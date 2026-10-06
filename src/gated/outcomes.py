import traceback

import numpy as np
import pandas as pd

from src import causal as C
from src import models, survival
from src.causal.sensitivity import benchmark_variables
from src.gated import protocol, runlog
from src.gated.design import backend_label, eligible_frames, prognostic_label, pseudo
from src.gated.protocol import (
    B_YEARS_X1,
    CATE_NOTE,
    CHEAP_LABEL,
    DAYS,
    E_NOTE,
    GATE_NAMES,
    GATES,
    HORIZON_YEARS,
    K_FOLDS,
    MIN_CONTROLS,
    N_REAL,
    SEED,
    SENS_LANDMARKS,
    SENSITIVITY,
    SUBGROUPS,
)
from src.gated.runlog import StopRun, audit, check_budget, gate, gate_status, log_fits, say
from src.rules import evaluate_rule


def stage_d(ctx, res, s, horizons, cache_dir, validate):
    """Unblind the trial outcomes; G9-G12 with the estimates, sensitivity rows and CATE.

    Only an error in the primary scope ends the run. Reference, descriptive, exploratory and report-only results (R in
    hybrid mode, the Hajek comparator, E, NC, X1-X4, CATE, G12) that raise are recorded as failed and the run goes on.
    """
    _unblind(ctx, s)
    hzs = _horizon_adequacy(ctx, res, s, horizons)
    s.update(F_t=C.features_trial(s["T"]), folds=C.make_folds(s["fine"], K_FOLDS, SEED))
    sens_frames = _sensitivity_frames(s) if s["r_scope"] else {}
    rows, sens = [], []
    for i, hz in enumerate(hzs):
        _estimate_horizon(ctx, res, s, hz, 0.4 + 0.4 * i / len(hzs), sens_frames, cache_dir, rows, sens)
    res["estimates"], res["sensitivity"] = rows, sens
    _g10(ctx, res, hzs, rows)
    _g11_fixed_sequence(ctx, res, s, hzs, rows, sens)
    if s["r_scope"]:
        res["cate"] = _cate(ctx, s, hzs)
    _g12(ctx, res, s, hzs[0], validate)


def _estimate_horizon(ctx, res, s, hz, frac, sens_frames, cache_dir, rows, sens):
    """Outcome models, estimate rows and sensitivity rows at one horizon, then the G9 IPCW report."""
    _fit_outcome_models(ctx, s, hz, frac, cache_dir)
    ipcw = {}
    if s["r_scope"]:
        ref = _scope_r_estimates(ctx, res, s, hz, rows)
        r_sens, ipcw["R"] = _scope_r_sensitivity(ctx, s, hz, sens_frames, ref)
        sens.extend(r_sens)
    if s["hybrid"]:
        _scope_h_estimates(ctx, s, hz, rows)
        sens.extend(_negative_controls(ctx, s, hz))
        e_est = _scope_e_estimates(ctx, s, hz, rows)
        h_sens, ipcw["H"] = _scope_h_sensitivity(s, hz)
        sens.extend(h_sens)
        sens.extend(_exploratory_sensitivity(ctx, s, hz, e_est))
    scope = _primary_scope(ctx)
    _g9_ipcw(ctx, hz, scope, ipcw[scope])


def _unblind(ctx, s):
    """Log the unblinding of the trial outcomes; from here on design actions are refused."""
    audit(ctx, "unblind", what="trial time/cens")
    ctx["unblinded"] = True
    say(ctx, "== UNBLINDED trial outcomes ==", 0.4)
    outcomes = s["fr"]["outcomes"]["trial"]
    s.update(trial_outcomes=outcomes, time=outcomes["time"].to_numpy(float), event=outcomes["cens"].to_numpy(float))
    s["fine"], s["coarse"] = survival.strata_labels("A", s["A"], s["post"])
    if s["hybrid"]:
        hist = s["fr"]["outcomes"]["hist"]
        s.update(time_B=hist["time"].to_numpy(float), event_B=hist["cens"].to_numpy(float))


def _horizon_adequacy(ctx, res, s, horizons):
    """Pseudo-observations, censoring fits and G9 status per horizon (one dict each); stops if the primary fails."""
    scope = _primary_scope(ctx)
    hzs, tables = [], []
    for h in horizons:
        hz = _pseudo_observations(s, h)
        hz["cens"] = _censoring_fits(s, hz)
        hz["g9"] = _g9_scopes(ctx, s, hz, tables)
        if h == horizons[0]:
            if hz["g9"][scope] == "FAIL":
                raise StopRun("G9", "the primary horizon fails horizon adequacy (no horizon switching)")
            hz["role"] = "primary"
        else:  # a failing secondary horizon becomes supplementary
            hz["role"] = (
                "secondary" if h == HORIZON_YEARS["secondary"] and hz["g9"][scope] != "FAIL" else "supplementary"
            )
        hzs.append(hz)
    res["horizon"] = pd.concat(tables, ignore_index=True)
    return hzs


def _pseudo_observations(s, h):
    """Pseudo-observations at horizon h per scope; kept and hidden controls are never pooled."""
    days = h * DAYS
    hz = {"h": h, "days": days}
    hz["theta"], hz["used"] = pseudo(s["time"], s["event"], days, s["fine"], s["coarse"])
    if s["hybrid"]:
        k = s["kept"]
        hz["theta_kept"], hz["used_kept"] = pseudo(s["time"][k], s["event"][k], days, s["fine"][k], s["coarse"][k])
    return hz


def _censoring_fits(s, hz):
    """Censoring model G(t) per source x arm cell (reverse Cox, or KM when there are too few censoring events)."""
    t, ev, T, days = s["time"], s["event"], s["T"], hz["days"]
    min_events = GATES["G9_cox_min_events_per_coef"]
    cells = {"A|1": s["trt"], "A|0": s["ctrl"]} if s["r_scope"] else {"A|1": s["trt"]}
    if s["hybrid"]:
        cells["kept"] = s["kept"]
    fits = {
        k: survival.censoring_G(t[i], ev[i], C.censoring_covariates(T.iloc[i], "A"), days, min_events)
        for k, i in cells.items()
    }
    if s["hybrid"]:
        fits["B|0"] = survival.censoring_G(
            s["time_B"], s["event_B"], C.censoring_covariates(s["B"], "B"), days, min_events
        )
    return fits


def _g9_scopes(ctx, s, hz, tables):
    """G9 at one horizon for scope R, scope H and the Rotterdam strata (report only); appends the check tables."""
    h, days, t, ev, trt, kept = hz["h"], hz["days"], s["time"], s["event"], s["trt"], s["kept"]
    lr_p = {k: v[1]["lr_p"] for k, v in hz["cens"].items()}
    status = {}
    if s["r_scope"]:
        hc = _horizon_check(t, ev, days, hz["used"])
        status["R"] = _g9(ctx, f"R {h:g}y", hc, _lr_note(lr_p, "A|1", "A|0"))
        tables.append(hc.assign(scope="R", horizon_years=h))
    if s["hybrid"]:
        hc = pd.concat(
            [
                _horizon_check(t[trt], ev[trt], days, hz["used"][trt]),
                _horizon_check(t[kept], ev[kept], days, hz["used_kept"]).assign(stratum=lambda x: "kept:" + x.stratum),
            ],
            ignore_index=True,
        )
        status["H"] = _g9(ctx, f"H {h:g}y", hc, _lr_note(lr_p, "A|1", "kept"))
        hc_B = _horizon_check(s["time_B"], s["event_B"], days, s["usedB"][h])
        _g9(ctx, f"B {h:g}y", hc_B, "Rotterdam strata, report only (m_B efficiency and E); " + _lr_note(lr_p, "B|0"))
        tables += [hc.assign(scope="H", horizon_years=h), hc_B.assign(scope="B", horizon_years=h)]
    return status


def _g9(ctx, scope, hc, note):
    """G9 from a ``survival.horizon_check`` table: worst stratum for G(H), at-risk count and KM identity."""
    hc = hc.reset_index(drop=True)
    ig, ir, ie = hc["G_H"].idxmin(), hc["at_risk_H"].idxmin(), hc["identity_err"].idxmax()
    g, n, err = float(hc.at[ig, "G_H"]), int(hc.at[ir, "at_risk_H"]), float(hc.at[ie, "identity_err"])
    checks = [
        (f"min_G_H[{hc.at[ig, 'stratum']}]", g, ">=", GATES["G9_G_min"], g >= GATES["G9_G_min"]),
        (f"min_at_risk_H[{hc.at[ir, 'stratum']}]", n, ">=", GATES["G9_min_at_risk"], n >= GATES["G9_min_at_risk"]),
        (
            f"max_identity_err[{hc.at[ie, 'stratum']}]",
            err,
            "<",
            GATES["G9_identity_tol"],
            err < GATES["G9_identity_tol"],
        ),
    ]
    return gate(ctx, "G9", GATE_NAMES["G9"], scope, gate_status(checks), checks, note=note)


def _lr_note(lr_p, *cells):
    """Reverse-Cox likelihood-ratio p-values of the censoring model per source x arm cell (G9 report)."""
    return "reverse-Cox LR p (report): " + "; ".join(
        f"{k} {format(lr_p[k], '.3g') if np.isfinite(lr_p[k]) else 'KM (too few censoring events)'}" for k in cells
    )


def _sensitivity_frames(s):
    """Eligible frames of S1 (landmark 0), S2 (landmark 182 d) and S3 (all ages)."""
    d, df_a, df_b = s["d"], s["df_a"], s["df_b"]
    return {
        "S1": eligible_frames(d, df_a, df_b, landmark=SENS_LANDMARKS["S1"]),
        "S2": eligible_frames(d, df_a, df_b, landmark=SENS_LANDMARKS["S2"]),
        "S3": eligible_frames(d, df_a, df_b, age_max=None),
    }


def _fit_outcome_models(ctx, s, hz, frac, cache_dir):
    """Cross-fitted T-learner mu (model backend) and its linear twin for the outcome-model diagnostics."""
    backend, h = ctx["backend"], hz["h"]
    arms = (0, 1) if s["r_scope"] else (1,)
    planned = K_FOLDS * len(arms) if backend.startswith("tabpfn") else 0
    check_budget(ctx, planned, f"outcome models {h:g} y")
    say(
        ctx,
        f"   fitting the outcome models (T-learner) at {h:g} y ({planned} TabPFN fits; {backend_label(backend)})",
        frac,
    )
    mu = C.crossfit_tlearner(
        s["F_t"],
        hz["theta"],
        s["A"],
        s["folds"],
        backend,
        SEED,
        C.TRIAL_CAT_IDX,
        arms=arms,
        purpose=f"mu_{h:g}y",
        cache_dir=cache_dir,
    )
    log_fits(ctx)
    mu_lin = (
        mu
        if backend == "logreg"
        else C.crossfit_tlearner(
            s["F_t"], hz["theta"], s["A"], s["folds"], "logreg", SEED, C.TRIAL_CAT_IDX, arms=arms, purpose=f"lin_{h:g}y"
        )
    )
    hz.update(mu=mu, mu_lin=mu_lin, preds=list(mu.values()))


def _scope_r_estimates(ctx, res, s, hz, rows):
    """Scope-R AIPW rows (primary in internal mode, demo reference in hybrid); returns the estimate per estimand."""
    hybrid, h, theta, mu = s["hybrid"], hz["h"], hz["theta"], hz["mu"]
    A, e = s["A"], s["e"]
    n, n1, n0 = len(A), len(s["trt"]), len(s["ctrl"])
    label = f"outcome: {backend_label(ctx['backend'])}; propensity: logistic (NOT TabPFN)"
    ref = {}
    for est in s["r_keep"]:
        es = _effect(ctx, hybrid, f"R {est} {h:g}y", lambda: _aipw(theta, A, e, mu, est))
        ov = C.overlap_summary(e, A, est, GATES["G4_ps_bounds"])
        n_target, n_comp, ess = {"ATT": (n1, n0, ov["ess_control"]), "ATC": (n0, n1, ov["ess_treated"])}.get(
            est, (n, n, min(ov["ess_treated"], ov["ess_control"]))
        )
        is_primary = est == s["r_primary"]
        base = "reference" if hybrid else ("primary" if is_primary else "secondary")
        note = "; ".join(
            x
            for x in (
                "demo reference: uses all trial controls (normally unknown)" if hybrid else "",
                "reference horizon fails G9" if hybrid and hz["g9"].get("R") == "FAIL" else "",
                "overlap population (estimand changed from ATT by action B4.1), not the treated"
                if is_primary and s["changed_from"]
                else "",
                _horizon_note(hz),
            )
            if x
        )
        _add(
            ctx,
            rows,
            est,
            "R",
            _role(base, hz["role"]),
            h,
            "AIPW, cross-fitted T-learner",
            label,
            es,
            n_target,
            n_comp,
            ess,
            s["changed_from"] if is_primary else None,
            note,
        )
        ref[est] = es.get("est")
    res["diagnostics"].setdefault("R", {})[f"{h:g}y"] = _effect(
        ctx, hybrid, f"R diagnostics {h:g}y", lambda: _outcome_diagnostics(theta, A, e, mu, hz["mu_lin"], s["folds"])
    )
    hz["preds"].append(e)
    return ref


def _scope_r_sensitivity(ctx, s, hz, sens_frames, ref):
    """S1-S6 rows of the main scope-R estimand (NOT TabPFN) and the (S4, S5, n floored) triple of the G9 report."""
    est = s["r_primary"] or "ATT"
    if est not in s["r_keep"]:
        return [], None
    h, days, ps_ext, T, t = hz["h"], hz["days"], s["ps_ext"], s["T"], s["time"]
    report_only, ref_est = s["hybrid"], ref.get(est)
    rows = []
    for sid in ("S1", "S2", "S3"):
        T_s, outcomes_s = sens_frames[sid]["trial_cov"], sens_frames[sid]["outcomes"]["trial"]
        es = _effect(
            ctx,
            report_only,
            f"{sid} R {h:g}y",
            lambda: _aipw_linear(T_s, _theta_frame(T_s, outcomes_s, days), est, ps_ext, f"sens_{sid}_{h:g}y"),
        )
        rows.append(_sens(sid, "R", h, est, es, ref_est, f"n = {len(T_s)}"))
    G = np.empty(len(s["A"]))
    G[s["trt"]], G[s["ctrl"]] = hz["cens"]["A|1"][0], hz["cens"]["A|0"][0]
    floor = GATES["G9_ipcw_G_floor"]
    n_floor = int(np.sum(G < floor))
    s4 = _effect(
        ctx,
        report_only,
        f"S4 R {h:g}y",
        lambda: _aipw_linear(T, survival.ipcw_outcome(t, days, G, floor), est, ps_ext, f"sens_S4_{h:g}y"),
    )
    s5 = _effect(ctx, report_only, f"S5 R {h:g}y", lambda: _aipw_linear(T, hz["theta"], est, ps_ext, f"sens_S5_{h:g}y"))
    s6 = _effect(
        ctx,
        report_only,
        f"S6 R {h:g}y",
        lambda: _aipw_linear(
            T, _theta_frame(T, s["trial_outcomes"], days, coarse_only=True), est, ps_ext, f"sens_S6_{h:g}y"
        ),
    )
    rows += [
        _sens("S4", "R", h, est, s4, ref_est, f"{n_floor} G values floored at {floor:g}"),
        _sens("S5", "R", h, est, s5, ref_est),
        _sens("S6", "R", h, est, s6, ref_est),
    ]
    return rows, (s4, s5, n_floor)


def _scope_h_estimates(ctx, s, hz, rows):
    """Scope H: robust hybrid ATT (primary) and ATC (secondary), and the Hajek entropy-balanced comparator."""
    h, theta_kept, mu = hz["h"], hz["theta_kept"], hz["mu"]
    T, Dt, trt, kept = s["T"], s["Dt"], s["trt"], s["kept"]
    choice, p = s["choice"][h], s["prog"][h]
    mB = p["m_T"] if choice == "model" else p["m_T_lin"] if choice == "linear" else None
    plab = prognostic_label(choice, ctx["backend"])
    Z = C.recalibration_design(T, mB)
    folds_C = C.make_folds(T["menostat"].to_numpy()[kept], K_FOLDS, SEED)
    theta_T, n1, n_kept, h_note = hz["theta"][trt], len(trt), len(kept), _horizon_note(hz)
    hy = C.robust_hybrid_att(theta_T, theta_kept, Dt[trt], Dt[kept], Z[trt], Z[kept], folds_C)
    _add(
        ctx,
        rows,
        "ATT",
        "H",
        _role("primary", hz["role"]),
        h,
        "robust hybrid (entropy balancing + recalibrated prognostic score; t interval)",
        f"outcome: OLS recalibration of the prognostic score (NOT TabPFN); prognostic score: {plab}; "
        f"weights: entropy balancing (NOT TabPFN)",
        hy,
        n1,
        n_kept,
        hy["ess"],
        None,
        h_note,
    )
    atc = C.robust_hybrid_atc(theta_T, theta_kept, Dt[trt], Dt[kept], mu[1][trt], mu[1][kept])
    _add(
        ctx,
        rows,
        "ATC",
        "H",
        _role("secondary", hz["role"]),
        h,
        "robust hybrid ATC (population = kept controls)",
        f"outcome: {backend_label(ctx['backend'])} (mu1 on the treated); weights: entropy balancing (NOT TabPFN)",
        atc,
        n_kept,
        n1,
        atc["ess"],
        None,
        h_note,
    )
    hajek = _effect(
        ctx, True, f"H Hajek {h:g}y", lambda: C.hajek_eb_att(theta_T, theta_kept, hy["w"], df=hy["ess"] - 1)
    )
    _add(
        ctx,
        rows,
        "ATT",
        "H",
        "descriptive",
        h,
        "Hajek entropy-balanced small arm (no outcome model)",
        "weights: entropy balancing (NOT TabPFN)",
        hajek,
        n1,
        n_kept,
        hy["ess"],
        None,
        "descriptive comparator",
    )
    hz.update(
        hy=hy,
        mB=mB,
        plab=plab,
        folds_C=folds_C,
        preds=hz["preds"] + [hy["mu0_T"], hy["mu0_C"]] + ([mB] if mB is not None else []),
    )


def _negative_controls(ctx, s, hz):
    """Negative-control beta of the trial controls on the Rotterdam prognostic score (report only, 90% CI)."""
    mB = hz["mB"]
    if mB is None:
        return []
    h, kept, ctrl = hz["h"], s["kept"], s["ctrl"]
    label = f"prognostic score: {hz['plab']}; weights: entropy balancing / ATT odds (NOT TabPFN)"
    change = "negative-control beta (trial vs external controls)"
    beta = _effect(ctx, True, f"NC H {h:g}y", lambda: _nc_beta(hz["theta_kept"], mB[kept], hz["hy"]["w"]))
    rows = [
        _sens(
            "NC",
            "H",
            h,
            "beta",
            beta,
            None,
            "90% CI; kept controls (entropy balancing) vs the prognostic score; TOST only if G8 is feasible",
            label,
            change,
        )
    ]
    if s["r_scope"]:
        odds = (s["e"] / (1 - s["e"]))[ctrl]
        beta = _effect(ctx, True, f"NC R {h:g}y", lambda: _nc_beta(hz["theta"][ctrl], mB[ctrl], odds))
        rows.append(
            _sens(
                "NC",
                "R",
                h,
                "beta",
                beta,
                None,
                "90% CI; all trial controls with ATT odds weights (demo only)",
                label,
                change,
            )
        )
    return rows


def _scope_e_estimates(ctx, s, hz, rows):
    """Scope-E rows (exploratory transport, E-pre); returns the main E estimate, the sign reference of X1-X4."""
    mB = hz["mB"]
    if mB is None:
        return None
    h, theta, trt, B = hz["h"], hz["theta"], s["trt"], s["B"]
    p, theta_B, Ds_T, Ds_B = s["prog"][h], s["thetaB"][h], s["Ds_T"], s["Ds_B"]
    oof = p["oof"] if s["choice"][h] == "model" else p["oof_lin"]
    pre_T, pre_B = _premenopausal(s)
    label = f"prognostic score: {hz['plab']}; weights: logistic source odds (NOT TabPFN)"
    e_note = E_NOTE + ("; postmenopausal support inadequate (G6)" if s["g6_post_fail"] else "")
    tr = _effect(ctx, True, f"E {h:g}y", lambda: C.transport_att(theta[trt], mB[trt], theta_B, oof, Ds_T[trt], Ds_B))
    failed = "failed" in tr
    tip = (
        ""
        if failed
        else (
            f"; tipping point of the CI limit {tr['tipping'][1] * 100:+.1f} points vs delta_max "
            f"{GATES['E_delta_max'] * 100:g}"
        )
    )
    _add(
        ctx,
        rows,
        "ATT",
        "E",
        "exploratory",
        h,
        "DR transport (logistic source odds)",
        label,
        tr,
        len(trt),
        len(B),
        tr.get("ess_B", np.nan),
        None,
        e_note + tip,
    )
    _add(
        ctx,
        rows,
        "ATT",
        "E",
        "exploratory",
        h,
        "g-formula: mean_T(theta - m_B) (report)",
        f"prognostic score: {hz['plab']}",
        tr if failed else {"est": tr["gformula"]},
        len(trt),
        len(B),
        np.nan,
        None,
        e_note,
    )
    pre = _effect(
        ctx,
        True,
        f"E-pre {h:g}y",
        lambda: C.transport_att(theta[pre_T], mB[pre_T], theta_B[pre_B], oof[pre_B], Ds_T[pre_T], Ds_B[pre_B]),
    )
    _add(
        ctx,
        rows,
        "ATT",
        "E",
        "exploratory",
        h,
        "DR transport, premenopausal only (E-pre)",
        label,
        pre,
        len(pre_T),
        len(pre_B),
        pre.get("ess_B", np.nan),
        None,
        E_NOTE,
    )
    return tr.get("est")


def _scope_h_sensitivity(s, hz):
    """S4-S6 rows of scope H with the linear score (NOT TabPFN) and the (S4, S5, n floored) triple of the G9 report."""
    h, days, t, ev, trt, kept = hz["h"], hz["days"], s["time"], s["event"], s["trt"], s["kept"]
    Z, folds_C, ref = C.recalibration_design(s["T"], s["prog"][h]["m_T_lin"]), hz["folds_C"], hz["hy"]["est"]
    G_T, G_K = hz["cens"]["A|1"][0], hz["cens"]["kept"][0]
    floor = GATES["G9_ipcw_G_floor"]
    n_floor = int(np.sum(G_T < floor) + np.sum(G_K < floor))
    s4 = _hybrid(
        s, survival.ipcw_outcome(t[trt], days, G_T, floor), survival.ipcw_outcome(t[kept], days, G_K, floor), Z, folds_C
    )
    s5 = _hybrid(s, hz["theta"][trt], hz["theta_kept"], Z, folds_C)
    s6 = _hybrid(s, survival.pseudo_km(t[trt], ev[trt], days), survival.pseudo_km(t[kept], ev[kept], days), Z, folds_C)
    rows = [
        _sens("S4", "H", h, "ATT", s4, ref, f"{n_floor} G values floored at {floor:g}"),
        _sens("S5", "H", h, "ATT", s5, ref),
        _sens("S6", "H", h, "ATT", s6, ref),
    ]
    return rows, (s4, s5, n_floor)


def _exploratory_sensitivity(ctx, s, hz, e_est):
    """X1-X4 with the linear prognostic score (exploratory, NOT TabPFN)."""
    h = hz["h"]
    x1 = _effect(ctx, True, f"X1 {h:g}y", lambda: _x1_effects(s, hz))
    x1_h, x1_e = (x1, x1) if "failed" in x1 else (x1["H"], x1["E"])
    rows = [
        _sens("X1", "H", h, "ATT", x1_h, hz["hy"]["est"], f"linear m_B trained on {x1.get('n_B')} Rotterdam patients"),
        _sens("X1", "E", h, "ATT", x1_e, e_est, E_NOTE),
    ]
    for sid, effect, note in (
        ("X2", _x2_effect, E_NOTE),
        ("X3", _x3_effect, E_NOTE + "; collider-biased: descriptive only"),
        ("X4", _x4_effect, E_NOTE),
    ):
        es = _effect(ctx, True, f"{sid} {h:g}y", lambda: effect(s, hz))
        rows.append(_sens(sid, "E", h, "ATT", es, e_est, note))
    return rows


def _x1_effects(s, hz):
    """X1: linear prognostic score refitted on Rotterdam patients operated in B_YEARS_X1, used in H and in E."""
    h, T, B, trt = hz["h"], s["T"], s["B"], s["trt"]
    years = B["year"].between(*B_YEARS_X1).to_numpy()
    B1 = B[years].reset_index(drop=True)
    fine, coarse = survival.strata_labels("B", np.zeros(len(B1), int), B1["post"].to_numpy(int))
    theta_1 = pseudo(s["time_B"][years], s["event_B"][years], hz["days"], fine, coarse)[0]
    p1 = C.prognostic_score(
        C.features_source(B1),
        s["Ds_B"][years],
        theta_1,
        C.features_source(T),
        s["Ds_T"],
        C.make_folds(B1["post"].to_numpy(), K_FOLDS, SEED),
        "logreg",
        SEED,
        C.SOURCE_CAT_IDX,
        purpose=f"sens_X1_{h:g}y",
    )
    theta_T = hz["theta"][trt]
    return {
        "H": _hybrid(s, theta_T, hz["theta_kept"], C.recalibration_design(T, p1["m_T_lin"]), hz["folds_C"]),
        "E": C.transport_att(theta_T, p1["m_T_lin"][trt], theta_1, p1["oof_lin"], s["Ds_T"][trt], s["Ds_B"][years]),
        "n_B": len(B1),
    }


def _x2_effect(s, hz):
    """X2: E with continuous log receptors in the source model."""
    h, trt = hz["h"], s["trt"]
    p = s["prog"][h]
    D_T, D_B = C.design_source_logrec(s["T"]), C.design_source_logrec(s["B"])
    return C.transport_att(hz["theta"][trt], p["m_T_lin"][trt], s["thetaB"][h], p["oof_lin"], D_T[trt], D_B)


def _x3_effect(s, hz):
    """X3: E with grade III added to the source model (collider-biased, descriptive only)."""
    h, trt = hz["h"], s["trt"]
    p = s["prog"][h]
    grade3_T = (s["T"]["tgrade"] == "III").to_numpy(float)[:, None]
    grade3_B = (s["B"]["tgrade"] == "III").to_numpy(float)[:, None]
    return C.transport_att(
        hz["theta"][trt],
        p["m_T_lin"][trt],
        s["thetaB"][h],
        p["oof_lin"],
        np.hstack([s["Ds_T"][trt], grade3_T[trt]]),
        np.hstack([s["Ds_B"], grade3_B]),
    )


def _x4_effect(s, hz):
    """X4: E restricted to the premenopausal treated and Rotterdam patients."""
    h = hz["h"]
    p = s["prog"][h]
    pre_T, pre_B = _premenopausal(s)
    return C.transport_att(
        hz["theta"][pre_T],
        p["m_T_lin"][pre_T],
        s["thetaB"][h][pre_B],
        p["oof_lin"][pre_B],
        s["Ds_T"][pre_T],
        s["Ds_B"][pre_B],
    )


def _g9_ipcw(ctx, hz, scope, ipcw):
    """G9 report: shift of the primary-scope estimate under IPCW (S4 vs S5, both linear), in standard errors."""
    s4, s5, n_floor = ipcw
    ratio = abs(s4["est"] - s5["est"]) / s5["se"]
    warn = GATES["G9_ipcw_warn_se"]
    gate(
        ctx,
        "G9",
        GATE_NAMES["G9"],
        f"{scope} {hz['h']:g}y IPCW",
        "WARN" if ratio > warn else "PASS",
        [("abs_delta_psi_over_se", ratio, "<=", warn, ratio <= warn)],
        note=f"S4 (IPCW, {n_floor} floored) vs S5 (pseudo-observations), both logistic/linear (NOT TabPFN); "
        f"report only",
    )


def _g10(ctx, res, hzs, rows):
    """G10 estimation integrity of the primary scope per horizon; the run stops if the primary horizon fails."""
    scope = _primary_scope(ctx)
    for hz in hzs:
        h = hz["h"]
        prow = [r for r in rows if r["scope"] == scope and r["horizon_years"] == h and r["role"] != "descriptive"]
        vals = np.array([[r["est"], r["se"], r["lo"], r["hi"]] for r in prow], float).reshape(-1, 4)
        finite = len(prow) > 0 and bool(np.isfinite(vals).all())
        se_pos = len(prow) > 0 and bool((vals[:, 1] > 0).all())
        in01 = all(bool(np.all((np.asarray(x) >= 0) & (np.asarray(x) <= 1))) for x in hz["preds"])
        checks = [
            ("estimates_finite", finite, "==", True, finite),
            ("se_positive", se_pos, "==", True, se_pos),
            ("predictions_in_0_1", in01, "==", True, in01),
        ]
        if ctx["mode"] == "hybrid":
            hy = hz["hy"]
            dev = abs(float(hy["w"].sum()) - 1)
            positive = bool((hy["w"] > 0).all())
            checks += [
                ("weights_sum_minus_1", dev, "<", 1e-9, dev < 1e-9),
                ("weights_positive", positive, "==", True, positive),
                (
                    "balance_imbalance",
                    hy["imbalance"],
                    "<",
                    GATES["G4H_imbalance_max"],
                    hy["imbalance"] < GATES["G4H_imbalance_max"],
                ),
            ]
        diag = f"{h:g}y" in res["diagnostics"].get("R" if scope == "R" else "prognostic", {})
        fits = runlog.tabpfn_fits(ctx)
        checks += [
            ("tabpfn_fits", fits, "<=", protocol.MAX_TABPFN_FITS, fits <= protocol.MAX_TABPFN_FITS),
            ("diagnostics_attached", diag, "==", True, diag),
        ]
        status = gate(ctx, "G10", GATE_NAMES["G10"], f"{scope} {h:g}y", gate_status(checks), checks)
        if hz is hzs[0] and status == "FAIL":
            raise StopRun(
                "G10",
                "NOT ACCEPTED (budget): the TabPFN fit budget was exceeded"
                if fits > protocol.MAX_TABPFN_FITS
                else "incomplete or failed computation",
            )


def _g11_fixed_sequence(ctx, res, s, hzs, rows, sens):
    """G11 at the primary horizon, then at the secondary horizon only if the primary passed."""
    ctx["frac"] = 0.85
    scope = _primary_scope(ctx)
    p_est = s["r_primary"] if scope == "R" else "ATT"
    prim = next(r for r in rows if r["scope"] == scope and r["estimand"] == p_est and r["role"] == "primary")
    res["primary"] = prim
    hz0 = hzs[0]
    res["g11"], bm, res["g11_detail"] = _g11(ctx, scope, prim, sens, _g11_sample(s, hz0, scope), hz0["h"])
    benchmarks = [bm]
    h3 = HORIZON_YEARS["secondary"]
    if res["g11"] != "PASS":
        for r in rows:
            if r["scope"] == scope and r["estimand"] == p_est and r["horizon_years"] == h3:
                r["note"] = "; ".join(
                    x
                    for x in (
                        r["note"],
                        ("fixed sequence not reached (2-y result not demonstrated): descriptive, not tested"),
                    )
                    if x
                )
    hz3 = next((hz for hz in hzs if hz["h"] == h3), None)
    if res["g11"] == "PASS" and hz3 is not None and hz3["role"] == "secondary":
        sec = next(
            r
            for r in rows
            if r["scope"] == scope and r["estimand"] == p_est and r["horizon_years"] == h3 and r["role"] == "secondary"
        )
        status, bm, _ = _g11(ctx, scope, sec, sens, _g11_sample(s, hz3, scope), h3)
        benchmarks.append(bm)
        res["secondary_claim"] = f"effect at {h3:g} years " + (
            "demonstrated under the stated assumptions"
            if status == "PASS"
            else "not demonstrated"
            if status == "NOT_APPLICABLE"
            else "estimated but not robust to unmeasured confounding"
        )
    res["benchmarks"] = pd.concat(benchmarks, ignore_index=True)


def _g11(ctx, scope, prim, sens, sample, h):
    """G11 at one horizon: not applicable if the RD CI includes 0, else robust vs benchmarks and signs."""
    label = f"{scope} {h:g}y"
    if prim["lo"] <= 0 <= prim["hi"]:
        gate(
            ctx,
            "G11",
            GATE_NAMES["G11"],
            label,
            "NOT_APPLICABLE",
            [("rd_ci_includes_0", True, "==", True, None)],
            note="the 95% CI includes 0: effect not demonstrated",
        )
        return "NOT_APPLICABLE", pd.DataFrame(), {}
    theta_s, A_s, D_s, T_s = sample
    bm = C.benchmark_bounding_factors(theta_s, A_s, D_s, benchmark_variables(T_s), GATES["G11_benchmark_min_cell"])
    named = C.bounding_factor(*GATES["G11_named_rr"])
    bf_max = float(np.nanmax(bm["bf"])) if bm["bf"].notna().any() else np.nan
    bf_ref = max(bf_max, named) if np.isfinite(bf_max) else named
    rule = [r for r in sens if r["scope"] == scope and r["horizon_years"] == h and r["in_sign_rule"]]
    same = sum(bool(np.sign(r["est"]) == np.sign(prim["est"])) for r in rule)
    sign_ok = bool(rule) and same == len(rule)
    rob = C.robustness(prim["rr"], prim["rr_lo"], prim["rr_hi"], bf_ref, sign_ok)
    rr_star = max(prim["rr"], 1 / prim["rr"])
    rr_au = bm.set_index("name")["rr_au"].get("menostat", np.nan)
    checks = [
        ("rr_star_lim", rob["rr_star_lim"], ">", bf_ref, rob["rr_star_lim"] > bf_ref),
        ("sign_stable_rows", same, "==", len(rule), sign_ok),
    ]
    note = (
        f"BF_ref = max(largest benchmark BF {bf_max:.3f}, named confounder {named:.2f}); outcome association "
        f"needed at the menostat RR_AU {rr_au:.2f}: {C.needed_rr_uy(rr_star, rr_au):.2f}"
    )
    status = gate(ctx, "G11", GATE_NAMES["G11"], label, "PASS" if rob["robust"] else "FAIL", checks, note=note)
    detail = {"rr_star_lim": rob["rr_star_lim"], "bf_ref": bf_ref, "bf_max": bf_max, "sign_ok": sign_ok}
    return status, bm.assign(horizon_years=h, bf_ref=bf_ref), detail


def _g11_sample(s, hz, scope):
    """Pseudo-observations, arm, design matrix and covariates of the primary scope's patients."""
    if scope == "R":
        return hz["theta"], s["A"], s["Dt"], s["T"]
    idx = np.r_[s["trt"], s["kept"]]
    return (
        np.r_[hz["theta"][s["trt"]], hz["theta_kept"]],
        s["A"][idx],
        s["Dt"][idx],
        s["T"].iloc[idx].reset_index(drop=True),
    )


def _cate(ctx, s, hzs):
    """CATE rows (scope R, exploratory) at the primary and secondary horizons; a failed horizon gives one row."""
    label = f"outcome: {backend_label(ctx['backend'])}; propensity: logistic (NOT TabPFN)"
    out = []
    for hz in hzs:
        if hz["h"] in (HORIZON_YEARS["primary"], HORIZON_YEARS["secondary"]):
            mu = hz["mu"]
            rows, failed = _report_only(
                ctx,
                f"CATE {hz['h']:g}y",
                lambda: _cate_horizon(s["T"], hz["theta"], s["A"], s["e"], mu[0], mu[1], hz["h"], label),
            )
            out += [_failed_cate_row(hz["h"], label, failed)] if failed else rows
    return out


def _cate_horizon(T, theta, A, e, mu0, mu1, h, label):
    """CATT_g and GATE_g per subgroup level, their contrasts, simultaneous bands and the BLP."""
    phi = C.ate_scores(theta, A, e, mu0, mu1)
    out = []
    for kind in ("CATT", "GATE"):
        ests, idx = [], []
        for name, rule_g, rule_not in SUBGROUPS:
            pair = []
            for rule in (rule_g, rule_not):
                m = evaluate_rule(rule, T).to_numpy(bool)
                r = C.subgroup_att(theta, A, e, mu0, m) if kind == "CATT" else C.subgroup_ate(phi, m)
                pair.append(r)
                ests.append(r)
                idx.append(len(out))
                out.append(
                    {
                        "horizon_years": h,
                        "estimand": kind,
                        "subgroup": name,
                        "level": rule,
                        "kind": "subgroup",
                        "est": r["est"],
                        "se": r["se"],
                        "lo": r["est"] - 1.96 * r["se"],
                        "hi": r["est"] + 1.96 * r["se"],
                        "n": int(m.sum()),
                        "n_treated": int((m & (A == 1)).sum()),
                    }
                )
            c = C.contrast(*pair)
            out.append(
                {
                    "horizon_years": h,
                    "estimand": kind,
                    "subgroup": name,
                    "level": f"({rule_g}) - ({rule_not})",
                    "kind": "contrast",
                    "est": c["est"],
                    "se": c["se"],
                    "lo": c["est"] - 1.96 * c["se"],
                    "hi": c["est"] + 1.96 * c["se"],
                    "p": c["p"],
                }
            )
        crit = C.simultaneous_band([r["if"] for r in ests], [r["se"] for r in ests], GATES["cate_band_draws"], SEED)
        for i in idx:
            out[i].update(
                band_c=crit, band_lo=out[i]["est"] - crit * out[i]["se"], band_hi=out[i]["est"] + crit * out[i]["se"]
            )
    b = C.blp(phi, mu1 - mu0)
    out.append(
        {
            "horizon_years": h,
            "estimand": "BLP",
            "subgroup": "all",
            "level": "beta2 (heterogeneity loading)",
            "kind": "BLP",
            "est": b["beta2"],
            "se": b["se2"],
            "lo": b["beta2"] - 1.96 * b["se2"],
            "hi": b["beta2"] + 1.96 * b["se2"],
            "p": b["p2"],
            "beta1": b["beta1"],
        }
    )
    return [{**r, "backend_label": label, "note": CATE_NOTE} for r in out]


def _failed_cate_row(h, label, failed):
    nan = np.nan
    return {
        "horizon_years": h,
        "estimand": "CATE",
        "subgroup": "all",
        "level": "",
        "kind": "failed",
        "est": nan,
        "se": nan,
        "lo": nan,
        "hi": nan,
        "n": nan,
        "n_treated": nan,
        "p": nan,
        "band_c": nan,
        "band_lo": nan,
        "band_hi": nan,
        "beta1": nan,
        "backend_label": label,
        "note": failed,
    }


def _g12(ctx, res, s, hz, validate):
    """G12 validation and operating characteristics at the primary horizon (report only, never gates)."""
    if not validate:
        gate(ctx, "G12", GATE_NAMES["G12"], "V", "SKIPPED", [], note="validate=False")
        return
    checks, failures = [], []
    if s["r_scope"] and len(s["ctrl"]) > GATES["placebo_n_treated"]:
        say(ctx, "   placebo test (logistic/linear, NOT TabPFN)", 0.9)
        out, failed = _report_only(ctx, "G12 placebo test", lambda: _placebo(s, hz))
        if failed:
            failures.append(f"placebo test {failed}")
        else:
            res["placebo"], placebo_checks = out
            checks += placebo_checks
    if s["demo"] and len(s["hidden"]) < MIN_CONTROLS:
        checks.append(("hidden_controls", len(s["hidden"]), ">=", MIN_CONTROLS, None))
    elif s["demo"]:
        say(ctx, "   hide-and-recover validation (logistic/linear, NOT TabPFN)", 0.94)
        out, failed = _report_only(ctx, "G12 hide-and-recover validation", lambda: _hide_and_recover(s, hz))
        if failed:
            failures.append(f"hide-and-recover validation {failed}")
        else:
            res["validation"], validation_checks = out
            checks += validation_checks
    note = f"report only, never gates; {models.CHEAP_LABEL}"
    if s["demo"]:
        note += f"; fixed prognostic-score predictions: {res['labels']['prognostic']}"
    note += "".join(f"; {f}" for f in failures)
    gate(ctx, "G12", GATE_NAMES["G12"], "V", "REPORT", checks, note=note)


def _placebo(s, hz):
    """Placebo test among the trial controls (true effect 0) and its G12 report checks."""
    placebo = C.placebo_test(
        s["T"], s["trial_outcomes"], hz["days"], GATES["placebo_reps"], GATES["placebo_n_treated"], SEED
    )
    summary = placebo["summary"].iloc[0]
    return placebo, [
        ("placebo_type_I_error", summary["type_I_error"], "report", None, None),
        ("placebo_bias_points", summary["bias"], "report", None, None),
    ]


def _hide_and_recover(s, hz):
    """Hide-and-recover validation with the hidden controls (fixed m_B predictions) and its G12 report checks."""
    choice, p = s["choice"][hz["h"]], s["prog"][hz["h"]]
    mB = p["m_T"] if choice == "model" else p["m_T_lin"] if choice == "linear" else None
    runs = C.hidden_control_validation(
        s["T"], s["trial_outcomes"], mB, hz["days"], N_REAL, GATES["validation_reps"], SEED
    )
    summary = C.summarize_validation_runs(runs)
    rh = summary[summary["method"] == "robust hybrid"].iloc[0]
    checks = [
        ("validation_bias_points", rh["bias"], "report", None, None),
        ("validation_coverage", rh["coverage"], "report", None, None),
    ]
    if s["stab"]:
        checks.append(("draw_stability_share_pass", s["stab"]["share_pass"], "report", None, None))
    return {"summary": summary, "runs": runs, "draw_stability": s["stab"]}, checks


def _effect(ctx, report_only, what, compute):
    """``compute()``; for a report-only row an exception gives ``{"failed": "failed: <error>"}`` instead."""
    if not report_only:
        return compute()
    out, failed = _report_only(ctx, what, compute)
    return {"failed": failed} if failed else out


def _report_only(ctx, what, compute):
    """``(compute(), "")``, or ``(None, "failed: <error>")`` after a ``row_failed`` audit line."""
    try:
        return compute(), ""
    except Exception as ex:  # noqa: BLE001 - a report-only result never ends the run
        audit(
            ctx,
            "row_failed",
            row=what,
            kind=type(ex).__name__,
            message=str(ex)[:2000],
            traceback=traceback.format_exc()[-4000:],
        )
        say(ctx, f"   {what}: failed ({type(ex).__name__}: {ex}); recorded, the run goes on")
        return None, f"failed: {type(ex).__name__}: {ex}"


def _add(ctx, rows, estimand, scope, role, h, method, label, es, n_target, n_comp, ess, changed_from=None, note=""):
    """Append one estimate row (a row of estimates.csv) and its ``estimate`` audit line; see :func:`_effect`."""
    row = {
        "estimand": estimand,
        "scope": scope,
        "role": role,
        "horizon_years": h,
        "method": method,
        "backend_label": label,
        **{
            k: es.get(k, np.nan) for k in ("est", "se", "lo", "hi", "df", "rr", "rr_lo", "rr_hi", "evalue", "evalue_ci")
        },
        "n_target": int(n_target),
        "n_comparison": int(n_comp),
        "ess": ess,
        "changed_from": changed_from,
        "note": es.get("failed", note),
    }
    rows.append(row)
    audit(
        ctx,
        "estimate",
        **{k: row[k] for k in ("estimand", "scope", "role", "horizon_years", "method", "est", "se", "lo", "hi")},
    )


def _sens(sid, scope, h, estimand, es, ref_est, note="", label=CHEAP_LABEL, change=None, in_rule=None):
    """One row of sensitivity.csv; ``same_sign`` compares it with the scope's main estimate."""
    spec = next((x for x in SENSITIVITY if x["id"] == sid), {"change": change, "in_sign_rule": False})
    rule = spec["in_sign_rule"] and scope in ("R", "H") if in_rule is None else in_rule
    failed = "failed" in es
    return {
        "id": sid,
        "scope": scope,
        "horizon_years": h,
        "change": change or spec["change"],
        "estimand": estimand,
        **{k: es.get(k, np.nan) for k in ("est", "se", "lo", "hi")},
        "in_sign_rule": bool(rule),
        "same_sign": None if ref_est is None or failed else bool(np.sign(es["est"]) == np.sign(ref_est)),
        "backend_label": label,
        "note": es["failed"] if failed else note,
    }


def _role(base, horizon_role):
    """Role of an estimate row: the estimand's role, demoted at the secondary and supplementary horizons."""
    if base == "primary":
        return horizon_role
    if base == "secondary" and horizon_role == "supplementary":
        return "supplementary"
    return base


def _horizon_note(hz):
    return "supplementary horizon: no claim" if hz["role"] == "supplementary" else ""


def _primary_scope(ctx):
    return "R" if ctx["mode"] == "internal" else "H"


def _horizon_check(t, ev, days, used):
    return survival.horizon_check(
        t, ev, days, used, GATES["G9_G_min"], GATES["G9_min_at_risk"], GATES["G9_identity_tol"]
    )


def _theta_frame(T, outcomes, days, coarse_only=False):
    """Pseudo-observations of one trial frame within arm x menostat strata, or within arm only (S6)."""
    fine, coarse = survival.strata_labels("A", T["A"].to_numpy(int), T["post"].to_numpy(int))
    return pseudo(
        outcomes["time"].to_numpy(float),
        outcomes["cens"].to_numpy(float),
        days,
        coarse if coarse_only else fine,
        coarse,
    )[0]


def _aipw(theta, A, e, mu, est):
    r = C.aipw(theta, A, e, mu[0], mu[1], est)
    return C.effect_summary(r["s1"], r["s0"], r["if_s1"], r["if_s0"])


def _aipw_linear(T, theta, est, ps_ext, purpose):
    """Scope-R AIPW with the logistic propensity score and a cross-fitted linear T-learner (NOT TabPFN)."""
    A, post = T["A"].to_numpy(int), T["post"].to_numpy(int)
    e = C.ps_logistic(C.design_trial_ext(T) if ps_ext else C.design_trial(T), A)
    folds = C.make_folds(survival.strata_labels("A", A, post)[0], K_FOLDS, SEED)
    mu = C.crossfit_tlearner(
        C.features_trial(T), theta, A, folds, "logreg", SEED, C.TRIAL_CAT_IDX, arms=(0, 1), purpose=purpose
    )
    return _aipw(theta, A, e, mu, est)


def _hybrid(s, theta_T, theta_K, Z, folds_C):
    """Robust-hybrid ATT of the treated vs the kept controls with recalibration covariates ``Z`` (all trial rows)."""
    trt, kept, Dt = s["trt"], s["kept"], s["Dt"]
    return C.robust_hybrid_att(theta_T, theta_K, Dt[trt], Dt[kept], Z[trt], Z[kept], folds_C)


def _nc_beta(theta, m, w):
    return dict(zip(("est", "se", "lo", "hi"), C.negative_control_beta(theta, m, w)))


def _premenopausal(s):
    """Positions of the premenopausal treated (trial) and of the premenopausal Rotterdam patients."""
    trt = s["trt"]
    return trt[s["post"][trt] == 0], np.flatnonzero(s["B"]["post"].to_numpy() == 0)


def _outcome_diagnostics(theta, A, e, mu, mu_lin, folds):
    return {
        "agreement": C.agreement(theta, A, e, mu[0]),
        "outcome_models": C.outcome_diagnostics(theta, A, folds, mu, mu_lin),
    }
