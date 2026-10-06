"""The gates, in order. Every gate returns a result dict; the first failing gate is handed to the repair loop.

Stage B (G1-G4) sees only covariates. Stage C (G5-G6) reads follow-up times (G5) and historical outcomes (G6).
Stage D reads the trial outcomes once, for the frozen design, and runs G7. No repair decision ever sees the
effect estimate: the repair loop gets gate diagnostics only.
"""

import operator

import numpy as np
import pandas as pd

from src import models, survival
from src.agent import frames
from src.causal.estimators import (
    effect_summary,
    hajek_att,
    mean_with_if,
    robust_hybrid_att,
)
from src.causal.nuisance import prognostic_score
from src.causal.sensitivity import evalue, evalue_ci, min_detectable_margin
from src.causal.weights import (
    balancing_weights,
    make_folds,
    overlap_summary,
    ps_logistic,
    smd_table,
    source_support,
)

LIMITS = {
    "min_treated": 30,
    "min_control": 30,
    "min_hist": 50,
    "min_events": 20,
    "min_events_hist": 15,
    "max_lost_complete_case": 0.10,
    "ps_bounds": (0.05, 0.95),
    "share_outside_max": 0.05,
    "ess_ratio_min": 0.5,
    "weight_share_max": 0.05,
    "smd_max": 0.10,
    "smd_ext_max": 0.25,
    "support_cut": 20.0,
    "unsupported_max": 0.10,
    "support_ess_min": 50.0,
    "support_weight_share_max": 0.10,
    "g_min": 0.70,
    "min_at_risk": 10,
    "min_stratum_n": 25,
    "prognostic_se_mult": 2.0,
    "eb_imbalance_max": 1e-6,
    "eb_ess_min_abs": 20.0,
    "eb_ess_min_frac": 1 / 3,
    "eb_weight_share_max": 0.15,
    "borrow_margin": 0.05,
    "folds": 5,
}
SEED = 0
GATE_NAMES = {
    "G1": "data fitness",
    "G2": "within-trial positivity",
    "G3": "covariate balance",
    "G4": "external support",
    "G5": "follow-up and horizon adequacy",
    "G6": "prognostic model credibility",
    "G7": "estimation integrity",
    "G8": "borrowing feasibility",
}
OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


def check(name, value, op, limit):
    """One numeric check; NaN fails."""
    value = float(value) if value is not None else float("nan")
    ok = None if op == "report" else bool(np.isfinite(value) and OPS[op](value, limit))
    return {"name": name, "value": value, "op": op, "limit": limit, "ok": ok}


def gate_result(gid, stage, checks, note="", diag=None, status=None):
    ok = all(c["ok"] for c in checks if c["ok"] is not None)
    return {
        "id": gid,
        "name": GATE_NAMES[gid],
        "stage": stage,
        "status": status or ("PASS" if ok else "FAIL"),
        "checks": checks,
        "note": note,
        "diag": diag or {},
    }


def run_gates(spec, trial, hist, state, ctx):
    """Run the gates in order on the current design state.

    Parameters
    ----------
    ctx : dict
        ``backend``, ``cache_dir``, ``say`` (callable(str)), ``audit`` (callable(event, **payload)).

    Returns
    -------
    dict
        ``gates``, ``failed`` (first failing gate or None), ``study``, ``state`` (copy, with an automatic
        prognostic choice applied), ``estimate`` (None if a gate failed before stage D).
    """
    state = {k: (list(v) if isinstance(v, list) else v) for k, v in state.items()}
    study = frames.build_study(spec, trial, hist, state)
    out = {"gates": [], "failed": None, "study": study, "state": state, "estimate": None}

    def add(g):
        out["gates"].append(g)
        ctx["say"](f"   {g['id']} {g['name']}: {g['status']}")
        ctx["audit"]("gate", **{k: g[k] for k in ("id", "status", "checks", "note")})
        if g["status"] == "FAIL":
            out["failed"] = g
        return g["status"] != "FAIL"

    steps = [lambda: g1_data(study, spec, state), lambda: g2_positivity(study), lambda: g3_balance(study)]
    if study["hybrid"]:
        steps.append(lambda: g4_support(study))
    steps.append(lambda: g5_horizon(study, spec, state))
    for step in steps:
        if not add(step()):
            return out
    prog = None
    if study["hybrid"]:
        g6, prog = g6_prognostic(study, spec, state, ctx)
        if g6["status"] == "PASS_AFTER_ACTION":
            state["prognostic"] = g6["diag"]["choice"]
        add(g6)
    ctx["audit"]("unblind_trial_outcomes", horizon_years=state["horizon_years"])
    ctx["say"]("   reading the trial outcomes (design is frozen for this attempt)")
    est = estimate(study, spec, state, prog)
    out["estimate"] = est
    if not add(g7_integrity(est, study)):
        return out
    if prog is not None:
        out["gates"].append(g8_borrowing(est, prog, state))
    return out


# ----------------------------------------------------------------------------
# Stage B: covariates only
# ----------------------------------------------------------------------------
def g1_data(study, spec, state):
    T, B, lim = study["T"], study["B"], LIMITS
    n_after_landmark = next(n for step, n in study["funnel"] if step.startswith("trial: event-free"))
    lost = 1 - len(T) / max(n_after_landmark, 1)
    a = T["__A"].to_numpy()
    checks = [
        check("n_treated", a.sum(), ">=", lim["min_treated"]),
        check("n_control", (1 - a).sum(), ">=", lim["min_control"]),
        check("events_trial (pooled)", T["__event"].sum(), ">=", lim["min_events"]),
        check("lost_to_missing_covariates", lost, "<=", lim["max_lost_complete_case"]),
    ]
    if study["hybrid"]:
        checks += [
            check("n_historical", len(B), ">=", lim["min_hist"]),
            check("events_historical", B["__event"].sum(), ">=", lim["min_events_hist"]),
        ]
    return gate_result(
        "G1",
        "B",
        checks,
        diag={
            "funnel": study["funnel"],
            "missing_share_trial": study["missing_trial"],
            "missing_share_historical": study["missing_hist"],
        },
    )


def ranges(study):
    """Per-arm 2.5% and 97.5% quantiles of every continuous covariate (what a range restriction can use)."""
    T, out = study["T"], {}
    for e in study["encs"]:
        if e["kind"] not in ("numeric", "ordinal") or e["levels"] is not None:
            continue
        v = pd.to_numeric(T[e["name"]])
        out[e["name"]] = {
            "treated": [float(v[T["__A"] == 1].quantile(0.025)), float(v[T["__A"] == 1].quantile(0.975))],
            "control": [float(v[T["__A"] == 0].quantile(0.025)), float(v[T["__A"] == 0].quantile(0.975))],
            "all": [float(v.min()), float(v.max())],
        }
    return out


def _propensity(study):
    T = study["T"]
    D = frames.ps_design(study, T)
    a = T["__A"].to_numpy()
    return D, a, ps_logistic(D, a)


def g2_positivity(study):
    lim = LIMITS
    try:
        _, a, e = _propensity(study)
    except Exception as ex:  # noqa: BLE001 - separation or a singular model is a positivity failure
        return gate_result(
            "G2",
            "B",
            [check("propensity_model_fits", 0, ">=", 1)],
            note=f"{type(ex).__name__}: {ex}",
            diag={"ranges": ranges(study), "smd_before": _smd_before(study)},
        )
    ov = overlap_summary(e, a, "ATT", lim["ps_bounds"])
    checks = [
        check("share_treated_outside_bounds", ov["share_outside"], "<=", lim["share_outside_max"]),
        check("ess_control / n_control", ov["ess_control"] / ov["n_control"], ">=", lim["ess_ratio_min"]),
        check("max_weight_share_control", ov["max_share_control"], "<=", lim["weight_share_max"]),
    ]
    return gate_result("G2", "B", checks, diag={"ranges": ranges(study), "smd_before": _smd_before(study)})


def _smd_before(study):
    T = study["T"]
    D, names, _ = frames.design(study["encs"], T)
    tab = smd_table(D, T["__A"].to_numpy(), np.ones(len(T)), names)
    return {r["name"]: round(float(r["smd_before"]), 3) for r in tab.to_dict("records")}


def g3_balance(study):
    lim = LIMITS
    T = study["T"]
    try:
        _, a, e = _propensity(study)
    except Exception as ex:  # noqa: BLE001
        return gate_result("G3", "B", [check("propensity_model_fits", 0, ">=", 1)], note=str(ex))
    w = balancing_weights(e, a, "ATT")
    D_ext, names, n_main = frames.design(study["encs"], T, squares=True, products=True)
    tab = smd_table(D_ext, a, w, names)
    main, ext = tab["smd_after"].abs().to_numpy()[:n_main], tab["smd_after"].abs().to_numpy()[n_main:]
    checks = [check("max_smd_main_effects", np.max(main) if len(main) else 0.0, "<=", lim["smd_max"])]
    if len(ext):
        checks.append(check("max_smd_squares_products", np.max(ext), "<=", lim["smd_ext_max"]))
    worst = tab.assign(a=tab["smd_after"].abs()).sort_values("a", ascending=False).head(5)
    return gate_result(
        "G3",
        "B",
        checks,
        diag={"worst_terms": {r["name"]: round(float(r["smd_after"]), 3) for r in worst.to_dict("records")}},
    )


def g4_support(study):
    lim = LIMITS
    T, B = study["T"], study["B"]
    treated = T[T["__A"] == 1]
    D_T, names, _ = frames.design(study["enc_s"], treated)
    D_B = frames.design(study["enc_s"], B)[0]
    src = source_support(
        D_T,
        D_B,
        np.array(["all"] * len(D_T)),
        np.array(["all"] * len(D_B)),
        seed=SEED,
        k=lim["folds"],
        cut=lim["support_cut"],
    )
    row = src["table"][src["table"]["stratum"] == "all"].iloc[0]
    X = np.vstack([D_T, D_B])
    arm = np.r_[np.ones(len(D_T)), np.zeros(len(D_B))]
    smd = smd_table(X, arm, np.ones(len(X)), names)
    checks = [
        check("share_treated_unsupported", row["share_unsupported"], "<=", lim["unsupported_max"]),
        check("ess_historical", row["ess_B"], ">=", min(lim["support_ess_min"], 0.5 * len(D_B))),
        check("max_weight_share_historical", row["max_share"], "<=", lim["support_weight_share_max"]),
        check("auc_trial_vs_historical", src["auc"], "report", None),
    ]
    return gate_result(
        "G4",
        "B",
        checks,
        diag={"smd_trial_vs_hist": {r["name"]: round(float(r["smd_before"]), 3) for r in smd.to_dict("records")}},
    )


# ----------------------------------------------------------------------------
# Stage C: follow-up and historical outcomes
# ----------------------------------------------------------------------------
def _arm_labels(study):
    T, B = study["T"], study["B"]
    time = T["__time"].to_numpy(float)
    event = T["__event"].to_numpy(float)
    labels = np.array([f"trial|{int(a)}" for a in T["__A"]], dtype=object)
    if B is not None:
        time = np.r_[time, B["__time"].to_numpy(float)]
        event = np.r_[event, B["__event"].to_numpy(float)]
        labels = np.r_[labels, np.array(["historical|0"] * len(B), dtype=object)]
    return time, event, labels


def _pseudo(time, event, h_days, labels):
    return survival.pseudo_by_strata(
        time, event, h_days, labels, labels, LIMITS["min_stratum_n"], LIMITS["min_at_risk"]
    )


def _horizon_table(study, h_years):
    time, event, labels = _arm_labels(study)
    _, used = _pseudo(time, event, h_years * survival.DAYS_PER_YEAR, labels)
    return survival.horizon_check(
        time, event, h_years * survival.DAYS_PER_YEAR, used, LIMITS["g_min"], LIMITS["min_at_risk"]
    )


def max_supported_horizon(study, h_years):
    """Largest horizon (in steps of 10% of the planned one, down to 30%) at which every arm passes."""
    for k in range(0, 8):
        h = h_years * (1 - 0.1 * k)
        if _horizon_table(study, h)["ok"].all():
            return float(h)
    return None


def g5_horizon(study, spec, state):
    h = state["horizon_years"]
    tab = _horizon_table(study, h)
    checks = []
    for r in tab.itertuples():
        c = check(f"G(H) in {r.stratum} (>= {LIMITS['g_min']:.2f}, {LIMITS['min_at_risk']}+ at risk)", r.G_H, ">=", 0)
        c["limit"], c["ok"] = LIMITS["g_min"], bool(r.ok)
        checks.append(c)
    diag = {"table": tab.to_dict("records")}
    if not tab["ok"].all():
        diag["max_supported_years"] = max_supported_horizon(study, h)
    return gate_result("G5", "C", checks, diag=diag, note=f"horizon {h:g} y")


def g6_prognostic(study, spec, state, ctx):
    """Prognostic score fitted on the historical data only; falls back to linear, then to none."""
    T, B, h = study["T"], study["B"], state["horizon_years"]
    theta_B, _ = _pseudo(
        B["__time"].to_numpy(float),
        B["__event"].to_numpy(float),
        h * survival.DAYS_PER_YEAR,
        np.array(["historical|0"] * len(B), dtype=object),
    )
    F_B, cat_idx = frames.features(study["enc_s"], B)
    F_T, _ = frames.features(study["enc_s"], T)
    D_B = frames.design(study["enc_s"], B)[0]
    D_T = frames.design(study["enc_s"], T)[0]
    folds_B = make_folds(frames.strata_labels(B, study["enc_s"]), LIMITS["folds"], SEED)
    ctx["say"](f"   fitting the prognostic score on {len(B)} historical patients ({ctx['backend']})")
    p = prognostic_score(
        F_B,
        D_B,
        theta_B,
        F_T,
        D_T,
        folds_B,
        ctx["backend"],
        SEED,
        cat_idx,
        purpose=f"mB_{h:g}y",
        cache_dir=ctx.get("cache_dir"),
    )
    limit = p["mse_lin"] + LIMITS["prognostic_se_mult"] * p["se_diff"]
    ok_rel, ok_null = p["mse"] <= limit, p["mse"] < p["mse_null"]
    checks = [
        check("cv_mse_model <= cv_mse_linear + 2 se", p["mse"], "<=", limit),
        check("cv_mse_model < cv_mse_null", p["mse"], "<", p["mse_null"]),
        check("cv_mse_linear", p["mse_lin"], "report", None),
        check("cal_slope", p["cal_slope"], "report", None),
    ]
    choice = state["prognostic"]
    status, note = "PASS", f"prognostic score: {models.BACKEND_LABELS.get(ctx['backend'], ctx['backend'])}"
    if choice == "model" and not (ok_rel and ok_null):
        choice = "linear" if p["mse_lin"] < p["mse_null"] else "none"
        status, note = "PASS_AFTER_ACTION", f"model not credible: switched to the {choice} prognostic score"
    elif choice != "model":
        note = f"prognostic score: {choice} (chosen earlier)"
    for c in checks[:2]:
        c["ok"] = c["ok"] if choice == "model" else None  # the failed model check is not a gate failure once handled
    return gate_result("G6", "C", checks, note=note, status=status, diag={"choice": choice}), p


# ----------------------------------------------------------------------------
# Stage D: estimation
# ----------------------------------------------------------------------------
def kept_controls(study, n_real):
    """Frozen sorted draw of the controls kept as 'real' (all of them unless ``n_real`` is set)."""
    ctrl = np.flatnonzero(study["T"]["__A"].to_numpy() == 0)
    if not n_real or n_real >= len(ctrl):
        return ctrl
    return np.sort(np.random.default_rng(SEED).choice(ctrl, size=n_real, replace=False))


def estimate(study, spec, state, prog):
    """Primary estimate: entropy-balanced controls plus a recalibrated outcome model (robust hybrid), ATT.

    Reference rows: unadjusted difference and a logistic-propensity IPW point estimate.
    """
    T, h = study["T"], state["horizon_years"]
    time, event, labels = _arm_labels(study)
    theta_all, used = _pseudo(time, event, h * survival.DAYS_PER_YEAR, labels)
    nT = len(T)
    theta = theta_all[:nT]
    a = T["__A"].to_numpy()
    kept = kept_controls(study, state["n_real"])
    treated = np.flatnonzero(a == 1)
    D = frames.design(study["encs"], T)[0]
    choice = state["prognostic"] if prog is not None else "none"
    if choice == "none":
        Z = D
    else:
        m = prog["m_T"] if choice == "model" else prog["m_T_lin"]
        t_only = frames.design(study["enc_t_only"], T)[0] if study["enc_t_only"] else np.zeros((nT, 0))
        Z = np.column_stack([m, t_only])
    folds_C = make_folds(frames.strata_labels(T.iloc[kept], study["encs"]), LIMITS["folds"], SEED)
    res = robust_hybrid_att(theta[treated], theta[kept], D[treated], D[kept], Z[treated], Z[kept], folds_C)
    s1, i1 = mean_with_if(theta[treated])
    s0, i0 = mean_with_if(theta[kept])
    unadj = effect_summary(s1, s0, np.r_[i1, np.zeros(len(kept))], np.r_[np.zeros(len(treated)), i0])
    rows = np.r_[treated, kept]
    e = ps_logistic(D[rows], a[rows])
    ipw = hajek_att(theta[rows], a[rows], e)
    return {
        "primary": res,
        "unadjusted": unadj,
        "ipw_point": ipw,
        "horizon_years": h,
        "n_treated": int(len(treated)),
        "n_controls_kept": int(len(kept)),
        "prognostic": choice,
        "pseudo_strata": sorted(set(used)),
        "sensitivity": sensitivity(res),
    }


def sensitivity(res):
    """E-values for the primary risk ratio (VanderWeele & Ding 2017) and the additive tipping point."""
    rr, lo, hi = res["rr"], res["rr_lo"], res["rr_hi"]
    return {
        "rr": rr,
        "evalue": evalue(rr),
        "evalue_ci": evalue_ci(lo, hi),
        "tipping_bias": float(res["est"]),
        "tipping_bias_ci": float(res["est"] - 1.96 * res["se"])
        if res["est"] > 0
        else float(res["est"] + 1.96 * res["se"]),
    }


def g7_integrity(est, study):
    lim, r = LIMITS, est["primary"]
    n_c = est["n_controls_kept"]
    checks = [
        check("entropy_balance_imbalance", r["imbalance"], "<=", lim["eb_imbalance_max"]),
        check("ess_controls", r["ess"], ">=", max(lim["eb_ess_min_abs"], lim["eb_ess_min_frac"] * n_c)),
        check("max_weight_share_controls", r["max_share"], "<=", lim["eb_weight_share_max"]),
    ]
    diag = {"smd_main": _smd_before(study)}
    return gate_result("G7", "D", checks, diag=diag)


def g8_borrowing(est, prog, state):
    """Information only: could borrowing ever show equivalence within a margin, given the controls that are kept?"""
    sigma = {"model": prog["sigma"], "linear": prog["sigma_lin"]}.get(est["prognostic"], np.nan)
    mdm = min_detectable_margin(sigma, est["primary"]["ess"]) if np.isfinite(sigma) else np.nan
    feasible = bool(np.isfinite(mdm) and mdm < LIMITS["borrow_margin"])
    return gate_result(
        "G8",
        "D",
        [check("min_detectable_margin", mdm, "report", None)],
        note="borrowing could show equivalence within 5 points" if feasible else "borrowing cannot show equivalence",
        status="INFO",
    )
