import ast

import numpy as np
import pandas as pd

from src import causal as C
from src import models, survival
from src.causal.data import SOURCE_COLUMNS, TRIAL_COLUMNS, gate_frames
from src.demo import prepare
from src.gated.protocol import (
    AGE_MAX,
    ASSUMPTIONS,
    CONTROL_VALUE,
    DAG_EDGES,
    DAYS,
    ELIGIBILITY,
    ELIGIBILITY_STATUSES,
    ESTIMAND_FIELDS,
    ESTIMAND_ROLES,
    ESTIMANDS,
    FOOTER,
    GATE_NAMES,
    GATES,
    K_FOLDS,
    LANDMARK_DAYS,
    MEASUREMENT_REGISTER,
    MIN_CONTROLS,
    MIN_HIST,
    MIN_TREATED,
    MISSING_MAX,
    N_REAL,
    PRIOR_OUTCOME_EXPOSURE,
    R_ESTIMANDS,
    RECEPTOR_CUTS,
    REQUIRED_MAPPING,
    REQUIRED_RULE_COLUMNS,
    REWEIGHTED_ARMS,
    SEED,
    SENSITIVITY,
    SUBGROUPS,
    TARGET_TRIAL,
    TREATMENT,
    Z_S,
    Z_T,
)
from src.gated.runlog import StopRun, apply_action, audit, can_act, check_budget, gate, gate_status, log_fits, say


def stage_a(ctx, res):
    """G0 protocol lock, G1 target trial and estimands, G2 DAG and identification.

    All three gates are evaluated; then the run stops at the first one that failed.
    """
    say(ctx, "== Stage A: protocol and identification (no data) ==", 0.01)
    status = {"G0": _g0(ctx), "G1": _g1(ctx), "G2": _g2(ctx, res)}
    reasons = {
        "G0": "protocol lock check failed",
        "G1": "target trial or estimand specification is incomplete",
        "G2": "adjustment sets do not follow from the DAG",
    }
    for gid, st in status.items():
        if st == "FAIL":
            raise StopRun(gid, reasons[gid])


def _g0(ctx):
    backend = ctx["backend"]
    consts = {
        "TARGET_TRIAL": TARGET_TRIAL,
        "ELIGIBILITY": ELIGIBILITY,
        "ESTIMANDS": ESTIMANDS,
        "ESTIMAND_ROLES": ESTIMAND_ROLES,
        "DAG_EDGES": DAG_EDGES,
        "ASSUMPTIONS": ASSUMPTIONS,
        "GATES": GATES,
        "SENSITIVITY": SENSITIVITY,
        "SUBGROUPS": SUBGROUPS,
        "Z_T": Z_T,
        "Z_S": Z_S,
        "FOOTER": FOOTER,
    }
    n_const = sum(bool(v) for v in consts.values())
    backends = models.BACKENDS + ["logreg"]
    n_inputs = sum(len(v) == 64 for v in ctx["inputs"].values())
    note = "tier capped at SUPPORTIVE (PRIOR_OUTCOME_EXPOSURE)" if PRIOR_OUTCOME_EXPOSURE else ""
    if backend == "standin":
        note += "; TEST RUN with the random-forest stand-in (NOT TabPFN): tier NOT INTERPRETABLE"
    note += f"; earlier runs with this digest: {len(ctx['earlier_runs'])}"
    checks = [
        ("constants_present", n_const, "==", len(consts), n_const == len(consts)),
        ("protocol_digest", ctx["protocol_digest"][:10], "is", "sha256", len(ctx["protocol_digest"]) == 64),
        ("code_digest", ctx["code_digest"][:10], "is", "sha256", len(ctx["code_digest"]) == 64),
        ("input_digests", n_inputs, "==", 4, n_inputs == 4),
        ("backend", backend, "in", backends, backend in backends),
        ("run_index", ctx["run_index"], ">=", 1, None),
    ]
    return gate(ctx, "G0", GATE_NAMES["G0"], "all", gate_status(checks), checks, note=note.strip("; "))


def _g1(ctx):
    mode = ctx["mode"]
    cells = [v for comp in TARGET_TRIAL.values() for v in comp.values()]
    shape_ok = len(TARGET_TRIAL) == 9 and all(
        set(c) == {"target", "emulation_A", "emulation_B", "deviation"} for c in TARGET_TRIAL.values()
    )
    n_cells = sum(isinstance(v, str) and bool(v.strip()) for v in cells)
    n_elig = sum(
        r.get("status") in ELIGIBILITY_STATUSES
        and all(str(r.get(k, "")).strip() for k in ("id", "criterion", "A", "B"))
        for r in ELIGIBILITY
    )
    names = ["ATT", "ATE", "ATC", "ATO", "CATE"]
    n_attr = sum(bool(str(ESTIMANDS.get(k, {}).get(f, "")).strip()) for k in names for f in ESTIMAND_FIELDS)
    n_primary = sum(r == "primary" for scope in ESTIMAND_ROLES[mode].values() for r in scope.values())
    checks = [
        ("target_trial_cells_filled", n_cells, "==", 36, shape_ok and n_cells == 36),
        ("eligibility_classified", n_elig, "==", len(ELIGIBILITY), len(ELIGIBILITY) == 9 == n_elig),
        (
            "estimand_attributes_filled",
            n_attr,
            "==",
            len(names) * len(ESTIMAND_FIELDS),
            n_attr == len(names) * len(ESTIMAND_FIELDS),
        ),
        (f"primary_estimands_{mode}", n_primary, "==", 1, n_primary == 1),
        ("landmark_days_both_sources", LANDMARK_DAYS, ">", 0, float(LANDMARK_DAYS) > 0),
    ]
    return gate(
        ctx,
        "G1",
        GATE_NAMES["G1"],
        "all",
        gate_status(checks),
        checks,
        note="one landmark constant for both sources; min time > L re-checked in G3",
    )


def _g2(ctx, res):
    rep = C.dag_report(
        DAG_EDGES, ASSUMPTIONS, Z_T, Z_S, used_trial=TRIAL_COLUMNS.keys(), used_source=SOURCE_COLUMNS.keys()
    )
    res["dag"] = rep
    checks = [
        (k, rep[k], "==", True, rep[k] is True)
        for k in (
            "acyclic_full",
            "acyclic_work",
            "zt_valid_work",
            "zs_adm_work",
            "features_ok_trial",
            "features_ok_source",
            "probes_ok",
        )
    ]
    checks += [
        (k, rep[k], "report", None, None)
        for k in ("zt_valid_full", "zs_adm_full", "zs_plus_grade_adm", "zs_valid_trial", "censoring_ok_work")
    ]
    breaks = {}
    for t in rep["threats"]:
        breaks.setdefault(t["assumption"], set()).update(
            z for z, b in (("Z_t", t["breaks_zt"]), ("Z_s", t["breaks_zs"])) if b
        )
    threats = ", ".join(f"{a} {'+'.join(sorted(z, reverse=True)) or 'none'}" for a, z in breaks.items())
    note = (
        f"G_work {rep['n_edges_work']} of {rep['n_edges_full']} edges; minimal trial sets "
        f"{rep['minimal_trial_sets']}; minimal S-admissible sets {rep['minimal_s_sets']}; zs_valid_trial False is "
        f"expected (W3, needed by E only); restoring an edge breaks: {threats}"
    )
    return gate(ctx, "G2", GATE_NAMES["G2"], "all", gate_status(checks), checks, note=note)


def stage_b(ctx, res, df_a, df_b, mapping, rules, validate):
    """G3 data fitness, the frozen draw of kept controls, G4-G5 for scopes R and H, G6 external support.

    Returns
    -------
    dict
        Design state used by stages C and D: frames, arms, kept controls, design matrices, propensity score.
    """
    say(ctx, "== Stage B: data and design (outcome-blind) ==", 0.05)
    if TREATMENT not in df_a.columns:
        gate(ctx, "G3", GATE_NAMES["G3"], "all", "FAIL", [("treatment_column", TREATMENT, "in", "A", False)])
        raise StopRun("G3", f"data not fit for the protocol: treatment column '{TREATMENT}' missing in A")
    d = prepare(df_a, df_b, mapping, rules, TREATMENT, CONTROL_VALUE)
    fr = eligible_frames(d, df_a, df_b)
    res["funnel"] = fr["funnel"]
    status, problems = _g3(ctx, d, fr, df_a, mapping, rules)
    if status == "FAIL":
        raise StopRun("G3", "data not fit for the protocol: " + "; ".join(problems[:3]))
    s = _design_state(ctx["mode"], d, df_a, df_b, fr)
    balance = []
    if s["r_scope"]:
        _scope_r_design(ctx, res, s, balance)
    if s["hybrid"]:
        _scope_h_design(ctx, s, balance, validate)
        _external_support(ctx, res, s, balance)
    else:
        gate(ctx, "G6", GATE_NAMES["G6"], "X", "SKIPPED", [], note="internal mode: Rotterdam not used")
    res["balance"] = pd.concat(balance, ignore_index=True) if balance else pd.DataFrame()
    return s


def eligible_frames(d, df_a, df_b, age_max=AGE_MAX, landmark=LANDMARK_DAYS):
    """Eligible trial and Rotterdam frames with the protocol constants, ending with a "complete covariates" step.

    That funnel step removes the patients with a missing model covariate (Z_t and the cross-source columns in the
    trial, as the treated enter the source model; Z_s in Rotterdam) or a missing time or cens. Nothing is imputed;
    G3 fails if the step removes more than MISSING_MAX of a source. A missing time already fails the landmark step.

    Returns
    -------
    dict
        ``src.causal.data.gate_frames`` output (trial_cov, hist_cov, outcomes, funnel, landmark_ok) after that step.
    """
    fr = gate_frames(
        d,
        df_a,
        df_b,
        age_max=age_max,
        landmark=landmark,
        b_years=None,
        treatment=TREATMENT,
        control_value=CONTROL_VALUE,
        receptor_cuts=RECEPTOR_CUTS,
    )
    trial_cols = list(dict.fromkeys(list(TRIAL_COLUMNS.values()) + list(SOURCE_COLUMNS.values())))
    out, rows = {**fr, "outcomes": dict(fr["outcomes"])}, []
    for src, key, cols in (("A", "trial", trial_cols), ("B", "hist", list(SOURCE_COLUMNS.values()))):
        cov, outcomes = fr[f"{key}_cov"], fr["outcomes"][key]
        complete = cov[cols].notna().all(axis=1) & outcomes[["time", "cens"]].notna().all(axis=1)
        ok = complete.to_numpy() if len(cov) else np.zeros(0, bool)
        out[f"{key}_cov"] = cov[ok].reset_index(drop=True)
        out["outcomes"][key] = outcomes[ok].reset_index(drop=True)
        kept = out[f"{key}_cov"]
        rows.append(
            {
                "source": src,
                "step": "complete covariates",
                "n_after": int(ok.sum()),
                "n_removed": int((~ok).sum()),
                "n_treated": int((kept["A"] == 1).sum()),
                "n_control": int((kept["A"] == 0).sum()),
            }
        )
    out["funnel"] = (
        pd.concat([fr["funnel"], pd.DataFrame(rows)], ignore_index=True)
        .sort_values("source", kind="stable")
        .reset_index(drop=True)
    )
    return out


def _g3(ctx, d, fr, df_a, mapping, rules):
    """G3 data fitness. Returns the status and the problems found."""
    mode = ctx["mode"]
    bad_map = []
    for a in REQUIRED_MAPPING:
        e, pc = mapping.get(a) or {}, d["info"]["post_checks"].get(a) or {}
        if e.get("status") != "approved" or not e.get("b_cols"):
            bad_map.append(f"mapping {a}: not approved")
        elif a in d["info"]["errors"]:
            bad_map.append(f"mapping {a}: transform error")
        elif not pc.get("ok"):
            bad_map.append(f"mapping {a}: post-check not ok")
    invalid = d["invalid"]
    used = _rule_columns([r for r in rules if r.get("rule") not in {x.get("rule") for x in invalid}])
    no_rule = [c for c in REQUIRED_RULE_COLUMNS if c not in used]
    labels = _label_problems(df_a, d)
    num = {"age", "pnodes", "progrec", "estrec", "time", "cens"}
    miss_a = _missing_shares(
        df_a, ["age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec", "time", "cens"], num | {"tsize"}
    )
    miss_b = _missing_shares(
        d["hist"], ["age", "menostat", "tsize", "pnodes", "progrec", "estrec", "time", "cens"], num
    )
    f = fr["funnel"]
    cc = f[f.step == "complete covariates"].set_index("source")
    removed = {
        s: float(cc.loc[s, "n_removed"] / max(cc.loc[s, "n_removed"] + cc.loc[s, "n_after"], 1)) for s in ("A", "B")
    }
    T, B = fr["trial_cov"], fr["hist_cov"]
    n_t, n_c = int((T["A"] == 1).sum()), int((T["A"] == 0).sum())
    n_c_used = min(n_c, N_REAL) if mode == "hybrid" else n_c
    steps = {s: set(f.loc[f.source == s, "step"]) for s in ("A", "B")}
    worst_a, worst_b = max(miss_a, key=miss_a.get), max(miss_b, key=miss_b.get)
    checks = [
        ("mapping_rows_ok", len(REQUIRED_MAPPING) - len(bad_map), "==", len(REQUIRED_MAPPING), not bad_map),
        ("stage2_rule_columns_missing", len(no_rule), "==", 0, not no_rule),
        ("invalid_rules", len(invalid), "==", 0, not invalid),
        ("label_problems", len(labels), "==", 0, not labels),
        (f"max_missing_A[{worst_a}]", miss_a[worst_a], "<=", MISSING_MAX, miss_a[worst_a] <= MISSING_MAX),
        (f"max_missing_B[{worst_b}]", miss_b[worst_b], "<=", MISSING_MAX, miss_b[worst_b] <= MISSING_MAX),
        ("complete_covariates_removed_A", removed["A"], "<=", MISSING_MAX, removed["A"] <= MISSING_MAX),
        ("complete_covariates_removed_B", removed["B"], "<=", MISSING_MAX, removed["B"] <= MISSING_MAX),
        ("n_treated", n_t, ">=", MIN_TREATED, n_t >= MIN_TREATED),
        (
            "n_controls_used" + ("_kept" if mode == "hybrid" else ""),
            n_c_used,
            ">=",
            MIN_CONTROLS,
            n_c_used >= MIN_CONTROLS,
        ),
        ("n_hist", len(B), ">=", MIN_HIST, len(B) >= MIN_HIST),
        ("landmark_ok", fr["landmark_ok"], "==", True, bool(fr["landmark_ok"])),
        ("funnel_symmetric", steps["A"] == steps["B"], "==", True, steps["A"] == steps["B"]),
    ]
    problems = (
        bad_map
        + [f"no valid Stage-2 rule on '{c}'" for c in no_rule]
        + [f"invalid rule {x.get('id')}: {x.get('error')}" for x in invalid]
        + labels
        + [
            f"{s} missing values above the {MISSING_MAX:.0%} limit ({c})"
            for s, m in (("A", miss_a), ("B", miss_b))
            for c, v in m.items()
            if v > MISSING_MAX
        ]
        + [
            f"{s}: complete-covariates step removes more than {MISSING_MAX:.0%}"
            for s, v in removed.items()
            if v > MISSING_MAX
        ]
        + [
            f"too few {what}"
            for what, ok in (
                ("treated patients", n_t >= MIN_TREATED),
                ("controls", n_c_used >= MIN_CONTROLS),
                ("Rotterdam patients", len(B) >= MIN_HIST),
            )
            if not ok
        ]
        + ([] if fr["landmark_ok"] else ["landmark not met in every frame (min time <= L or empty frame)"])
        + ([] if steps["A"] == steps["B"] else ["funnel not symmetric"])
    )
    note = (
        "; ".join(problems[:6])
        if problems
        else "measurement register: "
        + " | ".join(m.split(":")[0] for m in MEASUREMENT_REGISTER)
        + " (handling in report.md)"
    )
    return gate(ctx, "G3", GATE_NAMES["G3"], "all", gate_status(checks), checks, note=note), problems


def _rule_columns(rules):
    """Column names referenced by the enabled Stage-2 rules."""
    cols = set()
    for r in rules:
        if r.get("enabled", True) is False or not r.get("rule"):
            continue
        try:
            tree = ast.parse(str(r["rule"]).strip(), mode="eval")
        except SyntaxError:
            continue
        cols |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return cols


def _label_problems(df_a, d):
    """Unknown or missing labels the design matrices rely on (never coded silently as a reference or "treated")."""
    out = []
    tr = df_a[TREATMENT]
    labs = sorted({str(v) for v in tr.dropna()})
    if tr.isna().any():
        out.append(f"A {TREATMENT}: {int(tr.isna().sum())} missing values")
    if CONTROL_VALUE not in labs or len(labs) != 2:
        out.append(f"A {TREATMENT}: labels {labs}, expected '{CONTROL_VALUE}' and one treated label")
    a_h, hist = d["a_h"], d["hist"]
    for src, frame, col, allowed in (
        ("A", df_a, "menostat", C.MENO_CODES),
        ("A", df_a, "tgrade", C.GRADE_CODES),
        ("A", a_h, "tsize", C.SIZE_CODES),
        ("B", hist, "menostat", C.MENO_CODES),
        ("B", hist, "tsize", C.SIZE_CODES),
    ):
        if col not in frame.columns:
            out.append(f"{src} {col}: column missing")
            continue
        found = sorted({str(v) for v in frame[col].dropna()})
        unknown = [v for v in found if v not in allowed]
        if unknown:
            out.append(f"{src} {col}: unknown labels {unknown[:5]}, expected {list(allowed)}")
        elif col == "tsize" and set(found) != set(allowed):
            out.append(f"{src} tsize groups {found}, expected exactly {list(allowed)}")
    for src, frame in (("A", df_a), ("B", hist)):
        for col in ("time", "cens"):
            if col not in frame.columns or frame[col].isna().all():
                out.append(f"{src} {col}: missing")
        if "cens" in frame.columns:
            vals = sorted(set(pd.to_numeric(frame["cens"], errors="coerce").dropna().unique()))
            if not set(vals) <= {0, 1}:
                out.append(f"{src} cens: values {vals[:5]}, expected 0/1")
    return out


def _missing_shares(df, cols, numeric):
    """Share of missing (or non-numeric) values per column; a missing column counts as 1."""
    out = {}
    for c in cols:
        if c not in df.columns or len(df) == 0:
            out[c] = 1.0
            continue
        s = pd.to_numeric(df[c], errors="coerce") if c in numeric else df[c]
        out[c] = float(s.isna().mean())
    return out


def _design_state(mode, d, df_a, df_b, fr):
    T, B = fr["trial_cov"], fr["hist_cov"]
    A, post = T["A"].to_numpy(int), T["post"].to_numpy(int)
    trt, ctrl = np.flatnonzero(A == 1), np.flatnonzero(A == 0)
    demo = mode == "hybrid" and len(ctrl) > N_REAL
    kept = hidden = None
    if mode == "hybrid":
        # Frozen draw: sorted positions of N_REAL controls with seed SEED, never a sidebar value.
        kept = np.sort(np.random.default_rng(SEED).choice(ctrl, N_REAL, replace=False)) if demo else ctrl
        hidden = np.setdiff1d(ctrl, kept)
    return {
        "d": d,
        "df_a": df_a,
        "df_b": df_b,
        "fr": fr,
        "T": T,
        "B": B,
        "A": A,
        "post": post,
        "trt": trt,
        "ctrl": ctrl,
        "kept": kept,
        "hidden": hidden,
        "demo": demo,
        "hybrid": mode == "hybrid",
        "r_scope": "primary" if mode == "internal" else ("reference" if demo else None),
        "Dt": C.design_trial(T),
        "Dx": C.design_trial_ext(T),
        "e": None,
        "ps_ext": False,
        "r_keep": [],
        "r_primary": "ATT" if mode == "internal" else None,
        "changed_from": None,
        "eb": None,
        "stab": None,
        "g6_post_fail": False,
    }


def _scope_r_design(ctx, res, s, balance):
    """G4-G5 for the scope-R estimands with actions B4.1 and B5.1; stops if the internal-mode primary fails."""
    internal, A, Dx = ctx["mode"] == "internal", s["A"], s["Dx"]
    say(ctx, "   within-trial propensity score (logistic, NOT TabPFN)", 0.08)
    e = C.ps_logistic(s["Dt"], A)
    g4 = {est: _g4_r(ctx, e, A, est, 1) for est in R_ESTIMANDS}
    if internal and g4["ATT"] == "FAIL":
        if not can_act(ctx, "B4.1"):
            raise StopRun("G4", "the primary ATT fails within-trial positivity and action B4.1 is not available")
        apply_action(ctx, "B4.1", note="primary ATT fails G4")
        s["r_primary"], s["changed_from"] = "ATO", "ATT"
        res["estimand_changed"] = True
        gate(
            ctx,
            "G4",
            GATE_NAMES["G4"],
            "R ATO",
            "PASS_AFTER_ACTION",
            [],
            2,
            "B4.1",
            "primary estimand changed from ATT to ATO (overlap population, not the treated); tier capped at SUPPORTIVE",
        )
    keep = [est for est in R_ESTIMANDS if g4[est] != "FAIL"]
    g5 = {est: _g5_r(ctx, e, A, Dx, est, 1, balance) for est in keep}
    if any(st != "PASS" for st in g5.values()) and can_act(ctx, "B5.1"):
        apply_action(ctx, "B5.1", note="weighted |SMD| > 0.10 or variance ratio out of range")
        e, s["ps_ext"] = C.ps_logistic(Dx, A), True
        g4 = {est: _g4_r(ctx, e, A, est, 2, "B5.1") for est in keep}
        keep = [est for est in keep if g4[est] != "FAIL"]
        g5 = {est: _g5_r(ctx, e, A, Dx, est, 2, balance, "B5.1") for est in keep}
    s["e"], s["r_keep"] = e, [est for est in keep if g5[est] != "FAIL"]
    if internal and s["r_primary"] not in s["r_keep"]:
        gid = "G4" if s["r_primary"] not in keep else "G5"
        raise StopRun(
            gid,
            f"the primary estimand {s['r_primary']} fails "
            f"{'within-trial positivity' if gid == 'G4' else 'covariate balance'}",
        )


def _g4_r(ctx, e, A, est, attempt, after_action=None):
    """G4 for one scope-R estimand: share outside the PS bounds, ESS and largest weight (ATO: report only)."""
    ov = C.overlap_summary(e, A, est, GATES["G4_ps_bounds"])
    checks = [
        (
            "share_outside",
            ov["share_outside"],
            "<=",
            GATES["G4_share_outside_max"],
            ov["share_outside"] <= GATES["G4_share_outside_max"],
        )
    ]
    for arm in REWEIGHTED_ARMS[est]:
        ratio = ov[f"ess_{arm}"] / ov[f"n_{arm}"]
        checks += [
            (f"ess_ratio_{arm}", ratio, ">=", GATES["G4_ess_ratio_min"], ratio >= GATES["G4_ess_ratio_min"]),
            (
                f"max_share_{arm}",
                ov[f"max_share_{arm}"],
                "<=",
                GATES["G4_weight_share_max"],
                ov[f"max_share_{arm}"] <= GATES["G4_weight_share_max"],
            ),
        ]
    if est == "ATO":
        checks = [(m, v, op, t, None) for m, v, op, t, _ in checks]
        status, note = "REPORT", "ATO weights are bounded; reported only"
    else:
        status, note = gate_status(checks), ""
    if status == "PASS" and after_action:
        status = "PASS_AFTER_ACTION"
    return gate(ctx, "G4", GATE_NAMES["G4"], f"R {est}", status, checks, attempt, after_action, note)


def _g5_r(ctx, e, A, Dx, est, attempt, balance, after_action=None):
    """G5 for one scope-R estimand: weighted |SMD| on D_t,ext and variance ratios; appends the balance table."""
    tab = C.smd_table(Dx, A, C.balancing_weights(e, A, est), C.DESIGN_TRIAL_EXT_NAMES)
    smd = float(np.max(np.abs(tab["smd_after"].to_numpy())))  # np.max keeps NaN -> FAIL
    vr = tab["vr_after"].dropna()
    lo, hi = GATES["G5_vr_range"]
    vr_ok = bool(((vr >= lo) & (vr <= hi)).all())
    checks = [
        ("max_abs_smd", smd, "<=", GATES["G5_smd_max"], smd <= GATES["G5_smd_max"]),
        ("variance_ratio_range", (float(vr.min()), float(vr.max())) if len(vr) else None, "within", (lo, hi), vr_ok),
    ]
    if smd <= GATES["G5_smd_max"] and vr_ok:
        status = "PASS_AFTER_ACTION" if after_action else "PASS"
    elif smd <= GATES["G5_smd_conditional_max"]:
        status = "CONDITIONAL"
    else:
        status = "FAIL"
    note = "DR estimator used anyway; residual imbalance logged" if status == "CONDITIONAL" else ""
    gate(ctx, "G5", GATE_NAMES["G5"], f"R {est}", status, checks, attempt, after_action, note)
    balance.append(tab.assign(scope="R", estimand=est, attempt=attempt))
    return status


def _scope_h_design(ctx, s, balance, validate):
    """G4 and G5 for scope H: entropy balancing of the kept controls to the treated (no design actions in H)."""
    A, Dt, Dx, trt, kept = s["A"], s["Dt"], s["Dx"], s["trt"], s["kept"]
    tol = GATES["G4H_imbalance_max"]
    eb = C.entropy_balance(Dt[kept], Dt[trt].mean(0), tol)
    ess_min = max(GATES["G4H_ess_min_abs"], len(kept) * GATES["G4H_ess_min_frac"])
    checks = [
        ("imbalance", eb["imbalance"], "<", tol, eb["converged"]),
        ("ess", eb["ess"], ">=", ess_min, eb["ess"] >= ess_min),
        (
            "max_share",
            eb["max_share"],
            "<=",
            GATES["G4H_weight_share_max"],
            eb["max_share"] <= GATES["G4H_weight_share_max"],
        ),
    ]
    status = gate(
        ctx,
        "G4",
        GATE_NAMES["G4"],
        "H",
        gate_status(checks),
        checks,
        note=f"{len(kept)} kept controls (frozen draw: seed {SEED}, sorted positions); no actions in H",
    )
    if status == "FAIL":
        raise StopRun("G4", "hybrid not estimable for this sample")
    s["eb"] = eb
    idx = np.r_[trt, kept]
    tab = C.smd_table(Dx[idx], A[idx], np.r_[np.ones(len(trt)), eb["w"]], C.DESIGN_TRIAL_EXT_NAMES)
    smd = np.abs(tab["smd_after"].to_numpy())
    smd_dt, smd_ext = float(np.max(smd[: Dt.shape[1]])), float(np.max(smd[Dt.shape[1] :]))
    checks = [
        ("max_abs_smd_Dt", smd_dt, "<", tol, smd_dt < tol),
        ("max_abs_smd_extras", smd_ext, "<=", GATES["G5_smd_max"], None),
    ]
    if not smd_dt < tol:
        gate(ctx, "G5", GATE_NAMES["G5"], "H", "FAIL", checks, note="exact balance failed: code error")
        raise StopRun("G5", "entropy-balancing balance check failed (code error)")
    warn = smd_ext > GATES["G5_smd_max"]
    gate(
        ctx,
        "G5",
        GATE_NAMES["G5"],
        "H",
        "CONDITIONAL" if warn else "PASS",
        checks,
        note="WARN: D_t,ext extras above 0.10 (doubly robust estimator)" if warn else "exact balance on D_t",
    )
    balance.append(tab.assign(scope="H", estimand="ATT", attempt=1))
    if s["demo"] and validate:
        say(ctx, "   draw stability of the kept-control draw (outcome-free)")
        s["stab"] = C.draw_stability(Dt, A, len(kept), GATES["draw_stability_reps"], SEED, GATES)


def _external_support(ctx, res, s, balance):
    """G6 external support per menopausal stratum and the W3 probe (exploratory: never stops the run)."""
    T, B, A, trt = s["T"], s["B"], s["A"], s["trt"]
    Ds_T, Ds_B = C.design_source(T), C.design_source(B)
    strata_T = np.where(s["post"][trt] == 1, "post", "pre")
    strata_B = np.where(B["post"].to_numpy() == 1, "post", "pre")
    sup = C.source_support(Ds_T[trt], Ds_B, strata_T, strata_B, seed=SEED, k=K_FOLDS, cut=GATES["G6_ratio_cut"])
    rows = []
    for r in sup["table"].to_dict("records"):
        checks = [
            (
                "share_unsupported",
                r["share_unsupported"],
                "<=",
                GATES["G6_share_unsupported_max"],
                r["share_unsupported"] <= GATES["G6_share_unsupported_max"],
            ),
            ("ess_B", r["ess_B"], ">=", GATES["G6_ess_min"], r["ess_B"] >= GATES["G6_ess_min"]),
            (
                "max_share",
                r["max_share"],
                "<=",
                GATES["G6_weight_share_max"],
                r["max_share"] <= GATES["G6_weight_share_max"],
            ),
        ]
        note = f"AUC {sup['auc']:.3f}; density ratio cut {GATES['G6_ratio_cut']:g}" if r["stratum"] == "all" else ""
        status = gate(ctx, "G6", GATE_NAMES["G6"], f"X {r['stratum']}", gate_status(checks), checks, note=note)
        rows.append({**r, "auc": sup["auc"], "status": status})
    rows.append(_w3_probe(ctx, T, A, Ds_T, balance))
    res["support"] = pd.DataFrame(rows)
    s["g6_post_fail"] = any(r.get("stratum") == "post" and r["status"] == "FAIL" for r in rows)
    s["Ds_T"], s["Ds_B"] = Ds_T, Ds_B


def _w3_probe(ctx, T, A, Ds_T, balance):
    """G6 W3 probe: grade and size imbalance left after Z_s-only ATT weighting (outcome-blind)."""
    probe = pd.DataFrame({f"grade_{g}": (T["tgrade"] == g).astype(float) for g in C.GRADE_CODES})
    probe["tsize_mm"] = T["tsize_mm"].astype(float)
    w3 = C.w3_probe(Ds_T, A, probe)
    m = float(np.max(np.abs(w3["smd_after"].to_numpy())))
    checks = [("max_abs_smd_grade_size", m, "<=", GATES["G6_w3_smd_max"], m <= GATES["G6_w3_smd_max"])]
    status = gate(
        ctx,
        "G6",
        GATE_NAMES["G6"],
        "X W3",
        gate_status(checks),
        checks,
        note="W3 probe: grade and size after Z_s-only ATT weighting (outcome-blind)",
    )
    balance.append(w3.assign(scope="W3", estimand="ATT", attempt=1))
    return {"stratum": "W3 probe", "max_abs_smd": m, "status": status}


def stage_c(ctx, res, s, horizons, cache_dir):
    """Read the Rotterdam outcomes; G7 prognostic model credibility and G8 borrowing feasibility per horizon."""
    if not s["hybrid"]:
        for gid in ("G7", "G8"):
            gate(ctx, gid, GATE_NAMES[gid], "X", "SKIPPED", [], note="internal mode: Rotterdam not used")
        return
    audit(ctx, "read_external_outcomes", what="Rotterdam time/cens")
    say(ctx, "== Stage C: Rotterdam outcomes read (trial outcomes still blinded) ==", 0.15)
    B, T, outcomes = s["B"], s["T"], s["fr"]["outcomes"]["hist"]
    time_B, event_B = outcomes["time"].to_numpy(float), outcomes["cens"].to_numpy(float)
    fine_B, coarse_B = survival.strata_labels("B", np.zeros(len(B), int), B["post"].to_numpy(int))
    F_B, F_T = C.features_source(B), C.features_source(T)
    folds_B = C.make_folds(B["post"].to_numpy(), K_FOLDS, SEED)
    s.update(thetaB={}, usedB={}, prog={}, choice={}, g8={})
    for i, h in enumerate(horizons):
        s["thetaB"][h], s["usedB"][h] = pseudo(time_B, event_B, h * DAYS, fine_B, coarse_B)
        s["prog"][h] = _fit_prognostic_score(ctx, s, h, 0.15 + 0.2 * i / len(horizons), F_B, F_T, folds_B, cache_dir)
        s["choice"][h] = _g7(ctx, h, s["prog"][h])
        s["g8"][h] = _g8(ctx, s, h)
    h0 = horizons[0]
    res["labels"]["prognostic"] = prognostic_label(s["choice"][h0], ctx["backend"])
    res["g8"] = s["g8"][h0]
    res["diagnostics"]["prognostic"] = {
        f"{h:g}y": {
            k: s["prog"][h][k]
            for k in (
                "mse",
                "mse_lin",
                "mse_null",
                "se_diff",
                "sigma",
                "sigma_lin",
                "cal_int",
                "cal_int_ci",
                "cal_slope",
                "cal_slope_ci",
            )
        }
        | {"choice": s["choice"][h]}
        for h in horizons
    }


def _fit_prognostic_score(ctx, s, h, frac, F_B, F_T, folds_B, cache_dir):
    """Prognostic score m_B: cross-fitted on Rotterdam with the model backend, predicted for the trial patients."""
    backend = ctx["backend"]
    planned = K_FOLDS + 1 if backend.startswith("tabpfn") else 0
    check_budget(ctx, planned, f"prognostic score {h:g} y")
    say(ctx, f"   fitting the prognostic score m_B at {h:g} y ({planned} TabPFN fits; {backend_label(backend)})", frac)
    p = C.prognostic_score(
        F_B,
        s["Ds_B"],
        s["thetaB"][h],
        F_T,
        s["Ds_T"],
        folds_B,
        backend,
        SEED,
        C.SOURCE_CAT_IDX,
        purpose=f"mB_{h:g}y",
        cache_dir=cache_dir,
    )
    log_fits(ctx)
    return p


def _g7(ctx, h, p):
    """G7 prognostic model credibility, with action B7.1 or B7.2 on failure; returns "model", "linear" or "none"."""
    limit = p["mse_lin"] + GATES["G7_se_mult"] * p["se_diff"]
    ok_rel, ok_null = p["mse"] <= limit, p["mse"] < p["mse_null"]
    checks = [
        ("cv_mse_model", p["mse"], "<=", limit, ok_rel),
        ("cv_mse_model_vs_null", p["mse"], "<", p["mse_null"], ok_null),
        ("cv_mse_linear", p["mse_lin"], "report", None, None),
        ("cal_slope", p["cal_slope"], "report", None, None),
    ]
    choice, status, action, note = "model", "PASS", None, ""
    if not (ok_rel and ok_null):
        want, aid = ("linear", "B7.1") if p["mse_lin"] < p["mse_null"] else ("none", "B7.2")
        if aid in ctx["actions"]:
            choice, status, action, note = want, "PASS_AFTER_ACTION", aid, f"{aid} already applied"
        elif can_act(ctx, aid):
            apply_action(ctx, aid, note=f"m_B at {h:g} y")
            choice, status, action = want, "PASS_AFTER_ACTION", aid
        else:
            status, note = "FAIL", f"{aid} not applied: design-action cap reached; model kept (efficiency only)"
    gate(
        ctx,
        "G7",
        GATE_NAMES["G7"],
        f"H {h:g}y",
        status,
        checks,
        action=action,
        note=(note + "; " if note else "") + f"m_B: {prognostic_label(choice, ctx['backend'])}",
    )
    return choice


def _g8(ctx, s, h):
    """G8: minimum detectable margin of borrowing with the kept controls (borrowing itself is never used)."""
    choice, p = s["choice"][h], s["prog"][h]
    sigma = (
        p["sigma"]
        if choice == "model"
        else p["sigma_lin"]
        if choice == "linear"
        else float(np.std(s["thetaB"][h], ddof=1))
    )
    mdm = C.min_detectable_margin(sigma, s["eb"]["ess"], GATES["G8_alpha"])
    feasible = mdm < GATES["G8_margin"]
    g8 = {"status": "FEASIBLE" if feasible else "INFEASIBLE", "mdm": mdm, "sigma_B": sigma}
    gate(
        ctx,
        "G8",
        GATE_NAMES["G8"],
        f"X {h:g}y",
        g8["status"],
        [
            ("min_detectable_margin", mdm, "<", GATES["G8_margin"], feasible),
            ("sigma_B", sigma, "report", None, None),
            ("ess_C", s["eb"]["ess"], "report", None, None),
        ],
        note="borrowing not used: it needs a protocol amendment" if feasible else "borrowing not used (infeasible)",
    )
    return g8


def pseudo(t, ev, days, fine, coarse):
    """Kaplan-Meier pseudo-observations within strata (too small fine strata are merged into coarse ones)."""
    return survival.pseudo_by_strata(t, ev, days, fine, coarse, GATES["G9_min_n"], GATES["G9_min_at_risk"])


def backend_label(backend):
    return models.BACKEND_LABELS.get(backend, backend)


def prognostic_label(choice, backend):
    if choice == "model":
        return backend_label(backend)
    if choice == "linear":
        return "linear prognostic score on D_s (NOT TabPFN)"
    return "none: grade and size terms only (NOT TabPFN)"
