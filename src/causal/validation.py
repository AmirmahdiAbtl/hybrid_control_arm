import numpy as np
import pandas as pd

from src.causal.data import DESIGN_TRIAL_NAMES, TRIAL_CAT_IDX, design_trial, features_trial, recalibration_design
from src.causal.estimators import aipw, hajek_eb_att, robust_hybrid_att
from src.causal.nuisance import _crossfit_ols, crossfit_tlearner
from src.causal.weights import entropy_balance, make_folds, ps_logistic, smd_table
from src.models import CHEAP_LABEL
from src.survival import pseudo_by_strata, strata_labels

# Shown on every row of the validation and placebo tables, per repetition and in the summaries.
VALIDATION_NOTE = (
    CHEAP_LABEL + "; fixed prognostic-score predictions (if any); checks formulas and variance, not TabPFN"
)
PLACEBO_NOTE = CHEAP_LABEL + "; checks formulas and variance, not TabPFN"


def hidden_control_validation(trial_cov, outcomes, m_B_trial, horizon, n_real, reps, seed):
    """Hide-and-recover check of the robust hybrid (demo only; report only).

    Each repetition splits the eligible controls at random into ``n_real`` kept and the rest hidden. S0 is
    estimated on treated + kept with the robust hybrid (fixed m_B predictions) and with the Hajek entropy-balanced
    small arm. The truth is the AIPW-ATT S0 on treated + hidden, with a logistic propensity on D_t fitted once on
    all trial patients and mu0 a cross-fitted OLS on the hidden controls. Pseudo-observations are computed
    separately for the kept and the hidden controls.

    Parameters
    ----------
    trial_cov : pandas.DataFrame
    outcomes : pandas.DataFrame
        ``time`` and ``cens`` row-aligned with ``trial_cov``.
    m_B_trial : array-like or None
        Fixed prognostic-score predictions for all trial rows (None = no score).
    horizon : float
        Days.
    n_real, reps, seed : int

    Returns
    -------
    pandas.DataFrame
        One row per repetition and method: rep, method, est, se, truth, se_truth, diff (= est - truth), ess, note
        (``VALIDATION_NOTE``).
    """
    A, post = trial_cov["A"].to_numpy(int), trial_cov["post"].to_numpy(int)
    t, ev = outcomes["time"].to_numpy(float), outcomes["cens"].to_numpy(float)
    D, Z = design_trial(trial_cov), recalibration_design(trial_cov, m_B_trial)
    meno = trial_cov["menostat"].to_numpy(object)
    e_all = ps_logistic(D, A)
    trt, ctrl = np.flatnonzero(A == 1), np.flatnonzero(A == 0)
    if len(ctrl) <= n_real:
        raise ValueError("hide-and-recover needs more eligible controls than n_real")
    theta_T = _trial_theta(t, ev, A, post, trt, horizon)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        perm = rng.permutation(ctrl)
        kept, hidden = np.sort(perm[:n_real]), np.sort(perm[n_real:])
        theta_K = _trial_theta(t, ev, A, post, kept, horizon)
        hybrid = robust_hybrid_att(theta_T, theta_K, D[trt], D[kept], Z[trt], Z[kept], make_folds(meno[kept]))
        hajek = hajek_eb_att(theta_T, theta_K, hybrid["w"])
        idx = np.r_[trt, hidden]
        theta = np.r_[theta_T, _trial_theta(t, ev, A, post, hidden, horizon)]
        folds = make_folds(strata_labels("A", A[idx], post[idx])[0])
        mu0 = _crossfit_ols(D[idx], theta, A[idx] == 0, folds)
        truth = aipw(theta, A[idx], e_all[idx], mu0, mu0, "ATT")
        se_truth = float(np.sqrt(np.sum(truth["if_s0"] ** 2)))
        for method, r in (("robust hybrid", hybrid), ("Hajek EB small arm", hajek)):
            rows.append(
                {
                    "rep": rep,
                    "method": method,
                    "est": r["s0"],
                    "se": r["se_s0"],
                    "truth": truth["s0"],
                    "se_truth": se_truth,
                    "diff": r["s0"] - truth["s0"],
                    "ess": hybrid["ess"],
                    "note": VALIDATION_NOTE,
                }
            )
    return pd.DataFrame(rows)


def _trial_theta(t, ev, A, post, idx, horizon):
    """Pseudo-observations of trial units ``idx`` within arm x menostat strata (small strata merged)."""
    fine, coarse = strata_labels("A", A[idx], post[idx])
    return pseudo_by_strata(t[idx], ev[idx], horizon, fine, coarse)[0]


def summarize_validation_runs(runs):
    """Per-method summary of :func:`hidden_control_validation`, in percentage points (report only).

    Returns
    -------
    pandas.DataFrame
        method, reps, bias (mean diff) and bias_mcse (sd / sqrt(R)), SD, RMSE, coverage of
        |diff| <= 1.96 sqrt(se^2 + se_truth^2) and coverage_mcse (sqrt(p(1 - p) / R)), note (``VALIDATION_NOTE``).
    """
    out = []
    for method, g in runs.groupby("method", sort=False):
        d = g["diff"].to_numpy(float) * 100
        R = len(d)
        covered = np.abs(g["diff"]) <= 1.96 * np.sqrt(g["se"] ** 2 + g["se_truth"] ** 2)
        p = float(covered.mean())
        sd = float(d.std(ddof=1)) if R > 1 else np.nan
        out.append(
            {
                "method": method,
                "reps": R,
                "bias": float(d.mean()),
                "bias_mcse": sd / np.sqrt(R),
                "SD": sd,
                "RMSE": float(np.sqrt(np.mean(d**2))),
                "coverage": p,
                "coverage_mcse": float(np.sqrt(p * (1 - p) / R)),
                "note": VALIDATION_NOTE,
            }
        )
    return pd.DataFrame(out)


def placebo_test(trial_cov, outcomes, horizon, reps, n_pseudo, seed):
    """Placebo test among the eligible trial controls, where the true effect is 0 (report only).

    Each repetition draws ``n_pseudo`` pseudo-treated controls without replacement, with probability proportional
    to e / (1 - e) (logistic propensity on D_t of all trial patients). Pseudo-observations are recomputed within
    pseudo-arm x menostat strata, and the ATT is estimated by AIPW with a logistic propensity refitted on the
    controls and a cross-fitted "logreg" T-learner.

    Returns
    -------
    dict
        ``runs`` (rep, est, se, lo, hi, reject, note) and ``summary`` (one row: reps, type_I_error and MCSE,
        coverage and MCSE, bias / SD / mean SE in points with bias MCSE, note), with ``PLACEBO_NOTE`` on every row.
    """
    A = trial_cov["A"].to_numpy(int)
    D = design_trial(trial_cov)
    e = ps_logistic(D, A)
    ctrl = np.flatnonzero(A == 0)
    p = (e / (1 - e))[ctrl]
    controls = trial_cov.iloc[ctrl].reset_index(drop=True)
    t, ev = outcomes["time"].to_numpy(float)[ctrl], outcomes["cens"].to_numpy(float)[ctrl]
    post, D_C, F_C = controls["post"].to_numpy(int), D[ctrl], features_trial(controls)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        A_pseudo = np.zeros(len(ctrl), int)
        A_pseudo[rng.choice(len(ctrl), n_pseudo, replace=False, p=p / p.sum())] = 1
        fine, coarse = strata_labels("P", A_pseudo, post)
        theta = pseudo_by_strata(t, ev, horizon, fine, coarse)[0]
        mu = crossfit_tlearner(
            F_C, theta, A_pseudo, make_folds(fine, 5, seed), "logreg", seed, TRIAL_CAT_IDX, arms=(0,), purpose="placebo"
        )
        r = aipw(theta, A_pseudo, ps_logistic(D_C, A_pseudo), mu[0], mu[0], "ATT")
        rows.append(
            {
                "rep": rep,
                "est": r["est"],
                "se": r["se"],
                "lo": r["lo"],
                "hi": r["hi"],
                "reject": bool(not r["lo"] <= 0 <= r["hi"]),
                "note": PLACEBO_NOTE,
            }
        )
    runs = pd.DataFrame(rows)
    R, rejection_rate, est = len(runs), float(runs["reject"].mean()), runs["est"].to_numpy() * 100
    mcse = float(np.sqrt(rejection_rate * (1 - rejection_rate) / R))
    sd = float(est.std(ddof=1)) if R > 1 else np.nan
    summary = pd.DataFrame(
        [
            {
                "reps": R,
                "n_pseudo": n_pseudo,
                "type_I_error": rejection_rate,
                "type_I_mcse": mcse,
                "coverage": 1 - rejection_rate,
                "coverage_mcse": mcse,
                "bias": float(est.mean()),
                "bias_mcse": sd / np.sqrt(R),
                "SD": sd,
                "mean_se": float(runs["se"].mean() * 100),
                "note": PLACEBO_NOTE,
            }
        ]
    )
    return {"runs": runs, "summary": summary}


def draw_stability(D_t, A, n_real, reps, seed, thresholds):
    """Outcome-free share of random kept-control draws in which the H rows of gates G4 and G5 pass.

    Each draw is ``np.sort(rng.choice(controls, n_real, replace=False))`` from one ``default_rng(seed)``, i.e.
    sorted positions as in the frozen draw of the gated run.

    Parameters
    ----------
    D_t : numpy.ndarray
        :func:`design_trial` of all trial patients.
    A : array-like
    n_real, reps, seed : int
    thresholds : dict
        Needs ``G4H_imbalance_max``, ``G4H_ess_min_abs``, ``G4H_ess_min_frac`` and ``G4H_weight_share_max``
        (e.g. ``src.gated.protocol.GATES``).

    Returns
    -------
    dict
        reps, share_pass (and MCSE), share_exact (exact balance reached), and over the exactly balanced draws
        ess_p5, ess_median and max_share_p95.
    """
    D_t, A = np.asarray(D_t, float), np.asarray(A).astype(int)
    trt, ctrl = np.flatnonzero(A == 1), np.flatnonzero(A == 0)
    target = D_t[trt].mean(0)
    tol = thresholds["G4H_imbalance_max"]
    ess_min = max(thresholds["G4H_ess_min_abs"], n_real * thresholds["G4H_ess_min_frac"])
    arm = np.r_[np.ones(len(trt)), np.zeros(n_real)]
    rng = np.random.default_rng(seed)
    passed, exact, ess, share = [], [], [], []
    for _ in range(reps):
        kept = np.sort(rng.choice(ctrl, n_real, replace=False))
        eb = entropy_balance(D_t[kept], target, tol)
        smd = smd_table(np.vstack([D_t[trt], D_t[kept]]), arm, np.r_[np.ones(len(trt)), eb["w"]], DESIGN_TRIAL_NAMES)[
            "smd_after"
        ].to_numpy()
        # np.max keeps a NaN SMD, so such a draw fails (pandas max would skip it).
        max_smd = np.max(np.abs(smd))
        g4 = eb["converged"] and eb["ess"] >= ess_min and eb["max_share"] <= thresholds["G4H_weight_share_max"]
        passed.append(bool(g4 and max_smd < tol))
        exact.append(eb["converged"])
        ess.append(eb["ess"])
        share.append(eb["max_share"])
    p = float(np.mean(passed))
    ok = np.asarray(exact)
    ess, share = np.asarray(ess)[ok], np.asarray(share)[ok]
    return {
        "reps": reps,
        "n_real": n_real,
        "share_pass": p,
        "share_pass_mcse": float(np.sqrt(p * (1 - p) / reps)),
        "share_exact": float(ok.mean()),
        "ess_p5": float(np.percentile(ess, 5)) if ok.any() else np.nan,
        "ess_median": float(np.median(ess)) if ok.any() else np.nan,
        "max_share_p95": float(np.percentile(share, 95)) if ok.any() else np.nan,
        "note": "outcome-free (entropy balancing only)",
    }
