import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import causal
from src.gated.protocol import GATES
from src.models import (
    BACKEND_LABELS,
    CHEAP_LABEL,
    NON_COMPARABLE,
    encode_features,
    fit_predict,
    fit_predict_reg,
    is_categorical,
)
from src.survival import event_free_labels, km_at, km_ci, pseudo_km

# Support rule of tab 2: the thresholds of gate G6 (a treated patient with density ratio r above the cut has no support)
SUPPORT_RATIO_CUT = GATES["G6_ratio_cut"]
SUPPORT_SHARE_MAX = GATES["G6_share_unsupported_max"]
SUPPORT_ESS_MIN = GATES["G6_ess_min"]
SUPPORT_WEIGHT_SHARE_MAX = GATES["G6_weight_share_max"]

# Method rows of the estimate (tab 3) and validation (tab 4) tables
TREATED_ROW = "Treated, observed (KM)"
HYBRID_ROW = "Robust hybrid: DR + prognostic score"
SMALL_ARM_ROW = "Small control arm, ATT-weighted (entropy balancing)"
HISTORICAL_ROW = "Historical + model (no real controls; exploratory)"
LEGACY_ROWS = {
    "full": "Legacy: full correction (exploratory)",
    "dynamic": "Legacy: dynamic borrowing (exploratory)",
    "joint": "Legacy: joint model (exploratory)",
}
REFERENCE_ROW = "Reference (demo only): ATT-weighted mean of all trial controls"

_MAX_LOGIT_SHIFT = 6.0  # bisection bracket of the legacy logit shift


# ----------------------------------------------------------------------------
# Tab 2: comparability
# ----------------------------------------------------------------------------
def comparability(hist, trial, features, backend, seed=0, n_splits=5, progress=None):
    """Check whether the eligible historical patients cover the treated trial patients (tab 2).

    A classifier separates the treated (label 1) from the historical patients (label 0) with stratified
    out-of-fold probabilities pi. The density ratio r = pi / (1 - pi) * n_hist / n_treated does not depend on the
    group sizes; a treated patient with r > 20 has no support. The historical patients' r are the weights that
    make them look like the treated; their Kish ESS and largest share show how many of them really contribute.

    Parameters
    ----------
    hist, trial : pandas.DataFrame
        Eligible historical patients and trial patients (``arm`` = 'control' / 'treated'); only the treated
        trial patients are compared.
    features : list of str
        Baseline features that mean the same in both sources (see :func:`src.models.cross_source_features`).
    backend : str
        Classifier backend (see :func:`src.models.fit_predict`); one fit per fold.
    progress : callable, optional
        Called with the fraction of folds done.

    Returns
    -------
    dict
        ``auc`` (out-of-fold, treated vs historical); ``support`` (rows "all" and each ``menostat`` level of the
        treated: stratum, n_treated, n_hist, share_unsupported, ess_hist, max_share, ok; a row is ok if
        share_unsupported <= 0.10, ess_hist >= 50 and max_share <= 0.10); ``failing`` (strata that are not ok);
        ``verdict`` ("supported" or "limited support: <failing>"); ``share_unsupported`` (row "all"); ``oof``
        (out-of-fold P(treated) of the treated); ``pi_cut`` (P(treated) at r = 20, for plots); ``features``
        (:func:`feature_table`, historical vs treated).
    """
    treated = trial[trial["arm"] == "treated"]
    n_h, n_t = len(hist), len(treated)
    y = np.r_[np.zeros(n_h), np.ones(n_t)].astype(int)
    pi = _out_of_fold_probabilities(
        pd.concat([hist, treated], ignore_index=True), y, features, backend, seed, n_splits, progress
    )
    p = np.clip(pi, 1e-6, 1 - 1e-6)  # a forest can predict exactly 0 or 1
    ratio = p / (1 - p) * n_h / n_t
    support = _support_table(hist, treated, ratio[:n_h], ratio[n_h:])
    failing = support.loc[~support["ok"], "stratum"].tolist()
    return {
        "auc": float(roc_auc_score(y, pi)),
        "support": support,
        "failing": failing,
        "verdict": ("limited support: " + ", ".join(failing)) if failing else "supported",
        "share_unsupported": float(support["share_unsupported"].iloc[0]),
        "oof": pi[n_h:],
        "pi_cut": SUPPORT_RATIO_CUT * n_t / (n_h + SUPPORT_RATIO_CUT * n_t),
        "features": feature_table(hist, treated, features).rename(columns={"trial": "treated"}),
    }


def _out_of_fold_probabilities(patients, y, features, backend, seed, n_splits, progress):
    (X,), cat_idx = encode_features([patients], features)
    pi = np.zeros(len(y))
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    for k, (train, test) in enumerate(folds):
        pi[test] = fit_predict(backend, X[train], y[train], [X[test]], seed, cat_idx, purpose=f"comparability_fold{k}")[
            0
        ]
        if progress:
            progress((k + 1) / n_splits)
    return pi


def _support_table(hist, treated, ratio_hist, ratio_treated):
    """Support rule for all patients and for each menopausal status of the treated."""
    strata = [("all", np.ones(len(hist), bool), np.ones(len(treated), bool))]
    if "menostat" in hist.columns and "menostat" in treated.columns:
        meno_h, meno_t = hist["menostat"].astype(str).to_numpy(), treated["menostat"].astype(str).to_numpy()
        strata += [(lv, meno_h == lv, meno_t == lv) for lv in sorted(set(treated["menostat"].dropna().astype(str)))]
    rows = []
    for name, in_h, in_t in strata:
        share = float(np.mean(ratio_treated[in_t] > SUPPORT_RATIO_CUT))
        ess, top = causal.kish_ess(ratio_hist[in_h]), causal.max_weight_share(ratio_hist[in_h])
        rows.append(
            {
                "stratum": name,
                "n_treated": int(in_t.sum()),
                "n_hist": int(in_h.sum()),
                "share_unsupported": share,
                "ess_hist": ess,
                "max_share": top,
                "ok": bool(share <= SUPPORT_SHARE_MAX and ess >= SUPPORT_ESS_MIN and top <= SUPPORT_WEIGHT_SHARE_MAX),
            }
        )
    return pd.DataFrame(rows)


def feature_table(hist, trial, features):
    """Per-feature comparison of historical and trial patients.

    Returns
    -------
    pandas.DataFrame
        feature; hist and trial (mean, or the shares of the three most common levels for text columns); smd
        (standardized mean difference; for text columns the largest over levels); auc (single-feature AUC for
        telling the groups apart, >= 0.5).
    """
    y = np.r_[np.zeros(len(hist)), np.ones(len(trial))]
    rows = []
    for f in features:
        both = pd.concat([hist[f], trial[f]], ignore_index=True)
        compare = _compare_categorical if is_categorical([both]) else _compare_numeric
        rows.append({"feature": f, **compare(both, len(hist), y)})
    return pd.DataFrame(rows)


def _compare_categorical(values, n_hist, y):
    s = values.astype(str).where(values.notna())
    h, t = s[:n_hist], s[n_hist:]
    smds, shares = [], []
    for lv in sorted(values.dropna().astype(str).unique()):
        in_h, in_t = (h == lv).astype(float).to_numpy(), (t == lv).astype(float).to_numpy()
        smds.append(_standardized_mean_diff(in_h, in_t))
        shares.append((lv, in_h.mean(), in_t.mean()))
    # AUC score of a level: its share of trial patients; a missing value scores 0.5
    known = s.notna()
    score = s.map(pd.Series(y[known], index=s[known].index).groupby(s[known]).mean()).fillna(0.5)
    top = sorted(shares, key=lambda d: -(d[1] + d[2]))[:3]
    return {
        "hist": ", ".join(f"{lv} {a:.0%}" for lv, a, _ in top),
        "trial": ", ".join(f"{lv} {b:.0%}" for lv, _, b in top),
        "smd": max(smds, key=abs) if smds else 0.0,
        "auc": _symmetric_auc(y, score.to_numpy()),
    }


def _compare_numeric(values, n_hist, y):
    v = values.astype(float)
    return {
        "hist": f"{np.nanmean(v[:n_hist]):.2f}",
        "trial": f"{np.nanmean(v[n_hist:]):.2f}",
        "smd": _standardized_mean_diff(v[:n_hist], v[n_hist:]),
        "auc": _symmetric_auc(y, v.fillna(v.median()).to_numpy()),
    }


def _standardized_mean_diff(a, b):
    sd = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(b) - np.nanmean(a)) / sd) if sd > 0 else 0.0


def _symmetric_auc(y, score):
    """AUC flipped to >= 0.5: only how well the score separates the groups matters, not its direction."""
    auc = roc_auc_score(y, score)
    return float(max(auc, 1 - auc))


# ----------------------------------------------------------------------------
# Tabs 3-4: shared inputs
# ----------------------------------------------------------------------------
def prepare_arms(hist, trial, features, horizon_days):
    """Encode the features and compute the outcomes shared by the estimate and validation tabs.

    Parameters
    ----------
    hist : pandas.DataFrame
        Eligible historical patients (harmonized; with ``time`` and ``cens``).
    trial : pandas.DataFrame
        Trial patients (with ``time``, ``cens`` and ``arm`` = 'control' / 'treated').
    features : list of str
        Baseline features that mean the same in both sources (see :func:`src.models.cross_source_features`).
    horizon_days : float
        Horizon of the event-free probability.

    Returns
    -------
    dict
        ``Xh``, ``Xt``, ``cat_idx``: encoded ``features`` for the historical model. ``yh``, ``yt``: 0/1 event-free
        labels, NaN when censored before the horizon (complete-case; legacy rows only). ``theta_h``: Kaplan-Meier
        pseudo-observations of the historical patients (one stratum; NaN without follow-up time or event status).
        ``theta_t``: pseudo-observations of the trial patients (one stratum per arm). ``Dt``: within-trial one-hot
        design of ``features`` plus the trial columns in ``NON_COMPARABLE`` (e.g. grade, which compares across the
        trial arms but not across sources). ``note``: how many historical patients have no pseudo-observation
        (empty if none).

    Raises
    ------
    ValueError
        If a trial patient has no ``time`` or ``cens`` (a pseudo-observation needs both).
    """
    n_missing = int(trial[["time", "cens"]].isna().any(axis=1).sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} trial patients have no follow-up time or event status; remove them first "
            "(pseudo-observations need both)"
        )
    (Xh, Xt), cat_idx = encode_features([hist, trial], features)
    within_trial = list(features) + [c for c in sorted(NON_COMPARABLE) if c in trial.columns and c not in features]
    (Xw,), cat_w = encode_features([trial], within_trial)
    theta_h, note = _historical_pseudo_obs(hist, horizon_days)
    return {
        "Xh": Xh,
        "Xt": Xt,
        "cat_idx": cat_idx,
        "Dt": _onehot_design(Xw, cat_w),
        "yh": event_free_labels(hist["time"], hist["cens"], horizon_days),
        "yt": event_free_labels(trial["time"], trial["cens"], horizon_days),
        "theta_h": theta_h,
        "theta_t": _pseudo_obs_by_arm(trial, horizon_days),
        "note": note,
    }


def _pseudo_obs_by_arm(trial, horizon_days):
    t, e = trial["time"].to_numpy(float), trial["cens"].to_numpy(float)
    theta = np.empty(len(trial))
    for arm in ("treated", "control"):
        m = (trial["arm"] == arm).to_numpy()
        theta[m] = pseudo_km(t[m], e[m], horizon_days)
    return theta


def _historical_pseudo_obs(hist, horizon_days):
    """Pseudo-observations of the historical patients (NaN without follow-up) and a note on the missing ones."""
    # historical patients only feed models, so one without follow-up is left out instead of stopping the analysis
    ok = hist[["time", "cens"]].notna().all(axis=1).to_numpy()
    theta = np.full(len(hist), np.nan)
    theta[ok] = pseudo_km(hist["time"].to_numpy(float)[ok], hist["cens"].to_numpy(float)[ok], horizon_days)
    note = (
        ""
        if ok.all()
        else (
            f"{int(np.sum(~ok))} historical patients have no follow-up time or event status: "
            "left out of the pseudo-observation regressor"
        )
    )
    return theta, note


def _onehot_design(X, cat_idx):
    """Linear design for entropy balancing and the logistic propensity.

    Each categorical column becomes indicators of its levels (first level dropped). Missing values get the
    median / most frequent level, as in the ``logreg`` pipeline; a column with no observed value is dropped.
    """
    cols = []
    for j in range(X.shape[1]):
        x = X[:, j].copy()
        miss = np.isnan(x)
        if miss.all():
            continue
        if j in cat_idx:
            levels, counts = np.unique(x[~miss], return_counts=True)
            x[miss] = levels[np.argmax(counts)]
            cols += [(x == lv).astype(float) for lv in levels[1:]]
        else:
            x[miss] = np.median(x[~miss])
            cols.append(x)
    return np.column_stack(cols) if cols else np.empty((len(X), 0))


def historical_predictions(arms, backends, seed=0):
    """Fit each backend on the historical patients and predict every trial patient (the prognostic score p_hist).

    The classifier on the complete-case labels is the only p_hist used by tabs 3-4 (one fit per backend). For the
    cheap backends (``logreg``, ``standin``) a regressor on the historical pseudo-observations is fitted as well,
    for comparison only.

    Parameters
    ----------
    arms : dict
        Output of :func:`prepare_arms`.
    backends : list of str
        Model backends (see :func:`src.models.fit_predict`).

    Returns
    -------
    dict
        backend -> P(event-free at the horizon) for every trial patient; for a cheap backend also
        ``"<backend>_pseudo"`` -> the regressor's prediction, clipped to [0, 1].
    """
    labelled, has_theta = ~np.isnan(arms["yh"]), ~np.isnan(arms["theta_h"])
    Xh, Xt, cat_idx = arms["Xh"], arms["Xt"], arms["cat_idx"]
    out = {}
    for b in backends:
        out[b] = fit_predict(b, Xh[labelled], arms["yh"][labelled], [Xt], seed, cat_idx, purpose="p_hist")[0]
        if b in ("logreg", "standin"):
            pred = fit_predict_reg(
                b, Xh[has_theta], arms["theta_h"][has_theta], [Xt], seed, cat_idx, purpose="p_hist_pseudo"
            )[0]
            out[f"{b}_pseudo"] = np.clip(pred, 0, 1)
    return out


# ----------------------------------------------------------------------------
# Tab 3: estimate
# ----------------------------------------------------------------------------
def run_estimate(
    hist, trial, features, horizon_days, n_real, backend, seed=0, n_boot=2000, do_joint=False, arms=None, p_trial=None
):
    """Estimate the treated patients' control outcome with a small real control group (tab 3; exploratory).

    Not gated: the pre-specified analysis is the gated run. ``n_real`` random trial controls are
    kept. Outcomes are Kaplan-Meier pseudo-observations at the horizon, one stratum per arm (the kept controls form
    their own). Each row estimates S0, the event-free probability the treated would have had on control:

    - Treated, observed (KM): S1 with a log-log CI.
    - Robust hybrid: the kept controls entropy-balanced to the treated on the within-trial design ``Dt``; outcome
      model theta ~ 1 + p_hist (OLS, cross-fitted on the kept controls); doubly robust S0 with a t CI,
      df = ESS - 1 (:func:`src.causal.robust_hybrid_att`). Historical data enter only through p_hist.
    - Small control arm, ATT-weighted: Hajek mean of the kept controls with the same weights; normal CI.
    - Historical + model: mean p_hist over the treated; no CI.
    - Legacy full correction / dynamic borrowing (/ joint model): the old methods on complete-case labels; their
      bootstrap resamples the real controls only.
    - Reference (demo only): Hajek mean of all trial controls with logistic ATT odds e / (1 - e).

    A row that cannot be estimated has NaN estimate and CI and says why in its note.

    Parameters
    ----------
    hist, trial, features, horizon_days
        See :func:`prepare_arms`.
    n_real : int
        Number of randomly kept real controls.
    backend : str
        Backend of the historical model (see :func:`src.models.fit_predict`).
    seed : int
        Seed of the kept-control draw, the folds, the bootstrap and the models.
    n_boot : int
        Bootstrap draws for the legacy full / dynamic rows.
    do_joint : bool
        Also fit the legacy joint model (one more model fit).
    arms, p_trial : optional
        Precomputed :func:`prepare_arms` output and p_hist of the trial patients (:func:`historical_predictions`).

    Returns
    -------
    dict
        ``table`` (method, estimate, lo, hi, note; fractions); ``effects`` (method, effect = S1 - estimate, lo, hi
        for every row but the treated; only the robust hybrid has a CI); ``hybrid``
        (:func:`src.causal.robust_hybrid_att` output plus ``reason``: None, or why it is not estimable);
        ``shift_text``, ``shift_warn`` (:func:`shift_check_text`, weighted with the entropy-balancing weights when
        balance was reached); ``n_real_labelled`` (kept controls with a complete-case label); ``kept``.
    """
    arms = arms or prepare_arms(hist, trial, features, horizon_days)
    p_all = p_trial if p_trial is not None else historical_predictions(arms, [backend], seed)[backend]
    A = (trial["arm"] == "treated").to_numpy(int)
    trt, ctrl = np.flatnonzero(A == 1), np.flatnonzero(A == 0)
    kept = np.sort(np.random.default_rng(seed).choice(ctrl, size=min(n_real, len(ctrl)), replace=False))
    t, e, D, Z = trial["time"].to_numpy(float), trial["cens"].to_numpy(float), arms["Dt"], p_all[:, None]
    theta_T, theta_K = arms["theta_t"][trt], pseudo_km(t[kept], e[kept], horizon_days)
    km_trt = km_at(t[trt], e[trt], horizon_days)
    label = BACKEND_LABELS.get(backend, backend)

    folds = causal.make_folds(np.zeros(len(kept)), k=min(5, len(kept)), seed=seed)
    hyb, small = _hybrid_and_small_arm(theta_T, theta_K, D[trt], D[kept], Z[trt], Z[kept], folds)
    reference_row = _reference_row(arms["theta_t"], D, A)
    rows = [
        (TREATED_ROW, km_trt.surv, *km_ci(km_trt.surv, km_trt.se), f"{len(trt)} treated; Kaplan-Meier, log-log CI"),
        *_hybrid_rows(hyb, small, len(kept), label),
        (
            HISTORICAL_ROW,
            float(p_all[trt].mean()),
            np.nan,
            np.nan,
            f"{label} trained on historical patients (complete-case labels); no CI",
        ),
    ]
    labelled = kept[~np.isnan(arms["yt"][kept])]
    legacy_rows, _ = _legacy_rows(arms, p_all, trt, labelled, backend, seed, n_boot, do_joint)
    table = pd.DataFrame(rows + legacy_rows + [reference_row], columns=["method", "estimate", "lo", "hi", "note"])
    text, warn = shift_check_text(p_all[kept], theta_K, weights=hyb["w"] if hyb["converged"] else None)
    return {
        "table": table,
        "effects": _effects_table(table, hyb),
        "hybrid": hyb,
        "shift_text": text,
        "shift_warn": warn,
        "n_real_labelled": len(labelled),
        "kept": kept,
    }


def _hybrid_and_small_arm(theta_T, theta_K, D_T, D_K, Z_T, Z_K, folds):
    """Robust hybrid and small-arm results, or NaN with the reason when they are not estimable.

    Not estimable when the kept controls' outcomes are all equal (e.g. no event before the horizon: S0 = 1 with
    SE 0) or when entropy balancing does not reach balance (positivity, as in gate G4 H). The entropy-balancing
    diagnostics (``w``, ``ess``, ``max_share``, ``imbalance``, ``converged``) are returned either way; ``reason``
    is None when estimable.
    """
    eb = causal.entropy_balance(D_K, D_T.mean(0))
    if np.ptp(theta_K) == 0:
        reason = (
            "no events among the kept controls before the horizon"
            if theta_K[0] == 1
            else "the kept controls' outcomes are all equal"
        )
    elif not eb["converged"]:
        reason = "entropy balancing did not reach balance (positivity)"
    else:
        hyb = causal.robust_hybrid_att(theta_T, theta_K, D_T, D_K, Z_T, Z_K, folds)
        return {**hyb, "reason": None}, causal.hajek_eb_att(theta_T, theta_K, hyb["w"])
    nan = dict.fromkeys(
        ("est", "se", "lo", "hi", "df", "q", "s0", "se_s1", "se_s0", "rr", "rr_lo", "rr_hi", "evalue", "evalue_ci"),
        np.nan,
    )
    nan["s1"] = float(np.mean(theta_T))
    diagnostics = {k: eb[k] for k in ("w", "ess", "max_share", "imbalance", "converged")}
    return {**nan, **diagnostics, "reason": f"not estimable: {reason}"}, nan


def _hybrid_rows(hyb, small, n_kept, label):
    if hyb["reason"]:
        hybrid_note = (
            f"{hyb['reason']} ({n_kept} kept controls; entropy balancing ESS {hyb['ess']:.1f}, max "
            f"imbalance {hyb['imbalance']:.2g}); historical data used only as a prognostic score ({label})"
        )
        small_note = hyb["reason"]
    else:
        hybrid_note = (
            f"historical data used only as a prognostic score ({label}); {n_kept} kept controls, "
            f"entropy balancing (exact balance, ESS {hyb['ess']:.1f}); OLS recalibration (NOT TabPFN); "
            f"t CI, df {hyb['df']:.1f}"
        )
        small_note = "kept controls with the hybrid's weights; no historical data"
    return [
        (HYBRID_ROW, hyb["s0"], *_s0_interval(hyb), hybrid_note),
        (SMALL_ARM_ROW, small["s0"], *_s0_interval(small), small_note),
    ]


def _reference_row(theta, D, A):
    """Hajek mean of all trial controls with logistic ATT odds (normally unknown; demo only)."""
    trt, ctrl = np.flatnonzero(A == 1), np.flatnonzero(A == 0)
    note = f"normally unknown; all {len(ctrl)} trial controls, logistic ATT odds (NOT TabPFN)"
    if np.ptp(theta[ctrl]) > 0:
        e_ps = causal.ps_logistic(D, A)
        ref = causal.hajek_eb_att(theta[trt], theta[ctrl], e_ps[ctrl] / (1 - e_ps[ctrl]))
    else:  # e.g. no control event before the horizon: S0 = 1 with SE 0
        ref = dict.fromkeys(("s0", "q", "se_s0"), np.nan)
        note += "; not estimable: all trial controls have the same outcome at the horizon"
    return (REFERENCE_ROW, ref["s0"], *_s0_interval(ref), note)


def _s0_interval(res):
    """CI of S0 with the row's own quantile (t for the robust hybrid, normal otherwise); NaN if not estimable."""
    return res["s0"] - res["q"] * res["se_s0"], res["s0"] + res["q"] * res["se_s0"]


def _legacy_rows(arms, p_all, trt, labelled, backend, seed, n_boot, do_joint):
    """Rows of the legacy methods and the dynamic-borrowing result (None without labelled kept controls)."""
    if not len(labelled):
        return [
            (
                LEGACY_ROWS[m],
                np.nan,
                np.nan,
                np.nan,
                "not estimable: no kept control with a known status at the horizon",
            )
            for m in ["full", "dynamic"] + (["joint"] if do_joint else [])
        ], None
    dyn, boot, p_joint = _legacy_fits(arms, p_all, trt, labelled, backend, seed, n_boot, do_joint)
    legacy = f"{BACKEND_LABELS.get(backend, backend)}; complete-case labels; no type-I error control"
    rows = [
        (LEGACY_ROWS["full"], dyn["full"], *interval(boot["full"]), f"{legacy}; bootstrap ignores model uncertainty"),
        (
            LEGACY_ROWS["dynamic"],
            dyn["dynamic"],
            *interval(boot["dynamic"]),
            f"{legacy}; bootstrap ignores model uncertainty; applies {dyn['w']:.0%} of the shift",
        ),
    ]
    if do_joint:
        rows.append(
            (LEGACY_ROWS["joint"], p_joint, np.nan, np.nan, f"one model on historical + real controls; {legacy}")
        )
    return rows, dyn


def _legacy_fits(arms, p_trial, trt, labelled, backend, seed, n_boot, do_joint):
    """Dynamic borrowing, its bootstrap and the joint model's mean (None unless ``do_joint``) on labelled controls."""
    p_real, y_real, p_target = p_trial[labelled], arms["yt"][labelled], p_trial[trt]
    dyn = dynamic_borrowing(p_real, y_real, p_target)
    boot = bootstrap_hybrid(p_real, y_real, p_target, n_boot, seed)
    p_joint = None
    if do_joint:
        known = ~np.isnan(arms["yh"])
        p_joint = float(
            joint_predict(
                backend,
                arms["Xh"][known],
                arms["yh"][known],
                arms["Xt"][labelled],
                y_real,
                arms["Xt"][trt],
                seed,
                arms["cat_idx"],
            ).mean()
        )
    return dyn, boot, p_joint


def _effects_table(table, hyb):
    """S1 - S0 for every row but the treated; only the robust hybrid has a CI."""
    effects = pd.DataFrame(
        {"method": table["method"][1:], "effect": hyb["s1"] - table["estimate"][1:], "lo": np.nan, "hi": np.nan}
    ).reset_index(drop=True)
    effects.loc[effects["method"] == HYBRID_ROW, ["lo", "hi"]] = hyb["lo"], hyb["hi"]
    return effects


def shift_check_text(p_real, y_real, w_dynamic=None, weights=None):
    """Plain-language check of the historical model on the real controls, and whether to warn.

    Compares the real controls' observed outcome with the model's prediction for the same patients. In the
    estimate tab the weights are the entropy-balancing weights, so both means describe controls that look like the
    treated. The noise is the SE of the weighted mean difference, sqrt(sum w^2 (r - shift)^2) with r = y - p (as in
    :func:`src.causal.negative_control_beta`).

    Parameters
    ----------
    p_real : array-like
        Model predictions of event-free at the horizon for the real controls.
    y_real : array-like
        Their observed outcome (pseudo-observations, or 0/1 labels).
    w_dynamic : float, optional
        Legacy only: share of the shift that dynamic borrowing applied; adds that sentence when given.
    weights : array-like, optional
        Weights of the real controls; None gives equal weights.

    Returns
    -------
    text : str
    warn : bool
        True if |shift| > 2 * noise.
    """
    p, y = np.asarray(p_real, float), np.asarray(y_real, float)
    w = np.full(len(y), 1 / len(y)) if weights is None else np.asarray(weights, float) / np.sum(weights)
    obs, pred = float(w @ y), float(w @ p)
    shift = obs - pred
    se = float(np.sqrt(np.sum(w**2 * (y - p - shift) ** 2))) if len(y) > 1 else np.inf
    who = "Real controls" if weights is None else "Real controls weighted to the treated"
    text = (
        f"{who}: {obs:.0%} event-free, historical model predicted {pred:.0%}, shift {shift * 100:+.1f} points "
        f"(noise ± {se * 100:.1f})."
    )
    if w_dynamic is not None:
        text += f" Legacy dynamic borrowing applied {w_dynamic:.0%} of the shift."
    return text, bool(abs(shift) > 2 * se)


# ----------------------------------------------------------------------------
# Tab 4: validation
# ----------------------------------------------------------------------------
def run_validation(
    hist,
    trial,
    features,
    horizon_days,
    n_real,
    backend,
    reps=200,
    seed=0,
    n_boot=300,
    do_joint=False,
    arms=None,
    p_trial=None,
    progress=None,
):
    """Hide most trial controls and check how well each method recovers the treated's control outcome (tab 4).

    Demo only and report only. Each repetition splits the trial controls into ``n_real`` kept and the
    rest hidden. Every method estimates S0 from the treated and the kept controls (see :func:`run_estimate`). The
    truth is the AIPW-ATT S0 of the treated with the hidden controls: a logistic propensity on the one-hot design,
    fitted once on all trial patients (outcome-free), and an OLS outcome model cross-fitted on the hidden controls.
    Pseudo-observations are computed separately for treated, kept and hidden patients. The historical predictions
    are fitted once and held fixed, so every method that uses them shows one realization of their error.

    Parameters
    ----------
    hist, trial, features, horizon_days
        See :func:`prepare_arms`.
    n_real : int
        Kept controls per repetition (at most the number of trial controls minus 10).
    backend : str
        Backend of the historical model.
    reps : int
        Number of kept / hidden splits.
    seed : int
        Seed of the splits, folds, bootstrap and models.
    n_boot : int
        Bootstrap draws per repetition for the SE of the legacy full / dynamic rows.
    do_joint : bool
        Also run the legacy joint model (one model fit per repetition).
    arms, p_trial : optional
        Precomputed :func:`prepare_arms` output and p_hist of the trial patients.
    progress : callable, optional
        Called with the fraction of repetitions done.

    Returns
    -------
    dict
        ``runs`` (rep, method, est, se, truth, se_truth, diff = est - truth, covered, note; fractions; covered is
        |diff| <= 1.96 sqrt(se^2 + se_truth^2), NaN for methods without an SE), ``summary``
        (:func:`summarize_validation`), ``n_real`` and ``main`` (the robust-hybrid row).
    """
    arms = arms or prepare_arms(hist, trial, features, horizon_days)
    if p_trial is None:
        p_trial = historical_predictions(arms, [backend], seed)[backend]
    A = (trial["arm"] == "treated").to_numpy(int)
    trt, ctrl = np.flatnonzero(A == 1), np.flatnonzero(A == 0)
    n_real = min(n_real, len(ctrl) - 10)
    t, e = trial["time"].to_numpy(float), trial["cens"].to_numpy(float)
    e_ps = causal.ps_logistic(arms["Dt"], A)  # outcome-free, so the same propensity serves every split's truth
    notes = _validation_notes(BACKEND_LABELS.get(backend, backend))
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        perm = rng.permutation(ctrl)
        kept, hidden = np.sort(perm[:n_real]), np.sort(perm[n_real:])
        theta_K = pseudo_km(t[kept], e[kept], horizon_days)
        theta_H = pseudo_km(t[hidden], e[hidden], horizon_days)
        truth, se_truth = _hidden_control_truth(arms["theta_t"][trt], theta_H, arms["Dt"], trt, hidden, e_ps, seed)
        estimates = _split_estimates(
            arms, p_trial, trt, kept, theta_K, backend, n_boot, do_joint, fold_seed=seed, rep_seed=seed + rep
        )
        rows += [_score_estimate(rep, method, est, se, truth, se_truth, notes[method]) for method, est, se in estimates]
        if progress:
            progress((rep + 1) / reps)
    runs = pd.DataFrame(rows)
    return {"runs": runs, "summary": summarize_validation(runs), "n_real": n_real, "main": HYBRID_ROW}


def _validation_notes(label):
    """Note of each method; all of them say that the truth uses logistic / linear models."""
    fixed = f"historical predictions ({label}) held fixed: one realization of their error"
    legacy = f"complete-case labels; {fixed}; bootstrap SE ignores model uncertainty; no type-I error control"
    notes = {
        HYBRID_ROW: f"{fixed}; entropy balancing + OLS recalibration (NOT TabPFN)",
        SMALL_ARM_ROW: "kept controls only; entropy balancing (NOT TabPFN)",
        HISTORICAL_ROW: f"{fixed}; no SE",
        LEGACY_ROWS["full"]: legacy,
        LEGACY_ROWS["dynamic"]: legacy,
        LEGACY_ROWS["joint"]: f"{label} refitted per repetition; complete-case labels; no SE",
    }
    return {k: f"{v}; truth: {CHEAP_LABEL}" for k, v in notes.items()}


def _hidden_control_truth(theta_T, theta_H, D, trt, hidden, e_ps, seed):
    """AIPW-ATT S0 of the treated with the hidden controls, and its SE."""
    folds = causal.make_folds(np.zeros(len(hidden)), seed=seed)
    mu0_H, mu0_T = causal.recalibrate_mu0(theta_H, D[hidden], D[trt], folds)
    mu0 = np.r_[mu0_T, mu0_H]
    res = causal.aipw(
        np.r_[theta_T, theta_H],
        np.r_[np.ones(len(trt)), np.zeros(len(hidden))],
        e_ps[np.r_[trt, hidden]],
        mu0,
        mu0,
        "ATT",
    )
    return res["s0"], float(np.sqrt(np.sum(res["if_s0"] ** 2)))


def _split_estimates(arms, p_trial, trt, kept, theta_K, backend, n_boot, do_joint, fold_seed, rep_seed):
    """(method, S0 estimate, SE) of every method for one kept / hidden split."""
    theta_T, D, Z = arms["theta_t"][trt], arms["Dt"], p_trial[:, None]
    folds = causal.make_folds(np.zeros(len(kept)), k=min(5, len(kept)), seed=fold_seed)
    hyb = causal.robust_hybrid_att(theta_T, theta_K, D[trt], D[kept], Z[trt], Z[kept], folds)
    small = causal.hajek_eb_att(theta_T, theta_K, hyb["w"])
    labelled = kept[~np.isnan(arms["yt"][kept])]
    dyn, boot, p_joint = _legacy_fits(arms, p_trial, trt, labelled, backend, rep_seed, n_boot, do_joint)
    estimates = [
        (HYBRID_ROW, hyb["s0"], hyb["se_s0"]),
        (SMALL_ARM_ROW, small["s0"], small["se_s0"]),
        (HISTORICAL_ROW, float(p_trial[trt].mean()), np.nan),
        (LEGACY_ROWS["full"], dyn["full"], float(np.std(boot["full"], ddof=1))),
        (LEGACY_ROWS["dynamic"], dyn["dynamic"], float(np.std(boot["dynamic"], ddof=1))),
    ]
    if do_joint:
        estimates.append((LEGACY_ROWS["joint"], p_joint, np.nan))
    return estimates


def _score_estimate(rep, method, est, se, truth, se_truth, note):
    diff = est - truth
    covered = float(abs(diff) <= 1.96 * np.sqrt(se**2 + se_truth**2)) if np.isfinite(se) else np.nan
    return {
        "rep": rep,
        "method": method,
        "est": est,
        "se": se,
        "truth": truth,
        "se_truth": se_truth,
        "diff": diff,
        "covered": covered,
        "note": note,
    }


def summarize_validation(runs):
    """Per-method summary of :func:`run_validation` in percentage points (report only).

    Parameters
    ----------
    runs : pandas.DataFrame
        ``runs`` of :func:`run_validation`.

    Returns
    -------
    pandas.DataFrame
        method, reps, bias (mean of est - truth) and bias_mcse (SD / sqrt(R)), SD, RMSE, coverage (share of
        repetitions with |est - truth| <= 1.96 sqrt(se^2 + se_truth^2); NaN for methods without an SE),
        coverage_mcse (sqrt(p (1 - p) / R)), note.
    """
    out = []
    for method, g in runs.groupby("method", sort=False):
        d = g["diff"].to_numpy(float) * 100
        R = len(d)
        sd = float(d.std(ddof=1)) if R > 1 else np.nan
        cov = g["covered"].dropna().to_numpy(float)
        p = float(cov.mean()) if len(cov) else np.nan
        out.append(
            {
                "method": method,
                "reps": R,
                "bias": float(d.mean()),
                "bias_mcse": sd / np.sqrt(R),
                "SD": sd,
                "RMSE": float(np.sqrt(np.mean(d**2))),
                "coverage": p,
                "coverage_mcse": float(np.sqrt(p * (1 - p) / len(cov))) if len(cov) else np.nan,
                "note": g["note"].iloc[0],
            }
        )
    return pd.DataFrame(out)


def validation_headline(summary, n_real):
    """One sentence on the robust hybrid: bias and coverage with their Monte Carlo SEs; empty without that row."""
    row = summary[summary["method"] == HYBRID_ROW]
    if row.empty:
        return ""
    s = row.iloc[0]
    return (
        f"The robust hybrid with {n_real} kept controls: bias {s['bias']:+.1f} points (MCSE {s['bias_mcse']:.1f}), "
        f"coverage {s['coverage']:.0%} (MCSE {s['coverage_mcse']:.1%}) over {int(s['reps'])} splits; truth = ATT "
        "counterfactual of the hidden controls (demo only); methods using the historical model show one "
        f"realization of its error; truth: {CHEAP_LABEL}."
    )


# ----------------------------------------------------------------------------
# Legacy methods (complete-case labels; no type-I error control)
# ----------------------------------------------------------------------------
def logit_shift(p_real, y_real):
    """Maximum-likelihood logit intercept shift that calibrates the model to the real controls.

    Parameters
    ----------
    p_real : array-like
        Model probabilities of event-free at the horizon for the real controls.
    y_real : array-like
        Their observed 0/1 labels.

    Returns
    -------
    float
        The shift ``a``; +-6 if all labels are equal.
    """
    return float(_fit_shifts(_clipped_logit(p_real), np.asarray(y_real, float), np.ones((1, len(p_real))))[0])


def dynamic_borrowing(p_real, y_real, p_target):
    """Full correction and empirical-Bayes dynamic borrowing.

    Parameters
    ----------
    p_real, y_real : array-like
        Model probabilities and observed labels of the real controls.
    p_target : array-like
        Model probabilities of the target population.

    Returns
    -------
    dict
        ``a`` (:func:`logit_shift`), ``full`` (mean of expit(logit(p) + a) over the target), ``raw`` (mean
        p_target), ``shift`` (full - raw), ``se`` (noise of the real data), ``tau2``, ``w`` (share of the shift
        that is applied) and ``dynamic`` (the final estimate).
    """
    p_real, y_real, p_target = (np.asarray(v, float) for v in (p_real, y_real, p_target))
    a = logit_shift(p_real, y_real)
    full = float(expit(_clipped_logit(p_target) + a).mean())
    raw = float(p_target.mean())
    shift = full - raw
    se = float(np.std(y_real - p_real, ddof=1) / np.sqrt(len(y_real))) if len(y_real) > 1 else np.inf
    tau2 = max(shift**2 - se**2, 0.0)
    w = tau2 / (tau2 + se**2) if (tau2 + se**2) > 0 else 0.0
    return {
        "full": full,
        "raw": raw,
        "shift": shift,
        "se": se,
        "tau2": tau2,
        "w": float(w),
        "dynamic": raw + w * shift,
        "a": a,
    }


def bootstrap_hybrid(p_real, y_real, p_target, n_boot=2000, seed=0):
    """Bootstrap the real controls and recompute full correction and dynamic borrowing.

    Returns
    -------
    dict
        ``full`` and ``dynamic``: arrays of ``n_boot`` estimates (use percentiles for intervals).
    """
    p_real, y_real, p_target = (np.asarray(v, float) for v in (p_real, y_real, p_target))
    n = len(p_real)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)
    a = _fit_shifts(_clipped_logit(p_real), y_real, W)
    full = expit(_clipped_logit(p_target)[None, :] + a[:, None]).mean(axis=1)
    raw = p_target.mean()
    shift = full - raw
    r = y_real - p_real
    m = (W * r).sum(axis=1) / n
    var = (W * (r[None, :] - m[:, None]) ** 2).sum(axis=1) / max(n - 1, 1)
    se2 = var / n
    tau2 = np.maximum(shift**2 - se2, 0)
    w = np.divide(tau2, tau2 + se2, out=np.zeros_like(tau2), where=(tau2 + se2) > 0)
    return {"full": full, "dynamic": raw + w * shift}


def interval(draws, level=0.95):
    """Percentile interval of bootstrap draws."""
    q = (1 - level) / 2
    return float(np.quantile(draws, q)), float(np.quantile(draws, 1 - q))


def joint_predict(backend, X_hist, y_hist, X_real, y_real, X_target, seed=0, cat_idx=None):
    """Joint model: one classifier on historical + real controls with a ``source`` flag.

    Historical rows have source 0, real controls source 1; the target is predicted with source 1. Patients with
    unknown labels must already be removed.

    Returns
    -------
    numpy.ndarray
        P(event-free) for each target patient.
    """

    def with_source(X, source):
        return np.column_stack([X, np.full(len(X), source)])

    X_train = np.vstack([with_source(X_hist, 0), with_source(X_real, 1)])
    cat_with_source = list(cat_idx or []) + [X_hist.shape[1]]
    return fit_predict(backend, X_train, np.r_[y_hist, y_real], [with_source(X_target, 1)], seed, cat_with_source)[0]


def _fit_shifts(logit_real, y_real, W):
    """Logit intercept shift per weight row of ``W``: root of the weighted ML score equation, by bisection."""
    # The score sum_i w_i (y_i - expit(l_i + a)) decreases in a. With all labels equal it has no finite root,
    # and bisection ends at the bracket (+-6).
    lo = np.full(W.shape[0], -_MAX_LOGIT_SHIFT)
    hi = np.full(W.shape[0], _MAX_LOGIT_SHIFT)
    for _ in range(45):
        mid = (lo + hi) / 2
        score = (W * (y_real[None, :] - expit(logit_real[None, :] + mid[:, None]))).sum(axis=1)
        lo = np.where(score > 0, mid, lo)
        hi = np.where(score > 0, hi, mid)
    return (lo + hi) / 2


def _clipped_logit(p):
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))
