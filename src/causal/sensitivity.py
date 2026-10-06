import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.causal.data import TRIAL_COLUMNS, _as_float, _check_no_outcome_columns
from src.causal.weights import ps_logistic


def evalue(rr):
    """E-value of a risk ratio (VanderWeele & Ding 2017): rr* + sqrt(rr* (rr* - 1)), rr* = max(rr, 1/rr)."""
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return np.nan
    r = max(rr, 1 / rr)
    return float(r + np.sqrt(r * (r - 1)))


def evalue_ci(lo, hi):
    """E-value of a risk-ratio CI (VanderWeele & Ding 2017).

    Returns
    -------
    float
        1 if the CI contains 1, else the E-value of the limit closest to 1; NaN if a limit is not finite.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    if lo <= 1 <= hi:
        return 1.0
    return evalue(lo if lo > 1 else hi)


def bounding_factor(a, b):
    """Ding & VanderWeele (2016) bounding factor a b / (a + b - 1)."""
    return a * b / (a + b - 1)


def needed_rr_uy(rr_star, rr_au):
    """Outcome association needed to explain ``rr_star`` away, given the exposure association ``rr_au``.

    rr_uy = rr* (rr_au - 1) / (rr_au - rr*), or inf if rr_au <= rr*.
    """
    return rr_star * (rr_au - 1) / (rr_au - rr_star) if rr_au > rr_star else np.inf


def tipping_point(est, se, z=1.96):
    """Additive bias that moves the estimate or its CI limit to 0: (est, est - z se)."""
    return est, est - z * se


def min_detectable_margin(sigma, ess, alpha=0.05):
    """Smallest equivalence margin a TOST could ever pass: z_{1-alpha} sigma / sqrt(ess) (gate G8)."""
    return float(stats.norm.ppf(1 - alpha) * sigma / np.sqrt(ess))


def tost(est, se, margin, alpha=0.05):
    """Two one-sided tests of equivalence within +-margin (negative control; only run if gate G8 is feasible).

    Parameters
    ----------
    est, se : float
        Estimate and its standard error.
    margin : float
        Equivalence margin (same scale as ``est``).
    alpha : float
        Level of each one-sided test.

    Returns
    -------
    tuple
        (lo, hi, passed): the (1 - 2 alpha) normal CI and passed = -margin < lo and hi < margin.
    """
    z = stats.norm.ppf(1 - alpha)
    lo, hi = est - z * se, est + z * se
    return lo, hi, bool(-margin < lo and hi < margin)


def benchmark_variables(trial_cov):
    """Pre-specified benchmark covariates and blocks for the E-value rule.

    Parameters
    ----------
    trial_cov : pandas.DataFrame
        Trial covariates of the scope sample (rows aligned with the design).

    Returns
    -------
    list of tuple
        (name, labels (array of str), drop_cols (indices into the 8 :func:`design_trial` columns)).
    """
    _check_no_outcome_columns(trial_cov)
    c, df = TRIAL_COLUMNS, trial_cov
    age, nodes, size = _as_float(df[c["age"]]), _as_float(df[c["pnodes"]]), _as_float(df[c["tsize"]])
    meno = np.where((df[c["menostat"]] == "Post").to_numpy(), "post", "pre")
    er = np.where(_as_float(df[c["estrec"]]) >= 20, "ER+", "ER-")
    pgr = np.where(_as_float(df[c["progrec"]]) >= 20, "PgR+", "PgR-")
    age_band = np.where(age < 45, "<45", np.where(age < 60, "45-59", ">=60"))
    node_burden = np.where(nodes >= 4, "N4+", "N1-3")
    size_burden = np.where(size > 20, "T>20", "T<=20")

    def block(x, y):
        return np.array([f"{a}|{b}" for a, b in zip(x, y)], dtype=object)

    return [
        ("menostat", meno, [1]),
        ("age band", np.where(age < 50, "<50", np.where(age < 60, "50-59", ">=60")), [0]),
        ("nodes", np.where(nodes <= 3, "1-3", np.where(nodes <= 9, "4-9", ">=10")), [5]),
        ("tsize", np.where(size <= 20, "<=20", np.where(size <= 50, "21-50", ">50")), [2]),
        ("grade", df[c["tgrade"]].astype(str).to_numpy(object), [3, 4]),
        ("ER>=20", er, [7]),
        ("PgR>=20", pgr, [6]),
        ("age x menostat", block(meno, age_band), [0, 1]),
        ("ER x PgR", block(er, pgr), [6, 7]),
        ("nodes x tsize", block(node_burden, size_burden), [2, 5]),
    ]


def benchmark_bounding_factors(theta, A, D, benchmarks, min_cell=20):
    """Bounding factors of measured covariates and blocks, computed conditionally (gate G11).

    For each benchmark, only levels with >= ``min_cell`` treated and controls are kept.
    rr_au = max_k max(p1k/p0k, p0k/p1k), with p1k = P(level k | treated) and p0k the ATT-odds-weighted
    P(level k | control), the odds coming from a logistic propensity on D without the benchmark's columns.
    rr_uy = max_k r_k / min_k r_k, with r_k the mean predicted event risk (1 - theta) among controls when the
    level is set to k (OLS on D without the columns plus level dummies; r_k clipped to [0.01, 1]).
    bf = bounding_factor(rr_au, rr_uy).

    Parameters
    ----------
    theta : array-like
        Pseudo-observations (event-free) of the scope sample.
    A : array-like
        0/1 treatment.
    D : numpy.ndarray
        :func:`design_trial` of the scope sample (8 columns).
    benchmarks : list of tuple
        From :func:`benchmark_variables`.
    min_cell : int

    Returns
    -------
    pandas.DataFrame
        name, levels_kept, levels_dropped (with treated/control counts), rr_au, rr_uy, bf (NaN if fewer than two
        levels are kept).
    """
    theta, A, D = np.asarray(theta, float), np.asarray(A).astype(int), np.asarray(D, float)
    risk, ctrl = 1 - theta, A == 0
    rows = []
    for name, labels, drop in benchmarks:
        labels = np.asarray(labels).astype(str)
        counts = {k: ((labels[A == 1] == k).sum(), (labels[ctrl] == k).sum()) for k in np.unique(labels)}
        levels = [k for k, (n1, n0) in counts.items() if n1 >= min_cell and n0 >= min_cell]
        row = {
            "name": name,
            "levels_kept": ", ".join(levels),
            "levels_dropped": ", ".join(f"{k} ({n1}/{n0})" for k, (n1, n0) in counts.items() if k not in levels),
            "rr_au": np.nan,
            "rr_uy": np.nan,
            "bf": np.nan,
        }
        if len(levels) >= 2:
            X_rest = np.delete(D, drop, axis=1)
            e = ps_logistic(X_rest, A)
            odds = (e / (1 - e))[ctrl]
            p1 = np.array([np.mean(labels[A == 1] == k) for k in levels])
            p0 = np.array([np.average(labels[ctrl] == k, weights=odds) for k in levels])
            rr_au = float(max(np.max(p1 / p0), np.max(p0 / p1)))
            sub = ctrl & np.isin(labels, levels)
            dummies = np.column_stack([(labels[sub] == k).astype(float) for k in levels[1:]])
            model = LinearRegression().fit(np.column_stack([X_rest[sub], dummies]), risk[sub])

            def mean_risk_at(level):
                dummies_at_level = np.tile([float(level == k) for k in levels[1:]], (sub.sum(), 1))
                mean_risk = model.predict(np.column_stack([X_rest[sub], dummies_at_level])).mean()
                return float(np.clip(mean_risk, 0.01, 1.0))

            r = [mean_risk_at(k) for k in levels]
            rr_uy = max(r) / min(r)
            row.update(rr_au=rr_au, rr_uy=rr_uy, bf=bounding_factor(rr_au, rr_uy))
        rows.append(row)
    return pd.DataFrame(rows)


def robustness(rr, rr_lo, rr_hi, bf_ref, sign_ok):
    """Benchmark robustness rule of gate G11.

    robust iff the RR CI excludes 1, rr*_lim > bf_ref and ``sign_ok``, with rr*_lim = 1/rr_hi if rr < 1 else
    rr_lo (rr*_lim = 1 when the RR CI includes 1, which is never > bf_ref >= 1).

    ``applicable`` describes the RR CI only. G11 applies when the primary risk-difference CI excludes 0, and near
    the boundary the RD CI can exclude 0 while the RR CI includes 1. The caller must therefore decide
    applicability from the RD CI and use ``robust`` for PASS / FAIL, never map ``applicable = False`` to
    NOT_APPLICABLE.

    Parameters
    ----------
    rr, rr_lo, rr_hi : float
        Event risk ratio of the primary estimate and its CI.
    bf_ref : float
        Reference bounding factor (largest benchmark BF, floored by the named bounding factor).
    sign_ok : bool
        Sign stability over the sensitivity rows of the sign rule.

    Returns
    -------
    dict
        applicable (RR CI excludes 1), rr_star_lim, bf_ref, robust.
    """
    applicable = not (rr_lo <= 1 <= rr_hi)
    lim = (1 / rr_hi if rr < 1 else rr_lo) if applicable else 1.0
    return {
        "applicable": bool(applicable),
        "rr_star_lim": float(lim),
        "bf_ref": float(bf_ref),
        "robust": bool(applicable and lim > bf_ref and sign_ok),
    }
