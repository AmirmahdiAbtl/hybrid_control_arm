import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def make_folds(labels, k=5, seed=0):
    """Stratified fold id 0..k-1 per unit (``StratifiedKFold(k, shuffle=True, random_state=seed)``)."""
    y = np.asarray(labels).astype(str)
    folds = np.zeros(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        folds[te] = i
    return folds


def ps_logistic(D, a):
    """In-sample P(a = 1 | D) from an unpenalized logistic regression (MLE on standardized columns).

    Parameters
    ----------
    D : array-like
        Design matrix (n x p), no intercept column.
    a : array-like
        0/1 labels.

    Raises
    ------
    ValueError
        If ``D`` has a missing or non-finite value (no imputation is pre-specified).
    """
    return _logistic_fit_predict(D, a, [D])[0]


def _logistic_fit_predict(D_train, a_train, D_tests):
    """Unpenalized logistic MLE on standardized columns (constant columns dropped); P(a = 1) per test matrix.

    Newton-Cholesky with a tight tolerance gives the exact MLE, so the score equations hold to ~1e-9, which the
    exact ATO balance property needs. Missing values raise instead of silently dropping the column (sd = NaN).
    """
    D_train, a_train = np.asarray(D_train, float), np.asarray(a_train).astype(int)
    if not np.isfinite(D_train).all() or not all(np.isfinite(np.asarray(X, float)).all() for X in D_tests):
        raise ValueError(
            "logistic model: covariates contain missing or non-finite values (no imputation is pre-specified; gate G3)"
        )
    mu, sd = D_train.mean(0), D_train.std(0)
    keep = sd > 0
    if not keep.any():
        return [np.full(len(X), a_train.mean()) for X in D_tests]
    model = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=10000)
    model.fit((D_train[:, keep] - mu[keep]) / sd[keep], a_train)
    return [model.predict_proba((np.asarray(X, float)[:, keep] - mu[keep]) / sd[keep])[:, 1] for X in D_tests]


def balancing_weights(e, A, estimand):
    """Unnormalized balancing weight per unit (both arms).

    ATT: A + (1-A)e/(1-e); ATE: A/e + (1-A)/(1-e); ATC: A(1-e)/e + (1-A); ATO: A(1-e) + (1-A)e.

    Parameters
    ----------
    e : array-like
        Propensity score P(A = 1 | x), strictly inside (0, 1).
    A : array-like
        0/1 treatment.
    estimand : {"ATT", "ATE", "ATC", "ATO"}
    """
    e, A = np.asarray(e, float), np.asarray(A, float)
    if estimand == "ATT":
        return A + (1 - A) * e / (1 - e)
    if estimand == "ATE":
        return A / e + (1 - A) / (1 - e)
    if estimand == "ATC":
        return A * (1 - e) / e + (1 - A)
    if estimand == "ATO":
        return A * (1 - e) + (1 - A) * e
    raise ValueError(f"unknown estimand '{estimand}'")


def entropy_balance(X, target, tol=1e-6):
    """Entropy-balancing weights (Hainmueller 2012) so that sum_j w_j X_j = target.

    Solves the dual, min logsumexp(Z @ lam) with Z = (X - target) / sd(X). Constant columns are left out of Z;
    their gap to the target counts as imbalance. A target outside the convex hull of X gives
    ``converged = False``: a positivity failure, not a bug.

    Parameters
    ----------
    X : array-like
        Units to weight (n x p).
    target : array-like
        Target means (p).
    tol : float
        Convergence threshold on the largest standardized imbalance.

    Returns
    -------
    dict
        ``w`` (sums to 1), ``imbalance`` (max |w @ Z|), ``converged``, ``ess`` (1 / sum w^2), ``max_share`` and
        ``lam``.

    Raises
    ------
    ValueError
        If ``X`` or ``target`` has a missing or non-finite value.
    """
    X, target = np.asarray(X, float), np.asarray(target, float)
    if not (np.isfinite(X).all() and np.isfinite(target).all()):
        raise ValueError("entropy_balance: X or target contains missing or non-finite values")
    sd = X.std(0)
    varies = sd > 0
    Z = (X[:, varies] - target[varies]) / sd[varies]
    constant_gap = np.abs(X[0, ~varies] - target[~varies]).max(initial=0.0) if len(X) else np.inf
    lam = np.zeros(Z.shape[1])
    if Z.shape[1]:
        last_finite = {"lam": lam}

        def keep_finite(x):
            # Outside the hull the dual is unbounded and BFGS can overflow to NaN: keep the last finite iterate.
            if np.isfinite(logsumexp(Z @ x)):
                last_finite["lam"] = x.copy()

        with np.errstate(over="ignore", invalid="ignore"):
            res = minimize(
                lambda x: logsumexp(Z @ x),
                lam,
                jac=lambda x: softmax(Z @ x) @ Z,
                method="BFGS",
                callback=keep_finite,
                options={"gtol": 1e-12, "maxiter": 5000},
            )
            lam = res.x if np.isfinite(logsumexp(Z @ res.x)) else last_finite["lam"]
    w = softmax(Z @ lam) if Z.shape[1] else np.full(len(X), 1 / len(X))
    imbalance = max(float(np.abs(w @ Z).max(initial=0.0)), float(constant_gap))
    return {
        "w": w,
        "imbalance": imbalance,
        "converged": bool(imbalance < tol),
        "ess": float(1 / np.sum(w**2)),
        "max_share": float(w.max()),
        "lam": lam,
    }


def kish_ess(w):
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w**2)) if len(w) else 0.0


def max_weight_share(w):
    """Largest single weight as a share of the total."""
    w = np.asarray(w, float)
    return float(w.max() / w.sum()) if len(w) else np.nan


def smd_table(X, A, w, names, continuous=None):
    """Standardized mean differences before and after weighting.

    SMD = (m1 - m0) / sqrt((s1^2 + s0^2) / 2) with unweighted SDs (ddof = 1); the weighted means use the arm
    weights of the estimator (e.g. ATT: 1 for treated, w for controls). A column that is constant within each
    arm gets SMD 0 if both arms hold the same value and +-inf otherwise. Missing values give NaN SMDs, so take
    the maximum without skipping NaN (``np.max``, or pandas ``max(skipna=False)``) to make such a column fail.

    Parameters
    ----------
    X : array-like
        Covariates (n x p).
    A : array-like
        0/1 arm.
    w : array-like
        Weight per unit (both arms).
    names : list of str
    continuous : list of bool, optional
        Columns that get a variance ratio; default: columns with more than 2 distinct values.

    Returns
    -------
    pandas.DataFrame
        name, smd_before, smd_after, vr_after (weighted var1 / weighted var0; NaN for binary columns).
    """
    X, A, w = np.asarray(X, float), np.asarray(A).astype(bool), np.asarray(w, float)
    rows = []
    for j, name in enumerate(names):
        x1, x0, w1, w0 = X[A, j], X[~A, j], w[A], w[~A]
        sd = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
        m1, m0 = np.average(x1, weights=w1), np.average(x0, weights=w0)
        if np.ptp(x1) == 0 and np.ptp(x0) == 0:
            # Compare the constants directly: weighted means of a constant can be off by rounding.
            before = after = 0.0 if x1[0] == x0[0] else float(np.copysign(np.inf, x1[0] - x0[0]))
        else:
            before, after = (x1.mean() - x0.mean()) / sd, (m1 - m0) / sd
        is_continuous = continuous[j] if continuous is not None else len(np.unique(X[:, j])) > 2
        v0 = np.average((x0 - m0) ** 2, weights=w0)
        vr = np.average((x1 - m1) ** 2, weights=w1) / v0 if is_continuous and v0 > 0 else np.nan
        rows.append({"name": name, "smd_before": before, "smd_after": after, "vr_after": vr})
    return pd.DataFrame(rows)


def overlap_summary(e, A, estimand, bounds=(0.05, 0.95)):
    """Positivity diagnostics for one estimand (gate G4, scope R).

    "Outside" counts only the estimand's target units: ATT treated with e > hi; ATC controls with e < lo;
    ATE all units outside the bounds; ATO the same as ATE but report only (``gated`` False).

    Returns
    -------
    dict
        estimand, share_outside, gated, n / ess / max_share per arm (``_treated``, ``_control``).
    """
    e, A = np.asarray(e, float), np.asarray(A).astype(bool)
    lo, hi = bounds
    if estimand == "ATT":
        share = np.mean(e[A] > hi)
    elif estimand == "ATC":
        share = np.mean(e[~A] < lo)
    else:
        share = np.mean((e < lo) | (e > hi))
    w = balancing_weights(e, A, estimand)
    return {
        "estimand": estimand,
        "share_outside": float(share),
        "gated": estimand != "ATO",
        "n_treated": int(A.sum()),
        "n_control": int((~A).sum()),
        "ess_treated": kish_ess(w[A]),
        "ess_control": kish_ess(w[~A]),
        "max_share_treated": max_weight_share(w[A]),
        "max_share_control": max_weight_share(w[~A]),
    }


def source_support(D_T, D_B, strata_T, strata_B, seed=0, k=5, cut=20.0):
    """Support of the treated by Rotterdam via the density ratio r = f_T / f_B (gate G6).

    pi = out-of-fold P(treated | D_s) from an unpenalized logistic model of treated (1) vs Rotterdam (0) at
    natural class sizes, clipped to [1e-6, 1 - 1e-6]; r = pi / (1 - pi) * n_B / n_T.

    Parameters
    ----------
    D_T, D_B : numpy.ndarray
        :func:`design_source` of the treated and of Rotterdam.
    strata_T, strata_B : array-like of str
        Stratum labels (e.g. "pre" / "post").
    seed, k : int
    cut : float
        A treated patient is unsupported if r > cut.

    Returns
    -------
    dict
        ``auc``, ``table`` (rows "all" and each stratum: stratum, n_T, n_B, share_unsupported, ess_B,
        max_share), ``r_T``, ``r_B``.
    """
    D_T, D_B = np.asarray(D_T, float), np.asarray(D_B, float)
    n_T, n_B = len(D_T), len(D_B)
    X, y = np.vstack([D_T, D_B]), np.r_[np.ones(n_T), np.zeros(n_B)].astype(int)
    pi = np.empty(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        pi[te] = _logistic_fit_predict(X[tr], y[tr], [X[te]])[0]
    pi = np.clip(pi, 1e-6, 1 - 1e-6)
    r = pi / (1 - pi) * n_B / n_T
    r_T, r_B = r[:n_T], r[n_T:]
    s_T, s_B = np.asarray(strata_T).astype(str), np.asarray(strata_B).astype(str)
    groups = [("all", np.ones(n_T, bool), np.ones(n_B, bool))]
    groups += [(s, s_T == s, s_B == s) for s in np.unique(np.r_[s_T, s_B])]
    rows = []
    for name, in_T, in_B in groups:
        rows.append(
            {
                "stratum": name,
                "n_T": int(in_T.sum()),
                "n_B": int(in_B.sum()),
                "share_unsupported": float(np.mean(r_T[in_T] > cut)) if in_T.any() else np.nan,
                "ess_B": kish_ess(r_B[in_B]),
                "max_share": max_weight_share(r_B[in_B]),
            }
        )
    return {"auc": float(roc_auc_score(y, pi)), "table": pd.DataFrame(rows), "r_T": r_T, "r_B": r_B}


def w3_probe(D_s_trial, A, grade_cols, names=None):
    """W3 probe (outcome-free): residual SMD of grade (and size) after ATT weighting on D_s only, within trial.

    Parameters
    ----------
    D_s_trial : numpy.ndarray
        :func:`design_source` of all trial patients.
    A : array-like
        0/1 treatment.
    grade_cols : numpy.ndarray or pandas.DataFrame
        Probe columns, e.g. grade indicators and tsize_mm.
    names : list of str, optional
        Defaults to the DataFrame columns or ``probe_<j>``.

    Returns
    -------
    pandas.DataFrame
        :func:`smd_table` of the probe columns under logistic ATT weights fitted on D_s.
    """
    if names is None and isinstance(grade_cols, pd.DataFrame):
        names = list(grade_cols.columns)
    elif names is None:
        names = [f"probe_{j}" for j in range(np.asarray(grade_cols).shape[1])]
    e = ps_logistic(D_s_trial, A)
    return smd_table(np.asarray(grade_cols, float), A, balancing_weights(e, A, "ATT"), names)
