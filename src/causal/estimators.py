import numpy as np
import pandas as pd
from scipy import stats

from src.causal.nuisance import _crossfit_ols, _ols_predict
from src.causal.sensitivity import evalue, evalue_ci, tipping_point
from src.causal.weights import entropy_balance, kish_ess, ps_logistic


def aipw(theta, A, e, mu0, mu1, estimand):
    """AIPW estimate of the effect, with arm means and scaled influence functions.

    For example, ATT: S1 = mean_T(theta), S0 = [sum_T mu0 + sum_C o (theta - mu0)] / n_T with o = e / (1 - e).

    Parameters
    ----------
    theta : array-like
        Outcome (pseudo-observations).
    A : array-like
        0/1 treatment.
    e : array-like
        Propensity score P(A = 1 | x).
    mu0, mu1 : array-like
        Outcome regressions (out-of-fold); ``mu1`` is unused for ATT, ``mu0`` for ATC.
    estimand : {"ATE", "ATT", "ATC", "ATO"}
        ATO solves the efficient influence-function estimating equation, so it is orthogonal in e.

    Returns
    -------
    dict
        estimand, est, se, lo, hi (normal 95% CI), s1, s0, if_s1, if_s0.
    """
    theta, A, e, mu0, mu1 = (np.asarray(v, float) for v in (theta, A, e, mu0, mu1))
    n = len(A)
    n1 = A.sum()
    n0 = n - n1
    if estimand == "ATE":
        p1 = mu1 + A * (theta - mu1) / e
        p0 = mu0 + (1 - A) * (theta - mu0) / (1 - e)
        s1, s0 = p1.mean(), p0.mean()
        i1, i0 = (p1 - s1) / n, (p0 - s0) / n
    elif estimand == "ATT":
        o = e / (1 - e)
        s1 = (A * theta).sum() / n1
        s0 = (A * mu0 + (1 - A) * o * (theta - mu0)).sum() / n1
        i1 = A * (theta - s1) / n1
        i0 = (A * (mu0 - s0) + (1 - A) * o * (theta - mu0)) / n1
    elif estimand == "ATC":
        v = (1 - e) / e
        s1 = ((1 - A) * mu1 + A * v * (theta - mu1)).sum() / n0
        s0 = ((1 - A) * theta).sum() / n0
        i1 = ((1 - A) * (mu1 - s1) + A * v * (theta - mu1)) / n0
        i0 = (1 - A) * (theta - s0) / n0
    elif estimand == "ATO":
        h = e * (1 - e)
        c = (1 - 2 * e) * (A - e)  # h'(e)(A - e) makes the estimating equation orthogonal in e
        denom = (h + c).sum()
        s1 = (h * mu1 + A * (1 - e) * (theta - mu1) + c * mu1).sum() / denom
        s0 = (h * mu0 + (1 - A) * e * (theta - mu0) + c * mu0).sum() / denom
        i1 = ((h + c) * (mu1 - s1) + A * (1 - e) * (theta - mu1)) / h.sum()
        i0 = ((h + c) * (mu0 - s0) + (1 - A) * e * (theta - mu0)) / h.sum()
    else:
        raise ValueError(f"unknown estimand '{estimand}'")
    est = s1 - s0
    se = np.sqrt(np.sum((i1 - i0) ** 2))
    return {
        "estimand": estimand,
        "est": float(est),
        "se": float(se),
        "lo": float(est - 1.96 * se),
        "hi": float(est + 1.96 * se),
        "s1": float(s1),
        "s0": float(s0),
        "if_s1": i1,
        "if_s0": i0,
    }


def ato_population(X, e, names):
    """Covariate means of the sample and of the ATO population (weights h = e(1 - e))."""
    X, e = np.asarray(X, float), np.asarray(e, float)
    h = e * (1 - e)
    return pd.DataFrame({"name": names, "mean_all": X.mean(0), "mean_ato": h @ X / h.sum()})


def hajek_att(theta, A, e):
    """Hajek IPW ATT = mean_T theta - sum_C o theta / sum_C o, o = e/(1-e) (report only)."""
    theta, A, e = np.asarray(theta, float), np.asarray(A).astype(bool), np.asarray(e, float)
    o = e[~A] / (1 - e[~A])
    return float(theta[A].mean() - np.sum(o * theta[~A]) / o.sum())


def gformula_att(theta, A, mu0):
    """g-formula ATT = mean_T (theta - mu0) (report only)."""
    theta, A, mu0 = np.asarray(theta, float), np.asarray(A).astype(bool), np.asarray(mu0, float)
    return float(np.mean(theta[A] - mu0[A]))


def agreement(theta, A, e, mu0):
    """Agreement of the AIPW ATT with its Hajek IPW and g-formula versions (report only).

    Returns
    -------
    dict
        aipw, se (AIPW SE), ipw, gformula, diff_gformula, diff_ipw (AIPW minus each), diff_in_se (the two
        differences in units of the AIPW SE), se_diff_gformula / se_diff_ipw (SE of the IF difference, nuisances
        treated as fixed).
    """
    theta, A, e, mu0 = (np.asarray(v, float) for v in (theta, A, e, mu0))
    r = aipw(theta, A, e, mu0, mu0, "ATT")
    ipw, gf = hajek_att(theta, A, e), gformula_att(theta, A, mu0)
    n1, ctrl, o = A.sum(), A == 0, e / (1 - e)
    if_aipw = r["if_s1"] - r["if_s0"]
    if_ipw = r["if_s1"] - (1 - A) * o * (theta - (r["s1"] - ipw)) / o[ctrl].sum()
    if_g = r["if_s1"] - A * (mu0 - (r["s1"] - gf)) / n1
    dg, di = r["est"] - gf, r["est"] - ipw
    return {
        "aipw": r["est"],
        "se": r["se"],
        "ipw": ipw,
        "gformula": gf,
        "diff_gformula": dg,
        "diff_ipw": di,
        "diff_in_se": {"gformula": dg / r["se"], "ipw": di / r["se"]},
        "se_diff_gformula": float(np.sqrt(np.sum((if_aipw - if_g) ** 2))),
        "se_diff_ipw": float(np.sqrt(np.sum((if_aipw - if_ipw) ** 2))),
    }


def mean_with_if(theta):
    """Sample mean and its scaled IF (theta - mean) / n."""
    theta = np.asarray(theta, float)
    m = theta.mean()
    return float(m), (theta - m) / len(theta)


def dr_counterfactual_mean(mu_target, theta_comp, mu_comp, w_comp):
    """Doubly robust counterfactual mean M = mean(mu_target) + sum w (theta_comp - mu_comp), with sum w = 1.

    Returns
    -------
    M : float
    if_target : numpy.ndarray
        (mu_target - mean) / n_target.
    if_comp : numpy.ndarray
        w (r - sum w r), r = theta_comp - mu_comp.
    """
    mu_target = np.asarray(mu_target, float)
    w = np.asarray(w_comp, float)
    w = w / w.sum()
    r = np.asarray(theta_comp, float) - np.asarray(mu_comp, float)
    m, bias = mu_target.mean(), np.sum(w * r)
    return float(m + bias), (mu_target - m) / len(mu_target), w * (r - bias)


def recalibrate_mu0(theta_C, Zc, Zt, folds_C):
    """Robust-hybrid outcome model: OLS of theta on Z among the comparison units.

    Returns
    -------
    mu0_C : numpy.ndarray
        Cross-fitted (out-of-fold) predictions for the comparison units.
    mu0_T : numpy.ndarray
        Predictions of the OLS fitted on all comparison units, for the target units. Both clipped to [0, 1].
    """
    theta_C, Zc, Zt = np.asarray(theta_C, float), np.asarray(Zc, float), np.asarray(Zt, float)
    mu0_C = _crossfit_ols(Zc, theta_C, np.ones(len(theta_C), bool), np.asarray(folds_C))
    return mu0_C, np.clip(_ols_predict(Zc, theta_C, Zt), 0, 1)


def effect_summary(s1, s0, if1, if0, df=None, level=0.95):
    """Risk difference, event risk ratio and E-values from arm means and their scaled IFs.

    Parameters
    ----------
    s1, s0 : float
        Event-free probabilities with and without treatment.
    if1, if0 : array-like
        Scaled IFs of s1 and s0 on the same (union) index.
    df : float, optional
        Degrees of freedom of a t interval; None gives a normal interval.
    level : float

    Returns
    -------
    dict
        est (s1 - s0), se, lo, hi, df, q, s1, s0, se_s1, se_s0, rr = (1 - s1)/(1 - s0) with rr_lo / rr_hi
        (log scale, same quantile), evalue, evalue_ci.
    """
    if1, if0 = np.asarray(if1, float), np.asarray(if0, float)
    est = s1 - s0
    se = float(np.sqrt(np.sum((if1 - if0) ** 2)))
    a = 1 - (1 - level) / 2
    q = float(stats.t.ppf(a, df) if df is not None else stats.norm.ppf(a))
    # numpy floats, so that S0 = 1 (no event among the comparison units) gives inf/nan, not ZeroDivisionError
    e1, e0 = np.float64(1 - s1), np.float64(1 - s0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = e1 / e0
        se_lrr = np.sqrt(np.sum((-if1 / e1 + if0 / e0) ** 2))
        rr_lo, rr_hi = rr * np.exp(-q * se_lrr), rr * np.exp(q * se_lrr)
    return {
        "est": float(est),
        "se": se,
        "lo": float(est - q * se),
        "hi": float(est + q * se),
        "df": df,
        "q": q,
        "s1": float(s1),
        "s0": float(s0),
        "se_s1": float(np.sqrt(np.sum(if1**2))),
        "se_s0": float(np.sqrt(np.sum(if0**2))),
        "rr": float(rr),
        "rr_lo": float(rr_lo),
        "rr_hi": float(rr_hi),
        "evalue": evalue(rr),
        "evalue_ci": evalue_ci(rr_lo, rr_hi),
    }


def robust_hybrid_att(theta_T, theta_C, D_T, D_C, Z_T, Z_C, folds_C):
    """Robust-hybrid ATT: entropy-balanced kept controls plus a recalibrated prognostic-score outcome model.

    ATT = mean_T(theta) - [mean_T(mu0) + sum_C w (theta - mu0)], with w the entropy-balancing weights of the
    kept controls and mu0 the OLS of theta on Z (cross-fitted for the controls).

    Parameters
    ----------
    theta_T, theta_C : array-like
        Pseudo-observations of the treated and the kept controls.
    D_T, D_C : numpy.ndarray
        :func:`design_trial` of the treated and the kept controls (balanced exactly).
    Z_T, Z_C : numpy.ndarray
        :func:`recalibration_design` of both groups ([m_B, grade II, grade III, tsize_mm]).
    folds_C : array-like of int
        Folds of the kept controls (stratified by menostat).

    Returns
    -------
    dict
        :func:`effect_summary` (t interval, df = ESS - 1) plus ``w``, ``ess``, ``max_share``, ``imbalance``,
        ``converged``, ``mu0_T``, ``mu0_C``.
    """
    theta_T, theta_C = np.asarray(theta_T, float), np.asarray(theta_C, float)
    eb = entropy_balance(D_C, np.asarray(D_T, float).mean(0))
    mu0_C, mu0_T = recalibrate_mu0(theta_C, Z_C, Z_T, folds_C)
    s1, i1_T = mean_with_if(theta_T)
    s0, i0_T, i0_C = dr_counterfactual_mean(mu0_T, theta_C, mu0_C, eb["w"])
    res = effect_summary(s1, s0, np.r_[i1_T, np.zeros(len(theta_C))], np.r_[i0_T, i0_C], df=eb["ess"] - 1)
    return {
        **res,
        "w": eb["w"],
        "ess": eb["ess"],
        "max_share": eb["max_share"],
        "imbalance": eb["imbalance"],
        "converged": eb["converged"],
        "mu0_T": mu0_T,
        "mu0_C": mu0_C,
    }


def robust_hybrid_atc(theta_T, theta_C, D_T, D_C, mu1_T, mu1_C):
    """Robust-hybrid ATC (secondary estimand; the population is the kept controls).

    ATC = [mean_C(mu1) + sum_T w (theta - mu1)] - mean_C(theta), with w the weights that balance the treated
    to the kept controls' D_t means and mu1 the out-of-fold T-learner on the treated.

    Returns
    -------
    dict
        :func:`effect_summary` (t interval, df = min(ESS_T, n_C) - 1) plus ``w`` (treated weights), ``ess``,
        ``max_share``, ``imbalance``, ``converged``.
    """
    theta_T, theta_C = np.asarray(theta_T, float), np.asarray(theta_C, float)
    eb = entropy_balance(D_T, np.asarray(D_C, float).mean(0))
    s0, j0_C = mean_with_if(theta_C)
    s1, j1_C, j1_T = dr_counterfactual_mean(mu1_C, theta_T, mu1_T, eb["w"])
    res = effect_summary(
        s1, s0, np.r_[j1_T, j1_C], np.r_[np.zeros(len(theta_T)), j0_C], df=min(eb["ess"], len(theta_C)) - 1
    )
    return {
        **res,
        "w": eb["w"],
        "ess": eb["ess"],
        "max_share": eb["max_share"],
        "imbalance": eb["imbalance"],
        "converged": eb["converged"],
    }


def hajek_eb_att(theta_T, theta_C, w, df=None):
    """Descriptive comparator: treated mean minus the entropy-balanced (Hajek) mean of the kept controls.

    S0 = sum w theta_C with IF w (theta_C - S0), weights normalized to sum to 1.

    Parameters
    ----------
    theta_T, theta_C : array-like
        Pseudo-observations of the treated and the kept controls.
    w : array-like
        Entropy-balancing weights of the kept controls (e.g. ``robust_hybrid_att(...)["w"]``).
    df : float, optional
        Degrees of freedom of a t interval; None gives a normal interval.

    Returns
    -------
    dict
        :func:`effect_summary`.
    """
    theta_T, theta_C, w = np.asarray(theta_T, float), np.asarray(theta_C, float), np.asarray(w, float)
    w = w / w.sum()
    s1, i1 = mean_with_if(theta_T)
    s0 = float(np.sum(w * theta_C))
    return effect_summary(
        s1, s0, np.r_[i1, np.zeros(len(theta_C))], np.r_[np.zeros(len(theta_T)), w * (theta_C - s0)], df=df
    )


def negative_control_beta(theta_C, m_C, w):
    """Negative-control comparison of trial controls with the external prognostic score.

    beta = sum w (theta - m), se = sqrt(sum w^2 (theta - m - beta)^2), weights normalized to sum to 1.

    Returns
    -------
    tuple of float
        (beta, se, lo90, hi90).
    """
    w = np.asarray(w, float)
    w = w / w.sum()
    r = np.asarray(theta_C, float) - np.asarray(m_C, float)
    beta = float(np.sum(w * r))
    se = float(np.sqrt(np.sum(w**2 * (r - beta) ** 2)))
    z = float(stats.norm.ppf(0.95))
    return beta, se, beta - z * se, beta + z * se


def transport_att(theta_T, mB_T, theta_B, mB_B_oof, Ds_T, Ds_B):
    """Scope E (exploratory): treated vs Rotterdam controls transported with logistic source odds.

    q = P(treated | D_s) fitted on treated + Rotterdam at natural class sizes; w_B proportional to q / (1 - q);
    S0 = mean_T(m_B) + sum_B w_B (theta_B - m_B^oof). Relies on untestable cross-source exchangeability.


    Parameters
    ----------
    theta_T, mB_T : array-like
        Treated pseudo-observations and their prognostic score (full-B fit).
    theta_B, mB_B_oof : array-like
        Rotterdam pseudo-observations and out-of-fold prognostic score.
    Ds_T, Ds_B : numpy.ndarray
        :func:`design_source` of both groups.

    Returns
    -------
    dict
        :func:`effect_summary` (normal interval) plus ``gformula`` (mean_T theta - mean_T m_B), ``ess_B``,
        ``max_share_B``, ``tipping`` (point, CI-limit tipping points) and ``w_B``.
    """
    theta_T, theta_B = np.asarray(theta_T, float), np.asarray(theta_B, float)
    n_T, n_B = len(theta_T), len(theta_B)
    q = ps_logistic(np.vstack([Ds_T, Ds_B]), np.r_[np.ones(n_T), np.zeros(n_B)])[n_T:]
    w_B = q / (1 - q)
    w_B = w_B / w_B.sum()
    s1, i1 = mean_with_if(theta_T)
    s0, i0_T, i0_B = dr_counterfactual_mean(mB_T, theta_B, mB_B_oof, w_B)
    res = effect_summary(s1, s0, np.r_[i1, np.zeros(n_B)], np.r_[i0_T, i0_B])
    return {
        **res,
        "gformula": float(s1 - np.mean(mB_T)),
        "ess_B": kish_ess(w_B),
        "max_share_B": float(w_B.max()),
        "tipping": tipping_point(res["est"], res["se"]),
        "w_B": w_B,
    }
