import numpy as np
from scipy import stats


def ate_scores(theta, A, e, mu0, mu1):
    """Doubly robust ATE scores phi = mu1 - mu0 + A(theta - mu1)/e - (1 - A)(theta - mu0)/(1 - e)."""
    theta, A, e, mu0, mu1 = (np.asarray(v, float) for v in (theta, A, e, mu0, mu1))
    return mu1 - mu0 + A * (theta - mu1) / e - (1 - A) * (theta - mu0) / (1 - e)


def subgroup_att(theta, A, e, mu0, mask):
    """Subgroup ATT with its scaled IF (zero outside g).

    CATT_g = sum_g [A (theta - mu0) - (1 - A) o (theta - mu0)] / n_Tg, with o = e / (1 - e) and n_Tg the number
    of treated in g.

    Returns
    -------
    dict
        est, se, if.
    """
    theta, A, e, mu0 = (np.asarray(v, float) for v in (theta, A, e, mu0))
    g = np.asarray(mask, bool)
    o = e / (1 - e)
    n_treated_g = np.sum(A * g)
    psi = np.sum(g * (A * (theta - mu0) - (1 - A) * o * (theta - mu0))) / n_treated_g
    inf = g * (A * (theta - mu0 - psi) - (1 - A) * o * (theta - mu0)) / n_treated_g
    return {"est": float(psi), "se": float(np.sqrt(np.sum(inf**2))), "if": inf}


def subgroup_ate(phi_ate, mask):
    """Subgroup ATE: GATE_g = mean of the ATE scores in g, with IF 1{g} (phi - GATE_g) / n_g.

    Returns
    -------
    dict
        est, se, if.
    """
    phi, g = np.asarray(phi_ate, float), np.asarray(mask, bool)
    est = phi[g].mean()
    inf = g * (phi - est) / g.sum()
    return {"est": float(est), "se": float(np.sqrt(np.sum(inf**2))), "if": inf}


def contrast(a, b):
    """Difference between two disjoint subgroup estimates.

    The SE adds the two variances, which is exact for disjoint subgroups (their IFs have disjoint supports) with
    the nuisances treated as fixed.

    Parameters
    ----------
    a, b : dict
        Outputs of :func:`subgroup_att` or :func:`subgroup_ate` (need ``est`` and ``se``).

    Returns
    -------
    dict
        est (a - b), se = sqrt(se_a^2 + se_b^2), p (two-sided normal test of est = 0).
    """
    est = a["est"] - b["est"]
    se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
    return {"est": float(est), "se": se, "p": float(2 * stats.norm.sf(abs(est / se)))}


def simultaneous_band(ifs, ses, draws=2000, seed=0):
    """Critical value of simultaneous 95% intervals over several subgroup estimates (multiplier bootstrap).

    T_m = max_g |xi_m . IF_g| / SE_g with xi_m ~ N(0, I_n), and c = 95th percentile of T. In theory
    1.96 <= c <= the Bonferroni value, so c is clipped to that range to remove Monte Carlo noise.

    Parameters
    ----------
    ifs : list of numpy.ndarray
        Scaled IFs (one per subgroup estimate, on the same index).
    ses : array-like
        Their SEs.
    draws, seed : int

    Returns
    -------
    float
        The critical value c.
    """
    IF = np.column_stack(ifs)
    se = np.asarray(ses, float)
    xi = np.random.default_rng(seed).standard_normal((draws, IF.shape[0]))
    c = np.quantile(np.abs(xi @ IF / se).max(1), 0.95)
    return float(np.clip(c, stats.norm.ppf(0.975), stats.norm.ppf(1 - 0.025 / IF.shape[1])))


def blp(phi_ate, score):
    """Best linear predictor of the effect: OLS of the ATE scores phi on [1, score - mean(score)], HC3 SEs.

    Returns
    -------
    dict
        beta1 (mean effect), beta2 (heterogeneity loading), se1, se2, p2 (beta2 = 0).
    """
    phi, s = np.asarray(phi_ate, float), np.asarray(score, float)
    X = np.column_stack([np.ones(len(s)), s - s.mean()])
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ phi
    res = phi - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    V = bread @ (X.T * (res**2 / (1 - leverage) ** 2)) @ X @ bread
    se = np.sqrt(np.diag(V))
    return {
        "beta1": float(beta[0]),
        "beta2": float(beta[1]),
        "se1": float(se[0]),
        "se2": float(se[1]),
        "p2": float(2 * stats.norm.sf(abs(beta[1] / se[1]))),
    }
