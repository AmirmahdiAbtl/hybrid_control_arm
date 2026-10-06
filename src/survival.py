from collections import namedtuple

import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_YEAR = 365.25
KM = namedtuple("KM", "surv se")


def km_at(time, event, horizon):
    """Kaplan-Meier survival at ``horizon`` with Greenwood standard error.

    Parameters
    ----------
    time, event : array-like
        Follow-up time and event indicator (1 = event, 0 = censored).
    horizon : float
        Time point, same unit as ``time``.

    Returns
    -------
    KM
        ``surv`` and ``se`` (both NaN if there are no data). Beyond the last
        observed time the curve stays flat.
    """
    t = np.asarray(time, float)
    e = np.asarray(event, float)
    ok = ~(np.isnan(t) | np.isnan(e))
    t, e = t[ok], e[ok]
    if len(t) == 0:
        return KM(np.nan, np.nan)
    s, gw = 1.0, 0.0
    for u in np.unique(t[(e == 1) & (t <= horizon)]):
        n = np.sum(t >= u)
        d = np.sum((t == u) & (e == 1))
        s *= 1 - d / n
        if n > d:
            gw += d / (n * (n - d))
    return KM(float(s), float(s * np.sqrt(gw)))


def km_ci(surv, se, z=1.96):
    """Confidence interval for a Kaplan-Meier survival probability on the log(-log) scale.

    Unlike ``surv +- z * se`` the interval always stays inside [0, 1].

    Parameters
    ----------
    surv, se : float
        Survival probability and its Greenwood standard error (see :func:`km_at`).
    z : float
        Normal quantile; 1.96 gives a 95% interval.

    Returns
    -------
    tuple of float
        ``(lo, hi)``. ``(surv, surv)`` when ``surv`` is 0 or 1; NaN when ``surv`` or ``se`` is NaN.
    """
    if np.isnan(surv) or np.isnan(se):
        return np.nan, np.nan
    if surv <= 0 or surv >= 1:
        return float(surv), float(surv)
    se_t = se / (surv * abs(np.log(surv)))  # delta-method SE of log(-log S)
    return float(surv ** np.exp(z * se_t)), float(surv ** np.exp(-z * se_t))


def event_free_labels(time, event, horizon):
    """Binary label at ``horizon``: 1 = event-free, 0 = event by then, NaN = censored before (status unknown).

    Complete-case; used by the legacy rows only, never by the gated analysis.
    """
    t = np.asarray(time, float)
    e = np.asarray(event, float)
    y = np.full(len(t), np.nan)
    y[t > horizon] = 1.0
    y[(t <= horizon) & (e == 1)] = 0.0
    return y


def pseudo_km(time, event, horizon):
    """Jackknife Kaplan-Meier pseudo-observations of S(horizon) (vectorized, exact).

    Returns
    -------
    numpy.ndarray
        theta_i = n S(H) - (n - 1) S^(-i)(H); never clipped.
    """
    t = np.asarray(time, float)
    e = np.asarray(event, float)
    n = len(t)
    u = np.unique(t[(e == 1) & (t <= horizon)])
    if len(u) == 0:
        return np.ones(n)
    at = t[:, None] >= u[None, :]
    died = (t[:, None] == u[None, :]) & (e[:, None] == 1)
    nj, dj = at.sum(0), died.sum(0)
    den = nj - at  # at risk without subject i
    fac = np.where(den > 0, 1 - (dj - died) / np.maximum(den, 1), 1.0)
    return n * np.prod(1 - dj / nj) - (n - 1) * fac.prod(1)


def pseudo_by_strata(time, event, horizon, fine, coarse, min_n=25, min_at_risk=10):
    """Pseudo-observations within strata, merging the fine strata of a coarse cell when any is small.

    Pre-specified rule: within each coarse cell (source|arm), if any fine sub-stratum
    (source|arm|menostat) has < ``min_n`` patients or < ``min_at_risk`` with time > horizon, the coarse cell is used.

    Parameters
    ----------
    fine, coarse : array-like of str
        Stratum labels per unit (see :func:`strata_labels`).

    Returns
    -------
    theta : numpy.ndarray
    used : numpy.ndarray of str
        The stratum each unit's pseudo-observation was computed in.
    """
    time, event = np.asarray(time, float), np.asarray(event, float)
    fine, coarse = np.asarray(fine, object), np.asarray(coarse, object)
    used = fine.copy()
    for c in np.unique(coarse):
        in_c = coarse == c
        if any(
            ((fine == s) & in_c).sum() < min_n or (time[(fine == s) & in_c] > horizon).sum() < min_at_risk
            for s in np.unique(fine[in_c])
        ):
            used[in_c] = c
    theta = np.empty(len(time))
    for s in np.unique(used):
        m = used == s
        theta[m] = pseudo_km(time[m], event[m], horizon)
    return theta, used


def reverse_km(time, event, horizon):
    """Reverse Kaplan-Meier G(H) = P(still under follow-up at H) (censoring as the event)."""
    return km_at(time, 1 - np.asarray(event, float), horizon).surv


def strata_labels(source, A, post):
    """Fine and coarse pseudo-observation strata.

    Parameters
    ----------
    source : str or array-like of str
        E.g. "A" (trial) or "B" (Rotterdam).
    A, post : array-like
        Arm (0/1) and postmenopausal indicator (0/1).

    Returns
    -------
    fine, coarse : numpy.ndarray of str
        ``"source|a|pre"`` / ``"source|a|post"`` and ``"source|a"``.
    """
    A, post = np.asarray(A), np.asarray(post)
    src = np.broadcast_to(np.asarray(source, dtype=object), A.shape)
    fine = np.array([f"{s}|{int(a)}|{'post' if p else 'pre'}" for s, a, p in zip(src, A, post)], dtype=object)
    coarse = np.array([f"{s}|{int(a)}" for s, a in zip(src, A)], dtype=object)
    return fine, coarse


def horizon_check(time, event, horizon, used_labels, g_min=0.70, min_at_risk=10, tol=1e-8):
    """Horizon adequacy and integrity per final pseudo-observation stratum (gate G9).

    Parameters
    ----------
    used_labels : array-like of str
        Second output of :func:`pseudo_by_strata`.
    g_min, min_at_risk, tol : float
        Pass rules: G(H) >= g_min, at least ``min_at_risk`` with time > H, |mean theta - KM| < tol.

    Returns
    -------
    pandas.DataFrame
        One row per stratum: stratum, n, censored_before_H, at_risk_H, G_H, identity_err, ok.
    """
    t, e = np.asarray(time, float), np.asarray(event, float)
    used = np.asarray(used_labels, object)
    rows = []
    for s in np.unique(used):
        m = used == s
        g = reverse_km(t[m], e[m], horizon)
        err = abs(pseudo_km(t[m], e[m], horizon).mean() - km_at(t[m], e[m], horizon).surv)
        at_risk = int((t[m] > horizon).sum())
        rows.append(
            {
                "stratum": s,
                "n": int(m.sum()),
                "censored_before_H": int(((t[m] <= horizon) & (e[m] == 0)).sum()),
                "at_risk_H": at_risk,
                "G_H": g,
                "identity_err": err,
                "ok": bool(g >= g_min and at_risk >= min_at_risk and err < tol),
            }
        )
    return pd.DataFrame(rows)


def cox_ph(time, event, X, max_iter=50, tol=1e-9):
    """Cox proportional-hazards model (Breslow ties, Newton-Raphson on centred covariates).

    Parameters
    ----------
    X : array-like
        Covariates (n x p).
    tol : float
        Stop when the largest Newton step is below ``tol``.

    Returns
    -------
    dict
        ``beta``, ``se``, ``loglik``, ``loglik0`` (beta = 0), ``lr_p`` (likelihood-ratio chi^2 test, df = p),
        ``x_mean`` (centring), ``times`` (event times) and ``dLambda0`` (Breslow baseline hazard increments at
        the centred covariates, d_u / sum_{t_i >= u} exp((x_i - x_mean)'beta)).

    Raises
    ------
    ValueError
        If ``time``, ``event`` or ``X`` has a missing or non-finite value (no imputation is pre-specified).
    """
    t, d = np.asarray(time, float), np.asarray(event, float)
    X = np.asarray(X, float).reshape(len(t), -1)
    if not (np.isfinite(X).all() and np.isfinite(t).all() and np.isfinite(d).all()):
        raise ValueError("cox_ph: time, event or covariates contain missing or non-finite values")
    x_mean = X.mean(0)
    Xc = X - x_mean
    p = Xc.shape[1]
    u = np.unique(t[d == 1])
    if len(u) == 0:
        return {
            "beta": np.zeros(p),
            "se": np.full(p, np.nan),
            "loglik": 0.0,
            "loglik0": 0.0,
            "lr_p": 1.0,
            "x_mean": x_mean,
            "times": u,
            "dLambda0": np.zeros(0),
        }
    at = (t[:, None] >= u[None, :]).astype(float)  # n x J risk sets
    dm = ((t[:, None] == u[None, :]) & (d[:, None] == 1)).astype(float)  # n x J events
    dj, xd = dm.sum(0), dm.T @ Xc

    def likelihood_parts(b):
        w = np.exp(Xc @ b)
        s0, s1 = at.T @ w, at.T @ (w[:, None] * Xc)
        return w, s0, s1, float(np.sum(xd @ b - dj * np.log(s0)))

    def information(w, s0, s1):
        s2 = np.einsum("ij,ik,il->jkl", at * w[:, None], Xc, Xc)
        return np.einsum("j,jkl->kl", dj / s0, s2) - np.einsum("j,jk,jl->kl", dj / s0**2, s1, s1)

    b = np.zeros(p)
    w, s0, s1, ll = likelihood_parts(b)
    ll0 = ll
    for _ in range(max_iter):
        grad = (xd - dj[:, None] * s1 / s0[:, None]).sum(0)
        step = np.linalg.solve(information(w, s0, s1) + 1e-9 * np.eye(p), grad)
        for _half in range(30):  # step halving keeps the log-likelihood from decreasing
            w_new, s0_new, s1_new, ll_new = likelihood_parts(b + step)
            if ll_new >= ll - 1e-12:
                break
            step = step / 2
        b = b + step
        w, s0, s1, ll = w_new, s0_new, s1_new, ll_new
        if np.max(np.abs(step)) < tol:
            break
    cov = np.linalg.inv(information(w, s0, s1) + 1e-9 * np.eye(p))
    lr = max(2 * (ll - ll0), 0.0)
    return {
        "beta": b,
        "se": np.sqrt(np.diag(cov)),
        "loglik": ll,
        "loglik0": ll0,
        "lr_p": float(stats.chi2.sf(lr, p)),
        "x_mean": x_mean,
        "times": u,
        "dLambda0": dj / s0,
    }


def censoring_G(time, event, Xc, horizon, min_events_per_coef=5, fallback_G=None):
    """P(still under follow-up just before H | x) from a reverse Cox model (sensitivity S4, gate G9 report).

    Censoring data: ``tc = min(time, H)``, ``dc = (event == 0) & (time <= H)`` (events and the horizon censor the
    censoring process). With at least ``min_events_per_coef`` censoring events per covariate a Cox model is fitted
    and G_i = exp(-Lambda0(H-) exp((x_i - x_mean)'beta)); otherwise ``fallback_G`` (the stratum reverse KM).

    Parameters
    ----------
    Xc : array-like
        Censoring covariates (see :func:`src.causal.censoring_covariates`).
    fallback_G : float or array-like, optional
        Default: the reverse KM of the given data at ``horizon``.

    Returns
    -------
    G : numpy.ndarray
    info : dict
        ``model``, ``n_events``, ``lr_p`` (NaN for the KM fallback).
    """
    t, e = np.asarray(time, float), np.asarray(event, float)
    X = np.asarray(Xc, float).reshape(len(t), -1)
    tc, dc = np.minimum(t, horizon), ((e == 0) & (t <= horizon)).astype(float)
    n_ev = int(dc.sum())
    if n_ev >= min_events_per_coef * X.shape[1]:
        fit = cox_ph(tc, dc, X)
        lam = fit["dLambda0"][fit["times"] < horizon].sum()
        G = np.exp(-lam * np.exp((X - fit["x_mean"]) @ fit["beta"]))
        return G, {"model": "Cox (censoring as event)", "n_events": n_ev, "lr_p": fit["lr_p"]}
    fb = reverse_km(t, e, horizon) if fallback_G is None else fallback_G
    G = np.broadcast_to(np.asarray(fb, float), t.shape).copy()
    return G, {"model": "KM (too few censoring events)", "n_events": n_ev, "lr_p": np.nan}


def ipcw_outcome(time, horizon, G_H, floor=0.10):
    """IPCW outcome Y* = 1{time > H} / max(G_H, floor) (sensitivity S4).

    The number of floored units is ``np.sum(np.asarray(G_H) < floor)``; report it next to the estimate.
    """
    return (np.asarray(time, float) > horizon) / np.maximum(np.asarray(G_H, float), floor)
