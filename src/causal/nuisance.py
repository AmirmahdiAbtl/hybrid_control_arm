import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import models


def crossfit_tlearner(F, theta, A, folds, backend, seed=0, cat_idx=None, arms=(0, 1), purpose="", cache_dir=None):
    """Out-of-fold T-learner mu_a(x) = E[theta | x, A = a] for every unit.

    For each arm a and fold k, the model is fitted on ``(A == a) & (folds != k)`` and predicts all units of
    fold k in one batched call.

    Parameters
    ----------
    F : numpy.ndarray
        Features (e.g. :func:`features_trial`).
    theta : array-like
        Pseudo-observations.
    A : array-like
        0/1 arm.
    folds : array-like of int
    backend : str
        Model backend of ``models.fit_predict_reg``.
    seed : int
    cat_idx : list of int, optional
    arms : tuple of int
        Arms to model.
    purpose : str
        Prefix of the FIT_LOG purpose (``<purpose>_mu<a>_fold<k>``).
    cache_dir : str, optional
        TabPFN prediction cache, passed to ``models.fit_predict_reg``.

    Returns
    -------
    dict
        arm -> predictions clipped to [0, 1].
    """
    F, theta, A, folds = np.asarray(F, float), np.asarray(theta, float), np.asarray(A), np.asarray(folds)
    out = {}
    for a in arms:
        mu = np.empty(len(theta))
        for k in np.unique(folds):
            tr, te = (A == a) & (folds != k), folds == k
            tag = "_".join(p for p in (purpose, f"mu{a}", f"fold{k}") if p)
            mu[te] = models.fit_predict_reg(
                backend, F[tr], theta[tr], [F[te]], seed, cat_idx, purpose=tag, cache_dir=cache_dir
            )[0]
        out[a] = np.clip(mu, 0, 1)
    return out


def prognostic_score(F_B, D_B, theta_B, F_T, D_T, folds_B, backend, seed=0, cat_idx=None, purpose="mB", cache_dir=None):
    """Prognostic score m_B fitted on Rotterdam only (gate G7).

    Model path (``backend``): one fit per fold to predict Rotterdam out of fold, plus one fit on all of Rotterdam
    to predict the trial. Linear path (always fitted): OLS on ``D_B`` with the same folds, then on all of
    Rotterdam to predict ``D_T``.

    Parameters
    ----------
    F_B, D_B : numpy.ndarray
        Rotterdam features (model path) and design (linear path).
    theta_B : array-like
        Rotterdam pseudo-observations.
    F_T, D_T : numpy.ndarray
        The same for the trial rows to score.
    folds_B : array-like of int
    backend : str
    seed : int
    cat_idx : list of int, optional
        Categorical columns of ``F_B`` / ``F_T``.
    purpose : str
        FIT_LOG purpose prefix (``<purpose>_cv_fold<k>``, ``<purpose>_full``).
    cache_dir : str, optional

    Returns
    -------
    dict
        ``oof``, ``oof_lin`` (Rotterdam out-of-fold), ``m_T``, ``m_T_lin`` (trial scores), ``mse``, ``mse_lin``,
        ``mse_null`` (fold-wise training means), ``se_diff`` (SE of the paired MSE difference), ``sigma`` /
        ``sigma_lin`` (SD of the out-of-fold residual), ``cal_int``, ``cal_int_ci``, ``cal_slope``,
        ``cal_slope_ci`` (model path). All predictions are clipped to [0, 1].
    """
    F_B, D_B, F_T, D_T = (np.asarray(v, float) for v in (F_B, D_B, F_T, D_T))
    theta, folds = np.asarray(theta_B, float), np.asarray(folds_B)
    oof, oof_lin, null = np.empty(len(theta)), np.empty(len(theta)), np.empty(len(theta))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        oof[te] = models.fit_predict_reg(
            backend, F_B[tr], theta[tr], [F_B[te]], seed, cat_idx, purpose=f"{purpose}_cv_fold{k}", cache_dir=cache_dir
        )[0]
        oof_lin[te] = _ols_predict(D_B[tr], theta[tr], D_B[te])
        null[te] = theta[tr].mean()
    m_T = models.fit_predict_reg(
        backend, F_B, theta, [F_T], seed, cat_idx, purpose=f"{purpose}_full", cache_dir=cache_dir
    )[0]
    oof, oof_lin = np.clip(oof, 0, 1), np.clip(oof_lin, 0, 1)
    m_T, m_T_lin = np.clip(m_T, 0, 1), np.clip(_ols_predict(D_B, theta, D_T), 0, 1)
    paired_diff = (theta - oof) ** 2 - (theta - oof_lin) ** 2
    return {
        "oof": oof,
        "oof_lin": oof_lin,
        "m_T": m_T,
        "m_T_lin": m_T_lin,
        "mse": float(np.mean((theta - oof) ** 2)),
        "mse_lin": float(np.mean((theta - oof_lin) ** 2)),
        "mse_null": float(np.mean((theta - null) ** 2)),
        "se_diff": float(paired_diff.std(ddof=1) / np.sqrt(len(paired_diff))),
        "sigma": float(np.std(theta - oof, ddof=1)),
        "sigma_lin": float(np.std(theta - oof_lin, ddof=1)),
        **_calibration(theta, oof),
    }


def outcome_diagnostics(theta, A, folds, mu, mu_lin):
    """Out-of-fold accuracy and calibration of the outcome models, per arm (report only).

    Parameters
    ----------
    theta, A, folds : array-like
    mu, mu_lin : dict
        arm -> out-of-fold predictions of the model and of the linear baseline (from :func:`crossfit_tlearner`).

    Returns
    -------
    pandas.DataFrame
        arm, n, mse, mse_lin, mse_null (fold-wise training mean of the arm), cv_r2, cv_r2_lin, calibration.
    """
    theta, A, folds = np.asarray(theta, float), np.asarray(A), np.asarray(folds)
    rows = []
    for a in sorted(mu):
        in_arm = A == a
        null = np.empty(len(theta))
        for k in np.unique(folds):
            null[folds == k] = theta[in_arm & (folds != k)].mean()
        mse, mse_lin, mse_null = (float(np.mean((theta[in_arm] - p[in_arm]) ** 2)) for p in (mu[a], mu_lin[a], null))
        rows.append(
            {
                "arm": a,
                "n": int(in_arm.sum()),
                "mse": mse,
                "mse_lin": mse_lin,
                "mse_null": mse_null,
                "cv_r2": 1 - mse / mse_null,
                "cv_r2_lin": 1 - mse_lin / mse_null,
                **_calibration(theta[in_arm], mu[a][in_arm]),
            }
        )
    return pd.DataFrame(rows)


def _ols_predict(X_train, y_train, X_test):
    """OLS with intercept fitted on the training rows; predictions for ``X_test``."""
    model = LinearRegression().fit(np.asarray(X_train, float), np.asarray(y_train, float))
    return model.predict(np.asarray(X_test, float))


def _crossfit_ols(X, y, train, folds):
    """Out-of-fold OLS: fold k is predicted from ``train & (folds != k)``; clipped to [0, 1]."""
    pred = np.empty(len(y))
    for k in np.unique(folds):
        pred[folds == k] = _ols_predict(X[train & (folds != k)], y[train & (folds != k)], X[folds == k])
    return np.clip(pred, 0, 1)


def _calibration(theta, pred):
    """Calibration-in-the-large (mean residual, normal CI) and slope (OLS of theta on pred, CI from its SE)."""
    theta, pred = np.asarray(theta, float), np.asarray(pred, float)
    n = len(theta)
    resid = theta - pred
    half_width = 1.96 * resid.std(ddof=1) / np.sqrt(n)
    sxx = np.sum((pred - pred.mean()) ** 2)
    if sxx > 0 and n > 2:
        slope = np.sum((pred - pred.mean()) * (theta - theta.mean())) / sxx
        slope_resid = theta - theta.mean() - slope * (pred - pred.mean())
        se = np.sqrt(np.sum(slope_resid**2) / (n - 2) / sxx)
    else:
        slope, se = np.nan, np.nan
    return {
        "cal_int": float(resid.mean()),
        "cal_int_ci": (float(resid.mean() - half_width), float(resid.mean() + half_width)),
        "cal_slope": float(slope),
        "cal_slope_ci": (float(slope - 1.96 * se), float(slope + 1.96 * se)),
    }
