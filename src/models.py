import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BACKENDS = ["tabpfn-local", "tabpfn-client", "standin"]
BACKEND_LABELS = {
    "tabpfn-local": "TabPFN (local)",
    "tabpfn-client": "TabPFN (client, data leaves this machine)",
    "standin": "Random forest stand-in (testing only, NOT TabPFN)",
    "logreg": "logistic/linear regression baseline (NOT TabPFN)",
}
CHEAP_LABEL = "logistic/linear nuisance models (NOT TabPFN)"
NON_COMPARABLE = {"tgrade"}  # grade was recorded with different grading protocols in the two sources

GATED_HELP = (
    "TabPFN weights could not be downloaded (gated model). Fix, either:\n"
    "  1. accept the licence at https://huggingface.co/Prior-Labs/tabpfn_3_5 and run `hf auth login`, or\n"
    "  2. point HCA_TABPFN_MODEL_PATH at a local checkpoint, or\n"
    "  3. use the client backend: set TABPFN_TOKEN and HCA_BACKEND=tabpfn-client."
)

FIT_LOG = []  # one dict per fit: task, backend, purpose, n_train, n_test, seconds, cached, meta


class TabPFNSetupError(RuntimeError):
    """TabPFN is not usable yet (licence, login, token); the message says how to fix it."""


def default_backend():
    """Backend from ``HCA_BACKEND``; otherwise the TabPFN API client (``TABPFN_TOKEN``)."""
    return os.getenv("HCA_BACKEND") or "tabpfn-client"


def pick_device():
    """Device from ``HCA_DEVICE``; 'auto' = cuda, then mps (Apple), then cpu."""
    d = os.getenv("HCA_DEVICE", "auto")
    if d != "auto":
        return d
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def default_features(a_h, b_h, exclude):
    """Baseline features: columns present in both harmonized datasets minus treatment / outcome / ids."""
    return [c for c in a_h.columns if c in b_h.columns and c not in exclude]


def cross_source_features(features):
    """``features`` in the same order without the columns in ``NON_COMPARABLE`` (they differ between sources)."""
    return [f for f in features if f not in NON_COMPARABLE]


def encode_features(frames, features):
    """Encode several DataFrames with the same integer codes for text columns.

    Parameters
    ----------
    frames : list of pandas.DataFrame
        E.g. historical and trial patients; the categories are collected over all of them.
    features : list of str
        Columns to use.

    Returns
    -------
    arrays : list of numpy.ndarray
        One float matrix per frame (NaN stays NaN).
    cat_idx : list of int
        Indices of the categorical columns (to pass to TabPFN).
    """
    cat_idx, codes = [], {}
    for j, f in enumerate(features):
        if is_categorical([fr[f] for fr in frames]):
            levels = sorted({str(v) for fr in frames for v in fr[f].dropna().unique()})
            codes[f] = {v: i for i, v in enumerate(levels)}
            cat_idx.append(j)
    arrays = []
    for fr in frames:
        cols = []
        for f in features:
            if f in codes:
                cols.append(fr[f].map(lambda v: np.nan if pd.isna(v) else codes[f][str(v)]).astype(float).to_numpy())
            else:
                cols.append(fr[f].astype(float).to_numpy())
        arrays.append(np.column_stack(cols))
    return arrays, cat_idx


def is_categorical(cols):
    """True if any of the columns is non-numeric or boolean."""
    return any(not pd.api.types.is_numeric_dtype(c) or pd.api.types.is_bool_dtype(c) for c in cols)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit_predict(backend, X_train, y_train, X_tests, seed=0, cat_idx=None, purpose="", cache_dir=None):
    """Fit a classifier and return P(label = 1) for each test matrix.

    All test matrices are predicted in one batched call. Each fit is appended to ``FIT_LOG``.

    Parameters
    ----------
    backend : str
        ``tabpfn-local``, ``tabpfn-client``, ``standin`` or ``logreg``.
    X_train, y_train : numpy.ndarray
        Training data; ``y_train`` is 0/1 with no NaN.
    X_tests : list of numpy.ndarray
        Matrices to predict.
    cat_idx : list of int, optional
        Categorical column indices (one-hot encoded for ``logreg``).
    purpose : str
        What the fit is for; written to ``FIT_LOG``.
    cache_dir : str or pathlib.Path, optional
        TabPFN backends only: store predictions here and reuse them for identical inputs.

    Returns
    -------
    list of numpy.ndarray
        One probability vector per test matrix; an empty list if ``X_tests`` is empty. If ``y_train`` has a
        single class, that constant is returned without a fit.

    Raises
    ------
    TabPFNSetupError
        If the TabPFN weights are gated / missing or the API key is not set (the message has the fix).
    """
    y_train = np.asarray(y_train).astype(int)
    if len(X_tests) == 0:
        return []
    if len(np.unique(y_train)) < 2:
        return [np.full(len(X), float(y_train[0])) for X in X_tests]
    cat_idx = [int(c) for c in cat_idx or []]
    return _fit_and_predict(
        "classification",
        backend,
        lambda: _make_classifier(backend, seed, cat_idx),
        lambda m, X: m.predict_proba(X)[:, 1],
        X_train,
        y_train,
        X_tests,
        seed,
        cat_idx,
        purpose,
        cache_dir,
    )


def fit_predict_reg(backend, X_train, y_train, X_tests, seed=0, cat_idx=None, purpose="", cache_dir=None):
    """Fit a regressor and return its mean prediction for each test matrix.

    Same parameters, batching, ``FIT_LOG`` entry and cache as :func:`fit_predict`, except that ``y_train`` is a
    float outcome with no NaN (e.g. pseudo-observations). TabPFN returns the mean of its predictive distribution;
    ``logreg`` is least squares on the one-hot design; ``standin`` is a random forest.

    Returns
    -------
    list of numpy.ndarray
        One prediction vector per test matrix; an empty list if ``X_tests`` is empty. A constant ``y_train``
        gives that constant without a fit.
    """
    y_train = np.asarray(y_train, float)
    if len(X_tests) == 0:
        return []
    if len(y_train) and np.all(y_train == y_train[0]):
        return [np.full(len(X), y_train[0]) for X in X_tests]
    cat_idx = [int(c) for c in cat_idx or []]
    kw = {"output_type": "mean"} if backend.startswith("tabpfn") else {}
    return _fit_and_predict(
        "regression",
        backend,
        lambda: _make_regressor(backend, seed, cat_idx),
        lambda m, X: m.predict(X, **kw),
        X_train,
        y_train,
        X_tests,
        seed,
        cat_idx,
        purpose,
        cache_dir,
    )


def _fit_and_predict(task, backend, make, predict_fn, X_train, y_train, X_tests, seed, cat_idx, purpose, cache_dir):
    """One fit and one batched prediction of all test rows, logged in FIT_LOG; split back per test matrix.

    With ``cache_dir`` (TabPFN backends only) predictions are stored in ``<cache_dir>/<sha256>.npz`` and reused
    for identical inputs; a reuse is logged with ``cached=True`` and the stored metadata.
    """
    if any(np.ndim(X) != 2 for X in X_tests):  # vstack would accept a 1-D row and split wrongly
        raise ValueError("each test matrix must be 2-D (rows x features)")
    X_all = np.vstack(X_tests)
    cuts = np.cumsum([len(X) for X in X_tests])[:-1]
    entry = {"task": task, "backend": backend, "purpose": purpose, "n_train": len(y_train), "n_test": len(X_all)}
    t0 = time.perf_counter()
    path = None
    if cache_dir is not None and backend.startswith("tabpfn"):
        path = Path(cache_dir) / f"{_cache_key(task, backend, seed, cat_idx, X_train, y_train, X_tests)}.npz"
        if path.exists():
            with np.load(path) as z:
                preds = [z[f"arr_{i}"] for i in range(len(X_tests))]
                meta = json.loads(str(z["meta"]))
            FIT_LOG.append({**entry, "seconds": round(time.perf_counter() - t0, 3), "cached": True, "meta": meta})
            return preds
    model, pred = _fit_with_retry(backend, make, X_train, y_train, lambda m: predict_fn(m, X_all))
    meta = _fit_meta(backend, model)
    FIT_LOG.append({**entry, "seconds": round(time.perf_counter() - t0, 3), "cached": False, "meta": meta})
    preds = np.split(np.asarray(pred, float), cuts)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:  # write, then rename: an interrupted run leaves no broken cache file
            np.savez(f, *preds, meta=json.dumps(meta))
        os.replace(tmp, path)
    return preds


def _cache_key(task, backend, seed, cat_idx, X_train, y_train, X_tests):
    """SHA-256 of everything that determines a TabPFN prediction (model identity, seed, categorical columns, data)."""
    h = hashlib.sha256(json.dumps([task, backend, _model_id(backend), str(seed), list(cat_idx)]).encode())
    for a in (X_train, y_train, *X_tests):
        a = np.ascontiguousarray(a, dtype=float)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _fit_with_retry(backend, make, X, y, predict_fn):
    """Build a model with ``make()``, fit it on (X, y) and return ``(model, predict_fn(model))``.

    The TabPFN API sometimes answers with a temporary server error, so the client backend gets 3 tries.
    Gated-weights errors become :class:`TabPFNSetupError` with the fix.
    """
    tries = 3 if backend == "tabpfn-client" else 1
    for attempt in range(tries):
        try:
            model = make()
            model.fit(X, y)
            return model, predict_fn(model)
        except TabPFNSetupError:
            raise
        except Exception as e:
            if attempt + 1 < tries and not isinstance(e, (ValueError, TypeError)):
                time.sleep(2 * (attempt + 1))
                continue
            _raise_setup_error(backend, e)


def _raise_setup_error(backend, e):
    """Turn a gated-weights error into fix instructions; re-raise anything else unchanged."""
    msg = f"{type(e).__name__} {e}".lower()
    gated = ("gated", "401", "403", "huggingface", "hf auth", "access to model")
    if backend == "tabpfn-local" and any(k in msg for k in gated):
        raise TabPFNSetupError(GATED_HELP + f"\n\nOriginal error: {type(e).__name__}: {str(e)[:300]}") from e
    raise e


def _fit_meta(backend, model):
    """Model identity for the audit log; None for the cheap backends.

    TabPFN: package version and model name. Client: also the server's metadata of the last prediction (e.g. the
    billing model version) and the server timings.
    """
    if not backend.startswith("tabpfn"):
        return None
    meta = _model_id(backend)
    if backend == "tabpfn-client":
        try:  # _last_meta is private to tabpfn_client and may change between versions
            server = getattr(model, "_last_meta", None)
            meta["server"] = server.model_dump() if hasattr(server, "model_dump") else server
            meta["timings"] = model.get_timings() if callable(getattr(model, "get_timings", None)) else None
        except Exception as e:
            meta["meta_error"] = f"{type(e).__name__}: {e}"
    return json.loads(json.dumps(meta, default=str))  # JSON-safe for the audit log and the cache


def _model_id(backend):
    """Package version and model name of a TabPFN backend (logged with every fit; part of the cache key)."""
    pkg = "tabpfn-client" if backend == "tabpfn-client" else "tabpfn"
    return {pkg.replace("-", "_"): importlib.metadata.version(pkg), "model_path": _model_path(backend)}


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def _make_classifier(backend, seed, cat_idx):
    if backend == "standin":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=15, random_state=seed, n_jobs=-1),
        )
    if backend == "logreg":
        return _linear_pipeline(LogisticRegression(max_iter=1000), cat_idx)
    if backend == "tabpfn-local":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(
            device=pick_device(),
            ignore_pretraining_limits=True,
            random_state=seed,
            categorical_features_indices=cat_idx or None,
            model_path=_model_path(backend),
        )
    if backend == "tabpfn-client":
        return _tabpfn_client().TabPFNClassifier(
            model_path=_model_path(backend),
            ignore_pretraining_limits=True,
            random_state=seed,
            categorical_features_indices=cat_idx or None,
        )
    raise ValueError(f"unknown backend '{backend}'")


def _make_regressor(backend, seed, cat_idx):
    if backend == "standin":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=15, random_state=seed, n_jobs=-1),
        )
    if backend == "logreg":
        return _linear_pipeline(LinearRegression(), cat_idx)
    if backend == "tabpfn-local":
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            device=pick_device(),
            ignore_pretraining_limits=True,
            random_state=seed,
            categorical_features_indices=cat_idx or None,
            model_path=_model_path(backend),
        )
    if backend == "tabpfn-client":
        return _tabpfn_client().TabPFNRegressor(
            model_path=_model_path(backend),
            ignore_pretraining_limits=True,
            random_state=seed,
            categorical_features_indices=cat_idx or None,
        )
    raise ValueError(f"unknown backend '{backend}'")


def _linear_pipeline(estimator, cat_idx):
    """``estimator`` on one-hot encoded ``cat_idx`` columns and standardized other columns.

    Category codes are labels, not amounts (tumour size groups sorted as text give "20-50" = 0, "<=20" = 1,
    ">50" = 2), so a linear model must not treat them as numbers. Missing values: most frequent level / median.
    """
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    rest = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return make_pipeline(ColumnTransformer([("cat", cat, list(cat_idx))], remainder=rest), estimator)


def _model_path(backend):
    """TabPFN model passed to the constructor and logged.

    Client: ``HCA_TABPFN_CLIENT_MODEL``, else the pinned ``v3.5_default``. Local: ``HCA_TABPFN_MODEL_PATH``,
    else ``auto`` (the package default; from v3.5 on one checkpoint serves classifier and regressor).
    """
    if backend == "tabpfn-client":
        return os.getenv("HCA_TABPFN_CLIENT_MODEL") or "v3.5_default"
    return os.getenv("HCA_TABPFN_MODEL_PATH") or "auto"


def _tabpfn_client():
    """The ``tabpfn_client`` module, authorized with the API key in ``TABPFN_TOKEN``."""
    if not os.getenv("TABPFN_TOKEN"):
        raise TabPFNSetupError(
            "The TabPFN API client needs your API key: set TABPFN_TOKEN (in .env or the shell).\n"
            "To run TabPFN on this machine instead, use --backend tabpfn-local (needs the weights)."
        )
    import tabpfn_client

    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    return tabpfn_client
