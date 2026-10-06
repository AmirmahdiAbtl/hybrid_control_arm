"""src.models: regressor path, batching, one-hot logreg, FIT_LOG, cache, client model pin; and km_ci.

All offline: cheap backends or fake models.
"""

import importlib.metadata
import json
import socket

import numpy as np
import pytest
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder

from src import models
from src.models import (
    BACKEND_LABELS,
    NON_COMPARABLE,
    TabPFNSetupError,
    cross_source_features,
    fit_predict,
    fit_predict_reg,
)
from src.survival import km_at, km_ci


@pytest.fixture
def fit_log(monkeypatch):
    """A fresh FIT_LOG for this test (keeps fake TabPFN fits out of the session-wide log)."""
    log = []
    monkeypatch.setattr(models, "FIT_LOG", log)
    return log


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise OSError("network access in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    return X, X @ np.array([0.5, -0.2, 0.1]) + 0.3


class Recorder:
    """Fake model: records each predict call; the prediction is the first feature column."""

    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        self.calls.append(("predict_proba", len(X), None))
        return np.column_stack([1 - X[:, 0], X[:, 0]])

    def predict(self, X, **kw):
        self.calls.append(("predict", len(X), kw.get("output_type")))
        return X[:, 0].copy()


def test_logreg_regressor_recovers_linear_function(fit_log):
    rng = np.random.default_rng(0)
    code = rng.integers(0, 3, 300)
    X = np.column_stack([rng.normal(size=300), code, rng.normal(size=300)]).astype(float)
    y = 1 + 2 * X[:, 0] + np.array([0.0, 0.5, -0.3])[code] - X[:, 2]  # non-monotone in the category code
    tests = [X[:50], X[50:120], X[120:]]
    out = fit_predict_reg("logreg", X, y, tests, cat_idx=[1], purpose="linear check")
    assert isinstance(out, list) and [p.shape for p in out] == [(50,), (70,), (180,)]
    pred = np.concatenate(out)
    assert 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2) > 0.999
    assert fit_log[-1]["task"] == "regression" and fit_log[-1]["meta"] is None


def test_standin_regressor_runs(fit_log):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=200)
    out = fit_predict_reg("standin", X[:150], y[:150], [X[150:], X[:10]], seed=0)
    assert [p.shape for p in out] == [(50,), (10,)] and all(np.isfinite(p).all() for p in out)
    assert np.corrcoef(out[0], y[150:])[0, 1] > 0.8


def test_client_regressor_needs_token(monkeypatch, fit_log):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    X, y = _data()
    with pytest.raises(TabPFNSetupError, match="TABPFN_TOKEN"):
        fit_predict_reg("tabpfn-client", X, y, [X[:3]])
    assert fit_log == []


def test_batched_prediction_one_call(monkeypatch, fit_log):
    rec = Recorder()
    monkeypatch.setattr(models, "_make_classifier", lambda *a, **k: rec)
    monkeypatch.setattr(models, "_make_regressor", lambda *a, **k: rec)
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(40, 2))
    tests = [rng.uniform(size=(n, 2)) for n in (5, 3, 7)]

    out = fit_predict("standin", X, np.r_[np.zeros(20), np.ones(20)], tests)
    assert rec.calls == [("predict_proba", 15, None)]
    for p, T in zip(out, tests):
        np.testing.assert_array_equal(p, T[:, 0])

    rec.calls.clear()
    out = fit_predict_reg("tabpfn-local", X, X[:, 1], tests)
    assert rec.calls == [("predict", 15, "mean")]  # TabPFN: mean of the predictive distribution
    for p, T in zip(out, tests):
        np.testing.assert_array_equal(p, T[:, 0])

    rec.calls.clear()
    out = fit_predict_reg("logreg", X, X[:, 1], [tests[0], np.empty((0, 2)), tests[2]])
    assert rec.calls == [("predict", 12, None)] and [len(p) for p in out] == [5, 0, 7]
    np.testing.assert_array_equal(out[2], tests[2][:, 0])
    assert len(fit_log) == 3
    with pytest.raises(ValueError, match="2-D"):  # a 1-D row would be split wrongly after stacking
        fit_predict_reg("logreg", X, X[:, 1], [X[0]])


def test_empty_test_list_and_constant_outcome(monkeypatch, fit_log):
    monkeypatch.setattr(models, "_make_regressor", lambda *a, **k: pytest.fail("no fit expected"))
    monkeypatch.setattr(models, "_make_classifier", lambda *a, **k: pytest.fail("no fit expected"))
    X, y = _data()
    assert fit_predict("logreg", X, (y > y.mean()).astype(int), []) == []
    assert fit_predict_reg("logreg", X, y, []) == []
    const = fit_predict_reg("tabpfn-client", X, np.full(len(X), 0.8), [X[:4], X[:2]])
    assert [p.tolist() for p in const] == [[0.8] * 4, [0.8] * 2]
    assert fit_log == []  # no model was fitted


def test_logreg_classifier_one_hot_recovers_non_monotone_effect(fit_log):
    model = models._make_classifier("logreg", 0, [0])
    _, pipe, cols = model.steps[0][1].transformers[0]
    assert cols == [0] and any(isinstance(step, OneHotEncoder) for _, step in pipe.steps)
    assert isinstance(
        models._make_regressor("logreg", 0, [0]).steps[0][1].transformers[0][1].steps[-1][1], OneHotEncoder
    )

    rng = np.random.default_rng(2)
    code = rng.integers(0, 3, 3000)
    x = rng.normal(size=3000)
    y = (rng.uniform(size=3000) < expit(np.array([0.0, 2.0, -2.0])[code] + 0.3 * x)).astype(int)
    grid = np.column_stack([[0.0, 1.0, 2.0], np.zeros(3)])
    p = fit_predict("logreg", np.column_stack([code, x]).astype(float), y, [grid], cat_idx=[0])[0]
    assert p[1] > p[0] > p[2]  # a numeric (not one-hot) code could only give a monotone order


def test_fit_log_and_cache(monkeypatch, tmp_path, fit_log):
    fits = []

    class FakeTabPFN:
        def fit(self, X, y):
            fits.append(len(y))
            self.m = float(np.mean(y))
            return self

        def predict(self, X, output_type=None):
            assert output_type == "mean"
            return np.full(len(X), self.m)

    real_make = models._make_regressor
    monkeypatch.setattr(models, "_make_regressor", lambda *a, **k: FakeTabPFN())
    monkeypatch.delenv("HCA_TABPFN_MODEL_PATH", raising=False)
    X, y = _data()
    tests = [X[:5], X[5:9]]

    a = fit_predict_reg("tabpfn-local", X, y, tests, seed=1, purpose="m_B fold 1", cache_dir=tmp_path)
    b = fit_predict_reg("tabpfn-local", X, y, tests, seed=1, purpose="m_B fold 1", cache_dir=tmp_path)
    assert fits == [60] and [e["cached"] for e in fit_log] == [False, True]
    assert all(np.array_equal(p, q) for p, q in zip(a, b)) and [len(p) for p in b] == [5, 4]
    first = fit_log[0]
    assert set(first) == {"task", "backend", "purpose", "n_train", "n_test", "seconds", "cached", "meta"}
    assert (first["task"], first["backend"], first["purpose"], first["n_train"], first["n_test"]) == (
        "regression",
        "tabpfn-local",
        "m_B fold 1",
        60,
        9,
    )
    assert first["meta"] == {"tabpfn": importlib.metadata.version("tabpfn"), "model_path": "auto"}
    assert fit_log[1]["meta"] == first["meta"]  # stored metadata comes back on a hit
    assert len(list(tmp_path.glob("*.npz"))) == 1 and not list(tmp_path.glob("*.tmp"))
    json.dumps(fit_log)

    fit_predict_reg("tabpfn-local", X, y + 0.1, tests, seed=1, cache_dir=tmp_path)  # other y -> refit
    fit_predict_reg("tabpfn-local", X, y, tests, seed=2, cache_dir=tmp_path)  # other seed -> refit
    monkeypatch.setenv("HCA_TABPFN_MODEL_PATH", "/models/other.safetensors")  # other model -> refit
    fit_predict_reg("tabpfn-local", X, y, tests, seed=1, cache_dir=tmp_path)
    assert len(fits) == 4 and [e["cached"] for e in fit_log[2:]] == [False, False, False]
    assert fit_log[-1]["meta"]["model_path"] == "/models/other.safetensors"

    monkeypatch.setattr(models, "_make_regressor", real_make)
    fit_predict_reg("logreg", X, y, tests, cache_dir=tmp_path / "cheap")  # cheap backends are never cached
    assert not (tmp_path / "cheap").exists() and fit_log[-1]["cached"] is False and len(fits) == 4


def test_client_retry_for_regressor(monkeypatch, fit_log):
    attempts = []

    class Flaky:
        def fit(self, X, y):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("GCS upload permanently failed ... 500 InternalError")
            return self

        def predict(self, X, output_type=None):
            return np.full(len(X), 0.5)

    monkeypatch.setattr(models, "_make_regressor", lambda *a, **k: Flaky())
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    X, y = _data()
    p = fit_predict_reg("tabpfn-client", X, y, [X[:3]])[0]
    assert len(attempts) == 2 and p.shape == (3,) and len(fit_log) == 1

    class BadInput(Flaky):
        def fit(self, X, y):
            attempts.append(1)
            raise ValueError("bad input")  # not a server error: no retry

    attempts.clear()
    monkeypatch.setattr(models, "_make_regressor", lambda *a, **k: BadInput())
    with pytest.raises(ValueError, match="bad input"):
        fit_predict_reg("tabpfn-client", X, y, [X[:3]])
    assert len(attempts) == 1


def test_client_model_pinned(monkeypatch, no_network):
    import tabpfn_client

    assert "v3.5_default" in tabpfn_client.TabPFNClassifier.list_available_models()
    assert "v3.5_default" in tabpfn_client.TabPFNRegressor.list_available_models()
    monkeypatch.setenv("TABPFN_TOKEN", "test-token")
    monkeypatch.delenv("HCA_TABPFN_CLIENT_MODEL", raising=False)
    tokens = []
    monkeypatch.setattr(tabpfn_client, "set_access_token", tokens.append)

    reg = models._make_regressor("tabpfn-client", 3, [1])  # the real client estimators; nothing is sent
    clf = models._make_classifier("tabpfn-client", 3, [])
    assert isinstance(reg, tabpfn_client.TabPFNRegressor) and isinstance(clf, tabpfn_client.TabPFNClassifier)
    assert reg.model_path == clf.model_path == "v3.5_default" and tokens == ["test-token"] * 2
    assert reg.categorical_features_indices == [1] and clf.categorical_features_indices is None
    assert reg.ignore_pretraining_limits and reg.random_state == 3

    monkeypatch.setenv("HCA_TABPFN_CLIENT_MODEL", "v3_default")
    assert models._make_regressor("tabpfn-client", 0, []).model_path == "v3_default"


def test_local_regressor_construction(monkeypatch, no_network):
    tabpfn = pytest.importorskip("tabpfn")
    monkeypatch.setenv("HCA_DEVICE", "cpu")
    monkeypatch.delenv("HCA_TABPFN_MODEL_PATH", raising=False)
    reg = models._make_regressor("tabpfn-local", 4, [1, 3])  # constructing loads no weights
    assert isinstance(reg, tabpfn.TabPFNRegressor)
    assert (reg.model_path, reg.categorical_features_indices, reg.random_state) == ("auto", [1, 3], 4)
    monkeypatch.setenv("HCA_TABPFN_MODEL_PATH", "/models/tabpfn-v3.5.safetensors")
    assert models._make_regressor("tabpfn-local", 0, []).model_path == "/models/tabpfn-v3.5.safetensors"


def test_client_metadata_logged(monkeypatch, fit_log):
    class ServerMeta:  # pydantic-like
        def model_dump(self):
            return {"billing_model_version": "v3.5-test", "test_set_num_rows": 4}

    class FakeClientReg:
        def fit(self, X, y):
            return self

        def predict(self, X, output_type=None):
            self._last_meta = ServerMeta()
            return np.zeros(len(X))

        def get_timings(self):
            return {"fit": {"fit_s": 0.25}, "predict": None}

    class FakeClientClf(Recorder):
        def predict_proba(self, X):
            self._last_meta = {"billing_model_version": "v3.5-test"}  # tabpfn-client 0.6.1 stores a dict
            return super().predict_proba(X)

    monkeypatch.delenv("HCA_TABPFN_CLIENT_MODEL", raising=False)
    monkeypatch.setattr(models, "_make_regressor", lambda *a, **k: FakeClientReg())
    monkeypatch.setattr(models, "_make_classifier", lambda *a, **k: FakeClientClf())
    X, y = _data()
    fit_predict_reg("tabpfn-client", X, y, [X[:4]], purpose="mu1 fold 0")
    fit_predict("tabpfn-client", X, (y > np.median(y)).astype(int), [X[:4]], purpose="comparability")
    reg, clf = fit_log
    assert reg["meta"]["model_path"] == "v3.5_default"
    assert reg["meta"]["tabpfn_client"] == importlib.metadata.version("tabpfn-client")
    assert reg["meta"]["server"]["billing_model_version"] == "v3.5-test"
    assert reg["meta"]["timings"]["fit"]["fit_s"] == 0.25
    assert clf["task"] == "classification" and clf["meta"]["server"] == {"billing_model_version": "v3.5-test"}
    assert clf["meta"]["timings"] is None
    json.dumps(fit_log)


def test_labels_and_cross_source_features():
    assert BACKEND_LABELS["logreg"] == "logistic/linear regression baseline (NOT TabPFN)"
    assert all("NOT TabPFN" in BACKEND_LABELS[b] for b in ("logreg", "standin"))
    feats = ["age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec"]
    assert NON_COMPARABLE == {"tgrade"}
    assert cross_source_features(feats) == ["age", "menostat", "tsize", "pnodes", "progrec", "estrec"]


def test_km_ci_log_log():
    t = np.r_[np.arange(1, 41), np.arange(1, 41) + 0.5]
    e = np.r_[np.ones(40), np.zeros(40)]
    km = km_at(t, e, 12)
    lo, hi = km_ci(km.surv, km.se)
    assert 0 < lo < km.surv < hi < 1
    c = np.log(-np.log(km.surv))
    d = 1.96 * km.se / (km.surv * abs(np.log(km.surv)))
    assert lo == pytest.approx(np.exp(-np.exp(c + d)), abs=1e-12)
    assert hi == pytest.approx(np.exp(-np.exp(c - d)), abs=1e-12)
    lo, hi = km_ci(0.97, 0.03)  # the plain interval would end above 1
    assert hi < 1 and 0.97 + 1.96 * 0.03 > 1
    assert km_ci(1.0, 0.0) == (1.0, 1.0) and km_ci(0.0, 0.0) == (0.0, 0.0)
    assert all(np.isnan(v) for v in km_ci(np.nan, np.nan))
