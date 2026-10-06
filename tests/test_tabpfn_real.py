"""Runs the real tabpfn package. Skipped unless HCA_TABPFN_MODEL_PATH points to a checkpoint."""

import os

import numpy as np
import pytest

from src.exploratory import comparability, run_estimate
from src.models import default_features, fit_predict
from src.survival import DAYS_PER_YEAR

pytestmark = pytest.mark.skipif(
    not os.getenv("HCA_TABPFN_MODEL_PATH"), reason="set HCA_TABPFN_MODEL_PATH to run real TabPFN"
)


def test_fit_predict_real_tabpfn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    p = fit_predict("tabpfn-local", X[:150], y[:150], [X[150:]], seed=0)[0]
    assert p.shape == (50,) and ((p > 0.5) == y[150:]).mean() > 0.8


def test_estimate_real_tabpfn(demo):
    feats = default_features(demo["a_h"], demo["b_h"], {"horTh", "time", "cens"})
    est = run_estimate(demo["hist"], demo["trial"], feats, 2 * DAYS_PER_YEAR, 60, "tabpfn-local", seed=1, n_boot=200)
    assert est["table"].estimate.between(0.4, 0.95).all()
    c = comparability(demo["hist"], demo["trial"], feats, "tabpfn-local")
    assert 0.8 < c["auc"] < 0.97
