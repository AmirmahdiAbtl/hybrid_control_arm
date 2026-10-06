import numpy as np
import pytest
from scipy.special import expit, logit

from src.exploratory import bootstrap_hybrid, dynamic_borrowing, logit_shift, shift_check_text
from src.survival import event_free_labels, km_at


def test_km_hand_computed():
    km = km_at([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], 3.5)
    assert km.surv == pytest.approx(0.8 * 2 / 3)
    # Greenwood: S^2 * (1/(5*4) + 1/(3*2))
    assert km.se == pytest.approx(km.surv * np.sqrt(1 / 20 + 1 / 6))


def test_km_no_events_and_empty():
    assert km_at([1, 2, 3], [0, 0, 0], 2).surv == 1.0
    assert np.isnan(km_at([], [], 2).surv)


def test_labels():
    y = event_free_labels([10, 10, 10, 30], [1, 0, 0, 1], 20)
    assert y[0] == 0 and np.isnan(y[1]) and np.isnan(y[2]) and y[3] == 1


def test_shift_calibrates_to_observed_mean():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.9, 80)
    y = (rng.uniform(size=80) < p - 0.15).astype(float)
    a = logit_shift(p, y)
    assert expit(logit(p) + a).mean() == pytest.approx(y.mean(), abs=1e-6)


def test_shift_clamped_when_labels_equal():
    p = np.full(10, 0.5)
    assert logit_shift(p, np.ones(10)) == pytest.approx(6, abs=1e-3)
    assert logit_shift(p, np.zeros(10)) == pytest.approx(-6, abs=1e-3)


def test_w_zero_when_real_data_agree():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.4, 0.8, 200)
    y = (rng.uniform(size=200) < p).astype(float)
    d = dynamic_borrowing(p, y, p)
    assert d["w"] < 0.2 and abs(d["dynamic"] - d["raw"]) < abs(d["full"] - d["raw"]) + 1e-9
    y_exact = np.r_[np.ones(50), np.zeros(50)]
    d0 = dynamic_borrowing(np.full(100, 0.5), y_exact, np.full(30, 0.5))
    assert d0["w"] == 0 and d0["dynamic"] == pytest.approx(0.5)


def test_w_large_for_large_shift():
    rng = np.random.default_rng(2)
    p = np.full(200, 0.8)
    y = (rng.uniform(size=200) < 0.4).astype(float)
    d = dynamic_borrowing(p, y, np.full(100, 0.8))
    assert d["w"] > 0.8
    assert d["dynamic"] == pytest.approx(d["full"], abs=0.08)


def test_bootstrap_centered_and_text():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.4, 0.9, 60)
    y = (rng.uniform(size=60) < p - 0.1).astype(float)
    pt = rng.uniform(0.4, 0.9, 100)
    b = bootstrap_hybrid(p, y, pt, 500, seed=0)
    d = dynamic_borrowing(p, y, pt)
    assert abs(np.median(b["full"]) - d["full"]) < 0.04
    text, warn = shift_check_text(p, y, d["w"])
    assert "event-free" in text and "dynamic borrowing applied" in text and isinstance(warn, bool)
    assert shift_check_text(np.full(50, 0.9), np.r_[np.ones(25), np.zeros(25)], 1.0)[1] is True
