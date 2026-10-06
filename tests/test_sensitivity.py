"""Sensitivity analysis of src.causal: E-values, bounding factors, tipping points and equivalence tests.

Offline: closed-form formulas and logistic / linear benchmarks only.
"""

import numpy as np
import pytest
from scipy import stats

from src import causal as C
from src import survival

DAYS = 365.25
BENCHMARKS = [
    "menostat",
    "age band",
    "nodes",
    "tsize",
    "grade",
    "ER>=20",
    "PgR>=20",
    "age x menostat",
    "ER x PgR",
    "nodes x tsize",
]


@pytest.fixture(scope="module")
def r_sample(demo):
    """Scope R at 2 y: covariates, treatment and pseudo-observations of the 605 trial patients."""
    fr = C.gate_frames(demo, demo["df_a"], demo["df_b"])
    T, out = fr["trial_cov"], fr["outcomes"]["trial"]
    fine, coarse = survival.strata_labels("A", T["A"], T["post"])
    theta, _ = survival.pseudo_by_strata(out["time"], out["cens"], 2 * DAYS, fine, coarse)
    return T, theta


# E-values, bounding factors and the robustness rule
def test_evalues():
    assert C.evalue(0.5) == pytest.approx(2 + np.sqrt(2)) and C.evalue(2.0) == pytest.approx(C.evalue(0.5))
    assert C.evalue(1.0) == 1.0
    assert C.evalue_ci(0.6, 1.1) == 1.0
    assert C.evalue(3.9) == pytest.approx(7.26, abs=0.01) and C.evalue(1.8) == pytest.approx(3.0)
    assert C.evalue_ci(0.5, 0.8) == pytest.approx(C.evalue(0.8))  # limit closest to 1
    assert C.evalue_ci(1.2, 2.0) == pytest.approx(C.evalue(1.2))
    assert np.isnan(C.evalue(np.nan)) and np.isnan(C.evalue_ci(np.nan, 1.2))


def test_bounding_factor_and_robustness_rule():
    assert C.bounding_factor(2.0, 1.5) == pytest.approx(1.2)
    assert C.needed_rr_uy(1 / 0.70, 2.33) == pytest.approx(2.11, abs=0.02)
    assert C.needed_rr_uy(2.0, 1.5) == np.inf
    r = C.robustness(0.70, 0.57, 0.86, 1.2, True)
    assert r["applicable"] and r["rr_star_lim"] == pytest.approx(1 / 0.86) and r["robust"] is False  # 1.163 < 1.2
    assert C.robustness(0.70, 0.57, 0.80, 1.2, True)["robust"] is True  # 1.25 > 1.2
    assert C.robustness(0.70, 0.57, 0.80, 1.2, False)["robust"] is False  # sign rule
    assert C.robustness(1.5, 1.3, 1.8, 1.2, True)["rr_star_lim"] == pytest.approx(1.3)
    na = C.robustness(0.80, 0.55, 1.15, 1.2, True)
    assert not na["applicable"] and not na["robust"] and na["rr_star_lim"] == 1.0
    # The RD CI can exclude 0 while the RR CI includes 1: the gate takes G11 applicability from the RD CI, and
    # robust is False whatever bf_ref, so G11 is FAIL there (never NOT_APPLICABLE from `applicable`).
    assert C.robustness(0.85, 0.70, 1.02, 1.0, True)["robust"] is False


# Benchmark bounding factors on the demo
def test_benchmark_variables_definition(r_sample):
    T, _ = r_sample
    bench = C.benchmark_variables(T)
    assert [b[0] for b in bench] == BENCHMARKS
    drop = {name: cols for name, _, cols in bench}
    assert drop == {
        "menostat": [1],
        "age band": [0],
        "nodes": [5],
        "tsize": [2],
        "grade": [3, 4],
        "ER>=20": [7],
        "PgR>=20": [6],
        "age x menostat": [0, 1],
        "ER x PgR": [6, 7],
        "nodes x tsize": [2, 5],
    }
    labels = {name: set(lab) for name, lab, _ in bench}
    assert labels["age band"] == {"<50", "50-59", ">=60"} and labels["nodes"] == {"1-3", "4-9", ">=10"}
    assert labels["tsize"] == {"<=20", "21-50", ">50"} and labels["grade"] == {"I", "II", "III"}
    assert labels["age x menostat"] <= {f"{m}|{a}" for m in ("pre", "post") for a in ("<45", "45-59", ">=60")}
    assert all(len(lab) == len(T) for _, lab, _ in bench)


def test_benchmark_bounding_factors_on_the_demo(r_sample):
    T, theta = r_sample
    bm = C.benchmark_bounding_factors(theta, T["A"].to_numpy(), C.design_trial(T), C.benchmark_variables(T), 20)
    assert bm.name.tolist() == BENCHMARKS and (bm.bf >= 1).all() and (bm.rr_au >= 1).all() and (bm.rr_uy >= 1).all()
    block = bm.set_index("name").loc["age x menostat"]
    assert 1.1 <= block.bf <= 1.4 and round(block.bf, 2) == 1.25
    assert block.levels_kept.split(", ") == ["post|45-59", "post|>=60", "pre|45-59", "pre|<45"]
    assert "post|<45 (1/1)" in block.levels_dropped and "pre|>=60 (0/1)" in block.levels_dropped
    assert bm.iloc[:7].bf.max() <= 1.12  # singles
    assert round(max(bm.bf.max(), C.bounding_factor(2.0, 1.5)), 2) == 1.25  # bf_ref


def test_benchmark_with_a_single_kept_level_is_not_assessable():
    rng = np.random.default_rng(0)
    A = np.r_[np.ones(50), np.zeros(50)].astype(int)
    labels = np.r_[np.repeat("x", 90), np.repeat("y", 10)]  # "y": 0 treated -> dropped
    bm = C.benchmark_bounding_factors(rng.uniform(size=100), A, rng.normal(size=(100, 8)), [("u", labels, [0])], 20)
    assert np.isnan(bm.bf.iloc[0]) and bm.levels_kept.iloc[0] == "x" and bm.levels_dropped.iloc[0] == "y (0/10)"


# Borrowing pre-flight, TOST and tipping points
def test_min_detectable_margin():
    assert C.min_detectable_margin(0.437, 60) == pytest.approx(0.0928, abs=1e-3)
    assert C.min_detectable_margin(0.395, 24.5) == pytest.approx(0.131, abs=1e-3)  # demo pre-flight > 0.05


def test_tost():
    lo, hi, passed = C.tost(0.0, 0.01, 0.05)
    assert passed is True and lo == pytest.approx(-stats.norm.ppf(0.95) * 0.01) and hi == pytest.approx(-lo)
    assert C.tost(0.04, 0.02, 0.05)[2] is False


def test_tipping_point():
    point, ci = C.tipping_point(0.03, 0.055)
    assert point == 0.03 and ci == pytest.approx(0.03 - 1.96 * 0.055)


# Effect summaries (risk ratio, E-values, t intervals)
def test_effect_summary_two_independent_means():
    rng = np.random.default_rng(1)
    y1, y0 = rng.binomial(1, 0.8, 200).astype(float), rng.binomial(1, 0.7, 300).astype(float)
    s1, i1 = C.mean_with_if(y1)
    s0, i0 = C.mean_with_if(y0)
    if1, if0 = np.r_[i1, np.zeros(300)], np.r_[np.zeros(200), i0]
    r = C.effect_summary(s1, s0, if1, if0)
    assert r["est"] == pytest.approx(y1.mean() - y0.mean())
    assert r["se"] == pytest.approx(np.sqrt(y1.var() / 200 + y0.var() / 300))
    assert r["se_s1"] == pytest.approx(np.sqrt(y1.var() / 200)) and r["df"] is None
    assert r["rr"] == pytest.approx((1 - s1) / (1 - s0))
    se_log = np.sqrt(np.sum(i1**2) / (1 - s1) ** 2 + np.sum(i0**2) / (1 - s0) ** 2)
    assert r["rr_lo"] == pytest.approx(r["rr"] * np.exp(-1.96 * se_log), rel=1e-3)
    assert r["evalue"] == pytest.approx(C.evalue(r["rr"])) and r["evalue_ci"] == pytest.approx(
        C.evalue_ci(r["rr_lo"], r["rr_hi"])
    )
    t = C.effect_summary(s1, s0, if1, if0, df=10)
    assert t["q"] == pytest.approx(stats.t.ppf(0.975, 10)) and t["hi"] - t["lo"] > r["hi"] - r["lo"]


def test_g11_not_applicable_on_the_demo(r_sample):
    """The 2-y R ATT CI includes 0 and its RR CI includes 1, so the benchmark rule does not apply."""
    T, theta = r_sample
    A, D = T["A"].to_numpy(), C.design_trial(T)
    e = C.ps_logistic(D, A)
    mu0 = C.crossfit_tlearner(
        D, theta, A, C.make_folds(survival.strata_labels("A", A, T["post"])[0]), "logreg", 0, [], arms=(0,)
    )[0]
    r = C.aipw(theta, A, e, mu0, mu0, "ATT")
    s = C.effect_summary(r["s1"], r["s0"], r["if_s1"], r["if_s0"])
    assert s["lo"] < 0 < s["hi"] and s["rr_lo"] < 1 < s["rr_hi"] and s["evalue_ci"] == 1.0
    assert not C.robustness(s["rr"], s["rr_lo"], s["rr_hi"], 1.25, True)["applicable"]
