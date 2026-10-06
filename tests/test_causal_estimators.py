"""Weights, nuisance models, estimators, subgroup effects and operating checks of src.causal.

Offline: logistic / linear models or the 'logreg' backend only, so every pinned number is a cheap-model value.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression

from src import causal as C
from src import models, survival
from src.causal import weights
from src.rules import evaluate_rule

DAYS = 365.25
H2 = 2 * DAYS
SUBGROUPS = [
    ("ER", "estrec >= 10", "estrec < 10"),
    ("menostat", "menostat == 'Post'", "menostat == 'Pre'"),
    ("nodes", "pnodes >= 4", "pnodes <= 3"),
]
GATES_H = {"G4H_imbalance_max": 1e-6, "G4H_ess_min_abs": 20.0, "G4H_ess_min_frac": 1 / 3, "G4H_weight_share_max": 0.15}


@pytest.fixture(scope="module")
def frames(demo):
    return C.gate_frames(demo, demo["df_a"], demo["df_b"])


@pytest.fixture(scope="module")
def rsample(frames):
    """Scope R at 2 y: pseudo-observations, logistic PS and cross-fitted D_t-linear outcome models."""
    T, out = frames["trial_cov"], frames["outcomes"]["trial"]
    A = T["A"].to_numpy()
    fine, coarse = survival.strata_labels("A", A, T["post"])
    theta, _ = survival.pseudo_by_strata(out["time"], out["cens"], H2, fine, coarse)
    D = C.design_trial(T)
    folds = C.make_folds(fine)
    mu = C.crossfit_tlearner(D, theta, A, folds, "logreg", 0, [], purpose="R_lin")  # OLS on D_t
    return {
        "T": T,
        "A": A,
        "theta": theta,
        "D": D,
        "e": C.ps_logistic(D, A),
        "folds": folds,
        "mu0": mu[0],
        "mu1": mu[1],
        "time": out["time"].to_numpy(float),
        "cens": out["cens"].to_numpy(float),
    }


@pytest.fixture(scope="module")
def prognostic(frames):
    """Rotterdam pseudo-observations at 2 y and the cheap prognostic scores ('logreg' path + linear path)."""
    B, out_B, T = frames["hist_cov"], frames["outcomes"]["hist"], frames["trial_cov"]
    fine, coarse = survival.strata_labels("B", B["A"], B["post"])
    theta, _ = survival.pseudo_by_strata(out_B["time"], out_B["cens"], H2, fine, coarse)
    ps = C.prognostic_score(
        C.features_source(B),
        C.design_source(B),
        theta,
        C.features_source(T),
        C.design_source(T),
        C.make_folds(B["post"]),
        "logreg",
        0,
        C.SOURCE_CAT_IDX,
    )
    return {"theta": theta, **ps}


def _hybrid_sample(rsample, kept):
    """Pseudo-observations of treated + kept controls (small strata merged), as the gated run computes them."""
    A, T = rsample["A"], rsample["T"]
    trt = np.flatnonzero(A == 1)
    idx = np.r_[trt, kept]
    fine, coarse = survival.strata_labels("A", A[idx], T["post"].to_numpy()[idx])
    theta, used = survival.pseudo_by_strata(rsample["time"][idx], rsample["cens"][idx], H2, fine, coarse)
    return trt, theta[: len(trt)], theta[len(trt) :], used


def _robust_hybrid(rsample, m, kept):
    T, D = rsample["T"], rsample["D"]
    trt, th_T, th_K, _ = _hybrid_sample(rsample, kept)
    return C.robust_hybrid_att(
        th_T,
        th_K,
        D[trt],
        D[kept],
        C.recalibration_design(T.iloc[trt], m[trt]),
        C.recalibration_design(T.iloc[kept], m[kept]),
        C.make_folds(T["menostat"].to_numpy()[kept]),
    )


def _frozen_draw(rsample):
    """Kept controls of the frozen draw, as the gated run draws them: sorted positions, seed 0, 60 controls."""
    return np.sort(np.random.default_rng(0).choice(np.flatnonzero(rsample["A"] == 0), 60, replace=False))


# Simulation with known ATE / ATT / ATC / ATO
def _sim_nuisances(x):
    e = expit(-0.3 + 1.0 * x)
    return e, expit(0.4 - 0.5 * x), expit(0.7 + 0.4 * x)  # e, mu0, mu1 (effect heterogeneous in x)


def _sim_truths():
    """Population ATE / ATT / ATC / ATO by Gauss-Hermite quadrature over x ~ N(0, 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(120)
    w = w / w.sum()
    e, mu0, mu1 = _sim_nuisances(x)
    tau = mu1 - mu0
    return {
        "ATE": np.sum(w * tau),
        "ATT": np.sum(w * e * tau) / np.sum(w * e),
        "ATC": np.sum(w * (1 - e) * tau) / np.sum(w * (1 - e)),
        "ATO": np.sum(w * e * (1 - e) * tau) / np.sum(w * e * (1 - e)),
    }


def _simulate(rng, n):
    x = rng.normal(size=n)
    e, mu0, mu1 = _sim_nuisances(x)
    A = rng.binomial(1, e)
    return x, e, mu0, mu1, A, rng.binomial(1, np.where(A == 1, mu1, mu0)).astype(float)


# AIPW
def test_aipw_with_true_nuisances_is_centred_on_four_distinct_estimands():
    truth = _sim_truths()
    vals = list(truth.values())
    assert min(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1 :]) > 0.01
    _, e, mu0, mu1, A, Y = _simulate(np.random.default_rng(1), 3000)
    for k, v in truth.items():
        r = C.aipw(Y, A, e, mu0, mu1, k)
        assert abs(r["est"] - v) < 4 * r["se"]
        assert r["lo"] < r["est"] < r["hi"] and len(r["if_s1"]) == len(r["if_s0"]) == 3000
        assert r["est"] == pytest.approx(r["s1"] - r["s0"])


def test_aipw_coverage_with_fitted_logistic_ps_and_linear_outcomes():
    truth = _sim_truths()
    rng = np.random.default_rng(2024)
    hits = {k: 0 for k in truth}
    for _ in range(150):
        x, _, _, _, A, Y = _simulate(rng, 800)
        X = x[:, None]
        e = C.ps_logistic(X, A)
        m0 = LinearRegression().fit(X[A == 0], Y[A == 0]).predict(X)
        m1 = LinearRegression().fit(X[A == 1], Y[A == 1]).predict(X)
        for k in truth:
            r = C.aipw(Y, A, e, m0, m1, k)
            hits[k] += r["lo"] <= truth[k] <= r["hi"]
    for k, h in hits.items():
        assert 0.89 <= h / 150 <= 0.99, (k, h / 150)


@pytest.mark.parametrize("seed", range(5))
def test_ato_is_orthogonal_in_the_propensity(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=200_000)
    e = expit(-0.8 + 0.6 * x)
    mu0 = np.clip(0.6 + 0.1 * x, 0.05, 0.95)
    mu1 = np.clip(mu0 + 0.1 + 0.1 * x, 0, 1)
    tau = mu1 - mu0
    A = rng.binomial(1, e)
    theta = np.where(A == 1, mu1, mu0)  # noise-free: every residual term is 0
    h = e * (1 - e)
    psi = np.sum(h * tau) / h.sum()
    e_hat = expit(logit(e) + 0.5 * np.sign(tau - psi))  # propensity error correlated with the effect
    h_hat = e_hat * (1 - e_hat)
    bias_memo = np.sum(h_hat * tau) / h_hat.sum() - psi
    bias_orth = C.aipw(theta, A, e_hat, mu0, mu1, "ATO")["est"] - psi
    assert bias_memo > 0.005 and abs(bias_orth) < 0.25 * bias_memo


def test_ato_weights_balance_the_propensity_covariates_exactly(rsample):
    w = C.balancing_weights(rsample["e"], rsample["A"], "ATO")
    smd = C.smd_table(rsample["D"], rsample["A"], w, C.DESIGN_TRIAL_NAMES)
    assert smd.smd_after.abs().max() < 1e-6 and smd.smd_before.abs().max() > 0.5


def test_aipw_rejects_unknown_estimands():
    with pytest.raises(ValueError):
        C.aipw([1.0], [1], [0.5], [0.5], [0.5], "ATX")
    with pytest.raises(ValueError):
        C.balancing_weights([0.5], [1], "ATX")


# Entropy balancing and the doubly robust counterfactual mean
def test_entropy_balance_inside_outside_and_at_the_mean():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 3))
    target = X.mean(0) + 0.3 * X.std(0)
    inside = C.entropy_balance(X, target)
    assert inside["converged"] and inside["imbalance"] < 1e-8
    np.testing.assert_allclose(inside["w"] @ X, target, atol=1e-8)
    assert inside["w"].sum() == pytest.approx(1.0) and inside["ess"] < 200
    outside = target.copy()
    outside[0] = X[:, 0].max() + 0.1
    out = C.entropy_balance(X, outside)
    assert not out["converged"] and np.isfinite(out["w"]).all() and out["w"].sum() == pytest.approx(1.0)
    eq = C.entropy_balance(X, X.mean(0))
    np.testing.assert_allclose(eq["w"], 1 / 200, atol=1e-10)
    assert eq["ess"] == pytest.approx(200.0)


def test_entropy_balance_constant_column_and_joint_hull_failure(rsample):
    rng = np.random.default_rng(8)
    X = np.column_stack([rng.normal(size=50), np.ones(50)])
    assert C.entropy_balance(X, [0.1, 1.0])["converged"]
    assert not C.entropy_balance(X, [0.1, 0.5])["converged"]
    # Draw 90 of seed 0 (as draw_stability draws it): every coordinate is in range but the target lies outside
    # the joint hull, so the result must be finite and not converged.
    D, A = rsample["D"], rsample["A"]
    rng = np.random.default_rng(0)
    for _ in range(91):
        kept = np.sort(rng.choice(np.flatnonzero(A == 0), 60, replace=False))
    eb = C.entropy_balance(D[kept], D[A == 1].mean(0))
    assert not eb["converged"] and np.isfinite(eb["imbalance"]) and np.isfinite(eb["w"]).all()


def test_missing_covariates_stop_the_models_instead_of_dropping_the_column(frames, rsample):
    """No imputation is pre-specified, so the logistic models and entropy balancing must refuse a missing value.

    Dropping the column instead (its sd is NaN) would lose a confounder, and a NaN in the first kept row would let
    entropy balancing report converged=True.
    """
    T, A, D = rsample["T"], rsample["A"], rsample["D"]
    ctrl, trt = np.flatnonzero(A == 0), np.flatnonzero(A == 1)
    Dn = D.copy()
    Dn[ctrl[10], 5] = np.nan  # one missing log1p(pnodes)
    with pytest.raises(ValueError, match="missing or non-finite"):
        C.ps_logistic(Dn, A)
    with pytest.raises(ValueError, match="missing or non-finite"):
        weights._logistic_fit_predict(D, A, [Dn])  # a NaN in a prediction matrix only
    B = frames["hist_cov"]
    Ds_T, Ds_B = C.design_source(T.iloc[trt]), C.design_source(B)
    Ds_B[3, 4] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        C.source_support(Ds_T, Ds_B, np.repeat("all", len(trt)), np.repeat("all", len(B)))
    kept = _frozen_draw(rsample)
    target = D[trt].mean(0)
    for row in (0, 5):  # a NaN in row 0 must not pass as converged
        Dk = D[kept].copy()
        Dk[row, 5] = np.nan
        with pytest.raises(ValueError, match="missing or non-finite"):
            C.entropy_balance(Dk, target)
    Dk = D[kept].copy()
    Dk[2, 0] = np.inf
    for X, tgt in ((Dk, target), (D[kept], np.r_[target[:7], np.nan])):
        with pytest.raises(ValueError, match="missing or non-finite"):
            C.entropy_balance(X, tgt)
    assert C.entropy_balance(D[kept], target)["converged"]  # complete data: unchanged


def test_dr_counterfactual_mean_reduces_to_the_comparison_mean():
    rng = np.random.default_rng(7)
    n = 50
    theta = rng.uniform(size=n)
    M, if_t, if_c = C.dr_counterfactual_mean(np.zeros(30), theta, np.zeros(n), np.full(n, 3.0))
    assert M == pytest.approx(theta.mean()) and np.all(if_t == 0)
    se = np.sqrt(np.sum(if_t**2) + np.sum(if_c**2))
    assert se == pytest.approx(theta.std(ddof=1) / np.sqrt(n) * np.sqrt((n - 1) / n))


def test_mean_with_if():
    m, inf = C.mean_with_if([1.0, 2.0, 3.0, 6.0])
    assert m == 3.0 and np.allclose(inf, [-0.5, -0.25, 0.0, 0.75])


# Robust hybrid on the demo (frozen draw)
def test_robust_hybrid_on_the_demo(rsample, prognostic):
    kept = _frozen_draw(rsample)  # exact values are pinned in the next test
    h = _robust_hybrid(rsample, prognostic["m_T"], kept)
    assert h["converged"] and h["imbalance"] < 1e-6 and 22 <= h["ess"] <= 27 and h["max_share"] < 0.15
    assert -0.05 <= h["est"] <= 0.15 and 0.07 <= h["se"] <= 0.13
    assert h["df"] == pytest.approx(h["ess"] - 1) and h["lo"] < 0 < h["hi"]
    D = rsample["D"]
    np.testing.assert_allclose(h["w"] @ D[kept], D[rsample["A"] == 1].mean(0), rtol=1e-6, atol=1e-6)
    assert np.all((h["mu0_T"] >= 0) & (h["mu0_T"] <= 1)) and h["rr"] < 1 and h["evalue_ci"] == 1.0


def test_robust_hybrid_reference_numbers_on_the_frozen_draw(rsample, prognostic):
    """Pinned cheap-model values on the sorted frozen draw.

    The estimates depend on the order of the kept controls through the recalibration folds; ESS, max share,
    merged strata, Hajek-EB and beta do not.
    """
    kept = _frozen_draw(rsample)
    h = _robust_hybrid(rsample, prognostic["m_T"], kept)  # 'logreg' prognostic score
    assert (round(h["est"] * 100, 2), round(h["se"] * 100, 2)) == (5.42, 10.80)
    lin = _robust_hybrid(rsample, prognostic["m_T_lin"], kept)  # linear D_s prognostic score
    assert (round(lin["est"] * 100, 2), round(lin["se"] * 100, 2)) == (4.73, 10.81)
    assert (round(lin["lo"] * 100, 1), round(lin["hi"] * 100, 1)) == (-17.6, 27.1)
    assert h["ess"] == pytest.approx(24.5, abs=0.05) and h["max_share"] == pytest.approx(0.094, abs=0.001)
    _, th_T, th_K, used = _hybrid_sample(rsample, kept)
    assert set(used[len(th_T) :]) == {"A|0"}  # 22 postmenopausal kept -> merged
    assert round(C.hajek_eb_att(th_T, th_K, h["w"])["est"] * 100, 2) == 5.02
    beta, se, lo, hi = C.negative_control_beta(th_K, prognostic["m_T_lin"][kept], h["w"])
    assert (round(beta * 100, 2), round(se * 100, 2)) == (-2.79, 8.75) and lo < beta < hi


def test_robust_hybrid_atc_on_the_demo(rsample):
    T, A = rsample["T"], rsample["A"]
    kept = _frozen_draw(rsample)
    trt, th_T, th_K, _ = _hybrid_sample(rsample, kept)
    mu = C.crossfit_tlearner(
        C.features_trial(T), rsample["theta"], A, rsample["folds"], "logreg", 0, C.TRIAL_CAT_IDX, arms=(1,)
    )
    r = C.robust_hybrid_atc(th_T, th_K, rsample["D"][trt], rsample["D"][kept], mu[1][trt], mu[1][kept])
    assert r["converged"] and np.isfinite(r["est"]) and r["se"] > 0
    assert r["df"] == pytest.approx(min(r["ess"], 60) - 1) and r["w"].sum() == pytest.approx(1.0)


def test_negative_control_beta_on_all_controls(rsample, prognostic):
    A, e = rsample["A"], rsample["e"]
    ctrl = A == 0
    beta, se, lo, hi = C.negative_control_beta(rsample["theta"][ctrl], prognostic["m_T_lin"][ctrl], (e / (1 - e))[ctrl])
    assert (round(beta * 100, 1), round(se * 100, 1), round(lo * 100, 1), round(hi * 100, 1)) == (-3.7, 2.6, -8.0, 0.6)


def test_transport_e_row_on_the_demo(frames, rsample, prognostic):
    T, B, A = rsample["T"], frames["hist_cov"], rsample["A"]
    trt = np.flatnonzero(A == 1)
    E = C.transport_att(
        rsample["theta"][trt],
        prognostic["m_T_lin"][trt],
        prognostic["theta"],
        prognostic["oof_lin"],
        C.design_source(T.iloc[trt]),
        C.design_source(B),
    )
    assert round(E["est"] * 100, 1) == 3.0 and round(E["se"] * 100, 1) == 5.5
    assert round(E["ess_B"]) == 96 and round(E["gformula"] * 100, 1) == 1.9
    assert E["tipping"] == (E["est"], pytest.approx(E["est"] - 1.96 * E["se"]))
    pre_T, pre_B = trt[T["post"].to_numpy()[trt] == 0], np.flatnonzero(B["post"].to_numpy() == 0)
    E_pre = C.transport_att(
        rsample["theta"][pre_T],
        prognostic["m_T_lin"][pre_T],
        prognostic["theta"][pre_B],
        prognostic["oof_lin"][pre_B],
        C.design_source(T.iloc[pre_T]),
        C.design_source(B.iloc[pre_B]),
    )
    assert np.isfinite(E_pre["est"]) and E_pre["ess_B"] > E["ess_B"]


# Scope R reference numbers and diagnostics
def test_scope_r_design_diagnostics_on_the_demo(rsample):
    e, A = rsample["e"], rsample["A"]
    assert (round(e[A == 1].min(), 3), round(e[A == 1].max(), 3)) == (0.123, 0.567)
    assert (round(e[A == 0].min(), 3), round(e[A == 0].max(), 3)) == (0.109, 0.592)
    Dx = C.design_trial_ext(rsample["T"])
    expect = {
        "ATT": ("ess_control", 289, 0.034),
        "ATE": ("ess_treated", 170, 0.015),
        "ATC": ("ess_treated", 139, 0.023),
        "ATO": ("ess_treated", 197, 0.017),
    }
    for est, (key, ess, smd) in expect.items():
        ov = C.overlap_summary(e, A, est)
        assert ov["share_outside"] == 0 and round(ov[key]) == ess
        tab = C.smd_table(Dx, A, C.balancing_weights(e, A, est), C.DESIGN_TRIAL_EXT_NAMES)
        assert round(tab.smd_after.abs().max(), 3) == smd and round(tab.smd_before.abs().max(), 2) == 0.60
        vr = tab.set_index("name")["vr_after"]
        assert np.isnan(vr["post"]) and np.isfinite(vr["age"])  # variance ratio: continuous only
    assert round(C.overlap_summary(e, A, "ATO")["ess_control"]) == 340


def test_overlap_summary_counts_only_the_target_units():
    e, A = np.array([0.97, 0.5, 0.02, 0.5]), np.array([1, 1, 0, 0])
    assert C.overlap_summary(e, A, "ATT")["share_outside"] == 0.5
    assert C.overlap_summary(e, A, "ATC")["share_outside"] == 0.5
    assert C.overlap_summary(e, A, "ATE")["share_outside"] == 0.5
    assert C.overlap_summary(np.array([0.97, 0.5, 0.5, 0.5]), A, "ATC")["share_outside"] == 0.0
    assert not C.overlap_summary(e, A, "ATO")["gated"] and C.overlap_summary(e, A, "ATT")["gated"]


def test_smd_table_columns_without_spread():
    A = np.r_[np.ones(3), np.zeros(4)]
    w = np.r_[np.ones(3), 0.1, 0.7, 0.13, 0.07]
    x_e = np.r_[np.full(3, 0.1), 0.0, 0.1, 0.2, 0.3]
    X = np.column_stack([A, 1 - A, np.full(7, 0.1), np.full(7, 3.0), x_e])
    tab = C.smd_table(X, A, w, list("abcde")).set_index("name")
    assert tab.loc["a", "smd_before"] == tab.loc["a", "smd_after"] == np.inf  # complete imbalance
    assert tab.loc["b", "smd_before"] == tab.loc["b", "smd_after"] == -np.inf
    # same constant in both arms: 0, although the means differ by rounding (0.1 x 3 vs x 4; weighted 3.0 is 3 - 4e-16)
    assert (tab.loc[["c", "d"], ["smd_before", "smd_after"]] == 0.0).all().all()
    sd = np.sqrt((np.var(x_e[:3], ddof=1) + np.var(x_e[3:], ddof=1)) / 2)
    assert tab.loc["e", "smd_before"] == pytest.approx((x_e[:3].mean() - x_e[3:].mean()) / sd)
    assert tab.loc["e", "smd_after"] == pytest.approx((0.1 - np.average(x_e[3:], weights=w[3:])) / sd)


def test_scope_r_reference_numbers(rsample):
    """Pinned cheap-model values of the scope R estimators on the demo."""
    th, A, e, T = rsample["theta"], rsample["A"], rsample["e"], rsample["T"]
    lin = {k: C.aipw(th, A, e, rsample["mu0"], rsample["mu1"], k) for k in ("ATT", "ATE", "ATC", "ATO")}
    pinned_lin = {"ATT": 5.50, "ATE": 4.81, "ATC": 4.46, "ATO": 5.33}
    assert {k: round(v["est"] * 100, 2) for k, v in lin.items()} == pinned_lin
    assert round(lin["ATT"]["se"] * 100, 2) == 3.78
    h, tau = e * (1 - e), rsample["mu1"] - rsample["mu0"]
    memo = np.sum(h * tau + A * (1 - e) * (th - rsample["mu1"]) - (1 - A) * e * (th - rsample["mu0"])) / h.sum()
    assert round(memo * 100, 2) == 5.06
    att = lin["ATT"]
    s = C.effect_summary(att["s1"], att["s0"], att["if_s1"], att["if_s0"])
    assert (round(s["s1"], 3), round(s["s0"], 3), round(s["rr"], 2)) == (0.777, 0.722, 0.80)
    mu = C.crossfit_tlearner(C.features_trial(T), th, A, rsample["folds"], "logreg", 0, C.TRIAL_CAT_IDX)
    lr = {k: C.aipw(th, A, e, mu[0], mu[1], k) for k in ("ATT", "ATE", "ATC", "ATO")}
    pinned_logreg = {"ATT": 5.03, "ATE": 4.61, "ATC": 4.40, "ATO": 5.19}
    assert {k: round(v["est"] * 100, 2) for k, v in lr.items()} == pinned_logreg
    assert round(lr["ATT"]["se"] * 100, 2) == 3.89


def test_agreement_diagnostics(rsample):
    th, A, e, mu0 = rsample["theta"], rsample["A"], rsample["e"], rsample["mu0"]
    ag = C.agreement(th, A, e, mu0)
    assert ag["aipw"] == pytest.approx(C.aipw(th, A, e, mu0, mu0, "ATT")["est"])
    assert ag["ipw"] == pytest.approx(C.hajek_att(th, A, e))
    assert ag["gformula"] == pytest.approx(C.gformula_att(th, A, mu0))
    assert ag["diff_in_se"]["gformula"] == pytest.approx(ag["diff_gformula"] / ag["se"])
    assert ag["diff_in_se"]["ipw"] == pytest.approx(ag["diff_ipw"] / ag["se"])
    assert ag["se_diff_gformula"] > 0 and ag["se_diff_ipw"] > 0
    pop = C.ato_population(rsample["D"], e, C.DESIGN_TRIAL_NAMES)
    assert list(pop.columns) == ["name", "mean_all", "mean_ato"] and len(pop) == 8


# Nuisance models
def test_crossfit_tlearner_is_out_of_fold_batched_and_clipped(monkeypatch):
    calls = []

    def fake(backend, X_train, y_train, X_tests, seed=0, cat_idx=None, purpose="", cache_dir=None):
        calls.append((purpose, X_train[:, 0].astype(int), [X[:, 0].astype(int) for X in X_tests], cache_dir))
        return [np.full(len(X), 1.5) for X in X_tests]

    monkeypatch.setattr(models, "fit_predict_reg", fake)
    n = 50
    F = np.column_stack([np.arange(n), np.zeros(n)]).astype(float)  # column 0 = unit id
    A = np.r_[np.zeros(25), np.ones(25)]
    folds = np.arange(n) % 5
    mu = C.crossfit_tlearner(F, np.linspace(0, 1, n), A, folds, "tabpfn-local", purpose="R_2y", cache_dir="cache")
    assert len(calls) == 10 and calls[0][0] == "R_2y_mu0_fold0" and calls[-1][0] == "R_2y_mu1_fold4"
    for purpose, train, tests, cache_dir in calls:
        assert len(tests) == 1 and cache_dir == "cache"  # one batched prediction per fit
        k, a = int(purpose[-1]), int(purpose.split("_mu")[1][0])
        assert np.all(folds[train] != k) and np.all(A[train] == a) and np.all(folds[tests[0]] == k)
    assert set(mu) == {0, 1} and np.all(mu[0] == 1.0) and np.all(mu[1] == 1.0)


def test_prognostic_score_fits_and_metrics(frames, prognostic, monkeypatch):
    log = []
    monkeypatch.setattr(models, "FIT_LOG", log)
    B, T = frames["hist_cov"], frames["trial_cov"]
    ps = C.prognostic_score(
        C.features_source(B),
        C.design_source(B),
        prognostic["theta"],
        C.features_source(T),
        C.design_source(T),
        C.make_folds(B["post"]),
        "logreg",
        0,
        C.SOURCE_CAT_IDX,
        purpose="mB_2y",
    )
    assert [x["purpose"] for x in log] == [f"mB_2y_cv_fold{k}" for k in range(5)] + ["mB_2y_full"]
    assert all(x["backend"] == "logreg" and x["task"] == "regression" for x in log)
    for k in ("oof", "oof_lin", "m_T", "m_T_lin"):
        assert np.all((ps[k] >= 0) & (ps[k] <= 1))
    assert len(ps["m_T"]) == len(T) and len(ps["oof"]) == len(B)
    assert round(ps["mse"], 3) == 0.158 and round(ps["mse_null"], 3) == 0.174 and ps["mse_lin"] < ps["mse_null"]
    assert round(ps["sigma_lin"], 3) == 0.395 and round(ps["sigma"], 3) == 0.397 and ps["se_diff"] > 0
    lo, hi = ps["cal_slope_ci"]
    assert lo < ps["cal_slope"] < hi and ps["cal_int_ci"][0] < ps["cal_int"] < ps["cal_int_ci"][1]


def test_outcome_diagnostics_per_arm(rsample):
    th, A, folds = rsample["theta"], rsample["A"], rsample["folds"]
    mu = C.crossfit_tlearner(C.features_trial(rsample["T"]), th, A, folds, "logreg", 0, C.TRIAL_CAT_IDX)
    lin = {0: rsample["mu0"], 1: rsample["mu1"]}
    tab = C.outcome_diagnostics(th, A, folds, mu, lin)
    assert tab.arm.tolist() == [0, 1] and tab.n.tolist() == [400, 205]
    assert (tab.mse_null > 0).all() and np.isfinite(tab[["cv_r2", "cv_r2_lin", "cal_slope"]].to_numpy()).all()


# Subgroup effects (CATE)
def test_subgroup_att_partition_identity(rsample):
    th, A, e, mu0, T = rsample["theta"], rsample["A"], rsample["e"], rsample["mu0"], rsample["T"]
    att = C.aipw(th, A, e, mu0, mu0, "ATT")["est"]
    for _, rule, _ in SUBGROUPS:
        g = evaluate_rule(rule, T).to_numpy()
        a, b = C.subgroup_att(th, A, e, mu0, g), C.subgroup_att(th, A, e, mu0, ~g)
        assert ((A * g).sum() * a["est"] + (A * ~g).sum() * b["est"]) / A.sum() == pytest.approx(att, abs=1e-12)
        assert np.all(a["if"][~g] == 0) and a["se"] > 0
        c = C.contrast(a, b)
        assert c["est"] == pytest.approx(a["est"] - b["est"]) and c["se"] == pytest.approx(np.hypot(a["se"], b["se"]))
        assert 0 <= c["p"] <= 1


def test_subgroup_ate_is_the_mean_score(rsample):
    phi = C.ate_scores(rsample["theta"], rsample["A"], rsample["e"], rsample["mu0"], rsample["mu1"])
    assert phi.mean() == pytest.approx(
        C.aipw(rsample["theta"], rsample["A"], rsample["e"], rsample["mu0"], rsample["mu1"], "ATE")["est"]
    )
    g = evaluate_rule("menostat == 'Post'", rsample["T"]).to_numpy()
    r = C.subgroup_ate(phi, g)
    assert r["est"] == pytest.approx(phi[g].mean()) and r["se"] == pytest.approx(phi[g].std(ddof=0) / np.sqrt(g.sum()))


def test_simultaneous_band_between_pointwise_and_bonferroni(rsample):
    phi = C.ate_scores(rsample["theta"], rsample["A"], rsample["e"], rsample["mu0"], rsample["mu1"])
    groups = [
        C.subgroup_ate(phi, evaluate_rule(rule, rsample["T"]).to_numpy())
        for _, r1, r0 in SUBGROUPS
        for rule in (r1, r0)
    ]
    c = C.simultaneous_band([g["if"] for g in groups], [g["se"] for g in groups], draws=2000, seed=0)
    bonferroni = stats.norm.ppf(1 - 0.025 / 6)
    assert 1.96 < c < bonferroni and round(bonferroni, 2) == 2.64  # strictly inside: no clipping acted
    one = C.simultaneous_band([groups[0]["if"]], [groups[0]["se"]], draws=50, seed=1)
    assert one == pytest.approx(stats.norm.ppf(0.975))  # one test: c is the pointwise value


def test_blp_recovers_the_slope_with_hc3_errors():
    rng = np.random.default_rng(5)
    s = rng.normal(size=2000)
    phi = 0.1 + 0.5 * (s - s.mean()) + rng.normal(scale=2.0, size=2000)
    r = C.blp(phi, s)
    assert abs(r["beta2"] - 0.5) < 3 * r["se2"] and r["se1"] > 0 and r["se2"] > 0 and r["p2"] < 1e-3
    # HC3 equals the sum of squared leave-one-out coefficient changes
    s, phi = s[:30], phi[:30]
    X = np.column_stack([np.ones(30), s - s.mean()])
    full = np.linalg.lstsq(X, phi, rcond=None)[0]
    loo = np.array([np.linalg.lstsq(np.delete(X, i, 0), np.delete(phi, i), rcond=None)[0] for i in range(30)])
    r = C.blp(phi, s)
    np.testing.assert_allclose([r["se1"], r["se2"]], np.sqrt(((loo - full) ** 2).sum(0)), rtol=1e-8)


# Source support (gate G6)
def test_source_support_on_the_demo(frames):
    T, B = frames["trial_cov"], frames["hist_cov"]
    trt = (T["A"] == 1).to_numpy()
    sup = C.source_support(
        C.design_source(T[trt]),
        C.design_source(B),
        np.where(T["post"][trt] == 1, "post", "pre"),
        np.where(B["post"] == 1, "post", "pre"),
    )
    assert 0.80 <= sup["auc"] <= 0.87 and round(sup["auc"], 3) == 0.836
    tab = sup["table"].set_index("stratum")
    assert tab.index.tolist() == ["all", "post", "pre"]
    assert tab.loc["all", "share_unsupported"] <= 0.10 and tab.loc["all", "ess_B"] >= 50
    assert 35 <= tab.loc["post", "ess_B"] < 50 and tab.loc["pre", "ess_B"] > 200  # post FAIL, pre PASS
    assert round(tab.loc["all", "share_unsupported"], 3) == 0.049 and round(tab.loc["post", "max_share"], 3) == 0.078
    assert (tab.n_T.tolist(), tab.n_B.tolist()) == ([205, 147, 58], [547, 79, 468])


def test_w3_probe_passes_on_the_demo(frames):
    T = frames["trial_cov"]
    probe = pd.DataFrame({g: (T["tgrade"] == g).astype(float) for g in ("I", "II", "III")})
    probe["tsize_mm"] = T["tsize_mm"]
    tab = C.w3_probe(C.design_source(T), T["A"].to_numpy(), probe)
    assert tab.name.tolist() == ["I", "II", "III", "tsize_mm"]
    assert tab.smd_after.abs().max() <= 0.10 and round(tab.smd_after.abs().max(), 3) == 0.064


# Validation, placebo test and draw stability (gate G12, report only)
def test_hidden_control_validation_small(frames, prognostic):
    T, out = frames["trial_cov"], frames["outcomes"]["trial"]
    runs = C.hidden_control_validation(T, out, prognostic["m_T"], H2, 60, 5, 7)
    assert len(runs) == 10 and set(runs.method) == {"robust hybrid", "Hajek EB small arm"}
    assert (runs.note == C.VALIDATION_NOTE).all() and runs.note.str.contains("NOT TabPFN").all()  # per-rep rows
    assert np.isfinite(runs[["est", "se", "truth", "se_truth", "diff"]].to_numpy()).all()
    assert (runs.se > 0).all() and (runs.se_truth > 0).all()
    np.testing.assert_allclose(runs["diff"], runs.est - runs.truth)
    s = C.summarize_validation_runs(runs)
    assert list(s.columns) == ["method", "reps", "bias", "bias_mcse", "SD", "RMSE", "coverage", "coverage_mcse", "note"]
    assert (s.reps == 5).all() and s.coverage.between(0, 1).all() and s.note.str.contains("NOT TabPFN").all()
    assert s.note.str.contains("fixed prognostic-score predictions").all()
    assert s.note.str.contains("checks formulas and variance, not TabPFN").all()
    assert len(C.hidden_control_validation(T, out, None, H2, 60, 2, 0)) == 4  # no prognostic score (G7 none)


def test_placebo_test_small(frames, monkeypatch):
    log = []
    monkeypatch.setattr(models, "FIT_LOG", log)
    res = C.placebo_test(frames["trial_cov"], frames["outcomes"]["trial"], H2, 20, 100, 0)
    runs, s = res["runs"], res["summary"].iloc[0]
    assert len(runs) == 20 and np.isfinite(runs[["est", "se", "lo", "hi"]].to_numpy()).all() and (runs.se > 0).all()
    assert abs(s["bias"]) <= 3 * s["bias_mcse"] and abs(s["bias"]) < 3.0  # points; truth is 0
    assert 0 <= s["type_I_error"] <= 0.25 and s["coverage"] == pytest.approx(1 - s["type_I_error"])
    assert "NOT TabPFN" in s["note"] and (runs.note == C.PLACEBO_NOTE).all()  # summary and per-rep rows
    assert runs.note.str.contains("NOT TabPFN").all()
    assert len(log) == 20 * 5 and {x["backend"] for x in log} == {"logreg"}
    assert all(x["purpose"].startswith("placebo") for x in log)


def test_draw_stability_on_the_demo(rsample):
    ds = C.draw_stability(rsample["D"], rsample["A"], 60, 100, 0, GATES_H)
    assert ds["reps"] == 100 and ds["share_pass"] >= 0.95 and ds["share_exact"] >= 0.97
    assert ds["ess_p5"] > 20 and ds["max_share_p95"] < 0.15
    strict = C.draw_stability(rsample["D"], rsample["A"], 60, 10, 0, {**GATES_H, "G4H_ess_min_abs": 1000.0})
    assert strict["share_pass"] == 0.0
