"""Pseudo-observations, censoring models, analysis frames and the design contract of src.causal.

Offline: Kaplan-Meier, Cox and linear algebra only.
"""

import numpy as np
import pandas as pd
import pytest

from src import causal as C
from src import survival
from src.survival import km_at

DAYS = 365.25
Z_T = ["age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec"]
Z_S = ["age", "menostat", "tsize", "pnodes", "progrec", "estrec"]


@pytest.fixture(scope="module")
def frames(demo):
    return C.gate_frames(demo, demo["df_a"], demo["df_b"])


def _trial(frames):
    T, out = frames["trial_cov"], frames["outcomes"]["trial"]
    fine, coarse = survival.strata_labels("A", T["A"], T["post"])
    return out["time"].to_numpy(float), out["cens"].to_numpy(float), fine, coarse


def _hist(frames):
    B, out = frames["hist_cov"], frames["outcomes"]["hist"]
    fine, coarse = survival.strata_labels("B", B["A"], B["post"])
    return out["time"].to_numpy(float), out["cens"].to_numpy(float), fine, coarse


# Pseudo-observations
def test_pseudo_km_without_censoring_is_the_event_free_indicator():
    rng = np.random.default_rng(0)
    t = rng.exponential(500, 200).round()  # rounding creates ties
    H = float(np.median(t))
    theta = survival.pseudo_km(t, np.ones(200), H)
    assert np.max(np.abs(theta - (t > H))) < 1e-12  # exact up to floating point


def test_pseudo_km_mean_equals_km_with_ties_and_censoring():
    rng = np.random.default_rng(1)
    t = rng.integers(1, 31, 300).astype(float) * 30.0  # 30 distinct times
    e = rng.binomial(1, 0.6, 300).astype(float)
    H = 600.0
    assert (t > H).sum() >= 10
    assert abs(survival.pseudo_km(t, e, H).mean() - km_at(t, e, H).surv) < 1e-10


def test_pseudo_km_equals_brute_force_leave_one_out():
    rng = np.random.default_rng(2)
    n = 40
    t = rng.integers(1, 15, n).astype(float)
    e = rng.binomial(1, 0.7, n).astype(float)
    H = 9.0
    s = km_at(t, e, H).surv
    loo = np.array([km_at(np.delete(t, i), np.delete(e, i), H).surv for i in range(n)])
    np.testing.assert_allclose(survival.pseudo_km(t, e, H), n * s - (n - 1) * loo, rtol=0, atol=1e-12)


def test_pseudo_km_with_no_event_before_horizon_is_one():
    assert np.array_equal(survival.pseudo_km([10.0, 20.0, 30.0], [0, 1, 1], 5.0), np.ones(3))


def test_pseudo_by_strata_merge_rule():
    rng = np.random.default_rng(3)
    H = 500.0
    parts = []  # (fine label, coarse label, times)
    parts.append(("X|0|pre", "X|0", rng.uniform(1, 1000, 60)))
    parts.append(("X|0|post", "X|0", rng.uniform(1, 1000, 20)))  # < 25 patients -> merge
    parts.append(("X|1|pre", "X|1", rng.uniform(1, 1000, 60)))
    parts.append(("X|1|post", "X|1", np.r_[rng.uniform(1, 400, 35), np.full(5, 800.0)]))  # 5 at risk -> merge
    parts.append(("X|2|pre", "X|2", rng.uniform(1, 1000, 50)))
    parts.append(("X|2|post", "X|2", rng.uniform(1, 1000, 50)))  # both fine strata kept
    fine = np.concatenate([[f] * len(t) for f, _, t in parts])
    coarse = np.concatenate([[c] * len(t) for _, c, t in parts])
    t = np.concatenate([t for _, _, t in parts])
    e = rng.binomial(1, 0.7, len(t)).astype(float)
    theta, used = survival.pseudo_by_strata(t, e, H, fine, coarse)
    assert set(used[coarse == "X|0"]) == {"X|0"}
    assert set(used[coarse == "X|1"]) == {"X|1"}
    assert np.array_equal(used[coarse == "X|2"], fine[coarse == "X|2"])
    m = coarse == "X|1"
    np.testing.assert_allclose(theta[m], survival.pseudo_km(t[m], e[m], H))
    m = fine == "X|2|post"
    np.testing.assert_allclose(theta[m], survival.pseudo_km(t[m], e[m], H))


def test_strata_labels_format():
    fine, coarse = survival.strata_labels("A", [1, 0], [0, 1])
    assert list(fine) == ["A|1|pre", "A|0|post"] and list(coarse) == ["A|1", "A|0"]


# Analysis frames and horizon adequacy on the demo
def test_gate_frames_funnel_and_blinding(frames):
    f = frames["funnel"]
    a, b = f[f.source == "A"], f[f.source == "B"]
    assert a.n_after.tolist() == [686, 620, 605] and b.n_after.tolist() == [552, 550, 547]
    assert a.step.tolist() == ["start (Stage-2)", "age <= 65", "event-free and under follow-up at day 91"]
    assert (a.iloc[-1].n_treated, a.iloc[-1].n_control) == (205, 400)
    T, B = frames["trial_cov"], frames["hist_cov"]
    assert not {"time", "cens"} & (set(T.columns) | set(B.columns))
    assert frames["landmark_ok"]
    assert len(frames["outcomes"]["trial"]) == len(T) and len(frames["outcomes"]["hist"]) == len(B)
    assert T.orig_index.is_monotonic_increasing and list(T.index) == list(range(len(T)))
    assert set(T.tsize_grp) == set(C.SIZE_CODES) and set(T.tgrade) == set(C.GRADE_CODES)
    assert set(B.tgrade) <= {"II", "III"} and set(B.menostat) == set(C.MENO_CODES)
    assert (B.A == 0).all() and "tsize_mm" not in B.columns and "year" in B.columns
    assert T.er_cat.dtype.kind == "i" and set(T.er_cat) == {0, 1, 2}
    # receptor categories: <10 -> 0, 10-99 -> 1, >=100 -> 2
    assert ((T.er_cat == 0) == (T.estrec < 10)).all() and ((T.pgr_cat == 2) == (T.progrec >= 100)).all()


def test_gate_frames_options(demo):
    fr = C.gate_frames(demo, demo["df_a"], demo["df_b"], b_years=(1984, 1989))
    assert len(fr["hist_cov"]) == 297 and fr["hist_cov"].post.sum() == 48
    assert fr["funnel"].step.iloc[-1] == "surgery 1984-89"
    raw = C.gate_frames(demo, demo["df_a"], demo["df_b"], age_max=None, landmark=None)
    assert (len(raw["trial_cov"]), len(raw["hist_cov"])) == (686, 552) and raw["landmark_ok"]
    s1 = C.gate_frames(demo, demo["df_a"], demo["df_b"], landmark=0)  # sensitivity S1
    assert len(s1["trial_cov"]) == 620


def test_gate_frames_tolerates_bad_labels_and_missing_columns(demo):
    """Gate G3 reports label and column problems of the mapping, so building the frames must not crash."""
    hist = demo["hist"].drop(columns=["time", "cens"]).assign(menostat=lambda h: h.menostat.str.lower())
    d = {**demo, "hist": hist, "a_h": demo["a_h"].assign(tsize="20 to 50")}
    fr = C.gate_frames(d, demo["df_a"], demo["df_b"])
    assert len(fr["hist_cov"]) == 0 and not fr["landmark_ok"]  # no B time: nobody passes the landmark
    assert fr["funnel"][fr["funnel"].source == "B"].n_after.tolist() == [552, 550, 0]
    assert set(fr["trial_cov"].tsize_grp) == {"20 to 50"}
    assert np.isnan(C.features_source(fr["trial_cov"])[:, 2]).all()  # unknown size label -> NaN, not a code
    assert np.isnan(C.design_source(fr["trial_cov"])[:, 2:4]).all()  # ... and NaN dummies, not "<=20"
    raw = C.gate_frames({**d, "hist": hist.assign(time=1000.0, cens=0)}, demo["df_a"], demo["df_b"])
    assert (raw["hist_cov"].post == 0).all() and np.isnan(C.features_source(raw["hist_cov"])[:, 1]).all()
    assert np.isnan(C.design_source(raw["hist_cov"])[:, 1]).all()  # "post" is not a known label: NaN, not 0


def test_demo_strata_and_horizon_adequacy(frames):
    t, e, fine, coarse = _trial(frames)
    tb, eb, fb, cb = _hist(frames)
    assert pd.Series(fine).value_counts().to_dict() == {"A|0|pre": 224, "A|0|post": 176, "A|1|post": 147, "A|1|pre": 58}
    assert pd.Series(fb).value_counts().to_dict() == {"B|0|pre": 468, "B|0|post": 79}
    for tt, ee, ff, cc in ((t, e, fine, coarse), (tb, eb, fb, cb)):
        _, used = survival.pseudo_by_strata(tt, ee, 2 * DAYS, ff, cc)
        assert np.array_equal(used, ff)  # no merge at 2 y
        hc = survival.horizon_check(tt, ee, 2 * DAYS, used)
        assert (hc.G_H >= 0.86).all() and hc.ok.all() and (hc.identity_err < 1e-8).all()
    hc5 = survival.horizon_check(t, e, 5 * DAYS, fine)
    assert (hc5.G_H < 0.70).all() and not hc5.ok.any()
    hc3 = survival.horizon_check(t, e, 3 * DAYS, fine).set_index("stratum")
    assert 0.71 <= hc3.loc["A|0|pre", "G_H"] <= 0.73
    assert list(hc3.columns) == ["n", "censored_before_H", "at_risk_H", "G_H", "identity_err", "ok"]


def test_kept_controls_of_the_frozen_draw_merge(frames):
    T = frames["trial_cov"]
    t, e, _, _ = _trial(frames)
    ctrl = np.flatnonzero(T.A == 0)
    kept = np.sort(np.random.default_rng(0).choice(ctrl, 60, replace=False))
    assert T.post.to_numpy()[kept].sum() == 22
    fine, coarse = survival.strata_labels("A", T.A.to_numpy()[kept], T.post.to_numpy()[kept])
    _, used = survival.pseudo_by_strata(t[kept], e[kept], 2 * DAYS, fine, coarse)
    assert set(used) == {"A|0"}
    assert survival.reverse_km(t[kept], e[kept], 2 * DAYS) == pytest.approx(0.876, abs=0.001)
    assert survival.reverse_km(t[kept], e[kept], 3 * DAYS) == pytest.approx(0.744, abs=0.001)


# Censoring models
def test_cox_ph_recovers_coefficients_and_baseline_hazard():
    rng = np.random.default_rng(4)
    n, lam0 = 2000, 0.1
    X = np.column_stack([rng.normal(size=n), rng.binomial(1, 0.5, n)])
    T = rng.exponential(1 / (lam0 * np.exp(0.7 * X[:, 0])))
    cens = rng.uniform(0, 20, n)
    t, d = np.minimum(T, cens), (T <= cens).astype(float)
    fit = survival.cox_ph(t, d, X)
    assert abs(fit["beta"][0] - 0.7) < 0.1 and abs(fit["beta"][1]) < 0.15
    assert np.all(fit["se"] > 0) and fit["lr_p"] < 1e-6 and fit["loglik"] > fit["loglik0"]
    t0 = float(np.median(t))
    lam_hat = fit["dLambda0"][fit["times"] <= t0].sum()  # baseline at the centred covariates
    lam_true = lam0 * t0 * np.exp(0.7 * fit["x_mean"][0])
    assert abs(lam_hat - lam_true) < 0.1 * lam_true
    T0 = rng.exponential(1 / lam0, n)  # null: no covariate effect
    t, d = np.minimum(T0, cens), (T0 <= cens).astype(float)
    assert survival.cox_ph(t, d, X)["lr_p"] > 0.001


def test_cox_ph_without_events():
    fit = survival.cox_ph([1.0, 2.0], [0, 0], [[0.5], [1.0]])
    assert fit["lr_p"] == 1.0 and len(fit["times"]) == 0


def test_cox_ph_and_censoring_G_refuse_missing_values():
    """A NaN covariate must raise instead of giving NaN coefficients and NaN G."""
    t, e = np.tile([1.0, 2.0, 3.0], 10), np.tile([0.0, 1.0, 0.0], 10)
    X = np.tile([[0.5], [np.nan], [1.0]], (10, 1))
    with pytest.raises(ValueError, match="missing or non-finite"):
        survival.cox_ph(t, e, X)
    with pytest.raises(ValueError, match="missing or non-finite"):
        survival.censoring_G(t, e, X, 2.5)  # 10 censoring events: the Cox path
    with pytest.raises(ValueError, match="missing or non-finite"):
        survival.cox_ph(np.r_[t[:-1], np.nan], e, np.ones((30, 1)))


def test_ipcw_with_stratum_reverse_km_recovers_km_on_demo_controls(frames):
    T = frames["trial_cov"]
    t, e, _, _ = _trial(frames)
    H = 2 * DAYS
    for p in (0, 1):
        m = ((T.A == 0) & (T.post == p)).to_numpy()
        ystar = survival.ipcw_outcome(t[m], H, np.full(m.sum(), survival.reverse_km(t[m], e[m], H)))
        assert abs(ystar.mean() - km_at(t[m], e[m], H).surv) < 0.005


def test_ipcw_outcome_floor():
    y = survival.ipcw_outcome([10.0, 1.0, 10.0], 5.0, [0.5, 0.5, 0.01], floor=0.10)
    np.testing.assert_allclose(y, [2.0, 0.0, 10.0])


def test_censoring_G_cox_and_km_fallback(frames):
    T = frames["trial_cov"]
    t, e, _, _ = _trial(frames)
    H = 3 * DAYS
    m = (T.A == 0).to_numpy()
    X = C.censoring_covariates(T[m], "A")
    G, info = survival.censoring_G(t[m], e[m], X, H)
    assert info["model"].startswith("Cox") and 0 <= info["lr_p"] <= 1 and info["n_events"] >= 20
    assert np.all((G > 0) & (G <= 1)) and abs(G.mean() - survival.reverse_km(t[m], e[m], H)) < 0.05
    G2, info2 = survival.censoring_G(t[m], e[m], X, H, min_events_per_coef=1000)
    assert info2["model"].startswith("KM") and np.allclose(G2, survival.reverse_km(t[m], e[m], H))
    XB = C.censoring_covariates(frames["hist_cov"], "B")
    assert XB.shape == (547, 4) and set(XB[:, 3]) == {0.0, 1.0, 2.0}


# Design contract
def test_design_builders_refuse_outcomes(frames):
    T = frames["trial_cov"]
    builders = [
        C.design_trial,
        C.design_trial_ext,
        C.design_source,
        C.design_source_logrec,
        C.features_trial,
        C.features_source,
        C.benchmark_variables,
        C.recalibration_design,
        lambda df: C.censoring_covariates(df, "A"),
    ]
    for col in ("time", "cens"):
        bad = T.assign(**{col: 1.0})
        for build in builders:
            with pytest.raises(ValueError, match="outcome"):
                build(bad)


def test_design_shapes_and_fixed_codes(frames):
    T, B = frames["trial_cov"], frames["hist_cov"]
    assert C.design_trial(T).shape == (605, 8) and C.design_trial_ext(T).shape == (605, 11)
    assert C.design_source(T).shape == (605, 9) and C.design_source(B).shape == (547, 9)
    assert C.design_source_logrec(B).shape == (547, 7)
    assert len(C.DESIGN_TRIAL_NAMES) == 8 and len(C.DESIGN_TRIAL_EXT_NAMES) == 11 and len(C.DESIGN_SOURCE_NAMES) == 9
    D = C.design_trial(T)
    np.testing.assert_array_equal(D[:, 3], (T.tgrade == "II").to_numpy(float))
    np.testing.assert_allclose(D[:, 7], np.log1p(T.estrec.to_numpy(float)))
    F = C.features_trial(T)
    np.testing.assert_array_equal(F[:, 1], (T.menostat == "Post").to_numpy(float))  # Pre = 0, Post = 1
    np.testing.assert_array_equal(F[:, 3], T.tgrade.map({"I": 0, "II": 1, "III": 2}).to_numpy(float))
    sub = T[T.tgrade != "I"]  # same codes in any subset
    np.testing.assert_array_equal(C.features_trial(sub)[:, 3], F[(T.tgrade != "I").to_numpy(), 3])
    Fs = C.features_source(B)
    assert set(Fs[:, 2]) == {0.0, 1.0, 2.0} and not np.isnan(Fs).any()
    np.testing.assert_array_equal(Fs[:, 2], B.tsize_grp.map({"<=20": 0, "20-50": 1, ">50": 2}).to_numpy(float))
    Z = C.recalibration_design(T, np.full(605, 0.7))
    assert Z.shape == (605, 4) and C.recalibration_design(T).shape == (605, 3)


def test_missing_or_unknown_labels_give_nan_indicators_not_the_reference_level(frames):
    """A missing or unknown label must not be coded as the reference level (post = 0, grade I, size <=20,
    receptor <10), or a missing value within G3's allowance would reach the models unnoticed."""
    T, B = frames["trial_cov"], frames["hist_cov"]
    for X in (
        C.design_trial(T),
        C.design_trial_ext(T),
        C.design_source(T),
        C.design_source(B),
        C.design_source_logrec(B),
        C.recalibration_design(T),
        C.censoring_covariates(T, "A"),
        C.censoring_covariates(B, "B"),
    ):
        assert np.isfinite(X).all()  # the demo has no missing value
    bad = T.copy()
    bad.loc[0, "menostat"], bad.loc[1, "menostat"] = np.nan, "post"  # missing; unknown (lower case)
    bad.loc[2, "tgrade"], bad.loc[3, "tgrade"] = None, "IV"
    D, ok = C.design_trial(bad), C.design_trial(T)
    assert np.isnan(D[[0, 1], 1]).all() and np.isnan(D[2:4, 3:5]).all()
    assert np.isnan(C.design_trial_ext(bad)[[0, 1], 9]).all()  # age x post
    assert np.array_equal(D[4:], ok[4:]) and np.array_equal(D[:4, [0, 2, 5, 6, 7]], ok[:4, [0, 2, 5, 6, 7]])
    with pytest.raises(ValueError, match="missing or non-finite"):
        C.ps_logistic(D, bad["A"].to_numpy())  # stops instead of using post = 0
    assert np.isnan(C.recalibration_design(bad)[2:4, :2]).all()
    assert np.isnan(C.censoring_covariates(bad, "A")[[0, 1], 1]).all()
    src = B.astype({"er_cat": float})
    src.loc[0, "tsize_grp"], src.loc[1, "er_cat"], src.loc[2, "pgr_cat"], src.loc[3, "menostat"] = (
        "20 to 50",
        np.nan,
        7,
        None,
    )
    Ds = C.design_source(src)
    assert np.isnan(Ds[0, 2:4]).all() and np.isnan(Ds[1, 5:7]).all() and np.isnan(Ds[2, 7:9]).all()
    assert np.isnan(Ds[3, 1]) and np.array_equal(Ds[4:], C.design_source(B)[4:])
    assert np.isnan(C.censoring_covariates(src, "B")[0, 3])  # unknown size group -> NaN, not a code


def test_model_columns_are_the_dag_adjustment_sets(frames):
    assert set(C.TRIAL_COLUMNS) == set(Z_T) and set(C.SOURCE_COLUMNS) == set(Z_S)
    assert set(C.TRIAL_COLUMNS.values()) <= set(frames["trial_cov"].columns)
    for df in (frames["trial_cov"], frames["hist_cov"]):
        assert set(C.SOURCE_COLUMNS.values()) <= set(df.columns)
