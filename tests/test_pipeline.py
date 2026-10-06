"""Demo pipeline (tabs 2-4) with the stand-in and logistic backends: fast and offline."""

import numpy as np
import pytest

from src import exploratory, models
from src.exploratory import (
    HISTORICAL_ROW,
    HYBRID_ROW,
    LEGACY_ROWS,
    REFERENCE_ROW,
    SMALL_ARM_ROW,
    TREATED_ROW,
    comparability,
    historical_predictions,
    prepare_arms,
    run_estimate,
    run_validation,
    shift_check_text,
    validation_headline,
)
from src.gated import protocol
from src.models import (
    TabPFNSetupError,
    cross_source_features,
    default_backend,
    default_features,
    encode_features,
    fit_predict,
)
from src.survival import DAYS_PER_YEAR, km_at

H = 2 * DAYS_PER_YEAR
EXCL = {"horTh", "time", "cens"}


@pytest.fixture(scope="module")
def feats(demo):
    return default_features(demo["a_h"], demo["b_h"], EXCL)


def test_552_eligible_and_frames(demo):
    assert len(demo["hist"]) == 552 and demo["mask"].sum() == 552
    assert (demo["trial"].arm == "control").sum() == 440 and (demo["trial"].arm == "treated").sum() == 246
    assert demo["info"]["unmapped_a"] == []


def test_features_are_baseline_only(feats):
    assert feats == ["age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec"]


def test_encoding_consistent(demo, feats):
    (Xh, Xt), cat = encode_features([demo["hist"], demo["trial"]], feats)
    assert cat == [1, 2, 3] and Xh.shape == (552, 7) and Xt.shape == (686, 7)
    j = feats.index("tgrade")  # I/II/III share one code table across both datasets
    assert set(np.unique(Xt[:, j])) == {0, 1, 2} and set(np.unique(Xh[:, j])) <= {1, 2}


def test_comparability_standin(demo, feats):
    c = comparability(demo["hist"], demo["trial"], cross_source_features(feats), "standin", seed=0)
    assert 0.7 < c["auc"] < 0.95
    s = c["support"].set_index("stratum")
    assert set(s.index) == {"all", "Pre", "Post"}
    assert s.loc["all", ["n_treated", "n_hist"]].tolist() == [246, 552]
    assert "tgrade" not in set(c["features"]["feature"]) and "treated" in c["features"].columns
    assert len(c["oof"]) == 246 and 0 <= c["share_unsupported"] <= 1
    assert c["share_unsupported"] == s.loc["all", "share_unsupported"]
    # every row must pass; the verdict names the rows that do not
    assert c["failing"] == [r for r in s.index if not s.loc[r, "ok"]]
    assert c["verdict"] == ("limited support: " + ", ".join(c["failing"]) if c["failing"] else "supported")


def test_comparability_flags_postmenopausal_support(demo, feats):
    """Logistic backend: the density-ratio table reproduces the G6 pattern (postmenopausal treated weakly covered)."""
    c = comparability(demo["hist"], demo["trial"], cross_source_features(feats), "logreg", seed=0)
    s = c["support"].set_index("stratum")
    assert c["verdict"] == "limited support: Post" and c["failing"] == ["Post"]
    assert s.loc["Post", "ess_hist"] < 50 <= s.loc["Pre", "ess_hist"] and s.loc["all", "ok"]
    n_h, n_t = 552, 246
    assert c["pi_cut"] / (1 - c["pi_cut"]) * n_h / n_t == pytest.approx(20)  # density ratio at the plot line
    r_t = np.clip(c["oof"], 1e-6, 1 - 1e-6) / (1 - np.clip(c["oof"], 1e-6, 1 - 1e-6)) * n_h / n_t
    assert s.loc["all", "share_unsupported"] == pytest.approx(np.mean(r_t > 20))


def test_support_rule_matches_gate_g6():
    """Tab 2 repeats the gate G6 thresholds of src.gated.protocol; the two copies must agree."""
    g = protocol.GATES
    assert (
        exploratory.SUPPORT_RATIO_CUT,
        exploratory.SUPPORT_SHARE_MAX,
        exploratory.SUPPORT_ESS_MIN,
        exploratory.SUPPORT_WEIGHT_SHARE_MAX,
    ) == (g["G6_ratio_cut"], g["G6_share_unsupported_max"], g["G6_ess_min"], g["G6_weight_share_max"])


def test_estimate_standin(demo, feats):
    est = run_estimate(demo["hist"], demo["trial"], cross_source_features(feats), H, 60, "standin", seed=3, n_boot=300)
    t = est["table"].set_index("method")
    assert list(t.index) == [
        TREATED_ROW,
        HYBRID_ROW,
        SMALL_ARM_ROW,
        HISTORICAL_ROW,
        LEGACY_ROWS["full"],
        LEGACY_ROWS["dynamic"],
        REFERENCE_ROW,
    ]
    assert t.estimate.between(0.4, 0.95).all()
    lo, hi = t.loc[HYBRID_ROW, ["lo", "hi"]]
    assert lo < t.loc[HYBRID_ROW, "estimate"] < hi
    assert np.isnan(t.loc[HISTORICAL_ROW, ["lo", "hi"]].astype(float)).all()
    assert "NOT TabPFN" in t.loc[HYBRID_ROW, "note"] and "prognostic score" in t.loc[HYBRID_ROW, "note"]
    assert "dynamic borrowing" not in est["shift_text"] and "weighted to the treated" in est["shift_text"]
    assert est["n_real_labelled"] <= 60 and len(est["kept"]) == 60 and np.all(np.diff(est["kept"]) > 0)  # sorted draw
    # effects: S1 - S0 for every row but the treated; only the robust hybrid has a CI
    eff = est["effects"].set_index("method")
    assert list(eff.index) == list(t.index[1:])
    assert eff[["lo", "hi"]].notna().all(axis=1).tolist() == [m == HYBRID_ROW for m in eff.index]
    h = est["hybrid"]
    s1, s0 = t.loc[TREATED_ROW, "estimate"], t.loc[HYBRID_ROW, "estimate"]
    assert eff.loc[HYBRID_ROW, "effect"] == pytest.approx(s1 - s0)
    assert eff.loc[HYBRID_ROW, "effect"] == pytest.approx(h["est"]) and h["df"] == pytest.approx(h["ess"] - 1)
    assert h["converged"] and h["imbalance"] < 1e-6


def test_estimate_rows_logreg(demo, feats):
    """Treated row = KM with the log-log CI; reference = ATT-odds Hajek mean of all controls; legacy joint row."""
    trial = demo["trial"]
    est = run_estimate(
        demo["hist"], trial, cross_source_features(feats), H, 60, "logreg", seed=0, n_boot=200, do_joint=True
    )
    t = est["table"].set_index("method")
    trt = (trial["arm"] == "treated").to_numpy()
    km = km_at(trial["time"][trt], trial["cens"][trt], H)
    assert t.loc[TREATED_ROW, "estimate"] == pytest.approx(km.surv)
    assert t.loc[TREATED_ROW, "lo"] < km.surv < t.loc[TREATED_ROW, "hi"] <= 1
    assert LEGACY_ROWS["joint"] in t.index and t.estimate.between(0.4, 0.95).all()
    assert "normally unknown" in t.loc[REFERENCE_ROW, "note"]
    assert t.loc[REFERENCE_ROW, "lo"] < t.loc[REFERENCE_ROW, "estimate"] < t.loc[REFERENCE_ROW, "hi"]


def test_prepare_arms_outcomes_and_design(demo, feats):
    hist, trial = demo["hist"], demo["trial"]
    arms = prepare_arms(hist, trial, cross_source_features(feats), H)
    trt = (trial["arm"] == "treated").to_numpy()
    for m in (trt, ~trt):  # pseudo-observations within each arm average to that arm's KM
        assert arms["theta_t"][m].mean() == pytest.approx(km_at(trial["time"][m], trial["cens"][m], H).surv, abs=1e-10)
    assert arms["theta_h"].mean() == pytest.approx(km_at(hist["time"], hist["cens"], H).surv, abs=1e-10)
    # age, menostat (1 indicator), tsize (2), pnodes, progrec, estrec + within-trial tgrade (2 indicators)
    assert arms["Dt"].shape == (686, 9) and np.isfinite(arms["Dt"]).all()
    p = historical_predictions(arms, ["logreg"])
    assert set(p) == {"logreg", "logreg_pseudo"} and all(
        len(v) == 686 and v.min() >= 0 and v.max() <= 1 for v in p.values()
    )
    gappy = trial.copy()
    gappy.loc[gappy.index[:5], "age"] = np.nan
    gappy.loc[gappy.index[5:9], "menostat"] = np.nan
    assert np.isfinite(prepare_arms(hist, gappy, cross_source_features(feats), H)["Dt"]).all()  # imputed
    gappy.loc[gappy.index[0], "time"] = np.nan
    with pytest.raises(ValueError, match="follow-up time"):
        prepare_arms(hist, gappy, cross_source_features(feats), H)


def test_shift_check_text_weighted():
    text, warn = shift_check_text([0.5, 0.5], [1.0, 0.0], weights=[3, 1])
    assert "weighted to the treated" in text and "75% event-free" in text and "+25.0 points" in text
    assert "dynamic borrowing" not in text and warn is False  # noise = sqrt(0.75^2 0.25^2 + 0.25^2 0.75^2) = 0.27


def test_validation_standin(demo, feats):
    v = run_validation(
        demo["hist"], demo["trial"], cross_source_features(feats), H, 60, "standin", reps=10, seed=0, n_boot=100
    )
    s = v["summary"].set_index("method")
    methods = [HYBRID_ROW, SMALL_ARM_ROW, HISTORICAL_ROW, LEGACY_ROWS["full"], LEGACY_ROWS["dynamic"]]
    assert list(s.index) == methods and len(v["runs"]) == 10 * len(methods)
    assert "worth_real_controls" not in s.columns
    assert {"bias", "bias_mcse", "SD", "RMSE", "coverage", "coverage_mcse", "note"} <= set(s.columns)
    assert s.coverage.dropna().between(0, 1).all() and len(s.coverage.dropna()) == 4
    assert np.isnan(s.loc[HISTORICAL_ROW, "coverage"])
    assert s.RMSE.between(0.1, 20).all() and (s.reps == 10).all()
    runs = v["runs"]
    assert (runs.groupby("rep")["truth"].nunique() == 1).all()  # one ATT-targeted truth per split
    assert runs["diff"].to_numpy() == pytest.approx((runs["est"] - runs["truth"]).to_numpy())
    assert "NOT TabPFN" in s.loc[HYBRID_ROW, "note"] and "one realization" in s.loc[HISTORICAL_ROW, "note"]
    headline = validation_headline(v["summary"], v["n_real"])
    assert "robust hybrid" in headline and "MCSE" in headline and "10 splits" in headline


def test_setup_error_text_for_gated(monkeypatch):
    class Gated:
        def __init__(self, *a, **k):
            raise RuntimeError("Cannot access gated repo Prior-Labs/tabpfn_3_5 (401)")

    monkeypatch.setattr(models, "_make_classifier", lambda *a, **k: Gated())
    with pytest.raises(TabPFNSetupError, match="hf auth login"):
        fit_predict("tabpfn-local", np.random.rand(20, 2), np.r_[np.zeros(10), np.ones(10)], [np.random.rand(3, 2)])


def test_client_backend_needs_token(monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(TabPFNSetupError, match="TABPFN_TOKEN"):
        fit_predict("tabpfn-client", np.random.rand(20, 2), np.r_[np.zeros(10), np.ones(10)], [np.random.rand(3, 2)])


def test_default_backend(monkeypatch):
    monkeypatch.delenv("HCA_BACKEND", raising=False)
    assert default_backend() == "tabpfn-client"
    monkeypatch.setenv("HCA_BACKEND", "standin")
    assert default_backend() == "standin"


def test_client_retries_temporary_server_error(monkeypatch):
    attempts = []

    class Flaky:
        def fit(self, X, y):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("GCS upload permanently failed ... 500 InternalError")
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(models, "_make_classifier", lambda *a, **k: Flaky())
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    p = fit_predict("tabpfn-client", np.random.rand(20, 2), np.r_[np.zeros(10), np.ones(10)], [np.random.rand(3, 2)])[0]
    assert len(attempts) == 2 and p.shape == (3,)


def test_validation_no_crash_when_kept_controls_have_no_event(demo, feats):
    """S0 = 1 among few kept controls (short horizon) gives an undefined RR, not a ZeroDivisionError."""
    v = run_validation(
        demo["hist"], demo["trial"], cross_source_features(feats), 365.25, 10, "logreg", reps=40, seed=0, n_boot=50
    )
    assert len(v["summary"]) >= 2 and v["summary"]["note"].str.contains("NOT TabPFN").all()
