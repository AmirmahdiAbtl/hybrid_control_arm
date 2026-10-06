"""End-to-end gated runs on the demo, offline and with cheap backends: the numbers are NOT TabPFN results."""

import hashlib
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src import causal as C
from src import gated, models
from src.cli import run_gated as cli
from src.demo import APPROVED, DEMO, PATH_A, PATH_B
from src.gated import protocol, runlog
from src.harmonize import load_mapping

REDUCED = {"validation_reps": 20, "placebo_reps": 20, "draw_stability_reps": 20, "cate_band_draws": 200}
FILES = [
    "audit.jsonl",
    "verdict.json",
    "gates.csv",
    "estimates.csv",
    "sensitivity.csv",
    "cate.csv",
    "support.csv",
    "balance.csv",
    "validation.csv",
    "placebo.csv",
    "report.md",
]


@pytest.fixture(scope="module")
def inputs():
    rules = [c for c in json.loads((DEMO / "llm_eligibility.json").read_text())["criteria"] if c["kind"] == "rule"]
    return pd.read_csv(PATH_A), pd.read_csv(PATH_B), load_mapping(APPROVED), rules


@pytest.fixture(autouse=True)
def reduced(monkeypatch):
    for k, v in REDUCED.items():
        monkeypatch.setitem(protocol.GATES, k, v)


def _run(out, inputs, **kw):
    kw = {"backend": "logreg", "cache_dir": None, **kw}
    return gated.run_gated(*inputs, out_dir=str(out), **kw)


def _events(res):
    return [json.loads(line) for line in (Path(res["run_dir"]) / "audit.jsonl").read_text().splitlines()]


def _module_run(inputs, out, **kw):
    with pytest.MonkeyPatch.context() as mp:
        for k, v in REDUCED.items():
            mp.setitem(protocol.GATES, k, v)
        lines = []
        res = _run(out, inputs, progress=lambda f, m="": lines.append(m), **kw)
    return res, lines


@pytest.fixture(scope="module")
def hybrid(inputs, tmp_path_factory):
    return _module_run(inputs, tmp_path_factory.mktemp("hybrid"), mode="hybrid")


@pytest.fixture(scope="module")
def internal(inputs, tmp_path_factory):
    return _module_run(inputs, tmp_path_factory.mktemp("internal"), mode="internal")


def test_hybrid_run_order_and_staged_data_access(hybrid):
    res, lines = hybrid
    g = res["gates"]
    assert list(dict.fromkeys(g["gate"])) == [f"G{i}" for i in range(13)]
    ev = _events(res)
    names = [e["event"] for e in ev]
    gate_at = [(i, e["gate"]) for i, e in enumerate(ev) if e["event"] == "gate"]
    unblind = names.index("unblind")
    assert max(i for i, gid in gate_at if gid == "G8") < unblind < min(i for i, gid in gate_at if gid == "G9")
    assert names.index("read_external_outcomes") < min(i for i, gid in gate_at if gid == "G7")
    assert max(i for i, gid in gate_at if gid in ("G3", "G4", "G5", "G6")) < names.index("read_external_outcomes")
    assert names[0] == "run_start" and names[-1] == "verdict" and names.count("unblind") == 1
    fits = [e for e in ev if e["event"] == "model_fit"]
    assert fits and all(names.index("read_external_outcomes") < ev.index(e) for e in fits)
    assert all(ev.index(e) > unblind for e in fits if e["purpose"].startswith("mu_"))
    assert "== UNBLINDED trial outcomes ==" in lines and any(
        x.startswith("G4  within-trial positivity  [H]") for x in lines
    )


def test_hybrid_gate_results(hybrid):
    res, _ = hybrid
    g = res["gates"].set_index(["gate", "scope"])
    assert g.loc[("G8", "X 2y"), "status"] == "INFEASIBLE" and res["g8"]["mdm"] == pytest.approx(0.13, abs=0.01)
    assert g.loc[("G6", "X post"), "status"] == "FAIL" and g.loc[("G6", "X pre"), "status"] == "PASS"
    assert g.loc[("G6", "X all"), "status"] == "PASS" and g.loc[("G6", "X W3"), "status"] == "PASS"
    assert g.loc[("G4", "H"), "status"] == "PASS" and g.loc[("G5", "H"), "status"] == "PASS"
    assert g.loc[("G9", "H 2y"), "status"] == "PASS" and g.loc[("G10", "H 2y"), "status"] == "PASS"
    assert g.loc[("G11", "H 2y"), "status"] == "NOT_APPLICABLE" and g.loc[("G12", "V"), "status"] == "REPORT"
    assert "ess 24.48 >= 20" in g.loc[("G4", "H"), "checks"]  # frozen sorted draw: ESS 24.5


def test_hybrid_verdict_and_primary_estimate(hybrid):
    res, _ = hybrid
    assert (res["verdict"], res["tier"], res["claim"]) == ("ACCEPTED", "SUPPORTIVE", "effect not demonstrated")
    p = res["primary"]
    assert (p["scope"], p["estimand"], p["horizon_years"], p["role"]) == ("H", "ATT", 2.0, "primary")
    assert -0.05 <= p["est"] <= 0.15 and p["lo"] < 0 < p["hi"]
    # frozen sorted draw with the logreg prognostic score (NOT TabPFN)
    assert (round(p["est"] * 100, 2), round(p["se"] * 100, 2)) == (5.42, 10.80)
    assert p["df"] == pytest.approx(p["ess"] - 1) and res["tabpfn_fits"] == 0 and res["actions"] == []
    assert res["text"].startswith("ACCEPTED (SUPPORTIVE) - H ATT at 2 years: effect not demonstrated. Estimate +5.4")
    assert "borrowing INFEASIBLE" in res["text"] and res["text"].splitlines()[-1].startswith("Evidence type")
    est = res["estimates"]
    keys = set(zip(est.scope, est.estimand, est.horizon_years, est.role))
    assert {("R", "ATT", 2.0, "reference"), ("E", "ATT", 2.0, "exploratory"), ("H", "ATC", 2.0, "secondary")} <= keys


def test_hybrid_labels_say_not_tabpfn(hybrid):
    res, _ = hybrid
    assert all("NOT TabPFN" in v for v in res["labels"].values())
    labels = pd.concat([res["estimates"].backend_label, res["sensitivity"].backend_label, res["cate"].backend_label])
    for label in labels:
        assert all("NOT TabPFN" in part for part in label.split(";")), label
    assert res["validation"]["summary"].note.str.contains("NOT TabPFN").all()
    assert res["placebo"]["summary"].note.str.contains("NOT TabPFN").all()
    assert "NOT TabPFN" in res["gates"].set_index("gate").loc["G12", "note"]


def test_hybrid_outputs_and_audit_integrity(hybrid):
    res, _ = hybrid
    d = Path(res["run_dir"])
    assert d.parent.name == res["protocol_digest"][:10] and d.name == "run_001_hybrid"
    assert all((d / f).exists() for f in FILES)
    lines = (d / "audit.jsonl").read_text().splitlines()
    assert all(isinstance(json.loads(x), dict) for x in lines)
    v = json.loads((d / "verdict.json").read_text())
    assert v["audit_sha256"] == hashlib.sha256((d / "audit.jsonl").read_bytes()).hexdigest()
    assert (v["verdict"], v["tier"], v["claim"], v["tabpfn_fits"]) == (
        "ACCEPTED",
        "SUPPORTIVE",
        "effect not demonstrated",
        0,
    )
    assert v["primary"]["scope"] == "H" and v["first_failing_gate"] is None
    gcsv = pd.read_csv(d / "gates.csv")
    assert list(gcsv.columns) == [
        "gate",
        "name",
        "scope",
        "attempt",
        "status",
        "metric",
        "value",
        "op",
        "threshold",
        "action",
        "note",
    ]
    est = pd.read_csv(d / "estimates.csv")
    assert list(est.columns) == protocol.EST_COLUMNS and len(est) == len(res["estimates"])
    report = (d / "report.md").read_text()
    assert "## Estimates" in report and "Evidence type: non-randomized comparison" in report
    assert "## Assumptions and probes" in report and "W1" in report


def test_hybrid_sensitivity_and_validation_tables(hybrid):
    res, _ = hybrid
    sens = res["sensitivity"]
    at2 = sens[sens.horizon_years == 2.0]
    assert set(at2[at2.scope == "H"].id) == {"S4", "S5", "S6", "X1", "NC"}
    assert set(at2[at2.scope == "R"].id) == {"S1", "S2", "S3", "S4", "S5", "S6", "NC"}
    assert set(at2[at2.scope == "E"].id) == {"X1", "X2", "X3", "X4"}
    h5 = at2[(at2.scope == "H") & (at2.id == "S5")].iloc[0]  # linear D_s score, sorted draw: +4.73 (10.81)
    assert (round(h5.est * 100, 2), round(h5.se * 100, 2)) == (4.73, 10.81)
    assert np.isfinite(sens[["est", "se", "lo", "hi"]].to_numpy(float)).all()
    v = res["validation"]
    assert set(v["summary"].method) == {"robust hybrid", "Hajek EB small arm"} and (v["summary"].reps == 20).all()
    assert v["draw_stability"]["reps"] == 20 and len(res["placebo"]["runs"]) == 20


def test_internal_run(internal):
    res, _ = internal
    assert (res["verdict"], res["tier"], res["claim"]) == ("ACCEPTED", "SUPPORTIVE", "effect not demonstrated")
    est = res["estimates"]
    r2 = est[(est.scope == "R") & (est.horizon_years == 2.0)].set_index("estimand")
    assert set(r2.index) == {"ATT", "ATE", "ATC", "ATO"}
    assert np.isfinite(r2[["est", "se", "lo", "hi"]].to_numpy(float)).all()
    p = res["primary"]
    assert (p["scope"], p["estimand"], p["role"]) == ("R", "ATT", "primary") and p["changed_from"] is None
    assert 0.0 <= p["est"] <= 0.10 and p["lo"] < 0 < p["hi"] and p["se"] == pytest.approx(0.039, abs=0.005)
    assert r2.loc["ATE", "role"] == "secondary" and set(est.scope) == {"R"}
    cate = res["cate"]
    for (h, kind), grp in cate[cate.kind != "BLP"].groupby(["horizon_years", "estimand"]):
        assert (grp.kind == "subgroup").sum() == 6 and (grp.kind == "contrast").sum() == 3, (h, kind)
    assert set(cate.horizon_years) == {2.0, 3.0} and (cate.kind == "BLP").sum() == 2
    assert cate.note.str.contains("not evidence of effect modification").all()
    sub = cate[cate.kind == "subgroup"]
    assert (sub.band_lo <= sub.lo).all() and (sub.band_hi >= sub.hi).all()
    g = res["gates"].set_index("gate")
    assert (g.loc[["G6", "G7", "G8"], "status"] == "SKIPPED").all()
    assert len(res["placebo"]["summary"]) == 1 and res["placebo"]["summary"].note.str.contains("NOT TabPFN").all()
    assert res["validation"] == {} and "Historical data not used (internal mode)" in res["text"]
    assert res["text"].startswith("ACCEPTED (SUPPORTIVE) - R ATT at 2 years: effect not demonstrated.")


def test_standin_run_is_not_interpretable(tmp_path, inputs):
    res = _run(tmp_path, inputs, mode="internal", backend="standin", validate=False)
    assert res["tier"] == "NOT INTERPRETABLE" and res["claim"] is None and res["primary"] is None
    assert res["text"].startswith("TEST RUN (random-forest stand-in, NOT TabPFN) - ")
    assert len(res["estimates"]) == len(res["sensitivity"]) == len(res["cate"]) == 0
    report = (Path(res["run_dir"]) / "report.md").read_text()
    verdict_part = report.split("## Gates")[0]
    assert "points (95% CI" not in verdict_part and "Nothing to interpret" in verdict_part
    assert "## Estimates" not in report and len(pd.read_csv(Path(res["run_dir"]) / "estimates.csv")) == 0
    assert res["gates"].set_index("gate").loc["G0", "note"].count("TEST RUN") == 1


def test_fit_budget_overrun_is_not_accepted(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(protocol, "MAX_TABPFN_FITS", 0)
    monkeypatch.setattr(runlog, "tabpfn_fits", lambda ctx: 1)
    res = _run(tmp_path, inputs, mode="hybrid", validate=False)
    assert res["verdict"] == "NOT ACCEPTED" and "budget" in res["reason"]
    assert res["first_failing_gate"] == "G10" and res["tier"] == "NOT INTERPRETABLE"
    assert "unblind" not in [e["event"] for e in _events(res)]  # stopped before the first model call


def test_hybrid_positivity_failure(tmp_path, inputs, monkeypatch):
    monkeypatch.setitem(protocol.GATES, "G4H_ess_min_abs", 1000.0)
    res = _run(tmp_path, inputs, mode="hybrid", validate=False)
    assert (res["verdict"], res["tier"], res["first_failing_gate"]) == ("NOT ACCEPTED", "NOT INTERPRETABLE", "G4")
    assert res["reason"] == "hybrid not estimable for this sample"
    assert not res["estimand_changed"] and res["actions"] == [] and res["primary"] is None
    assert res["gates"].iloc[-1][["gate", "scope", "status"]].tolist() == ["G4", "H", "FAIL"]
    assert "No treatment-effect estimate is reported because G4 failed" in res["text"]
    assert {"unblind", "read_external_outcomes"}.isdisjoint(e["event"] for e in _events(res))


def test_internal_positivity_failure_switches_to_ato(tmp_path, inputs, monkeypatch):
    monkeypatch.setitem(protocol.GATES, "G4_ess_ratio_min", 0.99)
    res = _run(tmp_path, inputs, mode="internal", validate=False)
    assert res["actions"] == ["B4.1"] and res["estimand_changed"]
    p = res["primary"]
    assert (p["estimand"], p["changed_from"], p["scope"], p["role"]) == ("ATO", "ATT", "R", "primary")
    assert (res["verdict"], res["tier"]) == ("ACCEPTED", "SUPPORTIVE")
    assert "estimand changed from ATT to ATO" in res["text"] and "not to the treated" in res["text"]
    ev = _events(res)
    names = [e["event"] for e in ev]
    assert names.index("action") < names.index("unblind") and ev[names.index("action")]["id"] == "B4.1"
    g4 = res["gates"][res["gates"].gate == "G4"].set_index("scope")
    assert g4.loc["R ATT", "status"] == "FAIL" and "PASS_AFTER_ACTION" in g4.loc["R ATO", "status"].tolist()
    assert set(res["estimates"].estimand) == {"ATO"}  # failing estimands are dropped


def test_run_counter_lists_earlier_verdicts(tmp_path, inputs, monkeypatch):
    monkeypatch.setitem(protocol.GATES, "G4H_ess_min_abs", 1000.0)  # fast runs (stop at G4)
    first = _run(tmp_path, inputs, mode="hybrid", validate=False)
    second = _run(tmp_path, inputs, mode="hybrid", validate=False)
    assert Path(first["run_dir"]).name == "run_001_hybrid" and Path(second["run_dir"]).name == "run_002_hybrid"
    assert first["protocol_digest"] == second["protocol_digest"] and first["earlier_runs"] == []
    assert [(r["run"], r["verdict"]) for r in second["earlier_runs"]] == [(1, "NOT ACCEPTED")]
    assert _events(second)[0]["earlier_runs"][0]["run"] == 1
    assert "run_001_hybrid" in (Path(second["run_dir"]) / "report.md").read_text()


def test_gated_run_makes_no_network_call(tmp_path, inputs, monkeypatch):
    calls = []

    def refuse(self, *args, **kwargs):
        calls.append(args)
        raise OSError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    res = _run(tmp_path, inputs, mode="internal", validate=False)
    assert res["verdict"] == "ACCEPTED" and calls == []


def _refuse(self, *args, **kwargs):
    raise OSError("network disabled in tests")


def test_tabpfn_setup_error_writes_partial_outputs(tmp_path, inputs, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    with pytest.raises(models.TabPFNSetupError):
        _run(tmp_path, inputs, mode="hybrid", backend="tabpfn-client", validate=False)
    run_dir = next(tmp_path.glob("*/run_001_hybrid"))
    v = json.loads((run_dir / "verdict.json").read_text())
    assert v["verdict"] == "NOT ACCEPTED" and v["tier"] == "NOT INTERPRETABLE" and "TabPFN" in v["reason"]
    assert v["tabpfn_fits"] == 0


def test_cli_internal_logreg(tmp_path, capsys):
    code = cli.main(["--mode", "internal", "--backend", "logreg", "--out", str(tmp_path / "gated")])
    out = capsys.readouterr().out
    assert code == 0 and "Backend: logistic/linear regression baseline (NOT TabPFN)" in out
    assert "VERDICT: ACCEPTED (SUPPORTIVE) - R ATT at 2 years: effect not demonstrated" in out
    assert "== UNBLINDED trial outcomes ==" in out and "Expected TabPFN fits: 0" in out
    assert sum(line[:4].strip() in {f"G{i}" for i in range(13)} for line in out.splitlines()) >= 13
    assert not (tmp_path / "cache").exists()  # cheap backends are never cached


def test_balance_action_b51_restarts_g4_and_g5(tmp_path, inputs, monkeypatch):
    monkeypatch.setitem(protocol.GATES, "G5_smd_max", 0.01)  # every R estimand > 0.01 -> B5.1 (once)
    res = _run(tmp_path, inputs, mode="internal", validate=False)
    assert res["actions"] == ["B5.1"]
    g = res["gates"]
    assert set(g[(g.gate == "G4") & (g.attempt == 2)].action) == {"B5.1"}
    att2 = g[(g.gate == "G5") & (g.attempt == 2)].set_index("scope").status
    assert att2["R ATT"] == "CONDITIONAL" and att2["R ATO"] == "PASS_AFTER_ACTION"  # ATO balances D_t,ext exactly
    assert (res["verdict"], res["tier"]) == ("ACCEPTED", "SUPPORTIVE") and not res["estimand_changed"]
    ev = [e["event"] for e in _events(res)]
    assert ev.index("action") < ev.index("unblind")


def _primary_3y(monkeypatch):
    """Code-path check only: make 3 y the primary horizon, where the R ATT CI excludes 0 (+10.0, CI +1.3 to +18.7)."""
    monkeypatch.setitem(protocol.HORIZON_YEARS, "primary", 3.0)
    monkeypatch.setitem(protocol.HORIZON_YEARS, "secondary", 2.0)


def test_g11_failure_is_exploratory(tmp_path, inputs, monkeypatch):
    _primary_3y(monkeypatch)
    res = _run(tmp_path, inputs, mode="internal", validate=False)
    assert (res["verdict"], res["tier"], res["first_failing_gate"]) == ("NOT ACCEPTED", "EXPLORATORY", "G11")
    g11 = res["gates"].set_index("gate").loc["G11"]
    assert g11.status == "FAIL" and "rr_star_lim 1.029 > 1.202 -> no" in g11.checks  # RR CI limit 1.03 < BF_ref 1.20
    assert res["claim"] is None and len(res["estimates"]) > 0 and len(res["benchmarks"]) == 10
    assert "points" not in res["text"].splitlines()[0] and "not robust" in res["text"].splitlines()[0]
    report = (Path(res["run_dir"]) / "report.md").read_text()
    assert f"## Appendix: {protocol.EXPLORATORY_TITLE}" in report and "## Estimates" not in report


def test_g11_pass_gives_the_demonstrated_claim(tmp_path, inputs, monkeypatch):
    """Synthetic benchmarks (BF 1.0) and named floor 1.0, only to exercise the PASS branch and its wording."""
    _primary_3y(monkeypatch)
    monkeypatch.setitem(protocol.GATES, "G11_named_rr", (1.0, 1.0))
    real = C.benchmark_bounding_factors
    monkeypatch.setattr(C, "benchmark_bounding_factors", lambda *a, **k: real(*a, **k).assign(bf=1.0))
    res = _run(tmp_path, inputs, mode="internal", validate=False)
    assert (res["verdict"], res["tier"]) == ("ACCEPTED", "SUPPORTIVE")
    assert res["claim"] == "effect demonstrated under the stated assumptions"
    assert res["secondary_claim"] == "effect at 2 years not demonstrated"  # fixed sequence reached
    assert res["text"].startswith(
        "ACCEPTED (SUPPORTIVE) - R ATT at 3 years: effect demonstrated under the stated "
        "assumptions. Estimate +10.0 points (95% CI +1.3 to +18.7)."
    )
    report = (Path(res["run_dir"]) / "report.md").read_text()
    assert "would need a bias factor above 1.03" in report and "Pre-specified analysis; all gates met." in report


class _FakeTabPFN:
    """Stands in for tabpfn.TabPFNRegressor in fit counting only (linear model; NOT TabPFN)."""

    def __init__(self, cat_idx):
        self.model = models._linear_pipeline(LinearRegression(), cat_idx)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X, output_type="mean"):
        return self.model.predict(X)


def test_tabpfn_fit_count_and_cache_resume(tmp_path, inputs, monkeypatch):
    monkeypatch.delenv("HCA_TABPFN_MODEL_PATH", raising=False)
    monkeypatch.setattr(models, "_make_regressor", lambda backend, seed, cat_idx: _FakeTabPFN(cat_idx))
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    cache = tmp_path / "cache"
    first = _run(
        tmp_path / "gated", inputs, mode="hybrid", backend="tabpfn-local", cache_dir=str(cache), validate=False
    )
    assert first["tabpfn_fits"] == runlog.expected_tabpfn_fits("hybrid") == 32 and first["cache_hits"] == 0
    fits = [e for e in _events(first) if e["event"] == "model_fit"]
    assert len(fits) == 32 and all(e["backend"] == "tabpfn-local" and not e["cached"] for e in fits)
    assert all(e["meta"]["model_path"] == "auto" for e in fits)
    assert sum(e["purpose"].startswith("mB_") for e in fits) == 12 and len(list(cache.glob("*.npz"))) == 32
    second = _run(
        tmp_path / "gated", inputs, mode="hybrid", backend="tabpfn-local", cache_dir=str(cache), validate=False
    )
    assert second["tabpfn_fits"] == 0 and second["cache_hits"] == 32 and second["run_index"] == 2
    assert second["primary"]["est"] == first["primary"]["est"]
    v = json.loads((Path(second["run_dir"]) / "verdict.json").read_text())
    assert (v["tabpfn_fits"], v["cache_hits"]) == (0, 32)


@pytest.mark.parametrize("mode, n_controls, expected", [("hybrid", 60, 22), ("internal", None, 20)])
def test_tabpfn_fit_budget_per_mode(tmp_path, inputs, monkeypatch, mode, n_controls, expected):
    """Hybrid real use (no hidden controls, so no mu0): 11 fits per horizon; internal: 10 per horizon."""
    monkeypatch.delenv("HCA_TABPFN_MODEL_PATH", raising=False)
    monkeypatch.setattr(models, "_make_regressor", lambda backend, seed, cat_idx: _FakeTabPFN(cat_idx))
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    df_a, df_b, mapping, rules = inputs
    if n_controls:  # a trial with at most N_REAL eligible controls: all are kept, none hidden
        df_a = pd.concat([df_a[df_a.horTh == "yes"], df_a[df_a.horTh == "no"].sample(n_controls, random_state=1)])
    res = _run(tmp_path, (df_a.sort_index(), df_b, mapping, rules), mode=mode, backend="tabpfn-local")
    assert res["tabpfn_fits"] == expected == runlog.expected_tabpfn_fits(mode, demo=n_controls is None)
    if n_controls:
        assert "R" not in set(res["estimates"].scope) and len(res["cate"]) == 0
        assert res["validation"] == {} and res["placebo"] == {} and res["verdict"] in ("ACCEPTED", "NOT ACCEPTED")
        assert res["gates"].set_index("gate").loc["G4", "scope"] == "H"  # no R reference rows


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_an_exception_ends_the_run_incomplete(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(C, "robust_hybrid_att", _boom)
    res = _run(tmp_path, inputs, mode="hybrid", validate=False)
    assert (res["verdict"], res["tier"], res["first_failing_gate"]) == ("NOT ACCEPTED", "NOT INTERPRETABLE", None)
    assert res["reason"] == "incomplete: RuntimeError: boom" and res["primary"] is None
    ev = _events(res)
    assert next(e for e in ev if e["event"] == "error")["message"] == "boom" and ev[-1]["event"] == "verdict"
    assert json.loads((Path(res["run_dir"]) / "verdict.json").read_text())["reason"].startswith("incomplete")
    assert res["text"].startswith("NOT ACCEPTED (NOT INTERPRETABLE) - No treatment-effect estimate is reported")


@pytest.mark.parametrize("case", ["no surgery year", "source model fails"])
def test_a_failing_exploratory_row_is_recorded_and_the_run_goes_on(tmp_path, inputs, monkeypatch, case):
    """X1 needs the Rotterdam surgery year; E, E-pre and X1-X4 need the source model. Their errors never end the run."""
    df_a, df_b, mapping, rules = inputs
    if case == "no surgery year":
        df_b, failing = df_b.drop(columns="year"), ["X1"]
    else:
        monkeypatch.setattr(C, "transport_att", _boom)
        failing = ["E", "E-pre", "X1", "X2", "X3", "X4"]
    res = _run(tmp_path, (df_a, df_b, mapping, rules), mode="hybrid", validate=False)
    assert (res["verdict"], res["tier"], res["first_failing_gate"]) == ("ACCEPTED", "SUPPORTIVE", None)
    assert round(res["primary"]["est"] * 100, 2) == 5.42  # primary scope H is untouched
    failed = [e for e in _events(res) if e["event"] == "row_failed"]
    assert [e["row"] for e in failed] == [f"{row} {h}y" for h in (2, 3) for row in failing]
    assert all(e["message"] and e["traceback"] for e in failed)
    sens, est = res["sensitivity"], res["estimates"]
    bad = sens[sens.note.str.startswith("failed: ")]
    rows_per_horizon = sum(2 if row == "X1" else 1 for row in failing if row.startswith("X"))  # X1: H and E
    assert set(bad.id) == {row for row in failing if row.startswith("X")} and len(bad) == 2 * rows_per_horizon
    assert bad.est.isna().all() and bad.same_sign.isna().all() and len(sens) == 32
    bad_est = est[est.note.str.startswith("failed: ")]
    assert len(bad_est) == (6 if "E" in failing else 0) and set(bad_est.scope) <= {"E"} and len(est) == 20


def test_missing_outcomes_are_removed_like_missing_covariates(tmp_path, inputs):
    """A missing time or cens never reaches the models: the patient is removed in a logged funnel step."""
    df_a, df_b, mapping, rules = inputs
    eligible_a = df_a.index[(df_a.age <= 65) & (df_a.time > 91)]
    eligible_b = df_b.index[
        (df_b.nodes >= 1)
        & (df_b.chemo == 1)
        & (df_b.hormon == 0)
        & (df_b.age <= 65)
        & (df_b.recur == 0)
        & (df_b.dtime > 91)
    ]
    a, b = df_a.copy(), df_b.copy()
    a.loc[eligible_a[:5], "cens"] = np.nan
    a.loc[eligible_a[5], "time"] = np.nan  # fails the landmark step (time > 91 d)
    b.loc[eligible_b[:5], "death"] = np.nan  # cens = death for patients without recurrence
    res = _run(tmp_path, (a, b, mapping, rules), mode="hybrid", validate=False)
    removed = res["funnel"].set_index(["source", "step"]).n_removed
    assert removed[("A", "complete covariates")] == removed[("B", "complete covariates")] == 5
    assert removed[("A", "event-free and under follow-up at day 91")] == 16
    assert res["verdict"] == "ACCEPTED" and (res["gates"][res["gates"].gate == "G9"].status != "FAIL").all()
    a.loc[eligible_a[:32], "cens"] = np.nan  # 4.7% of A (raw check passes), 5.3% of the eligible
    res = _run(tmp_path, (a, df_b, mapping, rules), mode="internal", validate=False)
    assert res["first_failing_gate"] == "G3" and "A: complete-covariates step removes more than 5%" in res["reason"]
