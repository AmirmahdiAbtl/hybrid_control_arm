"""Protocol constants, digests, eligible frames, design actions and G3 checks of the gated analysis (offline)."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import causal as C
from src import gated
from src.demo import APPROVED, DEMO, PATH_A, PATH_B
from src.gated import design, outcomes, protocol, report, runlog
from src.harmonize import load_mapping


@pytest.fixture(scope="module")
def inputs():
    """Raw demo inputs as the CLI reads them: data, approved mapping, Stage-2 rules."""
    rules = [c for c in json.loads((DEMO / "llm_eligibility.json").read_text())["criteria"] if c["kind"] == "rule"]
    return pd.read_csv(PATH_A), pd.read_csv(PATH_B), load_mapping(APPROVED), rules


def _run(tmp_path, df_a, df_b, mapping, rules, **kw):
    return gated.run_gated(
        df_a, df_b, mapping, rules, backend="logreg", out_dir=str(tmp_path), cache_dir=None, validate=False, **kw
    )


def test_protocol_constants_match_the_sap():
    G = protocol.GATES
    assert G["G4H_ess_min_abs"] == 20 and G["G6_ratio_cut"] == 20 and G["G8_margin"] == 0.05
    assert G["G9_G_min"] == 0.70 and G["G11_named_rr"] == (2.0, 1.5) and G["max_design_actions"] == 2
    assert protocol.MAX_TABPFN_FITS == 60 and protocol.N_REAL == 60 and protocol.SEED == 0 and protocol.K_FOLDS == 5
    assert protocol.LANDMARK_DAYS == 91 and protocol.AGE_MAX == 65 and protocol.HORIZON_YEARS["primary"] == 2.0
    assert len(protocol.TARGET_TRIAL) == 9
    cells = [v for comp in protocol.TARGET_TRIAL.values() for v in comp.values()]
    assert len(cells) == 36 and all(isinstance(v, str) and v.strip() for v in cells)
    assert all(set(c) == {"target", "emulation_A", "emulation_B", "deviation"} for c in protocol.TARGET_TRIAL.values())
    for mode, scopes in protocol.ESTIMAND_ROLES.items():
        assert sum(r == "primary" for sc in scopes.values() for r in sc.values()) == 1, mode
    assert protocol.ESTIMAND_ROLES["internal"]["R"]["ATT"] == "primary" == protocol.ESTIMAND_ROLES["hybrid"]["H"]["ATT"]
    assert [r["id"] for r in protocol.ELIGIBILITY] == [f"E{i}" for i in range(1, 10)]
    assert all(r["status"] in protocol.ELIGIBILITY_STATUSES for r in protocol.ELIGIBILITY)
    assert set(protocol.ESTIMANDS) == {"ATT", "ATE", "ATC", "ATO", "CATE"}
    assert all(str(e[f]).strip() for e in protocol.ESTIMANDS.values() for f in protocol.ESTIMAND_FIELDS)
    assert [s["id"] for s in protocol.SENSITIVITY] == ["S1", "S2", "S3", "S4", "S5", "S6", "X1", "X2", "X3", "X4"]
    assert [s["in_sign_rule"] for s in protocol.SENSITIVITY] == [True] * 6 + [False] * 4
    assert len(protocol.DAG_EDGES) == 54 and sum(e[3] is not None for e in protocol.DAG_EDGES) == 11
    assert {e[3] for e in protocol.DAG_EDGES if e[3]} <= set(protocol.ASSUMPTIONS)
    # the gated package uses the single copy of the node -> column maps, gate_frames and benchmark_variables
    assert protocol.TRIAL_COLUMNS is C.TRIAL_COLUMNS and protocol.SOURCE_COLUMNS is C.SOURCE_COLUMNS
    assert design.gate_frames is C.gate_frames and outcomes.benchmark_variables is C.benchmark_variables
    assert "{outcome}" in protocol.FOOTER and "{prognostic}" in protocol.FOOTER


def test_expected_fit_budget():
    assert runlog.expected_tabpfn_fits("hybrid") == 32 and runlog.expected_tabpfn_fits("hybrid", True) == 48
    assert runlog.expected_tabpfn_fits("hybrid", demo=False) == 22 and runlog.expected_tabpfn_fits("internal") == 20
    assert max(runlog.expected_tabpfn_fits(m, True) for m in ("hybrid", "internal")) <= protocol.MAX_TABPFN_FITS


def test_protocol_digest_is_stable_and_tracks_every_threshold(monkeypatch):
    d0 = runlog.protocol_digest()
    assert d0 == runlog.protocol_digest() and len(d0) == 64
    monkeypatch.setitem(protocol.GATES, "G8_margin", 0.06)
    d1 = runlog.protocol_digest()
    assert d1 != d0
    monkeypatch.setattr(protocol, "MAX_TABPFN_FITS", 61)
    assert runlog.protocol_digest() not in (d0, d1)
    monkeypatch.undo()
    assert runlog.protocol_digest() == d0
    assert len(runlog.code_digest()) == 64 and runlog.frame_digest(pd.DataFrame({"a": [1]})) != runlog.frame_digest(
        pd.DataFrame({"a": [2]})
    )


def test_gate_frames_on_the_demo(demo):
    fr = design.eligible_frames(demo, demo["df_a"], demo["df_b"])
    f = fr["funnel"]
    a, b = f[f.source == "A"], f[f.source == "B"]
    assert a.n_after.tolist() == [686, 620, 605, 605] and b.n_after.tolist() == [552, 550, 547, 547]
    assert a.step.tolist()[-1] == b.step.tolist()[-1] == "complete covariates" and f.n_removed.iloc[[3, 7]].sum() == 0
    assert (a.iloc[-1].n_treated, a.iloc[-1].n_control) == (205, 400)
    T, B = fr["trial_cov"], fr["hist_cov"]
    assert not {"time", "cens"} & (set(T.columns) | set(B.columns))
    assert fr["landmark_ok"] and len(fr["outcomes"]["trial"]) == len(T) and len(fr["outcomes"]["hist"]) == len(B)
    assert list(T.index) == list(range(len(T))) and T.orig_index.is_monotonic_increasing


def test_complete_covariates_step_drops_and_logs_missing_rows(demo):
    """A missing covariate removes the patient in a logged funnel step; nothing is imputed."""
    df_a = demo["df_a"].copy()
    df_a.loc[[3, 10], "pnodes"] = np.nan
    fr = design.eligible_frames(demo, df_a, demo["df_b"])
    cc = fr["funnel"].set_index(["source", "step"]).loc[("A", "complete covariates")]
    assert cc.n_removed == int(df_a.loc[[3, 10]].pipe(lambda x: (x.age <= 65) & (x.time > 91)).sum())
    assert fr["trial_cov"].pnodes.notna().all() and len(fr["outcomes"]["trial"]) == len(fr["trial_cov"])


def test_apply_action_refuses_after_unblinding_reuse_and_the_cap(tmp_path, inputs):
    ctx = runlog.start_run(tmp_path, "internal", "logreg", *inputs)
    runlog.apply_action(ctx, "B5.1")
    with pytest.raises(RuntimeError, match="already used"):
        runlog.apply_action(ctx, "B5.1")
    runlog.apply_action(ctx, "B4.1")
    with pytest.raises(RuntimeError, match="2 design actions"):
        runlog.apply_action(ctx, "B7.1")
    ctx2 = runlog.start_run(tmp_path, "internal", "logreg", *inputs)
    ctx2["unblinded"] = True
    with pytest.raises(RuntimeError, match="unblinded"):
        runlog.apply_action(ctx2, "B7.1")
    with pytest.raises(ValueError):
        runlog.apply_action(ctx, "B9.9")
    events = [json.loads(x) for x in ctx["audit_path"].read_text().splitlines()]
    assert [e["id"] for e in events if e["event"] == "action"] == ["B5.1", "B4.1"]
    assert ctx["actions"] == ["B5.1", "B4.1"] and ctx2["actions"] == []
    assert ctx["run_dir"].name == "run_001_internal" and ctx2["run_dir"].name == "run_002_internal"


def test_source_support_on_the_demo_against_the_g6_thresholds(demo):
    fr = design.eligible_frames(demo, demo["df_a"], demo["df_b"])
    T, B = fr["trial_cov"], fr["hist_cov"]
    trt = (T["A"] == 1).to_numpy()
    G = protocol.GATES
    sup = C.source_support(
        C.design_source(T[trt]),
        C.design_source(B),
        np.where(T["post"][trt] == 1, "post", "pre"),
        np.where(B["post"] == 1, "post", "pre"),
        seed=protocol.SEED,
        k=protocol.K_FOLDS,
        cut=G["G6_ratio_cut"],
    )
    tab = sup["table"].set_index("stratum")
    assert 0.80 <= sup["auc"] <= 0.87
    passes = (
        (tab.share_unsupported <= G["G6_share_unsupported_max"])
        & (tab.ess_B >= G["G6_ess_min"])
        & (tab.max_share <= G["G6_weight_share_max"])
    )
    assert tab.loc["all", "share_unsupported"] <= 0.10 and passes["all"]
    assert 35 <= tab.loc["post", "ess_B"] < 50 and not passes["post"]  # post FAIL (ESS)
    assert tab.loc["pre", "ess_B"] > 200 and passes["pre"]


def _bad_inputs(inputs, case):
    df_a, df_b, mapping, rules = inputs
    df_a, mapping, rules = df_a.copy(), copy.deepcopy(mapping), copy.deepcopy(rules)
    if case == "A tgrade":
        df_a.loc[0, "tgrade"] = "IV"
    elif case == "A menostat":
        df_a.loc[0, "menostat"] = "post"
    elif case == "treatment missing":
        df_a.loc[0, "horTh"] = np.nan
    elif case == "treatment labels":
        df_a.loc[0, "horTh"] = "maybe"
    elif case == "B menostat":
        mapping["menostat"]["transform"]["mapping"] = {"0": "pre", "1": "post"}
    elif case == "tsize groups":
        mapping["tsize"]["transform"]["labels"] = ["small", "medium", "large"]
    elif case == "B time":
        mapping["time"]["status"] = "rejected"
    elif case == "rule":
        rules = [r for r in rules if r["column"] != "chemo"]
    elif case == "missing > 5%":
        df_a.loc[:59, "progrec"] = np.nan
    return df_a, df_b, mapping, rules


@pytest.mark.parametrize(
    "case, words",
    [
        ("A tgrade", "A tgrade: unknown labels ['IV']"),
        ("A menostat", "A menostat: unknown labels ['post']"),
        ("treatment missing", "A horTh: 1 missing values"),
        ("treatment labels", "A horTh: labels ['maybe', 'no', 'yes']"),
        ("B menostat", "B menostat: unknown labels ['post', 'pre']"),
        ("tsize groups", "A tsize: unknown labels"),
        ("B time", "mapping time: not approved"),
        ("rule", "no valid Stage-2 rule on 'chemo'"),
        ("missing > 5%", "A missing values above the 5% limit (progrec)"),
    ],
)
def test_g3_fails_with_a_clear_message(tmp_path, inputs, case, words):
    res = _run(tmp_path, *_bad_inputs(inputs, case), mode="hybrid")
    assert (res["verdict"], res["tier"], res["first_failing_gate"]) == ("NOT ACCEPTED", "NOT INTERPRETABLE", "G3")
    g3 = res["gates"][res["gates"].gate == "G3"].iloc[0]
    assert words in g3.note or words in res["reason"], (g3.note, res["reason"])
    assert res["primary"] is None and len(res["estimates"]) == 0 and "points" not in res["text"].splitlines()[0]
    events = [json.loads(x)["event"] for x in (Path(res["run_dir"]) / "audit.jsonl").read_text().splitlines()]
    assert "unblind" not in events and "read_external_outcomes" not in events  # outcomes never read


def test_treatment_column_missing_stops_at_g3(tmp_path, inputs):
    df_a, df_b, mapping, rules = inputs
    res = _run(tmp_path, df_a.drop(columns=["horTh"]), df_b, mapping, rules, mode="internal")
    assert res["first_failing_gate"] == "G3" and "treatment column 'horTh' missing" in res["reason"]


def test_gate_line_and_check_text():
    row = {
        "gate": "G4",
        "name": "within-trial positivity",
        "scope": "H",
        "attempt": 1,
        "status": "PASS",
        "checks": [
            ("ess", 24.48, ">=", 20.0, True),
            ("max_share", 0.0942, "<=", 0.15, True),
            ("cal_slope", 0.88, "report", None, None),
            ("ess_B", 41.8, ">=", 50.0, False),
        ],
        "action": None,
        "note": "",
    }
    assert runlog.gate_line(row) == (
        "G4  within-trial positivity  [H]  attempt 1  PASS  ess 24.48 >= 20; max_share "
        "0.0942 <= 0.15; cal_slope 0.88 (report); ess_B 41.8 >= 50 -> no"
    )


def test_not_interpretable_text_has_no_numbers():
    res = {
        "verdict": "NOT ACCEPTED",
        "tier": "NOT INTERPRETABLE",
        "backend": "logreg",
        "mode": "hybrid",
        "first_failing_gate": "G4",
        "reason": "hybrid not estimable for this sample",
        "primary": None,
        "labels": {"outcome": "x (NOT TabPFN)", "prognostic": "y (NOT TabPFN)", "weights": "w", "cheap": "c"},
        "g8": {"status": "INFEASIBLE", "mdm": 0.131},
    }
    text = report.verdict_text(res)
    assert text.startswith("NOT ACCEPTED (NOT INTERPRETABLE) - No treatment-effect estimate is reported because G4")
    assert not any(ch.isdigit() for ch in text.split("\n")[0].replace("G4", "")) and "13.1" not in text
    assert text.splitlines()[-1].startswith("Evidence type: non-randomized comparison.")
