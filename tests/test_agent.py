"""The automated agent, offline: a scripted LLM, the logistic backend, no network."""

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic import make_data

from src import online
from src.agent import frames, repair, run_agent
from src.agent.gates import run_gates
from src.agent.io import AutoAsker, ScriptedAsker
from src.agent.planner import CRITIC_SYSTEM, NARRATE_SYSTEM, PLAN_SYSTEM
from src.agent.spec import Covariate, StudySpec, normalize_spec, validate_spec
from src.online import HARMONIZE_SYSTEM
from src.rules import apply_rules


class FakeLLM:
    """Answers each kind of call from a script; counts the calls."""

    def __init__(self, plans, critic=None, choose=0):
        self.plans, self.critic, self.choose = list(plans), critic or {"verdict": "ok", "concerns": []}, choose
        self.calls = {}

    def __call__(self, system, user):
        kind = self.kind(system)
        self.calls[kind] = self.calls.get(kind, 0) + 1
        if kind == "harmonize":
            req = json.loads(user)
            name = req["a_column"]["name"]
            ok = name in req["candidates"]
            return json.dumps(
                {"match": [name] if ok else None, "transform": {"kind": "identity"}, "same_definition": "yes"}
            )
        if kind == "plan":
            plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
            return json.dumps(plan)
        if kind == "critic":
            return json.dumps(self.critic)
        if kind == "choose":
            return json.dumps({"choice": self.choose, "reason": "scripted"})
        return "Scripted summary."

    @staticmethod
    def kind(system):
        for kind, text in [
            ("harmonize", HARMONIZE_SYSTEM),
            ("plan", PLAN_SYSTEM),
            ("critic", CRITIC_SYSTEM),
            ("narrate", NARRATE_SYSTEM),
        ]:
            if system == text:
                return kind
        return "choose" if "repair an analysis" in system else "other"


def plan_for(horizon=4.0, covariates=None):
    covs = covariates or [
        ("age", "numeric"),
        ("sex", "binary"),
        ("grade", "ordinal"),
        ("marker_a", "numeric"),
        ("marker_b", "numeric"),
    ]
    return {
        "treatment": "arm",
        "control_value": "control",
        "time": "time",
        "event": "status",
        "event_values": ["2"],
        "time_unit": "days",
        "horizon_years": horizon,
        "landmark_days": 0,
        "covariates": [
            {"name": n, "kind": k, "cross_source": True, "essential": n in ("age", "marker_a")} for n, k in covs
        ],
        "rationale": "test plan",
    }


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(online, "jev_status", lambda: None)


@pytest.fixture(scope="module")
def data():
    return make_data()


def run(data, llm, asker=None, **kw):
    trial, hist = data
    return run_agent(
        trial,
        hist,
        llm=llm,
        asker=asker or AutoAsker(),
        backend="logreg",
        cache_dir=None,
        say=lambda m: None,
        **kw,
    )


# ----------------------------------------------------------------------------
# Spec validation (code, not LLM)
# ----------------------------------------------------------------------------
def test_validate_catches_bad_columns_and_values(data):
    trial, hist = data
    good = StudySpec(**plan_for())
    assert validate_spec(good, trial, hist)[0] == []
    bad_col = good.model_copy(update={"time": "no_such_column"})
    assert "not in the trial table" in validate_spec(bad_col, trial, hist)[0][0]
    bad_control = good.model_copy(update={"control_value": "3"})
    assert "control_value" in validate_spec(bad_control, trial, hist)[0][0]
    bad_event = good.model_copy(update={"event_values": ["7"]})
    assert "never occur" in validate_spec(bad_event, trial, hist)[0][0]
    leak = good.model_copy(update={"covariates": [*good.covariates, Covariate(name="status", kind="binary")]})
    assert any("treatment, time or event" in e for e in validate_spec(leak, trial, hist)[0])


def test_validate_catches_a_covariate_that_leaks_treatment(data):
    trial, hist = data
    trial = trial.assign(
        leak=(trial["arm"] == "drug").astype(float) + np.random.default_rng(0).normal(0, 0.01, len(trial))
    )
    spec = StudySpec(**plan_for(covariates=[("age", "numeric"), ("leak", "numeric")]))
    assert any("separates treated from control" in e for e in validate_spec(spec, trial, hist)[0])


def test_normalize_marks_historical_gaps_not_comparable(data):
    trial, hist = data
    spec = StudySpec(**plan_for(covariates=[("age", "numeric"), ("marker_a", "numeric"), ("lab", "numeric")]))
    fixed, notes = normalize_spec(spec, trial, hist)
    assert [c.cross_source for c in fixed.covariates] == [True, True, False]
    assert "lab" in notes[0]


# ----------------------------------------------------------------------------
# Data layer: no study-specific names
# ----------------------------------------------------------------------------
def test_design_encodes_each_kind(data):
    trial, hist = data
    spec = StudySpec(**plan_for(covariates=[("age", "numeric"), ("sex", "binary"), ("grade", "ordinal")]))
    state = frames.new_state(spec, hybrid=True)
    study = frames.build_study(spec, trial, hist, state)
    D, names, n_main = frames.design(study["encs"], study["T"], squares=True, products=True)
    assert names[:3] == ["age", "sex=m", "grade"] and n_main == 3
    assert "age^2" in names and "age x grade" in names
    assert set(np.unique(D[:, 1])) <= {0.0, 1.0}
    F, cat_idx = frames.features(study["enc_s"], study["T"])
    assert F.shape == (len(study["T"]), 3) and cat_idx == [1]


def test_complete_cases_and_landmark_are_in_the_funnel(data):
    trial, hist = data
    plan = plan_for(covariates=[("age", "numeric"), ("marker_a", "numeric"), ("marker_b", "numeric")])
    spec = StudySpec(**{**plan, "landmark_days": 100})
    study = frames.build_study(spec, trial, hist, frames.new_state(spec, hybrid=True))
    steps = dict(study["funnel"])
    assert steps["trial: all patients"] == len(trial)
    assert steps["trial: event-free and followed at day 100"] < len(trial)
    assert (study["T"]["__time"] > 100).all()


# ----------------------------------------------------------------------------
# Gates and repair
# ----------------------------------------------------------------------------
def gate_ctx():
    return {"backend": "logreg", "cache_dir": None, "say": lambda m: None, "audit": lambda e, **k: None}


def test_horizon_gate_fails_then_a_shorter_horizon_passes(data):
    trial, hist = data
    spec = StudySpec(**plan_for(horizon=4.0))
    spec, _ = normalize_spec(spec, trial, hist)
    state = frames.new_state(spec, hybrid=True)
    out = run_gates(spec, trial, hist, state, gate_ctx())
    assert out["failed"]["id"] == "G5" and out["estimate"] is None
    years = out["failed"]["diag"]["max_supported_years"]
    assert 3 < years < 4
    options = repair.build_options(out["failed"], spec, state, [], out["study"])
    assert options[0]["action"] == "shorten_horizon" and options[0]["args"]["years"] == pytest.approx(years, abs=0.01)
    assert options[-1]["action"] == "drop_borrowing"
    fixed = repair.apply_option(state, options[0])
    assert fixed["horizon_years"] == options[0]["args"]["years"] and state["horizon_years"] == 4.0
    out2 = run_gates(spec, trial, hist, fixed, gate_ctx())
    assert out2["failed"] is None and out2["estimate"] is not None
    assert [g["id"] for g in out2["gates"]] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]


def test_a_tried_option_is_not_offered_again(data):
    trial, hist = data
    spec, _ = normalize_spec(StudySpec(**plan_for(horizon=4.0)), trial, hist)
    state = frames.new_state(spec, hybrid=True)
    out = run_gates(spec, trial, hist, state, gate_ctx())
    first = repair.build_options(out["failed"], spec, state, [], out["study"])[0]
    state["tried"].append({"action": first["action"], "args": first["args"], "applied": False})
    again = repair.build_options(out["failed"], spec, state, [], out["study"])
    assert first["action"] not in [o["action"] for o in again]


def test_essential_covariates_are_never_offered_for_dropping(data):
    trial, hist = data
    spec, _ = normalize_spec(StudySpec(**plan_for()), trial, hist)
    state = frames.new_state(spec, hybrid=True)
    study = frames.build_study(spec, trial, hist, state)
    failed = {
        "id": "G3",
        "name": "x",
        "diag": {"worst_terms": {}, "smd_before": {"age": 0.9, "marker_a": 0.8, "grade": 0.7}},
    }
    options = repair.build_options(failed, spec, state, [], study)
    dropped = [o["args"]["name"] for o in options if o["action"] == "drop_covariate"]
    assert dropped and not {"age", "marker_a"} & set(dropped)


def test_rule_relaxation_returns_the_patients(data):
    trial, hist = data
    rules = [{"id": "I1", "text": "young", "rule": "age <= 40"}]
    mask, funnel, _ = apply_rules(hist, rules)
    assert mask.sum() < len(hist)
    mask2, _, _ = apply_rules(hist, [])
    assert mask2.sum() == len(hist)


# ----------------------------------------------------------------------------
# The whole agent
# ----------------------------------------------------------------------------
def test_agent_repairs_the_horizon_and_is_accepted(data, tmp_path):
    llm = FakeLLM([plan_for(horizon=4.0)])
    result = run(data, llm, out_dir=tmp_path)
    assert result["verdict"] == "ACCEPTED" and result["tier"] == "SUPPORTIVE"
    assert [t["action"] for t in result["repairs"]] == ["shorten_horizon"]
    assert [a["failed"] for a in result["attempts"]] == ["G5", None]
    assert result["attempts"][0]["estimate"] is None  # the failed design never reached the outcomes
    assert 3 < result["primary"]["horizon_years"] < 4
    assert result["plan_by"].startswith("AI-drafted, auto-confirmed")
    assert result["prognostic_label"].endswith("(NOT TabPFN)")
    assert llm.calls["plan"] == 1 and llm.calls["critic"] == 1 and llm.calls["choose"] == 1
    run_dir = Path(result["run_dir"])
    for name in ("report.md", "result.json", "audit.jsonl"):
        assert (run_dir / name).exists()
    events = [json.loads(line)["event"] for line in (run_dir / "audit.jsonl").read_text().splitlines()]
    assert events.index("plan_frozen") < events.index("unblind_trial_outcomes") < events.index("end")
    assert "repair" in events and "human" in events


def test_the_repair_choice_never_sees_the_estimate(data, tmp_path):
    llm = FakeLLM([plan_for(horizon=4.0)])
    seen = []
    inner = llm.__call__

    def spy(system, user):
        if "repair an analysis" in system:
            seen.append(user)
        return inner(system, user)

    run(data, spy, out_dir=tmp_path)
    assert seen and all("estimate" not in u and '"est"' not in u and "rr_lo" not in u for u in seen)


def test_invalid_plan_is_sent_back_with_the_errors(data, tmp_path):
    bad = {**plan_for(), "control_value": "9"}
    llm = FakeLLM([bad, plan_for(horizon=3.0)])
    result = run(data, llm, out_dir=tmp_path)
    assert llm.calls["plan"] == 2 and result["verdict"] == "ACCEPTED"
    audit = [json.loads(line) for line in (Path(result["run_dir"]) / "audit.jsonl").read_text().splitlines()]
    drafts = [r for r in audit if r["event"] == "plan_draft"]
    assert drafts[0]["errors"] and not drafts[1]["errors"]


def test_a_plan_that_never_validates_ends_not_accepted(data, tmp_path):
    llm = FakeLLM([{**plan_for(), "treatment": "no_such_column"}])
    result = run(data, llm, out_dir=tmp_path)
    assert result["verdict"] == "NOT ACCEPTED" and "plan could not be made valid" in result["reason"]
    assert llm.calls["plan"] == 3


def test_critic_concern_triggers_exactly_one_revision(data, tmp_path):
    critic = {
        "verdict": "revise",
        "concerns": [{"issue": "add copper", "severity": "high", "fix": "add_covariate", "column": "copper"}],
    }
    llm = FakeLLM([plan_for(horizon=3.0)], critic=critic)
    result = run(data, llm, out_dir=tmp_path)
    assert llm.calls["plan"] == 2 and llm.calls["critic"] == 1
    assert result["critic"]["concerns"][0]["column"] == "copper"


def test_human_can_stop_at_each_confirmation_point(data, tmp_path):
    for kind in ("mapping", "plan"):
        llm = FakeLLM([plan_for(horizon=3.0)])
        result = run(data, llm, asker=ScriptedAsker({kind: ["n"]}), out_dir=tmp_path)
        assert result["verdict"] == "NOT ACCEPTED" and "stopped by the user" in result["reason"]
        assert "primary" not in result


def test_human_feedback_goes_back_to_the_planner(data, tmp_path):
    llm = FakeLLM([plan_for(horizon=3.0)])
    asker = ScriptedAsker({"plan": ["please also adjust for platelet"]})
    result = run(data, llm, asker=asker, out_dir=tmp_path)
    assert llm.calls["plan"] == 2 and result["verdict"] == "ACCEPTED"
    assert [c["kind"] for c in result["confirmations"]].count("plan") == 2


def test_declining_every_repair_ends_not_accepted(data, tmp_path):
    llm = FakeLLM([plan_for(horizon=4.0)])
    result = run(data, llm, asker=ScriptedAsker({"repair": ["n"] * 5}), out_dir=tmp_path)
    assert result["verdict"] == "NOT ACCEPTED" and result["first_failing_gate"] == "G5"
    assert result["repairs"] == [] and "no allowed fix left" in result["reason"]


def test_repair_cap_stops_the_loop(data, tmp_path):
    llm = FakeLLM([plan_for(horizon=4.0)])
    result = run(data, llm, max_repairs=0, out_dir=tmp_path)
    assert result["verdict"] == "NOT ACCEPTED" and result["first_failing_gate"] == "G5"


def test_mapping_flags_a_column_whose_definition_differs(data, tmp_path):
    class Differing(FakeLLM):
        def __call__(self, system, user):
            if self.kind(system) == "harmonize" and json.loads(user)["a_column"]["name"] == "grade":
                return json.dumps({"match": ["grade"], "same_definition": "no", "definition_note": "different grading"})
            return super().__call__(system, user)

    asker = ScriptedAsker()
    result = run(data, Differing([plan_for(horizon=3.0)]), asker=asker, out_dir=tmp_path)
    assert result["verdict"] == "ACCEPTED"
    grade = [c for c in result["spec"]["covariates"] if c["name"] == "grade"]
    assert grade and grade[0]["cross_source"] is False  # not approved, so not used for borrowing


def test_agent_needs_an_llm(data, monkeypatch):
    monkeypatch.setattr(online, "get_llm", lambda: None)
    trial, hist = data
    with pytest.raises(RuntimeError, match="needs an LLM"):
        run_agent(trial, hist, asker=AutoAsker(), say=lambda m: None)


def test_narrative_headings_are_removed():
    from src.agent.planner import clean_narrative

    assert clean_narrative("# Summary\n\nFirst paragraph.\n## Result\nSecond.") == "First paragraph.\nSecond."


def test_a_deleted_run_folder_does_not_crash_the_run(data, tmp_path):
    import shutil

    class Deleting(FakeLLM):
        def __call__(self, system, user):
            if self.kind(system) == "critic":  # in the middle of the run
                for d in tmp_path.glob("run_*"):
                    shutil.rmtree(d)
            return super().__call__(system, user)

    result = run(data, Deleting([plan_for(horizon=3.0)]), out_dir=tmp_path)
    assert result["verdict"] == "ACCEPTED" and (Path(result["run_dir"]) / "report.md").exists()


def test_unusable_historical_data_is_a_visible_warning_not_a_silent_switch(data, tmp_path):
    trial, hist = data
    result = run((trial, hist.drop(columns=["time", "status"])), FakeLLM([plan_for(horizon=3.0)]), out_dir=tmp_path)
    assert (
        result["mode"] == "trial only" and "status" in result["borrowing_note"] and "time" in result["borrowing_note"]
    )
    assert "**Warning.**" in result["text"] and "The result uses the trial alone" in result["text"]
    ok = run(data, FakeLLM([plan_for(horizon=3.0)]), out_dir=tmp_path)
    assert ok["borrowing_note"] is None and "Warning" not in ok["text"]
