import asyncio
import io
import json

import httpx2
import pytest
from conftest import jev_response
from typesafe_sdk import AsyncTypeSafeClient

from src import online
from src.demo import demo_criteria, read_json
from src.harmonize import profile_columns
from src.online import (
    HarmonizeAnswer,
    extract_criteria,
    fetch_eligibility,
    jev_text_answers,
    jev_verify_rules,
    llm_json,
)
from src.rules import apply_rules


def test_jev_text_checks(jev_env):
    """Per-patient yes/no answers for a free-text column through the async client."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["state"])
        yes = "smoker" in body["state"]
        return jev_response({k: {"type": "noul", "noul": 0.95 if yes else 0.05} for k in body["questions"]})

    client = AsyncTypeSafeClient(transport=httpx2.MockTransport(handler))
    texts = ["former smoker, 20 pack years", "never smoked", "smoker"]
    questions = {"smoker": "Does the patient smoke?", "alcohol": "Heavy drinker?"}
    out = asyncio.run(jev_text_answers(client, texts, questions))
    assert [round(o["smoker"], 2) for o in out] == [0.95, 0.05, 0.95]
    assert sorted(seen) == sorted(texts)


def test_jev_verify_rules_flags_low(jev_env):
    def handler(request):
        body = json.loads(request.content)
        good = "nodes" in body["state"]["rule"]
        return jev_response({"ok": {"type": "noul", "noul": 0.97 if good else 0.4}})

    client = online.make_async_jev(httpx2.MockTransport(handler))
    items = [
        {"id": "I2", "text": "node positive", "rule": "nodes >= 1"},
        {"id": "I3", "text": "CMF", "rule": "chemo == 1"},
    ]
    s = jev_verify_rules(client, items)
    assert s["I2"] > 0.9 and s["I3"] < 0.7


def test_fetch_eligibility_mocked_urlopen(monkeypatch):
    payload = {"protocolSection": {"eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n- age >= 18"}}}
    called = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        called["url"] = req.full_url
        return Resp(json.dumps(payload).encode())

    monkeypatch.setattr(online, "urlopen", fake)
    assert fetch_eligibility(" nct00012345 ").startswith("Inclusion Criteria")
    assert called["url"] == "https://clinicaltrials.gov/api/v2/studies/NCT00012345"

    monkeypatch.setattr(online, "urlopen", lambda req, timeout=None: Resp(b'{"protocolSection": {}}'))
    with pytest.raises(ValueError):
        fetch_eligibility("NCT1")


def test_llm_json_validates_and_retries():
    answers = iter(
        ["not json", '```json\n{"match": "age", "transform": {"kind": "identity"}, "same_definition": "yes"}\n```']
    )
    prompts = []

    def llm(system, user):
        prompts.append(user)
        return next(answers)

    a = llm_json(llm, "sys", "user", HarmonizeAnswer)
    assert a.match == ["age"] and len(prompts) == 2 and "invalid" in prompts[1]
    with pytest.raises(ValueError):
        llm_json(lambda s, u: '{"transform": {"kind": "bin"}}', "s", "u", HarmonizeAnswer)  # bin needs edges


def test_extract_criteria_with_mocked_llm(raw):
    cache = read_json("llm_eligibility.json")
    crits = extract_criteria(lambda system, user: json.dumps(cache), "text", profile_columns(raw[1]))
    assert [c.kind for c in crits].count("rule") == 3 and [c.id for c in crits][:2] == ["I1", "I2"]
    assert any("approximation" in c.note.lower() for c in crits)


def test_demo_rules_give_552(raw):
    mask, funnel, invalid = apply_rules(raw[1], [c.model_dump() for c in demo_criteria() if c.kind == "rule"])
    assert mask.sum() == 552 and not invalid
    assert funnel.n_after.tolist() == [2982, 1546, 580, 552]


def test_bad_llm_rule_is_reported(raw):
    rules = [{"id": "x", "rule": "chemo", "text": "bad"}, {"id": "y", "rule": "nodes >= 1", "text": "ok"}]
    mask, funnel, invalid = apply_rules(raw[1], rules)
    assert invalid[0]["id"] == "x" and mask.sum() == (raw[1].nodes >= 1).sum()


def test_azure_status(monkeypatch):
    for k in (
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)
    assert online.llm_status() is None and online.get_llm() is None


def test_extract_criteria_reasks_invalid_rule(raw):
    """'hormon == 0 & chemo == 1' does not run (& binds before ==): the LLM is asked again with the error."""
    good = read_json("llm_eligibility.json")
    bad = json.loads(json.dumps(good))
    bad["criteria"][-1]["rule"] = "hormon == 0 & chemo == 1"
    seen = []

    def llm(system, user):
        seen.append(json.loads(user))
        return json.dumps(bad if len(seen) == 1 else good)

    crits = extract_criteria(llm, "text", profile_columns(raw[1]), raw[1])
    assert len(seen) == 2 and "C1" in seen[1]["rules_that_do_not_run"]
    assert crits[-1].rule == "hormon == 0"
