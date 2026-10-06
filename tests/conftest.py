import json
import sys
from pathlib import Path

import httpx2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo import TRUTH, demo_pipeline, load_demo_data  # noqa: E402


@pytest.fixture(scope="session")
def demo():
    return demo_pipeline()


@pytest.fixture(scope="session")
def raw():
    return load_demo_data()


@pytest.fixture(scope="session")
def truth():
    return json.loads(TRUTH.read_text())


def jev_response(answers):
    return httpx2.Response(
        200, json={"model": "jev-test", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}
    )


@pytest.fixture
def jev_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def fake_jev_transport(score_fn):
    """MockTransport at /v1/systemone; ``score_fn(a_col, b_col) -> P(same)`` answers pairwise Noul questions."""

    def handler(request):
        assert request.url.path == "/v1/systemone"
        body = json.loads(request.content)
        a = body["state"]["column_A (new trial)"]["name"]
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": score_fn(a, key)}
            else:  # choice (ranking mode)
                probs = {lab: score_fn(a, lab) if lab != "none" else 0.01 for lab in q["criteria"]}
                tot = sum(probs.values())
                probs = {k: v / tot for k, v in probs.items()}
                top = max(probs, key=probs.get)
                answers[key] = {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs}
        return jev_response(answers)

    return httpx2.MockTransport(handler)
