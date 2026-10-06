"""Human confirmation points and the audit log."""

import json
import time
from pathlib import Path


class ConsoleAsker:
    """Ask on the terminal. ``y`` accepts, ``n`` stops the run, any other text is a note the step understands."""

    def ask(self, kind, title, body, default="y"):
        print(f"\n--- {title} ---\n{body}")
        try:
            answer = input(f"[y = continue, n = stop, or type a change] (default {default}) > ").strip()
        except EOFError:
            answer = ""
        return answer or default


class AutoAsker:
    """Answer ``y`` everywhere (``--yes``). Every answer still goes to the audit log, marked as automatic."""

    def ask(self, kind, title, body, default="y"):
        return "y"


class ScriptedAsker:
    """Answers from a dict ``{kind: [answer, ...]}`` (tests); ``y`` once a list runs out."""

    def __init__(self, answers=None):
        self.answers, self.asked = {k: list(v) for k, v in (answers or {}).items()}, []

    def ask(self, kind, title, body, default="y"):
        self.asked.append((kind, title))
        queue = self.answers.get(kind, [])
        return queue.pop(0) if queue else "y"


class Audit:
    """Append-only JSON-lines log of everything the agent does: LLM calls, answers, gates, repairs."""

    def __init__(self, path):
        self.path, self.records = Path(path), []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def __call__(self, event, **payload):
        record = {"t": round(time.time(), 2), "event": event, **payload}
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)  # the run folder may have been deleted meanwhile
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def logged_llm(llm, audit):
    """Wrap ``llm(system, user) -> str`` so that every call and answer is written to the audit log."""
    if llm is None:
        return None

    def call(system, user):
        answer = llm(system, user)
        audit("llm_call", system=system[:120], user=user[:4000], answer=answer[:4000])
        return answer

    return call
