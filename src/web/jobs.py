"""Running jobs: start them, and turn the agent's log into what the job page shows (steps, gates, repairs)."""

import re
import threading
import time
import uuid

from src.agent.background import AgentJob

# (number in the agent's "== N. ..." lines, short name, what happens, does it ask the human?)
STEPS = [
    (1, "Read", "The tables and the trial record are read and profiled.", None),
    (2, "Match columns", "Jev scores the column pairs and the LLM decides the hard ones.", "mapping"),
    (3, "Eligibility", "The LLM turns the entry criteria into rules for the historical patients.", "rules"),
    (4, "Plan", "The LLM drafts the analysis plan; code checks it against the data.", None),
    (5, "Review", "A second LLM call looks for mistakes in the plan.", "plan"),
    (6, "Freeze", "The plan is hashed. From here nothing in it changes.", None),
    (7, "Gates", "Checks run in order; the trial outcomes are read once, at the end.", None),
    (8, "Repair", "A failed gate gets one fix, then the gates run again.", "repair"),
    (9, "Report", "Verdict, estimate, plain-language summary and the audit log.", None),
]
STEP_RE = re.compile(r"^== (\d+)\. ")
VERSION_RE = re.compile(r"^== 7\. Gates, design version (\d+)")
GATE_RE = re.compile(r"^\s+(G\d) (.+?): (PASS_AFTER_ACTION|PASS|FAIL|INFO)\s*$")
APPLIED_RE = re.compile(r"^\s+applied: (.+)$")
HINTS = {
    "mapping": "y = use the safe matches · n = stop · approve col1, col2 · reject col3",
    "rules": "y = apply the rules · n = stop · disable I2, I3",
    "plan": "y = freeze this plan · n = stop · or describe what to change",
    "repair": "y = apply this fix · n = skip this fix and try the next one",
}


def parse_progress(lines):
    """Current step, the gates of every design version and the repairs, read from the agent's progress lines."""
    step, versions, repairs = 0, [], []
    for line in lines:
        if m := STEP_RE.match(line):
            step = int(m.group(1))
        if m := VERSION_RE.match(line):
            versions.append({"n": int(m.group(1)), "gates": [], "failed": None})
        elif versions and (m := GATE_RE.match(line)):
            gate = {"id": m.group(1), "name": m.group(2), "status": m.group(3)}
            versions[-1]["gates"].append(gate)
            if gate["status"] == "FAIL":
                versions[-1]["failed"] = gate["id"]
        elif m := APPLIED_RE.match(line):
            repairs.append(m.group(1))
    return {"step": step, "versions": versions, "repairs": repairs}


class JobRecord:
    def __init__(self, job, mode, label, options):
        self.id = uuid.uuid4().hex[:12]
        self.job, self.mode, self.label, self.options = job, mode, label, options
        self.started = time.time()

    @property
    def status(self):
        if self.job.running:
            return "waiting" if self.job.pending else "running"
        return "failed" if self.job.error else "done"

    def snapshot(self, since=0):
        """JSON for the live page: new log lines from ``since`` on, progress, and the open question."""
        with self.job._lock:
            lines = list(self.job.log)
        pending = self.job.pending
        return {
            "id": self.id,
            "status": self.status,
            "mode": self.mode,
            "cursor": len(lines),
            "log": lines[since:],
            **parse_progress(lines),
            "pending": None if not pending else {**pending, "hint": HINTS.get(pending["kind"], "")},
            "error": self.job.error,
            "elapsed": round(time.time() - self.started),
        }


class JobRegistry:
    """All jobs of this server process (in memory; the files of a finished job stay in ``results/agent``)."""

    def __init__(self, max_running=3):
        self.max_running = max_running
        self._jobs = {}
        self._lock = threading.Lock()

    def running(self):
        return sum(1 for r in self._jobs.values() if r.job.running)

    def start(self, trial, hist, mode, label, options, **agent_kwargs):
        """Start a job. Raises ``RuntimeError`` when too many jobs run already."""
        with self._lock:
            if self.running() >= self.max_running:
                raise RuntimeError(f"{self.running()} runs are already in progress; wait for one to finish.")
            record = JobRecord(AgentJob(trial, hist, auto=mode == "single", **agent_kwargs), mode, label, options)
            self._jobs[record.id] = record
            return record

    def get(self, job_id):
        return self._jobs.get(job_id)

    def all(self):
        return sorted(self._jobs.values(), key=lambda r: -r.started)
