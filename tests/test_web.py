"""The Flask app: pages, docs, and both run modes through the HTTP endpoints (scripted LLM, no network)."""

import io
import json
import time

import pytest
from synthetic import LABELS, make_data
from test_agent import FakeLLM, plan_for

from src import online
from src.web import create_app
from src.web.jobs import parse_progress


@pytest.fixture
def make_app(monkeypatch, tmp_path):
    monkeypatch.setattr(online, "jev_status", lambda: None)

    def make(llm=None, **config):
        app = create_app({"RESULTS_DIR": str(tmp_path), "LLM": llm, "TESTING": True, **config})
        return app, app.test_client()

    return make


TRIAL, HIST = make_data()


def csv_file(df, name):
    return io.BytesIO(df.to_csv(index=False).encode()), name


def form(files=True, **kw):
    """Form fields of a run. The default data are the generated tables, uploaded as CSV files."""
    fields = {"mode": "single", "data": "upload", "backend": "logreg", "nct": "", "eligibility": "", "n_real": "0",
              "max_repairs": "6", **kw}  # fmt: skip
    if files and fields["data"] == "upload":
        fields.setdefault("trial", csv_file(TRIAL, "trial.csv"))
        fields.setdefault("hist", csv_file(HIST, "hist.csv"))
        fields.setdefault("trial_dict", (io.BytesIO(json.dumps(LABELS).encode()), "trial.json"))
    return fields


def start(client, **kw):
    r = client.post("/run", data=form(**kw), content_type="multipart/form-data")
    assert r.status_code == 303, r.get_data(as_text=True)[:600]
    return r.headers["Location"].rsplit("/", 1)[-1]


def snapshot(client, job_id):
    return client.get(f"/api/jobs/{job_id}").get_json()


def wait_for(client, job_id, cond, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        s = snapshot(client, job_id)
        if cond(s):
            return s
        time.sleep(0.05)
    raise AssertionError(f"timed out; last status {s['status']}, step {s['step']}")


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def test_landing_page_has_the_sections_and_the_brand(make_app):
    _, c = make_app(FakeLLM([plan_for()]))
    html = c.get("/").get_data(as_text=True)
    for text in (
        "A Control Arm You Can",
        "See how a run works",
        "Built so you can",
        "Run on real data",
        "Two ways to run it",
    ):
        assert text in html
    assert "logo-dark.png" in html and "InSilicoTrials" in html
    assert "Run in one loop" in html and "Run step by step" in html
    for path in (
        "/static/css/site.css",
        "/static/img/logo-dark.png",
        "/static/img/logo-white.png",
        "/static/js/landing.js",
    ):
        assert c.get(path).status_code == 200, path


def test_run_page_preselects_the_mode_and_lists_the_examples(make_app):
    _, c = make_app(FakeLLM([plan_for()]))
    html = c.get("/run?mode=step").get_data(as_text=True)
    assert 'value="step" checked' in html and "GBSG-2" in html and "Upload your own" in html


def test_run_page_without_an_llm_cannot_start(make_app, monkeypatch):
    monkeypatch.setattr(online, "llm_status", lambda: None)
    _, c = make_app(None)
    html = c.get("/run").get_data(as_text=True)
    assert "The LLM is not configured" in html and "disabled" in html
    r = c.post("/run", data=form(), content_type="multipart/form-data")
    assert r.status_code == 200 and "not configured" in r.get_data(as_text=True)


def test_docs_render_with_a_table_of_contents(make_app):
    _, c = make_app()
    assert c.get("/docs").status_code == 302
    html = c.get("/docs/agent").get_data(as_text=True)
    assert 'id="the-gates"' in html and 'href="#the-gates"' in html and "<table>" in html and "Single loop" in html
    assert c.get("/docs/sap").status_code == 404
    assert c.get("/docs/nothing").status_code == 404


def test_unknown_job_is_404(make_app):
    _, c = make_app()
    for path in ("/jobs/nope", "/api/jobs/nope", "/jobs/nope/files/report.md"):
        assert c.get(path).status_code == 404
    assert c.post("/api/jobs/nope/answer", json={"id": 1}).status_code == 404


# ----------------------------------------------------------------------------
# Single loop
# ----------------------------------------------------------------------------
def test_single_loop_runs_to_a_result_and_serves_the_files(make_app):
    _, c = make_app(FakeLLM([plan_for(horizon=4.0)]))
    job = start(c, mode="single")
    s = wait_for(c, job, lambda s: s["status"] in ("done", "failed"))
    assert s["status"] == "done" and s["pending"] is None and s["step"] == 9
    assert [v["failed"] for v in s["versions"]] == ["G5", None] and len(s["repairs"]) == 1
    page = c.get(f"/jobs/{job}").get_data(as_text=True)
    assert "ACCEPTED" in page and "shorten" in page.lower() and "Repairs (1)" in page and "auto-confirmed" in page
    for name, start_text in (("report.md", "# Hybrid control arm"), ("result.json", "{"), ("audit.jsonl", "{")):
        r = c.get(f"/jobs/{job}/files/{name}")
        assert r.status_code == 200 and r.get_data(as_text=True).startswith(start_text)
        assert "attachment" in r.headers["Content-Disposition"]
    assert c.get(f"/jobs/{job}/files/secret.txt").status_code == 404
    assert c.get(f"/jobs/{job}/files/..%2Fresult.json").status_code == 404


# ----------------------------------------------------------------------------
# Step by step
# ----------------------------------------------------------------------------
def test_step_by_step_waits_for_each_answer(make_app):
    _, c = make_app(FakeLLM([plan_for(horizon=4.0)]))
    job = start(c, mode="step")
    asked = []
    while True:
        s = wait_for(c, job, lambda s: s["status"] in ("waiting", "done", "failed"))
        if s["status"] != "waiting":
            break
        q = s["pending"]
        asked.append(q["kind"])
        assert q["hint"] and q["body"] and s["status"] == "waiting"
        assert c.post(f"/api/jobs/{job}/answer", json={"id": q["id"] + 5, "answer": "y"}).status_code == 409
        assert c.post(f"/api/jobs/{job}/answer", json={"id": q["id"], "answer": "y"}).get_json() == {"ok": True}
    assert asked == ["mapping", "plan", "repair"] and s["status"] == "done"
    assert "confirmed by a human" in c.get(f"/jobs/{job}").get_data(as_text=True)


def test_a_typed_change_reaches_the_planner(make_app):
    llm = FakeLLM([plan_for(horizon=3.0)])
    _, c = make_app(llm)
    job = start(c, mode="step")
    answers = {"mapping": "y", "plan": "also adjust for platelet"}
    seen = set()
    while True:
        s = wait_for(c, job, lambda s: s["status"] in ("waiting", "done", "failed"))
        if s["status"] != "waiting":
            break
        q = s["pending"]
        reply = answers.get(q["kind"], "y") if q["kind"] not in seen else "y"
        seen.add(q["kind"])
        c.post(f"/api/jobs/{job}/answer", json={"id": q["id"], "answer": reply})
    assert llm.calls["plan"] == 2 and s["status"] == "done"


def test_stopping_ends_the_run_not_accepted(make_app):
    _, c = make_app(FakeLLM([plan_for(horizon=3.0)]))
    job = start(c, mode="step")
    s = wait_for(c, job, lambda s: s["status"] == "waiting")
    assert c.post(f"/api/jobs/{job}/cancel").get_json() == {"ok": True}
    s = wait_for(c, job, lambda s: s["status"] in ("done", "failed"))
    page = c.get(f"/jobs/{job}").get_data(as_text=True)
    assert s["status"] == "done" and "NOT ACCEPTED" in page and "stopped by the user" in page


# ----------------------------------------------------------------------------
# The form
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "changes, message",
    [
        ({"nct": "NCT12"}, "ClinicalTrials.gov id"),
        ({"backend": "standin"}, "Unknown model backend"),
        ({"mode": "fast"}, "single loop or step by step"),
        ({"max_repairs": "99"}, "between 0 and 8"),
        ({"n_real": "abc"}, "whole number"),
        ({"data": "upload", "files": False}, "Upload the trial table"),
    ],
)
def test_bad_form_values_are_explained_and_start_nothing(make_app, changes, message):
    app, c = make_app(FakeLLM([plan_for()]))
    r = c.post("/run", data=form(**changes), content_type="multipart/form-data")
    assert r.status_code == 200 and message in r.get_data(as_text=True)
    assert app.extensions["jobs"].all() == []


def test_uploaded_csvs_are_used_and_the_filename_is_escaped(make_app):
    _, c = make_app(FakeLLM([plan_for(horizon=3.0)]))
    data = form(trial=csv_file(TRIAL, "<i>trial</i>.csv"))
    r = c.post("/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 303
    job_id = r.headers["Location"].rsplit("/", 1)[-1]
    wait_for(c, job_id, lambda s: s["status"] in ("done", "failed"))
    page = c.get(f"/jobs/{job_id}").get_data(as_text=True)
    assert "&lt;i&gt;trial&lt;/i&gt;.csv" in page and "<i>trial</i>" not in page


def test_a_broken_csv_is_a_message_not_a_crash(make_app):
    _, c = make_app(FakeLLM([plan_for()]))
    data = form(trial=(io.BytesIO(b"a,b\n1,2\n"), "t.csv"), hist=(io.BytesIO(b"a,b\n1,2\n"), "h.csv"))
    r = c.post("/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and "at least 3 columns and 20 rows" in r.get_data(as_text=True)


def test_too_many_parallel_runs_are_refused(make_app):
    _, c = make_app(FakeLLM([plan_for(horizon=3.0)]), MAX_RUNNING=0)
    r = c.post("/run", data=form(), content_type="multipart/form-data")
    assert r.status_code == 200 and "already in progress" in r.get_data(as_text=True)


# ----------------------------------------------------------------------------
# Progress parsing
# ----------------------------------------------------------------------------
def test_parse_progress_reads_steps_versions_and_repairs():
    lines = [
        "== 1. Reading the data and the trial record ==",
        "== 7. Gates, design version 1 ==",
        "   G1 data fitness: PASS",
        "   G5 follow-up and horizon adequacy: FAIL",
        "== 8. G5 follow-up and horizon adequacy failed: choosing a fix ==",
        "   applied: shorten_horizon {'years': 3.6}",
        "== 7. Gates, design version 2 ==",
        "   G6 prognostic model credibility: PASS_AFTER_ACTION",
        "   G8 borrowing feasibility: INFO",
    ]
    p = parse_progress(lines)
    assert p["step"] == 7 and p["repairs"] == ["shorten_horizon {'years': 3.6}"]
    assert [v["failed"] for v in p["versions"]] == ["G5", None]
    assert [g["status"] for g in p["versions"][1]["gates"]] == ["PASS_AFTER_ACTION", "INFO"]
    assert parse_progress([]) == {"step": 0, "versions": [], "repairs": []}


def test_default_example_has_a_stand_in_id_that_follows_the_choice(make_app):
    _, c = make_app(FakeLLM([plan_for()]))
    html = c.get("/run").get_data(as_text=True)
    assert 'value="NCT00929591"' in html and 'data-nct="NCT00929591"' in html and "stand-in" in html
    assert 'data-nct=""' in html  # the other examples and uploads start with no id
