"""Job page, the polling API and the downloads."""

from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_from_directory

from src.agent.gates import GATE_NAMES
from src.web.jobs import STEPS

bp = Blueprint("jobs", __name__)
FILES = {"report.md": "text/markdown", "result.json": "application/json", "audit.jsonl": "application/jsonl"}


def find(job_id):
    record = current_app.extensions["jobs"].get(job_id)
    if record is None:
        abort(404)
    return record


@bp.get("/jobs/<job_id>")
def job_page(job_id):
    record = find(job_id)
    return render_template(
        "job.html",
        record=record,
        snap=record.snapshot(),
        steps=STEPS,
        result=record.job.result,
        gate_names=GATE_NAMES,
    )


@bp.get("/api/jobs/<job_id>")
def api_job(job_id):
    since = request.args.get("since", 0, type=int)
    return jsonify(find(job_id).snapshot(max(since, 0)))


@bp.post("/api/jobs/<job_id>/answer")
def api_answer(job_id):
    record = find(job_id)
    pending = record.job.pending
    body = request.get_json(silent=True) or {}
    if pending is None or body.get("id") != pending["id"]:
        return jsonify({"ok": False, "error": "There is no open question with that id."}), 409
    record.job.answer(str(body.get("answer", "y"))[:2000])
    return jsonify({"ok": True})


@bp.post("/api/jobs/<job_id>/cancel")
def api_cancel(job_id):
    find(job_id).job.cancel()
    return jsonify({"ok": True})


@bp.get("/jobs/<job_id>/files/<name>")
def download(job_id, name):
    record = find(job_id)
    if name not in FILES or not record.job.result:
        abort(404)
    folder = Path(record.job.result["run_dir"]).resolve()
    return send_from_directory(folder, name, mimetype=FILES[name], as_attachment=True)
