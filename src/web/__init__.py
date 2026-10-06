"""The web app (Flask): landing page, the runner (single loop or step by step), live job pages and the docs."""

from pathlib import Path

from flask import Flask

from src.web.jobs import JobRegistry
from src.web.routes import register_routes

ROOT = Path(__file__).resolve().parent.parent.parent


def create_app(config=None):
    """Build the app.

    Parameters
    ----------
    config : dict, optional
        ``RESULTS_DIR`` (default ``results``), ``LLM`` (callable ``llm(system, user) -> str`` used instead of the
        configured one; tests), ``MAX_RUNNING`` (parallel runs), ``MAX_CONTENT_LENGTH`` (upload limit in bytes).
    """
    app = Flask(__name__)
    app.config.update(
        RESULTS_DIR=str(ROOT / "results"),
        LLM=None,
        MAX_RUNNING=3,
        MAX_CONTENT_LENGTH=60 * 1024 * 1024,
        DOCS_DIR=str(ROOT / "docs"),
    )
    app.config.update(config or {})
    app.extensions["jobs"] = JobRegistry(app.config["MAX_RUNNING"])
    register_routes(app)
    return app
