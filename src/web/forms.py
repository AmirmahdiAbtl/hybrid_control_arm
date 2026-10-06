"""Read and check the runner form (examples or uploaded CSVs)."""

import json
import re

import pandas as pd

from src.models import default_backend
from src.web.examples import EXAMPLES, load_example

BACKENDS = {
    "tabpfn-client": "TabPFN via API (data leaves this machine)",
    "tabpfn-local": "TabPFN (local weights)",
    "logreg": "Logistic baseline (quick, NOT TabPFN)",
}
MAX_ROWS = 200_000
NCT = re.compile(r"^NCT\d{8}$", re.I)


class FormError(ValueError):
    """Something in the form is wrong; the message is shown to the user."""


def default_form():
    backend = default_backend()
    return {
        "mode": "single",
        "data": "gbsg2",
        "nct": EXAMPLES["gbsg2"]["nct"],
        "eligibility": "",
        "backend": backend if backend in BACKENDS else "tabpfn-client",
        "n_real": "0",
        "max_repairs": "6",
    }


def read_inputs(form, files):
    """Validated inputs for a job.

    Returns
    -------
    dict
        ``trial``, ``hist``, ``trial_labels``, ``hist_labels``, ``mode``, ``label`` and ``agent`` (the keyword
        arguments of ``run_agent``: ``nct``, ``eligibility_text``, ``backend``, ``n_real``, ``max_repairs``).

    Raises
    ------
    FormError
    """
    mode = form.get("mode", "single")
    if mode not in ("single", "step"):
        raise FormError("Choose how to run: single loop or step by step.")
    backend = form.get("backend", "")
    if backend not in BACKENDS:
        raise FormError("Unknown model backend.")
    nct = form.get("nct", "").strip().upper()
    if nct and not NCT.match(nct):
        raise FormError("A ClinicalTrials.gov id looks like NCT00000000 (NCT and 8 digits).")
    text = form.get("eligibility", "").strip()
    if len(text) > 30_000:
        raise FormError("The eligibility text is too long (30,000 characters at most).")
    data = form.get("data", "")
    if data in EXAMPLES:
        trial, hist, labels_t, labels_h, example_text = load_example(data)
        text, label = text or example_text, EXAMPLES[data]["title"]
    elif data == "upload":
        trial, hist = _csv(files, "trial", "trial table"), _csv(files, "hist", "historical table")
        labels_t, labels_h = _json(files, "trial_dict"), _json(files, "hist_dict")
        label = f"{files['trial'].filename} + {files['hist'].filename}"
    else:
        raise FormError("Choose a dataset.")
    return {
        "trial": trial,
        "hist": hist,
        "trial_labels": labels_t,
        "hist_labels": labels_h,
        "mode": mode,
        "label": label,
        "agent": {
            "nct": nct or None,
            "eligibility_text": text or None,
            "backend": backend,
            "n_real": _integer(form, "n_real", 0, 2000) or None,
            "max_repairs": _integer(form, "max_repairs", 0, 8),
        },
    }


def _integer(form, name, low, high):
    try:
        value = int(form.get(name, low))
    except ValueError as e:
        raise FormError(f"{name} must be a whole number.") from e
    if not low <= value <= high:
        raise FormError(f"{name} must be between {low} and {high}.")
    return value


def _csv(files, field, what):
    f = files.get(field)
    if f is None or not f.filename:
        raise FormError(f"Upload the {what} (CSV).")
    try:
        df = pd.read_csv(f.stream)
    except Exception as e:  # noqa: BLE001 - any parse problem is a message for the user
        raise FormError(f"The {what} could not be read as CSV ({type(e).__name__}).") from e
    if df.shape[1] < 3 or len(df) < 20:
        raise FormError(f"The {what} needs at least 3 columns and 20 rows.")
    if len(df) > MAX_ROWS:
        raise FormError(f"The {what} has more than {MAX_ROWS:,} rows.")
    return df


def _json(files, field):
    f = files.get(field)
    if f is None or not f.filename:
        return {}
    try:
        data = json.load(f.stream)
    except ValueError as e:
        raise FormError(f"The dictionary '{f.filename}' is not valid JSON.") from e
    if not isinstance(data, dict):
        raise FormError('A column dictionary must be a JSON object: {"column": "description"}.')
    return {str(k): str(v) for k, v in data.items()}
