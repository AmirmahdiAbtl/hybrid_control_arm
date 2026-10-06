import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import models
from src.gated import protocol
from src.gated.protocol import (
    ACTIONS,
    GATE_NAMES,
    GATES,
    K_FOLDS,
    PRIOR_OUTCOME_EXPOSURE,
    PROTOCOL_VERSION,
    protocol_dict,
)


class StopRun(Exception):
    """A primary gate failed: the run ends NOT ACCEPTED at ``gate``."""

    def __init__(self, gate_id, reason):
        super().__init__(reason)
        self.gate, self.reason = gate_id, reason


def start_run(out_dir, mode, backend, df_a, df_b, mapping, rules):
    """Create the run folder and write the ``run_start`` audit line.

    The folder is ``out_dir/<protocol_digest[:10]>/run_<NNN>_<mode>/``; NNN counts the runs of this protocol digest.

    Parameters
    ----------
    out_dir : str or pathlib.Path
    mode : {"hybrid", "internal"}
    backend : str
    df_a, df_b : pandas.DataFrame
        Raw inputs (hashed, never written).
    mapping, rules : dict, list of dict
        Approved Stage-1 mapping and Stage-2 rules (hashed).

    Returns
    -------
    dict
        Run context: run_dir, audit_path, digests, run_index, earlier_runs, gates, actions, unblinded, the FIT_LOG
        positions and the progress callback.
    """
    digest = protocol_digest()
    base = Path(out_dir) / digest[:10]
    base.mkdir(parents=True, exist_ok=True)
    earlier = run_history(out_dir, digest)
    n = len([p for p in base.glob("run_*") if p.is_dir()]) + 1
    while True:  # two runs started at once must not share a folder
        run_dir = base / f"run_{n:03d}_{mode}"
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            n += 1
    inputs = {
        "a": frame_digest(df_a),
        "b": frame_digest(df_b),
        "mapping": sha256_hex(json.dumps(mapping, sort_keys=True, default=str)),
        "rules": sha256_hex(json.dumps(rules, sort_keys=True, default=str)),
    }
    ctx = {
        "run_dir": run_dir,
        "audit_path": run_dir / "audit.jsonl",
        "protocol_digest": digest,
        "code_digest": code_digest(),
        "inputs": inputs,
        "run_index": n,
        "earlier_runs": earlier,
        "mode": mode,
        "backend": backend,
        "gates": [],
        "actions": [],
        "unblinded": False,
        "fit_start": len(models.FIT_LOG),
        "fit_logged": len(models.FIT_LOG),
        "started": now_iso(),
        "t0": time.time(),
        "progress": None,
        "frac": 0.0,
    }
    audit(
        ctx,
        "run_start",
        protocol_digest=digest,
        protocol_version=PROTOCOL_VERSION,
        prior_outcome_exposure=PRIOR_OUTCOME_EXPOSURE,
        code_digest=ctx["code_digest"],
        inputs=inputs,
        mode=mode,
        backend=backend,
        backend_label=models.BACKEND_LABELS.get(backend, backend),
        versions=_versions(),
        model_file=_model_file(backend),
        run_index=n,
        earlier_runs=[{k: r.get(k) for k in ("run", "verdict", "tier", "mode", "backend")} for r in earlier],
    )
    return ctx


def audit(ctx, event, **fields):
    """Append ``{"t": <time>, "event": event, **fields}`` as one JSON line to the run's audit.jsonl.

    Parameters
    ----------
    ctx : dict
        Run context from :func:`start_run`.
    event : str
        E.g. run_start, gate, action, read_external_outcomes, model_fit, unblind, estimate, row_failed, verdict.
    **fields
        Any values; numpy types, NaN and DataFrames are converted by :func:`json_safe`.
    """
    line = json.dumps(json_safe({"t": now_iso(), "event": event, **fields}), allow_nan=False)
    with open(ctx["audit_path"], "a") as f:
        f.write(line + "\n")
        f.flush()


def say(ctx, msg, frac=None):
    """Send a progress message to the caller's ``progress(fraction, message)``, if any."""
    if frac is not None:
        ctx["frac"] = frac
    if ctx.get("progress"):
        ctx["progress"](ctx["frac"], msg)


def run_history(out_dir, digest=None):
    """Earlier runs of one protocol digest, read from their ``verdict.json``.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Results root (``results/gated``).
    digest : str, optional
        Protocol digest; default :func:`protocol_digest`.

    Returns
    -------
    list of dict
        run, dir, verdict, tier, claim, mode, backend, started; a run without verdict.json is "incomplete".
    """
    base = Path(out_dir) / (digest or protocol_digest())[:10]
    out = []
    for p in sorted(base.glob("run_*")) if base.is_dir() else []:
        if not (p.is_dir() and p.name.split("_")[1:2] and p.name.split("_")[1].isdigit()):
            continue
        row = {"run": int(p.name.split("_")[1]), "dir": p.name, "verdict": "incomplete (no verdict.json)"}
        try:
            v = json.loads((p / "verdict.json").read_text())
            row.update({k: v.get(k) for k in ("verdict", "tier", "claim", "mode", "backend", "started")})
        except (OSError, ValueError):
            pass
        out.append(row)
    return out


def protocol_digest():
    """SHA-256 of all protocol constants; its first 10 characters name the results folder of the protocol."""
    return sha256_hex(json.dumps(protocol_dict(), sort_keys=True, default=str))


def code_digest():
    """SHA-256 over the source of every module of the src package (sorted by path)."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def frame_digest(df):
    """SHA-256 of a DataFrame's CSV text."""
    return sha256_hex(df.to_csv(index=False))


def sha256_hex(data):
    """SHA-256 hex digest of bytes, or of the text of any other value."""
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


def now_iso():
    """Local time with its UTC offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_safe(v):
    """JSON-safe copy: numpy -> Python, NaN / inf -> None, tuples and sets -> lists, DataFrames -> records."""
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, pd.DataFrame):
        return json_safe(v.to_dict("records"))
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return [json_safe(x) for x in list(v)]
    if isinstance(v, set):
        return [json_safe(x) for x in sorted(v, key=str)]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if v is None or isinstance(v, str):
        return v
    return str(v)


def _versions():
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "tabpfn", "tabpfn-client"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _model_file(backend):
    """SHA-256 of the local TabPFN checkpoint when HCA_TABPFN_MODEL_PATH points to a file, else None."""
    path = os.getenv("HCA_TABPFN_MODEL_PATH")
    if backend != "tabpfn-local" or not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": path, "sha256": h.hexdigest()}


def gate(ctx, gid, name, scope, status, checks, attempt=1, action=None, note=""):
    """Record one gate evaluation: a row in ``ctx["gates"]``, an audit line and a progress line.

    Parameters
    ----------
    ctx : dict
    gid, name, scope : str
        Gate id (G0-G12), gate name and scope (e.g. "H", "R ATT", "X post", "H 2y").
    status : str
        PASS, FAIL, CONDITIONAL, PASS_AFTER_ACTION, NOT_APPLICABLE, SKIPPED, FEASIBLE, INFEASIBLE, WARN or REPORT.
    checks : list of tuple
        (metric, value, op, threshold, ok); ``ok`` None marks a report-only metric.
    attempt : int
    action : str, optional
        Design action applied before this evaluation.
    note : str

    Returns
    -------
    str
        ``status``.
    """
    row = {
        "gate": gid,
        "name": name,
        "scope": scope,
        "attempt": attempt,
        "status": status,
        "checks": [tuple(c) for c in checks],
        "action": action,
        "note": note,
    }
    ctx["gates"].append(row)
    audit(ctx, "gate", **row, tabpfn_fits=tabpfn_fits(ctx))
    say(ctx, gate_line(row))
    return status


def gate_status(checks):
    """PASS if every gated check holds (``ok`` None = report only), else FAIL."""
    return "PASS" if all(c[4] is None or bool(c[4]) for c in checks) else "FAIL"


def gate_line(row):
    """One printable line per gate evaluation.

    E.g. ``G4  within-trial positivity  [H]  attempt 1  PASS  ess 24.48 >= 20; max_share 0.0942 <= 0.15``; a failing
    check ends with ``-> no`` and a report-only metric with ``(report)``.
    """
    s = f"{row['gate']:<4}{row['name']}  [{row['scope']}]  attempt {row['attempt']}  {row['status']}"
    if row["checks"]:
        s += "  " + "; ".join(check_text(c) for c in row["checks"])
    if row["action"]:
        s += f"  (action {row['action']})"
    return s + (f"  | {row['note']}" if row["note"] else "")


def check_text(c):
    """Text of one gate check ``(metric, value, op, threshold, ok)``."""
    metric, value, op, threshold, ok = c
    if op == "report":
        return f"{metric} {fmt_value(value)} (report)"
    s = f"{metric} {fmt_value(value)} {op} {fmt_value(threshold)}"
    return s + (" (report)" if ok is None else "" if ok else " -> no")


def fmt_value(v):
    """Short text of a check value or threshold."""
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(fmt_value(x) for x in v) + ")"
    return str(v)


def can_act(ctx, aid):
    """True if design action ``aid`` may still be applied (before unblinding, once, within the action cap)."""
    return not ctx["unblinded"] and aid not in ctx["actions"] and len(ctx["actions"]) < GATES["max_design_actions"]


def apply_action(ctx, aid, note=""):
    """Log a pre-specified design action.

    Parameters
    ----------
    ctx : dict
    aid : {"B4.1", "B5.1", "B7.1", "B7.2"}
    note : str

    Raises
    ------
    RuntimeError
        After unblinding, on a second use of the same action, or when ``GATES["max_design_actions"]`` actions
        were already used.
    """
    if aid not in ACTIONS:
        raise ValueError(f"unknown design action '{aid}'")
    if ctx["unblinded"]:
        raise RuntimeError(f"design action {aid} refused: trial outcomes are already unblinded")
    if aid in ctx["actions"]:
        raise RuntimeError(f"design action {aid} refused: it was already used in this run")
    if len(ctx["actions"]) >= GATES["max_design_actions"]:
        raise RuntimeError(f"design action {aid} refused: {len(ctx['actions'])} design actions already used")
    ctx["actions"].append(aid)
    gid, what, restart = ACTIONS[aid]
    audit(ctx, "action", id=aid, gate=gid, restart=restart, stage="design", what=what, note=note)
    say(ctx, f"-> action {aid} (design): {what}" + (f", restart {restart}" if restart else ""))


def tabpfn_fits(ctx):
    """TabPFN fits of this run that were not served from the cache (G10 and the hard cap)."""
    return sum(
        1
        for e in models.FIT_LOG[ctx["fit_start"] :]
        if str(e.get("backend", "")).startswith("tabpfn") and not e.get("cached")
    )


def log_fits(ctx):
    """Write ``model_fit`` audit lines for the m_B and mu fits and every TabPFN fit since the last call.

    The many logistic/linear fits of the sensitivity rows, validation and placebo are not logged one by one.
    """
    log = models.FIT_LOG
    for e in log[ctx["fit_logged"] :]:
        purpose = str(e.get("purpose", ""))
        if str(e.get("backend", "")).startswith("tabpfn") or purpose.startswith(("mB_", "mu_")):
            audit(
                ctx,
                "model_fit",
                **{
                    k: e.get(k)
                    for k in ("purpose", "backend", "task", "n_train", "n_test", "seconds", "cached", "meta")
                },
            )
    ctx["fit_logged"] = len(log)


def check_budget(ctx, planned, what):
    """Stop the run (G10 FAIL) if the TabPFN fits so far plus ``planned`` would exceed ``MAX_TABPFN_FITS``."""
    n = tabpfn_fits(ctx)
    if n + planned > protocol.MAX_TABPFN_FITS:
        gate(
            ctx,
            "G10",
            GATE_NAMES["G10"],
            "budget",
            "FAIL",
            [("tabpfn_fits_done_plus_planned", n + planned, "<=", protocol.MAX_TABPFN_FITS, False)],
            note=f"NOT ACCEPTED (budget) before {what}",
        )
        raise StopRun("G10", f"NOT ACCEPTED (budget): the TabPFN fit budget would be exceeded ({what})")


def expected_tabpfn_fits(mode, supplementary_5y=False, demo=True):
    """TabPFN fits of one gated run with a TabPFN backend.

    Parameters
    ----------
    mode : {"hybrid", "internal"}
    supplementary_5y : bool
    demo : bool
        Hybrid mode with hidden controls (more than N_REAL eligible controls): mu0 is also fitted for the R reference.

    Returns
    -------
    int
        16 / 11 / 10 fits per horizon (hybrid demo / hybrid real use / internal) times 2 or 3 horizons.
    """
    per = (K_FOLDS + 1 + K_FOLDS * (2 if demo else 1)) if mode == "hybrid" else 2 * K_FOLDS
    return per * (3 if supplementary_5y else 2)
