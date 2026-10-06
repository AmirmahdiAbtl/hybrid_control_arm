import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import models
from src.gated import protocol, runlog
from src.gated.protocol import (
    ASSUMPTIONS,
    CATE_NOTE,
    CHEAP_LABEL,
    EST_COLUMNS,
    ESTIMANDS,
    EXPLORATORY_TITLE,
    FOOTER,
    GATES,
    MEASUREMENT_REGISTER,
    PRIOR_OUTCOME_EXPOSURE,
    PROTOCOL_VERSION,
    WEIGHTS_LABEL,
    Z_T,
)
from src.gated.runlog import audit, check_text, fmt_value, json_safe, log_fits, now_iso, say, sha256_hex


def new_result(ctx):
    """Empty result dict of a run; the stages fill it and :func:`finish` completes it."""
    lab = models.BACKEND_LABELS.get(ctx["backend"], ctx["backend"])
    hybrid = ctx["mode"] == "hybrid"
    labels = {
        "outcome": lab,
        "prognostic": lab if hybrid else "not used (internal mode)",
        "weights": WEIGHTS_LABEL if hybrid else "logistic propensity (NOT TabPFN)",
        "cheap": CHEAP_LABEL,
    }
    return {
        "verdict": None,
        "tier": None,
        "claim": None,
        "secondary_claim": None,
        "first_failing_gate": None,
        "reason": "",
        "mode": ctx["mode"],
        "backend": ctx["backend"],
        "labels": labels,
        "primary": None,
        "estimates": [],
        "gates": None,
        "sensitivity": [],
        "cate": [],
        "support": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "validation": {},
        "placebo": {},
        "dag": {},
        "funnel": pd.DataFrame(),
        "horizon": pd.DataFrame(),
        "benchmarks": pd.DataFrame(),
        "diagnostics": {},
        "actions": ctx["actions"],
        "estimand_changed": False,
        "g8": None,
        "g11": None,
        "g11_detail": {},
        "tabpfn_fits": 0,
        "cache_hits": 0,
        "seconds": 0.0,
        "run_dir": str(ctx["run_dir"]),
        "protocol_digest": ctx["protocol_digest"],
        "run_index": ctx["run_index"],
        "earlier_runs": ctx["earlier_runs"],
        "text": "",
    }


def finish(ctx, res, stop_gate, reason):
    """Set verdict, tier and claim, write the ``verdict`` audit line and all output files.

    Parameters
    ----------
    ctx : dict
        Run context.
    res : dict
        Result filled by the stages.
    stop_gate : str or None
        Gate that stopped the run.
    reason : str
        Why the run stopped; "incomplete: ..." after an exception, "" when it ran to the end.

    Returns
    -------
    dict
        ``res`` with the verdict fields and the text filled in.
    """
    log_fits(ctx)
    fits = runlog.tabpfn_fits(ctx)
    if stop_gate or reason.startswith("incomplete"):
        verdict, first, tier = "NOT ACCEPTED", stop_gate, "NOT INTERPRETABLE"
    elif fits > protocol.MAX_TABPFN_FITS:  # safety net: check_budget before each model call normally stops first
        verdict, first, tier = "NOT ACCEPTED", "G10", "NOT INTERPRETABLE"
        reason = "NOT ACCEPTED (budget): the TabPFN fit budget was exceeded"
    elif res["g11"] == "FAIL":
        verdict, first, tier = "NOT ACCEPTED", "G11", "EXPLORATORY"
        reason = "effect estimated but not robust to plausible unmeasured confounding"
    else:
        verdict, first = "ACCEPTED", None
        tier = "SUPPORTIVE" if PRIOR_OUTCOME_EXPOSURE or res["estimand_changed"] else "PRIMARY"
        reason = "all primary gates met" + (f" (design actions: {', '.join(ctx['actions'])})" if ctx["actions"] else "")
    test_run = res["backend"] == "standin"
    if test_run:
        tier = "NOT INTERPRETABLE"
    p = res["primary"]
    claim = None
    if verdict == "ACCEPTED" and not test_run and p is not None:
        claim = (
            "effect not demonstrated" if p["lo"] <= 0 <= p["hi"] else "effect demonstrated under the stated assumptions"
        )
    if tier == "NOT INTERPRETABLE":  # no treatment-effect numbers
        res.update(primary=None, estimates=[], sensitivity=[], cate=[], benchmarks=pd.DataFrame(), secondary_claim=None)
    res.update(
        verdict=verdict,
        tier=tier,
        claim=claim,
        first_failing_gate=first,
        reason=reason,
        tabpfn_fits=fits,
        cache_hits=sum(1 for e in models.FIT_LOG[ctx["fit_start"] :] if e.get("cached")),
        seconds=round(time.time() - ctx["t0"], 1),
        gates=_gate_table(ctx),
        estimates=_df(res["estimates"], EST_COLUMNS),
        sensitivity=_df(res["sensitivity"]),
        cate=_df(res["cate"]),
    )
    res["text"] = verdict_text(res)
    audit(
        ctx,
        "verdict",
        verdict=verdict,
        tier=tier,
        claim=claim,
        first_failing_gate=first,
        reason=reason,
        secondary_claim=res["secondary_claim"],
        tabpfn_fits=fits,
        cache_hits=res["cache_hits"],
        seconds=res["seconds"],
    )
    write_outputs(ctx, res)
    say(ctx, f"done: {verdict} ({tier})", 1.0)
    return res


def verdict_text(result):
    """Four-line verdict text: headline, models, borrowing, footer.

    A NOT INTERPRETABLE result has no numbers; a stand-in run starts with "TEST RUN". Effects are in percentage
    points of event-free probability.

    Parameters
    ----------
    result : dict
        Result of :func:`src.gated.run_gated`.

    Returns
    -------
    str
    """
    tier, p, labels = result["tier"], result.get("primary"), result["labels"]
    head = f"{result['verdict']} ({tier})"
    if result["backend"] == "standin":
        line1 = (
            f"TEST RUN (random-forest stand-in, NOT TabPFN) - {head}: nothing to interpret; no treatment-effect "
            f"estimate is reported."
        )
    elif tier == "NOT INTERPRETABLE":
        first = result["first_failing_gate"] or "the run"
        line1 = (
            f"{head} - No treatment-effect estimate is reported because {first} failed: {result['reason']}. "
            f"Descriptive summaries only."
        )
    elif tier == "EXPLORATORY":
        line1 = (
            f"{head} - {p['scope']} {p['estimand']} at {p['horizon_years']:g} years: effect estimated but not "
            f"robust to plausible unmeasured confounding (G11). Numbers appear only in the appendix "
            f"'{EXPLORATORY_TITLE}'."
        )
    else:
        line1 = (
            f"{head} - {p['scope']} {p['estimand']} at {p['horizon_years']:g} years: {result['claim']}. Estimate "
            f"{p['est'] * 100:+.1f} points (95% CI {p['lo'] * 100:+.1f} to {p['hi'] * 100:+.1f})."
        )
        if p.get("changed_from"):
            line1 += (
                f" The result applies to the overlap population (estimand changed from {p['changed_from']} to "
                f"{p['estimand']} because G4 failed), not to the treated trial population."
            )
        if tier == "SUPPORTIVE" and PRIOR_OUTCOME_EXPOSURE:
            line1 += " Outcomes had been examined before this protocol was frozen; this is a methods demonstration."
    line2 = (
        f"Models: outcome {labels['outcome']}; prognostic score {labels['prognostic']}; weights "
        f"{labels['weights']}; sensitivity, validation and placebo {labels['cheap']}."
    )
    g8 = result.get("g8")
    if result["mode"] == "internal":
        line3 = "Historical data not used (internal mode); no borrowing."
    elif not g8:
        line3 = "Historical data used only as a prognostic score; borrowing not assessed (the run stopped before G8)."
    elif tier == "NOT INTERPRETABLE":
        line3 = "Historical data used only as a prognostic score; borrowing not used."
    elif g8["status"] == "INFEASIBLE":
        line3 = (
            f"Historical data used only as a prognostic score; borrowing INFEASIBLE (minimum detectable margin "
            f"{g8['mdm'] * 100:.1f} > {GATES['G8_margin'] * 100:g} points)."
        )
    else:
        line3 = (
            f"Historical data used only as a prognostic score; borrowing FEASIBLE (minimum detectable margin "
            f"{g8['mdm'] * 100:.1f} < {GATES['G8_margin'] * 100:g} points) but not used: it needs a protocol "
            f"amendment."
        )
    return "\n".join([line1, line2, line3, footer(labels, result["mode"])])


def footer(labels, mode="hybrid"):
    """The mandatory footer with the model labels.

    Parameters
    ----------
    labels : dict
        Needs ``outcome`` and ``prognostic`` (backend labels).
    mode : {"hybrid", "internal"}
        In internal mode the Rotterdam sentence says that the historical data were not used.

    Returns
    -------
    str
    """
    text = FOOTER.format(outcome=labels["outcome"], prognostic=labels["prognostic"])
    if mode == "internal":
        text = text.replace(
            "Historical (Rotterdam) data were used only as a prognostic score.",
            "Historical (Rotterdam) data were not used (internal mode).",
        )
    return text


def write_outputs(ctx, result):
    """Write the run's files next to audit.jsonl.

    Always verdict.json, gates.csv, estimates.csv, sensitivity.csv, cate.csv, support.csv, balance.csv,
    validation.csv, placebo.csv and report.md; funnel.csv, horizon.csv, benchmarks.csv and the per-repetition
    validation / placebo tables when they have rows.
    """
    d = Path(ctx["run_dir"])
    verdict = {
        k: result.get(k)
        for k in (
            "verdict",
            "tier",
            "claim",
            "secondary_claim",
            "first_failing_gate",
            "reason",
            "primary",
            "mode",
            "backend",
            "protocol_digest",
            "run_index",
            "tabpfn_fits",
            "cache_hits",
            "seconds",
        )
    }
    verdict.update(
        protocol_version=PROTOCOL_VERSION,
        backend_label=models.BACKEND_LABELS.get(result["backend"]),
        audit_sha256=sha256_hex(Path(ctx["audit_path"]).read_bytes()),
        started=ctx["started"],
        finished=now_iso(),
    )
    (d / "verdict.json").write_text(json.dumps(json_safe(verdict), indent=2, allow_nan=False) + "\n")
    _gate_rows(ctx).to_csv(d / "gates.csv", index=False)
    _df(result["estimates"], EST_COLUMNS).to_csv(d / "estimates.csv", index=False)
    val, pl = result.get("validation") or {}, result.get("placebo") or {}
    tables = {
        "sensitivity": result["sensitivity"],
        "cate": result["cate"],
        "support": result["support"],
        "balance": result["balance"],
        "validation": val.get("summary"),
        "placebo": pl.get("summary"),
    }
    for name, tab in tables.items():
        _df(tab).to_csv(d / f"{name}.csv", index=False)
    extra = {
        "funnel": result["funnel"],
        "horizon": result["horizon"],
        "benchmarks": result["benchmarks"],
        "validation_runs": val.get("runs"),
        "placebo_runs": pl.get("runs"),
    }
    for name, tab in extra.items():
        if tab is not None and len(tab):
            _df(tab).to_csv(d / f"{name}.csv", index=False)
    (d / "report.md").write_text(_report_md(ctx, result))


def _claim_text(res):
    """Claim paragraphs of report.md."""
    tier, p = res["tier"], res.get("primary")
    if res["backend"] == "standin":
        return ["TEST RUN with the random-forest stand-in (NOT TabPFN). Nothing to interpret."]
    if tier == "NOT INTERPRETABLE":
        return [
            (
                f"No treatment-effect estimate is reported because {res['first_failing_gate'] or 'the run'} "
                f"failed: {res['reason']}. Descriptive summaries only."
            )
        ]
    if tier == "EXPLORATORY":
        return [
            (
                "Effect estimated but not robust to plausible unmeasured confounding (G11): NOT ACCEPTED. "
                f"Numbers appear only in the appendix '{EXPLORATORY_TITLE}'."
            )
        ]
    lo, hi = p["lo"] * 100, p["hi"] * 100
    out = []
    if res["claim"] == "effect not demonstrated":
        out.append(
            f"The data are compatible with effects from {lo:+.1f} to {hi:+.1f} percentage points; an effect was "
            f"not demonstrated."
        )
    else:
        d = res.get("g11_detail") or {}
        acts = f" [actions: {', '.join(res['actions'])}]" if res["actions"] else ""
        population = ESTIMANDS[p["estimand"]]["population"]
        out.append(
            f"Pre-specified analysis; all gates met{acts}. In {population[0].lower()}{population[1:]}, the "
            f"estimated difference in {p['horizon_years']:g}-year recurrence-free status (tamoxifen vs no "
            f"hormonal therapy) is {p['est'] * 100:+.1f} percentage points (95% CI {lo:+.1f} to {hi:+.1f}). "
            f"This assumes no unmeasured confounding of tamoxifen use given Z_t = {{{', '.join(Z_T)}}}. To "
            f"move the CI limit to the null, an unmeasured confounder would need a bias factor above "
            f"{d.get('rr_star_lim', np.nan):.2f}; the strongest measured covariate or block reaches "
            f"{d.get('bf_max', np.nan):.2f}."
        )
    if p.get("changed_from"):
        out.append(
            f"The result applies to the overlap population (estimand changed from {p['changed_from']} to "
            f"{p['estimand']} because G4 failed), not to the treated trial population."
        )
    if tier == "SUPPORTIVE" and PRIOR_OUTCOME_EXPOSURE:
        out.append("Outcomes had been examined before this protocol was frozen; this is a methods demonstration.")
    if res.get("secondary_claim"):
        out.append(f"Secondary (fixed sequence): {res['secondary_claim']}.")
    return out


def _gate_table(ctx):
    """One row per gate evaluation, checks as text."""
    rows = [
        {
            "gate": g["gate"],
            "name": g["name"],
            "scope": g["scope"],
            "attempt": g["attempt"],
            "status": g["status"],
            "checks": "; ".join(check_text(c) for c in g["checks"]),
            "action": g["action"] or "",
            "note": g["note"],
        }
        for g in ctx["gates"]
    ]
    return pd.DataFrame(rows, columns=["gate", "name", "scope", "attempt", "status", "checks", "action", "note"])


def _gate_rows(ctx):
    """gates.csv: one row per check; a gate evaluation without checks gets one row with an empty metric."""
    rows = []
    for g in ctx["gates"]:
        for metric, value, op, thr, _ok in g["checks"] or [("", None, "", None, None)]:
            rows.append(
                {
                    "gate": g["gate"],
                    "name": g["name"],
                    "scope": g["scope"],
                    "attempt": g["attempt"],
                    "status": g["status"],
                    "metric": metric,
                    "value": _csv_value(value),
                    "op": op,
                    "threshold": _csv_value(thr),
                    "action": g["action"] or "",
                    "note": g["note"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["gate", "name", "scope", "attempt", "status", "metric", "value", "op", "threshold", "action", "note"],
    )


def _report_md(ctx, res):
    """report.md: verdict, claim, gates, estimates (by tier), checks, funnel, assumptions, footer."""
    tier = res["tier"]
    lab = models.BACKEND_LABELS.get(res["backend"], res["backend"])
    out = [
        f"# Gated causal analysis: {res['verdict']} ({tier})",
        "",
        (
            f"Protocol {PROTOCOL_VERSION}, digest `{res['protocol_digest'][:10]}`, run {res['run_index']} "
            f"({res['mode']} mode), backend: {lab}. Started {ctx['started']}; {res['seconds']} s; TabPFN fits "
            f"{res['tabpfn_fits']} (cache hits {res['cache_hits']}).."
        ),
        "",
        "## Verdict",
        "",
    ]
    out += [f"> {line}" for line in res["text"].splitlines()[:3]]
    out += ["", "## Claim", ""] + _claim_text(res)
    out += ["", "## Gates", "", _md_table(res["gates"])]
    est = _points(res["estimates"])
    sens = res["sensitivity"]
    if tier in ("PRIMARY", "SUPPORTIVE"):
        out += ["## Estimates (percentage points; 95% CI)", "", _md_table(est)]
    if tier in ("PRIMARY", "SUPPORTIVE", "EXPLORATORY"):
        if tier == "EXPLORATORY":
            out += [f"## Appendix: {EXPLORATORY_TITLE}", "", _md_table(est)]
        if len(sens):
            s = sens.copy()
            for c in ("est", "se", "lo", "hi"):
                s[c] = (s[c].astype(float) * 100).round(1)
            out += [
                "### Sensitivity analyses (points; logistic/linear, NOT TabPFN; NC rows: 90% CI)",
                "",
                _md_table(s.drop(columns=["backend_label"])),
            ]
        if len(res["cate"]):
            c = res["cate"][res["cate"]["kind"] != "BLP"].copy()
            for k in ("est", "se", "lo", "hi", "band_lo", "band_hi"):
                c[k] = (c[k].astype(float) * 100).round(1)
            out += [
                "### CATE (exploratory, points)",
                "",
                CATE_NOTE,
                "",
                _md_table(
                    c[
                        [
                            "horizon_years",
                            "estimand",
                            "subgroup",
                            "level",
                            "kind",
                            "est",
                            "lo",
                            "hi",
                            "band_lo",
                            "band_hi",
                            "p",
                            "n",
                            "n_treated",
                        ]
                    ]
                ),
            ]
        if len(res["benchmarks"]):
            out += ["### Benchmark bounding factors (G11; logistic/linear)", "", _md_table(res["benchmarks"])]
    else:
        out += ["No treatment-effect estimate is reported (tier NOT INTERPRETABLE). Descriptive summaries only.", ""]
    if len(res["support"]):
        out += ["## External support (G6)", "", _md_table(res["support"])]
    val, pl = res.get("validation") or {}, res.get("placebo") or {}
    if val or pl:
        out += ["## Validation and placebo (G12, report only; logistic/linear nuisance models, NOT TabPFN)", ""]
        if pl:
            out += ["Placebo test:", "", _md_table(pl["summary"])]
        if val:
            out += ["Hide-and-recover (control-outcome scale, points):", "", _md_table(val["summary"])]
            if val.get("draw_stability"):
                out += [
                    "Draw stability (outcome-free): "
                    + ", ".join(f"{k} {fmt_value(v)}" for k, v in val["draw_stability"].items()),
                    "",
                ]
    out += ["## Eligibility funnel", "", _md_table(res["funnel"])]
    threats = {t["assumption"]: t for t in (res.get("dag") or {}).get("threats", [])}
    rows = [
        {
            "id": k,
            "assumption": v[0],
            "needed by": v[1],
            "probe": v[2],
            "restoring the edge breaks": ", ".join(
                z
                for z, b in (("Z_t", threats.get(k, {}).get("breaks_zt")), ("Z_s", threats.get(k, {}).get("breaks_zs")))
                if b
            )
            or ("-" if k not in threats else "none"),
        }
        for k, v in ASSUMPTIONS.items()
    ]
    out += ["## Assumptions and probes", "", _md_table(pd.DataFrame(rows))]
    out += ["## Measurement register", ""] + [f"- {m}" for m in MEASUREMENT_REGISTER] + [""]
    out += ["## Design actions", "", ", ".join(res["actions"]) or "none", ""]
    earlier = pd.DataFrame(res["earlier_runs"])
    out += ["## Earlier runs with this protocol digest", "", _md_table(earlier)]
    out += ["---", "", footer(res["labels"], res["mode"]), ""]
    return "\n".join(out)


def _points(est):
    """Estimates for display, effects in percentage points (one decimal)."""
    if est is None or len(est) == 0:
        return pd.DataFrame()
    out = est.copy()
    for c in ("est", "se", "lo", "hi"):
        out[c] = (out[c].astype(float) * 100).round(1)
    out["horizon_years"] = out["horizon_years"].map(lambda v: f"{v:g}")
    return out[
        [
            "estimand",
            "scope",
            "role",
            "horizon_years",
            "method",
            "est",
            "lo",
            "hi",
            "se",
            "rr",
            "evalue_ci",
            "n_target",
            "n_comparison",
            "ess",
            "changed_from",
            "backend_label",
            "note",
        ]
    ]


def _md_table(df):
    if df is None or len(df) == 0:
        return "_none_\n"

    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        s = f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
        return s.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _df(x, columns=None):
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x or [], columns=columns)


def _csv_value(v):
    return (
        v
        if isinstance(v, (int, float, np.integer, np.floating, np.bool_)) and not isinstance(v, bool)
        else (fmt_value(v) if v is not None else "")
    )
