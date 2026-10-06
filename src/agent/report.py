"""Verdict, claim and the written report of an agent run."""

import json

import numpy as np

from src import models
from src.agent.gates import GATE_NAMES

FOOTER = (
    "Evidence type: non-randomized comparison. The plan (treatment, outcome, covariates, horizon) was drafted by an "
    "LLM and checked by code, not by a clinician. Unmeasured confounding cannot be excluded; the E-value says how "
    "strong it would have to be. This analysis does not replace randomized evidence."
)


def claim_text(est, years):
    r = est["primary"]
    pts = lambda x: f"{100 * x:+.1f}"  # noqa: E731
    span = f"{pts(r['est'])} points (95% CI {pts(r['lo'])} to {pts(r['hi'])})"
    if r["lo"] > 0:
        what = "treated patients did better: more were event-free"
    elif r["hi"] < 0:
        what = "treated patients did worse: fewer were event-free"
    else:
        what = "effect not demonstrated (the interval includes 0)"
    return f"At {years:g} years, {what}. Difference in event-free probability {span}."


def tier(repairs, accepted):
    if not accepted:
        return "NONE"
    return "SUPPORTIVE" if repairs <= 2 else "EXPLORATORY"


def build_result(s, ctx):
    """Assemble the result dict from the final graph state."""
    run, spec = s.get("run"), s["spec"] if s.get("spec") else None
    est = run["estimate"] if run else None
    failed = run["failed"] if run else None
    accepted = bool(run and est and not failed and not s.get("stopped"))
    repairs = [t for t in s.get("design_state", {}).get("tried", []) if t["applied"]]
    reason = s.get("stopped") or (
        f"{failed['id']} {failed['name']}: not passed and no allowed fix left" if failed else ""
    )
    result = {
        "verdict": "ACCEPTED" if accepted else "NOT ACCEPTED",
        "tier": tier(len(repairs), accepted),
        "reason": reason,
        "plan_by": "AI-drafted, auto-confirmed (--yes)" if ctx["auto"] else "AI-drafted, confirmed by a human",
        "spec": spec.model_dump() if spec else None,
        "plan_digest": s.get("digest"),
        "borrowing_note": s.get("borrowing_note"),
        "design_state": s.get("design_state"),
        "mode": ("hybrid (historical prognostic score)" if run and run["study"]["hybrid"] else "trial only"),
        "backend": ctx["backend"],
        "prognostic_label": _prognostic_label(est, ctx["backend"]),
        "gates": run["gates"] if run else [],
        "first_failing_gate": failed["id"] if failed else None,
        "repairs": repairs,
        "attempts": s.get("attempts", []),
        "critic": s.get("critic"),
        "confirmations": s.get("confirmations", []),
        "mapping_coverage": s.get("coverage"),
        "rules": s.get("rules"),
        "funnel": run["study"]["funnel"] if run else [],
        "tabpfn_fits": len([f for f in models.FIT_LOG if not f.get("cached")]),
        "footer": FOOTER,
    }
    if est:
        r = est["primary"]
        keys = ("est", "se", "lo", "hi", "s1", "s0", "rr", "rr_lo", "rr_hi", "ess")
        result["primary"] = {k: float(r[k]) for k in keys} | {"horizon_years": est["horizon_years"]}
        result["reference"] = {
            "unadjusted_difference": est["unadjusted"]["est"],
            "unadjusted_ci": [est["unadjusted"]["lo"], est["unadjusted"]["hi"]],
            "ipw_difference_point": est["ipw_point"],
        }
        result["sensitivity"] = est["sensitivity"]
        result["claim"] = (
            claim_text(est, est["horizon_years"]) if accepted else "no claim: the analysis was not accepted"
        )
        result["n_treated"], result["n_controls_kept"] = est["n_treated"], est["n_controls_kept"]
    else:
        result["claim"] = "no claim: no estimate was produced"
    return result


def _prognostic_label(est, backend):
    if not est:
        return None
    choice = est["prognostic"]
    if choice == "model":
        return models.BACKEND_LABELS.get(backend, backend)
    return {"linear": "linear prognostic score (NOT TabPFN)", "none": "none: trial covariates only (NOT TabPFN)"}[
        choice
    ]


def _question_text(spec, years):
    if not spec:
        return None
    return (
        f"Does the treatment (column '{spec['treatment']}', control = {spec['control_value']}) change the probability "
        f"of being event-free at {years:g} years, compared with the control arm?"
    )


def _outcome_text(spec):
    if not spec:
        return None
    return (
        f"event = column '{spec['event']}' has a value in {spec['event_values']}; every other value is censored "
        f"(not an event); time column '{spec['time']}' in {spec['time_unit']}; treatment column '{spec['treatment']}', "
        f"control = {spec['control_value']}; result is the difference in the probability of being event-free"
    )


def facts_for_narration(result):
    """The numbers the LLM may quote, already formatted."""
    f = {
        "verdict": result["verdict"],
        "reason": result["reason"],
        "mode": result["mode"],
        "repairs": [f"after {t['gate']} failed, the agent {t['plain']}" for t in result["repairs"]],
        "question": _question_text(result["spec"], result.get("primary", {}).get("horizon_years", 0)),
        "gates": [f"{g['id']} {g['name']}: {g['status']}" for g in result["gates"]],
        "plan": result["spec"],
        "outcome": _outcome_text(result["spec"]),
        "claim": result["claim"],
        "plan_by": result["plan_by"],
        "limits": result["footer"],
    }
    if "primary" in result:
        s = result["sensitivity"]
        if np.isfinite(s["evalue"]) and np.isfinite(s["evalue_ci"]):
            f["unmeasured_confounding"] = (
                f"A hidden factor would have to raise both the chance of being treated and the chance of the event "
                f"by a risk ratio of at least {s['evalue']:.2f} to explain the estimate away "
                f"({s['evalue_ci']:.2f} for the confidence limit nearest no effect; 1.00 means the interval already "
                "includes no effect)."
            )
    return f


def render_markdown(result, narrative=""):
    L = [f"# Hybrid control arm: {result['verdict']}", ""]
    L += [f"**Claim.** {result['claim']}", ""]
    if result["reason"]:
        L += [f"**Why not accepted.** {result['reason']}", ""]
    L += [f"Tier: {result['tier']} | plan: {result['plan_by']} | mode: {result['mode']}", ""]
    if result.get("borrowing_note"):
        L += [f"**Warning.** {result['borrowing_note']}", ""]
    if narrative:
        L += ["## In plain words", narrative, ""]
    if result["spec"]:
        L += [
            "## Plan",
            "```json",
            json.dumps(result["spec"], indent=2),
            "```",
            f"Plan digest: `{result['plan_digest']}`",
            "",
        ]
    L += _critic_section(result) + _gate_section(result) + _repair_section(result) + _estimate_section(result)
    L += ["## Limits", result["footer"], ""]
    return "\n".join(L)


def _critic_section(result):
    c = result.get("critic")
    if not c:
        return []
    out = ["## Independent review of the plan (LLM critic)", f"Verdict: {c['verdict']}"]
    out += [f"- [{x['severity']}] {x['issue']} (fix: {x['fix']} {x['column']})" for x in c["concerns"]]
    return out + [""]


def _gate_section(result):
    out = ["## Gates (final design)", "| gate | status | checks |", "|---|---|---|"]
    for g in result["gates"]:
        checks = "; ".join(
            f"{c['name']} {c['value']:.3g}" + (f" {c['op']} {c['limit']:.3g}" if c["op"] != "report" else "")
            for c in g["checks"]
        )
        out.append(f"| {g['id']} {GATE_NAMES[g['id']]} | {g['status']} | {checks} {g['note']} |")
    return out + [""]


def _repair_section(result):
    if not result["repairs"]:
        return ["## Repairs", "None: the first design passed every gate.", ""]
    out = [
        "## Repairs",
        f"{len(result['repairs'])} change(s) were made after a gate failed. A repair is chosen from the gate's "
        "numbers only, never from the effect estimate. Every design version that reached estimation is listed so "
        "that nothing is hidden:",
    ]
    out += [
        f"{i + 1}. after {t['gate']} failed, the agent {t['plain']}. Reason given: {t['reason']}"
        for i, t in enumerate(result["repairs"])
    ]
    out += ["", "| version | failed gate | reached estimation | difference (points) |", "|---|---|---|---|"]
    for a in result["attempts"]:
        diff = "" if a["estimate"] is None else f"{100 * a['estimate']['est']:+.1f}"
        out.append(f"| {a['version']} | {a['failed'] or '-'} | {'yes' if a['estimate'] else 'no'} | {diff} |")
    return out + [""]


def _estimate_section(result):
    if "primary" not in result:
        return []
    p, ref, s = result["primary"], result["reference"], result["sensitivity"]
    pts = lambda x: f"{100 * x:+.1f}"  # noqa: E731
    return [
        "## Estimate",
        f"- Primary (ATT, robust hybrid, {p['horizon_years']:g} y): **{pts(p['est'])} points**, "
        f"95% CI {pts(p['lo'])} to {pts(p['hi'])}",
        f"- Event-free: treated {100 * p['s1']:.1f}%, estimated control {100 * p['s0']:.1f}%",
        f"- Reference (not used for the verdict): unadjusted {pts(ref['unadjusted_difference'])}, "
        f"IPW {pts(ref['ipw_difference_point'])}",
        f"- Prognostic score: {result['prognostic_label']}",
        f"- E-value (risk ratio {s['rr']:.2f}): {s['evalue']:.2f}; for the CI limit nearest 'no effect': "
        f"{s['evalue_ci']:.2f} (1.00 means the interval already includes no effect)",
        "",
    ]
