"""The agent: one LangGraph graph from raw tables to a verdict.

profile -> harmonize -> confirm mapping -> eligibility -> plan -> critic -> confirm plan -> freeze -> gates <-> repair
-> report. The LLM drafts (mapping, rules, plan, critique, repair choice, summary); code validates and computes; a
human confirms at the marked points (``--yes`` confirms automatically and says so in the report).
"""

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src import models, online
from src.agent import frames, planner, repair
from src.agent.gates import LIMITS, run_gates
from src.agent.io import Audit, AutoAsker, ConsoleAsker, logged_llm
from src.agent.report import build_result, facts_for_narration, render_markdown
from src.agent.spec import normalize_spec, validate_spec
from src.harmonize import coverage, harmonize_frames, profile_columns, propose_mapping
from src.online import extract_criteria
from src.rules import apply_rules

MAX_PLAN_TRIES = 3
MAX_HUMAN_REVISIONS = 2


class AgentState(TypedDict, total=False):
    trial: Any
    hist: Any
    prof_t: dict
    prof_h: dict
    brief: dict
    elig_text: str
    mapping: dict
    a_h: Any
    b_h: Any
    coverage: str
    notes: list
    rules: list
    funnel: Any
    hist_elig: Any
    disabled_rules: list
    spec: Any
    warnings: list
    feedback: list
    critic: dict
    revised: bool
    human_rounds: int
    digest: str
    design_state: dict
    run: dict
    attempts: list
    route: str
    stopped: str
    result: dict
    borrowing_note: str


def run_agent(
    trial,
    hist,
    *,
    nct=None,
    eligibility_text=None,
    trial_labels=None,
    hist_labels=None,
    llm=None,
    asker=None,
    backend=None,
    out_dir="results/agent",
    cache_dir="results/cache",
    n_real=None,
    max_repairs=6,
    plan=None,
    say=print,
):
    """Run the whole analysis; returns the result dict (also written to ``out_dir/run_*/``).

    Parameters
    ----------
    trial, hist : pandas.DataFrame
        The new trial (with treatment, time and event) and the historical study, in their own column names.
    nct : str, optional
        ClinicalTrials.gov id: the title, outcomes and eligibility text come from the registry.
    eligibility_text : str, optional
        Eligibility criteria text (used instead of the registry's).
    llm : callable, optional
        ``llm(system, user) -> str``; default ``online.get_llm()``. The agent needs one.
    asker : object with ``ask(kind, title, body, default)``, optional
        Human confirmation points; default: the terminal.
    n_real : int, optional
        Keep only this many trial controls as the real controls (simulates a small control arm).
    max_repairs : int
        Cap on applied repairs; the run ends NOT ACCEPTED when it is reached.
    plan : StudySpec, optional
        A frozen plan from an earlier run (``result["spec"]``): the LLM drafts and the critic are skipped.
    """
    llm = llm or online.get_llm()
    if llm is None:
        raise RuntimeError("The agent needs an LLM: set ANTHROPIC_FOUNDRY_API_KEY, _RESOURCE and ANTHROPIC_MODEL.")
    run_dir = Path(out_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    audit = Audit(run_dir / "audit.jsonl")
    asker = asker or ConsoleAsker()

    def announce(message):
        say(message)
        audit("say", message=message)

    ctx = {
        "llm": logged_llm(llm, audit),
        "asker": asker,
        "audit": audit,
        "say": announce,
        "backend": backend or models.default_backend(),
        "cache_dir": cache_dir,
        "auto": isinstance(asker, AutoAsker),
        "n_real": n_real,
        "max_repairs": max_repairs,
        "plan_override": plan,
        "nct": nct,
        "elig_text": eligibility_text,
        "trial_labels": trial_labels or {},
        "hist_labels": hist_labels or {},
        "confirmations": [],
        "run_dir": run_dir,
    }
    audit("start", backend=ctx["backend"], n_trial=len(trial), n_hist=len(hist), nct=nct, auto=ctx["auto"])
    graph = build_graph(ctx)
    final = graph.invoke({"trial": trial, "hist": hist}, {"recursion_limit": 100})
    return final["result"]


def confirm(ctx, kind, title, body):
    answer = ctx["asker"].ask(kind, title, body).strip()
    ctx["confirmations"].append({"kind": kind, "title": title, "answer": answer, "automatic": ctx["auto"]})
    ctx["audit"]("human", kind=kind, answer=answer, automatic=ctx["auto"])
    return answer


def build_graph(ctx):
    g = StateGraph(AgentState)
    nodes = {
        "profile": lambda s: n_profile(s, ctx),
        "harmonize": lambda s: n_harmonize(s, ctx),
        "confirm_mapping": lambda s: n_confirm_mapping(s, ctx),
        "eligibility": lambda s: n_eligibility(s, ctx),
        "plan": lambda s: n_plan(s, ctx),
        "critic": lambda s: n_critic(s, ctx),
        "confirm_plan": lambda s: n_confirm_plan(s, ctx),
        "freeze": lambda s: n_freeze(s, ctx),
        "gates": lambda s: n_gates(s, ctx),
        "repair": lambda s: n_repair(s, ctx),
        "report": lambda s: n_report(s, ctx),
    }
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "profile")
    g.add_edge("profile", "harmonize")
    g.add_edge("harmonize", "confirm_mapping")
    for src, dst in [("confirm_mapping", "eligibility"), ("eligibility", "plan"), ("freeze", "gates")]:
        g.add_conditional_edges(src, lambda s, dst=dst: "report" if s.get("stopped") else dst, [dst, "report"])
    g.add_conditional_edges("plan", lambda s: "report" if s.get("stopped") else "critic", ["critic", "report"])
    g.add_conditional_edges("critic", lambda s: s["route"], ["plan", "confirm_plan"])
    g.add_conditional_edges("confirm_plan", lambda s: s["route"], ["plan", "freeze", "report"])
    g.add_conditional_edges("gates", lambda s: s["route"], ["repair", "report"])
    g.add_conditional_edges("repair", lambda s: s["route"], ["gates", "report"])
    g.add_edge("report", END)
    return g.compile()


# ----------------------------------------------------------------------------
# Data and harmonization
# ----------------------------------------------------------------------------
def n_profile(s, ctx):
    ctx["say"]("== 1. Reading the data and the trial record ==")
    prof_t = profile_columns(s["trial"], ctx["trial_labels"])
    prof_h = profile_columns(s["hist"], ctx["hist_labels"])
    brief, text = None, ctx["elig_text"]
    if ctx["nct"]:
        try:
            brief = planner.fetch_trial_brief(ctx["nct"])
            text = text or brief["eligibility"]
            ctx["say"](f"   registry: {brief['title'][:100]}")
        except Exception as ex:  # noqa: BLE001 - registry down or bad id: continue without it, and say so
            ctx["say"](f"   could not read {ctx['nct']} from ClinicalTrials.gov ({ex}); continuing without it")
            ctx["audit"]("registry_error", message=str(ex))
    ctx["say"](f"   trial: {len(s['trial'])} patients, {s['trial'].shape[1]} columns; historical: {len(s['hist'])}")
    return {"prof_t": prof_t, "prof_h": prof_h, "brief": brief, "elig_text": text or "", "attempts": []}


def json_safe(value):
    """Copy of a profile with NaN / infinity replaced by None (Jev rejects them; all-empty columns have NaN stats)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def n_harmonize(s, ctx):
    ctx["say"]("== 2. Matching the historical columns to the trial's (Jev + LLM) ==")
    scores = None
    if online.jev_status():
        try:
            scores = online.jev_score_columns(
                online.make_async_jev(), json_safe(s["prof_t"]), json_safe(s["prof_h"]), mode="pairwise"
            )
        except Exception as ex:  # noqa: BLE001 - Jev is optional; the LLM then sees every candidate
            ctx["say"](f"   Jev failed ({type(ex).__name__}); every column goes to the LLM")
    else:
        ctx["say"]("   Jev is not configured (TYPESAFE_API_KEY): every column goes to the LLM")
    mapping = propose_mapping(s["trial"], s["hist"], s["prof_t"], s["prof_h"], scores, ctx["llm"])
    return {"mapping": mapping}


def classify_mapping(mapping):
    """(safe, flagged, unmatched): safe = same definition and checks pass; flagged = needs a human to approve."""
    safe, flagged, unmatched = [], [], []
    for a, e in mapping.items():
        if not e["b_cols"]:
            unmatched.append(a)
        elif e["checks"]["ok"] and e["same_definition"] == "yes":
            safe.append(a)
        else:
            flagged.append(a)
    return safe, flagged, unmatched


def parse_overrides(answer):
    """'approve a, b; reject c' -> ({'a', 'b'}, {'c'}); anything else -> empty sets."""
    out = {"approve": set(), "reject": set()}
    for word in out:
        m = re.search(rf"{word}\s+([^;]+)", answer, re.I)
        if m:
            out[word] = {x.strip() for x in m.group(1).split(",") if x.strip()}
    return out["approve"], out["reject"]


def n_confirm_mapping(s, ctx):
    mapping = s["mapping"]
    safe, flagged, unmatched = classify_mapping(mapping)
    lines = [f"  OK       {a} <- {' + '.join(mapping[a]['b_cols'])}" for a in safe]
    lines += [
        f"  FLAGGED  {a} <- {' + '.join(mapping[a]['b_cols'])}: definition {mapping[a]['same_definition']}; "
        f"{mapping[a]['definition_note'] or 'a check failed'} (not used unless you approve it)"
        for a in flagged
    ]
    lines += [f"  NO MATCH {a} (not used)" for a in unmatched]
    body = "\n".join(lines) + "\nReply: y | n | 'approve col1, col2' | 'reject col3'"
    answer = confirm(ctx, "mapping", "Column mapping: trial <- historical", body)
    if answer.lower() in ("n", "no"):
        return {"stopped": "stopped by the user at the column mapping"}
    approve, reject = parse_overrides(answer)
    use = (set(safe) | approve) - reject
    approved = {a: {**e, "status": "approved" if a in use and e["b_cols"] else "rejected"} for a, e in mapping.items()}
    a_h, b_h, info = harmonize_frames(s["trial"], s["hist"], approved)
    notes = [
        {
            "trial_column": a,
            "historical_columns": e["b_cols"],
            "same_definition": e["same_definition"],
            "note": e["definition_note"],
        }
        for a, e in approved.items()
        if e["b_cols"] and e["same_definition"] != "yes"
    ]
    notes += [{"trial_column": a, "note": "not in the historical data"} for a in a_h.columns if a not in b_h.columns]
    ctx["audit"]("mapping_approved", approved=sorted(use), info={k: v for k, v in info.items() if k != "post_checks"})
    return {
        "mapping": approved,
        "a_h": a_h,
        "b_h": b_h,
        "coverage": coverage(approved, list(s["trial"].columns)),
        "notes": notes,
    }


def eligible_history(s, disabled):
    """Historical patients (in the trial's column names) that pass the enabled rules."""
    rules = [r for r in s["rules"] if r["id"] not in disabled]
    mask, funnel, invalid = apply_rules(s["hist"], rules)
    return s["b_h"][mask.reindex(s["b_h"].index, fill_value=False)], funnel, invalid


def n_eligibility(s, ctx):
    ctx["say"]("== 3. Who in the historical data would have qualified for the trial ==")
    rules = []
    if s["elig_text"]:
        criteria = extract_criteria(ctx["llm"], s["elig_text"], s["prof_h"], s["hist"])
        rules = [c.model_dump() for c in criteria if c.kind == "rule" and c.rule]
    else:
        ctx["say"]("   no eligibility text: the historical patients are not filtered")
    s = {**s, "rules": rules}
    hist_elig, funnel, invalid = eligible_history(s, [])
    disabled = []
    if rules:
        body = funnel.to_string(index=False) + "\nReply: y | n | 'disable I2, I3'"
        answer = confirm(ctx, "rules", "Eligibility rules applied to the historical patients", body)
        if answer.lower() in ("n", "no"):
            return {"stopped": "stopped by the user at the eligibility rules"}
        m = re.search(r"disable\s+(.+)", answer, re.I)
        disabled = [x.strip() for x in m.group(1).split(",")] if m else []
        hist_elig, funnel, invalid = eligible_history(s, disabled)
    ctx["audit"]("eligibility", rules=rules, disabled=disabled, n_eligible=len(hist_elig), invalid=invalid)
    ctx["say"](f"   {len(hist_elig)} of {len(s['hist'])} historical patients are eligible")
    return {"rules": rules, "funnel": funnel, "hist_elig": hist_elig, "disabled_rules": disabled}


# ----------------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------------
def n_plan(s, ctx):
    ctx["say"]("== 4. The LLM drafts the analysis plan; code checks it ==")
    if ctx.get("plan_override") is not None:
        spec, fixes = normalize_spec(ctx["plan_override"], s["a_h"], s["hist_elig"])
        errors, warnings = validate_spec(spec, s["a_h"], s["hist_elig"])
        ctx["audit"]("plan_given", spec=spec.model_dump(), errors=errors, warnings=warnings, fixes=fixes)
        if errors:
            return {"stopped": "the given plan does not fit the data: " + "; ".join(errors)}
        ctx["say"]("   using the plan from --plan-file (no LLM draft)")
        return {"spec": spec, "warnings": warnings + fixes, "feedback": [], "revised": True}
    prof_t = profile_columns(s["a_h"], ctx["trial_labels"])
    prof_h = profile_columns(s["hist_elig"]) if len(s["hist_elig"]) else None
    feedback, previous = list(s.get("feedback") or []), s.get("spec")
    for attempt in range(MAX_PLAN_TRIES):
        spec = planner.draft_plan(ctx["llm"], s.get("brief"), prof_t, prof_h, s.get("notes", []), feedback, previous)
        spec, fixes = normalize_spec(spec, s["a_h"], s["hist_elig"])
        errors, warnings = validate_spec(spec, s["a_h"], s["hist_elig"])
        ctx["audit"](
            "plan_draft", attempt=attempt + 1, spec=spec.model_dump(), errors=errors, warnings=warnings, fixes=fixes
        )
        if not errors:
            ctx["say"](f"   plan accepted by the checks (draft {attempt + 1}); fixes applied by code: {len(fixes)}")
            return {"spec": spec, "warnings": warnings + fixes, "feedback": []}
        ctx["say"](f"   draft {attempt + 1} has {len(errors)} problem(s); sending them back to the LLM")
        feedback, previous = errors, spec
    return {"stopped": "the plan could not be made valid after 3 drafts: " + "; ".join(errors)}


def n_critic(s, ctx):
    if s.get("revised"):
        return {"route": "confirm_plan"}
    ctx["say"]("== 5. A second LLM call reviews the plan ==")
    critique = planner.critique_plan(
        ctx["llm"], s.get("brief"), s["spec"], profile_columns(s["a_h"], ctx["trial_labels"])
    )
    ctx["audit"]("critic", **critique.model_dump())
    actionable = [c for c in critique.concerns if c.severity in ("high", "medium") and c.fix != "other"]
    for c in critique.concerns:
        ctx["say"](f"   [{c.severity}] {c.issue}")
    out = {"critic": critique.model_dump(), "revised": True}
    if actionable:
        return {
            **out,
            "route": "plan",
            "feedback": [f"{c.issue} (suggested fix: {c.fix} {c.column})" for c in actionable],
        }
    return {**out, "route": "confirm_plan"}


def n_confirm_plan(s, ctx):
    spec = s["spec"]
    lines = json.dumps(spec.model_dump(), indent=2)
    extra = "".join(f"\n  note: {w}" for w in s.get("warnings", []))
    answer = confirm(
        ctx, "plan", "Analysis plan (will be frozen)", lines + extra + "\nReply: y | n | or describe what to change"
    )
    if answer.lower() in ("n", "no"):
        return {"stopped": "stopped by the user at the plan", "route": "report"}
    if answer.lower() in ("y", "yes", ""):
        return {"route": "freeze"}
    rounds = s.get("human_rounds", 0) + 1
    if rounds > MAX_HUMAN_REVISIONS:
        return {"stopped": "too many plan revisions requested", "route": "report"}
    return {"route": "plan", "feedback": [f"the human asks: {answer}"], "human_rounds": rounds}


def borrowing_problems(spec, hist):
    """Why the historical data cannot be used at all (empty list: they can)."""
    problems = []
    if not len(hist):
        problems.append("no historical patient is left after the eligibility rules")
    missing = [c for c in (spec.time, spec.event) if c not in hist.columns]
    if missing:
        problems.append(
            f"the column matching did not give the historical data a matched {' and '.join(missing)} column"
        )
    if not any(c.cross_source for c in spec.covariates):
        problems.append("no covariate is comparable across the two studies")
    return problems


def n_freeze(s, ctx):
    spec, hist = s["spec"], s["hist_elig"]
    problems = borrowing_problems(spec, hist)
    can_borrow = not problems
    note = (
        None
        if can_borrow
        else "Historical data not used: " + "; ".join(problems) + ". The result uses the trial alone."
    )
    payload = {"spec": spec.model_dump(), "rules": s["rules"], "limits": LIMITS, "disabled": s["disabled_rules"]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    ctx["audit"]("plan_frozen", digest=digest, borrowing=can_borrow, note=note)
    ctx["say"](f"== 6. Plan frozen ({digest}); borrowing {'on' if can_borrow else 'OFF'} ==")
    if note:
        ctx["say"](f"   WARNING: {note}")
    state = frames.new_state(spec, can_borrow, ctx["n_real"])
    state["disabled_rules"] = list(s["disabled_rules"])
    return {"digest": digest, "design_state": state, "borrowing_note": note}


# ----------------------------------------------------------------------------
# Gates and repair
# ----------------------------------------------------------------------------
def n_gates(s, ctx):
    version = len(s["attempts"]) + 1
    ctx["say"](f"== 7. Gates, design version {version} ==")
    gate_ctx = {k: ctx[k] for k in ("backend", "cache_dir", "say", "audit")}
    run = run_gates(s["spec"], s["a_h"], s["hist_elig"], s["design_state"], gate_ctx)
    failed, est = run["failed"], run["estimate"]
    attempt = {
        "version": version,
        "failed": failed["id"] if failed else None,
        "design_state": run["state"],
        "estimate": None
        if est is None
        else {"est": est["primary"]["est"], "lo": est["primary"]["lo"], "hi": est["primary"]["hi"]},
    }
    applied = [t for t in run["state"]["tried"] if t["applied"]]
    can_repair = failed is not None and len(applied) < ctx["max_repairs"]
    return {
        "run": run,
        "design_state": run["state"],
        "attempts": [*s["attempts"], attempt],
        "route": "repair" if can_repair else "report",
    }


def n_repair(s, ctx):
    failed, state = s["run"]["failed"], s["design_state"]
    ctx["say"](f"== 8. {failed['id']} {failed['name']} failed: choosing a fix ==")
    options = repair.build_options(failed, s["spec"], state, s["rules"], s["run"]["study"])
    while options:
        idx, why = repair.choose(ctx["llm"], failed, options, state)
        option = options[idx]
        body = (
            f"fix: {option['action']} {option['args']}\nwhy: {option['why']}\nthe agent's reason: {why}\n"
            "Reply: y | n (skip this fix)"
        )
        answer = confirm(ctx, "repair", f"{failed['id']} {failed['name']} failed", body)
        entry = {
            "plain": repair.describe(option, state),
            "gate": failed["id"],
            "action": option["action"],
            "args": option["args"],
            "reason": f"{option['why']}; {why}",
        }
        if answer.lower() in ("n", "no"):
            state = {**state, "tried": [*state["tried"], {**entry, "applied": False}]}
            options = [o for i, o in enumerate(options) if i != idx]
            continue
        new = repair.apply_option(state, option)
        new["tried"] = [*state["tried"], {**entry, "applied": True}]
        ctx["audit"]("repair", **entry)
        ctx["say"](f"   applied: {option['action']} {option['args']}")
        out = {"design_state": new, "route": "gates"}
        if option["action"] == "relax_rule":
            disabled = list(new["disabled_rules"])
            hist_elig, funnel, _ = eligible_history(s, disabled)
            out.update(hist_elig=hist_elig, funnel=funnel, disabled_rules=disabled)
        return out
    ctx["say"]("   no allowed fix is left")
    return {"design_state": state, "route": "report"}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def n_report(s, ctx):
    ctx["say"]("== 9. Writing the report ==")
    result = build_result({**s, "confirmations": ctx["confirmations"]}, ctx)
    result["confirmations"] = ctx["confirmations"]
    result["run_dir"] = str(ctx["run_dir"])
    narrative = ""
    if result["spec"] and ctx["llm"]:
        try:
            narrative = planner.narrate(ctx["llm"], facts_for_narration(result))
        except Exception as ex:  # noqa: BLE001 - the summary is a convenience; the numbers are in the report
            ctx["audit"]("narrative_error", message=str(ex))
    result["narrative"] = narrative
    result["text"] = render_markdown(result, narrative)
    run_dir = ctx["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    (run_dir / "report.md").write_text(result["text"])
    ctx["audit"]("end", verdict=result["verdict"], tier=result["tier"], reason=result["reason"])
    return {"result": result}
