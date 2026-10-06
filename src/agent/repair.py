"""Repair loop: what the agent may change when a gate fails, and how it picks.

Code builds a short list of concrete options from the gate diagnostics; the LLM picks one (or code picks the first
when there is no LLM or its answer is invalid). The estimate never reaches this module, so a repair cannot be
chosen because of the result it would give.
"""

import json

import numpy as np
from pydantic import BaseModel

from src.agent.frames import covariates
from src.online import llm_json

GATE_ACTIONS = {
    "G1": ["drop_covariate", "relax_rule", "drop_borrowing"],
    "G2": ["restrict_range", "drop_covariate"],
    "G3": ["richer_propensity", "restrict_range", "drop_covariate"],
    "G4": ["drop_cross_source", "relax_rule", "restrict_range", "drop_borrowing"],
    "G5": ["shorten_horizon", "drop_borrowing"],
    "G7": ["restrict_range", "drop_covariate", "drop_borrowing"],
}
MIN_KEPT_SHARE = 0.70  # a range restriction may not remove more than 30% of the treated patients
MIN_COVARIATES = 2


class Choice(BaseModel):
    choice: int
    reason: str = ""


def build_options(failed, spec, state, rules, study):
    """Concrete, not yet tried options for the failed gate, in default priority order."""
    gid, diag = failed["id"], failed["diag"]
    options = []
    for action in GATE_ACTIONS.get(gid, []):
        options += _options_for(action, gid, diag, spec, state, rules, study)
    tried = {(t["action"], json.dumps(t["args"], sort_keys=True)) for t in state["tried"]}
    return [o for o in options if (o["action"], json.dumps(o["args"], sort_keys=True)) not in tried]


def _options_for(action, gid, diag, spec, state, rules, study):
    smd = {**diag.get("smd_before", {}), **diag.get("smd_trial_vs_hist", {}), **diag.get("smd_main", {})}
    names = [c.name for c in covariates(spec, state)]

    def strength(name):  # largest |SMD| over this covariate's design columns
        vals = [abs(v) for k, v in smd.items() if k == name or k.startswith((f"{name}=", f"log1p({name})"))]
        return max(vals, default=0.0)

    if action == "drop_covariate":
        if len(names) <= MIN_COVARIATES:
            return []
        miss = diag.get("missing_share_trial", {})
        pool = [c for c in covariates(spec, state) if not c.essential]
        pool.sort(key=lambda c: -(miss.get(c.name, 0.0) if gid == "G1" else strength(c.name)))
        return [
            {"action": action, "args": {"name": c.name}, "why": f"{c.name}: strongest imbalance or most missing"}
            for c in pool[:3]
        ]
    if action == "restrict_range":
        return _range_options(diag.get("ranges", {}), study, strength)
    if action == "richer_propensity":
        if state["ps_level"] >= 2:
            return []
        return [{"action": action, "args": {"level": state["ps_level"] + 1}, "why": "add squares / products"}]
    if action == "drop_cross_source":
        pool = [c for c in covariates(spec, state) if c.cross_source]
        if len(pool) <= 1:
            return []
        pool.sort(key=lambda c: -strength(c.name))
        return [
            {"action": action, "args": {"name": c.name}, "why": f"{c.name}: differs most between the two sources"}
            for c in pool[:3]
        ]
    if action == "relax_rule":
        live = [r for r in rules if r["id"] not in state["disabled_rules"] and r.get("rule")]
        return [
            {"action": action, "args": {"id": r["id"]}, "why": f"{r['id']}: {r.get('text', '')[:80]}"} for r in live[:3]
        ]
    if action == "shorten_horizon":
        years = diag.get("max_supported_years")
        if years is None or years >= state["horizon_years"] - 1e-9:
            return []
        return [{"action": action, "args": {"years": round(years, 2)}, "why": "longest horizon every group supports"}]
    if action == "drop_borrowing" and state["hybrid"]:
        return [{"action": action, "args": {}, "why": "analyse the trial alone; no historical data"}]
    return []


def _range_options(ranges, study, strength):
    T, out = study["T"], []
    treated = T["__A"].to_numpy() == 1
    for name in sorted(ranges, key=lambda n: -strength(n)):
        r = ranges[name]
        lo, hi = max(r["treated"][0], r["control"][0]), min(r["treated"][1], r["control"][1])
        v = T[name].to_numpy(float)
        if lo <= r["all"][0] and hi >= r["all"][1]:
            continue
        if np.mean((v[treated] >= lo) & (v[treated] <= hi)) < MIN_KEPT_SHARE:
            continue
        out.append(
            {
                "action": "restrict_range",
                "args": {"col": name, "lo": round(lo, 4), "hi": round(hi, 4)},
                "why": f"keep {name} where both arms have patients",
            }
        )
    return out[:3]


CHOOSER_SYSTEM = """You repair an analysis that failed one pre-specified check (a 'gate').
You get the failed gate with its numbers and a numbered list of allowed fixes. Pick ONE fix.
The options are listed from the smallest change to the largest. Pick the smallest change that can plausibly fix the
failing numbers; pick a later option only when the earlier ones cannot (dropping the historical data is the last
resort). Never pick a fix because of what result you expect (you do not know the result); do not repeat a tried fix.
Answer JSON only: {"choice": <number>, "reason": "..."}"""


def choose(llm, failed, options, state):
    """The LLM's pick (index into ``options``) and its reason; falls back to the first option."""
    if llm is None or len(options) == 1:
        return 0, "default order" if llm is None else "only option left"
    request = {
        "failed_gate": {k: failed[k] for k in ("id", "name", "checks", "note")},
        "diagnostics": failed["diag"],
        "options": [{"number": i, **o} for i, o in enumerate(options)],
        "already_tried": state["tried"],
    }
    try:
        answer = llm_json(llm, CHOOSER_SYSTEM, json.dumps(request, default=str), Choice)
    except ValueError:
        return 0, "LLM answer invalid: default order"
    if not 0 <= answer.choice < len(options):
        return 0, "LLM chose a number outside the list: default order"
    return answer.choice, answer.reason


def describe(option, state):
    """One plain sentence for the report and the summary (written by code, not by the LLM)."""
    a, args = option["action"], option["args"]
    return {
        "drop_covariate": f"removed the covariate '{args.get('name')}' from the analysis",
        "restrict_range": f"limited the analysis to patients with {args.get('col')} between "
        f"{args.get('lo')} and {args.get('hi')}",
        "richer_propensity": "made the treatment-assignment model more flexible (squares and products)",
        "drop_cross_source": f"stopped comparing '{args.get('name')}' between the trial and the historical data",
        "relax_rule": f"stopped applying eligibility rule {args.get('id')} to the historical patients",
        "shorten_horizon": f"shortened the horizon from {state['horizon_years']:g} to {args.get('years')} years "
        "because follow-up in at least one group was too short",
        "drop_borrowing": "stopped using the historical data: the result uses the trial alone",
    }[a]


def apply_option(state, option):
    """New design state with the option applied."""
    new = {k: (list(v) if isinstance(v, list) else v) for k, v in state.items()}
    a, args = option["action"], option["args"]
    if a == "drop_covariate":
        new["drop_cov"].append(args["name"])
    elif a == "restrict_range":
        new["restrict"].append((args["col"], args["lo"], args["hi"]))
    elif a == "richer_propensity":
        new["ps_level"] = args["level"]
    elif a == "drop_cross_source":
        new["drop_cross"].append(args["name"])
    elif a == "relax_rule":
        new["disabled_rules"].append(args["id"])
    elif a == "shorten_horizon":
        new["horizon_years"] = args["years"]
    elif a == "drop_borrowing":
        new["hybrid"] = False
    else:
        raise ValueError(f"unknown action {a}")
    return new
