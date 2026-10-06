"""LLM calls of the agent: the plan, the critic's review and the plain-language summary."""

import json
from typing import Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel

from src.agent.spec import StudySpec
from src.online import llm_json

PLAN_SYSTEM = """You design the analysis plan for a hybrid-control study: a clinical trial table plus a historical
patient table that was already converted into the trial's column names. You get the trial record, the columns of both
tables (profiles) and notes on columns whose definitions differ. Return a plan as JSON only, with exactly these keys:
{"treatment": col, "control_value": "...", "time": col, "event": col, "event_values": ["..."],
 "time_unit": "days"|"weeks"|"months"|"years", "horizon_years": number, "landmark_days": number,
 "covariates": [{"name": col, "kind": "numeric"|"binary"|"categorical"|"ordinal", "cross_source": bool,
 "essential": bool, "why": "..."}], "rationale": "..."}
Rules:
- treatment: the trial column that says which arm each patient was in; control_value: the value of the control arm,
  written as text. Use the dictionary labels to decide what each code means.
- time and event: the time-to-first-event outcome that matches the trial's primary outcome. event_values lists the
  event column's values that mean the event happened (as text); every other value is censored. Read the labels: in a
  status column with 0 = censored, 1 = transplant, 2 = death, a death outcome is ["2"].
- horizon_years: the clinically meaningful time point (use the trial's primary outcome time frame if it gives one).
  Most patients must still be under follow-up there: stay below the 80th percentile of the time column.
- landmark_days: 0, unless the clock starts before treatment does (then the delay in days, so that nobody can be
  counted as treated only because they survived long enough).
- covariates: 4 to 12 baseline variables measured BEFORE treatment that predict the outcome or influenced who got
  treated. Include EVERY strong baseline predictor that exists in the trial table, even when the historical table
  lacks it or defines it differently: set cross_source to false for those (the trial-only adjustment needs them; only
  cross_source covariates are used to borrow from history). Never leave a variable out just because the historical
  table lacks it. Never include ids, dates, the treatment, the time, the event, anything measured after treatment
  starts, or another outcome or complication. essential is true for the 1 to 3 strongest known prognostic or
  assignment factors; the repair loop may never drop those.
- If 'feedback' is present, fix every point in it and keep everything else.
Return only JSON."""

CRITIC_SYSTEM = """You are a skeptical clinical epidemiologist reviewing an analysis plan before anyone looks at
outcomes by arm. You get the trial record, the plan and the column profiles. List real problems only: missing key
confounders that exist in the data, covariates that are consequences of treatment or outcome, a horizon that does not
match the trial's primary outcome, a wrong event definition, a missing landmark. Careful: control_value is a value of
the TREATMENT column; event_values are values of the EVENT column; they are unrelated. 'plan_in_words' states the plan
unambiguously. Missing-data handling is fixed by the pipeline (complete cases, at most 5% missing per covariate): do
not raise it. Answer JSON only:
{"verdict": "ok"|"revise", "concerns": [{"issue": "...", "severity": "high"|"medium"|"low",
 "fix": "add_covariate"|"remove_covariate"|"change_horizon"|"change_landmark"|"change_event"|"other",
 "column": "..."}]}"""

NARRATE_SYSTEM = """You explain the result of an automated causal analysis to a clinician in plain language: 6 to 9
short sentences, no jargon without a few words of explanation. Use ONLY the numbers in 'facts', copied exactly; do not
compute or add any number. Cover: what was asked, which data, what the checks found (and any repairs and why), the
result, and the main limit. Quote 'unmeasured_confounding' as written when it is present. Do not call the result
proof. Describe the outcome exactly as written in 'outcome'.
Plain paragraphs only: no headings, no bullet points, no markdown."""


class Concern(BaseModel):
    issue: str
    severity: Literal["high", "medium", "low"] = "medium"
    fix: Literal["add_covariate", "remove_covariate", "change_horizon", "change_landmark", "change_event", "other"] = (
        "other"
    )
    column: str = ""


class Critique(BaseModel):
    verdict: Literal["ok", "revise"] = "ok"
    concerns: list[Concern] = []


def fetch_trial_brief(nct_id):
    """Title, conditions, arms and primary outcomes of a registered trial (ClinicalTrials.gov API v2)."""
    nct_id = nct_id.strip().upper()
    req = Request(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}", headers={"Accept": "application/json"})
    with urlopen(req, timeout=20) as r:
        proto = json.loads(r.read().decode("utf-8"))["protocolSection"]
    ident, arms = proto.get("identificationModule", {}), proto.get("armsInterventionsModule", {})
    outcomes = proto.get("outcomesModule", {})
    return {
        "nct_id": nct_id,
        "title": ident.get("officialTitle") or ident.get("briefTitle", ""),
        "conditions": proto.get("conditionsModule", {}).get("conditions", []),
        "arms": [
            {"label": a.get("label"), "type": a.get("type"), "description": (a.get("description") or "")[:300]}
            for a in arms.get("armGroups", [])
        ],
        "primary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")} for o in outcomes.get("primaryOutcomes", [])
        ],
        "eligibility": proto.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
    }


def compact_profile(profile):
    """Smaller column profile for prompts (drops unit guesses and long value lists)."""
    keep = ("label", "dtype", "n_unique", "pct_missing", "min", "max", "median", "levels", "top_values")
    return {c: {k: p[k] for k in keep if p.get(k) not in (None, "")} for c, p in profile.items()}


def draft_plan(llm, brief, prof_trial, prof_hist, notes, feedback=None, previous=None):
    """The LLM's plan as a ``StudySpec`` (one retry on invalid JSON is inside ``llm_json``)."""
    request = {
        "trial_record": {k: v for k, v in (brief or {}).items() if k != "eligibility"},
        "eligibility_text": (brief or {}).get("eligibility", "")[:2500],
        "trial_columns": compact_profile(prof_trial),
        "historical_columns": compact_profile(prof_hist) if prof_hist else "no historical table",
        "definition_notes": notes,
    }
    if feedback:
        request["feedback"] = feedback
        request["previous_plan"] = previous.model_dump() if previous else None
    return llm_json(llm, PLAN_SYSTEM, json.dumps(request, default=str), StudySpec)


def plan_in_words(spec):
    """The plan as plain sentences, so that no field can be misread."""
    adjust = ", ".join(c.name + ("" if c.cross_source else " (trial only)") for c in spec.covariates)
    return [
        f"Patients are in the treated arm if column '{spec.treatment}' is not '{spec.control_value}'; "
        f"'{spec.control_value}' is the control arm.",
        f"The event is: column '{spec.event}' equals one of {spec.event_values}. Any other value is censored.",
        f"Time is column '{spec.time}' in {spec.time_unit}; horizon {spec.horizon_years:g} years; "
        f"landmark {spec.landmark_days:g} days.",
        f"Adjusted for: {adjust}.",
    ]


def critique_plan(llm, brief, spec, prof_trial):
    request = {
        "trial_record": {k: v for k, v in (brief or {}).items() if k != "eligibility"},
        "plan": spec.model_dump(),
        "plan_in_words": plan_in_words(spec),
        "trial_columns": compact_profile(prof_trial),
    }
    return llm_json(llm, CRITIC_SYSTEM, json.dumps(request, default=str), Critique)


def narrate(llm, facts):
    """Plain-language summary. The LLM only phrases numbers that the code computed."""
    return clean_narrative(llm(NARRATE_SYSTEM, json.dumps(facts, default=str)))


def clean_narrative(text):
    """Drop markdown headings the model adds despite the instruction; keep the paragraphs."""
    lines = [ln for ln in text.strip().splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines).strip()
