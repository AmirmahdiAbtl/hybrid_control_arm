import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.request import Request, urlopen  # module-level so tests can patch ``urlopen``

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.rules import RuleError, evaluate_rule

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Load ``.env`` from the project folder, if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


load_env()


# ----------------------------------------------------------------------------
# Pydantic schemas for LLM output
# ----------------------------------------------------------------------------
class Transform(BaseModel):
    """Declarative column transform; never free code.

    ``kind``: identity, rename, map_values (``mapping``), unit_scale (``multiply``/``add``), bin (``edges`` +
    ``labels``) or derive (``expr``, whitelisted expression over B columns). ``apply_to`` says which dataset gets
    the transform: normally B (converted into A's schema); a ``bin`` of a numeric A column into B's categories
    uses ``A``.
    """

    kind: Literal["identity", "rename", "map_values", "unit_scale", "bin", "derive"] = "identity"
    mapping: Optional[dict[str, Union[str, int, float]]] = None
    multiply: float = 1.0
    add: float = 0.0
    edges: Optional[list[float]] = None
    labels: Optional[list[str]] = None
    expr: Optional[str] = None
    apply_to: Literal["A", "B"] = "B"

    @model_validator(mode="after")
    def _complete(self):
        if self.kind == "map_values" and not self.mapping:
            raise ValueError("map_values needs 'mapping'")
        if self.kind == "bin" and not (self.edges and self.labels and len(self.labels) == len(self.edges) - 1):
            raise ValueError("bin needs 'edges' and 'labels' (one label fewer than edges)")
        if self.kind == "derive" and not self.expr:
            raise ValueError("derive needs 'expr'")
        return self


class HarmonizeAnswer(BaseModel):
    """LLM answer for one A column."""

    match: Optional[list[str]] = None
    transform: Transform = Field(default_factory=Transform)
    same_definition: Literal["yes", "no", "unsure"] = "unsure"
    definition_note: str = ""
    reason: str = ""

    @field_validator("match", mode="before")
    @classmethod
    def _as_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v or None


class GroupAnswer(BaseModel):
    """LLM answers for several A columns derived from the same B columns (e.g. time + event)."""

    answers: dict[str, HarmonizeAnswer]


class Criterion(BaseModel):
    """One eligibility criterion split out of the trial text."""

    id: str
    section: Literal["inclusion", "exclusion", "control_arm"] = "inclusion"
    text: str
    kind: Literal["rule", "text", "not_checkable"] = "not_checkable"
    rule: Optional[str] = None
    column: Optional[str] = None
    question: Optional[str] = None
    note: str = ""


class CriteriaList(BaseModel):
    """LLM answer for the eligibility text."""

    criteria: list[Criterion]


# ----------------------------------------------------------------------------
# LLM (Azure)
# ----------------------------------------------------------------------------
def llm_status():
    """Short description of the LLM configuration, or None if it is not configured."""
    has_key = os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
    has_target = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE") or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
    if has_key and has_target and os.getenv("ANTHROPIC_MODEL"):
        return f"Claude on Azure Foundry: {os.environ['ANTHROPIC_MODEL']}"
    return None


def get_llm():
    """Return ``llm(system, user) -> str`` using the Anthropic SDK on Azure Foundry, or None if not configured.

    ``anthropic.AnthropicFoundry()`` reads ``ANTHROPIC_FOUNDRY_API_KEY`` and ``ANTHROPIC_FOUNDRY_RESOURCE`` (or
    ``ANTHROPIC_FOUNDRY_BASE_URL``) itself; the deployment name is ``ANTHROPIC_MODEL``.
    """
    if not llm_status():
        return None
    model = os.environ["ANTHROPIC_MODEL"]

    def llm(system, user):  # the client is only created at the first real call
        import anthropic

        resource = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE", "")
        # the SDK wants only the resource name; accept a full URL too (https://<name>.services.ai.azure.com/anthropic/)
        m = re.match(r"https?://([^./]+)\.services\.ai\.azure\.com", resource)
        client = anthropic.AnthropicFoundry(**({"resource": m.group(1)} if m else {}))
        r = client.messages.create(
            model=model, max_tokens=4096, system=system, messages=[{"role": "user", "content": user}]
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    return llm


def llm_json(llm, system, user, model_cls, retries=1):
    """Call the LLM, parse JSON and validate it with pydantic (one retry that shows the error).

    Parameters
    ----------
    llm : callable
        ``llm(system, user) -> str``.
    model_cls : type
        Pydantic model of the answer.

    Returns
    -------
    pydantic.BaseModel
    """
    prompt, err = user, None
    for _ in range(retries + 1):
        try:
            return model_cls.model_validate(_extract_json(llm(system, prompt)))
        except (ValueError, ValidationError) as e:  # json errors are ValueError too
            err = e
            prompt = f"{user}\n\nYour previous answer was invalid ({str(e)[:400]}). Return only valid JSON."
    raise ValueError(f"LLM did not return valid JSON for {model_cls.__name__}: {err}")


def _extract_json(text):
    """The JSON object in an LLM answer (also inside a ```json fence)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    i, j = text.find("{"), text.rfind("}")
    return json.loads(text[i : j + 1])


HARMONIZE_SYSTEM = """You harmonize clinical trial datasets. Dataset \
A is the NEW trial (target schema), dataset B is an OLD study.
For ONE column of A you get its profile and the profiles of candidate columns of B. Answer with JSON only:
{"match": ["B_col", ...] or null, "transform": {...}, "same_definition": \
"yes"|"no"|"unsure", "definition_note": "...", "reason": "..."}
- match: one B column, several B columns when A's variable must be DERIVED from them, or null if none fits.
- transform converts B into A's format. Declarative only, one of:
  {"kind":"identity"} | {"kind":"rename"} |
  {"kind":"map_values","mapping":{"<B value>":"<A value>", ...}} (keys are B values as text) |
  {"kind":"unit_scale","multiply":x,"add":y} (A = B*x + y) |
  {"kind":"bin","edges":[lo,...,hi],"labels":[...],"apply_to":"A"|"B"} (intervals are (e0,e1], (e1,e2], ...; \
use apply_to "A" when A is numeric and only B's coarse categories exist, so A is binned to B's categories) |
  {"kind":"derive","expr":"<expression over B columns>"} (allowed: column names, \
numbers, strings, comparisons, and/or/not, + - * /, and `x if cond else y`).
- map_values targets must be A's values EXACTLY as written in \
A's profile (same case and spelling, e.g. "Pre", not "pre").
- Map codes by their MEANING, not by their rank: a numeric grade 2 is grade II even if B has no grade 1.
- Expressions may not call functions (no min, max, abs, ...): write `a if a < b else b` instead of min(a, b).
- Time-to-event outcomes: A's time column and event column are \
one pair (see all_A_columns). If B splits a composite event
  (e.g. recurrence or death) over several time and indicator \
columns, derive A's time AND event from B with the SAME condition,
  so they stay consistent: event = 1 at the first observed event. \
If one component's follow-up ends earlier without an event
  (e.g. no recurrence recorded up to its time, but death recorded \
later), events after that earlier time are not observed:
  censor at the earlier time (event 0). Use all B columns this needs as match.
- same_definition: do the two columns measure the same thing in the same way? \
Use a note for differences (e.g. different category ranges, scoring systems).
Return only JSON."""


HARMONIZE_GROUP_SYSTEM = (
    HARMONIZE_SYSTEM.replace("Return only JSON.", "")
    + """
You now get SEVERAL A columns that are derived from the SAME B columns (typically the time and the event column of one
outcome), with the answers given for each of them separately. Make them consistent and answer with JSON only:
{"answers": {"<A column>": {"match": [...], "transform": {...}, \
"same_definition": "...", "definition_note": "...", "reason": "..."}, ...}}
Both expressions must use the SAME condition, so that for \
every patient: if an event is counted, event = 1 and time = the
time of that event; otherwise event = 0 and time = the censoring time (the last time the patient was observed event-free
for EVERY component of the composite event).
Return only JSON."""
)


# ----------------------------------------------------------------------------
# Jev
# ----------------------------------------------------------------------------
SAME_VAR = "These two columns record the same clinical variable (units or coding may differ)."

_LOOP = None


def jev_status():
    """True if a Jev API key is configured."""
    return bool(os.getenv("TYPESAFE_API_KEY"))


def make_async_jev(transport=None):
    """Create an ``AsyncTypeSafeClient`` (reads ``TYPESAFE_API_KEY``; ``transport`` is for tests)."""
    from typesafe_sdk import AsyncTypeSafeClient

    return AsyncTypeSafeClient(transport=transport) if transport else AsyncTypeSafeClient()


def run_async(coro):
    """Run a coroutine from sync code on one shared background event loop.

    A single loop lets one Jev client be reused across calls, and also works when the caller (e.g. a web server thread)
    already has a running loop.
    """
    global _LOOP
    if _LOOP is None:
        _LOOP = asyncio.new_event_loop()
        threading.Thread(target=_LOOP.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _LOOP).result()


def jev_score_columns(client, prof_a, prof_b, mode="pairwise", concurrency=8, chunk=25):
    """Score how likely each (A column, B column) pair is the same variable.

    Parameters
    ----------
    client : typesafe_sdk.AsyncTypeSafeClient
    prof_a, prof_b : dict
        Column profiles (name -> profile).
    mode : {'pairwise', 'ranking'}
        ``pairwise``: one Noul per pair, independent probabilities. ``ranking``: one Choice per A column over all
        B columns plus 'none' (probabilities sum to 1).
    concurrency : int
        Parallel requests.
    chunk : int
        B columns per request in ``pairwise`` mode.

    Returns
    -------
    dict
        ``{a_col: {b_col: score}}``.
    """
    if mode == "ranking":
        if len(prof_b) > 254:
            raise ValueError("ranking mode supports at most 254 B columns (Choice limit is 255 labels)")
        return run_async(_ranking_scores(client, prof_a, prof_b, concurrency))
    return run_async(_pairwise_scores(client, prof_a, prof_b, concurrency, chunk))


async def _pairwise_scores(client, prof_a, prof_b, concurrency, chunk):
    from typesafe_sdk import Noul

    sem = asyncio.Semaphore(concurrency)
    scores = {a: {} for a in prof_a}

    async def score_chunk(a, b_cols):
        state = {"column_A (new trial)": prof_a[a], "columns_B (old study)": {b: prof_b[b] for b in b_cols}}
        questions = {
            b: Noul(
                instructions=f"{SAME_VAR} Column A is '{a}'; column B is '{b}'.",
                criteria={
                    "true": f"'{a}' and '{b}' measure the same clinical variable.",
                    "false": "They measure different things.",
                },
            )
            for b in b_cols
        }
        async with sem:
            resp = await client.system_one(state, questions)
        for b in b_cols:
            scores[a][b] = float(resp.nouls[b].noul)

    b_all = list(prof_b)
    await asyncio.gather(*[score_chunk(a, b_all[i : i + chunk]) for a in prof_a for i in range(0, len(b_all), chunk)])
    return scores


async def _ranking_scores(client, prof_a, prof_b, concurrency):
    from typesafe_sdk import Choice

    sem = asyncio.Semaphore(concurrency)
    scores = {}
    criteria = {b: _profile_summary(p) for b, p in prof_b.items()}
    criteria["none"] = "No column of B records this variable."

    async def score_column(a):
        state = {"column_A (new trial)": prof_a[a]}
        question = {
            "match": Choice(
                instructions="Which column of the old study records the same clinical variable "
                "as column A (units or coding may differ)?",
                criteria=criteria,
            )
        }
        async with sem:
            resp = await client.system_one(state, question)
        scores[a] = {b: float(resp.choices["match"].probabilities.get(b, 0.0)) for b in prof_b}

    await asyncio.gather(*[score_column(a) for a in prof_a])
    return scores


def _profile_summary(prof):
    """One-line text version of a column profile (used as choice description)."""
    bits = [prof.get("label") or "", prof.get("dtype", ""), f"{prof.get('n_unique')} unique"]
    if "min" in prof:
        bits.append(f"range {prof['min']}..{prof['max']}")
    if prof.get("top_values"):
        bits.append("values " + ", ".join(str(k) for k in list(prof["top_values"])[:5]))
    return "; ".join(str(b) for b in bits if b)


def jev_verify_rules(client, items, concurrency=8):
    """Jev checks that each rule implements its criterion.

    Parameters
    ----------
    items : list of dict
        ``id``, ``text``, ``rule``, optional ``note`` and ``profiles`` (used columns).

    Returns
    -------
    dict
        ``{id: P(faithful)}``; below 0.7 should be flagged.
    """
    return run_async(_rule_faithfulness(client, items, concurrency))


async def _rule_faithfulness(client, items, concurrency):
    from typesafe_sdk import Noul

    sem = asyncio.Semaphore(concurrency)
    out = {}

    async def verify(c):
        state = {"criterion": c["text"], "rule": c["rule"], "note": c.get("note", ""), "columns": c.get("profiles", {})}
        question = {
            "ok": Noul(
                instructions="The rule faithfully implements the criterion (given what the data records).",
                criteria={
                    "true": "The rule selects exactly the patients the criterion describes, or the note "
                    "documents a reasonable approximation.",
                    "false": "The rule does not match the criterion.",
                },
            )
        }
        async with sem:
            resp = await client.system_one(state, question)
        out[c["id"]] = float(resp.nouls["ok"].noul)

    await asyncio.gather(*[verify(c) for c in items])
    return out


def jev_text_eligibility(client, texts, questions, concurrency=8):
    """Sync wrapper of :func:`jev_text_answers`."""
    return run_async(jev_text_answers(client, texts, questions, concurrency))


async def jev_text_answers(client, texts, questions, concurrency=8):
    """Jev answers yes/no questions for each patient's free text.

    Parameters
    ----------
    client : typesafe_sdk.AsyncTypeSafeClient
    texts : list of str
        One free-text entry per patient.
    questions : dict
        ``{name: question}``, all asked in one call per patient.

    Returns
    -------
    list of dict
        Per patient ``{name: P(yes)}``.
    """
    from typesafe_sdk import Noul

    sem = asyncio.Semaphore(concurrency)
    nouls = {k: Noul(instructions=q) for k, q in questions.items()}

    async def answer(text):
        async with sem:
            resp = await client.system_one(str(text), nouls)
        return {k: float(resp.nouls[k].noul) for k in questions}

    return await asyncio.gather(*[answer(t) for t in texts])


def jev_column_roles(client, prof, concurrency=8):
    """Classify columns as baseline / treatment / outcome / identifier. Returns ``{col: role}``."""
    return run_async(_column_roles(client, prof, concurrency))


async def _column_roles(client, prof, concurrency):
    from typesafe_sdk import Choice

    sem = asyncio.Semaphore(concurrency)
    roles = {}

    async def classify(c):
        question = {
            "role": Choice(
                instructions="What role does this column play in a clinical trial dataset?",
                criteria={
                    "baseline": "patient characteristic measured before treatment (age, stage, receptors ...)",
                    "treatment": "treatment assignment or treatment received",
                    "outcome": "follow-up time, event or response after treatment start",
                    "identifier": "patient id, site, calendar year or other non-clinical key",
                },
            )
        }
        async with sem:
            resp = await client.system_one(prof[c], question)
        roles[c] = resp.choices["role"].choice

    await asyncio.gather(*[classify(c) for c in prof])
    return roles


# ----------------------------------------------------------------------------
# ClinicalTrials.gov
# ----------------------------------------------------------------------------
def fetch_eligibility(nct_id):
    """Fetch the eligibility text of a registered trial.

    Parameters
    ----------
    nct_id : str
        E.g. ``NCT00012345``.

    Returns
    -------
    str
        ``protocolSection.eligibilityModule.eligibilityCriteria``.
    """
    nct_id = nct_id.strip().upper()
    req = Request(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}", headers={"Accept": "application/json"})
    with urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    try:
        return data["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    except KeyError as e:
        raise ValueError(f"{nct_id} has no eligibility criteria in the registry") from e


# ----------------------------------------------------------------------------
# Eligibility: the LLM splits the trial text into criteria
# ----------------------------------------------------------------------------
ELIGIBILITY_SYSTEM = """You turn clinical trial eligibility text into checkable criteria for a historical patient table.
You get the text and the available columns (profiles). Split the text into individual criteria and answer JSON only:
{"criteria": [{"id": "I1", "section": "inclusion"|"exclusion"|"control_arm", \
"text": "...", "kind": "rule"|"text"|"not_checkable",
 "rule": "...", "column": "...", "question": "...", "note": "..."}]}
- kind "rule": a boolean expression over the available column \
names that is TRUE for ELIGIBLE patients (negate exclusions).
  Allowed: column names, numbers, strings, comparisons (also chained: \
18 <= age <= 65), `in [..]`, `not in [..]`, and/or/not, & | ~, + - * /.
  No function calls, no attributes. Compare a column to a value (write `chemo == 1`, never just `chemo`).
  Combine with `and` / `or` / `not` (write `hormon == 0 and chemo == \
1`; `hormon == 0 & chemo == 1` is wrong because & binds before ==).
- If no column records what the criterion asks, use kind \
"not_checkable". Never write a rule that is true for every patient
  just to have a rule (e.g. `size in [all size groups]` does not check operability or metastases).
- kind "text": the answer is only in a free-text column: give \
that `column` and a yes/no `question` about one patient's text.
- kind "not_checkable": nothing in the data records it (e.g. consent).
- note: say when the rule is an approximation (e.g. only chemotherapy yes/no is recorded, not the regimen).
- section "control_arm" is for criteria that define the control arm (e.g. no hormonal therapy).
Return only JSON."""


def extract_criteria(llm, text, profiles, df=None):
    """LLM splits the eligibility text into criteria with rules.

    Parameters
    ----------
    llm : callable
        ``llm(system, user) -> str``.
    text : str
        Eligibility text (pasted or from :func:`fetch_eligibility`).
    profiles : dict
        Column profiles of the historical data (what the rules may refer to).
    df : pandas.DataFrame, optional
        Historical data. If given, rules that do not run are sent back to the LLM once with the error.

    Returns
    -------
    list of Criterion
    """
    req = {"eligibility_text": text, "columns": profiles}
    for _ in range(2):
        criteria = llm_json(llm, ELIGIBILITY_SYSTEM, json.dumps(req, default=str), CriteriaList).criteria
        if df is None:
            return criteria
        errors = {}
        for c in criteria:
            if c.kind == "rule" and c.rule:
                try:
                    evaluate_rule(c.rule, df)
                except RuleError as e:
                    errors[c.id] = {"rule": c.rule, "error": str(e)}
        if not errors:
            return criteria
        req["your_previous_answer"] = [c.model_dump() for c in criteria]
        req["rules_that_do_not_run"] = errors
    return criteria
