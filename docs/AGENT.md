# Hybrid Control Arm: user guide

## What this is

A clinical trial often has a small control arm. An older study may have patients who look like the trial's controls. This tool decides, with checks you can read, whether those old patients can be used to improve the comparison, and then estimates the effect.

You give it two tables and, if the trial is registered, its ClinicalTrials.gov id. An AI agent then:

1. matches the old study's columns to the new trial's,
2. keeps only the old patients who would have qualified for the trial,
3. writes an analysis plan,
4. runs a fixed series of checks (the *gates*),
5. repairs a failed gate by changing one thing and running again,
6. writes a report with a verdict, an estimate and a full audit log.

The code is not tied to one study: no column name, disease or trial is written into it. The plan is written by an LLM for each dataset and validated by code. It has been run for real on one pair of studies only (GBSG-2 and Rotterdam, breast cancer), so treat any other data as untested.

> **Read this first.** The plan is drafted by an LLM and checked by code. A clinician has not reviewed it. Code can check that a plan is valid, not that it is clinically right: if the LLM leaves out an important confounder, every gate can still pass. Every report says so, and the best tier a run can reach is *SUPPORTIVE*. This is a research prototype, not a medical device.

## What you need

| Input | Required | Notes |
|---|---|---|
| New trial table (CSV) | yes | One row per patient. It needs a treatment column (which arm), a time column and an event column. |
| Historical study table (CSV) | yes | Patients from an older study. No treatment column is needed. Column names and codings may differ from the trial's. |
| Column dictionaries (JSON) | no | `{"column": "description"}` for each table. They improve the column matching a lot. |
| ClinicalTrials.gov id | no | `NCT` and 8 digits. Gives the agent the title, arms, primary outcome and eligibility text. |
| Eligibility text | no | Pasted criteria. Used instead of the registry's. Without any eligibility text, no historical patient is filtered out. |

Keys, in a file called `.env` in the project folder:

| Variable | What for |
|---|---|
| `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_MODEL` | The LLM (Claude on Azure Foundry). **Required.** |
| `TABPFN_TOKEN` | TabPFN through its API. Needed for the default backend. |
| `TYPESAFE_API_KEY` | Jev, used for column matching. Optional: without it every column goes to the LLM. |

## Install and start

```bash
uv venv .venv                                              # once
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m src.web                                # the web app on http://localhost:5050
```

Useful options: `--port 8080`, `--host 0.0.0.0` (to reach it from another machine; there is no login, so only do this on a trusted network), `--debug`.

The same agent is available without the browser:

```bash
.venv/bin/python -m src.cli.auto_run --trial trial.csv --hist old.csv --nct NCT00000000
.venv/bin/python -m src.cli.auto_run --trial trial.csv --hist old.csv --eligibility-file criteria.txt --yes
```

## Two ways to run

Both use the same agent, the same plan checks and the same gates. They differ only in who answers the agent's questions.

### Single loop

You press **Start**. The agent goes from the data to the verdict without stopping. Every question it would have asked you is answered "yes" automatically.

- Use it to try a dataset quickly, or when you will read the whole report afterwards anyway.
- The report says **"AI-drafted, auto-confirmed"**: no human saw the mapping, the rules or the plan before they were used. Read them in the report before you trust the result.
- On the command line this is `--yes`.

### Step by step

The agent stops and waits at the points that need judgement. The page shows the question, with **Continue**, **Stop the run** and a box to type a change. The run does not move until you answer.

| Step | What it shows you | What you can answer |
|---|---|---|
| **Column mapping** | Each trial column with the historical column the agent matched it to. Safe matches are marked OK. Matches where the definition differs, or where a check failed, are marked FLAGGED and are *not used* unless you approve them. | `y` use the safe matches. `n` stop. `approve col1, col2` also use these flagged ones. `reject col3` drop this one. |
| **Eligibility rules** | The rules the LLM wrote from the trial's criteria, and how many historical patients each removes. | `y` apply them. `n` stop. `disable I2, I3` apply all but these. |
| **The plan** | The full plan: treatment, control value, event coding, horizon, landmark, covariates. | `y` freeze it. `n` stop. Or describe a change in words ("also adjust for platelet"): the planner redrafts, up to two times. |
| **Each repair** | The gate that failed, the fix the agent chose, and why. | `y` apply it. `n` skip this fix; the agent proposes the next one. If none is left, the run ends NOT ACCEPTED. |

While you decide, the page shows the nine steps as a progress strip, the live log, and the gates as they pass or fail. **Stop at the next question** ends the run at the next confirmation point, not instantly.

Your answers are written to the audit log and appear in the report under *Confirmations*.

## The steps in detail

1. **Read.** Both tables are profiled (type, range, missing values). If you gave an id, the registry record is fetched.
2. **Match columns.** Jev scores every pair of trial and historical columns. High-scoring pairs still go through deterministic checks (type, unit, category overlap). The LLM decides the hard cases and writes a declarative transform (rename, recode, unit change, bins). No LLM-written code is ever executed. Columns that mean something different (for example two different grading systems) are flagged.
3. **Eligibility.** The LLM turns the eligibility text into rules over the historical columns. A safe expression engine applies them: comparisons, `and` / `or` / `not`, `in` lists and arithmetic, nothing else. A rule that does not run is sent back to the LLM once.
4. **Plan.** The LLM drafts the plan as structured data. Code then checks it against the data and sends every problem back, up to three drafts:
   - the named columns exist; the treatment has exactly two values and the control value is one of them;
   - the event values occur, and the event rate is plausible;
   - time is numeric and positive; the horizon is inside the follow-up; the landmark is well below the horizon;
   - every covariate exists, has the stated type, has at most 5% missing values, and does not by itself separate the treated from the controls (a sign of leakage);
   - a covariate missing from the historical data, or mostly missing there, is marked *not comparable across sources* by code.
5. **Review.** A second LLM call reviews the plan. Concerns about missing variables, the horizon, the landmark or the event definition go back to the planner for **one** revision.
6. **Freeze.** The plan, the eligibility rules and the gate limits are hashed. From here on, nothing in the plan changes. Only the repair loop may change the *design* (see below), and every change is logged.
7. **Gates.** See the next section. Trial outcomes are read once, after the design gates.
8. **Repair.** When a gate fails, one fix is applied and the gates run again.
9. **Report.** Verdict, estimate, summary, files.

## The gates

Gates run in order. The first four use covariates only; the trial's outcomes are not read until all earlier gates have passed. All thresholds are in `src/agent/gates.py` (`LIMITS`).

| Gate | What it checks | Passes when |
|---|---|---|
| **G1 Data fitness** | Enough patients and events after the landmark and complete-case selection. | at least 30 treated and 30 controls; at least 20 events; at most 10% of trial patients lost to missing covariates; with borrowing, at least 50 historical patients and 15 events |
| **G2 Positivity** | Treated and control patients overlap on the covariates. A propensity model is fitted. | at most 5% of treated outside propensity 0.05 to 0.95; effective control sample at least half of the controls; no single control weighs more than 5% |
| **G3 Balance** | After weighting, the arms look alike, including squares and products of continuous covariates. | standardized differences at most 0.10 (main effects) and 0.25 (squares and products) |
| **G4 External support** | The historical patients resemble the treated patients (only when borrowing). | at most 10% of treated patients unsupported (density ratio above 20); effective historical sample at least 50 (or half); no historical patient weighs more than 10% |
| **G5 Follow-up and horizon** | Enough patients are still followed at the horizon in every group. | the probability of still being under follow-up at the horizon is at least 0.70 in every group, with at least 10 patients at risk |
| **G6 Prognostic model** | The model fitted on the historical patients predicts their outcomes better than a constant and not worse than a linear model (cross-validated). | model error at most the linear model's plus 2 standard errors, and below the constant's. If not, the agent switches to a linear score, then to none. This is logged as *PASS AFTER ACTION*. |
| **G7 Estimation integrity** | The weighting that produces the estimate worked. | entropy balancing converged exactly; effective control sample at least 20 and a third of the controls; no control weighs more than 15% |
| **G8 Borrowing feasibility** | Information only: the smallest equivalence margin the kept controls could ever show. | never fails |

**Where TabPFN is used.** In G6 and in the estimate: TabPFN is fitted on the historical patients to predict the outcome from the covariates shared by both studies. That prediction (the *prognostic score*) is then used as an extra adjustment variable for the trial. It never replaces the trial's own controls.

## Repairs

When a gate fails, code lists the fixes allowed for that gate, built from the gate's own numbers. The LLM chooses one from the list. The list is ordered from the smallest change to the largest, and the LLM is told to pick the smallest that can fix the failing numbers.

| Gate | Allowed fixes, in order |
|---|---|
| G1 | drop a covariate that is mostly missing · relax an eligibility rule · stop borrowing |
| G2 | restrict a numeric covariate to the range both arms share · drop a covariate |
| G3 | make the propensity model richer (squares, then products) · restrict a range · drop a covariate |
| G4 | stop comparing one covariate across sources · relax an eligibility rule · restrict a range · stop borrowing |
| G5 | shorten the horizon to the longest one every group supports · stop borrowing |
| G7 | restrict a range · drop a covariate · stop borrowing |

Rules that keep the loop honest:

- **The repair step never sees the estimate.** It is given the failing gate and its diagnostics only, so it cannot choose a fix because of the result it would produce. This is tested.
- Covariates the plan marked *essential* (strong prognostic factors) are never offered for dropping.
- A range restriction cannot remove more than 30% of the treated patients.
- A fix that was tried (or declined) is not offered again.
- At most `max repairs` fixes (default 6), then the run ends NOT ACCEPTED.
- Every design version that reached the trial outcomes is listed in the report with its estimate, so you can see whether repairs moved the result. After more than two repairs the tier drops from SUPPORTIVE to EXPLORATORY.

## What the LLM does, and does not do

| The LLM | The code |
|---|---|
| matches the hard columns, writes recodes | scores and checks every match; applies recodes |
| turns criteria into rules | applies rules with a restricted engine |
| drafts the plan; reviews it once | validates it; freezes and hashes it |
| picks one repair from a list | builds the list; applies the fix; reruns |
| writes a plain-language summary from numbers the code gives it | computes every gate, estimate, interval, E-value and the verdict |

The LLM cannot change a threshold, skip a gate, or write the verdict. Its calls are written to the audit log.

## Reading the result

- **Verdict.** *ACCEPTED* means every gate passed for the final design and an estimate was produced. It does **not** mean the treatment works. *NOT ACCEPTED* says which gate failed, or that the run was stopped.
- **Tier.** *SUPPORTIVE* at best, because the plan was drafted by an LLM and not reviewed by a clinician. *EXPLORATORY* after more than two repairs. *NONE* if not accepted.
- **Claim.** One sentence from the interval: "treated did better" if the whole interval is above zero, "did worse" if below, "effect not demonstrated" if it includes zero.
- **Estimate.** The difference in the probability of being event-free at the horizon, in percentage points (treated minus estimated control), with a 95% interval. It is an *ATT*: the effect in the patients who were treated. The method is entropy balancing of the controls plus an outcome model that includes the prognostic score.
- **Reference rows.** An unadjusted difference and an inverse-probability-weighted point estimate. They are shown for comparison and are not used for the verdict.
- **E-value.** How strong a hidden factor would have to be, as a risk ratio with both treatment and outcome, to explain the estimate away. The second number is for the interval limit nearest "no effect"; 1.00 means the interval already includes no effect.
- **Plain words.** The LLM's summary of the numbers above. Check it against them: it can be loose in its wording.

## Files and the audit log

Every run writes a folder `results/agent/run_<date>_<time>/`:

| File | Content |
|---|---|
| `report.md` | The full report: verdict, plan, review, gates, repairs, estimate, limits. |
| `result.json` | The same, as data: every gate with its checks, the plan, the repairs, all design versions, the confirmations. |
| `audit.jsonl` | One JSON line per event: the data sizes, every LLM prompt and answer, every gate, every repair, your answers, the moment the plan was frozen, the moment the outcomes were read. |

In the audit log, `plan_frozen` always comes before `unblind_trial_outcomes`.

## Settings

| Setting | Where | Meaning |
|---|---|---|
| Prediction model | runner / `--backend` | `tabpfn-client` (API; **the historical data leaves your machine**), `tabpfn-local`, or `logreg` (quick logistic baseline, labelled NOT TabPFN everywhere). |
| Keep only this many trial controls | runner / `--n-real` | Keeps a fixed random subset of the controls, to simulate a small control arm. 0 = all. |
| Most repairs allowed | runner / `--max-repairs` | Default 6. |
| Reuse a plan | `--plan-file result.json` | Skips the LLM draft and the review and uses the frozen plan from an earlier run. The LLM's draft is not repeatable run to run, so use this to reproduce a run exactly. |

## The web app

| Page | Purpose |
|---|---|
| `/` | Landing page: how a run works, why it can be checked, results on real data, the two ways to run. |
| `/run` | The runner: choose the mode, the data (two bundled examples or your own CSVs) and the settings. |
| `/jobs/<id>` | The live page while a job runs; the result page when it ends. |
| `/docs` | This guide. |

JSON endpoints used by the live page:

| Endpoint | Purpose |
|---|---|
| `GET /api/jobs/<id>?since=<n>` | Status (`running`, `waiting`, `done`, `failed`), the current step, new log lines, the gates of each design version, the open question. |
| `POST /api/jobs/<id>/answer` | Body `{"id": <question id>, "answer": "y"}`. Rejected with 409 if that question is no longer open. |
| `POST /api/jobs/<id>/cancel` | Stops the run at the next confirmation point. |
| `GET /jobs/<id>/files/<name>` | Downloads `report.md`, `result.json` or `audit.jsonl`. |

Jobs live in the memory of the server process. If you restart the server, the live page for an old job is gone, but its files stay in `results/agent/`. Up to three jobs run at once.

## Limits you should know

- **The plan is not clinically reviewed.** See the note at the top. In step by step mode, reading the plan carefully before you press Continue is the main safeguard you have.
- **A valid plan can still be a bad one.** The checks catch broken or leaking plans, not missing knowledge.
- **The LLM is not repeatable.** Two runs on the same data can produce different plans. `--plan-file` fixes this for a rerun.
- **Non-randomized evidence.** Even when the trial is randomized, borrowing from historical patients and adjusting for covariates is an observational step.
- **Only a few datasets have been tried.** The README lists what was run. Other diseases, other outcomes (not time-to-event) and very small tables are untested.
- **The repair loop is a controlled search over designs.** It is limited, logged and blind to the estimate, but it is still several attempts on one dataset. The report lists them all.
- **No login, no HTTPS.** The web app is for a trusted local machine.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `No module named 'langgraph'` (or flask) | The packages went into a different Python than the one running the app. Install into the project's environment: `uv pip install --python .venv/bin/python -r requirements.txt`, then start with `.venv/bin/python -m src.web`. |
| "The LLM is not configured" | Set the three `ANTHROPIC_FOUNDRY_*` variables in `.env` (not `.env.example`) and restart. |
| "TabPFN is not set up" | Set `TABPFN_TOKEN`, or choose the logistic baseline to check the flow. |
| Jev fails | The run continues; every column then goes to the LLM. |
| Columns are missing after matching | Their definition differs or a check failed. In step by step mode approve them by name; in single loop they are left out. |
| A run is slow | The LLM calls (planning and review) take about a minute; TabPFN fits add up to a few minutes. The live log shows where it is. |
| The page says a job is waiting for you | You are in step by step mode and the agent has asked a question. Scroll to the yellow card. |

## Code map

```
src/web/            the Flask app
    __init__.py       create_app()
    routes/           pages.py (landing, runner, docs), jobs.py (job page, API, downloads)
    jobs.py           job registry; turns the agent's log into steps and gates
    forms.py          reads and validates the runner form
    docs.py           renders this file
    templates/ static/
src/agent/          the agent
    spec.py           the plan and its validator
    frames.py         plan to analysis frames and design matrices (no study-specific names)
    gates.py          the gates and the estimate
    repair.py         allowed fixes and how one is chosen
    planner.py        the LLM prompts
    graph.py          the LangGraph graph
    report.py         verdict, claim, tier, report.md
    io.py             confirmations and the audit log
    background.py     runs the agent in a thread for the web app
src/cli/auto_run.py the command line
tests/              test_agent.py, test_web.py (no network, no keys)
```
