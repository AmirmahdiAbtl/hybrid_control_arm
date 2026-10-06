# Hybrid Control Arm

A clinical trial often has a small control group. An older study may have patients who look like those controls. This
tool checks whether the old patients can be used to sharpen the comparison, and says so clearly when they can't.

## Where TabPFN is used

**TabPFN** is the prediction model in this pipeline. It is used for one thing, a **prognostic score**: a prediction of
how each patient is likely to do, learned from the old study's patients. That one score is used in three places.

1. **Training.** Each old patient's outcome at the chosen time horizon is turned into a number (a Kaplan-Meier
   pseudo-observation). **TabPFN** (a regressor) is trained on the old patients' covariates, the ones both studies
   recorded, to predict that number. It needs no tuning, which is why it fits a table of a few hundred patients.
2. **Checking it, gate G6.** Before the score is used, **TabPFN** is cross-validated on the old patients (5 folds). Its
   error must beat a constant guess and be no worse than plain linear regression plus 2 standard errors. If it fails,
   the agent falls back to a linear score, then to none, and logs that.
3. **The estimate.** One final **TabPFN** fit predicts a risk score for every trial patient. The trial's controls are
   weighted to match the treated patients, and a small model that includes this **TabPFN** score corrects the
   outcome. The result is the effect in the treated patients: the difference in the probability of being event-free,
   treated minus estimated control.

That is **5 cross-validation fits plus 1 final fit: 6 TabPFN fits per design version.** Predictions are cached, so
rerunning an unchanged design doesn't refit; a repair that changes the horizon or the variables does.

Everything else is plain logistic or linear code, not **TabPFN**: the overlap and balance checks, the weights, the final
correction and the E-value. The old patients are never pooled in as extra controls, and the LLM never computes a number.

## The agent

Give it two CSV files: the new trial and an older study. An AI agent then

1. matches the old study's columns to the trial's (Jev + LLM),
2. keeps only the old patients who would have qualified for the trial,
3. writes an analysis plan (LLM) that code checks against the data,
4. runs a series of checks, using **TabPFN** for the prediction,
5. fixes what fails and runs again,
6. writes a report with a verdict, the estimate and a full audit log.

You can run it in one go, or step by step and confirm each decision yourself. **TabPFN** is the prediction model at the
core of it.

## Run it

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env            # fill in your keys, see below
.venv/bin/python flask_app.py   # opens http://127.0.0.1:5050
```

In the app, pick the bundled example (GBSG-2 + Rotterdam breast cancer data) or upload your own two CSVs, choose
**single loop** or **step by step**, and press start.

Keys in `.env`:

| Variable | Used for |
|---|---|
| `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_MODEL` | the LLM (required) |
| `TABPFN_TOKEN` | TabPFN through its API |
| `TYPESAFE_API_KEY` | Jev (optional; without it the LLM matches every column) |

The **TabPFN** API sends the historical patient data to Prior Labs. To try the flow without **TabPFN**, choose the
"Logistic baseline" model in the app; its results are labelled NOT TabPFN.

## Two ways to run it

Same agent, same checks, same estimate. The only difference is who answers the agent's questions.

**Single loop.** Press start and wait. The agent goes from your two CSVs to the verdict without stopping. Every
question it would have asked you is answered "yes" automatically, and the report says "auto-confirmed", because no
human looked at the mapping, the rules or the plan. Good for a quick try.

**Step by step.** The agent pauses and waits for you at four decisions. You can continue, stop, or type a change:

1. **Column mapping**: which old column was matched to which trial column. Doubtful matches are flagged and not used
   unless you approve them.
2. **Eligibility rules**: the rules the LLM wrote, and how many old patients each one removes.
3. **The plan**: treatment, event definition, time horizon, adjustment variables.
4. **Each repair**: the fix proposed after a check fails. Accept it or skip it.

Use this one when the result matters. The plan is written by an LLM, and you are the only human reviewer.

## How the pipeline works

1. **Read.** Both tables are profiled. If you gave a ClinicalTrials.gov id, the trial's title, arms, outcome and
   criteria are fetched.
2. **Match columns.** The old study is renamed and recoded into the trial's format. Jev scores every column pair. A pair
   is accepted automatically only if the score is at least 0.90, the type, unit and category checks pass, and no other
   column also scores that high. Everything else goes to the LLM with its five best candidates. Columns that mean
   something different (say, two grading systems) are flagged.
3. **Eligibility.** The LLM turns the trial's entry criteria into rules and a restricted rule engine applies them to the
   old patients. On the bundled data this keeps 552 of 2,982.
4. **Plan.** The LLM drafts it. Code checks it against the data: columns exist, treatment is binary, the event is coded
   sensibly, missingness is low, no variable leaks the treatment, the horizon is inside the follow-up. Mistakes go back
   to the LLM (up to three drafts). A second LLM call reviews it once. Then it's frozen and hashed.
5. **Gates, in order.** The first four use covariates only and never touch the trial's outcomes.
   - G1: enough patients, events and complete data.
   - G2: treated and control patients overlap.
   - G3: after weighting, the two arms look alike.
   - G4: the old patients resemble the treated ones.
   - G5: enough patients are still followed at the horizon in every group.
   - G6: the **TabPFN** model is good enough to use.
   - Now the trial's outcomes are read, once, for the frozen design.
   - G7: the weighting behind the estimate actually worked.
6. **Repair.** When a gate fails, code lists the fixes allowed for it (shorten the horizon, restrict a range, drop a
   covariate, and as a last resort stop using the old data). The LLM picks one. It only sees the failing numbers, never
   the estimate. Then the gates run again, up to 6 times.
7. **Report.** Verdict, estimate with a 95% interval, E-value, every repair, every design version that reached an
   estimate, and a full audit log.

## Command line

```bash
.venv/bin/python -m src.cli.auto_run \
  --trial data/data_A_new_trial_GBSG2.csv --hist data/data_B_old_study_Rotterdam.csv \
  --trial-dict demo/dictionary_A.json --hist-dict demo/dictionary_B.json \
  --eligibility-file demo/eligibility_gbsg2.txt --yes
```

`--yes` confirms every step automatically; leave it out to be asked. Add `--backend logreg` for a quick run without
TabPFN. Each run writes `report.md`, `result.json` and `audit.jsonl` to `results/agent/run_<time>/`.

## More

- [docs/AGENT.md](docs/AGENT.md): the full guide (every step, every check, how to read a result). The app also serves
  it at `/docs`.
- Tests: `.venv/bin/python -m pytest -q` (no network or keys needed).
- Demo data: GBSG-2 breast cancer trial (Schumacher et al., J Clin Oncol 1994) and the Rotterdam tumour bank (R package
  `survival`; Royston and Altman, BMC Med Res Methodol 2013). The InSilicoTrials logo in the web app is theirs.
- Code: `src/web/` is the web app, `src/agent/` is the agent, `src/cli/` the command line.

The plan is written by an LLM and checked by code, not by a clinician. Treat the result as supporting evidence only.
