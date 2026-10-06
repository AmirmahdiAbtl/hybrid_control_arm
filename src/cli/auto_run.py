import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from src.agent import run_agent
from src.agent.io import AutoAsker, ConsoleAsker
from src.agent.spec import StudySpec
from src.models import BACKEND_LABELS, TabPFNSetupError, default_backend

HELP = """Fully automated hybrid-control analysis: the LLM plans, code checks, you confirm at a few points.

    python -m src.cli.auto_run --trial new_trial.csv --hist old_study.csv --nct NCT00000000
    python -m src.cli.auto_run --trial new_trial.csv --hist old_study.csv --eligibility-file criteria.txt --yes

Steps: read data -> match columns (Jev + LLM) -> eligibility rules (from the registry text) -> LLM drafts the plan ->
code validates it -> a second LLM call reviews it -> you confirm -> gates run in order. When a gate fails, the agent
changes one thing (from a short list, chosen from the gate's numbers only) and runs the gates again, up to
--max-repairs times. Output: <out>/run_<time>/ (report.md, result.json, audit.jsonl).
Exit code: 0 = ACCEPTED, 1 = NOT ACCEPTED, 2 = TabPFN is not set up.
"""


def read_dictionary(path):
    """Optional ``{column: description}`` JSON file."""
    return json.loads(Path(path).read_text()) if path else None


def load_plan(path):
    """A StudySpec from a result.json (its ``spec``) or from a file holding the spec itself."""
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return StudySpec(**(data.get("spec") or data))


def main(argv=None):
    p = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial", required=True, help="CSV of the new trial (treatment, time and event columns)")
    p.add_argument("--hist", required=True, help="CSV of the historical study")
    p.add_argument("--nct", help="ClinicalTrials.gov id of the new trial")
    p.add_argument("--eligibility-file", help="text file with the eligibility criteria (instead of --nct's)")
    p.add_argument("--trial-dict", help="JSON {column: description} for the trial table")
    p.add_argument("--hist-dict", help="JSON {column: description} for the historical table")
    p.add_argument("--backend", default=default_backend(), choices=list(BACKEND_LABELS), help="prediction model")
    p.add_argument("--yes", action="store_true", help="confirm every step automatically (logged as automatic)")
    p.add_argument("--n-real", type=int, help="keep only this many trial controls (simulates a small control arm)")
    p.add_argument("--max-repairs", type=int, default=6)
    p.add_argument("--plan-file", help="reuse a plan: the 'spec' of an earlier result.json (skips the LLM draft)")
    p.add_argument("--out", default="results/agent")
    p.add_argument("--cache", default="results/cache")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    text = Path(args.eligibility_file).read_text() if args.eligibility_file else None
    try:
        result = run_agent(
            pd.read_csv(args.trial),
            pd.read_csv(args.hist),
            nct=args.nct,
            eligibility_text=text,
            trial_labels=read_dictionary(args.trial_dict),
            hist_labels=read_dictionary(args.hist_dict),
            asker=AutoAsker() if args.yes else ConsoleAsker(),
            backend=args.backend,
            out_dir=args.out,
            cache_dir=args.cache,
            n_real=args.n_real,
            max_repairs=args.max_repairs,
            plan=load_plan(args.plan_file),
        )
    except TabPFNSetupError as e:
        print(f"TabPFN is not set up: {e}", file=sys.stderr)
        return 2
    print("\n" + result["text"])
    print(f"\nFiles: {result['run_dir']}")
    return 0 if result["verdict"] == "ACCEPTED" else 1


if __name__ == "__main__":
    sys.exit(main())
