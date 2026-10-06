import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src import gated
from src.demo import APPROVED, DEMO, PATH_A, PATH_B
from src.gated import protocol, runlog
from src.harmonize import load_mapping
from src.models import BACKEND_LABELS, BACKENDS, TabPFNSetupError, default_backend

HELP = """Command line run of the pre-specified gated causal analysis: one command, exit code = verdict.

    python -m src.cli.run_gated --backend logreg                    # quick offline check (logistic/linear, NOT TabPFN)
    python -m src.cli.run_gated --mode internal --backend logreg
    python -m src.cli.run_gated                                     # default backend: HCA_BACKEND, else TabPFN API

The horizon, n_real, seed, thresholds and covariate sets are protocol constants in src/gated/protocol.py; there are no
flags for them. Each run writes to <out>/<protocol digest>/run_<NNN>_<mode>/ (audit.jsonl, verdict.json, gates.csv,
estimates.csv, report.md, ...). Exit code: 0 = ACCEPTED, 1 = NOT ACCEPTED (including budget overruns and incomplete
runs), 2 = TabPFN is not set up.
"""

ROLE_ORDER = {"primary": 0, "secondary": 1, "reference": 2, "descriptive": 3, "exploratory": 4, "supplementary": 5}


def estimates_table(est):
    """Estimates table in percentage points, ordered by role (primary first)."""
    tab = est.assign(order=est["role"].map(ROLE_ORDER).fillna(9)).sort_values(
        ["order", "horizon_years", "scope"], kind="stable"
    )
    out = pd.DataFrame(
        {
            "estimand": tab["estimand"],
            "scope": tab["scope"],
            "role": tab["role"],
            "years": tab["horizon_years"].map(lambda v: f"{v:g}"),
            "method": tab["method"].str[:58],
        }
    )
    for c in ("est", "lo", "hi", "se"):
        out[c] = (tab[c].astype(float) * 100).round(1)
    out["n_target"], out["n_comparison"] = tab["n_target"], tab["n_comparison"]
    return out.to_string(index=False, na_rep="")


def main(argv=None):
    """Parse the arguments, run :func:`src.gated.run_gated` and print the gates, estimates and verdict.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments (default ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 ACCEPTED, 1 NOT ACCEPTED, 2 TabPFN setup error.
    """
    args = _parse_args(argv)
    _print_backend(args.backend)
    df_a, df_b = pd.read_csv(args.a), pd.read_csv(args.b)
    mapping = load_mapping(args.mapping)
    rules = [c for c in json.loads(Path(args.criteria).read_text())["criteria"] if c["kind"] == "rule"]
    _print_protocol(args, df_a)
    cache = None if args.no_cache else str(Path(args.out).parent / "cache")
    try:
        res = gated.run_gated(
            df_a,
            df_b,
            mapping,
            rules,
            mode=args.mode,
            backend=args.backend,
            supplementary_5y=args.supplementary_5y,
            out_dir=args.out,
            cache_dir=cache,
            progress=lambda f, m="": print(m, flush=True) if m else None,
        )
    except TabPFNSetupError as e:
        print(f"\nTabPFN is not set up:\n{e}", file=sys.stderr)
        return 2
    _print_result(res)
    return 0 if res["verdict"] == "ACCEPTED" else 1


def _parse_args(argv):
    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--mode",
        choices=["hybrid", "internal"],
        default="hybrid",
        help="hybrid: 60 kept controls + Rotterdam prognostic score (primary H); internal: all trial "
        "controls (primary R)",
    )
    ap.add_argument("--backend", choices=BACKENDS + ["logreg"], default=default_backend())
    ap.add_argument("--supplementary-5y", action="store_true", help="add the 5-year supplementary horizon")
    ap.add_argument("--a", default=str(PATH_A), help="trial CSV (GBSG-2)")
    ap.add_argument("--b", default=str(PATH_B), help="historical CSV (Rotterdam)")
    ap.add_argument("--mapping", default=str(APPROVED), help="approved mapping JSON (from the app)")
    ap.add_argument(
        "--criteria",
        default=str(DEMO / "llm_eligibility.json"),
        help="criteria JSON; the Stage-2 rules are the entries with kind 'rule'",
    )
    ap.add_argument("--out", default="results/gated", help="results folder (patient-derived; git-ignored)")
    ap.add_argument("--no-cache", action="store_true", help="do not reuse cached TabPFN predictions")
    return ap.parse_args(argv)


def _print_backend(backend):
    print(f"Backend: {BACKEND_LABELS[backend]}")
    if backend == "standin":
        print("!! STAND-IN MODEL: testing only. These numbers are NOT TabPFN results.")
    if backend == "tabpfn-client":
        print("The client backend sends the patient data to the Prior Labs servers (data leaves this machine).")


def _print_protocol(args, df_a):
    """Protocol digest, earlier runs of this digest and the expected number of TabPFN fits."""
    digest = runlog.protocol_digest()
    history = runlog.run_history(args.out, digest)
    print(f"Protocol {protocol.PROTOCOL_VERSION}, digest {digest[:10]}; this is run {len(history) + 1} of this digest")
    for r in history:
        print(f"   earlier run {r['run']:03d} ({r.get('mode')}, {r.get('backend')}): {r['verdict']} ({r.get('tier')})")
    controls = int((df_a.get(protocol.TREATMENT, pd.Series(dtype=str)).astype(str) == protocol.CONTROL_VALUE).sum())
    demo = args.mode == "hybrid" and controls > protocol.N_REAL
    fits = (
        runlog.expected_tabpfn_fits(args.mode, args.supplementary_5y, demo) if args.backend.startswith("tabpfn") else 0
    )
    print(f"Expected TabPFN fits: {fits} (hard cap {protocol.MAX_TABPFN_FITS})\n")


def _print_result(res):
    """Estimates (only for tiers that may show numbers), verdict text, fit count and output folder."""
    est = res["estimates"]
    if res["tier"] in ("PRIMARY", "SUPPORTIVE") and len(est):
        print(
            "\n=== Estimates: event-free difference, percentage points (primary, secondary, reference, exploratory) ==="
        )
        print(estimates_table(est))
    elif res["tier"] == "EXPLORATORY" and len(est):
        print(f"\n=== {protocol.EXPLORATORY_TITLE} ===")
        print(estimates_table(est))
    else:
        print("\nNo treatment-effect estimate is reported (tier NOT INTERPRETABLE).")
    lines = res["text"].splitlines()
    print("\nVERDICT: " + lines[0])
    print("\n".join(lines[1:]))
    print(f"\nTabPFN fits: {res['tabpfn_fits']} (cache hits {res['cache_hits']}); {res['seconds']} s")
    print(f"Outputs: {res['run_dir']}")


if __name__ == "__main__":
    sys.exit(main())
