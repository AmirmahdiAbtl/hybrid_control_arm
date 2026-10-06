import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.demo import APPROVED, DEMO, PATH_A, PATH_B, TRUTH, prepare
from src.exploratory import (
    comparability,
    historical_predictions,
    prepare_arms,
    run_estimate,
    run_validation,
    validation_headline,
)
from src.harmonize import coverage, evaluate_mapping, load_mapping
from src.models import (
    BACKEND_LABELS,
    BACKENDS,
    TabPFNSetupError,
    cross_source_features,
    default_backend,
    default_features,
)
from src.survival import DAYS_PER_YEAR

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"Backend: {BACKEND_LABELS[args.backend]}")
    if args.backend == "standin":
        print("!! STAND-IN MODEL: testing only. These numbers are NOT TabPFN results.")

    df_a, df_b = pd.read_csv(args.a), pd.read_csv(args.b)
    mapping = load_mapping(args.mapping)
    rules = [c for c in json.loads(Path(args.criteria).read_text())["criteria"] if c["kind"] == "rule"]
    d = harmonize_and_select(df_a, df_b, mapping, rules, args, out)

    feats = cross_source_features(default_features(d["a_h"], d["b_h"], {args.treatment, *args.outcome_cols}))
    print("Features:", feats, "(grade is excluded: the two studies used different grading protocols)")
    outcome_names = {args.outcome_cols[0]: "time", args.outcome_cols[1]: "cens"}
    hist, trial = d["hist"].rename(columns=outcome_names), d["trial"].rename(columns=outcome_names)
    horizon = args.years * DAYS_PER_YEAR
    try:
        report_comparability(hist, trial, feats, args, out)
        arms = prepare_arms(hist, trial, feats, horizon)
        preds = historical_predictions(arms, [args.backend], args.seed)
        report_historical_model(trial, preds, args.backend)
        report_estimate(hist, trial, feats, horizon, arms, preds[args.backend], args, out)
        report_validation(hist, trial, feats, horizon, arms, preds[args.backend], args, out)
    except TabPFNSetupError as e:
        print(f"\nTabPFN is not set up:\n{e}", file=sys.stderr)
        sys.exit(2)
    print(f"\nDone in {time.time() - t0:.0f} s. CSV files are in {out}/")
    print("For the gated causal analysis run: python -m src.cli.run_gated")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=[*BACKENDS, "logreg"], default=default_backend())
    ap.add_argument("--reps", type=int, default=200, help="validation splits (no model fit per split)")
    ap.add_argument("--n-real", type=int, default=60)
    ap.add_argument("--years", type=float, default=2, help="horizon in years (1, 2, 3, 5 ...)")
    ap.add_argument(
        "--joint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run the legacy joint model (one more model fit, plus one per validation split)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a", default=str(PATH_A), help="new trial CSV")
    ap.add_argument("--b", default=str(PATH_B), help="old study CSV")
    ap.add_argument("--mapping", default=str(APPROVED), help="approved mapping JSON (from the app)")
    ap.add_argument("--criteria", default=str(DEMO / "llm_eligibility.json"), help="criteria JSON (LLM output)")
    ap.add_argument("--truth", default=str(TRUTH), help="true mapping JSON (only to score the harmonizer; '' to skip)")
    ap.add_argument("--treatment", default="horTh")
    ap.add_argument("--control-value", default="no")
    ap.add_argument("--outcome-cols", nargs=2, default=["time", "cens"])
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def harmonize_and_select(df_a, df_b, mapping, rules, args, out):
    """Steps 0-1: harmonize with the approved mapping, then apply the eligibility rules (output of ``prepare``)."""
    print("\n=== 0 Harmonize ===")
    print(coverage(mapping, list(df_a.columns)))
    if args.truth and Path(args.truth).exists():
        ev = evaluate_mapping(mapping, args.truth)
        print(
            f"Harmonizer vs truth: precision {ev['precision']:.2f}, recall {ev['recall']:.2f}, "
            f"per route {ev['per_route']}"
        )
        ev["rows"].astype({"proposed": str, "truth": str}).to_csv(out / "harmonizer_evaluation.csv", index=False)
    d = prepare(df_a, df_b, mapping, rules, args.treatment, args.control_value)
    for a, err in d["info"]["errors"].items():
        print(f"  ERROR {a}: approved transform does not run, column left out ({err})")
    print("Unmapped A columns:", d["info"]["unmapped_a"] or "none", "| dropped B columns:", d["info"]["dropped_b"])
    for a, c in d["info"]["post_checks"].items():
        print(f"  {a}: {'ok' if c['ok'] else 'CHECK'} - {c['reason']}")

    show("1 Eligibility funnel", d["funnel"], out, "funnel")
    if d["invalid"]:
        print("Invalid rules (not applied):", d["invalid"])
    print(f"Eligible historical patients: {len(d['hist'])} of {len(df_b)}")
    return d


def report_comparability(hist, trial, feats, args, out):
    c = comparability(hist, trial, feats, args.backend, args.seed)
    print(
        f"\n=== 2 Comparability: do the historical patients cover the treated? ===\n"
        f"AUC {c['auc']:.3f} (treated vs historical) -> {c['verdict']}; {c['share_unsupported']:.1%} of the "
        f"treated have no support (density ratio > 20)"
    )
    show(
        "Support (each row needs: share unsupported <= 0.10, ESS of historical >= 50, max share <= 0.10)",
        c["support"],
        out,
        "comparability_support",
    )
    show("Features (historical vs treated)", c["features"], out, "comparability_features")


def report_historical_model(trial, preds, backend):
    if f"{backend}_pseudo" not in preds:  # the pseudo-observation regressor is fitted for the cheap backends only
        return
    trt = (trial["arm"] == "treated").to_numpy()
    print(
        f"\nHistorical model, mean over the treated: {preds[backend][trt].mean():.1%} (classifier on "
        f"complete-case labels, used below); {preds[backend + '_pseudo'][trt].mean():.1%} (regressor on "
        "pseudo-observations, comparison only)"
    )


def report_estimate(hist, trial, feats, horizon, arms, p_trial, args, out):
    est = run_estimate(
        hist,
        trial,
        feats,
        horizon,
        args.n_real,
        args.backend,
        args.seed,
        do_joint=args.joint,
        arms=arms,
        p_trial=p_trial,
    )
    show(
        f"3 Estimate (exploratory, not gated): control outcome of the treated, % (event-free at {args.years:g} years)",
        in_percent(est["table"], ("estimate", "lo", "hi")),
        out,
        "estimate",
    )
    show(
        "Difference vs treated, points (exploratory, not gated; only the robust hybrid has a valid CI)",
        in_percent(est["effects"], ("effect", "lo", "hi")),
        out,
        "estimate_effects",
    )
    print(est["shift_text"] + ("  WARNING: shift is larger than twice the noise." if est["shift_warn"] else ""))
    print("Pseudo-observations by arm; legacy rows use complete-case labels.")


def report_validation(hist, trial, feats, horizon, arms, p_trial, args, out):
    v = run_validation(
        hist,
        trial,
        feats,
        horizon,
        args.n_real,
        args.backend,
        args.reps,
        args.seed,
        do_joint=args.joint,
        arms=arms,
        p_trial=p_trial,
        progress=lambda f: print(f"\r  validation {f:4.0%}", end="", file=sys.stderr),
    )
    print(file=sys.stderr)
    show(
        f"4 Validation ({args.reps} splits; points; truth = ATT counterfactual of the hidden controls, demo only; "
        "report only)",
        v["summary"],
        out,
        "validation_summary",
    )
    v["runs"].to_csv(out / "validation_runs.csv", index=False)
    print("\n" + validation_headline(v["summary"], v["n_real"]))


def show(title, df, out, name):
    """Print a table (long notes under it, by method) and save it as ``<out>/<name>.csv``."""
    print(f"\n=== {title} ===")
    if "note" in df.columns:
        print(df.drop(columns="note").to_string(index=False))
        for method, note in zip(df["method"], df["note"]):
            print(f"  {method}: {note}")
    else:
        print(df.to_string(index=False))
    df.to_csv(out / f"{name}.csv", index=False)


def in_percent(df, cols):
    df = df.copy()
    for k in cols:
        df[k] = df[k] * 100
    return df


if __name__ == "__main__":
    main()
