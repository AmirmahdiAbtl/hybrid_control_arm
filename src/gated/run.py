import traceback

from src import models
from src.gated.design import stage_a, stage_b, stage_c
from src.gated.outcomes import stage_d
from src.gated.protocol import ESTIMAND_ROLES, HORIZON_YEARS
from src.gated.report import finish, new_result
from src.gated.runlog import StopRun, audit, start_run


def run_gated(
    df_a,
    df_b,
    mapping,
    rules,
    mode="hybrid",
    backend=None,
    supplementary_5y=False,
    out_dir="results/gated",
    cache_dir="results/cache",
    progress=None,
    validate=True,
):
    """Run the pre-specified gated causal analysis end to end.

    Parameters
    ----------
    df_a, df_b : pandas.DataFrame
        Raw trial (GBSG-2) and historical (Rotterdam) data.
    mapping, rules : dict, list of dict
        Approved Stage-1 mapping and Stage-2 rules (no LLM is called).
    mode : {"hybrid", "internal"}
        "hybrid": primary scope H (kept controls + Rotterdam prognostic score); "internal": primary scope R (all
        trial controls).
    backend : str, optional
        Backend of the outcome and prognostic models; default ``models.default_backend()``.
    supplementary_5y : bool
        Add the 5-year supplementary horizon.
    out_dir : str
        Results root; the run writes to ``out_dir/<protocol_digest[:10]>/run_<NNN>_<mode>/``.
    cache_dir : str or None
        TabPFN prediction cache (None = no cache).
    progress : callable, optional
        ``progress(fraction, message)``; receives every gate line, action and stage message.
    validate : bool
        Run the G12 validation and placebo test (report only).

    Returns
    -------
    dict
        verdict, tier, claim, secondary_claim, first_failing_gate, reason, mode, backend, labels, primary,
        estimates, gates, sensitivity, cate, support, balance, validation, placebo, dag, funnel, horizon,
        benchmarks, diagnostics, actions, tabpfn_fits, cache_hits, seconds, run_dir, protocol_digest, run_index,
        earlier_runs, text.

    Raises
    ------
    src.models.TabPFNSetupError
        TabPFN is not usable (licence, login, token); the partial outputs are written first.
    """
    if mode not in ESTIMAND_ROLES:
        raise ValueError(f"mode must be 'hybrid' or 'internal', not '{mode}'")
    backend = backend or models.default_backend()
    horizons = [HORIZON_YEARS["primary"], HORIZON_YEARS["secondary"]] + (
        [HORIZON_YEARS["supplementary"]] if supplementary_5y else []
    )
    ctx = start_run(out_dir, mode, backend, df_a, df_b, mapping, rules)
    ctx["progress"] = progress
    res = new_result(ctx)
    try:
        stage_a(ctx, res)  # G0-G2, no data
        s = stage_b(ctx, res, df_a, df_b, mapping, rules, validate)  # G3-G6, outcome-blind
        stage_c(ctx, res, s, horizons, cache_dir)  # G7-G8, Rotterdam outcomes only
        stage_d(ctx, res, s, horizons, cache_dir, validate)  # G9-G12, trial outcomes unblinded
    except StopRun as stop:
        return finish(ctx, res, stop.gate, stop.reason)
    except models.TabPFNSetupError as ex:
        audit(ctx, "error", kind=type(ex).__name__, message=str(ex)[:2000])
        finish(ctx, res, None, f"incomplete: TabPFN is not set up ({type(ex).__name__})")
        raise
    except Exception as ex:  # noqa: BLE001 - any other failure ends the run NOT ACCEPTED (incomplete)
        audit(ctx, "error", kind=type(ex).__name__, message=str(ex)[:2000], traceback=traceback.format_exc()[-4000:])
        return finish(ctx, res, None, f"incomplete: {type(ex).__name__}: {ex}")
    return finish(ctx, res, None, "")
