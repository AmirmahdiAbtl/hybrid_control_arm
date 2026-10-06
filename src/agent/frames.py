"""Spec -> analysis frames and design matrices. No column name is typed here: everything comes from the spec."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import skew

from src.agent.spec import keys


def new_state(spec, hybrid, n_real=None):
    """The design state the repair loop edits. The spec itself is never edited after it is frozen."""
    return {
        "horizon_years": spec.horizon_years,
        "restrict": [],  # (column, lo, hi) kept ranges
        "ps_level": 0,  # 0 main effects, 1 + squares, 2 + squares and products
        "drop_cov": [],
        "drop_cross": [],
        "disabled_rules": [],
        "hybrid": hybrid,
        "prognostic": "model",  # model | linear | none
        "n_real": n_real,
        "tried": [],
    }


def covariates(spec, state):
    """The covariates in use after the repair actions."""
    out = []
    for c in spec.covariates:
        if c.name in state["drop_cov"]:
            continue
        cross = c.cross_source and c.name not in state["drop_cross"] and state["hybrid"]
        out.append(c.model_copy(update={"cross_source": cross}))
    return out


def outcome_frame(spec, df):
    """Copy of ``df`` with ``__time`` (days), ``__event`` (0/1) and, if the treatment column exists, ``__A``."""
    f = df.copy()
    f["orig_index"] = df.index.to_numpy()
    f["__time"] = pd.to_numeric(f[spec.time], errors="coerce") * spec.unit_days()
    f["__event"] = keys(f[spec.event]).isin(spec.event_values).astype(int)
    if spec.treatment in f.columns:
        f["__A"] = (keys(f[spec.treatment]) != spec.control_value).astype(int)
    else:
        f["__A"] = 0
    return f


def build_study(spec, trial, hist, state):
    """Apply landmark, ranges and complete-case selection; build the encoders.

    Returns
    -------
    dict
        ``T`` and ``B`` (None without borrowing), ``covs``, ``encs``, ``enc_s`` (cross-source), ``enc_t_only``,
        ``funnel`` (list of (step, n)), ``hybrid``.
    """
    covs = covariates(spec, state)
    hybrid = bool(state["hybrid"] and hist is not None and any(c.cross_source for c in covs))
    funnel = []
    T = outcome_frame(spec, trial)
    funnel.append(("trial: all patients", len(T)))
    T = T[T["__time"] > spec.landmark_days]
    funnel.append((f"trial: event-free and followed at day {spec.landmark_days:g}", len(T)))
    T = _restrict(T, state["restrict"], [c.name for c in covs], funnel, "trial")
    T = T.dropna(subset=[c.name for c in covs]).reset_index(drop=True)
    funnel.append(("trial: complete covariates", len(T)))
    B = None
    if hybrid:
        s_names = [c.name for c in covs if c.cross_source]
        B = outcome_frame(spec, hist)
        funnel.append(("historical: eligible patients", len(B)))
        B = B[B["__time"] > spec.landmark_days]
        B = _restrict(B, state["restrict"], s_names, funnel, "historical")
        B = B.dropna(subset=s_names + ["__time"]).reset_index(drop=True)
        funnel.append(("historical: complete covariates and outcome", len(B)))
    encs = [_encoder(c, T, B) for c in covs]
    return {
        "T": T,
        "B": B,
        "covs": covs,
        "encs": encs,
        "enc_s": [e for e in encs if e["cross_source"]],
        "enc_t_only": [e for e in encs if not e["cross_source"]],
        "funnel": funnel,
        "hybrid": hybrid,
        "ps_level": state["ps_level"],
        "missing_trial": {c.name: round(float(trial[c.name].isna().mean()), 4) for c in covs},
        "missing_hist": {
            c.name: round(float(hist[c.name].isna().mean()), 4) for c in covs if hybrid and c.cross_source
        },
    }


def _restrict(frame, restrictions, allowed, funnel, label):
    for col, lo, hi in restrictions:
        if col in frame.columns and col in allowed:
            frame = frame[(frame[col] >= lo) & (frame[col] <= hi)]
            funnel.append((f"{label}: {col} in [{lo:.4g}, {hi:.4g}]", len(frame)))
    return frame


def _encoder(cov, T, B):
    pooled = T[cov.name] if B is None or not cov.cross_source else pd.concat([T[cov.name], B[cov.name]])
    enc = {"name": cov.name, "kind": cov.kind, "cross_source": cov.cross_source, "levels": None, "log": False}
    if cov.kind == "numeric":
        v = pd.to_numeric(pooled, errors="coerce").dropna()
        enc["log"] = bool(len(v) >= 20 and v.min() >= 0 and skew(v) > 2)
    elif cov.kind in ("binary", "categorical"):
        enc["levels"] = sorted(set(keys(pooled.dropna())))
    elif not pd.api.types.is_numeric_dtype(pooled):  # ordinal given as text: order by sorted value
        enc["levels"] = sorted(set(keys(pooled.dropna())))
    return enc


def design(encs, frame, squares=False, products=False):
    """Design matrix for the weighting models (no intercept).

    numeric: value (log1p if very skewed); binary: 0/1; categorical: one column per level except the first;
    ordinal: the number (or the rank of a text value). ``squares`` / ``products`` add the squares and pairwise
    products of the continuous columns (at most 6 of them for the products).

    Returns
    -------
    D : numpy.ndarray
    names : list of str
    n_main : int
        Number of main-effect columns (the first ``n_main`` columns of ``D``).
    """
    cols, names, cont = [], [], []
    for e in encs:
        x, n = frame[e["name"]], e["name"]
        if e["kind"] == "numeric":
            v = pd.to_numeric(x).to_numpy(float)
            cont.append(len(cols))
            cols.append(np.log1p(v) if e["log"] else v)
            names.append(f"log1p({n})" if e["log"] else n)
        elif e["kind"] == "binary":
            cols.append((keys(x) == e["levels"][-1]).to_numpy(float))
            names.append(f"{n}={e['levels'][-1]}")
        elif e["kind"] == "categorical":
            for lv in e["levels"][1:]:
                cols.append((keys(x) == lv).to_numpy(float))
                names.append(f"{n}={lv}")
        else:
            v = _ordinal(e, x)
            cont.append(len(cols))
            cols.append(v)
            names.append(n)
    D = np.column_stack(cols) if cols else np.zeros((len(frame), 0))
    n_main = D.shape[1]
    extra, extra_names = [], []
    if squares:
        for j in cont:
            extra.append(D[:, j] ** 2)
            extra_names.append(f"{names[j]}^2")
    if products:
        for i, j in itertools.combinations(cont[:6], 2):
            extra.append(D[:, i] * D[:, j])
            extra_names.append(f"{names[i]} x {names[j]}")
    if extra:
        D = np.column_stack([D, *extra])
    return D, names + extra_names, n_main


def _ordinal(e, x):
    if e["levels"] is None:
        return pd.to_numeric(x).to_numpy(float)
    rank = {lv: i for i, lv in enumerate(e["levels"])}
    return keys(x).map(rank).to_numpy(float)


def features(encs, frame):
    """Raw feature matrix for the prediction model (categories as integer codes) and its categorical columns."""
    cols, cat_idx = [], []
    for e in encs:
        x = frame[e["name"]]
        if e["kind"] in ("numeric",):
            cols.append(pd.to_numeric(x).to_numpy(float))
        elif e["kind"] == "ordinal":
            cols.append(_ordinal(e, x))
        else:
            rank = {lv: i for i, lv in enumerate(e["levels"])}
            cat_idx.append(len(cols))
            cols.append(keys(x).map(rank).to_numpy(float))
    return (np.column_stack(cols) if cols else np.zeros((len(frame), 0))), cat_idx


def ps_design(study, frame):
    """Design of the propensity model at the current richness level (see ``new_state``)."""
    return design(study["encs"], frame, squares=study["ps_level"] >= 1, products=study["ps_level"] >= 2)[0]


def strata_labels(frame, encs):
    """Fold-stratification labels: the first binary or categorical covariate, else one group."""
    for e in encs:
        if e["kind"] in ("binary", "categorical"):
            return keys(frame[e["name"]]).to_numpy()
    return np.zeros(len(frame), int)
