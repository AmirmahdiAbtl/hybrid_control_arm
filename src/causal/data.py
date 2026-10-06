import numpy as np
import pandas as pd

# DAG node -> column fed to the models; gate G2 checks the keys against the adjustment sets Z_T / Z_S.
TRIAL_COLUMNS = {
    "age": "age",
    "menostat": "menostat",
    "tsize": "tsize_mm",
    "tgrade": "tgrade",
    "pnodes": "pnodes",
    "progrec": "progrec",
    "estrec": "estrec",
}
SOURCE_COLUMNS = {
    "age": "age",
    "menostat": "menostat",
    "tsize": "tsize_grp",
    "pnodes": "pnodes",
    "progrec": "pgr_cat",
    "estrec": "er_cat",
}

# Fixed category codes, so that every frame and every fold is coded the same way.
MENO_CODES = {"Pre": 0, "Post": 1}
GRADE_CODES = {"I": 0, "II": 1, "III": 2}
SIZE_CODES = {"<=20": 0, "20-50": 1, ">50": 2}

TRIAL_CAT_IDX = [1, 3]  # categorical columns of features_trial (menostat, tgrade)
SOURCE_CAT_IDX = [1, 2]  # categorical columns of features_source (menostat, tsize group)

DESIGN_TRIAL_NAMES = [
    "age",
    "post",
    "tsize_mm",
    "grade_II",
    "grade_III",
    "log1p_pnodes",
    "log1p_progrec",
    "log1p_estrec",
]
DESIGN_TRIAL_EXT_NAMES = DESIGN_TRIAL_NAMES + ["age^2", "age_x_post", "log1p_pnodes^2"]
DESIGN_SOURCE_NAMES = [
    "age",
    "post",
    "tsize_20-50",
    "tsize_>50",
    "log1p_pnodes",
    "er_10-99",
    "er_>=100",
    "pgr_10-99",
    "pgr_>=100",
]


def gate_frames(
    d,
    df_a,
    df_b,
    age_max=65,
    landmark=91.0,
    b_years=None,
    treatment="horTh",
    control_value="no",
    receptor_cuts=(10.0, 100.0),
):
    """Eligible trial and historical frames, built from the approved Stage-2 data.



    Parameters
    ----------
    d : dict
        Output of ``src.demo.prepare`` (uses ``a_h`` and ``hist``).
    df_a, df_b : pandas.DataFrame
        Raw trial and historical data; ``df_b`` should have ``year``.
    age_max : float or None
        Keep ``age <= age_max``; None = no age filter.
    landmark : float or None
        Keep ``time > landmark`` (days: event-free and under follow-up at time zero); None = no landmark.
    b_years : tuple of int, optional
        Keep historical patients with surgery year in ``[lo, hi]``.
    treatment, control_value : str
        Treatment column of A and its control label. Every other value, including a missing one, counts as
        treated, so gate G3 must first check that the column is complete and has only the two expected labels.
    receptor_cuts : tuple of float
        Receptor category cuts (fmol/mg): <cut0 -> 0, cut0-cut1 -> 1, >=cut1 -> 2.

    The defaults repeat protocol constants of ``src.gated.protocol``, which cannot be imported here (it imports
    this module); tests/test_causal_dag.py checks that the two copies agree.

    Returns
    -------
    dict
        ``trial_cov`` and ``hist_cov`` (covariates without ``time`` / ``cens``, sorted by ``orig_index``, index
        0..n-1), ``outcomes`` (``{"trial": DataFrame[time, cens], "hist": ...}``, row-aligned; read only after
        the gate it belongs to), ``funnel`` (source, step, n_after, n_removed, n_treated, n_control),
        ``landmark_ok`` (min time > landmark in both frames) and ``params``.
    """
    a_h, h = d["a_h"], d["hist"]
    trial = pd.DataFrame(
        {
            "orig_index": df_a.index.to_numpy(),
            "age": _numeric_column(df_a, "age"),
            "menostat": _label_column(df_a, "menostat"),
            "tsize_mm": _numeric_column(df_a, "tsize"),
            "tsize_grp": _column_or_nan(a_h, "tsize").reindex(df_a.index).to_numpy(object),
            "tgrade": _label_column(df_a, "tgrade"),
            "pnodes": _numeric_column(df_a, "pnodes"),
            "progrec": _numeric_column(df_a, "progrec"),
            "estrec": _numeric_column(df_a, "estrec"),
            "A": (df_a[treatment].astype(str) != str(control_value)).astype(int).to_numpy(),
            "time": _numeric_column(df_a, "time"),
            "cens": _numeric_column(df_a, "cens"),
        }
    )
    hist = pd.DataFrame(
        {
            "orig_index": h.index.to_numpy(),
            "age": _numeric_column(h, "age"),
            "menostat": _label_column(h, "menostat"),
            "tsize_grp": _label_column(h, "tsize"),
            "tgrade": _label_column(h, "tgrade"),
            "pnodes": _numeric_column(h, "pnodes"),
            "progrec": _numeric_column(h, "progrec"),
            "estrec": _numeric_column(h, "estrec"),
            "A": 0,
            "year": _to_numeric(_column_or_nan(df_b, "year").reindex(h.index)),
            "time": _numeric_column(h, "time"),
            "cens": _numeric_column(h, "cens"),
        }
    )
    for frame in (trial, hist):
        frame.insert(3, "post", (frame["menostat"] == "Post").astype(int))
        frame.insert(frame.columns.get_loc("estrec") + 1, "er_cat", _receptor_category(frame["estrec"], receptor_cuts))
        frame.insert(
            frame.columns.get_loc("er_cat") + 1, "pgr_cat", _receptor_category(frame["progrec"], receptor_cuts)
        )

    funnel, eligible = [], {}

    def record_step(src, df, name, n_before):
        funnel.append(
            {
                "source": src,
                "step": name,
                "n_after": len(df),
                "n_removed": n_before - len(df),
                "n_treated": int((df["A"] == 1).sum()),
                "n_control": int((df["A"] == 0).sum()),
            }
        )

    for src, df in (("A", trial), ("B", hist)):
        record_step(src, df, "start (Stage-2)", len(df))
        if age_max is not None:
            n_before = len(df)
            df = df[df["age"] <= age_max]
            record_step(src, df, f"age <= {age_max:g}", n_before)
        if landmark is not None:
            n_before = len(df)
            df = df[df["time"] > landmark]
            record_step(src, df, f"event-free and under follow-up at day {landmark:g}", n_before)
        if src == "B" and b_years is not None:
            n_before = len(df)
            df = df[df["year"].between(*b_years)]
            record_step(src, df, f"surgery {b_years[0]}-{b_years[1] % 100:02d}", n_before)
        eligible[src] = df.sort_values("orig_index").reset_index(drop=True)

    trial, hist = eligible["A"], eligible["B"]
    landmark_ok = landmark is None or all(len(f) > 0 and f["time"].min() > landmark for f in (trial, hist))
    return {
        "trial_cov": trial.drop(columns=["time", "cens"]),
        "hist_cov": hist.drop(columns=["time", "cens"]),
        "outcomes": {"trial": trial[["time", "cens"]], "hist": hist[["time", "cens"]]},
        "funnel": pd.DataFrame(funnel),
        "landmark_ok": bool(landmark_ok),
        "params": {"age_max": age_max, "landmark": landmark, "b_years": b_years},
    }


def _column_or_nan(df, name):
    """Column ``name`` of ``df``, or all-NaN if it is missing (gate G3 reports missing columns)."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _numeric_column(df, name):
    """Column ``name`` as floats, with non-numeric values and a missing column as NaN."""
    return _to_numeric(_column_or_nan(df, name))


def _label_column(df, name):
    return _column_or_nan(df, name).to_numpy(object)


def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(float)


def _receptor_category(values, cuts):
    """0 / 1 / 2 for < cuts[0], [cuts[0], cuts[1]), >= cuts[1]; NaN stays NaN."""
    values = np.asarray(values, float)
    cat = np.where(values < cuts[0], 0.0, np.where(values < cuts[1], 1.0, 2.0))
    cat[np.isnan(values)] = np.nan
    return cat.astype(int) if not np.isnan(cat).any() else cat


def design_trial(df):
    """Within-trial design D_t (8 columns, see ``DESIGN_TRIAL_NAMES``).

    Parameters
    ----------
    df : pandas.DataFrame
        Trial covariates (``trial_cov`` rows); must not contain ``time`` / ``cens``.

    Returns
    -------
    numpy.ndarray
        age, post, tsize_mm, grade II, grade III, log1p(pnodes), log1p(progrec), log1p(estrec). A missing or
        unknown menostat / tgrade label gives NaN indicators (never the reference level); missing numbers stay NaN.
    """
    _check_no_outcome_columns(df)
    c = TRIAL_COLUMNS
    grade = df[c["tgrade"]]
    return np.column_stack(
        [
            _as_float(df[c["age"]]),
            _indicator(df[c["menostat"]], "Post", MENO_CODES),
            _as_float(df[c["tsize"]]),
            _indicator(grade, "II", GRADE_CODES),
            _indicator(grade, "III", GRADE_CODES),
            np.log1p(_as_float(df[c["pnodes"]])),
            np.log1p(_as_float(df[c["progrec"]])),
            np.log1p(_as_float(df[c["estrec"]])),
        ]
    )


def design_trial_ext(df):
    """D_t plus age^2, age x post and log1p(pnodes)^2 (11 columns, for balance checks)."""
    D = design_trial(df)
    return np.column_stack([D, D[:, 0] ** 2, D[:, 0] * D[:, 1], D[:, 5] ** 2])


def design_source(df):
    """Cross-source design D_s (9 columns, no grade; see ``DESIGN_SOURCE_NAMES``).

    Parameters
    ----------
    df : pandas.DataFrame
        Trial or historical covariates with ``tsize_grp``, ``er_cat`` and ``pgr_cat``.

    Returns
    -------
    numpy.ndarray
        age, post, size 20-50, size >50, log1p(pnodes), ER 10-99, ER >=100, PgR 10-99, PgR >=100. A missing or
        unknown menostat, size group or receptor category gives NaN indicators (never the reference level).
    """
    _check_no_outcome_columns(df)
    c = SOURCE_COLUMNS
    size, er, pgr, receptor_levels = df[c["tsize"]], df[c["estrec"]], df[c["progrec"]], (0, 1, 2)
    return np.column_stack(
        [
            _as_float(df[c["age"]]),
            _indicator(df[c["menostat"]], "Post", MENO_CODES),
            _indicator(size, "20-50", SIZE_CODES),
            _indicator(size, ">50", SIZE_CODES),
            np.log1p(_as_float(df[c["pnodes"]])),
            _indicator(er, 1, receptor_levels),
            _indicator(er, 2, receptor_levels),
            _indicator(pgr, 1, receptor_levels),
            _indicator(pgr, 2, receptor_levels),
        ]
    )


def design_source_logrec(df):
    """D_s with continuous receptors (sensitivity X2): age, post, size groups, log1p(pnodes), log1p(estrec),
    log1p(progrec) (7 columns)."""
    return np.column_stack(
        [design_source(df)[:, :5], np.log1p(_as_float(df["estrec"])), np.log1p(_as_float(df["progrec"]))]
    )


def features_trial(df):
    """TabPFN features F_t: age, menostat, tsize_mm, tgrade, pnodes, progrec, estrec (n x 7).

    menostat (Pre=0, Post=1) and tgrade (I=0, II=1, III=2) use the fixed codes and are categorical
    (``TRIAL_CAT_IDX``). Unknown labels become NaN.
    """
    _check_no_outcome_columns(df)
    c = TRIAL_COLUMNS
    return np.column_stack(
        [
            _as_float(df[c["age"]]),
            _as_float(df[c["menostat"]].map(MENO_CODES)),
            _as_float(df[c["tsize"]]),
            _as_float(df[c["tgrade"]].map(GRADE_CODES)),
            _as_float(df[c["pnodes"]]),
            _as_float(df[c["progrec"]]),
            _as_float(df[c["estrec"]]),
        ]
    )


def features_source(df):
    """TabPFN features F_s: age, menostat, tsize group, pnodes, ER category, PgR category (n x 6).

    menostat (Pre=0, Post=1) and the size group (<=20=0, 20-50=1, >50=2) are categorical (``SOURCE_CAT_IDX``);
    the receptor categories are numeric ordinal.
    """
    _check_no_outcome_columns(df)
    c = SOURCE_COLUMNS
    return np.column_stack(
        [
            _as_float(df[c["age"]]),
            _as_float(df[c["menostat"]].map(MENO_CODES)),
            _as_float(df[c["tsize"]].map(SIZE_CODES)),
            _as_float(df[c["pnodes"]]),
            _as_float(df[c["estrec"]]),
            _as_float(df[c["progrec"]]),
        ]
    )


def recalibration_design(df, m=None):
    """Covariates of the robust-hybrid outcome model: [m_B, 1{grade II}, 1{grade III}, tsize_mm].

    Parameters
    ----------
    df : pandas.DataFrame
        Trial covariates.
    m : array-like, optional
        Prognostic score for the rows of ``df``; None drops the column (gate G7 = "none").

    Returns
    -------
    numpy.ndarray
        A missing or unknown tgrade label gives NaN grade indicators (never grade I).
    """
    _check_no_outcome_columns(df)
    grade = df[TRIAL_COLUMNS["tgrade"]]
    score = [] if m is None else [_as_float(m)]
    return np.column_stack(
        score
        + [
            _indicator(grade, "II", GRADE_CODES),
            _indicator(grade, "III", GRADE_CODES),
            _as_float(df[TRIAL_COLUMNS["tsize"]]),
        ]
    )


def censoring_covariates(df, source):
    """Covariates of the censoring model (sensitivity S4): [age, post, log1p(pnodes), size].

    Parameters
    ----------
    df : pandas.DataFrame
        Trial or historical covariates; must not contain ``time`` / ``cens``.
    source : {"A", "B"}
        "A" uses ``tsize_mm``; "B" uses the size-group code 0/1/2 of ``tsize_grp`` (``SIZE_CODES``).

    Returns
    -------
    numpy.ndarray
        n x 4 matrix for ``src.survival.censoring_G``, to be fitted within source x arm cells. A missing or
        unknown menostat or size label gives NaN, which ``cox_ph`` refuses.
    """
    _check_no_outcome_columns(df)
    size = _as_float(df["tsize_mm"]) if source == "A" else _as_float(df["tsize_grp"].map(SIZE_CODES))
    return np.column_stack(
        [_as_float(df["age"]), _indicator(df["menostat"], "Post", MENO_CODES), np.log1p(_as_float(df["pnodes"])), size]
    )


def _check_no_outcome_columns(df):
    """Raise ``ValueError`` if ``time`` or ``cens`` reaches the outcome-free design stage."""
    bad = [c for c in ("time", "cens") if c in df.columns]
    if bad:
        raise ValueError(f"outcome columns {bad} must not reach the design stage")


def _as_float(s):
    return np.asarray(s, dtype=float)


def _indicator(s, level, known):
    """1{s == level} as float; NaN where ``s`` is missing or not one of the ``known`` labels.

    A missing or unknown label must not become the reference level: the NaN reaches the logistic models and
    entropy balancing, which refuse it (no imputation is pre-specified).
    """
    return pd.Series(s).map({k: float(k == level) for k in known}).to_numpy(float)
