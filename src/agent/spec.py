"""The study spec: everything the old gated code had typed in, as data the LLM writes and the code checks."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics import roc_auc_score

TIME_UNIT_DAYS = {"days": 1.0, "weeks": 7.0, "months": 30.4375, "years": 365.25}
DAYS = 365.25
MAX_MISSING = 0.05
MAX_CATEGORIES = 12


class Covariate(BaseModel):
    """One baseline covariate (measured before treatment)."""

    name: str
    kind: Literal["numeric", "binary", "categorical", "ordinal"]
    cross_source: bool = True  # recorded the same way in the trial and in the historical data
    essential: bool = False  # a known strong prognostic or assignment factor: the repair loop may not drop it
    why: str = ""


class StudySpec(BaseModel):
    """What to estimate and how; written by the LLM, validated by code, frozen before any gate runs."""

    treatment: str
    control_value: str
    time: str
    event: str
    event_values: list[str]  # values of the event column that mean "event happened"; everything else is censored
    time_unit: Literal["days", "weeks", "months", "years"] = "days"
    horizon_years: float = Field(gt=0)
    landmark_days: float = Field(default=0, ge=0)
    covariates: list[Covariate]
    rationale: str = ""

    @field_validator("control_value", mode="before")
    @classmethod
    def _control_as_text(cls, v):
        return key(v)

    @field_validator("event_values", mode="before")
    @classmethod
    def _events_as_text(cls, v):
        return [key(x) for x in (v if isinstance(v, list) else [v])]

    def days(self, years):
        return years * DAYS

    def unit_days(self):
        return TIME_UNIT_DAYS[self.time_unit]


def key(v):
    """Compare category values as text: 1, 1.0 and '1' are the same value."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def keys(series):
    return series.map(key)


def normalize_spec(spec, trial, hist):
    """Fix what code can fix without asking: a covariate missing (or mostly missing) in the history is not comparable.

    Returns the corrected spec and a list of notes.
    """
    notes, covs = [], []
    for c in spec.covariates:
        if c.cross_source and hist is not None:
            if c.name not in hist.columns:
                c, why = c.model_copy(update={"cross_source": False}), "not in the historical data"
            elif hist[c.name].isna().mean() > MAX_MISSING:
                c, why = c.model_copy(update={"cross_source": False}), "more than 5% missing in the historical data"
            else:
                why = ""
            if why:
                notes.append(f"{c.name}: cross_source set to false ({why})")
        covs.append(c)
    return spec.model_copy(update={"covariates": covs}), notes


def validate_spec(spec, trial, hist):
    """Check a spec against the data. Returns (errors, warnings); errors make the planner try again."""
    errors, warnings = [], []
    for col in (spec.treatment, spec.time, spec.event):
        if col not in trial.columns:
            errors.append(f"column '{col}' is not in the trial table (columns: {list(trial.columns)})")
    if errors:
        return errors, warnings
    _check_treatment(spec, trial, errors)
    _check_outcome(spec, trial, errors, warnings)
    _check_covariates(spec, trial, hist, errors, warnings)
    if hist is not None:
        for col in (spec.time, spec.event):
            if col not in hist.columns:
                warnings.append(f"'{col}' is not in the historical data: no borrowing is possible")
    return errors, warnings


def _check_treatment(spec, trial, errors):
    t = keys(trial[spec.treatment])
    levels = sorted(set(t))
    if "nan" in levels:
        errors.append(f"treatment '{spec.treatment}' has missing values")
    elif len(levels) != 2:
        errors.append(f"treatment '{spec.treatment}' must have exactly 2 values, it has {levels}")
    elif spec.control_value not in levels:
        errors.append(f"control_value '{spec.control_value}' is not a value of '{spec.treatment}' ({levels})")


def _check_outcome(spec, trial, errors, warnings):
    time = pd.to_numeric(trial[spec.time], errors="coerce")
    if time.isna().any():
        errors.append(f"time '{spec.time}' has missing or non-numeric values")
        return
    if (time <= 0).mean() > 0.01:
        errors.append(f"time '{spec.time}' has non-positive values")
    present = set(keys(trial[spec.event]))
    if not set(spec.event_values) & present:
        errors.append(
            f"event_values {spec.event_values} never occur in '{spec.event}' (values: {sorted(present)[:10]})"
        )
        return
    rate = keys(trial[spec.event]).isin(spec.event_values).mean()
    if not 0.02 < rate < 0.98:
        errors.append(f"event rate {rate:.0%} is implausible: check event_values")
    horizon = spec.days(spec.horizon_years)
    follow_up = time.max() * spec.unit_days()
    if horizon >= follow_up:
        errors.append(
            f"horizon {spec.horizon_years} y is not shorter than the longest follow-up ({follow_up / DAYS:.1f} y)"
        )
    if spec.landmark_days >= horizon / 2:
        errors.append("landmark_days must be well below the horizon")


def _check_covariates(spec, trial, hist, errors, warnings):
    names = [c.name for c in spec.covariates]
    if len(set(names)) != len(names):
        errors.append("a covariate is listed twice")
    if len(names) < 2:
        errors.append("at least 2 covariates are needed")
    banned = {spec.treatment, spec.time, spec.event}
    a = (keys(trial[spec.treatment]) != spec.control_value).astype(int) if spec.treatment in trial else None
    for c in spec.covariates:
        if c.name in banned:
            errors.append(f"covariate '{c.name}' is the treatment, time or event column")
        elif c.name not in trial.columns:
            errors.append(f"covariate '{c.name}' is not in the trial table")
        else:
            _check_one_covariate(c, trial[c.name], a, errors, warnings)
    if hist is not None and not any(c.cross_source for c in spec.covariates):
        warnings.append("no covariate is comparable across the two sources: no borrowing is possible")


def _check_one_covariate(c, s, a, errors, warnings):
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if s.isna().mean() > MAX_MISSING:
        errors.append(f"covariate '{c.name}' is {s.isna().mean():.0%} missing in the trial (limit {MAX_MISSING:.0%})")
    levels = s.dropna().map(key).nunique()
    if c.kind == "numeric" and not numeric:
        errors.append(f"covariate '{c.name}' is marked numeric but is not a number column")
    if c.kind == "binary" and levels != 2:
        errors.append(f"covariate '{c.name}' is marked binary but has {levels} values")
    if c.kind == "categorical" and levels > MAX_CATEGORIES:
        errors.append(f"covariate '{c.name}' has {levels} categories (limit {MAX_CATEGORIES}); bin it or drop it")
    if numeric and a is not None and s.notna().all() and s.nunique() > 1:
        auc = roc_auc_score(a, s)
        if max(auc, 1 - auc) >= 0.98:
            errors.append(f"covariate '{c.name}' alone separates treated from control (AUC {auc:.2f}): leak?")


def covariate_table(spec, trial, hist):
    """Per-covariate summary the planner and the report can show."""
    rows = []
    for c in spec.covariates:
        rows.append(
            {
                "name": c.name,
                "kind": c.kind,
                "cross_source": c.cross_source,
                "essential": c.essential,
                "missing_trial": float(trial[c.name].isna().mean()) if c.name in trial else None,
                "missing_hist": float(hist[c.name].isna().mean()) if hist is not None and c.name in hist else None,
            }
        )
    return pd.DataFrame(rows)
