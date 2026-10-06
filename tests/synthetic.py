"""A small generated dataset for the agent tests: a trial with two arms and a historical study.

No real disease and no bundled file. The trial has an arm (``drug`` / ``control``), a time in days and a status
(0 censored, 1 other, 2 event). The historical patients are followed for at most 1,280 days (3.5 years), so a 4-year
horizon fails the follow-up gate and the longest horizon every group supports is 3.2 years. ``lab`` is measured only
in the trial. The treatment has no effect.
"""

import numpy as np
import pandas as pd

HIST_FOLLOW_UP_DAYS = 1280
TRIAL_FOLLOW_UP_DAYS = 2000


def _patients(rng, n):
    age = rng.normal(58, 10, n).round(1)
    sex = rng.choice(["f", "m"], n, p=[0.6, 0.4])
    marker_a = np.exp(rng.normal(1.0, 0.6, n)).round(2)
    marker_b = rng.normal(10, 2, n).round(2)
    grade = rng.choice([1, 2, 3], n, p=[0.3, 0.45, 0.25])
    risk = 0.03 * (age - 58) + 0.4 * np.log(marker_a) + 0.25 * (grade - 2) + 0.1 * (sex == "m")
    return pd.DataFrame({"age": age, "sex": sex, "marker_a": marker_a, "marker_b": marker_b, "grade": grade}), risk


def _follow_up(rng, risk, limit):
    time = rng.exponential(2200 / np.exp(risk))
    drop = rng.exponential(9000, len(time))
    observed = np.minimum.reduce([time, drop, np.full(len(time), float(limit))])
    event = (time <= np.minimum(drop, limit)).astype(int) * 2
    other = (rng.random(len(time)) < 0.08) & (event == 0)
    return observed.round(0).clip(min=1), np.where(other, 1, event)


def make_data(seed=0, n_trial=320, n_hist=140):
    """``(trial, hist)`` DataFrames."""
    rng = np.random.default_rng(seed)
    trial, risk = _patients(rng, n_trial)
    trial["arm"] = rng.permutation(np.r_[np.full(n_trial // 2, "drug"), np.full(n_trial - n_trial // 2, "control")])
    trial["time"], trial["status"] = _follow_up(rng, risk, TRIAL_FOLLOW_UP_DAYS)
    trial["lab"] = rng.normal(5, 1, n_trial).round(2)
    trial.loc[rng.choice(n_trial, 6, replace=False), "marker_b"] = np.nan  # a few missing values
    hist, risk = _patients(rng, n_hist)
    hist["time"], hist["status"] = _follow_up(rng, risk, HIST_FOLLOW_UP_DAYS)
    hist["lab"] = np.nan  # not measured in the historical study
    cols = ["arm", "time", "status", "age", "sex", "marker_a", "marker_b", "grade", "lab"]
    return trial[cols], hist[[c for c in cols if c != "arm"]]


LABELS = {
    "arm": "treatment arm: drug or control",
    "time": "days from randomization to the event or last follow-up",
    "status": "0 = censored, 1 = other, 2 = event",
    "age": "age in years",
    "sex": "f / m",
    "marker_a": "baseline marker A (positive, skewed)",
    "marker_b": "baseline marker B",
    "grade": "grade 1-3",
    "lab": "baseline lab value",
}
