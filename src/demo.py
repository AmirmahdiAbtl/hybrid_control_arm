import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.harmonize import approve_all, cached_llm, harmonize_frames, load_mapping, profile_columns, propose_mapping
from src.online import Criterion
from src.rules import apply_rules

ROOT = Path(__file__).resolve().parent.parent
DATA, DEMO = ROOT / "data", ROOT / "demo"
PATH_A = DATA / "data_A_new_trial_GBSG2.csv"
PATH_B = DATA / "data_B_old_study_Rotterdam.csv"
TRUTH = DATA / "true_mapping.json"
APPROVED = DEMO / "approved_mapping.json"


def prepare(df_a, df_b, mapping, rules, treatment="horTh", control_value="no"):
    """Stages 1-2 for any two datasets: apply the approved mapping, then the eligibility rules.

    Parameters
    ----------
    df_a, df_b : pandas.DataFrame
        New trial and old study (raw).
    mapping : dict
        Approved mapping (see :func:`src.harmonize.propose_mapping`).
    rules : list of dict
        Eligibility rules (``id``, ``text``, ``rule``); they refer to B's raw column names.
    treatment, control_value : str
        Treatment column of A and the value that means control.

    Returns
    -------
    dict
        ``a_h``, ``b_h``, ``hist`` (eligible), ``trial``, ``funnel``, ``invalid``, ``mask``, ``info``.
    """
    a_h, b_h, info = harmonize_frames(df_a, df_b, mapping)
    mask, funnel, invalid = apply_rules(df_b, rules)
    hist, trial = build_frames(a_h, b_h, mask, treatment, control_value)
    return {
        "a_h": a_h,
        "b_h": b_h,
        "hist": hist,
        "trial": trial,
        "funnel": funnel,
        "invalid": invalid,
        "mask": mask,
        "info": info,
    }


def build_frames(a_h, b_h, mask, treatment="horTh", control_value="no"):
    """Historical (eligible) and trial frames used by stages 3-5.

    Parameters
    ----------
    a_h, b_h : pandas.DataFrame
        Harmonized trial and historical data (A's column names).
    mask : pandas.Series
        Eligible historical patients (index of ``b_h``).
    treatment, control_value : str
        Treatment column of the trial and the value that means control.

    Returns
    -------
    hist, trial : pandas.DataFrame
        ``trial`` gets an ``arm`` column ('control' / 'treated').
    """
    hist = b_h[mask.reindex(b_h.index, fill_value=False)].copy()
    trial = a_h.copy()
    trial["arm"] = np.where(trial[treatment].astype(str) == str(control_value), "control", "treated")
    return hist, trial


def demo_pipeline(mapping_path=None):
    """Harmonize and filter the demo with an approved mapping (the cached LLM answers if no path is given).

    Returns
    -------
    dict
        Output of :func:`prepare` plus ``df_a``, ``df_b`` and ``mapping``.
    """
    df_a, df_b, labels_a, labels_b = load_demo_data()
    mapping = load_mapping(mapping_path) if mapping_path else demo_mapping(df_a, df_b, labels_a, labels_b)
    rules = [c.model_dump() for c in demo_criteria() if c.kind == "rule"]
    return {"df_a": df_a, "df_b": df_b, "mapping": mapping, **prepare(df_a, df_b, mapping, rules)}


def load_demo_data():
    """Return ``(df_a, df_b, labels_a, labels_b)`` of the demo."""
    return pd.read_csv(PATH_A), pd.read_csv(PATH_B), read_json("dictionary_A.json"), read_json("dictionary_B.json")


def demo_criteria():
    """The cached LLM split of the demo eligibility text, as ``Criterion`` objects."""
    return [Criterion(**c) for c in read_json("llm_eligibility.json")["criteria"]]


def demo_mapping(df_a, df_b, labels_a, labels_b):
    """Approved mapping from the cached LLM answers (no Jev scores available offline)."""
    pa, pb = profile_columns(df_a, labels_a), profile_columns(df_b, labels_b)
    m = propose_mapping(df_a, df_b, pa, pb, scores=None, llm=cached_llm(read_json("llm_harmonize.json")))
    return approve_all(m)


def read_json(name):
    """Read a JSON file from the demo folder."""
    return json.loads((DEMO / name).read_text())
