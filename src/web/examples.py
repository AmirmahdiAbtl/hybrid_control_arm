"""The bundled datasets the runner offers, so that the app can be tried without any file."""

import json
from pathlib import Path

import pandas as pd

from src.demo import DEMO, PATH_A, PATH_B

EXAMPLES = {
    "gbsg2": {
        "title": "GBSG-2 + Rotterdam (breast cancer)",
        "text": "686 trial patients (tamoxifen or not) and 2,982 historical patients from another study with "
        "different column names and codings. The agent has to match them and find the 552 who would have qualified.",
        "trial": PATH_A,
        "hist": PATH_B,
        "trial_dict": DEMO / "dictionary_A.json",
        "hist_dict": DEMO / "dictionary_B.json",
        "nct": "NCT00929591",
        "registry": "GBSG-2 itself is not on ClinicalTrials.gov (it ran 1984-1989). NCT00929591 is a stand-in: the "
        "closest registered trial (tamoxifen, node-positive breast cancer, 1989). It only gives the reviewer context; "
        "the eligibility text is GBSG-2's own. Clear the field to run without it.",
        "eligibility": DEMO / "eligibility_gbsg2.txt",
    },
}


def load_example(key):
    """``(trial, hist, trial_labels, hist_labels, eligibility_text)`` of a bundled example."""
    e = EXAMPLES[key]
    text = Path(e["eligibility"]).read_text() if e["eligibility"] else ""
    labels = [json.loads(Path(e[k]).read_text()) for k in ("trial_dict", "hist_dict")]
    return pd.read_csv(e["trial"]), pd.read_csv(e["hist"]), labels[0], labels[1], text
