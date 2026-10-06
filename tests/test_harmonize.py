import json

import pandas as pd
import pytest
from conftest import fake_jev_transport

from src.demo import read_json
from src.harmonize import (
    _assign_one_to_one,
    apply_transform,
    approve_all,
    cached_llm,
    check_entry,
    check_pair,
    coverage,
    default_decision,
    evaluate_mapping,
    harmonize_frames,
    indicator_note,
    load_mapping,
    mapping_warnings,
    post_check,
    profile_columns,
    propose_mapping,
    route_column,
    save_mapping,
)
from src.online import jev_score_columns, make_async_jev

# What the fake Jev says for (A column, B column). Deliberately imperfect, like the real thing:
# overconfident (0.95 for 'size' although a unit/type conversion is needed), one weak match,
# two plausible rivals, and a decoy above the LLM threshold.
GOOD = {
    ("age", "age"): 0.99,
    ("menostat", "meno"): 0.97,
    ("tsize", "size"): 0.95,
    ("tgrade", "grade"): 0.75,
    ("pnodes", "nodes"): 0.98,
    ("progrec", "pgr"): 0.96,
    ("estrec", "er"): 0.96,
    ("horTh", "hormon"): 0.97,
    ("time", "rtime"): 0.95,
    ("time", "dtime"): 0.93,
    ("cens", "recur"): 0.94,
    ("cens", "death"): 0.92,
    ("horTh", "chemo"): 0.62,
}


def score(a, b):
    return GOOD.get((a, b), 0.03)


@pytest.fixture
def scores(raw, jev_env):
    df_a, df_b, la, lb = raw
    client = make_async_jev(fake_jev_transport(score))
    return jev_score_columns(client, profile_columns(df_a, la), profile_columns(df_b, lb), mode="pairwise")


def test_profiles(raw):
    df_a, df_b, la, lb = raw
    pa, pb = profile_columns(df_a, la), profile_columns(df_b, lb)
    assert set(pa["age"]) == set(pb["age"]) or set(pa["age"]) >= {
        "name",
        "dtype",
        "n_unique",
        "pct_missing",
        "min",
        "max",
        "mean",
    }
    meno = pa["menostat"]
    assert meno["kind"] == "categorical" and meno["top_values"]["Post"] > 0.5 and meno["levels"] == ["Post", "Pre"]
    assert pa["tsize"]["unit_guess"] == "mm" and pb["progrec" if "progrec" in pb else "pgr"]["unit_guess"] == "fmol/l"
    assert pb["size"]["kind"] == "categorical" and pa["tsize"]["kind"] == "numeric"
    json.dumps(pa)


def test_pairwise_scores_shape(scores, raw):
    assert scores["age"]["age"] == pytest.approx(0.99) and scores["age"]["year"] == pytest.approx(0.03)
    assert set(scores) == set(raw[0].columns) and set(scores["age"]) == set(raw[1].columns)


def test_ranking_mode(raw, jev_env):
    df_a, df_b, la, lb = raw
    client = make_async_jev(fake_jev_transport(score))
    s = jev_score_columns(client, profile_columns(df_a, la), profile_columns(df_b, lb), mode="ranking")
    assert max(s["pnodes"], key=s["pnodes"].get) == "nodes"
    assert sum(s["pnodes"].values()) <= 1.0001


def test_checks(raw):
    df_a, df_b, la, lb = raw
    pa, pb = profile_columns(df_a, la), profile_columns(df_b, lb)
    assert check_pair(pa["age"], pb["age"])["ok"]
    assert not check_pair(pa["tsize"], pb["size"])["ok"]  # numeric vs categorical
    assert not check_pair(pa["age"], pb["year"])["ok"]  # scale / range
    assert not check_pair(pa["menostat"], pb["meno"])["ok"]  # Pre/Post vs 0/1
    cm = profile_columns(pd.DataFrame({"h": [150.0, 160, 170, 180, 190, 175, 165, 155, 185, 172, 168, 178]}))["h"]
    m = profile_columns(pd.DataFrame({"h": [1.5, 1.6, 1.7, 1.8, 1.9, 1.75, 1.65, 1.55, 1.85, 1.72, 1.68, 1.78]}))["h"]
    c = check_pair(cm, m)
    assert not c["ok"] and "unit" in c["reason"]


def test_routing(scores, raw):
    df_a, df_b, la, lb = raw
    pa, pb = profile_columns(df_a, la), profile_columns(df_b, lb)

    def r(a):
        return route_column(scores[a], {b: check_pair(pa[a], pb[b]) for b in df_b.columns})

    assert r("age")["route"] == "auto" and r("pnodes")["route"] == "auto"
    assert r("tsize")["route"] == "llm"  # high score but failed checks
    assert r("tgrade")["route"] == "llm"  # mid score
    assert r("time")["route"] == "llm" and {"rtime", "dtime"} <= set(r("time")["candidates"])  # two rivals
    assert len(r("tgrade")["candidates"]) == 5  # the LLM always sees the 5 best columns
    assert route_column({"x": 0.2, "y": 0.1}, {"x": {"ok": True}, "y": {"ok": True}}, lo=0.3)["route"] == "none"
    # lo = 0: always ask the LLM
    assert route_column({"x": 0.2, "y": 0.1}, {"x": {"ok": True}, "y": {"ok": True}})["route"] == "llm"


def test_routing_real_jev_pattern():
    """Pattern seen with the real Jev: true match below 0.6, derived outcome's 2nd column at 0.25."""
    ok = {b: {"ok": True} for b in ["rtime", "dtime", "recur", "death", "pid", "year"]}
    sc = {"rtime": 0.70, "dtime": 0.25, "recur": 0.02, "death": 0.02, "pid": 0.01, "year": 0.01}
    r = route_column(sc, ok)
    assert r["route"] == "llm" and {"rtime", "dtime", "recur", "death"} <= set(r["candidates"])
    assert route_column({"grade": 0.51, "size": 0.02}, {"grade": {"ok": True}, "size": {"ok": True}})["route"] == "llm"


def test_full_proposal_precision_recall(scores, raw, truth):
    df_a, df_b, la, lb = raw
    pa, pb = profile_columns(df_a, la), profile_columns(df_b, lb)
    m = propose_mapping(df_a, df_b, pa, pb, scores, cached_llm(read_json("llm_harmonize.json")))
    ev = evaluate_mapping(m, truth)
    assert ev["precision"] == 1.0 and ev["recall"] == 1.0, ev["rows"]
    assert ev["per_route"]["Jev"]["n"] == 4 and ev["per_route"]["LLM"]["n"] == 6
    assert m["age"]["decided_by"] == "Jev" and m["tsize"]["decided_by"] == "LLM"
    assert m["tgrade"]["same_definition"] == "unsure" and "I-III" in m["tgrade"]["definition_note"]
    assert all(e["post_check"]["ok"] for e in m.values())
    assert default_decision(m["age"]) == "approve" and default_decision(m["tsize"]) == "pending"


def test_wrong_proposal_lowers_precision(truth):
    prop = {
        "age": {"b_cols": ["year"], "decided_by": "Jev"},
        "pnodes": {"b_cols": ["nodes"], "decided_by": "Jev"},
        "time": {"b_cols": ["rtime"], "decided_by": "LLM"},
    }
    ev = evaluate_mapping(prop, truth)
    assert ev["n_proposed"] == 3 and ev["n_correct"] == 1 and ev["precision"] == pytest.approx(1 / 3)
    assert ev["recall"] == pytest.approx(1 / 10) and ev["per_route"]["Jev"]["accuracy"] == 0.5


def test_no_jev_all_columns_to_llm(raw, truth):
    df_a, df_b, la, lb = raw
    m = propose_mapping(
        df_a,
        df_b,
        profile_columns(df_a, la),
        profile_columns(df_b, lb),
        None,
        cached_llm(read_json("llm_harmonize.json")),
    )
    ev = evaluate_mapping(m, truth)
    assert ev["recall"] == 1.0 and ev["per_route"] == {"LLM": {"n": 10, "accuracy": 1.0}}


def test_no_jev_no_llm_is_empty(raw):
    df_a, df_b, la, lb = raw
    m = propose_mapping(df_a, df_b, profile_columns(df_a, la), profile_columns(df_b, lb), None, None)
    assert all(not e["b_cols"] for e in m.values()) and "no LLM" in m["age"]["reason"]


def test_hungarian_prevents_duplicates():
    def entry(a, b, s, cands):
        return {
            "a_col": a,
            "b_cols": [b],
            "score": s,
            "decided_by": "Jev",
            "reason": "",
            "transform": {"kind": "identity"},
            "candidates": cands,
            "same_definition": "yes",
            "definition_note": "",
        }

    es = {
        "x": entry("x", "p", 0.95, {"p": 0.95, "q": 0.7}),
        "y": entry("y", "p", 0.99, {"p": 0.99}),
        "z": entry("z", "r", 0.9, {"r": 0.9}),
        "d": {**entry("d", "r", 0.9, {}), "b_cols": ["r", "s"]},
    }  # derived: exempt
    _assign_one_to_one(es)
    assert es["y"]["b_cols"] == ["p"] and es["x"]["b_cols"] == ["q"]
    assert es["z"]["b_cols"] == ["r"] and es["d"]["b_cols"] == ["r", "s"]
    singles = [e["b_cols"][0] for k, e in es.items() if k != "d" and e["b_cols"]]
    assert len(singles) == len(set(singles))


def test_hungarian_loser_without_alternative_unmapped():
    def mk(a, s):
        return {
            "a_col": a,
            "b_cols": ["p"],
            "score": s,
            "decided_by": "Jev",
            "reason": "",
            "transform": {"kind": "identity"},
            "candidates": {"p": s},
            "same_definition": "yes",
            "definition_note": "",
        }

    es = {"x": mk("x", 0.9), "y": mk("y", 0.95)}
    _assign_one_to_one(es)
    assert es["y"]["b_cols"] == ["p"] and es["x"]["b_cols"] == [] and es["x"]["decided_by"] == "none"


def test_post_check_is_case_sensitive(raw):
    """'pre' is not 'Pre': they would become two different categories downstream."""
    df_a, df_b, *_ = raw
    lower = apply_transform({"kind": "map_values", "mapping": {"0": "pre", "1": "post"}}, df_b, ["meno"])
    assert not post_check(df_a["menostat"], lower)["ok"]
    exact = apply_transform({"kind": "map_values", "mapping": {"0": "Pre", "1": "Post"}}, df_b, ["meno"])
    assert post_check(df_a["menostat"], exact)["ok"]


def test_transforms_on_real_data(raw):
    df_a, df_b, *_ = raw
    mv = apply_transform({"kind": "map_values", "mapping": {"0": "Pre", "1": "Post"}}, df_b, ["meno"])
    assert (mv == "Post").sum() == (df_b.meno == 1).sum() and mv.notna().all()
    b = apply_transform(
        {"kind": "bin", "edges": [0, 20, 50, 1000], "labels": ["<=20", "20-50", ">50"], "apply_to": "A"},
        df_a,
        ["tsize"],
    )
    assert (b == "<=20").sum() == (df_a.tsize <= 20).sum() and (b == ">50").sum() == (df_a.tsize > 50).sum()
    assert b[df_a.tsize == 20].eq("<=20").all() and b[df_a.tsize == 21].eq("20-50").all()
    us = apply_transform({"kind": "unit_scale", "multiply": 0.1, "add": 1}, df_a, ["tsize"])
    assert us.iloc[0] == pytest.approx(df_a.tsize.iloc[0] * 0.1 + 1)


def test_outcome_derivation_matches_survival_docs(demo):
    df_b, b_h = demo["df_b"], demo["b_h"]
    ignore = (df_b.recur == 0) & (df_b.death == 1) & (df_b.rtime < df_b.dtime)
    use_recur = (df_b.recur == 1) | ignore
    event = df_b.recur.where(use_recur, df_b.death)
    time = df_b.rtime.where(use_recur, df_b.dtime)
    assert (b_h["cens"] == event).all() and (b_h["time"] == time).all()
    assert ignore.sum() == 43


def test_harmonize_frames_only_approved(raw):
    df_a, df_b, la, lb = raw
    m = propose_mapping(
        df_a,
        df_b,
        profile_columns(df_a, la),
        profile_columns(df_b, lb),
        None,
        cached_llm(read_json("llm_harmonize.json")),
    )
    m = approve_all(m)
    m["tgrade"]["status"] = "rejected"
    a_h, b_h, info = harmonize_frames(df_a, df_b, m)
    assert "tgrade" not in b_h.columns and info["unmapped_a"] == ["tgrade"]
    assert set(info["dropped_b"]) == {"pid", "year", "chemo"} | {"grade"}
    assert coverage(m, list(df_a.columns)) == "9 of 10 A columns mapped, missing: tgrade"
    assert set(a_h.tsize.unique()) == {"<=20", "20-50", ">50"}


def test_failing_transform_is_reported_not_crash(raw):
    df_a, df_b, la, lb = raw
    m = approve_all(
        propose_mapping(
            df_a,
            df_b,
            profile_columns(df_a, la),
            profile_columns(df_b, lb),
            None,
            cached_llm(read_json("llm_harmonize.json")),
        )
    )
    m["time"]["transform"] = {"kind": "derive", "expr": "min(rtime, dtime)"}
    a_h, b_h, info = harmonize_frames(df_a, df_b, m)
    assert "time" not in b_h.columns and "not allowed" in info["errors"]["time"]


def test_llm_is_reasked_when_transform_fails(raw):
    df_a, df_b, la, lb = raw
    good = read_json("llm_harmonize.json")["time"]
    bad = {**good, "transform": {"kind": "derive", "expr": "min(rtime, dtime)"}}
    seen = []

    def llm(system, user):
        seen.append(json.loads(user))
        return json.dumps(bad if len(seen) == 1 else good)

    pa, pb = profile_columns(df_a, la), profile_columns(df_b, lb)
    m = propose_mapping(df_a[["time"]], df_b, {"time": pa["time"]}, pb, None, llm)
    assert len(seen) == 2 and "not allowed" in seen[1]["error_when_applying_it"]
    assert m["time"]["transform"]["expr"] == good["transform"]["expr"]


def test_mapping_roundtrip(tmp_path, raw):
    df_a, df_b, la, lb = raw
    m = approve_all(
        propose_mapping(
            df_a,
            df_b,
            profile_columns(df_a, la),
            profile_columns(df_b, lb),
            None,
            cached_llm(read_json("llm_harmonize.json")),
        )
    )
    save_mapping(m, tmp_path / "m.json")
    m2 = load_mapping(tmp_path / "m.json")
    a1, b1, _ = harmonize_frames(df_a, df_b, m)
    a2, b2, _ = harmonize_frames(df_a, df_b, m2)
    pd.testing.assert_frame_equal(b1, b2)


def test_roman_numeral_meaning_check(raw):
    assert mapping_warnings({"kind": "map_values", "mapping": {"2": "I", "3": "II"}})
    assert not mapping_warnings({"kind": "map_values", "mapping": {"2": "II", "3": "III"}})
    df_a, df_b, *_ = raw
    e = {"a_col": "tgrade", "b_cols": ["grade"], "transform": {"kind": "map_values", "mapping": {"2": "I", "3": "II"}}}
    c = check_entry(df_a, df_b, e)
    assert not c["ok"] and "changes the number" in c["reason"]


def test_indicator_note_counts_dropped_events(raw):
    df_b = raw[1]
    cols = ["recur", "death", "rtime", "dtime"]
    conservative = apply_transform(read_json("llm_harmonize.json")["cens"]["transform"], df_b, cols)
    assert "43 patients" in indicator_note(df_b, cols, conservative)


def test_time_and_event_are_made_consistent_in_one_call(raw):
    """Separate answers for time and cens that disagree are replaced by one joint answer."""
    df_a, df_b, la, lb = raw
    cache = read_json("llm_harmonize.json")
    sep_cens = {**cache["cens"], "transform": {"kind": "derive", "expr": "1 if (death == 1 or recur == 1) else 0"}}
    calls = []

    def llm(system, user):
        calls.append(system)
        if "SEVERAL A columns" in system:
            return json.dumps({"answers": {"time": cache["time"], "cens": cache["cens"]}})
        name = json.loads(user)["a_column"]["name"]
        return json.dumps(sep_cens if name == "cens" else cache[name])

    pa, pb = profile_columns(df_a, la), profile_columns(df_b, lb)
    m = propose_mapping(df_a, df_b, pa, pb, None, llm)
    assert sum("SEVERAL A columns" in c for c in calls) == 1
    assert m["cens"]["transform"]["expr"] == cache["cens"]["transform"]["expr"]
    assert "made consistent" in m["cens"]["reason"]
