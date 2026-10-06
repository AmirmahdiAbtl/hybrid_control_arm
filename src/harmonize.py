import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.online import HARMONIZE_GROUP_SYSTEM, HARMONIZE_SYSTEM, GroupAnswer, HarmonizeAnswer, Transform, llm_json
from src.rules import RuleError, evaluate_expression

# Auto-accept at a Jev score >= HI. If every B column scores below LO there is no match and the LLM is not asked
# (LO = 0: always ask).
HI, LO = 0.90, 0.0
TOP_K = 5  # the LLM always sees the K best B columns (a derived variable needs several)

_UNIT_PATTERNS = [
    (r"\bmm\b", "mm"),
    (r"\bcm\b", "cm"),
    (r"fmol", "fmol/l"),
    (r"\bdays?\b", "days"),
    (r"\byears?\b|\byrs?\b", "years"),
    (r"\bkg\b", "kg"),
    (r"\bmonths?\b", "months"),
    (r"%|percent", "%"),
]
_ROMAN_NUMERALS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


# ----------------------------------------------------------------------------
# Profiling
# ----------------------------------------------------------------------------
def profile_columns(df, labels=None):
    """Profile every column; the same layout is used for both datasets.

    Parameters
    ----------
    labels : dict, optional
        ``{column: description}`` (ADaM / SAS label style).

    Returns
    -------
    dict
        ``{column: profile}`` with name, label, dtype, kind (numeric / categorical), n_unique, pct_missing,
        min / max / mean / median for numeric columns, top_values (value -> frequency) and levels for categorical
        ones, and unit_guess.
    """
    labels = labels or {}
    out = {}
    for c in df.columns:
        s = df[c]
        n_unique = int(s.nunique())
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        p = {
            "name": c,
            "label": labels.get(c, ""),
            "dtype": str(s.dtype),
            "n_unique": n_unique,
            "pct_missing": round(float(s.isna().mean() * 100), 2),
            "kind": "numeric" if numeric and n_unique > 10 else "categorical",
            "unit_guess": guess_unit(c, labels.get(c, "")),
        }
        if numeric:
            p.update(
                min=_plain_scalar(s.min()),
                max=_plain_scalar(s.max()),
                mean=round(float(s.mean()), 3),
                median=_plain_scalar(s.median()),
            )
        if p["kind"] == "categorical":
            vc = s.value_counts(normalize=True, dropna=True)
            p["top_values"] = {_category_key(k): round(float(v), 3) for k, v in vc.head(5).items()}
            if n_unique <= 30:
                p["levels"] = sorted(_category_key(k) for k in vc.index)
        out[c] = p
    return out


def guess_unit(name, label=""):
    """Guess a unit from the label (and a few obvious names); '' if unknown."""
    text = f"{label} {name}".lower()
    for pattern, unit in _UNIT_PATTERNS:
        if re.search(pattern, text):
            return unit
    if re.fullmatch(r"age", name.lower()):
        return "years"
    return ""


def _plain_scalar(v):
    """Plain Python scalar for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _category_key(v):
    """Category as it will appear after harmonizing (1.0 -> '1'); case is kept."""
    if isinstance(v, (float, np.floating)) and not pd.isna(v) and float(v).is_integer():
        v = int(v)
    return str(v).strip()


def _loose_key(v):
    """Loose form for the routing check only (' Pre ' -> 'pre')."""
    return _category_key(v).lower()


# ----------------------------------------------------------------------------
# Deterministic checks and routing
# ----------------------------------------------------------------------------
def check_pair(pa, pb):
    """Type, range and category plausibility of an (A column, B column) pair.

    Returns
    -------
    dict
        ``ok`` (bool) and ``reason`` (text; explains the failure, e.g. a unit factor).
    """
    if pa["kind"] != pb["kind"]:
        return {"ok": False, "reason": f"type mismatch: A is {pa['kind']}, B is {pb['kind']}"}
    if pa["kind"] == "numeric":
        ma, mb = pa["median"], pb["median"]
        if ma and mb:
            r = abs(ma / mb)
            if r > 5 or r < 0.2:
                return {"ok": False, "reason": f"scale differs by about x{r:.3g}: possible unit conversion"}
        if pa["max"] < pb["min"] or pb["max"] < pa["min"]:
            return {
                "ok": False,
                "reason": f"ranges do not overlap (A {pa['min']}..{pa['max']}, B {pb['min']}..{pb['max']})",
            }
        return {"ok": True, "reason": "numeric, ranges plausible"}
    la, lb = {_loose_key(x) for x in pa.get("levels", [])}, {_loose_key(x) for x in pb.get("levels", [])}
    if not la or not lb:
        return {"ok": True, "reason": "too many categories to compare"}
    overlap = len(la & lb) / min(len(la), len(lb))
    if overlap < 0.5:
        return {
            "ok": False,
            "reason": f"category overlap {overlap:.0%} (A {sorted(la)[:5]}, B {sorted(lb)[:5]}): values need recoding",
        }
    return {"ok": True, "reason": f"category overlap {overlap:.0%}"}


def route_column(scores_a, checks_a, hi=HI, lo=LO, top_k=TOP_K):
    """Decide the route for one A column.

    Parameters
    ----------
    scores_a : dict
        ``{b_col: Jev score}`` for this A column.
    checks_a : dict
        ``{b_col: check dict}``.

    Returns
    -------
    dict
        ``route`` ('auto' / 'llm' / 'none'), ``top`` (b_col), ``candidates`` (b_cols the LLM sees: every one
        >= lo and at least the ``top_k`` best).

    Notes
    -----
    Auto-accept needs: top score >= hi, checks pass, and no second candidate >= hi (two equally good matches are
    ambiguous). With real Jev scores the true match is often well below 0.9 (age 0.80, grade 0.51 on the demo)
    and a derived outcome's second column can score 0.25, so the LLM gets the top K columns, not only those >= lo.
    """
    ranked = sorted(scores_a.items(), key=lambda kv: -kv[1])
    top, top_score = ranked[0]
    rivals = [b for b, s in ranked[1:] if s >= hi]
    if top_score >= hi and checks_a[top]["ok"] and not rivals:
        return {"route": "auto", "top": top, "candidates": [b for b, s in ranked if s >= lo]}
    if top_score >= lo and (top_score > 0 or lo == 0):
        candidates = [b for i, (b, s) in enumerate(ranked) if (s >= lo and lo > 0) or i < top_k]
        return {"route": "llm", "top": top, "candidates": candidates}
    return {"route": "none", "top": top, "candidates": []}


# ----------------------------------------------------------------------------
# Transforms and their checks
# ----------------------------------------------------------------------------
def apply_transform(spec, df, cols):
    """Apply a declarative transform (see :class:`src.online.Transform`) to ``cols`` of ``df``.

    Returns
    -------
    pandas.Series
    """
    t = Transform.model_validate(spec)
    if t.kind in ("identity", "rename"):
        return df[cols[0]]
    if t.kind == "map_values":
        m = {_category_key(k): v for k, v in t.mapping.items()}
        return df[cols[0]].map(lambda v: m.get(_category_key(v)) if not pd.isna(v) else np.nan)
    if t.kind == "unit_scale":
        return df[cols[0]] * t.multiply + t.add
    if t.kind == "bin":
        edges = [-np.inf] + list(t.edges[1:-1]) + [np.inf]
        return pd.cut(df[cols[0]], edges, labels=t.labels, right=True).astype(object).where(df[cols[0]].notna())
    if t.kind == "derive":
        return evaluate_expression(t.expr, df)
    raise ValueError(t.kind)


def mapping_warnings(spec):
    """Code mappings that change the number, e.g. grade '2' -> 'I' (should be 'II')."""
    t = Transform.model_validate(spec)
    if t.kind != "map_values":
        return []
    out = []
    for k, v in t.mapping.items():
        k, v = _category_key(k), str(v).strip().upper()
        if k.isdigit() and v in _ROMAN_NUMERALS and int(k) != _ROMAN_NUMERALS[v]:
            out.append(f"'{k}' -> '{v}' changes the number ({k} vs {_ROMAN_NUMERALS[v]})")
    return out


def indicator_note(df_b, cols, new_series):
    """For a derived 0/1 event: how many patients have a source indicator of 1 but end up as 0."""
    if not set(new_series.dropna().unique()) <= {0, 1}:
        return ""
    indicators = [c for c in cols if set(df_b[c].dropna().unique()) <= {0, 1}]
    if not indicators:
        return ""
    lost = int(((df_b[indicators] == 1).any(axis=1) & (new_series == 0)).sum())
    return (
        f"event rate {new_series.mean():.1%}; {lost} patients with "
        f"{' or '.join(c + '=1' for c in indicators)} are counted as event 0"
    )


def post_check(a_series, new_series):
    """Does the transformed column look like A's column? (category overlap / similar range)"""
    if new_series.notna().sum() == 0:
        return {"ok": False, "reason": "transformed column is empty (mapping does not cover the values?)"}
    numeric = pd.api.types.is_numeric_dtype(a_series) and pd.api.types.is_numeric_dtype(new_series)
    if numeric and a_series.nunique() > 10:
        pa = profile_columns(pd.DataFrame({"a": a_series}))["a"]
        pb = profile_columns(pd.DataFrame({"b": new_series}))["b"]
        if pb["kind"] != "numeric":
            return {"ok": False, "reason": "numeric vs categorical after transform"}
        return check_pair(pa, pb)
    la = {_category_key(v) for v in a_series.dropna().unique()}
    lb = {_category_key(v) for v in new_series.dropna().unique()}
    overlap = len(la & lb) / max(min(len(la), len(lb)), 1)
    reason = f"category overlap {overlap:.0%} (A: {sorted(la)[:6]}, transformed B: {sorted(lb)[:6]})"
    if lb - la:
        reason += f"; values not in A: {sorted(lb - la)[:4]}"
    return {"ok": overlap >= 0.5, "reason": reason}


def check_entry(df_a, df_b, e):
    """Post-transform check of one mapping entry (None if it has no B column)."""
    if not e["b_cols"]:
        return None
    try:
        a_series = df_a[e["a_col"]]
        t = Transform.model_validate(e["transform"])
        if t.apply_to == "A":
            return post_check(df_b[e["b_cols"][0]], apply_transform(e["transform"], df_a, [e["a_col"]]))
        new = apply_transform(e["transform"], df_b, e["b_cols"])
        res = post_check(a_series, new)
    except (RuleError, KeyError, ValueError, TypeError) as ex:
        return {"ok": False, "reason": f"transform failed: {ex}"}
    warnings = mapping_warnings(e["transform"])
    if warnings:
        res = {"ok": False, "reason": "code meaning changed: " + "; ".join(warnings) + " | " + res["reason"]}
    if t.kind == "derive":
        note = indicator_note(df_b, e["b_cols"], new)
        if note:
            res = {**res, "reason": res["reason"] + " | " + note}
    return res


# ----------------------------------------------------------------------------
# Mapping proposal
# ----------------------------------------------------------------------------
def propose_mapping(df_a, df_b, prof_a, prof_b, scores=None, llm=None, hi=HI, lo=LO, workers=4):
    """Propose a mapping from A columns to B columns.

    Parameters
    ----------
    prof_a, prof_b : dict
        Output of :func:`profile_columns`.
    scores : dict or None
        Jev scores ``{a_col: {b_col: score}}``; None if Jev is not available (then every A column goes to the LLM
        with all B columns as candidates).
    llm : callable or None
        ``llm(system, user) -> str``. If None, columns that need an LLM stay unmapped.
    hi, lo : float
        Routing thresholds.
    workers : int
        Parallel LLM calls.

    Returns
    -------
    dict
        ``{a_col: entry}``; entry has ``b_cols``, ``score``, ``decided_by`` ('Jev' / 'LLM' / 'none'),
        ``transform``, ``checks``, ``post_check``, ``same_definition``, ``definition_note``, ``reason``,
        ``candidates``, ``status``.
    """
    routes = _route_columns(df_a, df_b, prof_a, prof_b, scores, hi, lo)
    answers = _llm_answers(llm, routes, prof_a, prof_b, scores, df_a, df_b, workers)
    entries = {
        a: _proposal_entry(a, r, (scores or {}).get(a, {}), answers.get(a), df_b, hi, lo) for a, r in routes.items()
    }
    _assign_one_to_one(entries, lo)
    for e in entries.values():
        e["post_check"] = check_entry(df_a, df_b, e)
    return entries


def _route_columns(df_a, df_b, prof_a, prof_b, scores, hi, lo):
    routes = {}
    for a in df_a.columns:
        if scores is None:
            routes[a] = {"route": "llm", "top": None, "candidates": list(df_b.columns)}
            continue
        checks = {b: check_pair(prof_a[a], prof_b[b]) for b in df_b.columns}
        routes[a] = {**route_column(scores[a], checks, hi, lo), "checks": checks}
    return routes


def _llm_answers(llm, routes, prof_a, prof_b, scores, df_a, df_b, workers):
    """``{a_col: (answer or None, error text)}`` for the columns routed to the LLM (asked in parallel)."""

    def ask(a):
        if llm is None:
            return a, None, "no LLM available"
        try:
            answer = _ask_llm(
                llm, prof_a[a], routes[a]["candidates"], prof_b, (scores or {}).get(a), prof_a, df_a, df_b
            )
            return a, answer, ""
        except Exception as e:  # keep the column unmapped, show why
            return a, None, f"LLM failed: {str(e)[:200]}"

    llm_cols = [a for a, r in routes.items() if r["route"] == "llm"]
    with ThreadPoolExecutor(workers) as ex:
        answers = {a: (answer, err) for a, answer, err in ex.map(ask, llm_cols)}
    if llm is not None:
        answers.update(_make_groups_consistent(llm, answers, prof_a, prof_b, df_a, df_b))
    return answers


def _proposal_entry(a, route, scores_a, llm_answer, df_b, hi, lo):
    """Mapping entry of one A column from its route (and the LLM answer, if it was routed to the LLM)."""
    e = {
        "a_col": a,
        "b_cols": [],
        "score": None,
        "decided_by": "none",
        "transform": Transform().model_dump(),
        "checks": {"ok": True, "reason": ""},
        "post_check": None,
        "same_definition": "unsure",
        "definition_note": "",
        "reason": "",
        "candidates": {b: scores_a[b] for b in route["candidates"] if b in scores_a},
        "status": "proposed",
    }
    if route["route"] == "auto":
        top = route["top"]
        e.update(
            b_cols=[top],
            score=scores_a[top],
            decided_by="Jev",
            checks=route["checks"][top],
            reason="score >= %.2f and checks pass" % hi,
            same_definition="yes",
        )
    elif route["route"] == "llm":
        _apply_llm_answer(e, route, scores_a, *llm_answer, df_b)
    else:
        e["reason"] = "all Jev scores below %.2f" % lo
    return e


def _apply_llm_answer(e, route, scores_a, answer, error, df_b):
    if answer is None:
        e["reason"] = error
    elif not answer.match:
        e.update(reason=answer.reason or "LLM: no matching column", decided_by="LLM")
    elif any(b not in df_b.columns for b in answer.match):
        e["reason"] = f"LLM named unknown B columns {[b for b in answer.match if b not in df_b.columns]}; ignored"
    else:
        e.update(
            b_cols=answer.match,
            decided_by="LLM",
            transform=answer.transform.model_dump(),
            same_definition=answer.same_definition,
            definition_note=answer.definition_note,
            reason=answer.reason,
            score=max((scores_a.get(b, 0) for b in answer.match), default=None) if scores_a else None,
        )
        if "checks" in route and len(answer.match) == 1:
            e["checks"] = route["checks"][answer.match[0]]


def _ask_llm(llm, prof_a_col, candidates, prof_b, scores_a, prof_a, df_a, df_b):
    """Ask the LLM about one A column; if its transform does not run, ask once more with the error."""
    req = {
        "a_column": prof_a_col,
        "all_A_columns": {c: p.get("label", "") for c, p in prof_a.items()},
        "all_B_columns": {c: p.get("label", "") for c, p in prof_b.items()},
        "candidates": {b: {"profile": prof_b[b], "jev_score": (scores_a or {}).get(b)} for b in candidates},
    }
    for _ in range(2):
        answer = llm_json(llm, HARMONIZE_SYSTEM, json.dumps(req, default=str), HarmonizeAnswer)
        err = _transform_error(df_a, df_b, prof_a_col["name"], answer)
        if err is None:
            return answer
        req["your_previous_answer"] = answer.model_dump()
        req["error_when_applying_it"] = err
    return answer


def _transform_error(df_a, df_b, a, answer):
    """None if the answer's transform runs on the data, else the error text."""
    if not answer.match or any(b not in df_b.columns for b in answer.match):
        return None
    try:
        if answer.transform.apply_to == "A":
            apply_transform(answer.transform.model_dump(), df_a, [a])
        else:
            apply_transform(answer.transform.model_dump(), df_b, answer.match)
        return None
    except (RuleError, KeyError, ValueError, TypeError) as e:
        return str(e)


def _make_groups_consistent(llm, answers, prof_a, prof_b, df_a, df_b):
    """One LLM call per group of derived columns so that they use the same condition."""
    out = {}
    for group in _derived_groups(answers):
        b_cols = sorted({b for a in group for b in answers[a][0].match})
        req = {
            "a_columns": {a: prof_a[a] for a in group},
            "b_columns": {b: prof_b[b] for b in b_cols},
            "separate_answers": {a: answers[a][0].model_dump() for a in group},
        }
        try:
            res = llm_json(llm, HARMONIZE_GROUP_SYSTEM, json.dumps(req, default=str), GroupAnswer)
        except Exception as e:  # keep the separate answers, say why
            for a in group:
                out[a] = (answers[a][0], f"consistency check failed: {str(e)[:150]}")
            continue
        for a in group:
            answer = res.answers.get(a)
            if (
                answer is not None
                and answer.match
                and all(b in df_b.columns for b in answer.match)
                and _transform_error(df_a, df_b, a, answer) is None
            ):
                answer.reason = f"(made consistent with {', '.join(x for x in group if x != a)}) {answer.reason}"
                out[a] = (answer, "")
    return out


def _derived_groups(answers):
    """A columns whose derive transforms share B columns (e.g. time and event of one outcome)."""
    derived = {
        a: set(answer.match)
        for a, (answer, _) in answers.items()
        if answer is not None and answer.match and answer.transform.kind == "derive"
    }
    groups, seen = [], set()
    for a in derived:
        if a in seen:
            continue
        group, todo = {a}, [a]
        while todo:
            x = todo.pop()
            for y in derived:
                if y not in group and derived[x] & derived[y]:
                    group.add(y)
                    todo.append(y)
        seen |= group
        if len(group) > 1:
            groups.append(sorted(group))
    return groups


def _assign_one_to_one(entries, lo=LO):
    """Two A columns must not claim the same B column (Hungarian assignment on the scores).

    Derived mappings (several B columns) are exempt. A column that loses its B column falls back to its next-best
    candidate (identity transform, flagged) or to no match.
    """
    single = [a for a, e in entries.items() if len(e["b_cols"]) == 1]
    claimed = [e["b_cols"][0] for a, e in entries.items() if len(e["b_cols"]) == 1]
    if len(set(claimed)) == len(claimed):
        return
    b_cols = sorted({b for a in single for b in [entries[a]["b_cols"][0]] + list(entries[a]["candidates"])})
    n = len(single)
    value = np.full((n, len(b_cols) + n), -1e6)
    for i, a in enumerate(single):
        e = entries[a]
        for b, s in e["candidates"].items():
            if s >= lo:
                value[i, b_cols.index(b)] = s
        value[i, b_cols.index(e["b_cols"][0])] = (e["score"] or 0.5) + 1.0  # the decided match is preferred
        value[i, len(b_cols) + i] = 0.0  # "no match"
    rows, cols = linear_sum_assignment(value, maximize=True)
    for i, j in zip(rows, cols):
        e = entries[single[i]]
        new = b_cols[j] if j < len(b_cols) else None
        if new == e["b_cols"][0]:
            continue
        e["reason"] = (
            f"{e['b_cols'][0]} is better explained by another A column (one-to-one assignment). " + e["reason"]
        )
        if new is None:
            e.update(b_cols=[], decided_by="none", score=None, transform=Transform().model_dump())
        else:
            e.update(
                b_cols=[new],
                score=e["candidates"].get(new),
                decided_by="Jev",
                transform=Transform().model_dump(),
                same_definition="unsure",
                definition_note="reassigned by the one-to-one rule: check the transform",
            )


def cached_llm(cache):
    """Fake ``llm(system, user)`` that returns stored answers by A column name (demo / tests)."""

    def llm(system, user):
        name = json.loads(user)["a_column"]["name"]
        if name not in cache:
            raise KeyError(f"no cached LLM answer for column '{name}'")
        return json.dumps(cache[name])

    return llm


# ----------------------------------------------------------------------------
# Approval, coverage and application
# ----------------------------------------------------------------------------
def default_decision(e):
    """Initial decision in the approval table: Jev auto-accepts with passing checks; LLM proposals wait."""
    if not e["b_cols"]:
        return "reject"
    return "approve" if e["decided_by"] == "Jev" and e["checks"]["ok"] else "pending"


def approve_all(mapping):
    """Mark every proposal that has a B column as approved (returns a new dict)."""
    return {a: {**e, "status": "approved" if e["b_cols"] else "rejected"} for a, e in mapping.items()}


def coverage(mapping, a_cols):
    """Coverage text: '8 of 10 A columns mapped, missing: ...' (approved mappings only)."""
    ok = [a for a in a_cols if mapping.get(a, {}).get("status") == "approved" and mapping[a]["b_cols"]]
    miss = [a for a in a_cols if a not in ok]
    return f"{len(ok)} of {len(a_cols)} A columns mapped" + (f", missing: {', '.join(miss)}" if miss else "")


def harmonize_frames(df_a, df_b, mapping):
    """Convert B into A's schema using the approved mappings only.

    Parameters
    ----------
    mapping : dict
        From :func:`propose_mapping` (after human approval).

    Returns
    -------
    a_h : pandas.DataFrame
        A, with A-side transforms (e.g. tsize binned to B's categories) applied.
    b_h : pandas.DataFrame
        B in A's column names (approved columns only), same index as ``df_b``.
    info : dict
        ``unmapped_a`` (A columns without approved mapping), ``dropped_b`` (B columns not used by any approved
        mapping), ``post_checks`` and ``errors`` (approved transforms that failed to run; those columns are left
        out).
    """
    a_h, b_h, used, checks, errors = df_a.copy(), pd.DataFrame(index=df_b.index), set(), {}, {}
    for a, e in mapping.items():
        if e.get("status") != "approved" or not e["b_cols"]:
            continue
        try:
            t = Transform.model_validate(e["transform"])
            if t.apply_to == "A":
                a_new, b_new = apply_transform(e["transform"], df_a, [a]), df_b[e["b_cols"][0]]
            else:
                a_new, b_new = a_h[a], apply_transform(e["transform"], df_b, e["b_cols"])
        except (RuleError, KeyError, ValueError, TypeError) as ex:
            errors[a] = str(ex)
            continue
        a_h[a], b_h[a] = a_new, b_new
        used.update(e["b_cols"])
        checks[a] = post_check(a_h[a], b_h[a])
    unmapped = [a for a in df_a.columns if a not in b_h.columns]
    return (
        a_h,
        b_h,
        {
            "unmapped_a": unmapped,
            "dropped_b": [c for c in df_b.columns if c not in used],
            "post_checks": checks,
            "errors": errors,
        },
    )


def save_mapping(mapping, path):
    """Write the mapping (with approvals) as JSON."""
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2, default=str)


def load_mapping(path):
    """Read a mapping saved by :func:`save_mapping`."""
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Evaluation against the known truth
# ----------------------------------------------------------------------------
def evaluate_mapping(proposed, truth):
    """Score a proposed mapping against ``true_mapping.json`` (never used as input to the harmonizer).

    Parameters
    ----------
    proposed : dict
        Mapping from :func:`propose_mapping` (entries without B columns or rejected are ignored).
    truth : dict or str
        The truth, or the path to the JSON file.

    Returns
    -------
    dict
        ``precision`` (correct / proposed), ``recall`` (correct / true), counts, ``per_route`` (accuracy for 'Jev'
        auto-accepted vs 'LLM') and ``rows``. A derived variable counts as correct if it uses at least the true B
        columns.
    """
    if isinstance(truth, str):
        truth = json.load(open(truth))
    true_cols = _true_b_columns(truth)
    rows = []
    for a, e in proposed.items():
        if not e["b_cols"] or e.get("status") == "rejected":
            continue
        t, p = true_cols.get(a), set(e["b_cols"])
        ok = t is not None and (p == t or (len(t) > 1 and t <= p))
        rows.append(
            {
                "a_col": a,
                "proposed": sorted(p),
                "truth": sorted(t) if t else None,
                "correct": bool(ok),
                "decided_by": e["decided_by"],
            }
        )
    df = pd.DataFrame(rows, columns=["a_col", "proposed", "truth", "correct", "decided_by"])
    n_ok = int(df.correct.sum()) if len(df) else 0
    per_route = (
        {r: {"n": int(len(g)), "accuracy": float(g.correct.mean())} for r, g in df.groupby("decided_by")}
        if len(df)
        else {}
    )
    return {
        "precision": n_ok / len(df) if len(df) else float("nan"),
        "recall": n_ok / len(true_cols) if true_cols else float("nan"),
        "n_proposed": len(df),
        "n_correct": n_ok,
        "n_true": len(true_cols),
        "per_route": per_route,
        "rows": df,
    }


def _true_b_columns(truth):
    """``{a_col: set of B columns}`` from the truth file ("rtime + dtime (...)" -> {"rtime", "dtime"})."""
    out = {}
    for a, v in truth.items():
        if a.startswith("_"):
            continue
        out[a] = {s.strip() for s in re.sub(r"\(.*?\)", "", v["match"]).split("+") if s.strip()}
    return out
