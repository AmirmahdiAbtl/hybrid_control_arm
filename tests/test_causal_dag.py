"""DAG utilities of src.causal and the identification checks of gate G2. Pure Python, no data."""

import inspect

from src import causal as C
from src import survival
from src.gated import protocol

# Protocol DAG G_full: (parent, child, kind, assumption id or None, basis).
# Edges with an assumption id are removed in the working graph G_work.
DAG_EDGES = [
    ("age", "menostat", "causal", None, "menopause depends on age"),
    ("age", "Y", "causal", None, "age < 40 worse; non-cancer death rises with age"),
    ("menostat", "Y", "causal", None, "prognostic in Rotterdam, weak in GBSG"),
    ("menostat", "estrec", "causal", None, "premenopausal oestradiol occupies receptors"),
    ("menostat", "progrec", "causal", None, "PgR is oestrogen-induced"),
    ("U_bio", "tsize", "causal", None, "shared tumour biology"),
    ("U_bio", "pnodes", "causal", None, "shared tumour biology"),
    ("U_bio", "grade_true", "causal", None, "shared tumour biology"),
    ("U_bio", "estrec", "causal", None, "shared tumour biology"),
    ("U_bio", "progrec", "causal", None, "shared tumour biology"),
    ("U_bio", "Y", "causal", None, "HER2, proliferation, LVI (unmeasured)"),
    ("tsize", "pnodes", "causal", None, "nodal spread increases with size"),
    ("tsize", "Y", "causal", None, "prognostic factor"),
    ("pnodes", "Y", "causal", None, "prognostic factor"),
    ("estrec", "Y", "causal", None, "prognostic factor"),
    ("progrec", "Y", "causal", None, "prognostic factor"),
    ("grade_true", "Y", "causal", None, "prognostic factor"),
    ("grade_true", "tgrade", "measurement", None, "recorded grade measures true grade"),
    ("U_fit", "Y", "causal", None, "fitness acts through the death component of RFS"),
    ("horTh", "Y", "causal", None, "target effect"),
    ("age", "horTh", "assignment", None, "age influenced tamoxifen use (Sauerbrei 1999)"),
    ("menostat", "horTh", "assignment", None, "Dec 1986 amendment"),
    ("estrec", "horTh", "assignment", None, "receptor-driven preference"),
    ("progrec", "horTh", "assignment", None, "receptor-driven preference"),
    ("tsize", "horTh", "assignment", None, "weak"),
    ("pnodes", "horTh", "assignment", None, "weak"),
    ("tgrade", "horTh", "assignment", None, "weak; W3 matters for E only"),
    ("Period", "horTh", "assignment", None, "amendment acts by recruitment date"),
    ("U_choice", "horTh", "assignment", None, "preference / consent / centre"),
    ("U_choice", "Y", "threat", "W1", "preference or centre may be prognostic"),
    ("U_fit", "horTh", "threat", "W1", "tamoxifen withheld from frail patients"),
    ("Period", "Y", "threat", "W2", "calendar trend in outcome"),
    ("Period", "C", "censoring", None, "staggered entry"),
    ("S", "C", "censoring", None, "follow-up differs by source"),
    ("S", "SEL", "selection", None, "selection rules differ by source"),
    ("age", "SEL", "selection", None, "eligibility / B treatment selection"),
    ("menostat", "SEL", "selection", None, "B gave chemo mainly to premenopausal patients"),
    ("tsize", "SEL", "selection", None, "selection on observed size"),
    ("pnodes", "SEL", "selection", None, "nodes >= 1; chemo by nodal status"),
    ("estrec", "SEL", "selection", None, "hormones withheld by receptor status"),
    ("progrec", "SEL", "selection", None, "hormones withheld by receptor status"),
    ("S", "horTh", "design", None, "eligible B patients untreated by construction"),
    ("S", "Period", "source", None, "B 1979-93 vs A 1984-89"),
    ("S", "tgrade", "measurement", None, "different grading protocols (Royston & Altman 2013)"),
    ("S", "V_chemo", "source", None, "control chemotherapy differs"),
    ("S", "U_care", "source", None, "country, surgery/RT, follow-up, outcome definition"),
    ("S", "estrec", "measurement", "A9r", "receptor assays differ (DCC vs LBA/EIA)"),
    ("S", "progrec", "measurement", "A9r", "receptor assays differ"),
    ("S", "tsize", "measurement", "A9s", "mm vs groups, possibly clinical"),
    ("grade_true", "SEL", "latent_selection", "T1", "B selection on true grade"),
    ("V_chemo", "Y", "threat", "T2", "control-treatment versions differ"),
    ("U_care", "Y", "threat", "T3", "care / ascertainment / outcome definition differ"),
    ("U_fit", "SEL", "latent_selection", "T4", "B selection on unrecorded fitness"),
    ("U_bio", "SEL", "latent_selection", "T5", "B selection on unmeasured biology"),
]
ASSUMPTIONS = {k: k for k in ("W1", "W2", "A9r", "A9s", "T1", "T2", "T3", "T4", "T5", "W3", "C1")}
Z_T = ["age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec"]
Z_S = ["age", "menostat", "tsize", "pnodes", "progrec", "estrec"]
LATENT = ["U_bio", "grade_true", "U_choice", "U_fit", "Period", "V_chemo", "U_care"]

# Constants of src.gated.protocol that src.causal and src.survival repeat as argument defaults (the protocol
# imports src.causal, so not the other way round): (function, argument, protocol constant or GATES key, value).
CAUSAL_DEFAULTS = [
    ("gate_frames", "age_max", "AGE_MAX", 65),
    ("gate_frames", "landmark", "LANDMARK_DAYS", 91.0),
    ("gate_frames", "treatment", "TREATMENT", "horTh"),
    ("gate_frames", "control_value", "CONTROL_VALUE", "no"),
    ("gate_frames", "receptor_cuts", "RECEPTOR_CUTS", (10.0, 100.0)),
    ("make_folds", "k", "K_FOLDS", 5),
    ("make_folds", "seed", "SEED", 0),
    ("source_support", "k", "K_FOLDS", 5),
    ("source_support", "seed", "SEED", 0),
    ("source_support", "cut", "G6_ratio_cut", 20.0),
    ("overlap_summary", "bounds", "G4_ps_bounds", (0.05, 0.95)),
    ("entropy_balance", "tol", "G4H_imbalance_max", 1e-6),
    ("min_detectable_margin", "alpha", "G8_alpha", 0.05),
    ("pseudo_by_strata", "min_n", "G9_min_n", 25),
    ("pseudo_by_strata", "min_at_risk", "G9_min_at_risk", 10),
    ("horizon_check", "g_min", "G9_G_min", 0.70),
    ("horizon_check", "min_at_risk", "G9_min_at_risk", 10),
    ("horizon_check", "tol", "G9_identity_tol", 1e-8),
    ("censoring_G", "min_events_per_coef", "G9_cox_min_events_per_coef", 5),
    ("ipcw_outcome", "floor", "G9_ipcw_G_floor", 0.10),
    ("benchmark_bounding_factors", "min_cell", "G11_benchmark_min_cell", 20),
    ("simultaneous_band", "draws", "cate_band_draws", 2000),
]


def _default(func, arg):
    fn = getattr(C, func, None) or getattr(survival, func)
    return inspect.signature(fn).parameters[arg].default


def _report(**kw):
    args = {"used_trial": C.TRIAL_COLUMNS.keys(), "used_source": C.SOURCE_COLUMNS.keys(), **kw}
    return C.dag_report(DAG_EDGES, ASSUMPTIONS, Z_T, Z_S, **args)


# Textbook cases
def test_d_separation_textbook_cases():
    chain = [("a", "b"), ("b", "c")]
    assert not C.d_separated(chain, {"a"}, {"c"}, set()) and C.d_separated(chain, {"a"}, {"c"}, {"b"})
    fork = [("b", "a"), ("b", "c")]
    assert not C.d_separated(fork, {"a"}, {"c"}, set()) and C.d_separated(fork, {"a"}, {"c"}, {"b"})
    collider = [("a", "c"), ("b", "c"), ("c", "d")]
    assert C.d_separated(collider, {"a"}, {"b"}, set())
    assert not C.d_separated(collider, {"a"}, {"b"}, {"c"})
    assert not C.d_separated(collider, {"a"}, {"b"}, {"d"})  # conditioning on a descendant of the collider


def test_backdoor_cases():
    fork = [("Z", "T"), ("Z", "Y"), ("T", "Y")]
    assert C.is_backdoor_set(fork, "T", "Y", {"Z"}) and not C.is_backdoor_set(fork, "T", "Y", set())
    mediator = [("T", "M"), ("M", "Y")]
    assert C.is_backdoor_set(mediator, "T", "Y", set()) and not C.is_backdoor_set(mediator, "T", "Y", {"M"})
    m_bias = [("U1", "T"), ("U1", "M"), ("U2", "M"), ("U2", "Y"), ("T", "Y")]
    assert C.is_backdoor_set(m_bias, "T", "Y", set())
    assert not C.is_backdoor_set(m_bias, "T", "Y", {"M"})
    assert C.is_backdoor_set(m_bias, "T", "Y", {"M", "U1"})


def test_graph_helpers():
    par = C.dag_parents([("a", "b", "causal", None, ""), ("b", "c")])
    assert par == {"a": set(), "b": {"a"}, "c": {"b"}}
    assert C.ancestors(par, {"c"}) == {"a", "b"} and C.descendants(par, {"a"}) == {"b", "c"}
    assert C.find_cycle(par) == []
    cyc = C.find_cycle(C.dag_parents([("a", "b"), ("b", "c"), ("c", "a")]))
    assert cyc[0] == cyc[-1] and set(cyc) == {"a", "b", "c"}
    assert C.minimal_sets(lambda z: {"x"} <= z or {"y", "w"} <= z, ["x", "y", "w"]) == [["x"], ["y", "w"]]


# The protocol DAG
def test_protocol_dag_identification_checks():
    assert len(DAG_EDGES) == 54 and len(C.work_graph(DAG_EDGES)) == 43
    assert len(C.trial_graph(DAG_EDGES)) == 54 - sum("S" in e[:2] or "SEL" in e[:2] for e in DAG_EDGES)
    rep = _report()
    assert (rep["n_edges_full"], rep["n_edges_work"]) == (54, 43)
    assert rep["acyclic_full"] and rep["acyclic_work"]
    assert rep["zt_valid_work"] and not rep["zt_valid_full"]
    assert rep["zs_adm_work"] and not rep["zs_adm_full"]
    assert rep["zs_plus_grade_adm"] is False  # collider S -> tgrade <- grade_true -> Y
    assert rep["zs_valid_trial"] is False  # W3 path horTh <- tgrade <- grade_true -> Y
    assert [set(s) for s in rep["minimal_trial_sets"]] == [set(Z_T)]
    assert [set(s) for s in rep["minimal_s_sets"]] == [set(Z_S)]
    assert rep["censoring_ok_work"] and not rep["censoring_ok_full"]
    assert rep["features_ok_trial"] and rep["features_ok_source"] and rep["probes_ok"]


def test_protocol_dag_threat_report():
    threats = _report()["threats"]
    assert len(threats) == 11
    for t in threats:
        if t["assumption"] == "W1":
            assert (t["breaks_zt"], t["breaks_zs"]) == (True, False), t
        elif t["assumption"] == "W2":
            assert (t["breaks_zt"], t["breaks_zs"]) == (True, True), t
        else:  # A9r, A9s, T1-T5
            assert (t["breaks_zt"], t["breaks_zs"]) == (False, True), t
    assert {t["assumption"] for t in threats} == {"W1", "W2", "A9r", "A9s", "T1", "T2", "T3", "T4", "T5"}


def test_cycle_and_unnamed_assumption_fail_the_report():
    rep = C.dag_report(DAG_EDGES + [("Y", "age", "causal", None, "")], ASSUMPTIONS, Z_T, Z_S, Z_T, Z_S)
    assert not rep["acyclic_full"] and not rep["acyclic_work"] and rep["cycle_full"]
    rep = C.dag_report(DAG_EDGES + [("U_care", "horTh", "threat", "Z9", "")], ASSUMPTIONS, Z_T, Z_S, Z_T, Z_S)
    assert not rep["probes_ok"]


# Transportability and feature checks
def test_selection_on_observables_vs_measurement_shift():
    selection = [("U", "X"), ("U", "Y"), ("X", "SEL"), ("S", "SEL")]
    assert C.is_s_admissible(selection, {"X"})
    measurement = [("U", "X"), ("U", "Y"), ("S", "X")]
    assert not C.is_s_admissible(measurement, {"X"})


def test_grade_in_a_cross_source_model_fails_g2_and_dot_shows_every_node():
    rep = _report(used_trial=Z_T, used_source=Z_S + ["tgrade"])
    assert rep["features_ok_trial"] and not rep["features_ok_source"]
    dot = C.dag_dot(DAG_EDGES, LATENT)
    nodes = {e[0] for e in DAG_EDGES} | {e[1] for e in DAG_EDGES}
    assert all(f'"{v}"' in dot for v in nodes)
    assert dot.startswith("digraph") and dot.count("[style=dashed, color=red") == 11 and 'label="A9r"' in dot
    assert '"S" [shape=diamond]' in dot and '"SEL" [shape=diamond]' in dot
    assert '"U_bio" [shape=ellipse, style=dashed]' in dot and '"age" [shape=box]' in dot


def test_causal_defaults_repeat_the_protocol_constants():
    for func, arg, _, value in CAUSAL_DEFAULTS:
        assert _default(func, arg) == value, (func, arg)


def test_protocol_module_matches_this_dag_and_the_causal_defaults():
    assert [tuple(e[:4]) for e in protocol.DAG_EDGES] == [e[:4] for e in DAG_EDGES]
    assert list(protocol.Z_T) == Z_T and list(protocol.Z_S) == Z_S and set(protocol.ASSUMPTIONS) == set(ASSUMPTIONS)
    # A single copy of the node -> column maps: the protocol imports them from src.causal.
    for name in ("TRIAL_COLUMNS", "SOURCE_COLUMNS"):
        assert getattr(protocol, name, getattr(C, name)) is getattr(C, name), name
    for func, arg, name, value in CAUSAL_DEFAULTS:
        const = protocol.GATES[name] if name in protocol.GATES else getattr(protocol, name)
        assert const == value == _default(func, arg), (func, arg, name)
