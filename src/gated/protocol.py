from src.causal.data import SOURCE_COLUMNS, TRIAL_COLUMNS

PROTOCOL_VERSION = "1.0 (2026-10-05)"
PRIOR_OUTCOME_EXPOSURE = True  # outcomes were seen during design, so the tier is capped at SUPPORTIVE

TREATMENT, CONTROL_VALUE = "horTh", "no"
AGE_MAX = 65
LANDMARK_DAYS = 91.0
DAYS = 365.25
HORIZON_YEARS = {"primary": 2.0, "secondary": 3.0, "supplementary": 5.0}
N_REAL = 60  # kept trial controls in hybrid mode (frozen; the sidebar value is ignored)
SEED = 0
K_FOLDS = 5
RECEPTOR_CUTS = (10.0, 100.0)  # fmol/mg protein: <10, 10-99, >=100
MAX_TABPFN_FITS = 60

# Covariate sets as DAG node names. The columns fed to the models are TRIAL_COLUMNS / SOURCE_COLUMNS
# (src.causal.data); G2 checks that their keys equal Z_T / Z_S.
Z_T = ["age", "menostat", "tsize", "tgrade", "pnodes", "progrec", "estrec"]  # within-trial set
Z_S = ["age", "menostat", "tsize", "pnodes", "progrec", "estrec"]  # cross-source set (no grade)
LATENT = ["U_bio", "grade_true", "U_choice", "U_fit", "Period", "V_chemo", "U_care"]

GATES = {
    "G4_ps_bounds": (0.05, 0.95),
    "G4_share_outside_max": 0.05,
    "G4_ess_ratio_min": 0.5,
    "G4_weight_share_max": 0.05,
    "G4H_imbalance_max": 1e-6,
    "G4H_ess_min_abs": 20.0,
    "G4H_ess_min_frac": 1 / 3,
    "G4H_weight_share_max": 0.15,
    "G5_smd_max": 0.10,
    "G5_smd_conditional_max": 0.25,
    "G5_vr_range": (0.5, 2.0),
    "G6_ratio_cut": 20.0,
    "G6_share_unsupported_max": 0.10,
    "G6_ess_min": 50.0,
    "G6_weight_share_max": 0.10,
    "G6_w3_smd_max": 0.10,
    "G7_se_mult": 2.0,
    "G8_margin": 0.05,
    "G8_alpha": 0.05,
    "G9_G_min": 0.70,
    "G9_min_n": 25,
    "G9_min_at_risk": 10,
    "G9_identity_tol": 1e-8,
    "G9_ipcw_warn_se": 0.25,
    "G9_ipcw_G_floor": 0.10,
    "G9_cox_min_events_per_coef": 5,
    "G11_benchmark_min_cell": 20,
    "G11_named_rr": (2.0, 1.5),
    "G11_alpha": 0.05,
    "E_delta_max": 0.10,
    "max_design_actions": 2,
    "validation_reps": 200,
    "placebo_reps": 200,
    "placebo_n_treated": 100,
    "draw_stability_reps": 200,
    "cate_band_draws": 2000,
}

DAG_EDGES = [  # (parent, child, kind, assumption removed in G_work or None, basis)
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
]  # 54 edges; 11 carry an assumption id -> G_work has 43 edges

ASSUMPTIONS = {
    "W1": (
        "No unmeasured confounding of tamoxifen use by preference, consent, centre or fitness",
        "R,H,E",
        "G11: E-value vs benchmark bounding factors (untestable)",
    ),
    "W2": (
        "No calendar-period effect on outcome within GBSG-2 beyond Z_t",
        "R,H,E",
        "untestable (no entry date); E-value",
    ),
    "A9r": ("Receptor categories mean the same in both sources", "E", "sensitivity X2 (continuous receptors)"),
    "A9s": ("Tumour size groups mean the same in both sources", "E", "untestable"),
    "T1": ("Rotterdam controls not selected on true grade beyond Z_s", "E", "negative-control beta; tipping point"),
    "T2": (
        "Control chemotherapy versions exchangeable for Y(0) given Z_s",
        "E",
        "negative-control beta; tipping point",
    ),
    "T3": (
        "No source differences in care, follow-up or outcome definition affecting Y given Z_s",
        "E",
        "negative-control beta; tipping point",
    ),
    "T4": ("Rotterdam controls not selected on unrecorded fitness", "E", "negative-control beta; tipping point"),
    "T5": ("Rotterdam controls not selected on unmeasured biology", "E", "negative-control beta; tipping point"),
    "W3": (
        "Given Z_s, recorded grade does not drive tamoxifen use (edge kept; needed by E only)",
        "E",
        "G6: residual grade SMD after Z_s-only ATT weighting <= 0.10",
    ),
    "C1": (
        "Censoring independent of event time within source x arm x menopausal strata",
        "R,H,E",
        "G9 reverse-Cox report; sensitivity S4 (IPCW)",
    ),
}

# Target trial protocol and its emulation: 9 components x 4 cells. The emulation and deviation cells of "estimand" and
# "analysis" are short summaries, because G1 needs every cell filled.
TARGET_TRIAL = {
    "eligibility": {
        "target": "Women <= 65 y; primary operable invasive breast cancer T1a-3a, M0; >= 1 positive axillary node "
        "(>= 6 examined); modified radical mastectomy; Karnofsky >= 60; adjuvant CMF started "
        "perioperatively with 3 cycles completed; event-free and under follow-up at time zero",
        "emulation_A": "Protocol criteria. Applied in code: age <= 65 (removes 66, including 37 treated) and the "
        "landmark (time > 91 d)",
        "emulation_B": "Stage-2 rules: nodes >= 1, chemo == 1, hormon == 0. Applied in code: age <= 65 and landmark. "
        "T-stage, M0, Karnofsky, surgery type and nodes examined are not checkable.",
        "deviation": "Breast-conserving surgery + RT in B: fewer locoregional events, Y0 too high -> ATT toward null. "
        "Under-staging (nodes examined unknown): Y0 too low -> away from null. Selection of fitter "
        "patients for chemo in B: Y0 too high -> toward null. These matter only for E (exploratory) and "
        "for the prognostic score's efficiency, not for H validity.",
    },
    "treatment": {
        "target": "(1) Tamoxifen 30 mg/d for 2 y from time zero + planned CMF; (2) no adjuvant endocrine therapy + "
        "planned CMF",
        "emulation_A": "horTh yes/no. CMF 3 vs 6 cycles is a randomized co-intervention that is not recorded "
        "(RR 0.90-0.95)",
        "emulation_B": "Strategy (2) only. 'chemo' means any regimen, mainly CMF.",
        "deviation": "Treatment-version mismatch for controls (assumption T2). Direction unknown; magnitude small.",
    },
    "assignment": {
        "target": "Randomized 1:1, stratified by menopausal status",
        "emulation_A": "About 2/3 randomized and 1/3 by preference (unidentifiable), plus the Dec 1986 amendment. "
        "Emulated by conditional exchangeability given Z_t.",
        "emulation_B": "Physician choice of chemo without hormones; B patients are untreated by construction",
        "deviation": "Residual confounding by preference, centre, fitness (W1) and period (W2): sensitivity analysis "
        "in section 9",
    },
    "time_zero": {
        "target": "End of CMF cycle 3 = start of tamoxifen (about day 84-91 after surgery)",
        "emulation_A": "Clock starts at mastectomy. Landmark L = 91 d: keep patients with time > 91.",
        "emulation_B": "Clock starts at surgery. Same landmark.",
        "deviation": "Without the landmark: immortal time favouring tamoxifen (4 control vs 0 treated events by "
        "day 120). Tiny here, but fixed by design.",
    },
    "follow_up": {
        "target": "From time zero to the first RFS event, loss to follow-up, administrative end, or H",
        "emulation_A": "Heavy administrative censoring after 3 y (staggered entry)",
        "emulation_B": "Almost no censoring before 5 y",
        "deviation": "Handled by pseudo-observations (section 6)",
    },
    "outcome": {
        "target": "Y = 1{event-free at H = 730.5 d after surgery}, among those event-free at L",
        "emulation_A": "GBSG RFS event set",
        "emulation_B": "Recurrence or death (conservative rule). Contralateral and second primaries probably not "
        "counted [not verified].",
        "deviation": "Y0 in B probably too high by about 1-2 points at 2 y (expert judgement) -> toward null. "
        "Affects E only.",
    },
    "contrast": {
        "target": "Intention-to-treat analogue (assignment or initiation as recorded)",
        "emulation_A": "horTh as recorded. Whether it is allocated or received treatment is [not verified]. "
        "Compliance 93% in the randomized part [verified].",
        "emulation_B": "Not applicable",
        "deviation": "Close to ITT given high compliance",
    },
    "estimand": {
        "target": "ATT risk difference at 2 y (section 5)",
        "emulation_A": "ATT among the eligible treated GBSG-2 patients: scope R (all trial controls) and scope H "
        "(kept controls) target the same quantity (sections 4, 5.3)",
        "emulation_B": "No estimand of its own: Rotterdam enters H only through the prognostic score m_B; scope E "
        "(external-only transport) is exploratory (section 4)",
        "deviation": "None for R and H; ATO and trimming are forbidden in H and E because their population would "
        "depend on n_real (section 4)",
    },
    "analysis": {
        "target": "Sections 6-9",
        "emulation_A": "Kaplan-Meier pseudo-observations (section 6); AIPW in R and the robust hybrid in H (section 7)",
        "emulation_B": "Prognostic score m_B fitted on Rotterdam only (section 7.1); exploratory transport E "
        "(section 7.4)",
        "deviation": "Non-randomized comparison: unmeasured confounding is bounded with E-values against "
        "measured-covariate benchmarks (section 9)",
    },
}

ELIGIBILITY_STATUSES = ("both", "approximate", "not checkable in B", "not checkable")

ELIGIBILITY = [
    {"id": "E1", "criterion": "Age <= 65", "A": "applied in code", "B": "applied in code", "status": "both"},
    {
        "id": "E2",
        "criterion": ">= 1 positive node",
        "A": "trial eligibility",
        "B": "rule nodes >= 1 (I2)",
        "status": "both",
    },
    {
        "id": "E3",
        "criterion": "Adjuvant CMF",
        "A": "trial protocol",
        "B": "rule chemo == 1 (I3; regimen not checkable)",
        "status": "approximate",
    },
    {
        "id": "E4",
        "criterion": "No adjuvant hormonal therapy (control strategy)",
        "A": "horTh == 'no'",
        "B": "rule hormon == 0 (C1)",
        "status": "both",
    },
    {
        "id": "E5",
        "criterion": "Event-free and under follow-up at day 91",
        "A": "applied in code",
        "B": "applied in code",
        "status": "both",
    },
    {
        "id": "E6",
        "criterion": "Operable T1a-3a, M0",
        "A": "protocol",
        "B": "not recorded (tumour bank excludes M1)",
        "status": "not checkable in B",
    },
    {
        "id": "E7",
        "criterion": "Mastectomy, >= 6 nodes examined",
        "A": "protocol",
        "B": "not recorded",
        "status": "not checkable in B",
    },
    {"id": "E8", "criterion": "Karnofsky >= 60", "A": "protocol", "B": "not recorded", "status": "not checkable"},
    {
        "id": "E9",
        "criterion": "Informed consent",
        "A": "protocol",
        "B": "not recorded (no bias)",
        "status": "not checkable",
    },
]

# ICH E9(R1) attributes of the primary estimand; the other estimands differ only in their population.
_ATT = {
    "population": "Women aged <= 65 with operable node-positive breast cancer on adjuvant CMF after mastectomy, "
    "event-free and under follow-up 91 days after surgery, who received tamoxifen in GBSG-2",
    "treatment": "Tamoxifen 30 mg/d for 2 years, started after CMF cycle 3, added to CMF (3 or 6 cycles), vs CMF "
    "without adjuvant hormonal therapy",
    "variable": "Recurrence-free status 730.5 days after surgery (GBSG RFS event set)",
    "intercurrent": "Death without recurrence: composite (part of the event). Tamoxifen discontinuation, hormonal "
    "therapy started in controls (not recorded) and other anticancer therapy: treatment policy.",
    "missing": "Loss to follow-up and administrative censoring are missing data, not intercurrent events. They are "
    "handled with pseudo-observations under independent censoring within strata (C1); IPCW is a "
    "sensitivity analysis.",
    "summary": "Risk difference S1 - S0 in event-free probability (percentage points) with 95% CI. Secondary: event "
    "risk ratio RR = (1 - S1)/(1 - S0), used for E-values. No hazard ratio: it is non-collapsible and "
    "needs proportional hazards.",
}

_NO_TEST = "Risk difference S1 - S0 in event-free probability (percentage points) with 95% CI; no hypothesis test"

ESTIMANDS = {
    "ATT": _ATT,
    "ATE": {
        **_ATT,
        "population": "All eligible GBSG-2 patients (same eligibility as the ATT population), treated or not",
        "summary": _NO_TEST,
    },
    "ATC": {**_ATT, "population": "Eligible GBSG-2 patients who did not receive tamoxifen", "summary": _NO_TEST},
    "ATO": {
        **_ATT,
        "population": "Overlap population: eligible GBSG-2 patients weighted by e(1 - e) (clinical "
        "equipoise: older premenopausal and younger postmenopausal women), reported with "
        "an h-weighted baseline table",
        "summary": _NO_TEST,
    },
    "CATE": {
        **_ATT,
        "population": "Pre-specified subgroups of eligible GBSG-2 patients: ER >= 10 vs < 10 fmol/mg, "
        "menopausal status, nodes 1-3 vs >= 4 (section 8)",
        "summary": "Subgroup risk differences GATE_g and CATT_g with simultaneous 95% bands; exploratory "
        "(differences between subgroups are not evidence of effect modification)",
    },
}

ESTIMAND_FIELDS = ["population", "treatment", "variable", "intercurrent", "missing", "summary"]

ESTIMAND_ROLES = {
    "internal": {
        "R": {"ATT": "primary", "ATE": "secondary", "ATC": "secondary", "ATO": "secondary", "CATE": "exploratory"}
    },
    "hybrid": {
        "H": {"ATT": "primary", "ATC": "secondary"},
        "E": {"ATT": "exploratory"},
        "R": {"ATT": "reference", "ATE": "reference", "ATC": "reference", "ATO": "reference", "CATE": "exploratory"},
    },
}

# Pre-specified sensitivity analyses, all logistic/linear (NOT TabPFN).
SENSITIVITY = [
    {"id": "S1", "scopes": "R", "change": "L = 0", "in_sign_rule": True},
    {"id": "S2", "scopes": "R", "change": "L = 182", "in_sign_rule": True},
    {"id": "S3", "scopes": "R", "change": "all ages", "in_sign_rule": True},
    {"id": "S4", "scopes": "R,H", "change": "IPCW (Cox censoring model)", "in_sign_rule": True},
    {"id": "S5", "scopes": "R,H", "change": "linear nuisances / linear prognostic score", "in_sign_rule": True},
    {"id": "S6", "scopes": "R,H", "change": "strata source x arm", "in_sign_rule": True},
    {"id": "X1", "scopes": "H,E", "change": "B surgery years 1984-89", "in_sign_rule": False},
    {"id": "X2", "scopes": "E", "change": "continuous log receptors in Z_s", "in_sign_rule": False},
    {"id": "X3", "scopes": "E", "change": "grade III in Z_s (descriptive, collider-biased)", "in_sign_rule": False},
    {"id": "X4", "scopes": "E", "change": "premenopausal treated only", "in_sign_rule": False},
]

SENS_LANDMARKS = {"S1": 0.0, "S2": 182.0}  # days

B_YEARS_X1 = (1984, 1989)  # Pocock's "same period" (sensitivity X1)

SUBGROUPS = [
    ("ER", "estrec >= 10", "estrec < 10"),
    ("menostat", "menostat == 'Post'", "menostat == 'Pre'"),
    ("nodes", "pnodes >= 4", "pnodes <= 3"),
]

# Mandatory footer of every result.
FOOTER = (
    "Evidence type: non-randomized comparison. GBSG-2 is a partially randomized comprehensive cohort; tamoxifen "
    "use was partly chosen. Historical (Rotterdam) data were used only as a prognostic score. It does not "
    "replace randomized evidence. Models: outcome {outcome}; prognostic score {prognostic}; weights: logistic / "
    "entropy balancing."
)

# Gate G3 inputs.
REQUIRED_MAPPING = ["age", "menostat", "tsize", "pnodes", "progrec", "estrec", "horTh", "time", "cens"]
REQUIRED_RULE_COLUMNS = ["nodes", "chemo", "hormon"]
MISSING_MAX = 0.05  # per variable and source, and per source for the "complete covariates" step
MIN_TREATED, MIN_CONTROLS, MIN_HIST = 30, 30, 100

MEASUREMENT_REGISTER = [  # how each measurement difference between the sources is handled
    "grade: central Bloom-Richardson I/II/III (A) vs local 2/3 (B) -> excluded from cross-source models",
    "receptors: DCC (A) vs LBA/EIA (B), fmol/mg -> categories <10 / 10-99 / >=100 (A9r; sensitivity X2)",
    "tumour size: mm (A) vs groups (B) -> groups in cross-source models (A9s)",
    "outcome: GBSG RFS event set (A) vs recurrence or death (B) -> T3",
    "control chemotherapy: CMF 3 or 6 cycles (A) vs any chemo, mainly CMF (B) -> T2",
    "surgery and RT: mastectomy (A) vs about 47% breast-conserving with RT (B) -> T3",
]

GATE_NAMES = {
    "G0": "protocol lock",
    "G1": "target trial and estimands",
    "G2": "DAG and identification",
    "G3": "data fitness",
    "G4": "within-trial positivity",
    "G5": "covariate balance",
    "G6": "external support",
    "G7": "prognostic model credibility",
    "G8": "borrowing feasibility",
    "G9": "censoring and horizon adequacy",
    "G10": "estimation integrity",
    "G11": "robustness to unmeasured confounding",
    "G12": "validation and operating characteristics",
}

ACTIONS = {  # id -> (gate, what it does, gate restarted)
    "B4.1": ("G4", "R primary estimand ATT -> ATO (overlap population); tier capped at SUPPORTIVE", None),
    "B5.1": ("G5", "add age^2, age x post and log1p(nodes)^2 to the propensity model", "G4"),
    "B7.1": ("G7", "switch m_B to the linear prognostic score (NOT TabPFN)", None),
    "B7.2": ("G7", "no prognostic score: mu0 = alpha + grade + tsize terms", None),
}

R_ESTIMANDS = ["ATT", "ATE", "ATC", "ATO"]
REWEIGHTED_ARMS = {"ATT": ["control"], "ATC": ["treated"], "ATE": ["treated", "control"], "ATO": ["treated", "control"]}

CHEAP_LABEL = "logistic/linear (NOT TabPFN)"
WEIGHTS_LABEL = "logistic propensity (R) / entropy balancing (H) / logistic source odds (E) (NOT TabPFN)"
E_NOTE = "Exploratory; relies on untestable cross-source exchangeability (T1-T5, A9r, A9s, W3)"
CATE_NOTE = "Subgroup estimates are exploratory; differences between subgroups are not evidence of effect modification."
EXPLORATORY_TITLE = "EXPLORATORY - NOT ACCEPTED - not evidence of efficacy"

EST_COLUMNS = [
    "estimand",
    "scope",
    "role",
    "horizon_years",
    "method",
    "backend_label",
    "est",
    "se",
    "lo",
    "hi",
    "df",
    "rr",
    "rr_lo",
    "rr_hi",
    "evalue",
    "evalue_ci",
    "n_target",
    "n_comparison",
    "ess",
    "changed_from",
    "note",
]
OK_STATUSES = ("PASS", "CONDITIONAL", "PASS_AFTER_ACTION")


def protocol_dict():
    """All protocol constants as one dict, read at call time (the protocol digest hashes it)."""
    return {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "PRIOR_OUTCOME_EXPOSURE": PRIOR_OUTCOME_EXPOSURE,
        "TREATMENT": TREATMENT,
        "CONTROL_VALUE": CONTROL_VALUE,
        "AGE_MAX": AGE_MAX,
        "LANDMARK_DAYS": LANDMARK_DAYS,
        "DAYS": DAYS,
        "HORIZON_YEARS": HORIZON_YEARS,
        "N_REAL": N_REAL,
        "SEED": SEED,
        "K_FOLDS": K_FOLDS,
        "RECEPTOR_CUTS": RECEPTOR_CUTS,
        "MAX_TABPFN_FITS": MAX_TABPFN_FITS,
        "Z_T": Z_T,
        "Z_S": Z_S,
        "TRIAL_COLUMNS": TRIAL_COLUMNS,
        "SOURCE_COLUMNS": SOURCE_COLUMNS,
        "LATENT": LATENT,
        "GATES": GATES,
        "DAG_EDGES": DAG_EDGES,
        "ASSUMPTIONS": ASSUMPTIONS,
        "TARGET_TRIAL": TARGET_TRIAL,
        "ELIGIBILITY": ELIGIBILITY,
        "ESTIMANDS": ESTIMANDS,
        "ESTIMAND_ROLES": ESTIMAND_ROLES,
        "SENSITIVITY": SENSITIVITY,
        "SENS_LANDMARKS": SENS_LANDMARKS,
        "B_YEARS_X1": B_YEARS_X1,
        "SUBGROUPS": SUBGROUPS,
        "FOOTER": FOOTER,
        "REQUIRED_MAPPING": REQUIRED_MAPPING,
        "REQUIRED_RULE_COLUMNS": REQUIRED_RULE_COLUMNS,
        "MISSING_MAX": MISSING_MAX,
        "MIN_SIZES": (MIN_TREATED, MIN_CONTROLS, MIN_HIST),
    }
