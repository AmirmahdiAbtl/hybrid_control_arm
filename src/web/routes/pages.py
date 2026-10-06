"""Pages: landing, runner and documentation."""

from flask import Blueprint, abort, current_app, redirect, render_template, request, url_for

from src import online
from src.web.docs import DOCS, render_doc
from src.web.examples import EXAMPLES
from src.web.forms import BACKENDS, FormError, default_form, read_inputs
from src.web.jobs import STEPS

bp = Blueprint("pages", __name__)

RESULTS = [  # real runs (TabPFN API, real LLM and Jev, --yes); see README "What was and was not tested"
    {
        "title": "GBSG-2 + Rotterdam, breast cancer",
        "tag": "Two studies, different columns and codings",
        "headline": "+11.3 points",
        "interval": "95% interval +3.3 to +19.2 at 3 years",
        "verdict": "Treated did better",
        "text": "Columns matched without a single hardcoded name and 552 of 2,982 historical patients were "
        "selected. The support gate failed once; stopping the age comparison fixed it.",
    },
]


PIPELINE = [  # who does what in each step, for the explorer on the landing page
    (
        "Read",
        "Your two tables are profiled: types, ranges, missing values. If you give a trial id, the title, arms, "
        "primary outcome and eligibility text come from ClinicalTrials.gov.",
        "Nothing yet.",
        "Profiles every column; fetches the registry record.",
        "trial: 686 patients, 10 columns; historical: 2982",
    ),
    (
        "Match columns",
        "The historical table is renamed into the trial's column names, with unit and coding changes. "
        "Columns that mean something different are flagged and left out unless you approve them.",
        "Decides the hard matches and writes the transforms.",
        "Jev scores pairs; checks types, units and categories.",
        "approved: age, cens, estrec, pnodes, progrec, time",
    ),
    (
        "Eligibility",
        "Only the historical patients who would have qualified for the new trial are kept.",
        "Turns the entry criteria into rules.",
        "Applies the rules with a safe expression engine (no free code).",
        "552 of 2982 historical patients are eligible",
    ),
    (
        "Plan",
        "The analysis plan: treatment and control arm, outcome and event coding, horizon, landmark, and which "
        "baseline variables to adjust for.",
        "Drafts the plan.",
        "Validates it against the data (columns, coding, missingness, leaks) and sends errors back.",
        "plan accepted by the checks (draft 1); fixes applied by code: 0",
    ),
    (
        "Review",
        "A second, independent LLM call looks for what the first one missed, then the plan is shown to you.",
        "Reviews the plan once.",
        "Passes its concerns back to the planner, once.",
        "[medium] Menopausal status may be strongly tied to age ...  ->  one revision",
    ),
    (
        "Freeze",
        "The plan is hashed. After this point nothing in it can change.",
        "Nothing.",
        "Hashes the plan and the limits; starts the audit trail.",
        "== 6. Plan frozen (e1e000a53c1f); borrowing on ==",
    ),
    (
        "Gates",
        "Data fitness, positivity, balance, external support and follow-up are checked using covariates only. "
        "TabPFN fits the prognostic score on the historical patients. Only then are the trial outcomes read.",
        "Nothing: no model decides a gate.",
        "Every number, threshold and verdict.",
        "G3 covariate balance: PASS\nG4 external support: FAIL",
    ),
    (
        "Repair",
        "When a gate fails, the agent changes one thing from a short list, and the gates run again.",
        "Picks one fix from the failing numbers.",
        "Builds the list of allowed fixes; the estimate is never shown.",
        "applied: drop_cross_source {'name': 'age'}",
    ),
    (
        "Report",
        "Verdict, estimate with interval, E-value, the repairs and every design version, plus a summary in "
        "plain words.",
        "Writes the plain-language summary.",
        "Computes the verdict, claim and tier.",
        "ACCEPTED (SUPPORTIVE): treated patients did better\nDifference +11.3 points (95% CI +3.3 to +19.2)",
    ),
]


def llm_ready():
    return bool(current_app.config["LLM"] or online.llm_status())


@bp.get("/")
def landing():
    steps = [(n, name, text, ask, *PIPELINE[i][2:]) for i, (n, name, text, ask) in enumerate(STEPS)]
    return render_template("landing.html", steps=steps, results=RESULTS, llm_ready=llm_ready())


@bp.route("/run", methods=["GET", "POST"])
def run():
    form, error = default_form(), None
    if request.method == "GET":
        form["mode"] = request.args.get("mode") if request.args.get("mode") in ("single", "step") else form["mode"]
    else:
        form.update(request.form.to_dict())
        try:
            if not llm_ready():
                raise FormError("The LLM is not configured: set the ANTHROPIC_FOUNDRY_* variables in .env.")
            inputs = read_inputs(request.form, request.files)
            agent = inputs["agent"]
            record = current_app.extensions["jobs"].start(
                inputs["trial"],
                inputs["hist"],
                inputs["mode"],
                inputs["label"],
                {k: agent[k] for k in ("backend", "n_real", "max_repairs")},
                trial_labels=inputs["trial_labels"],
                hist_labels=inputs["hist_labels"],
                llm=current_app.config["LLM"] or online.get_llm(),
                out_dir=f"{current_app.config['RESULTS_DIR']}/agent",
                cache_dir=f"{current_app.config['RESULTS_DIR']}/cache",
                **agent,
            )
            return redirect(url_for("jobs.job_page", job_id=record.id), code=303)
        except (FormError, RuntimeError) as e:
            error = str(e)
    return render_template(
        "run.html",
        form=form,
        error=error,
        examples=EXAMPLES,
        backends=BACKENDS,
        llm_ready=llm_ready(),
        recent=current_app.extensions["jobs"].all()[:5],
    )


@bp.get("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="img/mark-dark.png"), code=301)


@bp.get("/docs")
def docs_index():
    return redirect(url_for("pages.docs", name="agent"))


@bp.get("/docs/<name>")
def docs(name):
    if name not in DOCS:
        abort(404)
    title, html, toc = render_doc(current_app.config["DOCS_DIR"], name)
    return render_template("docs.html", title=title, body=html, toc=toc, name=name, docs=DOCS)
