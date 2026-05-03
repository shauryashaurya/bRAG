import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import glob
    import json
    import pandas as pd
    import plotly.express as px
    return glob, json, mo, os, pd, px


@app.cell
def _():
    import dspy
    return (dspy,)


@app.cell
def _(mo):
    api_key_input = mo.ui.text(label="Gemini API Key", kind="password")
    return (api_key_input,)


@app.cell
def _(api_key_input):
    api_key_input
    return


@app.cell
def _(api_key_input, os):
    os.environ["GOOGLE_API_KEY"] = api_key_input.value
    return


@app.cell
def _():
    return


@app.cell
def _(glob, os, pd):
    # load all CSVs from ./ifrs17_data
    dfs = {}
    for _f in glob.glob("./ifrs17_data/*.csv"):
        _name = os.path.basename(_f).replace(".csv", "")
        try:
            dfs[_name] = pd.read_csv(_f)
        except Exception:
            pass
    return (dfs,)


@app.cell
def _(dfs, mo, px):
    # portfolio chart: rendered directly, no LLM involved
    _m = "ifrs17_metrics_output"
    if _m in dfs:
        _df = dfs[_m]
        _cols = [
            c
            for c in ["csm_closing", "insurance_revenue", "risk_adjustment"]
            if c in _df.columns
        ]
        if _cols and "policy_group_id" in _df.columns:
            _fig = px.bar(
                _df.melt(id_vars="policy_group_id", value_vars=_cols),
                x="policy_group_id",
                y="value",
                color="variable",
                barmode="group",
                title="IFRS17 Metrics by Policy Group",
            )
            portfolio_chart = _fig
        else:
            portfolio_chart = mo.md(
                "ifrs17_metrics_output loaded but missing expected columns."
            )
    elif dfs:
        _name = next(iter(dfs))
        _df = dfs[_name]
        _num = _df.select_dtypes("number").columns.tolist()
        portfolio_chart = (
            px.histogram(_df, x=_num[0], title=f"{_name}: {_num[0]}")
            if _num
            else mo.md("No numeric columns.")
        )
    else:
        portfolio_chart = mo.md("No CSVs in ./data/")
    portfolio_chart
    return


@app.cell
def _(dfs, json):
    # all tools are deterministic Python; LLMs read their docstrings as descriptions


    def retrieve_policy(policy_id: str) -> str:
        """Return all CSV rows for this policy_id as a JSON string, max 3000 chars."""
        rows = []
        for name, df in dfs.items():
            if "policy_id" in df.columns:
                m = df[df["policy_id"].astype(str) == str(policy_id)]
                if not m.empty:
                    rows.extend(m.head(3).to_dict("records"))
        return (
            json.dumps(rows, default=str)[:3000]
            if rows
            else f"No data for policy_id={policy_id}"
        )


    def csm_rollforward(opening: float, accretion: float, release: float) -> str:
        """Compute IFRS17 GMM CSM rollforward. closing = opening + accretion - release. Returns JSON."""
        closing = opening + accretion - release
        onerous = closing < 0
        return json.dumps(
            {
                "csm_opening": round(opening, 2),
                "csm_accretion": round(accretion, 2),
                "csm_release": round(release, 2),
                "csm_closing": round(max(closing, 0.0), 2),
                "loss_component": round(abs(closing), 2) if onerous else 0.0,
                "onerous": onerous,
            }
        )


    def revenue_decomposition(
        csm_release: float, ra_release: float, expected_claims: float
    ) -> str:
        """Compute IFRS17 insurance revenue. revenue = expected_claims + ra_release + csm_release. Returns JSON."""
        return json.dumps(
            {
                "expected_claims_incurred": round(expected_claims, 2),
                "risk_adjustment_release": round(ra_release, 2),
                "csm_release": round(csm_release, 2),
                "insurance_revenue": round(
                    expected_claims + ra_release + csm_release, 2
                ),
            }
        )


    def lob_stats(table_name: str) -> str:
        """Return descriptive statistics for a loaded table. Available tables are in dfs."""
        if table_name not in dfs:
            return f"Table not found. Available: {list(dfs.keys())}"
        num = dfs[table_name].select_dtypes("number")
        return (
            json.dumps(num.describe().round(2).to_dict())
            if not num.empty
            else "No numeric columns."
        )
    return csm_rollforward, lob_stats, retrieve_policy, revenue_decomposition


@app.cell
def _(mo, os):
    _key = os.environ.get("GOOGLE_API_KEY", "")
    mo.stop(not bool(_key), mo.md("Enter Gemini API key above."))
    return


@app.cell
def _(dspy, os):
    dspy.configure(
        lm=dspy.LM(
            "gemini/gemini-2.5-flash", api_key=os.environ.get("GOOGLE_API_KEY", "")
        )
    )
    return


@app.cell
def _():
    return


@app.cell
def _(dspy, mo, retrieve_policy):
    # programmer calls the tool; LLM only interprets the result
    # this is the baseline: no agency, explicit tool invocation


    class Interpret(dspy.Signature):
        context: str = dspy.InputField(desc="Policy data rows as JSON")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()


    _data = retrieve_policy("P001")  # programmer calls the tool
    _res = dspy.Predict(Interpret)(
        context=_data,
        question="What is the policy status and what IFRS17 model type is used?",
    )
    mo.md(f"""
    **Tool called by:** programmer (not LLM)

    **Data (truncated):** `{_data[:200]}...`

    **LLM interprets:** {_res.answer}

    Note: the LLM had no choice about whether or when to call the tool.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(dspy, mo, retrieve_policy):
    # LLM decides when to call retrieve_policy and with what argument
    # it may call it zero or multiple times before producing its answer


    class PolicyQA(dspy.Signature):
        """Answer questions about IFRS17 policies. Use retrieve_policy to fetch data first."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()


    _agent = dspy.ReAct(PolicyQA, tools=[retrieve_policy], max_iters=4)
    _res = _agent(
        question="What policies are in the data and what lines of business do they cover?"
    )
    mo.md(f"**Agent answer:** {_res.answer}")
    return


@app.cell
def _():
    return


@app.cell
def _(csm_rollforward, dspy, mo, retrieve_policy):
    # agent must: 1) retrieve data to find opening/accretion/release values
    #             2) call csm_rollforward with those values
    # argument extraction from retrieved data is the LLM's responsibility


    class CSMAnalyst(dspy.Signature):
        """Retrieve policy data then compute the CSM rollforward. Use retrieve_policy first, then csm_rollforward."""

        policy_id: str = dspy.InputField()
        analysis: str = dspy.OutputField(
            desc="CSM rollforward result with interpretation"
        )


    _agent = dspy.ReAct(
        CSMAnalyst, tools=[retrieve_policy, csm_rollforward], max_iters=5
    )
    _res = _agent(policy_id="P001")
    mo.md(f"**CSM analysis:** {_res.analysis}")
    return


@app.cell
def _():
    return


@app.cell
def _(
    csm_rollforward,
    dspy,
    lob_stats,
    mo,
    retrieve_policy,
    revenue_decomposition,
):
    # agent builds an implicit multi-step plan through its Thought steps
    # it calls tools in sequence: retrieve -> rollforward -> revenue -> lob_stats


    class IFRS17Analyst(dspy.Signature):
        """Full IFRS17 analysis. Use all available tools. Do not estimate values not found in data."""

        task: str = dspy.InputField()
        report: str = dspy.OutputField(
            desc="Structured analysis with all computed figures"
        )


    _agent = dspy.ReAct(
        IFRS17Analyst,
        tools=[retrieve_policy, csm_rollforward, revenue_decomposition, lob_stats],
        max_iters=8,
    )
    _res = _agent(
        task=(
            "For policy P001: retrieve data, compute CSM rollforward, decompose revenue. "
            "Also show summary statistics for the ifrs17_metrics_output table."
        )
    )
    mo.md(f"**Full report:** {_res.report}")
    return


@app.cell
def _():
    return


@app.cell
def _(csm_rollforward, dspy, mo, retrieve_policy, revenue_decomposition):
    class Extract(dspy.Signature):
        """Retrieve IFRS17 policy data and compute CSM and revenue metrics. Output must be valid JSON."""

        policy_id: str = dspy.InputField()
        metrics_json: str = dspy.OutputField(
            desc="JSON with csm_rollforward and revenue_decomposition results"
        )


    class Review(dspy.Signature):
        """
        Validate IFRS17 metrics for compliance.
        Rules: CSM closing must not be negative (loss component must be declared instead).
        Revenue must equal expected_claims_incurred + risk_adjustment_release + csm_release.
        Output Pass or Fail with specific rule violations listed.
        """

        metrics_json: str = dspy.InputField()
        verdict: str = dspy.OutputField()


    _extractor = dspy.ReAct(
        Extract,
        tools=[retrieve_policy, csm_rollforward, revenue_decomposition],
        max_iters=6,
    )
    _reviewer = dspy.Predict(Review)

    _step1 = _extractor(policy_id="P001")
    _step2 = _reviewer(metrics_json=_step1.metrics_json)

    mo.md(f"""
    **Extractor output:** `{_step1.metrics_json}`

    **Reviewer verdict:** {_step2.verdict}
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
