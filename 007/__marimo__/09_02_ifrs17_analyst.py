import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import glob
    import json
    import datetime
    import io
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from contextlib import redirect_stdout
    return datetime, glob, io, json, mo, os, pd, px, redirect_stdout


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
    _m = "ifrs17_metrics_output"
    if _m in dfs and "policy_group_id" in dfs[_m].columns:
        _df = dfs[_m]
        _metric_cols = [
            c
            for c in ["csm_closing", "insurance_revenue", "risk_adjustment"]
            if c in _df.columns
        ]
        if _metric_cols:
            _bar = px.bar(
                _df.melt(id_vars="policy_group_id", value_vars=_metric_cols),
                x="policy_group_id",
                y="value",
                color="variable",
                barmode="group",
                title="IFRS17 Metrics by Policy Group",
            )
            dashboard = _bar
        else:
            dashboard = mo.md("ifrs17_metrics_output missing metric columns.")
    elif "ifrs17_metrics_output" not in dfs and dfs:
        # fallback: show first numeric distribution
        _n = next(iter(dfs))
        _df = dfs[_n]
        _num = _df.select_dtypes("number").columns.tolist()
        dashboard = (
            px.histogram(_df, x=_num[0], title=f"{_n}: {_num[0]}")
            if _num
            else mo.md("No numeric columns.")
        )
    else:
        dashboard = mo.md("Add CSVs to ./data/ matching sample-ifrs17schema.yaml.")
    dashboard
    return


@app.cell
def _(datetime, dfs, json):
    # single source of truth for all tool implementations
    # all three frameworks import from this cell


    def retrieve_policy(policy_id: str) -> str:
        """Return all CSV rows for this policy_id as JSON, max 3000 chars."""
        rows = []
        for name, df in dfs.items():
            if "policy_id" in df.columns:
                m = df[df["policy_id"].astype(str) == str(policy_id)]
                if not m.empty:
                    rows.extend(m.head(3).to_dict("records"))
        return (
            json.dumps(rows, default=str)[:3000]
            if rows
            else f"No data for {policy_id}"
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
        """Return descriptive statistics for a loaded table as JSON."""
        if table_name not in dfs:
            return f"Not found. Available: {list(dfs.keys())}"
        num = dfs[table_name].select_dtypes("number")
        return (
            json.dumps(num.describe().round(2).to_dict())
            if not num.empty
            else "No numeric columns."
        )


    def build_audit(
        policy_group_id: str, amount: float, source_metric: str, framework: str
    ) -> dict:
        return {
            "entry_id": f"AI-{framework[:3].upper()}-{datetime.date.today().isoformat()}",
            "posting_date": datetime.date.today().isoformat(),
            "policy_group_id": policy_group_id,
            "account_code": "5001",
            "description": f"AI-extracted {source_metric} via {framework}",
            "amount": round(amount, 2),
            "dr_cr_flag": "CR",
            "source_metric": source_metric,
            "export_status": "Ready",
        }
    return (
        build_audit,
        csm_rollforward,
        lob_stats,
        retrieve_policy,
        revenue_decomposition,
    )


@app.cell
def _(mo):
    framework = mo.ui.dropdown(
        options=["DSPy", "Semantic Kernel", "AutoGen"],
        label="Analysis framework",
        value="DSPy",
    )
    framework
    return (framework,)


@app.cell
def _(dfs, mo):
    _opts = []
    if (
        "insurance_policy_master" in dfs
        and "policy_id" in dfs["insurance_policy_master"].columns
    ):
        _opts = list(
            dfs["insurance_policy_master"]["policy_id"].astype(str).unique()[:20]
        )
    if not _opts:
        _opts = ["P001"]
    policy_id = mo.ui.dropdown(options=_opts, label="Policy ID", value=_opts[0])
    policy_id
    return (policy_id,)


@app.cell
def _(
    build_audit,
    csm_rollforward,
    framework,
    json,
    lob_stats,
    mo,
    os,
    policy_id,
    retrieve_policy,
    revenue_decomposition,
):
    mo.stop(framework.value != "DSPy")
    mo.stop(
        not bool(os.environ.get("GOOGLE_API_KEY")), mo.md("Enter API key above.")
    )

    import dspy as _dspy

    _dspy.configure(
        lm=_dspy.LM(
            "gemini/gemini-2.0-flash", api_key=os.environ["GOOGLE_API_KEY"]
        )
    )


    class _FullAnalysis(_dspy.Signature):
        """
        Full IFRS17 analysis. Steps:
        1. Call retrieve_policy to get data.
        2. Extract csm opening, accretion, release values and call csm_rollforward.
        3. Extract csm_release, ra_release, expected_claims and call revenue_decomposition.
        4. Call lob_stats on ifrs17_metrics_output.
        Report all computed values. Do not estimate values not found in data.
        """

        policy_id: str = _dspy.InputField()
        report: str = _dspy.OutputField(
            desc="JSON string with all computed IFRS17 metrics"
        )


    _agent = _dspy.ReAct(
        _FullAnalysis,
        tools=[retrieve_policy, csm_rollforward, revenue_decomposition, lob_stats],
        max_iters=10,
    )
    try:
        _result = _agent(policy_id=policy_id.value)
        _raw = _result.report
        try:
            _parsed = json.loads(_raw)
        except Exception:
            _parsed = {"raw": _raw}
        _audit = build_audit(
            policy_id.value,
            float(_parsed.get("insurance_revenue", 0)),
            "insurance_revenue",
            "DSPy",
        )
        dspy_out = mo.md(
            f"### DSPy ReAct Result\n\n"
            f"```json\n{json.dumps(_parsed, indent=2, default=str)}\n```\n\n"
            f"### Audit Entry\n\n```json\n{json.dumps(_audit, indent=2)}\n```"
        )
    except Exception as _e:
        dspy_out = mo.md(f"**DSPy error:** `{_e}`")
    dspy_out
    return


@app.cell
async def _(build_audit, framework, json, mo, os, policy_id):
    mo.stop(framework.value != "Semantic Kernel")
    mo.stop(
        not bool(os.environ.get("GOOGLE_API_KEY")), mo.md("Enter API key above.")
    )

    import semantic_kernel as _sk
    from semantic_kernel.connectors.ai.google.google_ai import (
        GoogleAIChatCompletion as _GACC,
    )
    from semantic_kernel.connectors.ai.function_choice_behavior import (
        FunctionChoiceBehavior as _FCB,
    )
    from semantic_kernel.contents import ChatHistory as _CH
    from semantic_kernel.functions import kernel_function as _kf


    # wrap tools in a plugin class so SK can register them
    class _IFRS17Plugin:
        @_kf(description="Return all CSV rows for this policy_id as JSON.")
        def retrieve_policy(self, policy_id: str) -> str:
            return retrieve_policy(policy_id)

        @_kf(
            description="Compute IFRS17 GMM CSM rollforward from opening, accretion, release."
        )
        def csm_rollforward(
            self, opening: float, accretion: float, release: float
        ) -> str:
            return csm_rollforward(opening, accretion, release)

        @_kf(
            description="Compute IFRS17 insurance revenue from csm_release, ra_release, expected_claims."
        )
        def revenue_decomposition(
            self, csm_release: float, ra_release: float, expected_claims: float
        ) -> str:
            return revenue_decomposition(csm_release, ra_release, expected_claims)

        @_kf(description="Return descriptive stats for a named loaded table.")
        def lob_stats(self, table_name: str) -> str:
            return lob_stats(table_name)


    _kernel = _sk.Kernel()
    _kernel.add_service(
        _GACC(
            gemini_model_id="gemini-2.0-flash",
            api_key=os.environ["GOOGLE_API_KEY"],
        )
    )
    _kernel.add_plugin(_IFRS17Plugin(), plugin_name="ifrs17")

    _settings = _kernel.get_prompt_execution_settings_from_service_id("default")
    _settings.function_choice_behavior = _FCB.Auto()

    _history = _CH()
    _history.add_user_message(
        f"Full IFRS17 analysis for policy {policy_id.value}: "
        "retrieve data, compute CSM rollforward, decompose revenue, "
        "show stats for ifrs17_metrics_output. Report all computed values as JSON."
    )
    _svc = _kernel.get_service(type=_GACC)

    try:
        _result = await _svc.get_chat_message_content(
            chat_history=_history, settings=_settings, kernel=_kernel
        )
        _raw = str(_result)
        try:
            _parsed = json.loads(
                _raw.strip().lstrip("```json").rstrip("```").strip()
            )
        except Exception:
            _parsed = {"raw": _raw}
        _audit = build_audit(
            policy_id.value,
            float(_parsed.get("insurance_revenue", 0)),
            "insurance_revenue",
            "SemanticKernel",
        )
        sk_out = mo.md(
            f"### SK Result\n\n"
            f"```json\n{json.dumps(_parsed, indent=2, default=str)}\n```\n\n"
            f"### Audit Entry\n\n```json\n{json.dumps(_audit, indent=2)}\n```"
        )
    except Exception as _e:
        sk_out = mo.md(f"**SK error:** `{_e}`")
    sk_out
    return


@app.cell
def _(
    build_audit,
    csm_rollforward,
    framework,
    io,
    json,
    lob_stats,
    mo,
    os,
    policy_id,
    redirect_stdout,
    retrieve_policy,
    revenue_decomposition,
):
    mo.stop(framework.value != "AutoGen")
    mo.stop(
        not bool(os.environ.get("GOOGLE_API_KEY")), mo.md("Enter API key above.")
    )

    from autogen import (
        AssistantAgent as _AA,
        UserProxyAgent as _UPA,
        GroupChat as _GC,
        GroupChatManager as _GCM,
    )

    _key = os.environ["GOOGLE_API_KEY"]
    _cfg = {
        "config_list": [{"model": "gemini/gemini-2.0-flash", "api_key": _key}],
        "cache_seed": None,
    }

    _data_agent = _AA(
        "DataAgent",
        llm_config=_cfg,
        system_message="You retrieve IFRS17 data and statistics. Use your tools only.",
    )
    _finance_agent = _AA(
        "FinanceAgent",
        llm_config=_cfg,
        system_message="You compute CSM rollforward and revenue decomposition. Use your tools only.",
    )
    _reviewer = _AA(
        "Reviewer",
        llm_config=_cfg,
        system_message=(
            "You review and validate IFRS17 results. "
            "Check onerous flag and revenue identity. "
            "Summarize findings as JSON. Reply TERMINATE when done."
        ),
    )
    _proxy = _UPA(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
        max_consecutive_auto_reply=10,
    )

    # DataAgent tools
    _proxy.register_for_execution()(
        _data_agent.register_for_llm(
            description="Return CSV rows for a given policy_id."
        )(retrieve_policy)
    )
    _proxy.register_for_execution()(
        _data_agent.register_for_llm(
            description="Return descriptive stats for a named table."
        )(lob_stats)
    )

    # FinanceAgent tools
    _proxy.register_for_execution()(
        _finance_agent.register_for_llm(
            description="Compute IFRS17 CSM rollforward."
        )(csm_rollforward)
    )
    _proxy.register_for_execution()(
        _finance_agent.register_for_llm(
            description="Compute IFRS17 insurance revenue decomposition."
        )(revenue_decomposition)
    )

    _group = _GC(
        agents=[_proxy, _data_agent, _finance_agent, _reviewer],
        messages=[],
        max_round=12,
    )
    _manager = _GCM(groupchat=_group, llm_config=_cfg)

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _manager,
            message=(
                f"Analyze policy {policy_id.value}: "
                "DataAgent retrieves data and stats. FinanceAgent computes CSM and revenue. "
                "Reviewer validates and outputs final JSON summary."
            ),
        )
    _log = _buf.getvalue()

    # extract last JSON block from conversation
    _matches = list(
        __import__("re").finditer(r"\{[^{}]{20,}\}", _log, __import__("re").DOTALL)
    )
    _parsed = {}
    if _matches:
        try:
            _parsed = json.loads(_matches[-1].group())
        except Exception:
            _parsed = {"raw_snippet": _matches[-1].group()[:500]}

    _audit = build_audit(
        policy_id.value,
        float(_parsed.get("insurance_revenue", 0)),
        "insurance_revenue",
        "AutoGen",
    )
    ag_out = mo.md(
        f"### AutoGen GroupChat Result\n\n"
        f"```json\n{json.dumps(_parsed, indent=2, default=str)}\n```\n\n"
        f"### Audit Entry\n\n```json\n{json.dumps(_audit, indent=2)}\n```\n\n"
        f"### Chat Log (truncated)\n\n```\n{_log[:2000]}\n```"
    )
    ag_out
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
