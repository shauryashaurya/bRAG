import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import glob
    import json
    import io
    import re
    import pandas as pd
    import plotly.express as px
    from contextlib import redirect_stdout
    return glob, io, json, mo, os, pd, px, redirect_stdout


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
def _(mo, os):
    try:
        from autogen import (
            AssistantAgent,
            UserProxyAgent,
            GroupChat,
            GroupChatManager,
        )

        _ag_ok = True
    except ImportError:
        AssistantAgent = UserProxyAgent = GroupChat = GroupChatManager = None
        _ag_ok = False
    _key = os.environ.get("GOOGLE_API_KEY", "")
    mo.stop(not _ag_ok, mo.md("Install: `pip install pyautogen`"))
    mo.stop(not bool(_key), mo.md("Enter Gemini API key above."))
    # litellm routes gemini/ prefix to Google; cache_seed=None disables response caching
    AG_CFG = {
        "config_list": [{"model": "gemini/gemini-2.0-flash", "api_key": _key}],
        "cache_seed": None,
    }
    return AG_CFG, AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent


@app.cell
def _(dfs, json):
    # tool implementations are defined once; registered per tutorial on agent pairs


    def retrieve_policy(policy_id: str) -> str:
        """Return all CSV rows matching policy_id as JSON, max 3000 chars."""
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
        """Compute IFRS17 GMM CSM rollforward. closing = opening + accretion - release."""
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
        """Compute IFRS17 insurance revenue. revenue = expected_claims + ra_release + csm_release."""
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
    return csm_rollforward, lob_stats, retrieve_policy, revenue_decomposition


@app.cell
def _():
    return


@app.cell
def _(
    AG_CFG,
    AssistantAgent,
    UserProxyAgent,
    io,
    mo,
    redirect_stdout,
    retrieve_policy,
):
    # tools must be registered on BOTH agents:
    # register_for_llm: adds JSON schema to assistant's system prompt
    # register_for_execution: runs the function when assistant requests it

    _assistant = AssistantAgent(
        "analyst",
        llm_config=AG_CFG,
        system_message="You are an IFRS17 analyst. Use retrieve_policy to answer questions about policies. Reply TERMINATE when done.",
    )
    _proxy = UserProxyAgent(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
    )

    # register the tool on the pair
    _proxy.register_for_execution()(
        _assistant.register_for_llm(
            description="Return all CSV rows for a given policy_id."
        )(retrieve_policy)
    )

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _assistant, message="What IFRS17 model type is used for policy P001?"
        )
    _log = _buf.getvalue()
    mo.md(f"```\n{_log[:2000]}\n```")
    return


@app.cell
def _():
    return


@app.cell
def _(
    AG_CFG,
    AssistantAgent,
    UserProxyAgent,
    io,
    mo,
    redirect_stdout,
    retrieve_policy,
):
    # same setup as T1; the key difference is the question is open-ended
    # the agent decides: do I need to call retrieve_policy, and if so, with which policy_id?
    # compare with T1 where we told it exactly which policy

    _assistant = AssistantAgent(
        "analyst",
        llm_config=AG_CFG,
        system_message="You are an IFRS17 analyst with access to policy data. Use tools to answer. Reply TERMINATE when done.",
    )
    _proxy = UserProxyAgent(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
    )
    _proxy.register_for_execution()(
        _assistant.register_for_llm(
            description="Return CSV rows for a given policy_id."
        )(retrieve_policy)
    )

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _assistant,
            message="Look at the available policies and tell me which ones use PAA vs GMM measurement model.",
        )
    mo.md(f"```\n{_buf.getvalue()[:2000]}\n```")
    return


@app.cell
def _():
    return


@app.cell
def _(
    AG_CFG,
    AssistantAgent,
    UserProxyAgent,
    csm_rollforward,
    io,
    mo,
    redirect_stdout,
    retrieve_policy,
):
    # two tools registered; the task requires both
    # agent must: call retrieve_policy to get numbers, then call csm_rollforward with them

    _assistant = AssistantAgent(
        "analyst",
        llm_config=AG_CFG,
        system_message=(
            "You are an IFRS17 analyst. To compute CSM rollforward: first call retrieve_policy "
            "to get opening/accretion/release values, then call csm_rollforward with those values. "
            "Reply TERMINATE when done."
        ),
    )
    _proxy = UserProxyAgent(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
    )
    _proxy.register_for_execution()(
        _assistant.register_for_llm(
            description="Return CSV rows for a given policy_id."
        )(retrieve_policy)
    )
    _proxy.register_for_execution()(
        _assistant.register_for_llm(
            description="Compute IFRS17 CSM rollforward given opening, accretion, and release."
        )(csm_rollforward)
    )

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _assistant, message="Compute the CSM rollforward for policy P001."
        )
    mo.md(f"```\n{_buf.getvalue()[:2000]}\n```")
    return


@app.cell
def _():
    return


@app.cell
def _(
    AG_CFG,
    AssistantAgent,
    UserProxyAgent,
    csm_rollforward,
    io,
    lob_stats,
    mo,
    redirect_stdout,
    retrieve_policy,
    revenue_decomposition,
):
    # all 4 tools registered; agent uses multi-turn conversation to complete a compound task
    # note: max_consecutive_auto_reply limits runaway loops

    _assistant = AssistantAgent(
        "analyst",
        llm_config=AG_CFG,
        system_message=(
            "You are an IFRS17 analyst with full data access. "
            "Use tools to retrieve data, compute CSM rollforward, decompose revenue, and show table statistics. "
            "Reply TERMINATE when all tasks are complete."
        ),
    )
    _proxy = UserProxyAgent(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
        max_consecutive_auto_reply=8,
    )

    for _fn, _desc in [
        (retrieve_policy, "Return CSV rows for a given policy_id."),
        (
            csm_rollforward,
            "Compute IFRS17 GMM CSM rollforward from opening, accretion, release.",
        ),
        (
            revenue_decomposition,
            "Compute IFRS17 insurance revenue from csm_release, ra_release, expected_claims.",
        ),
        (lob_stats, "Return descriptive statistics for a named loaded table."),
    ]:
        _proxy.register_for_execution()(
            _assistant.register_for_llm(description=_desc)(_fn)
        )

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _assistant,
            message=(
                "For policy P001: retrieve its data, compute CSM rollforward, decompose revenue. "
                "Then provide statistics for the ifrs17_metrics_output table."
            ),
        )
    mo.md(f"```\n{_buf.getvalue()[:3000]}\n```")
    return


@app.cell
def _():
    return


@app.cell
def _(
    AG_CFG,
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    csm_rollforward,
    io,
    lob_stats,
    mo,
    redirect_stdout,
    retrieve_policy,
    revenue_decomposition,
):
    _data_agent = AssistantAgent(
        "DataAgent",
        llm_config=AG_CFG,
        system_message="You retrieve IFRS17 policy data and table statistics. Use only your registered tools.",
    )
    _finance_agent = AssistantAgent(
        "FinanceAgent",
        llm_config=AG_CFG,
        system_message="You compute IFRS17 CSM rollforward and revenue decomposition. Use only your registered tools.",
    )
    _reviewer = AssistantAgent(
        "Reviewer",
        llm_config=AG_CFG,
        system_message=(
            "You review IFRS17 analysis results. "
            "Check: CSM closing must not be negative (onerous flag). "
            "Revenue = expected_claims + ra_release + csm_release. "
            "Summarize findings and reply TERMINATE when done."
        ),
    )
    _proxy = UserProxyAgent(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
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
            description="Compute IFRS17 insurance revenue."
        )(revenue_decomposition)
    )

    _group = GroupChat(
        agents=[_proxy, _data_agent, _finance_agent, _reviewer],
        messages=[],
        max_round=10,
    )
    _manager = GroupChatManager(groupchat=_group, llm_config=AG_CFG)

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _manager,
            message="Analyze policy P001: retrieve data, compute CSM rollforward and revenue, then review the results.",
        )
    mo.md(f"```\n{_buf.getvalue()[:3000]}\n```")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
