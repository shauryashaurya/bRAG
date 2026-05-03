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
    # verify SK is importable before any tutorial runs
    try:
        import semantic_kernel as sk
        from semantic_kernel.connectors.ai.google.google_ai import (
            GoogleAIChatCompletion,
        )
        from semantic_kernel.functions import kernel_function
        from semantic_kernel.connectors.ai.function_choice_behavior import (
            FunctionChoiceBehavior,
        )
        from semantic_kernel.contents import ChatHistory
        from semantic_kernel.functions.kernel_arguments import KernelArguments

        _sk_ok = True
    except ImportError as _e:
        _sk_ok = False
        sk = GoogleAIChatCompletion = kernel_function = FunctionChoiceBehavior = (
            ChatHistory
        ) = KernelArguments = None

    _key = os.environ.get("GOOGLE_API_KEY", "")
    mo.stop(not _sk_ok, mo.md("Install: `pip install semantic-kernel[google]`"))
    mo.stop(not bool(_key), mo.md("Enter Gemini API key above."))
    return (
        ChatHistory,
        FunctionChoiceBehavior,
        GoogleAIChatCompletion,
        KernelArguments,
        kernel_function,
        sk,
    )


@app.cell
def _(GoogleAIChatCompletion, os, sk):
    # factory: new kernel per tutorial to avoid duplicate function registration
    def make_kernel():
        _k = sk.Kernel()
        _k.add_service(
            GoogleAIChatCompletion(
                gemini_model_id="gemini-2.0-flash",
                api_key=os.environ.get("GOOGLE_API_KEY", ""),
            )
        )
        return _k
    return (make_kernel,)


@app.cell
def _(dfs, json, kernel_function):
    # plugin classes group related tools
    # @kernel_function makes the method visible to SK's function calling


    class DataPlugin:
        @kernel_function(
            description="Return all CSV rows for this policy_id as a JSON string."
        )
        def retrieve_policy(self, policy_id: str) -> str:
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

        @kernel_function(
            description="Return descriptive statistics for a loaded table as JSON."
        )
        def lob_stats(self, table_name: str) -> str:
            if table_name not in dfs:
                return f"Not found. Available: {list(dfs.keys())}"
            num = dfs[table_name].select_dtypes("number")
            return (
                json.dumps(num.describe().round(2).to_dict())
                if not num.empty
                else "No numeric columns."
            )


    class FinancePlugin:
        @kernel_function(
            description="Compute IFRS17 GMM CSM rollforward. closing = opening + accretion - release."
        )
        def csm_rollforward(
            self, opening: float, accretion: float, release: float
        ) -> str:
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

        @kernel_function(
            description="Compute IFRS17 insurance revenue. revenue = expected_claims + ra_release + csm_release."
        )
        def revenue_decomposition(
            self, csm_release: float, ra_release: float, expected_claims: float
        ) -> str:
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
    return DataPlugin, FinancePlugin


@app.cell
def _():
    return


@app.cell
async def _(DataPlugin, KernelArguments, make_kernel, mo):
    # programmer picks which function to call and provides arguments explicitly
    # the LLM has no involvement in the function selection decision

    _kernel = make_kernel()
    _plugin = _kernel.add_plugin(DataPlugin(), plugin_name="data")
    _fn = _plugin["retrieve_policy"]

    _result = await _kernel.invoke(_fn, KernelArguments(policy_id="P001"))
    mo.md(f"""
    **Function called by:** programmer (no LLM involved in selection)

    **Result (truncated):** `{str(_result)[:300]}...`

    At this stage, SK is just a typed function registry. No LLM orchestration.
    """)
    return


@app.cell
def _():
    return


@app.cell
async def _(
    ChatHistory,
    DataPlugin,
    FunctionChoiceBehavior,
    GoogleAIChatCompletion,
    make_kernel,
    mo,
):
    # FunctionChoiceBehavior.Auto() attaches the function schema to the chat request
    # Gemini may issue a function call; SK dispatches it and feeds result back
    # the LLM decides: should I call retrieve_policy, or answer directly?

    _kernel = make_kernel()
    _kernel.add_plugin(DataPlugin(), plugin_name="data")

    _svc_id = "google_ai"
    _settings = _kernel.get_prompt_execution_settings_from_service_id("default")
    _settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    _history = ChatHistory()
    _history.add_user_message(
        "What policies are loaded and what lines of business do they cover?"
    )

    _svc = _kernel.get_service(type=GoogleAIChatCompletion)
    _result = await _svc.get_chat_message_content(
        chat_history=_history,
        settings=_settings,
        kernel=_kernel,
    )
    mo.md(f"**LLM response (with auto tool use):** {str(_result)}")
    return


@app.cell
def _():
    return


@app.cell
async def _(
    DataPlugin,
    FinancePlugin,
    FunctionChoiceBehavior,
    GoogleAIChatCompletion,
    make_kernel,
    mo,
):
    # both DataPlugin and FinancePlugin registered
    # question requires: retrieve policy data then compute CSM
    # LLM must call retrieve_policy first, extract numbers, then call csm_rollforward

    _kernel = make_kernel()
    _kernel.add_plugin(DataPlugin(), plugin_name="data")
    _kernel.add_plugin(FinancePlugin(), plugin_name="finance")

    _settings = _kernel.get_prompt_execution_settings_from_service_id("default")
    _settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    from semantic_kernel.contents import ChatHistory as _CH

    _history = _CH()
    _history.add_user_message(
        "Retrieve data for policy P001 and compute the CSM rollforward. "
        "Extract opening, accretion, and release values from the retrieved data."
    )

    _svc = _kernel.get_service(type=GoogleAIChatCompletion)
    _result = await _svc.get_chat_message_content(
        chat_history=_history, settings=_settings, kernel=_kernel
    )
    mo.md(f"**Result:** {str(_result)}")
    return


@app.cell
def _():
    return


@app.cell
async def _(
    DataPlugin,
    FinancePlugin,
    FunctionChoiceBehavior,
    GoogleAIChatCompletion,
    make_kernel,
    mo,
):
    # ChatHistory persists across turns; function results are included in history
    # the LLM can refer back to earlier tool results in later turns

    _kernel = make_kernel()
    _kernel.add_plugin(DataPlugin(), plugin_name="data")
    _kernel.add_plugin(FinancePlugin(), plugin_name="finance")

    _settings = _kernel.get_prompt_execution_settings_from_service_id("default")
    _settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    from semantic_kernel.contents import ChatHistory as _CH

    _history = _CH()
    _svc = _kernel.get_service(type=GoogleAIChatCompletion)


    async def _turn(msg):
        _history.add_user_message(msg)
        _r = await _svc.get_chat_message_content(
            chat_history=_history, settings=_settings, kernel=_kernel
        )
        _history.add_assistant_message(str(_r))
        return str(_r)


    _r1 = await _turn("Retrieve policy P001 data and compute its CSM rollforward.")
    _r2 = await _turn(
        "Now decompose the insurance revenue using the release figure from the CSM you just computed."
    )

    mo.md(f"""
    **Turn 1:** {_r1}

    **Turn 2 (references earlier result):** {_r2}
    """)
    return


@app.cell
def _():
    return


@app.cell
async def _(
    DataPlugin,
    FinancePlugin,
    FunctionChoiceBehavior,
    GoogleAIChatCompletion,
    make_kernel,
    mo,
):
    _kernel = make_kernel()
    _kernel.add_plugin(DataPlugin(), plugin_name="data")
    _kernel.add_plugin(FinancePlugin(), plugin_name="finance")

    _settings = _kernel.get_prompt_execution_settings_from_service_id("default")
    _settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    from semantic_kernel.contents import ChatHistory as _CH

    _history = _CH()
    _history.add_user_message(
        "Full IFRS17 analysis for policy P001: "
        "retrieve its data, compute CSM rollforward, decompose revenue, "
        "and provide summary statistics for the ifrs17_metrics_output table. "
        "Report all computed values explicitly."
    )
    _svc = _kernel.get_service(type=GoogleAIChatCompletion)
    _result = await _svc.get_chat_message_content(
        chat_history=_history, settings=_settings, kernel=_kernel
    )
    mo.md(f"**Full analysis:** {str(_result)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
