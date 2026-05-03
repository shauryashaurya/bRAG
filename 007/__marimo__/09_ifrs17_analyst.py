import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import json
    import glob
    import io
    import re
    import datetime
    import pandas as pd
    from contextlib import redirect_stdout
    return datetime, glob, io, json, mo, os, pd, re, redirect_stdout


@app.cell
def _(mo):
    api_key_input = mo.ui.text(label="Enter Gemini API Key", kind="password")
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
def _(glob, mo, os, pd):
    _files = glob.glob("./ifrs17_data/*.csv")
    dfs = {}
    for _f in _files:
        _name = os.path.basename(_f).replace(".csv", "")
        try:
            dfs[_name] = pd.read_csv(_f)
        except Exception:
            pass
    _opts = list(dfs.keys()) if dfs else ["(no CSVs in ./ifrs17_data)"]
    file_select = mo.ui.dropdown(
        options=_opts, label="Browse loaded tables", value=_opts[0]
    )
    load_status = mo.callout(
        mo.md(f"Loaded {len(dfs)} table(s): `{', '.join(dfs.keys()) or 'none'}`"),
        kind="info",
    )
    return dfs, file_select, load_status


@app.cell
def _(load_status):
    load_status
    return


@app.cell
def _(file_select):
    file_select
    return


@app.cell
def _(dfs, file_select, mo):
    if file_select.value in dfs:
        browse_view = mo.ui.table(dfs[file_select.value].head(20))
    else:
        browse_view = mo.md(
            "No data loaded. Add CSV files to `./ifrs17_data/` matching sample-ifrs17schema.yaml."
        )
    browse_view
    return


@app.cell
def _(dfs, mo):
    # pull policy IDs from master table if available
    _pid_opts = []
    if "insurance_policy_master" in dfs:
        _df = dfs["insurance_policy_master"]
        if "policy_id" in _df.columns:
            _pid_opts = list(_df["policy_id"].astype(str).unique()[:30])
    if not _pid_opts:
        _pid_opts = ["P001"]  # fallback placeholder

    policy_selector = mo.ui.dropdown(
        options=_pid_opts, label="Policy ID to analyze", value=_pid_opts[0]
    )
    return (policy_selector,)


@app.cell
def _(policy_selector):
    policy_selector
    return


@app.cell
def _(mo):
    framework_select = mo.ui.dropdown(
        options=["DSPy", "Semantic Kernel", "AutoGen"],
        label="Analysis framework",
        value="DSPy",
    )
    return (framework_select,)


@app.cell
def _(framework_select):
    framework_select
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run Analysis")
    return (run_btn,)


@app.cell
def _(run_btn):
    run_btn
    return


@app.cell
def _(dfs, policy_selector):
    def retrieve(policy_id=None):
        # pull rows matching policy_id from any table that has a policy_id column
        _pid = str(policy_id or policy_selector.value)
        _rows = []
        for _name, _df in dfs.items():
            if "policy_id" in _df.columns:
                _match = _df[_df["policy_id"].astype(str) == _pid]
                if not _match.empty:
                    _rows.extend(_match.head(3).to_dict("records"))
            elif "policy_group_id" in _df.columns:
                # include ifrs17_metrics_output rows regardless of policy_id
                _rows.extend(_df.head(3).to_dict("records"))
        if not _rows:
            return f"No rows found for policy_id={_pid} in loaded tables."
        return str(_rows)[:4000]
    return (retrieve,)


@app.function
def validate_csm(opening, accretion, release, closing, tol=0.01):
    # deterministic check: closing must equal opening + accretion - release
    # FCF delta omitted here; in production include it
    _expected = opening + accretion - release
    _diff = abs(_expected - closing)
    _onerous = closing < 0
    return {
        "formula": "CSM_close = CSM_open + accretion - release",
        "expected_closing": round(_expected, 2),
        "actual_closing": round(closing, 2),
        "difference": round(_diff, 2),
        "within_tolerance": _diff < tol,
        "onerous_flag": _onerous,
    }


@app.cell
def _(datetime):
    def build_audit(policy_group_id, amount, source_metric, framework):
        return {
            "entry_id": f"AI-{framework[:3].upper()}-{datetime.date.today().isoformat()}",
            "posting_date": datetime.date.today().isoformat(),
            "policy_group_id": str(policy_group_id),
            "account_code": "5001",
            "description": f"AI-extracted {source_metric} via {framework}",
            "amount": str(amount),
            "dr_cr_flag": "CR",
            "source_metric": source_metric,
            "export_status": "Ready",
        }
    return (build_audit,)


@app.cell
def _(build_audit, framework_select, json, mo, os, retrieve, run_btn):
    def _run_dspy():
        import dspy as _dspy

        _key = os.environ.get("GOOGLE_API_KEY", "")
        _lm = _dspy.LM("gemini/gemini-2.0-flash", api_key=_key)
        _dspy.configure(lm=_lm)

        class Extract(_dspy.Signature):
            context = _dspy.InputField(desc="Policy data rows as a string")
            policy_group_id = _dspy.OutputField()
            csm_opening = _dspy.OutputField(
                desc="Opening CSM balance as a plain number string"
            )
            csm_closing = _dspy.OutputField(
                desc="Closing CSM balance as a plain number string"
            )
            csm_release = _dspy.OutputField(
                desc="CSM released this period as a plain number string"
            )
            insurance_revenue = _dspy.OutputField(
                desc="Insurance revenue for the period as a plain number string"
            )

        class Validate(_dspy.Signature):
            data = _dspy.InputField(desc="Extracted IFRS17 metrics as a string")
            verdict = _dspy.OutputField(
                desc="Brief IFRS17 consistency assessment in 2 sentences"
            )

        _ext = _dspy.Predict(Extract)
        _val = _dspy.Predict(Validate)

        _ctx = retrieve()
        _out = _ext(context=_ctx)
        _verdict = _val(data=str(_out)).verdict

        _open = _parse_number(_out.csm_opening)
        _close = _parse_number(_out.csm_closing)
        _rel = _parse_number(_out.csm_release)
        _math = validate_csm(_open, 0.0, _rel, _close)

        _audit = build_audit(
            _out.policy_group_id,
            _out.insurance_revenue,
            "insurance_revenue",
            "DSPy",
        )
        return _out, _verdict, _math, _audit


    if (
        run_btn.value
        and framework_select.value == "DSPy"
        and os.environ.get("GOOGLE_API_KEY")
    ):
        try:
            _out, _verdict, _math, _audit = _run_dspy()
            dspy_result = mo.md(
                "### DSPy Extraction\n\n"
                f"- **policy_group_id:** `{_out.policy_group_id}`\n"
                f"- **csm_opening:** `{_out.csm_opening}`\n"
                f"- **csm_closing:** `{_out.csm_closing}`\n"
                f"- **csm_release:** `{_out.csm_release}`\n"
                f"- **insurance_revenue:** `{_out.insurance_revenue}`\n\n"
                "### LLM Consistency Verdict\n\n"
                f"{_verdict}\n\n"
                "### Deterministic Math Check\n\n"
                f"```json\n{json.dumps(_math, indent=2)}\n```\n\n"
                "### Audit Entry (journal_entries schema)\n\n"
                f"```json\n{json.dumps(_audit, indent=2)}\n```"
            )
        except Exception as _e:
            dspy_result = mo.md(f"**DSPy error:** `{_e}`")
    else:
        dspy_result = mo.md("_Select DSPy, enter API key, and click Run Analysis_")
    dspy_result
    return


@app.cell
async def _(build_audit, framework_select, json, mo, os, retrieve, run_btn):
    async def _run_sk():
        import semantic_kernel as _sk
        from semantic_kernel.connectors.ai.google.google_ai import (
            GoogleAIChatCompletion,
        )
        from semantic_kernel.functions.kernel_arguments import KernelArguments

        _key = os.environ.get("GOOGLE_API_KEY", "")
        _kernel = _sk.Kernel()
        _svc = GoogleAIChatCompletion(
            gemini_model_id="gemini-2.0-flash", api_key=_key
        )
        _kernel.add_service(_svc)

        _extract_prompt = (
            "Extract fields from the context and return ONLY valid JSON.\n"
            "Fields: policy_group_id (string), csm_opening (number), csm_closing (number), "
            "csm_release (number), insurance_revenue (number).\n"
            "Context: {{$context}}\nJSON:"
        )
        _validate_prompt = "In 2 sentences, assess the IFRS17 consistency of these extracted metrics:\n{{$data}}"

        _ext_fn = _kernel.add_function(
            "extract", "analyst", prompt=_extract_prompt
        )
        _val_fn = _kernel.add_function(
            "validate", "analyst", prompt=_validate_prompt
        )

        _ctx = retrieve()
        _raw = str(await _kernel.invoke(_ext_fn, KernelArguments(context=_ctx)))
        _clean = _raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            _parsed = json.loads(_clean)
        except Exception:
            _parsed = {
                "policy_group_id": "unknown",
                "csm_opening": 0,
                "csm_closing": 0,
                "csm_release": 0,
                "insurance_revenue": 0,
            }

        _verdict = str(
            await _kernel.invoke(_val_fn, KernelArguments(data=str(_parsed)))
        )

        _math = validate_csm(
            _parsed.get("csm_opening", 0),
            0.0,
            _parsed.get("csm_release", 0),
            _parsed.get("csm_closing", 0),
        )
        _audit = build_audit(
            _parsed.get("policy_group_id", "?"),
            _parsed.get("insurance_revenue", 0),
            "insurance_revenue",
            "SemanticKernel",
        )
        return _parsed, _verdict, _math, _audit


    if (
        run_btn.value
        and framework_select.value == "Semantic Kernel"
        and os.environ.get("GOOGLE_API_KEY")
    ):
        try:
            _parsed, _verdict, _math, _audit = await _run_sk()
            sk_result = mo.md(
                "### SK Extraction\n\n"
                f"```json\n{json.dumps(_parsed, indent=2)}\n```\n\n"
                "### LLM Consistency Verdict\n\n"
                f"{_verdict}\n\n"
                "### Deterministic Math Check\n\n"
                f"```json\n{json.dumps(_math, indent=2)}\n```\n\n"
                "### Audit Entry\n\n"
                f"```json\n{json.dumps(_audit, indent=2)}\n```"
            )
        except Exception as _e:
            sk_result = mo.md(f"**SK error:** `{_e}`")
    else:
        sk_result = mo.md(
            "_Select Semantic Kernel, enter API key, and click Run Analysis_"
        )
    sk_result
    return


@app.cell
def _(
    build_audit,
    framework_select,
    io,
    json,
    mo,
    os,
    re,
    redirect_stdout,
    retrieve,
    run_btn,
):
    def _run_autogen():
        from autogen import AssistantAgent, UserProxyAgent

        _key = os.environ.get("GOOGLE_API_KEY", "")
        _cfg = {
            "config_list": [{"model": "gemini/gemini-2.0-flash", "api_key": _key}],
            "cache_seed": None,
        }

        _schema = (
            '{"policy_group_id":"","csm_opening":0,"csm_closing":0,'
            '"csm_release":0,"insurance_revenue":0}'
        )
        _ctx = retrieve()
        _msg = (
            f"Extract and return ONLY valid JSON matching this schema:\n{_schema}\n\n"
            f"Source data:\n{_ctx}"
        )

        _extractor = AssistantAgent(
            "extractor",
            llm_config=_cfg,
            system_message="Return only valid JSON. No explanation, no markdown fences.",
        )
        _validator = AssistantAgent(
            "validator",
            llm_config=_cfg,
            system_message=(
                "You check IFRS17 data for consistency. "
                "Respond in 2 sentences then say TERMINATE."
            ),
        )
        _proxy = UserProxyAgent(
            "coordinator",
            code_execution_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: "TERMINATE"
            in msg.get("content", "").upper(),
            max_consecutive_auto_reply=3,
        )

        _buf = io.StringIO()
        with redirect_stdout(_buf):
            # turn 1: extract
            _proxy.initiate_chat(_extractor, message=_msg)
            _raw = _buf.getvalue()
            _match = re.search(r"\{[^{}]+\}", _raw, re.DOTALL)
            _parsed = {}
            if _match:
                try:
                    _parsed = json.loads(_match.group())
                except Exception:
                    pass

            # turn 2: validate
            _proxy.initiate_chat(
                _validator,
                message=f"Check this IFRS17 extraction for consistency:\n{json.dumps(_parsed)}",
            )

        _verdict = (
            _buf.getvalue().split("\n")[-3] if _buf.getvalue() else "no verdict"
        )
        _math = validate_csm(
            _parse_number(_parsed.get("csm_opening", 0)),
            0.0,
            _parse_number(_parsed.get("csm_release", 0)),
            _parse_number(_parsed.get("csm_closing", 0)),
        )
        _audit = build_audit(
            _parsed.get("policy_group_id", "?"),
            _parsed.get("insurance_revenue", 0),
            "insurance_revenue",
            "AutoGen",
        )
        return _parsed, _verdict, _math, _audit, _buf.getvalue()


    if (
        run_btn.value
        and framework_select.value == "AutoGen"
        and os.environ.get("GOOGLE_API_KEY")
    ):
        try:
            _parsed, _verdict, _math, _audit, _full_log = _run_autogen()
            ag_result = mo.md(
                "### AutoGen Extraction\n\n"
                f"```json\n{json.dumps(_parsed, indent=2)}\n```\n\n"
                "### Deterministic Math Check\n\n"
                f"```json\n{json.dumps(_math, indent=2)}\n```\n\n"
                "### Audit Entry\n\n"
                f"```json\n{json.dumps(_audit, indent=2)}\n```\n\n"
                "### Full Chat Log (truncated)\n\n"
                f"```\n{_full_log[:2000]}\n```"
            )
        except Exception as _e:
            ag_result = mo.md(f"**AutoGen error:** `{_e}`")
    else:
        ag_result = mo.md(
            "_Select AutoGen, enter API key, and click Run Analysis_"
        )
    ag_result
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
