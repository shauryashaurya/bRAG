import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import json
    import glob
    import pandas as pd
    return glob, mo, os, pd


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
    # load all CSVs from ./ifrs17_data
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
        options=_opts, label="Select table", value=_opts[0]
    )
    return dfs, file_select


@app.cell
def _(file_select):
    file_select
    return


@app.cell
def _(dfs, file_select, mo):
    if file_select.value in dfs:
        data_view = mo.ui.table(dfs[file_select.value].head(20))
    else:
        data_view = mo.md(
            "Place CSV files in `./ifrs17_data/` matching sample-ifrs17schema.yaml."
        )
    data_view
    return


@app.cell
def _(mo, os):
    import dspy

    _key = os.environ.get("GOOGLE_API_KEY", "")
    if _key:
        _lm = dspy.LM("gemini/gemini-2.5-flash", api_key=_key)
        dspy.configure(lm=_lm)
        lm_ready = True
    else:
        lm_ready = False
    mo.md(f"DSPy LM ready: `{lm_ready}` -- enter Gemini API key above first.")
    return dspy, lm_ready


@app.cell
def _(lm_ready):
    lm_ready
    return


@app.cell
def _(mo):
    run_t1 = mo.ui.run_button(label="Run Tutorial 1")
    return (run_t1,)


@app.cell
def _(run_t1):
    run_t1
    return


@app.cell
def _(dspy, lm_ready, mo, run_t1):
    if run_t1.value and lm_ready:

        class QA(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()

        _qa = dspy.Predict(QA)
        _res = _qa(
            question="What is the Contractual Service Margin and how is it released under IFRS17?"
        )
        t1_out = mo.md(f"**Answer:** {_res.answer}")
    else:
        t1_out = mo.md("_Enter API key and click Run Tutorial 1_")
    t1_out
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    run_t2 = mo.ui.run_button(label="Run Tutorial 2")
    return (run_t2,)


@app.cell
def _(run_t2):
    run_t2
    return


@app.cell
def _(dspy, lm_ready, mo, run_t2):
    _docs = [
        "IFRS17 measures insurance contracts using the CSM (Contractual Service Margin).",
        "The CSM represents unearned profit deferred and released over the coverage period.",
        "Insurance revenue is recognized as coverage services are provided each period.",
        "Revenue equals expected claims incurred plus risk adjustment release plus CSM release.",
        "A loss component is recognized immediately when a contract group is onerous.",
        "The PAA (Premium Allocation Approach) simplifies measurement for short-duration contracts.",
        "Reinsurance held is measured separately and cannot offset direct insurance liabilities.",
    ]


    def _retrieve(q, docs=_docs):
        _terms = set(q.lower().split())
        return (
            "\n".join(d for d in docs if any(t in d.lower() for t in _terms))
            or docs[0]
        )


    if run_t2.value and lm_ready:

        class RAG(dspy.Signature):
            context = dspy.InputField(desc="Retrieved IFRS17 passages")
            question = dspy.InputField()
            answer = dspy.OutputField()

        _rag = dspy.Predict(RAG)
        _q = "How is insurance revenue recognized under IFRS17?"
        _ctx = _retrieve(_q)
        _res = _rag(context=_ctx, question=_q)
        t2_out = mo.md(
            f"**Context used:**\n```\n{_ctx}\n```\n\n**Answer:** {_res.answer}"
        )
    else:
        t2_out = mo.md("_Enter API key and click Run Tutorial 2_")
    t2_out
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    run_t3 = mo.ui.run_button(label="Run Tutorial 3")
    return (run_t3,)


@app.cell
def _(run_t3):
    run_t3
    return


@app.cell
def _(dfs, dspy, file_select, lm_ready, mo, run_t3):
    if run_t3.value and lm_ready:

        class Extract(dspy.Signature):
            text = dspy.InputField(
                desc="Raw text or data row describing an IFRS17 policy group"
            )
            policy_group_id = dspy.OutputField()
            csm_opening = dspy.OutputField(
                desc="Opening CSM balance as a plain number"
            )
            csm_closing = dspy.OutputField(
                desc="Closing CSM balance as a plain number"
            )
            insurance_revenue = dspy.OutputField(
                desc="Insurance revenue for the period as a plain number"
            )

        _ext = dspy.Predict(Extract)

        # use loaded data if available, else use sample text
        if file_select.value in dfs:
            _row = dfs[file_select.value].iloc[0].to_dict()
            _txt = str(_row)
        else:
            _txt = "Policy group PG001 opened the quarter with CSM of 1,400,000 and closed with CSM 1,200,000. Insurance revenue recognized was 345,000 for the period ending 2024-03-31."

        _res = _ext(text=_txt)
        t3_out = mo.md(
            f"**Input text:**\n```\n{_txt[:300]}\n```\n\n"
            f"**policy_group_id:** `{_res.policy_group_id}`\n\n"
            f"**csm_opening:** `{_res.csm_opening}`\n\n"
            f"**csm_closing:** `{_res.csm_closing}`\n\n"
            f"**insurance_revenue:** `{_res.insurance_revenue}`"
        )
    else:
        t3_out = mo.md("_Enter API key and click Run Tutorial 3_")
    t3_out
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    chat_q = mo.ui.text(
        label="Follow-up question", placeholder="How is CSM released into revenue?"
    )
    send_btn = mo.ui.run_button(label="Send")
    return chat_q, send_btn


@app.cell
def _(chat_q):
    chat_q
    return


@app.cell
def _(send_btn):
    send_btn
    return


@app.cell
def _(chat_q):
    chat_q.value
    return


@app.cell
def _(chat_q, dspy, lm_ready, mo, send_btn):
    # seed the history with one prior exchange
    _seed_history = (
        "user: What is the Contractual Service Margin?\n"
        "ai: The CSM is the unearned profit deferred at contract inception and released "
        "over the coverage period as insurance services are provided."
    )

    if send_btn.value and lm_ready and chat_q.value:

        class ChatTurn(dspy.Signature):
            history = dspy.InputField(
                desc="Prior conversation turns, one per line"
            )
            message = dspy.InputField()
            reply = dspy.OutputField()

        _chat = dspy.Predict(ChatTurn)
        _res = _chat(history=_seed_history, message=chat_q.value)
        t4_out = mo.md(
            f"**Seeded history:**\n```\n{_seed_history}\n```\n\n"
            f"**You:** {chat_q.value}\n\n"
            f"**AI:** {_res.reply}"
        )
    else:
        t4_out = mo.md("_Enter API key, type a question, and click Send_")
    t4_out
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    run_t5 = mo.ui.run_button(label="Run Tutorial 5")
    return (run_t5,)


@app.cell
def _(run_t5):
    run_t5
    return


@app.cell
def _(dspy, lm_ready, mo, run_t5):
    if run_t5.value and lm_ready:

        class Plan(dspy.Signature):
            task = dspy.InputField()
            steps = dspy.OutputField(
                desc="Numbered list of IFRS17 analysis steps, one per line, max 4 steps"
            )

        class Execute(dspy.Signature):
            step = dspy.InputField()
            result = dspy.OutputField(
                desc="One-sentence IFRS17 insight for this step"
            )

        _planner = dspy.Predict(Plan)
        _executor = dspy.Predict(Execute)

        _task = "Analyze CSM movement drivers for a general insurance portfolio under GMM"
        _steps = _planner(task=_task).steps.strip().split("\n")
        _outputs = []
        for _s in _steps[:3]:  # cap at 3 to limit API calls
            if _s.strip():
                _r = _executor(step=_s).result
                _outputs.append(f"**Step:** {_s}\n\n**Insight:** {_r}")

        t5_out = mo.md(
            "\n\n---\n\n".join(_outputs) if _outputs else "No steps returned."
        )
    else:
        t5_out = mo.md("_Enter API key and click Run Tutorial 5_")
    t5_out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
