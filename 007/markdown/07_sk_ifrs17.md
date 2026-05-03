---
title: 07 Sk Ifrs17
marimo-version: 0.14.16
width: medium
---

```python {.marimo}
import marimo as mo
import os
import json
import glob
import pandas as pd
```

```python {.marimo}
api_key_input = mo.ui.text(label="Enter Gemini API Key", kind="password")
```

```python {.marimo}
api_key_input
```

```python {.marimo}
os.environ["GOOGLE_API_KEY"] = api_key_input.value
```

```python {.marimo}

```

```python {.marimo}
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
```

```python {.marimo}
file_select
```

```python {.marimo}
if file_select.value in dfs:
    data_view = mo.ui.table(dfs[file_select.value].head(20))
else:
    data_view = mo.md(
        "Place CSV files in `./ifrs17_data/` matching sample-ifrs17schema.yaml."
    )
data_view
```

```python {.marimo}
# verify packages are importable and key is present
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.google.google_ai import (
        GoogleAIChatCompletion,
    )
    from semantic_kernel.functions.kernel_arguments import KernelArguments
    from semantic_kernel.contents import ChatHistory

    sk_imported = True
except ImportError as _e:
    sk_imported = False
    sk = None
    GoogleAIChatCompletion = None
    KernelArguments = None
    ChatHistory = None

_key = os.environ.get("GOOGLE_API_KEY", "")
sk_ready = sk_imported and bool(_key)
mo.md(
    f"SK imported: `{sk_imported}` -- SK ready: `{sk_ready}`\n\n"
    "Install: `pip install semantic-kernel[google]`"
)
```

```python {.marimo}
# factory: fresh kernel per tutorial to avoid function name conflicts on re-run
def make_kernel():
    if not sk_ready:
        return None
    _kernel = sk.Kernel()
    _svc = GoogleAIChatCompletion(
        gemini_model_id="gemini-2.0-flash",
        api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )
    _kernel.add_service(_svc)
    return _kernel
```

```python {.marimo}

```

```python {.marimo}
run_t1 = mo.ui.run_button(label="Run Tutorial 1")
```

```python {.marimo}
run_t1
```

```python {.marimo}
if run_t1.value and sk_ready:
    _kernel = make_kernel()
    _fn = _kernel.add_function(
        function_name="explain",
        plugin_name="ifrs17_basics",
        prompt="Explain {{$topic}} in exactly 2 concise lines.",
    )
    _res = await _kernel.invoke(
        _fn, KernelArguments(topic="IFRS17 Contractual Service Margin")
    )
    t1_out = mo.md(f"**Result:** {str(_res)}")
else:
    t1_out = mo.md("_Enter API key and click Run Tutorial 1_")
t1_out
```

```python {.marimo}

```

```python {.marimo}
run_t2 = mo.ui.run_button(label="Run Tutorial 2")
```

```python {.marimo}
run_t2
```

````python {.marimo}
_docs = [
    "The CSM is the deferred profit recognized over the coverage period of a contract.",
    "Insurance revenue is recognized as the insurer provides coverage services each period.",
    "Revenue comprises expected claims incurred, risk adjustment release, and CSM release.",
    "Onerous contracts recognize a loss component immediately; CSM cannot go negative.",
    "The PAA is a simplified model permitted for contracts with coverage periods of 12 months or less.",
]


def _retrieve(q, docs=_docs):
    _terms = set(q.lower().split())
    _hits = [d for d in docs if any(t in d.lower() for t in _terms)]
    return "\n".join(_hits) if _hits else docs[0]


if run_t2.value and sk_ready:
    _kernel = make_kernel()
    _fn = _kernel.add_function(
        function_name="rag_answer",
        plugin_name="ifrs17_rag",
        prompt="Context:\n{{$context}}\n\nQuestion: {{$question}}\nAnswer:",
    )
    _q = "How is insurance revenue recognized and what does it include?"
    _ctx = _retrieve(_q)
    _res = await _kernel.invoke(
        _fn, KernelArguments(context=_ctx, question=_q)
    )
    t2_out = mo.md(
        f"**Context:**\n```\n{_ctx}\n```\n\n**Answer:** {str(_res)}"
    )
else:
    t2_out = mo.md("_Enter API key and click Run Tutorial 2_")
t2_out
````

```python {.marimo}

```

```python {.marimo}
run_t3 = mo.ui.run_button(label="Run Tutorial 3")
```

```python {.marimo}
run_t3
```

````python {.marimo}
_prompt = (
    "Extract the following fields from the text and return ONLY valid JSON with no extra text.\n"
    "Fields: policy_group_id (string), csm_closing (number), insurance_revenue (number).\n"
    "Text: {{$input}}\n"
    "JSON:"
)

if run_t3.value and sk_ready:
    _kernel = make_kernel()
    _fn = _kernel.add_function(
        function_name="extract",
        plugin_name="ifrs17_extract",
        prompt=_prompt,
    )
    if file_select.value in dfs:
        _txt = str(dfs[file_select.value].iloc[0].to_dict())[:400]
    else:
        _txt = "Policy group PG002 reports a closing CSM of 850000 and insurance revenue of 210000 for Q2 2024."

    _res = await _kernel.invoke(_fn, KernelArguments(input=_txt))
    _raw = str(_res).strip().lstrip("```json").rstrip("```").strip()
    try:
        _parsed = json.loads(_raw)
        t3_out = mo.md(
            f"**Parsed:**\n```json\n{json.dumps(_parsed, indent=2)}\n```"
        )
    except Exception:
        t3_out = mo.md(f"**Raw (parse failed):**\n```\n{_raw}\n```")
else:
    t3_out = mo.md("_Enter API key and click Run Tutorial 3_")
t3_out
````

```python {.marimo}

```

```python {.marimo}
followup_q = mo.ui.text(
    label="Follow-up question",
    placeholder="How does CSM release affect profit?",
)
run_t4 = mo.ui.run_button(label="Send")
```

```python {.marimo}
followup_q
```

```python {.marimo}
run_t4
```

```python {.marimo}
if run_t4.value and sk_ready and followup_q.value:
    # seed a prior exchange
    _hist = ChatHistory()
    _hist.add_user_message(
        "What is the Contractual Service Margin under IFRS17?"
    )
    _hist.add_assistant_message(
        "The CSM is the unearned profit on an insurance contract, deferred at inception "
        "and released over the coverage period as insurance services are provided."
    )

    _kernel = make_kernel()
    _fn = _kernel.add_function(
        function_name="chat",
        plugin_name="ifrs17_chat",
        prompt="{{$history}}\nUser: {{$message}}\nAssistant:",
    )
    _res = await _kernel.invoke(
        _fn,
        KernelArguments(history=str(_hist), message=followup_q.value),
    )
    t4_out = mo.md(
        f"**Seeded exchange:** CSM definition question and answer.\n\n"
        f"**You:** {followup_q.value}\n\n"
        f"**AI:** {str(_res)}"
    )
else:
    t4_out = mo.md("_Enter API key, type a follow-up, and click Send_")
t4_out
```

```python {.marimo}

```

```python {.marimo}
run_t5 = mo.ui.run_button(label="Run Tutorial 5")
```

```python {.marimo}
run_t5
```

```python {.marimo}
if run_t5.value and sk_ready:
    _kernel = make_kernel()
    _plan_fn = _kernel.add_function(
        function_name="plan",
        plugin_name="orchestrator",
        prompt="Break this IFRS17 task into exactly 3 analysis steps, one per line:\n{{$task}}",
    )
    _work_fn = _kernel.add_function(
        function_name="work",
        plugin_name="orchestrator",
        prompt="Provide one concise IFRS17 insight for this analysis step:\n{{$step}}",
    )

    _task = "Analyze CSM release patterns across a life insurance portfolio under GMM"
    _steps_raw = await _kernel.invoke(_plan_fn, KernelArguments(task=_task))
    _steps = str(_steps_raw).strip().split("\n")
    _outputs = []
    for _s in _steps[:3]:
        if _s.strip():
            _r = await _kernel.invoke(_work_fn, KernelArguments(step=_s))
            _outputs.append(f"**Step:** {_s}\n\n**Insight:** {str(_r)}")
    t5_out = mo.md(
        "\n\n---\n\n".join(_outputs) if _outputs else "No steps returned."
    )
else:
    t5_out = mo.md("_Enter API key and click Run Tutorial 5_")
t5_out
```

```python {.marimo}

```