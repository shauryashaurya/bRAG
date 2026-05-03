---
title: 08 Autogen Ifrs17
marimo-version: 0.14.16
width: medium
---

```python {.marimo}
import marimo as mo
import os
import json
import glob
import io
import re
import pandas as pd
from contextlib import redirect_stdout
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
try:
    from autogen import (
        AssistantAgent,
        UserProxyAgent,
        GroupChat,
        GroupChatManager,
    )

    ag_imported = True
except ImportError:
    AssistantAgent = UserProxyAgent = GroupChat = GroupChatManager = None
    ag_imported = False

_key = os.environ.get("GOOGLE_API_KEY", "")
# AutoGen 0.2 routes to Gemini via litellm; "gemini/" prefix triggers litellm's Gemini backend
ag_cfg = {
    "config_list": [{"model": "gemini/gemini-2.0-flash", "api_key": _key}],
    "cache_seed": None,
}
ag_ready = ag_imported and bool(_key)

mo.md(
    f"AutoGen imported: `{ag_imported}` -- AutoGen ready: `{ag_ready}`\n\n"
    "Install: `pip install pyautogen`"
)
```

```python {.marimo}

```

```python {.marimo}
run_t1 = mo.ui.run_button(label="Run Tutorial 1")
```

```python {.marimo}
run_t1
```

````python {.marimo}
if run_t1.value and ag_ready:
    _assistant = AssistantAgent(
        "assistant",
        llm_config=ag_cfg,
        system_message="You are an IFRS17 technical expert. Answer concisely.",
    )
    _user = UserProxyAgent(
        "user",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: True,
    )
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _user.initiate_chat(
            _assistant,
            message="Explain the CSM rollforward under IFRS17 GMM in 3 bullet points.",
        )
    t1_out = mo.md(f"```\n{_buf.getvalue()}\n```")
else:
    t1_out = mo.md("_Enter API key and click Run Tutorial 1_")
t1_out
````

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
    "CSM is the unearned profit on insurance contracts, released as services are provided.",
    "Insurance revenue = expected claims incurred + risk adjustment release + CSM release.",
    "Reinsurance held is measured separately; it cannot net against direct contract liabilities.",
    "Loss component: when a group of contracts is onerous, a loss is recognized immediately.",
]

if run_t2.value and ag_ready:
    _q = "What is insurance revenue under IFRS17 and what drives it?"
    _terms = set(_q.lower().split())
    _ctx = (
        "\n".join(d for d in _docs if any(t in d.lower() for t in _terms))
        or _docs[0]
    )
    _msg = f"Context (retrieved):\n{_ctx}\n\nQuestion: {_q}"

    _assistant = AssistantAgent(
        "assistant",
        llm_config=ag_cfg,
        system_message="Answer using only the provided context. Be concise.",
    )
    _user = UserProxyAgent(
        "user",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: True,
    )
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _user.initiate_chat(_assistant, message=_msg)
    t2_out = mo.md(
        f"**Context injected:**\n```\n{_ctx}\n```\n\n**Chat output:**\n```\n{_buf.getvalue()}\n```"
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
_schema = '{"policy_group_id": "", "csm_closing": 0, "insurance_revenue": 0}'

if run_t3.value and ag_ready:
    if file_select.value in dfs:
        _row_txt = str(dfs[file_select.value].iloc[0].to_dict())[:400]
    else:
        _row_txt = "Policy group PG003 closed with CSM 975000 and revenue 288000 for period 2024-06-30."

    _msg = (
        f"Extract and return ONLY valid JSON matching this schema:\n{_schema}\n\n"
        f"Source text: {_row_txt}"
    )

    _assistant = AssistantAgent(
        "extractor",
        llm_config=ag_cfg,
        system_message="Return only valid JSON. No explanation, no markdown fences.",
    )
    _user = UserProxyAgent(
        "user",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: True,
    )
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _user.initiate_chat(_assistant, message=_msg)

    _raw = _buf.getvalue()
    _match = re.search(r"\{[^{}]+\}", _raw, re.DOTALL)
    if _match:
        try:
            _parsed = json.loads(_match.group())
            t3_out = mo.md(
                f"**Extracted:**\n```json\n{json.dumps(_parsed, indent=2)}\n```"
            )
        except Exception:
            t3_out = mo.md(f"**Raw (parse failed):**\n```\n{_raw}\n```")
    else:
        t3_out = mo.md(f"**Raw output (no JSON found):**\n```\n{_raw}\n```")
else:
    t3_out = mo.md("_Enter API key and click Run Tutorial 3_")
t3_out
````

```python {.marimo}

```

```python {.marimo}
run_t4 = mo.ui.run_button(label="Run Tutorial 4 (2 turns)")
```

```python {.marimo}
run_t4
```

````python {.marimo}
if run_t4.value and ag_ready:
    _assistant = AssistantAgent(
        "assistant",
        llm_config=ag_cfg,
        system_message="You are an IFRS17 expert. Be concise.",
    )
    _user = UserProxyAgent(
        "user",
        code_execution_config=False,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
    )
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _user.initiate_chat(
            _assistant, message="What is the CSM under IFRS17?"
        )
        _user.send(
            "How is the CSM released into insurance revenue?", _assistant
        )
    t4_out = mo.md(f"```\n{_buf.getvalue()}\n```")
else:
    t4_out = mo.md("_Enter API key and click Run Tutorial 4_")
t4_out
````

```python {.marimo}

```

```python {.marimo}
run_t5 = mo.ui.run_button(label="Run Tutorial 5")
```

```python {.marimo}
run_t5
```

````python {.marimo}
if run_t5.value and ag_ready:
    _term = lambda msg: "TERMINATE" in msg.get("content", "").upper()

    _planner = AssistantAgent(
        "planner",
        llm_config=ag_cfg,
        system_message=(
            "You break IFRS17 analysis tasks into numbered steps. "
            "When the plan is complete, say TERMINATE."
        ),
    )
    _analyst = AssistantAgent(
        "analyst",
        llm_config=ag_cfg,
        system_message="You execute IFRS17 analysis steps and report findings concisely.",
    )
    _reviewer = AssistantAgent(
        "reviewer",
        llm_config=ag_cfg,
        system_message=(
            "You review IFRS17 analysis for accuracy against the standard. "
            "When review is complete, say TERMINATE."
        ),
    )
    _proxy = UserProxyAgent(
        "coordinator",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=_term,
    )

    _group = GroupChat(
        agents=[_proxy, _planner, _analyst, _reviewer],
        messages=[],
        max_round=8,
    )
    _manager = GroupChatManager(groupchat=_group, llm_config=ag_cfg)

    _buf = io.StringIO()
    with redirect_stdout(_buf):
        _proxy.initiate_chat(
            _manager,
            message="Analyze insurance revenue drivers for a life insurance GMM portfolio.",
        )
    # cap output to avoid overwhelming the cell
    _output = _buf.getvalue()[:3000]
    t5_out = mo.md(f"```\n{_output}\n```")
else:
    t5_out = mo.md("_Enter API key and click Run Tutorial 5_")
t5_out
````

```python {.marimo}

```