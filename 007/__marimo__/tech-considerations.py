import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/tech-considerations.slides.json",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## AI App vs Agentic App

    | Type | Definition | Loop? | Tools? |
    |---|---|---|---|
    | AI App | Single LLM call, deterministic flow | No | Optional |
    | Agentic App | LLM decides next action, dynamic flow | Yes | Required |

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## When to Use Which

    **Use an AI app when:**
    - Input/output is well-defined (summarize, classify, extract, translate)
    - Latency matters
    - You need predictable cost
    - Errors are easy to catch

    **Use an agentic app when:**
    - The path to the answer is unknown upfront
    - Multiple tools/data sources must be orchestrated
    - Tasks require planning, retry, or self-correction
    - The problem has sub-tasks that depend on prior results

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Core Agentic Patterns

    | Pattern | How it works | Good for |
    |---|---|---|
    | ReAct | Reason then Act, loop until done | General tool use |
    | Plan-and-Execute | Plan all steps upfront, then execute | Long multi-step tasks |
    | Reflection | Agent critiques its own output, iterates | Quality-sensitive outputs |
    | Multi-agent | Specialized agents collaborate | Complex domain separation |
    | Human-in-the-loop | Agent pauses for human approval | High-stakes actions |

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Frameworks

    | Framework | Style | Best for |
    |---|---|---|
    | LangChain | Modular, chain-based | Rapid prototyping, RAG |
    | LangGraph | Graph/state-machine | Complex multi-step agents |
    | DSPy | Declarative, auto-optimizes prompts | Research, prompt tuning |
    | CrewAI | Role-based multi-agent | Team-style agent workflows |
    | AutoGen (Microsoft) | Conversational multi-agent | Code generation, debate |
    | Haystack | Pipeline-based | Production NLP/RAG |
    | Semantic Kernel | SDK, enterprise-first | .NET/Python enterprise |
    | Agno | Lightweight, fast | Simple single agents |

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Memory Types

    | Type | What it stores | Example |
    |---|---|---|
    | In-context | Current conversation window | Chat history |
    | External | Vector DB, SQL, key-value | Long-term user facts |
    | Episodic | Past agent runs/traces | What the agent did before |
    | Procedural | How to do things (tools, prompts) | Tool definitions |

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tool / Data Patterns

    | Pattern | Description |
    |---|---|
    | RAG | Retrieve docs, inject into prompt |
    | Text-to-SQL | LLM writes SQL, DB executes |
    | Text-to-Pandas | LLM writes Python, runs locally |
    | Function calling | LLM triggers structured API calls |
    | Code interpreter | LLM writes + runs arbitrary code |
    | Browser/computer use | LLM controls UI directly |

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Orchestration Topologies

    ```
    Single agent     :  LLM -> tools -> answer
    Sequential chain :  A -> B -> C -> answer
    Router           :  input -> classifier -> specialist agent
    Hierarchical     :  orchestrator -> [worker1, worker2, worker3]
    Parallel         :  input -> [agent A, agent B] -> merge -> answer
    ```

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Key Production Concerns

    | Concern | Options |
    |---|---|
    | Observability | LangSmith, Arize, Helicone |
    | Cost control | Cache, limit tool calls, smaller models for sub-tasks |
    | Latency | Parallelism, streaming, async |
    | Safety | Tool sandboxing, human-in-the-loop, output validators |
    | Data privacy | On-prem LLMs, schema-only exposure, text-to-SQL |
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
