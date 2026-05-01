import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    layout_file="layouts/Agentic-AI-CheatSheet.slides.json",
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    # Agentic AI & LLMs  

    ## 1. Essential LLM Vocabulary  
    *   **LLM (Large Language Model):** The probabilistic engine that predicts the next "token" (word fragment).  
    *   **Token:** The basic unit of text processed by an LLM (~0.75 words).  
    *   **Context Window:** The "RAM" of the LLM. The total amount of text (input + output) it can consider at one time.  
    *   **Temperature:** A setting (0.0 to 1.0) that controls randomness.  
        *   **0.0:** Deterministic, best for coding/math.  
        *   **0.7+:** Creative, best for brainstorming/writing.  
    *   **Hallucination:** When a model generates factually incorrect information with high confidence.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Agentic AI Concepts  
    *   **Agent:** An LLM wrapped in a loop that can use **Tools** to change the state of the world.  
    *   **Orchestration:** The management of multiple tools, prompts, and memory (e.g., **LangChain**).  
    *   **Optimization:** Programmatically tuning the prompt/logic to improve performance (e.g., **DSPy**).  
    *   **ReAct (Reason + Act):** A prompting pattern: *Thought → Action → Observation → Repeat.*  
    *   **Human-in-the-Loop (HITL):** A safety checkpoint where the agent pauses for human approval.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. The Flow of an Agentic Application  
    The lifecycle of a single user request follows this circular path:  

    1.  **Input:** User provides a goal (e.g., *"Audit the music catalog"*).  
    2.  **Planning:** The Brain (LLM) breaks the goal into sub-tasks.  
    3.  **Tool Selection:** The Agent selects the best tool for the first sub-task.  
    4.  **Execution:** The tool runs (e.g., a Python script queries `songs.csv`).  
    5.  **Observation:** The Agent reads the tool's output.  
    6.  **Reflection:** The Agent asks: *"Did this solve the goal?"*  
        *   *If No:* Loop back to Step 2 with new information.  
        *   *If Yes:* Move to Step 7.  
    7.  **Final Response:** The Agent synthesizes all observations into a clear answer for the user.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Default Implementation Approach (The "Golden Path")  

    When building your first agentic app, follow this 4-step implementation sequence:  

    ### Phase 1: Define the "World" (The Data)  
    *   Clean your data (CSVs, PDFs).  
    *   Define what the Agent *can* and *cannot* see.  

    ### Phase 2: Build the "Tools" (The Skills)  
    *   Write Python functions for specific tasks (e.g., `calculate_duration()`, `search_artist_by_id()`).  
    *   **Crucial:** Write clear docstrings for every function. The LLM uses these to understand what the tool does.  

    ### Phase 3: Choose the "Framework" (The Logic)  
    *   **Use LangChain** if your app needs to interact with many APIs/Databases.  
    *   **Use DSPy** if your app is focused on high-accuracy data extraction or transformation.  

    ### Phase 4: Grounding & Evaluation (The Safety)  
    *   **RAG:** Ensure the model looks at your files before answering to prevent hallucinations.  
    *   **Iteration Limit:** Set `max_iterations=5` to prevent the agent from getting stuck in an infinite logic loop.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Quick Comparison: Framework Syntax  

    | Feature | **LangChain** (Orchestration) | **DSPy** (Optimization) |  
    | :--- | :--- | :--- |  
    | **Logic** | Defined by "Chains" of tools. | Defined by "Signatures" (In/Out). |  
    | **Prompting** | Manual (You write the prompt). | Automatic (The system "compiles" it). |  
    | **Data Flow** | Sequential (Tool A → Tool B). | Functional (Input → Program → Output). |  
    | **Best For** | Complex multi-tool apps. | High-reliability logic & pipelines. |
    """
    )
    return


if __name__ == "__main__":
    app.run()
