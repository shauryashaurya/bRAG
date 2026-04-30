import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 007 - licensed to quill

    ## The very first agentic app
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 0. Setup and suchlike""")
    return


@app.cell
def _():
    import os
    import marimo as mo
    return mo, os


@app.cell
def _():
    from google import genai
    return (genai,)


@app.cell
def _():
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.agents.agent_toolkits import create_python_agent
    from langchain_experimental.tools import PythonREPLTool
    return ChatGoogleGenerativeAI, PythonREPLTool, create_python_agent


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
def _(genai, os):
    # 0. List available models using new SDK
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return (client,)


@app.cell
def _(client):
    print("Available models:")
    for model in client.models.list():
        print(model.name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 1. Brain  

    The LLM (OpenAI, Gemini, Claude, etc. etc.) can act as a *brain* in that it can quickly convert your ask into a series of 'tasks' that will be executed by one or more 'tools'
    """
    )
    return


@app.cell
def _(ChatGoogleGenerativeAI):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 2. Agent with Python Tool

    Agents are basically ways in which an LLM can "call" or "invoke" a specialized tool.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    an **Agent** is the "Thinker" and a **Tool** is the "Doer."

    *   **The Agent:** A Large Language Model (LLM) that has been given a goal. It can reason, plan, and decide which actions to take.
    *   **The Tool:** A specific function or script that the Agent can call to interact with the real world (e.g., a calculator, a web search, or a database query).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Analogies:  
    **1. The Chef (Agent) and the Kitchen Utensils (Tools)**  
    * The Chef knows how to make a cake (Goal).    
    * However, the Chef needs bowl, whisk, oven (Tools) to get the job done.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **2. The Research Agent**  
    *   **Goal:** "Find the current stock price of Apple and tell me if it is up or down."  
    *   **Tool:** `Google_Search_API`  
    *   **Process:** The Agent realizes it doesn't know today's prices. It uses the Search Tool, reads the result, and then calculates the difference.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **3. The Data Cleaning Agent**  
    *   **Goal:** "Find all songs in this CSV that are over 10 minutes long."   
    *   **Tool:** `Python_Pandas_Interpreter`   
    *   **Process:** The Agent writes a small piece of code, sends it to the Python Tool, gets the list of songs back, and presents them to you.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **4. The Support Agent**   
    *   **Goal:** "Refund this customer if their order is late."   
    *   **Tool:** `Database_Query_Tool`   
    *   **Process:** The Agent uses the tool to look up the delivery date. If the date is past the deadline, it triggers a second tool called `Issue_Refund_API`.
    """
    )
    return


@app.cell
def _(PythonREPLTool, create_python_agent, llm):
    coding_agent = create_python_agent(
        llm=llm, tool=PythonREPLTool(), verbose=True, allow_dangerous_code=False
    )
    return (coding_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3. Execution""")
    return


@app.cell
def _(coding_agent):
    # Example Task
    coding_agent.run(
        "Print all the Fibonacci numbers till the 10th Fibonacci number using a function."
    )
    return


@app.cell
def _(coding_agent):
    coding_agent.run(
        "Create a list of 10 random integers and sort them in descending order"
    )
    return


@app.cell
def _(coding_agent):
    coding_agent.run("Generate a plot of a sine wave and save it as wave.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##... and that's all there is to it!""")
    return


@app.cell
def _(mo):
    mo.md(rf"""# Tool - Agent logic flow""")
    return


@app.cell
def _(mo):
    mo.md(
        rf"""
    ```mermaid  
    flowchart TD
        A["User Input: Can you fix the typo in my database?"] --> B["THE AGENT"]

        B -->|"Calls tool"| C["READ_DB_TOOL"]
        C -->|"Returns data"| B

        B --> D["Reasoning: Typo found. Now I will correct it"]

        D -->|"Calls tool"| E["WRITE_DB_TOOL"]
        E -->|"Confirms fix"| B

        B --> F["Final Output: Typo has been fixed!"]
    ```
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
