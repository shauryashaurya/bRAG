import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 007 - licensed to quill

    ## Structured data
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 0. Setup and suchlike""")
    return


@app.cell
def _():
    from google import genai
    return (genai,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    import os
    from langchain_google_genai import (
        GoogleGenerativeAIEmbeddings,
        ChatGoogleGenerativeAI,
    )
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_experimental.agents.agent_toolkits import (
        create_pandas_dataframe_agent,
    )
    return ChatGoogleGenerativeAI, create_pandas_dataframe_agent, os


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


@app.cell
def _(os):
    # Path relative to the marimo notebook
    DATA_DIR = os.path.join(os.path.dirname("./"), "songs_data")
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, os, pd):
    # Load CSVs
    df_songs = pd.read_csv(os.path.join(DATA_DIR, "songs.csv"))
    return (df_songs,)


@app.cell
def _(ChatGoogleGenerativeAI):
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return (llm,)


@app.cell
def _(create_pandas_dataframe_agent, df_songs, llm):
    # Agent
    music_historian = create_pandas_dataframe_agent(
        llm=llm, df=df_songs, verbose=True, allow_dangerous_code=False
    )
    return (music_historian,)


@app.cell
def _(music_historian):
    # Usage
    music_historian.invoke("How many songs were released before 1990?")
    return


@app.cell
def _(DATA_DIR, os, pd):
    df_albums = pd.read_csv(os.path.join(DATA_DIR, "albums.csv"))
    df_artists = pd.read_csv(os.path.join(DATA_DIR, "artists.csv"))
    return df_albums, df_artists


@app.cell
def _(create_pandas_dataframe_agent, df_albums, df_artists, df_songs, llm):
    music_historian = create_pandas_dataframe_agent(
        llm=llm,
        df=[df_songs, df_albums, df_artists],  # multi-df mode
        verbose=True,
        allow_dangerous_code=False,
    )
    return (music_historian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
  
    ## How the Agent Actually Works  
  
    `create_pandas_dataframe_agent` does **not** send all your data to Google:  
  
    1. It sends the dataframe **schema + a few sample rows** to the LLM as context  
    2. The LLM writes **Python/pandas code** to answer the question  
    3. That code runs **locally** in a Python REPL  
    4. If the result needs interpretation, only the **output** (a number, a small table) goes back to the LLM  
    5. The LLM returns a final natural language answer  
  
    So for `"How many songs were released before 1990?"`, Google only ever sees the column names, a sample row, and the scalar result (e.g. `342`) — not your full CSV.  
  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What Does Get Sent to Google  (or any other AI Service)

    - Column names and dtypes  
    - A few sample rows (usually 5)  
    - The generated Python code  
    - The output of that code  
    - Your question
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Approaches to Minimize LLM Exposure  """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **1. Restrict the schema — only pass columns the agent needs**  
    ```python  
    df_safe = df_songs[["title", "releaseDate", "duration"]]  # drop sensitive cols  

    music_historian = create_pandas_dataframe_agent(llm=llm, df=df_safe, ...)  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **2. Pre-aggregate before passing**  
    ```python  
    # Instead of raw transaction rows, send a summary  
    df_summary = df_songs.groupby(df_songs["releaseDate"].str[:4])["id"].count()  

    music_historian = create_pandas_dataframe_agent(llm=llm, df=df_summary, ...)  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **3. Anonymize or mask sensitive columns**  
    ```python  
    df_safe = df_songs.copy()  
    df_safe["artistIDs"] = df_safe["artistIDs"].apply(lambda x: "REDACTED")  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **4. Reduce sample rows sent to the LLM**  
    ```python  
    # The agent uses df.head() internally — you can truncate  
    df_sample = df_songs.sample(3)  # only 3 representative rows  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Scaling to Enterprise  """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""There are three main architectural approaches, in order of increasing robustness:"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Tier 1 — Local/Private LLM (no data leaves your infra)**  
    Swap Gemini for a self-hosted model like Llama 3 or Mistral via Ollama:  
    ```python  
    from langchain_ollama import ChatOllama  
    llm = ChatOllama(model="llama3")  
    # Everything runs on-premise, zero external API calls  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Tier 2 — Text-to-SQL instead of Text-to-Pandas**  
    Instead of sending dataframes, store data in a database and have the LLM generate SQL queries. Only the query and a small result set ever leave:  
    ```python  
    from langchain_community.utilities import SQLDatabase  
    from langchain_community.agent_toolkits import create_sql_agent  

    db = SQLDatabase.from_uri("postgresql://user:pass@localhost/musicdb")  
    agent = create_sql_agent(llm=llm, db=db, verbose=True)  
    # LLM only sees table schema + query result, never raw rows  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Tier 3 — Google's own enterprise stack (if you're staying in GCP)**  
    Use **BigQuery + Gemini** natively — the data never leaves Google's infrastructure and you get IAM, audit logs, and VPC controls:  
    ```python  
    from langchain_google_community import BigQueryVectorStore  
    # Query runs inside GCP, only the answer surfaces to the user  
    ```  
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary Recommendation  

    | Scale | Approach | Data Exposure |  
    |---|---|---|  
    | Prototype | pandas agent (current) | Schema + sample rows only |  
    | Internal tool | Text-to-SQL on local DB | Schema + query result only |  
    | Enterprise / sensitive data | Self-hosted LLM (Ollama) | Zero — fully on-premise |  
    | GCP enterprise | BigQuery + Gemini | Stays within your GCP project |  

    The biggest single win for most teams is moving from pandas agent >> **Text-to-SQL**, since it naturally limits what the LLM touches and scales to billions of rows without memory issues.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##... and that's all there is to it!""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
