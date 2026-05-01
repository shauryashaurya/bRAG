import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 007 - licensed to quill

    ## memory
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 0. Setup and suchlike""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from google import genai
    return (genai,)


@app.cell
def _():
    import pandas as pd
    return


@app.cell
def _():
    import os
    from langchain_google_genai import (
        GoogleGenerativeAIEmbeddings,
        ChatGoogleGenerativeAI,
    )
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_experimental.agents.agent_toolkits import (
        create_pandas_dataframe_agent,
    )
    return (
        ChatGoogleGenerativeAI,
        ChatMessageHistory,
        ChatPromptTemplate,
        MessagesPlaceholder,
        RunnableWithMessageHistory,
        StrOutputParser,
        os,
    )


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
def _(mo):
    mo.md(
        r"""
    # 1. LLM   
    Same idea as before...
    """
    )
    return


@app.cell
def _(ChatGoogleGenerativeAI):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return (llm,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 2. Prompt   
    `MessagesPlaceholder` handles history injection cleanly
    """
    )
    return


@app.cell
def _(ChatPromptTemplate, MessagesPlaceholder):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{human_input}"),
        ]
    )
    return (prompt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 3. Chain   
  
    LangChain Expression Language (**LCEL**) is a declarative, "pipe-based" syntax (using |) designed to compose LangChain components: **prompts, models, retrievers, and parsers** into a single 'chain' that can be easy to reason about.
    """
    )
    return


@app.cell
def _(StrOutputParser, llm, prompt):
    # express the chain as LCEL
    chain = prompt | llm | StrOutputParser()
    return (chain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    LCEL uses **pipe operator** `|`, borrowed from Unix shell.    
    Each component's output becomes the next component's input:     
     
    ```     
    prompt            # formats your variables into a full prompt message     
      |     
    llm               # receives the prompt, returns an AIMessage object     
      |     
    StrOutputParser() # pulls the plain string out of the AIMessage object     
    ```     
     
    Without the parser the chain returns an `AIMessage(content="...")` object. With it, you get a plain `str`.     
     
    It's equivalent to:     
    ```python     
    message = prompt.invoke({"human_input": "..."})     
    ai_message = llm.invoke(message)     
    result = StrOutputParser().invoke(ai_message)     
    ```     
     
    Just written as one line instead of three.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 4. Session store    
    One `ChatMessageHistory` per session_id
    """
    )
    return


@app.cell
def _(ChatMessageHistory):
    store = {}


    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    return (get_session_history,)


@app.cell
def _():
    # 5. Wrap with history
    return


@app.cell
def _(RunnableWithMessageHistory, chain, get_session_history):
    chat_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="human_input",
        history_messages_key="chat_history",
    )
    return (chat_chain,)


@app.cell
def _():
    # Usage — session_id lets you run multiple independent conversations
    config = {"configurable": {"session_id": "user_123"}}
    return (config,)


@app.cell
def _(chat_chain, config):
    chat_chain.invoke(
        {"human_input": "My name is Wha? Who? Chekit Chekit Slim Shady."},
        config=config,
    )
    return


@app.cell
def _(chat_chain, config):
    chat_chain.invoke({"human_input": "What is my name?"}, config=config)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 5. Enterprise idiosyncracies""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Problem Space     
    ```     
    Single user, demo    : dict in RAM is fine     
    Multi user, prod     : need persistence, isolation, TTL, scale     
    Enterprise           : + compliance, encryption, audit, PII handling     
    ```     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Storage Tiers     

    | Tier | What | When |     
    |---|---|---|     
    | In-process dict | RAM, lost on restart | Dev/demo only |     
    | Redis | Fast, TTL built-in, ephemeral | Session memory, high traffic |     
    | Postgres | Durable, queryable, row-level security | Audit trails, long-term history, multi-tenant |     
    | Elasticsearch | Full-text + vector search over history | Semantic recall, compliance search, analytics |     
    | Vector DB | Semantic search over history | "What did user say about X 3 weeks ago" |     
    | Hybrid | Redis (hot) + Postgres (cold) + Elastic (search) | Most production enterprise systems |     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Key Enterprise Concerns""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **1. TTL / Expiry**     
    ```python     
    RedisChatMessageHistory(session_id=id, ttl=3600)  # auto-expire after 1hr     
    # Postgres: run a cron DELETE WHERE created_at < NOW() - INTERVAL '30 days'     
    # Elastic: use ILM (Index Lifecycle Management) policies to auto-expire indices     
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **2. PII Scrubbing — strip before storage**     
    ```python     
    import re     
    def scrub(text):     
        text = re.sub(r'\b\d{16}\b', '[CARD]', text)      # credit cards     
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)         # emails     
        return text     

    history.add_user_message(scrub(user_input))     
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **3. Message Windowing — never send full history to LLM**     
    ```python     
    # Only last k messages go into prompt     
    recent = history.messages[-10:]     
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **4. Summarization — compress old history**     
    ```python     
    # When history exceeds threshold, summarize old turns with LLM     
    # keep summary + recent N messages     
    # common pattern: summary = llm.invoke(f"Summarize: {old_messages}")     
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **5. Audit Log — separate from working memory**     
    ```python     
    # Postgres: append-only table, never updated, only inserted     
    # Elastic: write-once index with ILM, fully searchable for compliance     
    # Both fulfill eDiscovery and regulatory requirements     
    ```     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### LangChain Integrations     

    ```python     
    # Redis     
    from langchain_community.chat_message_histories import RedisChatMessageHistory     
    RedisChatMessageHistory(session_id=id, url="redis://localhost:6379", ttl=3600)     

    # Postgres     
    from langchain_community.chat_message_histories import PostgresChatMessageHistory     
    PostgresChatMessageHistory(session_id=id, connection_string="postgresql://...")     

    # Elasticsearch     
    from langchain_community.chat_message_histories import ElasticsearchChatMessageHistory     
    ElasticsearchChatMessageHistory(     
        es_url="https://localhost:9200",     
        index="chat-history",     
        session_id=id     
    )     
    ```     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Architecture Pattern (Prod)     

    ```
    User request     
        |     
        V
    Redis (hot)              <-- LLM reads last N messages, fast, TTL-managed     
        |     
        V     
    Postgres (cold)          <-- full history, append-only, audit log     
        |                        row-level security for multi-tenant isolation     
        V     
    Elasticsearch (search)   <-- indexes full history for compliance search,     
        |                        analytics, and semantic/full-text recall     
        V     
    Vector DB (semantic)     <-- long-term episodic recall across sessions     
    ```

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### When to Use Each Store     

    | Store | Strengths | Avoid when |     
    |---|---|---|     
    | Redis | Sub-ms reads, TTL, pub/sub | Need durability or complex queries |     
    | Postgres | ACID, joins, row-level security, JSON | High-frequency reads at scale |     
    | Elasticsearch | Full-text search, analytics, ILM, scalable | Simple CRUD, tight budget |     
    | Vector DB | Semantic similarity search | Exact match or structured queries |     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Memory Scopes to Track Separately     

    | Scope | Key | Example |     
    |---|---|---|     
    | Session | session_id | Single conversation |     
    | User | user_id | Preferences across sessions |     
    | Org/Tenant | org_id | Shared context in a team |     
    | Agent run | run_id | What agent did in one task |     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Managed Options (avoid building it yourself)     

    | Tool | What it handles |     
    |---|---|     
    | Mem0 | User memory layer, API-first |     
    | Zep | Session + long-term memory, OSS |     
    | LangGraph Store | Built-in persistence for LangGraph agents |     
    | Elastic Cloud | Managed Elasticsearch with ILM + encryption |     
    | Redis Cloud | Managed Redis with TTL + encryption |     
    | Supabase / RDS | Managed Postgres with row-level security |     

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Compliance Quick checks     

    * PII never stored raw     
    * Encryption at rest + in transit     
    * Per-tenant data isolation (row-level security in Postgres, index-per-tenant in Elastic)     
    * TTL policy defined: Redis TTL, Postgres cron, Elastic ILM     
    * Audit log immutable and queryable (Postgres append-only, Elastic write-once)     
    * Full-text compliance search enabled (Elasticsearch)     
    * Right-to-erasure: can delete by user_id across all stores     
    * No history crosses tenant boundary
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
