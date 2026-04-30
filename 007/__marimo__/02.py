import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 007 - licensed to quill

    ## The 10 minute RAG...
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
    return (
        CharacterTextSplitter,
        ChatGoogleGenerativeAI,
        ChatPromptTemplate,
        GoogleGenerativeAIEmbeddings,
        InMemoryVectorStore,
        RunnablePassthrough,
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
    mo.md(r"""# 1. Process Text""")
    return


@app.cell
def _(CharacterTextSplitter):
    text = "The 'rocket-ship' project is a collaborative initiative for technical publishing."
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = splitter.create_documents([text])
    return (docs,)


@app.cell
def _(mo):
    mo.md(r"""# 2. Vector DB (In-memory)""")
    return


@app.cell
def _(GoogleGenerativeAIEmbeddings, InMemoryVectorStore, docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    return (retriever,)


@app.cell
def _(mo):
    mo.md(r"""# 3. RAG Chain — using LCEL (LangChain Expression Language)""")
    return


@app.cell
def _(ChatGoogleGenerativeAI):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return (llm,)


@app.cell
def _(ChatPromptTemplate):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the context below to answer the question. "
                "If you don't know, say so.\nContext: {context}",
            ),
            ("human", "{input}"),
        ]
    )
    return (prompt,)


@app.cell
def _(RunnablePassthrough, StrOutputParser, llm, prompt, retriever):
    qa_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return (qa_chain,)


@app.cell
def _(qa_chain):
    # Usage — now returns a plain string directly
    result = qa_chain.invoke("What is project rocket-ship?")
    print(result)
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
