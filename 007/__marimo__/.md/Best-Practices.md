---
title: Best Practices
marimo-version: 0.14.16
width: medium
layout_file: layouts/Start-Here.slides.json
---

```python {.marimo}
import marimo as mo
```

# Approach, Best Practices, and Use Cases
<!---->
## 1. The Core Approach: How to Build Agentic Apps
Unlike traditional "Chatbots," Agentic apps are designed to **act**. The development lifecycle follows a "Loop" rather than a "Line."

### A. The Perception-Reason-Act (PRA) Loop
1.  **Perception:** The agent receives a goal and gathers context (e.g., querying your `songs.csv` or `artists.csv`).
2.  **Reasoning:** The LLM uses a framework (like **LangChain** or **DSPy**) to break the goal into sub-tasks.
3.  **Acting:** The agent selects a "Tool" (a Python function, an API, or a database query) to execute a sub-task.
4.  **Observation:** The agent looks at the output of the tool. If there is an error, it loops back to "Reasoning" to fix it.

### B. Orchestration vs. Optimization
*   **Orchestration (LangChain):** Focus on the "Tools." You define the sequence of events and how the agent interacts with external systems.
*   **Optimization (DSPy):** Focus on the "Logic." You treat the prompt as a program, allowing the framework to optimize the instructions based on examples.
<!---->
## 2. Best Practices for Reliability
Agentic systems can be unpredictable. Follow these rules to keep them stable:

1.  **Limit the "Blast Radius":** Never give an agent full access to a database. Give it specific, read-only tools or narrow API endpoints.
2.  **Iterative Depth:** Set `max_iterations`. An agent should not be allowed to loop forever if it gets stuck.
3.  **Human-in-the-Loop (HITL):** For high-stakes actions (like deleting data or sending emails), require a human to click "Approve" before the agent proceeds.
4.  **Structured Output:** Use Pydantic or DSPy Signatures to ensure the agent returns data in a format (JSON/CSV) that your application can actually use.
5.  **Small Tools are Better:** Instead of one "Do Everything" tool, give the agent five "Do One Thing Well" tools. This reduces model confusion.
<!---->
## 3. High-Value Use Cases
Not every problem needs an Agent. Use Agentic AI when the task requires **multi-step reasoning** or **interaction with dynamic data.**

| Use Case Category | Description | Example |
| :--- | :--- | :--- |
| **Data Auditing** | Finding anomalies across multiple disconnected sources. | Finding "Temporal Paradoxes" in music release dates vs. artist birthdays. |
| **Dynamic Research** | Browsing the web or a codebase to answer a "How-to" question. | A bot that reads a GitHub repo and explains how to integrate its API. |
| **Self-Healing Pipelines** | Code or ETL jobs that fix themselves when they fail. | An agent that catches a SQL error, re-writes the query, and tries again. |
| **Personalized Synthesis** | Creating content that blends multiple complex data points. | Generating a marketing blurb for a song by cross-referencing its genre history and award wins. |
| **Workflow Automation** | Moving a task through several different software platforms. | An agent that reads an email, finds the invoice in a PDF, and logs it in QuickBooks. |
<!---->
## 4. When NOT to use Agents
*   **Speed is Critical:** Agents are slower than standard LLM calls because of the reasoning loops.
*   **Predictability is Mandatory:** If the output must be identical every time (e.g., tax calculations), use standard code.
*   **Simple Retrieval:** If a user just wants to find a specific row in a CSV, a simple search box is better than an agent.