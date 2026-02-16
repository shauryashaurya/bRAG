# The Evolution of RAG, Memory & Agentic AI: 2024–2026
## A Technical Comparison of Innovations and a Framework for Unified Evaluation

---

## Part 1: The Three Eras at a Glance

The period from 2024 to early 2026 represents a dramatic acceleration in how AI systems retrieve, reason over, and remember information. What began as incremental improvements to the basic retrieve-then-generate pipeline has transformed into a fundamentally different paradigm — one where AI systems manage their own context, reason recursively, and operate as autonomous agents with persistent memory.

The evolution can be summarised as three overlapping waves:

**2024 — The Adaptive & Structured Wave:** Systems learned *when* to retrieve, *how much* to retrieve, and began using structured representations (graphs, trees, hierarchies) rather than flat chunks.

**2025 — The Agentic & Memory Wave:** Retrieval became embedded inside autonomous agent loops. Memory systems matured from simple stores into managed, evolving architectures. Protocols standardised tool and agent communication.

**2026 — The Recursive & Context-as-Code Wave:** The prompt itself became a programmable object. Models learned to recursively decompose and process their own context through code execution, breaking free of fixed context windows entirely. RAG evolved into a general "Context Engine."

---

## Part 2: 2024 Innovations in Detail

### 2.1 Self-RAG — The Model Decides When to Retrieve

**Paper:** Asai et al., ICLR 2024 Oral · [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
**Code:** [github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)

**What changed from before:** Traditional RAG always retrieves — every query triggers a search regardless of whether external information is needed. Self-RAG trains the LLM itself to emit special *reflection tokens* that signal (a) whether retrieval is needed at all, (b) whether retrieved passages actually support the claim, and (c) whether the overall output is useful.

**How it works:** The model is fine-tuned with four types of reflection tokens — `[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]` — that it generates inline with its response. When the model generates `[Retrieve=Yes]`, retrieval is triggered. It then evaluates relevance and support before committing to the generated text.

**Key benefit:** Eliminates the "always-retrieve" overhead and the "noise from irrelevant retrieval" problem. On knowledge-intensive benchmarks, Self-RAG outperforms both vanilla RAG (which retrieves everything) and no-retrieval baselines.

**Limitation:** Requires fine-tuning the base model with reflection tokens — you can't just plug this into an off-the-shelf API model.

---

### 2.2 CRAG — Corrective Retrieval Augmented Generation

**Paper:** Yan et al., 2024 · [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
**Code:** [github.com/HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG)

**What changed from before:** Where Self-RAG teaches the *generator* to judge retrieval quality, CRAG adds a separate *lightweight evaluator* that sits between retriever and generator. This makes it plug-and-play — no model fine-tuning needed.

**How it works:** After retrieval, a small evaluator model scores document quality, returning a confidence signal. Three paths follow:
- **Correct** (high confidence): Documents pass through directly, with irrelevant sentences stripped via a decompose-then-recompose algorithm.
- **Incorrect** (low confidence): The system falls back to web search for fresh information.
- **Ambiguous** (medium confidence): Both the original documents and web search results are combined.

**Key differentiation from Self-RAG:** Self-RAG requires training the LLM itself with special tokens. CRAG is a modular wrapper — any retriever, any generator, any evaluator can be swapped. This makes it far more practical for production where one use API-based models (GPT-4, Claude) that one can't fine-tune.

**Key benefit:** Robust against retrieval failure. When the retriever returns garbage, the system self-corrects rather than hallucinating.

---

### 2.3 RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

**Paper:** Sarthi et al., ICLR 2024 · [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
**Code:** [github.com/parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)

**What changed from before:** All prior RAG systems retrieved *flat chunks* — fixed-size segments of the original document. This means they can only ever retrieve local, detail-level information. RAPTOR creates a hierarchy where you can retrieve at any level of abstraction.

**How it works (the recursion):**
1. Split documents into leaf-level text chunks.
2. Cluster semantically similar chunks using Gaussian Mixture Models.
3. Summarise each cluster with an LLM, producing parent nodes.
4. Repeat: cluster the summaries, summarise the clusters, creating grandparent nodes.
5. Continue until a single root summary exists.

At query time, one can retrieve from any level — specific leaf chunks for detail questions, mid-level summaries for thematic questions, or root-level summaries for "what is this corpus about?" questions.

**Key differentiation:** RAPTOR is the first system to apply recursion at *indexing time* (building the tree offline). This contrasts sharply with 2026's RLMs, which apply recursion at *inference time* (processing context online). RAPTOR's tree is static once built; RLMs generate their decomposition strategy dynamically for each query.

**Key benefit:** Multi-hop questions that span multiple documents finally work well, because mid-level summaries capture cross-document themes that flat chunks miss entirely.

**Limitation:** Expensive to build the tree (many LLM calls during indexing). The tree is static — adding new documents requires partial or full rebuilding.

---

### 2.4 GraphRAG — Knowledge Graph-Enhanced Retrieval

**Paper:** Edge et al. (Microsoft), 2024 · [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
**Code:** [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag) (20K+ stars)

**What changed from before:** RAPTOR creates abstraction hierarchies but doesn't capture *relationships* between entities. GraphRAG builds an explicit knowledge graph from the corpus — entities as nodes, relationships as edges — then uses community detection to create hierarchical summaries.

**How it works (two-stage pipeline):**
1. **Indexing:** An LLM reads every chunk and extracts entities (people, places, concepts) and their relationships. These form a knowledge graph. The Leiden algorithm detects communities (clusters of related entities). Each community gets a summary.
2. **Querying:** For *local* queries (specific facts), the system retrieves relevant entity subgraphs. For *global* queries (thematic, holistic), it uses a map-reduce strategy across community summaries.

**Key differentiation from RAPTOR:** RAPTOR's hierarchy is about *abstraction level* — the same content at different granularities. GraphRAG's structure is about *relationships* — it captures that "Entity A relates to Entity B via Relationship X," which RAPTOR cannot. GraphRAG excels at questions like "What are the main themes across all documents?" or "How does X relate to Y?" — questions where RAPTOR's tree retrieval would struggle because the answer spans many branches.

**Key benefit:** Handles "global sensemaking" queries that defeat all flat-chunk RAG systems. For the first time, you can ask questions about the *entire corpus* rather than just retrievable chunks.

**Limitation:** Expensive indexing (many LLM calls to extract entities). The knowledge graph can be noisy. Microsoft estimates indexing costs at ~$1–5 per million tokens.

---

### 2.5 HippoRAG — Neurobiologically Inspired Memory for RAG

**Paper:** Gutiérrez et al., NeurIPS 2024 · [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)
**Code:** [github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) (3.7K+ stars)

**What changed from before:** GraphRAG uses generic community detection. HippoRAG models itself explicitly on how the human hippocampus and neocortex collaborate to store and retrieve memories. It's the first system to bring *neuroscience architecture* into RAG.

**How it works:** Three components mirror brain regions:
- **LLM (as neocortex):** Processes and encodes new information, extracting knowledge graph triples.
- **Knowledge Graph (as hippocampal index):** Stores extracted relationships as a sparse, associative index.
- **Personalised PageRank (as retrieval):** When a query arrives, the system identifies entry points in the graph and runs Personalised PageRank to find associated information — mimicking how the hippocampus uses pattern completion to retrieve memories.

**Key differentiation from GraphRAG:** GraphRAG's community summaries are a *static, top-down* organisation. HippoRAG's PageRank retrieval is *dynamic and query-driven* — the same graph produces different retrieval paths depending on the query's entry point. It's also dramatically cheaper: 10–30× cheaper and 6–13× faster than iterative retrieval methods like IRCoT.

**HippoRAG 2 (Feb 2025, arXiv:2502.14802):** Extended with continual learning — the system can update its knowledge graph as new information arrives, without reprocessing the entire corpus.

---

### 2.6 LightRAG — Lightweight Graph-Based Retrieval

**Paper:** Guo et al. (HKUDS), EMNLP 2025 · [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
**Code:** [github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (30K+ stars)

**What changed:** GraphRAG is powerful but expensive and complex. LightRAG achieves similar benefits with a simpler, faster approach — making graph-based RAG practical for real deployments.

**How it works:** Dual-level retrieval combining:
- **Low-level:** Retrieve specific entities and their immediate relationships (like standard GraphRAG).
- **High-level:** Retrieve broader relational patterns and themes from the graph structure.

Both are backed by vector search over graph-derived key-value pairs, making retrieval fast. Crucially, it supports *incremental updates* — new documents can be indexed without rebuilding the entire graph.

**Key differentiation from GraphRAG:** LightRAG is simpler (no community detection algorithm), faster (incremental indexing), and cheaper to run. GraphRAG is more powerful for truly global queries but overkill for many use cases. LightRAG has become the pragmatic default for teams who want graph-enhanced RAG without Microsoft-scale infrastructure.

---

### 2.7 Adaptive-RAG — Complexity-Aware Routing

**Paper:** Jeong et al., 2024 · [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)
**Code:** [github.com/starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)

**What changed:** Self-RAG and CRAG decide whether to *use* retrieved documents. Adaptive-RAG goes further by deciding *what kind of retrieval strategy* to use based on query complexity.

**How it works:** A small classifier assesses each incoming query:
- **Simple query** → No retrieval, answer from parametric knowledge.
- **Moderate query** → Single-step retrieval (standard RAG).
- **Complex query** → Multi-hop iterative retrieval with chain-of-thought reasoning.

**Key benefit:** Optimises cost/latency vs. quality. Simple factual questions don't waste resources on multi-hop retrieval. Complex research questions get the full treatment. This routing pattern is now standard in production RAG systems.

---

### 2.8 DSPy — Programming (Not Prompting) Language Models

**Paper:** Khattab et al. (Stanford), ICLR 2024 Spotlight · [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
**Code:** [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)

**What changed:** Every system above requires hand-crafted prompts. DSPy replaces prompting with *programming* — you declare what the LLM should do (signatures like `"question -> answer"`) and DSPy automatically optimises the prompts, few-shot examples, and even fine-tuning data.

**How it works:** You build modules (Predict, ChainOfThought, ReAct, etc.) and compose them into pipelines. An optimizer (MIPRO, BootstrapFewShot, etc.) then automatically tunes the pipeline by searching over prompt variations and evaluating on a metric you define.

**Key benefit for RAG:** Instead of hand-tuning prompts for your retriever, reranker, and generator separately, you declare the pipeline and let DSPy optimise it end-to-end. This is especially powerful for Self-RAG and CRAG patterns, where the decision logic is complex.

**Why this matters for 2026:** DSPy is now the backbone of the RLM implementation — the `dspy.RLM` module uses DSPy's programming model to orchestrate recursive context processing. DSPy evolved from a prompt optimizer into the infrastructure layer for recursive AI.

---

### 2.9 MCP — Model Context Protocol

**Announcement:** Anthropic, November 2024
**Spec:** [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)

**What changed:** Before MCP, every AI tool integration was custom. Connecting Claude to a database required different code than connecting GPT-4 to the same database. MCP creates a universal, open standard (JSON-RPC 2.0) for connecting any LLM to any data source or tool.

**Key benefit:** Reduces integration complexity from M×N (M models × N tools) to M+N. A tool built once works with every MCP-compatible model.

**Why this matters for the evolution:** MCP is the *infrastructure layer* that makes Agentic RAG practical. When an agent needs to dynamically choose between a vector database, a SQL query, a web search, and a code interpreter, MCP provides the standardised interface for all of them.

---

## Part 3: 2025 Innovations in Detail

### 3.1 Agentic RAG — Retrieval Under Agent Control

**Survey:** Singh et al., Jan 2025 · [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
**Implementations:** LangGraph, CrewAI, Haystack, RAGFlow, and many others

**What changed from 2024:** In 2024, systems like Self-RAG and CRAG added decision points to the RAG pipeline. In 2025, those decision points became full *agent loops* — autonomous systems that can plan multi-step retrieval strategies, use tools, reflect on intermediate results, and iterate until satisfied.

**The key distinction (three levels):**

| Level | Description | Example |
|-------|-------------|---------|
| **Naive RAG** | Retrieve once → Generate | Basic vector search + LLM |
| **Workflow RAG** | Pre-defined multi-step pipeline | CRAG, Self-RAG, FLARE |
| **Agentic RAG** | Agent autonomously decides strategy | A-RAG, MA-RAG, RAGFlow |

A truly agentic system satisfies three principles: (1) **Autonomous Strategy** — the agent decides *how* to retrieve, not a hardcoded workflow; (2) **Iterative Execution** — it can retry, refine, and expand its search; (3) **Interleaved Tool Use** — retrieval is just one tool among many (SQL queries, web search, code execution, API calls).

**Concrete example:** A user asks "How did our Q3 revenue compare to the industry average?" An agentic RAG system would: (1) query the internal database for Q3 numbers, (2) web-search for industry benchmarks, (3) compare the two, (4) realise it needs Q2 numbers for context, (5) query again, (6) synthesise a final answer with citations.

**Why this matters:** RAG stopped being a "pipeline" and became a "capability" that agents invoke as needed. The RAG system doesn't own the workflow — the agent does.

---

### 3.2 DeepSeek-R1 — Reasoning Through Reinforcement Learning

**Paper:** DeepSeek AI, Jan 2025 · [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
**Code:** [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

**What changed:** OpenAI's O1 (Sep 2024) showed that chain-of-thought reasoning at inference time dramatically improves performance, but O1 was proprietary. DeepSeek-R1 achieved comparable results using pure RL on an open-source base model — and released everything under MIT license.

**How it works:** Starting from DeepSeek-V3-Base, they applied GRPO (Group Relative Policy Optimization) with only outcome-based rewards (is the final answer correct?). No human-annotated reasoning chains. The model *naturally emerged* with self-verification, reflection, and strategy adaptation — what the team called "aha moments" in training.

**Why this matters for RAG:** DeepSeek-R1 and its distilled variants (1.5B to 70B) bring reasoning capabilities to open-source models that can be deployed locally. This means your RAG system's generator can now reason over retrieved documents much more effectively — decomposing multi-hop questions, cross-referencing sources, detecting contradictions.

---

### 3.3 Mem0 — Production-Ready Persistent Memory

**Paper:** Chhikara et al., Apr 2025 · [arXiv:2504.19413](https://arxiv.org/abs/2504.19413)
**Code:** [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)

**What changed from before:** Earlier memory systems (MemGPT, MemoryBank) were research prototypes. Mem0 is engineered for production — it extracts, consolidates, and retrieves memories from conversations with 91% faster response times and 90% lower token usage compared to full-context approaches.

**How it works (two variants):**
- **Mem0 Base:** Extracts atomic facts from conversations. When new information contradicts old memories, it consolidates (updates, merges, or deletes). Uses vector search for retrieval.
- **Mem0 Graph:** Extends Base with a relationship graph — not just "user likes Italian food" but "user likes Italian food → prefers vegetarian → is allergic to shellfish" as connected nodes.

**Key differentiation from MemGPT/Letta:** MemGPT manages memory like an operating system (paging context in and out). Mem0 manages memory like a *knowledge base* — it's not about managing the context window, but about maintaining a persistent, evolving representation of what's been learned across many conversations.

**26% improvement** over OpenAI's memory on the LOCOMO benchmark (long-conversation memory evaluation).

---

### 3.4 A-MEM — Self-Organizing Zettelkasten Memory

**Paper:** Xu et al., Feb 2025 · [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
**Code:** [github.com/agentica-project/A-MEM](https://github.com/agentica-project/A-MEM)

**What changed:** Mem0 extracts facts and stores them. A-MEM goes further — memories aren't just stored, they *self-organise*. Inspired by the Zettelkasten note-taking method, memories form dynamic links and evolve their own structure over time.

**Key differentiation from Mem0:** Mem0 relies on predefined extraction rules. A-MEM's memories are *agentic* — the system autonomously decides how to link, group, and restructure memories as new information arrives. This means the memory structure adapts to the user's actual usage patterns rather than following a fixed schema.

---

### 3.5 MemOS — An Operating System for AI Memory

**Paper:** May 2025 · [arXiv:2505.22101](https://arxiv.org/abs/2505.22101)
**Code:** [github.com/MemTensor/MemOS](https://github.com/MemTensor/MemOS)

**What changed:** Where Mem0 is a memory layer and A-MEM adds self-organisation, MemOS provides the full *governance infrastructure* — lifecycle management, scheduling, access control, and cross-agent memory sharing.

**The memory hierarchy comparison:**

| System | Analogy | Manages |
|--------|---------|---------|
| **MemGPT** (2023) | Virtual memory paging | Context window overflow |
| **Mem0** (2024–25) | Personal knowledge base | Facts & relationships |
| **A-MEM** (2025) | Self-organising notebook | Evolving memory structure |
| **MemOS** (2025) | Operating system kernel | Memory lifecycle, sharing, governance |

MemOS is where you'd build if you have *multiple agents* that need to share, update, and govern a common memory pool — with policies about who can read what, when memories expire, and how conflicts between agent memories are resolved.

---

### 3.6 A2A — Agent-to-Agent Protocol

**Announcement:** Google, April 2025
**Spec:** [github.com/google/A2A](https://github.com/google/A2A)

**What changed:** MCP (2024) standardised how models talk to *tools*. A2A standardises how agents talk to *each other*. They're complementary: MCP is the agent-to-tool protocol, A2A is the agent-to-agent protocol.

**How it works:** Each agent publishes an "Agent Card" describing its capabilities. When Agent A needs help with a subtask, it discovers Agent B's card, negotiates a task, and communicates via structured, async messages. This works across vendors — a Claude-based agent can delegate to a GPT-based agent seamlessly.

---

### 3.7 The RAG-to-Context-Engine Transformation

A critical shift in 2025 thinking (articulated well by the RAGFlow team) is that RAG is evolving from a specific pattern ("retrieve documents, augment prompt, generate") into a general **Context Engine** — the infrastructure layer responsible for providing any AI system with the right context at the right time.

This means:
- RAG isn't just for question-answering anymore — it's the memory/knowledge layer for agents.
- The "R" in RAG expands from document retrieval to include SQL queries, API calls, memory lookups, web searches, and tool results.
- The "A" (augmented) becomes dynamic context management rather than simple prompt stuffing.
- The "G" (generation) becomes any downstream task, not just text generation.

This reframing is important because it explains why "RAG is dead" claims are wrong — RAG isn't dying, it's *generalising*.

---

## Part 4: 2026 Innovations (Emerging)

### 4.1 Recursive Language Models (RLMs) — The Paradigm Shift

**Paper:** Zhang & Khattab (MIT CSAIL), Dec 2025 · [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
**Code:** [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
**DSPy integration:** `dspy.RLM` module
**Blog:** [primeintellect.ai/blog/rlm](https://www.primeintellect.ai/blog/rlm) — "Recursive Language Models: the paradigm of 2026"

**What fundamentally changed:** Every system discussed so far — Self-RAG, CRAG, GraphRAG, RAPTOR, Agentic RAG — still puts the retrieved context *into the prompt*. The LLM must "read" all the context in its token window. RLMs break this assumption entirely.

**The core idea:** The prompt is treated as a *Python variable* stored in a REPL environment, not as tokens in the context window. The LLM writes Python code to inspect, search, partition, and recursively process this variable. When it needs to understand a portion of the context, it spawns a *sub-LLM call* with just that portion — a fresh instance with a clean context window.

```
Traditional RAG:   [Prompt + Retrieved Docs + Query] → LLM → Answer
RLM:               [Query + Code Environment] → LLM → writes Python →
                      Python examines context variable →
                      sub-LLM(chunk_1), sub-LLM(chunk_2), ... →
                      LLM aggregates results → Answer
```

**Concrete example:** You have 10 million tokens of legal documents and ask "What are the key liability clauses across all contracts?"
- **Traditional RAG:** Retrieves top-k chunks (maybe 20), missing most contracts entirely.
- **GraphRAG:** Indexes all entities and relationships (expensive), retrieves relevant subgraphs.
- **RLM:** The LLM writes code to list all files, batch-process each contract through a sub-LLM with the prompt "Extract liability clauses," then aggregates the results programmatically.

**Performance (from the paper):**
- **BrowseComp-Plus** (6–11M token inputs): RLM(GPT-5) achieved 91.33% — base GPT-5 scored 0%.
- **OOLONG-Pairs** (information-dense reasoning): 58% F1 — base GPT-5 scored 0.04%.
- RLM-Qwen3-8B (the first natively recursive open model): 28.3% average improvement over base Qwen3-8B.

**Why "paradigm of 2026":** RLMs represent a fundamentally different relationship between LLMs and context:

| Aspect | Traditional RAG | RLM |
|--------|----------------|-----|
| Context location | In the prompt (token space) | In a variable (code space) |
| Context limit | Bounded by context window | Unbounded (limited by compute budget) |
| Retrieval strategy | Pre-defined (vector search, graph search) | Learned/generated per query |
| Processing model | Read all context → generate | Write code → inspect selectively → recurse |
| Adaptability | Fixed pipeline | Fully dynamic, query-dependent |

**The RL training angle:** The trajectory of how an RLM chooses to decompose and recurse is entirely *learnable*. Prime Intellect and others are actively training models with RL to improve their recursive strategies — the same approach that made DeepSeek-R1 good at reasoning can make models good at *context management*.

---

### 4.2 A-RAG — Hierarchical Agentic Retrieval (Feb 2026)

**Paper:** [arXiv:2602.03442](https://arxiv.org/html/2602.03442v1) (very recent)

**What changed:** Fills the gap between Workflow RAG (pre-defined steps) and full autonomous agents. A-RAG gives the agent a hierarchical set of retrieval interfaces (chunk-level, entity-level, community-level, web search) and lets it *choose autonomously* which to use and in what order.

**Key insight:** True agentic RAG requires three things that prior systems lacked simultaneously: (1) autonomous strategy selection, (2) iterative execution with reflection, and (3) interleaved tool use. A-RAG satisfies all three while remaining grounded in structured retrieval.

---

### 4.3 RAG as Context Engine — The Convergence

By early 2026, the boundaries between categories are dissolving:
- **RAG systems** use agents for retrieval strategy (Agentic RAG).
- **Agent systems** use RAG for grounding and memory (Agent + Context Engine).
- **Memory systems** use graph structures from GraphRAG (Mem0 Graph, HippoRAG 2).
- **Recursive systems** use all of the above through code execution (RLMs).

The emerging architecture looks like:

```
┌─────────────────────────────────────────────────┐
│                 RLM / Agent Loop                 │
│   (Recursive decomposition, code generation,    │
│    strategy selection, self-reflection)          │
├─────────────────────────────────────────────────┤
│              Context Engine (RAG 3.0)            │
│  ┌──────────┬──────────┬──────────┬──────────┐  │
│  │  Vector  │  Graph   │  Memory  │   Tool   │  │
│  │  Search  │  Search  │  Store   │   Calls  │  │
│  │ (chunks) │(entities)│ (Mem0)   │  (MCP)   │  │
│  └──────────┴──────────┴──────────┴──────────┘  │
├─────────────────────────────────────────────────┤
│          Communication Protocols                 │
│    MCP (model↔tool)    A2A (agent↔agent)        │
├─────────────────────────────────────────────────┤
│         Reasoning Engine (DeepSeek-R1+)          │
│   (CoT, self-verification, reflection)          │
└─────────────────────────────────────────────────┘
```

---

## Part 5: Side-by-Side Technical Comparison

### 5.1 Retrieval Strategy Comparison

| System | Retrieval Type | When to Retrieve | What's Retrieved | Index Structure |
|--------|---------------|------------------|------------------|-----------------|
| Naive RAG | Always, single-pass | Every query | Top-k flat chunks | Flat vector index |
| Self-RAG | Adaptive (model decides) | When reflection token triggers | Relevant passages | Flat vector index |
| CRAG | Adaptive (evaluator decides) | Always, but corrects after | Filtered, corrected docs | Flat + web fallback |
| RAPTOR | Always, multi-level | Every query | Chunks at chosen abstraction level | Hierarchical tree |
| GraphRAG | Always, structure-aware | Every query | Entity subgraphs + community summaries | Knowledge graph + communities |
| HippoRAG | Always, associative | Every query | PageRank-activated graph regions | KG + PageRank |
| LightRAG | Always, dual-level | Every query | Low-level entities + high-level patterns | Lightweight KG + vectors |
| Adaptive-RAG | Routed by complexity | Depends on query class | Nothing / chunks / multi-hop chains | Flat + routing classifier |
| Agentic RAG | Agent-controlled | Agent decides per step | Anything (vector, graph, web, SQL, API) | Multiple, tool-based |
| RLM | Code-generated | LLM writes retrieval code | Programmatically selected context slices | Context-as-variable + sub-LLMs |

### 5.2 Memory System Comparison

| System | Memory Model | Persistence | Update Strategy | Multi-Agent |
|--------|-------------|-------------|-----------------|-------------|
| MemGPT/Letta | OS-style virtual memory | Session (paged) | LLM manages own context | No (single agent) |
| MemoryBank | Flat store + forgetting | Cross-session | Ebbinghaus decay curve | No |
| Mem0 Base | Fact extraction + consolidation | Permanent | Extract → consolidate → store | Via integrations |
| Mem0 Graph | Facts + relationship graph | Permanent | Graph updates on new info | Via integrations |
| HippoRAG | Neocortex + hippocampal index | Permanent | Continual learning (v2) | No |
| A-MEM | Self-organising Zettelkasten | Permanent | Autonomous restructuring | No |
| MemOS | Full memory OS | Permanent | Scheduled lifecycle management | Yes (governed sharing) |

### 5.3 Reasoning & Decomposition Comparison

| System | Decomposition Method | Recursive? | Learnable? | Context Limit |
|--------|---------------------|-----------|-----------|---------------|
| Chain-of-Thought | Linear step-by-step | No | No (prompting) | Context window |
| Tree of Thoughts | Branching + backtracking | Tree-structured | No | Context window |
| Graph of Thoughts | Arbitrary DAG | Graph-structured | No | Context window |
| ReAct | Thought→Action→Observe loop | Iterative (not recursive) | No | Context window |
| Reflexion | Try→Fail→Reflect→Retry | Iterative with memory | No | Context window |
| DeepSeek-R1 | RL-emergent CoT | No (but long) | Yes (RL) | Context window |
| RLM | Code-generated recursive decomposition | Truly recursive | Yes (RL-trainable) | **Unbounded** |

---

## Part 6: Designing a Unified Comparison Code Repository

The following is a conceptual architecture for a single codebase that lets you run, compare, and benchmark all these approaches side by side. We are not building this now — this is the blueprint.

### 6.1 Design Principles

1. **Common interface:** Every system implements the same `query(question, corpus) → answer + metadata` interface so results are directly comparable.
2. **Shared corpus:** All systems index and query the *same* document collection, eliminating data-related variance.
3. **Shared evaluation:** Every answer is scored with the same metrics (accuracy, faithfulness, latency, cost, token usage).
4. **Modular components:** Retrievers, generators, evaluators, and memory stores are interchangeable — you can combine GraphRAG's indexer with Self-RAG's adaptive retrieval, for example.

### 6.2 Proposed Repository Structure

```
rag-evolution-benchmark/
│
├── core/                          # Shared abstractions
│   ├── base.py                    # BaseRAGSystem interface
│   ├── corpus.py                  # Corpus loader + common preprocessing
│   ├── evaluator.py               # Answer quality scoring (accuracy, faithfulness, citation)
│   ├── metrics.py                 # Cost, latency, token usage tracking
│   └── types.py                   # Shared data types (Query, Answer, Chunk, Entity, etc.)
│
├── indexers/                      # How documents are processed for storage
│   ├── flat_chunker.py            # Basic fixed-size chunking (Naive RAG baseline)
│   ├── semantic_chunker.py        # Semantic-boundary chunking
│   ├── raptor_tree.py             # RAPTOR's recursive tree builder
│   ├── graph_extractor.py         # Entity/relationship extraction (GraphRAG, HippoRAG, LightRAG)
│   ├── community_detector.py      # Leiden community detection (GraphRAG)
│   └── lightrag_indexer.py        # LightRAG's simplified dual-level indexing
│
├── retrievers/                    # How relevant information is found
│   ├── vector_retriever.py        # Dense embedding search (DPR-style)
│   ├── bm25_retriever.py          # Sparse keyword search
│   ├── hybrid_retriever.py        # Combined vector + BM25 + reranking
│   ├── graph_retriever.py         # Subgraph retrieval from knowledge graph
│   ├── pagerank_retriever.py      # HippoRAG's Personalised PageRank
│   ├── community_retriever.py     # GraphRAG's community summary retrieval
│   └── adaptive_router.py         # Adaptive-RAG's complexity-based routing
│
├── generators/                    # How answers are produced from context
│   ├── basic_generator.py         # Simple "context + question → answer"
│   ├── self_rag_generator.py      # With reflection tokens (simulated via prompting)
│   ├── crag_generator.py          # With retrieval quality evaluation + correction
│   └── reasoning_generator.py     # Using DeepSeek-R1 or similar reasoning model
│
├── memory/                        # Persistent memory systems
│   ├── no_memory.py               # Stateless baseline
│   ├── memgpt_style.py            # Context paging simulation
│   ├── mem0_memory.py             # Fact extraction + consolidation
│   ├── graph_memory.py            # Mem0 Graph-style relationship memory
│   └── amem_memory.py             # Self-organising Zettelkasten memory
│
├── agents/                        # Agent-based orchestration
│   ├── single_pass.py             # No agent — direct pipeline
│   ├── react_agent.py             # ReAct loop with retrieval as a tool
│   ├── reflexion_agent.py         # Self-reflecting agent with retry
│   ├── agentic_rag.py             # Full agentic RAG with strategy selection
│   └── rlm_agent.py               # Recursive Language Model via DSPy
│
├── systems/                       # Complete end-to-end system configs
│   ├── naive_rag.py               # Flat chunks + vector retrieval + basic generation
│   ├── self_rag.py                # Self-RAG (adaptive retrieval with reflection)
│   ├── crag.py                    # CRAG (corrective retrieval with evaluator)
│   ├── raptor.py                  # RAPTOR (hierarchical tree retrieval)
│   ├── graphrag.py                # GraphRAG (knowledge graph + communities)
│   ├── hipporag.py                # HippoRAG (neurobiological memory retrieval)
│   ├── lightrag.py                # LightRAG (lightweight graph retrieval)
│   ├── adaptive_rag.py            # Adaptive-RAG (complexity-routed)
│   ├── agentic_rag.py             # Agentic RAG (agent-controlled retrieval)
│   ├── rlm.py                     # RLM (recursive context processing)
│   └── hybrid_experimental.py     # Mix-and-match custom configurations
│
├── benchmarks/                    # Standardised evaluation datasets
│   ├── hotpotqa.py                # Multi-hop QA
│   ├── musique.py                 # Multi-step reasoning
│   ├── narrativeqa.py             # Long-document comprehension
│   ├── locomo.py                  # Long-conversation memory
│   └── custom_corpus.py           # User-provided document collection
│
├── comparison/                    # Analysis and visualisation
│   ├── run_benchmark.py           # Run all systems on a benchmark
│   ├── compare_results.py         # Generate comparison tables + charts
│   ├── cost_analysis.py           # Token usage and API cost comparison
│   ├── latency_analysis.py        # Response time profiling
│   └── evolution_timeline.py      # Visualise how capabilities evolved
│
└── configs/                       # Configuration files
    ├── models.yaml                # LLM provider settings
    ├── benchmarks.yaml            # Which benchmarks to run
    └── systems.yaml               # Which systems to compare
```

### 6.3 The Common Interface

Every system implements this contract:

```python
class BaseRAGSystem(ABC):
    """Common interface for all RAG/retrieval/memory/agent systems."""

    def index(self, corpus: Corpus) -> IndexMetrics:
        """Process a corpus into the system's internal representation.
        Returns metrics: time, cost, storage size, index structure stats."""

    def query(self, question: str, context: QueryContext = None) -> Answer:
        """Answer a question using the system's full pipeline.
        Returns: answer text, supporting evidence, confidence,
                 retrieval trace, generation trace, cost, latency."""

    def query_batch(self, questions: List[str]) -> List[Answer]:
        """Batch processing for benchmark runs."""

    def get_capabilities(self) -> SystemCapabilities:
        """Declare what this system can do.
        e.g., supports_multi_hop, supports_global_queries,
              supports_memory, supports_streaming, max_context_tokens."""
```

### 6.4 What This Architecture Reveals             

By running all systems against the same queries and corpus, one can directly observe the evolution:

**Query type: Simple factual ("What year was the company founded?")**
- Naive RAG, Adaptive-RAG (no-retrieval path), and most systems perform similarly.
- RLM is overkill — unnecessary recursion adds latency and cost.
- **Lesson:** Simpler is better for simple questions.

**Query type: Multi-hop ("How did the CEO who founded the company in X year react to the Y policy?")**
- Naive RAG fails (can't connect multiple pieces).
- RAPTOR and GraphRAG succeed by different mechanisms (tree vs. graph).
- HippoRAG's PageRank finds the connection efficiently.
- Agentic RAG iteratively retrieves and connects.
- **Lesson:** Structured indexing pays off for complex queries.

**Query type: Global sensemaking ("What are the main themes across all 500 documents?")**
- Naive RAG, Self-RAG, CRAG all fail catastrophically (can only see top-k chunks).
- RAPTOR partially succeeds (root-level summaries capture themes).
- GraphRAG excels — this is exactly what community summaries are designed for.
- RLM succeeds by programmatically scanning all documents via sub-LLM calls.
- **Lesson:** GraphRAG is purpose-built for this; RLM achieves it through general capability.

**Query type: Massive context (10M+ tokens)**
- Everything except RLM hits context window limits.
- GraphRAG can index it but retrieval may miss long-range dependencies.
- RLM processes it natively through recursive decomposition.
- **Lesson:** RLMs are the only approach that scales to arbitrary context length.

**Multi-session memory ("Remember I said I'm vegetarian when recommending restaurants")**
- All RAG systems fail (they're stateless).
- Mem0 succeeds by extracting and storing the preference.
- A-MEM succeeds and additionally links this to dining history.
- MemOS succeeds and can share this across multiple agent instances.
- **Lesson:** Memory is orthogonal to retrieval — you need both.

### 6.5 The Evolution Dimensions      

The benchmark framework measures progress along five axes:

```
              Accuracy
                 ↑
                 │
                 │
  Adaptability ←─┼─→ Efficiency (cost/latency)
                 │
                 │
                 ↓
            Scale              Memory Persistence
        (context size)          (cross-session)
```

**2020 (Naive RAG):** Moderate accuracy, no adaptability, moderate efficiency, small scale, no memory.    
**2024 (Self-RAG + GraphRAG):** High accuracy, adaptive retrieval, moderate efficiency, medium scale, no memory.    
**2025 (Agentic RAG + Mem0):** High accuracy, fully adaptive agents, variable efficiency, medium scale, persistent memory.    
**2026 (RLM + Context Engine):** High accuracy, recursively adaptive, efficient (targeted sub-calls), unbounded scale, integrated memory.     

---

## Part 7: The Overarching Arc   

The story of 2024–2026 is one of progressive *agency* and *self-management*:

1. **2024 taught systems *when* to retrieve** (Self-RAG, CRAG, Adaptive-RAG) and *how to structure knowledge* (RAPTOR, GraphRAG, HippoRAG). The pipeline was still fixed — humans designed the retrieval strategy.

2. **2025 gave systems *autonomy*** (Agentic RAG, MCP/A2A protocols) and *persistence* (Mem0, A-MEM, MemOS). Agents chose their own retrieval strategies. Memory survived across sessions. But context was still bounded by the model's window.

3. **2026 broke the context barrier** (RLMs). The prompt became a programmable object. The LLM manages its own context through code, spawning sub-LLMs as needed. This is the first approach that scales to truly arbitrary input sizes without architectural changes to the base model.

The through-line: AI systems are progressively taking over responsibilities that used to belong to human engineers — deciding when to retrieve, what to retrieve, how to structure knowledge, how to manage memory, and now, how to manage their own context window. Each innovation in this timeline represents one more degree of freedom transferred from the human designer to the AI system itself.

---

*Document compiled February 2026.*      
*Covers innovations from 2024 through early 2026*    
*Barely past Valentine's Day 2026 and we've already got over 6000 papers on Arxiv*      
*Obvs. I would've missed some stuff...*     
