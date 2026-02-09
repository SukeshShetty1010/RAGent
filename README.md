# RAGent — Capability-Aware Agentic RAG for Gaming Intelligence
## Intent-Aware Routing | Evidence-Gated Responses | KPI-Proven Performance

### Why RAGent Exists
Generic RAG systems fail in specialized domains like gaming intelligence, where nuance, data quality, and user intent are paramount. RAGent was built to solve this. It is an opinionated, production-grade framework that proves a RAG system can be simultaneously intelligent, honest, and completely observable, providing trustworthy answers to complex domain-specific questions.

### Proven Guarantees: Outcomes Over Tools
RAGent is engineered to deliver specific, measurable outcomes.

*   **🛡️ Honest, Evidence-Gated Responses:** The system refuses to answer if evidence is weak or missing. It prioritizes safety over speculation.
*   **🎯 Intent-Aware Query Handling:** It correctly understands and routes complex user intents, distinguishing a `comparison` from a `factual` query.
*   **📊 Measurable, Resume-Grade Performance:** End-to-end performance—including retrieval quality, answer faithfulness, and latency—is tracked via a unified KPI dashboard.
*   **🎮 Domain-Native Intelligence:** The system's knowledge is rooted in a specialized gaming knowledge base (IGDB, RAWG, editorial), not generic web scrapes.
*   **⚙️ Deterministic & Reproducible:** The entire execution pipeline is deterministic and inspectable, ensuring that results are consistent and trustworthy.

### Designed to Answer
RAGent is explicitly designed to handle the types of questions gamers and analysts ask:

*   **Comparisons:** "What are the differences between *Far Cry 5* and *Assassin's Creed Valhalla*?"
*   **Factual Queries:** "What was the release date for *Far Cry 5*?"
*   **Listicles:** "What are the top 5 open-world games from the last year?"
*   **Temporal Questions:** "What are the latest patch notes for *Assassin's Creed Valhalla*?"

---

## System Architecture: The 7-Step Execution Engine

At its core, RAGent processes every query through a deterministic, 7-step pipeline within the `RageEngine`. This ensures that every decision, from retrieval to final response generation, is explicit, observable, and governed by the system's capabilities.

1.  **Intent Routing (`TaskRouter`)**: The user's query is first analyzed by the `IntentSignalExtractor` to identify its semantic purpose (e.g., comparison, factual query, listicle). The `TaskRouter` then maps these signals to a primary `TaskType`, forming a `RouterDecision` that guides the rest of the engine.

2.  **Strategy Selection (`StrategySelector`)**: Based on the `RouterDecision`, this pure-logic component selects a `RetrievalConfiguration`. It determines *how* to fetch evidence, such as using query decomposition for comparison tasks or expanding the context window for listicles.

3.  **Retrieval (`RetrievalOrchestrator`)**: The orchestrator executes the retrieval plan. It queries a Weaviate hybrid search (BM25 + Vector) backend for local, domain-specific data. A `RetrievalQualityGate` then assesses the results; if they are weak or the query has a temporal signal (e.g., "latest update"), it can trigger a fallback search to the public web via the `WebSearchTool`.

4.  **Capability Assessment (`CapabilityAssessor`)**: This is the **Honesty Gate**. It evaluates the retrieved evidence against the user's original intent to determine the system's `AnswerCapability`. It decides if the system has `FULL`, `PARTIAL`, or `INSUFFICIENT` evidence to form a safe and honest response.

5.  **Context Assembly (`ContextAssembler`)**: The retrieved evidence is passed to this component, which deduplicates, re-ranks, and budgets the content according to the `TaskType`. It enforces a hard character limit to ensure the final context is lean, relevant, and fits within the prompt window.

6.  **Prompt Construction (`PromptManager`)**: The `PromptManager` selects the appropriate instruction template based on both the `TaskType` and the `AnswerCapability`. It features a multi-stage fallback system (verbose -> concise -> truncated) to guarantee the final prompt respects a strict character budget. If the capability is `INSUFFICIENT`, it constructs a "safe refusal" prompt.

7.  **Guarded LLM Generation**: The final prompt is sent to the LLM *only if* the `AnswerCapability` is `FULL` or `PARTIAL`. If it is `INSUFFICIENT`, the system bypasses the LLM entirely and returns a pre-defined message explaining that it cannot answer safely.

## Running the KPI Dashboard

To get a full, real-time report of the system's performance, run the executive dashboard:

```bash
python3 KPI/Unified_KPI_Runner.py
```
