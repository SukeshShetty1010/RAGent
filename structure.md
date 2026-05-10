# RAGent - Directory Structure

This document provides an overview of the RAGent project layout, detailing the role of each directory and key file within the architecture.

```text
RAGent/
├── agent/                           # Agentic logic, routing, and control flow
│   ├── capability/                  # Honesty gate module
│   │   └── capability_assessor.py   # Assesses if retrieved evidence is FULL/PARTIAL/INSUFFICIENT
│   ├── intent/                      # Query intent detection
│   │   └── intent_extractor.py      # Extracts intent signals (e.g., comparison, listicle) using regex
│   ├── tools/                       # External fallback tools
│   │   └── web_search.py            # Tavily-based web search fallback
│   ├── context_algorithms.py        # Algorithms for deduplication and token budget application
│   ├── context_assembler.py         # Assembles and orders context chunks based on entity balance
│   ├── prompt_manager.py            # Manages prompt templates and budget compliance (verbose/concise/truncated)
│   └── task_router.py               # Deterministic query task router
├── chunking/                        # Text splitting logic
│   └── editorial_chunker.py         # Word-based chunking strategy for GameSpot editorial content
├── data/                            # Raw API clients and fetching scripts
│   ├── gamespot_data.py             # Fetches and merges reviews/articles from the GameSpot API
│   ├── igdb_data.py                 # Fetches relational metadata from the IGDB API
│   └── rawg_data.py                 # Fetches core game identity/data from the RAWG API
├── embed/                           # Embedding utilities
│   └── ...                          # Prepares payloads for embedding before Qdrant upserts
├── engine/                          # Core execution orchestration
│   └── execution_engine.py          # RageEngine orchestrating the 7-step RAG pipeline
├── ingest/                          # Ingestion layer pipelines
│   ├── igdb_metadata_ingest.py      # Pipeline for IGDB relational data ingestion
│   ├── ingest_gamespot.py           # Pipeline for GameSpot data ingestion
│   └── rawg_identity_ingest.py      # Pipeline for RAWG core identity ingestion
├── KPI/                             # Evaluation and observability dashboard
│   └── Unified_KPI_Runner.py        # Generates executive KPI dashboard across evaluation modules
├── llm/                             # Language Model and Embedding infrastructure
│   ├── modal_embed.py               # Serverless Modal E5-base-v2 embedder on T4 GPU
│   ├── modal_llm.py                 # Serverless Modal vLLM engine (Qwen 2.5 7B on L40S GPU)
│   └── ragent_client.py             # Client with lazy binding to call the remote LLM generator
├── pre_process/                     # Data normalization and merging
│   ├── cleaner.py                   # Cleans raw API payloads (RAWGCleaner, IGDBCleaner, GameSpotCleaner)
│   ├── loader.py                    # Orchestrates fetching from RAWG, IGDB, GameSpot and saving raw JSONs
│   └── merge.py                     # Schema transformation and merging of multi-source objects
├── retriever/                       # Retrieval orchestration and configuration
│   ├── orchestrator.py              # Executes multi-strategy retrieval
│   ├── quality_gate.py              # Evaluates retrieved evidence quality (OK/WEAK/EMPTY)
│   ├── rag_retriever.py             # Executes Qdrant hybrid search (BM25 sparse + Dense Vector)
│   └── strategy_selector.py         # Selects the optimal retrieval configuration based on query intent
├── scripts/                         # Operational pipelines and scripts
│   └── bulk_ingest.py               # Batch pipeline to ingest multiple games sequentially
├── tests/                           # Testing and evaluation framework
│   ├── evaluation_metrics.py        # Calculates grounding fidelity, safety, and hallucination rates
│   ├── regression_suite.py          # Automated regression tests for the RAG pipeline
│   └── verify_engine.py             # Script to verify the execution engine flow and outputs
├── ui/                              # User Interface
│   └── app_streaming.py             # Streamlit frontend observing the immutable UI Contract
├── upsert/                          # Database insertion orchestrators
│   └── upsert_all.py                # Orchestrates the 5-stage idempotent insertion to Qdrant DB
├── utils/                           # Common utility modules
│   ├── caching.py                   # Deterministic semantic caching (via @cacheable)
│   └── observability.py             # Thread-safe metrics registry and latency profiling (ProfileBlock)
├── vector/                          # Vector database definitions
│   └── create_schema.py             # Defines the 5 Qdrant collections (Game, Platform, etc.)
└── README.md                        # Project documentation and architecture mapping
```