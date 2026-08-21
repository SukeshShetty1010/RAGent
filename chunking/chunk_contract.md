# Editorial Chunk Contract

## Purpose

An **Editorial Chunk** is the atomic unit of retrieval for long-form editorial
content (reviews and articles) associated with a canonical Game.

Chunks are:
- Semantically self-contained
- Word-bounded for embedding compatibility (whitespace-delimited words,
  not model tokens — see "Tokenization Guarantees" below)
- Always scoped to a specific Game (no orphan text)
- Deterministically reproducible from source content

This contract ensures stable retrieval, debuggability, and traceability
across ingestion, embedding, and search layers.

---

## Chunk Identity Rules

- One chunk corresponds to a contiguous span of words from a single
  editorial body.
- Chunk boundaries are deterministic given:
  - normalized text
  - word splitter
  - chunk size (in words)
  - overlap (in words)
- Chunks never mix content from different editorials.

---

## Editorial Chunk Schema

| Field Name               | Type   | Description |
|--------------------------|--------|-------------|
| `chunk_id`               | UUID   | Deterministic UUID derived from content hash |
| `content`                | Text   | The chunk text (word-bounded) |
| `game_uuid`              | UUID   | Canonical Game UUID |
| `parent_editorial_uuid`  | UUID   | UUID of the source editorial container object |
| `source`                 | Text   | One of: `"gamespot"`, `"wikipedia"`, `"steam"` |
| `content_type`           | Enum   | `"review"` or `"article"` |
| `chunk_index`            | Int    | Order of this chunk within the source text |
| `source_title`           | Text   | Title of the original review or article |

---

## Tokenization Guarantees

- Chunk size is measured in **whitespace-delimited words**, not model
  tokens — there is no tokenizer in this path, only a word splitter
  (`chunking/editorial_chunker.py`'s `WordSplitter`)
- Production configuration is 300 words per chunk, 50 words of overlap
  (`embed/prepare_editorial_payloads.py`) — roughly 150-200 real model
  tokens, since English averages under a word per token
- Word counts are computed using the **local word splitter**
- Word windows must never exceed the configured maximum

---

## Architectural Invariants

- No chunk exists without a `game_uuid`
- No chunk exists without a `parent_editorial_uuid`
- Chunking is a **pure transformation**
- Chunk content must be index-ready without further mutation

---

## Non-Goals

- No embedding logic
- No vector DB interaction
- No persistence
- No summarization or rewriting

This contract defines the **retrieval boundary** for editorial knowledge.
