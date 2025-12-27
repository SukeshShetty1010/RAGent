# Editorial Chunk Contract

## Purpose

An **Editorial Chunk** is the atomic unit of retrieval for long-form editorial
content (reviews and articles) associated with a canonical Game.

Chunks are:
- Semantically self-contained
- Token-bounded for embedding compatibility
- Always scoped to a specific Game (no orphan text)
- Deterministically reproducible from source content

This contract ensures stable retrieval, debuggability, and traceability
across ingestion, embedding, and search layers.

---

## Chunk Identity Rules

- One chunk corresponds to a contiguous span of tokens from a single
  editorial body.
- Chunk boundaries are deterministic given:
  - normalized text
  - tokenizer
  - chunk size
  - overlap
- Chunks never mix content from different editorials.

---

## Editorial Chunk Schema

| Field Name               | Type   | Description |
|--------------------------|--------|-------------|
| `chunk_id`               | UUID   | Deterministic UUID derived from content hash |
| `content`                | Text   | The chunk text (token-bounded) |
| `game_uuid`              | UUID   | Canonical Game UUID |
| `parent_editorial_uuid`  | UUID   | UUID of the source GameSpot_Game object |
| `source`                 | Text   | Constant value: `"gamespot"` |
| `content_type`           | Enum   | `"review"` or `"article"` |
| `chunk_index`            | Int    | Order of this chunk within the source text |
| `source_title`           | Text   | Title of the original review or article |

---

## Tokenization Guarantees

- Chunk size targets ~500 tokens
- Overlap of ~50 tokens between adjacent chunks
- Token counts are computed using a **local tokenizer**
- Token windows must never exceed the configured maximum

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
