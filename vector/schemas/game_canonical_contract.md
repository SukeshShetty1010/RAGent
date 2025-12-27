# Game — Canonical Contract

## Purpose

This document defines what constitutes a **Game** in this system.

The contract exists to:
- Prevent **identity drift** across ingestion pipelines
- Establish a **single semantic anchor** for retrieval
- Enforce a stable shape **before vectorization**
- Ensure downstream systems (retrieval, filtering, joins) operate on a consistent entity

This contract is authoritative and must be satisfied before a Game object is persisted or embedded.

---

## Source of Truth

**RAWG is the single source of truth** for the canonical `Game` entity.

All Game objects:
- Originate from RAWG-derived data
- Are identified by RAWG’s game identifier
- Represent a single, authoritative interpretation of a video game

No other source is permitted to define or override Game identity.

---

## Identity

### Primary Identity Key

- **`game_id`**
  - Type: Integer
  - Definition: RAWG game ID
  - Guarantees global uniqueness within the system

This field alone determines whether two records refer to the same Game.

---

## Identity Descriptors

The following fields are classified as **Identity Descriptors**.

They are:
- Used for matching, filtering, and joins
- Stable enough to assist in identity confirmation
- Not treated as free-form content

**Identity Descriptors:**
- `title`
- `release_year`
- `genres`
- `developers`

These fields support human recognition and deterministic filtering but do **not** replace `game_id`.

---

## Descriptive Properties

The remaining fields enrich the Game object with metadata and descriptive context, but do not participate in identity resolution.

Examples include:
- Long-form descriptions
- Ratings and scores
- Tags and URLs
- Timestamps and capability flags

---

## Canonical Field Mapping

The table below defines the **exact mapping** from the Python transformation output (`create_game_object`) to the Weaviate `Game` schema.

| Python Key              | Weaviate Property       | Type        | Notes |
|-------------------------|-------------------------|-------------|-------|
| `game_id`               | `game_id`               | int         | Primary identity |
| `title`                 | `title`                 | text        | Identity descriptor |
| `description`           | `description`           | text        | Long-form content |
| `release_date`          | `release_date`          | date        | ISO date |
| `release_year`          | `release_year`          | int         | Identity descriptor |
| `genres`                | `genres`                | text[]      | Identity descriptor |
| `developers`            | `developers`            | text[]      | Identity descriptor |
| `publishers`            | `publishers`            | text[]      | Metadata |
| `tags`                  | `tags`                  | text[]      | Metadata |
| `average_rating`        | `average_rating`        | number      | Aggregated rating |
| `metacritic_score`      | `metacritic_score`      | int         | External score |
| `source_rawg_url`       | `source_rawg_url`       | text        | Source reference |
| `last_updated`          | `last_updated`          | date        | Source timestamp |
| `has_platform_specs`    | `has_platform_specs`    | boolean     | Capability flag |

---

## Contract Guarantees

A valid `Game` object:
1. Has a non-null `game_id`
2. Has a non-empty `title`
3. Originates from RAWG-derived data
4. Conforms exactly to this field structure
5. Is created **before** any embeddings or chunking occur

Any object that violates this contract must not be persisted or indexed.

---

## Non-Goals (Explicitly Out of Scope)

This contract does **not** define:
- Chunking strategies
- Embedding configuration
- Editorial content attachment
- Platform specification schemas

Those concerns are handled in later system stages.
