# Weaviate Class Responsibilities

## Purpose

This document defines **strict, non-overlapping responsibilities** for each Weaviate class in the system.

Its goal is to ensure:
- Clear separation of identity, content, metadata, and operational data
- Predictable vector behavior
- Zero ambiguity at query time about *where meaning lives*

This is an architectural contract. These responsibilities are not suggestions.

---

## Responsibility Table

| Class Name       | Role                  | Vector Strategy                     | Primary Responsibility |
|------------------|-----------------------|-------------------------------------|------------------------|
| **Game**         | Anchor                | Lightweight Semantic (Transformers) | Canonical identity and semantic anchor for a game |
| **GameSpot_Game**| Editorial Container   | Deep Semantic (OpenAI)               | Raw editorial source material for downstream chunking |
| **IGDB_Game**    | Relational Metadata   | None (Filter / Join only)            | Deep relational metadata (franchises, versions, DLCs) |
| **PlatformSpec** | Operational Data      | None (Filter only)                  | Platform-specific operational constraints |

---

## Class Role Definitions

### Game (Anchor)

**Role:** Canonical Anchor  
**Responsibility:**  
- Represents *one and only one* canonical game
- Owns the system-wide identity (`game_id`)
- Acts as the semantic pivot point for all joins and retrievals

**Key Characteristics:**
- Lightweight semantic vectorization
- Stable, identity-centric fields
- Optimized for filtering, joins, and high-level semantic grounding

This class answers:  
> “What game are we talking about?”

---

### GameSpot_Game (Editorial Container)

**Role:** Editorial Container  
**Responsibility:**  
- Stores long-form editorial content (summaries, reviews, articles)
- Serves as a **source layer**, not a final retrieval primitive
- Feeds controlled chunking and downstream embedding workflows

**Key Characteristics:**
- Deep semantic vectorization
- Nested, high-entropy text fields
- Not authoritative for identity

This class answers:  
> “What is being said *about* the game?”

---

### IGDB_Game (Relational Metadata)

**Role:** Relational Metadata  
**Responsibility:**  
- Provides structural and relational context
- Models franchises, editions, expansions, bundles, and lineage
- Enables deterministic graph traversal and filtering

**Key Characteristics:**
- No vectorization
- Foreign-key–style references
- Used exclusively for joins and metadata queries

This class answers:  
> “How is this game related to other games or entities?”

---

### PlatformSpec (Operational Data)

**Role:** Operational Data  
**Responsibility:**  
- Captures platform-specific constraints
- Stores hardware requirements and platform availability
- Supports strict filtering logic

**Key Characteristics:**
- No vectorization
- Always attached to a canonical Game
- Operational, not semantic

This class answers:  
> “Can this game run on a given platform under given constraints?”

---

## The “No Orphan” Rule

**No editorial content or relational data may exist in the Knowledge Graph without a resolved reference to a Canonical Game entity.**

This rule applies to:
- All `GameSpot_Game` objects
- All `IGDB_Game` objects
- All `PlatformSpec` objects

Implications:
- Identity resolution **must occur before persistence**
- No free-floating content nodes are allowed
- All non-Game classes are subordinate to `Game`

Violation of this rule invalidates the graph.

---

## Inputs & Outputs

### Input

- High-level business requirements
  - Search relevance
  - Rich editorial grounding
  - Accurate filtering
  - Stable identity resolution

### Output

- Hard architectural constraints on:
  - Class boundaries
  - Vectorization strategy
  - Retrieval responsibilities
  - Graph integrity

These constraints are binding for all downstream system design.

---

## Explicit Non-Goals (MUST NOT DO)

This document does **not**:
- Define ingestion order
- Define retrieval or query construction
- Define chunking strategies
- Define embedding parameters
- Contain implementation code

Those concerns are intentionally deferred.

---

## Why This Exists

This document exists to prevent **Vector Soup**.

Without strict class responsibilities:
- Identity bleeds into content
- Content competes with metadata
- Retrieval becomes probabilistic and fragile

By locking responsibilities:
- Identity is deterministic
- Context is layered
- Semantics are intentional

At query time, the system must always know:
- **Which class answers “what is it?”**
- **Which class answers “what is being said?”**
- **Which class answers “how is it related?”**
- **Which class answers “can it run?”**

This document makes that distinction non-negotiable.
