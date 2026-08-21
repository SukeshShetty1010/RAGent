"""
tests/test_rag_retriever_cli.py

Unit tests for T15 (AUDIT_TASKS.md): the CLI harness in
retriever/rag_retriever.py used to build a Llama-3 chat-template prompt
by hand (`format_llama3_prompt`) for a model this repo no longer runs.
It has been replaced by `_build_cli_prompt()`, which reproduces the
production engine's STEP 1/4/5/6 (routing, capability assessment,
context assembly, prompt construction) using the real collaborators --
so the CLI harness now exercises the same PromptManager path production
traffic does.

Hermetic: no Qdrant, no network. The only collaborator that can reach
disk/network on its own (RetrievalQualityGate's lazy CorpusEntityIndex)
is monkeypatched at its resolution point, `retriever.quality_gate.
_get_entity_index`, mirroring the `_entity_index_override` seam that
module already documents.
"""

from __future__ import annotations

import pytest

from retriever import rag_retriever
from agent.capability.capability_types import AnswerCapability

pytestmark = pytest.mark.unit


class _GroundedEntityIndex:
    def assess_grounding(self, query, evidence):
        return True


@pytest.fixture(autouse=True)
def _stub_entity_index(monkeypatch):
    monkeypatch.setattr(
        "retriever.quality_gate._get_entity_index",
        lambda: _GroundedEntityIndex(),
    )


def _chunk(source_title, content, score=5.0):
    return {
        "content": content,
        "source_title": source_title,
        "chunk_index": 0,
        "score": score,
    }


# ============================================================
# 1. The retired Modal/Llama-3 formatter is actually gone
# ============================================================

def test_format_llama3_prompt_removed():
    assert not hasattr(rag_retriever, "format_llama3_prompt")


def test_no_llama3_control_tokens_in_module_source():
    import inspect

    src = inspect.getsource(rag_retriever)
    for token in ("<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"):
        assert token not in src


# ============================================================
# 2. _build_cli_prompt reaches the real PromptManager path
# ============================================================

def test_build_cli_prompt_uses_real_prompt_path():
    query = "What is the setting of Elden Ring?"
    chunks = [
        _chunk(
            "Elden Ring Wiki",
            "Elden Ring is set in the Lands Between, a dark fantasy realm "
            "shattered after the destruction of the titular Elden Ring.",
        )
    ]

    prompt, task, capability, quality = rag_retriever._build_cli_prompt(query, chunks)

    assert query in prompt
    assert "Lands Between" in prompt
    assert "<|" not in prompt
    assert capability != AnswerCapability.INSUFFICIENT
    assert task is not None
    assert quality is not None


# ============================================================
# 3. Empty evidence reaches the real refusal path, not a
#    special case the harness routes around
# ============================================================

def test_build_cli_prompt_empty_chunks_is_insufficient():
    from agent.prompt_templates import insufficient_prompt

    query = "What is the setting of Elden Ring?"

    prompt, task, capability, quality = rag_retriever._build_cli_prompt(query, [])

    assert capability == AnswerCapability.INSUFFICIENT
    assert prompt == insufficient_prompt(query, reason="no_evidence")
