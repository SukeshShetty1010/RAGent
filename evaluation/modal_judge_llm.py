"""
evaluation/modal_judge_llm.py — RAGAS judge backed by the Modal
google/gemma-3-12b-it deployment (llm/modal_llm.py), as a second,
independent judge track alongside the Groq-backed one in
ragas_eval.py / ablation.py.

Motivation: Groq free-tier daily token quotas (TPD) have walled every
model tried as a judge (llama-3.3-70b, gpt-oss-120b, qwen3.6-27b all
share the same org-level cap). Modal bills GPU-seconds, not a daily
token budget, so a run here can't hit that wall. This is a parallel,
independent evaluation run (its own output file, its own checkpoint)
-- not a per-row substitute -- so a single RAGAS results file never
mixes judges.

Implements ragas's low-level BaseRagasLLM directly (not
LangchainLLMWrapper / ChatOpenAI) because llm/modal_llm.py exposes a
raw `generate(prompt: str) -> Iterator[str]` Modal method, not an
OpenAI-compatible HTTP server -- there is nothing for ChatOpenAI to
point at.
"""

from __future__ import annotations

import typing as t

import modal
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.llms.base import BaseRagasLLM

GEMMA_APP_NAME = "gemma-3-12b-it-vllm"
GEMMA_CLASS_NAME = "Gemma312BVLLM"

# Gemma 3 IT chat template. The Modal `generate` endpoint takes a raw
# prompt string, not a message list, so the template is applied here.
_GEMMA_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


class ModalGemmaLLM(BaseRagasLLM):
    """RAGAS judge LLM backed by the Modal-hosted gemma-3-12b-it (vLLM)."""

    def __init__(self, max_new_tokens: int = 2048) -> None:
        super().__init__()
        self.multiple_completion_supported = False
        self.max_new_tokens = max_new_tokens
        self._model = None

    def _get_model(self):
        if self._model is None:
            cls = modal.Cls.from_name(GEMMA_APP_NAME, GEMMA_CLASS_NAME)
            self._model = cls()
        return self._model

    def _complete_once(self, prompt_text: str, temperature: float) -> str:
        model = self._get_model()
        chunks = model.generate.remote_gen(
            _GEMMA_TEMPLATE.format(prompt=prompt_text),
            max_new_tokens=self.max_new_tokens,
            temperature=max(temperature, 0.0),
        )
        return "".join(chunks)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        prompt_text = prompt.to_string()
        # No provider-side n>1 limitation (unlike Groq) -- loop locally.
        # ragas reads non-langchain LLM results as generations[0][i] for
        # i in range(n) -- a single outer list holding all n samples, not
        # n separate single-sample outer entries.
        texts = [self._complete_once(prompt_text, temperature) for _ in range(n)]
        return LLMResult(generations=[[Generation(text=t_) for t_ in texts]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        import asyncio

        prompt_text = prompt.to_string()
        temp = temperature if temperature is not None else 0.01
        # Sequential, not gathered: metrics like answer_relevancy request
        # n>1 (self-consistency sampling). Firing those n reps concurrently
        # sends multiple simultaneous requests into the same warm Modal
        # container, which races on its shared event loop (see
        # llm/modal_llm.py generate()'s self._loop.run_until_complete).
        texts = [
            await asyncio.to_thread(self._complete_once, prompt_text, temp)
            for _ in range(n)
        ]
        return LLMResult(generations=[[Generation(text=t_) for t_ in texts]])

    def is_finished(self, response: LLMResult) -> bool:
        # The custom `generate` endpoint doesn't surface vLLM's
        # finish_reason; a generous max_new_tokens (2048) makes silent
        # truncation unlikely for RAGAS's short JSON-schema outputs.
        return True
