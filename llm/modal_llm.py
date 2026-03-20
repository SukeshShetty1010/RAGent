"""
Modal-hosted LLM service for RAGent
====================================
Model  : Qwen/Qwen2.5-7B-Instruct
Engine : vLLM (AsyncLLMEngine — PagedAttention + continuous batching)
GPU    : L40S  (48 GB GDDR6, Ada Lovelace, single card is sufficient)
App    : qwen2-5-7b-instruct-vllm

Design guarantees
-----------------
- SINGLE inference endpoint: `generate` — a Modal generator function
  that yields string token chunks and is consumed via remote_gen() on
  the client side (ragent_client_streaming.py).
- Engine is loaded once per container via @modal.enter; cleaned up
  via @modal.exit.
- Model weights and vLLM compilation artefacts are persisted on
  Modal Volumes to eliminate repeated downloads on cold starts.
- No quantisation, no LoRA, no weight modifications — pure bf16
  inference on the base Qwen2.5 instruct checkpoint.
- FAST_BOOT=True skips CUDA-graph capture and Torch-compile JIT so
  containers start in ~20 s instead of ~90 s.  Flip to False for
  maximum sustained throughput when replicas stay warm.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Iterator

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Cache directories inside the container — backed by Modal Volumes.
HF_CACHE_DIR   = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

# When True, skip CUDA-graph capture and torch.compile to cut cold-start
# latency at the cost of slightly lower steady-state throughput.
FAST_BOOT = True

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# Modal provides the NVIDIA CUDA drivers at the host level; we only need
# the Python-side vLLM stack.  We pin vLLM for reproducibility and pull
# FlashInfer for optimised attention kernels on Ada Lovelace (L40S).
# hf-transfer gives ~700 MB/s download speeds from the HF Hub.

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            # Accelerate HF Hub downloads via the rust-based hf-transfer.
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Point HF and vLLM caches at our mounted Volumes.
            "HF_HOME":         HF_CACHE_DIR,
            "HF_HUB_CACHE":    HF_CACHE_DIR,
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
            # Select FlashInfer attention kernel at the environment level.
            # This is the correct mechanism — NOT an AsyncEngineArgs kwarg.
            # FlashInfer delivers best decode throughput on Ada Lovelace (L40S).
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        }
    )
)

# ---------------------------------------------------------------------------
# Persistent Volumes  (survive container teardown and restarts)
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("hf-qwen2-5-7b-instruct", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-qwen2-5-7b-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------

app = modal.App("qwen2-5-7b-instruct-vllm")

# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L40S",
    image=vllm_image,
    timeout=600,                      # per-request timeout (seconds)
    scaledown_window=300,             # keep warm for 5 min after last request
    max_containers=5,                 # horizontal scale ceiling
    volumes={
        HF_CACHE_DIR:   hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=8)       # vLLM handles concurrency internally
class Qwen25VLLM:
    """
    Container-persistent vLLM inference engine for Qwen/Qwen2.5-7B-Instruct.

    Lifecycle
    ---------
    @modal.enter  — called once when the container first becomes live.
                    Downloads weights (if not cached), initialises the
                    AsyncLLMEngine, and runs a warmup request so the
                    first real request does not pay for lazy init.

    @modal.exit   — called once just before the container is torn down.
                    Gracefully shuts down the engine event loop.

    @modal.method — `generate` is the sole public endpoint; it is a
                    Python generator that yields decoded string chunks,
                    making it transparently consumable via remote_gen().
    """

    @modal.enter()
    def load_engine(self) -> None:
        """
        Initialise AsyncLLMEngine with settings tuned for an L40S.

        Key choices
        -----------
        dtype="bfloat16"
            Native bf16 on Ada Lovelace — no accuracy loss, best
            memory bandwidth utilisation.
        gpu_memory_utilization=0.92
            Leave ~8 % for CUDA driver/framework overhead; the rest
            is allocated to KV cache, maximising concurrency.
        max_model_len=32768
            Safe Qwen2.5 default on a single L40S. The model card
            advertises 128K support, but the current config defaults
            to 32K and recommends YaRN only when inputs exceed that.
        enable_prefix_caching=True
            Shares KV-cache blocks across requests with the same
            prompt prefix — valuable for RAG where a system prompt
            is repeated across many queries.
        enforce_eager=(FAST_BOOT)
            True  → skip CUDA-graph capture → faster cold start.
            False → capture graphs  → higher throughput when warm.
        """
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            max_model_len=32768,
            enable_prefix_caching=True,
            enforce_eager=FAST_BOOT,
            # Disable verbose request logging in production.
            disable_log_requests=True,
        )

        self._engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

        # Dedicate a persistent event loop to all async engine calls.
        self._loop = asyncio.new_event_loop()

        # Warmup: run one empty generation so CUDA kernels are compiled
        # and the first user request is not penalised.
        self._loop.run_until_complete(self._warmup())

    async def _warmup(self) -> None:
        """Send a minimal request through the engine to trigger lazy init."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        async for _ in self._engine.generate(
            prompt="ping",
            sampling_params=sampling_params,
            request_id="warmup-" + uuid.uuid4().hex,
        ):
            pass

    @modal.exit()
    def shutdown_engine(self) -> None:
        """Clean up the event loop when the container is about to stop."""
        try:
            self._loop.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public inference endpoint
    # ------------------------------------------------------------------

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> Iterator[str]:
        """
        Stream token chunks from Qwen/Qwen2.5-7B-Instruct via vLLM.

        This is a Modal generator method.  Each ``yield`` sends one
        decoded text chunk back to the caller over the Modal network
        boundary.  The client consumes it with ``remote_gen()``.

        Parameters
        ----------
        prompt : str
            Pre-formatted prompt string. The caller is responsible for
            applying Qwen-compatible chat formatting upstream if
            conversational templating is required.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature.  Use 0.0 for greedy decoding.
        top_p : float
            Nucleus sampling probability threshold.
        repetition_penalty : float
            Penalise repeated n-grams; 1.0 = no penalty.

        Yields
        ------
        str
            Incremental decoded text chunks as they are produced by
            the vLLM engine.
        """
        if not prompt:
            return

        # Run the async generator to completion on our persistent loop,
        # bridging the sync Modal generator boundary.
        gen = self._stream_tokens(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Drive the async generator step-by-step from the sync context
        # so each chunk can be yielded immediately without buffering.
        while True:
            try:
                chunk = self._loop.run_until_complete(gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    async def _stream_tokens(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> AsyncIterator[str]:
        """
        Async generator that drives AsyncLLMEngine and yields decoded
        incremental text deltas (not cumulative outputs).

        vLLM's AsyncLLMEngine.generate() yields RequestOutput objects
        where each output.outputs[0].text contains the *cumulative*
        text generated so far.  We compute the delta by tracking the
        previous length of the decoded string and slicing the new tail.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        request_id = "ragent-" + uuid.uuid4().hex
        prev_text_len: int = 0

        async for request_output in self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            # request_output.outputs is a list of CompletionOutput objects;
            # we always request n=1 so index 0 is the only candidate.
            output_text: str = request_output.outputs[0].text

            # Compute the incremental delta since the last yield.
            delta: str = output_text[prev_text_len:]
            prev_text_len = len(output_text)

            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Local entrypoint — smoke test
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """
    Quick smoke test.  Run with:
        modal run modal_llm.py
    """
    model = Qwen25VLLM()

    test_prompt = "Briefly explain PagedAttention in vLLM."
    print(f"Prompt: {test_prompt}\n")
    print("Response: ", end="", flush=True)

    accumulated: list[str] = []
    for chunk in model.generate.remote_gen(
        test_prompt,
        max_new_tokens=256,
        temperature=0.1,
    ):
        print(chunk, end="", flush=True)
        accumulated.append(chunk)

    print()  # newline after streaming output
    full_response = "".join(accumulated)
    print(f"\n[Total chars generated: {len(full_response)}]")
