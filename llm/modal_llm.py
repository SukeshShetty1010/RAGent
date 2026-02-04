"""
Modal-hosted LLM service for RAGent

Model:
- Qwen/Qwen2.5-1.5B-Instruct

Design:
- SINGLE endpoint only
- Runs on L40S GPU
- No secrets
- No class branches
- ragent_client.py unchanged
"""

from __future__ import annotations

import modal
import torch

# ------------------------------------------------------------------------------
# Modal App (MUST match ragent_client.py)
# ------------------------------------------------------------------------------

app = modal.App("rag-llama3-3b")

# ------------------------------------------------------------------------------
# Image
# ------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.6.3.post1",
        "torch==2.4.0",
        "transformers>=4.45.0",
        "huggingface-hub>=0.25.0",
    )
)

# ------------------------------------------------------------------------------
# HF Cache Volume (optional but recommended)
# ------------------------------------------------------------------------------

model_volume = modal.Volume.from_name(
    "hf-qwen-llm",
    create_if_missing=True,
)

# ------------------------------------------------------------------------------
# Global LLM (loaded once per container)
# ------------------------------------------------------------------------------

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        from vllm import LLM

        _llm = LLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            dtype=torch.bfloat16,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.80,
            trust_remote_code=True,
            enforce_eager=True,
            seed=42,
            download_dir="/models",
        )
    return _llm

# ------------------------------------------------------------------------------
# SINGLE GPU FUNCTION (THIS IS THE ONLY ENDPOINT)
# ------------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    image=image,
    timeout=300,
    volumes={"/models": model_volume},
    max_containers=5,
)
def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    Single GPU-backed text generation endpoint.
    """

    from vllm import SamplingParams

    if not prompt:
        return ""

    llm = _get_llm()

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    outputs = llm.generate([prompt], sampling)

    if not outputs or not outputs[0].outputs:
        return ""

    return outputs[0].outputs[0].text.strip()

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("🚀 Qwen2.5 LLM service ready (single GPU endpoint).")
