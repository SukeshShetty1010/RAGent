"""
Modal-hosted LLM service for RAGent

Model:
- HuggingFaceTB/SmolLM3-3B

Design guarantees:
- SINGLE endpoint only
- Runs on L40S GPU
- No secrets or auth logic
- No class branches
- Backend calls via ragent_client.py
"""

from __future__ import annotations

import modal
import torch

# ------------------------------------------------------------------------------
# Modal App (DEPLOYMENT IDENTITY)
# ------------------------------------------------------------------------------

app = modal.App("rag-smollm3-3b")

# ------------------------------------------------------------------------------
# Image (CUDA-ready, minimal)
# ------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.33.0",
        "huggingface-hub>=0.25.0",
    )
)

# ------------------------------------------------------------------------------
# Persistent HF Cache Volume
# ------------------------------------------------------------------------------

model_volume = modal.Volume.from_name(
    "hf-smollm3-llm",
    create_if_missing=True,
)

# ------------------------------------------------------------------------------
# Global Model (loaded once per container)
# ------------------------------------------------------------------------------

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer

    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM3-3B",
            trust_remote_code=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM3-3B",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

    return _model, _tokenizer


# ------------------------------------------------------------------------------
# SINGLE GPU FUNCTION (ONLY REMOTE ENDPOINT)
# ------------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    image=image,
    timeout=300,
    volumes={"/models": model_volume},
    max_containers=5,
    scaledown_window=300,  # Modal 1.0 replacement for container_idle_timeout
)
@modal.concurrent(max_inputs=4)  # Modal 1.0 replacement for allow_concurrent_inputs
def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    GPU-backed text generation endpoint.
    """

    if not prompt:
        return ""

    model, tokenizer = _get_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=32768,
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=temperature > 0,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


# ------------------------------------------------------------------------------
# Local Entrypoint (smoke test only)
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("🚀 SmolLM3-3B service ready (L40S · transformers backend)")
