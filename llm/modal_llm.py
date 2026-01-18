import modal
import torch

app = modal.App("rag-llama3-3b")

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

@app.function(
    gpu="L40S",  # ← Upgraded to Ada Lovelace (48GB VRAM)
    timeout=300,
    image=image,
    secrets=[modal.Secret.from_name("rag-secrets")],
    max_containers=5,
)
def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        dtype=torch.bfloat16,  # ← L40S supports BF16; this matches Llama 3 training and improves numerical stability
        tensor_parallel_size=1,  # Single L40S is sufficient for a 3B model
        max_model_len=8192,
        gpu_memory_utilization=0.85,  # Conservative headroom; well within 48GB
        trust_remote_code=True,
        enforce_eager=True,
        seed=42,
    )

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling)
    return outputs[0].outputs[0].text.strip()
