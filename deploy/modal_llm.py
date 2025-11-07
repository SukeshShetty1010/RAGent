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
    gpu="T4",
    timeout=300,
    image=image,
    secrets=[modal.Secret.from_name("rag-secrets")],
    max_containers=5,
)
def chat_completion_remote(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        dtype=torch.float16,  # ← FIXED: T4 needs float16, NOT bfloat16
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
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