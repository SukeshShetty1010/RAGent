# deploy/modal_app.py
import os
import modal
from huggingface_hub import login
from fastapi import FastAPI, Request, HTTPException
from uuid import uuid4
from vllm import LLM, SamplingParams

# Build the base image with necessary dependencies
image = (
    modal.Image.debian_slim()
      .apt_install("git")
      .pip_install(
          "vllm",
          "torch",
          "transformers",
          "huggingface_hub",
          "fastapi[standard]"
      )
)

app = modal.App("ragent-llm-server-api", image=image)
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Single job queue style (submit → poll)
@ app.function(
    gpu="A100-40GB",
    timeout=3600,
    secrets=[modal.Secret.from_name("ragent-secrets")]
)
def run_inference(request_payload: dict) -> dict:
    # Authenticate to HF if required
    login(token=os.environ["HF_TOKEN"])
    # Load model
    llm = LLM(MODEL_ID)
    # Build prompt
    messages = request_payload.get("messages", [])
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    params = SamplingParams(
        max_tokens=request_payload.get("max_tokens", 512),
        temperature=request_payload.get("temperature", 0.0),
    )
    outputs = llm.generate(prompt, sampling_params=params)
    if not outputs:
        text = ""
    else:
        first = outputs[0]
        try:
            text = first.outputs[0].text
        except Exception:
            text = getattr(first, "output_text", "")
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }
        ]
    }

web_app = FastAPI()
_jobs: dict[str, modal.FunctionCall] = {}

@ web_app.post("/v1/chat/completions")
async def submit_job(request: Request):
    data = await request.json()
    job_id = uuid4().hex
    fcall = run_inference.spawn(data)
    _jobs[job_id] = fcall
    print(f"[DEBUG] submit_job created job_id = {job_id}")
    print(f"[DEBUG] _jobs keys now = {list(_jobs.keys())}")
    return {"job_id": job_id, "status": "pending"}

@ web_app.get("/v1/chat/results/{job_id}")
async def get_result(job_id: str):
    print(f"[DEBUG] get_result called with job_id = {job_id}")
    print(f"[DEBUG] _jobs keys currently = {list(_jobs.keys())}")
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="job_id not found")
    fcall = _jobs[job_id]
    try:
        result = fcall.get(timeout=0)
    except Exception:
        return {"job_id": job_id, "status": "pending"}
    # Optionally keep job even after done, but for now remove
    del _jobs[job_id]
    return {"job_id": job_id, "status": "done", "result": result}

@ app.function()
@ modal.asgi_app(label="ragent-llm-server-api")
def web() -> FastAPI:
    return web_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8080)
