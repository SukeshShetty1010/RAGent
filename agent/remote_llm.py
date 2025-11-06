# agent/remote_llm.py
import os
import requests
import time
from typing import List, Dict, Any, Optional

class RemoteLLMError(Exception):
    pass

MODAL_LLM_URL = os.getenv(
    "MODAL_LLM_URL",
    "https://thesukeshshetty--ragent-llm-server-api.modal.run"
)
MODAL_LLM_KEY = os.getenv("MODAL_LLM_KEY", None)
DEFAULT_TIMEOUT = int(os.getenv("MODAL_LLM_TIMEOUT", "300"))
DEFAULT_MAX_TOKENS = int(os.getenv("MODAL_LLM_MAX_TOKENS", "512"))

def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MODAL_LLM_KEY:
        headers["Authorization"] = f"Bearer {MODAL_LLM_KEY}"
    return headers

def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "temperature": temperature
    }

    # Submit job
    try:
        resp = requests.post(
            f"{MODAL_LLM_URL}/v1/chat/completions",
            json=payload,
            headers=_headers(),
            timeout=timeout or DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise RemoteLLMError(f"Request to remote LLM failed (submit): {e}")

    if resp.status_code != 200:
        raise RemoteLLMError(f"Remote LLM submit returned status {resp.status_code}: {resp.text}")

    job_resp = resp.json()
    job_id = job_resp.get("job_id")
    if not job_id:
        raise RemoteLLMError(f"No job_id received in response: {job_resp}")

    # Poll for result
    start_time = time.time()
    poll_url = f"{MODAL_LLM_URL}/v1/chat/results/{job_id}"
    while True:
        try:
            resp2 = requests.get(
                poll_url,
                headers=_headers(),
                timeout=timeout or DEFAULT_TIMEOUT
            )
        except requests.RequestException as e:
            raise RemoteLLMError(f"Request to remote LLM failed (poll): {e}")

        if resp2.status_code == 200:
            result = resp2.json()
            if result.get("status") == "done":
                return result["result"]
        else:
            raise RemoteLLMError(f"Remote LLM poll returned status {resp2.status_code}: {resp2.text}")

        if time.time() - start_time > (timeout or DEFAULT_TIMEOUT):
            raise RemoteLLMError(f"Timeout waiting for remote LLM result (job_id {job_id})")

        time.sleep(1)

def get_text_from_response(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except KeyError:
        if "choices" in resp and len(resp["choices"]) > 0 and "text" in resp["choices"][0]:
            return resp["choices"][0]["text"]
        raise RemoteLLMError(f"Unexpected response format: {resp}")
