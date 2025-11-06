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
    model: str,
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
    print(f"[DEBUG] remote_llm.submit → URL: {MODAL_LLM_URL}/v1/chat/completions")
    print(f"[DEBUG] remote_llm.submit → payload: {payload}")
    try:
        resp = requests.post(
            f"{MODAL_LLM_URL}/v1/chat/completions",
            json=payload,
            headers=_headers(),
            timeout=timeout or DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise RemoteLLMError(f"Request to remote LLM failed (submit): {e}")
    print(f"[DEBUG] remote_llm.submit → status: {resp.status_code}, body: {resp.text}")
    if resp.status_code != 200:
        raise RemoteLLMError(f"Remote LLM submit returned status {resp.status_code}: {resp.text}")
    job_resp = resp.json()
    job_id = job_resp.get("job_id")
    if not job_id:
        raise RemoteLLMError(f"No job_id received in submit response: {job_resp}")

    # Poll for result
    poll_url = f"{MODAL_LLM_URL}/v1/chat/results/{job_id}"
    start_time = time.time()
    while True:
        print(f"[DEBUG] remote_llm.poll → URL: {poll_url}")
        try:
            resp2 = requests.get(
                poll_url,
                headers=_headers(),
                timeout=timeout or DEFAULT_TIMEOUT
            )
        except requests.RequestException as e:
            raise RemoteLLMError(f"Request to remote LLM failed (poll): {e}")
        print(f"[DEBUG] remote_llm.poll → status: {resp2.status_code}, body: {resp2.text}")
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
        raise RemoteLLMError(f"Unexpected response format: {resp}")
