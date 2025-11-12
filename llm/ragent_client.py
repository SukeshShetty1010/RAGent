# agent/ragent_client.py
import modal
chat_completion_remote = modal.Function.from_name("rag-llama3-3b", "chat_completion_remote")