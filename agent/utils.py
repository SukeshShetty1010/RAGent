import json
import os
import logging
from datetime import datetime

# ---------- Logger Setup ----------
def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------- File Save Helper ----------
def save_jsonl(data_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------- Base Ingestor ----------
class BaseIngestor:
    def __init__(self, api_client, output_path):
        self.api_client = api_client
        self.output_path = output_path
        self.logger = setup_logger(self.__class__.__name__)

    def fetch(self):
        raise NotImplementedError

    def normalize(self, data):
        raise NotImplementedError

    def run(self):
        self.logger.info("Starting ingestion...")
        raw_data = self.fetch()
        normalized = self.normalize(raw_data)
        save_jsonl(normalized, self.output_path)
        self.logger.info(f"Ingestion completed. Saved {len(normalized)} records to {self.output_path}")