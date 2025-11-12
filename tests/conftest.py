# tests/conftest.py
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()


@pytest.fixture(scope="session")
def rawg_client():
    api_key = os.getenv("RAWG_API_KEY")
    if not api_key:
        pytest.skip("RAWG_API_KEY not set – skipping live API tests")
    from auth.rawg_client import RAWGClient
    return RAWGClient(api_key=api_key)