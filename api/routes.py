# api/routes.py
"""
Legacy API routes module.

All active endpoints are now served from api/main.py (FastAPI).
This file is retained for backward compatibility only.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "ragent"}
