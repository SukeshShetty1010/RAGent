"""
tests/test_llm_config.py

Unit tests for LLM configuration consistency.
These tests are fully local — no Modal calls, no GPU required.
"""

import importlib
import types
import pytest

pytestmark = pytest.mark.unit


def _load_modal_llm_constants() -> dict:
    """
    Extract constants from modal_llm.py without executing Modal decorators.
    We read the module source and exec only the top-level constant assignments.
    """
    import ast, pathlib

    src = pathlib.Path("llm/modal_llm.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    constants: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    constants[target.id] = node.value.value
    return constants


def test_model_id_is_gemma3():
    """modal_llm.py must target the Gemma 3 12B model."""
    constants = _load_modal_llm_constants()
    assert "MODEL_ID" in constants, "MODEL_ID constant missing from modal_llm.py"
    assert constants["MODEL_ID"] == "google/gemma-3-12b-it", (
        f"Expected MODEL_ID='google/gemma-3-12b-it', got {constants['MODEL_ID']!r}"
    )


def test_app_name_in_modal_llm():
    """The Modal App name in modal_llm.py must be the Gemma 3 app slug."""
    import pathlib, re
    src = pathlib.Path("llm/modal_llm.py").read_text(encoding="utf-8")
    match = re.search(r'modal\.App\("([^"]+)"\)', src)
    assert match, "Could not find modal.App(...) call in modal_llm.py"
    assert match.group(1) == "gemma-3-12b-it-vllm", (
        f"Expected app name 'gemma-3-12b-it-vllm', got {match.group(1)!r}"
    )


def test_class_name_in_modal_llm():
    """The inference class in modal_llm.py must be named Gemma312BVLLM."""
    import pathlib, re
    src = pathlib.Path("llm/modal_llm.py").read_text(encoding="utf-8")
    assert "class Gemma312BVLLM" in src, (
        "Expected class name 'Gemma312BVLLM' not found in modal_llm.py"
    )




def test_fast_boot_is_bool():
    """FAST_BOOT constant must remain a boolean."""
    constants = _load_modal_llm_constants()
    assert "FAST_BOOT" in constants, "FAST_BOOT constant missing from modal_llm.py"
    assert isinstance(constants["FAST_BOOT"], bool), (
        f"FAST_BOOT must be a bool, got {type(constants['FAST_BOOT'])}"
    )
