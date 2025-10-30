# utils/gpu_utils.py
import torch

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def to_device(obj):
    """obj = model, tensor, list of tensors, etc."""
    device = get_device()
    return obj.to(device) if hasattr(obj, "to") else obj