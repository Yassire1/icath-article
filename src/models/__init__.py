"""
Model factory and registry
"""

from typing import Dict, Optional
from scripts.pipeline_config import MODEL_IDS, PATCHTST_MAX_STEPS, PATCHTST_BATCH_SIZE, PATCHTST_WINDOWS_BATCH_SIZE
from .base import BaseTSFMWrapper
from .moment import MOMENTWrapper
from .chronos import ChronosWrapper
from .lag_llama import LagLlamaWrapper
from .patchtst import PatchTSTWrapper

MODEL_REGISTRY = {
    'moment': MOMENTWrapper,
    'chronos': ChronosWrapper,
    'lag_llama': LagLlamaWrapper,
    'patchtst': PatchTSTWrapper,
}


def get_model(model_name: str, device: str = "cuda", **kwargs) -> BaseTSFMWrapper:
    """Factory function to get model wrapper"""
    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    if model_name == "moment":
        kwargs.setdefault("model_id", MODEL_IDS["moment"])
    elif model_name == "chronos":
        kwargs.setdefault("model_id", MODEL_IDS["chronos"])
    elif model_name == "lag_llama":
        kwargs.setdefault("model_id", MODEL_IDS["lag_llama"])
    elif model_name == "patchtst":
        kwargs.setdefault("max_steps", PATCHTST_MAX_STEPS)
        kwargs.setdefault("batch_size", PATCHTST_BATCH_SIZE)
        kwargs.setdefault("windows_batch_size", PATCHTST_WINDOWS_BATCH_SIZE)

    return MODEL_REGISTRY[model_name](device=device, **kwargs)


def list_models() -> list:
    """List available models"""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'get_model',
    'list_models',
    'BaseTSFMWrapper',
    'MOMENTWrapper',
    'ChronosWrapper',
    'LagLlamaWrapper',
    'PatchTSTWrapper',
]
