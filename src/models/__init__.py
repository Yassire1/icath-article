"""
Model factory and registry
"""

from typing import Dict, Optional
from .base import BaseTSFMWrapper
from .moment import MOMENTWrapper
from .chronos import ChronosWrapper
from .timegpt import TimeGPTWrapper
from .patchtst import PatchTSTWrapper

MODEL_REGISTRY = {
    'moment': MOMENTWrapper,
    'chronos': ChronosWrapper,
    'timegpt': TimeGPTWrapper,
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
    'TimeGPTWrapper',
    'PatchTSTWrapper',
]
