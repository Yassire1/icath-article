"""
Base model wrapper for unified TSFM interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseTSFMWrapper(ABC):
    """Abstract base class for all TSFM wrappers"""

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_loaded = False
        self.supports_few_shot = False

    @abstractmethod
    def load_model(self) -> None:
        """Load pretrained model"""
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate predictions"""
        pass

    def predict_batch(
        self,
        dataloader: DataLoader,
        horizon: int = 96
    ) -> Dict[str, np.ndarray]:
        """Predict on entire dataloader"""
        all_preds = []
        all_targets = []

        for X, y in dataloader:
            result = self.predict(X, horizon)
            all_preds.append(result['predictions'])
            all_targets.append(y.numpy())

        return {
            'predictions': np.concatenate(all_preds, axis=0),
            'targets': np.concatenate(all_targets, axis=0)
        }

    def zero_shot(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96
    ) -> Dict[str, np.ndarray]:
        """Zero-shot prediction (no adaptation)"""
        if not self.is_loaded:
            self.load_model()
        return self.predict(X, horizon)

    @abstractmethod
    def few_shot_adapt(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 10,
        **kwargs
    ) -> None:
        """Few-shot adaptation (e.g., LoRA)"""
        pass

    def get_model_info(self) -> Dict:
        """Return model metadata"""
        return {
            'name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded
        }
