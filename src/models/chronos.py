"""
Amazon Chronos Time Series Foundation Model Wrapper
Paper: arXiv:2403.07815
Note: Chronos is univariate - process each channel separately
"""

import torch
import numpy as np
from typing import Dict, Optional, Union
from .base import BaseTSFMWrapper


class ChronosWrapper(BaseTSFMWrapper):
    """Wrapper for Amazon Chronos model"""

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-large",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__("Chronos", device)
        self.model_id = model_id

    def load_model(self) -> None:
        """Load Chronos from HuggingFace"""
        try:
            from chronos import ChronosPipeline

            self.model = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32
            )
            self.is_loaded = True
            print(f"Chronos loaded on {self.device}")

        except ImportError:
            raise ImportError("Please install chronos: pip install chronos-forecasting")

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        num_samples: int = 20,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts - Chronos is univariate"""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        batch_size, seq_len, n_channels = X.shape

        # Average channels for univariate prediction
        X_avg = X.mean(dim=2)

        with torch.no_grad():
            forecasts = self.model.predict(
                X_avg,
                prediction_length=horizon,
                num_samples=num_samples
            )

        predictions = forecasts.mean(dim=1).cpu().numpy()
        uncertainties = forecasts.std(dim=1).cpu().numpy()

        predictions = np.expand_dims(predictions, -1).repeat(n_channels, axis=-1)

        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'model': self.model_name
        }

    def predict_per_channel(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        num_samples: int = 20
    ) -> Dict[str, np.ndarray]:
        """Predict each channel separately (slower but more accurate)"""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        batch_size, seq_len, n_channels = X.shape
        all_predictions = []

        for c in range(n_channels):
            X_c = X[:, :, c]

            with torch.no_grad():
                forecasts = self.model.predict(
                    X_c,
                    prediction_length=horizon,
                    num_samples=num_samples
                )

            pred_c = forecasts.mean(dim=1).cpu().numpy()
            all_predictions.append(pred_c)

        predictions = np.stack(all_predictions, axis=-1)

        return {
            'predictions': predictions,
            'model': self.model_name
        }

    def few_shot_adapt(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        **kwargs
    ) -> None:
        """Chronos doesn't support fine-tuning easily"""
        print("Warning: Chronos few-shot adaptation not implemented")
        print("Using zero-shot predictions")
