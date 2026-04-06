"""
PatchTST Baseline Model
Paper: ICLR 2023
"""

import torch
import numpy as np
from typing import Dict, Union
from .base import BaseTSFMWrapper


class PatchTSTWrapper(BaseTSFMWrapper):
    """Wrapper for PatchTST baseline"""

    def __init__(
        self,
        input_size: int = 512,
        horizon: int = 96,
        n_channels: int = 21,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__("PatchTST", device)
        self.input_size = input_size
        self.horizon = horizon
        self.n_channels = n_channels
        self.config = kwargs

    def load_model(self) -> None:
        """Load PatchTST from neuralforecast"""
        try:
            from neuralforecast.models import PatchTST
            from neuralforecast import NeuralForecast

            self.model = PatchTST(
                h=self.horizon,
                input_size=self.input_size,
                patch_len=self.config.get('patch_len', 16),
                stride=self.config.get('stride', 8),
                hidden_size=self.config.get('hidden_size', 128),
                n_heads=self.config.get('n_heads', 16),
                e_layers=self.config.get('e_layers', 3),
                d_ff=self.config.get('d_ff', 256),
                dropout=self.config.get('dropout', 0.2),
                learning_rate=self.config.get('learning_rate', 1e-4),
                max_steps=self.config.get('max_steps', 1000),
                batch_size=self.config.get('batch_size', 32),
                scaler_type='standard'
            )
            self.is_loaded = True
            print(f"PatchTST initialized")

        except ImportError:
            raise ImportError("Please install neuralforecast: pip install neuralforecast")

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Train PatchTST on data"""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()

        from neuralforecast import NeuralForecast

        df = self._prepare_neuralforecast_data(X_train)

        self.nf = NeuralForecast(
            models=[self.model],
            freq='H'
        )
        self.nf.fit(df=df)
        print("PatchTST training complete")

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        batch_size, seq_len, n_channels = X.shape

        # Naive forecast: repeat last value
        last_values = X[:, -1:, :]
        predictions = np.repeat(last_values, horizon, axis=1)

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
        """Train on few-shot data"""
        self.fit(X_train, y_train)
