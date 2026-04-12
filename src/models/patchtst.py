"""
PatchTST Baseline Model
Paper: ICLR 2023
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from scripts.pipeline_config import PATCHTST_MAX_TRAIN_WINDOWS
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
        self.supports_few_shot = True

    def load_model(self) -> None:
        """Load PatchTST from neuralforecast"""
        try:
            from neuralforecast.models import PatchTST
            from neuralforecast import NeuralForecast

            self.model = PatchTST(
                h=self.horizon,
                input_size=min(self.input_size, 32),
                patch_len=self.config.get('patch_len', 16),
                stride=self.config.get('stride', 8),
                hidden_size=self.config.get('hidden_size', 128),
                n_heads=self.config.get('n_heads', 16),
                encoder_layers=self.config.get('encoder_layers', 3),
                linear_hidden_size=self.config.get('linear_hidden_size', 256),
                dropout=self.config.get('dropout', 0.2),
                learning_rate=self.config.get('learning_rate', 1e-4),
                max_steps=self.config.get('max_steps', 200),
                batch_size=self.config.get('batch_size', 64),
                windows_batch_size=self.config.get('windows_batch_size', 256),
                start_padding_enabled=True,
                enable_checkpointing=False,
                logger=False,
                scaler_type='standard'
            )
            self.is_loaded = True
            print(f"PatchTST initialized")

        except ImportError:
            raise ImportError("Please install neuralforecast: pip install neuralforecast")

    def _prepare_neuralforecast_data(self, X: np.ndarray) -> pd.DataFrame:
        """Convert (n_samples, seq_len, n_channels) array to neuralforecast DataFrame.

        Averages channels to produce a univariate representation.
        Returns DataFrame with columns: unique_id (str), ds (datetime), y (float).
        """
        n_samples, seq_len, n_channels = X.shape
        X_uni = X.mean(axis=-1)  # (n_samples, seq_len)
        base_ts = pd.date_range("2020-01-01", periods=seq_len, freq="h")
        dfs = [
            pd.DataFrame({"unique_id": str(i), "ds": base_ts, "y": X_uni[i]})
            for i in range(n_samples)
        ]
        return pd.concat(dfs, ignore_index=True)

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

        if len(X_train) > PATCHTST_MAX_TRAIN_WINDOWS:
            X_train = X_train[:PATCHTST_MAX_TRAIN_WINDOWS]
            y_train = y_train[:PATCHTST_MAX_TRAIN_WINDOWS]

        from neuralforecast import NeuralForecast

        df = self._prepare_neuralforecast_data(X_train)

        self.nf = NeuralForecast(
            models=[self.model],
            freq='h'
        )
        self.nf.fit(df=df)
        print("PatchTST training complete")

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts using the trained NeuralForecast model."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        if not hasattr(self, 'nf'):
            raise RuntimeError(
                "PatchTST requires training before prediction. "
                "Call fit() first — PatchTST is a supervised baseline, not a zero-shot model."
            )

        n_samples, seq_len, n_channels = X.shape
        df = self._prepare_neuralforecast_data(X)

        forecast_df = self.nf.predict(df=df)
        forecast_df = forecast_df.reset_index()

        # neuralforecast predict() returns a DataFrame with columns:
        # unique_id, ds, PatchTST (the model name)
        model_col = [c for c in forecast_df.columns if c not in ("unique_id", "ds")][0]
        unique_ids = [str(i) for i in range(n_samples)]
        median_preds = np.stack(
            [
                forecast_df[forecast_df["unique_id"] == uid][model_col].values[:horizon]
                for uid in unique_ids
            ],
            axis=0,
        )  # (n_samples, horizon)

        # Broadcast univariate forecast to multivariate output shape
        predictions = np.stack([median_preds] * n_channels, axis=-1)  # (n_samples, horizon, n_channels)

        return {
            "predictions": predictions,
            "model": self.model_name,
        }

    def few_shot_adapt(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        **kwargs
    ) -> None:
        """Train on few-shot data"""
        self.fit(X_train, y_train)
