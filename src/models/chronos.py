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
        model_id: str = "amazon/chronos-t5-mini",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__("Chronos", device)
        self.model_id = model_id
        self.max_native_prediction_length = int(kwargs.get("max_native_prediction_length", 64))
        self._long_horizon_notice_printed = False

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
        num_samples: int = 5,
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

        forecast_mean, forecast_std = self._rollout_univariate(
            X_avg,
            horizon=horizon,
            num_samples=num_samples,
        )

        predictions = forecast_mean.cpu().numpy()
        uncertainties = forecast_std.cpu().numpy()

        predictions = np.expand_dims(predictions, -1).repeat(n_channels, axis=-1)

        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'model': self.model_name
        }

    def _forecast_univariate(
        self,
        context: torch.Tensor,
        prediction_length: int,
        num_samples: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.model.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )

    def _rollout_univariate(
        self,
        context: torch.Tensor,
        horizon: int,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if horizon <= self.max_native_prediction_length:
            forecasts = self._forecast_univariate(context, horizon, num_samples)
            return forecasts.mean(dim=1), forecasts.std(dim=1)

        if not self._long_horizon_notice_printed:
            print(
                f"Chronos horizon {horizon} exceeds native recommendation "
                f"{self.max_native_prediction_length}; using chunked rollout."
            )
            self._long_horizon_notice_printed = True

        seq_len = context.shape[1]
        current_context = context
        mean_segments = []
        std_segments = []
        remaining = horizon

        while remaining > 0:
            chunk_length = min(self.max_native_prediction_length, remaining)
            forecasts = self._forecast_univariate(current_context, chunk_length, num_samples)
            forecast_mean = forecasts.mean(dim=1)
            forecast_std = forecasts.std(dim=1)

            mean_segments.append(forecast_mean)
            std_segments.append(forecast_std)
            remaining -= chunk_length

            if remaining > 0:
                current_context = torch.cat(
                    [current_context, forecast_mean.to(current_context.device)],
                    dim=1,
                )[:, -seq_len:]

        return torch.cat(mean_segments, dim=1), torch.cat(std_segments, dim=1)

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

            pred_c, _ = self._rollout_univariate(X_c, horizon=horizon, num_samples=num_samples)
            pred_c = pred_c.cpu().numpy()
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
