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
        self.supports_few_shot = True
        self.few_shot_head = None
        self.few_shot_target_shape = None
        self.few_shot_hidden_dim = kwargs.get("few_shot_hidden_dim", 128)

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

    def _predict_zero_shot_base(
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

    def _build_few_shot_features(
        self,
        X: Union[np.ndarray, torch.Tensor],
        base_predictions: np.ndarray,
    ) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

        if X_np.ndim == 2:
            X_np = X_np[:, :, None]

        context_last = X_np[:, -1, :]
        context_mean = X_np.mean(axis=1)
        context_std = X_np.std(axis=1)
        base_flat = base_predictions.reshape(base_predictions.shape[0], -1)
        features = np.concatenate([base_flat, context_last, context_mean, context_std], axis=1)
        return torch.FloatTensor(features).to(self.device)

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        num_samples: int = 5,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        base_result = self._predict_zero_shot_base(
            X,
            horizon=horizon,
            num_samples=num_samples,
            **kwargs,
        )

        if self.few_shot_head is None or self.few_shot_target_shape is None:
            return base_result

        with torch.no_grad():
            features = self._build_few_shot_features(X, base_result["predictions"])
            adapted = self.few_shot_head(features).cpu().numpy()

        target_kind = self.few_shot_target_shape["kind"]
        if target_kind == "scalar":
            predictions = adapted.reshape(adapted.shape[0], 1, 1)
        else:
            target_horizon = self.few_shot_target_shape["horizon"]
            target_channels = self.few_shot_target_shape["channels"]
            predictions = adapted.reshape(adapted.shape[0], target_horizon, target_channels)

        return {
            "predictions": predictions,
            "model": self.model_name,
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
        epochs: int = 10,
        lr: float = 1e-4,
        **kwargs
    ) -> None:
        """Fit a lightweight calibration head on top of Chronos forecasts."""
        if not self.is_loaded:
            self.load_model()

        forecast_horizon = kwargs.get("forecast_horizon")
        if isinstance(X_train, np.ndarray):
            X_train = torch.FloatTensor(X_train)
        if isinstance(y_train, np.ndarray):
            y_train = torch.FloatTensor(y_train)

        if forecast_horizon is None:
            forecast_horizon = int(y_train.shape[1]) if y_train.ndim > 1 else 1

        with torch.no_grad():
            base_predictions = self._predict_zero_shot_base(
                X_train,
                horizon=forecast_horizon,
            )["predictions"]
        features = self._build_few_shot_features(X_train, base_predictions)

        if y_train.ndim == 1:
            targets = y_train.view(-1, 1).to(self.device)
            self.few_shot_target_shape = {"kind": "scalar"}
        elif y_train.ndim == 2:
            targets = y_train.to(self.device)
            self.few_shot_target_shape = {
                "kind": "sequence",
                "horizon": y_train.shape[1],
                "channels": 1,
            }
        else:
            targets = y_train.reshape(y_train.shape[0], -1).to(self.device)
            self.few_shot_target_shape = {
                "kind": "sequence",
                "horizon": y_train.shape[1],
                "channels": y_train.shape[2],
            }

        hidden_dim = min(self.few_shot_hidden_dim, max(64, features.shape[1] // 2))
        self.few_shot_head = torch.nn.Sequential(
            torch.nn.LayerNorm(features.shape[1]),
            torch.nn.Linear(features.shape[1], hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, targets.shape[1]),
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.few_shot_head.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.few_shot_head.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.few_shot_head(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % max(1, epochs // 2) == 0 or epoch == epochs - 1:
                print(f"Chronos few-shot epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

        self.few_shot_head.eval()
        print("Chronos few-shot adaptation complete")
