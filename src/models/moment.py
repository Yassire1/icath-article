"""
MOMENT Time Series Foundation Model Wrapper
Paper: arXiv:2402.03885
"""

import torch
import numpy as np
from typing import Dict, Optional, Union
from .base import BaseTSFMWrapper


class MOMENTWrapper(BaseTSFMWrapper):
    """Wrapper for MOMENT foundation model"""

    def __init__(
        self,
        model_id: str = "AutonLab/MOMENT-1-small",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__("MOMENT", device)
        self.model_id = model_id
        self.task_head = None
        self.seq_len = kwargs.get("seq_len", 512)
        self.supports_few_shot = True
        self.few_shot_head = None
        self.few_shot_target_shape = None
        self.few_shot_hidden_dim = kwargs.get("few_shot_hidden_dim", 256)

    def load_model(self) -> None:
        """Load MOMENT from HuggingFace"""
        try:
            from momentfm import MOMENTPipeline

            self.model = MOMENTPipeline.from_pretrained(
                self.model_id,
                cache_dir=None,
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': 96
                }
            )
            if hasattr(self.model, "init"):
                self.model.init()
            self.model.to(self.device)
            self.is_loaded = True
            print(f"MOMENT loaded on {self.device}")

        except ImportError:
            from transformers import AutoModel, AutoConfig

            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id,
                config=config,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.is_loaded = True
            print(f"MOMENT loaded via transformers on {self.device}")

    def _predict_zero_shot_base(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate the pretrained baseline forecast used by zero-shot mode."""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        if X.dim() == 3 and X.shape[2] != X.shape[1]:
            X = X.transpose(1, 2)

        # MOMENT-1 ships with a pretrained reconstruction head. Its forecasting
        # head is randomly initialized upstream, so for zero-shot/few-shot
        # benchmarking on arbitrary horizons we use the reconstruction path and
        # extrapolate from the trailing reconstructed context.
        if X.shape[-1] < self.seq_len:
            pad = self.seq_len - X.shape[-1]
            X = torch.nn.functional.pad(X, (pad, 0))
        elif X.shape[-1] > self.seq_len:
            X = X[:, :, -self.seq_len:]

        input_mask = torch.ones((X.shape[0], X.shape[-1]), device=X.device, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model.reconstruction(x_enc=X, input_mask=input_mask)
            recon = outputs.reconstruction.transpose(1, 2).cpu().numpy()

        last_step = recon[:, -1:, :]
        predictions = np.repeat(last_step, horizon, axis=1)

        return {
            'predictions': predictions,
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
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts, using the adapted head when available."""
        base_result = self._predict_zero_shot_base(X, horizon=horizon, **kwargs)

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
            'predictions': predictions,
            'model': self.model_name
        }

    def few_shot_adapt(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 10,
        lr: float = 1e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        **kwargs
    ) -> None:
        """Fit a lightweight supervised adaptation head on top of MOMENT outputs."""

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
            base_predictions = self._predict_zero_shot_base(X_train, horizon=forecast_horizon)["predictions"]
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
                print(f"MOMENT few-shot epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

        self.few_shot_head.eval()
        print("MOMENT few-shot adaptation complete")
