"""
Lag-Llama Time Series Foundation Model Wrapper
Paper: arXiv:2310.08278
HuggingFace: time-series-foundation-models/Lag-Llama

Installation:
    pip install git+https://github.com/time-series-foundation-models/lag-llama.git
"""

import pickle

import torch
import numpy as np
import pandas as pd
import inspect
from typing import Dict, Optional, Union
from .base import BaseTSFMWrapper


class LagLlamaWrapper(BaseTSFMWrapper):
    """Wrapper for Lag-Llama foundation model (probabilistic, univariate)."""

    def __init__(
        self,
        model_id: str = "time-series-foundation-models/Lag-Llama",
        context_length: int = 32,
        num_samples: int = 5,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__("Lag-Llama", device)
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.estimator = None
        self.predictor = None  # set after few_shot_adapt
        self.supports_few_shot = True
        self.few_shot_head = None
        self.few_shot_target_shape = None
        self.few_shot_hidden_dim = kwargs.get("few_shot_hidden_dim", 128)

    def load_model(self) -> None:
        """Download checkpoint from HuggingFace and initialise the estimator."""
        try:
            from huggingface_hub import hf_hub_download
            import torch
            try:
                from lag_llama.gluon.estimator import LagLlamaEstimator
            except ImportError:
                from lag_llama.gluonts.estimator import LagLlamaEstimator
        except ImportError:
            raise ImportError(
                "lag-llama not found. Install with:\n"
                "  pip install git+https://github.com/time-series-foundation-models/lag-llama.git"
            )

        ckpt_path = hf_hub_download(
            repo_id=self.model_id,
            filename="lag-llama.ckpt",
        )

        checkpoint = None
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except pickle.UnpicklingError as exc:
            if "Weights only load failed" not in str(exc):
                raise

            try:
                from gluonts.torch.distributions.studentT import StudentTOutput

                if hasattr(torch.serialization, "add_safe_globals"):
                    torch.serialization.add_safe_globals([StudentTOutput])
            except Exception:
                pass

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        ckpt_hparams = checkpoint.get("hyper_parameters", {})
        ckpt_model_kwargs = ckpt_hparams.get("model_kwargs", {})

        lags_seq = ckpt_model_kwargs.get("lags_seq", [])
        expanded_default_lags = ["Q", "M", "W", "D", "H", "T", "S"]
        use_default_lag_tokens = isinstance(lags_seq, list) and len(lags_seq) == 84

        estimator_kwargs = {
            "ckpt_path": ckpt_path,
            "prediction_length": int(ckpt_hparams.get("prediction_length", 1)),
            "context_length": int(ckpt_hparams.get("context_length", self.context_length)),
            "device": torch.device(self.device),
            "time_feat": ckpt_model_kwargs.get("time_feat", True),
            "lags_seq": expanded_default_lags if use_default_lag_tokens else lags_seq,
            "input_size": ckpt_model_kwargs.get("input_size", 1),
            "n_layer": ckpt_model_kwargs.get("n_layer", 8),
            "n_embd_per_head": ckpt_model_kwargs.get("n_embd_per_head", 16),
            "n_head": ckpt_model_kwargs.get("n_head", 9),
            "scaling": ckpt_model_kwargs.get("scaling", "robust"),
            "dropout": ckpt_model_kwargs.get("dropout", 0.0),
            "rope_scaling": ckpt_model_kwargs.get("rope_scaling", None),
            "max_context_length": ckpt_model_kwargs.get("max_context_length", 2048),
        }

        signature = inspect.signature(LagLlamaEstimator.__init__)
        if "num_parallel_samples" in signature.parameters:
            estimator_kwargs["num_parallel_samples"] = self.num_samples
        elif "num_samples" in signature.parameters:
            estimator_kwargs["num_samples"] = self.num_samples

        self.estimator = LagLlamaEstimator(**estimator_kwargs)
        self.is_loaded = True
        print(f"Lag-Llama loaded on {self.device}")

    def _to_gluonts_dataset(self, X: np.ndarray) -> "ListDataset":
        """Convert (n_samples, seq_len) float32 array to GluonTS ListDataset."""
        from gluonts.dataset.common import ListDataset

        base_start = pd.Timestamp("2020-01-01")
        entries = [
            {"start": base_start, "target": X[i].astype(np.float32)}
            for i in range(X.shape[0])
        ]
        return ListDataset(entries, freq="h")

    def _predict_zero_shot_base(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Base probabilistic forecast from the pretrained Lag-Llama model."""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Lag-Llama is univariate: average channels if multivariate input
        if X.ndim == 3:
            n_channels = X.shape[2]
            X_uni = X.mean(axis=-1)   # (n_samples, seq_len)
        else:
            n_channels = 1
            X_uni = X

        # Update horizon on the estimator
        self.estimator.prediction_length = horizon

        transformation = self.estimator.create_transformation()
        lightning_module = self.estimator.create_lightning_module()

        if self.predictor is None:
            # Zero-shot: use base pretrained weights
            predictor = self.estimator.create_predictor(
                transformation, lightning_module
            )
        else:
            predictor = self.predictor

        dataset = self._to_gluonts_dataset(X_uni)
        forecasts = list(predictor.predict(dataset))

        median_preds = np.stack(
            [f.quantile(0.5) for f in forecasts], axis=0
        )  # (n_samples, horizon)

        # Expand to (n_samples, horizon, n_channels) to match other wrappers
        predictions = np.stack([median_preds] * n_channels, axis=-1)

        return {
            "predictions": predictions,
            "model": self.model_name,
            "quantiles": {
                "q10": np.stack([f.quantile(0.1) for f in forecasts], axis=0),
                "q90": np.stack([f.quantile(0.9) for f in forecasts], axis=0),
            },
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
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Forecast with the pretrained model plus optional few-shot calibration head."""
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
            "predictions": predictions,
            "model": self.model_name,
        }

    def few_shot_adapt(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        epochs: int = 10,
        lr: float = 1e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        **kwargs,
    ) -> None:
        """Fit a lightweight calibration head on top of Lag-Llama predictions."""
        if not self.is_loaded:
            self.load_model()

        forecast_horizon = kwargs.get("forecast_horizon")
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train_tensor = y_train.detach().cpu()
        else:
            y_train_tensor = torch.FloatTensor(y_train)

        if forecast_horizon is None:
            forecast_horizon = int(y_train_tensor.shape[1]) if y_train_tensor.ndim > 1 else 1

        base_predictions = self._predict_zero_shot_base(X_train, horizon=forecast_horizon)["predictions"]
        features = self._build_few_shot_features(X_train, base_predictions)

        if y_train_tensor.ndim == 1:
            targets = y_train_tensor.view(-1, 1).to(self.device)
            self.few_shot_target_shape = {"kind": "scalar"}
        elif y_train_tensor.ndim == 2:
            targets = y_train_tensor.to(self.device)
            self.few_shot_target_shape = {
                "kind": "sequence",
                "horizon": y_train_tensor.shape[1],
                "channels": 1,
            }
        else:
            targets = y_train_tensor.reshape(y_train_tensor.shape[0], -1).to(self.device)
            self.few_shot_target_shape = {
                "kind": "sequence",
                "horizon": y_train_tensor.shape[1],
                "channels": y_train_tensor.shape[2],
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
                print(f"Lag-Llama few-shot epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

        self.few_shot_head.eval()
        print("Lag-Llama few-shot adaptation complete")
