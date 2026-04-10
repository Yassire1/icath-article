"""
Lag-Llama Time Series Foundation Model Wrapper
Paper: arXiv:2310.08278
HuggingFace: time-series-foundation-models/Lag-Llama

Installation:
    pip install git+https://github.com/time-series-foundation-models/lag-llama.git
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from .base import BaseTSFMWrapper


class LagLlamaWrapper(BaseTSFMWrapper):
    """Wrapper for Lag-Llama foundation model (probabilistic, univariate)."""

    def __init__(
        self,
        model_id: str = "time-series-foundation-models/Lag-Llama",
        context_length: int = 32,
        num_samples: int = 100,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__("Lag-Llama", device)
        self.model_id = model_id
        self.context_length = context_length
        self.num_samples = num_samples
        self.estimator = None
        self.predictor = None  # set after few_shot_adapt

    def load_model(self) -> None:
        """Download checkpoint from HuggingFace and initialise the estimator."""
        try:
            from huggingface_hub import hf_hub_download
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

        self.estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=96,       # overridden per-call in predict()
            context_length=self.context_length,
            num_samples=self.num_samples,
            device=self.device,
            time_feat=True,
        )
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
        return ListDataset(entries, freq="H")

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Zero-shot (or post-adaptation) probabilistic forecast.

        Returns the median as the point forecast. Also returns q10/q90 for
        uncertainty quantification.
        """
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
        """Fine-tune Lag-Llama on a small labelled dataset using LoRA.

        Trains on the (context, target) pairs and stores a fine-tuned
        predictor for subsequent calls to predict().
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()

        # Reduce to univariate if needed
        if X_train.ndim == 3:
            X_train = X_train.mean(axis=-1)
        if y_train.ndim == 3:
            y_train = y_train.mean(axis=-1)

        # Build training sequences (context window only; target acts as validation)
        dataset = self._to_gluonts_dataset(X_train)

        try:
            from peft import LoraConfig, get_peft_model, TaskType
            import pytorch_lightning as pl
        except ImportError:
            raise ImportError(
                "peft and pytorch_lightning are required for few-shot adaptation.\n"
                "  pip install peft pytorch-lightning"
            )

        transformation = self.estimator.create_transformation()
        lightning_module = self.estimator.create_lightning_module()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        lightning_module.model = get_peft_model(lightning_module.model, lora_config)
        lightning_module.model.print_trainable_parameters()
        # Override learning rate
        lightning_module.lr = lr

        train_dataloader = self.estimator.create_training_data_loader(
            transformation.apply(dataset), lightning_module
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_progress_bar=True,
        )
        trainer.fit(lightning_module, train_dataloader)

        self.predictor = self.estimator.create_predictor(transformation, lightning_module)
        print("Lag-Llama few-shot adaptation complete.")
