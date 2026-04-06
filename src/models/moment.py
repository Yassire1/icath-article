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
        model_id: str = "AutonLab/MOMENT-1-large",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__("MOMENT", device)
        self.model_id = model_id
        self.task_head = None

    def load_model(self) -> None:
        """Load MOMENT from HuggingFace"""
        try:
            from momentfm import MOMENTPipeline

            self.model = MOMENTPipeline.from_pretrained(
                self.model_id,
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': 96
                }
            )
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

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts"""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        if X.dim() == 3 and X.shape[2] != X.shape[1]:
            X = X.transpose(1, 2)

        with torch.no_grad():
            try:
                outputs = self.model(X, forecast_horizon=horizon)
                predictions = outputs.forecast.cpu().numpy()
            except:
                outputs = self.model(X)
                predictions = outputs.last_hidden_state[:, -horizon:, :].cpu().numpy()

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
        """Few-shot adaptation using LoRA"""
        from peft import LoraConfig, get_peft_model, TaskType

        if not self.is_loaded:
            self.load_model()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train.transpose(1, 2))
            predictions = outputs.last_hidden_state[:, -y_train.shape[1]:, :]
            loss = criterion(predictions, y_train.transpose(1, 2))
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.model.eval()
        print("Few-shot adaptation complete")
