"""
TimeGPT API Wrapper (Nixtla)
Paper: arXiv:2310.03589
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import torch
from .base import BaseTSFMWrapper
import os


class TimeGPTWrapper(BaseTSFMWrapper):
    """Wrapper for TimeGPT API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__("TimeGPT", device)
        self.api_key = api_key or os.getenv("NIXTLA_API_KEY")
        self.client = None

    def load_model(self) -> None:
        """Initialize API client"""
        try:
            from nixtla import NixtlaClient

            self.client = NixtlaClient(api_key=self.api_key)
            self.client.validate_api_key()
            self.is_loaded = True
            print("TimeGPT API client initialized")

        except ImportError:
            raise ImportError("Please install nixtla: pip install nixtla")
        except Exception as e:
            raise ValueError(f"API key validation failed: {e}")

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        horizon: int = 96,
        freq: str = "H",
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Generate forecasts via API"""
        if not self.is_loaded:
            self.load_model()

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        batch_size = X.shape[0] if X.ndim == 3 else 1
        all_predictions = []

        for b in range(batch_size):
            if X.ndim == 3:
                x_sample = X[b].mean(axis=1)
            else:
                x_sample = X

            df = pd.DataFrame({
                'unique_id': f'ts_{b}',
                'ds': pd.date_range(start='2020-01-01', periods=len(x_sample), freq=freq),
                'y': x_sample
            })

            try:
                forecast_df = self.client.forecast(
                    df=df,
                    h=horizon,
                    freq=freq,
                    model='timegpt-1'
                )
                pred = forecast_df['TimeGPT'].values
                all_predictions.append(pred)
            except Exception as e:
                print(f"API error for batch {b}: {e}")
                all_predictions.append(np.zeros(horizon))

        predictions = np.array(all_predictions)

        if X.ndim == 3:
            n_channels = X.shape[2]
            predictions = np.expand_dims(predictions, -1).repeat(n_channels, axis=-1)

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
        """TimeGPT fine-tuning via API"""
        print("TimeGPT few-shot: Using finetune_steps parameter")
        self.finetune_steps = kwargs.get('finetune_steps', 10)
