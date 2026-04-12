"""
SCADA-optimized preprocessing pipeline for industrial PdM datasets
Key innovations:
1. Chronological splits (no data leakage)
2. Per-sensor-family normalization
3. Kalman-based imputation
4. Health indicator extraction for RUL
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import interpolate
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import warnings
warnings.filterwarnings('ignore')


class SCADAPreprocessor:
    """Industrial SCADA preprocessing pipeline"""

    def __init__(
        self,
        lookback: int = 512,
        horizon: int = 96,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        normalization: str = "standard",
        seed: int = 42
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.normalization = normalization
        self.seed = seed
        self.scalers = {}

    def _get_scaler(self, method: str):
        """Get appropriate scaler"""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        return scalers.get(method, StandardScaler())

    def _kalman_impute(self, data: np.ndarray) -> np.ndarray:
        """Simple Kalman-like forward-backward imputation"""
        df = pd.DataFrame(data)
        df = df.ffill().bfill()
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(df.mean())
        return df.values

    def _chronological_split(
        self,
        data: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Chronological train/val/test split - NO shuffling"""
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        splits = {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }

        if targets is not None:
            splits["train_targets"] = targets[:train_end]
            splits["val_targets"] = targets[train_end:val_end]
            splits["test_targets"] = targets[val_end:]

        return splits

    def _normalize(
        self,
        train: np.ndarray,
        val: np.ndarray,
        test: np.ndarray,
        sensor_families: Optional[List[List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-sensor-family normalization - fit on train, transform val/test"""
        if sensor_families is None:
            sensor_families = [list(range(train.shape[1]))]

        train_norm = np.zeros_like(train)
        val_norm = np.zeros_like(val)
        test_norm = np.zeros_like(test)

        for i, family in enumerate(sensor_families):
            scaler = self._get_scaler(self.normalization)
            train_norm[:, family] = scaler.fit_transform(train[:, family])
            val_norm[:, family] = scaler.transform(val[:, family])
            test_norm[:, family] = scaler.transform(test[:, family])
            self.scalers[f"family_{i}"] = scaler

        return train_norm, val_norm, test_norm

    def _create_sequences(
        self,
        data: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences"""
        X, y = [], []
        n = len(data)

        if targets is not None:
            # Target-based tasks (e.g., RUL): one label per lookback window.
            # Use the shared valid length to avoid indexing past targets.
            max_start = min(n, len(targets)) - self.lookback
            for i in range(max(0, max_start)):
                X.append(data[i:i + self.lookback])
                y.append(targets[i + self.lookback])
        else:
            # Forecasting tasks: predict a horizon-sized future slice.
            max_start = n - self.lookback - self.horizon + 1
            for i in range(max(0, max_start)):
                X.append(data[i:i + self.lookback])
                y.append(data[i + self.lookback:i + self.lookback + self.horizon])

        return np.array(X), np.array(y)

    def _compute_rul_labels(
        self,
        data: np.ndarray,
        max_rul: int = 125,
        method: str = "piecewise_linear"
    ) -> np.ndarray:
        """Compute RUL labels with piecewise linear degradation"""
        n = len(data)

        if method == "piecewise_linear":
            rul = np.arange(n - 1, -1, -1)
            rul = np.minimum(rul, max_rul)
        else:
            rul = np.arange(n - 1, -1, -1)

        return rul

    def process_cmapss(
        self,
        data_path: Path,
        subset: str = "FD001"
    ) -> Dict[str, torch.Tensor]:
        """Process C-MAPSS dataset"""
        cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(3)] + [f'sensor_{i}' for i in range(21)]

        train_df = pd.read_csv(data_path / f"train_{subset}.txt", sep=r'\s+', header=None, names=cols)
        pd.read_csv(data_path / f"test_{subset}.txt", sep=r'\s+', header=None, names=cols)
        pd.read_csv(data_path / f"RUL_{subset}.txt", sep=r'\s+', header=None, names=['rul'])

        # Exclude constant sensors
        sensor_cols = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20]]

        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        for unit_id in train_df['unit'].unique():
            unit_data = train_df[train_df['unit'] == unit_id][sensor_cols].values
            unit_data = self._kalman_impute(unit_data)
            rul = self._compute_rul_labels(unit_data)

            splits = self._chronological_split(unit_data, rul)
            train_n, val_n, test_n = self._normalize(
                splits['train'], splits['val'], splits['test']
            )

            # Build windows on the full normalized unit trajectory while assigning
            # each window to a split based on where its target timestamp falls.
            # This preserves chronological split boundaries while allowing val/test
            # windows even when those segments are shorter than the lookback.
            unit_norm = np.concatenate([train_n, val_n, test_n], axis=0)
            train_end = len(train_n)
            val_end = train_end + len(val_n)

            def _windows_for_target_range(start_idx: int, end_idx: int):
                X_part, y_part = [], []
                label_start = max(start_idx, self.lookback)
                for t in range(label_start, end_idx):
                    X_part.append(unit_norm[t - self.lookback:t])
                    y_part.append(rul[t])
                if not X_part:
                    return None, None
                return np.array(X_part, dtype=np.float32), np.array(y_part, dtype=np.float32)

            X_tr, y_tr = _windows_for_target_range(0, train_end)
            X_va, y_va = _windows_for_target_range(train_end, val_end)
            X_te, y_te = _windows_for_target_range(val_end, len(unit_norm))

            if X_tr is not None:
                all_X_train.append(X_tr)
                all_y_train.append(y_tr)
            if X_va is not None:
                all_X_val.append(X_va)
                all_y_val.append(y_va)
            if X_te is not None:
                all_X_test.append(X_te)
                all_y_test.append(y_te)

        def _cat_or_empty(arrays: List[np.ndarray], tail_shape: Tuple[int, ...]) -> torch.Tensor:
            if arrays:
                return torch.FloatTensor(np.concatenate(arrays, axis=0))
            return torch.empty((0, *tail_shape), dtype=torch.float32)

        def _cat_targets_or_empty(arrays: List[np.ndarray]) -> torch.Tensor:
            if arrays:
                return torch.FloatTensor(np.concatenate(arrays, axis=0))
            return torch.empty((0,), dtype=torch.float32)

        result = {
            'train_X': _cat_or_empty(all_X_train, (self.lookback, len(sensor_cols))),
            'train_y': _cat_targets_or_empty(all_y_train),
            'val_X': _cat_or_empty(all_X_val, (self.lookback, len(sensor_cols))),
            'val_y': _cat_targets_or_empty(all_y_val),
            'test_X': _cat_or_empty(all_X_test, (self.lookback, len(sensor_cols))),
            'test_y': _cat_targets_or_empty(all_y_test),
            'task': 'rul',
            'dataset': 'cmapss',
            'subset': subset,
            'num_channels': len(sensor_cols),
            'lookback': self.lookback,
            'horizon': self.horizon
        }

        print(f"C-MAPSS {subset} processed:")
        print(f"  Train: {result['train_X'].shape}")
        print(f"  Val: {result['val_X'].shape}")
        print(f"  Test: {result['test_X'].shape}")

        return result

    def process_generic_csv(
        self,
        data_path: Path,
        timestamp_col: Optional[str] = None,
        target_col: Optional[str] = None,
        task: str = "forecasting"
    ) -> Dict[str, torch.Tensor]:
        """Process generic CSV time series data"""
        df = pd.read_csv(data_path)

        if timestamp_col and timestamp_col in df.columns:
            df = df.drop(columns=[timestamp_col])

        if target_col and target_col in df.columns:
            targets = df[target_col].values
            df = df.drop(columns=[target_col])
        else:
            targets = None

        data = df.values.astype(np.float32)
        data = self._kalman_impute(data)

        splits = self._chronological_split(data, targets)
        train_n, val_n, test_n = self._normalize(
            splits['train'], splits['val'], splits['test']
        )

        if task == "forecasting":
            X_tr, y_tr = self._create_sequences(train_n)
            X_va, y_va = self._create_sequences(val_n)
            X_te, y_te = self._create_sequences(test_n)
        else:
            X_tr, y_tr = self._create_sequences(train_n, splits.get('train_targets'))
            X_va, y_va = self._create_sequences(val_n, splits.get('val_targets'))
            X_te, y_te = self._create_sequences(test_n, splits.get('test_targets'))

        return {
            'train_X': torch.FloatTensor(X_tr),
            'train_y': torch.FloatTensor(y_tr),
            'val_X': torch.FloatTensor(X_va),
            'val_y': torch.FloatTensor(y_va),
            'test_X': torch.FloatTensor(X_te),
            'test_y': torch.FloatTensor(y_te),
            'task': task,
            'num_channels': data.shape[1],
            'lookback': self.lookback,
            'horizon': self.horizon
        }

    def save_processed(self, data: Dict, output_path: Path):
        """Save processed tensors"""
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path / "processed_data.pt")
        print(f"Saved to {output_path / 'processed_data.pt'}")

    def load_processed(self, input_path: Path) -> Dict:
        """Load processed tensors"""
        return torch.load(input_path / "processed_data.pt")


class TSFMDataset(Dataset):
    """PyTorch Dataset for TSFM benchmarking"""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, task: str = "forecasting"):
        self.X = X
        self.y = y
        self.task = task

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def preprocess_all_datasets(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    config_path: str = "config/config.yaml"
):
    """Preprocess all datasets"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    preprocessor = SCADAPreprocessor(
        lookback=config['preprocessing']['lookback_window'],
        horizon=config['preprocessing']['forecast_horizon'],
        train_ratio=config['preprocessing']['train_ratio'],
        val_ratio=config['preprocessing']['val_ratio'],
        normalization=config['preprocessing']['normalization']
    )

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    cmapss_path = raw_path / "cmapss"
    if cmapss_path.exists() and (cmapss_path / "train_FD001.txt").exists():
        print("\n" + "=" * 50)
        print("Processing C-MAPSS")
        print("=" * 50)
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            try:
                data = preprocessor.process_cmapss(cmapss_path, subset)
                preprocessor.save_processed(data, proc_path / "cmapss" / subset)
            except Exception as e:
                print(f"Error processing {subset}: {e}")

    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print("=" * 50)


if __name__ == "__main__":
    preprocess_all_datasets()
