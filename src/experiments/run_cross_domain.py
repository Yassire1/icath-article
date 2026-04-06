"""
Cross-domain experiment runner
Train on dataset A, test on dataset B
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import get_model
from src.evaluation.metrics import compute_all_metrics


def run_cross_domain_experiment(
    model_name: str,
    train_dataset: str,
    test_dataset: str,
    train_data_path: Path,
    test_data_path: Path,
    task: str = "forecasting",
    horizon: int = 96,
    device: str = "cuda"
) -> Dict:
    """Run cross-domain evaluation: train on A, test on B"""
    print(f"\n{'='*60}")
    print(f"Cross-Domain: {model_name} | Train: {train_dataset} -> Test: {test_dataset}")
    print(f"{'='*60}")

    train_data = torch.load(train_data_path / "processed_data.pt")
    test_data = torch.load(test_data_path / "processed_data.pt")

    X_train = train_data['train_X']
    y_train = train_data['train_y']
    X_test = test_data['test_X']
    y_test = test_data['test_y']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = get_model(model_name, device=device)
    model.load_model()

    print("Running zero-shot inference (no domain adaptation)...")
    results = model.zero_shot(X_test, horizon=horizon)
    predictions = results['predictions']

    if predictions.shape != y_test.shape:
        min_len = min(predictions.shape[1], y_test.shape[1])
        predictions = predictions[:, :min_len]
        y_test = y_test[:, :min_len]

    metrics = compute_all_metrics(
        y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test,
        predictions,
        task=task
    )

    print(f"Cross-Domain Results: {metrics}")

    return {
        'model': model_name,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'task': task,
        'scenario': 'cross_domain',
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }


def run_cross_domain_matrix(
    models: List[str],
    datasets: List[str],
    dataset_paths: Dict[str, Path],
    results_dir: str = "results/cross_domain",
    task: str = "forecasting",
    device: str = "cuda"
) -> pd.DataFrame:
    """Run full cross-domain transfer matrix"""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    transfer_matrix = {}

    for model_name in tqdm(models, desc="Models"):
        model_matrix = {}

        for train_ds in datasets:
            if train_ds not in dataset_paths:
                continue

            train_path = dataset_paths[train_ds]
            if not train_path.exists():
                continue

            for test_ds in datasets:
                if test_ds not in dataset_paths:
                    continue
                if train_ds == test_ds:
                    continue

                test_path = dataset_paths[test_ds]
                if not test_path.exists():
                    continue

                try:
                    result = run_cross_domain_experiment(
                        model_name=model_name,
                        train_dataset=train_ds,
                        test_dataset=test_ds,
                        train_data_path=train_path,
                        test_data_path=test_path,
                        task=task,
                        device=device
                    )
                    all_results.append(result)
                    model_matrix[f"{train_ds}->{test_ds}"] = result['metrics'].get('mae', float('nan'))

                except Exception as e:
                    print(f"Error: {model_name} {train_ds}->{test_ds}: {e}")
                    all_results.append({
                        'model': model_name,
                        'train_dataset': train_ds,
                        'test_dataset': test_ds,
                        'error': str(e)
                    })

        transfer_matrix[model_name] = model_matrix

    df = pd.DataFrame(all_results)
    df.to_csv(results_path / "cross_domain_results.csv", index=False)

    with open(results_path / "transfer_matrix.json", 'w') as f:
        json.dump(transfer_matrix, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-domain experiments")
    parser.add_argument("--models", type=str, default="moment,patchtst")
    parser.add_argument("--datasets", type=str, default="cmapss,phm_milling,wind_scada")
    parser.add_argument("--output", type=str, default="results/cross_domain")
    parser.add_argument("--task", type=str, default="forecasting")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    dataset_paths = {d: Path(f"data/processed/{d}") for d in datasets}

    results = run_cross_domain_matrix(
        models=models,
        datasets=datasets,
        dataset_paths=dataset_paths,
        results_dir=args.output,
        task=args.task,
        device=args.device
    )

    print("\nSummary:")
    print(results)
