"""
Few-shot experiment runner with LoRA adaptation
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


def run_few_shot_experiment(
    model_name: str,
    dataset_name: str,
    data_path: Path,
    task: str = "forecasting",
    horizon: int = 96,
    device: str = "cuda",
    train_ratio: float = 0.01,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    use_wandb: bool = True
) -> Dict:
    """Run few-shot evaluation with LoRA adaptation"""
    print(f"\n{'='*60}")
    print(f"Few-Shot: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    data = torch.load(data_path / "processed_data.pt")

    X_train = data['train_X']
    y_train = data['train_y']
    X_test = data['test_X']
    y_test = data['test_y']

    # Subsample training data
    n_train = max(int(len(X_train) * train_ratio), 32)
    X_train_sub = X_train[:n_train]
    y_train_sub = y_train[:n_train]

    print(f"Training samples (1%): {n_train}")
    print(f"Test samples: {len(X_test)}")

    model = get_model(model_name, device=device)
    model.load_model()

    print("Running few-shot adaptation with LoRA...")
    model.few_shot_adapt(
        X_train_sub, y_train_sub,
        epochs=epochs,
        lr=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    print("Running inference...")
    results = model.predict(X_test, horizon=horizon)
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

    print(f"Few-Shot Results: {metrics}")

    return {
        'model': model_name,
        'dataset': dataset_name,
        'task': task,
        'scenario': 'few_shot',
        'train_samples': n_train,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }


def run_all_few_shot(
    models: List[str],
    datasets: List[str],
    processed_dir: str = "data/processed",
    results_dir: str = "results/few_shot",
    task: str = "forecasting",
    device: str = "cuda",
    use_wandb: bool = True
) -> pd.DataFrame:
    """Run few-shot experiments for all model-dataset combinations"""
    processed_path = Path(processed_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset in tqdm(datasets, desc="Datasets"):
        dataset_path = processed_path / dataset

        if not dataset_path.exists():
            print(f"Warning: {dataset} not found, skipping")
            continue

        subsets = list(dataset_path.glob("*/processed_data.pt"))
        if not subsets:
            subsets = [dataset_path / "processed_data.pt"]

        for subset_path in subsets:
            subset_dir = subset_path.parent
            subset_name = f"{dataset}/{subset_dir.name}" if subset_dir != dataset_path else dataset

            for model_name in tqdm(models, desc=f"Models ({subset_name})", leave=False):
                try:
                    result = run_few_shot_experiment(
                        model_name=model_name,
                        dataset_name=subset_name,
                        data_path=subset_dir,
                        task=task,
                        device=device,
                        use_wandb=use_wandb
                    )
                    all_results.append(result)

                    with open(results_path / f"{model_name}_{dataset}_few_shot.json", 'w') as f:
                        json.dump(result, f, indent=2)

                except Exception as e:
                    print(f"Error with {model_name} on {subset_name}: {e}")
                    all_results.append({
                        'model': model_name,
                        'dataset': subset_name,
                        'error': str(e)
                    })

    df = pd.DataFrame(all_results)
    df.to_csv(results_path / "few_shot_results.csv", index=False)

    print(f"\nResults saved to {results_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few-shot experiments")
    parser.add_argument("--models", type=str, default="moment,sundial")
    parser.add_argument("--datasets", type=str, default="cmapss")
    parser.add_argument("--output", type=str, default="results/few_shot")
    parser.add_argument("--task", type=str, default="forecasting")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    results = run_all_few_shot(
        models=models,
        datasets=datasets,
        processed_dir="data/processed",
        results_dir=args.output,
        task=args.task,
        device=args.device
    )

    print("\nSummary:")
    print(results)
