"""
Zero-shot experiment runner
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

from src.models import get_model, list_models
from src.evaluation.metrics import compute_all_metrics
from src.data.preprocessing import SCADAPreprocessor


def run_zero_shot_experiment(
    model_name: str,
    dataset_name: str,
    data_path: Path,
    task: str = "forecasting",
    horizon: int = 96,
    device: str = "cuda",
    use_wandb: bool = True
) -> Dict:
    """Run zero-shot evaluation for a single model-dataset pair"""
    print(f"\n{'='*60}")
    print(f"Zero-Shot: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    data = torch.load(data_path / "processed_data.pt")

    X_test = data['test_X']
    y_test = data['test_y']

    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    model = get_model(model_name, device=device)
    model.load_model()

    print("Running inference...")
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

    print(f"Results: {metrics}")

    if use_wandb:
        try:
            import wandb
            wandb.log({
                f"{dataset_name}/{model_name}/{k}": v
                for k, v in metrics.items()
            })
        except Exception as e:
            print(f"WandB logging skipped: {e}")

    return {
        'model': model_name,
        'dataset': dataset_name,
        'task': task,
        'scenario': 'zero_shot',
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'predictions_shape': list(predictions.shape),
        'test_samples': len(X_test)
    }


def run_all_zero_shot(
    models: List[str],
    datasets: List[str],
    processed_dir: str = "data/processed",
    results_dir: str = "results/zero_shot",
    task: str = "forecasting",
    device: str = "cuda",
    use_wandb: bool = True
) -> pd.DataFrame:
    """Run zero-shot experiments for all model-dataset combinations"""
    processed_path = Path(processed_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="tsfm-pdm-bench",
                name=f"zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'scenario': 'zero_shot',
                    'models': models,
                    'datasets': datasets,
                    'task': task
                }
            )
        except Exception as e:
            print(f"WandB init skipped: {e}")

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
                    result = run_zero_shot_experiment(
                        model_name=model_name,
                        dataset_name=subset_name,
                        data_path=subset_dir,
                        task=task,
                        device=device,
                        use_wandb=use_wandb
                    )
                    all_results.append(result)

                    with open(results_path / f"{model_name}_{dataset}.json", 'w') as f:
                        json.dump(result, f, indent=2)

                except Exception as e:
                    print(f"Error with {model_name} on {subset_name}: {e}")
                    all_results.append({
                        'model': model_name,
                        'dataset': subset_name,
                        'error': str(e)
                    })

    df = pd.DataFrame(all_results)
    df.to_csv(results_path / "zero_shot_results.csv", index=False)

    if use_wandb:
        try:
            import wandb
            wandb.save(str(results_path / "zero_shot_results.csv"))
            wandb.finish()
        except:
            pass

    print(f"\nResults saved to {results_path}")
    return df


def create_results_table(
    results_df: pd.DataFrame,
    metric: str = "mae",
    output_path: Optional[Path] = None
) -> str:
    """Create LaTeX table from results"""
    pivot = results_df.pivot_table(
        values=f'metrics',
        index='model',
        columns='dataset',
        aggfunc=lambda x: x.iloc[0].get(metric, np.nan) if isinstance(x.iloc[0], dict) else np.nan
    )

    latex = pivot.to_latex(
        float_format="%.3f",
        caption=f"Zero-Shot {metric.upper()} Results",
        label=f"tab:zero_shot_{metric}"
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot experiments")
    parser.add_argument("--models", type=str, default="moment,chronos,lag_llama", help="Comma-separated model names")
    parser.add_argument("--datasets", type=str, default="cmapss", help="Comma-separated dataset names")
    parser.add_argument("--output", type=str, default="results/zero_shot", help="Output directory")
    parser.add_argument("--task", type=str, default="forecasting", help="Task type")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    results = run_all_zero_shot(
        models=models,
        datasets=datasets,
        processed_dir="data/processed",
        results_dir=args.output,
        task=args.task,
        device=args.device,
        use_wandb=not args.no_wandb
    )

    print("\nSummary:")
    print(results)
