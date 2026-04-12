"""
Step 3: Zero-shot experiments.

Evaluates all (model × dataset) combinations using pretrained weights only.
PatchTST is supervised — it trains on the training split first.

Idempotent — skips model-dataset pairs whose JSON result already exists.
Run:  python scripts/step_03_zero_shot.py
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pipeline_config import (
    PROC_DIR, RESULTS_DIR,
    MODELS_ZERO_SHOT, CMAPSS_HORIZON, HORIZON, PHM_MILLING_HORIZON,
    DEVICE, SEED, EVAL_BATCH_SIZE, MAX_EVAL_SAMPLES,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.models import get_model
from src.data.preprocessing import SCADAPreprocessor

log = setup_logging()

RES_ZS = RESULTS_DIR / "zero_shot"


def load_dataset(name: str, subset: str = None):
    """Load preprocessed tensors as numpy arrays."""
    if name == "cmapss":
        path = PROC_DIR / "cmapss" / (subset or "FD001")
    else:
        path = PROC_DIR / name
    pt_file = path / "processed_data.pt"
    if not pt_file.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {pt_file}")
    d = torch.load(pt_file, map_location="cpu", weights_only=False)
    to_np = lambda t: t.numpy() if isinstance(t, torch.Tensor) else t
    return to_np(d["train_X"]), to_np(d["train_y"]), to_np(d["test_X"]), to_np(d["test_y"])


def run_zero_shot(model_name, dataset_name, X_train, y_train, X_test, y_test, horizon):
    """Run a single zero-shot experiment."""
    if len(X_test) > MAX_EVAL_SAMPLES:
        X_test = X_test[:MAX_EVAL_SAMPLES]
        y_test = y_test[:MAX_EVAL_SAMPLES]

    model = get_model(model_name, device=DEVICE)

    if model_name == "patchtst":
        log.info(f"  PatchTST: fitting on training split ...")
        model.fit(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    else:
        model.load_model()

    pred_chunks = []
    for start in range(0, len(X_test), EVAL_BATCH_SIZE):
        batch = torch.FloatTensor(X_test[start:start + EVAL_BATCH_SIZE])
        results = model.zero_shot(batch, horizon=horizon)
        pred_chunks.append(results["predictions"])
    preds = np.concatenate(pred_chunks, axis=0)

    y = y_test
    if y.ndim == 1:
        preds_flat = preds.reshape(preds.shape[0], -1).mean(axis=1)
        y_flat = y
    else:
        min_h = min(preds.shape[1], y.shape[1] if y.ndim > 1 else horizon)
        preds_flat = preds[:, :min_h].flatten()
        y_flat = y[:, :min_h].flatten() if y.ndim > 1 else y.flatten()

    metrics = {
        "mae":  float(np.mean(np.abs(y_flat - preds_flat))),
        "rmse": float(np.sqrt(np.mean((y_flat - preds_flat) ** 2))),
        "mape": float(np.mean(np.abs((y_flat - preds_flat) / (np.abs(y_flat) + 1e-8))) * 100),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    ensure_dirs()
    set_seeds()
    RES_ZS.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 3: Zero-Shot Experiments")
    log.info("=" * 60)

    dataset_cfgs = [
        ("cmapss",     "FD001", CMAPSS_HORIZON),
        ("wind_scada", None,    HORIZON),
        ("phm_milling", None,   PHM_MILLING_HORIZON),
    ]

    missing_datasets = []
    zero_shot_results = {}
    total = 0
    skipped = 0

    for ds_name, subset, h in dataset_cfgs:
        try:
            X_tr, y_tr, X_te, y_te = load_dataset(ds_name, subset)
        except FileNotFoundError as e:
            log.warning(f"  {e} — skipping dataset.")
            missing_datasets.append(f"{ds_name}/{subset}" if subset else ds_name)
            continue

        ds_label = f"{ds_name}/{subset}" if subset else ds_name

        for model_name in MODELS_ZERO_SHOT:
            total += 1
            key = f"{model_name}_{ds_label.replace('/', '_')}"
            json_path = RES_ZS / f"{key}.json"

            if json_path.exists():
                log.info(f"  [cached] {key}")
                with open(json_path) as f:
                    zero_shot_results[key] = json.load(f)
                skipped += 1
                continue

            log.info(f"  Zero-shot: {model_name} on {ds_label}")
            try:
                metrics = run_zero_shot(model_name, ds_name, X_tr, y_tr, X_te, y_te, h)
                record = {
                    "model": model_name, "dataset": ds_label,
                    "scenario": "zero_shot", "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                zero_shot_results[key] = record
                with open(json_path, "w") as f:
                    json.dump(record, f, indent=2)
                log.info(f"    Result: {metrics}")
            except Exception as e:
                log.error(f"    ERROR: {e}")
                traceback.print_exc()
                zero_shot_results[key] = {
                    "model": model_name, "dataset": ds_label, "error": str(e),
                }

    mark_step_done("step_03_zero_shot", {
        "total_runs": total,
        "skipped_cached": skipped,
        "completed": sum(1 for v in zero_shot_results.values() if "error" not in v),
        "failed": sum(1 for v in zero_shot_results.values() if "error" in v),
    })
    if total == 0:
        log.warning("Step 3 completed with 0 runs; no preprocessed datasets were available.")
    else:
        log.info(f"Step 3 complete — {total} runs ({skipped} cached).")

    if missing_datasets:
        raise RuntimeError(
            "Zero-shot step is incomplete because required datasets are missing: "
            + ", ".join(missing_datasets)
        )


if __name__ == "__main__":
    main()
