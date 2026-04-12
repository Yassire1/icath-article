"""
Step 4: Few-shot adaptation experiments.

Fits lightweight few-shot adapters for the configured models on 1 % of
training data, then evaluates on the test set.

Idempotent — skips pairs whose JSON result already exists.
Run:  python scripts/step_04_few_shot.py
"""

import gc
import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pipeline_config import (
    PROC_DIR, RESULTS_DIR,
    MODELS_FEW_SHOT, CMAPSS_HORIZON, HORIZON, PHM_MILLING_HORIZON,
    LORA_R, LORA_ALPHA, LORA_EPOCHS, LORA_LR, TRAIN_RATIO,
    DEVICE, SEED, EVAL_BATCH_SIZE, MAX_EVAL_SAMPLES, MAX_FEW_SHOT_SAMPLES,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.models import get_model

log = setup_logging()

RES_FS = RESULTS_DIR / "few_shot"
FEW_SHOT_BATCH_SIZES = {
    "moment": 4,
    "lag_llama": 8,
    "patchtst": 16,
}


def clear_runtime_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def batched_few_shot_predict(model_name, model, X_test, horizon):
    if len(X_test) == 0:
        raise ValueError("X_test is empty; nothing to evaluate.")

    batch_size = min(FEW_SHOT_BATCH_SIZES.get(model_name, EVAL_BATCH_SIZE), len(X_test))
    pred_chunks = []
    start = 0

    while start < len(X_test):
        stop = min(start + batch_size, len(X_test))
        batch = torch.FloatTensor(X_test[start:stop])
        try:
            results = model.predict(batch, horizon=horizon)
            pred_chunks.append(results["predictions"])
            start = stop
        except Exception as exc:
            if not is_cuda_oom(exc) or batch_size == 1:
                raise
            clear_runtime_memory()
            batch_size = max(1, batch_size // 2)
            log.warning(f"    CUDA OOM at batch {stop - start}; retrying with batch_size={batch_size}")
        finally:
            del batch
            clear_runtime_memory()

    return np.concatenate(pred_chunks, axis=0)


def load_dataset(name, subset=None):
    if name == "cmapss":
        path = PROC_DIR / "cmapss" / (subset or "FD001")
    else:
        path = PROC_DIR / name
    d = torch.load(path / "processed_data.pt", map_location="cpu", weights_only=False)
    to_np = lambda t: t.numpy() if isinstance(t, torch.Tensor) else t
    return to_np(d["train_X"]), to_np(d["train_y"]), to_np(d["test_X"]), to_np(d["test_y"])


def run_few_shot(model_name, X_train, y_train, X_test, y_test, horizon):
    if len(X_test) > MAX_EVAL_SAMPLES:
        X_test = X_test[:MAX_EVAL_SAMPLES]
        y_test = y_test[:MAX_EVAL_SAMPLES]

    n_few = max(int(len(X_train) * TRAIN_RATIO), 32)
    n_few = min(n_few, MAX_FEW_SHOT_SAMPLES)
    X_few = torch.FloatTensor(X_train[:n_few])
    y_few = torch.FloatTensor(y_train[:n_few])

    model = None
    try:
        clear_runtime_memory()
        model = get_model(model_name, device=DEVICE)
        if not getattr(model, "supports_few_shot", False):
            raise NotImplementedError(
                f"{model_name} few-shot adaptation is not implemented in this repository."
            )
        model.load_model()
        log.info(f"    Few-shot adapting on {n_few} samples ...")
        model.few_shot_adapt(
            X_few, y_few,
            epochs=LORA_EPOCHS, lr=LORA_LR,
            lora_r=LORA_R, lora_alpha=LORA_ALPHA,
            forecast_horizon=horizon,
        )

        preds = batched_few_shot_predict(model_name, model, X_test, horizon)

        y = y_test
        if y.ndim == 1:
            preds_flat = preds.reshape(preds.shape[0], -1).mean(axis=1)
            y_flat = y
        else:
            min_h = min(preds.shape[1], y.shape[1] if y.ndim > 1 else horizon)
            preds_flat = preds[:, :min_h].flatten()
            y_flat = y[:, :min_h].flatten()

        metrics = {
            "mae":  float(np.mean(np.abs(y_flat - preds_flat))),
            "rmse": float(np.sqrt(np.mean((y_flat - preds_flat) ** 2))),
            "mape": float(np.mean(np.abs((y_flat - preds_flat) / (np.abs(y_flat) + 1e-8))) * 100),
            "train_samples": n_few,
        }

        return metrics
    finally:
        if model is not None:
            del model
        clear_runtime_memory()


def main():
    ensure_dirs()
    set_seeds()
    RES_FS.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 4: Few-Shot Adaptation Experiments")
    log.info("=" * 60)

    dataset_cfgs = [
        ("cmapss",     "FD001", CMAPSS_HORIZON),
        ("wind_scada", None,    HORIZON),
        ("phm_milling", None,   PHM_MILLING_HORIZON),
    ]

    total = 0
    skipped = 0
    succeeded = 0
    failed = 0
    missing_datasets = []

    for ds_name, subset, h in dataset_cfgs:
        try:
            X_tr, y_tr, X_te, y_te = load_dataset(ds_name, subset)
        except FileNotFoundError as e:
            log.warning(f"  {e} — skipping dataset.")
            missing_datasets.append(f"{ds_name}/{subset}" if subset else ds_name)
            continue

        ds_label = f"{ds_name}/{subset}" if subset else ds_name

        for model_name in MODELS_FEW_SHOT:
            total += 1
            key = f"{model_name}_{ds_label.replace('/', '_')}_few_shot"
            json_path = RES_FS / f"{key}.json"

            if json_path.exists():
                log.info(f"  [cached] {key}")
                skipped += 1
                continue

            log.info(f"  Few-shot: {model_name} on {ds_label}")
            try:
                metrics = run_few_shot(model_name, X_tr, y_tr, X_te, y_te, h)
                record = {
                    "model": model_name, "dataset": ds_label,
                    "scenario": "few_shot", "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(json_path, "w") as f:
                    json.dump(record, f, indent=2)
                log.info(f"    Result: {metrics}")
                succeeded += 1
            except Exception as e:
                log.error(f"    ERROR: {e}")
                traceback.print_exc()
                error_record = {
                    "model": model_name,
                    "dataset": ds_label,
                    "scenario": "few_shot",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(json_path, "w") as f:
                    json.dump(error_record, f, indent=2)
                failed += 1

    mark_step_done("step_04_few_shot", {
        "total": total,
        "cached": skipped,
        "completed": succeeded,
        "failed": failed,
    })
    if total == 0:
        log.warning("Step 4 completed with 0 runs; no preprocessed datasets were available.")
    else:
        log.info(f"Step 4 complete — {total} runs ({skipped} cached).")

    if missing_datasets:
        raise RuntimeError(
            "Few-shot step is incomplete because required datasets are missing: "
            + ", ".join(missing_datasets)
        )


if __name__ == "__main__":
    main()
