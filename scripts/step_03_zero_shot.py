"""
Step 3: Zero-shot experiments.

Evaluates all (model × dataset) combinations using pretrained weights only.
PatchTST is supervised — it trains on the training split first.

Idempotent — skips model-dataset pairs whose JSON result already exists.
Run:  python scripts/step_03_zero_shot.py
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
    MODELS_ZERO_SHOT, CMAPSS_HORIZON, HORIZON, PHM_MILLING_HORIZON,
    DEVICE, SEED, EVAL_BATCH_SIZE, MAX_EVAL_SAMPLES,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.models import get_model
from src.evaluation.metrics import FORECAST_METRIC_VERSION, compute_all_metrics

log = setup_logging()

RES_ZS = RESULTS_DIR / "zero_shot"
ZERO_SHOT_RESULT_VERSION = "chronos_chunked_rollout_v1"
MODEL_EVAL_BATCH_SIZES = {
    "moment": 4,
    "chronos": 16,
    "lag_llama": 16,
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


def batched_zero_shot_predict(model, model_name, X_test, horizon):
    if len(X_test) == 0:
        return np.empty((0, horizon, 0), dtype=np.float32)

    batch_size = min(MODEL_EVAL_BATCH_SIZES.get(model_name, EVAL_BATCH_SIZE), len(X_test))
    pred_chunks = []
    start = 0

    while start < len(X_test):
        stop = min(start + batch_size, len(X_test))
        batch = torch.FloatTensor(X_test[start:stop])
        try:
            results = model.zero_shot(batch, horizon=horizon)
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

    model = None
    try:
        clear_runtime_memory()
        model = get_model(model_name, device=DEVICE)

        if model_name == "patchtst":
            log.info(f"  PatchTST: fitting on training split ...")
            model.fit(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        else:
            model.load_model()

        preds = batched_zero_shot_predict(model, model_name, X_test, horizon)

        y = y_test
        if y.ndim == 1:
            preds_eval = preds.reshape(preds.shape[0], -1).mean(axis=1)
            y_eval = y
        else:
            min_h = min(preds.shape[1], y.shape[1] if y.ndim > 1 else horizon)
            preds_eval = preds[:, :min_h]
            y_eval = y[:, :min_h] if y.ndim > 1 else y

        metrics = compute_all_metrics(y_eval, preds_eval, task="forecasting")

        return metrics
    finally:
        if model is not None:
            del model
        clear_runtime_memory()


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
                with open(json_path) as f:
                    cached_record = json.load(f)
                if (
                    cached_record.get("metric_version") == FORECAST_METRIC_VERSION
                    and cached_record.get("result_version") == ZERO_SHOT_RESULT_VERSION
                ):
                    log.info(f"  [cached] {key}")
                    zero_shot_results[key] = cached_record
                    skipped += 1
                    continue
                log.info(f"  [stale cache] recomputing {key}")

            log.info(f"  Zero-shot: {model_name} on {ds_label}")
            try:
                metrics = run_zero_shot(model_name, ds_name, X_tr, y_tr, X_te, y_te, h)
                record = {
                    "model": model_name, "dataset": ds_label,
                    "scenario": "zero_shot", "metrics": metrics,
                    "metric_version": FORECAST_METRIC_VERSION,
                    "result_version": ZERO_SHOT_RESULT_VERSION,
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
                    "model": model_name,
                    "dataset": ds_label,
                    "scenario": "zero_shot",
                    "metric_version": FORECAST_METRIC_VERSION,
                    "result_version": ZERO_SHOT_RESULT_VERSION,
                    "error": str(e),
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
