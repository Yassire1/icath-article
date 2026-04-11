"""
Step 4: Few-shot LoRA experiments.

Adapts LoRA-capable models (MOMENT, Lag-Llama) on 1 % of training data,
then evaluates on the test set.

Idempotent — skips pairs whose JSON result already exists.
Run:  python scripts/step_04_few_shot.py
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pipeline_config import (
    PROC_DIR, RESULTS_DIR,
    MODELS_FEW_SHOT, CMAPSS_HORIZON, HORIZON,
    LORA_R, LORA_ALPHA, LORA_EPOCHS, LORA_LR, TRAIN_RATIO,
    DEVICE, SEED,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.models import get_model

log = setup_logging()

RES_FS = RESULTS_DIR / "few_shot"


def load_dataset(name, subset=None):
    if name == "cmapss":
        path = PROC_DIR / "cmapss" / (subset or "FD001")
    else:
        path = PROC_DIR / name
    d = torch.load(path / "processed_data.pt", map_location="cpu", weights_only=False)
    to_np = lambda t: t.numpy() if isinstance(t, torch.Tensor) else t
    return to_np(d["train_X"]), to_np(d["train_y"]), to_np(d["test_X"]), to_np(d["test_y"])


def run_few_shot(model_name, X_train, y_train, X_test, y_test, horizon):
    n_few = max(int(len(X_train) * TRAIN_RATIO), 32)
    X_few = torch.FloatTensor(X_train[:n_few])
    y_few = torch.FloatTensor(y_train[:n_few])

    model = get_model(model_name, device=DEVICE)
    model.load_model()
    log.info(f"    Few-shot adapting on {n_few} samples ...")
    model.few_shot_adapt(
        X_few, y_few,
        epochs=LORA_EPOCHS, lr=LORA_LR,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA,
    )

    results = model.predict(torch.FloatTensor(X_test), horizon=horizon)
    preds = results["predictions"]

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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    ensure_dirs()
    set_seeds()
    RES_FS.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 4: Few-Shot LoRA Experiments")
    log.info("=" * 60)

    dataset_cfgs = [
        ("cmapss",     "FD001", CMAPSS_HORIZON),
        ("wind_scada", None,    HORIZON),
        ("mimii",      None,    HORIZON),
    ]

    total = 0
    skipped = 0

    for ds_name, subset, h in dataset_cfgs:
        try:
            X_tr, y_tr, X_te, y_te = load_dataset(ds_name, subset)
        except FileNotFoundError as e:
            log.warning(f"  {e} — skipping dataset.")
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
            except Exception as e:
                log.error(f"    ERROR: {e}")
                traceback.print_exc()

    mark_step_done("step_04_few_shot", {"total": total, "cached": skipped})
    log.info(f"Step 4 complete — {total} runs ({skipped} cached).")


if __name__ == "__main__":
    main()
