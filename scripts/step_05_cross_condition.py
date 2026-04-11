"""
Step 5: Cross-condition transfer experiments.

Train on C-MAPSS FD001, test on FD002 / FD003 / FD004.
Also evaluates FD001 → FD001 in-domain baseline.

Idempotent — skips pairs whose JSON result already exists.
Run:  python scripts/step_05_cross_condition.py
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pipeline_config import (
    PROC_DIR, RESULTS_DIR,
    MODELS_ZERO_SHOT, CMAPSS_SUBSETS, CMAPSS_HORIZON,
    DEVICE, SEED,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.models import get_model

log = setup_logging()

RES_CC = RESULTS_DIR / "cross_condition"
SOURCE_SUBSET = "FD001"


def load_cmapss(subset):
    path = PROC_DIR / "cmapss" / subset / "processed_data.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    to_np = lambda t: t.numpy() if isinstance(t, torch.Tensor) else t
    return to_np(d["train_X"]), to_np(d["train_y"]), to_np(d["test_X"]), to_np(d["test_y"])


def run_single(model_name, X_train, y_train, X_test, y_test, horizon):
    model = get_model(model_name, device=DEVICE)
    if model_name == "patchtst":
        model.fit(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    else:
        model.load_model()

    results = model.predict(torch.FloatTensor(X_test), horizon=horizon)
    preds = results["predictions"]

    y = y_test
    if y.ndim == 1:
        preds_flat = preds.reshape(preds.shape[0], -1).mean(axis=1)
        y_flat = y
    else:
        min_h = min(preds.shape[1], y.shape[1])
        preds_flat = preds[:, :min_h].flatten()
        y_flat = y[:, :min_h].flatten()

    metrics = {
        "mae":  float(np.mean(np.abs(y_flat - preds_flat))),
        "rmse": float(np.sqrt(np.mean((y_flat - preds_flat) ** 2))),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    ensure_dirs()
    set_seeds()
    RES_CC.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 5: Cross-Condition Transfer (C-MAPSS)")
    log.info("=" * 60)

    try:
        X_src_tr, y_src_tr, X_src_te, y_src_te = load_cmapss(SOURCE_SUBSET)
    except FileNotFoundError:
        log.error(f"Source subset {SOURCE_SUBSET} not preprocessed — run step 2 first.")
        return

    target_subsets = [s for s in CMAPSS_SUBSETS if s != SOURCE_SUBSET]
    all_targets = [SOURCE_SUBSET] + target_subsets  # include in-domain baseline

    total = 0
    skipped = 0

    for target in all_targets:
        try:
            _, _, X_tgt_te, y_tgt_te = load_cmapss(target)
        except FileNotFoundError:
            log.warning(f"  {target} not preprocessed — skipping.")
            continue

        for model_name in MODELS_ZERO_SHOT:
            total += 1
            key = f"{model_name}_FD001_to_{target}"
            json_path = RES_CC / f"{key}.json"

            if json_path.exists():
                log.info(f"  [cached] {key}")
                skipped += 1
                continue

            label = "In-domain" if target == SOURCE_SUBSET else "Cross-condition"
            log.info(f"  {label}: {model_name}  FD001 → {target}")
            try:
                metrics = run_single(
                    model_name, X_src_tr, y_src_tr, X_tgt_te, y_tgt_te,
                    CMAPSS_HORIZON,
                )
                record = {
                    "model": model_name,
                    "source": SOURCE_SUBSET, "target": target,
                    "scenario": "cross_condition", "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(json_path, "w") as f:
                    json.dump(record, f, indent=2)
                log.info(f"    Result: {metrics}")
            except Exception as e:
                log.error(f"    ERROR: {e}")
                traceback.print_exc()

    mark_step_done("step_05_cross_condition", {"total": total, "cached": skipped})
    log.info(f"Step 5 complete — {total} runs ({skipped} cached).")


if __name__ == "__main__":
    main()
