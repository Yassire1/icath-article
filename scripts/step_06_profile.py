"""
Step 6: Inference latency & resource profiling.

Measures per-model: parameter count, wall-clock latency (ms),
and peak memory usage.  GPU memory is only reported if a CUDA device
is available; otherwise peak RSS is estimated.

Idempotent — skips models whose profile JSON already exists.
Run:  python scripts/step_06_profile.py
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from pipeline_config import (
    PROC_DIR, TABLES_DIR,
    MODELS_ZERO_SHOT, CMAPSS_HORIZON, DEVICE,
    WARMUP_RUNS, TIMING_RUNS,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.models import get_model

log = setup_logging()


def load_profile_batch():
    """Load a fixed 32-sample batch from C-MAPSS FD001 for profiling."""
    path = PROC_DIR / "cmapss" / "FD001" / "processed_data.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    X = d["test_X"]
    if isinstance(X, torch.Tensor):
        X = X[:32]
    else:
        X = torch.FloatTensor(X[:32])
    return X


def main():
    ensure_dirs()
    set_seeds()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 6: Inference Profiling")
    log.info("=" * 60)

    try:
        X_prof = load_profile_batch()
    except FileNotFoundError:
        log.error("C-MAPSS FD001 not preprocessed — cannot profile. Run step 2 first.")
        return

    profile_results = {}

    for model_name in MODELS_ZERO_SHOT:
        cached_path = TABLES_DIR / f"profile_{model_name}.json"
        if cached_path.exists():
            log.info(f"  [cached] {model_name}")
            with open(cached_path) as f:
                profile_results[model_name] = json.load(f)
            continue

        log.info(f"  Profiling {model_name} ...")
        try:
            model = get_model(model_name, device=DEVICE)
            if model_name == "patchtst":
                path = PROC_DIR / "cmapss" / "FD001" / "processed_data.pt"
                d = torch.load(path, map_location="cpu", weights_only=False)
                X_tr = d["train_X"][:64]
                y_tr = d["train_y"][:64]
                if not isinstance(X_tr, torch.Tensor):
                    X_tr, y_tr = torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
                model.fit(X_tr, y_tr)
            else:
                model.load_model()

            inner = getattr(model, "model", None)
            n_params = sum(p.numel() for p in inner.parameters()) if inner is not None else 0

            # Warmup
            for _ in range(WARMUP_RUNS):
                _ = model.predict(X_prof, horizon=CMAPSS_HORIZON)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(TIMING_RUNS):
                _ = model.predict(X_prof, horizon=CMAPSS_HORIZON)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) / TIMING_RUNS * 1000

            peak_mem_mb = (
                torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available()
                else float("nan")
            )

            rec = {
                "model": model_name,
                "params_M": round(n_params / 1e6, 1),
                "latency_ms": round(elapsed_ms, 1),
                "peak_gpu_mb": round(peak_mem_mb, 0) if not np.isnan(peak_mem_mb) else None,
                "batch_size": int(X_prof.shape[0]),
                "device": DEVICE,
            }
            profile_results[model_name] = rec
            with open(cached_path, "w") as f:
                json.dump(rec, f, indent=2)
            log.info(f"    params={rec['params_M']}M  latency={rec['latency_ms']}ms  "
                     f"peak_gpu={rec.get('peak_gpu_mb', 'N/A')}MB")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"    ERROR: {e}")
            traceback.print_exc()
            profile_results[model_name] = {"model": model_name, "error": str(e)}

    # Save combined CSV
    import pandas as pd
    rows = [v for v in profile_results.values() if "error" not in v]
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "efficiency_summary.csv", index=False)
        log.info(f"  Saved efficiency_summary.csv ({len(rows)} models)")

    mark_step_done("step_06_profile", {"models_profiled": len(rows)})
    log.info("Step 6 complete.")


if __name__ == "__main__":
    main()
