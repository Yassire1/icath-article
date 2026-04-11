"""
Main pipeline orchestrator — runs all steps in sequence.

Usage:
    python scripts/run_pipeline.py                  # run everything
    python scripts/run_pipeline.py --from 3          # resume from step 3
    python scripts/run_pipeline.py --only 2 7 8      # run only steps 2, 7, 8
    python scripts/run_pipeline.py --skip 6          # skip step 6

Each step is idempotent: cached results are reused automatically.
To force re-running a step, delete the relevant files in results/ or
data/processed/ before launching.
"""

import argparse
import sys
import time
from datetime import datetime

from pipeline_config import (
    setup_logging, ensure_dirs, set_seeds,
    load_manifest, save_manifest,
)

log = setup_logging()

# Step registry: (step_number, module_name, description)
STEPS = [
    (1, "step_01_download",        "Download datasets"),
    (2, "step_02_preprocess",      "Preprocess datasets"),
    (3, "step_03_zero_shot",       "Zero-shot experiments"),
    (4, "step_04_few_shot",        "Few-shot LoRA experiments"),
    (5, "step_05_cross_condition", "Cross-condition transfer"),
    (6, "step_06_profile",         "Inference profiling"),
    (7, "step_07_aggregate",       "Aggregate results → CSV"),
    (8, "step_08_visualize",       "Generate figures"),
]


def run_step(step_num: int, module_name: str, desc: str):
    """Import and execute a step module's main() function."""
    import importlib
    log.info("")
    log.info(f"{'#' * 70}")
    log.info(f"#  STEP {step_num}: {desc}")
    log.info(f"{'#' * 70}")

    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        log.info(f"  Step {step_num} finished in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        log.error(f"  Step {step_num} FAILED after {elapsed:.1f}s: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="TSFM-PDM Benchmark Pipeline Runner"
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Resume from this step number (default: 1)",
    )
    parser.add_argument(
        "--only", type=int, nargs="+", default=None,
        help="Run only these step numbers",
    )
    parser.add_argument(
        "--skip", type=int, nargs="+", default=None,
        help="Skip these step numbers",
    )
    args = parser.parse_args()

    ensure_dirs()
    set_seeds()

    # Determine which steps to run
    skip_set = set(args.skip or [])
    if args.only:
        steps_to_run = [(n, m, d) for n, m, d in STEPS if n in args.only]
    else:
        steps_to_run = [(n, m, d) for n, m, d in STEPS if n >= args.from_step]
    steps_to_run = [(n, m, d) for n, m, d in steps_to_run if n not in skip_set]

    log.info("=" * 70)
    log.info("  TSFM-PDM BENCHMARK PIPELINE")
    log.info(f"  Started: {datetime.now().isoformat()}")
    log.info(f"  Steps:   {[n for n, _, _ in steps_to_run]}")
    log.info("=" * 70)

    manifest = load_manifest()
    manifest["last_run"] = datetime.now().isoformat()
    save_manifest(manifest)

    t_total = time.time()
    failed = []

    for step_num, module_name, desc in steps_to_run:
        try:
            run_step(step_num, module_name, desc)
        except Exception:
            failed.append(step_num)
            log.error(f"  Stopping pipeline at step {step_num}.")
            break

    total_elapsed = time.time() - t_total

    log.info("")
    log.info("=" * 70)
    log.info("  PIPELINE SUMMARY")
    log.info("=" * 70)
    log.info(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if failed:
        log.error(f"  FAILED at step(s): {failed}")
    else:
        log.info("  All steps completed successfully.")
    log.info("")
    log.info("  Output locations:")
    log.info("    data/processed/          ← preprocessed .pt tensors")
    log.info("    results/zero_shot/       ← per-run JSON files")
    log.info("    results/few_shot/        ← per-run JSON files")
    log.info("    results/cross_condition/ ← per-run JSON files")
    log.info("    results/tables/          ← aggregated CSV summaries")
    log.info("    results/figures/         ← PDF + PNG figures")
    log.info("    results/manifest.json    ← step completion tracking")
    log.info("    results/pipeline.log     ← full execution log")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
