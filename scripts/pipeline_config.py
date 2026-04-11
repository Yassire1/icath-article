"""
Shared configuration for the VM pipeline.
All step scripts import from here — edit once, applies everywhere.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# ── Project root (one level up from scripts/) ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ── Paths ─────────────────────────────────────────────────────────────────
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROC_DIR      = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"
TABLES_DIR    = RESULTS_DIR  / "tables"
FIGURES_DIR   = RESULTS_DIR  / "figures"
MANIFEST_PATH = RESULTS_DIR  / "manifest.json"
LOG_PATH      = RESULTS_DIR  / "pipeline.log"

# ── Experiments ───────────────────────────────────────────────────────────
RUN_ZERO_SHOT       = True
RUN_FEW_SHOT        = True
RUN_CROSS_CONDITION = True

# Models
MODELS_ZERO_SHOT = ["moment", "chronos", "lag_llama", "patchtst"]
MODELS_FEW_SHOT  = ["moment", "lag_llama"]   # only LoRA-capable models

# Datasets
DATASETS = ["cmapss", "wind_scada", "mimii"]

# C-MAPSS
CMAPSS_SUBSETS   = ["FD001", "FD002", "FD003", "FD004"]
CMAPSS_LOOKBACK  = 64
CMAPSS_HORIZON   = 30

# Wind SCADA + MIMII
LOOKBACK = 512
HORIZON  = 96

# Few-shot LoRA
LORA_R      = 16
LORA_ALPHA  = 32
LORA_EPOCHS = 10
LORA_LR     = 1e-4
TRAIN_RATIO = 0.01  # 1 %

# MIMII audio
MIMII_MACHINES          = ["fan"]
MIMII_ZENODO_RECORD_ID  = "3384388"
MIMII_PREFERRED_VARIANT = "6_dB"
MIMII_MAX_FILES         = 500   # increase for full run
MIMII_N_MFCC            = 40
MIMII_SR                = 16_000
MIMII_N_FFT             = 1024
MIMII_HOP               = 512

# Hardware — VM has no GPU
DEVICE = "cpu"
SEED   = 42

# Profiling
WARMUP_RUNS = 2
TIMING_RUNS = 5


# ── Helpers ───────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    """Configure pipeline logger (console + file)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def set_seeds():
    """Set reproducibility seeds."""
    import numpy as np
    import torch
    import random
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def ensure_dirs():
    """Create all output directories."""
    for d in [
        RAW_DIR / "cmapss", RAW_DIR / "wind_scada", RAW_DIR / "mimii",
        PROC_DIR / "cmapss", PROC_DIR / "wind_scada", PROC_DIR / "mimii",
        RESULTS_DIR / "zero_shot",
        RESULTS_DIR / "few_shot",
        RESULTS_DIR / "cross_condition",
        TABLES_DIR,
        FIGURES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def load_manifest() -> dict:
    """Load the pipeline manifest (tracks completed steps)."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"steps": {}, "created": datetime.now().isoformat()}


def save_manifest(manifest: dict):
    """Persist the manifest."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def mark_step_done(step_name: str, details: dict = None):
    """Record that a step completed successfully."""
    manifest = load_manifest()
    safe_details = {
        k: v for k, v in (details or {}).items()
        if k not in ("status", "timestamp")
    }
    manifest["steps"][step_name] = {
        **safe_details,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    }
    save_manifest(manifest)


def is_step_done(step_name: str) -> bool:
    """Check whether a step was already completed."""
    manifest = load_manifest()
    return manifest.get("steps", {}).get(step_name, {}).get("status") == "completed"


# ── Kaggle credentials ────────────────────────────────────────────────────
def _parse_env_file() -> dict:
    """Parse KEY = VALUE lines from the project-root .env file."""
    env_path = PROJECT_ROOT / ".env"
    pairs: dict = {}
    if not env_path.exists():
        return pairs
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        pairs[key.strip()] = val.strip().strip('"').strip("'")
    return pairs


def setup_kaggle_credentials() -> None:
    """
    Read kaggle_username and kaggle_api from the project-root .env file and
    wire them up so the Kaggle CLI works without any manual configuration.

    - Sets KAGGLE_USERNAME and KAGGLE_KEY env vars for the current process
      and any subprocess (e.g. the `kaggle` CLI invoked via subprocess).
    - Writes ~/.kaggle/kaggle.json the first time so older CLI versions that
      do not read env vars can also authenticate.
    - Never overwrites credentials that are already set in the environment.
    """
    env = _parse_env_file()
    username = env.get("kaggle_username")
    key = env.get("kaggle_api")

    if not username or not key:
        return  # no credentials available — download step will surface the error

    os.environ.setdefault("KAGGLE_USERNAME", username)
    os.environ.setdefault("KAGGLE_KEY", key)

    # Create ~/.kaggle/kaggle.json only if it does not already exist.
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        kaggle_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        kaggle_json.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
        try:
            kaggle_json.chmod(0o600)  # required by the Kaggle CLI on Linux/macOS
        except Exception:
            pass  # silently skip on Windows — permissions work differently


# Auto-run at import time so every script that imports pipeline_config gets
# the credentials injected before it does anything with Kaggle.
setup_kaggle_credentials()
