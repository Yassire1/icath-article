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

DEFAULT_CACHE_ROOT = Path("/mnt/datasets") / "icath-cache"
CACHE_ROOT = DEFAULT_CACHE_ROOT if DEFAULT_CACHE_ROOT.exists() else PROJECT_ROOT / ".cache"
HF_HOME = CACHE_ROOT / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
TRANSFORMERS_CACHE = HF_HOME / "transformers"
TORCH_HOME = CACHE_ROOT / "torch"
XDG_CACHE_HOME = CACHE_ROOT / "xdg"

# ── Experiments ───────────────────────────────────────────────────────────
RUN_ZERO_SHOT       = True
RUN_FEW_SHOT        = True
RUN_CROSS_CONDITION = True

# Models
MODELS_ZERO_SHOT = ["moment", "chronos", "lag_llama", "patchtst"]
MODELS_FEW_SHOT  = ["moment", "lag_llama"]   # only LoRA-capable models
MODEL_IDS = {
    "moment": "AutonLab/MOMENT-1-small",
    "chronos": "amazon/chronos-t5-tiny",
    "lag_llama": "time-series-foundation-models/Lag-Llama",
}

# Runtime tuning for CPU-only end-to-end execution
EVAL_BATCH_SIZE = 128
MAX_EVAL_SAMPLES = 32
MAX_FEW_SHOT_SAMPLES = 32
PATCHTST_MAX_TRAIN_WINDOWS = 64
PATCHTST_MAX_STEPS = 20
PATCHTST_BATCH_SIZE = 32
PATCHTST_WINDOWS_BATCH_SIZE = 128

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
LORA_EPOCHS = 2
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
PROFILE_BATCH_SIZE = 8
WARMUP_RUNS = 1
TIMING_RUNS = 1


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
        CACHE_ROOT,
        HF_HOME,
        HF_HUB_CACHE,
        TRANSFORMERS_CACHE,
        TORCH_HOME,
        XDG_CACHE_HOME,
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
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

setup_kaggle_credentials()
