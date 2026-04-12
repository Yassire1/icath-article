"""
Step 2: Preprocess all datasets (C-MAPSS, Wind SCADA, PHM Milling).

- C-MAPSS: all four subsets (FD001-FD004), RUL task
- Wind SCADA: generic CSV, forecasting task
- PHM Milling: stream raw cut CSVs, summarize each cut into compact features,
    then build cut-level forecasting windows without crossing cutter boundaries.

Idempotent — skips datasets whose processed_data.pt already exists.
Run:  python scripts/step_02_preprocess.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline_config import (
    RAW_DIR, PROC_DIR,
    CMAPSS_SUBSETS, CMAPSS_LOOKBACK, CMAPSS_HORIZON,
    LOOKBACK, HORIZON, SEED,
    PHM_MILLING_LOOKBACK, PHM_MILLING_HORIZON, PHM_MILLING_CHUNK_ROWS,
    DATASETS,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.data.preprocessing import SCADAPreprocessor

warnings.filterwarnings("ignore")
log = setup_logging()

PHM_SENSOR_NAMES = [
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "ae_rms",
]
CUTTER_RE = re.compile(r"(c\d+)", re.IGNORECASE)
DIGIT_RE = re.compile(r"(\d+)")


def _cat_or_empty(arrays, tail_shape):
    if arrays:
        return torch.FloatTensor(np.concatenate(arrays, axis=0))
    return torch.empty((0, *tail_shape), dtype=torch.float32)


def _stack_or_empty(windows, tail_shape):
    if windows:
        return np.asarray(windows, dtype=np.float32)
    return np.empty((0, *tail_shape), dtype=np.float32)


# ── C-MAPSS ──────────────────────────────────────────────────────────────
def preprocess_cmapss() -> dict:
    """Process all C-MAPSS subsets. Returns {subset: data_dict}."""
    cmapss_raw = RAW_DIR / "cmapss"
    if not (cmapss_raw / "train_FD001.txt").exists():
        log.warning("C-MAPSS raw data not found — skipping.")
        return {}

    preprocessor = SCADAPreprocessor(
        lookback=CMAPSS_LOOKBACK, horizon=CMAPSS_HORIZON,
        train_ratio=0.70, val_ratio=0.15,
        normalization="standard", seed=SEED,
    )

    result = {}
    for subset in CMAPSS_SUBSETS:
        out_dir = PROC_DIR / "cmapss" / subset
        sentinel = out_dir / "processed_data.pt"
        if sentinel.exists():
            log.info(f"  C-MAPSS {subset}: cached")
            result[subset] = preprocessor.load_processed(out_dir)
        else:
            log.info(f"  C-MAPSS {subset}: preprocessing ...")
            data = preprocessor.process_cmapss(cmapss_raw, subset=subset)
            preprocessor.save_processed(data, out_dir)
            result[subset] = data
        d = result[subset]
        log.info(f"    train {tuple(d['train_X'].shape)}  "
                 f"val {tuple(d['val_X'].shape)}  test {tuple(d['test_X'].shape)}")
    return result


# ── Wind SCADA ───────────────────────────────────────────────────────────
def preprocess_wind_scada() -> dict:
    """Process Wind SCADA CSV. Returns data dict or {}."""
    scada_raw = RAW_DIR / "wind_scada"
    out_dir = PROC_DIR / "wind_scada"
    sentinel = out_dir / "processed_data.pt"

    preprocessor = SCADAPreprocessor(
        lookback=LOOKBACK, horizon=HORIZON,
        train_ratio=0.70, val_ratio=0.15,
        normalization="standard", seed=SEED,
    )

    if sentinel.exists():
        log.info("  Wind SCADA: cached")
        return preprocessor.load_processed(out_dir)

    csv_files = sorted(scada_raw.glob("*.csv"))
    if not csv_files:
        log.warning("  Wind SCADA: no CSV found — skipping.")
        return {}

    csv_path = csv_files[0]
    log.info(f"  Wind SCADA: preprocessing {csv_path.name} ...")

    df_raw = pd.read_csv(csv_path, nrows=5)
    ts_col = next(
        (c for c in df_raw.columns
         if any(k in c.lower() for k in ["time", "date", "timestamp"])),
        None,
    )

    data = preprocessor.process_generic_csv(csv_path, timestamp_col=ts_col, task="forecasting")
    preprocessor.save_processed(data, out_dir)
    log.info(f"    train {tuple(data['train_X'].shape)}  "
             f"val {tuple(data['val_X'].shape)}  test {tuple(data['test_X'].shape)}")
    return data


# ── PHM Milling ──────────────────────────────────────────────────────────
def _infer_cutter_id(path: Path):
    for candidate in [path.stem, path.parent.name, *[parent.name for parent in path.parents[:3]]]:
        match = CUTTER_RE.search(candidate)
        if match:
            return match.group(1).lower()
    return None


def _cut_sort_key(path: Path):
    digits = DIGIT_RE.findall(path.stem)
    if digits:
        return (0, int(digits[-1]), path.name.lower())
    return (1, path.name.lower())


def _iter_numeric_chunks(csv_path: Path):
    read_attempts = [
        {"header": None, "chunksize": PHM_MILLING_CHUNK_ROWS},
        {"header": None, "sep": r"[,\s]+", "engine": "python", "chunksize": PHM_MILLING_CHUNK_ROWS},
    ]

    last_error = None
    for kwargs in read_attempts:
        try:
            reader = pd.read_csv(csv_path, **kwargs)
            first = next(reader)
        except StopIteration:
            return
        except Exception as exc:
            last_error = exc
            continue

        first = first.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if first.shape[1] <= 1 and "sep" not in kwargs:
            continue

        yield first
        for chunk in reader:
            numeric = chunk.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if not numeric.empty:
                yield numeric
        return

    if last_error is not None:
        raise last_error
    raise ValueError(f"Could not parse numeric cut file: {csv_path}")


def _summarize_cut_file(csv_path: Path):
    count = 0
    sum_vec = None
    sumsq_vec = None
    min_vec = None
    max_vec = None

    for chunk in _iter_numeric_chunks(csv_path):
        arr = chunk.to_numpy(dtype=np.float64, copy=False)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] < len(PHM_SENSOR_NAMES):
            continue

        arr = arr[:, :len(PHM_SENSOR_NAMES)]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
            continue

        if sum_vec is None:
            sum_vec = arr.sum(axis=0)
            sumsq_vec = np.square(arr).sum(axis=0)
            min_vec = arr.min(axis=0)
            max_vec = arr.max(axis=0)
        else:
            sum_vec += arr.sum(axis=0)
            sumsq_vec += np.square(arr).sum(axis=0)
            min_vec = np.minimum(min_vec, arr.min(axis=0))
            max_vec = np.maximum(max_vec, arr.max(axis=0))
        count += arr.shape[0]

    if count == 0 or sum_vec is None:
        return None

    mean_vec = sum_vec / count
    var_vec = np.maximum(sumsq_vec / count - np.square(mean_vec), 0.0)
    std_vec = np.sqrt(var_vec)
    rms_vec = np.sqrt(sumsq_vec / count)

    features = {}
    for idx, sensor_name in enumerate(PHM_SENSOR_NAMES):
        features[f"{sensor_name}_mean"] = float(mean_vec[idx])
        features[f"{sensor_name}_std"] = float(std_vec[idx])
        features[f"{sensor_name}_min"] = float(min_vec[idx])
        features[f"{sensor_name}_max"] = float(max_vec[idx])
        features[f"{sensor_name}_rms"] = float(rms_vec[idx])
    return features


def _discover_phm_cutters(root: Path):
    groups = {}
    for csv_path in sorted(root.rglob("*.csv")):
        cutter_id = _infer_cutter_id(csv_path)
        if cutter_id is None:
            continue
        group = groups.setdefault(cutter_id, {"cuts": [], "wear": None})
        if "wear" in csv_path.stem.lower():
            group["wear"] = csv_path
        else:
            group["cuts"].append(csv_path)
    return {cutter_id: group for cutter_id, group in groups.items() if group["cuts"]}


def _split_forecasting_trajectory(preprocessor: SCADAPreprocessor, data: np.ndarray):
    splits = preprocessor._chronological_split(data)
    train_n, val_n, test_n = preprocessor._normalize(
        splits["train"], splits["val"], splits["test"]
    )

    full_series = np.concatenate([train_n, val_n, test_n], axis=0)
    train_end = len(train_n)
    val_end = train_end + len(val_n)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    max_start = len(full_series) - preprocessor.lookback - preprocessor.horizon + 1
    for start in range(max(0, max_start)):
        X_window = full_series[start:start + preprocessor.lookback]
        y_window = full_series[
            start + preprocessor.lookback:start + preprocessor.lookback + preprocessor.horizon
        ]
        target_end = start + preprocessor.lookback + preprocessor.horizon - 1

        if target_end < train_end:
            X_train.append(X_window)
            y_train.append(y_window)
        elif target_end < val_end:
            X_val.append(X_window)
            y_val.append(y_window)
        else:
            X_test.append(X_window)
            y_test.append(y_window)

    feature_dim = data.shape[1]
    return (
        _stack_or_empty(X_train, (preprocessor.lookback, feature_dim)),
        _stack_or_empty(y_train, (preprocessor.horizon, feature_dim)),
        _stack_or_empty(X_val, (preprocessor.lookback, feature_dim)),
        _stack_or_empty(y_val, (preprocessor.horizon, feature_dim)),
        _stack_or_empty(X_test, (preprocessor.lookback, feature_dim)),
        _stack_or_empty(y_test, (preprocessor.horizon, feature_dim)),
    )


def preprocess_phm_milling() -> dict:
    """Process PHM 2010 cutter data into memory-safe cut-level forecasting windows."""
    phm_raw = RAW_DIR / "phm_milling"
    out_dir = PROC_DIR / "phm_milling"
    sentinel = out_dir / "processed_data.pt"

    preprocessor = SCADAPreprocessor(
        lookback=PHM_MILLING_LOOKBACK, horizon=PHM_MILLING_HORIZON,
        train_ratio=0.70, val_ratio=0.15,
        normalization="standard", seed=SEED,
    )

    if sentinel.exists():
        log.info("  PHM Milling: cached")
        return preprocessor.load_processed(out_dir)

    cutter_groups = _discover_phm_cutters(phm_raw)
    if not cutter_groups:
        log.warning("  PHM Milling: no cutter CSV files found — skipping.")
        raise RuntimeError(
            "PHM Milling raw data is missing. Step 1 must complete fully before preprocessing."
        )

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
    feature_tables = []
    feature_dim = None

    for cutter_id, group in sorted(cutter_groups.items()):
        cut_files = sorted(group["cuts"], key=_cut_sort_key)
        log.info(f"  PHM Milling {cutter_id}: summarizing {len(cut_files)} cut files ...")

        rows = []
        for cut_number, cut_file in enumerate(cut_files, start=1):
            features = _summarize_cut_file(cut_file)
            if features is None:
                log.warning(f"    Skipping unreadable cut file: {cut_file.name}")
                continue
            features["cut_index"] = cut_number
            rows.append(features)

        if len(rows) < preprocessor.lookback + preprocessor.horizon:
            log.warning(
                f"    Skipping {cutter_id}: only {len(rows)} cut summaries, "
                f"need at least {preprocessor.lookback + preprocessor.horizon}."
            )
            continue

        feature_df = pd.DataFrame(rows).sort_values("cut_index").reset_index(drop=True)
        feature_tables.append(feature_df.assign(cutter_id=cutter_id))

        data = feature_df.drop(columns=["cut_index"]).to_numpy(dtype=np.float32)
        feature_dim = data.shape[1]
        X_tr, y_tr, X_va, y_va, X_te, y_te = _split_forecasting_trajectory(preprocessor, data)

        if len(X_tr):
            all_X_train.append(X_tr)
            all_y_train.append(y_tr)
        if len(X_va):
            all_X_val.append(X_va)
            all_y_val.append(y_va)
        if len(X_te):
            all_X_test.append(X_te)
            all_y_test.append(y_te)

        log.info(
            f"    windows train {len(X_tr)}  val {len(X_va)}  test {len(X_te)}"
        )

    if feature_dim is None or not feature_tables:
        raise RuntimeError(
            "PHM Milling preprocessing did not produce any usable cut-level sequences."
        )

    result = {
        "train_X": _cat_or_empty(all_X_train, (preprocessor.lookback, feature_dim)),
        "train_y": _cat_or_empty(all_y_train, (preprocessor.horizon, feature_dim)),
        "val_X": _cat_or_empty(all_X_val, (preprocessor.lookback, feature_dim)),
        "val_y": _cat_or_empty(all_y_val, (preprocessor.horizon, feature_dim)),
        "test_X": _cat_or_empty(all_X_test, (preprocessor.lookback, feature_dim)),
        "test_y": _cat_or_empty(all_y_test, (preprocessor.horizon, feature_dim)),
        "task": "forecasting",
        "dataset": "phm_milling",
        "num_channels": feature_dim,
        "lookback": preprocessor.lookback,
        "horizon": preprocessor.horizon,
        "cutters": sorted(cutter_groups.keys()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(feature_tables, ignore_index=True).to_csv(out_dir / "cut_level_features.csv", index=False)
    preprocessor.save_processed(result, out_dir)

    log.info(f"    train {tuple(result['train_X'].shape)}  "
             f"val {tuple(result['val_X'].shape)}  test {tuple(result['test_X'].shape)}")
    return result


# ── Main entry ───────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    set_seeds()
    log.info("=" * 60)
    log.info("STEP 2: Preprocess Datasets")
    log.info("=" * 60)

    cmapss_data = preprocess_cmapss() if "cmapss" in DATASETS else {}
    scada_data  = preprocess_wind_scada() if "wind_scada" in DATASETS else {}
    phm_data    = preprocess_phm_milling() if "phm_milling" in DATASETS else {}

    if not (cmapss_data or scada_data or phm_data):
        raise RuntimeError(
            "No datasets were preprocessed. Ensure step 1 downloaded raw data into data/raw/."
        )

    summary = {
        "cmapss_subsets": list(cmapss_data.keys()) if cmapss_data else [],
        "wind_scada": bool(scada_data),
        "phm_milling": bool(phm_data),
    }
    mark_step_done("step_02_preprocess", summary)
    log.info("Step 2 complete.")


if __name__ == "__main__":
    main()
