"""
Step 2: Preprocess all datasets (C-MAPSS, Wind SCADA, MIMII).

- C-MAPSS: all four subsets (FD001-FD004), RUL task
- Wind SCADA: generic CSV, forecasting task
- MIMII: WAV → MFCC → sliding-window, forecasting task
  Uses chunked processing to stay within RAM limits.

Idempotent — skips datasets whose processed_data.pt already exists.
Run:  python scripts/step_02_preprocess.py
"""

import gc
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline_config import (
    RAW_DIR, PROC_DIR,
    CMAPSS_SUBSETS, CMAPSS_LOOKBACK, CMAPSS_HORIZON,
    LOOKBACK, HORIZON, SEED,
    MIMII_MACHINES, MIMII_MAX_FILES, MIMII_N_MFCC,
    MIMII_SR, MIMII_N_FFT, MIMII_HOP, DATASETS,
    setup_logging, ensure_dirs, set_seeds, mark_step_done,
)
from src.data.preprocessing import SCADAPreprocessor

warnings.filterwarnings("ignore")
log = setup_logging()


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


# ── MIMII ────────────────────────────────────────────────────────────────
def preprocess_mimii() -> dict:
    """Process MIMII WAV files → MFCC timeline → sliding windows.

    Uses chunked WAV loading to keep peak RAM usage low.
    """
    import librosa

    mimii_raw = RAW_DIR / "mimii"
    out_dir = PROC_DIR / "mimii"
    sentinel = out_dir / "processed_data.pt"

    preprocessor = SCADAPreprocessor(
        lookback=LOOKBACK, horizon=HORIZON,
        train_ratio=0.70, val_ratio=0.15,
        normalization="standard", seed=SEED,
    )

    if sentinel.exists():
        log.info("  MIMII: cached")
        return preprocessor.load_processed(out_dir)

    all_wavs = sorted(mimii_raw.rglob("*.wav"))[:MIMII_MAX_FILES]
    if not all_wavs:
        log.warning("  MIMII: no WAV files found — skipping.")
        raise RuntimeError(
            "MIMII raw data is missing. Step 1 must complete fully before preprocessing."
        )

    log.info(f"  MIMII: extracting MFCC from {len(all_wavs)} WAV files (chunked) ...")

    # ── Chunked extraction — process WAVs in batches of CHUNK_SIZE,
    #    append MFCC frames to a memory-mapped temp file so we never
    #    hold all raw audio + all MFCCs in RAM simultaneously.
    CHUNK_SIZE = 50  # WAV files per chunk

    # First pass: compute total frames to pre-allocate.
    # librosa uses centered STFT by default, so frame count is:
    #   n_frames = 1 + floor(len(y) / hop_length)
    total_frames = 0
    for wav in all_wavs:
        y, sr = librosa.load(wav, sr=MIMII_SR, mono=True)
        n_frames = 1 + (len(y) // MIMII_HOP)
        total_frames += n_frames
        del y
        gc.collect()

    log.info(f"    Total MFCC frames to extract: {total_frames}")

    # Pre-allocate numpy array on disk via memmap
    tmp_dir = tempfile.mkdtemp(prefix="mimii_mfcc_")
    mmap_path = Path(tmp_dir) / "mfcc_timeline.npy"
    timeline = np.memmap(
        mmap_path, dtype=np.float32, mode="w+",
        shape=(total_frames, MIMII_N_MFCC),
    )

    write_pos = 0
    for chunk_start in range(0, len(all_wavs), CHUNK_SIZE):
        chunk_wavs = all_wavs[chunk_start : chunk_start + CHUNK_SIZE]
        for wav in chunk_wavs:
            y, sr = librosa.load(wav, sr=MIMII_SR, mono=True)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=MIMII_N_MFCC,
                n_fft=MIMII_N_FFT, hop_length=MIMII_HOP,
            ).T  # (frames, n_mfcc)
            n = mfcc.shape[0]
            if write_pos + n > total_frames:
                raise RuntimeError(
                    "MFCC pre-allocation was too small; "
                    "verify frame counting logic before continuing."
                )
            timeline[write_pos : write_pos + n] = mfcc
            write_pos += n
            del y, mfcc
        gc.collect()
        done = min(chunk_start + CHUNK_SIZE, len(all_wavs))
        log.info(f"    Processed {done}/{len(all_wavs)} WAV files")

    timeline.flush()

    log.info(f"    MFCC timeline shape: ({write_pos}, {MIMII_N_MFCC})")

    # Write timeline to temp CSV for process_generic_csv
    tmp_csv = Path(tmp_dir) / "mimii_mfcc.csv"
    # Write in chunks to avoid loading entire memmap into pandas at once
    CSV_CHUNK = 100_000
    header_written = False
    for start in range(0, write_pos, CSV_CHUNK):
        end = min(start + CSV_CHUNK, write_pos)
        chunk_df = pd.DataFrame(timeline[start:end])
        chunk_df.to_csv(
            tmp_csv, index=False,
            mode="a" if header_written else "w",
            header=not header_written,
        )
        header_written = True
        del chunk_df
    gc.collect()

    # Free the memmap
    del timeline
    gc.collect()
    mmap_path.unlink(missing_ok=True)

    log.info("    Running sliding-window preprocessing ...")
    data = preprocessor.process_generic_csv(tmp_csv, task="forecasting")
    preprocessor.save_processed(data, out_dir)

    # Cleanup temp files
    tmp_csv.unlink(missing_ok=True)
    try:
        Path(tmp_dir).rmdir()
    except OSError:
        pass

    log.info(f"    train {tuple(data['train_X'].shape)}  "
             f"val {tuple(data['val_X'].shape)}  test {tuple(data['test_X'].shape)}")
    return data


# ── Main entry ───────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    set_seeds()
    log.info("=" * 60)
    log.info("STEP 2: Preprocess Datasets")
    log.info("=" * 60)

    cmapss_data = preprocess_cmapss() if "cmapss" in DATASETS else {}
    scada_data  = preprocess_wind_scada() if "wind_scada" in DATASETS else {}
    mimii_data  = preprocess_mimii() if "mimii" in DATASETS else {}

    if not (cmapss_data or scada_data or mimii_data):
        raise RuntimeError(
            "No datasets were preprocessed. Ensure step 1 downloaded raw data into data/raw/."
        )

    summary = {
        "cmapss_subsets": list(cmapss_data.keys()) if cmapss_data else [],
        "wind_scada": bool(scada_data),
        "mimii": bool(mimii_data),
    }
    mark_step_done("step_02_preprocess", summary)
    log.info("Step 2 complete.")


if __name__ == "__main__":
    main()
