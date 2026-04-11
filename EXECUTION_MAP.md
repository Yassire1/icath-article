# EXECUTION MAP — TSFM-PDM Benchmark Pipeline

> **Purpose**: Guide an AI agent (or human) through executing the full experiment
> pipeline on a GCP VM (CPU only, high RAM).

---

## Quick Start

```bash
# From the project root directory:
cd /path/to/icath_conf_Article

# Run the entire pipeline end-to-end:
python scripts/run_pipeline.py

# Or run individual steps:
python scripts/step_01_download.py
python scripts/step_02_preprocess.py
# ... etc.
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    run_pipeline.py                           │
│              (orchestrator — chains all steps)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Step 1 ──► Step 2 ──► Step 3 ──► Step 4 ──► Step 5       │
│   Download   Preproc    Zero-shot  Few-shot   Cross-cond    │
│                           │           │           │         │
│                           └─────┬─────┘           │         │
│                                 ▼                 ▼         │
│                            Step 6 ──► Step 7 ──► Step 8     │
│                            Profile    Aggregate  Visualize  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step Dependencies

| Step | Script | Depends On | Can Skip? | Description |
|------|--------|------------|-----------|-------------|
| 1 | `step_01_download.py` | None | Yes, if data is already in `data/raw/` | Download C-MAPSS, Wind SCADA, MIMII |
| 2 | `step_02_preprocess.py` | Step 1 | Yes, if `.pt` files exist in `data/processed/` | Preprocess all datasets |
| 3 | `step_03_zero_shot.py` | Step 2 | No | Run zero-shot experiments |
| 4 | `step_04_few_shot.py` | Step 2 | No | Run few-shot LoRA experiments |
| 5 | `step_05_cross_condition.py` | Step 2 | No | Cross-condition transfer (C-MAPSS only) |
| 6 | `step_06_profile.py` | Step 2 | Yes | Inference latency profiling |
| 7 | `step_07_aggregate.py` | Steps 3, 4, 5 | No | Aggregate results → CSV tables |
| 8 | `step_08_visualize.py` | Step 7 | No | Generate PDF/PNG figures |

**Independent steps**: Steps 3, 4, 5, and 6 are independent of each other (all depend only on step 2). They can be run in any order.

---

## Step-by-Step Guide

### Step 1: Download Datasets
```bash
python scripts/step_01_download.py
```
- **Requires**: Kaggle CLI configured (`~/.kaggle/kaggle.json`)
- **Downloads**: C-MAPSS (Kaggle), Wind SCADA (Kaggle), MIMII fan (Zenodo)
- **Output**: `data/raw/{cmapss,wind_scada,mimii}/`
- **Idempotent**: Skips already-downloaded datasets

### Step 2: Preprocess Datasets
```bash
python scripts/step_02_preprocess.py
```
- **C-MAPSS**: All 4 subsets (FD001–FD004), lookback=64, horizon=30, RUL labels
- **Wind SCADA**: Generic CSV, lookback=512, horizon=96, forecasting task
- **MIMII**: WAV → MFCC (40 coefficients) → sliding windows
  - Uses **chunked processing** with memory-mapped arrays to avoid RAM overflow
  - Processes 50 WAV files at a time, writes MFCC to disk-backed numpy memmap
- **Output**: `data/processed/{dataset}/processed_data.pt`
- **Idempotent**: Skips datasets whose `processed_data.pt` already exists

### Step 3: Zero-Shot Experiments
```bash
python scripts/step_03_zero_shot.py
```
- **Models**: MOMENT, Chronos, Lag-Llama, PatchTST
- **Datasets**: C-MAPSS/FD001, Wind SCADA, MIMII
- **Note**: PatchTST is supervised — it trains on the training split first
- **Metrics**: MAE, RMSE, MAPE
- **Output**: `results/zero_shot/{model}_{dataset}.json`
- **Idempotent**: Skips model-dataset pairs whose JSON already exists

### Step 4: Few-Shot LoRA Experiments
```bash
python scripts/step_04_few_shot.py
```
- **Models**: MOMENT, Lag-Llama (LoRA-capable only)
- **Training data**: 1% of training set (minimum 32 samples)
- **LoRA config**: r=16, alpha=32, 10 epochs, lr=1e-4
- **Output**: `results/few_shot/{model}_{dataset}_few_shot.json`

### Step 5: Cross-Condition Transfer
```bash
python scripts/step_05_cross_condition.py
```
- **Source**: C-MAPSS FD001 (trained/loaded once)
- **Targets**: FD001 (in-domain baseline), FD002, FD003, FD004
- **All 4 models** evaluated on each target
- **Output**: `results/cross_condition/{model}_FD001_to_{target}.json`

### Step 6: Inference Profiling
```bash
python scripts/step_06_profile.py
```
- **Benchmark batch**: 32 samples from C-MAPSS FD001
- **Measures**: Parameter count, inference latency (ms), peak memory
- **Output**: `results/tables/profile_{model}.json`, `efficiency_summary.csv`

### Step 7: Aggregate Results
```bash
python scripts/step_07_aggregate.py
```
- Reads all JSON files from steps 3–5
- **Produces**:
  - `results/tables/zero_shot_mae.csv` — pivot table (model × dataset)
  - `results/tables/few_shot_mae.csv` — pivot table
  - `results/tables/cross_condition_mae.csv` — pivot table
  - `results/tables/zs_vs_fs_comparison.csv` — improvement percentages
  - `results/tables/all_results.csv` — master dataset
  - `results/tables/*_full.csv` — detailed per-run records

### Step 8: Generate Figures
```bash
python scripts/step_08_visualize.py
```
- **Produces** (in `results/figures/`):
  - `zero_shot_mae_bar.{pdf,png}` — Grouped bar chart
  - `zs_vs_fs_comparison.{pdf,png}` — Side-by-side comparison
  - `cross_condition_heatmap.{pdf,png}` — Transfer matrix heatmap

---

## Running the Pipeline

### Full Pipeline (Recommended)
```bash
python scripts/run_pipeline.py
```

### Resume from a Specific Step
```bash
# If step 2 completed but step 3 failed:
python scripts/run_pipeline.py --from 3
```

### Run Specific Steps Only
```bash
# Re-generate tables and figures only:
python scripts/run_pipeline.py --only 7 8
```

### Skip a Step
```bash
# Skip profiling (step 6):
python scripts/run_pipeline.py --skip 6
```

---

## Configuration

All experiment parameters are in **`scripts/pipeline_config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | `"cpu"` | Compute device (set to `"cuda"` if GPU available) |
| `MODELS_ZERO_SHOT` | `["moment", "chronos", "lag_llama", "patchtst"]` | Models for zero-shot |
| `MODELS_FEW_SHOT` | `["moment", "lag_llama"]` | LoRA-capable models |
| `CMAPSS_LOOKBACK` | `64` | Context window for C-MAPSS |
| `CMAPSS_HORIZON` | `30` | Forecast horizon for C-MAPSS |
| `LOOKBACK` | `512` | Context window for Wind SCADA / MIMII |
| `HORIZON` | `96` | Forecast horizon for Wind SCADA / MIMII |
| `MIMII_MAX_FILES` | `500` | Max WAV files to process (increase for full dataset) |
| `TRAIN_RATIO` | `0.01` | Few-shot training ratio (1%) |
| `SEED` | `42` | Random seed |

---

## Output Structure

```
results/
├── manifest.json              ← Step completion tracking (timestamps, status)
├── pipeline.log               ← Full execution log (console + file)
├── zero_shot/
│   ├── moment_cmapss_FD001.json
│   ├── chronos_cmapss_FD001.json
│   ├── moment_wind_scada.json
│   └── ...
├── few_shot/
│   ├── moment_cmapss_FD001_few_shot.json
│   └── ...
├── cross_condition/
│   ├── moment_FD001_to_FD001.json
│   ├── moment_FD001_to_FD002.json
│   └── ...
├── tables/
│   ├── zero_shot_mae.csv          ← Pivot: model × dataset
│   ├── few_shot_mae.csv           ← Pivot: model × dataset
│   ├── cross_condition_mae.csv    ← Pivot: model × target
│   ├── zs_vs_fs_comparison.csv    ← Improvement percentages
│   ├── all_results.csv            ← Master results file
│   ├── efficiency_summary.csv     ← Profiling results
│   └── profile_{model}.json       ← Per-model profiling
└── figures/
    ├── zero_shot_mae_bar.pdf
    ├── zero_shot_mae_bar.png
    ├── zs_vs_fs_comparison.pdf
    ├── zs_vs_fs_comparison.png
    ├── cross_condition_heatmap.pdf
    └── cross_condition_heatmap.png
```

---

## Tracking & Resumability

- **`results/manifest.json`**: Automatically updated after each step completes.
  Contains timestamps, status, and summary details per step.
- **Per-run JSON caching**: Each experiment (model × dataset) saves a JSON file.
  Re-running a step skips experiments whose JSON already exists.
- **To force re-run**: Delete the relevant JSON file(s) or `processed_data.pt`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root: `cd /path/to/icath_conf_Article` |
| Kaggle download fails | Ensure `~/.kaggle/kaggle.json` exists with valid credentials |
| MIMII OOM (RAM) | Reduce `MIMII_MAX_FILES` in `pipeline_config.py` |
| Slow inference on CPU | Expected — CPU-only inference is 10-50x slower than GPU |
| Step fails midway | Fix the issue, then resume: `python scripts/run_pipeline.py --from <step>` |
| Need to re-run one experiment | Delete its JSON in `results/` and re-run the step |

---

## For AI Agents

### Execution Order
1. `cd` to the project root directory
2. Ensure dependencies are installed: `pip install -r requirements.txt`
3. Run: `python scripts/run_pipeline.py`
4. Monitor progress via `results/pipeline.log`
5. Check completion via `results/manifest.json`
6. Final outputs are in `results/tables/*.csv` and `results/figures/*.{pdf,png}`

### Checking Status
```python
import json
manifest = json.load(open("results/manifest.json"))
for step, info in manifest["steps"].items():
    print(f"{step}: {info['status']} ({info['timestamp']})")
```

### Re-running Failed Steps
```bash
# Check which step failed:
cat results/manifest.json | python -m json.tool

# Resume from the failed step:
python scripts/run_pipeline.py --from <failed_step_number>
```
