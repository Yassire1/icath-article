# Execution Plan: TSFM Industrial PdM Benchmark Paper

**Deadline:** April 10, 2026  
**Scope:** 6 datasets × 5 open-source TSFMs + PatchTST baseline  
**Platform:** Google Colab Pro with GPU  
**Author:** Yassire Ammouri  

---

## Project Scope

### Datasets (6)

| Dataset | Domain | Primary Task | Priority |
|---------|--------|-------------|----------|
| C-MAPSS | Turbofan Engines | RUL Prediction | HIGH |
| PHM Milling | CNC Machining | Anomaly Detection | MEDIUM |
| PU Bearings | Rotating Machinery | Anomaly Detection | MEDIUM |
| Wind SCADA | Wind Energy | Forecasting | MEDIUM |
| MIMII | Factory Machinery | Anomaly Detection | HIGH |
| PRONOSTIA | Bearings | RUL Prediction | LOW |

### Models (6)

| Model | Type | Parameters | LoRA Support |
|-------|------|-----------|-------------|
| MOMENT | Foundation | 385M | Yes |
| Sundial | Foundation | 200M | Yes |
| Chronos | Foundation | 710M | No |
| Time-MoE | Foundation | 50M | Yes |
| Lag-Llama | Foundation | 7M | Yes |
| PatchTST | Baseline | ~5M | N/A (trained) |

### Experiment Types

1. **Zero-shot** - All 6 models × 6 datasets = 36 combinations
2. **Few-shot LoRA** - 4 fine-tunable models × 3 shot sizes (10, 50, 100) × 6 datasets
3. **Cross-domain** - Domain transfer experiments (C-MAPSS subsets, bearing-to-bearing, machine-to-machine)

---

## PHASE 1: Environment Setup (Day 1 - ~1 hour)

### Step 1.1: Upload Project to Google Drive

1. Zip the entire `icath_conf_Article/` folder
2. Upload to Google Drive → `My Drive/tsfm-pdm-benchmark/`
3. Verify upload completed successfully

### Step 1.2: Run Notebook 01 - Setup Environment

**File:** `notebooks/01_setup_environment.ipynb`

Actions:
- Connect to GPU runtime: `Runtime → Change runtime type → T4 or A100`
- Mount Google Drive
- Install dependencies from `requirements.txt`
- Verify GPU is detected (`nvidia-smi`)
- Verify all imports work (torch, momentfm, chronos, etc.)

Expected time: 10-15 minutes

---

## PHASE 2: Dataset Acquisition (Day 1 - ~2-3 hours)

### Step 2.1: Download All 6 Datasets

**File:** `notebooks/02_download_datasets.ipynb`

| Dataset | Download Method | URL |
|---------|-----------------|-----|
| **C-MAPSS** | Auto-download from NASA | https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6 |
| **MIMII** | Auto-download from Zenodo | https://zenodo.org/record/3384388 |
| **Wind SCADA** | Kaggle CLI | https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset |
| **PHM Milling** | Manual download | https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/ |
| **PU Bearings** | Manual request | https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/ |
| **PRONOSTIA** | GitHub clone | https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset |

**Kaggle Setup (required for Wind SCADA):**
```python
# Upload kaggle.json to Colab, then:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Step 2.2: Verify Downloads

At the end of notebook 02, check all datasets show **READY** status.

Expected directory structure after download:
```
data/raw/
├── cmapss/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_FD002.txt
│   └── ...
├── phm_milling/
├── pu_bearings/
├── wind_scada/
├── mimii/
│   ├── fan/
│   ├── pump/
│   ├── slider/
│   └── valve/
└── pronostia/
```

---

## PHASE 3: Data Preprocessing (Day 1-2 - ~1-2 hours)

### Step 3.1: Run Notebook 03 - Preprocessing

**File:** `notebooks/03_preprocessing.ipynb`

What it does:
- Runs SCADA preprocessing pipeline on all 6 datasets
- Creates chronological train/val/test splits (70/15/15) - NO shuffling
- Applies per-sensor-family normalization (fit on train, transform val/test)
- Generates sliding window sequences (lookback=512, horizon=96)
- Computes RUL labels with piecewise linear degradation
- Saves processed tensors to `data/processed/`

Expected time: 30-60 minutes

### Step 3.2: Verify Processed Data

Check that `data/processed/` contains subdirectories for each dataset with `processed_data.pt` files:

```
data/processed/
├── cmapss/
│   ├── FD001/processed_data.pt
│   ├── FD002/processed_data.pt
│   ├── FD003/processed_data.pt
│   └── FD004/processed_data.pt
├── phm_milling/processed_data.pt
├── pu_bearings/processed_data.pt
├── wind_scada/processed_data.pt
├── mimii/processed_data.pt
└── pronostia/processed_data.pt
```

---

## PHASE 4: Zero-Shot Experiments (Day 2 - ~4-6 hours)

### Step 4.1: Run Notebook 04 - Zero-Shot

**File:** `notebooks/04_zero_shot_experiments.ipynb`

What it does:
- Loads each pre-trained model (MOMENT, Sundial, Chronos, Time-MoE, Lag-Llama, PatchTST)
- Runs inference on all 6 datasets WITHOUT any fine-tuning
- Computes metrics: MAE, MSE, RMSE, MAPE for forecasting; F1, AUC-ROC, AUC-PR for anomaly detection; C-Index, RUL Score for RUL
- Logs results to WandB
- Saves results to `results/zero_shot/`

Expected time: 4-6 hours total (varies by model size and dataset)

**Tip:** Start this before sleeping - it can run overnight.

### Step 4.2: Check Results

Verify `results/zero_shot/zero_shot_results.csv` exists with all 36 model-dataset combinations.

---

## PHASE 5: Few-Shot LoRA Experiments (Day 3 - ~4-6 hours)

### Step 5.1: Run Notebook 05 - Few-Shot

**File:** `notebooks/05_few_shot_experiments.ipynb`

What it does:
- Applies LoRA adapters to fine-tunable models (MOMENT, Sundial, Time-MoE, Lag-Llama)
- Tests 3 shot sizes: 10, 50, 100 training samples
- Compares few-shot results against zero-shot baseline
- Measures improvement percentage per shot size
- Saves results to `results/few_shot/`

Expected time: 4-6 hours

---

## PHASE 6: Cross-Domain Experiments (Day 3-4 - ~3-4 hours)

### Step 6.1: Run Notebook 06 - Cross-Domain

**File:** `notebooks/06_cross_domain_experiments.ipynb`

Transfer pairs:
- C-MAPSS FD001 → FD002, FD003, FD004 (same domain, different conditions)
- PU Bearings → PRONOSTIA (bearing-to-bearing)
- MIMII fan → pump, slider, valve (machine type transfer)

What it does:
- Trains model on source domain, tests on target domain
- Measures performance degradation vs. in-domain baseline
- Quantifies domain gap for industrial PdM

Expected time: 3-4 hours

---

## PHASE 7: Generate Figures & Tables (Day 4 - ~1-2 hours)

### Step 7.1: Run Notebook 07 - Analysis & Visualization

**File:** `notebooks/07_analysis_visualization.ipynb`

Outputs generated:

| Figure | File | Purpose |
|--------|------|---------|
| Zero-shot comparison | `paper/figures/zero_shot_comparison.pdf` | MAE across models and datasets |
| Few-shot scaling | `paper/figures/few_shot_scaling.pdf` | Performance vs. shot size |
| Domain transfer | `paper/figures/domain_transfer.pdf` | Cross-domain degradation |
| Preprocessing impact | `paper/figures/preprocessing_impact.pdf` | Before/after SCADA preprocessing |

Also exports:
- Updated LaTeX tables with real numbers
- Summary statistics CSV

### Step 7.2: Download Results

Download from Colab to local machine:
- `results/` folder (all JSON/CSV files)
- `paper/figures/` folder (all generated PDF figures)

---

## PHASE 8: Update Paper with Real Results (Day 4-5 - ~2-3 hours)

### Step 8.1: Update Tables with Actual Numbers

Replace placeholder numbers in these files:

| File | Content |
|------|---------|
| `paper/tables/table1_capabilities.tex` | Model capabilities summary |
| `paper/tables/table2_datasets.tex` | Dataset statistics |
| `paper/tables/table3_zero_shot.tex` | Zero-shot MAE/MSE/RMSE results |
| `paper/tables/table4_few_shot.tex` | Few-shot improvement percentages |
| `paper/tables/table5_rul.tex` | RUL prediction scores |

### Step 8.2: Update Key Claims in Text

Search and replace these placeholder claims with actual numbers from your results:

| Claim | File |
|-------|------|
| "X% MAE degradation" | `paper/sections/abstract.tex`, `paper/sections/introduction.tex` |
| "X% RUL zero-shot failure" | `paper/sections/analysis.tex` |
| "X% preprocessing improvement" | `paper/sections/analysis.tex` |
| Summary findings | `paper/sections/conclusion.tex` |

### Step 8.3: Verify Figure References

Ensure figures are correctly referenced in:
- `paper/sections/experiments.tex`
- `paper/sections/analysis.tex`

Check that `\includegraphics` paths match the actual figure filenames.

---

## PHASE 9: Compile & Review (Day 5-6 - ~3-4 hours)

### Step 9.1: Compile LaTeX

**Local compilation:**
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Or use Overleaf:**
1. Create new project on Overleaf
2. Upload entire `paper/` folder contents
3. Compile and check for errors

### Step 9.2: Check Page Limit

- **Target:** 8-10 pages (IEEE conference format)
- **If over 10 pages:** Trim discussion section, reduce figure sizes, shorten related work
- **If under 8 pages:** Expand analysis, add ablation study, elaborate on methodology

### Step 9.3: Proofread Checklist

- [ ] Abstract matches actual findings
- [ ] All tables have correct numbers (no placeholders)
- [ ] All figures render properly and are readable
- [ ] All 21 citations are complete in `references.bib`
- [ ] Author name and affiliation are correct
- [ ] No TODO or placeholder text remains
- [ ] Grammar and spelling checked
- [ ] Consistent terminology throughout
- [ ] All acronyms defined on first use
- [ ] Figure captions are descriptive
- [ ] Table captions are self-contained

---

## PHASE 10: Final Submission (Day 6-7)

### Step 10.1: Final Review

- Read paper aloud for flow and clarity
- Check ICATH conference formatting requirements
- Verify PDF is under size limit (usually 10MB)
- Ensure all references are accessible

### Step 10.2: Submit

- Submit PDF to ICATH conference submission system before deadline
- Keep backup copies of everything (local + cloud)
- Note submission confirmation

---

## Quick Reference: Notebook Execution Order

| # | Notebook | Purpose | Est. Time | Dependencies |
|---|----------|---------|-----------|--------------|
| 01 | `01_setup_environment.ipynb` | GPU + dependencies | 15 min | None |
| 02 | `02_download_datasets.ipynb` | Get all 6 datasets | 2-3 hrs | Notebook 01 |
| 03 | `03_preprocessing.ipynb` | SCADA preprocessing | 1 hr | Notebook 02 |
| 04 | `04_zero_shot_experiments.ipynb` | Main benchmark | 4-6 hrs | Notebook 03 |
| 05 | `05_few_shot_experiments.ipynb` | LoRA fine-tuning | 4-6 hrs | Notebook 03 |
| 06 | `06_cross_domain_experiments.ipynb` | Transfer learning | 3-4 hrs | Notebook 03 |
| 07 | `07_analysis_visualization.ipynb` | Figures + tables | 1-2 hrs | Notebooks 04, 05, 06 |

**Total GPU time: ~15-20 hours**

---

## Suggested Daily Schedule

| Day | Tasks | Goal |
|-----|-------|------|
| **Day 1** | Notebooks 01, 02, 03 | Environment ready, data preprocessed |
| **Day 2** | Notebook 04 (zero-shot) | Main benchmark complete |
| **Day 3** | Notebooks 05, 06 | Few-shot + cross-domain complete |
| **Day 4** | Notebook 07 + update paper | Figures generated, tables updated |
| **Day 5** | Compile + proofread | Paper fully compiled and reviewed |
| **Day 6** | Final review + submit | Paper submitted |
| **Day 7** | Buffer day | Handle any issues |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU out of memory | Reduce `batch_size` in `config/config.yaml` |
| Model download fails from HuggingFace | Check internet, retry, or use `hf-cli login` |
| Dataset missing after download | Re-run notebook 02, check manual download URLs |
| WandB connection error | Run with `--no-wandb` flag or check API key |
| Colab disconnects | Enable Colab Pro, use "prevent disconnect" extension |
| LaTeX compilation fails | Check for missing packages, verify figure paths |
| Pandas deprecation warning | Already fixed (`.ffill().bfill()` instead of `fillna(method=...)`) |
| LoRA target modules error | Already fixed (dynamic detection added) |

---

## Files You Will Modify

**During experiments:** Nothing - notebooks handle everything automatically

**After experiments (manual updates):**
- `paper/tables/table3_zero_shot.tex` - Replace with actual MAE/MSE/RMSE
- `paper/tables/table4_few_shot.tex` - Replace with actual improvement %
- `paper/tables/table5_rul.tex` - Replace with actual RUL scores
- `paper/sections/abstract.tex` - Update key numbers
- `paper/sections/introduction.tex` - Update contribution claims
- `paper/sections/analysis.tex` - Update analysis with real results
- `paper/sections/conclusion.tex` - Update summary findings

---

## Important Notes

1. **Run notebooks in order** - Each depends on the previous one's output
2. **Save results frequently** - Download `results/` folder after each experiment phase
3. **Use Colab Pro** - Free tier will disconnect during long experiments
4. **WandB is enabled** - Monitor experiments at https://wandb.ai in real-time
5. **6 bugs already fixed** - models.yaml syntax, deprecated pandas, bare excepts, O(n²) concordance index, LoRA target modules
6. **Backup everything** - Keep copies on Google Drive and local machine
