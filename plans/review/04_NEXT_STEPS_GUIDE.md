# Step-by-Step Next Actions Guide

**For:** Yassire Ammouri  
**Context:** You are on your local Windows machine. The codebase was AI-generated. You need to execute real experiments and produce a submission-ready paper.  
**Date:** April 10, 2026

---

## Decision: Colab Free vs. Colab Pro

**Recommendation: Start with Colab Free. Upgrade to Pro ONLY if you hit session limits.**

Here is why:
- The reduced scope (3 datasets × 4 models) requires ~10-15 GPU hours total
- Colab Free gives T4 GPU for up to 12 hours per session (varies by demand)
- You can split experiments across multiple sessions
- Colab Pro ($10/mo) gives longer sessions + priority GPU — worth it if Free tier keeps disconnecting you during a 3-hour MOMENT run, or if you're in a timezone where free GPU is scarce

**If you can afford $10 without stress, get Pro. It removes friction. But it's NOT required for this scope.**

---

## STEP 0: Before You Touch Colab (Do This NOW on Your Local Machine)

**Time: 1-2 hours**

### 0.1 Delete all fabricated results from the paper
This is non-negotiable. You cannot have fake numbers in your repo.

Files to clean:
- `paper/tables/table3_zero_shot.tex` — delete all MAE/MSE values, leave table structure
- `paper/tables/table4_few_shot.tex` — delete all values
- `paper/tables/table5_rul.tex` — delete all values
- `paper/sections/abstract.tex` — replace specific numbers with `[TBD]`
- `paper/sections/introduction.tex` — replace specific percentages with `[TBD]`
- `paper/sections/analysis.tex` — replace 35%, 62%, 18% with `[TBD]`
- `paper/sections/conclusion.tex` — replace specific claims with `[TBD]`

### 0.2 Pass the optimized plan to your AI agent
Give `plans/review/03_OPTIMIZED_PLAN.md` to your coding agent with these instructions:
1. Fix PatchTST `predict()` — replace the naive stub with proper neuralforecast inference
2. Add `src/models/lag_llama.py` wrapper (open-source, HuggingFace: `time-series-foundation-models/Lag-Llama`)
3. Fix MOMENT wrapper duplication bug
4. Update `src/models/__init__.py` MODEL_REGISTRY: add lag_llama, remove timegpt
5. Update `config/models.yaml` to reflect 4 models (MOMENT, Chronos, Lag-Llama, PatchTST)
6. Update `config/datasets.yaml` to mark 3 active datasets (cmapss, mimii, wind_scada)

### 0.3 Push to GitHub
Create a private repo and push. You'll clone this in Colab.
```bash
cd icath_conf_Article
git init
git add .
git commit -m "Initial scaffold - pre-experimentation"
git remote add origin https://github.com/YOUR_USERNAME/tsfm-pdm-bench.git
git push -u origin main
```

---

## STEP 1: Colab Setup (Day 1, Session 1)

**Time: 30 minutes**

### 1.1 Open Colab, set runtime to T4 GPU
`Runtime → Change runtime type → T4 GPU`

### 1.2 Clone your repo
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/tsfm-pdm-bench.git /content/tsfm-pdm-bench
%cd /content/tsfm-pdm-bench
```

### 1.3 Install dependencies
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers accelerate peft
!pip install -q momentfm chronos-forecasting
!pip install -q gluonts[torch] neuralforecast
!pip install -q numpy pandas scikit-learn matplotlib seaborn plotly kaleido
!pip install -q wandb pyyaml tqdm scipy
# Lag-Llama:
!pip install -q git+https://github.com/time-series-foundation-models/lag-llama.git
```

### 1.4 Verify GPU + imports
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Test imports
from src.models import get_model, list_models
print(list_models())
```

---

## STEP 2: Download Datasets (Day 1, Session 1 continued)

**Time: 1-2 hours (some require manual steps)**

### 2.1 C-MAPSS (automated)
```python
# NASA C-MAPSS is publicly available
!python -c "from src.data.download import download_cmapss; download_cmapss('data/raw')"
```
If auto-download fails, manually download from:
https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip
Upload the zip to Colab and extract to `data/raw/cmapss/`

### 2.2 MIMII (automated from Zenodo)
```python
!python -c "from src.data.download import download_mimii; download_mimii('data/raw')"
```
Note: MIMII is ~1.2GB. This will take time on Colab's network.

### 2.3 Wind SCADA (Kaggle)
Option A (Kaggle API):
```python
# Upload your kaggle.json first
!mkdir -p ~/.kaggle
# Then:
!kaggle datasets download -d berkerisen/wind-turbine-scada-dataset -p data/raw/wind_scada/
!unzip data/raw/wind_scada/*.zip -d data/raw/wind_scada/
```

Option B (manual): Download from Kaggle, upload to Google Drive, copy to Colab.

### 2.4 Verify all data
```python
import os
for d in ['cmapss', 'mimii', 'wind_scada']:
    path = f'data/raw/{d}'
    files = os.listdir(path) if os.path.exists(path) else []
    print(f"{d}: {len(files)} files - {'OK' if files else 'MISSING'}")
```

---

## STEP 3: Preprocessing (Day 1-2)

**Time: 30-60 minutes**

```python
from src.data.preprocessing import SCADAPreprocessor, preprocess_all_datasets

preprocessor = SCADAPreprocessor()

# Process C-MAPSS (all 4 subsets)
for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
    data = preprocessor.process_cmapss('data/raw/cmapss', subset=subset)
    preprocessor.save_processed(data, f'data/processed/cmapss/{subset}')
    print(f"C-MAPSS {subset}: train_X shape = {data['train_X'].shape}")

# Process Wind SCADA
data = preprocessor.process_generic_csv('data/raw/wind_scada')
preprocessor.save_processed(data, 'data/processed/wind_scada')

# Process MIMII (may need adaptation for audio → spectrogram)
# NOTE: Check if preprocessing.py handles MIMII correctly
# MIMII is audio data — it needs mel-spectrogram conversion first
```

**IMPORTANT:** MIMII is an audio dataset. The preprocessing pipeline may not handle it correctly out of the box. If it fails, you have two options:
1. Fix the preprocessing to convert audio to mel-spectrograms first
2. Replace MIMII with PHM Milling (simpler tabular data)

### Save processed data to Drive (Checkpoint!)
```python
!cp -r data/processed /content/drive/MyDrive/tsfm-pdm-bench-data/
```

---

## STEP 4: Zero-Shot Experiments (Day 2-3)

**Time: 4-6 GPU hours across sessions**

### 4.1 Run one model at a time (safer for Colab session limits)

```python
# Session 1: MOMENT zero-shot (~1-2 hrs)
from src.experiments.run_zero_shot import run_zero_shot_experiment
from src.models import get_model
from src.data.preprocessing import SCADAPreprocessor

preprocessor = SCADAPreprocessor()

model = get_model('moment')
model.load_model()

datasets = {
    'cmapss_FD001': preprocessor.load_processed('data/processed/cmapss/FD001'),
    'wind_scada': preprocessor.load_processed('data/processed/wind_scada'),
    # Add MIMII if preprocessed
}

for name, data in datasets.items():
    result = run_zero_shot_experiment(model, data, dataset_name=name)
    print(f"MOMENT on {name}: MAE={result['mae']:.4f}")
    # Save individually
    import json
    with open(f'results/zero_shot/moment_{name}.json', 'w') as f:
        json.dump(result, f)

# CHECKPOINT: Save results to Drive
!cp -r results/ /content/drive/MyDrive/tsfm-pdm-bench-results/
```

```python
# Session 2: Chronos zero-shot (~2-3 hrs)
# Same pattern as above but with get_model('chronos')
```

```python
# Session 3: Lag-Llama zero-shot (~1 hr)
# Same pattern with get_model('lag_llama')
```

```python
# Session 4: PatchTST (train + evaluate) (~2-3 hrs)
# PatchTST requires training first, then evaluation
model = get_model('patchtst')
model.load_model()
# Train on each dataset
for name, data in datasets.items():
    model.fit(data['train_X'], data['train_y'])
    result = run_zero_shot_experiment(model, data, dataset_name=name)
    print(f"PatchTST on {name}: MAE={result['mae']:.4f}")
```

### 4.2 After all zero-shot runs, compile results
```python
import pandas as pd
import glob, json

results = []
for f in glob.glob('results/zero_shot/*.json'):
    with open(f) as fp:
        r = json.load(fp)
        results.append(r)

df = pd.DataFrame(results)
df.to_csv('results/zero_shot/zero_shot_results.csv', index=False)
print(df.pivot_table(values='mae', index='model', columns='dataset'))
```

---

## STEP 5: Few-Shot + Cross-Condition (Day 3-4)

**Time: 3-4 GPU hours**

### 5.1 Few-shot LoRA (MOMENT + Lag-Llama only)
```python
from src.experiments.run_few_shot import run_few_shot_experiment

for model_name in ['moment', 'lag_llama']:
    model = get_model(model_name)
    model.load_model()
    for dataset_name, data in datasets.items():
        result = run_few_shot_experiment(
            model, data,
            dataset_name=dataset_name,
            train_ratio=0.01,
            lora_r=16,
            epochs=10
        )
        # Save result

# CHECKPOINT
!cp -r results/ /content/drive/MyDrive/tsfm-pdm-bench-results/
```

### 5.2 Cross-condition transfer (C-MAPSS only)
```python
# Train/evaluate on FD001, test on FD002/FD003/FD004
source = preprocessor.load_processed('data/processed/cmapss/FD001')
for target_subset in ['FD002', 'FD003', 'FD004']:
    target = preprocessor.load_processed(f'data/processed/cmapss/{target_subset}')
    # Run MOMENT zero-shot on target after seeing source
    # This tests same-domain cross-condition transfer
```

---

## STEP 6: Generate Figures & Tables (Day 5)

**Time: 1-2 hours (CPU only, can do locally)**

### 6.1 Download results from Drive to local machine
Copy `results/` folder from Google Drive to your local `icath_conf_Article/results/`

### 6.2 Generate figures
```python
from src.visualization.figures import generate_all_figures
generate_all_figures('results/', 'paper/figures/')
```

### 6.3 Generate LaTeX tables with REAL numbers
```python
from src.visualization.tables import generate_all_tables
generate_all_tables('results/', 'paper/tables/')
```

---

## STEP 7: Rewrite Paper (Day 5-6)

**Time: 4-6 hours of focused writing**

### 7.1 Replace [TBD] placeholders with actual numbers
Go through every paper section and replace placeholder claims with real results.

### 7.2 Key sections to rewrite from scratch
- Abstract — write based on actual findings
- Introduction contributions list — match actual experiments
- Results section — all tables and analysis based on real data
- Discussion — interpret your REAL observations
- Conclusion — summarize what you ACTUALLY found

### 7.3 Pass to your coding agent for LaTeX compilation
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload to Overleaf for easier compilation.

---

## STEP 8: Review & Submit (Day 6-7)

### 8.1 Self-review checklist
- [ ] No fabricated/placeholder numbers remain
- [ ] All tables match actual experiment results
- [ ] All figures were generated from real data
- [ ] No overclaimed contributions ("first", "novel pipeline")
- [ ] All referenced prior work is cited (FoundTS, TSFM-Bench, Dintén 2025)
- [ ] Code repo is public and instructions work
- [ ] Paper is within page limits (6-8 pages)
- [ ] Author name and affiliation are correct

### 8.2 Pass the final PDF back to me (AI reviewer) for a final check
Before submitting, have the paper reviewed one more time.

### 8.3 Submit to ICATH
Follow the conference submission system instructions.

---

## Emergency Contingencies

### If MIMII preprocessing fails (audio data issue)
Replace MIMII with one of:
- PHM Milling — simpler tabular data, anomaly detection
- Use only C-MAPSS + Wind SCADA (2 datasets is still acceptable for a proceedings paper)

### If Colab keeps disconnecting
1. **Option A:** Get Colab Pro ($10)
2. **Option B:** Use Kaggle Notebooks (30 hrs/week free GPU)
3. **Option C:** Run smaller models locally if you have even a basic GPU

### If a model crashes on a dataset
Skip that model-dataset combination. Report "N/A — model failed to produce valid forecasts for this dataset" in your table. This is honest and actually interesting to reviewers.

### If results show TSFMs actually perform WELL on industrial data
Great — that's a positive finding. Reframe the paper: "TSFMs show promising transfer to industrial PdM, with caveats" instead of "TSFMs fail." The paper's contribution is the empirical evidence either way.

---

## Summary: Your Next 7 Days

| Day | Action | Deliverable |
|-----|--------|-------------|
| **Today** | Clean fabricated data, fix code bugs, push to GitHub | Clean codebase |
| **Day 2** | Colab: setup + download + preprocess | data/processed/ ready |
| **Day 3** | Colab: MOMENT + Chronos zero-shot runs | First real results |
| **Day 4** | Colab: Lag-Llama + PatchTST + few-shot | All experiments done |
| **Day 5** | Local: generate figures + tables + start rewriting paper | Paper v1 |
| **Day 6** | Write/polish paper, compile LaTeX | Paper v2 (near-final) |
| **Day 7** | Final review, submit | Submitted |

**Start now. Step 0 takes 1 hour and requires no GPU.**
