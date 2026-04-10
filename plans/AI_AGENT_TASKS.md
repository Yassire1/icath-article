# AI Agent Task Execution Plan

## Project: TSFM Industrial PdM Benchmark
## Deadline: April 10, 2026
## Total Duration: 7 Days

---

# TASK REGISTRY

Each task has:
- **ID**: Unique identifier
- **Priority**: P0 (critical), P1 (high), P2 (medium), P3 (low)
- **Dependencies**: Tasks that must complete first
- **Estimated Time**: Human hours / AI-assisted hours
- **Deliverables**: Expected outputs
- **Verification**: How to confirm completion

---

## DAY 1: FOUNDATION SETUP

### TASK-001: Create Project Structure
```
Priority: P0
Dependencies: None
Time: 0.5h
```

**Action**: Create directory structure
```bash
mkdir -p icath_conf_Article/{config,data/{raw,processed}/{cmapss,phm_milling,pu_bearings,wind_scada,mimii,pronostia},src/{data,models,evaluation,experiments,visualization},notebooks,results/{zero_shot,few_shot,cross_domain,tables,figures},paper/{sections,figures},scripts}

touch icath_conf_Article/src/__init__.py
touch icath_conf_Article/src/data/__init__.py
touch icath_conf_Article/src/models/__init__.py
touch icath_conf_Article/src/evaluation/__init__.py
touch icath_conf_Article/src/experiments/__init__.py
touch icath_conf_Article/src/visualization/__init__.py
```

**Deliverables**:
- [ ] All directories created
- [ ] All `__init__.py` files created

**Verification**: `ls -la icath_conf_Article/src/`

---

### TASK-002: Create Environment Files
```
Priority: P0
Dependencies: TASK-001
Time: 0.5h
```

**Action**: Create `environment.yml`
```yaml
name: tsfm-bench
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.1.0
  - pytorch-cuda=12.1
  - numpy=1.26.0
  - pandas=2.2.0
  - scikit-learn=1.5.0
  - matplotlib=3.9.0
  - seaborn=0.13.0
  - pip
  - pip:
    - transformers==4.44.0
    - accelerate==0.32.0
    - peft==0.12.0
    - wandb==0.17.0
    - nixtla==0.5.0
    - neuralforecast==1.7.0
    - plotly==5.22.0
    - kaleido==0.2.1
    - pyyaml==6.0.1
    - tqdm==4.66.0
```

**Action**: Create `requirements.txt`
```
torch==2.1.0
transformers==4.44.0
accelerate==0.32.0
peft==0.12.0
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.0
plotly==5.22.0
wandb==0.17.0
nixtla==0.5.0
neuralforecast==1.7.0
pyyaml==6.0.1
tqdm==4.66.0
```

**Deliverables**:
- [ ] `environment.yml` created
- [ ] `requirements.txt` created

**Verification**: File exists and is valid YAML/text

---

### TASK-003: Create Configuration Files
```
Priority: P0
Dependencies: TASK-001
Time: 1h
```

**Action**: Create three config files as specified in IMPLEMENTATION_PLAN.md:
1. `config/config.yaml` - Main configuration
2. `config/datasets.yaml` - Dataset specifications
3. `config/models.yaml` - Model specifications

**Deliverables**:
- [ ] `config/config.yaml`
- [ ] `config/datasets.yaml`
- [ ] `config/models.yaml`

**Verification**: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`

---

### TASK-004: Create Data Download Module
```
Priority: P0
Dependencies: TASK-003
Time: 1h
```

**Action**: Create `src/data/download.py` with functions:
- `download_cmapss()` - Instructions for NASA C-MAPSS
- `download_phm_milling()` - Instructions for PHM Society
- `download_pu_bearings()` - Instructions for Paderborn
- `download_wind_scada()` - Kaggle download
- `download_mimii()` - Zenodo auto-download
- `download_pronostia()` - GitHub clone
- `download_all_datasets()` - Master function

**Deliverables**:
- [ ] `src/data/download.py`

**Verification**: `python -m src.data.download` prints instructions

---

### TASK-005: Create Preprocessing Pipeline
```
Priority: P0
Dependencies: TASK-003
Time: 2h
```

**Action**: Create `src/data/preprocessing.py` with:
- `SCADAPreprocessor` class
- `_kalman_impute()` method
- `_chronological_split()` method
- `_normalize()` method
- `_create_sequences()` method
- `process_cmapss()` method
- `TSFMDataset` class

**Deliverables**:
- [ ] `src/data/preprocessing.py`
- [ ] `src/data/datasets.py` (PyTorch datasets)

**Verification**: Unit test with dummy data

---

### TASK-006: Download C-MAPSS Dataset
```
Priority: P0
Dependencies: TASK-004
Time: 1h (manual intervention required)
```

**Action**: 
1. Navigate to https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip
2. Download CMAPSSData.zip
3. Extract to `data/raw/cmapss/`

**Expected files**:
```
data/raw/cmapss/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
├── RUL_FD003.txt
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

**Deliverables**:
- [ ] All 12 C-MAPSS files downloaded

**Verification**: `ls data/raw/cmapss/*.txt | wc -l` returns 12

---

### TASK-007: Test Preprocessing Pipeline
```
Priority: P0
Dependencies: TASK-005, TASK-006
Time: 1h
```

**Action**: Run preprocessing on C-MAPSS FD001
```python
from src.data.preprocessing import SCADAPreprocessor
preprocessor = SCADAPreprocessor()
data = preprocessor.process_cmapss(Path("data/raw/cmapss"), subset="FD001")
preprocessor.save_processed(data, Path("data/processed/cmapss/FD001"))
```

**Deliverables**:
- [ ] `data/processed/cmapss/FD001/processed_data.pt`

**Verification**: 
```python
import torch
data = torch.load("data/processed/cmapss/FD001/processed_data.pt")
assert 'train_X' in data
assert data['train_X'].shape[1] == 512  # lookback
```

---

## DAY 2: DATA & MODEL SETUP

### TASK-008: Download Remaining Datasets
```
Priority: P1
Dependencies: TASK-004
Time: 2h (some manual)
```

**Action**: Download all remaining datasets:
1. PHM Milling - Manual from PHM Society
2. Wind SCADA - Kaggle CLI or manual
3. MIMII - Auto-download from Zenodo
4. PRONOSTIA - GitHub clone
5. PU Bearings - Manual request (optional, lower priority)

**Deliverables**:
- [ ] `data/raw/phm_milling/` populated
- [ ] `data/raw/wind_scada/` populated
- [ ] `data/raw/mimii/` populated
- [ ] `data/raw/pronostia/` populated

---

### TASK-009: Preprocess All Datasets
```
Priority: P1
Dependencies: TASK-007, TASK-008
Time: 2h
```

**Action**: Run `preprocess_all_datasets()` function

**Deliverables**:
- [ ] `data/processed/cmapss/{FD001,FD002,FD003,FD004}/processed_data.pt`
- [ ] `data/processed/phm_milling/processed_data.pt`
- [ ] `data/processed/wind_scada/processed_data.pt`
- [ ] `data/processed/mimii/processed_data.pt`

---

### TASK-010: Create Base Model Wrapper
```
Priority: P0
Dependencies: TASK-002
Time: 1h
```

**Action**: Create `src/models/base.py` with `BaseTSFMWrapper` abstract class

**Methods**:
- `load_model()` - Abstract
- `predict()` - Abstract
- `zero_shot()` - Implemented
- `few_shot_adapt()` - Abstract
- `predict_batch()` - Implemented

**Deliverables**:
- [ ] `src/models/base.py`

---

### TASK-011: Create MOMENT Wrapper
```
Priority: P0
Dependencies: TASK-010
Time: 1h
```

**Action**: Create `src/models/moment.py` with `MOMENTWrapper` class

**Deliverables**:
- [ ] `src/models/moment.py`

**Verification**:
```python
from src.models.moment import MOMENTWrapper
model = MOMENTWrapper()
model.load_model()
# Should print "MOMENT loaded on cuda"
```

---

### TASK-012: Create Chronos Wrapper
```
Priority: P0
Dependencies: TASK-010
Time: 1h
```

**Action**: Create `src/models/chronos.py` with `ChronosWrapper` class

**Deliverables**:
- [ ] `src/models/chronos.py`

---

### TASK-013: Create TimeGPT Wrapper
```
Priority: P1
Dependencies: TASK-010
Time: 0.5h
```

**Action**: Create `src/models/timegpt.py` with `TimeGPTWrapper` class

**Deliverables**:
- [ ] `src/models/timegpt.py`

---

### TASK-014: Create PatchTST Baseline
```
Priority: P0
Dependencies: TASK-010
Time: 1h
```

**Action**: Create `src/models/patchtst.py` with `PatchTSTWrapper` class

**Deliverables**:
- [ ] `src/models/patchtst.py`

---

### TASK-015: Create Model Factory
```
Priority: P0
Dependencies: TASK-011, TASK-012, TASK-013, TASK-014
Time: 0.5h
```

**Action**: Create `src/models/__init__.py` with:
- `MODEL_REGISTRY` dict
- `get_model()` factory function
- `list_models()` function

**Deliverables**:
- [ ] `src/models/__init__.py`

**Verification**:
```python
from src.models import get_model, list_models
print(list_models())  # ['moment', 'chronos', 'timegpt', 'patchtst']
model = get_model('moment')
```

---

## DAY 3: ZERO-SHOT EXPERIMENTS (PART 1)

### TASK-016: Create Metrics Module
```
Priority: P0
Dependencies: None
Time: 1h
```

**Action**: Create `src/evaluation/metrics.py` with:
- `mae()`, `mse()`, `rmse()`, `mape()`
- `crps()` for probabilistic forecasts
- `f1_at_threshold()`, `auc_roc()`, `auc_pr()`
- `concordance_index()`, `rul_score()`
- `compute_forecasting_metrics()`
- `compute_anomaly_metrics()`
- `compute_rul_metrics()`
- `compute_all_metrics()`

**Deliverables**:
- [ ] `src/evaluation/metrics.py`

---

### TASK-017: Create Zero-Shot Experiment Runner
```
Priority: P0
Dependencies: TASK-015, TASK-016
Time: 1h
```

**Action**: Create `src/experiments/run_zero_shot.py` with:
- `run_zero_shot_experiment()` - Single model-dataset
- `run_all_zero_shot()` - All combinations
- W&B logging integration
- Results saving to JSON/CSV

**Deliverables**:
- [ ] `src/experiments/run_zero_shot.py`

---

### TASK-018: Run MOMENT Zero-Shot
```
Priority: P0
Dependencies: TASK-009, TASK-017
Time: 2h (GPU)
```

**Action**:
```python
python -m src.experiments.run_zero_shot \
    --models moment \
    --datasets cmapss \
    --output results/zero_shot
```

**Deliverables**:
- [ ] `results/zero_shot/moment_cmapss_FD001.json`
- [ ] `results/zero_shot/moment_cmapss_FD002.json`
- [ ] `results/zero_shot/moment_cmapss_FD003.json`
- [ ] `results/zero_shot/moment_cmapss_FD004.json`

---

### TASK-019: Run Chronos Zero-Shot
```
Priority: P0
Dependencies: TASK-009, TASK-017
Time: 2h (GPU)
```

**Action**:
```python
python -m src.experiments.run_zero_shot \
    --models chronos \
    --datasets cmapss \
    --output results/zero_shot
```

**Deliverables**:
- [ ] `results/zero_shot/chronos_cmapss_*.json`

---

### TASK-020: Run PatchTST Baseline
```
Priority: P0
Dependencies: TASK-009, TASK-017
Time: 2h (GPU)
```

**Action**: Train and evaluate PatchTST
```python
python -m src.experiments.run_zero_shot \
    --models patchtst \
    --datasets cmapss \
    --output results/zero_shot
```

**Deliverables**:
- [ ] `results/zero_shot/patchtst_cmapss_*.json`

---

## DAY 4: ZERO-SHOT EXPERIMENTS (PART 2)

### TASK-021: Run TimeGPT Zero-Shot
```
Priority: P1
Dependencies: TASK-009, TASK-017
Time: 1h (API calls)
```

**Action**: Set NIXTLA_API_KEY and run
```bash
export NIXTLA_API_KEY="your_key"
python -m src.experiments.run_zero_shot \
    --models timegpt \
    --datasets cmapss \
    --output results/zero_shot
```

**Deliverables**:
- [ ] `results/zero_shot/timegpt_cmapss_*.json`

---

### TASK-022: Create Sundial Wrapper (Optional)
```
Priority: P2
Dependencies: TASK-010
Time: 1h
```

**Action**: Create `src/models/sundial.py`

---

### TASK-023: Create Time-MoE Wrapper (Optional)
```
Priority: P2
Dependencies: TASK-010
Time: 1h
```

**Action**: Create `src/models/time_moe.py`

---

### TASK-024: Create Lag-Llama Wrapper (Optional)
```
Priority: P2
Dependencies: TASK-010
Time: 1h
```

**Action**: Create `src/models/lag_llama.py`

---

### TASK-025: Run Remaining Zero-Shot Experiments
```
Priority: P1
Dependencies: TASK-021, TASK-022, TASK-023, TASK-024
Time: 3h (GPU)
```

**Action**: Run all models on all datasets
```python
python -m src.experiments.run_zero_shot \
    --models moment,chronos,timegpt,patchtst \
    --datasets cmapss,phm_milling,wind_scada,mimii \
    --output results/zero_shot
```

**Deliverables**:
- [ ] `results/zero_shot/zero_shot_results.csv` - Complete matrix

---

### TASK-026: Write Related Work Section
```
Priority: P1
Dependencies: None (can run parallel to experiments)
Time: 2h
```

**Action**: Create `paper/sections/related_work.tex`

**Content**:
1. TSFM Landscape (MOMENT, Sundial, Chronos, etc.)
2. PdM Benchmarks (C-MAPSS, PHM, PRONOSTIA)
3. Gap identification
4. Table 1: TSFM Capabilities vs PdM Requirements

**Deliverables**:
- [ ] `paper/sections/related_work.tex`

---

### TASK-027: Write Methodology Section
```
Priority: P1
Dependencies: None
Time: 2h
```

**Action**: Create `paper/sections/methodology.tex`

**Content**:
1. Section 3.1: Datasets (Table 2)
2. Section 3.2: Models
3. Section 3.3: Preprocessing Pipeline (Figure 2)
4. Section 3.4: Evaluation Protocol

**Deliverables**:
- [ ] `paper/sections/methodology.tex`

---

## DAY 5: ADVANCED EXPERIMENTS

### TASK-028: Create Few-Shot Experiment Runner
```
Priority: P1
Dependencies: TASK-017
Time: 1h
```

**Action**: Create `src/experiments/run_few_shot.py` with LoRA adaptation

**Deliverables**:
- [ ] `src/experiments/run_few_shot.py`

---

### TASK-029: Run Few-Shot Experiments
```
Priority: P1
Dependencies: TASK-028
Time: 3h (GPU)
```

**Action**:
```python
python -m src.experiments.run_few_shot \
    --models moment,sundial \
    --datasets cmapss \
    --train_ratio 0.01 \
    --lora_r 16 \
    --epochs 10 \
    --output results/few_shot
```

**Deliverables**:
- [ ] `results/few_shot/few_shot_results.csv`

---

### TASK-030: Create Cross-Domain Experiment Runner
```
Priority: P1
Dependencies: TASK-017
Time: 1h
```

**Action**: Create `src/experiments/run_cross_domain.py`
- Train on dataset A, test on dataset B
- Generate transfer matrix

**Deliverables**:
- [ ] `src/experiments/run_cross_domain.py`

---

### TASK-031: Run Cross-Domain Experiments
```
Priority: P1
Dependencies: TASK-030
Time: 2h (GPU)
```

**Action**:
```python
python -m src.experiments.run_cross_domain \
    --models moment,patchtst \
    --datasets cmapss,phm_milling,wind_scada \
    --output results/cross_domain
```

**Deliverables**:
- [ ] `results/cross_domain/transfer_matrix.csv`

---

### TASK-032: Write Introduction Section
```
Priority: P0
Dependencies: None
Time: 1.5h
```

**Action**: Create `paper/sections/introduction.tex`

**Content**:
1. Problem statement ($50B market)
2. TSFM promises vs reality
3. Industrial challenges (non-IID, privacy, etc.)
4. Contributions list

**Deliverables**:
- [ ] `paper/sections/introduction.tex`

---

## DAY 6: ANALYSIS & VISUALIZATION

### TASK-033: Create Figures Module
```
Priority: P0
Dependencies: TASK-025
Time: 1.5h
```

**Action**: Create `src/visualization/figures.py` with:
- `create_heatmap()` - Model vs Dataset MAE
- `create_bar_comparison()` - Grouped bars
- `create_radar_chart()` - Multi-dimensional
- `create_transfer_matrix()` - Cross-domain
- `create_failure_taxonomy_figure()` - Radar chart

**Deliverables**:
- [ ] `src/visualization/figures.py`

---

### TASK-034: Generate Figure 1 - Performance Heatmap
```
Priority: P0
Dependencies: TASK-033, TASK-025
Time: 0.5h
```

**Action**:
```python
from src.visualization.figures import create_heatmap
import pandas as pd

results = pd.read_csv("results/zero_shot/zero_shot_results.csv")
pivot = results.pivot_table(values='mae', index='model', columns='dataset')
create_heatmap(pivot, output_path="results/figures/fig1_heatmap.png")
```

**Deliverables**:
- [ ] `results/figures/fig1_heatmap.png`
- [ ] `paper/figures/fig1_heatmap.pdf`

---

### TASK-035: Generate Figure 3 - Transfer Matrix
```
Priority: P0
Dependencies: TASK-033, TASK-031
Time: 0.5h
```

**Action**: Create cross-domain transfer heatmap

**Deliverables**:
- [ ] `results/figures/fig3_transfer_matrix.png`

---

### TASK-036: Generate Figure 4 - Failure Taxonomy
```
Priority: P0
Dependencies: TASK-033
Time: 0.5h
```

**Action**: Create failure modes radar chart

**Deliverables**:
- [ ] `results/figures/fig4_failure_taxonomy.png`

---

### TASK-037: Create Tables Module
```
Priority: P0
Dependencies: TASK-025
Time: 1h
```

**Action**: Create `src/visualization/tables.py` with:
- `create_latex_table()` - Generic
- `create_zero_shot_table()` - Table 3
- `create_few_shot_table()` - Table 4
- `create_rul_table()` - Table 5

**Deliverables**:
- [ ] `src/visualization/tables.py`

---

### TASK-038: Generate All LaTeX Tables
```
Priority: P0
Dependencies: TASK-037
Time: 1h
```

**Action**: Generate Tables 1-5

**Deliverables**:
- [ ] `paper/tables/table1_capabilities.tex`
- [ ] `paper/tables/table2_datasets.tex`
- [ ] `paper/tables/table3_zero_shot.tex`
- [ ] `paper/tables/table4_few_shot.tex`
- [ ] `paper/tables/table5_rul.tex`

---

### TASK-039: Write Experiments Section
```
Priority: P0
Dependencies: TASK-025, TASK-029, TASK-031
Time: 2h
```

**Action**: Create `paper/sections/experiments.tex`

**Content**:
1. Section 4.1: Zero-Shot Performance (Table 3)
2. Section 4.2: Few-Shot Adaptation
3. Section 4.3: RUL Regression Failures (Table 5)
4. Section 4.4: Cross-Domain Transfer (Figure 3)
5. Section 4.5: Ablation - Preprocessing Impact

**Deliverables**:
- [ ] `paper/sections/experiments.tex`

---

### TASK-040: Write Analysis Section
```
Priority: P0
Dependencies: TASK-039
Time: 1.5h
```

**Action**: Create `paper/sections/analysis.tex`

**Content**:
1. Taxonomy of Failures
2. Distribution Shift analysis
3. Long-horizon drift
4. Privacy leakage concerns
5. Federation unreadiness
6. Industrial Checklist

**Deliverables**:
- [ ] `paper/sections/analysis.tex`

---

### TASK-041: Write Abstract
```
Priority: P0
Dependencies: TASK-039, TASK-040
Time: 0.5h
```

**Action**: Create `paper/sections/abstract.tex`
- 150 words max
- Key findings: 35% MAE gap, 62% RUL failure, 18% preprocessing boost

**Deliverables**:
- [ ] `paper/sections/abstract.tex`

---

## DAY 7: FINALIZATION & SUBMISSION

### TASK-042: Write Discussion Section
```
Priority: P0
Dependencies: TASK-040
Time: 1h
```

**Action**: Create `paper/sections/discussion.tex`

**Content**:
1. Limitations
2. Future work (federated direction)
3. Impact statement

**Deliverables**:
- [ ] `paper/sections/discussion.tex`

---

### TASK-043: Write Conclusion Section
```
Priority: P0
Dependencies: TASK-042
Time: 0.5h
```

**Action**: Create `paper/sections/conclusion.tex`

**Deliverables**:
- [ ] `paper/sections/conclusion.tex`

---

### TASK-044: Create Bibliography
```
Priority: P0
Dependencies: None
Time: 1h
```

**Action**: Create `paper/references.bib` with 25-30 references

**Key references**:
- MOMENT paper (arXiv:2402.03885)
- Chronos paper (arXiv:2403.07815)
- TimeGPT paper (arXiv:2310.03589)
- FoundTS benchmark (arXiv:2410.11802)
- C-MAPSS dataset
- PatchTST (ICLR 2023)

**Deliverables**:
- [ ] `paper/references.bib`

---

### TASK-045: Create Main LaTeX File
```
Priority: P0
Dependencies: TASK-041 through TASK-044
Time: 0.5h
```

**Action**: Create `paper/main.tex` that includes all sections

**Deliverables**:
- [ ] `paper/main.tex`

---

### TASK-046: Compile LaTeX Document
```
Priority: P0
Dependencies: TASK-045
Time: 0.5h
```

**Action**:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Deliverables**:
- [ ] `paper/main.pdf`

---

### TASK-047: Page Limit Check
```
Priority: P0
Dependencies: TASK-046
Time: 0.5h
```

**Action**: Verify paper is 8-10 pages

**If over limit**:
- Reduce figure sizes
- Condense tables
- Move details to appendix

**If under limit**:
- Add more analysis
- Expand discussion

---

### TASK-048: Final Proofread
```
Priority: P0
Dependencies: TASK-047
Time: 1h
```

**Checklist**:
- [ ] No spelling errors
- [ ] All figures referenced in text
- [ ] All tables referenced in text
- [ ] All citations present
- [ ] Abstract within word limit
- [ ] Author information correct
- [ ] Acknowledgments if needed

---

### TASK-049: Create GitHub Repository
```
Priority: P1
Dependencies: All code tasks
Time: 0.5h
```

**Action**:
```bash
git init
git add .
git commit -m "Initial commit: TSFM Industrial PdM Benchmark"
git remote add origin https://github.com/yassire/tsfm-industrial-bench.git
git push -u origin main
```

**Add**:
- README.md with instructions
- LICENSE (MIT)
- .gitignore

**Deliverables**:
- [ ] GitHub repository created and pushed

---

### TASK-050: Submit Paper
```
Priority: P0
Dependencies: TASK-048
Time: 0.5h
```

**Action**:
1. Convert PDF to required format if needed
2. Fill conference submission form
3. Upload paper
4. Verify submission confirmation

**Deliverables**:
- [ ] Submission confirmation email/screenshot

---

# PRIORITY MATRIX

## P0 - Must Complete (Paper won't compile without these)
- TASK-001 to TASK-007 (Setup)
- TASK-010 to TASK-015 (Core models)
- TASK-016 to TASK-020 (Core experiments)
- TASK-032 to TASK-050 (Writing & submission)

## P1 - Should Complete (Paper quality depends on these)
- TASK-008, TASK-009 (Additional datasets)
- TASK-021 (TimeGPT)
- TASK-026, TASK-027 (Sections)
- TASK-028 to TASK-031 (Advanced experiments)
- TASK-049 (GitHub)

## P2 - Nice to Have (Strengthens paper)
- TASK-022 to TASK-024 (Additional models)
- More datasets beyond minimum

## P3 - Optional
- Additional ablations
- Extended analysis

---

# FALLBACK STRATEGY

If running behind schedule:

**Minimum Viable Paper (MVP)**:
- 3 models: MOMENT, Chronos, PatchTST
- 2 datasets: C-MAPSS, Wind SCADA
- 1 scenario: Zero-shot only
- Estimated: 5 pages of results

This still provides:
- Novel industrial TSFM evaluation
- Concrete performance numbers
- Gap identification for journal paper

---

# EXECUTION NOTES FOR AI AGENT

1. **Parallel Execution**: Tasks with no dependencies can run simultaneously
2. **Checkpoint Frequently**: Save results after each experiment
3. **Error Recovery**: If a model fails, skip and continue with others
4. **Memory Management**: Clear GPU memory between models
5. **API Limits**: TimeGPT has rate limits - add delays if needed
6. **Verification**: Always verify task completion before marking done

---

*End of AI Agent Task Plan*
