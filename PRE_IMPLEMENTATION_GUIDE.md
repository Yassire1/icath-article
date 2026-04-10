# Pre-Implementation Guide: TSFM Industrial PdM Benchmark

This document explains the full project in practical terms and lists what you should understand and verify before running implementation and experiments.

## 1) Project Intent and Boundaries

The project benchmarks Time-Series Foundation Models (TSFMs) for industrial predictive maintenance (PdM), with three evaluation scenarios:

- Zero-shot inference
- Few-shot adaptation (LoRA)
- Cross-domain transfer

The objective is not to create a new TSFM architecture, but to evaluate deployment readiness on industrial data conditions (noise, missing values, non-IID behavior, drift, and domain mismatch).

## 2) Core Components (Names, Nature, Characteristics)

| Component | Location | Nature | Key Characteristics |
|---|---|---|---|
| Configuration Layer | `config/` | Declarative control | Centralizes preprocessing, experiment, hardware, and model metadata |
| Data Download Layer | `src/data/download.py` | Acquisition utility | Handles mixed auto/manual download workflows for 6 datasets |
| Preprocessing Layer | `src/data/preprocessing.py` | Data engineering pipeline | Chronological split, imputation, normalization, sequence generation, RUL label creation |
| Model Abstraction Layer | `src/models/base.py` | Interface contract | Unified API: `load_model`, `predict`, `few_shot_adapt`, `zero_shot` |
| Model Wrappers | `src/models/*.py` | Adapter pattern | Harmonizes model-specific I/O differences under one benchmark interface |
| Experiment Runners | `src/experiments/*.py` | Orchestration | Scenario-specific loops, metrics computation, result logging |
| Metrics Layer | `src/evaluation/metrics.py` | Evaluation core | Forecasting, anomaly, and RUL metrics with task-conditioned dispatch |
| Visualization Layer | `src/visualization/*.py` | Reporting utilities | Generates figures and LaTeX tables from experiment outputs |
| Notebook Execution Layer | `notebooks/01..07` | Human-in-the-loop workflow | Stepwise Colab execution from setup to analysis |
| Paper Layer | `paper/` | Publication output | LaTeX sections/tables that consume experiment outputs |

## 3) Models and Technical Specifications

### 3.1 Active Open-Source Models in Code Path

| Model | Wrapper | Params | Context | Native Type | Zero-shot | Few-shot |
|---|---|---:|---:|---|---|---|
| MOMENT | `src/models/moment.py` | 385M | 512 | Foundation TSFM | Yes | Yes (LoRA) |
| Chronos (T5 Large) | `src/models/chronos.py` | 710M | 512 | Probabilistic TSFM | Yes | Not practically supported in current wrapper |
| Lag-Llama | `src/models/lag_llama.py` | 7M | 32 | Probabilistic TSFM | Yes | Yes (LoRA-style adaptation path) |
| PatchTST (baseline) | `src/models/patchtst.py` | small (task-dependent) | 512 input / 96 horizon | Supervised baseline | No true zero-shot (must be fit) | Yes (via fit) |

### 3.2 Optional / Partial Models

- TimeGPT wrapper exists: `src/models/timegpt.py` (API/key-based, not wired in current model registry)
- Sundial and Time-MoE are referenced in docs/notebooks, but no active wrappers in current `src/models/__init__.py`

### 3.3 "Llama" Clarification

In this project, "Llama" refers to **Lag-Llama** (time-series foundation model), not a general LLM for text.

## 4) System Architecture and Setup

### 4.1 Logical Architecture

```text
Raw Data -> Preprocessing -> Processed Tensors -> Scenario Runners -> Metrics -> Results CSV/JSON -> Figures/Tables -> Paper

config/*.yaml controls each stage
```

### 4.2 Directory Contract

- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Scenario results: `results/zero_shot/`, `results/few_shot/`, `results/cross_domain/`
- Publication artifacts: `results/figures/`, `paper/tables/`, `paper/sections/`

### 4.3 Environment Setup Paths

Choose one:

1. Conda (`environment.yml`)
2. Pip (`requirements.txt`)
3. Scripted setup (`scripts/setup.sh`)
4. Colab notebook bootstrap (`notebooks/01_setup_environment.ipynb`)

### 4.4 Setup Knowledge You Need

- CUDA basics (GPU memory limits, device selection)
- PyTorch tensor shapes `(batch, seq_len, channels)`
- HuggingFace model pull mechanics
- Colab file persistence and Drive mounting

## 5) Data Processing and Conditional Techniques

### 5.1 Pipeline Steps

Implemented in `SCADAPreprocessor` (`src/data/preprocessing.py`):

1. Missing-value imputation (`ffill`/`bfill` + interpolation)
2. Chronological split (no shuffling)
3. Train-only fit normalization, val/test transform
4. Sliding window sequence construction
5. Optional RUL label generation (piecewise linear cap)

### 5.2 Conditional Logic Used in Pipeline

- If `targets is None`: forecasting targets are future windows
- Else: target is aligned scalar/label path
- If timestamp column exists: it is dropped before numeric conversion
- If model is univariate (Chronos/Lag-Llama): multivariate channels are collapsed (mean) then re-expanded for unified output shape

### 5.3 Why These Conditions Matter

- Prevents leakage (chronological split)
- Preserves compatibility across mixed model capabilities (uni vs multi-variate)
- Keeps evaluation API consistent despite heterogeneous model internals

## 6) Experiment Design (How You Will Execute)

### 6.1 Zero-Shot (`src/experiments/run_zero_shot.py`)

- Load processed test tensors
- Load model wrapper via factory
- Predict without adaptation
- Compute task-specific metrics
- Save per-run JSON + aggregate CSV

### 6.2 Few-Shot (`src/experiments/run_few_shot.py`)

- Subsample training data (default ratio, minimum sample floor)
- Apply adaptation (`few_shot_adapt`)
- Predict on held-out test set
- Compare against zero-shot baseline

### 6.3 Cross-Domain (`src/experiments/run_cross_domain.py`)

- Define source-target dataset pairs
- Evaluate transfer performance (A -> B)
- Build transfer matrix and export summary files

## 7) Metrics and Evaluation Concepts You Must Know

### 7.1 Forecasting

- MAE, MSE, RMSE, MAPE, optional CRPS

### 7.2 Anomaly Detection

- F1, Precision, Recall, AUC-ROC, AUC-PR

### 7.3 RUL

- MAE, RMSE, Concordance Index, NASA RUL score

### 7.4 Important Metric Caveat

Metric interpretation must always be tied to task and data regime. A better MAE on one subset does not imply robust transfer performance across domains.

## 8) Optimization Techniques (Current + Allowed Extensions)

### 8.1 Already Used

- LoRA for parameter-efficient adaptation (`moment.py`, `lag_llama.py`)
- Vectorized C-index computation (`metrics.py`)
- Device fallback (`cuda` if available, else `cpu`)
- Shape alignment guard in runners (truncate to minimum horizon)

### 8.2 Safe Next Optimizations (If Needed)

- Mixed precision (`fp16/bf16`) where numerically stable
- Gradient accumulation to fit larger effective batch size
- Smaller lookback/horizon for quick ablation and sanity checks
- Cached model weights and preprocessed tensors to avoid repeated overhead

### 8.3 Optimization Guardrails

- Never optimize by changing evaluation semantics silently
- Any optimization affecting accuracy must be documented and compared to baseline
- Compute-time improvements are valid only if metric behavior remains explainable

## 9) New Actions That Require Explicit Justification Before Use

Before applying any of the following, write a short justification note (what, why, risk, rollback):

1. Changing dataset splits (especially introducing random shuffle)
2. Changing lookback/horizon defaults
3. Replacing baseline or adding/removing core models
4. Injecting synthetic data augmentation
5. Applying task-specific post-processing not applied uniformly across models
6. Altering metric definitions or thresholds post hoc
7. Including closed API models in "reproducible" claims

Use this template:

```text
Action:
Reason:
Expected Benefit:
Risk/Trade-off:
Validation Plan:
Rollback Plan:
```

## 10) Pre-Implementation Checklist (Concept + Repo Readiness)

### 10.1 Conceptual Readiness

- I can explain zero-shot vs few-shot vs cross-domain in one paragraph each
- I understand uni vs multivariate model handling implications
- I know which metrics map to which task and why
- I can justify why chronological splitting is mandatory in PdM

### 10.2 Technical Readiness

- Environment resolves (`torch`, `transformers`, `peft`, `neuralforecast`, `chronos`)
- `data/raw/` contains expected files for selected datasets
- `data/processed/` tensors can be loaded without shape errors
- At least one smoke test run completes end-to-end

### 10.3 Consistency Checks to Perform Before Full Run

- Align model scope between `config/models.yaml`, `src/models/__init__.py`, and notebooks
- Align dataset scope between `config/datasets.yaml` and notebooks
- Confirm whether optional models (TimeGPT/Sundial/Time-MoE) are in scope or explicitly excluded
- Ensure paper placeholders are not treated as final findings

## 11) Recommended Learning Sequence (Before Heavy Runs)

1. Read `src/models/base.py` (interface contract)
2. Read `src/data/preprocessing.py` (data assumptions)
3. Read `src/evaluation/metrics.py` (what "good" means)
4. Run a small zero-shot smoke test on one dataset/model
5. Run a tiny few-shot adaptation experiment
6. Only then launch full benchmark matrix

## 12) What Success Looks Like for This Project

- Reproducible experiment outputs (CSV/JSON)
- Clear tables/figures tied to metrics and scenario definitions
- Claims in paper that are strictly supported by produced results
- Transparent limitations and justified methodological choices
