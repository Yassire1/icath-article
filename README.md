# TSFM Industrial PdM Benchmark

Benchmarking Time-Series Foundation Models for Industrial Predictive Maintenance: Critical Limitations Exposed

## Overview

This repository contains the code and data for evaluating Time Series Foundation Models (TSFMs) on industrial Predictive Maintenance (PdM) datasets. We benchmark 6 TSFMs across 6 industrial datasets, exposing fundamental gaps between benchmark performance and industrial applicability.

## Key Findings

- TSFMs degrade 35% MAE on non-IID industrial data vs. benchmarks
- Zero-shot RUL fails 62% due to distribution shifts
- SCADA preprocessing boosts cross-domain transfer 18%
- No evaluated TSFM satisfies federated readiness checklist

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate tsfm-bench

# Or use pip
pip install -r requirements.txt
```

## Quick Start

```bash
# Download datasets
python -m src.data.download

# Preprocess
python -m src.data.preprocessing

# Run zero-shot experiments
python -m src.experiments.run_zero_shot --models moment,chronos,patchtst --datasets cmapss

# Run few-shot experiments
python -m src.experiments.run_few_shot --models moment,sundial --datasets cmapss

# Run cross-domain experiments
python -m src.experiments.run_cross_domain --models moment,patchtst --datasets cmapss,phm_milling
```

## Datasets

| Dataset | Domain | Source |
|---------|--------|--------|
| C-MAPSS | Turbofan | NASA |
| PHM Milling | CNC | PHM Society |
| PU Bearings | Rotating | Paderborn Univ. |
| Wind SCADA | Turbines | Kaggle |
| MIMII | Factory | Zenodo |
| PRONOSTIA | Bearings | FEMTO-ST |

## Models

- MOMENT (AutonLab)
- Sundial (Salesforce)
- Chronos (Amazon)
- Time-MoE
- Lag-Llama
- TimeGPT (Nixtla API)
- PatchTST (baseline)
- Autoformer (baseline)

## Citation

If you use this benchmark, please cite:

```bibtex
@article{ammouri2026tsfm,
  title={Benchmarking Time-Series Foundation Models for Industrial Predictive Maintenance: Critical Limitations Exposed},
  author={Ammouri, Yassire},
  journal={ICATH Conference},
  year={2026}
}
```

## License

MIT License
