# TSFM Industrial PdM Benchmark

Benchmarking Time-Series Foundation Models for Industrial Predictive Maintenance: Critical Limitations Exposed

## Overview

This repository contains the code and paper assets for evaluating Time Series Foundation Models (TSFMs) on industrial Predictive Maintenance (PdM) datasets. The current runnable ICATH scope benchmarks 4 models across 3 datasets: C-MAPSS, Wind SCADA, and PHM Milling.

## Current Scope

- Zero-shot evaluation is the primary validated benchmark path.
- Cross-condition transfer on C-MAPSS is included as a core robustness study.
- PHM Milling replaces MIMII in the Colab-friendly pipeline to keep preprocessing memory-safe.
- Few-shot adapters for MOMENT and Lag-Llama remain exploratory and should not be reported as validated results until real parameter updates are implemented.

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
python scripts/step_01_download.py

# Preprocess
python scripts/step_02_preprocess.py

# Run zero-shot experiments
python scripts/step_03_zero_shot.py

# Run exploratory few-shot experiments
python scripts/step_04_few_shot.py

# Run cross-domain experiments
python scripts/step_05_cross_condition.py
```

## Datasets

| Dataset | Domain | Source |
|---------|--------|--------|
| C-MAPSS | Turbofan | NASA |
| Wind SCADA | Turbines | Kaggle |
| PHM Milling | CNC Tool Wear | PHM Society |

Inactive archival dataset entries remain in the repository configuration, but the active Colab-friendly benchmark path uses the three datasets above.

## Models

- MOMENT (AutonLab)
- Chronos (Amazon)
- Lag-Llama
- PatchTST (baseline)

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
