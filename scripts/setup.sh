#!/bin/bash
# Full environment setup script

set -e

echo "=========================================="
echo "TSFM Industrial PdM Benchmark Setup"
echo "=========================================="

if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "Creating conda environment..."
conda env create -f environment.yml -y || conda env update -f environment.yml

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tsfm-bench

echo "Installing additional packages..."
pip install momentfm --no-deps || true
pip install git+https://github.com/amazon-science/chronos-forecasting.git || true

echo "Creating directory structure..."
mkdir -p data/raw/{cmapss,phm_milling,pu_bearings,wind_scada,mimii,pronostia}
mkdir -p data/processed/{cmapss,phm_milling,pu_bearings,wind_scada,mimii,pronostia}
mkdir -p results/{zero_shot,few_shot,cross_domain,tables,figures}
mkdir -p paper/{sections,figures,tables}
mkdir -p src/{data,models,evaluation,experiments,visualization}
mkdir -p notebooks
mkdir -p config

touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/evaluation/__init__.py
touch src/experiments/__init__.py
touch src/visualization/__init__.py

echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo "=========================================="
echo "Setup complete!"
echo "Activate: conda activate tsfm-bench"
echo "=========================================="
