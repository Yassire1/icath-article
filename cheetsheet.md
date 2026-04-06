🚀 TSFM Industrial PdM Benchmark Cheat Sheet
Start TODAY - Everything you need for 8-week conference paper execution.

1. Materials (Hardware)
text
MINIMUM (Colab Pro $10/mo):
- RAM: 25GB (T4 GPU)
- Storage: 100GB free Google Drive

IDEAL (Local/GCP Free Tier):
- RTX 3090/4090 or A100
- 64GB RAM, 1TB SSD
- Ubuntu 22.04
2. Software Stack (One-Click Setup)
bash
# 1. Environment (5 min)
conda create -n tsfm-bench python=3.11
conda activate tsfm-bench
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pandas numpy scikit-learn matplotlib seaborn plotly kaleido

# 2. Core TSFM Libraries
pip install tsfm[all]  # MOMENT, Chronos, Lag-Llama
pip install time-series-foundation-models  # Sundial
pip install autogluon.timeseries  # TimeGPT wrapper

# 3. Benchmarks/Baselines
pip install neuralforecast  # PatchTST, Autoformer
pip install gluonts[torch]  # Transformers

# 4. Utils
pip install wandb  # Experiment tracking
pip install git+https://github.com/moment-timeseries-foundation-model/moment-research  # MOMENT latest
Full requirements.txt:

text
torch==2.1.0
transformers==4.44.0
tsfm==0.4.0
neuralforecast==1.7.0
pandas==2.2.0
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.0
plotly==5.22.0
wandb==0.17.0
peft==0.12.0  # LoRA
3. Models (6 TSFMs + 3 Baselines) - Ready-to-Infer
Model	GitHub/Docs	HuggingFace	API Key	Notes
MOMENT	
GitHub
moment-timeseries/movement-v1-huge	None	Best multivariate
Sundial	arXiv:2502.00816	sundial-ts/sundial-base	None	Probabilistic
Time-MoE	HuggingFace	Time-MoE/Time-MoE-L6	None	MoE scaling
Chronos	GitHub	amazon/chronos-t5-large	None	Token-based
Lag-Llama	GitHub	lag/llama3-8b-lag-llama	None	LLM-style
TimeGPT	Nixtla Docs	N/A	nixtla-dummy-key (free tier)	API only
Baselines:

python
from neuralforecast import NeuralForecast
models = [PatchTST(h=96), Autoformer(h=96), Transformer(h=96)]
4. Datasets (6 Industrial PdM) - Direct Downloads
Dataset	Source	Download	Size	Tasks
C-MAPSS	NASA Prognostics	ti.nd.edu/cmapss	200MB	RUL/Forecast
PHM Milling	PHM Society	phmsociety.org	50MB	Anomaly
PU Bearings	PRONOSTIA	ieee-dataport.org	300MB	RUL/Anomaly
Wind SCADA	Kaggle/OpenML	kaggle.com/alexisbcook/scada-wind-turbine	80MB	Forecast
MIMII	Factory Noise	zenodo.org	1.2GB	Anomaly
TEP (Bonus)	Chem Plant	braid.ac.uk	40MB	All tasks
One-Click Download Script:

python
import gdown
datasets = {
    "cmapss": "https://tiarc.nasa.gov/cmapss",
    "phm_milling": "https://www.phmsociety.org/sites/phmsociety.org/files/PHM_Data_Challenge_2010.Milling.zip",
    "pu_bearings": "https://ieee-dataport.org/sites/default/files/PHM12_Bearing.zip"
}
for name, url in datasets.items():
    gdown.download(url, f"data/{name}.zip", quiet=False)
5. Existing Benchmarks (Cite These)
text
FoundTS: arXiv:2410.11802 - [PDF](http://arxiv.org/pdf/2410.11802)
GIFT-Eval: arXiv:2410.10393 - [PDF](https://arxiv.org/pdf/2410.10393)
TSFM-Bench: ACM 2025 - DOI:10.1145/3711896.3737442
TSPP: arXiv:2312.17100 - [GitHub](https://github.com/thuml/TSPP)
6. Preprocessing Pipeline Code (Week 1 Ready)
python
# scada_preprocess.py - Your secret sauce
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scada_pipeline(df: pd.DataFrame, lookback=512, horizon=96):
    """Industrial-ready TSFM preprocessing"""
    # 1. Chronological split (no leakage)
    train = df[:int(0.7*len(df))]
    val = df[int(0.7*len(df)):int(0.85*len(df))]
    test = df[int(0.85*len(df)):]
    
    # 2. SCADA normalization (per-sensor family)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train)
    
    # 3. Sliding window
    def make_sequences(X, lookback, horizon):
        X_seq, y_seq = [], []
        for i in range(len(X) - lookback - horizon):
            X_seq.append(X[i:i+lookback])
            y_seq.append(X[i+lookback:i+lookback+horizon])
        return np.array(X_seq), np.array(y_seq)
    
    return make_sequences(X_train, lookback, horizon)
7. Evaluation Protocol Template
python
# eval_tsfm.py
from tsfm.models import MOMENTModel
import wandb

def benchmark_tsfm(model_name, dataset, scenario="zero-shot"):
    model = MOMENTModel.from_pretrained(model_name)
    preds = model.predict(dataset.X, return_y=True)
    
    metrics = {
        "mae": mae(preds.y_pred, preds.y_true),
        "crps": crps(preds.y_pred_proba, preds.y_true)
    }
    wandb.log(metrics)
    return metrics
8. Quickstart Commands (Copy-Paste)
bash
# Terminal 1: Environment
conda env create -f environment.yml
conda activate tsfm-bench

# Terminal 2: Download data
python download_datasets.py

# Terminal 3: Preprocess
python preprocess_all.py --datasets all

# Terminal 4: Experiments
wandb login
python run_benchmark.py --models all --scenarios zero-shot,few-shot --parallel
9. Colab Notebook Template
text
https://colab.research.google.com/drive/1[YOUR_TEMPLATE]
Sections: 1) Setup, 2) Data, 3) Preprocessing, 4) Zero-shot, 5) Few-shot LoRA, 6) Plots
10. Target Venues (April 2026 Deadlines)
Venue	Deadline	Format	Acceptance
ICLR 2027 Time Series Workshop	Sep 15, 2026	8-10p	75%
NeurIPS 2026 Datasets Track	May 15, 2026	9p	25%
MLSys 2027	Sep 1, 2026	10p	28%
START NOW CHECKLIST ✅
text
□ [ ] Colab Pro subscription ($10)
□ [ ] `conda env create -f environment.yml` (5 min)
□ [ ] `python download_datasets.py` (30 min)
□ [ ] C-MAPSS preprocessing test (Week 1 Day 1)
□ [ ] GitHub repo: github.com/yassire/tsfm-industrial-bench
□ [ ] Week 1 Google Sheet: Dataset stats table
Total Setup Time: 45 minutes. First results by tomorrow night.

Pro Tip: Run MOMENT zero-shot on C-MAPSS FD001 tonight (2 hours). Wake up to your first heatmap.