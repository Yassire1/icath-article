"""
Table generation utilities for paper
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def create_latex_table(
    data: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = "%.3f",
    output_path: Optional[Path] = None
) -> str:
    """Create LaTeX table from DataFrame"""
    latex = data.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        index=True
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def create_zero_shot_table(
    results: Dict[str, Dict[str, float]],
    metric: str = "mae",
    output_path: Optional[Path] = None
) -> str:
    """Create Table 3: Zero-shot MAE results"""
    df = pd.DataFrame(results).T
    df.columns = [col.replace('/', '_') for col in df.columns]

    latex = df.to_latex(
        float_format="%.3f",
        caption="Zero-Shot MAE Results (lower is better)",
        label="tab:zero_shot_mae"
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def create_few_shot_table(
    zero_shot: Dict[str, float],
    few_shot: Dict[str, float],
    supervised: Dict[str, float],
    output_path: Optional[Path] = None
) -> str:
    """Create Table 4: Few-shot vs supervised comparison"""
    data = {
        'Model': list(zero_shot.keys()),
        'Zero-Shot': list(zero_shot.values()),
        'Few-Shot (1\\%)': list(few_shot.values()),
        'Supervised': list(supervised.values())
    }
    df = pd.DataFrame(data).set_index('Model')

    latex = df.to_latex(
        float_format="%.3f",
        caption="Few-Shot vs. Supervised MAE Comparison",
        label="tab:few_shot"
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def create_rul_table(
    zero_shot: Dict[str, float],
    few_shot: Dict[str, float],
    supervised: Dict[str, float],
    output_path: Optional[Path] = None
) -> str:
    """Create Table 5: RUL prediction C-Index"""
    data = {
        'Model': list(zero_shot.keys()),
        'Zero-Shot': list(zero_shot.values()),
        'Few-Shot': list(few_shot.values()),
        'Supervised': list(supervised.values())
    }
    df = pd.DataFrame(data).set_index('Model')

    latex = df.to_latex(
        float_format="%.3f",
        caption="RUL Prediction C-Index (higher is better)",
        label="tab:rul"
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def create_capabilities_table(output_path: Optional[Path] = None) -> str:
    """Create Table 1: TSFM Capabilities vs PdM Requirements"""
    latex = r"""
\begin{table}[htbp]
\caption{TSFM Capabilities vs. PdM Requirements}
\label{tab:capabilities}
\centering
\begin{tabular}{lcccc}
\toprule
Model & Multivariate & Probabilistic & Long Context & Industrial Tested \\
\midrule
MOMENT & \checkmark & \checkmark & 512 & No \\
Sundial & \checkmark & \checkmark & 1024 & No \\
Chronos & \checkmark & \checkmark & 512 & No \\
Time-MoE & \checkmark & \checkmark & 512 & No \\
Lag-Llama & \checkmark & \checkmark & 32 & No \\
TimeGPT & \checkmark & \checkmark & 1024 & No \\
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def create_datasets_table(output_path: Optional[Path] = None) -> str:
    """Create Table 2: Dataset Summary"""
    latex = r"""
\begin{table}[htbp]
\caption{Industrial PdM Dataset Summary}
\label{tab:datasets}
\centering
\begin{tabular}{lccccc}
\toprule
Dataset & Domain & Samples & Channels & Tasks & Failure Rate \\
\midrule
C-MAPSS & Turbofan & 100 & 21 & RUL/Forecast & 20\% \\
PHM Milling & CNC & 16k & 8 & Anomaly & 15\% \\
PU Bearings & Rotating & 32 & 4 & RUL/Anomaly & 32\% \\
Wind SCADA & Turbines & 6k & 52 & Forecast & 8\% \\
MIMII & Factory & 10k & 8 & Anomaly & 10\% \\
PRONOSTIA & Bearings & 17 & 3 & RUL & 100\% \\
\bottomrule
\end{tabular}
\end{table}
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


def generate_all_tables(output_dir: Path) -> None:
    """Generate all paper tables"""
    output_dir.mkdir(parents=True, exist_ok=True)

    create_capabilities_table(output_dir / "table1_capabilities.tex")
    create_datasets_table(output_dir / "table2_datasets.tex")

    print(f"Tables saved to {output_dir}")


if __name__ == "__main__":
    generate_all_tables(Path("paper/tables"))
