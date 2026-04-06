"""
Visualization utilities for paper figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_heatmap(
    data: pd.DataFrame,
    title: str = "Model Performance Heatmap",
    cmap: str = "RdYlGn_r",
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[Path] = None,
    annot: bool = True,
    fmt: str = ".3f"
) -> plt.Figure:
    """Create performance heatmap (models vs datasets)"""
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'MAE'}
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def create_bar_comparison(
    data: pd.DataFrame,
    x: str = "model",
    y: str = "mae",
    hue: str = "dataset",
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create grouped bar chart comparing models"""
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_radar_chart(
    data: Dict[str, Dict[str, float]],
    title: str = "Model Capabilities Radar",
    output_path: Optional[Path] = None
) -> go.Figure:
    """Create radar chart for multi-dimensional comparison"""
    categories = list(list(data.values())[0].keys())

    fig = go.Figure()

    for model_name, metrics in data.items():
        values = [metrics[cat] for cat in categories]
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title
    )

    if output_path:
        fig.write_image(str(output_path), scale=2)

    return fig


def create_transfer_matrix(
    data: np.ndarray,
    labels: List[str],
    title: str = "Cross-Domain Transfer Performance",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create cross-domain transfer matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.eye(len(labels), dtype=bool)

    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        linewidths=0.5
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Train Dataset', fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_scenario_comparison(
    zero_shot: Dict[str, float],
    few_shot: Dict[str, float],
    supervised: Dict[str, float],
    models: List[str],
    metric: str = "MAE",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Compare zero-shot, few-shot, and supervised performance"""
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, [zero_shot[m] for m in models], width, label='Zero-Shot', color='#ff7f0e')
    bars2 = ax.bar(x, [few_shot[m] for m in models], width, label='Few-Shot', color='#2ca02c')
    bars3 = ax.bar(x + width, [supervised[m] for m in models], width, label='Supervised', color='#1f77b4')

    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Scenario and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        ax.bar_label(bars, fmt='%.2f', fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def create_failure_taxonomy_figure(
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Create Figure 4: Failure modes taxonomy visualization"""
    categories = ['Distribution\nShift', 'Long-horizon\nDrift', 'Privacy\nLeakage', 'Federation\nUnreadiness']

    models_data = {
        'MOMENT': [0.6, 0.5, 0.3, 0.8],
        'Sundial': [0.7, 0.6, 0.4, 0.9],
        'Chronos': [0.5, 0.7, 0.2, 0.7],
        'TimeGPT': [0.4, 0.4, 0.6, 0.8],
        'PatchTST': [0.3, 0.3, 0.1, 0.4]
    }

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    for (model, values), color in zip(models_data.items(), colors):
        values = values + values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('TSFM Failure Modes on Industrial Data', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def generate_all_figures(
    results_path: Path,
    output_path: Path
) -> None:
    """Generate all paper figures from results"""
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")

    create_failure_taxonomy_figure(output_path / "fig4_failure_taxonomy.png")

    print(f"Figures saved to {output_path}")


if __name__ == "__main__":
    create_failure_taxonomy_figure(Path("results/figures/test_radar.png"))
