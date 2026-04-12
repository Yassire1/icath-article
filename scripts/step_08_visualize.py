"""
Step 8: Generate publication-ready figures (PDF + PNG).

Reads aggregated CSVs from results/tables/ and produces plots in results/figures/.

Run:  python scripts/step_08_visualize.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed on VM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline_config import (
    RESULTS_DIR, TABLES_DIR, FIGURES_DIR,
    MODELS_ZERO_SHOT, MODELS_FEW_SHOT,
    setup_logging, ensure_dirs, mark_step_done,
)

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})
log = setup_logging()


def load_results_dir(res_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(res_dir.glob("*.json")):
        with open(f) as fp:
            r = json.load(fp)
        if "error" in r:
            continue
        m = r.get("metrics", {})
        row = {
            "scenario": r.get("scenario", res_dir.name),
            "model":    r.get("model"),
            "dataset":  r.get("dataset", r.get("target")),
            "mae":      m.get("mae"),
            "rmse":     m.get("rmse"),
            "mape":     m.get("mape"),
        }
        if "source" in r:
            row["source"] = r["source"]
            row["target"] = r["target"]
        rows.append(row)
    return pd.DataFrame(rows)


def fig_zero_shot_bar(df_zs):
    """Bar chart: zero-shot MAE by model × dataset."""
    if df_zs.empty:
        log.info("  Skipping zero-shot bar — no data.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    datasets_plot = df_zs["dataset"].unique()
    models_plot = [m for m in MODELS_ZERO_SHOT if m in df_zs["model"].values]
    x = range(len(models_plot))
    width = 0.8 / max(len(datasets_plot), 1)
    colors = plt.cm.tab10.colors

    for i, ds in enumerate(datasets_plot):
        vals = [
            df_zs[(df_zs["model"] == m) & (df_zs["dataset"] == ds)]["mae"].mean()
            for m in models_plot
        ]
        offset = (i - len(datasets_plot) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=ds, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models_plot, rotation=15, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Zero-Shot MAE by Model and Dataset")
    ax.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"zero_shot_mae_bar.{ext}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved zero_shot_mae_bar.pdf / .png")


def fig_zs_vs_fs(df_zs, df_fs):
    """Side-by-side zero-shot vs few-shot comparison."""
    if df_zs.empty or df_fs.empty:
        log.info("  Skipping ZS-vs-FS comparison — missing data.")
        return

    fs_models = [m for m in MODELS_FEW_SHOT if m in df_fs["model"].values]
    if not fs_models:
        return

    fig, axes = plt.subplots(1, len(fs_models), figsize=(5 * len(fs_models), 4), sharey=False)
    if len(fs_models) == 1:
        axes = [axes]

    for ax, model in zip(axes, fs_models):
        zs_row = df_zs[df_zs["model"] == model].set_index("dataset")["mae"]
        fs_row = df_fs[df_fs["model"] == model].set_index("dataset")["mae"]
        common_ds = sorted(set(zs_row.index) & set(fs_row.index))
        if not common_ds:
            ax.set_title(f"{model}\n(no common datasets)")
            continue
        x = np.arange(len(common_ds))
        ax.bar(x - 0.2, [zs_row.get(d, np.nan) for d in common_ds],
               0.35, label="Zero-shot", color="steelblue", alpha=0.85)
        ax.bar(x + 0.2, [fs_row.get(d, np.nan) for d in common_ds],
               0.35, label="Few-shot", color="coral", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(common_ds, rotation=15, ha="right")
        ax.set_ylabel("MAE")
        ax.set_title(f"{model.upper()}: ZS vs FS (1 %)")
        ax.legend()

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"zs_vs_fs_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved zs_vs_fs_comparison.pdf / .png")


def fig_cross_condition_heatmap(df_cc):
    """Heatmap: cross-condition MAE (FD001 → FD00x)."""
    if df_cc.empty:
        log.info("  Skipping cross-condition heatmap — no data.")
        return

    models_cc = [m for m in MODELS_ZERO_SHOT if m in df_cc["model"].values]
    targets = sorted(df_cc["target"].unique()) if "target" in df_cc.columns else sorted(df_cc["dataset"].unique())

    mat = []
    for model in models_cc:
        row = []
        col = "target" if "target" in df_cc.columns else "dataset"
        for tgt in targets:
            val = df_cc[(df_cc["model"] == model) & (df_cc[col] == tgt)]["mae"].mean()
            row.append(val)
        mat.append(row)

    mat = np.array(mat, dtype=float)

    fig, ax = plt.subplots(figsize=(max(4, len(targets) * 1.5), max(3, len(models_cc))))
    sns.heatmap(mat, annot=True, fmt=".4f",
                xticklabels=targets, yticklabels=models_cc,
                cmap="YlOrRd", ax=ax, cbar_kws={"label": "MAE"})
    ax.set_title("Cross-Condition MAE: C-MAPSS FD001 → FD00x")
    ax.set_xlabel("Target Condition")
    ax.set_ylabel("Model")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"cross_condition_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved cross_condition_heatmap.pdf / .png")


def main():
    ensure_dirs()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 8: Generate Figures")
    log.info("=" * 60)

    df_zs = load_results_dir(RESULTS_DIR / "zero_shot")
    df_fs = load_results_dir(RESULTS_DIR / "few_shot")
    df_cc = load_results_dir(RESULTS_DIR / "cross_condition")

    fig_zero_shot_bar(df_zs)
    fig_zs_vs_fs(df_zs, df_fs)
    fig_cross_condition_heatmap(df_cc)

    figures = list(FIGURES_DIR.glob("*.*"))
    if not figures:
        log.warning("No figures were generated; missing aggregated experiment data.")
    mark_step_done("step_08_visualize", {"figures": [f.name for f in figures]})
    log.info(f"Step 8 complete — {len(figures)} figure files.")


if __name__ == "__main__":
    main()
