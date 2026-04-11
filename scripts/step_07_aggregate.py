"""
Step 7: Aggregate results into publication-ready CSV tables.

Reads per-run JSON files from results/{zero_shot,few_shot,cross_condition}
and produces pivot tables + comparison CSVs in results/tables/.

Run:  python scripts/step_07_aggregate.py
"""

import json
from pathlib import Path

import pandas as pd

from pipeline_config import (
    RESULTS_DIR, TABLES_DIR,
    MODELS_FEW_SHOT,
    setup_logging, ensure_dirs, mark_step_done,
)

log = setup_logging()


def load_results_dir(res_dir: Path) -> pd.DataFrame:
    """Load all JSON result files from a directory into a DataFrame."""
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


def main():
    ensure_dirs()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 7: Aggregate Results → CSV Tables")
    log.info("=" * 60)

    df_zs = load_results_dir(RESULTS_DIR / "zero_shot")
    df_fs = load_results_dir(RESULTS_DIR / "few_shot")
    df_cc = load_results_dir(RESULTS_DIR / "cross_condition")
    df_all = pd.concat([df_zs, df_fs, df_cc], ignore_index=True)

    files_written = []

    # ── 1. Zero-shot pivot (model × dataset × MAE) ──────────────────────
    if not df_zs.empty:
        pivot = df_zs.pivot_table(index="model", columns="dataset", values="mae", aggfunc="mean")
        pivot["mean"] = pivot.mean(axis=1)
        pivot.to_csv(TABLES_DIR / "zero_shot_mae.csv")
        df_zs.to_csv(TABLES_DIR / "zero_shot_full.csv", index=False)
        files_written += ["zero_shot_mae.csv", "zero_shot_full.csv"]
        log.info(f"  Zero-shot: {len(df_zs)} runs")
        log.info(f"\n{pivot.round(4).to_string()}\n")
    else:
        log.info("  No zero-shot results found.")

    # ── 2. Few-shot pivot ────────────────────────────────────────────────
    if not df_fs.empty:
        pivot = df_fs.pivot_table(index="model", columns="dataset", values="mae", aggfunc="mean")
        pivot["mean"] = pivot.mean(axis=1)
        pivot.to_csv(TABLES_DIR / "few_shot_mae.csv")
        df_fs.to_csv(TABLES_DIR / "few_shot_full.csv", index=False)
        files_written += ["few_shot_mae.csv", "few_shot_full.csv"]
        log.info(f"  Few-shot: {len(df_fs)} runs")
        log.info(f"\n{pivot.round(4).to_string()}\n")
    else:
        log.info("  No few-shot results found.")

    # ── 3. Zero-shot vs. few-shot comparison ─────────────────────────────
    if not df_zs.empty and not df_fs.empty:
        rows_cmp = []
        for m in MODELS_FEW_SHOT:
            for ds in df_zs["dataset"].unique():
                zs = df_zs[(df_zs["model"] == m) & (df_zs["dataset"] == ds)]["mae"].mean()
                fs = df_fs[(df_fs["model"] == m) & (df_fs["dataset"] == ds)]["mae"].mean()
                if pd.notna(zs) or pd.notna(fs):
                    imp = ((zs - fs) / zs * 100) if pd.notna(zs) and pd.notna(fs) and zs > 0 else None
                    rows_cmp.append({
                        "model": m, "dataset": ds,
                        "zero_shot_mae": zs, "few_shot_mae": fs,
                        "improvement_pct": imp,
                    })
        df_cmp = pd.DataFrame(rows_cmp)
        df_cmp.to_csv(TABLES_DIR / "zs_vs_fs_comparison.csv", index=False)
        files_written.append("zs_vs_fs_comparison.csv")
        log.info(f"  ZS-vs-FS comparison: {len(df_cmp)} rows")

    # ── 4. Cross-condition pivot ─────────────────────────────────────────
    if not df_cc.empty:
        pivot = df_cc.pivot_table(index="model", columns="target", values="mae", aggfunc="mean")
        col_order = [c for c in ["FD001", "FD002", "FD003", "FD004"] if c in pivot.columns]
        pivot = pivot[col_order]
        pivot.to_csv(TABLES_DIR / "cross_condition_mae.csv")
        df_cc.to_csv(TABLES_DIR / "cross_condition_full.csv", index=False)
        files_written += ["cross_condition_mae.csv", "cross_condition_full.csv"]
        log.info(f"  Cross-condition: {len(df_cc)} runs")
        log.info(f"\n{pivot.round(4).to_string()}\n")
    else:
        log.info("  No cross-condition results found.")

    # ── 5. Master CSV ────────────────────────────────────────────────────
    df_all.to_csv(TABLES_DIR / "all_results.csv", index=False)
    files_written.append("all_results.csv")

    log.info(f"  Total records: {len(df_all)}")
    log.info(f"  Files written: {', '.join(files_written)}")

    mark_step_done("step_07_aggregate", {"files": files_written, "total_records": len(df_all)})
    log.info("Step 7 complete.")


if __name__ == "__main__":
    main()
