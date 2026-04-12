"""
Step 1: Download datasets (C-MAPSS, Wind SCADA, PHM Milling).

Idempotent — skips datasets that are already present.
Run:  python scripts/step_01_download.py
      python scripts/step_01_download.py --strict
"""

import argparse
import subprocess
import sys
import shutil
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from pipeline_config import (
    RAW_DIR, setup_logging, ensure_dirs, mark_step_done,
)

log = setup_logging()


def _kaggle_cmd():
    """Resolve a Kaggle CLI command that works in this environment."""
    which = shutil.which("kaggle")
    if which:
        return [which]

    sibling = Path(sys.executable).with_name("kaggle")
    if sibling.exists():
        return [sibling]

    # Fallback for environments where only the module entry is available.
    return [sys.executable, "-m", "kaggle.cli"]


def _flatten_single_subdir(dest: Path) -> None:
    """Flatten a single extracted wrapper directory, if present."""
    children = list(dest.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        return

    wrapper = children[0]
    for item in list(wrapper.iterdir()):
        target = dest / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))

    try:
        wrapper.rmdir()
    except OSError:
        pass


def _has_phm_milling_raw(dest: Path) -> bool:
    """Check whether PHM Milling raw files look complete enough to preprocess."""
    csv_files = list(dest.rglob("*.csv"))
    wear_files = [path for path in csv_files if "wear" in path.stem.lower()]
    acquisition_files = [path for path in csv_files if "wear" not in path.stem.lower()]
    return bool(wear_files) and len(acquisition_files) >= 10


# ── C-MAPSS ──────────────────────────────────────────────────────────────
def download_cmapss():
    dest = RAW_DIR / "cmapss"
    dest.mkdir(parents=True, exist_ok=True)
    sentinel = dest / "train_FD001.txt"

    if sentinel.exists():
        log.info("C-MAPSS already downloaded.")
        return True

    log.info("Downloading C-MAPSS from Kaggle (behrad3d/nasa-cmaps) ...")
    try:
        subprocess.run(
            _kaggle_cmd() + ["datasets", "download", "-d", "behrad3d/nasa-cmaps",
             "-p", str(dest), "--unzip"],
            capture_output=True, text=True, check=True,
        )
        log.info("C-MAPSS download complete.")
        # Flatten any sub-folder
        import shutil
        for sub in dest.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    try:
                        shutil.move(str(f), str(dest / f.name))
                    except Exception as e:
                        log.warning(f"Could not move {f.name}: {e}")
                try:
                    sub.rmdir()
                except OSError:
                    pass
        return sentinel.exists()
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or str(e)).strip()
        log.error(f"C-MAPSS download failed: {msg[:500]}")
        log.info("Manual: https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip")
        return False
    except Exception as e:
        log.error(f"C-MAPSS download failed: {e}")
        log.info("Manual: https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip")
        return False


# ── Wind SCADA ───────────────────────────────────────────────────────────
def download_wind_scada():
    dest = RAW_DIR / "wind_scada"
    dest.mkdir(parents=True, exist_ok=True)
    sentinel = next(dest.glob("*.csv"), None)

    if sentinel:
        log.info(f"Wind SCADA already downloaded: {sentinel.name}")
        return True

    slugs = [
        "berkerisen/wind-turbine-scada-dataset",
        "berkerisen/wind-power-forecasting",
    ]
    for slug in slugs:
        log.info(f"Trying Kaggle dataset: {slug}")
        try:
            proc = subprocess.run(
                _kaggle_cmd() + ["datasets", "download", "-d", slug,
                                 "-p", str(dest), "--unzip"],
                capture_output=True, text=True,
            )
        except Exception as e:
            log.warning(f"Failed to launch Kaggle CLI: {e}")
            break
        if proc.returncode == 0:
            import shutil
            for sub in dest.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        try:
                            shutil.move(str(f), str(dest / f.name))
                        except Exception:
                            pass
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
            log.info("Wind SCADA download complete.")
            return True
        log.warning(f"Failed for {slug}: {(proc.stderr or proc.stdout or '').strip()[:300]}")

    log.error("Wind SCADA download failed for all slugs.")
    log.info("Manual: https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset")
    return False


# ── PHM Milling ──────────────────────────────────────────────────────────
def download_phm_milling():
    dest = RAW_DIR / "phm_milling"
    dest.mkdir(parents=True, exist_ok=True)

    if _has_phm_milling_raw(dest):
        log.info("PHM Milling already downloaded.")
        return True

    slugs = ["rabahba/phm-data-challenge-2010"]
    for slug in slugs:
        log.info(f"Trying Kaggle dataset: {slug}")
        try:
            proc = subprocess.run(
                _kaggle_cmd() + ["datasets", "download", "-d", slug,
                                 "-p", str(dest), "--unzip"],
                capture_output=True, text=True,
            )
        except Exception as e:
            log.warning(f"Failed to launch Kaggle CLI: {e}")
            break

        if proc.returncode == 0:
            _flatten_single_subdir(dest)
            if _has_phm_milling_raw(dest):
                log.info("PHM Milling download complete.")
                return True

        log.warning(f"Failed for {slug}: {(proc.stderr or proc.stdout or '').strip()[:300]}")

    log.error("PHM Milling download failed for all automated sources.")
    log.info("Manual: https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/")
    return _has_phm_milling_raw(dest)


# ── Verify ───────────────────────────────────────────────────────────────
def verify_downloads():
    checks = {
        "C-MAPSS FD001 train": RAW_DIR / "cmapss" / "train_FD001.txt",
        "C-MAPSS FD001 test":  RAW_DIR / "cmapss" / "test_FD001.txt",
        "Wind SCADA CSV":      next((RAW_DIR / "wind_scada").glob("*.csv"), None),
        "PHM Milling CSV":     next((RAW_DIR / "phm_milling").rglob("*.csv"), None),
    }
    all_ok = True
    missing = []
    for label, path in checks.items():
        exists = path is not None and path.exists()
        status = "OK" if exists else "MISSING"
        log.info(f"  [{status}] {label}")
        if not exists:
            all_ok = False
            missing.append(label)
    return all_ok, missing


# ── Main entry ───────────────────────────────────────────────────────────
def main(strict: bool = False):
    ensure_dirs()
    log.info("=" * 60)
    log.info("STEP 1: Download Datasets")
    log.info("=" * 60)

    cmapss_ok = download_cmapss()
    wind_ok = download_wind_scada()
    phm_ok = download_phm_milling()

    ok, missing = verify_downloads()
    if ok:
        mark_step_done("step_01_download", {"datasets": ["cmapss", "wind_scada", "phm_milling"]})
        log.info("Step 1 complete — all datasets available.")
    else:
        log.warning("Step 1 partial — some datasets missing (see above).")
        log.warning(f"Download return flags: cmapss={cmapss_ok}, wind_scada={wind_ok}, phm_milling={phm_ok}")
        if strict:
            raise RuntimeError(
                "Strict mode enabled and required datasets are missing: "
                + ", ".join(missing)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Download datasets")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any required dataset is missing after download attempts.",
    )
    args = parser.parse_args()
    main(strict=args.strict)
