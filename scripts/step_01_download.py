"""
Step 1: Download datasets (C-MAPSS, Wind SCADA, MIMII).

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
    RAW_DIR, MIMII_MACHINES, MIMII_ZENODO_RECORD_ID,
    MIMII_PREFERRED_VARIANT, setup_logging, ensure_dirs, mark_step_done,
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


# ── MIMII ────────────────────────────────────────────────────────────────
def download_mimii():
    dest = RAW_DIR / "mimii"
    dest.mkdir(parents=True, exist_ok=True)
    CHUNK = 1 << 20

    zenodo_bases = [
        f"https://zenodo.org/records/{MIMII_ZENODO_RECORD_ID}/files",
        f"https://zenodo.org/record/{MIMII_ZENODO_RECORD_ID}/files",
    ]

    all_ok = True
    for machine in MIMII_MACHINES:
        machine_dir = dest / machine
        if machine_dir.exists() and any(machine_dir.rglob("*.wav")):
            log.info(f"MIMII/{machine} already downloaded.")
            continue

        candidates = list(dict.fromkeys([
            f"{MIMII_PREFERRED_VARIANT}_{machine}.zip",
            f"0_dB_{machine}.zip",
            f"6_dB_{machine}.zip",
            f"-6_dB_{machine}.zip",
            f"{machine}.zip",
        ]))

        downloaded = False
        for base in zenodo_bases:
            if downloaded:
                break
            for fname in candidates:
                url = f"{base}/{fname}"
                zip_path = dest / fname
                log.info(f"Trying: {url}")
                try:
                    with urllib.request.urlopen(url, timeout=30) as resp:
                        total = int(resp.headers.get("Content-Length", 0))
                        free = shutil.disk_usage(dest).free
                        if total > 0:
                            needed = int(total * 1.1)
                            if free < needed:
                                raise OSError(
                                    "Insufficient disk space for MIMII archive: "
                                    f"need ~{needed/1e9:.2f} GB, free {free/1e9:.2f} GB"
                                )
                        elif free < 2 * (1 << 30):
                            raise OSError(
                                "Insufficient disk space for unknown-size MIMII archive "
                                f"(free {free/1e9:.2f} GB)"
                            )

                        got = 0
                        with open(zip_path, "wb") as f:
                            while True:
                                chunk = resp.read(CHUNK)
                                if not chunk:
                                    break
                                f.write(chunk)
                                got += len(chunk)
                                if total > 0:
                                    pct = min(int(got * 100 / total), 100)
                                    print(f"\r  {pct}% ({got/1e6:.1f}/{total/1e6:.1f} MB)",
                                          end="", flush=True)
                    print()
                    log.info(f"Extracting {fname} ...")
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(dest)
                    zip_path.unlink(missing_ok=True)
                    downloaded = True
                    log.info(f"MIMII/{machine} ready.")
                    break
                except (urllib.error.HTTPError, urllib.error.URLError,
                        TimeoutError, OSError) as e:
                    if zip_path.exists():
                        zip_path.unlink(missing_ok=True)
                    log.warning(f"  Failed: {e}")

        if not downloaded:
            log.error(f"MIMII/{machine} download FAILED.")
            all_ok = False

    return all_ok


# ── Verify ───────────────────────────────────────────────────────────────
def verify_downloads():
    checks = {
        "C-MAPSS FD001 train": RAW_DIR / "cmapss" / "train_FD001.txt",
        "C-MAPSS FD001 test":  RAW_DIR / "cmapss" / "test_FD001.txt",
        "Wind SCADA CSV":      next((RAW_DIR / "wind_scada").glob("*.csv"), None),
        "MIMII WAV":           next((RAW_DIR / "mimii").rglob("*.wav"), None),
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
    mimii_ok = download_mimii()

    ok, missing = verify_downloads()
    if ok:
        mark_step_done("step_01_download", {"datasets": ["cmapss", "wind_scada", "mimii"]})
        log.info("Step 1 complete — all datasets available.")
    else:
        log.warning("Step 1 partial — some datasets missing (see above).")
        log.warning(f"Download return flags: cmapss={cmapss_ok}, wind_scada={wind_ok}, mimii={mimii_ok}")
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
