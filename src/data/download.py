"""
Dataset download utilities for TSFM Industrial PdM Benchmark
"""

import os
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config():
    """Load dataset configuration"""
    with open("config/datasets.yaml", "r") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar archive"""
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(dest_dir)
    print(f"Extracted to {dest_dir}")


def download_cmapss(data_dir: Path):
    """
    Download C-MAPSS dataset from NASA
    Manual download required from:
    https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
    """
    dest = data_dir / "cmapss"
    dest.mkdir(parents=True, exist_ok=True)

    print("C-MAPSS Dataset")
    print("=" * 50)
    print("Manual download required:")
    print("1. Go to: https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip")
    print("2. Download CMAPSSData.zip")
    print(f"3. Extract to: {dest}")
    print("")
    print("Expected files after extraction:")
    print("  - train_FD001.txt, test_FD001.txt, RUL_FD001.txt")
    print("  - train_FD002.txt, test_FD002.txt, RUL_FD002.txt")
    print("  - train_FD003.txt, test_FD003.txt, RUL_FD003.txt")
    print("  - train_FD004.txt, test_FD004.txt, RUL_FD004.txt")
    print("=" * 50)

    if (dest / "train_FD001.txt").exists():
        print("C-MAPSS already downloaded!")
        return True
    return False


def download_phm_milling(data_dir: Path):
    """Download PHM 2010 Milling dataset - Manual download from PHM Society"""
    dest = data_dir / "phm_milling"
    dest.mkdir(parents=True, exist_ok=True)

    print("PHM Milling Dataset")
    print("=" * 50)
    print("Manual download required:")
    print("1. Go to: https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/")
    print("2. Download the milling dataset")
    print(f"3. Extract to: {dest}")
    print("=" * 50)

    return False


def download_pu_bearings(data_dir: Path):
    """Download Paderborn University Bearing dataset"""
    dest = data_dir / "pu_bearings"
    dest.mkdir(parents=True, exist_ok=True)

    print("Paderborn University Bearings Dataset")
    print("=" * 50)
    print("Manual download required:")
    print("1. Go to: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/")
    print("2. Request access and download")
    print(f"3. Extract to: {dest}")
    print("=" * 50)

    return False


def download_wind_scada(data_dir: Path):
    """Download Wind Turbine SCADA dataset from Kaggle"""
    dest = data_dir / "wind_scada"
    dest.mkdir(parents=True, exist_ok=True)

    print("Wind SCADA Dataset")
    print("=" * 50)
    print("Download options:")
    print("Option 1 - Kaggle CLI:")
    print("  kaggle datasets download -d berkerisen/wind-turbine-scada-dataset")
    print(f"  Extract to: {dest}")
    print("")
    print("Option 2 - Manual:")
    print("  https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset")
    print("=" * 50)

    return False


def download_mimii(data_dir: Path):
    """Guide for MIMII dataset download from Zenodo (manual)."""
    dest = data_dir / "mimii"
    dest.mkdir(parents=True, exist_ok=True)

    print("MIMII Dataset")
    print("=" * 50)
    print("Manual download recommended (large dataset).")
    print("1. Go to: https://zenodo.org/record/3384388")
    print("2. Download required machine archives (fan, pump, slider, valve)")
    print("3. Extract under:")
    print(f"   {dest}")
    print("4. Ensure machine folders exist: fan, pump, slider, valve")
    print("=" * 50)

    expected_dirs = [
        dest / "fan",
        dest / "pump",
        dest / "slider",
        dest / "valve",
    ]
    return all(path.exists() for path in expected_dirs)


def download_pronostia(data_dir: Path):
    """Download PRONOSTIA/FEMTO bearing dataset"""
    dest = data_dir / "pronostia"
    dest.mkdir(parents=True, exist_ok=True)

    print("PRONOSTIA Dataset")
    print("=" * 50)
    print("Manual download required:")
    print("1. Go to: https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset")
    print("2. Clone or download the repository")
    print(f"3. Copy data to: {dest}")
    print("=" * 50)

    return False


def download_all_datasets(data_dir: str = "data/raw"):
    """Download all datasets"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DOWNLOADING ALL DATASETS")
    print("=" * 60 + "\n")

    results = {
        "cmapss": download_cmapss(data_path),
        "phm_milling": download_phm_milling(data_path),
        "pu_bearings": download_pu_bearings(data_path),
        "wind_scada": download_wind_scada(data_path),
        "mimii": download_mimii(data_path),
        "pronostia": download_pronostia(data_path)
    }

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        status_str = "READY" if status else "MANUAL DOWNLOAD REQUIRED"
        print(f"  {name}: {status_str}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    download_all_datasets()
