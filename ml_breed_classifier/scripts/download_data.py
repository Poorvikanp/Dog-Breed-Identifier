import argparse
import subprocess
import sys
from pathlib import Path
import zipfile
import shutil


def run_kaggle(args_list):
    try:
        kaggle_exe = shutil.which("kaggle")
        if kaggle_exe:
            subprocess.check_call([kaggle_exe, *args_list])
        else:
            subprocess.check_call([sys.executable, "-m", "kaggle", *args_list])
    except subprocess.CalledProcessError as e:
        print("Kaggle command failed. Ensure Kaggle CLI is configured (kaggle.json).")
        raise e


def unzip_all_in_dir(directory: Path):
    for z in directory.glob("*.zip"):
        with zipfile.ZipFile(z, 'r') as f:
            f.extractall(directory)


def download_dataset1(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle(["competitions", "download", "-c", "dog-breed-identification", "-p", str(out_dir)])
    unzip_all_in_dir(out_dir)


def download_dataset2(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle(["datasets", "download", "-d", "jessicali9530/stanford-dogs-dataset", "-p", str(out_dir)])
    unzip_all_in_dir(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Download datasets via Kaggle API")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2", "both"], default="both")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    base = Path(args.out)
    if args.dataset in ("dataset1", "both"):
        download_dataset1(base / "dataset1_raw")
    if args.dataset in ("dataset2", "both"):
        download_dataset2(base / "dataset2_raw")


if __name__ == "__main__":
    main()
