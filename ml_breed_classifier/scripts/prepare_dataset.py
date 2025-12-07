import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


def ensure_empty(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def prepare_dataset1(raw_dir: Path, out_dir: Path, val_ratio: float = 0.15, seed: int = 42):
    # Kaggle dog-breed-identification: labels.csv with columns id,breed and images under train/ and test/
    labels_path = raw_dir / "labels.csv"
    train_images_dir = raw_dir / "train"
    assert labels_path.exists(), f"Missing {labels_path}"
    assert train_images_dir.exists(), f"Missing {train_images_dir}"

    rows: List[Tuple[str, str]] = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_id = r["id"]
            breed = r["breed"].replace(" ", "_")
            rows.append((img_id, breed))

    classes = sorted({b for _, b in rows})
    by_class: Dict[str, List[str]] = {c: [] for c in classes}
    for img_id, breed in rows:
        by_class[breed].append(img_id)

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    ensure_empty(train_dir)
    ensure_empty(val_dir)

    random.seed(seed)
    for breed, ids in by_class.items():
        ids_train, ids_val = train_test_split(ids, test_size=val_ratio, random_state=seed, shuffle=True, stratify=None)
        for split, subset in ((train_dir, ids_train), (val_dir, ids_val)):
            class_dir = split / breed
            class_dir.mkdir(parents=True, exist_ok=True)
            for img_id in subset:
                src = train_images_dir / f"{img_id}.jpg"
                if src.exists():
                    shutil.copy2(src, class_dir / src.name)


def gather_images_from_class_tree(images_root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for class_dir in images_root.iterdir():
        if class_dir.is_dir():
            # Stanford Dogs has class directories like "n02085620-Chihuahua"
            name = class_dir.name
            label = name.split("-", 1)[-1] if "-" in name else name
            label = label.replace(" ", "_")
            files = [p for p in class_dir.rglob("*.jpg")]
            if files:
                mapping.setdefault(label, []).extend(files)
    return mapping


def _detect_images_root(raw_dir: Path) -> Path:
    """Detect the root folder that contains class subfolders with JPGs.
    Handles layouts like:
    - raw_dir/Images/<class>/*.jpg (expected)
    - raw_dir/images/Images/<class>/*.jpg (observed)
    - raw_dir/images/<class>/*.jpg (fallback)
    """
    candidates = [
        raw_dir / "Images",
        raw_dir / "images" / "Images",
        raw_dir / "images",
    ]
    for c in candidates:
        if c.exists() and any(p.is_dir() for p in c.iterdir() if p.is_dir()):
            # Check at least one jpg deep to be safe
            jpgs = list(c.rglob("*.jpg"))
            if len(jpgs) > 0:
                return c
    raise AssertionError(f"Could not locate Images root under {raw_dir}. Checked: {candidates}")


def prepare_dataset2(raw_dir: Path, out_dir: Path, val_ratio: float = 0.15, seed: int = 42, images_root_override: Path | None = None):
    # Stanford Dogs: find Images/ with class subfolders
    images_root = images_root_override if images_root_override else _detect_images_root(raw_dir)

    by_class = gather_images_from_class_tree(images_root)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    ensure_empty(train_dir)
    ensure_empty(val_dir)

    for label, paths in by_class.items():
        if len(paths) < 2:
            continue
        paths_train, paths_val = train_test_split(paths, test_size=val_ratio, random_state=seed, shuffle=True)
        for split, subset in ((train_dir, paths_train), (val_dir, paths_val)):
            class_dir = split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for src in subset:
                dst = class_dir / src.name
                shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets into classed train/val directories")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2", "both"], default="both")
    parser.add_argument("--inbase", default="data")
    parser.add_argument("--outbase", default="data")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--images_root", default=None, help="Override path to Images root for dataset2 (optional)")
    args = parser.parse_args()

    if args.dataset in ("dataset1", "both"):
        prepare_dataset1(Path(args.inbase) / "dataset1_raw", Path(args.outbase) / "dataset1" / "prepared", args.val_ratio)
    if args.dataset in ("dataset2", "both"):
        override = Path(args.images_root) if args.images_root else None
        prepare_dataset2(Path(args.inbase) / "dataset2_raw", Path(args.outbase) / "dataset2" / "prepared", args.val_ratio, images_root_override=override)


if __name__ == "__main__":
    main()
