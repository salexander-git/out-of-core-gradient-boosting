import csv
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np

ZIP_URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
ZIP_NAME = "yearpredictionmsd.zip"
TXT_NAME = "YearPredictionMSD.txt"

TRAIN_ROWS = 463_715
TEST_ROWS = 51_630
TOTAL_ROWS = TRAIN_ROWS + TEST_ROWS
NUM_FEATURES = 90


def download_if_needed(zip_path: Path) -> None:
    if zip_path.exists():
        print(f"Using existing archive: {zip_path}")
        return

    print(f"Downloading {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, zip_path)
    print(f"Saved archive to {zip_path}")


def extract_if_needed(zip_path: Path, txt_path: Path) -> None:
    if txt_path.exists():
        print(f"Using existing text file: {txt_path}")
        return

    print(f"Extracting {TXT_NAME} from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if TXT_NAME not in members:
            raise RuntimeError(f"{TXT_NAME} not found in archive. Found: {members[:10]}")
        zf.extract(TXT_NAME, path=txt_path.parent)
    print(f"Extracted to {txt_path}")


def compute_train_minmax(txt_path: Path):
    print("Pass 1/2: computing train-set feature min/max ...")
    fmin = np.full(NUM_FEATURES, np.inf, dtype=np.float32)
    fmax = np.full(NUM_FEATURES, -np.inf, dtype=np.float32)

    with txt_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if row_idx >= TRAIN_ROWS:
                break

            if len(row) != NUM_FEATURES + 1:
                raise RuntimeError(
                    f"Row {row_idx} has {len(row)} columns; expected {NUM_FEATURES + 1}"
                )

            feats = np.asarray(row[1:], dtype=np.float32)
            fmin = np.minimum(fmin, feats)
            fmax = np.maximum(fmax, feats)

            if (row_idx + 1) % 100_000 == 0:
                print(f"  processed {row_idx + 1:,} / {TRAIN_ROWS:,} train rows")

    span = fmax - fmin
    span[span == 0.0] = 1.0
    return fmin, span


def scale_to_engine_range(feats: np.ndarray, fmin: np.ndarray, span: np.ndarray) -> np.ndarray:
    scaled = ((feats - fmin) / span) * 20.0 - 10.0
    return np.clip(scaled, -10.0, 10.0).astype(np.float32, copy=False)


def float_to_qbin(feats_scaled: np.ndarray) -> np.ndarray:
    f_min, f_max = -10.0, 10.0
    scale = 255.0 / (f_max - f_min)
    return np.clip((feats_scaled - f_min) * scale, 0, 255).astype(np.uint8, copy=False)


def write_binary_files(txt_path: Path, fmin: np.ndarray, span: np.ndarray) -> None:
    train_feat_path = Path("yearpred_train_features.bin")
    train_feat_qbin_path = Path("yearpred_train_features_qbin.bin")
    train_label_path = Path("yearpred_train_labels.bin")

    test_feat_path = Path("yearpred_test_features.bin")
    test_feat_qbin_path = Path("yearpred_test_features_qbin.bin")
    test_label_path = Path("yearpred_test_labels.bin")

    print("Pass 2/2: writing scaled train/test binaries ...")

    train_feat = train_feat_path.open("wb")
    train_feat_qbin = train_feat_qbin_path.open("wb")
    train_label = train_label_path.open("wb")

    test_feat = test_feat_path.open("wb")
    test_feat_qbin = test_feat_qbin_path.open("wb")
    test_label = test_label_path.open("wb")

    try:
        with txt_path.open("r", newline="") as f:
            reader = csv.reader(f)

            row_count = 0
            for row_idx, row in enumerate(reader):
                if len(row) != NUM_FEATURES + 1:
                    raise RuntimeError(
                        f"Row {row_idx} has {len(row)} columns; expected {NUM_FEATURES + 1}"
                    )

                label = np.float32(row[0])
                feats = np.asarray(row[1:], dtype=np.float32)
                feats_scaled = scale_to_engine_range(feats, fmin, span)
                feats_qbin = float_to_qbin(feats_scaled)

                if row_idx < TRAIN_ROWS:
                    feats_scaled.tofile(train_feat)
                    feats_qbin.tofile(train_feat_qbin)
                    np.asarray([label], dtype=np.float32).tofile(train_label)
                else:
                    feats_scaled.tofile(test_feat)
                    feats_qbin.tofile(test_feat_qbin)
                    np.asarray([label], dtype=np.float32).tofile(test_label)

                row_count += 1
                if row_count % 100_000 == 0:
                    print(f"  wrote {row_count:,} / {TOTAL_ROWS:,} rows")

        if row_count != TOTAL_ROWS:
            raise RuntimeError(f"Expected {TOTAL_ROWS:,} rows, found {row_count:,}")

    finally:
        train_feat.close()
        train_feat_qbin.close()
        train_label.close()
        test_feat.close()
        test_feat_qbin.close()
        test_label.close()

    print("\nWrote:")
    for p in [
        train_feat_path, train_feat_qbin_path, train_label_path,
        test_feat_path, test_feat_qbin_path, test_label_path
    ]:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<32} {size_mb:9.2f} MB")


def main() -> None:
    zip_path = Path(ZIP_NAME)
    txt_path = Path(TXT_NAME)

    download_if_needed(zip_path)
    extract_if_needed(zip_path, txt_path)

    fmin, span = compute_train_minmax(txt_path)
    write_binary_files(txt_path, fmin, span)

    print("\nSummary")
    print(f"  Train rows:    {TRAIN_ROWS:,}")
    print(f"  Test rows:     {TEST_ROWS:,}")
    print(f"  Num features:  {NUM_FEATURES}")
    print("  Features scaled to approximately [-10, 10] using train-set min/max")
    print("  Labels preserved as raw years")
    print("  Wrote both float32 and pre-quantized uint8 feature files")

    print("\nReady-to-use commands")
    print("  python main.py")
    print(f"  load_train_qbin yearpred_train_features_qbin.bin yearpred_train_labels.bin {NUM_FEATURES}")
    print("  set_verbose_levels off")
    print("  train 20 6 0.1 1.0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)