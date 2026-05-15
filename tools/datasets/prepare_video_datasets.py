#!/usr/bin/env python3
"""Prepare natural-video datasets for V1 next-state prediction experiments.

The script intentionally keeps large dataset files outside the git repository.
It downloads small, reproducible starter subsets first, extracts only the video
frames needed by the V1 model, creates 32x32 grayscale frame arrays, and writes
CSV manifests for frame and one-step transition training.

Default root:
    /home/vishnu/datasets/v1_video

Override with:
    V1_VIDEO_DATA_ROOT=/path/to/root
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
KITTI_DEFAULT_DATE = "2011_09_26"
KITTI_STARTER_DRIVES = ("0001", "0002", "0020")
KITTI_CAMERA = "image_00"  # rectified left grayscale

DAVIS_TRAINVAL_480P_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    "DAVIS-2017-trainval-480p.zip"
)


@dataclass(frozen=True)
class KittiDrive:
    date: str
    drive: str

    @property
    def sequence(self) -> str:
        return f"{self.date}_drive_{self.drive}_sync"

    @property
    def url_dir(self) -> str:
        return f"{self.date}_drive_{self.drive}"

    @property
    def archive_name(self) -> str:
        return f"{self.sequence}.zip"

    @property
    def url(self) -> str:
        return f"{KITTI_BASE_URL}/{self.url_dir}/{self.archive_name}"


def default_root() -> Path:
    return Path(os.environ.get("V1_VIDEO_DATA_ROOT", "/home/vishnu/datasets/v1_video"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, *, force: bool = False) -> None:
    ensure_dir(dest.parent)
    if dest.exists() and not force:
        print(f"[skip] exists: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"[download] {url}")
    print(f"           -> {dest}")

    last = time.time()
    downloaded = 0

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last, downloaded
        downloaded = block_count * block_size
        now = time.time()
        if now - last < 5.0 and downloaded < total_size:
            return
        last = now
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(f"  {downloaded / 1e6:8.1f} / {total_size / 1e6:8.1f} MB ({pct:5.1f}%)")
        else:
            print(f"  {downloaded / 1e6:8.1f} MB")

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    tmp.replace(dest)


def extract_zip_members(archive: Path, dest_root: Path, member_predicate) -> int:
    ensure_dir(dest_root)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member_predicate(member.filename):
                continue
            zf.extract(member, dest_root)
            count += 1
    return count


def extract_kitti_drive(root: Path, drive: KittiDrive, *, camera: str = KITTI_CAMERA) -> Path:
    archive = root / "downloads" / "kitti_raw" / drive.archive_name
    raw_root = root / "raw" / "kitti_raw"
    sequence_dir = raw_root / drive.date / drive.sequence
    image_dir = sequence_dir / camera / "data"
    if image_dir.exists() and any(image_dir.glob("*.png")):
        print(f"[skip] extracted {drive.sequence} {camera}: {image_dir}")
        return image_dir

    print(f"[extract] {archive} camera={camera}")
    def wanted(name: str) -> bool:
        return f"/{camera}/data/" in f"/{name}" and name.endswith(".png")

    count = extract_zip_members(archive, raw_root, wanted)
    if count == 0:
        raise RuntimeError(f"No {camera}/data PNGs extracted from {archive}")
    print(f"[extract] {drive.sequence}: {count} PNG frames")
    return image_dir


def extract_kitti_calib(root: Path, date: str = KITTI_DEFAULT_DATE) -> None:
    archive = root / "downloads" / "kitti_raw" / f"{date}_calib.zip"
    raw_root = root / "raw" / "kitti_raw"
    if (raw_root / date / "calib_cam_to_cam.txt").exists():
        print(f"[skip] calibration exists for {date}")
        return
    print(f"[extract] {archive}")
    extract_zip_members(archive, raw_root, lambda name: name.endswith(".txt"))


def frame_split(sequence_index: int) -> str:
    # Deterministic, sequence-level split. Starter subset leaves one held-out drive.
    return "val" if sequence_index % 5 == 4 else "train"


def local_contrast_normalize(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32) / 255.0
    mean = float(frame.mean())
    std = float(frame.std())
    if std < 1e-6:
        return np.zeros_like(frame, dtype=np.float32)
    normed = (frame - mean) / (std + 1e-6)
    # Bounded range keeps downstream currents sane while preserving contrast.
    return np.clip(normed / 3.0, -1.0, 1.0).astype(np.float32)


def preprocess_frame(src: Path, dest: Path, *, size: int) -> None:
    ensure_dir(dest.parent)
    if dest.exists():
        return
    with Image.open(src) as img:
        gray = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = local_contrast_normalize(np.asarray(gray))
    np.save(dest, arr)


def build_kitti_manifests(
    root: Path,
    drives: list[KittiDrive],
    *,
    camera: str,
    size: int,
) -> None:
    manifests = root / "manifests"
    ensure_dir(manifests)
    processed_root = root / "processed" / f"kitti_raw_{size}x{size}"
    frame_csv = manifests / f"kitti_raw_{camera}_{size}x{size}_frames.csv"
    transition_csv = manifests / f"kitti_raw_{camera}_{size}x{size}_transitions.csv"

    frame_rows: list[dict[str, str | int | float]] = []
    transition_rows: list[dict[str, str | int | float]] = []
    for sequence_index, drive in enumerate(drives):
        split = frame_split(sequence_index)
        image_dir = root / "raw" / "kitti_raw" / drive.date / drive.sequence / camera / "data"
        frames = sorted(image_dir.glob("*.png"))
        if len(frames) < 2:
            raise RuntimeError(f"Need at least 2 frames for {drive.sequence}, found {len(frames)}")
        processed_seq = processed_root / drive.date / drive.sequence / camera
        processed_paths: list[Path] = []
        for frame_index, src in enumerate(frames):
            dest = processed_seq / f"{frame_index:010d}.npy"
            preprocess_frame(src, dest, size=size)
            processed_paths.append(dest)
            frame_rows.append(
                {
                    "dataset": "kitti_raw",
                    "date": drive.date,
                    "sequence": drive.sequence,
                    "camera": camera,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index * 0.1,  # synced KITTI raw camera stream is 10 Hz.
                    "split": split,
                    "raw_frame_path": str(src),
                    "processed_frame_path": str(dest),
                    "width": size,
                    "height": size,
                }
            )
        for frame_index in range(len(processed_paths) - 1):
            transition_rows.append(
                {
                    "dataset": "kitti_raw",
                    "date": drive.date,
                    "sequence": drive.sequence,
                    "camera": camera,
                    "t_index": frame_index,
                    "t_plus_1_index": frame_index + 1,
                    "dt_s": 0.1,
                    "split": split,
                    "frame_t_path": str(processed_paths[frame_index]),
                    "frame_t_plus_1_path": str(processed_paths[frame_index + 1]),
                }
            )

    write_csv(frame_csv, frame_rows)
    write_csv(transition_csv, transition_rows)
    print(f"[manifest] {frame_csv} rows={len(frame_rows)}")
    print(f"[manifest] {transition_csv} rows={len(transition_rows)}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_drives(values: list[str], date: str) -> list[KittiDrive]:
    drives = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if "_drive_" in value:
            # Accept full sequence-like input: 2011_09_26_drive_0001.
            parts = value.split("_drive_")
            drives.append(KittiDrive(parts[0], parts[1].replace("_sync", "")))
        else:
            drives.append(KittiDrive(date, value.zfill(4)))
    if not drives:
        raise ValueError("At least one drive is required")
    return drives


def prepare_kitti(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    drives = parse_drives(args.drives, args.date)
    print(f"[root] {root}")
    print("[kitti] drives=" + ",".join(d.sequence for d in drives))
    download_file(
        f"{KITTI_BASE_URL}/{args.date}_calib.zip",
        root / "downloads" / "kitti_raw" / f"{args.date}_calib.zip",
        force=args.force,
    )
    for drive in drives:
        download_file(drive.url, root / "downloads" / "kitti_raw" / drive.archive_name, force=args.force)
    extract_kitti_calib(root, args.date)
    for drive in drives:
        extract_kitti_drive(root, drive, camera=args.camera)
    build_kitti_manifests(root, drives, camera=args.camera, size=args.size)


def download_davis(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    archive = root / "downloads" / "davis" / "DAVIS-2017-trainval-480p.zip"
    download_file(DAVIS_TRAINVAL_480P_URL, archive, force=args.force)
    if args.extract:
        raw_root = root / "raw" / "davis"
        if (raw_root / "DAVIS" / "JPEGImages" / "480p").exists() and not args.force:
            print(f"[skip] DAVIS extracted: {raw_root}")
            return
        print(f"[extract] {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_root)


def build_davis_manifests(root: Path, *, size: int) -> None:
    image_root = root / "raw" / "davis" / "DAVIS" / "JPEGImages" / "480p"
    if not image_root.exists():
        raise FileNotFoundError(f"DAVIS image root not found: {image_root}")
    manifests = root / "manifests"
    ensure_dir(manifests)
    processed_root = root / "processed" / f"davis_2017_trainval_480p_{size}x{size}"
    frame_csv = manifests / f"davis_2017_trainval_480p_{size}x{size}_frames.csv"
    transition_csv = manifests / f"davis_2017_trainval_480p_{size}x{size}_transitions.csv"

    sequences = sorted(path for path in image_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"No DAVIS sequences found under {image_root}")

    frame_rows: list[dict[str, str | int | float]] = []
    transition_rows: list[dict[str, str | int | float]] = []
    for sequence_index, sequence_dir in enumerate(sequences):
        split = frame_split(sequence_index)
        frames = sorted(sequence_dir.glob("*.jpg"))
        if len(frames) < 2:
            continue
        processed_seq = processed_root / sequence_dir.name
        processed_paths: list[Path] = []
        for frame_index, src in enumerate(frames):
            dest = processed_seq / f"{frame_index:05d}.npy"
            preprocess_frame(src, dest, size=size)
            processed_paths.append(dest)
            frame_rows.append(
                {
                    "dataset": "davis_2017_trainval_480p",
                    "sequence": sequence_dir.name,
                    "frame_index": frame_index,
                    "timestamp_s": "",
                    "split": split,
                    "raw_frame_path": str(src),
                    "processed_frame_path": str(dest),
                    "width": size,
                    "height": size,
                }
            )
        for frame_index in range(len(processed_paths) - 1):
            transition_rows.append(
                {
                    "dataset": "davis_2017_trainval_480p",
                    "sequence": sequence_dir.name,
                    "t_index": frame_index,
                    "t_plus_1_index": frame_index + 1,
                    "dt_s": "",
                    "split": split,
                    "frame_t_path": str(processed_paths[frame_index]),
                    "frame_t_plus_1_path": str(processed_paths[frame_index + 1]),
                }
            )

    write_csv(frame_csv, frame_rows)
    write_csv(transition_csv, transition_rows)
    print(f"[manifest] {frame_csv} rows={len(frame_rows)}")
    print(f"[manifest] {transition_csv} rows={len(transition_rows)}")


def prepare_davis(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    archive = root / "downloads" / "davis" / "DAVIS-2017-trainval-480p.zip"
    if not archive.exists():
        download_file(DAVIS_TRAINVAL_480P_URL, archive, force=args.force)
    raw_root = root / "raw" / "davis"
    if not (raw_root / "DAVIS" / "JPEGImages" / "480p").exists():
        print(f"[extract] {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_root)
    build_davis_manifests(root, size=args.size)


def verify(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    print(f"[verify] root={root}")
    paths = [
        root / "manifests" / f"kitti_raw_{args.camera}_{args.size}x{args.size}_frames.csv",
        root / "manifests" / f"kitti_raw_{args.camera}_{args.size}x{args.size}_transitions.csv",
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as f:
            row_count = sum(1 for _ in f) - 1
        print(f"[ok] {path} rows={row_count}")
    transition_path = paths[1]
    with transition_path.open() as f:
        reader = csv.DictReader(f)
        first = next(reader)
    for key in ("frame_t_path", "frame_t_plus_1_path"):
        arr = np.load(first[key])
        if arr.shape != (args.size, args.size):
            raise RuntimeError(f"{key} shape {arr.shape}, expected {(args.size, args.size)}")
        if arr.dtype != np.float32:
            raise RuntimeError(f"{key} dtype {arr.dtype}, expected float32")
        print(f"[ok] {key}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.3f} max={arr.max():.3f}")


def write_bdd_note(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    ensure_dir(root / "downloads" / "bdd100k")
    note = root / "downloads" / "bdd100k" / "README_MANUAL_DOWNLOAD.txt"
    note.write_text(
        "BDD100K is a phase-2 dataset for this project. Download is manual from the official portal.\n"
        "Official homepage: https://www.bdd100k.com/\n"
        "Official docs/toolkit: https://doc.bdd100k.com/ and https://github.com/bdd100k/bdd100k\n"
        "Use videos only after the KITTI proof-of-concept is working; full videos are very large.\n",
        encoding="utf-8",
    )
    print(f"[note] {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(default_root()), help="Dataset root outside git")
    parser.add_argument("--size", type=int, default=32, help="Preprocessed square frame size")
    parser.add_argument("--camera", default=KITTI_CAMERA, help="KITTI camera stream, default image_00")
    parser.add_argument("--force", action="store_true", help="Redownload/re-extract existing files")

    sub = parser.add_subparsers(dest="command", required=True)

    kitti = sub.add_parser("prepare-kitti", help="Download/extract/preprocess KITTI Raw starter subset")
    kitti.add_argument("--date", default=KITTI_DEFAULT_DATE)
    kitti.add_argument("--drives", nargs="+", default=list(KITTI_STARTER_DRIVES))
    kitti.set_defaults(func=prepare_kitti)

    davis = sub.add_parser("download-davis", help="Download DAVIS 2017 trainval 480p held-out dataset")
    davis.add_argument("--extract", action="store_true", help="Extract the DAVIS archive after download")
    davis.set_defaults(func=download_davis)

    davis_prep = sub.add_parser("prepare-davis", help="Download/extract/preprocess DAVIS 2017 trainval 480p")
    davis_prep.set_defaults(func=prepare_davis)

    bdd = sub.add_parser("write-bdd-note", help="Create BDD100K manual-download note")
    bdd.set_defaults(func=write_bdd_note)

    check = sub.add_parser("verify", help="Verify generated KITTI manifests and sample arrays")
    check.set_defaults(func=verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
