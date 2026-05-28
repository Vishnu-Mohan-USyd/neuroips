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
import json
import math
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


L4E_PER_SITE = 16
DEFAULT_GABOR_ORIENTATIONS_DEG = (0.0, 45.0, 90.0, 135.0)
DEFAULT_GABOR_PHASES_RAD = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)


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


def bilinear_resize(frame: np.ndarray, out_size: int) -> np.ndarray:
    """Resize a 2D normalized frame to the V1 sheet grid without extra deps."""
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {frame.shape}")
    in_h, in_w = frame.shape
    if in_h == out_size and in_w == out_size:
        return frame.astype(np.float32, copy=False)
    y = np.linspace(0.0, float(in_h - 1), out_size, dtype=np.float32)
    x = np.linspace(0.0, float(in_w - 1), out_size, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    top = (1.0 - wx)[None, :] * frame[y0[:, None], x0[None, :]] + wx[None, :] * frame[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx)[None, :] * frame[y1[:, None], x0[None, :]] + wx[None, :] * frame[y1[:, None], x1[None, :]]
    return ((1.0 - wy)[:, None] * top + wy[:, None] * bottom).astype(np.float32)


def make_gabor_bank(*, kernel_size: int, sigma: float, wavelength: float) -> np.ndarray:
    """Return deterministic orientation x phase simple-cell kernels.

    The 16 channels per site mirror the model's L4E layout as four orientations
    and four contrast phases. Kernels are zero-mean/unit-norm so the drive scale
    argument controls current magnitude independently of kernel size.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("--kernel-size must be an odd integer >= 3")
    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    gaussian = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)

    kernels: list[np.ndarray] = []
    for orientation_deg in DEFAULT_GABOR_ORIENTATIONS_DEG:
        theta = math.radians(orientation_deg)
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        for phase in DEFAULT_GABOR_PHASES_RAD:
            kernel = gaussian * np.cos((2.0 * math.pi * x_theta / wavelength) + phase)
            kernel = kernel.astype(np.float32)
            kernel -= np.float32(kernel.mean())
            norm = float(np.sqrt(np.sum(kernel * kernel)))
            if norm < 1e-8:
                raise RuntimeError("Degenerate Gabor kernel")
            kernels.append((kernel / norm).astype(np.float32))
    return np.stack(kernels, axis=0)


def manifest_frame_column(fieldnames: list[str] | None, requested: str) -> str:
    if requested != "auto":
        return requested
    if not fieldnames:
        raise ValueError("Manifest has no header")
    for candidate in ("processed_frame_path", "frame_t_path", "frame_t_plus_1_path"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "Could not infer frame path column; pass --frame-path-column "
        "(expected processed_frame_path, frame_t_path, or frame_t_plus_1_path)"
    )


def load_manifest_frame_paths(args: argparse.Namespace) -> list[tuple[int, Path]]:
    manifest = Path(args.manifest).expanduser().resolve()
    rows: list[tuple[int, Path]] = []
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        column = manifest_frame_column(reader.fieldnames, args.frame_path_column)
        for row_index, row in enumerate(reader):
            if args.split and row.get("split") != args.split:
                continue
            raw_path = row.get(column, "")
            if not raw_path:
                raise ValueError(f"Manifest row {row_index} has empty {column}")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            rows.append((row_index, path))
            if args.max_frames and len(rows) >= args.max_frames:
                break
    if not rows:
        raise ValueError("No frames selected from manifest")
    return rows


def frame_to_l4e_drive(
    frame: np.ndarray,
    kernels: np.ndarray,
    *,
    sheet_side: int,
    drive_scale: float,
    drive_offset: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    if sheet_side <= 0:
        raise ValueError("--sheet-side must be positive")
    resized = bilinear_resize(frame.astype(np.float32, copy=False), sheet_side)
    radius = kernels.shape[1] // 2
    padded = np.pad(resized, radius, mode="reflect")
    drive = np.empty(sheet_side * sheet_side * L4E_PER_SITE, dtype=np.float32)

    for site_y in range(sheet_side):
        for site_x in range(sheet_side):
            patch = padded[site_y : site_y + kernels.shape[1], site_x : site_x + kernels.shape[2]]
            responses = np.einsum("kij,ij->k", kernels, patch, optimize=True)
            currents = drive_offset + drive_scale * np.maximum(responses, 0.0)
            site_id = site_y * sheet_side + site_x
            start = site_id * L4E_PER_SITE
            drive[start : start + L4E_PER_SITE] = np.clip(currents, clip_min, clip_max)
    return drive.astype(np.float32, copy=False)


def precompute_l4_drive(args: argparse.Namespace) -> None:
    """Precompute opt-in natural-video L4E current frames for GeNN replay."""
    output_bin = Path(args.output_bin).expanduser().resolve()
    output_manifest = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest
        else output_bin.with_suffix(output_bin.suffix + ".csv")
    )
    output_meta = (
        Path(args.output_meta).expanduser().resolve()
        if args.output_meta
        else output_bin.with_suffix(output_bin.suffix + ".json")
    )
    if args.clip_max <= args.clip_min:
        raise ValueError("--clip-max must be greater than --clip-min")
    if args.drive_scale < 0.0:
        raise ValueError("--drive-scale must be non-negative")

    frame_paths = load_manifest_frame_paths(args)
    kernels = make_gabor_bank(kernel_size=args.kernel_size, sigma=args.sigma, wavelength=args.wavelength)
    k_num_l4e = args.sheet_side * args.sheet_side * L4E_PER_SITE
    ensure_dir(output_bin.parent)

    manifest_rows: list[dict[str, object]] = []
    with output_bin.open("wb") as out:
        for frame_index, (source_row_index, frame_path) in enumerate(frame_paths):
            if not frame_path.exists():
                raise FileNotFoundError(frame_path)
            frame = np.load(frame_path)
            drive = frame_to_l4e_drive(
                frame,
                kernels,
                sheet_side=args.sheet_side,
                drive_scale=args.drive_scale,
                drive_offset=args.drive_offset,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
            )
            if drive.shape != (k_num_l4e,):
                raise RuntimeError(f"Internal drive shape {drive.shape}, expected {(k_num_l4e,)}")
            byte_offset = frame_index * k_num_l4e * np.dtype(np.float32).itemsize
            drive.tofile(out)
            manifest_rows.append(
                {
                    "frame_index": frame_index,
                    "source_row_index": source_row_index,
                    "source_frame_path": str(frame_path),
                    "drive_bin_path": str(output_bin),
                    "byte_offset": byte_offset,
                    "frame_size_float32": k_num_l4e,
                    "sheet_side": args.sheet_side,
                    "l4e_per_site": L4E_PER_SITE,
                    "k_num_l4e": k_num_l4e,
                    "drive_min": f"{float(drive.min()):.9g}",
                    "drive_max": f"{float(drive.max()):.9g}",
                    "drive_mean": f"{float(drive.mean()):.9g}",
                    "drive_std": f"{float(drive.std()):.9g}",
                }
            )

    write_csv(output_manifest, manifest_rows)
    meta = {
        "command": "precompute-l4-drive",
        "frame_count": len(manifest_rows),
        "drive_bin_path": str(output_bin),
        "manifest_path": str(output_manifest),
        "sheet_side": args.sheet_side,
        "l4e_per_site": L4E_PER_SITE,
        "k_num_l4e": k_num_l4e,
        "dtype": "float32",
        "filter_bank": {
            "type": "zero_mean_unit_norm_gabor",
            "orientation_deg": list(DEFAULT_GABOR_ORIENTATIONS_DEG),
            "phase_rad": list(DEFAULT_GABOR_PHASES_RAD),
            "kernel_size": args.kernel_size,
            "sigma": args.sigma,
            "wavelength": args.wavelength,
        },
        "drive_scale": args.drive_scale,
        "drive_offset": args.drive_offset,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "source_manifest": str(Path(args.manifest).expanduser().resolve()),
        "frame_path_column": args.frame_path_column,
        "split": args.split,
    }
    ensure_dir(output_meta.parent)
    output_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[l4-drive] {output_bin} frames={len(manifest_rows)} shape=({len(manifest_rows)}, {k_num_l4e})")
    print(f"[manifest] {output_manifest}")
    print(f"[meta] {output_meta}")


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

    l4_drive = sub.add_parser(
        "precompute-l4-drive",
        help="Convert normalized frame/transition manifests into raw float32 L4E replay drives",
    )
    l4_drive.add_argument("--manifest", required=True, help="Frame or transition CSV manifest")
    l4_drive.add_argument("--output-bin", required=True, help="Raw float32 output path")
    l4_drive.add_argument("--output-manifest", help="Per-frame drive manifest CSV")
    l4_drive.add_argument("--output-meta", help="JSON metadata path")
    l4_drive.add_argument("--sheet-side", type=int, default=32, help="V1 sheet side used by the model")
    l4_drive.add_argument("--max-frames", type=int, default=0, help="Optional frame cap; 0 means all selected")
    l4_drive.add_argument("--split", help="Optional manifest split filter, e.g. train or val")
    l4_drive.add_argument(
        "--frame-path-column",
        default="auto",
        help="Manifest path column, or auto for processed_frame_path/frame_t_path/frame_t_plus_1_path",
    )
    l4_drive.add_argument("--kernel-size", type=int, default=7, help="Odd Gabor kernel size in sheet sites")
    l4_drive.add_argument("--sigma", type=float, default=1.15, help="Gabor Gaussian sigma in sheet sites")
    l4_drive.add_argument("--wavelength", type=float, default=4.0, help="Gabor wavelength in sheet sites")
    l4_drive.add_argument("--drive-scale", type=float, default=0.25, help="Half-wave response to current scale")
    l4_drive.add_argument("--drive-offset", type=float, default=0.0, help="Additive current offset after filtering")
    l4_drive.add_argument("--clip-min", type=float, default=0.0, help="Minimum output current")
    l4_drive.add_argument("--clip-max", type=float, default=1.0, help="Maximum output current")
    l4_drive.set_defaults(func=precompute_l4_drive)

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
