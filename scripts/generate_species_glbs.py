#!/usr/bin/env python3
"""Generate runtime species GLB files from staged atlas sources.

This is a production artifact generator: outputs are binary GLB files under
`assets/` with node names matching `frontend_data/species_catalogs/*.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tarfile
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import trimesh
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
ATLAS_ROOT = ROOT / "external" / "atlases"
CATALOG_ROOT = ROOT / "frontend_data" / "species_catalogs"
ASSET_ROOT = ROOT / "assets"

DEFAULT_MAX_FACES_BY_SPECIES: dict[str, int] = {
    "mouse": 250,
    "rat": 1500,
    "marmoset": 1500,
    "macaque": 1500,
    "zebrafish": 500,
}
SIMPLIFICATION_AGGRESSION = 10


def load_catalog(species: str) -> dict:
    return json.loads((CATALOG_ROOT / f"{species}.json").read_text())


def catalog_mesh_names(species: str) -> set[str]:
    data = load_catalog(species)
    return set(data["regionsByMeshName"].keys())


def slug_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-") or "mesh"


def recenter_and_scale(scene: trimesh.Scene, target_size: float = 160.0) -> trimesh.Scene:
    bounds = scene.bounds
    if bounds is None or not np.isfinite(bounds).all():
        return scene
    center = bounds.mean(axis=0)
    size = bounds[1] - bounds[0]
    max_dim = float(size.max()) if size.size else 0.0
    scale = target_size / max_dim if max_dim > 0 else 1.0
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    scene.apply_transform(transform)
    return scene


def simplify_mesh(mesh: trimesh.Trimesh, max_faces: int | None) -> trimesh.Trimesh:
    if max_faces is None or len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces, aggression=SIMPLIFICATION_AGGRESSION)
    except Exception:
        return mesh


def add_named_mesh(scene: trimesh.Scene, mesh: trimesh.Trimesh, name: str, max_faces: int | None) -> None:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh = simplify_mesh(mesh, max_faces)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=[190, 190, 190, 255])
    scene.add_geometry(mesh, node_name=name, geom_name=name)


def export_scene(scene: trimesh.Scene, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    recenter_and_scale(scene)
    output.write_bytes(scene.export(file_type="glb"))


def generate_mouse(max_faces: int | None) -> Path:
    scene = trimesh.Scene()
    mesh_dir = ATLAS_ROOT / "mouse" / "source" / "structure_meshes"
    expected = catalog_mesh_names("mouse")
    for name in sorted(expected, key=lambda value: int(value) if value.isdigit() else value):
        path = mesh_dir / f"{name}.obj"
        mesh = trimesh.load_mesh(path, file_type="obj", process=False)
        add_named_mesh(scene, mesh, name, max_faces)
    output = ASSET_ROOT / "brain_mouse.glb"
    export_scene(scene, output)
    return output


def read_itksnap_labels(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        if '"' not in line:
            continue
        prefix, label_part = line.split('"', 1)
        parts = prefix.split()
        if not parts or not parts[0].isdigit():
            continue
        label = label_part.rsplit('"', 1)[0].strip()
        rows.append((int(parts[0]), label))
    return rows


def volume_mesh(
    data: np.ndarray,
    label_value: int,
    spacing: tuple[float, float, float],
    max_faces: int | None,
) -> trimesh.Trimesh | None:
    mask = data == label_value
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, np.array(mask.shape))
    cropped = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    padded = np.pad(cropped.astype(np.uint8), 1)
    verts, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
        allow_degenerate=False,
    )
    verts += (lo - 1) * np.array(spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    return simplify_mesh(mesh, max_faces)


def load_nifti_array(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.int32)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data, spacing


def generate_rat(max_faces: int | None) -> Path:
    scene = trimesh.Scene()
    species = "rat"
    catalog = load_catalog(species)
    by_id = {int(r["structureId"]): r for r in catalog["regions"] if r.get("meshName")}
    volume_path = ATLAS_ROOT / "rat" / "source" / "WHS_SD_rat_atlas_v4.01.nii.gz"
    data, spacing = load_nifti_array(volume_path)
    for label_id in sorted(by_id):
        mesh = volume_mesh(data, label_id, spacing, max_faces)
        if mesh is None:
            continue
        add_named_mesh(scene, mesh, by_id[label_id]["meshName"], None)
    output = ASSET_ROOT / "brain_rat.glb"
    export_scene(scene, output)
    return output


def extract_zip_member(zip_path: Path, member: str, temp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(member, temp_dir)
    return temp_dir / member


def generate_marmoset(max_faces: int | None) -> Path:
    scene = trimesh.Scene()
    catalog = load_catalog("marmoset")
    zip_path = ATLAS_ROOT / "marmoset" / "source" / "NIH_Marmoset_Atlas_V1_master.zip"
    volume_cache: dict[str, tuple[np.ndarray, tuple[float, float, float]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for region in catalog["regions"]:
            source = region["sourceMeshFile"]
            if ":" in source:
                archive_name, member = source.split(":", 1)
                key = source
                if key not in volume_cache:
                    extracted = extract_zip_member(ATLAS_ROOT / "marmoset" / archive_name, member, tmp_path)
                    volume_cache[key] = load_nifti_array(extracted)
            else:
                key = source
                if key not in volume_cache:
                    volume_cache[key] = load_nifti_array(ATLAS_ROOT / "marmoset" / source)
            data, spacing = volume_cache[key]
            mesh = volume_mesh(data, int(float(region["structureId"])), spacing, max_faces)
            if mesh is not None:
                add_named_mesh(scene, mesh, region["meshName"], None)
    output = ASSET_ROOT / "brain_marmoset.glb"
    export_scene(scene, output)
    return output


def macaque_surface_members() -> dict[str, str]:
    archive_path = ATLAS_ROOT / "macaque_prototype" / "source" / "D99_v2.0_dist.tgz"
    members: dict[str, str] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or "/surfs_right/" not in member.name or not member.name.endswith(".gii"):
                continue
            stem = Path(member.name).stem
            members[stem] = member.name
            members[slug_name(stem)] = member.name
    return members


def generate_macaque(max_faces: int | None) -> Path:
    scene = trimesh.Scene()
    catalog = load_catalog("macaque_prototype")
    archive_path = ATLAS_ROOT / "macaque_prototype" / "source" / "D99_v2.0_dist.tgz"
    surface_members = macaque_surface_members()
    volume_cache: dict[str, tuple[np.ndarray, tuple[float, float, float]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive_path, "r:gz") as archive:
            for region in catalog["regions"]:
                name = region["meshName"]
                if region["status"] == "archive-surface":
                    member = surface_members.get(name)
                    if not member:
                        continue
                    archive.extract(member, tmp_path)
                    loaded = nib.load(str(tmp_path / member))
                    arrays = loaded.darrays
                    verts = np.asarray(arrays[0].data, dtype=float)
                    faces = np.asarray(arrays[1].data, dtype=np.int64)
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    add_named_mesh(scene, mesh, name, max_faces)
                elif region["status"] == "volume-backed":
                    source = region["sourceMeshFile"]
                    if ":" not in source:
                        continue
                    _archive_name, member = source.split(":", 1)
                    if member not in volume_cache:
                        archive.extract(member, tmp_path)
                        volume_cache[member] = load_nifti_array(tmp_path / member)
                    data, spacing = volume_cache[member]
                    mesh = volume_mesh(data, int(float(region["structureId"])), spacing, max_faces)
                    if mesh is not None:
                        add_named_mesh(scene, mesh, name, None)
    output = ASSET_ROOT / "brain_macaque.glb"
    export_scene(scene, output)
    return output


def download_zebrafish_stls(limit: int | None = None) -> None:
    import subprocess
    args = [
        "python3",
        str(ROOT / "scripts" / "download_mapzebrain_regions.py"),
        "--source-dir",
        str(ATLAS_ROOT / "zebrafish" / "source"),
        "--download",
    ]
    if limit is not None:
        args += ["--limit", str(limit)]
    subprocess.run(args, check=True, cwd=ROOT)


def generate_zebrafish(max_faces: int | None, download: bool) -> Path:
    if download:
        download_zebrafish_stls()
    scene = trimesh.Scene()
    catalog = load_catalog("zebrafish")
    source_dir = ATLAS_ROOT / "zebrafish" / "source"
    for region in catalog["regions"]:
        name = region.get("meshName")
        source = region.get("sourceMeshFile")
        if not name or not source:
            continue
        path = source_dir / source
        if not path.exists():
            continue
        mesh = trimesh.load_mesh(path, file_type="stl", process=False)
        add_named_mesh(scene, mesh, name, max_faces)
    output = ASSET_ROOT / "brain_zebrafish.glb"
    export_scene(scene, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("species", nargs="+", choices=["mouse", "rat", "marmoset", "macaque", "zebrafish"])
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Override the default per-mesh face target for all requested species.",
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=ATLAS_ROOT,
        help="Root containing staged atlas source files.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=ASSET_ROOT,
        help="Directory where generated GLB files are written.",
    )
    parser.add_argument("--download-zebrafish", action="store_true")
    return parser.parse_args()


def max_faces_for_species(species: str, override: int | None) -> int | None:
    return override if override is not None else DEFAULT_MAX_FACES_BY_SPECIES[species]


def main() -> int:
    global ATLAS_ROOT, ASSET_ROOT
    args = parse_args()
    ATLAS_ROOT = args.atlas_root
    ASSET_ROOT = args.asset_root
    for species in args.species:
        max_faces = max_faces_for_species(species, args.max_faces)
        if species == "mouse":
            output = generate_mouse(max_faces)
        elif species == "rat":
            output = generate_rat(max_faces)
        elif species == "marmoset":
            output = generate_marmoset(max_faces)
        elif species == "macaque":
            output = generate_macaque(max_faces)
        elif species == "zebrafish":
            output = generate_zebrafish(max_faces, args.download_zebrafish)
        print(f"wrote {output} {output.stat().st_size} bytes max_faces={max_faces}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
