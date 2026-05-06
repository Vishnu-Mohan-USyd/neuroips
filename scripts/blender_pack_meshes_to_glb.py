#!/usr/bin/env python3
"""Pack OBJ/STL atlas meshes into a binary GLB with Blender.

Run with Blender so the `bpy` module is available:

    blender -b --python scripts/blender_pack_meshes_to_glb.py -- \
      --input-dir external/atlases/mouse/source/structure_meshes \
      --patterns "*.obj" \
      --output assets/brain_mouse.glb \
      --recenter
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_SUFFIXES = {".obj", ".stl"}


def blender_argv(argv: Sequence[str]) -> list[str]:
    """Return arguments after Blender's `--` separator."""
    if "--" not in argv:
        return []
    return list(argv[argv.index("--") + 1 :])


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line options intended for this script."""
    parser = argparse.ArgumentParser(
        description="Import OBJ/STL meshes as separate Blender objects and export one GLB."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.obj", "*.stl"],
        help="Glob patterns relative to --input-dir.",
    )
    parser.add_argument(
        "--decimate-ratio",
        type=float,
        default=1.0,
        help="Optional mesh decimation ratio in (0, 1]. Default preserves meshes.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform object scale applied before export. Default preserves units.",
    )
    parser.add_argument(
        "--recenter",
        action="store_true",
        help="Translate imported objects so the combined bounds are centered at origin.",
    )
    parser.add_argument(
        "--set-origin",
        choices=["none", "geometry", "bounds"],
        default="none",
        help="Optional origin adjustment per imported object.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import only the first N sorted meshes for preview builds.",
    )
    return parser.parse_args(argv)


def discover_meshes(input_dir: Path, patterns: Iterable[str], limit: int | None) -> list[Path]:
    """Return sorted OBJ/STL files matching the requested glob patterns."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    meshes: dict[Path, None] = {}
    for pattern in patterns:
        for path in input_dir.glob(pattern):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                meshes[path.resolve()] = None

    paths = sorted(meshes)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No OBJ/STL meshes found in {input_dir}")
    return paths


def sanitize_name(stem: str, used: set[str]) -> str:
    """Make a stable Blender object name while preserving the file stem."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    if not name:
        name = "mesh"
    base = name[:55]
    name = base
    index = 2
    while name in used:
        suffix = f"_{index}"
        name = f"{base[: 63 - len(suffix)]}{suffix}"
        index += 1
    used.add(name)
    return name


def clear_scene(bpy: object) -> None:
    """Remove Blender's default scene contents."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh_file(bpy: object, path: Path) -> list[object]:
    """Import one mesh file and return the newly created Blender objects."""
    before_names = {obj.name for obj in bpy.data.objects}
    suffix = path.suffix.lower()
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh suffix: {path.suffix}")
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def rename_imported_objects(objects: Sequence[object], path: Path, used: set[str]) -> None:
    """Assign stable names and record the source path on each imported object."""
    if len(objects) == 1:
        objects[0].name = sanitize_name(path.stem, used)
        objects[0]["source_file"] = str(path)
        return

    base = sanitize_name(path.stem, used)
    for index, obj in enumerate(objects, start=1):
        obj.name = sanitize_name(f"{base}_{index}", used)
        obj["source_file"] = str(path)


def apply_uniform_scale(objects: Iterable[object], scale: float) -> None:
    """Apply a uniform transform scale without changing source mesh data."""
    if scale <= 0:
        raise ValueError("--scale must be positive")
    if scale == 1.0:
        return
    for obj in objects:
        obj.scale = (obj.scale[0] * scale, obj.scale[1] * scale, obj.scale[2] * scale)


def set_origins(bpy: object, objects: Sequence[object], mode: str) -> None:
    """Optionally move object origins to geometry or bounds centers."""
    if mode == "none":
        return
    center = "MEDIAN" if mode == "geometry" else "BOUNDS"
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center=center)
        obj.select_set(False)


def apply_decimation(bpy: object, objects: Sequence[object], ratio: float) -> None:
    """Apply an optional decimation modifier to mesh objects."""
    if not 0 < ratio <= 1:
        raise ValueError("--decimate-ratio must be in (0, 1]")
    if ratio == 1.0:
        return

    for obj in objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        modifier = obj.modifiers.new(name="atlas_decimate", type="DECIMATE")
        modifier.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def combined_bounds_center(objects: Sequence[object]) -> tuple[float, float, float]:
    """Compute the center of all object world-space bounding boxes."""
    from mathutils import Vector  # type: ignore[import-not-found]

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                mins[axis] = min(mins[axis], world[axis])
                maxs[axis] = max(maxs[axis], world[axis])
    return tuple((mins[axis] + maxs[axis]) / 2.0 for axis in range(3))


def recenter_objects(objects: Sequence[object], center: tuple[float, float, float]) -> None:
    """Translate objects so the combined bounding-box center lands at origin."""
    for obj in objects:
        obj.location.x -= center[0]
        obj.location.y -= center[1]
        obj.location.z -= center[2]


def export_glb(bpy: object, output: Path) -> None:
    """Export the current scene as a binary GLB file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Blender import, optional simplification, and GLB export."""
    args = parse_args(blender_argv(sys.argv) if argv is None else argv)

    import bpy  # type: ignore[import-not-found]

    mesh_paths = discover_meshes(args.input_dir, args.patterns, args.limit)
    clear_scene(bpy)

    used_names: set[str] = set()
    imported_objects: list[object] = []
    for path in mesh_paths:
        objects = import_mesh_file(bpy, path)
        rename_imported_objects(objects, path, used_names)
        imported_objects.extend(objects)

    apply_uniform_scale(imported_objects, args.scale)
    bpy.context.view_layer.update()
    set_origins(bpy, imported_objects, args.set_origin)
    apply_decimation(bpy, imported_objects, args.decimate_ratio)
    bpy.context.view_layer.update()

    if args.recenter:
        recenter_objects(imported_objects, combined_bounds_center(imported_objects))

    export_glb(bpy, args.output)
    print(f"Exported {len(imported_objects)} objects from {len(mesh_paths)} files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
