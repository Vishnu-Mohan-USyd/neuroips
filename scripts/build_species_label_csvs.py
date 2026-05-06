#!/usr/bin/env python3
"""Build species atlas label catalogs from staged metadata.

The script uses only Python standard-library modules and writes CSV files with a
shared schema for downstream GLB mesh/catalog joining. It does not modify raw
atlas source files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


CSV_FIELDS = [
    "species",
    "source_asset",
    "source_mesh_file",
    "glb_mesh_name",
    "structure_id",
    "acronym",
    "label",
    "group",
    "parent_id",
    "hemisphere",
    "catalog_status",
    "notes",
]

XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate species atlas catalog CSVs from staged metadata."
    )
    parser.add_argument("--atlas-root", type=Path, default=Path("external/atlases"))
    return parser.parse_args()


def ascii_text(value: Any) -> str:
    """Return a compact ASCII-only string for CSV output."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def slug(value: Any, fallback: str = "mesh") -> str:
    """Return a stable ASCII identifier for expected mesh/object names."""
    text = ascii_text(value)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return clean or fallback


def row(**values: Any) -> dict[str, str]:
    """Create one normalized CSV row with the shared schema."""
    return {field: ascii_text(values.get(field, "")) for field in CSV_FIELDS}


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> int:
    """Write rows to a CSV path and return the row count."""
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def relative_source(species_root: Path, path: Path) -> str:
    """Return a source path relative to one species atlas directory."""
    return ascii_text(path.relative_to(species_root))


def hemisphere_name(value: Any) -> str:
    """Map common atlas hemisphere ids to names when known."""
    return {1: "left", 2: "right", 3: "bilateral"}.get(value, ascii_text(value))


def flatten_mouse_ontology(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten Allen ontology JSON and preserve path-derived group labels."""
    flat: dict[str, dict[str, Any]] = {}

    def visit(node: dict[str, Any], path_names: list[str], path_ids: list[str]) -> None:
        node_id = str(node.get("id", ""))
        names = path_names + [ascii_text(node.get("name", ""))]
        ids = path_ids + [node_id]
        group = names[1] if len(names) > 1 else ""
        flat[node_id] = {
            "node": node,
            "group": group,
            "path": " > ".join(name for name in names if name),
            "path_ids": "/".join(id_ for id_ in ids if id_),
        }
        for child in node.get("children", []) or []:
            visit(child, names, ids)

    visit(root, [], [])
    return flat


def build_mouse_catalog(atlas_root: Path) -> tuple[Path, int]:
    """Build a mouse catalog by joining CCF OBJ stems to structure ontology ids."""
    species_root = atlas_root / "mouse"
    graph_path = species_root / "source" / "structure_graph_1.json"
    mesh_dir = species_root / "source" / "structure_meshes"
    output = species_root / "catalog_mouse_labels.csv"
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    root_node = data["msg"][0]
    ontology = flatten_mouse_ontology(root_node)

    rows: list[dict[str, str]] = []
    for mesh_path in sorted(mesh_dir.glob("*.obj"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        structure_id = mesh_path.stem
        record = ontology.get(structure_id)
        node = record["node"] if record else {}
        rows.append(
            row(
                species="mouse",
                source_asset="source/structure_graph_1.json;source/structure_meshes",
                source_mesh_file=relative_source(species_root, mesh_path),
                glb_mesh_name=slug(mesh_path.stem),
                structure_id=structure_id,
                acronym=node.get("acronym", ""),
                label=node.get("name", ""),
                group=record.get("group", "") if record else "",
                parent_id=node.get("parent_structure_id", ""),
                hemisphere=hemisphere_name(node.get("hemisphere_id")) if node else "",
                catalog_status="mesh_with_ontology" if record else "mesh_without_ontology",
                notes=f"ontology_path={record['path']}" if record else "No matching ontology id.",
            )
        )
    return output, write_csv(output, rows)


def parse_itksnap_label_file(path: Path) -> list[tuple[str, str]]:
    """Parse ITK-SNAP label rows as `(structure_id, label)` pairs."""
    pattern = re.compile(
        r'^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+[-+0-9.]+\s+\d+\s+\d+\s+"(.*)"\s*$'
    )
    records: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            records.append((match.group(1), match.group(2)))
    return records


def build_rat_catalog(atlas_root: Path) -> tuple[Path, int]:
    """Build a rat catalog from the WHS ITK-SNAP label file."""
    species_root = atlas_root / "rat"
    label_path = species_root / "source" / "WHS_SD_rat_atlas_v4.01.label"
    volume_path = species_root / "source" / "WHS_SD_rat_atlas_v4.01.nii.gz"
    output = species_root / "catalog_rat_labels.csv"

    rows = []
    for structure_id, label in parse_itksnap_label_file(label_path):
        status = "background_label" if structure_id == "0" else "label_from_itksnap"
        rows.append(
            row(
                species="rat",
                source_asset=relative_source(species_root, label_path),
                source_mesh_file=relative_source(species_root, volume_path),
                glb_mesh_name="" if structure_id == "0" else slug(f"{structure_id}_{label}"),
                structure_id=structure_id,
                acronym="",
                label=label,
                group="",
                parent_id="",
                hemisphere="",
                catalog_status=status,
                notes="Volume label id; derive STL/OBJ before GLB packing.",
            )
        )
    return output, write_csv(output, rows)


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Read XLSX shared strings."""
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("x:si", XLSX_NS):
        parts = [text.text or "" for text in item.iter(f"{{{XLSX_NS['x']}}}t")]
        strings.append("".join(parts))
    return strings


def column_index(cell_ref: str) -> int:
    """Return zero-based column index from an Excel cell reference."""
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + ord(char.upper()) - 64
    return index - 1


def read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    """Read the first XLSX worksheet with enough support for atlas label tables."""
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet_names = sorted(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))
        if not sheet_names:
            return []
        root = ElementTree.fromstring(archive.read(sheet_names[0]))

    rows: list[list[str]] = []
    for row_node in root.findall(".//x:sheetData/x:row", XLSX_NS):
        values: list[str] = []
        for cell in row_node.findall("x:c", XLSX_NS):
            index = column_index(cell.get("r", "A"))
            while len(values) <= index:
                values.append("")
            cell_type = cell.get("t")
            if cell_type == "inlineStr":
                text_parts = [text.text or "" for text in cell.iter(f"{{{XLSX_NS['x']}}}t")]
                values[index] = "".join(text_parts)
                continue
            value_node = cell.find("x:v", XLSX_NS)
            if value_node is None or value_node.text is None:
                values[index] = ""
            elif cell_type == "s":
                values[index] = shared_strings[int(value_node.text)]
            else:
                values[index] = value_node.text
        rows.append(values)
    return rows


def marmoset_volume_for_block(species_root: Path, block_name: str) -> tuple[str, str]:
    """Return staged source volume and status for a marmoset workbook block."""
    zip_source = "source/NIH_Marmoset_Atlas_V1_master.zip"
    zip_root = "NIH_Marmoset_Atlas_V1-master/v1.1"
    zip_members = {
        "NIH_cortex_vL": "NIH_cortex_vL_150um.nii.gz",
        "NIH_cortex_vM": "NIH_cortex_vM_150um.nii.gz",
        "NIH_cortex_vH": "NIH_cortex_vH_150um.nii.gz",
    }
    for block_prefix, member_name in zip_members.items():
        if block_name.startswith(block_prefix):
            return f"{zip_source}:{zip_root}/{member_name}", "label_from_xlsx_volume"

    if "Paxinos" in block_name:
        volume = species_root / "source" / "NIH_cortex_vPaxinos_150um.nii.gz"
    elif "subcortical_beta" in block_name:
        volume = species_root / "source" / "NIH_subcortical_beta_150um.nii.gz"
    else:
        return "", "label_from_xlsx_volume_missing"
    return relative_source(species_root, volume), "label_from_xlsx_volume"


def build_marmoset_catalog(atlas_root: Path) -> tuple[Path, int]:
    """Build a marmoset catalog from the staged NIH labels XLSX workbook."""
    species_root = atlas_root / "marmoset"
    workbook = species_root / "source" / "NIH_labels_cortex_and_subcortical.xlsx"
    output = species_root / "catalog_marmoset_labels.csv"
    rows_data = read_xlsx_first_sheet(workbook)
    output_rows: list[dict[str, str]] = []

    if len(rows_data) < 3:
        return output, write_csv(
            output,
            [
                row(
                    species="marmoset",
                    source_asset=relative_source(species_root, workbook),
                    catalog_status="xlsx_no_label_rows",
                    notes="Workbook was present but no label table rows were parsed.",
                )
            ],
        )

    block_starts = [0, 3, 6, 9, 12]
    for start in block_starts:
        block_name = rows_data[0][start] if len(rows_data[0]) > start else ""
        if not block_name:
            continue
        source_mesh_file, status = marmoset_volume_for_block(species_root, block_name)
        for sheet_row in rows_data[2:]:
            name = sheet_row[start] if len(sheet_row) > start else ""
            acronym = sheet_row[start + 1] if len(sheet_row) > start + 1 else ""
            label_id = sheet_row[start + 2] if len(sheet_row) > start + 2 else ""
            if not label_id or not (name or acronym):
                continue
            output_rows.append(
                row(
                    species="marmoset",
                    source_asset=f"{relative_source(species_root, workbook)}:{block_name}",
                    source_mesh_file=source_mesh_file,
                    glb_mesh_name=slug(f"{label_id}_{acronym or name}"),
                    structure_id=label_id,
                    acronym=acronym,
                    label=name,
                    group=block_name,
                    parent_id="",
                    hemisphere="",
                    catalog_status=status,
                    notes=(
                        "Merged NIH Marmoset Atlas v1.1 workbook row; group is parcellation block. Matching NIfTI volume available."
                        if source_mesh_file
                        else "Merged NIH Marmoset Atlas v1.1 workbook row; matching NIfTI volume needs archive inspection."
                    ),
                )
            )
    return output, write_csv(output, output_rows)


def build_zebrafish_catalog(atlas_root: Path) -> tuple[Path, int]:
    """Build a zebrafish catalog from the staged mapZebrain manifest."""
    species_root = atlas_root / "zebrafish"
    manifest_path = species_root / "source" / "mapzebrain_manifest.json"
    output = species_root / "catalog_zebrafish_regions.csv"
    if not manifest_path.exists():
        return output, write_csv(
            output,
            [
                row(
                    species="zebrafish",
                    source_asset=relative_source(species_root, manifest_path),
                    catalog_status="manifest_missing",
                    notes="Run scripts/download_mapzebrain_regions.py before building this catalog.",
                )
            ],
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_rows: list[dict[str, str]] = []
    for region in manifest.get("regions", []):
        local_file = region.get("local_file") or ""
        stl_path = region.get("stl_path") or ""
        version = region.get("version") or {}
        output_rows.append(
            row(
                species="zebrafish",
                source_asset=relative_source(species_root, manifest_path),
                source_mesh_file=local_file or stl_path,
                glb_mesh_name=slug(Path(local_file).stem if local_file else f"{region.get('id')}_{region.get('name')}"),
                structure_id=region.get("id", ""),
                acronym=region.get("short_name", ""),
                label=region.get("name", ""),
                group=version.get("display_name", ""),
                parent_id=region.get("parent", ""),
                hemisphere="",
                catalog_status=region.get("download_status", "metadata_only"),
                notes=f"is_container={bool(region.get('is_container'))}; stl_path={stl_path}",
            )
        )
    return output, write_csv(output, output_rows)


def macaque_surface_index(archive: tarfile.TarFile) -> dict[str, str]:
    """Index D99 right-hemisphere GIFTI surface paths by structure id."""
    surfaces: dict[str, str] = {}
    pattern = re.compile(r"D99_v2\..+\.k(\d+)\.gii$")
    for member in archive.getmembers():
        if not member.isfile() or "/surfs_right/" not in member.name:
            continue
        match = pattern.search(member.name)
        if match and match.group(1) not in surfaces:
            surfaces[match.group(1)] = member.name
    return surfaces


def build_macaque_catalog(atlas_root: Path) -> tuple[Path, int]:
    """Build a macaque catalog from D99 labels inside the staged tgz archive."""
    species_root = atlas_root / "macaque_prototype"
    archive_path = species_root / "source" / "D99_v2.0_dist.tgz"
    output = species_root / "catalog_macaque_prototype_labels.csv"
    label_member = "D99_v2.0_dist/D99_v2.0_labels_semicolon.txt"
    right_volume_member = "D99_v2.0_dist/D99_atlas_v2.0_right.nii.gz"

    rows: list[dict[str, str]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        surfaces = macaque_surface_index(archive)
        label_file = archive.extractfile(label_member)
        if label_file is None:
            return output, write_csv(
                output,
                [
                    row(
                        species="macaque_prototype",
                        source_asset=f"{relative_source(species_root, archive_path)}:{label_member}",
                        catalog_status="archive_label_file_missing",
                        notes="D99 archive needs manual inspection before catalog generation.",
                    )
                ],
            )
        decoded = label_file.read().decode("utf-8", errors="replace").splitlines()

    for record in csv.reader(decoded, delimiter=";"):
        if len(record) < 3:
            continue
        structure_id, acronym, label = record[:3]
        group_parts = [part for part in record[3:] if part]
        surface_path = surfaces.get(structure_id)
        source_member = surface_path or right_volume_member
        source_mesh = f"{relative_source(species_root, archive_path)}:{source_member}"
        rows.append(
            row(
                species="macaque_prototype",
                source_asset=f"{relative_source(species_root, archive_path)}:{label_member}",
                source_mesh_file=source_mesh,
                glb_mesh_name=slug(Path(surface_path).stem if surface_path else f"{structure_id}_{acronym}"),
                structure_id=structure_id,
                acronym=acronym,
                label=label,
                group=" / ".join(group_parts),
                parent_id="",
                hemisphere="right" if surface_path else "",
                catalog_status=(
                    "archive_label_surface_right_available"
                    if surface_path
                    else "archive_label_volume_available"
                ),
                notes=(
                    "Right-hemisphere GIFTI surface available in D99 tgz; GLB still needs conversion."
                    if surface_path
                    else "No right-hemisphere GIFTI surface in D99 tgz; use right atlas volume member for extraction."
                ),
            )
        )
    return output, write_csv(output, rows)


def main() -> int:
    """Generate all feasible staged species catalogs."""
    args = parse_args()
    builders = [
        build_mouse_catalog,
        build_rat_catalog,
        build_zebrafish_catalog,
        build_marmoset_catalog,
        build_macaque_catalog,
    ]
    for builder in builders:
        path, count = builder(args.atlas_root)
        print(f"Wrote {count} rows: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
