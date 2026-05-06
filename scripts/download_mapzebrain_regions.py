#!/usr/bin/env python3
"""Fetch mapZebrain region metadata and optionally download region STL meshes.

Default behavior is metadata-only and writes a JSON manifest. STL downloads are
only performed when `--download` is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_BASE = "https://api.mapzebrain.org/api/"
MEDIA_BASE = "https://api.mapzebrain.org"
DEFAULT_SOURCE_DIR = Path("external/atlases/zebrafish/source")
USER_AGENT = "codextmp-atlas-handoff/1.0"


def parse_args() -> argparse.Namespace:
    """Parse downloader command-line options."""
    parser = argparse.ArgumentParser(
        description="Save mapZebrain API metadata and optionally download region STLs."
    )
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--media-base", default=MEDIA_BASE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output JSON manifest path. Defaults to <source-dir>/mapzebrain_manifest.json.",
    )
    parser.add_argument(
        "--region-version",
        default=None,
        help="Region version display name. Defaults to mapZebrain's newest version.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download region STL files. Omitted by default to avoid large transfers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call APIs and write the manifest, but do not download STL files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Download only first N regions.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between downloads.")
    return parser.parse_args()


def endpoint_url(api_base: str, endpoint: str) -> str:
    """Build a mapZebrain API endpoint URL with the trailing slash it expects."""
    return urllib.parse.urljoin(api_base.rstrip("/") + "/", endpoint.strip("/") + "/")


def post_json(api_base: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to a mapZebrain endpoint and return the decoded response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url(api_base, endpoint),
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def safe_slug(value: str, fallback: str) -> str:
    """Return a filesystem-safe ASCII slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return slug[:80] or fallback


def absolute_media_url(media_base: str, media_path: str) -> str:
    """Convert a mapZebrain media path to a quoted absolute URL."""
    if media_path.startswith("http://") or media_path.startswith("https://"):
        raw_url = media_path
    else:
        raw_url = urllib.parse.urljoin(media_base.rstrip("/") + "/", media_path.lstrip("/"))
    parts = urllib.parse.urlsplit(raw_url)
    quoted_path = urllib.parse.quote(parts.path, safe="/%()")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, quoted_path, parts.query, parts.fragment))


def region_stl_path(region: dict[str, Any]) -> str | None:
    """Return the preferred STL media path for one region record."""
    downloads = region.get("downloads") or {}
    if downloads.get("stl"):
        return str(downloads["stl"])
    visualization = region.get("visualization_data") or {}
    mesh = (visualization.get("_3d") or {}).get("mesh_file")
    return str(mesh) if mesh else None


def extract_regions(region_response: dict[str, Any], media_base: str) -> list[dict[str, Any]]:
    """Flatten mapZebrain region dictionary records for manifest output."""
    data = region_response.get("data") or {}
    dictionary = data.get("regions_dictionary") or {}
    records: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for region_id in sorted(dictionary, key=lambda key: int(key) if str(key).isdigit() else str(key)):
        region = dictionary[region_id]
        name = str(region.get("name") or region_id)
        stl_path = region_stl_path(region)
        filename_base = safe_slug(f"{region_id}_{name}", f"region_{region_id}")
        filename = f"{filename_base}.stl"
        index = 2
        while filename in used_names:
            filename = f"{filename_base}_{index}.stl"
            index += 1
        used_names.add(filename)

        records.append(
            {
                "id": region.get("id", region_id),
                "name": name,
                "short_name": region.get("short_name"),
                "parent": region.get("parent"),
                "is_container": bool(region.get("is_container", False)),
                "version": region.get("version"),
                "stl_path": stl_path,
                "stl_url": absolute_media_url(media_base, stl_path) if stl_path else None,
                "local_file": f"regions/{filename}" if stl_path else None,
                "download_status": "metadata_only" if stl_path else "no_stl_path",
            }
        )
    return records


def extract_brain_meshes(brain_response: dict[str, Any], media_base: str) -> list[dict[str, Any]]:
    """Extract whole-brain mesh metadata from `load_3D_brain_data`."""
    three_d = ((brain_response.get("data") or {}).get("threeD") or {})
    meshes: list[dict[str, Any]] = []
    for key in sorted(three_d):
        item = three_d[key]
        mesh_path = ((item.get("visualization_data") or {}).get("_3d") or {}).get("mesh_file")
        meshes.append(
            {
                "key": key,
                "name": item.get("name"),
                "mesh_path": mesh_path,
                "mesh_url": absolute_media_url(media_base, mesh_path) if mesh_path else None,
            }
        )
    return meshes


def download_file(url: str, destination: Path, overwrite: bool) -> str:
    """Download one URL to disk and return a manifest status string."""
    if destination.exists() and not overwrite:
        return "exists"
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    except urllib.error.URLError as exc:
        return f"error: {exc}"
    return "downloaded"


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a stable JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Fetch metadata, optionally download region STLs, and write a manifest."""
    args = parse_args()
    source_dir = args.source_dir
    manifest_path = args.manifest or source_dir / "mapzebrain_manifest.json"
    regions_dir = source_dir / "regions"

    newest = post_json(args.api_base, "load_newest_region_version_data", {})
    version_data = newest.get("versionData") or {}
    region_version = args.region_version or version_data.get("display_name") or "2.0 (MECE, 2024)"
    regions_response = post_json(
        args.api_base,
        "load_region_data_by_version",
        {"regionVersion": region_version},
    )
    brain_response = post_json(args.api_base, "load_3D_brain_data", {})

    regions = extract_regions(regions_response, args.media_base)
    downloadable = [region for region in regions if region.get("stl_url") and region.get("local_file")]
    to_download = downloadable[: args.limit] if args.limit is not None else downloadable

    if args.download and not args.dry_run:
        for region in to_download:
            destination = source_dir / str(region["local_file"])
            region["download_status"] = download_file(str(region["stl_url"]), destination, args.overwrite)
            if args.sleep > 0:
                time.sleep(args.sleep)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base": args.api_base,
        "media_base": args.media_base,
        "endpoints": {
            "newest_region_version": endpoint_url(args.api_base, "load_newest_region_version_data"),
            "region_data_by_version": endpoint_url(args.api_base, "load_region_data_by_version"),
            "brain_3d_data": endpoint_url(args.api_base, "load_3D_brain_data"),
        },
        "requested_region_version": region_version,
        "newest_region_version": version_data,
        "download_requested": bool(args.download),
        "dry_run": bool(args.dry_run or not args.download),
        "regions_dir": str(regions_dir),
        "counts": {
            "regions_total": len(regions),
            "regions_with_stl": len(downloadable),
            "download_attempts": len(to_download) if args.download and not args.dry_run else 0,
        },
        "brain_meshes": extract_brain_meshes(brain_response, args.media_base),
        "regions": regions,
    }
    write_manifest(manifest_path, manifest)

    print(
        "Saved manifest to "
        f"{manifest_path} with {len(regions)} regions; "
        f"downloads attempted: {manifest['counts']['download_attempts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
