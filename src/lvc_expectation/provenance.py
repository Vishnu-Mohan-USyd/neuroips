"""Versioned provenance helpers for run artifacts and frozen benchmark bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import platform
import subprocess
import sys

import torch


CONTRACT_VERSION = "2026-04-01.stage0.freeze.v1"
ARTIFACT_SCHEMA_VERSION = "2026-04-01.stage0.artifacts.v1"
METRIC_SCHEMA_VERSION = "2026-04-01.metrics.v1"
BENCHMARK_REGISTRY_VERSION = "2026-04-01.frozen-benchmarks.v1"
FROZEN_BENCHMARK_METRIC_VERSION = "benchmark_v1_frozen"

FROZEN_METRIC_KEY_DESCRIPTIONS = {
    "poststimulus_l23_template_specificity": "Latent target-aligned specificity on l23_readout during visible post-stimulus steps.",
    "prestimulus_template_specificity": "Template specificity on blank prestimulus steps scored against the generator-side expected target.",
    "probe_target_aligned_specificity_contrast": "Pair-balanced expected-minus-unexpected latent target-aligned specificity contrast on the local-global probe.",
    "probe_pooled_target_aligned_specificity_contrast": "Pair-balanced expected-minus-unexpected pooled target-aligned specificity contrast from gaussian_orientation_bank on the local-global probe.",
    "probe_context_comparator_nonuniformity_contrast": "Pair-balanced expected-minus-unexpected context-comparator nonuniformity contrast on probe steps.",
    "learned_probe_alignment_kl": "KL divergence from the learned context prediction to the generator-side expected distribution on scored probe rows.",
    "oracle_probe_alignment_kl": "KL divergence from the oracle context prediction to the generator-side expected distribution on scored probe rows.",
    "learned_probe_expected_logprob": "Expected log-probability assigned by the learned context path to the generator-side expected probe distribution.",
    "oracle_probe_expected_logprob": "Expected log-probability assigned by the oracle context path to the generator-side expected probe distribution.",
    "learned_probe_expected_target_top1_rate": "Rate at which the learned probe prediction top-1 matches the generator-side expected target bin.",
    "oracle_probe_expected_target_top1_rate": "Rate at which the oracle probe prediction top-1 matches the generator-side expected target bin.",
    "learned_probe_pair_flip_rate": "Rate at which learned probe predictions flip top-1 identity across the paired local-global contexts.",
    "oracle_probe_pair_flip_rate": "Rate at which oracle probe predictions flip top-1 identity across the paired local-global contexts.",
    "learned_probe_pair_flip_symmetry_consistency": "Consistency of learned probe pair-flip behavior across symmetry-mated probe pairs.",
    "oracle_probe_pair_flip_symmetry_consistency": "Consistency of oracle probe pair-flip behavior across symmetry-mated probe pairs.",
}

DEFAULT_LINEAGE = {
    "parent_run_ids": [],
    "child_run_ids": [],
    "benchmark_anchor_refs": [],
}

_FINGERPRINT_EXCLUDED_FILENAMES = frozenset({"run_fingerprint.json"})


@dataclass(frozen=True)
class BenchmarkAnchor:
    """Frozen benchmark bundle reference.

    Attributes
    ----------
    anchor_id:
        Stable benchmark identifier used in future provenance links.
    description:
        Human-readable meaning of the frozen bundle.
    classification:
        Human-readable frozen benchmark family classification.
    run_ids:
        Frozen run identifiers that comprise the benchmark bundle.
    bundle_fingerprint:
        SHA256 hash over the bundle's run-level fingerprints.
    """

    anchor_id: str
    description: str
    classification: str
    run_ids: tuple[str, ...]
    bundle_fingerprint: str
    metric_version: str = FROZEN_BENCHMARK_METRIC_VERSION
    subsets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    subset_fingerprints: dict[str, str] = field(default_factory=dict)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _json_safe(payload: object) -> object:
    if isinstance(payload, tuple):
        return [_json_safe(item) for item in payload]
    if isinstance(payload, list):
        return [_json_safe(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _json_safe(value) for key, value in payload.items()}
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot(cwd: Path) -> dict[str, object]:
    def _run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip() or None

    status = _run_git("status", "--short")
    return {
        "head": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(status),
        "status_short": status.splitlines()[:50] if status else [],
    }


def build_environment_snapshot(*, cwd: str | Path | None = None) -> dict[str, object]:
    """Return a small, JSON-safe environment snapshot for run provenance."""

    resolved_cwd = Path.cwd() if cwd is None else Path(cwd)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(resolved_cwd),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "git": _git_snapshot(resolved_cwd),
    }


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compute_run_fingerprint(run_dir: str | Path) -> str:
    """Compute a stable SHA256 over all substantive files in a run directory."""

    resolved_run_dir = Path(run_dir)
    file_rows: list[dict[str, str]] = []
    for path in sorted(path for path in resolved_run_dir.rglob("*") if path.is_file()):
        if path.name in _FINGERPRINT_EXCLUDED_FILENAMES:
            continue
        file_rows.append(
            {
                "path": path.relative_to(resolved_run_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": file_rows,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def compute_bundle_fingerprint(root_dir: str | Path, run_ids: list[str] | tuple[str, ...]) -> str:
    """Compute a stable SHA256 for a frozen benchmark bundle."""

    resolved_root = Path(root_dir)
    normalized_run_ids = sorted(dict.fromkeys(run_ids))
    payload = {
        "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
        "runs": [
            {
                "run_id": run_id,
                "run_fingerprint": compute_run_fingerprint(resolved_root / "runs" / run_id),
            }
            for run_id in normalized_run_ids
        ],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def write_run_provenance(
    run_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    environment: dict[str, object] | None = None,
) -> dict[str, str]:
    """Write config/environment snapshots plus a run fingerprint artifact."""

    resolved_run_dir = Path(run_dir)
    write_json(resolved_run_dir / "resolved_config.json", resolved_config)
    write_json(resolved_run_dir / "environment.json", environment or build_environment_snapshot())
    run_fingerprint = compute_run_fingerprint(resolved_run_dir)
    run_fingerprint_payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
    }
    write_json(resolved_run_dir / "run_fingerprint.json", run_fingerprint_payload)
    return run_fingerprint_payload


FROZEN_BENCHMARK_REGISTRY: dict[str, BenchmarkAnchor] = {
    "phase1_poststim_positive_v1": BenchmarkAnchor(
        anchor_id="phase1_poststim_positive_v1",
        description="Frozen phase-1 post-stim positive bundle.",
        classification="phase-1 post-stim positive",
        run_ids=(
            "7ca381644851",
            "c6d4f8c043b9",
            "ff56b2f53b0b",
            "48865dcda306",
            "340c4bbb2344",
        ),
        bundle_fingerprint="418ed5f6b0dc46554c891f0616358f3d7a26dba7380eacddae4da9025594c1d3",
    ),
    "phase1_prestim_negative_v1": BenchmarkAnchor(
        anchor_id="phase1_prestim_negative_v1",
        description="Frozen phase-1 prestim negative bundle.",
        classification="phase-1 prestim negative",
        run_ids=(
            "30dada327952",
            "1dd98eedc9d6",
            "76e83c341ce0",
            "fe27da082ca4",
            "1fb218e43bde",
        ),
        bundle_fingerprint="df62e83794323c69baf0290130127a6b775667e42f67f4bb3063a9106df10a1d",
    ),
    "phase15_probe_positive_v1": BenchmarkAnchor(
        anchor_id="phase15_probe_positive_v1",
        description="Frozen phase-1.5 decisive positive bundle.",
        classification="phase-1.5 decisive positive",
        run_ids=(
            "4fee7e5512c6",
            "15dc413c7865",
            "c146791dd90b",
            "436f74d942e1",
            "42fb81e97a17",
        ),
        bundle_fingerprint="678997de60354242260507b923b080cb0f94b6afda944c76e0323742d5c3f27e",
    ),
    "phase2_p2a_primary_negative_v1": BenchmarkAnchor(
        anchor_id="phase2_p2a_primary_negative_v1",
        description="Frozen phase-2 source-heldout negative bundle.",
        classification="phase-2 source-heldout primary negative",
        run_ids=(
            "db825bcc8783",
            "67143175a1fc",
            "ceb6a059a2c2",
            "96f3bfb7e7bf",
            "be9e669b0503",
            "a841ee2a0b1b",
            "f41c2371bab9",
            "85d5bf823847",
            "b5a6b46212f8",
            "052e87658432",
        ),
        bundle_fingerprint="cbc877478ba8024b9817382698ef08bb339f695afbe88d91a7546890b45e5daf",
        subsets={
            "direction_a": (
                "db825bcc8783",
                "67143175a1fc",
                "ceb6a059a2c2",
                "96f3bfb7e7bf",
                "be9e669b0503",
            ),
            "direction_b": (
                "a841ee2a0b1b",
                "f41c2371bab9",
                "85d5bf823847",
                "b5a6b46212f8",
                "052e87658432",
            ),
        },
        subset_fingerprints={
            "direction_a": "249ca1ec5593c8e5347d4fee21aafff54e35cf9845c55835b1da8acfd69ededf",
            "direction_b": "91da3a1fd12c622f59108eeb8bf6a3c63d58263efcad88a01ccf7eda8b8cbaa4",
        },
    ),
    "phase2_challenger_selection_negative_v1": BenchmarkAnchor(
        anchor_id="phase2_challenger_selection_negative_v1",
        description="Frozen failed challenger-selection sweep bundle set.",
        classification="phase-2 challenger-selection negative",
        run_ids=(
            "3004568f8330",
            "3d05cd635cb7",
            "721aa64dccc1",
            "1db8d7fd390b",
            "2683a48377da",
            "4e4a0c0ea202",
            "5a43af8c321d",
            "e07e225cce09",
            "15ad2d17439e",
            "196b2aeaf00e",
            "47a61c8bcca5",
            "5ccab54c7776",
            "1d38e6e713bf",
            "7eaf59c34ea4",
            "b338ab57aaf8",
            "c103041e796f",
        ),
        bundle_fingerprint="4493ace2d997c7da834456d2043830066c736cbfd56e803fc5d055063f57af33",
    ),
}


def frozen_benchmark_registry_payload() -> dict[str, object]:
    """Return the frozen benchmark registry as a JSON-safe payload."""

    return {
        "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
        "anchors": {
            anchor_id: _json_safe(asdict(anchor))
            for anchor_id, anchor in sorted(FROZEN_BENCHMARK_REGISTRY.items())
        },
    }


def frozen_benchmark_fingerprints_payload() -> dict[str, object]:
    """Return the frozen benchmark fingerprints artifact payload."""

    return {
        "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
        "fingerprints": {
            anchor_id: {
                "bundle_fingerprint": anchor.bundle_fingerprint,
                **(
                    {"subset_fingerprints": dict(sorted(anchor.subset_fingerprints.items()))}
                    if anchor.subset_fingerprints
                    else {}
                ),
            }
            for anchor_id, anchor in sorted(FROZEN_BENCHMARK_REGISTRY.items())
        },
    }


def frozen_metric_versions_payload() -> dict[str, object]:
    """Return the frozen benchmark metric-version registry payload."""

    return {
        "benchmark_registry_version": BENCHMARK_REGISTRY_VERSION,
        "metric_versions": {
            metric_key: {
                "version": "v1",
                "benchmark_metric_version": FROZEN_BENCHMARK_METRIC_VERSION,
                "semantic_description": description,
                "status": "frozen_v1_anchor",
            }
            for metric_key, description in sorted(FROZEN_METRIC_KEY_DESCRIPTIONS.items())
        },
        "notes": "Stage 0 freezes metric keys as v1 anchors only. No v2 metrics are introduced here.",
    }


def verify_frozen_benchmark_registry(root_dir: str | Path) -> dict[str, bool]:
    """Verify the live artifact tree against the frozen benchmark registry."""

    resolved_root = Path(root_dir)
    results: dict[str, bool] = {}
    for anchor_id, anchor in FROZEN_BENCHMARK_REGISTRY.items():
        anchor_ok = compute_bundle_fingerprint(resolved_root, anchor.run_ids) == anchor.bundle_fingerprint
        subset_ok = all(
            compute_bundle_fingerprint(resolved_root, subset_run_ids) == anchor.subset_fingerprints[subset_name]
            for subset_name, subset_run_ids in anchor.subsets.items()
        )
        results[anchor_id] = anchor_ok and subset_ok
    return results
