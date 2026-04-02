"""Artifact and manifest helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid

from .provenance import (
    ARTIFACT_SCHEMA_VERSION,
    BENCHMARK_REGISTRY_VERSION,
    CONTRACT_VERSION,
    METRIC_SCHEMA_VERSION,
    write_run_provenance,
)


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    config_name: str
    contract_version: str = CONTRACT_VERSION
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    metric_schema_version: str = METRIC_SCHEMA_VERSION
    benchmark_registry_version: str = BENCHMARK_REGISTRY_VERSION
    train_objectives: list[str] = field(default_factory=list)
    heldout_assays: list[str] = field(default_factory=list)
    lineage: dict[str, Any] = field(
        default_factory=lambda: {
            "parent_run_ids": [],
            "child_run_ids": [],
            "benchmark_anchor_refs": [],
        }
    )
    notes: dict[str, Any] = field(default_factory=dict)


class RunStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        config_name: str,
        *,
        train_objectives: list[str] | None = None,
        heldout_assays: list[str] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> tuple[RunManifest, Path]:
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.root_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "eval").mkdir(exist_ok=True)
        (run_dir / "ablations").mkdir(exist_ok=True)
        manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            config_name=config_name,
            train_objectives=list(train_objectives or []),
            heldout_assays=list(heldout_assays or []),
            notes=dict(notes or {}),
        )
        self.write_manifest(run_dir, manifest)
        return manifest, run_dir

    def reserve_run_path(self, config_name: str) -> tuple[RunManifest, Path]:
        return self.create_run(config_name)

    @staticmethod
    def write_provenance(
        run_dir: str | Path,
        manifest: RunManifest,
        *,
        resolved_config: dict[str, Any],
        parent_run_ids: list[str] | None = None,
        benchmark_anchor_refs: list[str] | None = None,
        child_run_ids: list[str] | None = None,
    ) -> None:
        manifest.lineage = {
            "parent_run_ids": list(parent_run_ids or manifest.lineage.get("parent_run_ids", [])),
            "child_run_ids": list(child_run_ids or manifest.lineage.get("child_run_ids", [])),
            "benchmark_anchor_refs": list(
                benchmark_anchor_refs or manifest.lineage.get("benchmark_anchor_refs", [])
            ),
        }
        RunStore.write_manifest(run_dir, manifest)
        write_run_provenance(run_dir, resolved_config=resolved_config)

    @staticmethod
    def write_manifest(run_dir: str | Path, manifest: RunManifest) -> None:
        manifest_path = Path(run_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

    @staticmethod
    def load_manifest(run_dir: str | Path) -> RunManifest:
        payload = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
        return RunManifest(**payload)
