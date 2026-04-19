"""Per-phase freeze manifest + boundary assertion.

Motivation (from scope-impact analysis §4(c)): the current codebase's Phase 1→2
transition used a blanket `net.parameters().requires_grad_(False)` which silently
dropped newly-added modules (alpha_net, VIP, deep_template — Phase 2.4.1 trap).
The v4 spec has **two** training phases + ablations + frozen-core SHA checks, so
a silent-freeze bug there would be even harder to debug.

This module implements an **explicit manifest**: for each phase, every named
parameter must be classified as `plastic` or `frozen`. At a phase boundary the
assertion verifies that:

  1. Every parameter in the network appears in the manifest (no silent drops).
  2. Parameters tagged `frozen` have `requires_grad == False` AND their
     `.data.clone()` hash is unchanged across the boundary.
  3. Parameters tagged `plastic` have `requires_grad == True`.

Real implementation lands once `network.py` exists. Scaffold here gives the
dataclass + yaml loader + assertion signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PhaseFreezeSpec:
    """Per-phase classification of every named parameter in the network.

    Attributes:
        phase_name: Canonical name ("phase_2_predictive", "phase_3_kok",
            "phase_3_richter", "null_control", etc.).
        plastic: Iterable of fully-qualified parameter names that are allowed to
            update during this phase (must have `requires_grad=True`).
        frozen: Iterable of fully-qualified parameter names that must be inert
            during this phase (`requires_grad=False` and bitwise-unchanged
            across the phase boundary).
    """
    phase_name: str
    plastic: list[str] = field(default_factory=list)
    frozen: list[str] = field(default_factory=list)


@dataclass
class FreezeManifest:
    """Full manifest across all phases.

    Loaded from yaml in `config/v2/freeze_manifest.yaml`. Storage format TBD
    in Task #9; scaffold here accepts a dict-of-dicts.
    """
    phases: dict[str, PhaseFreezeSpec] = field(default_factory=dict)


def load_manifest(yaml_path: Path | str) -> FreezeManifest:
    """Load a freeze manifest from yaml.

    Args:
        yaml_path: Path to yaml file matching the schema
            `{phase_name: {plastic: [...], frozen: [...]}}`.

    Raises:
        NotImplementedError: Scaffold. yaml schema + parser in Task #9.
    """
    raise NotImplementedError(
        "load_manifest is a scaffold; implement once config/v2/freeze_manifest.yaml "
        "schema is finalised (Task #9)."
    )


def assert_boundary(
    net: Any,
    manifest: FreezeManifest,
    phase_name: str,
    prior_checksums: dict[str, str] | None = None,
) -> dict[str, str]:
    """Verify that at a phase boundary the network matches the manifest.

    Checks performed (once implemented):
      1. Exhaustiveness — every `net.named_parameters()` appears in either the
         `plastic` or `frozen` list of `phase_name`.
      2. requires_grad flags match classification.
      3. If `prior_checksums` is provided, every `frozen` parameter's SHA-256
         hash matches the prior value (catches silent in-place writes).

    Args:
        net: The network module to check.
        manifest: Parsed freeze manifest.
        phase_name: Which phase to validate against.
        prior_checksums: Optional {param_name: sha256_hex} from a prior boundary;
            used to assert frozen params didn't drift.

    Returns:
        Fresh {param_name: sha256_hex} for use as the next `prior_checksums`.

    Raises:
        NotImplementedError: Scaffold.
        AssertionError: (once implemented) on any classification mismatch.
    """
    raise NotImplementedError(
        "assert_boundary is a scaffold; implement once network.py exposes "
        "named_parameters() (Task #11+)."
    )


def iter_expected_param_names(arch_config: Any) -> Iterable[str]:
    """Canonical list of every parameter the v2 network will expose.

    Used for manifest validation: if the network grows a new parameter that
    isn't in this list, `assert_boundary` should flag it as unclassified.

    Raises:
        NotImplementedError: Scaffold.
    """
    raise NotImplementedError(
        "iter_expected_param_names is a scaffold; implement once network.py "
        "defines the canonical parameter names (Task #11+)."
    )
