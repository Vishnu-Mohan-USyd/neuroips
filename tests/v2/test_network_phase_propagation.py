"""``V2Network.set_phase`` must propagate to every phase-aware child.

Training drivers switch phase at the top of the network; each
sub-module's ``plastic_weight_names()`` / ``frozen_weight_names()``
manifests depend on the child's own recorded phase. Missing propagation
was a real bug class in v1 — this test catches any regression.
"""

from __future__ import annotations

import pytest

from src.v2_model.network import V2Network


@pytest.fixture
def net(cfg):
    return V2Network(cfg, token_bank=None, seed=42)


@pytest.mark.parametrize(
    "phase", ["phase2", "phase3_kok", "phase3_richter"],
)
def test_set_phase_propagates_to_all_children(net, phase):
    """Every child with a ``phase`` attribute ends up at the new phase."""
    net.set_phase(phase)
    assert net.phase == phase

    for name, child in net.named_children():
        if name == "lgn_l4":
            continue  # has no phase concept
        child_phase = getattr(child, "phase", None)
        assert child_phase == phase, (
            f"child {name!r} stuck at phase {child_phase!r} after set_phase({phase!r})"
        )


def test_set_phase_rejects_unknown(net):
    """Unknown phase strings fail loudly on at least one child."""
    with pytest.raises(ValueError):
        net.set_phase("phase_invalid")
