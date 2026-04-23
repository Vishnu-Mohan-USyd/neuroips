"""Regression tests for :func:`scripts.v2.eval_kok._install_learned_w_q_gain`.

Task #74 β-Step-3: ``l23_e.W_q_gain`` is a non-persistent buffer, so the
Phase-3 Kok trainer writes it as a top-level ckpt key outside
``state_dict``. Earlier, ``scripts/v2/eval_kok.py`` used only the generic
:func:`_gates_common.load_checkpoint`, which restores ``state_dict`` but
ignores top-level keys — so any ``--enable-w-q-gain-rule`` ckpt was being
evaluated with ``W_q_gain`` silently reset to the default (all-1.0) init,
defeating the β mechanism during eval.

These tests lock in the two contracts of the new
``_install_learned_w_q_gain`` helper:

1. **Learned-ckpt path.** When the checkpoint contains a top-level
   ``W_q_gain`` tensor, the helper installs it bit-exactly onto
   ``bundle.net.l23_e.W_q_gain`` and returns ``True``.
2. **Backward-compat path.** When the checkpoint has no such key (a
   pre-β ckpt: phase-2 or legacy phase-3), the helper is a no-op —
   ``W_q_gain`` stays at its init value of 1.0 — and returns ``False``.

No real training run is needed: we hand-build a tiny synthetic ckpt in a
tmpdir from a freshly constructed :class:`V2Network` (same pattern used
by ``tests/v2/test_c_load_bearing_check_harness.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.v2._gates_common import load_checkpoint
from scripts.v2.eval_kok import _install_learned_w_q_gain
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _write_base_checkpoint(
    cfg, tmp_path: Path, *, seed: int = 42, name: str = "ckpt.pt",
) -> tuple[Path, V2Network]:
    """Write a minimal Phase-3-shaped ckpt without a top-level W_q_gain key.

    Returns the path and the source network so callers can re-use its
    ``l23_e.W_q_gain`` shape when synthesising a learned tensor.
    """
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.eval()
    path = tmp_path / name
    torch.save({
        "step": 0,
        "state_dict": net.state_dict(),
        "phase": "phase3_kok",
        "frozen_sha": net.frozen_sensory_core_sha(),
    }, path)
    return path, net


def test_install_learned_w_q_gain_copies_bit_exactly(cfg, tmp_path):
    """β-trained ckpt: helper installs learned W_q_gain bit-exactly."""
    path, src_net = _write_base_checkpoint(
        cfg, tmp_path, name="beta_ckpt.pt",
    )
    # Synthesise a "learned" W_q_gain: same shape as the live buffer, with
    # deliberately non-unity values so the assertion is meaningful (can't
    # accidentally pass by coincidence with the default init of 1.0).
    gain_shape = tuple(src_net.l23_e.W_q_gain.shape)
    torch.manual_seed(1234)
    learned = 0.1 + 0.8 * torch.rand(gain_shape, dtype=torch.float32)
    # Force row 0 and row 1 to distinct mean values so the verification
    # print would differ if the helper ever copied only part of the tensor.
    learned[0] = 0.35
    learned[1] = 0.55
    # Re-write the ckpt with the extra top-level key.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["W_q_gain"] = learned.clone()
    torch.save(payload, path)

    bundle = load_checkpoint(path, seed=42, device="cpu")
    # Precondition: before helper, W_q_gain is at init (all 1.0), not the
    # learned tensor. This proves load_checkpoint alone is insufficient.
    assert torch.all(bundle.net.l23_e.W_q_gain == 1.0), (
        "precondition broken: load_checkpoint unexpectedly restored "
        "W_q_gain from state_dict — helper would be redundant"
    )

    installed = _install_learned_w_q_gain(
        bundle, path, device="cpu", verbose=False,
    )

    assert installed is True, "helper should report install on β-ckpt"
    # Bit-exact match on the full tensor.
    assert torch.equal(bundle.net.l23_e.W_q_gain.detach().cpu(), learned), (
        "W_q_gain was not copied bit-exactly from the ckpt"
    )
    # Row-level sanity (mirrors the verbose stdout contract).
    assert float(bundle.net.l23_e.W_q_gain[0].mean().item()) == pytest.approx(
        0.35, abs=1e-6,
    )
    assert float(bundle.net.l23_e.W_q_gain[1].mean().item()) == pytest.approx(
        0.55, abs=1e-6,
    )


def test_install_learned_w_q_gain_noop_on_pre_beta_ckpt(cfg, tmp_path):
    """Pre-β ckpt: helper is a no-op, W_q_gain stays at 1.0 init."""
    path, _ = _write_base_checkpoint(cfg, tmp_path, name="pre_beta_ckpt.pt")
    # Sanity: the ckpt as saved must NOT carry a W_q_gain top-level key,
    # otherwise the test doesn't exercise the backward-compat path.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert "W_q_gain" not in raw, (
        "fixture invariant: pre-β ckpt must not contain a top-level "
        "W_q_gain key"
    )

    bundle = load_checkpoint(path, seed=42, device="cpu")
    # Precondition: live buffer is at its default init of 1.0.
    assert torch.all(bundle.net.l23_e.W_q_gain == 1.0)

    installed = _install_learned_w_q_gain(
        bundle, path, device="cpu", verbose=False,
    )

    assert installed is False, (
        "helper should return False on a ckpt without W_q_gain"
    )
    # Live buffer unchanged — still at init.
    assert torch.all(bundle.net.l23_e.W_q_gain == 1.0), (
        "W_q_gain changed despite ckpt having no W_q_gain key"
    )


def test_install_learned_w_q_gain_rejects_shape_mismatch(cfg, tmp_path):
    """Shape-mismatched W_q_gain must raise rather than silently truncate."""
    path, src_net = _write_base_checkpoint(
        cfg, tmp_path, name="bad_shape_ckpt.pt",
    )
    bad_shape = (src_net.l23_e.W_q_gain.shape[0] + 1,
                 src_net.l23_e.W_q_gain.shape[1])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["W_q_gain"] = torch.zeros(bad_shape, dtype=torch.float32)
    torch.save(payload, path)

    bundle = load_checkpoint(path, seed=42, device="cpu")
    with pytest.raises(ValueError, match="W_q_gain shape mismatch"):
        _install_learned_w_q_gain(bundle, path, device="cpu", verbose=False)
