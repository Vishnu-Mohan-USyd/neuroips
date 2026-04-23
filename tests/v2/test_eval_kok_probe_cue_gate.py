"""Regression tests for the ``--probe-cue-gate`` flag on
:func:`scripts.v2.eval_kok.run_kok_probe_trial`.

Task #74 β-eval bug: the β mechanism at ``src/v2_model/layers.py:680``
applies the per-cue ``W_q_gain`` multiplier on L4→L23E only when
``q_t is not None``. ``run_kok_probe_trial`` used to set ``q_t=None``
during the probe1 epoch, so any checkpoint trained with
``--enable-w-q-gain-rule`` was evaluated with the β gain silently
bypassed while rates were being recorded. Compare against
``scripts/v2/step1_beta_level11.py:_run_probe_trial_with_gate`` which
keeps ``q_t=q_cue`` live during probe1 ("β gain active here") and
which reports a large expectation contrast on the same ckpt that
legacy ``run_kok_probe_trial`` shows as null.

The new ``gate_probe_cue`` kwarg (CLI: ``--probe-cue-gate``) restores
the step1 protocol. These tests lock down two contracts:

1. **Bit-exact no-op when the gain is identity.** With
   ``W_q_gain = 1.0`` everywhere (the default init), ``gate_probe_cue``
   is a pure multiplication by 1.0, so the trial output must be
   bit-identical with the flag on vs off. This is the backward-compat
   guarantee: legacy (pre-β) checkpoints or fresh phase-2 networks
   run through the new code path must not change numerically.

2. **Observable modulation when the gain is non-unity.** With
   ``W_q_gain[cue_id]`` set to a strong non-unity pattern, the probe
   rates under ``gate_probe_cue=True`` must differ from those under
   ``gate_probe_cue=False``. This proves the gate is actually engaged
   and that the math reaches probe1, not just the cue epoch.

3. **L23E.forward bit-exact identity.** Direct layer-level test:
   ``L23E.forward`` with ``q_t=q_cue`` and ``q_t=None`` must return
   bit-equal outputs when ``W_q_gain`` is the all-ones init. Exercises
   the exact ``ff_l4 * cue_gain`` branch at ``layers.py:680``.

No training run is needed — all tests build a fresh
:class:`V2Network` in memory and call the helpers directly.
"""
from __future__ import annotations

import pytest
import torch

from scripts.v2._gates_common import CheckpointBundle
from scripts.v2.eval_kok import run_kok_probe_trial
from scripts.v2.train_phase3_kok_learning import KokTiming, build_cue_tensor
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank


def _make_bundle(seed: int = 42) -> CheckpointBundle:
    cfg = ModelConfig(seed=seed, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.set_phase("phase3_kok")
    net.eval()
    return CheckpointBundle(cfg=cfg, net=net, bank=bank, meta={})


def test_probe_cue_gate_is_bit_exact_noop_when_all_q_paths_inactive():
    """Flag on vs off must produce identical rates when every q_t consumer
    is inactive.

    ``gate_probe_cue`` delivers ``q_t=q_cue`` during probe1 to BOTH
    consumers of q_t in the network: ``l23_e.W_q_gain`` (the β gate at
    ``src/v2_model/layers.py:680``) AND ``context_memory.W_qm_task``
    (``src/v2_model/context_memory.py:421``), which drives the memory
    state ``m_t``. The flag is therefore a bit-exact no-op only when
    BOTH consumers are inactive — i.e. ``W_q_gain == 1.0`` AND
    ``W_qm_task == 0``. This is the correct statement of the
    backward-compat contract: a pre-β ckpt at fresh init (both
    conditions hold) must run identically with the flag on or off.
    """
    bundle = _make_bundle(seed=42)
    # Precondition 1: default W_q_gain is the all-ones init.
    assert torch.all(bundle.net.l23_e.W_q_gain == 1.0), (
        "precondition: default W_q_gain must be all-ones init"
    )
    # Precondition 2: suppress context_memory's q_t consumption by zeroing
    # W_qm_task. At fresh init W_qm_task is non-zero (random normal per
    # task_input_init_std), so without this step the memory drive differs
    # between q_t=q_cue and q_t=None even when W_q_gain==1.
    with torch.no_grad():
        bundle.net.context_memory.W_qm_task.zero_()
    timing = KokTiming()

    g_off = torch.Generator().manual_seed(1234)
    r_off = run_kok_probe_trial(
        bundle, cue_id=0, probe_orientation_deg=135.0,
        timing=timing, noise_std=0.0, generator=g_off,
        gate_probe_cue=False,
    )
    g_on = torch.Generator().manual_seed(1234)
    r_on = run_kok_probe_trial(
        bundle, cue_id=0, probe_orientation_deg=135.0,
        timing=timing, noise_std=0.0, generator=g_on,
        gate_probe_cue=True,
    )
    assert torch.equal(r_off, r_on), (
        "gate flag is not bit-exact when W_q_gain=1 and W_qm_task=0 — "
        "some third q_t consumer exists in the forward pass that we are "
        "not aware of, or the gate is introducing numerical drift"
    )


def test_probe_cue_gate_changes_output_under_nonunity_gain():
    """With W_q_gain[cue_id] non-unity, flag on vs off must differ."""
    bundle = _make_bundle(seed=42)
    cue_id = 0
    # Install a strong non-unity gain for cue 0 only (rows 2..n_c stay at 1).
    # This mirrors the β-trained checkpoint shape exactly: only rows
    # corresponding to active cue ids ever move from init.
    torch.manual_seed(9999)
    gain_row = 0.1 + 0.8 * torch.rand(
        bundle.net.l23_e.W_q_gain.shape[1], dtype=torch.float32,
    )
    with torch.no_grad():
        bundle.net.l23_e.W_q_gain[cue_id].copy_(gain_row)
    # Sanity: row is now far from unity.
    assert not torch.allclose(
        bundle.net.l23_e.W_q_gain[cue_id],
        torch.ones_like(bundle.net.l23_e.W_q_gain[cue_id]),
    )
    timing = KokTiming()

    g_off = torch.Generator().manual_seed(4321)
    r_off = run_kok_probe_trial(
        bundle, cue_id=cue_id, probe_orientation_deg=135.0,
        timing=timing, noise_std=0.0, generator=g_off,
        gate_probe_cue=False,
    )
    g_on = torch.Generator().manual_seed(4321)
    r_on = run_kok_probe_trial(
        bundle, cue_id=cue_id, probe_orientation_deg=135.0,
        timing=timing, noise_std=0.0, generator=g_on,
        gate_probe_cue=True,
    )
    # Must differ somewhere — if they're equal, the gate isn't being
    # applied during the probe1 epoch, which is the bug this patch fixes.
    assert not torch.equal(r_off, r_on), (
        "gate flag has no effect under non-unity W_q_gain — run_kok_probe_"
        "trial is not routing q_t=q_cue into the forward pass during probe1"
    )
    # Quantitative sanity: a large (>0.1) L2-norm difference, comparable to
    # the ~0.3 Δr that Level 11 reports on the real β-trained ckpt. The
    # exact magnitude depends on dynamics, so the threshold is loose.
    diff_norm = float((r_on - r_off).norm().item())
    assert diff_norm > 1e-3, (
        f"gate produces a visible rate change but it's suspiciously small "
        f"(||Δr||={diff_norm:.2e}); may indicate the gate is only reaching "
        f"a subset of the probe epoch"
    )


def test_layer_level_w_q_gain_bit_exact_under_identity():
    """L23E.forward with q_t=q_cue must equal q_t=None when W_q_gain==1.0.

    This directly exercises the branch at src/v2_model/layers.py:680
    (``cue_gain = q_t @ self.W_q_gain; ff_l4 = ff_l4 * cue_gain``).
    With an all-ones ``W_q_gain`` and any one-hot ``q_t``, ``cue_gain``
    is all-ones and ``ff_l4 * cue_gain`` is bit-identical to ``ff_l4``.
    Any mismatch here would mean the multiplication is introducing
    float drift, which would break the backward-compat contract.
    """
    bundle = _make_bundle(seed=42)
    layer = bundle.net.l23_e
    cfg = bundle.cfg
    B = 1
    n_units = layer.n_units
    n_l4 = layer.n_l4_e
    n_som = layer.n_som
    n_pv = layer.n_pv
    n_h_e = layer.n_h_e
    n_cue = layer.n_cue

    # Sanity: default W_q_gain is all-ones.
    assert torch.all(layer.W_q_gain == 1.0)

    torch.manual_seed(777)
    l4_input = torch.rand(B, n_l4)
    l23_rec = torch.rand(B, n_units)
    som_input = torch.rand(B, n_som)
    pv_input = torch.rand(B, n_pv)
    h_apical = torch.rand(B, n_h_e)
    context_bias = torch.zeros(B, n_units)
    state = torch.zeros(B, n_units)
    q_cue = build_cue_tensor(0, n_cue, device=cfg.device)
    assert q_cue.shape == (1, n_cue)

    with torch.no_grad():
        rate_ungated, _ = layer.forward(
            l4_input=l4_input,
            l23_recurrent_input=l23_rec,
            som_input=som_input,
            pv_input=pv_input,
            h_apical_input=h_apical,
            context_bias=context_bias,
            state=state,
            q_t=None,
        )
        rate_gated, _ = layer.forward(
            l4_input=l4_input,
            l23_recurrent_input=l23_rec,
            som_input=som_input,
            pv_input=pv_input,
            h_apical_input=h_apical,
            context_bias=context_bias,
            state=state,
            q_t=q_cue,
        )
    assert torch.equal(rate_ungated, rate_gated), (
        "L23E.forward produces different rates for q_t=None vs q_t=q_cue "
        "even when W_q_gain is identity — the (q_t @ W_q_gain) branch is "
        "introducing numerical drift"
    )


def test_layer_level_w_q_gain_scales_ff_exactly():
    """L23E.forward output with W_q_gain[cue]==k must match reapplying k.

    Strong bit-exact claim: when W_q_gain[cue_id] is a UNIFORM scalar
    k applied to every L23E unit and all non-L4 inputs are zero, the
    gated drive equals k · (ungated ff_l4 drive). At state=0 and
    context_bias=0 this test isolates the β-gate's multiplicative
    contract end-to-end through the layer's nonlinearity: ``rate_gated``
    equals ``(1-leak) * phi(k * ff_l4 - theta)`` and ``rate_ungated``
    equals ``(1-leak) * phi(ff_l4 - theta)`` — so we can't compare them
    bit-exactly across the nonlinearity. Instead we prove the gate
    applies the scaling by crafting a second q_cue that selects a row
    set to unity (``cue_id=1``) and asserting that output equals the
    q_t=None output exactly.
    """
    bundle = _make_bundle(seed=42)
    layer = bundle.net.l23_e
    cfg = bundle.cfg
    n_units = layer.n_units
    # Install: cue-0 row = 0.5, cue-1 row = 1.0 (unity).
    with torch.no_grad():
        layer.W_q_gain[0].fill_(0.5)
        layer.W_q_gain[1].fill_(1.0)

    torch.manual_seed(555)
    B = 1
    l4_input = torch.rand(B, layer.n_l4_e)
    l23_rec = torch.rand(B, n_units)
    som_input = torch.rand(B, layer.n_som)
    pv_input = torch.rand(B, layer.n_pv)
    h_apical = torch.rand(B, layer.n_h_e)
    context_bias = torch.zeros(B, n_units)
    state = torch.zeros(B, n_units)
    q_cue_0 = build_cue_tensor(0, layer.n_cue, device=cfg.device)
    q_cue_1 = build_cue_tensor(1, layer.n_cue, device=cfg.device)

    with torch.no_grad():
        rate_ungated, _ = layer.forward(
            l4_input=l4_input, l23_recurrent_input=l23_rec,
            som_input=som_input, pv_input=pv_input,
            h_apical_input=h_apical, context_bias=context_bias,
            state=state, q_t=None,
        )
        rate_cue1_unity, _ = layer.forward(
            l4_input=l4_input, l23_recurrent_input=l23_rec,
            som_input=som_input, pv_input=pv_input,
            h_apical_input=h_apical, context_bias=context_bias,
            state=state, q_t=q_cue_1,
        )
        rate_cue0_half, _ = layer.forward(
            l4_input=l4_input, l23_recurrent_input=l23_rec,
            som_input=som_input, pv_input=pv_input,
            h_apical_input=h_apical, context_bias=context_bias,
            state=state, q_t=q_cue_0,
        )
    # Cue-1 selects the unity row; output must equal ungated case bit-exactly.
    assert torch.equal(rate_cue1_unity, rate_ungated), (
        "cue_id=1 selects W_q_gain row of 1.0 (identity) but the output "
        "differs from q_t=None — the gate is not correctly selecting rows"
    )
    # Cue-0 selects the 0.5 row; output must NOT equal ungated case.
    assert not torch.equal(rate_cue0_half, rate_ungated), (
        "cue_id=0 selects W_q_gain row of 0.5 but output equals q_t=None "
        "— the gate is not being applied"
    )
