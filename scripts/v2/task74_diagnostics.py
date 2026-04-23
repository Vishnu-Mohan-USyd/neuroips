"""Task #74 Phase-0 diagnostics — three falsifiable scalars per checkpoint.

Purpose
-------
Produce one JSON per Phase-3 checkpoint carrying three orthogonal metrics
that directly probe the three root causes confirmed in Task #73:

1. **Coverage** — is the L2/3 preferred-orientation distribution spread across
   the 12 canonical orientation bins, or collapsed to a few modes?
   Metric: Shannon entropy (nats) of the preferred-orientation histogram.

2. **Rule magnitude** — does the trained task readout produce a bias whose
   magnitude is comparable to the localizer-evoked L2/3 response?
   Metric: ``||W_mh_task @ m_decision|| / ||r_l23_localizer||``, averaged
   across expected classes.

3. **Readout alignment** — is ``W_mh_task @ m`` aligned with the sensory
   tuning of the expected orientation, or arbitrary?
   Metric: Pearson(b_task_c, localizer_template_c), mean over expected
   classes.

Thresholds (Task #74 brief, ratified by Lead):
- coverage entropy PASS_STRONG > 1.74 nats (= 0.7 · ln 12),
                  PASS_WEAK   > 1.50 nats;
  and ≥ 8 of 12 orientation bins carry ≥ 5 % of units.
- rule_magnitude ratio > 0.05 (gate); aspirational > 1.7.
- readout_alignment cos > +0.30.

The CLI defaults to the Kok paradigm (2 expected classes = 45°, 135° — the
canonical anchors) for metrics 2+3, because that is the paradigm on which
the Task #73 diagnostics ran. Coverage is paradigm-independent (cue-free
localizer) and uses the full 12-orientation sweep.

Shape/unit contract
-------------------
- ``preferred_hist``   : list[int], length 12, units = # L2/3 E units.
- ``entropy_nats``     : float,  natural log of probabilities.
- ``entropy_bits``     : float,  log2 of probabilities.
- ``bias_added_norm``  : float,  ‖W_mh_task @ m_decision‖₂, per-trial mean.
- ``localizer_norm``   : float,  ‖r_l23 probe-epoch mean‖₂ at matched orient.
- ``cos_per_class_*``  : list[float], Pearson across L2/3 E units.

Randomness: controlled by ``--seed``; all torch and numpy RNG derived from it.

Exit code: 0 on success, nonzero on I/O or checkpoint error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import (
    CheckpointBundle, load_checkpoint,
    make_blank_frame, make_grating_frame,
)
from src.v2_model.context_memory import _SOM_GAIN_BIAS0, _SOM_GAIN_CLAMP_MAX
from src.v2_model.layers import _inhibitory_eff
from scripts.v2.eval_kok import (
    _compute_localizer_stats, run_kok_localizer_trial,
)
from scripts.v2.eval_richter import run_richter_localizer_trial
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)
from scripts.v2.train_phase3_richter_learning import (
    LEADER_TOKEN_IDX, N_LEAD_TRAIL, RichterTiming, TRAILER_TOKEN_IDX,
    build_leader_tensor, permutation_from_seed,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full 12-orientation sweep (15° spacing, [0°, 180°)) — canonical V1 basis.
LOCALIZER_ORIENTS_12 = np.linspace(0.0, 180.0, 12, endpoint=False).astype(np.float64)

# Lead-ratified thresholds (Task #74 brief).
THRESHOLDS: dict[str, float] = {
    "coverage_pass_strong_nats": 1.74,   # 0.7 · ln(12) ≈ 1.739
    "coverage_pass_weak_nats":   1.50,
    "coverage_n_bins_5pct":      8,      # bins with ≥ 5 % of units
    "rule_magnitude_gate":       0.05,
    "rule_magnitude_target":     1.70,   # aspirational: 0.5 · localizer_norm
    "readout_cos":               0.30,
}


# ---------------------------------------------------------------------------
# Coverage metric
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_coverage(
    bundle: CheckpointBundle,
    *,
    orients: np.ndarray = LOCALIZER_ORIENTS_12,
    n_trials_per_orient: int = 15,
    noise_std: float = 0.0,
    seed: int = 42,
) -> dict:
    """Run a 12-orientation cue-free localizer; histogram per-unit preferred
    orientations; report entropy + per-bin counts + pass/fail vs thresholds.

    Returns
    -------
    dict with keys:
      - ``n_orientations``: int (12)
      - ``n_units``: int
      - ``per_orientation_n_pref_units``: list[int], length 12
      - ``per_orientation_pct``: list[float], length 12 (sums to 100.0)
      - ``entropy_nats`` / ``entropy_bits``: float
      - ``max_entropy_nats``: float (= ln(12))
      - ``n_bins_geq_5pct``: int
      - ``thresholds``: dict
      - ``pass_strong`` / ``pass_weak`` / ``pass_bins``: bool
    """
    import time as _time
    timing = KokTiming()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    loc_trials: list[np.ndarray] = []
    loc_orient: list[float] = []
    _t0 = _time.monotonic()
    _total = int(len(orients)) * int(n_trials_per_orient)
    for i_o, o in enumerate(orients):
        for _ in range(int(n_trials_per_orient)):
            r = run_kok_localizer_trial(
                bundle,
                probe_orientation_deg=float(o),
                timing=timing, noise_std=float(noise_std), generator=gen,
            )
            loc_trials.append(r.cpu().numpy())
            loc_orient.append(float(o))
        _el = _time.monotonic() - _t0
        print(
            f"[coverage] orient={float(o):.1f}° done={i_o+1}/{len(orients)} "
            f"trials={len(loc_trials)}/{_total} elapsed={_el/60:.2f}min",
            flush=True,
        )
    loc_trials_np = np.stack(loc_trials, axis=0)             # [N, n_l23]
    loc_orient_np = np.asarray(loc_orient)
    stats = _compute_localizer_stats(loc_trials_np, loc_orient_np, orients)
    preferred_deg = stats["preferred_deg"]                   # [n_l23]

    # Histogram per-unit preferred bin (integer index into `orients`).
    n_units = int(preferred_deg.shape[0])
    bin_idx = np.array([
        int(np.argmin(np.abs(orients - p))) for p in preferred_deg
    ])
    hist = np.bincount(bin_idx, minlength=orients.size).astype(np.int64)
    pct = 100.0 * hist.astype(np.float64) / max(1, n_units)

    # Shannon entropy of preferred-orientation distribution.
    probs = hist.astype(np.float64) / max(1, n_units)
    nz = probs[probs > 0]
    entropy_nats = float(-(nz * np.log(nz)).sum())
    entropy_bits = float(-(nz * np.log2(nz)).sum())

    n_bins_geq_5pct = int((pct >= 5.0).sum())

    pass_strong = bool(
        entropy_nats > THRESHOLDS["coverage_pass_strong_nats"]
        and n_bins_geq_5pct >= THRESHOLDS["coverage_n_bins_5pct"]
    )
    pass_weak = bool(entropy_nats > THRESHOLDS["coverage_pass_weak_nats"])
    pass_bins = bool(n_bins_geq_5pct >= THRESHOLDS["coverage_n_bins_5pct"])

    return {
        "n_orientations": int(orients.size),
        "n_units": n_units,
        "orientations_deg": [float(o) for o in orients],
        "per_orientation_n_pref_units": [int(x) for x in hist],
        "per_orientation_pct": [float(x) for x in pct],
        "entropy_nats": entropy_nats,
        "entropy_bits": entropy_bits,
        "max_entropy_nats": float(np.log(orients.size)),
        "max_entropy_bits": float(np.log2(orients.size)),
        "n_bins_geq_5pct": n_bins_geq_5pct,
        "thresholds": {
            "pass_strong_nats": THRESHOLDS["coverage_pass_strong_nats"],
            "pass_weak_nats":   THRESHOLDS["coverage_pass_weak_nats"],
            "n_bins_5pct":      int(THRESHOLDS["coverage_n_bins_5pct"]),
        },
        "pass_strong": pass_strong,
        "pass_weak": pass_weak,
        "pass_bins": pass_bins,
    }


# ---------------------------------------------------------------------------
# Rule-magnitude + readout-alignment (Kok)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _kok_capture_at_decision(
    bundle: CheckpointBundle,
    *,
    cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, noise_std: float,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Run a Kok probe trial; capture ``m`` at end-of-delay and the
    probe-epoch mean ``r_l23``. Computes ``b_task = W_mh_task @ m`` and
    ``b_gen = W_mh_gen @ m``.

    Returns tensors on CPU:
      - ``m`` : [n_m]
      - ``b_task`` : [n_l23_e]
      - ``b_gen``  : [n_l23_e]
      - ``r_l23_probe`` : [n_l23_e]  probe-epoch mean.
    """
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    m_decision: Optional[Tensor] = None
    probe_rates: list[Tensor] = []
    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, None
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x, state, info = bundle.net(frame, state, q_t=q_t)
        if t == delay_end - 1:
            m_decision = state.m.clone()
        if delay_end <= t < probe1_end:
            probe_rates.append(info["r_l23"][0].clone())

    assert m_decision is not None
    cm = bundle.net.context_memory
    b_task_exc = (m_decision @ cm.W_mh_task_exc.t())[0].detach()
    # Task #74 Fix C-v2: W_mh_task_inh now produces a per-SOM-unit
    # multiplicative gain on the SOM→L23E synapses, not an additive SOM
    # drive. The "inh" task contribution to L23E is the *departure* of
    # the gain from its init-no-op value of 1.0, projected through the
    # (negative) SOM→L23E effective weight. This is an approximate
    # effective-drive proxy assuming unit-scale baseline SOM activity;
    # it recovers the correct sign, unit assignment, and rank-order
    # structure of the task readout's L23E impact.
    som_gain_pre = (m_decision @ cm.W_mh_task_inh.t())[0].detach()
    som_gain = torch.nn.functional.softplus(
        som_gain_pre + _SOM_GAIN_BIAS0
    ).clamp(max=_SOM_GAIN_CLAMP_MAX)
    w_som_l23 = _inhibitory_eff(bundle.net.l23_e.W_som_l23_raw).detach()
    b_task_inh_effective = w_som_l23 @ (som_gain - 1.0)
    b_task = b_task_exc + b_task_inh_effective
    b_gen = (m_decision @ cm.W_mh_gen.t())[0].detach()
    r_l23 = torch.stack(probe_rates, dim=0).mean(dim=0).detach()
    return {
        "m": m_decision[0].detach(),
        "b_task": b_task,
        "b_task_exc": b_task_exc,
        "b_task_inh": b_task_inh_effective,
        "som_gain": som_gain,
        "b_gen": b_gen,
        "r_l23_probe": r_l23,
    }


def _pearson(u: np.ndarray, v: np.ndarray) -> float:
    """Pearson r across units. NaN if either vector has zero variance."""
    u = u.astype(np.float64).ravel()
    v = v.astype(np.float64).ravel()
    if u.std() == 0 or v.std() == 0:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


@torch.no_grad()
def compute_rule_and_alignment_kok(
    bundle: CheckpointBundle,
    *,
    seed: int,
    n_probe_per_cond: int = 25,
    n_localizer: int = 20,
    noise_std: float = 0.0,
) -> dict:
    """Run a balanced 2-cue × 2-probe Kok block (cue_mapping from seed);
    capture ``m`` at end-of-delay → ``b_task = W_mh_task @ m``; run a matched
    cue-FREE localizer at the same probe orientations; compute rule magnitude
    and readout alignment per expected class.

    Returns dict with:
      - ``rule_magnitude``: {bias_added_norm_mean, localizer_norm_mean, ratio,
                              pass_gate, pass_target, per_class...}
      - ``readout_alignment``: {per_class_cos_same, per_class_cos_other_mean,
                                 mean_cos_same, pass, control_b_gen_same}
    """
    cfg = bundle.cfg
    timing = KokTiming()
    cue_map = cue_mapping_from_seed(seed)    # {cue_id: expected_orient_deg}
    probe_orients = list(CUE_ORIENTATIONS_DEG)   # [45.0, 135.0]
    ncls = len(probe_orients)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # ---- probe trials (balanced cue × probe) --------------------------
    records: list[dict] = []
    for cue_id in (0, 1):
        for p_o in probe_orients:
            for _ in range(int(n_probe_per_cond)):
                rec = _kok_capture_at_decision(
                    bundle, cue_id=cue_id,
                    probe_orientation_deg=float(p_o),
                    timing=timing, noise_std=float(noise_std), generator=gen,
                )
                rec["cue_id"] = int(cue_id)
                rec["probe_orient"] = float(p_o)
                # expected_class = index of the orientation the cue points to.
                rec["expected_class"] = int(
                    probe_orients.index(float(cue_map[cue_id]))
                )
                records.append(rec)

    b_task_arr = np.stack([r["b_task"].numpy() for r in records], axis=0)
    b_gen_arr = np.stack([r["b_gen"].numpy() for r in records], axis=0)
    y_exp = np.array([r["expected_class"] for r in records], dtype=int)

    # ---- localizer templates per class (cue-FREE) ---------------------
    loc_templates: dict[int, np.ndarray] = {}
    loc_norms: dict[int, float] = {}
    for c, o in enumerate(probe_orients):
        rates = []
        for _ in range(int(n_localizer)):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(o),
                timing=timing, noise_std=float(noise_std), generator=gen,
            )
            rates.append(r.cpu().numpy())
        loc_mean = np.stack(rates, axis=0).mean(axis=0)
        loc_templates[c] = loc_mean
        loc_norms[c] = float(np.linalg.norm(loc_mean))

    # ---- Rule magnitude: ||b_task|| per trial vs matched localizer_norm
    per_class_rule: dict[int, dict] = {}
    class_bias_norms: list[float] = []
    class_loc_norms: list[float] = []
    for c in range(ncls):
        sel = y_exp == c
        if not sel.any():
            continue
        bias_norms = np.linalg.norm(b_task_arr[sel], axis=1)
        mean_bias = float(bias_norms.mean())
        lnorm = loc_norms[c]
        per_class_rule[c] = {
            "expected_orient_deg": float(probe_orients[c]),
            "n_trials": int(sel.sum()),
            "bias_added_norm_mean": mean_bias,
            "bias_added_norm_std": float(bias_norms.std()),
            "localizer_norm": lnorm,
            "ratio": float(mean_bias / (lnorm + 1e-12)),
        }
        class_bias_norms.append(mean_bias)
        class_loc_norms.append(lnorm)

    bias_mean = float(np.mean(class_bias_norms))
    lnorm_mean = float(np.mean(class_loc_norms))
    ratio_agg = float(bias_mean / (lnorm_mean + 1e-12))
    rule_block = {
        "bias_added_norm_mean": bias_mean,
        "localizer_norm_mean": lnorm_mean,
        "ratio": ratio_agg,
        "threshold_gate": THRESHOLDS["rule_magnitude_gate"],
        "threshold_target": THRESHOLDS["rule_magnitude_target"],
        "pass_gate": bool(ratio_agg > THRESHOLDS["rule_magnitude_gate"]),
        "pass_target": bool(bias_mean > THRESHOLDS["rule_magnitude_target"]),
        "per_class": {str(k): v for k, v in per_class_rule.items()},
    }

    # ---- Readout alignment: Pearson(b_task_c, localizer[c])  ---------
    cos_same: list[float] = []
    cos_other: list[float] = []
    cos_bgen_same: list[float] = []
    for c in range(ncls):
        sel = y_exp == c
        if not sel.any():
            continue
        bt_c = b_task_arr[sel].mean(axis=0)
        bg_c = b_gen_arr[sel].mean(axis=0)
        cos_same.append(_pearson(bt_c, loc_templates[c]))
        others = [cc for cc in range(ncls) if cc != c]
        if others:
            cos_other.append(float(np.mean([
                _pearson(bt_c, loc_templates[cc]) for cc in others
            ])))
        cos_bgen_same.append(_pearson(bg_c, loc_templates[c]))

    mean_cos = float(np.mean(cos_same)) if cos_same else float("nan")
    mean_cos_other = float(np.mean(cos_other)) if cos_other else float("nan")
    mean_cos_bgen = (
        float(np.mean(cos_bgen_same)) if cos_bgen_same else float("nan")
    )
    align_block = {
        "per_class_cos_same": cos_same,
        "per_class_cos_other_mean": cos_other,
        "mean_cos_same": mean_cos,
        "mean_cos_other": mean_cos_other,
        "control_mean_cos_b_gen_same": mean_cos_bgen,
        "threshold": THRESHOLDS["readout_cos"],
        "pass": bool(mean_cos > THRESHOLDS["readout_cos"]),
    }

    return {"rule_magnitude": rule_block, "readout_alignment": align_block}


# ---------------------------------------------------------------------------
# Rule-magnitude + readout-alignment (Richter)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _richter_capture_at_decision(
    bundle: CheckpointBundle,
    *,
    leader_pos: int, trailer_pos: int,
    timing: RichterTiming, noise_std: float,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Run a Richter leader→trailer trial; capture ``m`` at end-of-leader
    (decision point) and the trailer-epoch mean ``r_l23``. Derives
    ``b_task = W_mh_task @ m`` and ``b_gen = W_mh_gen @ m``.
    """
    cfg = bundle.cfg
    device = cfg.device
    leader_tok = bundle.bank.tokens[
        LEADER_TOKEN_IDX[int(leader_pos)]:LEADER_TOKEN_IDX[int(leader_pos)] + 1
    ].to(device=device)
    trailer_tok = bundle.bank.tokens[
        TRAILER_TOKEN_IDX[int(trailer_pos)]:TRAILER_TOKEN_IDX[int(trailer_pos)] + 1
    ].to(device=device)
    leader_v = build_leader_tensor(
        int(leader_pos), bundle.net.context_memory.n_leader, device=device,
    )

    state = bundle.net.initial_state(batch_size=1)
    leader_end = timing.leader_steps
    trailer_end = leader_end + timing.trailer_steps
    m_decision: Optional[Tensor] = None
    trailer_rates: list[Tensor] = []
    for t in range(trailer_end):
        if t < leader_end:
            frame, ld_t = leader_tok, leader_v
        else:
            frame, ld_t = trailer_tok, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x, state, info = bundle.net(frame, state, leader_t=ld_t)
        if t == leader_end - 1:
            m_decision = state.m.clone()
        if leader_end <= t < trailer_end:
            trailer_rates.append(info["r_l23"][0].clone())
    assert m_decision is not None
    cm = bundle.net.context_memory
    b_task_exc = (m_decision @ cm.W_mh_task_exc.t())[0].detach()
    # Task #74 Fix C-v2: see _kok_capture_at_decision for rationale. The
    # inh task contribution is the gain-departure from 1 projected
    # through the SOM→L23E inhibitory effective weight.
    som_gain_pre = (m_decision @ cm.W_mh_task_inh.t())[0].detach()
    som_gain = torch.nn.functional.softplus(
        som_gain_pre + _SOM_GAIN_BIAS0
    ).clamp(max=_SOM_GAIN_CLAMP_MAX)
    w_som_l23 = _inhibitory_eff(bundle.net.l23_e.W_som_l23_raw).detach()
    b_task_inh_effective = w_som_l23 @ (som_gain - 1.0)
    b_task = b_task_exc + b_task_inh_effective
    b_gen = (m_decision @ cm.W_mh_gen.t())[0].detach()
    r_l23 = torch.stack(trailer_rates, dim=0).mean(dim=0).detach()
    return {
        "m": m_decision[0].detach(),
        "b_task": b_task,
        "b_task_exc": b_task_exc,
        "b_task_inh": b_task_inh_effective,
        "som_gain": som_gain,
        "b_gen": b_gen,
        "r_l23_probe": r_l23,
    }


@torch.no_grad()
def compute_rule_and_alignment_richter(
    bundle: CheckpointBundle,
    *,
    seed: int,
    n_trials_per_cond: int = 2,
    n_localizer: int = 15,
    noise_std: float = 0.0,
) -> dict:
    """Richter analog of :func:`compute_rule_and_alignment_kok`.

    Each of 6 leader positions has a seed-determined expected trailer
    ``σ(leader)``. For each (leader, trailer) condition we capture
    ``m_decision`` at end-of-leader and derive ``b_task = W_mh_task @ m``.
    Trailer-only (leader-free) localizer gives the paradigm sensory template
    per trailer class.
    """
    timing = RichterTiming()
    perm = permutation_from_seed(seed)            # [6] expected trailer[leader]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    ncls = int(N_LEAD_TRAIL)

    # Balanced leader × trailer trials.
    records: list[dict] = []
    for l_pos in range(ncls):
        for t_pos in range(ncls):
            for _ in range(int(n_trials_per_cond)):
                rec = _richter_capture_at_decision(
                    bundle, leader_pos=l_pos, trailer_pos=t_pos,
                    timing=timing, noise_std=float(noise_std),
                    generator=gen,
                )
                rec["leader_pos"] = int(l_pos)
                rec["trailer_pos"] = int(t_pos)
                rec["expected_class"] = int(perm[l_pos])
                records.append(rec)

    b_task_arr = np.stack([r["b_task"].numpy() for r in records], axis=0)
    b_gen_arr = np.stack([r["b_gen"].numpy() for r in records], axis=0)
    y_exp = np.array([r["expected_class"] for r in records], dtype=int)

    # Trailer-only localizer (leader-free) per trailer class.
    loc_templates: dict[int, np.ndarray] = {}
    loc_norms: dict[int, float] = {}
    for c in range(ncls):
        rates = []
        for _ in range(int(n_localizer)):
            r = run_richter_localizer_trial(
                bundle, trailer_pos=int(c),
                timing=timing, noise_std=float(noise_std), generator=gen,
            )
            rates.append(r.cpu().numpy())
        loc_mean = np.stack(rates, axis=0).mean(axis=0)
        loc_templates[c] = loc_mean
        loc_norms[c] = float(np.linalg.norm(loc_mean))

    # Rule magnitude.
    per_class_rule: dict[int, dict] = {}
    class_bias_norms: list[float] = []
    class_loc_norms: list[float] = []
    for c in range(ncls):
        sel = y_exp == c
        if not sel.any():
            continue
        bias_norms = np.linalg.norm(b_task_arr[sel], axis=1)
        mean_bias = float(bias_norms.mean())
        lnorm = loc_norms[c]
        per_class_rule[c] = {
            "expected_trailer": int(c),
            "n_trials": int(sel.sum()),
            "bias_added_norm_mean": mean_bias,
            "bias_added_norm_std": float(bias_norms.std()),
            "localizer_norm": lnorm,
            "ratio": float(mean_bias / (lnorm + 1e-12)),
        }
        class_bias_norms.append(mean_bias)
        class_loc_norms.append(lnorm)
    bias_mean = float(np.mean(class_bias_norms))
    lnorm_mean = float(np.mean(class_loc_norms))
    ratio_agg = float(bias_mean / (lnorm_mean + 1e-12))
    rule_block = {
        "bias_added_norm_mean": bias_mean,
        "localizer_norm_mean": lnorm_mean,
        "ratio": ratio_agg,
        "threshold_gate": THRESHOLDS["rule_magnitude_gate"],
        "threshold_target": THRESHOLDS["rule_magnitude_target"],
        "pass_gate": bool(ratio_agg > THRESHOLDS["rule_magnitude_gate"]),
        "pass_target": bool(bias_mean > THRESHOLDS["rule_magnitude_target"]),
        "per_class": {str(k): v for k, v in per_class_rule.items()},
    }

    # Readout alignment.
    cos_same, cos_other, cos_bgen_same = [], [], []
    for c in range(ncls):
        sel = y_exp == c
        if not sel.any():
            continue
        bt_c = b_task_arr[sel].mean(axis=0)
        bg_c = b_gen_arr[sel].mean(axis=0)
        cos_same.append(_pearson(bt_c, loc_templates[c]))
        others = [cc for cc in range(ncls) if cc != c]
        cos_other.append(float(np.mean([
            _pearson(bt_c, loc_templates[cc]) for cc in others
        ])))
        cos_bgen_same.append(_pearson(bg_c, loc_templates[c]))
    mean_cos = float(np.mean(cos_same)) if cos_same else float("nan")
    mean_cos_other = float(np.mean(cos_other)) if cos_other else float("nan")
    mean_cos_bgen = (
        float(np.mean(cos_bgen_same)) if cos_bgen_same else float("nan")
    )
    align_block = {
        "per_class_cos_same": cos_same,
        "per_class_cos_other_mean": cos_other,
        "mean_cos_same": mean_cos,
        "mean_cos_other": mean_cos_other,
        "control_mean_cos_b_gen_same": mean_cos_bgen,
        "threshold": THRESHOLDS["readout_cos"],
        "pass": bool(mean_cos > THRESHOLDS["readout_cos"]),
    }

    return {"rule_magnitude": rule_block, "readout_alignment": align_block}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _build_untrained_bundle(seed: int, device: str) -> CheckpointBundle:
    """Instantiate a fresh V2Network + TokenBank + ModelConfig with no
    trained weights loaded — represents the Phase-2 step-0 state.

    Used by Fix-A root-cause investigation: comparing preferred-orientation
    coverage at INITIAL (this path) vs FINAL (loaded checkpoint) states
    isolates whether coverage collapse originates in connectivity seeds
    (``A2`` fix) or in Phase-2 learning dynamics (``A3`` fix).
    """
    from src.v2_model.config import ModelConfig
    from src.v2_model.network import V2Network
    from src.v2_model.stimuli.feature_tokens import TokenBank

    cfg = ModelConfig(seed=seed, device=device)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device=device)
    net.set_phase("phase2")
    net.eval()
    return CheckpointBundle(
        cfg=cfg, net=net, bank=bank,
        meta={"phase": "untrained_init", "step": 0},
    )


def run_diagnostics(
    ckpt: Optional[Path],
    *,
    paradigm: str = "kok",
    seed: int = 42,
    n_coverage_trials: int = 15,
    n_probe_per_cond: int = 25,
    n_localizer: int = 20,
    noise_std: float = 0.0,
    skip_phase3_metrics: bool = False,
    untrained_init: bool = False,
    device: str = "cpu",
) -> dict:
    """Run all Task #74 Phase-0 diagnostics on one checkpoint.

    ``paradigm`` selects Kok (default) or Richter probe structure for
    metrics 2+3 (the coverage metric is paradigm-independent).
    When ``skip_phase3_metrics`` is True (e.g. Phase-2-only checkpoint),
    only the Coverage metric is computed; rule_magnitude and
    readout_alignment are returned as ``{"skipped": True}``.
    When ``untrained_init`` is True, the ``ckpt`` argument is ignored and
    a fresh V2Network is instantiated from ``seed`` — used by the Fix-A
    root-cause investigation.
    """
    if paradigm not in ("kok", "richter"):
        raise ValueError(
            f"paradigm must be 'kok' or 'richter'; got {paradigm!r}"
        )
    if untrained_init:
        bundle = _build_untrained_bundle(seed=seed, device=device)
        # Rule/alignment are undefined when W_mh_task is exact zero.
        skip_phase3_metrics = True
    else:
        if ckpt is None:
            raise ValueError(
                "ckpt must be provided unless untrained_init=True",
            )
        bundle = load_checkpoint(ckpt, seed=seed, device=device)

    coverage = compute_coverage(
        bundle,
        n_trials_per_orient=n_coverage_trials,
        noise_std=noise_std, seed=seed,
    )

    if skip_phase3_metrics:
        rule_align = {
            "rule_magnitude": {"skipped": True},
            "readout_alignment": {"skipped": True},
        }
    elif paradigm == "kok":
        rule_align = compute_rule_and_alignment_kok(
            bundle, seed=seed,
            n_probe_per_cond=n_probe_per_cond,
            n_localizer=n_localizer,
            noise_std=noise_std,
        )
    else:
        # n_probe_per_cond applies per (leader × trailer) cell in Richter.
        rule_align = compute_rule_and_alignment_richter(
            bundle, seed=seed,
            n_trials_per_cond=max(1, n_probe_per_cond // 12),
            n_localizer=n_localizer,
            noise_std=noise_std,
        )

    phase = bundle.meta.get("phase", "unknown")
    return {
        "task": "task74_diagnostics",
        "version": 1,
        "checkpoint": (str(ckpt) if ckpt is not None else None),
        "untrained_init": bool(untrained_init),
        "paradigm": paradigm,
        "phase": phase,
        "seed": int(seed),
        "coverage": coverage,
        **rule_align,
        "thresholds": THRESHOLDS,
    }


def _fmt_summary_line(out: dict) -> str:
    """One-line echo of the three headline scalars (for quick DM)."""
    cov = out["coverage"]
    rm = out.get("rule_magnitude", {})
    ra = out.get("readout_alignment", {})
    parts = [
        f"coverage_entropy={cov['entropy_nats']:.3f}",
        f"n_bins_5pct={cov['n_bins_geq_5pct']}/{cov['n_orientations']}",
    ]
    if not rm.get("skipped"):
        parts.extend([
            f"bias_norm={rm['bias_added_norm_mean']:.4f}",
            f"loc_norm={rm['localizer_norm_mean']:.3f}",
            f"ratio={rm['ratio']:.4f}",
        ])
    if not ra.get("skipped"):
        parts.append(f"cos_bt_same={ra['mean_cos_same']:.3f}")
    parts.extend([
        f"target_entropy>{THRESHOLDS['coverage_pass_strong_nats']:.2f}",
        f"target_ratio>{THRESHOLDS['rule_magnitude_gate']:.2f}",
        f"target_cos>{THRESHOLDS['readout_cos']:.2f}",
    ])
    return " ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Task #74 Phase-0 diagnostics (coverage, rule mag, alignment)",
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to Phase-2 or Phase-3 checkpoint. "
                        "Required unless --untrained-init is set.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paradigm", choices=["kok", "richter"], default="kok",
                   help="Paradigm for rule/alignment metrics; coverage is "
                        "paradigm-independent.")
    p.add_argument("--untrained-init", action="store_true",
                   help="Diagnose a fresh V2Network (Phase-2 step 0). "
                        "Coverage-only; rule/alignment skipped.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-coverage-trials", type=int, default=15,
                   help="localizer trials per orientation (12 orients)")
    p.add_argument("--n-probe-per-cond", type=int, default=25,
                   help="Kok probe trials per cue×probe condition")
    p.add_argument("--n-localizer", type=int, default=20,
                   help="Kok localizer trials per expected-class orientation")
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--skip-phase3-metrics", action="store_true",
                   help="For Phase-2-only checkpoints; skip rule/alignment.")
    args = p.parse_args(argv)

    if not args.untrained_init and args.checkpoint is None:
        p.error("--checkpoint is required unless --untrained-init is set")

    out = run_diagnostics(
        args.checkpoint,
        paradigm=args.paradigm,
        seed=args.seed,
        n_coverage_trials=args.n_coverage_trials,
        n_probe_per_cond=args.n_probe_per_cond,
        n_localizer=args.n_localizer,
        noise_std=args.noise_std,
        skip_phase3_metrics=args.skip_phase3_metrics,
        untrained_init=args.untrained_init,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(_fmt_summary_line(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
