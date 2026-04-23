"""Task #49 — Verify claims 4-6 + forensic probes A/B/C.

Shared setup:
    - Fresh V2Network with default config
    - Three stimulus conditions: blank, preferred grating, procedural world
    - Rate / θ / effective-softplus-weight logging

Claims (all results printed by the corresponding run_claim*):
    C4: Kok cue-memory learning dead from zero state — single Kok trial
    C5: Full-state Jacobian at blank op point
    C6: Homeostasis operating point at fresh init

Probes (all 3 stim × 1000 steps):
    A: no plast + no homeo — check stability at rest
    B: homeo only (1e-5) — θ drift
    C: plast only (default LRs) × {weight_decay=0, default} — effective weights

Usage:
    python3 -m scripts.v2._debug_task49_main --run all
    python3 -m scripts.v2._debug_task49_main --run c4
    python3 -m scripts.v2._debug_task49_main --run c5
    python3 -m scripts.v2._debug_task49_main --run c6
    python3 -m scripts.v2._debug_task49_main --run probeA
    python3 -m scripts.v2._debug_task49_main --run probeB
    python3 -m scripts.v2._debug_task49_main --run probeC
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.state import NetworkStateV2
from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world.procedural import ProceduralWorld
from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank, apply_plasticity_step, _forward_window,
)
from scripts.v2._gates_common import make_blank_frame, make_grating_frame
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, run_kok_trial, cue_mapping_from_seed,
    build_cue_tensor,
)


# ---------------------------------------------------------------------------
# Common setup helpers
# ---------------------------------------------------------------------------

def build_net(seed: int = 42) -> tuple[V2Network, ModelConfig]:
    cfg = ModelConfig()
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    return net, cfg


def zero_all_homeo(net: V2Network) -> None:
    """Post-construction bypass of constructor's lr > 0 check."""
    net.l23_e.homeostasis.lr = 0.0
    net.h_e.homeostasis.lr = 0.0


def make_blank(cfg: ModelConfig, B: int) -> Tensor:
    """[B, 1, H, W] blank frame (grey 0.5)."""
    return make_blank_frame(batch_size=B, cfg=cfg)


def make_grating(cfg: ModelConfig, B: int, orientation_deg: float = 45.0) -> Tensor:
    """[B, 1, H, W] oriented grating at full contrast."""
    return make_grating_frame(
        orientation_deg=orientation_deg, contrast=1.0, cfg=cfg, batch_size=B,
    )


def make_procedural(
    world: ProceduralWorld, n_steps: int, seed: int = 42, B: int = 4,
) -> Tensor:
    """[B, T, 1, H, W] procedural trajectories."""
    tracks = []
    for b in range(B):
        frames, _ = world.trajectory(seed + b, n_steps)
        tracks.append(frames)
    return torch.stack(tracks, dim=0)


def flat_rate_state(state: NetworkStateV2) -> Tensor:
    """Flatten [r_l4, r_l23, r_pv, r_som, r_h, h_pv, m] on dim=-1."""
    return torch.cat([
        state.r_l4, state.r_l23, state.r_pv, state.r_som,
        state.r_h, state.h_pv, state.m,
    ], dim=-1)


def split_rate_state(
    vec: Tensor, cfg: ModelConfig, base_state: NetworkStateV2,
) -> NetworkStateV2:
    """Inverse of flat_rate_state using cfg.arch dims."""
    a = cfg.arch
    sizes = [a.n_l4_e, a.n_l23_e, a.n_l23_pv, a.n_l23_som,
             a.n_h_e, a.n_h_pv, a.n_c]
    idx = 0
    fields = {}
    for name, n in zip(
        ["r_l4", "r_l23", "r_pv", "r_som", "r_h", "h_pv", "m"], sizes,
    ):
        fields[name] = vec[..., idx : idx + n].contiguous()
        idx += n
    return base_state._replace(**fields)


def rates_per_pop(state: NetworkStateV2) -> dict[str, dict[str, float]]:
    """mean/max/min per population, scalar floats."""
    def stat(t: Tensor) -> dict[str, float]:
        t = t.detach()
        return {
            "mean": float(t.mean().item()),
            "max": float(t.max().item()),
            "min": float(t.min().item()),
        }
    return {
        "r_l4": stat(state.r_l4), "r_l23": stat(state.r_l23),
        "r_pv": stat(state.r_pv), "r_som": stat(state.r_som),
        "r_h": stat(state.r_h), "h_pv": stat(state.h_pv),
        "m": stat(state.m),
    }


# ---------------------------------------------------------------------------
# Claim 4: Kok cue-memory learning dead from zero state
# ---------------------------------------------------------------------------

def run_claim4() -> None:
    print()
    print("=" * 72)
    print("=== Claim 4: Kok cue learning dead from zero state ===")
    print("=" * 72)
    net, cfg = build_net(seed=42)
    net.set_phase("phase3_kok")

    # Reset W_qm_task to zero (the claim is about cold-start Phase 3).
    with torch.no_grad():
        net.context_memory.W_qm_task.zero_()

    # Instrument: run cue phase by hand and probe memory
    timing = KokTiming()
    cue_id = 0
    cue_mapping = cue_mapping_from_seed(42)
    orientation_deg = cue_mapping[cue_id]
    q_cue = build_cue_tensor(cue_id=cue_id, n_cue=cfg.arch.n_c, device="cpu")

    state = net.initial_state(batch_size=1)
    blank = make_blank(cfg, 1)  # [1, 1, H, W]
    m_pre_cue = state.m.clone()
    print(f"m_pre_cue: mean={m_pre_cue.mean().item():+.6e} "
          f"max={m_pre_cue.abs().max().item():.6e} "
          f"norm={m_pre_cue.norm().item():.6e}")
    print(f"q_cue sum={q_cue.sum().item():.4f} "
          f"(one-hot, active index shows the cue)")

    # Track m during cue phase
    with torch.no_grad():
        for t in range(timing.cue_steps):
            x_hat, state, info = net.forward(blank, state, q_t=q_cue)
            if t in (0, 4, 9, 19, 29, 39):
                print(f"  cue t={t:>2}  m.mean={state.m.mean().item():+.4e}  "
                      f"m.max={state.m.abs().max().item():.4e}  "
                      f"m.norm={state.m.norm().item():.4e}  "
                      f"r_h.mean={state.r_h.mean().item():+.4e}")
        m_end_cue = state.m.clone()

    memory_error = m_end_cue - m_pre_cue
    print()
    print(f"m_end_cue: mean={m_end_cue.mean().item():+.6e} "
          f"max={m_end_cue.abs().max().item():.6e} "
          f"norm={m_end_cue.norm().item():.6e}")
    print(f"memory_error = m_end_cue − m_pre_cue:")
    print(f"    norm={memory_error.norm().item():.4e}  "
          f"max|.|={memory_error.abs().max().item():.4e}")

    # Now the delta_qm magnitude:
    gated = m_end_cue * memory_error                        # [1, n_m]
    hebb_outer = gated.t() @ q_cue / 1.0                    # [n_m, n_cue]
    print()
    print(f"gated = memory * memory_error:  norm={gated.norm().item():.4e}")
    print(f"hebb = outer(gated, cue):       norm={hebb_outer.norm().item():.4e}  "
          f"max|.|={hebb_outer.abs().max().item():.4e}")
    print(f"dw_qm at lr=1e-3, wd=1e-5: max|Δw| ≈ "
          f"{1e-3 * hebb_outer.abs().max().item():.4e} "
          f"(wd contribution is ZERO since W_qm_task=0)")
    print()
    print("=> If ||memory_error|| and ||m_end_cue|| are both tiny, the "
          "product is doubly tiny and learning is dead (Claim 4).")


# ---------------------------------------------------------------------------
# Claim 5: Full-state Jacobian at blank operating point
# ---------------------------------------------------------------------------

def _forward_flat(
    net: V2Network, frame: Tensor, state: NetworkStateV2, cfg: ModelConfig,
    vec: Tensor,
) -> Tensor:
    """Replace rate fields of `state` with `vec` and forward one step; return flat next state."""
    new_state = split_rate_state(vec.unsqueeze(0), cfg, state)
    _, next_state, _ = net.forward(frame, new_state)
    return flat_rate_state(next_state).squeeze(0)


def full_jacobian(
    net: V2Network, frame: Tensor, state: NetworkStateV2, cfg: ModelConfig,
    eps: float = 1e-3,
) -> Tensor:
    """Finite-difference Jacobian d r_{t+1} / d r_t at the given op point.

    Uses BATCHED perturbation: builds a batch where each column corresponds
    to one perturbation direction, so the full Jacobian falls out of 1
    forward pass (plus 1 for the baseline).
    """
    s0 = flat_rate_state(state).squeeze(0)                         # [n]
    n = s0.shape[0]

    with torch.no_grad():
        f0 = _forward_flat(net, frame, state, cfg, s0)             # [n]

        J = torch.zeros(n, n, dtype=s0.dtype)
        for i in range(n):
            s_pert = s0.clone()
            s_pert[i] += eps
            f_pert = _forward_flat(net, frame, state, cfg, s_pert) # [n]
            J[:, i] = (f_pert - f0) / eps

    return J


def run_claim5() -> None:
    print()
    print("=" * 72)
    print("=== Claim 5: Full-state Jacobian at blank op point ===")
    print("=" * 72)
    net, cfg = build_net(seed=42)
    net.eval()

    B = 1
    state0 = net.initial_state(batch_size=B)
    blank = make_blank(cfg, B)

    # Drive one step to leave exactly-zero degenerate kernel
    with torch.no_grad():
        _, state1, _ = net.forward(blank, state0)

    print("Baseline rates at blank op point (after 1 step from zero):")
    for name, stat in rates_per_pop(state1).items():
        print(f"  {name:>6}  mean={stat['mean']:+.4e}  max={stat['max']:.4e}")

    print()
    print("Computing full-state Jacobian (finite difference, eps=1e-3)...")
    t0 = time.monotonic()
    J = full_jacobian(net, blank, state1, cfg, eps=1e-3)
    print(f"J shape = {tuple(J.shape)}  (elapsed {time.monotonic() - t0:.2f}s)")

    # Eigenvalues (complex-valued for non-symmetric J)
    eigvals = torch.linalg.eigvals(J)
    eig_abs = eigvals.abs()
    order = torch.argsort(eig_abs, descending=True)
    top = eig_abs[order[:10]]
    print()
    print("Top 10 |eig|:")
    for i, k in enumerate(order[:10].tolist()):
        e = eigvals[k]
        print(f"  #{i+1}: |λ|={eig_abs[k].item():.6f}  "
              f"λ={e.real.item():+.6f}{'+' if e.imag>=0 else '-'}{abs(e.imag.item()):.6f}j")
    print(f"Spectral radius (full state) = {top[0].item():.6f}")

    # Eigenvector loading breakdown — use right eigenvectors
    eigvals_full, eigvecs = torch.linalg.eig(J)
    order_full = torch.argsort(eigvals_full.abs(), descending=True)

    # Block boundaries (same order as flat_rate_state):
    a = cfg.arch
    blocks = [
        ("r_l4", a.n_l4_e), ("r_l23", a.n_l23_e), ("r_pv", a.n_l23_pv),
        ("r_som", a.n_l23_som), ("r_h", a.n_h_e), ("h_pv", a.n_h_pv),
        ("m", a.n_c),
    ]
    starts = []
    idx = 0
    for name, n in blocks:
        starts.append((name, idx, idx + n))
        idx += n

    print()
    print("Top-eigenvector block loadings (L2 norm of loading within each block):")
    v = eigvecs[:, order_full[0]]
    v_abs = v.abs()
    v_norm = v_abs.norm()
    for (name, lo, hi) in starts:
        block_norm = v_abs[lo:hi].norm().item()
        print(f"  {name:>6}  ||v_block||/||v|| = {block_norm / v_norm.item():.4f}")

    # Per-population spectral radius of W_rec alone (the current guard)
    w_rec_l23 = F.softplus(net.l23_e.W_rec_raw) * net.l23_e.mask_rec
    w_rec_h = F.softplus(net.h_e.W_rec_raw) * net.h_e.mask_rec
    # Guard form: |eig(leak*I + w_rec)| < 1?  Actually layers.py uses
    # rate_next = leak * state + phi(drive - theta), so the linear-regime
    # d r_{t+1} / d r_t via rec is phi'(drive) * w_rec, and the total diag
    # includes + leak from the state carry. We compute both.
    leak_l23 = net.l23_e._leak
    leak_h = net.h_e._leak
    eig_l23 = torch.linalg.eigvals(w_rec_l23).abs().max().item()
    eig_h = torch.linalg.eigvals(w_rec_h).abs().max().item()
    print()
    print(f"Per-population guard (current): ")
    print(f"  L23E |eig(W_rec)| max = {eig_l23:.6f}  leak = {leak_l23:.3f}  "
          f"=> leak + phi'·max = {leak_l23 + eig_l23:.6f} (if phi'≈1)")
    print(f"  HE   |eig(W_rec)| max = {eig_h:.6f}  leak = {leak_h:.3f}  "
          f"=> leak + phi'·max = {leak_h + eig_h:.6f} (if phi'≈1)")
    print()
    print(f"Full-network spectral radius = {top[0].item():.6f}")


# ---------------------------------------------------------------------------
# Claim 6: Homeostasis operating point at fresh init
# ---------------------------------------------------------------------------

def run_claim6() -> None:
    print()
    print("=" * 72)
    print("=== Claim 6: Homeostasis operating-point delta at fresh init ===")
    print("=" * 72)
    net, cfg = build_net(seed=42)

    B = 4
    state = net.initial_state(batch_size=B)
    blank = make_blank(cfg, B)

    # Drive one step to get first-forward rates
    with torch.no_grad():
        _, state, _ = net.forward(blank, state)

    print(f"Fresh-init rates after 1 forward on blank frames:")
    print(f"  L23E target={net.l23_e.homeostasis.target_rate:.3f}  "
          f"actual mean={state.r_l23.mean().item():.6e}  "
          f"gap={state.r_l23.mean().item() - net.l23_e.homeostasis.target_rate:+.4f}")
    print(f"  HE   target={net.h_e.homeostasis.target_rate:.3f}  "
          f"actual mean={state.r_h.mean().item():.6e}  "
          f"gap={state.r_h.mean().item() - net.h_e.homeostasis.target_rate:+.4f}")
    print()
    lr_h = float(net.l23_e.homeostasis.lr)
    drift_l23 = lr_h * (state.r_l23.mean().item() - net.l23_e.homeostasis.target_rate)
    drift_h = float(net.h_e.homeostasis.lr) * (
        state.r_h.mean().item() - net.h_e.homeostasis.target_rate
    )
    print(f"θ drift per step (lr * (actual - target)):")
    print(f"  θ_L23E Δ/step = {drift_l23:+.4e}  (monotone {'negative' if drift_l23 < 0 else 'positive'})")
    print(f"  θ_HE   Δ/step = {drift_h:+.4e}  (monotone {'negative' if drift_h < 0 else 'positive'})")


# ---------------------------------------------------------------------------
# Probe A: no plast + no homeo
# ---------------------------------------------------------------------------

def _run_probe(
    net: V2Network, cfg: ModelConfig, frames_gen: Callable[[int], Tensor],
    n_steps: int, rules: Optional[PlasticityRuleBank],
    reset_state_each_step: bool = False,
    log_every: int = 100,
    condition_name: str = "",
) -> dict:
    """Generic probe loop.

    frames_gen(step) -> Tensor [B, 1, H, W] for that step
    rules=None => no plasticity
    """
    B = 4
    state = net.initial_state(batch_size=B)
    records = []

    for step in range(n_steps):
        frame = frames_gen(step)
        with torch.no_grad():
            _, state_next, info = net.forward(frame, state)

            # Build a 2-step state pair to call apply_plasticity_step
            if rules is not None:
                state_pre = state
                state_mid = state_next
                # second forward
                frame2 = frames_gen(step + n_steps)  # some fresh frame
                _, state_post, info_post = net.forward(frame2, state_mid)
                # Run plasticity + homeostasis
                from scripts.v2.train_phase2_predictive import _forward_window
                # Mirror driver structure: window of 2 frames
                window = torch.stack([frame, frame2], dim=1)  # [B, 2, 1, H, W]
                state0 = net.initial_state(batch_size=B) if reset_state_each_step else state
                (
                    state0, state1, state2, info0, info1, x_hat_0, _x_hat_1,
                ) = _forward_window(net, window, state0)
                apply_plasticity_step(
                    net, rules, state0, state1, state2, info0, info1, x_hat_0,
                )
                # Homeostasis updates (driver lines 374-375):
                net.l23_e.homeostasis.update(state2.r_l23)
                net.h_e.homeostasis.update(state2.r_h)
                state = state2 if not reset_state_each_step else state2
            else:
                state = state_next

        if step % log_every == 0 or step == n_steps - 1:
            per_pop = rates_per_pop(state)
            # θ
            theta_l23 = float(net.l23_e.homeostasis.theta.mean().item())
            theta_h = float(net.h_e.homeostasis.theta.mean().item())
            # Effective softplus weights (max)
            w_pv_l23_max = float(F.softplus(net.l23_e.W_pv_l23_raw).max().item())
            w_som_l23_max = float(F.softplus(net.l23_e.W_som_l23_raw).max().item())
            w_l4_l23_max = float(F.softplus(net.l23_e.W_l4_l23_raw).max().item())
            w_rec_l23_max = float(F.softplus(net.l23_e.W_rec_raw).max().item())
            w_l23_h_max = float(F.softplus(net.h_e.W_l23_h_raw).max().item())
            w_pred_h_max = float(F.softplus(net.prediction_head.W_pred_H_raw).max().item())

            records.append({
                "step": step,
                "rates": per_pop,
                "theta_l23": theta_l23, "theta_h": theta_h,
                "w_pv_l23_max": w_pv_l23_max, "w_som_l23_max": w_som_l23_max,
                "w_l4_l23_max": w_l4_l23_max, "w_rec_l23_max": w_rec_l23_max,
                "w_l23_h_max": w_l23_h_max, "w_pred_h_max": w_pred_h_max,
            })

    return {
        "condition": condition_name,
        "n_steps": n_steps,
        "records": records,
        "final_state": {
            "rates": rates_per_pop(state),
            "theta_l23": float(net.l23_e.homeostasis.theta.mean().item()),
            "theta_h": float(net.h_e.homeostasis.theta.mean().item()),
        },
    }


def run_probe_A(out_dir: Path) -> None:
    print()
    print("=" * 72)
    print("=== Probe A: no plasticity + no homeostasis, 3 stim × 1000 steps ===")
    print("=" * 72)

    from src.v2_model.world.procedural import ProceduralWorld
    _, cfg = build_net(seed=0)
    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, bank, seed_family="train")

    conditions = [
        ("blank", lambda step, cfg=cfg: make_blank(cfg, 4)),
        ("grating", lambda step, cfg=cfg: make_grating(cfg, 4, 45.0)),
    ]
    # procedural — pre-sample trajectory
    proc_frames = make_procedural(world, n_steps=1000, seed=42, B=4)  # [4, 1000, 1, H, W]
    conditions.append((
        "procedural",
        lambda step, pf=proc_frames: pf[:, step % pf.shape[1]].contiguous(),
    ))

    all_results = {}
    for name, gen in conditions:
        print(f"\n--- A-{name} ---")
        net, _ = build_net(seed=42)
        zero_all_homeo(net)   # homeo off
        result = _run_probe(
            net, cfg, gen, n_steps=1000, rules=None,
            reset_state_each_step=False, log_every=250,
            condition_name=f"A-{name}",
        )
        # Print concise summary
        for r in result["records"]:
            step = r["step"]
            rl23 = r["rates"]["r_l23"]["mean"]
            rh = r["rates"]["r_h"]["mean"]
            rpv = r["rates"]["r_pv"]["mean"]
            rsom = r["rates"]["r_som"]["mean"]
            print(f"  step={step:>4}  r_l23={rl23:+.4e}  r_h={rh:+.4e}  "
                  f"r_pv={rpv:+.4e}  r_som={rsom:+.4e}")
        # Jacobian at final op point (takes ~a few minutes)
        print(f"  -- computing full Jacobian at final op point ...", flush=True)
        state = net.initial_state(batch_size=1)
        # Re-run to final op with B=1
        for step in range(min(100, 1000)):   # short spin to reach op point
            f1 = gen(step)[0:1]
            _, state, _ = net.forward(f1, state)
        J = full_jacobian(net, gen(0)[0:1], state, cfg, eps=1e-3)
        eigs = torch.linalg.eigvals(J).abs()
        top = eigs.sort(descending=True).values[:5]
        print(f"  Top 5 |eig|: " + ", ".join(f"{v.item():.4f}" for v in top))
        result["jacobian_top5_abs"] = [float(v) for v in top]
        all_results[name] = result

    (out_dir / "probe_A.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir / 'probe_A.json'}")


# ---------------------------------------------------------------------------
# Probe B: homeo only, no plasticity
# ---------------------------------------------------------------------------

def run_probe_B(out_dir: Path) -> None:
    print()
    print("=" * 72)
    print("=== Probe B: homeo-only (1e-5), no plasticity, 3 stim × 1000 steps ===")
    print("=" * 72)

    from src.v2_model.world.procedural import ProceduralWorld
    _, cfg = build_net(seed=0)
    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, bank, seed_family="train")

    conditions = [
        ("blank", lambda step, cfg=cfg: make_blank(cfg, 4)),
        ("grating", lambda step, cfg=cfg: make_grating(cfg, 4, 45.0)),
    ]
    proc_frames = make_procedural(world, n_steps=1000, seed=42, B=4)
    conditions.append((
        "procedural",
        lambda step, pf=proc_frames: pf[:, step % pf.shape[1]].contiguous(),
    ))

    all_results = {}
    for name, gen in conditions:
        print(f"\n--- B-{name} ---")
        net, _ = build_net(seed=42)
        # homeo ON at default 1e-5, plast OFF (set LRs to 0 via rules=None + manual homeo calls)
        records = []
        B = 4
        state = net.initial_state(batch_size=B)
        for step in range(1000):
            frame = gen(step)
            with torch.no_grad():
                _, state, _ = net.forward(frame, state)
                net.l23_e.homeostasis.update(state.r_l23)
                net.h_e.homeostasis.update(state.r_h)
            if step % 100 == 0 or step == 999:
                rec = {
                    "step": step,
                    "r_l23_mean": float(state.r_l23.mean().item()),
                    "r_h_mean": float(state.r_h.mean().item()),
                    "r_pv_mean": float(state.r_pv.mean().item()),
                    "r_som_mean": float(state.r_som.mean().item()),
                    "theta_l23_mean": float(net.l23_e.homeostasis.theta.mean().item()),
                    "theta_h_mean": float(net.h_e.homeostasis.theta.mean().item()),
                    "theta_l23_min": float(net.l23_e.homeostasis.theta.min().item()),
                    "theta_h_min": float(net.h_e.homeostasis.theta.min().item()),
                }
                records.append(rec)
                print(f"  step={step:>4}  r_l23={rec['r_l23_mean']:+.4e}  "
                      f"r_h={rec['r_h_mean']:+.4e}  "
                      f"θ_l23={rec['theta_l23_mean']:+.4e}  "
                      f"θ_h={rec['theta_h_mean']:+.4e}")
        all_results[name] = {"records": records}

    (out_dir / "probe_B.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir / 'probe_B.json'}")


# ---------------------------------------------------------------------------
# Probe C: plasticity-only (2 decay settings)
# ---------------------------------------------------------------------------

def run_probe_C(out_dir: Path) -> None:
    print()
    print("=" * 72)
    print("=== Probe C: plast-only, 3 stim × 1000 steps × {wd=0, wd=1e-5} ===")
    print("=" * 72)

    from src.v2_model.world.procedural import ProceduralWorld
    _, cfg = build_net(seed=0)
    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, bank, seed_family="train")

    conditions = [
        ("blank", lambda step, cfg=cfg: make_blank(cfg, 4)),
        ("grating", lambda step, cfg=cfg: make_grating(cfg, 4, 45.0)),
    ]
    proc_frames = make_procedural(world, n_steps=1000, seed=42, B=4)
    conditions.append((
        "procedural",
        lambda step, pf=proc_frames: pf[:, step % pf.shape[1]].contiguous(),
    ))

    all_results = {}
    for wd in (0.0, 1e-5):
        for name, gen in conditions:
            tag = f"C-wd{wd}-{name}"
            print(f"\n--- {tag} ---")
            net, _ = build_net(seed=42)
            zero_all_homeo(net)
            rules = PlasticityRuleBank.from_config(
                cfg,
                lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
                weight_decay=wd, beta_syn=0.0,
            )

            records = []
            B = 4
            state = net.initial_state(batch_size=B)
            for step in range(1000):
                frame = gen(step)
                frame2 = gen((step + 1))
                window = torch.stack([frame, frame2], dim=1)  # [B, 2, 1, H, W]
                with torch.no_grad():
                    state0 = net.initial_state(batch_size=B)
                    state0, state1, state2, info0, info1, x_hat_0, _x_hat_1 = (
                        _forward_window(net, window, state0)
                    )
                    apply_plasticity_step(
                        net, rules, state0, state1, state2,
                        info0, info1, x_hat_0,
                    )
                    state = state2
                if step % 100 == 0 or step == 999:
                    rec = {
                        "step": step,
                        "r_l23_mean": float(state.r_l23.mean().item()),
                        "r_h_mean": float(state.r_h.mean().item()),
                        "w_l4_l23_max": float(F.softplus(net.l23_e.W_l4_l23_raw).max().item()),
                        "w_rec_l23_max": float(F.softplus(net.l23_e.W_rec_raw).max().item()),
                        "w_pv_l23_max": float(F.softplus(net.l23_e.W_pv_l23_raw).max().item()),
                        "w_som_l23_max": float(F.softplus(net.l23_e.W_som_l23_raw).max().item()),
                        "w_l23_h_max": float(F.softplus(net.h_e.W_l23_h_raw).max().item()),
                        "w_pred_h_max": float(F.softplus(net.prediction_head.W_pred_H_raw).max().item()),
                        "eps_abs_mean": float((state2.r_l4 - x_hat_0).abs().mean().item()),
                    }
                    records.append(rec)
                    print(f"  step={step:>4}  r_l23={rec['r_l23_mean']:+.3e}  "
                          f"r_h={rec['r_h_mean']:+.3e}  "
                          f"w_l4l23_max={rec['w_l4_l23_max']:.3e}  "
                          f"w_pvl23_max={rec['w_pv_l23_max']:.3e}  "
                          f"|ε|_mean={rec['eps_abs_mean']:.3e}")
            all_results[tag] = {"records": records}

    (out_dir / "probe_C.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_dir / 'probe_C.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default="all",
        choices=["all", "c4", "c5", "c6", "probeA", "probeB", "probeC"],
    )
    parser.add_argument("--out-dir", default="logs/task49")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    which = args.run
    if which in ("all", "c6"):
        run_claim6()
    if which in ("all", "c4"):
        run_claim4()
    if which in ("all", "c5"):
        run_claim5()
    if which in ("all", "probeA"):
        run_probe_A(out_dir)
    if which in ("all", "probeB"):
        run_probe_B(out_dir)
    if which in ("all", "probeC"):
        run_probe_C(out_dir)


if __name__ == "__main__":
    main()
