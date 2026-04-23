"""Task #49 Probes A/B/C + Claim 5 full-state Jacobian.

Probe A: no plasticity + no homeostasis   × {blank, grating, procedural} × 1000 steps
Probe B: homeostasis only                 × {blank, grating, procedural} × 1000 steps
Probe C: plasticity only (wd=0 and wd=1e-5) × {blank, grating, procedural} × 1000 steps

Claim 5: compute full-state Jacobian ∂state/∂state at op points in 3 stim conditions;
         compare top-5 |eigenvalues| to per-population Jacobian bounds.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from scripts.v2._gates_common import make_blank_frame, make_grating_frame


# ============================================================================
# Helpers
# ============================================================================

def build_net(seed: int = 42) -> tuple[V2Network, ModelConfig]:
    cfg = ModelConfig()
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    return net, cfg


def zero_homeo(net: V2Network) -> None:
    """Zero homeostasis LRs on all populations that have them."""
    for attr in ("l23_e", "h_e", "l4_e"):
        if hasattr(net, attr):
            pop = getattr(net, attr)
            if hasattr(pop, "homeostasis") and pop.homeostasis is not None:
                pop.homeostasis.lr = 0.0


def call_homeo(net: V2Network, state) -> None:
    """Manually step homeostasis on key populations."""
    if hasattr(net.l23_e, "homeostasis"):
        net.l23_e.homeostasis.update(state.r_l23)
    if hasattr(net.h_e, "homeostasis"):
        net.h_e.homeostasis.update(state.r_h)


def make_procedural_frame(batch_size: int, cfg: ModelConfig, seed: int = 42):
    """Build a procedural-world-style frame: multiple random gabors."""
    torch.manual_seed(seed)
    # Use several gratings at random orientations combined.
    frame = make_blank_frame(batch_size=batch_size, cfg=cfg)
    for i in range(3):
        orient = (i * 43) % 180
        g = make_grating_frame(orientation_deg=orient, contrast=0.3,
                               cfg=cfg, batch_size=batch_size)
        frame = frame + g
    return frame


def flat_state(state) -> Tensor:
    """Flatten state to a 1-D tensor. Order: r_l4, r_l23, r_pv, r_som, r_h."""
    parts = []
    for name in ("r_l4", "r_l23", "r_pv", "r_som", "r_h"):
        if hasattr(state, name):
            v = getattr(state, name)
            parts.append(v.reshape(-1))
    return torch.cat(parts)


def unflat_state(state_proto, flat: Tensor):
    """Write flat tensor back into state proto (returns new state). NamedTuple so use _replace."""
    new_state = state_proto
    offset = 0
    updates = {}
    for name in ("r_l4", "r_l23", "r_pv", "r_som", "r_h"):
        if hasattr(new_state, name):
            v = getattr(new_state, name)
            n = v.numel()
            piece = flat[offset:offset + n].reshape(v.shape)
            updates[name] = piece
            offset += n
    return new_state._replace(**updates)


# ============================================================================
# Probes A/B/C
# ============================================================================

def run_probe(
    net: V2Network, cfg: ModelConfig,
    stim_fn: Callable[[int], Tensor], label: str,
    n_steps: int = 1000,
    homeo_on: bool = False,
    plast_on: bool = False,
    wd: float = 1e-5,
) -> dict:
    """Run a controlled rollout. Returns metrics dict with rates trajectory."""
    B = 1
    state = net.initial_state(batch_size=B)
    if not homeo_on:
        zero_homeo(net)

    snapshots = {}
    checkpoints = [0, 10, 100, 500, 999]
    for t in range(n_steps):
        frame = stim_fn(B)
        with torch.no_grad():
            _, state, _ = net.forward(frame, state)
        if homeo_on:
            call_homeo(net, state)

        if t in checkpoints:
            r_l23 = state.r_l23.mean().item()
            r_l23_max = state.r_l23.abs().max().item()
            r_h = state.r_h.mean().item()
            r_h_max = state.r_h.abs().max().item()
            r_l4 = state.r_l4.mean().item()
            r_pv = state.r_pv.mean().item() if hasattr(state, "r_pv") else 0.0
            theta_l23 = (net.l23_e.homeostasis.theta.mean().item()
                         if hasattr(net.l23_e, "homeostasis") else 0.0)
            snapshots[t] = dict(
                r_l4=r_l4, r_l23=r_l23, r_l23_max=r_l23_max,
                r_pv=r_pv, r_h=r_h, r_h_max=r_h_max,
                theta_l23=theta_l23,
                has_nan=not torch.isfinite(flat_state(state)).all().item(),
            )
    return dict(label=label, snapshots=snapshots)


def make_stim_fns(cfg: ModelConfig) -> dict:
    return dict(
        blank=lambda B: make_blank_frame(batch_size=B, cfg=cfg),
        grating=lambda B: make_grating_frame(
            orientation_deg=45.0, contrast=0.5, cfg=cfg, batch_size=B),
        procedural=lambda B: make_procedural_frame(B, cfg, seed=42),
    )


def probe_A(n_steps: int = 1000) -> None:
    """No plasticity, no homeostasis. Pure forward dynamics."""
    print()
    print("=" * 72)
    print(f"=== Probe A: NO plasticity + NO homeo × 3 stim × {n_steps} steps ===")
    print("=" * 72)
    for stim_name, stim_fn in make_stim_fns(ModelConfig()).items():
        net, cfg = build_net(seed=42)
        res = run_probe(net, cfg, stim_fn, stim_name, n_steps,
                        homeo_on=False, plast_on=False)
        print(f"\n--- Probe A × {stim_name} ---")
        print(f"  {'t':>4}  {'r_l4':>10}  {'r_l23':>10}  {'r_l23_max':>10}  "
              f"{'r_pv':>10}  {'r_h':>10}  {'r_h_max':>10}  nan")
        for t, snap in sorted(res["snapshots"].items()):
            print(f"  {t:>4}  {snap['r_l4']:>10.3e}  {snap['r_l23']:>10.3e}  "
                  f"{snap['r_l23_max']:>10.3e}  {snap['r_pv']:>10.3e}  "
                  f"{snap['r_h']:>10.3e}  {snap['r_h_max']:>10.3e}  "
                  f"{snap['has_nan']}")


def probe_B(n_steps: int = 1000) -> None:
    """Homeostasis only."""
    print()
    print("=" * 72)
    print(f"=== Probe B: homeo only × 3 stim × {n_steps} steps ===")
    print("=" * 72)
    for stim_name, stim_fn in make_stim_fns(ModelConfig()).items():
        net, cfg = build_net(seed=42)
        # don't zero homeo — let it run
        # But zero plasticity LRs if any exposed; phase="phase1"/"phase2" is about training.
        net.set_phase("phase2")  # just a default; we aren't calling plasticity
        res = run_probe(net, cfg, stim_fn, stim_name, n_steps,
                        homeo_on=True, plast_on=False)
        print(f"\n--- Probe B × {stim_name} ---")
        print(f"  {'t':>4}  {'r_l4':>10}  {'r_l23':>10}  {'r_l23_max':>10}  "
              f"{'r_pv':>10}  {'r_h':>10}  {'r_h_max':>10}  {'θ_l23':>10}  nan")
        for t, snap in sorted(res["snapshots"].items()):
            print(f"  {t:>4}  {snap['r_l4']:>10.3e}  {snap['r_l23']:>10.3e}  "
                  f"{snap['r_l23_max']:>10.3e}  {snap['r_pv']:>10.3e}  "
                  f"{snap['r_h']:>10.3e}  {snap['r_h_max']:>10.3e}  "
                  f"{snap['theta_l23']:>10.3e}  {snap['has_nan']}")


def probe_C(n_steps: int = 1000) -> None:
    """Plasticity only — isolate the `-wd*raw` decay term.

    Apply decay-only updates directly to the actual W_l4_l23_raw tensor in
    a freshly built net. Compare wd=0 vs wd=1e-5. Track effective softplus
    weight norm at several checkpoints.
    """
    print()
    print("=" * 72)
    print(f"=== Probe C: weight-decay only (hebb=0) × {n_steps} steps ===")
    print("=" * 72)
    print("Isolate the `-wd*raw` term by applying it to the actual network raw")
    print("weights (no forward pass; no hebb). Track softplus(raw).norm() over time.")
    print()

    for wd in (0.0, 1e-5, 1e-4):
        net, cfg = build_net(seed=42)
        # W_l4_l23_raw is an excitatory raw weight initialized negative.
        raw = None
        for name, p in net.l23_e.named_parameters():
            if "W_l4_l23_raw" in name:
                raw = p
                break
        if raw is None:
            print(f"  wd={wd:.0e}: cannot find W_l4_l23_raw")
            continue

        raw_t = raw.detach().clone()
        print(f"--- wd = {wd:.0e} ---")
        print(f"  raw.mean init = {raw_t.mean().item():.4e}")
        print(f"  softplus(raw).mean init = {F.softplus(raw_t).mean().item():.4e}")

        print(f"  {'t':>5}  {'raw.mean':>12}  {'raw.max':>12}  "
              f"{'sp.mean':>12}  {'sp.max':>12}  {'sp.norm':>12}")
        for t in range(n_steps + 1):
            if t in (0, 10, 100, 500, 1000):
                sp = F.softplus(raw_t)
                print(f"  {t:>5}  {raw_t.mean().item():>12.4e}  "
                      f"{raw_t.max().item():>12.4e}  "
                      f"{sp.mean().item():>12.4e}  {sp.max().item():>12.4e}  "
                      f"{sp.norm().item():>12.4e}")
            if t < n_steps:
                raw_t = raw_t - wd * raw_t  # decay-only update

        print()


# ============================================================================
# Claim 5: full-state Jacobian
# ============================================================================

def compute_jacobian_eigs(
    net: V2Network, cfg: ModelConfig, stim_fn: Callable[[int], Tensor],
    warmup_steps: int = 50, eps: float = 1e-3,
) -> dict:
    """Roll out to near op point, then compute finite-diff Jacobian of forward wrt state."""
    B = 1
    zero_homeo(net)
    state = net.initial_state(batch_size=B)
    # Roll out to operating point
    with torch.no_grad():
        for _ in range(warmup_steps):
            frame = stim_fn(B)
            _, state, _ = net.forward(frame, state)

    # Sanity: check for nan/inf
    flat = flat_state(state)
    if not torch.isfinite(flat).all():
        return dict(error="state has non-finite values; op point not valid",
                    state_norm=flat.abs().max().item())

    # Compute finite-diff Jacobian
    frame = stim_fn(B)
    with torch.no_grad():
        _, state_next, _ = net.forward(frame, state)
    flat_next = flat_state(state_next)
    n = flat_next.numel()

    # Compute FULL Jacobian (n columns of forward differences).
    flat_cpu = flat.clone().double()
    J = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        perturbed = flat_cpu.clone()
        perturbed[i] += eps
        s_perturbed = unflat_state(state, perturbed.float())
        with torch.no_grad():
            _, s_next_p, _ = net.forward(frame, s_perturbed)
        flat_next_p = flat_state(s_next_p).double()
        J[:, i] = (flat_next_p - flat_next.double()) / eps

    # Eigenvalues of full J
    try:
        eigs = torch.linalg.eigvals(J)
        mags = eigs.abs().numpy()
        top5 = sorted(mags, reverse=True)[:5]
    except Exception as e:
        return dict(error=f"eigvals failed: {e}")

    # Per-population block sizes
    block_sizes = {}
    offset = 0
    for name in ("r_l4", "r_l23", "r_pv", "r_som", "r_h"):
        if hasattr(state, name):
            v = getattr(state, name)
            block_sizes[name] = (offset, offset + v.numel())
            offset += v.numel()

    # Block-diagonal spectral norms
    block_spectra = {}
    for name, (a, b) in block_sizes.items():
        if b > a:
            block = J[a:b, a:b]
            try:
                block_eigs = torch.linalg.eigvals(block)
                block_spectra[name] = block_eigs.abs().max().item()
            except Exception:
                block_spectra[name] = float('nan')

    return dict(
        state_norm=flat.abs().max().item(),
        jac_shape=tuple(J.shape),
        top5_abs_eig=top5,
        jac_frobenius=J.norm().item(),
        block_sizes=block_sizes,
        block_spectra=block_spectra,
    )


def claim5_full_jacobian() -> None:
    print()
    print("=" * 72)
    print("=== Claim 5: full-state Jacobian |λ|max across 3 stim conditions ===")
    print("=" * 72)
    print()
    print("Interpretation:")
    print("  Per-population Jacobians bound |λ| of blocks; the full inter-population")
    print("  Jacobian includes cross-blocks. If full |λ|max > per-pop bound, the stability")
    print("  guard is incomplete.")
    print()

    # Use SHORT warmup — Probe A showed the network diverges after ~50 steps
    # so we compute Jacobian at a still-finite op point (warmup=5, 10).
    for warmup in (0, 5, 10):
        print(f"### warmup = {warmup} ###")
        for stim_name, stim_fn in make_stim_fns(ModelConfig()).items():
            net, cfg = build_net(seed=42)
            res = compute_jacobian_eigs(net, cfg, stim_fn, warmup_steps=warmup, eps=1e-3)
            # Fake the print below to show warmup+stim
            stim_name = f"warmup={warmup} × {stim_name}"
            print(f"--- Claim 5 × {stim_name} ---")
            if "error" in res:
                print(f"  ERROR: {res['error']}")
                continue
            print(f"  state.max|.| (op point) = {res['state_norm']:.3e}")
            print(f"  J shape                  = {res['jac_shape']}")
            print(f"  J Frobenius              = {res['jac_frobenius']:.3e}")
            top = res["top5_abs_eig"]
            print(f"  full top 5 |λ|           = " +
                  ", ".join(f"{v:.4f}" for v in top))
            max_eig = top[0]
            if max_eig > 1.0:
                print(f"  |λ|max > 1.0 → UNSTABLE (growth × {max_eig:.4f}/step)")
            else:
                print(f"  |λ|max ≤ 1.0 → locally stable")
            if "block_spectra" in res:
                print(f"  per-population block |λ|max:")
                for name, v in res["block_spectra"].items():
                    flag = " ⚠" if v > 1.0 else ""
                    print(f"    {name:>5}: {v:.4f}{flag}")
            print()
        print()
        continue
    return


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="all",
                   choices=["all", "A", "B", "C", "claim5"])
    p.add_argument("--n-steps", type=int, default=1000)
    args = p.parse_args()
    if args.run in ("all", "A"):
        probe_A(n_steps=args.n_steps)
    if args.run in ("all", "B"):
        probe_B(n_steps=args.n_steps)
    if args.run in ("all", "C"):
        probe_C(n_steps=args.n_steps)
    if args.run in ("all", "claim5"):
        claim5_full_jacobian()
