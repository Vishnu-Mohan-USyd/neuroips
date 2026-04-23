"""Task #61 — per-step term decomposition + raw-max tracking on W_pv_l23_raw.

Addresses the team-lead's 7-line requested format by running:
  (1) Per-step snapshot on the SPECIFIC weight l23_e.W_pv_l23_raw,
      decomposing dw into {hebb, decay, shrink} components each step.
  (2) raw.max() trajectory to find when raw crosses +5.
  (3) Init-mean shift: override raw_init_means["W_pv_l23_raw"] from -5 → -4
      and compare explosion timing.
  (4) Falsification: rerun with weight_decay=1e-1 (×10000 normal 1e-5) and
      see if the raw-prior pull can arrest shrink domination.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from scripts.v2.train_phase2_predictive import (
    PlasticityRuleBank, build_world, sample_batch_window, _forward_window,
    _raw_prior, _soft_reset_state,
)


@dataclass
class TermSnap:
    step: int
    raw_max: float
    raw_mean: float
    softplus_mean: float
    hebb_mag: float           # mean |lr * hebb|
    decay_mag: float          # mean |wd * (w - prior)|
    shrink_mag: float         # mean |-beta * pre^2 * w|
    total_mag: float          # mean |total applied|
    r_pv_mean: float
    r_l23_max: float
    any_nan: bool


def run_with_terms(
    seed: int = 42,
    n_steps: int = 300,
    batch_size: int = 4,
    warmup_steps: int = 30,
    segment_length: int = 50,
    soft_reset_scale: float = 0.1,
    weight_decay: float = 1e-5,
    beta_syn: float = 1e-4,
    override_init_mean: float | None = None,
) -> list[TermSnap]:
    """Run Phase-2 with per-step term decomposition on W_pv_l23_raw.

    If override_init_mean is given, override net.l23_e.raw_init_means["W_pv_l23_raw"]
    AND initialize the weight tensor to that value (so raw_prior matches).
    """
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    world, bank = build_world(cfg, seed_family="train", held_out_regime=None)
    net = V2Network(cfg, token_bank=bank, seed=seed)
    net.set_phase("phase2")

    if override_init_mean is not None:
        # overwrite raw_init_means entry + reinit weight
        if hasattr(net.l23_e, "raw_init_means"):
            net.l23_e.raw_init_means["W_pv_l23_raw"] = float(override_init_mean)
        w = net.l23_e.W_pv_l23_raw
        with torch.no_grad():
            w.fill_(float(override_init_mean))

    rules = PlasticityRuleBank.from_config(
        cfg=cfg,
        lr_urbanczik=1e-4, lr_vogels=1e-4, lr_hebb=1e-4,
        weight_decay=weight_decay, beta_syn=beta_syn,
    )

    state = net.initial_state(batch_size=batch_size)

    # Warmup (forward only, no plasticity).
    for w in range(warmup_steps):
        seeds = [seed * 10_000 + w * batch_size + b for b in range(batch_size)]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)
        _, _, state, _, _, _, _ = _forward_window(net, frames, state)

    snaps: list[TermSnap] = []
    for step in range(n_steps):
        seeds = [
            seed * 10_000 + (warmup_steps + step) * batch_size + b
            for b in range(batch_size)
        ]
        frames = sample_batch_window(world, seeds, n_steps_per_window=2)

        try:
            state0, state1, state2, info0, info1, x_hat_0, _ = _forward_window(
                net, frames, state,
            )
        except Exception as e:
            print(f"[step {step}] forward exception: {e}")
            break

        # --- isolated term decomposition for W_pv_l23_raw (BEFORE updating) --
        w_raw = net.l23_e.W_pv_l23_raw
        pre = state1.r_pv           # Vogels pre_activity
        post = state2.r_l23         # Vogels post_activity
        target = cfg.plasticity.vogels_target_l23e_hz
        lr = rules.vogels_l23.lr
        wd = rules.vogels_l23.weight_decay
        beta = rules.energy.beta

        prior = _raw_prior(net, "l23_e", "W_pv_l23_raw", w_raw)
        post_dev = post - target                      # [B, 256]
        hebb = (post_dev.t() @ pre) / float(post.shape[0])  # [256, 16]
        hebb_term = lr * hebb
        decay_term = wd * (w_raw - prior)             # = -(dw decay contribution)
        # total shrink applied in _apply_update:
        pre_sq = (pre * pre).mean(dim=0)
        shrink_term = -beta * pre_sq * w_raw
        # applied-to-w value = lr*hebb - wd*(w-prior) + shrink
        total_applied = hebb_term - decay_term + shrink_term

        raw_max = float(w_raw.abs().max().item())
        raw_mean = float(w_raw.mean().item())
        softp = float(F.softplus(w_raw).mean().item())

        snap = TermSnap(
            step=step,
            raw_max=raw_max,
            raw_mean=raw_mean,
            softplus_mean=softp,
            hebb_mag=float(hebb_term.abs().mean().item()),
            decay_mag=float(decay_term.abs().mean().item()),
            shrink_mag=float(shrink_term.abs().mean().item()),
            total_mag=float(total_applied.abs().mean().item()),
            r_pv_mean=float(pre.mean().item()),
            r_l23_max=float(state2.r_l23.abs().max().item()),
            any_nan=bool(not torch.isfinite(w_raw).all().item()
                         or not torch.isfinite(state2.r_l23).all().item()),
        )
        snaps.append(snap)

        # actually apply the real Phase-2 plasticity step (uses the same rules)
        from scripts.v2.train_phase2_predictive import apply_plasticity_step
        try:
            apply_plasticity_step(
                net, rules, state0, state1, state2, info0, info1, x_hat_0,
            )
        except Exception as e:
            print(f"[step {step}] apply_plasticity exception: {e}")
            break

        state = state2
        if (
            segment_length > 0
            and (step + 1) % segment_length == 0
            and (step + 1) < n_steps
        ):
            state = _soft_reset_state(state, scale=soft_reset_scale)

        if snap.any_nan:
            print(f"[step {step}] NaN detected — halting")
            break

    return snaps


def print_snaps(label: str, snaps: list[TermSnap]) -> None:
    print(f"\n{'='*78}\n=== {label}  (steps={len(snaps)}) ===\n{'='*78}")
    print(f"{'step':>5} {'raw_max':>10} {'raw_mean':>10} {'softp':>10} "
          f"{'hebb':>11} {'decay':>11} {'shrink':>11} "
          f"{'total':>11} {'r_pv':>8} {'r_l23_mx':>8} {'nan':>4}")
    for s in snaps:
        if (s.step < 3 or s.step % 10 == 0 or s.step == len(snaps) - 1
                or s.any_nan):
            print(f"{s.step:>5} {s.raw_max:>10.3e} {s.raw_mean:>+10.3e} "
                  f"{s.softplus_mean:>10.3e} "
                  f"{s.hebb_mag:>11.3e} {s.decay_mag:>11.3e} "
                  f"{s.shrink_mag:>11.3e} {s.total_mag:>11.3e} "
                  f"{s.r_pv_mean:>8.2e} {s.r_l23_max:>8.2e} {s.any_nan!s:>4}")

    # When does raw_max cross 5?
    crossings = [s.step for s in snaps if s.raw_max > 5.0]
    first_cross = crossings[0] if crossings else None
    print(f"\n  raw_max first crosses 5.0 at step: {first_cross}")

    # first explosive step (|total| > 1e3)
    expl = next((s.step for s in snaps if s.total_mag > 1e3), None)
    print(f"  total |Δw|>1e3 first at step: {expl}")


def main() -> None:
    # --- baseline with decomposition ---
    print("\n\n[Baseline: default Task #56 config — lr=1e-4, wd=1e-5, beta=1e-4]")
    base = run_with_terms(seed=42, n_steps=300)
    print_snaps("baseline", base)

    # --- init mean shift: -5.0 → -4.0 ---
    print("\n\n[Init-mean shift: W_pv_l23_raw init -5.0 → -4.0]")
    init_shift = run_with_terms(seed=42, n_steps=300, override_init_mean=-4.0)
    print_snaps("init_mean=-4.0", init_shift)

    # --- weight_decay ×10000 to falsify raw-prior coefficient weakness ---
    print("\n\n[Falsification: weight_decay=1e-1 (×10000 normal)]")
    wd_huge = run_with_terms(seed=42, n_steps=300, weight_decay=1e-1)
    print_snaps("wd=1e-1", wd_huge)

    # --- summary ---
    print("\n\n" + "=" * 78)
    print("=== SUMMARY (for team-lead key=value) ===")
    print("=" * 78)

    def first_expl(snaps):
        s = next((x.step for x in snaps if x.total_mag > 1e3), None)
        return s

    def raw_cross5(snaps):
        return next((x.step for x in snaps if x.raw_max > 5.0), None)

    def pre_expl_ratio(snaps):
        """hebb / decay at step just BEFORE first explosion."""
        expl = first_expl(snaps)
        if expl is None or expl == 0:
            return float("nan"), float("nan"), float("nan")
        pre = snaps[expl - 1]
        return pre.hebb_mag, pre.decay_mag, pre.hebb_mag / max(1e-30, pre.decay_mag)

    for label, snaps in [("baseline", base), ("init=-4.0", init_shift),
                         ("wd=1e-1", wd_huge)]:
        expl = first_expl(snaps)
        cross = raw_cross5(snaps)
        h, d, r = pre_expl_ratio(snaps)
        halt = snaps[-1].step if snaps else -1
        print(f"\n[{label}]")
        print(f"  first_explosive_step = {expl}")
        print(f"  raw_max_first_crosses_5 = {cross}")
        print(f"  halted_step = {halt}  (NaN at end: {snaps[-1].any_nan if snaps else False})")
        print(f"  hebb_mag_pre_explosion = {h:.3e}")
        print(f"  decay_mag_pre_explosion = {d:.3e}")
        print(f"  hebb/decay_ratio_pre_explosion = {r:.3e}")


if __name__ == "__main__":
    main()
