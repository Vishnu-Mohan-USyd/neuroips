"""Task#74 SOM baseline investigation — root-cause r_som ≈ 505/unit on
Phase-2 step_3000 ckpt with task weights zeroed.

Evidence-only. 4 hypotheses:
  H1 fan-in miscalibration (w_l23_som_eff·r_l23 + w_fb_som_eff·r_h magnitude)
  H2 theta_som never moved during Phase-2 (NB: SOM uses fixed target_rate_hz,
     no adaptive theta — likely architecturally falsified)
  H3 softplus unbounded (drive >> threshold)
  H4 missing divisive stabilizer (architectural)

Procedure: forward 5 probes on Phase-2 ckpt (task weights = 0). Capture
r_l23, r_h, r_som per step during probe1. Decompose SOM drive into local +
feedback. Compute all four hypotheses' numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, build_cue_tensor, cue_mapping_from_seed,
)
from src.v2_model.layers import _excitatory_eff


def main() -> int:
    ckpt = ROOT / "checkpoints/v2/phase2/phase2_task70_s42/phase2_s42/step_3000.pt"
    bundle = load_checkpoint(ckpt, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    bundle.net.eval()
    with torch.no_grad():
        bundle.net.context_memory.W_mh_task_exc.zero_()
        bundle.net.context_memory.W_mh_task_inh.zero_()
        bundle.net.context_memory.W_qm_task.zero_()

    cfg = bundle.cfg
    cue_map = cue_mapping_from_seed(42)
    timing = KokTiming()
    net = bundle.net
    som = net.l23_som
    target = som.target_rate_hz
    w_l23_som_eff = _excitatory_eff(som.W_l23_som_raw).detach()   # [n_som, n_l23]
    w_fb_som_eff = _excitatory_eff(som.W_fb_som_raw).detach()     # [n_som, n_h]
    # Sign note: inhibitory pop — signs handled at L23E consumer side; the
    # local drive here is the positive pre-rectified input.
    n_som, n_l23 = w_l23_som_eff.shape
    _, n_h = w_fb_som_eff.shape
    print(f"[setup] n_som={n_som} n_l23={n_l23} n_h={n_h} "
          f"target_rate_hz={target}", file=sys.stderr)
    print(f"[setup] w_l23_som_eff: mean={w_l23_som_eff.mean().item():.4e} "
          f"max={w_l23_som_eff.max().item():.4e} "
          f"rowsum_mean={w_l23_som_eff.sum(1).mean().item():.4e}",
          file=sys.stderr)
    print(f"[setup] w_fb_som_eff:  mean={w_fb_som_eff.mean().item():.4e} "
          f"max={w_fb_som_eff.max().item():.4e} "
          f"rowsum_mean={w_fb_som_eff.sum(1).mean().item():.4e}",
          file=sys.stderr)

    # H4 — architectural check: does SOM forward subtract anything except
    # target_rate_hz? Already read code: no divisive stabilizer, no PV→SOM.
    has_div_stabilizer = False  # confirmed by reading src/v2_model/layers.py:670-705
    h2_theta_is_adaptive = hasattr(som, "theta") and isinstance(
        getattr(som, "theta", None), torch.Tensor)

    # Run probes and capture drive decomposition at each probe1 step
    drives_local = []
    drives_fb = []
    drives_total = []
    activated_all = []
    r_l23_all = []
    r_h_all = []
    r_som_all = []

    for i in range(5):
        cue_id = i % 2
        probe_deg = cue_map[cue_id]
        blank = make_blank_frame(1, cfg, device=cfg.device)
        probe = make_grating_frame(float(probe_deg), 1.0, cfg, device=cfg.device)
        q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=cfg.device)

        state = net.initial_state(batch_size=1)
        cue_end = timing.cue_steps
        delay_end = cue_end + timing.delay_steps
        probe1_end = delay_end + timing.probe1_steps
        n_total = timing.total

        with torch.no_grad():
            for t in range(n_total):
                if t < cue_end:
                    frame, q_t = blank, q_cue
                elif t < delay_end:
                    frame, q_t = blank, None
                elif t < probe1_end:
                    frame, q_t = probe, None
                else:
                    frame, q_t = blank, None
                _x, state, info = net(frame, state, q_t=q_t)

                if delay_end <= t < probe1_end:
                    r_l23 = info["r_l23"][0].detach()        # [n_l23]
                    r_h = info["r_h"][0].detach()            # [n_h]
                    r_som = info["r_som"][0].detach()        # [n_som]
                    d_local = F.linear(r_l23.unsqueeze(0), w_l23_som_eff).squeeze(0)
                    d_fb = F.linear(r_h.unsqueeze(0), w_fb_som_eff).squeeze(0)
                    d_total = d_local + d_fb
                    activated = F.softplus(d_total - target) if False else torch.clamp_min(
                        d_total - target, 0.0)  # approx magnitude (we want pre-leak size)
                    drives_local.append(d_local)
                    drives_fb.append(d_fb)
                    drives_total.append(d_total)
                    activated_all.append(activated)
                    r_l23_all.append(r_l23)
                    r_h_all.append(r_h)
                    r_som_all.append(r_som)

    # stack over (5 probes × probe1_steps) total timesteps
    drives_local = torch.stack(drives_local, dim=0)
    drives_fb = torch.stack(drives_fb, dim=0)
    drives_total = torch.stack(drives_total, dim=0)
    r_l23_t = torch.stack(r_l23_all, dim=0)
    r_h_t = torch.stack(r_h_all, dim=0)
    r_som_t = torch.stack(r_som_all, dim=0)

    # ========= H1 — fan-in miscalibration =========
    w_l23_som_rowsum_mean = float(w_l23_som_eff.sum(1).mean().item())
    w_fb_som_rowsum_mean = float(w_fb_som_eff.sum(1).mean().item())
    d_local_mean = float(drives_local.mean().item())
    d_fb_mean = float(drives_fb.mean().item())
    d_total_mean = float(drives_total.mean().item())
    # drive scales with r_l23 × fan-in_sum. Expected per Task#52 comment at
    # r_l23=0.012: local ≈ 0.96; here we measure actual.
    r_l23_mean = float(r_l23_t.mean().item())
    r_h_mean = float(r_h_t.mean().item())
    h1_expected_local = 256 * F.softplus(torch.tensor(-1.0)).item() * r_l23_mean
    h1_expected_fb = 64 * F.softplus(torch.tensor(-5.0)).item() * r_h_mean
    h1_expected_total = h1_expected_local + h1_expected_fb
    # pass = within factor 3 of design; fail otherwise
    h1_ratio_total = d_total_mean / max(target, 1e-12)
    h1_verdict = "fail" if h1_ratio_total > 10.0 else ("pass" if h1_ratio_total < 3.0 else "partial")
    print(f"\n[H1 fan-in] w_l23_som_rowsum_mean={w_l23_som_rowsum_mean:.4e} "
          f"w_fb_som_rowsum_mean={w_fb_som_rowsum_mean:.4e}", file=sys.stderr)
    print(f"[H1] drive_local_mean={d_local_mean:.4e} drive_fb_mean={d_fb_mean:.4e} "
          f"drive_total_mean={d_total_mean:.4e}", file=sys.stderr)
    print(f"[H1] r_l23_mean={r_l23_mean:.4e} r_h_mean={r_h_mean:.4e} "
          f"expected_total(design)={h1_expected_total:.4e} actual/design="
          f"{d_total_mean / max(h1_expected_total, 1e-12):.3f}×", file=sys.stderr)

    # ========= H2 — theta adaptive? =========
    if h2_theta_is_adaptive:
        theta_vec = som.theta.detach()
        theta_mean = float(theta_vec.mean().item())
        # init value — unknown without training log; skip delta
        h2_verdict = "unknown"
    else:
        theta_mean = float(target)
        h2_verdict = "falsified"  # SOM has no adaptive θ, only fixed target_rate_hz
    print(f"\n[H2] theta_is_adaptive={h2_theta_is_adaptive} "
          f"target_rate_hz(fixed)={target}", file=sys.stderr)

    # ========= H3 — softplus unbounded (drive ≫ threshold) =========
    drive_over_target = drives_total / max(target, 1e-12)
    drive_to_threshold_ratio_mean = float(drive_over_target.mean().item())
    drive_to_threshold_ratio_max = float(drive_over_target.max().item())
    h3_verdict = "fail" if drive_to_threshold_ratio_mean > 10.0 else (
        "pass" if drive_to_threshold_ratio_mean < 1.5 else "partial")
    print(f"\n[H3] drive_total_mean={d_total_mean:.4e} target={target} "
          f"ratio_mean={drive_to_threshold_ratio_mean:.2f} "
          f"ratio_max={drive_to_threshold_ratio_max:.2f}", file=sys.stderr)

    # ========= H4 — divisive stabilizer =========
    h4_verdict = "fail" if not has_div_stabilizer else "pass"
    # note: "fail" here = hypothesis CONFIRMED (stabilizer missing)
    print(f"\n[H4] divisive_stabilizer={'missing' if not has_div_stabilizer else 'exists'}",
          file=sys.stderr)

    # ========= Summary =========
    print(f"\nr_som observed: mean={float(r_som_t.mean().item()):.4e} "
          f"max={float(r_som_t.max().item()):.4e}")
    print(f"r_l23 observed: mean={r_l23_mean:.4e} "
          f"max={float(r_l23_t.max().item()):.4e}")
    print(f"r_h observed:   mean={r_h_mean:.4e} "
          f"max={float(r_h_t.max().item()):.4e}")
    print(f"\nH1: h1_verdict={h1_verdict} drive_som={d_total_mean:.4e} "
          f"w_l23_som_eff={w_l23_som_rowsum_mean:.4e} "
          f"w_fb_som_eff={w_fb_som_rowsum_mean:.4e}")
    print(f"H2: h2_verdict={h2_verdict} theta_som_mean={theta_mean:.4e} "
          f"(SOM has no adaptive theta; uses fixed target_rate_hz={target})")
    print(f"H3: h3_verdict={h3_verdict} drive_mean={d_total_mean:.4e} "
          f"theta_mean={target} ratio={drive_to_threshold_ratio_mean:.2f}")
    print(f"H4: h4_verdict={h4_verdict} "
          f"div_stabilizer={'missing' if not has_div_stabilizer else 'exists'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
