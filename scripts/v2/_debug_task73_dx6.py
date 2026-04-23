"""Task #73 Dx6 — time-resolved b_task(t), r_l23(t), Δamp(t).

Goal: localize IN TIME where the template fails to translate into a probe-
response effect. Competing predictions:
  H-template-weak : b_task present in delay; r_l23 Δamp emerges early in
                    probe window but small magnitude throughout
  H-template-decays-at-probe : b_task drops to 0 when probe arrives (m
                    reset), so no probe-window Δamp
  H-lateral-integration-slow : b_task persists but Δamp only emerges late
                    after SOM/PV circuit integrates
  H-amplified-late : with ×5 scale, Δamp emerges late (lateral integration
                    requires threshold)

Collect r_l23(t), ‖b_task(t)‖, ‖m(t)‖ for t across cue+delay+probe1.
Conditions: baseline (scale=1), amplified (scale=5).
Per-trial time courses aggregated over 20 trials × 4 sub-conds = 80 trials
per condition.

Time bins within probe1:
  early    = steps [0, 10)    (0-50ms)
  mid      = steps [10, 30)   (50-150ms)
  late     = steps [30, 100)  (150-500ms)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
)


@torch.no_grad()
def run_probe_trial_timecourse(bundle, cue_id, probe_deg, timing,
                                noise_std, generator):
    """Return r_l23(t), ||b_task(t)||, ||m(t)|| for ALL timesteps.
    Shape: r_l23_t [n_total, n_l23], b_norm_t [n_total], m_norm_t [n_total].
    """
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(float(probe_deg), 1.0, cfg, device=device)
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)
    W_task = bundle.net.context_memory.W_mh_task.data

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    r_t, b_t, m_t = [], [], []
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
                frame.shape, generator=generator, device=device)
        _x, state, info = bundle.net(frame, state, q_t=q_t)
        r_t.append(info["r_l23"][0].cpu().numpy())
        b = (W_task @ state.m[0]).cpu().numpy()
        b_t.append(float(np.linalg.norm(b)))
        m_t.append(float(np.linalg.norm(state.m[0].cpu().numpy())))
    return np.stack(r_t, axis=0), np.asarray(b_t), np.asarray(m_t)


def run_condition(ckpt_path, scale, n_trials=20, noise_std=0.01, seed=42):
    bundle = load_checkpoint(ckpt_path, seed=seed, device="cpu")
    bundle.net.set_phase("phase3_kok")
    W_orig = bundle.net.context_memory.W_mh_task.data.clone()
    bundle.net.context_memory.W_mh_task.data.copy_(scale * W_orig)

    cue_mapping = bundle.meta.get("cue_mapping", cue_mapping_from_seed(seed))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}
    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)

    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps

    # Bins within probe1:
    probe_len = probe1_end - delay_end
    bin_early = (delay_end + 0, delay_end + min(10, probe_len))
    bin_mid = (delay_end + min(10, probe_len),
               delay_end + min(30, probe_len))
    bin_late = (delay_end + min(30, probe_len), probe1_end)

    # Collect trials
    trial_data = {}  # (cue_id, is_exp) -> {r_t, b_t, m_t} stacked
    for cue_id in (0, 1):
        cue_probe = cue_mapping[cue_id]
        other_probe = cue_mapping[1 - cue_id]
        for probe_deg, is_exp in ((cue_probe, True), (other_probe, False)):
            all_r, all_b, all_m = [], [], []
            for _ in range(n_trials):
                r_t, b_t, m_t = run_probe_trial_timecourse(
                    bundle, cue_id, probe_deg, timing, noise_std, gen)
                all_r.append(r_t); all_b.append(b_t); all_m.append(m_t)
            trial_data[(cue_id, is_exp)] = {
                "r_t": np.stack(all_r, axis=0),   # [n_trials, n_total, n_l23]
                "b_t": np.stack(all_b, axis=0),   # [n_trials, n_total]
                "m_t": np.stack(all_m, axis=0),
            }

    # Aggregate across cues: expected vs unexpected
    r_exp = np.concatenate(
        [trial_data[(c, True)]["r_t"] for c in (0, 1)], axis=0)
    r_unexp = np.concatenate(
        [trial_data[(c, False)]["r_t"] for c in (0, 1)], axis=0)
    b_exp = np.concatenate(
        [trial_data[(c, True)]["b_t"] for c in (0, 1)], axis=0)
    b_unexp = np.concatenate(
        [trial_data[(c, False)]["b_t"] for c in (0, 1)], axis=0)

    # Mean r_l23 averaged across units and trials per time bin:
    def _mean_in_range(arr, lo, hi):
        return float(arr[:, lo:hi, ...].mean())

    delta_amp_early = (_mean_in_range(r_exp, bin_early[0], bin_early[1]) -
                       _mean_in_range(r_unexp, bin_early[0], bin_early[1]))
    delta_amp_mid = (_mean_in_range(r_exp, bin_mid[0], bin_mid[1]) -
                     _mean_in_range(r_unexp, bin_mid[0], bin_mid[1]))
    delta_amp_late = (_mean_in_range(r_exp, bin_late[0], bin_late[1]) -
                      _mean_in_range(r_unexp, bin_late[0], bin_late[1]))

    b_delay = float(b_exp[:, cue_end:delay_end].mean())
    b_probe_early = float(b_exp[:, bin_early[0]:bin_early[1]].mean())
    b_probe_mid = float(b_exp[:, bin_mid[0]:bin_mid[1]].mean())
    b_probe_late = float(b_exp[:, bin_late[0]:bin_late[1]].mean())

    return {
        "scale": scale,
        "W_mh_task_norm": float(
            bundle.net.context_memory.W_mh_task.data.norm().item()),
        "bin_early_steps": bin_early,
        "bin_mid_steps": bin_mid,
        "bin_late_steps": bin_late,
        "delta_amp_probe_early": delta_amp_early,
        "delta_amp_probe_mid": delta_amp_mid,
        "delta_amp_probe_late": delta_amp_late,
        "b_task_norm_delay_mean": b_delay,
        "b_task_norm_probe_early_mean": b_probe_early,
        "b_task_norm_probe_mid_mean": b_probe_mid,
        "b_task_norm_probe_late_mean": b_probe_late,
    }


def main(ckpt_path, n_trials=20):
    results = {}
    for scale in (1.0, 5.0):
        print(f"[Dx6] scale={scale}...", file=sys.stderr, flush=True)
        results[f"scale_{scale}"] = run_condition(
            ckpt_path, scale, n_trials=n_trials)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt"))
    p.add_argument("--n-trials", type=int, default=20)
    args = p.parse_args()
    main(args.ckpt, args.n_trials)
