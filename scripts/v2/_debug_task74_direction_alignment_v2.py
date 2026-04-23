"""Task#74 direction-alignment — v2 with DEMEANED cos + effective-gain test.

First-pass v1 found cos(W·m, r_som_loc)=1.0 but that is trivial: W is 89.6%
saturated at +8 (all positive), and r_som_loc is nearly uniform (SOM only
has 1 preferred-orient bin). Both are near-constant vectors; cos of
constants = 1.

The real test needs demeaning (remove the uniform offset) and must look at
the EFFECTIVE signal — som_gain = softplus(W·m + 0.5413).clamp(max=4.0)
post-softplus-clamp — because that's what actually modulates SOM.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

# reload saved JSON; recompute the couple of metrics we need from fresh data
prev = json.loads(
    (ROOT / "logs/task74/direction_alignment_s42.json").read_text())

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


def _run_probe_capture(bundle, cue_id, probe_deg, timing, noise_std, gen):
    cfg = bundle.cfg
    blank = make_blank_frame(1, cfg, device=cfg.device)
    probe = make_grating_frame(float(probe_deg), 1.0, cfg, device=cfg.device)
    q_cue = (build_cue_tensor(int(cue_id), cfg.arch.n_c, device=cfg.device)
             if cue_id is not None else None)
    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    som_p = []; m_dec = None
    with torch.no_grad():
        for t in range(timing.total):
            if t < cue_end:
                frame, q_t = blank, q_cue
            elif t < delay_end:
                frame, q_t = blank, None
            elif t < probe1_end:
                frame, q_t = probe, None
            else:
                frame, q_t = blank, None
            _x, state, info = bundle.net(frame, state, q_t=q_t)
            if t == delay_end - 1:
                m_dec = state.m.detach().clone()
            if delay_end <= t < probe1_end:
                som_p.append(info["r_som"][0].detach().clone())
    return torch.stack(som_p, 0).mean(0), m_dec.squeeze(0)


def main():
    ckpt = ROOT / "checkpoints/v2/phase3_kok_task74D/phase3_kok_s42.pt"
    bundle = load_checkpoint(ckpt, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    bundle.net.eval()
    cue_map = bundle.meta.get("cue_mapping") or cue_mapping_from_seed(42)
    cue_map = {int(k): float(v) for k, v in cue_map.items()}

    W_t = bundle.net.context_memory.W_mh_task_inh.detach().clone()
    W = W_t.numpy()
    n_som, n_m = W.shape
    bias0 = 0.5413
    clamp_max = 4.0

    timing = KokTiming()
    gen = torch.Generator(device="cpu").manual_seed(42)

    # m_c and r_som_loc_c
    m_c = {}
    r_som_loc = {}
    for c in (0, 1):
        ms = []
        for _ in range(20):
            _, m_dec = _run_probe_capture(
                bundle, c, cue_map[c], timing, 0.0, gen)
            ms.append(m_dec.numpy())
        m_c[c] = np.stack(ms, 0).mean(0)
        ss = []
        for _ in range(15):
            s_mean, _ = _run_probe_capture(
                bundle, None, cue_map[c], timing, 0.0, gen)
            ss.append(s_mean.numpy())
        r_som_loc[c] = np.stack(ss, 0).mean(0)

    def _cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float((a @ b) / (na * nb))

    # ========== H1-refined: DEMEANED cos + EFFECTIVE som_gain ==========
    Wm_raw = {c: W @ m_c[c] for c in (0, 1)}
    som_gain_eff = {c: np.minimum(
        np.log1p(np.exp(Wm_raw[c] + bias0)),
        clamp_max,
    ) for c in (0, 1)}

    # de-mean both sides → this is the ORIENTATION-specific contrast
    cos_raw_demeaned = {}
    cos_eff_demeaned = {}
    for c in (0, 1):
        a = Wm_raw[c] - Wm_raw[c].mean()
        b = r_som_loc[c] - r_som_loc[c].mean()
        cos_raw_demeaned[c] = _cos(a, b)
        a = som_gain_eff[c] - som_gain_eff[c].mean()
        b = r_som_loc[c] - r_som_loc[c].mean()
        cos_eff_demeaned[c] = _cos(a, b)

    mean_cos_raw = float(np.mean(list(cos_raw_demeaned.values())))
    mean_cos_eff = float(np.mean(list(cos_eff_demeaned.values())))

    # Contrast of som_gain_eff (the variance of the gain across SOM units)
    gain_std = {c: float(som_gain_eff[c].std()) for c in (0, 1)}
    gain_mean = {c: float(som_gain_eff[c].mean()) for c in (0, 1)}
    # Fraction of SOM units at gain_clamp=4
    gain_at_clamp = {c: float((som_gain_eff[c] >= clamp_max - 1e-6).mean())
                     for c in (0, 1)}

    # Shuffle ctrl with DEMEANED — redraw signs
    rng = np.random.default_rng(0)
    shuffle_cos_eff = []
    for _ in range(200):
        signs = rng.choice([-1.0, 1.0], size=W.shape)
        W_shuf = np.abs(W) * signs
        for c in (0, 1):
            Wm_s = W_shuf @ m_c[c]
            gain_s = np.minimum(np.log1p(np.exp(Wm_s + bias0)), clamp_max)
            a = gain_s - gain_s.mean()
            b = r_som_loc[c] - r_som_loc[c].mean()
            shuffle_cos_eff.append(_cos(a, b))
    shuf_mean = float(np.mean(shuffle_cos_eff))
    shuf_std = float(np.std(shuffle_cos_eff))
    gap_eff = mean_cos_eff - shuf_mean
    h1_verdict = "CONFIRMED" if abs(gap_eff) < 2 * shuf_std else "FALSIFIED"

    print(f"\n[H1-refined] W pct at +clamp={float((W >= 7.99).mean())*100:.1f}% "
          f"pct near 0={float((np.abs(W) < 0.5).mean())*100:.1f}% "
          f"pct at -clamp={float((W <= -7.99).mean())*100:.1f}%", file=sys.stderr)
    print(f"[H1-refined] cos(W·m_demeaned, r_som_loc_demeaned): "
          f"c0={cos_raw_demeaned[0]:.4f} c1={cos_raw_demeaned[1]:.4f} "
          f"mean={mean_cos_raw:.4f}", file=sys.stderr)
    print(f"[H1-refined] cos(som_gain_eff_demeaned, r_som_loc_demeaned): "
          f"c0={cos_eff_demeaned[0]:.4f} c1={cos_eff_demeaned[1]:.4f} "
          f"mean={mean_cos_eff:.4f}", file=sys.stderr)
    print(f"[H1-refined] som_gain_eff: mean c0={gain_mean[0]:.3f} c1={gain_mean[1]:.3f} "
          f"std c0={gain_std[0]:.4f} c1={gain_std[1]:.4f} "
          f"pct_at_clamp c0={gain_at_clamp[0]*100:.1f}% "
          f"c1={gain_at_clamp[1]*100:.1f}%", file=sys.stderr)
    print(f"[H1-refined] shuffle-ctrl (demeaned eff): mean={shuf_mean:.4f} "
          f"std={shuf_std:.4f} gap={gap_eff:.4f} verdict={h1_verdict}",
          file=sys.stderr)

    # Report synthesis numbers at end
    print(f"\n--- summary numbers for DM ---")
    print(f"H1 raw demeaned cos: c0={cos_raw_demeaned[0]:.4f} c1={cos_raw_demeaned[1]:.4f} mean={mean_cos_raw:.4f}")
    print(f"H1 eff demeaned cos: c0={cos_eff_demeaned[0]:.4f} c1={cos_eff_demeaned[1]:.4f} mean={mean_cos_eff:.4f}")
    print(f"H1 som_gain_eff std: c0={gain_std[0]:.4f} c1={gain_std[1]:.4f}")
    print(f"H1 som_gain_eff pct_at_clamp: c0={gain_at_clamp[0]:.3f} c1={gain_at_clamp[1]:.3f}")
    print(f"H1 shuffle: mean={shuf_mean:.4f} std={shuf_std:.4f}")
    print(f"H1 verdict={h1_verdict}")

    out = {
        "H1_refined": {
            "verdict": h1_verdict,
            "cos_raw_demeaned_per_class": cos_raw_demeaned,
            "cos_eff_demeaned_per_class": cos_eff_demeaned,
            "mean_cos_raw_demeaned": mean_cos_raw,
            "mean_cos_eff_demeaned": mean_cos_eff,
            "som_gain_eff_mean_per_class": gain_mean,
            "som_gain_eff_std_per_class": gain_std,
            "som_gain_eff_pct_at_clamp_per_class": gain_at_clamp,
            "shuffle_ctrl_mean": shuf_mean,
            "shuffle_ctrl_std": shuf_std,
            "gap": gap_eff,
            "W_pct_at_pos_clamp": float((W >= 7.99).mean()),
            "W_pct_near_zero": float((np.abs(W) < 0.5).mean()),
        },
    }
    Path("logs/task74/direction_alignment_v2_s42.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
