"""Task#74 direction-alignment investigation — why cos(W_mh_task_inh @ m_c,
r_som_localizer_c) ≈ 0 despite rule_ratio_effective ≈ 156.

Evidence only. 4 hypotheses:
  H1 W saturated pseudo-random (sign not structured by class)
  H2 modulator near zero (exp vs unexp r_som barely differs)
  H3 SOM tuning doesn't mirror L23E tuning
  H4 SNR = modulator / r_som_baseline ≪ 1

Ckpt: checkpoints/v2/phase3_kok_task74D/phase3_kok_s42.pt (post-Fix-Cv2 Kok)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path("/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, build_cue_tensor, cue_mapping_from_seed,
)


def _run_probe_capture(bundle, cue_id, probe_deg, timing, noise_std, gen):
    """Run a Kok probe trial; return (r_l23_probe_mean, r_som_probe_mean, m_end_of_delay)."""
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(float(probe_deg), 1.0, cfg, device=device)
    q_cue = (build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)
             if cue_id is not None else None)
    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    n_total = timing.total
    l23_p, som_p = [], []
    m_decision = None
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
            if noise_std > 0.0:
                frame = frame + noise_std * torch.randn(
                    frame.shape, generator=gen, device=device)
            _x, state, info = bundle.net(frame, state, q_t=q_t)
            if t == delay_end - 1:
                m_decision = state.m.detach().clone()
            if delay_end <= t < probe1_end:
                l23_p.append(info["r_l23"][0].detach().clone())
                som_p.append(info["r_som"][0].detach().clone())
    l23_mean = torch.stack(l23_p, 0).mean(0)
    som_mean = torch.stack(som_p, 0).mean(0)
    return l23_mean, som_mean, m_decision.squeeze(0)


def main():
    ckpt = ROOT / "checkpoints/v2/phase3_kok_task74D/phase3_kok_s42.pt"
    bundle = load_checkpoint(ckpt, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    bundle.net.eval()

    cue_map = bundle.meta.get("cue_mapping") or cue_mapping_from_seed(42)
    cue_map = {int(k): float(v) for k, v in cue_map.items()}
    print(f"[setup] cue_mapping={cue_map}", file=sys.stderr)

    W = bundle.net.context_memory.W_mh_task_inh.detach().clone().numpy()
    n_som, n_m = W.shape
    print(f"[setup] W_mh_task_inh shape=({n_som}, {n_m}) "
          f"min={W.min():.4f} max={W.max():.4f} mean={W.mean():.4f} "
          f"pct_at_pos_clamp={(W >= 7.99).mean()*100:.1f}% "
          f"pct_near_zero={(np.abs(W) < 0.5).mean()*100:.1f}% "
          f"pct_at_neg_clamp={(W <= -7.99).mean()*100:.1f}%",
          file=sys.stderr)

    timing = KokTiming()
    gen = torch.Generator(device="cpu").manual_seed(42)
    noise_std = 0.0

    # ========== Build m_c per class (matched probe = expected) ==========
    n_per_cue = 20
    m_by_cue = {0: [], 1: []}
    for cue_id in (0, 1):
        for _ in range(n_per_cue):
            _, _, m_dec = _run_probe_capture(
                bundle, cue_id, cue_map[cue_id], timing, noise_std, gen)
            m_by_cue[cue_id].append(m_dec.numpy())
    m_c = {c: np.stack(v, 0).mean(0) for c, v in m_by_cue.items()}

    # ========== Build r_som_localizer_c per expected orientation ==========
    n_loc = 15
    som_loc_by_cue = {0: [], 1: []}
    l23_loc_by_cue = {0: [], 1: []}
    for cue_id in (0, 1):
        orient = cue_map[cue_id]
        for _ in range(n_loc):
            l23_mean, som_mean, _ = _run_probe_capture(
                bundle, None, orient, timing, noise_std, gen)
            som_loc_by_cue[cue_id].append(som_mean.numpy())
            l23_loc_by_cue[cue_id].append(l23_mean.numpy())
    r_som_loc = {c: np.stack(v, 0).mean(0) for c, v in som_loc_by_cue.items()}
    r_l23_loc = {c: np.stack(v, 0).mean(0) for c, v in l23_loc_by_cue.items()}

    # ========== Expected vs unexpected r_som per class (for modulator) =====
    # Expected = cue c + probe at cue_map[c]. Unexpected = cue c + probe at cue_map[1-c].
    n_trials = 15
    r_som_exp = {0: [], 1: []}
    r_som_unexp = {0: [], 1: []}
    for cue_id in (0, 1):
        for _ in range(n_trials):
            _, s_exp, _ = _run_probe_capture(
                bundle, cue_id, cue_map[cue_id], timing, noise_std, gen)
            r_som_exp[cue_id].append(s_exp.numpy())
            _, s_une, _ = _run_probe_capture(
                bundle, cue_id, cue_map[1 - cue_id], timing, noise_std, gen)
            r_som_unexp[cue_id].append(s_une.numpy())
    modulator = {}
    for c in (0, 1):
        exp_mean = np.stack(r_som_exp[c], 0).mean(0)
        unexp_mean = np.stack(r_som_unexp[c], 0).mean(0)
        modulator[c] = exp_mean - unexp_mean

    def _cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return 0.0 if (na == 0 or nb == 0) else float((a @ b) / (na * nb))

    # ================================================================
    # H1 — W saturated pseudo-random test
    # ================================================================
    # For each class c, compute W @ m_c [n_som] and compare to r_som_loc[c].
    Wm = {c: W @ m_c[c] for c in (0, 1)}
    cos_Wm_loc = {c: _cos(Wm[c], r_som_loc[c]) for c in (0, 1)}
    # Shuffle control: randomize sign of W and see how cos changes
    rng = np.random.default_rng(0)
    shuffle_cos = []
    for _ in range(100):
        W_shuf = W.copy()
        signs = rng.choice([-1.0, 1.0], size=W.shape)
        W_shuf = np.abs(W) * signs
        for c in (0, 1):
            shuffle_cos.append(_cos(W_shuf @ m_c[c], r_som_loc[c]))
    shuffle_mean = float(np.mean(shuffle_cos))
    shuffle_std = float(np.std(shuffle_cos))
    # sign entropy: fraction of positive vs negative entries
    p_pos = float((W > 0).mean())
    h_sign = -(p_pos * np.log2(p_pos + 1e-12) +
               (1 - p_pos) * np.log2(1 - p_pos + 1e-12))
    mean_cos_Wm_loc = float(np.mean(list(cos_Wm_loc.values())))
    # verdict: if observed cos ≈ shuffle mean → CONFIRMED (pseudo-random)
    gap = mean_cos_Wm_loc - shuffle_mean
    h1_verdict = "CONFIRMED" if abs(gap) < 2 * shuffle_std else "FALSIFIED"
    print(f"\n[H1] cos(W·m_c, r_som_loc_c): c0={cos_Wm_loc[0]:.4f} "
          f"c1={cos_Wm_loc[1]:.4f} mean={mean_cos_Wm_loc:.4f}", file=sys.stderr)
    print(f"[H1] shuffle-control: mean={shuffle_mean:.4f} std={shuffle_std:.4f} "
          f"gap={gap:.4f}", file=sys.stderr)
    print(f"[H1] W sign entropy={h_sign:.4f} (1.0=random) p_pos={p_pos:.3f}",
          file=sys.stderr)

    # ================================================================
    # H2 — modulator near zero
    # ================================================================
    mod_norm = {c: float(np.linalg.norm(modulator[c])) for c in (0, 1)}
    mod_max_abs = {c: float(np.abs(modulator[c]).max()) for c in (0, 1)}
    mod_sparsity = {c: float((np.abs(modulator[c]) < 0.1 * mod_max_abs[c]).mean())
                    for c in (0, 1)}
    mod_norm_mean = float(np.mean(list(mod_norm.values())))
    mod_max_mean = float(np.mean(list(mod_max_abs.values())))
    mod_sparsity_mean = float(np.mean(list(mod_sparsity.values())))
    h2_verdict = "CONFIRMED" if mod_norm_mean < 1.0 else "FALSIFIED"
    print(f"\n[H2] modulator_norm c0={mod_norm[0]:.4f} c1={mod_norm[1]:.4f} "
          f"mean={mod_norm_mean:.4f}", file=sys.stderr)
    print(f"[H2] modulator_max c0={mod_max_abs[0]:.4f} c1={mod_max_abs[1]:.4f}",
          file=sys.stderr)

    # ================================================================
    # H3 — SOM tuning mirrors L23E tuning?
    # ================================================================
    # Run full 12-orient localizer to build per-unit preferred orient histograms
    n_orients = 12
    orients = np.linspace(0, 180, n_orients, endpoint=False)
    som_tuning = np.zeros((n_orients, n_som))
    l23_tuning = np.zeros((n_orients, 256))
    n_loc_per = 8
    for oi, o in enumerate(orients):
        tm_s, tm_l = [], []
        for _ in range(n_loc_per):
            l23_mean, som_mean, _ = _run_probe_capture(
                bundle, None, float(o), timing, noise_std, gen)
            tm_s.append(som_mean.numpy())
            tm_l.append(l23_mean.numpy())
        som_tuning[oi] = np.stack(tm_s, 0).mean(0)
        l23_tuning[oi] = np.stack(tm_l, 0).mean(0)
    som_pref = orients[np.argmax(som_tuning, axis=0)]
    l23_pref = orients[np.argmax(l23_tuning, axis=0)]
    som_hist = np.zeros(n_orients)
    l23_hist = np.zeros(n_orients)
    for oi in range(n_orients):
        som_hist[oi] = (som_pref == orients[oi]).sum() / n_som
        l23_hist[oi] = (l23_pref == orients[oi]).sum() / 256
    som_n_bins_5pct = int((som_hist >= 0.05).sum())
    l23e_n_bins_5pct = int((l23_hist >= 0.05).sum())
    cos_hist = _cos(som_hist - som_hist.mean(), l23_hist - l23_hist.mean())
    h3_verdict = "CONFIRMED" if cos_hist < 0.3 else "FALSIFIED"
    print(f"\n[H3] som_n_bins_5pct={som_n_bins_5pct} "
          f"l23e_n_bins_5pct={l23e_n_bins_5pct} "
          f"cos_som_l23_histograms={cos_hist:.4f}", file=sys.stderr)
    print(f"[H3] som_hist={som_hist.tolist()}", file=sys.stderr)
    print(f"[H3] l23_hist={l23_hist.tolist()}", file=sys.stderr)

    # ================================================================
    # H4 — SNR
    # ================================================================
    r_som_baseline = float(np.mean([np.stack(r_som_exp[c], 0).mean()
                                    for c in (0, 1)]))
    mod_abs_mean = mod_max_mean
    snr = mod_abs_mean / max(r_som_baseline, 1e-12)
    h4_verdict = "CONFIRMED" if snr < 0.01 else "FALSIFIED"
    print(f"\n[H4] r_som_baseline={r_som_baseline:.4f} "
          f"mod_abs={mod_abs_mean:.4f} snr={snr:.6f}", file=sys.stderr)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\nH1: h1_verdict={h1_verdict} w_sign_entropy={h_sign:.3f} "
          f"cos_W_col_loc={mean_cos_Wm_loc:.4f} "
          f"rand_shuffle_ctrl={shuffle_mean:.4f}")
    print(f"H2: h2_verdict={h2_verdict} mod_norm={mod_norm_mean:.4f} "
          f"mod_max={mod_max_mean:.4f} mod_sparsity={mod_sparsity_mean:.3f}")
    print(f"H3: h3_verdict={h3_verdict} som_n_bins={som_n_bins_5pct} "
          f"l23e_n_bins={l23e_n_bins_5pct} cos_som_l23_tuning={cos_hist:.4f}")
    print(f"H4: h4_verdict={h4_verdict} r_som_baseline={r_som_baseline:.4f} "
          f"mod_abs={mod_abs_mean:.4f} snr={snr:.6f}")

    out = {
        "H1": {"verdict": h1_verdict, "w_sign_entropy": h_sign,
               "cos_W_col_loc": mean_cos_Wm_loc,
               "cos_W_col_loc_per_class": cos_Wm_loc,
               "rand_shuffle_ctrl_mean": shuffle_mean,
               "rand_shuffle_ctrl_std": shuffle_std,
               "W_pct_at_pos_clamp": float((W >= 7.99).mean()),
               "W_pct_near_zero": float((np.abs(W) < 0.5).mean())},
        "H2": {"verdict": h2_verdict, "mod_norm_mean": mod_norm_mean,
               "mod_max_mean": mod_max_mean,
               "mod_sparsity_mean": mod_sparsity_mean,
               "mod_norm_per_class": mod_norm,
               "mod_max_per_class": mod_max_abs},
        "H3": {"verdict": h3_verdict,
               "som_n_bins_5pct": som_n_bins_5pct,
               "l23e_n_bins_5pct": l23e_n_bins_5pct,
               "cos_som_l23_tuning": cos_hist,
               "som_hist": som_hist.tolist(),
               "l23_hist": l23_hist.tolist()},
        "H4": {"verdict": h4_verdict,
               "r_som_baseline": r_som_baseline,
               "mod_abs": mod_abs_mean,
               "snr": snr},
    }
    Path("logs/task74").mkdir(parents=True, exist_ok=True)
    (Path("logs/task74") / "direction_alignment_s42.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
