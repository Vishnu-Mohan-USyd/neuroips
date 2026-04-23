"""Task#74 Phase-2 Fix-E degradation investigation.

4 hypotheses (evidence-only):
  H1 homeostasis θ drift
  H2 W_rec_l23 destructive growth (orientation structure decay)
  H3 prediction head stuck / weights saturating
  H4 frozen SOM removed stabilizing loop (compare vs Task#70 Phase-2)

Inputs:
  - fix E post-Phase-2: checkpoints/v2/phase2/phase2_fixE_s42/phase2_s42/step_{100,500,1500,3000}.pt
  - task#70 post-Phase-2: checkpoints/v2/phase2/phase2_task70_s42/phase2_s42/step_3000.pt
  - fresh init: construct from config (no ckpt needed)
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
from scripts.v2.train_phase3_kok_learning import KokTiming
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.layers import _excitatory_eff


def _fresh_init(seed=42):
    torch.manual_seed(seed)
    cfg = ModelConfig(seed=seed, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    return net, cfg


def _localizer_tuning(bundle_or_net, cfg, n_per=5, noise_std=0.0):
    """Compute L23E tuning [n_orient, n_l23] from cue-free probes."""
    if hasattr(bundle_or_net, "net"):
        net = bundle_or_net.net
    else:
        net = bundle_or_net
    net.eval()
    timing = KokTiming()
    n_orients = 12
    orients = np.linspace(0, 180, n_orients, endpoint=False)
    n_l23 = cfg.arch.n_l23_e
    tuning = np.zeros((n_orients, n_l23))
    with torch.no_grad():
        for oi, o in enumerate(orients):
            rs = []
            blank = make_blank_frame(1, cfg, device="cpu")
            probe = make_grating_frame(float(o), 1.0, cfg, device="cpu")
            for _ in range(n_per):
                state = net.initial_state(batch_size=1)
                cue_end = timing.cue_steps
                delay_end = cue_end + timing.delay_steps
                probe1_end = delay_end + timing.probe1_steps
                r_p = []
                for t in range(timing.total):
                    if t < delay_end:
                        frame = blank
                    elif t < probe1_end:
                        frame = probe
                    else:
                        frame = blank
                    _x, state, info = net(frame, state, q_t=None)
                    if delay_end <= t < probe1_end:
                        r_p.append(info["r_l23"][0].detach().clone())
                rs.append(torch.stack(r_p, 0).mean(0).numpy())
            tuning[oi] = np.stack(rs, 0).mean(0)
    return orients, tuning


def _eval_ckpt_metrics(ckpt_path):
    """Load ckpt, extract: theta_l23 stats, W_rec_l23_raw norm, W_pred_*_raw norms,
    r_l23 at blank (approximate baseline)."""
    if ckpt_path is None:
        net, cfg = _fresh_init()
        tag = "FRESH_INIT"
    else:
        bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
        net = bundle.net
        cfg = bundle.cfg
        tag = str(ckpt_path)
    out = {"tag": tag}
    # H1 θ on L23E
    try:
        theta = net.l23_e.homeostasis.theta.detach().numpy()
        out["theta_l23_mean"] = float(theta.mean())
        out["theta_l23_min"] = float(theta.min())
        out["theta_l23_max"] = float(theta.max())
        out["theta_l23_std"] = float(theta.std())
    except Exception as e:
        out["theta_l23_err"] = str(e)
    try:
        target_l23 = getattr(net.l23_e, "target_rate_hz", None)
        out["l23_target_rate_hz"] = float(target_l23) if target_l23 is not None else None
    except Exception:
        out["l23_target_rate_hz"] = None
    # H2 W_rec_l23_raw
    try:
        w_rec = net.l23_e.W_rec_raw.detach()
        w_rec_eff = _excitatory_eff(w_rec).numpy()
        out["W_rec_l23_raw_norm"] = float(w_rec.norm().item())
        out["W_rec_l23_eff_norm"] = float(np.linalg.norm(w_rec_eff))
        out["W_rec_l23_eff_mean"] = float(w_rec_eff.mean())
        out["W_rec_l23_eff_max"] = float(w_rec_eff.max())
    except Exception as e:
        out["W_rec_l23_err"] = str(e)
    # H3 prediction head
    try:
        for name in ("W_pred_H_raw", "W_pred_C_raw", "W_pred_apical_raw"):
            if hasattr(net.prediction_head, name):
                w = getattr(net.prediction_head, name).detach()
                out[f"{name}_norm"] = float(w.norm().item())
                out[f"{name}_mean"] = float(w.mean().item())
    except Exception as e:
        out["W_pred_err"] = str(e)
    # W_l23_som
    try:
        out["W_l23_som_raw_mean"] = float(
            net.l23_som.W_l23_som_raw.detach().mean().item())
        out["W_l23_som_eff_mean"] = float(
            _excitatory_eff(net.l23_som.W_l23_som_raw.detach()).mean().item())
    except Exception:
        pass
    return net, cfg, out


def _preferred_orient_bins(tuning, orients):
    """n_preferred_bins ≥ 5% of units."""
    pref = orients[np.argmax(tuning, axis=0)]
    bin_count = np.zeros(len(orients))
    for oi, o in enumerate(orients):
        bin_count[oi] = (pref == o).sum() / tuning.shape[1]
    return int((bin_count >= 0.05).sum()), bin_count, pref


def _per_unit_cos_w_rec_pref(net, tuning, orients):
    """For each unit i, compute cos(W_rec[i,:], tuning_at_i's_preferred)."""
    w_rec_eff = _excitatory_eff(net.l23_e.W_rec_raw.detach()).numpy()
    pref_idx = np.argmax(tuning, axis=0)  # [n_l23]
    cos_vals = []
    for i in range(w_rec_eff.shape[0]):
        target = tuning[pref_idx[i]]  # [n_l23]
        w_row = w_rec_eff[i]
        na = np.linalg.norm(w_row); nb = np.linalg.norm(target)
        if na == 0 or nb == 0:
            continue
        cos_vals.append((w_row @ target) / (na * nb))
    return float(np.mean(cos_vals)), float(np.std(cos_vals))


def main():
    results = {}
    steps_to_check = [
        ("fresh_init", None),
        ("step_100",  ROOT / "checkpoints/v2/phase2/phase2_fixE_s42/phase2_s42/step_100.pt"),
        ("step_500",  ROOT / "checkpoints/v2/phase2/phase2_fixE_s42/phase2_s42/step_500.pt"),
        ("step_1500", ROOT / "checkpoints/v2/phase2/phase2_fixE_s42/phase2_s42/step_1500.pt"),
        ("step_3000", ROOT / "checkpoints/v2/phase2/phase2_fixE_s42/phase2_s42/step_3000.pt"),
    ]
    trajectory = {}
    net_final = None; cfg_final = None
    tuning_at_steps = {}
    for label, pth in steps_to_check:
        net, cfg, metrics = _eval_ckpt_metrics(pth)
        # tuning (expensive but needed for n_pref_bins and H2 cos test)
        orients, tuning = _localizer_tuning(net, cfg, n_per=3, noise_std=0.0)
        n_bins, bin_count, pref = _preferred_orient_bins(tuning, orients)
        metrics["n_preferred_bins_5pct"] = n_bins
        metrics["rate_median_at_probe"] = float(np.median(tuning))
        metrics["rate_mean_at_probe"] = float(tuning.mean())
        tuning_at_steps[label] = tuning
        # H2 per-unit cos
        cos_mean, cos_std = _per_unit_cos_w_rec_pref(net, tuning, orients)
        metrics["cos_W_rec_pref_mean"] = cos_mean
        metrics["cos_W_rec_pref_std"] = cos_std
        trajectory[label] = metrics
        print(f"[{label}] "
              f"θ_mean={metrics.get('theta_l23_mean',np.nan):.4f} "
              f"W_rec_norm={metrics.get('W_rec_l23_eff_norm',np.nan):.2f} "
              f"cos_Wrec_pref={cos_mean:.3f}±{cos_std:.3f} "
              f"n_bins={n_bins} rate_med={metrics['rate_median_at_probe']:.4f} "
              f"W_pred_H={metrics.get('W_pred_H_raw_norm',np.nan):.2f} "
              f"W_l23_som_mean={metrics.get('W_l23_som_raw_mean',np.nan):.3f}",
              file=sys.stderr)
        if label == "step_3000":
            net_final = net; cfg_final = cfg
    results["fixE_trajectory"] = trajectory

    # ========== H4: Task#70 comparison ==========
    net_t70, cfg_t70, m_t70 = _eval_ckpt_metrics(
        ROOT / "checkpoints/v2/phase2/phase2_task70_s42/phase2_s42/step_3000.pt")
    o_t70, tuning_t70 = _localizer_tuning(net_t70, cfg_t70, n_per=3, noise_std=0.0)
    n_bins_t70, _, _ = _preferred_orient_bins(tuning_t70, o_t70)
    m_t70["n_preferred_bins_5pct"] = n_bins_t70
    m_t70["rate_median_at_probe"] = float(np.median(tuning_t70))
    cos_mean_t70, _ = _per_unit_cos_w_rec_pref(net_t70, tuning_t70, o_t70)
    m_t70["cos_W_rec_pref_mean"] = cos_mean_t70
    results["task70_post_phase2"] = m_t70
    print(f"\n[TASK#70 ref] n_bins={n_bins_t70} rate_med={m_t70['rate_median_at_probe']:.4f} "
          f"θ_mean={m_t70.get('theta_l23_mean',np.nan):.4f} "
          f"cos_Wrec_pref={cos_mean_t70:.3f}",
          file=sys.stderr)

    # ========== Hypothesis verdicts ==========
    init = trajectory["fresh_init"]
    final = trajectory["step_3000"]

    # H1 — θ drift
    theta_init = init.get("theta_l23_mean", np.nan)
    theta_final = final.get("theta_l23_mean", np.nan)
    theta_drift = theta_final - theta_init
    target_rate = init.get("l23_target_rate_hz", np.nan)
    observed_final_rate = final["rate_mean_at_probe"]
    # H1 passes if |drift| small and observed rate near target_rate
    h1_verdict = "FALSIFIED" if (abs(theta_drift) < 0.1 and
                                  abs(observed_final_rate - (target_rate or 1)) < 0.5
                                  ) else "CONFIRMED"

    # H2 — W_rec cos collapse
    cos_init = init["cos_W_rec_pref_mean"]
    cos_final = final["cos_W_rec_pref_mean"]
    w_rec_norm_init = init.get("W_rec_l23_eff_norm", np.nan)
    w_rec_norm_final = final.get("W_rec_l23_eff_norm", np.nan)
    h2_verdict = "CONFIRMED" if (cos_final < cos_init - 0.05) else "FALSIFIED"

    # H3 — prediction head: compare W_pred norms init vs final; also check
    # training-curve "loss growth" (lead reported first-10 median 0.094 →
    # last-10 median 0.697 — we confirm from metrics.jsonl below).
    # Read jsonl
    metrics_path = ROOT / "checkpoints/v2/phase2/phase2_fixE_s42/phase2_s42/metrics.jsonl"
    loss_lines = [json.loads(l) for l in metrics_path.read_text().splitlines()]
    loss_vals = [d["loss_pred"] for d in loss_lines]
    loss_init_med = float(np.median(loss_vals[:10]))
    loss_final_med = float(np.median(loss_vals[-10:]))
    loss_init_mean = float(np.mean(loss_vals[:10]))
    loss_final_mean = float(np.mean(loss_vals[-10:]))
    w_pred_H_init = init.get("W_pred_H_raw_norm", np.nan)
    w_pred_H_final = final.get("W_pred_H_raw_norm", np.nan)
    w_pred_C_init = init.get("W_pred_C_raw_norm", np.nan)
    w_pred_C_final = final.get("W_pred_C_raw_norm", np.nan)
    h3_verdict = "CONFIRMED" if loss_final_mean > 1.5 * loss_init_mean else "FALSIFIED"

    # H4 — Fix E bin collapse vs Task#70 bin collapse
    fixE_bin_init = init["n_preferred_bins_5pct"]
    fixE_bin_final = final["n_preferred_bins_5pct"]
    task70_bin = n_bins_t70
    fixE_collapse = (fixE_bin_final <= 2)
    # Compare: Task#70 with plastic SOM also has partial collapse
    h4_verdict = ("FALSIFIED" if task70_bin <= 2 else
                  ("CONFIRMED" if fixE_collapse else "FALSIFIED"))

    print(f"\n=== SUMMARY NUMBERS ===")
    print(f"H1: h1_verdict={h1_verdict} theta_init={theta_init:.4f} "
          f"theta_final={theta_final:.4f} theta_drift={theta_drift:.4f} "
          f"target_rate={target_rate} observed_final_rate={observed_final_rate:.4f}")
    print(f"H2: h2_verdict={h2_verdict} cos_init={cos_init:.4f} "
          f"cos_final={cos_final:.4f} "
          f"W_rec_norm_init={w_rec_norm_init:.4f} W_rec_norm_final={w_rec_norm_final:.4f} "
          f"bin_count_init={fixE_bin_init} bin_count_final={fixE_bin_final}")
    print(f"H3: h3_verdict={h3_verdict} loss_init_med={loss_init_med:.4f} "
          f"loss_final_med={loss_final_med:.4f} "
          f"loss_init_mean={loss_init_mean:.4f} loss_final_mean={loss_final_mean:.4f} "
          f"W_pred_H_norm=({w_pred_H_init:.2f},{w_pred_H_final:.2f}) "
          f"W_pred_C_norm=({w_pred_C_init:.2f},{w_pred_C_final:.2f})")
    print(f"H4: h4_verdict={h4_verdict} fixE_bin_collapse={fixE_collapse} "
          f"task70_bin_count={task70_bin} (task70_bin_collapse={task70_bin<=2})")

    Path("logs/task74").mkdir(parents=True, exist_ok=True)
    (Path("logs/task74") / "phase2_fixE_degradation.json").write_text(
        json.dumps({
            "hypotheses": {
                "H1": {"verdict": h1_verdict, "theta_init": theta_init,
                       "theta_final": theta_final, "theta_drift": theta_drift,
                       "target_rate": target_rate,
                       "observed_final_rate": observed_final_rate},
                "H2": {"verdict": h2_verdict, "cos_init": cos_init,
                       "cos_final": cos_final,
                       "W_rec_norm_init": w_rec_norm_init,
                       "W_rec_norm_final": w_rec_norm_final,
                       "bin_count_init": fixE_bin_init,
                       "bin_count_final": fixE_bin_final},
                "H3": {"verdict": h3_verdict, "loss_init_med": loss_init_med,
                       "loss_final_med": loss_final_med,
                       "loss_init_mean": loss_init_mean,
                       "loss_final_mean": loss_final_mean,
                       "W_pred_H_init": w_pred_H_init,
                       "W_pred_H_final": w_pred_H_final,
                       "W_pred_C_init": w_pred_C_init,
                       "W_pred_C_final": w_pred_C_final},
                "H4": {"verdict": h4_verdict,
                       "fixE_bin_collapse": fixE_collapse,
                       "task70_bin_count": task70_bin,
                       "task70_bin_collapse": bool(task70_bin <= 2)},
            },
            "fixE_trajectory": trajectory,
            "task70_ref": m_t70,
        }, indent=2))


if __name__ == "__main__":
    main()
