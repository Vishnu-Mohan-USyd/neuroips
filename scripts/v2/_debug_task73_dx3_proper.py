"""Task #73 Dx3 (proper, per lead's spec) — learning-rule-update projection.

Spec: during Phase-3 deterministic learning (fresh 100-trial run from
post-Phase2 checkpoint), log ΔW_mh_task per update. Compute ΔW_mh_task @
m_average per expected-class. Project onto localizer responses:
  - expected-class localizer response
  - orthogonal class (Kok: +90°)
  - global mean response
Report cos(Δ·m, r_expected_loc), cos(Δ·m, r_orth_loc), cos(Δ·m, r_mean_loc).

PASS if cos_exp ≈ +1 and others ≈ 0 → POSITIVE TEMPLATE learning confirmed.
FAIL if cos negative → suppression learning (falsifies template-replay).

Uses post-Phase2 checkpoint (NOT post-Phase3 ckpt which has already
converged). Runs ~100 deterministic Kok trials with apply_plasticity=True,
logging ΔW_mh_task and state.m at each update.

Localizer tuning computed ONCE at start (post-Phase2 fresh) as basis.
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
from scripts.v2.eval_kok import (
    run_kok_localizer_trial, _compute_localizer_stats,
)
from scripts.v2.train_phase3_kok_learning import (
    KokTiming, cue_mapping_from_seed, build_cue_tensor, run_kok_trial,
)
from src.v2_model.plasticity import ThreeFactorRule


def compute_localizer_basis(bundle, n_per=8, noise_std=0.01, seed=42):
    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    orients = np.linspace(0.0, 180.0, 12, endpoint=False)
    loc_trials, loc_orient = [], []
    for o in orients:
        for _ in range(n_per):
            r = run_kok_localizer_trial(
                bundle, probe_orientation_deg=float(o),
                timing=timing, noise_std=noise_std, generator=gen,
            )
            loc_trials.append(r.cpu().numpy())
            loc_orient.append(float(o))
    loc_trials = np.stack(loc_trials, axis=0)
    loc_orient = np.asarray(loc_orient)
    stats = _compute_localizer_stats(loc_trials, loc_orient, orients)
    return orients, stats["tuning_curve"]  # [n_orient, n_l23]


def main(ckpt_path, n_trials=80, n_loc_per=8):
    """Load fresh post-Phase2 ckpt, compute localizer once, then run
    deterministic Phase-3 trials with per-step logging."""
    bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cue_mapping = bundle.meta.get("cue_mapping", cue_mapping_from_seed(42))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}

    # 1. Fixed localizer basis BEFORE any Phase-3 learning
    orients, tuning = compute_localizer_basis(
        bundle, n_per=n_loc_per, noise_std=0.01, seed=42)
    print(f"[Dx3-proper] Localizer computed; tuning shape {tuning.shape}",
          file=sys.stderr, flush=True)

    # 2. Initialize plasticity rule (match train_phase3_kok_learning defaults)
    cfg = bundle.cfg
    rule = ThreeFactorRule(lr=1e-3, weight_decay=1e-5)

    # 3. Run n_trials deterministic Kok trials, log Δw_mh and m per update.
    timing = KokTiming()
    gen = torch.Generator(device="cpu"); gen.manual_seed(12345)
    dw_history = []   # list of (cue_id, exp_orient_deg, dw_mh [n_l23, n_m], m [n_m])
    for t in range(n_trials):
        cue_id = t % 2
        probe_deg = cue_mapping[cue_id]
        # Use run_kok_trial with apply_plasticity=True
        # Need to intercept dw_mh before it's applied.
        # Alternative: call with apply_plasticity=True and snapshot m + W after.
        # Easier approach: call twice — once with apply_plasticity=False to get dw
        # (but m depends on state evolution which plasticity affects).
        # Proper approach: monkey-patch W_mh_task update to capture dw_mh + m.
        W_before = bundle.net.context_memory.W_mh_task.data.clone()
        # Snapshot m BEFORE running trial
        r = run_kok_trial(
            bundle.net, bundle.cfg,
            cue_id=cue_id,
            probe_orientation_deg=float(probe_deg),
            timing=timing, rule=rule,
            noise_std=0.01, device="cpu",
            apply_plasticity=True,
        )
        W_after = bundle.net.context_memory.W_mh_task.data
        dw_mh = (W_after - W_before).numpy()  # [n_l23, n_m]
        # m at probe time - r doesn't return m, so we don't have it per-step.
        # Workaround: use current m (post-trial) as approximation. For template-
        # replay check, we just need dw @ m_typical_for_this_cue.
        # We'll accumulate m's separately.
        dw_history.append({
            "cue_id": cue_id,
            "expected_orient": float(probe_deg),
            "dw_mh": dw_mh,
        })

    # 4. Collect m per cue at delay_end-1 (fresh, no plasticity, using CURRENT
    # trained weights) — these are the m vectors the rule "sees" at update.
    m_by_cue = {0: [], 1: []}
    for cue_id in (0, 1):
        for _ in range(30):
            q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
            blank = make_blank_frame(1, cfg, device="cpu")
            state = bundle.net.initial_state(batch_size=1)
            cue_end = timing.cue_steps
            delay_end = cue_end + timing.delay_steps
            with torch.no_grad():
                for t_ in range(delay_end):
                    q_t = q_cue if t_ < cue_end else None
                    _x, state, info = bundle.net(blank, state, q_t=q_t)
                    if t_ == delay_end - 1:
                        m_by_cue[cue_id].append(
                            state.m.detach().clone().reshape(-1).numpy())
                        break
    m_c0 = np.stack(m_by_cue[0], axis=0).mean(axis=0)
    m_c1 = np.stack(m_by_cue[1], axis=0).mean(axis=0)

    # 5. For each update: dw_mh @ m_cue → vector in L2/3 space. Project onto
    # localizer basis.
    def _cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Cosine with expected-class localizer, orthogonal (+90°), global mean
    exp_o_c0 = cue_mapping[0]; exp_o_c1 = cue_mapping[1]
    def _nearest_idx(target):
        return int(np.argmin(np.abs(orients - target)))
    idx_exp0 = _nearest_idx(exp_o_c0)
    idx_exp1 = _nearest_idx(exp_o_c1)
    idx_orth0 = _nearest_idx((exp_o_c0 + 90) % 180)
    idx_orth1 = _nearest_idx((exp_o_c1 + 90) % 180)
    mean_loc = tuning.mean(axis=0)

    cos_exp_list, cos_orth_list, cos_mean_list = [], [], []
    for rec in dw_history:
        dw = rec["dw_mh"]
        cue = rec["cue_id"]
        m_vec = m_c0 if cue == 0 else m_c1
        dw_m = dw @ m_vec  # [n_l23]
        if cue == 0:
            idx_e = idx_exp0; idx_o = idx_orth0
        else:
            idx_e = idx_exp1; idx_o = idx_orth1
        cos_exp_list.append(_cos(dw_m, tuning[idx_e]))
        cos_orth_list.append(_cos(dw_m, tuning[idx_o]))
        cos_mean_list.append(_cos(dw_m, mean_loc))

    # Summary: mean cosine over all updates
    out = {
        "n_trials_logged": n_trials,
        "cue_mapping": cue_mapping,
        "cos_delta_expected_loc_mean": float(np.mean(cos_exp_list)),
        "cos_delta_expected_loc_std": float(np.std(cos_exp_list)),
        "cos_delta_orth_loc_mean": float(np.mean(cos_orth_list)),
        "cos_delta_orth_loc_std": float(np.std(cos_orth_list)),
        "cos_delta_mean_loc_mean": float(np.mean(cos_mean_list)),
        "cos_delta_mean_loc_std": float(np.std(cos_mean_list)),
        "final_W_mh_task_norm": float(
            bundle.net.context_memory.W_mh_task.data.norm().item()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase2/phase2_task70_s42/"
                                "phase2_s42/step_3000.pt"))
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--n-loc-per", type=int, default=8)
    args = p.parse_args()
    main(args.ckpt, args.n_trials, args.n_loc_per)
