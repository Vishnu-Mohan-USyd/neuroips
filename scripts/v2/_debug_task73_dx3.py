"""Task #73 Dx3 — project ΔW_mh_task·m onto the localizer tuning basis.

W_mh_task is zero at init, so ΔW_mh_task ≡ W_mh_task(trained).
For each cue c, compute b_task_mean = mean_trials(W_mh_task @ m_{cue=c}) at
delay_end-1, then project onto the 12-orientation localizer basis by
inner product with tuning[i] / ||tuning[i]||. The orientation with the
largest projection coefficient is the "template direction" the rule has
learned. Template replay hypothesis predicts template_direction(cue)
= expected_orientation(cue).
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
    KokTiming, cue_mapping_from_seed, build_cue_tensor,
)


def get_m_end_per_cue(bundle, cue_mapping, timing, n_trials=30):
    cfg = bundle.cfg
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    blank = make_blank_frame(1, cfg, device="cpu")
    m_by_cue = {0: [], 1: []}
    for cue_id in (0, 1):
        q_cue = build_cue_tensor(cue_id, cfg.arch.n_c, device="cpu")
        for _ in range(n_trials):
            state = bundle.net.initial_state(batch_size=1)
            for t in range(delay_end):
                q_t = q_cue if t < cue_end else None
                _x, state, info = bundle.net(blank, state, q_t=q_t)
                bundle.net.l23_e.homeostasis.update(state.r_l23)
                bundle.net.h_e.homeostasis.update(state.r_h)
                if t == delay_end - 1:
                    m_by_cue[cue_id].append(
                        state.m.detach().clone().reshape(-1).numpy())
                    break
    return {k: np.stack(v, axis=0) for k, v in m_by_cue.items()}


def localizer_tuning(bundle, timing, n_orients=12, n_per=5):
    cfg = bundle.cfg
    blank = make_blank_frame(1, cfg, device="cpu")
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    orients = np.linspace(0.0, 180.0, n_orients, endpoint=False)
    out = np.zeros((n_orients, bundle.net.cfg.arch.n_l23_e), dtype=np.float64)
    for i, o in enumerate(orients):
        probe = make_grating_frame(float(o), 1.0, cfg, device="cpu")
        accum = []
        for _ in range(n_per):
            state = bundle.net.initial_state(batch_size=1)
            probe_rs = []
            for t in range(probe1_end):
                frame = blank if t < delay_end else probe
                _x, state, info = bundle.net(frame, state, q_t=None)
                bundle.net.l23_e.homeostasis.update(state.r_l23)
                bundle.net.h_e.homeostasis.update(state.r_h)
                if delay_end <= t < probe1_end:
                    probe_rs.append(info["r_l23"][0].clone())
            accum.append(torch.stack(probe_rs, dim=0).mean(dim=0))
        out[i] = torch.stack(accum, dim=0).mean(dim=0).numpy()
    return orients, out


def main(ckpt_path: Path):
    bundle = load_checkpoint(ckpt_path, seed=42, device="cpu")
    bundle.net.set_phase("phase3_kok")
    cue_mapping = bundle.meta.get("cue_mapping", cue_mapping_from_seed(42))
    cue_mapping = {int(k): float(v) for k, v in cue_mapping.items()}
    cm = bundle.net.context_memory
    W_mh_task = cm.W_mh_task.data.numpy()  # [n_out=n_l23, n_m]
    timing = KokTiming()

    m_by_cue = get_m_end_per_cue(bundle, cue_mapping, timing, n_trials=20)
    m_c0_mean = m_by_cue[0].mean(axis=0)
    m_c1_mean = m_by_cue[1].mean(axis=0)

    # b_task_mean per cue
    b_c0 = W_mh_task @ m_c0_mean  # [n_l23]
    b_c1 = W_mh_task @ m_c1_mean

    orients, tuning = localizer_tuning(bundle, timing, n_orients=12, n_per=3)

    # Project b_task onto each orientation basis vector
    def project(b, basis_i):
        v = basis_i
        num = float(np.dot(b, v))
        den = float(np.dot(v, v))
        return num / den if den > 0 else 0.0

    proj_c0 = {float(orients[i]): project(b_c0, tuning[i]) for i in range(len(orients))}
    proj_c1 = {float(orients[i]): project(b_c1, tuning[i]) for i in range(len(orients))}

    arg_c0 = max(proj_c0, key=proj_c0.get)
    arg_c1 = max(proj_c1, key=proj_c1.get)
    exp_c0 = cue_mapping[0]; exp_c1 = cue_mapping[1]

    # Template-replay fraction: what fraction of b_task's norm projects onto the
    # expected-orientation basis direction (vs other 11)?
    def energy_fraction(b, basis_i, all_basis):
        v = basis_i
        proj_on_i = (np.dot(b, v) / np.linalg.norm(v)) ** 2
        total_proj = sum(
            (np.dot(b, all_basis[j]) / (np.linalg.norm(all_basis[j]) + 1e-12)) ** 2
            for j in range(len(all_basis))
        )
        return float(proj_on_i / total_proj) if total_proj > 0 else 0.0

    idx_exp_c0 = int(np.argmin(np.abs(orients - exp_c0)))
    idx_exp_c1 = int(np.argmin(np.abs(orients - exp_c1)))
    frac_c0 = energy_fraction(b_c0, tuning[idx_exp_c0], tuning)
    frac_c1 = energy_fraction(b_c1, tuning[idx_exp_c1], tuning)

    out = {
        "cue_mapping": cue_mapping,
        "projection_coefficients_cue0": proj_c0,
        "projection_coefficients_cue1": proj_c1,
        "argmax_orient_deg_cue0": arg_c0,
        "argmax_orient_deg_cue1": arg_c1,
        "expected_orient_cue0": exp_c0,
        "expected_orient_cue1": exp_c1,
        "template_match_cue0": bool(abs(arg_c0 - exp_c0) % 180 <= 15.0 or
                                    abs(arg_c0 - exp_c0) % 180 >= 165.0),
        "template_match_cue1": bool(abs(arg_c1 - exp_c1) % 180 <= 15.0 or
                                    abs(arg_c1 - exp_c1) % 180 >= 165.0),
        "energy_fraction_at_expected_cue0": frac_c0,
        "energy_fraction_at_expected_cue1": frac_c1,
        "b_task_norm_cue0": float(np.linalg.norm(b_c0)),
        "b_task_norm_cue1": float(np.linalg.norm(b_c1)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/v2/phase3_kok_task70/phase3_kok_s42.pt"))
    args = p.parse_args()
    main(args.ckpt)
