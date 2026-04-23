"""Task #73 — Diagnostic 3: Learning-rule projection on Phase-2 checkpoint.

Purpose
-------
The Phase-3 plasticity rule is:
    dw_mh = lr * outer(probe_error, m_start_probe) - wd * W_mh_task
    probe_error = r_l23_probe_mean - b_l23_pre_probe

If ``W_mh_task == 0`` (Phase-2 initial), then ``b_l23_pre_probe == b_gen``
(the generic readout alone), and:
    probe_error ≈ r_l23_probe_mean - b_gen

So the rule pushes W_mh_task toward a Rescorla-Wagner fixed point where
``W_mh_task @ m_cue ≈ r_l23_probe_mean - b_gen``.

This diagnostic directly measures the *direction* of accumulated dw_mh per
expected class c and projects it through the mean m_start_probe for that
class to produce:
    bias_added_c = avg_dw_c @ m_avg_c       # [n_l23]

Then compares this against:
  (a) localizer template for class c — should be highly positive if
      POSITIVE_TEMPLATE_REPLAY is true
  (b) localizer template for OTHER class — falsification control
  (c) grand mean of localizer templates — if ≈ +1, learning is GLOBAL_GAIN

Approach
--------
* Load Phase-2 step_3000.pt.
* Build ``ThreeFactorRule(lr=1e-3, weight_decay=1e-5)`` (matches training).
* Run ``apply_plasticity=False`` for 200 balanced Kok trials (and 180 Richter
  trials) so every trial independently measures the rule's gradient from the
  same untrained starting state.
* Log per trial: cue_id, probe_orient, m_start_probe, r_l23_probe_mean,
  b_l23_pre_probe, probe_error, dw_mh (pre-clamp).
* For each expected class c:
    avg_dw_c = mean(dw_mh) over trials with expected==c
    m_avg_c  = mean(m_start_probe) for those trials
    bias_added_c = avg_dw_c @ m_avg_c
    localizer[c] from ``run_kok_localizer_trial`` (cue-free) / ``run_richter_localizer_trial`` (leader-free)
* Compute across-unit Pearson vs same/other/grand_mean.

Outputs
-------
JSON at ``checkpoints/v2/_diag3_{paradigm}.json`` with per-class metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from src.v2_model.plasticity import ThreeFactorRule
from scripts.v2._gates_common import (
    load_checkpoint, make_blank_frame, make_grating_frame,
)
from scripts.v2.train_phase3_kok_learning import (
    CUE_ORIENTATIONS_DEG, KokTiming, build_cue_tensor, cue_mapping_from_seed,
)
from scripts.v2.train_phase3_richter_learning import (
    LEADER_TOKEN_IDX, TRAILER_TOKEN_IDX, RichterTiming,
    build_leader_tensor, permutation_from_seed,
)


def _pearson(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(np.float64).ravel()
    v = v.astype(np.float64).ravel()
    if u.std() == 0 or v.std() == 0:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


# ---------------------------------------------------------------------------
# Kok single trial — manual replication of ``_run_single_trial_kok``
# ---------------------------------------------------------------------------


@torch.no_grad()
def _kok_trial_and_dw(
    bundle, rule: ThreeFactorRule,
    cue_id: int, probe_orient_deg: float, timing: KokTiming,
) -> dict:
    cfg = bundle.cfg
    device = cfg.device
    net = bundle.net
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orient_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    m_start_probe: Optional[Tensor] = None
    b_l23_pre_probe: Optional[Tensor] = None
    probe1_l23: list[Tensor] = []

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
        _x, state, info = net(frame, state, q_t=q_t)
        if t == delay_end - 1:
            m_start_probe = state.m.clone()                             # [1,n_m]
        if t == delay_end:
            b_l23_pre_probe = info["b_l23"].clone()                     # [1,n_l23]
        if delay_end <= t < probe1_end:
            probe1_l23.append(info["r_l23"].clone())
    assert m_start_probe is not None and b_l23_pre_probe is not None

    r_l23_probe_mean = torch.stack(probe1_l23, dim=0).mean(dim=0)       # [1,n_l23]
    probe_error = r_l23_probe_mean - b_l23_pre_probe                    # [1,n_l23]
    # Replicate delta_mh WITHOUT clamp — we want the rule-direction signal.
    # ΔW = lr * outer(probe_error, m_start_probe)  - wd * W
    hebb = torch.einsum("bi,bj->ij", probe_error, m_start_probe) / max(
        1, probe_error.shape[0]
    )
    W = net.context_memory.W_mh_task
    dw_mh_pre = rule.lr * hebb - rule.weight_decay * W                  # [n_out,n_m]
    return {
        "m_start_probe": m_start_probe[0].detach(),
        "r_l23_probe": r_l23_probe_mean[0].detach(),
        "b_l23_pre_probe": b_l23_pre_probe[0].detach(),
        "probe_error": probe_error[0].detach(),
        "dw_mh_preclamp": dw_mh_pre.detach(),
    }


# ---------------------------------------------------------------------------
# Richter single trial — manual replication
# ---------------------------------------------------------------------------


@torch.no_grad()
def _richter_trial_and_dw(
    bundle, rule: ThreeFactorRule,
    leader_pos: int, trailer_pos: int, timing: RichterTiming,
) -> dict:
    cfg = bundle.cfg
    device = cfg.device
    net = bundle.net
    leader_tok = bundle.bank.tokens[
        LEADER_TOKEN_IDX[int(leader_pos)]:LEADER_TOKEN_IDX[int(leader_pos)] + 1
    ].to(device=device)
    trailer_tok = bundle.bank.tokens[
        TRAILER_TOKEN_IDX[int(trailer_pos)]:TRAILER_TOKEN_IDX[int(trailer_pos)] + 1
    ].to(device=device)
    leader_v = build_leader_tensor(
        int(leader_pos), net.context_memory.n_leader, device=device,
    )
    state = net.initial_state(batch_size=1)
    leader_end = timing.leader_steps
    trailer_end = leader_end + timing.trailer_steps
    m_start_trailer: Optional[Tensor] = None
    b_l23_pre_trailer: Optional[Tensor] = None
    trailer_l23: list[Tensor] = []
    for t in range(trailer_end):
        if t < leader_end:
            frame, ld_t = leader_tok, leader_v
        else:
            frame, ld_t = trailer_tok, None
        _x, state, info = net(frame, state, leader_t=ld_t)
        if t == leader_end - 1:
            m_start_trailer = state.m.clone()
        if t == leader_end:
            b_l23_pre_trailer = info["b_l23"].clone()
        if leader_end <= t < trailer_end:
            trailer_l23.append(info["r_l23"].clone())
    assert m_start_trailer is not None and b_l23_pre_trailer is not None
    r_l23_probe_mean = torch.stack(trailer_l23, dim=0).mean(dim=0)
    probe_error = r_l23_probe_mean - b_l23_pre_trailer
    hebb = torch.einsum("bi,bj->ij", probe_error, m_start_trailer) / max(
        1, probe_error.shape[0]
    )
    W = net.context_memory.W_mh_task
    dw_mh_pre = rule.lr * hebb - rule.weight_decay * W
    return {
        "m_start_probe": m_start_trailer[0].detach(),
        "r_l23_probe": r_l23_probe_mean[0].detach(),
        "b_l23_pre_probe": b_l23_pre_trailer[0].detach(),
        "probe_error": probe_error[0].detach(),
        "dw_mh_preclamp": dw_mh_pre.detach(),
    }


# ---------------------------------------------------------------------------
# Localizer templates (cue-free / leader-free)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _kok_localizer(bundle, orient_deg: float, timing: KokTiming,
                   n_trials: int, generator: torch.Generator) -> Tensor:
    from scripts.v2.eval_kok import run_kok_localizer_trial
    return torch.stack([
        run_kok_localizer_trial(
            bundle, probe_orientation_deg=float(orient_deg),
            timing=timing, noise_std=0.0, generator=generator,
        ) for _ in range(n_trials)
    ], dim=0).mean(dim=0)


@torch.no_grad()
def _richter_localizer(bundle, trailer_pos: int, timing: RichterTiming,
                       n_trials: int, generator: torch.Generator) -> Tensor:
    from scripts.v2.eval_richter import run_richter_localizer_trial
    return torch.stack([
        run_richter_localizer_trial(
            bundle, trailer_pos=int(trailer_pos),
            timing=timing, noise_std=0.0, generator=generator,
        ) for _ in range(n_trials)
    ], dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Paradigm drivers
# ---------------------------------------------------------------------------


def diag3_kok(ckpt: Path, out_path: Path, *, seed: int,
              n_trials_per_cond: int, n_localizer: int,
              lr: float, wd: float) -> dict:
    bundle = load_checkpoint(ckpt, seed=seed, device="cpu")
    bundle.net.set_phase("phase3_kok")   # expose W_mh_task / W_qm_task plastic
    rule = ThreeFactorRule(lr=lr, weight_decay=wd)
    timing = KokTiming()
    cue_map = cue_mapping_from_seed(seed)
    probe_orients = list(CUE_ORIENTATIONS_DEG)
    ncls = 2
    gen = torch.Generator(device="cpu").manual_seed(seed)

    records: list[dict] = []
    total = 2 * len(probe_orients) * n_trials_per_cond
    t_start = time.time()
    done = 0
    for cue_id in [0, 1]:
        for p_o in probe_orients:
            for _ in range(n_trials_per_cond):
                rec = _kok_trial_and_dw(
                    bundle, rule, cue_id=cue_id,
                    probe_orient_deg=float(p_o), timing=timing,
                )
                rec["cue_id"] = int(cue_id)
                rec["probe_orient"] = float(p_o)
                rec["expected_class"] = int(
                    probe_orients.index(float(cue_map[cue_id]))
                )
                records.append(rec)
                done += 1
                if done % 10 == 0:
                    el = time.time() - t_start
                    eta = el / done * (total - done)
                    print(f"[diag3 kok] {done}/{total}  elapsed={el:.1f}s  eta={eta:.1f}s",
                          flush=True)

    loc_templates: dict[int, Tensor] = {}
    for c, o in enumerate(probe_orients):
        loc_templates[c] = _kok_localizer(
            bundle, orient_deg=float(o), timing=timing,
            n_trials=n_localizer, generator=gen,
        )
        print(f"[diag3 kok localizer] class {c} done", flush=True)

    return _summarise_dw(records, loc_templates, ncls, out_path)


def diag3_richter(ckpt: Path, out_path: Path, *, seed: int,
                  n_trials_per_cond: int, n_localizer: int,
                  lr: float, wd: float) -> dict:
    bundle = load_checkpoint(ckpt, seed=seed, device="cpu")
    bundle.net.set_phase("phase3_richter")
    rule = ThreeFactorRule(lr=lr, weight_decay=wd)
    timing = RichterTiming()
    perm = permutation_from_seed(seed)
    ncls = 6
    gen = torch.Generator(device="cpu").manual_seed(seed)

    records: list[dict] = []
    total = 36 * n_trials_per_cond
    t_start = time.time()
    done = 0
    for l_pos in range(6):
        for t_pos in range(6):
            for _ in range(n_trials_per_cond):
                rec = _richter_trial_and_dw(
                    bundle, rule, leader_pos=l_pos, trailer_pos=t_pos,
                    timing=timing,
                )
                rec["leader_pos"] = int(l_pos)
                rec["trailer_pos"] = int(t_pos)
                rec["expected_class"] = int(perm[l_pos])
                records.append(rec)
                done += 1
                if done % 20 == 0:
                    el = time.time() - t_start
                    eta = el / done * (total - done)
                    print(f"[diag3 richter] {done}/{total}  elapsed={el:.1f}s  eta={eta:.1f}s",
                          flush=True)

    loc_templates: dict[int, Tensor] = {}
    for c in range(ncls):
        loc_templates[c] = _richter_localizer(
            bundle, trailer_pos=int(c), timing=timing,
            n_trials=n_localizer, generator=gen,
        )
        print(f"[diag3 richter localizer] class {c} done", flush=True)

    return _summarise_dw(records, loc_templates, ncls, out_path)


# ---------------------------------------------------------------------------
# Summary (projection + localizer correlation)
# ---------------------------------------------------------------------------


def _summarise_dw(records, loc_templates, ncls, out_path):
    y_exp = np.array([r["expected_class"] for r in records], dtype=int)

    # Per-expected-class aggregates
    cos_same, cos_other, cos_mean_all = [], [], []
    per_class = {}
    # Grand mean localizer (across classes)
    grand_mean_loc = torch.stack(
        [loc_templates[c] for c in range(ncls)], dim=0
    ).mean(dim=0).numpy()

    # Control: size of probe_error
    probe_err_stack = torch.stack(
        [r["probe_error"] for r in records], dim=0
    ).numpy()
    probe_err_norm_mean = float(np.linalg.norm(probe_err_stack, axis=1).mean())

    for c in range(ncls):
        sel = np.where(y_exp == c)[0]
        if len(sel) == 0:
            continue
        dws = torch.stack(
            [records[i]["dw_mh_preclamp"] for i in sel], dim=0
        )                                                        # [N,n_out,n_m]
        ms = torch.stack(
            [records[i]["m_start_probe"] for i in sel], dim=0
        )                                                        # [N,n_m]
        avg_dw = dws.mean(dim=0)                                 # [n_out,n_m]
        m_avg = ms.mean(dim=0)                                   # [n_m]
        bias_added = (avg_dw @ m_avg).numpy()                    # [n_out]

        loc_c = loc_templates[c].numpy()
        others = [cc for cc in range(ncls) if cc != c]
        cos_s = _pearson(bias_added, loc_c)
        cos_o = float(np.mean([
            _pearson(bias_added, loc_templates[cc].numpy()) for cc in others
        ]))
        cos_g = _pearson(bias_added, grand_mean_loc)
        cos_same.append(cos_s)
        cos_other.append(cos_o)
        cos_mean_all.append(cos_g)
        per_class[str(c)] = {
            "n_trials": int(len(sel)),
            "bias_added_norm": float(np.linalg.norm(bias_added)),
            "localizer_norm": float(np.linalg.norm(loc_c)),
            "pearson_same_class": cos_s,
            "pearson_other_class_mean": cos_o,
            "pearson_grand_mean": cos_g,
        }

    summary = {
        "paradigm": "—",
        "n_classes": int(ncls),
        "n_trials_total": int(len(records)),
        "probe_error_norm_mean": probe_err_norm_mean,
        "per_class": per_class,
        "aggregate": {
            "mean_pearson_same_class": float(np.mean(cos_same)) if cos_same else float("nan"),
            "mean_pearson_other_class": float(np.mean(cos_other)) if cos_other else float("nan"),
            "mean_pearson_grand_mean": float(np.mean(cos_mean_all)) if cos_mean_all else float("nan"),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm", choices=["kok", "richter"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Phase-2 checkpoint (step_3000.pt)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials-per-cond", type=int, default=50)
    p.add_argument("--n-localizer", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    args = p.parse_args()
    fn = diag3_kok if args.paradigm == "kok" else diag3_richter
    summary = fn(
        args.checkpoint, args.output,
        seed=args.seed, n_trials_per_cond=args.n_trials_per_cond,
        n_localizer=args.n_localizer,
        lr=args.lr, wd=args.weight_decay,
    )
    summary["paradigm"] = args.paradigm
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
