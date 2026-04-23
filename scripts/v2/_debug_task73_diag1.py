"""Task #73 — Diagnostic 1: Expectation-signal audit on trained Phase-3 ckpts.

Purpose
-------
Measure, at the decision point (end-of-delay for Kok, end-of-leader for
Richter), what signal the *trained* task readout (``W_mh_task``) produces, and
compare it against a *causal control* signal (``W_mh_gen`` — untrained in
Phase 3) and against localizer templates (pure sensory response per class).

Decision matrix (mutually exclusive outcomes)
---------------------------------------------
  cos(b_task_c, localizer[c])  cos(b_task_c, localizer[¬c])  cos(b_task_c, mean)  Verdict
    ≈ +1                          ≈ 0                            ≈ 0                 POSITIVE_TEMPLATE_REPLAY
    ≈ −1                          ≈ 0                            ≈ 0                 TARGETED_SUPPRESSION
    ≈ 0                           ≈ 0                            ≈ +1                GLOBAL_GAIN
    ≈ 0 and ||b_task||/||b_gen|| ≪ 1                                                 READOUT_DID_NOT_LEARN

Causal controls
---------------
* ``b_gen`` (generic readout, frozen in Phase 3) ⇒ should NOT correlate with
  localizer templates. If it does, the paradigm itself is confounded.
* ``cos(b_task_c, localizer[¬c])`` ⇒ if |corr| comparable to same-class corr,
  template specificity is absent.

Outputs
-------
JSON at ``checkpoints/v2/phase3_{paradigm}_task70/_diag1.json`` with scalars
and per-class tensors (lists).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pearson_across_units(u: np.ndarray, v: np.ndarray) -> float:
    """Across-units Pearson (shape [N])."""
    u = u.astype(np.float64).ravel()
    v = v.astype(np.float64).ravel()
    if u.std() == 0 or v.std() == 0:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


def _cv_linsvc_acc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """5-fold stratified LinearSVC accuracy (mean across folds)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    unique = np.unique(y)
    if len(unique) < 2:
        return float("nan")
    n_per_class = min(int(np.sum(y == u)) for u in unique)
    n_splits = min(5, n_per_class)
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(), LinearSVC(C=1.0, max_iter=5000, dual="auto"),
        )
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# Kok probing
# ---------------------------------------------------------------------------


@torch.no_grad()
def _kok_probe_and_capture(
    bundle, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, *, generator: torch.Generator,
    noise_std: float = 0.0,
) -> dict[str, Tensor]:
    """Single Kok probe trial. Returns at decision point: m, b_gen, b_task.
    Also returns probe-epoch mean r_l23.
    """
    cfg = bundle.cfg
    device = cfg.device
    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = bundle.net.initial_state(batch_size=1)
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps
    n_total = timing.total

    m_decision: Tensor | None = None
    probe_rates: list[Tensor] = []

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
                frame.shape, generator=generator, device=device,
            )
        _x_hat, state, info = bundle.net(frame, state, q_t=q_t)
        if t == delay_end - 1:
            m_decision = state.m.clone()                                # [1, n_m]
        if delay_end <= t < probe1_end:
            probe_rates.append(info["r_l23"][0].clone())                # [n_l23]

    assert m_decision is not None
    cm = bundle.net.context_memory
    b_gen = (m_decision @ cm.W_mh_gen.t())[0].clone()                   # [n_out]
    b_task = (m_decision @ cm.W_mh_task.t())[0].clone()                 # [n_out]
    r_l23_probe_mean = torch.stack(probe_rates, dim=0).mean(dim=0)      # [n_l23]
    return {
        "m": m_decision[0].detach(),
        "b_gen": b_gen.detach(),
        "b_task": b_task.detach(),
        "b_total": (b_gen + b_task).detach(),
        "r_l23_probe": r_l23_probe_mean.detach(),
    }


@torch.no_grad()
def _kok_localizer_mean(
    bundle, orient_deg: float, timing: KokTiming, *,
    n_trials: int, generator: torch.Generator, noise_std: float,
) -> Tensor:
    """Mean probe-epoch r_l23 across n_trials of a cue-FREE trial."""
    rates = []
    from scripts.v2.eval_kok import run_kok_localizer_trial
    for _ in range(n_trials):
        r = run_kok_localizer_trial(
            bundle, probe_orientation_deg=float(orient_deg),
            timing=timing, noise_std=noise_std, generator=generator,
        )
        rates.append(r)
    return torch.stack(rates, dim=0).mean(dim=0)                        # [n_l23]


# ---------------------------------------------------------------------------
# Richter probing
# ---------------------------------------------------------------------------


@torch.no_grad()
def _richter_probe_and_capture(
    bundle, leader_pos: int, trailer_pos: int,
    timing: RichterTiming, *, generator: torch.Generator,
    noise_std: float,
) -> dict[str, Tensor]:
    cfg = bundle.cfg
    device = cfg.device
    leader_tok = bundle.bank.tokens[
        LEADER_TOKEN_IDX[int(leader_pos)]:LEADER_TOKEN_IDX[int(leader_pos)] + 1
    ].to(device=device)
    trailer_tok = bundle.bank.tokens[
        TRAILER_TOKEN_IDX[int(trailer_pos)]:TRAILER_TOKEN_IDX[int(trailer_pos)] + 1
    ].to(device=device)
    leader_v = build_leader_tensor(
        int(leader_pos), bundle.net.context_memory.n_leader, device=device,
    )
    state = bundle.net.initial_state(batch_size=1)
    leader_end = timing.leader_steps
    trailer_end = leader_end + timing.trailer_steps
    m_decision: Tensor | None = None
    trailer_rates: list[Tensor] = []
    for t in range(trailer_end):
        if t < leader_end:
            frame, ld_t = leader_tok, leader_v
        else:
            frame, ld_t = trailer_tok, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn(
                frame.shape, generator=generator, device=device,
            )
        _x, state, info = bundle.net(frame, state, leader_t=ld_t)
        if t == leader_end - 1:
            m_decision = state.m.clone()
        if leader_end <= t < trailer_end:
            trailer_rates.append(info["r_l23"][0].clone())
    assert m_decision is not None
    cm = bundle.net.context_memory
    b_gen = (m_decision @ cm.W_mh_gen.t())[0].clone()
    b_task = (m_decision @ cm.W_mh_task.t())[0].clone()
    r_l23_probe_mean = torch.stack(trailer_rates, dim=0).mean(dim=0)
    return {
        "m": m_decision[0].detach(),
        "b_gen": b_gen.detach(),
        "b_task": b_task.detach(),
        "b_total": (b_gen + b_task).detach(),
        "r_l23_probe": r_l23_probe_mean.detach(),
    }


@torch.no_grad()
def _richter_localizer_mean(
    bundle, trailer_pos: int, timing: RichterTiming, *,
    n_trials: int, generator: torch.Generator, noise_std: float,
) -> Tensor:
    from scripts.v2.eval_richter import run_richter_localizer_trial
    rates = []
    for _ in range(n_trials):
        r = run_richter_localizer_trial(
            bundle, trailer_pos=int(trailer_pos),
            timing=timing, noise_std=noise_std, generator=generator,
        )
        rates.append(r)
    return torch.stack(rates, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Paradigm drivers
# ---------------------------------------------------------------------------


def diag1_kok(ckpt: Path, out_path: Path, *, seed: int,
              n_trials_per_cond: int, n_localizer: int) -> dict:
    t0 = time.time()
    bundle = load_checkpoint(ckpt, seed=seed, device="cpu")
    print(f"[kok] ckpt loaded in {time.time()-t0:.1f}s", flush=True)
    timing = KokTiming()
    cue_map = cue_mapping_from_seed(seed)               # {cue_id: orient_deg}
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Classes = expected orientation (element of {45.0, 135.0})
    probe_orients = list(CUE_ORIENTATIONS_DEG)          # [45.0, 135.0]
    ncls = len(probe_orients)

    # --- collect probe trials (balanced: cue×probe) -----------------------
    records: list[dict] = []
    total = 2 * len(probe_orients) * n_trials_per_cond
    t_start = time.time()
    done = 0
    for cue_id in [0, 1]:
        for p_o in probe_orients:
            for _ in range(n_trials_per_cond):
                rec = _kok_probe_and_capture(
                    bundle, cue_id=cue_id, probe_orientation_deg=float(p_o),
                    timing=timing, generator=gen, noise_std=0.0,
                )
                rec["cue_id"] = int(cue_id)
                rec["probe_orient"] = float(p_o)
                rec["expected_class"] = int(
                    probe_orients.index(float(cue_map[cue_id]))
                )  # class = expected orientation
                records.append(rec)
                done += 1
                if done % 10 == 0:
                    el = time.time() - t_start
                    eta = el / done * (total - done)
                    print(f"[kok probe] {done}/{total}  elapsed={el:.1f}s  eta={eta:.1f}s",
                          flush=True)

    # --- localizer templates ---------------------------------------------
    loc_templates: dict[int, Tensor] = {}
    for c, o in enumerate(probe_orients):
        loc_templates[c] = _kok_localizer_mean(
            bundle, orient_deg=float(o), timing=timing,
            n_trials=n_localizer, generator=gen, noise_std=0.0,
        )
        print(f"[kok localizer] class {c} ({o}°) done", flush=True)
    grand_mean_loc = torch.stack(
        [loc_templates[c] for c in range(ncls)], dim=0
    ).mean(dim=0)

    return _summarise(records, loc_templates, grand_mean_loc, ncls, out_path)


def diag1_richter(ckpt: Path, out_path: Path, *, seed: int,
                  n_trials_per_cond: int, n_localizer: int) -> dict:
    t0 = time.time()
    bundle = load_checkpoint(ckpt, seed=seed, device="cpu")
    print(f"[richter] ckpt loaded in {time.time()-t0:.1f}s", flush=True)
    timing = RichterTiming()
    perm = permutation_from_seed(seed)                  # expected_trailer[leader]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ncls = 6

    # Balanced across 6 leaders × 6 trailers
    records: list[dict] = []
    total = 36 * n_trials_per_cond
    t_start = time.time()
    done = 0
    for l_pos in range(6):
        for t_pos in range(6):
            for _ in range(n_trials_per_cond):
                rec = _richter_probe_and_capture(
                    bundle, leader_pos=l_pos, trailer_pos=t_pos,
                    timing=timing, generator=gen, noise_std=0.0,
                )
                rec["leader_pos"] = int(l_pos)
                rec["trailer_pos"] = int(t_pos)
                rec["expected_class"] = int(perm[l_pos])
                records.append(rec)
                done += 1
                if done % 20 == 0:
                    el = time.time() - t_start
                    eta = el / done * (total - done)
                    print(f"[richter probe] {done}/{total}  elapsed={el:.1f}s  eta={eta:.1f}s",
                          flush=True)

    loc_templates: dict[int, Tensor] = {}
    for c in range(ncls):
        loc_templates[c] = _richter_localizer_mean(
            bundle, trailer_pos=int(c), timing=timing,
            n_trials=n_localizer, generator=gen, noise_std=0.0,
        )
        print(f"[richter localizer] class {c} done", flush=True)
    grand_mean_loc = torch.stack(
        [loc_templates[c] for c in range(ncls)], dim=0
    ).mean(dim=0)

    return _summarise(records, loc_templates, grand_mean_loc, ncls, out_path)


def _summarise(records, loc_templates, grand_mean_loc, ncls, out_path):
    # ---- Collect arrays -------------------------------------------------
    m_arr = np.stack([r["m"].numpy() for r in records], axis=0)
    b_gen_arr = np.stack([r["b_gen"].numpy() for r in records], axis=0)
    b_task_arr = np.stack([r["b_task"].numpy() for r in records], axis=0)
    r_arr = np.stack([r["r_l23_probe"].numpy() for r in records], axis=0)
    y_exp = np.array([r["expected_class"] for r in records], dtype=int)

    # ---- Norms ----------------------------------------------------------
    norm_b_task = float(np.linalg.norm(b_task_arr, axis=1).mean())
    norm_b_gen = float(np.linalg.norm(b_gen_arr, axis=1).mean())
    ratio = norm_b_task / (norm_b_gen + 1e-12)

    # ---- Decoder accuracies ---------------------------------------------
    acc_m = _cv_linsvc_acc(m_arr, y_exp)
    acc_bt = _cv_linsvc_acc(b_task_arr, y_exp)
    acc_bg = _cv_linsvc_acc(b_gen_arr, y_exp)
    acc_r = _cv_linsvc_acc(r_arr, y_exp)

    # ---- Per-class b_task mean vs localizer Pearson --------------------
    loc_np = {c: loc_templates[c].numpy() for c in range(ncls)}
    gmean_np = grand_mean_loc.numpy()
    cos_same, cos_other, cos_mean = [], [], []
    cos_bgen_same = []  # causal control
    for c in range(ncls):
        sel = y_exp == c
        if sel.sum() == 0:
            continue
        bt_c = b_task_arr[sel].mean(axis=0)
        bg_c = b_gen_arr[sel].mean(axis=0)
        cs = _pearson_across_units(bt_c, loc_np[c])
        # orthogonal class: use the other class with max dissimilarity
        others = [cc for cc in range(ncls) if cc != c]
        co = float(np.mean([
            _pearson_across_units(bt_c, loc_np[cc]) for cc in others
        ]))
        cm_ = _pearson_across_units(bt_c, gmean_np)
        cos_same.append(cs)
        cos_other.append(co)
        cos_mean.append(cm_)
        cos_bgen_same.append(_pearson_across_units(bg_c, loc_np[c]))

    summary = {
        "n_trials": int(len(records)),
        "n_classes": int(ncls),
        "norm_b_task_mean": norm_b_task,
        "norm_b_gen_mean": norm_b_gen,
        "norm_ratio_task_over_gen": float(ratio),
        "decode_accuracy": {
            "from_m": acc_m,
            "from_b_task": acc_bt,
            "from_b_gen": acc_bg,
            "from_r_l23_probe": acc_r,
            "chance": 1.0 / ncls,
        },
        "per_class_pearson": {
            "b_task_vs_localizer_same_class": cos_same,
            "b_task_vs_localizer_other_class_mean": cos_other,
            "b_task_vs_grand_mean_localizer": cos_mean,
            "b_gen_vs_localizer_same_class_CONTROL": cos_bgen_same,
        },
        "aggregate_pearson": {
            "mean_b_task_vs_localizer_same": float(np.mean(cos_same)) if cos_same else float("nan"),
            "mean_b_task_vs_localizer_other": float(np.mean(cos_other)) if cos_other else float("nan"),
            "mean_b_task_vs_grand_mean": float(np.mean(cos_mean)) if cos_mean else float("nan"),
            "mean_b_gen_vs_localizer_same_CONTROL": float(np.mean(cos_bgen_same)) if cos_bgen_same else float("nan"),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm", choices=["kok", "richter"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials-per-cond", type=int, default=20)
    p.add_argument("--n-localizer", type=int, default=10)
    args = p.parse_args()

    if args.paradigm == "kok":
        summary = diag1_kok(
            args.checkpoint, args.output,
            seed=args.seed,
            n_trials_per_cond=args.n_trials_per_cond,
            n_localizer=args.n_localizer,
        )
    else:
        summary = diag1_richter(
            args.checkpoint, args.output,
            seed=args.seed,
            n_trials_per_cond=args.n_trials_per_cond,
            n_localizer=args.n_localizer,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
