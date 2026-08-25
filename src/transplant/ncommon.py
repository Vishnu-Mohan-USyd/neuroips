#!/usr/bin/env python3
"""Shared harness for the SURROUND-architecture transplant study (Phase 2).

Governing docs: /home/vishnu/neuroips_analysis/transplant_surround_20260823/
{PROTOCOL,DESIGN}.md. Conventions are inherited from the frozen
transplant_20260818 harness: this module imports that study's `tcommon` and
reuses its functions VERBATIM (markers, ce_profile, make_battery, ce_channels,
hybrid_state_dict, save_hybrid, k_from_raw, g_from_raw, numeric_leaf_diffs,
component tables, CE gate). Donors are swapped to the s=0.04 ladder endpoints.

Profile coordinates (center/flank/H/M/rate/vitality) use the flank study's
G0-anchor measurement core: `bin_mean` + `measure` copied VERBATIM from
/home/vishnu/scratch/flank_sharpening_20260819/ladder_eval_dampening.py
(the code path whose numbers the validator reproduced at abs diff 0.0).

DISCLOSED DEVIATION (DESIGN inheritance note): `instrumented_unroll` below is
the e2_replay clone with ONE line removed — the no-surround-era scope guard
`assert net.pred_inhib_strength == 0.0` (it would reject every s=0.04 net).
The pred_inhib term was already implemented correctly in the clone for any
strength, and the REAL gate — the per-step bitwise assert of the decomposition
against the repo forward `net.l23` — is retained unchanged on every step of
every net. All other scope asserts kept.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ORIG_SCRIPTS = Path("/home/vishnu/neuroips_analysis/transplant_20260818/scripts")
sys.path.insert(0, str(ORIG_SCRIPTS))
import tcommon as T0  # noqa: E402  (frozen original harness, reused verbatim)
from tcommon import assay, gates, tuned, simple  # noqa: E402

# ----------------------------------------------------------------- study paths
ROOT = Path("/home/vishnu/scratch/transplant_surround_20260823")
HYB = ROOT / "hybrids"
DONOR = Path("/home/vishnu/scratch/flank_sharpening_20260819/runs/ladder_s0p04")
FLANK_EVAL = Path("/home/vishnu/scratch/flank_sharpening_20260819/runs/ladder_s0p04")
RUN_LOG = ROOT / "RUN_LOG.md"

SEEDS = T0.SEEDS                      # (8, 9, 10, 11)
ARMS = T0.ARMS                        # ("alpha0.0", "alpha0.5")
ALPHA_OF, SLUG_OF, DIR_OF = T0.ALPHA_OF, T0.SLUG_OF, T0.DIR_OF
CELL_IDS = T0.CELL_IDS                # PPP..TTT (8 core cells)
COMPONENTS, COMP_TENSORS = T0.COMPONENTS, T0.COMP_TENSORS
CE_GATE = T0.CE_GATE                  # 3*ln36 = 10.7506
N = T0.N

# control cells (DESIGN 2.3): FB replaced by R / N / Q
CTRL_TRAINED = ("TRT", "TNT", "TQT")            # trained context, all seeds x arms
CTRL_HOST = ("PRP", "PNP", "PQP")               # host context, seed 8 only
SHARED_CELLS = ("PPP", "PRP")                   # regime-independent nets, dual filename
R_SEED, Q_SEED = 20260823, 20260824             # pinned RNG (DESIGN 2.3)

# G6 anchor: seed-8 pretrain STATE sha (harness state_sha256 recipe)
PRETRAIN_S8_STATE_SHA = "4c5b1a320300630cafcf1b2cbce77dd3c05abf7128aa3eb3eb24b46457bc4236"

# G0 anchor KEYS per arm (DESIGN 3.6). Anchor VALUES are loaded from the
# sha-pinned artifacts at full float precision — lead ruling 2026-08-23
# (PROTOCOL.md): "authoritative anchors are the cited ARTIFACT values …
# never document-transcribed decimals."
G0_ANCHOR_KEYS = {
    "alpha0.0": ("flank_ratio", "center_ratio", "H"),
    "alpha0.5": ("M_auc_ratio", "center_ratio", "flank_ratio", "H",
                 "continuation_mean_rate"),
}
EVAL_SHAS = {
    (8, "alpha0.0"): "fdf48fea678a0529e9461e233cc72cf82d55f4aac9f3a8fdd643fe250b7eae57",
    (8, "alpha0.5"): "d01f88f9692866e3ceccf3bbbafa1c9e1a24ce8b89f14e8004c7df7d8f4daae7",
    (9, "alpha0.0"): "ce9b1724d39945d40d799f986102713ffb6a79778a9a252bdbf9067005fbf98e",
    (9, "alpha0.5"): "cf97195f42194a94e4b4626b29950f33d0f1be16774ea8383ade81e7bc450eed",
    (10, "alpha0.0"): "af8cd13545ccbc859ac5953d542d2e40ce3b0b318c4f00cd780567aa1787278d",
    (10, "alpha0.5"): "b1f188018f21ffb46e015cae42bdd848b44088481a0a1645e507b60a26ce8d2f",
    (11, "alpha0.0"): "e986f5e6e8e87be02c4f35a17ac10f321c70ce9d186bf104112dc48a039d6257",
    (11, "alpha0.5"): "adee6b55b646b08e91d3edad335d3ee732aab3b33693716c15d8ed4ecd7e4a04",
}
FROZEN_M_ANCHOR_ORIGINAL = T0.FROZEN_M_ANCHOR   # 0.3320623037521497 (evaluator sanity)

# reused verbatim from the frozen original harness
markers = T0.markers
numeric_leaf_diffs = T0.numeric_leaf_diffs
ce_profile = T0.ce_profile
make_battery = T0.make_battery
ce_channels = T0.ce_channels
hybrid_state_dict = T0.hybrid_state_dict
save_hybrid = T0.save_hybrid
k_from_raw = T0.k_from_raw
g_from_raw = T0.g_from_raw
sha256_file = T0.sha256_file


# ------------------------------------------------------------- donor locations
def donor_dir(seed: int, arm: str) -> Path:
    return DONOR / f"{DIR_OF[arm]}_seed{seed}" / f"seed_{seed}"


def arm_path(seed: int, arm: str) -> Path:
    return donor_dir(seed, arm) / SLUG_OF[arm]


def pretrain_path(seed: int, arm: str = "alpha0.0") -> Path:
    """Seed pretrain. G2 proves the two regime dirs' pretrains bitwise
    identical; after G2 the alpha0.0 dir's copy is THE seed pretrain."""
    return donor_dir(seed, arm) / "common_pretrain_final.pt"


def stored_eval(seed: int, arm: str) -> dict:
    """Sha-pinned stored eval report (the flank study's coder evals, which the
    validator re-derived at zero mismatches). Sha verified on every read."""
    p = FLANK_EVAL / f"eval_{DIR_OF[arm]}_seed{seed}.json"
    sha = sha256_file(p)
    if sha != EVAL_SHAS[(seed, arm)]:
        raise SystemExit(f"stored eval sha mismatch for {p}: {sha} — STOP")
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- cell layout
def cell_dir(seed: int, arm: str, cell_id: str) -> Path:
    if cell_id in SHARED_CELLS:
        return HYB / f"seed_{seed}" / cell_id
    return HYB / f"seed_{seed}" / DIR_OF[arm] / cell_id


def cell_ckpt(seed: int, arm: str, cell_id: str) -> Path:
    return cell_dir(seed, arm, cell_id) / SLUG_OF[arm]


def all_table_cells():
    """Yield (seed, arm, cell_id) for all 94 table cells (DESIGN 2.2-2.3)."""
    for seed in SEEDS:
        for arm in ARMS:
            for cid in CELL_IDS:
                yield seed, arm, cid
            for cid in CTRL_TRAINED:
                yield seed, arm, cid
    for arm in ARMS:
        for cid in CTRL_HOST:
            yield 8, arm, cid


def distinct_net_key(seed: int, arm: str, cell_id: str):
    """Nets shared across arms (PPP, PRP) count once."""
    if cell_id in SHARED_CELLS:
        return (seed, "shared", cell_id)
    return (seed, arm, cell_id)


# ------------------------------------------------------------------- utilities
def state_sha256(state: dict) -> str:
    """Harness recipe verbatim (train_sweep.py state_sha256)."""
    import hashlib
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def mem_available_kb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable not found")


def mem_gate() -> None:
    kb = mem_available_kb()
    if kb < 25 * 1024 * 1024:
        raise SystemExit(f"MemAvailable {kb} kB < 25 GB — STOP (envelope)")


def heartbeat(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RUN_LOG, "a") as fh:
        fh.write(f"- [{stamp}] {msg}\n")


def release(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------- flank-study profile (verbatim)
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
VITALITY_OFFSETS = (-2, -1, 0, 1, 2)
VITALITY_FLOOR = 0.01
RATE_FLOOR = 0.01
PLOT_OFFSETS = tuple(range(-12, 13))


def bin_mean(curve: torch.Tensor, offsets) -> float:
    idx = [assay.OFFSETS.index(o) for o in offsets]
    return float(curve[:, idx].mean().item())


@torch.no_grad()
def measure(net, ck, device) -> dict:
    center_feedback = bool(ck.get("center_feedback", False))
    feedback_mode = tuned.resolve_feedback_mode(center_feedback,
                                                ck.get("feedback_mode"))
    theta_a, theta_b, finals = assay.matched_pairs(device)
    preds_a, rates_a = tuned.forward_seq_tuned(
        net, theta_a, 1.0, center_feedback=center_feedback,
        feedback_mode=feedback_mode)
    _, rates_b = tuned.forward_seq_tuned(
        net, theta_b, 1.0, center_feedback=center_feedback,
        feedback_mode=feedback_mode)
    H = float((preds_a[:, 3, :].argmax(-1) == finals).float().mean())
    final_rates = rates_a[:, -1, :]
    aligned_a = assay.align_rates(final_rates, finals).to(torch.float64)
    first_a = (theta_a[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    first_b = (theta_b[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    t0 = 0.5 * (assay.align_rates(rates_a[:, 0, :], first_a).to(torch.float64)
                + assay.align_rates(rates_b[:, 0, :], first_b).to(torch.float64))
    mean_profile = aligned_a.mean(dim=0)
    vit = {str(o * 5): float(mean_profile[assay.OFFSETS.index(o)].item())
           for o in VITALITY_OFFSETS}
    plot_idx = [assay.OFFSETS.index(o) for o in PLOT_OFFSETS]
    rate_A = float(aligned_a.mean().item())
    mean_rate_t0 = float(t0.mean().item())
    fpos = tuned.predictive_feedback_evidence(preds_a, center_feedback,
                                              feedback_mode)
    return {
        "H": H,
        "center_ratio": bin_mean(aligned_a, CENTER_OFFSETS)
        / bin_mean(t0, CENTER_OFFSETS),
        "flank_ratio": bin_mean(aligned_a, FLANK_OFFSETS)
        / bin_mean(t0, FLANK_OFFSETS),
        "M_auc_ratio": rate_A / mean_rate_t0,
        "continuation_mean_rate": rate_A,
        "mean_rate_t0": mean_rate_t0,
        "vitality_band": vit,
        "vitality_pass": all(v > VITALITY_FLOOR for v in vit.values()),
        "mean_profile_max": float(mean_profile.max().item()),
        "mean_profile_min": float(mean_profile.min().item()),
        "feedback_positive_mean": float(fpos.mean().item()),
        "curves_offsets_deg": [o * assay.STEP_DEG for o in PLOT_OFFSETS],
        "curve_adapted": mean_profile[plot_idx].cpu().tolist(),
        "curve_baseline_t0": t0.mean(dim=0)[plot_idx].cpu().tolist(),
    }


@torch.no_grad()
def measure_path(path: Path, device) -> dict:
    net, ck = assay.load_arm(path, device)
    out = measure(net, ck, device)
    release(net)
    return out


@torch.no_grad()
def measure_path_s0(path: Path, device) -> dict:
    """A4-style s->0 inference counterfactual (flank study convention)."""
    import copy as _copy
    ck = torch.load(path, map_location=device)
    cfg = _copy.deepcopy(ck["tuned_net_config"])
    cfg["pred_inhib_strength"] = 0.0
    net = tuned.build_tuned_from_config(cfg).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    out = measure(net, ck, device)
    release(net, ck)
    return out


# ------------------------------------- e2_replay clone (one-line deviation)
@torch.no_grad()
def instrumented_unroll(net, theta, feedback_mode, device):
    """Bitwise-gated clone of tuned.forward_seq_tuned (fb_scale=1.0).

    VERBATIM from e2_replay.instrumented_unroll EXCEPT the removed line
    `assert net.pred_inhib_strength == 0.0` (no-surround-era scope guard; the
    clone's own pred_inhib line handles any strength). Per-step bitwise gate
    vs the repo forward retained.
    """
    assert net.pred_feature_supp_strength == 0.0
    assert net.adapt_strength == 0.0
    assert net.rate_saturation_r_max == 0.0
    assert net.local_comp_mode == "divisive" and abs(net.local_comp_power - 1.0) < 1e-6
    batch = theta.shape[0]
    h = torch.zeros(batch, net.hidden, device=device)
    pred_down = torch.zeros(batch, N, device=device)
    adapt_state = torch.zeros(batch, N, device=device)
    g = F.softplus(net.circ_raw)
    k_scalar = g[3] - g[4] * torch.relu(g[1] - g[2] * g[0])
    strength = net.local_comp_effective_strength()
    rec = {"preds": [], "f": [], "h_seq": [], "rates": []}
    for t in range(theta.shape[1]):
        l4 = simple.l4_code(theta[:, t])
        fb = 1.0 * pred_down
        drive = net.feedforward(l4)
        fb_pos = F.relu(fb)
        vip = F.relu(g[0] * fb_pos)
        som = F.relu(g[1] * fb_pos - g[2] * vip)
        pred_inhib = net.pred_inhib_strength * (fb_pos @ net.pred_inhib_weight.t())
        pred_feature_supp = net.pred_feature_supp_strength * fb_pos
        adapt = net.adapt_strength * adapt_state
        u_pre = drive + g[3] * fb_pos - g[4] * som - pred_inhib - pred_feature_supp - adapt
        u = F.relu(u_pre)
        local_pool = u @ net.local_comp_weight.t()
        r = u / (1.0 + strength * local_pool).clamp_min(1e-6)
        r_repo = net.l23(l4, fb, adapt_state)
        assert torch.equal(r, r_repo), f"decomposition mismatch at t={t}"
        rec["rates"].append(r)
        adapt_state = net.update_adaptation(adapt_state, r)
        h = net.gru(r, h)
        pred = net.W_fb(h)
        rec["preds"].append(pred)
        rec["h_seq"].append(h)
        pred_down = tuned.predictive_feedback_evidence(pred, False, feedback_mode)
        rec["f"].append(pred_down)
    out = {k: torch.stack(v, 1) for k, v in rec.items()}
    out["k_scalar"] = k_scalar
    return out


def circ_offset(peak, target):
    """e2_replay.circ_offset, verbatim."""
    return ((peak - target + N // 2) % N) - N // 2


@torch.no_grad()
def placement_hits(path: Path, device, battery):
    """T0.placement_hits, verbatim except it calls the local clone above."""
    net, ck = assay.load_arm(path, device)
    mode = tuned.resolve_feedback_mode(bool(ck.get("center_feedback", False)),
                                       ck.get("feedback_mode"))
    recs = {c: instrumented_unroll(net, battery["theta"][c], mode, device)
            for c in ("A", "B")}
    ch, finals, velocities = battery["ch"], battery["finals"], battery["velocities"]
    extrap = (finals - 2 * velocities) % N
    hits = {"A_on_y": [], "B_on_y_minus_2v": [], "equality": []}
    for t in range(4):
        pa = recs["A"]["preds"][:, t, :].argmax(-1)
        pb = recs["B"]["preds"][:, t, :].argmax(-1)
        if t == 3:
            ha = float((circ_offset(pa, finals) == 0).float().mean())
            hb = float((circ_offset(pb, extrap) == 0).float().mean())
        else:
            ha = float((circ_offset(pa, ch["A"][:, t + 1]) == 0).float().mean())
            hb = float((circ_offset(pb, ch["B"][:, t + 1]) == 0).float().mean())
        hits["A_on_y"].append(ha)
        hits["B_on_y_minus_2v"].append(hb)
        hits["equality"].append(abs(ha - hb))
    hits["k_scalar"] = float(recs["A"]["k_scalar"])
    release(net, recs)
    return hits
