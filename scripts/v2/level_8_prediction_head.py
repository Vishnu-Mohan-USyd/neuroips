"""Level 8 component validation — prediction head sanity (Task #74).

Per Lead's bottom-up validation protocol. Scope: with plasticity OFF and the
full Fix-K + L2 + M + N substrate, drive the network through a held-out
procedural-world trajectory and ask whether :class:`PredictionHead` is
*structurally* engaged.

Verdict logic (two-tier per Lead's Level-7-style precedent)
-----------------------------------------------------------
**pass** (quantitatively competitive with baselines — expected only after
Phase-2 plasticity has trained the head):
  1. head_mse < 0.8 · random_mse
  2. head_mse ≤ 1.5 · copy_last_mse
  3. ‖x̂‖_mean ∈ [0.5, 2.0] × ‖x‖_mean

**neutral_baseline** (at-init, T29 calibration intentionally starts all raws
at -10.0 so ``softplus(-10) ≈ 4.54e-5`` → x̂ ≈ 1e-2 at blank; quantitative
gates will fail by design, but structural capability should already be
intact):
  1. ‖x̂‖_mean > 0                                — head is not zeroed.
  2. ‖x̂‖_mean < 5 · ‖x‖_mean                     — head is not exploded.
  3. x̂ has shape [B, n_l4_e] matching target    — shape correct.
  4. **Stream ablation** — each of (h_rate, c_bias, l23_apical_summary)
     demonstrably changes x̂ when zeroed, proving all three pathways engage
     (not just one dominating):
        relative L2 change ‖x̂_full − x̂_ablated‖ / ‖x̂_full‖ ≥ 1e-3.

**fail** otherwise.

Follow-up requirement (documented, not implemented here)
--------------------------------------------------------
After Phase-2 plasticity has trained the head, re-run this probe and require
the pass verdict (head_mse < 1.5 × copy_last_mse or better). Until then,
neutral_baseline is the honest verdict.

Target convention
-----------------
At forward step ``t`` (input frame ``x_t``):
  * ``r_l4_new_t`` is the L4 E rate computed *from* ``x_t`` (info["r_l4"]).
  * ``x_hat_next_t`` is the prediction of the *next* step's L4 E rate.

Pointwise comparison (for the quantitative tier):
  predict[t] = x_hat_next_t ;  target[t] = r_l4_new_{t+1} ; t ∈ [0, N-2].

Held-out trajectory: ``seed_family="eval"`` (SEED_BASE_EVAL=9000),
``trajectory_seed=1`` → world RNG seed 9001; 200 frames.

DM::
  level8_verdict=<pass|neutral_baseline|fail>
    head_mse=<#> copy_last_mse=<#> random_mse=<#> uniform_mse=<#>
    head_vs_copy_ratio=<#> ‖x̂‖_mean=<#> ‖x‖_mean=<#>
    ablate_h_dx=<#> ablate_c_dx=<#> ablate_apical_dx=<#>
    issue_if_fail=<short>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.world.procedural import ProceduralWorld
from src.v2_model.stimuli.feature_tokens import TokenBank


@torch.no_grad()
def _run_trajectory(
    net: V2Network, frames: Tensor, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Tensor, Tensor, Tensor]:
    """Step the network through ``frames``; collect sequences and a mid-trajectory
    snapshot of head-input tensors for the ablation probe.

    Parameters
    ----------
    net : V2Network
    frames : Tensor [N, 1, H, W]
    device : torch.device

    Returns
    -------
    x_hat_seq : np.ndarray [N, n_l4_e]  (float64)
    r_l4_seq : np.ndarray [N, n_l4_e]   (float64)
    h_snap, m_snap, l23_snap : Tensor
        Snapshot of ``r_h_new``, ``m`` (post-step), ``r_l23_new`` at the
        trajectory midpoint, for stream-ablation probing.
    """
    n_steps = int(frames.shape[0])
    state = net.initial_state(batch_size=1)

    x_hats: list[np.ndarray] = []
    r_l4s: list[np.ndarray] = []

    mid = n_steps // 2
    h_snap: Tensor | None = None
    m_snap: Tensor | None = None
    l23_snap: Tensor | None = None

    for t in range(n_steps):
        x_t = frames[t].unsqueeze(0).to(device)  # [1, 1, H, W]
        x_hat_next, state, info = net(x_t, state, q_t=None)
        x_hats.append(x_hat_next.squeeze(0).cpu().numpy().astype(np.float64))
        r_l4s.append(info["r_l4"].squeeze(0).cpu().numpy().astype(np.float64))
        if t == mid:
            h_snap = info["r_h"].detach().clone()
            m_snap = info["m"].detach().clone()
            l23_snap = info["r_l23"].detach().clone()

    assert h_snap is not None and m_snap is not None and l23_snap is not None
    return np.stack(x_hats), np.stack(r_l4s), h_snap, m_snap, l23_snap


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error across all (time, unit) entries."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


@torch.no_grad()
def _stream_ablation(
    net: V2Network, h: Tensor, m: Tensor, l23: Tensor,
) -> dict[str, float]:
    """Probe each of the three prediction-head input streams.

    Calls ``net.prediction_head`` four times:
      * full     : all three streams present
      * no_h     : h_rate set to zeros (required input, cannot be None)
      * no_c     : c_bias=None
      * no_apical: l23_apical_summary=None

    Returns relative L2 change  ‖x̂_full − x̂_ablated‖ / ‖x̂_full‖  for each
    stream. Values ≥ 1e-3 indicate that stream demonstrably affects x̂.
    """
    head = net.prediction_head
    h0 = torch.zeros_like(h)

    x_full = head(h_rate=h, c_bias=m, l23_apical_summary=l23)
    x_no_h = head(h_rate=h0, c_bias=m, l23_apical_summary=l23)
    x_no_c = head(h_rate=h, c_bias=None, l23_apical_summary=l23)
    x_no_a = head(h_rate=h, c_bias=m, l23_apical_summary=None)

    denom = float(torch.linalg.norm(x_full).cpu())
    if denom < 1e-30:
        return {
            "ablate_h_dx": float("nan"),
            "ablate_c_dx": float("nan"),
            "ablate_apical_dx": float("nan"),
            "x_full_norm": denom,
        }

    d_h = float(torch.linalg.norm(x_full - x_no_h).cpu()) / denom
    d_c = float(torch.linalg.norm(x_full - x_no_c).cpu()) / denom
    d_a = float(torch.linalg.norm(x_full - x_no_a).cpu()) / denom

    return {
        "ablate_h_dx": d_h,
        "ablate_c_dx": d_c,
        "ablate_apical_dx": d_a,
        "x_full_norm": denom,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory-seed", type=int, default=1,
                   help="added to SEED_BASE_EVAL=9000 → world RNG seed")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--ablation-threshold", type=float, default=1e-3,
                   help="min relative L2 change per stream for engagement")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = ModelConfig(seed=seed, device="cpu")
    device = torch.device("cpu")
    bank = TokenBank(cfg, seed=0)
    world = ProceduralWorld(cfg, token_bank=bank, seed_family="eval")
    net = V2Network(cfg, token_bank=bank, seed=seed, device="cpu")
    net.eval()
    n_l4_e_expected = int(net.prediction_head.n_l4_e)

    # --- Held-out trajectory --------------------------------------------------
    frames, _states = world.trajectory(
        trajectory_seed=int(args.trajectory_seed),
        n_steps=int(args.n_steps),
    )
    # frames: [n_steps, 1, 32, 32]

    x_hat_seq, r_l4_seq, h_snap, m_snap, l23_snap = _run_trajectory(
        net, frames, device=device,
    )
    # x_hat_seq, r_l4_seq: [n_steps, n_l4_e]

    # Shape sanity (required by the neutral_baseline tier).
    shape_ok = (
        x_hat_seq.ndim == 2
        and x_hat_seq.shape[1] == n_l4_e_expected
        and r_l4_seq.shape == x_hat_seq.shape
    )

    predict = x_hat_seq[:-1]
    target = r_l4_seq[1:]

    head_mse = _mse(predict, target)
    copy_last_mse = _mse(r_l4_seq[:-1], target)
    uniform_pred = np.broadcast_to(
        r_l4_seq.mean(axis=0, keepdims=True), target.shape,
    )
    uniform_mean_mse = _mse(uniform_pred, target)
    rng = np.random.default_rng(seed)
    random_pred = target[rng.permutation(target.shape[0])]
    random_mse = _mse(random_pred, target)

    x_hat_norm_mean = float(np.linalg.norm(predict, axis=1).mean())
    x_actual_norm_mean = float(np.linalg.norm(target, axis=1).mean())
    norm_ratio = (
        x_hat_norm_mean / x_actual_norm_mean
        if x_actual_norm_mean > 1e-30 else float("inf")
    )
    head_vs_copy_ratio = (
        head_mse / copy_last_mse if copy_last_mse > 1e-30 else float("inf")
    )
    head_vs_random_ratio = (
        head_mse / random_mse if random_mse > 1e-30 else float("inf")
    )

    # --- Stream ablation (Lead-mandated) --------------------------------------
    ablate = _stream_ablation(net, h_snap, m_snap, l23_snap)

    # --- Verdict logic: pass → neutral_baseline → fail ------------------------
    pass_fails: list[str] = []
    if not (head_mse < 0.8 * random_mse):
        pass_fails.append(
            f"head_mse {head_mse:.3e}≥0.8·random {0.8 * random_mse:.3e}"
        )
    if not (head_mse <= 1.5 * copy_last_mse):
        pass_fails.append(
            f"head_mse {head_mse:.3e}>1.5·copy_last {1.5 * copy_last_mse:.3e}"
        )
    if not (0.5 <= norm_ratio <= 2.0):
        pass_fails.append(f"norm_ratio {norm_ratio:.3f}∉[0.5,2.0]")

    # Neutral-baseline (at-init structural) checks.
    nb_fails: list[str] = []
    if not (x_hat_norm_mean > 0.0):
        nb_fails.append(f"‖x̂‖_mean {x_hat_norm_mean:.3e}≤0 (head zeroed)")
    if not (x_hat_norm_mean < 5.0 * x_actual_norm_mean):
        nb_fails.append(
            f"‖x̂‖_mean {x_hat_norm_mean:.3e}≥5·‖x‖ "
            f"{5.0 * x_actual_norm_mean:.3e} (head exploded)"
        )
    if not shape_ok:
        nb_fails.append(
            f"shape_mismatch: x_hat {x_hat_seq.shape} vs L4 "
            f"[_, {n_l4_e_expected}]"
        )
    for label, key in (
        ("h_rate", "ablate_h_dx"),
        ("c_bias", "ablate_c_dx"),
        ("l23_apical", "ablate_apical_dx"),
    ):
        val = ablate[key]
        if not (val >= args.ablation_threshold):
            nb_fails.append(
                f"{label}_ablation Δx/‖x‖={val:.3e}<{args.ablation_threshold}"
            )

    if not pass_fails:
        verdict = "pass"
        issue = "none"
    elif not nb_fails:
        verdict = "neutral_baseline"
        issue = (
            "none (structural capability intact; quantitative gates fail at "
            "init by T29 design — re-check post-Phase-2)"
        )
    else:
        verdict = "fail"
        issue = ";".join(nb_fails)

    # --- Write summary --------------------------------------------------------
    summary: dict[str, Any] = {
        "version": "level_8_prediction_head_v2",
        "seed": seed,
        "trajectory_seed": int(args.trajectory_seed),
        "world_rng_seed": 9000 + int(args.trajectory_seed),
        "n_steps": int(args.n_steps),
        "n_paired": int(target.shape[0]),
        "n_l4_e": int(target.shape[1]),
        "shape_ok": bool(shape_ok),
        "stats": {
            "head_mse": head_mse,
            "copy_last_mse": copy_last_mse,
            "uniform_mean_mse": uniform_mean_mse,
            "random_mse": random_mse,
            "head_vs_copy_ratio": head_vs_copy_ratio,
            "head_vs_random_ratio": head_vs_random_ratio,
            "x_hat_norm_mean": x_hat_norm_mean,
            "x_actual_norm_mean": x_actual_norm_mean,
            "norm_ratio": norm_ratio,
            **ablate,
        },
        "pass_fails": pass_fails,
        "nb_fails": nb_fails,
        "verdict": verdict,
        "issue_if_fail": issue,
        "followup_note": (
            "After Phase-2 trains the prediction head, re-run this probe "
            "and require verdict=pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    line = (
        f"level8_verdict={verdict} "
        f"head_mse={head_mse:.3e} "
        f"copy_last_mse={copy_last_mse:.3e} "
        f"random_mse={random_mse:.3e} "
        f"uniform_mse={uniform_mean_mse:.3e} "
        f"head_vs_copy_ratio={head_vs_copy_ratio:.3f} "
        f"‖x̂‖_mean={x_hat_norm_mean:.3e} "
        f"‖x‖_mean={x_actual_norm_mean:.3e} "
        f"ablate_h_dx={ablate['ablate_h_dx']:.3e} "
        f"ablate_c_dx={ablate['ablate_c_dx']:.3e} "
        f"ablate_apical_dx={ablate['ablate_apical_dx']:.3e} "
        f"issue_if_fail={issue}"
    )
    print(line)
    print(f"[wrote] {args.output}")
    return 0 if verdict in ("pass", "neutral_baseline") else 1


if __name__ == "__main__":
    raise SystemExit(main())
