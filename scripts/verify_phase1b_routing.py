"""Unit-level verification: does CompositeLoss route the 3 new Phase 1B terms?

Purely synthetic — no network, no stimulus, no training. Constructs a minimal
`outputs` dict, builds r_l23/q_pred windows with known structure, then calls
CompositeLoss.forward() under four routing configurations. Checks that each of
``sharp``, ``local_disc``, ``pred_suppress`` in loss_dict scales correctly with
the per-sample task_state mapped through the sweep_dual_1b task_routing.

For each term we use routing multipliers 0.0 (focused/routine) and 2.0
(routine/focused) — so:
    * routing=None            → baseline (legacy scalar form)
    * all-focused             → focused multiplier applied
    * all-routine             → routine multiplier applied
    * mixed 50/50             → (focused + routine) / 2

Configured Phase 1B task_routing:
    sharp:          focused=2.0, routine=0.0
    local_disc:     focused=2.0, routine=0.0
    pred_suppress:  focused=0.0, routine=2.0

Expected per-term results (baseline = legacy scalar value with routing off):
    sharp:
        all-focused → 2.0 * baseline
        all-routine → 0.0
        mixed       → 1.0 * baseline
    local_disc:
        all-focused → 2.0 * baseline
        all-routine → 0.0
        mixed       → 1.0 * baseline
    pred_suppress:
        all-focused → 0.0
        all-routine → 2.0 * baseline
        mixed       → 1.0 * baseline

If any actual value diverges from expected beyond 1e-5, routing is broken for
that term.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config
from src.training.losses import CompositeLoss


def main() -> None:
    cfg_path = "config/sweep/sweep_dual_1b.yaml"
    model_cfg, train_cfg, stim_cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"lambda_sharp         = {train_cfg.lambda_sharp}")
    print(f"lambda_local_disc    = {train_cfg.lambda_local_disc}")
    print(f"lambda_pred_suppress = {train_cfg.lambda_pred_suppress}")
    print(f"task_routing         = {train_cfg.task_routing}")
    print()

    B, W, N = 8, 4, model_cfg.n_orientations
    torch.manual_seed(42)

    # Build synthetic outputs dict — needed to satisfy forward() shape checks
    # but the tested losses only depend on r_l23_windows / q_pred_windows /
    # true_theta_windows. Keep these small and fixed so per-sample baselines
    # are identical across configurations.
    T = 12
    outputs = {
        "r_l4":          torch.rand(B, T, N, device=device) * 0.1,
        "r_l23":         torch.rand(B, T, N, device=device) * 0.1,
        "r_pv":          torch.rand(B, T, 1, device=device) * 0.1,
        "r_som":         torch.rand(B, T, N, device=device) * 0.1,
        "deep_template": torch.rand(B, T, N, device=device) * 0.1,
        "center_exc":    torch.ones(B, T, N, device=device),
    }

    # Windowed inputs with per-sample-identical structure:
    # r_l23_windows is ones — same per-sample magnitude everywhere, so any
    # variation in loss_dict is due to routing, not per-sample content drift.
    r_l23_windows = torch.ones(B, W, N, device=device)
    # Uniform q_pred so pred_suppress per-sample baseline is a constant.
    q_pred_windows = torch.full((B, W, N), 1.0 / N, device=device)
    # true_theta = 0 so the sharpness weight structure is identical per sample.
    true_theta = torch.zeros(B, W, device=device)
    true_next = torch.zeros(B, W, device=device)

    loss_fn = CompositeLoss(train_cfg, model_cfg).to(device)

    # First, capture the baseline (non-routed) values. Because r_l23_windows is
    # ones and true_theta is 0, every sample contributes identically, so the
    # scalar baseline equals any per-sample value.
    _, ld_base = loss_fn(
        outputs, true_theta, true_next,
        r_l23_windows, q_pred_windows,
        fb_scale=1.0,
        task_state=None,
        task_routing=None,
    )
    base_sharp = ld_base["sharp"]
    base_local = ld_base["local_disc"]
    base_pred  = ld_base["pred_suppress"]
    print("Baselines (routing=None):")
    print(f"  sharp         = {base_sharp:.6f}")
    print(f"  local_disc    = {base_local:.6f}")
    print(f"  pred_suppress = {base_pred:.6f}")
    print()

    # Task routing from the config
    tr = train_cfg.task_routing
    f_sharp = float(tr["focused"].get("sharp", 0.0))
    r_sharp = float(tr["routine"].get("sharp", 0.0))
    f_local = float(tr["focused"].get("local_disc", 0.0))
    r_local = float(tr["routine"].get("local_disc", 0.0))
    f_pred  = float(tr["focused"].get("pred_suppress", 0.0))
    r_pred  = float(tr["routine"].get("pred_suppress", 0.0))

    def exp_mult(focused_frac: float, f_mult: float, r_mult: float) -> float:
        """Expected routed value = (focused_frac*f + (1-focused_frac)*r) * base."""
        return focused_frac * f_mult + (1.0 - focused_frac) * r_mult

    configs = {
        "1. all-focused  [1,0]*B":     (torch.tensor([[1., 0.]] * B, device=device), 1.0),
        "2. all-routine  [0,1]*B":     (torch.tensor([[0., 1.]] * B, device=device), 0.0),
        "3. mixed 50/50":              (
            torch.tensor([[1., 0.]] * (B // 2) + [[0., 1.]] * (B // 2),
                         device=device), 0.5),
    }

    print(f"{'config':30s}  {'term':14s}  {'actual':>12s}  {'expected':>12s}  verdict")
    print("-" * 90)
    any_fail = False
    for name, (ts, focused_frac) in configs.items():
        _, ld = loss_fn(
            outputs, true_theta, true_next,
            r_l23_windows, q_pred_windows,
            fb_scale=1.0,
            task_state=ts,
            task_routing=tr,
        )
        checks = [
            ("sharp",         ld["sharp"],         exp_mult(focused_frac, f_sharp, r_sharp) * base_sharp),
            ("local_disc",    ld["local_disc"],    exp_mult(focused_frac, f_local, r_local) * base_local),
            ("pred_suppress", ld["pred_suppress"], exp_mult(focused_frac, f_pred,  r_pred)  * base_pred),
        ]
        for term, actual, expected in checks:
            ok = abs(actual - expected) < 1e-5
            if not ok:
                any_fail = True
            marker = "PASS" if ok else "FAIL"
            print(f"{name:30s}  {term:14s}  {actual:>12.6f}  {expected:>12.6f}  {marker}")
    print()
    if any_fail:
        print("RESULT: FAIL — one or more routed terms did not scale as expected.")
        sys.exit(1)
    else:
        print("RESULT: PASS — all three new terms route correctly per task_state.")


if __name__ == "__main__":
    main()
