"""Unit-level verification: does CompositeLoss route fb_energy per-sample?

Purely synthetic — no network, no stimulus, no training. Constructs a minimal
`outputs` dict with a KNOWN center_exc magnitude, calls CompositeLoss.forward()
under four routing configurations, and checks that loss_dict['fb_energy']
scales correctly with the per-sample task_state.

If routing works, we should see:
    * routing=None        → baseline mean magnitude
    * all-focused (0.0×)  → 0.0
    * all-routine (2.0×)  → 2.0 × baseline
    * 50/50 mixed         → 1.0 × baseline (mean of 0 and 2×0.5 = 1)
If any of these are equal where they should differ, routing is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config
from src.training.losses import CompositeLoss


def main() -> None:
    cfg_path = "config/sweep/sweep_dual_1a.yaml"
    model_cfg, train_cfg, stim_cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"lambda_fb_energy = {train_cfg.lambda_fb_energy}")
    print(f"task_routing     = {train_cfg.task_routing}")
    print()

    B, T, N = 8, 12, model_cfg.n_orientations
    torch.manual_seed(42)

    # Synthetic outputs with a known, non-zero center_exc. All samples have
    # identical |center_exc|.abs().mean() so any difference in reported
    # fb_energy loss is due to routing, not to sample-level variance.
    # Use ones to make the magnitude exactly 1.0.
    center_exc = torch.ones(B, T, N, device=device)  # abs().mean() == 1.0
    r_l23 = torch.rand(B, T, N, device=device) * 0.1  # small L2/3 activity
    r_l4 = torch.rand(B, T, N, device=device) * 0.1
    r_pv = torch.rand(B, T, 1, device=device) * 0.1
    r_som = torch.rand(B, T, N, device=device) * 0.1
    deep_template = torch.rand(B, T, N, device=device) * 0.1

    outputs = {
        "r_l4": r_l4,
        "r_l23": r_l23,
        "r_pv": r_pv,
        "r_som": r_som,
        "deep_template": deep_template,
        "center_exc": center_exc,   # KEY: abs().mean() == 1.0 per sample
    }

    loss_fn = CompositeLoss(train_cfg, model_cfg).to(device)

    # Minimal windowed inputs (shapes: [B, W, N] / [B, W])
    r_l23_windows = r_l23[:, :1]                                   # [B, 1, N]
    q_pred_windows = torch.softmax(
        torch.randn(B, 1, N, device=device), dim=-1
    )                                                              # [B, 1, N]
    true_theta = torch.zeros(B, 1, device=device)
    true_next = torch.zeros(B, 1, device=device)

    configs = {
        "1. routing=None (legacy path)":        (None, None),
        "2. all-focused (fb_energy mult=0.0)":  (
            torch.tensor([[1., 0.]] * B, device=device),
            train_cfg.task_routing,
        ),
        "3. all-routine (fb_energy mult=2.0)":  (
            torch.tensor([[0., 1.]] * B, device=device),
            train_cfg.task_routing,
        ),
        "4. mixed 50/50":                       (
            torch.tensor([[1., 0.]] * (B // 2) + [[0., 1.]] * (B // 2),
                         device=device),
            train_cfg.task_routing,
        ),
    }

    print(f"{'config':42s} {'fb_energy':>12s} {'expected':>12s}")
    print("-" * 68)
    expected = {
        "1. routing=None (legacy path)":       1.0,
        "2. all-focused (fb_energy mult=0.0)": 0.0,
        "3. all-routine (fb_energy mult=2.0)": 2.0,
        "4. mixed 50/50":                      1.0,
    }
    for name, (ts, tr) in configs.items():
        _, ld = loss_fn(
            outputs, true_theta, true_next,
            r_l23_windows, q_pred_windows,
            fb_scale=1.0,
            task_state=ts,
            task_routing=tr,
        )
        fe = ld["fb_energy"]
        exp = expected[name]
        marker = "PASS" if abs(fe - exp) < 1e-5 else "FAIL"
        print(f"{name:42s} {fe:>12.6f} {exp:>12.4f}  {marker}")

    print()
    print("Interpretation:")
    print("  1. Legacy path should give 1.0 (unweighted mean of |center_exc|)")
    print("  2. all-focused: fb_energy mult = 0.0 → per-sample weight = 0")
    print("     → mean(0 * ones) = 0.0  (proves routing IS applied)")
    print("  3. all-routine: fb_energy mult = 2.0 → mean(2 * ones) = 2.0")
    print("  4. mixed: mean of 0's and 2's = (0*4 + 2*4)/8 = 1.0")
    print()
    print("If (2) != 0 or (3) != 2 or (4) != 1, the routing is broken.")


if __name__ == "__main__":
    main()
