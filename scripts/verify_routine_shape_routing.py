"""Unit-level verification: does CompositeLoss route the Phase 2.4 routine_shape term?

Purely synthetic — no network, no stimulus, no training. Constructs a minimal
`outputs` dict with *known* ``center_exc`` and ``som_drive_fb`` magnitudes, then
calls CompositeLoss.forward() under four routing configurations. Checks that
``loss_dict["routine_shape"]`` scales correctly with the per-sample task_state
mapped through the sweep_dual_2_4 task_routing (focused: 0.0, routine: 2.0).

Analytic baseline (what CompositeLoss.forward should compute):
    # per-sample (with routing off), shape_per_sample is a scalar constant
    # because center_exc and som_drive_fb are constants across (B, T, N):
    shape_per_sample = |center_exc|.mean() - 0.5 * |som_drive_fb|.mean()

With ``center_exc = 0.04`` and ``som_drive_fb = 0.02`` everywhere:
    baseline_shape = 0.04 - 0.5 * 0.02 = 0.030

Expected ``loss_dict["routine_shape"]`` per configuration:
    routing=None  (legacy global-mean fallback): shape_per_sample.mean() = 0.030
    all-focused  (w=[0]*B):                       0.0
    all-routine  (w=[2]*B):                       2.0 * 0.030 = 0.060
    mixed 50/50  (half [0], half [2]):            1.0 * 0.030 = 0.030

If any actual value diverges from expected beyond 1e-6, routing is broken.

A secondary "sign" check verifies the gradient structure: the loss should
DECREASE when ``center_exc`` drops OR ``som_drive_fb`` rises on routine
samples. We run two extra forward passes with modified outputs and assert the
expected sign of Δloss.
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
    cfg_path = "config/sweep/sweep_dual_2_4.yaml"
    model_cfg, train_cfg, stim_cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"lambda_routine_shape  = {train_cfg.lambda_routine_shape}")
    print(f"lr_mult_alpha         = {train_cfg.lr_mult_alpha}")
    print(f"task_routing[focused] = {train_cfg.task_routing['focused']}")
    print(f"task_routing[routine] = {train_cfg.task_routing['routine']}")
    print()

    B, W, N = 8, 4, model_cfg.n_orientations
    T = 12
    torch.manual_seed(42)

    # Known-magnitude synthetic outputs:
    #   |center_exc|    = 0.04 everywhere  → per-sample mean = 0.04
    #   |som_drive_fb|  = 0.02 everywhere  → per-sample mean = 0.02
    #   → shape_per_sample = 0.04 - 0.5*0.02 = 0.030
    ce_mag = 0.04
    sdf_mag = 0.02
    outputs = {
        "r_l4":          torch.zeros(B, T, N, device=device),
        "r_l23":         torch.zeros(B, T, N, device=device),
        "r_pv":          torch.zeros(B, T, 1, device=device),
        "r_som":         torch.zeros(B, T, N, device=device),
        "deep_template": torch.zeros(B, T, N, device=device),
        "center_exc":    torch.full((B, T, N), ce_mag, device=device),
        "som_drive_fb":  torch.full((B, T, N), sdf_mag, device=device),
    }

    # Loss-side inputs that are irrelevant to routine_shape but required by
    # CompositeLoss.forward signature.
    r_l23_windows = torch.ones(B, W, N, device=device)
    q_pred_windows = torch.full((B, W, N), 1.0 / N, device=device)
    true_theta = torch.zeros(B, W, device=device)
    true_next = torch.zeros(B, W, device=device)

    loss_fn = CompositeLoss(train_cfg, model_cfg).to(device)

    baseline_shape = ce_mag - 0.5 * sdf_mag  # 0.030

    tr = train_cfg.task_routing
    f_shape = float(tr["focused"].get("routine_shape", 0.0))  # 0.0
    r_shape = float(tr["routine"].get("routine_shape", 0.0))  # 2.0

    configs = {
        "1. routing=None (legacy)":    (None,      None,        baseline_shape),
        "2. all-focused  [1,0]*B":     (torch.tensor([[1., 0.]] * B, device=device), tr, 0.0),
        "3. all-routine  [0,1]*B":     (torch.tensor([[0., 1.]] * B, device=device), tr, r_shape * baseline_shape),
        "4. mixed 50/50":              (
            torch.tensor([[1., 0.]] * (B // 2) + [[0., 1.]] * (B // 2), device=device),
            tr,
            0.5 * (f_shape * baseline_shape + r_shape * baseline_shape),
        ),
    }

    print(f"Expected routine_shape values (baseline_shape = {baseline_shape:.6f}):")
    for name, (_, _, expected) in configs.items():
        print(f"  {name:30s} → {expected:.6f}")
    print()

    print(f"{'config':30s}  {'actual':>12s}  {'expected':>12s}  verdict")
    print("-" * 72)
    any_fail = False
    actuals = {}
    for name, (ts, tr_arg, expected) in configs.items():
        _, ld = loss_fn(
            outputs, true_theta, true_next,
            r_l23_windows, q_pred_windows,
            fb_scale=1.0,
            task_state=ts,
            task_routing=tr_arg,
        )
        actual = ld["routine_shape"]
        actuals[name] = actual
        ok = abs(actual - expected) < 1e-6
        if not ok:
            any_fail = True
        marker = "PASS" if ok else "FAIL"
        print(f"{name:30s}  {actual:>12.6f}  {expected:>12.6f}  {marker}")
    print()

    # ------------------------------------------------------------------
    # Sign checks: verify the gradient structure of the loss.
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Sign checks (Δ routine_shape under known perturbations):")
    print("=" * 72)

    all_routine_ts = torch.tensor([[0., 1.]] * B, device=device)

    # (a) Lower center_exc → loss should DECREASE.
    outputs_low_ce = {**outputs, "center_exc": torch.full((B, T, N), ce_mag * 0.5, device=device)}
    _, ld_low_ce = loss_fn(
        outputs_low_ce, true_theta, true_next,
        r_l23_windows, q_pred_windows,
        fb_scale=1.0,
        task_state=all_routine_ts,
        task_routing=tr,
    )
    delta_low_ce = ld_low_ce["routine_shape"] - actuals["3. all-routine  [0,1]*B"]
    expected_delta_low_ce = r_shape * (-0.5 * ce_mag)   # 2.0 * (-0.02) = -0.04
    ok_sign_a = delta_low_ce < 0 and abs(delta_low_ce - expected_delta_low_ce) < 1e-6
    print(f"  lower center_exc  → Δ = {delta_low_ce:+.6f}  "
          f"(expected {expected_delta_low_ce:+.6f}) "
          f"{'PASS' if ok_sign_a else 'FAIL'}")

    # (b) Raise som_drive_fb → loss should DECREASE.
    outputs_high_sdf = {**outputs, "som_drive_fb": torch.full((B, T, N), sdf_mag * 3.0, device=device)}
    _, ld_high_sdf = loss_fn(
        outputs_high_sdf, true_theta, true_next,
        r_l23_windows, q_pred_windows,
        fb_scale=1.0,
        task_state=all_routine_ts,
        task_routing=tr,
    )
    delta_high_sdf = ld_high_sdf["routine_shape"] - actuals["3. all-routine  [0,1]*B"]
    # Δ per-sample shape = -0.5 * (sdf_new - sdf_old) = -0.5 * (0.06 - 0.02) = -0.02
    # Routed: -0.04 at w_routine_shape = 2.0
    expected_delta_high_sdf = r_shape * (-0.5 * (sdf_mag * 3.0 - sdf_mag))
    ok_sign_b = delta_high_sdf < 0 and abs(delta_high_sdf - expected_delta_high_sdf) < 1e-6
    print(f"  higher som_drive  → Δ = {delta_high_sdf:+.6f}  "
          f"(expected {expected_delta_high_sdf:+.6f}) "
          f"{'PASS' if ok_sign_b else 'FAIL'}")

    if not (ok_sign_a and ok_sign_b):
        any_fail = True

    # ------------------------------------------------------------------
    # Gradient check: autograd flows into center_exc / som_drive_fb with
    # the correct signs.
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Autograd check (gradient flows to center_exc / som_drive_fb):")
    print("=" * 72)
    ce_leaf = torch.full((B, T, N), ce_mag, device=device, requires_grad=True)
    sdf_leaf = torch.full((B, T, N), sdf_mag, device=device, requires_grad=True)
    outputs_grad = {**outputs, "center_exc": ce_leaf, "som_drive_fb": sdf_leaf}
    total, ld = loss_fn(
        outputs_grad, true_theta, true_next,
        r_l23_windows, q_pred_windows,
        fb_scale=1.0,
        task_state=all_routine_ts,
        task_routing=tr,
    )
    total.backward()
    ce_grad_mean = ce_leaf.grad.mean().item()
    sdf_grad_mean = sdf_leaf.grad.mean().item()
    # Expected signs (ignoring the magnitude contribution from fb_energy on
    # center_exc which will make ce gradient even more positive):
    #   d(l_routine_shape)/d(ce) > 0  → descent shrinks ce  ✓
    #   d(l_routine_shape)/d(sdf) < 0 → descent grows sdf   ✓
    ok_ce = ce_grad_mean > 0
    ok_sdf = sdf_grad_mean < 0
    print(f"  grad center_exc.mean()    = {ce_grad_mean:+.6e}  (want > 0) {'PASS' if ok_ce else 'FAIL'}")
    print(f"  grad som_drive_fb.mean()  = {sdf_grad_mean:+.6e}  (want < 0) {'PASS' if ok_sdf else 'FAIL'}")
    if not (ok_ce and ok_sdf):
        any_fail = True

    print()
    if any_fail:
        print("RESULT: FAIL — routine_shape routing or gradient flow is broken.")
        sys.exit(1)
    else:
        print("RESULT: PASS — routine_shape routing + gradient flow verified.")


if __name__ == "__main__":
    main()
