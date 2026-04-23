"""
Task #49 Claim 2 — raw-softplus weight-decay anti-shrinkage test.

Is the weight-decay term `- weight_decay * weights` shrinkage or anti-shrinkage
for an *initially negative* raw weight (typical excitatory init at -5.85)?

Experiment:
    raw_0 = -5.85
    100-step update with only decay active (Urbanczik hebb = 0):
        raw_{t+1} = raw_t + (lr * 0 - weight_decay * raw_t)
                  = raw_t * (1 - weight_decay)

Record:
    raw trajectory at t in {0, 10, 50, 100}
    softplus(raw) trajectory
    Effective excitatory weight = +softplus(raw)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def main() -> None:
    # Default weight_decay values from a few plasticity call sites. Check both.
    wd_values = [1e-3, 1e-2, 1e-1]
    raw_init = -5.85

    print("=== Claim 2: raw-softplus decay anti-shrinkage probe ===")
    print(f"raw_0={raw_init}, softplus(raw_0)={F.softplus(torch.tensor(raw_init)).item():.4e}")
    print()

    for wd in wd_values:
        raw = torch.tensor(raw_init, dtype=torch.float64)
        print(f"--- weight_decay={wd} ---")
        print(f"  {'step':>5}  {'raw':>12}  {'softplus(raw)':>14}  {'eff_w':>10}")
        for t in range(101):
            if t in (0, 1, 5, 10, 25, 50, 100):
                sp = F.softplus(raw).item()
                print(f"  {t:>5}  {raw.item():>12.6f}  {sp:>14.4e}  {sp:>10.4e}")
            # Decay-only update: hebb contribution = 0 (no error), only - wd * raw.
            raw = raw - wd * raw  # = raw * (1 - wd)
        print()

    # What about with a non-zero hebb term that would happen with ε~0 under
    # zero pre-activity? Nothing — at blank frames with zero rates,
    # hebb = mean_batch_outer(eps, pre) = 0, so decay alone drives raw.
    # But eps is NON-zero at init (r_l4_next - x_hat_prev). So let's check
    # under a realistic Phase-2 baseline: hebb is small but non-zero.
    print("=== With realistic hebb scale (|hebb| ~ 1e-3 * sign) ===")
    lr = 1e-4
    wd = 1e-3
    for hebb_sign in (+1, -1):
        raw = torch.tensor(raw_init, dtype=torch.float64)
        print(f"--- hebb_sign={hebb_sign:+d}, lr={lr}, wd={wd} ---")
        print(f"  {'step':>5}  {'raw':>12}  {'softplus(raw)':>14}")
        hebb_mag = 1e-3
        hebb = torch.tensor(hebb_sign * hebb_mag, dtype=torch.float64)
        for t in range(1001):
            if t in (0, 10, 100, 500, 1000):
                sp = F.softplus(raw).item()
                print(f"  {t:>5}  {raw.item():>12.6f}  {sp:>14.4e}")
            dw = lr * hebb - wd * raw
            raw = raw + dw
        print()


if __name__ == "__main__":
    main()
