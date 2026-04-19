"""Variant of _debug_phase2_divergence.py that freezes ThresholdHomeostasis.

Monkey-patches ThresholdHomeostasis.update to a no-op so `theta` stays at
its init value (0.0), then runs Phase-2 training and reports |eps|.
"""
from __future__ import annotations

import argparse

from src.v2_model.plasticity import ThresholdHomeostasis

# Freeze homeostasis BEFORE importing the debug harness or driver.
def _no_update(self, activity):  # type: ignore[no-redef]
    return None
ThresholdHomeostasis.update = _no_update  # type: ignore[assignment]

from scripts.v2._debug_phase2_divergence import run_with_probes, _fmt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-urbanczik", type=float, default=1e-3)
    ap.add_argument("--lr-vogels", type=float, default=5e-4)
    ap.add_argument("--lr-hebb", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--beta-syn", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--probe-every", type=int, default=1000)
    args = ap.parse_args()
    out = run_with_probes(
        lr_urbanczik=args.lr_urbanczik,
        n_steps=args.n_steps,
        probe_every=args.probe_every,
        beta_syn=args.beta_syn,
        lr_vogels=args.lr_vogels,
        lr_hebb=args.lr_hebb,
        weight_decay=args.weight_decay,
    )
    print(_fmt(out))


if __name__ == "__main__":
    main()
