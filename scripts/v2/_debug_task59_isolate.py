"""Task #59 — causal-isolation experiment for the shrink-dominated explosion.

Hypothesis: Plasticity explosion is caused by the energy shrinkage term
`-β·mean_b(pre²)·w` in `EnergyPenalty.current_weight_shrinkage`, NOT by the
Vogels Hebb term nor by bookkeeping accumulation.

Experimental test: Run the exact same Phase-2 stateful config with
`beta_syn=0.0` (shrinkage disabled) and compare explosion timing vs the
default (`beta_syn=1e-4`). If shrinkage is the cause, the β=0 run should
NOT explode. If it still explodes, shrinkage is not the root cause.
"""
from __future__ import annotations

from scripts.v2._debug_task59_explosion import run_instrumented, print_run_summary, \
    identify_first_explosive_rule, print_weight_evolution


def main() -> None:
    print("\n\n" + "#" * 76)
    print("# CAUSAL-ISOLATION: beta_syn=0.0 (shrinkage disabled)")
    print("#" * 76)
    snaps = run_instrumented(
        seed=42, n_steps=300, batch_size=4, warmup_steps=30,
        segment_length=50, soft_reset_scale=0.1, beta_syn=0.0,
    )
    print_run_summary("beta_syn=0", snaps)
    identify_first_explosive_rule(snaps)
    print_weight_evolution("beta_syn=0", snaps)


if __name__ == "__main__":
    main()
