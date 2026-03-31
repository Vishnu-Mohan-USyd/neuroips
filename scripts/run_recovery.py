#!/usr/bin/env python3
"""Run model recovery / sensitivity analysis.

Generates synthetic ground-truth responses for dampening, sharpening, and
center-surround mechanisms, then verifies the analysis pipeline correctly
identifies each planted mechanism.

This is a hard gate: must pass before the full mechanism comparison sweep.

Usage:
    python -m scripts.run_recovery
    python -m scripts.run_recovery --mechanism dampening
    python -m scripts.run_recovery --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MechanismType
from src.analysis.model_recovery import run_recovery, MECHANISM_TO_FIT_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model recovery analysis")
    parser.add_argument("--mechanism", type=str, default=None,
                        choices=["dampening", "sharpening", "center_surround"],
                        help="Run recovery for a single mechanism (default: all 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write results report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mechanisms = [
        MechanismType.DAMPENING,
        MechanismType.SHARPENING,
        MechanismType.CENTER_SURROUND,
    ]
    if args.mechanism:
        mechanisms = [MechanismType(args.mechanism)]

    all_passed = True
    report_lines = ["# Model Recovery Report\n"]

    for mech in mechanisms:
        logger.info(f"{'='*60}")
        logger.info(f"Running recovery for: {mech.value}")
        logger.info(f"{'='*60}")

        result = run_recovery(mech, seed=args.seed)

        # Report suppression profile
        logger.info(f"Suppression profile (first 5 bins):")
        for i in range(min(5, len(result.profile.delta_theta))):
            logger.info(
                f"  Δθ={result.profile.delta_theta[i]:.1f}°: "
                f"supp={result.profile.suppression[i]:.4f}, "
                f"surp={result.profile.surprise[i]:.4f}"
            )

        # Report fit results
        logger.info(f"Parametric fits:")
        for fit in result.fit_results:
            logger.info(f"  {fit.model_name}: R²={fit.r_squared:.4f}")

        logger.info(f"Identified mechanism: {result.identified_mechanism}")
        expected = MECHANISM_TO_FIT_MODEL.get(mech, "N/A")
        logger.info(f"Expected: {expected}")
        logger.info(f"Correct: {result.correctly_identified}")

        if not result.correctly_identified:
            logger.warning(f"RECOVERY FAILED for {mech.value}!")
            all_passed = False

        # Report MVPA results
        logger.info(f"Observation model results:")
        for key, mvpa in result.voxel_results.items():
            logger.info(f"  {key}: 2-way={mvpa['acc_2way']:.3f}, 3-way={mvpa['acc_3way']:.3f}")

        # Build report
        report_lines.append(f"\n## {mech.value}\n")
        report_lines.append(f"- Identified: {result.identified_mechanism}")
        report_lines.append(f"- Expected: {expected}")
        report_lines.append(f"- Correct: {result.correctly_identified}")
        report_lines.append(f"\nParametric fits:")
        for fit in result.fit_results:
            report_lines.append(f"- {fit.model_name}: R²={fit.r_squared:.4f}")
        report_lines.append(f"\nMVPA results:")
        for key, mvpa in result.voxel_results.items():
            report_lines.append(f"- {key}: 2-way={mvpa['acc_2way']:.3f}, 3-way={mvpa['acc_3way']:.3f}")

    logger.info(f"\n{'='*60}")
    if all_passed:
        logger.info("ALL MECHANISMS CORRECTLY RECOVERED. Gate PASSED.")
    else:
        logger.error("RECOVERY GATE FAILED. Analysis pipeline needs revision.")
    logger.info(f"{'='*60}")

    # Write report if requested
    if args.output:
        report_lines.append(f"\n## Summary\n")
        report_lines.append(f"Gate: {'PASSED' if all_passed else 'FAILED'}")
        Path(args.output).write_text("\n".join(report_lines))
        logger.info(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
