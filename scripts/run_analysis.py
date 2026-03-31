#!/usr/bin/env python3
"""Load experiment data and run all analyses.

Usage:
    python -m scripts.run_analysis --experiments results.pt --output analysis_out.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.suppression_profile import compute_mean_responses, compute_suppression_profile_from_experiment
from src.analysis.energy import compute_energy
from src.analysis.temporal_analysis import run_temporal_analysis
from src.analysis.v2_probes import run_v2_probes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis suite")
    parser.add_argument("--experiments", type=str, required=True,
                        help="Path to experiment results (.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save analysis results (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading experiments from {args.experiments}")
    results = torch.load(args.experiments, map_location="cpu", weights_only=False)

    analysis_out: dict = {}

    # Run analyses on each paradigm
    for paradigm_name, result in results.items():
        logger.info(f"Analysing {paradigm_name}...")

        # Mean responses (Analysis 1)
        mean_resp = compute_mean_responses(result)
        analysis_out[f"{paradigm_name}_mean_responses"] = mean_resp

        # Energy (Analysis 7)
        energy = compute_energy(result)
        analysis_out[f"{paradigm_name}_energy"] = energy

        # Temporal (Analysis 11)
        temporal = run_temporal_analysis(result)
        analysis_out[f"{paradigm_name}_temporal"] = temporal

        # V2 probes (Analysis 12)
        v2 = run_v2_probes(result)
        analysis_out[f"{paradigm_name}_v2_probes"] = v2

        logger.info(f"  Energy: total={energy.total_activity:.4f}")

    if args.output:
        torch.save(analysis_out, args.output)
        logger.info(f"Analysis saved to {args.output}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
