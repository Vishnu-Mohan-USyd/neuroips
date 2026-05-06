#!/usr/bin/env python3
"""Run the scaled-down V1 L4/L2/3 SNN."""

from __future__ import annotations

import argparse
from pathlib import Path

from v1_snn.config import load_config
from v1_snn.model import V1TwoLayerSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = V1TwoLayerSNN(config=config)
    result = model.run_protocol()
    output_dir = Path(args.output_dir)
    result.save(output_dir)
    print(f"saved results to {output_dir}")
    print(f"device={model.device}")
    for name, rate in result.rates.items():
        print(f"{name}: mean_rate_hz={rate.mean().item():.3f}")
    for name, osi in result.osi.items():
        print(f"{name}: mean_osi={osi.mean().item():.3f}")


if __name__ == "__main__":
    main()
