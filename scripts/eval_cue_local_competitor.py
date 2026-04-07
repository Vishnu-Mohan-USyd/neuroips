#!/usr/bin/env python3
"""Summarize cue-validity effects from cue-local-competitor experiments.

The script prefers existing ``cue_local_competitor`` artifacts. If an artifact
is unavailable, it can run the paradigm directly from a checkpoint/config pair
using the existing experiment code path. The summary stays intentionally small:

- per-condition prestimulus VIP and SOM means;
- per-condition sustained L2/3 peak;
- cue-kind aggregated match-vs-competitor peak gaps;
- valid-neutral and invalid-neutral gap deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_representation import load_model
from src.experiments.cue_local_competitor import CueLocalCompetitorParadigm
from src.experiments.paradigm_base import ExperimentResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_cue_local_competitor_artifact(path: str) -> ExperimentResult:
    """Load a saved cue-local-competitor artifact and return its result object."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "cue_local_competitor" in payload:
        result = payload["cue_local_competitor"]
    elif isinstance(payload, ExperimentResult):
        result = payload
    else:
        raise ValueError(f"{path} does not contain a cue_local_competitor result")
    if result.paradigm_name != "cue_local_competitor":
        raise ValueError(f"{path} contains paradigm={result.paradigm_name}, expected cue_local_competitor")
    return result


def run_cue_local_competitor(
    checkpoint: str,
    config: str,
    device: torch.device,
    n_trials: int,
    seed: int,
    batch_size: int,
    competitor_offset: float,
    cue_contrast: float,
) -> ExperimentResult:
    """Run the cue-local-competitor paradigm directly from a checkpoint."""
    net, model_cfg, _ = load_model(checkpoint, config, device)
    paradigm = CueLocalCompetitorParadigm(
        net,
        model_cfg,
        competitor_offset=competitor_offset,
        cue_contrast=cue_contrast,
    )
    result = paradigm.run(n_trials=n_trials, seed=seed, batch_size=batch_size)
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize_cue_validity(result: ExperimentResult) -> dict:
    """Compute compact cue-validity summaries from one experiment result.

    The summary is designed for publication hardening rather than exhaustive
    descriptive analysis. It keeps the metric surface close to the current
    screening discussion:

    - prestimulus VIP mean
    - prestimulus SOM mean
    - sustained L2/3 peak mean
    - sustained match-vs-competitor gap per cue validity
    """
    cue_start, cue_end = result.trial_info.get(
        "cue_window", result.temporal_windows["prestimulus"]
    )
    sustain_start, sustain_end = result.temporal_windows["sustained"]

    condition_summary: dict[str, dict] = {}
    for name, cond in result.conditions.items():
        rule, cue_kind, probe_kind = name.split("_")
        vip_mean = 0.0
        if cond.r_vip is not None:
            vip_mean = float(cond.r_vip[:, cue_start:cue_end].mean())
        condition_summary[name] = {
            "rule": rule,
            "cue_kind": cue_kind,
            "probe_kind": probe_kind,
            "prestim_vip_mean": vip_mean,
            "prestim_som_mean": float(cond.r_som[:, cue_start:cue_end].mean()),
            "sustained_peak_mean": float(cond.r_l23[:, sustain_start:sustain_end].amax(dim=(1, 2)).mean()),
        }

    cue_kind_summary: dict[str, dict] = {}
    for cue_kind in ["valid", "neutral", "invalid"]:
        cue_conditions = [
            summary for summary in condition_summary.values()
            if summary["cue_kind"] == cue_kind
        ]
        peak_gaps: list[float] = []
        for rule in ["cw", "ccw"]:
            match_key = f"{rule}_{cue_kind}_match"
            competitor_key = f"{rule}_{cue_kind}_competitor"
            peak_gaps.append(
                condition_summary[match_key]["sustained_peak_mean"]
                - condition_summary[competitor_key]["sustained_peak_mean"]
            )
        cue_kind_summary[cue_kind] = {
            "prestim_vip_mean": float(sum(c["prestim_vip_mean"] for c in cue_conditions) / len(cue_conditions)),
            "prestim_som_mean": float(sum(c["prestim_som_mean"] for c in cue_conditions) / len(cue_conditions)),
            "sustained_peak_gap": float(sum(peak_gaps) / len(peak_gaps)),
        }

    return {
        "paradigm_name": result.paradigm_name,
        "trial_info": dict(result.trial_info),
        "temporal_windows": dict(result.temporal_windows),
        "condition_summary": condition_summary,
        "cue_kind_summary": cue_kind_summary,
        "valid_neutral_gap_delta": (
            cue_kind_summary["valid"]["sustained_peak_gap"]
            - cue_kind_summary["neutral"]["sustained_peak_gap"]
        ),
        "invalid_neutral_gap_delta": (
            cue_kind_summary["invalid"]["sustained_peak_gap"]
            - cue_kind_summary["neutral"]["sustained_peak_gap"]
        ),
    }


def _write_csv(output_csv: Path, summary: dict) -> None:
    """Write flat CSV rows for per-condition and per-cue-kind summaries."""
    rows: list[dict[str, object]] = []
    for name, payload in summary["condition_summary"].items():
        rows.append({"scope": "condition", "name": name, **payload})
    for name, payload in summary["cue_kind_summary"].items():
        rows.append({"scope": "cue_kind", "name": name, **payload})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scope",
                "name",
                "rule",
                "cue_kind",
                "probe_kind",
                "prestim_vip_mean",
                "prestim_som_mean",
                "sustained_peak_mean",
                "sustained_peak_gap",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cue-validity summaries."""
    parser = argparse.ArgumentParser(description="Cue-local-competitor validity summary")
    parser.add_argument("--artifact", type=str, default=None, help="Existing cue_local_competitor .pt artifact")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint for on-demand generation")
    parser.add_argument("--config", type=str, default=None, help="Config path for on-demand generation")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--competitor-offset", type=float, default=10.0)
    parser.add_argument("--cue-contrast", type=float, default=1.0)
    parser.add_argument("--save-artifact", type=str, default=None, help="Optional path to save a generated artifact")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Load or run cue-local-competitor and emit a compact JSON/CSV summary."""
    args = parse_args()
    if args.artifact is None and (args.checkpoint is None or args.config is None):
        raise ValueError("provide either --artifact or both --checkpoint and --config")

    if args.artifact is not None:
        result = load_cue_local_competitor_artifact(args.artifact)
    else:
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        result = run_cue_local_competitor(
            checkpoint=args.checkpoint,
            config=args.config,
            device=device,
            n_trials=args.n_trials,
            seed=args.seed,
            batch_size=args.batch_size,
            competitor_offset=args.competitor_offset,
            cue_contrast=args.cue_contrast,
        )
        if args.save_artifact:
            save_path = Path(args.save_artifact)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({result.paradigm_name: result}, save_path)

    summary = summarize_cue_validity(result)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output_csv, summary)
    logger.info("Saved JSON to %s", output_json)
    logger.info("Saved CSV to %s", output_csv)


if __name__ == "__main__":
    main()
