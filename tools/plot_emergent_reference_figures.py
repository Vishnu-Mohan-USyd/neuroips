#!/usr/bin/env python3
"""Plot measured task–energy endpoints in four reference figure layouts.

The script reuses the fixed 216-pair assay and checkpoint loader from
``assay_emergent_task_energy_axis.py``. For each seed and endpoint it replays
feedback-on operational continuation A and reversal B histories. Colored
tuning curves are final A responses; the gray comparator pools ordinary-unroll
timestep-zero A/B responses aligned to their own first presented orientation,
where prior feedback state is naturally zero. It is not reversal B.

Decoding uses the assay's condition-blind, noise-held-out 36-class cosine
centroid protocol; histories are not held out. Tuning first averages 216 rows
within a seed. All plotted uncertainty is sample SEM across four independent
seeds, never pair-level variability. The script remeasures checkpoints but does
not apply a phenotype acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import assay_emergent_task_energy_axis as assay  # noqa: E402
import tuned_emergence_lib as tuned  # noqa: E402


ENERGY_COLOR = "#4C72B0"
TASK_COLOR = "#C44E52"
BASE_COLOR = "#888888"
PLOT_OFFSETS = tuple(range(-12, 13))
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
EXPECTED_RUN_DIRS = 4


@dataclass(frozen=True)
class SeedArmMeasurement:
    """One seed's endpoint measurements for one optimization arm.

    Curves contain raw L2/3 rates averaged over the 216 matched sequences. The
    expected and unexpected curves are aligned to the shared final orientation;
    the zero-context curve pools A/B timestep-zero responses aligned to each
    sequence's presented first orientation. Scalar rate quantities are in the
    same arbitrary rate units; decoding quantities are fractions in [0, 1].
    """

    seed: int
    run_dir: Path
    checkpoint: Path
    alpha: float
    feedback_mode: str
    expected_curve: torch.Tensor
    unexpected_curve: torch.Tensor
    zero_context_curve: torch.Tensor
    expected_center: float
    expected_flank: float
    unexpected_center: float
    unexpected_flank: float
    zero_context_center: float
    zero_context_flank: float
    delta_fq: float
    delta_q: float
    decode_expected: float
    decode_unexpected: float
    mean_rate_expected: float
    mean_rate_unexpected: float
    phase_y: float
    stored_saving: float

    @property
    def expected_center_flank_ratio(self) -> float:
        return self.expected_center / self.expected_flank

    @property
    def unexpected_center_flank_ratio(self) -> float:
        return self.unexpected_center / self.unexpected_flank

    @property
    def zero_context_center_flank_ratio(self) -> float:
        return self.zero_context_center / self.zero_context_flank

    @property
    def decode_delta(self) -> float:
        return self.decode_expected - self.decode_unexpected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Seed directory containing common_pretrain_final.pt and requested "
            "alpha finals. Repeat exactly four times for distinct seeds."
        ),
    )
    parser.add_argument(
        "--task-alpha",
        type=float,
        default=0.0,
        help="Low-rate-pressure endpoint used for task/sharpening displays.",
    )
    parser.add_argument(
        "--energy-alpha",
        type=float,
        default=0.9,
        help="High-rate-pressure endpoint used for energy/attenuation displays.",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="PyTorch replay device, for example cuda:0 or cpu."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures" / "emergent_reference_comparison",
        help=(
            "Directory for plot_data.json, tuning_dampening.png, "
            "tuning_sharpening.png, 1_decode_signflip.png, and "
            "3_decode_energy_phasespace.png."
        ),
    )
    args = parser.parse_args()
    if len(args.run_dir) != EXPECTED_RUN_DIRS:
        parser.error(
            f"--run-dir must be supplied exactly {EXPECTED_RUN_DIRS} times"
        )
    resolved_dirs = [path.expanduser().resolve() for path in args.run_dir]
    if len(set(resolved_dirs)) != len(resolved_dirs):
        parser.error("--run-dir values must be distinct")
    if math.isclose(args.task_alpha, args.energy_alpha, abs_tol=1e-12):
        parser.error("task and energy alpha values must be distinct")
    args.run_dir = resolved_dirs
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def mean_sem(values: Sequence[float]) -> dict[str, float | int]:
    """Return arithmetic mean and sample SEM across independent seeds."""

    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim != 1 or tensor.numel() < 2:
        raise ValueError("seed SEM requires at least two scalar observations")
    if not torch.isfinite(tensor).all():
        raise ValueError("cannot summarize non-finite values")
    return {
        "mean": float(tensor.mean().item()),
        "sem": float(
            (tensor.std(unbiased=True) / math.sqrt(tensor.numel())).item()
        ),
        "n_seeds": int(tensor.numel()),
    }


def curve_mean_sem(curves: Sequence[torch.Tensor]) -> tuple[list[float], list[float]]:
    """Aggregate seed-mean curves and return pointwise mean and seed SEM."""

    stack = torch.stack(
        [curve.detach().to(dtype=torch.float64, device="cpu") for curve in curves]
    )
    if stack.ndim != 2 or stack.shape[0] < 2:
        raise ValueError("curve SEM requires at least two seed curves")
    if stack.shape[1] != len(PLOT_OFFSETS):
        raise ValueError(
            f"expected {len(PLOT_OFFSETS)} plotted offsets, got {stack.shape[1]}"
        )
    if not torch.isfinite(stack).all():
        raise ValueError("cannot summarize non-finite curves")
    mean = stack.mean(dim=0)
    sem = stack.std(dim=0, unbiased=True) / math.sqrt(stack.shape[0])
    return mean.tolist(), sem.tolist()


def center_flank(curve: Sequence[float]) -> tuple[float, float, float]:
    """Compute C, F, and C/F using the locked displayed orientation bins."""

    if len(curve) != len(PLOT_OFFSETS):
        raise ValueError("center/flank calculation requires the displayed curve")
    values = {offset: float(value) for offset, value in zip(PLOT_OFFSETS, curve)}
    center = sum(values[offset] for offset in CENTER_OFFSETS) / len(CENTER_OFFSETS)
    flank = sum(values[offset] for offset in FLANK_OFFSETS) / len(FLANK_OFFSETS)
    if not flank > 0.0:
        raise ValueError("C/F requires positive mean flank activity")
    return center, flank, center / flank


def validate_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    expected_alpha: float,
    common_local_comp_raw: torch.Tensor,
) -> tuple[int, str]:
    """Fail fast when inputs are not the validated posterior-excess endpoints."""

    required = {
        "seed",
        "alpha",
        "step",
        "target_steps",
        "state_dict",
        "tuned_net_config",
        "references",
        "freeze_local_comp",
        "feedback_mode",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"{checkpoint_path} lacks keys: {sorted(missing)}")
    if not math.isclose(
        float(checkpoint["alpha"]), expected_alpha, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{checkpoint_path} alpha {checkpoint['alpha']} != {expected_alpha}"
        )
    if int(checkpoint["step"]) != int(checkpoint["target_steps"]):
        raise ValueError(f"{checkpoint_path} is not a completed endpoint")
    if not bool(checkpoint["freeze_local_comp"]):
        raise ValueError(f"{checkpoint_path} did not freeze local competition")
    local_comp_raw = checkpoint["state_dict"].get("local_comp_strength_raw")
    if local_comp_raw is None or not torch.equal(
        local_comp_raw, common_local_comp_raw
    ):
        raise ValueError(
            f"{checkpoint_path} local competition differs from common pretrain"
        )
    feedback_mode = tuned.resolve_feedback_mode(
        bool(checkpoint.get("center_feedback", False)),
        checkpoint.get("feedback_mode"),
    )
    if feedback_mode != tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS:
        raise ValueError(
            f"{checkpoint_path} uses feedback mode {feedback_mode!r}, expected "
            f"{tuned.FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS!r}"
        )
    return int(checkpoint["seed"]), feedback_mode


@torch.inference_mode()
def measure_seed_arm(
    run_dir: Path,
    alpha: float,
    device: torch.device,
    common_local_comp_raw: torch.Tensor,
) -> SeedArmMeasurement:
    """Measure one arm using the assay's exact matched-pair construction."""

    checkpoint_path = run_dir / f"alpha_{assay.alpha_slug(alpha)}_final.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    net, checkpoint = assay.load_arm(checkpoint_path, device)
    seed, feedback_mode = validate_checkpoint(
        checkpoint, checkpoint_path, alpha, common_local_comp_raw
    )

    theta_a, theta_b, finals = assay.matched_pairs(device)
    _, rates_a_all = tuned.forward_seq_tuned(
        net,
        theta_a,
        1.0,
        center_feedback=bool(checkpoint.get("center_feedback", False)),
        feedback_mode=feedback_mode,
    )
    _, rates_b_all = tuned.forward_seq_tuned(
        net,
        theta_b,
        1.0,
        center_feedback=bool(checkpoint.get("center_feedback", False)),
        feedback_mode=feedback_mode,
    )
    rates_a = rates_a_all[:, -1, :]
    rates_b = rates_b_all[:, -1, :]
    expected_shape = (assay.N * len(assay.VELOCITIES), assay.N)
    if tuple(rates_a.shape) != expected_shape or tuple(rates_b.shape) != expected_shape:
        raise RuntimeError(
            f"expected matched final-rate shape {expected_shape}, got "
            f"A={tuple(rates_a.shape)}, B={tuple(rates_b.shape)}"
        )

    aligned_a = assay.align_rates(rates_a, finals).to(torch.float64)
    aligned_b = assay.align_rates(rates_b, finals).to(torch.float64)
    first_channels_a = (
        (theta_a[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    )
    first_channels_b = (
        (theta_b[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
    )
    aligned_zero_context_a = assay.align_rates(
        rates_a_all[:, 0, :], first_channels_a
    ).to(torch.float64)
    aligned_zero_context_b = assay.align_rates(
        rates_b_all[:, 0, :], first_channels_b
    ).to(torch.float64)
    offset_to_index = {
        offset: index for index, offset in enumerate(assay.OFFSETS)
    }
    plot_indices = torch.tensor(
        [offset_to_index[offset] for offset in PLOT_OFFSETS],
        dtype=torch.long,
        device=device,
    )
    expected_curve = aligned_a.index_select(1, plot_indices).mean(dim=0)
    unexpected_curve = aligned_b.index_select(1, plot_indices).mean(dim=0)
    zero_context_curve = 0.5 * (
        aligned_zero_context_a.index_select(1, plot_indices).mean(dim=0)
        + aligned_zero_context_b.index_select(1, plot_indices).mean(dim=0)
    )
    exp_center, exp_flank, _ = center_flank(expected_curve.cpu().tolist())
    unx_center, unx_flank, _ = center_flank(unexpected_curve.cpu().tolist())
    zero_center, zero_flank, _ = center_flank(
        zero_context_curve.cpu().tolist()
    )

    r_ref = float(checkpoint["references"]["R_ref"])
    sigma = float(checkpoint["references"]["sigma_train"])
    shape_a = assay.shape_quantities(rates_a, finals, r_ref)
    shape_b = assay.shape_quantities(rates_b, finals, r_ref)
    decoding = assay.condition_blind_held_out_decoding(
        rates_a, rates_b, finals, sigma=sigma
    )

    mean_rate_a = rates_a.to(torch.float64).mean()
    mean_rate_b = rates_b.to(torch.float64).mean()
    epsilon = 1e-8 * assay.N * r_ref
    phase_y = (mean_rate_a - mean_rate_b) / (mean_rate_b + epsilon)
    stored_saving = (mean_rate_b - mean_rate_a) / (mean_rate_b + epsilon)
    if not torch.allclose(phase_y, -stored_saving, atol=1e-12, rtol=1e-12):
        raise RuntimeError("phase-space rate metric is not negative stored saving")

    return SeedArmMeasurement(
        seed=seed,
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        alpha=alpha,
        feedback_mode=feedback_mode,
        expected_curve=expected_curve.cpu(),
        unexpected_curve=unexpected_curve.cpu(),
        zero_context_curve=zero_context_curve.cpu(),
        expected_center=exp_center,
        expected_flank=exp_flank,
        unexpected_center=unx_center,
        unexpected_flank=unx_flank,
        zero_context_center=zero_center,
        zero_context_flank=zero_flank,
        delta_fq=float((shape_a["Fq"] - shape_b["Fq"]).mean().item()),
        delta_q=float((shape_a["Q"] - shape_b["Q"]).mean().item()),
        decode_expected=float(
            decoding["expected_A_held_out_top1_accuracy"]
        ),
        decode_unexpected=float(
            decoding["unexpected_B_held_out_top1_accuracy"]
        ),
        mean_rate_expected=float(mean_rate_a.item()),
        mean_rate_unexpected=float(mean_rate_b.item()),
        phase_y=float(phase_y.item()),
        stored_saving=float(stored_saving.item()),
    )


def load_common_local_comp(run_dir: Path, device: torch.device) -> tuple[int, torch.Tensor]:
    """Load the seed id and frozen local-competition parameter from pretraining."""

    common_path = run_dir / "common_pretrain_final.pt"
    if not common_path.is_file():
        raise FileNotFoundError(common_path)
    checkpoint = torch.load(common_path, map_location=device)
    if "seed" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"{common_path} lacks seed/state_dict metadata")
    local_comp_raw = checkpoint["state_dict"].get("local_comp_strength_raw")
    if local_comp_raw is None:
        raise ValueError(f"{common_path} lacks local_comp_strength_raw")
    return int(checkpoint["seed"]), local_comp_raw


def measurement_json(measurement: SeedArmMeasurement) -> dict[str, Any]:
    """Convert one measurement into a concise, JSON-safe record."""

    return {
        "seed": measurement.seed,
        "run_dir": str(measurement.run_dir),
        "checkpoint": str(measurement.checkpoint),
        "feedback_mode": measurement.feedback_mode,
        "tuning": {
            "first_stimulus_zero_context_center_C": (
                measurement.zero_context_center
            ),
            "first_stimulus_zero_context_flank_F": (
                measurement.zero_context_flank
            ),
            "first_stimulus_zero_context_C_over_F": (
                measurement.zero_context_center_flank_ratio
            ),
            "expected_center_C": measurement.expected_center,
            "expected_flank_F": measurement.expected_flank,
            "expected_C_over_F": measurement.expected_center_flank_ratio,
            "unexpected_center_C": measurement.unexpected_center,
            "unexpected_flank_F": measurement.unexpected_flank,
            "unexpected_C_over_F": measurement.unexpected_center_flank_ratio,
            "delta_C_expected_minus_zero_context": (
                measurement.expected_center - measurement.zero_context_center
            ),
            "delta_F_expected_minus_zero_context": (
                measurement.expected_flank - measurement.zero_context_flank
            ),
            "delta_C_expected_minus_unexpected": (
                measurement.expected_center - measurement.unexpected_center
            ),
            "delta_F_expected_minus_unexpected": (
                measurement.expected_flank - measurement.unexpected_flank
            ),
            "delta_Fq_expected_minus_unexpected": measurement.delta_fq,
            "delta_Q_expected_minus_unexpected": measurement.delta_q,
        },
        "decoding": {
            "expected_A_accuracy": measurement.decode_expected,
            "unexpected_B_accuracy": measurement.decode_unexpected,
            "delta_accuracy_A_minus_B": measurement.decode_delta,
        },
        "rates": {
            "expected_A_mean": measurement.mean_rate_expected,
            "unexpected_B_mean": measurement.mean_rate_unexpected,
            "phase_y_A_minus_B_over_B": measurement.phase_y,
            "stored_saving_B_minus_A_over_B": measurement.stored_saving,
        },
    }


def aggregate_arm(measurements: Sequence[SeedArmMeasurement]) -> dict[str, Any]:
    """Aggregate one arm over seed-level observations only."""

    if len(measurements) != EXPECTED_RUN_DIRS:
        raise ValueError(f"expected {EXPECTED_RUN_DIRS} seed measurements")
    expected_mean, expected_sem = curve_mean_sem(
        [measurement.expected_curve for measurement in measurements]
    )
    unexpected_mean, unexpected_sem = curve_mean_sem(
        [measurement.unexpected_curve for measurement in measurements]
    )
    zero_context_mean, zero_context_sem = curve_mean_sem(
        [measurement.zero_context_curve for measurement in measurements]
    )
    exp_c, exp_f, exp_ratio = center_flank(expected_mean)
    unx_c, unx_f, unx_ratio = center_flank(unexpected_mean)
    zero_c, zero_f, zero_ratio = center_flank(zero_context_mean)

    scalar_fields = {
        "expected_C": [m.expected_center for m in measurements],
        "expected_F": [m.expected_flank for m in measurements],
        "expected_C_over_F": [
            m.expected_center_flank_ratio for m in measurements
        ],
        "unexpected_C": [m.unexpected_center for m in measurements],
        "unexpected_F": [m.unexpected_flank for m in measurements],
        "unexpected_C_over_F": [
            m.unexpected_center_flank_ratio for m in measurements
        ],
        "first_stimulus_zero_context_C": [
            m.zero_context_center for m in measurements
        ],
        "first_stimulus_zero_context_F": [
            m.zero_context_flank for m in measurements
        ],
        "first_stimulus_zero_context_C_over_F": [
            m.zero_context_center_flank_ratio for m in measurements
        ],
        "displayed_delta_C_expected_minus_zero_context": [
            m.expected_center - m.zero_context_center for m in measurements
        ],
        "displayed_delta_F_expected_minus_zero_context": [
            m.expected_flank - m.zero_context_flank for m in measurements
        ],
        "delta_C": [
            m.expected_center - m.unexpected_center for m in measurements
        ],
        "delta_F": [
            m.expected_flank - m.unexpected_flank for m in measurements
        ],
        "delta_Fq": [m.delta_fq for m in measurements],
        "delta_Q": [m.delta_q for m in measurements],
    }
    return {
        "alpha": measurements[0].alpha,
        "seed_ids": [measurement.seed for measurement in measurements],
        "per_seed": [measurement_json(m) for m in measurements],
        "tuning_curve": {
            "offset_deg": [offset * assay.STEP_DEG for offset in PLOT_OFFSETS],
            "expected_mean": expected_mean,
            "expected_sem": expected_sem,
            "unexpected_mean": unexpected_mean,
            "unexpected_sem": unexpected_sem,
            "first_stimulus_zero_context_mean": zero_context_mean,
            "first_stimulus_zero_context_sem": zero_context_sem,
            "displayed_mean_curve_C_F": {
                "expected_C": exp_c,
                "expected_F": exp_f,
                "expected_C_over_F": exp_ratio,
                "first_stimulus_zero_context_C": zero_c,
                "first_stimulus_zero_context_F": zero_f,
                "first_stimulus_zero_context_C_over_F": zero_ratio,
            },
            "assay_expected_unexpected_mean_curve_C_F": {
                "expected_C": exp_c,
                "expected_F": exp_f,
                "expected_C_over_F": exp_ratio,
                "unexpected_C": unx_c,
                "unexpected_F": unx_f,
                "unexpected_C_over_F": unx_ratio,
            },
        },
        "shape_seed_summary": {
            name: mean_sem(values) for name, values in scalar_fields.items()
        },
        "decoding": {
            "expected_A_accuracy": mean_sem(
                [m.decode_expected for m in measurements]
            ),
            "unexpected_B_accuracy": mean_sem(
                [m.decode_unexpected for m in measurements]
            ),
            "delta_accuracy_A_minus_B": mean_sem(
                [m.decode_delta for m in measurements]
            ),
        },
        "phase_space": {
            "x_delta_accuracy_A_minus_B": mean_sem(
                [m.decode_delta for m in measurements]
            ),
            "y_mean_rate_A_minus_B_over_B": mean_sem(
                [m.phase_y for m in measurements]
            ),
            "stored_saving_B_minus_A_over_B": mean_sem(
                [m.stored_saving for m in measurements]
            ),
        },
    }


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def plot_tuning(
    aggregate: dict[str, Any],
    output_path: Path,
    *,
    arm_label: str,
    color: str,
    shared_ymax: float,
) -> None:
    curve = aggregate["tuning_curve"]
    ratios = curve["displayed_mean_curve_C_F"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(
        curve["offset_deg"],
        curve["first_stimulus_zero_context_mean"],
        yerr=curve["first_stimulus_zero_context_sem"],
        fmt="-o",
        ms=3,
        lw=1.2,
        color=BASE_COLOR,
        ecolor=BASE_COLOR,
        elinewidth=0.8,
        capsize=2,
        alpha=0.9,
        label="First stimulus (no prior context; feedback state = 0)",
    )
    ax.errorbar(
        curve["offset_deg"],
        curve["expected_mean"],
        yerr=curve["expected_sem"],
        fmt="-o",
        ms=3,
        lw=1.2,
        color=color,
        ecolor=color,
        elinewidth=0.8,
        capsize=2,
        alpha=0.95,
        label=(
            f"Expected after {arm_label} optimization "
            f"(α={aggregate['alpha']:g})"
        ),
    )
    ax.axvline(0.0, linestyle=":", color="black", linewidth=0.8)
    ax.set_xlabel(
        "nominal fixed feedforward orientation preference relative to "
        "presented orientation (°)"
    )
    ax.set_ylabel("Mean L2/3 response (a.u.)")
    ax.set_ylim(0.0, shared_ymax)
    ax.set_title(
        f"{arm_label.capitalize()}-optimized endpoint "
        f"(mean ± seed SEM, n=4; C/F: first = "
        f"{ratios['first_stimulus_zero_context_C_over_F']:.3f}, "
        f"expected = {ratios['expected_C_over_F']:.3f})"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_decoding(
    energy: dict[str, Any], task: dict[str, Any], output_path: Path
) -> None:
    groups = (energy, task)
    colors = (ENERGY_COLOR, TASK_COLOR)
    labels = ("Energy optimized", "Task optimized")
    expected = [group["decoding"]["expected_A_accuracy"]["mean"] for group in groups]
    unexpected = [
        group["decoding"]["unexpected_B_accuracy"]["mean"] for group in groups
    ]
    expected_sem = [
        group["decoding"]["expected_A_accuracy"]["sem"] for group in groups
    ]
    unexpected_sem = [
        group["decoding"]["unexpected_B_accuracy"]["sem"] for group in groups
    ]
    x = torch.arange(2, dtype=torch.float64).tolist()
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(
        [value - width / 2 for value in x],
        expected,
        width,
        yerr=expected_sem,
        capsize=4,
        color=colors,
        error_kw={"linewidth": 1.2, "ecolor": "#444444"},
    )
    ax.bar(
        [value + width / 2 for value in x],
        unexpected,
        width,
        yerr=unexpected_sem,
        capsize=4,
        color=colors,
        alpha=0.45,
        error_kw={"linewidth": 1.2, "ecolor": "#444444"},
    )
    chance = 1.0 / assay.N
    ax.axhline(chance, linestyle=":", color="gray", linewidth=1.0)
    ax.annotate(
        "chance (1/36)",
        (1.18, chance + 0.008),
        color="gray",
        fontsize=9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Orientation decoding accuracy")
    observed_top = max(
        value + error
        for value, error in zip(
            expected + unexpected, expected_sem + unexpected_sem
        )
    )
    ax.set_ylim(0.0, min(1.0, max(0.20, observed_top * 1.18)))
    ax.set_title(
        "Condition-blind held-out decoding (mean ± seed SEM, n=4)"
    )
    ax.legend(
        handles=[
            Patch(facecolor="#555555", label="Expected (feedback on)"),
            Patch(
                facecolor="#555555",
                alpha=0.45,
                label="Unexpected (matched; feedback on)",
            ),
        ],
        frameon=False,
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def padded_limits(
    centers: Sequence[float], errors: Sequence[float], *, minimum_span: float
) -> tuple[float, float]:
    low = min([0.0] + [value - error for value, error in zip(centers, errors)])
    high = max([0.0] + [value + error for value, error in zip(centers, errors)])
    span = max(high - low, minimum_span)
    return low - 0.18 * span, high + 0.18 * span


def plot_phase_space(
    energy: dict[str, Any], task: dict[str, Any], output_path: Path
) -> None:
    points = (
        ("Energy optimized", energy, ENERGY_COLOR),
        ("Task optimized", task, TASK_COLOR),
    )
    xs = [point[1]["phase_space"]["x_delta_accuracy_A_minus_B"]["mean"] for point in points]
    xerrs = [point[1]["phase_space"]["x_delta_accuracy_A_minus_B"]["sem"] for point in points]
    ys = [point[1]["phase_space"]["y_mean_rate_A_minus_B_over_B"]["mean"] for point in points]
    yerrs = [point[1]["phase_space"]["y_mean_rate_A_minus_B_over_B"]["sem"] for point in points]

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    for (name, _, color), x, xerr, y, yerr in zip(
        points, xs, xerrs, ys, yerrs
    ):
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="#555555",
            linewidth=1.2,
            capsize=3,
            zorder=2,
        )
        ax.scatter(x, y, s=240, color=color, edgecolor="black", zorder=3)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(10, 8),
            fontweight="bold",
        )
    ax.set_xlim(*padded_limits(xs, xerrs, minimum_span=0.10))
    ax.set_ylim(*padded_limits(ys, yerrs, minimum_span=0.10))
    ax.set_xlabel("Δ decode accuracy (expected−unexpected)")
    ax.set_ylabel(
        "Δ mean L2/3 rate (expected−unexpected, fraction)"
    )
    ax.set_title("Task–energy endpoint phase space (mean ± seed SEM, n=4)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    device = assay.choose_device(args.device)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    arms: dict[str, list[SeedArmMeasurement]] = {
        "task_optimized": [],
        "energy_optimized": [],
    }
    seen_seeds: set[int] = set()
    for run_dir in args.run_dir:
        common_seed, common_local_comp_raw = load_common_local_comp(run_dir, device)
        task = measure_seed_arm(
            run_dir, args.task_alpha, device, common_local_comp_raw
        )
        energy = measure_seed_arm(
            run_dir, args.energy_alpha, device, common_local_comp_raw
        )
        if task.seed != common_seed or energy.seed != common_seed:
            raise ValueError(f"seed metadata disagree within {run_dir}")
        if common_seed in seen_seeds:
            raise ValueError(f"duplicate seed {common_seed} in input run dirs")
        seen_seeds.add(common_seed)
        arms["task_optimized"].append(task)
        arms["energy_optimized"].append(energy)
        print(
            json.dumps(
                {
                    "status": "measured_seed",
                    "seed": common_seed,
                    "run_dir": str(run_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for measurements in arms.values():
        measurements.sort(key=lambda measurement: measurement.seed)
    baseline_reference = arms["task_optimized"][0].zero_context_curve
    for arm_name, measurements in arms.items():
        for measurement in measurements:
            if not torch.equal(
                measurement.zero_context_curve, baseline_reference
            ):
                raise RuntimeError(
                    "independently recomputed zero-context baselines differ: "
                    f"arm={arm_name}, seed={measurement.seed}"
                )
    task_aggregate = aggregate_arm(arms["task_optimized"])
    energy_aggregate = aggregate_arm(arms["energy_optimized"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "tuning_dampening": args.out_dir / "tuning_dampening.png",
        "tuning_sharpening": args.out_dir / "tuning_sharpening.png",
        "decode_signflip": args.out_dir / "1_decode_signflip.png",
        "decode_energy_phasespace": (
            args.out_dir / "3_decode_energy_phasespace.png"
        ),
        "plot_data": args.out_dir / "plot_data.json",
    }
    configure_plot_style()
    all_curve_upper = []
    for aggregate in (energy_aggregate, task_aggregate):
        curve = aggregate["tuning_curve"]
        all_curve_upper.extend(
            mean + sem
            for mean, sem in zip(curve["expected_mean"], curve["expected_sem"])
        )
        all_curve_upper.extend(
            mean + sem
            for mean, sem in zip(
                curve["first_stimulus_zero_context_mean"],
                curve["first_stimulus_zero_context_sem"],
            )
        )
    shared_ymax = max(all_curve_upper) * 1.16
    if not shared_ymax > 0.0:
        raise ValueError("tuning curves must contain positive activity")

    plot_tuning(
        energy_aggregate,
        output_paths["tuning_dampening"],
        arm_label="energy",
        color=ENERGY_COLOR,
        shared_ymax=shared_ymax,
    )
    plot_tuning(
        task_aggregate,
        output_paths["tuning_sharpening"],
        arm_label="task",
        color=TASK_COLOR,
        shared_ymax=shared_ymax,
    )
    plot_decoding(
        energy_aggregate, task_aggregate, output_paths["decode_signflip"]
    )
    plot_phase_space(
        energy_aggregate,
        task_aggregate,
        output_paths["decode_energy_phasespace"],
    )

    payload = {
        "metadata": {
            "script": str(Path(__file__).resolve()),
            "device": str(device),
            "seed_count": len(seen_seeds),
            "seed_ids": sorted(seen_seeds),
            "matched_pair_count_per_seed_condition": (
                assay.N * len(assay.VELOCITIES)
            ),
            "condition_A": "operational_continuation_A (code label: expected)",
            "condition_B": "operational_OOD_reversal_B (code label: unexpected)",
            "feedback": "on",
            "plotter_role": (
                "checkpoint remeasurement and presentation; no phenotype "
                "acceptance gate"
            ),
            "error_bars": (
                "sample standard deviation across four independent seed-level "
                "values divided by sqrt(4); tuning first averages 216 rows "
                "within each seed"
            ),
            "tuning_baseline": {
                "kind": "literal_first_stimulus_no_context",
                "source": (
                    "pooled_balanced_A_B_t0_aligned_to_own_first_orientation"
                ),
                "feedback_execution": (
                    "normal_feedback_on_unroll_with_naturally_zero_pred_down_at_t0"
                ),
                "alignment_reference": "presented_orientation_at_same_timepoint",
                "offset_degrees": [
                    offset * assay.STEP_DEG for offset in PLOT_OFFSETS
                ],
                "equality_check": (
                    "bit-identical across all four seeds and both endpoint arms"
                ),
            },
            "colored_tuning_curve": {
                "kind": "operational_continuation_A_final_contextual_response",
                "alignment_reference": "matched_final_orientation",
                "quantity": "raw L2/3 rates",
                "offset_range_deg": [-60.0, 60.0],
                "offset_step_deg": assay.STEP_DEG,
                "center_C_deg": [-5.0, 0.0, 5.0],
                "flank_F_deg": [
                    -30.0,
                    -25.0,
                    -20.0,
                    -15.0,
                    15.0,
                    20.0,
                    25.0,
                    30.0,
                ],
                "C_over_F": "mean(center_C_deg) / mean(flank_F_deg)",
            },
            "decoding": {
                "classifier": (
                    "condition-blind balanced pooled A+B cosine nearest-centroid"
                ),
                "held_out_scope": (
                    "independent additive noise tables only; histories and "
                    "stimuli are not held out"
                ),
                "classes": assay.N,
            },
            "endpoint_interpretation": {
                "task": "center enhancement with modest flank suppression",
                "energy": (
                    "broad attenuation with preferential center suppression; "
                    "absolute flanks are not preserved versus t0"
                ),
            },
            "phase_x": "expected_A_accuracy - unexpected_B_accuracy",
            "phase_y": (
                "(expected_A_mean_rate - unexpected_B_mean_rate) / "
                "unexpected_B_mean_rate; numerically negative stored saving"
            ),
            "chance_accuracy": 1.0 / assay.N,
            "output_files": {
                name: path.relative_to(args.out_dir).as_posix()
                for name, path in output_paths.items()
            },
        },
        "energy_optimized": energy_aggregate,
        "task_optimized": task_aggregate,
    }
    atomic_json_save(payload, output_paths["plot_data"])
    print(
        json.dumps(
            {
                "status": "complete",
                "output_files": {
                    name: str(path) for name, path in output_paths.items()
                },
                "energy_phase": energy_aggregate["phase_space"],
                "task_phase": task_aggregate["phase_space"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
