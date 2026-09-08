#!/usr/bin/env python3
"""Reproduce the current projected-output split-SST checkpoints.

Self-contained: every path is resolved relative to this file, and the only
external requirement is PyTorch (CPU is enough) and Matplotlib.

    python3 reproduce_figures.py

Writes individual and seed-mean raw response curves as PNG/SVG plus
figures/c6_curves.json. The mean comparison uses archived v8 curve data for
its original-sharpening reference. Checks the seed-8 values banked in BANKED
below and exits non-zero if a checkpoint is missing, incompatible, or fails
reproduction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
FIGS = HERE / "figures"

# Response bins, in units of 5-degree channels away from the presented
# orientation. The historical center_ratio averages -5, 0, +5 degrees;
# preferred_ratio separately measures the exact presented channel (0 degrees).
# Flanks are the 15-30 degree ring on both sides.
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
PLOT_OFFSETS = tuple(range(-12, 13))

ARMS = [
    ("alpha0p07", "0p07", "Sharpening", "sharpening", "#2b7bb9"),
    ("alpha0p7", "0p7", "Dampening", "dampening", "#cf3232"),
]
SEEDS = (8, 9, 10)

# Values produced by this script on the banked seed-8 checkpoints. Any drift
# means the model code and these checkpoints have come apart.
BANKED = {
    ("8", "0p07"): {"center_ratio": 0.8596, "flank_ratio": 0.8966,
                     "peak": 1.5841, "peak_at_deg": 0.0},
    ("8", "0p7"): {"center_ratio": 0.0790, "flank_ratio": 0.4921,
                    "peak": 0.2554, "peak_at_deg": 20.0},
}
TOL = 1e-3


def build_paths() -> None:
    sys.path.insert(0, str(HERE / "harness"))
    sys.path.insert(1, str(HERE / "tools"))


def plot_seed_mean(curves: dict[str, dict]) -> None:
    """Render the shared-axis comparison from the same checkpoint responses."""
    archived = json.loads(
        (HERE / "archive/v8/figures/c6_curves.json").read_text()
    )

    def mean_curve(slug: str, field: str) -> list[float]:
        return torch.tensor(
            [curves[f"seed{seed}_{slug}"][field] for seed in SEEDS],
            dtype=torch.float64,
        ).mean(dim=0).tolist()

    original = torch.tensor(
        [archived[f"seed{seed}_0p05"]["curve_adapted"] for seed in SEEDS],
        dtype=torch.float64,
    ).mean(dim=0).tolist()
    degrees = curves[f"seed{SEEDS[0]}_{ARMS[0][1]}"]["curves_offsets_deg"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ymax = max(original)
    for ax, (_, slug, phenotype, kind, _) in zip(axes, ARMS):
        first = mean_curve(slug, "curve_baseline_t0")
        unexpected = mean_curve(slug, "curve_unexpected")
        expected = mean_curve(slug, "curve_adapted")
        ymax = max(ymax, max(first), max(unexpected), max(expected))
        ax.plot(degrees, first, color="#777777", linestyle="--",
                linewidth=1.8, label="First response")
        if kind == "sharpening":
            ax.plot(degrees, original, color="#82b4d7", linestyle=":",
                    linewidth=2, label="Expected — original")
        ax.plot(degrees, unexpected, color="#dd8b3a", linewidth=1.7,
                label="Unexpected")
        ax.plot(degrees, expected,
                color="#236da8" if kind == "sharpening" else "#b64944",
                linewidth=2.3, label="Expected")
        alpha = float(slug.replace("p", "."))
        ax.set(title=f"{phenotype} · α = {alpha:.2f}", xlim=(-60, 60),
               xlabel="Orientation offset (°)")
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Activity (a.u.)")
    axes[0].set_ylim(0, ymax * 1.12)
    fig.tight_layout()
    fig.savefig(FIGS / "c6_shared_task_energy_seed_mean.png", dpi=180)
    fig.savefig(FIGS / "c6_shared_task_energy_seed_mean.svg")
    plt.close(fig)


def main() -> int:
    build_paths()
    import simple_net as simple           # noqa: E402
    import tuned_emergence_lib as tuned   # noqa: E402

    device = torch.device("cpu")
    simple.device = device
    simple.prefs = torch.arange(simple.N, device=device).float() * simple.STEP_DEG
    tuned.device = device
    torch.set_num_threads(2)

    import assay_emergent_task_energy_axis as assay  # noqa: E402
    assert assay.tuned is tuned, "assay must bind the harness library"

    FIGS.mkdir(exist_ok=True)
    dump: dict[str, dict] = {}
    failures: list[str] = []

    for seed in SEEDS:
        for armdir, aslug, phenotype, kind, colour in ARMS:
            ckpath = CKPT / f"seed{seed}" / armdir / f"alpha_{aslug}_final.pt"
            if not ckpath.exists():
                failures.append(f"{ckpath}: required checkpoint is missing")
                continue

            ck = torch.load(ckpath, map_location=device, weights_only=False)
            if ck.get("model_architecture_version") != tuned.MODEL_ARCHITECTURE_VERSION:
                failures.append(
                    f"{ckpath}: checkpoint architecture does not match current tuned circuit"
                )
                continue
            net = tuned.build_tuned_from_config(dict(ck["tuned_net_config"]))
            net = net.to(device)
            net.load_state_dict(ck["state_dict"], strict=True)
            net.eval()
            net.ref_rate.fill_(float(ck["references"]["R_ref"]))
            cf = bool(ck.get("center_feedback", False))
            fm = tuned.resolve_feedback_mode(cf, ck.get("feedback_mode"))

            with torch.no_grad():
                theta_a, theta_b, finals = assay.matched_pairs(device)
                _, ra = tuned.forward_seq_tuned(net, theta_a, 1.0,
                                                center_feedback=cf,
                                                feedback_mode=fm)
                _, rb = tuned.forward_seq_tuned(net, theta_b, 1.0,
                                                center_feedback=cf,
                                                feedback_mode=fm)
                adapted = assay.align_rates(ra[:, -1, :], finals).to(torch.float64)
                unexpected = assay.align_rates(rb[:, -1, :], finals).to(torch.float64)
                fa = (theta_a[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
                fb = (theta_b[:, 0] / assay.STEP_DEG).round().to(torch.long) % assay.N
                base = 0.5 * (assay.align_rates(ra[:, 0, :], fa).to(torch.float64)
                              + assay.align_rates(rb[:, 0, :], fb).to(torch.float64))

            idx = [assay.OFFSETS.index(o) for o in PLOT_OFFSETS]

            def bin_mean(curve, offs):
                j = [assay.OFFSETS.index(o) for o in offs]
                return float(curve[:, j].mean().item())

            deg = [o * assay.STEP_DEG for o in PLOT_OFFSETS]
            adapt_curve = adapted.mean(dim=0)[idx].cpu().tolist()
            unexpected_curve = unexpected.mean(dim=0)[idx].cpu().tolist()
            base_curve = base.mean(dim=0)[idx].cpu().tolist()
            expected_mean = float(adapted.mean().item())
            unexpected_mean = float(unexpected.mean().item())
            peak = max(adapt_curve)
            metrics = {
                "seed": seed,
                "alpha": aslug.replace("p", "."),
                "phenotype": kind,
                "checkpoint": str(ckpath.relative_to(HERE)),
                "curves_offsets_deg": deg,
                "curve_baseline_t0": base_curve,
                "curve_adapted": adapt_curve,
                "curve_unexpected": unexpected_curve,
                "preferred_ratio": bin_mean(adapted, (0,)) / bin_mean(base, (0,)),
                "shoulder_5deg_ratio": bin_mean(adapted, (-1, 1))
                / bin_mean(base, (-1, 1)),
                "flank_15deg_ratio": bin_mean(adapted, (-3, 3))
                / bin_mean(base, (-3, 3)),
                "center_ratio": bin_mean(adapted, CENTER_OFFSETS)
                / bin_mean(base, CENTER_OFFSETS),
                "flank_ratio": bin_mean(adapted, FLANK_OFFSETS)
                / bin_mean(base, FLANK_OFFSETS),
                "peak": peak,
                "peak_at_deg": deg[adapt_curve.index(peak)],
                "baseline_peak": max(base_curve),
                "expected_mean_rate": expected_mean,
                "unexpected_mean_rate": unexpected_mean,
                "expectation_suppression_percent": 100.0
                * (1.0 - expected_mean / unexpected_mean),
            }
            dump[f"seed{seed}_{aslug}"] = metrics

            png = FIGS / f"c6_{kind}_seed{seed}.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(deg, base_curve, color="0.45", linestyle=(0, (4, 3)),
                    linewidth=1.0, marker="o", markersize=2.6,
                    markeredgewidth=0, label="first response")
            ax.plot(deg, unexpected_curve, color="#d39a35", linestyle=":",
                    linewidth=1.2, label="unexpected")
            ax.plot(deg, adapt_curve, color=colour, linewidth=1.2, marker="o",
                    markersize=2.6, markeredgewidth=0, label="expected")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ymax = max(max(base_curve), max(adapt_curve), max(unexpected_curve))
            yticks = [t for t in ax.get_yticks() if 0.0 <= t <= ymax]
            xticks = [-60, -30, 0, 30, 60]
            ax.set_yticks(yticks)
            ax.set_xticks(xticks)
            ax.set_ylim(min(0.0, min(adapt_curve)), ymax * 1.06)
            ax.set_xlim(min(deg), max(deg))
            for side, bounds in (("left", (min(yticks), ymax)),
                                 ("bottom", (min(xticks), max(xticks)))):
                ax.spines[side].set_bounds(*bounds)
                ax.spines[side].set_position(("outward", 8))
                ax.spines[side].set_linewidth(0.8)
                ax.spines[side].set_color("0.3")
            ax.tick_params(direction="out", length=3, width=0.8, colors="0.3",
                           labelsize=9)
            ax.set_xlabel("Preferred orientation − stimulus (deg)", fontsize=9, color="0.3")
            ax.set_ylabel("Mean E activity (a.u.)", fontsize=9, color="0.3")
            ax.set_title(f"{phenotype} — seed {seed}", fontsize=11, loc="left",
                         color="0.15", pad=12)
            leg = ax.legend(frameon=False, fontsize=9, loc="upper right",
                            handlelength=1.6, borderpad=0, labelspacing=0.35)
            for txt in leg.get_texts():
                txt.set_color("0.3")
            fig.tight_layout()
            fig.savefig(png, dpi=200)
            fig.savefig(png.with_suffix(".svg"))
            plt.close(fig)

            print(f"  seed {seed:<3} alpha {aslug:<4} {kind:<11}"
                  f" center(0°) {metrics['preferred_ratio']:.4f}"
                  f" flank {metrics['flank_ratio']:.4f}"
                  f" ES {metrics['expectation_suppression_percent']:.2f}%"
                  f" peak {peak:.4f} at {metrics['peak_at_deg']:+.0f} deg"
                  f"  -> {png.name}")

            key = (str(seed), aslug)
            if key in BANKED:
                for field, want in BANKED[key].items():
                    got = metrics[field]
                    if abs(got - want) > TOL:
                        failures.append(
                            f"seed {seed} alpha {aslug} {field}: "
                            f"got {got:.6f}, banked {want:.6f}")

    print()
    if failures:
        print("REGRESSION — banked seed-8 values did not reproduce:")
        for f in failures:
            print("   ", f)
        return 1
    (FIGS / "c6_curves.json").write_text(json.dumps(dump, indent=1))
    plot_seed_mean(dump)
    print("Seed-8 values reproduce the banked figures within 1e-3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
