#!/usr/bin/env python3
"""Reproduce the Candidate-6 sharpening and dampening profiles from the
checkpoints in this repository.

Self-contained: every path is resolved relative to this file, and the only
external requirement is PyTorch (CPU is enough) and Matplotlib.

    python3 reproduce_figures.py

Writes figures/c6_<phenotype>_seed<N>.png and figures/c6_curves.json, then
checks the seed-8 numbers against the values banked in BANKED below and exits
non-zero if any of them has moved.
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
# orientation. Centre is the presented channel and its two neighbours; the
# flanks are the 15-30 degree ring on both sides.
CENTER_OFFSETS = (-1, 0, 1)
FLANK_OFFSETS = (-6, -5, -4, -3, 3, 4, 5, 6)
PLOT_OFFSETS = tuple(range(-12, 13))

ARMS = [
    ("alpha0p0", "0p0", "task optimized", "sharpening", "#2b7bb9"),
    ("alpha0p5", "0p5", "energy optimized", "dampening", "#cf3232"),
]
SEEDS = (8, 9, 10)

# Values produced by this script on the banked seed-8 checkpoints. Any drift
# means the model code and these checkpoints have come apart.
BANKED = {
    ("8", "0p0"): {"center_ratio": 1.2764, "flank_ratio": 0.7188,
                   "peak": 1.5969, "peak_at_deg": 0.0},
    ("8", "0p5"): {"center_ratio": 0.1007, "flank_ratio": 0.7063,
                   "peak": 0.3914, "peak_at_deg": 15.0},
}
TOL = 1e-3


def build_paths() -> None:
    sys.path.insert(0, str(HERE / "harness"))
    sys.path.insert(1, str(HERE / "tools"))


def main() -> int:
    build_paths()
    import simple_net as simple           # noqa: E402
    import tuned_emergence_lib as tuned   # noqa: E402

    device = torch.device("cpu")
    simple.device = device
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
                print(f"  skip: {ckpath} absent")
                continue

            ck = torch.load(ckpath, map_location=device, weights_only=False)
            net = tuned.build_tuned_from_config(dict(ck["tuned_net_config"]))
            net = net.to(device)
            _, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
            assert not unexpected, unexpected
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
            base_curve = base.mean(dim=0)[idx].cpu().tolist()
            peak = max(adapt_curve)
            metrics = {
                "seed": seed,
                "alpha": aslug.replace("p", "."),
                "phenotype": kind,
                "checkpoint": str(ckpath.relative_to(HERE)),
                "curves_offsets_deg": deg,
                "curve_baseline_t0": base_curve,
                "curve_adapted": adapt_curve,
                "center_ratio": bin_mean(adapted, CENTER_OFFSETS)
                / bin_mean(base, CENTER_OFFSETS),
                "flank_ratio": bin_mean(adapted, FLANK_OFFSETS)
                / bin_mean(base, FLANK_OFFSETS),
                "peak": peak,
                "peak_at_deg": deg[adapt_curve.index(peak)],
                "baseline_peak": max(base_curve),
            }
            dump[f"seed{seed}_{aslug}"] = metrics

            png = FIGS / f"c6_{kind}_seed{seed}.png"
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(deg, base_curve, color="0.45", linestyle=(0, (4, 3)),
                    linewidth=1.0, marker="o", markersize=2.6,
                    markeredgewidth=0, label="baseline")
            ax.plot(deg, adapt_curve, color=colour, linewidth=1.2, marker="o",
                    markersize=2.6, markeredgewidth=0, label="expected")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ymax = max(max(base_curve), max(adapt_curve))
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
            ax.set_xlabel("probe offset (deg)", fontsize=9, color="0.3")
            ax.set_ylabel("mean rate", fontsize=9, color="0.3")
            ax.set_title(f"{phenotype} — seed {seed}", fontsize=11, loc="left",
                         color="0.15", pad=12)
            leg = ax.legend(frameon=False, fontsize=9, loc="upper right",
                            handlelength=1.6, borderpad=0, labelspacing=0.35)
            for txt in leg.get_texts():
                txt.set_color("0.3")
            fig.tight_layout()
            fig.savefig(png, dpi=200)
            plt.close(fig)

            print(f"  seed {seed:<3} alpha {aslug:<4} {kind:<11}"
                  f" centre {metrics['center_ratio']:.4f}"
                  f" flank {metrics['flank_ratio']:.4f}"
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

    (FIGS / "c6_curves.json").write_text(json.dumps(dump, indent=1))

    print()
    if failures:
        print("REGRESSION — banked seed-8 values did not reproduce:")
        for f in failures:
            print("   ", f)
        return 1
    print("Seed-8 values reproduce the banked figures within 1e-3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
