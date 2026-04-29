"""Render task #54 Phase-A validation PNG artifacts (V1-V5) and run the
V4 phase-generalization classifier.

Inputs (JSON dumps from `v1_test --train-stdp ...`):
    /tmp/train_weight_snapshots.json
    /tmp/train_per_trial.json
    /tmp/v1_v2_phaseA_v1_rfs.json
    /tmp/v1_v2_phaseA_v2_osi.json
    /tmp/v1_v2_phaseA_v3_pi.json
    /tmp/v1_v2_phaseA_v4_decode.json
    /tmp/v1_v2_phaseA_v5_diag.json
    /tmp/v1_v2_phaseA_summary.json

Outputs (PNGs to /tmp; default --out path):
    train_weight_evolution.png
    v1_v2_phaseA_v1_rfs.png
    v1_v2_phaseA_v2_osi.png
    v1_v2_phaseA_v3_pi.png
    v1_v2_phaseA_v4_decode.png
    v1_v2_phaseA_v5_diag.png

Plus updates the V4 JSON with computed train/test accuracy.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID = 32
PI = math.pi
N_L4 = 32 * 32 * 8 * 16   # 131072
N_L23 = 32 * 32 * 16      # 16384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  default="/tmp")
    ap.add_argument("--out_dir", default="/tmp")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Train weight evolution ---
    snap_path = in_dir / "train_weight_snapshots.json"
    if snap_path.exists():
        snap = json.loads(snap_path.read_text())
        snaps = snap["snapshots"]
        trials = [s["trial"] for s in snaps]
        median = [s["median"] for s in snaps]
        mean   = [s["mean"]   for s in snaps]
        max_   = [s["max"]    for s in snaps]
        frac_z = [s["frac_at_zero"] for s in snaps]
        frac_c = [s["frac_at_cap"] for s in snaps]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        ax.plot(trials, mean,   "o-", label="mean")
        ax.plot(trials, median, "s-", label="median")
        ax.plot(trials, max_,   "^-", label="max")
        ax.set_xlabel("trial")
        ax.set_ylabel("weight (nS)")
        ax.set_title(
            f"Weight evolution (snapshots × {len(trials)})\n"
            f"A_plus={snap['stdp']['A_plus']}, "
            f"A_minus={snap['stdp']['A_minus']}, "
            f"τ⁺={snap['stdp']['tau_plus_ms']} ms, "
            f"τ⁻={snap['stdp']['tau_minus_ms']} ms",
            fontsize=9,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(trials, frac_z, "o-", color="C3", label="frac_at_zero")
        ax.plot(trials, frac_c, "s-", color="C2", label="frac_at_cap")
        ax.set_xlabel("trial")
        ax.set_ylabel("fraction")
        ax.set_title("Weights at bounds")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out / "train_weight_evolution.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out/'train_weight_evolution.png'}")

    # --- V1 RF locality ---
    v1_path = in_dir / "v1_v2_phaseA_v1_rfs.json"
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())
        sample_cells = v1["sample_cells"]
        sample_sta = v1["sample_cells_sta"]
        n_cells = len(sample_sta)

        fig, axes = plt.subplots(4, 4, figsize=(11, 11))
        axes = axes.flatten()
        for s, sc in enumerate(sample_sta):
            sta = np.asarray(sc["sta"], dtype=np.float64).reshape(GRID, GRID)
            ax = axes[s]
            ax.imshow(sta, origin="lower", cmap="viridis")
            ax.set_title(
                f"l23={sc['l23_idx']} (gx={sc['gx']},gy={sc['gy']})\n"
                f"total_spk={sc['total_spikes']}",
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            # Mark the cell's retinotopic position.
            ax.plot(sc['gx'], sc['gy'], 'r+', markersize=8, mew=1.5)
        for k in range(n_cells, len(axes)):
            axes[k].axis("off")
        fig.suptitle(
            "V1. Per-cell RF (small-Gabor-patch STA, "
            f"σ={v1.get('v1_sigma_px','?')} px aperture, "
            f"f={v1.get('v1_sf_cyc_per_px','?')} cyc/px, "
            f"{v1.get('v1_n_orientations','?')} orientations)\n"
            f"n_cells_with_rf={v1.get('n_cells_with_rf','?')}, "
            f"FWHM median={v1.get('fwhm_median_pixels','?')} px, "
            f"frac_connected={v1.get('frac_connected','?'):.3f}, "
            f"frac_peak_in_pool={v1.get('frac_peak_in_pool','?'):.3f}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out / "v1_v2_phaseA_v1_rfs.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out/'v1_v2_phaseA_v1_rfs.png'}")

    # --- V2 orientation tuning ---
    v2_path = in_dir / "v1_v2_phaseA_v2_osi.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text())
        osi = np.asarray(v2["osi_per_cell"], dtype=np.float64)
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        bins = np.linspace(0.0, 1.0, 41)
        ax.hist(osi, bins=bins, color="C0", edgecolor="0.4", alpha=0.85)
        ax.axvline(v2["osi_median"], color="k", lw=1.5,
                   label=f"median={v2['osi_median']:.3f}")
        ax.axvline(0.2, color="r", lw=1.0, ls="--",
                   label=f"OSI=0.2 (frac>0.2={v2['frac_osi_gt_0p2']:.3f})")
        ax.axvline(0.5, color="C2", lw=1.0, ls="--",
                   label=f"OSI=0.5 (frac>0.5={v2['frac_osi_gt_0p5']:.3f})")
        ax.set_xlabel("OSI = |Σ R(θ) e^(2iθ)| / Σ R(θ)")
        ax.set_ylabel("# L2/3 cells")
        ax.set_yscale("log")
        ax.set_title(
            f"V2. L2/3 orientation tuning (8 θ × 5 reps × 1s, post-training)\n"
            f"n_cells={len(osi)}",
            fontsize=11,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "v1_v2_phaseA_v2_osi.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out/'v1_v2_phaseA_v2_osi.png'}")

    # --- V3 phase invariance ---
    v3_path = in_dir / "v1_v2_phaseA_v3_pi.json"
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text())
        cells = v3["sample_cells"]
        l23_pis = [c["l23_pi"] for c in cells]
        l4_pis  = [c["l4_partner_pi_mean"] for c in cells]
        ratios  = [c["pi_ratio"] for c in cells]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.scatter(l4_pis, l23_pis, color="C0", s=40)
        max_lim = max(max(l4_pis + [0]), max(l23_pis + [0])) * 1.1 + 0.1
        ax.plot([0, max_lim], [0, max_lim], "k--", lw=0.8, label="y = x")
        for c in cells:
            ax.annotate(f"{c['l23_idx']}", (c["l4_partner_pi_mean"], c["l23_pi"]),
                        fontsize=6, alpha=0.6)
        ax.set_xlabel("L4-partner PI (mean over partners)")
        ax.set_ylabel("L2/3 PI")
        ax.set_xlim(0, max_lim)
        ax.set_ylim(0, max_lim)
        ax.set_title(
            f"V3. PI scatter (L23 vs L4-partner mean)\n"
            f"frac_ratio<1.0={v3['frac_pi_ratio_lt_1']:.3f}, "
            f"<0.5={v3['frac_pi_ratio_lt_0p5']:.3f}",
            fontsize=10,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.bar(range(len(ratios)), ratios, color="C4", edgecolor="0.4")
        ax.axhline(1.0, color="k", lw=1.0, ls="--", label="ratio = 1 (no pooling)")
        ax.axhline(0.5, color="r", lw=1.0, ls="--", label="ratio = 0.5 (substantial pooling)")
        ax.set_xlabel("sample-cell index")
        ax.set_ylabel("PI_L23 / PI_L4_partners")
        ax.set_title(f"V3. PI ratio per sample cell ({len(ratios)} cells)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out / "v1_v2_phaseA_v3_pi.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out/'v1_v2_phaseA_v3_pi.png'}")

    # --- V4 phase-generalization decoder ---
    v4_path = in_dir / "v1_v2_phaseA_v4_decode.json"
    if v4_path.exists():
        v4 = json.loads(v4_path.read_text())
        n_l4   = v4["n_l4"]
        n_l23  = v4["n_l23"]
        n_trials = v4["n_trials"]
        labels_theta = np.asarray(v4["label_theta_idx"], dtype=np.int64)
        labels_phi   = np.asarray(v4["label_phi_idx"],   dtype=np.int64)
        train_phi    = set(v4["train_phi_indices"])
        test_phi     = set(v4["test_phi_indices"])

        # Read binary feature dumps.
        l23_rates = np.fromfile(v4["l23_rates_bin"], dtype=np.float32).reshape(
            n_trials, n_l23)
        # L4 file is much larger; only read for the baseline comparison.
        # Memory: 1600 × 131072 × 4 = 800 MB.  Use float32.
        l4_rates = np.fromfile(v4["l4_rates_bin"], dtype=np.float32).reshape(
            n_trials, n_l4)

        train_mask = np.array([p in train_phi for p in labels_phi])
        test_mask  = np.array([p in test_phi  for p in labels_phi])

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        results = {}
        for label, X_full in [("L2/3", l23_rates), ("L4", l4_rates)]:
            X_train = X_full[train_mask]
            X_test  = X_full[test_mask]
            y_train = labels_theta[train_mask]
            y_test  = labels_theta[test_mask]

            scaler = StandardScaler(with_mean=True, with_std=False)
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            clf = LogisticRegression(
                penalty="l2", C=1.0,
                solver="lbfgs", max_iter=2000, tol=1e-4,
            )
            clf.fit(X_train_s, y_train)
            train_acc = float(clf.score(X_train_s, y_train))
            test_acc  = float(clf.score(X_test_s,  y_test))
            chance = 1.0 / 8
            generalization = (test_acc - chance) / max(1e-6, train_acc - chance)
            print(f"V4 {label}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
                  f"gen={generalization:.3f}")
            results[label] = dict(
                train_acc=train_acc, test_acc=test_acc,
                generalization=generalization,
            )

        # Update V4 JSON with results.
        v4["decode_results"] = results
        v4_path.write_text(json.dumps(v4, indent=2))
        print(f"updated {v4_path} with decoder results")

        # Plot.
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        names = list(results.keys())
        train_accs = [results[n]["train_acc"] for n in names]
        test_accs  = [results[n]["test_acc"]  for n in names]
        gens       = [results[n]["generalization"] for n in names]
        x = np.arange(len(names))
        w = 0.3
        ax.bar(x - w/2, train_accs, w, label="train_acc (ϕ ∈ {0, π})", color="C0")
        ax.bar(x + w/2, test_accs,  w, label="test_acc (ϕ ∈ {π/2, 3π/2})", color="C1")
        ax.axhline(1/8, color="k", lw=1.0, ls=":", label="chance (1/8)")
        for xi, n in enumerate(names):
            ax.annotate(f"gen={results[n]['generalization']:.3f}",
                        xy=(xi, max(train_accs[xi], test_accs[xi]) + 0.02),
                        ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("accuracy (8-way classification)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"V4. Phase-generalization decoder (logistic regression)\n"
            f"n_trials={n_trials}, train ϕ∈{{0,π}}, test ϕ∈{{π/2, 3π/2}}",
            fontsize=10,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "v1_v2_phaseA_v4_decode.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out/'v1_v2_phaseA_v4_decode.png'}")

    # --- V5 firing/weight diagnostics ---
    v5_path = in_dir / "v1_v2_phaseA_v5_diag.json"
    if v5_path.exists():
        v5 = json.loads(v5_path.read_text())
        rates = np.asarray(v5["l23_rate_per_cell_hz"], dtype=np.float64)
        weights = np.asarray(v5["l23_w_nS"], dtype=np.float64)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        bins_rate = np.linspace(0, max(1.0, rates.max() * 1.05), 80)
        ax.hist(rates, bins=bins_rate, color="C3", edgecolor="0.4", alpha=0.85)
        l23 = v5["l23"]
        ax.axvline(l23["mean_rate_hz"],   color="k", lw=1.2,
                   label=f"mean={l23['mean_rate_hz']:.3f} Hz")
        ax.axvline(l23["median_rate_hz"], color="0.4", lw=1.0, ls="--",
                   label=f"median={l23['median_rate_hz']:.3f} Hz")
        ax.axvline(l23["max_rate_hz"],    color="C2", lw=1.0, ls=":",
                   label=f"max={l23['max_rate_hz']:.2f} Hz")
        ax.set_xlabel("L2/3 firing rate (Hz, post-train, full-field θ=0)")
        ax.set_ylabel("# L2/3 cells")
        ax.set_yscale("log")
        ax.set_title(
            f"V5a. L2/3 firing-rate distribution\n"
            f"frac_silent={l23['frac_silent']:.3f}, n_l23={len(rates)}",
            fontsize=10,
        )
        ax.legend()

        ax = axes[1]
        # Convert to mV for human-readability.
        mv = weights * 0.974
        bins_w = np.logspace(np.log10(max(mv.min(), 1e-3)),
                             np.log10(mv.max() * 1.05 + 1e-3), 60)
        ax.hist(mv, bins=bins_w, color="C2", edgecolor="0.4", alpha=0.85)
        ws = v5["weights_nS"]
        ax.axvline(ws["median"] * 0.974, color="k", lw=1.2,
                   label=f"median={ws['median']*0.974:.3f} mV")
        ax.axvline(ws["mean"]   * 0.974, color="0.4", lw=1.0, ls="--",
                   label=f"mean={ws['mean']*0.974:.3f} mV")
        ax.axvline(ws["max"]    * 0.974, color="r", lw=1.0, ls=":",
                   label=f"max={ws['max']*0.974:.3f} mV (cap)")
        ax.set_xscale("log")
        ax.set_xlabel("weight (mV-equivalent EPSP, post-train)")
        ax.set_ylabel("# synapses")
        ax.set_title(
            f"V5b. Post-train weight distribution\n"
            f"frac_at_zero={ws['frac_at_zero']:.4f}, frac_at_cap={ws['frac_at_cap']:.4f}",
            fontsize=10,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "v1_v2_phaseA_v5_diag.png", dpi=120)
        plt.close(fig)
        print(f"wrote {out/'v1_v2_phaseA_v5_diag.png'}")


if __name__ == "__main__":
    main()
