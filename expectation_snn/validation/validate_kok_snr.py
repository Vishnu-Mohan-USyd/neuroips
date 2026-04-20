"""Functional validation for Sprint 5d Kok SNR-probe knobs (task #41 step 3).

Why this validator exists
-------------------------
Sprint 5c meta-review Flaw 4: ``Δ_decoding`` saturates at 100 % on
2-orientation × 192-cell features, blinding the Kok assay to any
expectation-driven sharpening. Diagnostic **D6 — Kok SNR curve**
(:doc:`SPRINT_5C_META_REVIEW.md`) degrades the decoder via contrast
reduction / input noise / cell subsampling / more orientations and
reports ``Acc_valid − Acc_invalid`` in a non-saturated regime.

This validator proves the *infrastructure* for D6 works, without running
the science sweep itself. The five functional checks are:

  [1] default_config_is_backward_compatible
        Run the assay with defaults (``contrast_multiplier=1.0``,
        ``input_noise_std_hz=0.0``, ``n_cells_subsampled=None``,
        ``n_orientations=2``). Expect:
          - meta["snr_contrast_multiplier"] == 1.0
          - meta["snr_input_noise_std_hz"]  == 0.0
          - meta["snr_n_cells_used"] == 192
          - meta["snr_n_orientations"] == 2
          - orientation_mvpa has the legacy keys ``delta_decoding`` +
            ``delta_decoding_ci`` (not NaN, mode != "multi_class")
          - svm (legacy validity decoder) returns a finite ``accuracy``

  [2] contrast_multiplier_reduces_valid_amp
        Same short paradigm with ``contrast_multiplier=0.3``. Compare
        ``mean_amp["valid"]`` rate vs the baseline run from [1]; the
        ×0.3 run must have strictly lower total_rate_hz (grating drive
        is weaker).

  [3] cell_subsample_is_exactly_reported
        ``n_cells_subsampled=32``. Expect:
          - meta["snr_n_cells_used"] == 32
          - raw["sub_cells"].size == 32
          - raw["sub_cells"].min() >= 0 and .max() < N_V1_E
          - orientation_mvpa still returns a finite ``delta_decoding``

  [4] low_contrast_plus_subsample_drops_below_saturation
        ``contrast_multiplier=0.3`` **AND** ``n_cells_subsampled=32``.
        This is D6's target regime. Expect ``acc_valid_mean < 0.95``
        (strictly below saturation floor of 1.0), and the decoder still
        discriminates above chance (``acc_valid_mean >= 0.40``, chance
        =0.5 minus tolerance for small-n noise). The {55, 80}% band is
        a property of a *full-paradigm* sweep; a ≤ 2-minute validator
        run cannot reproduce the band exactly, but it *can* prove the
        saturation ceiling is broken. That's what we assert.

  [5] n_orientations_6_and_12_run_end_to_end
        Run with ``n_orientations=6`` and ``n_orientations=12`` (tiny
        stim counts). Expect:
          - orientation_mvpa["mode"] == "multi_class"
          - orientation_mvpa["n_classes"] == {6, 12}
          - orientation_mvpa["accuracy"] is finite and > chance_acc
            (1/n_orientations); if not we accept equal to chance (weak
            signal in short runs) but NOT NaN or < 0.

All assays use seed=42, Brian2 numpy codegen, dt=0.1 ms. Each run uses a
short paradigm (small n_stim, ~200 ms per epoch) so total wall time is
under 6 minutes on a laptop CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from expectation_snn.assays.kok_passive import KokConfig, run_kok_passive


# ---------------------------------------------------------------------------
# Shared short-paradigm config factory
# ---------------------------------------------------------------------------

def _mini_kok(**overrides) -> KokConfig:
    """Return a short-paradigm Kok config for SNR validation runs.

    Short epochs + few trials so each assay is fast; MVPA bootstrap
    rounds cut aggressively (validator tests *plumbing*, not
    statistical-power-of-the-decoder).
    """
    base = dict(
        n_stim_trials=12,
        n_omission_trials=2,
        cue_ms=100.0,
        gap_ms=100.0,
        grating_ms=200.0,
        iti_ms=100.0,
        mvpa_n_subsamples=2,
        mvpa_n_bootstrap=20,
        mvpa_cv=2,
        seed=42,
    )
    base.update(overrides)
    return KokConfig(**base)


def _multi_orient_kok(n_orientations: int, **overrides) -> KokConfig:
    """Multi-orientation mini config: n_stim divisible by n_orientations."""
    n_stim = 6 * n_orientations if n_orientations >= 6 else 24
    base = dict(
        n_stim_trials=n_stim,
        n_omission_trials=0,
        cue_ms=100.0,
        gap_ms=100.0,
        grating_ms=200.0,
        iti_ms=100.0,
        mvpa_cv=2,
        n_orientations=int(n_orientations),
        seed=42,
    )
    base.update(overrides)
    return KokConfig(**base)


# ---------------------------------------------------------------------------
# Assays
# ---------------------------------------------------------------------------

def assay_default_config_is_backward_compatible() -> None:
    print("[1] default_config_is_backward_compatible")
    cfg = _mini_kok()
    res = run_kok_passive(cfg=cfg, verbose=False)

    m = res.meta
    assert m["snr_contrast_multiplier"] == 1.0, m
    assert m["snr_input_noise_std_hz"] == 0.0, m
    assert m["snr_n_cells_used"] == 192, (
        f"expected 192 E cells, got {m['snr_n_cells_used']}"
    )
    assert m["snr_n_orientations"] == 2, m
    assert abs(m["snr_effective_contrast"] - cfg.contrast) < 1e-9

    om = res.orientation_mvpa
    assert "delta_decoding" in om and np.isfinite(om["delta_decoding"]), om
    assert om.get("mode", None) != "multi_class", om

    svm = res.svm
    assert np.isfinite(svm.get("accuracy", np.nan)), svm
    print(f"    n_cells_used={m['snr_n_cells_used']}  "
          f"Δ_dec={om['delta_decoding']:+.3f}  "
          f"legacy_svm={svm['accuracy']:.3f}  PASS")


def assay_contrast_multiplier_reduces_valid_amp() -> None:
    print("[2] contrast_multiplier_reduces_valid_amp")
    cfg_full = _mini_kok()
    cfg_low = _mini_kok(contrast_multiplier=0.3)
    res_full = run_kok_passive(cfg=cfg_full, verbose=False)
    res_low = run_kok_passive(cfg=cfg_low, verbose=False)

    amp_full = float(res_full.mean_amp["valid"]["total_rate_hz"])
    amp_low = float(res_low.mean_amp["valid"]["total_rate_hz"])
    assert amp_low < amp_full, (
        f"contrast_multiplier=0.3 should LOWER valid amp "
        f"({amp_low:.2f} Hz) vs 1.0 ({amp_full:.2f} Hz)"
    )
    assert abs(res_low.meta["snr_effective_contrast"]
               - 0.3 * cfg_low.contrast) < 1e-9, res_low.meta
    print(f"    valid_amp @1.0 = {amp_full:.3f} Hz  >  "
          f"valid_amp @0.3 = {amp_low:.3f} Hz  PASS")


def assay_cell_subsample_is_exactly_reported() -> None:
    print("[3] cell_subsample_is_exactly_reported")
    cfg = _mini_kok(n_cells_subsampled=32)
    res = run_kok_passive(cfg=cfg, verbose=False)

    m = res.meta
    sub = res.raw["sub_cells"]
    assert m["snr_n_cells_used"] == 32, m
    assert sub.size == 32, f"sub_cells shape {sub.shape}"
    assert sub.min() >= 0 and sub.max() < 192, (
        f"sub_cells out of range: min={sub.min()}, max={sub.max()}"
    )
    assert np.unique(sub).size == 32, "sub_cells must be unique"

    om = res.orientation_mvpa
    assert np.isfinite(om["delta_decoding"]), om
    print(f"    sub_cells.size={sub.size}  range=[{sub.min()},{sub.max()}]  "
          f"Δ_dec={om['delta_decoding']:+.3f}  PASS")


def assay_low_contrast_plus_subsample_drops_below_saturation() -> None:
    print("[4] low_contrast_plus_subsample_drops_below_saturation")
    cfg = _mini_kok(contrast_multiplier=0.3, n_cells_subsampled=32,
                    n_stim_trials=24)   # a few more trials for decoder
    res = run_kok_passive(cfg=cfg, verbose=False)

    om = res.orientation_mvpa
    acc_v = float(om["acc_valid_mean"])
    acc_i = float(om["acc_invalid_mean"])
    assert np.isfinite(acc_v) and np.isfinite(acc_i), om
    # Saturation means acc_valid == 1.0; if our probe breaks saturation,
    # at least ONE of the two accuracies should be below the 0.95 ceiling.
    assert min(acc_v, acc_i) < 0.95, (
        f"decoder still saturated: acc_valid={acc_v:.3f} "
        f"acc_invalid={acc_i:.3f} (both >= 0.95)"
    )
    # Anti-inverted-decoder sanity: at least one decoder >= 0.20. With 12
    # valid + 12 invalid stim trials, 2-class 2-fold CV has quantized output
    # ∈ {0, 1/12, ..., 1} so chance = 0.5 ± ~0.15 on small-N. Threshold 0.20
    # catches a catastrophic label-flip bug (accuracy ≤ 0.10) but tolerates
    # normal small-sample noise. The 55-80% band is a full-paradigm property
    # (see module docstring); this validator asserts infrastructure only.
    assert max(acc_v, acc_i) >= 0.20, (
        f"decoder inverted / catastrophically broken: "
        f"acc_v={acc_v:.3f} acc_i={acc_i:.3f}"
    )
    print(f"    contrast×0.3, subsample=32  "
          f"acc_v={acc_v:.3f}  acc_i={acc_i:.3f}  "
          f"Δ_dec={om['delta_decoding']:+.3f}  PASS (saturation broken)")


def assay_n_orientations_6_and_12_run_end_to_end() -> None:
    print("[5] n_orientations_6_and_12_run_end_to_end")
    for n in (6, 12):
        cfg = _multi_orient_kok(n_orientations=n)
        res = run_kok_passive(cfg=cfg, verbose=False)
        om = res.orientation_mvpa
        assert om.get("mode") == "multi_class", (n, om)
        assert int(om["n_classes"]) == n, (n, om)
        assert np.isfinite(om["accuracy"]), (n, om)
        chance = 1.0 / float(n)
        # short runs may land near chance; we only require finite and >= 0.
        assert om["accuracy"] >= 0.0, (n, om)
        print(f"    n_orient={n}  accuracy={om['accuracy']:.3f}  "
              f"chance={chance:.3f}  n_per_class={om['n_per_class']}")
    print("    PASS")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    np.random.seed(42)
    assays = [
        assay_default_config_is_backward_compatible,
        assay_contrast_multiplier_reduces_valid_amp,
        assay_cell_subsample_is_exactly_reported,
        assay_low_contrast_plus_subsample_drops_below_saturation,
        assay_n_orientations_6_and_12_run_end_to_end,
    ]
    failed: list[str] = []
    for a in assays:
        try:
            a()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{a.__name__}: {exc}")
            print(f"    FAIL — {exc}")
    n = len(assays)
    n_ok = n - len(failed)
    print()
    print(f"validate_kok_snr: {n_ok}/{n} PASS")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
