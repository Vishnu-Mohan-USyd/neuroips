"""Component-level validation for the Sprint-3 stimulus schedule builders.

Covers three primitives added to `brian2_model/stimulus.py`:

1. `drifting_grating_input(theta_rad, contrast, duration_ms, dt_ms)`
   -> (n_steps, N_CHANNELS) Poisson-rate envelope.

2. `richter_crossover_training_schedule(rng, n_trials, ...)`
   -> Richter 2022 (DOI 10.1093/oons/kvac013) 6x6 leader->trailer paradigm.

3. `tang_rotating_sequence(rng, n_items, ...)`
   -> Tang 2023 (PMID 36864037) rotating orientation blocks with deviants.

Both trial builders must be deterministic under a seeded `np.random.Generator`
because Stage-1 gate evidence is reproducible only if the schedule is
reproducible.

Assays
------

A. Drifting-grating envelope
   A1. Shape == (n_steps, N_CHANNELS), n_steps = duration_ms / dt_ms.
   A2. argmax channel == channel closest to theta.
   A3. Envelope symmetric around the peak channel (Gaussian).
   A4. Peak rate == peak_rate_hz * contrast.
   A5. Width (FWHM) matches sigma_deg analytically
       (FWHM = 2 * sqrt(2*ln2) * sigma).

B. Richter schedule
   B1. Deterministic under seed=42 (two identical RNGs -> same `pairs`).
   B2. Balanced: each (leader, trailer) pair appears the same number of
       times (n_trials / 36).
   B3. Item kinds alternate leader -> trailer -> iti.
   B4. Timing: leader_ms + trailer_ms + iti_ms per trial.
   B5. Orientation coverage: all 6 orientations appear as leader AND as
       trailer at least once.

C. Tang schedule
   C1. Deterministic under seed=42.
   C2. Within each block, consecutive non-deviant items differ by exactly
       direction * 30 deg (mod 180) on the ring.
   C3. Deviant (last item of each block) != the expected next orientation.
   C4. Block lengths are in [block_len_range[0], block_len_range[1]],
       inclusive.
   C5. Deviant fraction is 1/mean_block_len within 20% (finite-sample).
   C6. Rotation direction is {+1, -1} only.

Run:
    python -m expectation_snn.validation.validate_stimulus_schedules
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..brian2_model.stimulus import (
    N_CHANNELS,
    CHANNEL_SPACING_RAD,
    DEFAULT_STIM_PEAK_HZ,
    DEFAULT_STIM_SIGMA_DEG,
    RICHTER_ORIENTATIONS_DEG,
    TANG_BLOCK_LEN_RANGE,
    TANG_ORIENTATIONS_DEG,
    drifting_grating_input,
    richter_crossover_training_schedule,
    tang_rotating_sequence,
)


STIM_DETERMINISM_SEED = 42


# --- small helpers ----------------------------------------------------------

def _fwhm_pixels(envelope: np.ndarray) -> int:
    """Full-width at half-maximum (in channel units) of a 1D ring envelope.

    Rolls the array so the peak is at center before computing the FWHM,
    so that Gaussians centred on channel 0 (wrapping to channels N-1, 1)
    are handled correctly.
    """
    peak_idx = int(np.argmax(envelope))
    centered = np.roll(envelope, len(envelope) // 2 - peak_idx)
    peak = float(centered.max())
    half = 0.5 * peak
    above = np.where(centered >= half)[0]
    if len(above) == 0:
        return 0
    return int(above.max() - above.min() + 1)


# --- assay A: drifting_grating_input ---------------------------------------

@dataclass
class GratingAssay:
    shape_ok: bool
    peak_channel_ok: bool
    symmetry_ok: bool
    peak_rate_ok: bool
    fwhm_channels: int
    fwhm_expected_channels: float
    fwhm_ok: bool

    @property
    def passed(self) -> bool:
        return (self.shape_ok and self.peak_channel_ok and self.symmetry_ok
                and self.peak_rate_ok and self.fwhm_ok)


def _check_grating() -> GratingAssay:
    theta = 0.0
    duration_ms = 250.0
    dt_ms = 1.0
    contrast = 0.8
    t, rates = drifting_grating_input(
        theta_rad=theta, contrast=contrast, duration_ms=duration_ms, dt_ms=dt_ms,
        peak_rate_hz=DEFAULT_STIM_PEAK_HZ, sigma_deg=DEFAULT_STIM_SIGMA_DEG,
    )
    # A1
    shape_ok = rates.shape == (int(duration_ms / dt_ms), N_CHANNELS)
    # A2
    peak_ch = int(np.argmax(rates[0]))
    # theta=0 -> channel closest to 0 -> channel 0.
    peak_channel_ok = (peak_ch == 0)
    # A3 symmetry on the ring: rates[0, k] == rates[0, N_CHANNELS - k] for k>0.
    env = rates[0]
    sym_ok = all(
        abs(env[k] - env[(N_CHANNELS - k) % N_CHANNELS]) < 1e-9
        for k in range(1, N_CHANNELS // 2 + 1)
    )
    # A4
    peak_expected = DEFAULT_STIM_PEAK_HZ * contrast
    peak_ok = abs(float(env.max()) - peak_expected) < 1e-6
    # A5 FWHM
    fwhm_deg_analytic = 2.0 * np.sqrt(2.0 * np.log(2.0)) * DEFAULT_STIM_SIGMA_DEG
    fwhm_channels_analytic = fwhm_deg_analytic / (180.0 / N_CHANNELS)  # 15 deg/ch
    fwhm_ch = _fwhm_pixels(env)
    # Allow +/-1 channel tolerance on an integer-sampled envelope.
    fwhm_ok = abs(fwhm_ch - fwhm_channels_analytic) <= 1.0

    return GratingAssay(
        shape_ok=shape_ok,
        peak_channel_ok=peak_channel_ok,
        symmetry_ok=sym_ok,
        peak_rate_ok=peak_ok,
        fwhm_channels=fwhm_ch,
        fwhm_expected_channels=float(fwhm_channels_analytic),
        fwhm_ok=fwhm_ok,
    )


# --- assay B: Richter schedule ---------------------------------------------

@dataclass
class RichterAssay:
    n_trials: int
    deterministic: bool
    balanced_counts_min: int
    balanced_counts_max: int
    balanced_ok: bool
    kinds_ok: bool
    timing_ok: bool
    leader_coverage_ok: bool
    trailer_coverage_ok: bool

    @property
    def passed(self) -> bool:
        return (self.deterministic and self.balanced_ok and self.kinds_ok
                and self.timing_ok and self.leader_coverage_ok
                and self.trailer_coverage_ok)


def _check_richter(n_trials: int = 360) -> RichterAssay:
    # B1 determinism
    rng_a = np.random.default_rng(STIM_DETERMINISM_SEED)
    rng_b = np.random.default_rng(STIM_DETERMINISM_SEED)
    plan_a = richter_crossover_training_schedule(rng_a, n_trials=n_trials)
    plan_b = richter_crossover_training_schedule(rng_b, n_trials=n_trials)
    pairs_a = plan_a.meta["pairs"]
    pairs_b = plan_b.meta["pairs"]
    deterministic = bool(np.array_equal(pairs_a, pairs_b))

    # B2 balanced 6x6
    counts = np.zeros((6, 6), dtype=np.int64)
    for li, ti in pairs_a:
        counts[li, ti] += 1
    expected = n_trials // 36
    balanced = bool(counts.min() == expected and counts.max() == expected)

    # B3 kinds: leader, trailer, iti, leader, trailer, iti, ...
    leader_ms = 500.0
    trailer_ms = 500.0
    iti_ms = 5000.0
    kinds_expected = ["leader", "trailer", "iti"] * 3
    kinds_ok = [it.kind for it in plan_a.items[:9]] == kinds_expected

    # B4 timing
    expected_total = n_trials * (leader_ms + trailer_ms + iti_ms)
    timing_ok = abs(plan_a.total_ms - expected_total) < 1e-6

    # B5 coverage
    leader_idxs = set(int(li) for li, _ in pairs_a)
    trailer_idxs = set(int(ti) for _, ti in pairs_a)
    leader_coverage = leader_idxs == set(range(6))
    trailer_coverage = trailer_idxs == set(range(6))

    return RichterAssay(
        n_trials=n_trials,
        deterministic=deterministic,
        balanced_counts_min=int(counts.min()),
        balanced_counts_max=int(counts.max()),
        balanced_ok=balanced,
        kinds_ok=kinds_ok,
        timing_ok=timing_ok,
        leader_coverage_ok=leader_coverage,
        trailer_coverage_ok=trailer_coverage,
    )


# --- assay C: Tang schedule ------------------------------------------------

@dataclass
class TangAssay:
    n_items: int
    deterministic: bool
    rotation_ok: bool
    deviant_ne_expected_ok: bool
    block_len_in_range_ok: bool
    deviant_frac: float
    deviant_frac_expected: float
    deviant_frac_ok: bool
    direction_in_set_ok: bool

    @property
    def passed(self) -> bool:
        return (self.deterministic and self.rotation_ok
                and self.deviant_ne_expected_ok
                and self.block_len_in_range_ok
                and self.deviant_frac_ok and self.direction_in_set_ok)


def _check_tang(n_items: int = 1000) -> TangAssay:
    # C1 determinism
    rng_a = np.random.default_rng(STIM_DETERMINISM_SEED)
    rng_b = np.random.default_rng(STIM_DETERMINISM_SEED)
    plan_a = tang_rotating_sequence(rng_a, n_items=n_items)
    plan_b = tang_rotating_sequence(rng_b, n_items=n_items)
    thetas_a = np.array([it.theta_rad for it in plan_a.items])
    thetas_b = np.array([it.theta_rad for it in plan_b.items])
    deterministic = bool(np.array_equal(thetas_a, thetas_b))

    block_ids = plan_a.meta["block_ids"]
    rot_dir = plan_a.meta["rotation_dir"]
    dev_mask = plan_a.meta["deviant_mask"]
    items = plan_a.items
    bids_unique = np.unique(block_ids)

    # C2 rotation within block.
    rotation_ok = True
    for bid in bids_unique:
        idxs = np.where(block_ids == bid)[0]
        if len(idxs) < 2:
            continue
        direction = int(rot_dir[idxs[0]])
        for k in range(1, len(idxs) - 1):
            prev = items[idxs[k - 1]]
            curr = items[idxs[k]]
            step = np.rad2deg(curr.theta_rad - prev.theta_rad)
            step = ((step + 90) % 180) - 90    # wrap to [-90, 90]
            if abs(step - direction * 30.0) > 1e-6:
                rotation_ok = False
                break
        if not rotation_ok:
            break

    # C3 deviant != expected next
    dev_ne_ok = True
    for bid in bids_unique:
        idxs = np.where(block_ids == bid)[0]
        if not dev_mask[idxs[-1]]:
            continue
        dev_item = items[idxs[-1]]
        first_item = items[idxs[0]]
        direction = int(rot_dir[idxs[0]])
        first_deg = float(np.rad2deg(first_item.theta_rad))
        n_pos = len(idxs) - 1
        expected_deg = (first_deg + direction * 30.0 * n_pos) % 180.0
        dev_deg = float(np.rad2deg(dev_item.theta_rad)) % 180.0
        if abs(dev_deg - expected_deg) < 1e-6:
            dev_ne_ok = False
            break

    # C4 block length range
    blen_lo, blen_hi = TANG_BLOCK_LEN_RANGE
    block_lens = np.array([np.sum(block_ids == b) for b in bids_unique])
    # Ignore the potentially-truncated last block.
    if len(block_lens) > 1:
        lens_internal = block_lens[:-1]
    else:
        lens_internal = block_lens
    in_range = bool((lens_internal.min() >= blen_lo)
                    and (lens_internal.max() <= blen_hi))

    # C5 deviant fraction ~ 1 / mean_block_len
    mean_blen = float(np.mean(lens_internal)) if len(lens_internal) > 0 else np.nan
    deviant_frac = float(dev_mask.mean())
    deviant_frac_expected = 1.0 / mean_blen if mean_blen > 0 else np.nan
    deviant_frac_ok = (
        not np.isnan(deviant_frac_expected)
        and abs(deviant_frac - deviant_frac_expected) / deviant_frac_expected < 0.2
    )

    # C6 direction in {-1, +1}
    dir_ok = bool(set(np.unique(rot_dir).tolist()) <= {-1, 1})

    return TangAssay(
        n_items=n_items,
        deterministic=deterministic,
        rotation_ok=rotation_ok,
        deviant_ne_expected_ok=dev_ne_ok,
        block_len_in_range_ok=in_range,
        deviant_frac=deviant_frac,
        deviant_frac_expected=deviant_frac_expected,
        deviant_frac_ok=deviant_frac_ok,
        direction_in_set_ok=dir_ok,
    )


# --- aggregator ------------------------------------------------------------

@dataclass
class StimulusValidationReport:
    grating: GratingAssay
    richter: RichterAssay
    tang: TangAssay

    @property
    def passed(self) -> bool:
        return self.grating.passed and self.richter.passed and self.tang.passed

    def summary(self) -> str:
        g = self.grating
        r = self.richter
        t = self.tang
        lines = ["Stimulus schedule validation:"]
        lines.append("  A. drifting_grating_input:")
        lines.append(f"    shape == (n_steps, N_CHANNELS):  "
                     f"{'PASS' if g.shape_ok else 'FAIL'}")
        lines.append(f"    peak channel == closest to theta: "
                     f"{'PASS' if g.peak_channel_ok else 'FAIL'}")
        lines.append(f"    symmetric envelope on ring:       "
                     f"{'PASS' if g.symmetry_ok else 'FAIL'}")
        lines.append(f"    peak rate == peak_rate_hz * c:    "
                     f"{'PASS' if g.peak_rate_ok else 'FAIL'}")
        lines.append(f"    FWHM = {g.fwhm_channels} channels "
                     f"(expected {g.fwhm_expected_channels:.2f}, +/-1): "
                     f"{'PASS' if g.fwhm_ok else 'FAIL'}")
        lines.append("  B. richter_crossover_training_schedule:")
        lines.append(f"    deterministic under seed={STIM_DETERMINISM_SEED}: "
                     f"{'PASS' if r.deterministic else 'FAIL'}")
        lines.append(f"    balanced 6x6 [min={r.balanced_counts_min}, "
                     f"max={r.balanced_counts_max}]: "
                     f"{'PASS' if r.balanced_ok else 'FAIL'}")
        lines.append(f"    item kinds (leader,trailer,iti)*3: "
                     f"{'PASS' if r.kinds_ok else 'FAIL'}")
        lines.append(f"    total timing == n_trials*6000 ms: "
                     f"{'PASS' if r.timing_ok else 'FAIL'}")
        lines.append(f"    full leader coverage (all 6):      "
                     f"{'PASS' if r.leader_coverage_ok else 'FAIL'}")
        lines.append(f"    full trailer coverage (all 6):     "
                     f"{'PASS' if r.trailer_coverage_ok else 'FAIL'}")
        lines.append("  C. tang_rotating_sequence:")
        lines.append(f"    deterministic under seed={STIM_DETERMINISM_SEED}: "
                     f"{'PASS' if t.deterministic else 'FAIL'}")
        lines.append(f"    rotation step == +/-30 deg in-block: "
                     f"{'PASS' if t.rotation_ok else 'FAIL'}")
        lines.append(f"    deviant != expected next:           "
                     f"{'PASS' if t.deviant_ne_expected_ok else 'FAIL'}")
        lines.append(f"    block length in {TANG_BLOCK_LEN_RANGE}: "
                     f"{'PASS' if t.block_len_in_range_ok else 'FAIL'}")
        lines.append(f"    deviant frac = {t.deviant_frac:.3f} "
                     f"(expected {t.deviant_frac_expected:.3f}, +/-20%): "
                     f"{'PASS' if t.deviant_frac_ok else 'FAIL'}")
        lines.append(f"    direction set {{-1, +1}}:           "
                     f"{'PASS' if t.direction_in_set_ok else 'FAIL'}")
        lines.append("  -----------------------------------")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def run_stimulus_validation(verbose: bool = True) -> StimulusValidationReport:
    rep = StimulusValidationReport(
        grating=_check_grating(),
        richter=_check_richter(n_trials=360),
        tang=_check_tang(n_items=1000),
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_stimulus_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
