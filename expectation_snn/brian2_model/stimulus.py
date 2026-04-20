"""Stimulus generators.

Core primitive:

- `drifting_grating_input(theta_rad, contrast, duration_ms, dt_ms)`:
    returns per-channel Poisson rate arrays (n_steps, N_CHANNELS) over the
    window. The caller applies these to `V1Ring.stim` via
    `v1_ring.set_stimulus`.

Stage-1/Stage-2 trial schedule builders:

- `richter_crossover_training_schedule(...)`:
    Richter 2022 (DOI 10.1093/oons/kvac013) 6 × 6 leader -> trailer cross-over.
    500 ms leader, 500 ms trailer, no ISI between, ~5 s ITI between trials.
    All 36 leader-trailer pairs, replicated to reach `n_trials`.
    Deterministic under provided `rng` (Plan pre-reg: seed=42 for current stage).

- `tang_rotating_sequence(...)`:
    Tang 2023 (PMID 36864037, PDF-verified in docs/research_log.md) rotating
    orientation paradigm. 6 orientations at 30° steps, 250 ms per item, 4 Hz
    (no ISI), rotating blocks of length in {5,6,7,8,9} with CW/CCW sampled
    per block, ending with a deviant sampled uniformly from the 5 remaining
    orientations. Deterministic under provided `rng`.

- `kok_trial(...)`:
    Kok 2012 cue -> delay -> grating. Currently a stub, deferred to Stage-2.

Conventions
-----------
- Orientations are in RADIANS on the 0..π ring (matches `V1Ring.thetas_rad`).
- `TrialItem.theta_rad=None` marks a blank / ISI epoch.
- Plans are pure descriptions; the driver (`train.py`) consumes them and
  drives `V1Ring.stim.rates` via `drifting_grating_input` on each epoch.

References
----------
- Tang 2023 PMID 36864037 (paradigm verified in docs/research_log.md Phase 0).
- Richter 2022 DOI 10.1093/oons/kvac013.
- Kok 2012 PMID 22841311.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Shared ring geometry: keep in sync with v1_ring / h_ring.
N_CHANNELS = 12
CHANNEL_SPACING_RAD = np.pi / N_CHANNELS  # 15 deg
DEFAULT_STIM_PEAK_HZ = 80.0
DEFAULT_STIM_SIGMA_DEG = 15.0

# Paper-verified constants.
TANG_ORIENTATIONS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
TANG_ITEM_MS = 250.0
TANG_BLOCK_LEN_RANGE = (5, 9)   # inclusive lower, inclusive upper

RICHTER_LEADER_MS = 500.0
RICHTER_TRAILER_MS = 500.0
RICHTER_ITI_MS = 5000.0
RICHTER_ORIENTATIONS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)


@dataclass
class TrialItem:
    """One timed stimulus epoch.

    Attributes
    ----------
    theta_rad : Optional[float]
        Orientation in radians on the 0..pi ring. None -> blank / ISI.
    contrast : float
        Linear scale on peak Poisson rate; 0..1 typical.
    duration_ms : float
    kind : str
        Free-form label: "grating", "blank", "cue", "leader", "trailer",
        "rotating", "deviant", "iti". Used for downstream analysis splits.
    meta : dict
        Optional per-item metadata (e.g., block index, rotation direction).
    """
    theta_rad: Optional[float]
    contrast: float
    duration_ms: float
    kind: str = "grating"
    meta: dict = field(default_factory=dict)


@dataclass
class TrialPlan:
    """A schedule = list of TrialItems + top-level metadata."""
    items: List[TrialItem]
    meta: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return float(sum(it.duration_ms for it in self.items))

    def n_items(self) -> int:
        return len(self.items)


# --- core Poisson-rate generator -------------------------------------------

def drifting_grating_input(
    theta_rad: float,
    contrast: float,
    duration_ms: float,
    dt_ms: float = 1.0,
    peak_rate_hz: float = DEFAULT_STIM_PEAK_HZ,
    sigma_deg: float = DEFAULT_STIM_SIGMA_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel Poisson rates for a drifting grating over a window.

    Parameters
    ----------
    theta_rad : float
        Grating orientation in radians (0..pi).
    contrast : float
    duration_ms : float
    dt_ms : float
    peak_rate_hz : float
    sigma_deg : float

    Returns
    -------
    t_ms : np.ndarray, shape (n_steps,)
    rates_hz : np.ndarray, shape (n_steps, N_CHANNELS)
        Constant across the window for drifting gratings in plan v5
        (we don't model phase/temporal-frequency modulation at rate level).
    """
    n_steps = max(1, int(round(duration_ms / dt_ms)))
    t_ms = np.arange(n_steps, dtype=np.float64) * dt_ms
    thetas = np.arange(N_CHANNELS) * CHANNEL_SPACING_RAD
    d = np.abs(thetas - float(theta_rad))
    d = np.minimum(d, np.pi - d)  # wrap-around on 0..pi ring
    sigma_rad = np.deg2rad(sigma_deg)
    per_channel = peak_rate_hz * float(contrast) * np.exp(
        -0.5 * (d / sigma_rad) ** 2
    )
    rates = np.broadcast_to(per_channel[None, :], (n_steps, N_CHANNELS)).copy()
    return t_ms, rates


# --- Richter 2022 cross-over training schedule -----------------------------

def richter_crossover_training_schedule(
    rng: np.random.Generator,
    n_trials: int = 360,
    leader_ms: float = RICHTER_LEADER_MS,
    trailer_ms: float = RICHTER_TRAILER_MS,
    iti_ms: float = RICHTER_ITI_MS,
    contrast: float = 1.0,
    orientations_deg: Sequence[float] = RICHTER_ORIENTATIONS_DEG,
) -> TrialPlan:
    """Richter 2022 6 × 6 leader -> trailer cross-over training schedule.

    Each trial = `leader_ms` of leader orientation -> immediate `trailer_ms`
    of trailer orientation (no ISI between leader and trailer) -> `iti_ms`
    of blank. Leaders and trailers are drawn from the full 6-orientation
    set; all 36 pairs appear equally often (balanced across repeats).

    Parameters
    ----------
    rng : np.random.Generator
        Deterministic RNG (plan pre-reg: seed=42 for current stage).
    n_trials : int
        Total trial count. 360 = 10 replications of each of 36 L-T pairs.
    leader_ms, trailer_ms : float
        Per-epoch duration (no ISI between them).
    iti_ms : float
        Blank ITI after the trailer.
    contrast : float
    orientations_deg : Sequence[float]
        Must have length 6 (paper 6 × 6 cross-over).

    Returns
    -------
    TrialPlan
        `meta` contains {"paradigm": "richter_crossover", "n_trials",
        "pairs": np.ndarray shape (n_trials, 2) — indices into
        orientations_deg for each (leader, trailer) pair}.

    References
    ----------
    Richter D, de Lange FP (2022) DOI 10.1093/oons/kvac013.
    """
    n_orients = len(orientations_deg)
    if n_orients != 6:
        raise ValueError(
            f"Richter paradigm requires 6 orientations, got {n_orients}"
        )
    n_pairs = n_orients * n_orients  # 36
    if n_trials % n_pairs != 0:
        raise ValueError(
            f"n_trials ({n_trials}) must be a multiple of "
            f"{n_pairs} (6 x 6 pairs) for a balanced schedule"
        )
    replicates = n_trials // n_pairs

    pairs = np.empty((n_trials, 2), dtype=np.int64)
    base = np.array([(i, j) for i in range(n_orients) for j in range(n_orients)],
                    dtype=np.int64)  # (36, 2)
    for r in range(replicates):
        block = base.copy()
        rng.shuffle(block, axis=0)
        pairs[r * n_pairs : (r + 1) * n_pairs] = block

    thetas_rad = np.deg2rad(np.asarray(orientations_deg, dtype=np.float64))

    items: List[TrialItem] = []
    for trial_idx, (li, ti) in enumerate(pairs):
        items.append(TrialItem(
            theta_rad=float(thetas_rad[li]),
            contrast=contrast,
            duration_ms=leader_ms,
            kind="leader",
            meta={"trial": trial_idx, "leader_idx": int(li),
                  "trailer_idx": int(ti)},
        ))
        items.append(TrialItem(
            theta_rad=float(thetas_rad[ti]),
            contrast=contrast,
            duration_ms=trailer_ms,
            kind="trailer",
            meta={"trial": trial_idx, "leader_idx": int(li),
                  "trailer_idx": int(ti)},
        ))
        if iti_ms > 0:
            items.append(TrialItem(
                theta_rad=None, contrast=0.0, duration_ms=iti_ms,
                kind="iti", meta={"trial": trial_idx},
            ))

    return TrialPlan(
        items=items,
        meta={"paradigm": "richter_crossover",
              "n_trials": n_trials,
              "pairs": pairs,
              "orientations_deg": tuple(orientations_deg)},
    )


# --- Richter biased training schedule (Sprint 5e Fix A) --------------------

def richter_biased_training_schedule(
    rng: np.random.Generator,
    n_trials: int = 360,
    p_bias: float = 0.80,
    derangement: Sequence[int] = (1, 2, 3, 4, 5, 0),
    leader_ms: float = RICHTER_LEADER_MS,
    trailer_ms: float = RICHTER_TRAILER_MS,
    iti_ms: float = RICHTER_ITI_MS,
    contrast: float = 1.0,
    orientations_deg: Sequence[float] = RICHTER_ORIENTATIONS_DEG,
) -> TrialPlan:
    """Biased Richter training schedule with learnable leader→trailer contingency.

    Sprint 5e Fix A: the legacy `richter_crossover_training_schedule` is
    balanced over all 36 leader-trailer pairs, so `P(T|L) = 1/6` for every
    `L` — a uniform transition matrix with **no** statistical regularity for
    STDP to latch onto. Per Sprint 5e-Diag (task #43, commit 5c8ad05), this
    schedule has `min_L entropy = log2(6) = 2.585` bits — maximal, i.e. zero
    contingency. The deranged-permutation variant implemented here induces
    a learnable prior: `P(L → f(L)) = p_bias`, with the remaining
    `1 − p_bias` split uniformly across the four trailers that are neither
    `L` nor `f(L)` (same-orientation trailers never occur under this
    schedule, avoiding the same-θ adaptation confound Sprint 5c R1 already
    removed from the test phase).

    The legacy balanced schedule remains the **test-phase** generator
    (`richter_crossover_training_schedule`); this function is
    training-phase only.

    Parameters
    ----------
    rng : np.random.Generator
        Deterministic RNG (seed=42 per current-stage pre-reg).
    n_trials : int
        Total trial count. Must be a multiple of `len(orientations_deg)` so
        that each leader appears exactly `n_trials / n_orients` times.
    p_bias : float
        Probability that `T = f(L)` given `L`. Must lie in (1/(n-1), 1) so
        that the expected-transition dominates the uniform alternative.
        Default 0.80 per Sprint 5e-Diag reference.
    derangement : Sequence[int]
        A derangement (fixed-point-free permutation) of `range(n_orients)`:
        `derangement[L] = f(L)` with `f(L) != L` for all `L`. Default
        `(1, 2, 3, 4, 5, 0)` — one-step rotation on the 6-orientation ring,
        matching the assay's expected direction (Sprint 5c R1).
    leader_ms, trailer_ms, iti_ms, contrast : float
    orientations_deg : Sequence[float]
        Must have length 6.

    Returns
    -------
    TrialPlan
        `meta` contains:
          - `paradigm` = "richter_biased"
          - `n_trials`
          - `pairs` : (n_trials, 2) int64 — (leader_idx, trailer_idx)
          - `expected_trailer_idx` : (n_trials,) int64 — `f(L)` for each trial
          - `is_expected` : (n_trials,) bool — `T == f(L)`
          - `p_bias` : float
          - `derangement` : tuple[int, ...]
          - `orientations_deg`

    References
    ----------
    Richter D, de Lange FP (2022) DOI 10.1093/oons/kvac013.
    Sprint 5e-Diag VERDICT_REPORT.md §Bug 1, §B5.
    """
    n_orients = len(orientations_deg)
    if n_orients != 6:
        raise ValueError(
            f"Richter paradigm requires 6 orientations, got {n_orients}"
        )
    if n_trials % n_orients != 0:
        raise ValueError(
            f"n_trials ({n_trials}) must be a multiple of {n_orients} "
            f"so each leader is presented {n_trials // n_orients} times"
        )
    if len(derangement) != n_orients:
        raise ValueError(
            f"derangement length {len(derangement)} != n_orients {n_orients}"
        )
    derangement = tuple(int(x) for x in derangement)
    if sorted(derangement) != list(range(n_orients)):
        raise ValueError(
            f"derangement {derangement} is not a permutation of 0..{n_orients-1}"
        )
    if any(derangement[i] == i for i in range(n_orients)):
        raise ValueError(
            f"derangement has a fixed point: {derangement} "
            f"(f(L) must differ from L for all L)"
        )
    p_other_total = 1.0 - float(p_bias)
    if not (0.0 < p_bias < 1.0):
        raise ValueError(f"p_bias must be in (0, 1), got {p_bias}")
    n_other = n_orients - 2  # exclude L itself AND f(L); spread 1-p_bias across these
    p_other = p_other_total / n_other    # per non-{L, f(L)} trailer
    if p_bias <= p_other:
        raise ValueError(
            f"p_bias={p_bias} is not greater than per-other mass "
            f"{p_other:.4f}; schedule would not be biased"
        )

    # Balanced leader sampling: each leader appears n_trials // n_orients times.
    reps_per_leader = n_trials // n_orients
    leaders = np.repeat(np.arange(n_orients, dtype=np.int64), reps_per_leader)
    rng.shuffle(leaders)

    # Per-trial trailer sampling from the biased multinomial.
    pairs = np.empty((n_trials, 2), dtype=np.int64)
    expected_trailer_idx = np.empty(n_trials, dtype=np.int64)
    is_expected = np.empty(n_trials, dtype=bool)
    # Precompute per-leader probability vector (shape (6,)).
    per_leader_p = np.zeros((n_orients, n_orients), dtype=np.float64)
    for L in range(n_orients):
        fL = derangement[L]
        per_leader_p[L, :] = 0.0
        for T in range(n_orients):
            if T == L:
                per_leader_p[L, T] = 0.0
            elif T == fL:
                per_leader_p[L, T] = p_bias
            else:
                per_leader_p[L, T] = p_other
        # Renormalise defensively (round-off).
        per_leader_p[L, :] /= per_leader_p[L, :].sum()

    for k, L in enumerate(leaders):
        L = int(L)
        T = int(rng.choice(n_orients, p=per_leader_p[L]))
        pairs[k] = (L, T)
        expected_trailer_idx[k] = derangement[L]
        is_expected[k] = (T == derangement[L])

    thetas_rad = np.deg2rad(np.asarray(orientations_deg, dtype=np.float64))

    items: List[TrialItem] = []
    for trial_idx, ((li, ti), exp_ti, is_exp) in enumerate(
        zip(pairs, expected_trailer_idx, is_expected)
    ):
        meta_base = {
            "trial": trial_idx,
            "leader_idx": int(li),
            "trailer_idx": int(ti),
            "expected_trailer_idx": int(exp_ti),
            "is_expected": bool(is_exp),
        }
        items.append(TrialItem(
            theta_rad=float(thetas_rad[li]),
            contrast=contrast,
            duration_ms=leader_ms,
            kind="leader",
            meta=dict(meta_base),
        ))
        items.append(TrialItem(
            theta_rad=float(thetas_rad[ti]),
            contrast=contrast,
            duration_ms=trailer_ms,
            kind="trailer",
            meta=dict(meta_base),
        ))
        if iti_ms > 0:
            items.append(TrialItem(
                theta_rad=None, contrast=0.0, duration_ms=iti_ms,
                kind="iti", meta={"trial": trial_idx},
            ))

    return TrialPlan(
        items=items,
        meta={"paradigm": "richter_biased",
              "n_trials": int(n_trials),
              "pairs": pairs,
              "expected_trailer_idx": expected_trailer_idx,
              "is_expected": is_expected,
              "p_bias": float(p_bias),
              "derangement": tuple(derangement),
              "orientations_deg": tuple(orientations_deg)},
    )


# --- Tang 2023 rotating sequence -------------------------------------------

def tang_rotating_sequence(
    rng: np.random.Generator,
    n_items: int = 1000,
    item_ms: float = TANG_ITEM_MS,
    block_len_range: Tuple[int, int] = TANG_BLOCK_LEN_RANGE,
    contrast: float = 1.0,
    orientations_deg: Sequence[float] = TANG_ORIENTATIONS_DEG,
) -> TrialPlan:
    """Tang 2023 rotating-orientation paradigm with per-block deviants.

    Mini-sequence structure (verified in docs/research_log.md):
      - Pick a random starting orientation from the 6-orientation set.
      - Pick a random rotation direction (CW vs CCW, +30° vs -30°).
      - Pick a block length uniformly from block_len_range (inclusive).
      - Emit `block_len` items rotating that direction at 30° steps,
        back-to-back (no ISI), each for `item_ms` (4 Hz if item_ms=250).
      - On the block's final item, replace with a deviant: an orientation
        drawn uniformly from the 5 that are NOT the one the rotation would
        have produced next.
      - Repeat mini-sequences until `n_items` items emitted; truncate
        partial final block.

    Parameters
    ----------
    rng : np.random.Generator
        Deterministic RNG (plan pre-reg: seed=42 for current stage).
    n_items : int
        Total items emitted. Default 1000 matches plan training scale.
    item_ms : float
        Per-item duration. 250 ms -> 4 Hz presentation rate.
    block_len_range : (int, int)
        Inclusive lower/upper bound on items per rotating block (deviant
        included in the count).
    contrast : float
    orientations_deg : Sequence[float]
        Must have length 6 (paper 30°-spaced set).

    Returns
    -------
    TrialPlan
        `meta` contains {"paradigm": "tang_rotating", "n_items",
        "deviant_mask": np.ndarray shape (n_items,) bool,
        "block_ids": np.ndarray shape (n_items,) int,
        "rotation_dir": np.ndarray shape (n_items,) int (+1/-1)}.

    References
    ----------
    Tang MF et al. (2023) Nat Commun 14:1196. PMID 36864037.
    """
    n_orients = len(orientations_deg)
    if n_orients != 6:
        raise ValueError(
            f"Tang paradigm requires 6 orientations, got {n_orients}"
        )
    blen_lo, blen_hi = block_len_range
    if not (blen_lo >= 2 and blen_hi >= blen_lo):
        raise ValueError(f"Invalid block_len_range {block_len_range}")

    thetas_rad = np.deg2rad(np.asarray(orientations_deg, dtype=np.float64))

    items: List[TrialItem] = []
    deviant_mask: List[bool] = []
    block_ids: List[int] = []
    rotation_dir: List[int] = []

    block_id = 0
    n_emitted = 0
    while n_emitted < n_items:
        start = int(rng.integers(0, n_orients))
        direction = int(rng.choice((-1, +1)))  # -1 = CCW (+30° -> -30°), +1 = CW
        block_len = int(rng.integers(blen_lo, blen_hi + 1))
        # Rotation positions 0..block_len-1; final item is a deviant.
        for k in range(block_len):
            if n_emitted >= n_items:
                break
            is_deviant = (k == block_len - 1)
            if is_deviant:
                # orientation that rotation would have produced next:
                expected = (start + direction * k) % n_orients
                # pick uniformly from the 5 remaining orientations
                options = np.array(
                    [o for o in range(n_orients) if o != expected],
                    dtype=np.int64,
                )
                idx = int(rng.choice(options))
            else:
                idx = (start + direction * k) % n_orients
            items.append(TrialItem(
                theta_rad=float(thetas_rad[idx]),
                contrast=contrast,
                duration_ms=item_ms,
                kind="deviant" if is_deviant else "rotating",
                meta={"block": block_id, "pos_in_block": k,
                      "block_len": block_len, "direction": direction,
                      "orient_idx": idx},
            ))
            deviant_mask.append(is_deviant)
            block_ids.append(block_id)
            rotation_dir.append(direction)
            n_emitted += 1
        block_id += 1

    return TrialPlan(
        items=items,
        meta={"paradigm": "tang_rotating",
              "n_items": len(items),
              "deviant_mask": np.asarray(deviant_mask, dtype=bool),
              "block_ids": np.asarray(block_ids, dtype=np.int64),
              "rotation_dir": np.asarray(rotation_dir, dtype=np.int64),
              "orientations_deg": tuple(orientations_deg)},
    )


# --- Kok 2012 (deferred) ---------------------------------------------------

def kok_trial(
    rng: np.random.Generator,
    expected_theta_rad: float,
    validity: float = 0.75,
    cue_ms: float = 500.0,
    gap_ms: float = 500.0,
    grating_ms: float = 500.0,
    iti_ms: float = 3500.0,
    include_omission: bool = False,
) -> TrialPlan:
    """Kok 2012 cue -> gap -> grating paradigm (PMID 22841311).

    NOT IMPLEMENTED — deferred to Stage-2 sprint when Kok-specific cue
    machinery (auditory analogue / feature cue) is wired into the H ring.
    """
    raise NotImplementedError(
        "kok_trial() deferred to Stage-2 sprint (requires cue pathway)"
    )


# --- legacy aliases (kept for backward compatibility) ----------------------

def richter_pair(rng, *args, **kwargs):  # pragma: no cover — legacy
    """Legacy alias. Use `richter_crossover_training_schedule` instead."""
    raise NotImplementedError(
        "richter_pair() renamed to richter_crossover_training_schedule()"
    )


def tang_rotating(rng, *args, **kwargs):  # pragma: no cover — legacy
    """Legacy alias. Use `tang_rotating_sequence` instead."""
    raise NotImplementedError(
        "tang_rotating() renamed to tang_rotating_sequence()"
    )


# --- self-check / smoke -----------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) drifting_grating_input basic sanity.
    theta = 0.0
    t, rates = drifting_grating_input(theta, contrast=1.0, duration_ms=250.0)
    assert rates.shape == (250, N_CHANNELS), rates.shape
    assert int(np.argmax(rates[0])) == 0
    assert np.allclose(rates[0], rates[-1])
    print(f"stimulus smoke: drifting_grating ch0={rates[0,0]:.2f} Hz")

    # 2) Richter schedule balanced, deterministic.
    plan_r = richter_crossover_training_schedule(rng, n_trials=360)
    assert plan_r.meta["paradigm"] == "richter_crossover"
    pairs = plan_r.meta["pairs"]
    assert pairs.shape == (360, 2)
    # Each (L, T) pair appears exactly 10 times.
    counts = np.zeros((6, 6), dtype=np.int64)
    for li, ti in pairs:
        counts[li, ti] += 1
    assert np.all(counts == 10), f"Unbalanced: min={counts.min()}, max={counts.max()}"
    # Timing: 500 + 500 + 5000 = 6000 ms per trial.
    assert abs(plan_r.total_ms - 360 * 6000.0) < 1e-6
    assert plan_r.n_items() == 360 * 3
    # Kinds: leader, trailer, iti alternating.
    kinds = [it.kind for it in plan_r.items[:6]]
    assert kinds == ["leader", "trailer", "iti",
                     "leader", "trailer", "iti"], kinds
    print(f"stimulus smoke: Richter schedule 360 trials, "
          f"{plan_r.total_ms/1000:.1f} s total, balanced 10 per pair")

    # 2b) Determinism under seed=42.
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    plan_a = richter_crossover_training_schedule(rng_a, n_trials=36)
    plan_b = richter_crossover_training_schedule(rng_b, n_trials=36)
    assert np.array_equal(plan_a.meta["pairs"], plan_b.meta["pairs"])
    print("stimulus smoke: Richter deterministic under seed=42")

    # 2c) Biased training schedule (Sprint 5e Fix A): empirical bias matches
    #     the target p_bias within sampling slack; same-orientation trailers
    #     never occur; deterministic under seed.
    rng_bias = np.random.default_rng(42)
    plan_bias = richter_biased_training_schedule(
        rng_bias, n_trials=360, p_bias=0.80,
    )
    assert plan_bias.meta["paradigm"] == "richter_biased"
    pairs_b = plan_bias.meta["pairs"]
    is_exp = plan_bias.meta["is_expected"]
    assert pairs_b.shape == (360, 2)
    # No same-orientation trailers under deranged-permutation schedule.
    assert np.all(pairs_b[:, 0] != pairs_b[:, 1]), (
        "same-orientation trailer leaked into biased schedule"
    )
    # Empirical P(T=f(L)|L) approx 0.80; allow ±0.05 slack on 360 trials.
    emp_bias = float(is_exp.mean())
    assert 0.75 <= emp_bias <= 0.85, (
        f"empirical bias {emp_bias:.3f} outside [0.75, 0.85]"
    )
    # max P(T|L) across all 6 leaders must be >= 0.70 (matches validator floor).
    from collections import Counter
    max_pt = 0.0
    min_ent = float("inf")
    for L in range(6):
        row_mask = pairs_b[:, 0] == L
        n_L = int(row_mask.sum())
        assert n_L > 0, f"leader {L} never appeared"
        ct = Counter(pairs_b[row_mask, 1].tolist())
        probs = np.array([ct.get(T, 0) / n_L for T in range(6)])
        max_pt = max(max_pt, float(probs.max()))
        p_nonzero = probs[probs > 0]
        ent = float(-(p_nonzero * np.log2(p_nonzero)).sum())
        min_ent = min(min_ent, ent)
    assert max_pt >= 0.70, f"max P(T|L) = {max_pt:.3f} < 0.70"
    assert min_ent <= (np.log2(6) - 0.5), (
        f"min_L entropy {min_ent:.3f} > log2(6) - 0.5"
    )
    # Determinism.
    rng_c = np.random.default_rng(42)
    rng_d = np.random.default_rng(42)
    pa = richter_biased_training_schedule(rng_c, n_trials=60)
    pb = richter_biased_training_schedule(rng_d, n_trials=60)
    assert np.array_equal(pa.meta["pairs"], pb.meta["pairs"])
    # Derangement validation: f(L) != L required.
    try:
        richter_biased_training_schedule(
            np.random.default_rng(0), n_trials=60, derangement=(0, 2, 3, 4, 5, 1),
        )
        raise AssertionError("fixed-point derangement should raise")
    except ValueError:
        pass
    print(f"stimulus smoke: biased schedule p_bias=0.80, empirical={emp_bias:.3f}, "
          f"max P(T|L)={max_pt:.3f}, min_L entropy={min_ent:.3f} bits")

    # 3) Tang rotating.
    rng2 = np.random.default_rng(42)
    plan_t = tang_rotating_sequence(rng2, n_items=1000)
    assert plan_t.meta["paradigm"] == "tang_rotating"
    assert plan_t.meta["n_items"] == 1000
    # Deviant rate is ~1 / mean_block_len. Expected ~ 1 / 7 = 14.3%.
    deviant_frac = plan_t.meta["deviant_mask"].mean()
    assert 0.10 < deviant_frac < 0.20, f"dev rate out of band: {deviant_frac:.3f}"
    # All item durations = 250 ms, 4 Hz.
    assert all(abs(it.duration_ms - 250.0) < 1e-6 for it in plan_t.items)
    assert abs(plan_t.total_ms - 1000 * 250.0) < 1e-6
    # Rotation block integrity: within a block, all non-deviant items
    # are separated by exactly +30° or -30° (mod 180°), in the block's
    # rotation direction.
    block_ids = plan_t.meta["block_ids"]
    rot_dir = plan_t.meta["rotation_dir"]
    dev_mask = plan_t.meta["deviant_mask"]
    for bid in np.unique(block_ids):
        idxs = np.where(block_ids == bid)[0]
        if len(idxs) < 2:
            continue
        direction = rot_dir[idxs[0]]
        for k in range(1, len(idxs) - 1):  # skip final deviant
            prev = plan_t.items[idxs[k - 1]]
            curr = plan_t.items[idxs[k]]
            step = np.rad2deg(curr.theta_rad - prev.theta_rad)
            step = ((step + 90) % 180) - 90   # wrap to [-90, 90]
            assert abs(step - direction * 30.0) < 1e-6, (
                f"block {bid} step {k}: got {step:.1f}° dir={direction}"
            )
    # The deviant must NOT equal the expected next orientation.
    for bid in np.unique(block_ids):
        idxs = np.where(block_ids == bid)[0]
        if not dev_mask[idxs[-1]]:
            continue
        dev_item = plan_t.items[idxs[-1]]
        # reconstruct expected
        first_item = plan_t.items[idxs[0]]
        direction = int(rot_dir[idxs[0]])
        first_theta_deg = float(np.rad2deg(first_item.theta_rad))
        n_pos = len(idxs) - 1   # position index for the deviant
        expected_deg = (first_theta_deg + direction * 30.0 * n_pos) % 180.0
        dev_deg = float(np.rad2deg(dev_item.theta_rad)) % 180.0
        assert abs(dev_deg - expected_deg) > 1e-6, (
            f"block {bid}: deviant equals expected ({dev_deg}, {expected_deg})"
        )
    print(f"stimulus smoke: Tang schedule 1000 items, "
          f"deviant frac = {deviant_frac:.3f}, block integrity OK")

    # 3b) Determinism under seed=42.
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    pa = tang_rotating_sequence(rng_a, n_items=100)
    pb = tang_rotating_sequence(rng_b, n_items=100)
    thetas_a = np.array([it.theta_rad for it in pa.items])
    thetas_b = np.array([it.theta_rad for it in pb.items])
    assert np.array_equal(thetas_a, thetas_b)
    print("stimulus smoke: Tang deterministic under seed=42")

    # 4) Legacy aliases raise.
    for fn, args in [(kok_trial, (rng, 0.0)),
                     (richter_pair, (rng,)),
                     (tang_rotating, (rng,))]:
        try:
            fn(*args)
            raise AssertionError(f"{fn.__name__} should raise")
        except NotImplementedError:
            pass
    print("stimulus smoke: PASS")
