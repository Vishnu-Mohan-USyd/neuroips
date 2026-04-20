"""Kok-inspired passive cueing assay (Sprint 5a Step 2, task #27;
Sprint 5c R2 — orientation-MVPA decoder, reviewer rec 5c-4).

Paradigm (plan §3.5, Kok 2012 PMID 22841311 logic, not an exact replication)::

    cue (500 ms) -> gap (500 ms) -> grating (500 ms) -> ITI

    240 stim trials: 180 valid (cue predicts the shown grating orientation)
                      + 60 invalid (orthogonal orientation).
    48 omission trials: cue present, no grating in the expected window.

Cues are the Stage-2 eligibility-trace-learned projections::

    cue_A -> H channel 3  ( 45°)
    cue_B -> H channel 9  (135°)

Valid:   cue_A -> grating 45°,   cue_B -> grating 135°.
Invalid: cue_A -> grating 135°,  cue_B -> grating 45°.

Primary neuron-level metrics (assays.metrics signatures):

1. Mean V1_E amplitude during the grating epoch (valid vs invalid);
   :func:`total_population_activity`.
2. **Sprint 5c R2**: orientation-MVPA decoding (45° vs 135°) computed
   *separately* on valid vs invalid grating-evoked V1_E vectors. To keep
   class counts equal across the two splits, valid trials are
   sub-sampled down to the invalid-trial count (default 60: 30 of 45° +
   30 of 135°). Repeated 20 times with different seeds; report
   ``Δ_decoding = Acc_valid − Acc_invalid`` as the expectation-MVPA
   index, with a percentile bootstrap CI over the 20 sub-samples.
   Original validity-decoder (predict valid vs invalid label from the
   V1 vector) is retained as ``svm_validity_legacy`` for backward compat.
3. Preference-rank suppression (Δ(valid − invalid) vs rank bin on
   cells sorted by |θ_pref − θ_presented|); :func:`suppression_vs_preference`.
4. Omission-subtracted response (valid − omission per cell);
   :func:`omission_subtracted_response`.

All per-seed metrics include bootstrap 95 % CIs where the underlying
metric provides them.

The caller provides a frozen :class:`FrozenBundle` (runtime.py), or lets
this module build one at seed=42, r=1.0, g_total=1.0.

References
----------
- Kok P et al. (2012) PMID 22841311 — expectation suppression + sharpening.
- Plan §3.5 — Kok-logic passive cueing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs
from brian2 import seed as b2_seed

from .runtime import (
    FrozenBundle, build_frozen_network,
    set_grating, v1_e_preferred_thetas,
    snapshot_h_counts, preprobe_h_rate_hz,
    STAGE2_CUE_CHANNELS, STAGE2_CUE_ACTIVE_HZ,
)
from ..brian2_model.h_ring import N_CHANNELS as H_N_CHANNELS
from ..brian2_model.v1_ring import (
    N_CHANNELS as V1_N_CHANNELS,
    stimulus_tuning_profile,
)
from brian2 import Hz
from .metrics import (
    suppression_vs_preference,
    total_population_activity,
    omission_subtracted_response,
    svm_decoding_accuracy,
)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KokConfig:
    """Kok passive-cueing paradigm configuration.

    Defaults match the plan §3.5 paradigm and task #27 Step-2 specification.

    Sprint 5c R2 additions:
      ``mvpa_n_subsamples``  — number of random valid sub-samples for
                                Δ_decoding bootstrap (default 20).
      ``mvpa_n_bootstrap``   — bootstrap reps over the per-subsample
                                Δ_decoding distribution (default 1000).
      ``mvpa_cv``            — k-fold CV inside each MVPA decoder fit
                                (default 5).

    Sprint 5d SNR-probe knobs (diagnostic D6, failure-case D):
      ``contrast_multiplier``  — multiplies ``contrast`` during grating
                                 epoch (default 1.0 = no change). Smaller
                                 → weaker grating drive.
      ``input_noise_std_hz``   — Gaussian σ (Hz) added to per-channel
                                 stimulus rate on every grating-epoch set
                                 (default 0.0 = clean). Clamped ≥ 0.
      ``n_cells_subsampled``   — if set, random subset of V1 E cells used
                                 as SVM feature vector (default None =
                                 use all 192). Applies to both legacy
                                 validity decoder and orientation-MVPA.
      ``n_orientations``       — number of orientations for the decoder
                                 (default 2 = {45°, 135°}, existing
                                 behaviour). If > 2, must be 6 or 12;
                                 schedule switches to evenly-spaced
                                 orientations over 180°, cue logic is
                                 dropped (all trials treated as "valid"
                                 with no omissions), and the decoder runs
                                 a multi-class CV on the full stim set.
    """
    n_stim_trials: int = 240
    n_omission_trials: int = 48
    validity: float = 0.75           # fraction of stim trials that are valid
    cue_ms: float = 500.0
    gap_ms: float = 500.0
    grating_ms: float = 500.0
    iti_ms: float = 2000.0           # ~3.5 s total gap with cue onset included
    cue_rate_hz: float = STAGE2_CUE_ACTIVE_HZ
    contrast: float = 1.0
    seed: int = 42
    mvpa_n_subsamples: int = 20
    mvpa_n_bootstrap: int = 1000
    mvpa_cv: int = 5
    # -- Sprint 5d SNR-probe knobs (diagnostic D6) --------------------------
    contrast_multiplier: float = 1.0
    input_noise_std_hz: float = 0.0
    n_cells_subsampled: Optional[int] = None
    n_orientations: int = 2
    # -- Sprint 5d pre-probe H instrumentation (diagnostic D1/D2) -----------
    # Measured in the *last* `preprobe_window_ms` of the cue→gap interval,
    # i.e. just before grating onset. Only recorded if bundle was built
    # with ``with_preprobe_h_mon=True``; otherwise ignored.
    preprobe_window_ms: float = 100.0


@dataclass
class KokResult:
    """Kok passive-cueing assay output (Sprint 5a primary metrics +
    Sprint 5c R2 orientation-MVPA decoder)."""
    mean_amp: Dict[str, Any]              # {"valid": ..., "invalid": ...}
    svm: Dict[str, Any]                   # legacy validity decoder (5a primary)
    orientation_mvpa: Dict[str, Any]      # Sprint 5c R2 — Δ_decoding result
    pref_rank: Dict[str, Any]
    omission: np.ndarray                  # per-cell valid − omission
    raw: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

_CUE_A = "A"
_CUE_B = "B"


def _cue_expected_theta_rad(cue: str) -> float:
    """Return the cue's expected stimulus orientation (0..π)."""
    if cue == _CUE_A:
        return float(STAGE2_CUE_CHANNELS[0]) * (np.pi / H_N_CHANNELS)
    if cue == _CUE_B:
        return float(STAGE2_CUE_CHANNELS[1]) * (np.pi / H_N_CHANNELS)
    raise ValueError(f"cue must be 'A' or 'B', got {cue!r}")


def _cue_invalid_theta_rad(cue: str) -> float:
    """Return the cue's invalid (orthogonal) stimulus orientation."""
    return _cue_expected_theta_rad(_CUE_B if cue == _CUE_A else _CUE_A)


def _evenly_spaced_orientations_rad(n: int) -> np.ndarray:
    """Return n orientations evenly spaced over 180° in radians (0..π).

    n=2  -> [0°, 90°]   — note: this does NOT match default cue 45°/135°
                          grid; the default n_orientations=2 path keeps
                          the existing Stage-2 cue channels via the
                          separate schedule branch.
    n=6  -> 30° steps {0°, 30°, 60°, 90°, 120°, 150°}
    n=12 -> 15° steps {0°, 15°, 30°, …, 165°} (matches H-ring channels).
    """
    if n < 2:
        raise ValueError(f"n_orientations must be >= 2, got {n}")
    return np.linspace(0.0, np.pi, n, endpoint=False)


def build_kok_schedule(
    cfg: KokConfig,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Assemble the trial list.

    Default (``n_orientations == 2``): 240 stim + 48 omission, balanced
    across cues A/B, 75%/25% valid/invalid split. Backward-compat.

    SNR-probe (``n_orientations > 2``): cue logic dropped, all trials
    marked valid with no omissions, ``n_stim_trials`` distributed evenly
    across the ``n_orientations`` orientations (evenly spaced over 180°).
    For multi-class orientation decoding only — cue attributes A/B are
    still emitted alternately for bookkeeping but carry no semantics.

    Each item is a dict with keys:
      - ``cue`` in {"A", "B"}
      - ``theta_rad`` : presented orientation (radians, float). NaN if omission.
      - ``expected_rad`` : cue-predicted orientation (radians), or
                           ``theta_rad`` itself in the SNR branch.
      - ``condition``  : 1 = valid (shown = expected), 0 = invalid.
      - ``is_omission``: bool — True if grating epoch is blank.

    Parameters
    ----------
    cfg : KokConfig
    rng : numpy Generator, optional
        Deterministic RNG for trial shuffling. Default: seed=cfg.seed.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    # -- SNR-probe branch: multi-orientation, no cue logic ------------------
    n_orient = int(cfg.n_orientations)
    if n_orient != 2:
        if n_orient not in (6, 12):
            raise ValueError(
                f"n_orientations must be 2, 6, or 12; got {n_orient}"
            )
        n_stim = int(cfg.n_stim_trials)
        if n_stim % n_orient != 0:
            raise ValueError(
                f"n_stim_trials={n_stim} must be divisible by "
                f"n_orientations={n_orient}"
            )
        thetas = _evenly_spaced_orientations_rad(n_orient)
        per_orient = n_stim // n_orient
        items: List[Dict[str, Any]] = []
        for k, theta in enumerate(thetas):
            cue = _CUE_A if (k % 2 == 0) else _CUE_B
            for _ in range(per_orient):
                items.append({
                    "cue": cue,
                    "theta_rad": float(theta),
                    "expected_rad": float(theta),
                    "condition": 1,
                    "is_omission": False,
                })
        order = np.arange(len(items))
        rng.shuffle(order)
        return [items[i] for i in order]

    n_stim = int(cfg.n_stim_trials)
    n_om = int(cfg.n_omission_trials)
    if n_stim % 2 != 0:
        raise ValueError("n_stim_trials must be even (cue-balanced)")
    if n_om % 2 != 0:
        raise ValueError("n_omission_trials must be even (cue-balanced)")

    per_cue = n_stim // 2                        # 120
    n_valid_per = int(round(per_cue * cfg.validity))  # 90
    n_invalid_per = per_cue - n_valid_per             # 30
    if n_valid_per + n_invalid_per != per_cue:
        raise AssertionError("validity split arithmetic")

    items: List[Dict[str, Any]] = []

    # Stim trials: 90 valid + 30 invalid per cue.
    for cue in (_CUE_A, _CUE_B):
        expected = _cue_expected_theta_rad(cue)
        invalid = _cue_invalid_theta_rad(cue)
        for _ in range(n_valid_per):
            items.append({
                "cue": cue,
                "theta_rad": expected,
                "expected_rad": expected,
                "condition": 1,
                "is_omission": False,
            })
        for _ in range(n_invalid_per):
            items.append({
                "cue": cue,
                "theta_rad": invalid,
                "expected_rad": expected,
                "condition": 0,
                "is_omission": False,
            })
    # Omission trials: cue present, no grating. Condition is still "expected"
    # (cue predicts; the unknown is whether the stimulus arrives at all).
    per_cue_om = n_om // 2
    for cue in (_CUE_A, _CUE_B):
        expected = _cue_expected_theta_rad(cue)
        for _ in range(per_cue_om):
            items.append({
                "cue": cue,
                "theta_rad": float("nan"),
                "expected_rad": expected,
                "condition": 1,
                "is_omission": True,
            })

    order = np.arange(len(items))
    rng.shuffle(order)
    items = [items[i] for i in order]
    return items


# ---------------------------------------------------------------------------
# Assay runner
# ---------------------------------------------------------------------------

def _snapshot_counts(mon: SpikeMonitor) -> np.ndarray:
    """Return a copy of per-neuron cumulative spike counts."""
    return np.asarray(mon.count[:], dtype=np.int64).copy()


def _orientation_label(theta_rad: float, expected_a_rad: float,
                       expected_b_rad: float) -> int:
    """Map presented theta to a binary orientation label (0 = 45°, 1 = 135°)."""
    da = abs(theta_rad - expected_a_rad)
    db = abs(theta_rad - expected_b_rad)
    return 0 if da <= db else 1


def _orientation_label_multi(theta_rad: float, grid_rad: np.ndarray) -> int:
    """Map presented theta to a multi-class label 0..n_orientations-1.

    Nearest grid point on the ring (orientation is π-periodic, so wrap
    differences into (-π/2, π/2] before measuring).
    """
    d = theta_rad - grid_rad
    # wrap to (-π/2, π/2]
    d = (d + np.pi / 2) % np.pi - np.pi / 2
    return int(np.argmin(np.abs(d)))


def _set_grating_snr(
    v1,
    theta_rad: Optional[float],
    contrast: float,
    noise_std_hz: float,
    rng: Optional[np.random.Generator],
) -> None:
    """Set V1 stimulus with optional per-channel Gaussian noise.

    When ``noise_std_hz <= 0.0`` and ``contrast * <tuning profile>`` is
    the same as the default path, this is numerically identical to
    :func:`.runtime.set_grating` (to preserve SNR-default backward
    compat). When ``noise_std_hz > 0.0``, adds a fresh Gaussian draw to
    every channel's rate and clamps to ≥ 0 Hz. A blank epoch
    (``theta_rad is None`` or ``contrast <= 0``) skips the noise path
    entirely so ITI/gap windows remain silent.
    """
    if theta_rad is None or contrast <= 0.0:
        per_channel = np.zeros(V1_N_CHANNELS, dtype=np.float64)
    else:
        per_channel = stimulus_tuning_profile(
            float(theta_rad), v1.config,
        ).astype(np.float64) * float(contrast)
        if noise_std_hz > 0.0:
            assert rng is not None, "rng required when noise_std_hz > 0"
            per_channel = per_channel + rng.normal(
                0.0, float(noise_std_hz), size=per_channel.shape,
            )
            per_channel = np.maximum(per_channel, 0.0)
    rates = per_channel[v1.stim_channel]
    v1.stim.rates = rates * Hz


def _orientation_mvpa(
    spike_vectors: np.ndarray,           # (n_trials, n_cells)
    orient_labels: np.ndarray,           # (n_trials,) ∈ {0,1}
    valid_mask: np.ndarray,              # (n_trials,) bool
    invalid_mask: np.ndarray,            # (n_trials,) bool
    n_subsamples: int,
    n_bootstrap: int,
    cv: int,
    seed: int,
) -> Dict[str, Any]:
    """Sprint 5c R2 orientation-decoding split: Δ = Acc_valid − Acc_invalid.

    Decoder: linear SVM (5-fold stratified CV inside each fit), trained
    to predict orientation (0=45° / 1=135°) from the V1_E spike vector.

    Sub-sampling: invalid trials are typically ~60 (30 per orientation
    at default). We sub-sample valid trials down to the same per-class
    count so both decoders see equal numbers of trials per class.
    Repeat ``n_subsamples`` times with different sub-sample seeds.

    Δ_decoding = Acc_valid − Acc_invalid is computed per sub-sample;
    the mean is reported with a percentile bootstrap CI.
    """
    from .metrics import svm_decoding_accuracy
    rng = np.random.default_rng(seed)

    valid_idx = np.where(valid_mask)[0]
    invalid_idx = np.where(invalid_mask)[0]
    if invalid_idx.size < 4 or valid_idx.size < invalid_idx.size:
        raise RuntimeError(
            f"Need at least 4 invalid + ≥invalid valid trials for MVPA; "
            f"got valid={valid_idx.size} invalid={invalid_idx.size}"
        )
    inv_labels = orient_labels[invalid_idx]
    n_per_class = int(min(np.bincount(inv_labels.astype(np.int64),
                                      minlength=2)))
    if n_per_class < 2:
        raise RuntimeError(
            f"Each orientation class needs ≥2 invalid trials; got "
            f"per-class counts {np.bincount(inv_labels.astype(np.int64), minlength=2)}"
        )

    # Index of invalid trials per orientation class (kept fixed across reps).
    inv_by_class: List[np.ndarray] = [
        invalid_idx[inv_labels == c] for c in (0, 1)
    ]
    val_labels_full = orient_labels[valid_idx]
    val_by_class: List[np.ndarray] = [
        valid_idx[val_labels_full == c] for c in (0, 1)
    ]
    if any(c.size < n_per_class for c in val_by_class):
        raise RuntimeError(
            f"Not enough valid per class ({[c.size for c in val_by_class]}) "
            f"to subsample to {n_per_class} per class"
        )

    acc_valid_subs = np.zeros(n_subsamples, dtype=np.float64)
    acc_invalid_subs = np.zeros(n_subsamples, dtype=np.float64)
    delta_subs = np.zeros(n_subsamples, dtype=np.float64)

    # Invalid decoder: same trials each subsample (no random subsetting needed),
    # but we re-fit per subsample with seed offset so CV folds vary too.
    inv_idx_balanced = np.concatenate([
        inv_by_class[0][:n_per_class], inv_by_class[1][:n_per_class],
    ])
    Xinv = spike_vectors[inv_idx_balanced]
    yinv = orient_labels[inv_idx_balanced].astype(np.int64)

    for s in range(n_subsamples):
        sub_seed = int(seed + s + 1)
        # Sub-sample valid trials
        rng_s = np.random.default_rng(sub_seed)
        sel_v = []
        for c in (0, 1):
            pool = val_by_class[c]
            pick = rng_s.choice(pool.size, size=n_per_class, replace=False)
            sel_v.append(pool[pick])
        val_idx_balanced = np.concatenate(sel_v)
        Xv = spike_vectors[val_idx_balanced]
        yv = orient_labels[val_idx_balanced].astype(np.int64)
        rv = svm_decoding_accuracy(Xv, yv, cv=cv, seed=sub_seed)
        ri = svm_decoding_accuracy(Xinv, yinv, cv=cv, seed=sub_seed)
        acc_valid_subs[s] = rv["accuracy"]
        acc_invalid_subs[s] = ri["accuracy"]
        delta_subs[s] = rv["accuracy"] - ri["accuracy"]

    # Bootstrap percentile CI on the mean of Δ_decoding across subsamples.
    boot = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_subsamples, size=n_subsamples)
        boot[b] = float(delta_subs[idx].mean())
    delta_mean = float(delta_subs.mean())
    delta_ci = (float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)))

    return {
        "delta_decoding": delta_mean,
        "delta_decoding_ci": delta_ci,
        "acc_valid_mean": float(acc_valid_subs.mean()),
        "acc_invalid_mean": float(acc_invalid_subs.mean()),
        "acc_valid_subs": acc_valid_subs,
        "acc_invalid_subs": acc_invalid_subs,
        "delta_subs": delta_subs,
        "n_subsamples": int(n_subsamples),
        "n_bootstrap": int(n_bootstrap),
        "n_per_class_subsample": int(n_per_class),
        "n_invalid_per_class": int(min(c.size for c in inv_by_class)),
        "cv": int(cv),
    }


def run_kok_passive(
    bundle: Optional[FrozenBundle] = None,
    cfg: Optional[KokConfig] = None,
    *,
    seed: int = 42,
    r: float = 1.0,
    g_total: float = 1.0,
    verbose: bool = False,
) -> KokResult:
    """Run the Kok-logic passive cueing assay end-to-end.

    Parameters
    ----------
    bundle : FrozenBundle, optional
        Prebuilt frozen network (from :func:`build_frozen_network`).
        Must have ``h_kind='hr'`` and ``with_cue=True``. Built fresh if None.
    cfg : KokConfig, optional
        Paradigm config. Defaults to ``KokConfig(seed=seed)``.
    seed : int
        RNG seed for trial shuffling and Brian2 state (default 42).
    r, g_total : float
        Feedback balance ratio + total. Only used if ``bundle is None``.
    verbose : bool
        If True, print progress every ~50 trials.

    Returns
    -------
    KokResult
    """
    cfg = cfg or KokConfig(seed=seed)
    if cfg.seed != seed:
        # Keep cfg.seed authoritative if caller passed an explicit cfg.
        seed = cfg.seed

    if bundle is None:
        bundle = build_frozen_network(
            h_kind="hr", seed=seed, r=r, g_total=g_total, with_cue=True,
        )
    elif bundle.h_kind != "hr" or bundle.cue_A is None:
        raise ValueError(
            "Kok assay requires an H_R bundle built with with_cue=True; "
            f"got h_kind={bundle.h_kind!r}, cue={'present' if bundle.cue_A else 'absent'}"
        )

    # Sprint 5c context_only mode: V1->H is silenced during the grating
    # window so H carries only the prior built up over cue+gap. Restored at
    # ITI start so the next cue+gap can still drive H. Mode "continuous"
    # (default) leaves the pathway always-on; "off" means no V1->H built.
    v1_to_h_mode = bundle.meta.get("v1_to_h_mode", "continuous")
    context_only = (v1_to_h_mode == "context_only")
    if context_only and bundle.v1_to_h is None:
        raise RuntimeError(
            "Kok assay context_only mode requires bundle.v1_to_h built"
        )

    # Determinism
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    # Dedicated RNG for per-trial stimulus noise; sits downstream of `rng`
    # but is never advanced when ``input_noise_std_hz == 0`` so SNR-default
    # numerics stay bit-exact with the legacy path.
    noise_rng = (
        np.random.default_rng(seed + 9001)
        if cfg.input_noise_std_hz > 0.0 else None
    )

    schedule = build_kok_schedule(cfg, rng)
    n_trials = len(schedule)
    n_e = int(bundle.v1_ring.e.N)
    eff_contrast = float(cfg.contrast) * float(cfg.contrast_multiplier)

    # --- Pre-probe H instrumentation (diagnostic D1/D2) -----------------
    # Present only when the bundle was built with ``with_preprobe_h_mon``.
    # We split the gap epoch into (gap_ms - preprobe_window_ms) + (pre-
    # probe window), snapshotting H_E cumulative counts at the boundary.
    # Window duration & placement: last `preprobe_window_ms` of gap →
    # matches Sprint 5d spec "Kok: final 100 ms of cue-stim gap".
    preprobe_on = bundle.h_e_mon is not None
    preprobe_win_ms = float(cfg.preprobe_window_ms) if preprobe_on else 0.0
    if preprobe_on and preprobe_win_ms >= cfg.gap_ms:
        raise ValueError(
            f"preprobe_window_ms={preprobe_win_ms} must be < gap_ms="
            f"{cfg.gap_ms} to leave a pre-probe block inside the gap"
        )
    n_h_channels = int(bundle.h_ring.e_channel.max()) + 1
    if preprobe_on:
        h_preprobe_rate_hz_arr = np.zeros(
            (n_trials, n_h_channels), dtype=np.float64,
        )
    else:
        h_preprobe_rate_hz_arr = None

    e_mon = SpikeMonitor(bundle.v1_ring.e, name=f"kok_e_mon_seed{seed}")
    net = Network(*bundle.groups, e_mon)

    # Allocate per-trial storage
    trial_grating_counts = np.zeros((n_e, n_trials), dtype=np.int64)
    trial_cue_epoch_counts = np.zeros((n_e, n_trials), dtype=np.int64)
    cue_per_trial = np.empty(n_trials, dtype="U1")
    theta_per_trial = np.empty(n_trials, dtype=np.float64)
    expected_per_trial = np.empty(n_trials, dtype=np.float64)
    cond_mask = np.empty(n_trials, dtype=np.int64)
    is_omission = np.empty(n_trials, dtype=bool)

    for k, item in enumerate(schedule):
        cue = item["cue"]
        theta = item["theta_rad"]
        expected = item["expected_rad"]
        valid = item["condition"]
        omit = item["is_omission"]

        cue_per_trial[k] = cue
        theta_per_trial[k] = theta
        expected_per_trial[k] = expected
        cond_mask[k] = valid
        is_omission[k] = omit

        bundle.reset_all()

        # --- cue epoch -------------------------------------------------
        bundle.cue_on(cue, rate_hz=cfg.cue_rate_hz)
        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        cnt_pre_cue = _snapshot_counts(e_mon)
        net.run(cfg.cue_ms * ms)
        cnt_post_cue = _snapshot_counts(e_mon)
        trial_cue_epoch_counts[:, k] = cnt_post_cue - cnt_pre_cue

        # --- gap epoch -------------------------------------------------
        # With pre-probe instrumentation on, split gap into
        # (gap_ms - preprobe_win_ms) then (preprobe_win_ms) so we can
        # snapshot H_E spike counts at the boundary. Only the final
        # `preprobe_win_ms` chunk is the "pre-probe prior" window.
        bundle.cue_off()
        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        if preprobe_on:
            pre_gap_ms = cfg.gap_ms - preprobe_win_ms
            if pre_gap_ms > 0.0:
                net.run(pre_gap_ms * ms)
            cnt_before_pp = snapshot_h_counts(bundle)
            net.run(preprobe_win_ms * ms)
            cnt_after_pp = snapshot_h_counts(bundle)
            h_preprobe_rate_hz_arr[k, :] = preprobe_h_rate_hz(
                cnt_before_pp, cnt_after_pp,
                bundle.h_ring, preprobe_win_ms,
            )
        else:
            net.run(cfg.gap_ms * ms)

        # --- grating epoch ---------------------------------------------
        # SNR-probe path: if contrast_multiplier != 1.0 OR input_noise > 0,
        # use the local helper so we can inject per-channel Gaussian noise
        # and apply the contrast multiplier. Default case (cm=1, noise=0)
        # routes through set_grating and is bit-exact with the legacy path.
        if not omit:
            if cfg.contrast_multiplier != 1.0 or cfg.input_noise_std_hz > 0.0:
                _set_grating_snr(
                    bundle.v1_ring, theta_rad=theta,
                    contrast=eff_contrast,
                    noise_std_hz=float(cfg.input_noise_std_hz),
                    rng=noise_rng,
                )
            else:
                set_grating(bundle.v1_ring, theta_rad=theta,
                            contrast=cfg.contrast)
        else:
            set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        if context_only:
            bundle.v1_to_h.set_active(False)
        cnt_pre_g = _snapshot_counts(e_mon)
        net.run(cfg.grating_ms * ms)
        cnt_post_g = _snapshot_counts(e_mon)
        trial_grating_counts[:, k] = cnt_post_g - cnt_pre_g

        # --- ITI -------------------------------------------------------
        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        if context_only:
            bundle.v1_to_h.set_active(True)
        net.run(cfg.iti_ms * ms)

        if verbose and (k + 1) % 50 == 0:
            print(f"[kok] trial {k+1}/{n_trials} done "
                  f"(cue={cue} cond={valid} omit={int(omit)})")

    # ----- derived masks -------------------------------------------------
    stim_mask = ~is_omission
    valid_stim_mask = stim_mask & (cond_mask == 1)
    invalid_stim_mask = stim_mask & (cond_mask == 0)

    if not valid_stim_mask.any():
        raise RuntimeError("no valid stim trials produced")
    # Multi-orientation branch has no invalid trials by construction.
    if cfg.n_orientations == 2 and not invalid_stim_mask.any():
        raise RuntimeError("no invalid stim trials produced")

    # ----- metric 1: mean amplitude (valid vs invalid) ------------------
    pop_mask = np.ones(n_e, dtype=bool)
    valid_amp = total_population_activity(
        trial_grating_counts[:, valid_stim_mask],
        pop_mask, window_ms=cfg.grating_ms,
    )
    if invalid_stim_mask.any():
        invalid_amp = total_population_activity(
            trial_grating_counts[:, invalid_stim_mask],
            pop_mask, window_ms=cfg.grating_ms,
        )
    else:
        invalid_amp = {}

    # ----- cell subsampling for SVM features ---------------------------
    # Applied to both legacy validity decoder and orientation-MVPA so the
    # knob cleanly reduces all decoders' signal-to-noise. Subsample seed
    # is deterministic from cfg.seed (distinct stream so it doesn't disturb
    # trial shuffling).
    if cfg.n_cells_subsampled is not None:
        n_sub = int(cfg.n_cells_subsampled)
        if n_sub < 2 or n_sub > n_e:
            raise ValueError(
                f"n_cells_subsampled={n_sub} must be in [2, {n_e}]"
            )
        sub_rng = np.random.default_rng(int(cfg.seed) + 7001)
        sub_cells = np.sort(sub_rng.choice(n_e, size=n_sub, replace=False))
    else:
        sub_cells = np.arange(n_e, dtype=np.int64)
    feature_counts = trial_grating_counts[sub_cells, :]  # (n_features, n_trials)

    # ----- metric 2 (legacy): SVM validity decoder ---------------------
    # Kept for backward compat; main 5c R2 metric is orientation_mvpa below.
    # Multi-orientation branch has no invalid class -> skip legacy decoder.
    if cfg.n_orientations == 2:
        X = feature_counts[:, stim_mask].T               # (n_stim, n_features)
        y = cond_mask[stim_mask]
        svm_res = svm_decoding_accuracy(X, y, cv=5, seed=cfg.seed)
    else:
        svm_res = {
            "accuracy": float("nan"),
            "accuracy_ci": (float("nan"), float("nan")),
            "fold_accuracy": np.full(5, np.nan),
            "n_trials": int(stim_mask.sum()),
            "n_classes": 0,
            "skipped_reason": "n_orientations != 2 (SNR multi-class branch)",
        }

    # ----- Orientation MVPA: 2-class (valid-vs-invalid) or multi-class --
    if cfg.n_orientations == 2:
        # Sprint 5c R2: binary orientation label (45° vs 135°).
        expected_a_rad = _cue_expected_theta_rad("A")
        expected_b_rad = _cue_expected_theta_rad("B")
        orient_labels = np.full(n_trials, -1, dtype=np.int64)
        for i in range(n_trials):
            if not is_omission[i]:
                orient_labels[i] = _orientation_label(
                    float(theta_per_trial[i]), expected_a_rad, expected_b_rad,
                )
        spike_vectors_all = feature_counts.T              # (n_trials, n_features)
        orient_mvpa = _orientation_mvpa(
            spike_vectors=spike_vectors_all,
            orient_labels=orient_labels,
            valid_mask=valid_stim_mask,
            invalid_mask=invalid_stim_mask,
            n_subsamples=int(cfg.mvpa_n_subsamples),
            n_bootstrap=int(cfg.mvpa_n_bootstrap),
            cv=int(cfg.mvpa_cv),
            seed=int(cfg.seed),
        )
    else:
        # Sprint 5d SNR-probe: multi-class orientation decoder over the
        # full stim set (no valid/invalid split in this branch).
        grid_rad = _evenly_spaced_orientations_rad(int(cfg.n_orientations))
        orient_labels = np.full(n_trials, -1, dtype=np.int64)
        for i in range(n_trials):
            if not is_omission[i]:
                orient_labels[i] = _orientation_label_multi(
                    float(theta_per_trial[i]), grid_rad,
                )
        spike_vectors_all = feature_counts.T              # (n_trials, n_features)
        Xm = spike_vectors_all[stim_mask]
        ym = orient_labels[stim_mask].astype(np.int64)
        mc_res = svm_decoding_accuracy(
            Xm, ym, cv=int(cfg.mvpa_cv), seed=int(cfg.seed),
        )
        orient_mvpa = {
            # chance-above accuracy is the SNR probe's meaningful quantity.
            "accuracy": float(mc_res["accuracy"]),
            "accuracy_ci": tuple(mc_res["accuracy_ci"]),
            "fold_accuracy": np.asarray(mc_res["fold_accuracy"]),
            "n_classes": int(cfg.n_orientations),
            "n_trials": int(stim_mask.sum()),
            "n_per_class": int(stim_mask.sum()) // int(cfg.n_orientations),
            "chance_acc": 1.0 / float(cfg.n_orientations),
            "cv": int(cfg.mvpa_cv),
            "mode": "multi_class",
            # Back-compat shape fields (2-class caller expects these keys):
            "delta_decoding": float("nan"),
            "delta_decoding_ci": (float("nan"), float("nan")),
            "acc_valid_mean": float(mc_res["accuracy"]),
            "acc_invalid_mean": float("nan"),
        }

    # ----- metric 3: preference-rank suppression ------------------------
    # Only defined for the 2-class valid/invalid paradigm; in the SNR
    # multi-orientation branch the "preference-rank vs valid-invalid" axis
    # is undefined, so we return an empty dict.
    pref_rad = v1_e_preferred_thetas(bundle.v1_ring)         # (n_e,)
    if cfg.n_orientations == 2:
        stim_counts = trial_grating_counts[:, stim_mask]
        stim_theta = theta_per_trial[stim_mask]
        stim_cond = cond_mask[stim_mask]
        pref_rank = suppression_vs_preference(
            stim_counts, pref_rad, stim_theta, stim_cond, n_bins=10,
        )
    else:
        pref_rank = {"skipped_reason": "multi_orientation_snr_branch"}

    # ----- metric 4: omission-subtracted response -----------------------
    # SNR multi-orientation branch has no omissions by construction.
    if is_omission.any() and valid_stim_mask.any():
        omit_counts = trial_grating_counts[:, is_omission]
        valid_counts = trial_grating_counts[:, valid_stim_mask]
        omission_delta = omission_subtracted_response(valid_counts, omit_counts)
    else:
        omission_delta = np.zeros(n_e, dtype=np.float64)

    return KokResult(
        mean_amp={"valid": valid_amp, "invalid": invalid_amp},
        svm=svm_res,
        orientation_mvpa=orient_mvpa,
        pref_rank=pref_rank,
        omission=omission_delta,
        raw={
            "trial_grating_counts": trial_grating_counts,
            "trial_cue_epoch_counts": trial_cue_epoch_counts,
            "cue_per_trial": cue_per_trial,
            "theta_per_trial": theta_per_trial,
            "expected_per_trial": expected_per_trial,
            "cond_mask": cond_mask,
            "is_omission": is_omission,
            "orient_labels": orient_labels,
            "pref_rad": pref_rad,
            "sub_cells": sub_cells,
            # Sprint 5d D1/D2 pre-probe H rate (if instrumented; else None).
            "h_preprobe_rate_hz": h_preprobe_rate_hz_arr,
            "preprobe_window_ms": float(preprobe_win_ms),
        },
        meta={
            "seed": int(cfg.seed),
            "n_trials": int(n_trials),
            "n_valid": int(valid_stim_mask.sum()),
            "n_invalid": int(invalid_stim_mask.sum()),
            "n_omission": int(is_omission.sum()),
            "config": cfg.__dict__,
            "bundle": {k: v for k, v in bundle.meta.items() if k != "config"},
            # Sprint 5d SNR-probe provenance:
            "snr_contrast_multiplier": float(cfg.contrast_multiplier),
            "snr_input_noise_std_hz": float(cfg.input_noise_std_hz),
            "snr_n_cells_used": int(sub_cells.size),
            "snr_n_orientations": int(cfg.n_orientations),
            "snr_effective_contrast": float(eff_contrast),
        },
    )


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke: tiny n to prove the pipeline runs end-to-end.
    cfg = KokConfig(
        n_stim_trials=40, n_omission_trials=4,
        iti_ms=500.0, grating_ms=500.0, cue_ms=500.0, gap_ms=500.0,
        mvpa_n_subsamples=4, mvpa_n_bootstrap=200,
        seed=42,
    )
    result = run_kok_passive(cfg=cfg, verbose=True)
    print(f"[kok smoke] n_trials={result.meta['n_trials']} "
          f"valid={result.meta['n_valid']} invalid={result.meta['n_invalid']} "
          f"om={result.meta['n_omission']}")
    print(f"  valid amp   = {result.mean_amp['valid']['total_rate_hz']:.3f} Hz "
          f"CI {result.mean_amp['valid']['total_rate_hz_ci']}")
    print(f"  invalid amp = {result.mean_amp['invalid']['total_rate_hz']:.3f} Hz "
          f"CI {result.mean_amp['invalid']['total_rate_hz_ci']}")
    print(f"  svm validity acc (legacy) = {result.svm['accuracy']:.3f}")
    om = result.orientation_mvpa
    print(f"  Δ_decoding (valid-invalid) = {om['delta_decoding']:+.3f} "
          f"CI {om['delta_decoding_ci']}  "
          f"(acc_v={om['acc_valid_mean']:.3f}, acc_i={om['acc_invalid_mean']:.3f})")
    print(f"  pref bin0 Δ = {result.pref_rank['bin_delta'][0]:.3f}")
    print(f"  omission-sub mean = {float(result.omission.mean()):.3f}")
