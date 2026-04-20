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
    STAGE2_CUE_CHANNELS, STAGE2_CUE_ACTIVE_HZ,
)
from ..brian2_model.h_ring import N_CHANNELS as H_N_CHANNELS
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


def build_kok_schedule(
    cfg: KokConfig,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Assemble the trial list: 240 stim + 48 omission, balanced across cues.

    Each item is a dict with keys:
      - ``cue`` in {"A", "B"}
      - ``theta_rad`` : presented orientation (radians, float). NaN if omission.
      - ``expected_rad`` : cue-predicted orientation (radians).
      - ``condition``  : 1 = valid (shown = expected), 0 = invalid.
      - ``is_omission``: bool — True if grating epoch is blank.

    Balance
    -------
    Stim trials are equally split across cue A / cue B (120 each at the
    default n_stim_trials=240); within each cue, 75 % valid / 25 % invalid
    (90/30). Omission trials are split 24 / 24.

    Parameters
    ----------
    cfg : KokConfig
    rng : numpy Generator, optional
        Deterministic RNG for trial shuffling. Default: seed=cfg.seed.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

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

    schedule = build_kok_schedule(cfg, rng)
    n_trials = len(schedule)
    n_e = int(bundle.v1_ring.e.N)

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
        bundle.cue_off()
        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        net.run(cfg.gap_ms * ms)

        # --- grating epoch ---------------------------------------------
        if not omit:
            set_grating(bundle.v1_ring, theta_rad=theta, contrast=cfg.contrast)
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
    if not invalid_stim_mask.any():
        raise RuntimeError("no invalid stim trials produced")

    # ----- metric 1: mean amplitude (valid vs invalid) ------------------
    pop_mask = np.ones(n_e, dtype=bool)
    valid_amp = total_population_activity(
        trial_grating_counts[:, valid_stim_mask],
        pop_mask, window_ms=cfg.grating_ms,
    )
    invalid_amp = total_population_activity(
        trial_grating_counts[:, invalid_stim_mask],
        pop_mask, window_ms=cfg.grating_ms,
    )

    # ----- metric 2 (legacy): SVM validity decoder ---------------------
    # Kept for backward compat; main 5c R2 metric is orientation_mvpa below.
    X = trial_grating_counts[:, stim_mask].T                # (n_stim, n_e)
    y = cond_mask[stim_mask]
    svm_res = svm_decoding_accuracy(X, y, cv=5, seed=cfg.seed)

    # ----- Sprint 5c R2: orientation MVPA, valid vs invalid split ------
    # Map presented theta to a binary orientation label (45° vs 135°).
    expected_a_rad = _cue_expected_theta_rad("A")
    expected_b_rad = _cue_expected_theta_rad("B")
    orient_labels = np.full(n_trials, -1, dtype=np.int64)
    for i in range(n_trials):
        if not is_omission[i]:
            orient_labels[i] = _orientation_label(
                float(theta_per_trial[i]), expected_a_rad, expected_b_rad,
            )
    spike_vectors_all = trial_grating_counts.T               # (n_trials, n_e)
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

    # ----- metric 3: preference-rank suppression ------------------------
    pref_rad = v1_e_preferred_thetas(bundle.v1_ring)         # (n_e,)
    stim_counts = trial_grating_counts[:, stim_mask]
    stim_theta = theta_per_trial[stim_mask]
    stim_cond = cond_mask[stim_mask]
    pref_rank = suppression_vs_preference(
        stim_counts, pref_rad, stim_theta, stim_cond, n_bins=10,
    )

    # ----- metric 4: omission-subtracted response -----------------------
    omit_counts = trial_grating_counts[:, is_omission]
    valid_counts = trial_grating_counts[:, valid_stim_mask]
    omission_delta = omission_subtracted_response(valid_counts, omit_counts)

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
        },
        meta={
            "seed": int(cfg.seed),
            "n_trials": int(n_trials),
            "n_valid": int(valid_stim_mask.sum()),
            "n_invalid": int(invalid_stim_mask.sum()),
            "n_omission": int(is_omission.sum()),
            "config": cfg.__dict__,
            "bundle": {k: v for k, v in bundle.meta.items() if k != "config"},
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
