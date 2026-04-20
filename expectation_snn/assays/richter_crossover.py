"""Richter 6 × 6 cross-over assay (Sprint 5a Step 3, task #27).

Sprint 5c R1 design (reviewer rec 5c-3 — addresses C2 same-θ adaptation
confound): deranged-permutation D=[1,2,3,4,5,0]::

    leader (500 ms, θ_L) -> trailer (500 ms, θ_T) -> ITI (1500 ms).

    expected   (6 pairs):  θ_T = θ_L + 30°        (Δθ_step = 1)
        Stage-1 trained the H_R rotational expectation; expected pairs
        always present a one-step rotation. This breaks the same-θ
        adaptation confound the original 5a design suffered (where
        expected ≡ θ_L = θ_T forced ~1 s of identical orientation).
    unexpected (24 pairs): θ_T = θ_L + k·30°,   k ∈ {2,3,4,5}
        Four offsets per leader, balanced across orientations.

    Default schedule:
        180 expected  =  6 pair types × 30 reps
        192 unexpected = 24 pair types ×  8 reps
        372 trials total.

V1 responses are measured on the trailer epoch; the leader serves as the
context that H_R feedback rides on. The Δθ-step covariate (1..5) is
written into each schedule item so downstream metrics can stratify by
expected vs unexpected step (and by step distance within unexpected).

Primary neuron-level metrics (assays.metrics):

1. Preference-rank suppression (:func:`suppression_vs_preference`,
   Δ(expected − unexpected) vs rank bin on cells ranked by
   |θ_pref − θ_T|).
2. Feature-distance surface (:func:`suppression_vs_distance_from_expected`,
   8 × 8 grid over d(pref, θ_L) × d(pref, θ_T)).
3. Cell-type × channel-distance gain matrix (3 cell types × 3 distances),
   computed with :func:`total_population_activity`.
4. Center-vs-flank redistribution (derived from metric 1).

Secondary: 6-family pseudo-voxel forward model
(:func:`pseudo_voxel_forward_model`).

References
----------
- Richter D, de Lange FP (2022) DOI 10.1093/oons/kvac013.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs
from brian2 import seed as b2_seed

from .runtime import (
    FrozenBundle, build_frozen_network,
    set_grating, v1_e_preferred_thetas,
)
from ..brian2_model.v1_ring import N_CHANNELS as V1_N_CHANNELS
from ..brian2_model.stimulus import RICHTER_ORIENTATIONS_DEG
from .metrics import (
    suppression_vs_preference,
    suppression_vs_distance_from_expected,
    pseudo_voxel_forward_model,
    _VOXEL_MODEL_FAMILIES,
)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RichterConfig:
    """Richter test-phase configuration (Sprint 5c R1: deranged-permutation).

    The schedule is determined by ``reps_expected`` and ``reps_unexpected``;
    ``n_trials`` is derived. Defaults:

      - 6 expected pair types  × 30 reps  = 180 trials   (Δθ_step = 1)
      - 24 unexpected pair types × 8 reps = 192 trials   (Δθ_step ∈ {2,3,4,5})
      - total                            = 372 trials.
    """
    leader_ms: float = 500.0
    trailer_ms: float = 500.0
    iti_ms: float = 1500.0
    contrast: float = 1.0
    n_voxels: int = 4                  # Richter pseudo-voxel default
    seed: int = 42
    reps_expected: int = 30            # × 6 expected pair types = 180
    reps_unexpected: int = 8           # × 24 unexpected pair types = 192

    @property
    def n_trials(self) -> int:
        return int(self.reps_expected) * 6 + int(self.reps_unexpected) * 24


@dataclass
class RichterResult:
    """Richter cross-over assay output."""
    pref_rank: Dict[str, Any]
    feature_distance: Dict[str, Any]
    cell_type_gain: Dict[str, Any]
    center_vs_flank: Dict[str, Any]
    voxel_forward: Dict[str, Any]        # 6-family secondary
    dtheta_stratified: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

def _richter_thetas_rad() -> np.ndarray:
    """Return the 6 Richter orientations as radians (0..π)."""
    return np.deg2rad(np.asarray(RICHTER_ORIENTATIONS_DEG, dtype=np.float64))


def build_richter_schedule(
    cfg: RichterConfig,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Assemble the deranged-permutation Richter test schedule.

    Sprint 5c R1 (reviewer rec 5c-3): the original 5a design used
    expected ≡ θ_L = θ_T, which left V1 driven by the *same* orientation
    for ~1 s straight (leader+trailer). At 30°-spaced channels that
    produces strong same-channel adaptation that is indistinguishable
    from "expectation suppression". The deranged-permutation design
    removes that confound: expected pairs always rotate 30° between
    leader and trailer.

    Schedule (defaults, 372 trials):
      - 6 expected pair types  (Δθ_step = 1)              × 30 reps = 180
      - 24 unexpected pair types (Δθ_step ∈ {2, 3, 4, 5})  × 8 reps = 192

    Each item is a dict with keys:
      - ``theta_L``     : leader orientation (rad)
      - ``theta_T``     : trailer orientation (rad)
      - ``condition``   : 1 = expected (Δθ_step=1), 0 = unexpected (step≥2)
      - ``pair_id``     : int in 0..29 (canonical pair index)
      - ``dtheta_step`` : int in {1,2,3,4,5} — the step distance (mod 6)
                         from leader to trailer on the 30°-spaced ring.

    Pair construction: leader index i ∈ {0..5}, trailer = (i+k) mod 6,
    k ∈ {1} for expected, k ∈ {2,3,4,5} for unexpected.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    thetas = _richter_thetas_rad()                      # (6,)
    n_orients = len(thetas)
    if n_orients != 6:
        raise ValueError(f"Richter requires 6 orientations, got {n_orients}")

    pairs: List[Dict[str, Any]] = []
    pid = 0
    # Expected: Δθ_step = 1 (single-step rotation)
    for i in range(n_orients):
        theta_L = float(thetas[i])
        theta_T = float(thetas[(i + 1) % n_orients])
        pairs.append({
            "theta_L": theta_L, "theta_T": theta_T,
            "condition": 1, "pair_id": pid, "dtheta_step": 1,
        })
        pid += 1
    # Unexpected: Δθ_step ∈ {2, 3, 4, 5}
    for i in range(n_orients):
        theta_L = float(thetas[i])
        for k in (2, 3, 4, 5):
            theta_T = float(thetas[(i + k) % n_orients])
            pairs.append({
                "theta_L": theta_L, "theta_T": theta_T,
                "condition": 0, "pair_id": pid, "dtheta_step": k,
            })
            pid += 1

    items: List[Dict[str, Any]] = []
    reps_exp = int(cfg.reps_expected)
    reps_unexp = int(cfg.reps_unexpected)
    for p in pairs:
        reps = reps_exp if p["condition"] == 1 else reps_unexp
        for _ in range(reps):
            items.append(dict(p))
    expected_total = reps_exp * 6 + reps_unexp * 24
    if len(items) != expected_total or expected_total != cfg.n_trials:
        raise ValueError(
            f"schedule items {len(items)} != n_trials {cfg.n_trials} "
            f"(reps_expected={reps_exp}, reps_unexpected={reps_unexp})"
        )
    order = np.arange(len(items))
    rng.shuffle(order)
    return [items[i] for i in order]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(mon: SpikeMonitor) -> np.ndarray:
    return np.asarray(mon.count[:], dtype=np.int64).copy()


def _nearest_channel(theta_rad: float, n_channels: int = V1_N_CHANNELS) -> int:
    """Return channel index whose preferred orientation is closest to theta."""
    chans = np.arange(n_channels) * (np.pi / n_channels)
    d = np.abs(chans - float(theta_rad))
    d = np.minimum(d, np.pi - d)
    return int(np.argmin(d))


def _voxel_spatial_bins(
    channel_per_cell: np.ndarray, n_voxels: int, n_channels: int,
) -> np.ndarray:
    """Assign each cell to one of ``n_voxels`` pseudo-voxels by channel.

    Partitions the 0..n_channels-1 channel index uniformly across voxels
    (contiguous channel blocks per voxel).
    """
    return (channel_per_cell.astype(np.int64) * n_voxels // n_channels).astype(np.int64)


# ---------------------------------------------------------------------------
# Specialized metric builders
# ---------------------------------------------------------------------------

def _cell_type_gain_matrix(
    counts_by_pop: Dict[str, np.ndarray],
    channel_by_pop: Dict[str, np.ndarray],
    trailer_theta_per_trial: np.ndarray,
    condition_mask: np.ndarray,
    window_ms: float,
    n_channels: int = V1_N_CHANNELS,
) -> Dict[str, Any]:
    """3 populations × 3 channel-distance gain matrix (Richter metric 3).

    For each cell population ``p`` (E/SOM/PV), split cells by channel
    distance to each trial's trailer orientation — {local(=0), nbr(=1),
    far(≥2)} — and report mean Hz per (pop × distance × condition) bin.

    PV is assigned channel −10 (out-of-ring); it therefore always lands in
    'far'. This matches the broad-pool biology where PV is not tuned.

    Returns
    -------
    dict with keys
      - ``rate_hz``   : (3 pops, 3 dists, 2 conds) mean Hz over (cell, trial)
      - ``delta_hz``  : (3 pops, 3 dists) expected − unexpected
      - ``pops``      : tuple ('E', 'SOM', 'PV')
      - ``dists``     : tuple ('local', 'nbr', 'far')
    """
    pops = ("E", "SOM", "PV")
    dist_labels = ("local", "nbr", "far")

    n_trials = trailer_theta_per_trial.shape[0]
    valid_mask = condition_mask == 1
    invalid_mask = condition_mask == 0

    rate = np.zeros((3, 3, 2), dtype=np.float64)

    trailer_ch = np.array([
        _nearest_channel(float(trailer_theta_per_trial[t]), n_channels)
        for t in range(n_trials)
    ], dtype=np.int64)

    for pi, p in enumerate(pops):
        counts = counts_by_pop[p]                # (n_cells_p, n_trials)
        ch_cell = channel_by_pop[p]              # (n_cells_p,)
        d_cell = np.abs(
            ch_cell[:, None].astype(np.int64) - trailer_ch[None, :]
        )                                         # (n_cells_p, n_trials)
        d_cell = np.minimum(d_cell, n_channels - d_cell)
        bin_cell = np.where(d_cell == 0, 0, np.where(d_cell == 1, 1, 2))

        for di in range(3):
            bin_mask = bin_cell == di                              # (n_cells_p, n_trials)
            spikes_per_trial_bin = (counts * bin_mask).sum(axis=0)  # (n_trials,)
            cells_per_trial_bin = bin_mask.sum(axis=0)              # (n_trials,)
            safe = np.maximum(cells_per_trial_bin, 1)
            per_trial_rate = spikes_per_trial_bin / safe / (window_ms / 1000.0)
            # Trials with 0 cells in this bin get per_trial_rate=0 but we
            # exclude those from the mean to avoid biasing toward 0.
            has_cells = cells_per_trial_bin > 0
            for ci, cmask in enumerate((valid_mask, invalid_mask)):
                sel = cmask & has_cells
                if sel.any():
                    rate[pi, di, ci] = float(per_trial_rate[sel].mean())

    delta = rate[:, :, 0] - rate[:, :, 1]

    return {
        "rate_hz": rate,
        "delta_hz": delta,
        "pops": pops,
        "dists": dist_labels,
    }


def _dtheta_step_stratified(
    trailer_counts_e: np.ndarray,        # (n_e, n_trials)
    pref_rad: np.ndarray,                # (n_e,)
    theta_T: np.ndarray,                 # (n_trials,)
    cond_mask: np.ndarray,               # (n_trials,)
    dtheta_step: np.ndarray,             # (n_trials,) ∈ {1..5}
    window_ms: float,
) -> Dict[str, Any]:
    """Per-Δθ_step preference-rank suppression report (Sprint 5c R1).

    For each unexpected step k ∈ {2,3,4,5}, compute the
    suppression-vs-preference Δ between expected (Δθ_step=1) and the
    given unexpected k. Returns the per-step ``bin_delta`` (10 bins) and
    the matched center/flank/redist scalars per step.

    Bins are computed by :func:`suppression_vs_preference` with the
    expected step (k=1) recoded as ``condition=1`` and the chosen
    unexpected k as ``condition=0``.
    """
    bin_deltas: Dict[int, np.ndarray] = {}
    redists: Dict[int, Dict[str, float]] = {}
    n_per_step: Dict[int, int] = {}
    expected_mask = (dtheta_step == 1) & (cond_mask == 1)
    n_per_step[1] = int(expected_mask.sum())
    for k in (2, 3, 4, 5):
        unexp_mask = (dtheta_step == k) & (cond_mask == 0)
        n_per_step[k] = int(unexp_mask.sum())
        sel = expected_mask | unexp_mask
        if not sel.any() or not unexp_mask.any():
            bin_deltas[k] = np.full(10, np.nan, dtype=np.float64)
            redists[k] = {"center_delta": float("nan"),
                          "flank_delta": float("nan"),
                          "redist": float("nan")}
            continue
        sub_cond = np.where(expected_mask[sel], 1, 0).astype(np.int64)
        pr = suppression_vs_preference(
            trailer_counts_e[:, sel], pref_rad,
            theta_T[sel], sub_cond, n_bins=10,
        )
        bd = np.asarray(pr["bin_delta"], dtype=np.float64)
        bin_deltas[k] = bd
        n_bins = bd.size
        redists[k] = {
            "center_delta": float(bd[0]),
            "flank_delta": float(bd[n_bins // 2:].mean()) if n_bins > 1 else 0.0,
            "redist": float(bd[0] - (bd[n_bins // 2:].mean() if n_bins > 1 else 0.0)),
        }
    return {
        "bin_delta_by_step": {int(k): bin_deltas[k] for k in (2, 3, 4, 5)},
        "redist_by_step": redists,
        "n_trials_per_step": {int(k): int(n_per_step.get(k, 0))
                              for k in (1, 2, 3, 4, 5)},
        "window_ms": float(window_ms),
    }


def _center_vs_flank_redistribution(pref_rank: Dict[str, Any]) -> Dict[str, Any]:
    """Compact redistribution readout from the preference-rank Δ per bin.

    - ``center_delta`` : Δ at bin 0 (most-preferred decile).
    - ``flank_delta``  : mean Δ across bins in the last half (bins ≥ n/2).
    - ``redist``       : center − flank (positive → expectation concentrates
                         response at preferred; negative → expectation
                         redistributes toward flanks).
    """
    bd = np.asarray(pref_rank["bin_delta"], dtype=np.float64)
    n_bins = bd.size
    center = float(bd[0])
    flank = float(bd[n_bins // 2:].mean()) if n_bins > 1 else 0.0
    return {
        "center_delta": center,
        "flank_delta": flank,
        "redist": center - flank,
        "n_bins": int(n_bins),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_richter_crossover(
    bundle: Optional[FrozenBundle] = None,
    cfg: Optional[RichterConfig] = None,
    *,
    seed: int = 42,
    r: float = 1.0,
    g_total: float = 1.0,
    verbose: bool = False,
) -> RichterResult:
    """Run the Richter cross-over assay end-to-end.

    Parameters
    ----------
    bundle : FrozenBundle, optional
        H_R-backed bundle (no cue needed). Built fresh if None.
    cfg : RichterConfig, optional
    seed : int
    r, g_total : float
        Feedback balance ratio / total. Only used if bundle is None.
    verbose : bool

    Returns
    -------
    RichterResult
    """
    cfg = cfg or RichterConfig(seed=seed)
    if cfg.seed != seed:
        seed = cfg.seed

    if bundle is None:
        bundle = build_frozen_network(
            h_kind="hr", seed=seed, r=r, g_total=g_total, with_cue=False,
        )
    elif bundle.h_kind != "hr":
        raise ValueError(f"Richter assay requires h_kind='hr', got {bundle.h_kind!r}")

    # Sprint 5c context_only mode: silence V1->H during the trailer so the
    # trailer-window H rate reflects only carry-over from the leader (a true
    # prior signal). Restored at ITI start.
    v1_to_h_mode = bundle.meta.get("v1_to_h_mode", "continuous")
    context_only = (v1_to_h_mode == "context_only")
    if context_only and bundle.v1_to_h is None:
        raise RuntimeError(
            "Richter assay context_only mode requires bundle.v1_to_h built"
        )

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    schedule = build_richter_schedule(cfg, rng)
    n_trials = len(schedule)
    n_e = int(bundle.v1_ring.e.N)
    n_som = int(bundle.v1_ring.som.N)
    n_pv = int(bundle.v1_ring.pv.N)

    e_mon = SpikeMonitor(bundle.v1_ring.e, name=f"rich_e_mon_seed{seed}")
    som_mon = SpikeMonitor(bundle.v1_ring.som, name=f"rich_som_mon_seed{seed}")
    pv_mon = SpikeMonitor(bundle.v1_ring.pv, name=f"rich_pv_mon_seed{seed}")
    net = Network(*bundle.groups, e_mon, som_mon, pv_mon)

    trailer_counts_e = np.zeros((n_e, n_trials), dtype=np.int64)
    trailer_counts_som = np.zeros((n_som, n_trials), dtype=np.int64)
    trailer_counts_pv = np.zeros((n_pv, n_trials), dtype=np.int64)
    leader_counts_e = np.zeros((n_e, n_trials), dtype=np.int64)

    theta_L = np.zeros(n_trials, dtype=np.float64)
    theta_T = np.zeros(n_trials, dtype=np.float64)
    cond_mask = np.zeros(n_trials, dtype=np.int64)
    pair_id = np.zeros(n_trials, dtype=np.int64)
    dtheta_step = np.zeros(n_trials, dtype=np.int64)

    for k, item in enumerate(schedule):
        theta_L[k] = item["theta_L"]
        theta_T[k] = item["theta_T"]
        cond_mask[k] = item["condition"]
        pair_id[k] = item["pair_id"]
        dtheta_step[k] = item["dtheta_step"]

        bundle.reset_all()

        # --- leader epoch --------------------------------------------------
        set_grating(bundle.v1_ring, theta_rad=item["theta_L"], contrast=cfg.contrast)
        pre_e = _snapshot(e_mon)
        net.run(cfg.leader_ms * ms)
        leader_counts_e[:, k] = _snapshot(e_mon) - pre_e

        # --- trailer epoch -------------------------------------------------
        set_grating(bundle.v1_ring, theta_rad=item["theta_T"], contrast=cfg.contrast)
        if context_only:
            bundle.v1_to_h.set_active(False)
        pre_e = _snapshot(e_mon)
        pre_som = _snapshot(som_mon)
        pre_pv = _snapshot(pv_mon)
        net.run(cfg.trailer_ms * ms)
        trailer_counts_e[:, k] = _snapshot(e_mon) - pre_e
        trailer_counts_som[:, k] = _snapshot(som_mon) - pre_som
        trailer_counts_pv[:, k] = _snapshot(pv_mon) - pre_pv

        # --- ITI -----------------------------------------------------------
        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        if context_only:
            bundle.v1_to_h.set_active(True)
        net.run(cfg.iti_ms * ms)

        if verbose and (k + 1) % 60 == 0:
            print(f"[richter] trial {k+1}/{n_trials} "
                  f"(pair={item['pair_id']} cond={item['condition']})")

    pref_rad = v1_e_preferred_thetas(bundle.v1_ring)              # (n_e,)
    channel_e = bundle.v1_ring.e_channel.astype(np.int64)
    channel_som = bundle.v1_ring.som_channel.astype(np.int64)
    # V1 PV is a broad pool without tuning; use −10 as an out-of-ring
    # placeholder so PV cells land in the 'far' bin.
    channel_pv_placeholder = np.full(n_pv, -10, dtype=np.int64)

    # ---- metric 1: preference-rank suppression on V1 E -----------------
    pref_rank = suppression_vs_preference(
        trailer_counts_e, pref_rad, theta_T, cond_mask, n_bins=10,
    )

    # ---- metric 2: feature-distance surface ----------------------------
    feat_dist = suppression_vs_distance_from_expected(
        trailer_counts_e, pref_rad,
        expected_theta_per_trial=theta_L,
        presented_theta_per_trial=theta_T,
        grid_bins=(8, 8),
    )

    # ---- metric 3: cell-type × channel-distance gain -------------------
    ctg = _cell_type_gain_matrix(
        counts_by_pop={
            "E": trailer_counts_e,
            "SOM": trailer_counts_som,
            "PV": trailer_counts_pv,
        },
        channel_by_pop={
            "E": channel_e,
            "SOM": channel_som,
            "PV": channel_pv_placeholder,
        },
        trailer_theta_per_trial=theta_T,
        condition_mask=cond_mask,
        window_ms=cfg.trailer_ms,
        n_channels=V1_N_CHANNELS,
    )

    # ---- metric 4: center-vs-flank redistribution ----------------------
    cvf = _center_vs_flank_redistribution(pref_rank)

    # ---- Sprint 5c R1: Δθ-step stratified report ----------------------
    dtheta_strat = _dtheta_step_stratified(
        trailer_counts_e, pref_rad, theta_T, cond_mask, dtheta_step,
        window_ms=cfg.trailer_ms,
    )

    # ---- secondary: pseudo-voxel 6-family predictions ------------------
    # Build per-theta (presented) average counts from expected trials only.
    thetas_unique = _richter_thetas_rad()
    n_unique = thetas_unique.size
    counts_by_theta = np.zeros((n_e, n_unique), dtype=np.float64)
    for ti, theta in enumerate(thetas_unique):
        sel = (np.abs(theta_T - theta) < 1e-9) & (cond_mask == 1)
        if sel.any():
            counts_by_theta[:, ti] = trailer_counts_e[:, sel].mean(axis=1)
    voxel_bins = _voxel_spatial_bins(channel_e, cfg.n_voxels, V1_N_CHANNELS)
    voxel_fwd: Dict[str, Any] = {}
    for fam in _VOXEL_MODEL_FAMILIES:
        voxel_fwd[fam] = pseudo_voxel_forward_model(
            counts_by_theta, voxel_bins, fam, thetas_unique, effect_size=0.2,
        )

    return RichterResult(
        pref_rank=pref_rank,
        feature_distance=feat_dist,
        cell_type_gain=ctg,
        center_vs_flank=cvf,
        voxel_forward=voxel_fwd,
        dtheta_stratified=dtheta_strat,
        raw={
            "trailer_counts_e": trailer_counts_e,
            "trailer_counts_som": trailer_counts_som,
            "trailer_counts_pv": trailer_counts_pv,
            "leader_counts_e": leader_counts_e,
            "theta_L": theta_L,
            "theta_T": theta_T,
            "cond_mask": cond_mask,
            "pair_id": pair_id,
            "dtheta_step": dtheta_step,
            "pref_rad": pref_rad,
        },
        meta={
            "seed": int(cfg.seed),
            "n_trials": int(n_trials),
            "n_expected": int((cond_mask == 1).sum()),
            "n_unexpected": int((cond_mask == 0).sum()),
            "n_per_step": dtheta_strat["n_trials_per_step"],
            "config": cfg.__dict__,
            "bundle": {k: v for k, v in bundle.meta.items() if k != "config"},
        },
    )


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = RichterConfig(
        reps_expected=2, reps_unexpected=1,
        leader_ms=300.0, trailer_ms=300.0, iti_ms=300.0,
        seed=42,
    )
    r = run_richter_crossover(cfg=cfg, verbose=True)
    print(f"[richter smoke] n_trials={r.meta['n_trials']}  "
          f"exp={r.meta['n_expected']}  unexp={r.meta['n_unexpected']}")
    print(f"  per-step n trials = {r.meta['n_per_step']}")
    print(f"  pref-rank bin0 Δ = {r.pref_rank['bin_delta'][0]:.3f}")
    print(f"  feat-dist grid shape = {r.feature_distance['grid'].shape}")
    print(f"  cell-type Δ E/SOM/PV × local/nbr/far =\n"
          f"    {r.cell_type_gain['delta_hz']}")
    print(f"  center-vs-flank redist = {r.center_vs_flank['redist']:+.3f}")
    for k, rd in r.dtheta_stratified["redist_by_step"].items():
        print(f"  step k={k} redist = {rd['redist']:+.3f}")
    print(f"  voxel families = {sorted(r.voxel_forward)}")
