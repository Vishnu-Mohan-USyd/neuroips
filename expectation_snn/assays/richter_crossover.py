"""Richter 6 × 6 cross-over assay (Sprint 5a Step 3, task #27).

Paradigm (plan §3.6, Richter 2022 logic)::

    leader (500 ms, θ_L) -> trailer (500 ms, θ_T) -> ITI (1500 ms).

    Test phase only. Stage 1 H_R was trained on balanced 6 × 6 pairs.
    In the test, we use 12 distinct (θ_L, θ_T) pair types at 50 % reliability:

        expected   (6 pairs): θ_L = θ_T        ("same-orientation continuity")
        unexpected (6 pairs): θ_L = θ_T + π/2  ("orthogonal violation")

    360 trials total = 30 reps × 12 pair types.

V1 responses are measured on the trailer epoch; the leader serves as the
context that H_R feedback rides on.

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
    """Richter test-phase configuration."""
    n_trials: int = 360
    reps_per_pair: int = 30            # 30 × 12 pair types = 360
    leader_ms: float = 500.0
    trailer_ms: float = 500.0
    iti_ms: float = 1500.0
    contrast: float = 1.0
    n_voxels: int = 4                  # Richter pseudo-voxel default
    seed: int = 42


@dataclass
class RichterResult:
    """Richter cross-over assay output."""
    pref_rank: Dict[str, Any]
    feature_distance: Dict[str, Any]
    cell_type_gain: Dict[str, Any]
    center_vs_flank: Dict[str, Any]
    voxel_forward: Dict[str, Any]        # 6-family secondary
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
    """Assemble the 360-trial test schedule.

    12 pair types × 30 reps (at defaults). Each item is a dict with keys:
      - ``theta_L``  : leader orientation (rad)
      - ``theta_T``  : trailer orientation (rad)
      - ``condition``: 1 = expected (θ_L = θ_T), 0 = unexpected (θ_L = θ_T + π/2)
      - ``pair_id``  : int in 0..11 (canonical pair index)

    Pair ordering: [exp_0, ..., exp_5, unexp_0, ..., unexp_5], then shuffled.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    thetas = _richter_thetas_rad()                      # (6,)
    n_orients = len(thetas)

    pairs: List[Dict[str, Any]] = []
    for i in range(n_orients):
        theta_T = float(thetas[i])
        # Expected: θ_L = θ_T
        theta_L_exp = theta_T
        pairs.append({
            "theta_L": theta_L_exp, "theta_T": theta_T,
            "condition": 1, "pair_id": i,
        })
    for i in range(n_orients):
        theta_T = float(thetas[i])
        # Unexpected: θ_L = θ_T + π/2 — shift by 3 bins (90° / 30° step) on
        # the 6-orientation grid, wrapping mod 6.
        theta_L_unexp = float(thetas[(i + 3) % n_orients])
        pairs.append({
            "theta_L": theta_L_unexp, "theta_T": theta_T,
            "condition": 0, "pair_id": n_orients + i,
        })

    n_pair = len(pairs)                        # 12
    reps = int(cfg.reps_per_pair)
    if reps * n_pair != cfg.n_trials:
        raise ValueError(
            f"n_trials={cfg.n_trials} must = reps_per_pair ({reps}) × "
            f"n_pairs ({n_pair})"
        )
    items: List[Dict[str, Any]] = []
    for p in pairs:
        for _ in range(reps):
            items.append(dict(p))
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

    for k, item in enumerate(schedule):
        theta_L[k] = item["theta_L"]
        theta_T[k] = item["theta_T"]
        cond_mask[k] = item["condition"]
        pair_id[k] = item["pair_id"]

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
        raw={
            "trailer_counts_e": trailer_counts_e,
            "trailer_counts_som": trailer_counts_som,
            "trailer_counts_pv": trailer_counts_pv,
            "leader_counts_e": leader_counts_e,
            "theta_L": theta_L,
            "theta_T": theta_T,
            "cond_mask": cond_mask,
            "pair_id": pair_id,
            "pref_rad": pref_rad,
        },
        meta={
            "seed": int(cfg.seed),
            "n_trials": int(n_trials),
            "n_expected": int((cond_mask == 1).sum()),
            "n_unexpected": int((cond_mask == 0).sum()),
            "config": cfg.__dict__,
            "bundle": {k: v for k, v in bundle.meta.items() if k != "config"},
        },
    )


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = RichterConfig(
        n_trials=24, reps_per_pair=2,
        leader_ms=300.0, trailer_ms=300.0, iti_ms=300.0,
        seed=42,
    )
    r = run_richter_crossover(cfg=cfg, verbose=True)
    print(f"[richter smoke] n_trials={r.meta['n_trials']}  "
          f"exp={r.meta['n_expected']}  unexp={r.meta['n_unexpected']}")
    print(f"  pref-rank bin0 Δ = {r.pref_rank['bin_delta'][0]:.3f}")
    print(f"  feat-dist grid shape = {r.feature_distance['grid'].shape}")
    print(f"  cell-type Δ E/SOM/PV × local/nbr/far =\n"
          f"    {r.cell_type_gain['delta_hz']}")
    print(f"  center-vs-flank redist = {r.center_vs_flank['redist']:+.3f}")
    print(f"  voxel families = {sorted(r.voxel_forward)}")
