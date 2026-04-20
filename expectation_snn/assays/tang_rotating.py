"""Tang-like rotating-orientation assay (Sprint 5a Step 4, task #27;
Sprint 5c R3 — +Random block + Δθ_prev covariate, reviewer rec 5c-1).

Paradigm (plan §3.7, Tang 2023 logic)::

    Two back-to-back blocks (default 500 items each, 250 ms per item):

      1) Random  block — IID-uniform draws from the 6 orientations;
                          NO rotational structure, NO deviants. Acts as
                          a baseline against which the rotating-block
                          deviant gain can be normalised.
      2) Rotating block — original Tang grammar (rotating blocks of
                          length 5-9, ±30°, last item per block replaced
                          by a deviant).

    Each item is labelled with one of three conditions:

      "random"            — drawn from the random block.
      "rotating_expected" — drawn from a rotating block, not a deviant.
      "rotating_deviant"  — drawn from a rotating block, is a deviant.

    Each item's per-trial covariate ``dtheta_prev_step`` records the
    (signed-magnitude) step distance from the previous item's
    orientation, k ∈ {0,1,2,3} (mod 6, wrapped to 0..3 magnitude).

Items ride on a frozen H_T ring pre-trained (Stage 1 H_T) on the
rotating grammar.

Primary neuron-level metrics (assays.metrics):

1. **Sprint 5c R3** matched-θ rate per condition (Random / Expected /
   Deviant), with per-Δθ_prev_step stratification.
2. Per-cell gain at θ_stim (deviant − expected, matched-θ only) — kept
   from 5a as a sanity check.
3. Population decoding (5-fold linear SVM on deviant vs expected,
   rotating-block items only).
4. Laminar profile: V1_E mean Hz per condition.

Secondary: FWHM via :func:`tuning_fit` (pre-registered null per plan H3).

References
----------
- Tang MF et al. (2023) Nat Commun 14:1196, PMID 36864037.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs
from brian2 import seed as b2_seed

from .runtime import (
    FrozenBundle, build_frozen_network,
    set_grating, v1_e_preferred_thetas,
)
from ..brian2_model.v1_ring import N_CHANNELS as V1_N_CHANNELS
from ..brian2_model.stimulus import (
    tang_rotating_sequence, TANG_ORIENTATIONS_DEG, TANG_ITEM_MS,
    TANG_BLOCK_LEN_RANGE,
)
from .metrics import (
    omission_subtracted_response,
    svm_decoding_accuracy,
    total_population_activity,
    tuning_fit,
)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TangConfig:
    """Tang rotating-orientation paradigm configuration.

    Sprint 5c R3 schedule:
      - ``n_random``   IID items from 6 orientations (no deviants)
      - ``n_rotating`` items via the original rotating grammar
      ``n_items`` is derived as their sum.

    Set ``n_random=0`` to recover the legacy 5a single-block design.
    """
    n_random: int = 500
    n_rotating: int = 500
    item_ms: float = TANG_ITEM_MS
    block_len_range: tuple = TANG_BLOCK_LEN_RANGE
    contrast: float = 1.0
    presettle_ms: float = 500.0          # warm-up before first item
    seed: int = 42

    @property
    def n_items(self) -> int:
        return int(self.n_random) + int(self.n_rotating)


@dataclass
class TangResult:
    """Tang rotating assay output (Sprint 5a primary metrics +
    Sprint 5c R3 Random-block + Δθ_prev covariate)."""
    cell_gain: Dict[str, Any]            # per-cell matched-θ deviant − expected
    svm: Dict[str, Any]                  # pop decoding (rotating block only)
    laminar: Dict[str, Any]              # V1_E mean Hz per condition
    tuning: Dict[str, Any]               # FWHM secondary (von Mises fits)
    three_condition: Dict[str, Any] = field(default_factory=dict)   # R3 primary
    dtheta_prev: Dict[str, Any] = field(default_factory=dict)       # R3 covariate
    raw: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(mon: SpikeMonitor) -> np.ndarray:
    return np.asarray(mon.count[:], dtype=np.int64).copy()


def _tang_thetas_rad() -> np.ndarray:
    return np.deg2rad(np.asarray(TANG_ORIENTATIONS_DEG, dtype=np.float64))


def _nearest_tang_idx(theta_rad: float, thetas_tang: np.ndarray) -> int:
    d = np.abs(thetas_tang - float(theta_rad))
    d = np.minimum(d, np.pi - d)
    return int(np.argmin(d))


def _theta_step_distance(theta_a: float, theta_b: float,
                         thetas_tang: np.ndarray) -> int:
    """Minimum step distance (mod 6) between two thetas on the 30°-grid.

    Returns k ∈ {0,1,2,3} (since steps 4,5 wrap to 2,1 by symmetry on
    the 0..π ring with 6 evenly-spaced bins).
    """
    n = thetas_tang.size                                 # 6
    ia = _nearest_tang_idx(theta_a, thetas_tang)
    ib = _nearest_tang_idx(theta_b, thetas_tang)
    raw = abs(ia - ib) % n
    return int(min(raw, n - raw))


def _build_random_block_plan(rng: np.random.Generator, n_items: int,
                              item_ms: float, contrast: float,
                              thetas_rad: np.ndarray):
    """Return a TrialPlan-shape (items + meta) for an IID random block."""
    from ..brian2_model.stimulus import TrialItem, TrialPlan
    n_orients = thetas_rad.size
    items = []
    for _ in range(n_items):
        idx = int(rng.integers(0, n_orients))
        items.append(TrialItem(
            theta_rad=float(thetas_rad[idx]), contrast=contrast,
            duration_ms=item_ms, kind="random",
            meta={"random": True, "orient_idx": idx},
        ))
    return TrialPlan(
        items=items,
        meta={"paradigm": "tang_random",
              "n_items": int(n_items),
              "deviant_mask": np.zeros(n_items, dtype=bool),
              "block_ids": np.full(n_items, -1, dtype=np.int64),
              "rotation_dir": np.zeros(n_items, dtype=np.int64)},
    )


def _condition_per_item(deviant_mask: np.ndarray,
                        is_random: np.ndarray) -> np.ndarray:
    """Return string-coded condition per item: 'random'|'rotating_expected'|
    'rotating_deviant'. Returned as integer codes 0/1/2 for storage."""
    cond = np.zeros(deviant_mask.size, dtype=np.int64)
    cond[is_random] = 0                                # random
    rot = ~is_random
    cond[rot & ~deviant_mask] = 1                       # rotating_expected
    cond[rot & deviant_mask] = 2                        # rotating_deviant
    return cond


_TANG_COND_NAMES = ("random", "rotating_expected", "rotating_deviant")


def _three_condition_matched_rate(
    counts_per_item: np.ndarray,         # (n_e, n_items)
    pref_rad: np.ndarray,                # (n_e,)
    theta_per_item: np.ndarray,          # (n_items,)
    cond_codes: np.ndarray,              # (n_items,) ∈ {0,1,2}
    window_ms: float,
) -> Dict[str, Any]:
    """Per-cell matched-θ Hz per condition. Sprint 5c R3 primary metric.

    For each cell c (preferring channel ch_c) and each condition, take
    the mean spikes/sec across items whose θ_presented falls on ch_c.
    Returns per-cell rates per condition + the population summaries.
    """
    n_e, n_items = counts_per_item.shape
    win_s = window_ms / 1000.0
    chans = np.arange(V1_N_CHANNELS) * (np.pi / V1_N_CHANNELS)
    d_item_chan = np.abs(theta_per_item[:, None] - chans[None, :])
    d_item_chan = np.minimum(d_item_chan, np.pi - d_item_chan)
    item_chan = np.argmin(d_item_chan, axis=1)
    d_cell_chan = np.abs(pref_rad[:, None] - chans[None, :])
    d_cell_chan = np.minimum(d_cell_chan, np.pi - d_cell_chan)
    cell_chan = np.argmin(d_cell_chan, axis=1)

    per_cell_rates = np.zeros((n_e, 3), dtype=np.float64)   # 3 conditions
    n_per_cell_per_cond = np.zeros((n_e, 3), dtype=np.int64)

    for c in range(n_e):
        matched = item_chan == cell_chan[c]
        for k in range(3):
            sel = matched & (cond_codes == k)
            n = int(sel.sum())
            n_per_cell_per_cond[c, k] = n
            if n > 0:
                per_cell_rates[c, k] = counts_per_item[c, sel].mean() / win_s

    # Population mean per condition (over cells with data in *all* 3 conds).
    valid_cells = (n_per_cell_per_cond > 0).all(axis=1)
    from .metrics import _bootstrap_mean_ci
    per_cond_summary: Dict[str, Any] = {}
    for k, name in enumerate(_TANG_COND_NAMES):
        if valid_cells.any():
            mu, ci = _bootstrap_mean_ci(per_cell_rates[valid_cells, k])
        else:
            mu, ci = float("nan"), (float("nan"), float("nan"))
        per_cond_summary[name] = {"mean_rate_hz": mu, "ci": ci}

    deltas = {}
    for k in (1, 2):       # rotating_expected, rotating_deviant
        d = per_cell_rates[:, k] - per_cell_rates[:, 0]   # vs random
        if valid_cells.any():
            mu, ci = _bootstrap_mean_ci(d[valid_cells])
        else:
            mu, ci = float("nan"), (float("nan"), float("nan"))
        deltas[f"{_TANG_COND_NAMES[k]}_minus_random"] = {
            "mean_delta_hz": mu, "ci": ci,
        }
    d_dev_exp = per_cell_rates[:, 2] - per_cell_rates[:, 1]
    if valid_cells.any():
        mu, ci = _bootstrap_mean_ci(d_dev_exp[valid_cells])
    else:
        mu, ci = float("nan"), (float("nan"), float("nan"))
    deltas["deviant_minus_expected"] = {"mean_delta_hz": mu, "ci": ci}

    return {
        "per_cond": per_cond_summary,
        "deltas": deltas,
        "per_cell_rates": per_cell_rates,
        "n_items_per_cell_per_cond": n_per_cell_per_cond,
        "n_cells_with_data_all_conds": int(valid_cells.sum()),
        "cond_names": _TANG_COND_NAMES,
    }


def _dtheta_prev_stratified(
    counts_per_item: np.ndarray,         # (n_e, n_items)
    pref_rad: np.ndarray,                # (n_e,)
    theta_per_item: np.ndarray,          # (n_items,)
    cond_codes: np.ndarray,              # (n_items,) ∈ {0,1,2}
    dtheta_prev_step: np.ndarray,        # (n_items,) ∈ {-1,0,1,2,3} (-1 = first item)
    window_ms: float,
) -> Dict[str, Any]:
    """Matched-θ rate stratified by Δθ_prev_step ∈ {0,1,2,3} per condition.

    Sprint 5c R3 covariate (reviewer concern: deviant gain may merely
    reflect a larger-than-expected Δθ from the previous item).

    Returns nested dict keyed by condition name → step → rate dict.
    """
    n_e, n_items = counts_per_item.shape
    win_s = window_ms / 1000.0
    chans = np.arange(V1_N_CHANNELS) * (np.pi / V1_N_CHANNELS)
    d_item_chan = np.abs(theta_per_item[:, None] - chans[None, :])
    d_item_chan = np.minimum(d_item_chan, np.pi - d_item_chan)
    item_chan = np.argmin(d_item_chan, axis=1)
    d_cell_chan = np.abs(pref_rad[:, None] - chans[None, :])
    d_cell_chan = np.minimum(d_cell_chan, np.pi - d_cell_chan)
    cell_chan = np.argmin(d_cell_chan, axis=1)

    out: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in _TANG_COND_NAMES}
    n_trials_grid = np.zeros((3, 4), dtype=np.int64)      # cond x step
    for k, name in enumerate(_TANG_COND_NAMES):
        for step in range(4):                              # 0..3
            sel_items = (cond_codes == k) & (dtheta_prev_step == step)
            n_trials_grid[k, step] = int(sel_items.sum())
            if not sel_items.any():
                out[name][step] = {"mean_rate_hz": float("nan"),
                                    "n_trials": 0, "n_cells": 0}
                continue
            # Per-cell mean rate over items in this (cond, step) bucket
            # restricted to matched orientation.
            per_cell = np.zeros(n_e, dtype=np.float64)
            n_per_cell = np.zeros(n_e, dtype=np.int64)
            for c in range(n_e):
                matched = (item_chan == cell_chan[c]) & sel_items
                n = int(matched.sum())
                n_per_cell[c] = n
                if n > 0:
                    per_cell[c] = counts_per_item[c, matched].mean() / win_s
            valid = n_per_cell > 0
            if valid.any():
                mu = float(per_cell[valid].mean())
            else:
                mu = float("nan")
            out[name][step] = {
                "mean_rate_hz": mu,
                "n_trials": int(sel_items.sum()),
                "n_cells": int(valid.sum()),
            }
    return {
        "by_cond_by_step": out,
        "n_trials_grid": n_trials_grid,
        "step_labels": (0, 1, 2, 3),
        "cond_names": _TANG_COND_NAMES,
    }


def _per_cell_matched_gain(
    counts_per_item: np.ndarray,      # (n_e, n_items)
    pref_rad: np.ndarray,             # (n_e,)
    theta_per_item: np.ndarray,       # (n_items,)
    deviant_mask: np.ndarray,         # (n_items,) bool
    window_ms: float,
) -> Dict[str, Any]:
    """Per-cell (deviant − expected) rate in Hz, restricted to items whose
    presented orientation is closest to the cell's preferred θ.

    This is the Tang primary signature: enhanced response on deviants at
    the cell's matched orientation.

    Returns
    -------
    dict with keys
      - ``delta_hz``           : (n_e,) per-cell (deviant − expected)
      - ``rate_deviant_hz``    : (n_e,) per-cell deviant rate at matched θ
      - ``rate_expected_hz``   : (n_e,) per-cell expected rate at matched θ
      - ``n_items_deviant``    : (n_e,) item counts used per cell
      - ``n_items_expected``   : (n_e,) idem
      - ``mean_delta_hz``      : scalar mean(delta_hz) over cells with data
      - ``mean_delta_hz_ci``   : bootstrap 95 % CI on mean
    """
    n_e, n_items = counts_per_item.shape
    win_s = window_ms / 1000.0

    # For each cell, find the presented thetas closest to its preferred θ
    # (using the V1 channel resolution). Count items whose presented θ
    # rounds to that cell's preferred θ.
    chans = np.arange(V1_N_CHANNELS) * (np.pi / V1_N_CHANNELS)
    # Map each item's theta to nearest V1 channel.
    d_item_chan = np.abs(theta_per_item[:, None] - chans[None, :])   # (n_items, N_CHAN)
    d_item_chan = np.minimum(d_item_chan, np.pi - d_item_chan)
    item_chan = np.argmin(d_item_chan, axis=1)                       # (n_items,)

    # Map each cell to its preferred channel.
    d_cell_chan = np.abs(pref_rad[:, None] - chans[None, :])         # (n_e, N_CHAN)
    d_cell_chan = np.minimum(d_cell_chan, np.pi - d_cell_chan)
    cell_chan = np.argmin(d_cell_chan, axis=1)                       # (n_e,)

    rate_dev = np.zeros(n_e, dtype=np.float64)
    rate_exp = np.zeros(n_e, dtype=np.float64)
    n_dev_items = np.zeros(n_e, dtype=np.int64)
    n_exp_items = np.zeros(n_e, dtype=np.int64)

    for c in range(n_e):
        matched = item_chan == cell_chan[c]
        dev_sel = matched & deviant_mask
        exp_sel = matched & ~deviant_mask
        if dev_sel.any():
            rate_dev[c] = counts_per_item[c, dev_sel].mean() / win_s
            n_dev_items[c] = int(dev_sel.sum())
        if exp_sel.any():
            rate_exp[c] = counts_per_item[c, exp_sel].mean() / win_s
            n_exp_items[c] = int(exp_sel.sum())

    delta = rate_dev - rate_exp
    valid_cells = (n_dev_items > 0) & (n_exp_items > 0)
    if valid_cells.any():
        from .metrics import _bootstrap_mean_ci
        mean_delta, ci = _bootstrap_mean_ci(delta[valid_cells])
    else:
        mean_delta = float("nan")
        ci = (float("nan"), float("nan"))

    return {
        "delta_hz": delta,
        "rate_deviant_hz": rate_dev,
        "rate_expected_hz": rate_exp,
        "n_items_deviant": n_dev_items,
        "n_items_expected": n_exp_items,
        "mean_delta_hz": mean_delta,
        "mean_delta_hz_ci": ci,
        "n_cells_with_data": int(valid_cells.sum()),
    }


def _laminar_profile(
    counts_per_item: np.ndarray,      # (n_e, n_items)
    deviant_mask: np.ndarray,
    window_ms: float,
) -> Dict[str, Any]:
    """Mean V1_E rate (Hz) in deviant vs expected items.

    Single-layer V1 ring: this is the "L2/3-equivalent" placeholder per the
    task note. Reports per-item per-cell-averaged rate and population stats.
    """
    pop_mask = np.ones(counts_per_item.shape[0], dtype=bool)
    dev = total_population_activity(
        counts_per_item[:, deviant_mask], pop_mask, window_ms=window_ms,
    )
    exp = total_population_activity(
        counts_per_item[:, ~deviant_mask], pop_mask, window_ms=window_ms,
    )
    return {
        "deviant_rate_hz": dev["total_rate_hz"],
        "deviant_rate_hz_ci": dev["total_rate_hz_ci"],
        "expected_rate_hz": exp["total_rate_hz"],
        "expected_rate_hz_ci": exp["total_rate_hz_ci"],
        "delta_hz": dev["total_rate_hz"] - exp["total_rate_hz"],
    }


def _tuning_secondary(
    counts_per_item: np.ndarray,      # (n_e, n_items)
    theta_per_item: np.ndarray,
    deviant_mask: np.ndarray,
    thetas_tang: np.ndarray,
) -> Dict[str, Any]:
    """Per-cell von-Mises tuning on expected (non-deviant) items only.

    This is the pre-registered null (plan H3): FWHM should NOT change
    systematically under deviance at intact balance — a narrowing/widening
    here would signal a real tuning deformation.
    """
    n_e = counts_per_item.shape[0]
    n_t = thetas_tang.size
    sc_by_theta_exp = np.zeros((n_e, n_t), dtype=np.float64)
    sc_by_theta_dev = np.zeros((n_e, n_t), dtype=np.float64)

    for ti, theta in enumerate(thetas_tang):
        matched = np.abs(theta_per_item - theta) < 1e-9
        exp_sel = matched & ~deviant_mask
        dev_sel = matched & deviant_mask
        if exp_sel.any():
            sc_by_theta_exp[:, ti] = counts_per_item[:, exp_sel].mean(axis=1)
        if dev_sel.any():
            sc_by_theta_dev[:, ti] = counts_per_item[:, dev_sel].mean(axis=1)

    # Only fit where there is data across all 6 thetas.
    exp_fit = tuning_fit(sc_by_theta_exp, thetas_tang)
    dev_fit = tuning_fit(sc_by_theta_dev, thetas_tang)
    return {
        "expected_fit": exp_fit,
        "deviant_fit": dev_fit,
        "sc_by_theta_expected": sc_by_theta_exp,
        "sc_by_theta_deviant": sc_by_theta_dev,
        "thetas_rad": thetas_tang,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tang_rotating(
    bundle: Optional[FrozenBundle] = None,
    cfg: Optional[TangConfig] = None,
    *,
    seed: int = 42,
    r: float = 1.0,
    g_total: float = 1.0,
    verbose: bool = False,
) -> TangResult:
    """Run the Tang-rotating deviant assay end-to-end on H_T.

    Parameters
    ----------
    bundle : FrozenBundle, optional
        H_T-backed bundle (no cue). Built fresh if None.
    cfg : TangConfig, optional
    seed : int
    r, g_total : float
        Feedback balance ratio / total. Only used if bundle is None.
    verbose : bool

    Returns
    -------
    TangResult
    """
    cfg = cfg or TangConfig(seed=seed)
    if cfg.seed != seed:
        seed = cfg.seed

    if bundle is None:
        bundle = build_frozen_network(
            h_kind="ht", seed=seed, r=r, g_total=g_total, with_cue=False,
        )
    elif bundle.h_kind != "ht":
        raise ValueError(f"Tang assay requires h_kind='ht', got {bundle.h_kind!r}")

    # Sprint 5c: Tang has no natural "context window" between items (250 ms
    # back-to-back, no ITI), so context_only mode is undefined. Reject
    # explicitly rather than silently equating to "off".
    v1_to_h_mode = bundle.meta.get("v1_to_h_mode", "continuous")
    if v1_to_h_mode == "context_only":
        raise ValueError(
            "Tang assay does not support v1_to_h_mode='context_only' "
            "(no natural context window — items are 250 ms back-to-back). "
            "Use 'continuous' or 'off' instead."
        )

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- build composite item sequence: random block then rotating block --
    thetas_tang = _tang_thetas_rad()
    items_all: list = []
    deviant_parts: list = []
    block_parts: list = []
    rotation_parts: list = []
    is_random_parts: list = []

    if cfg.n_random > 0:
        rng_rand = np.random.default_rng(seed + 101)   # split RNG streams
        plan_rand = _build_random_block_plan(
            rng_rand, n_items=int(cfg.n_random),
            item_ms=cfg.item_ms, contrast=cfg.contrast,
            thetas_rad=thetas_tang,
        )
        items_all.extend(plan_rand.items)
        deviant_parts.append(plan_rand.meta["deviant_mask"])
        block_parts.append(plan_rand.meta["block_ids"])
        rotation_parts.append(plan_rand.meta["rotation_dir"])
        is_random_parts.append(np.ones(int(cfg.n_random), dtype=bool))

    if cfg.n_rotating > 0:
        plan_rot = tang_rotating_sequence(
            rng, n_items=int(cfg.n_rotating), item_ms=cfg.item_ms,
            block_len_range=cfg.block_len_range, contrast=cfg.contrast,
        )
        items_all.extend(plan_rot.items)
        # Offset rotating block_ids so they don't collide with -1 (random sentinel).
        block_parts.append(plan_rot.meta["block_ids"])
        deviant_parts.append(plan_rot.meta["deviant_mask"])
        rotation_parts.append(plan_rot.meta["rotation_dir"])
        is_random_parts.append(np.zeros(plan_rot.meta["n_items"], dtype=bool))

    items = items_all
    n_items = len(items)
    deviant_mask = (np.concatenate(deviant_parts) if deviant_parts
                     else np.zeros(0, dtype=bool))
    block_ids = (np.concatenate(block_parts) if block_parts
                  else np.zeros(0, dtype=np.int64))
    rotation_dir = (np.concatenate(rotation_parts) if rotation_parts
                     else np.zeros(0, dtype=np.int64))
    is_random = (np.concatenate(is_random_parts) if is_random_parts
                  else np.zeros(0, dtype=bool))

    # Per-item presented theta (rad).
    theta_per_item = np.array(
        [it.theta_rad for it in items], dtype=np.float64,
    )

    # Δθ_prev_step: step distance between previous item θ and current θ.
    # First item (no prior) → -1 sentinel.
    dtheta_prev_step = np.full(n_items, -1, dtype=np.int64)
    for i in range(1, n_items):
        dtheta_prev_step[i] = _theta_step_distance(
            float(theta_per_item[i - 1]), float(theta_per_item[i]), thetas_tang,
        )

    cond_codes = _condition_per_item(deviant_mask, is_random)

    n_e = int(bundle.v1_ring.e.N)
    e_mon = SpikeMonitor(bundle.v1_ring.e, name=f"tang_e_mon_seed{seed}")
    net = Network(*bundle.groups, e_mon)

    # Pre-settle (blank) to let the bump stabilise; no items emitted.
    bundle.reset_all()
    if cfg.presettle_ms > 0:
        set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)
        net.run(cfg.presettle_ms * ms)

    counts_per_item = np.zeros((n_e, n_items), dtype=np.int64)
    for k, it in enumerate(items):
        set_grating(
            bundle.v1_ring, theta_rad=it.theta_rad, contrast=it.contrast,
        )
        pre_e = _snapshot(e_mon)
        net.run(cfg.item_ms * ms)
        counts_per_item[:, k] = _snapshot(e_mon) - pre_e

        if verbose and (k + 1) % 250 == 0:
            n_dev = int(deviant_mask[: k + 1].sum())
            print(f"[tang] item {k+1}/{n_items} (deviants so far={n_dev})")

    # Turn stim off after the sequence.
    set_grating(bundle.v1_ring, theta_rad=None, contrast=0.0)

    pref_rad = v1_e_preferred_thetas(bundle.v1_ring)     # (n_e,)
    thetas_tang = _tang_thetas_rad()

    # Restrict legacy metrics (cell-gain / SVM / laminar / tuning) to
    # rotating-block items: "deviant_mask" is only meaningful there
    # (random-block items are all flagged ~deviant by construction).
    rot_idx = np.where(~is_random)[0]

    # ---- metric 1: per-cell matched-θ gain (rotating only) -----------
    if rot_idx.size > 0:
        cell_gain = _per_cell_matched_gain(
            counts_per_item[:, rot_idx], pref_rad,
            theta_per_item[rot_idx], deviant_mask[rot_idx],
            window_ms=cfg.item_ms,
        )
    else:
        cell_gain = {
            "delta_hz": np.zeros(n_e),
            "rate_deviant_hz": np.zeros(n_e),
            "rate_expected_hz": np.zeros(n_e),
            "n_items_deviant": np.zeros(n_e, dtype=np.int64),
            "n_items_expected": np.zeros(n_e, dtype=np.int64),
            "mean_delta_hz": float("nan"),
            "mean_delta_hz_ci": (float("nan"), float("nan")),
            "n_cells_with_data": 0,
        }

    # ---- metric 2: population SVM decoding (rotating only) -----------
    if rot_idx.size > 1 and deviant_mask[rot_idx].any() and (~deviant_mask[rot_idx]).any():
        X = counts_per_item[:, rot_idx].T                  # (n_rot_items, n_e)
        y = deviant_mask[rot_idx].astype(np.int64)
        svm_res = svm_decoding_accuracy(X, y, cv=5, seed=cfg.seed)
    else:
        svm_res = {"accuracy": float("nan"), "ci": (float("nan"), float("nan"))}

    # ---- metric 3: laminar profile (rotating only) -------------------
    if rot_idx.size > 0:
        laminar = _laminar_profile(
            counts_per_item[:, rot_idx], deviant_mask[rot_idx],
            window_ms=cfg.item_ms,
        )
    else:
        laminar = {
            "deviant_rate_hz": float("nan"),
            "deviant_rate_hz_ci": (float("nan"), float("nan")),
            "expected_rate_hz": float("nan"),
            "expected_rate_hz_ci": (float("nan"), float("nan")),
            "delta_hz": float("nan"),
        }

    # ---- secondary: tuning FWHM (rotating only) ----------------------
    if rot_idx.size > 0:
        tuning = _tuning_secondary(
            counts_per_item[:, rot_idx], theta_per_item[rot_idx],
            deviant_mask[rot_idx], thetas_tang,
        )
    else:
        n_t = thetas_tang.size
        empty_fit = {"fwhm_rad": np.full(n_e, np.nan)}
        tuning = {
            "expected_fit": empty_fit,
            "deviant_fit": empty_fit,
            "sc_by_theta_expected": np.zeros((n_e, n_t)),
            "sc_by_theta_deviant": np.zeros((n_e, n_t)),
            "thetas_rad": thetas_tang,
        }

    # ---- R3 primary: 3-condition matched-θ rate ----------------------
    three_cond = _three_condition_matched_rate(
        counts_per_item, pref_rad, theta_per_item, cond_codes,
        window_ms=cfg.item_ms,
    )

    # ---- R3 covariate: Δθ_prev_step stratification -------------------
    dtheta_prev_out = _dtheta_prev_stratified(
        counts_per_item, pref_rad, theta_per_item, cond_codes,
        dtheta_prev_step, window_ms=cfg.item_ms,
    )

    return TangResult(
        cell_gain=cell_gain,
        svm=svm_res,
        laminar=laminar,
        tuning=tuning,
        three_condition=three_cond,
        dtheta_prev=dtheta_prev_out,
        raw={
            "counts_per_item": counts_per_item,
            "theta_per_item": theta_per_item,
            "deviant_mask": deviant_mask,
            "block_ids": block_ids,
            "rotation_dir": rotation_dir,
            "is_random": is_random,
            "dtheta_prev_step": dtheta_prev_step,
            "cond_codes": cond_codes,
            "pref_rad": pref_rad,
        },
        meta={
            "seed": int(cfg.seed),
            "n_items": int(n_items),
            "n_random": int(cfg.n_random),
            "n_rotating": int(cfg.n_rotating),
            "n_deviant": int(deviant_mask.sum()),
            "n_expected": int((~deviant_mask & ~is_random).sum()),
            "n_random_items": int(is_random.sum()),
            "config": cfg.__dict__,
            "bundle": {k: v for k, v in bundle.meta.items() if k != "config"},
        },
    )


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TangConfig(
        n_random=30, n_rotating=30, item_ms=150.0,
        presettle_ms=200.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg, verbose=True)
    print(f"[tang smoke] n_items={r.meta['n_items']} "
          f"random={r.meta['n_random_items']} dev={r.meta['n_deviant']} "
          f"exp={r.meta['n_expected']}")
    print(f"  cell-gain mean Δ Hz = {r.cell_gain['mean_delta_hz']:+.3f}")
    print(f"  svm acc             = {r.svm['accuracy']:.3f}")
    print(f"  laminar Δ Hz (dev-exp) = "
          f"{r.laminar['deviant_rate_hz']:.3f} − {r.laminar['expected_rate_hz']:.3f}"
          f" = {r.laminar['delta_hz']:+.3f}")
    fw = r.tuning["expected_fit"]["fwhm_rad"]
    n_ok = int(np.isfinite(fw).sum())
    print(f"  FWHM fits (expected): {n_ok}/{fw.size} cells (Tang null)")
    tc = r.three_condition["per_cond"]
    print(f"  3-cond rate (Hz): "
          f"random={tc['random']['mean_rate_hz']:.3f} "
          f"exp={tc['rotating_expected']['mean_rate_hz']:.3f} "
          f"dev={tc['rotating_deviant']['mean_rate_hz']:.3f}")
