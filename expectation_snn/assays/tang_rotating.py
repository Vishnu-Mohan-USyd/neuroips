"""Tang-like rotating-orientation assay (Sprint 5a Step 4, task #27).

Paradigm (plan §3.7, Tang 2023 logic)::

    1000 items × 250 ms (back-to-back, no ISI)
    rotating orientation blocks of length 5-9 (±30 °), last item of each
    block replaced by a deviant (orientation not along the rotation).

Items ride on a frozen H_T ring pre-trained (Stage 1 H_T) on the same
rotating grammar. V1 response is measured per item; the deviant_mask
from :func:`stimulus.tang_rotating_sequence` labels each item.

Primary neuron-level metrics (assays.metrics):

1. Per-cell gain at θ_stim (deviant − expected, matched-θ only).
2. Population decoding (5-fold linear SVM on deviant vs expected).
3. Laminar profile: V1_E mean Hz per condition
   (single-population V1 ⇒ L2/3-equivalent placeholder per task note).

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
    """Tang rotating-orientation paradigm configuration."""
    n_items: int = 1000
    item_ms: float = TANG_ITEM_MS
    block_len_range: tuple = TANG_BLOCK_LEN_RANGE
    contrast: float = 1.0
    presettle_ms: float = 500.0          # warm-up before first item
    seed: int = 42


@dataclass
class TangResult:
    """Tang rotating assay output (Sprint 5a primary metrics)."""
    cell_gain: Dict[str, Any]            # per-cell matched-θ deviant − expected
    svm: Dict[str, Any]                  # pop decoding
    laminar: Dict[str, Any]              # V1_E mean Hz per condition
    tuning: Dict[str, Any]               # FWHM secondary (von Mises fits)
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

    # --- build item sequence (back-to-back; no ITI between items) ---------
    plan = tang_rotating_sequence(
        rng, n_items=cfg.n_items, item_ms=cfg.item_ms,
        block_len_range=cfg.block_len_range, contrast=cfg.contrast,
    )
    items = plan.items
    n_items = len(items)
    deviant_mask = plan.meta["deviant_mask"]            # (n_items,)
    block_ids = plan.meta["block_ids"]                   # (n_items,)
    rotation_dir = plan.meta["rotation_dir"]             # (n_items,)

    # Per-item presented theta (rad).
    theta_per_item = np.array(
        [it.theta_rad for it in items], dtype=np.float64,
    )

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

    # ---- metric 1: per-cell matched-θ gain ----------------------------
    cell_gain = _per_cell_matched_gain(
        counts_per_item, pref_rad, theta_per_item, deviant_mask,
        window_ms=cfg.item_ms,
    )

    # ---- metric 2: population SVM decoding ----------------------------
    X = counts_per_item.T                                 # (n_items, n_e)
    y = deviant_mask.astype(np.int64)
    svm_res = svm_decoding_accuracy(X, y, cv=5, seed=cfg.seed)

    # ---- metric 3: laminar profile (V1_E) -----------------------------
    laminar = _laminar_profile(
        counts_per_item, deviant_mask, window_ms=cfg.item_ms,
    )

    # ---- secondary: tuning FWHM ---------------------------------------
    tuning = _tuning_secondary(
        counts_per_item, theta_per_item, deviant_mask, thetas_tang,
    )

    return TangResult(
        cell_gain=cell_gain,
        svm=svm_res,
        laminar=laminar,
        tuning=tuning,
        raw={
            "counts_per_item": counts_per_item,
            "theta_per_item": theta_per_item,
            "deviant_mask": deviant_mask,
            "block_ids": block_ids,
            "rotation_dir": rotation_dir,
            "pref_rad": pref_rad,
        },
        meta={
            "seed": int(cfg.seed),
            "n_items": int(n_items),
            "n_deviant": int(deviant_mask.sum()),
            "n_expected": int((~deviant_mask).sum()),
            "config": cfg.__dict__,
            "bundle": {k: v for k, v in bundle.meta.items() if k != "config"},
        },
    )


# ---------------------------------------------------------------------------
# CLI / smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TangConfig(
        n_items=60, item_ms=150.0, presettle_ms=200.0, seed=42,
    )
    r = run_tang_rotating(cfg=cfg, verbose=True)
    print(f"[tang smoke] n_items={r.meta['n_items']} "
          f"dev={r.meta['n_deviant']} exp={r.meta['n_expected']}")
    print(f"  cell-gain mean Δ Hz = {r.cell_gain['mean_delta_hz']:+.3f}")
    print(f"  svm acc             = {r.svm['accuracy']:.3f}")
    print(f"  laminar Δ Hz (dev-exp) = "
          f"{r.laminar['deviant_rate_hz']:.3f} − {r.laminar['expected_rate_hz']:.3f}"
          f" = {r.laminar['delta_hz']:+.3f}")
    fw = r.tuning["expected_fit"]["fwhm_rad"]
    n_ok = int(np.isfinite(fw).sum())
    print(f"  FWHM fits (expected): {n_ok}/{fw.size} cells (Tang null)")
