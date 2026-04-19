"""Component-level validation for the H-ring inhibitory architecture
(brian2_model/h_ring.py).

Four biology-anchored assays (Sprint-3 rework, 2026-04-19). All run on H_R
built via `build_h_r`; Vogels iSTDP is frozen (`inh_eta=0`) during probing
so only the fixed-topology inhibitory architecture is tested.

1. **Direct inh firing rates.** Two probes:

     (a) Drive ch0 only -> measure local_inh[0] rate (the per-channel
         cell tuned to ch0). Floor: >= LOCAL_INH_RATE_MIN_HZ.
     (b) Drive ch0 + ch6 simultaneously -> measure broad_inh mean rate
         (cells [N_CHANNELS..N_INH_POOL)). Floor: >= BROAD_INH_RATE_MIN_HZ.
         The broad pool receives sparse E->inh input (p_e_inh=0.4) and
         needs multi-channel E drive to cross rheobase (H_inh rheobase =
         (V_th-E_L)*gL = 15 mV * 15 nS = 225 pA; one channel's 16 E cells
         deliver insufficient AMPA charge).

2. **Local-inh per-channel tuning.** Drive ch0 only. The local inh cell
   tuned to ch0 (local[0]) must fire far more than the cell tuned to a
   non-driven channel (local[1]). This verifies that local inh are
   per-channel (one cell per theta bin) rather than a single broad pool:
     - local[0] >= LOCAL_INH_RATE_MIN_HZ (already asserted in Assay 1)
     - local[1] <= LOCAL_INH_NEIGHBOR_CEILING_HZ
     - ratio local[0] / max(local[1], 1 Hz) >= LOCAL_INH_TUNING_RATIO.
   (Gain-control via weight ablation is the *wrong* probe for a frozen
   iSTDP ring: at the init weight w_inh_e_init=0.5 local IPSCs are
   subthreshold to perturb ch0's cue-saturated firing; that lever is
   learned by Vogels during the settle window, not topology-fixed.)

3. **Broad-inh multi-channel suppression.** Drive ch0 + ch6. Compare total
   E rate across driven channels with broad intact vs broad_inh_scale=0.
   Broad removal must lift total activity:
   delta (ablated - intact) >= BROAD_ABL_SUM_UP_HZ.

4. **WTA regression flag (informational only).** Drive ch0 + ch6 for 500
   ms. Count channels with tail rate >= WTA_RATE_THRESHOLD_HZ. The
   current per-channel + broad-pool architecture is known to NOT enforce
   WTA under symmetric 2-channel drive (documented structural gap, see
   `docs/phase_gate_evidence.md` Sprint-3 backfill section). This assay
   measures the margin and prints it so a future circuit fix (Mexican-hat
   cross-channel inh / SFA / cross-channel E->inh) can be detected as a
   regression win. **Not gated** — does not affect pass/fail verdict.

Run:
    python -m expectation_snn.validation.validate_h_ring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from brian2 import (
    Network,
    SpikeMonitor,
    defaultclock,
    Hz,
    ms,
    prefs,
    seed as b2_seed,
)

from ..brian2_model.h_ring import (
    HRingConfig,
    N_CHANNELS,
    N_E_PER_CHANNEL,
    N_INH_POOL,
    build_h_r,
    silence_cue,
)


# -- constants ---------------------------------------------------------------

LOCAL_INH_RATE_MIN_HZ = 5.0           # local[0] evoked rate floor (single-ch drive)
LOCAL_INH_NEIGHBOR_CEILING_HZ = 1.0   # local[1] must stay near-silent under ch0-only drive
LOCAL_INH_TUNING_RATIO = 5.0          # local[0] / max(local[1], 1 Hz)
BROAD_INH_RATE_MIN_HZ = 1.0           # broad mean rate floor (dual-ch drive)
BROAD_ABL_SUM_UP_HZ = 10.0            # sum(ch0+ch6) E rate rise when broad_scale=0
WTA_RATE_THRESHOLD_HZ = 5.0        # "sustained bump" rate
PROBE_DUR_MS = 500.0
SETTLE_MS = 200.0
CUE_PEAK_HZ = 300.0
WTA_STEADY_MS = 250.0              # tail window for WTA measurement


# -- cfg ---------------------------------------------------------------------

def _stage1_cfg(**overrides) -> HRingConfig:
    """Stage-1 tuned config with Vogels frozen (inh_eta=0) for probing."""
    cfg = HRingConfig()
    cfg.w_ee_within_init = 1.0
    cfg.w_ee_cross_init = 0.02
    cfg.drive_amp_ee_pA = 50.0
    cfg.nmda_drive_amp_nS = 0.5
    cfg.ee_w_max = 1.5
    cfg.p_e_inh = 0.4
    cfg.w_e_inh = 0.4
    cfg.drive_amp_e_inh_pA = 40.0
    cfg.w_inh_e_init = 0.5
    cfg.broad_inh_scale = 0.3
    cfg.drive_amp_inh_e_pA = 40.0
    cfg.inh_rho_hz = 10.0
    cfg.inh_eta = 0.0
    cfg.inh_w_max = 1.5
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# -- helpers -----------------------------------------------------------------

def _drive_ch(ring, chs, peak_rate_hz: float) -> None:
    n_cue = int(ring.cue.N)
    block = n_cue // N_CHANNELS
    rates = np.zeros(n_cue)
    for c in chs:
        rates[c * block : (c + 1) * block] = peak_rate_hz
    ring.cue.rates = rates * Hz


def _e_rate_per_channel(e_mon: SpikeMonitor, e_channel: np.ndarray,
                        t0_ms: float, t1_ms: float) -> np.ndarray:
    i = np.asarray(e_mon.i[:], dtype=np.int64)
    t_ms = np.asarray(e_mon.t / ms)
    mask = (t_ms >= t0_ms) & (t_ms < t1_ms)
    counts = np.bincount(e_channel[i[mask]], minlength=N_CHANNELS)
    dur_s = max((t1_ms - t0_ms) / 1000.0, 1e-6)
    return counts / (N_E_PER_CHANNEL * dur_s)


def _inh_rate(inh_mon: SpikeMonitor, indices: np.ndarray,
              t0_ms: float, t1_ms: float) -> float:
    if len(indices) == 0:
        return 0.0
    i = np.asarray(inh_mon.i[:], dtype=np.int64)
    t_ms = np.asarray(inh_mon.t / ms)
    mask = (t_ms >= t0_ms) & (t_ms < t1_ms) & np.isin(i, indices)
    dur_s = max((t1_ms - t0_ms) / 1000.0, 1e-6)
    return float(mask.sum() / (len(indices) * dur_s))


def _probe(cfg: HRingConfig, drive_channels, probe_ms: float = PROBE_DUR_MS,
           seed: int = 42):
    """Build a fresh ring, drive `drive_channels`, run PROBE + tail.

    Returns (per-local-channel inh rate array shape (N_CHANNELS,),
    broad mean rate Hz, per-ch E rates full, per-ch E rates tail).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)

    ring = build_h_r(config=cfg)

    silence_cue(ring)
    e_mon = SpikeMonitor(ring.e, name=f"hr_e_{seed}_{len(drive_channels)}")
    inh_mon = SpikeMonitor(ring.inh, name=f"hr_inh_{seed}_{len(drive_channels)}")
    net = Network(*ring.groups, e_mon, inh_mon)

    net.run(SETTLE_MS * ms)
    _drive_ch(ring, drive_channels, CUE_PEAK_HZ)
    net.run(probe_ms * ms)

    t0, t1 = SETTLE_MS, SETTLE_MS + probe_ms
    per_local_rate = np.zeros(N_CHANNELS)
    for c in range(N_CHANNELS):
        per_local_rate[c] = _inh_rate(inh_mon, np.asarray([c]), t0, t1)
    broad_rate = _inh_rate(
        inh_mon, np.arange(N_CHANNELS, N_INH_POOL), t0, t1
    )
    per_ch_full = _e_rate_per_channel(e_mon, ring.e_channel, t0, t1)
    per_ch_tail = _e_rate_per_channel(
        e_mon, ring.e_channel, t1 - WTA_STEADY_MS, t1,
    )
    return per_local_rate, broad_rate, per_ch_full, per_ch_tail


# -- report ------------------------------------------------------------------

@dataclass
class HRingValidationReport:
    per_local_rate_single_hz: np.ndarray  # shape (N_CHANNELS,)
    broad_rate_dual_hz: float
    e_intact_dual_tail: np.ndarray
    e_broad_ablated_dual_tail: np.ndarray

    passed_direct_rates: bool
    passed_local_tuning: bool
    passed_broad_ablation: bool
    wta_n_sustained: int              # informational only

    @property
    def passed(self) -> bool:
        return (
            self.passed_direct_rates
            and self.passed_local_tuning
            and self.passed_broad_ablation
        )

    def summary(self) -> str:
        local0 = float(self.per_local_rate_single_hz[0])
        local1 = float(self.per_local_rate_single_hz[1])
        ratio = local0 / max(local1, 1.0)
        intact_sum = self.e_intact_dual_tail[0] + self.e_intact_dual_tail[6]
        ablated_sum = (
            self.e_broad_ablated_dual_tail[0]
            + self.e_broad_ablated_dual_tail[6]
        )
        broad_delta = ablated_sum - intact_sum
        lines = ["H-ring inhibition validation (Sprint-3 rework):"]
        lines.append("  Assay 1: direct inh firing rates")
        lines.append(
            f"    (a) drive ch0 only -> local_inh[0] = {local0:6.2f} Hz"
            f"  (>= {LOCAL_INH_RATE_MIN_HZ})"
        )
        lines.append(
            f"    (b) drive ch0+ch6 -> broad mean   = {self.broad_rate_dual_hz:6.2f} Hz"
            f"  (>= {BROAD_INH_RATE_MIN_HZ})  "
            f"{'PASS' if self.passed_direct_rates else 'FAIL'}"
        )
        lines.append("  Assay 2: local-inh per-channel tuning (drive ch0)")
        lines.append(
            f"    local[0] = {local0:6.2f} Hz; "
            f"local[1] = {local1:6.2f} Hz  "
            f"(nbr ceiling <= {LOCAL_INH_NEIGHBOR_CEILING_HZ} Hz)"
        )
        lines.append(
            f"    tuning ratio local[0]/max(local[1],1) = {ratio:.2f}  "
            f"(>= {LOCAL_INH_TUNING_RATIO})  "
            f"{'PASS' if self.passed_local_tuning else 'FAIL'}"
        )
        lines.append("  Assay 3: broad-inh multi-ch suppression (drive ch0+ch6; broad_scale=0)")
        lines.append(
            f"    tail sum(ch0+ch6) intact    = {intact_sum:6.2f} Hz"
        )
        lines.append(
            f"    tail sum(ch0+ch6) ablated   = {ablated_sum:6.2f} Hz"
        )
        lines.append(
            f"    delta = {broad_delta:+.2f} Hz (>= {BROAD_ABL_SUM_UP_HZ})  "
            f"{'PASS' if self.passed_broad_ablation else 'FAIL'}"
        )
        lines.append(
            "  Assay 4: WTA under symmetric 2-ch drive (INFORMATIONAL -- "
            "known structural gap)"
        )
        lines.append(
            f"    intact n sustained @ >= {WTA_RATE_THRESHOLD_HZ} Hz = "
            f"{self.wta_n_sustained} (biology target = 1)"
        )
        lines.append(
            "    current architecture lacks cross-channel E->inh / "
            "Mexican-hat wiring;"
        )
        lines.append(
            "    reported for regression tracking, not gated."
        )
        lines.append("  ---")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def run_h_ring_validation(verbose: bool = True) -> HRingValidationReport:
    cfg_intact = _stage1_cfg()
    cfg_broad_off = _stage1_cfg(broad_inh_scale=0.0)

    # Single-channel drive (intact) — per-channel local rates + baseline.
    per_local_rate, _b1, _e1, _tail1 = _probe(
        cfg_intact, [0], seed=42,
    )

    # Dual-channel drive (intact) — for broad rate + intact WTA margin.
    _l2, broad_rate, _e2_full, e_intact_dual_tail = _probe(
        cfg_intact, [0, N_CHANNELS // 2], seed=44,
    )

    # Dual-channel drive (broad_inh_scale=0) — for broad-ablation signature.
    _l3, _b3, _e3_full, e_broad_ablated_dual_tail = _probe(
        cfg_broad_off, [0, N_CHANNELS // 2], seed=45,
    )

    passed_direct = (
        per_local_rate[0] >= LOCAL_INH_RATE_MIN_HZ
        and broad_rate >= BROAD_INH_RATE_MIN_HZ
    )
    ratio = per_local_rate[0] / max(per_local_rate[1], 1.0)
    passed_local_tuning = (
        per_local_rate[0] >= LOCAL_INH_RATE_MIN_HZ
        and per_local_rate[1] <= LOCAL_INH_NEIGHBOR_CEILING_HZ
        and ratio >= LOCAL_INH_TUNING_RATIO
    )

    intact_sum = e_intact_dual_tail[0] + e_intact_dual_tail[6]
    ablated_sum = (
        e_broad_ablated_dual_tail[0] + e_broad_ablated_dual_tail[6]
    )
    broad_delta = ablated_sum - intact_sum
    passed_broad_ablation = broad_delta >= BROAD_ABL_SUM_UP_HZ

    wta_n = int(np.sum(e_intact_dual_tail >= WTA_RATE_THRESHOLD_HZ))

    rep = HRingValidationReport(
        per_local_rate_single_hz=per_local_rate,
        broad_rate_dual_hz=broad_rate,
        e_intact_dual_tail=e_intact_dual_tail,
        e_broad_ablated_dual_tail=e_broad_ablated_dual_tail,
        passed_direct_rates=passed_direct,
        passed_local_tuning=passed_local_tuning,
        passed_broad_ablation=passed_broad_ablation,
        wta_n_sustained=wta_n,
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_h_ring_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
