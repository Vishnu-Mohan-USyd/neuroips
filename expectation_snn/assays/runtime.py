"""Shared assay runtime: build the frozen full network (Sprint 5a).

Loads Stage-0 V1 bias + PV weights, Stage-1 H_R/H_T recurrent weights, and
(optionally) Stage-2 cue weights into fresh Brian2 groups, wires up the
H → V1 feedback routes at a given balance ratio, and returns a
`FrozenBundle` container. Each assay consumes the bundle, adds monitors,
and drives its own trial loop.

All checkpoints are loaded from
``expectation_snn/data/checkpoints/stage_{0,1_hr,1_ht,2}_seed{N}.npz`` by
default.

Plasticity is frozen on every plastic synapse so that nothing drifts
during the measurement phase.

References
----------
- Plan sec 3 — Stage 0 calibration + Stage 1 H_R / H_T + Stage 2 cue.
- Sprint 5a dispatch (task #27): intact, default balance r = 1.0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from brian2 import (
    Hz, mV, ms, nS, pA, PoissonGroup, SpikeMonitor, Synapses,
)

from ..brian2_model.h_ring import (
    HRing, HRingConfig, build_h_r, build_h_t,
    N_CHANNELS as H_N_CHANNELS, N_E_PER_CHANNEL as H_N_E_PER,
)
from ..brian2_model.v1_ring import (
    V1Ring, V1RingConfig, build_v1_ring,
    N_CHANNELS as V1_N_CHANNELS, N_E_PER_CHANNEL as V1_N_E_PER,
)
from ..brian2_model.feedback_routes import (
    FeedbackRoutes, FeedbackRoutesConfig, build_feedback_routes,
)
from ..brian2_model.feedforward_v1_to_h import (
    V1ToH, V1ToHConfig, build_v1_to_h_feedforward,
)
from ..brian2_model.h_clamp import (
    HClamp, HClampConfig, build_h_clamp,
)
from ..brian2_model.plasticity import eligibility_trace_cue_rule


# Paths --------------------------------------------------------------------

DEFAULT_CKPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "checkpoints",
)

# Cue geometry (matches train.py Stage-2 constants; duplicated here so
# runtime doesn't import from train.py — avoids pulling in the validation
# heavy-weight modules during assay setup).
STAGE2_CUE_CHANNELS: Tuple[int, int] = (3, 9)          # 45°, 135°
STAGE2_N_CUE_AFFERENTS: int = 32
STAGE2_CUE_ACTIVE_HZ: float = 80.0
STAGE2_W_INIT: float = 0.1
STAGE2_W_MAX: float = 2.0
STAGE2_LR: float = 0.0002
STAGE2_TAU_ELIG_MS: float = 1500.0
STAGE2_CUE_DRIVE_PA: float = 20.0


# Bundle ------------------------------------------------------------------

@dataclass
class FrozenBundle:
    """Container holding the frozen assay-time network.

    Attributes
    ----------
    h_ring, v1_ring : the two rings (H_R or H_T for `h_ring`).
    fb : H → V1 feedback routes at the given (g_total, r).
    cue_A, cue_B : Poisson cue groups (only if with_cue=True).
    cue_A_to_h, cue_B_to_h : frozen cue → H_E synapses (lr_eff = 0).
    groups : list of all Brian2 objects to splat into ``Network(...)``.
    meta : dict with loaded-checkpoint info + config.
    """
    h_ring: HRing
    h_kind: str
    v1_ring: V1Ring
    fb: FeedbackRoutes
    v1_to_h: Optional[V1ToH] = None
    h_clamp: Optional[HClamp] = None
    # Sprint 5d pre-probe H instrumentation (diagnostic D1/D2).
    # Present iff build_frozen_network(..., with_preprobe_h_mon=True).
    h_e_mon: Optional[SpikeMonitor] = None

    cue_A: Optional[PoissonGroup] = None
    cue_B: Optional[PoissonGroup] = None
    cue_A_to_h: Optional[Synapses] = None
    cue_B_to_h: Optional[Synapses] = None

    groups: List[object] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- per-trial state helpers ------------------------------------------

    def reset_h(self) -> None:
        """Kill any lingering bump / eligibility in H_R (H_T) so each trial
        starts from a clean resting state. Matches Stage-2 training reset.
        """
        self.h_ring.e.V = -70.0 * mV
        self.h_ring.e.I_e = 0 * pA
        self.h_ring.e.I_i = 0 * pA
        self.h_ring.e.g_nmda_h = 0 * nS
        self.h_ring.inh.V = -65.0 * mV
        self.h_ring.inh.I_e = 0 * pA
        self.h_ring.inh.I_i = 0 * pA
        if self.cue_A_to_h is not None:
            self.cue_A_to_h.elig = 0.0
        if self.cue_B_to_h is not None:
            self.cue_B_to_h.elig = 0.0

    def reset_v1(self) -> None:
        """Reset V1 soma/apical voltages + currents. (Optional; long ITIs
        usually suffice for full decay, but this removes any residual
        adaptation/apical state between trials.)"""
        self.v1_ring.e.V_soma = -70.0 * mV
        self.v1_ring.e.V_ap = -70.0 * mV
        self.v1_ring.e.I_e = 0 * pA
        self.v1_ring.e.I_i = 0 * pA
        self.v1_ring.e.I_ap_e = 0 * pA
        self.v1_ring.e.w_adapt = 0 * pA
        self.v1_ring.som.V = -65.0 * mV
        self.v1_ring.som.I_e = 0 * pA
        self.v1_ring.som.I_i = 0 * pA
        self.v1_ring.pv.V = -65.0 * mV
        self.v1_ring.pv.I_e = 0 * pA
        self.v1_ring.pv.I_i = 0 * pA

    def reset_all(self) -> None:
        self.reset_h()
        self.reset_v1()

    def cue_on(self, which: str, rate_hz: float = STAGE2_CUE_ACTIVE_HZ) -> None:
        """Turn cue A or B on at `rate_hz`; the other off.

        `which` ∈ {"A", "B"}. Raises if cue pathway not built.
        """
        if self.cue_A is None or self.cue_B is None:
            raise RuntimeError(
                "Cue pathway not built (pass with_cue=True to build_frozen_network)"
            )
        if which == "A":
            self.cue_A.rates = rate_hz * Hz
            self.cue_B.rates = 0 * Hz
        elif which == "B":
            self.cue_A.rates = 0 * Hz
            self.cue_B.rates = rate_hz * Hz
        else:
            raise ValueError(f"cue must be 'A' or 'B', got {which!r}")

    def cue_off(self) -> None:
        if self.cue_A is not None:
            self.cue_A.rates = 0 * Hz
        if self.cue_B is not None:
            self.cue_B.rates = 0 * Hz


# Checkpoint loaders -------------------------------------------------------

def _load_stage0_into_v1(v1: V1Ring, ckpt_path: str) -> Tuple[float, int]:
    """Restore V1 E I_bias + PV→E weights from Stage-0 checkpoint."""
    data = np.load(ckpt_path)
    bias = float(data["bias_pA"])
    v1.e.I_bias = bias * pA
    pv_w = np.asarray(data["pv_to_e_w"], dtype=np.float64)
    live = np.asarray(v1.pv_to_e.w[:])
    if pv_w.shape != live.shape:
        raise ValueError(
            f"Stage-0 pv_to_e_w shape {pv_w.shape} != live {live.shape}"
        )
    v1.pv_to_e.w[:] = pv_w
    return bias, pv_w.size


def _load_stage1_into_h(h: HRing, ckpt_path: str) -> int:
    """Restore H ring E→E weights from Stage-1 checkpoint.

    Works for both H_R (stage_1_hr) and H_T (stage_1_ht); both checkpoints
    store the same-shape ``ee_w_final`` on the shared 12-channel ring.
    """
    data = np.load(ckpt_path)
    ee_w = np.asarray(data["ee_w_final"], dtype=np.float64)
    live = np.asarray(h.ee.w[:])
    if ee_w.shape != live.shape:
        raise ValueError(
            f"Stage-1 ee_w_final shape {ee_w.shape} != live {live.shape}"
        )
    h.ee.w[:] = ee_w
    return ee_w.size


def _freeze_v1_plasticity(v1: V1Ring) -> None:
    v1.pv_to_e.active = False


def _freeze_h_plasticity(h: HRing) -> None:
    """Freeze pair-STDP on ``ee`` and Vogels iSTDP on ``inh_to_e``. Drive
    (AMPA + NMDA co-release; inh current) still flows — only the weight-
    update terms are set to zero.
    """
    h.ee.namespace["A_plus_eff"] = 0.0
    h.ee.namespace["A_minus_eff"] = 0.0
    h.inh_to_e.namespace["eta_eff"] = 0.0


def _build_frozen_cue_pathway(
    h: HRing,
) -> Tuple[PoissonGroup, PoissonGroup, Synapses, Synapses]:
    """Build cue_A / cue_B Poisson groups + frozen elig synapses.

    Same wiring shape as train._build_stage2_cue_pathway, but with
    learning_rate=0 so weights never drift when a cue fires during the
    assay. Caller loads weights from Stage-2 checkpoint.
    """
    cue_A = PoissonGroup(
        STAGE2_N_CUE_AFFERENTS, rates=0 * Hz, name="s5a_cue_A",
    )
    cue_B = PoissonGroup(
        STAGE2_N_CUE_AFFERENTS, rates=0 * Hz, name="s5a_cue_B",
    )
    cue_A_to_h = eligibility_trace_cue_rule(
        cue_A, h.e,
        connectivity="True",
        w_init=STAGE2_W_INIT,
        w_max=STAGE2_W_MAX,
        tau_elig=STAGE2_TAU_ELIG_MS * ms,
        learning_rate=0.0,                    # FROZEN — lr_eff = 0
        drive_amp_pA=STAGE2_CUE_DRIVE_PA,
        name="s5a_cue_A_to_h",
    )
    cue_B_to_h = eligibility_trace_cue_rule(
        cue_B, h.e,
        connectivity="True",
        w_init=STAGE2_W_INIT,
        w_max=STAGE2_W_MAX,
        tau_elig=STAGE2_TAU_ELIG_MS * ms,
        learning_rate=0.0,
        drive_amp_pA=STAGE2_CUE_DRIVE_PA,
        name="s5a_cue_B_to_h",
    )
    return cue_A, cue_B, cue_A_to_h, cue_B_to_h


# Builder ------------------------------------------------------------------

def build_frozen_network(
    h_kind: str = "hr",
    seed: int = 42,
    r: float = 1.0,
    g_total: float = 1.0,
    with_cue: bool = False,
    with_v1_to_h: object = "continuous",
    with_feedback_routes: bool = True,
    with_h_clamp: Optional[HClampConfig] = None,
    with_preprobe_h_mon: bool = False,
    ckpt_dir: Optional[str] = None,
    h_cfg: Optional[HRingConfig] = None,
    v1_cfg: Optional[V1RingConfig] = None,
    fb_cfg: Optional[FeedbackRoutesConfig] = None,
    v1_to_h_cfg: Optional[V1ToHConfig] = None,
) -> FrozenBundle:
    """Assemble the intact frozen assay-time network.

    Parameters
    ----------
    h_kind : str
        "hr" → H_R (Kok, Richter) loaded from stage_1_hr_seed{SEED}.npz.
        "ht" → H_T (Tang)            loaded from stage_1_ht_seed{SEED}.npz.
    seed : int
        Checkpoint seed suffix. Only seed=42 has been trained per task #27.
    r, g_total : float
        Feedback-balance ratio + total. Default r=1.0 (Sprint 5a default).
    with_cue : bool
        If True, build the Stage-2 cue pathway (cue_A/cue_B + frozen elig
        synapses) and load Stage-2 weights. Only valid for h_kind='hr'
        (Stage-2 was trained against H_R). Raises on 'ht' + with_cue=True.
    with_v1_to_h : str | bool
        V1_E → H_E feedforward pathway mode (task #31; Sprint 5c toggle):

          - ``"continuous"`` (default) — pathway active throughout the assay
            (required for H rings to emit spikes during grating; Sprint 5b
            diagnostic). Same as legacy ``True``.
          - ``"context_only"`` — pathway built and at full strength, but
            the assay loop is expected to silence it during the probe
            window via ``bundle.v1_to_h.set_active(False)`` (Sprint 5c
            prior-vs-amplifier separation; reviewer rec 5c-2).
          - ``"off"`` — pathway not built (legacy ``False``); H rings will
            be silent during grating epochs.

        Backward compat: ``True`` → ``"continuous"``, ``False`` → ``"off"``.

        Attach only at assay time — training paths keep this off to preserve
        Stage-2 DC-teacher isolation.
    with_feedback_routes : bool, default True
        If False, the H → V1 feedback routes (`feedback_routes.FeedbackRoutes`)
        are still built (for Brian2 object parity) but their gains
        (g_direct, g_SOM) are zeroed so no current flows on H spikes.
        Used by Sprint 5d D5 — run Richter/Tang with feedback OFF and
        V1→H OFF to measure the intrinsic V1-only adaptation baseline.
        Independently toggleable from `with_v1_to_h`.
    with_h_clamp : HClampConfig, optional
        If supplied, build a diagnostic H-clamp pathway (Sprint 5d D3):
        a Poisson afferent pool injecting AMPA drive into a specific H
        channel, weight-gated off by default (construct at build time
        with weights=0 — assay loop toggles `bundle.h_clamp.set_active`).
        See :mod:`expectation_snn.brian2_model.h_clamp`.
    with_preprobe_h_mon : bool, default False
        If True, attach a SpikeMonitor to ``h.e`` exposed as
        ``bundle.h_e_mon``. Used by Sprint 5d diagnostics D1/D2 (pre-probe
        prior index, forecast-vs-memory confusion matrix). Assays then
        call :func:`snapshot_h_counts` / :func:`preprobe_h_rate_hz` at
        the appropriate window boundary. Minimal overhead when off.
    ckpt_dir : str, optional
        Override for checkpoint directory. Defaults to
        ``expectation_snn/data/checkpoints``.
    h_cfg, v1_cfg, fb_cfg : optional configs.

    Returns
    -------
    FrozenBundle
        Ready for ``Network(*bundle.groups, *monitors)``.
    """
    if h_kind not in ("hr", "ht"):
        raise ValueError(f"h_kind must be 'hr' or 'ht', got {h_kind!r}")
    if with_cue and h_kind != "hr":
        raise ValueError(
            "with_cue=True requires h_kind='hr' (Stage-2 was trained on H_R)"
        )
    # Normalize with_v1_to_h: bool -> string (backward compat for diag scripts).
    if with_v1_to_h is True:
        with_v1_to_h = "continuous"
    elif with_v1_to_h is False:
        with_v1_to_h = "off"
    if with_v1_to_h not in ("continuous", "context_only", "off"):
        raise ValueError(
            f"with_v1_to_h must be 'continuous' | 'context_only' | 'off' "
            f"(or bool), got {with_v1_to_h!r}"
        )
    v1_to_h_mode = with_v1_to_h
    ckpt_dir = ckpt_dir or DEFAULT_CKPT_DIR
    stage0_ckpt = os.path.join(ckpt_dir, f"stage_0_seed{seed}.npz")
    stage1_ckpt = os.path.join(
        ckpt_dir,
        f"stage_1_{'hr' if h_kind == 'hr' else 'ht'}_seed{seed}.npz",
    )
    stage2_ckpt = os.path.join(ckpt_dir, f"stage_2_seed{seed}.npz")
    for p in (stage0_ckpt, stage1_ckpt):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"checkpoint missing: {p}")
    if with_cue and not os.path.isfile(stage2_ckpt):
        raise FileNotFoundError(f"Stage-2 checkpoint missing: {stage2_ckpt}")

    # --- rings -----------------------------------------------------------
    v1 = build_v1_ring(config=v1_cfg)
    if h_kind == "hr":
        h = build_h_r(config=h_cfg)
    else:
        h = build_h_t(config=h_cfg)

    # --- checkpoints ----------------------------------------------------
    bias_pA_val, pv_w_n = _load_stage0_into_v1(v1, stage0_ckpt)
    ee_w_n = _load_stage1_into_h(h, stage1_ckpt)

    # --- freeze plasticity ----------------------------------------------
    _freeze_v1_plasticity(v1)
    _freeze_h_plasticity(h)

    # --- feedback routes (H → V1) ---------------------------------------
    fb_cfg_eff = fb_cfg or FeedbackRoutesConfig(g_total=g_total, r=r)
    fb = build_feedback_routes(
        h, v1, fb_cfg_eff, name_prefix=f"s5a_{h_kind}",
    )
    if not with_feedback_routes:
        # Zero both H→V1 routes in-place (Sprint 5d D5 negative control).
        # Leaves Synapses objects attached to the Network (simpler) but
        # no current flows since on_pre deposits are scaled by w==0.
        from ..brian2_model.feedback_routes import set_balance
        set_balance(fb, r=fb_cfg_eff.r, g_total=0.0)

    # --- V1 → H feedforward (task #31; assay-time only) -----------------
    # Modes: "continuous" or "context_only" build the pathway identically
    # (caller toggles via bundle.v1_to_h.set_active(...) for context_only);
    # "off" skips construction entirely, leaving H rings unforced.
    v1_to_h_obj: Optional[V1ToH] = None
    if v1_to_h_mode != "off":
        ff_cfg = v1_to_h_cfg or V1ToHConfig()
        v1_to_h_obj = build_v1_to_h_feedforward(
            v1, h, ff_cfg, name_prefix=f"s5a_ffv1h_{h_kind}",
        )

    # --- H-clamp (Sprint 5d D3 diagnostic; optional) --------------------
    h_clamp_obj: Optional[HClamp] = None
    if with_h_clamp is not None:
        h_clamp_obj = build_h_clamp(
            h, with_h_clamp, name_prefix=f"s5d_hclamp_{h_kind}",
        )

    # --- cue pathway (optional) -----------------------------------------
    cue_A = cue_B = None
    cue_A_to_h = cue_B_to_h = None
    cue_meta = {}
    if with_cue:
        cue_A, cue_B, cue_A_to_h, cue_B_to_h = _build_frozen_cue_pathway(h)
        ck2 = np.load(stage2_ckpt)
        wA = np.asarray(ck2["cue_A_w_final"], dtype=np.float64)
        wB = np.asarray(ck2["cue_B_w_final"], dtype=np.float64)
        liveA = np.asarray(cue_A_to_h.w[:])
        liveB = np.asarray(cue_B_to_h.w[:])
        if wA.shape != liveA.shape or wB.shape != liveB.shape:
            raise ValueError(
                f"Stage-2 cue_{{A,B}}_w_final shape {wA.shape}/{wB.shape} "
                f"!= live {liveA.shape}/{liveB.shape}"
            )
        cue_A_to_h.w[:] = wA
        cue_B_to_h.w[:] = wB
        cue_meta = {
            "cue_A_theta_deg": STAGE2_CUE_CHANNELS[0] * (180.0 / H_N_CHANNELS),
            "cue_B_theta_deg": STAGE2_CUE_CHANNELS[1] * (180.0 / H_N_CHANNELS),
            "cue_A_channel": int(STAGE2_CUE_CHANNELS[0]),
            "cue_B_channel": int(STAGE2_CUE_CHANNELS[1]),
            "cue_A_w_mean": float(wA.mean()),
            "cue_B_w_mean": float(wB.mean()),
        }

    # --- Sprint 5d pre-probe H monitor (optional) -----------------------
    h_e_mon: Optional[SpikeMonitor] = None
    if with_preprobe_h_mon:
        h_e_mon = SpikeMonitor(
            h.e, name=f"s5d_preprobe_h_mon_{h_kind}_seed{seed}",
        )

    # --- groups list for Network ----------------------------------------
    groups: List[object] = []
    groups.extend(h.groups)
    groups.extend(v1.groups)
    groups.extend(fb.groups)
    if v1_to_h_obj is not None:
        groups.extend(v1_to_h_obj.groups)
    if h_clamp_obj is not None:
        groups.extend(h_clamp_obj.groups)
    if h_e_mon is not None:
        groups.append(h_e_mon)
    if with_cue:
        groups.extend([cue_A, cue_B, cue_A_to_h, cue_B_to_h])

    v1_to_h_meta = {}
    if v1_to_h_obj is not None:
        v1_to_h_meta = {
            "v1_to_h_g": float(v1_to_h_obj.g_v1_to_h),
            "v1_to_h_drive_pA": float(v1_to_h_obj.config.drive_amp_v1_to_h_pA),
            "v1_to_h_sigma_ch": float(v1_to_h_obj.config.sigma_channels),
            "v1_to_h_n_syn": int(v1_to_h_obj.kernel_w.size),
        }

    h_clamp_meta = {}
    if h_clamp_obj is not None:
        h_clamp_meta = {
            "h_clamp_target_channel": int(h_clamp_obj.config.target_channel),
            "h_clamp_rate_hz": float(h_clamp_obj.config.clamp_rate_hz),
            "h_clamp_drive_pA": float(h_clamp_obj.config.drive_amp_pA),
            "h_clamp_n_afferents": int(h_clamp_obj.config.n_afferents),
            "h_clamp_window_ms": (
                float(h_clamp_obj.config.window_start_ms),
                float(h_clamp_obj.config.window_end_ms),
            ),
        }

    bundle = FrozenBundle(
        h_ring=h, h_kind=h_kind, v1_ring=v1, fb=fb,
        v1_to_h=v1_to_h_obj,
        h_clamp=h_clamp_obj,
        h_e_mon=h_e_mon,
        cue_A=cue_A, cue_B=cue_B,
        cue_A_to_h=cue_A_to_h, cue_B_to_h=cue_B_to_h,
        groups=groups,
        meta={
            "seed": int(seed),
            "h_kind": h_kind,
            "r": float(r),
            "g_total": float(g_total),
            "v1_bias_pA": float(bias_pA_val),
            "n_pv_to_e": int(pv_w_n),
            "n_ee_w": int(ee_w_n),
            "n_fb_direct": int(fb.kernel_w_direct.size),
            "n_fb_som": int(fb.kernel_w_som.size),
            "g_direct": float(fb.g_direct),
            "g_SOM": float(fb.g_SOM),
            "with_v1_to_h": v1_to_h_mode,
            "v1_to_h_mode": v1_to_h_mode,
            "with_feedback_routes": bool(with_feedback_routes),
            "with_h_clamp": bool(with_h_clamp is not None),
            "with_preprobe_h_mon": bool(with_preprobe_h_mon),
            **v1_to_h_meta,
            **h_clamp_meta,
            **cue_meta,
        },
    )
    return bundle


# --- stimulus helpers (thin wrappers) ------------------------------------

def set_grating(v1: V1Ring, theta_rad: Optional[float],
                contrast: float = 1.0) -> None:
    """Turn V1 stimulus on/off. theta_rad=None → blank (contrast→0)."""
    from ..brian2_model.v1_ring import set_stimulus
    if theta_rad is None or contrast <= 0.0:
        set_stimulus(v1, theta_rad=0.0, contrast=0.0)
    else:
        set_stimulus(v1, theta_rad=float(theta_rad), contrast=float(contrast))


# --- spike-count helpers --------------------------------------------------

def count_spikes_in_window(
    spike_i: np.ndarray,
    spike_t_ms: np.ndarray,
    n_cells: int,
    t_start_ms: float,
    t_end_ms: float,
) -> np.ndarray:
    """Per-cell spike counts in [t_start, t_end). Returns shape (n_cells,)."""
    mask = (spike_t_ms >= t_start_ms) & (spike_t_ms < t_end_ms)
    if not mask.any():
        return np.zeros(n_cells, dtype=np.int64)
    return np.bincount(spike_i[mask], minlength=n_cells).astype(np.int64)


def v1_e_preferred_thetas(v1: V1Ring) -> np.ndarray:
    """Per-cell preferred orientation (rad), = channel index × π/N."""
    return v1.e_channel.astype(np.float64) * (np.pi / V1_N_CHANNELS)


# --- Sprint 5d pre-probe H instrumentation helpers -----------------------

def snapshot_h_counts(bundle: FrozenBundle) -> np.ndarray:
    """Return a copy of the cumulative H_E spike counts per cell.

    Requires ``build_frozen_network(with_preprobe_h_mon=True)`` so
    ``bundle.h_e_mon`` is present.

    Returns
    -------
    counts : np.ndarray, shape (h_ring.e.N,), dtype int64
        Cumulative spike counts at the time of call.
    """
    if bundle.h_e_mon is None:
        raise RuntimeError(
            "bundle.h_e_mon is None — rebuild with with_preprobe_h_mon=True"
        )
    return np.asarray(bundle.h_e_mon.count[:], dtype=np.int64).copy()


def preprobe_h_rate_hz(
    counts_before: np.ndarray,
    counts_after: np.ndarray,
    h_ring: HRing,
    window_ms: float,
) -> np.ndarray:
    """Per-H-channel firing rate during a [counts_before, counts_after) window.

    Parameters
    ----------
    counts_before, counts_after : np.ndarray, shape (h_ring.e.N,)
        Cumulative counts (from :func:`snapshot_h_counts`) at window
        boundaries. ``counts_after - counts_before`` gives per-cell
        spikes in the window.
    h_ring : HRing
        Used for ``e_channel`` mapping (cell index → H channel id) and
        channel count.
    window_ms : float
        Duration of the snapshot window in ms. Must be > 0.

    Returns
    -------
    rate_hz : np.ndarray, shape (N_CHANNELS,), dtype float64
        Mean firing rate (Hz) per H channel = total spikes / (n_cells
        in channel × window_s).
    """
    if window_ms <= 0.0:
        raise ValueError(f"window_ms must be > 0, got {window_ms}")
    delta = (np.asarray(counts_after, dtype=np.int64)
             - np.asarray(counts_before, dtype=np.int64))
    if delta.shape != (int(h_ring.e.N),):
        raise ValueError(
            f"counts shape {delta.shape} != h_ring.e.N={int(h_ring.e.N)}"
        )
    ch = np.asarray(h_ring.e_channel, dtype=np.int64)
    n_ch = int(ch.max()) + 1
    rate_hz = np.zeros(n_ch, dtype=np.float64)
    window_s = float(window_ms) * 1e-3
    for c in range(n_ch):
        m = (ch == c)
        if not m.any():
            rate_hz[c] = 0.0
            continue
        rate_hz[c] = float(delta[m].sum()) / (float(m.sum()) * window_s)
    return rate_hz


# --- self-check / smoke ---------------------------------------------------

if __name__ == "__main__":
    from brian2 import Network, SpikeMonitor, defaultclock, prefs
    from brian2 import seed as b2_seed

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42); np.random.seed(42)

    # 1) Build an H_R bundle with cue + verify load succeeds.
    b = build_frozen_network(h_kind="hr", seed=42, r=1.0, g_total=1.0,
                             with_cue=True)
    print(f"runtime smoke: h_kind=hr  V1 bias={b.meta['v1_bias_pA']:.1f} pA  "
          f"n_ee_w={b.meta['n_ee_w']}  n_pv={b.meta['n_pv_to_e']}  "
          f"cue_A_w_mean={b.meta['cue_A_w_mean']:.3f}  "
          f"cue_B_w_mean={b.meta['cue_B_w_mean']:.3f}")
    assert b.meta["n_ee_w"] == 36672
    assert b.meta["n_pv_to_e"] == 6144
    assert b.cue_A is not None and b.cue_B is not None

    # 2) Short run with cue A on: H_R matched channel should fire more.
    h_mon = SpikeMonitor(b.h_ring.e)
    net = Network(*b.groups, h_mon)
    b.reset_h()
    b.cue_on("A")
    net.run(500 * ms)
    b.cue_off()
    i = np.asarray(h_mon.i[:])
    per_ch = np.bincount(b.h_ring.e_channel[i], minlength=H_N_CHANNELS)
    matched_ch = b.meta["cue_A_channel"]
    peak = int(np.argmax(per_ch))
    print(f"runtime smoke: H_R cue_A 500 ms → peak ch{peak} "
          f"(expected {matched_ch}); per-ch = {per_ch}")
    assert peak == matched_ch, f"cue_A must evoke matched channel (got {peak})"

    # 3) H_T path builds and V1 integrates too.
    b2 = build_frozen_network(h_kind="ht", seed=42, r=1.0, g_total=1.0,
                              with_cue=False)
    print(f"runtime smoke: h_kind=ht  n_ee_w={b2.meta['n_ee_w']}  "
          f"n_fb_direct={b2.meta['n_fb_direct']}  "
          f"g_direct={b2.meta['g_direct']:.3f}  g_SOM={b2.meta['g_SOM']:.3f}")
    print("runtime smoke: PASS")
