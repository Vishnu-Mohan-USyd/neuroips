"""H_context + H_prediction architecture (Sprint 5e Fix C, Task #44).

Why this module exists
----------------------
Debugger's Sprint 5e-Diag (commit ``5c8ad05``) showed that a single Wang
bump-attractor ring (``h_ring.py``) is architecturally unable to form
a *pre-trailer* forecast. Under the Richter paradigm, a single ring
driven by the leader cue ends up with its argmax locked on the leader
channel (P = 1.000 on the B5 H-only unit test, biased 80/20 schedule,
360 trials); it cannot simultaneously carry a "current-input" bump and
a "next-expected" bump because a Wang ring admits exactly one attractor
state at a time.

The fix (reviewer's recommendation in
``SPRINT_5D_POST_VERDICT_REVIEW.md``, Researcher's spec for Task #45):
split H into two populations linked by a learned transform:

    H_context  — today's Wang ring; tracks the currently active input.
                 Cue-driven; also receives the bottom-up V1 → H_ctx
                 feedforward pathway (Sprint 5.5).
    H_prediction — parallel Wang ring, *not* cue-driven during the
                 leader epoch. Receives only the learned transform
                 W_ctx_pred of H_context output, plus a moderate
                 always-on V1 → H_pred teacher (the "third factor"
                 supplier for the eligibility-trace rule).

Connectivity (Researcher's spec, tagged ``[R]``):

    [R] V1_E                 → H_ctx_E       (Sprint 5.5, existing)
    [R] V1_E                 → H_pred_E      (teacher, moderate,
                                              built by the caller via
                                              ``feedforward_v1_to_h``)
    [R] H_ctx_E              → H_pred_E      (W_ctx_pred, plastic,
                                              built here)
    [R] H_pred_E             → V1_E apical   (feedback_routes rewired)
    [R] H_pred_E             → V1_SOM        (feedback_routes rewired)
    [R] D_in (2 ch CW/CCW)   → H_ctx_E       (Tang direction state,
                                              built here, Fix D)
    (  H_ctx → V1            — explicitly NONE, per reviewer)

Plasticity (W_ctx_pred) — Frémaux & Gerstner 2015 three-factor
---------------------------------------------------------------
Per-synapse eligibility:

    dxpre/dt    = -xpre  / tau_coinc                  (event-driven)
    dxpost/dt   = -xpost / tau_coinc                  (event-driven)
    delig/dt    = -elig  / tau_elig                   (event-driven)
    on_pre  :   I_e_post += w * drive_amp ; xpre += 1 ; elig += xpost
    on_post :   xpost += 1 ;                              elig += xpre

    Weight update, applied once at trailer onset as a NetworkOperation
    (M(t) integral covers the intended ±75 ms teaching pulse around trailer
    onset — cholinergic / dopaminergic learning gate, Yagishita 2014):

    w += eta * elig * M_integral  -  gamma * (w - w_target) * dt_trial
    w  = clip(w, 0, w_max)
    row_cap:  for each presyn i,  sum_j w[i, j] <= W_row_max
              (any violator rescaled in-place)

References
----------
- Frémaux N, Gerstner W (2015) Front Neural Circuits 9:85 —
  three-factor eligibility-trace learning.
- Yagishita S et al. (2014) Science 345:1616 —
  cholinergic / dopaminergic gating of eligibility consumption.
- Wang X-J (2001) Trends Neurosci 24:455 — ring bump attractor.
- Vogels TP et al. (2011) Science 334:1569 — iSTDP homeostasis,
  precedent for the row-cap + decay pair.
- Saponati M, Vinck M (2023) Nat Commun 14:4985 — predictive-neuron
  alternative rule (reserved as fallback; not used here).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from brian2 import (
    Hz,
    NetworkOperation,
    NeuronGroup,
    PoissonGroup,
    Synapses,
    defaultclock,
    ms,
    network_operation,
    pA,
    second,
)

from .h_ring import (
    HRing,
    HRingConfig,
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER,
    _build_h_ring,
)


# -- constants ---------------------------------------------------------------

# Direction state (Fix D). 2 channels: 0 = CW, 1 = CCW. Poisson source
# afferents are divided evenly across the two channels; the helper
# `set_direction(...)` activates exactly one side.
DEFAULT_N_DIRECTION_AFFERENTS = 32
DIRECTION_CHANNELS = 2  # CW, CCW

# Coincidence-trace time constant (fast pre/post covariance detector).
# Must span the leader->trailer gap (500 ms leader + 0 ms ITI within a
# pair) so x_pre from leader spikes is still non-trivial when the
# trailer post-spikes arrive: e^(-500/500) ~= 0.37 residual. A shorter
# value (e.g. 20 ms) decays to ~e^(-25) ~= 0 long before the trailer
# post-spikes occur, collapsing the coincidence term to self-pairs
# (amplifier signature: pred_argmax == leader, not trailer).
# Task #47 Debugger verdict (H1, empirical match on e^(-gap/tau)).
DEFAULT_TAU_COINC_MS = 500.0

# Researcher's defaults (see module docstring).
DEFAULT_TAU_ELIG_MS = 1000.0          # bridges 500-ms leader + 500-ms trailer.
DEFAULT_ETA = 1e-3                    # three-factor learning rate.
DEFAULT_GAMMA = 1e-4                  # weight-decay homeostasis rate.
DEFAULT_W_TARGET = 0.0075             # rest value of the decay term; must
                                      # match the POST-FIX init mean so
                                      # elig=0 synapses relax to init and
                                      # don't uniformly up-pump into the
                                      # row cap. With DEFAULT_W_INIT_FRAC
                                      # = 0.015, init mean = 0.015 * 1.0
                                      # / 2 = 0.0075. Previously 0.05
                                      # (matched the OLD 0.05 init frac);
                                      # after the 0.015 init fix it became
                                      # 6.7x above init, driving uniform
                                      # up-pump that collapsed W_ctx_pred
                                      # onto the 3.0/192 = 0.015625 row-
                                      # cap floor (attempt #3 evidence).
                                      # Task #47 Debugger verdict H4.
DEFAULT_W_ROW_MAX = 3.0               # cap on sum_j w[i, j] (per presyn).
DEFAULT_W_MAX = 1.0                   # per-synapse absolute ceiling.
DEFAULT_W_INIT_FRAC = 0.015           # uniform(0, w_init_frac * w_max).
                                      # Chosen so init row sum < w_row_max:
                                      # init mean = w_init_frac * w_max / 2 = 0.0075;
                                      # row sum = 0.0075 * n_pre (192) = 1.44 < 3.0.
                                      # Previous 0.05 gave row sum 4.80, triggering
                                      # the row-cap rescale on trial 1 and collapsing
                                      # W_ctx_pred to the uniform floor 3.0/192.
DEFAULT_M_WINDOW_MS = 75.0            # half-width of M(t) pulse around
                                      # each trailer onset.
DEFAULT_M_AMPLITUDE = 1.0             # amplitude of M(t) inside the window.
H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION = 1


# -- config -----------------------------------------------------------------

@dataclass
class HContextPredictionConfig:
    """H_context + H_prediction architecture configuration.

    Parameters
    ----------
    ctx_cfg : HRingConfig or None
        Override for the H_context Wang ring. Defaults to
        ``HRingConfig()`` — the same defaults as the legacy H_R.
    pred_cfg : HRingConfig or None
        Override for the H_prediction Wang ring. Defaults to
        ``HRingConfig()``.
    drive_amp_ctx_pred_pA : float
        Per-spike AMPA current deposited into H_pred E cells by each
        H_ctx → H_pred spike. Sized so a single pre spike delivers a
        sub-threshold EPSP at w = 1.0; the learned weight scales up
        from small init values.
    pred_e_uniform_bias_pA : float
        Uniform, label-blind excitability bias for H_prediction E cells.
        This changes tonic responsiveness but carries no orientation or
        transition content.
    tau_coinc_ms : float
        Fast coincidence-trace time constant (pre/post window).
    tau_elig_ms : float
        Slow eligibility-trace time constant. Must be >= leader_ms +
        ~trailer_ms so the leader-triggered pre trace is still alive
        at the M(t) window around trailer onset (bridges the
        leader-to-trailer gap).
    eta : float
        Three-factor learning rate; weight delta per unit M-integral
        per unit eligibility.
    gamma : float
        Weight-decay rate (per-second); pulls w toward ``w_target``.
    w_target : float
        Decay target for the homeostatic term.
    w_max : float
        Hard upper bound on any individual w.
    w_row_max : float
        Upper bound on per-presyn fan-out ``sum_j w[i, j]``. Violators
        are rescaled in-place (Vogels 2011 iSTDP homeostasis precedent).
    w_init_frac : float
        Initial weights uniform in ``[0, w_init_frac * w_max]``.
    m_window_ms : float
        Half-width of M(t) window around each trailer onset.
    m_amplitude : float
        Amplitude of M(t) inside the window (outside = 0).

    n_direction_afferents : int
        Poisson afferents total (half for CW, half for CCW) that
        project to H_ctx_E and carry rotational direction.
    drive_amp_dir_pA : float
        Per-spike AMPA current deposited into H_ctx_E by each direction
        afferent. Small (≈ 0.2 × V1→H drive) so direction biases the
        bump without overriding cue orientation.
    w_dir_init : float
        Initial direction-afferent weight. Fixed (not plastic).
    """

    ctx_cfg: Optional[HRingConfig] = None
    pred_cfg: Optional[HRingConfig] = None

    # H_ctx -> H_pred learned transform
    drive_amp_ctx_pred_pA: float = 25.0
    pred_e_uniform_bias_pA: float = 0.0
    tau_coinc_ms: float = DEFAULT_TAU_COINC_MS
    tau_elig_ms: float = DEFAULT_TAU_ELIG_MS
    eta: float = DEFAULT_ETA
    gamma: float = DEFAULT_GAMMA
    w_target: float = DEFAULT_W_TARGET
    w_max: float = DEFAULT_W_MAX
    w_row_max: float = DEFAULT_W_ROW_MAX
    w_init_frac: float = DEFAULT_W_INIT_FRAC
    m_window_ms: float = DEFAULT_M_WINDOW_MS
    m_amplitude: float = DEFAULT_M_AMPLITUDE

    # Tang direction state
    n_direction_afferents: int = DEFAULT_N_DIRECTION_AFFERENTS
    drive_amp_dir_pA: float = 16.0    # ≈ 0.2 × V1→H (80 pA)
    w_dir_init: float = 0.5


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable equivalent for scalar config values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _hring_config_to_dict(cfg: Optional[HRingConfig]) -> Optional[dict]:
    """Serialize an optional :class:`HRingConfig` into JSON-safe scalars."""
    if cfg is None:
        return None
    return {
        f.name: _json_safe(getattr(cfg, f.name))
        for f in fields(HRingConfig)
    }


def _hring_config_from_dict(data: Optional[dict]) -> Optional[HRingConfig]:
    """Restore an optional :class:`HRingConfig` from a tolerant dict payload."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise TypeError(f"HRingConfig payload must be dict or None, got {type(data)!r}")
    allowed = {f.name for f in fields(HRingConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return HRingConfig(**kwargs)


def h_context_prediction_config_to_dict(
    cfg: HContextPredictionConfig,
) -> dict:
    """Serialize :class:`HContextPredictionConfig` into JSON-safe metadata.

    The full nested config is stored so Stage-1 checkpoints can recreate
    the exact effective ctx/pred ring settings at assay runtime. Unknown
    future fields are ignored by the restore helper; missing legacy fields
    fall back to dataclass defaults.
    """
    payload: dict[str, Any] = {
        "schema_version": H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
    }
    for f in fields(HContextPredictionConfig):
        value = getattr(cfg, f.name)
        if f.name in ("ctx_cfg", "pred_cfg"):
            value = _hring_config_to_dict(value)
        payload[f.name] = _json_safe(value)
    return payload


def h_context_prediction_config_from_dict(
    payload: dict,
) -> HContextPredictionConfig:
    """Restore :class:`HContextPredictionConfig` from checkpoint metadata."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"HContextPredictionConfig payload must be dict, got {type(payload)!r}"
        )
    allowed = {f.name for f in fields(HContextPredictionConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key in ("ctx_cfg", "pred_cfg"):
            value = _hring_config_from_dict(value)
        kwargs[key] = value
    return HContextPredictionConfig(**kwargs)


def h_context_prediction_config_to_json(
    cfg: HContextPredictionConfig,
) -> str:
    """Return a stable JSON string for checkpoint storage."""
    return json.dumps(h_context_prediction_config_to_dict(cfg), sort_keys=True)


def h_context_prediction_config_from_json(
    payload: object,
) -> HContextPredictionConfig:
    """Restore config from a JSON string/bytes payload stored in ``npz``."""
    if isinstance(payload, np.ndarray):
        payload = payload.item()
    if isinstance(payload, np.bytes_):
        payload = bytes(payload).decode("utf-8")
    elif isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError(
            f"ctx_pred config JSON must be str/bytes/0d ndarray, got {type(payload)!r}"
        )
    return h_context_prediction_config_from_dict(json.loads(payload))


# -- container --------------------------------------------------------------

@dataclass
class HContextPrediction:
    """Bundle returned by :func:`build_h_context_prediction`.

    Attributes
    ----------
    ctx : HRing
        H_context Wang ring (cue + V1→H_ctx + direction inputs).
    pred : HRing
        H_prediction Wang ring (no cue; driven only by ctx→pred and by
        the caller's V1→H_pred teacher).
    ctx_pred : Synapses
        Plastic H_ctx_E → H_pred_E Synapses with per-synapse
        eligibility trace (see module docstring).
    direction : PoissonGroup
        2-channel direction afferent (CW / CCW) feeding H_ctx.
    dir_to_ctx : Synapses
        Fixed direction → H_ctx_E Synapses.
    config : HContextPredictionConfig
    w_ctx_pred_init : np.ndarray, shape (n_syn,)
        Snapshot of the initial uniform ``[0, w_init_frac * w_max]``
        weight vector so validators / replay can audit it.
    groups : list
        All Brian2 objects to splat into ``Network(...)``. Does NOT
        include the modulatory-gate ``NetworkOperation``; build that
        separately via :func:`make_modulatory_gate_operation` and pass
        to ``Network(...)`` alongside these groups.
    """
    ctx: HRing
    pred: HRing
    ctx_pred: Synapses
    direction: PoissonGroup
    dir_to_ctx: Synapses
    config: HContextPredictionConfig
    w_ctx_pred_init: np.ndarray = field(default_factory=lambda: np.array([]))
    groups: List[object] = field(default_factory=list)


# -- internal: W_ctx_pred plastic synapse -----------------------------------

def _build_ctx_pred_synapses(
    ctx: HRing, pred: HRing, cfg: HContextPredictionConfig,
    rng: np.random.Generator,
) -> Tuple[Synapses, np.ndarray]:
    """Build the H_ctx_E → H_pred_E plastic transform.

    All-to-all connectivity; per-synapse eligibility trace with
    coincidence accumulators; weight updates applied externally via
    a NetworkOperation (see :func:`make_modulatory_gate_operation`).

    Returns
    -------
    ctx_pred : Synapses
    w_init : np.ndarray
        Snapshot of the initial weight vector for validator auditing.
    """
    # NOTE: Brian2 reserves the '_pre' / '_post' suffixes on synaptic
    # variables for cross-population access, so we use 'xpre' / 'xpost'
    # (no underscore) as the local coincidence accumulators.
    model = """
    w : 1
    dxpre/dt  = -xpre / tau_coinc_eff : 1 (event-driven)
    dxpost/dt = -xpost / tau_coinc_eff : 1 (event-driven)
    delig/dt  = -elig / tau_elig_eff  : 1 (event-driven)
    """
    on_pre = f"""
    I_e_post += w * {cfg.drive_amp_ctx_pred_pA}*pA
    xpre += 1.0
    elig += xpost
    """
    on_post = """
    xpost += 1.0
    elig += xpre
    """
    namespace = {
        "tau_coinc_eff": cfg.tau_coinc_ms * ms,
        "tau_elig_eff":  cfg.tau_elig_ms  * ms,
    }
    ctx_pred = Synapses(
        ctx.e, pred.e,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="linear",
        namespace=namespace,
        name=f"{ctx.name}_to_{pred.name}",
    )
    # All-to-all (no self: ctx E cells are distinct from pred E cells
    # since they live in separate NeuronGroups, so no self-connection
    # risk; we keep connectivity explicit for clarity).
    ctx_pred.connect(True)
    # Brian2 standalone devices cannot read Synapses.N before build. This
    # connectivity is all-to-all between distinct E groups, so the size is
    # deterministic from the source/target group sizes.
    n_syn = int(len(ctx.e)) * int(len(pred.e))
    w_init = rng.uniform(
        low=0.0,
        high=cfg.w_init_frac * cfg.w_max,
        size=n_syn,
    ).astype(np.float64)
    ctx_pred.w[:] = w_init
    ctx_pred.xpre[:] = 0.0
    ctx_pred.xpost[:] = 0.0
    ctx_pred.elig[:] = 0.0
    return ctx_pred, w_init


# -- internal: direction afferent -------------------------------------------

def _build_direction_input(
    ctx: HRing, cfg: HContextPredictionConfig,
) -> Tuple[PoissonGroup, Synapses]:
    """Build the direction Poisson afferent and its link to H_ctx_E.

    Channels are split evenly across the afferent population:
        afferents [0, n/2)    -> CW  (channel 0)
        afferents [n/2, n)    -> CCW (channel 1)

    Connectivity: each direction afferent projects broadly to ALL
    H_ctx_E cells with a small fixed weight (bias, not orientation-
    selective). Per reviewer's spec this is a scalar rotational-
    direction signal, not a 2D orientation × direction grid.
    """
    n_dir = int(cfg.n_direction_afferents)
    if n_dir % DIRECTION_CHANNELS != 0:
        raise ValueError(
            f"n_direction_afferents ({n_dir}) must be divisible by "
            f"DIRECTION_CHANNELS ({DIRECTION_CHANNELS})."
        )
    direction = PoissonGroup(
        n_dir, rates=0 * Hz, name=f"{ctx.name}_direction",
    )
    dir_to_ctx = Synapses(
        direction, ctx.e,
        model="w : 1",
        on_pre=f"I_e_post += w * {cfg.drive_amp_dir_pA}*pA",
        name=f"{ctx.name}_dir_to_e",
    )
    # Broad fan-out: every direction afferent -> every H_ctx_E cell.
    dir_to_ctx.connect(True)
    dir_to_ctx.w[:] = cfg.w_dir_init
    return direction, dir_to_ctx


# -- factory ----------------------------------------------------------------

def build_h_context_prediction(
    config: Optional[HContextPredictionConfig] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    ctx_name: str = "h_ctx",
    pred_name: str = "h_pred",
) -> HContextPrediction:
    """Construct the H_context + H_prediction bundle.

    Two Wang rings are built via the private ``_build_h_ring`` helper
    (same topology as ``h_ring.build_h_r`` / ``build_h_t``) with
    distinct Brian2 name prefixes so the two populations don't collide.
    The H_ctx → H_pred learned transform and the direction afferent
    are added on top.

    Parameters
    ----------
    config : HContextPredictionConfig, optional
    rng : np.random.Generator, optional
        Numpy generator used for the uniform W_ctx_pred init. If None,
        a seeded default is used (seed=42, the project's convention).
    ctx_name, pred_name : str
        Brian2 name prefixes for the two rings. Kept distinct so both
        can co-exist in the same ``Network``.
    """
    cfg = config or HContextPredictionConfig()
    if rng is None:
        rng = np.random.default_rng(42)

    ctx_cfg = cfg.ctx_cfg or HRingConfig()
    pred_cfg = cfg.pred_cfg or HRingConfig()

    # Build both rings. Each comes with its own E, inh, cue (unused for
    # pred per spec), recurrent E↔E pair-STDP, and Vogels iSTDP.
    ctx = _build_h_ring(ctx_name, ctx_cfg)
    pred = _build_h_ring(pred_name, pred_cfg)
    pred.e.I_bias = cfg.pred_e_uniform_bias_pA * pA

    # Silence H_pred's cue afferents — by spec, H_pred is NOT
    # cue-driven. The cue PoissonGroup still exists (kept so the
    # pred ring is a drop-in HRing for other helpers that expect it);
    # we just clamp its rate to 0.
    pred.cue.rates = 0 * Hz

    # Plastic H_ctx_E → H_pred_E.
    ctx_pred, w_init = _build_ctx_pred_synapses(ctx, pred, cfg, rng)

    # Direction afferent → H_ctx_E (Tang / Fix D).
    direction, dir_to_ctx = _build_direction_input(ctx, cfg)

    bundle = HContextPrediction(
        ctx=ctx, pred=pred,
        ctx_pred=ctx_pred,
        direction=direction, dir_to_ctx=dir_to_ctx,
        config=cfg,
        w_ctx_pred_init=w_init,
    )
    bundle.groups = [
        *ctx.groups,
        *pred.groups,
        ctx_pred,
        direction, dir_to_ctx,
    ]
    return bundle


# -- direction helpers ------------------------------------------------------

def set_direction(bundle: HContextPrediction, direction: int,
                  rate_hz: float = 80.0) -> None:
    """Activate one direction channel (CW=0, CCW=1) at ``rate_hz``.

    Silences the other channel. Used per-Tang-block to tell H_ctx the
    rotational direction. During Richter/Kok assays the direction
    should stay silenced via :func:`silence_direction`.

    Parameters
    ----------
    bundle : HContextPrediction
    direction : {0, 1}
        0 = CW (channel 0 afferents active), 1 = CCW.
    rate_hz : float
        Poisson rate for the active afferent block.
    """
    if direction not in (0, 1):
        raise ValueError(f"direction must be 0 (CW) or 1 (CCW), got {direction}")
    n_dir = int(bundle.direction.N)
    half = n_dir // DIRECTION_CHANNELS
    rates = np.zeros(n_dir, dtype=np.float64)
    rates[direction * half: (direction + 1) * half] = float(rate_hz)
    bundle.direction.rates = rates * Hz


def silence_direction(bundle: HContextPrediction) -> None:
    """Zero all direction-afferent rates."""
    bundle.direction.rates = 0 * Hz


# -- eligibility / modulatory-gate update ------------------------------------

def apply_modulatory_update(
    bundle: HContextPrediction,
    *,
    m_integral: float,
    dt_trial_s: float,
) -> dict:
    """Apply one W_ctx_pred weight update from the current eligibility.

    Meant to be invoked at each trailer onset. Reads current elig per
    synapse (Brian2 updates event-driven traces lazily to the current
    time when read), applies the three-factor learning step, decay,
    clip, and row-cap, then resets elig to 0 inside the spent window.

    Parameters
    ----------
    bundle : HContextPrediction
    m_integral : float
        Integral of the modulatory gate M(t) over this event. For a
        rectangular pulse of amplitude ``m_amplitude`` and half-width
        ``m_window_ms``, ``m_integral = m_amplitude * (2 * m_window_ms) * 1e-3``
        (in seconds).
    dt_trial_s : float
        Time since the last weight update, in seconds; used to scale
        the weight-decay term. Typically ``(leader + trailer + iti) / 1000``.

    Returns
    -------
    stats : dict
        ``{"w_mean_before", "w_mean_after", "dw_sum", "n_capped",
           "elig_mean", "elig_max"}`` — compact telemetry for
        validators / training logs.
    """
    syn = bundle.ctx_pred
    cfg = bundle.config
    # Pull current state. Brian2 applies lazy decay so these values
    # reflect the current simulation time.
    w = np.asarray(syn.w[:], dtype=np.float64).copy()
    elig = np.asarray(syn.elig[:], dtype=np.float64).copy()
    i_pre = np.asarray(syn.i[:], dtype=np.int64).copy()

    # Three-factor update + homeostatic decay (Frémaux-Gerstner 2015).
    # Integrated over the M(t) window using m_integral (seconds * M).
    # gamma has units 1/s; decay over dt_trial_s.
    w_before_mean = float(w.mean())
    dw_learn = cfg.eta * elig * float(m_integral)
    dw_decay = -cfg.gamma * (w - cfg.w_target) * float(dt_trial_s)
    w_new = w + dw_learn + dw_decay

    # Per-synapse absolute clip.
    w_new = np.clip(w_new, 0.0, cfg.w_max)

    # Row-cap on per-presyn fan-out. If sum_j w[i, j] > w_row_max,
    # rescale row i uniformly.
    n_pre = int(i_pre.max()) + 1 if i_pre.size else 0
    if n_pre > 0:
        row_sum = np.bincount(i_pre, weights=w_new, minlength=n_pre)
        over = row_sum > cfg.w_row_max
        scales = np.where(over & (row_sum > 1e-12),
                          cfg.w_row_max / np.maximum(row_sum, 1e-12),
                          1.0)
        w_new = w_new * scales[i_pre]
        n_capped = int(over.sum())
    else:
        n_capped = 0

    syn.w[:] = w_new
    # Consume eligibility (M(t) has "used up" the trace).
    syn.elig[:] = 0.0

    return {
        "w_mean_before": w_before_mean,
        "w_mean_after": float(w_new.mean()),
        "dw_sum": float((w_new - w).sum()),
        "n_capped": n_capped,
        "elig_mean": float(elig.mean()),
        "elig_max": float(elig.max() if elig.size else 0.0),
    }


def make_modulatory_gate_operation(
    bundle: HContextPrediction,
    trailer_onsets_ms: Sequence[float],
    *,
    dt_trial_s: float,
    dt_op_ms: float = 5.0,
    log: Optional[list] = None,
) -> NetworkOperation:
    """Build a ``NetworkOperation`` that fires the M(t) gate update.

    The operation runs every ``dt_op_ms`` during the sim; when the
    current time reaches the NEXT unconsumed trailer onset and remains
    within its post-onset gate window, the update is applied once and
    that onset is marked consumed.

    Parameters
    ----------
    bundle : HContextPrediction
    trailer_onsets_ms : sequence of float
        Absolute simulation times (ms) at which a trailer started.
        Typically ``schedule_start_abs_ms + trial_k * trial_ms + leader_ms``
        for ``k = 0 .. n_trials - 1``.
    dt_trial_s : float
        Length of one trial in seconds (for the decay term).
    dt_op_ms : float
        Cadence of the NetworkOperation. Should be << m_window_ms so
        we don't miss the gate. Default 5 ms (vs 75-ms half-window).
    log : list, optional
        If given, each applied update appends its ``stats`` dict to
        this list for offline analysis.
    """
    cfg = bundle.config
    m_integral = cfg.m_amplitude * (2.0 * cfg.m_window_ms) * 1e-3  # seconds·M
    onsets = np.asarray(trailer_onsets_ms, dtype=np.float64)
    consumed = np.zeros(onsets.shape, dtype=bool)
    half_w = cfg.m_window_ms

    @network_operation(dt=dt_op_ms * ms, when="end",
                       name=f"{bundle.ctx.name}_{bundle.pred.name}_mgate")
    def _gate_step():
        t_now = float(defaultclock.t / ms)
        # Find the smallest unconsumed onset whose post-onset gate is open.
        remaining = np.flatnonzero(~consumed)
        if remaining.size == 0:
            return
        # Earliest unconsumed whose post-onset window has opened by now.
        for k in remaining:
            t_on = float(onsets[k])
            if t_on <= t_now <= t_on + half_w:
                stats = apply_modulatory_update(
                    bundle,
                    m_integral=m_integral,
                    dt_trial_s=dt_trial_s,
                )
                if log is not None:
                    log.append({"k": int(k), "t_ms": t_now, **stats})
                consumed[k] = True
                return
            if t_now < t_on:
                break   # next onsets haven't opened yet

    return _gate_step


# -- self-check / smoke -----------------------------------------------------

def _smoke() -> bool:  # pragma: no cover
    from brian2 import (
        Network, SpikeMonitor, defaultclock, prefs,
        seed as b2_seed,
    )
    from .h_ring import pulse_channel, silence_cue

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(42); np.random.seed(42)
    rng = np.random.default_rng(42)

    # (1) Build cleanly. No name collisions.
    bundle = build_h_context_prediction(rng=rng)
    assert len(bundle.ctx.e) == H_N_CHANNELS * H_N_E_PER
    assert len(bundle.pred.e) == H_N_CHANNELS * H_N_E_PER
    n_syn_expected = (H_N_CHANNELS * H_N_E_PER) ** 2
    assert len(bundle.ctx_pred) == n_syn_expected, (
        f"ctx→pred should be all-to-all (N*N), got {len(bundle.ctx_pred)}"
    )
    assert int(bundle.direction.N) == DEFAULT_N_DIRECTION_AFFERENTS
    print(f"smoke[1] built: ctx_E={len(bundle.ctx.e)}, "
          f"pred_E={len(bundle.pred.e)}, "
          f"ctx_pred_syn={len(bundle.ctx_pred)}, "
          f"dir_aff={int(bundle.direction.N)}")

    # (2) Init weights uniform in [0, w_init_frac * w_max].
    w0 = np.asarray(bundle.ctx_pred.w[:])
    assert w0.min() >= 0.0 and w0.max() <= bundle.config.w_init_frac * bundle.config.w_max + 1e-12
    print(f"smoke[2] init w_ctx_pred: min={w0.min():.4f}, "
          f"max={w0.max():.4f}, mean={w0.mean():.4f}")

    # (3) Quiet: zero input, both rings quiet.
    silence_cue(bundle.ctx); silence_cue(bundle.pred)
    silence_direction(bundle)
    ctx_mon = SpikeMonitor(bundle.ctx.e)
    pred_mon = SpikeMonitor(bundle.pred.e)
    net = Network(*bundle.groups, ctx_mon, pred_mon)
    net.run(500 * ms)
    ctx_rate = ctx_mon.num_spikes / (len(bundle.ctx.e) * 0.5)
    pred_rate = pred_mon.num_spikes / (len(bundle.pred.e) * 0.5)
    print(f"smoke[3] baseline: ctx_E={ctx_rate:.2f} Hz, "
          f"pred_E={pred_rate:.2f} Hz")
    assert ctx_rate < 5.0 and pred_rate < 5.0, (
        f"Both rings should be quiet pre-stage-0 calibration "
        f"(got ctx={ctx_rate}, pred={pred_rate})"
    )

    # (4) Drive ctx cue ch0 + teacher-drive pred ch2 → eligibility on
    # ctx[ch0] → pred[ch2] pairs should grow; we use
    # direction afferent as the teacher proxy for this smoke (in
    # production the teacher is V1→H_pred). Here we simply pulse ctx
    # ch0 and directly kick pred via the direction channel (which
    # targets ALL ctx cells broadly, so is a crude ctx-wide drive —
    # enough to generate post spikes on ctx, not pred; instead we'll
    # feed ctx then read elig to confirm it grew).
    b2_seed(43); np.random.seed(43)
    bundle2 = build_h_context_prediction(rng=np.random.default_rng(43))
    silence_cue(bundle2.pred)
    silence_direction(bundle2)
    pulse_channel(bundle2.ctx, channel=0, rate_hz=300.0)
    net2 = Network(*bundle2.groups)
    # Run 200 ms of ctx drive; confirm ctx→pred elig has grown
    # non-trivially on active ctx pre-synapses.
    net2.run(200 * ms)
    elig = np.asarray(bundle2.ctx_pred.elig[:])
    xpre_arr = np.asarray(bundle2.ctx_pred.xpre[:])
    print(f"smoke[4] after 200 ms ctx drive: elig mean={elig.mean():.4f} "
          f"max={elig.max():.4f}; xpre mean={xpre_arr.mean():.4f}")
    assert xpre_arr.max() > 0.0, "ctx drive should produce pre traces"

    # (5) M(t) gate update: apply a synthetic update using current elig.
    stats = apply_modulatory_update(
        bundle2, m_integral=0.150, dt_trial_s=2.5,
    )
    print(f"smoke[5] M-gate update: {stats}")
    assert stats["elig_mean"] >= 0.0
    assert stats["w_mean_after"] >= 0.0

    # (6) Row cap: inflate weights past w_row_max and confirm cap enforces.
    bundle2.ctx_pred.w[:] = bundle2.config.w_max  # every synapse at ceiling
    bundle2.ctx_pred.elig[:] = 0.0  # no further learning this tick
    stats = apply_modulatory_update(
        bundle2, m_integral=0.0, dt_trial_s=0.0,
    )
    w_after = np.asarray(bundle2.ctx_pred.w[:])
    i_pre = np.asarray(bundle2.ctx_pred.i[:])
    row_sums = np.bincount(i_pre, weights=w_after)
    print(f"smoke[6] row-cap: n_capped={stats['n_capped']}, "
          f"max row sum={row_sums.max():.3f} "
          f"(limit={bundle2.config.w_row_max:.3f})")
    assert row_sums.max() <= bundle2.config.w_row_max + 1e-9, (
        f"row cap violated: {row_sums.max()} > {bundle2.config.w_row_max}"
    )

    # (7) Direction switch: CW / CCW / silence.
    set_direction(bundle2, 0, rate_hz=50.0)
    r = np.asarray(bundle2.direction.rates / Hz)
    assert r[0] > 0 and r[int(bundle2.direction.N) // 2] == 0
    set_direction(bundle2, 1, rate_hz=50.0)
    r = np.asarray(bundle2.direction.rates / Hz)
    assert r[0] == 0 and r[int(bundle2.direction.N) // 2] > 0
    silence_direction(bundle2)
    r = np.asarray(bundle2.direction.rates / Hz)
    assert float(r.sum()) == 0.0
    print("smoke[7] direction CW/CCW/silence: PASS")

    print("h_context_prediction smoke: PASS")
    return True


if __name__ == "__main__":  # pragma: no cover
    ok = _smoke()
    raise SystemExit(0 if ok else 1)
