"""Fixed orientation-tuned L2/3 basis for task–energy experiments.

This module is intentionally separate from ``simple_net.py``.  It keeps the
same L4 code and sequence generator, but replaces learned dense L4->L2/3 and
dense decoder maps with a fixed local feedforward basis and constrained
orientation readouts. The recurrent GRU/``W_fb`` predictor and a
Dale-sign-constrained SOM/VIP-inspired rate motif supply feedback one abstract
time step later. Training remains ordinary momentum sequences only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_net import N, STEP_DEG, chan, device, l4_code, make_sequences  # noqa: F401


FEEDBACK_MODE_BASELINE = "baseline"
FEEDBACK_MODE_CENTERED = "centered"
FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS = "posterior_prior_excess"
FEEDBACK_MODES = (
    FEEDBACK_MODE_BASELINE,
    FEEDBACK_MODE_CENTERED,
    FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS,
)


def circular_distance_channels() -> torch.Tensor:
    idx = torch.arange(N, device=device)
    d = (idx[:, None] - idx[None, :]).abs()
    return torch.minimum(d, N - d).float()


def local_circular_matrix(sigma_channels: float) -> torch.Tensor:
    """Nonnegative circular Gaussian map, rows normalized to unit sum."""
    sigma = max(float(sigma_channels), 1e-6)
    w = torch.exp(-0.5 * (circular_distance_channels() / sigma).square())
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)


# --- Structural SOM/VIP population circuit (kcontext_20260825 DESIGN.md) ------
N_POP = 9  # SOM and VIP rate units tiling the 36-channel ring, centers c_i = 4i
CIRC_INDEX = {  # Candidate 6 (NOTE_DAMPENING_GEOMETRY addendum 4 section 4)
    "w_ef": 0,     # E->E    top-down feedback onto L2/3 pyramids (direct route)
    "theta_S": 1,  # SST rheobase
    "w_vd": 2,     # E->VIP  bottom-up excitation
    "w_vf": 3,     # E->VIP  top-down excitation
    "w_vs": 4,     # SOM-|VIP
    "w_sv": 5,     # VIP-|SOM blanket disinhibition
    "theta_V": 6,  # VIP rheobase
    "w_pv": 7,     # E->PV->E  co-tuned perisomatic divisive gain (channel-exact)
}
# HARMLESSNESS ladder exponent (addendum 3 section 5): w_pv_init =
# pred_inhib_strength * 2^K, LARGEST K <= 0 passing the C6 init-profile
# basin clause. K = -4 registered by results/init_check_c5.json; UNCHANGED
# for Candidate 6 (addendum 4 section 4).
W_PV_LADDER_K = -4
# Candidate 6 broad-blanket anatomy constant (addendum 4 sections 4-5):
# w_sf_fixed = m_fixed = sqrt(C_FIELD), both NON-TRAINABLE buffers, so
# inh <= C_FIELD * (A_out A_in f) for every reachable parameter setting.
# C_FIELD is MEASURED by the C7 field-equivalence calibration (matched to
# the validated family's own surround s*(fb_pos @ K_sigma4^T)); the value
# below is registered by results/kc_c7_field.json: part (c) fired — the
# Phase-2 alpha0.0 REALIZED inh field is materially stronger than the
# anatomical match c* = 0.03999628378299169 (norm ratio 2.143), so per the
# note section-5 pre-specified rule C_FIELD is RAISED to the realized level
# c_realized (one-scalar fit of the realized field on the same battery).
C_FIELD = 0.08435968302304604


def softplus_inverse(value: float) -> float:
    """raw = ln(expm1(w)); floored at 1e-8 so legacy zero-strength configs
    stay finite (softplus(raw) ~= 1e-8 ~= 0). No-op for this family's inits."""
    return math.log(math.expm1(max(float(value), 1e-8)))


def population_footprints(
    sigma_a: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed anatomy (DESIGN.md section 1): A_in [9,36], A_out [36,9], A_ss [9,9].

    G(d) = exp(-d^2 / 2 sigma_a^2) on circular channel distance; A_in rows
    (afferent pools) sum 1 over channels, A_out rows (per postsynaptic channel)
    sum 1 over the 9 SOM axons, A_ss rows sum 1 over the 9 SOM axons onto VIP.
    """
    sigma = max(float(sigma_a), 1e-6)
    centers = torch.arange(N_POP, device=device).float() * (float(N) / N_POP)
    channels = torch.arange(N, device=device).float()
    d = (channels[None, :] - centers[:, None]).abs()
    d = torch.minimum(d, N - d)  # [9,36] circular distance to unit centers
    g = torch.exp(-0.5 * (d / sigma).square())
    a_in = g / g.sum(dim=1, keepdim=True).clamp_min(1e-6)
    a_out = (g / g.sum(dim=0, keepdim=True).clamp_min(1e-6)).t()
    ds = (centers[None, :] - centers[:, None]).abs()
    ds = torch.minimum(ds, N - ds)  # [9,9] circular distance between centers
    gs = torch.exp(-0.5 * (ds / sigma).square())
    a_ss = gs / gs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return a_in, a_out, a_ss


class SimpleTunedNet(nn.Module):
    """Minimal recurrent predictive circuit with fixed orientation channels.

    L2/3 units retain stable orientation preference through a fixed, local,
    nonnegative feedforward basis.  Readout is constrained: either exact tied
    channel logits or a circular population-vector likelihood using all
    orientation channels.  The SOM/VIP circuit uses nonnegative gains and
    Dale-compliant signs.
    """

    def __init__(
        self,
        hidden: int = 64,
        ff_sigma_channels: float = 1.1,
        ff_gain: float = 1.6,
        decoder_gain: float = 8.0,
        readout: str = "channel",
        population_normalize: bool = True,
        pred_inhib_strength: float = 0.0,
        pred_inhib_sigma_channels: float = 0.65,
        pred_feature_supp_strength: float = 0.0,
        rate_saturation_r_max: float = 0.0,
        rate_saturation_r_half: float = 1.0,
        adapt_strength: float = 0.0,
        adapt_decay: float = 0.85,
        adapt_sigma_channels: float = 1.0,
        local_comp_strength: float = 0.0,
        local_comp_sigma_channels: float = 1.0,
        local_comp_power: float = 1.0,
        local_comp_mode: str = "divisive",
        local_comp_trainable: bool = False,
        recurrent_cell: str = "gru",
    ):
        super().__init__()
        self.hidden = hidden
        self.ff_sigma_channels = float(ff_sigma_channels)
        self.ff_gain = float(ff_gain)
        self.readout = str(readout)
        self.population_normalize = bool(population_normalize)
        self.pred_inhib_strength = float(pred_inhib_strength)
        self.pred_inhib_sigma_channels = float(pred_inhib_sigma_channels)
        self.pred_feature_supp_strength = float(pred_feature_supp_strength)
        self.rate_saturation_r_max = float(rate_saturation_r_max)
        self.rate_saturation_r_half = float(rate_saturation_r_half)
        self.adapt_strength = float(adapt_strength)
        self.adapt_decay = float(adapt_decay)
        self.adapt_sigma_channels = float(adapt_sigma_channels)
        self.local_comp_strength = float(local_comp_strength)
        self.local_comp_sigma_channels = float(local_comp_sigma_channels)
        self.local_comp_power = float(local_comp_power)
        self.local_comp_mode = str(local_comp_mode)
        self.local_comp_trainable = bool(local_comp_trainable)
        self.recurrent_cell = str(recurrent_cell)
        if self.recurrent_cell == "gru":
            self.gru = nn.GRUCell(N, hidden)
        elif self.recurrent_cell == "rnn_tanh":
            self.gru = nn.RNNCell(N, hidden, nonlinearity="tanh")
        else:
            raise ValueError(f"unknown recurrent_cell {self.recurrent_cell!r}")
        self.W_fb = nn.Linear(hidden, N)
        # Structural SOM/VIP population circuit (kcontext_20260825 DESIGN.md
        # sections 1-3). circ_raw keeps its NAME (optimizer / logging / policy
        # compatibility) but now holds the 8 Dale-positive synaptic magnitudes
        # in CIRC_INDEX order. Candidate 6 (NOTE_DAMPENING_GEOMETRY addendum 4
        # section 4): the broad SOM route is ANATOMY — its two gains are the
        # NON-TRAINABLE buffers w_sf_fixed / m_fixed = sqrt(C_FIELD) (the
        # sqrt split keeps the family's own w_sf,init = m_init = sqrt(strength)
        # convention, applied to the C7-CALIBRATED constant), and the 36
        # per-channel som_out_raw output gains are DELETED. Remaining constants
        # derive from the frozen config:
        #   sigma_a    = pred_inhib_sigma_channels / sqrt(2)
        #   w_ef,init  = the static circuit's k_init
        #              = softplus(0) - softplus(0)*relu(softplus(0) - softplus(0)^2)
        # softplus(0) is the STATIC circuit's own float32 gain value (its raws
        # are float32 zeros); evaluating the identity with it reproduces the
        # DESIGN's pinned 0.5457188206402068 bit-for-bit, which float64 ln 2
        # does not (0.545718818...).
        sp0 = float(F.softplus(torch.zeros(())).item())  # float32 softplus(0)
        w_ef_init = sp0 - sp0 * max(sp0 - sp0 * sp0, 0.0)
        sqrt_s = math.sqrt(self.pred_inhib_strength)
        theta_s_init = 0.1 * sqrt_s        # 0.02 — NUMERICALLY UNCHANGED from the registered init
        w_vs_init = 0.5
        w_vf_init = sqrt_s * w_vs_init     # value unchanged (0.1)
        w_vd_init = w_vf_init
        w_sv_init = 0.1
        theta_v_init = theta_s_init
        w_pv_init = self.pred_inhib_strength * (2.0 ** W_PV_LADDER_K)   # K = -4, UNCHANGED (addendum 4 section 4)
        circ_init = [0.0] * len(CIRC_INDEX)
        for name, value in (
            ("w_ef", w_ef_init),
            ("theta_S", theta_s_init), ("w_vd", w_vd_init), ("w_vf", w_vf_init),
            ("w_vs", w_vs_init), ("w_sv", w_sv_init), ("theta_V", theta_v_init),
            ("w_pv", w_pv_init),
        ):
            circ_init[CIRC_INDEX[name]] = softplus_inverse(value)
        self.circ_raw = nn.Parameter(torch.tensor(circ_init, dtype=torch.float32))
        # Candidate 6 (addendum 4 section 4): broad blanket = anatomy.
        self.register_buffer(
            "w_sf_fixed", torch.tensor(math.sqrt(C_FIELD), dtype=torch.float32))
        self.register_buffer(
            "m_fixed", torch.tensor(math.sqrt(C_FIELD), dtype=torch.float32))
        sigma_a = self.pred_inhib_sigma_channels / math.sqrt(2.0)
        a_in, a_out, a_ss = population_footprints(sigma_a)
        self.register_buffer("A_in", a_in, persistent=False)
        self.register_buffer("A_out", a_out, persistent=False)
        self.register_buffer("A_ss", a_ss, persistent=False)
        # Persistent scalar; the harness fills it with the measured R_ref after
        # reference_values() so a_t = r_prev / ref_rate is in baseline units.
        self.register_buffer("ref_rate", torch.ones(()))
        self.decoder_gain_raw = nn.Parameter(torch.tensor(math.log(math.exp(decoder_gain) - 1.0)))
        if self.local_comp_trainable:
            init_strength = max(self.local_comp_strength, 1e-8)
            self.local_comp_strength_raw = nn.Parameter(torch.tensor(math.log(math.expm1(init_strength))))
        self.register_buffer("ff_weight", local_circular_matrix(self.ff_sigma_channels))
        self.register_buffer(
            "pred_inhib_weight",
            local_circular_matrix(self.pred_inhib_sigma_channels),
            persistent=False,
        )
        self.register_buffer(
            "adapt_weight",
            local_circular_matrix(self.adapt_sigma_channels),
            persistent=False,
        )
        self.register_buffer(
            "local_comp_weight",
            local_circular_matrix(self.local_comp_sigma_channels),
            persistent=False,
        )
        angle = 2.0 * math.pi * torch.arange(N, device=device).float() / float(N)
        self.register_buffer("readout_cos", torch.cos(angle))
        self.register_buffer("readout_sin", torch.sin(angle))

    def feedforward(self, l4: torch.Tensor) -> torch.Tensor:
        return self.ff_gain * (l4 @ self.ff_weight.t())

    def decode(self, r: torch.Tensor) -> torch.Tensor:
        gain = F.softplus(self.decoder_gain_raw)
        if self.readout == "channel":
            return gain * r
        if self.readout == "population_vector":
            activity = F.relu(r)
            x = activity @ self.readout_cos
            y = activity @ self.readout_sin
            if self.population_normalize:
                norm = torch.sqrt(x.square() + y.square()).clamp_min(1e-6)
                x = x / norm
                y = y / norm
            logits = x.unsqueeze(-1) * self.readout_cos + y.unsqueeze(-1) * self.readout_sin
            return gain * logits
        raise ValueError(f"unknown tuned readout {self.readout!r}")

    def l23(
        self,
        l4: torch.Tensor,
        fb: torch.Tensor,
        adapt_state: torch.Tensor | None = None,
        r_prev: torch.Tensor | None = None,
        return_internals: bool = False,
    ):
        """Map ``[B,36]`` L4/feedback tensors to nonnegative L2/3 rates.

        Structural SOM/VIP population circuit (kcontext_20260825 DESIGN.md
        section 1). Candidate 6 (NOTE_DAMPENING_GEOMETRY addendum 4 section 4):
        the broad SOM route is ANATOMY — both broad gains are the non-trainable
        buffers ``w_sf_fixed``/``m_fixed`` and the SOM drive is the prediction
        ``f`` (the Candidate-5 recurrent-activity drive is RETIRED, section 6;
        ``r_prev`` is accepted for API compatibility and unused). With
        Dale-positive magnitudes ``g = softplus(circ_raw)`` (CIRC_INDEX order)
        and fixed footprints ``A_in``/``A_out``/``A_ss``:

            u_S = w_sf_fixed*(A_in f) - theta_S   [broad blanket, FIXED drive gain]
            S0  = relu(u_S)
            V   = relu(w_vd*(A_in drive) + w_vf*(A_in f) - w_vs*(A_ss S0) - theta_V)
            S   = relu(u_S - w_sv*mean(V))
            inh = m_fixed * (A_out S)             [FIXED output gain]
            rate = relu(drive + w_ef*f - inh)     [SOM broad subtractive]
            rate = rate / (1 + w_pv*f)            [PV channel-exact divisive —
            the SOLE learnable suppressive gain] -> existing local competition.

        Ceiling theorem (addendum 4 section 2): theta_S >= 0 and w_sv*mean(V)
        >= 0, so inh <= C_FIELD * (A_out A_in f) for EVERY reachable parameter
        setting, both arms. t0 identity is UNCONDITIONAL: fb = 0 => pool_f = 0
        => u_S = -theta_S < 0 => som = 0 => inh = 0, and pv = 0.
        ``return_internals`` additionally yields ``(S, V, inh)``.
        """

        drive = self.feedforward(l4)
        fb_pos = F.relu(fb)
        g = F.softplus(self.circ_raw)
        pool_f = fb_pos @ self.A_in.t()
        u_s = self.w_sf_fixed * pool_f - g[CIRC_INDEX["theta_S"]]   # SOM drive: f, FIXED gain
        som_ff = F.relu(u_s)
        vip = F.relu(
            g[CIRC_INDEX["w_vd"]] * (drive @ self.A_in.t())
            + g[CIRC_INDEX["w_vf"]] * pool_f
            - g[CIRC_INDEX["w_vs"]] * (som_ff @ self.A_ss.t())
            - g[CIRC_INDEX["theta_V"]]
        )
        som = F.relu(u_s - g[CIRC_INDEX["w_sv"]] * vip.mean(dim=-1, keepdim=True))
        inh = self.m_fixed * (som @ self.A_out.t())            # SOM: broad, subtractive, FIXED output gain
        pred_feature_supp = self.pred_feature_supp_strength * fb_pos
        adapt = 0.0 if adapt_state is None else self.adapt_strength * adapt_state
        rate = F.relu(
            drive + g[CIRC_INDEX["w_ef"]] * fb_pos - inh
            - pred_feature_supp - adapt
        )                                                      # '- inh' RESTORED inside the relu
        pv = g[CIRC_INDEX["w_pv"]] * fb_pos                    # NEW: PV drive, channel-exact, same signal
        rate = rate / (1.0 + pv).clamp_min(1e-6)               # NEW: PV divisive, perisomatic
        rate = self.apply_local_competition(rate)
        if self.rate_saturation_r_max > 0.0:
            half = max(self.rate_saturation_r_half, 1e-6)
            rate = self.rate_saturation_r_max * rate / (half + rate)
        if return_internals:
            return rate, (som, vip, inh)
        return rate

    def apply_local_competition(self, rate: torch.Tensor) -> torch.Tensor:
        """Apply current-step activity-driven local L2/3 competition."""
        strength = self.local_comp_effective_strength()
        if not self.local_comp_trainable and self.local_comp_strength <= 0.0:
            return rate
        power = max(self.local_comp_power, 1e-6)
        source = rate if abs(power - 1.0) < 1e-6 else rate.pow(power)
        local_pool = source @ self.local_comp_weight.t()
        if self.local_comp_mode == "divisive":
            return rate / (1.0 + strength * local_pool).clamp_min(1e-6)
        if self.local_comp_mode == "subtractive":
            return F.relu(rate - strength * local_pool)
        raise ValueError(f"unknown local_comp_mode {self.local_comp_mode!r}")

    def local_comp_effective_strength(self) -> torch.Tensor:
        """Return the current nonnegative local competition gain."""
        if self.local_comp_trainable:
            return F.softplus(self.local_comp_strength_raw)
        return torch.tensor(float(self.local_comp_strength), device=device)

    def update_adaptation(self, adapt_state: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Update temporal activity fatigue from previous L2/3 activity only."""
        if self.adapt_strength <= 0.0:
            return adapt_state
        decay = min(max(self.adapt_decay, 0.0), 0.999)
        smooth_r = r @ self.adapt_weight.t()
        return decay * adapt_state + (1.0 - decay) * smooth_r


def resolve_feedback_mode(
    center_over_classes: bool = False,
    feedback_mode: str | None = None,
) -> str:
    """Resolve legacy centering into one explicit shared feedback mode."""

    if feedback_mode is None:
        return (
            FEEDBACK_MODE_CENTERED
            if center_over_classes
            else FEEDBACK_MODE_BASELINE
        )
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"unknown feedback mode {feedback_mode!r}")
    if center_over_classes and feedback_mode != FEEDBACK_MODE_CENTERED:
        raise ValueError(
            "center_over_classes=True conflicts with explicit feedback mode "
            f"{feedback_mode!r}"
        )
    return feedback_mode


def predictive_feedback_evidence(
    raw_logits: torch.Tensor,
    center_over_classes: bool = False,
    feedback_mode: str | None = None,
) -> torch.Tensor:
    """Return nonnegative ``[B,36]`` evidence without changing CE logits.

    ``posterior_prior_excess`` computes ``relu(36*softmax(logits)-1)``. The
    result is used only as the next time step's fed-down state; the raw logits
    remain the next-channel prediction output.
    """

    mode = resolve_feedback_mode(center_over_classes, feedback_mode)
    if mode == FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS:
        posterior = F.softmax(raw_logits, dim=-1)
        return F.relu(float(N) * posterior - 1.0)
    if mode == FEEDBACK_MODE_CENTERED:
        raw_logits = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
    return F.relu(raw_logits)


def forward_seq_tuned(
    net: SimpleTunedNet,
    theta: torch.Tensor,
    fb_scale: float = 1.0,
    center_feedback: bool = False,
    feedback_mode: str | None = None,
    return_internals: bool = False,
):
    """Unroll the tuned network over degree-valued ``theta[B,S]``.

    Returns predictor logits ``[B,S,36]`` and L2/3 rates ``[B,S,36]``. Hidden,
    feedback, and adaptation states start at zero. At each time step L2/3 is
    evaluated first, then adaptation and GRU state update, then ``W_fb`` logits
    are transformed for the following step. Therefore the first-stimulus
    response has zero prior feedback context without disabling normal feedback
    execution. The previous step's L2/3 rate is carried as ``r_prev`` (None at
    t=0) for the population circuit's ongoing-activity route. With
    ``return_internals=True`` a third element ``(S, V, inh)`` is returned,
    each stacked to ``[B,S,·]``; the default two-tuple path is unchanged.
    """
    batch = theta.shape[0]
    h = torch.zeros(batch, net.hidden, device=device)
    pred_down = torch.zeros(batch, N, device=device)
    adapt_state = torch.zeros(batch, N, device=device)
    r_prev = None
    preds, r_seq = [], []
    som_seq, vip_seq, inh_seq = [], [], []
    for t in range(theta.shape[1]):
        out = net.l23(
            l4_code(theta[:, t]),
            fb_scale * pred_down,
            adapt_state,
            r_prev=r_prev,
            return_internals=return_internals,
        )
        if return_internals:
            r, (som, vip, inh) = out
            som_seq.append(som)
            vip_seq.append(vip)
            inh_seq.append(inh)
        else:
            r = out
        r_seq.append(r)
        r_prev = r
        adapt_state = net.update_adaptation(adapt_state, r)
        h = net.gru(r, h)
        pred = net.W_fb(h)
        preds.append(pred)
        pred_down = predictive_feedback_evidence(
            pred,
            center_feedback,
            feedback_mode,
        )
    if return_internals:
        return (
            torch.stack(preds, 1),
            torch.stack(r_seq, 1),
            (
                torch.stack(som_seq, 1),
                torch.stack(vip_seq, 1),
                torch.stack(inh_seq, 1),
            ),
        )
    return torch.stack(preds, 1), torch.stack(r_seq, 1)


def model_config(net: SimpleTunedNet) -> dict:
    return {
        "hidden": int(net.hidden),
        "ff_sigma_channels": float(net.ff_sigma_channels),
        "ff_gain": float(net.ff_gain),
        "decoder_gain": float(F.softplus(net.decoder_gain_raw).detach().cpu().item()),
        "readout": str(net.readout),
        "population_normalize": bool(net.population_normalize),
        "pred_inhib_strength": float(net.pred_inhib_strength),
        "pred_inhib_sigma_channels": float(net.pred_inhib_sigma_channels),
        "pred_feature_supp_strength": float(net.pred_feature_supp_strength),
        "rate_saturation_r_max": float(net.rate_saturation_r_max),
        "rate_saturation_r_half": float(net.rate_saturation_r_half),
        "adapt_strength": float(net.adapt_strength),
        "adapt_decay": float(net.adapt_decay),
        "adapt_sigma_channels": float(net.adapt_sigma_channels),
        "local_comp_strength": float(net.local_comp_strength),
        "local_comp_learned_strength": float(net.local_comp_effective_strength().detach().cpu().item()),
        "local_comp_trainable": bool(net.local_comp_trainable),
        "local_comp_sigma_channels": float(net.local_comp_sigma_channels),
        "local_comp_power": float(net.local_comp_power),
        "local_comp_mode": str(net.local_comp_mode),
        "recurrent_cell": str(net.recurrent_cell),
    }


def build_tuned_from_config(config: dict | None = None) -> SimpleTunedNet:
    config = dict(config or {})
    return SimpleTunedNet(
        hidden=int(config.get("hidden", 64)),
        ff_sigma_channels=float(config.get("ff_sigma_channels", 1.1)),
        ff_gain=float(config.get("ff_gain", 1.6)),
        decoder_gain=float(config.get("decoder_gain", 8.0)),
        readout=str(config.get("readout", "channel")),
        population_normalize=bool(config.get("population_normalize", True)),
        pred_inhib_strength=float(config.get("pred_inhib_strength", 0.0)),
        pred_inhib_sigma_channels=float(config.get("pred_inhib_sigma_channels", 0.65)),
        pred_feature_supp_strength=float(config.get("pred_feature_supp_strength", 0.0)),
        rate_saturation_r_max=float(config.get("rate_saturation_r_max", 0.0)),
        rate_saturation_r_half=float(config.get("rate_saturation_r_half", 1.0)),
        adapt_strength=float(config.get("adapt_strength", 0.0)),
        adapt_decay=float(config.get("adapt_decay", 0.85)),
        adapt_sigma_channels=float(config.get("adapt_sigma_channels", 1.0)),
        local_comp_strength=float(config.get("local_comp_strength", 0.0)),
        local_comp_sigma_channels=float(config.get("local_comp_sigma_channels", 1.0)),
        local_comp_power=float(config.get("local_comp_power", 1.0)),
        local_comp_mode=str(config.get("local_comp_mode", "divisive")),
        local_comp_trainable=bool(config.get("local_comp_trainable", False)),
        recurrent_cell=str(config.get("recurrent_cell", "gru")),
    )
