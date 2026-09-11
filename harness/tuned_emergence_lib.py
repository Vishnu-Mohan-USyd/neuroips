"""Fixed orientation-tuned L2/3 basis for task–energy experiments.

This module is intentionally separate from ``simple_net.py``.  It keeps the
same L4 code and sequence generator, but replaces learned dense L4->L2/3 and
dense decoder maps with a fixed local feedforward basis and constrained
orientation readouts. The recurrent GRU/``W_fb`` predictor and a
Dale-sign-constrained two-pool SOM/VIP-inspired rate motif supply feedback one
abstract time step later. Training remains ordinary momentum sequences only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_net import N, STEP_DEG, chan, device, l4_code, make_sequences  # noqa: F401


FEEDBACK_MODE_BASELINE = "baseline"
FEEDBACK_MODE_CENTERED = "centered"
FEEDBACK_MODE_POSTERIOR = "posterior"
FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS = "posterior_prior_excess"
FEEDBACK_MODES = (
    FEEDBACK_MODE_BASELINE,
    FEEDBACK_MODE_CENTERED,
    FEEDBACK_MODE_POSTERIOR,
    FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS,
)
MODEL_ARCHITECTURE_VERSION = "split_som_projected_output_tanh_v9"
TEMPORAL_MODEL_ARCHITECTURE_VERSION = "split_som_projected_output_temporal_v10"
LEARNED_TEMPORAL_MODEL_ARCHITECTURE_VERSION = "split_som_projected_output_learned_temporal_v11"
TEMPORAL_PROTOCOL = {
    "tau_p": 1.0,
    "on_duration": 4.0,
    "gap_duration": 5.0,
    "dt": 0.1,
    "early_times": [0.1, 0.2],
    "late_window": [3.0, 4.0],
    "time_units": "relative",
    "predictor_readout": "mean_early_noiseless",
    "current_task": "mean_noisy_instantaneous_ce",
    "on_integral": "trapezoid",
    "terminal_gap": True,
    "sst_boundary": "sequence_zero_then_blank_decay",
}


def validate_temporal_protocol(protocol: dict | None) -> dict | None:
    """Validate the temporal protocol and its optional noiseless peak readout."""
    if protocol is None:
        return None
    if not isinstance(protocol, dict) or set(protocol) != set(TEMPORAL_PROTOCOL):
        raise ValueError("temporal_protocol must contain the complete v10 protocol")
    for key, expected in TEMPORAL_PROTOCOL.items():
        if key not in ("dt", "predictor_readout") and protocol[key] != expected:
            raise ValueError(f"temporal_protocol {key!r} does not match v10")
    if protocol["predictor_readout"] not in (
        "mean_early_noiseless", "peak_evidence_noiseless"
    ):
        raise ValueError("unknown temporal predictor readout")
    dt = float(protocol["dt"])
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("temporal_protocol dt must be finite and positive")
    for boundary in (
        protocol["on_duration"],
        *protocol["early_times"],
        *protocol["late_window"],
    ):
        steps = float(boundary) / dt
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("temporal_protocol dt must sample every declared time")
    return {
        **protocol,
        "dt": dt,
        "early_times": list(protocol["early_times"]),
        "late_window": list(protocol["late_window"]),
    }


def circular_distance_channels() -> torch.Tensor:
    idx = torch.arange(N, device=device)
    d = (idx[:, None] - idx[None, :]).abs()
    return torch.minimum(d, N - d).float()


def local_circular_matrix(sigma_channels: float) -> torch.Tensor:
    """Nonnegative circular Gaussian map, rows normalized to unit sum."""
    sigma = max(float(sigma_channels), 1e-6)
    w = torch.exp(-0.5 * (circular_distance_channels() / sigma).square())
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)


PREDICTION_SOM_SIGMA_CHANNELS = math.sqrt(2.0)


def prediction_som_footprint() -> torch.Tensor:
    """Fixed feature-local prediction-recipient SOM footprint.

    Unlike ``local_circular_matrix``, this peak-normalized kernel is not
    row-normalized; each diagonal entry is exactly one before dtype rounding.
    """
    d = circular_distance_channels()
    w = torch.exp(
        -0.5 * (d / PREDICTION_SOM_SIGMA_CHANNELS).square()
    )
    return w / w.max().clamp_min(1e-6)


# --- Structural SOM/VIP population circuit (kcontext_20260825 DESIGN.md) ------
N_POP = 9  # SOM and VIP rate units tiling the 36-channel ring, centers c_i = 4i
CIRC_INDEX = {  # Candidate 6 (NOTE_DAMPENING_GEOMETRY addendum 4 section 4)
    "w_ef": 0,     # prediction-gated sensory gain onto L2/3 pyramids
    "theta_S": 1,  # SST rheobase
    "w_vd": 2,     # E->VIP  bottom-up excitation
    "w_sd": 3,     # E->SST/SOM bottom-up excitation
    "w_vs": 4,     # SOM-|VIP
    "w_sv": 5,     # VIP-|SOM local disinhibition
    "theta_V": 6,  # VIP rheobase
    "w_pv": 7,     # E->PV->E  perisomatic divisive gain
}
# HARMLESSNESS ladder exponent (addendum 3 section 5): w_pv_init =
# pred_inhib_strength * 2^K, LARGEST K <= 0 passing the C6 init-profile
# basin clause. K = -4 registered by results/init_check_c5.json; UNCHANGED
# for Candidate 6 (addendum 4 section 4).
W_PV_LADDER_K = -4
M_FIXED_MODE_LEGACY_CLAMP = "legacy_clamp"
M_FIXED_MODE_SOFTPLUS_RAW = "softplus_raw"
M_FIXED_MODES = (
    M_FIXED_MODE_LEGACY_CLAMP,
    M_FIXED_MODE_SOFTPLUS_RAW,
)
# Candidate 6 broad-blanket anatomy constant (addendum 4 sections 4-5):
# w_sf_fixed = m_fixed = sqrt(C_FIELD) at initialization. Both keep legacy
# state_dict keys; axis-stage policy may train their nonnegative effective
# values.
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
    Dale-compliant signs.  In the v4 footprint convention, SOM afferents,
    SOM->E outputs, and local VIP/SOM inhibition have explicit widths; the
    default SOM->E output footprint has twice the variance of the SOM input and
    VIP/SOM footprints.
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
        som_input_sigma_channels: float | None = 2.0,
        som_output_sigma_channels: float | None = None,
        vip_som_sigma_channels: float | None = None,
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
        recurrent_cell: str = "rnn_tanh",
        m_fixed_parameterization: str = M_FIXED_MODE_LEGACY_CLAMP,
        fixed_intrinsic_rheobases: bool = False,
        temporal_protocol: dict | None = None,
        learn_temporal_kinetics: bool = False,
        temporal_tau_e_init: float = 1.0,
        temporal_tau_p_init: float = 1.0,
        sst_response_gain: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.ff_sigma_channels = float(ff_sigma_channels)
        self.ff_gain = float(ff_gain)
        self.readout = str(readout)
        self.population_normalize = bool(population_normalize)
        self.pred_inhib_strength = float(pred_inhib_strength)
        self.pred_inhib_sigma_channels = float(pred_inhib_sigma_channels)
        if som_input_sigma_channels is None:
            som_input_sigma_channels = self.pred_inhib_sigma_channels
        self.som_input_sigma_channels = float(som_input_sigma_channels)
        if som_output_sigma_channels is None:
            som_output_sigma_channels = (
                math.sqrt(2.0) * self.som_input_sigma_channels
            )
        self.som_output_sigma_channels = float(som_output_sigma_channels)
        if vip_som_sigma_channels is None:
            vip_som_sigma_channels = self.som_input_sigma_channels
        self.vip_som_sigma_channels = float(vip_som_sigma_channels)
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
        self.m_fixed_parameterization = str(m_fixed_parameterization)
        if self.m_fixed_parameterization not in M_FIXED_MODES:
            raise ValueError(
                "m_fixed_parameterization must be one of "
                f"{M_FIXED_MODES}, got {self.m_fixed_parameterization!r}"
            )
        self.fixed_intrinsic_rheobases = bool(fixed_intrinsic_rheobases)
        self.temporal_protocol = validate_temporal_protocol(temporal_protocol)
        self.learn_temporal_kinetics = bool(learn_temporal_kinetics)
        self.sst_response_gain = bool(sst_response_gain)
        if self.sst_response_gain and not self.learn_temporal_kinetics:
            raise ValueError("SST response gain requires learned temporal kinetics")
        self.temporal_tau_e_init = float(temporal_tau_e_init)
        self.temporal_tau_p_init = float(temporal_tau_p_init)
        if self.learn_temporal_kinetics:
            if self.temporal_protocol is None:
                raise ValueError("learned temporal kinetics require a temporal protocol")
            for route, initial in (("e", self.temporal_tau_e_init), ("p", self.temporal_tau_p_init)):
                if not math.isfinite(initial) or initial <= 0.0:
                    raise ValueError("initial temporal time constants must be finite and positive")
                # Stable inverse softplus, with identical unrestricted maps for both routes.
                raw = initial + math.log(-math.expm1(-initial))
                self.register_parameter(
                    f"temporal_tau_{route}_raw",
                    nn.Parameter(torch.tensor(raw, dtype=torch.float32)),
                )
        self.model_architecture_version = (
            LEARNED_TEMPORAL_MODEL_ARCHITECTURE_VERSION
            if self.learn_temporal_kinetics else
            TEMPORAL_MODEL_ARCHITECTURE_VERSION
            if self.temporal_protocol is not None
            else MODEL_ARCHITECTURE_VERSION
        )
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
        # section 4): w_sf_fixed and m_fixed keep legacy state_dict keys while
        # using nonnegative effective values. SOM footprints
        # are explicit v4 anatomy:
        #   A_in  sigma_a = som_input_sigma_channels / sqrt(2)
        #   A_out sigma_a = som_output_sigma_channels / sqrt(2)
        #   A_ss  sigma_a = vip_som_sigma_channels / sqrt(2)
        # By default A_out has twice the variance of A_in/A_ss.
        # Remaining constants derive from the frozen config:
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
        w_sd_init = sqrt_s * w_vs_init     # reuses the retired prediction->VIP slot
        w_vd_init = w_sd_init
        w_sv_init = 0.1
        theta_v_init = theta_s_init
        w_pv_init = self.pred_inhib_strength * (2.0 ** W_PV_LADDER_K)   # K = -4, UNCHANGED (addendum 4 section 4)
        circ_init = [0.0] * len(CIRC_INDEX)
        for name, value in (
            ("w_ef", w_ef_init),
            ("theta_S", theta_s_init), ("w_vd", w_vd_init), ("w_sd", w_sd_init),
            ("w_vs", w_vs_init), ("w_sv", w_sv_init), ("theta_V", theta_v_init),
            ("w_pv", w_pv_init),
        ):
            circ_init[CIRC_INDEX[name]] = softplus_inverse(value)
        self.circ_raw = nn.Parameter(torch.tensor(circ_init, dtype=torch.float32))
        # Candidate 6 (addendum 4 section 4): initialized from anatomy; the
        # effective value is nonnegative and can be trained during axis fitting.
        w_sf_init = math.sqrt(C_FIELD)
        if self.sst_response_gain:
            w_sf_init = softplus_inverse(w_sf_init)
        self.w_sf_fixed = nn.Parameter(torch.tensor(w_sf_init, dtype=torch.float32))
        m_init = math.sqrt(C_FIELD)
        if self.m_fixed_parameterization == M_FIXED_MODE_SOFTPLUS_RAW:
            m_init = softplus_inverse(m_init)
        self.m_fixed = nn.Parameter(torch.tensor(m_init, dtype=torch.float32))
        self.register_buffer(
            "theta_s_fixed",
            torch.tensor(theta_s_init, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "theta_v_fixed",
            torch.tensor(theta_v_init, dtype=torch.float32),
            persistent=False,
        )
        som_input_sigma_a = self.som_input_sigma_channels / math.sqrt(2.0)
        som_output_sigma_a = self.som_output_sigma_channels / math.sqrt(2.0)
        vip_som_sigma_a = self.vip_som_sigma_channels / math.sqrt(2.0)
        a_in, _, _ = population_footprints(som_input_sigma_a)
        _, a_out, _ = population_footprints(som_output_sigma_a)
        _, _, a_ss = population_footprints(vip_som_sigma_a)
        self.register_buffer("A_in", a_in, persistent=False)
        self.register_buffer("A_out", a_out, persistent=False)
        self.register_buffer("A_ss", a_ss, persistent=False)
        self.register_buffer(
            "K_pred",
            prediction_som_footprint(),
            persistent=False,
        )
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
        *,
        som_p_state: torch.Tensor | None = None,
        som_p_times: torch.Tensor | None = None,
        prediction_excitation: torch.Tensor | None = None,
    ):
        """Map ``[B,36]`` L4/feedback tensors to nonnegative L2/3 rates.

        The circuit uses separate fixed 36-channel basal and prediction-recipient
        SOM/SST functional pools. Sensory drive reaches the basal pool through a
        broad 9-unit projection, while visual-drive-gated local prediction
        coincidence ``drive * (fb_pos @ K_pred.T)`` reaches the prediction pool.
        VIP remains a 9-unit local disinhibitory motif and targets both pools.
        Basal drive is divided by the basal SST pool, then modulated within
        ``[0, 2*basal]`` by the prediction pool's sigma-2 projected inhibitory
        output. The existing broad PV divisor acts on that modulated drive.
        ``S`` is the equal-mass mean of both raw SST firing pools, so activity
        accounting remains raw. ``return_internals`` additionally yields
        ``(S, V, som_gain, pre_pv_rate, post_pv_rate, exc_feedback_work,``
        ``S_B, S_P)``.

        With an actual ``som_p_state[B,36]``, use that firing state directly.
        Providing ``som_p_times[K]`` samples its exact exponential recruitment
        toward the fixed current target and returns ``[B,K,*]`` tensors. This
        relative-time recruitment is an engineering approximation.

        Learned dual-route kinetics independently filter excitatory prediction
        activation and actual SST firing, without imposing their relative speed.
        ``prediction_excitation`` is effective synaptic/apical activation, not
        another firing population; actual E remains the algebraic rate below.
        Optional SST response gain scales rectified recruitment before firing
        dynamics and activity accounting. This is a phenomenological firing-
        response gain, not equivalent presynaptic plasticity.
        """

        drive = self.feedforward(l4)
        fb_pos = F.relu(fb)
        g = self.circuit_gains()
        theta_s = g[CIRC_INDEX["theta_S"]]
        w_sv = g[CIRC_INDEX["w_sv"]]
        w_sf = self.w_sf_effective()
        b9 = drive @ self.A_in.t()
        b36 = b9 @ self.A_out.t()
        pool_f9 = fb_pos @ self.A_in.t()
        s_ff = F.relu(w_sf * pool_f9 - theta_s)
        vip = F.relu(
            g[CIRC_INDEX["w_vd"]] * b9
            - g[CIRC_INDEX["w_vs"]] * (s_ff @ self.A_ss.t())
            - g[CIRC_INDEX["theta_V"]]
        )
        v36 = vip @ self.A_out.t()
        p36 = fb_pos @ self.K_pred.t()
        pred_sens = drive * p36
        q_b = g[CIRC_INDEX["w_sd"]] * b36
        q_p = pred_sens if self.sst_response_gain else w_sf * pred_sens
        som_b = F.relu(q_b - theta_s - w_sv * v36)
        som_p = F.relu(q_p - theta_s - w_sv * v36)
        if self.sst_response_gain:
            som_p = w_sf * som_p
        if som_p_times is not None and som_p_state is None:
            raise ValueError("som_p_times requires som_p_state")
        if som_p_state is not None:
            if som_p_state.shape != som_p.shape:
                raise ValueError("som_p_state must match the [B,36] SST target")
            if som_p_times is None:
                som_p = som_p_state
            else:
                if self.temporal_protocol is None or som_p_times.ndim != 1:
                    raise ValueError("SST time samples require a temporal net and [K] times")
                tau_p = (
                    self.temporal_time_constants()[1]
                    if self.learn_temporal_kinetics else self.temporal_protocol["tau_p"]
                )
                scaled_times = som_p_times / tau_p
                decay = torch.exp(-scaled_times)[None, :, None]
                recruitment = -torch.expm1(-scaled_times)[None, :, None]
                som_p = som_p_state[:, None, :] * decay + som_p[:, None, :] * recruitment
                samples = som_p_times.numel()
                drive = drive[:, None, :].expand(-1, samples, -1)
                fb_pos = fb_pos[:, None, :].expand(-1, samples, -1)
                som_b = som_b[:, None, :].expand(-1, samples, -1)
                vip = vip[:, None, :].expand(-1, samples, -1)
        som = 0.5 * (som_b + som_p)
        m_effective = self.m_fixed_effective()
        som_gain = m_effective * som
        if prediction_excitation is not None and prediction_excitation.shape != fb_pos.shape:
            raise ValueError("prediction_excitation must match the sampled feedback shape")
        exc_feedback_work = (
            g[CIRC_INDEX["w_ef"]] * drive * fb_pos
            if prediction_excitation is None else drive * prediction_excitation
        )
        basal = drive / (1.0 + m_effective * som_b).clamp_min(1e-6)
        projected_som_p = som_p @ self.pred_inhib_weight.T
        u = (
            g[CIRC_INDEX["w_ef"]] * fb_pos
            if prediction_excitation is None else prediction_excitation
        ) - m_effective * projected_som_p
        pre_pv_rate = basal * (1.0 + torch.tanh(u))
        pv = (
            g[CIRC_INDEX["w_pv"]]
            * pre_pv_rate.mean(dim=-1, keepdim=True)
        ).expand_as(pre_pv_rate)
        post_pv_rate = pre_pv_rate / (1.0 + pv).clamp_min(1e-6)  # NEW: PV divisive, perisomatic
        rate = self.apply_local_competition(post_pv_rate)
        if self.rate_saturation_r_max > 0.0:
            half = max(self.rate_saturation_r_half, 1e-6)
            rate = self.rate_saturation_r_max * rate / (half + rate)
        if return_internals:
            return rate, (
                som,
                vip,
                som_gain,
                pre_pv_rate,
                post_pv_rate,
                exc_feedback_work,
                som_b,
                som_p,
            )
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

    def m_fixed_effective(self) -> torch.Tensor:
        """Return the Dale-positive effective SST/SOM output strength."""
        if self.m_fixed_parameterization == M_FIXED_MODE_SOFTPLUS_RAW:
            return F.softplus(self.m_fixed)
        return self.m_fixed.clamp_min(0.0)

    def w_sf_effective(self) -> torch.Tensor:
        """Return nonnegative SST input gain, or softplus firing-response gain."""
        if self.sst_response_gain:
            return F.softplus(self.w_sf_fixed)
        return self.w_sf_fixed.clamp_min(0.0)

    def temporal_time_constants(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return independent positive E-activation and SST time constants."""
        return F.softplus(self.temporal_tau_e_raw), F.softplus(self.temporal_tau_p_raw)

    def circuit_gains(self) -> torch.Tensor:
        """Return effective nonnegative circuit gains in ``CIRC_INDEX`` order."""
        gains = F.softplus(self.circ_raw)
        if self.fixed_intrinsic_rheobases:
            gains = gains.clone()
            gains[CIRC_INDEX["theta_S"]] = self.theta_s_fixed.to(
                device=gains.device,
                dtype=gains.dtype,
            )
            gains[CIRC_INDEX["theta_V"]] = self.theta_v_fixed.to(
                device=gains.device,
                dtype=gains.dtype,
            )
        return gains

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
            else FEEDBACK_MODE_POSTERIOR
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

    ``posterior`` computes ``softmax(logits)``.  The legacy
    ``posterior_prior_excess`` mode computes ``relu(36*softmax(logits)-1)``.
    The result is used only as the next time step's fed-down state; raw logits
    remain the next-channel prediction output.
    """

    mode = resolve_feedback_mode(center_over_classes, feedback_mode)
    if mode == FEEDBACK_MODE_POSTERIOR:
        return F.softmax(raw_logits, dim=-1)
    if mode == FEEDBACK_MODE_POSTERIOR_PRIOR_EXCESS:
        posterior = F.softmax(raw_logits, dim=-1)
        return F.relu(float(N) * posterior - 1.0)
    if mode == FEEDBACK_MODE_CENTERED:
        raw_logits = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
    return F.relu(raw_logits)


def select_peak_evidence(
    net: SimpleTunedNet, rates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one noiseless snapshot from [..., K, 36], excluding onset index 0.

    Squared orientation-resultant magnitude chooses the snapshot without a
    label. Indices refer to the original K axis; exact ties select the first.
    Gradients flow through the selected rates, not the discrete time choice.
    """
    if rates.ndim < 2 or rates.shape[-2] < 2 or rates.shape[-1] != N:
        raise ValueError("peak evidence requires rates shaped [..., K>=2, 36]")
    positive_rates = rates[..., 1:, :]
    evidence = (positive_rates @ net.readout_cos).square() + (
        positive_rates @ net.readout_sin
    ).square()
    indices = evidence.argmax(dim=-1) + 1
    selected = rates.gather(
        -2, indices[..., None, None].expand(*rates.shape[:-2], 1, N)
    ).squeeze(-2)
    return selected, indices


def _forward_seq_temporal(
    net: SimpleTunedNet,
    theta: torch.Tensor,
    fb_scale: float,
    center_feedback: bool,
    feedback_mode: str | None,
    return_internals: bool,
    return_timecourse: bool,
    clamp_final_probe_sst: bool,
):
    protocol = net.temporal_protocol
    dt = protocol["dt"]
    samples = round(protocol["on_duration"] / dt) + 1
    times = torch.arange(
        samples, device=theta.device, dtype=net.W_fb.weight.dtype
    ) * dt
    early_indices = [round(time / dt) for time in protocol["early_times"]]
    if net.learn_temporal_kinetics:
        tau_e, tau_p = net.temporal_time_constants()
        gap_decay = torch.exp(-protocol["gap_duration"] / tau_p)
        gap_factor = tau_p * (-torch.expm1(-protocol["gap_duration"] / tau_p)) / 40.0
        excitation_decay = torch.exp(-times / tau_e)[None, :, None]
        excitation_recruitment = -torch.expm1(-times / tau_e)[None, :, None]
        excitation_gap_decay = torch.exp(-protocol["gap_duration"] / tau_e)
    else:
        gap_decay = math.exp(-protocol["gap_duration"] / protocol["tau_p"])
        gap_factor = protocol["tau_p"] * (-math.expm1(
            -protocol["gap_duration"] / protocol["tau_p"]
        )) / 40.0
    h = torch.zeros(
        theta.shape[0], net.hidden, device=theta.device, dtype=net.W_fb.weight.dtype
    )
    pred_down = h.new_zeros(theta.shape[0], N)
    som_p_state = torch.zeros_like(pred_down)
    if net.learn_temporal_kinetics:
        excitation_state = torch.zeros_like(pred_down)
    predictions, endpoints, rate_timecourses, gap_integrals = [], [], [], []
    endpoint_internals = [[] for _ in range(8)]
    temporal_internals = [[] for _ in range(8)]
    for stimulus in range(theta.shape[1]):
        state_times = times
        if clamp_final_probe_sst and stimulus == theta.shape[1] - 1:
            state_times = times.clamp_max(protocol["early_times"][-1])
        excitation_kwargs = {}
        if net.learn_temporal_kinetics:
            target = net.circuit_gains()[CIRC_INDEX["w_ef"]] * F.relu(fb_scale * pred_down)
            excitation = (
                excitation_state[:, None, :] * excitation_decay
                + target[:, None, :] * excitation_recruitment
            )
            excitation_kwargs["prediction_excitation"] = excitation
        rates, internals = net.l23(
            l4_code(theta[:, stimulus]),
            fb_scale * pred_down,
            return_internals=True,
            som_p_state=som_p_state,
            som_p_times=state_times,
            **excitation_kwargs,
        )
        endpoints.append(rates[:, -1])
        if return_internals:
            for collected, values in zip(endpoint_internals, internals, strict=True):
                collected.append(values[:, -1])
        if return_timecourse:
            rate_timecourses.append(rates)
            for collected, values in zip(temporal_internals, internals, strict=True):
                collected.append(values)
        # Only the next real stimulus receives this new prediction. Snapshot
        # selection is noiseless; decoder noise is added only by the loss.
        if protocol["predictor_readout"] == "peak_evidence_noiseless":
            predictor_rates, _ = select_peak_evidence(net, rates)
        else:
            predictor_rates = rates[:, early_indices].mean(dim=1)
        h = net.gru(predictor_rates, h)
        prediction = net.W_fb(h)
        predictions.append(prediction)
        pred_down = predictive_feedback_evidence(
            prediction, center_feedback, feedback_mode
        )
        som_p_end = internals[-1][:, -1]
        if return_timecourse:
            gap_integrals.append(som_p_end.mean(dim=-1) * gap_factor)
        # Blank sensory input makes all other accounted rates zero. Include
        # the terminal blank as well; hidden predictor memory is held.
        som_p_state = som_p_end * gap_decay
        if net.learn_temporal_kinetics:
            # Both visually gated targets are zero during a true blank, even
            # though predictor memory is held. Activation adds no population cost.
            excitation_state = excitation[:, -1] * excitation_gap_decay
    result = (torch.stack(predictions, 1), torch.stack(endpoints, 1))
    if return_internals:
        result += (tuple(torch.stack(values, 1) for values in endpoint_internals),)
    if return_timecourse:
        result += ({
            "times": times,
            "rates": torch.stack(rate_timecourses, 1),
            "internals": tuple(torch.stack(values, 1) for values in temporal_internals),
            "gap_integral": torch.stack(gap_integrals, 1),
        },)
    return result


def forward_seq_tuned(
    net: SimpleTunedNet,
    theta: torch.Tensor,
    fb_scale: float = 1.0,
    center_feedback: bool = False,
    feedback_mode: str | None = None,
    return_internals: bool = False,
    *,
    return_timecourse: bool = False,
    clamp_final_probe_sst: bool = False,
):
    """Unroll the tuned network over degree-valued ``theta[B,S]``.

    Returns predictor logits ``[B,S,36]`` and L2/3 rates ``[B,S,36]``. Hidden,
    feedback, and adaptation states start at zero. At each time step L2/3 is
    evaluated first, then adaptation and GRU state update, then ``W_fb`` logits
    are transformed for the following step. Therefore the first-stimulus
    response has zero prior feedback context without disabling normal feedback
    execution. The previous step's L2/3 rate is carried as ``r_prev`` (None at
    t=0) for the population circuit's ongoing-activity route. With
    ``return_internals=True`` a third element
    ``(S, V, som_gain, pre_pv_rate, post_pv_rate, exc_feedback_work, S_B, S_P)``
    is returned, each stacked to ``[B,S,·]``. ``S_P`` is raw firing, before
    its inhibitory output projection; the default
    two-tuple path is unchanged.

    In v10, each stimulus has a fixed-input dwell followed by a true blank.
    The predictor updates once using the mean noiseless E at 0.1 and 0.2.
    ``return_timecourse`` appends a dict containing ``times[K]``,
    ``rates[B,S,K,36]``, the eight corresponding internals, and the already
    population-weighted ``gap_integral[B,S]``. The final-probe clamp is an
    assay intervention: freeze actual S_P after 0.2 until offset, then decay.
    Learned kinetics retain this schedule and learn both independent route
    time constants; excitatory activation also decays naturally through blanks.
    """
    if net.temporal_protocol is not None:
        return _forward_seq_temporal(
            net, theta, fb_scale, center_feedback, feedback_mode,
            return_internals, return_timecourse, clamp_final_probe_sst,
        )
    if return_timecourse or clamp_final_probe_sst:
        raise ValueError("timecourse and SST clamp require a temporal v10 network")
    batch = theta.shape[0]
    h = torch.zeros(batch, net.hidden, device=device)
    pred_down = torch.zeros(batch, N, device=device)
    adapt_state = torch.zeros(batch, N, device=device)
    r_prev = None
    preds, r_seq = [], []
    som_seq, vip_seq, som_gain_seq = [], [], []
    pre_pv_seq, post_pv_seq, exc_feedback_work_seq = [], [], []
    som_b_seq, som_p_seq = [], []
    for t in range(theta.shape[1]):
        out = net.l23(
            l4_code(theta[:, t]),
            fb_scale * pred_down,
            adapt_state,
            r_prev=r_prev,
            return_internals=return_internals,
        )
        if return_internals:
            r, (
                som,
                vip,
                som_gain,
                pre_pv_rate,
                post_pv_rate,
                exc_feedback_work,
                som_b,
                som_p,
            ) = out
            som_seq.append(som)
            vip_seq.append(vip)
            som_gain_seq.append(som_gain)
            pre_pv_seq.append(pre_pv_rate)
            post_pv_seq.append(post_pv_rate)
            exc_feedback_work_seq.append(exc_feedback_work)
            som_b_seq.append(som_b)
            som_p_seq.append(som_p)
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
                torch.stack(som_gain_seq, 1),
                torch.stack(pre_pv_seq, 1),
                torch.stack(post_pv_seq, 1),
                torch.stack(exc_feedback_work_seq, 1),
                torch.stack(som_b_seq, 1),
                torch.stack(som_p_seq, 1),
            ),
        )
    return torch.stack(preds, 1), torch.stack(r_seq, 1)


def model_config(net: SimpleTunedNet) -> dict:
    config = {
        "hidden": int(net.hidden),
        "ff_sigma_channels": float(net.ff_sigma_channels),
        "ff_gain": float(net.ff_gain),
        "decoder_gain": float(F.softplus(net.decoder_gain_raw).detach().cpu().item()),
        "readout": str(net.readout),
        "population_normalize": bool(net.population_normalize),
        "pred_inhib_strength": float(net.pred_inhib_strength),
        "pred_inhib_sigma_channels": float(net.pred_inhib_sigma_channels),
        "som_input_sigma_channels": float(net.som_input_sigma_channels),
        "som_output_sigma_channels": float(net.som_output_sigma_channels),
        "vip_som_sigma_channels": float(net.vip_som_sigma_channels),
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
        "m_fixed_parameterization": str(net.m_fixed_parameterization),
        "fixed_intrinsic_rheobases": bool(net.fixed_intrinsic_rheobases),
        "model_architecture_version": net.model_architecture_version,
    }
    if net.temporal_protocol is not None:
        config["temporal_protocol"] = validate_temporal_protocol(net.temporal_protocol)
    if net.learn_temporal_kinetics:
        config.update(
            learn_temporal_kinetics=True,
            temporal_tau_e_init=net.temporal_tau_e_init,
            temporal_tau_p_init=net.temporal_tau_p_init,
        )
    if net.sst_response_gain:
        config["sst_response_gain"] = True
    return config


def build_tuned_from_config(config: dict | None = None) -> SimpleTunedNet:
    config = dict(config or {})
    requested_architecture = config.get("model_architecture_version")
    temporal_protocol = validate_temporal_protocol(config.get("temporal_protocol"))
    learn_temporal_kinetics = bool(config.get("learn_temporal_kinetics", False))
    expected_architecture = (
        LEARNED_TEMPORAL_MODEL_ARCHITECTURE_VERSION
        if learn_temporal_kinetics else
        TEMPORAL_MODEL_ARCHITECTURE_VERSION
        if temporal_protocol is not None
        else MODEL_ARCHITECTURE_VERSION
    )
    if (
        requested_architecture is not None
        and requested_architecture != expected_architecture
    ):
        raise ValueError(
            "model_architecture_version mismatch: "
            f"expected {expected_architecture!r}, "
            f"got {requested_architecture!r}"
        )
    som_input_sigma_channels = float(
        config.get(
            "som_input_sigma_channels",
            config.get("pred_inhib_sigma_channels", 2.0),
        )
    )
    som_output_sigma_channels = float(
        config.get(
            "som_output_sigma_channels",
            math.sqrt(2.0) * som_input_sigma_channels,
        )
    )
    vip_som_sigma_channels = float(
        config.get("vip_som_sigma_channels", som_input_sigma_channels)
    )
    return SimpleTunedNet(
        hidden=int(config.get("hidden", 64)),
        ff_sigma_channels=float(config.get("ff_sigma_channels", 1.1)),
        ff_gain=float(config.get("ff_gain", 1.6)),
        decoder_gain=float(config.get("decoder_gain", 8.0)),
        readout=str(config.get("readout", "channel")),
        population_normalize=bool(config.get("population_normalize", True)),
        pred_inhib_strength=float(config.get("pred_inhib_strength", 0.0)),
        pred_inhib_sigma_channels=float(config.get("pred_inhib_sigma_channels", 0.65)),
        som_input_sigma_channels=som_input_sigma_channels,
        som_output_sigma_channels=som_output_sigma_channels,
        vip_som_sigma_channels=vip_som_sigma_channels,
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
        recurrent_cell=str(config.get("recurrent_cell", "rnn_tanh")),
        m_fixed_parameterization=str(
            config.get("m_fixed_parameterization", M_FIXED_MODE_LEGACY_CLAMP)
        ),
        fixed_intrinsic_rheobases=bool(
            config.get("fixed_intrinsic_rheobases", False)
        ),
        temporal_protocol=temporal_protocol,
        learn_temporal_kinetics=learn_temporal_kinetics,
        temporal_tau_e_init=float(config.get("temporal_tau_e_init", 1.0)),
        temporal_tau_p_init=float(config.get("temporal_tau_p_init", 1.0)),
        sst_response_gain=bool(config.get("sst_response_gain", False)),
    )
