"""Fixed orientation-tuned L2/3 basis for natural emergence experiments.

This module is intentionally separate from ``simple_net.py``.  It keeps the
same L4 code and sequence generator, but replaces learned dense L4->L2/3 and
dense decoder maps with a fixed local feedforward basis and constrained
orientation readouts.  Training remains ordinary momentum sequences only.
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
        self.gru = nn.GRUCell(N, hidden)
        self.W_fb = nn.Linear(hidden, N)
        self.circ_raw = nn.Parameter(torch.zeros(5))
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

    def l23(self, l4: torch.Tensor, fb: torch.Tensor, adapt_state: torch.Tensor | None = None) -> torch.Tensor:
        drive = self.feedforward(l4)
        fb_pos = F.relu(fb)
        g = F.softplus(self.circ_raw)
        vip = F.relu(g[0] * fb_pos)
        som = F.relu(g[1] * fb_pos - g[2] * vip)
        pred_inhib = self.pred_inhib_strength * (fb_pos @ self.pred_inhib_weight.t())
        pred_feature_supp = self.pred_feature_supp_strength * fb_pos
        adapt = 0.0 if adapt_state is None else self.adapt_strength * adapt_state
        rate = F.relu(drive + g[3] * fb_pos - g[4] * som - pred_inhib - pred_feature_supp - adapt)
        rate = self.apply_local_competition(rate)
        if self.rate_saturation_r_max > 0.0:
            half = max(self.rate_saturation_r_half, 1e-6)
            rate = self.rate_saturation_r_max * rate / (half + rate)
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
    """Return nonnegative feedback evidence without changing raw CE logits."""

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
):
    """Unroll tuned network over [B,S] orientations."""
    batch = theta.shape[0]
    h = torch.zeros(batch, net.hidden, device=device)
    pred_down = torch.zeros(batch, N, device=device)
    adapt_state = torch.zeros(batch, N, device=device)
    preds, r_seq = [], []
    for t in range(theta.shape[1]):
        r = net.l23(l4_code(theta[:, t]), fb_scale * pred_down, adapt_state)
        r_seq.append(r)
        adapt_state = net.update_adaptation(adapt_state, r)
        h = net.gru(r, h)
        pred = net.W_fb(h)
        preds.append(pred)
        pred_down = predictive_feedback_evidence(
            pred,
            center_feedback,
            feedback_mode,
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
    )
