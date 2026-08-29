"""
simple_net.py - minimal 3-component V1/V2-style model, from scratch (pure PyTorch, single file).

A handcoded sensory layer feeds a trainable cortical layer whose top-down feedback can either
SHARPEN or DAMPEN the expected stimulus through a Dale-compliant SOM/VIP microcircuit. This is the
shared core imported by both experiment directories beside it:
  - phaseA_somvip/ : the sharpen/dampen regime EMERGES from the energy-vs-task ratio (one fixed circuit).
  - phaseB_somvip/ : ONE trained net SWITCHES regime at runtime via a context bit (no retraining).
See each directory's RESULTS.md for the scientific results.

Three components:
  1. L4 ring  : handcoded, FIXED circular-Gaussian population code of the input orientation
                (N=36 channels, 5 deg/channel). No parameters.
  2. L2/3 ring: trainable rectified feedforward map W_ff: L4 -> L2/3 (the representation), into which
                the top-down prediction is fed back (see the l23 operator below).
  3. GRU      : reads L2/3, carries temporal context, and projects a next-orientation prediction (W_fb)
                back DOWN into L2/3 at the following step.

The l23 feedback operator (ONE shared knob-set; the regime lives in learned gain VALUES, not in extra
machinery). With drive = W_ff(l4):
  - use_circuit=True  -> Dale SOM/VIP microcircuit (the headline used by Phase A/B):
        vip = relu(g_v*fb [+ softplus(g_ctx)*ctx if context]);  som = relu(g_s*fb - g_sv*vip);
        r   = relu(drive + g_e*fb - g_ps*som).  All five gains >=0 (softplus of circ_raw); the minus
        signs are STRUCTURAL (cell type / Dale's law), never negative weights. VIP-dominant -> net
        excitation (sharpen); SOM-dominant -> net inhibition (dampen). context=True (task #43B) adds a
        runtime context drive into VIP, the only ctx->output path.
  - else feedback_mode='subtractive' -> r = relu(drive - fb)  (Rao-Ballard predictive-coding variant,
        task #41) -- an ALTERNATIVE operator, not used by the Phase A/B headline.
  - else (default 'additive')        -> r = relu(drive + fb)  (the original reinforcing operator).
  signed_fb (in forward_seq) gates only the fed-DOWN copy: False=relu(pred) (default), True=signed
  push-pull (task #38). The additive/subtractive/signed_fb operators are kept for the record; the
  committed results all use the SOM/VIP circuit (use_circuit=True).

Two losses only:
  - representation : cross-entropy decoding of orientation from L2/3
                     (phase 1 -> CURRENT orientation; phase 2 -> NEXT orientation = "predict next step").
  - energy budget  : mean |L2/3 activity|   (its weight lam_energy is the sharpen<->dampen knob).

Two phases:
  - Phase 1 (2000 steps): static random orientations; train W_ff + decoder (feedback off).
  - Phase 2 (4000 steps in the results): temporal sequences; train GRU + W_fb (and the SOM/VIP gains
                            circ_raw when use_circuit=True) to predict the next step, with W_ff +
                            decoder frozen so the feedback itself must carry the prediction. (Phase B's
                            driver uses a custom loop in the same spirit that ALSO trains the context
                            gain g_ctx_raw plus an external noisy readout -- see phaseB_somvip/.)

NOTE on sequence defaults: make_sequences()/phase2() DEFAULT to mode='markov' (S=8, p_stay=0.8, 3000
steps), but every committed Phase A/B result uses mode='momentum', S=12, p_stay=0.9, 4000 steps, which
the driver scripts pass EXPLICITLY. 'momentum' (a sticky hidden acceleration the velocity integrates)
is the regime a memoryful net can anticipate and beat the 1-step persistence baseline on.

Reproducibility: this module seeds torch at import (torch.manual_seed(0) below) and builds `prefs` on
`device` at import; `device` auto-selects CUDA if available else CPU (CPU works, just slower).
"""
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

N = 36                      # orientation channels
STEP_DEG = 180.0 / N        # 5 deg per channel
SIGMA = 12.0                # L4 tuning width (deg)
prefs = torch.arange(N, device=device).float() * STEP_DEG     # preferred orientation of each channel


def circ_dist(a, b, period=180.0):
    d = (a - b).abs() % period
    return torch.minimum(d, period - d)


def l4_code(theta_deg):
    """Handcoded FIXED L4 ring: [B] orientations(deg) -> [B,N] circular-Gaussian population code."""
    d = circ_dist(theta_deg[:, None], prefs[None, :])         # [B,N]
    return torch.exp(-0.5 * (d / SIGMA) ** 2)


def chan(theta_deg):
    """orientation(deg) -> nearest channel index."""
    return (theta_deg / STEP_DEG).round().long() % N


class SimpleNet(nn.Module):
    def __init__(self, hidden=64, use_circuit=False, context=False):
        super().__init__()
        self.W_ff    = nn.Linear(N, N)          # L4  -> L2/3            (phase 1)
        self.gru     = nn.GRUCell(N, hidden)    # L2/3 -> RNN            (phase 2)
        self.W_fb    = nn.Linear(hidden, N)     # RNN -> L2/3 feedback   (phase 2)
        self.decoder = nn.Linear(N, N)          # L2/3 -> orientation logits (representation readout)
        self.hidden  = hidden
        self.signed_fb = False                  # #38: fed-down relu gate; True -> SIGNED push-pull feedback
        self.feedback_mode = 'additive'         # #41: 'additive' = reinforce prediction (ORIGINAL); 'subtractive' = cancel it
        self.use_circuit = use_circuit          # #43: SOM/VIP Dale microcircuit -> sharpen/dampen emerges from learned gains
        self.context = context                  # #43B: runtime context drive into VIP -> regime switches live (no retrain)
        # Optional L2/3 divisive competition. Defaults are inactive so the
        # committed checkpoints and canonical reproduction scripts keep the
        # original model path unless a driver explicitly opts in.
        self.l23_competition_strength = 0.0
        self.l23_competition_sigma_channels = 2.0
        self.l23_competition_radius = 4
        self.l23_competition_global_strength = 0.0
        # Optional local recurrent L2/3 inhibition. This is a default-off,
        # topology-only channel interaction applied before the firing-rate
        # ReLU, using the same circular kernel for every stimulus and regime.
        self.l23_local_inhibition_strength = 0.0
        self.l23_local_inhibition_sigma_channels = 1.5
        self.l23_local_inhibition_radius = 3
        self.l23_local_inhibition_center_weight = 0.0
        # Optional feedback-gated L2/3 inhibition. Defaults inactive; when
        # enabled, the top-down prediction recruits the same circular local
        # inhibitory kernel at every timestep and for every stimulus.
        self.l23_feedback_gated_inhibition_strength = 0.0
        self.l23_feedback_gated_inhibition_sigma_channels = 1.5
        self.l23_feedback_gated_inhibition_radius = 3
        self.l23_feedback_gated_inhibition_center_weight = 1.0
        # Optional feedback-to-SOM local inhibitory projection. Defaults
        # inactive; when enabled, top-down feedback is pooled over the circular
        # orientation topology and recruits SOM inhibition at every timestep.
        self.som_feedback_pool_strength = 0.0
        self.som_feedback_pool_sigma_channels = 1.5
        self.som_feedback_pool_radius = 3
        self.som_feedback_pool_center_weight = 0.0
        # Optional topographic SOM/VIP routing. Defaults inactive; when
        # enabled, feedback recruits a narrow SOM inhibitory drive and a
        # broader VIP/disinhibitory drive through fixed circular kernels.
        self.somvip_topographic_som_strength = 0.0
        self.somvip_topographic_som_sigma_channels = 0.75
        self.somvip_topographic_som_radius = 2
        self.somvip_topographic_som_center_weight = 1.0
        self.somvip_topographic_vip_strength = 0.0
        self.somvip_topographic_vip_sigma_channels = 2.5
        self.somvip_topographic_vip_radius = 5
        self.somvip_topographic_vip_center_weight = 0.0
        # Optional prediction-error-style feedback inside the circuit path.
        # Defaults inactive; when enabled, the same predicted top-down drive
        # subtracts from feedforward input before the learned SOM/VIP balance.
        self.l23_prediction_error_strength = 0.0
        if use_circuit:
            # 5 non-negative gains via softplus(raw): [g_v, g_s, g_sv, g_e, g_ps]. Signs are STRUCTURAL (cell type),
            # only magnitudes learn:  g_v top-down->VIP,  g_s top-down->SOM,  g_sv VIP-|SOM,  g_e top-down->Pyr(exc),
            # g_ps SOM-|Pyr.   VIP-dominant -> net excitation (sharpen);  SOM-dominant -> net inhibition (dampen).
            self.circ_raw = nn.Parameter(torch.zeros(5))
            if context:
                # #43B: external context EXCITES VIP (softplus>=0 drive). ctx high -> VIP up -> SOM off -> sharpen;
                # ctx low -> SOM dominates -> dampen. The ONLY context->output path is through VIP.
                self.g_ctx_raw = nn.Parameter(torch.zeros(1))

    def _apply_l23_competition(self, r):
        """General local divisive normalization over the L2/3 orientation ring."""
        local_strength = float(getattr(self, 'l23_competition_strength', 0.0))
        global_strength = float(getattr(self, 'l23_competition_global_strength', 0.0))
        if local_strength <= 0.0 and global_strength <= 0.0:
            return r

        denom = torch.ones_like(r)
        if local_strength > 0.0:
            sigma = max(float(getattr(self, 'l23_competition_sigma_channels', 2.0)), 1e-6)
            radius = max(0, int(getattr(self, 'l23_competition_radius', 4)))
            local = torch.zeros_like(r)
            norm = 0.0
            for off in range(-radius, radius + 1):
                w = torch.exp(torch.tensor(-0.5 * (off / sigma) ** 2, device=r.device, dtype=r.dtype))
                local = local + w * torch.roll(r, shifts=off, dims=1)
                norm += float(w.item())
            denom = denom + local_strength * (local / max(norm, 1e-6))

        if global_strength > 0.0:
            denom = denom + global_strength * r.mean(dim=1, keepdim=True)
        return r / denom.clamp_min(1e-6)

    def _apply_l23_local_inhibition(self, pre):
        """Subtractive local recurrent inhibition over the orientation ring."""
        strength = float(getattr(self, 'l23_local_inhibition_strength', 0.0))
        if strength <= 0.0:
            return F.relu(pre)

        sigma = max(float(getattr(self, 'l23_local_inhibition_sigma_channels', 1.5)), 1e-6)
        radius = max(0, int(getattr(self, 'l23_local_inhibition_radius', 3)))
        center_weight = max(0.0, float(getattr(self, 'l23_local_inhibition_center_weight', 0.0)))
        source = F.relu(pre)
        inhibition = torch.zeros_like(source)
        norm = 0.0
        for off in range(-radius, radius + 1):
            w = torch.exp(torch.tensor(-0.5 * (off / sigma) ** 2, device=source.device, dtype=source.dtype))
            if off == 0:
                w = w * center_weight
            inhibition = inhibition + w * torch.roll(source, shifts=off, dims=1)
            norm += float(w.item())
        if norm <= 1e-6:
            return source
        return F.relu(pre - strength * (inhibition / norm))

    def _apply_l23_feedback_gated_inhibition(self, pre, fb):
        """Prediction-driven local inhibitory recruitment over the L2/3 ring."""
        strength = float(getattr(self, 'l23_feedback_gated_inhibition_strength', 0.0))
        if strength <= 0.0:
            return pre

        sigma = max(float(getattr(self, 'l23_feedback_gated_inhibition_sigma_channels', 1.5)), 1e-6)
        radius = max(0, int(getattr(self, 'l23_feedback_gated_inhibition_radius', 3)))
        center_weight = max(0.0, float(getattr(self, 'l23_feedback_gated_inhibition_center_weight', 1.0)))
        source = F.relu(fb)
        inhibition = torch.zeros_like(source)
        norm = 0.0
        for off in range(-radius, radius + 1):
            w = torch.exp(torch.tensor(-0.5 * (off / sigma) ** 2, device=source.device, dtype=source.dtype))
            if off == 0:
                w = w * center_weight
            inhibition = inhibition + w * torch.roll(source, shifts=off, dims=1)
            norm += float(w.item())
        if norm <= 1e-6:
            return pre
        return pre - strength * (inhibition / norm)

    def _circular_feedback_pool(self, fb, sigma, radius, center_weight):
        """Fixed nonnegative circular pooling of top-down feedback."""
        source = F.relu(fb)
        pool = torch.zeros_like(source)
        norm = 0.0
        for off in range(-radius, radius + 1):
            w = torch.exp(torch.tensor(-0.5 * (off / sigma) ** 2, device=source.device, dtype=source.dtype))
            if off == 0:
                w = w * center_weight
            pool = pool + w * torch.roll(source, shifts=off, dims=1)
            norm += float(w.item())
        if norm <= 1e-6:
            return torch.zeros_like(source)
        return pool / norm

    def _som_feedback_pool(self, fb):
        """Circular feedback pool for prediction-driven SOM recruitment."""
        strength = max(0.0, float(getattr(self, 'som_feedback_pool_strength', 0.0)))
        if strength <= 0.0:
            return torch.zeros_like(fb)

        sigma = max(float(getattr(self, 'som_feedback_pool_sigma_channels', 1.5)), 1e-6)
        radius = max(0, int(getattr(self, 'som_feedback_pool_radius', 3)))
        center_weight = max(0.0, float(getattr(self, 'som_feedback_pool_center_weight', 0.0)))
        return strength * self._circular_feedback_pool(fb, sigma, radius, center_weight)

    def _topographic_somvip_feedback(self, fb):
        """Shared narrow-SOM / broad-VIP routing of top-down feedback."""
        som_strength = max(0.0, float(getattr(self, 'somvip_topographic_som_strength', 0.0)))
        vip_strength = max(0.0, float(getattr(self, 'somvip_topographic_vip_strength', 0.0)))
        if som_strength <= 0.0 and vip_strength <= 0.0:
            z = torch.zeros_like(fb)
            return z, z

        som_drive = torch.zeros_like(fb)
        if som_strength > 0.0:
            som_sigma = max(float(getattr(self, 'somvip_topographic_som_sigma_channels', 0.75)), 1e-6)
            som_radius = max(0, int(getattr(self, 'somvip_topographic_som_radius', 2)))
            som_center = max(0.0, float(getattr(self, 'somvip_topographic_som_center_weight', 1.0)))
            som_drive = som_strength * self._circular_feedback_pool(fb, som_sigma, som_radius, som_center)

        vip_drive = torch.zeros_like(fb)
        if vip_strength > 0.0:
            vip_sigma = max(float(getattr(self, 'somvip_topographic_vip_sigma_channels', 2.5)), 1e-6)
            vip_radius = max(0, int(getattr(self, 'somvip_topographic_vip_radius', 5)))
            vip_center = max(0.0, float(getattr(self, 'somvip_topographic_vip_center_weight', 0.0)))
            vip_drive = vip_strength * self._circular_feedback_pool(fb, vip_sigma, vip_radius, vip_center)

        return som_drive, vip_drive

    def l23(self, l4, fb, ctx=0.0):
        # #41 feedback operator.  WIRING (forward_seq): the fed-down `fb` at step t is the prediction made
        # one step earlier of the CURRENT input theta_t, so it lands on exactly theta_t's channel.
        #   'additive'    (default): r = relu(W_ff(l4) + fb)  -> reinforces the predicted feature (EXACT original line).
        #   'subtractive'          : r = relu(W_ff(l4) - fb)  -> CANCELS the prediction (Rao-Ballard predictive coding):
        #       a correctly-anticipated current stimulus subtracts at its own channel (center -> trough), the
        #       unpredicted surround passes through at its FF level; the OUTER relu keeps rates >=0 (positive
        #       prediction ERROR / surprise).  Pure sign-flip -- single variable, no re-indexing.
        drive = self.W_ff(l4)
        if getattr(self, 'use_circuit', False):
            # #43 SOM/VIP microcircuit. All gains >=0 (softplus); the inhibition is the STRUCTURAL minus sign
            # (cell type), NOT a negative weight -- Dale-compliant. Sharpen vs dampen is whatever the learned
            # SOM-vs-VIP balance settles to; the architecture is identical in both regimes.
            g      = F.softplus(self.circ_raw)            # [5] = g_v,g_s,g_sv,g_e,g_ps (non-negative magnitudes)
            topo_som, topo_vip = self._topographic_somvip_feedback(fb)
            vip_in = g[0] * fb + topo_vip                 # top-down -> VIP plus broad topographic disinhibition
            if getattr(self, 'context', False):
                vip_in = vip_in + F.softplus(self.g_ctx_raw) * ctx   # #43B context EXCITES VIP -> disinhibits SOM -> sharpen
            vip = F.relu(vip_in)
            fb_pool = self._som_feedback_pool(fb)
            som = F.relu(g[1] * fb + fb_pool + topo_som - g[2] * vip)  # top-down -> SOM, VIP -| SOM (disinhibition)
            pe_strength = float(getattr(self, 'l23_prediction_error_strength', 0.0))
            pe_drive = pe_strength * F.relu(fb)
            pre = self._apply_l23_feedback_gated_inhibition(
                drive - pe_drive + g[3] * fb - g[4] * som,
                fb,
            )
            r = self._apply_l23_local_inhibition(
                pre
            )  # Pyr: FF drive + circuit feedback/error, then local L2/3 inhibition
            return self._apply_l23_competition(r)
        if getattr(self, 'feedback_mode', 'additive') == 'subtractive':
            pre = self._apply_l23_feedback_gated_inhibition(drive - fb, fb)
            return self._apply_l23_competition(self._apply_l23_local_inhibition(pre))
        pre = self._apply_l23_feedback_gated_inhibition(drive + fb, fb)
        return self._apply_l23_competition(self._apply_l23_local_inhibition(pre))


# ---------------- Phase 1: static representation ----------------
def phase1(net, steps=2000, batch=128, lam_energy=0.02, lr=1e-3):
    opt = torch.optim.Adam(list(net.W_ff.parameters()) + list(net.decoder.parameters()), lr=lr)
    print("\n=== PHASE 1: learn L4->L2/3 representation (static orientations) ===")
    curve = []
    for s in range(1, steps + 1):
        theta = torch.randint(0, N, (batch,), device=device).float() * STEP_DEG
        r = net.l23(l4_code(theta), torch.zeros(batch, N, device=device))
        logits = net.decoder(r)
        target = chan(theta)
        loss = F.cross_entropy(logits, target) + lam_energy * r.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 1 or s % 200 == 0:
            acc = (logits.argmax(-1) == target).float().mean().item()
            print(f"  step {s:5d}  loss {loss.item():.3f}  rep_acc {acc*100:5.1f}%  mean|r| {r.abs().mean().item():.3f}")
            curve.append((s, loss.item(), acc))
    return curve


# ---------------- sequences: easy march / memoryless-switch Markov / structured momentum ----------------
def _seq_latents(ch):
    """Observable per-step kinematics of a channel trajectory ``ch`` [B,S] (integer
    channel indices), used by ``make_sequences(..., return_latents=True)``.

    Both are CAUSAL (backward) finite differences in the signed shortest-path channel
    metric, zero-padded at t=0 (the boundary where they are undefined):

        vel[:, t]   = signed(ch[:, t] - ch[:, t-1])   for t>=1,  vel[:, 0]   = 0
        accel[:, t] = vel[:, t] - vel[:, t-1]          for t>=1,  accel[:, 0] = 0

    so ``vel`` is the signed channel velocity carried INTO step t (channels/step) and
    ``accel`` its per-step difference (channels/step^2). For ``mode='momentum'`` these
    equal the generator's realised velocity / acceleration (shifted one step and
    clamp-aware); they are well-defined for every mode. Returns (vel, accel), both
    int64 [B,S] on ``ch``'s device.
    """
    vel = torch.zeros_like(ch)
    vel[:, 1:] = (ch[:, 1:] - ch[:, :-1] + N // 2) % N - N // 2
    accel = torch.zeros_like(ch)
    accel[:, 1:] = vel[:, 1:] - vel[:, :-1]
    return vel, accel


def make_sequences(batch, S, mode='markov', p_stay=0.8, vels=(-3, -1, 1, 3), step_channels=3, vmax=4,
                   return_latents=False):
    """Three regimes selected by `mode`:
      'march'   : EASY baseline. Random start, one random direction (+/-), CONSTANT 15 deg/step.
                  Fully predictable once the direction is known -> the RNN can lock onto one direction.
      'markov'  : HARD, MEMORYLESS switches. A hidden drift velocity on a sticky Markov chain -- each step
                  it persists w.p. p_stay, else switches to a RANDOM, INDEPENDENT one in `vels`. Because a
                  switch is independent of the past, NO model can predict it -> persistence is unbeatable.
      'momentum': HARD, STRUCTURED switches. A hidden ACCELERATION (sticky Markov on {-1,0,+1} channels/step^2,
                  persists w.p. p_stay) that the velocity integrates: v_{t+1}=clip(v_t+a_t, +/-vmax). The
                  velocity changes SMOOTHLY, so a turn is telegraphed by the recent trend -- a model with
                  memory can extrapolate it and BEAT the 1-step persistence baseline.
    Returns [B,S] orientations in degrees, landing exactly on channels. If
    ``return_latents=True`` it ALSO returns (theta, vel, accel) where ``vel``/``accel`` are
    the signed per-step channel velocity and its per-step difference (see ``_seq_latents``);
    the DEFAULT return value is unchanged (theta only), so existing callers are unaffected."""
    if mode == 'march':
        c0 = torch.randint(0, N, (batch, 1), device=device)
        d  = torch.randint(0, 2, (batch, 1), device=device) * 2 - 1   # -1 or +1
        t  = torch.arange(S, device=device)[None, :]
        ch = (c0 + d * step_channels * t) % N
        theta = ch.float() * STEP_DEG
        if return_latents:
            vel, accel = _seq_latents(ch)
            return theta, vel, accel
        return theta
    if mode == 'markov':
        V  = torch.tensor(vels, device=device)
        nv = len(vels)
        v_idx = torch.zeros(batch, S, dtype=torch.long, device=device)
        v_idx[:, 0] = torch.randint(0, nv, (batch,), device=device)
        for t in range(1, S):                                        # sticky Markov chain on the latent velocity
            stay = torch.rand(batch, device=device) < p_stay
            v_idx[:, t] = torch.where(stay, v_idx[:, t - 1],
                                      torch.randint(0, nv, (batch,), device=device))
        vel    = V[v_idx]                                            # [B,S] velocity governing step t -> t+1
        c0     = torch.randint(0, N, (batch, 1), device=device)
        offset = torch.cat([torch.zeros(batch, 1, dtype=torch.long, device=device),
                            torch.cumsum(vel[:, :-1], dim=1)], dim=1)  # exclusive prefix sum of velocity [B,S]
        ch = (c0 + offset) % N
        theta = ch.float() * STEP_DEG
        if return_latents:
            vel, accel = _seq_latents(ch)
            return theta, vel, accel
        return theta
    if mode == 'momentum':
        accs = torch.tensor([-1, 0, 1], device=device)               # acceleration alphabet (channels/step^2)
        na   = 3
        a_idx = torch.zeros(batch, S, dtype=torch.long, device=device)
        a_idx[:, 0] = torch.randint(0, na, (batch,), device=device)
        for t in range(1, S):                                        # sticky Markov chain on the ACCELERATION
            stay = torch.rand(batch, device=device) < p_stay
            a_idx[:, t] = torch.where(stay, a_idx[:, t - 1],
                                      torch.randint(0, na, (batch,), device=device))
        a = accs[a_idx]                                              # [B,S] acceleration per step
        v = torch.zeros(batch, S, dtype=torch.long, device=device)
        v[:, 0] = torch.randint(-vmax, vmax + 1, (batch,), device=device)
        for t in range(1, S):                                        # velocity INTEGRATES acceleration (smooth)
            v[:, t] = (v[:, t - 1] + a[:, t - 1]).clamp(-vmax, vmax)
        c0     = torch.randint(0, N, (batch, 1), device=device)
        offset = torch.cat([torch.zeros(batch, 1, dtype=torch.long, device=device),
                            torch.cumsum(v[:, :-1], dim=1)], dim=1)   # position INTEGRATES velocity
        ch = (c0 + offset) % N
        theta = ch.float() * STEP_DEG
        if return_latents:
            vel, accel = _seq_latents(ch)
            return theta, vel, accel
        return theta
    raise ValueError(f"unknown mode {mode!r}")


# ---------------- shared sequence forward pass ----------------
def forward_seq(net, theta, fb_scale=1.0, signed_fb=None, ctx=None):
    """Unroll the net over a [B,S] orientation sequence, feeding each step's prediction DOWN to L2/3
    at the next step. Returns preds [B,S,N] (logits; preds[:,t] predicts theta_{t+1}) and r_all [B,S,N].

    `signed_fb` gates ONLY the fed-down copy of the prediction (the CE-logits path `preds` is unchanged):
      None  -> use net.signed_fb (default False);
      False -> pred_down = relu(pred)  (non-negative top-down drive; the original behaviour);
      True  -> pred_down = pred        (SIGNED push-pull: feedback may be negative and pull r BELOW the
                                        feed-forward floor relu(W_ff(l4)), enabling genuine dampening)."""
    use_signed = (getattr(net, 'signed_fb', False) if signed_fb is None else signed_fb)
    B = theta.shape[0]
    h         = torch.zeros(B, net.hidden, device=device)
    pred_down = torch.zeros(B, N, device=device)                     # prediction fed DOWN from previous step
    preds, r_seq = [], []
    for t in range(theta.shape[1]):
        r = net.l23(l4_code(theta[:, t]), fb_scale * pred_down,
                    ctx=(0.0 if ctx is None else ctx))               # L2/3 = bottom-up(theta_t) + predicted top-down (+ctx->VIP)
        r_seq.append(r)
        h = net.gru(r, h)                                            # update temporal context
        pred = net.W_fb(h)                                           # predict the NEXT element  [B,N] logits
        preds.append(pred)
        pred_down = pred if use_signed else F.relu(pred)            # fed down next step (signed if use_signed; else relu)
    return torch.stack(preds, 1), torch.stack(r_seq, 1)


@torch.no_grad()
def quick_acc(net, mode, batch=2048, S=8, fb_scale=1.0, p_stay=0.8):
    """Held-out next-element accuracy (all transitions) on a fresh batch -- a clean, low-noise read."""
    theta = make_sequences(batch, S, mode=mode, p_stay=p_stay)
    preds, _ = forward_seq(net, theta, fb_scale)
    ok = (preds[:, :-1].argmax(-1) == chan(theta)[:, 1:])
    return ok.float().mean().item()


# ---------------- Phase 2: predict the next element, feed the prediction DOWN to L2/3 ----------------
def phase2(net, mode='markov', steps=3000, batch=128, S=8, lam_energy=0.02, lr=1e-3, fb_scale=1.0, p_stay=0.8,
           ce_weight=1.0):
    # Freeze the phase-1 representation (W_ff + decoder). The RNN's prediction is the ONLY top-down
    # signal into L2/3, so it has to do the predicting itself.
    for p in list(net.W_ff.parameters()) + list(net.decoder.parameters()):
        p.requires_grad_(False)
    params = list(net.gru.parameters()) + list(net.W_fb.parameters())
    if getattr(net, 'use_circuit', False):
        params.append(net.circ_raw)                                  # #43 SOM/VIP gains train in phase 2 too
    opt = torch.optim.Adam(params, lr=lr)
    print(f"\n=== PHASE 2 ({mode}, p_stay={p_stay}, S={S}): RNN predicts next element -> fed DOWN into L2/3 ===")
    curve = []
    for s in range(1, steps + 1):
        theta = make_sequences(batch, S, mode=mode, p_stay=p_stay)   # [B,S]
        preds, r_all = forward_seq(net, theta, fb_scale)
        pl  = preds[:, :-1, :].reshape(-1, N)                        # prediction made at step t ...
        tgt = chan(theta[:, 1:]).reshape(-1)                         # ... must match the NEXT element theta_{t+1}
        pred_err = F.cross_entropy(pl, tgt)                          # prediction error
        loss = ce_weight * pred_err + lam_energy * r_all.abs().mean()   # ce_weight = task pressure (default 1.0 == original)
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 1 or s % 200 == 0:
            held = quick_acc(net, mode, S=S, fb_scale=fb_scale, p_stay=p_stay) * 100  # clean held-out accuracy
            print(f"  step {s:5d}  pred_error(CE) {pred_err.item():.3f}  held-out_acc {held:5.1f}%  "
                  f"mean|r| {r_all.abs().mean().item():.3f}  (chance {100/N:.1f}%)")
            curve.append((s, pred_err.item(), held))
    return curve


# ---------------- evaluation: trained RNN vs the OPTIMAL persistence tracker ----------------
@torch.no_grad()
def evaluate(net, mode='markov', batch=4096, S=8, fb_scale=1.0, p_stay=0.8):
    """Compare the trained net's next-element accuracy to the OPTIMAL tracker for this process
    (predict next = current + last-observed velocity). What remains is the SWITCH rate: transitions
    where the hidden velocity just changed are inherently unpredictable from the past."""
    theta = make_sequences(batch, S, mode=mode, p_stay=p_stay)
    preds, _ = forward_seq(net, theta, fb_scale)
    c      = chan(theta)                                             # [B,S] true channels
    rnn_ok = (preds[:, :-1].argmax(-1) == c[:, 1:])                  # [B,S-1] per-transition correct
    # optimal persistence tracker: predict velocity_t = velocity_{t-1} (defined for transitions t>=1)
    dvel       = (c[:, 1:] - c[:, :-1] + N // 2) % N - N // 2        # [B,S-1] signed velocity per transition
    persist_ok = (dvel[:, 1:] == dvel[:, :-1])                       # [B,S-2] did velocity persist?
    switch_rate = 1.0 - persist_ok.float().mean().item()
    print(f"\n=== EVAL ({mode}) on held-out sequences (batch={batch}, S={S}) ===")
    print(f"  RNN next-element acc, ALL transitions            : {rnn_ok.float().mean().item()*100:5.1f}%")
    print(f"  RNN next-element acc, trackable only (t>=1)       : {rnn_ok[:, 1:].float().mean().item()*100:5.1f}%")
    print(f"  OPTIMAL persistence tracker, trackable (t>=1)     : {persist_ok.float().mean().item()*100:5.1f}%")
    print(f"  measured switch rate (irreducible error)          : {switch_rate*100:5.1f}%   (chance {100/N:.1f}%)")


if __name__ == "__main__":
    print(f"device = {device}")
    for mode in ('march', 'markov'):
        print(f"\n################  CASE: {mode}  ################")
        torch.manual_seed(0)                       # identical init + phase-1; ONLY the sequence stats differ
        net = SimpleNet().to(device)
        phase1(net)
        phase2(net, mode=mode)
        evaluate(net, mode=mode)
    print("\nDone.")
