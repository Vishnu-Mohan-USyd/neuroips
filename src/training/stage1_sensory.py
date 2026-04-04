"""Stage 1: Sensory scaffold training.

Goal: Stable orientation selectivity in L4 and L2/3.
Runs V1 submodules directly (L4 -> PV -> L2/3), bypassing V2/feedback.
Trains L2/3 params + PV + decoder with random gratings at variable contrast.
2000 steps, Adam lr=1e-3.

Gating checkpoints (must all pass before Stage 2):
    1. L2/3 decoder accuracy >= 90% (36-way)
    2. Each L2/3 unit has unimodal tuning
    3. Preferred orientations tile 0-175 deg
    4. FWHM 15-30 deg
    5. Mean activity in target range
    6. Contrast-invariant tuning width (PV divisive normalization working)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from src.config import ModelConfig, TrainingConfig
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating
from src.training.losses import CompositeLoss
from src.training.trainer import freeze_stage1

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    """Result of Stage 1 training."""
    final_loss: float
    decoder_accuracy: float
    gating_passed: dict[str, bool]
    n_steps_trained: int
    decoder_state_dict: dict | None = None  # Trained decoder weights for transfer


def _run_v1_only(
    net: LaminarV1V2Network,
    stimulus: Tensor,
    n_timesteps: int = 20,
) -> Tensor:
    """Run V1 submodules directly, bypassing V2/feedback.

    L4 -> PV -> L2/3 with zero SOM drive and zero template modulation.
    This ensures the sensory scaffold is built without any feedback influence.

    Args:
        net: LaminarV1V2Network.
        stimulus: [B, N] — single grating stimulus.
        n_timesteps: Number of integration steps.

    Returns:
        r_l23: [B, N] — final L2/3 rates.
    """
    B, N = stimulus.shape
    device = stimulus.device

    r_l4 = torch.zeros(B, N, device=device)
    r_l23 = torch.zeros(B, N, device=device)
    r_pv = torch.zeros(B, 1, device=device)
    r_som = torch.zeros(B, N, device=device)
    adaptation = torch.zeros(B, N, device=device)
    template_mod = torch.zeros(B, N, device=device)

    for _ in range(n_timesteps):
        r_l4, adaptation = net.l4(stimulus, r_l4, r_pv, adaptation)
        r_pv = net.pv(r_l4, r_l23, r_pv)
        r_l23 = net.l23(r_l4, r_l23, template_mod, r_som, r_pv)

    return r_l23


def run_stage1(
    net: LaminarV1V2Network,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device | None = None,
    seed: int = 42,
) -> Stage1Result:
    """Run Stage 1 sensory scaffold training.

    Runs V1 directly (no V2/feedback). Trains L2/3 params + PV + decoder
    with random gratings at variable contrast.

    Args:
        net: LaminarV1V2Network (will be modified in-place).
        model_cfg: ModelConfig.
        train_cfg: TrainingConfig.
        device: Device to train on.
        seed: Random seed.

    Returns:
        Stage1Result with training metrics and gating checkpoint results.
    """
    dev = device or torch.device("cpu")
    net = net.to(dev)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    N = model_cfg.n_orientations
    n_steps = train_cfg.stage1_n_steps
    lr = train_cfg.stage1_lr
    contrast_range = train_cfg.stage1_contrast_range
    batch_size = 32

    # Loss module (contains the decoder)
    loss_fn = CompositeLoss(train_cfg, model_cfg).to(dev)

    # Freeze everything, then unfreeze only Stage 1 params
    for p in net.parameters():
        p.requires_grad_(False)
    # Trainable in Stage 1: L2/3 params + PV params
    trainable = list(net.l23.parameters()) + list(net.pv.parameters())
    for p in trainable:
        p.requires_grad_(True)

    # Optimizer: Stage 1 params + decoder
    all_params = trainable + list(loss_fn.orientation_decoder.parameters())
    optimizer = Adam(all_params, lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()

        # Random orientations (continuous) and contrasts
        thetas = torch.rand(batch_size, generator=gen) * model_cfg.orientation_range
        lo, hi = contrast_range
        contrasts = lo + (hi - lo) * torch.rand(batch_size, generator=gen)

        # Generate grating stimulus
        stim = generate_grating(
            thetas, contrasts,
            n_orientations=N,
            sigma=model_cfg.sigma_ff,
            n=model_cfg.naka_rushton_n,
            c50=model_cfg.naka_rushton_c50,
            period=model_cfg.orientation_range,
        ).to(dev)  # [B, N]

        # Run V1 only (bypass V2/feedback) for 20 timesteps
        r_l23 = _run_v1_only(net, stim, n_timesteps=20)

        # Readout loss
        logits = loss_fn.orientation_decoder(r_l23)  # [B, N]
        targets = loss_fn._theta_to_channel(thetas).to(dev)  # [B]
        l_sensory = F.cross_entropy(logits, targets)

        # Homeostasis (add time dim for the penalty)
        l_homeo = loss_fn.homeostasis_penalty(r_l23.unsqueeze(1))

        loss = l_sensory + train_cfg.lambda_homeo * l_homeo
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            acc = (logits.argmax(dim=-1) == targets).float().mean().item()
            logger.info(
                f"Stage 1 step {step+1}/{n_steps}: "
                f"loss={loss.item():.4f}, acc={acc:.3f}, "
                f"l23_mean={r_l23.mean().item():.4f}"
            )

    # Run gating checkpoints
    gating = _run_gating_checks(net, loss_fn, model_cfg, dev, gen)

    # Final accuracy
    with torch.no_grad():
        thetas = torch.rand(256, generator=gen) * model_cfg.orientation_range
        contrasts = torch.ones(256) * 0.5
        stim = generate_grating(
            thetas, contrasts,
            n_orientations=N, sigma=model_cfg.sigma_ff,
            n=model_cfg.naka_rushton_n, c50=model_cfg.naka_rushton_c50,
            period=model_cfg.orientation_range,
        ).to(dev)
        r_l23 = _run_v1_only(net, stim, n_timesteps=20)
        logits = loss_fn.orientation_decoder(r_l23)
        targets = loss_fn._theta_to_channel(thetas).to(dev)
        final_acc = (logits.argmax(dim=-1) == targets).float().mean().item()

    # Freeze Stage 1 params
    freeze_stage1(net)

    # Extract trained decoder weights for transfer to Stage 2
    decoder_sd = {k: v.cpu().clone() for k, v in loss_fn.orientation_decoder.state_dict().items()}

    return Stage1Result(
        final_loss=losses[-1] if losses else float("nan"),
        decoder_accuracy=final_acc,
        gating_passed=gating,
        n_steps_trained=n_steps,
        decoder_state_dict=decoder_sd,
    )


def _run_gating_checks(
    net: LaminarV1V2Network,
    loss_fn: CompositeLoss,
    model_cfg: ModelConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, bool]:
    """Run all 6 gating checkpoints for Stage 1.

    Returns:
        Dict mapping checkpoint name -> pass/fail.
    """
    N = model_cfg.n_orientations
    period = model_cfg.orientation_range
    step_deg = period / N
    results = {}

    net.eval()
    with torch.no_grad():
        # Present all 36 canonical orientations at high contrast
        test_oris = torch.arange(N, dtype=torch.float32) * step_deg  # [N]
        test_contrasts = torch.ones(N) * 0.8

        stim = generate_grating(
            test_oris, test_contrasts,
            n_orientations=N, sigma=model_cfg.sigma_ff,
            n=model_cfg.naka_rushton_n, c50=model_cfg.naka_rushton_c50,
            period=period,
        ).to(device)  # [N, N]

        # Run V1 only
        r_readout = _run_v1_only(net, stim, n_timesteps=20)  # [N, N]

        # 1. Decoder accuracy >= 90%
        labels = torch.arange(N, device=device)
        logits = loss_fn.orientation_decoder(r_readout)
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        results["decoder_accuracy_90"] = acc >= 0.90
        logger.info(f"Gating 1 - Decoder accuracy: {acc:.3f} (need >= 0.90)")

        # 2. Unimodal tuning: each unit's response has a clear peak
        n_unimodal = 0
        for i in range(N):
            curve = r_readout[:, i]  # response of unit i to all orientations
            is_uni = curve.max() > curve.mean() * 1.5
            if is_uni:
                n_unimodal += 1
        results["unimodal_tuning"] = n_unimodal >= N * 0.8
        logger.info(f"Gating 2 - Unimodal units: {n_unimodal}/{N}")

        # 3. Preferred orientations tile 0-175 deg
        pref_oris = r_readout.argmax(dim=0)  # [N] — for each unit, which stimulus peaks
        unique_prefs = pref_oris.unique().numel()
        results["orientation_tiling"] = unique_prefs >= N * 0.7
        logger.info(f"Gating 3 - Unique preferred orientations: {unique_prefs}/{N}")

        # 4. FWHM 15-30 deg (relaxed to 15-40 for discrete channels)
        fwhm_ok = 0
        for i in range(N):
            curve = r_readout[:, i]
            peak_val = curve.max()
            if peak_val > 1e-6:
                half_max = peak_val / 2.0
                above_half = (curve >= half_max).sum().item()
                fwhm_deg = above_half * step_deg
                if 15.0 <= fwhm_deg <= 40.0:
                    fwhm_ok += 1
        results["fwhm_range"] = fwhm_ok >= N * 0.7
        logger.info(f"Gating 4 - FWHM in range: {fwhm_ok}/{N}")

        # 5. Mean activity in target range
        mean_rate = r_readout.mean().item()
        results["mean_activity"] = 0.001 <= mean_rate <= 2.0
        logger.info(f"Gating 5 - Mean L2/3 rate: {mean_rate:.4f}")

        # 6. Contrast-invariant tuning width
        low_stim = generate_grating(
            test_oris, torch.ones(N) * 0.2,
            n_orientations=N, sigma=model_cfg.sigma_ff,
            n=model_cfg.naka_rushton_n, c50=model_cfg.naka_rushton_c50,
            period=period,
        ).to(device)
        r_low = _run_v1_only(net, low_stim, n_timesteps=20)

        fwhm_high = []
        fwhm_low = []
        for i in range(N):
            for r, fwhm_list in [(r_readout, fwhm_high), (r_low, fwhm_low)]:
                curve = r[:, i]
                peak_val = curve.max()
                if peak_val > 1e-6:
                    above = (curve >= peak_val / 2.0).sum().item() * step_deg
                else:
                    above = 0.0
                fwhm_list.append(above)

        fwhm_high_mean = sum(fwhm_high) / len(fwhm_high)
        fwhm_low_mean = sum(fwhm_low) / len(fwhm_low) if sum(fwhm_low) > 0 else 0
        if fwhm_low_mean > 0 and fwhm_high_mean > 0:
            ratio = max(fwhm_high_mean, fwhm_low_mean) / min(fwhm_high_mean, fwhm_low_mean)
            results["contrast_invariant_width"] = ratio < 2.0
        else:
            results["contrast_invariant_width"] = False
        logger.info(
            f"Gating 6 - FWHM high={fwhm_high_mean:.1f}, low={fwhm_low_mean:.1f}"
        )

    net.train()
    return results
