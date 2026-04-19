"""Phase-3 Kok task-learning driver (plan v4 / Task #40).

Freezes the sensory core and every *generic* circuit weight; only the
two task-specific context-memory weights ``W_qm_task`` and ``W_mh_task``
adapt, via :class:`ThreeFactorRule`.

Kok 2012 trial timing (``dt=5 ms``):
  * 200 ms cue    → 40 steps    (cue token ``q_t`` active, blank frame)
  * 550 ms delay  → 110 steps   (blank frame, zero cue)
  * 500 ms probe1 → 100 steps   (oriented grating, zero cue)
  * 100 ms blank  → 20 steps    (blank frame)
  * 500 ms probe2 → 100 steps   (oriented grating, zero cue)

Cue-mapping counterbalance: seed parity decides which of the two cues
points at which orientation (odd → cue0=45°, cue1=135°; even → reversed),
so the mapping is not confounded with rate-space asymmetries in the
post-Phase-2 network.

Learning sub-phase: deterministic (cue predicts probe orientation 100 %).
Scan sub-phase:     probabilistic (75 % valid, 25 % mismatched).

Plasticity (closed-form, no ``backward``):
  * ``W_qm_task`` via :meth:`ThreeFactorRule.delta_qm` with
    ``cue = q_t``, ``memory = m_end_cue``,
    ``memory_error = m_end_cue − m_pre_cue`` (cue-evoked change).
  * ``W_mh_task`` via :meth:`ThreeFactorRule.delta_mh` with
    ``memory = m_start_probe``,
    ``probe_error = r_l23_probe1_mean − b_l23_pre_probe``.

Frozen-sensory-core SHA is asserted identical before/after training; every
non-plastic Parameter under the current phase manifest is also checksummed
to catch accidental writes outside the two task weights.

Usage:
    python -m scripts.v2.train_phase3_kok_learning \\
        --phase2-checkpoint checkpoints/v2/phase2/phase2_s42/step_1000.pt \\
        --seed 42 --n-trials-learning 5000 --n-trials-scan 10000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import make_blank_frame, make_grating_frame
from scripts.v2.train_phase2_predictive import PhaseFrozenError
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.plasticity import ThreeFactorRule
from src.v2_model.stimuli.feature_tokens import TokenBank


__all__ = [
    "CUE_ORIENTATIONS_DEG", "KokTiming", "cue_mapping_from_seed",
    "build_cue_tensor", "run_kok_trial", "run_phase3_kok_training",
]


# ---------------------------------------------------------------------------
# Paradigm constants
# ---------------------------------------------------------------------------

CUE_ORIENTATIONS_DEG: tuple[float, float] = (45.0, 135.0)


@dataclass
class KokTiming:
    """Per-trial step budget (integer ``dt=5 ms`` steps)."""
    cue_steps: int = 40          # 200 ms
    delay_steps: int = 110       # 550 ms
    probe1_steps: int = 100      # 500 ms
    blank_steps: int = 20        # 100 ms
    probe2_steps: int = 100      # 500 ms

    @property
    def total(self) -> int:
        return (self.cue_steps + self.delay_steps + self.probe1_steps
                + self.blank_steps + self.probe2_steps)


def cue_mapping_from_seed(seed: int) -> dict[int, float]:
    """Cue-id → orientation (deg) map. Counterbalanced by seed parity.

    Odd seed  ⇒ {0: 45°, 1: 135°}.
    Even seed ⇒ {0: 135°, 1: 45°}.
    """
    if int(seed) % 2 == 1:
        return {0: CUE_ORIENTATIONS_DEG[0], 1: CUE_ORIENTATIONS_DEG[1]}
    return {0: CUE_ORIENTATIONS_DEG[1], 1: CUE_ORIENTATIONS_DEG[0]}


def build_cue_tensor(cue_id: int, n_cue: int, device: str = "cpu") -> Tensor:
    """One-hot cue tensor ``[1, n_cue]`` with a single 1.0 at position ``cue_id``.

    Only the first two slots (0, 1) are used by the 2-class Kok paradigm;
    the remaining ``n_cue - 2`` dims stay exactly zero so W_qm_task does
    not accrue spurious outer-product entries outside the used subspace.
    """
    if cue_id not in (0, 1):
        raise ValueError(f"cue_id must be 0 or 1; got {cue_id}")
    q = torch.zeros(1, int(n_cue), device=device, dtype=torch.float32)
    q[0, int(cue_id)] = 1.0
    return q


# ---------------------------------------------------------------------------
# Frozen-weight snapshot (for the end-of-training integrity check)
# ---------------------------------------------------------------------------


def _frozen_params(net: V2Network) -> list[tuple[str, str]]:
    """Every named Parameter *not* in the current phase's plastic manifest.

    LGN/L4 is skipped — it has no Parameters (frozen-by-construction) and
    is covered by :meth:`V2Network.frozen_sensory_core_sha`.
    """
    plastic = set(net.plastic_weight_names())
    out: list[tuple[str, str]] = []
    for mod_name, child in net.named_children():
        if mod_name == "lgn_l4":
            continue
        for p_name, _p in child.named_parameters(recurse=False):
            if (mod_name, p_name) in plastic:
                continue
            out.append((mod_name, p_name))
    return out


def _snapshot_weights(
    net: V2Network, keys: list[tuple[str, str]],
) -> dict[tuple[str, str], Tensor]:
    snaps: dict[tuple[str, str], Tensor] = {}
    for mod, wname in keys:
        snaps[(mod, wname)] = (
            getattr(getattr(net, mod), wname).detach().clone()
        )
    return snaps


def _assert_snapshot_unchanged(
    net: V2Network, snaps: dict[tuple[str, str], Tensor],
) -> None:
    for (mod, wname), w_before in snaps.items():
        w_after = getattr(getattr(net, mod), wname).data
        if not torch.equal(w_before, w_after):
            raise RuntimeError(
                f"frozen weight {mod}.{wname} was mutated during Phase-3 "
                f"Kok training (phase={net.phase!r})"
            )


def _assert_plastic(net: V2Network, module: str, wname: str) -> None:
    if (module, wname) not in net.plastic_weight_names():
        raise PhaseFrozenError(
            f"refusing to update {module}.{wname}: not plastic under "
            f"phase={net.phase!r}"
        )


# ---------------------------------------------------------------------------
# Trial execution + plasticity
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_kok_trial(
    net: V2Network, cfg: ModelConfig,
    *, cue_id: int, probe_orientation_deg: float,
    timing: KokTiming, rule: ThreeFactorRule,
    noise_std: float = 0.0, device: str = "cpu",
) -> dict[str, Tensor]:
    """Run one Kok trial through the network; apply plasticity at end.

    Returns a dict with the four signals consumed by the update so the
    caller can log them; the update has already been applied in-place.
    """
    _assert_plastic(net, "context_memory", "W_qm_task")
    _assert_plastic(net, "context_memory", "W_mh_task")

    blank = make_blank_frame(1, cfg, device=device)
    probe = make_grating_frame(
        float(probe_orientation_deg), 1.0, cfg, device=device,
    )
    q_cue = build_cue_tensor(int(cue_id), cfg.arch.n_c, device=device)

    state = net.initial_state(batch_size=1)
    m_pre_cue = state.m.clone()                                 # [1, n_m]
    m_end_cue: Optional[Tensor] = None
    m_start_probe: Optional[Tensor] = None
    b_l23_pre_probe: Optional[Tensor] = None
    probe1_l23: list[Tensor] = []

    n_total = timing.total
    cue_end = timing.cue_steps
    delay_end = cue_end + timing.delay_steps
    probe1_end = delay_end + timing.probe1_steps
    blank2_end = probe1_end + timing.blank_steps

    for t in range(n_total):
        if t < cue_end:
            frame, q_t = blank, q_cue
        elif t < delay_end:
            frame, q_t = blank, None
        elif t < probe1_end:
            frame, q_t = probe, None
        elif t < blank2_end:
            frame, q_t = blank, None
        else:
            frame, q_t = probe, None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn_like(frame)

        _x_hat, state, info = net(frame, state, q_t=q_t)

        # Task #43 — θ homeostasis must track activity during Phase-3
        # assays (matches Phase-2 driver). Homeostasis mutates buffers,
        # NOT nn.Parameters — the frozen-sensory-core-SHA invariant
        # (Parameter-only) is preserved; θ is a running setpoint, not a
        # learned weight.
        with torch.no_grad():
            net.l23_e.homeostasis.update(state.r_l23)
            net.h_e.homeostasis.update(state.r_h)

        if t == cue_end - 1:
            m_end_cue = state.m.clone()
        if t == delay_end - 1:
            m_start_probe = state.m.clone()
        if t == delay_end:
            b_l23_pre_probe = info["b_l23"].clone()
        if delay_end <= t < probe1_end:
            probe1_l23.append(info["r_l23"].clone())

    assert m_end_cue is not None and m_start_probe is not None
    assert b_l23_pre_probe is not None and probe1_l23

    memory_error_qm = m_end_cue - m_pre_cue                      # [1, n_m]
    dw_qm = rule.delta_qm(
        cue=q_cue, memory=m_end_cue, memory_error=memory_error_qm,
        weights=net.context_memory.W_qm_task,
    )
    net.context_memory.W_qm_task.data.add_(dw_qm)
    # Runaway safeguard: cap |W_qm_task| at 1.0 per element to prevent
    # the cue → memory → cue positive-feedback loop from diverging during
    # early Phase-3 training (Task #58 / debugger Task #49 Claim 4).
    net.context_memory.W_qm_task.data.clamp_(min=-1.0, max=1.0)

    r_l23_probe1_mean = torch.stack(probe1_l23, dim=0).mean(dim=0)  # [1, n_l23]
    probe_error_mh = r_l23_probe1_mean - b_l23_pre_probe         # [1, n_l23]
    dw_mh = rule.delta_mh(
        memory=m_start_probe, probe_error=probe_error_mh,
        weights=net.context_memory.W_mh_task,
    )
    net.context_memory.W_mh_task.data.add_(dw_mh)

    return {
        "dw_qm_abs_mean": dw_qm.abs().mean().detach(),
        "dw_mh_abs_mean": dw_mh.abs().mean().detach(),
        "probe1_r_l23_mean": r_l23_probe1_mean.mean().detach(),
        "m_end_cue_mean": m_end_cue.mean().detach(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainStepMetrics:
    phase_name: str
    trial: int
    cue_id: int
    probe_orientation_deg: float
    dw_qm_abs_mean: float
    dw_mh_abs_mean: float
    probe1_r_l23_mean: float
    wall_time_s: float


@dataclass
class KokHistory:
    steps: list[TrainStepMetrics] = field(default_factory=list)


def _pick_scan_probe(
    cue_id: int, cue_mapping: dict[int, float], validity: float,
    np_rng: np.random.Generator,
) -> float:
    """Sample a probe orientation: ``validity`` probability of matching cue."""
    if float(validity) >= 1.0 or np_rng.random() < float(validity):
        return float(cue_mapping[cue_id])
    other = 1 - int(cue_id)
    return float(cue_mapping[other])


def run_phase3_kok_training(
    net: V2Network,
    *, n_trials_learning: int,
    n_trials_scan: int,
    validity_scan: float = 0.75,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    timing: Optional[KokTiming] = None,
    noise_std: float = 0.0,
    cue_mapping: Optional[dict[int, float]] = None,
    metrics_path: Optional[Path] = None,
    log_every: int = 50,
) -> KokHistory:
    """Run the two-sub-phase Kok trainer in-place on ``net``.

    * Learning sub-phase — ``n_trials_learning`` trials, 100 % valid.
    * Scan sub-phase     — ``n_trials_scan`` trials, ``validity_scan`` valid.

    The caller owns ``net``; this function mutates ``W_qm_task`` and
    ``W_mh_task`` in place. No other Parameter is touched — an integrity
    check at exit asserts this.
    """
    if net.phase != "phase3_kok":
        raise PhaseFrozenError(
            f"run_phase3_kok_training requires phase='phase3_kok'; "
            f"got {net.phase!r}"
        )
    if n_trials_learning < 0 or n_trials_scan < 0:
        raise ValueError("trial counts must be ≥ 0")
    if not (0.0 <= validity_scan <= 1.0):
        raise ValueError(f"validity_scan must be in [0, 1]; got {validity_scan}")

    cfg = net.cfg
    timing = timing or KokTiming()
    cue_mapping = cue_mapping or cue_mapping_from_seed(seed)
    rule = ThreeFactorRule(lr=float(lr), weight_decay=float(weight_decay))

    sha_at_start = net.frozen_sensory_core_sha()
    frozen_keys = _frozen_params(net)
    frozen_snaps = _snapshot_weights(net, frozen_keys)

    np_rng = np.random.default_rng(int(seed))
    history = KokHistory()
    metrics_fh = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = metrics_path.open("w")

    t_start = time.monotonic()
    try:
        for sub_phase, n_trials, validity in (
            ("learning", int(n_trials_learning), 1.0),
            ("scan",     int(n_trials_scan),     float(validity_scan)),
        ):
            for k in range(n_trials):
                cue_id = int(np_rng.integers(0, 2))
                probe_deg = _pick_scan_probe(cue_id, cue_mapping, validity, np_rng)
                info = run_kok_trial(
                    net, cfg,
                    cue_id=cue_id, probe_orientation_deg=probe_deg,
                    timing=timing, rule=rule,
                    noise_std=float(noise_std), device=str(cfg.device),
                )
                if k % max(int(log_every), 1) == 0 or k == n_trials - 1:
                    m = TrainStepMetrics(
                        phase_name=sub_phase, trial=k, cue_id=cue_id,
                        probe_orientation_deg=float(probe_deg),
                        dw_qm_abs_mean=float(info["dw_qm_abs_mean"].item()),
                        dw_mh_abs_mean=float(info["dw_mh_abs_mean"].item()),
                        probe1_r_l23_mean=float(info["probe1_r_l23_mean"].item()),
                        wall_time_s=time.monotonic() - t_start,
                    )
                    history.steps.append(m)
                    if metrics_fh is not None:
                        metrics_fh.write(json.dumps(m.__dict__) + "\n")
                        metrics_fh.flush()
    finally:
        if metrics_fh is not None:
            metrics_fh.close()

    sha_at_end = net.frozen_sensory_core_sha()
    if sha_at_end != sha_at_start:
        raise RuntimeError(
            "frozen sensory core SHA changed during Phase-3 Kok training — "
            "LGN/L4 mutated (forbidden)"
        )
    _assert_snapshot_unchanged(net, frozen_snaps)
    return history


def _save_checkpoint(
    net: V2Network, trial: int, out_path: Path, cue_mapping: dict[int, float],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(trial),
            "state_dict": net.state_dict(),
            "phase": net.phase,
            "frozen_sha": net.frozen_sensory_core_sha(),
            "cue_mapping": {int(k): float(v) for k, v in cue_mapping.items()},
        },
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-3 Kok task-learning driver.")
    p.add_argument("--phase2-checkpoint", type=Path, default=None,
                   help="Load a Phase-2 checkpoint as the starting net (optional).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-learning", type=int, default=5000)
    p.add_argument("--n-trials-scan", type=int, default=10_000)
    p.add_argument("--validity-scan", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("checkpoints/v2/phase3_kok"),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli().parse_args(argv)
    torch.manual_seed(int(args.seed))

    cfg = ModelConfig(seed=int(args.seed), device=args.device)
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=int(args.seed), device=args.device)
    if args.phase2_checkpoint is not None:
        ckpt = torch.load(args.phase2_checkpoint, map_location=args.device,
                          weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
    net.set_phase("phase3_kok")

    cue_mapping = cue_mapping_from_seed(int(args.seed))
    out_path = args.out_dir / f"phase3_kok_s{int(args.seed)}.pt"
    metrics_path = out_path.parent / f"phase3_kok_s{int(args.seed)}_metrics.jsonl"
    run_phase3_kok_training(
        net=net,
        n_trials_learning=int(args.n_trials_learning),
        n_trials_scan=int(args.n_trials_scan),
        validity_scan=float(args.validity_scan),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        noise_std=float(args.noise_std),
        cue_mapping=cue_mapping,
        metrics_path=metrics_path,
        log_every=int(args.log_every),
    )
    _save_checkpoint(
        net, args.n_trials_learning + args.n_trials_scan,
        out_path, cue_mapping,
    )
    print(f"phase3_kok checkpoint written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
