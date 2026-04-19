"""Phase-3 Richter task-learning driver (plan v4 / Task #40).

Freezes the sensory core and every generic circuit weight; only the
two task-specific context-memory weights ``W_lm_task`` and ``W_mh_task``
adapt, via :class:`ThreeFactorRule`.

Richter 2019 trial timing (``dt=5 ms``):
  * 500 ms leader  → 100 steps  (leading identity token; ``leader_t`` active)
  * 0 ms ISI       → 0 steps    (immediate transition to trailer)
  * 500 ms trailer → 100 steps  (trailing identity token; zero leader)
  * 5 s ITI        → 0 steps    (represented by an inter-trial state reset;
                                 not stepped because the 500 ms τ_m memory
                                 has decayed to ``exp(-10) ≈ 4.5e-5`` by then)

Leading × trailing matrix: six of the twelve :class:`TokenBank` tokens
act as leaders (indices 0-5); the remaining six as trailers (indices
6-11). The 6×6 identity pairs define all trial conditions.

Learning sub-phase (``100 %``): leader ``i`` deterministically precedes
trailer ``σ(i)`` for a fixed permutation ``σ`` taken from the seed.
Scan sub-phase (``50 %``): leader predicts trailer ``σ(i)`` on half the
trials, an arbitrary trailer on the other half.

Plasticity (closed-form, no ``backward``):
  * ``W_lm_task`` via :meth:`ThreeFactorRule.delta_qm` (same math; leader
    takes the role of the Kok cue):
    ``cue = leader_t``, ``memory = m_end_leader``,
    ``memory_error = m_end_leader − m_pre_trial``.
  * ``W_mh_task`` via :meth:`ThreeFactorRule.delta_mh`:
    ``memory = m_start_trailer``,
    ``probe_error = r_l23_trailer_mean − b_l23_pre_trailer``.

Frozen-sensory-core SHA is asserted identical before/after training;
every non-plastic Parameter in the current phase's manifest is also
checksummed for accidental writes.

Usage:
    python -m scripts.v2.train_phase3_richter_learning \\
        --phase2-checkpoint checkpoints/v2/phase2/phase2_s42/step_1000.pt \\
        --seed 42 --n-trials-learning 20000 --n-trials-scan 10000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from scripts.v2.train_phase2_predictive import PhaseFrozenError
from src.v2_model.config import ModelConfig
from src.v2_model.network import V2Network
from src.v2_model.plasticity import ThreeFactorRule
from src.v2_model.stimuli.feature_tokens import TokenBank


__all__ = [
    "RichterTiming", "LEADER_TOKEN_IDX", "TRAILER_TOKEN_IDX",
    "permutation_from_seed", "build_leader_tensor", "run_richter_trial",
    "run_phase3_richter_training",
]


# ---------------------------------------------------------------------------
# Paradigm constants
# ---------------------------------------------------------------------------

LEADER_TOKEN_IDX: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
TRAILER_TOKEN_IDX: tuple[int, ...] = (6, 7, 8, 9, 10, 11)
N_LEAD_TRAIL: int = 6


@dataclass
class RichterTiming:
    """Per-trial step budget (integer ``dt=5 ms`` steps)."""
    leader_steps: int = 100      # 500 ms
    trailer_steps: int = 100     # 500 ms
    # ITI is represented by an explicit state reset between trials (see
    # module docstring). ``iti_steps`` is kept for completeness but the
    # driver does *not* simulate it — it sets ``iti_steps = 0`` by default
    # to keep per-trial wall-time bounded.
    iti_steps: int = 0

    @property
    def total(self) -> int:
        return self.leader_steps + self.trailer_steps + self.iti_steps


def permutation_from_seed(seed: int) -> tuple[int, ...]:
    """Return a stable 6-permutation ``σ`` mapping leader-pos i → trailer-pos σ(i).

    Uses :class:`numpy.random.default_rng` for reproducibility — the same
    ``seed`` always yields the same permutation, so the learned 6×6
    leading/trailing association is seed-deterministic.
    """
    rng = np.random.default_rng(int(seed))
    return tuple(int(x) for x in rng.permutation(N_LEAD_TRAIL))


def build_leader_tensor(
    leader_pos: int, n_leader: int, device: str = "cpu",
) -> Tensor:
    """One-hot leader vector ``[1, n_leader]`` with a 1.0 at ``leader_pos``.

    ``leader_pos ∈ {0, …, 5}`` indexes into :data:`LEADER_TOKEN_IDX`.
    Remaining slots (``leader_pos + 1 … n_leader − 1``) stay exactly zero.
    """
    if not (0 <= int(leader_pos) < N_LEAD_TRAIL):
        raise ValueError(
            f"leader_pos must be in [0, {N_LEAD_TRAIL}); got {leader_pos}"
        )
    l = torch.zeros(1, int(n_leader), device=device, dtype=torch.float32)
    l[0, int(leader_pos)] = 1.0
    return l


# ---------------------------------------------------------------------------
# Frozen-weight snapshot helpers
# ---------------------------------------------------------------------------


def _frozen_params(net: V2Network) -> list[tuple[str, str]]:
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
    return {
        (m, w): getattr(getattr(net, m), w).detach().clone()
        for (m, w) in keys
    }


def _assert_snapshot_unchanged(
    net: V2Network, snaps: dict[tuple[str, str], Tensor],
) -> None:
    for (mod, wname), w_before in snaps.items():
        w_after = getattr(getattr(net, mod), wname).data
        if not torch.equal(w_before, w_after):
            raise RuntimeError(
                f"frozen weight {mod}.{wname} was mutated during Phase-3 "
                f"Richter training (phase={net.phase!r})"
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
def run_richter_trial(
    net: V2Network, bank: TokenBank,
    *, leader_pos: int, trailer_pos: int,
    timing: RichterTiming, rule: ThreeFactorRule,
    noise_std: float = 0.0, device: str = "cpu",
) -> dict[str, Tensor]:
    """Run one Richter trial through the network; apply plasticity at end.

    Returns a dict with the four signals consumed by the update so the
    caller can log them; the update has already been applied in-place.
    """
    _assert_plastic(net, "context_memory", "W_lm_task")
    _assert_plastic(net, "context_memory", "W_mh_task")

    if not (0 <= leader_pos < N_LEAD_TRAIL):
        raise ValueError(f"leader_pos out of range: {leader_pos}")
    if not (0 <= trailer_pos < N_LEAD_TRAIL):
        raise ValueError(f"trailer_pos out of range: {trailer_pos}")

    leader_tok = bank.tokens[
        LEADER_TOKEN_IDX[int(leader_pos)]:LEADER_TOKEN_IDX[int(leader_pos)] + 1
    ].to(device=device)                                             # [1,1,H,W]
    trailer_tok = bank.tokens[
        TRAILER_TOKEN_IDX[int(trailer_pos)]:TRAILER_TOKEN_IDX[int(trailer_pos)] + 1
    ].to(device=device)                                             # [1,1,H,W]
    leader_v = build_leader_tensor(
        int(leader_pos), net.context_memory.n_leader, device=device,
    )

    state = net.initial_state(batch_size=1)
    m_pre_trial = state.m.clone()                                   # [1, n_m]
    m_end_leader: Optional[Tensor] = None
    m_start_trailer: Optional[Tensor] = None
    b_l23_pre_trailer: Optional[Tensor] = None
    trailer_l23: list[Tensor] = []

    leader_end = timing.leader_steps
    trailer_end = leader_end + timing.trailer_steps

    for t in range(timing.total):
        if t < leader_end:
            frame = leader_tok
            ld_t = leader_v
        elif t < trailer_end:
            frame = trailer_tok
            ld_t = None
        else:
            frame = torch.full_like(trailer_tok, 0.5)               # ITI grey
            ld_t = None
        if noise_std > 0.0:
            frame = frame + noise_std * torch.randn_like(frame)

        _x_hat, state, info = net(frame, state, leader_t=ld_t)

        # Task #43 — θ homeostasis must track activity during Phase-3
        # assays (matches Phase-2 driver). Homeostasis mutates buffers,
        # NOT nn.Parameters — the frozen-sensory-core-SHA invariant
        # (Parameter-only) is preserved; θ is a running setpoint, not a
        # learned weight.
        with torch.no_grad():
            net.l23_e.homeostasis.update(state.r_l23)
            net.h_e.homeostasis.update(state.r_h)

        if t == leader_end - 1:
            m_end_leader = state.m.clone()
            m_start_trailer = state.m.clone()
        if t == leader_end:
            b_l23_pre_trailer = info["b_l23"].clone()
        if leader_end <= t < trailer_end:
            trailer_l23.append(info["r_l23"].clone())

    assert m_end_leader is not None and m_start_trailer is not None
    assert b_l23_pre_trailer is not None and trailer_l23

    memory_error_lm = m_end_leader - m_pre_trial                    # [1, n_m]
    dw_lm = rule.delta_qm(
        cue=leader_v, memory=m_end_leader, memory_error=memory_error_lm,
        weights=net.context_memory.W_lm_task,
    )
    net.context_memory.W_lm_task.data.add_(dw_lm)
    # Runaway safeguard: cap |W_lm_task| at 1.0 per element — mirrors the
    # Kok-side cap (Task #58 / debugger Task #49 Claim 4).
    net.context_memory.W_lm_task.data.clamp_(min=-1.0, max=1.0)

    r_l23_trailer_mean = torch.stack(trailer_l23, dim=0).mean(dim=0)  # [1, n_l23]
    probe_error_mh = r_l23_trailer_mean - b_l23_pre_trailer         # [1, n_l23]
    dw_mh = rule.delta_mh(
        memory=m_start_trailer, probe_error=probe_error_mh,
        weights=net.context_memory.W_mh_task,
    )
    net.context_memory.W_mh_task.data.add_(dw_mh)

    return {
        "dw_lm_abs_mean": dw_lm.abs().mean().detach(),
        "dw_mh_abs_mean": dw_mh.abs().mean().detach(),
        "trailer_r_l23_mean": r_l23_trailer_mean.mean().detach(),
        "m_end_leader_mean": m_end_leader.mean().detach(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainStepMetrics:
    phase_name: str
    trial: int
    leader_pos: int
    trailer_pos: int
    dw_lm_abs_mean: float
    dw_mh_abs_mean: float
    trailer_r_l23_mean: float
    wall_time_s: float


@dataclass
class RichterHistory:
    steps: list[TrainStepMetrics] = field(default_factory=list)


def _pick_scan_trailer(
    leader_pos: int, permutation: tuple[int, ...], reliability: float,
    np_rng: np.random.Generator,
) -> int:
    """Trailer position: matches ``σ(leader)`` with prob ``reliability``."""
    if float(reliability) >= 1.0 or np_rng.random() < float(reliability):
        return int(permutation[int(leader_pos)])
    # Pick any other trailer position uniformly from the remaining 5.
    canon = int(permutation[int(leader_pos)])
    options = [i for i in range(N_LEAD_TRAIL) if i != canon]
    return int(np_rng.choice(options))


def run_phase3_richter_training(
    net: V2Network, bank: TokenBank,
    *, n_trials_learning: int,
    n_trials_scan: int,
    reliability_scan: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    timing: Optional[RichterTiming] = None,
    noise_std: float = 0.0,
    permutation: Optional[tuple[int, ...]] = None,
    metrics_path: Optional[Path] = None,
    log_every: int = 50,
) -> RichterHistory:
    """Run the two-sub-phase Richter trainer in-place on ``net``.

    * Learning sub-phase — ``n_trials_learning`` trials, leader→trailer
      follows ``permutation`` 100 % of the time.
    * Scan sub-phase — ``n_trials_scan`` trials, follows ``permutation``
      with probability ``reliability_scan``; otherwise trailer is drawn
      uniformly from the other five.

    The caller owns ``net``; this function mutates ``W_lm_task`` and
    ``W_mh_task`` in place. No other Parameter is touched — an integrity
    check at exit asserts this.
    """
    if net.phase != "phase3_richter":
        raise PhaseFrozenError(
            f"run_phase3_richter_training requires phase='phase3_richter'; "
            f"got {net.phase!r}"
        )
    if n_trials_learning < 0 or n_trials_scan < 0:
        raise ValueError("trial counts must be ≥ 0")
    if not (0.0 <= reliability_scan <= 1.0):
        raise ValueError(
            f"reliability_scan must be in [0, 1]; got {reliability_scan}"
        )

    cfg = net.cfg
    timing = timing or RichterTiming()
    permutation = permutation or permutation_from_seed(seed)
    if len(permutation) != N_LEAD_TRAIL:
        raise ValueError(
            f"permutation must have length {N_LEAD_TRAIL}; got {len(permutation)}"
        )
    rule = ThreeFactorRule(lr=float(lr), weight_decay=float(weight_decay))

    sha_at_start = net.frozen_sensory_core_sha()
    frozen_keys = _frozen_params(net)
    frozen_snaps = _snapshot_weights(net, frozen_keys)

    np_rng = np.random.default_rng(int(seed))
    history = RichterHistory()
    metrics_fh = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = metrics_path.open("w")

    t_start = time.monotonic()
    try:
        for sub_phase, n_trials, reliab in (
            ("learning", int(n_trials_learning), 1.0),
            ("scan",     int(n_trials_scan),     float(reliability_scan)),
        ):
            for k in range(n_trials):
                leader_pos = int(np_rng.integers(0, N_LEAD_TRAIL))
                trailer_pos = _pick_scan_trailer(
                    leader_pos, permutation, reliab, np_rng,
                )
                info = run_richter_trial(
                    net, bank,
                    leader_pos=leader_pos, trailer_pos=trailer_pos,
                    timing=timing, rule=rule,
                    noise_std=float(noise_std), device=str(cfg.device),
                )
                if k % max(int(log_every), 1) == 0 or k == n_trials - 1:
                    m = TrainStepMetrics(
                        phase_name=sub_phase, trial=k,
                        leader_pos=int(leader_pos),
                        trailer_pos=int(trailer_pos),
                        dw_lm_abs_mean=float(info["dw_lm_abs_mean"].item()),
                        dw_mh_abs_mean=float(info["dw_mh_abs_mean"].item()),
                        trailer_r_l23_mean=float(
                            info["trailer_r_l23_mean"].item()
                        ),
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
            "frozen sensory core SHA changed during Phase-3 Richter "
            "training — LGN/L4 mutated (forbidden)"
        )
    _assert_snapshot_unchanged(net, frozen_snaps)
    return history


def _save_checkpoint(
    net: V2Network, trial: int, out_path: Path,
    permutation: tuple[int, ...],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(trial),
            "state_dict": net.state_dict(),
            "phase": net.phase,
            "frozen_sha": net.frozen_sensory_core_sha(),
            "permutation": [int(x) for x in permutation],
        },
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-3 Richter task-learning driver.",
    )
    p.add_argument("--phase2-checkpoint", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-trials-learning", type=int, default=20_000)
    p.add_argument("--n-trials-scan", type=int, default=10_000)
    p.add_argument("--reliability-scan", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("checkpoints/v2/phase3_richter"),
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
    net.set_phase("phase3_richter")

    permutation = permutation_from_seed(int(args.seed))
    out_path = args.out_dir / f"phase3_richter_s{int(args.seed)}.pt"
    metrics_path = (
        out_path.parent / f"phase3_richter_s{int(args.seed)}_metrics.jsonl"
    )
    run_phase3_richter_training(
        net=net, bank=bank,
        n_trials_learning=int(args.n_trials_learning),
        n_trials_scan=int(args.n_trials_scan),
        reliability_scan=float(args.reliability_scan),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        noise_std=float(args.noise_std),
        permutation=permutation,
        metrics_path=metrics_path,
        log_every=int(args.log_every),
    )
    _save_checkpoint(
        net, args.n_trials_learning + args.n_trials_scan,
        out_path, permutation,
    )
    print(f"phase3_richter checkpoint written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
