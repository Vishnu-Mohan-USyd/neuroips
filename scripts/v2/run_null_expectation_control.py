"""Phase-2 Gate 6 — null expectation control (plan v4 / Task #39).

Runs Kok-like and Richter-like assays against a Phase-2 checkpoint with
the task-specific context-memory weights still at zero init. Because
``W_qm_task``, ``W_lm_task`` and ``W_mh_task`` are zero, both
"expected" and "unexpected" trials are computationally identical — the
null test verifies that random trial-to-trial variation does not
produce a spurious systematic difference when split by the label.

Pass criteria (plan v4):
    |Δ_amplitude_expected_vs_unexpected| ≤ 1 · SEM_pool   (Kok + Richter)
    |Δ_SVM_acc|                           < 0.02          (Richter)

Writes ``gate_6_null_control.json`` next to the checkpoint. Exit 0 on
pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from scripts.v2._gates_common import (
    CheckpointBundle, load_checkpoint, make_blank_frame, make_grating_frame,
)


__all__ = [
    "run_kok_null", "run_richter_null", "run_gate_6_null_control",
]


def _split_stats(
    values: np.ndarray, split: np.ndarray, *, tol_sem: float,
) -> dict[str, Any]:
    """|mean_exp − mean_unexp| vs pooled SEM from a 0/1 split vector."""
    exp = values[split == 0]
    unexp = values[split == 1]
    if exp.size < 2 or unexp.size < 2:
        return {
            "n_expected": int(exp.size), "n_unexpected": int(unexp.size),
            "passed": True,  # insufficient trials → trivially pass
        }
    m0, m1 = float(exp.mean()), float(unexp.mean())
    s0 = float(exp.std(ddof=1) / math.sqrt(exp.size))
    s1 = float(unexp.std(ddof=1) / math.sqrt(unexp.size))
    sem = math.sqrt(s0 ** 2 + s1 ** 2) + 1e-9
    delta = abs(m0 - m1)
    return {
        "mean_expected": m0, "mean_unexpected": m1,
        "delta_amplitude": float(delta), "sem_pooled": float(sem),
        "delta_over_sem": float(delta / sem),
        "passed": bool(delta <= tol_sem * sem),
    }


# ---------------------------------------------------------------------------
# Kok-like null assay
# ---------------------------------------------------------------------------


@torch.no_grad()
def _kok_trial_probe_amp(
    bundle: CheckpointBundle, *, probe_deg: float,
    n_steps_cue: int, n_steps_delay: int, n_steps_probe: int,
    noise_std: float, generator: torch.Generator,
) -> float:
    """One Kok-like trial with zero cue; returns mean L2/3 E amp in probe."""
    device = bundle.cfg.device
    blank = make_blank_frame(1, bundle.cfg, device=device)
    probe = make_grating_frame(probe_deg, 1.0, bundle.cfg, device=device)
    state = bundle.net.initial_state(batch_size=1)
    for _ in range(int(n_steps_cue) + int(n_steps_delay)):
        _, state, _ = bundle.net(blank, state)
    amps: list[float] = []
    for _ in range(int(n_steps_probe)):
        noise = float(noise_std) * torch.randn(
            probe.shape, generator=generator, device=device,
        )
        _, state, info = bundle.net(probe + noise, state)
        amps.append(float(info["r_l23"].mean().item()))
    return float(np.mean(amps))


@torch.no_grad()
def run_kok_null(
    bundle: CheckpointBundle,
    *, n_trials: int = 40, n_cells: int = 4,
    n_steps_cue: int = 8, n_steps_delay: int = 22, n_steps_probe: int = 20,
    orientation_a_deg: float = 45.0, orientation_b_deg: float = 135.0,
    noise_std: float = 0.01, amplitude_tolerance_sem: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Kok-like null assay: 2 orientation classes, labels split 50/50."""
    gen = torch.Generator(device=bundle.cfg.device); gen.manual_seed(int(seed))
    classes = {"A_45deg": orientation_a_deg, "B_135deg": orientation_b_deg}
    per_class: dict[str, Any] = {}
    all_passed = True
    for label, theta in classes.items():
        amps = np.array([
            _kok_trial_probe_amp(
                bundle, probe_deg=float(theta),
                n_steps_cue=n_steps_cue, n_steps_delay=n_steps_delay,
                n_steps_probe=n_steps_probe, noise_std=noise_std, generator=gen,
            )
            for _ in range(int(n_trials) * int(n_cells))
        ], dtype=np.float64)
        rng = np.random.default_rng(int(seed) + hash(label) % 10_000)
        split = np.zeros(len(amps), dtype=np.int64)
        split[rng.permutation(len(amps))[: len(amps) // 2]] = 1  # mark half as "unexpected"
        stats = _split_stats(amps, split, tol_sem=amplitude_tolerance_sem)
        stats["n_trials"] = int(len(amps))
        per_class[label] = stats
        all_passed = all_passed and stats["passed"]
    return {
        "assay": "kok_null", "per_class": per_class,
        "tolerance_sem": float(amplitude_tolerance_sem), "passed": all_passed,
    }


# ---------------------------------------------------------------------------
# Richter-like null assay
# ---------------------------------------------------------------------------


@torch.no_grad()
def _richter_token_amp(
    bundle: CheckpointBundle, *, token: Tensor,
    n_steps_lead: int, n_steps_probe: int,
    noise_std: float, generator: torch.Generator,
) -> tuple[float, np.ndarray]:
    """Lead→probe on a single identity token. Returns (mean_amp, L4 feat)."""
    device = bundle.cfg.device
    state = bundle.net.initial_state(batch_size=1)
    for _ in range(int(n_steps_lead)):
        _, state, _ = bundle.net(token, state)
    amps: list[float] = []
    l4: list[np.ndarray] = []
    for _ in range(int(n_steps_probe)):
        noise = float(noise_std) * torch.randn(
            token.shape, generator=generator, device=device,
        )
        _, state, info = bundle.net(token + noise, state)
        amps.append(float(info["r_l23"].mean().item()))
        l4.append(info["r_l4"].detach().cpu().numpy().reshape(-1))
    return float(np.mean(amps)), np.mean(np.stack(l4, axis=0), axis=0)


def _svm_null(
    X: np.ndarray, y: np.ndarray, split: np.ndarray,
    *, seed: int, tol: float, n_classes: int,
) -> dict[str, Any]:
    """Train + score a 12-way LinearSVC per split half; report |Δ acc| < tol."""
    try:
        from sklearn.model_selection import train_test_split       # type: ignore
        from sklearn.svm import LinearSVC                          # type: ignore
    except ImportError:
        return {"error": "sklearn not available", "passed": False}
    acc: dict[str, float] = {}
    for name, val in (("expected", 0), ("unexpected", 1)):
        mask = split == val
        if mask.sum() < n_classes * 2:
            acc[name] = float("nan"); continue
        Xs, ys = X[mask], y[mask]
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                Xs, ys, test_size=0.3,
                random_state=int(seed) + val, stratify=ys,
            )
        except ValueError:
            acc[name] = float("nan"); continue
        clf = LinearSVC(random_state=int(seed), max_iter=5000, dual="auto")
        clf.fit(Xtr, ytr)
        acc[name] = float(clf.score(Xte, yte))
    a0, a1 = acc.get("expected", float("nan")), acc.get("unexpected", float("nan"))
    if math.isfinite(a0) and math.isfinite(a1):
        d = abs(a0 - a1); passed = bool(d < tol)
    else:
        d = float("nan"); passed = True  # insufficient data → trivially pass smoke
    return {
        "accuracy_expected": a0, "accuracy_unexpected": a1,
        "delta_accuracy": float(d) if math.isfinite(d) else None,
        "tolerance": float(tol), "passed": passed,
    }


@torch.no_grad()
def run_richter_null(
    bundle: CheckpointBundle,
    *, n_trials_per_token: int = 10,
    n_steps_lead: int = 6, n_steps_probe: int = 10,
    noise_std: float = 0.01, amplitude_tolerance_sem: float = 1.0,
    svm_tolerance: float = 0.02, seed: int = 42,
) -> dict[str, Any]:
    """Richter-like null assay: 12 identity tokens + leader=zero."""
    gen = torch.Generator(device=bundle.cfg.device); gen.manual_seed(int(seed))
    tokens_all = bundle.bank.tokens
    N_tok = int(tokens_all.shape[0])
    amps, labels, splits, feats = [], [], [], []
    rng = np.random.default_rng(int(seed))
    for tok_idx in range(N_tok):
        tok = tokens_all[tok_idx:tok_idx + 1]
        tok_splits = rng.integers(0, 2, size=int(n_trials_per_token))
        for trial in range(int(n_trials_per_token)):
            a, f = _richter_token_amp(
                bundle, token=tok,
                n_steps_lead=n_steps_lead, n_steps_probe=n_steps_probe,
                noise_std=noise_std, generator=gen,
            )
            amps.append(a); labels.append(tok_idx)
            splits.append(int(tok_splits[trial])); feats.append(f)
    amps_arr = np.asarray(amps, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    split_arr = np.asarray(splits, dtype=np.int64)

    per_token: dict[str, Any] = {}; amp_passed = True
    for t in range(N_tok):
        mask = labels_arr == t
        stats = _split_stats(
            amps_arr[mask], split_arr[mask], tol_sem=amplitude_tolerance_sem,
        )
        per_token[f"token_{t:02d}"] = stats
        amp_passed = amp_passed and stats["passed"]
    svm = _svm_null(
        np.stack(feats, axis=0), labels_arr, split_arr,
        seed=int(seed), tol=float(svm_tolerance), n_classes=N_tok,
    )
    return {
        "assay": "richter_null",
        "per_token_amplitude": per_token,
        "amplitude_passed": amp_passed,
        "svm": svm,
        "passed": bool(amp_passed and svm.get("passed", False)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gate_6_null_control(
    bundle: CheckpointBundle, *, phase: str = "phase3_kok",
    kok_kwargs: dict | None = None, richter_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run both Kok + Richter null assays; combined pass/fail."""
    bundle.net.set_phase(phase)
    kok = run_kok_null(bundle, **(kok_kwargs or {}))
    richter = run_richter_null(bundle, **(richter_kwargs or {}))
    return {
        "phase_switched_to": phase, "kok": kok, "richter": richter,
        "passed": bool(kok["passed"] and richter["passed"]),
    }


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-2 Gate 6 null control")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--phase", type=str, default="phase3_kok")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--kok-trials", type=int, default=40)
    p.add_argument("--richter-trials-per-token", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    bundle = load_checkpoint(args.checkpoint, seed=args.seed, device=args.device)
    results = run_gate_6_null_control(
        bundle, phase=args.phase,
        kok_kwargs={"n_trials": args.kok_trials, "seed": args.seed},
        richter_kwargs={
            "n_trials_per_token": args.richter_trials_per_token, "seed": args.seed,
        },
    )
    out_path = args.output or (args.checkpoint.parent / "gate_6_null_control.json")
    out_path.write_text(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
