#!/usr/bin/env python3
"""Short pilot: Stage 1 + 15K Stage 2 steps for A/B/C/D.

Runs all 4 mechanisms IN PARALLEL on GPU using torch.multiprocessing.

Go/no-go criteria:
  - V2 prediction accuracy above chance (>8.3% = 1/12 anchors)
  - Sensory readout improving
  - No NaN
  - No stuck models (loss decreasing)
"""
import sys
import logging
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.multiprocessing as mp

from src.config import ModelConfig, TrainingConfig, StimulusConfig, MechanismType
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.training.stage1_sensory import run_stage1
from src.training.stage2_feedback import run_stage2
from src.training.trainer import freeze_stage1

MECHANISMS = [
    MechanismType.DAMPENING,
    MechanismType.SHARPENING,
    MechanismType.CENTER_SURROUND,
    MechanismType.ADAPTATION_ONLY,
]

STAGE2_STEPS = 15000
SEED = 42


def run_mechanism(mech: MechanismType, gpu_id: int, result_dict: dict):
    """Run Stage 1 + Stage 2 for a single mechanism on a specific GPU stream."""
    # Set up per-process logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{mech.value:20s}] %(message)s",
        force=True,
    )
    logger = logging.getLogger(mech.value)

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    logger.info(f"Starting on {device} (GPU: {torch.cuda.get_device_name(device)})")

    model_cfg = ModelConfig(mechanism=mech)
    train_cfg = TrainingConfig(stage2_n_steps=STAGE2_STEPS)
    stim_cfg = StimulusConfig()

    torch.manual_seed(SEED)
    net = LaminarV1V2Network(model_cfg)
    loss_fn = CompositeLoss(train_cfg, model_cfg)

    # Stage 1
    logger.info("--- Stage 1 ---")
    t0 = time.time()
    result1 = run_stage1(net, model_cfg, train_cfg, device, SEED)
    s1_time = time.time() - t0
    logger.info(f"Stage 1: loss={result1.final_loss:.4f}, acc={result1.decoder_accuracy:.3f}, "
                f"gating={result1.gating_passed}, time={s1_time:.1f}s")

    # Stage 2
    logger.info("--- Stage 2 ---")
    freeze_stage1(net)
    t0 = time.time()
    result2 = run_stage2(
        net, loss_fn, model_cfg, train_cfg, stim_cfg,
        device=device, seed=SEED, log_interval=500,
    )
    s2_time = time.time() - t0

    logger.info(f"Stage 2: loss={result2.final_loss:.4f}, "
                f"s_acc={result2.final_sensory_acc:.3f}, "
                f"p_acc={result2.final_pred_acc:.3f}, "
                f"time={s2_time:.1f}s ({s2_time/STAGE2_STEPS*1000:.1f}ms/step)")

    result_dict[mech.value] = {
        "s1_loss": float(result1.final_loss),
        "s1_acc": float(result1.decoder_accuracy),
        "s1_gating": bool(result1.gating_passed),
        "s2_loss": float(result2.final_loss),
        "s2_sensory_acc": float(result2.final_sensory_acc),
        "s2_pred_acc": float(result2.final_pred_acc),
        "s2_time": float(s2_time),
        "s2_loss_history": [float(x) for x in result2.loss_history],
        "has_nan": any(x != x for x in result2.loss_history),
    }
    logger.info("DONE")


def summarize(results: dict):
    """Print summary and go/no-go verdict."""
    logger = logging.getLogger("SUMMARY")

    logger.info("\n" + "=" * 60)
    logger.info("PILOT SUMMARY")
    logger.info("=" * 60)

    all_pass = True
    for name, r in results.items():
        nan_status = "NaN!" if r["has_nan"] else "OK"
        hist = r["s2_loss_history"]
        early_loss = sum(hist[:500]) / max(len(hist[:500]), 1)
        late_loss = sum(hist[-500:]) / max(len(hist[-500:]), 1)
        improving = late_loss < early_loss

        logger.info(
            f"  {name:20s}: s1_acc={r['s1_acc']:.3f}, "
            f"s2_loss={r['s2_loss']:.4f}, "
            f"s_acc={r['s2_sensory_acc']:.3f}, "
            f"p_acc={r['s2_pred_acc']:.3f}, "
            f"early={early_loss:.4f}->late={late_loss:.4f}, "
            f"improving={'YES' if improving else 'NO'}, "
            f"nan={nan_status}"
        )

        checks = {
            "no_nan": not r["has_nan"],
            "loss_improving": improving,
            "pred_above_chance": r["s2_pred_acc"] > 0.05,
        }
        for check, passed in checks.items():
            if not passed:
                logger.warning(f"  {name} FAILED: {check}")
                all_pass = False

    if all_pass:
        logger.info("\nVERDICT: GO — all models learning, proceed to full 80K")
    else:
        logger.warning("\nVERDICT: NO-GO — some checks failed, investigate before full run")

    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("MAIN")

    if not torch.cuda.is_available():
        logger.error("CUDA not available — this script requires GPU")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Use spawn for CUDA multiprocessing safety
    mp.set_start_method("spawn", force=True)

    # Shared dict for results across processes
    manager = mp.Manager()
    result_dict = manager.dict()

    # Launch all 4 mechanisms in parallel
    # All share GPU 0 (each gets its own CUDA stream implicitly)
    t_start = time.time()
    processes = []
    for mech in MECHANISMS:
        gpu_id = 0  # All on same GPU — each process gets own CUDA context
        p = mp.Process(target=run_mechanism, args=(mech, gpu_id, result_dict))
        p.start()
        processes.append((mech.value, p))
        logger.info(f"Launched {mech.value} (PID={p.pid}) on GPU {gpu_id}")

    # Wait for all to finish
    for name, p in processes:
        p.join()
        if p.exitcode != 0:
            logger.error(f"{name} FAILED with exit code {p.exitcode}")

    total_time = time.time() - t_start
    logger.info(f"All mechanisms finished in {total_time:.1f}s")

    # Convert manager dict to regular dict
    results = dict(result_dict)

    if not results:
        logger.error("No results collected — all processes may have failed")
        sys.exit(1)

    # Save raw results
    output_path = Path(__file__).parent.parent / "pilot_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Raw results saved to {output_path}")

    # Print summary
    summarize(results)
