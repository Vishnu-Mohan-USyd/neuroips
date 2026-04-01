#!/usr/bin/env python3
"""Benchmark torch.compile modes for the packed-input forward().

Tests: uncompiled, default, max-autotune-no-cudagraphs, reduce-overhead.
Reports compile time + per-step forward+backward timing.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.config import ModelConfig, TrainingConfig, StimulusConfig, MechanismType
from src.model.network import LaminarV1V2Network
from src.training.losses import CompositeLoss
from src.stimulus.sequences import HMMSequenceGenerator
from src.training.trainer import (
    freeze_stage1, unfreeze_stage2,
    build_stimulus_sequence, compute_readout_indices, extract_readout_data,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

model_cfg = ModelConfig(mechanism=MechanismType.DAMPENING)
train_cfg = TrainingConfig()
stim_cfg = StimulusConfig()
N = model_cfg.n_orientations
B = 32
seq_length = 50
T = seq_length * (train_cfg.steps_on + train_cfg.steps_isi)
print(f'B={B}, T={T}, N={N}')

gen = torch.Generator(device='cpu')
gen.manual_seed(42)
hmm_gen = HMMSequenceGenerator(
    n_orientations=N, p_self=stim_cfg.p_self,
    p_transition_cw=stim_cfg.p_transition_cw,
    p_transition_ccw=stim_cfg.p_transition_ccw,
    n_anchors=stim_cfg.n_anchors, jitter_range=stim_cfg.jitter_range,
    transition_step=stim_cfg.transition_step, period=model_cfg.orientation_range,
    contrast_range=train_cfg.stage2_contrast_range,
    ambiguous_fraction=train_cfg.ambiguous_fraction,
)
readout_indices = compute_readout_indices(
    seq_length, train_cfg.steps_on, train_cfg.steps_isi,
    window_start=4, window_end=7,
)


def make_batch():
    metadata = hmm_gen.generate(B, seq_length, gen)
    stim_seq, cue_seq, task_seq, true_thetas, true_next_thetas = (
        build_stimulus_sequence(metadata, model_cfg, train_cfg)
    )
    return (stim_seq.to(device), cue_seq.to(device), task_seq.to(device),
            true_thetas.to(device), true_next_thetas.to(device))


def run_step(model, loss_fn, optimizer):
    """Full training step: datagen + pack + forward + loss + backward + step."""
    optimizer.zero_grad()
    stim_seq, cue_seq, task_seq, true_thetas, true_next_thetas = make_batch()
    packed = LaminarV1V2Network.pack_inputs(stim_seq, cue_seq, task_seq)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    r_l23_all, _, aux = model(packed)
    outputs = {
        'r_l23': r_l23_all, 'q_pred': aux['q_pred_all'],
        'r_l4': aux['r_l4_all'], 'r_pv': aux['r_pv_all'],
        'r_som': aux['r_som_all'], 'deep_template': aux['deep_template_all'],
    }
    r_l23_win, q_pred_win = extract_readout_data(outputs, readout_indices)
    total_loss, loss_dict = loss_fn(
        outputs, true_thetas, true_next_thetas, r_l23_win, q_pred_win,
    )
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
    optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time() - t0, loss_dict


def benchmark_mode(mode_name, compile_mode=None, n_warmup=3, n_bench=10):
    print(f'\n{"=" * 60}')
    print(f'MODE: {mode_name}')
    print(f'{"=" * 60}')

    torch.manual_seed(42)
    net = LaminarV1V2Network(model_cfg).to(device)
    freeze_stage1(net)
    unfreeze_stage2(net)
    loss_fn = CompositeLoss(train_cfg, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=1e-4
    )

    if compile_mode is not None:
        t0 = time.time()
        model = torch.compile(net, mode=compile_mode)
        print(f'  torch.compile() call: {time.time() - t0:.2f}s')
    else:
        model = net

    # Warmup
    for i in range(n_warmup):
        t0_w = time.time()
        elapsed, ld = run_step(model, loss_fn, optimizer)
        wall = time.time() - t0_w
        print(f'  Warmup {i+1}/{n_warmup}: step={elapsed:.3f}s wall={wall:.3f}s loss={ld["total"]:.4f}')

    # Benchmark
    times = []
    for i in range(n_bench):
        elapsed, ld = run_step(model, loss_fn, optimizer)
        times.append(elapsed)

    mean_t = sum(times) / len(times)
    print(f'  Bench steps: {[f"{t:.3f}" for t in times]}')
    print(f'  Mean: {mean_t:.3f}s  Min: {min(times):.3f}s  Max: {max(times):.3f}s')
    print(f'  Projected 80K: {mean_t * 80000 / 3600:.1f} hours')
    return mean_t


# Run benchmarks
results = {}
results['uncompiled'] = benchmark_mode('Uncompiled (baseline)', compile_mode=None)
results['default'] = benchmark_mode('torch.compile(default)', compile_mode='default')
results['max-autotune-no-cudagraphs'] = benchmark_mode(
    'torch.compile(max-autotune-no-cudagraphs)',
    compile_mode='max-autotune-no-cudagraphs',
)

print(f'\n{"=" * 60}')
print('SUMMARY')
print(f'{"=" * 60}')
baseline = results['uncompiled']
for name, t in results.items():
    speedup = baseline / t if t > 0 else 0
    print(f'  {name:40s}: {t:.3f}s/step  ({speedup:.1f}x vs uncompiled)  '
          f'80K={t * 80000 / 3600:.1f}h')
