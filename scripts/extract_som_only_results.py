#!/usr/bin/env python3
"""Extract results from SOM-only experiments for report."""
import sys, json, glob
sys.path.insert(0, '.')

import torch
from src.config import ModelConfig
from src.model.network import LaminarV1V2Network
from src.analysis.feedback_discovery import extract_profiles, classify_profile


def extract_run(run_dir, label, seed):
    """Extract results from a single run."""
    # Find checkpoint
    ckpt_files = glob.glob(f'{run_dir}/*/checkpoint.pt')
    if not ckpt_files:
        print(f"  MISSING checkpoint: {run_dir}")
        return None

    # Load model
    ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
    cfg = ModelConfig(**ckpt['config']['model'])
    net = LaminarV1V2Network(cfg)
    net.load_state_dict(ckpt['model_state'], strict=False)
    net.eval()

    # Extract profile
    K_inh = extract_profiles(net)
    classification = classify_profile(K_inh)

    # Get final metrics
    metrics_files = glob.glob(f'{run_dir}/*/metrics.jsonl')
    final_metrics = {}
    if metrics_files:
        with open(metrics_files[0]) as f:
            lines = f.readlines()
            if lines:
                final_metrics = json.loads(lines[-1])

    # Alpha weights
    alpha_weights = net.feedback.alpha_inh.data.tolist()
    a_inh_norm = net.feedback.alpha_inh.abs().sum().item()

    # Profile values at key orientations
    inh_center = K_inh[0].item()
    inh_near = K_inh[1].item()
    inh_flank = K_inh[9].item()  # 45 deg away

    result = {
        'label': label,
        'seed': seed,
        'a_inh': a_inh_norm,
        'inh_center': inh_center,
        'inh_near': inh_near,
        'inh_flank': inh_flank,
        's_acc': final_metrics.get('s_acc', 0),
        'cw_acc': final_metrics.get('state_acc', 0),
        'alpha_weights': [round(x, 4) for x in alpha_weights],
        'correlations': classification['correlations'],
        'winning_class': classification['winning_class'],
        'winning_r': classification['winning_r'],
    }

    print(f"  {label} s{seed}: a_inh={a_inh_norm:.3f}, "
          f"inh_center={inh_center:.5f}, inh_flank={inh_flank:.5f}, "
          f"s_acc={result['s_acc']:.3f}, cw_acc={result['cw_acc']:.3f}, "
          f"class={result['winning_class']} (R={result['winning_r']:.3f})")
    print(f"    alphas: {result['alpha_weights']}")
    print(f"    R_damp={classification['correlations']['dampening']:.3f}, "
          f"R_sharp={classification['correlations']['sharpening']:.3f}, "
          f"R_cs={classification['correlations']['center_surround']:.3f}")

    return result


print("=" * 70)
print("SOM-ONLY ARCHITECTURE RESULTS")
print("=" * 70)

results = {}

print("\n--- Baseline (config/simple.yaml) ---")
for seed in [42, 123, 456]:
    r = extract_run(f'results/som_only/baseline_s{seed}', 'baseline', seed)
    if r:
        results[f'baseline_s{seed}'] = r

print("\n--- Experiment C: Detection + Discrimination ---")
for seed in [42, 123, 456]:
    r = extract_run(f'results/som_only/expC_s{seed}', 'expC', seed)
    if r:
        results[f'expC_s{seed}'] = r

# Save raw results
with open('results/som_only/som_only_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nRaw results saved to results/som_only/som_only_results.json")
