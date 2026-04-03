"""Extract results from noise sweep experiments."""
import torch, json, glob, sys
sys.path.insert(0, '.')
from src.analysis.feedback_discovery import extract_profiles, classify_profile
from src.config import ModelConfig
from src.model.network import LaminarV1V2Network

noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
results = []

for noise in noise_levels:
    name = f'noise_{noise:.2f}'.replace('.', 'p')

    # Load metrics
    metrics_files = glob.glob(f'results/sweep/{name}/*/metrics.jsonl')
    if not metrics_files:
        print(f'MISSING: {name}')
        continue

    # Get final metrics
    with open(metrics_files[0]) as f:
        lines = f.readlines()
        final = json.loads(lines[-1])

    # Load model and extract profile
    ckpt_files = glob.glob(f'results/sweep/{name}/*/checkpoint.pt')
    if ckpt_files:
        ckpt = torch.load(ckpt_files[0], map_location='cpu', weights_only=False)
        cfg = ModelConfig(**ckpt['config']['model'])
        net = LaminarV1V2Network(cfg)
        net.load_state_dict(ckpt['model_state'], strict=False)
        net.eval()
        K_inh, K_exc = extract_profiles(net)
        classification = classify_profile(K_inh, K_exc)

        net_modulation_center = (K_exc[0] - K_inh[0]).item()
        net_modulation_flank = (K_exc[3] - K_inh[3]).item()

        alpha_inh_weights = [round(x, 4) for x in net.feedback.alpha_inh.data.tolist()]
        alpha_exc_weights = [round(x, 4) for x in net.feedback.alpha_exc.data.tolist()]

        # Use signed R for best classification
        corrs = classification['correlations']
        best_by_signed_r = max(corrs, key=lambda k: corrs[k])
        profile_signed = best_by_signed_r
        r_signed = corrs[best_by_signed_r]
    else:
        classification = {'winning_class': 'MISSING', 'correlations': {}}
        net_modulation_center = 0
        net_modulation_flank = 0
        alpha_inh_weights = []
        alpha_exc_weights = []
        profile_signed = 'MISSING'
        r_signed = 0

    row = {
        'noise': noise,
        's_acc': final.get('s_acc', 0),
        'cw_acc': final.get('state_acc', 0),
        'a_inh': final.get('a_inh_norm', final.get('a_inh', 0)),
        'a_exc': final.get('a_exc_norm', final.get('a_exc', 0)),
        'net_center': round(net_modulation_center, 5),
        'net_flank': round(net_modulation_flank, 5),
        'profile_r2': classification.get('winning_class', '?'),
        'profile_signed': profile_signed,
        'r_signed': round(r_signed, 4),
        'alpha_inh': alpha_inh_weights,
        'alpha_exc': alpha_exc_weights,
    }
    results.append(row)
    sign = '+' if net_modulation_center >= 0 else ''
    print(f"noise={noise:.2f}: s_acc={row['s_acc']:.3f}, cw_acc={row['cw_acc']:.3f}, "
          f"a_inh={row['a_inh']:.3f}, a_exc={row['a_exc']:.3f}, "
          f"net_center={sign}{net_modulation_center:.5f}, profile={row['profile_signed']} (R={r_signed:+.4f})")

# Save results
with open('results/sweep/sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to results/sweep/sweep_results.json')
