"""evaluate the two static networks: response, activity, and decoding."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import assay_emergent_task_energy_axis as assay


def measure_responses(root, device):
    saved = json.loads((root / 'figures/c6_curves.json').read_text())
    theta_a, theta_b, finals = assay.matched_pairs(device)
    repeats = 128
    noise_seed = 940001
    records = {}
    with torch.no_grad():
        for kind, slug in (('sharpening', '0p07'), ('dampening', '0p7')):
            records[kind] = []
            for seed in (8, 9, 10):
                path = root / f'checkpoints/seed{seed}/alpha{slug}/alpha_{slug}_final.pt'
                net, ck = assay.load_arm(path, device)
                assert ck['axis_current_readout'] == 'population_vector'
                ref = ck['references']
                generator = assay.train_sweep.make_generator(device, noise_seed)
                noise = torch.randn((repeats, len(finals), 36), generator=generator, device=device) * ref['sigma_train']
                targets = finals.unsqueeze(0).expand(repeats, -1)
                row = {'seed': seed, 'alpha': float(ck['alpha']), 'checkpoint': str(path.relative_to(root)),
                       'R_ref': ref['R_ref'], 'sigma': ref['sigma_train']}
                base_curves = []
                for condition, theta in (('expected', theta_a), ('unexpected', theta_b)):
                    _, rates, internals = assay.tuned.forward_seq_tuned(
                        net, theta, 1.0, center_feedback=ck['center_feedback'],
                        feedback_mode=ck['feedback_mode'], return_internals=True)
                    final = rates[:, -1]
                    som, vip, som_gain, pre_pv, _, feedback_work, _, _ = [v[:, -1] for v in internals]
                    activity = assay.train_sweep.modeled_population_activity_components(
                        final, som, vip, assay.train_sweep.pv_scalar_from_pre_pv(net, pre_pv),
                        som_gain, feedback_work)
                    metrics = assay.temporal_current_metrics(
                        net, final.unsqueeze(0).expand(repeats, -1, -1), targets, noise, ref, generator)
                    aligned = assay.align_rates(final, finals).double()
                    first_labels = (theta[:, 0] / 5).round().long() % 36
                    base_curves.append(assay.align_rates(rates[:, 0], first_labels).double())
                    row[condition] = {
                        'normalized_activity': float(activity['modeled_population_activity_numerator']) / ref['R_ref'],
                        'accuracy_percent': 100 * metrics['current_accuracy'],
                        'current_ce': metrics['current_ce'],
                        'population_components': {k:float(v) for k,v in activity.items()},
                        'curve': aligned.mean(0).tolist(),
                        'curve_min_across_histories': None,
                    }
                row['baseline_curve'] = (0.5 * (base_curves[0] + base_curves[1])).mean(0).tolist()
                row['offset_degrees'] = [5 * k for k in assay.OFFSETS]
                select = [assay.OFFSETS.index(k) for k in range(-12, 13)]
                banked = saved[f'seed{seed}_{slug}']
                for key, expected_key in (('expected', 'curve_adapted'), ('unexpected', 'curve_unexpected')):
                    assert np.allclose(np.asarray(row[key]['curve'])[select], banked[expected_key], atol=1e-12, rtol=0)
                records[kind].append(row)
    result = {'noise_seed': noise_seed, 'noise_repeats': repeats, 'pair_count': len(finals),
              'decoder': 'fixed training population-vector readout; 36-way current-orientation top-1; paired additive noise',
              'energy': 'R/R_ref from weighted E, PV, SST and VIP firing at the final probe stimulus',
              'probe': 'matched continuation/reversal histories; identical final orientation; all 36 orientations and six signed velocities',
              'records': records}
    return result



def measure_decoding(root, device):
    theta_a, theta_b, finals = assay.matched_pairs(device)
    velocities = torch.tensor(assay.VELOCITIES, device=device).repeat(36)
    train_repeats, test_repeats = 32, 128
    train_noise_seed, test_noise_seed = 910001, 940001
    rows = []
    with torch.no_grad():
        for kind, slug in (('sharpening', '0p07'), ('dampening', '0p7')):
            for seed in (8, 9, 10):
                checkpoint_path = root / f'checkpoints/seed{seed}/alpha{slug}/alpha_{slug}_final.pt'
                net, ck = assay.load_arm(checkpoint_path, device)
                rates = []
                for theta in (theta_a, theta_b):
                    _, response = assay.tuned.forward_seq_tuned(
                        net, theta, center_feedback=ck['center_feedback'],
                        feedback_mode=ck['feedback_mode'])
                    rates.append(response[:, -1])
                sigma = ck['references']['sigma_train']
                test_features = assay.paired_noisy_features(
                    rates[0], rates[1], sigma, test_repeats, test_noise_seed)
                test_features = [f.reshape(test_repeats, len(finals), 36) for f in test_features]
                counts = torch.zeros(2, dtype=torch.int64)
                folds = []
                visited = torch.zeros(len(finals), dtype=torch.int64)
                for speed in (1, 2, 3):
                    held = velocities.abs() == speed
                    fit = ~held
                    assert not (fit & held).any()
                    assert torch.bincount(finals[fit], minlength=36).eq(4).all()
                    assert torch.bincount(finals[held], minlength=36).eq(2).all()
                    visited += held.long()
                    train_a, train_b = assay.paired_noisy_features(
                        rates[0][fit], rates[1][fit], sigma, train_repeats, train_noise_seed)
                    centroids = assay.fit_balanced_cosine_centroids(
                        train_a, train_b, finals[fit], train_repeats)
                    labels = finals[held].repeat(test_repeats)
                    fold = {'held_out_speed_deg_per_step': 5*speed}
                    for c, condition in enumerate(('expected', 'unexpected')):
                        features = test_features[c][:, held].reshape(-1, 36)
                        predictions = (features @ centroids.T).argmax(-1)
                        correct = (predictions == labels).sum()
                        counts[c] += correct.cpu()
                        fold[condition + '_accuracy_percent'] = 100 * float(correct) / len(labels)
                    folds.append(fold)
                assert visited.eq(1).all()
                total = test_repeats * len(finals)
                row = {'kind':kind, 'seed':seed, 'checkpoint':str(checkpoint_path.relative_to(root)),
                       'expected_accuracy_percent': 100*int(counts[0])/total,
                       'unexpected_accuracy_percent':100*int(counts[1])/total,
                       'test_trials_per_condition':total, 'folds':folds}
                rows.append(row)
    result = {
        'decoder':'one balanced pooled expected+unexpected cosine nearest-centroid readout per checkpoint and training fold',
        'condition_input':False,
        'cross_validation':'three leave-one-absolute-velocity-out folds; each holds both velocity signs, all orientations, and both conditions out of decoder fitting',
        'train_noise_seed':train_noise_seed,'test_noise_seed':test_noise_seed,
        'train_repeats':train_repeats,'test_repeats':test_repeats,
        'noise':'checkpoint sigma_train unchanged; paired expected/unexpected test noise identical to the previous figure',
        'rows': rows,
    }
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "static_evaluation" / "results.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = assay.choose_device("cpu")
    result = measure_responses(ROOT, device)
    decoding = measure_decoding(ROOT, device)
    result["decoder"] = decoding["decoder"]
    result["decoder_evaluation"] = decoding
    by_model = {(row["kind"], row["seed"]): row for row in decoding["rows"]}
    for kind, records in result["records"].items():
        for row in records:
            decoded = by_model[kind, row["seed"]]
            for condition in ("expected", "unexpected"):
                values = row[condition]
                values["population_vector_accuracy_percent"] = values["accuracy_percent"]
                values["accuracy_percent"] = decoded[condition + "_accuracy_percent"]
            print(json.dumps({"network": kind, "seed": row["seed"],
                              "expected_decoding": row["expected"]["accuracy_percent"],
                              "unexpected_decoding": row["unexpected"]["accuracy_percent"],
                              "expected_activity": row["expected"]["normalized_activity"],
                              "unexpected_activity": row["unexpected"]["normalized_activity"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
