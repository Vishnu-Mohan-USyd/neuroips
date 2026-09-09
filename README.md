# Sharpening, dampening, and learned temporal responses

A simple orientation-selective rate circuit learns different response shapes
under task-accuracy and population-activity pressure. This branch includes
**all three network types**, trained checkpoints, results, and reproduction code.

| Network | Training regime | Response |
|---|---|---|
| **Sharpening, v9** | Low activity penalty (`alpha=0.07`). | Expected orientation enhanced; neighboring orientations and flanks suppressed. |
| **Dampening, v9** | Higher activity penalty (`alpha=0.70`), same architecture. | Central suppression with relative flank sparing. |
| **Temporal, v11** | Early decoding protected, activity charged throughout; both pathway kinetics learned. | Early sharpening followed by selective dampening relative to broader flanks; a narrow central peak persists. |

**[Read the network guide](NETWORKS.md)** for architecture, input and training,
scientific assumptions, per-condition results, checkpoint loading, and usage.

The current temporal model learns the relative timing of excitatory prediction
activation and SST recruitment. It is distinct from the retained historical
v10 model, which prescribed slow SST kinetics.

![Learned temporal response](figures/temporal_v11/temporal_response_minimal.png)

![Accuracy and activity controls](figures/temporal_v11/temporal_controls_minimal.png)

## Quick start

```bash
git clone --branch networks-sharpening-dampening-temporal-v11 --single-branch \
    https://github.com/Vishnu-Mohan-USyd/neuroips.git
cd neuroips
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Evaluate all six static sharpening/dampening checkpoints:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python reproduce_figures.py
```

Evaluate all fifteen learned temporal checkpoints, including the matched controls:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python tools/evaluate_temporal.py \
    --out outputs/temporal_v11_evaluation/results.json
```

Recreate the two temporal figures from packaged results without evaluating or training:

```bash
python tools/plot_learned_temporal.py \
    --results figures/temporal_v11/results.json \
    --out-dir outputs/temporal_v11_figures
```

Three-seed fits and controls show that rewarding accuracy throughout removes
late dampening at the tested activity weight; energy-only training suppresses
responses immediately. These are conditional results in a rate model using a
firing-activity proxy for energy. Full response shapes, early versus sustained
accuracy, and modeling limitations are documented in [NETWORKS.md](NETWORKS.md).

- [Static checkpoints](checkpoints/) and [numerical profiles](figures/c6_curves.json)
- [Learned temporal checkpoints](checkpoints/temporal_v11/)
- [Learned temporal numerical results](figures/temporal_v11/results.json)
- [Shared circuit](harness/tuned_emergence_lib.py), [training](harness/train_sweep.py),
  and [temporal training entry point](harness/train_temporal.py)
