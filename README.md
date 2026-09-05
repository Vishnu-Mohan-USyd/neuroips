# Expectation suppression: sharpening and dampening in one circuit

This branch contains the current **split-SST, tanh-RNN model**, its trained
sharpening and dampening checkpoints for seeds **8, 9, and 10**, and the code
needed to reproduce their activity–orientation curves.

Both outcomes use the same architecture. For each seed, two copies of a shared
task-pretrained network are trained with different population-activity penalties:
**alpha = 0.05** produces sharpening; **alpha = 0.20** produces dampening.
The architecture identifier is `split_som_tanh_modulation_v8`.

This is a minimal rate-level modeling hypothesis, not a claim that the circuit
matches every aspect of cortical biology. In particular, sensory–prediction
coincidence and the separation of SST into functional pools are explicit
architectural assumptions, not discoveries made by training.

## Reproduce the published results

```bash
git clone --branch c6-biological-sharpening-dampening https://github.com/Vishnu-Mohan-USyd/neuroips.git
cd neuroips
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES="" python reproduce_figures.py
```

The last command loads the six included checkpoints on CPU; it does not train.
It writes six PNGs, six SVGs, and `figures/c6_curves.json`, containing raw curves
and numerical measurements. Missing or incompatible checkpoints cause failure.
Seed-8 reference measurements are checked to tolerance `1e-3`. Paths resolve
relative to the script, not the calling directory.

Checked environment: Python **3.13.7**, PyTorch **2.10.0+cu130**, Matplotlib
**3.10.8**, NumPy **2.3.5**, and pytest **9.0.2**. CUDA is not needed for
checkpoint evaluation. Requirements pin the package versions; the installed
PyTorch CPU/CUDA build depends on the installation environment.

| Sharpening, seed 8 | Dampening, seed 8 |
|---|---|
| ![Sharpening](figures/c6_sharpening_seed8.png) | ![Dampening](figures/c6_dampening_seed8.png) |

[Sharpening SVG](figures/c6_sharpening_seed8.svg) ·
[Dampening SVG](figures/c6_dampening_seed8.svg)

## Architecture

Input consists of orientation angles in degrees, with 180-degree periodicity,
not images. Activity is in arbitrary nonnegative rate units; timesteps have no
assigned duration in milliseconds.

| Part | Representation and role |
|---|---|
| L4 sensory encoder | 36 fixed circular-Gaussian channels, spaced by 5 degrees, with a tuning width of 12 degrees. |
| L2/3 excitatory population, E | 36 orientation channels. A fixed nonnegative feedforward map supplies sensory drive. These final E rates are plotted. |
| Temporal predictor | A 64-unit, ungated `torch.nn.RNNCell` with `tanh` activation. A learned linear projection produces 36 next-orientation logits. |
| Sensory-driven SST, `S_B` | A 36-channel inhibitory field receiving sensory drive pooled through a nine-channel basis. Divisively inhibits E. |
| Prediction-recipient SST, `S_P` | A separate 36-channel inhibitory field driven by sensory input multiplied by spatially smoothed prediction. Lowers E gain. |
| Prediction-driven inhibitory relay | Nine rectified SST-like channels, `s_ff`, that receive feedback and inhibit VIP. Separate from `S_P`. |
| VIP | Nine sensory-excited channels. Inhibits both `S_B` and `S_P` through local orientation footprints. |
| PV | One scalar driven by mean pre-PV E activity; uniformly divides the E response. Not an explicit population of PV neurons. |

### Signal flow

At the first timestep, predictor state and feedback are zero. Subsequently,
feedback is the **ordinary softmax prediction made at the previous step**.
The present stimulus is not used to compute its own incoming prediction.

1. L4 passes through the fixed feedforward map to give sensory drive `D`.
2. Sensory input recruits `S_B` and VIP. Prediction recruits the inhibitory
   relay, reducing VIP, and overlaps with sensory drive to recruit `S_P`.
3. VIP inhibits both SST pools. Sensory-driven SST divides the sensory response.
4. Direct excitatory feedback and prediction-recipient SST compete to increase
   or decrease the remaining response.
5. The broad PV divisor scales the result. The final E response updates the
   RNN, which predicts the next orientation.

After the SST/VIP rates are computed, the core E calculation is:

```text
basal = D / (1 + m * S_B)
modulation = 1 + tanh(w_ef * feedback - m * S_P)
pre_PV = basal * modulation
PV = w_pv * mean(pre_PV over orientation channels)
E = pre_PV / (1 + PV)
```

`S_P` receives `D * (feedback @ K_pred.T)`, followed by a threshold and VIP
inhibition. `K_pred` is a fixed, circular, peak-normalized Gaussian map, applied
identically at every orientation. No expected/unexpected flag is supplied.
Nevertheless, **sensory–prediction multiplication is built into the circuit**.
Modulation is bounded between zero and twice the basal response. Without
sensory drive it cannot generate E activity.

Local pathway magnitudes are nonnegative, with signs assigned explicitly.
This does not make the whole model Dale-compliant: the abstract RNN has
unconstrained signed weights and states. There are no explicit dendritic
compartments, spikes, within-step interneuron dynamics, or local recurrent
E-to-E connections. Adaptation, extra local competition, and optional rate
saturation are disabled. PV is weak in the fitted networks; VIP has the
prescribed disinhibitory route, but the response shapes do not establish its
necessity.

### What learns

Common pretraining updates the RNN and its feedback projection. Task/activity
training updates those same weights plus **one shared prediction-to-SST
recruitment strength**, named `w_sf_fixed` for checkpoint compatibility. Despite
its name, it is trainable in these arms. It controls both the prediction-driven
relay and `S_P` recruitment.

Feedforward maps, anatomical footprints, SST-to-E strength `m`, thresholds,
and other local gains are fixed across arms. For example, `w_ef` is about
0.546, `m` about 0.290, and `w_pv` is 0.0025. Learned `w_sf` is 0.824–0.840
in sharpening and 3.360–3.400 in dampening. RNN and feedback weights also differ
after training; these are not checkpoints with a manually switched SST gain.

## Training pressure

The task equally weights next-orientation cross entropy and current-orientation
cross entropy from a noisy, confidence-weighted circular E readout. Each is
normalized by `log(36)`. Activity is averaged over all sequences, timesteps,
and orientation channels, not restricted to an expected orientation or flanks.

```text
T = 0.5 * normalized_next_CE + 0.5 * normalized_current_CE
R = (5/6) * mean(E) + (37/480) * mean(PV)
    + (1/20) * mean(SST) + (19/480) * mean(VIP)
loss = (1 - alpha) * T + alpha * R / R_ref
```

SST activity is the equal-mass mean of `S_B` and `S_P`. `R` is a dimensionless
weighted activity proxy, **not ATP consumption or a full metabolic budget**.
It does not charge for RNN state, the inhibitory relay, or every synaptic
operation. Neither response shape is a term in the loss.

Training uses momentum sequences of length 12, with velocity bounded at four
channels per step and sticky acceleration in `{-1, 0, 1}`. During pretraining and arm training,
a 2% halt probability applies to eligible transitions whose preceding speed
is at least two channels per step. This changes the sequence; no halt or
expectation label enters the model or loss. Each arm uses the same seed-specific
data and readout-noise streams.

## Results and measurement definitions

The assay uses **216 matched pairs**: all 36 final orientations and six signed
velocities. Expected histories continue their trajectory; unexpected histories
reverse at the final transition. Both have the same final stimulus. The reversal
is an operational out-of-distribution sequence violation, not every experimental
definition of expectation.

Responses are aligned to the final stimulus and averaged over histories. The
horizontal axis denotes **neurons' preferred orientations relative to the
stimulus**, not different probe stimuli presented to one neuron. Plots show
raw first-response, expected, and unexpected activity.

Shape ratios compare the expected final response with the network's own first
response, before feedback arrives. That first response includes sensory
SST/VIP/PV processing; it is not bare feedforward drive. Ratios are ratios of
pooled means, not averages of per-channel ratios.

| Seed | Type / alpha | At 0° / first | At ±5° / first | Flanks ±15–30° / first | Mean expectation suppression |
|---|---|---:|---:|---:|---:|
| 8 | Sharpening / 0.05 | 1.1415 | 0.7886 | 0.9931 | 4.72% |
| 9 | Sharpening / 0.05 | 1.1366 | 0.7885 | 0.9930 | 4.80% |
| 10 | Sharpening / 0.05 | 1.1478 | 0.7901 | 0.9934 | 4.62% |
| 8 | Dampening / 0.20 | 0.2932 | 0.2573 | 0.9560 | 31.39% |
| 9 | Dampening / 0.20 | 0.2879 | 0.2527 | 0.9559 | 31.53% |
| 10 | Dampening / 0.20 | 0.2857 | 0.2496 | 0.9564 | 31.68% |

Sharpening raises the exact expected channel while suppressing its neighbors;
more distant flanks are almost spared. Dampening strongly suppresses the
expected region and relatively spares distant flanks. It is not a perfectly
monotonic central notch: ±5-degree neighbors are slightly more suppressed than
the exact center, and raw expected activity peaks at ±15 degrees.

Expectation suppression is `100 * (1 - mean_expected_E / mean_unexpected_E)`,
using all 36 channels at the last timestep. The expected population mean is
lower in all six checkpoints. This is **not** a claim of lower expected activity
at every orientation. In the JSON, `preferred_ratio` measures exactly 0 degrees;
the historical key `center_ratio` pools -5, 0, and +5 degrees instead.

## Train from scratch

Run from the repository root, using a fresh output directory so the trainer
does not resume an existing run. The full published settings are supplied below;
the general-purpose trainer's default alpha values are not this published pair.

```bash
for seed in 8 9 10; do
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python harness/train_sweep.py \
        --seed "$seed" --device cuda:0 --out outputs/retrained \
        --pretrain-steps 3000 --axis-steps 32000 \
        --batch 128 --sequence-length 12 --mismatch-prob 0.02 \
        --lr 0.001 --clip 5 --alphas 0.05 0.20 \
        --feedback-mode posterior --recurrent-cell rnn_tanh \
        --freeze-local-comp
done
```

CPU training is supported by changing `--device cuda:0` to `--device cpu`.
CPU and CUDA random-number streams can produce different fitted weights. The
trainer enables deterministic PyTorch algorithms and saves optimizer and RNG
states for same-backend resumption. Numerical identity across different devices,
PyTorch builds, or CUDA libraries is not guaranteed. Included checkpoints are
the reference for the published figures; the full training campaign was not
rerun for this release.

Evaluate a freshly trained pair with the existing assay:

```bash
python tools/assay_emergent_task_energy_axis.py \
    --run-dir outputs/retrained/seed_8 --alphas 0.05 0.20 \
    --device cpu --out outputs/retrained/seed_8/assay.json
```

It reports activity, aligned response shape, and held-out noisy decoding. For
included checkpoints, use `--run-dir checkpoints/seed8/alpha0p05 --alphas 0.05`
or `--run-dir checkpoints/seed8/alpha0p2 --alphas 0.20`, with an output path
of your choice.

## Load a network in Python

```python
import sys
import torch

sys.path.insert(0, "harness")
import simple_net as simple
import tuned_emergence_lib as tuned

device = torch.device("cpu")
simple.device = tuned.device = device
simple.prefs = torch.arange(36, device=device).float() * 5.0
checkpoint = torch.load(
    "checkpoints/seed8/alpha0p05/alpha_0p05_final.pt",
    map_location=device, weights_only=False,
)
assert checkpoint["model_architecture_version"] == tuned.MODEL_ARCHITECTURE_VERSION
net = tuned.build_tuned_from_config(checkpoint["tuned_net_config"]).to(device)
net.load_state_dict(checkpoint["state_dict"], strict=True)
net.eval()
orientations = torch.tensor([[0., 10., 20., 30., 40.]], device=device)
with torch.no_grad():
    logits, rates = tuned.forward_seq_tuned(
        net, orientations, 1.0,
        center_feedback=checkpoint["center_feedback"],
        feedback_mode=checkpoint["feedback_mode"],
    )
# logits and rates have shape [batch, timesteps, 36].
# rates[:, -1] is the final E response; logits[:, -1] predicts the next stimulus.
```

Checkpoints contain training state as well as weights; only load ones you trust.
Changing `alpha0p05/alpha_0p05_final.pt` to `alpha0p2/alpha_0p2_final.pt` loads
the dampening network.

## Code and artifact layout

| Path | Contents |
|---|---|
| `harness/tuned_emergence_lib.py` | Current circuit, fixed maps, feedback transforms, model configuration, and recurrent forward pass. `SimpleTunedNet.l23` implements SST/VIP/PV processing. |
| `harness/train_sweep.py` | Sequences, losses, common pretraining, alpha-arm training, and checkpoints. Retains an optional constrained-training mode not used for the published pair. |
| `harness/simple_net.py` | Imported orientation-encoding helpers and older reference implementations. Its legacy `SimpleNet`/GRU is not the released model. |
| `tools/assay_emergent_task_energy_axis.py` | Continuation/reversal histories, orientation alignment, activity and decoding measurements. |
| `reproduce_figures.py` | CPU evaluation and plotting of all six endpoints; checks reference numbers. |
| `tests/test_minimal_biology_circuit.py` | 38 existing tests covering circuit equations/signs, feedback timing, training policies/losses, resume behavior, and reproduction failure handling. |
| `checkpoints/seed{8,9,10}/alpha{0p05,0p2}/` | Final model, seed-shared common pretrain, and original summary. Pretrains are duplicated per arm so each assay directory is self-contained. |
| `figures/` | Reproduced PNG/SVG profiles and JSON curve data. |
| `requirements.txt` | Pinned package versions. |

Training summaries retain their original run paths and settings; corresponding
checkpoints are packaged beside them. Historical alpha-0.0/0.5 models belong to
the earlier architecture and remain on the
[`c6-interneuron-networks` branch](https://github.com/Vishnu-Mohan-USyd/neuroips/tree/c6-interneuron-networks).
Exploratory runs and session notes are not part of this release.

Run the existing tests with:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -m pytest -q tests/test_minimal_biology_circuit.py
```
