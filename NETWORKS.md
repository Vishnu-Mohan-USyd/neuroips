# network guide: task-prioritized, energy-prioritized, and temporal

this guide describes the three main networks, the measurements supporting them,
how to run their code, and the earlier experiments that led to the current
temporal model. the main temporal model uses **peak decoding with an uncapped
2× or 3× initialization**. it does not use the earlier early-decoding objective.

## main package

| network | training | principal response |
|---|---|---|
| task-prioritized static | activity weight `alpha=0.07` | expected center enhanced; neighboring channels and broader flanks suppressed |
| energy-prioritized static | same static circuit, `alpha=0.70` | expected center strongly suppressed; broader flanks relatively spared |
| temporal | `alpha=0.30`, peak decoding, two independently learned time constants | early center enhancement and flank suppression, followed by greater proportional center suppression late |

the static networks are two trained regimes of the same architecture. the 2×
and 3× temporal starts are two initializations of one temporal architecture.
there are six static final checkpoints and six temporal final checkpoints,
covering seeds 8, 9, and 10. the repeated 2× run is documented as a repeat of the
same seeds, not counted as three additional independent seeds.

```text
harness/
  simple_net.py                      orientation encoding and sequence utilities
  tuned_emergence_lib.py              shared static and temporal rate circuit
  train_sweep.py                      static training and shared task/activity loss
  train_temporal.py                   main uncapped peak-decoding temporal recipe
tools/
  assay_emergent_task_energy_axis.py   shared response, activity, and decoder helpers
  evaluate_static.py                 fresh static activity/decoding/response assay
  evaluate_temporal.py               fresh native-grid peak temporal assay
  plot_static_networks.py             main static paper figure
  plot_learned_temporal.py            main temporal profile with linked snapshots
reproduce_figures.py                  original static response-curve reproduction
checkpoints/
  seed{8,9,10}/alpha0p07/             task-prioritized static + common pretrain
  seed{8,9,10}/alpha0p7/              energy-prioritized static + common pretrain
  temporal_peak/
    init_2x/seed_{8,9,10}/            main temporal start: tau_E=.5, tau_S=1
    init_3x/seed_{8,9,10}/            main temporal start: tau_E=1/3, tau_S=1
results/
  static/main.json                    corrected static decoding, activity, curves
  temporal/init_2x.json               full three-seed temporal measurements
  temporal/init_3x.json               same measurements for the 3× start
figures/
  c6_curves.json                      original static response-curve reference
  static/figure1_static_networks.*    current static paper figure
  temporal/init_{2,3}x/               current temporal response figures
tests/test_minimal_biology_circuit.py existing circuit checks
```

each temporal seed directory contains `common_initial.pt`,
`alpha_0p3_final.pt`, `training_summary.json`, and
`peak_reference_limits.json`. the checkpoint stores model configuration,
weights, optimizer state, random-generator states, activity/noise references,
training step, and the learned ce multiplier. these files are copied from the
completed experiments without altering the checkpoints.

historical v8, fixed-timing v10, early-priority v11, control, capped, and scratch
checkpoint families are excluded from this commit preparation. their existing
local files remain on disk under ignored paths. the shared library retains its
existing compatibility code; the main entry points require only the files above.
internal version strings are compatibility identifiers: both earlier learned
temporal models and the current model use `learned_temporal_v11` architecture
metadata. the current model is distinguished by
`predictor_readout=peak_evidence_noiseless`,
`temporal_current_window=peak`, and a `dual` ce constraint.

## install and run

run these commands from this checkout. the validated environment is python
3.13.7, pytorch 2.10.0+cu130, numpy 2.3.5, matplotlib 3.10.8, and pytest 9.0.2.
the release versions are pinned in `requirements.txt`; evaluation supports cpu.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### evaluate all packaged networks

these commands perform fresh inference. saved results are references, not an
inference cache. outputs go under the ignored `outputs/` directory.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python tools/evaluate_static.py \
    --out outputs/static_evaluation/results.json

for ratio in 2 3; do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python tools/evaluate_temporal.py --init-ratio "$ratio" \
        --out "outputs/temporal_evaluation/init_${ratio}x.json"
done
```

the static evaluator measures all six endpoints. the temporal evaluator
measures all three seeds for the selected initialization; `--seeds 8` selects
one. temporal inference uses the trained sampling grid, `dt=0.1`.

the static output separates `accuracy_percent` from
`population_vector_accuracy_percent`. its `current_ce` is the
population-vector diagnostic, not a ce score from the figure's centroid decoder.
in temporal output, `selected` contains the actual peak-decoding performance
and activity used for the reported objective. `held_out` contains separately
measured instantaneous accuracy curves and window averages.
`held_out.current_ce` is an early-window diagnostic, not the peak objective.

### reproduce the main figures

```bash
python tools/plot_static_networks.py \
    --results results/static/main.json --out outputs/static_figures

for ratio in 2 3; do
    python tools/plot_learned_temporal.py \
        --results "results/temporal/init_${ratio}x.json" \
        --out-dir "outputs/temporal_figures/init_${ratio}x"
done
```

replace the input paths with fresh evaluation results to plot those results.
the plotters write png, svg, and pdf files. all labels are lowercase, axes are
minimal, and shaded bands show the range across the three seeds, not confidence
intervals. the temporal figure links snapshots at `t=.1,1,2,4` to their
locations on the full timecourse and includes a magnified late response.

the original static raw-curve check remains available:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python reproduce_figures.py --out outputs/static_response_check
```

it evaluates the six static checkpoints, writes individual and mean response
curves, and checks the original seed-8 banked measurements to `1e-3`.
the obsolete archived-network overlay has been removed.

## shared circuit, stimulus, and prediction

the input is an **orientation angle**, not an image or spike train. orientations
wrap after 180°. there are 36 channels spaced by 5°. a fixed circular-gaussian
layer-4 encoder has a 12° tuning width. a fixed nonnegative feedforward map,
with width 1.1 channels and gain 1.6, supplies sensory drive `D`.

| component | representation and role |
|---|---|
| excitatory response | 36 nonnegative orientation-channel rates; these are the plotted responses |
| sequence predictor | 64-unit tanh `RNNCell`, followed by the learned linear projection `W_fb` to 36 next-orientation logits |
| basal sst | sensory-driven inhibitory field, using a nine-channel pooling basis, that divides the sensory response |
| prediction sst | sensory/prediction coincidence recruits a 36-channel inhibitory field, projected onto excitatory channels |
| sst-like relay and vip | nine rectified prediction-driven relay channels inhibit nine vip channels; vip inhibits both sst fields |
| pv | scalar proportional to mean pre-pv excitation; divisively scales all orientation channels |

the predictor's legacy attribute name is `gru`, but these checkpoints use a
**tanh rnn, not a gru**. incoming feedback `f` is the softmax prediction from
the previous real stimulus. the first stimulus starts with zero predictor state
and zero feedback. the network never receives an expected/unexpected label.

let `p = f @ K_pred.T`. the prediction footprint `K_pred` is circular,
peak-normalized, and has width sqrt(2) channels. the fixed positive sst output
map `P` is row-normalized with width two channels, or 10°. the core computation is:

```text
basal = D / (1 + m * S_B)
u = excitatory_prediction_activation - m * (S_P @ P.T)
pre_PV = basal * (1 + tanh(u))
PV = w_pv * mean(pre_PV)
E = pre_PV / (1 + PV)
```

the modulation before pv lies between zero and twice the basal response.
without sensory drive, this circuit cannot generate excitatory firing.
the two sst fields, fixed spatial maps, multiplicative coincidence, and
divisive normalization are architectural assumptions.

for static networks the prediction routes act immediately:

```text
excitatory_prediction_activation = g_E * f
S_P = relu(g_S * D * p - theta_S - w_sv * VIP_36)
```

training sequences contain 12 orientations with momentum, velocity up to four
channels per step, and sticky acceleration in `{-1,0,1}`. a 2% halt probability
is applied on eligible transitions after a speed of at least two channels.
the generator changes the sequence itself; it does not supply a surprise flag.

common static pretraining updates the rnn and `W_fb` for 12,000 steps. each
static arm then trains for 12,000 steps, also updating the sst input gain.
the excitatory feedback gain, local anatomical maps, thresholds, local
competition, and other circuit gains are fixed in those static arms.

## main temporal architecture

the temporal extension adds only two independent first-order states:
effective excitatory prediction activation `x`, and actual prediction-sst
firing `z`. both time constants have positive softplus parameterizations.
their magnitudes and ordering are learned freely.

```text
x_target = g_E * f
z_target = g_S * relu(D * p - theta_S - w_sv * VIP_36)

x(t) = x(0) * exp(-t/tau_E) + x_target * (1 - exp(-t/tau_E))
z(t) = z(0) * exp(-t/tau_S) + z_target * (1 - exp(-t/tau_S))

excitatory_prediction_activation = x(t)
S_P = z(t)
```

the exponential dynamics are evaluated analytically at the sampling times.
`x` represents effective synaptic/apical activation, not another firing
population. `z` is the sst activity used for both inhibition and activity cost.

the temporal sst gain acts **after rectification**, whereas the static gain
scales the input before threshold. this response-gain parameterization was
introduced because the earlier form could leave sst completely silent with
zero gradients. it preserves a trainable gain on an active underlying drive.
it is a phenomenological recruitment approximation, not an established local
plasticity mechanism. the same parameter retains its input-gain role in the
auxiliary relay to vip. its checkpoint name is `w_sf_fixed`; the temporal
configuration `sst_response_gain=true` gives it positive softplus semantics.

| protocol quantity | main setting |
|---|---|
| stimulus duration | 4 relative time units |
| following zero-input blank | 5 relative time units, including the terminal blank |
| samples | `t=0,.1,.2,...,4` |
| decoding candidates | all 40 positive samples, `.1,...,4` |
| early reporting window | mean of `.1,.2` |
| late reporting window | mean of `3.1,...,4` |
| 2× initialization | `tau_E=.5, tau_S=1` |
| 3× initialization | `tau_E=1/3, tau_S=1` |
| timing bound | none |

the stimulus and incoming prediction remain fixed throughout the on-period.
during the blank both visually gated targets are zero, states decay naturally,
and predictor memory is held. residual sst firing during the blank is charged.
states reset at independent-sequence boundaries, not between successive
stimuli. time units have no established mapping to milliseconds.

### how peak decoding removes explicit early priority

for every positive sample, compute the magnitude squared of the doubled-angle
orientation resultant of the **noiseless** population response:

```text
score(t) = sum_j(E_j(t) * cos(2*orientation_j))^2
         + sum_j(E_j(t) * sin(2*orientation_j))^2
t_star = argmax over t=.1,...,4 of score(t)
selected_response = E(t_star)
```

one entire actual population snapshot is selected. this is not a separate
maximum for each neuron, and it is not a label-dependent minimum of decoding
error. the selected time is allowed to be early or late. exact ties use the
first candidate; gradients flow through the selected response, not through
the discrete time choice.

current decoding receives one gaussian noise draw **after** snapshot selection.
the rnn receives the same selected snapshot without decoder noise and updates
once, predicting the next real stimulus. the 40 internal samples are not
40 additional sequence elements. the current implementation uses 40 candidates,
not the illustrative five timesteps proposed during discussion.

this objective rewards one informative sample per stimulus. it does not reward
accuracy equally at every sample and does not impose an early accuracy window.
selecting the maximum over a completed response is an offline engineering
readout approximation; it is not an implemented causal biological detector.
the architecture and initialization still influence which strategy optimization
finds.

### initialization and trainable parameters

the main starts use each seed's original static common pretrain at
`checkpoints/seedN/alpha0p07/common_pretrain_final.pt`. they do not initialize
from an already trained early-priority temporal endpoint. the static weights
are copied, sst gain is converted to its inverse-softplus coordinate, and the
two new time constants are initialized as above.

temporal arm training updates the rnn, `W_fb`, excitatory feedback gain, sst
response gain, and both time constants. local competition and the other local
circuit parameters remain fixed. the 2× and 3× recipes differ only in the
initial excitatory time constant; there is no hard ratio cap.

## training objective and energy accounting

the current decoder used in training is a fixed confidence-weighted
population-vector readout. rates receive additive gaussian noise with
`sigma_train=.25*A_ref`, are rectified, and are projected onto doubled-angle
sine/cosine directions. resultant magnitude determines confidence in the
36 orientation logits. next-stimulus prediction uses ordinary cross entropy
on `W_fb` logits. both targets are actual orientation classes.

```text
T = (current_CE + next_CE) / (2 * log(36))

R(t) = (5/6)*mean(E)
     + (37/480)*mean(PV)
     + (1/20)*mean((S_B + S_P)/2)
     + (19/480)*mean(VIP)

static loss = (1-alpha)*T + alpha*R/R_ref

J = integral of R over stimulus plus following blank
temporal base loss = .7*T + .3*J/J_ref
temporal training loss = temporal base loss + lambda*(current_CE - ceiling)
lambda <- max(0, lambda + detached(current_CE - ceiling))
```

`lambda` starts at zero and is updated after each optimizer step. it is an
engineering dual-constraint procedure, not a proposed neural learning rule.
the main fits use this adaptive multiplier rather than the historical fixed
quadratic penalty. old penalty-coefficient metadata may remain in a checkpoint;
it is inactive when the constraint method is `dual`.

the ceiling is calibrated separately from a static-source temporal reference
with **equal initial time constants of 1**, using peak decoding. the chosen
2×/3× initialization does not change the calibration reference. calibration
uses 1,024 twelve-stimulus sequences, data seed 920001, and noise seed 920002.
the original per-seed current-ce ceilings are approximately 1.79534334,
1.79532997, and 1.79532011. small cpu/cuda numerical differences are possible.

the energy weight is uniform per unit time. there is no increasing late penalty.
on-period activity uses trapezoidal integration at the trained grid; blank sst
decay is integrated analytically. `J_ref=4*R_ref` is a fixed normalization,
not the measured cost of the fitted temporal model. minimizing total activity
and its mean is equivalent here because every cycle has the same duration.

`R` and `J` are population-weighted activity proxies. they are not atp,
joules, or metabolic measurements. they include modeled excitatory, pv, sst,
and vip firing, but omit the rnn, auxiliary relay, and detailed synaptic work.
the effective excitatory state has no separate population charge. these
assumptions define what a reduction in the reported “energy” means.

## results and their interpretation

### response-shape measurement

the shape assay uses 216 matched continuation/reversal history pairs:
36 final orientations and six signed velocities. both histories end at the
same presented orientation. the first response, before predictive feedback,
is the sensory baseline; it already includes local inhibition.

the horizontal axis shows population channel preference relative to the final
stimulus, not a stimulus sweep through one recorded neuron. “center” is the
0° channel. broad flanks are ±15°, ±20°, ±25°, and ±30°. each group is averaged
before dividing by its corresponding baseline, then results are averaged
across seeds. a ratio of 1 means unchanged.

### static networks

| three-seed mean | task-prioritized | energy-prioritized |
|---|---:|---:|
| expected center / baseline | 1.1118 | 0.0907 |
| expected broad flanks / baseline | 0.8949 | 0.4910 |
| expected activity / reference | 0.8703 | 0.3637 |
| unexpected activity / reference | 0.9571 | 0.7050 |
| expected decoding | 82.00% | 8.74% |
| unexpected decoding | 59.55% | 22.98% |

![static response, activity, and decoding](figures/static/figure1_static_networks.png)

the static energy-prioritized profile has off-center maxima near ±20°.
the task-prioritized profile enhances the preferred channel and suppresses
nearby channels. the dashed activity line at 1 denotes the fixed activity
reference; the decoding chance line is 1/36, or 2.78%.

the static paper figure uses a **single balanced cosine nearest-centroid
decoder per checkpoint and training fold**, fitted jointly to expected and
unexpected responses. condition identity is not an input. three folds each
hold out one absolute velocity, 5°, 10°, or 15° per step, including both signs
and both conditions. all 36 orientation classes remain represented in training.
this holds out velocity histories and noise, not orientation classes.

noisy rates are rectified and normalized by l1 then l2 norm. fitting uses
32 noise repeats with seed 910001; testing uses 128 with seed 940001.
expected/unexpected test noise is paired. there are 27,648 test trials per
condition per checkpoint across the folds. the noise scale is unchanged.

this was an **evaluation change, not network retraining**. the original
population-vector measurement gave task-prioritized expected accuracy 39.61%
and unexpected accuracy 44.51%. that decoder compresses the full pattern into
a circular resultant; it does not exploit the narrow center enhancement as
effectively as the fitted full-pattern classifier. both measurements are
retained in the numerical results. the ordering is a measured result of the
specified decoder and data, not a universal consequence of a sharper profile.
static paper accuracy and temporal training-decoder accuracy should not be
compared as if they were the same assay.

### main temporal network: 2× and 3× starts

| three-seed mean | 2× start | 3× start |
|---|---:|---:|
| early center / baseline | 1.7897 | 1.7896 |
| early broad flanks / baseline | 0.8530 | 0.8530 |
| late center / baseline | 0.0991 | 0.0991 |
| late broad flanks / baseline | 0.3269 | 0.3269 |
| selected-snapshot decoding | 49.1889% | 49.1943% |
| next-stimulus accuracy | 80.3474% | 80.3593% |
| cycle activity / reference | 0.54529 | 0.54530 |
| learned tau_E | 0.005886 | 0.005886 |
| learned tau_S | 0.89347 | 0.89350 |
| learned tau_S/tau_E | 151.80× | 151.80× |
| selected time after the first stimulus | 0.1 | 0.1 |

all six fits completed 24,000 updates. the 2× repeat used the same three seeds,
initial states, devices, and training budget. its measured expected/unexpected
curves, selected decoding, activity, and time constants exactly matched the
original 2× evaluation in the tested environment. this establishes repetition
of those runs, not additional independent-seed robustness.

for the 2× start, the first positive sample alone has center/baseline 1.8427
and flanks/baseline 0.8976. these differ from the table's average over the two
early samples. late center activity is about 10% of baseline, whereas broad
flanks retain about 33%.

![main 2× temporal response](figures/temporal/init_2x/temporal_response.png)

the late response has greater **proportional center suppression relative to
baseline** than the broad-flank average. it still contains a small raw central
peak: late center/±15° flank is approximately 1.58–1.66. it is not a complete
inversion of tuning at every spatial scale. the near-center channels and
broader flanks must not be treated as interchangeable.

held-out temporal evaluation uses 1,024 twelve-stimulus sequences, data seed
930001 and peak-decoder noise seed 930002. the instantaneous accuracy curve
uses separately paired early and other-time noise, with auxiliary seed 930003.
mean accuracy across the on-period is approximately 15.12%, and late accuracy
approximately 9.73%. only the selected snapshot is rewarded by the main task.

the held-out selected ce is approximately 1.79615, slightly above the calibrated
ceiling of approximately 1.79533. the files correctly record
`ceiling_feasible=false` at tolerance `1e-6`. the dual procedure achieved
near-ceiling performance, not exact constraint satisfaction.

**2× and 3× describe initialization only.** the fitted gap grows to about
152×. the excitatory constant becomes much smaller than `dt=.1`, making that
route effectively settled by the first positive sample. this ratio is not an
empirical biological estimate. changing the sampling grid changes the set
of candidate snapshots and can change recurrent predictions; a new peak grid
is not just a finer measurement of an otherwise identical readout policy.

## retraining

### static networks

```bash
for seed in 8 9 10; do
    CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        python harness/train_sweep.py \
        --seed "$seed" --device cuda:0 --out outputs/static_retrained \
        --pretrain-steps 12000 --axis-steps 12000 \
        --batch 128 --sequence-length 12 --mismatch-prob 0.02 \
        --lr 0.001 --clip 5 --alphas 0.07 0.70 \
        --feedback-mode posterior --recurrent-cell rnn_tanh \
        --axis-current-readout population_vector --freeze-local-comp
done
```

### main temporal network

```bash
for ratio in 2 3; do
    for seed in 8 9 10; do
        CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
            python harness/train_temporal.py \
            --init-ratio "$ratio" --seed "$seed" --device cuda:0 \
            --out outputs/temporal_retrained --axis-steps 24000 --lr 0.003
    done
done
```

the temporal wrapper fixes peak decoding, dual ce, alpha .3, batch 128,
sequence length 12, mismatch probability .02, and gradient clipping 5.
optimization uses adam. it learns the two pathway gains and time constants
alongside the predictor. it reconstructs the initial weights from the packaged
static pretrain and uses the matching packaged temporal initial configuration.
this reproduces temporal arm training, not a new temporal pretraining stage.

output directories are `<out>/init_2x/seed_N/` or
`<out>/init_3x/seed_N/`. use a fresh output directory for a fresh fit.
an existing run resumes its saved optimizer and random-generator state.
`--device cpu` is supported. exact training agreement across different
devices or software builds is not promised. the original 2×/3× fits used
cuda:0 for seeds 8/9 and cuda:1 for seed 10.

## loading and using the networks in python

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, "tools")
import assay_emergent_task_energy_axis as assay

paths = {
    "task_prioritized": "checkpoints/seed8/alpha0p07/alpha_0p07_final.pt",
    "energy_prioritized": "checkpoints/seed8/alpha0p7/alpha_0p7_final.pt",
    "temporal": "checkpoints/temporal_peak/init_2x/seed_8/alpha_0p3_final.pt",
}
device = assay.choose_device("cpu")
net, checkpoint = assay.load_arm(Path(paths["temporal"]), device)
theta = torch.tensor([[0., 10., 20., 30., 40.]], device=device)
with torch.no_grad():
    result = assay.tuned.forward_seq_tuned(
        net, theta,
        center_feedback=checkpoint["center_feedback"],
        feedback_mode=checkpoint["feedback_mode"],
        return_timecourse=net.temporal_protocol is not None,
    )
logits, rates = result[:2]  # [batch, real stimulus, 36]
if net.temporal_protocol is not None:
    trace = result[2]
    times = trace["times"]  # [41]
    timecourse = trace["rates"]  # [batch, real stimulus, 41, 36]
    selected, indices = assay.tuned.select_peak_evidence(net, timecourse)
```

static `rates` are instantaneous responses; temporal `rates` are the
end-of-stimulus responses at `t=4`. use `selected` for the main temporal
readout or `timecourse` for the full response. load configuration as well as
weights. `assay.choose_device` also sets the fixed orientation preferences
on the correct device. checkpoint files are trusted project artifacts that
contain pytorch training state.

## experimental development and outcomes

these records explain the model choices. historical checkpoints and scratch
runners are not part of the main package. comparisons across stages are not
all matched interventions: the timing, task, initialization source, or gain
parameterization changed between some stages.

### 1. fixed slow sst and early accuracy

the first temporal extension used instantaneous excitation and a prescribed
sst recruitment constant of 1. early current accuracy was rewarded at .1/.2;
activity was charged over the stimulus and blank. increasing activity pressure
strengthened late suppression, but delayed sst already supplied a preferred
temporal ordering. those results could not establish learned relative timing.

| activity alpha | early center | late center | late broad flanks | early decoding | cycle activity |
|---|---:|---:|---:|---:|---:|
| 0 | 1.432 | 1.432 | 1.002 | 50.88% | 1.065 |
| .07 | 1.158 | .083 | .476 | 42.53% | .544 |
| .30 | 1.131 | .057 | .444 | 41.63% | .529 |
| .70 | 1.130 | .054 | .440 | 41.51% | .527 |

these were three-seed, 12,000-step arms after the original common pretraining.
the stronger transition coincided with worse early decoding. this was not an
accuracy-matched dose-response demonstration. local source:
`outputs/temporal_energy_dose/results.json`.

### 2. rewarding accuracy throughout the held stimulus

with fixed sst timing and the original predictor's early readout, current ce
was averaged over all 40 positive samples while total task weight stayed fixed.

| activity alpha | early center | late center | late broad flanks | mean on-period decoding | cycle activity |
|---|---:|---:|---:|---:|---:|
| 0 | 1.432 | 1.432 | 1.002 | 50.99% | 1.065 |
| .07 | 1.376 | 1.027 | .869 | 41.22% | .901 |
| .30 | 1.276 | .376 | .657 | 25.73% | .672 |
| .70 | 1.146 | .070 | .461 | 16.27% | .536 |

sustained accuracy reward opposed suppression most at low activity weight;
it did not eliminate suppression at all weights. these fixed-timing results
must not be described as the current peak model. local source:
`outputs/temporal_sustained_accuracy/results.json`.

### 3. independently learned timing, with early accuracy still protected

the next stage made both route kinetics learnable, moved sst gain after
rectification, and learned excitatory gain. an early ce ceiling used a
quadratic penalty, eventually with coefficient 10000. the initial states
were derived from historical temporal common pretrains, unlike today's
static-source peak initial states.

| condition | early center | early flanks | late center | late flanks | cycle activity |
|---|---:|---:|---:|---:|---:|
| joint, equal initial taus | 1.805 | .895 | .281 | .406 | .599 |
| joint, sst initially 10× faster | 1.796 | .904 | .363 | .441 | .615 |
| accuracy only | 1.841 | 1.003 | 1.871 | 1.003 | 1.087 |
| energy only | .025 | .209 | .025 | .209 | .347 |
| sustained accuracy + energy | 1.125 | 1.000 | 1.466 | .999 | 1.012 |

all five conditions had three 24,000-step fits. the main and reversed-start
conditions learned faster excitation and slower sst and met their early ce
ceilings. removing energy pressure preserved enhancement; removing task
pressure caused immediate suppression. sustained accuracy plus its mean-ce
ceiling preserved enhancement at alpha .3. that result differs from the
earlier fixed-timing sustained experiment because its setup differs.

freezing sst recruitment after .2 increased probe-cycle activity: continued
recruitment saved 31.82–33.77% relative to that intervention. swapping or
equalizing fitted kinetics violated the early ce ceiling in the tested
fixed-weight comparisons. these supported the utility of fitted timing but
still depended on explicit early priority. strict gates requiring a late raw
central trough did not pass. historical records are in
`figures/temporal_v11/results.json`, retained locally outside the package.

### 4. peak decoding and freely learned timing

replacing the early readout with the label-free peak rule removed that explicit
priority. equal initial kinetics initially used the quadratic ce penalty;
the dual method was then used for the main initialization experiments.

| peak experiment | early center | late center | late flanks | decoding | cycle activity |
|---|---:|---:|---:|---:|---:|
| equal taus, quadratic penalty | .945 | 1.653 | .970 | 50.13% | 1.023 |
| equal taus, dual | .864 | 1.752 | .949 | 49.29% | 1.003 |
| sst initially 10× faster, dual | .864 | 1.752 | .949 | 49.28% | 1.003 |
| excitation initially 10× faster, dual | 1.791 | .099 | .327 | 49.19% | .545 |
| same 10× start, no activity penalty | 1.854 | 1.855 | 1.003 | 53.65% | 1.131 |

equal/reversed starts learned fast sst and slow excitation, selected a snapshot
near the end of the stimulus, and developed late enhancement. the faster
excitation start found the early-enhancement/late-suppression strategy and
selected .1. thus the peak rule can select different times; an early response
is not imposed by an early loss window.

the matched 10× joint/no-energy comparison reduced cycle activity by about
51.8% with energy pressure. the no-energy network learned negligible sst gain
despite slower sst kinetics. this supports an energy-pressure contribution,
rather than treating slow sst alone as sufficient. however, decoding also
declined from 53.65% to 49.19%; it was not an accuracy-matched comparison.
the matched no-energy control was run for the 10× start, not separately for
the current 2×/3× starts.

### 5. smaller initial timing differences and hard caps

all rows below used three seeds, 24,000 steps, peak decoding, dual ce, and
alpha .3. ratios refer to initial tau_S/tau_E unless a trained cap is stated.

| experiment | early center | early flanks | late center | late flanks | cycle activity | outcome |
|---|---:|---:|---:|---:|---:|---|
| uncapped 3× start | 1.790 | .853 | .099 | .327 | .545 | desired broad-flank-relative transition |
| uncapped 2× start | 1.790 | .853 | .099 | .327 | .545 | same transition |
| repeated uncapped 2× | 1.790 | .853 | .099 | .327 | .545 | same-seed measurements exactly repeated |
| uncapped 1.5× start | .863 | .964 | 1.752 | .949 | 1.003 | fast sst, slow excitation; late enhancement |
| 3× start, trained ratio capped at 5× | 1.525 | .488 | .333 | .250 | .578 | late flanks proportionally more suppressed; desired late shape absent |
| 3× start, trained ratio capped at 20× | 1.731 | .659 | .245 | .298 | .523 | weaker relative center suppression; raw central peak remains |

the 5× cap used symmetric projection in log-time-constant space after each
optimizer step, preserving the geometric mean and allowing either ordering.
the 20× experiment used the same method. next-stimulus accuracy was 76.07%
with the 5× cap and 79.01% with the 20× cap, versus approximately 80.35% for
the uncapped main model. lower activity alone did not mean a better overall
task/activity objective. both cap experiments were abandoned as requested;
the main implementation has no hard ratio projection.

starting at 2× or 3× did not keep the trained ratio at 2–3×: it grew to about
152×. the 1.5× result shows initialization sensitivity under this optimizer,
budget, and model. these experiments do not locate an exact boundary between
basins or prove that no other strategy exists. local records use
`outputs/temporal_peak_dual_e2`, `e3`, `e2_repeat`, `e1p5`,
`cap5_e3`, and `cap20_e3` as the corresponding directory suffixes.

## biological interpretation and limits

fast effective excitation followed by recruited inhibition is a defensible
qualitative circuit prior. previous literature review found direct feedback
examples of early excitation followed by inhibition, including
[zhang et al., 2014](https://doi.org/10.1126/science.1254126) and
[shen et al., 2022](https://doi.org/10.1038/s41467-022-33883-9).
[fişek et al., 2023](https://doi.org/10.1038/s41586-023-06007-6) supports
apical feedback activation, and
[grier et al., 2023](https://doi.org/10.1038/s41467-023-42968-y) reports
heterogeneous fast/slow excitatory input to som neurons. these are qualitative
motivations, not measurements of this model's two time constants. feedback
latencies, synaptic kinetics, recruitment time constants, and dendritic
activation are different quantities.

the network is a rate model, not a spiking network. the rnn has signed
weights/states and is not a dale-compliant implementation. fixed anatomy,
first-order kinetics, response-gain placement, noiseless peak selection,
gradient-based training, and incomplete activity accounting are engineering
approximations. no sharpening target, dampening target, timed switch, or
explicit early-decoding reward is used in the main temporal loss.

the supported finding is conditional: this circuit can learn a useful
early-enhancement/late-relative-suppression strategy under the specified
peak-information and activity objective, from 2× and 3× starts. energy controls
in the closely related 10× experiment support a contribution from activity
pressure. the result does not establish universal convergence, physiological
timing values, exact ce-constraint feasibility, a monotonic dose response for
the current main recipe, or measured metabolic savings.

## package validation

validation completed in the environment listed above:

- all 43 existing circuit tests passed; the test suite was not changed.
- fresh evaluation of all six static endpoints and all six temporal endpoints
  exactly reproduced 33,121 numerical values and seven boolean values in the
  packaged result files. maximum numerical difference was zero.
- all six temporal final checkpoints and their six initial checkpoints are
  byte-identical to their experiment sources. each final completed 24,000
  updates and records peak decoding, dual ce, and uncapped learned kinetics.
- regeneration of the seed-8 initializer for each temporal recipe exactly
  matched all 16 saved state tensors, the configuration, and the references.
  the initial time constants are .5/1 and 1/3/1 as documented.
- each temporal recipe completed a fresh one-update cpu training check at
  seed 8. losses and gradients were finite; both time constants and pathway
  gains updated; the dual multiplier and training state were saved correctly.
- the documented python loading example ran for all three network types,
  including temporal traces and the selected population response.
- the original static response-reproduction command completed successfully,
  including its banked-value checks and the mean plot without archived inputs.
- the static paper figure and both main temporal profiles were regenerated as
  png, svg, and pdf. the static and 2× temporal renderings were visually checked.

these package checks reuse the completed scientific fits; they do not add new
full training experiments or independent evidence about convergence. validation
outputs are local under `outputs/commit_preparation_validation/` and are not
included as another network family.
