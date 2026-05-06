# Inhibitory Subtype Research And Implementation Plan

## Scope

This branch now contains the research/design record for splitting the generic
`L4I` and `L23I` pools into biologically motivated inhibitory subtypes without
regressing the already validated orientation-selectivity workflow. The concrete
implementation branch is `inhibition-subtypes-implementation`.

Saved generic-inhibition baseline:

- Generic inhibitory pools: `L4I`, `L23I`.
- Full default GeNN run:
  - baseline `L23E` median OSI: `0.492307`
  - post-STDP `L23E` median OSI: `0.846834`
  - zero-learning control post `L23E` median OSI: `0.490047`

Current subtype implementation branch:

- Branch: `inhibition-subtypes-implementation`.
- Populations: `L4E/L4PV/L4SOM` and `L23E/L23PV/L23SOM/L23VIP`.
- Default subtype full run:
  - baseline `L23E` median OSI: `0.141110`
  - post-STDP `L23E` median OSI: `0.826583`
  - `L23E` median OSI delta: `0.685473`
  - plastic weight mean: `0.005743 -> 0.003543`
- Matched zero-training control:
  - post `L23E` median OSI: `0.148311`
  - plastic weight mean unchanged: `0.005743 -> 0.005743`

Debugger-confirmed implementation note: four default training epochs
over-depressed the plastic `L4E -> L23E` pathway and silenced most `L23E`
sites. The validated subtype default is therefore one training epoch with the
same local STDP amplitudes.

## Biological constraints

### PV / Pvalb

Function:
PV interneurons are fast-spiking, perisomatic inhibitory cells. In V1 they are
best treated as feedforward/recurrent gain-control and timing-stabilization
cells rather than as the source of orientation tuning itself.

Connectivity abstraction:

- Receive strong local excitation from nearby E populations.
- Inhibit E strongly and locally.
- Inhibit PV locally.
- Provide weak or absent inhibition to SOM/VIP in the simplified circuit.

Implementation mapping:

- Split generic `L4I` and `L23I` into `L4PV` and `L23PV`.
- Use faster LIF parameters than E and than SOM/VIP.
- Use short local radii and strong negative E-target weights.
- Preserve the existing feedforward inhibition path as mostly `L4E -> L23PV`.

Validation:

- Activating or strengthening PV should reduce E rates broadly with only modest
  change in preferred orientation.
- PV should prevent runaway excitation during STDP.
- PV perturbation should mostly affect gain/mean rate, not create the OSI.

Key sources:

- Atallah BV, Bruns W, Carandini M, Scanziani M. 2012. *Parvalbumin-expressing
  interneurons linearly transform cortical responses to visual stimuli.*
  Neuron. DOI: https://doi.org/10.1016/j.neuron.2011.12.013
- Pfeffer CK, Xue M, He M, Huang ZJ, Scanziani M. 2013. *Inhibition of
  inhibition in visual cortex: the logic of connections between molecularly
  distinct interneurons.* Nature Neuroscience. DOI:
  https://doi.org/10.1038/nn.3446

### SOM / SST

Function:
SOM interneurons, especially Martinotti-like cells, are dendrite-targeting and
support surround/lateral suppression through broader recruitment by recurrent
and horizontal excitation.

Connectivity abstraction:

- Receive recurrent E drive, especially from broader L2/3 E neighborhoods.
- Inhibit E dendritic compartments; in this point-neuron model this maps to a
  slower inhibitory current onto E.
- Strongly inhibit VIP and PV in the Pfeffer-style inhibitory circuit.
- Avoid strong SOM -> SOM self-inhibition in the first approximation.

Implementation mapping:

- Add `L23SOM` as the main dendritic/surround inhibitory subtype.
- Add `L4SOM` only as a small stabilizing pool; L4 should remain PV-heavy.
- Use slower synaptic decay than PV. Broader L2/3 SOM recruitment remains a
  planned center-surround extension; the current implementation uses local
  point-neuron SOM inhibition and should not be overclaimed as a full
  surround-context model.
- Connect `L23E -> L23SOM`; increase its radius relative to `L23E -> L23PV`
  only when center-surround stimuli and validation are added.
- Connect `L23SOM -> L23E` and optionally `L23SOM -> L23PV/VIP`.

Validation:

- Increasing surround/context drive should recruit SOM more than PV.
- SOM ablation/weakening should reduce surround suppression and increase
  recurrent E amplification.
- SOM should sharpen or dampen tuning depending on spatial context, not simply
  produce uniform gain suppression.

Key sources:

- Adesnik H, Bruns W, Taniguchi H, Huang ZJ, Scanziani M. 2012. *A neural
  circuit for spatial summation in visual cortex.* Nature. DOI:
  https://doi.org/10.1038/nature11526
- Keller AJ, Dipoppa M, Roth MM, Caudill MS, Ingrosso A, Miller KD, Scanziani
  M. 2020. *A disinhibitory circuit for contextual modulation in primary visual
  cortex.* Neuron. DOI: https://doi.org/10.1016/j.neuron.2020.11.013

### VIP

Function:
VIP interneurons are primarily disinhibitory. In the simplified V1 model they
should not be implemented as a major direct inhibitory drive onto pyramidal
cells. Their first-order role should be to inhibit SOM, allowing behavioral or
top-down gates to modulate dendritic inhibition.

Connectivity abstraction:

- Receive modulatory/top-down/context input and some local E input.
- Preferentially inhibit SOM.
- Weak or absent direct inhibition to E, PV, and VIP in the first
  implementation.

Implementation mapping:

- Add `L23VIP`; keep `L4VIP` absent initially unless a specific L4 state-control
  mechanism is implemented.
- Add optional external current gates `V1_L23PV_GATE_NA`,
  `V1_L23SOM_GATE_NA`, and `V1_L23VIP_GATE_NA` for validation experiments.
- Connect `L23VIP -> L23SOM` strongly and locally.

Validation:

- Activating VIP should reduce SOM activity and increase L23E responsiveness,
  especially under surround/context conditions.
- VIP activation should not directly sharpen OSI by itself; it should modulate
  the gain/context dependence through SOM disinhibition.
- A zero-VIP control should recover the current PV/SOM-only behavior.

Key sources:

- Pfeffer CK et al. 2013. DOI: https://doi.org/10.1038/nn.3446
- Fu Y, Tucciarone JM, Espinosa JS, Sheng N, Darcy DP, Nicoll RA, Huang ZJ,
  Stryker MP. 2014. *A cortical circuit for gain control by behavioral state.*
  Cell. DOI: https://doi.org/10.1016/j.cell.2014.01.050
- Pi HJ, Hangya B, Kvitsiani D, Sanders JI, Huang ZJ, Kepecs A. 2013.
  *Cortical interneurons that specialize in disinhibitory control.* Nature.
  DOI: https://doi.org/10.1038/nature12676

### Lamp5 / NDNF / Neurogliaform

Function:
Lamp5/NDNF/neurogliaform-like interneurons are sparse, layer-1/distal dendrite
inhibitory cells. They are most relevant when the model includes top-down
feedback onto apical tufts. In the current two-layer point-neuron model they
should be represented cautiously.

Connectivity abstraction:

- Receive top-down/feedback/context drive.
- Provide slow, broad dendritic inhibition.
- Can inhibit nearby interneurons and dendrites; do not model as a fast PV-like
  perisomatic pool.

Implementation mapping:

- Do not add Lamp5 as a required baseline population until feedback/context
  experiments exist.
- If included early, add a small `L23Lamp5`/`L1Lamp5` proxy with slow inhibitory
  synapses to `L23E` and possibly `L23SOM`.
- Gate it with explicit feedback/context input, not with the bottom-up grating
  drive.

Validation:

- Lamp5 activation should mainly suppress distal/context amplification and have
  slower kinetics than PV.
- Without feedback/context drive, Lamp5 should not materially change baseline
  OSI emergence.
- With feedback enabled, Lamp5 should constrain feedback-driven amplification.

Key sources:

- Tremblay R, Lee S, Rudy B. 2016. *GABAergic interneurons in the neocortex:
  from cellular properties to circuits.* Neuron. DOI:
  https://doi.org/10.1016/j.neuron.2016.06.033
- Jiang X et al. 2013. *The organization of two new cortical interneuronal
  circuits.* Nature Neuroscience. DOI: https://doi.org/10.1038/nn.3305
- Cohen-Kashi Malina K, Mohar B, Rappaport AN, Lampl I. 2021. *Bottom-up
  inputs are required for establishment of top-down connectivity onto cortical
  layer 1 neurogliaform cells.* Neuron. DOI:
  https://doi.org/10.1016/j.neuron.2021.07.027

## Implemented first model split

```text
L4:   L4E, L4PV, L4SOM
L2/3: L23E, L23PV, L23SOM, L23VIP
```

Leave Lamp5/NGF as a documented feedback/context extension unless the next task
explicitly adds top-down feedback.

Layer-specific rationale:

- L4 should be PV-heavy and feedforward-stabilized.
- L2/3 should contain PV for gain/timing, SOM for dendritic/surround
  suppression, and VIP for disinhibition of SOM.
- Lamp5 is biologically real but premature without apical-tuft or feedback
  compartments.

## Proportions for the scaled model

The current model has `4E + 1I` per site in each layer. A literal subtype split
cannot preserve 80/20 and represent PV/SOM/VIP at every site with integer
counts.

Two safe implementation options:

1. Preserve current population size approximately.
   Use sparse subtype placement across sites, e.g. PV on every site, SOM/VIP on
   alternating sites. This is harder to implement cleanly with current local
   patch code.

2. Increase per-site scale.
   Use enough neurons per site to preserve 80/20 with integer subtype counts.
   Recommended first scaled subtype model:

```text
L4 per site:   16 L4E, 3 L4PV, 1 L4SOM       = 20 total, 20% I
L2/3 per site: 16 L23E, 2 L23PV, 1 L23SOM,
               1 L23VIP                      = 20 total, 20% I
```

This increases neuron count by about 4x relative to the current model but keeps
the H200 target reasonable. It also keeps the biology interpretable.

## Implementation sequence

1. Add subtype constants and LIF parameter sets.
   Keep `PV` fastest, `SOM/VIP` slightly slower, with slower SOM inhibitory
   synapses onto E.

2. Replace generic `L4I/L23I` populations with subtype populations.
   Do this in one branch only, not in the saved `validated-osi-stdp` branch.

3. Rewire local inhibitory motifs.
   Use this first-pass connection table:

```text
L4E   -> L4PV, L4SOM
L4PV  -> L4E, L4PV
L4SOM -> L4E, L4PV

L4E   -> L23E       existing STDP orientation-biased path
L4E   -> L23PV      feedforward inhibition

L23E   -> L23E, L23PV, L23SOM, L23VIP
L23PV  -> L23E, L23PV
L23SOM -> L23E, L23PV, L23VIP
L23VIP -> L23SOM
```

Avoid initially:

```text
VIP -> E
VIP -> PV
VIP -> VIP
SOM -> SOM
PV  -> SOM
```

These omissions are consistent with the simplified V1 inhibitory logic used by
Pfeffer-style and Keller-style abstractions.

4. Preserve the current OSI/STDP experiment outputs.
   Keep baseline/post sweeps and no-learning controls identical.

5. Add subtype activity outputs.
   Save per-trial rates for PV/SOM/VIP populations and summary ratios.

## Validation plan

The subtype implementation is not acceptable unless all of these pass:

### Regression validations

- Full default OSI/STDP run remains finite.
- Post-training `L23E` median OSI remains above the no-learning control by at
  least `0.05`.
- No-learning control keeps weights unchanged and does not show artificial OSI
  gain.
- Mean L23E firing rate stays finite and nonzero.

### PV validations

- PV strengthening reduces L23E mean rate more than it shifts L23E preferred
  orientation.
- PV weakening increases rate or instability risk.
- PV activity tracks local feedforward/recurrent E drive.

### SOM validations

- A larger surround/context stimulus recruits SOM more than PV.
- SOM weakening reduces surround suppression.
- SOM effects are stronger on recurrent/context amplification than on the
  bottom-up single-center grating response.

### VIP validations

- VIP gate increases L23E response by suppressing SOM.
- VIP gate effect is reduced or absent if `VIP -> SOM` is removed.
- VIP gate should not directly alter weights or hardcode orientation tuning.

### Lamp5/NGF validations

Only run after adding feedback/context:

- Lamp5/NGF activation suppresses feedback/context amplification with slower
  kinetics.
- Lamp5/NGF has minimal effect on the baseline feedforward OSI run when feedback
  drive is absent.

## Success criteria for the next coding branch

The next implementation branch should be considered successful only if it
reports:

- branch name and commit hash,
- exact GeNN build command,
- full OSI/STDP metrics,
- matched no-learning control metrics,
- subtype activity summaries,
- perturbation results for PV/SOM/VIP,
- finite-output checks for all generated CSVs,
- whether `post_l23_median_osi - control_post_l23_median_osi >= 0.05`.

## H200 validation results

Validated on `dev2` with GeNN entrypoint
`/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh`.

Commands:

```bash
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_subtype_full \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

```bash
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_subtype_control \
V1_TRAINING_EPOCHS=0 \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh v1TwoLayerModel.cc
```

Regression results:

```text
full:    baseline_l23_median_osi=0.141110, post_l23_median_osi=0.826583
control: baseline_l23_median_osi=0.141110, post_l23_median_osi=0.148311
delta over control post OSI: 0.678272
full weights mean:    0.005743 -> 0.003543
control weights mean: 0.005743 -> 0.005743
```

Zero-training perturbation checks (`V1_L23*_GATE_NA=0.2`, one gate at a time):

```text
no gate:  L23E mean rate 29.63 Hz, L23PV 94.36 Hz, L23SOM 59.47 Hz, L23VIP 0.40 Hz
PV gate:  L23E mean rate 11.02 Hz, L23PV 101.36 Hz, L23SOM 11.02 Hz, L23VIP 0.37 Hz
SOM gate: L23E mean rate 17.22 Hz, L23PV 61.47 Hz, L23SOM 88.57 Hz, L23VIP 0.00 Hz
VIP gate: L23E mean rate 38.54 Hz, L23PV 119.63 Hz, L23SOM 34.50 Hz, L23VIP 98.81 Hz
```

Interpretation:

- PV and SOM gates suppress L2/3E rate, consistent with inhibitory gain
  control and dendritic inhibitory roles in this point-neuron abstraction.
- VIP gate increases L2/3E rate and reduces SOM activity, consistent with the
  implemented `VIP -> SOM` disinhibitory motif.
- The current SOM validation is a local-gate check only. It does not yet prove
  biological surround suppression, because the model does not include an
  explicit center-surround stimulus or broader SOM recruitment radius.
