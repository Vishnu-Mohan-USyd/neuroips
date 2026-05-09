# V1 L4 to L2/3 Soft-Prior SOM and Recurrence Report

Branch prepared for this report: `v1-soft-prior-som-recurrence-report`

Primary implementation files:

- `genn/v1TwoLayerConfig.h`
- `genn/v1Biology.h`
- `genn/v1TwoLayerModel.cc`
- `tools/validate_full_plasticity.py`

Closest previous report:

- `docs/v1_full_plasticity_system_report.md`

Latest validation family documented here:

- H200 run prefix family: `v1_recur_causal1_*`
- Full trained network: `v1_recur_causal1_full`
- No-learning control: `v1_recur_causal1_control`
- Same-trained-network SOM-output ablation: `v1_recur_causal1_somoff`
- Same-trained-network recurrent-output ablation: `v1_recur_causal1_recoff`

This document records the current state of the reduced GeNN/C++ V1 scaffold
after replacing the old hard orientation feedforward mask with a softer
probabilistic orientation prior, then adding stronger SOM size-tuning and L2/3
recurrent-specificity validation.

## 1. Purpose

The model is a two-layer spiking scaffold for later expectation/top-down
experiments. The immediate requirement is not full anatomical density. The
requirement is a biologically plausible reduced L4 to L2/3 circuit whose
orientation selectivity, inhibitory stabilization, broad/size-dependent
suppression, and recurrent coactive/co-tuned structure can be measured before
adding top-down feedback.

The current branch is therefore a pre-top-down audit branch. It keeps the
accepted hardcoded V1 orientation drive in L4, weakens the L4 to L2/3
orientation prior into a soft probability bias, and validates whether the rest
of the circuit still develops useful L2/3 selectivity without breaking SOM,
PV, recurrent, and rate-safety behavior.

## 2. Biological Scope

The biological target is columnar V1, closer to cat/tree-shrew/macaque-like
orientation-map logic than mouse salt-and-pepper V1. Direct anatomical work in
tree shrew supports local, retinotopic, orientation/axis-biased L4 to L2/3
projection geometry rather than completely random feedforward mixing. The
current model treats this as a graded local prior, not as an all-or-none rule.

The model is still a reduced scaffold. It does not reproduce full cortical
density, conductance-based synapses, dendritic compartments, detailed
interneuron subtypes, LGN spiking afferents, long developmental timescales, or
long-range L2/3 horizontal patch systems. It should be interpreted as a
mechanistic test bed for a two-sheet L4 to L2/3 circuit, not a full V1
reconstruction.

Key biological references that motivate the implemented mechanisms include:

- Mooser, Bosking, and Fitzpatrick, "A morphological basis for orientation
  tuning in primary visual cortex", Nature Neuroscience, 2004. This motivates
  local orientation/axis-biased L4 to L2/3 feedforward structure.
- Yoshimura, Dantzker, and Callaway, "Excitatory cortical neurons form
  fine-scale functional networks", Nature, 2005. This motivates nonrandom local
  subnetworks and shared input structure.
- Ko et al., "Functional specificity of local synaptic connections in
  neocortical networks", Nature, 2011. This motivates preferential connectivity
  among nearby neurons with similar visual responses.
- Cossell et al., "Functional organization of excitatory synaptic strength in
  primary visual cortex", Nature, 2015. This motivates strong-synapse
  enrichment among correlated feature-selective cells.
- Wilson et al., "Division and subtraction by distinct cortical inhibitory
  networks in vivo", Nature, 2012. This motivates distinct PV and SOM
  functional roles.
- Adesnik et al., "A neural circuit for spatial summation in visual cortex",
  Nature, 2012. This motivates SOM/SST involvement in size/surround
  suppression.
- Vogels et al., "Inhibitory plasticity balances excitation and inhibition in
  sensory pathways and memory networks", Science, 2011. This motivates local
  inhibitory plasticity as a stabilizing mechanism.

## 3. Network Architecture

The model has two aligned retinotopic sheets:

- L4: a driven input sheet with hardcoded orientation-tuned simple-cell-like
  drive.
- L2/3: an overlaid cortical sheet with excitatory recurrence plus PV, SOM, and
  VIP inhibitory subtypes.

The sheet is `32 x 32`, so there are `1024` retinotopic sites. Each site is a
scaled hypercolumn-like unit, not a full anatomical hypercolumn.

Population counts:

| Population | Per site | Total |
|---|---:|---:|
| L4E | 16 | 16,384 |
| L4PV | 3 | 3,072 |
| L4SOM | 1 | 1,024 |
| L23E | 16 | 16,384 |
| L23PV | 2 | 2,048 |
| L23SOM | 1 | 1,024 |
| L23VIP | 1 | 1,024 |
| Total | 40 | 40,960 |

L4 therefore contains inhibition, but no L4 VIP population. L2/3 contains
explicit PV, SOM, and VIP populations. The layer-wise E/I ratio is roughly
80/20.

All populations use current-based LIF dynamics in GeNN. The main time step is
`0.1 ms`; the default trial duration is `250 ms`; the measurement settle window
is `50 ms`; and the standard orientation sweep uses `12` orientations.

Subtype LIF intent:

- Excitatory cells: slower regular-spiking point-neuron approximation.
- PV: fast, lower membrane time constant, fast inhibition, local stabilization.
- SOM: slower than PV, broader output, slower inhibitory synapse.
- VIP: disinhibitory population targeting SOM, present structurally but not yet
  driven by top-down input or trained in this branch.

## 4. Hardcoded, Learned, and Emergent Components

Hardcoded by design:

- The L4 orientation map.
- The L4 simple-cell/Gabor-like drive.
- The existence of E/PV/SOM/VIP cell classes.
- The local retinotopic sheet geometry.
- Population sizes, radii, base weights, time constants, and plasticity
  parameter defaults.

Hardcoded as a biological prior, but now softened:

- L4E to L23E feedforward wiring is still orientation-biased, but it is no
  longer a hard same-orientation threshold. The current default
  `V1_FF_ORIENTATION_BIAS_STRENGTH` is `0.3`, with a probability floor of
  `0.15`. This means local cross-orientation input can exist; similar
  orientation simply has higher connection probability.

Learned or plastic:

- L4E to L23E feedforward weights change through STDP.
- L23E to L23E recurrent excitatory weights change through STDP.
- L23PV to L23E inhibitory weights change through homeostatic inhibitory
  plasticity.
- L23SOM to L23E inhibitory weights change through homeostatic inhibitory
  plasticity.

Emergent or measured rather than directly imposed:

- Post-training L2/3 OSI.
- Recurrent strong-synapse enrichment among coactive/co-tuned L23E cells.
- Recurrent contribution to L23E coactivation measured by recurrence-on versus
  recurrence-off context sweeps.
- Size-dependent L23E response profile and large-field suppression.

Not implemented yet:

- Top-down expectation feedback.
- VIP learning.
- Feedback-driven VIP to SOM disinhibition.
- True annular surround stimuli.
- Long-range patchy L2/3 horizontal axons.
- Conductance-based synapses and dendritic compartments.

## 5. Connectivity

L4 local connectivity:

- `L4E -> L4E`
- `L4E -> L4PV`
- `L4E -> L4SOM`
- `L4PV -> L4E`
- `L4PV -> L4PV`
- `L4SOM -> L4E`
- `L4SOM -> L4PV`

L4 to L2/3 feedforward connectivity:

- `L4E -> L23E`: local, soft orientation-biased, plastic.
- `L4E -> L23PV`: local, static, fast feedforward PV recruitment.

L2/3 recurrent and inhibitory connectivity:

- `L23E -> L23E`: sparse, local, distance-dependent, plastic.
- `L23E -> L23PV`: local excitatory recruitment of PV.
- `L23E -> L23SOM`: broader excitatory recruitment of SOM.
- `L23E -> L23VIP`: local excitatory recruitment of VIP.
- `L23PV -> L23E`: local fast inhibition, homeostatically plastic.
- `L23PV -> L23PV`: static PV self-network inhibition.
- `L23SOM -> L23E`: broad, slower inhibition, homeostatically plastic.
- `L23SOM -> L23PV`: broad inhibition of PV.
- `L23SOM -> L23VIP`: broad inhibition of VIP.
- `L23VIP -> L23SOM`: disinhibitory motif, static.

Main radii:

| Projection family | Radius in sites |
|---|---:|
| L4 local | 1 |
| L4E to L23E/PV feedforward | 1 |
| L23 local/PV | 2 |
| L23SOM input | 3 |
| L23SOM output | 6 |

The L23E to L23E recurrent path uses radius `2`, peak connection probability
`0.12`, and distance sigma squared `3.0`. It is not globally random and it is
not dense. The current topology remains local enough that many connected pairs
sample similar orientation domains; this is biologically plausible for local
columnar tissue, but it also means true orthogonal-pair comparisons are still
less powered than in a model with explicit long-range patchy horizontal axons.

## 6. Soft Feedforward Orientation Prior

The previous hard orientation mask was too strong as a biological claim because
it made L4E to L23E connectivity nearly a binary same-orientation filter. The
current branch changes that structure into a probability function.

For each local candidate L4E to L23E edge, the code computes:

- L4 site preferred orientation.
- L2/3 site preferred orientation.
- Circular orientation difference.
- Similarity `0.5 * (1 + cos(2 * delta))`.
- Manhattan distance within the local feedforward radius.

The connection probability is:

```text
biased_similarity = (1 - bias_strength) * 0.5 + bias_strength * similarity
probability = floor + (1 - floor) * biased_similarity
probability -= distance_penalty * manhattan_distance
probability = clamp(probability, floor, 1)
```

Current defaults:

| Parameter | Value |
|---|---:|
| `kOrientationSoftProbabilityFloor` | 0.15 |
| `kOrientationSoftBiasStrength` | 0.3 |
| `kOrientationDistancePenalty` | 0.08 |

At `bias_strength=0`, orientation similarity no longer contributes and the
projection becomes locally retinotopic with a nonzero random-like probability
floor. At `bias_strength=0.3`, similar orientations are favored but
cross-orientation local input remains possible. This is the current compromise:
biologically motivated soft prior, not a hard-coded proof of L2/3 orientation
selectivity.

## 7. Plasticity and Training Protocol

The current run is not a single undifferentiated training sweep. It uses staged
training:

1. Baseline sweep: all plasticity off, full-field orientation sweep.
2. Main training: feedforward STDP, recurrent STDP, and PV/SOM inhibitory
   homeostasis on.
3. Recurrent consolidation: feedforward STDP off; recurrent STDP and inhibitory
   homeostasis on.
4. Recurrent-only consolidation: feedforward and inhibitory plasticity off;
   recurrent STDP on.
5. Post sweep: all plasticity off.
6. SOM context and size tuning sweeps: plasticity off; optional SOM-output
   context ablation applied after the trained weights are captured.
7. Recurrence context sweep: plasticity off; optional L23E to L23E output scale
   applied after the trained weights are captured.

Default training parameters:

| Parameter | Value |
|---|---:|
| Training epochs | 1 |
| Recurrent consolidation epochs | 3 |
| Recurrent-only consolidation epochs | 18 |
| Feedforward STDP `Aplus` | 0.000100 |
| Feedforward STDP `Aminus` | 0.0000875 |
| Feedforward STDP tau | 20 ms |
| Feedforward weight bounds | 0.0005 to 0.020 |
| L23E to L23E STDP `Aplus` | 0.000100 |
| L23E to L23E STDP `Aminus` | 0.000100 |
| L23E to L23E STDP tau | 60 ms |
| L23E to L23E weight bounds | 0.0010 to 0.0100 |
| PV homeostatic eta | 0.000020 |
| PV homeostatic target | 25 Hz |
| PV to E bounds | -0.050 to -0.002 |
| SOM homeostatic eta | 0.000050 |
| SOM homeostatic target | 5 Hz |
| SOM to E bounds | -0.040 to -0.001 |

The current SOM baseline current is:

```text
V1_L23SOM_GATE_NA = 0.18
```

This is an explicit engineering approximation. It is not a literal biological
claim that SOM cells receive a constant current source. It stands in for
omitted background, lateral, deeper-layer, and top-down sources that can keep
SOM cells recruitable. This should be replaced in the next biological
refinement with an explicit source of SOM drive.

## 8. Output and Validation Files

The model writes site-level, cell-level, context, size-tuning, and weight CSVs.
The important current outputs include:

- `*_baseline_l4_sites.csv`
- `*_baseline_l23_sites.csv`
- `*_post_l4_sites.csv`
- `*_post_l23_sites.csv`
- `*_post_l23pv_sites.csv`
- `*_post_l23som_sites.csv`
- `*_post_l23vip_sites.csv`
- `*_l23e_cell_tuning.csv`
- `*_l23e_recurrence_context_tuning.csv`
- `*_som_context_validation.csv`
- `*_size_tuning.csv`
- `*_weights_before.csv`
- `*_weights_after.csv`
- `*_l23ee_weights_before.csv`
- `*_l23ee_weights_after.csv`
- `*_l23pv_to_l23e_weights_before.csv`
- `*_l23pv_to_l23e_weights_after.csv`
- `*_l23som_to_l23e_weights_before.csv`
- `*_l23som_to_l23e_weights_after.csv`
- `*_l23ee_specificity.csv`
- `*_summary.csv`
- `*_summary.txt`

`tools/validate_full_plasticity.py` now validates not just OSI and weight
movement, but also size tuning, SOM ablation, recurrent feature specificity,
cell-level response correlation, strong-synapse enrichment, and recurrence-on
versus recurrence-off contribution.

## 9. Current Validation Results

The latest complete validation with the current soft-prior branch passes all
reported gates.

### 9.1 OSI and Rate Safety

The trained full model produces strong L2/3 orientation selectivity:

| Metric | Value |
|---|---:|
| Full post L2/3 median OSI | 0.745356 |
| No-learning control post L2/3 median OSI | 0.000000 |
| OSI delta | 0.745356 |

Rate safety also passes. L23E remains sparse but bounded; PV and SOM remain
active and bounded.

| Run | Population | Median Hz | Fraction below 1 Hz | p99 Hz | Limit Hz |
|---|---|---:|---:|---:|---:|
| Full | L23E | 0.052083 | 0.940430 | 1.666667 | 100 |
| Full | L23PV | 36.250000 | 0.073242 | 68.045833 | 150 |
| Full | L23SOM | 9.583333 | 0.007812 | 16.666667 | 150 |
| Control | L23E | 0.000000 | 0.956055 | 1.875000 | 100 |
| Control | L23PV | 40.833333 | 0.048828 | 70.320834 | 150 |
| Control | L23SOM | 6.250000 | 0.074219 | 17.500000 | 150 |
| SOM-off | L23E | 0.052083 | 0.940430 | 1.706771 | 100 |
| SOM-off | L23PV | 36.250000 | 0.076172 | 69.070834 | 150 |
| SOM-off | L23SOM | 9.166667 | 0.009766 | 17.083333 | 150 |

Interpretation: the OSI result is real for this scaffold, but the correct claim
is still "plasticity-dependent L2/3 selectivity on a hardcoded L4 orientation
map and soft orientation-biased L4 to L2/3 prior." The model did not discover
orientation-tuned L4 drive from untuned input.

### 9.2 Feedforward, PV, SOM, and Recurrent Weight Plasticity

The validator confirms that recurrent and inhibitory weights changed in the
trained run and remained unchanged in the no-learning control.

| Projection | Active synapses | Changed fraction | p95 abs change | Threshold | Bounds/sign |
|---|---:|---:|---:|---:|---|
| L23E to L23E | 406,530 | 0.051598 | 0.000095 | 0.000090 | pass |
| L23PV to L23E | 758,912 | 0.938301 | 0.007460 | 0.000480 | pass |
| L23SOM to L23E | 2,238,016 | 0.406453 | 0.000678 | 0.000390 | pass |

The no-learning control reports `max_abs_change=0.000000000000` for all three
of these plastic paths. VIP weight-learning files are absent, as expected.

### 9.3 SOM Size Tuning and Broad Suppression

The SOM validation now measures a size-tuning curve rather than relying only on
a center-versus-broad binary comparison. The full model shows the biologically
expected direction: response rises from a tiny stimulus to an intermediate
preferred size, then suppresses for large stimuli.

Latest validator pass:

| Gate | Result |
|---|---|
| Preferred orientation for size test | 105 deg |
| Selected orientations | 90, 105, 120 deg |
| Radii tested | 0.5, 1, 2, 3, 4, 6 sites |
| L23E rates by radius | 0.000000, 5.312500, 24.270833, 0.000000, 0.000000, 0.000000 Hz |
| Peak radius | 2 sites |
| Large-size suppression | 1.000000 |
| L4E suppression | 0.071970 |
| L23E minus L4E suppression delta | 0.928030 |

Same-trained-network SOM-output ablation partially rescues the large-field
suppression:

| Metric | Value |
|---|---:|
| Full L23E size suppression | 1.000000 |
| SOM-off L23E size suppression | 0.490814 |
| SOM causal delta | 0.509186 |
| SOM-off L23E rates by radius | 0.000000, 30.416667, 79.375000, 56.562500, 59.583333, 40.416667 Hz |

Interpretation: the model now captures a U/inverted-size-tuning-like behavior
in the central L23E response: weak response to too-small stimuli, maximal
response at an intermediate aperture, and suppression at larger apertures. The
SOM pathway contributes causally to the large-size suppression. This is closer
to V1 size-tuning biology than the earlier single broad-field BSI gate.

Important caveat: this is still aperture-size tuning, not a true annular
surround protocol. A stronger next validation should test center alone,
surround alone, and center plus annular surround while ablating SOM and PV
separately.

### 9.4 L2/3 Recurrent Specificity

The recurrent audit now goes beyond "weights changed." The model exports
cell-level L23E tuning curves and annotates each active L23E to L23E synapse
with pre/post site, distance, preferred orientation difference, pre/post peak
rate, response correlation, initial weight, final weight, and delta weight.

Orientation-difference specificity gate:

| Metric | Value |
|---|---:|
| Active L23E to L23E rows | 406,530 |
| Low-delta count | 101,632 |
| High-delta count | 101,632 |
| Low-delta range | 0.000000 to 2.046702 deg |
| High-delta range | 8.191690 to 73.550731 deg |
| Low mean delta_w | 0.000003 |
| High mean delta_w | 0.000001 |
| Low mean final weight | 0.004503 |
| High mean final weight | 0.004501 |
| p95 abs delta_w | 0.000095 |

Response-correlation specificity gate:

| Metric | Value |
|---|---:|
| High-corr mean delta_w | 0.000004 |
| Low-corr mean delta_w | 0.000002 |
| High-corr mean final weight | 0.004504 |
| Low-corr mean final weight | 0.004502 |
| Best margin | 0.000001 |

Strong-synapse enrichment gate:

| Metric | Value |
|---|---:|
| Active endpoints analyzed | 19,780 |
| Top 10 percent count | 1,978 |
| Correlation threshold | 0.2 |
| Top fraction with corr > 0.2 | 0.551567 |
| Baseline active-endpoint fraction with corr > 0.2 | 0.322902 |
| Corr odds ratio | 2.904037 |
| Top fraction with corr > 0.2 and low delta | 0.257331 |
| Baseline fraction with corr > 0.2 and low delta | 0.108291 |
| Combined odds ratio | 3.432038 |
| Top weight range | 0.004897 to 0.009958 |

Interpretation: the recurrent path now shows the correct biological direction:
the strongest recurrent synapses are enriched among coactive and co-tuned L23E
pairs. This is closer to Ko/Cossell-style functional specificity than a mere
weight-change check.

The effect size is still modest in absolute weight units. That is expected
given the narrow bounded weights and short training. The claim should be
"recurrent plasticity reinforces coactive/co-tuned structure in a reduced
local circuit", not "the model fully reproduces the development of cortical
recurrent feature-specific assemblies."

### 9.5 Causal Contribution of Recurrence

The latest validation adds a same-trained-network recurrence context ablation.
The trained network is measured with recurrent output scale `1.0` and again
with recurrent output scale `0.0`, using exported L23E cell tuning to compare
pairwise response correlations.

Result:

| Metric | Value |
|---|---:|
| Mapped recurrent pairs | 406,530 |
| Active pairs | 121,585 |
| Focus pairs | 35,369 |
| Full recurrent scale | 1.000000 |
| Recurrence-off scale | 0.000000 |
| Mean corr with recurrence on | 0.106386 |
| Mean corr with recurrence off | 0.088412 |
| Mean corr delta | 0.017973 |
| Fraction corr > 0.2 with recurrence on | 0.175464 |
| Fraction corr > 0.2 with recurrence off | 0.148661 |
| Fraction corr > 0.2 delta | 0.026803 |

Rate/OSI safety under recurrence ablation:

| Metric | Value |
|---|---:|
| Active pairs | 121,585 |
| Mean peak rate with recurrence on | 5.813937 Hz |
| Mean peak rate with recurrence off | 5.415244 Hz |
| Off/on peak ratio | 0.931425 |
| Mean OSI with recurrence on | 0.520390 |
| Mean OSI with recurrence off | 0.452364 |

Interpretation: recurrence contributes positively to coactivation and OSI, but
it is not the only source. Shared L4 input and the smooth orientation map still
explain much of the coactivity. This matches the likely biology better than an
all-or-none story: recurrent L2/3 circuitry amplifies and stabilizes
co-tuned/coactive subnetworks on top of structured feedforward drive.

## 10. What Improved Relative to the Previous Report

The previous full-plasticity report validated OSI, weight movement, basic rate
safety, and a narrow broad-field SOM ablation. This branch adds or changes:

- L4E to L23E orientation structure changed from hard-thresholded orientation
  filtering to a soft probabilistic orientation prior.
- The default soft-prior strength is fixed at `0.3`, based on the sweep showing
  it preserves OSI while weakening the structural shortcut.
- Size-tuning validation now tests multiple aperture radii and requires an
  interior optimum plus large-size suppression.
- SOM causal validation now compares size suppression under full versus
  SOM-output-off conditions.
- Cell-level L23E tuning export was added.
- L23E to L23E specificity export was extended with response correlation and
  pre/post peak rates.
- Strong-synapse enrichment is validated among active recurrent endpoints.
- A recurrence-output-off context run can be passed to the validator through
  `--recoff`.
- Recurrence contribution is now measured causally, not just inferred from
  weight movement.

## 11. Current Biological Comparison

The L4 to L2/3 path is now more defensible. It is local and retinotopic, with a
graded orientation probability bias and nonzero cross-orientation input. That
matches the qualitative biological claim better than the old hard mask:
feedforward anatomy is structured, but not purely same-orientation-only.

The L2/3 recurrent path is now a stronger biological approximation than before.
It is local, sparse, plastic, and enriched among coactive/co-tuned pairs after
training. Strong recurrent synapses are more likely to connect cells whose
responses are correlated. That aligns directionally with Ko et al. and Cossell
et al., while remaining much simpler than real L2/3 recurrence.

The SOM pathway now captures the main functional direction expected from V1
size tuning: an intermediate preferred size and strong suppression at larger
apertures. SOM-output ablation weakens that large-field suppression. The
remaining mismatch is the tonic SOM support current and the lack of a true
annular surround circuit.

The PV pathway is implemented as fast local inhibition with homeostatic
plasticity. It is validated mainly as a stabilizing pathway: weights change,
sign and bounds are preserved, and rates do not run away. This is a useful
functional approximation, but not a detailed PV subtype model.

The VIP pathway is not ready for top-down expectation experiments. VIP exists
structurally and targets SOM, but there is no top-down input, no VIP learning,
and no validated feedback-driven disinhibition yet.

## 12. Cheat Ledger and Limitations

Acceptable by current project assumptions:

- L4 orientation tuning is hardcoded.
- The orientation map is deterministic.
- LIF neurons are point neurons.
- The model is scaled down from real V1 density.

Biological prior, but must be stated honestly:

- L4E to L23E connectivity is still orientation-biased. It is soft and
  probabilistic, but not discovered from fully unbiased wiring.

Engineering approximations:

- `V1_L23SOM_GATE_NA=0.18` is tonic SOM support for missing biological drive.
- Static VIP-to-SOM motif exists before top-down feedback is implemented.
- Current source-based synapses omit conductance and dendritic targeting.
- Short training does not match developmental timescales.

Not yet validated:

- True annular surround suppression.
- Separate PV versus SOM causal roles across center, surround, and center-plus-
  surround stimuli.
- VIP-mediated expectation/top-down effects.
- Distance-bin shuffled/null controls for recurrent specificity.
- Multi-seed robustness of the full soft-prior/SOM/recurrent validation suite.
- Long-range patchy horizontal L2/3 axons.

## 13. Commands for Reproducing the Validation

Build/run commands are executed on the H200 pod, where `/scratch` is mounted.
The local workstation used to prepare this report may not have `/scratch`.

Full run:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_recur_causal1_full \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

No-learning control:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_recur_causal1_control \
V1_STDP_APLUS=0 \
V1_STDP_AMINUS=0 \
V1_L23EE_STDP_ENABLE=0 \
V1_L23PV_HOMEO_ENABLE=0 \
V1_L23SOM_HOMEO_ENABLE=0 \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

Same-trained-network SOM-output ablation:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_recur_causal1_somoff \
V1_L23SOM_CONTEXT_OUTPUT_SCALE=0 \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

Same-trained-network recurrence-output ablation:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_recur_causal1_recoff \
V1_L23EE_CONTEXT_OUTPUT_SCALE=0 \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh -f v1TwoLayerModel.cc
```

Validator:

```bash
cd /scratch/proj/v1_snn_l4_l23
python3 tools/validate_full_plasticity.py \
  --genn-dir /scratch/proj/v1_snn_l4_l23/genn \
  --full v1_recur_causal1_full \
  --control v1_recur_causal1_control \
  --somoff v1_recur_causal1_somoff \
  --recoff v1_recur_causal1_recoff
```

Local sanity checks used before committing this branch:

```bash
python3 -m py_compile tools/validate_full_plasticity.py
git diff --check -- genn/v1TwoLayerModel.cc tools/validate_full_plasticity.py genn/v1Biology.h docs/v1_soft_prior_som_recurrence_report.md README.md
```

## 14. Bottom Line

The current model is a better pre-top-down scaffold than the earlier hard-mask
version. It preserves strong post-training L2/3 OSI with a softer feedforward
orientation prior, shows SOM-dependent size suppression with an intermediate
preferred aperture, and shows recurrent enrichment among coactive/co-tuned
L23E cells with a measured causal recurrent contribution.

The correct scientific claim is narrow but useful: this is a reduced,
orientation-map V1 L4 to L2/3 scaffold in which plasticity sharpens L2/3
selectivity and reinforces coactive/co-tuned recurrent structure under a soft
biological feedforward prior. It is not yet a fully self-organizing V1
microcircuit, and it is not yet ready for biological claims about top-down
expectation until VIP/feedback and stronger surround/null validations are added.
