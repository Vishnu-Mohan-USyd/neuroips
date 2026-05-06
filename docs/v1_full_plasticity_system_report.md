# V1 L4 to L2/3 Full-Plasticity SNN Report

Branch: `neuroips-v1-full-plasticity-report`

Validated base commit: `560773c` (`Validate full-plasticity V1 SNN`)

Primary implementation files:

- `genn/v1TwoLayerConfig.h`
- `genn/v1TwoLayerModel.cc`
- `genn/v1Biology.h`
- `genn/v1GenerateL4Drive.cc`
- `tools/validate_full_plasticity.py`
- `docs/full_plasticity_research.md`

Primary H200 validation artifacts:

- `/scratch/proj/v1_snn_l4_l23/genn/v1_fp_pass_full_*`
- `/scratch/proj/v1_snn_l4_l23/genn/v1_fp_pass_control_*`
- `/scratch/proj/v1_snn_l4_l23/genn/v1_fp_pass_somoff_*`
- `/scratch/proj/v1_snn_l4_l23/genn/validation_fp_pass_validator.log`
- `/scratch/proj/v1_snn_l4_l23/genn/validation_fp_pass_validator.status`

## 1. Goal

This branch implements and validates a scaled two-layer spiking model of V1
feedforward processing from layer 4 to layer 2/3. The model is intended to be a
biologically inspired, H200-runnable starting point, not a full-density primate
V1 reconstruction.

The user-level target was:

- Two overlaid cortical sheets corresponding to L4 and L2/3.
- A 32 x 32 hypercolumn-like sheet layout.
- Orientation-selective L4 drive based on Gabor-like tuning.
- L2/3 topographic feedforward input from L4.
- Explicit inhibitory populations, including PV, SOM, and VIP.
- Recurrent excitatory plasticity.
- Inhibitory plasticity and E/I homeostatic stabilization.
- SOM-mediated broad suppression validated causally.
- OSI emergence validated before and after training.
- No top-down or VIP learning in this branch.

## 2. Scope And Non-Scope

In scope:

- GeNN/C++ implementation.
- LIF neurons with excitatory and inhibitory subtype parameters.
- L4 and L2/3 E/PV/SOM/VIP population structure.
- Orientation-biased plastic `L4E -> L23E` feedforward projection.
- Sparse local plastic `L23E -> L23E` recurrent projection.
- Trace-based homeostatic inhibitory plasticity on `L23PV -> L23E` and
  `L23SOM -> L23E`.
- SOM broad-output geometry and post-training context-only SOM ablation.
- Strict artifact validator over H200 CSV outputs.

Out of scope:

- Full primate/cat neuron counts or realistic density.
- Long-range multi-hypercolumn horizontal axons.
- Six-layer cortical architecture.
- LGN model beyond generated L4 drive.
- VIP/top-down/contextual learning.
- Conductance-based cell models or synapse models.
- Full developmental timescale.

## 3. Network Architecture

### 3.1 Sheet Geometry

The model uses two aligned 32 x 32 sheets:

- L4 sheet: retinotopic/orientation-tuned input sheet.
- L2/3 sheet: parallel retinotopic output/recurrent sheet.

Each lattice site is a scaled hypercolumn-like unit rather than a literal
anatomical hypercolumn. This lets the model preserve topographic and
feature-biased structure while staying tractable on the H200.

Constants:

| Parameter | Value |
|---|---:|
| Sheet side | `32` |
| Sites | `1024` |
| Time step | `0.1 ms` |
| Default trial duration | `250 ms` |
| Default measured orientations | `12` |
| Default training epochs | `1` |

### 3.2 Population Counts

Per-site populations:

| Layer | E | PV | SOM | VIP | Total per site |
|---|---:|---:|---:|---:|---:|
| L4 | 16 | 3 | 1 | 0 | 20 |
| L2/3 | 16 | 2 | 1 | 1 | 20 |

Whole-model populations:

| Population | Count |
|---|---:|
| L4E | 16,384 |
| L4PV | 3,072 |
| L4SOM | 1,024 |
| L23E | 16,384 |
| L23PV | 2,048 |
| L23SOM | 1,024 |
| L23VIP | 1,024 |
| Total | 40,960 |

The per-layer E/I ratio is approximately 80/20. L2/3 inhibition is split into
PV, SOM, and VIP, with PV numerically dominant among inhibitory subtypes.

### 3.3 Neuron Model

All populations use current-based LIF dynamics in GeNN:

- Membrane leak integrated with exponential time constant.
- Absolute refractory period.
- Spike threshold and reset.
- External current variable `Iext` used for stimulus drive and subtype gates.

Subtype parameters:

| Type | C nF | TauM ms | Vrest mV | Vreset mV | Vthresh mV | Refrac ms |
|---|---:|---:|---:|---:|---:|---:|
| Excitatory | 0.25 | 20 | -65 | -60 | -50 | 2.0 |
| PV | 0.20 | 8 | -62 | -55 | -45 | 1.0 |
| SOM | 0.22 | 18 | -62 | -56 | -47 | 2.0 |
| VIP | 0.20 | 15 | -62 | -55 | -46 | 1.5 |

Biological mapping:

- PV is fast and high-gain.
- SOM is slower and broader.
- VIP is present as a disinhibitory subtype but not actively learned in this
  branch.

## 4. Connectivity

### 4.1 Radii

| Projection family | Radius |
|---|---:|
| L4 local | `1` |
| L2/3 local | `2` |
| L2/3 SOM input | `3` |
| L2/3 SOM output | `6` |
| L4 to L2/3 feedforward | `1` |

The widened SOM output radius is an explicit biological abstraction: SOM output
is broader than local PV-like inhibition.

### 4.2 L4 Recurrent And Inhibitory Connectivity

Implemented L4 projections:

- `L4E -> L4E`
- `L4E -> L4PV`
- `L4E -> L4SOM`
- `L4PV -> L4E`
- `L4PV -> L4PV`
- `L4SOM -> L4E`
- `L4SOM -> L4PV`

L4 inhibition is static in this branch. L4 is not the main plastic target; it is
the oriented input layer.

### 4.3 L4 to L2/3 Feedforward Connectivity

Implemented feedforward projections:

- `L4E -> L23E`
- `L4E -> L23PV`

`L4E -> L23E` is plastic and orientation-biased. Its sparse connectivity uses a
local retinotopic neighborhood plus an orientation similarity threshold. This is
the one place where orientation tuning is hardcoded by design, matching the
project constraint that V1 orientation tuning can be hardcoded while the rest
should be learned or emergent where possible.

`L4E -> L23PV` is static and local, giving PV fast feedforward recruitment.

No `L4E -> L23SOM` projection is included in the final branch. A probe showed
that adding it with the correct SOM geometry rescued SOM activity but did not
pass the causal broad-suppression gate. SOM recruitment is instead supported by
recurrent L2/3 excitation plus a small default SOM baseline current.

### 4.4 L2/3 Recurrent Excitatory Connectivity

Implemented:

- `L23E -> L23E`

This projection uses sparse distance-dependent local connectivity, not dense
all-to-all local connectivity and not hardcoded orientation-pruned recurrence.

Key defaults:

- Radius: `2`.
- Peak probability: `0.12`.
- Distance sigma squared: `3.0`.
- Mean indegree from local deterministic sparse rule: approximately `24.8`.
- STDP bounds: `[0.0010, 0.0100]`.
- Initial weight: `0.0045`.

Biological mapping:

- Local L2/3 recurrence is sparse and distance-dependent.
- Feature-biased strengthening should arise from activity and plasticity, not
  from assigning orientation tags to recurrent synapses.
- This does not model long-range horizontal/patchy axons.

### 4.5 L2/3 Inhibitory Connectivity

Implemented:

- `L23E -> L23PV`
- `L23E -> L23SOM`
- `L23E -> L23VIP`
- `L23PV -> L23E`
- `L23PV -> L23PV`
- `L23SOM -> L23E`
- `L23SOM -> L23PV`
- `L23SOM -> L23VIP`
- `L23VIP -> L23SOM`

Subtype roles:

- PV: local, fast, perisomatic-like stabilization.
- SOM: broader, slower, broad-field/dendritic suppression abstraction.
- VIP: static disinhibitory motif through `VIP -> SOM`; no learning or top-down
  control is implemented.

## 5. Plasticity And Homeostasis

### 5.1 Feedforward Excitatory STDP

Projection:

- `L4E -> L23E`

Default parameters:

- `Aplus = 0.0001`
- `Aminus = 0.0000875`
- `tauPlus = 20 ms`
- `tauMinus = 20 ms`
- Bounds: `[0.0005, 0.020]`

Role:

- Main driver of L2/3 OSI emergence.
- Preserves L4 tuning while letting L2/3 response selectivity change after
  training.

### 5.2 Recurrent L2/3 Excitatory STDP

Projection:

- `L23E -> L23E`

Default parameters:

- Enabled by default.
- `Aplus = 0.000100`
- `Aminus = 0.000110`
- Slight LTD bias.
- `tauPlus = 20 ms`
- `tauMinus = 20 ms`
- Bounds: `[0.0010, 0.0100]`

Validation result:

- Recurrent plasticity is engaged but bounded.
- Control run leaves weights unchanged.

Biological comparison:

- The tens-of-ms timing window matches canonical cortical STDP timescales.
- The LTD bias is a stabilizing approximation for recurrent E plasticity.
- The model validates engagement, not full self-organization of long-range
  assemblies.

### 5.3 Homeostatic Inhibitory Plasticity

Projection families:

- `L23PV -> L23E`
- `L23SOM -> L23E`

Rule:

- Trace-based inhibitory plasticity.
- Pre and post traces decay with `20 ms` time constant.
- Inhibitory weights remain inhibitory by bounded clipping.
- High postsynaptic activity increases inhibitory magnitude; low activity can
  weaken it.

Default parameters:

| Parameter | PV -> E | SOM -> E |
|---|---:|---:|
| Enabled | yes | yes |
| Eta | `0.000020` | `0.000020` |
| Target Hz | `25.0` | `15.0` |
| Bounds | `[-0.0500, -0.0020]` | `[-0.0400, -0.0010]` |

Biological comparison:

- This follows the functional role of inhibitory plasticity in maintaining E/I
  balance.
- It is a simplified local rule, not a full model of interneuron-specific
  developmental plasticity or neuromodulation.

### 5.4 SOM Baseline And Context Ablation

Default:

- `V1_L23SOM_GATE_NA = 0.18`

Reason:

- Full STDP training depressed L2/3 excitation enough to silence SOM when no
  independent baseline was present.
- A small SOM baseline was chosen as an abstraction for omitted background,
  lateral, and deeper-layer drive. It was preferred over adding
  `L4E -> L23SOM`, because SOM cells are not best represented as purely
  feedforward middle-layer targets.

Context ablation:

- `V1_L23SOM_CONTEXT_OUTPUT_SCALE`
- Default: `1.0`.
- Validation somoff run: `0.0`.

Important implementation detail:

- Context output scaling is applied after baseline, training, and post sweeps.
- This means `somoff` uses the same trained network and only changes SOM output
  during center/broad context validation.

## 6. Experiment Protocol

Default H200 protocol:

- 12 orientations.
- 250 ms per orientation trial.
- 50 ms settle window.
- 1 training epoch.
- Baseline sweep without learning.
- Training sweep with plasticity enabled.
- Post sweep without learning.
- Center-only context validation.
- Broad-field context validation.

The L4 drive is generated internally from Gabor-like orientation selectivity and
spatial position. Context validation uses:

- Center-only aperture radius: `2` sites.
- Broad-field aperture: full field.

The center aperture was reduced from `5` to `2` after debugging showed that the
larger aperture already saturated the central SOM input field and made the
center-vs-broad comparison invalid.

## 7. H200 Run Commands

Full run:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_fp_pass_full \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh v1TwoLayerModel.cc
```

No-learning control:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_fp_pass_control \
V1_STDP_APLUS=0 \
V1_STDP_AMINUS=0 \
V1_L23EE_STDP_ENABLE=0 \
V1_L23PV_HOMEO_ENABLE=0 \
V1_L23SOM_HOMEO_ENABLE=0 \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh v1TwoLayerModel.cc
```

Same-trained-network SOM context ablation:

```bash
cd /scratch/proj/v1_snn_l4_l23/genn
PKG_CONFIG_PATH=/scratch/miniconda3/lib/pkgconfig \
CUDA_PATH=/usr/local/cuda \
LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
V1_OUTPUT_PREFIX=/scratch/proj/v1_snn_l4_l23/genn/v1_fp_pass_somoff \
V1_L23SOM_CONTEXT_OUTPUT_SCALE=0 \
/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh v1TwoLayerModel.cc
```

Strict validator:

```bash
cd /scratch/proj/v1_snn_l4_l23
python3 tools/validate_full_plasticity.py \
  --genn-dir /scratch/proj/v1_snn_l4_l23/genn \
  --full v1_fp_pass_full \
  --control v1_fp_pass_control \
  --somoff v1_fp_pass_somoff
```

Validator status file:

```text
/scratch/proj/v1_snn_l4_l23/genn/validation_fp_pass_validator.status = 0
```

## 8. Validation Gates And Results

### 8.1 OSI Emergence

Gate:

- Full post L2/3 median OSI must be at least `0.70`.
- Full post L2/3 median OSI must exceed no-learning control by at least `0.10`.

Result:

| Metric | Value |
|---|---:|
| Full post L2/3 median OSI | `0.745356` |
| Control post L2/3 median OSI | `0.000000` |
| Delta | `0.745356` |

Interpretation:

- L2/3 selectivity emerged only when plasticity was enabled.
- This is stronger and faster than biological development; it is a computational
  validation of mechanism engagement, not a developmental timescale match.

### 8.2 Feedforward Plasticity

Validated through:

- OSI emergence.
- Feedforward weight summary.
- No-learning control with feedforward STDP disabled.

Key full-run weight summary:

| Metric | Before | After |
|---|---:|---:|
| `L4E -> L23E` mean | `0.005743` | `0.005163` |
| `L4E -> L23E` max | `0.006000` | `0.006126` |

Interpretation:

- Feedforward weights changed during training.
- The no-learning control left them unchanged.

### 8.3 Recurrent Excitatory Plasticity

Gate:

- Active weights keep positive sign.
- At least 5 percent change by at least 1 percent of allowed range.
- p95 absolute change exceeds threshold.
- Lower/upper bound pileup below 10 percent.
- Control max absolute change is zero.

Result:

| Metric | Value |
|---|---:|
| Active synapses | `406530` |
| Changed fraction | `0.109119` |
| p95 absolute change | `0.000109` |
| Threshold | `0.000090` |
| Lower bound fraction | `0.000000` |
| Upper bound fraction | `0.000000` |
| Control max abs change | `0.000000000000` |

Interpretation:

- Recurrent E plasticity is engaged and bounded.
- This supports the claim that the recurrent E pathway is plastic, not merely
  static decoration.

### 8.4 PV Inhibitory Plasticity

Gate:

- Active weights keep inhibitory sign.
- At least 5 percent change by threshold.
- p95 absolute change exceeds threshold.
- Bound pileup below 10 percent.
- Control unchanged.

Result:

| Metric | Value |
|---|---:|
| Active synapses | `758912` |
| Changed fraction | `0.768706` |
| p95 absolute change | `0.001230` |
| Threshold | `0.000480` |
| Lower bound fraction | `0.000000` |
| Upper bound fraction | `0.000000` |
| Control max abs change | `0.000000000000` |

Interpretation:

- PV inhibition is strongly homeostatically engaged.
- This is consistent with PV as a fast stabilizing gain-control pathway.

### 8.5 SOM Inhibitory Plasticity

Gate:

- Active weights keep inhibitory sign.
- At least 5 percent change by threshold.
- p95 absolute change exceeds threshold.
- Bound pileup below 10 percent.
- Control unchanged.

Result:

| Metric | Value |
|---|---:|
| Active synapses | `2238016` |
| Changed fraction | `0.177072` |
| p95 absolute change | `0.000451` |
| Threshold | `0.000390` |
| Lower bound fraction | `0.000000` |
| Upper bound fraction | `0.000000` |
| Control max abs change | `0.000000000000` |

Interpretation:

- SOM inhibition is plastic and bounded.
- This supports the branch claim that SOM is not just a static inhibitory
  population.

### 8.6 Rate Safety

Gate:

- All rates finite.
- L23E must not run away, measured by p99 upper bound.
- PV/SOM keep median-rate and low-silence checks.

Result:

| Run | Pop | Median Hz | Frac below 1 Hz | p99 Hz | Limit |
|---|---|---:|---:|---:|---:|
| Full | L23E | `0.416667` | `0.682617` | `5.040104` | `100.0` |
| Full | L23PV | `22.916667` | `0.093750` | `61.795833` | `150.0` |
| Full | L23SOM | `19.166667` | `0.000000` | `26.666667` | `150.0` |
| Control | L23E | `0.000000` | `0.730469` | `21.253386` | `100.0` |
| Control | L23PV | `17.083333` | `0.100586` | `59.070834` | `150.0` |
| Control | L23SOM | `27.916667` | `0.014648` | `52.083333` | `150.0` |
| Somoff | L23E | `0.416667` | `0.674805` | `5.000000` | `100.0` |
| Somoff | L23PV | `22.916667` | `0.092773` | `61.700000` | `150.0` |
| Somoff | L23SOM | `19.166667` | `0.000000` | `27.404167` | `150.0` |

Interpretation:

- L23E is sparse. This was accepted because OSI and context responses show
  driven activity, and the anti-runaway gate passes.
- PV and SOM remain active and bounded.

### 8.7 SOM Broad Suppression

Gate:

- Center-preferred L23E response must be at least `5 Hz`.
- Driven center-response orientations are selected by:
  `center L23E >= max(10 Hz, 0.25 * preferred center L23E)`.
- Mean broad suppression index over those driven orientations must be at least
  `0.20`.
- Context-only SOM output ablation must reduce BSI by at least `0.05` absolute
  or at least 50 percent relative.

Result:

| Metric | Value |
|---|---:|
| Preferred center orientation | `120 deg` |
| Center preferred L23E | `59.375 Hz` |
| Driven threshold | `14.843750 Hz` |
| Relevant orientations | `10` |
| Min center SOM over driven orientations | `20.000000 Hz` |
| Min broad SOM over driven orientations | `15.000000 Hz` |
| Full mean BSI | `1.000000` |
| Somoff mean BSI | `0.949126` |
| BSI delta | `0.050874` |

Interpretation:

- Broad-field stimulation strongly suppresses central L23E in the intact model.
- Reducing SOM output after training weakens that suppression enough to pass the
  causal gate.
- The margin is narrow and passes by the absolute-delta criterion, not by a
  large relative reduction.

### 8.8 VIP Exclusion

Gate:

- No VIP weight-learning CSVs can exist.
- VIP rates are reported but not required to show learning.

Result:

| Metric | Value |
|---|---:|
| VIP weight files | none |
| Full baseline L23VIP mean rate | `0.000000` |
| Full post L23VIP mean rate | `0.000000` |
| Control L23VIP mean rate | `0.000000` |
| Somoff L23VIP mean rate | `0.000000` |

Interpretation:

- VIP exists structurally but has no learning or top-down function in this
  branch.

## 9. Biological Comparison

### 9.1 What Matches Biology Reasonably

- Two retinotopically overlaid sheets approximate L4 to L2/3 organization.
- E/I proportions are approximately 80/20.
- PV, SOM, and VIP are separated functionally.
- PV is faster and more local.
- SOM is slower and broader.
- L4 to L2/3 feedforward drive is local and orientation-biased.
- L2/3 recurrent E connectivity is sparse and local, not dense global random.
- Recurrent E plasticity uses bounded STDP with a tens-of-ms window.
- Inhibitory plasticity preserves sign and supports E/I stabilization.
- SOM contribution to broad suppression is tested causally by output ablation.

### 9.2 What Is Simplified

- The model has 40,960 neurons, far below biological densities for even a small
  macaque/cat V1 patch.
- Each lattice site is a scaled computational unit, not a one-to-one
  anatomical hypercolumn.
- The L4 orientation map and Gabor-like drive are hardcoded.
- The model does not include LGN, conductance synapses, dendrites, or detailed
  interneuron subclasses.
- The model does not include long-range horizontal axons or top-down feedback.
- The training timescale is computationally short.
- SOM broad suppression is validated in a center-vs-full-field assay, not a
  full annular surround assay.

### 9.3 Source Mapping

The design is documented against the following biological literature in
`docs/full_plasticity_research.md`:

- Yoshimura, Dantzker, Callaway 2005: local L2/3 connectivity.
- Ko et al. 2011: feature-specific local connectivity.
- Cossell et al. 2015: strong recurrent excitation among similarly responsive
  neurons.
- Song et al. 2005: heavy-tailed cortical E/E synaptic weights.
- Stettler et al. 2002 and Bosking et al. 1997: long-range horizontal
  iso-orientation structure, noted as out of scope here.
- Bi and Poo 1998, Sjostrom et al. 2001, van Rossum et al. 2000, Clopath et al.
  2010: STDP constraints.
- Vogels et al. 2011, D'Amour and Froemke 2015, Vickers et al. 2018, Xue et al.
  2014: inhibitory plasticity and E/I balance.
- Adesnik et al. 2012, Wilson et al. 2012, Kapfer et al. 2007, Ozeki et al.
  2009: SOM/PV functional roles and broad suppression.

## 10. Code Additions And Changes

### 10.1 `genn/v1TwoLayerConfig.h`

Key updates:

- Added explicit L4 and L2/3 inhibitory subtype counts.
- Added L2/3 SOM input and output radii.
- Widened `kL23SOMOutputRadius` to `6`.
- Preserved 32 x 32 sheet structure.

### 10.2 `genn/v1TwoLayerModel.cc`

Key additions:

- `HomeostaticInhibitory` GeNN weight-update snippet.
- `SparseDistancePatch` GeNN sparse connectivity snippet.
- Plastic sparse-distance `L23E -> L23E`.
- Homeostatic `L23PV -> L23E`.
- Homeostatic `L23SOM -> L23E`.
- Static subtype projections among L23E/PV/SOM/VIP.
- SOM context output scale applied after training and before context validation.
- CSV exports for subtype rates, site metrics, context validation, and before/after
  weight families.

Important environment controls:

| Variable | Purpose |
|---|---|
| `V1_OUTPUT_PREFIX` | Output CSV prefix |
| `V1_ORIENTATION_COUNT` | Orientation sweep count |
| `V1_TRIAL_MS` | Trial duration |
| `V1_SETTLE_MS` | Measurement settle time |
| `V1_TRAINING_EPOCHS` | Number of training epochs |
| `V1_STDP_APLUS` | Feedforward STDP LTP amplitude |
| `V1_STDP_AMINUS` | Feedforward STDP LTD amplitude |
| `V1_L23EE_STDP_ENABLE` | Toggle recurrent E STDP |
| `V1_L23EE_STDP_APLUS` | Recurrent E STDP LTP amplitude |
| `V1_L23EE_STDP_AMINUS` | Recurrent E STDP LTD amplitude |
| `V1_L23PV_HOMEO_ENABLE` | Toggle PV inhibitory homeostasis |
| `V1_L23SOM_HOMEO_ENABLE` | Toggle SOM inhibitory homeostasis |
| `V1_L23PV_HOMEO_ETA` | PV homeostasis learning rate |
| `V1_L23SOM_HOMEO_ETA` | SOM homeostasis learning rate |
| `V1_L23PV_HOMEO_TARGET_HZ` | PV homeostasis target |
| `V1_L23SOM_HOMEO_TARGET_HZ` | SOM homeostasis target |
| `V1_L23PV_GATE_NA` | PV baseline current |
| `V1_L23SOM_GATE_NA` | SOM baseline current |
| `V1_L23VIP_GATE_NA` | VIP baseline current |
| `V1_L23SOM_OUTPUT_SCALE` | Global SOM output scale |
| `V1_L23SOM_CONTEXT_OUTPUT_SCALE` | Post-training context-only SOM output scale |

### 10.3 `tools/validate_full_plasticity.py`

The validator loads three prefixes:

- Full plasticity run.
- No-learning control.
- Same-trained-network SOM context ablation.

It checks:

- Summary metrics.
- Context CSVs.
- Post site-rate CSVs.
- Before/after weight CSVs.
- Missing or malformed files.
- Non-finite values.
- OSI gate.
- Weight-change gates.
- Rate-safety gates.
- SOM causal broad-suppression gate.
- VIP learning exclusion.

### 10.4 Documentation And Memory

Added:

- `docs/full_plasticity_research.md`
- `docs/v1_full_plasticity_system_report.md`

Updated:

- `MEMORY.md` with project-specific rules:
  - Use GeNN/C++ by default.
  - If the H200 pod is stopped, use `mygpu resume` and wait.
  - Do not claim SOM validation without causal SOM-output evidence.

## 11. Local Verification Commands

These were run after implementation:

```bash
git diff --check
python -m py_compile tools/validate_full_plasticity.py
PYTHONPATH=src python -m pytest -q
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  genn/v1GenerateL4Drive.cc -o /tmp/v1GenerateL4Drive
```

Results:

- `git diff --check`: clean.
- Validator Python compile: passed.
- Unit tests: `4 passed`.
- L4 drive generator compile: passed.

## 12. Residual Risks

- SOM ablation passes by a narrow absolute BSI delta: `0.050874`.
- L23E activity is sparse; the validator treats this as acceptable because
  driven context responses and OSI pass, and runaway does not occur.
- Baseline L2/3 OSI is `0.000000`; the branch validates post-training emergence,
  not realistic pre-training L2/3 selectivity.
- Recurrent E weight initialization remains simple, not explicitly lognormal.
- Long-range horizontal/contextual V1 circuitry remains unimplemented.
- VIP/top-down learning remains unimplemented by design.

## 13. Recommended Next Branches

1. Add long-range horizontal L2/3 projections across multi-hypercolumn patches.
2. Add annular surround stimuli instead of only center-vs-full-field context.
3. Add neighborhood-level SOM recruitment metrics instead of central-site-only
   context CSVs.
4. Add optional conductance-based synapses or current decomposition for E/I
   balance analysis.
5. Add VIP top-down/disinhibitory gating once feedback inputs exist.
6. Add recurrent E co-tuning enrichment analysis using explicit edge identity
   and site-level preferred orientations.
