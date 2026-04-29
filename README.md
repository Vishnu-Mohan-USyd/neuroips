# neuroips — V1 spiking-network model (Phase A + Phase B + grading sweep)

CUDA-accelerated AdEx spiking-neuron model of cat/primate primary visual
cortex (V1), built for studying predictive-coding emergent expectation in a
small, debuggable circuit.  Branch:
`phase-a-graded-iso-orientation-2026-04-29` (snapshot of Apr 29 2026).

## Project goal

Reproduce, in a numerically-tractable spiking model, the basic V1 features
that biological L4 + L2/3 circuits exhibit (orientation tuning, phase
invariance, retinotopic RFs) and probe whether minimal-mechanism plasticity
in L2/3 recurrence is sufficient to learn Gavornik-&-Bear-style sequence
expectation signatures.  The target species is **cat / primate** (sharp
tuning, OSI ≈ 0.5–0.8); mouse data is reference only.

## Architecture

### L4 (`adex_v1.cu`)
- 131 072 AdEx cells = **32 × 32 retinotopic hypercolumns × 8 preferred
  orientations × 16 clones** per hypercolumn.
- Each cell is a "simple-cell-like" Gabor-filter neuron with a baked-in
  receptive-field template (`build_gabor_templates_kernel`).  γ jitter
  across clones gives natural OSI variance.
- Driven by a closed-form drifting grating (closed-form Poisson rate per
  cell per timestep) or a direct mode (per-pixel Poisson).
- L4 weights are **static** (Gabor templates are constants of the build).
- Empirical L4 OSI: **median 0.516**, 100 % > 0.2 (`v1_test --measure-l4-osi`).

### L2/3 (`adex_v1.cu`)
- 16 384 AdEx excitatory cells = 32 × 32 hypercolumns × 16 clones.
- **L4 → L2/3 feed-forward**: each L2/3 cell pools from a 3 × 3 hypercolumn
  patch in L4 (≈ 1152 candidates), with sampling probability per candidate
  determined by the **`--l4-l23-grading`** mode (see below).  Weights drawn
  lognormal in EPSP-mV, hard-clipped at 5.0 mV, converted to nS via
  0.974 mV/nS.  Mean fan-in normalized to ≈ 40 partners per L2/3 cell
  (interior, full 3×3 patch).
- **L4 → L2/3 plasticity**: multiplicative pair-STDP, τ⁺ = 15 ms,
  τ⁻ = 30 ms, A⁺ = 0.005, A⁻ = 0.003, w_max = 5.13 nS, w_min = 0.
- **L2/3 → L2/3 recurrent (optional, Phase B)**: distance-dependent
  static recurrence with 4× reciprocity boost, p(d) = 0.12·exp(-d/1.5),
  d_max = 4 hypercolumns, lognormal weights (median 0.3 mV, cap 1.5 mV),
  1 ms conduction delay.
- Optional Gütig multiplicative pair-STDP on L2/3 ↔ L2/3 for Gavornik-style
  ABCD sequence training.

### `--l4-l23-grading` (task #11 sweep)

Each L2/3 cell is assigned a target preferred orientation
`target_ori = (l23_clone_idx % 8)`, giving 8 orientation buckets ×
2 cells per bucket per hypercolumn.  The L4 → L2/3 sampling probability per
candidate is then `p_connect[Δθ_bin]` where Δθ = wrapped half-angle
between the L4 candidate's `ori_idx` and `target_ori`.  Five modes
({0°, 22.5°, 45°, 67.5°, 90°} bins, all normalized to expected fan-in 40):

| mode    | w(0°) | w(22.5°) | w(45°) | w(67.5°) | w(90°) | post-STDP L2/3 median gOSI |
|---------|-------|----------|--------|----------|--------|----------------------------|
| random  | 1.0   | 1.0      | 1.0    | 1.0      | 1.0    | 0.170                      |
| am      | 0.55  | 0.50     | 0.20   | 0.31     | 0.0    | 0.365                      |
| **sharp (default)** | 1.0 | 0.40 | 0.10 | 0.05 | 0.0 | **0.575** (PASS L4 ref 0.516) |
| strict  | 1.0   | 0.20     | 0.0    | 0.0      | 0.0    | 0.615 (PASS L4 ref 0.516)  |
| gentle  | 1.0   | 0.80     | 0.50   | 0.30     | 0.10   | 0.409                      |

`am` reproduces the cat L4-simple → L2/3-complex empirical curve from
[Alonso & Martinez 1998 *Nat Neurosci*]; the others are interpolations /
extrapolations.  `sharp` is the new default in this branch — pushes L2/3
emergent OSI past L4's 0.516 reference while still admitting a unimodal
distribution.  All five modes remain selectable via the flag for ablation
work.

## Phase A validation (V1–V5, post-STDP)

Run via `v1_test --train-stdp [--n-train-trials N] [--save-trained-weights PATH]`,
then the same binary writes the V1–V5 closure to `--out_dir`:

| test | what it measures | pass criterion | last-run result (sharp default) |
|------|------------------|----------------|---------------------------------|
| V1 | sparse-pixel RF locality | ≥ 30 % cells with measurable RF | 49 % single-CC RFs |
| V2 | orientation tuning | L2/3 median gOSI ≥ 0.2 | **0.575** (sharp); 0.170 (random); 0.615 (strict) |
| V3 | phase invariance | L2/3 PI < L4-partner mean PI | 87.5 % of sample cells |
| V4 | phase-generalization decoding (100 ms) | accuracy ≥ 0.7 | 0.94 |
| V5 | firing + weight diagnostics | mean rate > 1 Hz, no runaway | 28.5 % silent at θ = 0 |

## Phase B validation (negative result)

L2/3 → L2/3 STDP trained on Gavornik & Bear ABCD sequence
(A = 0°, B = 90°, C = 45°, D = 135°; 150 ms each, 200 ms ISI within
sequence, 1500 ms ITI between sequences) for 2000 sequences.  Validation
suite (`--train-l23-stdp` then four V_x tests):

| test       | what it tests                      | result                                    |
|------------|------------------------------------|-------------------------------------------|
| V_order    | trained ABCD vs reverse DCBA       | DCBA > ABCD (t = −3.47, p = 0.00051) — **wrong direction** |
| V_timing   | response peak at trained 200 ms ISI | peak at 800 ms ISI (longest tested)      |
| V_omission | A_C_D omission → fill-in?          | A_CD = 0 spikes (suppressed not filled-in) |
| V_lesion   | un-train rev → restored            | sign-flip pass by technicality           |

Bottom line: **predictive-coding signature failure** with current minimal
mechanism set.  Open question for the user: which mechanism (LTD-dominant
STDP, eligibility traces, dendritic gating, etc.) is the missing
ingredient.

## Build

```sh
cd <repo-root>
mkdir build && cd build
cmake ..                    # configures sm_86 + sm_89 (CUDA_ARCHITECTURES "86;89")
make -j v1_test             # main binary
```

## CLI flag overview

```
v1_test --enable-l23 --verify                 # Phase A structure only (no plasticity)
v1_test --enable-l23-recurrent --verify       # Phase B B1 (static recurrence sanity)
v1_test --train-stdp                          # Phase A train + V1-V5 validation
v1_test --train-l23-stdp                      # Phase B train + V_order/timing/omission/lesion
v1_test --measure-l4-osi                      # L4-only OSI distribution sweep (no L2/3)
v1_test --l4-l23-grading {random,am,sharp,strict,gentle}  # L4→L2/3 sampling prior; default sharp
v1_test --save-trained-weights PATH           # persist post-train weights .bin + sibling .json
v1_test --load-trained-weights PATH           # load weights, skip training, run validation only
v1_test --n-train-trials N                    # default 1000
v1_test --train-stim-ms MS                    # default 500
v1_test --seed N                              # default 42
v1_test --out_dir PATH                        # default /tmp
v1_test --help                                # full usage
```

Auxiliary Python plotting helpers in repo root (`plot_*.py`) consume the
JSON outputs from the C++ binary.

## Repository layout

```
adex_v1.cu                         main CUDA source (~9000 lines)
stim_kernels.cuh                   shared stim header
l4_test.cu / stim_test.cu          smaller test binaries
CMakeLists.txt                     build (sm_86 + sm_89)
plot_l4_osi.py                     L4 OSI histogram (task #7)
plot_phaseA.py                     Phase A V1-V5 closure rendering
plot_phaseA_grading.py             grading-sweep aggregation (task #11)
plot_l23_recurrent.py              Phase B B1 sanity panels
plot_l23_verify.py                 L2/3 verify panels
plot_stim_protocol.py              stim-protocol-check renders
run_phaseA_grading_sweep.sh        bash orchestrator for the 5-variant Phase A sweep
docs/                              research notes (cat / primate biology, evidence)
```

## Documentation

- `docs/v1_l4_l23_osi_biology.md` — biological L4 vs L2/3 OSI distributions
  in cat / primate (target species) and reasons for treating mouse data
  as reference only.
- `docs/v1_cat_primate_connectivity_osi.md` — cat/primate V1 like-to-like
  connectivity, OSI, L2/3 connectivity defaults, modelling implications.
- `docs/v1_l4_l23_orientation_match_evidence.md` — direct evidence on
  L4-simple → L2/3 orientation matching (Alonso & Martinez 1998 etc.) used
  to motivate the `am` and `sharp` grading curves in task #11.

(The fourth research note `v1_gavornik_bear_protocol.md` describing the
Gavornik & Bear 2014 ABCD sequence-learning protocol was on /tmp at the
time of training but had been cleaned from disk by commit time; the
protocol's parameters are documented inline in the Phase B section of
`adex_v1.cu` and reproduced in the Phase B description above.)

## Reproducing the headline results

```sh
# Build
mkdir build && cd build && cmake .. && make -j v1_test && cd ..

# L4 OSI distribution baseline (task #7)
./build/v1_test --measure-l4-osi
# -> /tmp/l4_osi.json: median_osi=0.516

# Phase A train + V1-V5 with default sharp grading (task #11/#12)
./build/v1_test --train-stdp \
  --save-trained-weights /tmp/phaseA/trained_weights.bin \
  --out_dir /tmp/phaseA
# -> /tmp/phaseA/v1_v2_phaseA_v2_osi.json: osi_median ~ 0.575

# Phase A grading sweep (5 variants, task #11)
./run_phaseA_grading_sweep.sh
# -> /tmp/phaseA_grading_summary.{json,png}
```

## Status (as of 2026-04-29)

Phase A: **DONE & validated** (V1–V5 all pass, default sharp grading).
Phase B: **negative result**; predictive-coding signatures fail with current
minimal-mechanism set.  Open mechanism question.
