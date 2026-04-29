# neuroips — V1 spiking-network model (Phase A + Phase B + grading + STDP-form sweep)

CUDA-accelerated AdEx spiking-neuron model of cat/primate primary visual
cortex (V1), built for studying predictive-coding emergent expectation in a
small, debuggable circuit.  Latest branch:
`phase-a-additive-stdp-fix-2026-04-29` (snapshot of Apr 29 2026; supersedes
the earlier `phase-a-graded-iso-orientation-2026-04-29` branch with the
multiplicative-attractor diagnosis and additive-STDP fix added).

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
- **L4 → L2/3 plasticity**: bounded pair-STDP, τ⁺ = 15 ms,
  τ⁻ = 30 ms, A⁺ = 0.005, A⁻ = 0.003 (default; runtime-overridable via
  `--stdp-a-minus`), w_max = 5.13 nS, w_min = 0.  Two weight-update forms
  selectable via `--stdp-form {multiplicative, additive}` (default
  multiplicative for production / regression-stable; additive recommended
  per the task #15-#17 diagnosis below).
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

## `--stdp-form` (task #15-#17 diagnosis + fix)

Two pair-STDP weight-update rules are implemented; both gated by hard
clipping to `[w_min, w_max]` and both wired through the same trace
dynamics (τ⁺=15 ms, τ⁻=30 ms, A⁺=0.005, A⁻=0.003).  Selectable via
`--stdp-form {multiplicative, additive}` (default `multiplicative` —
production behaviour unchanged; `additive` recommended for new work).

| update step | multiplicative (Gütig form, default) | additive (new, task #17) |
|-------------|--------------------------------------|--------------------------|
| post-spike LTP    | `w += A⁺ · (w_max − w) · x_pre`      | `w += A⁺ · x_pre`        |
| pre-spike LTD     | `w -= A⁻ · w · y_post`               | `w -= A⁻ · y_post`       |

Why the new option exists.  Forensic diagnosis (task #15) showed that the
multiplicative rule converges every Δθ-bin to a single uniform attractor
at A⁺/(A⁺+A⁻)·w_max ≈ 3.21 nS.  After 1000 trials of mixed-orientation
training, the post-trained per-Δθ-bin weight medians agree to within
0.03 nS — i.e., orientation-correlated pre-post pairings (which should
strengthen iso-Δθ synapses and weaken cross-Δθ ones) are washed out by
the multiplicative drift.  Causal proof (task #16 C2): retraining the
strict variant with A⁻ shifted from 0.003 to 0.005 (so A⁺ = A⁻) shifts
the trained median from 3.18 nS to 2.08 nS, in the predicted direction
of A⁺/(A⁺+A⁻)·w_max = 2.57 nS.  The remaining ~0.5 nS gap is consistent
with τ-asymmetry biasing the LTP/LTD event-rate ratio away from 1:1.

The additive rule (task #17) drops the per-weight scaling factors and
relies on the hard clip alone.  The result is a partially bimodal weight
distribution that preserves orientation correlation through training.

| variant          | pre-train OSI | multiplicative post-train OSI | **additive post-train OSI** |
|------------------|---------------|-------------------------------|-----------------------------|
| random           | 0.578         | 0.170                         | **0.612**                   |
| strict           | 0.798         | 0.615                         | **0.826**                   |

Additive STDP raises L2/3 median OSI **above** the connectivity-prior-only
pre-train value in both random and strict — i.e. the rule actually builds
selectivity from training (the goal).  Multiplicative does the opposite:
post-train OSI is below pre-train in every variant tested in task #11
(random / am / sharp / strict / gentle).  Per-cell breakdown for the two
variants tested in task #17:

| metric             | random + mult | random + add | strict + mult | strict + add |
|--------------------|---------------|--------------|---------------|--------------|
| osi_median         | 0.170         | 0.612        | 0.615         | 0.826        |
| frac OSI > 0.2     | 0.439         | 0.728        | 0.973         | 0.962        |
| frac OSI > 0.5     | 0.165         | 0.585        | 0.973         | 0.960        |
| frac OSI > 0.8     | 0.062         | 0.317        | 0.019         | 0.581        |
| L2/3 frac silent   | 0.285         | 0.579        | 0.304         | 0.552        |
| weight median (nS) | 2.20          | 0.98         | 3.18          | 0.94         |
| weight std (nS)    | ~0.4          | 0.83         | ~0.4          | 1.15         |
| frac at zero       | 0             | 0.021        | 0             | 0.065        |
| frac at cap        | 2.4e-4        | 3.5e-3       | 1.1e-5        | 2.4e-2       |

Trade-off: additive roughly doubles the silent fraction (~30 % → ~55 %)
because synapses can hard-clip to zero; the survivors are markedly more
selective (frac OSI > 0.8 jumps 5-30×).  Default remains multiplicative
in the source so existing trained-weight `.bin` files are byte-identical
under regression tests; switch to additive at run time when you want
the better tuning.

Full diagnostic write-up: `docs/v1_stdp_multiplicative_attractor_diagnosis.md`.

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
v1_test --stdp-form {multiplicative,additive} # pair-STDP update form; default multiplicative (prod)
v1_test --stdp-a-minus VAL                    # runtime A_minus override (default 0.003)
v1_test --dump-conn-csr                       # dump L4→L2/3 CSR + initial weights + meta to --out_dir
v1_test --validate-only-v2                    # skip V1/V3/V4/V5 in the train-stdp validation suite
v1_test --save-trained-weights PATH           # persist post-train weights .bin + sibling .json
v1_test --load-trained-weights PATH           # load weights, skip training, run validation only
v1_test --n-train-trials N                    # default 1000  (0 allowed: skips training loop)
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
- `docs/v1_stdp_multiplicative_attractor_diagnosis.md` — full forensic
  diagnosis of the multiplicative-STDP attractor failure (task #15-#17):
  per-Δθ-bin uniform convergence at ≈ 3.21 nS, the C2 (A⁺ = A⁻)
  attractor-shift causal proof, and the additive-STDP resolution.

(The fifth research note `v1_gavornik_bear_protocol.md` describing the
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

# Additive-STDP fix (task #17 — recommended for new work)
./build/v1_test --train-stdp --l4-l23-grading strict --stdp-form additive \
  --save-trained-weights /tmp/phaseA_strict_add/trained_weights.bin \
  --out_dir /tmp/phaseA_strict_add
# -> /tmp/phaseA_strict_add/v1_v2_phaseA_v2_osi.json: osi_median ~ 0.826

# Connectivity CSR dump for offline Python analysis (task #16 C1)
./build/v1_test --l4-l23-grading strict --dump-conn-csr --out_dir /tmp/phaseA_strict_diag
# -> /tmp/phaseA_strict_diag/{row_ptr.bin, col_idx.bin, w_init_nS.bin,
#                              target_ori_per_l23.bin, csr_meta.json}
```

## Status (as of 2026-04-29)

Phase A: **DONE & validated** (V1–V5 all pass, default sharp grading).
Multiplicative pair-STDP attractor diagnosed (task #15) and a working
fix landed as `--stdp-form additive` (task #17): random L2/3 OSI 0.578 →
0.612 (with training) and strict 0.798 → 0.826.  Default left at
`multiplicative` so existing trained-weight `.bin` files remain
byte-identical under regression tests.  Phase B: **negative result**;
predictive-coding signatures fail with current minimal-mechanism set.
Open mechanism question.
