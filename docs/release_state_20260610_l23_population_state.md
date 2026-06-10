# Release state: L2/3 population-state validation

Date: 2026-06-10 Australia/Sydney

This file is the release index for the current GitHub branch. It points to the
current code state, dev2/H200 usage notes, validation evidence, biological
interpretation, and known limitations. Read it before using this branch for the
next higher-area predictor work.

## Branch purpose

The current branch validates the lower V1 L4 -> L2/3 GeNN/C++ model as a
natural-video-driven, biologically framed source of stable distributed L2/3
population states. The branch deliberately moves the primary biological target
away from exact top-k tile identity and toward population-state stability,
because biological L2/3 responses are sparse and variable at single-cell level
while downstream-readable population geometry can remain reliable.

This branch does not add top-down feedback. It prepares the lower V1 target
representation for later higher-area prediction.

## Current code state

Main implementation:

```text
genn/v1TwoLayerModel.cc
genn/v1TwoLayerConfig.h
genn/v1Biology.h
scripts/run_local_genn.sh
tools/validate_l23_population_state.py
tests/test_l23_population_state_validator.py
```

The current model remains a two-sheet V1 scaffold:

- L4 and L2/3 are aligned retinotopic sheets.
- L4 contains excitatory and PV-like inhibitory populations.
- L2/3 contains excitatory, PV, SOM, and VIP populations.
- VIP remains out of scope/silent for the current pre-feedback validation.
- Natural video is encoded as fixed L4 simple-cell/Gabor-like drive.
- L4 -> L2/3 feedforward, L2/3 recurrent, PV, and SOM mechanisms remain
  active under the validated lower-V1 regime.
- The accepted L2/3 recurrent plasticity path is the triplet/homeostatic
  mechanism documented in `docs/l23ee_triplet_homeostatic_plasticity_report.md`.

Current release docs:

```text
docs/l23_population_state_validation.md
docs/l23_population_state_dev2_validation_report.md
docs/restart_checkpoint_20260610_population_state.md
docs/dev2_video_genn_setup_checkpoint.md
docs/pre_top_down_v1_biology_alignment_report.md
docs/rate_homeostasis_remediation_report.md
docs/l23ee_triplet_homeostatic_plasticity_report.md
```

## Local verification

The population-state validator and Python-side test suite were verified locally
with:

```bash
python -m py_compile tools/validate_l23_population_state.py
python -m unittest tests.test_l23_population_state_validator
PYTHONPATH=src python -m unittest discover -s tests
git diff --check -- tools/validate_l23_population_state.py docs/l23_population_state_validation.md tests/test_l23_population_state_validator.py
```

Observed result:

```text
Ran 4 tests OK
Ran 8 tests OK
git diff --check: pass
```

## dev2 pod state and usage

Primary workspace:

```text
dev2
```

Use these commands after a restart:

```bash
MYGPU_WS=dev2 mygpu resume
# wait around 5 minutes if the workload was stopped
MYGPU_WS=dev2 mygpu connect
```

For non-interactive commands, use:

```bash
MYGPU_WS=dev2 mygpu exec env BASH_ENV=/scratch/.bashrc bash -lc '<command>'
```

Important operational caveats:

- Do not kill unrelated processes on the pod.
- `mygpu exec bash -lc ...` does not source `/scratch/.bashrc`; pass
  `BASH_ENV=/scratch/.bashrc`.
- `/scratch/.bashrc` changes directory to `/scratch`; nested scripts that must
  preserve their caller working directory should use `env -u BASH_ENV`.
- Use the absolute GeNN path on dev2:
  `/scratch/proj/v1_snn_l4_l23/neuroips/.local_genn/genn`.

Current verified dev2 setup is documented in:

```text
docs/dev2_video_genn_setup_checkpoint.md
docs/restart_checkpoint_20260610_population_state.md
```

Known dev2 paths:

```text
Repo:      /scratch/proj/v1_snn_l4_l23/neuroips
Data root: /scratch/proj/v1_snn_l4_l23/data/v1_video
GeNN:      /scratch/proj/v1_snn_l4_l23/neuroips/.local_genn/genn
Runs:      /scratch/proj/v1_snn_l4_l23/runs
Logs:      /scratch/proj/v1_snn_l4_l23/logs
```

The latest successful dev2 run artifacts are:

```text
Run:
v1_l23_popstate4_dev2_20260609T154837Z

Run dir:
/scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z

Run log:
/scratch/proj/v1_snn_l4_l23/logs/v1_l23_popstate4_dev2_20260609T154837Z.log

Validator log:
/scratch/proj/v1_snn_l4_l23/logs/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_validator.log

Validator JSON:
/scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_metrics.json
```

## Dataset state

The dev2 video setup checkpoint records the prepared data:

- KITTI Raw synced drives `0001`, `0002`, and `0020` from `2011_09_26`.
- DAVIS 2017 trainval 480p.
- 32x32 frame and transition manifests.
- Precomputed KITTI L4E drive bins for 32x32 and 40x40 sheet runs.

BDD100K is not automatically downloaded because it requires the official/manual
download path. The setup script writes a manual-download note under the dev2
data root.

## Latest validated lower-V1 run

The latest strict validation used a 40x40 simulated sheet with a central 32x32
validation core:

```text
validation_sheet_side=40
validation_core_enabled=1
validation_core_side=32
validation_core_offset_x_sites=4
validation_core_offset_y_sites=4
video_frame_count=64
video_repeat_count=4
video_l4_drive_scale=0.850000
video_feedback_disabled=1
lower_v1_video_consolidation_hva_feedback_enabled=0
final_post_video_l23_median_osi=0.790750
```

No future-frame, target-label, held-out-frame, validation-metric, or HVA
feedback leakage was used by the event-trace feedforward path or recurrent
triplet/homeostatic path in this run.

## Population-state validation metrics

Strict validator result:

```text
VALIDATOR_EXIT=0
thresholds_passed=True
```

Key metrics:

| Metric | Value | Gate |
| --- | ---: | ---: |
| `repeat_count` | `4` | `>=4` |
| `frame_count` | `64` | `>=64` |
| `site_count` | `1024` | central 32x32 core |
| `repeat_vector_corr_mean` | `0.672147` | `>=0.45` |
| `repeat_frame_shuffle_gap_mean` | `0.389603` | `>=0.20` |
| `repeat_site_shuffle_gap_mean` | `0.672036` | `>=0.15` |
| `odd_even_rsm_corr` | `0.901051` | `>=0.50` |
| `odd_even_rsm_frame_shuffle_gap_mean` | `0.895266` | `>=0.15` |
| `odd_even_rsm_site_shuffle_gap_mean` | `0.897045` | `>=0.15` |
| `heldout_decoder_top1_accuracy` | `0.750000` | `>=0.20` |
| `heldout_decoder_top5_accuracy` | `0.996094` | `>=0.55` |
| `heldout_decoder_top1_shuffle_gap_mean` | `0.734023` | `>=0.10` |
| `heldout_decoder_top5_shuffle_gap_mean` | `0.916406` | `>=0.15` |
| `fano_mean` | `0.108380` | diagnostic |
| `fano_early_minus_late` | `-0.003603` | diagnostic |

The strict report is:

```text
docs/l23_population_state_dev2_validation_report.md
```

## Biological interpretation and scope

The validated claim is narrow:

> The reduced lower-V1 GeNN model produces stable, downstream-readable,
> distributed L2/3 population states under repeated natural-video drive, with
> preserved OSI and no top-down/HVA feedback.

This is the correct pre-feedback target for a later higher-area predictor.

The branch should not claim:

- exact top-k tile identity is the primary biological target,
- a completed higher-area predictor,
- implemented expectation feedback,
- full macaque/cat density realism,
- dendritic compartments or conductance-faithful PV/SOM physiology,
- a complete biological V1 microcircuit.

## Known limitations

- The model is still a reduced point-neuron GeNN/C++ scaffold.
- The validation uses precomputed natural-video L4E drive, not a retinal/LGN
  spiking front end.
- The central 32x32 core is validated inside a 40x40 halo; full-sheet claims
  should not ignore edge effects.
- The Fano value is a diagnostic because this run was not designed as a strict
  blank-to-stimulus variability-quenching assay.
- VIP/top-down expectation feedback remains out of scope for this branch.
- Multi-seed lower-V1 population-state validation has not yet been completed.

## Next task

The next task should train and validate a higher-area predictor against the
distributed L2/3 population state, not against literal exact top-k tile identity
as the primary biology metric. The predictor validation should include:

- predicted-vs-actual future L2/3 population-vector correlation,
- predicted-vs-actual future representational geometry preservation,
- held-out future-state decoder/readout transfer,
- persistence, time-shuffle, spatial-shuffle, and no-learning controls,
- explicit no-leakage checks for future frames, labels, validation metrics, and
  feedback into lower V1.

