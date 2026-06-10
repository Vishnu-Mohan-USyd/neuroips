# Restart checkpoint: L2/3 population-state validation

Date: 2026-06-10 Australia/Sydney

This checkpoint records the current state before machine shutdown/repair. It is
the file to read first when resuming the next session.

## Current local state

Repository:

```text
/home/vishnu/coding_proj/deepsnn_claude
```

Current branch:

```text
v1-l23-population-state-validation
```

Current local HEAD before this checkpoint file:

```text
70dd754 docs: record dev2 l23 population state validation
```

Important recent local commits:

```text
70dd754 docs: record dev2 l23 population state validation
277a664 validate l23 population state geometry
a4003b3 tools: add l23 population state validator
e4dfdc2 docs: clarify dev2 genn path usage
6a856eb docs: checkpoint dev2 video genn setup
852e384 Checkpoint restart state for population metrics
5679cf3 Replace L23EE top-k competition with triplet homeostasis
```

Files added/updated for the latest task:

```text
tools/validate_l23_population_state.py
docs/l23_population_state_validation.md
tests/test_l23_population_state_validator.py
docs/l23_population_state_dev2_validation_report.md
```

## What was accomplished

The task was to move away from exact top-k tile identity as the main biological
target and validate stable distributed L2/3 population-state responses to
natural video.

Implemented a standalone validator that reads existing GeNN
`*_video_site_rates.csv` outputs and measures:

- repeated-frame L2/3E population-vector correlation,
- temporal frame-shuffle controls,
- spatial site-shuffle controls,
- odd/even repeat representational similarity matrix stability,
- held-out-repeat nearest-template decoder transfer,
- Fano/variability diagnostics when frame durations or spike counts are
  available,
- central validation-core cropping with `--sheet-side 40 --core-side 32`.

The validator does not alter the model and does not use HVA/top-down feedback.

## Local verification commands already passed

```bash
python -m py_compile tools/validate_l23_population_state.py
python -m unittest tests.test_l23_population_state_validator
PYTHONPATH=src python -m unittest discover -s tests
git diff --check -- tools/validate_l23_population_state.py docs/l23_population_state_validation.md tests/test_l23_population_state_validator.py
```

Observed test summary:

```text
Ran 4 tests in 0.038s OK
Ran 8 tests in 0.671s OK
git diff --check: pass
```

## dev2 state

The verified `dev2` branch state at the time of validation was:

```text
/scratch/proj/v1_snn_l4_l23/neuroips
branch: v1-l23-population-state-validation
HEAD: 449ef48 validate l23 population state geometry
```

`dev2` was initially unreachable while this checkpoint was being written:

```text
mygpu exec: workload is not ready to stream: pod not found
```

It was re-verified reachable immediately afterward, after the user reported that
the pod was back:

```text
STDOUT_PROBE=1
/scratch
dev2-0-3
DONE
```

The remote repo and metrics artifact were also re-verified:

```text
## v1-l23-population-state-validation
449ef48 validate l23 population state geometry
2e76e5a tools: add l23 population state validator
852e384 Checkpoint restart state for population metrics
5679cf3 Replace L23EE top-k competition with triplet homeostasis
4cdcf8b Clarify final rate homeostasis report

METRICS_JSON_PRESENT=1
-rw-rw-r-- 1 2029910 dgxgroup 7495 Jun  9 15:55 /scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_metrics.json
```

Interpret the empty-output attempt as a transient `mygpu`/pod streaming issue,
not as evidence of missing code or missing artifacts.

If needed after restart:

```bash
MYGPU_WS=dev2 mygpu resume
# wait around 5 minutes
MYGPU_WS=dev2 mygpu connect
```

Use this command prefix for future non-interactive `dev2` commands:

```bash
MYGPU_WS=dev2 mygpu exec env BASH_ENV=/scratch/.bashrc bash -lc '<command>'
```

## dev2 data/build setup

See:

```text
docs/dev2_video_genn_setup_checkpoint.md
```

Known setup:

- Repo: `/scratch/proj/v1_snn_l4_l23/neuroips`
- Data root: `/scratch/proj/v1_snn_l4_l23/data/v1_video`
- GeNN: `/scratch/proj/v1_snn_l4_l23/neuroips/.local_genn/genn`
- KITTI/DAVIS data prepared; BDD100K still manual only.
- Use absolute GeNN path on `dev2`; do not rely on `.local_genn/...` from
  arbitrary working directories.

## Latest validated dev2 run

Run:

```text
v1_l23_popstate4_dev2_20260609T154837Z
```

Artifacts:

```text
Run dir:
/scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z

Run log:
/scratch/proj/v1_snn_l4_l23/logs/v1_l23_popstate4_dev2_20260609T154837Z.log

Validator log:
/scratch/proj/v1_snn_l4_l23/logs/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_validator.log

Validator JSON:
/scratch/proj/v1_snn_l4_l23/runs/v1_l23_popstate4_dev2_20260609T154837Z/v1_l23_popstate4_dev2_20260609T154837Z_l23_population_state_metrics.json
```

Run configuration:

```text
validation_sheet_side=40
validation_core_side=32
video_frame_count=64
video_repeat_count=4
video_feedback_disabled=1
lower_v1_video_consolidation_hva_feedback_enabled=0
video_ff_event_trace_future_frame_used=0
video_ff_event_trace_target_label_used=0
video_ff_event_trace_heldout_frames_used=0
video_ff_event_trace_hva_feedback_enabled=0
video_l23ee_triplet_homeostatic_plasticity_future_frame_used=0
video_l23ee_triplet_homeostatic_plasticity_target_label_used=0
video_l23ee_triplet_homeostatic_plasticity_heldout_frames_used=0
video_l23ee_triplet_homeostatic_plasticity_validation_metric_used=0
final_post_video_l23_median_osi=0.790750
```

Strict population-state validation result:

```text
VALIDATOR_EXIT=0
thresholds_passed=True
repeat_count=4
frame_count=64
site_count=1024
repeat_vector_corr_mean=0.6721468178771056
repeat_frame_shuffle_gap_mean=0.3896025306461446
repeat_site_shuffle_gap_mean=0.6720363896677986
odd_even_rsm_corr=0.9010511711207637
odd_even_rsm_frame_shuffle_gap_mean=0.8952655606094582
odd_even_rsm_site_shuffle_gap_mean=0.89704456052766
heldout_decoder_top1_accuracy=0.75
heldout_decoder_top5_accuracy=0.99609375
heldout_decoder_top1_shuffle_gap_mean=0.7340234375
heldout_decoder_top5_shuffle_gap_mean=0.91640625
fano_mean=0.10837981631534857
fano_early_minus_late=-0.0036030986726683645
```

Thresholds used:

```text
repeat_count>=4
frame_count>=64
repeat_vector_corr_mean>=0.45
repeat_frame_shuffle_gap_mean>=0.20
repeat_site_shuffle_gap_mean>=0.15
odd_even_rsm_corr>=0.50
odd_even_rsm_frame_shuffle_gap_mean>=0.15
odd_even_rsm_site_shuffle_gap_mean>=0.15
heldout_decoder_top1_accuracy>=0.20
heldout_decoder_top5_accuracy>=0.55
heldout_decoder_top1_shuffle_gap_mean>=0.10
heldout_decoder_top5_shuffle_gap_mean>=0.15
```

Full report:

```text
docs/l23_population_state_dev2_validation_report.md
```

## Interpretation to preserve

The lower V1 model now has evidence for stable distributed L2/3 population
states under repeated natural video. This is the biologically appropriate target
for the future higher-area predictor.

Do not describe this as exact same top-5 tile identity being required by
biology. Exact top-k remains a strict diagnostic, not the primary biological
target.

The latest validation still does not implement or validate the higher-area
predictor or top-down feedback. It validates the lower-V1 target representation
that the higher area should later predict.

## Next likely task

The next sensible task is to build/train the higher-area predictor against this
distributed L2/3 population state, using held-out-repeat/future-state metrics:

- predicted-vs-actual future L2/3 population-vector correlation,
- predicted-vs-actual future RSM/geometry preservation,
- downstream decoder transfer from predicted state,
- persistence/time-shuffle/spatial-shuffle controls,
- no direct future-frame, label, or validation-metric leakage.

Raw top-k exact tile overlap can remain diagnostic only.

## Resume checklist

After reboot:

1. Open this file.
2. Verify local branch:

```bash
cd /home/vishnu/coding_proj/deepsnn_claude
git status --short --branch
git log --oneline -5
```

3. If `dev2` is needed:

```bash
MYGPU_WS=dev2 mygpu resume
# wait approximately 5 minutes if it was stopped
MYGPU_WS=dev2 mygpu exec env BASH_ENV=/scratch/.bashrc bash -lc 'cd /scratch/proj/v1_snn_l4_l23/neuroips && git status --short --branch && git log --oneline -3'
```

4. Do not restart from older raw-oracle/top-k framing. Continue from the
   population-state target framing above.
