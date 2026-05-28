# Video Dataset Preparation

This branch prepares natural-video data for the next-state prediction branch of
the V1 SNN. Data files are intentionally stored outside the git repository.

Default data root:

```bash
/home/vishnu/datasets/v1_video
```

Override with:

```bash
export V1_VIDEO_DATA_ROOT=/path/to/v1_video
```

## Dataset Order

1. **KITTI Raw first.** Use synced/rectified left grayscale frames at 10 Hz as
   the proof-of-concept natural-video source.
2. **DAVIS held-out probe.** Download DAVIS 2017 train/val 480p for later
   object-centric and occlusion stress tests.
3. **BDD100K later.** Treat BDD100K videos as phase 2 because the full video
   dataset is large and normally downloaded through the official portal.

## Commands

Prepare the KITTI starter subset:

```bash
python3 tools/datasets/prepare_video_datasets.py prepare-kitti \
  --drives 0001 0002 0020
python3 tools/datasets/prepare_video_datasets.py verify
```

This downloads:

- `2011_09_26_calib.zip`
- `2011_09_26_drive_0001_sync.zip`
- `2011_09_26_drive_0002_sync.zip`
- `2011_09_26_drive_0020_sync.zip`

It extracts only `image_00/data/*.png`, preprocesses each frame to a
local-contrast-normalized `32x32` `float32` NumPy array, and writes:

```text
manifests/kitti_raw_image_00_32x32_frames.csv
manifests/kitti_raw_image_00_32x32_transitions.csv
```

Precompute a lower-V1 L4E replay drive for the `40 x 40` validated sheet:

```bash
python3 tools/datasets/prepare_video_datasets.py precompute-l4-drive \
  --manifest /home/vishnu/datasets/v1_video/manifests/kitti_raw_image_00_32x32_frames.csv \
  --output-bin /home/vishnu/datasets/v1_video/drives/kitti_raw_image_00_l4e_drive_40x40_128f.bin \
  --sheet-side 40 \
  --max-frames 128
```

The validated natural-video replay used the calibrated variant:

```text
/home/vishnu/datasets/v1_video/drives/kitti_raw_image_00_l4e_drive_40x40_128f_scale1_offset012_clip1.bin
/scratch/proj/v1_snn_l4_l23/data/kitti_raw_image_00_l4e_drive_40x40_128f_scale1_offset012_clip1.bin
```

The validated HVA predictor milestone used the 256-frame calibrated variant:

```text
/home/vishnu/datasets/v1_video/drives/kitti_raw_image_00_l4e_drive_40x40_256f_scale1_offset012_clip1.bin
/scratch/proj/v1_snn_l4_l23/data/kitti_raw_image_00_l4e_drive_40x40_256f_scale1_offset012_clip1.bin
sha256=b13b1c4f06bd22dca351dad4627dae9b3c2d730c345f94efd792c38dabb0912f
shape=(256, 25600)
size=26214400 bytes
```

Its metadata records:

```text
command=precompute-l4-drive
source_manifest=/home/vishnu/datasets/v1_video/manifests/kitti_raw_image_00_32x32_frames.csv
frame_count=128
sheet_side=40
k_num_l4e=25600
l4e_per_site=16
filter_bank=4 orientations x 4 phases, zero_mean_unit_norm_gabor
drive_scale=1.0
drive_offset=0.12
clip=[0,1]
```

The first validated replay consumed only the first `64` frames and repeated
them three times (`V1_VIDEO_MAX_FRAMES=64`,
`V1_VIDEO_REPLAY_REPEAT_COUNT=3`) with `100 ms` presentation windows.

Prepare DAVIS held-out data:

```bash
python3 tools/datasets/prepare_video_datasets.py prepare-davis
```

This writes:

```text
manifests/davis_2017_trainval_480p_32x32_frames.csv
manifests/davis_2017_trainval_480p_32x32_transitions.csv
```

Create the BDD100K manual-download note:

```bash
python3 tools/datasets/prepare_video_datasets.py write-bdd-note
```

## Local Layout

```text
/home/vishnu/datasets/v1_video/
  downloads/
    kitti_raw/
    davis/
    bdd100k/
  raw/
    kitti_raw/
    davis/
  processed/
    kitti_raw_32x32/
  manifests/
```

## Sources

- KITTI Raw: https://www.cvlibs.net/datasets/kitti/raw_data.php
- KITTI official S3 mirror:
  `https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/`
- BDD100K: https://www.bdd100k.com/ and https://github.com/bdd100k/bdd100k
- DAVIS 2017: https://davischallenge.org/

## Current Scope

This preparation step stages video frames, one-frame transition manifests, and
raw `float32` L4E drive arrays for replay-only lower-V1 validation. The GeNN
model now supports opt-in replay with:

```text
V1_VIDEO_REPLAY_ENABLE=1
V1_VIDEO_DRIVE_BIN=<raw float32 drive>
V1_VIDEO_FRAME_COUNT=<stored frames>
V1_VIDEO_MAX_FRAMES=<optional validation cap>
V1_VIDEO_FRAME_MS=100
V1_VIDEO_REPLAY_REPEAT_COUNT=<repeat count>
```

Replay exports:

```text
<prefix>_video_population_rates.csv
<prefix>_video_site_rates.csv
<prefix>_video_frame_summary.csv
```

The same lower-V1 replay drive now also supports opt-in millisecond
event-aligned timing validation. This is still replay-only, with trained
lower-V1 weights frozen and feedback/top-down disabled:

```text
V1_VIDEO_EVENT_TIMING_ENABLE=1
V1_VIDEO_EVENT_BIN_MS=2
V1_VIDEO_EVENT_REPEAT_COUNT=6
V1_VIDEO_EVENT_PRE_MS=50
V1_VIDEO_EVENT_POST_MS=100
V1_VIDEO_EVENT_GRAY_CURRENT=-1
```

Event-timing exports:

```text
<prefix>_video_event_population_bins.csv
<prefix>_video_event_site_bins.csv
```

The H200 lower-V1 replay pass was:

```text
prefix=v1_natvideo_replay4_sombroad_repeat3
validator_rerun_rc=0
frames=64
repeats=3
feedback_disabled=1
video_training_enabled=0
```

The H200 strict event-aligned timing pass was:

```text
prefix=v1_natvideo_event_timing2_bin2_repeat6
validator_rerun_rc=0
bin_ms=2
event_repeats=6
l4_reference_peak_ms=2
l23pv_peak_ms=8
l23e_peak_ms=28
l23som_peak_ms=28
timing_source=event_minus_gray_control
```

The event-timing validator gates causal latency from matched
`event - gray_control` traces when gray controls exist, so gray-control
fluctuations are not interpreted as event-specific onset. Raw event-only traces
remain diagnostic output.

The first HVA predictor milestone now uses the same lower-V1 replay drive as a
predictor-only task. It remains feedback-free: the sidecar reads causal L23E
tile history and predicts the future top-k L23E tile activity pattern without
feeding current back into V1. The final passing run is documented in
`docs/hva_predictor_system_report.md`.

Full top-down expectation, feedback projections, VIP feedback disinhibition,
and prediction-error learning remain out of scope for this
dataset-preparation/replay layer. The event assay measures timing relative to
direct L4E drive onset, not absolute retina/LGN latency, and it uses selected
site/population bins rather than full cell-level latency maps.
