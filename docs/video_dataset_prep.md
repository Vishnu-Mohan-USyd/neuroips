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

This preparation step only stages video frames and one-frame transition
manifests. It does not yet implement `fillL4EDriveFromFrame(...)`, video
feedback projections, or prediction-error learning in GeNN.
