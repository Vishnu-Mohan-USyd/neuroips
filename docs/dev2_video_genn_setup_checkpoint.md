# dev2 video + GeNN setup checkpoint

Date: 2026-06-09

This checkpoint records the verified `dev2` H200 setup used for the V1 GeNN
video work. It is intentionally operational: paths, commands, and caveats that
matter after a pod rebuild/restart.

## Remote workspace

- Workspace: `dev2`
- Connect command: `MYGPU_WS=dev2 mygpu connect`
- Exec prefix for non-interactive commands:

```bash
MYGPU_WS=dev2 mygpu exec env BASH_ENV=/scratch/.bashrc bash -lc '<command>'
```

`mygpu exec bash -lc ...` does not source `/scratch/.bashrc`, so `git`,
`python`, and `conda` will not be on `PATH` unless `BASH_ENV` is passed as
above.

Important caveat: `/scratch/.bashrc` changes directory to `/scratch`. When a
nested Bash script must preserve its caller working directory, launch that
nested command with `env -u BASH_ENV`, for example:

```bash
env -u BASH_ENV bash /scratch/proj/v1_snn_l4_l23/neuroips/scripts/run_local_genn.sh dev2_smoke
```

## Repository

- Remote path: `/scratch/proj/v1_snn_l4_l23/neuroips`
- Branch: `v1-biophysical-l23ee-plasticity`
- Verified commit: `852e3847e9c88db0e2f14adb77bb1475308732e8`
- Verified state: clean worktree on `dev2`
- Remote URL is clean HTTPS and does not store the GitHub token.

## Conda/tooling

- Conda root: `/scratch/miniconda3`
- Python: `3.12.13`
- Conda: `26.5.2`
- Packages installed for dataset preparation include `numpy`, `pillow`,
  `pandas`, and `scipy`.

## Data

- Data root: `/scratch/proj/v1_snn_l4_l23/data/v1_video`
- Setup script: `/scratch/proj/v1_snn_l4_l23/scripts/setup_video_data_dev2.sh`
- Setup log: `/scratch/proj/v1_snn_l4_l23/logs/setup_video_data_dev2_20260609T103306Z.log`
- Final data size: `2.9G`

Downloaded/prepared automatically:

- KITTI Raw synced drives `0001`, `0002`, and `0020` from `2011_09_26`.
- DAVIS 2017 trainval 480p.
- 32x32 frame and transition manifests.
- Precomputed KITTI L4E drive bins for 32x32 and 40x40 sheet runs.

Key verified artifacts:

```text
/scratch/proj/v1_snn_l4_l23/data/v1_video/manifests/kitti_raw_image_00_32x32_frames.csv
/scratch/proj/v1_snn_l4_l23/data/v1_video/manifests/kitti_raw_image_00_32x32_transitions.csv
/scratch/proj/v1_snn_l4_l23/data/v1_video/manifests/davis_2017_trainval_480p_32x32_frames.csv
/scratch/proj/v1_snn_l4_l23/data/v1_video/manifests/davis_2017_trainval_480p_32x32_transitions.csv
/scratch/proj/v1_snn_l4_l23/data/v1_video/drives/kitti_raw_image_00_32x32_l4e_drive_32x32_128f_scale1_offset012_clip1.bin
/scratch/proj/v1_snn_l4_l23/data/v1_video/drives/kitti_raw_image_00_32x32_l4e_drive_32x32_256f_scale1_offset012_clip1.bin
/scratch/proj/v1_snn_l4_l23/data/v1_video/drives/kitti_raw_image_00_l4e_drive_40x40_128f_scale1_offset012_clip1.bin
/scratch/proj/v1_snn_l4_l23/data/v1_video/drives/kitti_raw_image_00_l4e_drive_40x40_256f_scale1_offset012_clip1.bin
```

BDD100K full videos were not downloaded by script because they require the
official/manual download path. The setup script writes:

```text
/scratch/proj/v1_snn_l4_l23/data/v1_video/downloads/bdd100k/README_MANUAL_DOWNLOAD.txt
```

Fresh rebuild command on a new `dev2` scratch volume:

```bash
MYGPU_WS=dev2 mygpu exec env BASH_ENV=/scratch/.bashrc bash -lc \
  'bash /scratch/proj/v1_snn_l4_l23/scripts/setup_video_data_dev2.sh'
```

## GeNN

- Installed path: `/scratch/proj/v1_snn_l4_l23/neuroips/.local_genn/genn`
- Remote: `https://github.com/genn-team/genn.git`
- Verified tag: `5.4.0`
- Verified commit: `dd258075263c4b2bcb6607d230add658bcc23127`
- Build script: `/scratch/proj/v1_snn_l4_l23/neuroips/.local_genn/genn/bin/genn-buildmodel.sh`

CUDA compatibility for this pod requires the CUDA compat library path when
building/running generated GeNN CUDA code:

```bash
LDFLAGS="-L/usr/local/cuda/compat"
LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:/scratch/miniconda3/lib:${LD_LIBRARY_PATH:-}"
```

## Verified smoke run

Smoke run directory:

```text
/scratch/proj/v1_snn_l4_l23/runs/dev2_genn_smoke32_compat_20260609T110643Z
```

Smoke log:

```text
/scratch/proj/v1_snn_l4_l23/logs/dev2_genn_smoke32_compat_20260609T110643Z.log
```

The log ends with:

```text
model build complete
```

Required non-empty smoke outputs were verified:

```text
dev2_genn_smoke32_compat_20260609T110643Z_summary.csv
dev2_genn_smoke32_compat_20260609T110643Z_video_frame_summary.csv
dev2_genn_smoke32_compat_20260609T110643Z_video_population_rates.csv
dev2_genn_smoke32_compat_20260609T110643Z_video_site_rates.csv
```

Minimal smoke command pattern:

```bash
MYGPU_WS=dev2 mygpu exec env BASH_ENV=/scratch/.bashrc bash -lc '
repo=/scratch/proj/v1_snn_l4_l23/neuroips
run_root=/scratch/proj/v1_snn_l4_l23/runs
run_name=dev2_smoke
cd "$repo"
env -u BASH_ENV \
  GENN_DIR="$repo/.local_genn/genn" \
  CUDA_PATH=/usr/local/cuda \
  LIBFFI_PREFIX=/scratch/miniconda3 \
  LDFLAGS="-L/usr/local/cuda/compat" \
  LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:/scratch/miniconda3/lib:${LD_LIBRARY_PATH:-}" \
  RUN_ROOT="$run_root" \
  V1_SHEET_SIDE=32 \
  V1_TRAINING_EPOCHS=0 \
  V1_VIDEO_REPLAY_ENABLE=1 \
  V1_VIDEO_DRIVE_BIN=/scratch/proj/v1_snn_l4_l23/data/v1_video/drives/kitti_raw_image_00_32x32_l4e_drive_32x32_128f_scale1_offset012_clip1.bin \
  V1_VIDEO_FRAME_COUNT=128 \
  V1_VIDEO_MAX_FRAMES=2 \
  V1_VIDEO_REPLAY_REPEAT_COUNT=1 \
  V1_VIDEO_FRAME_MS=10 \
  V1_VIDEO_CONSOLIDATION_ENABLE=0 \
  V1_VIDEO_RECURRENT_ONLY_CONSOLIDATION_ENABLE=0 \
  V1_VIDEO_EVENT_TIMING_ENABLE=0 \
  V1_HVA_PREDICTOR_ENABLE=0 \
  V1_OUTPUT_PREFIX="$run_root/$run_name/$run_name" \
  bash "$repo/scripts/run_local_genn.sh" "$run_name"
'
```
