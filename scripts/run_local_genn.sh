#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_local_genn.sh [run-name] [extra genn-buildmodel args...]

Build/run the GeNN V1 C++ model using the local GeNN checkout in
.local_genn/genn and write generated files plus V1 CSV outputs under .runs.

Environment overrides:
  GENN_DIR       GeNN checkout path, default: <repo>/.local_genn/genn
  CUDA_PATH      CUDA install path, default: $CUDA_HOME or /usr/local/cuda
  LIBFFI_PREFIX  libffi prefix, default: $CONDA_PREFIX or $HOME/miniconda3
  V1_SHEET_SIDE  Sheet side compiled into the model, default: 32
  CXXFLAGS       Extra compiler flags appended after -DV1_SHEET_SIDE
  RUN_ROOT       Output root, default: <repo>/.runs

Example:
  V1_SHEET_SIDE=16 V1_TRAINING_EPOCHS=0 scripts/run_local_genn.sh smoke16
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${1:-local_genn_$(date +%Y%m%dT%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi

genn_dir="${GENN_DIR:-$repo_root/.local_genn/genn}"
cuda_path="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
libffi_prefix="${LIBFFI_PREFIX:-${CONDA_PREFIX:-$HOME/miniconda3}}"
sheet_side="${V1_SHEET_SIDE:-32}"
run_root="${RUN_ROOT:-$repo_root/.runs}"
run_dir="$run_root/$run_name"
model_src="$repo_root/genn/v1TwoLayerModel.cc"
build_script="$genn_dir/bin/genn-buildmodel.sh"

if [[ ! -x "$build_script" ]]; then
  echo "Missing executable GeNN build script: $build_script" >&2
  exit 2
fi
if [[ ! -d "$cuda_path" ]]; then
  echo "Missing CUDA path: $cuda_path" >&2
  exit 2
fi
if [[ ! -f "$libffi_prefix/lib/pkgconfig/libffi.pc" ]]; then
  echo "Missing libffi pkg-config metadata: $libffi_prefix/lib/pkgconfig/libffi.pc" >&2
  exit 2
fi

mkdir -p "$run_dir"

export CUDA_PATH="$cuda_path"
export PATH="$CUDA_PATH/bin:$PATH"
export PKG_CONFIG_PATH="$libffi_prefix/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$libffi_prefix/lib:${LD_LIBRARY_PATH:-}"
export CXXFLAGS="-DV1_SHEET_SIDE=$sheet_side ${CXXFLAGS:-}"
export V1_OUTPUT_PREFIX="${V1_OUTPUT_PREFIX:-$run_dir/$run_name}"

cd "$run_dir"

echo "run_name=$run_name"
echo "run_dir=$run_dir"
echo "GENN_DIR=$genn_dir"
echo "CUDA_PATH=$CUDA_PATH"
echo "LIBFFI_PREFIX=$libffi_prefix"
echo "CXXFLAGS=$CXXFLAGS"
echo "V1_OUTPUT_PREFIX=$V1_OUTPUT_PREFIX"
echo "model_src=$model_src"

"$build_script" -f "$@" "$model_src"
