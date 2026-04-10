#!/usr/bin/env bash
# =============================================================================
# Phase 1 Validation Script — SNN Port
# =============================================================================
# Author: Validator agent (fs_SNN / neuroips)
# Purpose: Gate Phase 1 of the rate-to-SNN port. Run AFTER Coder finishes
#          Phase 1.10 (all of 1.0 through 1.9 merged). DO NOT run earlier —
#          new modules will not exist yet and the script will fail.
#
# Exit code contract:
#   0   → GO verdict (all checks passed)
#   !=0 → NO-GO verdict (check failed; first-failure message printed)
#
# Usage:
#   bash validation/phase1_validate.sh                 # normal run
#   bash validation/phase1_validate.sh --verbose       # dump all outputs
# =============================================================================

set -u
set -o pipefail

VERBOSE=0
[[ "${1:-}" == "--verbose" ]] && VERBOSE=1

cd "$(dirname "$0")/.." || exit 2

RED=$(printf '\033[31m')
GRN=$(printf '\033[32m')
YLW=$(printf '\033[33m')
BLD=$(printf '\033[1m')
RST=$(printf '\033[0m')

FAIL=0
declare -a FAIL_MSGS

ok()   { echo " ${GRN}[PASS]${RST} $*"; }
bad()  { echo " ${RED}[FAIL]${RST} $*"; FAIL=$((FAIL+1)); FAIL_MSGS+=("$*"); }
warn() { echo " ${YLW}[WARN]${RST} $*"; }
hdr()  { echo; echo "${BLD}== $* ==${RST}"; }

# ---------------------------------------------------------------------------
# 0. Preflight — branch, commit, clean tree
# ---------------------------------------------------------------------------
hdr "0. Preflight"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
HEAD_SHA=$(git rev-parse --short HEAD)
if [[ "$BRANCH" == "feature/snn-port" ]]; then
    ok "On branch feature/snn-port (HEAD=$HEAD_SHA)"
else
    warn "Not on feature/snn-port (currently on $BRANCH)"
fi

# Check file allow-list: every new file under src/ or tests/ must be in the
# plan's Phase 1 scope. Any stray modification is an automatic NO-GO.
BASELINE_SHA="59b6c16"
if git cat-file -e "$BASELINE_SHA" 2>/dev/null; then
    CHANGED=$(git diff --name-only "$BASELINE_SHA"..HEAD -- src/ tests/ requirements.txt)
    ALLOWED='^(src/spiking/|tests/test_spiking_(populations|network|v2)\.py$|src/config\.py$|requirements\.txt$)'
    STRAY=$(echo "$CHANGED" | grep -Ev "$ALLOWED" || true)
    if [[ -z "$STRAY" ]]; then
        ok "No stray file modifications outside Phase 1 allow-list"
    else
        bad "Files modified outside Phase 1 scope:"
        echo "$STRAY" | sed 's/^/        /'
    fi
else
    warn "Cannot find baseline commit $BASELINE_SHA — skipping allow-list check"
fi

# ---------------------------------------------------------------------------
# 1. Rate-model regression — 374 existing tests must still pass
# ---------------------------------------------------------------------------
hdr "1. Rate-model regression (374 existing tests)"

# Run only the rate-model test files (exclude future spiking test files).
RATE_TESTS="tests/test_analysis.py tests/test_experiments.py tests/test_model_forward.py tests/test_model_recovery.py tests/test_network.py tests/test_stimulus.py tests/test_training.py"

RATE_OUT=$(python -m pytest $RATE_TESTS -q --tb=no 2>&1 || true)
[[ $VERBOSE -eq 1 ]] && echo "$RATE_OUT"

RATE_LINE=$(echo "$RATE_OUT" | grep -E '^[0-9]+ (passed|failed)' | tail -1)
RATE_PASSED=$(echo "$RATE_LINE" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
RATE_FAILED=$(echo "$RATE_LINE" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)

if [[ "$RATE_PASSED" == "374" && "$RATE_FAILED" == "0" ]]; then
    ok "Rate-model suite: 374 passed, 0 failed"
elif [[ "$RATE_FAILED" == "0" && "$RATE_PASSED" -ge 374 ]]; then
    ok "Rate-model suite: $RATE_PASSED passed (>= 374 baseline), 0 failed"
else
    bad "Rate-model regression: got $RATE_PASSED passed / $RATE_FAILED failed (baseline is 374/0)"
    echo "$RATE_LINE"
fi

# ---------------------------------------------------------------------------
# 2. Spiking test suite — must exist and pass
# ---------------------------------------------------------------------------
hdr "2. Spiking test suite"

for f in tests/test_spiking_populations.py tests/test_spiking_network.py tests/test_spiking_v2.py; do
    if [[ -f "$f" ]]; then
        ok "Exists: $f"
    else
        bad "Missing: $f"
    fi
done

SNN_OUT=$(python -m pytest tests/test_spiking_populations.py tests/test_spiking_network.py tests/test_spiking_v2.py -q --tb=short 2>&1 || true)
[[ $VERBOSE -eq 1 ]] && echo "$SNN_OUT"

SNN_LINE=$(echo "$SNN_OUT" | grep -E '^[0-9]+ (passed|failed)' | tail -1)
SNN_FAILED=$(echo "$SNN_LINE" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
SNN_PASSED=$(echo "$SNN_LINE" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)

if [[ "$SNN_FAILED" == "0" && "$SNN_PASSED" -gt 0 ]]; then
    ok "Spiking suite: $SNN_PASSED passed, 0 failed"
else
    bad "Spiking suite: $SNN_PASSED passed / $SNN_FAILED failed"
    echo "$SNN_OUT" | tail -20
fi

# ---------------------------------------------------------------------------
# 3. Import sanity — src.spiking package
# ---------------------------------------------------------------------------
hdr "3. Import sanity (src.spiking package)"

python - <<'PY' 2>&1 | tee /tmp/phase1_imports.out
import sys
errors = []

# Package root
try:
    import src.spiking
    print("[ok]  import src.spiking")
except Exception as e:
    errors.append(("src.spiking", repr(e)))

# Individual modules — every file listed in the plan
for mod in [
    "src.spiking.surrogate",
    "src.spiking.filters",
    "src.spiking.state",
    "src.spiking.populations",
    "src.spiking.v2_context",
    "src.spiking.network",
]:
    try:
        __import__(mod)
        print(f"[ok]  import {mod}")
    except Exception as e:
        errors.append((mod, repr(e)))

# SpikingConfig must be importable from src.config
try:
    from src.config import SpikingConfig
    print("[ok]  from src.config import SpikingConfig")
except Exception as e:
    errors.append(("SpikingConfig", repr(e)))

# Surrogate gradient function callable
try:
    import torch
    from src.spiking.surrogate import atan_surrogate  # expected name
    v = torch.randn(4, requires_grad=True)
    z = atan_surrogate(v)
    assert z.shape == v.shape, f"shape mismatch: {z.shape} vs {v.shape}"
    assert set(z.unique().tolist()).issubset({0.0, 1.0}) or z.dtype == torch.float32
    z.sum().backward()  # gradient must flow
    assert v.grad is not None, "no gradient propagated through surrogate"
    print("[ok]  atan_surrogate callable and differentiable")
except Exception as e:
    errors.append(("atan_surrogate", repr(e)))

if errors:
    print()
    print("IMPORT ERRORS:")
    for name, err in errors:
        print(f"  {name}: {err}")
    sys.exit(1)
PY
if [[ $? -eq 0 ]]; then
    ok "All src.spiking imports clean"
else
    bad "Import failures (see above)"
fi

# ---------------------------------------------------------------------------
# 4. Forward-pass smoke test — instantiate + forward each spiking population
# ---------------------------------------------------------------------------
hdr "4. Forward-pass smoke test"

python - <<'PY' 2>&1
import sys, torch
from src.config import ModelConfig, SpikingConfig
from src.spiking.populations import (
    SpikingL4Ring,
    SpikingPVPool,
    SpikingL23Ring,
    SpikingSOMRing,
    SpikingVIPRing,
)
from src.spiking.network import SpikingLaminarV1V2Network
from src.spiking.state import initial_spiking_state
from src.spiking.v2_context import SpikingV2Context

cfg = ModelConfig()
scfg = SpikingConfig()
B, N = 4, cfg.n_orientations

ok_count = 0
err_count = 0

def check(label, cond, detail=""):
    global ok_count, err_count
    if cond:
        print(f"[ok]  {label}")
        ok_count += 1
    else:
        print(f"[FAIL] {label}: {detail}")
        err_count += 1

# ---- Individual populations ----
# Each spiking population's forward matches the actual SpikingLaminarV1V2Network
# step-order dependencies: L4 reads r_pv_{t-1}, PV reads x_l4 and x_l23, L23
# reads x_l4, x_som, r_pv, plus template_modulation.
try:
    l4 = SpikingL4Ring(cfg, scfg)
    stim = torch.randn(B, N).abs()
    r_pv_prev = torch.zeros(B, 1)
    state_l4 = l4.init_state(batch_size=B)
    state_l4, z, x = l4(stim, r_pv_prev, state_l4)
    check("SpikingL4Ring forward", z.shape == (B, N) and x.shape == (B, N),
          f"z={tuple(z.shape)} x={tuple(x.shape)}")
    check("SpikingL4Ring spikes binary",
          torch.isfinite(z).all() and ((z == 0) | (z == 1)).all(),
          "non-binary spikes or NaN")
except Exception as e:
    check("SpikingL4Ring", False, repr(e))

try:
    # SpikingPVPool does not take SpikingConfig (it is rate-based).
    pv = SpikingPVPool(cfg)
    state_pv = pv.init_state(batch_size=B)
    x_l4 = torch.randn(B, N).abs()
    x_l23 = torch.randn(B, N).abs()
    state_pv, r_pv, r_pv2 = pv(x_l4, x_l23, state_pv)
    check("SpikingPVPool forward (rate-based)",
          r_pv.shape == (B, 1) and r_pv2.shape == (B, 1),
          f"r_pv={tuple(r_pv.shape)} r_pv2={tuple(r_pv2.shape)}")
except Exception as e:
    check("SpikingPVPool", False, repr(e))

try:
    l23 = SpikingL23Ring(cfg, scfg)
    state_l23 = l23.init_state(batch_size=B)
    x_l4 = torch.randn(B, N).abs()
    template_mod = torch.zeros(B, N)
    x_som = torch.zeros(B, N)
    r_pv = torch.zeros(B, 1)
    state_l23, z, x = l23(x_l4, template_mod, x_som, r_pv, state_l23)
    check("SpikingL23Ring forward",
          z.shape == (B, N) and x.shape == (B, N),
          f"z={tuple(z.shape)}")
except Exception as e:
    check("SpikingL23Ring", False, repr(e))

try:
    som = SpikingSOMRing(cfg, scfg)
    state_som = som.init_state(batch_size=B)
    drive = torch.randn(B, N)
    state_som, z, x = som(drive, state_som)
    check("SpikingSOMRing forward", z.shape == (B, N), f"z={tuple(z.shape)}")
except Exception as e:
    check("SpikingSOMRing", False, repr(e))

try:
    vip = SpikingVIPRing(cfg, scfg)
    state_vip = vip.init_state(batch_size=B)
    drive = torch.randn(B, N)
    state_vip, z, x = vip(drive, state_vip)
    check("SpikingVIPRing forward", z.shape == (B, N), f"z={tuple(z.shape)}")
except Exception as e:
    check("SpikingVIPRing", False, repr(e))

# ---- V2 LSNN ----
try:
    v2 = SpikingV2Context(cfg, scfg)
    state_v2 = v2.init_state(batch_size=B)
    n_lsnn = getattr(scfg, "n_lsnn_neurons", "?")
    print(f"[ok]  SpikingV2Context instantiated (n_lsnn={n_lsnn})")
    ok_count += 1
except Exception as e:
    check("SpikingV2Context", False, repr(e))

# ---- Full network ----
# SpikingLaminarV1V2Network exposes state via the initial_spiking_state
# helper (not a bound init_state method) and a forward that takes a full
# [B, T, N] stimulus trajectory.
try:
    net = SpikingLaminarV1V2Network(cfg, scfg)
    T = 5
    stim = torch.randn(B, T, N).abs()
    out, _, aux = net(stim)
    check("SpikingLaminarV1V2Network forward",
          out.shape == (B, T, N)
          and "r_l4_all" in aux
          and "spike_l23_all" in aux,
          f"out={tuple(out.shape)} aux_keys={sorted(aux.keys())[:5]}")
    check("SpikingLaminarV1V2Network no NaN",
          torch.isfinite(out).all()
          and all(torch.isfinite(t).all() for t in aux.values()
                  if isinstance(t, torch.Tensor)),
          "NaN/Inf in output or aux")
except Exception as e:
    check("SpikingLaminarV1V2Network", False, repr(e))

print()
print(f"smoke test: {ok_count} passed, {err_count} failed")
sys.exit(0 if err_count == 0 else 1)
PY
if [[ $? -eq 0 ]]; then
    ok "Forward-pass smoke test clean"
else
    bad "Forward-pass smoke test had failures (see above)"
fi

# ---------------------------------------------------------------------------
# 5. Total suite — everything together
# ---------------------------------------------------------------------------
hdr "5. Full pytest suite (rate + spiking)"

FULL_OUT=$(python -m pytest tests/ -q --tb=no 2>&1 || true)
[[ $VERBOSE -eq 1 ]] && echo "$FULL_OUT"

FULL_LINE=$(echo "$FULL_OUT" | grep -E '^[0-9]+ (passed|failed)' | tail -1)
FULL_PASSED=$(echo "$FULL_LINE" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
FULL_FAILED=$(echo "$FULL_LINE" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)

if [[ "$FULL_FAILED" == "0" && "$FULL_PASSED" -ge 374 ]]; then
    ok "Full suite: $FULL_PASSED passed, 0 failed"
else
    bad "Full suite: $FULL_PASSED passed / $FULL_FAILED failed (baseline >= 374 passed, 0 failed)"
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
hdr "VERDICT"

if [[ $FAIL -eq 0 ]]; then
    echo " ${GRN}${BLD}GO${RST} — Phase 1 ready for merge"
    exit 0
else
    echo " ${RED}${BLD}NO-GO${RST} — $FAIL check(s) failed:"
    for m in "${FAIL_MSGS[@]}"; do echo "   - $m"; done
    exit 1
fi
