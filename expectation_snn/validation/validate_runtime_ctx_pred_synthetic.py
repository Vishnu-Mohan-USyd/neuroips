"""Synthetic ctx_pred frozen-runtime construction smoke.

This validator creates temporary Stage-0 and Stage-1 ctx_pred checkpoints
with deterministic shapes and values, then exercises
``build_frozen_network(architecture="ctx_pred")`` without relying on trained
artifacts. It is a construction/checkpoint-loading smoke only; synthetic
weights are not biologically or scientifically meaningful.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from brian2 import Network, SpikeMonitor, defaultclock, ms, prefs
from brian2 import seed as b2_seed, start_scope

from expectation_snn.assays.runtime import (
    _expected_ctx_pred_shape,
    _expected_h_ee_shape,
    _expected_v1_pv_to_e_shape,
    build_frozen_network,
    set_grating,
)


SEED = 42


def _write_synthetic_checkpoints(ckpt_dir: Path) -> None:
    start_scope()
    # Use the runtime shape helpers on real builders through a temporary
    # synthetic bundle so this validator stays aligned with model topology.
    from expectation_snn.brian2_model.h_context_prediction import (
        build_h_context_prediction,
    )
    from expectation_snn.brian2_model.v1_ring import build_v1_ring

    v1 = build_v1_ring(name_prefix="synthetic_shape_v1")
    ctx_pred = build_h_context_prediction(
        ctx_name="synthetic_shape_ctx",
        pred_name="synthetic_shape_pred",
    )
    shape_net = Network(*v1.groups, *ctx_pred.groups)
    shape_net.run(0 * ms)

    stage0_shape = _expected_v1_pv_to_e_shape(v1)
    h_ee_shape = _expected_h_ee_shape(ctx_pred.ctx)
    ctx_pred_shape = _expected_ctx_pred_shape(ctx_pred)

    np.savez(
        ckpt_dir / f"stage_0_seed{SEED}.npz",
        bias_pA=0.0,
        pv_to_e_w=np.zeros(stage0_shape, dtype=np.float64),
    )
    np.savez(
        ckpt_dir / f"stage_1_ctx_pred_seed{SEED}.npz",
        ctx_ee_w_final=np.zeros(h_ee_shape, dtype=np.float64),
        pred_ee_w_final=np.zeros(h_ee_shape, dtype=np.float64),
        W_ctx_pred_final=np.full(ctx_pred_shape, 0.005, dtype=np.float64),
    )


def main() -> int:
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)

    with tempfile.TemporaryDirectory(prefix="ctx_pred_synthetic_ckpt_") as tmp_s:
        ckpt_dir = Path(tmp_s)
        _write_synthetic_checkpoints(ckpt_dir)

        start_scope()
        prefs.codegen.target = "numpy"
        defaultclock.dt = 0.1 * ms
        b2_seed(SEED)
        np.random.seed(SEED)

        bundle = build_frozen_network(
            architecture="ctx_pred",
            seed=SEED,
            ckpt_dir=str(ckpt_dir),
            with_cue=False,
            with_v1_to_h="continuous",
            with_feedback_routes=True,
        )
        assert bundle.meta["architecture"] == "ctx_pred"
        assert bundle.ctx_pred is not None
        assert bundle.h_ring is bundle.ctx_pred.pred
        assert bundle.meta["n_pv_to_e"] == 6144
        assert bundle.meta["n_ee_w"] == 36672
        assert bundle.meta["n_ctx_pred_w"] == 36864
        assert bundle.v1_to_h is not None
        assert bundle.v1_to_h.v1_to_he.target is bundle.ctx_pred.ctx.e

        v1_mon = SpikeMonitor(bundle.v1_ring.e, name="synthetic_ctx_pred_v1_e")
        ctx_mon = SpikeMonitor(bundle.ctx_pred.ctx.e, name="synthetic_ctx_pred_ctx_e")
        pred_mon = SpikeMonitor(bundle.ctx_pred.pred.e, name="synthetic_ctx_pred_pred_e")
        net = Network(*bundle.groups, v1_mon, ctx_mon, pred_mon)
        set_grating(bundle.v1_ring, theta_rad=0.0, contrast=0.0)
        net.run(1.0 * ms)

        print(
            "validate_runtime_ctx_pred_synthetic:",
            f"n_pv_to_e={bundle.meta['n_pv_to_e']}",
            f"n_ctx_ee_w={bundle.meta['n_ctx_ee_w']}",
            f"n_pred_ee_w={bundle.meta['n_pred_ee_w']}",
            f"n_ctx_pred_w={bundle.meta['n_ctx_pred_w']}",
            f"v1_spikes={int(v1_mon.num_spikes)}",
            f"ctx_spikes={int(ctx_mon.num_spikes)}",
            f"pred_spikes={int(pred_mon.num_spikes)}",
        )
    print("validate_runtime_ctx_pred_synthetic: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
