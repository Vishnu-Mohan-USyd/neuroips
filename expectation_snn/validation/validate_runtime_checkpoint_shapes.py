"""Validate frozen-runtime checkpoint shape checks with synthetic arrays.

The CUDA migration needs checkpoint loaders to reject malformed artifacts
without reading live Brian2 ``Synapses`` state before standalone build. This
validator uses tiny synthetic ``.npz`` files and the real model builders, but
does not require trained checkpoints or run a simulation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from brian2 import Network, ms, prefs, start_scope

from expectation_snn.assays.runtime import (
    _expected_ctx_pred_shape,
    _expected_h_ee_shape,
    _expected_stage2_cue_shape,
    _expected_v1_pv_to_e_shape,
    _load_stage0_into_v1,
    _load_stage1_ctx_pred_into,
    _load_stage1_into_h,
    _required_checkpoint_vector,
)
from expectation_snn.brian2_model.h_context_prediction import (
    build_h_context_prediction,
)
from expectation_snn.brian2_model.h_ring import build_h_r
from expectation_snn.brian2_model.v1_ring import build_v1_ring


def _save_npz(path: Path, **arrays: object) -> None:
    np.savez(path, **arrays)


def _expect_error(fn: Callable[[], object], exc_type: type[Exception], text: str) -> None:
    try:
        fn()
    except exc_type as exc:
        if text not in str(exc):
            raise AssertionError(f"expected {text!r} in {exc!r}") from exc
        return
    raise AssertionError(f"expected {exc_type.__name__} containing {text!r}")


def _validate_stage0(tmp: Path) -> None:
    start_scope()
    v1 = build_v1_ring(name_prefix="shape_v1")
    net = Network(*v1.groups)
    net.run(0 * ms)
    expected = _expected_v1_pv_to_e_shape(v1)
    assert expected == (6144,), expected

    good = tmp / "stage0_good.npz"
    _save_npz(good, bias_pA=125.0, pv_to_e_w=np.zeros(expected))
    bias, n_w = _load_stage0_into_v1(v1, str(good))
    assert bias == 125.0
    assert n_w == expected[0]

    bad = tmp / "stage0_bad_shape.npz"
    _save_npz(bad, bias_pA=125.0, pv_to_e_w=np.zeros((1, expected[0])))
    _expect_error(
        lambda: _load_stage0_into_v1(v1, str(bad)),
        ValueError,
        f"pv_to_e_w shape {(1, expected[0])} != expected {expected}",
    )


def _validate_stage1_h(tmp: Path) -> None:
    start_scope()
    h = build_h_r()
    net = Network(*h.groups)
    net.run(0 * ms)
    expected = _expected_h_ee_shape(h)
    assert expected == (36672,), expected

    good = tmp / "stage1_h_good.npz"
    _save_npz(good, ee_w_final=np.full(expected, 0.25))
    assert _load_stage1_into_h(h, str(good)) == expected[0]

    bad = tmp / "stage1_h_bad_shape.npz"
    _save_npz(bad, ee_w_final=np.zeros((expected[0], 1)))
    _expect_error(
        lambda: _load_stage1_into_h(h, str(bad)),
        ValueError,
        f"ee_w_final shape {(expected[0], 1)} != expected {expected}",
    )


def _validate_ctx_pred(tmp: Path) -> None:
    start_scope()
    bundle = build_h_context_prediction(ctx_name="shape_ctx", pred_name="shape_pred")
    net = Network(*bundle.groups)
    net.run(0 * ms)
    h_expected = _expected_h_ee_shape(bundle.ctx)
    cp_expected = _expected_ctx_pred_shape(bundle)
    assert h_expected == (36672,), h_expected
    assert cp_expected == (36864,), cp_expected

    good = tmp / "ctx_pred_good.npz"
    _save_npz(
        good,
        ctx_ee_w_final=np.full(h_expected, 0.1),
        pred_ee_w_final=np.full(h_expected, 0.2),
        W_ctx_pred_final=np.full(cp_expected, 0.01),
    )
    meta = _load_stage1_ctx_pred_into(bundle, str(good))
    assert meta["n_ctx_ee_w"] == h_expected[0]
    assert meta["n_pred_ee_w"] == h_expected[0]
    assert meta["n_ctx_pred_w"] == cp_expected[0]

    bad = tmp / "ctx_pred_bad_shape.npz"
    _save_npz(
        bad,
        ctx_ee_w_final=np.full(h_expected, 0.1),
        pred_ee_w_final=np.full(h_expected, 0.2),
        W_ctx_pred_final=np.zeros((cp_expected[0], 1)),
    )
    _expect_error(
        lambda: _load_stage1_ctx_pred_into(bundle, str(bad)),
        ValueError,
        f"W_ctx_pred_final shape {(cp_expected[0], 1)} != expected {cp_expected}",
    )

    missing = tmp / "ctx_pred_missing_key.npz"
    _save_npz(
        missing,
        ctx_ee_w_final=np.full(h_expected, 0.1),
        pred_ee_w_final=np.full(h_expected, 0.2),
    )
    _expect_error(
        lambda: _load_stage1_ctx_pred_into(bundle, str(missing)),
        KeyError,
        "W_ctx_pred_final",
    )


def _validate_stage2_cue_shape(tmp: Path) -> None:
    start_scope()
    h = build_h_r()
    net = Network(*h.groups)
    net.run(0 * ms)
    expected = _expected_stage2_cue_shape(h)
    assert expected == (6144,), expected

    cue = tmp / "stage2_cue.npz"
    _save_npz(cue, cue_A_w_final=np.zeros(expected))
    data = np.load(cue)
    arr = _required_checkpoint_vector(
        data, "cue_A_w_final", expected, "Stage-2",
    )
    assert arr.shape == expected


def main() -> int:
    prefs.codegen.target = "numpy"
    with tempfile.TemporaryDirectory(prefix="runtime_ckpt_shapes_") as tmp_s:
        tmp = Path(tmp_s)
        _validate_stage0(tmp)
        _validate_stage1_h(tmp)
        _validate_ctx_pred(tmp)
        _validate_stage2_cue_shape(tmp)
    print("validate_runtime_checkpoint_shapes: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
