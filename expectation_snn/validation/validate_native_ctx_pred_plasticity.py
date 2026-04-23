"""Validate native ctx_pred eligibility and delayed-gate plasticity primitive.

This is a GPU-native Stage1 foundation test, not a full Stage1 trainer. It
uses the real ctx_pred all-to-all dimensions (192 x 192 synapses) with a tiny
controlled event schedule so CPU reference and CUDA kernels can be compared
before adding the full trial loop.
"""
from __future__ import annotations

import numpy as np

from expectation_snn.cuda_sim.native import (
    backend_info,
    run_ctx_pred_plasticity_test,
)


TOL = 1e-10


def _edge(pre: int, post: int, n_post: int) -> int:
    return int(pre) * int(n_post) + int(post)


def _max_error(result: dict) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _assert_cpu_cuda_match(result: dict) -> None:
    assert _max_error(result) <= TOL, result["max_abs_error"]
    for key in (
        "w",
        "elig_before_gate",
        "elig_after_gate",
        "xpre_after_gate",
        "xpost_after_gate",
        "row_sums",
    ):
        cpu = np.asarray(result[f"cpu_{key}"], dtype=np.float64)
        cuda = np.asarray(result[f"cuda_{key}"], dtype=np.float64)
        assert np.allclose(cpu, cuda, atol=TOL, rtol=0.0), key
    assert int(result["cpu_n_capped"]) == int(result["cuda_n_capped"])


def _assert_repeat_deterministic(a: dict, b: dict) -> None:
    assert int(a["seed"]) == int(b["seed"]) == 42
    assert int(a["n_steps"]) == int(b["n_steps"])
    for key in (
        "initial_w",
        "cpu_w",
        "cuda_w",
        "cpu_elig_before_gate",
        "cuda_elig_before_gate",
        "cpu_row_sums",
        "cuda_row_sums",
    ):
        assert np.array_equal(
            np.asarray(a[key], dtype=np.float64),
            np.asarray(b[key], dtype=np.float64),
        ), key


def main() -> int:
    result = run_ctx_pred_plasticity_test(seed=42, n_steps=640)
    repeat = run_ctx_pred_plasticity_test(seed=42, n_steps=640)

    assert int(result["n_pre"]) == 192
    assert int(result["n_post"]) == 192
    assert int(result["n_syn"]) == 192 * 192
    assert int(result["n_steps"]) == 640
    assert np.isclose(float(result["dt_ms"]), 0.1, atol=0.0, rtol=0.0)
    assert np.isclose(float(result["tau_coinc_ms"]), 500.0, atol=0.0, rtol=0.0)
    assert np.isclose(float(result["tau_elig_ms"]), 1000.0, atol=0.0, rtol=0.0)

    _assert_cpu_cuda_match(result)
    _assert_repeat_deterministic(result, repeat)

    n_post = int(result["n_post"])
    paired_idx = _edge(
        int(result["paired_pre"]),
        int(result["paired_post"]),
        n_post,
    )
    pre_rule_idx = _edge(
        int(result["pre_rule_pre"]),
        int(result["paired_post"]),
        n_post,
    )
    silent_idx = _edge(
        int(result["silent_pre"]),
        int(result["silent_post"]),
        n_post,
    )

    initial_w = np.asarray(result["initial_w"], dtype=np.float64)
    cpu_w = np.asarray(result["cpu_w"], dtype=np.float64)
    elig_before = np.asarray(result["cpu_elig_before_gate"], dtype=np.float64)
    elig_after = np.asarray(result["cpu_elig_after_gate"], dtype=np.float64)
    row_sums = np.asarray(result["cpu_row_sums"], dtype=np.float64)

    assert elig_before[paired_idx] > 0.0
    assert elig_before[pre_rule_idx] > 0.0
    assert cpu_w[paired_idx] > initial_w[paired_idx]
    assert cpu_w[pre_rule_idx] > initial_w[pre_rule_idx]

    expected_silent = initial_w[silent_idx] - float(result["gamma"]) * (
        initial_w[silent_idx] - float(result["w_target"])
    ) * float(result["dt_trial_s"])
    expected_silent = np.clip(expected_silent, 0.0, float(result["w_max"]))
    assert np.isclose(cpu_w[silent_idx], expected_silent, atol=TOL, rtol=0.0), (
        cpu_w[silent_idx],
        expected_silent,
    )
    assert np.isclose(elig_before[silent_idx], 0.0, atol=TOL, rtol=0.0)

    capped_pre = int(result["capped_pre"])
    assert int(result["cpu_n_capped"]) == 1
    assert np.isclose(
        row_sums[capped_pre],
        float(result["w_row_max"]),
        atol=5e-12,
        rtol=0.0,
    )
    assert float(row_sums.max()) <= float(result["w_row_max"]) + 5e-12
    assert np.all(cpu_w >= -TOL)
    assert np.all(cpu_w <= float(result["w_max"]) + TOL)
    assert np.allclose(elig_after, 0.0, atol=TOL, rtol=0.0)

    print(
        "validate_native_ctx_pred_plasticity: PASS",
        f"backend_info={backend_info()}",
        f"n_syn={result['n_syn']}",
        f"pre_events={np.asarray(result['pre_event_steps']).tolist()}",
        f"post_events={np.asarray(result['post_event_steps']).tolist()}",
        f"paired_elig={elig_before[paired_idx]:.6e}",
        f"pre_rule_elig={elig_before[pre_rule_idx]:.6e}",
        f"silent_dw={cpu_w[silent_idx] - initial_w[silent_idx]:.6e}",
        f"capped_row_sum={row_sums[capped_pre]:.12f}",
        f"max_err={_max_error(result):.3e}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
