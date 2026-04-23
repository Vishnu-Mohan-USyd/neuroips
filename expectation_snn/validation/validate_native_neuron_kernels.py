"""Validate first native network-specific neuron CUDA kernels."""
from __future__ import annotations

from typing import Any

import numpy as np

from expectation_snn.cuda_sim.native import backend_info, run_decay_test


TOL = 1e-9


def _max_error(result: dict[str, Any]) -> float:
    return max(float(v) for v in result["max_abs_error"].values())


def _assert_decay_case(
    population: str,
    *,
    n_steps: int,
    threshold_case: bool,
    expect_spikes: bool,
) -> float:
    result = run_decay_test(
        population,
        n_steps=n_steps,
        threshold_case=threshold_case,
    )
    max_err = _max_error(result)
    assert max_err <= TOL, (population, threshold_case, max_err, result["max_abs_error"])
    assert int(result["cpu_total_spikes"]) == int(result["cuda_total_spikes"])
    if expect_spikes:
        assert int(result["cpu_total_spikes"]) > 0, (population, result)
    else:
        assert int(result["cpu_total_spikes"]) == 0, (population, result)
    assert np.array_equal(result["cpu_spike_counts"], result["cuda_spike_counts"])
    for key, cpu_values in result["cpu_state"].items():
        assert np.allclose(cpu_values, result["cuda_state"][key], atol=TOL, rtol=0.0), (
            population, key, threshold_case,
        )
    return max_err


def main() -> int:
    cases = [
        ("v1_e", 100, False, False),
        ("h_e", 100, False, False),
        ("v1_e", 25, True, True),
        ("h_e", 25, True, True),
    ]
    errors: dict[str, float] = {}
    for population, n_steps, threshold_case, expect_spikes in cases:
        label = f"{population}|steps={n_steps}|threshold={int(threshold_case)}"
        errors[label] = _assert_decay_case(
            population,
            n_steps=n_steps,
            threshold_case=threshold_case,
            expect_spikes=expect_spikes,
        )

    err_text = " ".join(f"{k}:max_err={v:.3e}" for k, v in errors.items())
    print(
        "validate_native_neuron_kernels: PASS",
        f"backend_info={backend_info()}",
        err_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

