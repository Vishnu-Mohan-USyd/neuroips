"""Validate post-hoc SpikeMonitor window extraction semantics.

This is a checkpoint-free CPU smoke path for the Richter CUDA migration. It
checks the helper semantics used by ``scripts/diag_ctx_pred_richter_balance.py``
without building the full frozen ctx_pred network.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from brian2 import Network, SpikeGeneratorGroup, SpikeMonitor, defaultclock, ms, prefs
from brian2 import start_scope

from scripts.diag_ctx_pred_richter_balance import (
    _counts_from_windows,
    _h_rate_from_windows,
)


def main() -> int:
    start_scope()
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    spike_i = np.asarray([0, 1, 2, 3, 0, 1, 2], dtype=np.int32)
    spike_t = np.asarray([1.0, 4.9, 5.0, 7.5, 9.9, 10.0, 12.0]) * ms
    group = SpikeGeneratorGroup(4, spike_i, spike_t)
    mon = SpikeMonitor(group)
    Network(group, mon).run(13.0 * ms)

    windows_ms = [(0.0, 5.0), (5.0, 10.0), (10.0, 13.0)]
    counts = _counts_from_windows(mon, 4, windows_ms)
    expected_counts = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(counts, expected_counts)

    ring = SimpleNamespace(e=group, e_channel=np.asarray([0, 0, 1, 1], dtype=np.int64))
    rates = _h_rate_from_windows(mon, ring, windows_ms)
    expected_rates = np.asarray(
        [
            [200.0, 0.0],
            [100.0, 200.0],
            [1.0 / 2.0 / 0.003, 1.0 / 2.0 / 0.003],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(rates, expected_rates, rtol=1e-12, atol=1e-12)

    try:
        _h_rate_from_windows(mon, ring, [(2.0, 2.0)])
    except ValueError as exc:
        if "invalid posthoc window" not in str(exc):
            raise
    else:
        raise AssertionError("_h_rate_from_windows accepted a zero-length window")

    print("validate_posthoc_counts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
