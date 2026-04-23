"""Tiny Brian2 backend-selection smoke validator.

This is intentionally smaller than the biological validators. It proves that
the repository backend helper can select NumPy by default and opt into Brian2
standalone backends for a minimal one-neuron network.

Run:
    EXPECTATION_SNN_BACKEND=numpy python -m expectation_snn.validation.validate_backend
    EXPECTATION_SNN_BACKEND=cuda python -m expectation_snn.validation.validate_backend
"""
from __future__ import annotations

import numpy as np

from brian2 import (
    NeuronGroup,
    SpikeMonitor,
    defaultclock,
    ms,
    run,
    seed as b2_seed,
    start_scope,
)

from expectation_snn.brian2_model.backend import configure_backend


SEED = 42


def main() -> int:
    cfg = configure_backend()
    start_scope()
    defaultclock.dt = 0.1 * ms
    b2_seed(SEED)
    np.random.seed(SEED)

    group = NeuronGroup(
        1,
        "dv/dt = -v / (10*ms) : 1",
        threshold="v > 1",
        reset="v = 0",
        method="exact",
    )
    group.v = 2
    monitor = SpikeMonitor(group)

    run(1 * ms)
    n_spikes = int(monitor.num_spikes)
    passed = n_spikes == 1
    print(
        f"validate_backend: backend={cfg.name} spikes={n_spikes} "
        f"directory={cfg.directory}"
    )
    print("validate_backend: PASS" if passed else "validate_backend: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
