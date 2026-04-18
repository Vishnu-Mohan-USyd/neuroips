"""Brian2 smoke test for expectation_snn Phase 0.

Builds a minimal 100-neuron LIF network driven by Poisson input with pair-STDP on
the input->LIF synapses, runs for 100 ms, prints spike counts and weight changes.

Pass criterion: Brian2 imports, simulation runs to completion, at least one neuron
spikes, and STDP updates at least one synaptic weight.

Usage:
    conda activate expectation_snn
    python expectation_snn/scripts/smoke_test.py
"""
from __future__ import annotations

import numpy as np

from brian2 import (
    Hz,
    NeuronGroup,
    PoissonGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    mV,
    ms,
    prefs,
    run,
    seed,
)


def main() -> int:
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    seed(42)
    np.random.seed(42)

    N = 100
    v_rest = -70 * mV
    v_th = -50 * mV
    v_reset = -65 * mV
    tau = 10 * ms

    tau_pre = 20 * ms
    tau_post = 20 * ms
    A_plus = 0.01
    A_minus = -0.0105
    w_init = 5.0
    w_max = 10.0

    eqs = "dv/dt = (v_rest - v) / tau : volt (unless refractory)"

    neurons = NeuronGroup(
        N,
        model=eqs,
        threshold="v > v_th",
        reset="v = v_reset",
        refractory=2 * ms,
        method="exact",
    )
    neurons.v = v_rest

    poisson = PoissonGroup(N, rates=200 * Hz)

    stdp_model = """
    w : 1
    dApre/dt  = -Apre / tau_pre   : 1 (event-driven)
    dApost/dt = -Apost / tau_post : 1 (event-driven)
    """
    on_pre = """
    v_post += w * mV
    Apre += A_plus
    w = clip(w + Apost, 0, w_max)
    """
    on_post = """
    Apost += A_minus
    w = clip(w + Apre, 0, w_max)
    """

    syn = Synapses(poisson, neurons, model=stdp_model, on_pre=on_pre, on_post=on_post)
    syn.connect(j="i")
    syn.w = w_init

    mon_neurons = SpikeMonitor(neurons)
    mon_poisson = SpikeMonitor(poisson)

    w_before = np.array(syn.w[:])
    run(100 * ms)
    w_after = np.array(syn.w[:])

    n_neuron_spikes = int(mon_neurons.num_spikes)
    n_poisson_spikes = int(mon_poisson.num_spikes)
    n_weights_changed = int(np.sum(np.abs(w_after - w_before) > 1e-9))

    print(f"smoke_test: neuron spikes = {n_neuron_spikes}")
    print(f"smoke_test: poisson input spikes = {n_poisson_spikes}")
    print(f"smoke_test: synapses w changed = {n_weights_changed} / {len(w_before)}")
    print(f"smoke_test: w mean before -> after = {w_before.mean():.6f} -> {w_after.mean():.6f}")

    ok = n_neuron_spikes > 0 and n_poisson_spikes > 0 and n_weights_changed > 0
    print("smoke_test: PASS" if ok else "smoke_test: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
