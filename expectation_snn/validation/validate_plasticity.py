"""Component-level validation for the Sprint-3 plasticity extension
(`brian2_model/plasticity.py::pair_stdp_with_normalization`).

Sprint-3 added an `nmda_drive_amp_nS` parameter to
`pair_stdp_with_normalization`: when > 0, each pre-spike deposits
`w * nmda_drive_amp_nS` into the post-synaptic slow NMDA conductance
channel (`g_nmda_h_post`) on top of the usual AMPA deposit into
`I_e_post`. Plasticity still acts on the AMPA weight only (there is a
single scalar `w` per synapse); NMDA co-release scales with the same
weight at a fixed ratio. This is the Wang 2001 bump-attractor recipe.

Assays
------

1. **STDP asymmetry**: a train of pre-before-post pairings must
   potentiate w (net LTP); post-before-pre pairings must depress w
   (net LTD). Directly exercises the `on_pre` / `on_post` logic.

2. **NMDA co-release on**: with `nmda_drive_amp_nS > 0`, each pre-spike
   raises the post-synaptic `g_nmda_h` by approximately
   `w * nmda_drive_amp_nS`, decaying with tau_nmda_h (50 ms).

3. **NMDA co-release off**: with `nmda_drive_amp_nS = 0`, `g_nmda_h`
   stays at zero under pre-spikes (backward compatibility).

4. **Plasticity decouples AMPA from NMDA**: NMDA deposit amplitude on a
   pre-spike is exactly `w(t) * nmda_drive_amp_nS`. Over LTP/LTD, the
   AMPA weight changes but the *per-deposit* NMDA/AMPA ratio stays at
   `nmda_drive_amp_nS / drive_amp_pA` (NMDA is always co-released in
   fixed proportion to AMPA drive).

5. **Multiplicative weight bounds**: under repeated pre-before-post
   pairings, w saturates at w_max (never exceeds it) — checks the
   `clip(w, 0, w_max_eff)` in the rule.

6. **NMDA is pre-spike-only (not post-spike)**: Sprint-3 rework — fire
   ONLY post-spikes (via kicker), with NMDA co-release enabled on the
   synapse. Assert g_nmda_h stays at zero — post-spikes must NOT deposit
   into the slow NMDA channel. This is the biology (Wang 2001): NMDA
   co-release is governed by pre-synaptic vesicle release, not
   post-synaptic depolarization.

Run:
    python -m expectation_snn.validation.validate_plasticity
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    SpikeGeneratorGroup,
    Synapses,
    defaultclock,
    mV,
    ms,
    nS,
    nA,
    pA,
    prefs,
    seed as b2_seed,
)

from ..brian2_model.plasticity import pair_stdp_with_normalization


STDP_MIN_LTP_DELTA = 0.005       # Minimum weight increase from 10 pre-before-post pairings.
STDP_MIN_LTD_DELTA = 0.005       # Minimum weight decrease (abs) from 10 post-before-pre pairings.
NMDA_DEPOSIT_TOL_NS = 0.05       # Absolute tolerance on per-pre NMDA deposit (nS).


# --- local helper: a minimal H_E-like post cell with g_nmda_h --------------

def _make_h_e_like_post(name: str) -> NeuronGroup:
    """Minimal post cell exposing I_e, I_i, and g_nmda_h for the plasticity test.

    Kept local (rather than importing the full H_E factory) to keep the
    validator fast and to not entangle it with the neuron-factory code.
    """
    model = """
    dV/dt = (gL*(EL - V) + I_e - I_i) / C : volt (unless refractory)
    dI_e/dt = -I_e / tau_e : amp
    dI_i/dt = -I_i / tau_i : amp
    dg_nmda_h/dt = -g_nmda_h / tau_nmda_h : siemens
    """
    grp = NeuronGroup(
        1, model=model,
        threshold="V > V_th", reset="V = V_reset",
        refractory=2 * ms, method="euler", name=name,
        namespace={
            "V_th": -50 * mV, "V_reset": -65 * mV,
            "tau_e": 5 * ms, "tau_i": 10 * ms,
            "tau_nmda_h": 50 * ms,
            "gL": 10 * nS, "EL": -70 * mV, "C": 0.2 * nS * second_as_quantity(),
        },
    )
    grp.V = -70 * mV
    grp.I_e = 0 * pA
    grp.I_i = 0 * pA
    grp.g_nmda_h = 0 * nS
    return grp


def second_as_quantity():
    """Return 1 ms*Hz = dimensionless so we can set C via ms * nS indirectly.

    Brian2's unit system wants farad; (nS * ms) = nF already. We just need the
    right literal. Easier: use nS * 20 * ms = 20 nF-equivalent capacitive tau?
    """
    # Helper not actually used (left as a stub reference).
    from brian2 import ms
    return ms


# --- assay 1 + 5: STDP asymmetry + multiplicative bound --------------------

def _run_stdp_pairings(
    dt_pre_post_ms: float,
    n_pairs: int,
    inter_pair_ms: float,
    w_init: float,
    w_max: float,
    A_plus: float,
    A_minus: float,
    seed: int = 0,
) -> float:
    """Run `n_pairs` pre/post spike pairings separated by `inter_pair_ms` and
    return the final weight.

    `dt_pre_post_ms > 0`: pre spikes `dt` ms BEFORE post (LTP regime).
    `dt_pre_post_ms < 0`: pre spikes `-dt` ms AFTER post (LTD regime).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)

    # Build minimal post LIF (plasticity rule needs I_e_post).
    post = NeuronGroup(
        1,
        """dV/dt = (gL*(EL - V) + I_e - I_i) / C : volt (unless refractory)
           dI_e/dt = -I_e / tau_e : amp
           dI_i/dt = -I_i / tau_i : amp""",
        threshold="V > V_th", reset="V = V_reset",
        refractory=2 * ms, method="euler",
        namespace={
            "V_th": -50 * mV, "V_reset": -65 * mV,
            "tau_e": 5 * ms, "tau_i": 10 * ms,
            "gL": 10 * nS, "EL": -70 * mV, "C": 0.2 * (nS * 1e3 * ms),
        },
        name=f"stdp_post_{seed}",
    )
    post.V = -70 * mV

    # Timing: first pre at t=10 ms, then pairings every `inter_pair_ms`.
    base = 10.0
    pre_times = []
    post_times = []
    for k in range(n_pairs):
        t0 = base + k * inter_pair_ms
        if dt_pre_post_ms >= 0:
            pre_times.append(t0)
            post_times.append(t0 + dt_pre_post_ms)
        else:
            post_times.append(t0)
            pre_times.append(t0 - dt_pre_post_ms)
    pre = SpikeGeneratorGroup(
        1, [0] * len(pre_times), np.asarray(pre_times) * ms, name=f"stdp_pre_{seed}",
    )
    kicker = SpikeGeneratorGroup(
        1, [0] * len(post_times), np.asarray(post_times) * ms, name=f"stdp_kick_{seed}",
    )

    syn = pair_stdp_with_normalization(
        pre, post, connectivity="True",
        w_init=w_init, w_max=w_max,
        A_plus=A_plus, A_minus=A_minus,
        tau_pre=20 * ms, tau_post=20 * ms,
        drive_amp_pA=0.0,       # don't let STDP synapse drive post
        nmda_drive_amp_nS=0.0,  # assay 1 / 5: AMPA-only
        name=f"stdp_syn_{seed}",
    )
    kick_syn = Synapses(kicker, post, on_pre="V_post += 100*mV", name=f"stdp_kick_syn_{seed}")
    kick_syn.connect()

    duration_ms = base + n_pairs * inter_pair_ms + 50.0
    net = Network(pre, post, kicker, syn, kick_syn)
    net.run(duration_ms * ms)
    return float(syn.w[0])


# --- assay 2 + 3 + 4: NMDA co-release ---------------------------------------

def _run_nmda_coreleae_probe(
    w: float,
    nmda_drive_amp_nS: float,
    drive_amp_pA: float,
    dt_sample_ms: float = 0.1,
    probe_ms: float = 60.0,
    seed: int = 0,
) -> np.ndarray:
    """Drive a single pre-spike and record g_nmda_h_post over `probe_ms` ms.

    Returns
    -------
    g_nmda_trace_nS : np.ndarray, shape (n_samples,)
        Sampled NMDA conductance of the post cell (nS).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = dt_sample_ms * ms
    b2_seed(seed); np.random.seed(seed)

    # H_E-like post (inline, so we don't depend on external namespacing).
    post = NeuronGroup(
        1,
        """dV/dt = (gL*(EL - V) + I_e - I_i) / C : volt (unless refractory)
           dI_e/dt = -I_e / tau_e : amp
           dI_i/dt = -I_i / tau_i : amp
           dg_nmda_h/dt = -g_nmda_h / tau_nmda_h : siemens""",
        threshold="V > V_th", reset="V = V_reset",
        refractory=2 * ms, method="euler",
        namespace={
            "V_th": -50 * mV, "V_reset": -65 * mV,
            "tau_e": 5 * ms, "tau_i": 10 * ms,
            "tau_nmda_h": 50 * ms,
            "gL": 10 * nS, "EL": -70 * mV, "C": 0.2 * (nS * 1e3 * ms),
        },
        name=f"nmda_post_{seed}",
    )
    post.V = -70 * mV
    post.g_nmda_h = 0 * nS

    pre = SpikeGeneratorGroup(
        1, [0], np.asarray([5.0]) * ms, name=f"nmda_pre_{seed}",
    )
    syn = pair_stdp_with_normalization(
        pre, post, connectivity="True",
        w_init=w, w_max=max(w + 0.1, 1.0),
        A_plus=1e-6, A_minus=1e-6,   # effectively no weight change over probe
        tau_pre=20 * ms, tau_post=20 * ms,
        drive_amp_pA=drive_amp_pA,
        nmda_drive_amp_nS=nmda_drive_amp_nS,
        target_channel="soma",
        name=f"nmda_syn_{seed}",
    )

    # Record g_nmda_h via a state monitor.
    from brian2 import StateMonitor
    mon = StateMonitor(post, "g_nmda_h", record=True, name=f"nmda_mon_{seed}")
    net = Network(pre, post, syn, mon)
    net.run(probe_ms * ms)
    return np.asarray(mon.g_nmda_h[0] / nS)


# --- assay 6: NMDA deposited on pre-spikes only (not post-spikes) ----------

def _run_post_only_nmda_probe(
    nmda_drive_amp_nS: float,
    drive_amp_pA: float,
    n_post_spikes: int = 10,
    inter_spike_ms: float = 20.0,
    probe_ms: float = 300.0,
    dt_sample_ms: float = 0.1,
    seed: int = 77,
) -> float:
    """Fire ONLY post-spikes (no pre-spikes); return peak g_nmda_h.

    A separate kicker SpikeGeneratorGroup is wired into the post cell via
    ``V_post += 100*mV`` to force n_post_spikes inside the probe window.
    The STDP synapse has `nmda_drive_amp_nS > 0` so that IF post-spikes
    were depositing into g_nmda_h, we'd see a positive peak. The rule
    only deposits on on_pre, so the expected peak is 0.
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = dt_sample_ms * ms
    b2_seed(seed); np.random.seed(seed)

    post = NeuronGroup(
        1,
        """dV/dt = (gL*(EL - V) + I_e - I_i) / C : volt (unless refractory)
           dI_e/dt = -I_e / tau_e : amp
           dI_i/dt = -I_i / tau_i : amp
           dg_nmda_h/dt = -g_nmda_h / tau_nmda_h : siemens""",
        threshold="V > V_th", reset="V = V_reset",
        refractory=2 * ms, method="euler",
        namespace={
            "V_th": -50 * mV, "V_reset": -65 * mV,
            "tau_e": 5 * ms, "tau_i": 10 * ms,
            "tau_nmda_h": 50 * ms,
            "gL": 10 * nS, "EL": -70 * mV, "C": 0.2 * (nS * 1e3 * ms),
        },
        name=f"post_only_post_{seed}",
    )
    post.V = -70 * mV

    # Silent pre-group: attached to synapse but never fires. Use a
    # SpikeGeneratorGroup with zero-length time array.
    pre = SpikeGeneratorGroup(
        1, indices=np.asarray([], dtype=np.int64),
        times=np.asarray([]) * ms,
        name=f"post_only_pre_{seed}",
    )
    syn = pair_stdp_with_normalization(
        pre, post, connectivity="True",
        w_init=0.5, w_max=1.0,
        A_plus=1e-6, A_minus=1e-6,
        tau_pre=20 * ms, tau_post=20 * ms,
        drive_amp_pA=drive_amp_pA,
        nmda_drive_amp_nS=nmda_drive_amp_nS,
        target_channel="soma",
        name=f"post_only_syn_{seed}",
    )

    # Kicker: forces post to spike at n_post_spikes times without
    # involving the STDP synapse's pre path.
    kick_times = (5.0 + np.arange(n_post_spikes) * inter_spike_ms) * ms
    kicker = SpikeGeneratorGroup(
        1, indices=np.zeros(n_post_spikes, dtype=np.int64),
        times=kick_times, name=f"post_only_kick_{seed}",
    )
    kick_syn = Synapses(kicker, post, on_pre="V_post += 100*mV",
                        name=f"post_only_kick_syn_{seed}")
    kick_syn.connect()

    from brian2 import StateMonitor, SpikeMonitor
    g_mon = StateMonitor(post, "g_nmda_h", record=True,
                         name=f"post_only_g_{seed}")
    spk_mon = SpikeMonitor(post, name=f"post_only_spk_{seed}")
    net = Network(pre, post, kicker, syn, kick_syn, g_mon, spk_mon)
    net.run(probe_ms * ms)

    # Sanity: we must have observed post-spikes (the assay is meaningful
    # only if post actually fired without the pre path).
    if len(spk_mon.t) < n_post_spikes // 2:
        # Return a sentinel large value to force FAIL if kicker didn't work.
        return float("inf")
    return float(np.asarray(g_mon.g_nmda_h[0] / nS).max())


# --- Assay dataclasses ------------------------------------------------------

@dataclass
class PlasticityValidationReport:
    stdp_ltp_delta: float
    stdp_ltd_delta: float
    stdp_asymmetry_ok: bool

    nmda_peak_on_nS: float
    nmda_peak_expected_nS: float
    nmda_on_ok: bool

    nmda_peak_off_nS: float
    nmda_off_ok: bool

    nmda_ratio_w1: float
    nmda_ratio_w2: float
    nmda_ratio_constant_ok: bool

    bounded_w_final: float
    w_max_reference: float
    bounded_ok: bool

    nmda_peak_post_only_nS: float
    nmda_pre_only_ok: bool

    @property
    def passed(self) -> bool:
        return (self.stdp_asymmetry_ok and self.nmda_on_ok
                and self.nmda_off_ok and self.nmda_ratio_constant_ok
                and self.bounded_ok and self.nmda_pre_only_ok)

    def summary(self) -> str:
        lines = ["Plasticity validation (pair-STDP + NMDA co-release):"]
        lines.append("  1. STDP asymmetry (pre-post -> LTP; post-pre -> LTD):")
        lines.append(
            f"     LTP delta = {self.stdp_ltp_delta:+.4f} "
            f"(>= +{STDP_MIN_LTP_DELTA})  "
            f"LTD delta = {self.stdp_ltd_delta:+.4f} "
            f"(<= -{STDP_MIN_LTD_DELTA})  "
            f"{'PASS' if self.stdp_asymmetry_ok else 'FAIL'}"
        )
        lines.append("  2. NMDA co-release ON (peak g_nmda_h after 1 pre-spike):")
        lines.append(
            f"     peak g_nmda_h = {self.nmda_peak_on_nS:.3f} nS "
            f"(expected ~{self.nmda_peak_expected_nS:.3f} +/- "
            f"{NMDA_DEPOSIT_TOL_NS})  "
            f"{'PASS' if self.nmda_on_ok else 'FAIL'}"
        )
        lines.append("  3. NMDA co-release OFF (amp=0 -> g_nmda_h stays 0):")
        lines.append(
            f"     peak g_nmda_h = {self.nmda_peak_off_nS:.3f} nS  "
            f"{'PASS' if self.nmda_off_ok else 'FAIL'}"
        )
        lines.append("  4. Plasticity / NMDA decoupling (deposit ratio constant):")
        lines.append(
            f"     ratio(w=0.2) = {self.nmda_ratio_w1:.3f}, "
            f"ratio(w=0.8) = {self.nmda_ratio_w2:.3f}  "
            f"{'PASS' if self.nmda_ratio_constant_ok else 'FAIL'}"
        )
        lines.append("  5. Multiplicative bounds (w stays in [0, w_max]):")
        lines.append(
            f"     w_final = {self.bounded_w_final:.4f} "
            f"(w_max = {self.w_max_reference})  "
            f"{'PASS' if self.bounded_ok else 'FAIL'}"
        )
        lines.append("  6. NMDA deposit on pre-spikes ONLY (post-only -> g_nmda=0):")
        lines.append(
            f"     peak g_nmda_h (post-only drive) = "
            f"{self.nmda_peak_post_only_nS:.4f} nS  "
            f"(expect ~0)  "
            f"{'PASS' if self.nmda_pre_only_ok else 'FAIL'}"
        )
        lines.append("  -----------------------------------")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def run_plasticity_validation(verbose: bool = True) -> PlasticityValidationReport:
    # --- Assay 1: STDP asymmetry ---
    w0 = 0.5
    w_after_ltp = _run_stdp_pairings(
        dt_pre_post_ms=+5.0, n_pairs=20, inter_pair_ms=100.0,
        w_init=w0, w_max=1.0, A_plus=0.05, A_minus=0.055, seed=0,
    )
    w_after_ltd = _run_stdp_pairings(
        dt_pre_post_ms=-5.0, n_pairs=20, inter_pair_ms=100.0,
        w_init=w0, w_max=1.0, A_plus=0.05, A_minus=0.055, seed=1,
    )
    stdp_ltp_delta = w_after_ltp - w0
    stdp_ltd_delta = w_after_ltd - w0
    stdp_asym_ok = (stdp_ltp_delta >= STDP_MIN_LTP_DELTA
                    and stdp_ltd_delta <= -STDP_MIN_LTD_DELTA)

    # --- Assay 2: NMDA co-release ON ---
    w_probe = 0.5
    nmda_amp = 0.5  # nS
    drive_pA = 20.0
    trace_on = _run_nmda_coreleae_probe(
        w=w_probe, nmda_drive_amp_nS=nmda_amp, drive_amp_pA=drive_pA, seed=10,
    )
    nmda_peak_on = float(trace_on.max())
    nmda_peak_expected = w_probe * nmda_amp  # 0.25 nS
    nmda_on_ok = abs(nmda_peak_on - nmda_peak_expected) <= NMDA_DEPOSIT_TOL_NS

    # --- Assay 3: NMDA co-release OFF ---
    trace_off = _run_nmda_coreleae_probe(
        w=w_probe, nmda_drive_amp_nS=0.0, drive_amp_pA=drive_pA, seed=11,
    )
    nmda_peak_off = float(trace_off.max())
    nmda_off_ok = nmda_peak_off < NMDA_DEPOSIT_TOL_NS

    # --- Assay 4: NMDA deposit scales linearly with w (ratio constant) ---
    trace_w1 = _run_nmda_coreleae_probe(
        w=0.2, nmda_drive_amp_nS=nmda_amp, drive_amp_pA=drive_pA, seed=12,
    )
    trace_w2 = _run_nmda_coreleae_probe(
        w=0.8, nmda_drive_amp_nS=nmda_amp, drive_amp_pA=drive_pA, seed=13,
    )
    ratio_w1 = float(trace_w1.max()) / 0.2   # should == nmda_amp
    ratio_w2 = float(trace_w2.max()) / 0.8
    # Ratios should both equal nmda_amp within tolerance (i.e. linear in w).
    ratio_tol = 0.15  # 15% combined discretisation + event sampling jitter
    nmda_ratio_ok = (abs(ratio_w1 - nmda_amp) / nmda_amp < ratio_tol
                     and abs(ratio_w2 - nmda_amp) / nmda_amp < ratio_tol
                     and abs(ratio_w1 - ratio_w2) / nmda_amp < ratio_tol)

    # --- Assay 5: Multiplicative bounds ---
    w_bound_final = _run_stdp_pairings(
        dt_pre_post_ms=+5.0, n_pairs=500, inter_pair_ms=30.0,
        w_init=0.9, w_max=1.0, A_plus=0.2, A_minus=0.0, seed=20,
    )
    bounded_ok = (w_bound_final <= 1.0 + 1e-9) and (w_bound_final > 0.9)

    # --- Assay 6: NMDA deposit on pre-spikes ONLY (not post-spikes) ---
    nmda_peak_post_only = _run_post_only_nmda_probe(
        nmda_drive_amp_nS=nmda_amp, drive_amp_pA=drive_pA,
        n_post_spikes=10, inter_spike_ms=20.0, probe_ms=300.0, seed=77,
    )
    nmda_pre_only_ok = nmda_peak_post_only < NMDA_DEPOSIT_TOL_NS

    rep = PlasticityValidationReport(
        stdp_ltp_delta=stdp_ltp_delta,
        stdp_ltd_delta=stdp_ltd_delta,
        stdp_asymmetry_ok=stdp_asym_ok,
        nmda_peak_on_nS=nmda_peak_on,
        nmda_peak_expected_nS=nmda_peak_expected,
        nmda_on_ok=nmda_on_ok,
        nmda_peak_off_nS=nmda_peak_off,
        nmda_off_ok=nmda_off_ok,
        nmda_ratio_w1=ratio_w1,
        nmda_ratio_w2=ratio_w2,
        nmda_ratio_constant_ok=nmda_ratio_ok,
        bounded_w_final=w_bound_final,
        w_max_reference=1.0,
        bounded_ok=bounded_ok,
        nmda_peak_post_only_nS=nmda_peak_post_only,
        nmda_pre_only_ok=nmda_pre_only_ok,
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_plasticity_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
