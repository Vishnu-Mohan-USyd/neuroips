"""Component-level validation for the Stage-2 cue-rule plasticity (Sprint 4).

Isolated test of `eligibility_trace_cue_rule` from `brian2_model/plasticity.py`
BEFORE wiring it into the Stage-2 driver. Three-factor rule:

    on_pre:  I_e_post += w * drive_amp_pA;  elig = 1.
    elig decays at tau_elig (default 1500 ms here per plan section 2).
    on_post: w += lr * elig  (teacher-forced post spike is the third factor).

Four biology-anchored assays at seed=42:

1. **Selectivity ratio (matched vs unmatched).** Train 50 trials of
   (cue_A @ t=0 -> teacher @ H_E[0] @ t=TEACHER_DELAY_MS), then 50 trials
   of (cue_B @ t=0 -> teacher @ H_E[1]). Assertion:
   w(cue_A -> H_E[0]) / w(cue_A -> H_E[1]) >= SELECTIVITY_RATIO_MIN
   and symmetrically for cue_B.

2. **Cue-alone delay firing (matched >> unmatched).** After training,
   freeze learning (namespace lr -> 0) and present cue_A as a
   TEST_CUE_RATE_HZ Poisson burst for DELAY_WINDOW_MS. Measure H_E rates.
   Assertion: rate(H_E[0]) >= DELAY_RATE_MIN_HZ and
   rate(H_E[0]) / max(rate(H_E[1]), 1 Hz) >= SELECTIVITY_RATIO_MIN.

3. **Eligibility trace tau in band [1200, 1800] ms.** Parallel probe:
   for each of N_TAU_DELAYS delays t_k, wire a fresh (cue_k -> H_E_k)
   pair; cue_k fires once at t=0, single teacher kick at t_k. Read back
   Delta w_k = lr * exp(-t_k / tau_elig). Log-linear fit on (t_k, log
   Delta w_k) recovers tau. Assertion: tau_fit in [1200, 1800] ms.

4. **Cross-pair leakage control.** The opposing weights (cue_A -> H_E[1]
   and cue_B -> H_E[0]) must remain near w_init because no teacher
   spikes into the unmatched H_E during their cue phase. Assertion:
   max(w_cross) - w_init <= CROSS_LEAK_MAX (structural safety check).

Run:
    python -m expectation_snn.validation.validate_cue_rule
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from brian2 import (
    Hz,
    Network,
    NeuronGroup,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Synapses,
    defaultclock,
    mV,
    ms,
    pA,
    prefs,
    seed as b2_seed,
)

from ..brian2_model.neurons import make_h_e_population
from ..brian2_model.plasticity import eligibility_trace_cue_rule


# --- constants --------------------------------------------------------------

SEED = 42
TAU_ELIG_MS = 1500.0
TAU_ELIG_BAND_MS: Tuple[float, float] = (1200.0, 1800.0)

N_TRIALS_PER_CUE = 50
TRIAL_MS = 2500.0                 # trial length (elig decays well before next)
TEACHER_DELAY_MS = 1000.0         # cue -> teacher latency per Lead spec
SETTLE_MS = 50.0                  # pre-first-trial settle
TEACHER_KICK_MV = 30.0            # strong kick to force H_E post-spike

CUE_W_INIT = 0.2
CUE_W_MAX = 2.0
CUE_LR = 0.08
CUE_DRIVE_AMP_PA = 300.0          # per-spike peak I_e push (plan value ~20 pA
                                  # is Stage-2 wiring; here we use a higher
                                  # value so a cue-alone Poisson burst can
                                  # drive H_E above its 200 pA rheobase)

SELECTIVITY_RATIO_MIN = 2.0       # matched / unmatched weight
DELAY_RATE_MIN_HZ = 5.0           # matched H_E cue-alone firing floor
DELAY_WINDOW_MS = 500.0
TEST_CUE_RATE_HZ = 200.0
CROSS_LEAK_MAX = 0.05             # max drift of unmatched w above w_init

N_TAU_DELAYS = 6                  # parallel probes for tau fit


# --- assay 1+2+4: training + freeze + cue-alone test -----------------------

def _build_train_network(seed: int):
    """2 cues x 2 H_E cells; plastic cue->H_E; kicker teacher->H_E."""
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)

    h_e = make_h_e_population(2, name="cue_rule_he")

    # Cues (single-cell each; spike times set per-trial)
    cue_a = SpikeGeneratorGroup(1, [], [] * ms, name="cue_a")
    cue_b = SpikeGeneratorGroup(1, [], [] * ms, name="cue_b")

    # Teacher generators + spike-forcing synapses
    teacher_a = SpikeGeneratorGroup(1, [], [] * ms, name="teacher_a")
    teacher_b = SpikeGeneratorGroup(1, [], [] * ms, name="teacher_b")
    teacher_a_syn = Synapses(
        teacher_a, h_e,
        on_pre=f"V_post += {TEACHER_KICK_MV}*mV",
        name="teacher_a_syn",
    )
    teacher_a_syn.connect(i=0, j=0)
    teacher_b_syn = Synapses(
        teacher_b, h_e,
        on_pre=f"V_post += {TEACHER_KICK_MV}*mV",
        name="teacher_b_syn",
    )
    teacher_b_syn.connect(i=0, j=1)

    # Plastic cue -> H_E (each cue connects to BOTH H_E cells)
    cue_a_syn = eligibility_trace_cue_rule(
        cue_a, h_e, connectivity="True",
        w_init=CUE_W_INIT, w_max=CUE_W_MAX,
        tau_elig=TAU_ELIG_MS * ms, learning_rate=CUE_LR,
        drive_amp_pA=CUE_DRIVE_AMP_PA, name="cue_a_syn",
    )
    cue_b_syn = eligibility_trace_cue_rule(
        cue_b, h_e, connectivity="True",
        w_init=CUE_W_INIT, w_max=CUE_W_MAX,
        tau_elig=TAU_ELIG_MS * ms, learning_rate=CUE_LR,
        drive_amp_pA=CUE_DRIVE_AMP_PA, name="cue_b_syn",
    )

    return {
        "h_e": h_e,
        "cue_a": cue_a, "cue_b": cue_b,
        "teacher_a": teacher_a, "teacher_b": teacher_b,
        "cue_a_syn": cue_a_syn, "cue_b_syn": cue_b_syn,
        "teacher_a_syn": teacher_a_syn, "teacher_b_syn": teacher_b_syn,
    }


def _schedule_training(bundle, n_trials: int):
    """Set spike times for (cue_A+teacher_A) then (cue_B+teacher_B) phases."""
    phase_a_times_cue, phase_a_times_teacher = [], []
    for i in range(n_trials):
        t0 = SETTLE_MS + i * TRIAL_MS + 10.0
        phase_a_times_cue.append(t0)
        phase_a_times_teacher.append(t0 + TEACHER_DELAY_MS)

    phase_b_times_cue, phase_b_times_teacher = [], []
    phase_b_start = SETTLE_MS + n_trials * TRIAL_MS
    for i in range(n_trials):
        t0 = phase_b_start + i * TRIAL_MS + 10.0
        phase_b_times_cue.append(t0)
        phase_b_times_teacher.append(t0 + TEACHER_DELAY_MS)

    total_ms = phase_b_start + n_trials * TRIAL_MS

    bundle["cue_a"].set_spikes(
        np.zeros(n_trials, dtype=np.int64),
        np.asarray(phase_a_times_cue) * ms,
    )
    bundle["teacher_a"].set_spikes(
        np.zeros(n_trials, dtype=np.int64),
        np.asarray(phase_a_times_teacher) * ms,
    )
    bundle["cue_b"].set_spikes(
        np.zeros(n_trials, dtype=np.int64),
        np.asarray(phase_b_times_cue) * ms,
    )
    bundle["teacher_b"].set_spikes(
        np.zeros(n_trials, dtype=np.int64),
        np.asarray(phase_b_times_teacher) * ms,
    )
    return total_ms


def _run_training(seed: int = SEED, n_trials: int = N_TRIALS_PER_CUE):
    """Returns (bundle, weights_after) where weights_after is dict of 4 edges."""
    bundle = _build_train_network(seed)
    total_ms = _schedule_training(bundle, n_trials)

    net = Network(
        bundle["h_e"],
        bundle["cue_a"], bundle["cue_b"],
        bundle["teacher_a"], bundle["teacher_b"],
        bundle["cue_a_syn"], bundle["cue_b_syn"],
        bundle["teacher_a_syn"], bundle["teacher_b_syn"],
    )
    net.run(total_ms * ms)

    cue_a_syn = bundle["cue_a_syn"]
    cue_b_syn = bundle["cue_b_syn"]
    # cue -> H_E connectivity is "True": index order (i=0, j=0) and (i=0, j=1).
    # For the single-cell cue group, syn.j tells us the post index.
    j_a = np.asarray(cue_a_syn.j[:], dtype=np.int64)
    w_a = np.asarray(cue_a_syn.w[:])
    j_b = np.asarray(cue_b_syn.j[:], dtype=np.int64)
    w_b = np.asarray(cue_b_syn.w[:])

    weights = {
        "a_to_he0": float(w_a[j_a == 0][0]),
        "a_to_he1": float(w_a[j_a == 1][0]),
        "b_to_he0": float(w_b[j_b == 0][0]),
        "b_to_he1": float(w_b[j_b == 1][0]),
    }
    return bundle, net, weights


def _cue_alone_probe(bundle, net, cue_name: str, probe_ms: float = DELAY_WINDOW_MS,
                     rate_hz: float = TEST_CUE_RATE_HZ,
                     rng: np.random.Generator = None):
    """Inject a Poisson burst into `cue_name` with learning frozen.

    Sets lr_eff namespace to 0 on both plastic synapses so the test does
    not modify the trained weights.
    Returns per-cell H_E rate (Hz) over the probe window.
    """
    t_now_ms = float(net.t / ms)
    # Freeze learning. namespace is a regular dict on the Synapses; we
    # mutate `lr_eff` and rely on Brian2's re-resolution at next run.
    bundle["cue_a_syn"].namespace["lr_eff"] = 0.0
    bundle["cue_b_syn"].namespace["lr_eff"] = 0.0

    # Schedule a Poisson burst on the named cue
    if rng is None:
        rng = np.random.default_rng(SEED + 17)
    n_expected = int(rate_hz * probe_ms / 1000.0)
    # Homogeneous Poisson: inter-spike intervals ~ Exp(lambda=rate)
    isis = rng.exponential(1000.0 / rate_hz, size=n_expected * 3)
    ts = np.cumsum(isis)
    ts = ts[ts < probe_ms]
    if len(ts) == 0:
        ts = np.array([1.0])
    # Enforce minimum ISI >= defaultclock.dt (0.1 ms). Round to dt grid and
    # dedupe, since the SpikeGeneratorGroup forbids >1 spike per dt.
    dt_ms = 0.1
    ts = np.unique(np.round(ts / dt_ms).astype(np.int64)) * dt_ms
    spike_times = (t_now_ms + ts)
    bundle[cue_name].set_spikes(
        np.zeros(len(spike_times), dtype=np.int64),
        spike_times * ms,
    )
    # Silence the other cue and both teachers
    other = "cue_b" if cue_name == "cue_a" else "cue_a"
    bundle[other].set_spikes([], [] * ms)
    bundle["teacher_a"].set_spikes([], [] * ms)
    bundle["teacher_b"].set_spikes([], [] * ms)

    # Add a fresh SpikeMonitor on H_E
    h_mon = SpikeMonitor(bundle["h_e"], name=f"hmon_{cue_name}_{int(t_now_ms)}")
    net.add(h_mon)
    net.run(probe_ms * ms)
    i = np.asarray(h_mon.i[:], dtype=np.int64)
    t = np.asarray(h_mon.t / ms)
    mask = (t >= t_now_ms) & (t < t_now_ms + probe_ms)
    rates = np.array([
        float((i[mask] == 0).sum()) / (probe_ms / 1000.0),
        float((i[mask] == 1).sum()) / (probe_ms / 1000.0),
    ])
    net.remove(h_mon)
    return rates


# --- assay 3: eligibility tau fit ------------------------------------------

def _measure_tau_elig(seed: int = SEED) -> Tuple[float, np.ndarray, np.ndarray]:
    """Parallel probe: for each delay t_k, one (cue_k -> H_E_k) pair with a
    single pre spike at t=0 and a single teacher kick at t=t_k. Returns
    (tau_fit_ms, delays_ms, delta_w_array).
    """
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(seed); np.random.seed(seed)

    delays_ms = np.linspace(100.0, 2500.0, N_TAU_DELAYS)
    N = len(delays_ms)
    h = make_h_e_population(N, name="tau_he")
    # one cue cell per probe, paired i -> j = i
    cue = SpikeGeneratorGroup(
        N, np.arange(N, dtype=np.int64), np.full(N, 10.0) * ms, name="tau_cue",
    )
    teacher = SpikeGeneratorGroup(
        N, np.arange(N, dtype=np.int64),
        (10.0 + delays_ms) * ms, name="tau_teacher",
    )
    teacher_syn = Synapses(
        teacher, h, on_pre=f"V_post += {TEACHER_KICK_MV}*mV", name="tau_tsyn",
    )
    teacher_syn.connect(j="i")
    cue_syn = eligibility_trace_cue_rule(
        cue, h, connectivity="i==j",
        w_init=CUE_W_INIT, w_max=CUE_W_MAX,
        tau_elig=TAU_ELIG_MS * ms, learning_rate=CUE_LR,
        drive_amp_pA=0.0,                 # isolate elig: no pre-driven currents
        name="tau_cue_syn",
    )
    net = Network(h, cue, teacher, cue_syn, teacher_syn)
    net.run((10.0 + float(delays_ms.max()) + 50.0) * ms)

    w_final = np.asarray(cue_syn.w[:])
    # Synapses were built via connect(i==j), so ordering may not be by i.
    # Reorder by i to align with delays.
    i_vec = np.asarray(cue_syn.i[:], dtype=np.int64)
    order = np.argsort(i_vec)
    w_sorted = w_final[order]
    delta_w = w_sorted - CUE_W_INIT

    # log-linear fit on log(delta_w) vs delay -> slope = -1/tau
    valid = delta_w > 1e-9
    slope, intercept = np.polyfit(
        delays_ms[valid], np.log(delta_w[valid]), 1,
    )
    tau_fit = float(-1.0 / slope)
    return tau_fit, delays_ms, delta_w


# --- report ----------------------------------------------------------------

@dataclass
class CueRuleValidationReport:
    weights: dict
    selectivity_a: float
    selectivity_b: float
    rate_matched_a: float
    rate_unmatched_a: float
    rate_matched_b: float
    rate_unmatched_b: float
    tau_fit_ms: float
    tau_band_ms: Tuple[float, float]
    max_cross_leak: float

    passed_selectivity: bool
    passed_delay_rate: bool
    passed_tau: bool
    passed_cross_leak: bool

    @property
    def passed(self) -> bool:
        return (self.passed_selectivity and self.passed_delay_rate
                and self.passed_tau and self.passed_cross_leak)

    def summary(self) -> str:
        w = self.weights
        lines = ["Cue-rule validation (Sprint-4 Step 1, seed=42):"]
        lines.append("  Assay 1: selectivity ratio (matched/unmatched weight)")
        lines.append(
            f"    cue_A: w_matched={w['a_to_he0']:.3f} "
            f"w_unmatched={w['a_to_he1']:.3f}  "
            f"ratio = {self.selectivity_a:.2f}"
        )
        lines.append(
            f"    cue_B: w_matched={w['b_to_he1']:.3f} "
            f"w_unmatched={w['b_to_he0']:.3f}  "
            f"ratio = {self.selectivity_b:.2f}"
        )
        lines.append(
            f"    floor >= {SELECTIVITY_RATIO_MIN}:  "
            f"{'PASS' if self.passed_selectivity else 'FAIL'}"
        )
        lines.append("  Assay 2: cue-alone delay firing (learning frozen)")
        lines.append(
            f"    cue_A alone: rate(H_E[0])={self.rate_matched_a:6.2f} Hz "
            f"rate(H_E[1])={self.rate_unmatched_a:6.2f} Hz"
        )
        lines.append(
            f"    cue_B alone: rate(H_E[1])={self.rate_matched_b:6.2f} Hz "
            f"rate(H_E[0])={self.rate_unmatched_b:6.2f} Hz"
        )
        lines.append(
            f"    matched >= {DELAY_RATE_MIN_HZ} Hz; ratio matched/unmatched "
            f">= {SELECTIVITY_RATIO_MIN}:  "
            f"{'PASS' if self.passed_delay_rate else 'FAIL'}"
        )
        lines.append("  Assay 3: eligibility trace tau")
        lines.append(
            f"    tau_fit = {self.tau_fit_ms:.1f} ms  "
            f"band = {self.tau_band_ms}  "
            f"{'PASS' if self.passed_tau else 'FAIL'}"
        )
        lines.append("  Assay 4: cross-pair leakage control")
        lines.append(
            f"    max(w_cross - w_init) = {self.max_cross_leak:.4f}  "
            f"(<= {CROSS_LEAK_MAX})  "
            f"{'PASS' if self.passed_cross_leak else 'FAIL'}"
        )
        lines.append("  ---")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def run_cue_rule_validation(verbose: bool = True) -> CueRuleValidationReport:
    # Assays 1 + 4: training, read weights.
    bundle, net, weights = _run_training(seed=SEED, n_trials=N_TRIALS_PER_CUE)
    sel_a = weights["a_to_he0"] / max(weights["a_to_he1"], 1e-6)
    sel_b = weights["b_to_he1"] / max(weights["b_to_he0"], 1e-6)
    max_cross_leak = max(
        weights["a_to_he1"] - CUE_W_INIT,
        weights["b_to_he0"] - CUE_W_INIT,
    )

    # Assay 2: cue-alone delay firing (learning frozen; uses trained network).
    rates_a = _cue_alone_probe(bundle, net, "cue_a")
    rates_b = _cue_alone_probe(bundle, net, "cue_b")

    # Assay 3: tau fit (separate mini-network).
    tau_fit_ms, _delays, _dw = _measure_tau_elig(seed=SEED)

    passed_selectivity = (
        sel_a >= SELECTIVITY_RATIO_MIN and sel_b >= SELECTIVITY_RATIO_MIN
    )
    passed_delay_rate = (
        rates_a[0] >= DELAY_RATE_MIN_HZ
        and rates_b[1] >= DELAY_RATE_MIN_HZ
        and (rates_a[0] / max(rates_a[1], 1.0)) >= SELECTIVITY_RATIO_MIN
        and (rates_b[1] / max(rates_b[0], 1.0)) >= SELECTIVITY_RATIO_MIN
    )
    passed_tau = TAU_ELIG_BAND_MS[0] <= tau_fit_ms <= TAU_ELIG_BAND_MS[1]
    passed_cross_leak = max_cross_leak <= CROSS_LEAK_MAX

    rep = CueRuleValidationReport(
        weights=weights,
        selectivity_a=sel_a,
        selectivity_b=sel_b,
        rate_matched_a=float(rates_a[0]),
        rate_unmatched_a=float(rates_a[1]),
        rate_matched_b=float(rates_b[1]),
        rate_unmatched_b=float(rates_b[0]),
        tau_fit_ms=tau_fit_ms,
        tau_band_ms=TAU_ELIG_BAND_MS,
        max_cross_leak=float(max_cross_leak),
        passed_selectivity=passed_selectivity,
        passed_delay_rate=passed_delay_rate,
        passed_tau=passed_tau,
        passed_cross_leak=passed_cross_leak,
    )
    if verbose:
        print(rep.summary())
    return rep


if __name__ == "__main__":
    rep = run_cue_rule_validation(verbose=True)
    if not rep.passed:
        raise SystemExit(1)
