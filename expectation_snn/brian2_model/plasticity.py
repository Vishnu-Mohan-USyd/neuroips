"""Plasticity rules for the expectation_snn prototype.

Three rules implemented, one reserved as a fallback:

1. `pair_stdp_with_normalization` — standard symmetric pair-STDP with
   multiplicative weight-dependent bounds on LTP and LTD. Target: H recurrent
   E<->E synapses during Stage 1 (co-firing drives bump formation).
   Multiplicative bounds avoid runaway growth. Postsynaptic sum normalization
   is exposed as a separate helper `normalize_postsyn_sum` so the caller can
   wire it into a Brian2 `NetworkOperation` (which needs the full Python
   runtime, not Brian2's code-generation path).

2. `vogels_istdp` — Vogels 2011 inhibitory STDP. Event-driven symmetric trace
   rule that drives the postsynaptic E cell toward a target firing rate rho.
   Target: PV -> E within V1 and H rings; E/I balance across the Stage-0
   calibration sweep without hand-tuning per cell.

3. `eligibility_trace_cue_rule` — three-factor rule for cue -> H_R synapses
   used during Stage 2 cue learning (Frémaux & Gerstner 2015). Pre-spikes set
   an eligibility trace; H_E post-spikes (driven by the teacher pathway, wired
   externally) consume the trace and update the weight.

4. `saponati_vinck_rule` — reserved as fallback only, NotImplementedError.

Every rule factory returns a Brian2 `Synapses` object with its `connect()`
already called. The wiring-layer modules (v1_ring.py, h_ring.py,
feedback_routes.py) compose these factories into the larger circuit.

References
----------
- Bi GQ, Poo MM (1998) J Neurosci 18:10464.
- Royer S, Pare D (2003) Nature 422:518.
- Turrigiano GG (2008) Cell 135:422.
- Vogels TP et al. (2011) Science 334:1569.
- Frémaux N, Gerstner W (2015) Front Neural Circuits 9:85.
- Saponati M, Vinck M (2023) Nat Commun 14:4985.
"""
from __future__ import annotations

import numpy as np

from brian2 import (
    Hz,
    Synapses,
    ms,
)


# ---------- 1. pair-STDP with multiplicative bounds -------------------------

def pair_stdp_with_normalization(
    source_group,
    target_group,
    *,
    connectivity: str = "i != j",
    w_init: float = 0.5,
    w_max: float = 1.0,
    A_plus: float = 0.01,
    A_minus: float = 0.0105,
    tau_pre=20 * ms,
    tau_post=20 * ms,
    drive_amp_pA: float = 20.0,
    target_channel: str = "soma",
    name: str = "pair_stdp",
) -> Synapses:
    """Pair-STDP synapses with multiplicative weight-dependent bounds.

    Rule (event-driven, Brian2 idiom):

        on_pre:   Apre  += A_plus;
                  w     = clip(w - A_minus * w * Apost, 0, w_max).
        on_post:  Apost += 1;
                  w     = clip(w + A_plus * (w_max - w) * Apre, 0, w_max).

    with Apre and Apost decaying at tau_pre and tau_post. Multiplicative bounds
    prevent runaway: LTP saturates as w -> w_max, LTD saturates as w -> 0.

    Parameters
    ----------
    source_group, target_group : NeuronGroup
    connectivity : str
        Brian2 connect condition.
    w_init : float
    w_max : float
    A_plus, A_minus : float
        LTP and LTD amplitudes. A_minus > A_plus is the classic stability
        ratio for pair-STDP under Poisson drive (Bi & Poo 1998 ~1.05).
    tau_pre, tau_post : Quantity
    drive_amp_pA : float
        Amplitude of the postsynaptic current step on each pre-spike, in pA.
        The effective drive is `w * drive_amp_pA` (so a unit-weight synapse
        deposits drive_amp_pA picoamps).
    target_channel : {"soma", "apical"}
        Which post-synaptic current variable to write into. "soma" targets
        ``I_e_post``; "apical" targets ``I_ap_e_post`` (only valid when the
        target cell has an apical compartment — V1_E).
    name : str

    Returns
    -------
    Synapses
        Connected synapses with weights initialised to w_init.

    See Also
    --------
    `normalize_postsyn_sum` — call periodically via NetworkOperation to
        enforce a postsynaptic weight-sum target.
    """
    model = """
    w : 1
    dApre/dt  = -Apre  / tau_pre  : 1 (event-driven)
    dApost/dt = -Apost / tau_post : 1 (event-driven)
    """
    if "I_ap_e" in target_group.variables and target_channel == "apical":
        on_pre_drive = f"I_ap_e_post += w * {drive_amp_pA}*pA"
    elif "I_e" in target_group.variables:
        on_pre_drive = f"I_e_post += w * {drive_amp_pA}*pA"
    else:
        on_pre_drive = "v_post += w * mV"
    on_pre = f"""
    {on_pre_drive}
    Apre += A_plus_eff
    w = clip(w - A_minus_eff * w * Apost, 0, w_max_eff)
    """
    on_post = """
    Apost += 1.0
    w = clip(w + A_plus_eff * (w_max_eff - w) * Apre, 0, w_max_eff)
    """

    namespace = {
        "A_plus_eff": A_plus,
        "A_minus_eff": A_minus,
        "w_max_eff": w_max,
        "tau_pre": tau_pre,
        "tau_post": tau_post,
    }
    syn = Synapses(
        source_group,
        target_group,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="linear",
        namespace=namespace,
        name=name,
    )
    syn.connect(condition=connectivity)
    syn.w = w_init
    return syn


def normalize_postsyn_sum(syn: Synapses, target_sum: float) -> None:
    """Rescale each postsynaptic cell's incoming weights so their sum = target.

    Intended to be called inside a Brian2 `NetworkOperation` at a chosen
    cadence (e.g. every 100 ms) by the caller. We do the groupwise rescale in
    numpy (O(#synapses)); Brian2's code-generation path cannot express this
    directly, which is why it lives as a Python helper rather than inside the
    Synapses model.

    Parameters
    ----------
    syn : Synapses
        Must expose `.w[:]` and `.j[:]` (postsyn index) arrays.
    target_sum : float
        Desired sum of w over synapses incident on each postsyn cell.
    """
    w = np.asarray(syn.w[:], dtype=np.float64)
    j = np.asarray(syn.j[:], dtype=np.int64)
    n_post = int(j.max()) + 1 if j.size else 0
    if n_post == 0:
        return
    sums = np.bincount(j, weights=w, minlength=n_post)
    scales = np.where(sums > 1e-12, target_sum / sums, 1.0)
    syn.w[:] = w * scales[j]


# ---------- 2. Vogels 2011 iSTDP ---------------------------------------------

def vogels_istdp(
    source_inh_group,
    target_e_group,
    *,
    connectivity: str = "True",
    w_init: float = 0.1,
    w_max: float = 10.0,
    eta: float = 1e-3,
    rho=5 * Hz,
    tau=20 * ms,
    drive_amp_pA: float = 20.0,
    name: str = "vogels_istdp",
) -> Synapses:
    """Vogels 2011 inhibitory STDP.

    Presynaptic trace `xpre` and postsynaptic trace `xpost` both decay at tau.
    On each presynaptic spike:

        w += eta * (xpost - alpha),   alpha = 2 * rho * tau.

    On each postsynaptic spike:

        w += eta * xpre.

    Over long runs, postsynaptic rate converges to rho.

    Parameters
    ----------
    source_inh_group, target_e_group : NeuronGroup
    connectivity : str
    w_init : float
    w_max : float
        Weight cap.
    eta : float
        Learning rate.
    rho : Quantity (Hz)
        Target postsynaptic firing rate.
    tau : Quantity (ms)
        Trace decay time constant.
    drive_amp_pA : float
        Amplitude of inhibitory current (subtracted) on each pre spike.
    name : str

    Notes
    -----
    Names `xpre` / `xpost` are deliberately chosen to avoid Brian2's reserved
    `_pre` / `_post` suffixes.
    """
    model = """
    w : 1
    dxpre/dt  = -xpre  / tau_vogels : 1 (event-driven)
    dxpost/dt = -xpost / tau_vogels : 1 (event-driven)
    """
    if "I_i" in target_e_group.variables:
        inh_drive_pre = f"I_i_post += w * {drive_amp_pA}*pA"
    else:
        inh_drive_pre = "v_post -= w * mV"
    on_pre = f"""
    {inh_drive_pre}
    xpre += 1.0
    w = clip(w + eta_eff * (xpost - alpha_eff), 0, w_max_eff)
    """
    on_post = """
    xpost += 1.0
    w = clip(w + eta_eff * xpre, 0, w_max_eff)
    """
    alpha = 2.0 * float(rho * tau)  # dimensionless constant
    namespace = {
        "eta_eff": eta,
        "alpha_eff": alpha,
        "w_max_eff": w_max,
        "tau_vogels": tau,
    }
    syn = Synapses(
        source_inh_group,
        target_e_group,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="linear",
        namespace=namespace,
        name=name,
    )
    syn.connect(condition=connectivity)
    syn.w = w_init
    return syn


# ---------- 3. Eligibility-trace cue rule (three-factor) ---------------------

def eligibility_trace_cue_rule(
    cue_group,
    h_e_group,
    teacher_group=None,
    *,
    connectivity: str = "True",
    w_init: float = 0.2,
    w_max: float = 2.0,
    tau_elig=500 * ms,
    learning_rate: float = 0.05,
    drive_amp_pA: float = 20.0,
    name: str = "cue_elig",
) -> Synapses:
    """Cue -> H_E synapses with eligibility traces; teacher gates post spikes.

    Event-driven three-factor rule:

        on_pre:  drive_post; elig = 1.
        elig decays: d elig / dt = -elig / tau_elig.
        on_post: w += learning_rate * elig.

    The teacher is assumed to drive postsynaptic H_E cells directly (wired by
    the caller outside this factory) during the reward / teacher window; a
    post spike in that window coincides with a still-elevated eligibility
    from the cue, yielding a weight update. Outside the window, elig has
    decayed to near zero so spontaneous post spikes barely move the weight.

    Parameters
    ----------
    cue_group, h_e_group : NeuronGroup
    teacher_group : NeuronGroup or None
        Reserved for future explicit-teacher wiring; currently unused inside
        the factory (the caller wires teacher->h_e directly).
    connectivity : str
    w_init : float
    w_max : float
    tau_elig : Quantity (ms)
        Eligibility trace decay. ~500 ms matches the cue-to-H window used in
        Stage 2 (plan sec 2).
    learning_rate : float
    drive_amp_pA : float
    name : str

    Notes
    -----
    Variable name is `elig` rather than `e` because `e` collides with
    Brian2's built-in Euler constant.
    """
    _ = teacher_group
    model = """
    w : 1
    delig/dt = -elig / tau_elig_eff : 1 (event-driven)
    """
    if "I_e" in h_e_group.variables:
        drive_pre = f"I_e_post += w * {drive_amp_pA}*pA"
    else:
        drive_pre = "v_post += w * mV"
    on_pre = f"""
    {drive_pre}
    elig = 1.0
    """
    on_post = """
    w = clip(w + lr_eff * elig, 0, w_max_eff)
    """
    namespace = {
        "tau_elig_eff": tau_elig,
        "lr_eff": learning_rate,
        "w_max_eff": w_max,
    }
    syn = Synapses(
        cue_group,
        h_e_group,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="linear",
        namespace=namespace,
        name=name,
    )
    syn.connect(condition=connectivity)
    syn.w = w_init
    return syn


# ---------- 4. Saponati & Vinck 2023 predictive rule (placeholder) -----------

def saponati_vinck_rule(*args, **kwargs):
    """Saponati & Vinck 2023 predictive-learning rule (placeholder).

    Reserved as a fallback; raises `NotImplementedError`.
    """
    raise NotImplementedError(
        "Saponati & Vinck 2023 rule not implemented yet — "
        "reserved as a fallback per plan sec 2 if pair-STDP fails."
    )


# ---------- __main__ unit tests ---------------------------------------------

def _test_pair_stdp() -> bool:
    """Pair-STDP: isolated pre-before-post pairs produce net LTP.

    Spacing pairs by >> 5*tau so the Apost trace from the previous post
    spike has decayed; each pair then acts independently.
    """
    from brian2 import (
        Network, NeuronGroup, SpikeGeneratorGroup, Synapses,
        defaultclock, ms, mV, nS, prefs, seed as b2_seed,
    )
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(0); np.random.seed(0)

    # Minimal single-compartment LIF post with the new I_e/I_i channels.
    post = NeuronGroup(
        1,
        """dv/dt = (-(v - (-65*mV)) + (I_e - I_i)/(20*nS))/(10*ms) : volt
           dI_e/dt = -I_e/(5*ms) : amp
           dI_i/dt = -I_i/(10*ms) : amp""",
        threshold="v > -50*mV",
        reset="v = -65*mV",
        method="euler",
    )
    post.v = -65 * mV

    # Pairs at 100 ms gap (>> 5 tau=100 ms), pre-before-post with 2 ms delay.
    pre_times_ms = [1.0, 101.0, 201.0, 301.0]
    post_times_ms = [3.0, 103.0, 203.0, 303.0]
    pre = SpikeGeneratorGroup(1, [0] * len(pre_times_ms), np.asarray(pre_times_ms) * ms,
                              name="pstdp_pre")
    post_kick = SpikeGeneratorGroup(1, [0] * len(post_times_ms),
                                     np.asarray(post_times_ms) * ms, name="pstdp_postkick")

    # Use the factory; note: target has I_syn, not I_rec.
    syn = pair_stdp_with_normalization(
        pre, post, connectivity="True", w_init=0.5, w_max=1.0,
        A_plus=0.05, A_minus=0.055, tau_pre=20 * ms, tau_post=20 * ms,
        drive_amp_pA=0.0,  # don't let the STDP synapse itself kick post
        name="test_pstdp",
    )
    # A separate fixed synapse to force the post spike at exactly the kick time.
    kick = Synapses(post_kick, post, on_pre="v_post += 100*mV", name="test_kick")
    kick.connect()

    w_before = float(syn.w[0])
    net = Network(pre, post, post_kick, syn, kick)
    net.run(320 * ms)
    w_after = float(syn.w[0])
    print(f"pair_stdp[pre-before-post x4]: w {w_before:.4f} -> {w_after:.4f} "
          f"(delta={w_after - w_before:+.4f})")
    assert w_after > w_before, "Pair-STDP pre-before-post should increase w"
    return True


def _test_vogels() -> bool:
    """Vogels iSTDP qualitative check: weights grow when post overshoots rho."""
    from brian2 import (
        Network, NeuronGroup, PoissonGroup, SpikeMonitor, Synapses,
        defaultclock, Hz, ms, prefs, seed as b2_seed, mV, pA, nS,
    )
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(1); np.random.seed(1)

    # E cell with new I_e/I_i synaptic channels.
    e = NeuronGroup(
        1,
        """dv/dt = (-(v - (-65*mV)) + (I_e - I_i)/(20*nS))/(10*ms) : volt (unless refractory)
           dI_e/dt = -I_e/(5*ms) : amp
           dI_i/dt = -I_i/(10*ms) : amp""",
        threshold="v > -50*mV",
        reset="v = -65*mV",
        refractory=2 * ms,
        method="euler",
    )
    e.v = -65 * mV
    exc_src = PoissonGroup(50, rates=50 * Hz)
    exc_syn = Synapses(exc_src, e, on_pre="I_e_post += 40*pA")
    exc_syn.connect()
    inh_src = PoissonGroup(20, rates=20 * Hz)
    inh_syn = vogels_istdp(
        inh_src, e, connectivity="True",
        w_init=0.01, w_max=10.0, eta=1e-3, rho=5 * Hz, tau=20 * ms,
        drive_amp_pA=40.0, name="test_vogels",
    )
    mon = SpikeMonitor(e)
    net = Network(e, exc_src, exc_syn, inh_src, inh_syn, mon)
    w0 = float(np.asarray(inh_syn.w[:]).mean())
    net.run(2000 * ms)
    w1 = float(np.asarray(inh_syn.w[:]).mean())
    rate = mon.num_spikes / 2.0
    print(f"vogels: w_mean {w0:.4f} -> {w1:.4f}, post rate = {rate:.2f} Hz")
    assert w1 > w0, "Vogels iSTDP should grow PV weights when post > rho"
    return True


def _test_eligibility() -> bool:
    """Eligibility trace: weight delta on a post spike reflects the current
    trace, which decays with tau_elig between spikes.

    Trace is event-driven, so we probe it indirectly: two post spikes at
    different delays after a single pre spike should produce markedly
    different weight updates (first ~1.0, second ~e^{-k}).
    """
    from brian2 import (
        Network, NeuronGroup, SpikeGeneratorGroup, Synapses,
        defaultclock, ms, prefs, seed as b2_seed,
    )
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(2); np.random.seed(2)

    # Single pre spike at t=5ms; post spikes at t=10ms (trace ~1) and
    # t=205ms (trace ~e^{-200/50}~0.018).
    cue = SpikeGeneratorGroup(1, [0], np.asarray([5.0]) * ms, name="elig_cue")
    h = NeuronGroup(
        1,
        """dv/dt = -v/(10*ms) : 1
           dI_e/dt = -I_e/(5*ms) : amp""",
        threshold="v > 0.5",
        reset="v = 0",
        method="exact",
    )
    post_kick = SpikeGeneratorGroup(
        1, [0, 0], np.asarray([10.0, 205.0]) * ms, name="elig_kick"
    )
    kick = Synapses(post_kick, h, on_pre="v_post += 1.0", name="elig_kick_syn")
    kick.connect()

    syn = eligibility_trace_cue_rule(
        cue, h, teacher_group=None, connectivity="True",
        w_init=0.5, w_max=10.0, tau_elig=50 * ms,
        learning_rate=1.0, drive_amp_pA=0.0, name="test_elig",
    )
    net = Network(cue, h, post_kick, kick, syn)
    w0 = float(syn.w[0])
    net.run(11 * ms)              # after first post spike
    w1 = float(syn.w[0])
    net.run(300 * ms)             # after second post spike
    w2 = float(syn.w[0])
    d1 = w1 - w0
    d2 = w2 - w1
    print(f"eligibility: w {w0:.4f} -> {w1:.4f} -> {w2:.4f} "
          f"(delta_fresh={d1:+.4f}, delta_decayed={d2:+.4f})")
    assert d1 > 0.5, f"First post spike should consume ~full trace (got {d1})"
    assert d2 < 0.2 * d1, (
        f"Second post spike after 200 ms should see decayed trace "
        f"(got d2={d2}, d1={d1})"
    )
    return True


def _test_saponati() -> bool:
    """Saponati & Vinck placeholder raises NotImplementedError."""
    try:
        saponati_vinck_rule()
    except NotImplementedError as exc:
        print(f"saponati_vinck: correctly raised NotImplementedError: {exc}")
        return True
    raise AssertionError("saponati_vinck_rule should raise NotImplementedError")


def _test_normalize() -> bool:
    """normalize_postsyn_sum: each postsyn cell's incoming sum -> target."""
    from brian2 import (
        Network, NeuronGroup, SpikeGeneratorGroup,
        defaultclock, ms, prefs, seed as b2_seed,
    )
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(3); np.random.seed(3)

    pre = SpikeGeneratorGroup(4, [], np.asarray([]) * ms, name="norm_pre")
    post = NeuronGroup(
        2,
        """dv/dt = 0*Hz : 1
           dI_e/dt = -I_e/(5*ms) : amp""",
        threshold="v > 9999",
        reset="v = 0",
        method="exact",
    )
    syn = pair_stdp_with_normalization(
        pre, post, connectivity="True", w_init=0.5, name="test_norm",
    )
    # Set non-uniform weights: post 0 gets [0.1, 0.2, 0.3, 0.4] (sum 1.0),
    # post 1 gets [0.5, 0.5, 0.5, 0.5] (sum 2.0).
    # Brian2 .w vector is ordered by (i, j) pairs from connect("True").
    # We set by j to make it crystal clear.
    w_vec = np.zeros(len(syn))
    for k, (i, j) in enumerate(zip(np.asarray(syn.i[:]), np.asarray(syn.j[:]))):
        if int(j) == 0:
            w_vec[k] = 0.1 + 0.1 * int(i)   # 0.1, 0.2, 0.3, 0.4
        else:
            w_vec[k] = 0.5
    syn.w[:] = w_vec
    normalize_postsyn_sum(syn, target_sum=1.0)
    w_new = np.asarray(syn.w[:])
    j_vec = np.asarray(syn.j[:])
    s0 = w_new[j_vec == 0].sum()
    s1 = w_new[j_vec == 1].sum()
    print(f"normalize: sum_j0 = {s0:.4f}, sum_j1 = {s1:.4f} (target 1.0)")
    assert abs(s0 - 1.0) < 1e-6, f"post 0 sum {s0} != 1.0"
    assert abs(s1 - 1.0) < 1e-6, f"post 1 sum {s1} != 1.0"
    return True


if __name__ == "__main__":
    ok = True
    tests = [
        ("pair_stdp", _test_pair_stdp),
        ("vogels", _test_vogels),
        ("eligibility", _test_eligibility),
        ("normalize", _test_normalize),
        ("saponati", _test_saponati),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"{name} FAIL: {exc!r}")
            ok = False
    print("plasticity smoke:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
