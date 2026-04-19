"""Neuron populations for the expectation_snn V1 <-> H laminar prototype.

Every population exposes three synaptic-current channels with exponential
decay:

- `I_e`  : excitatory post-synaptic current, decays with `tau_e` (~5 ms).
- `I_i`  : inhibitory post-synaptic current, decays with `tau_i` (~10 ms).
- `I_bias` : static DC bias (no dynamics; for Stage 0 tonic calibration).

V1 excitatory cells add an apical compartment with its own excitatory
synaptic channel `I_ap_e` (top-down target), plus an algebraic apical-to-
soma gate current `I_ap`.

Design decisions
----------------
- Sub-threshold apical: apical input is modulatory, multiplying the soma's
  above-rest depolarization via a sigmoid on V_ap. No BAC-style apical spike.
- Brette-Gerstner 2005 spike-frequency adaptation (SFA): `w_adapt` integrates
  with tau_adapt, jumps by b_adapt on each post-spike. We stop short of full
  AdEx (exponential upswing) — LIF+SFA suffices for Stage-0 rate calibration.
- Inhibitory (PV, SOM) and H_E / H_inh cells are single-compartment LIF with
  the same three current channels. Differentiation between "recurrent E",
  "cue E", "feedback E" is handled by WHICH synapse deposits, not WHICH
  current variable.
- tau_e and tau_i are module-level constants (5 ms / 10 ms) matching AMPA /
  GABA-A kinetics.

References
----------
- Brette R, Gerstner W (2005) J Neurophysiol 94:3637.
- Hu H, Gan J, Jonas P (2014) Science 345:1255263.
- Urban-Ciecko J, Barth AL (2016) Nat Rev Neurosci 17:401.
- Niell CM, Stryker MP (2008) J Neurosci 28:7520.
- Larkum ME (2013) Trends Neurosci 36:141.
- Wang X-J (2001) Trends Neurosci 24:455.
"""
from __future__ import annotations

from brian2 import (
    Equations,
    Hz,
    NeuronGroup,
    PoissonGroup,
    mV,
    ms,
    nF,
    nS,
    pA,
)

# Synaptic kinetics (module-level, shared across populations).
TAU_E = 5.0 * ms        # AMPA-like excitatory current decay
TAU_I = 10.0 * ms       # GABA-A-like inhibitory current decay

# NMDA slow recurrent on H_E (Wang 2001 working-memory bump attractor).
TAU_NMDA_H = 50.0 * ms   # NMDA gating time constant. 50 ms is in the NR2A
                         # subunit-dominated range (Vicini et al. 1998); gives
                         # bump persistence in the [200, 500] ms working-memory
                         # band under our recurrent-gain regime. NR2B-dominated
                         # synapses have slower (~150-400 ms) decay.
V_NMDA_REV = 0.0 * mV    # NMDA reversal potential (mixed cation)
# Mg2+ block (Jahr & Stevens 1990, [Mg]=1 mM):
#   s_nmda(V) = 1 / (1 + exp(-0.062 * V/mV) / 3.57)
#   s_nmda(-70 mV) = 0.044  (mostly blocked at rest)
#   s_nmda(-50 mV) = 0.138  (partial unblock at threshold)
#   s_nmda(  0 mV) = 0.781  (mostly open under strong depolarization)


# --- V1 excitatory pyramidal (LIF soma + passive apical + SFA) --------------

V1E_C_SOMA = 0.2 * nF
V1E_GL_SOMA = 10.0 * nS
V1E_EL = -70.0 * mV
V1E_VT = -50.0 * mV
V1E_VR = -65.0 * mV
V1E_REFRACTORY = 2.0 * ms
V1E_C_AP = 0.1 * nF
V1E_GL_AP = 4.0 * nS
V1E_EL_AP = -70.0 * mV
V1E_G_AP_SOMA = 2.0 * nS
V1E_V_AP_TH = -55.0 * mV
V1E_V_AP_SLOPE = 4.0 * mV
V1E_A_ADAPT = 0.0 * nS
V1E_B_ADAPT = 30.0 * pA
V1E_TAU_ADAPT = 150.0 * ms

_V1E_EQS = Equations(
    """
    dV_soma/dt = (
        gL_soma*(EL_soma - V_soma)
        + I_bias + I_e + I_ap - I_i - w_adapt
    ) / C_soma : volt (unless refractory)
    dV_ap/dt = (
        gL_ap*(EL_ap - V_ap) + I_ap_e
    ) / C_ap : volt
    I_ap = g_ap_soma * 1.0 / (1.0 + exp(-(V_ap - V_ap_th)/V_ap_slope)) * (V_soma - EL_soma) : amp
    dI_e/dt     = -I_e     / tau_e : amp
    dI_i/dt     = -I_i     / tau_i : amp
    dI_ap_e/dt  = -I_ap_e  / tau_e : amp
    dw_adapt/dt = (a_adapt*(V_soma - EL_soma) - w_adapt) / tau_adapt : amp
    I_bias : amp
    C_soma : farad
    gL_soma : siemens
    EL_soma : volt
    C_ap : farad
    gL_ap : siemens
    EL_ap : volt
    g_ap_soma : siemens
    V_ap_th : volt
    V_ap_slope : volt
    a_adapt : siemens
    tau_adapt : second
    """
)


def make_v1_e_population(n_cells: int, name: str = "v1_e") -> NeuronGroup:
    """Build the V1 excitatory population: LIF soma + passive apical + SFA.

    Post-synapses write into ``I_e_post`` (soma-targeted excitatory),
    ``I_i_post`` (soma-targeted inhibitory), or ``I_ap_e_post`` (apical
    excitatory / top-down). ``I_bias`` is a static DC parameter for
    Stage-0 tonic bias calibration.

    Parameters
    ----------
    n_cells : int
    name : str
        Brian2 group name; must be unique within the enclosing Network.

    Returns
    -------
    NeuronGroup
        V1_E population at rest, zero adaptation, zero synaptic currents.
    """
    group = NeuronGroup(
        n_cells,
        model=_V1E_EQS,
        threshold="V_soma > V_th",
        reset="V_soma = V_reset; w_adapt += b_adapt",
        refractory=V1E_REFRACTORY,
        method="euler",
        name=name,
        namespace={
            "V_th": V1E_VT,
            "V_reset": V1E_VR,
            "b_adapt": V1E_B_ADAPT,
            "tau_e": TAU_E,
            "tau_i": TAU_I,
        },
    )
    group.V_soma = V1E_EL
    group.V_ap = V1E_EL_AP
    group.I_e = 0 * pA
    group.I_i = 0 * pA
    group.I_ap_e = 0 * pA
    group.I_bias = 0 * pA
    group.w_adapt = 0 * pA
    group.C_soma = V1E_C_SOMA
    group.gL_soma = V1E_GL_SOMA
    group.EL_soma = V1E_EL
    group.C_ap = V1E_C_AP
    group.gL_ap = V1E_GL_AP
    group.EL_ap = V1E_EL_AP
    group.g_ap_soma = V1E_G_AP_SOMA
    group.V_ap_th = V1E_V_AP_TH
    group.V_ap_slope = V1E_V_AP_SLOPE
    group.a_adapt = V1E_A_ADAPT
    group.tau_adapt = V1E_TAU_ADAPT
    return group


# --- V1 inhibitory PV (fast-spiking basket) ----------------------------------

V1PV_C = 0.1 * nF
V1PV_GL = 20.0 * nS
V1PV_EL = -65.0 * mV
V1PV_VT = -50.0 * mV
V1PV_VR = -65.0 * mV
V1PV_REFRACTORY = 1.0 * ms

_LIF_WITH_EI_EQS = Equations(
    """
    dV/dt = (gL*(EL - V) + I_bias + I_e - I_i) / C : volt (unless refractory)
    dI_e/dt = -I_e / tau_e : amp
    dI_i/dt = -I_i / tau_i : amp
    I_bias : amp
    C : farad
    gL : siemens
    EL : volt
    """
)


def make_v1_pv_population(n_cells: int, name: str = "v1_pv") -> NeuronGroup:
    """V1 PV fast-spiking basket: LIF, tau_m ~5 ms (Hu 2014)."""
    group = NeuronGroup(
        n_cells,
        model=_LIF_WITH_EI_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=V1PV_REFRACTORY,
        method="euler",
        name=name,
        namespace={
            "V_th": V1PV_VT, "V_reset": V1PV_VR,
            "tau_e": TAU_E, "tau_i": TAU_I,
        },
    )
    group.V = V1PV_EL
    group.I_e = 0 * pA
    group.I_i = 0 * pA
    group.I_bias = 0 * pA
    group.C = V1PV_C
    group.gL = V1PV_GL
    group.EL = V1PV_EL
    return group


# --- V1 inhibitory SOM (Martinotti / slow) ----------------------------------

V1SOM_C = 0.15 * nF
V1SOM_GL = 10.0 * nS
V1SOM_EL = -65.0 * mV
V1SOM_VT = -50.0 * mV
V1SOM_VR = -65.0 * mV
V1SOM_REFRACTORY = 2.0 * ms


def make_v1_som_population(n_cells: int, name: str = "v1_som") -> NeuronGroup:
    """V1 SOM Martinotti: LIF, tau_m ~15 ms (Urban-Ciecko & Barth 2016)."""
    group = NeuronGroup(
        n_cells,
        model=_LIF_WITH_EI_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=V1SOM_REFRACTORY,
        method="euler",
        name=name,
        namespace={
            "V_th": V1SOM_VT, "V_reset": V1SOM_VR,
            "tau_e": TAU_E, "tau_i": TAU_I,
        },
    )
    group.V = V1SOM_EL
    group.I_e = 0 * pA
    group.I_i = 0 * pA
    group.I_bias = 0 * pA
    group.C = V1SOM_C
    group.gL = V1SOM_GL
    group.EL = V1SOM_EL
    return group


# --- H excitatory (prior / memory ring) --------------------------------------
#
# H_E carries the context / prediction bump. Needs a slow recurrent channel
# (NMDA with Mg2+ block) so that a cue-evoked bump can persist 200-500 ms
# after cue offset, matching the working-memory attractor regime of Wang
# (2001) Trends Neurosci 24:455 and Compte et al. (2000) Cereb Cortex 10:910.
# AMPA-only recurrence (tau_e=5 ms) cannot sustain a bump at biologically
# plausible firing rates.
#
# Three synaptic channels here:
#   I_e       : AMPA-like fast excitatory (tau_e=5 ms), receives cue + recurrent AMPA.
#   I_nmda    : NMDA slow excitatory with Mg2+ block (tau_nmda_h=100 ms).
#   I_i       : GABA-A inhibitory (tau_i=10 ms).
#
# Recurrent H E->E synapses deposit into BOTH I_e (AMPA) and g_nmda_h (NMDA),
# at the Wang 2001 ratio (NMDA ~60-80 pct of integrated recurrent
# conductance). Cue / feedforward inputs stay AMPA-only.

HE_C = 0.2 * nF
HE_GL = 10.0 * nS
HE_EL = -70.0 * mV
HE_VT = -50.0 * mV
HE_VR = -65.0 * mV
HE_REFRACTORY = 2.0 * ms

_HE_NMDA_EQS = Equations(
    """
    dV/dt = (gL*(EL - V) + I_bias + I_e + I_nmda - I_i) / C : volt (unless refractory)
    dI_e/dt = -I_e / tau_e : amp
    dI_i/dt = -I_i / tau_i : amp
    dg_nmda_h/dt = -g_nmda_h / tau_nmda_h : siemens
    s_nmda = 1 / (1 + exp(-0.062 * V/mV) / 3.57) : 1
    I_nmda = g_nmda_h * s_nmda * (V_nmda_rev - V) : amp
    I_bias : amp
    C : farad
    gL : siemens
    EL : volt
    """
)


def make_h_e_population(n_cells: int, name: str = "h_e") -> NeuronGroup:
    """H excitatory LIF + NMDA slow recurrent (Wang 2001 bump-attractor).

    Postsynaptic variables used by the wiring layer:

    - ``I_e_post``    : AMPA-like fast excitatory current (cue + recurrent
      AMPA co-release).
    - ``g_nmda_h_post`` : NMDA gating conductance (recurrent NMDA co-release
      only). Mg2+ block applied algebraically via ``s_nmda``.
    - ``I_i_post``    : GABA-A inhibitory current (inh pool).
    - ``I_bias``      : static DC bias (calibrated in Stage-0).

    Parameters
    ----------
    n_cells : int
    name : str

    Returns
    -------
    NeuronGroup
        H_E at rest, zero synaptic currents, zero NMDA conductance.
    """
    group = NeuronGroup(
        n_cells,
        model=_HE_NMDA_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=HE_REFRACTORY,
        method="euler",
        name=name,
        namespace={
            "V_th": HE_VT, "V_reset": HE_VR,
            "tau_e": TAU_E, "tau_i": TAU_I,
            "tau_nmda_h": TAU_NMDA_H,
            "V_nmda_rev": V_NMDA_REV,
        },
    )
    group.V = HE_EL
    group.I_e = 0 * pA
    group.I_i = 0 * pA
    group.I_bias = 0 * pA
    group.g_nmda_h = 0 * nS
    group.C = HE_C
    group.gL = HE_GL
    group.EL = HE_EL
    return group


# --- H inhibitory (balancing interneurons) ----------------------------------

HINH_C = 0.1 * nF
HINH_GL = 15.0 * nS
HINH_EL = -65.0 * mV
HINH_VT = -50.0 * mV
HINH_VR = -65.0 * mV
HINH_REFRACTORY = 1.0 * ms


def make_h_inh_population(n_cells: int, name: str = "h_inh") -> NeuronGroup:
    """H inhibitory LIF (generic; not split into PV/SOM in the prototype)."""
    group = NeuronGroup(
        n_cells,
        model=_LIF_WITH_EI_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=HINH_REFRACTORY,
        method="euler",
        name=name,
        namespace={
            "V_th": HINH_VT, "V_reset": HINH_VR,
            "tau_e": TAU_E, "tau_i": TAU_I,
        },
    )
    group.V = HINH_EL
    group.I_e = 0 * pA
    group.I_i = 0 * pA
    group.I_bias = 0 * pA
    group.C = HINH_C
    group.gL = HINH_GL
    group.EL = HINH_EL
    return group


# --- Cue input (Poisson driver) ---------------------------------------------

CUE_DEFAULT_RATE = 40.0 * Hz


def make_cue_input_population(n_cues: int, rate=CUE_DEFAULT_RATE,
                              name: str = "cue_in") -> PoissonGroup:
    """Cue Poisson input. Caller updates `.rates` as trial progresses."""
    return PoissonGroup(n_cues, rates=rate, name=name)


# --- Self-check / smoke --------------------------------------------------------

if __name__ == "__main__":
    from brian2 import Network, SpikeMonitor, prefs, defaultclock, seed as b2_seed, nA
    import numpy as np
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms
    b2_seed(0); np.random.seed(0)

    v1_e = make_v1_e_population(8, name="smoke_v1_e")
    v1_pv = make_v1_pv_population(4, name="smoke_v1_pv")
    v1_som = make_v1_som_population(4, name="smoke_v1_som")
    h_e = make_h_e_population(8, name="smoke_h_e")
    h_inh = make_h_inh_population(4, name="smoke_h_inh")
    cue = make_cue_input_population(3, name="smoke_cue")
    print(f"neurons smoke: v1_e={len(v1_e)} v1_pv={len(v1_pv)} "
          f"v1_som={len(v1_som)} h_e={len(h_e)} h_inh={len(h_inh)} "
          f"cue={len(cue)}")

    # Runtime sanity: V1_E fires at a finite rate under tonic I_bias, SFA
    # rate-limits. I_bias replaces the old I_bu.
    v1_e.I_bias = 300 * pA
    mon = SpikeMonitor(v1_e)
    net = Network(v1_e, mon)
    net.run(500 * ms)
    rate = mon.num_spikes / (8 * 0.5)
    assert rate > 0, "V1_E must spike under strong tonic bias"
    assert rate < 100, "V1_E must not run away under 300 pA (SFA caps rate)"
    print(f"neurons smoke: V1_E tonic-bias rate = {rate:.2f} Hz")
    print("neurons smoke: PASS")
