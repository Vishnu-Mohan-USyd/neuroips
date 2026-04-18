"""Neuron populations for the expectation_snn V1 <-> H laminar prototype.

Six factory functions, one per population, each returning a Brian2 NeuronGroup
preconfigured with the equations, thresholds, resets, and initial state for
that cell type. The heavy lifting is in the module-level parameter block — every
constant carries units (Brian2 physical units) and an inline citation to the
biological range.

Design decisions
----------------
- V1 excitatory pyramidal cells carry a simplified two-compartment model:
  `V_soma` receives bottom-up stimulus drive and evokes spikes; `V_ap` is a
  passive apical integrator receiving top-down feedback. Apical current feeds
  the soma only through a sigmoidal gate `g_ap_soma * sigmoid((V_ap-V_ap_th)/V_ap_s)`,
  so apical input is sub-threshold on its own (apical spikes would require the
  full two-compartment BAC burst we defer to a later phase). The scaling follows
  the intent of Larkum's "two-layer" detector idea without the NMDA Mg2+ block.
- Spike-frequency adaptation uses the Brette & Gerstner 2005 AdEx adaptation
  current: `dw_adapt/dt = (a*(V_soma - E_L) - w_adapt)/tau_adapt`, with a jump
  `w_adapt += b_adapt` on each spike. We stop short of the AdEx exponential
  upswing — LIF + adaptation is sufficient for the Stage 0 rate/FWHM calibration
  and keeps the per-cell state small. Upgrade path to full AdEx is reserved.
- Inhibitory populations (PV, SOM) are single-compartment LIF. PV has fast
  membrane (tau_m ~5 ms, Hu 2014); SOM has slower membrane (tau_m ~15 ms,
  Urban-Ciecko & Barth 2016) matching their cortical roles.
- H excitatory cells have a strong self-recurrent option (built in v1_ring.py /
  h_ring.py, not here) to sustain persistent bumps; the neuron itself is a
  plain LIF. Apical is unnecessary in H for the current plan.
- The cue input group is a Brian2 PoissonGroup — it is the dendrite-free input
  driver for H during cue presentation.

All parameters are exposed as module-level constants. Callers are expected to
tune them (most notably `g_ap_soma`, balance-sweep gain constants) via the
plan's Stage 0 calibration, not by editing this file.

References
----------
- Brette R, Gerstner W (2005) J Neurophysiol 94:3637. Adaptive exponential
  integrate-and-fire model.
- Hu H, Gan J, Jonas P (2014) Science 345:1255263. PV+ fast-spiking basket cells.
- Urban-Ciecko J, Barth AL (2016) Nat Rev Neurosci 17:401. Somatostatin
  interneurons.
- Niell CM, Stryker MP (2008) J Neurosci 28:7520. Mouse V1 awake firing rates.
- Larkum ME (2013) Trends Neurosci 36:141. Cellular two-layer hypothesis.
- Wang X-J (2001) Trends Neurosci 24:455. Synaptic reverberation / persistent
  activity in cortex.
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


# --- V1 excitatory pyramidal (LIF soma + passive apical + SFA) --------------

V1E_C_SOMA = 0.2 * nF           # somatic capacitance
V1E_GL_SOMA = 10.0 * nS         # somatic leak conductance (tau_m ~20 ms)
V1E_EL = -70.0 * mV             # leak reversal
V1E_VT = -50.0 * mV             # spike threshold
V1E_VR = -65.0 * mV             # reset voltage
V1E_REFRACTORY = 2.0 * ms       # absolute refractory
# Apical compartment (passive RC, no spiking of its own)
V1E_C_AP = 0.1 * nF
V1E_GL_AP = 4.0 * nS            # tau_ap = 25 ms: integrates top-down slowly
V1E_EL_AP = -70.0 * mV
# Sigmoidal apical-to-soma coupling
V1E_G_AP_SOMA = 2.0 * nS        # peak effective conductance
V1E_V_AP_TH = -55.0 * mV        # half-activation voltage of apical sigmoid
V1E_V_AP_SLOPE = 4.0 * mV       # slope factor
# Spike-frequency adaptation (Brette & Gerstner 2005)
V1E_A_ADAPT = 0.0 * nS          # subthreshold adaptation (0 = pure spike-triggered)
V1E_B_ADAPT = 30.0 * pA         # spike-triggered jump
V1E_TAU_ADAPT = 150.0 * ms      # adaptation decay (pyramidal range 100-200 ms)

_V1E_EQS = Equations(
    """
    dV_soma/dt = (
        gL_soma*(EL_soma - V_soma)
        + I_bu + I_rec + I_ap - w_adapt
    ) / C_soma : volt (unless refractory)
    dV_ap/dt = (
        gL_ap*(EL_ap - V_ap) + I_td
    ) / C_ap : volt
    I_ap = g_ap_soma * 1.0 / (1.0 + exp(-(V_ap - V_ap_th)/V_ap_slope)) * (V_soma - EL_soma) : amp
    dw_adapt/dt = (a_adapt*(V_soma - EL_soma) - w_adapt) / tau_adapt : amp
    I_bu : amp
    I_rec : amp
    I_td : amp
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

    Parameters
    ----------
    n_cells : int
        Number of V1 excitatory cells.
    name : str
        Brian2 group name; must be unique within the enclosing Network.

    Returns
    -------
    NeuronGroup
        V1_E population, resting at EL_soma, apical resting at EL_ap,
        adaptation at zero.

    Notes
    -----
    Bottom-up stimulus drive goes into `I_bu`, recurrent drive into `I_rec`,
    top-down feedback into `I_td`. Downstream synapse models are expected to
    write into these current variables (summed by Brian2 internally).
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
        },
    )
    group.V_soma = V1E_EL
    group.V_ap = V1E_EL_AP
    group.w_adapt = 0.0 * pA
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
V1PV_GL = 20.0 * nS             # tau_m ~5 ms (Hu et al. 2014)
V1PV_EL = -65.0 * mV
V1PV_VT = -50.0 * mV
V1PV_VR = -65.0 * mV
V1PV_REFRACTORY = 1.0 * ms

_V1PV_EQS = Equations(
    """
    dV/dt = (gL*(EL - V) + I_syn) / C : volt (unless refractory)
    I_syn : amp
    C : farad
    gL : siemens
    EL : volt
    """
)


def make_v1_pv_population(n_cells: int, name: str = "v1_pv") -> NeuronGroup:
    """Build the V1 PV (fast-spiking basket) population.

    Single-compartment LIF with fast membrane (tau_m ~5 ms). Synaptic drive
    arrives via `I_syn`. Reference: Hu, Gan & Jonas 2014.
    """
    group = NeuronGroup(
        n_cells,
        model=_V1PV_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=V1PV_REFRACTORY,
        method="exact",
        name=name,
        namespace={"V_th": V1PV_VT, "V_reset": V1PV_VR},
    )
    group.V = V1PV_EL
    group.C = V1PV_C
    group.gL = V1PV_GL
    group.EL = V1PV_EL
    return group


# --- V1 inhibitory SOM (Martinotti / slow) ----------------------------------

V1SOM_C = 0.15 * nF
V1SOM_GL = 10.0 * nS            # tau_m ~15 ms (Urban-Ciecko & Barth 2016)
V1SOM_EL = -65.0 * mV
V1SOM_VT = -50.0 * mV
V1SOM_VR = -65.0 * mV
V1SOM_REFRACTORY = 2.0 * ms

_V1SOM_EQS = _V1PV_EQS


def make_v1_som_population(n_cells: int, name: str = "v1_som") -> NeuronGroup:
    """Build the V1 SOM (Martinotti) population.

    Single-compartment LIF with slower membrane (tau_m ~15 ms). Targets apical
    dendrites of V1_E in the circuit wiring (in feedback_routes.py), not here.
    """
    group = NeuronGroup(
        n_cells,
        model=_V1SOM_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=V1SOM_REFRACTORY,
        method="exact",
        name=name,
        namespace={"V_th": V1SOM_VT, "V_reset": V1SOM_VR},
    )
    group.V = V1SOM_EL
    group.C = V1SOM_C
    group.gL = V1SOM_GL
    group.EL = V1SOM_EL
    return group


# --- H excitatory (prior / memory ring) --------------------------------------

HE_C = 0.2 * nF
HE_GL = 10.0 * nS               # tau_m ~20 ms (pyramidal range)
HE_EL = -70.0 * mV
HE_VT = -50.0 * mV
HE_VR = -65.0 * mV
HE_REFRACTORY = 2.0 * ms

_HE_EQS = Equations(
    """
    dV/dt = (gL*(EL - V) + I_rec + I_cue + I_fb) / C : volt (unless refractory)
    I_rec : amp
    I_cue : amp
    I_fb : amp
    C : farad
    gL : siemens
    EL : volt
    """
)


def make_h_e_population(n_cells: int, name: str = "h_e") -> NeuronGroup:
    """Build the H excitatory population.

    Single-compartment LIF with three current inputs: `I_rec` for intra-H
    recurrence (the substrate of persistent bumps, wired with E->E STDP in
    h_ring.py), `I_cue` for cue-group drive, `I_fb` for top-down feedback
    returning from V2/other. Reference: Wang 2001 for persistent-activity
    substrate; Compte et al. 2000 for ring-attractor version.
    """
    group = NeuronGroup(
        n_cells,
        model=_HE_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=HE_REFRACTORY,
        method="exact",
        name=name,
        namespace={"V_th": HE_VT, "V_reset": HE_VR},
    )
    group.V = HE_EL
    group.C = HE_C
    group.gL = HE_GL
    group.EL = HE_EL
    return group


# --- H inhibitory (balancing interneurons) ----------------------------------

HINH_C = 0.1 * nF
HINH_GL = 15.0 * nS             # tau_m ~7 ms, between PV and SOM; generic inh
HINH_EL = -65.0 * mV
HINH_VT = -50.0 * mV
HINH_VR = -65.0 * mV
HINH_REFRACTORY = 1.0 * ms

_HINH_EQS = _V1PV_EQS


def make_h_inh_population(n_cells: int, name: str = "h_inh") -> NeuronGroup:
    """Build the H inhibitory population.

    Single-compartment LIF used for feedback inhibition on H_E and global
    balance of the ring. Not split into PV/SOM for the prototype.
    """
    group = NeuronGroup(
        n_cells,
        model=_HINH_EQS,
        threshold="V > V_th",
        reset="V = V_reset",
        refractory=HINH_REFRACTORY,
        method="exact",
        name=name,
        namespace={"V_th": HINH_VT, "V_reset": HINH_VR},
    )
    group.V = HINH_EL
    group.C = HINH_C
    group.gL = HINH_GL
    group.EL = HINH_EL
    return group


# --- Cue input (Poisson driver) ---------------------------------------------

CUE_DEFAULT_RATE = 40.0 * Hz    # per-channel on-rate during cue presentation


def make_cue_input_population(n_cues: int, rate=CUE_DEFAULT_RATE, name: str = "cue_in") -> PoissonGroup:
    """Build the cue input Poisson group.

    Each cue channel fires at a configurable rate during its on-window; the
    caller (stimulus.py) is expected to update `rates` as the trial progresses
    (ramp up during cue, ramp down after). Reference: pattern matches Brian2
    `PoissonGroup` docs and is the standard cortical afferent stand-in used
    across rate-coded cue paradigms.
    """
    return PoissonGroup(n_cues, rates=rate, name=name)


# --- Self-check / smoke --------------------------------------------------------

if __name__ == "__main__":
    # Minimal construction check: build one of each, verify counts + no errors.
    v1_e = make_v1_e_population(8, name="smoke_v1_e")
    v1_pv = make_v1_pv_population(4, name="smoke_v1_pv")
    v1_som = make_v1_som_population(4, name="smoke_v1_som")
    h_e = make_h_e_population(8, name="smoke_h_e")
    h_inh = make_h_inh_population(4, name="smoke_h_inh")
    cue = make_cue_input_population(3, name="smoke_cue")
    print(f"neurons smoke: v1_e={len(v1_e)} v1_pv={len(v1_pv)} v1_som={len(v1_som)} "
          f"h_e={len(h_e)} h_inh={len(h_inh)} cue={len(cue)}")
    print("neurons smoke: PASS")
