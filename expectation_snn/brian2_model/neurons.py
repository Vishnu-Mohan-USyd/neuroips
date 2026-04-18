"""Neuron models.

Two-variable LIF with V_soma (bottom-up drive) and V_ap (top-down modulatory bias),
spike-frequency adaptation (Brette & Gerstner 2005 form), and LIF inhibitory cells
(SOM, PV). Upgrade path to AdEx / true 2-compartment reserved for Stage 0 if the
LIF + apical variable proves crude.
"""
