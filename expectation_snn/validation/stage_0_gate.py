"""Stage 0 calibration gate.

Checks:
- V1 E rates in 2-8 Hz (Niell & Stryker 2008).
- Stable responses; no runaway; no silent layer > 50 ms.
- Tuning FWHM ~30-60 deg.
- H baseline has no persistent bump in absence of input.
- g_total produces sub-threshold apical modulation (does not force V1 E spikes alone).

Pass requires 3 seeds sign-consistent 3/3.
"""
