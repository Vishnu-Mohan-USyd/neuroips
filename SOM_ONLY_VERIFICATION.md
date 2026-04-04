# SOM-Only Exp C: L2/3 Verification

## Key answer

**SI(0°) is NEGATIVE (facilitation), not positive (suppression).** The SOM-only model did NOT learn dampening. It learned to REDUCE SOM at the predicted orientation — creating center-surround via SOM reduction, not SOM increase.

---

## Why: the learned alpha_inh values are NEGATIVE

All 3 seeds learned **negative** Gaussian α_inh weights:

| Basis | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| G(σ=5°) | **−0.163** | −0.041 | −0.038 |
| G(σ=15°) | **−0.156** | −0.036 | −0.032 |
| G(σ=30°) | **−0.140** | −0.030 | −0.023 |
| G(σ=60°) | **−0.123** | −0.025 | −0.017 |
| MexHat | **−0.179** | −0.054 | −0.053 |
| Const | +0.010 | +0.010 | +0.010 |
| Odd/Sin | +0.038 | +0.059 | −0.011 |
| ‖α‖₁ | **0.810** | 0.255 | 0.184 |

This produces a **negative** K_inh at center (e.g., seed 42: K_inh(0°) = −0.124). Through softplus:
- At predicted (q_centered > 0): inh_field < 0 → softplus(neg) < softplus(0) → **LESS SOM** → facilitation
- At flanks (q_centered < 0): inh_field > 0 → softplus(pos) > softplus(0) → **MORE SOM** → suppression

The SOM-only operator discovered that the optimal strategy is to **reduce inhibition at predicted** rather than increase it.

---

## Softplus bias kills L2/3 at oracle pi=3.0

| pi_eff | SOM drive (max) | L2/3 max (seed 42) | Status |
|---|---|---|---|
| 0.1 | 0.070 | 0.278 | OK |
| 0.3 | 0.210 | 0.241 | OK |
| 0.5 | 0.350 | 0.202 | Reduced |
| 1.0 | 0.701 | 0.106 | Heavily suppressed |
| 1.5 | 1.051 | 0.020 | Nearly dead |
| 2.0 | 1.401 | 0.002 | Dead |
| **3.0** | **2.102** | **0.000** | **Killed** |

The softplus bias (pi × ln2) creates constant SOM that monotonically kills L2/3. With no center_exc pathway to compensate, the SOM-only model is only functional at LOW precision. The model's actual V2 produces very low pi (0.04–0.26) to survive.

---

## L2/3 suppression-by-tuning (at realistic pi)

### Seed 42 (pi=0.50, 2× realistic)

| Δθ | R_on | R_off | Delta | SI |
|---|---|---|---|---|
| 0° | 0.2023 | 0.1983 | +0.0039 | **−2.0% (facilitation)** |
| 5° | 0.2019 | 0.1983 | +0.0036 | −1.8% |
| 10° | 0.2011 | 0.1983 | +0.0028 | −1.4% |
| 15° | 0.2001 | 0.1983 | +0.0017 | −0.9% |
| 20° | 0.1991 | 0.1983 | +0.0008 | −0.4% |
| 30° | 0.1979 | 0.1983 | −0.0005 | +0.2% |
| 45° | 0.1973 | 0.1983 | −0.0010 | **+0.5% (suppression)** |
| 60° | 0.1973 | 0.1983 | −0.0011 | +0.5% |
| 90° | 0.1973 | 0.1983 | −0.0010 | +0.5% |

**Classification: CENTER-SURROUND (SOM-mediated).** Facilitation at predicted, suppression at flanks.

### Seeds 123 and 456 (pi=0.08–0.16)

Effects too small to classify (SI < 0.1% everywhere). The V2 precision is too low and the alpha weights too small to produce measurable L2/3 modulation.

---

## Cross-seed consistency

| Metric | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Realistic V2 pi_eff | 0.258 | 0.041 | 0.080 |
| ‖α_inh‖₁ | 0.810 | 0.255 | 0.184 |
| SI(0°) at test pi | −2.0% (pi=0.5) | −0.1% (pi=0.08) | −0.1% (pi=0.16) |
| Pattern direction | Facilitation | Facilitation | Facilitation |

All seeds show the same DIRECTION (facilitation at center), but seed 42 has ~4× larger weights and produces a measurable effect. Seeds 123/456 have weaker feedback and negligible L2/3 modulation.

---

## Comparison to lead's expectation

| Expected | Observed |
|---|---|
| SI(0°) positive (suppression) | **SI(0°) negative (facilitation)** |
| Dampening (more SOM at predicted) | Center-surround (less SOM at predicted) |
| Pathway gain mismatch gone | Pathway gain mismatch replaced by **sign inversion** of alpha weights |

The model circumvented the "SOM-only = dampening" assumption by learning **negative** α_inh weights. Instead of increasing SOM at the predicted channel, it decreases SOM there — achieving facilitation through reduced inhibition, with suppression at flanks from increased SOM.

---

## Architectural implications

1. **The softplus nonlinearity enables negative alpha to create facilitation.** softplus(negative field) < softplus(0) = constant bias. The model exploits this asymmetry to create an effective "excitatory" signal through SOM reduction.

2. **The SOM-only pathway cannot achieve dampening** (suppression at predicted) because:
   - Positive alpha_inh → positive K_inh → more SOM everywhere (dominated by softplus bias) → global suppression, not orientation-selective dampening
   - The only way to get orientation-selective modulation is negative alpha (less SOM at predicted), which produces facilitation, not suppression

3. **The softplus bias creates a fundamental operating-range problem.** At pi > 1.5, the constant SOM bias kills L2/3 regardless of alpha. The model's V2 learns very low precision (0.04–0.26) to keep L2/3 alive, but this also minimizes the feedback effect magnitude.
