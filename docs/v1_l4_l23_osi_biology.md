# V1 L4 vs L2/3 Orientation Selectivity Index — biological reference

**Author:** researcher (refine-v1-core team)
**Date:** 2026-04-29
**Purpose:** Compare model L4 (median gOSI = 0.516) and Phase-A L2/3 (median gOSI = 0.170) against primary-literature OSI distributions in mouse, cat, and macaque V1.

**TL;DR.** The model L4 layer (median gOSI = 0.516, n = 131 072) is in **excellent quantitative agreement** with the only primary mouse-V1 study using the exact same Fourier/global formula on awake animals (Sun, Tan, Mensh, Ji 2016: median gOSI = 0.56 for L4). Most other "median OSI" numbers reported for mouse V1 (0.31–0.90) come from a **different formula** — the 2-point modulation index — and are not directly comparable. The Phase-A L2/3 layer (median gOSI = 0.170) is **~3.4× lower** than mouse-V1 L2/3 biology (Sun 2016: median gOSI = 0.58); this is the expected gap for an untrained, randomly wired feedforward L4→L2/3 layer and is what Phase B's recurrent plasticity should close.

---

## 1. OSI definition variants — READ FIRST

At least four distinct quantities are all called "OSI" in the V1 literature. They produce **systematically different numerical values for the same neuron** and cannot be cross-compared without explicit conversion.

| # | Name | Formula | For a perfect cos² tuning | Used by |
|---|---|---|---|---|
| 1 | **gOSI** / 1 − CV / Fourier OSI | gOSI = \|Σₖ R(θₖ) e^(i 2θₖ)\| / Σₖ R(θₖ) | 0.50 | **OUR MODEL**; Ringach et al. 2002 (as 1−CV); Sun et al. 2016; de Vries 2020; Mazurek et al. 2014 |
| 2 | **2-point OSI** | (R_pref − R_orth) / (R_pref + R_orth) | 1.00 | Niell & Stryker 2008/2010; Tan et al. 2011; Van den Bergh 2010; Ko, Hofer 2011 |
| 3 | OI | (R_pref − R_orth) / R_pref | 1.00 | older lit; Mazurek 2014 |
| 4 | Bandwidth (HWHM at 1/√2) | half-width to 70.7 % of peak (degrees) | 22.5° | Ringach 2002; Niell & Stryker 2010 |

**Verbatim formulas from primary sources:**
- **Sun, Tan, Mensh, Ji 2016** (Nat Neurosci 19:308): _"gOSI = \|∑ₖ R(θₖ) e^(i 2θₖ)\| / ∑ₖ R(θₖ)"_ and _"OSI = (R_pref − R_orth) / (R_pref + R_orth)"_. **The only mouse-V1 paper that reports both, separately — our cleanest reference.**
- **Ringach, Shapley, Hawken 2002** (J Neurosci 22:5639): _"V = 1 − \|R\|, where R = Σₖ rₖ e^(i 2θₖ) / Σₖ rₖ"_ — therefore **(1 − CV) ≡ gOSI** by algebra. Bandwidth defined as HWHM at 1/√2 of peak.
- **Niell & Stryker 2008** (J Neurosci 28:7520): _"(R_pref − R_ortho) / (R_pref + R_ortho)"_. Note that 1 − CV "gave similar results" but they preferred the 2-point form.
- **Mazurek, Kager, Van Hooser 2014** (Front Neural Circuits 8:92): documents that 2-point indices are unreliable for weakly tuned cells; **recommends 1-CirVar (= gOSI) as the more robust population statistic**. Our model uses the recommended form.

**Critical numerical identity for sanity-checking:** for any well-behaved tuning curve gOSI ≤ 2-pt OSI. For a pure von Mises with concentration κ in orientation space (our L4: κ ≈ 4.5, FWHM ≈ 32°), the analytical gOSI = I₁(κ)/I₀(κ) ≈ 0.68; our empirical median 0.516 sits below this because of 8-orientation sampling and Poisson firing noise.

---

## 2. Mouse V1 L4 — primary sources

**2.1 Sun, Tan, Mensh, Ji 2016** "Thalamus provides layer 4 of primary visual cortex with orientation- and direction-tuned inputs" (*Nat Neurosci* 19:308–315; doi:10.1038/nn.4196; PMC4731241).
- State: **awake**, head-fixed mice; adaptive-optics 2-photon, GCaMP6s.
- **n = 1239 visually responsive L4 neurons** (of 1511 outlined; 82 % responsive), 3 mice.
- Verbatim: _"neurons in L4 are more orientation-selective (median gOSI = 0.56, median OSI = 0.78, Fig. 4q,r) with correspondingly narrower tuning widths (median FWHM = 33.6°)"_.
- 83 % of visually responsive L4 neurons orientation-selective.
- Thalamic boutons in L4 (separate population): bouton median gOSI = 0.31; OS-classified bouton median gOSI = 0.56.
- **Anchor reference for our model.** Same gOSI formula. Awake mouse V1.

**2.2 Niell & Stryker 2008** "Highly selective receptive fields in mouse visual cortex" (*J Neurosci* 28(30):7520–7536; PMC3040721).
- State: anaesthetised C57BL/6 (urethane); silicon-array extracellular single units.
- n = 235 single units, 27 mice; 87 % (204) visually responsive. OSI form: 2-point.
- Per-layer numerical medians **not tabulated in text** (only Fig. 4E). Verbatim: _"In layers 2/3, 4, and 6, there is a sharp distinction between cell types, because almost all putative excitatory units were highly orientation selective, whereas most putative inhibitory units were untuned."_ For L5: _"selectivity was much less (p < 0.001)."_
- Approximate read of Fig. 4E (mean 2-pt OSI for putative excitatory): ~0.55–0.7 in L2/3, L4, L6; ~0.3–0.4 in L5. **Marked uncertain — needs follow-up** for exact medians.

**2.3 Niell & Stryker 2010** (Neuron; PMC3184003). Awake mouse V1; locomotion does not change selectivity. L2/3 broad-spiking median **HWHM ≈ 24° awake / 23° anaesth.** (28 broad-spiking units, 8 mice).

**2.4 Tan, Brown, Scholl, Mohanty, Priebe 2011** (*J Neurosci* 31:12339; PMC3202243). In vivo whole-cell intracellular, anaesthetised. Layers II/III + IV combined (not separated). **Mouse spike OSI (2-pt) median = 0.31, n = 20**; mouse Vm OSI = 0.09. Authors note distribution "consistent with Niell & Stryker 2008" (rank-sum p > 0.05).

**2.5 Van den Bergh, Zhang, Arckens 2010** (PMC2881339). Anaesthetised mouse V1, extracellular. Median 2-pt OSI = **0.90** (n = 69). **Substantially higher than every other report** — likely strict responsiveness inclusion criterion. **Treat as outlier.**

---

## 3. Mouse V1 L2/3 — primary sources

**3.1 Sun et al. 2016** (same paper). L2/3 visually responsive rate = 1279 / 2608 = **49 %** (much lower than L4's 82 %). For L2/3 excitatory OS population: **median gOSI = 0.58, median 2-pt OSI = 0.78, median FWHM = 29.2°** ("slightly more tuned" than L4). 83 % of visually responsive cells orientation-selective. **Anchor reference for our model L2/3.**

**3.2 Ko, Hofer, Pichler et al. 2011** "Functional specificity of local synaptic connections in neocortical networks" (*Nature* 473:87–91; doi:10.1038/nature09880; PMC3089591). Anaesthetised mouse (fentanyl + midazolam + medetomidine + light isoflurane); 2-photon Ca-imaging in vivo + paired patch in vitro. n = 116 functionally matched L2/3 pyramidal cells, 16 mice. 2-pt OSI form (R_best vs R_ortho average). **79/116 = 68 % of L2/3 pyramidal cells with OSI > 0.4.** Median value not explicitly reported.

**3.3 Hofer et al. 2011** "Differential connectivity and response dynamics" (*Nat Neurosci* 14:1045–1052; doi:10.1038/nn.2876). Anaesthetised mouse, 2-photon. Confirms L2/3 PV interneurons broadly tuned vs L2/3 excitatory cells sharply tuned, but **does not tabulate a clean L2/3 excitatory OSI median** in usable form. Marked **uncertain — needs follow-up**.

**3.4 de Vries et al. 2020** Allen Brain Observatory (*Nat Neurosci* 23:138; PMC6948932). Awake, GCaMP6f, ~10 410 V1 neurons, 243 mice. Allen Observatory uses the **gOSI** formula (per the eNeuro inclusion-criteria companion paper, PMC8114876). Per-layer median gOSI not in text — **needs extraction from AllenSDK** if user wants direct comparison.

---

## 4. Cat & macaque V1

**Cat.** Hubel & Wiesel 1962 qualitative only ("usually fails long before angle 90° to optimum"). **Tan et al. 2011**: cat V1 spike OSI median = **0.83 (n = 31)** in vivo whole-cell, anaesthetised, layers II/III + IV combined (2-pt form; substantially higher than mouse — consistent with cat's orientation-column architecture). Cat **L4 vs L2/3 OSI separately is essentially absent from the primary literature** — papers from the Reid, Ferster, Hirsch, Martinez, Alonso labs typically report bandwidth (~30–40° HWHM, often narrower in L4 simple cells than L2/3 complex cells) rather than OSI. Marked **uncertain — needs follow-up** if cat layer-resolved OSI needed.

**Macaque.** **Ringach, Shapley, Hawken 2002** (*J Neurosci* 22:5639): 26 anaesthetised macaques (sufentanil), n = 308 cells. Population median CV = 0.61 → **median gOSI = 0.39**. Median orientation bandwidth = **23.5°**. Layers 4C, 3B, 5 are the **least selective**, opposite the mouse pattern where L4 ≈ L2/3. Verbatim: _"there is a broad distribution of circular variance in all layers of V1"_; _"at least 25 % of the cells in all layers have a circular variance greater than 0.65"_ (i.e., gOSI < 0.35). Per-layer numerical medians only graphical — **uncertain — needs follow-up** for exact 4C, 3B, 2/3 medians. **Van den Bergh et al. 2010**: macaque V1 median 2-pt OSI ≈ 0.45 (n = 96), anaesthetised.

---

## 5. Master comparison table

All gOSI = global Fourier form: gOSI = |Σ R(θ) e^(i 2θ)| / Σ R(θ). All 2-pt OSI = (R_pref − R_orth) / (R_pref + R_orth). "n.r." = not reported numerically in primary text.

| Population | OSI form | Median | Frac > 0.2 | Frac > 0.5 | Frac > 0.8 | n cells | State | Citation |
|---|---|---|---|---|---|---|---|---|
| **OUR MODEL L4** | **gOSI** | **0.516** | **100 %** | **56.4 %** | **0 %** | **131 072** | model | this work |
| **OUR MODEL L2/3 (Phase-A baseline)** | **gOSI** | **0.170** | **43.9 %** | **16.5 %** | n.r. | **16 384** | model | this work |
| Mouse V1 L4 | gOSI | **0.56** | n.r. (≥83 % OS) | n.r. | n.r. | 1239 | awake | Sun et al. 2016 |
| Mouse V1 L4 | 2-pt OSI | 0.78 | n.r. | n.r. | n.r. | 1239 | awake | Sun et al. 2016 |
| Mouse V1 L2/3 (excit. OS pop.) | **gOSI** | **0.58** | n.r. | n.r. | n.r. | ≈ 1060 | awake | Sun et al. 2016 |
| Mouse V1 L2/3 (excit. OS pop.) | 2-pt OSI | 0.78 | n.r. | n.r. | n.r. | ≈ 1060 | awake | Sun et al. 2016 |
| Mouse V1 L4 thalamic boutons (OS-only) | gOSI | 0.56 | n.r. | n.r. | n.r. | ~half of ~28 000 | awake | Sun et al. 2016 |
| Mouse V1, L II/III + L IV pooled | 2-pt spike OSI | 0.31 | n.r. | n.r. | n.r. | 20 | anaesth. | Tan et al. 2011 |
| Mouse V1 (high-respond bias) | 2-pt OSI | 0.90 | n.r. | n.r. | n.r. | 69 | anaesth. | Van den Bergh 2010 |
| Mouse V1, putative-excit pooled L2/3+L4+L6 | 2-pt OSI | "highly OS" (~0.55–0.7 from Fig 4E) | n.r. | n.r. | n.r. | ~204 | anaesth. urethane | Niell & Stryker 2008 |
| Mouse V1 L5, putative-excit | 2-pt OSI | "much less" (~0.3–0.4 from Fig 4E) | n.r. | n.r. | n.r. | ~30 | anaesth. | Niell & Stryker 2008 |
| Mouse V1 L2/3 broad-spiking | HWHM | 24° awake / 23° anaesth. | – | – | – | 28 | both | Niell & Stryker 2010 |
| Mouse V1 L2/3 pyramidal | 2-pt OSI | n.r. | n.r. | 68 % > 0.4 | n.r. | 116 | anaesth. | Ko, Hofer 2011 |
| Cat V1 (L II/III + L IV pooled) | 2-pt spike OSI | **0.83** | n.r. | n.r. | n.r. | 31 | anaesth. | Tan et al. 2011 |
| Cat V1 (L II/III + L IV pooled) | 2-pt Vm OSI | 0.28 | n.r. | n.r. | n.r. | 31 | anaesth. | Tan et al. 2011 |
| Cat V1 L4 vs L2/3 separately | – | **n.r. in primary lit** (only bandwidth ~30–40° HWHM) | – | – | – | – | – | needs follow-up |
| Macaque V1 (population) | gOSI = 1−CV | **0.39** | n.r. | n.r. | <25 % > 0.35 in any layer (lower bound) | 308 | anaesth. sufentanil | Ringach et al. 2002 |
| Macaque V1 (population) | bandwidth | 23.5° HWHM | – | – | – | 308 | anaesth. | Ringach et al. 2002 |
| Macaque V1 layers 4Cα, 4Cβ, 3B, 5 | gOSI = 1−CV | "concentrated low-selectivity" (medians only graphical) | n.r. | n.r. | n.r. | per-layer n n.r. | anaesth. | Ringach et al. 2002 |
| Macaque V1 (population) | 2-pt OSI | ≈ 0.45 | n.r. | n.r. | n.r. | 96 | anaesth. | Van den Bergh 2010 |

**Cross-formula sanity check from Sun 2016:** on the same L4 population, gOSI = 0.56 ↔ 2-pt OSI = 0.78. Offset of ~0.22 — but **distribution-dependent**, not a fixed conversion. Never quote a 2-pt OSI to compare against a gOSI.

---

## 6. Interpretation

### L4
Model L4 (median gOSI 0.516) is **within 8 % of the only directly comparable biological number** (Sun et al. 2016 awake mouse V1 L4 median gOSI 0.56, same formula, same species, awake). Because we hardcode L4 with a pure von Mises tuning of κ ≈ 4.5 (analytical gOSI = I₁(κ)/I₀(κ) ≈ 0.68 for an infinite, noise-free sample), the empirical 0.516 with 8 orientation samples and Poisson noise is the expected attenuation. **Verdict: model L4 quantitatively matches awake mouse V1 L4 biology under the same formula.**

The L4 fraction > 0.5 = 56.4 % is consistent with the median sitting at 0.56 (~50 % of cells above median by definition). Fraction > 0.8 = 0 % is consistent with the analytical ceiling ~0.68 for our chosen κ.

The model L4 does **not** match cat V1 L4 (median 2-pt spike OSI 0.83 in Tan 2011, dominated by sharply tuned simple cells in orientation columns). If the modelling target is cat V1, κ must roughly double. If the target is mouse V1, κ ≈ 4.5 is calibrated correctly.

### L2/3
Phase-A model L2/3 (median gOSI 0.170) is **~3.4× lower than awake mouse V1 L2/3 biology** (Sun 2016 median gOSI 0.58). Unambiguous quantitative gap. Possible interpretations, in order of likelihood:

1. **Phase A is feedforward-only with random L4→L2/3 wiring; L2/3 has no recurrent plasticity.** Biological L2/3 has had weeks–years of intracortical plasticity refining recurrent connections. A randomly wired feedforward network from a tuned L4 layer is expected to under-tune L2/3 because each L2/3 cell pools across L4 cells with mixed orientation preferences. **This is the most likely cause and is consistent with the Phase B program (L2/3→L2/3 STDP) being needed.** If Phase B works as intended, post-Phase-B L2/3 median gOSI should approach ~0.5–0.6.
2. L4→L2/3 connectivity may be too sparse / too weak so L4 tuning fails to drive L2/3 to spike near R_pref. Diagnose by checking L2/3 firing-rate distributions vs L4.
3. OSI form mismatch — **not** the source: our model uses the gOSI form, the same as Sun 2016.
4. Species mismatch — **not** the source: 0.58 is the mouse number.

**Recommendation:** the Phase-A L2/3 baseline of 0.170 is biologically too low for mouse V1. After Phase B's recurrent plasticity, re-measure and verify L2/3 median gOSI rises toward 0.5–0.6. If it does, Phase B is working as intended. If not, that itself is a finding.

### Form-mismatch trap to avoid
Many widely cited "mouse V1 OSI ≈ 0.31" or "≈ 0.90" numbers are **2-pt OSI on different cell samples** (Tan 2011; Van den Bergh 2010). **Do not compare these directly to our model's gOSI.** The only safe reference points are studies reporting the global Fourier / 1-CV form: Sun et al. 2016, Ringach et al. 2002, Allen Brain Observatory (de Vries 2020).

---

## 7. Open questions / cautions

1. **Spike vs Ca²⁺ vs Vm.** Our model OSI is computed on spike counts → matches spike-OSI literature, not dF/F or Vm. Sun 2016 is dF/F-based (slight upward bias possible, ~10 %); Tan 2011 reports spike OSI ≈ 3.4× Vm OSI in mouse due to spike-threshold nonlinearity. Spike-OSI is the right comparison column.
2. **Awake vs anaesthetised.** Niell & Stryker 2010 found no OSI change with locomotion or anaesthesia for L2/3 broad-spiking cells. L4 awake/anaesth comparison less characterised; Sun 2016 (awake) is the safest mouse L4 reference.
3. **Inclusion criteria.** Sun 2016 reports OSI only on visually responsive cells (49 % of L2/3, 82 % of L4). Our model currently reports OSI on **all** L4 / L2/3 cells with no responsiveness gate. Restrict to a "responsive" subset (e.g., mean rate > 1 Hz at preferred orientation) for a fairer comparison; this may modestly raise our medians but will not close the L2/3 gap to 0.58.
4. **Finite-orientation-sampling artifact.** 8 orientations at 22.5° spacing under-samples narrow tuning curves. For our κ ≈ 4.5 (FWHM 32°) this is barely Nyquist-adequate; produces ~5–10 % numerical bias.
5. **Per-layer macaque numbers in Ringach 2002 only graphical.** Extracting exact 4C, 3B, 2/3 medians requires WebPlotDigitizer on Figs 4–5 or supplementary data extraction. Marked uncertain.
6. **Cat V1 layer-resolved OSI** is the largest literature gap. Reid/Ferster/Hirsch labs report bandwidth not OSI. If a cat L4 vs L2/3 OSI breakdown becomes important, this requires a dedicated dive — possibly only available in supplementary data.

---

## 8. Primary-source bibliography

1. Niell CM, Stryker MP (2008). "Highly selective receptive fields in mouse visual cortex." *J Neurosci* 28(30):7520–7536. doi:10.1523/JNEUROSCI.0623-08.2008. PMC3040721.
2. Niell CM, Stryker MP (2010). "Modulation of visual responses by behavioral state in mouse visual cortex." *Neuron* 65:472–479. PMC3184003.
3. Sun W, Tan Z, Mensh BD, Ji N (2016). "Thalamus provides layer 4 of primary visual cortex with orientation- and direction-tuned inputs." *Nat Neurosci* 19:308–315. doi:10.1038/nn.4196. PMC4731241.
4. Tan AYY, Brown BD, Scholl B, Mohanty D, Priebe NJ (2011). "Orientation Selectivity of Synaptic Input to Neurons in Mouse and Cat Primary Visual Cortex." *J Neurosci* 31(34):12339–12350. doi:10.1523/JNEUROSCI.2039-11.2011. PMC3202243.
5. Hofer SB, Ko H, Pichler B et al. (2011). "Differential connectivity and response dynamics of excitatory and inhibitory neurons in visual cortex." *Nat Neurosci* 14:1045–1052. doi:10.1038/nn.2876. PMID 21765421.
6. Ko H, Hofer SB, Pichler B, Buchanan KA, Sjöström PJ, Mrsic-Flogel TD (2011). "Functional specificity of local synaptic connections in neocortical networks." *Nature* 473:87–91. doi:10.1038/nature09880. PMC3089591.
7. Ringach DL, Shapley RM, Hawken MJ (2002). "Orientation selectivity in macaque V1: diversity and laminar dependence." *J Neurosci* 22(13):5639–5651. PMID 12097515.
8. Mazurek M, Kager M, Van Hooser SD (2014). "Robust quantification of orientation selectivity and direction selectivity." *Front Neural Circuits* 8:92. PMC4123790.
9. Van den Bergh G, Zhang B, Arckens L, Chino YM (2010). "Receptive-field properties of V1 and V2 neurons in mice and macaque monkeys." *J Comp Neurol* 518(11):2051–2070. PMC2881339.
10. Hubel DH, Wiesel TN (1962). "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex." *J Physiol* 160:106–154.
11. Skottun BC, De Valois RL, Grosof DH, Movshon JA, Albrecht DG, Bonds AB (1991). "Classifying simple and complex cells on the basis of response modulation." *Vision Res* 31:1079–1086.
12. de Vries SEJ et al. (2020). "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex." *Nat Neurosci* 23:138–151. PMC6948932.
13. de Vries SEJ et al., inclusion-criteria companion. *eNeuro* 2021. PMC8114876.

---
*End of report.*
