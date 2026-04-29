# Cat & primate V1 — like-to-like emergence, layered OSI, L2/3 connectivity, modeling defaults

**Author:** researcher (refine-v1-core team)
**Date:** 2026-04-29
**Target species:** cat (immediate calibration) + macaque (ultimate goal). Mouse cited only as contrast, or where cat/primate data is genuinely absent.

---

## TL;DR

1. **The "random + STDP → like-to-like" story is mouse-specific.** Empirically established in rodent L2/3 (Ko 2013, Ko 2014, Cossell 2015, Lee 2016). Not directly demonstrated in cat or primate, and largely doesn't need to apply: cat/primate V1 has columnar architecture (~50 µm orientation columns inside ~1 mm hypercolumns), so within-column connectivity is automatically iso-orientation **by spatial geometry**, regardless of the developmental rule.
2. **Cat/primate long-range L2/3 horizontals are explicitly iso-orientation** at ≥500 µm scale: Gilbert & Wiesel 1989 (cat, 6–8 mm), Malach et al. 1993 (macaque, ~mm clusters), Bosking et al. 1997 (tree shrew, 2–5 mm, **57.6 % iso-orientation contacts within ±35°**, 4× collinear bias), Stettler et al. 2002 (macaque, V1 intrinsics iso-orientation, V2→V1 feedback unspecific). These are anatomical facts in adult; **whether refined from random is not directly tested in cat/primate.**
3. **Macaque L4C is NOT robustly the least selective layer.** Ringach et al. 2002's pairwise Wilcoxon test was **p > 0.1 for all layer pairs** — no significant inter-layer differences (n = 308). Hong et al. 2024 macaque Neuropixels (n = 661) finds **L4C 2-pt OSI = 0.68, L2/3 = 0.70, "other" = 0.70** — statistically indistinguishable.
4. **Cat L4 simple cells: Vm HWHM ≈ 30–35° (broad), spike OSI sharpened to ≈ 0.83 by spike threshold (the iceberg effect)** (Carandini & Ferster 2000, Anderson et al. 2000). The data-driven Antolík et al. 2024 cat-V1 model targets HWHH = 21.1° (L4 excit) / 25.2° (L2/3 excit).
5. **The user's two requested targets — gOSI ≈ 0.7–0.8 AND HWHM ≈ 20–25° — are mutually inconsistent for a noise-free pure von Mises tuning curve.** For gOSI = 0.7, HWHM = 14.5°; for HWHM = 20°, gOSI ≈ 0.55. Biology resolves this with the iceberg: broad subthreshold Vm, sharp spike output. Recommended: keep κ ≈ 4.0–5.0 (HWHM 14–18°) and add a baseline subtraction in OSI computation, OR accept κ ≈ 4.5 → gOSI ≈ 0.70 ≈ macaque L4C 0.68 (Hong 2024).
6. **L4 → L2/3 connectivity in cat/primate** should respect columnar geometry: the 3 × 3 hypercolumn patch should restrict L4 sampling to iso-orientation L4 cells (within ~±22.5°), matching cat's columnar architecture directly. No STDP refinement of orientation needed.
7. **L2/3 → L2/3 long-range horizontal recurrents** for cat/primate should be **hard-coded iso-orientation patches across hypercolumns** at ≥500 µm scale (Bosking 1997 statistics). Not refined from random — that's the mouse story, the wrong template for the species we model.

---

## §1 Like-to-like pooling — pre-wired or refined from random?

### Mouse (reference only)

- **Ko et al. 2013** (*Nature* 496:96; PMC4843961): connectivity between similarly tuned L2/3 neurons emerges between P13–15 (eye-opening; trend P = 0.092) and P22–26 (significant P = 4.6 × 10⁻⁴). Connection probability for highly correlated pairs: **19.4 % at P13–15 → 41.5 % at P22–26** (χ² P = 0.0035). **Overall connectivity rate ~16–22 % constant — connections REDISTRIBUTE rather than emerge from absent**; functional refinement, not de novo emergence. n = 31 mice, 283 neurons.
- **Ko et al. 2014** (*J Neurosci* 34:9812; PMC4099553): like-to-like emerges **in dark-reared mice** (similar-orientation pairs >2× more likely connected, P = 0.028; n = 6). **Visual experience is not strictly required**; spontaneous activity sufficient.
- **Cossell et al. 2015** (*Nature* 518:399; PMC4843963): heavy-tailed EPSP distribution (median 0.19 mV, mean 0.45 mV); **strongest connections between most-correlated cells**. n = 203 cells, 75 connections, 17 mice.
- **Lee et al. 2016** (*Nature* 532:370): saturated EM of mouse L2/3; **pyramidal neurons with similar orientation selectivity preferentially form synapses with each other at the spine level**.

**What the mouse story does NOT prove:** that connectivity was ever truly random. Already at eye-opening, ~20 % connectivity exists with a non-significant trend toward like-to-like. Connections aren't created from nothing; they refine.

### Cat

- **Hubel & Wiesel 1962, 1974**: orientation columns ~50 µm wide, hypercolumn ~0.5–1 mm.
- **Gilbert & Wiesel 1989** (*J Neurosci* 9:2432; PMID 2746337): cat V1 horizontal axons run **6–8 mm parallel to the cortical surface** with clustered distribution. Bead retrograde + 2-deoxyglucose orientation maps demonstrate that **retrogradely labeled cells are confined to regions of orientation specificity similar to the injection site** — both intrinsic and corticocortical. **Iso-orientation columnar specificity is anatomically established in adult cat.**

There is no cat developmental study analogous to Ko 2013/2014. The cat literature establishes the iso-orientation pattern in adult, not its developmental origin. Within an orientation column (~50 µm), local sampling is iso-orientation **by geometry alone**, regardless of plasticity.

### Primate (macaque, plus tree shrew as primate-near reference)

- **Malach et al. 1993** (*PNAS* 90:10469): 4 anaesth. macaques (*M. fascicularis*); biocytin + optical imaging; intrinsic horizontals form **patches/clusters extending ~mm** with iso-orientation specificity confirmed by alignment to imaged orientation domains.
- **Sincich & Blasdel 2001** (*J Neurosci* 21:4416): squirrel monkey V1; intralaminar L2/3 connections clustered, iso-orientation, with **collinear bias** (long axis of projection field aligned with preferred orientation).
- **Stettler et al. 2002** (*Neuron* 36:739; PMID 12441061): macaque V1; GFP adenovirus labeling. **V1 intrinsic horizontals link iso-orientation domains** spanning "regions of visual space up to eight times larger than receptive fields." **V2→V1 feedback shows no orientation specificity** (key dissociation).
- **Bosking et al. 1997** (*J Neurosci* 17:2112; PMC6793759): tree shrew (*Tupaia glis*) L2/3; biocytin + optical imaging. Connections **2–5 mm parallel to surface, up to 4 mm**. **Inside 500 µm: not orientation-specific. Outside 500 µm: 57.6 % of boutons contact sites with orientation preference within ±35°** (P < 0.02 vs random); **4× more terminals along the preferred-orientation axis than orthogonal**.

**Like-to-like in cat/primate is established as a structural fact in adult V1 at multiple scales:**
- within-column (~50 µm): iso-orientation by columnar geometry,
- short-range (<500 µm): less specific (consistent with pinwheel-center mixing),
- long-range (≥500 µm to several mm): explicitly iso-orientation, often co-axial.

**Whether the long-range pattern is innate, activity-dependent, or STDP-refined in cat/primate is essentially unknown from primary data.**

### Computational models

- **Antolík et al. 2024** comprehensive cat V1 model (PMC11371232): 5 × 5 mm patch around area centralis, layers 4 + 2/3. **Connectivity is pre-wired stochastically based on RF templates and anatomical constraints — NOT STDP.** Recurrent L2/3 has both distance-dependent and orientation-functional bias. The authors deliberately bypass the developmental question. **This is the canonical modern cat-V1 model approach: hard-code the architecture.**
- **Clopath et al. 2010** (voltage-based STDP) shows in principle that STDP can produce orientation selectivity and like-to-like topography, but on simplified networks not calibrated to cat/primate.

**Bottom line for §1.** For mouse, "random/less-specific initial wiring + STDP refinement" is empirically supported. **For cat and primate, the analogous developmental experiment has not been done.** Cat/primate models in the literature universally pre-wire the iso-orientation columnar geometry. Empirical default = pre-wire.

---

## §2 Cat / primate L4 OSI

| Source | Species | Layer | OSI form | Median | Bandwidth (HWHM) | n | State |
|---|---|---|---|---|---|---|---|
| Anderson, Lampl, Gillespie, Ferster 2000 *Science* 290:1968 | cat | L4 simple | Vm HWHM | – | ~30–35° (Vm); spike sharpened by threshold | – | anaesth. |
| Carandini & Ferster 2000 *J Neurophysiol* 84:909 | cat | L4 simple | Vm HWHM | – | ~30–35° (Vm); spike narrower | – | anaesth. |
| Tan et al. 2011 (PMC3202243) | cat | L II/III + IV pooled | 2-pt spike OSI | **0.83** | – | 31 | anaesth. (intra. WC) |
| Ringach et al. 2002 *J Neurosci* 22:5639 | macaque | population | gOSI = 1−CV | **0.39** | 23.5° | 308 | anaesth. (sufentanil) |
| Ringach 2002 (per-layer) | macaque | 4Cα/4Cβ/3B vs others | gOSI | **layer differences NS (Wilcoxon p > 0.1)** | – | 308 | anaesth. |
| Hong et al. 2024 *Front Neural Circuits* | macaque (rhesus) | L4C | 2-pt OSI | **0.68** | 17.0° | 247 | anaesth. (Neuropixels) |
| Hong et al. 2024 | macaque | L2/3 | 2-pt OSI | **0.70** | – | 116 | anaesth. |
| Hong et al. 2024 | macaque | "other" (4A/B + 5/6) | 2-pt OSI | **0.70** | 15.7° | 414 | anaesth. |
| Schiller, Finlay, Volman 1976 | macaque | all layers | qualitative | simple > complex; complex broaden in deep layers; simple uniform | – | – | anaesth. |
| Antolík et al. 2024 (model target) | cat | L4 excit | HWHH | – | **21.1°** | – | (model) |

**Key cat L4:** HWHM (Vm) ≈ 30–35°; spike OSI ≈ 0.83 (Tan 2011, layer-pooled); cat L4 simple-cell sharpening relative to broad Vm is the **iceberg effect** — spike threshold cuts off subthreshold response at non-preferred orientations.

**Key macaque L4:** Ringach 2002 — population gOSI 0.39, bandwidth 23.5° HWHM, no statistically significant per-layer differences (p > 0.1 in n = 308). Hong 2024 (n = 661, anesthetised, Neuropixels): **L4C ≈ L2/3 ≈ "other" in 2-pt OSI (0.68 vs 0.70 vs 0.70)**, statistically indistinguishable. Bandwidth 17° in L4C, 15.7° elsewhere (small but significant, p = 0.012). The popular "L4 is least selective in primate" narrative is weakly supported by primary data.

---

## §3 Cat / primate L2/3 OSI

| Source | Species | Layer | OSI form | Median | Bandwidth | n | State |
|---|---|---|---|---|---|---|---|
| Hong et al. 2024 | macaque | L2/3 | 2-pt OSI | **0.70** | – | 116 | anaesth. |
| Tan et al. 2011 (combined L II/III + IV) | cat | combined | 2-pt spike | 0.83 | – | 31 | anaesth. |
| Schiller et al. 1976 | macaque | L2/3 (complex-dominated) | qualitative | broader than simple; broadens with depth | – | – | anaesth. |
| Antolík et al. 2024 (model target) | cat | L2/3 excit | HWHH | – | **25.2°** | – | (model) |
| Hubel & Wiesel 1962 | cat | L2/3 complex | qualitative | "fails long before 90°" | ~30–40° HWHM (qualitative) | – | anaesth. |

**Punchline.** In cat and primate, L2/3 OSI is **comparable to or slightly higher than L4** — opposite to the simplistic "L4 sharper" intuition. Hong 2024 macaque is the cleanest direct primary measurement: L2/3 = 0.70, L4C = 0.68, statistically indistinguishable. Cat layer-resolved spike OSI is essentially absent from primary literature; Schiller's qualitative "complex cells broaden in deep layers; simple cells uniform" is consistent with L2/3 ≳ L4 in superficial layers.

---

## §4 L2/3 connectivity geometry

### Within-column / short-range (≤500 µm)

- **Holmgren et al. 2003** (*J Physiol* 551:139; rat L2/3 paired-recording): ~30 inputs per pyramidal cell from local 200 × 200 µm cylinder; distance-dependent connection probability. (Closest available rodent paired data; cat/primate paired at this scale effectively absent.)
- Within a cat orientation column (~50 µm), local cells are by definition iso-orientation; a 200 µm radius spans ~3–4 columns and yields **mixed-orientation local sampling** when hit randomly with respect to the orientation map. Bosking 1997 confirms: *"inside of 500 µm, the pattern of connections is much less specific."*

### Long-range horizontal (≥500 µm)

| Source | Species | Range | Iso-orientation specificity | Layer | Method |
|---|---|---|---|---|---|
| Gilbert & Wiesel 1989 | cat | 6–8 mm | Yes (retro labels confined to iso-orient columns) | L2/3, L5 | bead retro + 2-DG |
| Malach et al. 1993 | macaque | up to ~mm clusters | Yes (aligned to imaged orient domains) | L2/3 | biocytin + optical imaging |
| Sincich & Blasdel 2001 | squirrel monkey | several mm | Yes; **collinear bias** | L2/3 | biocytin |
| Bosking 1997 | tree shrew | 2–5 mm (max 4 mm) | **57.6 % iso-orient within ±35° outside 500 µm** (P < 0.02); 4× collinear bias | L2/3 | biocytin + optical imaging |
| Stettler et al. 2002 | macaque | up to 8× classical RF | Yes for V1 intrinsic; **NO for V2→V1 feedback** | L2/3 | GFP adenovirus |

**Universal cat/primate finding:** long-range L2/3 horizontals preferentially target iso-orientation patches across hypercolumns, with co-axial collinear bias. **Anatomically established, well-replicated.** The mouse-style "STDP refines from random" experiment has never been done in cat or primate; the cat/primate literature treats this connectivity as architectural.

---

## §5 Modeling defaults for our cat/primate calibration

Architecture: 32 × 32 retinotopic hypercolumns × 8 orientations × 16 cells/orientation (131k L4, 16k L2/3).

### A) L4 von Mises tuning sharpness — **the gOSI/HWHM constraint is internally inconsistent**

For R(θ) ∝ exp(κ cos(2(θ − θ_pref))):
- analytical gOSI = I₁(κ)/I₀(κ)
- HWHM = ½ × (180/π) × arccos(1 + ln(0.5)/κ)

| κ | HWHM | analytical gOSI |
|---|---|---|
| 2.0 | 27° | 0.46 |
| 3.0 | 20° | 0.55 |
| 4.0 | 16° | 0.64 |
| 4.5 (current) | 15° | 0.68 |
| 5.0 | 14° | 0.71 |
| 7.0 | 12° | 0.78 |

**Targets gOSI 0.7–0.8 AND HWHM 20–25° cannot both be hit by κ alone in a pure von Mises.** For gOSI 0.7, HWHM = 14.5°; for HWHM 20°, gOSI ≈ 0.55.

**Biological resolution: the iceberg effect.** Cat L4 has Vm HWHM ≈ 30–35° but spike OSI ≈ 0.83. The spike threshold cuts off subthreshold response at non-preferred orientations.

**Recommended approach:**
1. **Option A (simplest):** keep κ ≈ 4.5–5.0 (HWHM 13–15°), accept that the underlying tuning is sharper than cat Vm but lands at gOSI ≈ 0.70 — matching macaque L4C empirical 0.68 (Hong 2024) **almost exactly**. Empirically defensible.
2. **Option B (more biologically faithful):** keep κ ≈ 3.5–4.0 (HWHM 16–22°, matching Antolík model target HWHH 21.1°) plus a spontaneous-rate baseline subtraction or non-zero firing-rate threshold below which R(θ) → 0 in spike output. This mimics the iceberg, sharpens measured gOSI to 0.7+, and produces realistic frac_silent at non-preferred orientations.
3. **Trade-off:** option B raises silent fraction but is the path biology takes; option A is simpler and lands the OSI target without adding a baseline mechanism.

### B) L4 → L2/3 connectivity in cat/primate

Three architectural choices:

1. **Mixed-orientation L4 sampling + STDP refinement** (mouse-style). Pool across all 8 orientations × 9 spatial positions; STDP later eliminates mismatches. **Empirically supported only in mouse.**
2. **Iso-orientation L4 sampling by geometric prior** (cat/primate-style). Each L2/3 cell pools L4 cells with orientations within ~±22.5° of its own preference. Within the 3 × 3 patch this is 9 spatial positions × 16 cells ≈ 144 partners. **Matches cat/primate columnar geometry directly.** No STDP needed for orientation.
3. **Hybrid:** mixed local (~80 µm scale, within-pinwheel) + biased iso-orientation across the columnar map.

**Recommended for cat/primate calibration: option 2.** The cat/primate data unambiguously shows the architecture is columnar; reproducing this geometry directly is more biologically faithful than rederiving it from random + STDP. Option 1 is appropriate only if the explicit goal is to model the developmental refinement process itself.

Partner count: cat/primate L4 input to L2/3 is on the order of hundreds per cell. Current ~40 L4 partners is on the low end but acceptable as a tractable sparse approximation.

### C) L2/3 → L2/3 recurrent geometry

**Local recurrent (within hypercolumn, ≤500 µm):** keep current B1 distance-decaying form. Holmgren 2003 + Cossell 2015 support distance decay; Bosking 1997 explicitly notes **inside 500 µm connections are NOT orientation-specific** — local random sampling is biologically appropriate.

**Long-range horizontal (≥500 µm = ≥several hypercolumns):** cat/primate biology unambiguous — iso-orientation, co-axial.

Three options:
1. Keep local-only; rely on STDP for long-range iso-orientation. **Not supported by mouse Ko data (only short-range pairs tested) and never tested in cat/primate.**
2. **Hard-code iso-orientation horizontal connections at hypercolumn scale** (Bosking 1997-style: ≥500 µm range, ~57.6 % iso-orientation hit rate within ±35°, with collinear axis bias if direction-preference is meaningful). **Matches cat/primate biology directly.**
3. Hybrid — local random + sparse long-range iso-orientation patches.

**Recommended for cat/primate calibration: option 2.** Cat/primate adult connectivity is anatomical and well-replicated. Mouse-style "random + STDP" is the wrong template. Implementation: each L2/3 cell sends sparse axons (~5–10) to distant hypercolumns, biased so destination cells share preferred orientation within ±35° (Bosking number) and biased toward the visual axis collinear with the cell's preferred orientation (Bosking 4× collinear bias).

---

## §6 Open questions / cautions

1. **The "L4 is least selective in primate" claim is weak.** Ringach 2002 pairwise Wilcoxon p > 0.1; Hong 2024 finds L4C ≈ L2/3 (0.68 vs 0.70, statistically indistinguishable, n = 661). Reviews overstate this.
2. **Cat developmental connectivity (random→like-to-like via STDP) is not directly tested.** Cat/primate models pre-wire. Using STDP to LEARN iso-orientation in a cat-style model goes beyond established cat literature — that's a model design choice, not a biological fact.
3. **Iceberg effect must be modeled to match both gOSI and HWHM targets.** A pure von Mises spike-rate generator forces a trade-off; biology resolves it via spike threshold on a broader Vm.
4. **Tan 2011 cat OSI = 0.83 is layer-pooled (II/III + IV) intracellular WC, n = 31.** Cat layer-resolved spike OSI is essentially absent from primary literature.
5. **Bosking 1997 is tree shrew** (primate-near, columnar architecture). Sincich & Blasdel 2001 (squirrel monkey) and Stettler 2002 (macaque) confirm qualitatively but with less rigorous quantitation. Bosking is the canonical quantitative iso-orientation horizontal-connection reference for the primate-style architecture.
6. **Holmgren 2003 connectivity numbers are rat.** Distance-dependent connection probability is plausibly conserved across mammals; cat/primate paired patch is rare.

---

## §7 References (primary)

1. Hubel DH, Wiesel TN (1962). "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex." *J Physiol* 160:106–154.
2. Hubel DH, Wiesel TN (1974). "Sequence regularity and geometry of orientation columns in the monkey striate cortex." *J Comp Neurol* 158:267–293.
3. Schiller PH, Finlay BL, Volman SF (1976). "Quantitative studies of single-cell properties in monkey striate cortex. II. Orientation specificity and ocular dominance." *J Neurophysiol* 39(6):1320–1333. PMID 825622.
4. Gilbert CD, Wiesel TN (1989). "Columnar specificity of intrinsic horizontal and corticocortical connections in cat visual cortex." *J Neurosci* 9(7):2432–2442. PMID 2746337.
5. Malach R, Amir Y, Harel M, Grinvald A (1993). "Relationship between intrinsic connections and functional architecture revealed by optical imaging and in vivo targeted biocytin injections in primate striate cortex." *PNAS* 90(22):10469–10473.
6. Bosking WH, Zhang Y, Schofield B, Fitzpatrick D (1997). "Orientation Selectivity and the Arrangement of Horizontal Connections in Tree Shrew Striate Cortex." *J Neurosci* 17(6):2112–2127. PMC6793759.
7. Carandini M, Ferster D (2000). "Orientation Tuning of Input Conductance, Excitation, and Inhibition in Cat Primary Visual Cortex." *J Neurophysiol* 84(2):909–926.
8. Anderson JS, Lampl I, Gillespie DC, Ferster D (2000). "The Contribution of Noise to Contrast Invariance of Orientation Tuning in Cat Visual Cortex." *Science* 290(5498):1968–1972. PMID 11110664.
9. Sincich LC, Blasdel GG (2001). "Oriented axon projections in primary visual cortex of the monkey." *J Neurosci* 21(12):4416–4426.
10. Ringach DL, Shapley RM, Hawken MJ (2002). "Orientation selectivity in macaque V1: diversity and laminar dependence." *J Neurosci* 22(13):5639–5651. PMID 12097515.
11. Stettler DD, Das A, Bennett J, Gilbert CD (2002). "Lateral Connectivity and Contextual Interactions in Macaque Primary Visual Cortex." *Neuron* 36:739–750. PMID 12441061.
12. Holmgren C, Harkany T, Svennenfors B, Zilberter Y (2003). "Pyramidal cell communication within local networks in layer 2/3 of rat neocortex." *J Physiol* 551:139–153.
13. Hirsch JA, Martinez LM, Pillai C, Alonso JM, Wang Q, Sommer FT (2003). "Functionally distinct inhibitory neurons at the first stage of visual cortical processing." *Nat Neurosci* 6(12):1300–1308.
14. Clopath C, Büsing L, Vasilaki E, Gerstner W (2010). "Connectivity reflects coding: a model of voltage-based STDP with homeostasis." *Nat Neurosci* 13:344–352.
15. Tan AYY, Brown BD, Scholl B, Mohanty D, Priebe NJ (2011). "Orientation Selectivity of Synaptic Input to Neurons in Mouse and Cat Primary Visual Cortex." *J Neurosci* 31(34):12339–12350. PMC3202243.
16. Ko H, Hofer SB, Pichler B, Buchanan KA, Sjöström PJ, Mrsic-Flogel TD (2011). "Functional specificity of local synaptic connections in neocortical networks." *Nature* 473:87–91. doi:10.1038/nature09880. PMC3089591.
17. Ko H, Cossell L, Baragli C, Antolik J, Clopath C, Hofer SB, Mrsic-Flogel TD (2013). "The emergence of functional microcircuits in visual cortex." *Nature* 496:96–100. PMC4843961.
18. Ko H, Mrsic-Flogel TD, Hofer SB (2014). "Emergence of Feature-Specific Connectivity in Cortical Microcircuits in the Absence of Visual Experience." *J Neurosci* 34(29):9812–9816. PMC4099553.
19. Cossell L, Iacaruso MF, Muir DR, Houlton R, Sader EN, Ko H, Hofer SB, Mrsic-Flogel TD (2015). "Functional organization of excitatory synaptic strength in primary visual cortex." *Nature* 518:399–403. PMC4843963.
20. Lee WCA, Bonin V, Reed M, et al. (2016). "Anatomy and function of an excitatory network in the visual cortex." *Nature* 532:370–374. doi:10.1038/nature17192.
21. Antolík J et al. (2024). "A comprehensive data-driven model of cat primary visual cortex." PMC11371232.
22. Hong YL et al. (2024). "Comparison of orientation encoding across layers within single columns of primate V1 revealed by high-density recordings." *Front Neural Circuits*. doi:10.3389/fncir.2024.1399571.

---
*End of report.*
