# Research Log

> **HISTORICAL / NONCANONICAL.** This file is retained as development lineage.
> The canonical current workflow, equations, results, and reproduction guide are
> [`emergent_task_energy_axis.md`](emergent_task_energy_axis.md). The brief
> pointer below is non-normative; the remainder records legacy repair work.

## Current-workflow pointer (non-normative)

Use the [canonical guide](emergent_task_energy_axis.md) for the current
architecture, equations, exact values, seals, and reproduction. Use the
machine-readable
[`endpoint_selection_record.json`](../figures/emergent_reference_comparison/endpoint_selection_record.json)
for endpoint history and hashes. The current model is `SimpleTunedNet`; root
`simple_net.SimpleNet` belongs to the legacy Phase A/B lineage.

Chronological current-workflow discoveries:

1. `alpha=.6` produced a viable development signal on seeds `0–3` and advanced
   to a fresh cohort.
2. On fresh seeds `4–7`, `.6` retained the stored energy, decoding, `dC<dF`,
   and `dQ<0` signs but failed the amplitude-sensitive M criteria. It was
   rejected; no `Cret/Fret` conclusion was inferred from unstored leaves.
3. Balanced `alpha=.5` passed revised development seeds `4–7` and independent
   from-scratch seeds `8–11`, so `.5` became the selected endpoint.
4. The first fresh evaluator had bound `M` to A/B saving. The evaluator was
   corrected to `M=AUC(A final)/AUC(literal t0)`; checkpoints, assays,
   thresholds, profiles, and training remained unchanged.

Biological interpretation remains deliberately limited. SOM/VIP signs and
normalization are qualitative analogies; the GRU and fixed circular basis are
engineering abstractions. Energy–information studies
([PMID 10195106](https://pubmed.ncbi.nlm.nih.gov/10195106/),
[PMID 17373859](https://pubmed.ncbi.nlm.nih.gov/17373859/)) and activity-linked
local ATP demand
([PMID 24529383](https://pubmed.ncbi.nlm.nih.gov/24529383/)) motivate an
energetic constraint but do not make `mean(r)/R_ref` a biophysical energy
measurement. “Relative flank sparing” does not claim absolute flank
enhancement above the literal first-stimulus response.

## Legacy remainder

## Expectation-Suppression Repair

Mechanism summary:

- Prediction-derived metabolic pressure encourages expected stimuli to use less
  L2/3 activity than matched 90-degree unexpected stimuli.
- A current-orientation decoding term keeps expected representations usable in
  sharpen/attend instead of allowing global activity collapse.
- Dampen/save adds floor-relative expected-content suppression using a
  model-derived prediction mask, not a fixed flank or response-profile target.
- Phase B uses SOM/VIP-style context gating: ctx=+1 supports sharpen/attend,
  and ctx=-1 supports dampen/save.
- Noisy readout stress tests whether the decoded expected representation remains
  precise under perturbation.

Biological mapping:

- Predictive coding motivates top-down prediction signals and reduced activity
  for expected sensory input.
- Expectation sharpening and expectation suppression motivate the distinct
  attend/sharpen and save/dampen regimes.
- Trial-to-trial noise and attention effects motivate noisy readout validation.
- VIP/SOM disinhibitory context control motivates a context-gated circuit
  interpretation.

Citation anchors:

- Rao & Ballard 1999, DOI `10.1038/4580`.
- Kok, Jehee & de Lange 2012, DOI `10.1016/j.neuron.2012.04.034`.
- Alink et al. 2010, DOI `10.1523/JNEUROSCI.3730-10.2010`.
- Tolhurst, Movshon & Dean 1983, DOI `10.1016/0042-6989(83)90200-6`.
- Cohen & Maunsell 2009, DOI `10.1038/nn.2439`.
- Pi et al. 2013, DOI `10.1038/nature12676`.
- Lee et al. 2013, DOI `10.1038/nn.3544`.
- Zhang et al. 2014, DOI `10.1126/science.1254126`.

Engineering approximations:

- `SimpleNet` is a compact recurrent engineering model, not a compartmental or
  spiking cortical simulation.
- The recurrent predictor uses GRU-style machinery.
- Decoding uses cross-entropy over orientation channels.
- Prediction masks and objective terms are differentiable proxies for
  prediction-weighted metabolic pressure.
- Gaussian noisy readout is a controlled stress test, not a fitted cortical
  noise model.

Validation interpretation:

- Passing validators show that repaired checkpoints satisfy the specified
  deterministic paired-sequence criteria.
- Independent sharpen/dampen and combined ctx=+1/ctx=-1 checkpoints reproduce
  the intended energy/precision tradeoff.
- Generated checkpoints and logs remain local artifacts; canonical repository
  checkpoints were not overwritten.
