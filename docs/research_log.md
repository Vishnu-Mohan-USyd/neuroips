# Research Log

> **HISTORICAL / NONCANONICAL.** This file is retained as development lineage.
> The canonical current workflow, equations, results, and reproduction guide are
> [`emergent_task_energy_axis.md`](emergent_task_energy_axis.md). The brief
> pointer below is non-normative; the remainder records legacy repair work.

## Current-workflow pointer (non-normative)

The current four-seed workflow is documented in
[`emergent_task_energy_axis.md`](emergent_task_energy_axis.md). It uses one
fixed orientation substrate, a recurrent predictor, a SOM/VIP-inspired
Dale-sign-constrained rate motif, and frozen divisive local competition. The
only arm coordinate is the task-versus-normalized-L2/3-mean-rate weight; matched
operational continuation/reversal geometry is post-training only.

Biological mappings are deliberately limited. VIP/SOM signs are motivated by
Pi et al. (2013, doi:10.1038/nature12676) and Pfeffer et al. (2013,
doi:10.1038/nn.3446); divisive competition is motivated by normalization work
including Heeger (1992, doi:10.1017/S0952523800009640) and Carandini et al.
(1997, doi:10.1523/JNEUROSCI.17-21-08621.1997). The GRU, mean-rate proxy,
fixed circular basis, and exact loss/assay definitions remain engineering
approximations. The high-alpha endpoint is broad attenuation with preferential
center suppression, not absolute flank-preserving dampening.

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
