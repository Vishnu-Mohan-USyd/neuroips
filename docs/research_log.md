# Research Log

> **HISTORICAL / NONCANONICAL.** This file is retained as development lineage.
> The canonical current workflow, equations, results, and reproduction guide are
> [`emergent_task_energy_axis.md`](emergent_task_energy_axis.md). The brief
> pointer below is non-normative; the remainder records legacy repair work.

## Current-workflow pointer (non-normative)

The current workflow is documented in
[`emergent_task_energy_axis.md`](emergent_task_energy_axis.md). It uses one
fixed orientation substrate, a recurrent predictor, a SOM/VIP-inspired
Dale-sign-constrained rate motif, and frozen divisive local competition. The
only arm coordinate is the task-versus-normalized-L2/3-mean-rate weight; matched
operational continuation/reversal geometry is post-training only.

Every arm minimizes the same two-term objective,
`J(alpha)=(1-alpha)T+alpha E`. `T` is normalized next-step prediction plus noisy
current-orientation precision; `E=mean(r)/R_ref` is a normalized L2/3 mean-rate
proxy. `alpha=.5` is the balanced coordinate, not a new loss. No loss knows the
operational A/B labels, expected orientation, center/flank windows, desired
shape, or amplitude threshold.

### Dampening-amplitude calibration record

- `alpha=.6` passed the development screen on seeds `0–3`, with mean
  whole-profile retention `M=.2740319260531955`.
- On the first fresh cohort, seeds `4–7`, `.6` passed the stored energy,
  decoding, `dC<dF`, and `dQ<0` checks, but failed amplitude retention. Mean
  `M(.6)=.21336743856557555` versus `M(.9)=.15607987727316153`; seed 5's M
  ratio was `1.2243398680161324<1.25`; seed 7's ratio was
  `1.206259035990063<1.25` and its difference was
  `.029167501148376213<.040`; cohort `M(.6)<.250`. It was not accepted. No
  stored `Cret/Fret` claim is made for this cohort.
- `alpha=.5` passed revised development seeds `4–7` and an independent
  from-scratch confirmation on seeds `8–11`.
- In the fresh cohort, all `48/48` per-seed checks across the three scientific
  readout families passed. Whole-profile retention values were
  `.2901656344312839`, `.29020974714964337`, `.2722376079208284`, and
  `.34392168241773374`; their mean `.29913366797987234` exceeded the `.250`
  boundary by `.04913366797987234`.
- Final all-assays seal:
  `027feb665537e1f54628e9e7af1ff5b25bdb759e067ff02e6b751fb42e37cd51`.
  Assay ledger:
  `04404bd8efdaba8a506b686d746c79bbb03b4212799ced43fd3c8ef2c3fb77a4`
  (`32/32` verified).

The shape term **relative flank sparing** means `F_A/F_t0 > C_A/C_t0`; it does
not mean that absolute flanks exceed the literal `t=0` response. The gray
baseline is the ordinary first response of the same arm with recurrent state
and feedback exactly zero.

The initial fresh-cohort evaluator accidentally bound `M` to stored A/B saving.
The corrected, development-consistent definition is
`M=AUC(A final aligned 36-bin profile)/AUC(t0 aligned 36-bin profile)`, exactly
equivalent to `rate_A/rate_t0`. This was an evaluator-only defect: no model,
checkpoint, assay, profile, threshold, or training output changed. The
hash-bound supersession is
`c3fa958cba809e0aafef0c2a8db6de4224c05b8610ddf5320ac01feaba0284ee`.

Biological mappings are deliberately limited. VIP/SOM signs are motivated by
Pi et al. (2013, doi:10.1038/nature12676) and Pfeffer et al. (2013,
doi:10.1038/nn.3446); divisive competition is motivated by normalization work
including Heeger (1992, doi:10.1017/S0952523800009640) and Carandini et al.
(1997, doi:10.1523/JNEUROSCI.17-21-08621.1997). The GRU, mean-rate proxy,
fixed circular basis, and exact loss/assay definitions remain engineering
approximations. The energy–information premise is grounded qualitatively by
Laughlin, de Ruyter van Steveninck & Anderson (1998,
[PMID 10195106](https://pubmed.ncbi.nlm.nih.gov/10195106/)) and Niven, Anderson
& Laughlin (2007,
[PMID 17373859](https://pubmed.ncbi.nlm.nih.gov/17373859/)); activity-linked
local ATP demand is supported by Rangaraju, Calloway & Ryan (2014,
[PMID 24529383](https://pubmed.ncbi.nlm.nih.gov/24529383/)). None of those
studies turns `mean(r)/R_ref` into a biophysical energy measurement. The
current endpoint shows center suppression with relative flank sparing, not
absolute flank preservation.

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
