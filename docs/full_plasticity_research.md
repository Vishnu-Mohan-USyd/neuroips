# Full Plasticity Research Notes

## Scope

This branch extends the validated subtype model beyond static recurrent and
inhibitory circuits. The target is still a tractable two-layer L4/L2/3 model,
not a full six-layer or feedback V1 model.

Out of scope for this branch:

- Top-down or contextual VIP learning.
- Long-range multi-hypercolumn horizontal axons beyond the current sheet.
- Full conductance/current decomposition unless needed after rate validation.

## Recurrent L2/3 Excitatory Connectivity

Biological constraints:

- L2/3 recurrent excitation is sparse, local, and non-random.
- Same-feature or co-active excitatory neurons are more likely to be connected
  and can have stronger synapses, but this should emerge from activity and
  plasticity in this model rather than from hardcoded orientation labels.
- Strong E->E weights should remain sparse/log-tailed rather than all saturating.
- Long-range patchy horizontal/iso-orientation projections are real in
  orientation-map species, but they are a separate mechanism from local
  recurrent E->E inside this small two-layer patch.

Implementation implications:

- Use local `L23E -> L23E` recurrence, not orientation-pruned recurrence.
- Use weak bounded STDP on recurrent E->E, smaller than feedforward STDP.
- Validate that recurrent E weights change more for co-active/co-tuned pairs
  without destroying L2/3 OSI or causing rate runaway.

Primary sources:

- Yoshimura, Dantzker, Callaway 2005, Nature. DOI:
  https://doi.org/10.1038/nature03252
- Ko et al. 2011, Nature. DOI: https://doi.org/10.1038/nature09880
- Cossell et al. 2015, Nature. DOI: https://doi.org/10.1038/nature14182
- Song et al. 2005, PLOS Biology. DOI:
  https://doi.org/10.1371/journal.pbio.0030068
- Stettler et al. 2002, Neuron. DOI:
  https://doi.org/10.1016/S0896-6273(02)01029-2
- Bosking et al. 1997, Journal of Neuroscience. DOI:
  https://doi.org/10.1523/JNEUROSCI.17-06-02112.1997

## Recurrent Excitatory Plasticity Rule

Biological constraints:

- Cortical E->E plasticity has a tens-of-ms timing window.
- Recurrent plasticity must be weaker and more guarded than feedforward
  plasticity in this small recurrent network.
- Homeostatic guardrails are required because recurrent Hebbian plasticity is
  otherwise prone to runaway or collapse.

Implementation implications:

- Use local pair-based STDP with `tauPlus` around 20 ms and `tauMinus` around
  20-40 ms.
- Use slight LTD bias and narrow bounds around the existing recurrent E weight.
- Keep recurrent plasticity separately toggleable for ablation.

Primary sources:

- Bi & Poo 1998, Journal of Neuroscience. DOI:
  https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998
- Sjostrom et al. 2001, Neuron. DOI:
  https://doi.org/10.1016/S0896-6273(01)00542-6
- van Rossum et al. 2000, Journal of Neuroscience. DOI:
  https://doi.org/10.1523/JNEUROSCI.20-23-08812.2000
- Clopath et al. 2010, Nature Neuroscience. DOI:
  https://doi.org/10.1038/nn.2479

## Inhibitory Plasticity And Homeostasis

Biological constraints:

- Inhibitory plasticity can maintain E/I balance and sparse cortical responses.
- The sign of inhibitory synapses must remain inhibitory; plasticity changes
  magnitude, not sign.
- A practical first target is plastic `PV -> E` and slower/lower-rate
  `SOM -> E` adaptation. VIP learning is excluded.
- The rule should be local and postsynaptic-rate seeking: high postsynaptic E
  activity strengthens inhibition, low activity weakens it.

Implementation implications:

- Add trace-based inhibitory plasticity to `L23PV -> L23E` and `L23SOM -> L23E`.
- Bound inhibitory magnitude around the existing static weights.
- Validate that inhibitory weights change in the direction predicted by
  postsynaptic L23E activity and that weights do not pile up at bounds.

Primary sources:

- Vogels et al. 2011, Science. DOI:
  https://doi.org/10.1126/science.1211095
- D'Amour & Froemke 2015, Neuron. DOI:
  https://doi.org/10.1016/j.neuron.2015.03.014
- Vickers et al. 2018, Neuron. DOI:
  https://doi.org/10.1016/j.neuron.2018.07.018
- Xue et al. 2014, Nature. DOI: https://doi.org/10.1038/nature13321

## SOM Broad Suppression

Biological constraints:

- SOM should not be represented as merely stronger local PV-like inhibition.
- SOM recruitment is broader and slower, supporting broad/context/lateral
  suppression through dendritic inhibition.
- The first validation should compare center-only and broad-field stimulation,
  then verify that SOM ablation weakens broad suppression.

Implementation implications:

- Use a broader L2/3 SOM radius than PV.
- Export center-vs-broad response metrics.
- Use SOM-output ablation as a validation control.

Primary sources:

- Adesnik et al. 2012, Nature. DOI: https://doi.org/10.1038/nature11526
- Wilson et al. 2012, Nature. DOI: https://doi.org/10.1038/nature11347
- Kapfer et al. 2007, Nature Neuroscience. DOI:
  https://doi.org/10.1038/nn1909
- Ozeki et al. 2009, Neuron. DOI:
  https://doi.org/10.1016/j.neuron.2009.03.028

## Acceptance Gates

- Full default run must keep `post_l23_median_osi >= 0.70`.
- Full default run must exceed the no-learning control by at least `0.10` OSI.
- Recurrent E->E weights must change measurably without saturating.
- Inhibitory PV/SOM->E weights must change measurably without sign flips.
- L23E rates must not collapse or run away.
- Broad-field stimulation must suppress central L23E responses causally:
  mean broad suppression index over driven center-response orientations should
  remain strong in the intact model, and SOM-output ablation must reduce that
  suppression.
