# Research Log

## Scope

Initial v0 for a scaled-down, two-layer V1 model with:

- fixed L4 Gabor-like orientation tuning,
- local many-to-one L4 to L2/3 convergence,
- feature-biased local recurrence,
- explicit L4 and L2/3 inhibition,
- emergent L2/3 orientation selectivity from wiring and dynamics,
- gated local STDP on L4E to L2/3E feedforward synapses.

## Mechanism choices

### 1. L4 excitatory drive uses fixed Gabor-like receptive fields

Biological mapping:
Simple-cell-like L4 receptive fields are approximated by fixed Gabor filters
anchored at retinotopic positions.

Why this is in v0:
It is the only strong domain-specific prior hardcoded into the model. It gives a
clean, interpretable orientation scaffold for L4 while leaving downstream L2/3
selectivity to emerge.

Primary sources:

- Jones JP, Palmer LA. 1987. *An evaluation of the two-dimensional Gabor filter
  model of simple receptive fields in cat striate cortex.* Journal of
  Neurophysiology. DOI: https://doi.org/10.1152/jn.1987.58.6.1233
- Hubel DH, Wiesel TN. 1962. *Receptive fields, binocular interaction and
  functional architecture in the cat's visual cortex.* Journal of Physiology.
  DOI: https://doi.org/10.1113/jphysiol.1962.sp006837

### 2. L4 to L2/3 is local, convergent, and feature-biased

Biological mapping:
L2/3 excitatory cells receive many local feedforward contacts from L4 inside a
mostly column-restricted domain, with an orientation bias rather than fully
random pooling.

Why this is in v0:
The model samples local L4 inputs within a Gaussian retinotopic neighborhood and
weights sampling toward similar orientation tags. This is the minimum structure
needed to reflect columnar V1 without hardcoding L2/3 tuning curves directly.

Primary sources:

- Chisum HJ, Mooser F, Fitzpatrick D. 2004. *A morphological basis for
  orientation tuning in primary visual cortex.* Nature Neuroscience.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/15258585/
- Binzegger T, Douglas RJ, Martin KAC. 2004. *A quantitative map of the circuit
  of cat primary visual cortex.* Journal of Neuroscience.
  DOI: https://doi.org/10.1523/JNEUROSCI.1400-04.2004

### 3. Inhibition is explicit in both layers and kept local

Biological mapping:
L4 inhibition is fast and stabilizing, while L2/3 inhibition constrains local
recurrent amplification and keeps tuning from collapsing into runaway
excitation.

Why this is in v0:
A simple E/I split with faster inhibitory dynamics is the smallest biologically
reasonable stabilizer for the two-layer network. Detailed PV/SOM/VIP subtyping
is deferred until the base model is validated.

Primary sources:

- Atallah BV, Bruns W, Carandini M, Scanziani M. 2012.
  *Parvalbumin-expressing interneurons linearly transform cortical responses to
  visual stimuli.* Neuron. DOI: https://doi.org/10.1016/j.neuron.2011.12.013
- Potjans TC, Diesmann M. 2014. *The cell-type specific cortical microcircuit:
  relating structure and activity in a full-scale spiking network model.*
  Cerebral Cortex. DOI: https://doi.org/10.1093/cercor/bhs358

### 4. L4E to L2/3E training uses gated local STDP

Biological mapping:
Feedforward excitatory weights are updated from pre/post spike timing using a
bounded pair-based STDP rule. The training gate is an experiment-control switch,
not a nonlocal normalization rule.

Why this is in v0:
The baseline and post-training orientation sweeps are measured with learning
disabled. During training trials only, dynamic `Aplus/Aminus` enable local
plasticity on `L4E -> L23E`; a zero-learning control keeps the same stimulus
history but freezes the weights.

Validation evidence:
The full H200 run produced baseline `L23E` median OSI `0.492307` and
post-training `0.846834` (`+0.354527`). The matched zero-learning control
produced post `L23E` median OSI `0.490047` (`-0.002260`), while weights stayed
unchanged to printed precision.

Primary sources:

- Markram H, Lubke J, Frotscher M, Sakmann B. 1997. *Regulation of synaptic
  efficacy by coincidence of postsynaptic APs and EPSPs.* Science.
  DOI: https://doi.org/10.1126/science.275.5297.213
- Bi GQ, Poo MM. 1998. *Synaptic modifications in cultured hippocampal neurons:
  dependence on spike timing, synaptic strength, and postsynaptic cell type.*
  Journal of Neuroscience. PubMed: https://pubmed.ncbi.nlm.nih.gov/9852584/

## Assumptions

- Each lattice site is treated as a scaled hypercolumn proxy rather than a full
  morphological reconstruction.
- Excitatory and inhibitory counts preserve coarse layer proportions rather than
  absolute macaque neuron counts.
- L2/3 orientation selectivity is not directly injected as a tuning curve; the
  only explicit feature tag is used to bias developmental wiring.

## Deferred items

- Subtyped inhibition implementation is planned in
  `docs/inhibition_subtypes_research.md`. The recommended next model split is
  L4 PV/SOM plus L2/3 PV/SOM/VIP, with Lamp5/NGF deferred until feedback or
  apical-tuft context is explicitly modeled.
- Three-factor neuromodulated plasticity and homeostatic stabilization.
- Multi-hypercolumn contextual feedback and long-range L2/3 horizontals.
