# expectation_snn

Minimal predictive-feedback spiking neural network (V1 + H module) for dissociating
sharpening vs dampening regimes of expectation, following plan v5.

## Scientific question

Given an already-existing sensory code and a learned prior (cue or context),
does feedback make the V1 representation look more like sharpening or more like
dampening?

Paradigms bridged (held-out assays, NOT training targets):
- **Kok 2012** (Neuron 75:265) — explicit-cue passive cueing
- **Richter et al. 2022** (Oxford Open Neurosci 1:kvac013) — incidental-context cross-over
- **Tang et al. 2023** (Nat Commun 14:1196) — rotating deviants, gain-not-width signature

## Falsifiable claim

The feedback balance `r = g_direct / g_SOM` (with `g_total` held constant) determines
whether sharpening, dampening, or hybrid regimes emerge. Pre-registered 5-point sweep
across conditions S1–S5.

## Environment

```bash
conda env create -f env.yml
conda activate expectation_snn
python -c "import brian2; print(brian2.__version__)"
```

## Repository layout

See plan §10. Core modules in `brian2_model/`, held-out assays in `assays/`,
phase gates in `validation/`, analysis in `analysis/`, docs + research log in `docs/`.

## Plan

Full architecture, training stages, metrics, ablations, pre-registered hypotheses:
`/home/vysoforlife/.claude/plans/polished-foraging-narwhal.md` (plan v5).
