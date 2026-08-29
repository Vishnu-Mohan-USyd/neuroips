# Candidate 6 — sharpening and dampening from an interneuron circuit

Both predictive-coding phenotypes emerge in one network from a single shared
pretraining, with **the energy weight `alpha` as the only difference between
the two arms**. Nothing else about the model, the data, the seed or the
training schedule changes between them.

| | expected orientation | flanks (15–30°) | peak of the response |
|---|---|---|---|
| **α = 0.0** — task optimized | **1.28×** baseline | 0.72× | at 0° |
| **α = 0.5** — energy optimized | **0.10×** baseline | 0.71× | displaced to **±15°** |

The dampening arm shows both signatures at once: the predicted orientation is
suppressed to a tenth of its unmodulated response, while the flanks retain
seven tenths of theirs. The profile's maximum is no longer at the presented
orientation at all — it has moved onto the flanks.

Reproduced across three seeds:

| seed | sharpening centre / flank | dampening centre / flank | dampening peak |
|---|---|---|---|
| 8 | 1.2764 / 0.7188 | 0.1007 / 0.7063 | +15° |
| 9 | 1.2914 / 0.7196 | 0.1064 / 0.7062 | +15° |
| 10 | 1.3103 / 0.7451 | 0.0886 / 0.7432 | −15° |

## The circuit

L2/3 excitatory units across 36 orientation channels, with a **fixed SOM
blanket** providing broad subtractive inhibition and a **PV** population
providing channel-exact divisive suppression — the sole learnable suppressive
gain. Top-down feedback carries the prediction. There are no hand-set
orientation-specific weights and no per-orientation gain terms; the
phenotypes are learned from the task and energy pressures alone.

## Reproducing the figures

```
python3 reproduce_figures.py
```

Requires PyTorch (CPU is sufficient) and Matplotlib. Runs in about a minute,
writes `figures/`, and checks the seed-8 numbers against the values banked in
the script. It exits non-zero if any of them has moved by more than 1e-3.

That check is not decorative. The model code and these checkpoints have to
match: substituting a later revision of `tuned_emergence_lib.py` — one
carrying the apical-gating change made after these runs — loads the same
checkpoints without error but moves the flank ratio from 0.71 to 1.04 and
fails the check.

## Layout

```
harness/
  tuned_emergence_lib.py   the Candidate-6 model
  train_sweep.py           the trainer that produced the checkpoints
  simple_net.py            reference implementation, unmodified
tools/
  assay_emergent_task_energy_axis.py   matched-pair probe assay
checkpoints/seed{8,9,10}/alpha{0p0,0p5}/
  common_pretrain_final.pt   the shared pretrain both arms start from
  alpha_0p{0,5}_final.pt     the two arms
  training_summary.json
figures/
  c6_{sharpening,dampening}_seed{8,9,10}.png
  c6_curves.json             every plotted curve, as data
reproduce_figures.py
```

## Scope

This branch contains only the Candidate-6 network. Code for other candidate
architectures explored in the same campaign is deliberately excluded, and the
harness here is the revision that these checkpoints were trained under.

## How the curves are measured

Matched pairs of trials are presented; the response is aligned to the
presented orientation and averaged over trials. The **baseline** is the
network's own response at the first timestep, before feedback has arrived —
so each arm is compared against itself, not against the other arm. The
**expected** trace is the response at the final timestep, once the prediction
is in place. Centre is the presented channel and its two neighbours; the
flank band is ±15–30°.
