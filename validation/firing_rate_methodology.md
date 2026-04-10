# Firing Rate Measurement Methodology — SNN Validation

**Author:** Validator
**Purpose:** Define the *exact* procedure by which population firing rates are measured from spike trains of the spiking V1-V2 network. This methodology is the arbiter for every biological-plausibility firing-rate gate, from Phase 1 through Phase 4.
**Status:** Authoritative — any GO/NO-GO verdict citing firing rates must use this procedure. If this document conflicts with the SNN port plan file (`~/.claude/plans/quirky-humming-giraffe.md` lines 182-188), **this document wins**; the plan file has known wrong attribution on firing-rate ranges.

**Revision log:**
- 2026-04-10, v1 (draft): Primary gate = plan/macaque values, warn band = mouse values.
- 2026-04-10, v2: Team lead inverted the policy — primary = mouse handout values, warn = plan values.
- 2026-04-10, v3: Team lead tightened again after the Researcher delivered the **primary-source evidence pack** (`~/.claude/plans/snn_port_evidence_pack.md`). Primary gate moved to **STATIONARY mouse V1 (quiet wake)**. SOM/VIP values (5-15 Hz each) were annotated "via team-lead ruling (pending primary source)".
- 2026-04-10, v4 (**current**): Team lead ruled "Option A" after the Researcher delivered a follow-up SOM/VIP citation pack (`~/.claude/plans/snn_som_vip_citations.md`). SOM evoked widened 5-15 → **5-20 Hz** (Kvitsiani 2013 ACC 16±4 Hz NS-SOM tail + Ma 2010 V1 fold-ratio); VIP evoked widened 5-15 → **10-25 Hz** (Mesik 2015 V1 urethane 8.7±8.2 Hz × 2.5× awake-scaling from Niell & Stryker 2010). SOM/VIP spontaneous widened 1-5 → **2-10 Hz**. Every v4 number is primary-sourced; v3's round-number VIP 5 Hz lower bound is retracted — team lead acknowledged it was not sourced. Inference gaps for SOM/VIP are documented in the new §4.5.

---

## 1. Time-Unit Convention

| Quantity | Value | Source |
|---|---|---|
| Simulation timestep `dt` | **1 ms** | Team lead directive (2026-04-10) + `ModelConfig.dt = 1` (`src/config.py:83`) |
| Conversion | `rate_Hz = spike_count / window_ms * 1000` | Definition |
| `tau_l4 = 5` | → 5 ms membrane τ | `src/config.py:31` |
| `tau_l23 = 10` | → 10 ms membrane τ | `src/config.py:43` |
| `tau_adaptation = 200` | → 200 ms SSA τ | `src/config.py:32` |
| Per-stimulus window (training) | 50 timesteps = 50 ms | Plan line 159 ("Increase from 20 to 50") |

All SNN modules use `beta = exp(-dt/tau)` (plan line 35). With `dt = 1 ms` this reads:
- `beta_l4 = exp(-1/5) = 0.819` (plan line 47)
- `beta_l23 = exp(-1/10) = 0.905` (plan line 49)
- `rho_ssa = exp(-1/200) = 0.995` (plan line 53)

## 2. Stimulus Protocol for Rate Measurement

### 2.1 Evoked rate (primary gate)

For each of the 36 orientation channels `k` in `[0 ... 35]`:

1. **Target orientation:** `theta_k = k * 5.0` degrees (period 180°).
2. **Contrast:** `c = 1.0` (full contrast — "unit amplitude" per team lead's directive).
3. **Stimulus tensor:** `generate_grating(orientation=theta_k, contrast=1.0, n_orientations=36, sigma=12.0, n=2.0, c50=0.3)` from `src/stimulus/gratings.py:77`.
4. **Batch:** replicate `B = 16` times with independent network init states (to get a batch-level mean/std).
5. **Run:** feed the same stimulus for `T_total` consecutive timesteps with no interstimulus interval.
6. **Measurement window:** see §3.

Sweep over all 36 channels → stack into a `[36, B, N]` rate matrix per population.

### 2.2 Spontaneous rate

Feed a zero-valued stimulus (no population drive) for the same window. Spontaneous rates are reported per population, not per channel.

### 2.3 Max sustained rate

Drive the network with a single-channel saturating input at `5×` the unit amplitude, sustained for `T_total = 1000` timesteps (1 s) to catch any runaway. Used only as an upper-bound check; not the primary gate.

## 3. Warmup and Measurement Windows

The training-time window (50 timesteps) is **too short for reliable rate statistics**. Firing-rate validation uses a dedicated, longer window.

| Phase | T_warmup | T_measure | T_total | Rationale |
|---|---|---|---|---|
| **Primary evoked gate** | 50 ms | 450 ms | 500 ms | 5 × τ_l23 = 50 ms warmup lets the slowest fast population reach quasi-steady-state; 450 ms of spikes at 5 Hz = ~2.25 spikes/channel, enough for batch-level means but still short of the 200 ms SSA timescale so adaptation transients are bounded. |
| **Spontaneous** | 50 ms | 450 ms | 500 ms | Same. |
| **Max sustained** | 200 ms | 800 ms | 1000 ms | Longer window spans ≥4 × τ_adaptation so SSA has fully engaged; this is the steady-state rate *after* adaptation, which matches the biological "sustained" value. |

```python
rate_Hz = spike_count[:, T_warmup:T_total].sum(dim=time_axis) / (T_total - T_warmup) * 1000
# shape: [channel_sweep, batch, n_orientations]
```

**Reporting:** for each population, report
- `mean_rate_preferred`: rate on the channel that matches the stimulus orientation (diagonal element across the sweep)
- `std_rate_preferred`: std across the `B` batch dimension at the preferred channel
- `mean_rate_off_preferred`: mean of all non-preferred channels (off-diagonal)
- `peak_rate_any`: max over all (channel, batch) combinations — used for the "max sustained" gate

## 4. Biological Ranges (the gates themselves)

**Authoritative ruling (team lead, 2026-04-10, v3):** the primary gate is **stationary mouse V1** (quiet-wake, head-fixed, non-running) with primary-source citations from the Researcher's evidence pack. The older "running-state" ranges become the warn band upper bound. The plan file's macaque-like numbers are obsolete for gating and will be corrected later — **the truth lives in this document, not in the plan file.**

The network is assumed to be in stationary mode for all of Phase 1-3 validation. The Coder has added `SpikingConfig.stationary_mode: bool = True` (Task #17). The Validator reads this flag and picks the corresponding gate set:
- `stationary_mode = True`  → **§4.1 primary gate** (stationary)
- `stationary_mode = False` → the §4.2 warn upper bounds are promoted to primary (running regime); a separate evidence pack must be cited before validation can proceed in running mode.

### 4.1 Primary gate — STATIONARY mouse V1 (hard NO-GO if outside)

Every row cites a primary source. Rates are preferred-channel evoked responses to full-contrast drifting gratings in quiet-wake, head-fixed mice. For SOM/VIP, see §4.5 for the inference chain.

| Population | Evoked (Hz) — **PRIMARY GATE** | Spontaneous (Hz) | Primary source |
|---|---|---|---|
| L4 excitatory | **2 – 5** | 0.1 – 0.8 | Niell & Stryker 2010, *Neuron* 65:472-479, Fig 3D: evoked `2.9 ± 0.4 Hz`, spont `0.17 ± 0.07 Hz` (mean ± SEM), stationary awake mouse V1 |
| L2/3 pyramidal | **1 – 4** | 0.1 – 0.8 | Niell & Stryker 2010, *Neuron* 65:472-479: supragranular excitatory, stationary |
| PV fast-spiking | **2 – 25** | 0.5 – 10 | Atallah et al. 2012, *Neuron* 73:159-170: evoked mean `12.1 ± 9.6 Hz`, spont `2.1 ± 3.1 Hz`, stationary mouse V1 |
| SOM | **5 – 20** | 2 – 10 | **Primary:** Kvitsiani et al. 2013, *Nature* 498:363-366 (ACC, opto-tagged, awake behaving): NS-SOM `16±4 Hz` (n=13), WS-SOM `4±1 Hz` (n=22), weighted mean ≈7.6 Hz. **Supporting:** Ma et al. 2010, *J Neurosci* 30(43):14371-14379 (V1, urethane, cell-attached): "SOM evoked ∼3- to 5-fold weaker than PV". **Cross-area qualitative:** Gentet et al. 2012, *Nat Neurosci* 15:607 (barrel cortex, awake): SOM "tonically active in quiet wake". **See §4.5 for inference chain (ACC→V1 anchor).** |
| VIP | **10 – 25** | 2 – 10 | **Primary:** Mesik et al. 2015, *Front Neural Circuits* 9:22 (V1, urethane, loose-patch): VIP spont `1.3±2.2 Hz`, evoked `8.7±8.2 Hz`. **Awake scaling ×2.5:** Niell & Stryker 2010, *Neuron* 65:472-479 (stationary→running broad-spiking: `2.9→8.2 Hz`). Awake-corrected: spont ≈3–4 Hz, evoked ≈17–22 Hz; ranges widened to 2–10 / 10–25 for SD headroom. **Supporting:** Reimer et al. 2014, *Neuron* 84:355-362 (V1, whole-cell, awake): VIP+ depolarize `2.1±0.6 mV` during pupil dilation. **See §4.5 for inference chain (urethane→awake scaling).** |

Preferred-channel rates must fall inside the **Evoked** column. Off-preferred rates must fall inside the **Spontaneous** column (or lower).

### 4.2 Warn band — RUNNING mouse V1 (warn only, still a GO)

If a preferred-channel rate exceeds the stationary upper limit but stays within the running-state range below, the Validator emits a yellow `[WARN]` and continues. This interprets "above stationary" as "the network has drifted toward a locomotion-like activity regime", which is non-pathological but non-ideal for a stationary-mode test.

| Population | Warn upper bound (Hz) | Source for the upper bound |
|---|---|---|
| L4 excitatory | 15 | Niell & Stryker 2010, running broad-spiking: `8.2±0.9 Hz` evoked + SD headroom |
| L2/3 pyramidal | 10 | Niell & Stryker 2010, running supragranular equivalent |
| PV fast-spiking | 100 | Atallah 2012 + published PV running maxima (upper literature bound) |
| SOM | 40 | **No V1 running-state SOM Hz data.** 2× primary upper (§4.1 20 Hz × 2) as a principled extension; still well below the §4.3 biophysical ceiling of 80 Hz. |
| VIP | 50 | **No V1 running-state VIP Hz data.** 2× primary upper (§4.1 25 Hz × 2); supported qualitatively by Reimer et al. 2014, which shows VIP depolarizes with arousal — arousal-driven firing extensions can exceed the stationary primary range without being pathological. Still below the §4.3 biophysical ceiling of 100 Hz. |

L4/L2/3/PV warn bounds come from Niell & Stryker 2010's locomotion condition (same paper as the primary gate). SOM/VIP have no equivalent awake V1 dataset for running state — see §4.5 for the reasoning.

### 4.3 Hard NO-GO ceiling (pathological, regardless of state)

| Condition | Threshold | Rationale |
|---|---|---|
| Excitatory preferred-channel rate | > 300 Hz | Physical upper bound for sustained spiking in pyramidal / L4 stellate neurons |
| PV preferred-channel rate | > 500 Hz | Absolute physical limit for fast-spiking interneurons |
| Any responsive channel | = 0 Hz during stimulus | Dead unit — network broken |
| Any rate | NaN / Inf | Numerical instability |
| Spikes | not binary `{0, 1}` | Forward pass returning something other than Heaviside output |
| Cross-population asymmetry | L4 rate > 10 × L2/3 rate | Feedforward amplification broken (L4 → L2/3 should roughly preserve rate magnitude in stationary mode) |

### 4.4 Gate policy (three tiers)

1. **GO** — preferred-channel rate inside the **§4.1 primary (stationary)** evoked range, AND off-preferred inside the §4.1 spontaneous range, AND no §4.3 pathology triggered.
2. **WARN** — preferred-channel rate above the §4.1 upper limit but still within the §4.2 running-state ceiling. Logged yellow; overall verdict remains GO unless something else fails.
3. **NO-GO** — any of:
    - Preferred rate **below** the §4.1 lower limit (e.g., L4 at 1 Hz).
    - Preferred rate **above** the §4.2 warn upper bound (e.g., L4 at 20 Hz).
    - Any §4.3 pathological condition triggered.
    - Off-preferred rate above the §4.1 spontaneous ceiling.

### 4.5 Inference chain for SOM/VIP (provenance disclosure)

**Why this section exists:** The SOM and VIP ranges in §4.1 are not drawn from a single paper in the way the L4/L2/3/PV ranges are. A future Validator reading this doc in six months should be able to reconstruct the full evidence chain in one hop. This section lays it out explicitly.

**Gap 1 — No awake + V1 + opto-tagged + Hz + error bars exists for SOM or VIP** (as of 2026-04-10).

The Researcher audited the modern V1 SOM/VIP literature for the citation pack (`~/.claude/plans/snn_som_vip_citations.md` §E). Every widely-cited paper converged on two-photon calcium imaging (ΔF/F), which reports event rates that are not directly convertible to spike Hz:
- Dipoppa et al. 2018 *Neuron* — calcium
- Pakan et al. 2016 *eLife* — calcium
- Millman et al. 2020 *eLife* — calcium
- Fu et al. 2014 *Cell* — calcium

**This is a real literature gap, not a search failure.** The v4 gate accepts that SOM/VIP Hz targets must be assembled from multiple adjacent datasets rather than a single gold-standard paper.

**Gap 2 — SOM primary Hz anchor is cross-area (ACC, not V1).**

Kvitsiani et al. 2013 *Nature* 498:363 is the **only opto-tagged, awake, extracellular SOM dataset with error bars** in rodent cortex. It reports PFC/ACC, not V1. The v4 gate accepts this anchor because:
1. SOM intrinsic biophysics (low max firing rate, strong spike-frequency accommodation, thin apical inhibition) are **conserved across cortical areas** — Kvitsiani's 16±4 Hz NS-SOM / 4±1 Hz WS-SOM split matches the qualitative description in V1-specific papers (Ma 2010, Adesnik 2012, Gentet 2012 barrel).
2. The V1-specific electrophysiology (Ma 2010) is **urethane-anesthetized, with Hz values only in figures** the Researcher could not extract. The strongest text-level quote is "SOM evoked ∼3- to 5-fold weaker than PV", which anchors a *ratio*, not an absolute Hz.
3. Using Ma 2010's fold-ratio + Atallah 2012's PV numbers gives SOM awake evoked ≈ Atallah PV (12.1 Hz) ÷ 3-5 = **2.4–4 Hz under urethane**. Scaled to awake via Niell & Stryker's ×2.5 factor → **~6–10 Hz**. This overlaps the Kvitsiani weighted-mean of 7.6 Hz, so the cross-area anchor is consistent with the V1-specific fold-ratio inference.

**Conclusion:** SOM 5-20 Hz evoked brackets both the Kvitsiani direct measurement (NS+WS) and the Ma 2010 inferred V1 range. A network outside 5-20 Hz is inconsistent with *both* independent evidence streams.

**Gap 3 — VIP primary data is urethane-anesthetized; awake values are extrapolated.**

Mesik et al. 2015 *Front Neural Circuits* 9:22 is the best V1-specific VIP electrophysiology dataset: **urethane anesthesia, loose-patch, direct Hz**. It reports VIP spont `1.3 ± 2.2 Hz` / evoked `8.7 ± 8.2 Hz`. The v4 gate scales these to awake via a ×2.5 factor:

- **Where the ×2.5 comes from:** Niell & Stryker 2010 *Neuron* 65:472 measured stationary vs. running V1 broad-spiking (putative pyramidal) rates: `2.9 → 8.2 Hz` (evoked), a ratio of ~2.8. The Researcher rounded to ×2.5 for conservative scaling.
- **Limitation:** Niell & Stryker's ratio is measured on *pyramidal* cells. Applying it to VIP interneurons is an extrapolation. However, broad arousal modulation is known to affect VIP as much or more than pyramidal cells (Fu 2014 — locomotion-gated VIP; Reimer 2014 *Neuron* 84:355 — VIP+ depolarize `2.1±0.6 mV` during pupil dilation), so the extrapolation is plausible and likely conservative.
- **Check:** Mesik 2015 × 2.5 gives VIP awake evoked ~17–22 Hz. The v4 gate uses **10–25 Hz** to widen for SD headroom on both sides. The lower bound of 10 Hz is conservatively pulled down from the ~17 Hz awake-scaled mean to accommodate low-contrast stimuli; the upper bound of 25 Hz is the awake-scaled mean + ~0.4 SD.

**Conclusion:** VIP 10-25 Hz evoked is the awake-corrected Mesik 2015 range with SD headroom. Dropping the lower bound to 5 Hz (as v3 had) is not supported by any primary source — a 5 Hz VIP would imply urethane-equivalent suppression, which contradicts the awake-mode assumption of this entire gate.

**What this §4.5 commits the Validator to:**
1. Any v4+ revision that changes SOM or VIP ranges must extend the revision log with the new primary source that justifies the change — no "round-number cleanup" allowed.
2. If a future paper delivers a direct awake V1 opto-tagged Hz measurement for SOM or VIP, that paper replaces the current primary source and the gate is re-derived; gaps 1-3 in this section are updated or removed accordingly.
3. If the SNN port ever runs in **non-stationary** mode (`SpikingConfig.stationary_mode = False`), the gate is re-derived from the Niell & Stryker running-state values — the current §4.2 warn band is *not* a running-mode primary gate.

### 4.6 Additional biological metrics (not primary gates, tracked for reporting)

From the Researcher's handout:

| Metric | Target | Gate level |
|---|---|---|
| Fano factor (alert V1) | 0.5 – 1.5 | warn |
| ISI CV, regular-spiking | 0.5 – 1.0 | warn |
| ISI CV, fast-spiking (PV) | 0.3 – 0.5 | warn |
| Orientation selectivity index (OSI) | > 0.3 | warn |
| E/I ratio (spikes) | ~4:1 (excitatory : inhibitory) | warn |
| Gamma-band oscillation | 30–80 Hz | informational |

These are computed on the same 450 ms evoked window.

## 5. Implementation Template (Python pseudocode)

```python
import torch
from src.config import ModelConfig, SpikingConfig
from src.spiking.network import SpikingLaminarV1V2Network
from src.stimulus.gratings import generate_grating

def measure_firing_rates(net, cfg, *, condition, B=16,
                         T_warmup=50, T_measure=450, seed=0):
    """Measure per-population firing rates on a 36-channel orientation sweep.

    Returns a dict: population_name -> {preferred_mean, preferred_std,
                                        off_pref_mean, peak}
    """
    torch.manual_seed(seed)
    T_total = T_warmup + T_measure
    N = cfg.n_orientations

    # Spike accumulators: [36 sweep, batch, N channels]
    spike_buf = {pop: torch.zeros(N, B, N) for pop in
                 ("l4", "l23", "som", "vip")}
    pv_buf    = torch.zeros(N, B, 1)  # rate-based

    for k in range(N):
        theta = torch.full((B,), k * 5.0)
        if condition == "evoked":
            stim = generate_grating(theta, contrast=torch.ones(B),
                                    n_orientations=N, sigma=12.0)
        elif condition == "spontaneous":
            stim = torch.zeros(B, N)
        elif condition == "max":
            stim = 5.0 * generate_grating(theta, contrast=torch.ones(B),
                                          n_orientations=N, sigma=12.0)
        else:
            raise ValueError(condition)

        state = net.init_state(batch_size=B)
        for t in range(T_total):
            state, aux = net.step(stim, state)
            if t >= T_warmup:
                spike_buf["l4"][k]  += aux["z_l4"]
                spike_buf["l23"][k] += aux["z_l23"]
                spike_buf["som"][k] += aux["z_som"]
                spike_buf["vip"][k] += aux["z_vip"]
                pv_buf[k]           += aux["r_pv"]  # rate integral

    # Convert to Hz
    rates = {}
    for pop, buf in spike_buf.items():
        hz = buf / (T_measure / 1000.0)  # spikes / seconds
        preferred = hz.diagonal(dim1=0, dim2=2)   # [B, 36]
        off_pref_mask = 1 - torch.eye(N).unsqueeze(1).expand(-1, B, -1)
        off_pref = (hz * off_pref_mask).sum((0, 2)) / off_pref_mask.sum((0, 2))
        rates[pop] = {
            "preferred_mean": preferred.mean().item(),
            "preferred_std":  preferred.std().item(),
            "off_pref_mean":  off_pref.mean().item(),
            "peak":           hz.max().item(),
        }

    # PV: average rate × 1 (already Hz if tau is in ms)
    rates["pv"] = {
        "preferred_mean": (pv_buf / T_measure).mean().item() * 1000,
        "preferred_std":  (pv_buf / T_measure).std().item() * 1000,
        "off_pref_mean":  float("nan"),  # PV has no orientation channels
        "peak":           (pv_buf / T_measure).max().item() * 1000,
    }
    return rates
```

The Validator will run this once Phase 1.10 lands (or an adapted version based on the Coder's final API). It is **not** run during Phase 1.0-1.9; firing rates are a Phase 2 / post-training gate because an untrained spiking network will not fire at biologically plausible rates.

## 6. Per-Phase Application

| Phase | When to run | Passing condition |
|---|---|---|
| Phase 1.10 | Instantiation smoke test only (no training) | Binary spikes, no NaN/Inf, forward pass clean, correct shapes, gradient flow. **No firing-rate gate.** (Untrained networks do not fire at biological rates — see Researcher evidence pack §A.6 / team-lead ruling.) |
| Phase 2 (Pass 1 sensory scaffold) | After scaffold training converges, with `SpikingConfig.stationary_mode = True` | Preferred-channel rates inside **§4.1 stationary primary gate**; off-preferred inside stationary spontaneous range. WARN if inside §4.2 running upper band. Any §4.3 pathology = NO-GO. Gate policy per §4.4. |
| Phase 3 (Pass 2/3 V2+feedback) | After each pass, stationary mode | Same gate as Phase 2, plus rates must stay stable across the feedback ramp (no population drifts out of the §4.1 stationary range during the ramp). |
| Phase 4 (Pass 4 objective sweep) | Per config, stationary mode | All three regimes (dampening, transition, sharpening) must satisfy the §4.1 stationary primary gate. Even the dampening regime's L2/3 must stay ≥ 1 Hz at preferred (the §4.1 stationary lower bound) — "it's dampening" is not an excuse for a dead population. |

## 7. Assumptions That Must Be Verified

The methodology above assumes four things about the Coder's implementation that the Validator will verify by inspection:

1. **`net.step(stim, state) → (state, aux)` where `aux` contains `z_*` spike tensors.** If the Coder uses a different API (e.g., returning spikes as part of `state` itself), the methodology template is updated, not the gates.
2. **`dt = 1 ms` is baked into the time constants.** If the Coder chooses a different `dt` (e.g., 0.5 ms for stability), all `T_warmup` / `T_measure` values must scale, and so do the firing-rate conversion formulas in §3.
3. **Spikes are binary `{0, 1}`, not graded.** The validation script checks this explicitly.
4. **`SpikingConfig.stationary_mode: bool = True` exists and is honored** (Task #17). The Validator reads this flag at gate-selection time: `True` → §4.1 primary stationary gate; `False` → running-mode validation is not yet supported and the Validator will abort with NO-GO and an explicit "running-mode evidence pack required" message.

If any of these assumptions is violated, the Validator will report NO-GO with the specific assumption that failed, and the team lead must decide whether to patch the methodology or the implementation.
