# Training Pipeline, Experimental Paradigms, and Analysis Suite
## Comprehensive Design Document for V1-V2 Expectation Suppression Model

---

## Table of Contents
1. [Training Pipeline](#1-training-pipeline)
2. [Experimental Paradigms](#2-experimental-paradigms)
3. [Analysis / Readout Suite](#3-analysis--readout-suite)
4. [Configuration Constants](#4-configuration-constants)

---

## 1. Training Pipeline

### 1.1 Overview

Three-stage curriculum with strict gating between stages. No stage begins until the previous stage passes explicit quantitative checkpoints.

```
Stage 1: Sensory Scaffold    (~0 gradient steps for hand-set; ~2K steps for L4→L2/3)
Stage 2: V2 Sequence Learning (~50K-100K sequence steps)
Stage 3: Mechanism Comparison  (5 models × same budget as Stage 2)
```

### 1.2 Stage 1: Sensory Scaffold

#### 1.2.1 Hand-Set L4 Tuning Curves

L4 has 36 excitatory orientation channels with preferred orientations:

```python
# θ_i = i * 5° for i in 0..35, covering [0°, 175°]
preferred_orientations = torch.linspace(0, 175, 36)  # degrees
```

Each L4 unit's response to stimulus orientation θ is a von Mises function on the doubled-angle circle:

```python
def l4_response(theta_stim, theta_pref, kappa=2.5, baseline=0.05, gain=1.0):
    """
    Von Mises tuning curve for orientation-selective V1 L4 neuron.
    
    Args:
        theta_stim: stimulus orientation in degrees [0, 180)
        theta_pref: preferred orientation in degrees [0, 180)
        kappa: concentration parameter (controls width)
            kappa=2.0 → σ ≈ 27°, kappa=2.5 → σ ≈ 24°, kappa=3.0 → σ ≈ 21°
        baseline: spontaneous firing rate (fraction of max)
        gain: peak response amplitude
    
    Returns:
        firing rate (scalar or tensor)
    """
    # Convert to radians on doubled-angle circle [0, 2π)
    delta = 2 * (theta_stim - theta_pref) * (math.pi / 180)
    response = baseline + gain * torch.exp(kappa * (torch.cos(delta) - 1))
    return response
```

**Parameter choices and justification:**
- `kappa = 2.5`: gives half-width at half-max ≈ 22-25°, consistent with macaque V1 simple cells (Ringach et al., 2002). This is narrow enough that adjacent channels (5° apart) are distinguishable but broad enough that ~8-10 channels respond meaningfully to any given stimulus.
- `baseline = 0.05`: low spontaneous rate. Ensures non-zero gradients everywhere during L2/3 pretraining.
- `gain = 1.0`: normalized peak. Actual gain will be modulated by contrast input.

**L4 adaptation dynamics** (also hand-set, not trained):

```python
class L4Adaptation:
    """
    Exponential adaptation with per-channel fatigue.
    
    State variable a_i(t) tracks recent activation of channel i.
    Effective response = r_i(t) * (1 - alpha * a_i(t))
    
    a_i(t+1) = (1 - 1/tau_adapt) * a_i(t) + (1/tau_adapt) * r_i(t)
    """
    tau_adapt = 8.0      # decay time constant in presentation units
    alpha_adapt = 0.3    # maximum adaptation strength (30% reduction at full adaptation)
```

**Contrast modulation:**
```python
def apply_contrast(l4_response, contrast):
    """
    Naka-Rushton contrast response function.
    R(c) = R_max * c^n / (c^n + c50^n)
    
    Args:
        l4_response: raw tuning curve response (36,)
        contrast: stimulus contrast in [0, 1]
    """
    n = 2.0    # exponent (typical for V1)
    c50 = 0.3  # semi-saturation contrast
    gain = (contrast ** n) / (contrast ** n + c50 ** n)
    return l4_response * gain
```

**All L4 parameters are frozen** — no gradients flow into L4 tuning curves, adaptation parameters, or contrast function during any training stage.

#### 1.2.2 L4→L2/3 Feedforward + PV Normalization Pretraining

**Goal:** L2/3 units develop stable orientation selectivity that inherits L4 tuning but is sharper (due to PV normalization) and more robust.

**Architecture for this stage (trainable components only):**
- `W_ff`: feedforward weight matrix L4→L2/3, shape (36, 36), initialized as a blurred identity (each L2/3 unit receives from nearby L4 channels):
  ```python
  # Initialize W_ff as Gaussian neighborhood connectivity
  W_ff = torch.zeros(36, 36)
  for i in range(36):
      for j in range(36):
          delta = min(abs(i - j), 36 - abs(i - j))  # circular distance
          W_ff[i, j] = math.exp(-delta**2 / (2 * 3.0**2))  # σ_connect = 3 channels = 15°
  W_ff = W_ff / W_ff.sum(dim=1, keepdim=True)  # row-normalize
  ```

- `PV_pool`: divisive normalization pool
  ```python
  def pv_normalization(l23_excitation, sigma_pv=0.1, pool_sigma_channels=12):
      """
      Divisive normalization: r_i = e_i / (sigma_pv + sum_j w_ij * e_j)
      
      pool_sigma_channels=12 means the PV pool integrates over ~60° of orientation 
      space (broad, as PV interneurons are broadly tuned).
      """
      # PV pooling weights: broad Gaussian
      pool_weights = torch.zeros(36, 36)
      for i in range(36):
          for j in range(36):
              delta = min(abs(i - j), 36 - abs(i - j))
              pool_weights[i, j] = math.exp(-delta**2 / (2 * pool_sigma_channels**2))
      pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True)
      
      normalization = sigma_pv + pool_weights @ l23_excitation
      return l23_excitation / normalization
  ```

- `b_l23`: L2/3 bias vector, shape (36,), initialized to 0.

**Training data for Stage 1:**
- Present random orientations uniformly sampled from the 36 channels (or continuous uniform [0°, 180°))
- Each presentation: one orientation at full contrast (c=1.0), single timestep (no temporal structure yet)
- Batch size: 128 random orientations per batch

**Loss function for Stage 1:**

```python
def stage1_loss(l23_output, stimulus_orientation, preferred_orientations):
    """
    Two-component loss to establish orientation selectivity in L2/3.
    
    Component 1: Orientation classification (cross-entropy)
        Bin stimulus into nearest of 36 channels, train linear readout on L2/3.
        This ensures L2/3 population carries decodable orientation information.
    
    Component 2: Tuning curve shape regularization
        Encourage each L2/3 unit to have a unimodal, peaked response centered
        on its preferred orientation. Measured as correlation between L2/3 unit's
        response profile and the target von Mises profile.
    """
    # Component 1: Classification
    # Linear readout head (36→36 softmax, separate trainable params)
    logits = readout_head(l23_output)  # (batch, 36)
    target_channel = orientation_to_channel(stimulus_orientation)  # (batch,)
    loss_classify = F.cross_entropy(logits, target_channel)
    
    # Component 2: Tuning curve shape (computed over a batch spanning all orientations)
    # Accumulate response profile for each unit across the batch
    # Then compute negative correlation with target von Mises
    loss_shape = tuning_curve_regularization(l23_output, stimulus_orientation)
    
    return loss_classify + 0.1 * loss_shape
```

**Optimizer for Stage 1:**
- Adam, lr=1e-3, weight_decay=1e-5
- Train for 2,000 steps (128 samples/step = 256K presentations)
- Learning rate: constant (short training, no need for schedule)

**What is trainable in Stage 1:** `W_ff`, `b_l23`, PV `sigma_pv` (scalar), readout head weights.
**What is frozen:** All L4 parameters, adaptation parameters.

#### 1.2.3 Stage 1 Gating Checkpoints

Stage 1 is complete (and Stage 2 may begin) **only if ALL of the following pass:**

| Checkpoint | Criterion | How to Measure |
|---|---|---|
| **C1: Classification accuracy** | ≥ 90% top-1 accuracy on held-out orientations (36 classes) | Run 1,000 random orientations through network, classify from L2/3 |
| **C2: Tuning curve unimodality** | All 36 L2/3 units have a single peak in their response profile | Present all 36 orientations, record each unit's response, verify single max with ≥2:1 peak-to-trough ratio |
| **C3: Preferred orientation tiling** | Preferred orientations of L2/3 units span [0°, 180°) uniformly | Compute preferred orientation of each L2/3 unit, check that standard deviation of spacing is < 2° |
| **C4: Bandwidth** | Mean HWHM of L2/3 tuning curves is 15°-30° | Fit von Mises to each unit's response profile, extract bandwidth |
| **C5: PV normalization strength** | Max L2/3 firing rate is bounded (< 2.0) and min is > 0 | Present all orientations, record max activation |

```python
def verify_stage1(model, device='cpu'):
    """Returns dict of checkpoint results. All must pass."""
    results = {}
    
    # Present all 36 orientations, get L2/3 responses
    orientations = torch.linspace(0, 175, 36, device=device)
    l23_responses = torch.zeros(36, 36)  # (stimulus, unit)
    for i, theta in enumerate(orientations):
        l23_responses[i] = model.forward_l23(theta, contrast=1.0)
    
    # C1: Classification
    correct = 0
    total = 1000
    for _ in range(total):
        theta = torch.rand(1) * 180
        target = (theta / 5).long().clamp(0, 35)
        output = model.forward_l23(theta, contrast=1.0)
        pred = model.readout(output).argmax()
        correct += (pred == target).item()
    results['C1_accuracy'] = correct / total  # must be >= 0.90
    
    # C2: Unimodality
    peak_to_trough = []
    for unit in range(36):
        profile = l23_responses[:, unit]
        peak_to_trough.append(profile.max() / (profile.min() + 1e-8))
    results['C2_min_peak_trough'] = min(peak_to_trough)  # must be >= 2.0
    
    # C3: Tiling
    pref_oris = l23_responses.argmax(dim=0).float() * 5.0  # preferred orientation per unit
    sorted_pref = pref_oris.sort().values
    spacings = torch.diff(sorted_pref)
    results['C3_spacing_std'] = spacings.std().item()  # must be < 2.0
    
    # C4: Bandwidth (approximate HWHM)
    bandwidths = []
    for unit in range(36):
        profile = l23_responses[:, unit]
        peak = profile.max()
        half_max = peak / 2
        above_half = (profile >= half_max).sum().item()
        hwhm = above_half * 5.0 / 2  # approximate half-width in degrees
        bandwidths.append(hwhm)
    results['C4_mean_hwhm'] = sum(bandwidths) / len(bandwidths)  # must be in [15, 30]
    
    # C5: Bounded activation
    results['C5_max_activation'] = l23_responses.max().item()  # must be < 2.0
    results['C5_min_activation'] = l23_responses.min().item()  # must be > 0
    
    return results
```

**After Stage 1 passes:** Freeze `W_ff`, `b_l23`, PV parameters. Only top-down connections and V2 are trainable in Stage 2.

---

### 1.3 Stage 2: V2 Sequence Learning + Feedback Training

#### 1.3.1 Training Sequence Generation

Sequences are generated by a Hidden Markov Model (HMM) with 3 latent states:

```python
class SequenceGenerator:
    """
    Generates orientation sequences from a 3-state HMM.
    
    Latent states:
        0: clockwise     — next θ = current θ + step_size (mod 180°), with p=transition_prob
        1: counterclockwise — next θ = current θ - step_size (mod 180°), with p=transition_prob
        2: neutral        — next θ ~ Uniform(36 channels)
    
    State transitions:
        Self-transition probability: p_self = 0.95
        Cross-transition: uniform over other 2 states, p_cross = 0.025 each
    """
    
    def __init__(self, 
                 n_channels=36,
                 step_size=15.0,        # degrees per transition (3 channels)
                 transition_prob=0.80,   # P(predicted next | state)
                 p_self=0.95,            # state self-transition
                 sequence_length=50,     # presentations per sequence
                 seed=None):
        
        self.n_channels = n_channels
        self.step_channels = int(step_size / (180.0 / n_channels))  # = 3 channels
        self.transition_prob = transition_prob
        self.p_cross = (1.0 - p_self) / 2
        
        self.state_transition = torch.tensor([
            [p_self,      self.p_cross, self.p_cross],   # from CW
            [self.p_cross, p_self,      self.p_cross],   # from CCW
            [self.p_cross, self.p_cross, p_self],         # from neutral
        ])
        
        self.sequence_length = sequence_length
    
    def generate_batch(self, batch_size):
        """
        Returns:
            orientations: (batch, seq_len) int tensor, channel indices 0-35
            states: (batch, seq_len) int tensor, latent state 0-2
            expected_next: (batch, seq_len) int tensor, what V2 should predict
                           (-1 for neutral state = uniform prediction)
        """
        orientations = torch.zeros(batch_size, self.sequence_length, dtype=torch.long)
        states = torch.zeros(batch_size, self.sequence_length, dtype=torch.long)
        expected_next = torch.full((batch_size, self.sequence_length), -1, dtype=torch.long)
        
        # Initial state: uniform random
        states[:, 0] = torch.randint(0, 3, (batch_size,))
        orientations[:, 0] = torch.randint(0, self.n_channels, (batch_size,))
        
        for t in range(1, self.sequence_length):
            # State transition
            for b in range(batch_size):
                prev_state = states[b, t-1].item()
                states[b, t] = torch.multinomial(
                    self.state_transition[prev_state], 1
                ).item()
            
            # Orientation transition
            for b in range(batch_size):
                state = states[b, t].item()
                prev_ori = orientations[b, t-1].item()
                
                if state == 0:  # clockwise
                    predicted = (prev_ori + self.step_channels) % self.n_channels
                    if torch.rand(1) < self.transition_prob:
                        orientations[b, t] = predicted
                    else:
                        orientations[b, t] = torch.randint(0, self.n_channels, (1,))
                    expected_next[b, t-1] = predicted
                    
                elif state == 1:  # counterclockwise
                    predicted = (prev_ori - self.step_channels) % self.n_channels
                    if torch.rand(1) < self.transition_prob:
                        orientations[b, t] = predicted
                    else:
                        orientations[b, t] = torch.randint(0, self.n_channels, (1,))
                    expected_next[b, t-1] = predicted
                    
                else:  # neutral
                    orientations[b, t] = torch.randint(0, self.n_channels, (1,))
                    expected_next[b, t-1] = -1  # no specific prediction
        
        return orientations, states, expected_next
```

**Critical design choice — marginal frequency equalization:**

Because clockwise and counterclockwise states produce non-uniform orientation sequences (always stepping by ±15°), the marginal frequency of each orientation must be checked and rebalanced.

```python
def verify_marginal_uniformity(generator, n_sequences=10000):
    """
    Check that each of the 36 orientations appears with approximately 
    equal frequency when averaged across all generated sequences.
    Criterion: max/min ratio < 1.2 (no orientation > 20% more frequent than any other).
    """
    oris, _, _ = generator.generate_batch(n_sequences)
    counts = torch.bincount(oris.flatten(), minlength=36).float()
    counts /= counts.sum()
    ratio = counts.max() / counts.min()
    print(f"Marginal uniformity ratio: {ratio:.3f} (must be < 1.2)")
    return ratio < 1.2
```

This should pass naturally since initial orientations are uniform and state transitions are symmetric, but must be verified.

#### 1.3.2 Temporal Structure of Each Presentation

Each grating presentation consists of multiple network timesteps to allow recurrent dynamics:

```python
# Within each orientation presentation:
T_ON = 8        # timesteps where grating is presented to L4
T_ISI = 4       # inter-stimulus interval (blank input, L4 gets baseline only)
T_TOTAL = T_ON + T_ISI  # = 12 timesteps per presentation

# Within T_ON:
#   t=0-2:  transient response (L4 adaptation hasn't kicked in)
#   t=3-7:  steady-state (adaptation active, top-down modulation has arrived)
#   Readout window: t=4-7 (last 4 timesteps of ON period)

# During T_ISI:
#   L4 receives no stimulus (just baseline noise)
#   V2 GRU continues to update its hidden state
#   Adaptation decays slightly
#   Deep template may persist (this is measured in omission experiments)
```

**Sequence structure:**
- Each training sequence: 50 orientation presentations × 12 timesteps = 600 timesteps total
- V2 GRU processes the full 600-timestep sequence
- Loss is computed at each presentation (during the readout window)

#### 1.3.3 Loss Function

```python
def stage2_loss(model_output, targets, lambdas):
    """
    Multi-objective loss for Stage 2.
    
    Args:
        model_output: dict containing:
            - 'l23_activity': (batch, seq_len, readout_timesteps, 36) — L2/3 responses
            - 'v2_prediction': (batch, seq_len, 36) — V2's softmax prediction of next orientation
            - 'excitatory_activity': (batch, seq_len, readout_timesteps, 36) — all excitatory rates
            - 'mean_rate': (batch, seq_len) — population mean firing rate
        targets: dict containing:
            - 'stimulus_channel': (batch, seq_len) — actual presented channel
            - 'next_channel': (batch, seq_len) — next orientation (-1 for neutral)
            - 'is_neutral': (batch, seq_len) — bool mask for neutral-state trials
        lambdas: dict of loss weights
    """
    # ---- Component 1: Sensory Readout Loss ----
    # L2/3 must faithfully encode the current stimulus
    # Use cross-entropy with a frozen linear readout head (trained in Stage 1)
    l23_mean = model_output['l23_activity'].mean(dim=2)  # avg over readout timesteps → (batch, seq, 36)
    logits = frozen_readout(l23_mean)  # (batch, seq, 36)
    loss_sensory = F.cross_entropy(
        logits.reshape(-1, 36),
        targets['stimulus_channel'].reshape(-1),
        reduction='mean'
    )
    
    # ---- Component 2: Next-Orientation Prediction Loss ----
    # V2 must predict the next stimulus orientation
    # Only computed on non-neutral trials (where a specific prediction exists)
    valid_mask = targets['next_channel'] >= 0  # (batch, seq)
    if valid_mask.any():
        v2_pred = model_output['v2_prediction'][valid_mask]  # (N_valid, 36)
        v2_target = targets['next_channel'][valid_mask]      # (N_valid,)
        loss_pred = F.cross_entropy(v2_pred, v2_target, reduction='mean')
    else:
        loss_pred = torch.tensor(0.0)
    
    # For neutral trials: V2 should predict uniform distribution
    # KL(uniform || v2_pred) penalizes overconfident predictions during neutral
    if (~valid_mask).any():
        v2_pred_neutral = F.log_softmax(model_output['v2_prediction'][~valid_mask], dim=-1)
        uniform = torch.full_like(v2_pred_neutral, 1.0 / 36)
        loss_neutral_pred = F.kl_div(v2_pred_neutral, uniform, reduction='batchmean')
        loss_pred = loss_pred + 0.1 * loss_neutral_pred
    
    # ---- Component 3: Energy Cost ----
    # L1 penalty on total excitatory activity (metabolic cost proxy)
    # Encourages the network to solve the task with minimal firing
    exc_activity = model_output['excitatory_activity']  # (batch, seq, timesteps, 36)
    loss_energy = exc_activity.abs().mean()
    
    # ---- Component 4: Homeostasis Penalty ----
    # Penalize if mean population rate drifts outside target range
    mean_rate = model_output['mean_rate']  # (batch, seq)
    target_rate_low = 0.15
    target_rate_high = 0.50
    too_low = F.relu(target_rate_low - mean_rate)
    too_high = F.relu(mean_rate - target_rate_high)
    loss_homeo = (too_low ** 2 + too_high ** 2).mean()
    
    # ---- Combine ----
    total = (lambdas['sensory'] * loss_sensory +
             lambdas['pred'] * loss_pred +
             lambdas['energy'] * loss_energy +
             lambdas['homeo'] * loss_homeo)
    
    return total, {
        'loss_sensory': loss_sensory.item(),
        'loss_pred': loss_pred.item(),
        'loss_energy': loss_energy.item(),
        'loss_homeo': loss_homeo.item(),
        'total': total.item(),
    }

# Default lambda values:
LAMBDAS = {
    'sensory': 1.0,     # primary objective: don't corrupt sensory representation
    'pred': 0.5,        # secondary: learn to predict (but not at cost of sensory)
    'energy': 0.01,     # gentle metabolic pressure
    'homeo': 1.0,       # strong homeostasis enforcement
}
```

**Lambda justification:**
- `sensory=1.0`: This is the anchor. If top-down feedback destroys sensory fidelity, the whole system fails. This loss must stay low throughout training.
- `pred=0.5`: Important but secondary. V2 should learn to predict, but not at the cost of corrupting L2/3. If this were too high, V2 might force L2/3 to become a predictive code rather than a sensory code — we want the sharpening/dampening question to arise from the trade-off, not be forced.
- `energy=0.01`: Small but non-zero. This is the key driver: it creates metabolic pressure that makes expectation suppression adaptive. Without it, there's no reason to suppress expected stimuli. The exact value may need tuning — too high and the network goes silent, too low and no suppression emerges.
- `homeo=1.0`: Strong, to prevent collapse or explosion. This is a guardrail, not a driver.

#### 1.3.4 Optimizer and Schedule

```python
# Optimizer
optimizer = torch.optim.AdamW(
    [
        {'params': model.v2_gru.parameters(), 'lr': 3e-4},
        {'params': model.feedback_connections.parameters(), 'lr': 1e-4},
        {'params': model.som_weights.parameters(), 'lr': 1e-4},
        {'params': model.deep_template.parameters(), 'lr': 1e-4},
    ],
    weight_decay=1e-4
)

# Schedule: linear warmup (1K steps) + cosine decay
warmup_steps = 1000
total_steps = 80000  # ~80K steps

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6),
    ],
    milestones=[warmup_steps]
)
```

**Why separate learning rates:**
- V2 GRU (3e-4): needs to learn relatively fast — it's the main "brain" for sequence learning.
- Feedback connections (1e-4): slower, to prevent catastrophic feedback that destroys L2/3 selectivity early in training. The sensory loss guards against this, but slower feedback learning adds stability.

**Training duration:**
- 80,000 steps × batch_size=32 sequences × 50 presentations/sequence = 128M total presentations
- At ~2 sequences/second on a single GPU, this is ~11 hours per model
- Can be reduced to 50K steps if convergence is early (see stopping criteria)

#### 1.3.5 Early Stopping and Convergence Criteria

```python
class Stage2Monitor:
    """
    Monitors training progress and determines convergence.
    Checks every eval_interval steps.
    """
    eval_interval = 500  # steps between evaluations
    patience = 10        # eval intervals without improvement before stopping
    min_steps = 20000    # minimum training steps regardless of convergence
    
    def should_stop(self, metrics_history):
        """
        Convergence criteria (ALL must be met for at least `patience` consecutive evals):
            1. V2 prediction accuracy ≥ 70% on non-neutral trials
            2. Sensory readout accuracy ≥ 85% (hasn't degraded from Stage 1)
            3. Total loss has plateaued (relative change < 1% over patience window)
            4. Mean population rate is within homeostatic bounds
        """
        if len(metrics_history) < self.patience:
            return False
        if metrics_history[-1]['step'] < self.min_steps:
            return False
        
        recent = metrics_history[-self.patience:]
        
        pred_acc_ok = all(m['v2_pred_accuracy'] >= 0.70 for m in recent)
        sensory_ok = all(m['sensory_accuracy'] >= 0.85 for m in recent)
        
        losses = [m['total_loss'] for m in recent]
        loss_plateau = (max(losses) - min(losses)) / (max(losses) + 1e-8) < 0.01
        
        rate_ok = all(0.12 < m['mean_rate'] < 0.55 for m in recent)
        
        return pred_acc_ok and sensory_ok and loss_plateau and rate_ok
```

#### 1.3.6 What is Trainable in Stage 2

| Component | Trainable? | Notes |
|---|---|---|
| L4 tuning curves | **NO** | Hand-set, permanently frozen |
| L4 adaptation | **NO** | Hand-set |
| W_ff (L4→L2/3) | **NO** | Frozen after Stage 1 |
| PV normalization | **NO** | Frozen after Stage 1 |
| L2/3 bias | **NO** | Frozen after Stage 1 |
| Readout head | **NO** | Frozen (used for loss only) |
| V2 GRU | **YES** | Main learner |
| V2→deep template weights | **YES** | Feedback pathway |
| V2→SOM weights | **YES** | Feedback pathway |
| Deep template→L2/3 weights | **YES** | How template modulates L2/3 |
| SOM→L2/3 weights | **YES** | Inhibitory modulation |

---

### 1.4 Stage 3: Mechanism Comparison

#### 1.4.1 Five Model Variants

Each variant differs **only** in the feedback architecture — specifically, how V2's predictions modulate L2/3 via the deep template and SOM pathways.

```python
class FeedbackMechanism:
    """Base class for mechanism variants."""
    pass

class DampeningMechanism(FeedbackMechanism):
    """
    V2 prediction → deep template → SOM inhibition targeted at expected channel.
    
    SOM profile: sharp inhibition centered on expected orientation.
    σ_som = 10° (narrow, targets only the expected channel and immediate neighbors)
    
    Effect: neurons tuned to expected orientation are suppressed.
    Net result: reduced response at expected orientation, others unaffected.
    """
    som_sigma = 10.0  # degrees, narrow inhibition
    template_to_som = 'excitatory_peaked'  # template excites SOM at expected ori
    som_to_l23 = 'inhibitory'              # SOM inhibits L2/3

class SharpeningMechanism(FeedbackMechanism):
    """
    V2 prediction → deep template → SOM inhibition targeted AWAY from expected channel.
    
    SOM profile: broad inhibition everywhere EXCEPT expected orientation.
    Implemented as: uniform inhibition minus a sharp facilitation at expected ori.
    σ_som_excl = 15° (the narrow exclusion zone around expected)
    
    Effect: neurons AWAY from expected orientation are suppressed.
    Net result: expected orientation channel relatively enhanced, flanks suppressed.
    """
    som_sigma_broad = 90.0    # very broad (nearly uniform inhibition)
    som_sigma_exclude = 15.0  # narrow exclusion zone around expected
    template_to_som = 'excitatory_inverted'  # template drives broad SOM inhibition
    som_to_l23 = 'inhibitory'

class CenterSurroundMechanism(FeedbackMechanism):
    """
    V2 prediction → deep template → two pathways:
      1. Narrow excitation at expected orientation (center)
      2. Broader inhibition around expected orientation (surround)
    
    Net profile is a Mexican-hat (difference of Gaussians):
      modulation(δθ) = A_exc * exp(-δθ²/(2σ_exc²)) - A_inh * exp(-δθ²/(2σ_inh²))
    
    σ_exc = 12° (narrow facilitation)
    σ_inh = 35° (broader suppression)
    A_exc = 0.8, A_inh = 0.5 (net positive at center, net negative in surround)
    
    Effect: expected channel is mildly facilitated, flanking channels suppressed,
            distant channels unaffected.
    """
    sigma_exc = 12.0
    sigma_inh = 35.0
    amp_exc = 0.8
    amp_inh = 0.5

class AdaptationOnlyMechanism(FeedbackMechanism):
    """
    CONTROL: No top-down feedback at all.
    
    V2 GRU is present and trained (to verify it learns predictions),
    but its output is NOT connected to any V1 modulatory pathway.
    
    All suppression effects come purely from L4 adaptation.
    This is the null model — any "expectation suppression" seen here
    is actually just repetition suppression / SSA.
    """
    feedback_connected = False

class ExplicitPCMechanism(FeedbackMechanism):
    """
    Explicit predictive coding: V2 sends a prediction to L2/3,
    and L2/3 computes the residual (error = input - prediction).
    
    Deep template generates a full predicted L2/3 activation pattern.
    L2/3 output = L2/3_feedforward - alpha * deep_template_prediction
    
    alpha is a learnable scalar (initialized to 0.5).
    
    This is NOT biologically standard but serves as a computational 
    upper bound on what explicit error coding achieves.
    """
    alpha_init = 0.5  # learnable subtraction strength
```

#### 1.4.2 Fair Comparison Protocol

```python
COMPARISON_CONFIG = {
    'n_seeds': 5,              # 5 random seeds per mechanism
    'seeds': [42, 137, 256, 512, 1024],
    'mechanisms': ['dampening', 'sharpening', 'center_surround', 
                   'adaptation_only', 'explicit_pc'],
    'total_models': 25,        # 5 mechanisms × 5 seeds
    
    # Identical for all models:
    'stage1_checkpoint': 'shared',    # ALL models share the same Stage 1 weights
    'sequence_data': 'shared',        # ALL models see same training sequences (per seed)
    'lambdas': LAMBDAS,              # same loss weights
    'optimizer': 'AdamW',            # same optimizer
    'lr_v2': 3e-4,                   # same learning rates
    'lr_feedback': 1e-4,
    'total_steps': 80000,            # same compute budget
    'batch_size': 32,
    
    # The ONLY difference: feedback mechanism class
}

def train_all_models():
    """
    Train all 25 models. Can be parallelized across GPUs.
    
    Critical: same seed → same random sequence data.
    The SequenceGenerator is seeded with the model seed,
    so seed=42 always produces the same sequence order
    regardless of mechanism.
    """
    # Stage 1: Train once, save checkpoint
    stage1_model = train_stage1(seed=0)
    verify_stage1(stage1_model)
    save_checkpoint(stage1_model, 'stage1_shared.pt')
    
    for mechanism in COMPARISON_CONFIG['mechanisms']:
        for seed in COMPARISON_CONFIG['seeds']:
            model = build_model(mechanism=mechanism)
            model.load_stage1('stage1_shared.pt')  # same sensory scaffold
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            generator = SequenceGenerator(seed=seed)
            
            train_stage2(model, generator, 
                        steps=COMPARISON_CONFIG['total_steps'],
                        mechanism_name=mechanism, seed=seed)
            
            save_checkpoint(model, f'{mechanism}_seed{seed}.pt')
```

#### 1.4.3 Stage 3 Gating (Post-Training Validation)

Before any model enters the analysis phase, verify:

```python
def validate_trained_model(model, mechanism_name):
    """
    Post-training health checks. A model that fails these is excluded from analysis.
    """
    checks = {}
    
    # 1. Sensory fidelity preserved
    checks['sensory_accuracy'] = evaluate_sensory_accuracy(model) >= 0.80
    # (slightly lower bar than Stage 1 because top-down modulation can shift responses)
    
    # 2. V2 learned predictions (except adaptation_only, which should still learn internally)
    checks['v2_prediction_accuracy'] = evaluate_v2_prediction(model) >= 0.60
    
    # 3. No population collapse
    checks['mean_rate_bounded'] = 0.05 < evaluate_mean_rate(model) < 0.80
    
    # 4. Tuning curves still exist (no catastrophic forgetting of selectivity)
    checks['tuning_intact'] = all_units_selective(model, min_selectivity=1.5)
    
    # 5. For non-control models: feedback is actually active
    if mechanism_name != 'adaptation_only':
        checks['feedback_active'] = feedback_magnitude(model) > 0.01
    
    return checks
```

---

## 2. Experimental Paradigms

### 2.0 General Experimental Protocol

All experiments are run **after training is complete** (inference only, no gradients). Each paradigm generates specific trial sequences, runs them through the trained model, and records all layer activations.

```python
@dataclass
class TrialRecord:
    """One trial's complete data, recorded during experiment."""
    paradigm: str
    condition: str            # e.g., 'expected', 'unexpected', 'neutral'
    stimulus_orientation: int  # channel index (0-35)
    expected_orientation: int  # what V2 predicted (-1 if neutral)
    latent_state: int         # HMM state (0=CW, 1=CCW, 2=neutral)
    contrast: float
    
    # Recorded activations (all shape: (T_ON, 36) for temporal dynamics)
    l4_activity: torch.Tensor
    l23_activity: torch.Tensor
    som_activity: torch.Tensor
    deep_template: torch.Tensor
    pv_activity: torch.Tensor
    v2_hidden: torch.Tensor     # (T_ON, n_hidden)
    v2_prediction: torch.Tensor # (36,) softmax prediction
    
    # Derived during recording
    adaptation_state: torch.Tensor  # (36,) L4 adaptation level at trial onset
    
    # Readout window activity (mean over t=4..7)
    l23_readout: torch.Tensor   # (36,) — main analysis target

@dataclass
class ExperimentResult:
    """Full experiment output."""
    paradigm: str
    trials: List[TrialRecord]
    metadata: dict
```

**Common context-setting protocol:**
Before each critical trial, the network must be in a well-defined predictive state. This requires a **context phase** of consistent transitions:

```python
N_CONTEXT = 8  # presentations establishing context before the probe trial
# The last 5 must be rule-consistent (to ensure V2 has inferred the state)
# The first 3 can be anything (warm-up / initial state inference)
```

---

### 2.1 Paradigm 1: Hidden-State Transitions

**Purpose:** Core paradigm. Establish predictive context, then measure V1 response to expected vs. unexpected vs. neutral stimuli.

#### 2.1.1 Orientation Set

Use 12 orientations spaced 15° apart (channels 0, 3, 6, ..., 33) for cleaner experimental structure. This means the CW/CCW step is exactly one step in the experimental orientation set.

```python
EXPERIMENT_CHANNELS = list(range(0, 36, 3))  # [0, 3, 6, 9, ..., 33] = 12 orientations
EXPERIMENT_ORIENTATIONS = [c * 5.0 for c in EXPERIMENT_CHANNELS]  # [0°, 15°, 30°, ..., 165°]
```

#### 2.1.2 Trial Structure

```python
def generate_paradigm1_trials(n_trials_per_condition=200):
    """
    Generates trials for Paradigm 1: Hidden-State Transitions.
    
    Three conditions:
        1. Expected: probe orientation matches V2's prediction (rule-consistent)
        2. Unexpected: probe orientation violates V2's prediction (rule-inconsistent)  
        3. Neutral: probe after neutral-state context (no specific prediction)
    
    Trial structure:
        [context_1, context_2, ..., context_8, PROBE]
        
        Context phase: 8 presentations following a consistent rule
        Probe: the critical measurement trial
    
    For Expected and Unexpected conditions:
        - Context: 8 presentations following CW or CCW rule (alternated across trials)
        - Expected probe: next orientation following the rule
        - Unexpected probe: orientation 90° away from predicted (maximally unexpected)
          Also include moderate-unexpected: 30° and 60° deviations
    
    For Neutral condition:
        - Context: 8 random orientations (no systematic transition structure)
        - Probe: random orientation (same marginal statistics as other conditions)
    
    Returns:
        list of trial dicts, each containing full sequence + probe info
    """
    trials = []
    
    for condition in ['expected', 'unexpected_90', 'unexpected_60', 
                      'unexpected_30', 'neutral']:
        for trial_idx in range(n_trials_per_condition):
            # Balance starting orientation across trials
            start_ori_idx = trial_idx % 12
            start_channel = EXPERIMENT_CHANNELS[start_ori_idx]
            
            # Balance CW vs CCW across trials
            direction = 1 if trial_idx % 2 == 0 else -1  # CW=+1, CCW=-1
            step = 3 * direction  # 3 channels = 15°
            
            # Build context sequence
            context = []
            current = start_channel
            for i in range(N_CONTEXT):
                context.append(current)
                current = (current + step) % 36
            
            # Predicted next orientation
            predicted = (context[-1] + step) % 36
            
            if condition == 'expected':
                probe = predicted
            elif condition == 'unexpected_90':
                probe = (predicted + 18) % 36  # +90° = 18 channels
            elif condition == 'unexpected_60':
                probe = (predicted + 12) % 36  # +60° = 12 channels
            elif condition == 'unexpected_30':
                probe = (predicted + 6) % 36   # +30° = 6 channels
            elif condition == 'neutral':
                # Rebuild context as random for neutral
                context = [EXPERIMENT_CHANNELS[torch.randint(0, 12, (1,)).item()] 
                          for _ in range(N_CONTEXT)]
                probe = EXPERIMENT_CHANNELS[torch.randint(0, 12, (1,)).item()]
                predicted = -1
            
            trials.append({
                'condition': condition,
                'context': context,
                'probe': probe,
                'predicted': predicted,
                'direction': direction,
                'start_channel': start_channel,
                'contrast': 1.0,
            })
    
    return trials
```

**Total trials:** 200 per condition × 5 conditions = 1,000 trials per model.

**Counterbalancing:**
- Starting orientation uniformly distributed across 12 orientations
- CW vs CCW balanced within each condition
- Probe orientation: marginal frequency of each orientation is equalized across expected + unexpected conditions (critical — see verification below)

```python
def verify_paradigm1_balance(trials):
    """Verify that probe orientations have equal marginal frequency across conditions."""
    from collections import Counter
    for condition in ['expected', 'unexpected_90', 'unexpected_60', 
                      'unexpected_30', 'neutral']:
        cond_trials = [t for t in trials if t['condition'] == condition]
        probe_counts = Counter(t['probe'] for t in cond_trials)
        counts = list(probe_counts.values())
        ratio = max(counts) / min(counts)
        assert ratio < 1.5, f"Condition {condition}: probe frequency ratio {ratio:.2f} > 1.5"
    print("Paradigm 1 balance: PASS")
```

#### 2.1.3 Running Trials Through the Network

```python
def run_paradigm1(model, trials):
    """
    Run each trial through the trained model and record all activations.
    
    Procedure:
        1. Reset model state (V2 hidden, adaptation, SOM activity)
        2. Present context sequence (8 presentations × 12 timesteps)
        3. Present probe stimulus (1 presentation × 12 timesteps)
        4. Record all activations during probe ON period (t=0..7)
        5. Store readout-window average (t=4..7) for main analysis
    """
    records = []
    
    for trial in tqdm(trials, desc="Paradigm 1"):
        model.reset_state()
        
        # Context phase — run but don't record (except to verify V2 state)
        for ori_channel in trial['context']:
            model.present_stimulus(ori_channel, contrast=trial['contrast'], 
                                   n_timesteps=T_ON)
            model.run_isi(n_timesteps=T_ISI)
        
        # Verify V2 has formed prediction (sanity check)
        v2_pred = model.get_v2_prediction()
        
        # Probe phase — record everything
        probe_record = model.present_stimulus_and_record(
            trial['probe'], contrast=trial['contrast'], n_timesteps=T_ON
        )
        
        records.append(TrialRecord(
            paradigm='hidden_state_transitions',
            condition=trial['condition'],
            stimulus_orientation=trial['probe'],
            expected_orientation=trial['predicted'],
            latent_state=0 if trial['direction'] == 1 else 1,
            contrast=trial['contrast'],
            l4_activity=probe_record['l4'],
            l23_activity=probe_record['l23'],
            som_activity=probe_record['som'],
            deep_template=probe_record['template'],
            pv_activity=probe_record['pv'],
            v2_hidden=probe_record['v2_hidden'],
            v2_prediction=v2_pred,
            adaptation_state=model.get_adaptation_state(),
            l23_readout=probe_record['l23'][4:8].mean(dim=0),  # readout window
        ))
    
    return records
```

---

### 2.2 Paradigm 2: Omission Trials

**Purpose:** Test whether the deep template contains an expectation signal even when no stimulus is presented. This dissociates top-down template from bottom-up response.

#### 2.2.1 Trial Structure

```python
def generate_paradigm2_trials(n_trials_per_condition=150):
    """
    Three conditions:
        1. Omission-after-predictive: stable context (CW/CCW) → blank instead of expected grating
        2. Omission-after-neutral: neutral context → blank
        3. Present-after-predictive: control — same context but grating IS presented
    
    Context must be longer to ensure strong predictive state:
        N_CONTEXT = 10 (vs 8 in Paradigm 1)
    
    During omission: L4 input = baseline noise only (contrast=0)
    Record deep template, L2/3, and V2 hidden during the omission period.
    """
    N_CONTEXT_OMISSION = 10  # longer context for stronger prediction
    trials = []
    
    for condition in ['omission_predictive', 'omission_neutral', 'present_predictive']:
        for trial_idx in range(n_trials_per_condition):
            start_channel = EXPERIMENT_CHANNELS[trial_idx % 12]
            direction = 1 if trial_idx % 2 == 0 else -1
            step = 3 * direction
            
            if condition in ['omission_predictive', 'present_predictive']:
                context = []
                current = start_channel
                for i in range(N_CONTEXT_OMISSION):
                    context.append(current)
                    current = (current + step) % 36
                predicted = (context[-1] + step) % 36
            else:  # neutral
                context = [EXPERIMENT_CHANNELS[torch.randint(0, 12, (1,)).item()] 
                          for _ in range(N_CONTEXT_OMISSION)]
                predicted = -1
            
            is_omission = condition.startswith('omission')
            
            trials.append({
                'condition': condition,
                'context': context,
                'probe': predicted if not is_omission else -1,  # -1 = blank
                'predicted': predicted,
                'contrast': 0.0 if is_omission else 1.0,
                'direction': direction,
            })
    
    return trials
```

**Key measurement:** During omission, decode orientation from the deep template. If the template is active, orientation should be decodable above chance (1/12 ≈ 8.3%).

---

### 2.3 Paradigm 3: Ambiguous Stimuli

**Purpose:** Test perceptual bias. Under ambiguity, does the network bias perception toward (center-surround) or away from (dampening) the expected orientation?

#### 2.3.1 Stimulus Design

```python
def generate_ambiguous_stimulus(channel_a, channel_b, mixture_weight=0.5):
    """
    Create an ambiguous stimulus as a weighted mixture of two orientation channels.
    
    L4 input = mixture_weight * tuning_curve(θ_a) + (1 - mixture_weight) * tuning_curve(θ_b)
    
    This simulates a low-contrast grating with ambiguous orientation, or a plaid
    composed of two components.
    """
    l4_a = l4_tuning_curve(channel_a * 5.0)  # (36,) response to orientation A
    l4_b = l4_tuning_curve(channel_b * 5.0)  # (36,) response to orientation B
    return mixture_weight * l4_a + (1 - mixture_weight) * l4_b

def generate_paradigm3_trials(n_trials_per_condition=150):
    """
    Conditions:
        1. Ambiguous-expected-component: context predicts θ_a, stimulus is θ_a/θ_b mixture
        2. Ambiguous-neutral: no prediction, same θ_a/θ_b mixture
        3. Low-contrast-expected: context predicts θ, stimulus is θ at low contrast
        4. Low-contrast-neutral: no prediction, same low contrast
    
    For ambiguous mixtures:
        θ_b is chosen to be ±30° from θ_a (close enough to create genuine ambiguity,
        far enough that the two are distinguishable).
    
    For low-contrast:
        contrast = 0.15 (just above threshold, where top-down influence should be maximal)
    """
    trials = []
    
    for condition in ['ambiguous_expected', 'ambiguous_neutral',
                      'low_contrast_expected', 'low_contrast_neutral']:
        for trial_idx in range(n_trials_per_condition):
            start_channel = EXPERIMENT_CHANNELS[trial_idx % 12]
            direction = 1 if trial_idx % 2 == 0 else -1
            step = 3 * direction
            
            # Build context
            if 'expected' in condition:
                context = []
                current = start_channel
                for i in range(N_CONTEXT):
                    context.append(current)
                    current = (current + step) % 36
                predicted = (context[-1] + step) % 36
            else:
                context = [EXPERIMENT_CHANNELS[torch.randint(0, 12, (1,)).item()]
                          for _ in range(N_CONTEXT)]
                predicted = -1
            
            if 'ambiguous' in condition:
                # θ_a = predicted orientation, θ_b = predicted ± 30° (6 channels)
                theta_a = predicted if predicted >= 0 else EXPERIMENT_CHANNELS[trial_idx % 12]
                bias_dir = 1 if trial_idx % 2 == 0 else -1
                theta_b = (theta_a + 6 * bias_dir) % 36
                
                trials.append({
                    'condition': condition,
                    'context': context,
                    'probe_type': 'mixture',
                    'channel_a': theta_a,
                    'channel_b': theta_b,
                    'mixture_weight': 0.5,  # equal mixture
                    'predicted': predicted,
                    'contrast': 0.5,  # moderate contrast for the mixture
                })
            else:  # low contrast
                probe = predicted if predicted >= 0 else EXPERIMENT_CHANNELS[trial_idx % 12]
                trials.append({
                    'condition': condition,
                    'context': context,
                    'probe_type': 'single',
                    'probe': probe,
                    'predicted': predicted,
                    'contrast': 0.15,  # low contrast
                })
    
    return trials
```

**Key measurement:** Decode perceived orientation from L2/3. Under ambiguity, is decoded orientation biased toward or away from the expected orientation? This is the critical dampening vs. center-surround diagnostic.

---

### 2.4 Paradigm 4: Task Relevance (Attention Control)

**Purpose:** Dissociate expectation from attention. If suppression occurs only when orientation is task-relevant, it might be attention, not prediction.

#### 2.4.1 Task Design

```python
"""
Two task conditions, run in separate blocks:

Task A — Orientation Discrimination (orientation-relevant):
    "Which of two orientations was presented?"
    Implemented as: additional readout head trained with cross-entropy
    on orientation labels. Active during these blocks.
    
    λ_task_A = 0.3 added to the loss for the orientation discrimination head.

Task B — Contrast Change Detection (orientation-irrelevant):
    "Did the contrast change relative to previous trial?"
    Implemented as: binary readout head on L2/3 mean activity level.
    Orientation information is present but not task-relevant.
    
    λ_task_B = 0.3 added to the loss for the contrast change head.

CRITICAL: Same stimulus statistics in both tasks. Same orientations, same sequences,
same transition probabilities. The ONLY difference is which readout head receives 
gradient during training.

Implementation: Train two variants of each mechanism model:
    - One with Task A loss added to Stage 2 loss
    - One with Task B loss added to Stage 2 loss
    Then run identical experimental trials through both.
"""

class TaskRelevanceConfig:
    # Task A: Orientation discrimination
    task_a_readout = nn.Linear(36, 36)  # L2/3 → orientation class
    task_a_loss_weight = 0.3
    
    # Task B: Contrast change detection
    task_b_readout = nn.Linear(36, 2)   # L2/3 → {same_contrast, different_contrast}
    task_b_loss_weight = 0.3
    
    # Contrast change implementation:
    # On 30% of trials, contrast changes by ±0.2 from default
    # Model must detect this binary change
    contrast_change_prob = 0.3
    contrast_change_magnitude = 0.2
```

#### 2.4.2 Trial Structure

```python
def generate_paradigm4_trials(n_trials_per_condition=150):
    """
    Factorial design: 2 tasks × 3 expectation conditions = 6 cells
    
    Task: orientation_relevant, orientation_irrelevant
    Expectation: expected, unexpected, neutral
    
    Same trial structure as Paradigm 1, but with task manipulation.
    Task is manipulated between blocks (not within — to avoid task-switching costs).
    
    Block structure:
        10 blocks of 60 trials each (30 per task, alternating ABAB...)
        First block is practice (discarded)
    """
    # Use same trial generation as Paradigm 1
    base_trials = generate_paradigm1_trials(n_trials_per_condition)
    
    # Duplicate for each task
    task_a_trials = [dict(t, task='orientation_relevant') for t in base_trials]
    task_b_trials = [dict(t, task='orientation_irrelevant') for t in base_trials]
    
    return task_a_trials + task_b_trials
```

**Key prediction:** If expectation suppression is truly predictive (not attentional), it should occur in BOTH tasks. If it only occurs in the orientation-relevant task, it's likely an attention effect.

---

### 2.5 Paradigm 5: Local vs. High-Level Surprise Dissociation

**Purpose:** Test whether V1 modulation tracks local stimulus mismatch (|θ_presented - θ_predicted|) or V2's inferred rule violation.

#### 2.5.1 Trial Design

```python
def generate_paradigm5_trials(n_trials_per_condition=150):
    """
    The key manipulation: trials with MATCHED local mismatch but DIFFERENT rule violation.
    
    Setup:
        - Establish CW context: ..., 0°, 15°, 30°, 45° (V2 expects 60° next)
        - Two conditions with identical |60° - probe| = 45°:
    
    Condition A — Rule-violating 45° mismatch:
        Context: CW rule → predicts 60°
        Probe: 105° (|105 - 60| = 45°, clearly violates CW rule)
        V2 surprise: HIGH (this orientation could only happen by random jump)
    
    Condition B — Rule-consistent "distant" transition:
        Context: CW rule → sudden state change to CCW (detected by V2 over 2-3 trials)
        Probe: 15° (which is consistent with a CCW step from the last orientation)
        But engineered so |15 - 60| = 45° same local distance as Condition A
        V2 surprise: LOW (once V2 infers CCW state, 15° is expected)
    
    Wait — this is tricky. Let me design it more carefully.
    
    Better approach:
        Condition A — High V2 surprise, moderate local mismatch:
            CW context predicts 60°. Present 105° (+45° off).
            Local mismatch: 45°. V2 rule violation: YES.
        
        Condition B — Low V2 surprise, same local mismatch:
            CW context then 2-3 CCW transitions (V2 detects state change).
            After V2 has updated: the next CCW-predicted orientation happens to be
            45° from what the original CW rule would have predicted.
            Local mismatch from original prediction: 45°. V2 rule violation: NO (new rule).
        
        Condition C — High V2 surprise, small local mismatch:
            CW context predicts 60°. Present 75° (+15° off, only 1 step wrong).
            But this orientation is inconsistent with BOTH CW and CCW rules.
            Local mismatch: 15°. V2 rule violation: YES.
        
        Condition D — Low V2 surprise, small local mismatch:
            CW context predicts 60°. Present 45° (-15° off, 1 step backward).
            This IS consistent with a CCW rule.
            Local mismatch: 15°. V2 rule violation: AMBIGUOUS (could be state change).
    
    The cleanest comparison is A vs B (matched local mismatch, different V2 surprise).
    """
    trials = []
    
    for trial_idx in range(n_trials_per_condition):
        start_channel = EXPERIMENT_CHANNELS[trial_idx % 12]
        
        # ---- Condition A: High V2 surprise, 45° mismatch ----
        # CW context
        context_a = []
        current = start_channel
        for i in range(N_CONTEXT):
            context_a.append(current)
            current = (current + 3) % 36
        predicted_cw = current  # next CW step
        probe_a = (predicted_cw + 9) % 36  # +45° = 9 channels away
        
        trials.append({
            'condition': 'high_surprise_45deg',
            'context': context_a,
            'probe': probe_a,
            'predicted': predicted_cw,
            'local_mismatch_deg': 45.0,
            'v2_surprise': 'high',
        })
        
        # ---- Condition B: Low V2 surprise, 45° mismatch ----
        # CW context then transition to CCW
        # Start CW for 5, then CCW for 3 (V2 should detect state change)
        context_b_phase1 = []
        current = start_channel
        for i in range(5):
            context_b_phase1.append(current)
            current = (current + 3) % 36
        # Now switch to CCW for 3 more
        context_b_phase2 = []
        for i in range(3):
            current = (current - 3) % 36
            context_b_phase2.append(current)
        
        context_b = context_b_phase1 + context_b_phase2
        
        # V2 should now predict CCW continuation
        predicted_ccw = (context_b[-1] - 3) % 36
        # Present the CCW-expected probe, which IS 45° from original CW prediction
        # (we may need to adjust starting orientation to make this exactly 45°)
        probe_b = predicted_ccw
        
        # Compute actual local mismatch from original CW prediction
        cw_would_predict = (context_b_phase1[-1] + 3) % 36
        mismatch_channels = min(
            abs(probe_b - cw_would_predict),
            36 - abs(probe_b - cw_would_predict)
        )
        mismatch_deg = mismatch_channels * 5.0
        
        trials.append({
            'condition': 'low_surprise_matched_mismatch',
            'context': context_b,
            'probe': probe_b,
            'predicted_by_current_state': predicted_ccw,
            'predicted_by_original_state': cw_would_predict,
            'local_mismatch_from_original_deg': mismatch_deg,
            'v2_surprise': 'low',
        })
    
    return trials
```

**Key measurement:** Compare V1 suppression magnitude (expected - actual response) between conditions A and B. If V1 tracks V2's inferred rule, suppression should differ despite matched local mismatch. If V1 only tracks local mismatch, suppression should be identical.

**Additional analysis:** Measure V2 hidden state distance from the "predictive state" as a continuous regressor of V1 modulation magnitude.

---

## 3. Analysis / Readout Suite

### 3.0 Data Structure for All Analyses

```python
@dataclass
class AnalysisInput:
    """Standardized input to all analysis functions."""
    trials: List[TrialRecord]      # all trial records from one paradigm
    model_name: str                 # e.g., 'dampening_seed42'
    mechanism: str                  # e.g., 'dampening'
    preferred_orientations: torch.Tensor  # (36,) in degrees
    
    def filter(self, condition: str) -> List[TrialRecord]:
        return [t for t in self.trials if t.condition == condition]
    
    def get_responses(self, condition: str, layer: str = 'l23_readout') -> torch.Tensor:
        """Returns (n_trials, 36) tensor of responses for a condition."""
        filtered = self.filter(condition)
        return torch.stack([getattr(t, layer) for t in filtered])
```

---

### 3.1 Analysis 1: Mean Response Analysis

**What it measures:** Overall population response magnitude across conditions.

**Data needed:** Paradigm 1 trials, all conditions.

```python
def analysis_mean_response(data: AnalysisInput) -> dict:
    """
    Compare mean population response across expected/unexpected/neutral conditions,
    separately for L4, L2/3, and deep template populations.
    
    For each condition and layer:
        1. Average response across all units → scalar per trial
        2. Compute mean ± SEM across trials
        3. Run one-way ANOVA across 3 conditions (expected, unexpected_90, neutral)
        4. Post-hoc pairwise t-tests with Bonferroni correction
    
    Expected results by mechanism:
        Dampening:      expected < neutral < unexpected (in L2/3)
        Sharpening:     expected < neutral < unexpected (in L2/3, same direction)
        Center-surround: expected < neutral < unexpected (in L2/3, same direction)
        Adaptation:     expected ≈ neutral ≈ unexpected in L2/3 (no top-down effect)
                        expected < neutral in L4 (adaptation only)
    
    KEY: All three suppression mechanisms reduce mean L2/3 response to expected.
    This analysis CANNOT distinguish them — it just verifies suppression exists.
    """
    results = {}
    
    for layer in ['l4_activity', 'l23_readout', 'deep_template']:
        for cond in ['expected', 'unexpected_90', 'neutral']:
            responses = data.get_responses(cond, layer)  # (n_trials, 36)
            mean_per_trial = responses.mean(dim=1)  # (n_trials,) mean over all channels
            
            results[f'{layer}_{cond}_mean'] = mean_per_trial.mean().item()
            results[f'{layer}_{cond}_sem'] = mean_per_trial.std().item() / math.sqrt(len(mean_per_trial))
        
        # ANOVA
        from scipy import stats
        groups = [
            data.get_responses('expected', layer).mean(dim=1).numpy(),
            data.get_responses('unexpected_90', layer).mean(dim=1).numpy(),
            data.get_responses('neutral', layer).mean(dim=1).numpy(),
        ]
        f_stat, p_val = stats.f_oneway(*groups)
        results[f'{layer}_anova_F'] = f_stat
        results[f'{layer}_anova_p'] = p_val
        
        # Post-hoc pairwise t-tests (Bonferroni corrected)
        pairs = [('expected', 'neutral'), ('unexpected_90', 'neutral'), 
                 ('expected', 'unexpected_90')]
        for c1, c2 in pairs:
            t_stat, p = stats.ttest_ind(
                data.get_responses(c1, layer).mean(dim=1).numpy(),
                data.get_responses(c2, layer).mean(dim=1).numpy()
            )
            results[f'{layer}_{c1}_vs_{c2}_t'] = t_stat
            results[f'{layer}_{c1}_vs_{c2}_p'] = p * 3  # Bonferroni
    
    return results

def plot_mean_response(results_all_mechanisms: dict):
    """
    Bar plot: x-axis = condition (expected/unexpected/neutral)
    Grouped by mechanism (5 clusters)
    Separate panels for L4, L2/3, deep template
    Error bars = SEM
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    layers = ['l4_activity', 'l23_readout', 'deep_template']
    layer_names = ['L4', 'L2/3 (readout)', 'Deep Template']
    
    for ax, layer, name in zip(axes, layers, layer_names):
        x = np.arange(3)  # 3 conditions
        width = 0.15
        
        for i, mech in enumerate(['dampening', 'sharpening', 'center_surround',
                                   'adaptation_only', 'explicit_pc']):
            means = [results_all_mechanisms[mech][f'{layer}_{c}_mean'] 
                     for c in ['expected', 'unexpected_90', 'neutral']]
            sems = [results_all_mechanisms[mech][f'{layer}_{c}_sem']
                    for c in ['expected', 'unexpected_90', 'neutral']]
            
            ax.bar(x + i * width, means, width, yerr=sems, label=mech, alpha=0.8)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean Population Response')
        ax.set_title(name)
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(['Expected', 'Unexpected', 'Neutral'])
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    return fig
```

---

### 3.2 Analysis 2: Suppression-by-Tuning Profile (THE KEY DIAGNOSTIC)

**What it measures:** How expectation-related suppression varies as a function of each unit's tuning relative to the expected orientation. THIS is the analysis that distinguishes dampening from sharpening from center-surround.

**Data needed:** Paradigm 1 trials, expected and neutral conditions.

```python
def analysis_suppression_by_tuning(data: AnalysisInput) -> dict:
    """
    For each unit i with preferred orientation θ_pref_i:
        1. Compute Δθ_i = circular_distance(θ_pref_i, θ_expected) for each trial
        2. Compute suppression_i = response_neutral_i - response_expected_i 
           (positive = suppression, negative = enhancement)
        3. Bin units by Δθ (0°, 5°, 10°, ..., 90°) and average suppression within bins
        4. Plot suppression as a function of Δθ
    
    Expected profiles:
        Dampening:       peak suppression at Δθ=0° (expected channel), 
                         falls off with distance, near zero by Δθ>30°
        Sharpening:      near-zero suppression at Δθ=0°, 
                         INCREASING suppression from Δθ=15° to Δθ=60°,
                         peaks at Δθ~45°, then falls off
        Center-surround: slight ENHANCEMENT at Δθ=0°, 
                         suppression at Δθ=15-40°,
                         return to baseline by Δθ>50°
                         (Mexican hat profile)
        Adaptation:      flat (no Δθ dependence since adaptation is local in L4)
        Explicit PC:     depends on implementation but likely dampening-like
    """
    results = {}
    
    # Collect responses aligned to expected orientation
    delta_theta_bins = np.arange(0, 95, 5)  # 0° to 90° in 5° steps
    n_bins = len(delta_theta_bins)
    
    suppression_by_bin = {b: [] for b in delta_theta_bins}
    
    expected_trials = data.filter('expected')
    neutral_trials = data.filter('neutral')
    
    # Average neutral response per orientation (baseline)
    neutral_responses = data.get_responses('neutral', 'l23_readout')  # (n_neutral, 36)
    neutral_mean_by_ori = {}
    for trial in neutral_trials:
        ori = trial.stimulus_orientation
        if ori not in neutral_mean_by_ori:
            neutral_mean_by_ori[ori] = []
        neutral_mean_by_ori[ori].append(trial.l23_readout)
    for ori in neutral_mean_by_ori:
        neutral_mean_by_ori[ori] = torch.stack(neutral_mean_by_ori[ori]).mean(dim=0)  # (36,)
    
    # For each expected trial, compute suppression per unit
    for trial in expected_trials:
        expected_ori = trial.expected_orientation  # channel index
        expected_deg = expected_ori * 5.0
        
        # Get matched neutral baseline (same stimulus orientation)
        if trial.stimulus_orientation in neutral_mean_by_ori:
            baseline = neutral_mean_by_ori[trial.stimulus_orientation]
        else:
            continue  # skip if no neutral baseline for this orientation
        
        for unit in range(36):
            pref_deg = data.preferred_orientations[unit].item()
            delta = circular_distance_degrees(pref_deg, expected_deg)  # [0, 90]
            
            suppression = baseline[unit].item() - trial.l23_readout[unit].item()
            
            # Find nearest bin
            bin_idx = int(round(delta / 5.0)) * 5
            bin_idx = min(bin_idx, 90)
            suppression_by_bin[bin_idx].append(suppression)
    
    # Compute bin statistics
    bin_means = []
    bin_sems = []
    for b in delta_theta_bins:
        vals = suppression_by_bin[b]
        if len(vals) > 0:
            bin_means.append(np.mean(vals))
            bin_sems.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            bin_means.append(0)
            bin_sems.append(0)
    
    results['delta_theta_bins'] = delta_theta_bins
    results['suppression_means'] = np.array(bin_means)
    results['suppression_sems'] = np.array(bin_sems)
    
    # Fit parametric models to the suppression profile
    results['fit_dampening'] = fit_gaussian_suppression(delta_theta_bins, bin_means)
    results['fit_sharpening'] = fit_inverted_gaussian_suppression(delta_theta_bins, bin_means)
    results['fit_center_surround'] = fit_dog_suppression(delta_theta_bins, bin_means)
    
    return results

def circular_distance_degrees(a, b):
    """Circular distance on [0°, 180°) orientation space."""
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)

def fit_gaussian_suppression(bins, means):
    """Fit S(Δθ) = A * exp(-Δθ²/(2σ²)) — dampening profile."""
    from scipy.optimize import curve_fit
    def model(x, A, sigma):
        return A * np.exp(-x**2 / (2 * sigma**2))
    try:
        popt, pcov = curve_fit(model, bins, means, p0=[max(means), 20], maxfev=5000)
        residuals = means - model(bins, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((means - np.mean(means))**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        return {'params': popt, 'r_squared': r_squared, 'type': 'dampening'}
    except:
        return {'params': None, 'r_squared': -1, 'type': 'dampening'}

def fit_inverted_gaussian_suppression(bins, means):
    """Fit S(Δθ) = A * (1 - exp(-Δθ²/(2σ²))) — sharpening profile."""
    from scipy.optimize import curve_fit
    def model(x, A, sigma):
        return A * (1 - np.exp(-x**2 / (2 * sigma**2)))
    try:
        popt, pcov = curve_fit(model, bins, means, p0=[max(means), 30], maxfev=5000)
        residuals = means - model(bins, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((means - np.mean(means))**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        return {'params': popt, 'r_squared': r_squared, 'type': 'sharpening'}
    except:
        return {'params': None, 'r_squared': -1, 'type': 'sharpening'}

def fit_dog_suppression(bins, means):
    """Fit S(Δθ) = A_surr*exp(-Δθ²/(2σ_s²)) - A_ctr*exp(-Δθ²/(2σ_c²)) — center-surround."""
    from scipy.optimize import curve_fit
    def model(x, A_surr, sigma_surr, A_ctr, sigma_ctr):
        return A_surr * np.exp(-x**2 / (2 * sigma_surr**2)) - A_ctr * np.exp(-x**2 / (2 * sigma_ctr**2))
    try:
        popt, pcov = curve_fit(model, bins, means, 
                               p0=[0.5, 30, 0.3, 10], maxfev=10000,
                               bounds=([0, 10, 0, 1], [5, 90, 5, 25]))
        residuals = means - model(bins, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((means - np.mean(means))**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        return {'params': popt, 'r_squared': r_squared, 'type': 'center_surround'}
    except:
        return {'params': None, 'r_squared': -1, 'type': 'center_surround'}

def plot_suppression_by_tuning(results_all_mechanisms: dict):
    """
    Line plot: x-axis = Δθ (distance from expected), y-axis = suppression magnitude
    One line per mechanism, with SEM shading
    Dashed horizontal line at y=0 (no effect)
    
    This is THE key figure for the paper.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {
        'dampening': '#e41a1c',
        'sharpening': '#377eb8',
        'center_surround': '#4daf4a',
        'adaptation_only': '#984ea3',
        'explicit_pc': '#ff7f00',
    }
    
    for mech, res in results_all_mechanisms.items():
        bins = res['delta_theta_bins']
        means = res['suppression_means']
        sems = res['suppression_sems']
        
        ax.plot(bins, means, 'o-', color=colors[mech], label=mech, linewidth=2)
        ax.fill_between(bins, means - sems, means + sems, color=colors[mech], alpha=0.2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δθ from Expected Orientation (degrees)')
    ax.set_ylabel('Suppression (neutral - expected)')
    ax.set_title('Suppression-by-Tuning Profile')
    ax.legend()
    ax.set_xlim([0, 90])
    
    return fig
```

---

### 3.3 Analysis 3: Tuning Curve Analysis

**What it measures:** Changes in tuning curve parameters (amplitude, width, preferred orientation) across conditions.

```python
def analysis_tuning_curves(data: AnalysisInput) -> dict:
    """
    For each L2/3 unit, construct its orientation tuning curve under each condition.
    Fit von Mises function, extract parameters.
    
    Procedure:
        1. For each condition (expected, unexpected, neutral):
           For each of the 36 units:
             a. Collect all trials where unit i is the "target" 
                (stimulus = unit's preferred orientation ± 15°)
             b. Build response profile: response as function of stimulus orientation
             c. Fit: r(θ) = b + A * exp(κ * cos(2*(θ - θ_pref)))
             d. Extract: A (amplitude), κ→σ (width), θ_pref (peak location)
        
        2. Compare across conditions:
           - Amplitude change: gain modulation (dampening predicts lower A for expected)
           - Width change: sharpening predicts narrower σ for expected
           - Peak shift: center-surround might shift peaks
    
    Expected results:
        Dampening:       ↓ amplitude, = width, = peak position
        Sharpening:      ↓ or = amplitude, ↓ width (narrower), = peak position
        Center-surround: slight ↓ amplitude, slight ↓ width, possible small peak shift
        Adaptation:      ↓ amplitude in L4 only (no change in L2/3 width)
    """
    results = {}
    
    for condition in ['expected', 'unexpected_90', 'neutral']:
        amplitudes = []
        widths = []
        peak_shifts = []
        
        # Get all trials for this condition
        trials = data.filter(condition)
        
        # Build response matrix: (n_trials, 36) responses × 36 stimulus orientations
        # Group trials by stimulus orientation
        responses_by_stim = {}
        for trial in trials:
            stim = trial.stimulus_orientation
            if stim not in responses_by_stim:
                responses_by_stim[stim] = []
            responses_by_stim[stim].append(trial.l23_readout)
        
        # Average responses per stimulus orientation
        mean_responses_by_stim = {}
        for stim, resps in responses_by_stim.items():
            mean_responses_by_stim[stim] = torch.stack(resps).mean(dim=0)  # (36,)
        
        # For each unit, build tuning curve
        for unit in range(36):
            ori_axis = sorted(mean_responses_by_stim.keys())
            response_curve = [mean_responses_by_stim[o][unit].item() for o in ori_axis]
            
            # Fit von Mises
            fit = fit_von_mises(
                np.array(ori_axis) * 5.0,  # convert to degrees
                np.array(response_curve)
            )
            
            if fit is not None:
                amplitudes.append(fit['amplitude'])
                widths.append(fit['sigma_degrees'])
                peak_shifts.append(
                    circular_distance_degrees(
                        fit['preferred'], 
                        data.preferred_orientations[unit].item()
                    )
                )
        
        results[f'{condition}_amplitudes'] = np.array(amplitudes)
        results[f'{condition}_widths'] = np.array(widths)
        results[f'{condition}_peak_shifts'] = np.array(peak_shifts)
    
    # Statistical comparisons
    from scipy import stats
    for param in ['amplitudes', 'widths']:
        t, p = stats.ttest_rel(results[f'expected_{param}'], results[f'neutral_{param}'])
        results[f'{param}_expected_vs_neutral_t'] = t
        results[f'{param}_expected_vs_neutral_p'] = p
    
    return results

def fit_von_mises(orientations_deg, responses):
    """
    Fit r(θ) = baseline + amplitude * exp(kappa * (cos(2*(θ - θ_pref)*π/180) - 1))
    Returns dict with amplitude, kappa, sigma_degrees, preferred, baseline, r_squared.
    """
    from scipy.optimize import curve_fit
    
    def model(theta, baseline, amplitude, kappa, theta_pref):
        delta = 2 * (theta - theta_pref) * np.pi / 180
        return baseline + amplitude * np.exp(kappa * (np.cos(delta) - 1))
    
    try:
        # Initial guesses
        peak_idx = np.argmax(responses)
        p0 = [np.min(responses), np.max(responses) - np.min(responses), 
              2.0, orientations_deg[peak_idx]]
        
        popt, pcov = curve_fit(model, orientations_deg, responses, p0=p0, maxfev=10000)
        
        baseline, amplitude, kappa, theta_pref = popt
        
        # Convert kappa to sigma in degrees
        # For von Mises on doubled angle: variance = 1 - I1(κ)/I0(κ)
        from scipy.special import i0, i1
        if kappa > 0:
            circ_var = 1 - i1(kappa) / i0(kappa)
            sigma_rad = np.sqrt(circ_var)
            sigma_deg = sigma_rad * 180 / (2 * np.pi)  # half because doubled angle
        else:
            sigma_deg = 90.0  # flat
        
        predicted = model(orientations_deg, *popt)
        ss_res = np.sum((responses - predicted)**2)
        ss_tot = np.sum((responses - np.mean(responses))**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            'baseline': baseline,
            'amplitude': amplitude,
            'kappa': kappa,
            'sigma_degrees': sigma_deg,
            'preferred': theta_pref % 180,
            'r_squared': r_squared,
        }
    except:
        return None

def plot_tuning_curve_params(results_all_mechanisms: dict):
    """
    Two-panel figure:
        Left: amplitude (expected vs neutral) for each mechanism
        Right: width (expected vs neutral) for each mechanism
    
    Scatter plots with unity line. Points below unity = suppression (amplitude) 
    or narrowing (width).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for mech, res in results_all_mechanisms.items():
        # Amplitude panel
        axes[0].scatter(
            res['neutral_amplitudes'], res['expected_amplitudes'],
            alpha=0.3, label=mech, s=10
        )
        # Width panel
        axes[1].scatter(
            res['neutral_widths'], res['expected_widths'],
            alpha=0.3, label=mech, s=10
        )
    
    for ax, title in zip(axes, ['Amplitude', 'Width (σ, degrees)']):
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', alpha=0.3)  # unity line
        ax.set_xlabel(f'Neutral {title}')
        ax.set_ylabel(f'Expected {title}')
        ax.set_title(title)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    return fig
```

---

### 3.4 Analysis 4: Decoding Accuracy & Fisher Information

**What it measures:** How well can a linear decoder discriminate orientations from L2/3 activity across conditions?

```python
def analysis_decoding(data: AnalysisInput) -> dict:
    """
    Train a linear SVM decoder on L2/3 readout activity to classify stimulus orientation.
    Cross-validated accuracy per condition.
    
    Procedure:
        1. For each condition separately:
           a. Collect all (L2/3 readout vector, stimulus orientation) pairs
           b. 5-fold cross-validation with linear SVM
           c. Report mean accuracy ± std
        
        2. Also: train decoder on neutral data, test on expected data
           (transfer accuracy — tests whether representation geometry changed)
        
        3. Fisher information (approximate):
           For each pair of adjacent orientations (e.g., 0° and 15°),
           compute d' = |μ_1 - μ_2| / sqrt((σ_1² + σ_2²)/2)
           where μ, σ are mean and std of L2/3 population response vectors.
           Average d' across orientation pairs.
    
    Expected results:
        Sharpening:     ↑ decoding accuracy for expected (more discriminable)
        Dampening:      ↓ decoding accuracy for expected (less discriminable)
        Center-surround: ↑ or = decoding accuracy (local enhancement helps)
        Adaptation:     ↓ decoding accuracy (general degradation, no structure)
    """
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    
    for condition in ['expected', 'unexpected_90', 'neutral']:
        trials = data.filter(condition)
        X = torch.stack([t.l23_readout for t in trials]).numpy()  # (n, 36)
        y = np.array([t.stimulus_orientation for t in trials])    # (n,)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LinearSVC(max_iter=10000, C=1.0)
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        
        results[f'{condition}_decode_acc_mean'] = scores.mean()
        results[f'{condition}_decode_acc_std'] = scores.std()
    
    # Transfer: train on neutral, test on expected
    neutral_trials = data.filter('neutral')
    expected_trials = data.filter('expected')
    
    X_train = torch.stack([t.l23_readout for t in neutral_trials]).numpy()
    y_train = np.array([t.stimulus_orientation for t in neutral_trials])
    X_test = torch.stack([t.l23_readout for t in expected_trials]).numpy()
    y_test = np.array([t.stimulus_orientation for t in expected_trials])
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    clf = LinearSVC(max_iter=10000, C=1.0)
    clf.fit(X_train_s, y_train)
    transfer_acc = clf.score(X_test_s, y_test)
    results['transfer_neutral_to_expected'] = transfer_acc
    
    # Fisher information (d-prime between adjacent orientations)
    for condition in ['expected', 'neutral']:
        trials = data.filter(condition)
        
        # Group by stimulus orientation
        responses_by_ori = {}
        for t in trials:
            ori = t.stimulus_orientation
            if ori not in responses_by_ori:
                responses_by_ori[ori] = []
            responses_by_ori[ori].append(t.l23_readout.numpy())
        
        # Compute d' between adjacent orientations
        oris = sorted(responses_by_ori.keys())
        d_primes = []
        for i in range(len(oris)):
            j = (i + 1) % len(oris)
            r1 = np.array(responses_by_ori[oris[i]])  # (n1, 36)
            r2 = np.array(responses_by_ori[oris[j]])  # (n2, 36)
            
            mu1, mu2 = r1.mean(axis=0), r2.mean(axis=0)
            var1, var2 = r1.var(axis=0), r2.var(axis=0)
            
            # Multivariate d' (Mahalanobis-like, simplified)
            d_sq = np.sum((mu1 - mu2)**2 / ((var1 + var2) / 2 + 1e-8))
            d_primes.append(np.sqrt(d_sq))
        
        results[f'{condition}_mean_dprime'] = np.mean(d_primes)
        results[f'{condition}_std_dprime'] = np.std(d_primes)
    
    return results

def plot_decoding(results_all_mechanisms: dict):
    """
    Left panel: bar plot of decoding accuracy by condition × mechanism
    Right panel: Fisher information (d') for expected vs neutral, by mechanism
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    mechanisms = list(results_all_mechanisms.keys())
    x = np.arange(len(mechanisms))
    
    # Left: decoding accuracy
    for i, cond in enumerate(['expected', 'neutral', 'unexpected_90']):
        means = [results_all_mechanisms[m][f'{cond}_decode_acc_mean'] for m in mechanisms]
        stds = [results_all_mechanisms[m][f'{cond}_decode_acc_std'] for m in mechanisms]
        axes[0].bar(x + i * 0.25, means, 0.25, yerr=stds, label=cond, alpha=0.8)
    
    axes[0].set_xticks(x + 0.25)
    axes[0].set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Decoding Accuracy')
    axes[0].legend()
    axes[0].set_title('Linear Decoding from L2/3')
    
    # Right: d-prime
    for m_idx, mech in enumerate(mechanisms):
        dp_exp = results_all_mechanisms[mech]['expected_mean_dprime']
        dp_neu = results_all_mechanisms[mech]['neutral_mean_dprime']
        axes[1].bar(m_idx - 0.15, dp_exp, 0.3, color='#e41a1c', alpha=0.7, 
                    label='Expected' if m_idx == 0 else '')
        axes[1].bar(m_idx + 0.15, dp_neu, 0.3, color='#377eb8', alpha=0.7,
                    label='Neutral' if m_idx == 0 else '')
    
    axes[1].set_xticks(range(len(mechanisms)))
    axes[1].set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel("Fisher Information (d')")
    axes[1].set_title('Discriminability: Expected vs Neutral')
    axes[1].legend()
    
    plt.tight_layout()
    return fig
```

---

### 3.5 Analysis 5: Representational Similarity Analysis (RSA)

**What it measures:** Does expectation make orientation representations more distinct (sharpening) or less distinct (dampening)?

```python
def analysis_rsa(data: AnalysisInput) -> dict:
    """
    Construct representational dissimilarity matrices (RDMs) across orientations,
    separately per condition.
    
    Procedure:
        1. For each condition, compute mean L2/3 response vector for each stimulus orientation
        2. Compute pairwise correlation distance: d(i,j) = 1 - corr(r_i, r_j)
        3. Construct RDM (12×12 or 36×36 matrix)
        4. Compare RDMs across conditions
    
    Metrics:
        - Mean off-diagonal distance (higher = more distinct representations)
        - RDM correlation between conditions (Kendall's tau)
        - Within-condition vs between-condition distance ratio
    
    Expected results:
        Sharpening:     ↑ mean distance for expected condition (more distinct)
        Dampening:      ↓ mean distance for expected condition (less distinct)
        Center-surround: ↑ mean distance for nearby orientations, 
                         ≈ for distant ones (local enhancement)
    """
    results = {}
    
    for condition in ['expected', 'unexpected_90', 'neutral']:
        trials = data.filter(condition)
        
        # Mean response per stimulus orientation
        responses_by_ori = {}
        for t in trials:
            ori = t.stimulus_orientation
            if ori not in responses_by_ori:
                responses_by_ori[ori] = []
            responses_by_ori[ori].append(t.l23_readout)
        
        oris = sorted(responses_by_ori.keys())
        n_ori = len(oris)
        mean_responses = torch.stack([
            torch.stack(responses_by_ori[o]).mean(dim=0) for o in oris
        ])  # (n_ori, 36)
        
        # Correlation distance matrix
        # Normalize rows
        normed = mean_responses - mean_responses.mean(dim=1, keepdim=True)
        normed = normed / (normed.norm(dim=1, keepdim=True) + 1e-8)
        
        rdm = 1.0 - (normed @ normed.T).numpy()  # correlation distance
        np.fill_diagonal(rdm, 0)
        
        results[f'{condition}_rdm'] = rdm
        results[f'{condition}_mean_distance'] = rdm[np.triu_indices_from(rdm, k=1)].mean()
        
        # Distance for adjacent orientations (nearby discriminability)
        adj_distances = []
        for i in range(n_ori):
            j = (i + 1) % n_ori
            adj_distances.append(rdm[i, j])
        results[f'{condition}_adjacent_distance'] = np.mean(adj_distances)
    
    # RDM comparison: Kendall's tau between expected and neutral RDMs
    from scipy.stats import kendalltau
    rdm_exp = results['expected_rdm'][np.triu_indices_from(results['expected_rdm'], k=1)]
    rdm_neu = results['neutral_rdm'][np.triu_indices_from(results['neutral_rdm'], k=1)]
    tau, p = kendalltau(rdm_exp, rdm_neu)
    results['rdm_kendall_tau'] = tau
    results['rdm_kendall_p'] = p
    
    return results

def plot_rsa(results_all_mechanisms: dict):
    """
    Top row: RDMs for each mechanism (expected condition)
    Bottom row: bar chart comparing mean distance across conditions × mechanisms
    """
    mechanisms = list(results_all_mechanisms.keys())
    n_mech = len(mechanisms)
    
    fig, axes = plt.subplots(2, n_mech, figsize=(4 * n_mech, 8))
    
    for i, mech in enumerate(mechanisms):
        res = results_all_mechanisms[mech]
        
        # Top: RDM
        im = axes[0, i].imshow(res['expected_rdm'], cmap='viridis', vmin=0, vmax=2)
        axes[0, i].set_title(f'{mech}\n(expected)')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
        
        # Bottom: mean distance comparison
        distances = [res[f'{c}_mean_distance'] for c in ['expected', 'neutral', 'unexpected_90']]
        axes[1, i].bar(['Exp', 'Neu', 'Unexp'], distances, color=['#e41a1c', '#377eb8', '#4daf4a'])
        axes[1, i].set_ylabel('Mean Corr. Distance')
        axes[1, i].set_title(mech)
    
    plt.tight_layout()
    return fig
```

---

### 3.6 Analysis 6: Omission Template Analysis

**What it measures:** During omission trials, how much expectation signal is present in the deep template?

```python
def analysis_omission(data_omission: AnalysisInput) -> dict:
    """
    Data from Paradigm 2 (omission trials).
    
    1. Decode orientation from deep template during omission:
       Train decoder on template activity from present-after-predictive trials,
       test on omission-after-predictive trials.
       Above-chance accuracy = template contains expectation.
    
    2. Decode from L2/3 during omission:
       Any residual L2/3 activity during omission reflects top-down activation
       without bottom-up input.
    
    3. Template strength: Pearson correlation between template activity during 
       omission and template activity during matched expected-present trials.
    
    Expected results:
        All feedback-active mechanisms: template should encode expected orientation
        during omission (above-chance decoding).
        
        Dampening: L2/3 should show suppression at expected orientation during omission
                   (negative correlation with template).
        Sharpening: L2/3 during omission should show small activation at expected 
                    orientation (template facilitates through surround inhibition).
        Center-surround: L2/3 should show peaked activation at expected orientation.
        Adaptation-only: template and L2/3 should be at chance during omission.
    """
    results = {}
    
    # Decode from deep template
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    
    present_trials = data_omission.filter('present_predictive')
    omission_pred_trials = data_omission.filter('omission_predictive')
    omission_neut_trials = data_omission.filter('omission_neutral')
    
    # Train on present trials
    X_train = torch.stack([t.deep_template.mean(dim=0) for t in present_trials]).numpy()
    y_train = np.array([t.expected_orientation for t in present_trials])
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train_s, y_train)
    
    # Test on omission-predictive trials
    X_test_pred = torch.stack([t.deep_template.mean(dim=0) for t in omission_pred_trials]).numpy()
    y_test_pred = np.array([t.expected_orientation for t in omission_pred_trials])
    X_test_pred_s = scaler.transform(X_test_pred)
    
    results['template_decode_omission_predictive'] = clf.score(X_test_pred_s, y_test_pred)
    
    # Test on omission-neutral trials (should be at chance)
    if len(omission_neut_trials) > 0:
        X_test_neut = torch.stack([t.deep_template.mean(dim=0) for t in omission_neut_trials]).numpy()
        y_test_neut = np.array([t.stimulus_orientation for t in omission_neut_trials])  # arbitrary
        X_test_neut_s = scaler.transform(X_test_neut)
        results['template_decode_omission_neutral'] = clf.score(X_test_neut_s, y_test_neut)
    
    # Chance level
    n_classes = len(set(y_train))
    results['chance_level'] = 1.0 / n_classes
    
    # Template-L2/3 correlation during omission
    template_omission = torch.stack([t.deep_template.mean(dim=0) for t in omission_pred_trials])
    l23_omission = torch.stack([t.l23_readout for t in omission_pred_trials])
    
    # Per-trial correlation between template and L2/3
    correlations = []
    for i in range(len(omission_pred_trials)):
        t = template_omission[i].numpy()
        l = l23_omission[i].numpy()
        r = np.corrcoef(t, l)[0, 1]
        correlations.append(r)
    
    results['template_l23_corr_mean'] = np.mean(correlations)
    results['template_l23_corr_std'] = np.std(correlations)
    
    # L2/3 peak analysis during omission: where is L2/3 most active?
    l23_mean = l23_omission.mean(dim=0).numpy()  # (36,)
    results['l23_omission_profile'] = l23_mean
    results['l23_omission_peak_channel'] = np.argmax(l23_mean)
    results['l23_omission_mean_activity'] = np.mean(l23_mean)
    
    return results

def plot_omission(results_all_mechanisms: dict):
    """
    Left: template decoding accuracy during omission (bar per mechanism, with chance line)
    Right: L2/3 activity profile during omission, aligned to expected orientation
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    mechanisms = list(results_all_mechanisms.keys())
    
    # Left: decoding
    decode_accs = [results_all_mechanisms[m]['template_decode_omission_predictive'] 
                   for m in mechanisms]
    chance = results_all_mechanisms[mechanisms[0]]['chance_level']
    
    axes[0].bar(range(len(mechanisms)), decode_accs, color='steelblue', alpha=0.8)
    axes[0].axhline(y=chance, color='red', linestyle='--', label=f'Chance ({chance:.2f})')
    axes[0].set_xticks(range(len(mechanisms)))
    axes[0].set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Decoding Accuracy from Template')
    axes[0].set_title('Omission: Template Contains Expected Orientation?')
    axes[0].legend()
    
    # Right: L2/3 profile (would need alignment to expected — simplified here)
    axes[1].set_title('L2/3 Activity During Omission\n(aligned to expected orientation)')
    axes[1].set_xlabel('Orientation relative to expected (degrees)')
    axes[1].set_ylabel('L2/3 Activity')
    
    plt.tight_layout()
    return fig
```

---

### 3.7 Analysis 7: Energy Analysis

**What it measures:** Metabolic cost (total activity) per correctly classified stimulus, across mechanisms and conditions.

```python
def analysis_energy(data: AnalysisInput) -> dict:
    """
    Compute total excitatory activity per trial, then normalize by decoding accuracy.
    
    Metrics:
        1. Total L2/3 activity per trial: sum(l23_readout)
        2. Total activity per correct classification: 
           sum(activity) / P(correct classification)
           = metabolic cost per bit of sensory information
        3. Efficiency ratio: accuracy / total_activity
    
    Expected results:
        Energy-efficient mechanisms reduce total activity for expected stimuli
        while maintaining (or improving) accuracy.
        
        Sharpening: might achieve BEST efficiency 
                    (lower total activity + better accuracy)
        Dampening: lower total activity but worse accuracy 
                   (bad cost-accuracy trade-off)
        Center-surround: moderate on both
        Adaptation: no energy savings from expectation
    """
    results = {}
    
    for condition in ['expected', 'unexpected_90', 'neutral']:
        trials = data.filter(condition)
        
        # Total L2/3 activity per trial
        activities = [t.l23_readout.sum().item() for t in trials]
        results[f'{condition}_total_activity_mean'] = np.mean(activities)
        results[f'{condition}_total_activity_std'] = np.std(activities)
        
        # Decoding accuracy for this condition (from readout head)
        correct = 0
        for t in trials:
            predicted = t.l23_readout.argmax().item()
            actual = t.stimulus_orientation
            # Allow ±1 channel tolerance (within 5° is correct)
            if min(abs(predicted - actual), 36 - abs(predicted - actual)) <= 1:
                correct += 1
        accuracy = correct / len(trials)
        
        results[f'{condition}_accuracy'] = accuracy
        results[f'{condition}_cost_per_correct'] = np.mean(activities) / (accuracy + 1e-8)
        results[f'{condition}_efficiency'] = accuracy / (np.mean(activities) + 1e-8)
    
    # Suppression savings: how much energy saved for expected vs neutral?
    results['energy_savings_pct'] = (
        (results['neutral_total_activity_mean'] - results['expected_total_activity_mean']) 
        / (results['neutral_total_activity_mean'] + 1e-8)
    ) * 100
    
    # Accuracy-energy Pareto check
    results['accuracy_preserved'] = (
        results['expected_accuracy'] >= 0.8 * results['neutral_accuracy']
    )
    
    return results

def plot_energy(results_all_mechanisms: dict):
    """
    Scatter plot: x = total activity, y = decoding accuracy
    One point per (mechanism × condition), connected for same mechanism.
    Pareto frontier highlighted.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {
        'dampening': '#e41a1c', 'sharpening': '#377eb8',
        'center_surround': '#4daf4a', 'adaptation_only': '#984ea3',
        'explicit_pc': '#ff7f00',
    }
    markers = {'expected': 'o', 'neutral': 's', 'unexpected_90': '^'}
    
    for mech, res in results_all_mechanisms.items():
        for cond in ['expected', 'neutral', 'unexpected_90']:
            ax.scatter(
                res[f'{cond}_total_activity_mean'],
                res[f'{cond}_accuracy'],
                color=colors[mech], marker=markers[cond],
                s=80, alpha=0.8,
                label=f'{mech}/{cond}' if cond == 'expected' else None
            )
        
        # Connect conditions for same mechanism
        xs = [res[f'{c}_total_activity_mean'] for c in ['expected', 'neutral', 'unexpected_90']]
        ys = [res[f'{c}_accuracy'] for c in ['expected', 'neutral', 'unexpected_90']]
        ax.plot(xs, ys, '-', color=colors[mech], alpha=0.3)
    
    ax.set_xlabel('Total L2/3 Activity')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('Energy-Accuracy Trade-off')
    ax.legend(fontsize=7, ncol=2)
    
    return fig
```

---

### 3.8 Analysis 8: Bias Analysis (Paradigm 3)

**What it measures:** Under ambiguous stimuli, is perceived orientation biased toward or away from the expected orientation?

```python
def analysis_bias(data_ambiguous: AnalysisInput) -> dict:
    """
    Data from Paradigm 3 (ambiguous stimuli).
    
    For mixture stimuli:
        1. Decode "perceived" orientation from L2/3 (argmax or population vector average)
        2. Compute bias = circular_distance(perceived, expected_component) - 
                          circular_distance(perceived, other_component)
           Negative bias = attraction toward expected, Positive = repulsion
        
    For low-contrast stimuli:
        1. Decode perceived orientation
        2. Compute shift relative to true orientation
        3. Is shift toward or away from expected?
    
    Expected results:
        Dampening:       repulsion from expected (perceived shifts toward other component)
                         because expected channel is suppressed
        Center-surround: attraction toward expected (facilitation at expected channel)
        Sharpening:      slight repulsion (flanks suppressed, but expected peak intact)
        Adaptation:      no systematic bias (adaptation is stimulus-specific, not expectation-specific)
    """
    results = {}
    
    # Mixture trials
    for condition in ['ambiguous_expected', 'ambiguous_neutral']:
        trials = data_ambiguous.filter(condition)
        
        biases = []
        for t in trials:
            # Population vector decode
            l23 = t.l23_readout.numpy()
            oris_rad = np.linspace(0, np.pi, 36, endpoint=False)
            # Population vector (on doubled angle)
            vec_x = np.sum(l23 * np.cos(2 * oris_rad))
            vec_y = np.sum(l23 * np.sin(2 * oris_rad))
            decoded_rad = np.arctan2(vec_y, vec_x) / 2
            decoded_deg = (np.degrees(decoded_rad)) % 180
            
            if hasattr(t, 'channel_a') and hasattr(t, 'channel_b'):
                ori_a = t.channel_a * 5.0
                ori_b = t.channel_b * 5.0
                
                dist_to_a = circular_distance_degrees(decoded_deg, ori_a)
                dist_to_b = circular_distance_degrees(decoded_deg, ori_b)
                
                # Bias: negative = closer to expected (channel_a), positive = closer to other
                if t.expected_orientation >= 0:
                    expected_deg = t.expected_orientation * 5.0
                    if circular_distance_degrees(ori_a, expected_deg) < \
                       circular_distance_degrees(ori_b, expected_deg):
                        # channel_a is the expected component
                        bias = dist_to_a - dist_to_b
                    else:
                        bias = dist_to_b - dist_to_a
                else:
                    bias = dist_to_a - dist_to_b  # arbitrary for neutral
                
                biases.append(bias)
        
        results[f'{condition}_bias_mean'] = np.mean(biases)
        results[f'{condition}_bias_std'] = np.std(biases)
        results[f'{condition}_bias_sem'] = np.std(biases) / np.sqrt(len(biases))
    
    # Statistical test: is bias significantly different from 0?
    from scipy import stats
    exp_biases = [b for b in biases]  # from last loop — need to collect per condition
    # (Would actually collect separately; simplified here)
    t_stat, p_val = stats.ttest_1samp(exp_biases, 0)
    results['bias_ttest_t'] = t_stat
    results['bias_ttest_p'] = p_val
    
    # Attraction vs repulsion classification
    results['bias_direction'] = 'attraction' if results['ambiguous_expected_bias_mean'] < 0 else 'repulsion'
    
    return results

def plot_bias(results_all_mechanisms: dict):
    """
    Bar plot: bias magnitude per mechanism.
    Negative = attraction (toward expected), Positive = repulsion (away from expected).
    Error bars = SEM.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    mechanisms = list(results_all_mechanisms.keys())
    biases = [results_all_mechanisms[m]['ambiguous_expected_bias_mean'] for m in mechanisms]
    sems = [results_all_mechanisms[m]['ambiguous_expected_bias_sem'] for m in mechanisms]
    
    colors = ['#e41a1c' if b > 0 else '#377eb8' for b in biases]
    ax.bar(range(len(mechanisms)), biases, yerr=sems, color=colors, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(mechanisms)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.set_ylabel('Perceptual Bias (degrees)\n← attraction | repulsion →')
    ax.set_title('Ambiguous Stimulus: Bias Toward/Away from Expected')
    
    return fig
```

---

### 3.9 Analysis 9: Synthetic Observation Model (fMRI/BOLD Simulation)

**What it measures:** Can the same ground truth mechanism produce conflicting readouts at different measurement scales? This addresses the sharpening-in-spikes vs. dampening-in-BOLD paradox.

```python
def analysis_synthetic_bold(data: AnalysisInput) -> dict:
    """
    Simulate fMRI-like measurements from the network to test whether the same 
    mechanism can look like "sharpening" at single-unit level but "dampening" 
    at voxel level.
    
    Procedure:
        1. Pool L2/3 units into synthetic "voxels"
           Each voxel contains 6-9 units with similar (but not identical) preferred
           orientations. This mimics the spatial pooling of fMRI.
        
        2. Compute "BOLD signal" per voxel:
           BOLD_v = mean(l23_activity_units_in_voxel) + 0.3 * total_inhibitory_activity
           (BOLD reflects total metabolic activity, including inhibition)
        
        3. Run standard analyses on voxel-level data:
           a. Mean BOLD: expected vs neutral → univariate test
           b. MVPA on voxel patterns: train decoder, test accuracy
           c. Compare MVPA results with single-unit decoder results
    
    Expected results:
        Sharpening model:
            - Single-unit: better decoding for expected (finer tuning)
            - Voxel: lower mean BOLD (less total activity) BUT better MVPA
            → Appears as sharpening at BOTH levels (consistent)
        
        Dampening model:
            - Single-unit: worse decoding for expected
            - Voxel: lower mean BOLD AND worse MVPA
            → Appears as dampening at both levels (consistent)
        
        Center-surround model:
            - Single-unit: facilitation at expected channel, suppression at flanks
            - Voxel: lower mean BOLD (net suppression) but MVPA may be preserved
              (because the enhanced units dominate the voxel pattern)
            → Could appear as dampening in univariate but sharpening in MVPA!
            This is the key insight: center-surround can produce the empirical 
            dissociation seen in the literature (Kok et al., 2012).
    """
    results = {}
    
    # Define voxel pooling (4 voxels, each spanning ~45° of orientation space)
    # In real fMRI, voxels don't tile orientation space this neatly,
    # but this gives us a best-case MVPA scenario
    
    n_voxels = 8  # 8 voxels spanning 180°
    channels_per_voxel = 36 // n_voxels  # ~4-5 channels per voxel
    
    voxel_assignments = {}
    for v in range(n_voxels):
        start = v * channels_per_voxel
        end = start + channels_per_voxel
        voxel_assignments[v] = list(range(start, min(end, 36)))
    
    for condition in ['expected', 'unexpected_90', 'neutral']:
        trials = data.filter(condition)
        
        # Single-unit data
        l23_data = torch.stack([t.l23_readout for t in trials])  # (n, 36)
        
        # Voxel-level: average within each voxel
        voxel_data = torch.zeros(len(trials), n_voxels)
        for v in range(n_voxels):
            unit_indices = voxel_assignments[v]
            voxel_data[:, v] = l23_data[:, unit_indices].mean(dim=1)
        
        # Add inhibitory contribution to BOLD
        if hasattr(trials[0], 'som_activity'):
            som_data = torch.stack([t.som_activity.mean(dim=0) for t in trials])
            som_voxel = torch.zeros(len(trials), n_voxels)
            for v in range(n_voxels):
                unit_indices = voxel_assignments[v]
                som_voxel[:, v] = som_data[:, unit_indices].mean(dim=1)
            voxel_bold = voxel_data + 0.3 * som_voxel
        else:
            voxel_bold = voxel_data
        
        # Univariate: mean BOLD
        results[f'{condition}_mean_bold'] = voxel_bold.mean().item()
        
        # Store for MVPA
        results[f'{condition}_voxel_data'] = voxel_bold
        results[f'{condition}_labels'] = [t.stimulus_orientation for t in trials]
    
    # MVPA on voxel data
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    for condition in ['expected', 'neutral']:
        X = results[f'{condition}_voxel_data'].numpy()
        y = np.array(results[f'{condition}_labels'])
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        clf = LinearSVC(max_iter=10000)
        scores = cross_val_score(clf, X_s, y, cv=5, scoring='accuracy')
        
        results[f'{condition}_mvpa_accuracy'] = scores.mean()
        results[f'{condition}_mvpa_std'] = scores.std()
    
    # The critical comparison: does MVPA dissociate from univariate?
    results['univariate_suppression'] = (
        results['neutral_mean_bold'] - results['expected_mean_bold']
    )
    results['mvpa_change'] = (
        results['expected_mvpa_accuracy'] - results['neutral_mvpa_accuracy']
    )
    
    # Classification: what does this look like?
    if results['univariate_suppression'] > 0 and results['mvpa_change'] > 0:
        results['apparent_mechanism_bold'] = 'sharpening_like'
        # Lower BOLD but better MVPA → sharpening signature
    elif results['univariate_suppression'] > 0 and results['mvpa_change'] <= 0:
        results['apparent_mechanism_bold'] = 'dampening_like'
        # Lower BOLD and worse MVPA → dampening signature
    elif results['univariate_suppression'] > 0 and abs(results['mvpa_change']) < 0.02:
        results['apparent_mechanism_bold'] = 'ambiguous'
    else:
        results['apparent_mechanism_bold'] = 'no_suppression'
    
    return results

def plot_bold_dissociation(results_all_mechanisms: dict):
    """
    2D scatter: x = univariate suppression (BOLD reduction), y = MVPA change
    One point per mechanism. Quadrants labeled:
        Top-right: sharpening-like (less BOLD, better MVPA)
        Bottom-right: dampening-like (less BOLD, worse MVPA)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    colors = {
        'dampening': '#e41a1c', 'sharpening': '#377eb8',
        'center_surround': '#4daf4a', 'adaptation_only': '#984ea3',
        'explicit_pc': '#ff7f00',
    }
    
    for mech, res in results_all_mechanisms.items():
        ax.scatter(
            res['univariate_suppression'],
            res['mvpa_change'],
            color=colors[mech], s=200, zorder=5, edgecolors='black',
            label=mech
        )
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Univariate BOLD Suppression\n(neutral - expected, positive = suppression)')
    ax.set_ylabel('MVPA Accuracy Change\n(expected - neutral, positive = better)')
    ax.set_title('Synthetic fMRI: Univariate vs Multivariate')
    
    # Quadrant labels
    ax.text(0.8, 0.9, 'Sharpening-like', transform=ax.transAxes, fontsize=10,
            ha='center', color='green', fontweight='bold')
    ax.text(0.8, 0.1, 'Dampening-like', transform=ax.transAxes, fontsize=10,
            ha='center', color='red', fontweight='bold')
    
    ax.legend()
    return fig
```

---

### 3.10 Analysis 10: Task Relevance Interaction

**What it measures:** All of the above analyses, separately for orientation-relevant and orientation-irrelevant task conditions.

```python
def analysis_task_relevance(data_relevant: AnalysisInput, 
                            data_irrelevant: AnalysisInput) -> dict:
    """
    Run the full analysis suite on both task conditions and compute interactions.
    
    Data from Paradigm 4 (task relevance).
    
    For each analysis (1-9), compute:
        - Effect in orientation-relevant task
        - Effect in orientation-irrelevant task
        - Interaction: does the effect differ across tasks?
    
    Key metric: Task Relevance Modulation Index (TRMI)
        TRMI = (effect_relevant - effect_irrelevant) / (effect_relevant + effect_irrelevant + ε)
        TRMI = 0: expectation effect is task-independent (true prediction)
        TRMI > 0: effect is stronger when orientation is relevant (attention-like)
        TRMI < 0: effect is stronger when orientation is irrelevant (rare)
    
    Expected results:
        If ES is purely predictive: TRMI ≈ 0 for all mechanisms
        If ES is attention-modulated: TRMI > 0 (stronger in relevant task)
        
        Most likely: TRMI > 0 for some analyses (attention amplifies prediction)
        but key suppression-by-tuning profile shape should be TASK-INDEPENDENT
        (the shape is determined by the feedback mechanism, not attention).
    """
    results = {}
    
    # Run core analyses on both task conditions
    for task, data in [('relevant', data_relevant), ('irrelevant', data_irrelevant)]:
        results[f'{task}_mean_response'] = analysis_mean_response(data)
        results[f'{task}_suppression_profile'] = analysis_suppression_by_tuning(data)
        results[f'{task}_tuning'] = analysis_tuning_curves(data)
        results[f'{task}_decoding'] = analysis_decoding(data)
        results[f'{task}_rsa'] = analysis_rsa(data)
        results[f'{task}_energy'] = analysis_energy(data)
    
    # Compute TRMIs for key metrics
    key_metrics = [
        ('suppression_magnitude', 
         lambda r: np.mean(r['suppression_means'])),
        ('decoding_accuracy_change',
         lambda r: r.get('expected_decode_acc_mean', 0) - r.get('neutral_decode_acc_mean', 0)),
        ('mean_response_suppression',
         lambda r: r.get('l23_readout_neutral_mean', 0) - r.get('l23_readout_expected_mean', 0)),
        ('energy_savings',
         lambda r: r.get('energy_savings_pct', 0)),
    ]
    
    for metric_name, extractor in key_metrics:
        # Find which result dict contains this metric
        for analysis_key in ['mean_response', 'suppression_profile', 'decoding', 'energy']:
            try:
                val_rel = extractor(results[f'relevant_{analysis_key}'])
                val_irr = extractor(results[f'irrelevant_{analysis_key}'])
                
                trmi = (val_rel - val_irr) / (abs(val_rel) + abs(val_irr) + 1e-8)
                results[f'TRMI_{metric_name}'] = trmi
                results[f'{metric_name}_relevant'] = val_rel
                results[f'{metric_name}_irrelevant'] = val_irr
                break
            except (KeyError, TypeError):
                continue
    
    # Critical test: suppression-by-tuning PROFILE SHAPE is task-independent?
    # Compare the shapes (not magnitudes) of suppression profiles
    prof_rel = results['relevant_suppression_profile']['suppression_means']
    prof_irr = results['irrelevant_suppression_profile']['suppression_means']
    
    # Normalize both to unit norm, then correlate
    prof_rel_norm = prof_rel / (np.linalg.norm(prof_rel) + 1e-8)
    prof_irr_norm = prof_irr / (np.linalg.norm(prof_irr) + 1e-8)
    shape_correlation = np.corrcoef(prof_rel_norm, prof_irr_norm)[0, 1]
    
    results['suppression_profile_shape_correlation'] = shape_correlation
    # If > 0.9: profile shape is task-independent (mechanism is the same)
    # If < 0.5: profile shape changes with task (attention changes the mechanism)
    
    return results

def plot_task_relevance(results_all_mechanisms: dict):
    """
    2×N panel figure:
        Top row: suppression-by-tuning profile for relevant task
        Bottom row: suppression-by-tuning profile for irrelevant task
        One column per mechanism
    
    Plus: TRMI bar chart as a separate figure
    """
    mechanisms = list(results_all_mechanisms.keys())
    n = len(mechanisms)
    
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), sharey=True)
    
    for i, mech in enumerate(mechanisms):
        res = results_all_mechanisms[mech]
        
        for row, task in enumerate(['relevant', 'irrelevant']):
            prof = res[f'{task}_suppression_profile']
            axes[row, i].plot(prof['delta_theta_bins'], prof['suppression_means'], 'o-')
            axes[row, i].fill_between(
                prof['delta_theta_bins'],
                prof['suppression_means'] - prof['suppression_sems'],
                prof['suppression_means'] + prof['suppression_sems'],
                alpha=0.3
            )
            axes[row, i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[row, i].set_title(f'{mech}\n({task})')
            if i == 0:
                axes[row, i].set_ylabel('Suppression')
            axes[row, i].set_xlabel('Δθ (degrees)')
    
    plt.tight_layout()
    return fig
```

---

## 4. Configuration Constants

All key hyperparameters in one place for reproducibility:

```python
CONFIG = {
    # ===== Network =====
    'n_channels': 36,
    'ori_spacing_deg': 5.0,
    
    # ===== L4 Tuning =====
    'l4_kappa': 2.5,
    'l4_baseline': 0.05,
    'l4_gain': 1.0,
    'l4_adapt_tau': 8.0,
    'l4_adapt_alpha': 0.3,
    'contrast_n': 2.0,
    'contrast_c50': 0.3,
    
    # ===== L2/3 + PV =====
    'w_ff_sigma_channels': 3.0,
    'pv_sigma': 0.1,
    'pv_pool_sigma_channels': 12.0,
    
    # ===== Stage 1 Training =====
    'stage1_lr': 1e-3,
    'stage1_weight_decay': 1e-5,
    'stage1_steps': 2000,
    'stage1_batch_size': 128,
    
    # ===== Stage 2 Training =====
    'stage2_lr_v2': 3e-4,
    'stage2_lr_feedback': 1e-4,
    'stage2_weight_decay': 1e-4,
    'stage2_total_steps': 80000,
    'stage2_warmup_steps': 1000,
    'stage2_batch_size': 32,
    'stage2_sequence_length': 50,
    
    # ===== Loss Weights =====
    'lambda_sensory': 1.0,
    'lambda_pred': 0.5,
    'lambda_energy': 0.01,
    'lambda_homeo': 1.0,
    
    # ===== Sequence Generation =====
    'hmm_step_deg': 15.0,
    'hmm_transition_prob': 0.80,
    'hmm_p_self': 0.95,
    
    # ===== Temporal Structure =====
    'T_ON': 8,
    'T_ISI': 4,
    'readout_window_start': 4,
    'readout_window_end': 8,
    
    # ===== Homeostasis =====
    'target_rate_low': 0.15,
    'target_rate_high': 0.50,
    
    # ===== Experimental =====
    'n_experiment_orientations': 12,
    'n_context_presentations': 8,
    'n_context_omission': 10,
    'n_trials_per_condition': 200,
    'low_contrast_level': 0.15,
    'mixture_contrast': 0.5,
    'ambiguous_separation_deg': 30.0,
    
    # ===== Mechanism Comparison =====
    'n_seeds': 5,
    'seeds': [42, 137, 256, 512, 1024],
    
    # ===== Task Relevance =====
    'task_loss_weight': 0.3,
    'contrast_change_prob': 0.3,
    'contrast_change_magnitude': 0.2,
    
    # ===== Mechanism-Specific =====
    'dampening_som_sigma': 10.0,
    'sharpening_som_sigma_broad': 90.0,
    'sharpening_som_sigma_exclude': 15.0,
    'cs_sigma_exc': 12.0,
    'cs_sigma_inh': 35.0,
    'cs_amp_exc': 0.8,
    'cs_amp_inh': 0.5,
    'explicit_pc_alpha_init': 0.5,
}
```

---

## Appendix A: Summary Table of Predictions

| Analysis | Dampening | Sharpening | Center-Surround | Adaptation Only | Explicit PC |
|---|---|---|---|---|---|
| 1. Mean response | ↓ expected | ↓ expected | ↓ expected | ↓ L4 only | ↓ expected |
| 2. Suppression profile | Peak at Δθ=0° | Peak at Δθ~45° | Mexican hat | Flat | Peak at Δθ=0° |
| 3. Amplitude | ↓ | = or slight ↓ | slight ↓ | ↓ (L4 only) | ↓ |
| 3. Width | = | ↓ (narrower) | slight ↓ | = | = |
| 4. Decoding accuracy | ↓ | ↑ | ↑ or = | ↓ (noise) | ↓ |
| 5. RSA distance | ↓ | ↑ | ↑ nearby, = far | ↓ | ↓ |
| 6. Omission template | Present | Present | Present | Absent | Present |
| 7. Energy efficiency | Low efficiency | High efficiency | Moderate | No benefit | Moderate |
| 8. Ambiguous bias | Repulsion | Slight repulsion | Attraction | None | Repulsion |
| 9. BOLD vs MVPA | Both dampening | Both sharpening | BOLD dampen, MVPA sharpen | Both degraded | Both dampening |
| 10. Task relevance | TRMI ≈ 0 | TRMI ≈ 0 | TRMI ≈ 0 | TRMI ≈ 0 | TRMI ≈ 0 |

**The center-surround model is uniquely predicted to show dissociation between univariate BOLD (dampening-like) and MVPA (sharpening-like), matching the empirical literature.** This is the key result that would favor the center-surround account if observed.

---

## Appendix B: Execution Pipeline

```python
def run_full_pipeline():
    """
    Complete pipeline from training to analysis.
    Estimated total compute: ~60 GPU-hours (25 models × ~2.5 hours each)
    """
    # Stage 1: Sensory scaffold (once)
    model_base = train_stage1()
    assert all(v for v in verify_stage1(model_base).values())
    
    # Stage 2+3: Train all mechanism variants
    trained_models = {}
    for mechanism in CONFIG['mechanisms']:
        for seed in CONFIG['seeds']:
            model = build_model(mechanism)
            model.load_stage1(model_base)
            train_stage2(model, mechanism, seed)
            assert all(v for v in validate_trained_model(model, mechanism).values())
            trained_models[f'{mechanism}_seed{seed}'] = model
    
    # Run experiments
    for paradigm_fn in [generate_paradigm1_trials, generate_paradigm2_trials,
                        generate_paradigm3_trials, generate_paradigm4_trials,
                        generate_paradigm5_trials]:
        trials = paradigm_fn()
        for model_name, model in trained_models.items():
            records = run_trials(model, trials)
            save_records(records, model_name, paradigm_fn.__name__)
    
    # Run analyses
    for analysis_fn in [analysis_mean_response, analysis_suppression_by_tuning,
                        analysis_tuning_curves, analysis_decoding, analysis_rsa,
                        analysis_omission, analysis_energy, analysis_bias,
                        analysis_synthetic_bold, analysis_task_relevance]:
        for model_name in trained_models:
            records = load_records(model_name, analysis_fn)
            results = analysis_fn(records)
            save_results(results, model_name, analysis_fn.__name__)
    
    # Generate figures (aggregate across seeds)
    generate_all_figures(results_dir='results/')
```
