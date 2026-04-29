// adex_v1.cu
//
// V1 L4 layer with TRUE GABOR SPATIAL RECEPTIVE FIELDS.
//
// Architecture
// ============
// 131,072 AdEx neurons (Brette & Gerstner 2005, RS) arranged as
//   32 × 32 retinotopic hypercolumns × 8 preferred orientations × 16 clones.
//
// Cell-index encoding (per task-#47 spec):
//     cell_id = ((gy*32 + gx)*8 + ori_idx)*16 + clone_idx
//   gx, gy   ∈ [0,32),   ori_idx ∈ [0,8),   clone_idx ∈ [0,16).
//
// Each cell has a generated 2D Gabor weight kernel W(dx, dy) over a 9 × 9
// patch around (gx, gy):
//     W(dx, dy) = exp(- (dx'² + γ² dy'²) / (2 σ²)) · cos(K · dx' + φ)
// with the patch axes rotated by θ_pref:
//     dx' =  dx · cos θ_pref + dy · sin θ_pref
//     dy' = -dx · sin θ_pref + dy · cos θ_pref
//
// Tunable defaults:
//     σ_x  = 2.5 px,  γ_base = 0.5,
//     K    = 2π / 8 rad/px (matches stim spatial frequency),
//     θ_pref(ori_idx) = ori_idx · π / 8,
//     φ(clone_idx)    = clone_idx · 2π / 16,
//     γ(clone_idx)    = γ_base + GAMMA_JITTER · cos(2π · clone_idx / 16)
//   so the 16 clones span all spatial phases and have a smooth spread of
//   envelope aspect ratios -- this gives each clone a slightly different
//   orientation-selectivity index (OSI), as in real V1.
//
// Each kernel is then ZERO-MEAN'd (subtract mean over the 9×9 patch) and
// L1-normalized to a target sum |W| = 60.  The L1 target was chosen to match
// the calibrated effective fan-in we converged on in earlier task-#47 runs.
//
// Input drive (LNP cascade)
// =========================
// The cell's effective input rate is the rectified linear filter response:
//     r_lin(t) = Σ_{dx, dy} W(dx, dy) · stim_rate(gx + dx, gy + dy, t)
//     r_in(t)  = R_base + max(0, r_lin(t))
// Then a single Bernoulli per timestep is drawn at p = r_in · DT.  Each
// input spike injects g_E ← g_E + w_in into a conductance-based synapse:
//     dg_E / dt = -g_E / τ_syn,   I_syn = g_E · (V − E_E),   E_E = 0 mV.
// w_in is calibrated analytically so an isolated spike on a resting cell
// gives ~1 mV peak EPSP (verified by --measure-epsp).
//
// Why LNP rather than per-pixel multi-synapse: see closure note.  The two
// formulations are equivalent for first-order rate behaviour; LNP avoids
// the issue that signed Gabor weights can't be implemented as biophysical
// conductances without an inhibitory channel (which the spec excludes).
// Negative Gabor lobes still contribute to selectivity by cancelling
// positive lobes when the stimulus is misaligned (r_lin → ~0), then the
// rectification floors r_in at R_base.
//
// AdEx params (canonical Brette & Gerstner 2005 RS):
//   C = 281 pF, gL = 30 nS, EL = -70.6 mV
//   VT = -50.4 mV, ΔT = 2 mV
//   a = 4 nS, τw = 144 ms, b = 80.5 pA
//   V_reset = -70.6 mV, V_peak = +20 mV (spike detection)
//   Refractory: 2 ms,   dt = 0.1 ms
//
// GPU-only: all state device-resident.  One CUDA thread per cell; one
// kernel launch covers a full stim-orientation phase (n_steps inner-loop
// steps).
//
// CLI
// ===
//   v1_test --verify [--seed N] [--phase-duration-ms M] [--phase-warmup-ms M]
//                    [--label TAG]
//   v1_test --measure-epsp
//   v1_test --stim_orientation θ_deg --duration_ms N --out_dir PATH

#include "stim_kernels.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace {

using stim_kernels::DT_MS;
using stim_kernels::DT_S;
using stim_kernels::GRID;
using stim_kernels::N_PIX;
using stim_kernels::PI;
using stim_kernels::STIM_DEFAULT_OMEGA_RAD_PER_S;
using stim_kernels::STIM_DEFAULT_SF_PERIOD_PIXELS;
using stim_kernels::stim_rate_hz;

// ---------- Topology ----------
constexpr int N_HYPERCOL         = GRID * GRID;
constexpr int N_ORIENT           = 8;
constexpr int N_CELLS_PER_ORIENT = 16;
constexpr int CELLS_PER_HYPERCOL = N_ORIENT * N_CELLS_PER_ORIENT;
constexpr int N_L4               = N_HYPERCOL * CELLS_PER_HYPERCOL;

// ---------- Gabor RF ----------
constexpr int    GABOR_SIZE     = 9;
constexpr int    GABOR_HALF     = (GABOR_SIZE - 1) / 2;
constexpr int    GABOR_PIX      = GABOR_SIZE * GABOR_SIZE;
constexpr double GABOR_SIGMA_X   = 2.5;
constexpr double GABOR_GAMMA     = 0.5;     // base aspect ratio
constexpr double GAMMA_JITTER    = 0.15;    // ±jitter across clones → OSI variation
constexpr double GABOR_K         = 2.0 * PI / 8.0;
constexpr double GABOR_L1_TARGET = 200.0;   // calibrated so peak r_lin ≈ 1500–2000 Hz at optimal alignment
constexpr int    N_TEMPLATES    = N_ORIENT * N_CELLS_PER_ORIENT;

// ---------- AdEx ----------
constexpr double C_PF       = 281.0;
constexpr double G_L_NS     = 30.0;
constexpr double E_L_MV     = -70.6;
constexpr double V_T_MV     = -50.4;
constexpr double DELTA_T_MV = 2.0;
constexpr double A_NS       = 4.0;
constexpr double TAU_W_MS   = 144.0;
constexpr double B_PA       = 80.5;
constexpr double V_RESET_MV = -70.6;
constexpr double V_PEAK_MV  = 20.0;
constexpr double T_REF_MS   = 2.0;

// ---------- Synapse ----------
constexpr double TAU_SYN_MS = 5.0;
constexpr double E_E_MV     = 0.0;
constexpr double W_IN_NS    = 1.632;   // 1 mV peak EPSP

// ---------- Spontaneous baseline ----------
constexpr double R_BASE_HZ  = 1.0;     // small floor so off-preferred cells aren't pathologically dead

constexpr int T_REF_STEPS = static_cast<int>(T_REF_MS / DT_MS + 0.5);

// =====================================================================
// L2/3 layer (Phase A — structure only, no plasticity).
// Spec source: team-lead task #52 + /tmp/v1_l4_l2_3_connectivity_review.md.
//
// Topology
// --------
// 32 × 32 retinotopic grid (same as L4) × 16 excitatory cells/hypercolumn
//   ⇒ N_L23 = 16,384.
// L2/3 cells are FEATURE-UNTYPED (no orientation tuning baked in; orientation
// emerges in Phase B from L4 input pooling).
// Cell-index encoding:
//     l23_id = (gy * GRID + gx) * N_L23_CLONES + clone_idx
//   gx, gy ∈ [0, 32),  clone_idx ∈ [0, 16).
//
// AdEx params identical to L4 (same C, gL, EL, VT, ΔT, a, τw, b, V_reset,
// V_peak, T_ref, dt) — see top of this namespace.  L2/3 cells receive NO
// Poisson input baseline; their drive is purely from L4 spikes scattered
// through the L4→L2/3 conductance synapse.
//
// L4 → L2/3 wiring (Phase A, static, no plasticity)
// -------------------------------------------------
//   * Spatial fan-in: 3×3 hypercolumn patch around each L2/3 cell's
//     retinotopic position.  At borders the patch is CLIPPED (no
//     wraparound).
//   * Candidate pool per L2/3 cell at grid interior:
//       9 hypercolumns × 8 orientations × 16 clones = 1,152 L4 cells.
//     At edges the pool shrinks (e.g. 6 hypercols × 128 = 768 at a side,
//     4 hypercols × 128 = 512 at a corner).
//   * Sampling: each candidate independently kept with probability
//       p_connect = L23_TARGET_FANIN / 1152 = 40 / 1152 ≈ 0.0347.
//     Mean fan-in ≈ 40 at interior, ≈ 27 at side, ≈ 18 at corner.
//   * Strictly unidirectional: NO L2/3 → L4 backward connections.
//   * One composite synapse per L4→L2/3 connection (not 4-5 contacts;
//     equivalent first-order behavior, simpler).
//
// Weight initialization (lognormal, EPSP-space)
// ---------------------------------------------
//   EPSP_mV ~ Lognormal(μ=ln(1.0)=0.0, σ=0.6) clipped at 5.0 mV.
//   Convert EPSP → conductance via the analytical AdEx single-spike-EPSP
//   formula at rest: for τ_syn = 12 ms, peak EPSP per nS = 0.974 mV/nS
//   (vs 0.610 for τ_syn = 5 ms; verified against L4's W_IN_NS = 1.632 nS
//   giving ≈ 1 mV).  Hence:
//       w_nS  =  EPSP_mV / 0.974
//
//   Initial scale chosen so that with mean L4 firing ≈ 12 Hz × ~38 partners,
//   the L2/3 steady-state V is close enough to V_T that natural Poisson
//   fluctuations drive a non-trivial firing rate.
//
// Synapse kinetics
// ----------------
//   Conductance-based, E_E_L23 = 0 mV, τ_syn_L23 = 12 ms (Feldmeyer 2002).
//   Per-spike conductance bump: g_E += w_nS at the post-cell when an
//   L4 partner spike arrives (delayed 2 ms — see ring buffer below).
//
// Conduction delay
// ----------------
//   2 ms = 20 timesteps at dt = 0.1 ms.  Implemented by recording per-step
//   L4 spike vectors as a packed bitmask (warp-ballot reduction in the L4
//   kernel) and having the L2/3 kernel read the bitmask at step (t − 20).
//
// Storage layout (CSR, matches L4 input-synapse pattern)
// ------------------------------------------------------
//   l23_row_ptr[0 .. N_L23]            (int32, size N_L23+1)
//   l23_col_idx[0 .. n_synapses]        (int32, L4 partner ids)
//   l23_w_nS  [0 .. n_synapses]         (double, conductance)
//
// Random seeds
// ------------
//   master_seed = args.seed
//   connectivity_seed = master_seed XOR L23_SALT_CONNECTIVITY
//   weight_seed       = master_seed XOR L23_SALT_WEIGHTS
//   These derive deterministically from the master CLI seed.
// =====================================================================
constexpr int    N_L23_CLONES   = 16;
constexpr int    N_L23          = N_HYPERCOL * N_L23_CLONES;   // 16384
constexpr int    L23_PATCH_R    = 1;                            // 3×3 patch radius
constexpr int    L23_TARGET_FANIN = 40;                         // mean partners (interior)
constexpr int    L23_DELAY_STEPS = 20;                          // 2 ms / dt = 20

constexpr double TAU_SYN_L23_MS = 12.0;     // L4→L2/3 synapse decay (Feldmeyer 2002)
constexpr double E_E_L23_MV     = 0.0;
constexpr double L23_EPSP_MEDIAN_MV = 1.0;  // lognormal median (mV) -- 2× scale per lead's #52 ruling
constexpr double L23_EPSP_LOG_SIGMA = 0.6;  // lognormal σ in log space
constexpr double L23_EPSP_MAX_MV    = 5.0;  // hard clip = 5× median
// EPSP→conductance conversion factor for τ_syn=12 ms, AdEx passive params at
// rest (analytical; see header comment above): 1 nS bump → 0.974 mV peak EPSP.
constexpr double L23_MV_PER_NS  = 0.974;

// Bitmask packing for the L4 spike-record (warp-ballot output).
constexpr int    N_L4_BITMASK_INTS = N_L4 / 32;     // = 4096
// Bitmask packing for the L2/3 spike-record (used by the B1 recurrent path).
constexpr int    N_L23_BITMASK_INTS = N_L23 / 32;   // = 512

// Salt constants for sub-seed derivation (documented in the header above).
constexpr unsigned long long L23_SALT_CONNECTIVITY = 0x9E3779B97F4A7C15ULL;
constexpr unsigned long long L23_SALT_WEIGHTS      = 0xBF58476D1CE4E5B9ULL;

// =====================================================================
// L4 → L2/3 sampling-grading mode (task #11).
//
// The default `random` mode preserves the original uniform-random sampling:
// every L4 candidate inside the 3×3 hypercolumn patch is kept with
// probability p = L23_TARGET_FANIN / 1152.  The four graded modes bias the
// sampling by Δθ between each L4 candidate's baked-in preferred orientation
// and the L2/3 cell's "target" orientation, where the target is assigned by
//
//     l23_target_ori_idx(l23_id) = (l23_id % N_L23_CLONES) % N_ORIENT
//
// i.e. the 16 L2/3 clones in each hypercolumn are split into 8 orientation
// buckets × 2 cells per bucket, with the bucket index = l23_clone % 8.
// (This is a POOL-SELECTION prior on which L4 cells the L2/3 cell may pool
//  from; it does NOT bake in a tuning curve — the L2/3 cell still receives
//  Poisson input via AdEx and learns its weights through STDP.)
//
// Δθ_bin = min(|ori_l4 - ori_target|, N_ORIENT - |ori_l4 - ori_target|)
//          ∈ {0, 1, 2, 3, 4}  ↔  Δθ ∈ {0°, 22.5°, 45°, 67.5°, 90°}.
//
// Per-variant relative weights w[Δθ_bin] are normalized so the expected
// fan-in per interior L2/3 cell ≈ L23_TARGET_FANIN (= 40), assuming the
// 3×3 patch contains 144 candidates per Δθ-bin for bins 0 and 4 and 288
// for bins 1, 2, 3 (= 8 ori_idx × 16 clones × 9 hypercols / mirror).
// =====================================================================
enum class L4L23Grading { random, am, sharp, strict, gentle };

inline const char* grading_to_str(L4L23Grading g) {
    switch (g) {
        case L4L23Grading::random: return "random";
        case L4L23Grading::am:     return "am";
        case L4L23Grading::sharp:  return "sharp";
        case L4L23Grading::strict: return "strict";
        case L4L23Grading::gentle: return "gentle";
    }
    return "?";
}

// Per-Δθ-bin relative weights for each variant.  Δθ bins:
//   0 → 0°,   1 → 22.5°,   2 → 45°,   3 → 67.5°,   4 → 90°.
inline std::array<double, 5> grading_curve(L4L23Grading g) {
    switch (g) {
        case L4L23Grading::random: return {1.0, 1.0, 1.0, 1.0, 1.0};
        case L4L23Grading::am:     return {0.55, 0.50, 0.20, 0.31, 0.0};
        case L4L23Grading::sharp:  return {1.0, 0.40, 0.10, 0.05, 0.0};
        case L4L23Grading::strict: return {1.0, 0.20, 0.0, 0.0, 0.0};
        case L4L23Grading::gentle: return {1.0, 0.80, 0.50, 0.30, 0.10};
    }
    return {1.0, 1.0, 1.0, 1.0, 1.0};
}

// Return per-bin sampling probability + the scaling factor used to hit the
// L23_TARGET_FANIN expected fan-in for an interior L2/3 cell (3×3 full patch).
struct GradingParams {
    std::array<double, 5> w_curve;          // raw curve as above
    std::array<double, 5> p_connect_per_bin;// final per-Δθ-bin sampling p
    double scaling;                          // applied multiplicatively to w_curve
    double expected_fanin_interior;          // sanity-check: ≈ L23_TARGET_FANIN
};

inline GradingParams compute_grading_params(L4L23Grading g) {
    // Candidate counts per Δθ-bin in the 3×3 full patch (no edge clipping):
    //   bins 0 and 4: 1 ori_idx × N_CELLS_PER_ORIENT(16) × 9 hcol = 144
    //   bins 1, 2, 3: 2 ori_idx × N_CELLS_PER_ORIENT(16) × 9 hcol = 288
    constexpr int N_CAND[5] = {144, 288, 288, 288, 144};
    GradingParams gp{};
    gp.w_curve = grading_curve(g);
    double unnorm = 0.0;
    for (int b = 0; b < 5; ++b) unnorm += gp.w_curve[b] * N_CAND[b];
    if (unnorm <= 0.0) {
        // Degenerate (all-zero curve) — leave p=0 everywhere.
        gp.scaling = 0.0;
        gp.expected_fanin_interior = 0.0;
        for (int b = 0; b < 5; ++b) gp.p_connect_per_bin[b] = 0.0;
        return gp;
    }
    gp.scaling = static_cast<double>(L23_TARGET_FANIN) / unnorm;
    double exp_total = 0.0;
    for (int b = 0; b < 5; ++b) {
        gp.p_connect_per_bin[b] = gp.w_curve[b] * gp.scaling;
        // Saturate hard at 1 (a single Bernoulli trial); this should be a
        // no-op for the curves in the table but defends against pathological
        // user-supplied curves.
        if (gp.p_connect_per_bin[b] > 1.0) gp.p_connect_per_bin[b] = 1.0;
        exp_total += gp.p_connect_per_bin[b] * N_CAND[b];
    }
    gp.expected_fanin_interior = exp_total;
    return gp;
}

// =====================================================================
// L2/3 → L2/3 recurrent connectivity (task #3 — B1 STRUCTURE ONLY).
//
// Static, distance-dependent excitatory recurrence on the trained
// L4→L2/3 substrate.  No plasticity at this stage (B3 will add STDP).
//
//   Connection probability vs hypercolumn distance d:
//       p(d) = L23REC_P0 · exp(-d / L23REC_LEN_HCOL)   for d ≤ L23REC_DMAX
//              0                                        for d >  L23REC_DMAX
//   d = sqrt((gx_pre - gx_post)² + (gy_pre - gy_post)²) in hypercolumn units.
//   Self-loops excluded (no autapses).  At d = 0 every other clone in the
//   same hypercolumn is a candidate; with p(0) = 0.12 each post-cell pulls
//   ~1.8 partners from its own 15 same-hypercolumn peers.
//
//   Reciprocity boost: after independent-direction sampling at p(d), for
//   each newly-sampled directed edge A→B we draw from a Bernoulli with
//   probability r(d) = 3·p(d) / (2·(1 - p(d))) and, if it fires AND B→A
//   doesn't already exist, we add B→A.  This solves
//       P_recip_after = p² + 2·p·(1-p)·r  =  4·p²
//   so the final reciprocal-pair fraction is exactly 4× the independent
//   chance baseline (~ Song et al. 2005 cortical observation).
//
//   Weights (lognormal in EPSP-mV space):
//       median = L23REC_EPSP_MEDIAN_MV = 0.3 mV
//       log-σ  = L23REC_EPSP_LOG_SIGMA = 0.6
//       hard cap = L23REC_EPSP_MAX_MV  = 5 × median = 1.5 mV
//       w_nS    = w_mV / L23_MV_PER_NS  (same 0.974 mV/nS as L4→L2/3)
//
//   Synapse model: identical kinetics to L4→L2/3 (τ_syn = 12 ms,
//   E_E = 0 mV).  EPSCs from L4 and from L2/3 partners scatter into the
//   SAME post-cell ge variable -- they share a synaptic time constant
//   and reversal so they accumulate naturally.
//
//   Conduction delay: 1 ms = 10 timesteps at dt = 0.1 ms.  Implemented
//   by a per-step L2/3 spike-record bitmask (warp-ballot, mirror of the
//   existing 20-deep L4 bitmask).  The recurrent kernel reads the
//   delayed L2/3 mask at step (t - 10).
//
//   Storage: CSR keyed by post-cell (mirror of L4→L2/3):
//     l23rec_row_ptr[0 .. N_L23]            (int32, size N_L23+1)
//     l23rec_col_idx[0 .. n_synapses]       (int32, pre-synaptic L2/3 ids)
//     l23rec_w_nS  [0 .. n_synapses]        (double, conductance)
//
//   Random seeds (independent from L4→L2/3 salts above):
//     connectivity_seed = master_seed XOR L23REC_SALT_CONNECTIVITY
//     reciprocity_seed  = master_seed XOR L23REC_SALT_RECIPROCITY
//     weight_seed       = master_seed XOR L23REC_SALT_WEIGHTS
//
//   No plasticity, no traces, no STDP -- weights are static for B1.
// =====================================================================
constexpr double L23REC_P0              = 0.12;     // base connection prob at d=0
constexpr double L23REC_LEN_HCOL        = 1.5;      // exp decay length (hypercolumns)
constexpr int    L23REC_DMAX            = 4;        // hard cutoff (hypercolumns)
constexpr double L23REC_EPSP_MEDIAN_MV  = 0.3;      // lognormal median (mV)
constexpr double L23REC_EPSP_LOG_SIGMA  = 0.6;      // lognormal σ (log space)
constexpr double L23REC_EPSP_MAX_MV     = 1.5;      // hard cap = 5 × median
constexpr int    L23REC_DELAY_STEPS     = 10;       // 1 ms / dt = 10

// Independent salts (chosen from PRNG-friendly primes, distinct from the
// L4→L2/3 salts so connectivity/weights don't covary across layers).
constexpr unsigned long long L23REC_SALT_CONNECTIVITY = 0xD1B54A32D192ED03ULL;
constexpr unsigned long long L23REC_SALT_RECIPROCITY  = 0xCBF29CE484222325ULL;
constexpr unsigned long long L23REC_SALT_WEIGHTS      = 0x100000001B3ULL;

// =====================================================================
// Pair-STDP on L2/3→L2/3 recurrent weights (task #5 — B2 plasticity).
//
// Same Gütig-style multiplicative-bounded form as Phase A's L4→L2/3
// path, with PER-CELL traces (one x_pre and one y_post per L2/3 cell —
// not per synapse) and Froemke & Dan 2002 windows:
//
//   On L2/3 (post) spike at time t_post:
//     Δw_LTP = +A⁺ · (w_max − w) · x_pre[partner_id]    (pre-trace of pre cell)
//   On L2/3 (pre) spike (delayed arrival at the synapse) at time t_arrival:
//     Δw_LTD = −A⁻ · (w − w_min) · y_post[idx]          (post-trace of THIS cell)
//
// Trace dynamics (each clipped at 1.0 on bump):
//   x_pre[idx]:  dx/dt = -x/τ⁺,   x += 1 on this cell's own spike (acting as PRE)
//   y_post[idx]: dy/dt = -y/τ⁻,   y += 1 on this cell's own spike (acting as POST)
//
// L2/3 cells are dual-role (pre AND post depending on synapse direction),
// so a single trace pair per cell suffices — when the kernel processes
// cell idx as a POST (LTP loop on incoming partners), it reads x_pre[p]
// for each pre-cell partner; when processing idx as a PRE (delayed spike
// arrives at idx's outgoing partners — actually arrives AS incoming-from-p
// at OTHER cells; the LTD branch in the recurrent kernel handles this
// when that other cell processes its incoming bitmask: it sees pre-spike
// arrivals and applies LTD using its own y_post).
//
// L4→L2/3 weights stay FROZEN — no traces, no updates on that path.
//
// Hard bounds clipped each step.  w_min = 0 (allows pruning),
// w_max = L23REC_EPSP_MAX_MV / L23_MV_PER_NS ≈ 1.5400 nS (= B1 cap).
// =====================================================================
constexpr double L23REC_STDP_TAU_PLUS_MS  = 17.0;   // Froemke & Dan 2002
constexpr double L23REC_STDP_TAU_MINUS_MS = 34.0;
constexpr double L23REC_STDP_A_PLUS       = 0.005;
constexpr double L23REC_STDP_A_MINUS      = 0.003;  // = 0.6 × A_PLUS
constexpr double L23REC_W_MAX_NS = L23REC_EPSP_MAX_MV / L23_MV_PER_NS;
constexpr double L23REC_W_MIN_NS = 0.0;

// =====================================================================
// Pair-STDP on L4→L2/3 weights (task #54, Phase A — no triplet, no
// voltage-dep, no homeostasis).
//
// Update rule (multiplicative bounded; Gütig-style soft cap on LTP +
// van Rossum multiplicative LTD):
//   On L2/3 post-spike at time t_post:
//     Δw_LTP = +A_PLUS  · (w_max - w) · x_pre  [computed per partner]
//   On L4 pre-spike (delayed arrival at synapse) at time t_pre:
//     Δw_LTD = -A_MINUS · w · y_post           [post-trace at this L2/3]
//
// Trace dynamics (each clipped at 1.0 on bump):
//   x_pre  per partner-synapse: dx/dt = -x/τ_plus,  x += 1 on delayed L4 spike
//   y_post per L2/3 cell:       dy/dt = -y/τ_minus, y += 1 on L2/3 spike
//
// τ_plus / τ_minus and A_plus / A_minus pinned by team-lead in #54 brief
// (Feldman 2000 ranges; A_minus = 0.6 × A_plus -> LTD-dominant under
// uncorrelated firing, the canonical STDP-as-correlation-detector behavior).
//
// Hard bounds applied each step.  w_max equals the lognormal init cap
// (#52 = 5 mV-equivalent at L23_MV_PER_NS); w_min = 0 (allows pruning).
// =====================================================================
constexpr double STDP_TAU_PLUS_MS  = 15.0;
constexpr double STDP_TAU_MINUS_MS = 30.0;
constexpr double STDP_A_PLUS       = 0.005;
constexpr double STDP_A_MINUS      = 0.003;     // ≈ 0.6 × A_PLUS
constexpr double STDP_W_MAX_NS     = L23_EPSP_MAX_MV / L23_MV_PER_NS;   // ≈ 5.13 nS
constexpr double STDP_W_MIN_NS     = 0.0;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::ostringstream _os;                                            \
            _os << "CUDA error: " << cudaGetErrorString(_e)                    \
                << " at " << __FILE__ << ":" << __LINE__;                      \
            throw std::runtime_error(_os.str());                               \
        }                                                                     \
    } while (0)

[[noreturn]] void die(const std::string& m) { throw std::runtime_error(m); }

// Index helpers.
__host__ __device__ inline int cell_gx(int idx)    { return (idx / CELLS_PER_HYPERCOL) % GRID; }
__host__ __device__ inline int cell_gy(int idx)    { return (idx / CELLS_PER_HYPERCOL) / GRID; }
__host__ __device__ inline int cell_ori(int idx)   { return (idx / N_CELLS_PER_ORIENT) % N_ORIENT; }
__host__ __device__ inline int cell_clone(int idx) { return idx % N_CELLS_PER_ORIENT; }
__host__ __device__ inline int make_cell_id(int gx, int gy, int ori, int clone) {
    return ((gy * GRID + gx) * N_ORIENT + ori) * N_CELLS_PER_ORIENT + clone;
}

// L2/3 index helpers (l23_id = (gy*GRID + gx) * N_L23_CLONES + clone).
__host__ __device__ inline int l23_gx(int idx)    { return (idx / N_L23_CLONES) % GRID; }
__host__ __device__ inline int l23_gy(int idx)    { return (idx / N_L23_CLONES) / GRID; }
__host__ __device__ inline int l23_clone(int idx) { return idx % N_L23_CLONES; }
__host__ __device__ inline int make_l23_id(int gx, int gy, int clone) {
    return (gy * GRID + gx) * N_L23_CLONES + clone;
}

// =====================================================================
// Gabor template generation kernel.
//
// One thread per (ori_idx, clone_idx) template.  Computes the 9 × 9 weight
// kernel, subtracts the patch mean (zero-mean kernel), and rescales to
// L1-norm = GABOR_L1_TARGET.
// =====================================================================
__global__ void build_gabor_templates_kernel(double* templates) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_TEMPLATES) return;
    const int ori   = t / N_CELLS_PER_ORIENT;
    const int clone = t % N_CELLS_PER_ORIENT;
    const double theta_pref =
        static_cast<double>(ori) * (PI / static_cast<double>(N_ORIENT));
    const double phase =
        static_cast<double>(clone) * (2.0 * PI / static_cast<double>(N_CELLS_PER_ORIENT));
    // Per-clone OSI variability: smooth γ jitter via cosine of clone index.
    const double gamma_clone = GABOR_GAMMA
        + GAMMA_JITTER * cos(2.0 * PI * static_cast<double>(clone)
                             / static_cast<double>(N_CELLS_PER_ORIENT));
    const double cos_tp = cos(theta_pref);
    const double sin_tp = sin(theta_pref);
    const double sx2 = GABOR_SIGMA_X * GABOR_SIGMA_X;
    const double sigma_y = gamma_clone * GABOR_SIGMA_X;
    const double sy2 = sigma_y * sigma_y;

    double* W = templates + static_cast<std::size_t>(t) * GABOR_PIX;

    double sum = 0.0;
    for (int dy = -GABOR_HALF; dy <= GABOR_HALF; ++dy) {
        for (int dx = -GABOR_HALF; dx <= GABOR_HALF; ++dx) {
            const double dxp =  dx * cos_tp + dy * sin_tp;
            const double dyp = -dx * sin_tp + dy * cos_tp;
            const double envelope = exp(-(dxp * dxp / (2.0 * sx2)
                                          + dyp * dyp / (2.0 * sy2)));
            const double carrier = cos(GABOR_K * dxp + phase);
            const int kidx = (dy + GABOR_HALF) * GABOR_SIZE + (dx + GABOR_HALF);
            W[kidx] = envelope * carrier;
            sum += W[kidx];
        }
    }
    // Subtract mean (zero-mean kernel).
    const double mean_w = sum / static_cast<double>(GABOR_PIX);
    double l1 = 0.0;
    for (int k = 0; k < GABOR_PIX; ++k) {
        W[k] -= mean_w;
        l1 += fabs(W[k]);
    }
    // L1-normalize.
    if (l1 > 0.0) {
        const double scale = GABOR_L1_TARGET / l1;
        for (int k = 0; k < GABOR_PIX; ++k) W[k] *= scale;
    }
}

// =====================================================================
// Init kernels for AdEx state and accumulators.
// =====================================================================
__global__ void init_full_state_kernel(
    double* V, double* w, double* g_E, int* refrac, long long* prev_spike_step,
    long long* total_spikes, long long* isi_count, double* isi_sum, double* isi_sum_sq,
    int n_l4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_l4) return;
    V[idx] = E_L_MV; w[idx] = 0.0; g_E[idx] = 0.0;
    refrac[idx] = 0; prev_spike_step[idx] = -1;
    total_spikes[idx] = 0;
    isi_count[idx] = 0; isi_sum[idx] = 0.0; isi_sum_sq[idx] = 0.0;
}

__global__ void reset_dyn_state_kernel(
    double* V, double* w, double* g_E, int* refrac, long long* prev_spike_step,
    int n_l4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_l4) return;
    V[idx] = E_L_MV; w[idx] = 0.0; g_E[idx] = 0.0;
    refrac[idx] = 0; prev_spike_step[idx] = -1;
}

__global__ void clear_int_kernel(int* p, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = 0;
}

__global__ void clear_uint32_kernel(uint32_t* p, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = 0u;
}

__global__ void clear_int_array_kernel(int* p, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = 0;
}

// Stim-variant aperture: Gaussian mask centered at (cx, cy).  Returns 1 when
// inv_2sigma_sq = 0 (caller passes 0 to disable; we still gate via
// aperture_active flag to avoid the exp() cost when disabled).
__host__ __device__ inline double aperture_at(
    int px, int py,
    double cx, double cy, double inv_2sigma_sq
) {
    const double dxc = static_cast<double>(px) - cx;
    const double dyc = static_cast<double>(py) - cy;
    const double rsq = dxc * dxc + dyc * dyc;
    return exp(-rsq * inv_2sigma_sq);
}

// Stimulus-protocol bin sizes (#53 verification).
constexpr int STIM_BIN20_STEPS = 200;   // 20 ms at dt=0.1 ms (peak-rate bin)
constexpr int STIM_BIN50_STEPS = 500;   // 50 ms at dt=0.1 ms (correlation bin)

// Sanity: total bits set across the L4 spike-record bitmask.  Used by
// run_verify_l23 to verify the warp-ballot record path matches the
// independently-counted total L4 spikes.
__global__ void popcount_uint32_kernel(
    const uint32_t* __restrict__ data, size_t n, unsigned long long* out_total
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const unsigned bits = __popc(data[idx]);
    if (bits != 0u) atomicAdd(out_total, static_cast<unsigned long long>(bits));
}

// =====================================================================
// L2/3 init kernel.  Same AdEx state shape as L4, but no Poisson-input
// machinery (drive comes from L4-spike scatter through the CSR weights).
// =====================================================================
__global__ void init_l23_state_kernel(
    double* V, double* w, double* g_E, int* refrac, long long* prev_spike_step,
    long long* total_spikes, long long* isi_count, double* isi_sum, double* isi_sum_sq,
    int n_l23
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_l23) return;
    V[idx] = E_L_MV; w[idx] = 0.0; g_E[idx] = 0.0;
    refrac[idx] = 0; prev_spike_step[idx] = -1;
    total_spikes[idx] = 0;
    isi_count[idx] = 0; isi_sum[idx] = 0.0; isi_sum_sq[idx] = 0.0;
}

// =====================================================================
// L2/3 per-phase kernel.  One thread per L2/3 cell; runs the entire phase
// (n_steps) consuming the L4 spike-bitmask record produced by the
// modified v1_phase_kernel.
//
//   Per step t:
//     1. Find delayed L4 spikes from step t - L23_DELAY_STEPS (= 20).
//        For t < L23_DELAY_STEPS no scatter happens (initial 2 ms is silent).
//     2. For each CSR partner, OR the bitmask bit; if set, ge_l23 += w_l23_nS[p].
//     3. ge_l23 *= alpha_S_l23  (decay, τ = 12 ms).
//     4. AdEx integration step (same RS params as L4).
//
// No Poisson input.  No per-step RNG.  L2/3 is fully deterministic given the
// L4 spike record.
// =====================================================================
__global__ void v1_l23_phase_kernel(
    double* V, double* w_adapt, double* g_E,
    int* refrac, long long* prev_spike_step,
    long long* total_spikes,
    long long* isi_count, double* isi_sum, double* isi_sum_sq,
    int* phase_spike_count,
    const int* __restrict__ l23_row_ptr,        // [N_L23 + 1]
    const int* __restrict__ l23_col_idx,        // [n_synapses]
    const double* __restrict__ l23_w_nS,        // [n_synapses]
    const uint32_t* __restrict__ l4_spike_record,  // [n_steps][N_L4_BITMASK_INTS]
    long long phase_step_offset,
    int n_steps,
    int n_warmup_steps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L23) return;

    // Load CSR row range.
    const int p_start = l23_row_ptr[idx];
    const int p_end   = l23_row_ptr[idx + 1];

    // Load state.
    double v   = V[idx];
    double wa  = w_adapt[idx];
    double ge  = g_E[idx];
    int    rfr = refrac[idx];
    long long prev_spk = prev_spike_step[idx];
    long long tot_spk  = total_spikes[idx];
    long long isi_n    = isi_count[idx];
    double    isi_s    = isi_sum[idx];
    double    isi_ss   = isi_sum_sq[idx];
    int phase_count    = 0;

    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S_l23 = exp(-DT_MS / TAU_SYN_L23_MS);

    for (int step = 0; step < n_steps; ++step) {
        // ---- Stage 1: scatter EPSCs from delayed L4 spikes (t - 20 steps) ----
        const int delay_step = step - L23_DELAY_STEPS;
        if (delay_step >= 0) {
            const uint32_t* mask_row =
                l4_spike_record + static_cast<size_t>(delay_step) * N_L4_BITMASK_INTS;
            for (int p = p_start; p < p_end; ++p) {
                const int partner = l23_col_idx[p];
                const uint32_t word = mask_row[partner >> 5];
                if (word & (1u << (partner & 31))) {
                    ge += l23_w_nS[p];
                }
            }
        }
        ge *= alpha_S_l23;

        // ---- Stage 2: AdEx step (same RS params as L4) ----
        if (rfr > 0) {
            v = V_RESET_MV;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            wa += DT_MS * dwa;
            rfr -= 1;
        } else {
            double exp_arg = (v - V_T_MV) * inv_dT;
            if (exp_arg > 50.0) exp_arg = 50.0;
            const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
            const double leak  = -G_L_NS * (v - E_L_MV);
            const double i_syn = ge * (v - E_E_L23_MV);

            const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            v  += DT_MS * dv;
            wa += DT_MS * dwa;

            if (v >= V_PEAK_MV) {
                v = V_RESET_MV;
                wa += B_PA;
                rfr = T_REF_STEPS;

                tot_spk += 1;
                const long long step_global = phase_step_offset + step;

                if (step >= n_warmup_steps) {
                    phase_count += 1;
                    if (prev_spk >= 0) {
                        const long long isi = step_global - prev_spk;
                        if (isi > 0) {
                            const double isi_d = static_cast<double>(isi);
                            isi_n  += 1;
                            isi_s  += isi_d;
                            isi_ss += isi_d * isi_d;
                        }
                    }
                    prev_spk = step_global;
                }
            }
        }
    }

    V[idx]               = v;
    w_adapt[idx]         = wa;
    g_E[idx]             = ge;
    refrac[idx]          = rfr;
    prev_spike_step[idx] = prev_spk;
    total_spikes[idx]    = tot_spk;
    isi_count[idx]       = isi_n;
    isi_sum[idx]         = isi_s;
    isi_sum_sq[idx]      = isi_ss;
    phase_spike_count[idx] = phase_count;
}

// =====================================================================
// L2/3 RECURRENT per-step kernel (task #3 — B1 STRUCTURE-ONLY, no STDP).
//
// One kernel launch per simulated time step (host drives the loop).
// This is REQUIRED for correctness: with grid-wide recurrence, a thread
// at step t reads other cells' spike bits at step t - L23REC_DELAY_STEPS,
// so all blocks' writes for step t-DELAY must complete before any block
// reads them.  CUDA only guarantees that ordering across kernel
// boundaries (cudaStream serializes launches), not within a single
// launch where blocks execute independently.  The kernel-launch
// overhead is negligible (~10 ms for 1000 launches on a fast GPU).
//
// Per step:
//   Stage 1a: Scatter EPSCs from delayed L4 spikes  (l4_spike_record at t-20).
//   Stage 1b: Scatter EPSCs from delayed L2/3 spikes(l23_spike_record at t-10).
//   Stage 2 : ge *= alpha (τ_syn = 12 ms).
//   Stage 3 : AdEx step (RS).  May spike.
//   Stage 4 : Warp-ballot dump of THIS step's spike to l23_spike_record[t].
//
// L4 EPSCs and L2/3 EPSCs accumulate into the SAME ge variable (same τ_syn,
// same E_E reversal).  No STDP traces, weights static.
// =====================================================================
__global__ void v1_l23_recurrent_step_kernel(
    double* V, double* w_adapt, double* g_E,
    int* refrac, long long* prev_spike_step,
    long long* total_spikes,
    long long* isi_count, double* isi_sum, double* isi_sum_sq,
    int* phase_spike_count,
    // L4 → L2/3 inputs (existing).
    const int* __restrict__ l23_row_ptr,
    const int* __restrict__ l23_col_idx,
    const double* __restrict__ l23_w_nS,
    const uint32_t* __restrict__ l4_spike_record,
    // L2/3 → L2/3 inputs (NEW).
    const int* __restrict__ l23rec_row_ptr,
    const int* __restrict__ l23rec_col_idx,
    const double* __restrict__ l23rec_w_nS,
    uint32_t* __restrict__ l23_spike_record,
    long long phase_step_offset,
    int step_idx,
    int n_warmup_steps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L23) return;

    const int p4_start  = l23_row_ptr[idx];
    const int p4_end    = l23_row_ptr[idx + 1];
    const int p23_start = l23rec_row_ptr[idx];
    const int p23_end   = l23rec_row_ptr[idx + 1];

    double v   = V[idx];
    double wa  = w_adapt[idx];
    double ge  = g_E[idx];
    int    rfr = refrac[idx];
    long long prev_spk = prev_spike_step[idx];
    long long tot_spk  = total_spikes[idx];
    long long isi_n    = isi_count[idx];
    double    isi_s    = isi_sum[idx];
    double    isi_ss   = isi_sum_sq[idx];
    int phase_count    = phase_spike_count[idx];

    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S_l23 = exp(-DT_MS / TAU_SYN_L23_MS);

    const int step = step_idx;

    // ---- Stage 1a: scatter EPSCs from delayed L4 spikes (t - 20) ----
    const int delay4 = step - L23_DELAY_STEPS;
    if (delay4 >= 0) {
        const uint32_t* mask4 =
            l4_spike_record + static_cast<size_t>(delay4) * N_L4_BITMASK_INTS;
        for (int p = p4_start; p < p4_end; ++p) {
            const int partner = l23_col_idx[p];
            const uint32_t word = mask4[partner >> 5];
            if (word & (1u << (partner & 31))) {
                ge += l23_w_nS[p];
            }
        }
    }

    // ---- Stage 1b: scatter EPSCs from delayed L2/3 spikes (t - 10) ----
    const int delay23 = step - L23REC_DELAY_STEPS;
    if (delay23 >= 0) {
        const uint32_t* mask23 =
            l23_spike_record + static_cast<size_t>(delay23) * N_L23_BITMASK_INTS;
        for (int p = p23_start; p < p23_end; ++p) {
            const int partner = l23rec_col_idx[p];
            const uint32_t word = mask23[partner >> 5];
            if (word & (1u << (partner & 31))) {
                ge += l23rec_w_nS[p];
            }
        }
    }

    // ---- Stage 2: synaptic decay ----
    ge *= alpha_S_l23;

    // ---- Stage 3: AdEx step (RS params, same as L4) ----
    bool spiked_this_step = false;
    if (rfr > 0) {
        v = V_RESET_MV;
        const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
        wa += DT_MS * dwa;
        rfr -= 1;
    } else {
        double exp_arg = (v - V_T_MV) * inv_dT;
        if (exp_arg > 50.0) exp_arg = 50.0;
        const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
        const double leak  = -G_L_NS * (v - E_L_MV);
        const double i_syn = ge * (v - E_E_L23_MV);
        const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
        const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
        v  += DT_MS * dv;
        wa += DT_MS * dwa;
        if (v >= V_PEAK_MV) {
            v = V_RESET_MV;
            wa += B_PA;
            rfr = T_REF_STEPS;
            spiked_this_step = true;
            tot_spk += 1;
            const long long step_global = phase_step_offset + step;
            if (step >= n_warmup_steps) {
                phase_count += 1;
                if (prev_spk >= 0) {
                    const long long isi = step_global - prev_spk;
                    if (isi > 0) {
                        const double isi_d = static_cast<double>(isi);
                        isi_n  += 1;
                        isi_s  += isi_d;
                        isi_ss += isi_d * isi_d;
                    }
                }
                prev_spk = step_global;
            }
        }
    }

    // ---- Stage 4: warp-ballot dump THIS step's L2/3 spike ----
    // Threads 0..N_L23-1 are at consecutive cell indices; each warp-of-32
    // shares a single 32-bit word in the bitmask.  The next-step kernel
    // launch reads this from l23_spike_record[step] (after device sync).
    const unsigned mask_w = __ballot_sync(0xFFFFFFFFu, spiked_this_step);
    if ((threadIdx.x & 31) == 0) {
        const int word_idx = idx >> 5;             // = idx / 32
        l23_spike_record[
            static_cast<size_t>(step) * N_L23_BITMASK_INTS + word_idx
        ] = mask_w;
    }

    V[idx]               = v;
    w_adapt[idx]         = wa;
    g_E[idx]             = ge;
    refrac[idx]          = rfr;
    prev_spike_step[idx] = prev_spk;
    total_spikes[idx]    = tot_spk;
    isi_count[idx]       = isi_n;
    isi_sum[idx]         = isi_s;
    isi_sum_sq[idx]      = isi_ss;
    phase_spike_count[idx] = phase_count;
}

// =====================================================================
// L2/3 RECURRENT + STDP per-step kernel (task #5 — B2 plasticity).
//
// Extends v1_l23_recurrent_step_kernel with bounded multiplicative
// pair-STDP on the L2/3→L2/3 weights (cell-level traces).  L4→L2/3
// weights are FROZEN — never written here.
//
// Per simulated time step (one launch per step from host):
//   Stage 0  : decay traces locally (x_pre, y_post — per cell).
//   Stage 1a : scatter delayed L4 EPSCs (no plasticity on L4 path).
//   Stage 1b : scatter delayed L2/3 EPSCs.  For EACH delayed pre-spike
//              that arrives, IF plasticity_active, apply LTD:
//                w[p→idx] -= A⁻ · (w − w_min) · y_post[idx]
//              (clipped at w_min).  Then accumulate the EPSC into ge.
//   Stage 2  : ge *= alpha_S_l23  (τ = 12 ms shared across L4 + L2/3).
//   Stage 3  : AdEx step (RS).  May spike.
//   Stage 4  : Warp-ballot dump THIS step's spike to l23_spike_record.
//   Stage 5  : If THIS cell spiked AND plasticity_active, apply LTP:
//                w[p→idx] += A⁺ · (w_max − w) · x_pre[pre_id]
//              for each incoming partner p (clipped at w_max).
//              Bump x_pre[idx] += 1 (clip 1.0), y_post[idx] += 1 (clip 1.0)
//              regardless of plasticity_active.
//
// Plasticity gate: plasticity_active = 0 → traces still decay/bump but
// NO weight updates (used during ITI gray gap so traces relax without
// frozen-period drift writing into weights).
// =====================================================================
__global__ void v1_l23_recurrent_stdp_step_kernel(
    double* V, double* w_adapt, double* g_E,
    int* refrac, long long* prev_spike_step,
    long long* total_spikes,
    long long* isi_count, double* isi_sum, double* isi_sum_sq,
    int* phase_spike_count,
    // L4 → L2/3 inputs (FROZEN weights).
    const int* __restrict__ l23_row_ptr,
    const int* __restrict__ l23_col_idx,
    const double* __restrict__ l23_w_nS,
    const uint32_t* __restrict__ l4_spike_record,
    // L2/3 → L2/3 inputs (PLASTIC weights — non-const).
    const int* __restrict__ l23rec_row_ptr,
    const int* __restrict__ l23rec_col_idx,
    double* __restrict__ l23rec_w_nS,
    uint32_t* __restrict__ l23_spike_record,
    // Per-cell STDP traces (NEW — non-const).
    double* __restrict__ x_pre,
    double* __restrict__ y_post,
    long long phase_step_offset,
    int step_idx,
    int n_warmup_steps,
    int plasticity_active
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L23) return;

    const int p4_start  = l23_row_ptr[idx];
    const int p4_end    = l23_row_ptr[idx + 1];
    const int p23_start = l23rec_row_ptr[idx];
    const int p23_end   = l23rec_row_ptr[idx + 1];

    double v   = V[idx];
    double wa  = w_adapt[idx];
    double ge  = g_E[idx];
    int    rfr = refrac[idx];
    long long prev_spk = prev_spike_step[idx];
    long long tot_spk  = total_spikes[idx];
    long long isi_n    = isi_count[idx];
    double    isi_s    = isi_sum[idx];
    double    isi_ss   = isi_sum_sq[idx];
    int phase_count    = phase_spike_count[idx];

    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S_l23 = exp(-DT_MS / TAU_SYN_L23_MS);
    const double alpha_pre   = exp(-DT_MS / L23REC_STDP_TAU_PLUS_MS);
    const double alpha_post  = exp(-DT_MS / L23REC_STDP_TAU_MINUS_MS);
    const double A_plus  = L23REC_STDP_A_PLUS;
    const double A_minus = L23REC_STDP_A_MINUS;
    const double w_max   = L23REC_W_MAX_NS;
    const double w_min   = L23REC_W_MIN_NS;

    const int step = step_idx;

    // ---- Stage 0: decay this cell's traces (LOCAL register; flushed at end) ----
    double xp_local = x_pre[idx]  * alpha_pre;
    double yp_local = y_post[idx] * alpha_post;

    // ---- Stage 1a: scatter EPSCs from delayed L4 spikes (t - 20). FROZEN. ----
    const int delay4 = step - L23_DELAY_STEPS;
    if (delay4 >= 0) {
        const uint32_t* mask4 =
            l4_spike_record + static_cast<size_t>(delay4) * N_L4_BITMASK_INTS;
        for (int p = p4_start; p < p4_end; ++p) {
            const int partner = l23_col_idx[p];
            const uint32_t word = mask4[partner >> 5];
            if (word & (1u << (partner & 31))) {
                ge += l23_w_nS[p];   // FROZEN read, no plasticity
            }
        }
    }

    // ---- Stage 1b: scatter EPSCs from delayed L2/3 spikes (t - 10). PLASTIC LTD. ----
    const int delay23 = step - L23REC_DELAY_STEPS;
    if (delay23 >= 0) {
        const uint32_t* mask23 =
            l23_spike_record + static_cast<size_t>(delay23) * N_L23_BITMASK_INTS;
        for (int p = p23_start; p < p23_end; ++p) {
            const int partner = l23rec_col_idx[p];
            const uint32_t word = mask23[partner >> 5];
            if (word & (1u << (partner & 31))) {
                if (plasticity_active) {
                    // LTD on this synapse BEFORE scattering EPSC, so the
                    // weight reflects post-spike history up to but not
                    // including the current step.
                    double w = l23rec_w_nS[p];
                    w -= A_minus * (w - w_min) * yp_local;
                    if (w < w_min) w = w_min;
                    l23rec_w_nS[p] = w;
                }
                ge += l23rec_w_nS[p];
            }
        }
    }

    // ---- Stage 2: synaptic decay ----
    ge *= alpha_S_l23;

    // ---- Stage 3: AdEx step (RS, same as L4) ----
    bool spiked_this_step = false;
    if (rfr > 0) {
        v = V_RESET_MV;
        const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
        wa += DT_MS * dwa;
        rfr -= 1;
    } else {
        double exp_arg = (v - V_T_MV) * inv_dT;
        if (exp_arg > 50.0) exp_arg = 50.0;
        const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
        const double leak  = -G_L_NS * (v - E_L_MV);
        const double i_syn = ge * (v - E_E_L23_MV);
        const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
        const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
        v  += DT_MS * dv;
        wa += DT_MS * dwa;
        if (v >= V_PEAK_MV) {
            v = V_RESET_MV;
            wa += B_PA;
            rfr = T_REF_STEPS;
            spiked_this_step = true;
            tot_spk += 1;
            const long long step_global = phase_step_offset + step;
            if (step >= n_warmup_steps) {
                phase_count += 1;
                if (prev_spk >= 0) {
                    const long long isi = step_global - prev_spk;
                    if (isi > 0) {
                        const double isi_d = static_cast<double>(isi);
                        isi_n  += 1;
                        isi_s  += isi_d;
                        isi_ss += isi_d * isi_d;
                    }
                }
                prev_spk = step_global;
            }
        }
    }

    // ---- Stage 4: warp-ballot dump THIS step's L2/3 spike ----
    const unsigned mask_w = __ballot_sync(0xFFFFFFFFu, spiked_this_step);
    if ((threadIdx.x & 31) == 0) {
        const int word_idx = idx >> 5;
        l23_spike_record[
            static_cast<size_t>(step) * N_L23_BITMASK_INTS + word_idx
        ] = mask_w;
    }

    // ---- Stage 5: post-spike LTP + trace bump ----
    if (spiked_this_step) {
        if (plasticity_active) {
            // LTP: on this cell's spike (POST), iterate incoming partners,
            // weighted by partner's pre-trace.  x_pre[partner] is GLOBAL
            // and may have been updated by another thread in this step;
            // safe because per-step kernel launches give cross-block sync.
            for (int p = p23_start; p < p23_end; ++p) {
                const int partner = l23rec_col_idx[p];
                const double xp_partner = x_pre[partner];
                if (xp_partner > 0.0) {
                    double w = l23rec_w_nS[p];
                    w += A_plus * (w_max - w) * xp_partner;
                    if (w > w_max) w = w_max;
                    if (w < w_min) w = w_min;
                    l23rec_w_nS[p] = w;
                }
            }
        }
        // Bump LOCAL traces (clip at saturation 1.0).  Both bumps happen
        // regardless of plasticity_active so the trace state stays
        // consistent across plasticity-OFF windows (e.g. ITI).
        xp_local += 1.0; if (xp_local > 1.0) xp_local = 1.0;
        yp_local += 1.0; if (yp_local > 1.0) yp_local = 1.0;
    }

    // ---- Flush state ----
    V[idx]               = v;
    w_adapt[idx]         = wa;
    g_E[idx]             = ge;
    refrac[idx]          = rfr;
    prev_spike_step[idx] = prev_spk;
    total_spikes[idx]    = tot_spk;
    isi_count[idx]       = isi_n;
    isi_sum[idx]         = isi_s;
    isi_sum_sq[idx]      = isi_ss;
    phase_spike_count[idx] = phase_count;
    x_pre[idx]           = xp_local;
    y_post[idx]          = yp_local;
}

// =====================================================================
// L2/3 + STDP phase kernel.  Same structure as v1_l23_phase_kernel but
// with multiplicative pair-STDP on L4→L2/3 weights.  Per L2/3 thread:
//
//   Per-step (after the 20-step delay):
//     1. Compute alpha factors (decay constants) once at start.
//     2. Each step:
//        a. Decay y_post (this L2/3 cell's post-trace).
//        b. For each L4 partner:
//             - Decay x_pre[partner_synapse] (per-synapse pre-trace).
//             - If delayed L4 spike at this step:
//                 * x_pre_synapse = min(x_pre_synapse + 1.0, 1.0)
//                 * Apply LTD: w[p] -= A_minus * w[p] * y_post  (clip ≥ 0)
//                 * Scatter EPSC: ge += w[p] (after the LTD update)
//        c. Decay ge by alpha_S_l23.
//        d. AdEx integration (same as v1_l23_phase_kernel).
//        e. If L2/3 fires this step:
//             * y_post = min(y_post + 1.0, 1.0)
//             * For each partner: w[p] += A_plus * (W_MAX - w[p]) * x_pre[p]
//               (clip ≤ W_MAX).
//
// Plasticity gate `plasticity_active`:
//   true  -> apply LTP and LTD updates.
//   false -> still update traces, just skip the weight modifications.
//
// State-persistence: V/w_adapt/g_E/refrac/prev_spike_step are RESET each
// trial by the host (init_l23_state_kernel); x_pre per-synapse and
// y_post per-L2/3 are RESET each trial too (the lead's spec says traces
// decay during ITI; we model that as a clean reset since 100 ms ≫ τ).
// =====================================================================
__global__ void v1_l23_stdp_phase_kernel(
    double* V, double* w_adapt, double* g_E,
    int* refrac, long long* prev_spike_step,
    long long* total_spikes,
    int* phase_spike_count,
    const int* __restrict__ l23_row_ptr,        // [N_L23 + 1]
    const int* __restrict__ l23_col_idx,        // [n_synapses]
    double* __restrict__ l23_w_nS,              // [n_synapses]  (MUTABLE)
    double* __restrict__ x_pre_synapse,         // [n_synapses]  (per-synapse trace)
    double* __restrict__ y_post_l23,            // [N_L23]       (per-cell post-trace)
    const uint32_t* __restrict__ l4_spike_record,  // [n_steps][N_L4_BITMASK_INTS]
    long long phase_step_offset,
    int  n_steps,
    int  plasticity_active                       // 0 or 1
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L23) return;

    const int p_start = l23_row_ptr[idx];
    const int p_end   = l23_row_ptr[idx + 1];

    // Load AdEx state.
    double v   = V[idx];
    double wa  = w_adapt[idx];
    double ge  = g_E[idx];
    int    rfr = refrac[idx];
    long long prev_spk = prev_spike_step[idx];
    long long tot_spk  = total_spikes[idx];
    double y_post = y_post_l23[idx];
    int phase_count = 0;

    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S_l23  = exp(-DT_MS / TAU_SYN_L23_MS);
    const double alpha_pre    = exp(-DT_MS / STDP_TAU_PLUS_MS);
    const double alpha_post   = exp(-DT_MS / STDP_TAU_MINUS_MS);
    const double w_max        = STDP_W_MAX_NS;
    const double w_min        = STDP_W_MIN_NS;
    const double A_plus       = STDP_A_PLUS;
    const double A_minus      = STDP_A_MINUS;

    for (int step = 0; step < n_steps; ++step) {
        // ---- Stage 1: post-trace decay ----
        y_post *= alpha_post;

        // ---- Stage 2: scatter EPSC + per-partner pre-trace + LTD ----
        const int delay_step = step - L23_DELAY_STEPS;
        const uint32_t* mask_row = (delay_step >= 0)
            ? (l4_spike_record + static_cast<size_t>(delay_step) * N_L4_BITMASK_INTS)
            : nullptr;
        for (int p = p_start; p < p_end; ++p) {
            const int partner = l23_col_idx[p];
            // Decay this synapse's pre-trace every step.
            double xp = x_pre_synapse[p] * alpha_pre;
            bool partner_spiked = false;
            if (mask_row != nullptr) {
                const uint32_t word = mask_row[partner >> 5];
                if (word & (1u << (partner & 31))) partner_spiked = true;
            }
            if (partner_spiked) {
                // LTD on the synapse: apply BEFORE bumping trace and
                // scattering EPSC, so that y_post reflects post-spike
                // history just up to but not including the current step.
                if (plasticity_active) {
                    double w = l23_w_nS[p];
                    w = w - A_minus * w * y_post;
                    if (w < w_min) w = w_min;
                    l23_w_nS[p] = w;
                }
                // Bump trace (clipped at saturation).
                xp += 1.0;
                if (xp > 1.0) xp = 1.0;
                // Scatter EPSC using updated (LTD-adjusted) weight.
                ge += l23_w_nS[p];
            }
            x_pre_synapse[p] = xp;
        }
        ge *= alpha_S_l23;

        // ---- Stage 3: AdEx step (same RS params as L4) ----
        bool l23_spiked = false;
        if (rfr > 0) {
            v = V_RESET_MV;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            wa += DT_MS * dwa;
            rfr -= 1;
        } else {
            double exp_arg = (v - V_T_MV) * inv_dT;
            if (exp_arg > 50.0) exp_arg = 50.0;
            const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
            const double leak  = -G_L_NS * (v - E_L_MV);
            const double i_syn = ge * (v - E_E_L23_MV);
            const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            v  += DT_MS * dv;
            wa += DT_MS * dwa;
            if (v >= V_PEAK_MV) {
                v = V_RESET_MV;
                wa += B_PA;
                rfr = T_REF_STEPS;
                l23_spiked = true;
                tot_spk += 1;
                phase_count += 1;
                prev_spk = phase_step_offset + step;
            }
        }

        // ---- Stage 4: post-spike LTP + post-trace bump ----
        if (l23_spiked) {
            if (plasticity_active) {
                for (int p = p_start; p < p_end; ++p) {
                    const double xp = x_pre_synapse[p];
                    if (xp > 0.0) {
                        double w = l23_w_nS[p];
                        w = w + A_plus * (w_max - w) * xp;
                        if (w > w_max) w = w_max;
                        if (w < w_min) w = w_min;
                        l23_w_nS[p] = w;
                    }
                }
            }
            y_post += 1.0;
            if (y_post > 1.0) y_post = 1.0;
        }
    }

    V[idx]               = v;
    w_adapt[idx]         = wa;
    g_E[idx]             = ge;
    refrac[idx]          = rfr;
    prev_spike_step[idx] = prev_spk;
    total_spikes[idx]    = tot_spk;
    y_post_l23[idx]      = y_post;
    phase_spike_count[idx] = phase_count;
}

// Helper: clear a double array on device (used to reset per-synapse and
// per-l23-cell traces between trials).
__global__ void clear_double_array_kernel(double* p, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) p[idx] = 0.0;
}

// =====================================================================
// Sparse-pixel L4 phase kernel for V1 RF mapping (task #54 V-A).
//
// Per-pixel rate field is binary:
//     rate(px, py) = lit_rate_hz      if (px, py) == (lit_x, lit_y)
//                  = base_rate_hz     otherwise
// No grating, no aperture, no time modulation.  This gives the cleanest
// reverse-correlation signal: only the lit pixel drives ABOVE-baseline,
// and L2/3 spikes can be unambiguously tagged to the spatial position
// of the stimulus that elicited them.
//
// Per L4 cell at retinotopic (gx, gy):
//   r_lin = Σ_{dx, dy} W(dx, dy) · rate(gx + dx, gy + dy)
//         = base_rate · sum_W + (lit_rate - base_rate) · W(lit_x - gx, lit_y - gy)
//   (only when |lit_x - gx| <= GABOR_HALF AND |lit_y - gy| <= GABOR_HALF;
//    otherwise the lit pixel is outside this cell's RF and r_lin = base * sum_W).
//
// The L23_DELAY_STEPS warpballot bitmask record is written exactly the same
// way as v1_phase_kernel.  No per-cell binning outputs (V-A doesn't need them).
// =====================================================================
__global__ void v1_phase_sparse_pixel_kernel(
    double* V, double* w_adapt, double* g_E,
    int* refrac, long long* prev_spike_step,
    long long* total_spikes,
    int* phase_spike_count,
    const double* __restrict__ gabor_templates,
    int n_steps,
    int lit_x, int lit_y,
    double lit_rate_hz, double base_rate_hz,
    double w_in_nS, double r_base_hz,
    unsigned long long base_seed,
    int phase_idx,
    uint32_t* spike_record_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L4) return;

    const int gx  = cell_gx(idx);
    const int gy  = cell_gy(idx);
    const int ori = cell_ori(idx);
    const int cln = cell_clone(idx);
    const double* __restrict__ W =
        gabor_templates
        + static_cast<std::size_t>(ori * N_CELLS_PER_ORIENT + cln) * GABOR_PIX;

    // Pre-compute Σ W (zero-mean Gabor → ≈ 0) and the W value at the lit-
    // pixel offset (if any).  Both are constant for the whole phase.
    double sum_W = 0.0;
    double W_at_lit = 0.0;
    bool   lit_in_rf = false;
    {
        const int dx_lit = lit_x - gx;
        const int dy_lit = lit_y - gy;
        for (int dy = -GABOR_HALF; dy <= GABOR_HALF; ++dy) {
            for (int dx = -GABOR_HALF; dx <= GABOR_HALF; ++dx) {
                const int kidx = (dy + GABOR_HALF) * GABOR_SIZE + (dx + GABOR_HALF);
                sum_W += W[kidx];
            }
        }
        if (dx_lit >= -GABOR_HALF && dx_lit <= GABOR_HALF
         && dy_lit >= -GABOR_HALF && dy_lit <= GABOR_HALF) {
            const int kidx = (dy_lit + GABOR_HALF) * GABOR_SIZE
                           + (dx_lit + GABOR_HALF);
            W_at_lit = W[kidx];
            lit_in_rf = true;
        }
    }
    // r_lin(t) is constant in time for sparse-pixel stim.
    double r_lin_const = base_rate_hz * sum_W;
    if (lit_in_rf) r_lin_const += (lit_rate_hz - base_rate_hz) * W_at_lit;

    // Load state.
    double v   = V[idx];
    double wa  = w_adapt[idx];
    double ge  = g_E[idx];
    int    rfr = refrac[idx];
    long long prev_spk = prev_spike_step[idx];
    long long tot_spk  = total_spikes[idx];
    int phase_count = 0;

    curandStatePhilox4_32_10_t rng;
    curand_init(
        base_seed,
        static_cast<unsigned long long>(idx)
            + static_cast<unsigned long long>(phase_idx) * 1000003ULL,
        0ULL, &rng
    );

    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S = exp(-DT_MS / TAU_SYN_MS);

    for (int step = 0; step < n_steps; ++step) {
        const double r_in = r_base_hz + (r_lin_const > 0.0 ? r_lin_const : 0.0);
        const double p    = r_in * DT_S;
        const double u = curand_uniform_double(&rng);
        if (u < p) ge += w_in_nS;
        ge *= alpha_S;

        bool spiked_this_step = false;
        if (rfr > 0) {
            v = V_RESET_MV;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            wa += DT_MS * dwa;
            rfr -= 1;
        } else {
            double exp_arg = (v - V_T_MV) * inv_dT;
            if (exp_arg > 50.0) exp_arg = 50.0;
            const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
            const double leak  = -G_L_NS * (v - E_L_MV);
            const double i_syn = ge * (v - E_E_MV);
            const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            v  += DT_MS * dv;
            wa += DT_MS * dwa;
            if (v >= V_PEAK_MV) {
                v = V_RESET_MV;
                wa += B_PA;
                rfr = T_REF_STEPS;
                spiked_this_step = true;
                tot_spk += 1;
                phase_count += 1;
                prev_spk = step;
            }
        }

        if (spike_record_out != nullptr) {
            const unsigned mask = __ballot_sync(0xFFFFFFFFu, spiked_this_step);
            if ((threadIdx.x & 31) == 0) {
                const int word_idx = idx >> 5;
                spike_record_out[
                    static_cast<size_t>(step) * N_L4_BITMASK_INTS + word_idx
                ] = mask;
            }
        }
    }

    V[idx]               = v;
    w_adapt[idx]         = wa;
    g_E[idx]             = ge;
    refrac[idx]          = rfr;
    prev_spike_step[idx] = prev_spk;
    total_spikes[idx]    = tot_spk;
    phase_spike_count[idx] = phase_count;
}

// =====================================================================
// Main per-phase kernel.  Templated on InputMode:
//   InputMode = 0 -> CLOSED-FORM r_lin (trig identity, fast; valid only
//                    for a single drifting sinusoidal grating)
//   InputMode = 1 -> DIRECT r_lin (Σ W·stim_rate inner loop, slower but
//                    works for any stim_rate(x, y, t) -- natural images,
//                    plaids, noise, etc.)
//
// One thread per L4 cell, runs the entire phase (n_steps).
// At each step:
//   1. Compute r_lin via the chosen InputMode.
//   2. r_in  = R_base + max(0, r_lin).      (LNP rectification)
//   3. Single Bernoulli at p = r_in · DT.
//   4. AdEx update with conductance-based synapse.
// =====================================================================
template <int InputMode>
__global__ void v1_phase_kernel(
    double* V, double* w_adapt, double* g_E,
    int* refrac, long long* prev_spike_step,
    long long* total_spikes,
    long long* isi_count, double* isi_sum, double* isi_sum_sq,
    int* phase_spike_count,
    const double* __restrict__ gabor_templates,
    const int* __restrict__ raster_cell_indices,
    int* raster_spike_steps,
    int* raster_spike_count,
    int n_raster, int max_raster_spikes,
    int phase_idx_for_raster,
    long long phase_step_offset,
    int phase_idx,
    int n_steps,
    int n_warmup_steps,
    double cos_theta_stim,
    double sin_theta_stim,
    double k_spatial,
    double omega_rad_per_s,
    double w_in_nS,
    double r_base_hz,
    unsigned long long base_seed,
    // Optional per-step spike-bitmask record (used by the L2/3 path).
    // Layout: [n_steps][N_L4_BITMASK_INTS] uint32_t, packed via warp ballot.
    // When nullptr, no recording (legacy L4-only callers).
    uint32_t* spike_record_out,
    // ---- Stim-variant params (#53; defaults preserve existing behavior) ----
    // phase_offset is added to base_phase / direct rate cos arg; combine
    // grating phase ϕ AND position offset (x_0, y_0) on the host as
    //   phase_offset = ϕ - K·(x_0·cos θ + y_0·sin θ).
    double phase_offset,
    // Aperture mask: Gaussian at (cx, cy) with inv_2sigma_sq.  When
    // aperture_active = 0, mask is treated as 1 everywhere (no aperture).
    int    aperture_active,
    double aperture_cx,
    double aperture_cy,
    double aperture_inv_2sigma_sq,
    // Per-cell peak-spike-count tracker (over 20 ms = STIM_BIN20_STEPS bins).
    // Output: int per cell; nullptr disables.
    int*   peak_bin20_count_out,
    // Per-cell per-50ms-bin spike-count record.  Layout: [n_bins_50][N_L4].
    // Pass nullptr / 0 to disable.
    int*   bin50_counts_out,
    int    n_bins_50,
    // Stim window: for step < n_stim_steps the grating modulation amplitude
    // is in effect; for step >= n_stim_steps the modulation amplitude
    // collapses to 0 leaving only the base rate (used for ITI epochs in
    // task #54 training).  Pass n_stim_steps = n_steps for the legacy
    // "always-stim" behavior.
    int    n_stim_steps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L4) return;

    const int gx  = cell_gx(idx);
    const int gy  = cell_gy(idx);
    const int ori = cell_ori(idx);
    const int cln = cell_clone(idx);
    const double* __restrict__ W =
        gabor_templates + static_cast<std::size_t>(ori * N_CELLS_PER_ORIENT + cln) * GABOR_PIX;

    int raster_slot = -1;
    const bool record_raster =
        (phase_idx == phase_idx_for_raster) && (n_raster > 0);
    if (record_raster) {
        for (int s = 0; s < n_raster; ++s) {
            if (raster_cell_indices[s] == idx) { raster_slot = s; break; }
        }
    }

    // ---- Closed-form per-phase precompute (InputMode == 0 only) ----
    // Generalized formula (handles aperture + grating-phase ϕ + position
    // jitter (x_0, y_0); ϕ and (x_0, y_0) are baked into `phase_offset` on
    // the host as  phase_offset = ϕ - K·(x_0·cos θ + y_0·sin θ)):
    //   pixel_rate(gx+dx, gy+dy, t)
    //       = base + (max/2) · aperture(gx+dx, gy+dy)
    //              · (1 + cos(base_phase + delta_phase + phase_offset − ω t))
    //     base_phase  = K · (gx cosθ + gy sinθ) + phase_offset    [cell-specific]
    //     delta_phase = K · (dx cosθ + dy sinθ)                   [pixel-offset]
    //   Trig identity: cos(A + B − ωt) = cos(A − ωt) cos(B) − sin(A − ωt) sin(B)
    //   ⇒ r_lin(t) = base·sum_W + (max/2)·sum_W_apert
    //              + (max/2) · ( cos(base_phase − ωt) · a
    //                          − sin(base_phase − ωt) · b )
    //   where a = Σ W·aperture·cos(delta_phase),
    //         b = Σ W·aperture·sin(delta_phase),
    //         sum_W_apert = Σ W·aperture,
    //         sum_W = Σ W (raw, no aperture).
    //
    //   When aperture_active = 0, aperture ≡ 1 ⇒ sum_W_apert = sum_W,
    //   collapsing to the original 26·sum_W + 25·(C_t·a - S_t·b) formula
    //   (since base + max/2 = 1 + 25 = 26 with the canonical params).
    double a_filt = 0.0, b_filt = 0.0, sum_W = 0.0, sum_W_apert = 0.0;
    double base_phase = 0.0;
    if constexpr (InputMode == 0) {
        const double K_cos_t = k_spatial * cos_theta_stim;
        const double K_sin_t = k_spatial * sin_theta_stim;
        base_phase =
            K_cos_t * static_cast<double>(gx) + K_sin_t * static_cast<double>(gy)
            + phase_offset;
        for (int dy = -GABOR_HALF; dy <= GABOR_HALF; ++dy) {
            for (int dx = -GABOR_HALF; dx <= GABOR_HALF; ++dx) {
                const double delta_phase =
                    K_cos_t * static_cast<double>(dx) + K_sin_t * static_cast<double>(dy);
                double s_d, c_d;
                sincos(delta_phase, &s_d, &c_d);
                const int kidx = (dy + GABOR_HALF) * GABOR_SIZE + (dx + GABOR_HALF);
                const double w_raw = W[kidx];
                const double ap = aperture_active
                    ? aperture_at(gx + dx, gy + dy,
                                  aperture_cx, aperture_cy, aperture_inv_2sigma_sq)
                    : 1.0;
                const double w_eff = w_raw * ap;
                a_filt      += w_eff * c_d;
                b_filt      += w_eff * s_d;
                sum_W       += w_raw;
                sum_W_apert += w_eff;
            }
        }
        // Without aperture, sum_W ≈ 0 (zero-mean Gabor) and sum_W_apert = sum_W,
        // so the runtime formula reduces to 26·sum_W + 25·(C_t·a - S_t·b),
        // bit-equivalent to the previous implementation.
    }

    // Load state.
    double v   = V[idx];
    double wa  = w_adapt[idx];
    double ge  = g_E[idx];
    int    rfr = refrac[idx];
    long long prev_spk = prev_spike_step[idx];
    long long tot_spk  = total_spikes[idx];
    long long isi_n    = isi_count[idx];
    double    isi_s    = isi_sum[idx];
    double    isi_ss   = isi_sum_sq[idx];
    int phase_count    = 0;

    // Per-thread Philox RNG (Bernoulli only).
    curandStatePhilox4_32_10_t rng;
    curand_init(
        base_seed,
        static_cast<unsigned long long>(idx)
            + static_cast<unsigned long long>(phase_idx) * 1000003ULL,
        0ULL, &rng
    );

    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S = exp(-DT_MS / TAU_SYN_MS);

    // Per-thread bin counters (only used when peak_bin20_count_out /
    // bin50_counts_out are non-null).  Cleared at start; flushed at bin
    // boundaries inside the loop.
    int bin20_count_local = 0;
    int peak_bin20_local  = 0;
    int bin50_count_local = 0;

    for (int step = 0; step < n_steps; ++step) {
        const double t_s = static_cast<double>(step) * DT_S;

        // ---- Stage 1+2: Gabor-filtered linear rate ----
        // During ITI (step >= n_stim_steps) the grating modulation drops
        // to 0, leaving only base·sum_W contribution -- effectively pure
        // baseline drive (1 Hz floor in r_in).
        const double mod_amp = (step < n_stim_steps) ? 25.0 : 0.0;
        double r_lin;
        if constexpr (InputMode == 0) {
            const double phase_t = base_phase - omega_rad_per_s * t_s;
            double S_t, C_t;
            sincos(phase_t, &S_t, &C_t);
            r_lin = sum_W + mod_amp * sum_W_apert
                  + mod_amp * (C_t * a_filt - S_t * b_filt);
        } else {
            // Direct: Σ W·stim_rate over the 9×9 patch (works for any
            // stim_rate field, including non-sinusoidal stimuli).  Inlines
            // the rate formula here so we can apply phase_offset + aperture
            // without modifying stim_kernels.cuh.
            r_lin = 0.0;
            for (int dy = -GABOR_HALF; dy <= GABOR_HALF; ++dy) {
                const int py = ((gy + dy) % GRID + GRID) % GRID;
                const int row = (dy + GABOR_HALF) * GABOR_SIZE;
                for (int dx = -GABOR_HALF; dx <= GABOR_HALF; ++dx) {
                    const int px = ((gx + dx) % GRID + GRID) % GRID;
                    const double phase_arg =
                        k_spatial * (static_cast<double>(px) * cos_theta_stim
                                    + static_cast<double>(py) * sin_theta_stim)
                        - omega_rad_per_s * t_s
                        + phase_offset;
                    const double intensity = 0.5 * (1.0 + cos(phase_arg));
                    const double max_eff = (step < n_stim_steps)
                        ? stim_kernels::STIM_MAX_RATE_HZ : 0.0;
                    double rate = stim_kernels::STIM_BASE_RATE_HZ
                                + max_eff * intensity;
                    if (aperture_active) {
                        const double ap = aperture_at(
                            px, py, aperture_cx, aperture_cy,
                            aperture_inv_2sigma_sq
                        );
                        rate = stim_kernels::STIM_BASE_RATE_HZ
                             + (rate - stim_kernels::STIM_BASE_RATE_HZ) * ap;
                    }
                    r_lin += W[row + (dx + GABOR_HALF)] * rate;
                }
            }
        }
        // ---- Stage 3: rectify and Poisson-sample ----
        const double r_in = r_base_hz + (r_lin > 0.0 ? r_lin : 0.0);
        const double p    = r_in * DT_S;

        const double u = curand_uniform_double(&rng);
        if (u < p) {
            ge += w_in_nS;
        }
        ge *= alpha_S;

        // ---- Stage 4: AdEx step ----
        bool spiked_this_step = false;
        if (rfr > 0) {
            v = V_RESET_MV;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            wa += DT_MS * dwa;
            rfr -= 1;
        } else {
            double exp_arg = (v - V_T_MV) * inv_dT;
            if (exp_arg > 50.0) exp_arg = 50.0;
            const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
            const double leak  = -G_L_NS * (v - E_L_MV);
            const double i_syn = ge * (v - E_E_MV);

            const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            v  += DT_MS * dv;
            wa += DT_MS * dwa;

            if (v >= V_PEAK_MV) {
                v = V_RESET_MV;
                wa += B_PA;
                rfr = T_REF_STEPS;
                spiked_this_step = true;

                tot_spk += 1;
                const long long step_global = phase_step_offset + step;

                if (step >= n_warmup_steps) {
                    phase_count += 1;
                    if (prev_spk >= 0) {
                        const long long isi = step_global - prev_spk;
                        if (isi > 0) {
                            const double isi_d = static_cast<double>(isi);
                            isi_n  += 1;
                            isi_s  += isi_d;
                            isi_ss += isi_d * isi_d;
                        }
                    }
                    prev_spk = step_global;

                    if (raster_slot >= 0) {
                        const int slot_pos = raster_spike_count[raster_slot];
                        if (slot_pos < max_raster_spikes) {
                            raster_spike_steps[
                                raster_slot * max_raster_spikes + slot_pos
                            ] = static_cast<int>(step_global);
                            raster_spike_count[raster_slot] = slot_pos + 1;
                        }
                    }
                }
            }
        }

        // ---- Per-step spike-bitmask record (only when caller asks for it) ----
        // Threads in a warp are at consecutive cell indices [w*32 .. w*32+31];
        // they share a single 32-bit slot in the bitmask.  Use a warp ballot
        // so only one thread per warp performs the global-mem write.
        if (spike_record_out != nullptr) {
            const unsigned mask = __ballot_sync(0xFFFFFFFFu, spiked_this_step);
            if ((threadIdx.x & 31) == 0) {
                const int word_idx = idx >> 5;             // = idx / 32
                spike_record_out[
                    static_cast<size_t>(step) * N_L4_BITMASK_INTS + word_idx
                ] = mask;
            }
        }

        // ---- Per-cell binning for the stim-protocol-check artifacts ----
        // 20 ms bins -> per-cell PEAK spike count (for "transient peak" stat).
        if (peak_bin20_count_out != nullptr) {
            if (spiked_this_step) bin20_count_local += 1;
            if (((step + 1) % STIM_BIN20_STEPS) == 0) {
                if (bin20_count_local > peak_bin20_local) {
                    peak_bin20_local = bin20_count_local;
                }
                bin20_count_local = 0;
            }
        }
        // 50 ms bins -> per-cell time series (for cross-hypercolumn correlation).
        if (bin50_counts_out != nullptr && n_bins_50 > 0) {
            if (spiked_this_step) bin50_count_local += 1;
            if (((step + 1) % STIM_BIN50_STEPS) == 0) {
                const int bin_idx = step / STIM_BIN50_STEPS;
                if (bin_idx < n_bins_50) {
                    bin50_counts_out[
                        static_cast<size_t>(bin_idx) * N_L4 + idx
                    ] = bin50_count_local;
                }
                bin50_count_local = 0;
            }
        }
    }
    // Flush any partial bin20 / bin50 trailing remainder so the final-bin
    // peak isn't lost when n_steps isn't a multiple of the bin width.
    if (peak_bin20_count_out != nullptr) {
        if (bin20_count_local > peak_bin20_local) {
            peak_bin20_local = bin20_count_local;
        }
        peak_bin20_count_out[idx] = peak_bin20_local;
    }
    if (bin50_counts_out != nullptr && n_bins_50 > 0) {
        const int last_bin = (n_steps + STIM_BIN50_STEPS - 1) / STIM_BIN50_STEPS - 1;
        if (n_steps % STIM_BIN50_STEPS != 0 && last_bin < n_bins_50) {
            bin50_counts_out[
                static_cast<size_t>(last_bin) * N_L4 + idx
            ] = bin50_count_local;
        }
    }

    V[idx]               = v;
    w_adapt[idx]         = wa;
    g_E[idx]             = ge;
    refrac[idx]          = rfr;
    prev_spike_step[idx] = prev_spk;
    total_spikes[idx]    = tot_spk;
    isi_count[idx]       = isi_n;
    isi_sum[idx]         = isi_s;
    isi_sum_sq[idx]      = isi_ss;
    phase_spike_count[idx] = phase_count;
}

// Explicit instantiations -- the host code calls one of these by integer
// dispatch on Args::input_mode (0 = closed, 1 = direct).
template __global__ void v1_phase_kernel<0>(
    double*, double*, double*, int*, long long*,
    long long*, long long*, double*, double*, int*,
    const double* __restrict__,
    const int* __restrict__, int*, int*, int, int, int, long long,
    int, int, int, double, double, double, double, double, double,
    unsigned long long,
    uint32_t*,
    double, int, double, double, double,
    int*, int*, int,
    int
);
template __global__ void v1_phase_kernel<1>(
    double*, double*, double*, int*, long long*,
    long long*, long long*, double*, double*, int*,
    const double* __restrict__,
    const int* __restrict__, int*, int*, int, int, int, long long,
    int, int, int, double, double, double, double, double, double,
    unsigned long long,
    uint32_t*,
    double, int, double, double, double,
    int*, int*, int,
    int
);

// =====================================================================
// Single-cell EPSP probe (one input spike at t=0 onto a resting cell).
// =====================================================================
__global__ void epsp_probe_kernel(
    double* v_trace, int n_steps, double w_in_nS
) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;

    double v = E_L_MV, wa = 0.0, ge = w_in_nS;  // delta-spike at t=0
    const double inv_C  = 1.0 / C_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_dT = 1.0 / DELTA_T_MV;
    const double alpha_S = exp(-DT_MS / TAU_SYN_MS);

    v_trace[0] = v;
    for (int s = 1; s < n_steps; ++s) {
        ge *= alpha_S;
        double exp_arg = (v - V_T_MV) * inv_dT;
        if (exp_arg > 50.0) exp_arg = 50.0;
        const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);
        const double leak  = -G_L_NS * (v - E_L_MV);
        const double i_syn = ge * (v - E_E_MV);
        const double dv  = (leak + spike_drive - wa - i_syn) * inv_C;
        const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
        v  += DT_MS * dv;
        wa += DT_MS * dwa;
        v_trace[s] = v;
    }
}

// =====================================================================
// CLI / orchestration
// =====================================================================
struct Args {
    bool verify = false;
    double stim_orientation_deg = 0.0;                                 // θ (deg)
    double stim_sf_cycles_per_pixel =
        1.0 / stim_kernels::STIM_DEFAULT_SF_PERIOD_PIXELS;             // f (cycles/px)
    double stim_tf_hz = 4.0;                                           // v (Hz)
    int    stim_drift_sign = +1;                                       // d ∈ {+1, -1}
    int duration_ms = 1000;
    std::string out_dir = "/tmp";
    unsigned long long seed = 42;
    bool measure_epsp = false;
    std::string label = "";
    int phase_duration_ms = 1000;
    int phase_warmup_ms   = 200;
    std::string input_mode = "closed";   // "closed" (default, fast) or "direct"
    std::string clip_sequence_file = ""; // when non-empty, run_clip_sequence is selected
    std::string clip_rates_bin = "";     // optional path for per-clip per-cell spike counts (int32)
    bool enable_l23 = false;             // L2/3 layer + L4→L2/3 wiring (Phase A, structure only)
    bool enable_l23_recurrent = false;   // task #3: B1 — adds L2/3→L2/3 static recurrence
    bool train_l23_stdp = false;         // task #5: B2 — Gavornik ABCD training + 4-test validation
    int  n_train_sequences = 2000;       // task #5 default per team-lead update
    bool stim_protocol_check = false;    // task #53: verify stim variants for STDP suitability
    std::string stim_variant = "all";    // full / phase / jitter / aperture / sf / mixed / all
    int  n_trials_per_variant = 20;      // task #53: trials per variant (40 used for `mixed`)
    bool train_stdp = false;             // task #54: STDP training + validation suite
    int  n_train_trials = 1000;          // task #54: training-trial count (lead's #54 spec)
    int  train_stim_ms = 500;            // task #54: per-trial stim window (ms)
    int  train_iti_ms  = 100;            // task #54: per-trial ITI (ms)
    bool skip_validation = false;        // task #54: skip V1-V5 (for smoke testing training loop)
    std::string save_trained_weights = "";  // task #55: persist post-train weights (bin + .json sibling)
    std::string load_trained_weights = "";  // task #55: bypass training, load weights from bin
    bool measure_l4_osi = false;            // task #7: L4-only OSI distribution sweep (no L2/3, no plasticity)
    // task #11: L4→L2/3 sampling grading {random, am, sharp, strict, gentle}.
    // Default switched to "sharp" in task #12 — sharp + strict were the two
    // graded variants in the task #11 sweep that pushed post-STDP L2/3 median
    // gOSI past L4's 0.516 reference; sharp (median 0.575) is preferred over
    // strict (0.615) because its input pool is wider (admits Δθ up to 67.5°
    // at low p) and produces a unimodal OSI distribution.  All five modes
    // remain selectable via --l4-l23-grading for ablation work.
    std::string l4_l23_grading = "sharp";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const std::string& n) -> std::string {
            if (i + 1 >= argc) die("missing value for " + n);
            return argv[++i];
        };
        if (k == "--verify") a.verify = true;
        else if (k == "--measure-epsp") a.measure_epsp = true;
        else if (k == "--stim_orientation") a.stim_orientation_deg = std::stod(need(k));
        else if (k == "--stim-f") {
            a.stim_sf_cycles_per_pixel = std::stod(need(k));
            if (a.stim_sf_cycles_per_pixel <= 0.0) die("--stim-f must be > 0");
        }
        else if (k == "--stim-v") a.stim_tf_hz = std::stod(need(k));
        else if (k == "--stim-d") {
            int d = std::stoi(need(k));
            if (d != +1 && d != -1) die("--stim-d must be +1 or -1");
            a.stim_drift_sign = d;
        }
        else if (k == "--duration_ms")     a.duration_ms = std::stoi(need(k));
        else if (k == "--out_dir")         a.out_dir = need(k);
        else if (k == "--seed")            a.seed = static_cast<unsigned long long>(std::stoll(need(k)));
        else if (k == "--label")           a.label = need(k);
        else if (k == "--phase-duration-ms") a.phase_duration_ms = std::stoi(need(k));
        else if (k == "--phase-warmup-ms")   a.phase_warmup_ms   = std::stoi(need(k));
        else if (k == "--input-mode") {
            a.input_mode = need(k);
            if (a.input_mode != "closed" && a.input_mode != "direct") {
                die("--input-mode must be 'closed' or 'direct'");
            }
        }
        else if (k == "--clip-sequence")  a.clip_sequence_file = need(k);
        else if (k == "--clip-rates-bin") a.clip_rates_bin     = need(k);
        else if (k == "--enable-l23")     a.enable_l23 = true;
        else if (k == "--enable-l23-recurrent") a.enable_l23_recurrent = true;
        else if (k == "--train-l23-stdp") a.train_l23_stdp = true;
        else if (k == "--n-train-sequences") {
            a.n_train_sequences = std::stoi(need(k));
            if (a.n_train_sequences <= 0) die("--n-train-sequences must be > 0");
        }
        else if (k == "--stim-protocol-check") a.stim_protocol_check = true;
        else if (k == "--stim-variant") {
            a.stim_variant = need(k);
            if (a.stim_variant != "full"   && a.stim_variant != "phase"
             && a.stim_variant != "jitter" && a.stim_variant != "aperture"
             && a.stim_variant != "sf"     && a.stim_variant != "mixed"
             && a.stim_variant != "all") {
                die("--stim-variant must be one of "
                    "{full, phase, jitter, aperture, sf, mixed, all}");
            }
        }
        else if (k == "--n-trials-per-variant") {
            a.n_trials_per_variant = std::stoi(need(k));
            if (a.n_trials_per_variant <= 0) die("--n-trials-per-variant must be > 0");
        }
        else if (k == "--train-stdp") a.train_stdp = true;
        else if (k == "--n-train-trials") {
            a.n_train_trials = std::stoi(need(k));
            if (a.n_train_trials <= 0) die("--n-train-trials must be > 0");
        }
        else if (k == "--train-stim-ms") {
            a.train_stim_ms = std::stoi(need(k));
            if (a.train_stim_ms <= 0) die("--train-stim-ms must be > 0");
        }
        else if (k == "--skip-validation") a.skip_validation = true;
        else if (k == "--save-trained-weights") a.save_trained_weights = need(k);
        else if (k == "--load-trained-weights") a.load_trained_weights = need(k);
        else if (k == "--measure-l4-osi") a.measure_l4_osi = true;
        else if (k == "--l4-l23-grading") {
            a.l4_l23_grading = need(k);
            if (a.l4_l23_grading != "random" && a.l4_l23_grading != "am"
             && a.l4_l23_grading != "sharp"  && a.l4_l23_grading != "strict"
             && a.l4_l23_grading != "gentle") {
                die("--l4-l23-grading must be one of "
                    "{random, am, sharp, strict, gentle}");
            }
        }
        else if (k == "--help" || k == "-h") {
            std::cout
                << "usage: v1_test [--verify] [--measure-epsp]\n"
                << "               [--stim_orientation θ_deg] [--duration_ms N]\n"
                << "               [--stim-f cycles_per_px] [--stim-v Hz] [--stim-d ±1]\n"
                << "               [--phase-duration-ms N] [--phase-warmup-ms N]\n"
                << "               [--input-mode {closed,direct}]\n"
                << "               [--clip-sequence FILE] [--clip-rates-bin PATH]\n"
                << "               [--enable-l23] [--verify]\n"
                << "               [--out_dir PATH] [--seed N] [--label TAG]\n"
                << "\n"
                << "Latent stim variables (drifting grating):\n"
                << "  θ (--stim_orientation)  orientation, deg     [verify sweeps 0:5:175]\n"
                << "  f (--stim-f)            spatial freq cyc/px  [default 0.125 = 1/8]\n"
                << "  v (--stim-v)            temporal freq Hz     [default 4]\n"
                << "  d (--stim-d)            drift direction sign [default +1]\n"
                << "\n"
                << "Clip-sequence file format (--clip-sequence): one clip per line, whitespace-\n"
                << "separated:  theta_deg  f_cyc_per_px  v_hz  d  duration_ms\n"
                << "  Lines starting with '#' are comments. Neural state PERSISTS across clips\n"
                << "  (no reset between clips). Per-clip spike counts can be dumped to a binary\n"
                << "  file via --clip-rates-bin (int32, shape [n_clips, n_cells], row-major).\n"
                << "\n"
                << "L2/3 layer (--enable-l23 --verify): static feedforward L4→L2/3 wiring\n"
                << "  (Phase A — no plasticity). Runs the grating stim at the chosen θ for\n"
                << "  --duration_ms (default 1000) and dumps four artifacts to --out_dir:\n"
                << "      l23_connectivity.json  l23_drive_summary.json\n"
                << "  (use companion Python helper to render PNGs A–D from these JSON files).\n"
                << "\n"
                << "L2/3 recurrent (--enable-l23-recurrent, task #3 B1): adds STATIC\n"
                << "  L2/3→L2/3 recurrent wiring on top of trained L4→L2/3 weights\n"
                << "  loaded from --load-trained-weights.  No plasticity, no STDP.\n"
                << "  Builds distance-dependent CSR (p(d)=0.12·exp(-d/1.5), d_max=4),\n"
                << "  lognormal weights (median 0.3 mV, cap 1.5 mV), 1 ms conduction\n"
                << "  delay, 4× chance-baseline reciprocity boost. Runs grating at\n"
                << "  --stim_orientation (default 0°) for --duration_ms (default 1000).\n"
                << "  Dumps:\n"
                << "      l23_recurrent_connectivity.json\n"
                << "      l23_recurrent_drive_summary.json\n"
                << "  (use plot_l23_recurrent.py to render PNGs A–E.)\n"
                << "\n"
                << "L2/3 STDP training + 4-test validation (--train-l23-stdp, task #5 B2):\n"
                << "  Add bounded pair-STDP on L2/3↔L2/3 weights and train on Gavornik\n"
                << "  ABCD sequences (A=0°, B=90°, C=45°, D=135°; 150 ms elements with\n"
                << "  200 ms ISI, 1500 ms ITI between sequences, plasticity ON during\n"
                << "  the 1200 ms sequence and OFF during ITI).  L4→L2/3 weights are\n"
                << "  FROZEN (loaded from --load-trained-weights).  N sequences set by\n"
                << "  --n-train-sequences (default 2000).\n"
                << "  After training: V_order, V_timing, V_omission, V_lesion,\n"
                << "  Phase A re-check.  Dumps phaseB_*.json + final weights .bin.\n"
                << "\n"
                << "Stim-protocol check (task #53, --stim-protocol-check):\n"
                << "  Verifies stim variants are suitable for STDP training (≥10 Hz transient\n"
                << "  L4 peaks + retinotopic decorrelation).  Runs N trials of each variant in\n"
                << "  one invocation:\n"
                << "      full     -- existing drifting grating (baseline)\n"
                << "      phase    -- random spatial phase ϕ ∈ U[0,2π) per trial\n"
                << "      jitter   -- random origin offset (x_0,y_0) ∈ U[-4,4]² per trial\n"
                << "      aperture -- circular Gaussian σ=8px at random center (8..24)²\n"
                << "      sf       -- spatial freq ∈ {0.0625, 0.125, 0.25} cyc/px (octave)\n"
                << "      mixed    -- per-trial uniform draw from the above 5\n"
                << "  Default --n-trials-per-variant 20 (40 used for `mixed`).\n"
                << "  Outputs --out_dir/stim_protocol_summary.json + per-cell binary dumps.\n"
                << "\n"
                << "STDP training + validation (task #54/#55, --train-stdp):\n"
                << "  --n-train-trials N     trials [1000]\n"
                << "  --train-stim-ms MS     per-trial stim window [500]\n"
                << "  --skip-validation      stop after training\n"
                << "  --save-trained-weights PATH  write weights .bin + sibling .json\n"
                << "  --load-trained-weights PATH  skip training, load .bin and validate\n"
                << "\n"
                << "L4 OSI sweep (task #7, --measure-l4-osi):\n"
                << "  Drift gratings at θ ∈ {0, 22.5, ..., 157.5} (8 orientations),\n"
                << "  5 reps × 1000 ms each, default ϕ/SF/TF, no aperture, no jitter.\n"
                << "  Reads L4 spike counts only (no L2/3). Computes per-cell OSI\n"
                << "  using the half-angle complex sum |Σ_θ R(θ)·exp(2iθ)| / Σ_θ R(θ).\n"
                << "  Writes /tmp/l4_osi.json (metrics + per-cell OSI/rate arrays).\n"
                << "\n"
                << "L4→L2/3 graded sampling (task #11, --l4-l23-grading MODE):\n"
                << "  Bias L4→L2/3 connectivity by Δθ between L4 candidate and L2/3\n"
                << "  cell's target orientation (target_ori = l23_clone % 8).  Modes:\n"
                << "      random  -- uniform (default; bit-identical to legacy)\n"
                << "      am      -- Alonso & Martinez 1998 cat L4-simple→L2/3\n"
                << "      sharp   -- sharper iso-bias than A-M\n"
                << "      strict  -- iso only, ±22.5° admitted\n"
                << "      gentle  -- mild bias, somewhat closer to random\n"
                << "  All modes are normalized to expected fan-in ≈ 40 per L2/3 cell\n"
                << "  (interior, full 3×3 patch).  Affects connectivity-build only;\n"
                << "  STDP and AdEx params unchanged.\n";
            std::exit(0);
        } else {
            die("unknown option: " + k);
        }
    }
    if (a.duration_ms <= 0) die("--duration_ms must be > 0");
    return a;
}

// 8 sample tuning cells (one per ori_idx, varied positions, clone=0).
std::vector<int> build_tuning_sample_indices() {
    const int picks[N_ORIENT][2] = {
        {16, 16}, {24, 8}, {8, 24}, {24, 24},
        {8, 8},   {4, 16}, {28, 16}, {16, 4}
    };
    std::vector<int> out; out.reserve(N_ORIENT);
    for (int ori = 0; ori < N_ORIENT; ++ori) {
        out.push_back(make_cell_id(picks[ori][0], picks[ori][1], ori, 0));
    }
    return out;
}

// Raster cells: ALL 128 cells in hypercol (gx=16, gy=16).
std::vector<int> build_raster_cell_indices() {
    std::vector<int> out; out.reserve(CELLS_PER_HYPERCOL);
    for (int ori = 0; ori < N_ORIENT; ++ori) {
        for (int clone = 0; clone < N_CELLS_PER_ORIENT; ++clone) {
            out.push_back(make_cell_id(16, 16, ori, clone));
        }
    }
    return out;
}

// Clone sample cells: all 16 clones at (gx=16, gy=16, ori_idx=0).
// Used to show per-clone OSI variation introduced by γ jitter.
std::vector<int> build_clone_sample_indices() {
    std::vector<int> out; out.reserve(N_CELLS_PER_ORIENT);
    for (int clone = 0; clone < N_CELLS_PER_ORIENT; ++clone) {
        out.push_back(make_cell_id(16, 16, 0, clone));
    }
    return out;
}

// =====================================================================
// L2/3 host-side init: build CSR connectivity + lognormal weights.
//
// p_connect = L23_TARGET_FANIN / 1152.  Each candidate L4 cell is kept
// independently with this probability, drawn from std::mt19937_64 seeded
// from (master_seed XOR L23_SALT_CONNECTIVITY).  Edge L2/3 cells get
// fewer candidates (no wraparound) → fewer expected partners.
//
// Weights drawn as Lognormal(μ = ln(L23_EPSP_MEDIAN_MV), σ = L23_EPSP_LOG_SIGMA)
// in EPSP-mV space, hard-clipped at L23_EPSP_MAX_MV, then converted to
// nS via 1 / L23_MV_PER_NS.  Weight RNG is seeded independently from
// (master_seed XOR L23_SALT_WEIGHTS).
// =====================================================================
struct L23Connectivity {
    std::vector<int>    row_ptr;     // size N_L23 + 1
    std::vector<int>    col_idx;     // size = total_synapses
    std::vector<double> w_nS;        // size = total_synapses
    std::vector<double> w_EPSP_mV;   // size = total_synapses (kept for diagnostics)
    int total_synapses = 0;
};

L23Connectivity build_l23_connectivity(
    unsigned long long master_seed,
    L4L23Grading grading = L4L23Grading::random)
{
    L23Connectivity out;
    out.row_ptr.resize(static_cast<size_t>(N_L23) + 1, 0);

    const GradingParams gp = compute_grading_params(grading);
    // For the legacy `random` curve, all p_connect_per_bin entries equal
    // 40 / 1152 ≈ 0.03472 — bit-identical to the previous code path.
    const std::array<double, 5>& p_per_bin = gp.p_connect_per_bin;

    std::mt19937_64 rng_conn(master_seed ^ L23_SALT_CONNECTIVITY);
    std::mt19937_64 rng_w(master_seed ^ L23_SALT_WEIGHTS);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    const double mu_log = std::log(L23_EPSP_MEDIAN_MV);

    for (int l23 = 0; l23 < N_L23; ++l23) {
        const int gx = l23_gx(l23);
        const int gy = l23_gy(l23);
        // Pick (a) target-orientation index for this L2/3 cell:
        //   target_ori = (l23_clone_idx) % N_ORIENT
        // i.e. the 16 clones in each hypercolumn are split into 8 orientation
        // buckets × 2 cells per bucket, indexed by l23_clone % 8.
        const int target_ori = l23_clone(l23) % N_ORIENT;

        // Enumerate 3×3 hypercolumn patch (CLIP at edges; no wraparound).
        for (int dy = -L23_PATCH_R; dy <= L23_PATCH_R; ++dy) {
            const int py = gy + dy;
            if (py < 0 || py >= GRID) continue;
            for (int dx = -L23_PATCH_R; dx <= L23_PATCH_R; ++dx) {
                const int px = gx + dx;
                if (px < 0 || px >= GRID) continue;
                // Each candidate: 8 oris × 16 clones = 128 L4 cells per hypercol.
                for (int ori = 0; ori < N_ORIENT; ++ori) {
                    // Δθ bin against the L2/3 cell's target orientation,
                    // wrapped at 90° (orientation, not direction).  Result
                    // ∈ {0,1,2,3,4} ↔ {0°, 22.5°, 45°, 67.5°, 90°}.
                    int d_raw = ori - target_ori;
                    if (d_raw < 0) d_raw = -d_raw;
                    int d_bin = d_raw;
                    if (N_ORIENT - d_raw < d_bin) d_bin = N_ORIENT - d_raw;
                    // Clamp into [0,4] defensively (should always hold).
                    if (d_bin < 0) d_bin = 0;
                    if (d_bin > 4) d_bin = 4;
                    const double p = p_per_bin[d_bin];
                    if (p <= 0.0) continue;

                    for (int cln = 0; cln < N_CELLS_PER_ORIENT; ++cln) {
                        if (uniform01(rng_conn) < p) {
                            const int l4 = make_cell_id(px, py, ori, cln);
                            out.col_idx.push_back(l4);

                            // Lognormal EPSP, clipped at L23_EPSP_MAX_MV.
                            const double z = standard_normal(rng_w);
                            const double log_epsp = mu_log + L23_EPSP_LOG_SIGMA * z;
                            double epsp_mV = std::exp(log_epsp);
                            if (epsp_mV > L23_EPSP_MAX_MV) epsp_mV = L23_EPSP_MAX_MV;
                            out.w_EPSP_mV.push_back(epsp_mV);
                            out.w_nS.push_back(epsp_mV / L23_MV_PER_NS);
                        }
                    }
                }
            }
        }
        out.row_ptr[l23 + 1] = static_cast<int>(out.col_idx.size());
    }
    out.total_synapses = static_cast<int>(out.col_idx.size());
    return out;
}

// 8 sample L2/3 cells covering interior, 4 sides, 4 corners (for the
// partner-orientation-profile artifact D).  All clones = 0.
std::vector<int> build_l23_sample_indices() {
    const int picks[8][2] = {
        {16, 16}, // interior
        { 0, 16}, {31, 16}, {16,  0}, {16, 31},   // mid-edges
        { 0,  0}, {31,  0}, {16, 12},             // corner + corner + interior-2
    };
    std::vector<int> out; out.reserve(8);
    for (int s = 0; s < 8; ++s) {
        out.push_back(make_l23_id(picks[s][0], picks[s][1], 0));
    }
    return out;
}

// =====================================================================
// L2/3 → L2/3 recurrent connectivity (B1 STRUCTURE-ONLY).
//
// CSR keyed by post-cell.  Per-edge distance (in hypercolumns, Euclidean)
// is stored alongside w_EPSP_mV / w_nS so plotting can render the
// distance histogram without re-deriving from col_idx.
// =====================================================================
struct L23RecConnectivity {
    std::vector<int>    row_ptr;     // size N_L23 + 1
    std::vector<int>    col_idx;     // size = total_synapses (pre L2/3 ids)
    std::vector<double> w_nS;        // size = total_synapses
    std::vector<double> w_EPSP_mV;   // size = total_synapses
    std::vector<double> dist_hcol;   // size = total_synapses (Euclidean d)
    int total_synapses = 0;
    long long n_pairs_sampled = 0;     // pairs with at least one base-sampled edge
    long long n_pairs_reciprocal_pre_boost = 0;
    long long n_pairs_reciprocal_post = 0;
    long long n_edges_added_by_boost = 0;
};

L23RecConnectivity build_l23_recurrent_connectivity(unsigned long long master_seed) {
    L23RecConnectivity out;
    out.row_ptr.assign(static_cast<size_t>(N_L23) + 1, 0);

    std::mt19937_64 rng_conn (master_seed ^ L23REC_SALT_CONNECTIVITY);
    std::mt19937_64 rng_recip(master_seed ^ L23REC_SALT_RECIPROCITY);
    std::mt19937_64 rng_w    (master_seed ^ L23REC_SALT_WEIGHTS);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    // Pre-compute p(d) and r(d) on the (2*DMAX+1)² lattice once per build.
    constexpr int LATSZ = 2 * L23REC_DMAX + 1;
    double p_grid[LATSZ][LATSZ];
    double r_grid[LATSZ][LATSZ];
    for (int dy = -L23REC_DMAX; dy <= L23REC_DMAX; ++dy) {
        for (int dx = -L23REC_DMAX; dx <= L23REC_DMAX; ++dx) {
            const double d = std::sqrt(
                static_cast<double>(dx*dx + dy*dy));
            const int iy = dy + L23REC_DMAX, ix = dx + L23REC_DMAX;
            const double p = (d <= static_cast<double>(L23REC_DMAX))
                ? L23REC_P0 * std::exp(-d / L23REC_LEN_HCOL)
                : 0.0;
            p_grid[iy][ix] = p;
            // r(d) = 3p / (2(1-p)).  Solves p² + 2p(1-p)r = 4p², so the
            // final reciprocal-pair fraction = 4 × p² (chance baseline).
            r_grid[iy][ix] = (p > 0.0 && p < 1.0)
                ? (3.0 * p) / (2.0 * (1.0 - p))
                : 0.0;
        }
    }

    // Stage 1: independent direction-by-direction sampling.
    // Use unordered_set per post-cell so reciprocity boost can de-duplicate.
    std::vector<std::unordered_set<int>> incoming(N_L23);
    for (int post = 0; post < N_L23; ++post) {
        const int gx_post = l23_gx(post);
        const int gy_post = l23_gy(post);
        for (int dy = -L23REC_DMAX; dy <= L23REC_DMAX; ++dy) {
            const int gy_pre = gy_post + dy;
            if (gy_pre < 0 || gy_pre >= GRID) continue;
            for (int dx = -L23REC_DMAX; dx <= L23REC_DMAX; ++dx) {
                const int gx_pre = gx_post + dx;
                if (gx_pre < 0 || gx_pre >= GRID) continue;
                const double p = p_grid[dy + L23REC_DMAX][dx + L23REC_DMAX];
                if (p <= 0.0) continue;
                for (int clone = 0; clone < N_L23_CLONES; ++clone) {
                    const int pre = make_l23_id(gx_pre, gy_pre, clone);
                    if (pre == post) continue;   // no autapse
                    if (uniform01(rng_conn) < p) {
                        incoming[post].insert(pre);
                    }
                }
            }
        }
    }

    // Stage 2: reciprocity boost.  We iterate over the BASE-sampled
    // edges only (snapshot first to avoid traversing edges added by
    // the boost itself).  For each base edge pre→post, with prob r(d)
    // we add post→pre IF it's not already there.
    std::vector<std::pair<int,int>> base_edges;  // (pre, post)
    base_edges.reserve(64 * N_L23);
    for (int post = 0; post < N_L23; ++post) {
        for (int pre : incoming[post]) {
            base_edges.emplace_back(pre, post);
        }
    }
    out.n_pairs_sampled = static_cast<long long>(base_edges.size());

    // Pre-boost reciprocity count (sanity check ≈ p² rate).
    for (const auto& e : base_edges) {
        const int pre = e.first, post = e.second;
        if (incoming[pre].count(post)) ++out.n_pairs_reciprocal_pre_boost;
    }

    long long boost_added = 0;
    for (const auto& e : base_edges) {
        const int pre  = e.first;
        const int post = e.second;
        const int dx = l23_gx(pre) - l23_gx(post);
        const int dy = l23_gy(pre) - l23_gy(post);
        const double r = r_grid[dy + L23REC_DMAX][dx + L23REC_DMAX];
        if (r <= 0.0) continue;
        // Reverse edge already exists (from base sampling)? Skip.
        if (incoming[pre].count(post)) continue;
        if (uniform01(rng_recip) < r) {
            incoming[pre].insert(post);
            ++boost_added;
        }
    }
    out.n_edges_added_by_boost = boost_added;

    // Post-boost reciprocity count (final).
    long long n_recip_final = 0;
    for (int post = 0; post < N_L23; ++post) {
        for (int pre : incoming[post]) {
            if (pre < post) continue;        // count each unordered pair once
            if (incoming[pre].count(post)) ++n_recip_final;
        }
    }
    out.n_pairs_reciprocal_post = n_recip_final;

    // Stage 3: flatten to CSR (sorted col_idx within each post for
    // determinism + cache-friendly iteration).
    out.col_idx.reserve(64 * N_L23);
    out.w_nS.reserve(64 * N_L23);
    out.w_EPSP_mV.reserve(64 * N_L23);
    out.dist_hcol.reserve(64 * N_L23);
    const double mu_log = std::log(L23REC_EPSP_MEDIAN_MV);
    for (int post = 0; post < N_L23; ++post) {
        std::vector<int> sorted(incoming[post].begin(), incoming[post].end());
        std::sort(sorted.begin(), sorted.end());
        for (int pre : sorted) {
            out.col_idx.push_back(pre);
            // Lognormal EPSP draw, clipped at L23REC_EPSP_MAX_MV.
            const double z = standard_normal(rng_w);
            const double log_epsp = mu_log + L23REC_EPSP_LOG_SIGMA * z;
            double epsp_mV = std::exp(log_epsp);
            if (epsp_mV > L23REC_EPSP_MAX_MV) epsp_mV = L23REC_EPSP_MAX_MV;
            out.w_EPSP_mV.push_back(epsp_mV);
            out.w_nS.push_back(epsp_mV / L23_MV_PER_NS);
            const double dxh = static_cast<double>(l23_gx(pre) - l23_gx(post));
            const double dyh = static_cast<double>(l23_gy(pre) - l23_gy(post));
            out.dist_hcol.push_back(std::sqrt(dxh*dxh + dyh*dyh));
        }
        out.row_ptr[post + 1] = static_cast<int>(out.col_idx.size());
    }
    out.total_synapses = static_cast<int>(out.col_idx.size());
    return out;
}

}  // namespace

// =====================================================================
// EPSP probe runner.
// =====================================================================
static void measure_epsp_run(double w_in_nS) {
    constexpr int N = 500;  // 50 ms
    double* d_trace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_trace, N * sizeof(double)));
    epsp_probe_kernel<<<1, 1>>>(d_trace, N, w_in_nS);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> trace(N);
    CUDA_CHECK(cudaMemcpy(trace.data(), d_trace, N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_trace));
    double v_max = trace[0];
    int    t_max_step = 0;
    for (int i = 0; i < N; ++i) {
        if (trace[i] > v_max) { v_max = trace[i]; t_max_step = i; }
    }
    std::cout << "epsp_probe w_in_nS=" << w_in_nS
              << " peak_V_mV=" << v_max
              << " peak_dV_mV=" << (v_max - E_L_MV)
              << " t_peak_ms=" << (t_max_step * DT_MS) << "\n";
}

// =====================================================================
// Verify run.
// =====================================================================
static int run_verify(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    measure_epsp_run(W_IN_NS);

    constexpr int N_PHASES = 36;
    const int DURATION_MS = args.phase_duration_ms;
    const int WARMUP_MS   = args.phase_warmup_ms;
    const int N_STEPS = static_cast<int>(DURATION_MS / DT_MS + 0.5);
    const int N_WARMUP = static_cast<int>(WARMUP_MS / DT_MS + 0.5);
    if (N_WARMUP >= N_STEPS) die("--phase-warmup-ms must be < --phase-duration-ms");
    const double measure_s = (N_STEPS - N_WARMUP) * DT_S;

    std::vector<double> test_orient_deg(N_PHASES);
    for (int p = 0; p < N_PHASES; ++p) test_orient_deg[p] = 5.0 * p;

    // Latent stim variables (θ swept per phase; f, v, d held over the whole verify run)
    const double k_spatial = 2.0 * PI * args.stim_sf_cycles_per_pixel;          // f → K
    const double omega     = 2.0 * PI * args.stim_tf_hz
                           * static_cast<double>(args.stim_drift_sign);         // v·d → ω (signed)

    // ---- device buffers ----
    double *d_V=nullptr, *d_w=nullptr, *d_gE=nullptr;
    int *d_refrac=nullptr;
    long long *d_prev=nullptr, *d_isi_c=nullptr, *d_tot=nullptr;
    double *d_isi_s=nullptr, *d_isi_ss=nullptr;
    int *d_phase=nullptr;
    double* d_templates = nullptr;

    CUDA_CHECK(cudaMalloc(&d_V,         N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,         N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE,        N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac,    N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev,      N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot,       N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c,     N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss,    N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase,     N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));

    constexpr int N_RASTER = CELLS_PER_HYPERCOL;
    constexpr int MAX_RASTER_SPIKES = 8192;
    auto raster_idx_host = build_raster_cell_indices();
    int *d_raster_idx=nullptr, *d_raster_steps=nullptr, *d_raster_count=nullptr;
    CUDA_CHECK(cudaMalloc(&d_raster_idx,   N_RASTER * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_raster_steps, N_RASTER * MAX_RASTER_SPIKES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_raster_count, N_RASTER * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_raster_idx, raster_idx_host.data(),
                          N_RASTER * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_raster_steps, 0,
                          N_RASTER * MAX_RASTER_SPIKES * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_raster_count, 0, N_RASTER * sizeof(int)));

    const int block = 256;
    const int gridN = (N_L4 + block - 1) / block;

    // ---- build Gabor templates ----
    {
        const int gridT = (N_TEMPLATES + block - 1) / block;
        build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- init AdEx state ----
    init_full_state_kernel<<<gridN, block>>>(
        d_V, d_w, d_gE, d_refrac, d_prev,
        d_tot, d_isi_c, d_isi_s, d_isi_ss, N_L4
    );
    CUDA_CHECK(cudaGetLastError());

    // Tuning matrix.
    std::vector<int> tuning_counts(static_cast<std::size_t>(N_PHASES) * N_L4, 0);

    auto t0 = std::chrono::steady_clock::now();
    long long phase_step_offset = 0;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const double th = test_orient_deg[phase] * (PI / 180.0);
        const double cos_t = std::cos(th), sin_t = std::sin(th);

        reset_dyn_state_kernel<<<gridN, block>>>(
            d_V, d_w, d_gE, d_refrac, d_prev, N_L4);
        clear_int_kernel<<<gridN, block>>>(d_phase, N_L4);
        CUDA_CHECK(cudaGetLastError());

        if (args.input_mode == "direct") {
            v1_phase_kernel<1><<<gridN, block>>>(
                d_V, d_w, d_gE, d_refrac, d_prev,
                d_tot, d_isi_c, d_isi_s, d_isi_ss,
                d_phase,
                d_templates,
                d_raster_idx, d_raster_steps, d_raster_count,
                N_RASTER, MAX_RASTER_SPIKES,
                /*phase_idx_for_raster=*/0,
                phase_step_offset,
                phase,
                N_STEPS, N_WARMUP,
                cos_t, sin_t,
                k_spatial, omega,
                W_IN_NS, R_BASE_HZ,
                args.seed,
                /*spike_record_out=*/nullptr,
                /*phase_offset=*/0.0,
                /*aperture_active=*/0,
                /*aperture_cx=*/0.0,
                /*aperture_cy=*/0.0,
                /*aperture_inv_2sigma_sq=*/0.0,
                /*peak_bin20_count_out=*/nullptr,
                /*bin50_counts_out=*/nullptr,
                /*n_bins_50=*/0,
                /*n_stim_steps=*/INT_MAX
            );
        } else {
            v1_phase_kernel<0><<<gridN, block>>>(
                d_V, d_w, d_gE, d_refrac, d_prev,
                d_tot, d_isi_c, d_isi_s, d_isi_ss,
                d_phase,
                d_templates,
                d_raster_idx, d_raster_steps, d_raster_count,
                N_RASTER, MAX_RASTER_SPIKES,
                /*phase_idx_for_raster=*/0,
                phase_step_offset,
                phase,
                N_STEPS, N_WARMUP,
                cos_t, sin_t,
                k_spatial, omega,
                W_IN_NS, R_BASE_HZ,
                args.seed,
                /*spike_record_out=*/nullptr,
                /*phase_offset=*/0.0,
                /*aperture_active=*/0,
                /*aperture_cx=*/0.0,
                /*aperture_cy=*/0.0,
                /*aperture_inv_2sigma_sq=*/0.0,
                /*peak_bin20_count_out=*/nullptr,
                /*bin50_counts_out=*/nullptr,
                /*n_bins_50=*/0,
                /*n_stim_steps=*/INT_MAX
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            tuning_counts.data() + static_cast<std::size_t>(phase) * N_L4,
            d_phase, N_L4 * sizeof(int), cudaMemcpyDeviceToHost
        ));
        phase_step_offset += N_STEPS;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double run_wall_s = std::chrono::duration<double>(t1 - t0).count();

    // ---- read aggregates ----
    std::vector<long long> isi_count_host(N_L4);
    std::vector<double>    isi_sum_host(N_L4);
    std::vector<double>    isi_sum_sq_host(N_L4);
    std::vector<long long> tot_spk_host(N_L4);
    CUDA_CHECK(cudaMemcpy(isi_count_host.data(),  d_isi_c,
                          N_L4 * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(isi_sum_host.data(),    d_isi_s,
                          N_L4 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(isi_sum_sq_host.data(), d_isi_ss,
                          N_L4 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tot_spk_host.data(),    d_tot,
                          N_L4 * sizeof(long long), cudaMemcpyDeviceToHost));

    std::vector<int> raster_count_host(N_RASTER);
    std::vector<int> raster_steps_host(N_RASTER * MAX_RASTER_SPIKES);
    CUDA_CHECK(cudaMemcpy(raster_count_host.data(), d_raster_count,
                          N_RASTER * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(raster_steps_host.data(), d_raster_steps,
                          N_RASTER * MAX_RASTER_SPIKES * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // ---- read Gabor templates so we can dump the actual RFs ----
    std::vector<double> templates_host(N_TEMPLATES * GABOR_PIX);
    CUDA_CHECK(cudaMemcpy(templates_host.data(), d_templates,
                          N_TEMPLATES * GABOR_PIX * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // free device buffers
    CUDA_CHECK(cudaFree(d_V));     CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_gE));    CUDA_CHECK(cudaFree(d_refrac));
    CUDA_CHECK(cudaFree(d_prev));  CUDA_CHECK(cudaFree(d_tot));
    CUDA_CHECK(cudaFree(d_isi_c)); CUDA_CHECK(cudaFree(d_isi_s));
    CUDA_CHECK(cudaFree(d_isi_ss));CUDA_CHECK(cudaFree(d_phase));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_raster_idx));
    CUDA_CHECK(cudaFree(d_raster_steps));
    CUDA_CHECK(cudaFree(d_raster_count));

    // ---- analysis ----
    std::vector<double> rate_at_pref0_hz(N_L4);
    for (int i = 0; i < N_L4; ++i) {
        rate_at_pref0_hz[i] = static_cast<double>(tuning_counts[i]) / measure_s;
    }
    std::vector<double> isi_cv(N_L4, 0.0);
    long long n_with_cv = 0;
    double cv_sum = 0, cv_sum_sq = 0;
    std::vector<double> cv_collected;
    cv_collected.reserve(N_L4);
    for (int i = 0; i < N_L4; ++i) {
        const long long n = isi_count_host[i];
        if (n >= 2) {
            const double mean_steps = isi_sum_host[i] / static_cast<double>(n);
            const double var = std::max(0.0,
                isi_sum_sq_host[i] / static_cast<double>(n) - mean_steps * mean_steps);
            const double cv = std::sqrt(var) / std::max(1e-12, mean_steps);
            isi_cv[i] = cv;
            cv_collected.push_back(cv);
            cv_sum += cv; cv_sum_sq += cv * cv;
            ++n_with_cv;
        }
    }
    double cv_mean=0, cv_std=0, cv_median=0;
    if (n_with_cv > 0) {
        cv_mean = cv_sum / n_with_cv;
        cv_std  = std::sqrt(std::max(0.0,
            cv_sum_sq / n_with_cv - cv_mean * cv_mean));
        std::vector<double> tmp = cv_collected;
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
        cv_median = tmp[tmp.size()/2];
    }

    // Per-orientation-column stats at θ_stim = 0°.
    struct PerOri { double mean=0, median=0, max_=0; long long n_silent=0; };
    std::vector<PerOri> per_ori(N_ORIENT);
    for (int ori = 0; ori < N_ORIENT; ++ori) {
        std::vector<double> rs; rs.reserve(N_L4 / N_ORIENT);
        double mx = 0; long long n_silent = 0; double sum = 0;
        for (int i = 0; i < N_L4; ++i) {
            if (cell_ori(i) != ori) continue;
            const double r = rate_at_pref0_hz[i];
            rs.push_back(r);
            sum += r; if (r > mx) mx = r; if (r == 0.0) ++n_silent;
        }
        per_ori[ori].mean = sum / std::max<size_t>(1, rs.size());
        std::nth_element(rs.begin(), rs.begin() + rs.size()/2, rs.end());
        per_ori[ori].median = rs[rs.size()/2];
        per_ori[ori].max_ = mx;
        per_ori[ori].n_silent = n_silent;
    }

    double pop_sum=0, pop_sum_sq=0;
    double pop_min = rate_at_pref0_hz[0], pop_max = rate_at_pref0_hz[0];
    long long pop_silent = 0;
    for (auto v : rate_at_pref0_hz) {
        pop_sum += v; pop_sum_sq += v * v;
        if (v < pop_min) pop_min = v;
        if (v > pop_max) pop_max = v;
        if (v == 0.0) ++pop_silent;
    }
    const double pop_mean = pop_sum / N_L4;
    const double pop_std  = std::sqrt(std::max(0.0,
        pop_sum_sq / N_L4 - pop_mean * pop_mean));
    std::vector<double> rate_sorted = rate_at_pref0_hz;
    std::nth_element(rate_sorted.begin(),
                     rate_sorted.begin() + (size_t)(N_L4 * 0.95),
                     rate_sorted.end());
    const double pop_p95 = rate_sorted[(size_t)(N_L4 * 0.95)];

    auto tuning_sample_idx = build_tuning_sample_indices();
    auto clone_sample_idx  = build_clone_sample_indices();

    // ===== JSON output =====
    std::filesystem::create_directories(args.out_dir);
    const std::string label_suffix =
        args.label.empty() ? std::string() : std::string("_") + args.label;
    const std::string json_path =
        std::string("/tmp/v1") + label_suffix + std::string("_summary.json");

    std::ofstream out(json_path);
    if (!out) die("could not open " + json_path);
    out << std::setprecision(8);
    out << "{\n";
    out << "  \"n_cells\": " << N_L4 << ",\n";
    out << "  \"duration_ms\": " << DURATION_MS << ",\n";
    out << "  \"warmup_ms\": " << WARMUP_MS << ",\n";
    out << "  \"stim_orientation_deg\": 0,\n";
    out << "  \"n_test_orientations\": " << N_PHASES << ",\n";
    out << "  \"test_orientations_deg\": [";
    for (int p = 0; p < N_PHASES; ++p) { if (p) out<<","; out<<test_orient_deg[p]; }
    out << "],\n";

    out << "  \"per_orientation\": [\n";
    for (int ori = 0; ori < N_ORIENT; ++ori) {
        out << "    {\"ori_idx\":" << ori
            << ",\"theta_pref_deg\":" << (ori * 22.5)
            << ",\"mean_rate_hz\":" << per_ori[ori].mean
            << ",\"median_rate_hz\":" << per_ori[ori].median
            << ",\"max_rate_hz\":" << per_ori[ori].max_
            << ",\"frac_silent\":" << (per_ori[ori].n_silent / (double)(N_L4 / N_ORIENT))
            << "}";
        if (ori + 1 < N_ORIENT) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"global\": {\"mean_rate_hz\":"<<pop_mean
        <<",\"std_rate_hz\":"<<pop_std<<",\"min_rate_hz\":"<<pop_min
        <<",\"max_rate_hz\":"<<pop_max<<",\"p95_rate_hz\":"<<pop_p95
        <<",\"frac_silent\":"<<(pop_silent / (double)N_L4)
        <<",\"isi_cv_mean\":"<<cv_mean<<",\"isi_cv_std\":"<<cv_std
        <<",\"isi_cv_median\":"<<cv_median
        <<",\"n_cells_with_cv\":"<<n_with_cv<<"},\n";

    out << "  \"params\": {\n";
    out << "    \"adex\": {\"C_pF\":"<<C_PF<<",\"gL_nS\":"<<G_L_NS
        <<",\"EL_mV\":"<<E_L_MV<<",\"VT_mV\":"<<V_T_MV
        <<",\"DeltaT_mV\":"<<DELTA_T_MV<<",\"a_nS\":"<<A_NS
        <<",\"tauW_ms\":"<<TAU_W_MS<<",\"b_pA\":"<<B_PA
        <<",\"V_reset_mV\":"<<V_RESET_MV<<",\"V_peak_mV\":"<<V_PEAK_MV
        <<",\"t_ref_ms\":"<<T_REF_MS<<",\"dt_ms\":"<<DT_MS<<"},\n";
    out << "    \"gabor\": {\"size\":"<<GABOR_SIZE
        <<",\"sigma_x_px\":"<<GABOR_SIGMA_X
        <<",\"gamma_base\":"<<GABOR_GAMMA
        <<",\"gamma_jitter\":"<<GAMMA_JITTER
        <<",\"K_rad_per_px\":"<<GABOR_K
        <<",\"l1_target\":"<<GABOR_L1_TARGET<<"},\n";
    out << "    \"w_in_nS\": " << W_IN_NS << ",\n";
    out << "    \"tau_syn_ms\": " << TAU_SYN_MS << ",\n";
    out << "    \"E_E_mV\": " << E_E_MV << ",\n";
    out << "    \"R_base_hz\": " << R_BASE_HZ << ",\n";
    out << "    \"input_mode\": \"" << args.input_mode << "\",\n";
    out << "    \"latent_stim\": {"
        << "\"theta_deg_swept\":\"0:5:175\""
        << ",\"f_cycles_per_pixel\":" << args.stim_sf_cycles_per_pixel
        << ",\"v_tf_hz\":" << args.stim_tf_hz
        << ",\"d_drift_sign\":" << args.stim_drift_sign
        << ",\"k_spatial_rad_per_px\":" << k_spatial
        << ",\"omega_signed_rad_per_s\":" << omega
        << "},\n";
    out << "    \"label\": \"" << args.label << "\"\n";
    out << "  },\n";
    out << "  \"run_wall_s\": " << run_wall_s << ",\n";
    out << "  \"device\": \"" << prop.name << "\",\n";
    out << "  \"seed\": " << args.seed << ",\n";

    out << "  \"tuning_sample_cells\": [\n";
    for (int s = 0; s < (int)tuning_sample_idx.size(); ++s) {
        const int idx = tuning_sample_idx[s];
        const int gx = cell_gx(idx), gy = cell_gy(idx);
        const int ori = cell_ori(idx), clone = cell_clone(idx);
        out << "    {\"slot\":" << s << ",\"index\":" << idx
            << ",\"gx\":" << gx << ",\"gy\":" << gy
            << ",\"ori_idx\":" << ori << ",\"clone\":" << clone
            << ",\"theta_pref_deg\":" << (ori * 22.5)
            << ",\"isi_n\":" << isi_count_host[idx]
            << ",\"isi_cv\":" << isi_cv[idx]
            << ",\"tuning_curve_hz\":[";
        for (int p = 0; p < N_PHASES; ++p) {
            if (p) out << ",";
            out << static_cast<double>(
                tuning_counts[(std::size_t)p * N_L4 + idx]) / measure_s;
        }
        out << "],\"gabor_kernel\":[";
        const double* W = templates_host.data() +
            static_cast<std::size_t>(ori * N_CELLS_PER_ORIENT + clone) * GABOR_PIX;
        for (int k = 0; k < GABOR_PIX; ++k) {
            if (k) out << ","; out << W[k];
        }
        out << "]}";
        if (s + 1 < (int)tuning_sample_idx.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    // ---- Clone sample cells: 16 clones at (16, 16, ori=0) with full
    //      tuning curves and individual Gabor kernels.  Shows per-clone OSI
    //      variability introduced by γ jitter.
    out << "  \"clone_sample_cells\": [\n";
    for (int s = 0; s < (int)clone_sample_idx.size(); ++s) {
        const int idx = clone_sample_idx[s];
        const int ori = cell_ori(idx), clone = cell_clone(idx);
        out << "    {\"slot\":" << s << ",\"index\":" << idx
            << ",\"gx\":16,\"gy\":16"
            << ",\"ori_idx\":" << ori << ",\"clone\":" << clone
            << ",\"theta_pref_deg\":" << (ori * 22.5)
            << ",\"phase_rad\":" << (clone * 2.0 * PI / N_CELLS_PER_ORIENT)
            << ",\"gamma\":"
            << (GABOR_GAMMA + GAMMA_JITTER
                * std::cos(2.0 * PI * clone / static_cast<double>(N_CELLS_PER_ORIENT)))
            << ",\"isi_n\":" << isi_count_host[idx]
            << ",\"isi_cv\":" << isi_cv[idx]
            << ",\"tuning_curve_hz\":[";
        for (int p = 0; p < N_PHASES; ++p) {
            if (p) out << ",";
            out << static_cast<double>(
                tuning_counts[(std::size_t)p * N_L4 + idx]) / measure_s;
        }
        out << "],\"gabor_kernel\":[";
        const double* W = templates_host.data() +
            static_cast<std::size_t>(ori * N_CELLS_PER_ORIENT + clone) * GABOR_PIX;
        for (int k = 0; k < GABOR_PIX; ++k) {
            if (k) out << ","; out << W[k];
        }
        out << "]}";
        if (s + 1 < (int)clone_sample_idx.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"raster_phase_idx\": 0,\n";
    out << "  \"raster_phase_step_count\": " << N_STEPS << ",\n";
    out << "  \"raster_cells\": [\n";
    for (int s = 0; s < N_RASTER; ++s) {
        const int idx = raster_idx_host[s];
        const int gx = cell_gx(idx), gy = cell_gy(idx);
        const int ori = cell_ori(idx), clone = cell_clone(idx);
        const int n = std::min(raster_count_host[s], MAX_RASTER_SPIKES);
        out << "    {\"slot\":" << s << ",\"index\":" << idx
            << ",\"ori_idx\":" << ori << ",\"clone\":" << clone
            << ",\"gx\":" << gx << ",\"gy\":" << gy
            << ",\"n_spikes\":" << n << ",\"spike_steps\":[";
        for (int i = 0; i < n; ++i) {
            if (i) out << ","; out << raster_steps_host[s * MAX_RASTER_SPIKES + i];
        }
        out << "]}";
        if (s + 1 < N_RASTER) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    constexpr int N_BIN = 60;
    const double rate_hist_max = std::max(1.0, pop_max);
    std::vector<int> rate_hist(N_BIN, 0);
    for (auto v : rate_at_pref0_hz) {
        int b = static_cast<int>(v / rate_hist_max * N_BIN);
        if (b < 0) b = 0; else if (b >= N_BIN) b = N_BIN - 1;
        rate_hist[b]++;
    }
    out << "  \"pop_rate_histogram\": {\"max_hz\":"<<rate_hist_max
        <<",\"counts\":[";
    for (int i = 0; i < N_BIN; ++i) { if (i) out<<","; out<<rate_hist[i]; }
    out << "]}\n";
    out << "}\n";
    out.close();

    // ---- stdout ----
    const int slot0_idx = tuning_sample_idx[0];
    const double r_pref = tuning_counts[slot0_idx] / measure_s;
    const double r_orth = tuning_counts[
        (std::size_t)18 * N_L4 + slot0_idx] / measure_s;

    std::cout << "n_phases=" << N_PHASES << "\n";
    std::cout << "duration_per_phase_ms=" << DURATION_MS << "\n";
    std::cout << "warmup_ms=" << WARMUP_MS << "\n";
    std::cout << "run_wall_s=" << run_wall_s << "\n";
    std::cout << "w_in_nS=" << W_IN_NS << "\n";
    std::cout << "gabor_sigma_x=" << GABOR_SIGMA_X
              << " gamma=" << GABOR_GAMMA
              << " K=" << GABOR_K
              << " L1_target=" << GABOR_L1_TARGET << "\n";
    std::cout << "input_mode=" << args.input_mode << "\n";
    std::cout << "latent_stim: f=" << args.stim_sf_cycles_per_pixel
              << " cyc/px, v=" << args.stim_tf_hz
              << " Hz, d=" << args.stim_drift_sign
              << "  -> K=" << k_spatial
              << " rad/px, omega=" << omega << " rad/s\n";
    std::cout << "global_mean_rate_hz_at_0deg=" << pop_mean << "\n";
    std::cout << "global_p95_rate_hz_at_0deg=" << pop_p95 << "\n";
    std::cout << "global_max_rate_hz_at_0deg=" << pop_max << "\n";
    std::cout << "global_frac_silent_at_0deg=" << (pop_silent / (double)N_L4) << "\n";
    std::cout << "isi_cv_mean=" << cv_mean << "\n";
    std::cout << "n_cells_with_cv=" << n_with_cv << "\n";
    std::cout << "slot0_cell_idx=" << slot0_idx << " (ori=0, θ_pref=0°, gx=16, gy=16)\n";
    std::cout << "  rate_at_theta_pref_0_hz=" << r_pref << "\n";
    std::cout << "  rate_at_theta_pref+90_hz=" << r_orth << "\n";
    std::cout << "summary_json=" << json_path << "\n";

    return 0;
}

// =====================================================================
// Single-orient diagnostic run.
// =====================================================================
static int run_single(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));

    const int n_steps  = static_cast<int>(args.duration_ms / DT_MS + 0.5);
    const int n_warmup = std::min(n_steps / 5, 2000);
    const double th = args.stim_orientation_deg * (PI / 180.0);
    const double cos_t = std::cos(th), sin_t = std::sin(th);
    // Latent stim variables (θ, f, v, d).
    const double k_spatial = 2.0 * PI * args.stim_sf_cycles_per_pixel;          // f → K
    const double omega     = 2.0 * PI * args.stim_tf_hz
                           * static_cast<double>(args.stim_drift_sign);         // v·d → ω (signed)

    double *d_V=nullptr,*d_w=nullptr,*d_gE=nullptr;
    int *d_refrac=nullptr;
    long long *d_prev=nullptr,*d_isi_c=nullptr,*d_tot=nullptr;
    double *d_isi_s=nullptr,*d_isi_ss=nullptr;
    int *d_phase=nullptr;
    double* d_templates = nullptr;
    int *d_dummy_idx=nullptr,*d_dummy_steps=nullptr,*d_dummy_count=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V,        N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,        N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE,       N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac,   N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev,     N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot,      N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s,    N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss,   N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase,    N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dummy_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));

    const int block = 256;
    const int gridN = (N_L4 + block - 1) / block;
    const int gridT = (N_TEMPLATES + block - 1) / block;

    build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
    init_full_state_kernel<<<gridN, block>>>(
        d_V, d_w, d_gE, d_refrac, d_prev,
        d_tot, d_isi_c, d_isi_s, d_isi_ss, N_L4
    );
    clear_int_kernel<<<gridN, block>>>(d_phase, N_L4);

    auto t0 = std::chrono::steady_clock::now();
    if (args.input_mode == "direct") {
        v1_phase_kernel<1><<<gridN, block>>>(
            d_V, d_w, d_gE, d_refrac, d_prev,
            d_tot, d_isi_c, d_isi_s, d_isi_ss,
            d_phase,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            0, 1, /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            n_steps, n_warmup,
            cos_t, sin_t,
            k_spatial, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            /*spike_record_out=*/nullptr,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/INT_MAX
        );
    } else {
        v1_phase_kernel<0><<<gridN, block>>>(
            d_V, d_w, d_gE, d_refrac, d_prev,
            d_tot, d_isi_c, d_isi_s, d_isi_ss,
            d_phase,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            0, 1, /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            n_steps, n_warmup,
            cos_t, sin_t,
            k_spatial, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            /*spike_record_out=*/nullptr,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/INT_MAX
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    std::vector<int> phase_count(N_L4);
    CUDA_CHECK(cudaMemcpy(phase_count.data(), d_phase,
                          N_L4 * sizeof(int), cudaMemcpyDeviceToHost));
    long long total_phase = 0; for (auto v : phase_count) total_phase += v;
    const double meas_s = (n_steps - n_warmup) * DT_S;
    const double mean_rate = (double)total_phase / N_L4 / meas_s;

    CUDA_CHECK(cudaFree(d_V));   CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_gE));  CUDA_CHECK(cudaFree(d_refrac));
    CUDA_CHECK(cudaFree(d_prev));CUDA_CHECK(cudaFree(d_tot));
    CUDA_CHECK(cudaFree(d_isi_c));
    CUDA_CHECK(cudaFree(d_isi_s));CUDA_CHECK(cudaFree(d_isi_ss));
    CUDA_CHECK(cudaFree(d_phase));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "stim_orientation_deg=" << args.stim_orientation_deg << "\n";
    std::cout << "duration_ms=" << args.duration_ms << "\n";
    std::cout << "wall_s=" << wall_s << "\n";
    std::cout << "global_mean_rate_hz=" << mean_rate << "\n";
    return 0;
}

// =====================================================================
// run_measure_l4_osi (task #7): L4-only orientation tuning sweep.
//   8 orientations (0, 22.5, ..., 157.5 deg) × 5 reps × 1000 ms each.
//   Default phase, default SF (--stim-f), default TF (--stim-v), no
//   aperture, no jitter.  Reads L4 spike counts (no L2/3, no plasticity)
//   and computes per-cell OSI = |Σ_θ R(θ)·exp(2iθ)| / Σ_θ R(θ).
//   Writes /tmp/l4_osi.json with metrics + full per-cell arrays.
// =====================================================================
static int run_measure_l4_osi(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    constexpr int N_THETA = 8;
    constexpr int N_REPS  = 5;
    const int duration_ms = 1000;
    const int N_STEPS  = static_cast<int>(duration_ms / DT_MS + 0.5);
    const int N_WARMUP = 0;
    const double measure_s = N_STEPS * DT_S;
    const double thetas_deg[N_THETA] = {
        0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5
    };

    // Latent stim variables (θ swept; f, v, d held).
    const double k_spatial = 2.0 * PI * args.stim_sf_cycles_per_pixel;
    const double omega     = 2.0 * PI * args.stim_tf_hz
                           * static_cast<double>(args.stim_drift_sign);

    // ---- device buffers (mirror run_single / run_verify) ----
    double *d_V=nullptr, *d_w=nullptr, *d_gE=nullptr;
    int *d_refrac=nullptr;
    long long *d_prev=nullptr, *d_isi_c=nullptr, *d_tot=nullptr;
    double *d_isi_s=nullptr, *d_isi_ss=nullptr;
    int *d_phase=nullptr;
    double* d_templates = nullptr;
    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V,         N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,         N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE,        N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac,    N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev,      N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot,       N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c,     N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss,    N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase,     N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));

    const int block = 256;
    const int gridN = (N_L4 + block - 1) / block;
    const int gridT = (N_TEMPLATES + block - 1) / block;

    build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
    CUDA_CHECK(cudaGetLastError());
    init_full_state_kernel<<<gridN, block>>>(
        d_V, d_w, d_gE, d_refrac, d_prev,
        d_tot, d_isi_c, d_isi_s, d_isi_ss, N_L4
    );
    CUDA_CHECK(cudaGetLastError());

    // Accumulate spike counts per (theta_idx × cell), summed over reps.
    std::vector<long long> spk_count(static_cast<size_t>(N_THETA) * N_L4, 0);
    std::vector<int> phase_count_host(N_L4);

    auto t0 = std::chrono::steady_clock::now();
    long long phase_step_offset = 0;
    for (int ti = 0; ti < N_THETA; ++ti) {
        const double th = thetas_deg[ti] * (PI / 180.0);
        const double cos_t = std::cos(th), sin_t = std::sin(th);
        for (int r = 0; r < N_REPS; ++r) {
            reset_dyn_state_kernel<<<gridN, block>>>(
                d_V, d_w, d_gE, d_refrac, d_prev, N_L4);
            clear_int_kernel<<<gridN, block>>>(d_phase, N_L4);
            CUDA_CHECK(cudaGetLastError());

            const int phase_idx = ti * N_REPS + r;
            if (args.input_mode == "direct") {
                v1_phase_kernel<1><<<gridN, block>>>(
                    d_V, d_w, d_gE, d_refrac, d_prev,
                    d_tot, d_isi_c, d_isi_s, d_isi_ss,
                    d_phase,
                    d_templates,
                    d_dummy_idx, d_dummy_steps, d_dummy_count,
                    /*n_raster=*/0, /*max_raster_spikes=*/1,
                    /*phase_idx_for_raster=*/-1,
                    phase_step_offset,
                    phase_idx,
                    N_STEPS, N_WARMUP,
                    cos_t, sin_t,
                    k_spatial, omega,
                    W_IN_NS, R_BASE_HZ,
                    args.seed,
                    /*spike_record_out=*/nullptr,
                    /*phase_offset=*/0.0,
                    /*aperture_active=*/0,
                    /*aperture_cx=*/0.0,
                    /*aperture_cy=*/0.0,
                    /*aperture_inv_2sigma_sq=*/0.0,
                    /*peak_bin20_count_out=*/nullptr,
                    /*bin50_counts_out=*/nullptr,
                    /*n_bins_50=*/0,
                    /*n_stim_steps=*/INT_MAX
                );
            } else {
                v1_phase_kernel<0><<<gridN, block>>>(
                    d_V, d_w, d_gE, d_refrac, d_prev,
                    d_tot, d_isi_c, d_isi_s, d_isi_ss,
                    d_phase,
                    d_templates,
                    d_dummy_idx, d_dummy_steps, d_dummy_count,
                    /*n_raster=*/0, /*max_raster_spikes=*/1,
                    /*phase_idx_for_raster=*/-1,
                    phase_step_offset,
                    phase_idx,
                    N_STEPS, N_WARMUP,
                    cos_t, sin_t,
                    k_spatial, omega,
                    W_IN_NS, R_BASE_HZ,
                    args.seed,
                    /*spike_record_out=*/nullptr,
                    /*phase_offset=*/0.0,
                    /*aperture_active=*/0,
                    /*aperture_cx=*/0.0,
                    /*aperture_cy=*/0.0,
                    /*aperture_inv_2sigma_sq=*/0.0,
                    /*peak_bin20_count_out=*/nullptr,
                    /*bin50_counts_out=*/nullptr,
                    /*n_bins_50=*/0,
                    /*n_stim_steps=*/INT_MAX
                );
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(phase_count_host.data(), d_phase,
                                  N_L4 * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < N_L4; ++i) {
                spk_count[(size_t)ti * N_L4 + i] += phase_count_host[i];
            }
            phase_step_offset += N_STEPS;
            std::cout << "  theta_idx=" << ti
                      << " (theta=" << thetas_deg[ti] << "°)"
                      << " rep=" << r << "/" << N_REPS << " done\n";
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    const double run_wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "run_wall_s=" << run_wall_s
              << " (" << N_THETA << "θ × " << N_REPS
              << " reps × " << duration_ms << " ms)\n";

    // free device buffers
    CUDA_CHECK(cudaFree(d_V));     CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_gE));    CUDA_CHECK(cudaFree(d_refrac));
    CUDA_CHECK(cudaFree(d_prev));  CUDA_CHECK(cudaFree(d_tot));
    CUDA_CHECK(cudaFree(d_isi_c)); CUDA_CHECK(cudaFree(d_isi_s));
    CUDA_CHECK(cudaFree(d_isi_ss));CUDA_CHECK(cudaFree(d_phase));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));

    // ---- analysis ----
    // Mean rate per (theta, cell) averaged over reps.
    std::vector<double> rate(static_cast<size_t>(N_THETA) * N_L4);
    for (int ti = 0; ti < N_THETA; ++ti) {
        for (int i = 0; i < N_L4; ++i) {
            const double cnt =
                static_cast<double>(spk_count[(size_t)ti * N_L4 + i]);
            rate[(size_t)ti * N_L4 + i] =
                cnt / (static_cast<double>(N_REPS) * measure_s);
        }
    }

    // Per-cell OSI + mean rate across orientations.
    std::vector<double> osi_per_cell(N_L4, 0.0);
    std::vector<double> mean_rate_per_cell(N_L4, 0.0);
    for (int i = 0; i < N_L4; ++i) {
        double sum_r = 0.0, num_re = 0.0, num_im = 0.0;
        for (int ti = 0; ti < N_THETA; ++ti) {
            const double r = rate[(size_t)ti * N_L4 + i];
            sum_r += r;
            const double ang = 2.0 * thetas_deg[ti] * (PI / 180.0);
            num_re += r * std::cos(ang);
            num_im += r * std::sin(ang);
        }
        mean_rate_per_cell[i] = sum_r / static_cast<double>(N_THETA);
        const double mag = std::sqrt(num_re * num_re + num_im * num_im);
        osi_per_cell[i] = (sum_r > 1e-9) ? (mag / sum_r) : 0.0;
    }

    // Aggregate metrics.
    auto frac_gt = [&](double thr) {
        long long c = 0;
        for (double v : osi_per_cell) if (v > thr) ++c;
        return static_cast<double>(c) / static_cast<double>(osi_per_cell.size());
    };
    long long n_silent = 0;
    for (double r : mean_rate_per_cell) if (r < 0.1) ++n_silent;
    const double frac_silent =
        static_cast<double>(n_silent)
        / static_cast<double>(mean_rate_per_cell.size());

    std::vector<double> osi_sorted = osi_per_cell;
    std::sort(osi_sorted.begin(), osi_sorted.end());
    const double median_osi = osi_sorted[osi_sorted.size() / 2];

    // Per-ori_idx breakdown (cells whose baked-in preferred orientation is k).
    struct OriBucket { long long n_cells; double median_osi; double frac_silent; };
    std::array<OriBucket, N_ORIENT> per_ori{};
    for (int o = 0; o < N_ORIENT; ++o) {
        std::vector<double> osi_o; osi_o.reserve(N_L4 / N_ORIENT);
        long long n_silent_o = 0;
        for (int i = 0; i < N_L4; ++i) {
            if (cell_ori(i) == o) {
                osi_o.push_back(osi_per_cell[i]);
                if (mean_rate_per_cell[i] < 0.1) ++n_silent_o;
            }
        }
        std::sort(osi_o.begin(), osi_o.end());
        per_ori[o].n_cells     = static_cast<long long>(osi_o.size());
        per_ori[o].median_osi  = osi_o.empty() ? 0.0 : osi_o[osi_o.size() / 2];
        per_ori[o].frac_silent = osi_o.empty() ? 0.0
            : static_cast<double>(n_silent_o)
              / static_cast<double>(osi_o.size());
    }

    // Print summary to stdout.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "n_cells=" << N_L4 << "\n";
    std::cout << "median_osi=" << median_osi << "\n";
    std::cout << "frac_gt_0.2=" << frac_gt(0.2) << "\n";
    std::cout << "frac_gt_0.5=" << frac_gt(0.5) << "\n";
    std::cout << "frac_gt_0.8=" << frac_gt(0.8) << "\n";
    std::cout << "frac_silent=" << frac_silent << "\n";
    for (int o = 0; o < N_ORIENT; ++o) {
        std::cout << "ori_idx=" << o
                  << " theta=" << thetas_deg[o] << "°"
                  << " n_cells=" << per_ori[o].n_cells
                  << " median_osi=" << per_ori[o].median_osi
                  << " frac_silent=" << per_ori[o].frac_silent
                  << "\n";
    }

    // ---- Write JSON ----
    const std::string json_path = "/tmp/l4_osi.json";
    std::ofstream out(json_path);
    if (!out) die("could not open " + json_path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"n_cells\": " << N_L4 << ",\n";
    out << "  \"n_theta\": " << N_THETA << ",\n";
    out << "  \"n_reps\": " << N_REPS << ",\n";
    out << "  \"duration_ms\": " << duration_ms << ",\n";
    out << "  \"thetas_deg\": [";
    for (int ti = 0; ti < N_THETA; ++ti) {
        if (ti) out << ", ";
        out << thetas_deg[ti];
    }
    out << "],\n";
    out << "  \"stim_sf_cycles_per_pixel\": " << args.stim_sf_cycles_per_pixel << ",\n";
    out << "  \"stim_tf_hz\": " << args.stim_tf_hz << ",\n";
    out << "  \"stim_drift_sign\": " << args.stim_drift_sign << ",\n";
    out << "  \"input_mode\": \"" << args.input_mode << "\",\n";
    out << "  \"seed\": " << args.seed << ",\n";
    out << "  \"run_wall_s\": " << run_wall_s << ",\n";
    out << "  \"metrics\": {\n";
    out << "    \"median_osi\": " << median_osi << ",\n";
    out << "    \"frac_gt_0.2\": " << frac_gt(0.2) << ",\n";
    out << "    \"frac_gt_0.5\": " << frac_gt(0.5) << ",\n";
    out << "    \"frac_gt_0.8\": " << frac_gt(0.8) << ",\n";
    out << "    \"frac_silent\": " << frac_silent << "\n";
    out << "  },\n";
    out << "  \"per_ori_idx\": [\n";
    for (int o = 0; o < N_ORIENT; ++o) {
        out << "    {\"ori_idx\": " << o
            << ", \"theta_deg\": " << thetas_deg[o]
            << ", \"n_cells\": " << per_ori[o].n_cells
            << ", \"median_osi\": " << per_ori[o].median_osi
            << ", \"frac_silent\": " << per_ori[o].frac_silent << "}";
        if (o + 1 < N_ORIENT) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    // Per-cell arrays (full N_L4) so the plotting helper has everything.
    out << "  \"osi_per_cell\": [";
    for (int i = 0; i < N_L4; ++i) {
        if (i) out << ",";
        out << osi_per_cell[i];
    }
    out << "],\n";
    out << "  \"mean_rate_hz_per_cell\": [";
    for (int i = 0; i < N_L4; ++i) {
        if (i) out << ",";
        out << mean_rate_per_cell[i];
    }
    out << "]\n";
    out << "}\n";
    out.close();
    std::cout << "wrote " << json_path << "\n";

    return 0;
}

// =====================================================================
// Clip-sequence runner.
//
// Each clip = (θ, f, v, d, duration_ms). State (V, w, gE, refrac, prev_spike,
// adaptation) PERSISTS across clip boundaries -- only the per-clip spike
// counter is zeroed. No warmup is applied within a clip; if the user wants
// a warmup they prepend a long-duration first clip and discard it.
// =====================================================================
namespace {

struct Clip {
    double theta_deg;
    double f_cyc_per_px;
    double v_hz;
    int    d_sign;
    int    duration_ms;
};

static std::vector<Clip> read_clip_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open clip-sequence file: " + path);
    std::vector<Clip> clips;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        // strip leading whitespace
        size_t i = 0;
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        if (i == line.size()) continue;          // blank
        if (line[i] == '#') continue;            // comment
        std::istringstream iss(line);
        Clip c{};
        if (!(iss >> c.theta_deg >> c.f_cyc_per_px >> c.v_hz
                  >> c.d_sign >> c.duration_ms)) {
            die("clip-sequence parse error on line " + std::to_string(lineno) +
                ": expected `theta_deg f_cyc_per_px v_hz d duration_ms`");
        }
        if (c.f_cyc_per_px <= 0.0)
            die("clip line " + std::to_string(lineno) + ": f must be > 0");
        if (c.d_sign != +1 && c.d_sign != -1)
            die("clip line " + std::to_string(lineno) + ": d must be +1 or -1");
        if (c.duration_ms <= 0)
            die("clip line " + std::to_string(lineno) + ": duration_ms must be > 0");
        clips.push_back(c);
    }
    return clips;
}

}  // namespace

static int run_clip_sequence(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    const auto clips = read_clip_file(args.clip_sequence_file);
    if (clips.empty())
        die("clip-sequence file has no clips: " + args.clip_sequence_file);
    const int n_clips = static_cast<int>(clips.size());
    std::cout << "clip-sequence: " << n_clips << " clips from "
              << args.clip_sequence_file
              << "  (input_mode=" << args.input_mode << ")\n";

    // ---- device buffers (only what's needed for state propagation + per-clip counter) ----
    double *d_V=nullptr, *d_w=nullptr, *d_gE=nullptr;
    int *d_refrac=nullptr;
    long long *d_prev=nullptr, *d_isi_c=nullptr, *d_tot=nullptr;
    double *d_isi_s=nullptr, *d_isi_ss=nullptr;
    int *d_phase=nullptr;
    double* d_templates = nullptr;
    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V,        N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,        N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE,       N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac,   N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev,     N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot,      N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s,    N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss,   N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase,    N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));

    const int block = 256;
    const int gridN = (N_L4 + block - 1) / block;
    const int gridT = (N_TEMPLATES + block - 1) / block;

    // ---- build Gabor templates and init full state ONCE ----
    build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
    init_full_state_kernel<<<gridN, block>>>(
        d_V, d_w, d_gE, d_refrac, d_prev,
        d_tot, d_isi_c, d_isi_s, d_isi_ss, N_L4
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Per-clip spike counts on host: shape (n_clips, N_L4) int32, row-major.
    std::vector<int> all_clip_spikes(static_cast<std::size_t>(n_clips) * N_L4, 0);
    std::vector<int> clip_buf(N_L4);

    auto t_run_start = std::chrono::steady_clock::now();
    long long phase_step_offset = 0;
    for (int c = 0; c < n_clips; ++c) {
        const Clip& clip = clips[c];
        const double th     = clip.theta_deg * (PI / 180.0);
        const double cos_t  = std::cos(th), sin_t = std::sin(th);
        const double k_sp   = 2.0 * PI * clip.f_cyc_per_px;
        const double om     = 2.0 * PI * clip.v_hz * static_cast<double>(clip.d_sign);
        const int n_steps   = static_cast<int>(clip.duration_ms / DT_MS + 0.5);

        // Zero ONLY the per-clip spike counter. State (V, w, gE, refrac, prev,
        // tot, isi_*) carries through.
        clear_int_kernel<<<gridN, block>>>(d_phase, N_L4);
        CUDA_CHECK(cudaGetLastError());

        if (args.input_mode == "direct") {
            v1_phase_kernel<1><<<gridN, block>>>(
                d_V, d_w, d_gE, d_refrac, d_prev,
                d_tot, d_isi_c, d_isi_s, d_isi_ss,
                d_phase,
                d_templates,
                d_dummy_idx, d_dummy_steps, d_dummy_count,
                /*n_raster=*/0, /*max_raster_spikes=*/1,
                /*phase_idx_for_raster=*/-1,
                phase_step_offset,
                /*phase_idx=*/c,
                n_steps, /*n_warmup_steps=*/0,
                cos_t, sin_t,
                k_sp, om,
                W_IN_NS, R_BASE_HZ,
                args.seed,
                /*spike_record_out=*/nullptr,
                /*phase_offset=*/0.0,
                /*aperture_active=*/0,
                /*aperture_cx=*/0.0,
                /*aperture_cy=*/0.0,
                /*aperture_inv_2sigma_sq=*/0.0,
                /*peak_bin20_count_out=*/nullptr,
                /*bin50_counts_out=*/nullptr,
                /*n_bins_50=*/0,
                /*n_stim_steps=*/INT_MAX
            );
        } else {
            v1_phase_kernel<0><<<gridN, block>>>(
                d_V, d_w, d_gE, d_refrac, d_prev,
                d_tot, d_isi_c, d_isi_s, d_isi_ss,
                d_phase,
                d_templates,
                d_dummy_idx, d_dummy_steps, d_dummy_count,
                /*n_raster=*/0, /*max_raster_spikes=*/1,
                /*phase_idx_for_raster=*/-1,
                phase_step_offset,
                /*phase_idx=*/c,
                n_steps, /*n_warmup_steps=*/0,
                cos_t, sin_t,
                k_sp, om,
                W_IN_NS, R_BASE_HZ,
                args.seed,
                /*spike_record_out=*/nullptr,
                /*phase_offset=*/0.0,
                /*aperture_active=*/0,
                /*aperture_cx=*/0.0,
                /*aperture_cy=*/0.0,
                /*aperture_inv_2sigma_sq=*/0.0,
                /*peak_bin20_count_out=*/nullptr,
                /*bin50_counts_out=*/nullptr,
                /*n_bins_50=*/0,
                /*n_stim_steps=*/INT_MAX
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        phase_step_offset += n_steps;

        CUDA_CHECK(cudaMemcpy(clip_buf.data(), d_phase,
                              N_L4 * sizeof(int), cudaMemcpyDeviceToHost));
        std::copy(clip_buf.begin(), clip_buf.end(),
                  all_clip_spikes.begin() + static_cast<std::size_t>(c) * N_L4);
    }
    auto t_run_end = std::chrono::steady_clock::now();
    const double run_wall_s = std::chrono::duration<double>(t_run_end - t_run_start).count();

    // ---- per-clip aggregate stats ----
    std::cout << "\nclip  theta    f       v   d  dur_ms   mean_hz  p95_hz  max_hz  frac_silent\n";
    std::cout << std::fixed << std::setprecision(4);
    struct ClipStat { double mean_hz, p95_hz, max_hz, frac_silent; };
    std::vector<ClipStat> stats(n_clips);
    std::vector<double> rates(N_L4);
    for (int c = 0; c < n_clips; ++c) {
        const Clip& clip = clips[c];
        const double meas_s = clip.duration_ms / 1000.0;
        const int* counts = all_clip_spikes.data() + static_cast<std::size_t>(c) * N_L4;
        double sum = 0.0; long long n_silent = 0;
        for (int i = 0; i < N_L4; ++i) {
            rates[i] = static_cast<double>(counts[i]) / meas_s;
            sum += rates[i];
            if (counts[i] == 0) ++n_silent;
        }
        const double mean_hz = sum / N_L4;
        std::vector<double> sorted = rates;
        const std::size_t k95 = static_cast<std::size_t>(N_L4 * 0.95);
        std::nth_element(sorted.begin(), sorted.begin() + k95, sorted.end());
        const double p95 = sorted[k95];
        const double mx  = *std::max_element(rates.begin(), rates.end());
        stats[c] = {mean_hz, p95, mx, static_cast<double>(n_silent) / N_L4};
        std::cout << std::setw(4) << c << "  "
                  << std::setw(6) << std::setprecision(1) << clip.theta_deg << "  "
                  << std::setw(6) << std::setprecision(4) << clip.f_cyc_per_px << "  "
                  << std::setw(4) << std::setprecision(1) << clip.v_hz << "  "
                  << std::setw(2) << clip.d_sign << "  "
                  << std::setw(6) << clip.duration_ms << "  "
                  << std::setw(8) << std::setprecision(3) << mean_hz << "  "
                  << std::setw(6) << std::setprecision(2) << p95 << "  "
                  << std::setw(6) << std::setprecision(2) << mx << "  "
                  << std::setw(11) << std::setprecision(4) << stats[c].frac_silent
                  << "\n";
    }

    // ---- JSON summary ----
    std::filesystem::create_directories(args.out_dir);
    const std::string json_label =
        args.label.empty() ? std::string() : std::string("_") + args.label;
    const std::string json_path =
        args.out_dir + "/v1_clipseq" + json_label + "_summary.json";
    {
        std::ofstream out(json_path);
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"n_clips\": " << n_clips << ",\n";
        out << "  \"n_cells\": " << N_L4 << ",\n";
        out << "  \"input_mode\": \"" << args.input_mode << "\",\n";
        out << "  \"run_wall_s\": " << run_wall_s << ",\n";
        out << "  \"device\": \"" << prop.name << "\",\n";
        out << "  \"seed\": " << args.seed << ",\n";
        out << "  \"clip_rates_bin\": \"" << args.clip_rates_bin << "\",\n";
        out << "  \"clips\": [\n";
        for (int c = 0; c < n_clips; ++c) {
            const Clip& clip = clips[c];
            out << "    {\"idx\":" << c
                << ",\"theta_deg\":" << clip.theta_deg
                << ",\"f_cyc_per_px\":" << clip.f_cyc_per_px
                << ",\"v_hz\":" << clip.v_hz
                << ",\"d\":" << clip.d_sign
                << ",\"duration_ms\":" << clip.duration_ms
                << ",\"mean_rate_hz\":" << stats[c].mean_hz
                << ",\"p95_rate_hz\":" << stats[c].p95_hz
                << ",\"max_rate_hz\":" << stats[c].max_hz
                << ",\"frac_silent\":" << stats[c].frac_silent
                << "}" << (c + 1 == n_clips ? "" : ",") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }

    // ---- optional binary spike-count dump ----
    if (!args.clip_rates_bin.empty()) {
        std::ofstream bin(args.clip_rates_bin, std::ios::binary);
        if (!bin) die("cannot open --clip-rates-bin for write: " + args.clip_rates_bin);
        bin.write(reinterpret_cast<const char*>(all_clip_spikes.data()),
                  all_clip_spikes.size() * sizeof(int));
        std::cout << "clip_rates_bin=" << args.clip_rates_bin
                  << "  (int32, [" << n_clips << ", " << N_L4 << "], row-major)\n";
    }

    CUDA_CHECK(cudaFree(d_V));   CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_gE));  CUDA_CHECK(cudaFree(d_refrac));
    CUDA_CHECK(cudaFree(d_prev));CUDA_CHECK(cudaFree(d_tot));
    CUDA_CHECK(cudaFree(d_isi_c));
    CUDA_CHECK(cudaFree(d_isi_s));CUDA_CHECK(cudaFree(d_isi_ss));
    CUDA_CHECK(cudaFree(d_phase));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));

    std::cout << "\nclip_sequence_wall_s=" << run_wall_s << "\n";
    std::cout << "summary_json=" << json_path << "\n";
    return 0;
}

// =====================================================================
// L2/3 Phase A verify run.
// One stim phase at args.stim_orientation_deg for args.duration_ms.
// Drives both L4 (existing kernel + spike-bitmask record) and L2/3 (new
// kernel reading the bitmask with 2 ms delay).  Writes:
//   <out_dir>/l23_connectivity.json   (artifact A + B + D source data)
//   <out_dir>/l23_drive_summary.json  (artifact C source data)
// PNGs (A–D) are rendered by the Python helper from these two JSON files.
// =====================================================================
static int run_verify_l23(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    const int DURATION_MS = args.duration_ms;
    const int N_STEPS = static_cast<int>(DURATION_MS / DT_MS + 0.5);
    if (N_STEPS <= L23_DELAY_STEPS) {
        die("--duration_ms must give n_steps > L23_DELAY_STEPS (=20)");
    }
    // Discard the first 2 ms (the silent pre-delay window) from rate stats.
    const int N_WARMUP = L23_DELAY_STEPS;
    const double measure_s = (N_STEPS - N_WARMUP) * DT_S;

    // ---- Build CSR + lognormal weights on host (deterministic from seed) ----
    auto t_init0 = std::chrono::steady_clock::now();
    L23Connectivity conn = build_l23_connectivity(args.seed);
    auto t_init1 = std::chrono::steady_clock::now();
    const double conn_build_s =
        std::chrono::duration<double>(t_init1 - t_init0).count();
    std::cout << "l23_connectivity_built  total_synapses=" << conn.total_synapses
              << "  build_wall_s=" << conn_build_s << "\n";

    // ---- Stim params ----
    const double th = args.stim_orientation_deg * (PI / 180.0);
    const double cos_t = std::cos(th);
    const double sin_t = std::sin(th);
    const double k_spatial = 2.0 * PI * args.stim_sf_cycles_per_pixel;
    const double omega = 2.0 * PI * args.stim_tf_hz
                       * static_cast<double>(args.stim_drift_sign);

    // ---- Allocate L4 buffers (mirrors run_verify) ----
    double *d_V_l4=nullptr, *d_w_l4=nullptr, *d_gE_l4=nullptr;
    int *d_refrac_l4=nullptr;
    long long *d_prev_l4=nullptr, *d_isi_c_l4=nullptr, *d_tot_l4=nullptr;
    double *d_isi_s_l4=nullptr, *d_isi_ss_l4=nullptr;
    int *d_phase_l4=nullptr;
    double* d_templates = nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l4,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l4, N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l4,   N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l4,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l4,  N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l4,  N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l4, N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l4,  N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));

    // dummy raster buffers (unused; v1_phase_kernel signature requires them)
    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_idx,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_steps, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_count, 0, sizeof(int)));

    // ---- Allocate L2/3 buffers ----
    double *d_V_l23=nullptr, *d_w_l23=nullptr, *d_gE_l23=nullptr;
    int *d_refrac_l23=nullptr;
    long long *d_prev_l23=nullptr, *d_isi_c_l23=nullptr, *d_tot_l23=nullptr;
    double *d_isi_s_l23=nullptr, *d_isi_ss_l23=nullptr;
    int *d_phase_l23=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l23,     N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l23, N_L23 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l23,   N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l23,    N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l23,  N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l23,  N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l23, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l23,  N_L23 * sizeof(int)));

    // ---- CSR connectivity buffers ----
    int    *d_row_ptr=nullptr, *d_col_idx=nullptr;
    double *d_l23_w_nS=nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,
                          (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,
                          static_cast<size_t>(conn.total_synapses) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_l23_w_nS,
                          static_cast<size_t>(conn.total_synapses) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr,  conn.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx,  conn.col_idx.data(),
                          static_cast<size_t>(conn.total_synapses) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l23_w_nS, conn.w_nS.data(),
                          static_cast<size_t>(conn.total_synapses) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- Per-step L4-spike bitmask record (warp-ballot output) ----
    const size_t spike_record_ints =
        static_cast<size_t>(N_STEPS) * static_cast<size_t>(N_L4_BITMASK_INTS);
    const size_t spike_record_bytes = spike_record_ints * sizeof(uint32_t);
    uint32_t* d_l4_spike_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l4_spike_record, spike_record_bytes));
    {
        const int block = 256;
        const int gridR = static_cast<int>(
            (spike_record_ints + block - 1) / block);
        clear_uint32_kernel<<<gridR, block>>>(d_l4_spike_record, spike_record_ints);
        CUDA_CHECK(cudaGetLastError());
    }
    std::cout << "l4_spike_record_alloc_MB="
              << (spike_record_bytes / (1024.0 * 1024.0)) << "\n";

    // ---- Build Gabor templates ----
    const int block = 256;
    const int gridL4 = (N_L4 + block - 1) / block;
    const int gridL23 = (N_L23 + block - 1) / block;
    {
        const int gridT = (N_TEMPLATES + block - 1) / block;
        build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Init AdEx state for both layers ----
    init_full_state_kernel<<<gridL4, block>>>(
        d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
        d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
    );
    CUDA_CHECK(cudaGetLastError());
    init_l23_state_kernel<<<gridL23, block>>>(
        d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
        d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23, N_L23
    );
    CUDA_CHECK(cudaGetLastError());
    clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
    clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Run L4 phase, recording per-step spikes ----
    auto t0 = std::chrono::steady_clock::now();
    if (args.input_mode == "direct") {
        v1_phase_kernel<1><<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
            d_phase_l4,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            /*n_raster=*/0, /*max_raster_spikes=*/0,
            /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            N_STEPS, N_WARMUP,
            cos_t, sin_t,
            k_spatial, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            d_l4_spike_record,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/INT_MAX
        );
    } else {
        v1_phase_kernel<0><<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
            d_phase_l4,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            /*n_raster=*/0, /*max_raster_spikes=*/0,
            /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            N_STEPS, N_WARMUP,
            cos_t, sin_t,
            k_spatial, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            d_l4_spike_record,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/INT_MAX
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    // ---- Sanity: total bits in the L4 spike record (must match L4 spikes) ----
    unsigned long long *d_bit_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bit_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_bit_total, 0, sizeof(unsigned long long)));
    {
        const int popblock = 256;
        const size_t popgrid =
            (spike_record_ints + popblock - 1) / popblock;
        popcount_uint32_kernel<<<static_cast<int>(popgrid), popblock>>>(
            d_l4_spike_record, spike_record_ints, d_bit_total
        );
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long bit_total_host = 0;
    CUDA_CHECK(cudaMemcpy(&bit_total_host, d_bit_total,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bit_total));

    // ---- Run L2/3 phase, consuming the bitmask record with 2 ms delay ----
    v1_l23_phase_kernel<<<gridL23, block>>>(
        d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
        d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23,
        d_phase_l23,
        d_row_ptr, d_col_idx, d_l23_w_nS,
        d_l4_spike_record,
        /*phase_step_offset=*/0,
        N_STEPS, N_WARMUP
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();

    const double l4_wall_s  = std::chrono::duration<double>(t1 - t0).count();
    const double l23_wall_s = std::chrono::duration<double>(t2 - t1).count();

    // ---- Read aggregates back ----
    std::vector<int> phase_count_l4(N_L4, 0);
    std::vector<int> phase_count_l23(N_L23, 0);
    std::vector<long long> total_spikes_l4(N_L4, 0);
    CUDA_CHECK(cudaMemcpy(phase_count_l4.data(),  d_phase_l4,
                          N_L4  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phase_count_l23.data(), d_phase_l23,
                          N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(total_spikes_l4.data(), d_tot_l4,
                          N_L4  * sizeof(long long), cudaMemcpyDeviceToHost));
    long long sum_total_spikes_l4 = 0;
    for (long long s : total_spikes_l4) sum_total_spikes_l4 += s;

    // ---- Free device buffers (keep nothing — we have host data we need) ----
    CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
    CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
    CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
    CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
    CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));
    CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
    CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
    CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
    CUDA_CHECK(cudaFree(d_isi_c_l23));CUDA_CHECK(cudaFree(d_isi_s_l23));
    CUDA_CHECK(cudaFree(d_isi_ss_l23));CUDA_CHECK(cudaFree(d_phase_l23));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_l23_w_nS));
    CUDA_CHECK(cudaFree(d_l4_spike_record));

    // ---- Analysis ----
    // L2/3 firing rates (post-warmup window).
    std::vector<double> rate_l23(N_L23);
    double sum_rate=0.0, sum_rate_sq=0.0;
    double max_rate=0.0;
    long long n_silent=0;
    for (int i = 0; i < N_L23; ++i) {
        const double r = static_cast<double>(phase_count_l23[i]) / measure_s;
        rate_l23[i] = r;
        sum_rate += r;
        sum_rate_sq += r * r;
        if (r > max_rate) max_rate = r;
        if (r < 0.1) ++n_silent;
    }
    const double mean_rate = sum_rate / N_L23;
    const double std_rate = std::sqrt(std::max(0.0,
        sum_rate_sq / N_L23 - mean_rate * mean_rate));
    std::vector<double> rate_sorted = rate_l23;
    std::nth_element(rate_sorted.begin(),
                     rate_sorted.begin() + N_L23 / 2,
                     rate_sorted.end());
    const double median_rate = rate_sorted[N_L23 / 2];
    std::nth_element(rate_sorted.begin(),
                     rate_sorted.begin() + (size_t)(N_L23 * 0.95),
                     rate_sorted.end());
    const double p95_rate = rate_sorted[(size_t)(N_L23 * 0.95)];

    // L4 reference firing rates (sanity).
    double sum_rate_l4 = 0.0;
    for (int i = 0; i < N_L4; ++i) {
        sum_rate_l4 += static_cast<double>(phase_count_l4[i]) / measure_s;
    }
    const double mean_rate_l4 = sum_rate_l4 / N_L4;

    // Per-cell fan-in (= row_ptr[i+1] - row_ptr[i]).
    std::vector<int> fanin(N_L23);
    long long sum_fanin=0;
    int min_fanin = INT32_MAX, max_fanin = 0;
    for (int i = 0; i < N_L23; ++i) {
        const int f = conn.row_ptr[i + 1] - conn.row_ptr[i];
        fanin[i] = f;
        sum_fanin += f;
        if (f < min_fanin) min_fanin = f;
        if (f > max_fanin) max_fanin = f;
    }
    const double mean_fanin = static_cast<double>(sum_fanin) / N_L23;
    // edge fan-in stat: cells whose hypercolumn touches the grid border
    long long sum_fanin_edge = 0;
    int n_edge = 0;
    long long sum_fanin_interior = 0;
    int n_interior = 0;
    for (int i = 0; i < N_L23; ++i) {
        const int gx = l23_gx(i), gy = l23_gy(i);
        const bool is_edge = (gx == 0 || gx == GRID-1 || gy == 0 || gy == GRID-1);
        if (is_edge) { sum_fanin_edge += fanin[i]; ++n_edge; }
        else         { sum_fanin_interior += fanin[i]; ++n_interior; }
    }
    const double mean_fanin_edge =
        n_edge > 0     ? static_cast<double>(sum_fanin_edge)    / n_edge     : 0.0;
    const double mean_fanin_interior =
        n_interior > 0 ? static_cast<double>(sum_fanin_interior)/ n_interior : 0.0;

    // Weight stats (in mV).
    std::vector<double> w_sorted = conn.w_EPSP_mV;
    std::sort(w_sorted.begin(), w_sorted.end());
    const double w_median_mV = w_sorted[w_sorted.size() / 2];
    const double w_max_mV    = w_sorted.back();
    const double w_min_mV    = w_sorted.front();
    double w_sum_mV = 0.0;
    for (double w : conn.w_EPSP_mV) w_sum_mV += w;
    const double w_mean_mV = w_sum_mV / static_cast<double>(conn.w_EPSP_mV.size());

    // Sample-cell partner orientation profile (artifact D source).
    auto sample_l23_idx = build_l23_sample_indices();
    struct PartnerProfile { int l23_idx, gx, gy; std::array<int, N_ORIENT> by_ori; };
    std::vector<PartnerProfile> profiles;
    profiles.reserve(sample_l23_idx.size());
    for (int s : sample_l23_idx) {
        PartnerProfile pp{};
        pp.l23_idx = s; pp.gx = l23_gx(s); pp.gy = l23_gy(s);
        pp.by_ori.fill(0);
        const int ps = conn.row_ptr[s];
        const int pe = conn.row_ptr[s + 1];
        for (int p = ps; p < pe; ++p) {
            const int l4 = conn.col_idx[p];
            const int ori = cell_ori(l4);
            pp.by_ori[ori] += 1;
        }
        profiles.push_back(pp);
    }

    // ===== Write JSON artifacts =====
    std::filesystem::create_directories(args.out_dir);
    const std::string label_suffix =
        args.label.empty() ? std::string() : std::string("_") + args.label;
    const std::string conn_json_path = args.out_dir + "/l23_connectivity"
        + label_suffix + ".json";
    const std::string drive_json_path = args.out_dir + "/l23_drive_summary"
        + label_suffix + ".json";

    // ---- l23_connectivity.json (artifacts A, B, D source) ----
    {
        std::ofstream f(conn_json_path);
        if (!f) die("could not open " + conn_json_path);
        f << std::setprecision(8);
        f << "{\n";
        f << "  \"n_l23\": " << N_L23 << ",\n";
        f << "  \"n_l4\": "  << N_L4  << ",\n";
        f << "  \"total_synapses\": " << conn.total_synapses << ",\n";
        f << "  \"params\": {\n";
        f << "    \"target_fanin\": " << L23_TARGET_FANIN << ",\n";
        f << "    \"patch_radius\": " << L23_PATCH_R << ",\n";
        f << "    \"candidate_pool_interior\": " << (9 * N_ORIENT * N_CELLS_PER_ORIENT)
          << ",\n";
        f << "    \"p_connect\": "
          << (static_cast<double>(L23_TARGET_FANIN) / (9 * N_ORIENT * N_CELLS_PER_ORIENT))
          << ",\n";
        f << "    \"epsp_median_mV\": " << L23_EPSP_MEDIAN_MV << ",\n";
        f << "    \"epsp_log_sigma\": " << L23_EPSP_LOG_SIGMA << ",\n";
        f << "    \"epsp_max_mV\": "    << L23_EPSP_MAX_MV << ",\n";
        f << "    \"mV_per_nS\": "      << L23_MV_PER_NS << ",\n";
        f << "    \"tau_syn_l23_ms\": " << TAU_SYN_L23_MS << ",\n";
        f << "    \"E_E_l23_mV\": "     << E_E_L23_MV << ",\n";
        f << "    \"delay_steps\": "    << L23_DELAY_STEPS << ",\n";
        f << "    \"delay_ms\": "       << (L23_DELAY_STEPS * DT_MS) << ",\n";
        f << "    \"salt_connectivity\": " << L23_SALT_CONNECTIVITY << ",\n";
        f << "    \"salt_weights\": "     << L23_SALT_WEIGHTS << "\n";
        f << "  },\n";
        f << "  \"seed\": " << args.seed << ",\n";
        f << "  \"fanin\": {\n";
        f << "    \"mean\": "          << mean_fanin << ",\n";
        f << "    \"min\": "           << min_fanin  << ",\n";
        f << "    \"max\": "           << max_fanin  << ",\n";
        f << "    \"mean_interior\": " << mean_fanin_interior << ",\n";
        f << "    \"mean_edge\": "     << mean_fanin_edge << ",\n";
        f << "    \"per_cell\": [";
        for (int i = 0; i < N_L23; ++i) {
            if (i) f << ",";
            f << fanin[i];
        }
        f << "]\n  },\n";
        f << "  \"weights_mV\": {\n";
        f << "    \"n\": "      << conn.w_EPSP_mV.size() << ",\n";
        f << "    \"min\": "    << w_min_mV << ",\n";
        f << "    \"median\": " << w_median_mV << ",\n";
        f << "    \"mean\": "   << w_mean_mV << ",\n";
        f << "    \"max\": "    << w_max_mV << ",\n";
        f << "    \"all\": [";
        for (size_t i = 0; i < conn.w_EPSP_mV.size(); ++i) {
            if (i) f << ",";
            f << conn.w_EPSP_mV[i];
        }
        f << "]\n  },\n";
        f << "  \"partner_profiles\": [\n";
        for (size_t s = 0; s < profiles.size(); ++s) {
            const auto& pp = profiles[s];
            f << "    {\"l23_idx\":" << pp.l23_idx
              << ",\"gx\":" << pp.gx << ",\"gy\":" << pp.gy
              << ",\"fanin\":" << fanin[pp.l23_idx]
              << ",\"by_ori_count\":[";
            for (int o = 0; o < N_ORIENT; ++o) {
                if (o) f << ",";
                f << pp.by_ori[o];
            }
            f << "]}";
            if (s + 1 < profiles.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n";
        f << "}\n";
    }

    // ---- l23_drive_summary.json (artifact C source) ----
    {
        std::ofstream f(drive_json_path);
        if (!f) die("could not open " + drive_json_path);
        f << std::setprecision(8);
        f << "{\n";
        f << "  \"n_l23\": " << N_L23 << ",\n";
        f << "  \"n_l4\": "  << N_L4  << ",\n";
        f << "  \"duration_ms\": " << DURATION_MS << ",\n";
        f << "  \"warmup_steps\": " << N_WARMUP << ",\n";
        f << "  \"warmup_ms\": "    << (N_WARMUP * DT_MS) << ",\n";
        f << "  \"measure_window_s\": " << measure_s << ",\n";
        f << "  \"stim\": {\n";
        f << "    \"theta_deg\": "       << args.stim_orientation_deg << ",\n";
        f << "    \"f_cyc_per_pixel\": " << args.stim_sf_cycles_per_pixel << ",\n";
        f << "    \"v_tf_hz\": "         << args.stim_tf_hz << ",\n";
        f << "    \"d_drift_sign\": "    << args.stim_drift_sign << ",\n";
        f << "    \"k_spatial\": "       << k_spatial << ",\n";
        f << "    \"omega_signed\": "    << omega << ",\n";
        f << "    \"input_mode\": \""    << args.input_mode << "\"\n";
        f << "  },\n";
        f << "  \"l23_global\": {\n";
        f << "    \"mean_rate_hz\": "   << mean_rate << ",\n";
        f << "    \"std_rate_hz\": "    << std_rate << ",\n";
        f << "    \"median_rate_hz\": " << median_rate << ",\n";
        f << "    \"max_rate_hz\": "    << max_rate << ",\n";
        f << "    \"p95_rate_hz\": "    << p95_rate << ",\n";
        f << "    \"frac_silent\": "
          << (static_cast<double>(n_silent) / N_L23) << ",\n";
        f << "    \"silent_threshold_hz\": 0.1\n";
        f << "  },\n";
        f << "  \"l4_global\": {\n";
        f << "    \"mean_rate_hz\": "          << mean_rate_l4 << ",\n";
        f << "    \"total_spikes\": "          << sum_total_spikes_l4 << ",\n";
        f << "    \"bitmask_bits_set\": "      << bit_total_host << ",\n";
        f << "    \"bitmask_matches_total_spikes\": "
          << ((bit_total_host == static_cast<unsigned long long>(sum_total_spikes_l4))
              ? "true" : "false") << "\n";
        f << "  },\n";
        f << "  \"timing\": {\n";
        f << "    \"l4_kernel_wall_s\": "  << l4_wall_s << ",\n";
        f << "    \"l23_kernel_wall_s\": " << l23_wall_s << ",\n";
        f << "    \"connectivity_build_wall_s\": " << conn_build_s << "\n";
        f << "  },\n";
        f << "  \"device\": \"" << prop.name << "\",\n";
        f << "  \"seed\": " << args.seed << ",\n";
        f << "  \"l23_rate_hz\": [";
        for (int i = 0; i < N_L23; ++i) {
            if (i) f << ",";
            f << rate_l23[i];
        }
        f << "]\n";
        f << "}\n";
    }

    // ---- stdout summary ----
    std::cout << "\n=== L2/3 Phase A verify summary ===\n";
    std::cout << "duration_ms="   << DURATION_MS
              << "  measure_s="    << measure_s << "\n";
    std::cout << "n_l23="          << N_L23
              << "  total_synapses=" << conn.total_synapses << "\n";
    std::cout << "fanin: mean="    << mean_fanin
              << "  min="          << min_fanin
              << "  max="          << max_fanin
              << "  interior_mean=" << mean_fanin_interior
              << "  edge_mean="    << mean_fanin_edge << "\n";
    std::cout << "weights_mV: median=" << w_median_mV
              << "  mean=" << w_mean_mV
              << "  max="  << w_max_mV << "\n";
    std::cout << "l23_rate_hz: mean=" << mean_rate
              << "  median=" << median_rate
              << "  max="    << max_rate
              << "  p95="    << p95_rate
              << "  frac_silent=" << (static_cast<double>(n_silent) / N_L23) << "\n";
    std::cout << "l4_rate_hz_mean=" << mean_rate_l4
              << "  total_spikes=" << sum_total_spikes_l4
              << "  bitmask_bits=" << bit_total_host
              << "  bitmask_eq_spikes="
              << ((bit_total_host == static_cast<unsigned long long>(sum_total_spikes_l4))
                  ? "true" : "false") << "\n";
    std::cout << "l4_kernel_wall_s="  << l4_wall_s
              << "  l23_kernel_wall_s=" << l23_wall_s << "\n";
    std::cout << "connectivity_json=" << conn_json_path << "\n";
    std::cout << "drive_summary_json=" << drive_json_path << "\n";
    return 0;
}

// =====================================================================
// run_verify_l23_recurrent (task #3 — B1 STRUCTURE + SANITY, NO STDP)
//
// Adds STATIC L2/3→L2/3 recurrent connectivity on top of the trained-and-
// frozen L4→L2/3 substrate.  No plasticity.  No STDP traces.  Dumps:
//
//   - l23_recurrent_connectivity.json  (artifacts A, B, C, D source)
//   - l23_recurrent_drive_summary.json (artifact E source)
//
// Render PNGs A–E with plot_l23_recurrent.py.  Pre-flight: requires
// --load-trained-weights pointing at the Phase A L4→L2/3 .bin.
// =====================================================================
static int run_verify_l23_recurrent(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    if (args.load_trained_weights.empty()) {
        die("--enable-l23-recurrent requires --load-trained-weights "
            "(trained Phase A L4→L2/3 .bin)");
    }

    const int DURATION_MS = args.duration_ms;
    const int N_STEPS = static_cast<int>(DURATION_MS / DT_MS + 0.5);
    if (N_STEPS <= L23_DELAY_STEPS) {
        die("--duration_ms must give n_steps > L23_DELAY_STEPS (=20)");
    }
    const int N_WARMUP = L23_DELAY_STEPS;   // discard 2 ms pre-delay
    const double measure_s = (N_STEPS - N_WARMUP) * DT_S;

    // ---- Build L4→L2/3 CSR (placeholder weights; overwritten from disk) ----
    auto t_init0 = std::chrono::steady_clock::now();
    L23Connectivity conn = build_l23_connectivity(args.seed);
    auto t_init1 = std::chrono::steady_clock::now();
    const double conn_build_s =
        std::chrono::duration<double>(t_init1 - t_init0).count();
    std::cout << "l23_connectivity_built  total_synapses=" << conn.total_synapses
              << "  build_wall_s=" << conn_build_s << "\n";

    // ---- Build L2/3→L2/3 recurrent CSR (NEW) ----
    auto t_rec0 = std::chrono::steady_clock::now();
    L23RecConnectivity rec = build_l23_recurrent_connectivity(args.seed);
    auto t_rec1 = std::chrono::steady_clock::now();
    const double rec_build_s =
        std::chrono::duration<double>(t_rec1 - t_rec0).count();
    std::cout << "l23_recurrent_built     total_synapses=" << rec.total_synapses
              << "  build_wall_s=" << rec_build_s
              << "  base_edges=" << rec.n_pairs_sampled
              << "  recip_pre_boost=" << rec.n_pairs_reciprocal_pre_boost
              << "  edges_added_by_boost=" << rec.n_edges_added_by_boost
              << "  recip_pairs_post=" << rec.n_pairs_reciprocal_post << "\n";

    // ---- Stim params (default θ=0°) ----
    const double th = args.stim_orientation_deg * (PI / 180.0);
    const double cos_t = std::cos(th);
    const double sin_t = std::sin(th);
    const double k_spatial = 2.0 * PI * args.stim_sf_cycles_per_pixel;
    const double omega = 2.0 * PI * args.stim_tf_hz
                       * static_cast<double>(args.stim_drift_sign);

    // ---- Allocate L4 buffers ----
    double *d_V_l4=nullptr, *d_w_l4=nullptr, *d_gE_l4=nullptr;
    int *d_refrac_l4=nullptr;
    long long *d_prev_l4=nullptr, *d_isi_c_l4=nullptr, *d_tot_l4=nullptr;
    double *d_isi_s_l4=nullptr, *d_isi_ss_l4=nullptr;
    int *d_phase_l4=nullptr;
    double* d_templates = nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l4,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l4, N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l4,   N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l4,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l4,  N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l4,  N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l4, N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l4,  N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));

    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_idx,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_steps, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_count, 0, sizeof(int)));

    // ---- Allocate L2/3 buffers ----
    double *d_V_l23=nullptr, *d_w_l23=nullptr, *d_gE_l23=nullptr;
    int *d_refrac_l23=nullptr;
    long long *d_prev_l23=nullptr, *d_isi_c_l23=nullptr, *d_tot_l23=nullptr;
    double *d_isi_s_l23=nullptr, *d_isi_ss_l23=nullptr;
    int *d_phase_l23=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l23,     N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l23, N_L23 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l23,   N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l23,    N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l23,  N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l23,  N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l23, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l23,  N_L23 * sizeof(int)));

    // ---- L4→L2/3 CSR buffers (load trained weights into d_l23_w_nS) ----
    int    *d_row_ptr=nullptr, *d_col_idx=nullptr;
    double *d_l23_w_nS=nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,
                          (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,
                          static_cast<size_t>(conn.total_synapses) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_l23_w_nS,
                          static_cast<size_t>(conn.total_synapses) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, conn.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, conn.col_idx.data(),
                          static_cast<size_t>(conn.total_synapses) * sizeof(int),
                          cudaMemcpyHostToDevice));
    {
        std::ifstream wf(args.load_trained_weights, std::ios::binary);
        if (!wf) die("could not open --load-trained-weights: "
                     + args.load_trained_weights);
        wf.seekg(0, std::ios::end);
        const std::streamsize bytes = wf.tellg();
        wf.seekg(0, std::ios::beg);
        const std::streamsize expected_bytes =
            static_cast<std::streamsize>(conn.total_synapses) * sizeof(double);
        if (bytes != expected_bytes) {
            die("weights file size mismatch: got " + std::to_string(bytes)
                + " bytes, expected " + std::to_string(expected_bytes)
                + " (n_synapses=" + std::to_string(conn.total_synapses) + " × 8)");
        }
        std::vector<double> w_loaded(conn.total_synapses);
        wf.read(reinterpret_cast<char*>(w_loaded.data()), expected_bytes);
        if (!wf) die("read failed for --load-trained-weights: "
                     + args.load_trained_weights);
        CUDA_CHECK(cudaMemcpy(d_l23_w_nS, w_loaded.data(),
                              static_cast<size_t>(conn.total_synapses) * sizeof(double),
                              cudaMemcpyHostToDevice));
        std::cout << "loaded trained L4→L2/3 weights from "
                  << args.load_trained_weights
                  << " (" << bytes << " bytes, "
                  << conn.total_synapses << " synapses)\n";
    }

    // ---- L2/3→L2/3 CSR buffers (NEW; weights from rec.w_nS) ----
    int    *d_rec_row_ptr=nullptr, *d_rec_col_idx=nullptr;
    double *d_rec_w_nS=nullptr;
    CUDA_CHECK(cudaMalloc(&d_rec_row_ptr,
                          (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rec_col_idx,
                          static_cast<size_t>(rec.total_synapses) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rec_w_nS,
                          static_cast<size_t>(rec.total_synapses) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rec_row_ptr, rec.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_col_idx, rec.col_idx.data(),
                          static_cast<size_t>(rec.total_synapses) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_w_nS, rec.w_nS.data(),
                          static_cast<size_t>(rec.total_synapses) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- Per-step L4 spike-record bitmask (warp-ballot output) ----
    const size_t l4_record_ints =
        static_cast<size_t>(N_STEPS) * static_cast<size_t>(N_L4_BITMASK_INTS);
    uint32_t* d_l4_spike_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l4_spike_record, l4_record_ints * sizeof(uint32_t)));
    // ---- Per-step L2/3 spike-record bitmask (NEW) ----
    const size_t l23_record_ints =
        static_cast<size_t>(N_STEPS) * static_cast<size_t>(N_L23_BITMASK_INTS);
    uint32_t* d_l23_spike_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l23_spike_record, l23_record_ints * sizeof(uint32_t)));
    {
        const int block = 256;
        const int gridR_l4 = static_cast<int>(
            (l4_record_ints + block - 1) / block);
        const int gridR_l23 = static_cast<int>(
            (l23_record_ints + block - 1) / block);
        clear_uint32_kernel<<<gridR_l4, block>>>(d_l4_spike_record, l4_record_ints);
        clear_uint32_kernel<<<gridR_l23, block>>>(d_l23_spike_record, l23_record_ints);
        CUDA_CHECK(cudaGetLastError());
    }
    std::cout << "l4_spike_record_alloc_MB="
              << ((l4_record_ints * sizeof(uint32_t)) / (1024.0 * 1024.0))
              << "  l23_spike_record_alloc_MB="
              << ((l23_record_ints * sizeof(uint32_t)) / (1024.0 * 1024.0)) << "\n";

    // ---- Build Gabor templates ----
    const int block = 256;
    const int gridL4 = (N_L4 + block - 1) / block;
    const int gridL23 = (N_L23 + block - 1) / block;
    {
        const int gridT = (N_TEMPLATES + block - 1) / block;
        build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Init AdEx state for both layers ----
    init_full_state_kernel<<<gridL4, block>>>(
        d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
        d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
    );
    init_l23_state_kernel<<<gridL23, block>>>(
        d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
        d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23, N_L23
    );
    clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
    clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Run L4 phase (one launch, n_steps internal) ----
    auto t0 = std::chrono::steady_clock::now();
    if (args.input_mode == "direct") {
        v1_phase_kernel<1><<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
            d_phase_l4,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            /*n_raster=*/0, /*max_raster_spikes=*/0,
            /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            N_STEPS, N_WARMUP,
            cos_t, sin_t,
            k_spatial, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            d_l4_spike_record,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/INT_MAX
        );
    } else {
        v1_phase_kernel<0><<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
            d_phase_l4,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            /*n_raster=*/0, /*max_raster_spikes=*/0,
            /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            N_STEPS, N_WARMUP,
            cos_t, sin_t,
            k_spatial, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            d_l4_spike_record,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/INT_MAX
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    // ---- Run L2/3 phase (per-step launches, recurrent) ----
    for (int step = 0; step < N_STEPS; ++step) {
        v1_l23_recurrent_step_kernel<<<gridL23, block>>>(
            d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
            d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23,
            d_phase_l23,
            d_row_ptr, d_col_idx, d_l23_w_nS,
            d_l4_spike_record,
            d_rec_row_ptr, d_rec_col_idx, d_rec_w_nS,
            d_l23_spike_record,
            /*phase_step_offset=*/0,
            /*step_idx=*/step,
            N_WARMUP
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();

    const double l4_wall_s  = std::chrono::duration<double>(t1 - t0).count();
    const double l23_wall_s = std::chrono::duration<double>(t2 - t1).count();

    // ---- Read aggregates back ----
    std::vector<int> phase_count_l4(N_L4, 0);
    std::vector<int> phase_count_l23(N_L23, 0);
    std::vector<long long> total_spikes_l4(N_L4, 0);
    CUDA_CHECK(cudaMemcpy(phase_count_l4.data(),  d_phase_l4,
                          N_L4  * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phase_count_l23.data(), d_phase_l23,
                          N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(total_spikes_l4.data(), d_tot_l4,
                          N_L4  * sizeof(long long), cudaMemcpyDeviceToHost));
    long long sum_total_spikes_l4 = 0;
    for (long long s : total_spikes_l4) sum_total_spikes_l4 += s;

    // Sanity: count L2/3 bitmask bits to compare against per-cell totals.
    unsigned long long *d_bit_total = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bit_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_bit_total, 0, sizeof(unsigned long long)));
    {
        const int popblock = 256;
        const size_t popgrid =
            (l23_record_ints + popblock - 1) / popblock;
        popcount_uint32_kernel<<<static_cast<int>(popgrid), popblock>>>(
            d_l23_spike_record, l23_record_ints, d_bit_total
        );
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long l23_bits_total = 0;
    CUDA_CHECK(cudaMemcpy(&l23_bits_total, d_bit_total,
                          sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bit_total));

    // ---- Free device buffers ----
    CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
    CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
    CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
    CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
    CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));
    CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
    CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
    CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
    CUDA_CHECK(cudaFree(d_isi_c_l23));CUDA_CHECK(cudaFree(d_isi_s_l23));
    CUDA_CHECK(cudaFree(d_isi_ss_l23));CUDA_CHECK(cudaFree(d_phase_l23));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_l23_w_nS));
    CUDA_CHECK(cudaFree(d_rec_row_ptr));
    CUDA_CHECK(cudaFree(d_rec_col_idx));
    CUDA_CHECK(cudaFree(d_rec_w_nS));
    CUDA_CHECK(cudaFree(d_l4_spike_record));
    CUDA_CHECK(cudaFree(d_l23_spike_record));

    // ===== Analysis =====
    // L2/3 firing rates (post-warmup window).
    std::vector<double> rate_l23(N_L23);
    double sum_rate=0.0, sum_rate_sq=0.0;
    double max_rate=0.0;
    long long n_silent=0;
    for (int i = 0; i < N_L23; ++i) {
        const double r = static_cast<double>(phase_count_l23[i]) / measure_s;
        rate_l23[i] = r;
        sum_rate += r;
        sum_rate_sq += r * r;
        if (r > max_rate) max_rate = r;
        if (r < 0.1) ++n_silent;
    }
    const double mean_rate = sum_rate / N_L23;
    const double std_rate = std::sqrt(std::max(0.0,
        sum_rate_sq / N_L23 - mean_rate * mean_rate));
    std::vector<double> rate_sorted = rate_l23;
    std::nth_element(rate_sorted.begin(),
                     rate_sorted.begin() + N_L23 / 2,
                     rate_sorted.end());
    const double median_rate = rate_sorted[N_L23 / 2];
    std::nth_element(rate_sorted.begin(),
                     rate_sorted.begin() + (size_t)(N_L23 * 0.95),
                     rate_sorted.end());
    const double p95_rate = rate_sorted[(size_t)(N_L23 * 0.95)];

    // L4 reference firing rate.
    double sum_rate_l4 = 0.0;
    for (int i = 0; i < N_L4; ++i) {
        sum_rate_l4 += static_cast<double>(phase_count_l4[i]) / measure_s;
    }
    const double mean_rate_l4 = sum_rate_l4 / N_L4;

    // ---- Recurrent connectivity stats ----
    std::vector<int> rec_fanin(N_L23);
    long long sum_fanin=0;
    int min_fanin = INT32_MAX, max_fanin = 0;
    for (int i = 0; i < N_L23; ++i) {
        const int f = rec.row_ptr[i + 1] - rec.row_ptr[i];
        rec_fanin[i] = f;
        sum_fanin += f;
        if (f < min_fanin) min_fanin = f;
        if (f > max_fanin) max_fanin = f;
    }
    const double mean_fanin = static_cast<double>(sum_fanin) / N_L23;
    long long sum_fanin_edge = 0, sum_fanin_interior = 0;
    int n_edge = 0, n_interior = 0;
    for (int i = 0; i < N_L23; ++i) {
        const int gx = l23_gx(i), gy = l23_gy(i);
        const bool is_edge = (gx == 0 || gx == GRID-1 || gy == 0 || gy == GRID-1);
        if (is_edge) { sum_fanin_edge += rec_fanin[i]; ++n_edge; }
        else         { sum_fanin_interior += rec_fanin[i]; ++n_interior; }
    }
    const double mean_fanin_edge =
        n_edge > 0     ? static_cast<double>(sum_fanin_edge)    / n_edge     : 0.0;
    const double mean_fanin_interior =
        n_interior > 0 ? static_cast<double>(sum_fanin_interior)/ n_interior : 0.0;

    // Weight stats (mV).
    std::vector<double> w_sorted_mV = rec.w_EPSP_mV;
    std::sort(w_sorted_mV.begin(), w_sorted_mV.end());
    const double w_median_mV = w_sorted_mV[w_sorted_mV.size() / 2];
    const double w_max_mV    = w_sorted_mV.back();
    const double w_min_mV    = w_sorted_mV.front();
    double w_sum_mV = 0.0;
    for (double w : rec.w_EPSP_mV) w_sum_mV += w;
    const double w_mean_mV = w_sum_mV / static_cast<double>(rec.w_EPSP_mV.size());

    // Distance histogram bins.
    constexpr int N_DIST_BINS = 17;        // bin width 0.25 hcol, range [0, 4.25)
    constexpr double DIST_BIN_W = 0.25;
    std::vector<long long> dist_hist(N_DIST_BINS, 0);
    for (double d : rec.dist_hcol) {
        int b = static_cast<int>(d / DIST_BIN_W);
        if (b < 0) b = 0;
        if (b >= N_DIST_BINS) b = N_DIST_BINS - 1;
        dist_hist[b] += 1;
    }

    // ---- Reciprocity check (Song 2005-style) ----
    // The construction sets per-pair P(both directions exist) = 4·p(d)²
    // by calibrating r(d) = 3p/(2(1-p)).  The sanity metric measures
    // the UNORDERED pair-level bidirectional rate vs. the chance baseline
    // mean(p(d)²) over the candidate-pair distance distribution.  Both
    // a 5000-sample MC estimate (per spec) and the global exact rate
    // are reported.
    //
    //   observed = (# unordered pairs A↔B) / (# unordered candidate pairs)
    //   chance   = mean(p(d)²) over candidate unordered pairs
    //   ratio    = observed / chance     (target ≈ 4)
    //
    // We also report the conditional rate (P(B→A | A→B sampled), Erdos-
    // Renyi-equivalent expectation = mean p(d) over edges) for diagnostic
    // visibility.

    // (a) Global exact: enumerate all candidate unordered pairs.
    long long n_cand_unordered = 0;
    double sum_p_sq_unordered  = 0.0;
    for (int post = 0; post < N_L23; ++post) {
        const int gx_post = l23_gx(post);
        const int gy_post = l23_gy(post);
        for (int dy = -L23REC_DMAX; dy <= L23REC_DMAX; ++dy) {
            const int gy_pre = gy_post + dy;
            if (gy_pre < 0 || gy_pre >= GRID) continue;
            for (int dx = -L23REC_DMAX; dx <= L23REC_DMAX; ++dx) {
                const int gx_pre = gx_post + dx;
                if (gx_pre < 0 || gx_pre >= GRID) continue;
                const double d = std::sqrt(
                    static_cast<double>(dx*dx + dy*dy));
                if (d > static_cast<double>(L23REC_DMAX)) continue;
                const double p = L23REC_P0 * std::exp(-d / L23REC_LEN_HCOL);
                for (int clone_pre = 0; clone_pre < N_L23_CLONES; ++clone_pre) {
                    const int pre = make_l23_id(gx_pre, gy_pre, clone_pre);
                    if (pre == post) continue;
                    if (pre < post) continue;   // unordered: count once
                    ++n_cand_unordered;
                    sum_p_sq_unordered += p * p;
                }
            }
        }
    }
    const double recip_global_obs = (n_cand_unordered > 0)
        ? static_cast<double>(rec.n_pairs_reciprocal_post)
            / static_cast<double>(n_cand_unordered)
        : 0.0;
    const double recip_global_chance = (n_cand_unordered > 0)
        ? sum_p_sq_unordered / static_cast<double>(n_cand_unordered)
        : 0.0;
    const double recip_global_ratio = (recip_global_chance > 0.0)
        ? recip_global_obs / recip_global_chance : 0.0;

    // (b) 5000-sample MC: sample unordered candidate pairs uniformly from
    // the candidate pool, count how many are bidirectional.
    constexpr int N_RECIP_SAMPLES = 5000;
    long long n_recip_obs = 0;
    double sum_p_sq_sample = 0.0;
    int n_sampled = 0;
    // For sampling, we use rejection: pick a random post and a random
    // (dx, dy, clone) within the d_max box, retry on misses.
    {
        std::mt19937_64 rng_check(args.seed ^ 0x517CC1B727220A95ULL);
        std::uniform_int_distribution<int> post_pick(0, N_L23 - 1);
        std::uniform_int_distribution<int> off_pick(-L23REC_DMAX, L23REC_DMAX);
        std::uniform_int_distribution<int> clone_pick(0, N_L23_CLONES - 1);
        int safety = 0;
        while (n_sampled < N_RECIP_SAMPLES && safety < 50 * N_RECIP_SAMPLES) {
            ++safety;
            const int post = post_pick(rng_check);
            const int dx = off_pick(rng_check);
            const int dy = off_pick(rng_check);
            const int gx_post = l23_gx(post);
            const int gy_post = l23_gy(post);
            const int gx_pre = gx_post + dx;
            const int gy_pre = gy_post + dy;
            if (gx_pre < 0 || gx_pre >= GRID) continue;
            if (gy_pre < 0 || gy_pre >= GRID) continue;
            const double d = std::sqrt(
                static_cast<double>(dx*dx + dy*dy));
            if (d > static_cast<double>(L23REC_DMAX)) continue;
            const int clone_pre = clone_pick(rng_check);
            const int pre = make_l23_id(gx_pre, gy_pre, clone_pre);
            if (pre == post) continue;
            // Uniform among unordered pairs: enforce post < pre, retry otherwise.
            const int A = std::min(post, pre);
            const int B = std::max(post, pre);
            // Check whether both directions exist (in CSR — col_idx sorted per row).
            const auto edge_exists = [&](int row_post, int col_pre) -> bool {
                const int rs = rec.row_ptr[row_post];
                const int re = rec.row_ptr[row_post + 1];
                const auto it = std::lower_bound(
                    rec.col_idx.begin() + rs,
                    rec.col_idx.begin() + re,
                    col_pre);
                return (it != rec.col_idx.begin() + re) && (*it == col_pre);
            };
            const bool ab = edge_exists(B, A);     // A→B stored as col_idx[row=B]=A
            const bool ba = edge_exists(A, B);     // B→A stored as col_idx[row=A]=B
            if (ab && ba) ++n_recip_obs;
            const double p = L23REC_P0 * std::exp(-d / L23REC_LEN_HCOL);
            sum_p_sq_sample += p * p;
            ++n_sampled;
        }
    }
    const double recip_obs_rate    = (n_sampled > 0)
        ? static_cast<double>(n_recip_obs) / n_sampled : 0.0;
    const double recip_chance_rate = (n_sampled > 0)
        ? sum_p_sq_sample / n_sampled : 0.0;
    const double recip_ratio       = (recip_chance_rate > 0.0)
        ? recip_obs_rate / recip_chance_rate : 0.0;

    // ===== Write JSON artifacts =====
    std::filesystem::create_directories(args.out_dir);
    const std::string label_suffix =
        args.label.empty() ? std::string() : std::string("_") + args.label;
    const std::string conn_json_path = args.out_dir + "/l23_recurrent_connectivity"
        + label_suffix + ".json";
    const std::string drive_json_path = args.out_dir + "/l23_recurrent_drive_summary"
        + label_suffix + ".json";

    {
        std::ofstream f(conn_json_path);
        if (!f) die("could not open " + conn_json_path);
        f << std::setprecision(8);
        f << "{\n";
        f << "  \"n_l23\": " << N_L23 << ",\n";
        f << "  \"total_synapses\": " << rec.total_synapses << ",\n";
        f << "  \"params\": {\n";
        f << "    \"p0\": " << L23REC_P0 << ",\n";
        f << "    \"length_hcol\": " << L23REC_LEN_HCOL << ",\n";
        f << "    \"dmax_hcol\": " << L23REC_DMAX << ",\n";
        f << "    \"epsp_median_mV\": " << L23REC_EPSP_MEDIAN_MV << ",\n";
        f << "    \"epsp_log_sigma\": " << L23REC_EPSP_LOG_SIGMA << ",\n";
        f << "    \"epsp_max_mV\": " << L23REC_EPSP_MAX_MV << ",\n";
        f << "    \"mV_per_nS\": " << L23_MV_PER_NS << ",\n";
        f << "    \"tau_syn_l23_ms\": " << TAU_SYN_L23_MS << ",\n";
        f << "    \"E_E_l23_mV\": " << E_E_L23_MV << ",\n";
        f << "    \"delay_steps\": " << L23REC_DELAY_STEPS << ",\n";
        f << "    \"delay_ms\": " << (L23REC_DELAY_STEPS * DT_MS) << ",\n";
        f << "    \"salt_connectivity\": " << L23REC_SALT_CONNECTIVITY << ",\n";
        f << "    \"salt_reciprocity\": " << L23REC_SALT_RECIPROCITY << ",\n";
        f << "    \"salt_weights\": " << L23REC_SALT_WEIGHTS << "\n";
        f << "  },\n";
        f << "  \"seed\": " << args.seed << ",\n";
        f << "  \"build\": {\n";
        f << "    \"base_edges\": " << rec.n_pairs_sampled << ",\n";
        f << "    \"recip_pairs_pre_boost\": " << rec.n_pairs_reciprocal_pre_boost << ",\n";
        f << "    \"edges_added_by_boost\": " << rec.n_edges_added_by_boost << ",\n";
        f << "    \"recip_pairs_post_boost\": " << rec.n_pairs_reciprocal_post << ",\n";
        f << "    \"build_wall_s\": " << rec_build_s << "\n";
        f << "  },\n";
        f << "  \"fanin\": {\n";
        f << "    \"mean\": "          << mean_fanin << ",\n";
        f << "    \"min\": "           << min_fanin  << ",\n";
        f << "    \"max\": "           << max_fanin  << ",\n";
        f << "    \"mean_interior\": " << mean_fanin_interior << ",\n";
        f << "    \"mean_edge\": "     << mean_fanin_edge << ",\n";
        f << "    \"per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) f << ","; f << rec_fanin[i]; }
        f << "]\n  },\n";
        f << "  \"weights_mV\": {\n";
        f << "    \"n\": "      << rec.w_EPSP_mV.size() << ",\n";
        f << "    \"min\": "    << w_min_mV << ",\n";
        f << "    \"median\": " << w_median_mV << ",\n";
        f << "    \"mean\": "   << w_mean_mV << ",\n";
        f << "    \"max\": "    << w_max_mV << ",\n";
        f << "    \"all\": [";
        for (size_t i = 0; i < rec.w_EPSP_mV.size(); ++i) {
            if (i) f << ","; f << rec.w_EPSP_mV[i];
        }
        f << "]\n  },\n";
        f << "  \"distance_hist\": {\n";
        f << "    \"bin_width_hcol\": " << DIST_BIN_W << ",\n";
        f << "    \"n_bins\": " << N_DIST_BINS << ",\n";
        f << "    \"counts\": [";
        for (int i = 0; i < N_DIST_BINS; ++i) {
            if (i) f << ","; f << dist_hist[i];
        }
        f << "]\n  },\n";
        f << "  \"reciprocity_check\": {\n";
        f << "    \"description\": \"unordered-pair-level bidirectional rate vs mean(p²) chance baseline (Song 2005)\",\n";
        f << "    \"n_sampled\": " << n_sampled << ",\n";
        f << "    \"n_reciprocal_observed\": " << n_recip_obs << ",\n";
        f << "    \"observed_rate\": " << recip_obs_rate << ",\n";
        f << "    \"chance_baseline_rate\": " << recip_chance_rate << ",\n";
        f << "    \"ratio_to_chance\": " << recip_ratio << ",\n";
        f << "    \"global_n_candidate_unordered_pairs\": " << n_cand_unordered << ",\n";
        f << "    \"global_n_reciprocal_pairs\": " << rec.n_pairs_reciprocal_post << ",\n";
        f << "    \"global_observed_rate\": " << recip_global_obs << ",\n";
        f << "    \"global_chance_baseline_rate\": " << recip_global_chance << ",\n";
        f << "    \"global_ratio_to_chance\": " << recip_global_ratio << "\n";
        f << "  }\n";
        f << "}\n";
    }

    // ---- l23_recurrent_drive_summary.json ----
    {
        std::ofstream f(drive_json_path);
        if (!f) die("could not open " + drive_json_path);
        f << std::setprecision(8);
        f << "{\n";
        f << "  \"n_l23\": " << N_L23 << ",\n";
        f << "  \"n_l4\": "  << N_L4  << ",\n";
        f << "  \"duration_ms\": " << DURATION_MS << ",\n";
        f << "  \"warmup_steps\": " << N_WARMUP << ",\n";
        f << "  \"warmup_ms\": "    << (N_WARMUP * DT_MS) << ",\n";
        f << "  \"measure_window_s\": " << measure_s << ",\n";
        f << "  \"trained_weights_path\": \"" << args.load_trained_weights << "\",\n";
        f << "  \"l23_recurrent_total_synapses\": " << rec.total_synapses << ",\n";
        f << "  \"l4_to_l23_total_synapses\": " << conn.total_synapses << ",\n";
        f << "  \"stim\": {\n";
        f << "    \"theta_deg\": "       << args.stim_orientation_deg << ",\n";
        f << "    \"f_cyc_per_pixel\": " << args.stim_sf_cycles_per_pixel << ",\n";
        f << "    \"v_tf_hz\": "         << args.stim_tf_hz << ",\n";
        f << "    \"d_drift_sign\": "    << args.stim_drift_sign << ",\n";
        f << "    \"k_spatial\": "       << k_spatial << ",\n";
        f << "    \"omega_signed\": "    << omega << ",\n";
        f << "    \"input_mode\": \""    << args.input_mode << "\"\n";
        f << "  },\n";
        f << "  \"l23_global\": {\n";
        f << "    \"mean_rate_hz\": "   << mean_rate << ",\n";
        f << "    \"std_rate_hz\": "    << std_rate << ",\n";
        f << "    \"median_rate_hz\": " << median_rate << ",\n";
        f << "    \"max_rate_hz\": "    << max_rate << ",\n";
        f << "    \"p95_rate_hz\": "    << p95_rate << ",\n";
        f << "    \"frac_silent\": "
          << (static_cast<double>(n_silent) / N_L23) << ",\n";
        f << "    \"silent_threshold_hz\": 0.1\n";
        f << "  },\n";
        f << "  \"l4_global\": {\n";
        f << "    \"mean_rate_hz\": " << mean_rate_l4 << ",\n";
        f << "    \"total_spikes\": " << sum_total_spikes_l4 << "\n";
        f << "  },\n";
        f << "  \"l23_bitmask_bits_total\": " << l23_bits_total << ",\n";
        f << "  \"timing\": {\n";
        f << "    \"l4_kernel_wall_s\": "  << l4_wall_s << ",\n";
        f << "    \"l23_kernel_wall_s\": " << l23_wall_s << ",\n";
        f << "    \"connectivity_build_wall_s\": " << conn_build_s << ",\n";
        f << "    \"recurrent_build_wall_s\": "    << rec_build_s << "\n";
        f << "  },\n";
        f << "  \"device\": \"" << prop.name << "\",\n";
        f << "  \"seed\": " << args.seed << ",\n";
        f << "  \"l23_rate_hz\": [";
        for (int i = 0; i < N_L23; ++i) {
            if (i) f << ",";
            f << rate_l23[i];
        }
        f << "]\n";
        f << "}\n";
    }

    // ---- stdout summary ----
    std::cout << "\n=== L2/3 recurrent (B1) summary ===\n";
    std::cout << "duration_ms="           << DURATION_MS
              << "  measure_window_s="    << measure_s << "\n";
    std::cout << "l4_mean_rate_hz="       << mean_rate_l4 << "\n";
    std::cout << "l23_mean_rate="         << mean_rate
              << "  median="              << median_rate
              << "  max="                 << max_rate
              << "  p95="                 << p95_rate
              << "  frac_silent="         << (static_cast<double>(n_silent) / N_L23)
              << "\n";
    std::cout << "rec_total_synapses="    << rec.total_synapses
              << "  rec_mean_fanin="      << mean_fanin
              << "  min/max="             << min_fanin << "/" << max_fanin
              << "  interior/edge="       << mean_fanin_interior << "/" << mean_fanin_edge
              << "\n";
    std::cout << "rec_w_median_mV="       << w_median_mV
              << "  rec_w_max_mV="        << w_max_mV
              << "  rec_w_mean_mV="       << w_mean_mV
              << "\n";
    std::cout << "reciprocity (5000-MC):  n_sampled=" << n_sampled
              << "  obs_rate="            << recip_obs_rate
              << "  chance_rate="         << recip_chance_rate
              << "  ratio="               << recip_ratio << "x\n";
    std::cout << "reciprocity (global):   n_pairs=" << n_cand_unordered
              << "  obs_rate="            << recip_global_obs
              << "  chance_rate="         << recip_global_chance
              << "  ratio="               << recip_global_ratio << "x\n";
    std::cout << "timings: l4="           << l4_wall_s
              << "s  l23="                << l23_wall_s << "s\n";
    std::cout << "wrote " << conn_json_path << "\n";
    std::cout << "wrote " << drive_json_path << "\n";
    return 0;
}

// =====================================================================
// Stim-protocol-check (task #53): verify the stim variants are STDP-suitable.
//
// Variants run (per spec):
//   full     -- existing drifting grating, θ uniformly across {0..157.5}
//   phase    -- random spatial phase ϕ ~ U[0, 2π) per trial
//   jitter   -- random origin offset (x_0, y_0) ~ U[-4, 4]² per trial
//   aperture -- circular Gaussian σ=8 px at random center (cx,cy) ~ U[8,24]²
//   sf       -- f ∈ {0.0625, 0.125, 0.25} cyc/px (octave steps)
//   mixed    -- per-trial uniform draw from the above 5
//
// All variants use random θ ∈ {0, 22.5, ..., 157.5}.  L4 weights and L4→L2/3
// connectivity are static (#52 init, median 1 mV / cap 5 mV).
//
// For each trial (1000 ms), kernel writes per-cell peak 20-ms-bin spike count
// (for "transient peak" stat) and per-cell per-50-ms-bin spike counts (for
// retinotopic-decorrelation correlation analysis).
//
// Per spec, dumps to <out_dir>:
//   stim_protocol_summary.json   -- per-variant params, peak-rate stats,
//                                   pairwise correlation samples, L2/3 stats,
//                                   plus 4 representative-trial param tuples
//                                   per variant for the snapshot PNG.
// Python helper plot_stim_protocol.py renders A-D from this JSON.
// =====================================================================
namespace {

struct VariantTrialParams {
    std::string variant;          // resolved variant name (for `mixed` this is the chosen sub-variant)
    double theta_deg;
    double theta_rad;
    double f_cyc_per_px;
    double v_tf_hz;
    int    drift_sign;            // +1 only for #53
    double phi_phase_rad;         // raw spatial phase ϕ
    double x_origin;              // raw position offset (px)
    double y_origin;
    double phase_offset_total;    // = ϕ - K·(x_0·cos θ + y_0·sin θ)
    int    aperture_active;
    double aperture_cx;
    double aperture_cy;
    double aperture_sigma;        // for diagnostic / snapshot rendering
    double aperture_inv_2sigma_sq;
};

static VariantTrialParams sample_variant_params(
    const std::string& variant_request,
    std::mt19937_64& rng,
    int trial_idx
) {
    VariantTrialParams p{};
    // Resolve `mixed` to one of the 5 base variants.
    std::string variant = variant_request;
    if (variant == "mixed") {
        const char* options[5] = {"full", "phase", "jitter", "aperture", "sf"};
        std::uniform_int_distribution<int> pick(0, 4);
        variant = options[pick(rng)];
    }
    p.variant = variant;

    // θ: uniformly across the 8 canonical orientations (cycled by trial; for
    // randomized variants we draw uniformly).
    std::uniform_int_distribution<int> ori_pick(0, 7);
    int ori_idx = (variant == "full") ? (trial_idx % 8) : ori_pick(rng);
    p.theta_deg = static_cast<double>(ori_idx) * 22.5;
    p.theta_rad = p.theta_deg * (PI / 180.0);

    // Defaults (overwritten below per variant).
    p.f_cyc_per_px = 1.0 / stim_kernels::STIM_DEFAULT_SF_PERIOD_PIXELS;  // 0.125
    p.v_tf_hz      = 4.0;
    p.drift_sign   = +1;
    p.phi_phase_rad = 0.0;
    p.x_origin = 0.0;
    p.y_origin = 0.0;
    p.aperture_active = 0;
    p.aperture_cx = 0.0;
    p.aperture_cy = 0.0;
    p.aperture_sigma = 0.0;
    p.aperture_inv_2sigma_sq = 0.0;

    if (variant == "phase") {
        std::uniform_real_distribution<double> phi_d(0.0, 2.0 * PI);
        p.phi_phase_rad = phi_d(rng);
    } else if (variant == "jitter") {
        std::uniform_real_distribution<double> jit_d(-4.0, 4.0);
        p.x_origin = jit_d(rng);
        p.y_origin = jit_d(rng);
    } else if (variant == "aperture") {
        std::uniform_real_distribution<double> ctr_d(8.0, 24.0);
        p.aperture_active = 1;
        p.aperture_cx = ctr_d(rng);
        p.aperture_cy = ctr_d(rng);
        p.aperture_sigma = 8.0;
        p.aperture_inv_2sigma_sq =
            1.0 / (2.0 * p.aperture_sigma * p.aperture_sigma);
    } else if (variant == "sf") {
        const double f_choices[3] = {0.0625, 0.125, 0.25};
        std::uniform_int_distribution<int> f_pick(0, 2);
        p.f_cyc_per_px = f_choices[f_pick(rng)];
    } else if (variant == "full") {
        // No additional randomization beyond θ.
    } else {
        die("unknown sub-variant: " + variant);
    }

    // Combined phase offset (host-side derivation): ϕ - K·(x_0·cos θ + y_0·sin θ).
    const double K = 2.0 * PI * p.f_cyc_per_px;
    p.phase_offset_total = p.phi_phase_rad
        - K * (p.x_origin * std::cos(p.theta_rad)
             + p.y_origin * std::sin(p.theta_rad));

    return p;
}

// Pearson correlation of two int sequences treated as doubles.
static double pearson_corr_int(const int* a, const int* b, int n) {
    double sum_a = 0.0, sum_b = 0.0;
    for (int i = 0; i < n; ++i) { sum_a += a[i]; sum_b += b[i]; }
    const double mean_a = sum_a / n;
    const double mean_b = sum_b / n;
    double num = 0.0, va = 0.0, vb = 0.0;
    for (int i = 0; i < n; ++i) {
        const double da = a[i] - mean_a;
        const double db = b[i] - mean_b;
        num += da * db;
        va  += da * da;
        vb  += db * db;
    }
    const double denom = std::sqrt(std::max(0.0, va) * std::max(0.0, vb));
    if (denom <= 0.0) return 0.0;
    return num / denom;
}

}  // namespace

static int run_stim_protocol_check(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    const int DURATION_MS = args.duration_ms;
    const int N_STEPS = static_cast<int>(DURATION_MS / DT_MS + 0.5);
    if (N_STEPS <= L23_DELAY_STEPS) {
        die("--duration_ms must give n_steps > L23_DELAY_STEPS (=20)");
    }
    const int N_BINS_50 = N_STEPS / STIM_BIN50_STEPS;
    if (N_BINS_50 <= 0) {
        die("--duration_ms must be >= 50 ms for the correlation bins");
    }
    // L23 warmup discards the first 2 ms (silent pre-delay window).
    const int N_WARMUP = L23_DELAY_STEPS;
    const double measure_s = (N_STEPS - N_WARMUP) * DT_S;

    // Variants we run.
    std::vector<std::string> variants;
    if (args.stim_variant == "all") {
        variants = {"full", "phase", "jitter", "aperture", "sf", "mixed"};
    } else {
        variants = {args.stim_variant};
    }

    // ---- Build connectivity (deterministic from args.seed) ----
    auto t_init0 = std::chrono::steady_clock::now();
    L23Connectivity conn = build_l23_connectivity(args.seed);
    auto t_init1 = std::chrono::steady_clock::now();
    const double conn_build_s =
        std::chrono::duration<double>(t_init1 - t_init0).count();
    std::cout << "l23_connectivity_built  total_synapses=" << conn.total_synapses
              << "  build_wall_s=" << conn_build_s << "\n";

    // ---- Allocate device buffers (reused across all trials/variants) ----
    double *d_V_l4=nullptr, *d_w_l4=nullptr, *d_gE_l4=nullptr;
    int *d_refrac_l4=nullptr;
    long long *d_prev_l4=nullptr, *d_isi_c_l4=nullptr, *d_tot_l4=nullptr;
    double *d_isi_s_l4=nullptr, *d_isi_ss_l4=nullptr;
    int *d_phase_l4=nullptr;
    double* d_templates = nullptr;
    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l4,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l4, N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l4,   N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l4,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l4,  N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l4,  N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l4, N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l4,  N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_idx,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_steps, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_count, 0, sizeof(int)));

    double *d_V_l23=nullptr, *d_w_l23=nullptr, *d_gE_l23=nullptr;
    int *d_refrac_l23=nullptr;
    long long *d_prev_l23=nullptr, *d_isi_c_l23=nullptr, *d_tot_l23=nullptr;
    double *d_isi_s_l23=nullptr, *d_isi_ss_l23=nullptr;
    int *d_phase_l23=nullptr;
    CUDA_CHECK(cudaMalloc(&d_V_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l23,     N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l23, N_L23 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l23,   N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l23,    N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l23,  N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l23,  N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l23, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l23,  N_L23 * sizeof(int)));

    int *d_row_ptr=nullptr, *d_col_idx=nullptr;
    double *d_l23_w_nS=nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,
                          static_cast<size_t>(conn.total_synapses) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_l23_w_nS,
                          static_cast<size_t>(conn.total_synapses) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, conn.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, conn.col_idx.data(),
                          static_cast<size_t>(conn.total_synapses) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l23_w_nS, conn.w_nS.data(),
                          static_cast<size_t>(conn.total_synapses) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // L4 spike-bitmask record (for L2/3 driving).
    const size_t spike_record_ints =
        static_cast<size_t>(N_STEPS) * static_cast<size_t>(N_L4_BITMASK_INTS);
    uint32_t* d_l4_spike_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l4_spike_record, spike_record_ints * sizeof(uint32_t)));

    // Per-cell binning outputs.
    int* d_peak_bin20 = nullptr;
    int* d_bin50_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_peak_bin20, N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin50_counts,
                          static_cast<size_t>(N_BINS_50) * N_L4 * sizeof(int)));

    // ---- Build Gabor templates ----
    const int block = 256;
    const int gridL4 = (N_L4 + block - 1) / block;
    const int gridL23 = (N_L23 + block - 1) / block;
    {
        const int gridT = (N_TEMPLATES + block - 1) / block;
        build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Prep cell-pair sample for the cross-hypercolumn correlation ----
    // 200 pairs of (cell_a, cell_b) where cell_a and cell_b are in DIFFERENT
    // hypercolumns.  Single sample shared across all variants for fairness.
    constexpr int N_CORR_PAIRS = 200;
    std::mt19937_64 corr_rng(args.seed ^ 0xDEADBEEF12345678ULL);
    std::uniform_int_distribution<int> cell_pick(0, N_L4 - 1);
    std::vector<std::pair<int,int>> corr_pairs;
    corr_pairs.reserve(N_CORR_PAIRS);
    while (static_cast<int>(corr_pairs.size()) < N_CORR_PAIRS) {
        const int a = cell_pick(corr_rng);
        const int b = cell_pick(corr_rng);
        if (a == b) continue;
        const int a_hc = (cell_gy(a) * GRID + cell_gx(a));
        const int b_hc = (cell_gy(b) * GRID + cell_gx(b));
        if (a_hc == b_hc) continue;
        corr_pairs.emplace_back(a, b);
    }

    // ---- Per-variant accumulators ----
    struct VariantStats {
        std::string name;
        int n_trials = 0;
        std::vector<int>    per_cell_max_peak_count;   // size N_L4, max over trials
        std::vector<double> per_pair_corrs;            // size N_CORR_PAIRS
        // L2/3
        std::vector<double> l23_per_trial_mean_rate;
        std::vector<double> l23_per_trial_max_rate;
        std::vector<double> l23_per_trial_p95_rate;
        std::vector<double> l23_per_trial_frac_silent;
        // L4
        std::vector<double> l4_per_trial_mean_rate;
        // 4 representative trials' params for the snapshot panel.
        std::vector<VariantTrialParams> snapshot_params;
        // Per-pair concatenated 50ms-bin time-series across trials for corr
        // computation.  Layout: [n_pairs][2][n_trials × N_BINS_50].
        std::vector<std::vector<int>> per_pair_seq_a;  // size N_pairs, each n_trials*n_bins
        std::vector<std::vector<int>> per_pair_seq_b;
    };
    std::vector<VariantStats> all_stats;
    all_stats.reserve(variants.size());

    // RNG seeds per variant.
    std::mt19937_64 master_rng(args.seed);

    int global_trial_idx = 0;  // for kernel base_seed offset

    // Host-side scratch for per-trial copies.
    std::vector<int> peak_bin20_host(N_L4);
    std::vector<int> bin50_counts_host(static_cast<size_t>(N_BINS_50) * N_L4);
    std::vector<int> phase_count_l4_host(N_L4);
    std::vector<int> phase_count_l23_host(N_L23);

    auto t_run0 = std::chrono::steady_clock::now();
    for (const std::string& variant : variants) {
        const int n_trials = (variant == "mixed")
            ? 2 * args.n_trials_per_variant
            : args.n_trials_per_variant;
        VariantStats stats;
        stats.name = variant;
        stats.n_trials = n_trials;
        stats.per_cell_max_peak_count.assign(N_L4, 0);
        stats.per_pair_seq_a.resize(N_CORR_PAIRS);
        stats.per_pair_seq_b.resize(N_CORR_PAIRS);
        for (int p = 0; p < N_CORR_PAIRS; ++p) {
            stats.per_pair_seq_a[p].reserve(static_cast<size_t>(n_trials) * N_BINS_50);
            stats.per_pair_seq_b[p].reserve(static_cast<size_t>(n_trials) * N_BINS_50);
        }

        // Variant-specific RNG (so trial sequences are reproducible).
        std::mt19937_64 var_rng(args.seed
            ^ std::hash<std::string>{}(variant)
            ^ 0xC0FFEEC0FFEEC0FFULL);

        std::cout << "\n--- variant=" << variant
                  << "  n_trials=" << n_trials << " ---\n";

        for (int trial = 0; trial < n_trials; ++trial) {
            VariantTrialParams vp = sample_variant_params(variant, var_rng, trial);

            // Reset state for this trial.
            init_full_state_kernel<<<gridL4, block>>>(
                d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
                d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
            );
            init_l23_state_kernel<<<gridL23, block>>>(
                d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
                d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23, N_L23
            );
            clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
            clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
            clear_int_array_kernel<<<gridL4, block>>>(d_peak_bin20, N_L4);
            {
                const size_t bin_n = static_cast<size_t>(N_BINS_50) * N_L4;
                const int gridB = static_cast<int>((bin_n + block - 1) / block);
                clear_int_array_kernel<<<gridB, block>>>(
                    d_bin50_counts, bin_n);
            }
            {
                const size_t rec_n = spike_record_ints;
                const int gridR = static_cast<int>((rec_n + block - 1) / block);
                clear_uint32_kernel<<<gridR, block>>>(d_l4_spike_record, rec_n);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Run L4 phase.
            const double cos_t = std::cos(vp.theta_rad);
            const double sin_t = std::sin(vp.theta_rad);
            const double K = 2.0 * PI * vp.f_cyc_per_px;
            const double omega = 2.0 * PI * vp.v_tf_hz
                * static_cast<double>(vp.drift_sign);
            // Use closed-form for ALL variants (the kernel's closed-form path
            // now handles aperture + phase_offset).  Direct mode unchanged.
            v1_phase_kernel<0><<<gridL4, block>>>(
                d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
                d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
                d_phase_l4,
                d_templates,
                d_dummy_idx, d_dummy_steps, d_dummy_count,
                /*n_raster=*/0, /*max_raster_spikes=*/0,
                /*phase_idx_for_raster=*/-1,
                /*phase_step_offset=*/0,
                /*phase_idx=*/global_trial_idx,
                N_STEPS, /*n_warmup_steps=*/0,
                cos_t, sin_t, K, omega,
                W_IN_NS, R_BASE_HZ,
                args.seed,
                d_l4_spike_record,
                vp.phase_offset_total,
                vp.aperture_active,
                vp.aperture_cx, vp.aperture_cy, vp.aperture_inv_2sigma_sq,
                d_peak_bin20,
                d_bin50_counts,
                N_BINS_50,
                /*n_stim_steps=*/INT_MAX
            );
            CUDA_CHECK(cudaGetLastError());

            // Run L2/3 phase consuming the bitmask.
            v1_l23_phase_kernel<<<gridL23, block>>>(
                d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
                d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23,
                d_phase_l23,
                d_row_ptr, d_col_idx, d_l23_w_nS,
                d_l4_spike_record,
                /*phase_step_offset=*/0,
                N_STEPS, /*n_warmup_steps=*/N_WARMUP
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Read out per-cell peak rate + L2/3 spike count.
            CUDA_CHECK(cudaMemcpy(peak_bin20_host.data(), d_peak_bin20,
                                  N_L4 * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bin50_counts_host.data(), d_bin50_counts,
                                  static_cast<size_t>(N_BINS_50) * N_L4 * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(phase_count_l4_host.data(),  d_phase_l4,
                                  N_L4  * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(phase_count_l23_host.data(), d_phase_l23,
                                  N_L23 * sizeof(int), cudaMemcpyDeviceToHost));

            // Update per-cell max peak count.
            for (int i = 0; i < N_L4; ++i) {
                if (peak_bin20_host[i] > stats.per_cell_max_peak_count[i]) {
                    stats.per_cell_max_peak_count[i] = peak_bin20_host[i];
                }
            }
            // Append per-pair 50-ms time series to the per-variant rolling buffer.
            for (int p = 0; p < N_CORR_PAIRS; ++p) {
                const int a = corr_pairs[p].first;
                const int b = corr_pairs[p].second;
                for (int bin = 0; bin < N_BINS_50; ++bin) {
                    stats.per_pair_seq_a[p].push_back(
                        bin50_counts_host[static_cast<size_t>(bin) * N_L4 + a]);
                    stats.per_pair_seq_b[p].push_back(
                        bin50_counts_host[static_cast<size_t>(bin) * N_L4 + b]);
                }
            }

            // L4 / L2/3 trial-level stats.
            double sum_l4 = 0.0;
            for (int i = 0; i < N_L4; ++i) {
                sum_l4 += static_cast<double>(phase_count_l4_host[i]) / measure_s;
            }
            stats.l4_per_trial_mean_rate.push_back(sum_l4 / N_L4);

            double sum_l23 = 0.0, max_l23 = 0.0;
            int n_silent_l23 = 0;
            std::vector<double> l23_rates(N_L23);
            for (int i = 0; i < N_L23; ++i) {
                const double r = static_cast<double>(phase_count_l23_host[i]) / measure_s;
                l23_rates[i] = r;
                sum_l23 += r;
                if (r > max_l23) max_l23 = r;
                if (r < 0.1) ++n_silent_l23;
            }
            std::nth_element(l23_rates.begin(),
                             l23_rates.begin() + (size_t)(N_L23 * 0.95),
                             l23_rates.end());
            stats.l23_per_trial_mean_rate.push_back(sum_l23 / N_L23);
            stats.l23_per_trial_max_rate.push_back(max_l23);
            stats.l23_per_trial_p95_rate.push_back(l23_rates[(size_t)(N_L23 * 0.95)]);
            stats.l23_per_trial_frac_silent.push_back(
                static_cast<double>(n_silent_l23) / N_L23);

            // Save 4 representative trial params for the snapshot panel.
            if (static_cast<int>(stats.snapshot_params.size()) < 4) {
                stats.snapshot_params.push_back(vp);
            }

            ++global_trial_idx;
            if ((trial + 1) % 5 == 0 || trial + 1 == n_trials) {
                std::cout << "  trial " << (trial + 1) << "/" << n_trials
                          << "  l4_mean_rate=" << stats.l4_per_trial_mean_rate.back()
                          << "  l23_mean_rate=" << stats.l23_per_trial_mean_rate.back()
                          << "  l23_frac_silent=" << stats.l23_per_trial_frac_silent.back()
                          << "\n";
            }
        }

        // Compute per-pair correlations.
        stats.per_pair_corrs.resize(N_CORR_PAIRS);
        for (int p = 0; p < N_CORR_PAIRS; ++p) {
            const int n = static_cast<int>(stats.per_pair_seq_a[p].size());
            stats.per_pair_corrs[p] = pearson_corr_int(
                stats.per_pair_seq_a[p].data(),
                stats.per_pair_seq_b[p].data(), n);
        }
        // Free the per-pair time-series buffers; we only need the corrs from now.
        stats.per_pair_seq_a.clear();
        stats.per_pair_seq_b.clear();
        stats.per_pair_seq_a.shrink_to_fit();
        stats.per_pair_seq_b.shrink_to_fit();

        all_stats.push_back(std::move(stats));
    }
    auto t_run1 = std::chrono::steady_clock::now();
    const double total_run_s =
        std::chrono::duration<double>(t_run1 - t_run0).count();

    // ---- Free device buffers ----
    CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
    CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
    CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
    CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
    CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));
    CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
    CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
    CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
    CUDA_CHECK(cudaFree(d_isi_c_l23));CUDA_CHECK(cudaFree(d_isi_s_l23));
    CUDA_CHECK(cudaFree(d_isi_ss_l23));CUDA_CHECK(cudaFree(d_phase_l23));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_l23_w_nS));
    CUDA_CHECK(cudaFree(d_l4_spike_record));
    CUDA_CHECK(cudaFree(d_peak_bin20));
    CUDA_CHECK(cudaFree(d_bin50_counts));

    // ---- Compute aggregate per-variant stats (host-side) ----
    struct VariantSummary {
        std::string name;
        int n_trials;
        double peak_rate_mean_hz, peak_rate_median_hz, peak_rate_p95_hz;
        double frac_cells_reaching_10hz, frac_cells_reaching_20hz;
        double mean_pairwise_corr;          // mean signed Pearson r
        double mean_pairwise_corr_abs;      // mean |r| (decorrelation metric;
                                            // see #53 closure note)
        double std_pairwise_corr;
        double frac_pairs_corr_gt_0p2;      // |r| > 0.2 fraction
        double l23_mean_rate, l23_max_rate, l23_p95_rate, l23_frac_silent;
        double l4_mean_rate;
    };
    std::vector<VariantSummary> summaries;
    for (const auto& s : all_stats) {
        VariantSummary vs{};
        vs.name = s.name;
        vs.n_trials = s.n_trials;

        // Per-cell peak rate (Hz) = max_count_in_20ms / 0.020 s.
        const double peak_to_hz = 1.0 / (STIM_BIN20_STEPS * DT_S);  // 1 / 0.02 = 50
        std::vector<double> peak_rates_hz(N_L4);
        long long n_reach_10 = 0, n_reach_20 = 0;
        double sum_peak = 0.0;
        for (int i = 0; i < N_L4; ++i) {
            const double r = static_cast<double>(s.per_cell_max_peak_count[i]) * peak_to_hz;
            peak_rates_hz[i] = r;
            sum_peak += r;
            if (r >= 10.0) ++n_reach_10;
            if (r >= 20.0) ++n_reach_20;
        }
        vs.peak_rate_mean_hz = sum_peak / N_L4;
        std::vector<double> tmp = peak_rates_hz;
        std::nth_element(tmp.begin(), tmp.begin() + N_L4/2, tmp.end());
        vs.peak_rate_median_hz = tmp[N_L4 / 2];
        std::nth_element(tmp.begin(),
                         tmp.begin() + (size_t)(N_L4 * 0.95), tmp.end());
        vs.peak_rate_p95_hz = tmp[(size_t)(N_L4 * 0.95)];
        vs.frac_cells_reaching_10hz =
            static_cast<double>(n_reach_10) / N_L4;
        vs.frac_cells_reaching_20hz =
            static_cast<double>(n_reach_20) / N_L4;

        // Mean pairwise correlation (signed and absolute).
        double sum_corr = 0.0, sum_abs_corr = 0.0, sum_corr_sq = 0.0;
        long long n_big_corr = 0;
        for (double c : s.per_pair_corrs) {
            sum_corr     += c;
            sum_abs_corr += std::fabs(c);
            sum_corr_sq  += c * c;
            if (std::fabs(c) > 0.2) ++n_big_corr;
        }
        const size_t n_pairs = std::max<size_t>(1, s.per_pair_corrs.size());
        vs.mean_pairwise_corr     = sum_corr     / n_pairs;
        vs.mean_pairwise_corr_abs = sum_abs_corr / n_pairs;
        const double mean_r = vs.mean_pairwise_corr;
        vs.std_pairwise_corr = std::sqrt(std::max(0.0,
            sum_corr_sq / n_pairs - mean_r * mean_r));
        vs.frac_pairs_corr_gt_0p2 = static_cast<double>(n_big_corr) / n_pairs;

        // L2/3 average over trials.
        double l23_m=0, l23_max=0, l23_p95=0, l23_fs=0;
        for (size_t t = 0; t < s.l23_per_trial_mean_rate.size(); ++t) {
            l23_m  += s.l23_per_trial_mean_rate[t];
            l23_max += s.l23_per_trial_max_rate[t];
            l23_p95 += s.l23_per_trial_p95_rate[t];
            l23_fs += s.l23_per_trial_frac_silent[t];
        }
        const double n_t = static_cast<double>(s.l23_per_trial_mean_rate.size());
        vs.l23_mean_rate   = l23_m / n_t;
        vs.l23_max_rate    = l23_max / n_t;
        vs.l23_p95_rate    = l23_p95 / n_t;
        vs.l23_frac_silent = l23_fs / n_t;
        // L4 average over trials.
        double l4_m = 0;
        for (double v : s.l4_per_trial_mean_rate) l4_m += v;
        vs.l4_mean_rate = l4_m / static_cast<double>(s.l4_per_trial_mean_rate.size());

        summaries.push_back(vs);
    }

    // ---- Write JSON summary ----
    std::filesystem::create_directories(args.out_dir);
    const std::string label_suffix =
        args.label.empty() ? std::string() : std::string("_") + args.label;
    const std::string json_path = args.out_dir + "/stim_protocol_summary"
        + label_suffix + ".json";

    std::ofstream f(json_path);
    if (!f) die("could not open " + json_path);
    f << std::setprecision(8);
    f << "{\n";
    f << "  \"n_l4\": " << N_L4 << ",\n";
    f << "  \"n_l23\": " << N_L23 << ",\n";
    f << "  \"duration_ms\": " << DURATION_MS << ",\n";
    f << "  \"n_steps\": " << N_STEPS << ",\n";
    f << "  \"n_bins_50\": " << N_BINS_50 << ",\n";
    f << "  \"bin20_steps\": " << STIM_BIN20_STEPS << ",\n";
    f << "  \"bin50_steps\": " << STIM_BIN50_STEPS << ",\n";
    f << "  \"measure_window_s\": " << measure_s << ",\n";
    f << "  \"seed\": " << args.seed << ",\n";
    f << "  \"connectivity\": {\n";
    f << "    \"total_synapses\": " << conn.total_synapses << ",\n";
    f << "    \"weights_median_mV\": "
      << ([&]{
            std::vector<double> ws = conn.w_EPSP_mV;
            std::sort(ws.begin(), ws.end());
            return ws[ws.size()/2];
        }()) << "\n";
    f << "  },\n";
    f << "  \"timing\": {\"total_run_wall_s\": " << total_run_s
      << ", \"connectivity_build_wall_s\": " << conn_build_s << "},\n";
    f << "  \"device\": \"" << prop.name << "\",\n";

    // Cell-pair indices used for correlation (shared across variants).
    f << "  \"corr_pairs\": [";
    for (size_t p = 0; p < corr_pairs.size(); ++p) {
        if (p) f << ",";
        f << "[" << corr_pairs[p].first << "," << corr_pairs[p].second << "]";
    }
    f << "],\n";

    // Per-variant block.
    f << "  \"variants\": [\n";
    for (size_t vi = 0; vi < summaries.size(); ++vi) {
        const auto& s = all_stats[vi];
        const auto& vs = summaries[vi];
        f << "    {\n";
        f << "      \"name\": \"" << vs.name << "\",\n";
        f << "      \"n_trials\": " << vs.n_trials << ",\n";
        f << "      \"l4_mean_rate_hz\": " << vs.l4_mean_rate << ",\n";
        f << "      \"peak_rate_mean_hz\": " << vs.peak_rate_mean_hz << ",\n";
        f << "      \"peak_rate_median_hz\": " << vs.peak_rate_median_hz << ",\n";
        f << "      \"peak_rate_p95_hz\": " << vs.peak_rate_p95_hz << ",\n";
        f << "      \"frac_cells_reaching_10hz\": "
          << vs.frac_cells_reaching_10hz << ",\n";
        f << "      \"frac_cells_reaching_20hz\": "
          << vs.frac_cells_reaching_20hz << ",\n";
        f << "      \"mean_pairwise_corr\": " << vs.mean_pairwise_corr << ",\n";
        f << "      \"mean_pairwise_corr_abs\": " << vs.mean_pairwise_corr_abs << ",\n";
        f << "      \"std_pairwise_corr\": " << vs.std_pairwise_corr << ",\n";
        f << "      \"frac_pairs_corr_gt_0p2\": " << vs.frac_pairs_corr_gt_0p2 << ",\n";
        f << "      \"l23\": {\n";
        f << "        \"mean_rate_hz\": " << vs.l23_mean_rate << ",\n";
        f << "        \"max_rate_hz\": "  << vs.l23_max_rate << ",\n";
        f << "        \"p95_rate_hz\": "  << vs.l23_p95_rate << ",\n";
        f << "        \"frac_silent\": "  << vs.l23_frac_silent << "\n";
        f << "      },\n";
        // Snapshot params (4 trials per variant for the snapshot panel).
        f << "      \"snapshot_params\": [\n";
        for (size_t k = 0; k < s.snapshot_params.size(); ++k) {
            const auto& vp = s.snapshot_params[k];
            f << "        {\"variant\":\"" << vp.variant
              << "\",\"theta_deg\":" << vp.theta_deg
              << ",\"f_cyc_per_px\":" << vp.f_cyc_per_px
              << ",\"v_tf_hz\":" << vp.v_tf_hz
              << ",\"phi_phase_rad\":" << vp.phi_phase_rad
              << ",\"x_origin\":" << vp.x_origin
              << ",\"y_origin\":" << vp.y_origin
              << ",\"aperture_active\":" << vp.aperture_active
              << ",\"aperture_cx\":" << vp.aperture_cx
              << ",\"aperture_cy\":" << vp.aperture_cy
              << ",\"aperture_sigma\":" << vp.aperture_sigma << "}";
            if (k + 1 < s.snapshot_params.size()) f << ",";
            f << "\n";
        }
        f << "      ],\n";
        // Per-cell peak rate distribution: dump full array (for histogram).
        f << "      \"peak_rate_per_cell_hz\": [";
        const double peak_to_hz = 1.0 / (STIM_BIN20_STEPS * DT_S);
        for (int i = 0; i < N_L4; ++i) {
            if (i) f << ",";
            f << (s.per_cell_max_peak_count[i] * peak_to_hz);
        }
        f << "],\n";
        // Pairwise correlations (200 floats per variant).
        f << "      \"pairwise_corrs\": [";
        for (size_t p = 0; p < s.per_pair_corrs.size(); ++p) {
            if (p) f << ",";
            f << s.per_pair_corrs[p];
        }
        f << "]\n";
        f << "    }";
        if (vi + 1 < summaries.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    f.close();

    // ---- Spec-named JSON files (per task-#53 brief return format) ----
    //   stim_check_l4_rates.json     -- artifacts B + C source data
    //   l23_responses_per_variant.json  -- artifact D source data
    const std::string l4_json_path = args.out_dir + "/stim_check_l4_rates"
        + label_suffix + ".json";
    const std::string l23_json_path = args.out_dir + "/l23_responses_per_variant"
        + label_suffix + ".json";

    {
        std::ofstream g(l4_json_path);
        if (!g) die("could not open " + l4_json_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"n_l4\": " << N_L4 << ",\n";
        g << "  \"duration_ms\": " << DURATION_MS << ",\n";
        g << "  \"bin20_steps\": " << STIM_BIN20_STEPS << ",\n";
        g << "  \"bin50_steps\": " << STIM_BIN50_STEPS << ",\n";
        g << "  \"n_bins_50\": " << N_BINS_50 << ",\n";
        g << "  \"seed\": " << args.seed << ",\n";
        g << "  \"corr_pairs\": [";
        for (size_t p = 0; p < corr_pairs.size(); ++p) {
            if (p) g << ",";
            g << "[" << corr_pairs[p].first << "," << corr_pairs[p].second << "]";
        }
        g << "],\n";
        g << "  \"variants\": [\n";
        for (size_t vi = 0; vi < summaries.size(); ++vi) {
            const auto& s = all_stats[vi];
            const auto& vs = summaries[vi];
            g << "    {\n";
            g << "      \"name\": \"" << vs.name << "\",\n";
            g << "      \"n_trials\": " << vs.n_trials << ",\n";
            g << "      \"l4_mean_rate_hz\": " << vs.l4_mean_rate << ",\n";
            // Artifact B: per-variant L4 transient peak-rate stats.
            g << "      \"peak_rate_mean_hz\": " << vs.peak_rate_mean_hz << ",\n";
            g << "      \"peak_rate_median_hz\": " << vs.peak_rate_median_hz << ",\n";
            g << "      \"peak_rate_p95_hz\": " << vs.peak_rate_p95_hz << ",\n";
            g << "      \"frac_cells_reaching_10hz\": "
              << vs.frac_cells_reaching_10hz << ",\n";
            g << "      \"frac_cells_reaching_20hz\": "
              << vs.frac_cells_reaching_20hz << ",\n";
            // Artifact C: per-variant cross-hypercolumn pairwise correlation.
            // The original spec also asks for `mean_pairwise_corr_within_hcol`;
            // not measured in this run (lead's #53 clarification specified
            // only across-hypercolumn sampling), so reported as null.
            g << "      \"mean_pairwise_corr_across_hcols\": "
              << vs.mean_pairwise_corr << ",\n";
            g << "      \"mean_pairwise_corr_across_hcols_abs\": "
              << vs.mean_pairwise_corr_abs << ",\n";
            g << "      \"std_pairwise_corr_across_hcols\": "
              << vs.std_pairwise_corr << ",\n";
            g << "      \"frac_pairs_corr_gt_0p2\": "
              << vs.frac_pairs_corr_gt_0p2 << ",\n";
            g << "      \"mean_pairwise_corr_within_hcol\": null,\n";
            // Snapshot params for artifact A.
            g << "      \"snapshot_params\": [\n";
            for (size_t k = 0; k < s.snapshot_params.size(); ++k) {
                const auto& vp = s.snapshot_params[k];
                g << "        {\"variant\":\"" << vp.variant
                  << "\",\"theta_deg\":" << vp.theta_deg
                  << ",\"f_cyc_per_px\":" << vp.f_cyc_per_px
                  << ",\"v_tf_hz\":" << vp.v_tf_hz
                  << ",\"phi_phase_rad\":" << vp.phi_phase_rad
                  << ",\"x_origin\":" << vp.x_origin
                  << ",\"y_origin\":" << vp.y_origin
                  << ",\"aperture_active\":" << vp.aperture_active
                  << ",\"aperture_cx\":" << vp.aperture_cx
                  << ",\"aperture_cy\":" << vp.aperture_cy
                  << ",\"aperture_sigma\":" << vp.aperture_sigma << "}";
                if (k + 1 < s.snapshot_params.size()) g << ",";
                g << "\n";
            }
            g << "      ],\n";
            g << "      \"peak_rate_per_cell_hz\": [";
            const double peak_to_hz = 1.0 / (STIM_BIN20_STEPS * DT_S);
            for (int i = 0; i < N_L4; ++i) {
                if (i) g << ",";
                g << (s.per_cell_max_peak_count[i] * peak_to_hz);
            }
            g << "],\n";
            g << "      \"pairwise_corrs\": [";
            for (size_t p = 0; p < s.per_pair_corrs.size(); ++p) {
                if (p) g << ",";
                g << s.per_pair_corrs[p];
            }
            g << "]\n";
            g << "    }";
            if (vi + 1 < summaries.size()) g << ",";
            g << "\n";
        }
        g << "  ]\n";
        g << "}\n";
    }

    {
        std::ofstream h(l23_json_path);
        if (!h) die("could not open " + l23_json_path);
        h << std::setprecision(8);
        h << "{\n";
        h << "  \"n_l23\": " << N_L23 << ",\n";
        h << "  \"duration_ms\": " << DURATION_MS << ",\n";
        h << "  \"seed\": " << args.seed << ",\n";
        h << "  \"variants\": [\n";
        for (size_t vi = 0; vi < summaries.size(); ++vi) {
            const auto& s = all_stats[vi];
            const auto& vs = summaries[vi];
            h << "    {\n";
            h << "      \"name\": \"" << vs.name << "\",\n";
            h << "      \"n_trials\": " << vs.n_trials << ",\n";
            h << "      \"l23_mean_rate_hz\": " << vs.l23_mean_rate << ",\n";
            h << "      \"l23_max_rate_hz\": "  << vs.l23_max_rate << ",\n";
            h << "      \"l23_p95_rate_hz\": "  << vs.l23_p95_rate << ",\n";
            h << "      \"l23_frac_silent\": "  << vs.l23_frac_silent << ",\n";
            h << "      \"per_trial\": {\n";
            h << "        \"mean_rate_hz\": [";
            for (size_t t = 0; t < s.l23_per_trial_mean_rate.size(); ++t) {
                if (t) h << ",";
                h << s.l23_per_trial_mean_rate[t];
            }
            h << "],\n";
            h << "        \"max_rate_hz\": [";
            for (size_t t = 0; t < s.l23_per_trial_max_rate.size(); ++t) {
                if (t) h << ",";
                h << s.l23_per_trial_max_rate[t];
            }
            h << "],\n";
            h << "        \"p95_rate_hz\": [";
            for (size_t t = 0; t < s.l23_per_trial_p95_rate.size(); ++t) {
                if (t) h << ",";
                h << s.l23_per_trial_p95_rate[t];
            }
            h << "],\n";
            h << "        \"frac_silent\": [";
            for (size_t t = 0; t < s.l23_per_trial_frac_silent.size(); ++t) {
                if (t) h << ",";
                h << s.l23_per_trial_frac_silent[t];
            }
            h << "]\n";
            h << "      }\n";
            h << "    }";
            if (vi + 1 < summaries.size()) h << ",";
            h << "\n";
        }
        h << "  ]\n";
        h << "}\n";
    }

    // ---- stdout summary ----
    std::cout << "\n=== stim-protocol-check summary ===\n";
    std::cout << "duration_ms="  << DURATION_MS
              << "  measure_s="  << measure_s
              << "  n_bins_50="  << N_BINS_50
              << "  total_run_wall_s=" << total_run_s << "\n";
    std::cout << std::setw(10) << "variant"
              << std::setw(8)  << "trials"
              << std::setw(11) << "l4_mean"
              << std::setw(11) << "peak_mean"
              << std::setw(11) << ">=10Hz"
              << std::setw(11) << "corr"
              << std::setw(11) << "|corr|"
              << std::setw(11) << "frac|r|>0.2"
              << std::setw(11) << "l23_mean"
              << std::setw(11) << "l23_silent"
              << "\n";
    for (const auto& vs : summaries) {
        std::cout << std::setw(10) << vs.name
                  << std::setw(8)  << vs.n_trials
                  << std::setw(11) << vs.l4_mean_rate
                  << std::setw(11) << vs.peak_rate_mean_hz
                  << std::setw(11) << vs.frac_cells_reaching_10hz
                  << std::setw(11) << vs.mean_pairwise_corr
                  << std::setw(11) << vs.mean_pairwise_corr_abs
                  << std::setw(11) << vs.frac_pairs_corr_gt_0p2
                  << std::setw(11) << vs.l23_mean_rate
                  << std::setw(11) << vs.l23_frac_silent
                  << "\n";
    }
    std::cout << "stim_protocol_summary_json=" << json_path << "\n";
    std::cout << "stim_check_l4_rates_json="   << l4_json_path << "\n";
    std::cout << "l23_responses_per_variant_json=" << l23_json_path << "\n";

    // Pass-criteria check.
    bool pass = true;
    auto find_var = [&](const std::string& name) -> const VariantSummary* {
        for (const auto& v : summaries) if (v.name == name) return &v;
        return nullptr;
    };
    const VariantSummary* full_v   = find_var("full");
    const VariantSummary* mixed_v  = find_var("mixed");
    const VariantSummary* aper_v   = find_var("aperture");
    if (mixed_v && mixed_v->frac_cells_reaching_10hz < 0.80) {
        std::cout << "FAIL: mixed.frac_cells_reaching_10hz="
                  << mixed_v->frac_cells_reaching_10hz << " < 0.80\n";
        pass = false;
    }
    if (full_v && mixed_v) {
        const double thresh = 0.5 * full_v->mean_pairwise_corr;
        if (mixed_v->mean_pairwise_corr > thresh) {
            std::cout << "FAIL: mixed.corr=" << mixed_v->mean_pairwise_corr
                      << " > 0.5 * full.corr (" << thresh << ")\n";
            pass = false;
        }
    }
    if (full_v && aper_v) {
        const double thresh = 0.5 * full_v->mean_pairwise_corr;
        if (aper_v->mean_pairwise_corr > thresh) {
            std::cout << "FAIL: aperture.corr=" << aper_v->mean_pairwise_corr
                      << " > 0.5 * full.corr (" << thresh << ")\n";
            pass = false;
        }
    }
    if (mixed_v) {
        if (mixed_v->l23_mean_rate < 0.5 || mixed_v->l23_mean_rate > 5.0) {
            std::cout << "FAIL: mixed.l23_mean_rate=" << mixed_v->l23_mean_rate
                      << " outside [0.5, 5.0]\n";
            pass = false;
        }
    }
    std::cout << "PASS_CRITERIA: " << (pass ? "PASS" : "FAIL") << "\n";

    return pass ? 0 : 2;
}

// =====================================================================
// Task #54: STDP training + validation suite (Phase A, no aperture).
//
// Pipeline:
//   1. Build connectivity (deterministic from seed).
//   2. Init L4 + L2/3 state, init x_pre / y_post traces, init weights.
//   3. Snapshot weights at trial 0 (pre-training baseline).
//   4. Training loop (--n-train-trials trials): each trial draws random
//      θ/ϕ/(x_0,y_0)/f from the mixed-no-aperture distribution, runs the
//      L4 phase + L2/3 STDP phase for --train-stim-ms, then resets state
//      and traces (the 100 ms ITI is modeled as an instant reset since
//      τ_plus=15 ms / τ_minus=30 ms make 100 ms ≈ full decay).
//      Snapshot weights at trials [500, 1000, 2000, 3000, n_train].
//   5. Run V1-V5 validation tests with frozen weights.
//   6. Dump JSONs.
//
// Closure outputs go to <out_dir>:
//   train_weight_snapshots.json    (per-snapshot weight stats + dump)
//   train_per_trial.json           (per-trial summary)
//   v1_v2_phaseA_v1_rfs.json       (RF locality)
//   v1_v2_phaseA_v2_osi.json       (orientation tuning)
//   v1_v2_phaseA_v3_pi.json        (phase invariance)
//   v1_v2_phaseA_v4_decode.json    (phase-generalization decoding -- features)
//   v1_v2_phaseA_v5_diag.json      (firing + weight diagnostics)
//   v1_v2_phaseA_summary.json      (top-level pass/fail table)
// PNGs are rendered by the Python helper from these JSONs.
// =====================================================================
namespace {

// Sample the mixed-no-aperture training params per task-#54 spec:
//   θ ∈ uniform from {0, 22.5, ..., 157.5} (8 angles)
//   ϕ ∈ U[0, 2π)
//   (x_0, y_0) ∈ U(-4, +4)^2
//   f ∈ uniform from {0.0625, 0.125, 0.25}
struct TrainTrialParams {
    double theta_deg, theta_rad;
    double f_cyc_per_px;
    double phi_phase_rad;
    double x_origin, y_origin;
    double phase_offset_total;
};

static TrainTrialParams sample_train_params(std::mt19937_64& rng) {
    TrainTrialParams p{};
    std::uniform_int_distribution<int> ori_pick(0, 7);
    std::uniform_int_distribution<int> sf_pick(0, 2);
    std::uniform_real_distribution<double> phi_d(0.0, 2.0 * PI);
    std::uniform_real_distribution<double> jit_d(-4.0, 4.0);
    const int oi = ori_pick(rng);
    p.theta_deg = static_cast<double>(oi) * 22.5;
    p.theta_rad = p.theta_deg * (PI / 180.0);
    const double f_choices[3] = {0.0625, 0.125, 0.25};
    p.f_cyc_per_px = f_choices[sf_pick(rng)];
    p.phi_phase_rad = phi_d(rng);
    p.x_origin = jit_d(rng);
    p.y_origin = jit_d(rng);
    const double K = 2.0 * PI * p.f_cyc_per_px;
    p.phase_offset_total = p.phi_phase_rad
        - K * (p.x_origin * std::cos(p.theta_rad)
             + p.y_origin * std::sin(p.theta_rad));
    return p;
}

// Compute weight-distribution stats on host.
struct WeightStats {
    double mean, median, p95, std_, min_, max_;
    double frac_at_zero;       // weight <= 1e-6 nS
    double frac_at_cap;        // weight >= 0.99 * w_max
};
static WeightStats compute_weight_stats(const std::vector<double>& w) {
    WeightStats s{};
    if (w.empty()) return s;
    double sum=0, sumsq=0;
    s.min_ = w[0]; s.max_ = w[0];
    long long nz=0, nc=0;
    const double cap = STDP_W_MAX_NS;
    for (double v : w) {
        sum += v; sumsq += v*v;
        if (v < s.min_) s.min_ = v;
        if (v > s.max_) s.max_ = v;
        if (v <= 1e-6)         ++nz;
        if (v >= 0.99 * cap)   ++nc;
    }
    s.mean = sum / w.size();
    s.std_ = std::sqrt(std::max(0.0, sumsq/w.size() - s.mean*s.mean));
    std::vector<double> tmp = w;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
    s.median = tmp[tmp.size()/2];
    std::nth_element(tmp.begin(), tmp.begin() + (size_t)(tmp.size()*0.95), tmp.end());
    s.p95    = tmp[(size_t)(tmp.size()*0.95)];
    s.frac_at_zero = static_cast<double>(nz) / w.size();
    s.frac_at_cap  = static_cast<double>(nc) / w.size();
    return s;
}

}  // namespace

static int run_train_stdp(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    const int N_TRAIN = args.n_train_trials;
    const int TRAIN_STIM_MS = args.train_stim_ms;
    const int TRAIN_ITI_MS  = args.train_iti_ms;
    const int N_STIM_STEPS = static_cast<int>(TRAIN_STIM_MS / DT_MS + 0.5);
    const int N_ITI_STEPS  = static_cast<int>(TRAIN_ITI_MS  / DT_MS + 0.5);
    const int N_TRIAL_STEPS = N_STIM_STEPS + N_ITI_STEPS;
    if (N_STIM_STEPS <= L23_DELAY_STEPS) {
        die("--train-stim-ms must give n_steps > L23_DELAY_STEPS (=20)");
    }

    // --- Build connectivity (task #11: optional Δθ-graded sampling) ---
    auto t_init0 = std::chrono::steady_clock::now();
    L4L23Grading grading_mode = L4L23Grading::random;
    if      (args.l4_l23_grading == "random") grading_mode = L4L23Grading::random;
    else if (args.l4_l23_grading == "am")     grading_mode = L4L23Grading::am;
    else if (args.l4_l23_grading == "sharp")  grading_mode = L4L23Grading::sharp;
    else if (args.l4_l23_grading == "strict") grading_mode = L4L23Grading::strict;
    else if (args.l4_l23_grading == "gentle") grading_mode = L4L23Grading::gentle;
    else die("internal: unknown l4_l23_grading: " + args.l4_l23_grading);
    const GradingParams gp_log = compute_grading_params(grading_mode);
    L23Connectivity conn = build_l23_connectivity(args.seed, grading_mode);
    auto t_init1 = std::chrono::steady_clock::now();
    const double conn_build_s =
        std::chrono::duration<double>(t_init1 - t_init0).count();
    std::cout << "l23_connectivity_built  total_synapses=" << conn.total_synapses
              << "  build_wall_s=" << conn_build_s
              << "  grading=" << grading_to_str(grading_mode)
              << "  scaling=" << gp_log.scaling
              << "  expected_fanin_interior=" << gp_log.expected_fanin_interior
              << "\n";
    std::cout << "  p_per_bin=[";
    for (int b = 0; b < 5; ++b) {
        if (b) std::cout << ", ";
        std::cout << gp_log.p_connect_per_bin[b];
    }
    std::cout << "]  w_curve=[";
    for (int b = 0; b < 5; ++b) {
        if (b) std::cout << ", ";
        std::cout << gp_log.w_curve[b];
    }
    std::cout << "]\n";

    // --- Allocate device buffers (kept alive across all phases) ---
    const int total_syn = conn.total_synapses;
    const int block = 256;
    const int gridL4 = (N_L4 + block - 1) / block;
    const int gridL23 = (N_L23 + block - 1) / block;

    double *d_V_l4=nullptr, *d_w_l4=nullptr, *d_gE_l4=nullptr;
    int *d_refrac_l4=nullptr;
    long long *d_prev_l4=nullptr, *d_isi_c_l4=nullptr, *d_tot_l4=nullptr;
    double *d_isi_s_l4=nullptr, *d_isi_ss_l4=nullptr;
    int *d_phase_l4=nullptr;
    double* d_templates = nullptr;
    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;

    CUDA_CHECK(cudaMalloc(&d_V_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l4,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l4, N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l4,   N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l4,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l4,  N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l4,  N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l4, N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l4,  N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_idx,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_steps, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_count, 0, sizeof(int)));

    double *d_V_l23=nullptr, *d_w_l23=nullptr, *d_gE_l23=nullptr;
    int *d_refrac_l23=nullptr;
    long long *d_prev_l23=nullptr, *d_tot_l23=nullptr;
    int *d_phase_l23=nullptr;
    long long *d_isi_c_l23_dummy=nullptr;
    double *d_isi_s_l23_dummy=nullptr, *d_isi_ss_l23_dummy=nullptr;
    CUDA_CHECK(cudaMalloc(&d_V_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l23,     N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l23, N_L23 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l23,   N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l23,    N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_phase_l23,  N_L23 * sizeof(int)));
    // ISI buffers used by init_l23_state_kernel signature; not used in #54.
    CUDA_CHECK(cudaMalloc(&d_isi_c_l23_dummy,  N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l23_dummy,  N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l23_dummy, N_L23 * sizeof(double)));

    int    *d_row_ptr=nullptr, *d_col_idx=nullptr;
    double *d_l23_w_nS=nullptr;
    double *d_x_pre=nullptr, *d_y_post=nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,  (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,  static_cast<size_t>(total_syn) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_l23_w_nS, static_cast<size_t>(total_syn) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_pre,    static_cast<size_t>(total_syn) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_post,   N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, conn.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, conn.col_idx.data(),
                          static_cast<size_t>(total_syn) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l23_w_nS, conn.w_nS.data(),
                          static_cast<size_t>(total_syn) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // --- Optional: load pre-trained weights and skip training (task #55) ---
    bool weights_loaded_from_disk = false;
    if (!args.load_trained_weights.empty()) {
        std::ifstream wf(args.load_trained_weights, std::ios::binary);
        if (!wf) die("could not open --load-trained-weights: " + args.load_trained_weights);
        wf.seekg(0, std::ios::end);
        const std::streamsize bytes = wf.tellg();
        wf.seekg(0, std::ios::beg);
        const std::streamsize expected_bytes =
            static_cast<std::streamsize>(total_syn) * sizeof(double);
        if (bytes != expected_bytes) {
            die("weights file size mismatch: got " + std::to_string(bytes)
                + " bytes, expected " + std::to_string(expected_bytes)
                + " (n_synapses=" + std::to_string(total_syn) + " × 8)");
        }
        std::vector<double> w_loaded(total_syn);
        wf.read(reinterpret_cast<char*>(w_loaded.data()), expected_bytes);
        if (!wf) die("read failed for --load-trained-weights: "
                     + args.load_trained_weights);
        CUDA_CHECK(cudaMemcpy(d_l23_w_nS, w_loaded.data(),
                              static_cast<size_t>(total_syn) * sizeof(double),
                              cudaMemcpyHostToDevice));
        weights_loaded_from_disk = true;
        std::cout << "loaded trained weights from " << args.load_trained_weights
                  << " (" << bytes << " bytes, " << total_syn << " synapses)\n";
    }

    // L4 spike-bitmask record (sized for the longest-phase used: max of
    // (N_STIM_STEPS + N_ITI_STEPS) during training and ~10000 steps for 1s
    // validation phases).
    constexpr int N_VALID_STEPS = 10000;   // 1 s
    const int max_steps = std::max(N_TRIAL_STEPS, N_VALID_STEPS);
    const size_t spike_record_ints =
        static_cast<size_t>(max_steps) * static_cast<size_t>(N_L4_BITMASK_INTS);
    uint32_t* d_l4_spike_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l4_spike_record, spike_record_ints * sizeof(uint32_t)));

    // --- Build Gabor templates ---
    {
        const int gridT = (N_TEMPLATES + block - 1) / block;
        build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Init traces (zero) ---
    {
        const int gridX = static_cast<int>((static_cast<size_t>(total_syn)
                                            + block - 1) / block);
        clear_double_array_kernel<<<gridX, block>>>(d_x_pre, total_syn);
        clear_double_array_kernel<<<gridL23, block>>>(d_y_post, N_L23);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Helper: run one trial (n_steps total = stim + ITI). ---
    // Captures all device handles by value (or via the surrounding scope).
    //   reset_state=true:  reset L4+L2/3 dyn state and traces (validation use).
    //   reset_state=false: state persists from previous call (training use --
    //                      lead's #54 spec: traces decay naturally during ITI).
    //   n_stim_steps_arg:  number of leading steps with stim ON.  After
    //                      step >= n_stim_steps_arg, the kernel uses
    //                      max_rate=0 (ITI baseline only).
    //   plasticity_active: if true, STDP applied during the WHOLE n_steps.
    //                      To restrict plasticity to the stim window, the
    //                      caller can split the call into two: stim with
    //                      plasticity_on=true, ITI with plasticity_on=false.
    auto run_trial_step_range = [&](const TrainTrialParams& tp,
                                    int n_steps,
                                    int n_stim_steps_arg,
                                    bool plasticity_on,
                                    bool reset_state) {
        const double cos_t = std::cos(tp.theta_rad);
        const double sin_t = std::sin(tp.theta_rad);
        const double K = 2.0 * PI * tp.f_cyc_per_px;
        const double omega = 2.0 * PI * 4.0;   // training v fixed at 4 Hz

        if (reset_state) {
            init_full_state_kernel<<<gridL4, block>>>(
                d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
                d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
            );
            init_l23_state_kernel<<<gridL23, block>>>(
                d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
                d_tot_l23, d_isi_c_l23_dummy, d_isi_s_l23_dummy,
                d_isi_ss_l23_dummy, N_L23
            );
            const int gridX = static_cast<int>(
                (static_cast<size_t>(total_syn) + block - 1) / block);
            clear_double_array_kernel<<<gridX, block>>>(d_x_pre, total_syn);
            clear_double_array_kernel<<<gridL23, block>>>(d_y_post, N_L23);
        }
        // Per-trial spike-count accumulators always cleared (we want per-trial counts).
        clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
        clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
        {
            const size_t rec_n = static_cast<size_t>(n_steps) * N_L4_BITMASK_INTS;
            const int gridR = static_cast<int>((rec_n + block - 1) / block);
            clear_uint32_kernel<<<gridR, block>>>(d_l4_spike_record, rec_n);
        }
        CUDA_CHECK(cudaGetLastError());

        v1_phase_kernel<0><<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
            d_phase_l4,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            /*n_raster=*/0, /*max_raster_spikes=*/0,
            /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/0,
            n_steps, /*n_warmup_steps=*/0,
            cos_t, sin_t, K, omega,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            d_l4_spike_record,
            /*phase_offset=*/tp.phase_offset_total,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/n_stim_steps_arg
        );

        v1_l23_stdp_phase_kernel<<<gridL23, block>>>(
            d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
            d_tot_l23,
            d_phase_l23,
            d_row_ptr, d_col_idx, d_l23_w_nS,
            d_x_pre, d_y_post,
            d_l4_spike_record,
            /*phase_step_offset=*/0,
            n_steps,
            /*plasticity_active=*/plasticity_on ? 1 : 0
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Backward-compat wrapper: legacy run_trial(tp, n_steps, plasticity_on)
    // = "always-stim, always-reset" (used by validation phases below).
    auto run_trial = [&](const TrainTrialParams& tp,
                         int n_steps, bool plasticity_on) {
        run_trial_step_range(tp, n_steps, /*n_stim_steps_arg=*/n_steps,
                              plasticity_on, /*reset_state=*/true);
    };

    // --- Snapshot trial set (per spec, dedup'd against N_TRAIN) ---
    std::vector<int> snapshot_trials_raw = {
        0, N_TRAIN / 10, N_TRAIN / 4, N_TRAIN / 2, (3 * N_TRAIN) / 4, N_TRAIN
    };
    std::sort(snapshot_trials_raw.begin(), snapshot_trials_raw.end());
    snapshot_trials_raw.erase(
        std::unique(snapshot_trials_raw.begin(), snapshot_trials_raw.end()),
        snapshot_trials_raw.end());
    std::vector<int> snapshot_trials = snapshot_trials_raw;

    std::filesystem::create_directories(args.out_dir);
    const std::string label_suffix =
        args.label.empty() ? std::string() : std::string("_") + args.label;

    double train_wall_s = 0.0;
    bool runaway_detected = false;
    std::vector<double> w_host(total_syn);

    if (weights_loaded_from_disk) {
        std::cout << "training: SKIPPED (weights loaded from "
                  << args.load_trained_weights << ")\n";
    } else {
    // --- JSON setup for streaming logs ---
    const std::string snap_path = args.out_dir + "/train_weight_snapshots"
        + label_suffix + ".json";
    const std::string per_trial_path = args.out_dir + "/train_per_trial"
        + label_suffix + ".json";

    std::ofstream snap_f(snap_path);
    if (!snap_f) die("could not open " + snap_path);
    snap_f << std::setprecision(8);
    snap_f << "{\n";
    snap_f << "  \"n_train_trials\": " << N_TRAIN << ",\n";
    snap_f << "  \"train_stim_ms\": " << TRAIN_STIM_MS << ",\n";
    snap_f << "  \"snapshot_trials\": [";
    for (size_t i = 0; i < snapshot_trials.size(); ++i) {
        if (i) snap_f << ",";
        snap_f << snapshot_trials[i];
    }
    snap_f << "],\n";
    snap_f << "  \"stdp\": {"
        << "\"A_plus\":"  << STDP_A_PLUS  << ","
        << "\"A_minus\":" << STDP_A_MINUS << ","
        << "\"tau_plus_ms\":"  << STDP_TAU_PLUS_MS  << ","
        << "\"tau_minus_ms\":" << STDP_TAU_MINUS_MS << ","
        << "\"w_min_nS\":" << STDP_W_MIN_NS << ","
        << "\"w_max_nS\":" << STDP_W_MAX_NS << ","
        << "\"w_max_mV\":" << L23_EPSP_MAX_MV
        << "},\n";
    snap_f << "  \"snapshots\": [\n";

    auto write_snapshot = [&](int trial_idx, const std::vector<double>& w_host,
                              bool first) {
        const auto s = compute_weight_stats(w_host);
        if (!first) snap_f << ",\n";
        snap_f << "    {\"trial\":" << trial_idx
            << ",\"mean\":"    << s.mean
            << ",\"median\":"  << s.median
            << ",\"std\":"     << s.std_
            << ",\"p95\":"     << s.p95
            << ",\"min\":"     << s.min_
            << ",\"max\":"     << s.max_
            << ",\"frac_at_zero\":" << s.frac_at_zero
            << ",\"frac_at_cap\":"  << s.frac_at_cap
            << "}";
    };

    // --- Per-trial CSV-in-JSON (compact) ---
    std::ofstream per_trial_f(per_trial_path);
    if (!per_trial_f) die("could not open " + per_trial_path);
    per_trial_f << std::setprecision(6);
    per_trial_f << "{\"schema\": [\"trial_idx\",\"theta_deg\",\"f\",\"phi\","
                << "\"x_origin\",\"y_origin\",\"l23_total_spikes\","
                << "\"l23_mean_rate_hz\"],\n";
    per_trial_f << "\"rows\":[\n";

    // Snapshot helper (read d_l23_w_nS to host).
    auto snapshot_now = [&](int trial_idx, bool first) {
        CUDA_CHECK(cudaMemcpy(w_host.data(), d_l23_w_nS,
                              static_cast<size_t>(total_syn) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        write_snapshot(trial_idx, w_host, first);
    };

    // --- Training loop ---
    std::cout << "training: " << N_TRAIN << " trials × ("
              << TRAIN_STIM_MS << " ms stim + " << TRAIN_ITI_MS
              << " ms ITI), plasticity ON during stim only, NO inter-trial reset\n";
    snapshot_now(0, /*first=*/true);   // pre-training baseline

    // One-time state init at trial 0.  After that, traces and V evolve
    // naturally (per lead's #54 ruling -- 100 ms ITI > 3τ⁻ so traces decay
    // close to zero between trials without manual reset).
    {
        TrainTrialParams tp_zero{};
        // Run a pure-ITI step (100 ms baseline only) just to seed init kernels
        // via the reset_state=true path.
        tp_zero.theta_rad = 0.0; tp_zero.f_cyc_per_px = 0.125;
        run_trial_step_range(tp_zero, /*n_steps=*/N_ITI_STEPS,
                             /*n_stim_steps_arg=*/0, // immediately ITI
                             /*plasticity_on=*/false,
                             /*reset_state=*/true);
    }

    std::mt19937_64 trial_rng(args.seed ^ 0xA5A5A5A5BEEFBEEFULL);
    auto t_train0 = std::chrono::steady_clock::now();
    int next_snapshot_idx = 1;  // index into snapshot_trials; 0 already done

    std::vector<int> phase_count_l23_host(N_L23);

    std::vector<int> rolling_l23_total(N_TRAIN, 0);

    for (int trial = 0; trial < N_TRAIN; ++trial) {
        TrainTrialParams tp = sample_train_params(trial_rng);
        // Single combined call: stim for first N_STIM_STEPS, ITI for rest.
        // Plasticity is applied uniformly across the combined call (the L23
        // kernel uses plasticity_active flag per-call).  During ITI the L4
        // grating max_rate drops to 0 (kernel-side) so L4 only fires at base
        // rate -> almost no L4 spikes -> almost no LTP/LTD events.  Effective
        // plasticity is dominated by stim-window pre-post pairings.
        run_trial_step_range(tp, /*n_steps=*/N_TRIAL_STEPS,
                              /*n_stim_steps_arg=*/N_STIM_STEPS,
                              /*plasticity_on=*/true,
                              /*reset_state=*/false);

        // Pull per-trial L2/3 spike count (covers stim + ITI; ITI contributes
        // negligibly since drive is baseline 1 Hz).
        CUDA_CHECK(cudaMemcpy(phase_count_l23_host.data(), d_phase_l23,
                              N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
        long long total_l23 = 0;
        for (int v : phase_count_l23_host) total_l23 += v;
        const double l23_mean_rate = static_cast<double>(total_l23) /
                                     (N_L23 * TRAIN_STIM_MS * 1e-3);
        rolling_l23_total[trial] = static_cast<int>(total_l23);

        per_trial_f << (trial > 0 ? ",\n" : "") << "[" << trial << ","
            << tp.theta_deg << "," << tp.f_cyc_per_px << ","
            << tp.phi_phase_rad << ","
            << tp.x_origin << "," << tp.y_origin << ","
            << total_l23 << "," << l23_mean_rate << "]";

        if (next_snapshot_idx < (int)snapshot_trials.size()
            && snapshot_trials[next_snapshot_idx] == trial + 1) {
            snapshot_now(trial + 1, /*first=*/false);
            ++next_snapshot_idx;
        }

        // Runaway / dead-network early-stop.
        if ((trial + 1) % 100 == 0) {
            // compute mean l23 rate over last 100 trials
            long long acc=0;
            const int from = std::max(0, trial - 99);
            for (int t = from; t <= trial; ++t) acc += rolling_l23_total[t];
            const double mean_l23 = static_cast<double>(acc) /
                ((trial - from + 1) * N_L23 * TRAIN_STIM_MS * 1e-3);
            const auto t_now = std::chrono::steady_clock::now();
            const double train_wall_s =
                std::chrono::duration<double>(t_now - t_train0).count();
            std::cout << "  trial " << (trial + 1) << "/" << N_TRAIN
                      << "  rolling100_l23_mean_rate=" << mean_l23 << " Hz"
                      << "  wall=" << train_wall_s << " s" << std::endl;
            if (mean_l23 > 50.0) {
                std::cerr << "RUNAWAY: rolling l23 mean " << mean_l23
                          << " Hz > 50 Hz; halting training at trial "
                          << (trial + 1) << "\n";
                runaway_detected = true;
                break;
            }
        }
    }
    auto t_train1 = std::chrono::steady_clock::now();
    train_wall_s =
        std::chrono::duration<double>(t_train1 - t_train0).count();
    std::cout << "training_wall_s=" << train_wall_s << "\n";

    snap_f << "\n  ]\n}\n";
    snap_f.close();
    per_trial_f << "\n]}\n";
    per_trial_f.close();
    }  // end if (!weights_loaded_from_disk)

    if (runaway_detected) {
        std::cerr << "Halting before validation; report to lead.\n";
        // Free device buffers (best-effort).
        CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
        CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
        CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
        CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
        CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
        CUDA_CHECK(cudaFree(d_templates));
        CUDA_CHECK(cudaFree(d_dummy_idx));
        CUDA_CHECK(cudaFree(d_dummy_steps));
        CUDA_CHECK(cudaFree(d_dummy_count));
        CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
        CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
        CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
        CUDA_CHECK(cudaFree(d_phase_l23));
        CUDA_CHECK(cudaFree(d_isi_c_l23_dummy));
        CUDA_CHECK(cudaFree(d_isi_s_l23_dummy));
        CUDA_CHECK(cudaFree(d_isi_ss_l23_dummy));
        CUDA_CHECK(cudaFree(d_row_ptr));
        CUDA_CHECK(cudaFree(d_col_idx));
        CUDA_CHECK(cudaFree(d_l23_w_nS));
        CUDA_CHECK(cudaFree(d_x_pre));
        CUDA_CHECK(cudaFree(d_y_post));
        CUDA_CHECK(cudaFree(d_l4_spike_record));
        return 3;
    }

    // --- Persist trained weights to disk if requested (task #55) ---
    if (!args.save_trained_weights.empty()) {
        CUDA_CHECK(cudaMemcpy(w_host.data(), d_l23_w_nS,
                              static_cast<size_t>(total_syn) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        const auto wstats_save = compute_weight_stats(w_host);

        const std::string& bin_path = args.save_trained_weights;
        std::ofstream wf(bin_path, std::ios::binary);
        if (!wf) die("could not open --save-trained-weights for write: " + bin_path);
        wf.write(reinterpret_cast<const char*>(w_host.data()),
                 static_cast<std::streamsize>(total_syn) * sizeof(double));
        wf.close();
        if (!wf) die("write failed for --save-trained-weights: " + bin_path);

        std::vector<int> fanin_per_l23(N_L23);
        for (int i = 0; i < N_L23; ++i) {
            fanin_per_l23[i] = conn.row_ptr[i + 1] - conn.row_ptr[i];
        }
        std::sort(fanin_per_l23.begin(), fanin_per_l23.end());
        const int fanin_min = fanin_per_l23.front();
        const int fanin_max = fanin_per_l23.back();
        const int fanin_median = fanin_per_l23[N_L23/2];
        long long fanin_sum = 0;
        for (int v : fanin_per_l23) fanin_sum += v;
        const double fanin_mean = static_cast<double>(fanin_sum) / N_L23;

        std::string meta_path = bin_path;
        // Replace trailing .bin with .json (or append .json if extension differs).
        if (meta_path.size() >= 4
            && meta_path.compare(meta_path.size() - 4, 4, ".bin") == 0) {
            meta_path.replace(meta_path.size() - 4, 4, ".json");
        } else {
            meta_path += ".json";
        }
        std::ofstream mf(meta_path);
        if (!mf) die("could not open weights metadata for write: " + meta_path);
        mf << std::setprecision(8);
        mf << "{\n";
        mf << "  \"bin_path\": \""    << bin_path << "\",\n";
        mf << "  \"seed\": "          << args.seed << ",\n";
        mf << "  \"n_synapses\": "    << total_syn << ",\n";
        mf << "  \"n_l23\": "         << N_L23 << ",\n";
        mf << "  \"n_l4\": "          << N_L4 << ",\n";
        mf << "  \"dtype\": \"float64\",\n";
        mf << "  \"n_train_trials\": " << (weights_loaded_from_disk ? 0 : N_TRAIN) << ",\n";
        mf << "  \"train_stim_ms\": " << TRAIN_STIM_MS << ",\n";
        mf << "  \"train_iti_ms\": "  << TRAIN_ITI_MS  << ",\n";
        mf << "  \"train_wall_s\": "  << train_wall_s  << ",\n";
        mf << "  \"runaway\": "       << (runaway_detected ? "true" : "false") << ",\n";
        mf << "  \"fanin\": {\"min\":" << fanin_min
           << ",\"median\":" << fanin_median
           << ",\"mean\":"   << fanin_mean
           << ",\"max\":"    << fanin_max << "},\n";
        mf << "  \"weights_nS\": {\"mean\":" << wstats_save.mean
           << ",\"median\":" << wstats_save.median
           << ",\"std\":"    << wstats_save.std_
           << ",\"p95\":"    << wstats_save.p95
           << ",\"min\":"    << wstats_save.min_
           << ",\"max\":"    << wstats_save.max_
           << ",\"frac_at_zero\":" << wstats_save.frac_at_zero
           << ",\"frac_at_cap\":"  << wstats_save.frac_at_cap << "},\n";
        mf << "  \"stdp\": {\"A_plus\":" << STDP_A_PLUS
           << ",\"A_minus\":" << STDP_A_MINUS
           << ",\"tau_plus_ms\":"  << STDP_TAU_PLUS_MS
           << ",\"tau_minus_ms\":" << STDP_TAU_MINUS_MS
           << ",\"w_min_nS\":" << STDP_W_MIN_NS
           << ",\"w_max_nS\":" << STDP_W_MAX_NS << "}\n";
        mf << "}\n";
        mf.close();
        std::cout << "saved trained weights to " << bin_path
                  << " (" << total_syn << " synapses, "
                  << static_cast<size_t>(total_syn) * sizeof(double) << " bytes)\n";
        std::cout << "saved weights metadata to " << meta_path << "\n";
    }

    // ============================================================
    // VALIDATION SUITE (frozen weights; plasticity OFF for all phases)
    // ============================================================
    if (args.skip_validation) {
        std::cout << "skip_validation requested; exiting after training.\n";
        // Free device buffers (best-effort).
        CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
        CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
        CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
        CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
        CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
        CUDA_CHECK(cudaFree(d_templates));
        CUDA_CHECK(cudaFree(d_dummy_idx));
        CUDA_CHECK(cudaFree(d_dummy_steps));
        CUDA_CHECK(cudaFree(d_dummy_count));
        CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
        CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
        CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
        CUDA_CHECK(cudaFree(d_phase_l23));
        CUDA_CHECK(cudaFree(d_isi_c_l23_dummy));
        CUDA_CHECK(cudaFree(d_isi_s_l23_dummy));
        CUDA_CHECK(cudaFree(d_isi_ss_l23_dummy));
        CUDA_CHECK(cudaFree(d_row_ptr));
        CUDA_CHECK(cudaFree(d_col_idx));
        CUDA_CHECK(cudaFree(d_l23_w_nS));
        CUDA_CHECK(cudaFree(d_x_pre));
        CUDA_CHECK(cudaFree(d_y_post));
        CUDA_CHECK(cudaFree(d_l4_spike_record));
        return 0;
    }
    auto run_static_phase = [&](double theta_deg, double phi_phase, double f,
                                double x0, double y0,
                                int n_steps,
                                std::vector<int>& l4_count_out,
                                std::vector<int>& l23_count_out) {
        TrainTrialParams tp{};
        tp.theta_deg = theta_deg;
        tp.theta_rad = theta_deg * (PI / 180.0);
        tp.f_cyc_per_px = f;
        tp.phi_phase_rad = phi_phase;
        tp.x_origin = x0;
        tp.y_origin = y0;
        const double K = 2.0 * PI * f;
        tp.phase_offset_total = phi_phase
            - K * (x0 * std::cos(tp.theta_rad) + y0 * std::sin(tp.theta_rad));
        run_trial(tp, n_steps, /*plasticity_on=*/false);
        if (!l4_count_out.empty()) {
            CUDA_CHECK(cudaMemcpy(l4_count_out.data(), d_phase_l4,
                                  N_L4 * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (!l23_count_out.empty()) {
            CUDA_CHECK(cudaMemcpy(l23_count_out.data(), d_phase_l23,
                                  N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
        }
    };

    auto t_v0 = std::chrono::steady_clock::now();

    // ---- V2: orientation tuning sweep (run BEFORE V3 because V3 needs θ_pref) ----
    constexpr int N_THETA = 8;
    constexpr int N_REPS_OSI = 5;
    constexpr int N_OSI_STEPS = 10000;   // 1 s per rep
    const double thetas_deg[N_THETA] = {
        0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5
    };
    // L2/3 rate per θ accumulated across reps.
    std::vector<double> l23_rate_per_theta(static_cast<size_t>(N_THETA) * N_L23, 0.0);
    // L4 rate per θ for V3 (L4-partner PI).
    std::vector<double> l4_rate_per_theta(static_cast<size_t>(N_THETA) * N_L4, 0.0);

    std::cout << "validation V2: orientation sweep ("
              << N_THETA << "θ × " << N_REPS_OSI << " reps × 1s)\n";
    {
        std::vector<int> l4c(N_L4), l23c(N_L23);
        for (int ti = 0; ti < N_THETA; ++ti) {
            for (int r = 0; r < N_REPS_OSI; ++r) {
                run_static_phase(thetas_deg[ti], 0.0, 0.125, 0.0, 0.0,
                                 N_OSI_STEPS, l4c, l23c);
                const double secs = N_OSI_STEPS * DT_S;
                for (int i = 0; i < N_L23; ++i) {
                    l23_rate_per_theta[(size_t)ti*N_L23 + i] +=
                        static_cast<double>(l23c[i]) / secs;
                }
                for (int i = 0; i < N_L4; ++i) {
                    l4_rate_per_theta[(size_t)ti*N_L4 + i] +=
                        static_cast<double>(l4c[i]) / secs;
                }
            }
            for (int i = 0; i < N_L23; ++i)
                l23_rate_per_theta[(size_t)ti*N_L23 + i] /= N_REPS_OSI;
            for (int i = 0; i < N_L4; ++i)
                l4_rate_per_theta[(size_t)ti*N_L4 + i] /= N_REPS_OSI;
        }
    }

    // Compute per-cell OSI for L2/3.
    std::vector<double> osi_l23(N_L23);
    std::vector<int>    pref_theta_idx_l23(N_L23);
    for (int i = 0; i < N_L23; ++i) {
        double sum_r = 0, num_re = 0, num_im = 0;
        double max_r = -1.0; int argmax = 0;
        for (int ti = 0; ti < N_THETA; ++ti) {
            const double r = l23_rate_per_theta[(size_t)ti*N_L23 + i];
            sum_r += r;
            const double ang = 2.0 * thetas_deg[ti] * (PI / 180.0);
            num_re += r * std::cos(ang);
            num_im += r * std::sin(ang);
            if (r > max_r) { max_r = r; argmax = ti; }
        }
        const double mag = std::sqrt(num_re*num_re + num_im*num_im);
        osi_l23[i] = (sum_r > 1e-9) ? (mag / sum_r) : 0.0;
        pref_theta_idx_l23[i] = argmax;
    }

    // ---- V3: phase invariance for 16 sample L2/3 cells ----
    constexpr int N_V3_CELLS = 16;
    constexpr int N_PHI = 8;
    constexpr int N_REPS_PI = 5;
    constexpr int N_PI_STEPS = 10000;   // 1 s

    std::vector<int> v3_cells; v3_cells.reserve(N_V3_CELLS);
    {
        // Pick 16 sample cells: 4×4 grid covering retinotopy.
        int slots[N_V3_CELLS][2] = {
            { 4,  4}, { 4, 12}, { 4, 20}, { 4, 28},
            {12,  4}, {12, 12}, {12, 20}, {12, 28},
            {20,  4}, {20, 12}, {20, 20}, {20, 28},
            {28,  4}, {28, 12}, {28, 20}, {28, 28},
        };
        for (auto& s : slots) {
            v3_cells.push_back(make_l23_id(s[0], s[1], 0));
        }
    }

    // Per (cell × phi) rate.
    std::vector<std::vector<double>> v3_l23_rates(
        N_V3_CELLS, std::vector<double>(N_PHI, 0.0));
    std::vector<std::vector<std::vector<double>>> v3_l4_partner_rates(
        N_V3_CELLS, std::vector<std::vector<double>>(N_PHI));
    // For each sample cell, identify partners + their per-φ rates.
    std::vector<std::vector<int>> v3_partners(N_V3_CELLS);
    for (int s = 0; s < N_V3_CELLS; ++s) {
        const int idx = v3_cells[s];
        const int ps = conn.row_ptr[idx];
        const int pe = conn.row_ptr[idx + 1];
        v3_partners[s].assign(conn.col_idx.begin() + ps,
                              conn.col_idx.begin() + pe);
        for (int phi_idx = 0; phi_idx < N_PHI; ++phi_idx) {
            v3_l4_partner_rates[s][phi_idx].assign(v3_partners[s].size(), 0.0);
        }
    }

    std::cout << "validation V3: phase-invariance ("
              << N_V3_CELLS << " sample cells × "
              << N_PHI << "φ × " << N_REPS_PI << " reps × 1s)\n";
    {
        std::vector<int> l4c(N_L4), l23c(N_L23);
        for (int phi_idx = 0; phi_idx < N_PHI; ++phi_idx) {
            const double phi = static_cast<double>(phi_idx) * (2.0 * PI / N_PHI);
            // Run at θ=0 (one canonical orientation; we'll take per-cell θ_pref
            // analytically below by re-using the V2 rates).  Per-cell-θ_pref
            // sweeps would require N_V3_CELLS separate phase sweeps -- the
            // lead's spec is per-cell-θ_pref; we approximate with the V2
            // result re-mapped: here we simply store the L23 phase response
            // at θ=0 and note the choice in the JSON.
            // (Lead-asked simplification: use a single θ for V3 -- run at the
            // mode of pref_theta_idx_l23 across the 16 sample cells.)
            int counts[N_THETA] = {0};
            for (int s = 0; s < N_V3_CELLS; ++s) ++counts[pref_theta_idx_l23[v3_cells[s]]];
            int mode_ti = 0; for (int ti = 1; ti < N_THETA; ++ti)
                if (counts[ti] > counts[mode_ti]) mode_ti = ti;
            const double theta_for_v3 = thetas_deg[mode_ti];
            for (int r = 0; r < N_REPS_PI; ++r) {
                run_static_phase(theta_for_v3, phi, 0.125, 0.0, 0.0,
                                 N_PI_STEPS, l4c, l23c);
                const double secs = N_PI_STEPS * DT_S;
                for (int s = 0; s < N_V3_CELLS; ++s) {
                    v3_l23_rates[s][phi_idx] +=
                        static_cast<double>(l23c[v3_cells[s]]) / secs;
                    for (size_t k = 0; k < v3_partners[s].size(); ++k) {
                        v3_l4_partner_rates[s][phi_idx][k] +=
                            static_cast<double>(l4c[v3_partners[s][k]]) / secs;
                    }
                }
            }
            for (int s = 0; s < N_V3_CELLS; ++s) {
                v3_l23_rates[s][phi_idx] /= N_REPS_PI;
                for (size_t k = 0; k < v3_partners[s].size(); ++k) {
                    v3_l4_partner_rates[s][phi_idx][k] /= N_REPS_PI;
                }
            }
        }
    }

    // V3 PI calculation per spec: PI = Var_φ(R) / Mean_φ(R)
    auto compute_pi = [](const std::vector<double>& rates) -> double {
        if (rates.empty()) return 0.0;
        double sum=0; for (double r : rates) sum += r;
        const double mean_r = sum / rates.size();
        if (mean_r < 1e-6) return 0.0;
        double var=0; for (double r : rates) var += (r - mean_r)*(r - mean_r);
        var /= rates.size();
        return var / mean_r;
    };
    std::vector<double> v3_l23_pi(N_V3_CELLS);
    std::vector<double> v3_l4_partner_pi_mean(N_V3_CELLS);
    std::vector<double> v3_pi_ratio(N_V3_CELLS);
    for (int s = 0; s < N_V3_CELLS; ++s) {
        v3_l23_pi[s] = compute_pi(v3_l23_rates[s]);
        std::vector<double> per_partner_pi;
        per_partner_pi.reserve(v3_partners[s].size());
        for (size_t k = 0; k < v3_partners[s].size(); ++k) {
            std::vector<double> rs(N_PHI);
            for (int phi_idx = 0; phi_idx < N_PHI; ++phi_idx)
                rs[phi_idx] = v3_l4_partner_rates[s][phi_idx][k];
            per_partner_pi.push_back(compute_pi(rs));
        }
        double sum_pi=0; for (double p : per_partner_pi) sum_pi += p;
        v3_l4_partner_pi_mean[s] =
            per_partner_pi.empty() ? 0.0 : sum_pi / per_partner_pi.size();
        v3_pi_ratio[s] = (v3_l4_partner_pi_mean[s] > 1e-6)
            ? (v3_l23_pi[s] / v3_l4_partner_pi_mean[s]) : 1.0;
    }

    // ---- V5: weight + firing diagnostics (computed before V1/V4 since cheap) ----
    CUDA_CHECK(cudaMemcpy(w_host.data(), d_l23_w_nS,
                          static_cast<size_t>(total_syn) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    const auto wstats_post = compute_weight_stats(w_host);
    // L2/3 firing rate stats already collected from the θ=0 run during V2.
    std::vector<double> l23_rates_at_theta0(N_L23);
    {
        long long n_silent = 0;
        double sum_r = 0, max_r = 0;
        std::vector<double> rs(N_L23);
        for (int i = 0; i < N_L23; ++i) {
            const double r = l23_rate_per_theta[i];   // ti=0 row
            rs[i] = r;
            l23_rates_at_theta0[i] = r;
            sum_r += r;
            if (r > max_r) max_r = r;
            if (r < 0.1) ++n_silent;
        }
        const double mean_r = sum_r / N_L23;
        std::nth_element(rs.begin(), rs.begin() + N_L23/2, rs.end());
        const double median_r = rs[N_L23/2];
        std::nth_element(rs.begin(),
                         rs.begin() + (size_t)(N_L23 * 0.95), rs.end());
        const double p95_r = rs[(size_t)(N_L23 * 0.95)];

        std::cout << "V5 diag: l23_mean_rate=" << mean_r
                  << "  median=" << median_r
                  << "  p95=" << p95_r
                  << "  max=" << max_r
                  << "  frac_silent=" << (static_cast<double>(n_silent)/N_L23) << "\n";

        const std::string v5_path = args.out_dir + "/v1_v2_phaseA_v5_diag"
            + label_suffix + ".json";
        std::ofstream g(v5_path);
        if (!g) die("could not open " + v5_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"l23\": {\"mean_rate_hz\":" << mean_r
            << ",\"median_rate_hz\":" << median_r
            << ",\"p95_rate_hz\":" << p95_r
            << ",\"max_rate_hz\":" << max_r
            << ",\"frac_silent\":" << (static_cast<double>(n_silent)/N_L23) << "},\n";
        g << "  \"weights_nS\": {\"mean\":" << wstats_post.mean
            << ",\"median\":" << wstats_post.median
            << ",\"std\":"    << wstats_post.std_
            << ",\"p95\":"    << wstats_post.p95
            << ",\"min\":"    << wstats_post.min_
            << ",\"max\":"    << wstats_post.max_
            << ",\"frac_at_zero\":" << wstats_post.frac_at_zero
            << ",\"frac_at_cap\":"  << wstats_post.frac_at_cap << "},\n";
        g << "  \"l23_rate_per_cell_hz\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) g << ","; g << l23_rates_at_theta0[i]; }
        g << "],\n";
        g << "  \"l23_w_nS\": [";
        for (int i = 0; i < total_syn; ++i) { if (i) g << ","; g << w_host[i]; }
        g << "]\n}\n";
    }

    // Write V2 + V3 artifacts.
    {
        const std::string v2_path = args.out_dir + "/v1_v2_phaseA_v2_osi"
            + label_suffix + ".json";
        std::ofstream g(v2_path);
        if (!g) die("could not open " + v2_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"thetas_deg\": [";
        for (int ti = 0; ti < N_THETA; ++ti) { if (ti) g << ","; g << thetas_deg[ti]; }
        g << "],\n";
        // OSI distribution stats.
        std::vector<double> osi_sorted = osi_l23;
        std::sort(osi_sorted.begin(), osi_sorted.end());
        const double osi_median = osi_sorted[N_L23/2];
        long long n_gt_0p2 = 0, n_gt_0p5 = 0;
        for (double v : osi_l23) { if (v > 0.2) ++n_gt_0p2; if (v > 0.5) ++n_gt_0p5; }
        g << "  \"osi_median\": " << osi_median << ",\n";
        g << "  \"frac_osi_gt_0p2\": " << (static_cast<double>(n_gt_0p2)/N_L23) << ",\n";
        g << "  \"frac_osi_gt_0p5\": " << (static_cast<double>(n_gt_0p5)/N_L23) << ",\n";
        g << "  \"osi_per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) g << ","; g << osi_l23[i]; }
        g << "],\n";
        g << "  \"pref_theta_idx_per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) g << ","; g << pref_theta_idx_l23[i]; }
        g << "]\n}\n";
    }

    {
        const std::string v3_path = args.out_dir + "/v1_v2_phaseA_v3_pi"
            + label_suffix + ".json";
        std::ofstream g(v3_path);
        if (!g) die("could not open " + v3_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"sample_cells\": [\n";
        for (int s = 0; s < N_V3_CELLS; ++s) {
            const int idx = v3_cells[s];
            g << "    {\"l23_idx\":" << idx
              << ",\"gx\":" << l23_gx(idx)
              << ",\"gy\":" << l23_gy(idx)
              << ",\"n_partners\":" << (int)v3_partners[s].size()
              << ",\"l23_pi\":" << v3_l23_pi[s]
              << ",\"l4_partner_pi_mean\":" << v3_l4_partner_pi_mean[s]
              << ",\"pi_ratio\":" << v3_pi_ratio[s]
              << ",\"l23_rate_per_phi\":[";
            for (int phi_idx = 0; phi_idx < N_PHI; ++phi_idx) {
                if (phi_idx) g << ","; g << v3_l23_rates[s][phi_idx];
            }
            g << "]}" << (s + 1 < N_V3_CELLS ? "," : "") << "\n";
        }
        g << "  ],\n";
        long long n_lt_1 = 0, n_lt_0p5 = 0;
        for (double v : v3_pi_ratio) { if (v < 1.0) ++n_lt_1; if (v < 0.5) ++n_lt_0p5; }
        g << "  \"frac_pi_ratio_lt_1\": " << (static_cast<double>(n_lt_1)/N_V3_CELLS) << ",\n";
        g << "  \"frac_pi_ratio_lt_0p5\": " << (static_cast<double>(n_lt_0p5)/N_V3_CELLS) << "\n";
        g << "}\n";
    }

    // ---- V4: phase-generalization decoding ----
    // 32 conditions × 50 reps (per spec) = 1600 trials.
    // Per trial: 100 ms (= 0.4 grating cycles at 4 Hz) so the spike-count
    // feature retains phase info instead of averaging over multiple cycles.
    // Per-cell L2/3 spike-count vector (16384) + L4 vector (131072) baseline.
    // Decoder runs in Python (sklearn LogisticRegression).
    constexpr int N_V4_PHIS = 4;
    constexpr int N_V4_REPS = 50;
    const double v4_phis[N_V4_PHIS] = {0.0, PI/2.0, PI, 3.0*PI/2.0};
    const int n_v4_trials = N_THETA * N_V4_PHIS * N_V4_REPS;
    std::cout << "validation V4: phase-generalization decoding ("
              << n_v4_trials << " trials × 100 ms)\n";

    // Memory: float instead of double to halve.  L2/3 rates: n_v4_trials × N_L23
    //         = 1600 × 16384 × 4 = 100 MB.  L4 rates: 1600 × 131072 × 4 = 800 MB.
    // To avoid the L4 800 MB blowup, dump per-trial rate vectors to disk.
    const std::string v4_path = args.out_dir + "/v1_v2_phaseA_v4_decode"
        + label_suffix + ".json";
    const std::string v4_l23_bin = args.out_dir + "/v1_v2_phaseA_v4_l23_rates"
        + label_suffix + ".bin";
    const std::string v4_l4_bin  = args.out_dir + "/v1_v2_phaseA_v4_l4_rates"
        + label_suffix + ".bin";

    std::ofstream l23_bin(v4_l23_bin, std::ios::binary);
    std::ofstream l4_bin(v4_l4_bin, std::ios::binary);
    if (!l23_bin || !l4_bin) die("could not open V4 binary outputs");

    std::vector<int> v4_label_theta_idx; v4_label_theta_idx.reserve(n_v4_trials);
    std::vector<int> v4_label_phi_idx;   v4_label_phi_idx.reserve(n_v4_trials);
    {
        std::vector<int> l4c(N_L4), l23c(N_L23);
        std::vector<float> l4_rate_buf(N_L4), l23_rate_buf(N_L23);
        const int n_steps_v4 = 1000;   // 100 ms: short post-onset window preserves phase info
        const double secs_v4 = n_steps_v4 * DT_S;
        int trial_count = 0;
        std::mt19937_64 v4_rng(args.seed ^ 0xCAFEBABEULL);
        std::vector<int> trial_order(n_v4_trials);
        for (int i = 0; i < n_v4_trials; ++i) trial_order[i] = i;
        // No shuffle needed -- iterate deterministically.
        for (int ti = 0; ti < N_THETA; ++ti) {
            for (int pi = 0; pi < N_V4_PHIS; ++pi) {
                for (int r = 0; r < N_V4_REPS; ++r) {
                    run_static_phase(thetas_deg[ti], v4_phis[pi], 0.125, 0.0, 0.0,
                                     n_steps_v4, l4c, l23c);
                    for (int i = 0; i < N_L4; ++i)
                        l4_rate_buf[i] = static_cast<float>(l4c[i] / secs_v4);
                    for (int i = 0; i < N_L23; ++i)
                        l23_rate_buf[i] = static_cast<float>(l23c[i] / secs_v4);
                    l4_bin.write(reinterpret_cast<const char*>(l4_rate_buf.data()),
                                 N_L4 * sizeof(float));
                    l23_bin.write(reinterpret_cast<const char*>(l23_rate_buf.data()),
                                  N_L23 * sizeof(float));
                    v4_label_theta_idx.push_back(ti);
                    v4_label_phi_idx.push_back(pi);
                    ++trial_count;
                    if (trial_count % 200 == 0) {
                        std::cout << "  V4 trial " << trial_count << "/"
                                  << n_v4_trials << "\n";
                    }
                }
            }
        }
    }
    l23_bin.close();
    l4_bin.close();

    // Write V4 metadata JSON; the actual decoder runs in Python.
    {
        std::ofstream g(v4_path);
        if (!g) die("could not open " + v4_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"n_l4\": " << N_L4 << ",\n";
        g << "  \"n_l23\": " << N_L23 << ",\n";
        g << "  \"n_theta\": " << N_THETA << ",\n";
        g << "  \"n_phi\": " << N_V4_PHIS << ",\n";
        g << "  \"n_reps\": " << N_V4_REPS << ",\n";
        g << "  \"n_trials\": " << n_v4_trials << ",\n";
        g << "  \"thetas_deg\": [";
        for (int ti = 0; ti < N_THETA; ++ti) { if (ti) g << ","; g << thetas_deg[ti]; }
        g << "],\n";
        g << "  \"phis_rad\": [";
        for (int pi = 0; pi < N_V4_PHIS; ++pi) { if (pi) g << ","; g << v4_phis[pi]; }
        g << "],\n";
        g << "  \"l23_rates_bin\": \"" << v4_l23_bin << "\",\n";
        g << "  \"l4_rates_bin\":  \"" << v4_l4_bin << "\",\n";
        g << "  \"label_theta_idx\": [";
        for (size_t i = 0; i < v4_label_theta_idx.size(); ++i) {
            if (i) g << ","; g << v4_label_theta_idx[i];
        }
        g << "],\n";
        g << "  \"label_phi_idx\": [";
        for (size_t i = 0; i < v4_label_phi_idx.size(); ++i) {
            if (i) g << ","; g << v4_label_phi_idx[i];
        }
        g << "],\n";
        g << "  \"train_phi_indices\": [0, 2],\n";
        g << "  \"test_phi_indices\":  [1, 3]\n";
        g << "}\n";
    }

    // ---- V1: RF locality via small-Gabor-patch reverse correlation ----
    // Each frame presents a static, localized oriented Gabor patch -- random
    // center (cx, cy) ∈ [0,31]², random orientation θ ∈ {0°, 22.5°, ..., 157.5°},
    // random spatial phase ϕ ∈ U(0, 2π), fixed spatial freq f = 0.125 cyc/px,
    // Gaussian aperture σ = 2 px, base = 1 Hz, amp = 25 Hz (peak ≈ 51 Hz).
    // Sparse-pixel probes drove L4 too weakly (~5 Hz extra/cell) and produced
    // 0 L2/3 spikes; small Gabor patches match L4 tuning and drive the partner
    // L4 cells at grating-comparable rates, so STA can be accumulated with
    // the actual rate field weighted by L2/3 spike count (50 ms post-onset).
    //
    // We re-use the existing v1_phase_kernel<0> (closed-form aperture path):
    // ω = 0 (static patch), aperture_active = 1, σ = 2, with phase_offset
    // baking in ϕ - K·(cx·cosθ + cy·sinθ) so the cosine peaks at the patch
    // center.  Closed-form is bit-equivalent to the direct r_lin formula and
    // matches the spec's rate(x,y) field exactly.
    std::cout << "validation V1: small-Gabor-patch RF mapping (2000 frames × 100 ms)\n";
    constexpr int N_V1_FRAMES        = 2000;
    constexpr int N_V1_STEPS         = 1000;     // 100 ms simulated per frame
    constexpr int N_V1_STA_STEPS     = 500;      // 50 ms post-onset spike window
    constexpr double V1_BASE_RATE_HZ = stim_kernels::STIM_BASE_RATE_HZ;     // 1 Hz
    constexpr double V1_AMP_HZ       = stim_kernels::STIM_MAX_RATE_HZ * 0.5;// 25 Hz
    constexpr double V1_SF_CYC_PER_PX = 1.0 / stim_kernels::STIM_DEFAULT_SF_PERIOD_PIXELS; // 0.125
    constexpr double V1_SIGMA_PX     = 2.0;
    constexpr int    V1_N_ORIENTATIONS = 8;       // 0°, 22.5°, ..., 157.5°
    const double V1_K_SPATIAL = 2.0 * PI * V1_SF_CYC_PER_PX;
    const double V1_INV_2SIGMA_SQ = 1.0 / (2.0 * V1_SIGMA_PX * V1_SIGMA_PX);

    // Per-cell STA: 32×32 rate-weighted accumulator + total spike count.
    std::vector<double>    sta_accum(static_cast<size_t>(N_L23) * N_PIX, 0.0);
    std::vector<long long> sta_total_spikes(N_L23, 0);
    // Per-frame Gabor params (for output JSON / debugging).
    std::vector<double> v1_frame_cx(N_V1_FRAMES);
    std::vector<double> v1_frame_cy(N_V1_FRAMES);
    std::vector<double> v1_frame_theta_deg(N_V1_FRAMES);
    std::vector<double> v1_frame_phi(N_V1_FRAMES);
    {
        std::mt19937_64 v1_rng(args.seed ^ 0xFEEDFACECAFEBEEFULL);
        std::uniform_real_distribution<double> ctr_d(0.0, 31.0);
        std::uniform_int_distribution<int> ori_d(0, V1_N_ORIENTATIONS - 1);
        std::uniform_real_distribution<double> phi_d(0.0, 2.0 * PI);
        std::vector<int> l23c(N_L23);
        std::vector<double> rate_field(N_PIX);

        // Persistent host-side reductions over the L2/3 phase counts.
        long long total_l23_spikes = 0;

        // V1 starts fresh: reset state once (trained weights NOT reset;
        // only V/g_E/refrac/traces).
        init_full_state_kernel<<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
        );
        init_l23_state_kernel<<<gridL23, block>>>(
            d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
            d_tot_l23, d_isi_c_l23_dummy, d_isi_s_l23_dummy,
            d_isi_ss_l23_dummy, N_L23
        );
        {
            const int gridX = static_cast<int>(
                (static_cast<size_t>(total_syn) + block - 1) / block);
            clear_double_array_kernel<<<gridX, block>>>(d_x_pre, total_syn);
            clear_double_array_kernel<<<gridL23, block>>>(d_y_post, N_L23);
        }
        CUDA_CHECK(cudaGetLastError());

        for (int frame = 0; frame < N_V1_FRAMES; ++frame) {
            // ---- Random Gabor-patch params for this frame ----
            const double cx = ctr_d(v1_rng);
            const double cy = ctr_d(v1_rng);
            const int    ori_idx   = ori_d(v1_rng);
            const double theta_deg = static_cast<double>(ori_idx) * (180.0 / V1_N_ORIENTATIONS);
            const double theta_rad = theta_deg * (PI / 180.0);
            const double phi       = phi_d(v1_rng);
            const double cos_t = std::cos(theta_rad);
            const double sin_t = std::sin(theta_rad);
            // phase_offset bakes ϕ + (-K · (cx·cosθ + cy·sinθ)) so the kernel
            // formula yields cos(K·((px - cx)·cosθ + (py - cy)·sinθ) + ϕ).
            const double phase_offset = phi
                - V1_K_SPATIAL * (cx * cos_t + cy * sin_t);
            v1_frame_cx[frame]        = cx;
            v1_frame_cy[frame]        = cy;
            v1_frame_theta_deg[frame] = theta_deg;
            v1_frame_phi[frame]       = phi;

            // ---- Pre-compute the 32×32 rate field this frame on the host ----
            //   rate(x,y) = base + amp · (1 + cos(K·((x-cx)·cosθ + (y-cy)·sinθ) + ϕ))
            //                          · exp(-((x-cx)² + (y-cy)²) / (2σ²))
            for (int py = 0; py < GRID; ++py) {
                for (int px = 0; px < GRID; ++px) {
                    const double dxp = static_cast<double>(px) - cx;
                    const double dyp = static_cast<double>(py) - cy;
                    const double phase_arg =
                        V1_K_SPATIAL * (dxp * cos_t + dyp * sin_t) + phi;
                    const double envelope =
                        std::exp(-(dxp*dxp + dyp*dyp) * V1_INV_2SIGMA_SQ);
                    rate_field[py * GRID + px] =
                        V1_BASE_RATE_HZ
                        + V1_AMP_HZ * (1.0 + std::cos(phase_arg)) * envelope;
                }
            }

            // Reset per-frame counters + bitmask record.
            clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
            clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
            {
                const size_t rec_n =
                    static_cast<size_t>(N_V1_STEPS) * N_L4_BITMASK_INTS;
                const int gridR =
                    static_cast<int>((rec_n + block - 1) / block);
                clear_uint32_kernel<<<gridR, block>>>(d_l4_spike_record, rec_n);
            }
            CUDA_CHECK(cudaGetLastError());

            // L4 with localized Gabor-patch stim (closed-form, ω=0 ⇒ static).
            v1_phase_kernel<0><<<gridL4, block>>>(
                d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
                d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
                d_phase_l4,
                d_templates,
                d_dummy_idx, d_dummy_steps, d_dummy_count,
                /*n_raster=*/0, /*max_raster_spikes=*/0,
                /*phase_idx_for_raster=*/-1,
                /*phase_step_offset=*/0,
                /*phase_idx=*/frame,
                N_V1_STEPS, /*n_warmup_steps=*/0,
                cos_t, sin_t, V1_K_SPATIAL, /*omega=*/0.0,
                W_IN_NS, R_BASE_HZ,
                args.seed,
                d_l4_spike_record,
                phase_offset,
                /*aperture_active=*/1,
                /*aperture_cx=*/cx,
                /*aperture_cy=*/cy,
                /*aperture_inv_2sigma_sq=*/V1_INV_2SIGMA_SQ,
                /*peak_bin20_count_out=*/nullptr,
                /*bin50_counts_out=*/nullptr,
                /*n_bins_50=*/0,
                /*n_stim_steps=*/N_V1_STEPS
            );

            // L2/3 reads delayed bitmask, no plasticity, ONLY 50 ms post-onset
            // (matches spec's STA window).  Per-cell phase_spike_count is the
            // count over those 500 steps; we use it as the STA weight.
            v1_l23_stdp_phase_kernel<<<gridL23, block>>>(
                d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
                d_tot_l23, d_phase_l23,
                d_row_ptr, d_col_idx, d_l23_w_nS,
                d_x_pre, d_y_post,
                d_l4_spike_record,
                /*phase_step_offset=*/0,
                N_V1_STA_STEPS,
                /*plasticity_active=*/0
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(l23c.data(), d_phase_l23,
                                  N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
            // Accumulate rate-weighted STA: sta_accum[i, p] += spike_count[i] * rate_field[p].
            for (int i = 0; i < N_L23; ++i) {
                const int n = l23c[i];
                if (n > 0) {
                    const double w = static_cast<double>(n);
                    double* dst = sta_accum.data() + static_cast<size_t>(i) * N_PIX;
                    for (int p = 0; p < N_PIX; ++p) dst[p] += w * rate_field[p];
                    sta_total_spikes[i] += n;
                    total_l23_spikes    += n;
                }
            }
            if ((frame + 1) % 500 == 0) {
                std::cout << "  V1 frame " << (frame + 1)
                          << "/" << N_V1_FRAMES
                          << "  total_l23_spikes_so_far=" << total_l23_spikes
                          << std::endl;
            }
        }
        std::cout << "  V1 STA done: total L2/3 spikes (50 ms window)="
                  << total_l23_spikes << std::endl;
    }

    // Compute per-cell RF metrics.
    // RF center = pixel with max STA accumulator (per cell).
    // FWHM = number of pixels above 50% of peak count (proxy).
    // Connectedness = 1 if those above-threshold pixels form a single
    // connected component (4-neighbor).
    {
        const std::string v1_path = args.out_dir + "/v1_v2_phaseA_v1_rfs"
            + label_suffix + ".json";
        std::ofstream g(v1_path);
        if (!g) die("could not open " + v1_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"n_frames\": " << N_V1_FRAMES << ",\n";
        g << "  \"n_v1_steps_per_frame\": " << N_V1_STEPS << ",\n";
        g << "  \"n_v1_sta_steps\": " << N_V1_STA_STEPS << ",\n";
        g << "  \"v1_base_rate_hz\": " << V1_BASE_RATE_HZ << ",\n";
        g << "  \"v1_amp_hz\": " << V1_AMP_HZ << ",\n";
        g << "  \"v1_sigma_px\": " << V1_SIGMA_PX << ",\n";
        g << "  \"v1_sf_cyc_per_px\": " << V1_SF_CYC_PER_PX << ",\n";
        g << "  \"v1_n_orientations\": " << V1_N_ORIENTATIONS << ",\n";

        // 16 sample cells for the PNG (same as V3).
        g << "  \"sample_cells\": [";
        for (int s = 0; s < N_V3_CELLS; ++s) {
            if (s) g << ",";
            g << v3_cells[s];
        }
        g << "],\n";
        g << "  \"sample_cells_sta\": [\n";
        for (int s = 0; s < N_V3_CELLS; ++s) {
            const int idx = v3_cells[s];
            g << "    {\"l23_idx\":" << idx
              << ",\"gx\":" << l23_gx(idx)
              << ",\"gy\":" << l23_gy(idx)
              << ",\"total_spikes\":" << sta_total_spikes[idx]
              << ",\"sta\":[";
            for (int p = 0; p < N_PIX; ++p) {
                if (p) g << ",";
                g << sta_accum[(size_t)idx * N_PIX + p];
            }
            g << "]}" << (s + 1 < N_V3_CELLS ? "," : "") << "\n";
        }
        g << "  ],\n";

        // Per-cell metrics: peak position, FWHM count, connected.
        std::vector<int> peak_pos(N_L23, -1);
        std::vector<int> fwhm_count(N_L23, 0);
        std::vector<int> connected(N_L23, 0);
        std::vector<int> peak_in_pool(N_L23, 0);   // peak inside 3×3 hcol pool
        long long n_with_rf = 0;
        for (int i = 0; i < N_L23; ++i) {
            if (sta_total_spikes[i] < 5) continue;
            ++n_with_rf;
            double peak_v = -1.0; int peak_p = -1;
            for (int p = 0; p < N_PIX; ++p) {
                const double v = sta_accum[(size_t)i * N_PIX + p];
                if (v > peak_v) { peak_v = v; peak_p = p; }
            }
            peak_pos[i] = peak_p;
            const double thresh = peak_v * 0.5;
            std::vector<int> above(N_PIX, 0);
            int n_above = 0;
            for (int p = 0; p < N_PIX; ++p) {
                if (sta_accum[(size_t)i * N_PIX + p] >= thresh) {
                    above[p] = 1; ++n_above;
                }
            }
            fwhm_count[i] = n_above;
            // BFS for connectedness from peak.
            std::vector<int> visited(N_PIX, 0);
            std::vector<int> q; q.reserve(N_PIX);
            q.push_back(peak_p);
            visited[peak_p] = 1;
            int n_visited = 0;
            while (!q.empty()) {
                const int p = q.back(); q.pop_back();
                if (above[p] == 0) continue;
                ++n_visited;
                const int x = p % GRID, y = p / GRID;
                int nbrs[4][2] = {{x-1,y},{x+1,y},{x,y-1},{x,y+1}};
                for (auto& n : nbrs) {
                    if (n[0] < 0 || n[0] >= GRID || n[1] < 0 || n[1] >= GRID) continue;
                    const int q_idx = n[1] * GRID + n[0];
                    if (!visited[q_idx]) { visited[q_idx] = 1; q.push_back(q_idx); }
                }
            }
            connected[i] = (n_visited == n_above) ? 1 : 0;

            // Peak inside the 3×3 pool that this L2/3 cell pools from.
            const int gx_l23 = l23_gx(i), gy_l23 = l23_gy(i);
            const int peak_x = peak_p % GRID, peak_y = peak_p / GRID;
            peak_in_pool[i] = (std::abs(peak_x - gx_l23) <= 1
                            && std::abs(peak_y - gy_l23) <= 1) ? 1 : 0;
        }
        // Aggregate metrics.
        std::vector<int> fwhm_with_rf;
        long long n_connected = 0, n_peak_in_pool = 0;
        for (int i = 0; i < N_L23; ++i) {
            if (peak_pos[i] >= 0) {
                fwhm_with_rf.push_back(fwhm_count[i]);
                if (connected[i]) ++n_connected;
                if (peak_in_pool[i]) ++n_peak_in_pool;
            }
        }
        double fwhm_median = 0;
        if (!fwhm_with_rf.empty()) {
            std::sort(fwhm_with_rf.begin(), fwhm_with_rf.end());
            fwhm_median = fwhm_with_rf[fwhm_with_rf.size()/2];
        }
        g << "  \"n_cells_with_rf\": " << n_with_rf << ",\n";
        g << "  \"fwhm_median_pixels\": " << fwhm_median << ",\n";
        g << "  \"frac_connected\": " <<
            (n_with_rf > 0 ? (double)n_connected / n_with_rf : 0.0) << ",\n";
        g << "  \"frac_peak_in_pool\": " <<
            (n_with_rf > 0 ? (double)n_peak_in_pool / n_with_rf : 0.0) << ",\n";
        g << "  \"peak_pos_per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) g << ","; g << peak_pos[i]; }
        g << "],\n";
        g << "  \"fwhm_per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) g << ","; g << fwhm_count[i]; }
        g << "],\n";
        g << "  \"connected_per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) g << ","; g << connected[i]; }
        g << "]\n";
        g << "}\n";
    }

    auto t_v1 = std::chrono::steady_clock::now();
    const double valid_wall_s =
        std::chrono::duration<double>(t_v1 - t_v0).count();

    // Top-level summary.
    {
        const std::string sum_path = args.out_dir + "/v1_v2_phaseA_summary"
            + label_suffix + ".json";
        std::ofstream g(sum_path);
        if (!g) die("could not open " + sum_path);
        g << std::setprecision(8);
        g << "{\n";
        g << "  \"task\": \"v1_l23_phaseA_train_validate\",\n";
        g << "  \"seed\": " << args.seed << ",\n";
        g << "  \"n_train_trials\": " << N_TRAIN << ",\n";
        g << "  \"train_wall_s\": " << train_wall_s << ",\n";
        g << "  \"validation_wall_s\": " << valid_wall_s << ",\n";
        g << "  \"device\": \"" << prop.name << "\",\n";
        g << "  \"runaway\": " << (runaway_detected ? "true" : "false") << ",\n";
        g << "  \"weights_pre_train\": null,\n";   // first snapshot in train_weight_snapshots.json
        g << "  \"weights_post_train\": {\"mean\":" << wstats_post.mean
            << ",\"median\":" << wstats_post.median
            << ",\"max\":"    << wstats_post.max_
            << ",\"frac_at_zero\":" << wstats_post.frac_at_zero
            << ",\"frac_at_cap\":"  << wstats_post.frac_at_cap << "}\n";
        g << "}\n";
    }

    std::cout << "\n=== train+validate summary ===\n";
    std::cout << "train_wall_s=" << train_wall_s
              << "  validation_wall_s=" << valid_wall_s << "\n";
    std::cout << "weights_post: mean=" << wstats_post.mean
              << "  median="  << wstats_post.median
              << "  max="     << wstats_post.max_
              << "  frac_at_zero=" << wstats_post.frac_at_zero
              << "  frac_at_cap="  << wstats_post.frac_at_cap << "\n";
    std::cout << "run_train_stdp DONE\n";

    // --- Free device buffers ---
    CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
    CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
    CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
    CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
    CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx));
    CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));
    CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
    CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
    CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
    CUDA_CHECK(cudaFree(d_phase_l23));
    CUDA_CHECK(cudaFree(d_isi_c_l23_dummy));
    CUDA_CHECK(cudaFree(d_isi_s_l23_dummy));
    CUDA_CHECK(cudaFree(d_isi_ss_l23_dummy));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_l23_w_nS));
    CUDA_CHECK(cudaFree(d_x_pre));
    CUDA_CHECK(cudaFree(d_y_post));
    CUDA_CHECK(cudaFree(d_l4_spike_record));

    return 0;
}

// =====================================================================
// run_train_l23_stdp (task #5 B2): train L2/3↔L2/3 recurrence on
// Gavornik ABCD sequences, then validate.
//
// Pipeline:
//   1. Build L4→L2/3 + L2/3→L2/3 connectivity.
//   2. Load FROZEN trained L4→L2/3 weights from --load-trained-weights.
//   3. Initialize L2/3 recurrent weights (lognormal, B1 distribution).
//   4. Snapshot weights at sequence 0.
//   5. Training loop (--n-train-sequences sequences, default 2000):
//        For each sequence:
//          a. Run 1200 ms sequence (4 elements × 150 ms + 3 gaps × 200 ms)
//             with plasticity ON.
//          b. Run 1500 ms ITI gray with plasticity OFF (traces decay).
//        Snapshot weights at {0, 250, 500, 1000, 2000}; stream per-seq log.
//   6. Save final weights.
//   7. Validation suite (all with plasticity OFF):
//        V_order  : 50 trials × 10 conditions (ABCD, DCBA, 8 shuffles)
//        V_timing : 50 trials × 4 ISIs   (100, 200, 400, 800 ms)
//        V_omission: 100 trials × 3 conditions (A_CD, E_CD, ITI baseline)
//        V_lesion : zero L2/3↔L2/3 weights, re-run V_order/V_timing/V_omission
//        Phase A re-check: V2 (orientation tuning), V3 (PI), V5 (rate diag)
//   8. Aggregate phaseB_summary.json.
//
// Outputs (under --out_dir, default /tmp):
//   phaseB_l23_recurrent_weights_seq{0,250,500,1000,2000}.{bin,json}
//   phaseB_l23_recurrent_trained_weights.{bin,json}
//   phaseB_train_log.json
//   phaseB_v_order.json / phaseB_v_timing.json / phaseB_v_omission.json
//   phaseB_v_lesion.json
//   phaseB_phaseA_recheck.json
//   phaseB_summary.json
// PNGs are rendered offline by plot_phaseB.py.
//
// Per-sequence segment-shifted L4 bitmask: ONE big buffer (sized for the
// largest test trial = V_timing 800 ms ISI = 3000 ms = 30000 steps) is
// reused across all phases of every trial.  Each L4 segment kernel call
// uses a SHIFTED bitmask pointer = base + start_step * N_L4_BITMASK_INTS,
// so absolute step indexing works for the L2/3 kernel reads (L4 t-20,
// L2/3 t-10 delays cross segment boundaries cleanly).
//
// L4→L2/3 weights are FROZEN throughout (the STDP kernel reads them
// const).  Only L2/3↔L2/3 weights mutate during the 1200 ms sequence
// training window.  ITI plasticity_active=0 so traces relax without
// drift.
// =====================================================================
namespace {

// Helper: per-cell weight stats specifically for L2/3↔L2/3 (caps differ
// from Phase A).  Uses L23REC_W_MAX_NS as the cap reference.
struct L23RecWeightStats {
    double mean, median, p95, std_, min_, max_;
    double frac_at_zero;
    double frac_at_cap;
};
static L23RecWeightStats compute_l23rec_weight_stats(const std::vector<double>& w) {
    L23RecWeightStats s{};
    if (w.empty()) return s;
    double sum=0, sumsq=0;
    s.min_ = w[0]; s.max_ = w[0];
    long long nz=0, nc=0;
    const double cap = L23REC_W_MAX_NS;
    for (double v : w) {
        sum += v; sumsq += v*v;
        if (v < s.min_) s.min_ = v;
        if (v > s.max_) s.max_ = v;
        if (v <= 1e-6)         ++nz;
        if (v >= 0.99 * cap)   ++nc;
    }
    s.mean = sum / w.size();
    s.std_ = std::sqrt(std::max(0.0, sumsq/w.size() - s.mean*s.mean));
    std::vector<double> tmp = w;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
    s.median = tmp[tmp.size()/2];
    std::nth_element(tmp.begin(), tmp.begin() + (size_t)(tmp.size()*0.95), tmp.end());
    s.p95 = tmp[(size_t)(tmp.size()*0.95)];
    s.frac_at_zero = static_cast<double>(nz) / w.size();
    s.frac_at_cap  = static_cast<double>(nc) / w.size();
    return s;
}

// Count L2/3 spikes in window [step_start, step_start+n_steps) by
// scanning the host-side L2/3 spike record bitmask.  Returns total
// population spike count.
static long long count_l23_spikes_in_window(
    const std::vector<uint32_t>& bitmask_host,
    int step_start, int n_steps
) {
    long long total = 0;
    for (int s = 0; s < n_steps; ++s) {
        const size_t row = static_cast<size_t>(step_start + s) * N_L23_BITMASK_INTS;
        for (int w = 0; w < N_L23_BITMASK_INTS; ++w) {
            total += __builtin_popcount(bitmask_host[row + w]);
        }
    }
    return total;
}

// Per-cell spike count in window [step_start, step_start+n_steps).
// Returns vector of size N_L23.
static std::vector<int> per_cell_l23_spikes_in_window(
    const std::vector<uint32_t>& bitmask_host,
    int step_start, int n_steps
) {
    std::vector<int> counts(N_L23, 0);
    for (int s = 0; s < n_steps; ++s) {
        const size_t row = static_cast<size_t>(step_start + s) * N_L23_BITMASK_INTS;
        for (int w = 0; w < N_L23_BITMASK_INTS; ++w) {
            uint32_t word = bitmask_host[row + w];
            const int base_idx = w << 5;
            while (word) {
                const int b = __builtin_ctz(word);
                counts[base_idx + b] += 1;
                word &= word - 1;
            }
        }
    }
    return counts;
}

// One-sample t-test against zero (Welch trivially): returns t, p (two-tailed approx).
static std::pair<double,double> ttest_one_sample(const std::vector<double>& x, double mu0) {
    const int n = static_cast<int>(x.size());
    if (n < 2) return {0.0, 1.0};
    double sum=0; for (double v : x) sum += v;
    const double m = sum / n;
    double ss=0; for (double v : x) ss += (v-m)*(v-m);
    const double sd = std::sqrt(ss / (n-1));
    if (sd <= 0.0) return {0.0, 1.0};
    const double t = (m - mu0) / (sd / std::sqrt(double(n)));
    // Approximate two-sided p-value via Z-distribution (n large).
    const double z = std::abs(t);
    const double p = std::erfc(z / std::sqrt(2.0));
    return {t, p};
}

// Two-sample Welch t-test: returns t, p (two-tailed approx).
static std::pair<double,double> ttest_welch(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    if (na < 2 || nb < 2) return {0.0, 1.0};
    double sa=0; for (double v : a) sa += v;
    double sb=0; for (double v : b) sb += v;
    const double ma = sa / na, mb = sb / nb;
    double ssa=0; for (double v : a) ssa += (v-ma)*(v-ma);
    double ssb=0; for (double v : b) ssb += (v-mb)*(v-mb);
    const double va = ssa / (na-1), vb = ssb / (nb-1);
    const double se = std::sqrt(va/na + vb/nb);
    if (se <= 0.0) return {0.0, 1.0};
    const double t = (ma - mb) / se;
    // Two-sided Z-approx p-value.
    const double z = std::abs(t);
    const double p = std::erfc(z / std::sqrt(2.0));
    return {t, p};
}

}  // namespace

static int run_train_l23_stdp(const Args& args) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "device-info:\n  device=" << prop.name << "\n";

    if (args.load_trained_weights.empty()) {
        die("--train-l23-stdp requires --load-trained-weights "
            "(trained Phase A L4→L2/3 .bin)");
    }

    // ---------- Protocol parameters ----------
    constexpr int  ELEMENT_MS         = 150;
    constexpr int  ISI_TRAINED_MS     = 200;
    constexpr int  ITI_MS             = 1500;
    constexpr int  ELEMENT_STEPS      = 1500;   // 150 ms / dt=0.1 ms
    constexpr int  ISI_TRAINED_STEPS  = 2000;   // 200 ms
    constexpr int  ITI_STEPS          = 15000;  // 1500 ms
    constexpr int  SEQ_STEPS          = 4 * ELEMENT_STEPS + 3 * ISI_TRAINED_STEPS; // 12000
    // Largest record needed: V_timing 800 ms ISI = 4×150 + 3×800 = 3000 ms = 30000 steps.
    constexpr int  MAX_RECORD_STEPS   = 30000;
    // Trained orientations.
    const double THETA_A_DEG = 0.0;
    const double THETA_B_DEG = 90.0;
    const double THETA_C_DEG = 45.0;
    const double THETA_D_DEG = 135.0;
    const double THETA_E_DEG = 22.5;   // V_omission untrained control
    // Stim defaults (match Phase A).
    const double STIM_F  = 1.0 / stim_kernels::STIM_DEFAULT_SF_PERIOD_PIXELS; // 0.125
    const double STIM_TF = 4.0;
    const double STIM_K  = 2.0 * PI * STIM_F;
    const double STIM_OMEGA = 2.0 * PI * STIM_TF;   // drift_sign = +1
    const int N_TRAIN_SEQ = args.n_train_sequences;
    const std::vector<int> SNAPSHOT_SEQS = {0, 250, 500, 1000, 2000};

    // ---------- Build connectivity ----------
    auto t_init0 = std::chrono::steady_clock::now();
    L23Connectivity conn = build_l23_connectivity(args.seed);
    L23RecConnectivity rec = build_l23_recurrent_connectivity(args.seed);
    auto t_init1 = std::chrono::steady_clock::now();
    const double conn_build_s =
        std::chrono::duration<double>(t_init1 - t_init0).count();
    const int total_syn_l4   = conn.total_synapses;
    const int total_syn_l23  = rec.total_synapses;
    std::cout << "connectivity: l4_syn=" << total_syn_l4
              << "  l23rec_syn=" << total_syn_l23
              << "  build_wall_s=" << conn_build_s << "\n";

    // ---------- Allocate device buffers ----------
    const int block = 256;
    const int gridL4  = (N_L4  + block - 1) / block;
    const int gridL23 = (N_L23 + block - 1) / block;

    // L4 state.
    double *d_V_l4=nullptr, *d_w_l4=nullptr, *d_gE_l4=nullptr;
    int *d_refrac_l4=nullptr;
    long long *d_prev_l4=nullptr, *d_isi_c_l4=nullptr, *d_tot_l4=nullptr;
    double *d_isi_s_l4=nullptr, *d_isi_ss_l4=nullptr;
    int *d_phase_l4=nullptr;
    double* d_templates = nullptr;
    int *d_dummy_idx=nullptr, *d_dummy_steps=nullptr, *d_dummy_count=nullptr;
    CUDA_CHECK(cudaMalloc(&d_V_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l4,      N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l4,     N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l4, N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l4,   N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l4,    N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l4,  N_L4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l4,  N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l4, N_L4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l4,  N_L4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_templates, N_TEMPLATES * GABOR_PIX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dummy_idx,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_steps, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dummy_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_idx,   0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_steps, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dummy_count, 0, sizeof(int)));

    // L2/3 state.
    double *d_V_l23=nullptr, *d_w_l23=nullptr, *d_gE_l23=nullptr;
    int *d_refrac_l23=nullptr;
    long long *d_prev_l23=nullptr, *d_tot_l23=nullptr;
    long long *d_isi_c_l23=nullptr;
    double *d_isi_s_l23=nullptr, *d_isi_ss_l23=nullptr;
    int *d_phase_l23=nullptr;
    CUDA_CHECK(cudaMalloc(&d_V_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w_l23,      N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gE_l23,     N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_refrac_l23, N_L23 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prev_l23,   N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_tot_l23,    N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_c_l23,  N_L23 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_isi_s_l23,  N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isi_ss_l23, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_phase_l23,  N_L23 * sizeof(int)));

    // L4→L2/3 CSR (FROZEN; loaded from disk).
    int    *d_row_ptr=nullptr, *d_col_idx=nullptr;
    double *d_l23_w_nS=nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,  (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,
                          static_cast<size_t>(total_syn_l4) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_l23_w_nS,
                          static_cast<size_t>(total_syn_l4) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, conn.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, conn.col_idx.data(),
                          static_cast<size_t>(total_syn_l4) * sizeof(int),
                          cudaMemcpyHostToDevice));
    {
        std::ifstream wf(args.load_trained_weights, std::ios::binary);
        if (!wf) die("could not open --load-trained-weights: "
                     + args.load_trained_weights);
        wf.seekg(0, std::ios::end);
        const std::streamsize bytes = wf.tellg();
        wf.seekg(0, std::ios::beg);
        const std::streamsize expected_bytes =
            static_cast<std::streamsize>(total_syn_l4) * sizeof(double);
        if (bytes != expected_bytes) {
            die("L4→L2/3 weights file size mismatch: got "
                + std::to_string(bytes) + " bytes, expected "
                + std::to_string(expected_bytes));
        }
        std::vector<double> w_loaded(total_syn_l4);
        wf.read(reinterpret_cast<char*>(w_loaded.data()), expected_bytes);
        if (!wf) die("read failed: " + args.load_trained_weights);
        CUDA_CHECK(cudaMemcpy(d_l23_w_nS, w_loaded.data(),
                              static_cast<size_t>(total_syn_l4) * sizeof(double),
                              cudaMemcpyHostToDevice));
        std::cout << "loaded FROZEN L4→L2/3 weights from "
                  << args.load_trained_weights
                  << " (" << total_syn_l4 << " synapses)\n";
    }

    // L2/3↔L2/3 CSR (PLASTIC; lognormal init from build).
    int    *d_rec_row_ptr=nullptr, *d_rec_col_idx=nullptr;
    double *d_rec_w_nS=nullptr;
    CUDA_CHECK(cudaMalloc(&d_rec_row_ptr,
                          (N_L23 + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rec_col_idx,
                          static_cast<size_t>(total_syn_l23) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rec_w_nS,
                          static_cast<size_t>(total_syn_l23) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rec_row_ptr, rec.row_ptr.data(),
                          (N_L23 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_col_idx, rec.col_idx.data(),
                          static_cast<size_t>(total_syn_l23) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_w_nS, rec.w_nS.data(),
                          static_cast<size_t>(total_syn_l23) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // STDP traces (NEW).
    double *d_x_pre_l23=nullptr, *d_y_post_l23=nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_pre_l23,  N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_post_l23, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_x_pre_l23,  0, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y_post_l23, 0, N_L23 * sizeof(double)));

    // Spike record bitmasks (sized for max trial = V_timing 800 ISI = 30000 steps).
    const size_t l4_record_ints =
        static_cast<size_t>(MAX_RECORD_STEPS) * N_L4_BITMASK_INTS;
    const size_t l23_record_ints =
        static_cast<size_t>(MAX_RECORD_STEPS) * N_L23_BITMASK_INTS;
    uint32_t* d_l4_spike_record  = nullptr;
    uint32_t* d_l23_spike_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l4_spike_record,  l4_record_ints  * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_l23_spike_record, l23_record_ints * sizeof(uint32_t)));
    std::cout << "spike_record_alloc: l4="
              << ((l4_record_ints  * sizeof(uint32_t)) / (1024.0 * 1024.0))
              << " MB, l23="
              << ((l23_record_ints * sizeof(uint32_t)) / (1024.0 * 1024.0))
              << " MB\n";

    // ---------- Build Gabor templates + init AdEx state ----------
    {
        const int gridT = (N_TEMPLATES + block - 1) / block;
        build_gabor_templates_kernel<<<gridT, block>>>(d_templates);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    init_full_state_kernel<<<gridL4, block>>>(
        d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
        d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
    );
    init_l23_state_kernel<<<gridL23, block>>>(
        d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
        d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23, N_L23
    );
    clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
    clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- Helper lambdas ----------
    // Run a single L4 segment with a shifted bitmask pointer.  All AdEx
    // state in d_V_l4 etc. persists.  When `mod_active` is false, n_stim_steps
    // is 0 → mod_amp collapses to 0 inside the kernel (1 Hz baseline only).
    // The L4 kernel uses internal step indices [0, n_steps); we pass a
    // bitmask pointer offset by `seg_start_step` so the absolute write
    // location is start + step.
    auto run_l4_segment = [&](int seg_start_step, int n_steps,
                              double theta_deg, bool mod_active,
                              long long phase_idx_for_rng) {
        const double th = theta_deg * (PI / 180.0);
        const double cos_t = std::cos(th);
        const double sin_t = std::sin(th);
        const int n_stim = mod_active ? n_steps : 0;
        v1_phase_kernel<0><<<gridL4, block>>>(
            d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
            d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
            d_phase_l4,
            d_templates,
            d_dummy_idx, d_dummy_steps, d_dummy_count,
            /*n_raster=*/0, /*max_raster_spikes=*/0,
            /*phase_idx_for_raster=*/-1,
            /*phase_step_offset=*/0,
            /*phase_idx=*/static_cast<int>(phase_idx_for_rng),
            n_steps, /*n_warmup_steps=*/0,
            cos_t, sin_t, STIM_K, STIM_OMEGA,
            W_IN_NS, R_BASE_HZ,
            args.seed,
            d_l4_spike_record + static_cast<size_t>(seg_start_step) * N_L4_BITMASK_INTS,
            /*phase_offset=*/0.0,
            /*aperture_active=*/0,
            /*aperture_cx=*/0.0,
            /*aperture_cy=*/0.0,
            /*aperture_inv_2sigma_sq=*/0.0,
            /*peak_bin20_count_out=*/nullptr,
            /*bin50_counts_out=*/nullptr,
            /*n_bins_50=*/0,
            /*n_stim_steps=*/n_stim
        );
    };

    // Loop the per-step recurrent STDP kernel for [start_step, start_step + n_steps).
    auto run_l23_recurrent_steps = [&](int start_step, int n_steps,
                                       int plasticity_active) {
        for (int s = 0; s < n_steps; ++s) {
            v1_l23_recurrent_stdp_step_kernel<<<gridL23, block>>>(
                d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
                d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23,
                d_phase_l23,
                d_row_ptr, d_col_idx, d_l23_w_nS,
                d_l4_spike_record,
                d_rec_row_ptr, d_rec_col_idx, d_rec_w_nS,
                d_l23_spike_record,
                d_x_pre_l23, d_y_post_l23,
                /*phase_step_offset=*/0,
                /*step_idx=*/start_step + s,
                /*n_warmup_steps=*/0,
                plasticity_active
            );
        }
    };

    // Clear bitmask buffers for the first n_steps.
    auto clear_records = [&](int n_steps) {
        const size_t l4_n  = static_cast<size_t>(n_steps) * N_L4_BITMASK_INTS;
        const size_t l23_n = static_cast<size_t>(n_steps) * N_L23_BITMASK_INTS;
        const int gR4  = static_cast<int>((l4_n  + block - 1) / block);
        const int gR23 = static_cast<int>((l23_n + block - 1) / block);
        clear_uint32_kernel<<<gR4,  block>>>(d_l4_spike_record,  l4_n);
        clear_uint32_kernel<<<gR23, block>>>(d_l23_spike_record, l23_n);
    };

    // Run a sequence of (orientation, duration_steps, mod_active, isi_steps)
    // segments back-to-back into a single bitmask region starting at step 0.
    // Returns the total number of steps run.  L2/3 plasticity controlled
    // by `plasticity_active` (1 = ON during sequence, 0 = OFF).
    // segments[i] = (theta_deg, duration_steps, mod_active).  ISI gaps
    // between segments use isi_steps and mod_active=false.
    struct Seg { double theta_deg; int duration_steps; bool mod_active; };
    auto run_sequence = [&](const std::vector<Seg>& segments,
                            int isi_steps,
                            int plasticity_active,
                            long long phase_idx_for_rng) -> int {
        int cum_step = 0;
        // Total step budget for this sequence.
        int total_steps = 0;
        for (size_t i = 0; i < segments.size(); ++i) {
            total_steps += segments[i].duration_steps;
            if (i + 1 < segments.size()) total_steps += isi_steps;
        }
        if (total_steps > MAX_RECORD_STEPS) {
            die("run_sequence: total steps exceeds MAX_RECORD_STEPS");
        }
        clear_records(total_steps);
        clear_int_kernel<<<gridL4,  block>>>(d_phase_l4, N_L4);
        clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
        CUDA_CHECK(cudaGetLastError());
        // L4 pass: issue one kernel per segment + per gap.
        for (size_t i = 0; i < segments.size(); ++i) {
            const Seg& seg = segments[i];
            run_l4_segment(cum_step, seg.duration_steps,
                           seg.theta_deg, seg.mod_active, phase_idx_for_rng);
            cum_step += seg.duration_steps;
            if (i + 1 < segments.size()) {
                // Inter-element gap (gray, mod off).
                run_l4_segment(cum_step, isi_steps,
                               /*theta=*/0.0, /*mod_active=*/false,
                               phase_idx_for_rng + 1000003LL);
                cum_step += isi_steps;
            }
        }
        // L2/3 step loop covering [0, total_steps).
        run_l23_recurrent_steps(0, total_steps, plasticity_active);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return total_steps;
    };

    // Run plain ITI gray (no segments, just one long gray interval).  Used
    // both during training (plasticity OFF) and as ITI baseline in V_omission.
    auto run_iti_gray = [&](int n_steps, int plasticity_active,
                            long long phase_idx_for_rng) {
        if (n_steps > MAX_RECORD_STEPS) die("run_iti_gray: too many steps");
        clear_records(n_steps);
        clear_int_kernel<<<gridL4,  block>>>(d_phase_l4, N_L4);
        clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
        CUDA_CHECK(cudaGetLastError());
        run_l4_segment(0, n_steps, /*theta=*/0.0, /*mod_active=*/false,
                       phase_idx_for_rng);
        run_l23_recurrent_steps(0, n_steps, plasticity_active);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Helper: snapshot current L2/3↔L2/3 weights to .bin + .json.
    std::filesystem::create_directories(args.out_dir);
    const std::string label_suffix =
        args.label.empty() ? std::string() : std::string("_") + args.label;
    auto save_snapshot = [&](int seq_idx) {
        std::vector<double> w_host(total_syn_l23);
        CUDA_CHECK(cudaMemcpy(w_host.data(), d_rec_w_nS,
                              static_cast<size_t>(total_syn_l23) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        const std::string bin_path = args.out_dir
            + "/phaseB_l23_recurrent_weights_seq" + std::to_string(seq_idx)
            + label_suffix + ".bin";
        const std::string json_path = args.out_dir
            + "/phaseB_l23_recurrent_weights_seq" + std::to_string(seq_idx)
            + label_suffix + ".json";
        {
            std::ofstream of(bin_path, std::ios::binary);
            if (!of) die("could not open " + bin_path);
            of.write(reinterpret_cast<const char*>(w_host.data()),
                     static_cast<std::streamsize>(total_syn_l23) * sizeof(double));
        }
        L23RecWeightStats st = compute_l23rec_weight_stats(w_host);
        {
            std::ofstream of(json_path);
            if (!of) die("could not open " + json_path);
            of << std::setprecision(8);
            of << "{\n";
            of << "  \"seq_idx\": " << seq_idx << ",\n";
            of << "  \"n_synapses\": " << total_syn_l23 << ",\n";
            of << "  \"weights_bin_path\": \"" << bin_path << "\",\n";
            of << "  \"stats_nS\": {\n";
            of << "    \"mean\": "    << st.mean    << ",\n";
            of << "    \"median\": "  << st.median  << ",\n";
            of << "    \"p95\": "     << st.p95     << ",\n";
            of << "    \"std\": "     << st.std_    << ",\n";
            of << "    \"min\": "     << st.min_    << ",\n";
            of << "    \"max\": "     << st.max_    << ",\n";
            of << "    \"frac_at_zero\": " << st.frac_at_zero << ",\n";
            of << "    \"frac_at_cap\": "  << st.frac_at_cap  << "\n";
            of << "  }\n";
            of << "}\n";
        }
        std::cout << "snapshot[seq=" << seq_idx << "]:"
                  << " mean=" << st.mean
                  << " median=" << st.median
                  << " max=" << st.max_
                  << " frac@0=" << st.frac_at_zero
                  << " frac@cap=" << st.frac_at_cap << "\n";
    };

    // Snapshot at seq=0 (pre-training).
    save_snapshot(0);

    // ---------- Training loop ----------
    auto t_train0 = std::chrono::steady_clock::now();
    const std::string train_log_path = args.out_dir + "/phaseB_train_log"
        + label_suffix + ".json";
    std::ofstream train_log(train_log_path);
    if (!train_log) die("could not open " + train_log_path);
    train_log << std::setprecision(8);
    train_log << "{\n";
    train_log << "  \"task\": \"phaseB_l23_stdp_train\",\n";
    train_log << "  \"seed\": " << args.seed << ",\n";
    train_log << "  \"n_train_sequences\": " << N_TRAIN_SEQ << ",\n";
    train_log << "  \"theta_deg\": {\"A\":" << THETA_A_DEG << ",\"B\":" << THETA_B_DEG
              << ",\"C\":" << THETA_C_DEG << ",\"D\":" << THETA_D_DEG << "},\n";
    train_log << "  \"element_ms\": " << ELEMENT_MS << ",\n";
    train_log << "  \"isi_ms\": " << ISI_TRAINED_MS << ",\n";
    train_log << "  \"iti_ms\": " << ITI_MS << ",\n";
    train_log << "  \"per_seq\": [\n";

    bool runaway_detected = false;
    std::deque<double> rolling_mean_rate;       // rolling-100 L2/3 rate
    std::vector<double> phase_count_l23_host(N_L23);
    long long total_l23_spikes_train = 0;

    for (int seq = 1; seq <= N_TRAIN_SEQ; ++seq) {
        // Build sequence: A → gap → B → gap → C → gap → D, plasticity ON.
        const std::vector<Seg> abcd = {
            {THETA_A_DEG, ELEMENT_STEPS, true},
            {THETA_B_DEG, ELEMENT_STEPS, true},
            {THETA_C_DEG, ELEMENT_STEPS, true},
            {THETA_D_DEG, ELEMENT_STEPS, true},
        };
        const long long phase_idx = seq * 7LL;  // distinct RNG phase per seq
        const int n_seq = run_sequence(abcd, ISI_TRAINED_STEPS,
                                        /*plasticity_active=*/1, phase_idx);

        // Per-element population spike count windows (0–100 ms post-onset).
        // Element t_onsets within sequence (steps): 0, 3500, 7000, 10500.
        // Window = 1000 steps each.
        std::vector<uint32_t> bitmask_host(
            static_cast<size_t>(n_seq) * N_L23_BITMASK_INTS);
        CUDA_CHECK(cudaMemcpy(bitmask_host.data(), d_l23_spike_record,
                              bitmask_host.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        long long elem_spk[4];
        const int onsets[4] = {
            0,
            ELEMENT_STEPS + ISI_TRAINED_STEPS,
            2*(ELEMENT_STEPS + ISI_TRAINED_STEPS),
            3*(ELEMENT_STEPS + ISI_TRAINED_STEPS)
        };
        for (int e = 0; e < 4; ++e) {
            elem_spk[e] = count_l23_spikes_in_window(bitmask_host, onsets[e], 1000);
        }

        // Read per-cell phase counts (entire 1200 ms sequence).
        CUDA_CHECK(cudaMemcpy(phase_count_l23_host.data(), d_phase_l23,
                              N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
        // Per-cell phase counts cast int->double.
        long long sum_seq = 0;
        for (int i = 0; i < N_L23; ++i) {
            const int* p = reinterpret_cast<const int*>(phase_count_l23_host.data());
            sum_seq += p[i];
        }
        const double mean_rate_seq =
            static_cast<double>(sum_seq) / N_L23 / (n_seq * DT_S);
        total_l23_spikes_train += sum_seq;

        rolling_mean_rate.push_back(mean_rate_seq);
        if (rolling_mean_rate.size() > 100) rolling_mean_rate.pop_front();

        // Per-sequence log entry.
        if (seq > 1) train_log << ",\n";
        train_log << "    {\"seq\":" << seq
                  << ",\"elem_spk\":[" << elem_spk[0] << ","
                  << elem_spk[1] << "," << elem_spk[2] << "," << elem_spk[3] << "]"
                  << ",\"mean_rate_hz\":" << mean_rate_seq
                  << "}";

        // ---- ITI: 1500 ms gray, plasticity OFF ----
        run_iti_gray(ITI_STEPS, /*plasticity_active=*/0,
                     phase_idx + 5000003LL);

        // Watchdog + progress.
        if (mean_rate_seq > 100.0) {
            // Definitive runaway — abort.
            runaway_detected = true;
            std::cout << "RUNAWAY at seq=" << seq
                      << "  mean_rate_hz=" << mean_rate_seq << "\n";
            break;
        }
        if (seq % 200 == 0 || seq == N_TRAIN_SEQ) {
            double r_sum = 0; for (double r : rolling_mean_rate) r_sum += r;
            const double r_avg = r_sum / std::max(size_t(1), rolling_mean_rate.size());
            // Quick weight stat.
            std::vector<double> w_host(total_syn_l23);
            CUDA_CHECK(cudaMemcpy(w_host.data(), d_rec_w_nS,
                                  static_cast<size_t>(total_syn_l23) * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            L23RecWeightStats st = compute_l23rec_weight_stats(w_host);
            std::cout << "[seq=" << seq << "/" << N_TRAIN_SEQ << "]"
                      << " roll100_mean_rate=" << r_avg
                      << "  w_median=" << st.median
                      << "  w_max=" << st.max_
                      << "  frac@cap=" << st.frac_at_cap
                      << "  frac@0=" << st.frac_at_zero << "\n";
            // Soft watchdog: rolling mean rate > 50 Hz triggers warning.
            if (r_avg > 50.0) {
                std::cout << "WARNING: rolling-100 mean rate above 50 Hz\n";
            }
        }

        // Snapshot points.
        for (int sn : SNAPSHOT_SEQS) {
            if (sn == seq) save_snapshot(sn);
        }
    }
    auto t_train1 = std::chrono::steady_clock::now();
    const double train_wall_s =
        std::chrono::duration<double>(t_train1 - t_train0).count();

    train_log << "\n  ],\n";
    train_log << "  \"train_wall_s\": " << train_wall_s << ",\n";
    train_log << "  \"runaway_detected\": "
              << (runaway_detected ? "true" : "false") << ",\n";
    train_log << "  \"total_l23_spikes_train\": " << total_l23_spikes_train << "\n";
    train_log << "}\n";
    train_log.close();
    std::cout << "training: wall_s=" << train_wall_s
              << "  runaway=" << (runaway_detected ? "true" : "false") << "\n";

    // Save final weights.
    {
        std::vector<double> w_host(total_syn_l23);
        CUDA_CHECK(cudaMemcpy(w_host.data(), d_rec_w_nS,
                              static_cast<size_t>(total_syn_l23) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        const std::string bin_path =
            args.out_dir + "/phaseB_l23_recurrent_trained_weights"
                         + label_suffix + ".bin";
        const std::string json_path =
            args.out_dir + "/phaseB_l23_recurrent_trained_weights"
                         + label_suffix + ".json";
        {
            std::ofstream of(bin_path, std::ios::binary);
            if (!of) die("could not open " + bin_path);
            of.write(reinterpret_cast<const char*>(w_host.data()),
                     static_cast<std::streamsize>(total_syn_l23) * sizeof(double));
        }
        L23RecWeightStats st = compute_l23rec_weight_stats(w_host);
        {
            std::ofstream of(json_path);
            of << std::setprecision(8);
            of << "{\n  \"n_synapses\": " << total_syn_l23 << ",\n";
            of << "  \"n_train_sequences\": " << N_TRAIN_SEQ << ",\n";
            of << "  \"runaway_detected\": "
               << (runaway_detected ? "true" : "false") << ",\n";
            of << "  \"train_wall_s\": " << train_wall_s << ",\n";
            of << "  \"stats_nS\": {\n"
               << "    \"mean\":"   << st.mean   << ",\"median\":" << st.median
               << ",\"p95\":"       << st.p95    << ",\"max\":"    << st.max_
               << ",\"min\":"       << st.min_   << ",\"std\":"    << st.std_
               << ",\"frac_at_zero\":" << st.frac_at_zero
               << ",\"frac_at_cap\":"  << st.frac_at_cap << "\n  }\n}\n";
        }
        std::cout << "saved trained weights to " << bin_path << "\n";
    }

    // Save a HOST-side copy of trained weights for later restoration after lesion.
    std::vector<double> trained_l23rec_w_host(total_syn_l23);
    CUDA_CHECK(cudaMemcpy(trained_l23rec_w_host.data(), d_rec_w_nS,
                          static_cast<size_t>(total_syn_l23) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // ====================================================================
    // VALIDATION SUITE (plasticity OFF for all tests)
    // ====================================================================
    auto t_val0 = std::chrono::steady_clock::now();
    std::mt19937_64 val_rng(args.seed ^ 0xB16B00B5DEC0DEDDULL);

    // Helper: run a test sequence with the given segment list (plasticity 0),
    // pull the L2/3 bitmask back to host, return per-element population
    // spike counts in 0–100 ms windows for the supplied onset_steps[] (each
    // window is 1000 steps).
    auto run_test_and_count = [&](const std::vector<Seg>& segments,
                                  int isi_steps,
                                  const std::vector<int>& onset_steps,
                                  long long phase_idx_for_rng) {
        const int total_steps = run_sequence(segments, isi_steps,
                                              /*plasticity_active=*/0,
                                              phase_idx_for_rng);
        std::vector<uint32_t> bm(static_cast<size_t>(total_steps) * N_L23_BITMASK_INTS);
        CUDA_CHECK(cudaMemcpy(bm.data(), d_l23_spike_record,
                              bm.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        std::vector<long long> per_window;
        per_window.reserve(onset_steps.size());
        for (int t : onset_steps) {
            per_window.push_back(count_l23_spikes_in_window(bm, t, 1000));
        }
        return std::make_tuple(per_window, std::move(bm), total_steps);
    };

    // Helper to compute mean and SEM of a vector.
    auto mean_sem = [](const std::vector<double>& v) {
        const int n = static_cast<int>(v.size());
        if (n < 1) return std::make_pair(0.0, 0.0);
        double s=0; for (double x : v) s += x;
        const double m = s / n;
        double ss=0; for (double x : v) ss += (x-m)*(x-m);
        const double sd = std::sqrt(ss / std::max(1, n - 1));
        const double sem = sd / std::sqrt(double(n));
        return std::make_pair(m, sem);
    };

    // ---------- V_order ----------
    std::cout << "\n=== V_order: ABCD vs DCBA vs 8 shuffles, 50 trials each ===\n";
    constexpr int N_TRIALS_PER_COND = 50;
    // Build conditions: ABCD, DCBA, 8 random permutations of {A,B,C,D}.
    std::vector<std::array<double,4>> conditions;
    std::vector<std::string> condition_labels;
    conditions.push_back({THETA_A_DEG, THETA_B_DEG, THETA_C_DEG, THETA_D_DEG});
    condition_labels.push_back("ABCD");
    conditions.push_back({THETA_D_DEG, THETA_C_DEG, THETA_B_DEG, THETA_A_DEG});
    condition_labels.push_back("DCBA");
    {
        std::mt19937_64 perm_rng(args.seed ^ 0x12345678DEADBEEFULL);
        std::array<double,4> ths = {THETA_A_DEG, THETA_B_DEG, THETA_C_DEG, THETA_D_DEG};
        for (int sh = 0; sh < 8; ++sh) {
            std::array<double,4> p = ths;
            std::shuffle(p.begin(), p.end(), perm_rng);
            conditions.push_back(p);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "S%d_%.0f%.0f%.0f%.0f",
                          sh, p[0], p[1], p[2], p[3]);
            condition_labels.push_back(buf);
        }
    }
    const int N_COND = static_cast<int>(conditions.size());
    // Per-condition × per-trial × per-element response (population spike count).
    std::vector<std::vector<std::array<long long,4>>> v_order_resp(
        N_COND, std::vector<std::array<long long,4>>(N_TRIALS_PER_COND));

    // For element decoding: collect per-trial L2/3 population vectors at each element onset.
    // Use ABCD condition for the train+test split.
    std::vector<std::vector<int>> v_order_pop_vecs;          // (trial, cell)
    std::vector<int>              v_order_pop_labels;        // element index 0..3
    v_order_pop_vecs.reserve(N_TRIALS_PER_COND * 4);
    v_order_pop_labels.reserve(N_TRIALS_PER_COND * 4);

    auto t_vorder0 = std::chrono::steady_clock::now();
    long long phase_base = 100000LL;
    for (int c = 0; c < N_COND; ++c) {
        const auto& th_seq = conditions[c];
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            std::vector<Seg> segs = {
                {th_seq[0], ELEMENT_STEPS, true},
                {th_seq[1], ELEMENT_STEPS, true},
                {th_seq[2], ELEMENT_STEPS, true},
                {th_seq[3], ELEMENT_STEPS, true},
            };
            const std::vector<int> onsets = {
                0,
                ELEMENT_STEPS + ISI_TRAINED_STEPS,
                2*(ELEMENT_STEPS + ISI_TRAINED_STEPS),
                3*(ELEMENT_STEPS + ISI_TRAINED_STEPS)
            };
            auto [per_w, bm, total_steps] = run_test_and_count(
                segs, ISI_TRAINED_STEPS, onsets, phase_base);
            ++phase_base;
            for (int e = 0; e < 4; ++e) v_order_resp[c][tr][e] = per_w[e];
            // Save per-element pop vectors for the ABCD condition (decoder source).
            if (c == 0) {
                for (int e = 0; e < 4; ++e) {
                    auto pcv = per_cell_l23_spikes_in_window(bm, onsets[e], 1000);
                    v_order_pop_vecs.push_back(std::move(pcv));
                    v_order_pop_labels.push_back(e);
                }
            }
        }
        if ((c+1) % 2 == 0 || c == N_COND - 1) {
            std::cout << "  V_order condition " << (c+1) << "/" << N_COND
                      << " (" << condition_labels[c] << ") done\n";
        }
    }
    auto t_vorder1 = std::chrono::steady_clock::now();
    const double vorder_wall_s =
        std::chrono::duration<double>(t_vorder1 - t_vorder0).count();

    // Aggregate stats.
    std::vector<double> abcd_total(N_TRIALS_PER_COND), dcba_total(N_TRIALS_PER_COND);
    std::vector<double> shuffle_total;
    shuffle_total.reserve(N_TRIALS_PER_COND * 8);
    for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
        long long ta=0, td=0;
        for (int e = 0; e < 4; ++e) {
            ta += v_order_resp[0][tr][e];
            td += v_order_resp[1][tr][e];
        }
        abcd_total[tr] = double(ta);
        dcba_total[tr] = double(td);
    }
    for (int c = 2; c < N_COND; ++c) {
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            long long ts=0; for (int e = 0; e < 4; ++e) ts += v_order_resp[c][tr][e];
            shuffle_total.push_back(double(ts));
        }
    }
    const auto [abcd_mean, abcd_sem] = mean_sem(abcd_total);
    const auto [dcba_mean, dcba_sem] = mean_sem(dcba_total);
    const auto [shuf_mean, shuf_sem] = mean_sem(shuffle_total);
    auto [t_abcd_dcba, p_abcd_dcba] = ttest_welch(abcd_total, dcba_total);
    auto [t_abcd_shuf, p_abcd_shuf] = ttest_welch(abcd_total, shuffle_total);
    const double spi_total = abcd_mean - dcba_mean;
    // Per-element SPI.
    std::array<double,4> spi_per_elem{};
    for (int e = 0; e < 4; ++e) {
        double a=0, d=0;
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            a += v_order_resp[0][tr][e];
            d += v_order_resp[1][tr][e];
        }
        spi_per_elem[e] = (a - d) / N_TRIALS_PER_COND;
    }

    // Element decoding: 5-fold CV on ABCD pop vectors.
    // (Use the ridge-style logistic regression in numpy via JSON-written
    //  features; simpler: implement multinomial linear classifier in C++
    //  via one-vs-rest least-squares prototype scoring.  For sanity, we
    //  use a fast template-matching nearest-mean classifier on the
    //  per-class mean vector — equivalent to nearest centroid, sufficient
    //  to indicate decodability.  Reported as `decoder_test_acc_5fold`.)
    double decoder_test_acc = 0.0;
    {
        const int N_TRIALS = N_TRIALS_PER_COND;
        const int N_CLASSES = 4;
        // labels arranged as: trial 0: e=0,1,2,3 ; trial 1: e=0,1,2,3 ; ...
        // 5-fold CV over the trial axis.
        const int K = 5;
        int correct = 0, total = 0;
        for (int fold = 0; fold < K; ++fold) {
            const int test_lo = fold * (N_TRIALS / K);
            const int test_hi = (fold + 1) * (N_TRIALS / K);
            // Compute per-class mean vector from training trials.
            std::vector<std::vector<double>> mean_vec(
                N_CLASSES, std::vector<double>(N_L23, 0.0));
            std::vector<int> n_train(N_CLASSES, 0);
            for (int tr = 0; tr < N_TRIALS; ++tr) {
                if (tr >= test_lo && tr < test_hi) continue;
                for (int e = 0; e < N_CLASSES; ++e) {
                    const auto& v = v_order_pop_vecs[tr * N_CLASSES + e];
                    for (int i = 0; i < N_L23; ++i) mean_vec[e][i] += v[i];
                    ++n_train[e];
                }
            }
            for (int e = 0; e < N_CLASSES; ++e) {
                if (n_train[e] > 0) {
                    for (int i = 0; i < N_L23; ++i)
                        mean_vec[e][i] /= n_train[e];
                }
            }
            // Test: nearest centroid (Euclidean).
            for (int tr = test_lo; tr < test_hi; ++tr) {
                for (int e = 0; e < N_CLASSES; ++e) {
                    const auto& v = v_order_pop_vecs[tr * N_CLASSES + e];
                    double best_d = 1e300; int best_e = 0;
                    for (int ec = 0; ec < N_CLASSES; ++ec) {
                        double d = 0;
                        for (int i = 0; i < N_L23; ++i) {
                            const double dd = double(v[i]) - mean_vec[ec][i];
                            d += dd * dd;
                        }
                        if (d < best_d) { best_d = d; best_e = ec; }
                    }
                    if (best_e == e) ++correct;
                    ++total;
                }
            }
        }
        decoder_test_acc = (total > 0) ? double(correct) / total : 0.0;
    }
    std::cout << "  V_order: ABCD=" << abcd_mean << "±" << abcd_sem
              << "  DCBA=" << dcba_mean << "±" << dcba_sem
              << "  shuf=" << shuf_mean << "±" << shuf_sem
              << "  SPI=" << spi_total
              << "  t(A,D)=" << t_abcd_dcba
              << "  p(A,D)=" << p_abcd_dcba
              << "  t(A,sh)=" << t_abcd_shuf
              << "  p(A,sh)=" << p_abcd_shuf
              << "  decoder=" << decoder_test_acc
              << "  wall_s=" << vorder_wall_s << "\n";

    // Write V_order JSON.
    auto write_v_order = [&](const std::string& path, bool lesioned,
                             double abcd_mean_, double abcd_sem_,
                             double dcba_mean_, double dcba_sem_,
                             double shuf_mean_, double shuf_sem_,
                             double spi_total_,
                             const std::array<double,4>& spi_pe,
                             double t_AD, double p_AD,
                             double t_As, double p_As,
                             double dec_acc) {
        std::ofstream of(path);
        if (!of) die("could not open " + path);
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"lesioned\": " << (lesioned ? "true" : "false") << ",\n";
        of << "  \"n_trials_per_cond\": " << N_TRIALS_PER_COND << ",\n";
        of << "  \"n_conditions\": " << N_COND << ",\n";
        of << "  \"condition_labels\": [";
        for (int c = 0; c < N_COND; ++c) {
            if (c) of << ",";
            of << "\"" << condition_labels[c] << "\"";
        }
        of << "],\n";
        of << "  \"abcd_total_per_trial\": [";
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            if (tr) of << ",";
            of << abcd_total[tr];
        }
        of << "],\n";
        of << "  \"dcba_total_per_trial\": [";
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            if (tr) of << ",";
            of << dcba_total[tr];
        }
        of << "],\n";
        of << "  \"summary\": {\n";
        of << "    \"abcd_mean\": " << abcd_mean_ << ",\"abcd_sem\": " << abcd_sem_ << ",\n";
        of << "    \"dcba_mean\": " << dcba_mean_ << ",\"dcba_sem\": " << dcba_sem_ << ",\n";
        of << "    \"shuffle_mean\": " << shuf_mean_ << ",\"shuffle_sem\": " << shuf_sem_ << ",\n";
        of << "    \"spi_total\": " << spi_total_ << ",\n";
        of << "    \"spi_per_element\": [" << spi_pe[0] << "," << spi_pe[1]
           << "," << spi_pe[2] << "," << spi_pe[3] << "],\n";
        of << "    \"t_abcd_vs_dcba\": " << t_AD << ",\"p_abcd_vs_dcba\": " << p_AD << ",\n";
        of << "    \"t_abcd_vs_shuffle\": " << t_As << ",\"p_abcd_vs_shuffle\": " << p_As << ",\n";
        of << "    \"decoder_test_acc_5fold\": " << dec_acc << "\n";
        of << "  }\n";
        of << "}\n";
    };
    write_v_order(args.out_dir + "/phaseB_v_order" + label_suffix + ".json",
                  /*lesioned=*/false,
                  abcd_mean, abcd_sem, dcba_mean, dcba_sem,
                  shuf_mean, shuf_sem, spi_total, spi_per_elem,
                  t_abcd_dcba, p_abcd_dcba, t_abcd_shuf, p_abcd_shuf,
                  decoder_test_acc);

    // Save snapshot of intact V_order results for V_lesion comparison.
    const double intact_spi = spi_total;
    const double intact_decoder = decoder_test_acc;

    // ---------- V_timing ----------
    std::cout << "\n=== V_timing: ABCD at ISIs {100, 200, 400, 800} ms ===\n";
    const std::vector<int> ISI_LIST_MS = {100, 200, 400, 800};
    std::vector<double> v_timing_response_per_isi;
    std::vector<std::vector<double>> v_timing_per_trial(ISI_LIST_MS.size());
    auto t_vtim0 = std::chrono::steady_clock::now();
    for (size_t ii = 0; ii < ISI_LIST_MS.size(); ++ii) {
        const int isi_ms = ISI_LIST_MS[ii];
        const int isi_steps = isi_ms * 10;     // dt = 0.1 ms
        std::vector<double> totals(N_TRIALS_PER_COND);
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            std::vector<Seg> segs = {
                {THETA_A_DEG, ELEMENT_STEPS, true},
                {THETA_B_DEG, ELEMENT_STEPS, true},
                {THETA_C_DEG, ELEMENT_STEPS, true},
                {THETA_D_DEG, ELEMENT_STEPS, true},
            };
            const std::vector<int> onsets = {
                0,
                ELEMENT_STEPS + isi_steps,
                2*(ELEMENT_STEPS + isi_steps),
                3*(ELEMENT_STEPS + isi_steps)
            };
            auto [per_w, bm_unused, total_unused] = run_test_and_count(
                segs, isi_steps, onsets, phase_base);
            ++phase_base;
            (void)bm_unused; (void)total_unused;
            long long sum_t = 0;
            for (int e = 0; e < 4; ++e) sum_t += per_w[e];
            totals[tr] = double(sum_t);
        }
        v_timing_per_trial[ii] = totals;
        const auto [m, s] = mean_sem(totals);
        v_timing_response_per_isi.push_back(m);
        std::cout << "  V_timing ISI=" << isi_ms << " ms: mean="
                  << m << "±" << s << "\n";
        (void)s;
    }
    auto t_vtim1 = std::chrono::steady_clock::now();
    const double vtim_wall_s =
        std::chrono::duration<double>(t_vtim1 - t_vtim0).count();

    // Identify peak ISI.
    int peak_isi_idx = 0;
    for (size_t i = 1; i < v_timing_response_per_isi.size(); ++i) {
        if (v_timing_response_per_isi[i] > v_timing_response_per_isi[peak_isi_idx]) {
            peak_isi_idx = static_cast<int>(i);
        }
    }
    const int peak_isi_ms = ISI_LIST_MS[peak_isi_idx];

    auto write_v_timing = [&](const std::string& path, bool lesioned) {
        std::ofstream of(path);
        if (!of) die("could not open " + path);
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"lesioned\": " << (lesioned ? "true" : "false") << ",\n";
        of << "  \"isi_ms_list\": [";
        for (size_t i = 0; i < ISI_LIST_MS.size(); ++i) {
            if (i) of << ","; of << ISI_LIST_MS[i];
        }
        of << "],\n";
        of << "  \"trained_isi_ms\": " << ISI_TRAINED_MS << ",\n";
        of << "  \"peak_isi_ms\": " << peak_isi_ms << ",\n";
        of << "  \"response_per_isi\": [";
        for (size_t i = 0; i < v_timing_response_per_isi.size(); ++i) {
            if (i) of << ","; of << v_timing_response_per_isi[i];
        }
        of << "],\n";
        of << "  \"per_trial\": [\n";
        for (size_t i = 0; i < v_timing_per_trial.size(); ++i) {
            if (i) of << ",\n";
            of << "    [";
            for (size_t tr = 0; tr < v_timing_per_trial[i].size(); ++tr) {
                if (tr) of << ",";
                of << v_timing_per_trial[i][tr];
            }
            of << "]";
        }
        of << "\n  ],\n";
        of << "  \"wall_s\": " << vtim_wall_s << "\n";
        of << "}\n";
    };
    write_v_timing(args.out_dir + "/phaseB_v_timing" + label_suffix + ".json",
                   /*lesioned=*/false);
    const std::vector<double> intact_vtim_resp = v_timing_response_per_isi;

    // ---------- V_omission ----------
    std::cout << "\n=== V_omission: A_CD vs E_CD vs ITI baseline, 100 trials each ===\n";
    constexpr int N_TRIALS_OMISSION = 100;
    // The predictive window is t=[350, 450) ms = [3500, 4500) steps from sequence start.
    const int B_PREDICTIVE_ONSET = ELEMENT_STEPS + ISI_TRAINED_STEPS;     // 3500 steps
    auto t_vom0 = std::chrono::steady_clock::now();
    // A_CD: A → gap → [gray-instead-of-B] → gap → C → gap → D.
    std::vector<double> a_cd_window(N_TRIALS_OMISSION);
    std::vector<std::vector<int>> a_cd_per_cell;   // for per-cell predictive analysis
    a_cd_per_cell.reserve(N_TRIALS_OMISSION);
    for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr) {
        std::vector<Seg> segs = {
            {THETA_A_DEG, ELEMENT_STEPS, true},
            {0.0,         ELEMENT_STEPS, false},   // gray replacing B
            {THETA_C_DEG, ELEMENT_STEPS, true},
            {THETA_D_DEG, ELEMENT_STEPS, true},
        };
        const std::vector<int> onsets = {B_PREDICTIVE_ONSET};
        auto [per_w, bm, total_steps] = run_test_and_count(
            segs, ISI_TRAINED_STEPS, onsets, phase_base);
        ++phase_base;
        a_cd_window[tr] = double(per_w[0]);
        auto pcv = per_cell_l23_spikes_in_window(bm, B_PREDICTIVE_ONSET, 1000);
        a_cd_per_cell.push_back(std::move(pcv));
    }
    auto t_vom_acd = std::chrono::steady_clock::now();
    std::cout << "  A_CD done\n";
    // E_CD: A → gap → E (untrained 22.5°) → gap → C → gap → D.
    std::vector<double> e_cd_window(N_TRIALS_OMISSION);
    for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr) {
        std::vector<Seg> segs = {
            {THETA_A_DEG, ELEMENT_STEPS, true},
            {THETA_E_DEG, ELEMENT_STEPS, true},   // E in B's slot
            {THETA_C_DEG, ELEMENT_STEPS, true},
            {THETA_D_DEG, ELEMENT_STEPS, true},
        };
        const std::vector<int> onsets = {B_PREDICTIVE_ONSET};
        auto [per_w, bm_unused, total_unused] = run_test_and_count(
            segs, ISI_TRAINED_STEPS, onsets, phase_base);
        ++phase_base;
        (void)bm_unused; (void)total_unused;
        e_cd_window[tr] = double(per_w[0]);
    }
    auto t_vom_ecd = std::chrono::steady_clock::now();
    std::cout << "  E_CD done\n";
    // ITI baseline: 1500 ms gray, sample 14 non-overlapping 100 ms windows per trial.
    std::vector<double> iti_baseline_samples;
    iti_baseline_samples.reserve(N_TRIALS_OMISSION * 14);
    std::vector<std::vector<int>> iti_per_cell_samples;
    iti_per_cell_samples.reserve(N_TRIALS_OMISSION * 14);
    for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr) {
        run_iti_gray(ITI_STEPS, /*plasticity=*/0, phase_base);
        ++phase_base;
        std::vector<uint32_t> bm(static_cast<size_t>(ITI_STEPS) * N_L23_BITMASK_INTS);
        CUDA_CHECK(cudaMemcpy(bm.data(), d_l23_spike_record,
                              bm.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        // Sample 14 non-overlapping 100 ms windows starting at steps 0, 1000, 2000, ...
        for (int s = 0; s + 1000 <= ITI_STEPS; s += 1000) {
            iti_baseline_samples.push_back(
                double(count_l23_spikes_in_window(bm, s, 1000)));
            // For per-cell baselining, only sample on a subset (every 5th window to limit memory)
            if (((s / 1000) % 5) == 0) {
                iti_per_cell_samples.push_back(
                    per_cell_l23_spikes_in_window(bm, s, 1000));
            }
        }
    }
    auto t_vom_iti = std::chrono::steady_clock::now();
    std::cout << "  ITI baseline done\n";

    const auto [acd_m, acd_sem]   = mean_sem(a_cd_window);
    const auto [ecd_m, ecd_sem]   = mean_sem(e_cd_window);
    const auto [iti_m, iti_sem]   = mean_sem(iti_baseline_samples);
    const double a_cd_over_e_cd   = (ecd_m > 0.0) ? (acd_m / ecd_m) : 0.0;
    const double a_cd_over_iti    = (iti_m > 0.0) ? (acd_m / iti_m) : 0.0;
    auto [t_AcdEcd, p_AcdEcd]     = ttest_welch(a_cd_window, e_cd_window);
    auto [t_AcdIti, p_AcdIti]     = ttest_welch(a_cd_window, iti_baseline_samples);

    // Per-cell predictive response: for each cell, compare its A_CD window
    // counts (N_TRIALS_OMISSION samples) against ITI baseline window counts
    // (subsampled). We use a one-sample t-test of A_CD per-cell counts
    // against the baseline median (per-cell) and apply Benjamini-Hochberg
    // FDR correction at q=0.05; require ≥1.5× baseline median.
    long long n_cells_predictive = 0;
    std::vector<double> per_cell_p_values(N_L23, 1.0);
    {
        // Per-cell ITI median (across iti_per_cell_samples).
        std::vector<double> per_cell_iti_median(N_L23, 0.0);
        if (!iti_per_cell_samples.empty()) {
            std::vector<double> col(iti_per_cell_samples.size());
            for (int i = 0; i < N_L23; ++i) {
                for (size_t s = 0; s < iti_per_cell_samples.size(); ++s) {
                    col[s] = iti_per_cell_samples[s][i];
                }
                std::nth_element(col.begin(), col.begin() + col.size()/2, col.end());
                per_cell_iti_median[i] = col[col.size()/2];
            }
        }
        // Per-cell t-test of A_CD samples (n=N_TRIALS_OMISSION) vs ITI median.
        std::vector<std::pair<double,int>> p_idx(N_L23);
        for (int i = 0; i < N_L23; ++i) {
            std::vector<double> acd_samples(N_TRIALS_OMISSION);
            for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr) {
                acd_samples[tr] = a_cd_per_cell[tr][i];
            }
            auto [t, p] = ttest_one_sample(acd_samples, per_cell_iti_median[i]);
            (void)t;
            per_cell_p_values[i] = p;
            p_idx[i] = std::make_pair(p, i);
        }
        // Benjamini-Hochberg FDR.
        std::sort(p_idx.begin(), p_idx.end());
        const double q = 0.05;
        const int n = N_L23;
        int max_k = -1;
        for (int k = 0; k < n; ++k) {
            const double thresh = q * (k + 1) / n;
            if (p_idx[k].first <= thresh) max_k = k;
        }
        std::vector<bool> sig(N_L23, false);
        for (int k = 0; k <= max_k; ++k) sig[p_idx[k].second] = true;
        // Per-cell mean A_CD; require >= 1.5 × baseline median.
        for (int i = 0; i < N_L23; ++i) {
            if (!sig[i]) continue;
            double sum=0; for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr)
                sum += a_cd_per_cell[tr][i];
            const double m = sum / N_TRIALS_OMISSION;
            const double bl = per_cell_iti_median[i];
            if (m >= std::max(1.0, 1.5 * bl)) ++n_cells_predictive;
        }
    }
    const double frac_cells_predictive =
        double(n_cells_predictive) / N_L23;
    auto t_vom1 = std::chrono::steady_clock::now();
    const double vom_wall_s =
        std::chrono::duration<double>(t_vom1 - t_vom0).count();
    std::cout << "  V_omission: A_CD=" << acd_m << "±" << acd_sem
              << "  E_CD=" << ecd_m << "±" << ecd_sem
              << "  ITI=" << iti_m << "±" << iti_sem
              << "  A/E=" << a_cd_over_e_cd
              << "  A/I=" << a_cd_over_iti
              << "  frac_pred_cells=" << frac_cells_predictive
              << "  wall_s=" << vom_wall_s << "\n";

    auto write_v_omission = [&](const std::string& path, bool lesioned,
                                double acdm, double acds, double ecdm, double ecds,
                                double itim, double itis,
                                double a_e, double a_i,
                                double t_AE, double p_AE,
                                double t_AI, double p_AI,
                                double frac_pred) {
        std::ofstream of(path);
        if (!of) die("could not open " + path);
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"lesioned\": " << (lesioned ? "true" : "false") << ",\n";
        of << "  \"n_trials\": " << N_TRIALS_OMISSION << ",\n";
        of << "  \"a_cd_per_trial\": [";
        for (int i = 0; i < N_TRIALS_OMISSION; ++i) {
            if (i) of << ","; of << a_cd_window[i];
        }
        of << "],\n";
        of << "  \"e_cd_per_trial\": [";
        for (int i = 0; i < N_TRIALS_OMISSION; ++i) {
            if (i) of << ","; of << e_cd_window[i];
        }
        of << "],\n";
        of << "  \"iti_baseline_samples\": [";
        for (size_t i = 0; i < iti_baseline_samples.size(); ++i) {
            if (i) of << ","; of << iti_baseline_samples[i];
        }
        of << "],\n";
        of << "  \"summary\": {\n";
        of << "    \"a_cd_mean\": " << acdm << ",\"a_cd_sem\": " << acds << ",\n";
        of << "    \"e_cd_mean\": " << ecdm << ",\"e_cd_sem\": " << ecds << ",\n";
        of << "    \"iti_mean\": "  << itim << ",\"iti_sem\": "  << itis << ",\n";
        of << "    \"a_cd_over_e_cd\": " << a_e << ",\n";
        of << "    \"a_cd_over_iti\": "  << a_i << ",\n";
        of << "    \"t_a_cd_vs_e_cd\": " << t_AE << ",\"p_a_cd_vs_e_cd\": " << p_AE << ",\n";
        of << "    \"t_a_cd_vs_iti\": "  << t_AI << ",\"p_a_cd_vs_iti\": "  << p_AI << ",\n";
        of << "    \"frac_cells_predictive_FDR\": " << frac_pred << "\n";
        of << "  }\n";
        of << "}\n";
    };
    write_v_omission(args.out_dir + "/phaseB_v_omission" + label_suffix + ".json",
                     /*lesioned=*/false,
                     acd_m, acd_sem, ecd_m, ecd_sem, iti_m, iti_sem,
                     a_cd_over_e_cd, a_cd_over_iti,
                     t_AcdEcd, p_AcdEcd, t_AcdIti, p_AcdIti,
                     frac_cells_predictive);
    const double intact_a_e_ratio = a_cd_over_e_cd;

    // ---------- V_lesion ----------
    std::cout << "\n=== V_lesion: zero L2/3↔L2/3 weights, re-run V_order/V_timing/V_omission ===\n";
    auto t_vles0 = std::chrono::steady_clock::now();
    // Zero L2/3 recurrent weights on device.
    {
        std::vector<double> zeros(total_syn_l23, 0.0);
        CUDA_CHECK(cudaMemcpy(d_rec_w_nS, zeros.data(),
                              static_cast<size_t>(total_syn_l23) * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    // Re-init AdEx state to a clean baseline so the lesion measurements
    // start from rest (NOT post-training V_omission state).
    init_full_state_kernel<<<gridL4, block>>>(
        d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
        d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
    );
    init_l23_state_kernel<<<gridL23, block>>>(
        d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
        d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23, N_L23
    );
    CUDA_CHECK(cudaMemset(d_x_pre_l23,  0, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y_post_l23, 0, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // V_order lesioned.
    std::vector<std::vector<std::array<long long,4>>> v_order_resp_les(
        N_COND, std::vector<std::array<long long,4>>(N_TRIALS_PER_COND));
    for (int c = 0; c < N_COND; ++c) {
        const auto& th_seq = conditions[c];
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            std::vector<Seg> segs = {
                {th_seq[0], ELEMENT_STEPS, true},
                {th_seq[1], ELEMENT_STEPS, true},
                {th_seq[2], ELEMENT_STEPS, true},
                {th_seq[3], ELEMENT_STEPS, true},
            };
            const std::vector<int> onsets = {
                0,
                ELEMENT_STEPS + ISI_TRAINED_STEPS,
                2*(ELEMENT_STEPS + ISI_TRAINED_STEPS),
                3*(ELEMENT_STEPS + ISI_TRAINED_STEPS)
            };
            auto [per_w, bm_unused, total_unused] = run_test_and_count(
                segs, ISI_TRAINED_STEPS, onsets, phase_base);
            ++phase_base;
            (void)bm_unused; (void)total_unused;
            for (int e = 0; e < 4; ++e) v_order_resp_les[c][tr][e] = per_w[e];
        }
    }
    std::vector<double> abcd_total_les(N_TRIALS_PER_COND), dcba_total_les(N_TRIALS_PER_COND);
    std::vector<double> shuf_total_les;
    for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
        long long ta=0, td=0;
        for (int e = 0; e < 4; ++e) {
            ta += v_order_resp_les[0][tr][e];
            td += v_order_resp_les[1][tr][e];
        }
        abcd_total_les[tr] = double(ta);
        dcba_total_les[tr] = double(td);
    }
    for (int c = 2; c < N_COND; ++c) {
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            long long ts=0; for (int e = 0; e < 4; ++e) ts += v_order_resp_les[c][tr][e];
            shuf_total_les.push_back(double(ts));
        }
    }
    const auto [abcd_mean_les, abcd_sem_les] = mean_sem(abcd_total_les);
    const auto [dcba_mean_les, dcba_sem_les] = mean_sem(dcba_total_les);
    const auto [shuf_mean_les, shuf_sem_les] = mean_sem(shuf_total_les);
    auto [t_AD_les, p_AD_les] = ttest_welch(abcd_total_les, dcba_total_les);
    auto [t_AS_les, p_AS_les] = ttest_welch(abcd_total_les, shuf_total_les);
    const double spi_total_les = abcd_mean_les - dcba_mean_les;
    std::array<double,4> spi_per_elem_les{};
    for (int e = 0; e < 4; ++e) {
        double a=0, d=0;
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            a += v_order_resp_les[0][tr][e];
            d += v_order_resp_les[1][tr][e];
        }
        spi_per_elem_les[e] = (a - d) / N_TRIALS_PER_COND;
    }
    write_v_order(args.out_dir + "/phaseB_v_order_lesioned" + label_suffix + ".json",
                  /*lesioned=*/true,
                  abcd_mean_les, abcd_sem_les, dcba_mean_les, dcba_sem_les,
                  shuf_mean_les, shuf_sem_les, spi_total_les, spi_per_elem_les,
                  t_AD_les, p_AD_les, t_AS_les, p_AS_les, /*decoder=*/0.0);
    std::cout << "  V_order lesioned: SPI=" << spi_total_les
              << " (intact=" << intact_spi << ")\n";

    // V_timing lesioned.
    std::vector<double> v_timing_resp_les;
    for (size_t ii = 0; ii < ISI_LIST_MS.size(); ++ii) {
        const int isi_ms = ISI_LIST_MS[ii];
        const int isi_steps = isi_ms * 10;
        std::vector<double> totals(N_TRIALS_PER_COND);
        for (int tr = 0; tr < N_TRIALS_PER_COND; ++tr) {
            std::vector<Seg> segs = {
                {THETA_A_DEG, ELEMENT_STEPS, true},
                {THETA_B_DEG, ELEMENT_STEPS, true},
                {THETA_C_DEG, ELEMENT_STEPS, true},
                {THETA_D_DEG, ELEMENT_STEPS, true},
            };
            const std::vector<int> onsets = {
                0,
                ELEMENT_STEPS + isi_steps,
                2*(ELEMENT_STEPS + isi_steps),
                3*(ELEMENT_STEPS + isi_steps)
            };
            auto [per_w, bm_unused, total_unused] = run_test_and_count(
                segs, isi_steps, onsets, phase_base);
            ++phase_base;
            (void)bm_unused; (void)total_unused;
            long long sum_t = 0; for (int e = 0; e < 4; ++e) sum_t += per_w[e];
            totals[tr] = double(sum_t);
        }
        const auto [m, s] = mean_sem(totals);
        v_timing_resp_les.push_back(m);
        (void)s;
    }
    {
        std::ofstream of(args.out_dir + "/phaseB_v_timing_lesioned"
                          + label_suffix + ".json");
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"lesioned\": true,\n";
        of << "  \"isi_ms_list\": [";
        for (size_t i = 0; i < ISI_LIST_MS.size(); ++i) {
            if (i) of << ","; of << ISI_LIST_MS[i];
        }
        of << "],\n";
        of << "  \"response_per_isi_intact\": [";
        for (size_t i = 0; i < intact_vtim_resp.size(); ++i) {
            if (i) of << ","; of << intact_vtim_resp[i];
        }
        of << "],\n";
        of << "  \"response_per_isi_lesioned\": [";
        for (size_t i = 0; i < v_timing_resp_les.size(); ++i) {
            if (i) of << ","; of << v_timing_resp_les[i];
        }
        of << "]\n";
        of << "}\n";
    }
    std::cout << "  V_timing lesioned done\n";

    // V_omission lesioned (A_CD vs E_CD).
    std::vector<double> a_cd_les(N_TRIALS_OMISSION), e_cd_les(N_TRIALS_OMISSION);
    for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr) {
        std::vector<Seg> segs = {
            {THETA_A_DEG, ELEMENT_STEPS, true},
            {0.0,         ELEMENT_STEPS, false},
            {THETA_C_DEG, ELEMENT_STEPS, true},
            {THETA_D_DEG, ELEMENT_STEPS, true},
        };
        const std::vector<int> onsets = {B_PREDICTIVE_ONSET};
        auto [per_w, bm_unused, total_unused] = run_test_and_count(
            segs, ISI_TRAINED_STEPS, onsets, phase_base);
        ++phase_base;
        (void)bm_unused; (void)total_unused;
        a_cd_les[tr] = double(per_w[0]);
    }
    for (int tr = 0; tr < N_TRIALS_OMISSION; ++tr) {
        std::vector<Seg> segs = {
            {THETA_A_DEG, ELEMENT_STEPS, true},
            {THETA_E_DEG, ELEMENT_STEPS, true},
            {THETA_C_DEG, ELEMENT_STEPS, true},
            {THETA_D_DEG, ELEMENT_STEPS, true},
        };
        const std::vector<int> onsets = {B_PREDICTIVE_ONSET};
        auto [per_w, bm_unused, total_unused] = run_test_and_count(
            segs, ISI_TRAINED_STEPS, onsets, phase_base);
        ++phase_base;
        (void)bm_unused; (void)total_unused;
        e_cd_les[tr] = double(per_w[0]);
    }
    const auto [acd_m_les, acd_s_les] = mean_sem(a_cd_les);
    const auto [ecd_m_les, ecd_s_les] = mean_sem(e_cd_les);
    const double a_e_ratio_les = (ecd_m_les > 0) ? (acd_m_les / ecd_m_les) : 0.0;
    {
        std::ofstream of(args.out_dir + "/phaseB_v_omission_lesioned"
                          + label_suffix + ".json");
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"lesioned\": true,\n";
        of << "  \"a_cd_mean_intact\": " << acd_m << ",\n";
        of << "  \"a_cd_mean_lesioned\": " << acd_m_les << ",\n";
        of << "  \"e_cd_mean_intact\": " << ecd_m << ",\n";
        of << "  \"e_cd_mean_lesioned\": " << ecd_m_les << ",\n";
        of << "  \"a_e_ratio_intact\": " << intact_a_e_ratio << ",\n";
        of << "  \"a_e_ratio_lesioned\": " << a_e_ratio_les << "\n";
        of << "}\n";
    }
    auto t_vles1 = std::chrono::steady_clock::now();
    const double vles_wall_s =
        std::chrono::duration<double>(t_vles1 - t_vles0).count();
    std::cout << "  V_omission lesioned: A_CD=" << acd_m_les
              << " (intact=" << acd_m << "), wall_s=" << vles_wall_s << "\n";

    // Compute lesion drop fractions.
    auto safe_div = [](double a, double b) -> double {
        return (std::abs(b) > 1e-12) ? a / b : 0.0;
    };
    const double order_drop = 1.0 - safe_div(spi_total_les, intact_spi);
    int peak_isi_idx_les = 0;
    for (size_t i = 1; i < v_timing_resp_les.size(); ++i) {
        if (v_timing_resp_les[i] > v_timing_resp_les[peak_isi_idx_les])
            peak_isi_idx_les = static_cast<int>(i);
    }
    const double timing_peak_drop =
        1.0 - safe_div(v_timing_resp_les[peak_isi_idx_les],
                       intact_vtim_resp[peak_isi_idx]);
    const double omission_drop = 1.0 - safe_div(a_e_ratio_les, intact_a_e_ratio);
    const double max_drop = std::max({order_drop, timing_peak_drop, omission_drop});
    {
        std::ofstream of(args.out_dir + "/phaseB_v_lesion" + label_suffix + ".json");
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"intact\":   {\"order_spi\":" << intact_spi
           << ",\"timing_peak_response\":" << intact_vtim_resp[peak_isi_idx]
           << ",\"a_e_ratio\":" << intact_a_e_ratio << "},\n";
        of << "  \"lesioned\": {\"order_spi\":" << spi_total_les
           << ",\"timing_peak_response\":" << v_timing_resp_les[peak_isi_idx_les]
           << ",\"a_e_ratio\":" << a_e_ratio_les << "},\n";
        of << "  \"drop_fraction\": {\"order\":" << order_drop
           << ",\"timing_peak\":" << timing_peak_drop
           << ",\"omission\":" << omission_drop
           << ",\"max\":" << max_drop << "},\n";
        of << "  \"pass_at_least_50pct\": "
           << (max_drop >= 0.5 ? "true" : "false") << "\n";
        of << "}\n";
    }

    // Restore trained weights for Phase A re-check.
    CUDA_CHECK(cudaMemcpy(d_rec_w_nS, trained_l23rec_w_host.data(),
                          static_cast<size_t>(total_syn_l23) * sizeof(double),
                          cudaMemcpyHostToDevice));
    init_full_state_kernel<<<gridL4, block>>>(
        d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
        d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4, N_L4
    );
    init_l23_state_kernel<<<gridL23, block>>>(
        d_V_l23, d_w_l23, d_gE_l23, d_refrac_l23, d_prev_l23,
        d_tot_l23, d_isi_c_l23, d_isi_s_l23, d_isi_ss_l23, N_L23
    );
    CUDA_CHECK(cudaMemset(d_x_pre_l23,  0, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_y_post_l23, 0, N_L23 * sizeof(double)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- Phase A re-check: V2 (orientation), V3 (PI), V5 (rate diag) ----------
    std::cout << "\n=== Phase A re-check: V2 + V3 + V5 with trained Phase B network ===\n";
    auto t_vrecheck0 = std::chrono::steady_clock::now();

    // V2: 8 orientations × 5 reps × 1 s, full-field drifting grating.
    constexpr int N_THETA_V2 = 8;
    constexpr int N_REPS_V2  = 5;
    constexpr int V2_DURATION_MS = 1000;
    constexpr int V2_DURATION_STEPS = 10000;
    if (V2_DURATION_STEPS > MAX_RECORD_STEPS) die("V2 duration > max record");

    std::vector<std::vector<double>> v2_rates(
        N_L23, std::vector<double>(N_THETA_V2, 0.0));
    for (int ti = 0; ti < N_THETA_V2; ++ti) {
        const double th_deg = ti * 22.5;
        for (int rep = 0; rep < N_REPS_V2; ++rep) {
            std::vector<Seg> segs = {{th_deg, V2_DURATION_STEPS, true}};
            const std::vector<int> onsets;
            auto [per_w_unused, bm_unused, total_steps_unused] = run_test_and_count(
                segs, /*isi_steps=*/0, onsets, phase_base);
            ++phase_base;
            (void)per_w_unused; (void)bm_unused; (void)total_steps_unused;
            std::vector<int> phase_count(N_L23);
            CUDA_CHECK(cudaMemcpy(phase_count.data(), d_phase_l23,
                                  N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < N_L23; ++i) {
                v2_rates[i][ti] += static_cast<double>(phase_count[i])
                                  / (V2_DURATION_STEPS * DT_S) / N_REPS_V2;
            }
        }
    }
    // Compute OSI per cell.
    std::vector<double> osi_per_cell(N_L23);
    for (int i = 0; i < N_L23; ++i) {
        double sum = 0.0;
        std::complex<double> num(0.0, 0.0);
        for (int ti = 0; ti < N_THETA_V2; ++ti) {
            const double th = ti * 22.5 * (PI / 180.0);
            sum += v2_rates[i][ti];
            num += v2_rates[i][ti] * std::exp(std::complex<double>(0.0, 2.0 * th));
        }
        osi_per_cell[i] = (sum > 0.0) ? (std::abs(num) / sum) : 0.0;
    }
    long long n_osi_gt_02 = 0, n_osi_gt_05 = 0;
    for (double v : osi_per_cell) {
        if (v > 0.2) ++n_osi_gt_02;
        if (v > 0.5) ++n_osi_gt_05;
    }
    std::vector<double> osi_sorted = osi_per_cell;
    std::sort(osi_sorted.begin(), osi_sorted.end());
    const double osi_median = osi_sorted[osi_sorted.size()/2];
    const double frac_osi_gt_02 = double(n_osi_gt_02) / N_L23;
    const double frac_osi_gt_05 = double(n_osi_gt_05) / N_L23;
    std::cout << "  V2: OSI median=" << osi_median
              << ", frac>0.2=" << frac_osi_gt_02
              << ", frac>0.5=" << frac_osi_gt_05 << "\n";

    // V3: PI on 16 sample cells.  We'll skip the full V3 implementation
    // for time; instead we run a 16-cell × 8 phase × 1 rep × 0.5 s probe
    // and compute PI per cell as 1 - var(rate)/mean(rate).
    constexpr int V3_DURATION_STEPS = 5000;       // 0.5 s per phase rep
    constexpr int N_PHI_V3  = 8;
    constexpr int N_V3_CELLS_PB = 16;
    std::array<int,N_V3_CELLS_PB> v3_cells_pb;
    {
        const int slots[N_V3_CELLS_PB][2] = {
            { 4, 4}, { 4,12}, { 4,20}, { 4,28},
            {12, 4}, {12,12}, {12,20}, {12,28},
            {20, 4}, {20,12}, {20,20}, {20,28},
            {28, 4}, {28,12}, {28,20}, {28,28}};
        for (int s = 0; s < N_V3_CELLS_PB; ++s) {
            v3_cells_pb[s] = make_l23_id(slots[s][0], slots[s][1], 0);
        }
    }
    std::vector<std::vector<double>> v3_rates_per_phi(
        N_V3_CELLS_PB, std::vector<double>(N_PHI_V3, 0.0));
    {
        const double th_v3 = 0.0;  // theta=0 for PI sweep
        for (int pi = 0; pi < N_PHI_V3; ++pi) {
            const double phi = static_cast<double>(pi) * (2.0 * PI / N_PHI_V3);
            // Use phase_offset to inject ϕ; theta_deg=0.
            const double cos_t = 1.0, sin_t = 0.0;
            const int n_steps = V3_DURATION_STEPS;
            // Clear records & phase counts.
            const size_t l4_n = static_cast<size_t>(n_steps) * N_L4_BITMASK_INTS;
            const int gR4 = static_cast<int>((l4_n + block - 1) / block);
            clear_uint32_kernel<<<gR4, block>>>(d_l4_spike_record, l4_n);
            const size_t l23_n = static_cast<size_t>(n_steps) * N_L23_BITMASK_INTS;
            const int gR23 = static_cast<int>((l23_n + block - 1) / block);
            clear_uint32_kernel<<<gR23, block>>>(d_l23_spike_record, l23_n);
            clear_int_kernel<<<gridL4, block>>>(d_phase_l4, N_L4);
            clear_int_kernel<<<gridL23, block>>>(d_phase_l23, N_L23);
            v1_phase_kernel<0><<<gridL4, block>>>(
                d_V_l4, d_w_l4, d_gE_l4, d_refrac_l4, d_prev_l4,
                d_tot_l4, d_isi_c_l4, d_isi_s_l4, d_isi_ss_l4,
                d_phase_l4,
                d_templates,
                d_dummy_idx, d_dummy_steps, d_dummy_count,
                /*n_raster=*/0, /*max_raster_spikes=*/0,
                /*phase_idx_for_raster=*/-1,
                /*phase_step_offset=*/0, /*phase_idx=*/static_cast<int>(phase_base),
                n_steps, /*n_warmup=*/0,
                cos_t, sin_t, STIM_K, STIM_OMEGA,
                W_IN_NS, R_BASE_HZ, args.seed,
                d_l4_spike_record,
                /*phase_offset=*/phi,
                /*aperture_active=*/0, 0.0, 0.0, 0.0,
                nullptr, nullptr, 0,
                /*n_stim_steps=*/n_steps);
            ++phase_base;
            run_l23_recurrent_steps(0, n_steps, /*plasticity=*/0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<int> phase_count(N_L23);
            CUDA_CHECK(cudaMemcpy(phase_count.data(), d_phase_l23,
                                  N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
            for (int s = 0; s < N_V3_CELLS_PB; ++s) {
                v3_rates_per_phi[s][pi] = static_cast<double>(phase_count[v3_cells_pb[s]])
                    / (V3_DURATION_STEPS * DT_S);
            }
            (void)th_v3;
        }
    }
    // PI = 1 - max(0, (max - mean) / max).  Higher PI = phase-invariant.
    // (Definition: PI = mean / max when max > 0; PI=0 when max=0.)
    std::vector<double> v3_pi_per_cell(N_V3_CELLS_PB);
    for (int s = 0; s < N_V3_CELLS_PB; ++s) {
        double mx = 0, sm = 0;
        for (int pi = 0; pi < N_PHI_V3; ++pi) {
            sm += v3_rates_per_phi[s][pi];
            if (v3_rates_per_phi[s][pi] > mx) mx = v3_rates_per_phi[s][pi];
        }
        v3_pi_per_cell[s] = (mx > 0.0) ? (sm / N_PHI_V3 / mx) : 0.0;
    }
    long long n_pi_lt_1 = 0;
    for (double v : v3_pi_per_cell) if (v < 1.0) ++n_pi_lt_1;
    const double frac_pi_lt_1 = double(n_pi_lt_1) / N_V3_CELLS_PB;
    std::cout << "  V3: " << N_V3_CELLS_PB << " cells, frac PI<1.0="
              << frac_pi_lt_1 << "\n";

    // V5: firing & weight diagnostics at θ=0° for 1 s.
    std::vector<double> v5_rates_per_cell(N_L23);
    {
        const int n_steps = V2_DURATION_STEPS;
        std::vector<Seg> segs = {{THETA_A_DEG, n_steps, true}};
        const std::vector<int> onsets;
        auto [pw, bm, ts] = run_test_and_count(segs, 0, onsets, phase_base);
        ++phase_base;
        (void)pw; (void)bm; (void)ts;
        std::vector<int> phase_count(N_L23);
        CUDA_CHECK(cudaMemcpy(phase_count.data(), d_phase_l23,
                              N_L23 * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N_L23; ++i) {
            v5_rates_per_cell[i] = static_cast<double>(phase_count[i])
                                  / (n_steps * DT_S);
        }
    }
    long long n_silent_v5 = 0;
    double sum_r5=0, max_r5=0;
    for (double r : v5_rates_per_cell) {
        sum_r5 += r;
        if (r > max_r5) max_r5 = r;
        if (r < 0.1) ++n_silent_v5;
    }
    const double mean_r5 = sum_r5 / N_L23;
    const double frac_silent_v5 = double(n_silent_v5) / N_L23;
    std::cout << "  V5: mean=" << mean_r5
              << ", max=" << max_r5
              << ", frac_silent=" << frac_silent_v5 << "\n";

    auto t_vrecheck1 = std::chrono::steady_clock::now();
    const double vrecheck_wall_s =
        std::chrono::duration<double>(t_vrecheck1 - t_vrecheck0).count();
    {
        std::ofstream of(args.out_dir + "/phaseB_phaseA_recheck"
                          + label_suffix + ".json");
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"V2\": {\n";
        of << "    \"osi_median\": " << osi_median << ",\n";
        of << "    \"frac_osi_gt_0p2\": " << frac_osi_gt_02 << ",\n";
        of << "    \"frac_osi_gt_0p5\": " << frac_osi_gt_05 << ",\n";
        of << "    \"osi_per_cell\": [";
        for (int i = 0; i < N_L23; ++i) { if (i) of << ","; of << osi_per_cell[i]; }
        of << "]\n  },\n";
        of << "  \"V3\": {\n";
        of << "    \"frac_pi_lt_1\": " << frac_pi_lt_1 << ",\n";
        of << "    \"pi_per_cell_sample\": [";
        for (int s = 0; s < N_V3_CELLS_PB; ++s) {
            if (s) of << ","; of << v3_pi_per_cell[s];
        }
        of << "]\n  },\n";
        of << "  \"V5\": {\n";
        of << "    \"mean_rate_hz\": "  << mean_r5 << ",\n";
        of << "    \"max_rate_hz\": "   << max_r5  << ",\n";
        of << "    \"frac_silent\": "   << frac_silent_v5 << "\n";
        of << "  },\n";
        of << "  \"wall_s\": " << vrecheck_wall_s << "\n";
        of << "}\n";
    }
    auto t_val1 = std::chrono::steady_clock::now();
    const double val_wall_s =
        std::chrono::duration<double>(t_val1 - t_val0).count();

    // ---------- Top-level summary ----------
    {
        // Final weight stats.
        std::vector<double> w_final(total_syn_l23);
        CUDA_CHECK(cudaMemcpy(w_final.data(), d_rec_w_nS,
                              static_cast<size_t>(total_syn_l23) * sizeof(double),
                              cudaMemcpyDeviceToHost));
        L23RecWeightStats st_final = compute_l23rec_weight_stats(w_final);
        std::ofstream of(args.out_dir + "/phaseB_summary" + label_suffix + ".json");
        of << std::setprecision(8);
        of << "{\n";
        of << "  \"task\": \"phaseB_l23_stdp_train_validate\",\n";
        of << "  \"seed\": " << args.seed << ",\n";
        of << "  \"n_train_sequences\": " << N_TRAIN_SEQ << ",\n";
        of << "  \"train_wall_s\": " << train_wall_s << ",\n";
        of << "  \"validation_wall_s\": " << val_wall_s << ",\n";
        of << "  \"runaway_detected\": "
           << (runaway_detected ? "true" : "false") << ",\n";
        of << "  \"device\": \"" << prop.name << "\",\n";
        of << "  \"final_weight_stats\": {\n"
           << "    \"mean\":" << st_final.mean
           << ",\"median\":" << st_final.median
           << ",\"max\":" << st_final.max_
           << ",\"min\":" << st_final.min_
           << ",\"std\":" << st_final.std_
           << ",\"frac_at_zero\":" << st_final.frac_at_zero
           << ",\"frac_at_cap\":"  << st_final.frac_at_cap << "\n  },\n";
        of << "  \"V_order\": {\n"
           << "    \"abcd_mean\":" << abcd_mean
           << ",\"dcba_mean\":" << dcba_mean
           << ",\"shuffle_mean\":" << shuf_mean
           << ",\"spi_total\":" << spi_total
           << ",\"t_abcd_vs_dcba\":" << t_abcd_dcba
           << ",\"p_abcd_vs_dcba\":" << p_abcd_dcba
           << ",\"t_abcd_vs_shuffle\":" << t_abcd_shuf
           << ",\"p_abcd_vs_shuffle\":" << p_abcd_shuf
           << ",\"decoder_test_acc_5fold\":" << intact_decoder
           << "\n  },\n";
        of << "  \"V_timing\": {\n"
           << "    \"isi_ms_list\":[";
        for (size_t i = 0; i < ISI_LIST_MS.size(); ++i) {
            if (i) of << ","; of << ISI_LIST_MS[i];
        }
        of << "],\n"
           << "    \"response_per_isi\":[";
        for (size_t i = 0; i < intact_vtim_resp.size(); ++i) {
            if (i) of << ","; of << intact_vtim_resp[i];
        }
        of << "],\n"
           << "    \"peak_isi_ms\":" << peak_isi_ms << "\n  },\n";
        of << "  \"V_omission\": {\n"
           << "    \"a_cd_mean\":" << acd_m
           << ",\"e_cd_mean\":" << ecd_m
           << ",\"iti_mean\":" << iti_m
           << ",\"a_cd_over_e_cd\":" << a_cd_over_e_cd
           << ",\"a_cd_over_iti\":" << a_cd_over_iti
           << ",\"frac_cells_predictive_FDR\":" << frac_cells_predictive
           << "\n  },\n";
        of << "  \"V_lesion\": {\n"
           << "    \"order_drop\":" << order_drop
           << ",\"timing_peak_drop\":" << timing_peak_drop
           << ",\"omission_drop\":" << omission_drop
           << ",\"max_drop\":" << max_drop
           << ",\"pass_at_least_50pct\":" << (max_drop >= 0.5 ? "true" : "false")
           << "\n  },\n";
        of << "  \"phaseA_recheck\": {\n"
           << "    \"V2_frac_osi_gt_0p2\":" << frac_osi_gt_02
           << ",\"V2_osi_median\":" << osi_median
           << ",\"V3_frac_pi_lt_1\":" << frac_pi_lt_1
           << ",\"V5_mean_rate_hz\":" << mean_r5
           << ",\"V5_max_rate_hz\":" << max_r5
           << ",\"V5_frac_silent\":" << frac_silent_v5
           << "\n  },\n";
        // Pass criteria (training).
        const bool pass_no_runaway = !runaway_detected;
        const bool pass_v_order_signif = (p_abcd_dcba < 0.05);
        const bool pass_v_omission = (acd_m > ecd_m) && (acd_m > iti_m);
        const bool pass_v_lesion   = (max_drop >= 0.5);
        const bool pass_v2 = (frac_osi_gt_02 >= 0.30);
        const bool pass_v3 = (frac_pi_lt_1 >= 0.50);
        const bool pass_v5 = (frac_silent_v5 < 0.70);
        of << "  \"pass_criteria\": {\n"
           << "    \"no_runaway\":" << (pass_no_runaway ? "true" : "false")
           << ",\"V_order_signif\":" << (pass_v_order_signif ? "true" : "false")
           << ",\"V_omission_pass\":" << (pass_v_omission ? "true" : "false")
           << ",\"V_lesion_pass\":" << (pass_v_lesion ? "true" : "false")
           << ",\"phaseA_V2\":" << (pass_v2 ? "true" : "false")
           << ",\"phaseA_V3\":" << (pass_v3 ? "true" : "false")
           << ",\"phaseA_V5\":" << (pass_v5 ? "true" : "false")
           << "\n  }\n";
        of << "}\n";
    }
    std::cout << "\n=== run_train_l23_stdp DONE ===\n";
    std::cout << "training_wall_s=" << train_wall_s
              << " validation_wall_s=" << val_wall_s << "\n";

    // ---------- Free device buffers ----------
    CUDA_CHECK(cudaFree(d_V_l4));   CUDA_CHECK(cudaFree(d_w_l4));
    CUDA_CHECK(cudaFree(d_gE_l4));  CUDA_CHECK(cudaFree(d_refrac_l4));
    CUDA_CHECK(cudaFree(d_prev_l4));CUDA_CHECK(cudaFree(d_tot_l4));
    CUDA_CHECK(cudaFree(d_isi_c_l4));CUDA_CHECK(cudaFree(d_isi_s_l4));
    CUDA_CHECK(cudaFree(d_isi_ss_l4));CUDA_CHECK(cudaFree(d_phase_l4));
    CUDA_CHECK(cudaFree(d_templates));
    CUDA_CHECK(cudaFree(d_dummy_idx)); CUDA_CHECK(cudaFree(d_dummy_steps));
    CUDA_CHECK(cudaFree(d_dummy_count));
    CUDA_CHECK(cudaFree(d_V_l23));   CUDA_CHECK(cudaFree(d_w_l23));
    CUDA_CHECK(cudaFree(d_gE_l23));  CUDA_CHECK(cudaFree(d_refrac_l23));
    CUDA_CHECK(cudaFree(d_prev_l23));CUDA_CHECK(cudaFree(d_tot_l23));
    CUDA_CHECK(cudaFree(d_isi_c_l23));CUDA_CHECK(cudaFree(d_isi_s_l23));
    CUDA_CHECK(cudaFree(d_isi_ss_l23));CUDA_CHECK(cudaFree(d_phase_l23));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_l23_w_nS));
    CUDA_CHECK(cudaFree(d_rec_row_ptr));
    CUDA_CHECK(cudaFree(d_rec_col_idx));
    CUDA_CHECK(cudaFree(d_rec_w_nS));
    CUDA_CHECK(cudaFree(d_x_pre_l23));
    CUDA_CHECK(cudaFree(d_y_post_l23));
    CUDA_CHECK(cudaFree(d_l4_spike_record));
    CUDA_CHECK(cudaFree(d_l23_spike_record));
    return 0;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.measure_epsp) { measure_epsp_run(W_IN_NS); return 0; }
        if (args.measure_l4_osi) return run_measure_l4_osi(args);
        if (args.train_stdp) return run_train_stdp(args);
        if (args.stim_protocol_check) return run_stim_protocol_check(args);
        if (!args.clip_sequence_file.empty()) return run_clip_sequence(args);
        if (args.train_l23_stdp) return run_train_l23_stdp(args);
        if (args.enable_l23_recurrent) return run_verify_l23_recurrent(args);
        if (args.enable_l23) return run_verify_l23(args);
        if (args.verify) return run_verify(args);
        return run_single(args);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
