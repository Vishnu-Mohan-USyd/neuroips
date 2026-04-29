// l4_test.cu
//
// V1 L4 layer: 131,072 AdEx neurons arranged as
//   32x32 retinotopic hypercolumns x 8 orientation columns x 16 cells/orient
//
// Each cell has a von-Mises orientation tuning curve at preferred
// orientation theta_pref = k_orient * pi/8.  Per-cell Poisson input rate is
//
//     rate(t) = R_base + R_max * tuning(theta_stim - theta_pref)
//                              * intensity(x_ret, y_ret, t)
//
// where intensity(x,y,t) = (1 + cos(K(x cosθ + y sinθ) - ω t))/2 is the same
// moving-grating signal as stim_test.cu's pixel layer.  Each L4 cell receives
// its own Poisson source at this modulated rate; spikes inject an exponential
// current pulse (tau_E = 5 ms, Q_E = 60 pA) into the AdEx soma.
//
// All state lives on the device.  One thread per L4 cell; a single kernel
// covers an entire test-orientation phase (warmup + measurement steps in
// one launch).  The host orchestrates: clear per-phase counters, launch
// kernel, read back accumulators, repeat.
//
// Outputs:
//   - per-cell tuning matrix (rate Hz at each test orientation)
//   - per-cell ISI mean / CV (across the full run)
//   - sample-cell spike rasters (20 cells: 5 retinotopic positions x 4
//     orientation columns)
//   - aggregate rate-stat and ISI-CV histograms
//
// CLI:  l4_test --seed N --n-test-orientations N --duration_per_ms M
//                --warmup_ms M --out PATH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------- topology ----------
constexpr int GRID = 32;
constexpr int N_HYPERCOL = GRID * GRID;          // 1024
constexpr int N_ORIENT = 8;
constexpr int N_CELLS_PER_ORIENT = 16;
constexpr int CELLS_PER_HYPERCOL = N_ORIENT * N_CELLS_PER_ORIENT;  // 128
constexpr int N_L4 = N_HYPERCOL * CELLS_PER_HYPERCOL;              // 131,072
constexpr double PI = 3.14159265358979323846;

// ---------- AdEx (Brette-Gerstner regular spiking) ----------
constexpr double C_M_PF      = 200.0;
constexpr double G_L_NS      = 10.0;
constexpr double E_L_MV      = -70.0;
constexpr double V_T_MV      = -50.0;
constexpr double DELTA_T_MV  = 2.0;
// Adaptation: keep subthreshold "a" coupling (small) and make spike-triggered
// "b" mild so cells can sustain ~20-60 Hz at preferred orientation; the
// canonical Brette-Gerstner regular-spiking b=80.5 pA over-silences here
// because the input is single-source Poisson rather than thousands of small
// inputs.
constexpr double A_NS        = 2.0;
constexpr double TAU_W_MS    = 144.0;
constexpr double B_PA        = 20.0;
constexpr double V_R_MV      = -70.6;
constexpr double V_PEAK_MV   = 0.0;
constexpr double T_REF_MS    = 2.0;

// ---------- synaptic input ----------
// Calibrated so that at preferred orientation + peak intensity (Poisson rate
// = R_base + R_max ≈ 1005 Hz) the steady-state mean I_syn ≈ λ·Q·τ ≈ 750 pA,
// giving an output firing rate around 30-50 Hz in the AdEx after the
// adaptation balance (w_ss ≈ b·r·τ_w + a·(V_T - E_L)).
constexpr double TAU_E_MS    = 5.0;
constexpr double Q_E_PA      = 150.0;

// ---------- stimulus ----------
constexpr double K_SPATIAL          = 2.0 * PI / 8.0;  // one cycle / 8 px
constexpr double OMEGA_TEMPORAL_DEFAULT = 2.0 * PI * 4.0;  // 4 Hz drift
constexpr double R_BASE_HZ   = 5.0;
constexpr double R_MAX_HZ    = 1000.0;   // peak per-cell input rate (preferred orient * I=1)
constexpr double KAPPA       = 4.0;

// ---------- timing ----------
constexpr double DT_MS = 0.1;
constexpr double DT_S  = DT_MS / 1000.0;

// ---------- sample cells (20 = 5 positions x 4 orientations, all at k_cell=0) ----------
constexpr int N_SAMPLE_CELLS = 20;
constexpr int MAX_SAMPLE_SPIKES = 4096;  // per sample cell, across full run

// ---------- CUDA helpers ----------
#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::ostringstream _os;                                           \
            _os << "CUDA error: " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__;                     \
            throw std::runtime_error(_os.str());                              \
        }                                                                     \
    } while (0)

[[noreturn]] void die(const std::string& msg) {
    throw std::runtime_error(msg);
}

__device__ inline double von_mises_tuning(double dtheta_rad, double kappa) {
    // exp(kappa * (cos(2*dtheta) - 1)) -- in [exp(-2*kappa), 1].
    return exp(kappa * (cos(2.0 * dtheta_rad) - 1.0));
}

__device__ inline int decompose_x(int idx) { return (idx / CELLS_PER_HYPERCOL) % GRID; }
__device__ inline int decompose_y(int idx) { return (idx / CELLS_PER_HYPERCOL) / GRID; }
__device__ inline int decompose_korient(int idx) {
    return (idx % CELLS_PER_HYPERCOL) / N_CELLS_PER_ORIENT;
}
__device__ inline int decompose_kcell(int idx) {
    return (idx % CELLS_PER_HYPERCOL) % N_CELLS_PER_ORIENT;
}

__global__ void l4_init_state_kernel(
    double* V, double* w_adapt, double* I_syn,
    int* refrac_remaining, long long* prev_spike_step,
    int n_l4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_l4) return;
    V[idx] = E_L_MV;
    w_adapt[idx] = 0.0;
    I_syn[idx] = 0.0;
    refrac_remaining[idx] = 0;
    prev_spike_step[idx] = -1;
}

__global__ void l4_init_aggregators_kernel(
    long long* isi_count, double* isi_sum, double* isi_sum_sq,
    long long* total_spikes_full_run, int n_l4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_l4) return;
    isi_count[idx] = 0;
    isi_sum[idx] = 0.0;
    isi_sum_sq[idx] = 0.0;
    total_spikes_full_run[idx] = 0;
}

__global__ void clear_int_array(int* p, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    p[idx] = 0;
}

// Run an entire test-orientation phase (warmup + measurement steps) for all
// 131k cells.  One thread per cell.  RNG state is register-resident (Philox
// counter-based, keyed by (seed, idx, phase_idx)).
//
// Per step:
//   1. Compute time-varying intensity at this cell's retinotopic position.
//   2. Compute per-cell input rate = R_base + R_max * tuning * intensity.
//   3. Bernoulli draw -- if input spike, I_syn += Q_E.
//   4. Decay I_syn by exp(-dt/tau_E).
//   5. AdEx forward-Euler update for V, w_adapt.
//   6. Spike detection / reset / refractory bookkeeping.
//   7. If step >= n_warmup: count toward measurement spike_counts.
//   8. Update ISI accumulators (when not in warmup).
//   9. If sample cell, append spike step to per-sample buffer.
__global__ void l4_run_phase_kernel(
    // persistent state
    double* V, double* w_adapt, double* I_syn,
    int* refrac_remaining, long long* prev_spike_step,
    // persistent accumulators (across all phases)
    long long* isi_count, double* isi_sum, double* isi_sum_sq,
    long long* total_spikes_full_run,
    // per-phase measurement spike count (cleared by host before launch)
    int* spike_counts_phase,
    // sample-cell raster bookkeeping (persistent buffer; running offsets)
    const int* __restrict__ sample_indices,
    int* sample_spike_steps,
    int* sample_spike_count,
    int n_sample,
    int max_sample_spikes,
    long long phase_step_offset,   // global step offset for raster timestamping
    // params
    int n_steps_total,
    int n_warmup_steps,
    double cos_theta_stim,
    double sin_theta_stim,
    double theta_stim_rad,
    double drift_omega_rad_per_s,
    std::uint64_t base_seed,
    int phase_idx
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_L4) return;

    // Decompose cell index.
    const int x_ret    = decompose_x(idx);
    const int y_ret    = decompose_y(idx);
    const int k_orient = decompose_korient(idx);
    // k_cell unused in dynamics (cells in same column share tuning, differ
    // only by RNG seed -> different Poisson realizations -> diverging spike
    // trains).
    const double theta_pref = static_cast<double>(k_orient) * (PI / static_cast<double>(N_ORIENT));

    // Per-cell tuning factor (constant for this phase).
    const double dtheta = theta_stim_rad - theta_pref;
    const double tuning = von_mises_tuning(dtheta, KAPPA);

    // Spatial phase (constant for this phase since cell position is fixed).
    const double spatial_phase =
        K_SPATIAL * (static_cast<double>(x_ret) * cos_theta_stim
                    + static_cast<double>(y_ret) * sin_theta_stim);

    // Per-cell decay factor (precomputed once).
    const double alpha_E = exp(-DT_MS / TAU_E_MS);

    // Load persistent state into registers.
    double v   = V[idx];
    double wa  = w_adapt[idx];
    double isn = I_syn[idx];
    int    rfr = refrac_remaining[idx];
    long long prev_spk = prev_spike_step[idx];

    long long isi_n   = isi_count[idx];
    double    isi_s   = isi_sum[idx];
    double    isi_ss  = isi_sum_sq[idx];
    long long tot_spk = total_spikes_full_run[idx];
    int       phase_count = 0;

    // Identify sample slot (linear scan over 20 entries -- trivial).
    int sample_slot = -1;
    for (int s = 0; s < n_sample; ++s) {
        if (sample_indices[s] == idx) { sample_slot = s; break; }
    }

    // Per-thread Philox RNG -- independent stream per (cell, phase).
    curandStatePhilox4_32_10_t rng;
    // sequence = idx; offset advances inside the generator naturally, but we
    // also bump the subsequence using phase_idx so different phases see
    // independent sub-streams.
    curand_init(base_seed,
                static_cast<unsigned long long>(idx)
                    + static_cast<unsigned long long>(phase_idx) * 1000003ULL,
                0ULL,
                &rng);

    constexpr int T_REF_STEPS = 20;  // T_REF_MS / DT_MS = 2/0.1
    const double inv_C  = 1.0 / C_M_PF;
    const double inv_TW = 1.0 / TAU_W_MS;
    const double inv_DT_DELTA = 1.0 / DELTA_T_MV;

    for (int step = 0; step < n_steps_total; ++step) {
        const double t_in_phase_s = static_cast<double>(step) * DT_S;
        const double phase_arg =
            spatial_phase - drift_omega_rad_per_s * t_in_phase_s;
        const double intensity = 0.5 * (1.0 + cos(phase_arg));

        // Per-cell input rate.
        const double rate_hz = R_BASE_HZ + R_MAX_HZ * tuning * intensity;
        const double p_input = rate_hz * DT_S;

        // Bernoulli draw.
        const double u = curand_uniform_double(&rng);
        if (u < p_input) {
            isn += Q_E_PA;
        }

        // Decay synaptic current after potentially adding -- this gives a
        // single-exponential PSC kernel where the spike's contribution
        // decays over subsequent timesteps.
        isn *= alpha_E;

        // AdEx integration (forward Euler).
        if (rfr > 0) {
            // Refractory clamp.
            v = V_R_MV;
            // Adaptation continues to evolve during refractory.
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;
            wa += DT_MS * dwa;
            rfr -= 1;
        } else {
            // Spike-initiation exponential, capped to prevent overflow.
            double exp_arg = (v - V_T_MV) * inv_DT_DELTA;
            if (exp_arg > 50.0) exp_arg = 50.0;
            const double spike_drive = G_L_NS * DELTA_T_MV * exp(exp_arg);

            // dV/dt = (-g_L*(V-E_L) + g_L*Δ_T*exp((V-V_T)/Δ_T) - w + I_syn)/C
            const double leak = -G_L_NS * (v - E_L_MV);
            const double dv = (leak + spike_drive - wa + isn) * inv_C;
            const double dwa = (A_NS * (v - E_L_MV) - wa) * inv_TW;

            v  += DT_MS * dv;
            wa += DT_MS * dwa;

            if (v > V_PEAK_MV) {
                // Emit spike.
                v = V_R_MV;
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

                    if (sample_slot >= 0) {
                        const int slot_pos = sample_spike_count[sample_slot];
                        if (slot_pos < max_sample_spikes) {
                            sample_spike_steps[
                                sample_slot * max_sample_spikes + slot_pos
                            ] = static_cast<int>(step_global);
                            sample_spike_count[sample_slot] = slot_pos + 1;
                        }
                    }
                }
            }
        }
    }

    // Write back persistent state.
    V[idx]               = v;
    w_adapt[idx]         = wa;
    I_syn[idx]           = isn;
    refrac_remaining[idx]= rfr;
    prev_spike_step[idx] = prev_spk;
    isi_count[idx]       = isi_n;
    isi_sum[idx]         = isi_s;
    isi_sum_sq[idx]      = isi_ss;
    total_spikes_full_run[idx] = tot_spk;
    spike_counts_phase[idx] = phase_count;
}

struct Args {
    std::uint64_t seed = 42;
    int n_test_orient = 16;
    int duration_per_ms = 500;
    int warmup_ms = 100;
    bool drift = true;       // moving grating during tuning test
    std::string out_path;
};

Args parse_args(int argc, char** argv) {
    Args a;
    bool got_out = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) die("missing value for " + name);
            return argv[++i];
        };
        if (k == "--seed") {
            a.seed = static_cast<std::uint64_t>(std::stoll(need(k)));
        } else if (k == "--n-test-orientations") {
            a.n_test_orient = std::stoi(need(k));
        } else if (k == "--duration_per_ms") {
            a.duration_per_ms = std::stoi(need(k));
        } else if (k == "--warmup_ms") {
            a.warmup_ms = std::stoi(need(k));
        } else if (k == "--no-drift") {
            a.drift = false;
        } else if (k == "--out") {
            a.out_path = need(k);
            got_out = true;
        } else if (k == "--help" || k == "-h") {
            std::cout
                << "usage: l4_test [--seed N] [--n-test-orientations N] "
                   "[--duration_per_ms M] [--warmup_ms M] [--no-drift] "
                   "--out PATH\n";
            std::exit(0);
        } else {
            die("unknown option: " + k);
        }
    }
    if (!got_out) die("--out required");
    if (a.n_test_orient <= 0) die("--n-test-orientations must be > 0");
    if (a.duration_per_ms <= 0) die("--duration_per_ms must be > 0");
    if (a.warmup_ms < 0 || a.warmup_ms >= a.duration_per_ms)
        die("--warmup_ms must be in [0, duration_per_ms)");
    return a;
}

// 20 sample cells: 5 retinotopic positions x 4 orientations (k_cell = 0).
std::vector<int> build_sample_indices() {
    const int positions[5][2] = {
        {16, 16}, {8, 8}, {8, 24}, {24, 8}, {24, 24}
    };
    const int orient_picks[4] = {0, 2, 4, 6};  // 0°, 45°, 90°, 135°
    std::vector<int> out;
    out.reserve(N_SAMPLE_CELLS);
    for (int p = 0; p < 5; ++p) {
        const int x = positions[p][0];
        const int y = positions[p][1];
        for (int oi = 0; oi < 4; ++oi) {
            const int k_orient = orient_picks[oi];
            const int idx =
                ((y * GRID + x) * N_ORIENT + k_orient) * N_CELLS_PER_ORIENT
                + 0;  // k_cell = 0
            out.push_back(idx);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        int dev_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        if (dev_count <= 0) {
            std::cerr << "error: no CUDA device available\n";
            return 1;
        }
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        CUDA_CHECK(cudaSetDevice(0));

        const int n_steps_phase = static_cast<int>(args.duration_per_ms / DT_MS + 0.5);
        const int n_warmup_steps = static_cast<int>(args.warmup_ms / DT_MS + 0.5);
        const int n_measure_steps = n_steps_phase - n_warmup_steps;
        const double measure_duration_s = n_measure_steps * DT_S;
        const double drift_omega = args.drift ? OMEGA_TEMPORAL_DEFAULT : 0.0;

        // Sample cell setup.
        const std::vector<int> sample_indices = build_sample_indices();
        if ((int)sample_indices.size() != N_SAMPLE_CELLS)
            die("sample-cell builder produced wrong count");

        // ---- device allocations ----
        double *d_V = nullptr, *d_w = nullptr, *d_Isyn = nullptr;
        int *d_refrac = nullptr;
        long long *d_prev = nullptr;
        long long *d_isi_count = nullptr;
        double *d_isi_sum = nullptr, *d_isi_sum_sq = nullptr;
        long long *d_tot_spk = nullptr;
        int *d_spike_phase = nullptr;
        int *d_sample_idx = nullptr;
        int *d_sample_steps = nullptr;
        int *d_sample_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_V,         N_L4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_w,         N_L4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Isyn,      N_L4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_refrac,    N_L4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_prev,      N_L4 * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_isi_count, N_L4 * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_isi_sum,   N_L4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_isi_sum_sq,N_L4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_tot_spk,   N_L4 * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_spike_phase, N_L4 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sample_idx, N_SAMPLE_CELLS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sample_steps,
                              N_SAMPLE_CELLS * MAX_SAMPLE_SPIKES * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sample_count, N_SAMPLE_CELLS * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_sample_idx, sample_indices.data(),
                              N_SAMPLE_CELLS * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sample_steps, 0,
                              N_SAMPLE_CELLS * MAX_SAMPLE_SPIKES * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_sample_count, 0,
                              N_SAMPLE_CELLS * sizeof(int)));

        const int block = 256;
        const int gridN = (N_L4 + block - 1) / block;

        l4_init_state_kernel<<<gridN, block>>>(
            d_V, d_w, d_Isyn, d_refrac, d_prev, N_L4);
        CUDA_CHECK(cudaGetLastError());
        l4_init_aggregators_kernel<<<gridN, block>>>(
            d_isi_count, d_isi_sum, d_isi_sum_sq, d_tot_spk, N_L4);
        CUDA_CHECK(cudaGetLastError());

        // ---- per-orientation sweep ----
        std::vector<double> test_orient_deg(args.n_test_orient);
        // Span [0, 180) so all preferred orientations are covered.
        for (int i = 0; i < args.n_test_orient; ++i) {
            test_orient_deg[i] = (180.0 * i) / args.n_test_orient;
        }

        // Tuning matrix: [n_test_orient][N_L4]  (kept on host as flat array).
        std::vector<int> tuning_counts_host(
            static_cast<std::size_t>(args.n_test_orient) * N_L4, 0);

        const auto t_run0 = std::chrono::steady_clock::now();
        long long phase_step_offset = 0;
        for (int phase = 0; phase < args.n_test_orient; ++phase) {
            const double theta_stim_rad =
                test_orient_deg[phase] * PI / 180.0;
            const double cos_t = std::cos(theta_stim_rad);
            const double sin_t = std::sin(theta_stim_rad);

            // Reset per-phase state (V, w, I_syn, refrac, prev_spike).
            l4_init_state_kernel<<<gridN, block>>>(
                d_V, d_w, d_Isyn, d_refrac, d_prev, N_L4);
            // Clear per-phase spike count.
            clear_int_array<<<gridN, block>>>(d_spike_phase, N_L4);
            CUDA_CHECK(cudaGetLastError());

            l4_run_phase_kernel<<<gridN, block>>>(
                d_V, d_w, d_Isyn, d_refrac, d_prev,
                d_isi_count, d_isi_sum, d_isi_sum_sq,
                d_tot_spk,
                d_spike_phase,
                d_sample_idx, d_sample_steps, d_sample_count,
                N_SAMPLE_CELLS, MAX_SAMPLE_SPIKES,
                phase_step_offset,
                n_steps_phase, n_warmup_steps,
                cos_t, sin_t, theta_stim_rad,
                drift_omega,
                args.seed, phase
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Read back the per-phase spike counts.
            CUDA_CHECK(cudaMemcpy(
                tuning_counts_host.data()
                    + static_cast<std::size_t>(phase) * N_L4,
                d_spike_phase, N_L4 * sizeof(int),
                cudaMemcpyDeviceToHost));

            phase_step_offset += n_steps_phase;
        }
        const auto t_run1 = std::chrono::steady_clock::now();
        const double run_wall_s =
            std::chrono::duration<double>(t_run1 - t_run0).count();

        // ---- read back aggregates ----
        std::vector<long long> isi_count_host(N_L4);
        std::vector<double>    isi_sum_host(N_L4);
        std::vector<double>    isi_sum_sq_host(N_L4);
        std::vector<long long> tot_spk_host(N_L4);
        std::vector<int>       sample_count_host(N_SAMPLE_CELLS);
        std::vector<int>       sample_steps_host(
            N_SAMPLE_CELLS * MAX_SAMPLE_SPIKES);

        CUDA_CHECK(cudaMemcpy(isi_count_host.data(), d_isi_count,
                              N_L4 * sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(isi_sum_host.data(), d_isi_sum,
                              N_L4 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(isi_sum_sq_host.data(), d_isi_sum_sq,
                              N_L4 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tot_spk_host.data(), d_tot_spk,
                              N_L4 * sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sample_count_host.data(), d_sample_count,
                              N_SAMPLE_CELLS * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sample_steps_host.data(), d_sample_steps,
                              N_SAMPLE_CELLS * MAX_SAMPLE_SPIKES * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // free device buffers
        CUDA_CHECK(cudaFree(d_V));         CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_Isyn));      CUDA_CHECK(cudaFree(d_refrac));
        CUDA_CHECK(cudaFree(d_prev));      CUDA_CHECK(cudaFree(d_isi_count));
        CUDA_CHECK(cudaFree(d_isi_sum));   CUDA_CHECK(cudaFree(d_isi_sum_sq));
        CUDA_CHECK(cudaFree(d_tot_spk));   CUDA_CHECK(cudaFree(d_spike_phase));
        CUDA_CHECK(cudaFree(d_sample_idx));
        CUDA_CHECK(cudaFree(d_sample_steps));
        CUDA_CHECK(cudaFree(d_sample_count));

        // ---- analysis ----
        // Per-cell rate at each test orientation = phase_count / measure_dur.
        // Already have integer counts in tuning_counts_host.

        // Per-cell ISI mean/CV across full run (only for cells with >=2 ISIs).
        std::vector<double> isi_mean_ms_per_cell(N_L4, 0.0);
        std::vector<double> isi_cv_per_cell(N_L4, 0.0);
        std::vector<int>    has_cv(N_L4, 0);
        long long n_cells_with_cv = 0;
        double cv_sum = 0.0, cv_sum_sq = 0.0;
        std::vector<double> cv_collected;
        cv_collected.reserve(N_L4);
        for (int i = 0; i < N_L4; ++i) {
            const long long n = isi_count_host[i];
            if (n >= 2) {
                const double mean_steps = isi_sum_host[i] / static_cast<double>(n);
                const double var = std::max(
                    0.0,
                    isi_sum_sq_host[i] / static_cast<double>(n)
                        - mean_steps * mean_steps);
                const double std_ = std::sqrt(var);
                const double cv = std_ / mean_steps;
                isi_mean_ms_per_cell[i] = mean_steps * DT_MS;
                isi_cv_per_cell[i] = cv;
                has_cv[i] = 1;
                cv_collected.push_back(cv);
                cv_sum += cv;
                cv_sum_sq += cv * cv;
                ++n_cells_with_cv;
            }
        }
        double cv_mean = 0.0, cv_std = 0.0, cv_median = 0.0;
        if (n_cells_with_cv > 0) {
            cv_mean = cv_sum / n_cells_with_cv;
            const double mean_sq = cv_mean * cv_mean;
            cv_std = std::sqrt(std::max(
                0.0, cv_sum_sq / n_cells_with_cv - mean_sq));
            std::vector<double> tmp = cv_collected;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
            cv_median = tmp[tmp.size()/2];
        }

        // Mean rate per cell across the full run (Hz).
        const double total_meas_s = measure_duration_s * args.n_test_orient;
        std::vector<double> mean_rate_hz_per_cell(N_L4, 0.0);
        // Sum spike counts across all phases for each cell.
        for (int i = 0; i < N_L4; ++i) {
            long long s = 0;
            for (int p = 0; p < args.n_test_orient; ++p) {
                s += tuning_counts_host[
                    static_cast<std::size_t>(p) * N_L4 + i];
            }
            mean_rate_hz_per_cell[i] = static_cast<double>(s) / total_meas_s;
        }

        // Aggregate rate stats across all cells.
        double rate_sum = 0.0, rate_sum_sq = 0.0;
        double rate_min = mean_rate_hz_per_cell[0];
        double rate_max = mean_rate_hz_per_cell[0];
        for (auto v : mean_rate_hz_per_cell) {
            rate_sum += v;
            rate_sum_sq += v * v;
            if (v < rate_min) rate_min = v;
            if (v > rate_max) rate_max = v;
        }
        const double rate_mean = rate_sum / N_L4;
        const double rate_std = std::sqrt(std::max(
            0.0, rate_sum_sq / N_L4 - rate_mean * rate_mean));

        // Rate per orientation -- mean across cells per phase.
        std::vector<double> rate_per_orient_mean_hz(args.n_test_orient, 0.0);
        for (int p = 0; p < args.n_test_orient; ++p) {
            long long s = 0;
            for (int i = 0; i < N_L4; ++i) {
                s += tuning_counts_host[
                    static_cast<std::size_t>(p) * N_L4 + i];
            }
            rate_per_orient_mean_hz[p] =
                static_cast<double>(s) / (measure_duration_s * N_L4);
        }

        // Sample-cell tuning curves and metadata.
        // ----- write JSON -----
        std::ofstream out(args.out_path);
        if (!out) die("could not open out path: " + args.out_path);
        out << std::setprecision(8);
        out << "{\n";
        out << "  \"device\": \"" << prop.name << "\",\n";
        out << "  \"seed\": " << args.seed << ",\n";
        out << "  \"n_l4_cells\": " << N_L4 << ",\n";
        out << "  \"grid\": " << GRID << ",\n";
        out << "  \"n_orient_columns\": " << N_ORIENT << ",\n";
        out << "  \"cells_per_orient\": " << N_CELLS_PER_ORIENT << ",\n";
        out << "  \"n_test_orientations\": " << args.n_test_orient << ",\n";
        out << "  \"duration_per_ms\": " << args.duration_per_ms << ",\n";
        out << "  \"warmup_ms\": " << args.warmup_ms << ",\n";
        out << "  \"measure_ms\": " << (args.duration_per_ms - args.warmup_ms)
            << ",\n";
        out << "  \"drift\": " << (args.drift ? "true" : "false") << ",\n";
        out << "  \"dt_ms\": " << DT_MS << ",\n";
        out << "  \"adex_params\": {";
        out << "\"C_pF\":"<<C_M_PF<<",\"gL_nS\":"<<G_L_NS
            <<",\"EL_mV\":"<<E_L_MV<<",\"VT_mV\":"<<V_T_MV
            <<",\"DeltaT_mV\":"<<DELTA_T_MV<<",\"a_nS\":"<<A_NS
            <<",\"tauW_ms\":"<<TAU_W_MS<<",\"b_pA\":"<<B_PA
            <<",\"VR_mV\":"<<V_R_MV<<",\"Vpeak_mV\":"<<V_PEAK_MV
            <<",\"tRef_ms\":"<<T_REF_MS<<"},\n";
        out << "  \"input_params\": {\"R_base_Hz\":"<<R_BASE_HZ
            <<",\"R_max_Hz\":"<<R_MAX_HZ<<",\"kappa\":"<<KAPPA
            <<",\"tau_E_ms\":"<<TAU_E_MS<<",\"Q_E_pA\":"<<Q_E_PA<<"},\n";
        out << "  \"stim_params\": {\"k_spatial\":"<<K_SPATIAL
            <<",\"omega_drift_rad_per_s\":"<<drift_omega<<"},\n";
        out << "  \"run_wall_s\": " << run_wall_s << ",\n";

        out << "  \"test_orientations_deg\": [";
        for (int i = 0; i < args.n_test_orient; ++i) {
            if (i) out << ",";
            out << test_orient_deg[i];
        }
        out << "],\n";

        out << "  \"all_cells_rate_stats\": {";
        out << "\"mean_hz\":"<<rate_mean<<",\"std_hz\":"<<rate_std
            <<",\"min_hz\":"<<rate_min<<",\"max_hz\":"<<rate_max<<"},\n";

        out << "  \"all_cells_isi_cv_stats\": {";
        out << "\"n_cells\":"<<n_cells_with_cv<<",\"mean\":"<<cv_mean
            <<",\"std\":"<<cv_std<<",\"median\":"<<cv_median<<"},\n";

        out << "  \"rate_per_orient_mean_hz\": [";
        for (int p = 0; p < args.n_test_orient; ++p) {
            if (p) out << ",";
            out << rate_per_orient_mean_hz[p];
        }
        out << "],\n";

        // Histograms: rate (50 bins from 0 to rate_max) and ISI CV (50 bins 0..2).
        const int N_BIN = 50;
        std::vector<int> rate_hist(N_BIN, 0);
        const double rate_hist_max = std::max(1.0, rate_max);
        for (auto v : mean_rate_hz_per_cell) {
            int b = static_cast<int>(v / rate_hist_max * N_BIN);
            if (b < 0) b = 0; else if (b >= N_BIN) b = N_BIN - 1;
            rate_hist[b]++;
        }
        std::vector<int> cv_hist(N_BIN, 0);
        const double cv_hist_max = 2.0;
        for (auto v : cv_collected) {
            int b = static_cast<int>(v / cv_hist_max * N_BIN);
            if (b < 0) b = 0; else if (b >= N_BIN) b = N_BIN - 1;
            cv_hist[b]++;
        }
        out << "  \"rate_histogram\": {\"max_hz\":"<<rate_hist_max
            <<",\"counts\":[";
        for (int i = 0; i < N_BIN; ++i) { if (i) out<<","; out<<rate_hist[i]; }
        out << "]},\n";
        out << "  \"cv_histogram\": {\"max\":"<<cv_hist_max<<",\"counts\":[";
        for (int i = 0; i < N_BIN; ++i) { if (i) out<<","; out<<cv_hist[i]; }
        out << "]},\n";

        // Sample cells with metadata + tuning curves + spike trains.
        out << "  \"sample_cells\": [\n";
        for (int s = 0; s < N_SAMPLE_CELLS; ++s) {
            const int idx = sample_indices[s];
            const int x  = (idx / CELLS_PER_HYPERCOL) % GRID;
            const int y  = (idx / CELLS_PER_HYPERCOL) / GRID;
            const int ko = (idx % CELLS_PER_HYPERCOL) / N_CELLS_PER_ORIENT;
            const int kc = (idx % CELLS_PER_HYPERCOL) % N_CELLS_PER_ORIENT;
            const double theta_pref_deg = ko * (180.0 / N_ORIENT);
            out << "    {\"slot\":" << s
                << ",\"index\":" << idx
                << ",\"x\":" << x << ",\"y\":" << y
                << ",\"k_orient\":" << ko << ",\"k_cell\":" << kc
                << ",\"theta_pref_deg\":" << theta_pref_deg
                << ",\"isi_mean_ms\":" << isi_mean_ms_per_cell[idx]
                << ",\"isi_cv\":" << isi_cv_per_cell[idx]
                << ",\"isi_n_samples\":" << isi_count_host[idx]
                << ",\"tuning_curve_hz\":[";
            for (int p = 0; p < args.n_test_orient; ++p) {
                if (p) out << ",";
                const int c = tuning_counts_host[
                    static_cast<std::size_t>(p) * N_L4 + idx];
                out << static_cast<double>(c) / measure_duration_s;
            }
            out << "],\"spike_count_total\":" << sample_count_host[s]
                << ",\"spike_steps\":[";
            const int spike_n = std::min(
                sample_count_host[s], MAX_SAMPLE_SPIKES);
            for (int i = 0; i < spike_n; ++i) {
                if (i) out << ",";
                out << sample_steps_host[s * MAX_SAMPLE_SPIKES + i];
            }
            out << "]}";
            if (s + 1 < N_SAMPLE_CELLS) out << ",";
            out << "\n";
        }
        out << "  ],\n";

        out << "  \"phase_step_offsets\": [";
        for (int p = 0; p < args.n_test_orient; ++p) {
            if (p) out << ",";
            out << static_cast<long long>(p) * n_steps_phase;
        }
        out << "],\n";
        out << "  \"n_steps_per_phase\": " << n_steps_phase << "\n";
        out << "}\n";
        out.close();

        // ---- stdout summary ----
        std::cout << "device=" << prop.name << "\n";
        std::cout << "n_l4_cells=" << N_L4 << "\n";
        std::cout << "n_test_orientations=" << args.n_test_orient << "\n";
        std::cout << "duration_per_ms=" << args.duration_per_ms << "\n";
        std::cout << "warmup_ms=" << args.warmup_ms << "\n";
        std::cout << "drift=" << (args.drift ? 1 : 0) << "\n";
        std::cout << "run_wall_s=" << run_wall_s << "\n";
        std::cout << "all_cells_mean_rate_hz=" << rate_mean << "\n";
        std::cout << "all_cells_std_rate_hz=" << rate_std << "\n";
        std::cout << "all_cells_min_rate_hz=" << rate_min << "\n";
        std::cout << "all_cells_max_rate_hz=" << rate_max << "\n";
        std::cout << "n_cells_with_cv=" << n_cells_with_cv << "\n";
        std::cout << "isi_cv_mean=" << cv_mean << "\n";
        std::cout << "isi_cv_std=" << cv_std << "\n";
        std::cout << "isi_cv_median=" << cv_median << "\n";
        std::cout << "rate_per_orient_mean_hz=";
        for (int p = 0; p < args.n_test_orient; ++p) {
            if (p) std::cout << ",";
            std::cout << rate_per_orient_mean_hz[p];
        }
        std::cout << "\n";
        std::cout << "out_path=" << args.out_path << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
