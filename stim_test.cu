// stim_test.cu
//
// Moving grating Poisson spike generator on a 32x32 retinotopic grid.
// GPU-only: every pixel's full timestep loop runs on one CUDA thread; only
// final counts/rate-sums and a small batch of intensity snapshots get pulled
// back to the host. Single kernel launch covers the entire run.
//
// CLI:
//     stim_test --orientation <deg> --duration_ms <int> --out <path> [--seed <int>]
//
// Hard-fixed parameters (per task #46 spec):
//     32x32 grid, dt = 0.1 ms, base_rate = 1 Hz, max_rate = 50 Hz,
//     k_spatial = 2*pi/8 rad/pixel (1 cycle / 8 px),
//     omega_temporal = 2*pi*4 rad/s (4 Hz drift).

#include <cuda_runtime.h>
#include <curand_kernel.h>

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

constexpr int GRID = 32;
constexpr int N_PIX = GRID * GRID;
constexpr double PI = 3.14159265358979323846;
constexpr double DT_MS = 0.1;
constexpr double DT_S = DT_MS / 1000.0;
constexpr double BASE_RATE_HZ = 1.0;
constexpr double MAX_RATE_HZ = 50.0;
// Default spatial period: one full grating cycle every 8 pixels.
// Override with --sf-period-pixels N (larger N => lower SF, longer bars).
constexpr double DEFAULT_SF_PERIOD_PIXELS = 8.0;
constexpr double OMEGA_TEMPORAL = 2.0 * PI * 4.0;
constexpr int N_SNAPSHOTS = 8;
constexpr double EXPECTED_MEAN_RATE_HZ = BASE_RATE_HZ + 0.5 * MAX_RATE_HZ;

// CUDA error helper. Throws on non-success so cudaMalloc / launch / memcpy
// failures abort the run with a useful message.
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

// Per-thread Philox4_32_10 state: counter-based RNG that is the standard
// for parallel Monte Carlo on GPUs. curand_init(seed, sequence, offset,
// state) gives each thread an independent stream keyed by `sequence`, so
// thread N's draws are uncorrelated with thread M's (unlike a streaming
// generator with naive offset seeding, which can produce shifted-copy
// correlations between threads).

// Single kernel launch handles the full duration: each thread owns one pixel,
// loops n_steps times, draws one Bernoulli (Poisson approx for p << 1) per
// step, and writes the integrated spike count + integrated rate at the end.
//
// At dt = 0.1 ms with max rate 50 Hz, p_max = 0.005 — Bernoulli ≈ Poisson.
__global__ void stim_run_kernel(
    int* __restrict__ spike_counts,
    double* __restrict__ rate_time_sum,
    std::uint64_t base_seed,
    int n_steps,
    double cos_theta,
    double sin_theta,
    double k_spatial
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PIX) return;
    const int x = idx % GRID;
    const int y = idx / GRID;

    // Per-thread Philox state — independent stream per pixel via the
    // sequence parameter. Register-resident; no global state array.
    curandStatePhilox4_32_10_t rng;
    curand_init(base_seed, static_cast<unsigned long long>(idx), 0ULL, &rng);

    int local_count = 0;
    double local_rate_sum = 0.0;

    const double xf = static_cast<double>(x);
    const double yf = static_cast<double>(y);
    const double spatial_phase = k_spatial * (xf * cos_theta + yf * sin_theta);

    for (int step = 0; step < n_steps; ++step) {
        const double t_s = static_cast<double>(step) * DT_S;
        const double phase = spatial_phase - OMEGA_TEMPORAL * t_s;
        const double intensity = 0.5 * (1.0 + cos(phase));   // [0, 1]
        const double rate_hz = BASE_RATE_HZ + MAX_RATE_HZ * intensity;
        const double p = rate_hz * DT_S;

        local_rate_sum += rate_hz;

        // curand_uniform_double returns in (0, 1].
        const double u = curand_uniform_double(&rng);
        if (u < p) {
            ++local_count;
        }
    }

    spike_counts[idx] = local_count;
    rate_time_sum[idx] = local_rate_sum;
}

// Deterministic intensity field at a single timepoint, used to dump
// snapshots that prove the grating moves.
__global__ void snapshot_kernel(
    double* __restrict__ intensity_grid,
    double t_s,
    double cos_theta,
    double sin_theta,
    double k_spatial
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_PIX) return;
    const int x = idx % GRID;
    const int y = idx / GRID;
    const double phase =
        k_spatial * (static_cast<double>(x) * cos_theta
                     + static_cast<double>(y) * sin_theta)
        - OMEGA_TEMPORAL * t_s;
    intensity_grid[idx] = 0.5 * (1.0 + cos(phase));
}

struct Args {
    double orientation_deg = 0.0;
    int duration_ms = 500;
    std::string out_path;
    std::uint64_t seed = 42;
    double sf_period_pixels = DEFAULT_SF_PERIOD_PIXELS;
};

[[noreturn]] void die(const std::string& msg) {
    throw std::runtime_error(msg);
}

Args parse_args(int argc, char** argv) {
    Args a;
    bool got_orientation = false;
    bool got_duration = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) die("missing value for " + name);
            return argv[++i];
        };
        if (k == "--orientation") {
            a.orientation_deg = std::stod(need(k));
            got_orientation = true;
        } else if (k == "--duration_ms") {
            a.duration_ms = std::stoi(need(k));
            got_duration = true;
        } else if (k == "--out") {
            a.out_path = need(k);
        } else if (k == "--seed") {
            a.seed = static_cast<std::uint64_t>(std::stoll(need(k)));
        } else if (k == "--sf-period-pixels") {
            a.sf_period_pixels = std::stod(need(k));
        } else if (k == "--help" || k == "-h") {
            std::cout << "usage: stim_test --orientation <deg> "
                         "--duration_ms <int> --out <path> [--seed <int>] "
                         "[--sf-period-pixels <float>]\n";
            std::exit(0);
        } else {
            die("unknown option: " + k);
        }
    }
    if (!got_orientation) die("--orientation required");
    if (!got_duration) die("--duration_ms required");
    if (a.out_path.empty()) die("--out required");
    if (a.duration_ms <= 0) die("--duration_ms must be > 0");
    if (a.sf_period_pixels <= 0.0) die("--sf-period-pixels must be > 0");
    return a;
}

void write_grid_int(std::ostream& os, const std::vector<int>& data) {
    os << "[";
    for (int r = 0; r < GRID; ++r) {
        if (r > 0) os << ",";
        os << "[";
        for (int c = 0; c < GRID; ++c) {
            if (c > 0) os << ",";
            os << data[r * GRID + c];
        }
        os << "]";
    }
    os << "]";
}

void write_grid_double(std::ostream& os, const std::vector<double>& data) {
    os << "[";
    for (int r = 0; r < GRID; ++r) {
        if (r > 0) os << ",";
        os << "[";
        for (int c = 0; c < GRID; ++c) {
            if (c > 0) os << ",";
            os << data[r * GRID + c];
        }
        os << "]";
    }
    os << "]";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        // Hard-fail with no GPU — this binary is GPU-only by construction.
        int dev_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        if (dev_count <= 0) {
            std::cerr << "error: no CUDA device available\n";
            return 1;
        }
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        CUDA_CHECK(cudaSetDevice(0));

        const int n_steps = static_cast<int>(args.duration_ms / DT_MS + 0.5);
        const double duration_s = args.duration_ms / 1000.0;
        const double theta_rad = args.orientation_deg * PI / 180.0;
        const double cos_theta = std::cos(theta_rad);
        const double sin_theta = std::sin(theta_rad);
        const double k_spatial = 2.0 * PI / args.sf_period_pixels;

        // Single allocation set, freed at end. Nothing else lives on host
        // until the readbacks below.
        int* d_spike_counts = nullptr;
        double* d_rate_sum = nullptr;
        double* d_snap = nullptr;
        CUDA_CHECK(cudaMalloc(&d_spike_counts, N_PIX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rate_sum, N_PIX * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_snap, N_PIX * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_spike_counts, 0, N_PIX * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_rate_sum, 0, N_PIX * sizeof(double)));

        const int block = 256;
        const int grid = (N_PIX + block - 1) / block;

        // --- Main run: one kernel covers the whole duration ---
        const auto t0 = std::chrono::steady_clock::now();
        stim_run_kernel<<<grid, block>>>(
            d_spike_counts, d_rate_sum, args.seed,
            n_steps, cos_theta, sin_theta, k_spatial
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- Snapshots: 8 evenly spaced timepoints in [0, duration_ms) ---
        std::vector<double> snap_t_ms(N_SNAPSHOTS);
        std::vector<std::vector<double>> snapshots(
            N_SNAPSHOTS, std::vector<double>(N_PIX)
        );
        for (int s = 0; s < N_SNAPSHOTS; ++s) {
            const double t_ms =
                (static_cast<double>(args.duration_ms) * s) / N_SNAPSHOTS;
            const double t_s = t_ms / 1000.0;
            snap_t_ms[s] = t_ms;
            snapshot_kernel<<<grid, block>>>(
                d_snap, t_s, cos_theta, sin_theta, k_spatial
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(
                snapshots[s].data(), d_snap,
                N_PIX * sizeof(double), cudaMemcpyDeviceToHost
            ));
        }

        // --- Readback ---
        std::vector<int> spike_counts(N_PIX);
        std::vector<double> rate_sum(N_PIX);
        CUDA_CHECK(cudaMemcpy(
            spike_counts.data(), d_spike_counts,
            N_PIX * sizeof(int), cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(
            rate_sum.data(), d_rate_sum,
            N_PIX * sizeof(double), cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaFree(d_spike_counts));
        CUDA_CHECK(cudaFree(d_rate_sum));
        CUDA_CHECK(cudaFree(d_snap));

        // --- Aggregates ---
        std::vector<double> expected_rate(N_PIX);
        std::vector<double> empirical_rate(N_PIX);
        for (int i = 0; i < N_PIX; ++i) {
            expected_rate[i] = rate_sum[i] / static_cast<double>(n_steps);
            empirical_rate[i] = spike_counts[i] / duration_s;
        }
        long total_spikes = 0;
        for (auto v : spike_counts) total_spikes += v;
        const double total_expected = EXPECTED_MEAN_RATE_HZ * N_PIX * duration_s;
        const double mean_count = static_cast<double>(total_spikes) / N_PIX;
        double sse = 0.0;
        for (auto v : spike_counts) {
            const double d = v - mean_count;
            sse += d * d;
        }
        const double std_count = std::sqrt(sse / N_PIX);
        const double mean_emp_rate = mean_count / duration_s;

        // Min/max for sanity.
        int spike_min = spike_counts[0];
        int spike_max = spike_counts[0];
        for (auto v : spike_counts) {
            if (v < spike_min) spike_min = v;
            if (v > spike_max) spike_max = v;
        }

        // --- JSON dump ---
        std::ofstream out(args.out_path);
        if (!out) die("could not open out path: " + args.out_path);
        out << std::setprecision(8);
        out << "{\n";
        out << "  \"orientation_deg\": " << args.orientation_deg << ",\n";
        out << "  \"duration_ms\": " << args.duration_ms << ",\n";
        out << "  \"n_steps\": " << n_steps << ",\n";
        out << "  \"dt_ms\": " << DT_MS << ",\n";
        out << "  \"grid\": " << GRID << ",\n";
        out << "  \"base_rate_hz\": " << BASE_RATE_HZ << ",\n";
        out << "  \"max_rate_hz\": " << MAX_RATE_HZ << ",\n";
        out << "  \"k_spatial_rad_per_pixel\": " << k_spatial << ",\n";
        out << "  \"sf_period_pixels\": " << args.sf_period_pixels << ",\n";
        out << "  \"omega_temporal_rad_per_s\": " << OMEGA_TEMPORAL << ",\n";
        out << "  \"seed\": " << args.seed << ",\n";
        out << "  \"expected_mean_rate_hz_per_pixel\": "
            << EXPECTED_MEAN_RATE_HZ << ",\n";
        out << "  \"expected_total_spikes\": " << total_expected << ",\n";
        out << "  \"total_spikes\": " << total_spikes << ",\n";
        out << "  \"mean_spike_count_per_pixel\": " << mean_count << ",\n";
        out << "  \"std_spike_count_per_pixel\": " << std_count << ",\n";
        out << "  \"min_spike_count\": " << spike_min << ",\n";
        out << "  \"max_spike_count\": " << spike_max << ",\n";
        out << "  \"mean_empirical_rate_hz\": " << mean_emp_rate << ",\n";
        out << "  \"gpu_kernel_wall_ms\": " << wall_ms << ",\n";
        out << "  \"device_name\": \"" << prop.name << "\",\n";
        out << "  \"spike_counts\": ";
        write_grid_int(out, spike_counts);
        out << ",\n";
        out << "  \"expected_rate_hz\": ";
        write_grid_double(out, expected_rate);
        out << ",\n";
        out << "  \"empirical_rate_hz\": ";
        write_grid_double(out, empirical_rate);
        out << ",\n";
        out << "  \"snapshots\": [\n";
        for (int s = 0; s < N_SNAPSHOTS; ++s) {
            out << "    {\"t_ms\": " << snap_t_ms[s]
                << ", \"intensity\": ";
            write_grid_double(out, snapshots[s]);
            out << "}";
            if (s + 1 < N_SNAPSHOTS) out << ",";
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        out.close();

        // --- Stdout summary ---
        std::cout << "device=" << prop.name << "\n";
        std::cout << "orientation_deg=" << args.orientation_deg << "\n";
        std::cout << "sf_period_pixels=" << args.sf_period_pixels << "\n";
        std::cout << "k_spatial_rad_per_pixel=" << k_spatial << "\n";
        std::cout << "duration_ms=" << args.duration_ms << "\n";
        std::cout << "n_steps=" << n_steps << "\n";
        std::cout << "dt_ms=" << DT_MS << "\n";
        std::cout << "seed=" << args.seed << "\n";
        std::cout << "total_spikes=" << total_spikes << "\n";
        std::cout << "expected_total_spikes=" << total_expected << "\n";
        std::cout << "mean_spike_count_per_pixel=" << mean_count << "\n";
        std::cout << "std_spike_count_per_pixel=" << std_count << "\n";
        std::cout << "min_spike_count=" << spike_min << "\n";
        std::cout << "max_spike_count=" << spike_max << "\n";
        std::cout << "mean_empirical_rate_hz=" << mean_emp_rate << "\n";
        std::cout << "expected_mean_rate_hz=" << EXPECTED_MEAN_RATE_HZ << "\n";
        std::cout << "ratio_emp_over_expected="
                  << (mean_emp_rate / EXPECTED_MEAN_RATE_HZ) << "\n";
        std::cout << "gpu_kernel_wall_ms=" << wall_ms << "\n";
        std::cout << "out_path=" << args.out_path << "\n";

        // --- Spike-count grid (32x32) for visual grep ---
        std::cout << "\nspike_count_grid (row=y top->bottom, col=x left->right):\n";
        for (int y = 0; y < GRID; ++y) {
            for (int x = 0; x < GRID; ++x) {
                std::cout << std::setw(4) << spike_counts[y * GRID + x];
            }
            std::cout << "\n";
        }

        // --- Snapshot intensity quantised to 0..9 for an ASCII glance ---
        std::cout << "\nsnapshot intensity (quantised 0..9, t shown above each):\n";
        for (int s = 0; s < N_SNAPSHOTS; ++s) {
            std::cout << "t_ms=" << snap_t_ms[s] << "\n";
            for (int y = 0; y < GRID; ++y) {
                for (int x = 0; x < GRID; ++x) {
                    const double v = snapshots[s][y * GRID + x];
                    int q = static_cast<int>(v * 9.999);
                    if (q < 0) q = 0;
                    if (q > 9) q = 9;
                    std::cout << q;
                }
                std::cout << "\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
