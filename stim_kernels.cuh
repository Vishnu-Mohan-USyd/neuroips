// stim_kernels.cuh
//
// Shared moving-grating stim formula used by stim_test.cu (Poisson per-pixel
// generator from task #46) and adex_v1.cu (per-cell input rate after tuning
// modulation, task #47).
//
// stim_test.cu retains its own inline implementation for backward compat;
// adex_v1.cu calls the helpers defined here so the formula is provably the
// same -- if the team wants stim_test.cu to call into the helper too, that's
// a one-line refactor.
//
// Rate at a 32x32 retinotopic pixel:
//     intensity(x, y, t) = (1 + cos(k*(x cosθ + y sinθ) - ω t)) / 2
//     rate(x, y, t)      = base + max * intensity(x, y, t)
// with task-#46 defaults base=1 Hz, max=50 Hz, k=2π/8 rad/px, ω=2π·4 rad/s.

#pragma once

#include <cmath>

namespace stim_kernels {

constexpr int    GRID                          = 32;
constexpr int    N_PIX                         = GRID * GRID;
constexpr double PI                            = 3.14159265358979323846;
constexpr double DT_MS                         = 0.1;
constexpr double DT_S                          = DT_MS / 1000.0;
constexpr double STIM_BASE_RATE_HZ             = 1.0;
constexpr double STIM_MAX_RATE_HZ              = 50.0;
constexpr double STIM_DEFAULT_SF_PERIOD_PIXELS = 8.0;
constexpr double STIM_DEFAULT_OMEGA_RAD_PER_S  = 2.0 * PI * 4.0;

__host__ __device__ inline double stim_intensity(
    int x, int y, double t_s,
    double cos_theta, double sin_theta,
    double k_spatial, double omega_rad_per_s
) {
    const double phase =
        k_spatial * (static_cast<double>(x) * cos_theta
                    + static_cast<double>(y) * sin_theta)
        - omega_rad_per_s * t_s;
    return 0.5 * (1.0 + cos(phase));
}

__host__ __device__ inline double stim_rate_hz(
    int x, int y, double t_s,
    double cos_theta, double sin_theta,
    double k_spatial, double omega_rad_per_s
) {
    return STIM_BASE_RATE_HZ + STIM_MAX_RATE_HZ * stim_intensity(
        x, y, t_s, cos_theta, sin_theta, k_spatial, omega_rad_per_s
    );
}

}  // namespace stim_kernels
