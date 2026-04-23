#include "expectation_snn_cuda/manifest.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>

namespace expectation_snn_cuda {

namespace {

constexpr int V1_E_N = 192;
constexpr int V1_SOM_N = 48;
constexpr int H_E_N = 192;
constexpr int H_N_CHANNELS = 12;
constexpr int H_E_PER_CHANNEL = 16;
constexpr int H_INH_N = 16;
constexpr int H_BROAD_INH_START = 12;
constexpr double DT_MS = 0.1;

constexpr double TAU_E_MS = 5.0;
constexpr double TAU_I_MS = 10.0;
constexpr double TAU_NMDA_H_MS = 50.0;
constexpr double V_NMDA_REV_MV = 0.0;

constexpr double V1E_C_SOMA_NF = 0.2;
constexpr double V1E_GL_SOMA_NS = 10.0;
constexpr double V1E_EL_MV = -70.0;
constexpr double V1E_VT_MV = -50.0;
constexpr double V1E_VR_MV = -65.0;
constexpr double V1E_REFACTORY_MS = 2.0;
constexpr double V1E_C_AP_NF = 0.1;
constexpr double V1E_GL_AP_NS = 4.0;
constexpr double V1E_EL_AP_MV = -70.0;
constexpr double V1E_G_AP_SOMA_NS = 2.0;
constexpr double V1E_V_AP_TH_MV = -55.0;
constexpr double V1E_V_AP_SLOPE_MV = 4.0;
constexpr double V1E_A_ADAPT_NS = 0.0;
constexpr double V1E_B_ADAPT_PA = 30.0;
constexpr double V1E_TAU_ADAPT_MS = 150.0;

constexpr double V1SOM_C_NF = 0.15;
constexpr double V1SOM_GL_NS = 10.0;
constexpr double V1SOM_EL_MV = -65.0;
constexpr double V1SOM_VT_MV = -50.0;
constexpr double V1SOM_VR_MV = -65.0;
constexpr double V1SOM_REFACTORY_MS = 2.0;

constexpr double HE_C_NF = 0.2;
constexpr double HE_GL_NS = 10.0;
constexpr double HE_EL_MV = -70.0;
constexpr double HE_VT_MV = -50.0;
constexpr double HE_VR_MV = -65.0;
constexpr double HE_REFACTORY_MS = 2.0;
constexpr double HI_C_NF = 0.15;
constexpr double HI_GL_NS = 10.0;
constexpr double HI_EL_MV = -65.0;
constexpr double HI_VT_MV = -50.0;
constexpr double HI_VR_MV = -65.0;
constexpr double HI_REFACTORY_MS = 2.0;

constexpr double H_RING_EXT_DRIVE_PA = 800.0;
constexpr double H_RING_EE_AMPA_PA = 28.0;
constexpr double H_RING_EE_NMDA_NS = 1.06;
constexpr double H_RING_EI_LOCAL_PA = 130.0;
constexpr double H_RING_EI_BROAD_PA = 35.0;
constexpr double H_RING_IE_LOCAL_PA = 130.0;
constexpr double H_RING_IE_BROAD_PA = 10.0;
constexpr double CTX_PRED_GATE_DRIVE_PA = 400.0;

constexpr double PA_PER_NF_TO_MV_PER_MS = 0.001;
constexpr int COUNT_BEFORE_WINDOW_CELL = 100;
constexpr int COUNT_START_WINDOW_CELL = 101;
constexpr int COUNT_END_WINDOW_CELL = 102;
constexpr int COUNT_AFTER_WINDOW_CELL = 103;

void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(label) + ": " + cudaGetErrorString(status)
        );
    }
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("max_abs_diff size mismatch");
    }
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out = std::max(out, std::abs(a[i] - b[i]));
    }
    return out;
}

template <typename T>
T* device_alloc_copy(const std::vector<T>& host) {
    T* dev = nullptr;
    check_cuda(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc");
    check_cuda(
        cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy host->device"
    );
    return dev;
}

template <typename T>
void copy_device_to_host(T* dev, std::vector<T>& host) {
    check_cuda(
        cudaMemcpy(host.data(), dev, host.size() * sizeof(T), cudaMemcpyDeviceToHost),
        "cudaMemcpy device->host"
    );
}

template <typename T>
void device_free(T* dev) {
    if (dev != nullptr) {
        check_cuda(cudaFree(dev), "cudaFree");
    }
}

__device__ __host__ inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ __host__ inline double refractory_until_ms(
    int step,
    double refractory_ms
) {
    const int refractory_steps = static_cast<int>(refractory_ms / DT_MS + 0.5);
    return static_cast<double>(step + refractory_steps) * DT_MS;
}

__device__ __host__ inline std::uint64_t mix_counter64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __host__ inline double counter_uniform01(
    std::int64_t seed,
    int step,
    int source
) {
    const std::uint64_t x =
        static_cast<std::uint64_t>(seed)
        ^ (static_cast<std::uint64_t>(step) * 0xD1B54A32D192ED03ull)
        ^ (static_cast<std::uint64_t>(source) * 0xABC98388FB8FAC03ull);
    const std::uint64_t mixed = mix_counter64(x);
    return static_cast<double>(mixed >> 11) / 9007199254740992.0;
}

__device__ __host__ inline int richter_phase_index(
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    if (step >= leader_start_step && step < leader_end_step) {
        return 0;
    }
    if (step >= preprobe_start_step && step < preprobe_end_step) {
        return 1;
    }
    if (step >= trailer_start_step && step < trailer_end_step) {
        return 2;
    }
    return -1;
}

__device__ __host__ inline double grating_rate_for_step_channel(
    int step,
    int channel,
    int expected_channel,
    int unexpected_channel,
    double grating_rate_hz,
    double baseline_rate_hz,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    if ((phase == 0 || phase == 1) && channel == expected_channel) {
        return grating_rate_hz;
    }
    if (phase == 2 && channel == unexpected_channel) {
        return grating_rate_hz;
    }
    if (phase >= 0) {
        return baseline_rate_hz;
    }
    return 0.0;
}

__device__ __host__ inline bool seeded_source_event(
    std::int64_t seed,
    int step,
    int source,
    int channel,
    int expected_channel,
    int unexpected_channel,
    double grating_rate_hz,
    double baseline_rate_hz,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    const double rate_hz = grating_rate_for_step_channel(
        step, channel, expected_channel, unexpected_channel, grating_rate_hz,
        baseline_rate_hz, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const double p = fmin(fmax(rate_hz * DT_MS / 1000.0, 0.0), 1.0);
    return counter_uniform01(seed, step, source) < p;
}

void init_v1e_state(
    std::map<std::string, std::vector<double>>& state,
    std::vector<std::int32_t>& spikes,
    bool threshold_case
) {
    state["V_soma_mV"].resize(V1_E_N);
    state["V_ap_mV"].resize(V1_E_N);
    state["I_e_pA"].resize(V1_E_N);
    state["I_i_pA"].resize(V1_E_N);
    state["I_ap_e_pA"].resize(V1_E_N);
    state["w_adapt_pA"].resize(V1_E_N);
    state["I_bias_pA"].resize(V1_E_N);
    state["refrac_until_ms"].resize(V1_E_N);
    spikes.assign(V1_E_N, 0);
    for (int i = 0; i < V1_E_N; ++i) {
        state["V_soma_mV"][i] = -69.5 + 0.002 * static_cast<double>(i % 17);
        state["V_ap_mV"][i] = -70.0 + 0.001 * static_cast<double>(i % 11);
        state["I_e_pA"][i] = 0.5 + 0.01 * static_cast<double>(i % 5);
        state["I_i_pA"][i] = 0.2 + 0.01 * static_cast<double>(i % 3);
        state["I_ap_e_pA"][i] = 0.3 + 0.005 * static_cast<double>(i % 7);
        state["w_adapt_pA"][i] = 0.1 + 0.002 * static_cast<double>(i % 13);
        state["I_bias_pA"][i] = 0.0;
        state["refrac_until_ms"][i] = -1.0;
    }
    if (threshold_case) {
        state["V_soma_mV"][0] = -49.0;
        state["V_ap_mV"][0] = -55.0;
    }
}

void init_he_state(
    std::map<std::string, std::vector<double>>& state,
    std::vector<std::int32_t>& spikes,
    bool threshold_case
) {
    state["V_mV"].resize(H_E_N);
    state["I_e_pA"].resize(H_E_N);
    state["I_i_pA"].resize(H_E_N);
    state["g_nmda_h_nS"].resize(H_E_N);
    state["I_bias_pA"].resize(H_E_N);
    state["refrac_until_ms"].resize(H_E_N);
    spikes.assign(H_E_N, 0);
    for (int i = 0; i < H_E_N; ++i) {
        state["V_mV"][i] = -69.8 + 0.001 * static_cast<double>(i % 19);
        state["I_e_pA"][i] = 0.4 + 0.01 * static_cast<double>(i % 5);
        state["I_i_pA"][i] = 0.2 + 0.01 * static_cast<double>(i % 3);
        state["g_nmda_h_nS"][i] = 0.01 * static_cast<double>(i % 7);
        state["I_bias_pA"][i] = 0.0;
        state["refrac_until_ms"][i] = -1.0;
    }
    if (threshold_case) {
        state["V_mV"][0] = -49.0;
    }
}

void init_he_quiet_state(
    std::map<std::string, std::vector<double>>& state,
    std::vector<std::int32_t>& spikes
) {
    state["V_mV"].assign(H_E_N, HE_EL_MV);
    state["I_e_pA"].assign(H_E_N, 0.0);
    state["I_i_pA"].assign(H_E_N, 0.0);
    state["g_nmda_h_nS"].assign(H_E_N, 0.0);
    state["I_bias_pA"].assign(H_E_N, 0.0);
    state["refrac_until_ms"].assign(H_E_N, -1.0);
    spikes.assign(H_E_N, 0);
}

void init_v1e_feedback_state(
    std::map<std::string, std::vector<double>>& state,
    std::vector<std::int32_t>& spikes
) {
    state["V_soma_mV"].assign(V1_E_N, -69.0);
    state["V_ap_mV"].assign(V1_E_N, V1E_EL_AP_MV);
    state["I_e_pA"].assign(V1_E_N, 0.0);
    state["I_i_pA"].assign(V1_E_N, 0.0);
    state["I_ap_e_pA"].assign(V1_E_N, 0.0);
    state["w_adapt_pA"].assign(V1_E_N, 0.0);
    state["I_bias_pA"].assign(V1_E_N, 0.0);
    state["refrac_until_ms"].assign(V1_E_N, -1.0);
    spikes.assign(V1_E_N, 0);
}

void init_v1som_feedback_state(
    std::map<std::string, std::vector<double>>& state
) {
    state["V_mV"].assign(V1_SOM_N, V1SOM_EL_MV);
    state["I_e_pA"].assign(V1_SOM_N, 0.0);
    state["I_i_pA"].assign(V1_SOM_N, 0.0);
    state["I_bias_pA"].assign(V1_SOM_N, 0.0);
    state["refrac_until_ms"].assign(V1_SOM_N, -1.0);
}

void cpu_step_v1e(std::map<std::string, std::vector<double>>& s,
                  std::vector<std::int32_t>& spikes,
                  int n_steps) {
    auto& v = s["V_soma_mV"];
    auto& vap = s["V_ap_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& iape = s["I_ap_e_pA"];
    auto& wadapt = s["w_adapt_pA"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    for (int step = 0; step < n_steps; ++step) {
        const double t_ms = step * DT_MS;
        for (int i = 0; i < V1_E_N; ++i) {
            const bool in_refrac = t_ms < refrac[i];
            const double old_v = v[i];
            const double old_vap = vap[i];
            const double old_ie = ie[i];
            const double old_ii = ii[i];
            const double old_iape = iape[i];
            const double old_w = wadapt[i];
            const double i_ap = V1E_G_AP_SOMA_NS
                * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
                * (old_v - V1E_EL_MV);
            if (!in_refrac) {
                const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
                    + ibias[i] + old_ie + i_ap - old_ii - old_w;
                v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
                    * PA_PER_NF_TO_MV_PER_MS;
            }
            vap[i] = old_vap + DT_MS
                * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
                / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
            ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
            ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
            iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
            wadapt[i] = old_w + DT_MS
                * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
                   / V1E_TAU_ADAPT_MS);
            if (!in_refrac && v[i] > V1E_VT_MV) {
                spikes[i] += 1;
                v[i] = V1E_VR_MV;
                wadapt[i] += V1E_B_ADAPT_PA;
                refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
            }
        }
    }
}

void cpu_step_he(std::map<std::string, std::vector<double>>& s,
                 std::vector<std::int32_t>& spikes,
                 int n_steps) {
    auto& v = s["V_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& gnmda = s["g_nmda_h_nS"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    for (int step = 0; step < n_steps; ++step) {
        const double t_ms = step * DT_MS;
        for (int i = 0; i < H_E_N; ++i) {
            const bool in_refrac = t_ms < refrac[i];
            const double old_v = v[i];
            const double old_ie = ie[i];
            const double old_ii = ii[i];
            const double old_g = gnmda[i];
            if (!in_refrac) {
                const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
                const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
                const double current = HE_GL_NS * (HE_EL_MV - old_v)
                    + ibias[i] + old_ie + i_nmda - old_ii;
                v[i] = old_v + DT_MS * current / HE_C_NF
                    * PA_PER_NF_TO_MV_PER_MS;
            }
            ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
            ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
            gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
            if (!in_refrac && v[i] > HE_VT_MV) {
                spikes[i] += 1;
                v[i] = HE_VR_MV;
                refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
            }
        }
    }
}

void cpu_force_count_boundary_voltage(
    std::map<std::string, std::vector<double>>& s,
    int step,
    int window_start_step,
    int window_end_step
) {
    if (window_start_step > 0 && step == window_start_step - 1) {
        s["V_mV"][COUNT_BEFORE_WINDOW_CELL] = -49.0;
    }
    if (step == window_start_step) {
        s["V_mV"][COUNT_START_WINDOW_CELL] = -49.0;
    }
    if (step == window_end_step) {
        s["V_mV"][COUNT_END_WINDOW_CELL] = -49.0;
    }
}

void cpu_he_count_step(
    std::map<std::string, std::vector<double>>& s,
    std::vector<std::int32_t>& window_counts,
    int step,
    int window_start_step,
    int window_end_step
) {
    auto& v = s["V_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& gnmda = s["g_nmda_h_nS"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    const bool in_count_window = (
        step >= window_start_step && step < window_end_step
    );
    for (int i = 0; i < H_E_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_g = gnmda[i];
        if (!in_refrac) {
            const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
            const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
            const double current = HE_GL_NS * (HE_EL_MV - old_v)
                + ibias[i] + old_ie + i_nmda - old_ii;
            v[i] = old_v + DT_MS * current / HE_C_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
        if (!in_refrac && v[i] > HE_VT_MV) {
            v[i] = HE_VR_MV;
            refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
            if (in_count_window) {
                window_counts[i] += 1;
            }
        }
    }
}

void cpu_force_v1e_count_boundary_voltage(
    std::map<std::string, std::vector<double>>& s,
    int step,
    int window_start_step,
    int window_end_step
) {
    if (window_start_step > 0 && step == window_start_step - 1) {
        s["V_soma_mV"][COUNT_BEFORE_WINDOW_CELL] = -49.0;
    }
    if (step == window_start_step) {
        s["V_soma_mV"][COUNT_START_WINDOW_CELL] = -49.0;
    }
    if (step == window_end_step) {
        s["V_soma_mV"][COUNT_END_WINDOW_CELL] = -49.0;
    }
}

void cpu_v1e_count_step(
    std::map<std::string, std::vector<double>>& s,
    std::vector<std::int32_t>& window_counts,
    int step,
    int window_start_step,
    int window_end_step
) {
    auto& v = s["V_soma_mV"];
    auto& vap = s["V_ap_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& iape = s["I_ap_e_pA"];
    auto& wadapt = s["w_adapt_pA"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    const bool in_count_window = (
        step >= window_start_step && step < window_end_step
    );
    for (int i = 0; i < V1_E_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_vap = vap[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_iape = iape[i];
        const double old_w = wadapt[i];
        const double i_ap = V1E_G_AP_SOMA_NS
            * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
            * (old_v - V1E_EL_MV);
        if (!in_refrac) {
            const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
                + ibias[i] + old_ie + i_ap - old_ii - old_w;
            v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        vap[i] = old_vap + DT_MS
            * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
            / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
        wadapt[i] = old_w + DT_MS
            * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
               / V1E_TAU_ADAPT_MS);
        if (!in_refrac && v[i] > V1E_VT_MV) {
            v[i] = V1E_VR_MV;
            wadapt[i] += V1E_B_ADAPT_PA;
            refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
            if (in_count_window) {
                window_counts[i] += 1;
            }
        }
    }
}

void cpu_v1som_step(
    std::map<std::string, std::vector<double>>& s,
    int step
) {
    auto& v = s["V_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    for (int i = 0; i < V1_SOM_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        if (!in_refrac) {
            const double current = V1SOM_GL_NS * (V1SOM_EL_MV - old_v)
                + ibias[i] + old_ie - old_ii;
            v[i] = old_v + DT_MS * current / V1SOM_C_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        if (!in_refrac && v[i] > V1SOM_VT_MV) {
            v[i] = V1SOM_VR_MV;
            refrac[i] = refractory_until_ms(step, V1SOM_REFACTORY_MS);
        }
    }
}

bool deterministic_stim_event(
    int step,
    int start_step,
    int end_step,
    int offset_steps,
    int period_steps
) {
    return step >= start_step
        && step < end_step
        && step >= start_step + offset_steps
        && ((step - start_step - offset_steps) % period_steps) == 0;
}

int count_deterministic_events(
    int start_step,
    int end_step,
    int offset_steps,
    int period_steps
) {
    int n = 0;
    for (int step = start_step; step < end_step; ++step) {
        if (deterministic_stim_event(
                step, start_step, end_step, offset_steps, period_steps
            )) {
            n += 1;
        }
    }
    return n;
}

void cpu_force_richter_boundary_v1e(
    std::map<std::string, std::vector<double>>& s,
    int step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_end_step
) {
    if (step == preprobe_start_step - 1) {
        s["V_soma_mV"][COUNT_BEFORE_WINDOW_CELL] = -49.0;
    }
    if (step == preprobe_start_step) {
        s["V_soma_mV"][COUNT_START_WINDOW_CELL] = -49.0;
    }
    if (step == preprobe_end_step) {
        s["V_soma_mV"][COUNT_END_WINDOW_CELL] = -49.0;
    }
    if (step == trailer_end_step) {
        s["V_soma_mV"][COUNT_AFTER_WINDOW_CELL] = -49.0;
    }
}

void cpu_force_richter_boundary_he(
    std::map<std::string, std::vector<double>>& s,
    int step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_end_step
) {
    if (step == preprobe_start_step - 1) {
        s["V_mV"][COUNT_BEFORE_WINDOW_CELL] = -49.0;
    }
    if (step == preprobe_start_step) {
        s["V_mV"][COUNT_START_WINDOW_CELL] = -49.0;
    }
    if (step == preprobe_end_step) {
        s["V_mV"][COUNT_END_WINDOW_CELL] = -49.0;
    }
    if (step == trailer_end_step) {
        s["V_mV"][COUNT_AFTER_WINDOW_CELL] = -49.0;
    }
}

void cpu_v1e_richter_count_step(
    std::map<std::string, std::vector<double>>& s,
    std::vector<std::int32_t>& leader_counts,
    std::vector<std::int32_t>& preprobe_counts,
    std::vector<std::int32_t>& trailer_counts,
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    auto& v = s["V_soma_mV"];
    auto& vap = s["V_ap_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& iape = s["I_ap_e_pA"];
    auto& wadapt = s["w_adapt_pA"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    for (int i = 0; i < V1_E_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_vap = vap[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_iape = iape[i];
        const double old_w = wadapt[i];
        const double i_ap = V1E_G_AP_SOMA_NS
            * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
            * (old_v - V1E_EL_MV);
        if (!in_refrac) {
            const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
                + ibias[i] + old_ie + i_ap - old_ii - old_w;
            v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        vap[i] = old_vap + DT_MS
            * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
            / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
        wadapt[i] = old_w + DT_MS
            * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
               / V1E_TAU_ADAPT_MS);
        if (!in_refrac && v[i] > V1E_VT_MV) {
            v[i] = V1E_VR_MV;
            wadapt[i] += V1E_B_ADAPT_PA;
            refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
            if (phase == 0) {
                leader_counts[i] += 1;
            } else if (phase == 1) {
                preprobe_counts[i] += 1;
            } else if (phase == 2) {
                trailer_counts[i] += 1;
            }
        }
    }
}

void cpu_he_richter_count_step(
    std::map<std::string, std::vector<double>>& s,
    std::vector<std::int32_t>& leader_counts,
    std::vector<std::int32_t>& preprobe_counts,
    std::vector<std::int32_t>& trailer_counts,
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    auto& v = s["V_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& gnmda = s["g_nmda_h_nS"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    for (int i = 0; i < H_E_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_g = gnmda[i];
        if (!in_refrac) {
            const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
            const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
            const double current = HE_GL_NS * (HE_EL_MV - old_v)
                + ibias[i] + old_ie + i_nmda - old_ii;
            v[i] = old_v + DT_MS * current / HE_C_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
        if (!in_refrac && v[i] > HE_VT_MV) {
            v[i] = HE_VR_MV;
            refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
            if (phase == 0) {
                leader_counts[i] += 1;
            } else if (phase == 1) {
                preprobe_counts[i] += 1;
            } else if (phase == 2) {
                trailer_counts[i] += 1;
            }
        }
    }
}

void cpu_v1e_richter_count_step_flags(
    std::map<std::string, std::vector<double>>& s,
    std::vector<std::int32_t>& leader_counts,
    std::vector<std::int32_t>& preprobe_counts,
    std::vector<std::int32_t>& trailer_counts,
    std::vector<std::int32_t>& step_spikes,
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    std::fill(step_spikes.begin(), step_spikes.end(), 0);
    auto& v = s["V_soma_mV"];
    auto& vap = s["V_ap_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& iape = s["I_ap_e_pA"];
    auto& wadapt = s["w_adapt_pA"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    for (int i = 0; i < V1_E_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_vap = vap[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_iape = iape[i];
        const double old_w = wadapt[i];
        const double i_ap = V1E_G_AP_SOMA_NS
            * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
            * (old_v - V1E_EL_MV);
        if (!in_refrac) {
            const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
                + ibias[i] + old_ie + i_ap - old_ii - old_w;
            v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        vap[i] = old_vap + DT_MS
            * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
            / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
        wadapt[i] = old_w + DT_MS
            * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
               / V1E_TAU_ADAPT_MS);
        if (!in_refrac && v[i] > V1E_VT_MV) {
            v[i] = V1E_VR_MV;
            wadapt[i] += V1E_B_ADAPT_PA;
            refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
            step_spikes[i] = 1;
            if (phase == 0) {
                leader_counts[i] += 1;
            } else if (phase == 1) {
                preprobe_counts[i] += 1;
            } else if (phase == 2) {
                trailer_counts[i] += 1;
            }
        }
    }
}

void cpu_he_richter_count_step_flags(
    std::map<std::string, std::vector<double>>& s,
    std::vector<std::int32_t>& leader_counts,
    std::vector<std::int32_t>& preprobe_counts,
    std::vector<std::int32_t>& trailer_counts,
    std::vector<std::int32_t>& step_spikes,
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step
) {
    std::fill(step_spikes.begin(), step_spikes.end(), 0);
    auto& v = s["V_mV"];
    auto& ie = s["I_e_pA"];
    auto& ii = s["I_i_pA"];
    auto& gnmda = s["g_nmda_h_nS"];
    auto& ibias = s["I_bias_pA"];
    auto& refrac = s["refrac_until_ms"];
    const double t_ms = step * DT_MS;
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    for (int i = 0; i < H_E_N; ++i) {
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_g = gnmda[i];
        if (!in_refrac) {
            const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
            const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
            const double current = HE_GL_NS * (HE_EL_MV - old_v)
                + ibias[i] + old_ie + i_nmda - old_ii;
            v[i] = old_v + DT_MS * current / HE_C_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
        if (!in_refrac && v[i] > HE_VT_MV) {
            v[i] = HE_VR_MV;
            refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
            step_spikes[i] = 1;
            if (phase == 0) {
                leader_counts[i] += 1;
            } else if (phase == 1) {
                preprobe_counts[i] += 1;
            } else if (phase == 2) {
                trailer_counts[i] += 1;
            }
        }
    }
}

__global__ void v1e_decay_kernel(
    int n_steps, double* v, double* vap, double* ie, double* ii,
    double* iape, double* wadapt, double* ibias, double* refrac,
    std::int32_t* spikes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V1_E_N) return;
    for (int step = 0; step < n_steps; ++step) {
        const double t_ms = step * DT_MS;
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_vap = vap[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_iape = iape[i];
        const double old_w = wadapt[i];
        const double i_ap = V1E_G_AP_SOMA_NS
            * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
            * (old_v - V1E_EL_MV);
        if (!in_refrac) {
            const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
                + ibias[i] + old_ie + i_ap - old_ii - old_w;
            v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        vap[i] = old_vap + DT_MS
            * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
            / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
        wadapt[i] = old_w + DT_MS
            * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
               / V1E_TAU_ADAPT_MS);
        if (!in_refrac && v[i] > V1E_VT_MV) {
            spikes[i] += 1;
            v[i] = V1E_VR_MV;
            wadapt[i] += V1E_B_ADAPT_PA;
            refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
        }
    }
}

__global__ void he_decay_kernel(
    int n_steps, double* v, double* ie, double* ii, double* gnmda,
    double* ibias, double* refrac, std::int32_t* spikes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_E_N) return;
    for (int step = 0; step < n_steps; ++step) {
        const double t_ms = step * DT_MS;
        const bool in_refrac = t_ms < refrac[i];
        const double old_v = v[i];
        const double old_ie = ie[i];
        const double old_ii = ii[i];
        const double old_g = gnmda[i];
        if (!in_refrac) {
            const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
            const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
            const double current = HE_GL_NS * (HE_EL_MV - old_v)
                + ibias[i] + old_ie + i_nmda - old_ii;
            v[i] = old_v + DT_MS * current / HE_C_NF
                * PA_PER_NF_TO_MV_PER_MS;
        }
        ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
        if (!in_refrac && v[i] > HE_VT_MV) {
            spikes[i] += 1;
            v[i] = HE_VR_MV;
            refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
        }
    }
}

__global__ void csr_scatter_one_source_kernel(
    std::int32_t pre_index,
    const std::int32_t* row_ptr,
    const std::int32_t* post,
    const double* weight,
    double drive_amp,
    double* target
) {
    const int start = row_ptr[pre_index];
    const int end = row_ptr[pre_index + 1];
    for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
        atomicAdd(&target[post[edge]], weight[edge] * drive_amp);
    }
}

__global__ void force_count_boundary_voltage_kernel(
    int step,
    int window_start_step,
    int window_end_step,
    double* v
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (window_start_step > 0 && step == window_start_step - 1) {
        v[COUNT_BEFORE_WINDOW_CELL] = -49.0;
    }
    if (step == window_start_step) {
        v[COUNT_START_WINDOW_CELL] = -49.0;
    }
    if (step == window_end_step) {
        v[COUNT_END_WINDOW_CELL] = -49.0;
    }
}

__global__ void force_single_voltage_kernel(
    int index,
    double forced_voltage_mV,
    double* v
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    v[index] = forced_voltage_mV;
}

__global__ void he_count_step_kernel(
    int step,
    int window_start_step,
    int window_end_step,
    double* v,
    double* ie,
    double* ii,
    double* gnmda,
    double* ibias,
    double* refrac,
    std::int32_t* window_counts
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_E_N) return;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const bool in_count_window = (
        step >= window_start_step && step < window_end_step
    );
    const double old_v = v[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_g = gnmda[i];
    if (!in_refrac) {
        const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
        const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
        const double current = HE_GL_NS * (HE_EL_MV - old_v)
            + ibias[i] + old_ie + i_nmda - old_ii;
        v[i] = old_v + DT_MS * current / HE_C_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
    if (!in_refrac && v[i] > HE_VT_MV) {
        v[i] = HE_VR_MV;
        refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
        if (in_count_window) {
            window_counts[i] += 1;
        }
    }
}

__global__ void v1e_count_step_kernel(
    int step,
    int window_start_step,
    int window_end_step,
    double* v,
    double* vap,
    double* ie,
    double* ii,
    double* iape,
    double* wadapt,
    double* ibias,
    double* refrac,
    std::int32_t* window_counts
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V1_E_N) return;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const bool in_count_window = (
        step >= window_start_step && step < window_end_step
    );
    const double old_v = v[i];
    const double old_vap = vap[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_iape = iape[i];
    const double old_w = wadapt[i];
    const double i_ap = V1E_G_AP_SOMA_NS
        * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
        * (old_v - V1E_EL_MV);
    if (!in_refrac) {
        const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
            + ibias[i] + old_ie + i_ap - old_ii - old_w;
        v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    vap[i] = old_vap + DT_MS
        * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
        / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
    wadapt[i] = old_w + DT_MS
        * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
           / V1E_TAU_ADAPT_MS);
    if (!in_refrac && v[i] > V1E_VT_MV) {
        v[i] = V1E_VR_MV;
        wadapt[i] += V1E_B_ADAPT_PA;
        refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
        if (in_count_window) {
            window_counts[i] += 1;
        }
    }
}

__global__ void v1e_richter_count_step_kernel(
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step,
    double* v,
    double* vap,
    double* ie,
    double* ii,
    double* iape,
    double* wadapt,
    double* ibias,
    double* refrac,
    std::int32_t* leader_counts,
    std::int32_t* preprobe_counts,
    std::int32_t* trailer_counts
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V1_E_N) return;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const double old_v = v[i];
    const double old_vap = vap[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_iape = iape[i];
    const double old_w = wadapt[i];
    const double i_ap = V1E_G_AP_SOMA_NS
        * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
        * (old_v - V1E_EL_MV);
    if (!in_refrac) {
        const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
            + ibias[i] + old_ie + i_ap - old_ii - old_w;
        v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    vap[i] = old_vap + DT_MS
        * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
        / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
    wadapt[i] = old_w + DT_MS
        * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
           / V1E_TAU_ADAPT_MS);
    if (!in_refrac && v[i] > V1E_VT_MV) {
        v[i] = V1E_VR_MV;
        wadapt[i] += V1E_B_ADAPT_PA;
        refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
        if (phase == 0) {
            leader_counts[i] += 1;
        } else if (phase == 1) {
            preprobe_counts[i] += 1;
        } else if (phase == 2) {
            trailer_counts[i] += 1;
        }
    }
}

__global__ void he_richter_count_step_kernel(
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step,
    double* v,
    double* ie,
    double* ii,
    double* gnmda,
    double* ibias,
    double* refrac,
    std::int32_t* leader_counts,
    std::int32_t* preprobe_counts,
    std::int32_t* trailer_counts
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_E_N) return;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const double old_v = v[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_g = gnmda[i];
    if (!in_refrac) {
        const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
        const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
        const double current = HE_GL_NS * (HE_EL_MV - old_v)
            + ibias[i] + old_ie + i_nmda - old_ii;
        v[i] = old_v + DT_MS * current / HE_C_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
    if (!in_refrac && v[i] > HE_VT_MV) {
        v[i] = HE_VR_MV;
        refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
        if (phase == 0) {
            leader_counts[i] += 1;
        } else if (phase == 1) {
            preprobe_counts[i] += 1;
        } else if (phase == 2) {
            trailer_counts[i] += 1;
        }
    }
}

__global__ void v1e_richter_count_step_flags_kernel(
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step,
    double* v,
    double* vap,
    double* ie,
    double* ii,
    double* iape,
    double* wadapt,
    double* ibias,
    double* refrac,
    std::int32_t* leader_counts,
    std::int32_t* preprobe_counts,
    std::int32_t* trailer_counts,
    std::int32_t* step_spikes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V1_E_N) return;
    step_spikes[i] = 0;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const double old_v = v[i];
    const double old_vap = vap[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_iape = iape[i];
    const double old_w = wadapt[i];
    const double i_ap = V1E_G_AP_SOMA_NS
        * sigmoid((old_vap - V1E_V_AP_TH_MV) / V1E_V_AP_SLOPE_MV)
        * (old_v - V1E_EL_MV);
    if (!in_refrac) {
        const double current = V1E_GL_SOMA_NS * (V1E_EL_MV - old_v)
            + ibias[i] + old_ie + i_ap - old_ii - old_w;
        v[i] = old_v + DT_MS * current / V1E_C_SOMA_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    vap[i] = old_vap + DT_MS
        * (V1E_GL_AP_NS * (V1E_EL_AP_MV - old_vap) + old_iape)
        / V1E_C_AP_NF * PA_PER_NF_TO_MV_PER_MS;
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    iape[i] = old_iape + DT_MS * (-old_iape / TAU_E_MS);
    wadapt[i] = old_w + DT_MS
        * ((V1E_A_ADAPT_NS * (old_v - V1E_EL_MV) - old_w)
           / V1E_TAU_ADAPT_MS);
    if (!in_refrac && v[i] > V1E_VT_MV) {
        v[i] = V1E_VR_MV;
        wadapt[i] += V1E_B_ADAPT_PA;
        refrac[i] = refractory_until_ms(step, V1E_REFACTORY_MS);
        step_spikes[i] = 1;
        if (phase == 0) {
            leader_counts[i] += 1;
        } else if (phase == 1) {
            preprobe_counts[i] += 1;
        } else if (phase == 2) {
            trailer_counts[i] += 1;
        }
    }
}

__global__ void he_richter_count_step_flags_kernel(
    int step,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step,
    double* v,
    double* ie,
    double* ii,
    double* gnmda,
    double* ibias,
    double* refrac,
    std::int32_t* leader_counts,
    std::int32_t* preprobe_counts,
    std::int32_t* trailer_counts,
    std::int32_t* step_spikes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_E_N) return;
    step_spikes[i] = 0;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const double old_v = v[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_g = gnmda[i];
    if (!in_refrac) {
        const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
        const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
        const double current = HE_GL_NS * (HE_EL_MV - old_v)
            + ibias[i] + old_ie + i_nmda - old_ii;
        v[i] = old_v + DT_MS * current / HE_C_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
    if (!in_refrac && v[i] > HE_VT_MV) {
        v[i] = HE_VR_MV;
        refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
        step_spikes[i] = 1;
        if (phase == 0) {
            leader_counts[i] += 1;
        } else if (phase == 1) {
            preprobe_counts[i] += 1;
        } else if (phase == 2) {
            trailer_counts[i] += 1;
        }
    }
}

__global__ void h_ring_external_pulse_kernel(
    int channel,
    double drive_pA,
    double* e_ie
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_E_PER_CHANNEL) return;
    e_ie[channel * H_E_PER_CHANNEL + i] += drive_pA;
}

__global__ void h_ring_e_step_kernel(
    int step,
    int leader_start,
    int leader_end,
    int persistence_start,
    int persistence_end,
    int late_start,
    int late_end,
    double* v,
    double* ie,
    double* ii,
    double* gnmda,
    double* refrac,
    std::int32_t* leader_counts,
    std::int32_t* persistence_counts,
    std::int32_t* late_counts,
    std::int32_t* total_counts,
    std::int32_t* step_spikes,
    std::int32_t* last_persistence_step
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_E_N) return;
    step_spikes[i] = 0;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const double old_v = v[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    const double old_g = gnmda[i];
    if (!in_refrac) {
        const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
        const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
        const double current = HE_GL_NS * (HE_EL_MV - old_v)
            + old_ie + i_nmda - old_ii;
        v[i] = old_v + DT_MS * current / HE_C_NF * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    gnmda[i] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
    if (!in_refrac && v[i] > HE_VT_MV) {
        step_spikes[i] = 1;
        v[i] = HE_VR_MV;
        refrac[i] = refractory_until_ms(step, HE_REFACTORY_MS);
        total_counts[i] += 1;
        if (step >= leader_start && step < leader_end) {
            leader_counts[i] += 1;
        }
        if (step >= persistence_start && step < persistence_end) {
            persistence_counts[i] += 1;
            atomicMax(last_persistence_step, step);
        }
        if (step >= late_start && step < late_end) {
            late_counts[i] += 1;
        }
    }
}

__global__ void h_ring_inh_step_kernel(
    int step,
    double* v,
    double* ie,
    double* ii,
    double* refrac,
    std::int32_t* total_counts,
    std::int32_t* step_spikes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_INH_N) return;
    step_spikes[i] = 0;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const double old_v = v[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    if (!in_refrac) {
        const double current = HI_GL_NS * (HI_EL_MV - old_v) + old_ie - old_ii;
        v[i] = old_v + DT_MS * current / HI_C_NF * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    if (!in_refrac && v[i] > HI_VT_MV) {
        step_spikes[i] = 1;
        v[i] = HI_VR_MV;
        refrac[i] = refractory_until_ms(step, HI_REFACTORY_MS);
        total_counts[i] += 1;
    }
}

__global__ void h_ring_e_recurrent_scatter_kernel(
    const std::int32_t* e_spikes,
    double* e_ie,
    double* e_gnmda,
    double* inh_ie
) {
    const int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= H_E_N || e_spikes[src] == 0) return;
    const int channel = src / H_E_PER_CHANNEL;
    const int start = channel * H_E_PER_CHANNEL;
    for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
        const int post = start + k;
        if (post == src) continue;
        atomicAdd(&e_ie[post], H_RING_EE_AMPA_PA);
        atomicAdd(&e_gnmda[post], H_RING_EE_NMDA_NS);
    }
    atomicAdd(&inh_ie[channel], H_RING_EI_LOCAL_PA);
    atomicAdd(&inh_ie[H_BROAD_INH_START + (src % 4)], H_RING_EI_BROAD_PA);
}

__global__ void h_ring_inh_scatter_kernel(
    const std::int32_t* inh_spikes,
    double* e_ii
) {
    const int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= H_INH_N || inh_spikes[src] == 0) return;
    if (src < H_BROAD_INH_START) {
        const int start = src * H_E_PER_CHANNEL;
        for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
            atomicAdd(&e_ii[start + k], H_RING_IE_LOCAL_PA);
        }
    } else {
        for (int post = 0; post < H_E_N; ++post) {
            atomicAdd(&e_ii[post], H_RING_IE_BROAD_PA);
        }
    }
}

__global__ void h_ring_external_pulse_batch_kernel(
    int n_trials,
    const std::int32_t* channels,
    double drive_pA,
    double* e_ie
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_trials * H_E_PER_CHANNEL;
    if (idx >= n) return;
    const int trial = idx / H_E_PER_CHANNEL;
    const int offset = idx % H_E_PER_CHANNEL;
    const int channel = channels[trial];
    if (channel < 0 || channel >= H_BROAD_INH_START) return;
    e_ie[trial * H_E_N + channel * H_E_PER_CHANNEL + offset] += drive_pA;
}

__global__ void h_ring_e_step_batch_kernel(
    int n_trials,
    int step,
    int leader_start,
    int leader_end,
    int mid_start,
    int mid_end,
    int late_start,
    int late_end,
    double* v,
    double* ie,
    double* ii,
    double* gnmda,
    double* refrac,
    std::int32_t* leader_counts,
    std::int32_t* mid_counts,
    std::int32_t* late_counts,
    std::int32_t* total_counts,
    std::int32_t* step_spikes,
    std::int32_t* last_mid_step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_trials * H_E_N;
    if (idx >= n) return;
    const int trial = idx / H_E_N;
    step_spikes[idx] = 0;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[idx];
    const double old_v = v[idx];
    const double old_ie = ie[idx];
    const double old_ii = ii[idx];
    const double old_g = gnmda[idx];
    if (!in_refrac) {
        const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
        const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
        const double current = HE_GL_NS * (HE_EL_MV - old_v)
            + old_ie + i_nmda - old_ii;
        v[idx] = old_v + DT_MS * current / HE_C_NF * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[idx] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[idx] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    gnmda[idx] = old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
    if (!in_refrac && v[idx] > HE_VT_MV) {
        step_spikes[idx] = 1;
        v[idx] = HE_VR_MV;
        refrac[idx] = refractory_until_ms(step, HE_REFACTORY_MS);
        total_counts[idx] += 1;
        if (step >= leader_start && step < leader_end) {
            leader_counts[idx] += 1;
        }
        if (step >= mid_start && step < mid_end) {
            mid_counts[idx] += 1;
            atomicMax(&last_mid_step[trial], step);
        }
        if (step >= late_start && step < late_end) {
            late_counts[idx] += 1;
        }
    }
}

__global__ void h_ring_inh_step_batch_kernel(
    int n_trials,
    int step,
    double* v,
    double* ie,
    double* ii,
    double* refrac,
    std::int32_t* total_counts,
    std::int32_t* step_spikes
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_trials * H_INH_N;
    if (idx >= n) return;
    step_spikes[idx] = 0;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[idx];
    const double old_v = v[idx];
    const double old_ie = ie[idx];
    const double old_ii = ii[idx];
    if (!in_refrac) {
        const double current = HI_GL_NS * (HI_EL_MV - old_v) + old_ie - old_ii;
        v[idx] = old_v + DT_MS * current / HI_C_NF * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[idx] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[idx] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    if (!in_refrac && v[idx] > HI_VT_MV) {
        step_spikes[idx] = 1;
        v[idx] = HI_VR_MV;
        refrac[idx] = refractory_until_ms(step, HI_REFACTORY_MS);
        total_counts[idx] += 1;
    }
}

__global__ void h_ring_e_recurrent_scatter_batch_kernel(
    int n_trials,
    const std::int32_t* e_spikes,
    double* e_ie,
    double* e_gnmda,
    double* inh_ie
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_trials * H_E_N;
    if (idx >= n || e_spikes[idx] == 0) return;
    const int trial = idx / H_E_N;
    const int src = idx % H_E_N;
    const int e_base = trial * H_E_N;
    const int inh_base = trial * H_INH_N;
    const int channel = src / H_E_PER_CHANNEL;
    const int start = channel * H_E_PER_CHANNEL;
    for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
        const int post = start + k;
        if (post == src) continue;
        atomicAdd(&e_ie[e_base + post], H_RING_EE_AMPA_PA);
        atomicAdd(&e_gnmda[e_base + post], H_RING_EE_NMDA_NS);
    }
    atomicAdd(&inh_ie[inh_base + channel], H_RING_EI_LOCAL_PA);
    atomicAdd(&inh_ie[inh_base + H_BROAD_INH_START + (src % 4)], H_RING_EI_BROAD_PA);
}

__global__ void h_ring_inh_scatter_batch_kernel(
    int n_trials,
    const std::int32_t* inh_spikes,
    double* e_ii
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_trials * H_INH_N;
    if (idx >= n || inh_spikes[idx] == 0) return;
    const int trial = idx / H_INH_N;
    const int src = idx % H_INH_N;
    const int e_base = trial * H_E_N;
    if (src < H_BROAD_INH_START) {
        const int start = src * H_E_PER_CHANNEL;
        for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
            atomicAdd(&e_ii[e_base + start + k], H_RING_IE_LOCAL_PA);
        }
    } else {
        for (int post = 0; post < H_E_N; ++post) {
            atomicAdd(&e_ii[e_base + post], H_RING_IE_BROAD_PA);
        }
    }
}

__global__ void h_ctx_pred_dense_scatter_batch_kernel(
    int n_trials,
    const std::int32_t* ctx_spikes,
    const double* w_ctx_pred,
    double drive_pA,
    double* pred_e_ie
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_trials * H_E_N;
    if (idx >= n || ctx_spikes[idx] == 0) return;
    const int trial = idx / H_E_N;
    const int src = idx % H_E_N;
    const int pred_base = trial * H_E_N;
    const int w_base = src * H_E_N;
    for (int post = 0; post < H_E_N; ++post) {
        const double current = w_ctx_pred[w_base + post] * drive_pA;
        if (current != 0.0) {
            atomicAdd(&pred_e_ie[pred_base + post], current);
        }
    }
}

__global__ void seeded_stim_source_scatter_kernel(
    std::int64_t seed,
    int step,
    int n_sources,
    const std::int32_t* source_channel,
    int expected_channel,
    int unexpected_channel,
    double grating_rate_hz,
    double baseline_rate_hz,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step,
    int iti_start_step,
    int iti_end_step,
    const std::int32_t* row_ptr,
    const std::int32_t* post,
    const double* weight,
    double drive_amp,
    double* target,
    std::int32_t* source_events_by_step,
    std::int32_t* source_events_by_afferent,
    std::int32_t* source_events_by_channel,
    std::int32_t* source_phase_counts
) {
    const int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_sources) return;
    const int channel = source_channel[src];
    if (!seeded_source_event(
            seed, step, src, channel, expected_channel, unexpected_channel,
            grating_rate_hz, baseline_rate_hz, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        )) {
        return;
    }
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const int phase_index =
        phase >= 0 ? phase : ((step >= iti_start_step && step < iti_end_step) ? 3 : 4);
    atomicAdd(&source_events_by_step[step], 1);
    atomicAdd(&source_events_by_afferent[src], 1);
    if (channel >= 0) {
        atomicAdd(&source_events_by_channel[channel], 1);
    }
    atomicAdd(&source_phase_counts[phase_index], 1);
    atomicAdd(&source_phase_counts[4], 1);
    for (int edge = row_ptr[src]; edge < row_ptr[src + 1]; ++edge) {
        atomicAdd(&target[post[edge]], weight[edge] * drive_amp);
    }
}

__global__ void controlled_stim_source_scatter_kernel(
    int step,
    int n_events,
    const std::int32_t* event_steps,
    const std::int32_t* event_sources,
    int n_sources,
    const std::int32_t* source_channel,
    int leader_start_step,
    int leader_end_step,
    int preprobe_start_step,
    int preprobe_end_step,
    int trailer_start_step,
    int trailer_end_step,
    int iti_start_step,
    int iti_end_step,
    const std::int32_t* row_ptr,
    const std::int32_t* post,
    const double* weight,
    double drive_amp,
    double* target,
    std::int32_t* source_events_by_step,
    std::int32_t* source_events_by_afferent,
    std::int32_t* source_events_by_channel,
    std::int32_t* source_phase_counts
) {
    const int event = blockIdx.x * blockDim.x + threadIdx.x;
    if (event >= n_events || event_steps[event] != step) return;
    const int src = event_sources[event];
    if (src < 0 || src >= n_sources) return;
    const int channel = source_channel[src];
    const int phase = richter_phase_index(
        step, leader_start_step, leader_end_step, preprobe_start_step,
        preprobe_end_step, trailer_start_step, trailer_end_step
    );
    const int phase_index =
        phase >= 0 ? phase : ((step >= iti_start_step && step < iti_end_step) ? 3 : 4);
    atomicAdd(&source_events_by_step[step], 1);
    atomicAdd(&source_events_by_afferent[src], 1);
    if (channel >= 0) {
        atomicAdd(&source_events_by_channel[channel], 1);
    }
    atomicAdd(&source_phase_counts[phase_index], 1);
    atomicAdd(&source_phase_counts[4], 1);
    for (int edge = row_ptr[src]; edge < row_ptr[src + 1]; ++edge) {
        atomicAdd(&target[post[edge]], weight[edge] * drive_amp);
    }
}

__global__ void csr_scatter_spike_flags_kernel(
    int n_pre,
    const std::int32_t* spike_flags,
    const std::int32_t* row_ptr,
    const std::int32_t* post,
    const double* weight,
    double drive_amp,
    double* target
) {
    const int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_pre || spike_flags[src] == 0) return;
    for (int edge = row_ptr[src]; edge < row_ptr[src + 1]; ++edge) {
        atomicAdd(&target[post[edge]], weight[edge] * drive_amp);
    }
}

__global__ void v1som_step_kernel(
    int step,
    double* v,
    double* ie,
    double* ii,
    double* ibias,
    double* refrac
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V1_SOM_N) return;
    const double t_ms = step * DT_MS;
    const bool in_refrac = t_ms < refrac[i];
    const double old_v = v[i];
    const double old_ie = ie[i];
    const double old_ii = ii[i];
    if (!in_refrac) {
        const double current = V1SOM_GL_NS * (V1SOM_EL_MV - old_v)
            + ibias[i] + old_ie - old_ii;
        v[i] = old_v + DT_MS * current / V1SOM_C_NF
            * PA_PER_NF_TO_MV_PER_MS;
    }
    ie[i] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
    ii[i] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
    if (!in_refrac && v[i] > V1SOM_VT_MV) {
        v[i] = V1SOM_VR_MV;
        refrac[i] = refractory_until_ms(step, V1SOM_REFACTORY_MS);
    }
}

std::map<std::string, double> compare_state(
    const std::map<std::string, std::vector<double>>& cpu,
    const std::map<std::string, std::vector<double>>& cuda
) {
    std::map<std::string, double> out;
    for (const auto& [key, cpu_values] : cpu) {
        out[key] = max_abs_diff(cpu_values, cuda.at(key));
    }
    return out;
}

void append_prefixed_state(
    std::map<std::string, std::vector<double>>& out,
    const std::string& prefix,
    const std::map<std::string, std::vector<double>>& state
) {
    for (const auto& [key, values] : state) {
        out[prefix + "." + key] = values;
    }
}

struct CsrBank {
    std::string name;
    std::vector<std::int32_t> row_ptr;
    std::vector<std::int32_t> post;
    std::vector<double> weight;
    std::int32_t n_pre;
    std::int32_t n_target;
};

struct HNativeRingState {
    std::vector<double> e_v;
    std::vector<double> e_ie;
    std::vector<double> e_ii;
    std::vector<double> e_gnmda;
    std::vector<double> e_refrac;
    std::vector<double> inh_v;
    std::vector<double> inh_ie;
    std::vector<double> inh_ii;
    std::vector<double> inh_refrac;
    std::vector<std::int32_t> e_spikes;
    std::vector<std::int32_t> inh_spikes;
};

int h_channel_for_e(int index) {
    return index / H_E_PER_CHANNEL;
}

int h_channel_start(int channel) {
    return channel * H_E_PER_CHANNEL;
}

void init_h_native_ring_state(HNativeRingState& s) {
    s.e_v.assign(H_E_N, HE_EL_MV);
    s.e_ie.assign(H_E_N, 0.0);
    s.e_ii.assign(H_E_N, 0.0);
    s.e_gnmda.assign(H_E_N, 0.0);
    s.e_refrac.assign(H_E_N, -1.0);
    s.inh_v.assign(H_INH_N, HI_EL_MV);
    s.inh_ie.assign(H_INH_N, 0.0);
    s.inh_ii.assign(H_INH_N, 0.0);
    s.inh_refrac.assign(H_INH_N, -1.0);
    s.e_spikes.assign(H_E_N, 0);
    s.inh_spikes.assign(H_INH_N, 0);
}

void append_h_native_state(
    std::map<std::string, std::vector<double>>& out,
    const std::string& prefix,
    const HNativeRingState& s
) {
    out[prefix + ".e_v_mV"] = s.e_v;
    out[prefix + ".e_I_e_pA"] = s.e_ie;
    out[prefix + ".e_I_i_pA"] = s.e_ii;
    out[prefix + ".e_g_nmda_nS"] = s.e_gnmda;
    out[prefix + ".inh_v_mV"] = s.inh_v;
    out[prefix + ".inh_I_e_pA"] = s.inh_ie;
}

void cpu_h_ring_external_pulse(
    HNativeRingState& s,
    int channel,
    double drive_pA
) {
    const int start = h_channel_start(channel);
    for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
        s.e_ie[static_cast<std::size_t>(start + k)] += drive_pA;
    }
}

void cpu_h_ring_e_step(
    HNativeRingState& s,
    int step,
    int leader_start,
    int leader_end,
    int persistence_start,
    int persistence_end,
    int late_start,
    int late_end,
    std::vector<std::int32_t>& leader_counts,
    std::vector<std::int32_t>& persistence_counts,
    std::vector<std::int32_t>& late_counts,
    std::vector<std::int32_t>& total_counts,
    int* last_persistence_step
) {
    std::fill(s.e_spikes.begin(), s.e_spikes.end(), 0);
    const double t_ms = step * DT_MS;
    for (int i = 0; i < H_E_N; ++i) {
        const bool in_refrac = t_ms < s.e_refrac[static_cast<std::size_t>(i)];
        const double old_v = s.e_v[static_cast<std::size_t>(i)];
        const double old_ie = s.e_ie[static_cast<std::size_t>(i)];
        const double old_ii = s.e_ii[static_cast<std::size_t>(i)];
        const double old_g = s.e_gnmda[static_cast<std::size_t>(i)];
        if (!in_refrac) {
            const double s_nmda = 1.0 / (1.0 + exp(-0.062 * old_v) / 3.57);
            const double i_nmda = old_g * s_nmda * (V_NMDA_REV_MV - old_v);
            const double current = HE_GL_NS * (HE_EL_MV - old_v)
                + old_ie + i_nmda - old_ii;
            s.e_v[static_cast<std::size_t>(i)] =
                old_v + DT_MS * current / HE_C_NF * PA_PER_NF_TO_MV_PER_MS;
        }
        s.e_ie[static_cast<std::size_t>(i)] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        s.e_ii[static_cast<std::size_t>(i)] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        s.e_gnmda[static_cast<std::size_t>(i)] =
            old_g + DT_MS * (-old_g / TAU_NMDA_H_MS);
        if (!in_refrac && s.e_v[static_cast<std::size_t>(i)] > HE_VT_MV) {
            s.e_spikes[static_cast<std::size_t>(i)] = 1;
            s.e_v[static_cast<std::size_t>(i)] = HE_VR_MV;
            s.e_refrac[static_cast<std::size_t>(i)] =
                refractory_until_ms(step, HE_REFACTORY_MS);
            total_counts[static_cast<std::size_t>(i)] += 1;
            if (step >= leader_start && step < leader_end) {
                leader_counts[static_cast<std::size_t>(i)] += 1;
            }
            if (step >= persistence_start && step < persistence_end) {
                persistence_counts[static_cast<std::size_t>(i)] += 1;
                *last_persistence_step = std::max(*last_persistence_step, step);
            }
            if (step >= late_start && step < late_end) {
                late_counts[static_cast<std::size_t>(i)] += 1;
            }
        }
    }
}

void cpu_h_ring_inh_step(
    HNativeRingState& s,
    int step,
    std::vector<std::int32_t>& total_counts
) {
    std::fill(s.inh_spikes.begin(), s.inh_spikes.end(), 0);
    const double t_ms = step * DT_MS;
    for (int i = 0; i < H_INH_N; ++i) {
        const bool in_refrac = t_ms < s.inh_refrac[static_cast<std::size_t>(i)];
        const double old_v = s.inh_v[static_cast<std::size_t>(i)];
        const double old_ie = s.inh_ie[static_cast<std::size_t>(i)];
        const double old_ii = s.inh_ii[static_cast<std::size_t>(i)];
        if (!in_refrac) {
            const double current = HI_GL_NS * (HI_EL_MV - old_v) + old_ie - old_ii;
            s.inh_v[static_cast<std::size_t>(i)] =
                old_v + DT_MS * current / HI_C_NF * PA_PER_NF_TO_MV_PER_MS;
        }
        s.inh_ie[static_cast<std::size_t>(i)] = old_ie + DT_MS * (-old_ie / TAU_E_MS);
        s.inh_ii[static_cast<std::size_t>(i)] = old_ii + DT_MS * (-old_ii / TAU_I_MS);
        if (!in_refrac && s.inh_v[static_cast<std::size_t>(i)] > HI_VT_MV) {
            s.inh_spikes[static_cast<std::size_t>(i)] = 1;
            s.inh_v[static_cast<std::size_t>(i)] = HI_VR_MV;
            s.inh_refrac[static_cast<std::size_t>(i)] =
                refractory_until_ms(step, HI_REFACTORY_MS);
            total_counts[static_cast<std::size_t>(i)] += 1;
        }
    }
}

void cpu_h_ring_scatter_spikes(HNativeRingState& s) {
    for (int src = 0; src < H_E_N; ++src) {
        if (s.e_spikes[static_cast<std::size_t>(src)] == 0) {
            continue;
        }
        const int channel = h_channel_for_e(src);
        const int start = h_channel_start(channel);
        for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
            const int post = start + k;
            if (post == src) {
                continue;
            }
            s.e_ie[static_cast<std::size_t>(post)] += H_RING_EE_AMPA_PA;
            s.e_gnmda[static_cast<std::size_t>(post)] += H_RING_EE_NMDA_NS;
        }
        s.inh_ie[static_cast<std::size_t>(channel)] += H_RING_EI_LOCAL_PA;
        s.inh_ie[static_cast<std::size_t>(H_BROAD_INH_START + (src % 4))] +=
            H_RING_EI_BROAD_PA;
    }
    for (int src = 0; src < H_INH_N; ++src) {
        if (s.inh_spikes[static_cast<std::size_t>(src)] == 0) {
            continue;
        }
        if (src < H_BROAD_INH_START) {
            const int start = h_channel_start(src);
            for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
                s.e_ii[static_cast<std::size_t>(start + k)] += H_RING_IE_LOCAL_PA;
            }
        } else {
            for (int post = 0; post < H_E_N; ++post) {
                s.e_ii[static_cast<std::size_t>(post)] += H_RING_IE_BROAD_PA;
            }
        }
    }
}

void cpu_h_ctx_pred_dense_scatter(
    const HNativeRingState& ctx,
    const std::vector<double>& w_ctx_pred,
    HNativeRingState& pred
) {
    if (w_ctx_pred.size() != static_cast<std::size_t>(H_E_N * H_E_N)) {
        throw std::runtime_error("W_ctx_pred must have shape 192x192");
    }
    for (int src = 0; src < H_E_N; ++src) {
        if (ctx.e_spikes[static_cast<std::size_t>(src)] == 0) {
            continue;
        }
        const std::size_t row = static_cast<std::size_t>(src) * H_E_N;
        for (int post = 0; post < H_E_N; ++post) {
            pred.e_ie[static_cast<std::size_t>(post)] +=
                w_ctx_pred[row + static_cast<std::size_t>(post)]
                * CTX_PRED_GATE_DRIVE_PA;
        }
    }
}

CsrBank build_csr_bank(
    const std::string& name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight
) {
    if (pre.empty()) {
        throw std::runtime_error("CSR bank requires nonempty edge arrays: " + name);
    }
    if (pre.size() != post.size() || pre.size() != weight.size()) {
        throw std::runtime_error("CSR bank shape mismatch: " + name);
    }
    const int n_pre = static_cast<int>(*std::max_element(pre.begin(), pre.end())) + 1;
    const int n_target = static_cast<int>(*std::max_element(post.begin(), post.end())) + 1;
    CsrBank bank;
    bank.name = name;
    bank.n_pre = n_pre;
    bank.n_target = n_target;
    bank.row_ptr.assign(static_cast<std::size_t>(n_pre) + 1, 0);
    for (const auto src : pre) {
        if (src < 0) {
            throw std::runtime_error("CSR bank negative pre index: " + name);
        }
        bank.row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < n_pre; ++i) {
        bank.row_ptr[static_cast<std::size_t>(i) + 1] += bank.row_ptr[i];
    }
    std::vector<std::int32_t> fill = bank.row_ptr;
    bank.post.resize(post.size());
    bank.weight.resize(weight.size());
    for (std::size_t edge = 0; edge < pre.size(); ++edge) {
        const int src = pre[edge];
        const int dst_slot = fill[static_cast<std::size_t>(src)]++;
        bank.post[static_cast<std::size_t>(dst_slot)] = post[edge];
        bank.weight[static_cast<std::size_t>(dst_slot)] = weight[edge];
    }
    return bank;
}

std::int32_t csr_fanout(const CsrBank& bank, std::int32_t pre_index) {
    if (pre_index < 0 || pre_index >= bank.n_pre) {
        throw std::runtime_error("CSR pre index out of range: " + bank.name);
    }
    return bank.row_ptr[static_cast<std::size_t>(pre_index) + 1]
        - bank.row_ptr[static_cast<std::size_t>(pre_index)];
}

double csr_sum_to_target(
    const CsrBank& bank,
    std::int32_t pre_index,
    std::int32_t target_index,
    double drive_amp
) {
    double total = 0.0;
    for (int edge = bank.row_ptr[pre_index]; edge < bank.row_ptr[pre_index + 1]; ++edge) {
        if (bank.post[static_cast<std::size_t>(edge)] == target_index) {
            total += bank.weight[static_cast<std::size_t>(edge)] * drive_amp;
        }
    }
    return total;
}

void cpu_scatter_one_source(
    const CsrBank& bank,
    std::int32_t pre_index,
    double drive_amp,
    std::vector<double>& target
) {
    for (int edge = bank.row_ptr[pre_index]; edge < bank.row_ptr[pre_index + 1]; ++edge) {
        target[static_cast<std::size_t>(bank.post[static_cast<std::size_t>(edge)])] +=
            bank.weight[static_cast<std::size_t>(edge)] * drive_amp;
    }
}

void cpu_scatter_spike_flags(
    const CsrBank& bank,
    const std::vector<std::int32_t>& spike_flags,
    double drive_amp,
    std::vector<double>& target
) {
    const int n_pre = std::min<int>(
        bank.n_pre, static_cast<int>(spike_flags.size())
    );
    for (int src = 0; src < n_pre; ++src) {
        if (spike_flags[static_cast<std::size_t>(src)] != 0) {
            cpu_scatter_one_source(bank, src, drive_amp, target);
        }
    }
}

std::int32_t sum_counts(const std::vector<std::int32_t>& counts) {
    return std::accumulate(counts.begin(), counts.end(), 0);
}

double max_count_abs_diff(
    const std::vector<std::int32_t>& cpu,
    const std::vector<std::int32_t>& cuda
) {
    if (cpu.size() != cuda.size()) {
        throw std::runtime_error("count vector size mismatch");
    }
    double out = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        out = std::max(
            out,
            static_cast<double>(std::abs(cpu[i] - cuda[i]))
        );
    }
    return out;
}

std::vector<double> counts_to_rate_hz(
    const std::vector<std::int32_t>& counts,
    int start_step,
    int end_step
) {
    const double duration_s = static_cast<double>(end_step - start_step)
        * DT_MS / 1000.0;
    std::vector<double> out(counts.size(), 0.0);
    if (duration_s <= 0.0) {
        return out;
    }
    for (std::size_t i = 0; i < counts.size(); ++i) {
        out[i] = static_cast<double>(counts[i]) / duration_s;
    }
    return out;
}

DecayPrimitiveResult run_v1e_decay(int n_steps, bool threshold_case) {
    DecayPrimitiveResult result;
    result.population = "v1_e";
    result.n_cells = V1_E_N;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    init_v1e_state(result.cpu_state, result.cpu_spike_counts, threshold_case);
    result.cuda_state = result.cpu_state;
    result.cuda_spike_counts = result.cpu_spike_counts;
    cpu_step_v1e(result.cpu_state, result.cpu_spike_counts, n_steps);

    auto* d_v = device_alloc_copy(result.cuda_state["V_soma_mV"]);
    auto* d_vap = device_alloc_copy(result.cuda_state["V_ap_mV"]);
    auto* d_ie = device_alloc_copy(result.cuda_state["I_e_pA"]);
    auto* d_ii = device_alloc_copy(result.cuda_state["I_i_pA"]);
    auto* d_iape = device_alloc_copy(result.cuda_state["I_ap_e_pA"]);
    auto* d_w = device_alloc_copy(result.cuda_state["w_adapt_pA"]);
    auto* d_ibias = device_alloc_copy(result.cuda_state["I_bias_pA"]);
    auto* d_refrac = device_alloc_copy(result.cuda_state["refrac_until_ms"]);
    auto* d_spikes = device_alloc_copy(result.cuda_spike_counts);
    v1e_decay_kernel<<<1, 256>>>(
        n_steps, d_v, d_vap, d_ie, d_ii, d_iape, d_w, d_ibias, d_refrac,
        d_spikes
    );
    check_cuda(cudaGetLastError(), "v1e_decay_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "v1e_decay_kernel sync");
    copy_device_to_host(d_v, result.cuda_state["V_soma_mV"]);
    copy_device_to_host(d_vap, result.cuda_state["V_ap_mV"]);
    copy_device_to_host(d_ie, result.cuda_state["I_e_pA"]);
    copy_device_to_host(d_ii, result.cuda_state["I_i_pA"]);
    copy_device_to_host(d_iape, result.cuda_state["I_ap_e_pA"]);
    copy_device_to_host(d_w, result.cuda_state["w_adapt_pA"]);
    copy_device_to_host(d_ibias, result.cuda_state["I_bias_pA"]);
    copy_device_to_host(d_refrac, result.cuda_state["refrac_until_ms"]);
    copy_device_to_host(d_spikes, result.cuda_spike_counts);
    device_free(d_v); device_free(d_vap); device_free(d_ie); device_free(d_ii);
    device_free(d_iape); device_free(d_w); device_free(d_ibias);
    device_free(d_refrac); device_free(d_spikes);
    result.max_abs_error = compare_state(result.cpu_state, result.cuda_state);
    result.cpu_total_spikes = std::accumulate(
        result.cpu_spike_counts.begin(), result.cpu_spike_counts.end(), 0
    );
    result.cuda_total_spikes = std::accumulate(
        result.cuda_spike_counts.begin(), result.cuda_spike_counts.end(), 0
    );
    return result;
}

DecayPrimitiveResult run_he_decay(int n_steps, bool threshold_case) {
    DecayPrimitiveResult result;
    result.population = "h_e";
    result.n_cells = H_E_N;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    init_he_state(result.cpu_state, result.cpu_spike_counts, threshold_case);
    result.cuda_state = result.cpu_state;
    result.cuda_spike_counts = result.cpu_spike_counts;
    cpu_step_he(result.cpu_state, result.cpu_spike_counts, n_steps);

    auto* d_v = device_alloc_copy(result.cuda_state["V_mV"]);
    auto* d_ie = device_alloc_copy(result.cuda_state["I_e_pA"]);
    auto* d_ii = device_alloc_copy(result.cuda_state["I_i_pA"]);
    auto* d_g = device_alloc_copy(result.cuda_state["g_nmda_h_nS"]);
    auto* d_ibias = device_alloc_copy(result.cuda_state["I_bias_pA"]);
    auto* d_refrac = device_alloc_copy(result.cuda_state["refrac_until_ms"]);
    auto* d_spikes = device_alloc_copy(result.cuda_spike_counts);
    he_decay_kernel<<<1, 256>>>(
        n_steps, d_v, d_ie, d_ii, d_g, d_ibias, d_refrac, d_spikes
    );
    check_cuda(cudaGetLastError(), "he_decay_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "he_decay_kernel sync");
    copy_device_to_host(d_v, result.cuda_state["V_mV"]);
    copy_device_to_host(d_ie, result.cuda_state["I_e_pA"]);
    copy_device_to_host(d_ii, result.cuda_state["I_i_pA"]);
    copy_device_to_host(d_g, result.cuda_state["g_nmda_h_nS"]);
    copy_device_to_host(d_ibias, result.cuda_state["I_bias_pA"]);
    copy_device_to_host(d_refrac, result.cuda_state["refrac_until_ms"]);
    copy_device_to_host(d_spikes, result.cuda_spike_counts);
    device_free(d_v); device_free(d_ie); device_free(d_ii); device_free(d_g);
    device_free(d_ibias); device_free(d_refrac); device_free(d_spikes);
    result.max_abs_error = compare_state(result.cpu_state, result.cuda_state);
    result.cpu_total_spikes = std::accumulate(
        result.cpu_spike_counts.begin(), result.cpu_spike_counts.end(), 0
    );
    result.cuda_total_spikes = std::accumulate(
        result.cuda_spike_counts.begin(), result.cuda_spike_counts.end(), 0
    );
    return result;
}

constexpr int CTX_PRED_N_PRE = H_E_N;
constexpr int CTX_PRED_N_POST = H_E_N;
constexpr int CTX_PRED_N_SYN = CTX_PRED_N_PRE * CTX_PRED_N_POST;
constexpr double CTX_PRED_TAU_COINC_MS = 500.0;
constexpr double CTX_PRED_TAU_ELIG_MS = 1000.0;
constexpr double CTX_PRED_ETA = 1e-3;
constexpr double CTX_PRED_GAMMA = 1e-4;
constexpr double CTX_PRED_W_TARGET = 0.0075;
constexpr double CTX_PRED_W_MAX = 1.0;
constexpr double CTX_PRED_W_ROW_MAX = 3.0;
constexpr double CTX_PRED_W_INIT_FRAC = 0.015;
constexpr double CTX_PRED_M_INTEGRAL = 0.150;
constexpr double CTX_PRED_DT_TRIAL_S = 2.5;
constexpr int CTX_PRED_PAIRED_PRE = 3;
constexpr int CTX_PRED_PAIRED_POST = 4;
constexpr int CTX_PRED_PRE_RULE_PRE = 8;
constexpr int CTX_PRED_CAPPED_PRE = 17;
constexpr int CTX_PRED_SILENT_PRE = 5;
constexpr int CTX_PRED_SILENT_POST = 6;
constexpr int CTX_PRED_TRIAL_LEADER_PRE = 21;
constexpr int CTX_PRED_TRIAL_BOUNDARY_PRE = 22;
constexpr int CTX_PRED_TRIAL_TRAILER_POST = 31;
constexpr int CTX_PRED_TRIAL_LATE_TRAILER_POST = 32;

std::size_t ctx_pred_edge_index(int pre, int post) {
    return static_cast<std::size_t>(pre) * CTX_PRED_N_POST
        + static_cast<std::size_t>(post);
}

std::vector<double> init_ctx_pred_test_weights(std::int64_t seed) {
    std::vector<double> w(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    const double high = CTX_PRED_W_INIT_FRAC * CTX_PRED_W_MAX;
    for (int edge = 0; edge < CTX_PRED_N_SYN; ++edge) {
        w[static_cast<std::size_t>(edge)] =
            counter_uniform01(seed, 0, edge) * high;
    }
    for (int post = 0; post < CTX_PRED_N_POST; ++post) {
        w[ctx_pred_edge_index(CTX_PRED_CAPPED_PRE, post)] = 0.05;
    }
    return w;
}

void cpu_ctx_pred_decay_step(
    std::vector<double>& xpre,
    std::vector<double>& xpost,
    std::vector<double>& elig
) {
    const double decay_coinc = std::exp(-DT_MS / CTX_PRED_TAU_COINC_MS);
    const double decay_elig = std::exp(-DT_MS / CTX_PRED_TAU_ELIG_MS);
    for (int edge = 0; edge < CTX_PRED_N_SYN; ++edge) {
        const std::size_t idx = static_cast<std::size_t>(edge);
        xpre[idx] *= decay_coinc;
        xpost[idx] *= decay_coinc;
        elig[idx] *= decay_elig;
    }
}

void cpu_ctx_pred_pre_event(
    std::vector<double>& xpre,
    std::vector<double>& xpost,
    std::vector<double>& elig,
    int pre
) {
    for (int post = 0; post < CTX_PRED_N_POST; ++post) {
        const std::size_t idx = ctx_pred_edge_index(pre, post);
        xpre[idx] += 1.0;
        elig[idx] += xpost[idx];
    }
}

void cpu_ctx_pred_post_event(
    std::vector<double>& xpre,
    std::vector<double>& xpost,
    std::vector<double>& elig,
    int post
) {
    for (int pre = 0; pre < CTX_PRED_N_PRE; ++pre) {
        const std::size_t idx = ctx_pred_edge_index(pre, post);
        xpost[idx] += 1.0;
        elig[idx] += xpre[idx];
    }
}

std::int32_t cpu_ctx_pred_gate_update(
    std::vector<double>& w,
    std::vector<double>& elig
) {
    for (int edge = 0; edge < CTX_PRED_N_SYN; ++edge) {
        const std::size_t idx = static_cast<std::size_t>(edge);
        const double dw =
            CTX_PRED_ETA * elig[idx] * CTX_PRED_M_INTEGRAL
            - CTX_PRED_GAMMA * (w[idx] - CTX_PRED_W_TARGET)
                * CTX_PRED_DT_TRIAL_S;
        w[idx] = std::min(CTX_PRED_W_MAX, std::max(0.0, w[idx] + dw));
    }

    std::int32_t n_capped = 0;
    for (int pre = 0; pre < CTX_PRED_N_PRE; ++pre) {
        double row_sum = 0.0;
        for (int post = 0; post < CTX_PRED_N_POST; ++post) {
            row_sum += w[ctx_pred_edge_index(pre, post)];
        }
        if (row_sum > CTX_PRED_W_ROW_MAX && row_sum > 1e-12) {
            const double scale = CTX_PRED_W_ROW_MAX / row_sum;
            for (int post = 0; post < CTX_PRED_N_POST; ++post) {
                w[ctx_pred_edge_index(pre, post)] *= scale;
            }
            ++n_capped;
        }
    }

    std::fill(elig.begin(), elig.end(), 0.0);
    return n_capped;
}

std::vector<double> ctx_pred_row_sums(const std::vector<double>& w) {
    std::vector<double> row_sums(static_cast<std::size_t>(CTX_PRED_N_PRE), 0.0);
    for (int pre = 0; pre < CTX_PRED_N_PRE; ++pre) {
        double sum = 0.0;
        for (int post = 0; post < CTX_PRED_N_POST; ++post) {
            sum += w[ctx_pred_edge_index(pre, post)];
        }
        row_sums[static_cast<std::size_t>(pre)] = sum;
    }
    return row_sums;
}

double vector_sum(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

double vector_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return vector_sum(values) / static_cast<double>(values.size());
}

double vector_max_value(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return *std::max_element(values.begin(), values.end());
}

double sum_delta(
    const std::vector<double>& before,
    const std::vector<double>& after
) {
    if (before.size() != after.size()) {
        throw std::runtime_error("sum_delta size mismatch");
    }
    double out = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        out += after[i] - before[i];
    }
    return out;
}

std::vector<double> init_h_ee_placeholder_weights(
    std::int64_t seed,
    std::int64_t stream
) {
    constexpr int h_ee_n_syn = H_E_N * (H_E_N - 1);
    std::vector<double> out(static_cast<std::size_t>(h_ee_n_syn), 0.0);
    for (int edge = 0; edge < h_ee_n_syn; ++edge) {
        out[static_cast<std::size_t>(edge)] =
            counter_uniform01(seed + stream, 1, edge) * 0.02;
    }
    return out;
}

int ctx_pred_stage1_phase(
    int step,
    int leader_start_step,
    int leader_end_step,
    int trailer_start_step,
    int trailer_end_step,
    int iti_start_step,
    int iti_end_step
) {
    if (step >= leader_start_step && step < leader_end_step) {
        return 0;
    }
    if (step >= trailer_start_step && step < trailer_end_step) {
        return 1;
    }
    if (step >= iti_start_step && step < iti_end_step) {
        return 2;
    }
    return -1;
}

void add_stage1_event_count(
    std::map<std::string, std::int32_t>& counts,
    const std::string& prefix,
    int step,
    int leader_start_step,
    int leader_end_step,
    int trailer_start_step,
    int trailer_end_step,
    int iti_start_step,
    int iti_end_step
) {
    const int phase = ctx_pred_stage1_phase(
        step,
        leader_start_step,
        leader_end_step,
        trailer_start_step,
        trailer_end_step,
        iti_start_step,
        iti_end_step
    );
    if (phase == 0) {
        ++counts[prefix + ".leader"];
    } else if (phase == 1) {
        ++counts[prefix + ".trailer"];
    } else if (phase == 2) {
        ++counts[prefix + ".iti"];
    } else {
        ++counts[prefix + ".outside"];
    }
}

__global__ void ctx_pred_decay_kernel(
    double* xpre,
    double* xpost,
    double* elig,
    int n_syn,
    double decay_coinc,
    double decay_elig
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) {
        return;
    }
    xpre[idx] *= decay_coinc;
    xpost[idx] *= decay_coinc;
    elig[idx] *= decay_elig;
}

__global__ void ctx_pred_pre_event_kernel(
    double* xpre,
    double* xpost,
    double* elig,
    int pre,
    int n_post
) {
    const int post = blockIdx.x * blockDim.x + threadIdx.x;
    if (post >= n_post) {
        return;
    }
    const int idx = pre * n_post + post;
    xpre[idx] += 1.0;
    elig[idx] += xpost[idx];
}

__global__ void ctx_pred_post_event_kernel(
    double* xpre,
    double* xpost,
    double* elig,
    int post,
    int n_pre,
    int n_post
) {
    const int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n_pre) {
        return;
    }
    const int idx = pre * n_post + post;
    xpost[idx] += 1.0;
    elig[idx] += xpre[idx];
}

__global__ void ctx_pred_gate_preclip_kernel(
    double* w,
    double* elig,
    int n_syn,
    double eta,
    double m_integral,
    double gamma,
    double w_target,
    double dt_trial_s,
    double w_max
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_syn) {
        return;
    }
    const double dw =
        eta * elig[idx] * m_integral
        - gamma * (w[idx] - w_target) * dt_trial_s;
    w[idx] = fmin(w_max, fmax(0.0, w[idx] + dw));
    elig[idx] = 0.0;
}

__global__ void ctx_pred_row_cap_kernel(
    double* w,
    int n_pre,
    int n_post,
    double w_row_max,
    int* capped_flags
) {
    const int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n_pre) {
        return;
    }
    double row_sum = 0.0;
    const int row_start = pre * n_post;
    for (int post = 0; post < n_post; ++post) {
        row_sum += w[row_start + post];
    }
    if (row_sum > w_row_max && row_sum > 1e-12) {
        const double scale = w_row_max / row_sum;
        for (int post = 0; post < n_post; ++post) {
            w[row_start + post] *= scale;
        }
        capped_flags[pre] = 1;
    } else {
        capped_flags[pre] = 0;
    }
}

}  // namespace

std::string backend_info() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    std::ostringstream out;
    out << "cudaGetDeviceCount_status=" << static_cast<int>(status);
    if (status == cudaSuccess) {
        out << " device_count=" << device_count;
    } else {
        out << " error=" << cudaGetErrorString(status);
    }
    return out.str();
}

DecayPrimitiveResult run_decay_primitive(
    const std::string& population,
    std::int32_t n_steps,
    bool threshold_case
) {
    if (n_steps < 1) {
        throw std::runtime_error("n_steps must be >= 1");
    }
    if (population == "v1_e") {
        return run_v1e_decay(n_steps, threshold_case);
    }
    if (population == "h_e") {
    return run_he_decay(n_steps, threshold_case);
    }
    throw std::runtime_error(
        "unknown population for decay primitive: " + population
        + " (expected 'v1_e' or 'h_e')"
    );
}

HRingDynamicsTestResult run_h_ring_dynamics_test(std::int64_t seed) {
    (void)seed;
    const int leader_start = 0;
    const int leader_end = 800;
    const int persistence_start = leader_end;
    const int persistence_end = 3800;
    const int late_start = persistence_end;
    const int late_end = 5600;
    const int pretrailer_start = 3000;
    const int pretrailer_end = 3800;
    const int trailer_start = 4000;
    const int trailer_end = 4800;
    const int n_steps = 6000;
    const int ctx_channel = 0;
    const int pred_channel = 1;
    const int pulse_period_steps = 20;

    HRingDynamicsTestResult result;
    result.seed = static_cast<std::int32_t>(seed);
    result.n_e = H_E_N;
    result.n_inh = H_INH_N;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    result.phase_steps = {
        {"leader_start_step", leader_start},
        {"leader_end_step", leader_end},
        {"persistence_start_step", persistence_start},
        {"persistence_end_step", persistence_end},
        {"late_start_step", late_start},
        {"late_end_step", late_end},
        {"pretrailer_start_step", pretrailer_start},
        {"pretrailer_end_step", pretrailer_end},
        {"trailer_start_step", trailer_start},
        {"trailer_end_step", trailer_end},
    };

    HNativeRingState cpu_ctx;
    HNativeRingState cpu_pred;
    HNativeRingState cuda_ctx;
    HNativeRingState cuda_pred;
    init_h_native_ring_state(cpu_ctx);
    init_h_native_ring_state(cpu_pred);
    init_h_native_ring_state(cuda_ctx);
    init_h_native_ring_state(cuda_pred);

    result.cpu_ctx_leader_counts.assign(H_E_N, 0);
    result.cpu_ctx_persistence_counts.assign(H_E_N, 0);
    result.cpu_ctx_late_counts.assign(H_E_N, 0);
    result.cpu_ctx_total_counts.assign(H_E_N, 0);
    result.cpu_pred_leader_counts.assign(H_E_N, 0);
    result.cpu_pred_pretrailer_counts.assign(H_E_N, 0);
    result.cpu_pred_trailer_counts.assign(H_E_N, 0);
    result.cpu_pred_total_counts.assign(H_E_N, 0);
    result.cpu_ctx_inh_total_counts.assign(H_INH_N, 0);
    result.cpu_pred_inh_total_counts.assign(H_INH_N, 0);

    int cpu_ctx_last_persistence_step = -1;
    int cpu_pred_last_pretrailer_step = -1;
    for (int step = 0; step < n_steps; ++step) {
        if (step >= leader_start && step < leader_end
            && step % pulse_period_steps == 0) {
            cpu_h_ring_external_pulse(cpu_ctx, ctx_channel, H_RING_EXT_DRIVE_PA);
        }
        if (step >= trailer_start && step < trailer_end
            && step % pulse_period_steps == 0) {
            cpu_h_ring_external_pulse(cpu_pred, pred_channel, H_RING_EXT_DRIVE_PA);
        }
        cpu_h_ring_e_step(
            cpu_ctx, step, leader_start, leader_end, persistence_start,
            persistence_end, late_start, late_end, result.cpu_ctx_leader_counts,
            result.cpu_ctx_persistence_counts, result.cpu_ctx_late_counts,
            result.cpu_ctx_total_counts, &cpu_ctx_last_persistence_step
        );
        cpu_h_ring_inh_step(cpu_ctx, step, result.cpu_ctx_inh_total_counts);
        cpu_h_ring_e_step(
            cpu_pred, step, leader_start, leader_end, pretrailer_start,
            pretrailer_end, trailer_start, trailer_end,
            result.cpu_pred_leader_counts, result.cpu_pred_pretrailer_counts,
            result.cpu_pred_trailer_counts, result.cpu_pred_total_counts,
            &cpu_pred_last_pretrailer_step
        );
        cpu_h_ring_inh_step(cpu_pred, step, result.cpu_pred_inh_total_counts);
        cpu_h_ring_scatter_spikes(cpu_ctx);
        cpu_h_ring_scatter_spikes(cpu_pred);
    }

    result.cuda_ctx_leader_counts.assign(H_E_N, 0);
    result.cuda_ctx_persistence_counts.assign(H_E_N, 0);
    result.cuda_ctx_late_counts.assign(H_E_N, 0);
    result.cuda_ctx_total_counts.assign(H_E_N, 0);
    result.cuda_pred_leader_counts.assign(H_E_N, 0);
    result.cuda_pred_pretrailer_counts.assign(H_E_N, 0);
    result.cuda_pred_trailer_counts.assign(H_E_N, 0);
    result.cuda_pred_total_counts.assign(H_E_N, 0);
    result.cuda_ctx_inh_total_counts.assign(H_INH_N, 0);
    result.cuda_pred_inh_total_counts.assign(H_INH_N, 0);
    std::vector<std::int32_t> cuda_ctx_last(1, -1);
    std::vector<std::int32_t> cuda_pred_last(1, -1);

    auto* d_ctx_ev = device_alloc_copy(cuda_ctx.e_v);
    auto* d_ctx_eie = device_alloc_copy(cuda_ctx.e_ie);
    auto* d_ctx_eii = device_alloc_copy(cuda_ctx.e_ii);
    auto* d_ctx_eg = device_alloc_copy(cuda_ctx.e_gnmda);
    auto* d_ctx_er = device_alloc_copy(cuda_ctx.e_refrac);
    auto* d_ctx_iv = device_alloc_copy(cuda_ctx.inh_v);
    auto* d_ctx_iie = device_alloc_copy(cuda_ctx.inh_ie);
    auto* d_ctx_iii = device_alloc_copy(cuda_ctx.inh_ii);
    auto* d_ctx_ir = device_alloc_copy(cuda_ctx.inh_refrac);
    auto* d_ctx_es = device_alloc_copy(cuda_ctx.e_spikes);
    auto* d_ctx_is = device_alloc_copy(cuda_ctx.inh_spikes);
    auto* d_ctx_leader = device_alloc_copy(result.cuda_ctx_leader_counts);
    auto* d_ctx_persist = device_alloc_copy(result.cuda_ctx_persistence_counts);
    auto* d_ctx_late = device_alloc_copy(result.cuda_ctx_late_counts);
    auto* d_ctx_total = device_alloc_copy(result.cuda_ctx_total_counts);
    auto* d_ctx_i_total = device_alloc_copy(result.cuda_ctx_inh_total_counts);
    auto* d_ctx_last = device_alloc_copy(cuda_ctx_last);

    auto* d_pred_ev = device_alloc_copy(cuda_pred.e_v);
    auto* d_pred_eie = device_alloc_copy(cuda_pred.e_ie);
    auto* d_pred_eii = device_alloc_copy(cuda_pred.e_ii);
    auto* d_pred_eg = device_alloc_copy(cuda_pred.e_gnmda);
    auto* d_pred_er = device_alloc_copy(cuda_pred.e_refrac);
    auto* d_pred_iv = device_alloc_copy(cuda_pred.inh_v);
    auto* d_pred_iie = device_alloc_copy(cuda_pred.inh_ie);
    auto* d_pred_iii = device_alloc_copy(cuda_pred.inh_ii);
    auto* d_pred_ir = device_alloc_copy(cuda_pred.inh_refrac);
    auto* d_pred_es = device_alloc_copy(cuda_pred.e_spikes);
    auto* d_pred_is = device_alloc_copy(cuda_pred.inh_spikes);
    auto* d_pred_leader = device_alloc_copy(result.cuda_pred_leader_counts);
    auto* d_pred_pre = device_alloc_copy(result.cuda_pred_pretrailer_counts);
    auto* d_pred_trailer = device_alloc_copy(result.cuda_pred_trailer_counts);
    auto* d_pred_total = device_alloc_copy(result.cuda_pred_total_counts);
    auto* d_pred_i_total = device_alloc_copy(result.cuda_pred_inh_total_counts);
    auto* d_pred_last = device_alloc_copy(cuda_pred_last);

    const int block = 256;
    for (int step = 0; step < n_steps; ++step) {
        if (step >= leader_start && step < leader_end
            && step % pulse_period_steps == 0) {
            h_ring_external_pulse_kernel<<<1, 32>>>(
                ctx_channel, H_RING_EXT_DRIVE_PA, d_ctx_eie
            );
            check_cuda(cudaGetLastError(), "h ctx external pulse launch");
        }
        if (step >= trailer_start && step < trailer_end
            && step % pulse_period_steps == 0) {
            h_ring_external_pulse_kernel<<<1, 32>>>(
                pred_channel, H_RING_EXT_DRIVE_PA, d_pred_eie
            );
            check_cuda(cudaGetLastError(), "h pred external pulse launch");
        }
        h_ring_e_step_kernel<<<1, block>>>(
            step, leader_start, leader_end, persistence_start, persistence_end,
            late_start, late_end, d_ctx_ev, d_ctx_eie, d_ctx_eii, d_ctx_eg,
            d_ctx_er, d_ctx_leader, d_ctx_persist, d_ctx_late, d_ctx_total,
            d_ctx_es, d_ctx_last
        );
        check_cuda(cudaGetLastError(), "h ctx e step launch");
        h_ring_inh_step_kernel<<<1, block>>>(
            step, d_ctx_iv, d_ctx_iie, d_ctx_iii, d_ctx_ir, d_ctx_i_total,
            d_ctx_is
        );
        check_cuda(cudaGetLastError(), "h ctx inh step launch");
        h_ring_e_step_kernel<<<1, block>>>(
            step, leader_start, leader_end, pretrailer_start, pretrailer_end,
            trailer_start, trailer_end, d_pred_ev, d_pred_eie, d_pred_eii,
            d_pred_eg, d_pred_er, d_pred_leader, d_pred_pre, d_pred_trailer,
            d_pred_total, d_pred_es, d_pred_last
        );
        check_cuda(cudaGetLastError(), "h pred e step launch");
        h_ring_inh_step_kernel<<<1, block>>>(
            step, d_pred_iv, d_pred_iie, d_pred_iii, d_pred_ir, d_pred_i_total,
            d_pred_is
        );
        check_cuda(cudaGetLastError(), "h pred inh step launch");
        h_ring_e_recurrent_scatter_kernel<<<1, block>>>(
            d_ctx_es, d_ctx_eie, d_ctx_eg, d_ctx_iie
        );
        check_cuda(cudaGetLastError(), "h ctx e scatter launch");
        h_ring_inh_scatter_kernel<<<1, block>>>(d_ctx_is, d_ctx_eii);
        check_cuda(cudaGetLastError(), "h ctx inh scatter launch");
        h_ring_e_recurrent_scatter_kernel<<<1, block>>>(
            d_pred_es, d_pred_eie, d_pred_eg, d_pred_iie
        );
        check_cuda(cudaGetLastError(), "h pred e scatter launch");
        h_ring_inh_scatter_kernel<<<1, block>>>(d_pred_is, d_pred_eii);
        check_cuda(cudaGetLastError(), "h pred inh scatter launch");
    }
    check_cuda(cudaDeviceSynchronize(), "h ring dynamics sync");

    copy_device_to_host(d_ctx_ev, cuda_ctx.e_v);
    copy_device_to_host(d_ctx_eie, cuda_ctx.e_ie);
    copy_device_to_host(d_ctx_eii, cuda_ctx.e_ii);
    copy_device_to_host(d_ctx_eg, cuda_ctx.e_gnmda);
    copy_device_to_host(d_ctx_iv, cuda_ctx.inh_v);
    copy_device_to_host(d_ctx_iie, cuda_ctx.inh_ie);
    copy_device_to_host(d_ctx_leader, result.cuda_ctx_leader_counts);
    copy_device_to_host(d_ctx_persist, result.cuda_ctx_persistence_counts);
    copy_device_to_host(d_ctx_late, result.cuda_ctx_late_counts);
    copy_device_to_host(d_ctx_total, result.cuda_ctx_total_counts);
    copy_device_to_host(d_ctx_i_total, result.cuda_ctx_inh_total_counts);
    copy_device_to_host(d_ctx_last, cuda_ctx_last);

    copy_device_to_host(d_pred_ev, cuda_pred.e_v);
    copy_device_to_host(d_pred_eie, cuda_pred.e_ie);
    copy_device_to_host(d_pred_eii, cuda_pred.e_ii);
    copy_device_to_host(d_pred_eg, cuda_pred.e_gnmda);
    copy_device_to_host(d_pred_iv, cuda_pred.inh_v);
    copy_device_to_host(d_pred_iie, cuda_pred.inh_ie);
    copy_device_to_host(d_pred_leader, result.cuda_pred_leader_counts);
    copy_device_to_host(d_pred_pre, result.cuda_pred_pretrailer_counts);
    copy_device_to_host(d_pred_trailer, result.cuda_pred_trailer_counts);
    copy_device_to_host(d_pred_total, result.cuda_pred_total_counts);
    copy_device_to_host(d_pred_i_total, result.cuda_pred_inh_total_counts);
    copy_device_to_host(d_pred_last, cuda_pred_last);

    device_free(d_ctx_ev); device_free(d_ctx_eie); device_free(d_ctx_eii);
    device_free(d_ctx_eg); device_free(d_ctx_er); device_free(d_ctx_iv);
    device_free(d_ctx_iie); device_free(d_ctx_iii); device_free(d_ctx_ir);
    device_free(d_ctx_es); device_free(d_ctx_is); device_free(d_ctx_leader);
    device_free(d_ctx_persist); device_free(d_ctx_late); device_free(d_ctx_total);
    device_free(d_ctx_i_total); device_free(d_ctx_last);
    device_free(d_pred_ev); device_free(d_pred_eie); device_free(d_pred_eii);
    device_free(d_pred_eg); device_free(d_pred_er); device_free(d_pred_iv);
    device_free(d_pred_iie); device_free(d_pred_iii); device_free(d_pred_ir);
    device_free(d_pred_es); device_free(d_pred_is); device_free(d_pred_leader);
    device_free(d_pred_pre); device_free(d_pred_trailer); device_free(d_pred_total);
    device_free(d_pred_i_total); device_free(d_pred_last);

    append_h_native_state(result.cpu_final_state, "ctx", cpu_ctx);
    append_h_native_state(result.cpu_final_state, "pred", cpu_pred);
    append_h_native_state(result.cuda_final_state, "ctx", cuda_ctx);
    append_h_native_state(result.cuda_final_state, "pred", cuda_pred);
    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    result.max_abs_error["ctx_leader_counts"] = max_count_abs_diff(
        result.cpu_ctx_leader_counts, result.cuda_ctx_leader_counts
    );
    result.max_abs_error["ctx_persistence_counts"] = max_count_abs_diff(
        result.cpu_ctx_persistence_counts, result.cuda_ctx_persistence_counts
    );
    result.max_abs_error["ctx_late_counts"] = max_count_abs_diff(
        result.cpu_ctx_late_counts, result.cuda_ctx_late_counts
    );
    result.max_abs_error["ctx_total_counts"] = max_count_abs_diff(
        result.cpu_ctx_total_counts, result.cuda_ctx_total_counts
    );
    result.max_abs_error["pred_leader_counts"] = max_count_abs_diff(
        result.cpu_pred_leader_counts, result.cuda_pred_leader_counts
    );
    result.max_abs_error["pred_pretrailer_counts"] = max_count_abs_diff(
        result.cpu_pred_pretrailer_counts, result.cuda_pred_pretrailer_counts
    );
    result.max_abs_error["pred_trailer_counts"] = max_count_abs_diff(
        result.cpu_pred_trailer_counts, result.cuda_pred_trailer_counts
    );
    result.max_abs_error["pred_total_counts"] = max_count_abs_diff(
        result.cpu_pred_total_counts, result.cuda_pred_total_counts
    );
    result.max_abs_error["ctx_inh_total_counts"] = max_count_abs_diff(
        result.cpu_ctx_inh_total_counts, result.cuda_ctx_inh_total_counts
    );
    result.max_abs_error["pred_inh_total_counts"] = max_count_abs_diff(
        result.cpu_pred_inh_total_counts, result.cuda_pred_inh_total_counts
    );

    const double duration_s = static_cast<double>(n_steps) * DT_MS / 1000.0;
    const int ctx_leader_total = sum_counts(result.cpu_ctx_leader_counts);
    const int ctx_persist_total = sum_counts(result.cpu_ctx_persistence_counts);
    const int ctx_late_total = sum_counts(result.cpu_ctx_late_counts);
    const int pred_leader_total = sum_counts(result.cpu_pred_leader_counts);
    const int pred_pre_total = sum_counts(result.cpu_pred_pretrailer_counts);
    const int pred_trailer_total = sum_counts(result.cpu_pred_trailer_counts);
    const double ctx_max_rate_hz =
        static_cast<double>(*std::max_element(
            result.cpu_ctx_total_counts.begin(), result.cpu_ctx_total_counts.end()
        )) / duration_s;
    const double pred_max_rate_hz =
        static_cast<double>(*std::max_element(
            result.cpu_pred_total_counts.begin(), result.cpu_pred_total_counts.end()
        )) / duration_s;
    const double max_rate_hz = std::max(ctx_max_rate_hz, pred_max_rate_hz);
    const double ctx_persistence_ms =
        cpu_ctx_last_persistence_step >= persistence_start
        ? static_cast<double>(cpu_ctx_last_persistence_step - leader_end) * DT_MS
        : 0.0;
    const double cuda_ctx_persistence_ms =
        cuda_ctx_last[0] >= persistence_start
        ? static_cast<double>(cuda_ctx_last[0] - leader_end) * DT_MS
        : 0.0;
    result.metrics = {
        {"ctx_leader_total_spikes", static_cast<double>(ctx_leader_total)},
        {"ctx_persistence_total_spikes", static_cast<double>(ctx_persist_total)},
        {"ctx_late_total_spikes", static_cast<double>(ctx_late_total)},
        {"ctx_inh_total_spikes",
         static_cast<double>(sum_counts(result.cpu_ctx_inh_total_counts))},
        {"ctx_persistence_ms", ctx_persistence_ms},
        {"cuda_ctx_persistence_ms", cuda_ctx_persistence_ms},
        {"pred_leader_total_spikes", static_cast<double>(pred_leader_total)},
        {"pred_pretrailer_total_spikes", static_cast<double>(pred_pre_total)},
        {"pred_trailer_total_spikes", static_cast<double>(pred_trailer_total)},
        {"pred_inh_total_spikes",
         static_cast<double>(sum_counts(result.cpu_pred_inh_total_counts))},
        {"ctx_max_rate_hz", ctx_max_rate_hz},
        {"pred_max_rate_hz", pred_max_rate_hz},
        {"max_rate_hz", max_rate_hz},
        {"no_runaway_pass", max_rate_hz <= 80.0 ? 1.0 : 0.0},
        {"ctx_persistence_window_pass",
         (ctx_persistence_ms >= 200.0 && ctx_persistence_ms <= 500.0) ? 1.0 : 0.0},
        {"pred_silent_leader_pass", pred_leader_total == 0 ? 1.0 : 0.0},
        {"pred_trailer_driven_pass", pred_trailer_total > 0 ? 1.0 : 0.0},
    };
    return result;
}

Stage1HGateDynamicsResult run_stage1_h_gate_dynamics_test(
    std::int64_t seed,
    const std::vector<std::int32_t>& leader_cells,
    const std::vector<std::int32_t>& trailer_cells,
    const std::vector<double>& w_ctx_pred
) {
    if (leader_cells.empty()) {
        throw std::runtime_error("Stage1 H gate dynamics requires nonempty schedule");
    }
    if (leader_cells.size() != trailer_cells.size()) {
        throw std::runtime_error("leader_cells and trailer_cells must have equal size");
    }
    if (w_ctx_pred.size() != static_cast<std::size_t>(H_E_N * H_E_N)) {
        throw std::runtime_error("W_ctx_pred must have length 36864");
    }

    const int n_trials = static_cast<int>(leader_cells.size());
    const int leader_start = 0;
    const int leader_end = 800;
    const int persistence_start = leader_end;
    const int persistence_end = 3800;
    const int pretrailer_start = 2000;
    const int pretrailer_end = 3000;
    const int trailer_start = 4000;
    const int trailer_end = 4800;
    const int late_start = persistence_end;
    const int late_end = 5600;
    const int n_steps = 6000;
    const int pulse_period_steps = 20;

    Stage1HGateDynamicsResult result;
    result.seed = seed;
    result.n_trials = n_trials;
    result.n_e = H_E_N;
    result.n_inh = H_INH_N;
    result.n_steps_per_trial = n_steps;
    result.dt_ms = DT_MS;
    result.phase_steps = {
        {"leader_start_step", leader_start},
        {"leader_end_step", leader_end},
        {"persistence_start_step", persistence_start},
        {"persistence_end_step", persistence_end},
        {"pretrailer_start_step", pretrailer_start},
        {"pretrailer_end_step", pretrailer_end},
        {"trailer_start_step", trailer_start},
        {"trailer_end_step", trailer_end},
        {"late_start_step", late_start},
        {"late_end_step", late_end},
    };
    result.leader_channels.reserve(static_cast<std::size_t>(n_trials));
    result.trailer_channels.reserve(static_cast<std::size_t>(n_trials));
    for (int trial = 0; trial < n_trials; ++trial) {
        const int leader_cell = leader_cells[static_cast<std::size_t>(trial)];
        const int trailer_cell = trailer_cells[static_cast<std::size_t>(trial)];
        if (leader_cell < 0 || leader_cell >= H_E_N
            || trailer_cell < 0 || trailer_cell >= H_E_N) {
            throw std::runtime_error("Stage1 H gate cell id out of range");
        }
        result.leader_channels.push_back(leader_cell / H_E_PER_CHANNEL);
        result.trailer_channels.push_back(trailer_cell / H_E_PER_CHANNEL);
    }

    const std::size_t e_total = static_cast<std::size_t>(n_trials) * H_E_N;
    const std::size_t inh_total = static_cast<std::size_t>(n_trials) * H_INH_N;
    std::vector<std::int32_t> cpu_ctx_leader(e_total, 0);
    std::vector<std::int32_t> cpu_ctx_persist(e_total, 0);
    std::vector<std::int32_t> cpu_ctx_late(e_total, 0);
    result.cpu_ctx_total_counts.assign(e_total, 0);
    std::vector<std::int32_t> cpu_pred_leader(e_total, 0);
    std::vector<std::int32_t> cpu_pred_pre(e_total, 0);
    std::vector<std::int32_t> cpu_pred_trailer(e_total, 0);
    result.cpu_pred_total_counts.assign(e_total, 0);
    result.cpu_ctx_inh_total_counts.assign(inh_total, 0);
    result.cpu_pred_inh_total_counts.assign(inh_total, 0);
    result.cpu_ctx_persistence_ms_by_trial.assign(
        static_cast<std::size_t>(n_trials), 0.0
    );
    result.cpu_pred_pretrailer_target_counts.assign(
        static_cast<std::size_t>(n_trials), 0
    );

    for (int trial = 0; trial < n_trials; ++trial) {
        HNativeRingState ctx;
        HNativeRingState pred;
        init_h_native_ring_state(ctx);
        init_h_native_ring_state(pred);
        std::vector<std::int32_t> ctx_leader(H_E_N, 0);
        std::vector<std::int32_t> ctx_persist(H_E_N, 0);
        std::vector<std::int32_t> ctx_late(H_E_N, 0);
        std::vector<std::int32_t> ctx_total(H_E_N, 0);
        std::vector<std::int32_t> pred_leader(H_E_N, 0);
        std::vector<std::int32_t> pred_pre(H_E_N, 0);
        std::vector<std::int32_t> pred_trailer(H_E_N, 0);
        std::vector<std::int32_t> pred_total(H_E_N, 0);
        std::vector<std::int32_t> ctx_inh_total(H_INH_N, 0);
        std::vector<std::int32_t> pred_inh_total(H_INH_N, 0);
        int ctx_last_persistence_step = -1;
        int pred_last_pretrailer_step = -1;

        for (int step = 0; step < n_steps; ++step) {
            if (step >= leader_start && step < leader_end
                && step % pulse_period_steps == 0) {
                cpu_h_ring_external_pulse(
                    ctx,
                    result.leader_channels[static_cast<std::size_t>(trial)],
                    H_RING_EXT_DRIVE_PA
                );
            }
            if (step >= trailer_start && step < trailer_end
                && step % pulse_period_steps == 0) {
                cpu_h_ring_external_pulse(
                    pred,
                    result.trailer_channels[static_cast<std::size_t>(trial)],
                    H_RING_EXT_DRIVE_PA
                );
            }
            cpu_h_ring_e_step(
                ctx, step, leader_start, leader_end, persistence_start,
                persistence_end, late_start, late_end, ctx_leader, ctx_persist,
                ctx_late, ctx_total, &ctx_last_persistence_step
            );
            cpu_h_ring_inh_step(ctx, step, ctx_inh_total);
            cpu_h_ring_e_step(
                pred, step, leader_start, leader_end, pretrailer_start,
                pretrailer_end, trailer_start, trailer_end, pred_leader,
                pred_pre, pred_trailer, pred_total, &pred_last_pretrailer_step
            );
            cpu_h_ring_inh_step(pred, step, pred_inh_total);
            cpu_h_ring_scatter_spikes(ctx);
            cpu_h_ctx_pred_dense_scatter(ctx, w_ctx_pred, pred);
            cpu_h_ring_scatter_spikes(pred);
        }

        const std::size_t e_base = static_cast<std::size_t>(trial) * H_E_N;
        const std::size_t inh_base = static_cast<std::size_t>(trial) * H_INH_N;
        for (int i = 0; i < H_E_N; ++i) {
            cpu_ctx_leader[e_base + static_cast<std::size_t>(i)] = ctx_leader[i];
            cpu_ctx_persist[e_base + static_cast<std::size_t>(i)] = ctx_persist[i];
            cpu_ctx_late[e_base + static_cast<std::size_t>(i)] = ctx_late[i];
            result.cpu_ctx_total_counts[e_base + static_cast<std::size_t>(i)] =
                ctx_total[i];
            cpu_pred_leader[e_base + static_cast<std::size_t>(i)] = pred_leader[i];
            cpu_pred_pre[e_base + static_cast<std::size_t>(i)] = pred_pre[i];
            cpu_pred_trailer[e_base + static_cast<std::size_t>(i)] = pred_trailer[i];
            result.cpu_pred_total_counts[e_base + static_cast<std::size_t>(i)] =
                pred_total[i];
        }
        for (int i = 0; i < H_INH_N; ++i) {
            result.cpu_ctx_inh_total_counts[inh_base + static_cast<std::size_t>(i)] =
                ctx_inh_total[i];
            result.cpu_pred_inh_total_counts[inh_base + static_cast<std::size_t>(i)] =
                pred_inh_total[i];
        }
        result.cpu_ctx_persistence_ms_by_trial[static_cast<std::size_t>(trial)] =
            ctx_last_persistence_step >= persistence_start
            ? static_cast<double>(ctx_last_persistence_step - leader_end) * DT_MS
            : 0.0;
        const int target_channel =
            result.trailer_channels[static_cast<std::size_t>(trial)];
        const int target_start = target_channel * H_E_PER_CHANNEL;
        int target_pre_count = 0;
        for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
            target_pre_count += pred_pre[static_cast<std::size_t>(target_start + k)];
        }
        result.cpu_pred_pretrailer_target_counts[static_cast<std::size_t>(trial)] =
            target_pre_count;
    }

    result.cuda_ctx_total_counts.assign(e_total, 0);
    result.cuda_pred_total_counts.assign(e_total, 0);
    result.cuda_ctx_inh_total_counts.assign(inh_total, 0);
    result.cuda_pred_inh_total_counts.assign(inh_total, 0);
    result.cuda_ctx_persistence_ms_by_trial.assign(
        static_cast<std::size_t>(n_trials), 0.0
    );
    result.cuda_pred_pretrailer_target_counts.assign(
        static_cast<std::size_t>(n_trials), 0
    );
    std::vector<std::int32_t> cuda_ctx_leader(e_total, 0);
    std::vector<std::int32_t> cuda_ctx_persist(e_total, 0);
    std::vector<std::int32_t> cuda_ctx_late(e_total, 0);
    std::vector<std::int32_t> cuda_pred_leader(e_total, 0);
    std::vector<std::int32_t> cuda_pred_pre(e_total, 0);
    std::vector<std::int32_t> cuda_pred_trailer(e_total, 0);
    std::vector<std::int32_t> cuda_ctx_last(static_cast<std::size_t>(n_trials), -1);
    std::vector<std::int32_t> cuda_pred_last(static_cast<std::size_t>(n_trials), -1);

    std::vector<double> ctx_ev(e_total, HE_EL_MV);
    std::vector<double> ctx_eie(e_total, 0.0);
    std::vector<double> ctx_eii(e_total, 0.0);
    std::vector<double> ctx_eg(e_total, 0.0);
    std::vector<double> ctx_er(e_total, -1.0);
    std::vector<double> ctx_iv(inh_total, HI_EL_MV);
    std::vector<double> ctx_iie(inh_total, 0.0);
    std::vector<double> ctx_iii(inh_total, 0.0);
    std::vector<double> ctx_ir(inh_total, -1.0);
    std::vector<std::int32_t> ctx_es(e_total, 0);
    std::vector<std::int32_t> ctx_is(inh_total, 0);
    std::vector<double> pred_ev(e_total, HE_EL_MV);
    std::vector<double> pred_eie(e_total, 0.0);
    std::vector<double> pred_eii(e_total, 0.0);
    std::vector<double> pred_eg(e_total, 0.0);
    std::vector<double> pred_er(e_total, -1.0);
    std::vector<double> pred_iv(inh_total, HI_EL_MV);
    std::vector<double> pred_iie(inh_total, 0.0);
    std::vector<double> pred_iii(inh_total, 0.0);
    std::vector<double> pred_ir(inh_total, -1.0);
    std::vector<std::int32_t> pred_es(e_total, 0);
    std::vector<std::int32_t> pred_is(inh_total, 0);

    auto* d_leader_channels = device_alloc_copy(result.leader_channels);
    auto* d_trailer_channels = device_alloc_copy(result.trailer_channels);
    auto* d_w = device_alloc_copy(w_ctx_pred);
    auto* d_ctx_ev = device_alloc_copy(ctx_ev);
    auto* d_ctx_eie = device_alloc_copy(ctx_eie);
    auto* d_ctx_eii = device_alloc_copy(ctx_eii);
    auto* d_ctx_eg = device_alloc_copy(ctx_eg);
    auto* d_ctx_er = device_alloc_copy(ctx_er);
    auto* d_ctx_iv = device_alloc_copy(ctx_iv);
    auto* d_ctx_iie = device_alloc_copy(ctx_iie);
    auto* d_ctx_iii = device_alloc_copy(ctx_iii);
    auto* d_ctx_ir = device_alloc_copy(ctx_ir);
    auto* d_ctx_es = device_alloc_copy(ctx_es);
    auto* d_ctx_is = device_alloc_copy(ctx_is);
    auto* d_ctx_leader = device_alloc_copy(cuda_ctx_leader);
    auto* d_ctx_persist = device_alloc_copy(cuda_ctx_persist);
    auto* d_ctx_late = device_alloc_copy(cuda_ctx_late);
    auto* d_ctx_total = device_alloc_copy(result.cuda_ctx_total_counts);
    auto* d_ctx_i_total = device_alloc_copy(result.cuda_ctx_inh_total_counts);
    auto* d_ctx_last = device_alloc_copy(cuda_ctx_last);
    auto* d_pred_ev = device_alloc_copy(pred_ev);
    auto* d_pred_eie = device_alloc_copy(pred_eie);
    auto* d_pred_eii = device_alloc_copy(pred_eii);
    auto* d_pred_eg = device_alloc_copy(pred_eg);
    auto* d_pred_er = device_alloc_copy(pred_er);
    auto* d_pred_iv = device_alloc_copy(pred_iv);
    auto* d_pred_iie = device_alloc_copy(pred_iie);
    auto* d_pred_iii = device_alloc_copy(pred_iii);
    auto* d_pred_ir = device_alloc_copy(pred_ir);
    auto* d_pred_es = device_alloc_copy(pred_es);
    auto* d_pred_is = device_alloc_copy(pred_is);
    auto* d_pred_leader = device_alloc_copy(cuda_pred_leader);
    auto* d_pred_pre = device_alloc_copy(cuda_pred_pre);
    auto* d_pred_trailer = device_alloc_copy(cuda_pred_trailer);
    auto* d_pred_total = device_alloc_copy(result.cuda_pred_total_counts);
    auto* d_pred_i_total = device_alloc_copy(result.cuda_pred_inh_total_counts);
    auto* d_pred_last = device_alloc_copy(cuda_pred_last);

    const int block = 256;
    const int e_grid = (static_cast<int>(e_total) + block - 1) / block;
    const int inh_grid = (static_cast<int>(inh_total) + block - 1) / block;
    const int pulse_grid = (n_trials * H_E_PER_CHANNEL + block - 1) / block;
    for (int step = 0; step < n_steps; ++step) {
        if (step >= leader_start && step < leader_end
            && step % pulse_period_steps == 0) {
            h_ring_external_pulse_batch_kernel<<<pulse_grid, block>>>(
                n_trials, d_leader_channels, H_RING_EXT_DRIVE_PA, d_ctx_eie
            );
            check_cuda(cudaGetLastError(), "stage1 h ctx pulse batch launch");
        }
        if (step >= trailer_start && step < trailer_end
            && step % pulse_period_steps == 0) {
            h_ring_external_pulse_batch_kernel<<<pulse_grid, block>>>(
                n_trials, d_trailer_channels, H_RING_EXT_DRIVE_PA, d_pred_eie
            );
            check_cuda(cudaGetLastError(), "stage1 h pred pulse batch launch");
        }
        h_ring_e_step_batch_kernel<<<e_grid, block>>>(
            n_trials, step, leader_start, leader_end, persistence_start,
            persistence_end, late_start, late_end, d_ctx_ev, d_ctx_eie,
            d_ctx_eii, d_ctx_eg, d_ctx_er, d_ctx_leader, d_ctx_persist,
            d_ctx_late, d_ctx_total, d_ctx_es, d_ctx_last
        );
        check_cuda(cudaGetLastError(), "stage1 h ctx e batch launch");
        h_ring_inh_step_batch_kernel<<<inh_grid, block>>>(
            n_trials, step, d_ctx_iv, d_ctx_iie, d_ctx_iii, d_ctx_ir,
            d_ctx_i_total, d_ctx_is
        );
        check_cuda(cudaGetLastError(), "stage1 h ctx inh batch launch");
        h_ring_e_step_batch_kernel<<<e_grid, block>>>(
            n_trials, step, leader_start, leader_end, pretrailer_start,
            pretrailer_end, trailer_start, trailer_end, d_pred_ev, d_pred_eie,
            d_pred_eii, d_pred_eg, d_pred_er, d_pred_leader, d_pred_pre,
            d_pred_trailer, d_pred_total, d_pred_es, d_pred_last
        );
        check_cuda(cudaGetLastError(), "stage1 h pred e batch launch");
        h_ring_inh_step_batch_kernel<<<inh_grid, block>>>(
            n_trials, step, d_pred_iv, d_pred_iie, d_pred_iii, d_pred_ir,
            d_pred_i_total, d_pred_is
        );
        check_cuda(cudaGetLastError(), "stage1 h pred inh batch launch");
        h_ring_e_recurrent_scatter_batch_kernel<<<e_grid, block>>>(
            n_trials, d_ctx_es, d_ctx_eie, d_ctx_eg, d_ctx_iie
        );
        check_cuda(cudaGetLastError(), "stage1 h ctx recurrent batch launch");
        h_ctx_pred_dense_scatter_batch_kernel<<<e_grid, block>>>(
            n_trials, d_ctx_es, d_w, CTX_PRED_GATE_DRIVE_PA, d_pred_eie
        );
        check_cuda(cudaGetLastError(), "stage1 h ctx->pred batch launch");
        h_ring_inh_scatter_batch_kernel<<<inh_grid, block>>>(
            n_trials, d_ctx_is, d_ctx_eii
        );
        check_cuda(cudaGetLastError(), "stage1 h ctx inh scatter batch launch");
        h_ring_e_recurrent_scatter_batch_kernel<<<e_grid, block>>>(
            n_trials, d_pred_es, d_pred_eie, d_pred_eg, d_pred_iie
        );
        check_cuda(cudaGetLastError(), "stage1 h pred recurrent batch launch");
        h_ring_inh_scatter_batch_kernel<<<inh_grid, block>>>(
            n_trials, d_pred_is, d_pred_eii
        );
        check_cuda(cudaGetLastError(), "stage1 h pred inh scatter batch launch");
    }
    check_cuda(cudaDeviceSynchronize(), "stage1 h gate dynamics sync");

    copy_device_to_host(d_ctx_persist, cuda_ctx_persist);
    copy_device_to_host(d_ctx_total, result.cuda_ctx_total_counts);
    copy_device_to_host(d_ctx_i_total, result.cuda_ctx_inh_total_counts);
    copy_device_to_host(d_ctx_last, cuda_ctx_last);
    copy_device_to_host(d_pred_pre, cuda_pred_pre);
    copy_device_to_host(d_pred_total, result.cuda_pred_total_counts);
    copy_device_to_host(d_pred_i_total, result.cuda_pred_inh_total_counts);

    device_free(d_leader_channels); device_free(d_trailer_channels); device_free(d_w);
    device_free(d_ctx_ev); device_free(d_ctx_eie); device_free(d_ctx_eii);
    device_free(d_ctx_eg); device_free(d_ctx_er); device_free(d_ctx_iv);
    device_free(d_ctx_iie); device_free(d_ctx_iii); device_free(d_ctx_ir);
    device_free(d_ctx_es); device_free(d_ctx_is); device_free(d_ctx_leader);
    device_free(d_ctx_persist); device_free(d_ctx_late); device_free(d_ctx_total);
    device_free(d_ctx_i_total); device_free(d_ctx_last);
    device_free(d_pred_ev); device_free(d_pred_eie); device_free(d_pred_eii);
    device_free(d_pred_eg); device_free(d_pred_er); device_free(d_pred_iv);
    device_free(d_pred_iie); device_free(d_pred_iii); device_free(d_pred_ir);
    device_free(d_pred_es); device_free(d_pred_is); device_free(d_pred_leader);
    device_free(d_pred_pre); device_free(d_pred_trailer); device_free(d_pred_total);
    device_free(d_pred_i_total); device_free(d_pred_last);

    for (int trial = 0; trial < n_trials; ++trial) {
        result.cuda_ctx_persistence_ms_by_trial[static_cast<std::size_t>(trial)] =
            cuda_ctx_last[static_cast<std::size_t>(trial)] >= persistence_start
            ? static_cast<double>(
                cuda_ctx_last[static_cast<std::size_t>(trial)] - leader_end
              ) * DT_MS
            : 0.0;
        const int target_channel =
            result.trailer_channels[static_cast<std::size_t>(trial)];
        const int target_start = target_channel * H_E_PER_CHANNEL;
        const std::size_t e_base = static_cast<std::size_t>(trial) * H_E_N;
        int target_pre_count = 0;
        for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
            target_pre_count +=
                cuda_pred_pre[e_base + static_cast<std::size_t>(target_start + k)];
        }
        result.cuda_pred_pretrailer_target_counts[static_cast<std::size_t>(trial)] =
            target_pre_count;
    }

    auto max_flat_rate_hz = [&](const std::vector<std::int32_t>& counts) {
        const double duration_s = static_cast<double>(n_steps) * DT_MS / 1000.0;
        return static_cast<double>(*std::max_element(counts.begin(), counts.end()))
            / duration_s;
    };
    auto max_channel_rate_hz = [&](const std::vector<std::int32_t>& counts) {
        const double duration_s = static_cast<double>(n_steps) * DT_MS / 1000.0;
        double out = 0.0;
        for (int trial = 0; trial < n_trials; ++trial) {
            const std::size_t e_base = static_cast<std::size_t>(trial) * H_E_N;
            for (int channel = 0; channel < H_BROAD_INH_START; ++channel) {
                int total = 0;
                const int start = channel * H_E_PER_CHANNEL;
                for (int k = 0; k < H_E_PER_CHANNEL; ++k) {
                    total += counts[e_base + static_cast<std::size_t>(start + k)];
                }
                out = std::max(
                    out,
                    static_cast<double>(total)
                        / (static_cast<double>(H_E_PER_CHANNEL) * duration_s)
                );
            }
        }
        return out;
    };
    const double mean_ctx_persistence_ms =
        std::accumulate(
            result.cpu_ctx_persistence_ms_by_trial.begin(),
            result.cpu_ctx_persistence_ms_by_trial.end(),
            0.0
        ) / static_cast<double>(n_trials);
    const int forecast_trials = std::count_if(
        result.cpu_pred_pretrailer_target_counts.begin(),
        result.cpu_pred_pretrailer_target_counts.end(),
        [](std::int32_t value) { return value > 0; }
    );
    const double forecast_probability =
        static_cast<double>(forecast_trials) / static_cast<double>(n_trials);
    const double ctx_max_cell_rate = max_flat_rate_hz(result.cpu_ctx_total_counts);
    const double pred_max_cell_rate = max_flat_rate_hz(result.cpu_pred_total_counts);
    const double max_cell_rate = std::max(ctx_max_cell_rate, pred_max_cell_rate);
    const double ctx_max_channel_rate =
        max_channel_rate_hz(result.cpu_ctx_total_counts);
    const double pred_max_channel_rate =
        max_channel_rate_hz(result.cpu_pred_total_counts);
    const double max_channel_rate =
        std::max(ctx_max_channel_rate, pred_max_channel_rate);
    const int ctx_total_spikes = sum_counts(result.cpu_ctx_total_counts);
    const int pred_total_spikes = sum_counts(result.cpu_pred_total_counts);
    const int pred_pre_target_spikes =
        sum_counts(result.cpu_pred_pretrailer_target_counts);
    result.metrics = {
        {"h_context_persistence_ms", mean_ctx_persistence_ms},
        {"h_context_persistence_min_ms", 200.0},
        {"h_context_persistence_max_ms", 500.0},
        {"h_context_persistence_pass",
         (mean_ctx_persistence_ms >= 200.0 && mean_ctx_persistence_ms <= 500.0)
            ? 1.0 : 0.0},
        {"h_prediction_pretrailer_forecast_probability", forecast_probability},
        {"h_prediction_pretrailer_forecast_threshold", 0.25},
        {"h_prediction_pretrailer_forecast_pass",
         forecast_probability >= 0.25 ? 1.0 : 0.0},
        {"no_runaway_max_rate_hz", max_cell_rate},
        {"no_runaway_max_channel_rate_hz", max_channel_rate},
        {"no_runaway_threshold_hz", 80.0},
        {"no_runaway_pass", max_cell_rate <= 80.0 ? 1.0 : 0.0},
        {"ctx_max_cell_rate_hz", ctx_max_cell_rate},
        {"pred_max_cell_rate_hz", pred_max_cell_rate},
        {"ctx_max_channel_rate_hz", ctx_max_channel_rate},
        {"pred_max_channel_rate_hz", pred_max_channel_rate},
        {"ctx_total_spikes", static_cast<double>(ctx_total_spikes)},
        {"pred_total_spikes", static_cast<double>(pred_total_spikes)},
        {"pred_pretrailer_target_spikes", static_cast<double>(pred_pre_target_spikes)},
        {"forecast_trial_count", static_cast<double>(forecast_trials)},
        {"h_prediction_pretrailer_start_step", static_cast<double>(pretrailer_start)},
        {"h_prediction_pretrailer_end_step", static_cast<double>(pretrailer_end)},
        {"n_trials", static_cast<double>(n_trials)},
        {"gate_metric_source_native_h_dynamics", 1.0},
        {"ctx_pred_gate_drive_amp_pA", CTX_PRED_GATE_DRIVE_PA},
        {"h_context_peak_bin_ms", 10.0},
    };
    result.max_abs_error = {
        {
            "ctx_persistence_ms_by_trial",
            max_abs_diff(
                result.cpu_ctx_persistence_ms_by_trial,
                result.cuda_ctx_persistence_ms_by_trial
            ),
        },
        {
            "pred_pretrailer_target_counts",
            max_count_abs_diff(
                result.cpu_pred_pretrailer_target_counts,
                result.cuda_pred_pretrailer_target_counts
            ),
        },
        {
            "ctx_total_counts",
            max_count_abs_diff(
                result.cpu_ctx_total_counts,
                result.cuda_ctx_total_counts
            ),
        },
        {
            "pred_total_counts",
            max_count_abs_diff(
                result.cpu_pred_total_counts,
                result.cuda_pred_total_counts
            ),
        },
        {
            "ctx_inh_total_counts",
            max_count_abs_diff(
                result.cpu_ctx_inh_total_counts,
                result.cuda_ctx_inh_total_counts
            ),
        },
        {
            "pred_inh_total_counts",
            max_count_abs_diff(
                result.cpu_pred_inh_total_counts,
                result.cuda_pred_inh_total_counts
            ),
        },
    };
    return result;
}

CsrScatterPrimitiveResult run_csr_scatter_primitive(
    const std::string& bank_name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight,
    double drive_amp,
    std::int32_t pre_index
) {
    if (pre.empty()) {
        throw std::runtime_error("CSR scatter primitive requires nonempty edge arrays");
    }
    if (pre.size() != post.size() || pre.size() != weight.size()) {
        throw std::runtime_error("CSR scatter primitive edge array shape mismatch");
    }
    const auto max_pre_it = std::max_element(pre.begin(), pre.end());
    const auto max_post_it = std::max_element(post.begin(), post.end());
    const int n_pre = static_cast<int>(*max_pre_it) + 1;
    const int n_target = static_cast<int>(*max_post_it) + 1;
    if (pre_index < 0 || pre_index >= n_pre) {
        throw std::runtime_error("CSR scatter primitive pre_index out of range");
    }

    std::vector<std::int32_t> row_ptr(n_pre + 1, 0);
    for (const auto src : pre) {
        if (src < 0) {
            throw std::runtime_error("CSR scatter primitive negative pre index");
        }
        row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < n_pre; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }
    std::vector<std::int32_t> fill = row_ptr;
    std::vector<std::int32_t> post_csr(post.size());
    std::vector<double> weight_csr(weight.size());
    for (std::size_t edge = 0; edge < pre.size(); ++edge) {
        const int src = pre[edge];
        const int dst_slot = fill[src]++;
        post_csr[dst_slot] = post[edge];
        weight_csr[dst_slot] = weight[edge];
    }

    CsrScatterPrimitiveResult result;
    result.bank_name = bank_name;
    result.pre_index = pre_index;
    result.n_edges = static_cast<std::int32_t>(pre.size());
    result.n_pre = n_pre;
    result.n_target = n_target;
    result.n_edges_for_source = row_ptr[pre_index + 1] - row_ptr[pre_index];
    result.drive_amp = drive_amp;
    result.cpu_target.assign(n_target, 0.0);
    result.cuda_target.assign(n_target, 0.0);

    for (int edge = row_ptr[pre_index]; edge < row_ptr[pre_index + 1]; ++edge) {
        result.cpu_target[post_csr[edge]] += weight_csr[edge] * drive_amp;
    }

    auto* d_row_ptr = device_alloc_copy(row_ptr);
    auto* d_post = device_alloc_copy(post_csr);
    auto* d_weight = device_alloc_copy(weight_csr);
    auto* d_target = device_alloc_copy(result.cuda_target);
    csr_scatter_one_source_kernel<<<1, 256>>>(
        pre_index, d_row_ptr, d_post, d_weight, drive_amp, d_target
    );
    check_cuda(cudaGetLastError(), "csr_scatter_one_source_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "csr_scatter_one_source_kernel sync");
    copy_device_to_host(d_target, result.cuda_target);
    device_free(d_row_ptr);
    device_free(d_post);
    device_free(d_weight);
    device_free(d_target);

    result.max_abs_error = max_abs_diff(result.cpu_target, result.cuda_target);
    return result;
}

EventOrderingSliceResult run_event_ordering_slice(
    const std::string& bank_name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight,
    double drive_amp,
    std::int32_t pre_index
) {
    if (pre.empty()) {
        throw std::runtime_error("event ordering slice requires nonempty edge arrays");
    }
    if (pre.size() != post.size() || pre.size() != weight.size()) {
        throw std::runtime_error("event ordering slice edge array shape mismatch");
    }
    const int n_pre = static_cast<int>(*std::max_element(pre.begin(), pre.end())) + 1;
    const int n_target = static_cast<int>(*std::max_element(post.begin(), post.end())) + 1;
    if (n_target != H_E_N) {
        throw std::runtime_error(
            "event ordering slice currently targets H E and requires 192 targets"
        );
    }
    if (pre_index < 0 || pre_index >= n_pre) {
        throw std::runtime_error("event ordering slice pre_index out of range");
    }

    std::vector<std::int32_t> row_ptr(n_pre + 1, 0);
    for (const auto src : pre) {
        if (src < 0) {
            throw std::runtime_error("event ordering slice negative pre index");
        }
        row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < n_pre; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }
    if (row_ptr[pre_index] == row_ptr[pre_index + 1]) {
        throw std::runtime_error("event ordering slice selected source has no edges");
    }

    std::vector<std::int32_t> fill = row_ptr;
    std::vector<std::int32_t> post_csr(post.size());
    std::vector<double> weight_csr(weight.size());
    for (std::size_t edge = 0; edge < pre.size(); ++edge) {
        const int src = pre[edge];
        const int dst_slot = fill[src]++;
        post_csr[dst_slot] = post[edge];
        weight_csr[dst_slot] = weight[edge];
    }

    EventOrderingSliceResult result;
    result.bank_name = bank_name;
    result.pre_index = pre_index;
    result.target_index = post_csr[row_ptr[pre_index]];
    result.n_edges_for_source = row_ptr[pre_index + 1] - row_ptr[pre_index];
    result.drive_amp = drive_amp;

    std::map<std::string, std::vector<double>> cpu_event;
    std::map<std::string, std::vector<double>> cpu_no_event;
    std::vector<std::int32_t> cpu_event_spikes;
    std::vector<std::int32_t> cpu_no_event_spikes;
    init_he_quiet_state(cpu_event, cpu_event_spikes);
    init_he_quiet_state(cpu_no_event, cpu_no_event_spikes);
    result.cpu_v_initial = cpu_event["V_mV"][result.target_index];

    cpu_step_he(cpu_no_event, cpu_no_event_spikes, 1);
    cpu_step_he(cpu_event, cpu_event_spikes, 1);
    result.cpu_no_event_v_after_step0 = cpu_no_event["V_mV"][result.target_index];
    result.cpu_v_after_step0 = cpu_event["V_mV"][result.target_index];

    result.event_sum_to_target = 0.0;
    for (int edge = row_ptr[pre_index]; edge < row_ptr[pre_index + 1]; ++edge) {
        const double increment = weight_csr[edge] * drive_amp;
        cpu_event["I_e_pA"][post_csr[edge]] += increment;
        if (post_csr[edge] == result.target_index) {
            result.event_sum_to_target += increment;
        }
    }
    result.cpu_i_e_after_scatter = cpu_event["I_e_pA"][result.target_index];

    cpu_step_he(cpu_no_event, cpu_no_event_spikes, 1);
    cpu_step_he(cpu_event, cpu_event_spikes, 1);
    result.cpu_no_event_v_after_step1 = cpu_no_event["V_mV"][result.target_index];
    result.cpu_v_after_step1 = cpu_event["V_mV"][result.target_index];
    result.cpu_total_spikes = std::accumulate(
        cpu_event_spikes.begin(), cpu_event_spikes.end(), 0
    );

    std::map<std::string, std::vector<double>> cuda_state;
    std::vector<std::int32_t> cuda_spikes;
    init_he_quiet_state(cuda_state, cuda_spikes);
    result.cuda_v_initial = cuda_state["V_mV"][result.target_index];

    auto* d_v = device_alloc_copy(cuda_state["V_mV"]);
    auto* d_ie = device_alloc_copy(cuda_state["I_e_pA"]);
    auto* d_ii = device_alloc_copy(cuda_state["I_i_pA"]);
    auto* d_g = device_alloc_copy(cuda_state["g_nmda_h_nS"]);
    auto* d_ibias = device_alloc_copy(cuda_state["I_bias_pA"]);
    auto* d_refrac = device_alloc_copy(cuda_state["refrac_until_ms"]);
    auto* d_spikes = device_alloc_copy(cuda_spikes);
    auto* d_row_ptr = device_alloc_copy(row_ptr);
    auto* d_post = device_alloc_copy(post_csr);
    auto* d_weight = device_alloc_copy(weight_csr);

    he_decay_kernel<<<1, 256>>>(
        1, d_v, d_ie, d_ii, d_g, d_ibias, d_refrac, d_spikes
    );
    check_cuda(cudaGetLastError(), "event ordering step0 launch");
    check_cuda(cudaDeviceSynchronize(), "event ordering step0 sync");
    copy_device_to_host(d_v, cuda_state["V_mV"]);
    result.cuda_v_after_step0 = cuda_state["V_mV"][result.target_index];

    csr_scatter_one_source_kernel<<<1, 256>>>(
        pre_index, d_row_ptr, d_post, d_weight, drive_amp, d_ie
    );
    check_cuda(cudaGetLastError(), "event ordering scatter launch");
    check_cuda(cudaDeviceSynchronize(), "event ordering scatter sync");
    copy_device_to_host(d_ie, cuda_state["I_e_pA"]);
    result.cuda_i_e_after_scatter = cuda_state["I_e_pA"][result.target_index];

    he_decay_kernel<<<1, 256>>>(
        1, d_v, d_ie, d_ii, d_g, d_ibias, d_refrac, d_spikes
    );
    check_cuda(cudaGetLastError(), "event ordering step1 launch");
    check_cuda(cudaDeviceSynchronize(), "event ordering step1 sync");
    copy_device_to_host(d_v, cuda_state["V_mV"]);
    copy_device_to_host(d_spikes, cuda_spikes);
    result.cuda_v_after_step1 = cuda_state["V_mV"][result.target_index];
    result.cuda_total_spikes = std::accumulate(
        cuda_spikes.begin(), cuda_spikes.end(), 0
    );

    device_free(d_v);
    device_free(d_ie);
    device_free(d_ii);
    device_free(d_g);
    device_free(d_ibias);
    device_free(d_refrac);
    device_free(d_spikes);
    device_free(d_row_ptr);
    device_free(d_post);
    device_free(d_weight);

    result.max_abs_error = 0.0;
    result.max_abs_error = std::max(
        result.max_abs_error,
        std::abs(result.cpu_v_initial - result.cuda_v_initial)
    );
    result.max_abs_error = std::max(
        result.max_abs_error,
        std::abs(result.cpu_v_after_step0 - result.cuda_v_after_step0)
    );
    result.max_abs_error = std::max(
        result.max_abs_error,
        std::abs(result.cpu_i_e_after_scatter - result.cuda_i_e_after_scatter)
    );
    result.max_abs_error = std::max(
        result.max_abs_error,
        std::abs(result.cpu_v_after_step1 - result.cuda_v_after_step1)
    );
    return result;
}

CtxToPredCountTestResult run_ctx_to_pred_count_test(
    const std::string& bank_name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight,
    double drive_amp,
    std::int32_t pre_index,
    const std::vector<std::int32_t>& event_steps,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
) {
    if (pre.empty()) {
        throw std::runtime_error("ctx_to_pred count test requires nonempty edges");
    }
    if (pre.size() != post.size() || pre.size() != weight.size()) {
        throw std::runtime_error("ctx_to_pred count test edge shape mismatch");
    }
    if (event_steps.empty()) {
        throw std::runtime_error("ctx_to_pred count test requires event steps");
    }
    if (n_steps < 2) {
        throw std::runtime_error("ctx_to_pred count test requires n_steps >= 2");
    }
    if (window_start_step < 0 || window_end_step <= window_start_step
        || window_end_step >= n_steps) {
        throw std::runtime_error(
            "ctx_to_pred count test requires 0 <= start < end < n_steps"
        );
    }
    const int n_pre = static_cast<int>(*std::max_element(pre.begin(), pre.end())) + 1;
    const int n_target = static_cast<int>(*std::max_element(post.begin(), post.end())) + 1;
    if (n_target != H_E_N) {
        throw std::runtime_error(
            "ctx_to_pred count test currently requires 192 H_pred E targets"
        );
    }
    if (pre_index < 0 || pre_index >= n_pre) {
        throw std::runtime_error("ctx_to_pred count test pre_index out of range");
    }
    const auto first_event_it = std::min_element(event_steps.begin(), event_steps.end());
    const int first_event_step = *first_event_it;
    if (first_event_step < 0 || first_event_step + 1 >= n_steps) {
        throw std::runtime_error(
            "ctx_to_pred count test first event must have a following step"
        );
    }
    for (const int event_step : event_steps) {
        if (event_step < 0 || event_step >= n_steps) {
            throw std::runtime_error("ctx_to_pred count test event step out of range");
        }
    }

    std::vector<std::int32_t> row_ptr(n_pre + 1, 0);
    for (const auto src : pre) {
        if (src < 0) {
            throw std::runtime_error("ctx_to_pred count test negative pre index");
        }
        row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < n_pre; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }
    if (row_ptr[pre_index] == row_ptr[pre_index + 1]) {
        throw std::runtime_error("ctx_to_pred count test selected source has no edges");
    }

    std::vector<std::int32_t> fill = row_ptr;
    std::vector<std::int32_t> post_csr(post.size());
    std::vector<double> weight_csr(weight.size());
    for (std::size_t edge = 0; edge < pre.size(); ++edge) {
        const int src = pre[edge];
        const int dst_slot = fill[src]++;
        post_csr[dst_slot] = post[edge];
        weight_csr[dst_slot] = weight[edge];
    }

    CtxToPredCountTestResult result;
    result.bank_name = bank_name;
    result.pre_index = pre_index;
    result.target_index = post_csr[row_ptr[pre_index]];
    result.n_steps = n_steps;
    result.window_start_step = window_start_step;
    result.window_end_step = window_end_step;
    result.n_edges_for_source = row_ptr[pre_index + 1] - row_ptr[pre_index];
    result.drive_amp = drive_amp;
    result.event_steps = event_steps;
    result.event_sum_to_target = 0.0;
    for (int edge = row_ptr[pre_index]; edge < row_ptr[pre_index + 1]; ++edge) {
        if (post_csr[edge] == result.target_index) {
            result.event_sum_to_target += weight_csr[edge] * drive_amp;
        }
    }

    std::map<std::string, std::vector<double>> cpu_event;
    std::map<std::string, std::vector<double>> cpu_no_event;
    std::vector<std::int32_t> cpu_event_counts(H_E_N, 0);
    std::vector<std::int32_t> cpu_no_event_counts(H_E_N, 0);
    init_he_quiet_state(cpu_event, result.cpu_counts);
    init_he_quiet_state(cpu_no_event, result.cuda_counts);
    result.cpu_counts.assign(H_E_N, 0);
    result.cuda_counts.assign(H_E_N, 0);
    cpu_no_event_counts.assign(H_E_N, 0);
    cpu_event_counts.assign(H_E_N, 0);

    for (int step = 0; step < n_steps; ++step) {
        cpu_force_count_boundary_voltage(
            cpu_no_event, step, window_start_step, window_end_step
        );
        cpu_force_count_boundary_voltage(
            cpu_event, step, window_start_step, window_end_step
        );
        cpu_he_count_step(
            cpu_no_event, cpu_no_event_counts, step,
            window_start_step, window_end_step
        );
        cpu_he_count_step(
            cpu_event, cpu_event_counts, step,
            window_start_step, window_end_step
        );
        if (step == first_event_step) {
            result.cpu_target_v_after_event_step =
                cpu_event["V_mV"][result.target_index];
            result.cpu_no_event_target_v_after_event_step =
                cpu_no_event["V_mV"][result.target_index];
        }
        if (std::find(event_steps.begin(), event_steps.end(), step) != event_steps.end()) {
            for (int edge = row_ptr[pre_index]; edge < row_ptr[pre_index + 1]; ++edge) {
                cpu_event["I_e_pA"][post_csr[edge]] += weight_csr[edge] * drive_amp;
            }
            if (step == first_event_step) {
                result.cpu_target_i_e_after_event_scatter =
                    cpu_event["I_e_pA"][result.target_index];
            }
        }
        if (step == first_event_step + 1) {
            result.cpu_target_v_after_next_step =
                cpu_event["V_mV"][result.target_index];
            result.cpu_no_event_target_v_after_next_step =
                cpu_no_event["V_mV"][result.target_index];
        }
    }
    result.cpu_counts = cpu_event_counts;
    result.cpu_final_state = cpu_event;
    result.cpu_total_window_spikes = std::accumulate(
        result.cpu_counts.begin(), result.cpu_counts.end(), 0
    );

    std::map<std::string, std::vector<double>> cuda_state;
    std::vector<std::int32_t> cuda_dummy_spikes;
    init_he_quiet_state(cuda_state, cuda_dummy_spikes);
    std::vector<std::int32_t> cuda_counts(H_E_N, 0);

    auto* d_v = device_alloc_copy(cuda_state["V_mV"]);
    auto* d_ie = device_alloc_copy(cuda_state["I_e_pA"]);
    auto* d_ii = device_alloc_copy(cuda_state["I_i_pA"]);
    auto* d_g = device_alloc_copy(cuda_state["g_nmda_h_nS"]);
    auto* d_ibias = device_alloc_copy(cuda_state["I_bias_pA"]);
    auto* d_refrac = device_alloc_copy(cuda_state["refrac_until_ms"]);
    auto* d_counts = device_alloc_copy(cuda_counts);
    auto* d_row_ptr = device_alloc_copy(row_ptr);
    auto* d_post = device_alloc_copy(post_csr);
    auto* d_weight = device_alloc_copy(weight_csr);

    for (int step = 0; step < n_steps; ++step) {
        force_count_boundary_voltage_kernel<<<1, 1>>>(
            step, window_start_step, window_end_step, d_v
        );
        check_cuda(cudaGetLastError(), "ctx_to_pred force voltage launch");
        he_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_v, d_ie, d_ii, d_g, d_ibias, d_refrac, d_counts
        );
        check_cuda(cudaGetLastError(), "ctx_to_pred count step launch");
        check_cuda(cudaDeviceSynchronize(), "ctx_to_pred count step sync");
        if (step == first_event_step) {
            copy_device_to_host(d_v, cuda_state["V_mV"]);
            result.cuda_target_v_after_event_step =
                cuda_state["V_mV"][result.target_index];
        }
        if (std::find(event_steps.begin(), event_steps.end(), step) != event_steps.end()) {
            csr_scatter_one_source_kernel<<<1, 256>>>(
                pre_index, d_row_ptr, d_post, d_weight, drive_amp, d_ie
            );
            check_cuda(cudaGetLastError(), "ctx_to_pred count scatter launch");
            check_cuda(cudaDeviceSynchronize(), "ctx_to_pred count scatter sync");
            if (step == first_event_step) {
                copy_device_to_host(d_ie, cuda_state["I_e_pA"]);
                result.cuda_target_i_e_after_event_scatter =
                    cuda_state["I_e_pA"][result.target_index];
            }
        }
        if (step == first_event_step + 1) {
            copy_device_to_host(d_v, cuda_state["V_mV"]);
            result.cuda_target_v_after_next_step =
                cuda_state["V_mV"][result.target_index];
        }
    }

    copy_device_to_host(d_v, cuda_state["V_mV"]);
    copy_device_to_host(d_ie, cuda_state["I_e_pA"]);
    copy_device_to_host(d_ii, cuda_state["I_i_pA"]);
    copy_device_to_host(d_g, cuda_state["g_nmda_h_nS"]);
    copy_device_to_host(d_ibias, cuda_state["I_bias_pA"]);
    copy_device_to_host(d_refrac, cuda_state["refrac_until_ms"]);
    copy_device_to_host(d_counts, cuda_counts);
    result.cuda_counts = cuda_counts;
    result.cuda_final_state = cuda_state;
    result.cuda_total_window_spikes = std::accumulate(
        result.cuda_counts.begin(), result.cuda_counts.end(), 0
    );

    device_free(d_v);
    device_free(d_ie);
    device_free(d_ii);
    device_free(d_g);
    device_free(d_ibias);
    device_free(d_refrac);
    device_free(d_counts);
    device_free(d_row_ptr);
    device_free(d_post);
    device_free(d_weight);

    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    double count_error = 0.0;
    for (int i = 0; i < H_E_N; ++i) {
        count_error = std::max(
            count_error,
            static_cast<double>(std::abs(result.cpu_counts[i] - result.cuda_counts[i]))
        );
    }
    result.max_abs_error["counts"] = count_error;
    result.max_abs_error["target_v_after_event_step"] = std::abs(
        result.cpu_target_v_after_event_step - result.cuda_target_v_after_event_step
    );
    result.max_abs_error["target_i_e_after_event_scatter"] = std::abs(
        result.cpu_target_i_e_after_event_scatter
        - result.cuda_target_i_e_after_event_scatter
    );
    result.max_abs_error["target_v_after_next_step"] = std::abs(
        result.cpu_target_v_after_next_step - result.cuda_target_v_after_next_step
    );
    return result;
}

FeedbackV1CountTestResult run_feedback_v1_count_test(
    const std::string& direct_bank_name,
    const std::vector<std::int32_t>& direct_pre,
    const std::vector<std::int32_t>& direct_post,
    const std::vector<double>& direct_weight,
    double direct_drive_amp,
    const std::string& som_bank_name,
    const std::vector<std::int32_t>& som_pre,
    const std::vector<std::int32_t>& som_post,
    const std::vector<double>& som_weight,
    double som_drive_amp,
    std::int32_t pre_index,
    const std::vector<std::int32_t>& event_steps,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
) {
    if (direct_pre.empty() || som_pre.empty()) {
        throw std::runtime_error("feedback V1 count test requires nonempty edges");
    }
    if (direct_pre.size() != direct_post.size()
        || direct_pre.size() != direct_weight.size()
        || som_pre.size() != som_post.size()
        || som_pre.size() != som_weight.size()) {
        throw std::runtime_error("feedback V1 count test edge shape mismatch");
    }
    if (event_steps.empty()) {
        throw std::runtime_error("feedback V1 count test requires event steps");
    }
    if (n_steps < 3) {
        throw std::runtime_error("feedback V1 count test requires n_steps >= 3");
    }
    if (window_start_step < 0 || window_end_step <= window_start_step
        || window_end_step >= n_steps) {
        throw std::runtime_error(
            "feedback V1 count test requires 0 <= start < end < n_steps"
        );
    }

    const int direct_n_pre =
        static_cast<int>(*std::max_element(direct_pre.begin(), direct_pre.end())) + 1;
    const int direct_n_target =
        static_cast<int>(*std::max_element(direct_post.begin(), direct_post.end())) + 1;
    const int som_n_pre =
        static_cast<int>(*std::max_element(som_pre.begin(), som_pre.end())) + 1;
    const int som_n_target =
        static_cast<int>(*std::max_element(som_post.begin(), som_post.end())) + 1;
    if (direct_n_target != V1_E_N) {
        throw std::runtime_error(
            "feedback direct bank must target 192 V1 E cells"
        );
    }
    if (som_n_target != V1_SOM_N) {
        throw std::runtime_error(
            "feedback SOM bank must target 48 V1 SOM cells"
        );
    }
    if (pre_index < 0 || pre_index >= direct_n_pre || pre_index >= som_n_pre) {
        throw std::runtime_error("feedback V1 count test pre_index out of range");
    }
    const auto first_event_it = std::min_element(event_steps.begin(), event_steps.end());
    const int first_event_step = *first_event_it;
    if (first_event_step < 0 || first_event_step + 2 >= n_steps) {
        throw std::runtime_error(
            "feedback V1 count test first event needs two following steps"
        );
    }
    for (const int event_step : event_steps) {
        if (event_step < 0 || event_step >= n_steps) {
            throw std::runtime_error("feedback V1 count test event step out of range");
        }
    }

    std::vector<std::int32_t> direct_row_ptr(direct_n_pre + 1, 0);
    for (const auto src : direct_pre) {
        if (src < 0) {
            throw std::runtime_error("feedback direct bank negative pre index");
        }
        direct_row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < direct_n_pre; ++i) {
        direct_row_ptr[i + 1] += direct_row_ptr[i];
    }
    std::vector<std::int32_t> direct_fill = direct_row_ptr;
    std::vector<std::int32_t> direct_post_csr(direct_post.size());
    std::vector<double> direct_weight_csr(direct_weight.size());
    for (std::size_t edge = 0; edge < direct_pre.size(); ++edge) {
        const int src = direct_pre[edge];
        const int dst_slot = direct_fill[src]++;
        direct_post_csr[dst_slot] = direct_post[edge];
        direct_weight_csr[dst_slot] = direct_weight[edge];
    }

    std::vector<std::int32_t> som_row_ptr(som_n_pre + 1, 0);
    for (const auto src : som_pre) {
        if (src < 0) {
            throw std::runtime_error("feedback SOM bank negative pre index");
        }
        som_row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < som_n_pre; ++i) {
        som_row_ptr[i + 1] += som_row_ptr[i];
    }
    std::vector<std::int32_t> som_fill = som_row_ptr;
    std::vector<std::int32_t> som_post_csr(som_post.size());
    std::vector<double> som_weight_csr(som_weight.size());
    for (std::size_t edge = 0; edge < som_pre.size(); ++edge) {
        const int src = som_pre[edge];
        const int dst_slot = som_fill[src]++;
        som_post_csr[dst_slot] = som_post[edge];
        som_weight_csr[dst_slot] = som_weight[edge];
    }
    if (direct_row_ptr[pre_index] == direct_row_ptr[pre_index + 1]
        || som_row_ptr[pre_index] == som_row_ptr[pre_index + 1]) {
        throw std::runtime_error(
            "feedback V1 count test selected source has no feedback edges"
        );
    }

    FeedbackV1CountTestResult result;
    result.direct_bank_name = direct_bank_name;
    result.som_bank_name = som_bank_name;
    result.pre_index = pre_index;
    result.target_v1e_index = direct_post_csr[direct_row_ptr[pre_index]];
    result.target_som_index = som_post_csr[som_row_ptr[pre_index]];
    result.n_steps = n_steps;
    result.window_start_step = window_start_step;
    result.window_end_step = window_end_step;
    result.direct_edge_count = static_cast<std::int32_t>(direct_pre.size());
    result.som_edge_count = static_cast<std::int32_t>(som_pre.size());
    result.direct_edges_for_source =
        direct_row_ptr[pre_index + 1] - direct_row_ptr[pre_index];
    result.som_edges_for_source = som_row_ptr[pre_index + 1] - som_row_ptr[pre_index];
    result.direct_drive_amp = direct_drive_amp;
    result.som_drive_amp = som_drive_amp;
    result.event_steps = event_steps;
    result.direct_event_sum_to_target = 0.0;
    result.som_event_sum_to_target = 0.0;
    for (int edge = direct_row_ptr[pre_index]; edge < direct_row_ptr[pre_index + 1]; ++edge) {
        if (direct_post_csr[edge] == result.target_v1e_index) {
            result.direct_event_sum_to_target +=
                direct_weight_csr[edge] * direct_drive_amp;
        }
    }
    for (int edge = som_row_ptr[pre_index]; edge < som_row_ptr[pre_index + 1]; ++edge) {
        if (som_post_csr[edge] == result.target_som_index) {
            result.som_event_sum_to_target += som_weight_csr[edge] * som_drive_amp;
        }
    }

    std::map<std::string, std::vector<double>> cpu_v1e_event;
    std::map<std::string, std::vector<double>> cpu_v1e_no_event;
    std::map<std::string, std::vector<double>> cpu_som_event;
    std::map<std::string, std::vector<double>> cpu_som_no_event;
    std::vector<std::int32_t> cpu_dummy_spikes;
    std::vector<std::int32_t> cpu_event_counts(V1_E_N, 0);
    std::vector<std::int32_t> cpu_no_event_counts(V1_E_N, 0);
    init_v1e_feedback_state(cpu_v1e_event, cpu_dummy_spikes);
    init_v1e_feedback_state(cpu_v1e_no_event, cpu_dummy_spikes);
    init_v1som_feedback_state(cpu_som_event);
    init_v1som_feedback_state(cpu_som_no_event);

    for (int step = 0; step < n_steps; ++step) {
        cpu_force_v1e_count_boundary_voltage(
            cpu_v1e_event, step, window_start_step, window_end_step
        );
        cpu_force_v1e_count_boundary_voltage(
            cpu_v1e_no_event, step, window_start_step, window_end_step
        );
        cpu_v1e_count_step(
            cpu_v1e_event, cpu_event_counts, step,
            window_start_step, window_end_step
        );
        cpu_v1e_count_step(
            cpu_v1e_no_event, cpu_no_event_counts, step,
            window_start_step, window_end_step
        );
        cpu_v1som_step(cpu_som_event, step);
        cpu_v1som_step(cpu_som_no_event, step);
        if (step == first_event_step) {
            result.cpu_v1e_soma_after_event_step =
                cpu_v1e_event["V_soma_mV"][result.target_v1e_index];
            result.cpu_no_event_v1e_soma_after_event_step =
                cpu_v1e_no_event["V_soma_mV"][result.target_v1e_index];
        }
        if (std::find(event_steps.begin(), event_steps.end(), step) != event_steps.end()) {
            for (int edge = direct_row_ptr[pre_index]; edge < direct_row_ptr[pre_index + 1]; ++edge) {
                cpu_v1e_event["I_ap_e_pA"][direct_post_csr[edge]] +=
                    direct_weight_csr[edge] * direct_drive_amp;
            }
            for (int edge = som_row_ptr[pre_index]; edge < som_row_ptr[pre_index + 1]; ++edge) {
                cpu_som_event["I_e_pA"][som_post_csr[edge]] +=
                    som_weight_csr[edge] * som_drive_amp;
            }
            if (step == first_event_step) {
                result.cpu_v1e_i_ap_after_event_scatter =
                    cpu_v1e_event["I_ap_e_pA"][result.target_v1e_index];
                result.cpu_v1som_i_e_after_event_scatter =
                    cpu_som_event["I_e_pA"][result.target_som_index];
            }
        }
        if (step == first_event_step + 1) {
            result.cpu_v1e_ap_after_next_step =
                cpu_v1e_event["V_ap_mV"][result.target_v1e_index];
            result.cpu_no_event_v1e_ap_after_next_step =
                cpu_v1e_no_event["V_ap_mV"][result.target_v1e_index];
            result.cpu_v1som_v_after_next_step =
                cpu_som_event["V_mV"][result.target_som_index];
            result.cpu_no_event_v1som_v_after_next_step =
                cpu_som_no_event["V_mV"][result.target_som_index];
        }
        if (step == first_event_step + 2) {
            result.cpu_v1e_soma_after_late_step =
                cpu_v1e_event["V_soma_mV"][result.target_v1e_index];
            result.cpu_no_event_v1e_soma_after_late_step =
                cpu_v1e_no_event["V_soma_mV"][result.target_v1e_index];
        }
    }
    result.cpu_counts = cpu_event_counts;
    result.cpu_total_window_spikes = std::accumulate(
        result.cpu_counts.begin(), result.cpu_counts.end(), 0
    );
    append_prefixed_state(result.cpu_final_state, "v1e", cpu_v1e_event);
    append_prefixed_state(result.cpu_final_state, "v1som", cpu_som_event);

    std::map<std::string, std::vector<double>> cuda_v1e_state;
    std::map<std::string, std::vector<double>> cuda_som_state;
    std::vector<std::int32_t> cuda_dummy_spikes;
    std::vector<std::int32_t> cuda_counts(V1_E_N, 0);
    init_v1e_feedback_state(cuda_v1e_state, cuda_dummy_spikes);
    init_v1som_feedback_state(cuda_som_state);

    auto* d_v = device_alloc_copy(cuda_v1e_state["V_soma_mV"]);
    auto* d_vap = device_alloc_copy(cuda_v1e_state["V_ap_mV"]);
    auto* d_ie = device_alloc_copy(cuda_v1e_state["I_e_pA"]);
    auto* d_ii = device_alloc_copy(cuda_v1e_state["I_i_pA"]);
    auto* d_iape = device_alloc_copy(cuda_v1e_state["I_ap_e_pA"]);
    auto* d_w = device_alloc_copy(cuda_v1e_state["w_adapt_pA"]);
    auto* d_ibias = device_alloc_copy(cuda_v1e_state["I_bias_pA"]);
    auto* d_refrac = device_alloc_copy(cuda_v1e_state["refrac_until_ms"]);
    auto* d_counts = device_alloc_copy(cuda_counts);
    auto* d_som_v = device_alloc_copy(cuda_som_state["V_mV"]);
    auto* d_som_ie = device_alloc_copy(cuda_som_state["I_e_pA"]);
    auto* d_som_ii = device_alloc_copy(cuda_som_state["I_i_pA"]);
    auto* d_som_ibias = device_alloc_copy(cuda_som_state["I_bias_pA"]);
    auto* d_som_refrac = device_alloc_copy(cuda_som_state["refrac_until_ms"]);
    auto* d_direct_row_ptr = device_alloc_copy(direct_row_ptr);
    auto* d_direct_post = device_alloc_copy(direct_post_csr);
    auto* d_direct_weight = device_alloc_copy(direct_weight_csr);
    auto* d_som_row_ptr = device_alloc_copy(som_row_ptr);
    auto* d_som_post = device_alloc_copy(som_post_csr);
    auto* d_som_weight = device_alloc_copy(som_weight_csr);

    for (int step = 0; step < n_steps; ++step) {
        force_count_boundary_voltage_kernel<<<1, 1>>>(
            step, window_start_step, window_end_step, d_v
        );
        check_cuda(cudaGetLastError(), "feedback V1 force voltage launch");
        v1e_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_v, d_vap, d_ie, d_ii, d_iape, d_w, d_ibias, d_refrac, d_counts
        );
        check_cuda(cudaGetLastError(), "feedback V1 E count step launch");
        v1som_step_kernel<<<1, 64>>>(
            step, d_som_v, d_som_ie, d_som_ii, d_som_ibias, d_som_refrac
        );
        check_cuda(cudaGetLastError(), "feedback V1 SOM step launch");
        check_cuda(cudaDeviceSynchronize(), "feedback V1 step sync");
        if (step == first_event_step) {
            copy_device_to_host(d_v, cuda_v1e_state["V_soma_mV"]);
            result.cuda_v1e_soma_after_event_step =
                cuda_v1e_state["V_soma_mV"][result.target_v1e_index];
        }
        if (std::find(event_steps.begin(), event_steps.end(), step) != event_steps.end()) {
            csr_scatter_one_source_kernel<<<1, 256>>>(
                pre_index, d_direct_row_ptr, d_direct_post, d_direct_weight,
                direct_drive_amp, d_iape
            );
            check_cuda(cudaGetLastError(), "feedback direct scatter launch");
            csr_scatter_one_source_kernel<<<1, 256>>>(
                pre_index, d_som_row_ptr, d_som_post, d_som_weight,
                som_drive_amp, d_som_ie
            );
            check_cuda(cudaGetLastError(), "feedback SOM scatter launch");
            check_cuda(cudaDeviceSynchronize(), "feedback scatter sync");
            if (step == first_event_step) {
                copy_device_to_host(d_iape, cuda_v1e_state["I_ap_e_pA"]);
                copy_device_to_host(d_som_ie, cuda_som_state["I_e_pA"]);
                result.cuda_v1e_i_ap_after_event_scatter =
                    cuda_v1e_state["I_ap_e_pA"][result.target_v1e_index];
                result.cuda_v1som_i_e_after_event_scatter =
                    cuda_som_state["I_e_pA"][result.target_som_index];
            }
        }
        if (step == first_event_step + 1) {
            copy_device_to_host(d_vap, cuda_v1e_state["V_ap_mV"]);
            copy_device_to_host(d_som_v, cuda_som_state["V_mV"]);
            result.cuda_v1e_ap_after_next_step =
                cuda_v1e_state["V_ap_mV"][result.target_v1e_index];
            result.cuda_v1som_v_after_next_step =
                cuda_som_state["V_mV"][result.target_som_index];
        }
        if (step == first_event_step + 2) {
            copy_device_to_host(d_v, cuda_v1e_state["V_soma_mV"]);
            result.cuda_v1e_soma_after_late_step =
                cuda_v1e_state["V_soma_mV"][result.target_v1e_index];
        }
    }

    copy_device_to_host(d_v, cuda_v1e_state["V_soma_mV"]);
    copy_device_to_host(d_vap, cuda_v1e_state["V_ap_mV"]);
    copy_device_to_host(d_ie, cuda_v1e_state["I_e_pA"]);
    copy_device_to_host(d_ii, cuda_v1e_state["I_i_pA"]);
    copy_device_to_host(d_iape, cuda_v1e_state["I_ap_e_pA"]);
    copy_device_to_host(d_w, cuda_v1e_state["w_adapt_pA"]);
    copy_device_to_host(d_ibias, cuda_v1e_state["I_bias_pA"]);
    copy_device_to_host(d_refrac, cuda_v1e_state["refrac_until_ms"]);
    copy_device_to_host(d_counts, cuda_counts);
    copy_device_to_host(d_som_v, cuda_som_state["V_mV"]);
    copy_device_to_host(d_som_ie, cuda_som_state["I_e_pA"]);
    copy_device_to_host(d_som_ii, cuda_som_state["I_i_pA"]);
    copy_device_to_host(d_som_ibias, cuda_som_state["I_bias_pA"]);
    copy_device_to_host(d_som_refrac, cuda_som_state["refrac_until_ms"]);
    result.cuda_counts = cuda_counts;
    result.cuda_total_window_spikes = std::accumulate(
        result.cuda_counts.begin(), result.cuda_counts.end(), 0
    );
    append_prefixed_state(result.cuda_final_state, "v1e", cuda_v1e_state);
    append_prefixed_state(result.cuda_final_state, "v1som", cuda_som_state);

    device_free(d_v);
    device_free(d_vap);
    device_free(d_ie);
    device_free(d_ii);
    device_free(d_iape);
    device_free(d_w);
    device_free(d_ibias);
    device_free(d_refrac);
    device_free(d_counts);
    device_free(d_som_v);
    device_free(d_som_ie);
    device_free(d_som_ii);
    device_free(d_som_ibias);
    device_free(d_som_refrac);
    device_free(d_direct_row_ptr);
    device_free(d_direct_post);
    device_free(d_direct_weight);
    device_free(d_som_row_ptr);
    device_free(d_som_post);
    device_free(d_som_weight);

    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    double count_error = 0.0;
    for (int i = 0; i < V1_E_N; ++i) {
        count_error = std::max(
            count_error,
            static_cast<double>(std::abs(result.cpu_counts[i] - result.cuda_counts[i]))
        );
    }
    result.max_abs_error["counts"] = count_error;
    result.max_abs_error["v1e_soma_after_event_step"] = std::abs(
        result.cpu_v1e_soma_after_event_step
        - result.cuda_v1e_soma_after_event_step
    );
    result.max_abs_error["v1e_i_ap_after_event_scatter"] = std::abs(
        result.cpu_v1e_i_ap_after_event_scatter
        - result.cuda_v1e_i_ap_after_event_scatter
    );
    result.max_abs_error["v1som_i_e_after_event_scatter"] = std::abs(
        result.cpu_v1som_i_e_after_event_scatter
        - result.cuda_v1som_i_e_after_event_scatter
    );
    result.max_abs_error["v1e_ap_after_next_step"] = std::abs(
        result.cpu_v1e_ap_after_next_step - result.cuda_v1e_ap_after_next_step
    );
    result.max_abs_error["v1som_v_after_next_step"] = std::abs(
        result.cpu_v1som_v_after_next_step
        - result.cuda_v1som_v_after_next_step
    );
    result.max_abs_error["v1e_soma_after_late_step"] = std::abs(
        result.cpu_v1e_soma_after_late_step
        - result.cuda_v1e_soma_after_late_step
    );
    return result;
}

V1StimFeedforwardCountTestResult run_v1_stim_feedforward_count_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::string& feedforward_bank_name,
    const std::vector<std::int32_t>& feedforward_pre,
    const std::vector<std::int32_t>& feedforward_post,
    const std::vector<double>& feedforward_weight,
    double feedforward_drive_amp,
    std::int32_t stim_pre_index,
    const std::vector<std::int32_t>& stim_event_steps,
    std::int32_t force_v1e_step,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
) {
    if (stim_pre.empty() || feedforward_pre.empty()) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test requires nonempty edges"
        );
    }
    if (stim_pre.size() != stim_post.size() || stim_pre.size() != stim_weight.size()
        || feedforward_pre.size() != feedforward_post.size()
        || feedforward_pre.size() != feedforward_weight.size()) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test edge shape mismatch"
        );
    }
    if (stim_event_steps.empty()) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test requires stimulus events"
        );
    }
    if (n_steps < 4) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test requires n_steps >= 4"
        );
    }
    if (window_start_step < 0 || window_end_step <= window_start_step
        || window_end_step >= n_steps) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test requires 0 <= start < end < n_steps"
        );
    }
    const auto first_stim_it =
        std::min_element(stim_event_steps.begin(), stim_event_steps.end());
    const int first_stim_step = *first_stim_it;
    if (first_stim_step < 0 || first_stim_step + 1 >= n_steps) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test first stimulus needs a following step"
        );
    }
    if (force_v1e_step < 0 || force_v1e_step + 1 >= n_steps) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test forced V1 spike needs a following step"
        );
    }
    for (const int event_step : stim_event_steps) {
        if (event_step < 0 || event_step >= n_steps) {
            throw std::runtime_error(
                "V1 stimulus/feedforward count test stimulus step out of range"
            );
        }
    }

    const int stim_n_pre =
        static_cast<int>(*std::max_element(stim_pre.begin(), stim_pre.end())) + 1;
    const int stim_n_target =
        static_cast<int>(*std::max_element(stim_post.begin(), stim_post.end())) + 1;
    const int ff_n_pre =
        static_cast<int>(*std::max_element(feedforward_pre.begin(), feedforward_pre.end())) + 1;
    const int ff_n_target =
        static_cast<int>(*std::max_element(feedforward_post.begin(), feedforward_post.end())) + 1;
    if (stim_n_target != V1_E_N) {
        throw std::runtime_error(
            "stimulus bank must target 192 V1 E cells"
        );
    }
    if (ff_n_target != H_E_N) {
        throw std::runtime_error(
            "V1->H bank must target 192 H_ctx E cells"
        );
    }
    if (stim_pre_index < 0 || stim_pre_index >= stim_n_pre) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test stim_pre_index out of range"
        );
    }

    std::vector<std::int32_t> stim_row_ptr(stim_n_pre + 1, 0);
    for (const auto src : stim_pre) {
        if (src < 0) {
            throw std::runtime_error("stimulus bank negative pre index");
        }
        stim_row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < stim_n_pre; ++i) {
        stim_row_ptr[i + 1] += stim_row_ptr[i];
    }
    if (stim_row_ptr[stim_pre_index] == stim_row_ptr[stim_pre_index + 1]) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test selected stimulus source has no edges"
        );
    }
    std::vector<std::int32_t> stim_fill = stim_row_ptr;
    std::vector<std::int32_t> stim_post_csr(stim_post.size());
    std::vector<double> stim_weight_csr(stim_weight.size());
    for (std::size_t edge = 0; edge < stim_pre.size(); ++edge) {
        const int src = stim_pre[edge];
        const int dst_slot = stim_fill[src]++;
        stim_post_csr[dst_slot] = stim_post[edge];
        stim_weight_csr[dst_slot] = stim_weight[edge];
    }

    const int forced_v1e_index = stim_post_csr[stim_row_ptr[stim_pre_index]];
    if (forced_v1e_index < 0 || forced_v1e_index >= ff_n_pre) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test forced V1 source absent from V1->H bank"
        );
    }

    std::vector<std::int32_t> ff_row_ptr(ff_n_pre + 1, 0);
    for (const auto src : feedforward_pre) {
        if (src < 0) {
            throw std::runtime_error("V1->H bank negative pre index");
        }
        ff_row_ptr[static_cast<std::size_t>(src) + 1] += 1;
    }
    for (int i = 0; i < ff_n_pre; ++i) {
        ff_row_ptr[i + 1] += ff_row_ptr[i];
    }
    if (ff_row_ptr[forced_v1e_index] == ff_row_ptr[forced_v1e_index + 1]) {
        throw std::runtime_error(
            "V1 stimulus/feedforward count test forced V1 source has no H edges"
        );
    }
    std::vector<std::int32_t> ff_fill = ff_row_ptr;
    std::vector<std::int32_t> ff_post_csr(feedforward_post.size());
    std::vector<double> ff_weight_csr(feedforward_weight.size());
    for (std::size_t edge = 0; edge < feedforward_pre.size(); ++edge) {
        const int src = feedforward_pre[edge];
        const int dst_slot = ff_fill[src]++;
        ff_post_csr[dst_slot] = feedforward_post[edge];
        ff_weight_csr[dst_slot] = feedforward_weight[edge];
    }

    V1StimFeedforwardCountTestResult result;
    result.stim_bank_name = stim_bank_name;
    result.feedforward_bank_name = feedforward_bank_name;
    result.stim_pre_index = stim_pre_index;
    result.forced_v1e_index = forced_v1e_index;
    result.target_h_index = ff_post_csr[ff_row_ptr[forced_v1e_index]];
    result.n_steps = n_steps;
    result.window_start_step = window_start_step;
    result.window_end_step = window_end_step;
    result.force_v1e_step = force_v1e_step;
    result.stim_edge_count = static_cast<std::int32_t>(stim_pre.size());
    result.feedforward_edge_count = static_cast<std::int32_t>(feedforward_pre.size());
    result.stim_edges_for_source =
        stim_row_ptr[stim_pre_index + 1] - stim_row_ptr[stim_pre_index];
    result.feedforward_edges_for_source =
        ff_row_ptr[forced_v1e_index + 1] - ff_row_ptr[forced_v1e_index];
    result.stim_drive_amp = stim_drive_amp;
    result.feedforward_drive_amp = feedforward_drive_amp;
    result.stim_event_steps = stim_event_steps;
    result.stim_event_sum_to_v1e_target = 0.0;
    for (int edge = stim_row_ptr[stim_pre_index]; edge < stim_row_ptr[stim_pre_index + 1]; ++edge) {
        if (stim_post_csr[edge] == forced_v1e_index) {
            result.stim_event_sum_to_v1e_target +=
                stim_weight_csr[edge] * stim_drive_amp;
        }
    }
    result.feedforward_event_sum_to_h_target = 0.0;
    for (int edge = ff_row_ptr[forced_v1e_index]; edge < ff_row_ptr[forced_v1e_index + 1]; ++edge) {
        if (ff_post_csr[edge] == result.target_h_index) {
            result.feedforward_event_sum_to_h_target +=
                ff_weight_csr[edge] * feedforward_drive_amp;
        }
    }

    std::map<std::string, std::vector<double>> cpu_v1_event;
    std::map<std::string, std::vector<double>> cpu_v1_no_stim;
    std::map<std::string, std::vector<double>> cpu_h_event;
    std::map<std::string, std::vector<double>> cpu_h_no_ff;
    std::vector<std::int32_t> cpu_dummy_spikes;
    std::vector<std::int32_t> cpu_v1_counts(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_no_stim_counts(V1_E_N, 0);
    std::vector<std::int32_t> cpu_h_counts(H_E_N, 0);
    std::vector<std::int32_t> cpu_h_no_ff_counts(H_E_N, 0);
    init_v1e_feedback_state(cpu_v1_event, cpu_dummy_spikes);
    init_v1e_feedback_state(cpu_v1_no_stim, cpu_dummy_spikes);
    init_he_quiet_state(cpu_h_event, cpu_dummy_spikes);
    init_he_quiet_state(cpu_h_no_ff, cpu_dummy_spikes);

    for (int step = 0; step < n_steps; ++step) {
        if (step == force_v1e_step) {
            cpu_v1_event["V_soma_mV"][forced_v1e_index] = -49.0;
            cpu_v1_no_stim["V_soma_mV"][forced_v1e_index] = -49.0;
        }
        cpu_force_count_boundary_voltage(
            cpu_h_event, step, window_start_step, window_end_step
        );
        cpu_force_count_boundary_voltage(
            cpu_h_no_ff, step, window_start_step, window_end_step
        );
        cpu_v1e_count_step(
            cpu_v1_event, cpu_v1_counts, step, window_start_step, window_end_step
        );
        cpu_v1e_count_step(
            cpu_v1_no_stim, cpu_v1_no_stim_counts,
            step, window_start_step, window_end_step
        );
        cpu_he_count_step(
            cpu_h_event, cpu_h_counts, step, window_start_step, window_end_step
        );
        cpu_he_count_step(
            cpu_h_no_ff, cpu_h_no_ff_counts,
            step, window_start_step, window_end_step
        );
        if (step == first_stim_step) {
            result.cpu_v1e_soma_after_stim_step =
                cpu_v1_event["V_soma_mV"][forced_v1e_index];
            result.cpu_no_stim_v1e_soma_after_stim_step =
                cpu_v1_no_stim["V_soma_mV"][forced_v1e_index];
        }
        if (std::find(stim_event_steps.begin(), stim_event_steps.end(), step)
            != stim_event_steps.end()) {
            for (int edge = stim_row_ptr[stim_pre_index]; edge < stim_row_ptr[stim_pre_index + 1]; ++edge) {
                cpu_v1_event["I_e_pA"][stim_post_csr[edge]] +=
                    stim_weight_csr[edge] * stim_drive_amp;
            }
            if (step == first_stim_step) {
                result.cpu_v1e_i_e_after_stim_scatter =
                    cpu_v1_event["I_e_pA"][forced_v1e_index];
            }
        }
        if (step == first_stim_step + 1) {
            result.cpu_v1e_soma_after_stim_next_step =
                cpu_v1_event["V_soma_mV"][forced_v1e_index];
            result.cpu_no_stim_v1e_soma_after_stim_next_step =
                cpu_v1_no_stim["V_soma_mV"][forced_v1e_index];
        }
        if (step == force_v1e_step) {
            result.cpu_h_v_after_force_v1e_step =
                cpu_h_event["V_mV"][result.target_h_index];
            result.cpu_no_ff_h_v_after_force_v1e_step =
                cpu_h_no_ff["V_mV"][result.target_h_index];
            for (int edge = ff_row_ptr[forced_v1e_index]; edge < ff_row_ptr[forced_v1e_index + 1]; ++edge) {
                cpu_h_event["I_e_pA"][ff_post_csr[edge]] +=
                    ff_weight_csr[edge] * feedforward_drive_amp;
            }
            result.cpu_h_i_e_after_ff_scatter =
                cpu_h_event["I_e_pA"][result.target_h_index];
        }
        if (step == force_v1e_step + 1) {
            result.cpu_h_v_after_ff_next_step =
                cpu_h_event["V_mV"][result.target_h_index];
            result.cpu_no_ff_h_v_after_ff_next_step =
                cpu_h_no_ff["V_mV"][result.target_h_index];
        }
    }
    result.cpu_v1_counts = cpu_v1_counts;
    result.cpu_h_counts = cpu_h_counts;
    result.cpu_total_v1_window_spikes = std::accumulate(
        result.cpu_v1_counts.begin(), result.cpu_v1_counts.end(), 0
    );
    result.cpu_total_h_window_spikes = std::accumulate(
        result.cpu_h_counts.begin(), result.cpu_h_counts.end(), 0
    );
    append_prefixed_state(result.cpu_final_state, "v1e", cpu_v1_event);
    append_prefixed_state(result.cpu_final_state, "hctx", cpu_h_event);

    std::map<std::string, std::vector<double>> cuda_v1_state;
    std::map<std::string, std::vector<double>> cuda_h_state;
    std::vector<std::int32_t> cuda_dummy_spikes;
    std::vector<std::int32_t> cuda_v1_counts(V1_E_N, 0);
    std::vector<std::int32_t> cuda_h_counts(H_E_N, 0);
    init_v1e_feedback_state(cuda_v1_state, cuda_dummy_spikes);
    init_he_quiet_state(cuda_h_state, cuda_dummy_spikes);

    auto* d_v1_v = device_alloc_copy(cuda_v1_state["V_soma_mV"]);
    auto* d_v1_vap = device_alloc_copy(cuda_v1_state["V_ap_mV"]);
    auto* d_v1_ie = device_alloc_copy(cuda_v1_state["I_e_pA"]);
    auto* d_v1_ii = device_alloc_copy(cuda_v1_state["I_i_pA"]);
    auto* d_v1_iape = device_alloc_copy(cuda_v1_state["I_ap_e_pA"]);
    auto* d_v1_w = device_alloc_copy(cuda_v1_state["w_adapt_pA"]);
    auto* d_v1_ibias = device_alloc_copy(cuda_v1_state["I_bias_pA"]);
    auto* d_v1_refrac = device_alloc_copy(cuda_v1_state["refrac_until_ms"]);
    auto* d_v1_counts = device_alloc_copy(cuda_v1_counts);
    auto* d_h_v = device_alloc_copy(cuda_h_state["V_mV"]);
    auto* d_h_ie = device_alloc_copy(cuda_h_state["I_e_pA"]);
    auto* d_h_ii = device_alloc_copy(cuda_h_state["I_i_pA"]);
    auto* d_h_g = device_alloc_copy(cuda_h_state["g_nmda_h_nS"]);
    auto* d_h_ibias = device_alloc_copy(cuda_h_state["I_bias_pA"]);
    auto* d_h_refrac = device_alloc_copy(cuda_h_state["refrac_until_ms"]);
    auto* d_h_counts = device_alloc_copy(cuda_h_counts);
    auto* d_stim_row_ptr = device_alloc_copy(stim_row_ptr);
    auto* d_stim_post = device_alloc_copy(stim_post_csr);
    auto* d_stim_weight = device_alloc_copy(stim_weight_csr);
    auto* d_ff_row_ptr = device_alloc_copy(ff_row_ptr);
    auto* d_ff_post = device_alloc_copy(ff_post_csr);
    auto* d_ff_weight = device_alloc_copy(ff_weight_csr);

    for (int step = 0; step < n_steps; ++step) {
        if (step == force_v1e_step) {
            force_single_voltage_kernel<<<1, 1>>>(
                forced_v1e_index, -49.0, d_v1_v
            );
            check_cuda(cudaGetLastError(), "V1 stimulus/feedforward force V1 launch");
        }
        force_count_boundary_voltage_kernel<<<1, 1>>>(
            step, window_start_step, window_end_step, d_h_v
        );
        check_cuda(cudaGetLastError(), "V1 stimulus/feedforward force H launch");
        v1e_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_v1_v, d_v1_vap, d_v1_ie, d_v1_ii, d_v1_iape,
            d_v1_w, d_v1_ibias, d_v1_refrac, d_v1_counts
        );
        check_cuda(cudaGetLastError(), "V1 stimulus/feedforward V1 step launch");
        he_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_h_v, d_h_ie, d_h_ii, d_h_g, d_h_ibias, d_h_refrac, d_h_counts
        );
        check_cuda(cudaGetLastError(), "V1 stimulus/feedforward H step launch");
        check_cuda(cudaDeviceSynchronize(), "V1 stimulus/feedforward step sync");
        if (step == first_stim_step) {
            copy_device_to_host(d_v1_v, cuda_v1_state["V_soma_mV"]);
            result.cuda_v1e_soma_after_stim_step =
                cuda_v1_state["V_soma_mV"][forced_v1e_index];
        }
        if (std::find(stim_event_steps.begin(), stim_event_steps.end(), step)
            != stim_event_steps.end()) {
            csr_scatter_one_source_kernel<<<1, 256>>>(
                stim_pre_index, d_stim_row_ptr, d_stim_post, d_stim_weight,
                stim_drive_amp, d_v1_ie
            );
            check_cuda(cudaGetLastError(), "V1 stimulus scatter launch");
            check_cuda(cudaDeviceSynchronize(), "V1 stimulus scatter sync");
            if (step == first_stim_step) {
                copy_device_to_host(d_v1_ie, cuda_v1_state["I_e_pA"]);
                result.cuda_v1e_i_e_after_stim_scatter =
                    cuda_v1_state["I_e_pA"][forced_v1e_index];
            }
        }
        if (step == first_stim_step + 1) {
            copy_device_to_host(d_v1_v, cuda_v1_state["V_soma_mV"]);
            result.cuda_v1e_soma_after_stim_next_step =
                cuda_v1_state["V_soma_mV"][forced_v1e_index];
        }
        if (step == force_v1e_step) {
            copy_device_to_host(d_h_v, cuda_h_state["V_mV"]);
            result.cuda_h_v_after_force_v1e_step =
                cuda_h_state["V_mV"][result.target_h_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                forced_v1e_index, d_ff_row_ptr, d_ff_post, d_ff_weight,
                feedforward_drive_amp, d_h_ie
            );
            check_cuda(cudaGetLastError(), "V1->H feedforward scatter launch");
            check_cuda(cudaDeviceSynchronize(), "V1->H feedforward scatter sync");
            copy_device_to_host(d_h_ie, cuda_h_state["I_e_pA"]);
            result.cuda_h_i_e_after_ff_scatter =
                cuda_h_state["I_e_pA"][result.target_h_index];
        }
        if (step == force_v1e_step + 1) {
            copy_device_to_host(d_h_v, cuda_h_state["V_mV"]);
            result.cuda_h_v_after_ff_next_step =
                cuda_h_state["V_mV"][result.target_h_index];
        }
    }

    copy_device_to_host(d_v1_v, cuda_v1_state["V_soma_mV"]);
    copy_device_to_host(d_v1_vap, cuda_v1_state["V_ap_mV"]);
    copy_device_to_host(d_v1_ie, cuda_v1_state["I_e_pA"]);
    copy_device_to_host(d_v1_ii, cuda_v1_state["I_i_pA"]);
    copy_device_to_host(d_v1_iape, cuda_v1_state["I_ap_e_pA"]);
    copy_device_to_host(d_v1_w, cuda_v1_state["w_adapt_pA"]);
    copy_device_to_host(d_v1_ibias, cuda_v1_state["I_bias_pA"]);
    copy_device_to_host(d_v1_refrac, cuda_v1_state["refrac_until_ms"]);
    copy_device_to_host(d_v1_counts, cuda_v1_counts);
    copy_device_to_host(d_h_v, cuda_h_state["V_mV"]);
    copy_device_to_host(d_h_ie, cuda_h_state["I_e_pA"]);
    copy_device_to_host(d_h_ii, cuda_h_state["I_i_pA"]);
    copy_device_to_host(d_h_g, cuda_h_state["g_nmda_h_nS"]);
    copy_device_to_host(d_h_ibias, cuda_h_state["I_bias_pA"]);
    copy_device_to_host(d_h_refrac, cuda_h_state["refrac_until_ms"]);
    copy_device_to_host(d_h_counts, cuda_h_counts);
    result.cuda_v1_counts = cuda_v1_counts;
    result.cuda_h_counts = cuda_h_counts;
    result.cuda_total_v1_window_spikes = std::accumulate(
        result.cuda_v1_counts.begin(), result.cuda_v1_counts.end(), 0
    );
    result.cuda_total_h_window_spikes = std::accumulate(
        result.cuda_h_counts.begin(), result.cuda_h_counts.end(), 0
    );
    append_prefixed_state(result.cuda_final_state, "v1e", cuda_v1_state);
    append_prefixed_state(result.cuda_final_state, "hctx", cuda_h_state);

    device_free(d_v1_v);
    device_free(d_v1_vap);
    device_free(d_v1_ie);
    device_free(d_v1_ii);
    device_free(d_v1_iape);
    device_free(d_v1_w);
    device_free(d_v1_ibias);
    device_free(d_v1_refrac);
    device_free(d_v1_counts);
    device_free(d_h_v);
    device_free(d_h_ie);
    device_free(d_h_ii);
    device_free(d_h_g);
    device_free(d_h_ibias);
    device_free(d_h_refrac);
    device_free(d_h_counts);
    device_free(d_stim_row_ptr);
    device_free(d_stim_post);
    device_free(d_stim_weight);
    device_free(d_ff_row_ptr);
    device_free(d_ff_post);
    device_free(d_ff_weight);

    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    double v1_count_error = 0.0;
    double h_count_error = 0.0;
    for (int i = 0; i < V1_E_N; ++i) {
        v1_count_error = std::max(
            v1_count_error,
            static_cast<double>(std::abs(result.cpu_v1_counts[i] - result.cuda_v1_counts[i]))
        );
    }
    for (int i = 0; i < H_E_N; ++i) {
        h_count_error = std::max(
            h_count_error,
            static_cast<double>(std::abs(result.cpu_h_counts[i] - result.cuda_h_counts[i]))
        );
    }
    result.max_abs_error["v1_counts"] = v1_count_error;
    result.max_abs_error["h_counts"] = h_count_error;
    result.max_abs_error["v1e_soma_after_stim_step"] = std::abs(
        result.cpu_v1e_soma_after_stim_step
        - result.cuda_v1e_soma_after_stim_step
    );
    result.max_abs_error["v1e_i_e_after_stim_scatter"] = std::abs(
        result.cpu_v1e_i_e_after_stim_scatter
        - result.cuda_v1e_i_e_after_stim_scatter
    );
    result.max_abs_error["v1e_soma_after_stim_next_step"] = std::abs(
        result.cpu_v1e_soma_after_stim_next_step
        - result.cuda_v1e_soma_after_stim_next_step
    );
    result.max_abs_error["h_v_after_force_v1e_step"] = std::abs(
        result.cpu_h_v_after_force_v1e_step
        - result.cuda_h_v_after_force_v1e_step
    );
    result.max_abs_error["h_i_e_after_ff_scatter"] = std::abs(
        result.cpu_h_i_e_after_ff_scatter
        - result.cuda_h_i_e_after_ff_scatter
    );
    result.max_abs_error["h_v_after_ff_next_step"] = std::abs(
        result.cpu_h_v_after_ff_next_step
        - result.cuda_h_v_after_ff_next_step
    );
    return result;
}

ClosedLoopDeterministicCountTestResult run_closed_loop_deterministic_count_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    std::int32_t stim_pre_index,
    std::int32_t stim_step,
    std::int32_t v1_force_step,
    std::int32_t hctx_force_step,
    std::int32_t hpred_force_step,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
) {
    if (n_steps < 4 || hpred_force_step + 2 >= n_steps) {
        throw std::runtime_error(
            "closed-loop deterministic test requires hpred_force_step + 2 < n_steps"
        );
    }
    if (window_start_step < 0 || window_end_step <= window_start_step
        || window_end_step >= n_steps) {
        throw std::runtime_error(
            "closed-loop deterministic test requires 0 <= start < end < n_steps"
        );
    }
    if (stim_step < 0 || stim_step + 1 >= n_steps
        || v1_force_step < 0 || v1_force_step + 1 >= n_steps
        || hctx_force_step < 0 || hctx_force_step + 1 >= n_steps
        || hpred_force_step < 0 || hpred_force_step + 2 >= n_steps) {
        throw std::runtime_error("closed-loop deterministic test step out of range");
    }

    const CsrBank stim_bank = build_csr_bank(
        stim_bank_name, stim_pre, stim_post, stim_weight
    );
    const CsrBank v1_to_h_bank = build_csr_bank(
        v1_to_h_bank_name, v1_to_h_pre, v1_to_h_post, v1_to_h_weight
    );
    const CsrBank ctx_to_pred_bank = build_csr_bank(
        ctx_to_pred_bank_name, ctx_to_pred_pre, ctx_to_pred_post,
        ctx_to_pred_weight
    );
    const CsrBank fb_direct_bank = build_csr_bank(
        feedback_direct_bank_name, feedback_direct_pre, feedback_direct_post,
        feedback_direct_weight
    );
    const CsrBank fb_som_bank = build_csr_bank(
        feedback_som_bank_name, feedback_som_pre, feedback_som_post,
        feedback_som_weight
    );
    if (stim_bank.n_target != V1_E_N
        || v1_to_h_bank.n_target != H_E_N
        || ctx_to_pred_bank.n_target != H_E_N
        || fb_direct_bank.n_target != V1_E_N
        || fb_som_bank.n_target != V1_SOM_N) {
        throw std::runtime_error("closed-loop deterministic test target size mismatch");
    }
    if (csr_fanout(stim_bank, stim_pre_index) <= 0) {
        throw std::runtime_error("closed-loop deterministic test stimulus source empty");
    }

    ClosedLoopDeterministicCountTestResult result;
    result.n_steps = n_steps;
    result.window_start_step = window_start_step;
    result.window_end_step = window_end_step;
    result.stim_step = stim_step;
    result.v1_force_step = v1_force_step;
    result.hctx_force_step = hctx_force_step;
    result.hpred_force_step = hpred_force_step;
    result.stim_pre_index = stim_pre_index;
    result.v1e_index = stim_bank.post[stim_bank.row_ptr[stim_pre_index]];
    if (result.v1e_index < 0 || result.v1e_index >= v1_to_h_bank.n_pre
        || csr_fanout(v1_to_h_bank, result.v1e_index) <= 0) {
        throw std::runtime_error("closed-loop deterministic test V1 source empty");
    }
    result.hctx_index = v1_to_h_bank.post[v1_to_h_bank.row_ptr[result.v1e_index]];
    if (result.hctx_index < 0 || result.hctx_index >= ctx_to_pred_bank.n_pre
        || csr_fanout(ctx_to_pred_bank, result.hctx_index) <= 0) {
        throw std::runtime_error("closed-loop deterministic test H_ctx source empty");
    }
    result.hpred_index = ctx_to_pred_bank.post[
        ctx_to_pred_bank.row_ptr[result.hctx_index]
    ];
    if (result.hpred_index < 0 || result.hpred_index >= fb_direct_bank.n_pre
        || result.hpred_index >= fb_som_bank.n_pre
        || csr_fanout(fb_direct_bank, result.hpred_index) <= 0
        || csr_fanout(fb_som_bank, result.hpred_index) <= 0) {
        throw std::runtime_error("closed-loop deterministic test H_pred source empty");
    }
    result.feedback_v1e_index = fb_direct_bank.post[
        fb_direct_bank.row_ptr[result.hpred_index]
    ];
    result.feedback_som_index = fb_som_bank.post[
        fb_som_bank.row_ptr[result.hpred_index]
    ];

    result.edge_counts = {
        {"v1_stim_to_e", static_cast<std::int32_t>(stim_pre.size())},
        {"v1_to_h_ctx", static_cast<std::int32_t>(v1_to_h_pre.size())},
        {"ctx_to_pred", static_cast<std::int32_t>(ctx_to_pred_pre.size())},
        {"fb_pred_to_v1e_apical", static_cast<std::int32_t>(feedback_direct_pre.size())},
        {"fb_pred_to_v1som", static_cast<std::int32_t>(feedback_som_pre.size())},
    };
    result.source_fanouts = {
        {"v1_stim_to_e", csr_fanout(stim_bank, stim_pre_index)},
        {"v1_to_h_ctx", csr_fanout(v1_to_h_bank, result.v1e_index)},
        {"ctx_to_pred", csr_fanout(ctx_to_pred_bank, result.hctx_index)},
        {"fb_pred_to_v1e_apical", csr_fanout(fb_direct_bank, result.hpred_index)},
        {"fb_pred_to_v1som", csr_fanout(fb_som_bank, result.hpred_index)},
    };
    result.drive_amps = {
        {"v1_stim_to_e", stim_drive_amp},
        {"v1_to_h_ctx", v1_to_h_drive_amp},
        {"ctx_to_pred", ctx_to_pred_drive_amp},
        {"fb_pred_to_v1e_apical", feedback_direct_drive_amp},
        {"fb_pred_to_v1som", feedback_som_drive_amp},
    };
    result.event_sums = {
        {"v1_stim_to_e", csr_sum_to_target(
            stim_bank, stim_pre_index, result.v1e_index, stim_drive_amp
        )},
        {"v1_to_h_ctx", csr_sum_to_target(
            v1_to_h_bank, result.v1e_index, result.hctx_index,
            v1_to_h_drive_amp
        )},
        {"ctx_to_pred", csr_sum_to_target(
            ctx_to_pred_bank, result.hctx_index, result.hpred_index,
            ctx_to_pred_drive_amp
        )},
        {"fb_pred_to_v1e_apical", csr_sum_to_target(
            fb_direct_bank, result.hpred_index, result.feedback_v1e_index,
            feedback_direct_drive_amp
        )},
        {"fb_pred_to_v1som", csr_sum_to_target(
            fb_som_bank, result.hpred_index, result.feedback_som_index,
            feedback_som_drive_amp
        )},
    };

    std::map<std::string, std::vector<double>> cpu_v1;
    std::map<std::string, std::vector<double>> cpu_v1_no_stim;
    std::map<std::string, std::vector<double>> cpu_v1_no_fb;
    std::map<std::string, std::vector<double>> cpu_som;
    std::map<std::string, std::vector<double>> cpu_som_no_fb;
    std::map<std::string, std::vector<double>> cpu_hctx;
    std::map<std::string, std::vector<double>> cpu_hctx_no_v1;
    std::map<std::string, std::vector<double>> cpu_hpred;
    std::map<std::string, std::vector<double>> cpu_hpred_no_ctx;
    std::vector<std::int32_t> cpu_dummy_spikes;
    std::vector<std::int32_t> cpu_v1_counts(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_no_counts(V1_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_counts(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_no_counts(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_counts(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_no_counts(H_E_N, 0);
    init_v1e_feedback_state(cpu_v1, cpu_dummy_spikes);
    init_v1e_feedback_state(cpu_v1_no_stim, cpu_dummy_spikes);
    init_v1e_feedback_state(cpu_v1_no_fb, cpu_dummy_spikes);
    init_v1som_feedback_state(cpu_som);
    init_v1som_feedback_state(cpu_som_no_fb);
    init_he_quiet_state(cpu_hctx, cpu_dummy_spikes);
    init_he_quiet_state(cpu_hctx_no_v1, cpu_dummy_spikes);
    init_he_quiet_state(cpu_hpred, cpu_dummy_spikes);
    init_he_quiet_state(cpu_hpred_no_ctx, cpu_dummy_spikes);

    for (int step = 0; step < n_steps; ++step) {
        if (step == v1_force_step) {
            cpu_v1["V_soma_mV"][result.v1e_index] = -49.0;
            cpu_v1_no_stim["V_soma_mV"][result.v1e_index] = -49.0;
            cpu_v1_no_fb["V_soma_mV"][result.v1e_index] = -49.0;
        }
        if (step == hctx_force_step) {
            cpu_hctx["V_mV"][result.hctx_index] = -49.0;
            cpu_hctx_no_v1["V_mV"][result.hctx_index] = -49.0;
        }
        if (step == hpred_force_step) {
            cpu_hpred["V_mV"][result.hpred_index] = -49.0;
            cpu_hpred_no_ctx["V_mV"][result.hpred_index] = -49.0;
        }

        cpu_v1e_count_step(cpu_v1, cpu_v1_counts, step, window_start_step, window_end_step);
        cpu_v1e_count_step(
            cpu_v1_no_stim, cpu_v1_no_counts, step, window_start_step, window_end_step
        );
        cpu_v1e_count_step(
            cpu_v1_no_fb, cpu_v1_no_counts, step, window_start_step, window_end_step
        );
        cpu_v1som_step(cpu_som, step);
        cpu_v1som_step(cpu_som_no_fb, step);
        cpu_he_count_step(cpu_hctx, cpu_hctx_counts, step, window_start_step, window_end_step);
        cpu_he_count_step(
            cpu_hctx_no_v1, cpu_hctx_no_counts, step, window_start_step,
            window_end_step
        );
        cpu_he_count_step(cpu_hpred, cpu_hpred_counts, step, window_start_step, window_end_step);
        cpu_he_count_step(
            cpu_hpred_no_ctx, cpu_hpred_no_counts, step, window_start_step,
            window_end_step
        );

        if (step == stim_step) {
            result.cpu_v1_soma_after_stim_step =
                cpu_v1["V_soma_mV"][result.v1e_index];
            result.cpu_no_stim_v1_soma_after_stim_step =
                cpu_v1_no_stim["V_soma_mV"][result.v1e_index];
            cpu_scatter_one_source(stim_bank, stim_pre_index, stim_drive_amp, cpu_v1["I_e_pA"]);
            cpu_scatter_one_source(
                stim_bank, stim_pre_index, stim_drive_amp, cpu_v1_no_fb["I_e_pA"]
            );
            result.cpu_v1_i_e_after_stim_scatter =
                cpu_v1["I_e_pA"][result.v1e_index];
        }
        if (step == stim_step + 1) {
            result.cpu_v1_soma_after_stim_next_step =
                cpu_v1["V_soma_mV"][result.v1e_index];
            result.cpu_no_stim_v1_soma_after_stim_next_step =
                cpu_v1_no_stim["V_soma_mV"][result.v1e_index];
        }
        if (step == v1_force_step) {
            result.cpu_hctx_v_after_v1_step =
                cpu_hctx["V_mV"][result.hctx_index];
            result.cpu_no_v1_hctx_v_after_v1_step =
                cpu_hctx_no_v1["V_mV"][result.hctx_index];
            cpu_scatter_one_source(
                v1_to_h_bank, result.v1e_index, v1_to_h_drive_amp,
                cpu_hctx["I_e_pA"]
            );
            result.cpu_hctx_i_e_after_v1_scatter =
                cpu_hctx["I_e_pA"][result.hctx_index];
        }
        if (step == v1_force_step + 1) {
            result.cpu_hctx_v_after_v1_next_step =
                cpu_hctx["V_mV"][result.hctx_index];
            result.cpu_no_v1_hctx_v_after_v1_next_step =
                cpu_hctx_no_v1["V_mV"][result.hctx_index];
        }
        if (step == hctx_force_step) {
            result.cpu_hpred_v_after_hctx_step =
                cpu_hpred["V_mV"][result.hpred_index];
            result.cpu_no_ctx_hpred_v_after_hctx_step =
                cpu_hpred_no_ctx["V_mV"][result.hpred_index];
            cpu_scatter_one_source(
                ctx_to_pred_bank, result.hctx_index, ctx_to_pred_drive_amp,
                cpu_hpred["I_e_pA"]
            );
            result.cpu_hpred_i_e_after_hctx_scatter =
                cpu_hpred["I_e_pA"][result.hpred_index];
        }
        if (step == hctx_force_step + 1) {
            result.cpu_hpred_v_after_hctx_next_step =
                cpu_hpred["V_mV"][result.hpred_index];
            result.cpu_no_ctx_hpred_v_after_hctx_next_step =
                cpu_hpred_no_ctx["V_mV"][result.hpred_index];
        }
        if (step == hpred_force_step) {
            result.cpu_v1_soma_after_hpred_step =
                cpu_v1["V_soma_mV"][result.feedback_v1e_index];
            result.cpu_no_fb_v1_soma_after_hpred_step =
                cpu_v1_no_fb["V_soma_mV"][result.feedback_v1e_index];
            cpu_scatter_one_source(
                fb_direct_bank, result.hpred_index, feedback_direct_drive_amp,
                cpu_v1["I_ap_e_pA"]
            );
            cpu_scatter_one_source(
                fb_som_bank, result.hpred_index, feedback_som_drive_amp,
                cpu_som["I_e_pA"]
            );
            result.cpu_v1_i_ap_after_fb_scatter =
                cpu_v1["I_ap_e_pA"][result.feedback_v1e_index];
            result.cpu_som_i_e_after_fb_scatter =
                cpu_som["I_e_pA"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 1) {
            result.cpu_v1_ap_after_fb_next_step =
                cpu_v1["V_ap_mV"][result.feedback_v1e_index];
            result.cpu_no_fb_v1_ap_after_fb_next_step =
                cpu_v1_no_fb["V_ap_mV"][result.feedback_v1e_index];
            result.cpu_som_v_after_fb_next_step =
                cpu_som["V_mV"][result.feedback_som_index];
            result.cpu_no_fb_som_v_after_fb_next_step =
                cpu_som_no_fb["V_mV"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 2) {
            result.cpu_v1_soma_after_fb_late_step =
                cpu_v1["V_soma_mV"][result.feedback_v1e_index];
            result.cpu_no_fb_v1_soma_after_fb_late_step =
                cpu_v1_no_fb["V_soma_mV"][result.feedback_v1e_index];
        }
    }

    result.cpu_v1_counts = cpu_v1_counts;
    result.cpu_hctx_counts = cpu_hctx_counts;
    result.cpu_hpred_counts = cpu_hpred_counts;
    result.cpu_total_v1_window_spikes = std::accumulate(
        result.cpu_v1_counts.begin(), result.cpu_v1_counts.end(), 0
    );
    result.cpu_total_hctx_window_spikes = std::accumulate(
        result.cpu_hctx_counts.begin(), result.cpu_hctx_counts.end(), 0
    );
    result.cpu_total_hpred_window_spikes = std::accumulate(
        result.cpu_hpred_counts.begin(), result.cpu_hpred_counts.end(), 0
    );
    append_prefixed_state(result.cpu_final_state, "v1e", cpu_v1);
    append_prefixed_state(result.cpu_final_state, "v1som", cpu_som);
    append_prefixed_state(result.cpu_final_state, "hctx", cpu_hctx);
    append_prefixed_state(result.cpu_final_state, "hpred", cpu_hpred);

    std::map<std::string, std::vector<double>> cuda_v1;
    std::map<std::string, std::vector<double>> cuda_som;
    std::map<std::string, std::vector<double>> cuda_hctx;
    std::map<std::string, std::vector<double>> cuda_hpred;
    std::vector<std::int32_t> cuda_dummy_spikes;
    std::vector<std::int32_t> cuda_v1_counts(V1_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_counts(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_counts(H_E_N, 0);
    init_v1e_feedback_state(cuda_v1, cuda_dummy_spikes);
    init_v1som_feedback_state(cuda_som);
    init_he_quiet_state(cuda_hctx, cuda_dummy_spikes);
    init_he_quiet_state(cuda_hpred, cuda_dummy_spikes);

    auto* d_v1_v = device_alloc_copy(cuda_v1["V_soma_mV"]);
    auto* d_v1_vap = device_alloc_copy(cuda_v1["V_ap_mV"]);
    auto* d_v1_ie = device_alloc_copy(cuda_v1["I_e_pA"]);
    auto* d_v1_ii = device_alloc_copy(cuda_v1["I_i_pA"]);
    auto* d_v1_iape = device_alloc_copy(cuda_v1["I_ap_e_pA"]);
    auto* d_v1_w = device_alloc_copy(cuda_v1["w_adapt_pA"]);
    auto* d_v1_ibias = device_alloc_copy(cuda_v1["I_bias_pA"]);
    auto* d_v1_refrac = device_alloc_copy(cuda_v1["refrac_until_ms"]);
    auto* d_v1_counts = device_alloc_copy(cuda_v1_counts);
    auto* d_som_v = device_alloc_copy(cuda_som["V_mV"]);
    auto* d_som_ie = device_alloc_copy(cuda_som["I_e_pA"]);
    auto* d_som_ii = device_alloc_copy(cuda_som["I_i_pA"]);
    auto* d_som_ibias = device_alloc_copy(cuda_som["I_bias_pA"]);
    auto* d_som_refrac = device_alloc_copy(cuda_som["refrac_until_ms"]);
    auto* d_hctx_v = device_alloc_copy(cuda_hctx["V_mV"]);
    auto* d_hctx_ie = device_alloc_copy(cuda_hctx["I_e_pA"]);
    auto* d_hctx_ii = device_alloc_copy(cuda_hctx["I_i_pA"]);
    auto* d_hctx_g = device_alloc_copy(cuda_hctx["g_nmda_h_nS"]);
    auto* d_hctx_ibias = device_alloc_copy(cuda_hctx["I_bias_pA"]);
    auto* d_hctx_refrac = device_alloc_copy(cuda_hctx["refrac_until_ms"]);
    auto* d_hctx_counts = device_alloc_copy(cuda_hctx_counts);
    auto* d_hpred_v = device_alloc_copy(cuda_hpred["V_mV"]);
    auto* d_hpred_ie = device_alloc_copy(cuda_hpred["I_e_pA"]);
    auto* d_hpred_ii = device_alloc_copy(cuda_hpred["I_i_pA"]);
    auto* d_hpred_g = device_alloc_copy(cuda_hpred["g_nmda_h_nS"]);
    auto* d_hpred_ibias = device_alloc_copy(cuda_hpred["I_bias_pA"]);
    auto* d_hpred_refrac = device_alloc_copy(cuda_hpred["refrac_until_ms"]);
    auto* d_hpred_counts = device_alloc_copy(cuda_hpred_counts);
    auto* d_stim_row_ptr = device_alloc_copy(stim_bank.row_ptr);
    auto* d_stim_post = device_alloc_copy(stim_bank.post);
    auto* d_stim_weight = device_alloc_copy(stim_bank.weight);
    auto* d_v1_to_h_row_ptr = device_alloc_copy(v1_to_h_bank.row_ptr);
    auto* d_v1_to_h_post = device_alloc_copy(v1_to_h_bank.post);
    auto* d_v1_to_h_weight = device_alloc_copy(v1_to_h_bank.weight);
    auto* d_ctx_to_pred_row_ptr = device_alloc_copy(ctx_to_pred_bank.row_ptr);
    auto* d_ctx_to_pred_post = device_alloc_copy(ctx_to_pred_bank.post);
    auto* d_ctx_to_pred_weight = device_alloc_copy(ctx_to_pred_bank.weight);
    auto* d_fb_direct_row_ptr = device_alloc_copy(fb_direct_bank.row_ptr);
    auto* d_fb_direct_post = device_alloc_copy(fb_direct_bank.post);
    auto* d_fb_direct_weight = device_alloc_copy(fb_direct_bank.weight);
    auto* d_fb_som_row_ptr = device_alloc_copy(fb_som_bank.row_ptr);
    auto* d_fb_som_post = device_alloc_copy(fb_som_bank.post);
    auto* d_fb_som_weight = device_alloc_copy(fb_som_bank.weight);

    for (int step = 0; step < n_steps; ++step) {
        if (step == v1_force_step) {
            force_single_voltage_kernel<<<1, 1>>>(result.v1e_index, -49.0, d_v1_v);
            check_cuda(cudaGetLastError(), "closed loop force V1 launch");
        }
        if (step == hctx_force_step) {
            force_single_voltage_kernel<<<1, 1>>>(result.hctx_index, -49.0, d_hctx_v);
            check_cuda(cudaGetLastError(), "closed loop force H_ctx launch");
        }
        if (step == hpred_force_step) {
            force_single_voltage_kernel<<<1, 1>>>(result.hpred_index, -49.0, d_hpred_v);
            check_cuda(cudaGetLastError(), "closed loop force H_pred launch");
        }
        v1e_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_v1_v, d_v1_vap, d_v1_ie, d_v1_ii, d_v1_iape, d_v1_w,
            d_v1_ibias, d_v1_refrac, d_v1_counts
        );
        check_cuda(cudaGetLastError(), "closed loop V1 step launch");
        v1som_step_kernel<<<1, 64>>>(
            step, d_som_v, d_som_ie, d_som_ii, d_som_ibias, d_som_refrac
        );
        check_cuda(cudaGetLastError(), "closed loop SOM step launch");
        he_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_hctx_v, d_hctx_ie, d_hctx_ii, d_hctx_g, d_hctx_ibias,
            d_hctx_refrac, d_hctx_counts
        );
        check_cuda(cudaGetLastError(), "closed loop H_ctx step launch");
        he_count_step_kernel<<<1, 256>>>(
            step, window_start_step, window_end_step,
            d_hpred_v, d_hpred_ie, d_hpred_ii, d_hpred_g, d_hpred_ibias,
            d_hpred_refrac, d_hpred_counts
        );
        check_cuda(cudaGetLastError(), "closed loop H_pred step launch");
        check_cuda(cudaDeviceSynchronize(), "closed loop step sync");

        if (step == stim_step) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            result.cuda_v1_soma_after_stim_step =
                cuda_v1["V_soma_mV"][result.v1e_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                stim_pre_index, d_stim_row_ptr, d_stim_post, d_stim_weight,
                stim_drive_amp, d_v1_ie
            );
            check_cuda(cudaGetLastError(), "closed loop stimulus scatter launch");
            check_cuda(cudaDeviceSynchronize(), "closed loop stimulus scatter sync");
            copy_device_to_host(d_v1_ie, cuda_v1["I_e_pA"]);
            result.cuda_v1_i_e_after_stim_scatter =
                cuda_v1["I_e_pA"][result.v1e_index];
        }
        if (step == stim_step + 1) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            result.cuda_v1_soma_after_stim_next_step =
                cuda_v1["V_soma_mV"][result.v1e_index];
        }
        if (step == v1_force_step) {
            copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
            result.cuda_hctx_v_after_v1_step =
                cuda_hctx["V_mV"][result.hctx_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.v1e_index, d_v1_to_h_row_ptr, d_v1_to_h_post,
                d_v1_to_h_weight, v1_to_h_drive_amp, d_hctx_ie
            );
            check_cuda(cudaGetLastError(), "closed loop V1->H scatter launch");
            check_cuda(cudaDeviceSynchronize(), "closed loop V1->H scatter sync");
            copy_device_to_host(d_hctx_ie, cuda_hctx["I_e_pA"]);
            result.cuda_hctx_i_e_after_v1_scatter =
                cuda_hctx["I_e_pA"][result.hctx_index];
        }
        if (step == v1_force_step + 1) {
            copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
            result.cuda_hctx_v_after_v1_next_step =
                cuda_hctx["V_mV"][result.hctx_index];
        }
        if (step == hctx_force_step) {
            copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
            result.cuda_hpred_v_after_hctx_step =
                cuda_hpred["V_mV"][result.hpred_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.hctx_index, d_ctx_to_pred_row_ptr, d_ctx_to_pred_post,
                d_ctx_to_pred_weight, ctx_to_pred_drive_amp, d_hpred_ie
            );
            check_cuda(cudaGetLastError(), "closed loop ctx->pred scatter launch");
            check_cuda(cudaDeviceSynchronize(), "closed loop ctx->pred scatter sync");
            copy_device_to_host(d_hpred_ie, cuda_hpred["I_e_pA"]);
            result.cuda_hpred_i_e_after_hctx_scatter =
                cuda_hpred["I_e_pA"][result.hpred_index];
        }
        if (step == hctx_force_step + 1) {
            copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
            result.cuda_hpred_v_after_hctx_next_step =
                cuda_hpred["V_mV"][result.hpred_index];
        }
        if (step == hpred_force_step) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            result.cuda_v1_soma_after_hpred_step =
                cuda_v1["V_soma_mV"][result.feedback_v1e_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.hpred_index, d_fb_direct_row_ptr, d_fb_direct_post,
                d_fb_direct_weight, feedback_direct_drive_amp, d_v1_iape
            );
            check_cuda(cudaGetLastError(), "closed loop feedback direct scatter launch");
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.hpred_index, d_fb_som_row_ptr, d_fb_som_post,
                d_fb_som_weight, feedback_som_drive_amp, d_som_ie
            );
            check_cuda(cudaGetLastError(), "closed loop feedback SOM scatter launch");
            check_cuda(cudaDeviceSynchronize(), "closed loop feedback scatter sync");
            copy_device_to_host(d_v1_iape, cuda_v1["I_ap_e_pA"]);
            copy_device_to_host(d_som_ie, cuda_som["I_e_pA"]);
            result.cuda_v1_i_ap_after_fb_scatter =
                cuda_v1["I_ap_e_pA"][result.feedback_v1e_index];
            result.cuda_som_i_e_after_fb_scatter =
                cuda_som["I_e_pA"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 1) {
            copy_device_to_host(d_v1_vap, cuda_v1["V_ap_mV"]);
            copy_device_to_host(d_som_v, cuda_som["V_mV"]);
            result.cuda_v1_ap_after_fb_next_step =
                cuda_v1["V_ap_mV"][result.feedback_v1e_index];
            result.cuda_som_v_after_fb_next_step =
                cuda_som["V_mV"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 2) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            result.cuda_v1_soma_after_fb_late_step =
                cuda_v1["V_soma_mV"][result.feedback_v1e_index];
        }
    }

    copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
    copy_device_to_host(d_v1_vap, cuda_v1["V_ap_mV"]);
    copy_device_to_host(d_v1_ie, cuda_v1["I_e_pA"]);
    copy_device_to_host(d_v1_ii, cuda_v1["I_i_pA"]);
    copy_device_to_host(d_v1_iape, cuda_v1["I_ap_e_pA"]);
    copy_device_to_host(d_v1_w, cuda_v1["w_adapt_pA"]);
    copy_device_to_host(d_v1_ibias, cuda_v1["I_bias_pA"]);
    copy_device_to_host(d_v1_refrac, cuda_v1["refrac_until_ms"]);
    copy_device_to_host(d_v1_counts, cuda_v1_counts);
    copy_device_to_host(d_som_v, cuda_som["V_mV"]);
    copy_device_to_host(d_som_ie, cuda_som["I_e_pA"]);
    copy_device_to_host(d_som_ii, cuda_som["I_i_pA"]);
    copy_device_to_host(d_som_ibias, cuda_som["I_bias_pA"]);
    copy_device_to_host(d_som_refrac, cuda_som["refrac_until_ms"]);
    copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
    copy_device_to_host(d_hctx_ie, cuda_hctx["I_e_pA"]);
    copy_device_to_host(d_hctx_ii, cuda_hctx["I_i_pA"]);
    copy_device_to_host(d_hctx_g, cuda_hctx["g_nmda_h_nS"]);
    copy_device_to_host(d_hctx_ibias, cuda_hctx["I_bias_pA"]);
    copy_device_to_host(d_hctx_refrac, cuda_hctx["refrac_until_ms"]);
    copy_device_to_host(d_hctx_counts, cuda_hctx_counts);
    copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
    copy_device_to_host(d_hpred_ie, cuda_hpred["I_e_pA"]);
    copy_device_to_host(d_hpred_ii, cuda_hpred["I_i_pA"]);
    copy_device_to_host(d_hpred_g, cuda_hpred["g_nmda_h_nS"]);
    copy_device_to_host(d_hpred_ibias, cuda_hpred["I_bias_pA"]);
    copy_device_to_host(d_hpred_refrac, cuda_hpred["refrac_until_ms"]);
    copy_device_to_host(d_hpred_counts, cuda_hpred_counts);

    result.cuda_v1_counts = cuda_v1_counts;
    result.cuda_hctx_counts = cuda_hctx_counts;
    result.cuda_hpred_counts = cuda_hpred_counts;
    result.cuda_total_v1_window_spikes = std::accumulate(
        result.cuda_v1_counts.begin(), result.cuda_v1_counts.end(), 0
    );
    result.cuda_total_hctx_window_spikes = std::accumulate(
        result.cuda_hctx_counts.begin(), result.cuda_hctx_counts.end(), 0
    );
    result.cuda_total_hpred_window_spikes = std::accumulate(
        result.cuda_hpred_counts.begin(), result.cuda_hpred_counts.end(), 0
    );
    append_prefixed_state(result.cuda_final_state, "v1e", cuda_v1);
    append_prefixed_state(result.cuda_final_state, "v1som", cuda_som);
    append_prefixed_state(result.cuda_final_state, "hctx", cuda_hctx);
    append_prefixed_state(result.cuda_final_state, "hpred", cuda_hpred);

    device_free(d_v1_v); device_free(d_v1_vap); device_free(d_v1_ie);
    device_free(d_v1_ii); device_free(d_v1_iape); device_free(d_v1_w);
    device_free(d_v1_ibias); device_free(d_v1_refrac); device_free(d_v1_counts);
    device_free(d_som_v); device_free(d_som_ie); device_free(d_som_ii);
    device_free(d_som_ibias); device_free(d_som_refrac);
    device_free(d_hctx_v); device_free(d_hctx_ie); device_free(d_hctx_ii);
    device_free(d_hctx_g); device_free(d_hctx_ibias);
    device_free(d_hctx_refrac); device_free(d_hctx_counts);
    device_free(d_hpred_v); device_free(d_hpred_ie); device_free(d_hpred_ii);
    device_free(d_hpred_g); device_free(d_hpred_ibias);
    device_free(d_hpred_refrac); device_free(d_hpred_counts);
    device_free(d_stim_row_ptr); device_free(d_stim_post); device_free(d_stim_weight);
    device_free(d_v1_to_h_row_ptr); device_free(d_v1_to_h_post);
    device_free(d_v1_to_h_weight);
    device_free(d_ctx_to_pred_row_ptr); device_free(d_ctx_to_pred_post);
    device_free(d_ctx_to_pred_weight);
    device_free(d_fb_direct_row_ptr); device_free(d_fb_direct_post);
    device_free(d_fb_direct_weight);
    device_free(d_fb_som_row_ptr); device_free(d_fb_som_post);
    device_free(d_fb_som_weight);

    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    double v1_count_error = 0.0;
    double hctx_count_error = 0.0;
    double hpred_count_error = 0.0;
    for (int i = 0; i < V1_E_N; ++i) {
        v1_count_error = std::max(
            v1_count_error,
            static_cast<double>(std::abs(result.cpu_v1_counts[i] - result.cuda_v1_counts[i]))
        );
    }
    for (int i = 0; i < H_E_N; ++i) {
        hctx_count_error = std::max(
            hctx_count_error,
            static_cast<double>(
                std::abs(result.cpu_hctx_counts[i] - result.cuda_hctx_counts[i])
            )
        );
        hpred_count_error = std::max(
            hpred_count_error,
            static_cast<double>(
                std::abs(result.cpu_hpred_counts[i] - result.cuda_hpred_counts[i])
            )
        );
    }
    result.max_abs_error["v1_counts"] = v1_count_error;
    result.max_abs_error["hctx_counts"] = hctx_count_error;
    result.max_abs_error["hpred_counts"] = hpred_count_error;
    result.max_abs_error["v1_soma_after_stim_step"] = std::abs(
        result.cpu_v1_soma_after_stim_step - result.cuda_v1_soma_after_stim_step
    );
    result.max_abs_error["v1_i_e_after_stim_scatter"] = std::abs(
        result.cpu_v1_i_e_after_stim_scatter
        - result.cuda_v1_i_e_after_stim_scatter
    );
    result.max_abs_error["v1_soma_after_stim_next_step"] = std::abs(
        result.cpu_v1_soma_after_stim_next_step
        - result.cuda_v1_soma_after_stim_next_step
    );
    result.max_abs_error["hctx_v_after_v1_step"] = std::abs(
        result.cpu_hctx_v_after_v1_step - result.cuda_hctx_v_after_v1_step
    );
    result.max_abs_error["hctx_i_e_after_v1_scatter"] = std::abs(
        result.cpu_hctx_i_e_after_v1_scatter
        - result.cuda_hctx_i_e_after_v1_scatter
    );
    result.max_abs_error["hctx_v_after_v1_next_step"] = std::abs(
        result.cpu_hctx_v_after_v1_next_step
        - result.cuda_hctx_v_after_v1_next_step
    );
    result.max_abs_error["hpred_v_after_hctx_step"] = std::abs(
        result.cpu_hpred_v_after_hctx_step
        - result.cuda_hpred_v_after_hctx_step
    );
    result.max_abs_error["hpred_i_e_after_hctx_scatter"] = std::abs(
        result.cpu_hpred_i_e_after_hctx_scatter
        - result.cuda_hpred_i_e_after_hctx_scatter
    );
    result.max_abs_error["hpred_v_after_hctx_next_step"] = std::abs(
        result.cpu_hpred_v_after_hctx_next_step
        - result.cuda_hpred_v_after_hctx_next_step
    );
    result.max_abs_error["v1_soma_after_hpred_step"] = std::abs(
        result.cpu_v1_soma_after_hpred_step
        - result.cuda_v1_soma_after_hpred_step
    );
    result.max_abs_error["v1_i_ap_after_fb_scatter"] = std::abs(
        result.cpu_v1_i_ap_after_fb_scatter
        - result.cuda_v1_i_ap_after_fb_scatter
    );
    result.max_abs_error["som_i_e_after_fb_scatter"] = std::abs(
        result.cpu_som_i_e_after_fb_scatter
        - result.cuda_som_i_e_after_fb_scatter
    );
    result.max_abs_error["v1_ap_after_fb_next_step"] = std::abs(
        result.cpu_v1_ap_after_fb_next_step
        - result.cuda_v1_ap_after_fb_next_step
    );
    result.max_abs_error["som_v_after_fb_next_step"] = std::abs(
        result.cpu_som_v_after_fb_next_step
        - result.cuda_som_v_after_fb_next_step
    );
    result.max_abs_error["v1_soma_after_fb_late_step"] = std::abs(
        result.cpu_v1_soma_after_fb_late_step
        - result.cuda_v1_soma_after_fb_late_step
    );
    return result;
}

FrozenRichterDeterministicTrialResult run_frozen_richter_deterministic_trial_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    std::int32_t expected_stim_pre_index,
    std::int32_t unexpected_stim_pre_index,
    std::int32_t stim_period_steps,
    std::int32_t n_steps,
    std::int32_t leader_start_step,
    std::int32_t leader_end_step,
    std::int32_t preprobe_start_step,
    std::int32_t preprobe_end_step,
    std::int32_t trailer_start_step,
    std::int32_t trailer_end_step,
    std::int32_t iti_start_step,
    std::int32_t iti_end_step
) {
    constexpr int stim_offset_steps = 2;
    if (stim_period_steps <= 0) {
        throw std::runtime_error("frozen Richter deterministic test period must be > 0");
    }
    if (leader_start_step < 0
        || leader_start_step >= leader_end_step
        || leader_end_step > preprobe_start_step
        || preprobe_start_step >= preprobe_end_step
        || preprobe_end_step > trailer_start_step
        || trailer_start_step >= trailer_end_step
        || trailer_end_step > iti_start_step
        || iti_start_step >= iti_end_step
        || iti_end_step != n_steps) {
        throw std::runtime_error(
            "frozen Richter deterministic test requires ordered [start,end) phases"
        );
    }
    if (preprobe_start_step <= 0) {
        throw std::runtime_error(
            "frozen Richter deterministic test requires preprobe_start_step > 0"
        );
    }
    const int first_stim_step = leader_start_step + stim_offset_steps;
    const int v1_force_step = preprobe_start_step + 4;
    const int hctx_force_step = trailer_start_step + 6;
    const int hpred_force_step = trailer_start_step + 28;
    if (first_stim_step < leader_start_step || first_stim_step + 1 >= n_steps
        || v1_force_step + 1 >= n_steps
        || hctx_force_step + 1 >= n_steps
        || hpred_force_step + 2 >= n_steps) {
        throw std::runtime_error(
            "frozen Richter deterministic test probe steps are out of range"
        );
    }

    const CsrBank stim_bank = build_csr_bank(
        stim_bank_name, stim_pre, stim_post, stim_weight
    );
    const CsrBank v1_to_h_bank = build_csr_bank(
        v1_to_h_bank_name, v1_to_h_pre, v1_to_h_post, v1_to_h_weight
    );
    const CsrBank ctx_to_pred_bank = build_csr_bank(
        ctx_to_pred_bank_name, ctx_to_pred_pre, ctx_to_pred_post,
        ctx_to_pred_weight
    );
    const CsrBank fb_direct_bank = build_csr_bank(
        feedback_direct_bank_name, feedback_direct_pre, feedback_direct_post,
        feedback_direct_weight
    );
    const CsrBank fb_som_bank = build_csr_bank(
        feedback_som_bank_name, feedback_som_pre, feedback_som_post,
        feedback_som_weight
    );
    if (stim_bank.n_target != V1_E_N
        || v1_to_h_bank.n_target != H_E_N
        || ctx_to_pred_bank.n_target != H_E_N
        || fb_direct_bank.n_target != V1_E_N
        || fb_som_bank.n_target != V1_SOM_N) {
        throw std::runtime_error("frozen Richter deterministic test target size mismatch");
    }
    if (expected_stim_pre_index < 0 || expected_stim_pre_index >= stim_bank.n_pre
        || unexpected_stim_pre_index < 0
        || unexpected_stim_pre_index >= stim_bank.n_pre
        || csr_fanout(stim_bank, expected_stim_pre_index) <= 0
        || csr_fanout(stim_bank, unexpected_stim_pre_index) <= 0) {
        throw std::runtime_error(
            "frozen Richter deterministic test stimulus source is empty/out of range"
        );
    }

    FrozenRichterDeterministicTrialResult result;
    result.n_steps = n_steps;
    result.expected_stim_pre_index = expected_stim_pre_index;
    result.unexpected_stim_pre_index = unexpected_stim_pre_index;
    result.stim_period_steps = stim_period_steps;
    result.phase_steps = {
        {"leader_start_step", leader_start_step},
        {"leader_end_step", leader_end_step},
        {"preprobe_start_step", preprobe_start_step},
        {"preprobe_end_step", preprobe_end_step},
        {"trailer_start_step", trailer_start_step},
        {"trailer_end_step", trailer_end_step},
        {"iti_start_step", iti_start_step},
        {"iti_end_step", iti_end_step},
        {"first_stim_step", first_stim_step},
        {"v1_force_step", v1_force_step},
        {"hctx_force_step", hctx_force_step},
        {"hpred_force_step", hpred_force_step},
    };
    result.edge_counts = {
        {"v1_stim_to_e", static_cast<std::int32_t>(stim_pre.size())},
        {"v1_to_h_ctx", static_cast<std::int32_t>(v1_to_h_pre.size())},
        {"ctx_to_pred", static_cast<std::int32_t>(ctx_to_pred_pre.size())},
        {"fb_pred_to_v1e_apical", static_cast<std::int32_t>(feedback_direct_pre.size())},
        {"fb_pred_to_v1som", static_cast<std::int32_t>(feedback_som_pre.size())},
    };
    result.source_event_counts = {
        {"expected.leader", count_deterministic_events(
            leader_start_step, leader_end_step, stim_offset_steps, stim_period_steps
        )},
        {"expected.preprobe", count_deterministic_events(
            preprobe_start_step, preprobe_end_step, stim_offset_steps,
            stim_period_steps
        )},
        {"expected.trailer", 0},
        {"unexpected.leader", 0},
        {"unexpected.preprobe", 0},
        {"unexpected.trailer", count_deterministic_events(
            trailer_start_step, trailer_end_step, stim_offset_steps, stim_period_steps
        )},
    };
    result.source_event_counts["total"] =
        result.source_event_counts["expected.leader"]
        + result.source_event_counts["expected.preprobe"]
        + result.source_event_counts["unexpected.trailer"];
    result.v1e_index = stim_bank.post[stim_bank.row_ptr[expected_stim_pre_index]];
    const int unexpected_v1e_index =
        stim_bank.post[stim_bank.row_ptr[unexpected_stim_pre_index]];
    if (result.v1e_index < 0 || result.v1e_index >= v1_to_h_bank.n_pre
        || csr_fanout(v1_to_h_bank, result.v1e_index) <= 0) {
        throw std::runtime_error("frozen Richter deterministic test V1 source empty");
    }
    result.hctx_index = v1_to_h_bank.post[v1_to_h_bank.row_ptr[result.v1e_index]];
    if (result.hctx_index < 0 || result.hctx_index >= ctx_to_pred_bank.n_pre
        || csr_fanout(ctx_to_pred_bank, result.hctx_index) <= 0) {
        throw std::runtime_error("frozen Richter deterministic test H_ctx source empty");
    }
    result.hpred_index = ctx_to_pred_bank.post[
        ctx_to_pred_bank.row_ptr[result.hctx_index]
    ];
    if (result.hpred_index < 0 || result.hpred_index >= fb_direct_bank.n_pre
        || result.hpred_index >= fb_som_bank.n_pre
        || csr_fanout(fb_direct_bank, result.hpred_index) <= 0
        || csr_fanout(fb_som_bank, result.hpred_index) <= 0) {
        throw std::runtime_error("frozen Richter deterministic test H_pred source empty");
    }
    result.feedback_v1e_index = fb_direct_bank.post[
        fb_direct_bank.row_ptr[result.hpred_index]
    ];
    result.feedback_som_index = fb_som_bank.post[
        fb_som_bank.row_ptr[result.hpred_index]
    ];
    result.source_fanouts = {
        {"v1_stim_to_e.expected", csr_fanout(stim_bank, expected_stim_pre_index)},
        {"v1_stim_to_e.unexpected", csr_fanout(stim_bank, unexpected_stim_pre_index)},
        {"v1_to_h_ctx", csr_fanout(v1_to_h_bank, result.v1e_index)},
        {"ctx_to_pred", csr_fanout(ctx_to_pred_bank, result.hctx_index)},
        {"fb_pred_to_v1e_apical", csr_fanout(fb_direct_bank, result.hpred_index)},
        {"fb_pred_to_v1som", csr_fanout(fb_som_bank, result.hpred_index)},
    };
    result.drive_amps = {
        {"v1_stim_to_e", stim_drive_amp},
        {"v1_to_h_ctx", v1_to_h_drive_amp},
        {"ctx_to_pred", ctx_to_pred_drive_amp},
        {"fb_pred_to_v1e_apical", feedback_direct_drive_amp},
        {"fb_pred_to_v1som", feedback_som_drive_amp},
    };
    result.event_sums = {
        {"v1_stim_to_e.expected_target", csr_sum_to_target(
            stim_bank, expected_stim_pre_index, result.v1e_index, stim_drive_amp
        )},
        {"v1_stim_to_e.unexpected_first_target", csr_sum_to_target(
            stim_bank, unexpected_stim_pre_index, unexpected_v1e_index,
            stim_drive_amp
        )},
        {"v1_to_h_ctx", csr_sum_to_target(
            v1_to_h_bank, result.v1e_index, result.hctx_index,
            v1_to_h_drive_amp
        )},
        {"ctx_to_pred", csr_sum_to_target(
            ctx_to_pred_bank, result.hctx_index, result.hpred_index,
            ctx_to_pred_drive_amp
        )},
        {"fb_pred_to_v1e_apical", csr_sum_to_target(
            fb_direct_bank, result.hpred_index, result.feedback_v1e_index,
            feedback_direct_drive_amp
        )},
        {"fb_pred_to_v1som", csr_sum_to_target(
            fb_som_bank, result.hpred_index, result.feedback_som_index,
            feedback_som_drive_amp
        )},
    };

    std::map<std::string, std::vector<double>> cpu_v1;
    std::map<std::string, std::vector<double>> cpu_v1_no_stim;
    std::map<std::string, std::vector<double>> cpu_v1_no_fb;
    std::map<std::string, std::vector<double>> cpu_som;
    std::map<std::string, std::vector<double>> cpu_som_no_fb;
    std::map<std::string, std::vector<double>> cpu_hctx;
    std::map<std::string, std::vector<double>> cpu_hctx_no_v1;
    std::map<std::string, std::vector<double>> cpu_hpred;
    std::map<std::string, std::vector<double>> cpu_hpred_no_ctx;
    std::vector<std::int32_t> cpu_dummy_spikes;
    init_v1e_feedback_state(cpu_v1, cpu_dummy_spikes);
    init_v1e_feedback_state(cpu_v1_no_stim, cpu_dummy_spikes);
    init_v1e_feedback_state(cpu_v1_no_fb, cpu_dummy_spikes);
    init_v1som_feedback_state(cpu_som);
    init_v1som_feedback_state(cpu_som_no_fb);
    init_he_quiet_state(cpu_hctx, cpu_dummy_spikes);
    init_he_quiet_state(cpu_hctx_no_v1, cpu_dummy_spikes);
    init_he_quiet_state(cpu_hpred, cpu_dummy_spikes);
    init_he_quiet_state(cpu_hpred_no_ctx, cpu_dummy_spikes);

    std::vector<std::int32_t> cpu_v1_leader(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_preprobe(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_trailer(V1_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_leader(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_trailer(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_leader(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_trailer(H_E_N, 0);
    std::vector<std::int32_t> cpu_v1_discard(V1_E_N, 0);
    std::vector<std::int32_t> cpu_h_discard(H_E_N, 0);

    double cpu_v1_soma_after_stim_step = 0.0;
    double cpu_no_stim_v1_soma_after_stim_step = 0.0;
    double cpu_v1_i_e_after_stim_scatter = 0.0;
    double cpu_v1_soma_after_stim_next_step = 0.0;
    double cpu_no_stim_v1_soma_after_stim_next_step = 0.0;
    double cpu_hctx_v_after_v1_step = 0.0;
    double cpu_no_v1_hctx_v_after_v1_step = 0.0;
    double cpu_hctx_i_e_after_v1_scatter = 0.0;
    double cpu_hctx_v_after_v1_next_step = 0.0;
    double cpu_no_v1_hctx_v_after_v1_next_step = 0.0;
    double cpu_hpred_v_after_hctx_step = 0.0;
    double cpu_no_ctx_hpred_v_after_hctx_step = 0.0;
    double cpu_hpred_i_e_after_hctx_scatter = 0.0;
    double cpu_hpred_v_after_hctx_next_step = 0.0;
    double cpu_no_ctx_hpred_v_after_hctx_next_step = 0.0;
    double cpu_v1_soma_after_hpred_step = 0.0;
    double cpu_no_fb_v1_soma_after_hpred_step = 0.0;
    double cpu_v1_i_ap_after_fb_scatter = 0.0;
    double cpu_som_i_e_after_fb_scatter = 0.0;
    double cpu_v1_ap_after_fb_next_step = 0.0;
    double cpu_no_fb_v1_ap_after_fb_next_step = 0.0;
    double cpu_som_v_after_fb_next_step = 0.0;
    double cpu_no_fb_som_v_after_fb_next_step = 0.0;
    double cpu_v1_soma_after_fb_late_step = 0.0;
    double cpu_no_fb_v1_soma_after_fb_late_step = 0.0;

    for (int step = 0; step < n_steps; ++step) {
        cpu_force_richter_boundary_v1e(
            cpu_v1, step, preprobe_start_step, preprobe_end_step, trailer_end_step
        );
        cpu_force_richter_boundary_v1e(
            cpu_v1_no_stim, step, preprobe_start_step, preprobe_end_step,
            trailer_end_step
        );
        cpu_force_richter_boundary_v1e(
            cpu_v1_no_fb, step, preprobe_start_step, preprobe_end_step,
            trailer_end_step
        );
        cpu_force_richter_boundary_he(
            cpu_hctx, step, preprobe_start_step, preprobe_end_step,
            trailer_end_step
        );
        cpu_force_richter_boundary_he(
            cpu_hctx_no_v1, step, preprobe_start_step, preprobe_end_step,
            trailer_end_step
        );
        cpu_force_richter_boundary_he(
            cpu_hpred, step, preprobe_start_step, preprobe_end_step,
            trailer_end_step
        );
        cpu_force_richter_boundary_he(
            cpu_hpred_no_ctx, step, preprobe_start_step, preprobe_end_step,
            trailer_end_step
        );

        if (step == v1_force_step) {
            cpu_v1["V_soma_mV"][result.v1e_index] = -49.0;
            cpu_v1_no_stim["V_soma_mV"][result.v1e_index] = -49.0;
            cpu_v1_no_fb["V_soma_mV"][result.v1e_index] = -49.0;
        }
        if (step == hctx_force_step) {
            cpu_hctx["V_mV"][result.hctx_index] = -49.0;
            cpu_hctx_no_v1["V_mV"][result.hctx_index] = -49.0;
        }
        if (step == hpred_force_step) {
            cpu_hpred["V_mV"][result.hpred_index] = -49.0;
            cpu_hpred_no_ctx["V_mV"][result.hpred_index] = -49.0;
        }

        cpu_v1e_richter_count_step(
            cpu_v1, cpu_v1_leader, cpu_v1_preprobe, cpu_v1_trailer, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );
        cpu_v1e_richter_count_step(
            cpu_v1_no_stim, cpu_v1_discard, cpu_v1_discard, cpu_v1_discard, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );
        cpu_v1e_richter_count_step(
            cpu_v1_no_fb, cpu_v1_discard, cpu_v1_discard, cpu_v1_discard, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );
        cpu_v1som_step(cpu_som, step);
        cpu_v1som_step(cpu_som_no_fb, step);
        cpu_he_richter_count_step(
            cpu_hctx, cpu_hctx_leader, cpu_hctx_preprobe, cpu_hctx_trailer, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );
        cpu_he_richter_count_step(
            cpu_hctx_no_v1, cpu_h_discard, cpu_h_discard, cpu_h_discard, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );
        cpu_he_richter_count_step(
            cpu_hpred, cpu_hpred_leader, cpu_hpred_preprobe, cpu_hpred_trailer, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );
        cpu_he_richter_count_step(
            cpu_hpred_no_ctx, cpu_h_discard, cpu_h_discard, cpu_h_discard, step,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step
        );

        const bool expected_event =
            deterministic_stim_event(
                step, leader_start_step, leader_end_step, stim_offset_steps,
                stim_period_steps
            )
            || deterministic_stim_event(
                step, preprobe_start_step, preprobe_end_step, stim_offset_steps,
                stim_period_steps
            );
        const bool unexpected_event = deterministic_stim_event(
            step, trailer_start_step, trailer_end_step, stim_offset_steps,
            stim_period_steps
        );
        if (step == first_stim_step) {
            cpu_v1_soma_after_stim_step =
                cpu_v1["V_soma_mV"][result.v1e_index];
            cpu_no_stim_v1_soma_after_stim_step =
                cpu_v1_no_stim["V_soma_mV"][result.v1e_index];
        }
        if (expected_event) {
            cpu_scatter_one_source(
                stim_bank, expected_stim_pre_index, stim_drive_amp,
                cpu_v1["I_e_pA"]
            );
            cpu_scatter_one_source(
                stim_bank, expected_stim_pre_index, stim_drive_amp,
                cpu_v1_no_fb["I_e_pA"]
            );
            if (step == first_stim_step) {
                cpu_v1_i_e_after_stim_scatter =
                    cpu_v1["I_e_pA"][result.v1e_index];
            }
        }
        if (unexpected_event) {
            cpu_scatter_one_source(
                stim_bank, unexpected_stim_pre_index, stim_drive_amp,
                cpu_v1["I_e_pA"]
            );
            cpu_scatter_one_source(
                stim_bank, unexpected_stim_pre_index, stim_drive_amp,
                cpu_v1_no_fb["I_e_pA"]
            );
        }
        if (step == first_stim_step + 1) {
            cpu_v1_soma_after_stim_next_step =
                cpu_v1["V_soma_mV"][result.v1e_index];
            cpu_no_stim_v1_soma_after_stim_next_step =
                cpu_v1_no_stim["V_soma_mV"][result.v1e_index];
        }
        if (step == v1_force_step) {
            cpu_hctx_v_after_v1_step =
                cpu_hctx["V_mV"][result.hctx_index];
            cpu_no_v1_hctx_v_after_v1_step =
                cpu_hctx_no_v1["V_mV"][result.hctx_index];
            cpu_scatter_one_source(
                v1_to_h_bank, result.v1e_index, v1_to_h_drive_amp,
                cpu_hctx["I_e_pA"]
            );
            cpu_hctx_i_e_after_v1_scatter =
                cpu_hctx["I_e_pA"][result.hctx_index];
        }
        if (step == v1_force_step + 1) {
            cpu_hctx_v_after_v1_next_step =
                cpu_hctx["V_mV"][result.hctx_index];
            cpu_no_v1_hctx_v_after_v1_next_step =
                cpu_hctx_no_v1["V_mV"][result.hctx_index];
        }
        if (step == hctx_force_step) {
            cpu_hpred_v_after_hctx_step =
                cpu_hpred["V_mV"][result.hpred_index];
            cpu_no_ctx_hpred_v_after_hctx_step =
                cpu_hpred_no_ctx["V_mV"][result.hpred_index];
            cpu_scatter_one_source(
                ctx_to_pred_bank, result.hctx_index, ctx_to_pred_drive_amp,
                cpu_hpred["I_e_pA"]
            );
            cpu_hpred_i_e_after_hctx_scatter =
                cpu_hpred["I_e_pA"][result.hpred_index];
        }
        if (step == hctx_force_step + 1) {
            cpu_hpred_v_after_hctx_next_step =
                cpu_hpred["V_mV"][result.hpred_index];
            cpu_no_ctx_hpred_v_after_hctx_next_step =
                cpu_hpred_no_ctx["V_mV"][result.hpred_index];
        }
        if (step == hpred_force_step) {
            cpu_v1_soma_after_hpred_step =
                cpu_v1["V_soma_mV"][result.feedback_v1e_index];
            cpu_no_fb_v1_soma_after_hpred_step =
                cpu_v1_no_fb["V_soma_mV"][result.feedback_v1e_index];
            cpu_scatter_one_source(
                fb_direct_bank, result.hpred_index, feedback_direct_drive_amp,
                cpu_v1["I_ap_e_pA"]
            );
            cpu_scatter_one_source(
                fb_som_bank, result.hpred_index, feedback_som_drive_amp,
                cpu_som["I_e_pA"]
            );
            cpu_v1_i_ap_after_fb_scatter =
                cpu_v1["I_ap_e_pA"][result.feedback_v1e_index];
            cpu_som_i_e_after_fb_scatter =
                cpu_som["I_e_pA"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 1) {
            cpu_v1_ap_after_fb_next_step =
                cpu_v1["V_ap_mV"][result.feedback_v1e_index];
            cpu_no_fb_v1_ap_after_fb_next_step =
                cpu_v1_no_fb["V_ap_mV"][result.feedback_v1e_index];
            cpu_som_v_after_fb_next_step =
                cpu_som["V_mV"][result.feedback_som_index];
            cpu_no_fb_som_v_after_fb_next_step =
                cpu_som_no_fb["V_mV"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 2) {
            cpu_v1_soma_after_fb_late_step =
                cpu_v1["V_soma_mV"][result.feedback_v1e_index];
            cpu_no_fb_v1_soma_after_fb_late_step =
                cpu_v1_no_fb["V_soma_mV"][result.feedback_v1e_index];
        }
    }

    result.cpu_raw_counts = {
        {"v1_e.leader", cpu_v1_leader},
        {"v1_e.preprobe", cpu_v1_preprobe},
        {"v1_e.trailer", cpu_v1_trailer},
        {"hctx_e.leader", cpu_hctx_leader},
        {"hctx_e.preprobe", cpu_hctx_preprobe},
        {"hctx_e.trailer", cpu_hctx_trailer},
        {"hpred_e.leader", cpu_hpred_leader},
        {"hpred_e.preprobe", cpu_hpred_preprobe},
        {"hpred_e.trailer", cpu_hpred_trailer},
    };
    append_prefixed_state(result.cpu_final_state, "v1e", cpu_v1);
    append_prefixed_state(result.cpu_final_state, "v1som", cpu_som);
    append_prefixed_state(result.cpu_final_state, "hctx", cpu_hctx);
    append_prefixed_state(result.cpu_final_state, "hpred", cpu_hpred);

    std::map<std::string, std::vector<double>> cuda_v1;
    std::map<std::string, std::vector<double>> cuda_som;
    std::map<std::string, std::vector<double>> cuda_hctx;
    std::map<std::string, std::vector<double>> cuda_hpred;
    std::vector<std::int32_t> cuda_dummy_spikes;
    init_v1e_feedback_state(cuda_v1, cuda_dummy_spikes);
    init_v1som_feedback_state(cuda_som);
    init_he_quiet_state(cuda_hctx, cuda_dummy_spikes);
    init_he_quiet_state(cuda_hpred, cuda_dummy_spikes);
    std::vector<std::int32_t> cuda_v1_leader(V1_E_N, 0);
    std::vector<std::int32_t> cuda_v1_preprobe(V1_E_N, 0);
    std::vector<std::int32_t> cuda_v1_trailer(V1_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_leader(H_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_trailer(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_leader(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_trailer(H_E_N, 0);

    double cuda_v1_soma_after_stim_step = 0.0;
    double cuda_v1_i_e_after_stim_scatter = 0.0;
    double cuda_v1_soma_after_stim_next_step = 0.0;
    double cuda_hctx_v_after_v1_step = 0.0;
    double cuda_hctx_i_e_after_v1_scatter = 0.0;
    double cuda_hctx_v_after_v1_next_step = 0.0;
    double cuda_hpred_v_after_hctx_step = 0.0;
    double cuda_hpred_i_e_after_hctx_scatter = 0.0;
    double cuda_hpred_v_after_hctx_next_step = 0.0;
    double cuda_v1_soma_after_hpred_step = 0.0;
    double cuda_v1_i_ap_after_fb_scatter = 0.0;
    double cuda_som_i_e_after_fb_scatter = 0.0;
    double cuda_v1_ap_after_fb_next_step = 0.0;
    double cuda_som_v_after_fb_next_step = 0.0;
    double cuda_v1_soma_after_fb_late_step = 0.0;

    auto* d_v1_v = device_alloc_copy(cuda_v1["V_soma_mV"]);
    auto* d_v1_vap = device_alloc_copy(cuda_v1["V_ap_mV"]);
    auto* d_v1_ie = device_alloc_copy(cuda_v1["I_e_pA"]);
    auto* d_v1_ii = device_alloc_copy(cuda_v1["I_i_pA"]);
    auto* d_v1_iape = device_alloc_copy(cuda_v1["I_ap_e_pA"]);
    auto* d_v1_w = device_alloc_copy(cuda_v1["w_adapt_pA"]);
    auto* d_v1_ibias = device_alloc_copy(cuda_v1["I_bias_pA"]);
    auto* d_v1_refrac = device_alloc_copy(cuda_v1["refrac_until_ms"]);
    auto* d_v1_leader = device_alloc_copy(cuda_v1_leader);
    auto* d_v1_preprobe = device_alloc_copy(cuda_v1_preprobe);
    auto* d_v1_trailer = device_alloc_copy(cuda_v1_trailer);
    auto* d_som_v = device_alloc_copy(cuda_som["V_mV"]);
    auto* d_som_ie = device_alloc_copy(cuda_som["I_e_pA"]);
    auto* d_som_ii = device_alloc_copy(cuda_som["I_i_pA"]);
    auto* d_som_ibias = device_alloc_copy(cuda_som["I_bias_pA"]);
    auto* d_som_refrac = device_alloc_copy(cuda_som["refrac_until_ms"]);
    auto* d_hctx_v = device_alloc_copy(cuda_hctx["V_mV"]);
    auto* d_hctx_ie = device_alloc_copy(cuda_hctx["I_e_pA"]);
    auto* d_hctx_ii = device_alloc_copy(cuda_hctx["I_i_pA"]);
    auto* d_hctx_g = device_alloc_copy(cuda_hctx["g_nmda_h_nS"]);
    auto* d_hctx_ibias = device_alloc_copy(cuda_hctx["I_bias_pA"]);
    auto* d_hctx_refrac = device_alloc_copy(cuda_hctx["refrac_until_ms"]);
    auto* d_hctx_leader = device_alloc_copy(cuda_hctx_leader);
    auto* d_hctx_preprobe = device_alloc_copy(cuda_hctx_preprobe);
    auto* d_hctx_trailer = device_alloc_copy(cuda_hctx_trailer);
    auto* d_hpred_v = device_alloc_copy(cuda_hpred["V_mV"]);
    auto* d_hpred_ie = device_alloc_copy(cuda_hpred["I_e_pA"]);
    auto* d_hpred_ii = device_alloc_copy(cuda_hpred["I_i_pA"]);
    auto* d_hpred_g = device_alloc_copy(cuda_hpred["g_nmda_h_nS"]);
    auto* d_hpred_ibias = device_alloc_copy(cuda_hpred["I_bias_pA"]);
    auto* d_hpred_refrac = device_alloc_copy(cuda_hpred["refrac_until_ms"]);
    auto* d_hpred_leader = device_alloc_copy(cuda_hpred_leader);
    auto* d_hpred_preprobe = device_alloc_copy(cuda_hpred_preprobe);
    auto* d_hpred_trailer = device_alloc_copy(cuda_hpred_trailer);
    auto* d_stim_row_ptr = device_alloc_copy(stim_bank.row_ptr);
    auto* d_stim_post = device_alloc_copy(stim_bank.post);
    auto* d_stim_weight = device_alloc_copy(stim_bank.weight);
    auto* d_v1_to_h_row_ptr = device_alloc_copy(v1_to_h_bank.row_ptr);
    auto* d_v1_to_h_post = device_alloc_copy(v1_to_h_bank.post);
    auto* d_v1_to_h_weight = device_alloc_copy(v1_to_h_bank.weight);
    auto* d_ctx_to_pred_row_ptr = device_alloc_copy(ctx_to_pred_bank.row_ptr);
    auto* d_ctx_to_pred_post = device_alloc_copy(ctx_to_pred_bank.post);
    auto* d_ctx_to_pred_weight = device_alloc_copy(ctx_to_pred_bank.weight);
    auto* d_fb_direct_row_ptr = device_alloc_copy(fb_direct_bank.row_ptr);
    auto* d_fb_direct_post = device_alloc_copy(fb_direct_bank.post);
    auto* d_fb_direct_weight = device_alloc_copy(fb_direct_bank.weight);
    auto* d_fb_som_row_ptr = device_alloc_copy(fb_som_bank.row_ptr);
    auto* d_fb_som_post = device_alloc_copy(fb_som_bank.post);
    auto* d_fb_som_weight = device_alloc_copy(fb_som_bank.weight);

    for (int step = 0; step < n_steps; ++step) {
        if (step == preprobe_start_step - 1) {
            force_single_voltage_kernel<<<1, 1>>>(COUNT_BEFORE_WINDOW_CELL, -49.0, d_v1_v);
            check_cuda(cudaGetLastError(), "frozen Richter force V1 before launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_BEFORE_WINDOW_CELL, -49.0, d_hctx_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_ctx before launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_BEFORE_WINDOW_CELL, -49.0, d_hpred_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_pred before launch");
        }
        if (step == preprobe_start_step) {
            force_single_voltage_kernel<<<1, 1>>>(COUNT_START_WINDOW_CELL, -49.0, d_v1_v);
            check_cuda(cudaGetLastError(), "frozen Richter force V1 start launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_START_WINDOW_CELL, -49.0, d_hctx_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_ctx start launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_START_WINDOW_CELL, -49.0, d_hpred_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_pred start launch");
        }
        if (step == preprobe_end_step) {
            force_single_voltage_kernel<<<1, 1>>>(COUNT_END_WINDOW_CELL, -49.0, d_v1_v);
            check_cuda(cudaGetLastError(), "frozen Richter force V1 end launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_END_WINDOW_CELL, -49.0, d_hctx_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_ctx end launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_END_WINDOW_CELL, -49.0, d_hpred_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_pred end launch");
        }
        if (step == trailer_end_step) {
            force_single_voltage_kernel<<<1, 1>>>(COUNT_AFTER_WINDOW_CELL, -49.0, d_v1_v);
            check_cuda(cudaGetLastError(), "frozen Richter force V1 after launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_AFTER_WINDOW_CELL, -49.0, d_hctx_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_ctx after launch");
            force_single_voltage_kernel<<<1, 1>>>(COUNT_AFTER_WINDOW_CELL, -49.0, d_hpred_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_pred after launch");
        }
        if (step == v1_force_step) {
            force_single_voltage_kernel<<<1, 1>>>(result.v1e_index, -49.0, d_v1_v);
            check_cuda(cudaGetLastError(), "frozen Richter force V1 source launch");
        }
        if (step == hctx_force_step) {
            force_single_voltage_kernel<<<1, 1>>>(result.hctx_index, -49.0, d_hctx_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_ctx source launch");
        }
        if (step == hpred_force_step) {
            force_single_voltage_kernel<<<1, 1>>>(result.hpred_index, -49.0, d_hpred_v);
            check_cuda(cudaGetLastError(), "frozen Richter force H_pred source launch");
        }
        v1e_richter_count_step_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_v1_v, d_v1_vap, d_v1_ie, d_v1_ii, d_v1_iape, d_v1_w,
            d_v1_ibias, d_v1_refrac, d_v1_leader, d_v1_preprobe,
            d_v1_trailer
        );
        check_cuda(cudaGetLastError(), "frozen Richter V1 step launch");
        v1som_step_kernel<<<1, 64>>>(
            step, d_som_v, d_som_ie, d_som_ii, d_som_ibias, d_som_refrac
        );
        check_cuda(cudaGetLastError(), "frozen Richter SOM step launch");
        he_richter_count_step_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_hctx_v, d_hctx_ie, d_hctx_ii, d_hctx_g, d_hctx_ibias,
            d_hctx_refrac, d_hctx_leader, d_hctx_preprobe, d_hctx_trailer
        );
        check_cuda(cudaGetLastError(), "frozen Richter H_ctx step launch");
        he_richter_count_step_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_hpred_v, d_hpred_ie, d_hpred_ii, d_hpred_g, d_hpred_ibias,
            d_hpred_refrac, d_hpred_leader, d_hpred_preprobe, d_hpred_trailer
        );
        check_cuda(cudaGetLastError(), "frozen Richter H_pred step launch");
        check_cuda(cudaDeviceSynchronize(), "frozen Richter step sync");

        const bool expected_event =
            deterministic_stim_event(
                step, leader_start_step, leader_end_step, stim_offset_steps,
                stim_period_steps
            )
            || deterministic_stim_event(
                step, preprobe_start_step, preprobe_end_step, stim_offset_steps,
                stim_period_steps
            );
        const bool unexpected_event = deterministic_stim_event(
            step, trailer_start_step, trailer_end_step, stim_offset_steps,
            stim_period_steps
        );
        if (step == first_stim_step) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            cuda_v1_soma_after_stim_step =
                cuda_v1["V_soma_mV"][result.v1e_index];
        }
        if (expected_event) {
            csr_scatter_one_source_kernel<<<1, 256>>>(
                expected_stim_pre_index, d_stim_row_ptr, d_stim_post,
                d_stim_weight, stim_drive_amp, d_v1_ie
            );
            check_cuda(cudaGetLastError(), "frozen Richter expected stim scatter launch");
            check_cuda(cudaDeviceSynchronize(), "frozen Richter expected stim sync");
            if (step == first_stim_step) {
                copy_device_to_host(d_v1_ie, cuda_v1["I_e_pA"]);
                cuda_v1_i_e_after_stim_scatter =
                    cuda_v1["I_e_pA"][result.v1e_index];
            }
        }
        if (unexpected_event) {
            csr_scatter_one_source_kernel<<<1, 256>>>(
                unexpected_stim_pre_index, d_stim_row_ptr, d_stim_post,
                d_stim_weight, stim_drive_amp, d_v1_ie
            );
            check_cuda(
                cudaGetLastError(), "frozen Richter unexpected stim scatter launch"
            );
            check_cuda(cudaDeviceSynchronize(), "frozen Richter unexpected stim sync");
        }
        if (step == first_stim_step + 1) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            cuda_v1_soma_after_stim_next_step =
                cuda_v1["V_soma_mV"][result.v1e_index];
        }
        if (step == v1_force_step) {
            copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
            cuda_hctx_v_after_v1_step =
                cuda_hctx["V_mV"][result.hctx_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.v1e_index, d_v1_to_h_row_ptr, d_v1_to_h_post,
                d_v1_to_h_weight, v1_to_h_drive_amp, d_hctx_ie
            );
            check_cuda(cudaGetLastError(), "frozen Richter V1->H scatter launch");
            check_cuda(cudaDeviceSynchronize(), "frozen Richter V1->H sync");
            copy_device_to_host(d_hctx_ie, cuda_hctx["I_e_pA"]);
            cuda_hctx_i_e_after_v1_scatter =
                cuda_hctx["I_e_pA"][result.hctx_index];
        }
        if (step == v1_force_step + 1) {
            copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
            cuda_hctx_v_after_v1_next_step =
                cuda_hctx["V_mV"][result.hctx_index];
        }
        if (step == hctx_force_step) {
            copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
            cuda_hpred_v_after_hctx_step =
                cuda_hpred["V_mV"][result.hpred_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.hctx_index, d_ctx_to_pred_row_ptr, d_ctx_to_pred_post,
                d_ctx_to_pred_weight, ctx_to_pred_drive_amp, d_hpred_ie
            );
            check_cuda(cudaGetLastError(), "frozen Richter ctx->pred scatter launch");
            check_cuda(cudaDeviceSynchronize(), "frozen Richter ctx->pred sync");
            copy_device_to_host(d_hpred_ie, cuda_hpred["I_e_pA"]);
            cuda_hpred_i_e_after_hctx_scatter =
                cuda_hpred["I_e_pA"][result.hpred_index];
        }
        if (step == hctx_force_step + 1) {
            copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
            cuda_hpred_v_after_hctx_next_step =
                cuda_hpred["V_mV"][result.hpred_index];
        }
        if (step == hpred_force_step) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            cuda_v1_soma_after_hpred_step =
                cuda_v1["V_soma_mV"][result.feedback_v1e_index];
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.hpred_index, d_fb_direct_row_ptr, d_fb_direct_post,
                d_fb_direct_weight, feedback_direct_drive_amp, d_v1_iape
            );
            check_cuda(
                cudaGetLastError(), "frozen Richter feedback direct scatter launch"
            );
            csr_scatter_one_source_kernel<<<1, 256>>>(
                result.hpred_index, d_fb_som_row_ptr, d_fb_som_post,
                d_fb_som_weight, feedback_som_drive_amp, d_som_ie
            );
            check_cuda(cudaGetLastError(), "frozen Richter feedback SOM scatter launch");
            check_cuda(cudaDeviceSynchronize(), "frozen Richter feedback sync");
            copy_device_to_host(d_v1_iape, cuda_v1["I_ap_e_pA"]);
            copy_device_to_host(d_som_ie, cuda_som["I_e_pA"]);
            cuda_v1_i_ap_after_fb_scatter =
                cuda_v1["I_ap_e_pA"][result.feedback_v1e_index];
            cuda_som_i_e_after_fb_scatter =
                cuda_som["I_e_pA"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 1) {
            copy_device_to_host(d_v1_vap, cuda_v1["V_ap_mV"]);
            copy_device_to_host(d_som_v, cuda_som["V_mV"]);
            cuda_v1_ap_after_fb_next_step =
                cuda_v1["V_ap_mV"][result.feedback_v1e_index];
            cuda_som_v_after_fb_next_step =
                cuda_som["V_mV"][result.feedback_som_index];
        }
        if (step == hpred_force_step + 2) {
            copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
            cuda_v1_soma_after_fb_late_step =
                cuda_v1["V_soma_mV"][result.feedback_v1e_index];
        }
    }

    copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
    copy_device_to_host(d_v1_vap, cuda_v1["V_ap_mV"]);
    copy_device_to_host(d_v1_ie, cuda_v1["I_e_pA"]);
    copy_device_to_host(d_v1_ii, cuda_v1["I_i_pA"]);
    copy_device_to_host(d_v1_iape, cuda_v1["I_ap_e_pA"]);
    copy_device_to_host(d_v1_w, cuda_v1["w_adapt_pA"]);
    copy_device_to_host(d_v1_ibias, cuda_v1["I_bias_pA"]);
    copy_device_to_host(d_v1_refrac, cuda_v1["refrac_until_ms"]);
    copy_device_to_host(d_v1_leader, cuda_v1_leader);
    copy_device_to_host(d_v1_preprobe, cuda_v1_preprobe);
    copy_device_to_host(d_v1_trailer, cuda_v1_trailer);
    copy_device_to_host(d_som_v, cuda_som["V_mV"]);
    copy_device_to_host(d_som_ie, cuda_som["I_e_pA"]);
    copy_device_to_host(d_som_ii, cuda_som["I_i_pA"]);
    copy_device_to_host(d_som_ibias, cuda_som["I_bias_pA"]);
    copy_device_to_host(d_som_refrac, cuda_som["refrac_until_ms"]);
    copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
    copy_device_to_host(d_hctx_ie, cuda_hctx["I_e_pA"]);
    copy_device_to_host(d_hctx_ii, cuda_hctx["I_i_pA"]);
    copy_device_to_host(d_hctx_g, cuda_hctx["g_nmda_h_nS"]);
    copy_device_to_host(d_hctx_ibias, cuda_hctx["I_bias_pA"]);
    copy_device_to_host(d_hctx_refrac, cuda_hctx["refrac_until_ms"]);
    copy_device_to_host(d_hctx_leader, cuda_hctx_leader);
    copy_device_to_host(d_hctx_preprobe, cuda_hctx_preprobe);
    copy_device_to_host(d_hctx_trailer, cuda_hctx_trailer);
    copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
    copy_device_to_host(d_hpred_ie, cuda_hpred["I_e_pA"]);
    copy_device_to_host(d_hpred_ii, cuda_hpred["I_i_pA"]);
    copy_device_to_host(d_hpred_g, cuda_hpred["g_nmda_h_nS"]);
    copy_device_to_host(d_hpred_ibias, cuda_hpred["I_bias_pA"]);
    copy_device_to_host(d_hpred_refrac, cuda_hpred["refrac_until_ms"]);
    copy_device_to_host(d_hpred_leader, cuda_hpred_leader);
    copy_device_to_host(d_hpred_preprobe, cuda_hpred_preprobe);
    copy_device_to_host(d_hpred_trailer, cuda_hpred_trailer);

    result.cuda_raw_counts = {
        {"v1_e.leader", cuda_v1_leader},
        {"v1_e.preprobe", cuda_v1_preprobe},
        {"v1_e.trailer", cuda_v1_trailer},
        {"hctx_e.leader", cuda_hctx_leader},
        {"hctx_e.preprobe", cuda_hctx_preprobe},
        {"hctx_e.trailer", cuda_hctx_trailer},
        {"hpred_e.leader", cuda_hpred_leader},
        {"hpred_e.preprobe", cuda_hpred_preprobe},
        {"hpred_e.trailer", cuda_hpred_trailer},
    };
    append_prefixed_state(result.cuda_final_state, "v1e", cuda_v1);
    append_prefixed_state(result.cuda_final_state, "v1som", cuda_som);
    append_prefixed_state(result.cuda_final_state, "hctx", cuda_hctx);
    append_prefixed_state(result.cuda_final_state, "hpred", cuda_hpred);

    device_free(d_v1_v); device_free(d_v1_vap); device_free(d_v1_ie);
    device_free(d_v1_ii); device_free(d_v1_iape); device_free(d_v1_w);
    device_free(d_v1_ibias); device_free(d_v1_refrac);
    device_free(d_v1_leader); device_free(d_v1_preprobe);
    device_free(d_v1_trailer);
    device_free(d_som_v); device_free(d_som_ie); device_free(d_som_ii);
    device_free(d_som_ibias); device_free(d_som_refrac);
    device_free(d_hctx_v); device_free(d_hctx_ie); device_free(d_hctx_ii);
    device_free(d_hctx_g); device_free(d_hctx_ibias);
    device_free(d_hctx_refrac); device_free(d_hctx_leader);
    device_free(d_hctx_preprobe); device_free(d_hctx_trailer);
    device_free(d_hpred_v); device_free(d_hpred_ie); device_free(d_hpred_ii);
    device_free(d_hpred_g); device_free(d_hpred_ibias);
    device_free(d_hpred_refrac); device_free(d_hpred_leader);
    device_free(d_hpred_preprobe); device_free(d_hpred_trailer);
    device_free(d_stim_row_ptr); device_free(d_stim_post); device_free(d_stim_weight);
    device_free(d_v1_to_h_row_ptr); device_free(d_v1_to_h_post);
    device_free(d_v1_to_h_weight);
    device_free(d_ctx_to_pred_row_ptr); device_free(d_ctx_to_pred_post);
    device_free(d_ctx_to_pred_weight);
    device_free(d_fb_direct_row_ptr); device_free(d_fb_direct_post);
    device_free(d_fb_direct_weight);
    device_free(d_fb_som_row_ptr); device_free(d_fb_som_post);
    device_free(d_fb_som_weight);

    result.ordering_deltas = {
        {"stim_same_step_abs", std::abs(
            cpu_v1_soma_after_stim_step - cpu_no_stim_v1_soma_after_stim_step
        )},
        {"stim_next_delta",
            cpu_v1_soma_after_stim_next_step
            - cpu_no_stim_v1_soma_after_stim_next_step},
        {"v1_h_same_step_abs", std::abs(
            cpu_hctx_v_after_v1_step - cpu_no_v1_hctx_v_after_v1_step
        )},
        {"v1_h_next_delta",
            cpu_hctx_v_after_v1_next_step
            - cpu_no_v1_hctx_v_after_v1_next_step},
        {"ctx_pred_same_step_abs", std::abs(
            cpu_hpred_v_after_hctx_step - cpu_no_ctx_hpred_v_after_hctx_step
        )},
        {"ctx_pred_next_delta",
            cpu_hpred_v_after_hctx_next_step
            - cpu_no_ctx_hpred_v_after_hctx_next_step},
        {"feedback_same_step_abs", std::abs(
            cpu_v1_soma_after_hpred_step - cpu_no_fb_v1_soma_after_hpred_step
        )},
        {"feedback_apical_next_delta",
            cpu_v1_ap_after_fb_next_step - cpu_no_fb_v1_ap_after_fb_next_step},
        {"feedback_som_next_delta",
            cpu_som_v_after_fb_next_step - cpu_no_fb_som_v_after_fb_next_step},
        {"feedback_soma_late_delta",
            cpu_v1_soma_after_fb_late_step
            - cpu_no_fb_v1_soma_after_fb_late_step},
    };
    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    for (const auto& [key, cpu_counts] : result.cpu_raw_counts) {
        result.max_abs_error["counts." + key] =
            max_count_abs_diff(cpu_counts, result.cuda_raw_counts.at(key));
    }
    result.max_abs_error["v1_soma_after_stim_step"] = std::abs(
        cpu_v1_soma_after_stim_step - cuda_v1_soma_after_stim_step
    );
    result.max_abs_error["v1_i_e_after_stim_scatter"] = std::abs(
        cpu_v1_i_e_after_stim_scatter - cuda_v1_i_e_after_stim_scatter
    );
    result.max_abs_error["v1_soma_after_stim_next_step"] = std::abs(
        cpu_v1_soma_after_stim_next_step - cuda_v1_soma_after_stim_next_step
    );
    result.max_abs_error["hctx_v_after_v1_step"] = std::abs(
        cpu_hctx_v_after_v1_step - cuda_hctx_v_after_v1_step
    );
    result.max_abs_error["hctx_i_e_after_v1_scatter"] = std::abs(
        cpu_hctx_i_e_after_v1_scatter - cuda_hctx_i_e_after_v1_scatter
    );
    result.max_abs_error["hctx_v_after_v1_next_step"] = std::abs(
        cpu_hctx_v_after_v1_next_step - cuda_hctx_v_after_v1_next_step
    );
    result.max_abs_error["hpred_v_after_hctx_step"] = std::abs(
        cpu_hpred_v_after_hctx_step - cuda_hpred_v_after_hctx_step
    );
    result.max_abs_error["hpred_i_e_after_hctx_scatter"] = std::abs(
        cpu_hpred_i_e_after_hctx_scatter - cuda_hpred_i_e_after_hctx_scatter
    );
    result.max_abs_error["hpred_v_after_hctx_next_step"] = std::abs(
        cpu_hpred_v_after_hctx_next_step - cuda_hpred_v_after_hctx_next_step
    );
    result.max_abs_error["v1_soma_after_hpred_step"] = std::abs(
        cpu_v1_soma_after_hpred_step - cuda_v1_soma_after_hpred_step
    );
    result.max_abs_error["v1_i_ap_after_fb_scatter"] = std::abs(
        cpu_v1_i_ap_after_fb_scatter - cuda_v1_i_ap_after_fb_scatter
    );
    result.max_abs_error["som_i_e_after_fb_scatter"] = std::abs(
        cpu_som_i_e_after_fb_scatter - cuda_som_i_e_after_fb_scatter
    );
    result.max_abs_error["v1_ap_after_fb_next_step"] = std::abs(
        cpu_v1_ap_after_fb_next_step - cuda_v1_ap_after_fb_next_step
    );
    result.max_abs_error["som_v_after_fb_next_step"] = std::abs(
        cpu_som_v_after_fb_next_step - cuda_som_v_after_fb_next_step
    );
    result.max_abs_error["v1_soma_after_fb_late_step"] = std::abs(
        cpu_v1_soma_after_fb_late_step - cuda_v1_soma_after_fb_late_step
    );
    result.max_abs_error["count_sum_v1"] = std::abs(
        static_cast<double>(
            sum_counts(result.cpu_raw_counts["v1_e.leader"])
            + sum_counts(result.cpu_raw_counts["v1_e.preprobe"])
            + sum_counts(result.cpu_raw_counts["v1_e.trailer"])
            - sum_counts(result.cuda_raw_counts["v1_e.leader"])
            - sum_counts(result.cuda_raw_counts["v1_e.preprobe"])
            - sum_counts(result.cuda_raw_counts["v1_e.trailer"])
        )
    );
    return result;
}

FrozenRichterSeededSourceResult run_frozen_richter_seeded_source_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    std::int64_t seed,
    std::int32_t expected_channel,
    std::int32_t unexpected_channel,
    double grating_rate_hz,
    double baseline_rate_hz,
    std::int32_t n_steps,
    std::int32_t leader_start_step,
    std::int32_t leader_end_step,
    std::int32_t preprobe_start_step,
    std::int32_t preprobe_end_step,
    std::int32_t trailer_start_step,
    std::int32_t trailer_end_step,
    std::int32_t iti_start_step,
    std::int32_t iti_end_step
) {
    if (n_steps <= 0 || iti_end_step != n_steps
        || leader_start_step < 0
        || leader_start_step >= leader_end_step
        || leader_end_step > preprobe_start_step
        || preprobe_start_step >= preprobe_end_step
        || preprobe_end_step > trailer_start_step
        || trailer_start_step >= trailer_end_step
        || trailer_end_step > iti_start_step
        || iti_start_step >= iti_end_step) {
        throw std::runtime_error(
            "seeded source test requires ordered [start,end) Richter phases"
        );
    }
    if (grating_rate_hz < 0.0 || baseline_rate_hz < 0.0) {
        throw std::runtime_error("seeded source rates must be nonnegative");
    }

    const CsrBank stim_bank = build_csr_bank(
        stim_bank_name, stim_pre, stim_post, stim_weight
    );
    const CsrBank v1_to_h_bank = build_csr_bank(
        v1_to_h_bank_name, v1_to_h_pre, v1_to_h_post, v1_to_h_weight
    );
    const CsrBank ctx_to_pred_bank = build_csr_bank(
        ctx_to_pred_bank_name, ctx_to_pred_pre, ctx_to_pred_post,
        ctx_to_pred_weight
    );
    const CsrBank fb_direct_bank = build_csr_bank(
        feedback_direct_bank_name, feedback_direct_pre, feedback_direct_post,
        feedback_direct_weight
    );
    const CsrBank fb_som_bank = build_csr_bank(
        feedback_som_bank_name, feedback_som_pre, feedback_som_post,
        feedback_som_weight
    );
    if (stim_bank.n_target != V1_E_N
        || v1_to_h_bank.n_target != H_E_N
        || ctx_to_pred_bank.n_target != H_E_N
        || fb_direct_bank.n_target != V1_E_N
        || fb_som_bank.n_target != V1_SOM_N) {
        throw std::runtime_error("seeded source test target size mismatch");
    }
    if (static_cast<int>(stim_channel.size()) != stim_bank.n_pre) {
        throw std::runtime_error("seeded source test stim_channel size mismatch");
    }
    const int n_channels =
        static_cast<int>(*std::max_element(stim_channel.begin(), stim_channel.end())) + 1;
    if (expected_channel < 0 || expected_channel >= n_channels
        || unexpected_channel < 0 || unexpected_channel >= n_channels) {
        throw std::runtime_error("seeded source test channel index out of range");
    }

    FrozenRichterSeededSourceResult result;
    result.seed = seed;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    result.expected_channel = expected_channel;
    result.unexpected_channel = unexpected_channel;
    result.phase_steps = {
        {"leader_start_step", leader_start_step},
        {"leader_end_step", leader_end_step},
        {"preprobe_start_step", preprobe_start_step},
        {"preprobe_end_step", preprobe_end_step},
        {"trailer_start_step", trailer_start_step},
        {"trailer_end_step", trailer_end_step},
        {"iti_start_step", iti_start_step},
        {"iti_end_step", iti_end_step},
    };
    result.edge_counts = {
        {"v1_stim_to_e", static_cast<std::int32_t>(stim_pre.size())},
        {"v1_to_h_ctx", static_cast<std::int32_t>(v1_to_h_pre.size())},
        {"ctx_to_pred", static_cast<std::int32_t>(ctx_to_pred_pre.size())},
        {"fb_pred_to_v1e_apical", static_cast<std::int32_t>(feedback_direct_pre.size())},
        {"fb_pred_to_v1som", static_cast<std::int32_t>(feedback_som_pre.size())},
    };
    result.rates_hz = {
        {"grating", grating_rate_hz},
        {"baseline", baseline_rate_hz},
    };

    std::map<std::string, std::vector<double>> cpu_v1;
    std::map<std::string, std::vector<double>> cpu_som;
    std::map<std::string, std::vector<double>> cpu_hctx;
    std::map<std::string, std::vector<double>> cpu_hpred;
    std::vector<std::int32_t> dummy_spikes;
    init_v1e_feedback_state(cpu_v1, dummy_spikes);
    init_v1som_feedback_state(cpu_som);
    init_he_quiet_state(cpu_hctx, dummy_spikes);
    init_he_quiet_state(cpu_hpred, dummy_spikes);

    std::vector<std::int32_t> cpu_v1_leader(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_preprobe(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_trailer(V1_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_leader(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_trailer(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_leader(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_trailer(H_E_N, 0);
    std::vector<std::int32_t> cpu_v1_flags(V1_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_flags(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_flags(H_E_N, 0);
    std::vector<std::int32_t> cpu_source_by_step(n_steps, 0);
    std::vector<std::int32_t> cpu_source_by_afferent(stim_bank.n_pre, 0);
    std::vector<std::int32_t> cpu_source_by_channel(n_channels, 0);
    std::vector<std::int32_t> cpu_source_by_phase(5, 0);

    for (int step = 0; step < n_steps; ++step) {
        cpu_v1e_richter_count_step_flags(
            cpu_v1, cpu_v1_leader, cpu_v1_preprobe, cpu_v1_trailer,
            cpu_v1_flags, step, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        );
        cpu_v1som_step(cpu_som, step);
        cpu_he_richter_count_step_flags(
            cpu_hctx, cpu_hctx_leader, cpu_hctx_preprobe, cpu_hctx_trailer,
            cpu_hctx_flags, step, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        );
        cpu_he_richter_count_step_flags(
            cpu_hpred, cpu_hpred_leader, cpu_hpred_preprobe, cpu_hpred_trailer,
            cpu_hpred_flags, step, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        );
        for (int src = 0; src < stim_bank.n_pre; ++src) {
            const int channel = stim_channel[static_cast<std::size_t>(src)];
            if (!seeded_source_event(
                    seed, step, src, channel, expected_channel, unexpected_channel,
                    grating_rate_hz, baseline_rate_hz, leader_start_step,
                    leader_end_step, preprobe_start_step, preprobe_end_step,
                    trailer_start_step, trailer_end_step
                )) {
                continue;
            }
            const int phase = richter_phase_index(
                step, leader_start_step, leader_end_step, preprobe_start_step,
                preprobe_end_step, trailer_start_step, trailer_end_step
            );
            const int phase_index =
                phase >= 0
                ? phase
                : ((step >= iti_start_step && step < iti_end_step) ? 3 : 4);
            cpu_source_by_step[static_cast<std::size_t>(step)] += 1;
            cpu_source_by_afferent[static_cast<std::size_t>(src)] += 1;
            cpu_source_by_channel[static_cast<std::size_t>(channel)] += 1;
            cpu_source_by_phase[static_cast<std::size_t>(phase_index)] += 1;
            cpu_source_by_phase[4] += 1;
            cpu_scatter_one_source(
                stim_bank, src, stim_drive_amp, cpu_v1["I_e_pA"]
            );
        }
        cpu_scatter_spike_flags(
            v1_to_h_bank, cpu_v1_flags, v1_to_h_drive_amp, cpu_hctx["I_e_pA"]
        );
        cpu_scatter_spike_flags(
            ctx_to_pred_bank, cpu_hctx_flags, ctx_to_pred_drive_amp,
            cpu_hpred["I_e_pA"]
        );
        cpu_scatter_spike_flags(
            fb_direct_bank, cpu_hpred_flags, feedback_direct_drive_amp,
            cpu_v1["I_ap_e_pA"]
        );
        cpu_scatter_spike_flags(
            fb_som_bank, cpu_hpred_flags, feedback_som_drive_amp,
            cpu_som["I_e_pA"]
        );
    }

    result.cpu_raw_counts = {
        {"v1_e.leader", cpu_v1_leader},
        {"v1_e.preprobe", cpu_v1_preprobe},
        {"v1_e.trailer", cpu_v1_trailer},
        {"hctx_e.leader", cpu_hctx_leader},
        {"hctx_e.preprobe", cpu_hctx_preprobe},
        {"hctx_e.trailer", cpu_hctx_trailer},
        {"hpred_e.leader", cpu_hpred_leader},
        {"hpred_e.preprobe", cpu_hpred_preprobe},
        {"hpred_e.trailer", cpu_hpred_trailer},
    };
    result.cpu_source_counts = {
        {"source.events_by_step", cpu_source_by_step},
        {"source.events_by_afferent", cpu_source_by_afferent},
        {"source.events_by_channel", cpu_source_by_channel},
        {"source.events_by_phase", cpu_source_by_phase},
    };
    result.source_event_counts = {
        {"leader", cpu_source_by_phase[0]},
        {"preprobe", cpu_source_by_phase[1]},
        {"trailer", cpu_source_by_phase[2]},
        {"iti", cpu_source_by_phase[3]},
        {"total", cpu_source_by_phase[4]},
    };
    result.cpu_diagnostic_rates_hz = {
        {"v1_e.leader", counts_to_rate_hz(cpu_v1_leader, leader_start_step, leader_end_step)},
        {"v1_e.preprobe", counts_to_rate_hz(cpu_v1_preprobe, preprobe_start_step, preprobe_end_step)},
        {"v1_e.trailer", counts_to_rate_hz(cpu_v1_trailer, trailer_start_step, trailer_end_step)},
        {"hctx_e.preprobe", counts_to_rate_hz(cpu_hctx_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hctx_e.trailer", counts_to_rate_hz(cpu_hctx_trailer, trailer_start_step, trailer_end_step)},
        {"hpred_e.preprobe", counts_to_rate_hz(cpu_hpred_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hpred_e.trailer", counts_to_rate_hz(cpu_hpred_trailer, trailer_start_step, trailer_end_step)},
    };
    append_prefixed_state(result.cpu_final_state, "v1e", cpu_v1);
    append_prefixed_state(result.cpu_final_state, "v1som", cpu_som);
    append_prefixed_state(result.cpu_final_state, "hctx", cpu_hctx);
    append_prefixed_state(result.cpu_final_state, "hpred", cpu_hpred);

    std::map<std::string, std::vector<double>> cuda_v1;
    std::map<std::string, std::vector<double>> cuda_som;
    std::map<std::string, std::vector<double>> cuda_hctx;
    std::map<std::string, std::vector<double>> cuda_hpred;
    init_v1e_feedback_state(cuda_v1, dummy_spikes);
    init_v1som_feedback_state(cuda_som);
    init_he_quiet_state(cuda_hctx, dummy_spikes);
    init_he_quiet_state(cuda_hpred, dummy_spikes);
    std::vector<std::int32_t> cuda_v1_leader(V1_E_N, 0);
    std::vector<std::int32_t> cuda_v1_preprobe(V1_E_N, 0);
    std::vector<std::int32_t> cuda_v1_trailer(V1_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_leader(H_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_trailer(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_leader(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_trailer(H_E_N, 0);
    std::vector<std::int32_t> cuda_v1_flags(V1_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_flags(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_flags(H_E_N, 0);
    std::vector<std::int32_t> cuda_source_by_step(n_steps, 0);
    std::vector<std::int32_t> cuda_source_by_afferent(stim_bank.n_pre, 0);
    std::vector<std::int32_t> cuda_source_by_channel(n_channels, 0);
    std::vector<std::int32_t> cuda_source_by_phase(5, 0);

    auto* d_v1_v = device_alloc_copy(cuda_v1["V_soma_mV"]);
    auto* d_v1_vap = device_alloc_copy(cuda_v1["V_ap_mV"]);
    auto* d_v1_ie = device_alloc_copy(cuda_v1["I_e_pA"]);
    auto* d_v1_ii = device_alloc_copy(cuda_v1["I_i_pA"]);
    auto* d_v1_iape = device_alloc_copy(cuda_v1["I_ap_e_pA"]);
    auto* d_v1_w = device_alloc_copy(cuda_v1["w_adapt_pA"]);
    auto* d_v1_ibias = device_alloc_copy(cuda_v1["I_bias_pA"]);
    auto* d_v1_refrac = device_alloc_copy(cuda_v1["refrac_until_ms"]);
    auto* d_v1_leader = device_alloc_copy(cuda_v1_leader);
    auto* d_v1_preprobe = device_alloc_copy(cuda_v1_preprobe);
    auto* d_v1_trailer = device_alloc_copy(cuda_v1_trailer);
    auto* d_v1_flags = device_alloc_copy(cuda_v1_flags);
    auto* d_som_v = device_alloc_copy(cuda_som["V_mV"]);
    auto* d_som_ie = device_alloc_copy(cuda_som["I_e_pA"]);
    auto* d_som_ii = device_alloc_copy(cuda_som["I_i_pA"]);
    auto* d_som_ibias = device_alloc_copy(cuda_som["I_bias_pA"]);
    auto* d_som_refrac = device_alloc_copy(cuda_som["refrac_until_ms"]);
    auto* d_hctx_v = device_alloc_copy(cuda_hctx["V_mV"]);
    auto* d_hctx_ie = device_alloc_copy(cuda_hctx["I_e_pA"]);
    auto* d_hctx_ii = device_alloc_copy(cuda_hctx["I_i_pA"]);
    auto* d_hctx_g = device_alloc_copy(cuda_hctx["g_nmda_h_nS"]);
    auto* d_hctx_ibias = device_alloc_copy(cuda_hctx["I_bias_pA"]);
    auto* d_hctx_refrac = device_alloc_copy(cuda_hctx["refrac_until_ms"]);
    auto* d_hctx_leader = device_alloc_copy(cuda_hctx_leader);
    auto* d_hctx_preprobe = device_alloc_copy(cuda_hctx_preprobe);
    auto* d_hctx_trailer = device_alloc_copy(cuda_hctx_trailer);
    auto* d_hctx_flags = device_alloc_copy(cuda_hctx_flags);
    auto* d_hpred_v = device_alloc_copy(cuda_hpred["V_mV"]);
    auto* d_hpred_ie = device_alloc_copy(cuda_hpred["I_e_pA"]);
    auto* d_hpred_ii = device_alloc_copy(cuda_hpred["I_i_pA"]);
    auto* d_hpred_g = device_alloc_copy(cuda_hpred["g_nmda_h_nS"]);
    auto* d_hpred_ibias = device_alloc_copy(cuda_hpred["I_bias_pA"]);
    auto* d_hpred_refrac = device_alloc_copy(cuda_hpred["refrac_until_ms"]);
    auto* d_hpred_leader = device_alloc_copy(cuda_hpred_leader);
    auto* d_hpred_preprobe = device_alloc_copy(cuda_hpred_preprobe);
    auto* d_hpred_trailer = device_alloc_copy(cuda_hpred_trailer);
    auto* d_hpred_flags = device_alloc_copy(cuda_hpred_flags);
    auto* d_stim_channel = device_alloc_copy(stim_channel);
    auto* d_source_by_step = device_alloc_copy(cuda_source_by_step);
    auto* d_source_by_afferent = device_alloc_copy(cuda_source_by_afferent);
    auto* d_source_by_channel = device_alloc_copy(cuda_source_by_channel);
    auto* d_source_by_phase = device_alloc_copy(cuda_source_by_phase);
    auto* d_stim_row_ptr = device_alloc_copy(stim_bank.row_ptr);
    auto* d_stim_post = device_alloc_copy(stim_bank.post);
    auto* d_stim_weight = device_alloc_copy(stim_bank.weight);
    auto* d_v1_to_h_row_ptr = device_alloc_copy(v1_to_h_bank.row_ptr);
    auto* d_v1_to_h_post = device_alloc_copy(v1_to_h_bank.post);
    auto* d_v1_to_h_weight = device_alloc_copy(v1_to_h_bank.weight);
    auto* d_ctx_to_pred_row_ptr = device_alloc_copy(ctx_to_pred_bank.row_ptr);
    auto* d_ctx_to_pred_post = device_alloc_copy(ctx_to_pred_bank.post);
    auto* d_ctx_to_pred_weight = device_alloc_copy(ctx_to_pred_bank.weight);
    auto* d_fb_direct_row_ptr = device_alloc_copy(fb_direct_bank.row_ptr);
    auto* d_fb_direct_post = device_alloc_copy(fb_direct_bank.post);
    auto* d_fb_direct_weight = device_alloc_copy(fb_direct_bank.weight);
    auto* d_fb_som_row_ptr = device_alloc_copy(fb_som_bank.row_ptr);
    auto* d_fb_som_post = device_alloc_copy(fb_som_bank.post);
    auto* d_fb_som_weight = device_alloc_copy(fb_som_bank.weight);

    const int source_blocks = (stim_bank.n_pre + 127) / 128;
    for (int step = 0; step < n_steps; ++step) {
        v1e_richter_count_step_flags_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_v1_v, d_v1_vap, d_v1_ie, d_v1_ii, d_v1_iape, d_v1_w,
            d_v1_ibias, d_v1_refrac, d_v1_leader, d_v1_preprobe,
            d_v1_trailer, d_v1_flags
        );
        check_cuda(cudaGetLastError(), "seeded source V1 step launch");
        v1som_step_kernel<<<1, 64>>>(
            step, d_som_v, d_som_ie, d_som_ii, d_som_ibias, d_som_refrac
        );
        check_cuda(cudaGetLastError(), "seeded source SOM step launch");
        he_richter_count_step_flags_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_hctx_v, d_hctx_ie, d_hctx_ii, d_hctx_g, d_hctx_ibias,
            d_hctx_refrac, d_hctx_leader, d_hctx_preprobe, d_hctx_trailer,
            d_hctx_flags
        );
        check_cuda(cudaGetLastError(), "seeded source H_ctx step launch");
        he_richter_count_step_flags_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_hpred_v, d_hpred_ie, d_hpred_ii, d_hpred_g, d_hpred_ibias,
            d_hpred_refrac, d_hpred_leader, d_hpred_preprobe, d_hpred_trailer,
            d_hpred_flags
        );
        check_cuda(cudaGetLastError(), "seeded source H_pred step launch");
        seeded_stim_source_scatter_kernel<<<source_blocks, 128>>>(
            seed, step, stim_bank.n_pre, d_stim_channel, expected_channel,
            unexpected_channel, grating_rate_hz, baseline_rate_hz,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            iti_start_step, iti_end_step, d_stim_row_ptr, d_stim_post,
            d_stim_weight, stim_drive_amp, d_v1_ie, d_source_by_step,
            d_source_by_afferent, d_source_by_channel, d_source_by_phase
        );
        check_cuda(cudaGetLastError(), "seeded source stimulus scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            v1_to_h_bank.n_pre, d_v1_flags, d_v1_to_h_row_ptr, d_v1_to_h_post,
            d_v1_to_h_weight, v1_to_h_drive_amp, d_hctx_ie
        );
        check_cuda(cudaGetLastError(), "seeded source V1->H scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            ctx_to_pred_bank.n_pre, d_hctx_flags, d_ctx_to_pred_row_ptr,
            d_ctx_to_pred_post, d_ctx_to_pred_weight, ctx_to_pred_drive_amp,
            d_hpred_ie
        );
        check_cuda(cudaGetLastError(), "seeded source ctx->pred scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            fb_direct_bank.n_pre, d_hpred_flags, d_fb_direct_row_ptr,
            d_fb_direct_post, d_fb_direct_weight, feedback_direct_drive_amp,
            d_v1_iape
        );
        check_cuda(cudaGetLastError(), "seeded source feedback direct scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            fb_som_bank.n_pre, d_hpred_flags, d_fb_som_row_ptr, d_fb_som_post,
            d_fb_som_weight, feedback_som_drive_amp, d_som_ie
        );
        check_cuda(cudaGetLastError(), "seeded source feedback SOM scatter launch");
        check_cuda(cudaDeviceSynchronize(), "seeded source step sync");
    }

    copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
    copy_device_to_host(d_v1_vap, cuda_v1["V_ap_mV"]);
    copy_device_to_host(d_v1_ie, cuda_v1["I_e_pA"]);
    copy_device_to_host(d_v1_ii, cuda_v1["I_i_pA"]);
    copy_device_to_host(d_v1_iape, cuda_v1["I_ap_e_pA"]);
    copy_device_to_host(d_v1_w, cuda_v1["w_adapt_pA"]);
    copy_device_to_host(d_v1_ibias, cuda_v1["I_bias_pA"]);
    copy_device_to_host(d_v1_refrac, cuda_v1["refrac_until_ms"]);
    copy_device_to_host(d_v1_leader, cuda_v1_leader);
    copy_device_to_host(d_v1_preprobe, cuda_v1_preprobe);
    copy_device_to_host(d_v1_trailer, cuda_v1_trailer);
    copy_device_to_host(d_som_v, cuda_som["V_mV"]);
    copy_device_to_host(d_som_ie, cuda_som["I_e_pA"]);
    copy_device_to_host(d_som_ii, cuda_som["I_i_pA"]);
    copy_device_to_host(d_som_ibias, cuda_som["I_bias_pA"]);
    copy_device_to_host(d_som_refrac, cuda_som["refrac_until_ms"]);
    copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
    copy_device_to_host(d_hctx_ie, cuda_hctx["I_e_pA"]);
    copy_device_to_host(d_hctx_ii, cuda_hctx["I_i_pA"]);
    copy_device_to_host(d_hctx_g, cuda_hctx["g_nmda_h_nS"]);
    copy_device_to_host(d_hctx_ibias, cuda_hctx["I_bias_pA"]);
    copy_device_to_host(d_hctx_refrac, cuda_hctx["refrac_until_ms"]);
    copy_device_to_host(d_hctx_leader, cuda_hctx_leader);
    copy_device_to_host(d_hctx_preprobe, cuda_hctx_preprobe);
    copy_device_to_host(d_hctx_trailer, cuda_hctx_trailer);
    copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
    copy_device_to_host(d_hpred_ie, cuda_hpred["I_e_pA"]);
    copy_device_to_host(d_hpred_ii, cuda_hpred["I_i_pA"]);
    copy_device_to_host(d_hpred_g, cuda_hpred["g_nmda_h_nS"]);
    copy_device_to_host(d_hpred_ibias, cuda_hpred["I_bias_pA"]);
    copy_device_to_host(d_hpred_refrac, cuda_hpred["refrac_until_ms"]);
    copy_device_to_host(d_hpred_leader, cuda_hpred_leader);
    copy_device_to_host(d_hpred_preprobe, cuda_hpred_preprobe);
    copy_device_to_host(d_hpred_trailer, cuda_hpred_trailer);
    copy_device_to_host(d_source_by_step, cuda_source_by_step);
    copy_device_to_host(d_source_by_afferent, cuda_source_by_afferent);
    copy_device_to_host(d_source_by_channel, cuda_source_by_channel);
    copy_device_to_host(d_source_by_phase, cuda_source_by_phase);

    result.cuda_raw_counts = {
        {"v1_e.leader", cuda_v1_leader},
        {"v1_e.preprobe", cuda_v1_preprobe},
        {"v1_e.trailer", cuda_v1_trailer},
        {"hctx_e.leader", cuda_hctx_leader},
        {"hctx_e.preprobe", cuda_hctx_preprobe},
        {"hctx_e.trailer", cuda_hctx_trailer},
        {"hpred_e.leader", cuda_hpred_leader},
        {"hpred_e.preprobe", cuda_hpred_preprobe},
        {"hpred_e.trailer", cuda_hpred_trailer},
    };
    result.cuda_source_counts = {
        {"source.events_by_step", cuda_source_by_step},
        {"source.events_by_afferent", cuda_source_by_afferent},
        {"source.events_by_channel", cuda_source_by_channel},
        {"source.events_by_phase", cuda_source_by_phase},
    };
    result.cuda_diagnostic_rates_hz = {
        {"v1_e.leader", counts_to_rate_hz(cuda_v1_leader, leader_start_step, leader_end_step)},
        {"v1_e.preprobe", counts_to_rate_hz(cuda_v1_preprobe, preprobe_start_step, preprobe_end_step)},
        {"v1_e.trailer", counts_to_rate_hz(cuda_v1_trailer, trailer_start_step, trailer_end_step)},
        {"hctx_e.preprobe", counts_to_rate_hz(cuda_hctx_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hctx_e.trailer", counts_to_rate_hz(cuda_hctx_trailer, trailer_start_step, trailer_end_step)},
        {"hpred_e.preprobe", counts_to_rate_hz(cuda_hpred_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hpred_e.trailer", counts_to_rate_hz(cuda_hpred_trailer, trailer_start_step, trailer_end_step)},
    };
    append_prefixed_state(result.cuda_final_state, "v1e", cuda_v1);
    append_prefixed_state(result.cuda_final_state, "v1som", cuda_som);
    append_prefixed_state(result.cuda_final_state, "hctx", cuda_hctx);
    append_prefixed_state(result.cuda_final_state, "hpred", cuda_hpred);

    device_free(d_v1_v); device_free(d_v1_vap); device_free(d_v1_ie);
    device_free(d_v1_ii); device_free(d_v1_iape); device_free(d_v1_w);
    device_free(d_v1_ibias); device_free(d_v1_refrac);
    device_free(d_v1_leader); device_free(d_v1_preprobe);
    device_free(d_v1_trailer); device_free(d_v1_flags);
    device_free(d_som_v); device_free(d_som_ie); device_free(d_som_ii);
    device_free(d_som_ibias); device_free(d_som_refrac);
    device_free(d_hctx_v); device_free(d_hctx_ie); device_free(d_hctx_ii);
    device_free(d_hctx_g); device_free(d_hctx_ibias);
    device_free(d_hctx_refrac); device_free(d_hctx_leader);
    device_free(d_hctx_preprobe); device_free(d_hctx_trailer);
    device_free(d_hctx_flags);
    device_free(d_hpred_v); device_free(d_hpred_ie); device_free(d_hpred_ii);
    device_free(d_hpred_g); device_free(d_hpred_ibias);
    device_free(d_hpred_refrac); device_free(d_hpred_leader);
    device_free(d_hpred_preprobe); device_free(d_hpred_trailer);
    device_free(d_hpred_flags);
    device_free(d_stim_channel); device_free(d_source_by_step);
    device_free(d_source_by_afferent); device_free(d_source_by_channel);
    device_free(d_source_by_phase);
    device_free(d_stim_row_ptr); device_free(d_stim_post); device_free(d_stim_weight);
    device_free(d_v1_to_h_row_ptr); device_free(d_v1_to_h_post);
    device_free(d_v1_to_h_weight);
    device_free(d_ctx_to_pred_row_ptr); device_free(d_ctx_to_pred_post);
    device_free(d_ctx_to_pred_weight);
    device_free(d_fb_direct_row_ptr); device_free(d_fb_direct_post);
    device_free(d_fb_direct_weight);
    device_free(d_fb_som_row_ptr); device_free(d_fb_som_post);
    device_free(d_fb_som_weight);

    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    for (const auto& [key, cpu_counts] : result.cpu_raw_counts) {
        result.max_abs_error["raw_counts." + key] =
            max_count_abs_diff(cpu_counts, result.cuda_raw_counts.at(key));
    }
    for (const auto& [key, cpu_counts] : result.cpu_source_counts) {
        result.max_abs_error[key] =
            max_count_abs_diff(cpu_counts, result.cuda_source_counts.at(key));
    }
    for (const auto& [key, cpu_rates] : result.cpu_diagnostic_rates_hz) {
        result.max_abs_error["rates." + key] =
            max_abs_diff(cpu_rates, result.cuda_diagnostic_rates_hz.at(key));
    }
    return result;
}

FrozenRichterSeededSourceResult run_frozen_richter_controlled_source_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    const std::vector<std::int32_t>& event_steps,
    const std::vector<std::int32_t>& event_sources,
    std::int32_t expected_channel,
    std::int32_t unexpected_channel,
    std::int32_t n_steps,
    std::int32_t leader_start_step,
    std::int32_t leader_end_step,
    std::int32_t preprobe_start_step,
    std::int32_t preprobe_end_step,
    std::int32_t trailer_start_step,
    std::int32_t trailer_end_step,
    std::int32_t iti_start_step,
    std::int32_t iti_end_step
) {
    if (n_steps <= 0 || iti_end_step != n_steps
        || leader_start_step < 0
        || leader_start_step >= leader_end_step
        || leader_end_step > preprobe_start_step
        || preprobe_start_step >= preprobe_end_step
        || preprobe_end_step > trailer_start_step
        || trailer_start_step >= trailer_end_step
        || trailer_end_step > iti_start_step
        || iti_start_step >= iti_end_step) {
        throw std::runtime_error(
            "controlled source test requires ordered [start,end) Richter phases"
        );
    }
    if (event_steps.size() != event_sources.size()) {
        throw std::runtime_error("controlled source event arrays must match");
    }
    if (event_steps.empty()) {
        throw std::runtime_error("controlled source test requires at least one event");
    }

    const CsrBank stim_bank = build_csr_bank(
        stim_bank_name, stim_pre, stim_post, stim_weight
    );
    const CsrBank v1_to_h_bank = build_csr_bank(
        v1_to_h_bank_name, v1_to_h_pre, v1_to_h_post, v1_to_h_weight
    );
    const CsrBank ctx_to_pred_bank = build_csr_bank(
        ctx_to_pred_bank_name, ctx_to_pred_pre, ctx_to_pred_post,
        ctx_to_pred_weight
    );
    const CsrBank fb_direct_bank = build_csr_bank(
        feedback_direct_bank_name, feedback_direct_pre, feedback_direct_post,
        feedback_direct_weight
    );
    const CsrBank fb_som_bank = build_csr_bank(
        feedback_som_bank_name, feedback_som_pre, feedback_som_post,
        feedback_som_weight
    );
    if (stim_bank.n_target != V1_E_N
        || v1_to_h_bank.n_target != H_E_N
        || ctx_to_pred_bank.n_target != H_E_N
        || fb_direct_bank.n_target != V1_E_N
        || fb_som_bank.n_target != V1_SOM_N) {
        throw std::runtime_error("controlled source test target size mismatch");
    }
    if (static_cast<int>(stim_channel.size()) != stim_bank.n_pre) {
        throw std::runtime_error("controlled source test stim_channel size mismatch");
    }
    const int n_channels =
        static_cast<int>(*std::max_element(stim_channel.begin(), stim_channel.end())) + 1;
    if (expected_channel < 0 || expected_channel >= n_channels
        || unexpected_channel < 0 || unexpected_channel >= n_channels) {
        throw std::runtime_error("controlled source test channel index out of range");
    }
    std::vector<std::vector<std::int32_t>> events_by_step(
        static_cast<std::size_t>(n_steps)
    );
    for (std::size_t e = 0; e < event_steps.size(); ++e) {
        const int step = event_steps[e];
        const int src = event_sources[e];
        if (step < 0 || step >= n_steps) {
            throw std::runtime_error("controlled source event step out of range");
        }
        if (src < 0 || src >= stim_bank.n_pre) {
            throw std::runtime_error("controlled source event source out of range");
        }
        events_by_step[static_cast<std::size_t>(step)].push_back(src);
    }

    FrozenRichterSeededSourceResult result;
    result.seed = 0;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    result.expected_channel = expected_channel;
    result.unexpected_channel = unexpected_channel;
    result.phase_steps = {
        {"leader_start_step", leader_start_step},
        {"leader_end_step", leader_end_step},
        {"preprobe_start_step", preprobe_start_step},
        {"preprobe_end_step", preprobe_end_step},
        {"trailer_start_step", trailer_start_step},
        {"trailer_end_step", trailer_end_step},
        {"iti_start_step", iti_start_step},
        {"iti_end_step", iti_end_step},
    };
    result.edge_counts = {
        {"v1_stim_to_e", static_cast<std::int32_t>(stim_pre.size())},
        {"v1_to_h_ctx", static_cast<std::int32_t>(v1_to_h_pre.size())},
        {"ctx_to_pred", static_cast<std::int32_t>(ctx_to_pred_pre.size())},
        {"fb_pred_to_v1e_apical", static_cast<std::int32_t>(feedback_direct_pre.size())},
        {"fb_pred_to_v1som", static_cast<std::int32_t>(feedback_som_pre.size())},
    };
    result.rates_hz = {
        {"grating", 0.0},
        {"baseline", 0.0},
        {"controlled_event_count", static_cast<double>(event_steps.size())},
    };

    std::map<std::string, std::vector<double>> cpu_v1;
    std::map<std::string, std::vector<double>> cpu_som;
    std::map<std::string, std::vector<double>> cpu_hctx;
    std::map<std::string, std::vector<double>> cpu_hpred;
    std::vector<std::int32_t> dummy_spikes;
    init_v1e_feedback_state(cpu_v1, dummy_spikes);
    init_v1som_feedback_state(cpu_som);
    init_he_quiet_state(cpu_hctx, dummy_spikes);
    init_he_quiet_state(cpu_hpred, dummy_spikes);

    std::vector<std::int32_t> cpu_v1_leader(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_preprobe(V1_E_N, 0);
    std::vector<std::int32_t> cpu_v1_trailer(V1_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_leader(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_trailer(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_leader(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_trailer(H_E_N, 0);
    std::vector<std::int32_t> cpu_v1_flags(V1_E_N, 0);
    std::vector<std::int32_t> cpu_hctx_flags(H_E_N, 0);
    std::vector<std::int32_t> cpu_hpred_flags(H_E_N, 0);
    std::vector<std::int32_t> cpu_source_by_step(n_steps, 0);
    std::vector<std::int32_t> cpu_source_by_afferent(stim_bank.n_pre, 0);
    std::vector<std::int32_t> cpu_source_by_channel(n_channels, 0);
    std::vector<std::int32_t> cpu_source_by_phase(5, 0);

    for (int step = 0; step < n_steps; ++step) {
        cpu_v1e_richter_count_step_flags(
            cpu_v1, cpu_v1_leader, cpu_v1_preprobe, cpu_v1_trailer,
            cpu_v1_flags, step, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        );
        cpu_v1som_step(cpu_som, step);
        cpu_he_richter_count_step_flags(
            cpu_hctx, cpu_hctx_leader, cpu_hctx_preprobe, cpu_hctx_trailer,
            cpu_hctx_flags, step, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        );
        cpu_he_richter_count_step_flags(
            cpu_hpred, cpu_hpred_leader, cpu_hpred_preprobe, cpu_hpred_trailer,
            cpu_hpred_flags, step, leader_start_step, leader_end_step,
            preprobe_start_step, preprobe_end_step, trailer_start_step,
            trailer_end_step
        );
        for (const int src : events_by_step[static_cast<std::size_t>(step)]) {
            const int channel = stim_channel[static_cast<std::size_t>(src)];
            const int phase = richter_phase_index(
                step, leader_start_step, leader_end_step, preprobe_start_step,
                preprobe_end_step, trailer_start_step, trailer_end_step
            );
            const int phase_index =
                phase >= 0
                ? phase
                : ((step >= iti_start_step && step < iti_end_step) ? 3 : 4);
            cpu_source_by_step[static_cast<std::size_t>(step)] += 1;
            cpu_source_by_afferent[static_cast<std::size_t>(src)] += 1;
            cpu_source_by_channel[static_cast<std::size_t>(channel)] += 1;
            cpu_source_by_phase[static_cast<std::size_t>(phase_index)] += 1;
            cpu_source_by_phase[4] += 1;
            cpu_scatter_one_source(
                stim_bank, src, stim_drive_amp, cpu_v1["I_e_pA"]
            );
        }
        cpu_scatter_spike_flags(
            v1_to_h_bank, cpu_v1_flags, v1_to_h_drive_amp, cpu_hctx["I_e_pA"]
        );
        cpu_scatter_spike_flags(
            ctx_to_pred_bank, cpu_hctx_flags, ctx_to_pred_drive_amp,
            cpu_hpred["I_e_pA"]
        );
        cpu_scatter_spike_flags(
            fb_direct_bank, cpu_hpred_flags, feedback_direct_drive_amp,
            cpu_v1["I_ap_e_pA"]
        );
        cpu_scatter_spike_flags(
            fb_som_bank, cpu_hpred_flags, feedback_som_drive_amp,
            cpu_som["I_e_pA"]
        );
    }

    result.cpu_raw_counts = {
        {"v1_e.leader", cpu_v1_leader},
        {"v1_e.preprobe", cpu_v1_preprobe},
        {"v1_e.trailer", cpu_v1_trailer},
        {"hctx_e.leader", cpu_hctx_leader},
        {"hctx_e.preprobe", cpu_hctx_preprobe},
        {"hctx_e.trailer", cpu_hctx_trailer},
        {"hpred_e.leader", cpu_hpred_leader},
        {"hpred_e.preprobe", cpu_hpred_preprobe},
        {"hpred_e.trailer", cpu_hpred_trailer},
    };
    result.cpu_source_counts = {
        {"source.events_by_step", cpu_source_by_step},
        {"source.events_by_afferent", cpu_source_by_afferent},
        {"source.events_by_channel", cpu_source_by_channel},
        {"source.events_by_phase", cpu_source_by_phase},
    };
    result.source_event_counts = {
        {"leader", cpu_source_by_phase[0]},
        {"preprobe", cpu_source_by_phase[1]},
        {"trailer", cpu_source_by_phase[2]},
        {"iti", cpu_source_by_phase[3]},
        {"total", cpu_source_by_phase[4]},
    };
    result.cpu_diagnostic_rates_hz = {
        {"v1_e.leader", counts_to_rate_hz(cpu_v1_leader, leader_start_step, leader_end_step)},
        {"v1_e.preprobe", counts_to_rate_hz(cpu_v1_preprobe, preprobe_start_step, preprobe_end_step)},
        {"v1_e.trailer", counts_to_rate_hz(cpu_v1_trailer, trailer_start_step, trailer_end_step)},
        {"hctx_e.preprobe", counts_to_rate_hz(cpu_hctx_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hctx_e.trailer", counts_to_rate_hz(cpu_hctx_trailer, trailer_start_step, trailer_end_step)},
        {"hpred_e.preprobe", counts_to_rate_hz(cpu_hpred_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hpred_e.trailer", counts_to_rate_hz(cpu_hpred_trailer, trailer_start_step, trailer_end_step)},
    };
    append_prefixed_state(result.cpu_final_state, "v1e", cpu_v1);
    append_prefixed_state(result.cpu_final_state, "v1som", cpu_som);
    append_prefixed_state(result.cpu_final_state, "hctx", cpu_hctx);
    append_prefixed_state(result.cpu_final_state, "hpred", cpu_hpred);

    std::map<std::string, std::vector<double>> cuda_v1;
    std::map<std::string, std::vector<double>> cuda_som;
    std::map<std::string, std::vector<double>> cuda_hctx;
    std::map<std::string, std::vector<double>> cuda_hpred;
    init_v1e_feedback_state(cuda_v1, dummy_spikes);
    init_v1som_feedback_state(cuda_som);
    init_he_quiet_state(cuda_hctx, dummy_spikes);
    init_he_quiet_state(cuda_hpred, dummy_spikes);
    std::vector<std::int32_t> cuda_v1_leader(V1_E_N, 0);
    std::vector<std::int32_t> cuda_v1_preprobe(V1_E_N, 0);
    std::vector<std::int32_t> cuda_v1_trailer(V1_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_leader(H_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_trailer(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_leader(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_preprobe(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_trailer(H_E_N, 0);
    std::vector<std::int32_t> cuda_v1_flags(V1_E_N, 0);
    std::vector<std::int32_t> cuda_hctx_flags(H_E_N, 0);
    std::vector<std::int32_t> cuda_hpred_flags(H_E_N, 0);
    std::vector<std::int32_t> cuda_source_by_step(n_steps, 0);
    std::vector<std::int32_t> cuda_source_by_afferent(stim_bank.n_pre, 0);
    std::vector<std::int32_t> cuda_source_by_channel(n_channels, 0);
    std::vector<std::int32_t> cuda_source_by_phase(5, 0);

    auto* d_v1_v = device_alloc_copy(cuda_v1["V_soma_mV"]);
    auto* d_v1_vap = device_alloc_copy(cuda_v1["V_ap_mV"]);
    auto* d_v1_ie = device_alloc_copy(cuda_v1["I_e_pA"]);
    auto* d_v1_ii = device_alloc_copy(cuda_v1["I_i_pA"]);
    auto* d_v1_iape = device_alloc_copy(cuda_v1["I_ap_e_pA"]);
    auto* d_v1_w = device_alloc_copy(cuda_v1["w_adapt_pA"]);
    auto* d_v1_ibias = device_alloc_copy(cuda_v1["I_bias_pA"]);
    auto* d_v1_refrac = device_alloc_copy(cuda_v1["refrac_until_ms"]);
    auto* d_v1_leader = device_alloc_copy(cuda_v1_leader);
    auto* d_v1_preprobe = device_alloc_copy(cuda_v1_preprobe);
    auto* d_v1_trailer = device_alloc_copy(cuda_v1_trailer);
    auto* d_v1_flags = device_alloc_copy(cuda_v1_flags);
    auto* d_som_v = device_alloc_copy(cuda_som["V_mV"]);
    auto* d_som_ie = device_alloc_copy(cuda_som["I_e_pA"]);
    auto* d_som_ii = device_alloc_copy(cuda_som["I_i_pA"]);
    auto* d_som_ibias = device_alloc_copy(cuda_som["I_bias_pA"]);
    auto* d_som_refrac = device_alloc_copy(cuda_som["refrac_until_ms"]);
    auto* d_hctx_v = device_alloc_copy(cuda_hctx["V_mV"]);
    auto* d_hctx_ie = device_alloc_copy(cuda_hctx["I_e_pA"]);
    auto* d_hctx_ii = device_alloc_copy(cuda_hctx["I_i_pA"]);
    auto* d_hctx_g = device_alloc_copy(cuda_hctx["g_nmda_h_nS"]);
    auto* d_hctx_ibias = device_alloc_copy(cuda_hctx["I_bias_pA"]);
    auto* d_hctx_refrac = device_alloc_copy(cuda_hctx["refrac_until_ms"]);
    auto* d_hctx_leader = device_alloc_copy(cuda_hctx_leader);
    auto* d_hctx_preprobe = device_alloc_copy(cuda_hctx_preprobe);
    auto* d_hctx_trailer = device_alloc_copy(cuda_hctx_trailer);
    auto* d_hctx_flags = device_alloc_copy(cuda_hctx_flags);
    auto* d_hpred_v = device_alloc_copy(cuda_hpred["V_mV"]);
    auto* d_hpred_ie = device_alloc_copy(cuda_hpred["I_e_pA"]);
    auto* d_hpred_ii = device_alloc_copy(cuda_hpred["I_i_pA"]);
    auto* d_hpred_g = device_alloc_copy(cuda_hpred["g_nmda_h_nS"]);
    auto* d_hpred_ibias = device_alloc_copy(cuda_hpred["I_bias_pA"]);
    auto* d_hpred_refrac = device_alloc_copy(cuda_hpred["refrac_until_ms"]);
    auto* d_hpred_leader = device_alloc_copy(cuda_hpred_leader);
    auto* d_hpred_preprobe = device_alloc_copy(cuda_hpred_preprobe);
    auto* d_hpred_trailer = device_alloc_copy(cuda_hpred_trailer);
    auto* d_hpred_flags = device_alloc_copy(cuda_hpred_flags);
    auto* d_stim_channel = device_alloc_copy(stim_channel);
    auto* d_event_steps = device_alloc_copy(event_steps);
    auto* d_event_sources = device_alloc_copy(event_sources);
    auto* d_source_by_step = device_alloc_copy(cuda_source_by_step);
    auto* d_source_by_afferent = device_alloc_copy(cuda_source_by_afferent);
    auto* d_source_by_channel = device_alloc_copy(cuda_source_by_channel);
    auto* d_source_by_phase = device_alloc_copy(cuda_source_by_phase);
    auto* d_stim_row_ptr = device_alloc_copy(stim_bank.row_ptr);
    auto* d_stim_post = device_alloc_copy(stim_bank.post);
    auto* d_stim_weight = device_alloc_copy(stim_bank.weight);
    auto* d_v1_to_h_row_ptr = device_alloc_copy(v1_to_h_bank.row_ptr);
    auto* d_v1_to_h_post = device_alloc_copy(v1_to_h_bank.post);
    auto* d_v1_to_h_weight = device_alloc_copy(v1_to_h_bank.weight);
    auto* d_ctx_to_pred_row_ptr = device_alloc_copy(ctx_to_pred_bank.row_ptr);
    auto* d_ctx_to_pred_post = device_alloc_copy(ctx_to_pred_bank.post);
    auto* d_ctx_to_pred_weight = device_alloc_copy(ctx_to_pred_bank.weight);
    auto* d_fb_direct_row_ptr = device_alloc_copy(fb_direct_bank.row_ptr);
    auto* d_fb_direct_post = device_alloc_copy(fb_direct_bank.post);
    auto* d_fb_direct_weight = device_alloc_copy(fb_direct_bank.weight);
    auto* d_fb_som_row_ptr = device_alloc_copy(fb_som_bank.row_ptr);
    auto* d_fb_som_post = device_alloc_copy(fb_som_bank.post);
    auto* d_fb_som_weight = device_alloc_copy(fb_som_bank.weight);

    const int event_blocks =
        (static_cast<int>(event_steps.size()) + 127) / 128;
    for (int step = 0; step < n_steps; ++step) {
        v1e_richter_count_step_flags_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_v1_v, d_v1_vap, d_v1_ie, d_v1_ii, d_v1_iape, d_v1_w,
            d_v1_ibias, d_v1_refrac, d_v1_leader, d_v1_preprobe,
            d_v1_trailer, d_v1_flags
        );
        check_cuda(cudaGetLastError(), "controlled source V1 step launch");
        v1som_step_kernel<<<1, 64>>>(
            step, d_som_v, d_som_ie, d_som_ii, d_som_ibias, d_som_refrac
        );
        check_cuda(cudaGetLastError(), "controlled source SOM step launch");
        he_richter_count_step_flags_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_hctx_v, d_hctx_ie, d_hctx_ii, d_hctx_g, d_hctx_ibias,
            d_hctx_refrac, d_hctx_leader, d_hctx_preprobe, d_hctx_trailer,
            d_hctx_flags
        );
        check_cuda(cudaGetLastError(), "controlled source H_ctx step launch");
        he_richter_count_step_flags_kernel<<<1, 256>>>(
            step, leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            d_hpred_v, d_hpred_ie, d_hpred_ii, d_hpred_g, d_hpred_ibias,
            d_hpred_refrac, d_hpred_leader, d_hpred_preprobe, d_hpred_trailer,
            d_hpred_flags
        );
        check_cuda(cudaGetLastError(), "controlled source H_pred step launch");
        controlled_stim_source_scatter_kernel<<<event_blocks, 128>>>(
            step, static_cast<int>(event_steps.size()), d_event_steps,
            d_event_sources, stim_bank.n_pre, d_stim_channel,
            leader_start_step, leader_end_step, preprobe_start_step,
            preprobe_end_step, trailer_start_step, trailer_end_step,
            iti_start_step, iti_end_step, d_stim_row_ptr, d_stim_post,
            d_stim_weight, stim_drive_amp, d_v1_ie, d_source_by_step,
            d_source_by_afferent, d_source_by_channel, d_source_by_phase
        );
        check_cuda(cudaGetLastError(), "controlled source stimulus scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            v1_to_h_bank.n_pre, d_v1_flags, d_v1_to_h_row_ptr, d_v1_to_h_post,
            d_v1_to_h_weight, v1_to_h_drive_amp, d_hctx_ie
        );
        check_cuda(cudaGetLastError(), "controlled source V1->H scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            ctx_to_pred_bank.n_pre, d_hctx_flags, d_ctx_to_pred_row_ptr,
            d_ctx_to_pred_post, d_ctx_to_pred_weight, ctx_to_pred_drive_amp,
            d_hpred_ie
        );
        check_cuda(cudaGetLastError(), "controlled source ctx->pred scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            fb_direct_bank.n_pre, d_hpred_flags, d_fb_direct_row_ptr,
            d_fb_direct_post, d_fb_direct_weight, feedback_direct_drive_amp,
            d_v1_iape
        );
        check_cuda(cudaGetLastError(), "controlled source feedback direct scatter launch");
        csr_scatter_spike_flags_kernel<<<2, 128>>>(
            fb_som_bank.n_pre, d_hpred_flags, d_fb_som_row_ptr, d_fb_som_post,
            d_fb_som_weight, feedback_som_drive_amp, d_som_ie
        );
        check_cuda(cudaGetLastError(), "controlled source feedback SOM scatter launch");
        check_cuda(cudaDeviceSynchronize(), "controlled source step sync");
    }

    copy_device_to_host(d_v1_v, cuda_v1["V_soma_mV"]);
    copy_device_to_host(d_v1_vap, cuda_v1["V_ap_mV"]);
    copy_device_to_host(d_v1_ie, cuda_v1["I_e_pA"]);
    copy_device_to_host(d_v1_ii, cuda_v1["I_i_pA"]);
    copy_device_to_host(d_v1_iape, cuda_v1["I_ap_e_pA"]);
    copy_device_to_host(d_v1_w, cuda_v1["w_adapt_pA"]);
    copy_device_to_host(d_v1_ibias, cuda_v1["I_bias_pA"]);
    copy_device_to_host(d_v1_refrac, cuda_v1["refrac_until_ms"]);
    copy_device_to_host(d_v1_leader, cuda_v1_leader);
    copy_device_to_host(d_v1_preprobe, cuda_v1_preprobe);
    copy_device_to_host(d_v1_trailer, cuda_v1_trailer);
    copy_device_to_host(d_som_v, cuda_som["V_mV"]);
    copy_device_to_host(d_som_ie, cuda_som["I_e_pA"]);
    copy_device_to_host(d_som_ii, cuda_som["I_i_pA"]);
    copy_device_to_host(d_som_ibias, cuda_som["I_bias_pA"]);
    copy_device_to_host(d_som_refrac, cuda_som["refrac_until_ms"]);
    copy_device_to_host(d_hctx_v, cuda_hctx["V_mV"]);
    copy_device_to_host(d_hctx_ie, cuda_hctx["I_e_pA"]);
    copy_device_to_host(d_hctx_ii, cuda_hctx["I_i_pA"]);
    copy_device_to_host(d_hctx_g, cuda_hctx["g_nmda_h_nS"]);
    copy_device_to_host(d_hctx_ibias, cuda_hctx["I_bias_pA"]);
    copy_device_to_host(d_hctx_refrac, cuda_hctx["refrac_until_ms"]);
    copy_device_to_host(d_hctx_leader, cuda_hctx_leader);
    copy_device_to_host(d_hctx_preprobe, cuda_hctx_preprobe);
    copy_device_to_host(d_hctx_trailer, cuda_hctx_trailer);
    copy_device_to_host(d_hpred_v, cuda_hpred["V_mV"]);
    copy_device_to_host(d_hpred_ie, cuda_hpred["I_e_pA"]);
    copy_device_to_host(d_hpred_ii, cuda_hpred["I_i_pA"]);
    copy_device_to_host(d_hpred_g, cuda_hpred["g_nmda_h_nS"]);
    copy_device_to_host(d_hpred_ibias, cuda_hpred["I_bias_pA"]);
    copy_device_to_host(d_hpred_refrac, cuda_hpred["refrac_until_ms"]);
    copy_device_to_host(d_hpred_leader, cuda_hpred_leader);
    copy_device_to_host(d_hpred_preprobe, cuda_hpred_preprobe);
    copy_device_to_host(d_hpred_trailer, cuda_hpred_trailer);
    copy_device_to_host(d_source_by_step, cuda_source_by_step);
    copy_device_to_host(d_source_by_afferent, cuda_source_by_afferent);
    copy_device_to_host(d_source_by_channel, cuda_source_by_channel);
    copy_device_to_host(d_source_by_phase, cuda_source_by_phase);

    result.cuda_raw_counts = {
        {"v1_e.leader", cuda_v1_leader},
        {"v1_e.preprobe", cuda_v1_preprobe},
        {"v1_e.trailer", cuda_v1_trailer},
        {"hctx_e.leader", cuda_hctx_leader},
        {"hctx_e.preprobe", cuda_hctx_preprobe},
        {"hctx_e.trailer", cuda_hctx_trailer},
        {"hpred_e.leader", cuda_hpred_leader},
        {"hpred_e.preprobe", cuda_hpred_preprobe},
        {"hpred_e.trailer", cuda_hpred_trailer},
    };
    result.cuda_source_counts = {
        {"source.events_by_step", cuda_source_by_step},
        {"source.events_by_afferent", cuda_source_by_afferent},
        {"source.events_by_channel", cuda_source_by_channel},
        {"source.events_by_phase", cuda_source_by_phase},
    };
    result.cuda_diagnostic_rates_hz = {
        {"v1_e.leader", counts_to_rate_hz(cuda_v1_leader, leader_start_step, leader_end_step)},
        {"v1_e.preprobe", counts_to_rate_hz(cuda_v1_preprobe, preprobe_start_step, preprobe_end_step)},
        {"v1_e.trailer", counts_to_rate_hz(cuda_v1_trailer, trailer_start_step, trailer_end_step)},
        {"hctx_e.preprobe", counts_to_rate_hz(cuda_hctx_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hctx_e.trailer", counts_to_rate_hz(cuda_hctx_trailer, trailer_start_step, trailer_end_step)},
        {"hpred_e.preprobe", counts_to_rate_hz(cuda_hpred_preprobe, preprobe_start_step, preprobe_end_step)},
        {"hpred_e.trailer", counts_to_rate_hz(cuda_hpred_trailer, trailer_start_step, trailer_end_step)},
    };
    append_prefixed_state(result.cuda_final_state, "v1e", cuda_v1);
    append_prefixed_state(result.cuda_final_state, "v1som", cuda_som);
    append_prefixed_state(result.cuda_final_state, "hctx", cuda_hctx);
    append_prefixed_state(result.cuda_final_state, "hpred", cuda_hpred);

    device_free(d_v1_v); device_free(d_v1_vap); device_free(d_v1_ie);
    device_free(d_v1_ii); device_free(d_v1_iape); device_free(d_v1_w);
    device_free(d_v1_ibias); device_free(d_v1_refrac);
    device_free(d_v1_leader); device_free(d_v1_preprobe);
    device_free(d_v1_trailer); device_free(d_v1_flags);
    device_free(d_som_v); device_free(d_som_ie); device_free(d_som_ii);
    device_free(d_som_ibias); device_free(d_som_refrac);
    device_free(d_hctx_v); device_free(d_hctx_ie); device_free(d_hctx_ii);
    device_free(d_hctx_g); device_free(d_hctx_ibias);
    device_free(d_hctx_refrac); device_free(d_hctx_leader);
    device_free(d_hctx_preprobe); device_free(d_hctx_trailer);
    device_free(d_hctx_flags);
    device_free(d_hpred_v); device_free(d_hpred_ie); device_free(d_hpred_ii);
    device_free(d_hpred_g); device_free(d_hpred_ibias);
    device_free(d_hpred_refrac); device_free(d_hpred_leader);
    device_free(d_hpred_preprobe); device_free(d_hpred_trailer);
    device_free(d_hpred_flags);
    device_free(d_stim_channel); device_free(d_event_steps);
    device_free(d_event_sources); device_free(d_source_by_step);
    device_free(d_source_by_afferent); device_free(d_source_by_channel);
    device_free(d_source_by_phase);
    device_free(d_stim_row_ptr); device_free(d_stim_post); device_free(d_stim_weight);
    device_free(d_v1_to_h_row_ptr); device_free(d_v1_to_h_post);
    device_free(d_v1_to_h_weight);
    device_free(d_ctx_to_pred_row_ptr); device_free(d_ctx_to_pred_post);
    device_free(d_ctx_to_pred_weight);
    device_free(d_fb_direct_row_ptr); device_free(d_fb_direct_post);
    device_free(d_fb_direct_weight);
    device_free(d_fb_som_row_ptr); device_free(d_fb_som_post);
    device_free(d_fb_som_weight);

    result.max_abs_error = compare_state(result.cpu_final_state, result.cuda_final_state);
    for (const auto& [key, cpu_counts] : result.cpu_raw_counts) {
        result.max_abs_error["raw_counts." + key] =
            max_count_abs_diff(cpu_counts, result.cuda_raw_counts.at(key));
    }
    for (const auto& [key, cpu_counts] : result.cpu_source_counts) {
        result.max_abs_error[key] =
            max_count_abs_diff(cpu_counts, result.cuda_source_counts.at(key));
    }
    for (const auto& [key, cpu_rates] : result.cpu_diagnostic_rates_hz) {
        result.max_abs_error["rates." + key] =
            max_abs_diff(cpu_rates, result.cuda_diagnostic_rates_hz.at(key));
    }
    return result;
}

CtxPredPlasticityTestResult run_ctx_pred_plasticity_test(
    std::int64_t seed,
    std::int32_t n_steps
) {
    const std::vector<std::int32_t> pre_event_steps{10, 540};
    const std::vector<std::int32_t> pre_event_cells{
        CTX_PRED_PAIRED_PRE,
        CTX_PRED_PRE_RULE_PRE,
    };
    const std::vector<std::int32_t> post_event_steps{510};
    const std::vector<std::int32_t> post_event_cells{CTX_PRED_PAIRED_POST};
    if (n_steps <= pre_event_steps.back()) {
        throw std::runtime_error(
            "ctx_pred plasticity test n_steps must exceed final event step"
        );
    }

    CtxPredPlasticityTestResult result;
    result.seed = seed;
    result.n_pre = CTX_PRED_N_PRE;
    result.n_post = CTX_PRED_N_POST;
    result.n_syn = CTX_PRED_N_SYN;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    result.tau_coinc_ms = CTX_PRED_TAU_COINC_MS;
    result.tau_elig_ms = CTX_PRED_TAU_ELIG_MS;
    result.eta = CTX_PRED_ETA;
    result.gamma = CTX_PRED_GAMMA;
    result.w_target = CTX_PRED_W_TARGET;
    result.w_max = CTX_PRED_W_MAX;
    result.w_row_max = CTX_PRED_W_ROW_MAX;
    result.m_integral = CTX_PRED_M_INTEGRAL;
    result.dt_trial_s = CTX_PRED_DT_TRIAL_S;
    result.paired_pre = CTX_PRED_PAIRED_PRE;
    result.paired_post = CTX_PRED_PAIRED_POST;
    result.pre_rule_pre = CTX_PRED_PRE_RULE_PRE;
    result.capped_pre = CTX_PRED_CAPPED_PRE;
    result.silent_pre = CTX_PRED_SILENT_PRE;
    result.silent_post = CTX_PRED_SILENT_POST;
    result.pre_event_steps = pre_event_steps;
    result.pre_event_cells = pre_event_cells;
    result.post_event_steps = post_event_steps;
    result.post_event_cells = post_event_cells;

    result.initial_w = init_ctx_pred_test_weights(seed);

    std::vector<double> cpu_w = result.initial_w;
    std::vector<double> cpu_xpre(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cpu_xpost(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cpu_elig(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);

    for (int step = 0; step < n_steps; ++step) {
        cpu_ctx_pred_decay_step(cpu_xpre, cpu_xpost, cpu_elig);
        for (std::size_t e = 0; e < pre_event_steps.size(); ++e) {
            if (pre_event_steps[e] == step) {
                cpu_ctx_pred_pre_event(
                    cpu_xpre, cpu_xpost, cpu_elig, pre_event_cells[e]
                );
            }
        }
        for (std::size_t e = 0; e < post_event_steps.size(); ++e) {
            if (post_event_steps[e] == step) {
                cpu_ctx_pred_post_event(
                    cpu_xpre, cpu_xpost, cpu_elig, post_event_cells[e]
                );
            }
        }
    }

    result.cpu_elig_before_gate = cpu_elig;
    result.cpu_n_capped = cpu_ctx_pred_gate_update(cpu_w, cpu_elig);
    result.cpu_w = cpu_w;
    result.cpu_elig_after_gate = cpu_elig;
    result.cpu_xpre_after_gate = cpu_xpre;
    result.cpu_xpost_after_gate = cpu_xpost;
    result.cpu_row_sums = ctx_pred_row_sums(cpu_w);

    std::vector<double> cuda_w = result.initial_w;
    std::vector<double> cuda_xpre(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cuda_xpost(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cuda_elig(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<int> cuda_capped_flags(static_cast<std::size_t>(CTX_PRED_N_PRE), 0);

    double* d_w = device_alloc_copy(cuda_w);
    double* d_xpre = device_alloc_copy(cuda_xpre);
    double* d_xpost = device_alloc_copy(cuda_xpost);
    double* d_elig = device_alloc_copy(cuda_elig);
    int* d_capped_flags = device_alloc_copy(cuda_capped_flags);

    const int block = 256;
    const int syn_grid = (CTX_PRED_N_SYN + block - 1) / block;
    const int row_grid = (CTX_PRED_N_PRE + block - 1) / block;
    const double decay_coinc = std::exp(-DT_MS / CTX_PRED_TAU_COINC_MS);
    const double decay_elig = std::exp(-DT_MS / CTX_PRED_TAU_ELIG_MS);

    for (int step = 0; step < n_steps; ++step) {
        ctx_pred_decay_kernel<<<syn_grid, block>>>(
            d_xpre, d_xpost, d_elig, CTX_PRED_N_SYN, decay_coinc, decay_elig
        );
        check_cuda(cudaGetLastError(), "ctx_pred_decay_kernel launch");
        for (std::size_t e = 0; e < pre_event_steps.size(); ++e) {
            if (pre_event_steps[e] == step) {
                ctx_pred_pre_event_kernel<<<1, block>>>(
                    d_xpre, d_xpost, d_elig, pre_event_cells[e], CTX_PRED_N_POST
                );
                check_cuda(
                    cudaGetLastError(), "ctx_pred_pre_event_kernel launch"
                );
            }
        }
        for (std::size_t e = 0; e < post_event_steps.size(); ++e) {
            if (post_event_steps[e] == step) {
                ctx_pred_post_event_kernel<<<1, block>>>(
                    d_xpre,
                    d_xpost,
                    d_elig,
                    post_event_cells[e],
                    CTX_PRED_N_PRE,
                    CTX_PRED_N_POST
                );
                check_cuda(
                    cudaGetLastError(), "ctx_pred_post_event_kernel launch"
                );
            }
        }
    }

    check_cuda(cudaDeviceSynchronize(), "ctx_pred plasticity trace sync");
    copy_device_to_host(d_elig, cuda_elig);
    result.cuda_elig_before_gate = cuda_elig;

    ctx_pred_gate_preclip_kernel<<<syn_grid, block>>>(
        d_w,
        d_elig,
        CTX_PRED_N_SYN,
        CTX_PRED_ETA,
        CTX_PRED_M_INTEGRAL,
        CTX_PRED_GAMMA,
        CTX_PRED_W_TARGET,
        CTX_PRED_DT_TRIAL_S,
        CTX_PRED_W_MAX
    );
    check_cuda(cudaGetLastError(), "ctx_pred_gate_preclip_kernel launch");
    ctx_pred_row_cap_kernel<<<row_grid, block>>>(
        d_w,
        CTX_PRED_N_PRE,
        CTX_PRED_N_POST,
        CTX_PRED_W_ROW_MAX,
        d_capped_flags
    );
    check_cuda(cudaGetLastError(), "ctx_pred_row_cap_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "ctx_pred plasticity gate sync");

    copy_device_to_host(d_w, cuda_w);
    copy_device_to_host(d_xpre, cuda_xpre);
    copy_device_to_host(d_xpost, cuda_xpost);
    copy_device_to_host(d_elig, cuda_elig);
    copy_device_to_host(d_capped_flags, cuda_capped_flags);

    result.cuda_w = cuda_w;
    result.cuda_elig_after_gate = cuda_elig;
    result.cuda_xpre_after_gate = cuda_xpre;
    result.cuda_xpost_after_gate = cuda_xpost;
    result.cuda_row_sums = ctx_pred_row_sums(cuda_w);
    result.cuda_n_capped = std::accumulate(
        cuda_capped_flags.begin(), cuda_capped_flags.end(), 0
    );

    device_free(d_w);
    device_free(d_xpre);
    device_free(d_xpost);
    device_free(d_elig);
    device_free(d_capped_flags);

    result.max_abs_error = {
        {"w", max_abs_diff(result.cpu_w, result.cuda_w)},
        {
            "elig_before_gate",
            max_abs_diff(result.cpu_elig_before_gate, result.cuda_elig_before_gate),
        },
        {
            "elig_after_gate",
            max_abs_diff(result.cpu_elig_after_gate, result.cuda_elig_after_gate),
        },
        {
            "xpre_after_gate",
            max_abs_diff(result.cpu_xpre_after_gate, result.cuda_xpre_after_gate),
        },
        {
            "xpost_after_gate",
            max_abs_diff(result.cpu_xpost_after_gate, result.cuda_xpost_after_gate),
        },
        {"row_sums", max_abs_diff(result.cpu_row_sums, result.cuda_row_sums)},
        {
            "n_capped",
            std::abs(
                static_cast<double>(result.cpu_n_capped - result.cuda_n_capped)
            ),
        },
    };
    return result;
}

CtxPredTrainingTrialSliceResult run_ctx_pred_training_trial_slice_test(
    std::int64_t seed
) {
    const int leader_start_step = 0;
    const int leader_end_step = 80;
    const int trailer_start_step = 80;
    const int trailer_end_step = 180;
    const int iti_start_step = 180;
    const int iti_end_step = 220;
    const int n_steps = iti_end_step;
    const int gate_step = trailer_end_step;

    const std::vector<std::int32_t> hctx_pre_event_steps{
        leader_start_step + 10,
        leader_end_step - 1,
    };
    const std::vector<std::int32_t> hctx_pre_event_cells{
        CTX_PRED_TRIAL_LEADER_PRE,
        CTX_PRED_TRIAL_BOUNDARY_PRE,
    };
    const std::vector<std::int32_t> hpred_post_event_steps{
        trailer_start_step,
        trailer_end_step - 1,
    };
    const std::vector<std::int32_t> hpred_post_event_cells{
        CTX_PRED_TRIAL_TRAILER_POST,
        CTX_PRED_TRIAL_LATE_TRAILER_POST,
    };

    CtxPredTrainingTrialSliceResult result;
    result.seed = seed;
    result.n_pre = CTX_PRED_N_PRE;
    result.n_post = CTX_PRED_N_POST;
    result.n_syn = CTX_PRED_N_SYN;
    result.n_steps = n_steps;
    result.dt_ms = DT_MS;
    result.phase_steps = {
        {"leader_start_step", leader_start_step},
        {"leader_end_step", leader_end_step},
        {"trailer_start_step", trailer_start_step},
        {"trailer_end_step", trailer_end_step},
        {"iti_start_step", iti_start_step},
        {"iti_end_step", iti_end_step},
    };
    result.gate_step = gate_step;
    result.leader_pre = CTX_PRED_TRIAL_LEADER_PRE;
    result.boundary_pre = CTX_PRED_TRIAL_BOUNDARY_PRE;
    result.trailer_post = CTX_PRED_TRIAL_TRAILER_POST;
    result.late_trailer_post = CTX_PRED_TRIAL_LATE_TRAILER_POST;
    result.capped_pre = CTX_PRED_CAPPED_PRE;
    result.silent_pre = CTX_PRED_SILENT_PRE;
    result.silent_post = CTX_PRED_SILENT_POST;
    result.hctx_pre_event_steps = hctx_pre_event_steps;
    result.hctx_pre_event_cells = hctx_pre_event_cells;
    result.hpred_post_event_steps = hpred_post_event_steps;
    result.hpred_post_event_cells = hpred_post_event_cells;
    result.event_counts = {
        {"hctx_pre.leader", 0},
        {"hctx_pre.trailer", 0},
        {"hctx_pre.iti", 0},
        {"hctx_pre.outside", 0},
        {"hpred_post.leader", 0},
        {"hpred_post.trailer", 0},
        {"hpred_post.iti", 0},
        {"hpred_post.outside", 0},
    };
    for (const int step : hctx_pre_event_steps) {
        add_stage1_event_count(
            result.event_counts,
            "hctx_pre",
            step,
            leader_start_step,
            leader_end_step,
            trailer_start_step,
            trailer_end_step,
            iti_start_step,
            iti_end_step
        );
    }
    for (const int step : hpred_post_event_steps) {
        add_stage1_event_count(
            result.event_counts,
            "hpred_post",
            step,
            leader_start_step,
            leader_end_step,
            trailer_start_step,
            trailer_end_step,
            iti_start_step,
            iti_end_step
        );
    }

    result.initial_w_ctx_pred = init_ctx_pred_test_weights(seed);

    std::vector<double> cpu_w = result.initial_w_ctx_pred;
    std::vector<double> cpu_xpre(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cpu_xpost(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cpu_elig(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);

    for (int step = 0; step < n_steps; ++step) {
        if (step == gate_step) {
            result.cpu_elig_before_gate = cpu_elig;
            result.cpu_n_capped = cpu_ctx_pred_gate_update(cpu_w, cpu_elig);
        }
        cpu_ctx_pred_decay_step(cpu_xpre, cpu_xpost, cpu_elig);
        for (std::size_t e = 0; e < hctx_pre_event_steps.size(); ++e) {
            if (hctx_pre_event_steps[e] == step) {
                cpu_ctx_pred_pre_event(
                    cpu_xpre, cpu_xpost, cpu_elig, hctx_pre_event_cells[e]
                );
            }
        }
        for (std::size_t e = 0; e < hpred_post_event_steps.size(); ++e) {
            if (hpred_post_event_steps[e] == step) {
                cpu_ctx_pred_post_event(
                    cpu_xpre, cpu_xpost, cpu_elig, hpred_post_event_cells[e]
                );
            }
        }
    }
    result.cpu_w_ctx_pred_final = cpu_w;
    result.cpu_elig_after_iti = cpu_elig;
    result.cpu_xpre_after_iti = cpu_xpre;
    result.cpu_xpost_after_iti = cpu_xpost;
    result.cpu_row_sums = ctx_pred_row_sums(cpu_w);

    std::vector<double> cuda_w = result.initial_w_ctx_pred;
    std::vector<double> cuda_xpre(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cuda_xpost(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cuda_elig(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<int> cuda_capped_flags(static_cast<std::size_t>(CTX_PRED_N_PRE), 0);

    double* d_w = device_alloc_copy(cuda_w);
    double* d_xpre = device_alloc_copy(cuda_xpre);
    double* d_xpost = device_alloc_copy(cuda_xpost);
    double* d_elig = device_alloc_copy(cuda_elig);
    int* d_capped_flags = device_alloc_copy(cuda_capped_flags);

    const int block = 256;
    const int syn_grid = (CTX_PRED_N_SYN + block - 1) / block;
    const int row_grid = (CTX_PRED_N_PRE + block - 1) / block;
    const double decay_coinc = std::exp(-DT_MS / CTX_PRED_TAU_COINC_MS);
    const double decay_elig = std::exp(-DT_MS / CTX_PRED_TAU_ELIG_MS);

    for (int step = 0; step < n_steps; ++step) {
        if (step == gate_step) {
            check_cuda(cudaDeviceSynchronize(), "ctx_pred trial pre-gate sync");
            copy_device_to_host(d_elig, cuda_elig);
            result.cuda_elig_before_gate = cuda_elig;
            ctx_pred_gate_preclip_kernel<<<syn_grid, block>>>(
                d_w,
                d_elig,
                CTX_PRED_N_SYN,
                CTX_PRED_ETA,
                CTX_PRED_M_INTEGRAL,
                CTX_PRED_GAMMA,
                CTX_PRED_W_TARGET,
                CTX_PRED_DT_TRIAL_S,
                CTX_PRED_W_MAX
            );
            check_cuda(
                cudaGetLastError(),
                "ctx_pred trial gate preclip kernel launch"
            );
            ctx_pred_row_cap_kernel<<<row_grid, block>>>(
                d_w,
                CTX_PRED_N_PRE,
                CTX_PRED_N_POST,
                CTX_PRED_W_ROW_MAX,
                d_capped_flags
            );
            check_cuda(
                cudaGetLastError(),
                "ctx_pred trial row cap kernel launch"
            );
            check_cuda(cudaDeviceSynchronize(), "ctx_pred trial gate sync");
        }
        ctx_pred_decay_kernel<<<syn_grid, block>>>(
            d_xpre, d_xpost, d_elig, CTX_PRED_N_SYN, decay_coinc, decay_elig
        );
        check_cuda(cudaGetLastError(), "ctx_pred trial decay kernel launch");
        for (std::size_t e = 0; e < hctx_pre_event_steps.size(); ++e) {
            if (hctx_pre_event_steps[e] == step) {
                ctx_pred_pre_event_kernel<<<1, block>>>(
                    d_xpre,
                    d_xpost,
                    d_elig,
                    hctx_pre_event_cells[e],
                    CTX_PRED_N_POST
                );
                check_cuda(
                    cudaGetLastError(),
                    "ctx_pred trial pre event kernel launch"
                );
            }
        }
        for (std::size_t e = 0; e < hpred_post_event_steps.size(); ++e) {
            if (hpred_post_event_steps[e] == step) {
                ctx_pred_post_event_kernel<<<1, block>>>(
                    d_xpre,
                    d_xpost,
                    d_elig,
                    hpred_post_event_cells[e],
                    CTX_PRED_N_PRE,
                    CTX_PRED_N_POST
                );
                check_cuda(
                    cudaGetLastError(),
                    "ctx_pred trial post event kernel launch"
                );
            }
        }
    }

    check_cuda(cudaDeviceSynchronize(), "ctx_pred trial final sync");
    copy_device_to_host(d_w, cuda_w);
    copy_device_to_host(d_xpre, cuda_xpre);
    copy_device_to_host(d_xpost, cuda_xpost);
    copy_device_to_host(d_elig, cuda_elig);
    copy_device_to_host(d_capped_flags, cuda_capped_flags);

    result.cuda_w_ctx_pred_final = cuda_w;
    result.cuda_elig_after_iti = cuda_elig;
    result.cuda_xpre_after_iti = cuda_xpre;
    result.cuda_xpost_after_iti = cuda_xpost;
    result.cuda_row_sums = ctx_pred_row_sums(cuda_w);
    result.cuda_n_capped = std::accumulate(
        cuda_capped_flags.begin(), cuda_capped_flags.end(), 0
    );

    device_free(d_w);
    device_free(d_xpre);
    device_free(d_xpost);
    device_free(d_elig);
    device_free(d_capped_flags);

    result.max_abs_error = {
        {
            "w_ctx_pred_final",
            max_abs_diff(
                result.cpu_w_ctx_pred_final,
                result.cuda_w_ctx_pred_final
            ),
        },
        {
            "elig_before_gate",
            max_abs_diff(result.cpu_elig_before_gate, result.cuda_elig_before_gate),
        },
        {
            "elig_after_iti",
            max_abs_diff(result.cpu_elig_after_iti, result.cuda_elig_after_iti),
        },
        {
            "xpre_after_iti",
            max_abs_diff(result.cpu_xpre_after_iti, result.cuda_xpre_after_iti),
        },
        {
            "xpost_after_iti",
            max_abs_diff(result.cpu_xpost_after_iti, result.cuda_xpost_after_iti),
        },
        {"row_sums", max_abs_diff(result.cpu_row_sums, result.cuda_row_sums)},
        {
            "n_capped",
            std::abs(
                static_cast<double>(result.cpu_n_capped - result.cuda_n_capped)
            ),
        },
    };
    return result;
}

static CtxPredTinyTrainerTestResult run_ctx_pred_controlled_trainer_impl(
    std::int64_t seed,
    std::int32_t schedule_variant,
    const std::vector<std::int32_t>& leader_pre_cells,
    const std::vector<std::int32_t>& trailer_post_cells
) {
    if (leader_pre_cells.empty()) {
        throw std::runtime_error("native Stage1 trainer schedule cannot be empty");
    }
    if (leader_pre_cells.size() != trailer_post_cells.size()) {
        throw std::runtime_error(
            "leader_pre_cells and trailer_post_cells must have equal length"
        );
    }
    for (const int cell : leader_pre_cells) {
        if (cell < 0 || cell >= CTX_PRED_N_PRE) {
            throw std::runtime_error("leader_pre_cells contains out-of-range cell");
        }
    }
    for (const int cell : trailer_post_cells) {
        if (cell < 0 || cell >= CTX_PRED_N_POST) {
            throw std::runtime_error("trailer_post_cells contains out-of-range cell");
        }
    }

    const int n_trials = static_cast<int>(leader_pre_cells.size());
    const int leader_start_rel = 0;
    const int leader_end_rel = 80;
    const int trailer_start_rel = 80;
    const int trailer_end_rel = 180;
    const int iti_start_rel = 180;
    const int iti_end_rel = 220;
    const int trial_steps = iti_end_rel;
    const int n_steps = n_trials * trial_steps;

    CtxPredTinyTrainerTestResult result;
    result.seed = seed;
    result.schedule_variant = schedule_variant;
    result.n_trials = n_trials;
    result.n_pre = CTX_PRED_N_PRE;
    result.n_post = CTX_PRED_N_POST;
    result.n_syn = CTX_PRED_N_SYN;
    result.h_ee_n_syn = H_E_N * (H_E_N - 1);
    result.n_steps = n_steps;
    result.trial_steps = trial_steps;
    result.dt_ms = DT_MS;
    result.phase_steps = {
        {"leader_start_step", leader_start_rel},
        {"leader_end_step", leader_end_rel},
        {"trailer_start_step", trailer_start_rel},
        {"trailer_end_step", trailer_end_rel},
        {"iti_start_step", iti_start_rel},
        {"iti_end_step", iti_end_rel},
    };
    result.event_counts = {
        {"hctx_pre.leader", 0},
        {"hctx_pre.trailer", 0},
        {"hctx_pre.iti", 0},
        {"hctx_pre.outside", 0},
        {"hpred_post.leader", 0},
        {"hpred_post.trailer", 0},
        {"hpred_post.iti", 0},
        {"hpred_post.outside", 0},
    };

    for (int trial = 0; trial < n_trials; ++trial) {
        const int trial_offset = trial * trial_steps;
        const int leader_pre = leader_pre_cells[static_cast<std::size_t>(trial)];
        const int trailer_post = trailer_post_cells[static_cast<std::size_t>(trial)];
        const int boundary_pre = (
            schedule_variant == 0
            ? 60 + trial
            : (schedule_variant == 1
               ? 120 + trial
               : (leader_pre + 32) % CTX_PRED_N_PRE)
        );
        const int late_trailer_post = (
            schedule_variant == 0 || schedule_variant == 1
            ? 90 + trial
            : (trailer_post + 64) % CTX_PRED_N_POST
        );
        result.trial_leader_pre_cells.push_back(leader_pre);
        result.trial_trailer_post_cells.push_back(trailer_post);
        result.gate_steps.push_back(trial_offset + trailer_end_rel);

        result.hctx_pre_event_steps.push_back(trial_offset + 10);
        result.hctx_pre_event_cells.push_back(leader_pre);
        result.hctx_pre_event_steps.push_back(trial_offset + leader_end_rel - 1);
        result.hctx_pre_event_cells.push_back(boundary_pre);
        result.hpred_post_event_steps.push_back(trial_offset + trailer_start_rel);
        result.hpred_post_event_cells.push_back(trailer_post);
        result.hpred_post_event_steps.push_back(trial_offset + trailer_end_rel - 1);
        result.hpred_post_event_cells.push_back(late_trailer_post);
    }

    for (const int step : result.hctx_pre_event_steps) {
        const int rel_step = step % trial_steps;
        add_stage1_event_count(
            result.event_counts,
            "hctx_pre",
            rel_step,
            leader_start_rel,
            leader_end_rel,
            trailer_start_rel,
            trailer_end_rel,
            iti_start_rel,
            iti_end_rel
        );
    }
    for (const int step : result.hpred_post_event_steps) {
        const int rel_step = step % trial_steps;
        add_stage1_event_count(
            result.event_counts,
            "hpred_post",
            rel_step,
            leader_start_rel,
            leader_end_rel,
            trailer_start_rel,
            trailer_end_rel,
            iti_start_rel,
            iti_end_rel
        );
    }

    result.initial_w_ctx_pred = init_ctx_pred_test_weights(seed);
    result.cpu_ctx_ee_w_final = init_h_ee_placeholder_weights(seed, 101);
    result.cuda_ctx_ee_w_final = result.cpu_ctx_ee_w_final;
    result.cpu_pred_ee_w_final = init_h_ee_placeholder_weights(seed, 202);
    result.cuda_pred_ee_w_final = result.cpu_pred_ee_w_final;

    std::vector<double> cpu_w = result.initial_w_ctx_pred;
    std::vector<double> cpu_xpre(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cpu_xpost(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cpu_elig(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);

    auto append_cpu_gate_telemetry =
        [&](const std::vector<double>& w_before,
            const std::vector<double>& w_after,
            const std::vector<double>& elig_before,
            int n_capped) {
            result.cpu_gate_w_before.push_back(vector_mean(w_before));
            result.cpu_gate_w_after.push_back(vector_mean(w_after));
            result.cpu_gate_dw_sum.push_back(sum_delta(w_before, w_after));
            result.cpu_gate_elig_mean.push_back(vector_mean(elig_before));
            result.cpu_gate_elig_max.push_back(vector_max_value(elig_before));
            result.cpu_gate_n_capped.push_back(n_capped);
            result.cpu_gate_row_sum_max.push_back(
                vector_max_value(ctx_pred_row_sums(w_after))
            );
        };

    for (int step = 0; step < n_steps; ++step) {
        if (std::find(result.gate_steps.begin(), result.gate_steps.end(), step)
            != result.gate_steps.end()) {
            const std::vector<double> w_before = cpu_w;
            const std::vector<double> elig_before = cpu_elig;
            const int n_capped = cpu_ctx_pred_gate_update(cpu_w, cpu_elig);
            append_cpu_gate_telemetry(w_before, cpu_w, elig_before, n_capped);
        }
        cpu_ctx_pred_decay_step(cpu_xpre, cpu_xpost, cpu_elig);
        for (std::size_t e = 0; e < result.hctx_pre_event_steps.size(); ++e) {
            if (result.hctx_pre_event_steps[e] == step) {
                cpu_ctx_pred_pre_event(
                    cpu_xpre,
                    cpu_xpost,
                    cpu_elig,
                    result.hctx_pre_event_cells[e]
                );
            }
        }
        for (std::size_t e = 0; e < result.hpred_post_event_steps.size(); ++e) {
            if (result.hpred_post_event_steps[e] == step) {
                cpu_ctx_pred_post_event(
                    cpu_xpre,
                    cpu_xpost,
                    cpu_elig,
                    result.hpred_post_event_cells[e]
                );
            }
        }
    }
    result.cpu_w_ctx_pred_final = cpu_w;
    result.cpu_elig_after_training = cpu_elig;
    result.cpu_xpre_after_training = cpu_xpre;
    result.cpu_xpost_after_training = cpu_xpost;
    result.cpu_row_sums = ctx_pred_row_sums(cpu_w);

    std::vector<double> cuda_w = result.initial_w_ctx_pred;
    std::vector<double> cuda_xpre(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cuda_xpost(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<double> cuda_elig(static_cast<std::size_t>(CTX_PRED_N_SYN), 0.0);
    std::vector<int> cuda_capped_flags(static_cast<std::size_t>(CTX_PRED_N_PRE), 0);

    auto* d_w = device_alloc_copy(cuda_w);
    auto* d_xpre = device_alloc_copy(cuda_xpre);
    auto* d_xpost = device_alloc_copy(cuda_xpost);
    auto* d_elig = device_alloc_copy(cuda_elig);
    auto* d_capped_flags = device_alloc_copy(cuda_capped_flags);

    const int block = 256;
    const int syn_grid = (CTX_PRED_N_SYN + block - 1) / block;
    const int row_grid = (CTX_PRED_N_PRE + block - 1) / block;
    const double decay_coinc = std::exp(-DT_MS / CTX_PRED_TAU_COINC_MS);
    const double decay_elig = std::exp(-DT_MS / CTX_PRED_TAU_ELIG_MS);

    auto append_cuda_gate_telemetry =
        [&](const std::vector<double>& w_before,
            const std::vector<double>& w_after,
            const std::vector<double>& elig_before,
            int n_capped) {
            result.cuda_gate_w_before.push_back(vector_mean(w_before));
            result.cuda_gate_w_after.push_back(vector_mean(w_after));
            result.cuda_gate_dw_sum.push_back(sum_delta(w_before, w_after));
            result.cuda_gate_elig_mean.push_back(vector_mean(elig_before));
            result.cuda_gate_elig_max.push_back(vector_max_value(elig_before));
            result.cuda_gate_n_capped.push_back(n_capped);
            result.cuda_gate_row_sum_max.push_back(
                vector_max_value(ctx_pred_row_sums(w_after))
            );
        };

    for (int step = 0; step < n_steps; ++step) {
        if (std::find(result.gate_steps.begin(), result.gate_steps.end(), step)
            != result.gate_steps.end()) {
            std::vector<double> w_before(static_cast<std::size_t>(CTX_PRED_N_SYN));
            std::vector<double> elig_before(static_cast<std::size_t>(CTX_PRED_N_SYN));
            check_cuda(cudaDeviceSynchronize(), "ctx_pred tiny pre-gate sync");
            copy_device_to_host(d_w, w_before);
            copy_device_to_host(d_elig, elig_before);
            ctx_pred_gate_preclip_kernel<<<syn_grid, block>>>(
                d_w,
                d_elig,
                CTX_PRED_N_SYN,
                CTX_PRED_ETA,
                CTX_PRED_M_INTEGRAL,
                CTX_PRED_GAMMA,
                CTX_PRED_W_TARGET,
                CTX_PRED_DT_TRIAL_S,
                CTX_PRED_W_MAX
            );
            check_cuda(
                cudaGetLastError(),
                "ctx_pred tiny gate preclip kernel launch"
            );
            ctx_pred_row_cap_kernel<<<row_grid, block>>>(
                d_w,
                CTX_PRED_N_PRE,
                CTX_PRED_N_POST,
                CTX_PRED_W_ROW_MAX,
                d_capped_flags
            );
            check_cuda(
                cudaGetLastError(),
                "ctx_pred tiny row cap kernel launch"
            );
            check_cuda(cudaDeviceSynchronize(), "ctx_pred tiny gate sync");
            std::vector<double> w_after(static_cast<std::size_t>(CTX_PRED_N_SYN));
            copy_device_to_host(d_w, w_after);
            copy_device_to_host(d_capped_flags, cuda_capped_flags);
            append_cuda_gate_telemetry(
                w_before,
                w_after,
                elig_before,
                std::accumulate(cuda_capped_flags.begin(), cuda_capped_flags.end(), 0)
            );
        }
        ctx_pred_decay_kernel<<<syn_grid, block>>>(
            d_xpre, d_xpost, d_elig, CTX_PRED_N_SYN, decay_coinc, decay_elig
        );
        check_cuda(cudaGetLastError(), "ctx_pred tiny decay kernel launch");
        for (std::size_t e = 0; e < result.hctx_pre_event_steps.size(); ++e) {
            if (result.hctx_pre_event_steps[e] == step) {
                ctx_pred_pre_event_kernel<<<1, block>>>(
                    d_xpre,
                    d_xpost,
                    d_elig,
                    result.hctx_pre_event_cells[e],
                    CTX_PRED_N_POST
                );
                check_cuda(
                    cudaGetLastError(),
                    "ctx_pred tiny pre event kernel launch"
                );
            }
        }
        for (std::size_t e = 0; e < result.hpred_post_event_steps.size(); ++e) {
            if (result.hpred_post_event_steps[e] == step) {
                ctx_pred_post_event_kernel<<<1, block>>>(
                    d_xpre,
                    d_xpost,
                    d_elig,
                    result.hpred_post_event_cells[e],
                    CTX_PRED_N_PRE,
                    CTX_PRED_N_POST
                );
                check_cuda(
                    cudaGetLastError(),
                    "ctx_pred tiny post event kernel launch"
                );
            }
        }
    }

    check_cuda(cudaDeviceSynchronize(), "ctx_pred tiny final sync");
    copy_device_to_host(d_w, cuda_w);
    copy_device_to_host(d_xpre, cuda_xpre);
    copy_device_to_host(d_xpost, cuda_xpost);
    copy_device_to_host(d_elig, cuda_elig);
    copy_device_to_host(d_capped_flags, cuda_capped_flags);

    result.cuda_w_ctx_pred_final = cuda_w;
    result.cuda_elig_after_training = cuda_elig;
    result.cuda_xpre_after_training = cuda_xpre;
    result.cuda_xpost_after_training = cuda_xpost;
    result.cuda_row_sums = ctx_pred_row_sums(cuda_w);

    device_free(d_w);
    device_free(d_xpre);
    device_free(d_xpost);
    device_free(d_elig);
    device_free(d_capped_flags);

    result.max_abs_error = {
        {
            "w_ctx_pred_final",
            max_abs_diff(
                result.cpu_w_ctx_pred_final,
                result.cuda_w_ctx_pred_final
            ),
        },
        {
            "ctx_ee_w_final",
            max_abs_diff(result.cpu_ctx_ee_w_final, result.cuda_ctx_ee_w_final),
        },
        {
            "pred_ee_w_final",
            max_abs_diff(result.cpu_pred_ee_w_final, result.cuda_pred_ee_w_final),
        },
        {
            "elig_after_training",
            max_abs_diff(
                result.cpu_elig_after_training,
                result.cuda_elig_after_training
            ),
        },
        {
            "xpre_after_training",
            max_abs_diff(
                result.cpu_xpre_after_training,
                result.cuda_xpre_after_training
            ),
        },
        {
            "xpost_after_training",
            max_abs_diff(
                result.cpu_xpost_after_training,
                result.cuda_xpost_after_training
            ),
        },
        {"row_sums", max_abs_diff(result.cpu_row_sums, result.cuda_row_sums)},
        {
            "gate_w_before",
            max_abs_diff(result.cpu_gate_w_before, result.cuda_gate_w_before),
        },
        {
            "gate_w_after",
            max_abs_diff(result.cpu_gate_w_after, result.cuda_gate_w_after),
        },
        {
            "gate_dw_sum",
            max_abs_diff(result.cpu_gate_dw_sum, result.cuda_gate_dw_sum),
        },
        {
            "gate_elig_mean",
            max_abs_diff(result.cpu_gate_elig_mean, result.cuda_gate_elig_mean),
        },
        {
            "gate_elig_max",
            max_abs_diff(result.cpu_gate_elig_max, result.cuda_gate_elig_max),
        },
        {
            "gate_row_sum_max",
            max_abs_diff(
                result.cpu_gate_row_sum_max,
                result.cuda_gate_row_sum_max
            ),
        },
        {
            "gate_n_capped",
            max_count_abs_diff(
                result.cpu_gate_n_capped,
                result.cuda_gate_n_capped
            ),
        },
    };
    return result;
}

CtxPredTinyTrainerTestResult run_ctx_pred_tiny_trainer_test(
    std::int64_t seed,
    std::int32_t schedule_variant
) {
    if (schedule_variant < 0 || schedule_variant > 1) {
        throw std::runtime_error("tiny trainer schedule_variant must be 0 or 1");
    }
    std::vector<std::int32_t> leader_pre_cells;
    std::vector<std::int32_t> trailer_post_cells;
    leader_pre_cells.reserve(5);
    trailer_post_cells.reserve(5);
    for (int trial = 0; trial < 5; ++trial) {
        leader_pre_cells.push_back((schedule_variant == 0 ? 20 : 90) + trial);
        trailer_post_cells.push_back(30 + trial);
    }
    return run_ctx_pred_controlled_trainer_impl(
        seed,
        schedule_variant,
        leader_pre_cells,
        trailer_post_cells
    );
}

CtxPredTinyTrainerTestResult run_ctx_pred_generated_schedule_test(
    std::int64_t seed,
    const std::vector<std::int32_t>& leader_pre_cells,
    const std::vector<std::int32_t>& trailer_post_cells
) {
    const std::size_t n_trials = leader_pre_cells.size();
    if (n_trials < 6 || n_trials > 360) {
        throw std::runtime_error(
            "generated Stage1 schedule test expects 6 to 360 trials"
        );
    }
    return run_ctx_pred_controlled_trainer_impl(
        seed,
        -1,
        leader_pre_cells,
        trailer_post_cells
    );
}

ManifestSummary inspect_manifest_path(const std::string& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("manifest path is not a regular file: " + path);
    }

    // Placeholder until the NPZ reader and device-side manifest loader land.
    // The Python exporter/validator is the source of truth for schema v1.
    return ManifestSummary{
        .schema_version = 1,
        .synapse_bank_count = 0,
    };
}

}  // namespace expectation_snn_cuda
