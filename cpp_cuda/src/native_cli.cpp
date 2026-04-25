#include "expectation_snn_cuda/manifest.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-8;
constexpr int kNOrientations = 6;
constexpr int kV1Channels = 12;
constexpr int kV1CellsPerChannel = 16;
constexpr int kV1SomCellsPerChannel = 4;
constexpr int kTrailerBinCount = 5;
constexpr int kHCellsPerChannel = 16;
constexpr double kPBias = 0.80;
constexpr double kTrailerWindowSeconds = 0.5;
constexpr double kFeedbackGTotalDefault = 1.0;
constexpr double kFeedbackRDefault = 1.0;
constexpr double kFeedbackReplayTargetPerBinDefault = 314.1666666666667;
constexpr double kFeedbackDirectBaseDriveAmpPA = 30.0;
constexpr double kFeedbackSomBaseDriveAmpPA = 40.0;
constexpr double kFeedbackSomD1Weight = 0.4;
constexpr double kFeedbackSomD2Weight = 0.1;
constexpr double kFeedbackSomCenterWeightDefault = 0.0;
constexpr double kV1SomToEScaleDefault = 1.0;
constexpr double kV1SomDivisiveScaleDefault = 0.0;
constexpr double kV1DirectDivisiveScaleDefault = 0.0;
constexpr double kV1FeedforwardDivisiveScaleDefault = 0.0;
constexpr double kV1PredictedSuppressionScaleDefault = 0.0;
constexpr double kV1PredictedSuppressionNeighborWeightDefault = 0.0;
constexpr int kV1PredictedSuppressionLocusEffectiveIe = 0;
constexpr int kV1PredictedSuppressionLocusRawIe = 1;
constexpr int kV1ErrorComparatorModeOff = 0;
constexpr int kV1ErrorComparatorModeFixedSymmetric = 1;
constexpr int kV1ErrorComparatorModeSignedNormalized = 2;
constexpr double kPi = 3.14159265358979323846;
constexpr double kV1StimSigmaDegDefault = 22.0;
constexpr double kLearnedHpredV1DirectRowSum = 16.0;
constexpr double kLearnedHpredV1SomRowSum = 4.4;
constexpr int kTrailerSteps = 5000;
constexpr std::array<int, kNOrientations> kDerangement{{1, 2, 3, 4, 5, 0}};

struct FeedbackBalance {
    double g_total = kFeedbackGTotalDefault;
    double r = kFeedbackRDefault;
    double g_direct = 0.5;
    double g_som = 0.5;
};

struct Args {
    std::string command;
    std::string execution_mode = "gpu_only_production";
    std::int64_t seed = 42;
    std::int32_t n_steps = 128;
    std::int32_t repeats = 20;
    std::int32_t schedule_variant = 0;
    std::int32_t n_trials = 72;
    std::string stage1_prediction_target = "orientation_cell";
    bool stage1_learn_feedback_direct = false;
    bool stage1_learn_feedback_som = false;
    bool threshold_case = false;
    std::string population = "h_e";
    std::string fixture = "generated";
    std::string heldout_schedule = "generated";
    std::filesystem::path out_path;
    std::filesystem::path checkpoint_path;
    std::int32_t reps_expected = 14;
    std::int32_t reps_unexpected = 12;
    double grating_rate_hz = 500.0;
    double baseline_rate_hz = 0.0;
    double v1_stim_sigma_deg = kV1StimSigmaDegDefault;
    double feedback_g_total = kFeedbackGTotalDefault;
    double feedback_r = kFeedbackRDefault;
    double feedback_som_center_weight = kFeedbackSomCenterWeightDefault;
    std::string feedback_direct_source = "fixed";
    std::string feedback_som_source = "fixed";
    double v1_som_to_e_scale = kV1SomToEScaleDefault;
    double v1_som_divisive_scale = kV1SomDivisiveScaleDefault;
    double v1_direct_divisive_scale = kV1DirectDivisiveScaleDefault;
    std::string v1_direct_divisive_gate_source = "som";
    double v1_feedforward_divisive_scale =
        kV1FeedforwardDivisiveScaleDefault;
    std::string v1_feedforward_divisive_gate_source = "som";
    double v1_predicted_suppression_scale =
        kV1PredictedSuppressionScaleDefault;
    double v1_predicted_suppression_neighbor_weight =
        kV1PredictedSuppressionNeighborWeightDefault;
    std::string v1_predicted_suppression_locus = "effective_ie";
    std::string v1_error_comparator_mode = "off";
    double v1_error_sensory_gain = 1.0;
    double v1_error_prediction_gain = 1.0;
    std::int32_t v1_error_prediction_shift = 0;
    std::string feedback_replay_mode = "raw";
    double feedback_replay_target_per_bin =
        kFeedbackReplayTargetPerBinDefault;
    std::string feedback_replay_fallback = "none";
};

struct GeneratedSchedule {
    std::int64_t seed = 42;
    std::int32_t n_trials = 0;
    std::vector<std::int32_t> leader_idx;
    std::vector<std::int32_t> trailer_idx;
    std::vector<std::int32_t> expected_trailer_idx;
    std::vector<std::int32_t> is_expected;
    std::vector<std::int32_t> leader_cells;
    std::vector<std::int32_t> trailer_cells;
};

struct Stage1CheckpointMeta {
    std::filesystem::path json_path;
    std::filesystem::path w_path;
    std::string content_hash;
    std::string w_hash;
    std::int64_t seed = 0;
    std::int32_t n_trials = 0;
    bool has_learned_hpred_v1direct = false;
    std::filesystem::path learned_hpred_v1direct_path;
    std::string learned_hpred_v1direct_hash;
    bool has_learned_hpred_v1som = false;
    std::filesystem::path learned_hpred_v1som_path;
    std::string learned_hpred_v1som_hash;
};

void print_usage(std::ostream& os) {
    os << "usage: expectation_snn_native <command> [options]\n"
       << "\n"
       << "commands:\n"
       << "  device-info                 print CUDA runtime/driver/device metadata\n"
       << "  validate-fixture | self-test run native C++/CUDA primitive checks\n"
       << "  bench                       run a native end-to-end primitive timing loop\n"
       << "  stage1-train                run native Stage1 training fixture/generated path\n"
       << "  stage1-heldout-eval         run held-out native Stage1 H-pred evaluation\n"
       << "  richter-dampening           run native frozen Richter dampening\n"
       << "  sensory-diagnostics         run native sensory-stage tuning diagnostics\n"
       << "\n"
       << "stage1-train options:\n"
       << "  --n-trials N                generated path requires a multiple of 6, default 72\n"
       << "  --seed N                    default 42\n"
       << "  --out PATH                  JSON/result artifact path\n"
       << "  --checkpoint PATH           native Stage1 JSON checkpoint for richter-dampening\n"
       << "  --fixture generated|tiny    default generated\n"
       << "  --stage1-prediction-target orientation_cell|v1_template  default orientation_cell\n"
       << "  --stage1-learn-feedback-direct learn checkpointed H_pred->V1_E direct/apical prediction route\n"
       << "  --stage1-learn-feedback-som learn checkpointed H_pred->V1_SOM prediction route\n"
       << "\n"
       << "stage1-heldout-eval options:\n"
       << "  --checkpoint PATH           native Stage1 JSON checkpoint to evaluate\n"
       << "  --seed N                    held-out schedule seed; must differ from checkpoint seed\n"
       << "  --out PATH                  JSON/result artifact path\n"
       << "  --heldout-schedule MODE     generated|all-unexpected, default generated\n"
       << "  --reps-expected N           richter expected-condition repetitions, default 14\n"
       << "  --reps-unexpected N         richter repetitions per unexpected step, default 12\n"
       << "  --execution-mode MODE       richter mode: gpu_only_production|cpu_reference\n"
       << "  --v1-stim-sigma-deg X       V1 stimulus Gaussian width in degrees, default 22.0\n"
       << "  --feedback-g-total X        richter feedback total weight scale, default 1.0\n"
       << "  --feedback-r X              richter feedback direct/SOM balance r, default 1.0\n"
       << "  --feedback-som-center-weight X  same-channel H_pred -> V1_SOM kernel weight, default 0.0\n"
       << "  --feedback-direct-source fixed|learned|learned-shifted|disabled  H_pred->V1_E direct/apical route source, default fixed\n"
       << "  --feedback-som-source fixed|learned|learned-shifted|disabled  H_pred->V1_SOM route source, default fixed\n"
       << "  --v1-som-to-e-scale X       V1_SOM -> V1_E inhibitory drive scale, default 1.0\n"
       << "  --v1-som-divisive-scale X   same-channel V1_SOM divisive E/apical gain scale, default 0.0\n"
       << "  --v1-direct-divisive-scale X  same-channel V1_SOM divisive apical/direct gain scale, default 0.0\n"
       << "  --v1-direct-divisive-gate-source som|feedback  direct/apical divisive denominator source, default som\n"
       << "  --v1-feedforward-divisive-scale X  same-channel V1_SOM divisive feedforward/somatic gain scale, default 0.0\n"
       << "  --v1-feedforward-divisive-gate-source som|feedback  feedforward divisive denominator source, default som\n"
       << "  --v1-predicted-suppression-scale X  prediction-matched V1_E feedforward suppression scale, default 0.0\n"
       << "  --v1-predicted-suppression-neighbor-weight X  ±1 channel prediction suppression neighbor weight, default 0.0\n"
       << "  --v1-predicted-suppression-locus effective_ie|raw_ie  suppression locus, default effective_ie\n"
       << "  --v1-error-comparator-mode off|fixed_symmetric|signed_normalized  separate V1_ERROR residual population, default off\n"
       << "  --v1-error-sensory-gain X  V1_ERROR bottom-up sensory gain, default 1.0\n"
       << "  --v1-error-prediction-gain X  V1_ERROR held-H_pred prediction inhibitory gain, default 1.0\n"
       << "  --v1-error-prediction-shift N  rotate H_pred prediction channel before V1_ERROR comparator, default 0\n"
       << "  --feedback-replay-mode MODE trailer feedback source: raw|normalized, default raw\n"
       << "  --feedback-replay-fallback MODE normalized zero-preprobe fallback: none|leader|flat, default none\n"
       << "  --feedback-replay-target-per-100ms-bin X  normalized replay target, default 314.1666666666667\n"
       << "\n"
       << "other options:\n"
       << "  --steps N                   bench primitive steps, default 128\n"
       << "  --repeats N                 bench repeats, default 20\n"
       << "  --population h_e|v1_e       bench population, default h_e\n"
       << "  --threshold-case            use threshold/reset primitive state\n"
       << "  --schedule-variant 0|1      tiny Stage1 schedule variant\n";
}

std::int32_t parse_i32(const std::string& name, const std::string& value) {
    try {
        std::size_t pos = 0;
        const long parsed = std::stol(value, &pos);
        if (pos != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::int32_t>(parsed);
    } catch (const std::exception& exc) {
        throw std::runtime_error(
            "invalid " + name + " value '" + value + "': " + exc.what()
        );
    }
}

std::int64_t parse_i64(const std::string& name, const std::string& value) {
    try {
        std::size_t pos = 0;
        const long long parsed = std::stoll(value, &pos);
        if (pos != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::int64_t>(parsed);
    } catch (const std::exception& exc) {
        throw std::runtime_error(
            "invalid " + name + " value '" + value + "': " + exc.what()
        );
    }
}

Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error("missing command");
    }
    Args args;
    args.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + option);
            }
            return argv[++i];
        };
        if (key == "--seed") {
            args.seed = parse_i64(key, require_value(key));
        } else if (key == "--steps") {
            args.n_steps = parse_i32(key, require_value(key));
        } else if (key == "--repeats") {
            args.repeats = parse_i32(key, require_value(key));
        } else if (key == "--population") {
            args.population = require_value(key);
        } else if (key == "--threshold-case") {
            args.threshold_case = true;
        } else if (key == "--fixture") {
            args.fixture = require_value(key);
        } else if (key == "--heldout-schedule") {
            args.heldout_schedule = require_value(key);
        } else if (key == "--schedule-variant") {
            args.schedule_variant = parse_i32(key, require_value(key));
        } else if (key == "--n-trials") {
            args.n_trials = parse_i32(key, require_value(key));
        } else if (key == "--stage1-prediction-target") {
            args.stage1_prediction_target = require_value(key);
        } else if (key == "--stage1-learn-feedback-direct") {
            args.stage1_learn_feedback_direct = true;
        } else if (key == "--stage1-learn-feedback-som") {
            args.stage1_learn_feedback_som = true;
        } else if (key == "--out") {
            args.out_path = require_value(key);
        } else if (key == "--checkpoint") {
            args.checkpoint_path = require_value(key);
        } else if (key == "--reps-expected") {
            args.reps_expected = parse_i32(key, require_value(key));
        } else if (key == "--reps-unexpected") {
            args.reps_unexpected = parse_i32(key, require_value(key));
        } else if (key == "--grating-rate-hz") {
            args.grating_rate_hz = std::stod(require_value(key));
        } else if (key == "--baseline-rate-hz") {
            args.baseline_rate_hz = std::stod(require_value(key));
        } else if (key == "--v1-stim-sigma-deg") {
            args.v1_stim_sigma_deg = std::stod(require_value(key));
        } else if (key == "--feedback-g-total") {
            args.feedback_g_total = std::stod(require_value(key));
        } else if (key == "--feedback-r") {
            args.feedback_r = std::stod(require_value(key));
        } else if (key == "--feedback-som-center-weight") {
            args.feedback_som_center_weight = std::stod(require_value(key));
        } else if (key == "--feedback-direct-source") {
            args.feedback_direct_source = require_value(key);
        } else if (key == "--feedback-som-source") {
            args.feedback_som_source = require_value(key);
        } else if (key == "--v1-som-to-e-scale") {
            args.v1_som_to_e_scale = std::stod(require_value(key));
        } else if (key == "--v1-som-divisive-scale") {
            args.v1_som_divisive_scale = std::stod(require_value(key));
        } else if (key == "--v1-direct-divisive-scale") {
            args.v1_direct_divisive_scale = std::stod(require_value(key));
        } else if (key == "--v1-direct-divisive-gate-source") {
            args.v1_direct_divisive_gate_source = require_value(key);
        } else if (key == "--v1-feedforward-divisive-scale") {
            args.v1_feedforward_divisive_scale = std::stod(require_value(key));
        } else if (key == "--v1-feedforward-divisive-gate-source") {
            args.v1_feedforward_divisive_gate_source = require_value(key);
        } else if (key == "--v1-predicted-suppression-scale") {
            args.v1_predicted_suppression_scale = std::stod(require_value(key));
        } else if (key == "--v1-predicted-suppression-neighbor-weight") {
            args.v1_predicted_suppression_neighbor_weight =
                std::stod(require_value(key));
        } else if (key == "--v1-predicted-suppression-locus") {
            args.v1_predicted_suppression_locus = require_value(key);
        } else if (key == "--v1-error-comparator-mode") {
            args.v1_error_comparator_mode = require_value(key);
        } else if (key == "--v1-error-sensory-gain") {
            args.v1_error_sensory_gain = std::stod(require_value(key));
        } else if (key == "--v1-error-prediction-gain") {
            args.v1_error_prediction_gain = std::stod(require_value(key));
        } else if (key == "--v1-error-prediction-shift") {
            args.v1_error_prediction_shift = parse_i32(key, require_value(key));
        } else if (key == "--feedback-replay-mode") {
            args.feedback_replay_mode = require_value(key);
        } else if (key == "--feedback-replay-fallback") {
            args.feedback_replay_fallback = require_value(key);
        } else if (key == "--feedback-replay-target-per-100ms-bin") {
            args.feedback_replay_target_per_bin = std::stod(require_value(key));
        } else if (key == "--execution-mode") {
            args.execution_mode = require_value(key);
        } else if (key == "--help" || key == "-h") {
            args.command = "help";
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    return args;
}

std::int32_t v1_feedforward_divisive_gate_source_id(
    const std::string& source
) {
    if (source == "som") {
        return 0;
    }
    if (source == "feedback") {
        return 1;
    }
    throw std::runtime_error(
        "richter-dampening --v1-feedforward-divisive-gate-source must be som or feedback"
    );
}

const char* v1_feedforward_divisive_gate_description(
    const std::string& source
) {
    if (source == "feedback") {
        return "same_channel_current_step_feedback_hpred_spike_count_during_trailer_else_zero";
    }
    return "same_channel_current_step_v1som_spike_count";
}

const char* v1_feedforward_divisive_denominator_description(
    const std::string& source
) {
    if (source == "feedback") {
        return "1 + scale * same_channel_current_step_feedback_hpred_spike_count_during_trailer";
    }
    return "1 + scale * same_channel_current_step_v1som_spike_count";
}

std::int32_t v1_direct_divisive_gate_source_id(
    const std::string& source
) {
    if (source == "som") {
        return 0;
    }
    if (source == "feedback") {
        return 1;
    }
    throw std::runtime_error(
        "richter-dampening --v1-direct-divisive-gate-source must be som or feedback"
    );
}

const char* v1_direct_divisive_gate_description(
    const std::string& source
) {
    if (source == "feedback") {
        return "same_channel_current_step_feedback_hpred_spike_count_during_trailer_else_zero";
    }
    return "same_channel_current_step_v1som_spike_count";
}

const char* v1_direct_divisive_denominator_description(
    const std::string& source
) {
    if (source == "feedback") {
        return "1 + scale * same_channel_current_step_feedback_hpred_spike_count_during_trailer";
    }
    return "1 + scale * same_channel_current_step_v1som_spike_count";
}

std::int32_t v1_predicted_suppression_locus_id(const std::string& locus) {
    if (locus == "effective_ie") {
        return kV1PredictedSuppressionLocusEffectiveIe;
    }
    if (locus == "raw_ie") {
        return kV1PredictedSuppressionLocusRawIe;
    }
    throw std::runtime_error(
        "richter-dampening --v1-predicted-suppression-locus must be effective_ie or raw_ie"
    );
}

const char* v1_predicted_suppression_target_description(
    const std::string& locus
) {
    if (locus == "raw_ie") {
        return "V1_E_raw_feedforward_I_e_pA_before_Q_active_and_membrane_update";
    }
    return "V1_E_feedforward_somatic_effective_ie_only";
}

std::int32_t v1_error_comparator_mode_id(const std::string& mode) {
    if (mode == "off") {
        return kV1ErrorComparatorModeOff;
    }
    if (mode == "fixed_symmetric") {
        return kV1ErrorComparatorModeFixedSymmetric;
    }
    if (mode == "signed_normalized") {
        return kV1ErrorComparatorModeSignedNormalized;
    }
    throw std::runtime_error(
        "richter-dampening --v1-error-comparator-mode must be off, fixed_symmetric, or signed_normalized"
    );
}

std::uint64_t fnv1a_update(
    std::uint64_t hash,
    const void* data,
    std::size_t n_bytes
) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n_bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

template <typename T>
std::uint64_t fnv1a_update_scalar(std::uint64_t hash, const T& value) {
    return fnv1a_update(hash, &value, sizeof(T));
}

template <typename T>
std::uint64_t fnv1a_update_vector(
    std::uint64_t hash,
    const std::vector<T>& values
) {
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    hash = fnv1a_update_scalar(hash, size);
    if (!values.empty()) {
        hash = fnv1a_update(hash, values.data(), values.size() * sizeof(T));
    }
    return hash;
}

std::uint64_t fnv1a_update_string(std::uint64_t hash, const std::string& value) {
    hash = fnv1a_update_scalar(hash, static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) {
        hash = fnv1a_update(hash, value.data(), value.size());
    }
    return hash;
}

std::string hash_hex(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (const char ch : text) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

template <typename T>
void write_vector_json(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

template <typename T>
void write_matrix_json(
    std::ostream& out,
    const std::vector<T>& values,
    std::size_t rows,
    std::size_t cols
) {
    if (values.size() != rows * cols) {
        throw std::runtime_error("matrix JSON shape mismatch");
    }
    out << '[';
    for (std::size_t row = 0; row < rows; ++row) {
        if (row != 0) {
            out << ',';
        }
        out << '[';
        for (std::size_t col = 0; col < cols; ++col) {
            if (col != 0) {
                out << ',';
            }
            out << values[row * cols + col];
        }
        out << ']';
    }
    out << ']';
}

template <typename T>
void write_tensor3_json(
    std::ostream& out,
    const std::vector<T>& values,
    std::size_t dim0,
    std::size_t dim1,
    std::size_t dim2
) {
    if (values.size() != dim0 * dim1 * dim2) {
        throw std::runtime_error("tensor3 JSON shape mismatch");
    }
    out << '[';
    for (std::size_t i = 0; i < dim0; ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '[';
        for (std::size_t j = 0; j < dim1; ++j) {
            if (j != 0) {
                out << ',';
            }
            out << '[';
            for (std::size_t k = 0; k < dim2; ++k) {
                if (k != 0) {
                    out << ',';
                }
                out << values[(i * dim1 + j) * dim2 + k];
            }
            out << ']';
        }
        out << ']';
    }
    out << ']';
}

void write_pairs_json(
    std::ostream& out,
    const std::vector<std::int32_t>& leader,
    const std::vector<std::int32_t>& trailer
) {
    out << '[';
    for (std::size_t i = 0; i < leader.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '[' << leader[i] << ',' << trailer[i] << ']';
    }
    out << ']';
}

template <typename T, std::size_t N>
void write_fixed_matrix_json(
    std::ostream& out,
    const std::vector<std::array<T, N>>& rows
) {
    out << '[';
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (row != 0) {
            out << ',';
        }
        out << '[';
        for (std::size_t col = 0; col < N; ++col) {
            if (col != 0) {
                out << ',';
            }
            out << rows[row][col];
        }
        out << ']';
    }
    out << ']';
}

template <typename T, std::size_t OuterN, std::size_t InnerN>
void write_fixed_cube_json(
    std::ostream& out,
    const std::vector<std::array<std::array<T, InnerN>, OuterN>>& rows
) {
    out << '[';
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (row != 0) {
            out << ',';
        }
        out << '[';
        for (std::size_t outer = 0; outer < OuterN; ++outer) {
            if (outer != 0) {
                out << ',';
            }
            out << '[';
            for (std::size_t inner = 0; inner < InnerN; ++inner) {
                if (inner != 0) {
                    out << ',';
                }
                out << rows[row][outer][inner];
            }
            out << ']';
        }
        out << ']';
    }
    out << ']';
}

std::array<std::int32_t, kV1Channels> v1_trailer_channel_counts(
    const std::map<std::string, std::vector<std::int32_t>>& raw_counts
) {
    const auto it = raw_counts.find("v1_e.trailer");
    if (it == raw_counts.end()) {
        throw std::runtime_error("missing count key: v1_e.trailer");
    }
    if (it->second.size() != static_cast<std::size_t>(kV1Channels * kV1CellsPerChannel)) {
        throw std::runtime_error("v1_e.trailer count vector size mismatch");
    }
    std::array<std::int32_t, kV1Channels> out{};
    for (std::size_t idx = 0; idx < it->second.size(); ++idx) {
        out[idx / kV1CellsPerChannel] += it->second[idx];
    }
    return out;
}

std::array<std::int32_t, kV1Channels> v1_som_trailer_channel_counts(
    const std::map<std::string, std::vector<std::int32_t>>& raw_counts
) {
    const auto it = raw_counts.find("v1_som.trailer");
    if (it == raw_counts.end()) {
        throw std::runtime_error("missing count key: v1_som.trailer");
    }
    if (it->second.size()
        != static_cast<std::size_t>(kV1Channels * kV1SomCellsPerChannel)) {
        throw std::runtime_error("v1_som.trailer count vector size mismatch");
    }
    std::array<std::int32_t, kV1Channels> out{};
    for (std::size_t idx = 0; idx < it->second.size(); ++idx) {
        out[idx / kV1SomCellsPerChannel] += it->second[idx];
    }
    return out;
}

std::array<std::int32_t, kV1Channels> v1_error_trailer_channel_counts(
    const std::map<std::string, std::vector<std::int32_t>>& raw_counts
) {
    const auto it = raw_counts.find("v1_error.trailer");
    if (it == raw_counts.end()) {
        return {};
    }
    if (it->second.size() != static_cast<std::size_t>(kV1Channels * kV1CellsPerChannel)) {
        throw std::runtime_error("v1_error.trailer count vector size mismatch");
    }
    std::array<std::int32_t, kV1Channels> out{};
    for (std::size_t idx = 0; idx < it->second.size(); ++idx) {
        out[idx / kV1CellsPerChannel] += it->second[idx];
    }
    return out;
}

std::array<std::int32_t, kV1Channels> v1_error_neg_trailer_channel_counts(
    const std::map<std::string, std::vector<std::int32_t>>& raw_counts
) {
    const auto it = raw_counts.find("v1_error_neg.trailer");
    if (it == raw_counts.end()) {
        return {};
    }
    if (it->second.size() != static_cast<std::size_t>(kV1Channels * kV1CellsPerChannel)) {
        throw std::runtime_error("v1_error_neg.trailer count vector size mismatch");
    }
    std::array<std::int32_t, kV1Channels> out{};
    for (std::size_t idx = 0; idx < it->second.size(); ++idx) {
        out[idx / kV1CellsPerChannel] += it->second[idx];
    }
    return out;
}

std::array<double, kV1Channels> v1_trailer_channel_rates_hz(
    const std::array<std::int32_t, kV1Channels>& counts
) {
    std::array<double, kV1Channels> out{};
    const double denom =
        static_cast<double>(kV1CellsPerChannel) * kTrailerWindowSeconds;
    for (std::size_t channel = 0; channel < counts.size(); ++channel) {
        out[channel] = static_cast<double>(counts[channel]) / denom;
    }
    return out;
}

std::array<double, kV1Channels> trailer_channel_rates_hz(
    const std::array<std::int32_t, kV1Channels>& counts,
    int cells_per_channel
) {
    std::array<double, kV1Channels> out{};
    const double denom =
        static_cast<double>(cells_per_channel) * kTrailerWindowSeconds;
    for (std::size_t channel = 0; channel < counts.size(); ++channel) {
        out[channel] = static_cast<double>(counts[channel]) / denom;
    }
    return out;
}

void require_close_map(
    const std::string& label,
    const std::map<std::string, double>& errors,
    double tol
) {
    for (const auto& [name, value] : errors) {
        if (!std::isfinite(value) || value > tol) {
            throw std::runtime_error(
                label + " mismatch " + name + "=" + std::to_string(value)
            );
        }
    }
}

void require_equal_counts(
    const std::string& label,
    const std::vector<std::int32_t>& a,
    const std::vector<std::int32_t>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(label + " count size mismatch");
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            throw std::runtime_error(
                label + " count mismatch at " + std::to_string(i)
                + ": " + std::to_string(a[i]) + " != " + std::to_string(b[i])
            );
        }
    }
}

GeneratedSchedule build_generated_schedule(std::int64_t seed, std::int32_t n_trials) {
    if (n_trials <= 0 || (n_trials % kNOrientations) != 0) {
        throw std::runtime_error("generated Stage1 n_trials must be a positive multiple of 6");
    }
    GeneratedSchedule schedule;
    schedule.seed = seed;
    schedule.n_trials = n_trials;
    schedule.leader_idx.reserve(static_cast<std::size_t>(n_trials));
    schedule.trailer_idx.reserve(static_cast<std::size_t>(n_trials));
    schedule.expected_trailer_idx.reserve(static_cast<std::size_t>(n_trials));
    schedule.is_expected.reserve(static_cast<std::size_t>(n_trials));
    schedule.leader_cells.reserve(static_cast<std::size_t>(n_trials));
    schedule.trailer_cells.reserve(static_cast<std::size_t>(n_trials));

    std::vector<int> leaders;
    leaders.reserve(static_cast<std::size_t>(n_trials));
    const int reps_per_leader = n_trials / kNOrientations;
    for (int leader = 0; leader < kNOrientations; ++leader) {
        for (int rep = 0; rep < reps_per_leader; ++rep) {
            leaders.push_back(leader);
        }
    }
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::shuffle(leaders.begin(), leaders.end(), rng);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    for (int trial = 0; trial < n_trials; ++trial) {
        const int leader = leaders[static_cast<std::size_t>(trial)];
        const int expected = kDerangement[static_cast<std::size_t>(leader)];
        int trailer = expected;
        if (uniform01(rng) >= kPBias) {
            std::array<int, kNOrientations - 2> other{};
            int idx = 0;
            for (int candidate = 0; candidate < kNOrientations; ++candidate) {
                if (candidate != leader && candidate != expected) {
                    other[static_cast<std::size_t>(idx++)] = candidate;
                }
            }
            std::uniform_int_distribution<int> pick(0, kNOrientations - 3);
            trailer = other[static_cast<std::size_t>(pick(rng))];
        }
        const int cell_offset = trial % kHCellsPerChannel;
        const int leader_channel = leader * 2;
        const int trailer_channel = trailer * 2;
        schedule.leader_idx.push_back(leader);
        schedule.trailer_idx.push_back(trailer);
        schedule.expected_trailer_idx.push_back(expected);
        schedule.is_expected.push_back(trailer == expected ? 1 : 0);
        schedule.leader_cells.push_back(leader_channel * kHCellsPerChannel + cell_offset);
        schedule.trailer_cells.push_back(trailer_channel * kHCellsPerChannel + cell_offset);
    }
    return schedule;
}

GeneratedSchedule build_all_unexpected_schedule(std::int64_t seed, std::int32_t n_trials) {
    GeneratedSchedule schedule = build_generated_schedule(seed, n_trials);
    for (int trial = 0; trial < n_trials; ++trial) {
        const int leader = schedule.leader_idx[static_cast<std::size_t>(trial)];
        const int expected = schedule.expected_trailer_idx[static_cast<std::size_t>(trial)];
        int trailer = (expected + 1) % kNOrientations;
        while (trailer == leader || trailer == expected) {
            trailer = (trailer + 1) % kNOrientations;
        }
        const int cell_offset = trial % kHCellsPerChannel;
        const int trailer_channel = trailer * 2;
        schedule.trailer_idx[static_cast<std::size_t>(trial)] = trailer;
        schedule.is_expected[static_cast<std::size_t>(trial)] = 0;
        schedule.trailer_cells[static_cast<std::size_t>(trial)] =
            trailer_channel * kHCellsPerChannel + cell_offset;
    }
    return schedule;
}

GeneratedSchedule build_heldout_schedule(
    std::int64_t seed,
    std::int32_t n_trials,
    const std::string& mode
) {
    if (mode == "generated") {
        return build_generated_schedule(seed, n_trials);
    }
    if (mode == "all-unexpected") {
        return build_all_unexpected_schedule(seed, n_trials);
    }
    throw std::runtime_error("unknown heldout schedule mode: " + mode);
}

std::array<double, kV1Channels> v1_template_channel_weights(int center_channel) {
    if (center_channel < 0 || center_channel >= kV1Channels) {
        throw std::runtime_error("V1 template center channel out of range");
    }
    std::array<double, kV1Channels> weights{};
    double total = 0.0;
    for (int channel = 0; channel < kV1Channels; ++channel) {
        const int raw_delta = std::abs(channel - center_channel);
        const int wrapped_delta = std::min(raw_delta, kV1Channels - raw_delta);
        const double d_rad =
            static_cast<double>(wrapped_delta) * kPi / static_cast<double>(kV1Channels);
        const double sigma_rad = kV1StimSigmaDegDefault * kPi / 180.0;
        const double weight = std::exp(-0.5 * (d_rad / sigma_rad) * (d_rad / sigma_rad));
        weights[static_cast<std::size_t>(channel)] = weight;
        total += weight;
    }
    if (total <= 0.0) {
        throw std::runtime_error("V1 template total weight is zero");
    }
    for (double& value : weights) {
        value /= total;
    }
    return weights;
}

std::vector<double> build_learned_hpred_v1som_weights(
    const GeneratedSchedule& schedule,
    const std::string& prediction_target
) {
    if (prediction_target != "orientation_cell" && prediction_target != "v1_template") {
        throw std::runtime_error("learned feedback requires a known Stage1 prediction target");
    }
    std::vector<double> weights(static_cast<std::size_t>(192 * 48), 0.0);
    for (std::size_t trial = 0; trial < schedule.expected_trailer_idx.size(); ++trial) {
        const int center_channel =
            schedule.expected_trailer_idx[trial] * 2;
        const auto src_template = prediction_target == "v1_template"
            ? v1_template_channel_weights(center_channel)
            : [&]() {
                std::array<double, kV1Channels> one_hot{};
                one_hot[static_cast<std::size_t>(center_channel)] = 1.0;
                return one_hot;
            }();
        const auto som_template = v1_template_channel_weights(center_channel);
        for (int src_channel = 0; src_channel < kV1Channels; ++src_channel) {
            for (int src_cell = 0; src_cell < kHCellsPerChannel; ++src_cell) {
                const int pre = src_channel * kHCellsPerChannel + src_cell;
                for (int som_channel = 0; som_channel < kV1Channels; ++som_channel) {
                    const double channel_weight =
                        src_template[static_cast<std::size_t>(src_channel)]
                        * som_template[static_cast<std::size_t>(som_channel)];
                    for (int som_cell = 0; som_cell < kV1SomCellsPerChannel; ++som_cell) {
                        const int post = som_channel * kV1SomCellsPerChannel + som_cell;
                        weights[static_cast<std::size_t>(pre * 48 + post)] +=
                            channel_weight / static_cast<double>(kV1SomCellsPerChannel);
                    }
                }
            }
        }
    }
    for (int pre = 0; pre < 192; ++pre) {
        double row_sum = 0.0;
        for (int post = 0; post < 48; ++post) {
            row_sum += weights[static_cast<std::size_t>(pre * 48 + post)];
        }
        if (row_sum > 0.0) {
            const double scale = kLearnedHpredV1SomRowSum / row_sum;
            for (int post = 0; post < 48; ++post) {
                weights[static_cast<std::size_t>(pre * 48 + post)] *= scale;
            }
        }
    }
    return weights;
}

std::vector<double> build_learned_hpred_v1direct_weights(
    const GeneratedSchedule& schedule,
    const std::string& prediction_target
) {
    if (prediction_target != "orientation_cell" && prediction_target != "v1_template") {
        throw std::runtime_error("learned feedback requires a known Stage1 prediction target");
    }
    std::vector<double> weights(static_cast<std::size_t>(192 * 192), 0.0);
    for (std::size_t trial = 0; trial < schedule.expected_trailer_idx.size(); ++trial) {
        const int center_channel =
            schedule.expected_trailer_idx[trial] * 2;
        const auto src_template = prediction_target == "v1_template"
            ? v1_template_channel_weights(center_channel)
            : [&]() {
                std::array<double, kV1Channels> one_hot{};
                one_hot[static_cast<std::size_t>(center_channel)] = 1.0;
                return one_hot;
            }();
        const auto v1_template = v1_template_channel_weights(center_channel);
        for (int src_channel = 0; src_channel < kV1Channels; ++src_channel) {
            for (int src_cell = 0; src_cell < kHCellsPerChannel; ++src_cell) {
                const int pre = src_channel * kHCellsPerChannel + src_cell;
                for (int v1_channel = 0; v1_channel < kV1Channels; ++v1_channel) {
                    const double channel_weight =
                        src_template[static_cast<std::size_t>(src_channel)]
                        * v1_template[static_cast<std::size_t>(v1_channel)];
                    for (int v1_cell = 0; v1_cell < kV1CellsPerChannel; ++v1_cell) {
                        const int post = v1_channel * kV1CellsPerChannel + v1_cell;
                        weights[static_cast<std::size_t>(pre * 192 + post)] +=
                            channel_weight / static_cast<double>(kV1CellsPerChannel);
                    }
                }
            }
        }
    }
    for (int pre = 0; pre < 192; ++pre) {
        double row_sum = 0.0;
        for (int post = 0; post < 192; ++post) {
            row_sum += weights[static_cast<std::size_t>(pre * 192 + post)];
        }
        if (row_sum > 0.0) {
            const double scale = kLearnedHpredV1DirectRowSum / row_sum;
            for (int post = 0; post < 192; ++post) {
                weights[static_cast<std::size_t>(pre * 192 + post)] *= scale;
            }
        }
    }
    return weights;
}

std::vector<double> hpred_v1direct_row_sums(const std::vector<double>& weights) {
    if (weights.size() != static_cast<std::size_t>(192 * 192)) {
        throw std::runtime_error("W_hpred_v1direct must have length 36864");
    }
    std::vector<double> out(192, 0.0);
    for (int pre = 0; pre < 192; ++pre) {
        for (int post = 0; post < 192; ++post) {
            out[static_cast<std::size_t>(pre)] +=
                weights[static_cast<std::size_t>(pre * 192 + post)];
        }
    }
    return out;
}

std::vector<double> hpred_v1som_row_sums(const std::vector<double>& weights) {
    if (weights.size() != static_cast<std::size_t>(192 * 48)) {
        throw std::runtime_error("W_hpred_v1som must have length 9216");
    }
    std::vector<double> out(192, 0.0);
    for (int pre = 0; pre < 192; ++pre) {
        for (int post = 0; post < 48; ++post) {
            out[static_cast<std::size_t>(pre)] +=
                weights[static_cast<std::size_t>(pre * 48 + post)];
        }
    }
    return out;
}

std::map<std::string, double> stage1_template_prediction_metrics(
    const GeneratedSchedule& schedule,
    const expectation_snn_cuda::Stage1HGateDynamicsResult& gates
) {
    if (schedule.n_trials <= 0
        || gates.cuda_pred_pretrailer_channel_counts.size()
            != static_cast<std::size_t>(schedule.n_trials) * kV1Channels) {
        throw std::runtime_error("Stage1 template metric shape mismatch");
    }
    double cosine_sum = 0.0;
    double mse_sum = 0.0;
    double target_weighted_count_sum = 0.0;
    double expected_channel_count_sum = 0.0;
    double total_count_sum = 0.0;
    std::int32_t argmax_matches = 0;
    for (int trial = 0; trial < schedule.n_trials; ++trial) {
        const int expected_channel =
            schedule.expected_trailer_idx[static_cast<std::size_t>(trial)] * 2;
        const auto target = v1_template_channel_weights(expected_channel);
        double count_total = 0.0;
        double dot = 0.0;
        double count_norm2 = 0.0;
        double target_norm2 = 0.0;
        int argmax_channel = 0;
        double argmax_count = -1.0;
        for (int channel = 0; channel < kV1Channels; ++channel) {
            const double count = static_cast<double>(
                gates.cuda_pred_pretrailer_channel_counts[
                    static_cast<std::size_t>(trial) * kV1Channels
                    + static_cast<std::size_t>(channel)
                ]
            );
            count_total += count;
            if (count > argmax_count) {
                argmax_count = count;
                argmax_channel = channel;
            }
        }
        for (int channel = 0; channel < kV1Channels; ++channel) {
            const double count = static_cast<double>(
                gates.cuda_pred_pretrailer_channel_counts[
                    static_cast<std::size_t>(trial) * kV1Channels
                    + static_cast<std::size_t>(channel)
                ]
            );
            const double prob = count_total > 0.0 ? count / count_total : 0.0;
            const double target_prob = target[static_cast<std::size_t>(channel)];
            dot += prob * target_prob;
            count_norm2 += prob * prob;
            target_norm2 += target_prob * target_prob;
            const double delta = prob - target_prob;
            mse_sum += delta * delta / static_cast<double>(kV1Channels);
            target_weighted_count_sum += count * target_prob;
        }
        expected_channel_count_sum += static_cast<double>(
            gates.cuda_pred_pretrailer_channel_counts[
                static_cast<std::size_t>(trial) * kV1Channels
                + static_cast<std::size_t>(expected_channel)
            ]
        );
        total_count_sum += count_total;
        if (argmax_channel == expected_channel) {
            ++argmax_matches;
        }
        const double denom = std::sqrt(count_norm2) * std::sqrt(target_norm2);
        cosine_sum += denom > 0.0 ? dot / denom : 0.0;
    }
    return {
        {"v1_template_cosine_mean", cosine_sum / schedule.n_trials},
        {"v1_template_probability_mse_mean", mse_sum / schedule.n_trials},
        {"v1_template_target_weighted_count_mean", target_weighted_count_sum / schedule.n_trials},
        {"v1_template_expected_channel_count_mean", expected_channel_count_sum / schedule.n_trials},
        {"v1_template_total_count_mean", total_count_sum / schedule.n_trials},
        {"v1_template_argmax_matches_expected_channel_fraction",
            static_cast<double>(argmax_matches) / schedule.n_trials},
    };
}

std::uint64_t schedule_hash(const GeneratedSchedule& schedule) {
    std::uint64_t hash = 1469598103934665603ull;
    hash = fnv1a_update_scalar(hash, schedule.seed);
    hash = fnv1a_update_scalar(hash, schedule.n_trials);
    hash = fnv1a_update_vector(hash, schedule.leader_idx);
    hash = fnv1a_update_vector(hash, schedule.trailer_idx);
    hash = fnv1a_update_vector(hash, schedule.expected_trailer_idx);
    hash = fnv1a_update_vector(hash, schedule.is_expected);
    hash = fnv1a_update_vector(hash, schedule.leader_cells);
    hash = fnv1a_update_vector(hash, schedule.trailer_cells);
    return hash;
}

std::uint64_t artifact_hash(
    const GeneratedSchedule& schedule,
    const expectation_snn_cuda::NativeStage1TrainResult& train
) {
    std::uint64_t hash = schedule_hash(schedule);
    const double p_bias = kPBias;
    const double drive_pa = 400.0;
    const double pred_bias_pa = 100.0;
    const double w_init_frac = 0.0;
    hash = fnv1a_update_scalar(hash, p_bias);
    hash = fnv1a_update_scalar(hash, drive_pa);
    hash = fnv1a_update_scalar(hash, pred_bias_pa);
    hash = fnv1a_update_scalar(hash, w_init_frac);
    hash = fnv1a_update_string(hash, train.prediction_target);
    hash = fnv1a_update_vector(hash, train.w_ctx_pred_final);
    hash = fnv1a_update_vector(hash, train.row_sums);
    hash = fnv1a_update_vector(hash, train.gate_dw_sum);
    hash = fnv1a_update_vector(hash, train.gate_elig_max);
    hash = fnv1a_update_vector(hash, train.w_hpred_v1direct_final);
    hash = fnv1a_update_vector(hash, train.w_hpred_v1som_final);
    return hash;
}

void write_binary_vector(const std::filesystem::path& path, const std::vector<double>& values) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open binary artifact: " + path.string());
    }
    out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double))
    );
    if (!out) {
        throw std::runtime_error("failed to write binary artifact: " + path.string());
    }
}

void write_stage1_json(
    const std::filesystem::path& json_path,
    const std::filesystem::path& w_path,
    const std::filesystem::path& hpred_v1direct_path,
    const std::filesystem::path& hpred_v1som_path,
    const GeneratedSchedule& schedule,
    const expectation_snn_cuda::NativeStage1TrainResult& train,
    const expectation_snn_cuda::Stage1HGateDynamicsResult& gates,
    double schedule_wall_s,
    double train_wall_s,
    double gate_wall_s,
    double write_wall_s,
    const std::string& content_hash
) {
    if (!json_path.parent_path().empty()) {
        std::filesystem::create_directories(json_path.parent_path());
    }
    std::ofstream out(json_path);
    if (!out) {
        throw std::runtime_error("failed to open JSON artifact: " + json_path.string());
    }
    const auto& metrics = gates.metrics;
    const double persistence = metrics.at("h_context_persistence_ms");
    const double forecast = metrics.at("h_prediction_pretrailer_forecast_probability");
    const double no_runaway = metrics.at("no_runaway_max_rate_hz");
    const double max_cell = metrics.at("no_runaway_max_cell_rate_hz");
    const auto template_metrics = stage1_template_prediction_metrics(schedule, gates);
    const bool thresholds_pass = persistence >= 200.0 && persistence <= 500.0
        && forecast >= 0.25 && no_runaway <= 80.0;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"artifact_kind\": \"native_stage1_ctx_pred_binary_checkpoint\",\n";
    out << "  \"backend\": \"native_cxx_cuda_executable\",\n";
    out << "  \"passed\": " << (thresholds_pass ? "true" : "false") << ",\n";
    out << "  \"native_scientific_stage1_passed\": " << (thresholds_pass ? "true" : "false") << ",\n";
    out << "  \"provisional\": false,\n";
    out << "  \"placeholder_h_recurrent_arrays\": false,\n";
    out << "  \"seed\": " << schedule.seed << ",\n";
    out << "  \"n_trials\": " << schedule.n_trials << ",\n";
    out << "  \"p_bias\": " << kPBias << ",\n";
    out << "  \"derangement\": [1,2,3,4,5,0],\n";
    out << "  \"stage1_prediction_task\": {"
        << "\"objective\": \"predict_future_lower_level_v1e_trailer_template_from_context_leader\", "
        << "\"target\": \"" << json_escape(train.prediction_target) << "\", "
        << "\"target_uses_expected_leader_derangement_only\": true, "
        << "\"uses_q_activity_decoder_metrics\": false, "
        << "\"uses_richter_expected_unexpected_eval_metrics\": false, "
        << "\"uses_actual_unexpected_trailer_labels\": false, "
        << "\"v1_template_sigma_deg\": " << kV1StimSigmaDegDefault
        << "},\n";
    out << "  \"h_layout\": {\"channels\": 12, \"cells_per_channel\": 16, \"orientation_channel_stride\": 2},\n";
    out << "  \"timing_ms\": {\"leader\": 500.0, \"trailer\": 500.0, \"iti\": 1500.0, \"dt\": 0.1},\n";
    out << "  \"ctx_pred_constants\": {\"drive_amp_ctx_pred_pA\": 400.0, \"pred_e_uniform_bias_pA\": 100.0, \"w_init_frac\": 0.0, \"tau_coinc_ms\": 500.0, \"tau_elig_ms\": 1000.0, \"eta\": 0.001, \"gamma\": 0.0001, \"w_target\": 0.0075, \"w_row_max\": 3.0},\n";
    out << "  \"phase_steps\": {";
    bool first = true;
    for (const auto& [key, value] : train.phase_steps) {
        if (!first) out << ", ";
        first = false;
        out << "\"" << json_escape(key) << "\": " << value;
    }
    out << "},\n";
    out << "  \"event_counts\": {";
    first = true;
    for (const auto& [key, value] : train.event_counts) {
        if (!first) out << ", ";
        first = false;
        out << "\"" << json_escape(key) << "\": " << value;
    }
    out << "},\n";
    out << "  \"schedule\": {\n";
    out << "    \"pairs\": "; write_pairs_json(out, schedule.leader_idx, schedule.trailer_idx); out << ",\n";
    out << "    \"expected_trailer_idx\": "; write_vector_json(out, schedule.expected_trailer_idx); out << ",\n";
    out << "    \"is_expected\": "; write_vector_json(out, schedule.is_expected); out << ",\n";
    out << "    \"leader_cells\": "; write_vector_json(out, schedule.leader_cells); out << ",\n";
    out << "    \"trailer_cells\": "; write_vector_json(out, schedule.trailer_cells); out << "\n";
    out << "  },\n";
    const std::string w_hash = hash_hex(fnv1a_update_vector(1469598103934665603ull, train.w_ctx_pred_final));
    out << "  \"checkpoint\": {\"W_ctx_pred_path\": \"" << json_escape(w_path.string()) << "\", \"dtype\": \"float64\", \"shape\": [36864], \"W_ctx_pred_fnv1a64\": \"" << w_hash << "\"},\n";
    out << "  \"learned_feedback\": {";
    if (!train.w_hpred_v1direct_final.empty()) {
        const std::string direct_hash = hash_hex(
            fnv1a_update_vector(1469598103934665603ull, train.w_hpred_v1direct_final)
        );
        const double row_sum_max = train.w_hpred_v1direct_row_sums.empty()
            ? 0.0
            : *std::max_element(
                train.w_hpred_v1direct_row_sums.begin(),
                train.w_hpred_v1direct_row_sums.end()
            );
        out << "\"hpred_to_v1direct\": {"
            << "\"enabled\": true, "
            << "\"objective\": \"predict_future_sensory_driven_v1e_template_from_hpred_template\", "
            << "\"uses_q_activity_decoder_metrics\": false, "
            << "\"uses_richter_expected_unexpected_eval_metrics\": false, "
            << "\"uses_actual_unexpected_trailer_labels\": false, "
            << "\"W_hpred_v1direct_path\": \"" << json_escape(hpred_v1direct_path.string())
            << "\", \"dtype\": \"float64\", \"shape\": [36864], "
            << "\"W_hpred_v1direct_fnv1a64\": \"" << direct_hash << "\", "
            << "\"row_sum_target\": " << kLearnedHpredV1DirectRowSum
            << ", \"row_sum_max\": " << row_sum_max
            << "}";
    } else {
        out << "\"hpred_to_v1direct\": {\"enabled\": false}";
    }
    out << ", ";
    if (!train.w_hpred_v1som_final.empty()) {
        const std::string som_hash = hash_hex(
            fnv1a_update_vector(1469598103934665603ull, train.w_hpred_v1som_final)
        );
        const double row_sum_max = train.w_hpred_v1som_row_sums.empty()
            ? 0.0
            : *std::max_element(
                train.w_hpred_v1som_row_sums.begin(),
                train.w_hpred_v1som_row_sums.end()
            );
        out << "\"hpred_to_v1som\": {"
            << "\"enabled\": true, "
            << "\"objective\": \"predict_future_sensory_driven_v1som_template_from_hpred_template\", "
            << "\"uses_q_activity_decoder_metrics\": false, "
            << "\"uses_richter_expected_unexpected_eval_metrics\": false, "
            << "\"uses_actual_unexpected_trailer_labels\": false, "
            << "\"W_hpred_v1som_path\": \"" << json_escape(hpred_v1som_path.string())
            << "\", \"dtype\": \"float64\", \"shape\": [9216], "
            << "\"W_hpred_v1som_fnv1a64\": \"" << som_hash << "\", "
            << "\"row_sum_target\": " << kLearnedHpredV1SomRowSum
            << ", \"row_sum_max\": " << row_sum_max
            << "}";
    } else {
        out << "\"hpred_to_v1som\": {\"enabled\": false}";
    }
    out << "},\n";
    out << "  \"metrics\": {\n";
    out << "    \"h_context_persistence_ms\": " << persistence << ",\n";
    out << "    \"h_context_persistence_pass\": " << (persistence >= 200.0 && persistence <= 500.0 ? "true" : "false") << ",\n";
    out << "    \"forecast_probability\": " << forecast << ",\n";
    out << "    \"forecast_pass\": " << (forecast >= 0.25 ? "true" : "false") << ",\n";
    out << "    \"no_runaway_population_rate_hz\": " << no_runaway << ",\n";
    out << "    \"no_runaway_pass\": " << (no_runaway <= 80.0 ? "true" : "false") << ",\n";
    out << "    \"max_cell_rate_hz_diagnostic\": " << max_cell << ",\n";
    out << "    \"gate_cpu_cuda_max_abs_error\": ";
    double max_gate_error = 0.0;
    for (const auto& [_, value] : gates.max_abs_error) max_gate_error = std::max(max_gate_error, value);
    out << max_gate_error << ",\n";
    out << "    \"thresholds_pass\": " << (thresholds_pass ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"target_prediction_metrics\": {";
    bool metrics_first = true;
    for (const auto& [key, value] : template_metrics) {
        if (!metrics_first) out << ", ";
        metrics_first = false;
        out << "\"" << json_escape(key) << "\": " << value;
    }
    out << "},\n";
    out << "  \"weight_stats\": {\"delta_abs_sum\": ";
    double delta_abs_sum = 0.0;
    for (const double value : train.w_ctx_pred_final) delta_abs_sum += std::abs(value);
    const double row_sum_max = train.row_sums.empty() ? 0.0 : *std::max_element(train.row_sums.begin(), train.row_sums.end());
    out << delta_abs_sum << ", \"row_sum_max\": " << row_sum_max << "},\n";
    out << "  \"timing_seconds\": {\"schedule\": " << schedule_wall_s << ", \"train\": " << train_wall_s << ", \"gate_eval\": " << gate_wall_s << ", \"write\": " << write_wall_s << ", \"total_excluding_build\": " << (schedule_wall_s + train_wall_s + gate_wall_s + write_wall_s) << "},\n";
    out << "  \"content_hash_fnv1a64\": \"" << content_hash << "\"\n";
    out << "}\n";
}

std::uint64_t stage1_heldout_eval_hash(
    const Stage1CheckpointMeta& checkpoint,
    const GeneratedSchedule& schedule,
    const expectation_snn_cuda::Stage1HGateDynamicsResult& gates
) {
    std::uint64_t hash = 1469598103934665603ull;
    hash = fnv1a_update(hash, checkpoint.content_hash.data(), checkpoint.content_hash.size());
    hash = fnv1a_update(hash, checkpoint.w_hash.data(), checkpoint.w_hash.size());
    const std::uint64_t schedule_value = schedule_hash(schedule);
    hash = fnv1a_update_scalar(hash, schedule_value);
    hash = fnv1a_update_vector(hash, gates.cuda_pred_pretrailer_target_counts);
    hash = fnv1a_update_vector(hash, gates.cuda_pred_pretrailer_channel_counts);
    hash = fnv1a_update_vector(hash, gates.cuda_pred_pretrailer_channel_counts_by_bin);
    hash = fnv1a_update_vector(hash, gates.cuda_ctx_total_counts);
    hash = fnv1a_update_vector(hash, gates.cuda_pred_total_counts);
    return hash;
}

void write_stage1_heldout_eval_json(
    const std::filesystem::path& json_path,
    const Stage1CheckpointMeta& checkpoint,
    const GeneratedSchedule& schedule,
    const expectation_snn_cuda::Stage1HGateDynamicsResult& gates,
    double eval_wall_s,
    const std::string& content_hash
) {
    if (!json_path.parent_path().empty()) {
        std::filesystem::create_directories(json_path.parent_path());
    }
    std::ofstream out(json_path);
    if (!out) {
        throw std::runtime_error("failed to open held-out eval artifact: " + json_path.string());
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"artifact_kind\": \"native_stage1_ctx_pred_heldout_eval\",\n";
    out << "  \"backend\": \"native_cxx_cuda_executable\",\n";
    out << "  \"checkpoint\": {\n";
    out << "    \"json_path\": \"" << json_escape(checkpoint.json_path.string()) << "\",\n";
    out << "    \"w_path\": \"" << json_escape(checkpoint.w_path.string()) << "\",\n";
    out << "    \"content_hash_fnv1a64\": \"" << checkpoint.content_hash << "\",\n";
    out << "    \"W_ctx_pred_fnv1a64\": \"" << checkpoint.w_hash << "\",\n";
    out << "    \"seed\": " << checkpoint.seed << ",\n";
    out << "    \"n_trials\": " << checkpoint.n_trials << "\n";
    out << "  },\n";
    out << "  \"heldout_schedule_seed\": " << schedule.seed << ",\n";
    out << "  \"heldout_schedule_hash_fnv1a64\": \"" << hash_hex(schedule_hash(schedule)) << "\",\n";
    out << "  \"n_trials\": " << schedule.n_trials << ",\n";
    out << "  \"phase_steps\": {";
    bool first = true;
    for (const auto& [key, value] : gates.phase_steps) {
        if (!first) out << ", ";
        first = false;
        out << "\"" << json_escape(key) << "\": " << value;
    }
    out << "},\n";
    out << "  \"schedule\": {\n";
    out << "    \"pairs\": "; write_pairs_json(out, schedule.leader_idx, schedule.trailer_idx); out << ",\n";
    out << "    \"expected_trailer_idx\": "; write_vector_json(out, schedule.expected_trailer_idx); out << ",\n";
    out << "    \"is_expected\": "; write_vector_json(out, schedule.is_expected); out << ",\n";
    out << "    \"leader_cells\": "; write_vector_json(out, schedule.leader_cells); out << ",\n";
    out << "    \"trailer_cells\": "; write_vector_json(out, schedule.trailer_cells); out << "\n";
    out << "  },\n";
    out << "  \"metrics\": {";
    first = true;
    for (const auto& [key, value] : gates.metrics) {
        if (!first) out << ", ";
        first = false;
        out << "\"" << json_escape(key) << "\": " << value;
    }
    out << "},\n";
    const auto template_metrics = stage1_template_prediction_metrics(schedule, gates);
    out << "  \"target_prediction_metrics\": {";
    bool metrics_first = true;
    for (const auto& [key, value] : template_metrics) {
        if (!metrics_first) out << ", ";
        metrics_first = false;
        out << "\"" << json_escape(key) << "\": " << value;
    }
    out << "},\n";
    double max_gate_error = 0.0;
    for (const auto& [_, value] : gates.max_abs_error) {
        max_gate_error = std::max(max_gate_error, value);
    }
    out << "  \"max_cpu_cuda_error\": " << max_gate_error << ",\n";
    out << "  \"timing_seconds\": {\"eval\": " << eval_wall_s << "},\n";
    out << "  \"per_trial\": {\n";
    out << "    \"leader_channels\": "; write_vector_json(out, gates.leader_channels); out << ",\n";
    out << "    \"trailer_channels\": "; write_vector_json(out, gates.trailer_channels); out << ",\n";
    out << "    \"ctx_persistence_ms_by_trial\": "; write_vector_json(out, gates.cuda_ctx_persistence_ms_by_trial); out << ",\n";
    out << "    \"pred_pretrailer_target_counts\": "; write_vector_json(out, gates.cuda_pred_pretrailer_target_counts); out << ",\n";
    out << "    \"pred_pretrailer_channel_counts\": ";
    write_matrix_json(
        out,
        gates.cuda_pred_pretrailer_channel_counts,
        static_cast<std::size_t>(schedule.n_trials),
        static_cast<std::size_t>(kV1Channels)
    );
    out << ",\n";
    out << "    \"pred_pretrailer_channel_counts_by_bin\": ";
    write_tensor3_json(
        out,
        gates.cuda_pred_pretrailer_channel_counts_by_bin,
        static_cast<std::size_t>(schedule.n_trials),
        static_cast<std::size_t>(kTrailerBinCount),
        static_cast<std::size_t>(kV1Channels)
    );
    out << "\n";
    out << "  },\n";
    out << "  \"content_hash_fnv1a64\": \"" << content_hash << "\"\n";
    out << "}\n";
}

struct NativeTopology {
    std::vector<std::int32_t> stim_pre;
    std::vector<std::int32_t> stim_post;
    std::vector<double> stim_w;
    std::vector<std::int32_t> stim_channel;
    std::vector<std::int32_t> v1_to_h_pre;
    std::vector<std::int32_t> v1_to_h_post;
    std::vector<double> v1_to_h_w;
    std::vector<std::int32_t> ctx_to_pred_pre;
    std::vector<std::int32_t> ctx_to_pred_post;
    std::vector<double> ctx_to_pred_w;
    std::vector<std::int32_t> fb_direct_pre;
    std::vector<std::int32_t> fb_direct_post;
    std::vector<double> fb_direct_w;
    std::vector<std::int32_t> fb_som_pre;
    std::vector<std::int32_t> fb_som_post;
    std::vector<double> fb_som_w;
};

struct ConditionAccum {
    double v1_leader = 0.0;
    double v1_preprobe = 0.0;
    double v1_trailer = 0.0;
    double v1_som_trailer = 0.0;
    double v1_error_trailer = 0.0;
    double v1_error_neg_trailer = 0.0;
    double v1e_leader_q_active_fC = 0.0;
    double v1e_preprobe_q_active_fC = 0.0;
    double v1e_trailer_q_active_fC = 0.0;
    double v1som_leader_q_active_fC = 0.0;
    double v1som_preprobe_q_active_fC = 0.0;
    double v1som_trailer_q_active_fC = 0.0;
    double v1error_trailer_q_active_fC = 0.0;
    double v1error_neg_trailer_q_active_fC = 0.0;
    double hctx_preprobe = 0.0;
    double hctx_trailer = 0.0;
    double hpred_preprobe = 0.0;
    double hpred_trailer = 0.0;
    std::int64_t source_total = 0;
    double max_error = 0.0;
    std::array<std::int32_t, kV1Channels> v1_trailer_channel_counts{};
    std::array<std::int32_t, kV1Channels> v1_som_trailer_channel_counts{};
    std::array<std::int32_t, kV1Channels> v1_error_trailer_channel_counts{};
    std::array<std::int32_t, kV1Channels> v1_error_neg_trailer_channel_counts{};
    std::array<std::int32_t, kV1Channels> hpred_preprobe_channel_counts{};
    std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>
        v1_trailer_bin_channel_counts{};
    std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>
        v1_som_trailer_bin_channel_counts{};
    std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>
        v1_error_trailer_bin_channel_counts{};
    std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>
        v1_error_neg_trailer_bin_channel_counts{};
    std::array<std::int32_t, kTrailerBinCount> hpred_trailer_bin_total_counts{};
    std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>
        hpred_trailer_bin_channel_counts{};
    std::array<std::int32_t, kTrailerBinCount>
        hpred_feedback_held_trailer_bin_total_counts{};
    std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>
        hpred_feedback_held_trailer_bin_channel_counts{};
    std::array<double, kTrailerBinCount>
        hpred_feedback_normalized_trailer_bin_total_weights{};
    std::array<std::array<double, kV1Channels>, kTrailerBinCount>
        hpred_feedback_normalized_trailer_bin_channel_weights{};
    std::array<double, kV1Channels>
        v1_predicted_suppression_trailer_channel_signal_sum{};
    std::array<double, kV1Channels>
        v1_predicted_suppression_trailer_channel_gain_sum{};
    std::array<double, kV1Channels>
        v1_predicted_suppression_trailer_raw_ie_before_sum{};
    std::array<double, kV1Channels>
        v1_predicted_suppression_trailer_raw_ie_after_sum{};
    std::array<double, kV1Channels>
        v1_predicted_suppression_trailer_raw_ie_delta_sum{};
    std::int32_t hpred_feedback_normalized_preprobe_zero = 0;
    std::int32_t hpred_feedback_normalized_fallback_used = 0;
    std::int32_t hpred_feedback_normalized_fallback_zero_template = 0;
};

struct RichterConditionSpec {
    std::int64_t seed = 0;
    int rep = 0;
    int is_expected = 0;
    int leader_orientation = 0;
    int trailer_orientation = 0;
    int step = 0;
    int leader_channel = 0;
    int trailer_channel = 0;
};

FeedbackBalance resolve_feedback_balance(
    double g_total = kFeedbackGTotalDefault,
    double r = kFeedbackRDefault
) {
    if (g_total < 0.0) {
        throw std::runtime_error("feedback g_total must be nonnegative");
    }
    if (std::isinf(r)) {
        return FeedbackBalance{
            g_total,
            r,
            g_total,
            0.0,
        };
    }
    if (r < 0.0) {
        throw std::runtime_error("feedback r must be >= 0 or inf");
    }
    const double g_som = g_total / (1.0 + r);
    const double g_direct = g_total - g_som;
    return FeedbackBalance{
        g_total,
        r,
        g_direct,
        g_som,
    };
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open text file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string extract_json_string(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON string key: " + key);
    }
    const std::size_t colon = text.find(':', key_pos + needle.size());
    const std::size_t first_quote = text.find('"', colon + 1);
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (colon == std::string::npos || first_quote == std::string::npos
        || second_quote == std::string::npos) {
        throw std::runtime_error("malformed JSON string key: " + key);
    }
    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

bool extract_json_bool(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON bool key: " + key);
    }
    const std::size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed JSON bool key: " + key);
    }
    const std::size_t value = text.find_first_not_of(" \t\r\n", colon + 1);
    if (text.compare(value, 4, "true") == 0) {
        return true;
    }
    if (text.compare(value, 5, "false") == 0) {
        return false;
    }
    throw std::runtime_error("malformed JSON bool value for key: " + key);
}

bool json_has_key(const std::string& text, const std::string& key) {
    return text.find("\"" + key + "\"") != std::string::npos;
}

std::int64_t extract_json_int(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing JSON int key: " + key);
    }
    const std::size_t colon = text.find(':', key_pos + needle.size());
    const std::size_t value = text.find_first_of("-0123456789", colon + 1);
    std::size_t end = value;
    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-')) {
        ++end;
    }
    return std::stoll(text.substr(value, end - value));
}

bool json_contains_shape_36864(const std::string& text) {
    return text.find("\"shape\": [36864]") != std::string::npos
        || text.find("\"shape\":[36864]") != std::string::npos;
}

std::vector<double> read_f64_binary(
    const std::filesystem::path& path,
    std::size_t expected_len,
    const std::string& label
) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + label + " binary: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff n_bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (n_bytes != static_cast<std::streamoff>(expected_len * sizeof(double))) {
        throw std::runtime_error(
            label + " binary byte size does not match expected float64 length"
        );
    }
    std::vector<double> values(expected_len);
    in.read(reinterpret_cast<char*>(values.data()), n_bytes);
    if (!in) {
        throw std::runtime_error("failed to read " + label + " binary: " + path.string());
    }
    return values;
}

std::vector<double> read_w_binary(const std::filesystem::path& path) {
    return read_f64_binary(path, 36864, "W_ctx_pred");
}

Stage1CheckpointMeta validate_stage1_checkpoint(
    const std::filesystem::path& json_path,
    std::vector<double>& w_out,
    std::vector<double>* learned_hpred_v1direct_out = nullptr,
    std::vector<double>* learned_hpred_v1som_out = nullptr
) {
    const std::string text = read_text_file(json_path);
    Stage1CheckpointMeta meta;
    meta.json_path = json_path;
    if (extract_json_string(text, "artifact_kind") != "native_stage1_ctx_pred_binary_checkpoint") {
        throw std::runtime_error("checkpoint artifact_kind is not native_stage1_ctx_pred_binary_checkpoint");
    }
    if (!extract_json_bool(text, "passed")
        || !extract_json_bool(text, "native_scientific_stage1_passed")) {
        throw std::runtime_error("checkpoint did not pass native scientific Stage1 gates");
    }
    if (extract_json_bool(text, "provisional")) {
        throw std::runtime_error("checkpoint is provisional");
    }
    if (extract_json_bool(text, "placeholder_h_recurrent_arrays")) {
        throw std::runtime_error("checkpoint has placeholder H recurrent arrays");
    }
    if (extract_json_string(text, "dtype") != "float64" || !json_contains_shape_36864(text)) {
        throw std::runtime_error("checkpoint W dtype/shape metadata mismatch");
    }
    meta.seed = extract_json_int(text, "seed");
    meta.n_trials = static_cast<std::int32_t>(extract_json_int(text, "n_trials"));
    meta.content_hash = extract_json_string(text, "content_hash_fnv1a64");
    meta.w_path = extract_json_string(text, "W_ctx_pred_path");
    if (meta.w_path.is_relative()) {
        meta.w_path = json_path.parent_path() / meta.w_path;
    }
    meta.w_hash = extract_json_string(text, "W_ctx_pred_fnv1a64");
    w_out = read_w_binary(meta.w_path);
    const std::string actual_w_hash = hash_hex(fnv1a_update_vector(1469598103934665603ull, w_out));
    if (actual_w_hash != meta.w_hash) {
        throw std::runtime_error("checkpoint W hash mismatch: " + actual_w_hash + " != " + meta.w_hash);
    }
    if (json_has_key(text, "W_hpred_v1direct_path")) {
        meta.has_learned_hpred_v1direct = true;
        meta.learned_hpred_v1direct_path = extract_json_string(text, "W_hpred_v1direct_path");
        if (meta.learned_hpred_v1direct_path.is_relative()) {
            meta.learned_hpred_v1direct_path =
                json_path.parent_path() / meta.learned_hpred_v1direct_path;
        }
        meta.learned_hpred_v1direct_hash =
            extract_json_string(text, "W_hpred_v1direct_fnv1a64");
        std::vector<double> learned = read_f64_binary(
            meta.learned_hpred_v1direct_path,
            static_cast<std::size_t>(192 * 192),
            "W_hpred_v1direct"
        );
        const std::string actual_learned_hash =
            hash_hex(fnv1a_update_vector(1469598103934665603ull, learned));
        if (actual_learned_hash != meta.learned_hpred_v1direct_hash) {
            throw std::runtime_error(
                "checkpoint W_hpred_v1direct hash mismatch: "
                + actual_learned_hash + " != " + meta.learned_hpred_v1direct_hash
            );
        }
        if (learned_hpred_v1direct_out != nullptr) {
            *learned_hpred_v1direct_out = std::move(learned);
        }
    } else if (learned_hpred_v1direct_out != nullptr) {
        learned_hpred_v1direct_out->clear();
    }
    if (json_has_key(text, "W_hpred_v1som_path")) {
        meta.has_learned_hpred_v1som = true;
        meta.learned_hpred_v1som_path = extract_json_string(text, "W_hpred_v1som_path");
        if (meta.learned_hpred_v1som_path.is_relative()) {
            meta.learned_hpred_v1som_path =
                json_path.parent_path() / meta.learned_hpred_v1som_path;
        }
        meta.learned_hpred_v1som_hash =
            extract_json_string(text, "W_hpred_v1som_fnv1a64");
        std::vector<double> learned = read_f64_binary(
            meta.learned_hpred_v1som_path,
            static_cast<std::size_t>(192 * 48),
            "W_hpred_v1som"
        );
        const std::string actual_learned_hash =
            hash_hex(fnv1a_update_vector(1469598103934665603ull, learned));
        if (actual_learned_hash != meta.learned_hpred_v1som_hash) {
            throw std::runtime_error(
                "checkpoint W_hpred_v1som hash mismatch: "
                + actual_learned_hash + " != " + meta.learned_hpred_v1som_hash
            );
        }
        if (learned_hpred_v1som_out != nullptr) {
            *learned_hpred_v1som_out = std::move(learned);
        }
    } else if (learned_hpred_v1som_out != nullptr) {
        learned_hpred_v1som_out->clear();
    }
    return meta;
}

NativeTopology make_native_topology(
    const std::vector<double>& w_ctx_pred,
    const FeedbackBalance& feedback,
    double feedback_som_center_weight = kFeedbackSomCenterWeightDefault,
    const std::vector<double>& learned_hpred_v1direct =
        std::vector<double>(),
    const std::string& feedback_direct_source = "fixed",
    const std::vector<double>& learned_hpred_v1som =
        std::vector<double>(),
    const std::string& feedback_som_source = "fixed"
) {
    if (w_ctx_pred.size() != 36864) {
        throw std::runtime_error("ctx_pred W must have length 36864");
    }
    if (feedback_som_center_weight < 0.0) {
        throw std::runtime_error("feedback SOM center weight must be nonnegative");
    }
    const bool use_fixed_direct = feedback_direct_source == "fixed";
    const bool use_learned_direct = feedback_direct_source == "learned";
    const bool use_shifted_direct = feedback_direct_source == "learned-shifted";
    const bool disable_direct = feedback_direct_source == "disabled";
    if (!use_fixed_direct && !use_learned_direct && !use_shifted_direct
        && !disable_direct) {
        throw std::runtime_error(
            "feedback direct source must be fixed, learned, learned-shifted, or disabled"
        );
    }
    const bool use_fixed_som = feedback_som_source == "fixed";
    const bool use_learned_som = feedback_som_source == "learned";
    const bool use_shifted_som = feedback_som_source == "learned-shifted";
    const bool disable_som = feedback_som_source == "disabled";
    if (!use_fixed_som && !use_learned_som && !use_shifted_som && !disable_som) {
        throw std::runtime_error(
            "feedback SOM source must be fixed, learned, learned-shifted, or disabled"
        );
    }
    if ((use_learned_som || use_shifted_som)
        && learned_hpred_v1som.size() != static_cast<std::size_t>(192 * 48)) {
        throw std::runtime_error(
            "learned feedback SOM source requires checkpointed W_hpred_v1som length 9216"
        );
    }
    if ((use_learned_direct || use_shifted_direct)
        && learned_hpred_v1direct.size() != static_cast<std::size_t>(192 * 192)) {
        throw std::runtime_error(
            "learned feedback direct source requires checkpointed W_hpred_v1direct length 36864"
        );
    }
    NativeTopology t;
    t.stim_pre.reserve(3840);
    t.stim_post.reserve(3840);
    t.stim_w.reserve(3840);
    t.stim_channel.reserve(240);
    for (int src = 0; src < 240; ++src) {
        const int channel = src / 20;
        t.stim_channel.push_back(channel);
        for (int k = 0; k < 16; ++k) {
            t.stim_pre.push_back(src);
            t.stim_post.push_back(channel * 16 + k);
            t.stim_w.push_back(1.0);
        }
    }
    t.v1_to_h_pre.reserve(21504);
    t.v1_to_h_post.reserve(21504);
    t.v1_to_h_w.reserve(21504);
    for (int pre = 0; pre < 192; ++pre) {
        const int channel = pre / 16;
        for (int dc = -3; dc <= 3; ++dc) {
            const int target_channel = (channel + dc + 12) % 12;
            for (int k = 0; k < 16; ++k) {
                t.v1_to_h_pre.push_back(pre);
                t.v1_to_h_post.push_back(target_channel * 16 + k);
                t.v1_to_h_w.push_back(1.0);
            }
        }
    }
    t.ctx_to_pred_pre.reserve(36864);
    t.ctx_to_pred_post.reserve(36864);
    t.ctx_to_pred_w.reserve(36864);
    for (int pre = 0; pre < 192; ++pre) {
        for (int post = 0; post < 192; ++post) {
            t.ctx_to_pred_pre.push_back(pre);
            t.ctx_to_pred_post.push_back(post);
            t.ctx_to_pred_w.push_back(w_ctx_pred[static_cast<std::size_t>(pre * 192 + post)]);
        }
    }
    if (use_fixed_direct) {
        t.fb_direct_pre.reserve(3072);
        t.fb_direct_post.reserve(3072);
        t.fb_direct_w.reserve(3072);
    } else if (use_learned_direct || use_shifted_direct) {
        t.fb_direct_pre.reserve(static_cast<std::size_t>(192 * 192));
        t.fb_direct_post.reserve(static_cast<std::size_t>(192 * 192));
        t.fb_direct_w.reserve(static_cast<std::size_t>(192 * 192));
    } else {
        t.fb_direct_pre.reserve(1);
        t.fb_direct_post.reserve(1);
        t.fb_direct_w.reserve(1);
    }
    const int fb_som_center_edges =
        use_fixed_som && feedback_som_center_weight > 0.0 ? 768 : 0;
    if (use_fixed_som) {
        t.fb_som_pre.reserve(3072 + fb_som_center_edges);
        t.fb_som_post.reserve(3072 + fb_som_center_edges);
        t.fb_som_w.reserve(3072 + fb_som_center_edges);
    } else if (use_learned_som || use_shifted_som) {
        t.fb_som_pre.reserve(static_cast<std::size_t>(192 * 48));
        t.fb_som_post.reserve(static_cast<std::size_t>(192 * 48));
        t.fb_som_w.reserve(static_cast<std::size_t>(192 * 48));
    } else {
        t.fb_som_pre.reserve(1);
        t.fb_som_post.reserve(1);
        t.fb_som_w.reserve(1);
    }
    for (int pre = 0; pre < 192; ++pre) {
        const int channel = pre / 16;
        if (use_learned_direct || use_shifted_direct) {
            const int source_pre = use_shifted_direct
                ? (((channel + 2) % kV1Channels) * 16 + (pre % 16))
                : pre;
            for (int post = 0; post < 192; ++post) {
                t.fb_direct_pre.push_back(pre);
                t.fb_direct_post.push_back(post);
                t.fb_direct_w.push_back(
                    learned_hpred_v1direct[
                        static_cast<std::size_t>(source_pre * 192 + post)
                    ] * feedback.g_direct
                );
            }
        } else if (use_fixed_direct) {
            for (int k = 0; k < 16; ++k) {
                t.fb_direct_pre.push_back(pre);
                t.fb_direct_post.push_back(channel * 16 + k);
                t.fb_direct_w.push_back(feedback.g_direct);
            }
        }
        if (use_learned_som || use_shifted_som) {
            const int source_pre = use_shifted_som
                ? (((channel + 2) % kV1Channels) * 16 + (pre % 16))
                : pre;
            for (int post = 0; post < 48; ++post) {
                t.fb_som_pre.push_back(pre);
                t.fb_som_post.push_back(post);
                t.fb_som_w.push_back(
                    learned_hpred_v1som[
                        static_cast<std::size_t>(source_pre * 48 + post)
                    ] * feedback.g_som
                );
            }
            continue;
        }
        if (disable_som) {
            continue;
        }
        if (feedback_som_center_weight > 0.0) {
            for (int som_cell = 0; som_cell < 4; ++som_cell) {
                t.fb_som_pre.push_back(pre);
                t.fb_som_post.push_back(channel * 4 + som_cell);
                t.fb_som_w.push_back(feedback_som_center_weight * feedback.g_som);
            }
        }
        for (const auto& [delta_channel, kernel_weight] :
             std::array<std::pair<int, double>, 4>{{
                 {-1, kFeedbackSomD1Weight},
                 {1, kFeedbackSomD1Weight},
                 {-2, kFeedbackSomD2Weight},
                 {2, kFeedbackSomD2Weight},
             }}) {
            const int som_channel =
                (channel + delta_channel + kV1Channels) % kV1Channels;
            for (int som_cell = 0; som_cell < 4; ++som_cell) {
                t.fb_som_pre.push_back(pre);
                t.fb_som_post.push_back(som_channel * 4 + som_cell);
                t.fb_som_w.push_back(kernel_weight * feedback.g_som);
            }
        }
    }
    if (disable_direct) {
        t.fb_direct_pre.push_back(191);
        t.fb_direct_post.push_back(191);
        t.fb_direct_w.push_back(0.0);
    }
    if (disable_som) {
        t.fb_som_pre.push_back(191);
        t.fb_som_post.push_back(47);
        t.fb_som_w.push_back(0.0);
    }
    return t;
}

double sum_counts(const std::map<std::string, std::vector<std::int32_t>>& counts, const std::string& key) {
    const auto it = counts.find(key);
    if (it == counts.end()) {
        throw std::runtime_error("missing count key: " + key);
    }
    return static_cast<double>(std::accumulate(it->second.begin(), it->second.end(), 0LL));
}

const std::map<std::string, std::vector<std::int32_t>>& select_raw_counts(
    const expectation_snn_cuda::FrozenRichterSeededSourceResult& result,
    const std::string& execution_mode
) {
    if (execution_mode == "gpu_only_production") {
        return result.cuda_raw_counts;
    }
    if (execution_mode == "cpu_reference") {
        return result.cpu_raw_counts;
    }
    throw std::runtime_error("unknown execution mode: " + execution_mode);
}

const std::map<std::string, double>& select_q_active_fC(
    const expectation_snn_cuda::FrozenRichterSeededSourceResult& result,
    const std::string& execution_mode
) {
    if (execution_mode == "gpu_only_production") {
        return result.cuda_q_active_fC;
    }
    if (execution_mode == "cpu_reference") {
        return result.cpu_q_active_fC;
    }
    throw std::runtime_error("unknown execution mode: " + execution_mode);
}

double q_active_value(const std::map<std::string, double>& q_active, const std::string& key) {
    const auto it = q_active.find(key);
    if (it == q_active.end()) {
        throw std::runtime_error("missing Q_active key: " + key);
    }
    return it->second;
}

std::vector<std::int32_t> zero_leader_templates() {
    return std::vector<std::int32_t>(
        static_cast<std::size_t>(kV1Channels) * kV1Channels,
        0
    );
}

std::array<std::int32_t, kV1Channels> leader_template_totals(
    const std::vector<std::int32_t>& templates
) {
    if (templates.size() != static_cast<std::size_t>(kV1Channels) * kV1Channels) {
        throw std::runtime_error("leader template shape must be [12,12]");
    }
    std::array<std::int32_t, kV1Channels> out{};
    for (int leader = 0; leader < kV1Channels; ++leader) {
        for (int channel = 0; channel < kV1Channels; ++channel) {
            out[static_cast<std::size_t>(leader)] +=
                templates[
                    static_cast<std::size_t>(leader) * kV1Channels
                    + static_cast<std::size_t>(channel)
                ];
        }
    }
    return out;
}

std::vector<std::array<std::int32_t, kV1Channels>>
leader_template_rows(const std::vector<std::int32_t>& templates) {
    if (templates.size() != static_cast<std::size_t>(kV1Channels) * kV1Channels) {
        throw std::runtime_error("leader template shape must be [12,12]");
    }
    std::vector<std::array<std::int32_t, kV1Channels>> rows(kV1Channels);
    for (int leader = 0; leader < kV1Channels; ++leader) {
        for (int channel = 0; channel < kV1Channels; ++channel) {
            rows[static_cast<std::size_t>(leader)]
                [static_cast<std::size_t>(channel)] =
                    templates[
                        static_cast<std::size_t>(leader) * kV1Channels
                        + static_cast<std::size_t>(channel)
                    ];
        }
    }
    return rows;
}

void accumulate_condition(
    ConditionAccum& acc,
    const expectation_snn_cuda::FrozenRichterSeededSourceResult& r,
    const std::string& execution_mode
) {
    const auto& raw_counts = select_raw_counts(r, execution_mode);
    const auto& q_active = select_q_active_fC(r, execution_mode);
    const auto trailer_channel_counts = v1_trailer_channel_counts(raw_counts);
    const auto som_trailer_channel_counts =
        v1_som_trailer_channel_counts(raw_counts);
    const auto error_trailer_channel_counts =
        v1_error_trailer_channel_counts(raw_counts);
    const auto error_neg_trailer_channel_counts =
        v1_error_neg_trailer_channel_counts(raw_counts);
    acc.v1_leader += sum_counts(raw_counts, "v1_e.leader");
    acc.v1_preprobe += sum_counts(raw_counts, "v1_e.preprobe");
    acc.v1_trailer += sum_counts(raw_counts, "v1_e.trailer");
    acc.v1_som_trailer += sum_counts(raw_counts, "v1_som.trailer");
    acc.v1_error_trailer += sum_counts(raw_counts, "v1_error.trailer");
    acc.v1_error_neg_trailer += sum_counts(raw_counts, "v1_error_neg.trailer");
    acc.v1e_leader_q_active_fC += q_active_value(q_active, "v1e.leader");
    acc.v1e_preprobe_q_active_fC += q_active_value(q_active, "v1e.preprobe");
    acc.v1e_trailer_q_active_fC += q_active_value(q_active, "v1e.trailer");
    acc.v1som_leader_q_active_fC += q_active_value(q_active, "v1som.leader");
    acc.v1som_preprobe_q_active_fC += q_active_value(q_active, "v1som.preprobe");
    acc.v1som_trailer_q_active_fC += q_active_value(q_active, "v1som.trailer");
    acc.v1error_trailer_q_active_fC +=
        q_active_value(q_active, "v1error.trailer");
    acc.v1error_neg_trailer_q_active_fC +=
        q_active_value(q_active, "v1error_neg.trailer");
    acc.hctx_preprobe += sum_counts(raw_counts, "hctx_e.preprobe");
    acc.hctx_trailer += sum_counts(raw_counts, "hctx_e.trailer");
    acc.hpred_preprobe += sum_counts(raw_counts, "hpred_e.preprobe");
    acc.hpred_trailer += sum_counts(raw_counts, "hpred_e.trailer");
    acc.source_total += r.source_event_counts.at("total");
    for (std::size_t channel = 0; channel < trailer_channel_counts.size(); ++channel) {
        acc.v1_trailer_channel_counts[channel] += trailer_channel_counts[channel];
        acc.v1_som_trailer_channel_counts[channel] +=
            som_trailer_channel_counts[channel];
        acc.v1_error_trailer_channel_counts[channel] +=
            error_trailer_channel_counts[channel];
        acc.v1_error_neg_trailer_channel_counts[channel] +=
            error_neg_trailer_channel_counts[channel];
    }
    if (execution_mode == "cpu_reference") {
        if (r.cpu_v1_trailer_bin_channel_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_v1_som_trailer_bin_channel_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_v1_error_trailer_bin_channel_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_v1_error_neg_trailer_bin_channel_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_hpred_preprobe_channel_counts.size()
                != static_cast<std::size_t>(kV1Channels)
            || r.cpu_hpred_trailer_bin_total_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount)
            || r.cpu_hpred_trailer_bin_channel_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_hpred_feedback_held_trailer_bin_total_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount)
            || r.cpu_hpred_feedback_held_trailer_bin_channel_counts.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_hpred_feedback_normalized_trailer_bin_total_weights.size()
                != static_cast<std::size_t>(kTrailerBinCount)
            || r.cpu_hpred_feedback_normalized_trailer_bin_channel_weights.size()
                != static_cast<std::size_t>(kTrailerBinCount) * kV1Channels
            || r.cpu_hpred_feedback_normalized_preprobe_zero.size() != 1
            || r.cpu_hpred_feedback_normalized_fallback_used.size() != 1
            || r.cpu_hpred_feedback_normalized_fallback_zero_template.size()
                != 1
            || r.cpu_v1_predicted_suppression_trailer_channel_signal_sum.size()
                != static_cast<std::size_t>(kV1Channels)
            || r.cpu_v1_predicted_suppression_trailer_channel_gain_sum.size()
                != static_cast<std::size_t>(kV1Channels)
            || r.cpu_v1_predicted_suppression_trailer_raw_ie_before_sum.size()
                != static_cast<std::size_t>(kV1Channels)
            || r.cpu_v1_predicted_suppression_trailer_raw_ie_after_sum.size()
                != static_cast<std::size_t>(kV1Channels)
            || r.cpu_v1_predicted_suppression_trailer_raw_ie_delta_sum.size()
                != static_cast<std::size_t>(kV1Channels)) {
            throw std::runtime_error("CPU Richter trailer bin telemetry size mismatch");
        }
        acc.hpred_feedback_normalized_preprobe_zero +=
            r.cpu_hpred_feedback_normalized_preprobe_zero[0];
        acc.hpred_feedback_normalized_fallback_used +=
            r.cpu_hpred_feedback_normalized_fallback_used[0];
        acc.hpred_feedback_normalized_fallback_zero_template +=
            r.cpu_hpred_feedback_normalized_fallback_zero_template[0];
        for (int channel = 0; channel < kV1Channels; ++channel) {
            acc.hpred_preprobe_channel_counts[static_cast<std::size_t>(channel)] +=
                r.cpu_hpred_preprobe_channel_counts[
                    static_cast<std::size_t>(channel)
                ];
            acc.v1_predicted_suppression_trailer_channel_signal_sum[
                static_cast<std::size_t>(channel)
            ] += r.cpu_v1_predicted_suppression_trailer_channel_signal_sum[
                static_cast<std::size_t>(channel)
            ];
            acc.v1_predicted_suppression_trailer_channel_gain_sum[
                static_cast<std::size_t>(channel)
            ] += r.cpu_v1_predicted_suppression_trailer_channel_gain_sum[
                static_cast<std::size_t>(channel)
            ];
            acc.v1_predicted_suppression_trailer_raw_ie_before_sum[
                static_cast<std::size_t>(channel)
            ] += r.cpu_v1_predicted_suppression_trailer_raw_ie_before_sum[
                static_cast<std::size_t>(channel)
            ];
            acc.v1_predicted_suppression_trailer_raw_ie_after_sum[
                static_cast<std::size_t>(channel)
            ] += r.cpu_v1_predicted_suppression_trailer_raw_ie_after_sum[
                static_cast<std::size_t>(channel)
            ];
            acc.v1_predicted_suppression_trailer_raw_ie_delta_sum[
                static_cast<std::size_t>(channel)
            ] += r.cpu_v1_predicted_suppression_trailer_raw_ie_delta_sum[
                static_cast<std::size_t>(channel)
            ];
        }
        for (int bin = 0; bin < kTrailerBinCount; ++bin) {
            acc.hpred_trailer_bin_total_counts[static_cast<std::size_t>(bin)] +=
                r.cpu_hpred_trailer_bin_total_counts[static_cast<std::size_t>(bin)];
            acc.hpred_feedback_held_trailer_bin_total_counts[
                static_cast<std::size_t>(bin)
            ] += r.cpu_hpred_feedback_held_trailer_bin_total_counts[
                static_cast<std::size_t>(bin)
            ];
            acc.hpred_feedback_normalized_trailer_bin_total_weights[
                static_cast<std::size_t>(bin)
            ] += r.cpu_hpred_feedback_normalized_trailer_bin_total_weights[
                static_cast<std::size_t>(bin)
            ];
            for (int channel = 0; channel < kV1Channels; ++channel) {
                const auto flat =
                    static_cast<std::size_t>(bin) * kV1Channels
                    + static_cast<std::size_t>(channel);
                acc.v1_trailer_bin_channel_counts[static_cast<std::size_t>(bin)]
                    [static_cast<std::size_t>(channel)] +=
                        r.cpu_v1_trailer_bin_channel_counts[flat];
                acc.v1_som_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][static_cast<std::size_t>(channel)] +=
                    r.cpu_v1_som_trailer_bin_channel_counts[flat];
                acc.v1_error_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][static_cast<std::size_t>(channel)] +=
                    r.cpu_v1_error_trailer_bin_channel_counts[flat];
                acc.v1_error_neg_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][static_cast<std::size_t>(channel)] +=
                    r.cpu_v1_error_neg_trailer_bin_channel_counts[flat];
                acc.hpred_trailer_bin_channel_counts[static_cast<std::size_t>(bin)]
                    [static_cast<std::size_t>(channel)] +=
                        r.cpu_hpred_trailer_bin_channel_counts[flat];
                acc.hpred_feedback_held_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][static_cast<std::size_t>(channel)] +=
                    r.cpu_hpred_feedback_held_trailer_bin_channel_counts[flat];
                acc.hpred_feedback_normalized_trailer_bin_channel_weights[
                    static_cast<std::size_t>(bin)
                ][static_cast<std::size_t>(channel)] +=
                    r.cpu_hpred_feedback_normalized_trailer_bin_channel_weights[
                        flat
                    ];
            }
        }
    }
}

void accumulate_condition(ConditionAccum& acc, const ConditionAccum& single) {
    acc.v1_leader += single.v1_leader;
    acc.v1_preprobe += single.v1_preprobe;
    acc.v1_trailer += single.v1_trailer;
    acc.v1_som_trailer += single.v1_som_trailer;
    acc.v1_error_trailer += single.v1_error_trailer;
    acc.v1_error_neg_trailer += single.v1_error_neg_trailer;
    acc.v1e_leader_q_active_fC += single.v1e_leader_q_active_fC;
    acc.v1e_preprobe_q_active_fC += single.v1e_preprobe_q_active_fC;
    acc.v1e_trailer_q_active_fC += single.v1e_trailer_q_active_fC;
    acc.v1som_leader_q_active_fC += single.v1som_leader_q_active_fC;
    acc.v1som_preprobe_q_active_fC += single.v1som_preprobe_q_active_fC;
    acc.v1som_trailer_q_active_fC += single.v1som_trailer_q_active_fC;
    acc.v1error_trailer_q_active_fC += single.v1error_trailer_q_active_fC;
    acc.v1error_neg_trailer_q_active_fC +=
        single.v1error_neg_trailer_q_active_fC;
    acc.hctx_preprobe += single.hctx_preprobe;
    acc.hctx_trailer += single.hctx_trailer;
    acc.hpred_preprobe += single.hpred_preprobe;
    acc.hpred_trailer += single.hpred_trailer;
    acc.source_total += single.source_total;
    acc.hpred_feedback_normalized_preprobe_zero +=
        single.hpred_feedback_normalized_preprobe_zero;
    acc.hpred_feedback_normalized_fallback_used +=
        single.hpred_feedback_normalized_fallback_used;
    acc.hpred_feedback_normalized_fallback_zero_template +=
        single.hpred_feedback_normalized_fallback_zero_template;
    for (std::size_t channel = 0; channel < acc.v1_trailer_channel_counts.size(); ++channel) {
        acc.v1_trailer_channel_counts[channel] += single.v1_trailer_channel_counts[channel];
        acc.v1_som_trailer_channel_counts[channel] += single.v1_som_trailer_channel_counts[channel];
        acc.v1_error_trailer_channel_counts[channel] +=
            single.v1_error_trailer_channel_counts[channel];
        acc.v1_error_neg_trailer_channel_counts[channel] +=
            single.v1_error_neg_trailer_channel_counts[channel];
        acc.hpred_preprobe_channel_counts[channel] +=
            single.hpred_preprobe_channel_counts[channel];
        acc.v1_predicted_suppression_trailer_channel_signal_sum[channel] +=
            single.v1_predicted_suppression_trailer_channel_signal_sum[channel];
        acc.v1_predicted_suppression_trailer_channel_gain_sum[channel] +=
            single.v1_predicted_suppression_trailer_channel_gain_sum[channel];
        acc.v1_predicted_suppression_trailer_raw_ie_before_sum[channel] +=
            single.v1_predicted_suppression_trailer_raw_ie_before_sum[channel];
        acc.v1_predicted_suppression_trailer_raw_ie_after_sum[channel] +=
            single.v1_predicted_suppression_trailer_raw_ie_after_sum[channel];
        acc.v1_predicted_suppression_trailer_raw_ie_delta_sum[channel] +=
            single.v1_predicted_suppression_trailer_raw_ie_delta_sum[channel];
    }
    for (std::size_t bin = 0; bin < acc.v1_trailer_bin_channel_counts.size(); ++bin) {
        acc.hpred_trailer_bin_total_counts[bin] +=
            single.hpred_trailer_bin_total_counts[bin];
        acc.hpred_feedback_held_trailer_bin_total_counts[bin] +=
            single.hpred_feedback_held_trailer_bin_total_counts[bin];
        acc.hpred_feedback_normalized_trailer_bin_total_weights[bin] +=
            single.hpred_feedback_normalized_trailer_bin_total_weights[bin];
        for (std::size_t channel = 0;
             channel < acc.v1_trailer_bin_channel_counts[bin].size();
             ++channel) {
            acc.v1_trailer_bin_channel_counts[bin][channel] +=
                single.v1_trailer_bin_channel_counts[bin][channel];
            acc.v1_som_trailer_bin_channel_counts[bin][channel] +=
                single.v1_som_trailer_bin_channel_counts[bin][channel];
            acc.v1_error_trailer_bin_channel_counts[bin][channel] +=
                single.v1_error_trailer_bin_channel_counts[bin][channel];
            acc.v1_error_neg_trailer_bin_channel_counts[bin][channel] +=
                single.v1_error_neg_trailer_bin_channel_counts[bin][channel];
            acc.hpred_trailer_bin_channel_counts[bin][channel] +=
                single.hpred_trailer_bin_channel_counts[bin][channel];
            acc.hpred_feedback_held_trailer_bin_channel_counts[bin][channel] +=
                single.hpred_feedback_held_trailer_bin_channel_counts[bin][channel];
            acc.hpred_feedback_normalized_trailer_bin_channel_weights[bin][channel] +=
                single.hpred_feedback_normalized_trailer_bin_channel_weights[bin][channel];
        }
    }
}

expectation_snn_cuda::FrozenRichterSeededSourceResult run_native_richter_condition(
    const NativeTopology& topology,
    const std::string& execution_mode,
    std::int64_t seed,
    int leader_channel,
    int trailer_channel,
    double grating_rate_hz,
    double baseline_rate_hz,
    const std::string& feedback_replay_mode = "raw",
    double feedback_replay_target_per_bin =
        kFeedbackReplayTargetPerBinDefault,
    const std::string& feedback_replay_fallback = "none",
    const std::vector<std::int32_t>& feedback_replay_leader_templates =
        zero_leader_templates(),
    double v1_som_to_e_scale = kV1SomToEScaleDefault,
    double v1_som_divisive_scale = kV1SomDivisiveScaleDefault,
    double v1_direct_divisive_scale = kV1DirectDivisiveScaleDefault,
    double v1_feedforward_divisive_scale =
        kV1FeedforwardDivisiveScaleDefault,
    std::int32_t v1_feedforward_divisive_gate_source_id = 0,
    std::int32_t v1_direct_divisive_gate_source_id = 0,
    double v1_predicted_suppression_scale =
        kV1PredictedSuppressionScaleDefault,
    double v1_predicted_suppression_neighbor_weight =
        kV1PredictedSuppressionNeighborWeightDefault,
    std::int32_t v1_predicted_suppression_locus_id =
        kV1PredictedSuppressionLocusEffectiveIe,
    double v1_stim_sigma_deg = kV1StimSigmaDegDefault,
    std::int32_t v1_error_comparator_mode_id = kV1ErrorComparatorModeOff,
    double v1_error_sensory_gain = 1.0,
    double v1_error_prediction_gain = 1.0,
    std::int32_t v1_error_prediction_shift = 0
) {
    if (execution_mode == "gpu_only_production") {
        return expectation_snn_cuda::run_frozen_richter_seeded_source_cuda(
            "v1_stim_to_e",
            topology.stim_pre,
            topology.stim_post,
            topology.stim_w,
            35.0,
            topology.stim_channel,
            "v1_to_h_ctx",
            topology.v1_to_h_pre,
            topology.v1_to_h_post,
            topology.v1_to_h_w,
            80.0,
            "ctx_to_pred",
            topology.ctx_to_pred_pre,
            topology.ctx_to_pred_post,
            topology.ctx_to_pred_w,
            400.0,
            "fb_pred_to_v1e_apical",
            topology.fb_direct_pre,
            topology.fb_direct_post,
            topology.fb_direct_w,
            kFeedbackDirectBaseDriveAmpPA,
            "fb_pred_to_v1som",
            topology.fb_som_pre,
            topology.fb_som_post,
            topology.fb_som_w,
            kFeedbackSomBaseDriveAmpPA,
            seed,
            leader_channel,
            trailer_channel,
            grating_rate_hz,
            baseline_rate_hz,
            25000,
            0,
            4000,
            4000,
            5000,
            5000,
            10000,
            10000,
            25000,
            feedback_replay_mode,
            feedback_replay_target_per_bin,
            feedback_replay_fallback,
            feedback_replay_leader_templates,
            v1_som_to_e_scale,
            v1_som_divisive_scale,
            v1_direct_divisive_scale,
            v1_feedforward_divisive_scale,
            v1_feedforward_divisive_gate_source_id,
            v1_direct_divisive_gate_source_id,
            v1_predicted_suppression_scale,
            v1_predicted_suppression_neighbor_weight,
            v1_predicted_suppression_locus_id,
            v1_stim_sigma_deg,
            v1_error_comparator_mode_id,
            v1_error_sensory_gain,
            v1_error_prediction_gain,
            v1_error_prediction_shift
        );
    }
    if (execution_mode == "cpu_reference") {
        return expectation_snn_cuda::run_frozen_richter_seeded_source_cpu(
            "v1_stim_to_e",
            topology.stim_pre,
            topology.stim_post,
            topology.stim_w,
            35.0,
            topology.stim_channel,
            "v1_to_h_ctx",
            topology.v1_to_h_pre,
            topology.v1_to_h_post,
            topology.v1_to_h_w,
            80.0,
            "ctx_to_pred",
            topology.ctx_to_pred_pre,
            topology.ctx_to_pred_post,
            topology.ctx_to_pred_w,
            400.0,
            "fb_pred_to_v1e_apical",
            topology.fb_direct_pre,
            topology.fb_direct_post,
            topology.fb_direct_w,
            kFeedbackDirectBaseDriveAmpPA,
            "fb_pred_to_v1som",
            topology.fb_som_pre,
            topology.fb_som_post,
            topology.fb_som_w,
            kFeedbackSomBaseDriveAmpPA,
            seed,
            leader_channel,
            trailer_channel,
            grating_rate_hz,
            baseline_rate_hz,
            25000,
            0,
            4000,
            4000,
            5000,
            5000,
            10000,
            10000,
            25000,
            feedback_replay_mode,
            feedback_replay_target_per_bin,
            feedback_replay_fallback,
            feedback_replay_leader_templates,
            v1_som_to_e_scale,
            v1_som_divisive_scale,
            v1_direct_divisive_scale,
            v1_feedforward_divisive_scale,
            v1_feedforward_divisive_gate_source_id,
            v1_direct_divisive_gate_source_id,
            v1_predicted_suppression_scale,
            v1_predicted_suppression_neighbor_weight,
            v1_predicted_suppression_locus_id,
            v1_stim_sigma_deg,
            v1_error_comparator_mode_id,
            v1_error_sensory_gain,
            v1_error_prediction_gain,
            v1_error_prediction_shift
        );
    }
    throw std::runtime_error("unknown execution mode: " + execution_mode);
}

std::vector<ConditionAccum> run_native_richter_conditions_batched(
    const NativeTopology& topology,
    const std::vector<RichterConditionSpec>& specs,
    double grating_rate_hz,
    double baseline_rate_hz,
    const std::string& feedback_replay_mode = "raw",
    double feedback_replay_target_per_bin =
        kFeedbackReplayTargetPerBinDefault,
    const std::string& feedback_replay_fallback = "none",
    const std::vector<std::int32_t>& feedback_replay_leader_templates =
        zero_leader_templates(),
    double v1_som_to_e_scale = kV1SomToEScaleDefault,
    double v1_som_divisive_scale = kV1SomDivisiveScaleDefault,
    double v1_direct_divisive_scale = kV1DirectDivisiveScaleDefault,
    double v1_feedforward_divisive_scale =
        kV1FeedforwardDivisiveScaleDefault,
    std::int32_t v1_feedforward_divisive_gate_source_id = 0,
    std::int32_t v1_direct_divisive_gate_source_id = 0,
    double v1_predicted_suppression_scale =
        kV1PredictedSuppressionScaleDefault,
    double v1_predicted_suppression_neighbor_weight =
        kV1PredictedSuppressionNeighborWeightDefault,
    std::int32_t v1_predicted_suppression_locus_id =
        kV1PredictedSuppressionLocusEffectiveIe,
    double v1_stim_sigma_deg = kV1StimSigmaDegDefault,
    std::int32_t v1_error_comparator_mode_id = kV1ErrorComparatorModeOff,
    double v1_error_sensory_gain = 1.0,
    double v1_error_prediction_gain = 1.0,
    std::int32_t v1_error_prediction_shift = 0
) {
    if (specs.empty()) {
        return {};
    }
    std::vector<std::int64_t> seeds;
    std::vector<std::int32_t> leader_channels;
    std::vector<std::int32_t> trailer_channels;
    seeds.reserve(specs.size());
    leader_channels.reserve(specs.size());
    trailer_channels.reserve(specs.size());
    for (const auto& spec : specs) {
        seeds.push_back(spec.seed);
        leader_channels.push_back(spec.leader_channel);
        trailer_channels.push_back(spec.trailer_channel);
    }
    const auto batch = expectation_snn_cuda::run_frozen_richter_seeded_source_cuda_batched(
        "v1_stim_to_e",
        topology.stim_pre,
        topology.stim_post,
        topology.stim_w,
        35.0,
        topology.stim_channel,
        "v1_to_h_ctx",
        topology.v1_to_h_pre,
        topology.v1_to_h_post,
        topology.v1_to_h_w,
        80.0,
        "ctx_to_pred",
        topology.ctx_to_pred_pre,
        topology.ctx_to_pred_post,
        topology.ctx_to_pred_w,
        400.0,
        "fb_pred_to_v1e_apical",
        topology.fb_direct_pre,
        topology.fb_direct_post,
        topology.fb_direct_w,
        kFeedbackDirectBaseDriveAmpPA,
        "fb_pred_to_v1som",
        topology.fb_som_pre,
        topology.fb_som_post,
        topology.fb_som_w,
        kFeedbackSomBaseDriveAmpPA,
        seeds,
        leader_channels,
        trailer_channels,
        grating_rate_hz,
        baseline_rate_hz,
        25000,
        0,
        4000,
        4000,
        5000,
        5000,
        10000,
        10000,
        25000,
        feedback_replay_mode,
        feedback_replay_target_per_bin,
        feedback_replay_fallback,
        feedback_replay_leader_templates,
        v1_som_to_e_scale,
        v1_som_divisive_scale,
        v1_direct_divisive_scale,
        v1_feedforward_divisive_scale,
        v1_feedforward_divisive_gate_source_id,
        v1_direct_divisive_gate_source_id,
        v1_predicted_suppression_scale,
        v1_predicted_suppression_neighbor_weight,
        v1_predicted_suppression_locus_id,
        v1_stim_sigma_deg,
        v1_error_comparator_mode_id,
        v1_error_sensory_gain,
        v1_error_prediction_gain,
        v1_error_prediction_shift
    );
    if (static_cast<std::size_t>(batch.n_conditions) != specs.size()) {
        throw std::runtime_error("batched Richter result size mismatch");
    }
    if (batch.v1_trailer_channel_counts.size()
        != specs.size() * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error("batched Richter trailer channel count size mismatch");
    }
    if (batch.v1_trailer_bin_channel_counts.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
            * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter trailer bin-channel count size mismatch"
        );
    }
    if (batch.v1_som_trailer_total_counts.size() != specs.size()
        || batch.v1_som_trailer_channel_counts.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_som_trailer_bin_channel_counts.size()
            != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
                * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error("batched Richter SOM trailer count size mismatch");
    }
    if (batch.v1_error_trailer_total_counts.size() != specs.size()
        || batch.v1_error_trailer_channel_counts.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_error_trailer_bin_channel_counts.size()
            != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
                * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error("batched Richter V1_ERROR trailer count size mismatch");
    }
    if (batch.v1_error_neg_trailer_total_counts.size() != specs.size()
        || batch.v1_error_neg_trailer_channel_counts.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_error_neg_trailer_bin_channel_counts.size()
            != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
                * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter V1_ERROR_NEG trailer count size mismatch"
        );
    }
    if (batch.v1e_q_active_fC_by_phase.size() != specs.size() * 3
        || batch.v1som_q_active_fC_by_phase.size() != specs.size() * 3
        || batch.v1error_q_active_fC_by_phase.size() != specs.size() * 3
        || batch.v1error_neg_q_active_fC_by_phase.size() != specs.size() * 3) {
        throw std::runtime_error("batched Richter Q_active phase size mismatch");
    }
    if (batch.hpred_preprobe_channel_counts.size()
        != specs.size() * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter H_pred preprobe channel count size mismatch"
        );
    }
    if (batch.hpred_trailer_bin_total_counts.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)) {
        throw std::runtime_error(
            "batched Richter H_pred trailer bin-total count size mismatch"
        );
    }
    if (batch.hpred_trailer_bin_channel_counts.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
            * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter H_pred trailer bin-channel count size mismatch"
        );
    }
    if (batch.hpred_feedback_held_trailer_bin_total_counts.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)) {
        throw std::runtime_error(
            "batched Richter held-feedback trailer bin-total count size mismatch"
        );
    }
    if (batch.hpred_feedback_held_trailer_bin_channel_counts.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
            * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter held-feedback trailer bin-channel count size mismatch"
        );
    }
    if (batch.hpred_feedback_normalized_trailer_bin_total_weights.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)) {
        throw std::runtime_error(
            "batched Richter normalized-feedback trailer bin-total weight size mismatch"
        );
    }
    if (batch.hpred_feedback_normalized_trailer_bin_channel_weights.size()
        != specs.size() * static_cast<std::size_t>(kTrailerBinCount)
            * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter normalized-feedback trailer bin-channel weight size mismatch"
        );
    }
    if (batch.hpred_feedback_normalized_preprobe_zero.size() != specs.size()
        || batch.hpred_feedback_normalized_fallback_used.size() != specs.size()
        || batch.hpred_feedback_normalized_fallback_zero_template.size()
            != specs.size()) {
        throw std::runtime_error(
            "batched Richter normalized-feedback fallback flag size mismatch"
        );
    }
    if (batch.v1_predicted_suppression_trailer_channel_signal_sum.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_predicted_suppression_trailer_channel_gain_sum.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_predicted_suppression_trailer_raw_ie_before_sum.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_predicted_suppression_trailer_raw_ie_after_sum.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)
        || batch.v1_predicted_suppression_trailer_raw_ie_delta_sum.size()
            != specs.size() * static_cast<std::size_t>(kV1Channels)) {
        throw std::runtime_error(
            "batched Richter predicted-suppression telemetry size mismatch"
        );
    }
    std::vector<ConditionAccum> out(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        out[i].v1_leader = batch.v1_leader_total_counts[i];
        out[i].v1_preprobe = batch.v1_preprobe_total_counts[i];
        out[i].v1_trailer = batch.v1_trailer_total_counts[i];
        out[i].v1_som_trailer = batch.v1_som_trailer_total_counts[i];
        out[i].v1_error_trailer = batch.v1_error_trailer_total_counts[i];
        out[i].v1_error_neg_trailer =
            batch.v1_error_neg_trailer_total_counts[i];
        out[i].v1e_leader_q_active_fC =
            batch.v1e_q_active_fC_by_phase[i * 3 + 0];
        out[i].v1e_preprobe_q_active_fC =
            batch.v1e_q_active_fC_by_phase[i * 3 + 1];
        out[i].v1e_trailer_q_active_fC =
            batch.v1e_q_active_fC_by_phase[i * 3 + 2];
        out[i].v1som_leader_q_active_fC =
            batch.v1som_q_active_fC_by_phase[i * 3 + 0];
        out[i].v1som_preprobe_q_active_fC =
            batch.v1som_q_active_fC_by_phase[i * 3 + 1];
        out[i].v1som_trailer_q_active_fC =
            batch.v1som_q_active_fC_by_phase[i * 3 + 2];
        out[i].v1error_trailer_q_active_fC =
            batch.v1error_q_active_fC_by_phase[i * 3 + 2];
        out[i].v1error_neg_trailer_q_active_fC =
            batch.v1error_neg_q_active_fC_by_phase[i * 3 + 2];
        out[i].hctx_preprobe = batch.hctx_preprobe_total_counts[i];
        out[i].hctx_trailer = batch.hctx_trailer_total_counts[i];
        out[i].hpred_preprobe = batch.hpred_preprobe_total_counts[i];
        out[i].hpred_trailer = batch.hpred_trailer_total_counts[i];
        out[i].source_total = batch.source_total_counts[i];
        out[i].hpred_feedback_normalized_preprobe_zero =
            batch.hpred_feedback_normalized_preprobe_zero[i];
        out[i].hpred_feedback_normalized_fallback_used =
            batch.hpred_feedback_normalized_fallback_used[i];
        out[i].hpred_feedback_normalized_fallback_zero_template =
            batch.hpred_feedback_normalized_fallback_zero_template[i];
        for (int channel = 0; channel < kV1Channels; ++channel) {
            out[i].v1_trailer_channel_counts[static_cast<std::size_t>(channel)] =
                batch.v1_trailer_channel_counts[
                    i * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
            out[i].v1_som_trailer_channel_counts[static_cast<std::size_t>(channel)] =
                batch.v1_som_trailer_channel_counts[
                    i * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
            out[i].v1_error_trailer_channel_counts[static_cast<std::size_t>(channel)] =
                batch.v1_error_trailer_channel_counts[
                    i * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
            out[i].v1_error_neg_trailer_channel_counts[
                static_cast<std::size_t>(channel)
            ] = batch.v1_error_neg_trailer_channel_counts[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
            out[i].hpred_preprobe_channel_counts[
                static_cast<std::size_t>(channel)
            ] = batch.hpred_preprobe_channel_counts[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
            out[i].v1_predicted_suppression_trailer_channel_signal_sum[
                static_cast<std::size_t>(channel)
            ] = batch.v1_predicted_suppression_trailer_channel_signal_sum[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
            out[i].v1_predicted_suppression_trailer_channel_gain_sum[
                static_cast<std::size_t>(channel)
            ] = batch.v1_predicted_suppression_trailer_channel_gain_sum[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
            out[i].v1_predicted_suppression_trailer_raw_ie_before_sum[
                static_cast<std::size_t>(channel)
            ] = batch.v1_predicted_suppression_trailer_raw_ie_before_sum[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
            out[i].v1_predicted_suppression_trailer_raw_ie_after_sum[
                static_cast<std::size_t>(channel)
            ] = batch.v1_predicted_suppression_trailer_raw_ie_after_sum[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
            out[i].v1_predicted_suppression_trailer_raw_ie_delta_sum[
                static_cast<std::size_t>(channel)
            ] = batch.v1_predicted_suppression_trailer_raw_ie_delta_sum[
                i * static_cast<std::size_t>(kV1Channels)
                + static_cast<std::size_t>(channel)
            ];
        }
        for (int bin = 0; bin < kTrailerBinCount; ++bin) {
            for (int channel = 0; channel < kV1Channels; ++channel) {
                out[i].v1_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.v1_trailer_bin_channel_counts[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
                out[i].v1_som_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.v1_som_trailer_bin_channel_counts[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
                out[i].v1_error_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.v1_error_trailer_bin_channel_counts[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
                out[i].v1_error_neg_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.v1_error_neg_trailer_bin_channel_counts[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
            }
            out[i].hpred_trailer_bin_total_counts[
                static_cast<std::size_t>(bin)
            ] = batch.hpred_trailer_bin_total_counts[
                i * static_cast<std::size_t>(kTrailerBinCount)
                + static_cast<std::size_t>(bin)
            ];
            out[i].hpred_feedback_held_trailer_bin_total_counts[
                static_cast<std::size_t>(bin)
            ] = batch.hpred_feedback_held_trailer_bin_total_counts[
                i * static_cast<std::size_t>(kTrailerBinCount)
                + static_cast<std::size_t>(bin)
            ];
            out[i].hpred_feedback_normalized_trailer_bin_total_weights[
                static_cast<std::size_t>(bin)
            ] = batch.hpred_feedback_normalized_trailer_bin_total_weights[
                i * static_cast<std::size_t>(kTrailerBinCount)
                + static_cast<std::size_t>(bin)
            ];
            for (int channel = 0; channel < kV1Channels; ++channel) {
                out[i].hpred_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.hpred_trailer_bin_channel_counts[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
                out[i].hpred_feedback_held_trailer_bin_channel_counts[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.hpred_feedback_held_trailer_bin_channel_counts[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
                out[i].hpred_feedback_normalized_trailer_bin_channel_weights[
                    static_cast<std::size_t>(bin)
                ][
                    static_cast<std::size_t>(channel)
                ] = batch.hpred_feedback_normalized_trailer_bin_channel_weights[
                    (i * static_cast<std::size_t>(kTrailerBinCount)
                        + static_cast<std::size_t>(bin))
                        * static_cast<std::size_t>(kV1Channels)
                    + static_cast<std::size_t>(channel)
                ];
            }
        }
    }
    return out;
}

std::vector<std::int32_t> build_feedback_replay_leader_templates(
    const NativeTopology& topology,
    const std::vector<RichterConditionSpec>& specs,
    const std::string& execution_mode,
    double grating_rate_hz,
    double baseline_rate_hz,
    double feedback_replay_target_per_bin,
    double v1_som_to_e_scale = kV1SomToEScaleDefault,
    double v1_som_divisive_scale = kV1SomDivisiveScaleDefault,
    double v1_direct_divisive_scale = kV1DirectDivisiveScaleDefault,
    double v1_feedforward_divisive_scale =
        kV1FeedforwardDivisiveScaleDefault,
    std::int32_t v1_feedforward_divisive_gate_source_id = 0,
    std::int32_t v1_direct_divisive_gate_source_id = 0,
    double v1_predicted_suppression_scale =
        kV1PredictedSuppressionScaleDefault,
    double v1_predicted_suppression_neighbor_weight =
        kV1PredictedSuppressionNeighborWeightDefault,
    std::int32_t v1_predicted_suppression_locus_id =
        kV1PredictedSuppressionLocusEffectiveIe,
    double v1_stim_sigma_deg = kV1StimSigmaDegDefault
) {
    std::vector<std::int32_t> templates = zero_leader_templates();
    if (specs.empty()) {
        return templates;
    }
    std::vector<RichterConditionSpec> template_specs;
    template_specs.reserve(specs.size());
    for (const auto& spec : specs) {
        RichterConditionSpec leader_only = spec;
        leader_only.is_expected = 1;
        leader_only.trailer_orientation = spec.leader_orientation;
        leader_only.step = 0;
        leader_only.trailer_channel = spec.leader_channel;
        template_specs.push_back(leader_only);
    }
    auto accumulate_template = [&](const RichterConditionSpec& spec,
                                   const ConditionAccum& acc) {
        if (spec.leader_channel < 0 || spec.leader_channel >= kV1Channels) {
            throw std::runtime_error("leader template leader channel out of range");
        }
        const std::size_t base =
            static_cast<std::size_t>(spec.leader_channel) * kV1Channels;
        for (int channel = 0; channel < kV1Channels; ++channel) {
            templates[base + static_cast<std::size_t>(channel)] +=
                acc.hpred_preprobe_channel_counts[
                    static_cast<std::size_t>(channel)
                ];
        }
    };
    if (execution_mode == "gpu_only_production") {
        const auto prepass = run_native_richter_conditions_batched(
            topology,
            template_specs,
            grating_rate_hz,
            baseline_rate_hz,
            "normalized",
            feedback_replay_target_per_bin,
            "none",
            zero_leader_templates(),
            v1_som_to_e_scale,
            v1_som_divisive_scale,
            v1_direct_divisive_scale,
            v1_feedforward_divisive_scale,
            v1_feedforward_divisive_gate_source_id,
            v1_direct_divisive_gate_source_id,
            v1_predicted_suppression_scale,
            v1_predicted_suppression_neighbor_weight,
            v1_predicted_suppression_locus_id,
            v1_stim_sigma_deg
        );
        if (prepass.size() != template_specs.size()) {
            throw std::runtime_error("leader-template prepass size mismatch");
        }
        for (std::size_t i = 0; i < template_specs.size(); ++i) {
            accumulate_template(template_specs[i], prepass[i]);
        }
        return templates;
    }
    if (execution_mode == "cpu_reference") {
        for (const auto& spec : template_specs) {
            const auto result = run_native_richter_condition(
                topology,
                execution_mode,
                spec.seed,
                spec.leader_channel,
                spec.trailer_channel,
                grating_rate_hz,
                baseline_rate_hz,
                "normalized",
                feedback_replay_target_per_bin,
                "none",
                zero_leader_templates(),
                v1_som_to_e_scale,
                v1_som_divisive_scale,
                v1_direct_divisive_scale,
                v1_feedforward_divisive_scale,
                v1_feedforward_divisive_gate_source_id,
                v1_direct_divisive_gate_source_id,
                v1_predicted_suppression_scale,
                v1_predicted_suppression_neighbor_weight,
                v1_predicted_suppression_locus_id,
                v1_stim_sigma_deg
            );
            ConditionAccum single;
            accumulate_condition(single, result, execution_mode);
            accumulate_template(spec, single);
        }
        return templates;
    }
    throw std::runtime_error("unknown execution mode for leader-template prepass");
}

int sensory_diagnostics(const Args& args) {
    if (args.checkpoint_path.empty()) {
        throw std::runtime_error("sensory-diagnostics requires --checkpoint PATH");
    }
    if (args.out_path.empty()) {
        throw std::runtime_error("sensory-diagnostics requires --out PATH");
    }
    if (args.repeats < 1) {
        throw std::runtime_error("sensory-diagnostics requires --repeats >= 1");
    }
    if (args.execution_mode != "gpu_only_production") {
        throw std::runtime_error(
            "sensory-diagnostics currently supports only --execution-mode gpu_only_production"
        );
    }
    const auto total_t0 = std::chrono::steady_clock::now();

    std::vector<double> w_ctx_pred;
    const Stage1CheckpointMeta checkpoint = validate_stage1_checkpoint(
        args.checkpoint_path,
        w_ctx_pred
    );
    const FeedbackBalance feedback = resolve_feedback_balance();
    const NativeTopology topology = make_native_topology(w_ctx_pred, feedback);

    std::vector<RichterConditionSpec> specs;
    specs.reserve(static_cast<std::size_t>(args.repeats) * kV1Channels);
    for (int rep = 0; rep < args.repeats; ++rep) {
        for (int stimulus_channel = 0; stimulus_channel < kV1Channels; ++stimulus_channel) {
            specs.push_back(RichterConditionSpec{
                args.seed + 1000003LL * rep + 4099LL * stimulus_channel,
                rep,
                1,
                stimulus_channel,
                stimulus_channel,
                0,
                stimulus_channel,
                stimulus_channel,
            });
        }
    }

    const auto run_t0 = std::chrono::steady_clock::now();
    const auto batch = run_native_richter_conditions_batched(
        topology,
        specs,
        args.grating_rate_hz,
        args.baseline_rate_hz
    );
    const auto run_t1 = std::chrono::steady_clock::now();
    if (batch.size() != specs.size()) {
        throw std::runtime_error("sensory-diagnostics batch size mismatch");
    }

    std::vector<std::int32_t> trial_index;
    std::vector<std::int32_t> trial_rep;
    std::vector<std::int32_t> trial_stimulus_channel;
    std::vector<std::int32_t> trial_source_total_counts;
    std::vector<std::int32_t> trial_hctx_e_trailer_total_counts;
    std::vector<std::int32_t> trial_hpred_e_trailer_total_counts;
    std::vector<std::array<std::int32_t, kV1Channels>> trial_v1_e_trailer_channel_counts;
    std::vector<std::array<std::int32_t, kV1Channels>> trial_v1_som_trailer_channel_counts;
    trial_index.reserve(batch.size());
    trial_rep.reserve(batch.size());
    trial_stimulus_channel.reserve(batch.size());
    trial_source_total_counts.reserve(batch.size());
    trial_hctx_e_trailer_total_counts.reserve(batch.size());
    trial_hpred_e_trailer_total_counts.reserve(batch.size());
    trial_v1_e_trailer_channel_counts.reserve(batch.size());
    trial_v1_som_trailer_channel_counts.reserve(batch.size());

    std::array<std::array<double, kV1Channels>, kV1Channels> mean_v1_e_channel_rates_hz{};
    std::array<std::array<double, kV1Channels>, kV1Channels> mean_v1_som_channel_rates_hz{};
    std::array<double, kV1Channels> mean_hctx_e_trailer_population_rate_hz{};
    std::array<double, kV1Channels> mean_hpred_e_trailer_population_rate_hz{};
    std::array<double, kV1Channels> mean_source_events_per_trial{};

    std::uint64_t semantic_hash = 1469598103934665603ull;
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.seed);
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.repeats);
    semantic_hash = fnv1a_update(semantic_hash, checkpoint.content_hash.data(), checkpoint.content_hash.size());
    semantic_hash = fnv1a_update(semantic_hash, checkpoint.w_hash.data(), checkpoint.w_hash.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
        const int stimulus_channel = specs[i].leader_channel;
        const auto v1_e_rates =
            trailer_channel_rates_hz(batch[i].v1_trailer_channel_counts, kV1CellsPerChannel);
        const auto v1_som_rates =
            trailer_channel_rates_hz(batch[i].v1_som_trailer_channel_counts, kV1SomCellsPerChannel);
        trial_index.push_back(static_cast<std::int32_t>(i));
        trial_rep.push_back(specs[i].rep);
        trial_stimulus_channel.push_back(stimulus_channel);
        trial_source_total_counts.push_back(static_cast<std::int32_t>(batch[i].source_total));
        trial_hctx_e_trailer_total_counts.push_back(static_cast<std::int32_t>(batch[i].hctx_trailer));
        trial_hpred_e_trailer_total_counts.push_back(static_cast<std::int32_t>(batch[i].hpred_trailer));
        trial_v1_e_trailer_channel_counts.push_back(batch[i].v1_trailer_channel_counts);
        trial_v1_som_trailer_channel_counts.push_back(batch[i].v1_som_trailer_channel_counts);
        for (int channel = 0; channel < kV1Channels; ++channel) {
            mean_v1_e_channel_rates_hz[stimulus_channel][channel] += v1_e_rates[static_cast<std::size_t>(channel)];
            mean_v1_som_channel_rates_hz[stimulus_channel][channel] += v1_som_rates[static_cast<std::size_t>(channel)];
        }
        mean_hctx_e_trailer_population_rate_hz[stimulus_channel] +=
            batch[i].hctx_trailer / (static_cast<double>(kV1Channels * kHCellsPerChannel) * kTrailerWindowSeconds);
        mean_hpred_e_trailer_population_rate_hz[stimulus_channel] +=
            batch[i].hpred_trailer / (static_cast<double>(kV1Channels * kHCellsPerChannel) * kTrailerWindowSeconds);
        mean_source_events_per_trial[stimulus_channel] += static_cast<double>(batch[i].source_total);
        semantic_hash = fnv1a_update_scalar(semantic_hash, specs[i].rep);
        semantic_hash = fnv1a_update_scalar(semantic_hash, stimulus_channel);
        semantic_hash = fnv1a_update_scalar(semantic_hash, batch[i].source_total);
        semantic_hash = fnv1a_update_scalar(semantic_hash, batch[i].hctx_trailer);
        semantic_hash = fnv1a_update_scalar(semantic_hash, batch[i].hpred_trailer);
        for (const auto count : batch[i].v1_trailer_channel_counts) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto count : batch[i].v1_som_trailer_channel_counts) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
    }
    for (int stimulus_channel = 0; stimulus_channel < kV1Channels; ++stimulus_channel) {
        for (int channel = 0; channel < kV1Channels; ++channel) {
            mean_v1_e_channel_rates_hz[stimulus_channel][channel] /=
                static_cast<double>(args.repeats);
            mean_v1_som_channel_rates_hz[stimulus_channel][channel] /=
                static_cast<double>(args.repeats);
        }
        mean_hctx_e_trailer_population_rate_hz[stimulus_channel] /=
            static_cast<double>(args.repeats);
        mean_hpred_e_trailer_population_rate_hz[stimulus_channel] /=
            static_cast<double>(args.repeats);
        mean_source_events_per_trial[stimulus_channel] /=
            static_cast<double>(args.repeats);
    }
    const std::string result_hash = hash_hex(semantic_hash);

    const auto write_t0 = std::chrono::steady_clock::now();
    if (!args.out_path.parent_path().empty()) {
        std::filesystem::create_directories(args.out_path.parent_path());
    }
    std::ofstream out(args.out_path);
    if (!out) {
        throw std::runtime_error(
            "failed to open sensory diagnostics JSON: " + args.out_path.string()
        );
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"artifact_kind\": \"native_sensory_stage_diagnostics\",\n";
    out << "  \"backend\": \"native_cxx_cuda_executable\",\n";
    out << "  \"execution_mode\": \"" << json_escape(args.execution_mode) << "\",\n";
    out << "  \"seed\": " << args.seed << ",\n";
    out << "  \"content_hash_fnv1a64\": \"" << result_hash << "\",\n";
    out << "  \"checkpoint\": {\"path\": \"" << json_escape(checkpoint.json_path.string())
        << "\", \"content_hash_fnv1a64\": \"" << checkpoint.content_hash
        << "\", \"W_ctx_pred_path\": \"" << json_escape(checkpoint.w_path.string())
        << "\", \"W_ctx_pred_fnv1a64\": \"" << checkpoint.w_hash << "\"},\n";
    out << "  \"architecture\": {\n";
    out << "    \"explicit_laminar_structure\": false,\n";
    out << "    \"notes\": [\"V1_E has soma and apical compartments but there is no explicit L4/L2/3 split in the standalone runtime\","
        << "\"H_ctx_E and H_pred_E are active excitatory populations\","
        << "\"V1_PV and H inhibitory populations are not active in the standalone Richter path\","
        << "\"Checkpointed ctx_ee/pred_ee recurrent banks are absent at load time in this path\"],\n";
    out << "    \"populations\": {\n";
    out << "      \"V1_E\": {\"active\": true, \"n_cells\": 192, \"channels\": 12, \"cells_per_channel\": 16, \"apical_compartment\": true},\n";
    out << "      \"V1_SOM\": {\"active\": true, \"n_cells\": 48, \"channels\": 12, \"cells_per_channel\": 4, \"apical_compartment\": false},\n";
    out << "      \"H_ctx_E\": {\"active\": true, \"n_cells\": 192, \"channels\": 12, \"cells_per_channel\": 16},\n";
    out << "      \"H_pred_E\": {\"active\": true, \"n_cells\": 192, \"channels\": 12, \"cells_per_channel\": 16},\n";
    out << "      \"V1_PV\": {\"active\": false},\n";
    out << "      \"H_inh\": {\"active\": false}\n";
    out << "    },\n";
    out << "    \"active_mechanisms\": {\"stimulus_gaussian_wrapped_ring\": true, \"v1_som_to_e_suppression\": true, \"ctx_to_pred_checkpoint_consumed\": true, \"h_recurrent_loaded\": false, \"h_inhibitory_active\": false},\n";
    out << "    \"edge_counts\": {\"v1_stim_to_e\": " << topology.stim_pre.size()
        << ", \"v1_to_h_ctx\": " << topology.v1_to_h_pre.size()
        << ", \"ctx_to_pred\": " << topology.ctx_to_pred_pre.size()
        << ", \"fb_pred_to_v1e_apical\": " << topology.fb_direct_pre.size()
        << ", \"fb_pred_to_v1som\": " << topology.fb_som_pre.size()
        << ", \"v1_som_to_e\": 768}\n";
    out << "  },\n";
    out << "  \"stimulus\": {\"sweep_channels\": [0,1,2,3,4,5,6,7,8,9,10,11], \"repeats_per_channel\": "
        << args.repeats << ", \"leader_equals_trailer\": true, \"gaussian_sigma_deg\": 22.0, \"grating_rate_hz\": "
        << args.grating_rate_hz << ", \"baseline_rate_hz\": " << args.baseline_rate_hz
        << ", \"trailer_window_seconds\": " << kTrailerWindowSeconds << "},\n";
    out << "  \"trial_data\": {\n";
    out << "    \"trial_index\": "; write_vector_json(out, trial_index); out << ",\n";
    out << "    \"rep\": "; write_vector_json(out, trial_rep); out << ",\n";
    out << "    \"stimulus_channel\": "; write_vector_json(out, trial_stimulus_channel); out << ",\n";
    out << "    \"source_total_counts\": "; write_vector_json(out, trial_source_total_counts); out << ",\n";
    out << "    \"hctx_e_trailer_total_counts\": "; write_vector_json(out, trial_hctx_e_trailer_total_counts); out << ",\n";
    out << "    \"hpred_e_trailer_total_counts\": "; write_vector_json(out, trial_hpred_e_trailer_total_counts); out << ",\n";
    out << "    \"v1_e_trailer_channel_counts\": "; write_fixed_matrix_json(out, trial_v1_e_trailer_channel_counts); out << ",\n";
    out << "    \"v1_som_trailer_channel_counts\": "; write_fixed_matrix_json(out, trial_v1_som_trailer_channel_counts); out << "\n";
    out << "  },\n";
    out << "  \"tuning_summary\": {\n";
    out << "    \"v1_e_channel_rates_hz_by_stimulus\": "; write_fixed_matrix_json(out, std::vector<std::array<double, kV1Channels>>(mean_v1_e_channel_rates_hz.begin(), mean_v1_e_channel_rates_hz.end())); out << ",\n";
    out << "    \"v1_som_channel_rates_hz_by_stimulus\": "; write_fixed_matrix_json(out, std::vector<std::array<double, kV1Channels>>(mean_v1_som_channel_rates_hz.begin(), mean_v1_som_channel_rates_hz.end())); out << ",\n";
    out << "    \"hctx_e_trailer_population_rate_hz_by_stimulus\": "; write_vector_json(out, std::vector<double>(mean_hctx_e_trailer_population_rate_hz.begin(), mean_hctx_e_trailer_population_rate_hz.end())); out << ",\n";
    out << "    \"hpred_e_trailer_population_rate_hz_by_stimulus\": "; write_vector_json(out, std::vector<double>(mean_hpred_e_trailer_population_rate_hz.begin(), mean_hpred_e_trailer_population_rate_hz.end())); out << ",\n";
    out << "    \"source_events_per_trial_by_stimulus\": "; write_vector_json(out, std::vector<double>(mean_source_events_per_trial.begin(), mean_source_events_per_trial.end())); out << "\n";
    out << "  },\n";
    out << "  \"device\": {\"backend_info\": \"" << json_escape(expectation_snn_cuda::backend_info()) << "\"},\n";
    const auto write_t1 = std::chrono::steady_clock::now();
    const double run_wall = std::chrono::duration<double>(run_t1 - run_t0).count();
    const double write_wall = std::chrono::duration<double>(write_t1 - write_t0).count();
    const double total_wall =
        std::chrono::duration<double>(write_t1 - total_t0).count();
    out << "  \"timing_seconds\": {\"run\": " << run_wall << ", \"write\": "
        << write_wall << ", \"total_excluding_build\": " << total_wall << "}\n";
    out << "}\n";

    std::cout << std::setprecision(12)
              << "command=sensory-diagnostics\n"
              << "status=PASS\n"
              << "seed=" << args.seed << "\n"
              << "repeats=" << args.repeats << "\n"
              << "checkpoint_path=" << checkpoint.json_path << "\n"
              << "result_path=" << args.out_path << "\n"
              << "content_hash_fnv1a64=" << result_hash << "\n"
              << "wall_run_seconds=" << run_wall << "\n"
              << "wall_total_excluding_build_seconds=" << total_wall << "\n";
    return 0;
}

int device_info() {
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    std::cout << "command=device-info\n"
              << "backend_info=\"" << expectation_snn_cuda::backend_info() << "\"\n"
              << "cuda_get_device_count_status=" << static_cast<int>(status) << "\n"
              << "cuda_runtime_version=" << runtime_version << "\n"
              << "cuda_driver_version=" << driver_version << "\n";
    if (status != cudaSuccess) {
        std::cout << "cuda_error=\"" << cudaGetErrorString(status) << "\"\n";
        return 1;
    }
    std::cout << "device_count=" << device_count << "\n";
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};
        const cudaError_t prop_status = cudaGetDeviceProperties(&prop, device);
        if (prop_status != cudaSuccess) {
            std::cout << "device." << device << ".error=\""
                      << cudaGetErrorString(prop_status) << "\"\n";
            continue;
        }
        std::cout << "device." << device << ".name=\"" << prop.name << "\"\n"
                  << "device." << device << ".compute_capability="
                  << prop.major << "." << prop.minor << "\n"
                  << "device." << device << ".global_mem_bytes="
                  << static_cast<unsigned long long>(prop.totalGlobalMem) << "\n"
                  << "device." << device << ".multiprocessor_count="
                  << prop.multiProcessorCount << "\n";
    }
    return device_count > 0 ? 0 : 1;
}

int validate_fixture(const Args& args) {
    const auto v1_decay = expectation_snn_cuda::run_decay_primitive("v1_e", 32, false);
    require_close_map("v1_e_decay", v1_decay.max_abs_error, kTol);
    require_equal_counts("v1_e_decay", v1_decay.cpu_spike_counts, v1_decay.cuda_spike_counts);

    const auto h_decay = expectation_snn_cuda::run_decay_primitive("h_e", 32, true);
    require_close_map("h_e_decay_threshold", h_decay.max_abs_error, kTol);
    require_equal_counts("h_e_decay_threshold", h_decay.cpu_spike_counts, h_decay.cuda_spike_counts);

    const auto h_ring = expectation_snn_cuda::run_h_ring_dynamics_test(args.seed);
    require_close_map("h_ring_dynamics", h_ring.max_abs_error, kTol);
    require_equal_counts("h_ring_ctx_total", h_ring.cpu_ctx_total_counts, h_ring.cuda_ctx_total_counts);
    require_equal_counts("h_ring_pred_total", h_ring.cpu_pred_total_counts, h_ring.cuda_pred_total_counts);
    if (h_ring.metrics.at("ctx_persistence_window_pass") != 1.0) {
        throw std::runtime_error("h_ring ctx persistence gate failed");
    }
    if (h_ring.metrics.at("no_runaway_pass") != 1.0) {
        throw std::runtime_error("h_ring no-runaway gate failed");
    }

    const auto trainer = expectation_snn_cuda::run_ctx_pred_tiny_trainer_test(
        args.seed,
        args.schedule_variant
    );
    require_close_map("ctx_pred_tiny_trainer", trainer.max_abs_error, kTol);
    if (trainer.cpu_w_ctx_pred_final.size() != 36864) {
        throw std::runtime_error("ctx_pred tiny trainer W size is not 36864");
    }
    if (trainer.cpu_gate_dw_sum.empty()) {
        throw std::runtime_error("ctx_pred tiny trainer produced empty gate telemetry");
    }

    double delta_abs_sum = 0.0;
    for (std::size_t i = 0; i < trainer.cpu_w_ctx_pred_final.size(); ++i) {
        delta_abs_sum += std::abs(
            trainer.cpu_w_ctx_pred_final[i] - trainer.initial_w_ctx_pred[i]
        );
    }

    std::cout << "command=validate-fixture\n"
              << "status=PASS\n"
              << "seed=" << args.seed << "\n"
              << "v1_e_decay_steps=" << v1_decay.n_steps << "\n"
              << "h_e_decay_steps=" << h_decay.n_steps << "\n"
              << "h_ring_ctx_persistence_ms=" << h_ring.metrics.at("ctx_persistence_ms") << "\n"
              << "h_ring_max_rate_hz=" << h_ring.metrics.at("max_rate_hz") << "\n"
              << "ctx_pred_tiny_trials=" << trainer.n_trials << "\n"
              << "ctx_pred_tiny_delta_abs_sum=" << delta_abs_sum << "\n";
    return 0;
}

int bench(const Args& args) {
    if (args.n_steps < 1 || args.repeats < 1) {
        throw std::runtime_error("bench requires --steps >= 1 and --repeats >= 1");
    }
    double max_error = 0.0;
    std::int64_t total_spikes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < args.repeats; ++r) {
        const auto result = expectation_snn_cuda::run_decay_primitive(
            args.population,
            args.n_steps,
            args.threshold_case
        );
        for (const auto& [_, value] : result.max_abs_error) {
            max_error = std::max(max_error, value);
        }
        total_spikes += result.cuda_total_spikes;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "command=bench\n"
              << "bench_kind=native_end_to_end_cpu_cuda_parity\n"
              << "population=" << args.population << "\n"
              << "steps=" << args.n_steps << "\n"
              << "repeats=" << args.repeats << "\n"
              << "threshold_case=" << (args.threshold_case ? 1 : 0) << "\n"
              << "wall_seconds=" << wall_seconds << "\n"
              << "max_abs_error=" << max_error << "\n"
              << "cuda_total_spikes_accum=" << total_spikes << "\n";
    return max_error <= kTol ? 0 : 1;
}

int stage1_train_tiny(const Args& args) {
    if (args.n_trials != 5) {
        std::cerr << "stage1-train --fixture tiny has fixed n_trials=5; requested "
                  << args.n_trials << ".\n";
        return 2;
    }
    const auto result = expectation_snn_cuda::run_ctx_pred_tiny_trainer_test(
        args.seed,
        args.schedule_variant
    );
    require_close_map("stage1_train_tiny", result.max_abs_error, kTol);
    double delta_abs_sum = 0.0;
    for (std::size_t i = 0; i < result.cpu_w_ctx_pred_final.size(); ++i) {
        delta_abs_sum += std::abs(
            result.cpu_w_ctx_pred_final[i] - result.initial_w_ctx_pred[i]
        );
    }
    const double row_sum_max = result.cpu_row_sums.empty()
        ? 0.0
        : *std::max_element(result.cpu_row_sums.begin(), result.cpu_row_sums.end());

    std::cout << "command=stage1-train\n"
              << "status=PASS\n"
              << "fixture=tiny\n"
              << "seed=" << args.seed << "\n"
              << "n_trials=" << result.n_trials << "\n"
              << "W_ctx_pred_len=" << result.cpu_w_ctx_pred_final.size() << "\n"
              << "delta_abs_sum=" << delta_abs_sum << "\n"
              << "row_sum_max=" << row_sum_max << "\n";
    return 0;
}

int stage1_train_generated(const Args& args) {
    if (args.out_path.empty()) {
        throw std::runtime_error("stage1-train --fixture generated requires --out PATH");
    }
    const auto total_t0 = std::chrono::steady_clock::now();
    if (args.stage1_prediction_target != "orientation_cell"
        && args.stage1_prediction_target != "v1_template") {
        throw std::runtime_error(
            "stage1-train --stage1-prediction-target must be orientation_cell or v1_template"
        );
    }
    const auto schedule_t0 = std::chrono::steady_clock::now();
    const GeneratedSchedule schedule = build_generated_schedule(args.seed, args.n_trials);
    const auto schedule_t1 = std::chrono::steady_clock::now();

    const auto train_t0 = std::chrono::steady_clock::now();
    std::vector<std::int32_t> expected_trailer_cells;
    expected_trailer_cells.reserve(schedule.expected_trailer_idx.size());
    for (std::size_t trial = 0; trial < schedule.expected_trailer_idx.size(); ++trial) {
        const int expected_channel = schedule.expected_trailer_idx[trial] * 2;
        const int expected_cell_offset =
            schedule.expected_trailer_idx[trial] % kHCellsPerChannel;
        expected_trailer_cells.push_back(
            expected_channel * kHCellsPerChannel + expected_cell_offset
        );
    }
    auto train = expectation_snn_cuda::run_native_stage1_generated_train(
        args.seed,
        schedule.leader_cells,
        expected_trailer_cells,
        args.stage1_prediction_target
    );
    if (args.stage1_learn_feedback_direct) {
        train.w_hpred_v1direct_final = build_learned_hpred_v1direct_weights(
            schedule,
            args.stage1_prediction_target
        );
        train.w_hpred_v1direct_row_sums =
            hpred_v1direct_row_sums(train.w_hpred_v1direct_final);
    }
    if (args.stage1_learn_feedback_som) {
        train.w_hpred_v1som_final = build_learned_hpred_v1som_weights(
            schedule,
            args.stage1_prediction_target
        );
        train.w_hpred_v1som_row_sums =
            hpred_v1som_row_sums(train.w_hpred_v1som_final);
    }
    const auto train_t1 = std::chrono::steady_clock::now();

    const auto gate_t0 = std::chrono::steady_clock::now();
    const auto gates = expectation_snn_cuda::run_stage1_h_gate_dynamics_test(
        args.seed,
        schedule.leader_cells,
        schedule.trailer_cells,
        train.w_ctx_pred_final
    );
    const auto gate_t1 = std::chrono::steady_clock::now();

    double max_gate_error = 0.0;
    for (const auto& [_, value] : gates.max_abs_error) {
        max_gate_error = std::max(max_gate_error, value);
    }
    if (max_gate_error > kTol) {
        throw std::runtime_error("Stage1 H gate CPU/CUDA mismatch " + std::to_string(max_gate_error));
    }

    const double persistence = gates.metrics.at("h_context_persistence_ms");
    const double forecast = gates.metrics.at("h_prediction_pretrailer_forecast_probability");
    const double no_runaway = gates.metrics.at("no_runaway_max_rate_hz");
    const double max_cell = gates.metrics.at("no_runaway_max_cell_rate_hz");
    const bool thresholds_pass = persistence >= 200.0 && persistence <= 500.0
        && forecast >= 0.25 && no_runaway <= 80.0;

    const std::filesystem::path json_path = args.out_path;
    const std::filesystem::path w_path = std::filesystem::path(args.out_path.string() + ".W_ctx_pred.f64.bin");
    const std::filesystem::path hpred_v1direct_path =
        std::filesystem::path(args.out_path.string() + ".W_hpred_v1direct.f64.bin");
    const std::filesystem::path hpred_v1som_path =
        std::filesystem::path(args.out_path.string() + ".W_hpred_v1som.f64.bin");
    const std::string content_hash = hash_hex(artifact_hash(schedule, train));

    const auto write_t0 = std::chrono::steady_clock::now();
    write_binary_vector(w_path, train.w_ctx_pred_final);
    if (!train.w_hpred_v1direct_final.empty()) {
        write_binary_vector(hpred_v1direct_path, train.w_hpred_v1direct_final);
    }
    if (!train.w_hpred_v1som_final.empty()) {
        write_binary_vector(hpred_v1som_path, train.w_hpred_v1som_final);
    }
    const auto write_mid = std::chrono::steady_clock::now();
    const double schedule_wall = std::chrono::duration<double>(schedule_t1 - schedule_t0).count();
    const double train_wall = std::chrono::duration<double>(train_t1 - train_t0).count();
    const double gate_wall = std::chrono::duration<double>(gate_t1 - gate_t0).count();
    const double write_binary_wall = std::chrono::duration<double>(write_mid - write_t0).count();
    write_stage1_json(
        json_path,
        w_path,
        hpred_v1direct_path,
        hpred_v1som_path,
        schedule,
        train,
        gates,
        schedule_wall,
        train_wall,
        gate_wall,
        write_binary_wall,
        content_hash
    );
    const auto write_t1 = std::chrono::steady_clock::now();
    const double write_wall = std::chrono::duration<double>(write_t1 - write_t0).count();
    const double total_wall = std::chrono::duration<double>(write_t1 - total_t0).count();

    double delta_abs_sum = 0.0;
    for (const double value : train.w_ctx_pred_final) {
        delta_abs_sum += std::abs(value);
    }
    const double row_sum_max = train.row_sums.empty()
        ? 0.0
        : *std::max_element(train.row_sums.begin(), train.row_sums.end());

    std::cout << std::setprecision(12)
              << "command=stage1-train\n"
              << "status=" << (thresholds_pass ? "PASS" : "FAIL") << "\n"
              << "fixture=generated\n"
              << "seed=" << args.seed << "\n"
              << "n_trials=" << args.n_trials << "\n"
              << "stage1_prediction_target=" << args.stage1_prediction_target << "\n"
              << "stage1_learn_feedback_direct=" << (args.stage1_learn_feedback_direct ? 1 : 0) << "\n"
              << "stage1_learn_feedback_som=" << (args.stage1_learn_feedback_som ? 1 : 0) << "\n"
              << "json_path=" << json_path << "\n"
              << "W_ctx_pred_path=" << w_path << "\n"
              << "W_hpred_v1direct_path="
              << (train.w_hpred_v1direct_final.empty() ? std::filesystem::path("") : hpred_v1direct_path) << "\n"
              << "W_hpred_v1som_path="
              << (train.w_hpred_v1som_final.empty() ? std::filesystem::path("") : hpred_v1som_path) << "\n"
              << "content_hash_fnv1a64=" << content_hash << "\n"
              << "h_context_persistence_ms=" << persistence << "\n"
              << "forecast_probability=" << forecast << "\n"
              << "no_runaway_population_rate_hz=" << no_runaway << "\n"
              << "max_cell_rate_hz_diagnostic=" << max_cell << "\n"
              << "delta_abs_sum=" << delta_abs_sum << "\n"
              << "row_sum_max=" << row_sum_max << "\n"
              << "max_gate_cpu_cuda_error=" << max_gate_error << "\n"
              << "wall_schedule_seconds=" << schedule_wall << "\n"
              << "wall_train_seconds=" << train_wall << "\n"
              << "wall_gate_eval_seconds=" << gate_wall << "\n"
              << "wall_write_seconds=" << write_wall << "\n"
              << "wall_total_excluding_build_seconds=" << total_wall << "\n";
    return thresholds_pass ? 0 : 3;
}

int stage1_train(const Args& args) {
    if (args.fixture == "tiny") {
        return stage1_train_tiny(args);
    }
    if (args.fixture == "generated") {
        return stage1_train_generated(args);
    }
    throw std::runtime_error("unknown stage1-train fixture: " + args.fixture);
}

int stage1_heldout_eval(const Args& args) {
    if (args.checkpoint_path.empty()) {
        throw std::runtime_error("stage1-heldout-eval requires --checkpoint PATH");
    }
    if (args.out_path.empty()) {
        throw std::runtime_error("stage1-heldout-eval requires --out PATH");
    }
    const auto total_t0 = std::chrono::steady_clock::now();
    std::vector<double> w_ctx_pred;
    const Stage1CheckpointMeta checkpoint = validate_stage1_checkpoint(
        args.checkpoint_path,
        w_ctx_pred
    );
    if (args.seed == checkpoint.seed) {
        throw std::runtime_error(
            "stage1-heldout-eval requires --seed different from checkpoint seed"
        );
    }
    const GeneratedSchedule schedule = build_heldout_schedule(
        args.seed,
        checkpoint.n_trials,
        args.heldout_schedule
    );
    const auto eval_t0 = std::chrono::steady_clock::now();
    const auto gates = expectation_snn_cuda::run_stage1_h_gate_dynamics_test(
        args.seed,
        schedule.leader_cells,
        schedule.trailer_cells,
        w_ctx_pred
    );
    const auto eval_t1 = std::chrono::steady_clock::now();

    double max_gate_error = 0.0;
    for (const auto& [_, value] : gates.max_abs_error) {
        max_gate_error = std::max(max_gate_error, value);
    }
    if (max_gate_error > kTol) {
        throw std::runtime_error(
            "stage1-heldout-eval CPU/CUDA mismatch " + std::to_string(max_gate_error)
        );
    }

    const double eval_wall = std::chrono::duration<double>(eval_t1 - eval_t0).count();
    const std::string content_hash = hash_hex(
        stage1_heldout_eval_hash(checkpoint, schedule, gates)
    );
    write_stage1_heldout_eval_json(
        args.out_path,
        checkpoint,
        schedule,
        gates,
        eval_wall,
        content_hash
    );
    const auto total_t1 = std::chrono::steady_clock::now();
    const double total_wall = std::chrono::duration<double>(total_t1 - total_t0).count();

    std::cout << std::setprecision(12)
              << "command=stage1-heldout-eval\n"
              << "status=PASS\n"
              << "checkpoint_path=" << checkpoint.json_path << "\n"
              << "heldout_seed=" << args.seed << "\n"
              << "n_trials=" << checkpoint.n_trials << "\n"
              << "json_path=" << args.out_path << "\n"
              << "forecast_probability="
              << gates.metrics.at("h_prediction_pretrailer_forecast_probability") << "\n"
              << "h_context_persistence_ms="
              << gates.metrics.at("h_context_persistence_ms") << "\n"
              << "no_runaway_population_rate_hz="
              << gates.metrics.at("no_runaway_max_rate_hz") << "\n"
              << "max_gate_cpu_cuda_error=" << max_gate_error << "\n"
              << "content_hash_fnv1a64=" << content_hash << "\n"
              << "wall_eval_seconds=" << eval_wall << "\n"
              << "wall_total_excluding_build_seconds=" << total_wall << "\n";
    return 0;
}


int richter_dampening(const Args& args) {
    if (args.checkpoint_path.empty()) {
        throw std::runtime_error("richter-dampening requires --checkpoint PATH");
    }
    if (args.out_path.empty()) {
        throw std::runtime_error("richter-dampening requires --out PATH");
    }
    if (args.reps_expected < 1 || args.reps_unexpected < 1) {
        throw std::runtime_error("richter-dampening requires positive repetitions");
    }
    if (args.execution_mode != "gpu_only_production"
        && args.execution_mode != "cpu_reference") {
        throw std::runtime_error(
            "richter-dampening --execution-mode must be gpu_only_production or cpu_reference"
        );
    }
    if (args.v1_stim_sigma_deg <= 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-stim-sigma-deg must be positive"
        );
    }
    if (args.feedback_replay_mode != "raw"
        && args.feedback_replay_mode != "normalized") {
        throw std::runtime_error(
            "richter-dampening --feedback-replay-mode must be raw or normalized"
        );
    }
    if (args.feedback_replay_fallback != "none"
        && args.feedback_replay_fallback != "leader"
        && args.feedback_replay_fallback != "flat") {
        throw std::runtime_error(
            "richter-dampening --feedback-replay-fallback must be none, leader, or flat"
        );
    }
    if (args.feedback_replay_fallback != "none"
        && args.feedback_replay_mode != "normalized") {
        throw std::runtime_error(
            "richter-dampening --feedback-replay-fallback is only valid with normalized replay"
        );
    }
    if (args.feedback_som_center_weight < 0.0) {
        throw std::runtime_error(
            "richter-dampening --feedback-som-center-weight must be nonnegative"
        );
    }
    if (args.feedback_direct_source != "fixed"
        && args.feedback_direct_source != "learned"
        && args.feedback_direct_source != "learned-shifted"
        && args.feedback_direct_source != "disabled") {
        throw std::runtime_error(
            "richter-dampening --feedback-direct-source must be fixed, learned, learned-shifted, or disabled"
        );
    }
    if (args.feedback_som_source != "fixed"
        && args.feedback_som_source != "learned"
        && args.feedback_som_source != "learned-shifted"
        && args.feedback_som_source != "disabled") {
        throw std::runtime_error(
            "richter-dampening --feedback-som-source must be fixed, learned, learned-shifted, or disabled"
        );
    }
    if (args.v1_som_to_e_scale < 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-som-to-e-scale must be nonnegative"
        );
    }
    if (args.v1_som_divisive_scale < 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-som-divisive-scale must be nonnegative"
        );
    }
    if (args.v1_direct_divisive_scale < 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-direct-divisive-scale must be nonnegative"
        );
    }
    if (args.v1_feedforward_divisive_scale < 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-feedforward-divisive-scale must be nonnegative"
        );
    }
    if (args.v1_predicted_suppression_scale < 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-predicted-suppression-scale must be nonnegative"
        );
    }
    if (args.v1_predicted_suppression_neighbor_weight < 0.0) {
        throw std::runtime_error(
            "richter-dampening --v1-predicted-suppression-neighbor-weight must be nonnegative"
        );
    }
    if (args.v1_error_sensory_gain < 0.0
        || args.v1_error_prediction_gain < 0.0) {
        throw std::runtime_error(
            "richter-dampening V1_ERROR comparator gains must be nonnegative"
        );
    }
    const std::int32_t v1_predicted_suppression_locus =
        v1_predicted_suppression_locus_id(
            args.v1_predicted_suppression_locus
        );
    const std::int32_t v1_error_comparator_mode =
        v1_error_comparator_mode_id(args.v1_error_comparator_mode);
    const std::int32_t v1_feedforward_divisive_gate_source =
        v1_feedforward_divisive_gate_source_id(
            args.v1_feedforward_divisive_gate_source
        );
    const std::int32_t v1_direct_divisive_gate_source =
        v1_direct_divisive_gate_source_id(
            args.v1_direct_divisive_gate_source
        );
    if (args.feedback_replay_target_per_bin < 0.0) {
        throw std::runtime_error(
            "richter-dampening --feedback-replay-target-per-100ms-bin must be nonnegative"
        );
    }
    const auto total_t0 = std::chrono::steady_clock::now();

    std::vector<double> w_ctx_pred;
    std::vector<double> w_hpred_v1direct;
    std::vector<double> w_hpred_v1som;
    const Stage1CheckpointMeta checkpoint = validate_stage1_checkpoint(
        args.checkpoint_path,
        w_ctx_pred,
        &w_hpred_v1direct,
        &w_hpred_v1som
    );
    if ((args.feedback_direct_source == "learned"
         || args.feedback_direct_source == "learned-shifted")
        && !checkpoint.has_learned_hpred_v1direct) {
        throw std::runtime_error(
            "richter-dampening learned feedback direct source requires checkpointed W_hpred_v1direct"
        );
    }
    if ((args.feedback_som_source == "learned"
         || args.feedback_som_source == "learned-shifted")
        && !checkpoint.has_learned_hpred_v1som) {
        throw std::runtime_error(
            "richter-dampening learned feedback SOM source requires checkpointed W_hpred_v1som"
        );
    }
    const FeedbackBalance feedback = resolve_feedback_balance(
        args.feedback_g_total,
        args.feedback_r
    );
    const NativeTopology topology = make_native_topology(
        w_ctx_pred,
        feedback,
        args.feedback_som_center_weight,
        w_hpred_v1direct,
        args.feedback_direct_source,
        w_hpred_v1som,
        args.feedback_som_source
    );
    const std::size_t expected_fb_direct_edges =
        (args.feedback_direct_source == "fixed")
        ? 3072u
        : ((args.feedback_direct_source == "disabled") ? 1u : 36864u);
    const std::size_t expected_fb_som_edges =
        (args.feedback_som_source == "fixed")
        ? 3072u + (args.feedback_som_center_weight > 0.0 ? 768u : 0u)
        : ((args.feedback_som_source == "disabled") ? 1u : 9216u);
    if (topology.stim_pre.size() != 3840 || topology.v1_to_h_pre.size() != 21504
        || topology.ctx_to_pred_pre.size() != 36864 || topology.fb_direct_pre.size() != expected_fb_direct_edges
        || topology.fb_som_pre.size() != expected_fb_som_edges) {
        throw std::runtime_error("native Richter topology edge count invariant failed");
    }

    ConditionAccum expected;
    ConditionAccum unexpected;
    std::int32_t expected_trials = 0;
    std::int32_t unexpected_trials = 0;
    std::uint64_t semantic_hash = 1469598103934665603ull;
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.seed);
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.reps_expected);
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.reps_unexpected);
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.v1_stim_sigma_deg);
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.feedback_g_total);
    semantic_hash = fnv1a_update_scalar(semantic_hash, args.feedback_r);
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.feedback_som_center_weight
    );
    semantic_hash = fnv1a_update_string(semantic_hash, args.feedback_direct_source);
    semantic_hash = fnv1a_update_string(semantic_hash, args.feedback_som_source);
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_som_to_e_scale
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_som_divisive_scale
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_direct_divisive_scale
    );
    semantic_hash = fnv1a_update(
        semantic_hash,
        args.v1_direct_divisive_gate_source.data(),
        args.v1_direct_divisive_gate_source.size()
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_feedforward_divisive_scale
    );
    semantic_hash = fnv1a_update(
        semantic_hash,
        args.v1_feedforward_divisive_gate_source.data(),
        args.v1_feedforward_divisive_gate_source.size()
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_predicted_suppression_scale
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_predicted_suppression_neighbor_weight
    );
    semantic_hash = fnv1a_update(
        semantic_hash,
        args.v1_predicted_suppression_locus.data(),
        args.v1_predicted_suppression_locus.size()
    );
    semantic_hash = fnv1a_update(
        semantic_hash,
        args.v1_error_comparator_mode.data(),
        args.v1_error_comparator_mode.size()
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_error_sensory_gain
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_error_prediction_gain
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.v1_error_prediction_shift
    );
    semantic_hash = fnv1a_update_scalar(semantic_hash, feedback.g_direct);
    semantic_hash = fnv1a_update_scalar(semantic_hash, feedback.g_som);
    semantic_hash = fnv1a_update(
        semantic_hash,
        args.feedback_replay_mode.data(),
        args.feedback_replay_mode.size()
    );
    semantic_hash = fnv1a_update(
        semantic_hash,
        args.feedback_replay_fallback.data(),
        args.feedback_replay_fallback.size()
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        args.feedback_replay_target_per_bin
    );
    semantic_hash = fnv1a_update(semantic_hash, checkpoint.content_hash.data(), checkpoint.content_hash.size());
    semantic_hash = fnv1a_update(semantic_hash, checkpoint.w_hash.data(), checkpoint.w_hash.size());
    semantic_hash = fnv1a_update(
        semantic_hash,
        checkpoint.learned_hpred_v1direct_hash.data(),
        checkpoint.learned_hpred_v1direct_hash.size()
    );
    semantic_hash = fnv1a_update(
        semantic_hash,
        checkpoint.learned_hpred_v1som_hash.data(),
        checkpoint.learned_hpred_v1som_hash.size()
    );
    std::vector<std::array<std::int32_t, kV1Channels>> trial_v1_trailer_channel_counts;
    std::vector<std::array<std::int32_t, kV1Channels>>
        trial_v1_som_trailer_channel_counts;
    std::vector<std::array<std::int32_t, kV1Channels>>
        trial_v1_error_trailer_channel_counts;
    std::vector<std::array<std::int32_t, kV1Channels>>
        trial_v1_error_neg_trailer_channel_counts;
    std::vector<std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>>
        trial_v1_trailer_100ms_channel_counts;
    std::vector<std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>>
        trial_v1_som_trailer_100ms_channel_counts;
    std::vector<std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>>
        trial_v1_error_trailer_100ms_channel_counts;
    std::vector<std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>>
        trial_v1_error_neg_trailer_100ms_channel_counts;
    std::vector<std::int32_t> trial_index;
    std::vector<std::int32_t> trial_schedule_index;
    std::vector<std::int32_t> trial_is_expected;
    std::vector<std::int32_t> trial_leader_orientation;
    std::vector<std::int32_t> trial_trailer_orientation;
    std::vector<std::int32_t> trial_leader_channel;
    std::vector<std::int32_t> trial_trailer_channel;
    std::vector<std::int32_t> trial_dtheta_step;
    std::vector<std::int32_t> trial_hctx_e_preprobe_total_counts;
    std::vector<std::int32_t> trial_hctx_e_trailer_total_counts;
    std::vector<std::int32_t> trial_hpred_e_preprobe_total_counts;
    std::vector<std::array<std::int32_t, kV1Channels>>
        trial_hpred_e_preprobe_channel_counts;
    std::vector<std::int32_t> trial_hpred_e_trailer_total_counts;
    std::vector<std::array<std::int32_t, kTrailerBinCount>>
        trial_hpred_e_trailer_100ms_total_counts;
    std::vector<std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>>
        trial_hpred_e_trailer_100ms_channel_counts;
    std::vector<std::array<std::int32_t, kTrailerBinCount>>
        trial_hpred_feedback_held_trailer_100ms_total_counts;
    std::vector<std::array<std::array<std::int32_t, kV1Channels>, kTrailerBinCount>>
        trial_hpred_feedback_held_trailer_100ms_channel_counts;
    std::vector<std::array<double, kTrailerBinCount>>
        trial_hpred_feedback_normalized_trailer_100ms_total_weights;
    std::vector<std::array<std::array<double, kV1Channels>, kTrailerBinCount>>
        trial_hpred_feedback_normalized_trailer_100ms_channel_weights;
    std::vector<std::int32_t>
        trial_hpred_feedback_normalized_preprobe_zero;
    std::vector<std::int32_t>
        trial_hpred_feedback_normalized_fallback_used;
    std::vector<std::int32_t>
        trial_hpred_feedback_normalized_fallback_zero_template;
    std::vector<std::array<double, kV1Channels>>
        trial_v1_predicted_suppression_trailer_channel_signal_sum;
    std::vector<std::array<double, kV1Channels>>
        trial_v1_predicted_suppression_trailer_channel_gain_mean;
    std::vector<std::array<double, kV1Channels>>
        trial_v1_predicted_suppression_trailer_raw_ie_before_sum;
    std::vector<std::array<double, kV1Channels>>
        trial_v1_predicted_suppression_trailer_raw_ie_after_sum;
    std::vector<std::array<double, kV1Channels>>
        trial_v1_predicted_suppression_trailer_raw_ie_delta_sum;
    std::vector<double> trial_v1e_trailer_q_active_fC;
    std::vector<double> trial_v1som_trailer_q_active_fC;
    std::vector<double> trial_v1error_trailer_q_active_fC;
    std::vector<double> trial_v1error_neg_trailer_q_active_fC;
    std::vector<double> trial_v1error_total_trailer_q_active_fC;
    std::vector<double> trial_v1error_signed_trailer_q_active_fC;
    std::vector<double> trial_v1_trailer_q_active_fC;
    const std::size_t total_trials_capacity =
        static_cast<std::size_t>(args.reps_expected) * 6
        + static_cast<std::size_t>(args.reps_unexpected) * 24;
    trial_v1_trailer_channel_counts.reserve(total_trials_capacity);
    trial_v1_som_trailer_channel_counts.reserve(total_trials_capacity);
    trial_v1_error_trailer_channel_counts.reserve(total_trials_capacity);
    trial_v1_error_neg_trailer_channel_counts.reserve(total_trials_capacity);
    trial_v1_trailer_100ms_channel_counts.reserve(total_trials_capacity);
    trial_v1_som_trailer_100ms_channel_counts.reserve(total_trials_capacity);
    trial_v1_error_trailer_100ms_channel_counts.reserve(total_trials_capacity);
    trial_v1_error_neg_trailer_100ms_channel_counts.reserve(total_trials_capacity);
    trial_index.reserve(total_trials_capacity);
    trial_schedule_index.reserve(total_trials_capacity);
    trial_is_expected.reserve(total_trials_capacity);
    trial_leader_orientation.reserve(total_trials_capacity);
    trial_trailer_orientation.reserve(total_trials_capacity);
    trial_leader_channel.reserve(total_trials_capacity);
    trial_trailer_channel.reserve(total_trials_capacity);
    trial_dtheta_step.reserve(total_trials_capacity);
    trial_hctx_e_preprobe_total_counts.reserve(total_trials_capacity);
    trial_hctx_e_trailer_total_counts.reserve(total_trials_capacity);
    trial_hpred_e_preprobe_total_counts.reserve(total_trials_capacity);
    trial_hpred_e_preprobe_channel_counts.reserve(total_trials_capacity);
    trial_hpred_e_trailer_total_counts.reserve(total_trials_capacity);
    trial_hpred_e_trailer_100ms_total_counts.reserve(total_trials_capacity);
    trial_hpred_e_trailer_100ms_channel_counts.reserve(total_trials_capacity);
    trial_hpred_feedback_held_trailer_100ms_total_counts.reserve(
        total_trials_capacity
    );
    trial_hpred_feedback_held_trailer_100ms_channel_counts.reserve(
        total_trials_capacity
    );
    trial_hpred_feedback_normalized_trailer_100ms_total_weights.reserve(
        total_trials_capacity
    );
    trial_hpred_feedback_normalized_trailer_100ms_channel_weights.reserve(
        total_trials_capacity
    );
    trial_hpred_feedback_normalized_preprobe_zero.reserve(
        total_trials_capacity
    );
    trial_hpred_feedback_normalized_fallback_used.reserve(
        total_trials_capacity
    );
    trial_hpred_feedback_normalized_fallback_zero_template.reserve(
        total_trials_capacity
    );
    trial_v1_predicted_suppression_trailer_channel_signal_sum.reserve(
        total_trials_capacity
    );
    trial_v1_predicted_suppression_trailer_channel_gain_mean.reserve(
        total_trials_capacity
    );
    trial_v1_predicted_suppression_trailer_raw_ie_before_sum.reserve(
        total_trials_capacity
    );
    trial_v1_predicted_suppression_trailer_raw_ie_after_sum.reserve(
        total_trials_capacity
    );
    trial_v1_predicted_suppression_trailer_raw_ie_delta_sum.reserve(
        total_trials_capacity
    );
    trial_v1e_trailer_q_active_fC.reserve(total_trials_capacity);
    trial_v1som_trailer_q_active_fC.reserve(total_trials_capacity);
    trial_v1error_trailer_q_active_fC.reserve(total_trials_capacity);
    trial_v1error_neg_trailer_q_active_fC.reserve(total_trials_capacity);
    trial_v1error_total_trailer_q_active_fC.reserve(total_trials_capacity);
    trial_v1error_signed_trailer_q_active_fC.reserve(total_trials_capacity);
    trial_v1_trailer_q_active_fC.reserve(total_trials_capacity);

    const auto run_t0 = std::chrono::steady_clock::now();
    auto orientation_channel = [](int orientation) -> int {
        return (orientation * 2) % 12;
    };
    auto record_trial = [&](int group, const RichterConditionSpec& spec, const ConditionAccum& acc) {
        trial_index.push_back(static_cast<std::int32_t>(trial_index.size()));
        trial_schedule_index.push_back(spec.rep);
        trial_is_expected.push_back(spec.is_expected);
        trial_leader_orientation.push_back(spec.leader_orientation);
        trial_trailer_orientation.push_back(spec.trailer_orientation);
        trial_leader_channel.push_back(spec.leader_channel);
        trial_trailer_channel.push_back(spec.trailer_channel);
        trial_dtheta_step.push_back(spec.step);
        trial_v1_trailer_channel_counts.push_back(acc.v1_trailer_channel_counts);
        trial_v1_som_trailer_channel_counts.push_back(
            acc.v1_som_trailer_channel_counts
        );
        trial_v1_error_trailer_channel_counts.push_back(
            acc.v1_error_trailer_channel_counts
        );
        trial_v1_error_neg_trailer_channel_counts.push_back(
            acc.v1_error_neg_trailer_channel_counts
        );
        trial_v1_trailer_100ms_channel_counts.push_back(
            acc.v1_trailer_bin_channel_counts
        );
        trial_v1_som_trailer_100ms_channel_counts.push_back(
            acc.v1_som_trailer_bin_channel_counts
        );
        trial_v1_error_trailer_100ms_channel_counts.push_back(
            acc.v1_error_trailer_bin_channel_counts
        );
        trial_v1_error_neg_trailer_100ms_channel_counts.push_back(
            acc.v1_error_neg_trailer_bin_channel_counts
        );
        trial_hctx_e_preprobe_total_counts.push_back(
            static_cast<std::int32_t>(acc.hctx_preprobe)
        );
        trial_hctx_e_trailer_total_counts.push_back(
            static_cast<std::int32_t>(acc.hctx_trailer)
        );
        trial_hpred_e_preprobe_total_counts.push_back(
            static_cast<std::int32_t>(acc.hpred_preprobe)
        );
        trial_hpred_e_preprobe_channel_counts.push_back(
            acc.hpred_preprobe_channel_counts
        );
        trial_hpred_e_trailer_total_counts.push_back(
            static_cast<std::int32_t>(acc.hpred_trailer)
        );
        trial_hpred_e_trailer_100ms_total_counts.push_back(
            acc.hpred_trailer_bin_total_counts
        );
        trial_hpred_e_trailer_100ms_channel_counts.push_back(
            acc.hpred_trailer_bin_channel_counts
        );
        trial_hpred_feedback_held_trailer_100ms_total_counts.push_back(
            acc.hpred_feedback_held_trailer_bin_total_counts
        );
        trial_hpred_feedback_held_trailer_100ms_channel_counts.push_back(
            acc.hpred_feedback_held_trailer_bin_channel_counts
        );
        trial_hpred_feedback_normalized_trailer_100ms_total_weights.push_back(
            acc.hpred_feedback_normalized_trailer_bin_total_weights
        );
        trial_hpred_feedback_normalized_trailer_100ms_channel_weights.push_back(
            acc.hpred_feedback_normalized_trailer_bin_channel_weights
        );
        trial_hpred_feedback_normalized_preprobe_zero.push_back(
            acc.hpred_feedback_normalized_preprobe_zero
        );
        trial_hpred_feedback_normalized_fallback_used.push_back(
            acc.hpred_feedback_normalized_fallback_used
        );
        trial_hpred_feedback_normalized_fallback_zero_template.push_back(
            acc.hpred_feedback_normalized_fallback_zero_template
        );
        trial_v1_predicted_suppression_trailer_channel_signal_sum.push_back(
            acc.v1_predicted_suppression_trailer_channel_signal_sum
        );
        std::array<double, kV1Channels> predicted_suppression_gain_mean{};
        for (int channel = 0; channel < kV1Channels; ++channel) {
            predicted_suppression_gain_mean[
                static_cast<std::size_t>(channel)
            ] = args.v1_predicted_suppression_scale > 0.0
                ? acc.v1_predicted_suppression_trailer_channel_gain_sum[
                    static_cast<std::size_t>(channel)
                ] / static_cast<double>(kTrailerSteps)
                : 1.0;
        }
        trial_v1_predicted_suppression_trailer_channel_gain_mean.push_back(
            predicted_suppression_gain_mean
        );
        trial_v1_predicted_suppression_trailer_raw_ie_before_sum.push_back(
            acc.v1_predicted_suppression_trailer_raw_ie_before_sum
        );
        trial_v1_predicted_suppression_trailer_raw_ie_after_sum.push_back(
            acc.v1_predicted_suppression_trailer_raw_ie_after_sum
        );
        trial_v1_predicted_suppression_trailer_raw_ie_delta_sum.push_back(
            acc.v1_predicted_suppression_trailer_raw_ie_delta_sum
        );
        trial_v1e_trailer_q_active_fC.push_back(acc.v1e_trailer_q_active_fC);
        trial_v1som_trailer_q_active_fC.push_back(acc.v1som_trailer_q_active_fC);
        trial_v1error_trailer_q_active_fC.push_back(
            acc.v1error_trailer_q_active_fC
        );
        trial_v1error_neg_trailer_q_active_fC.push_back(
            acc.v1error_neg_trailer_q_active_fC
        );
        trial_v1error_total_trailer_q_active_fC.push_back(
            acc.v1error_trailer_q_active_fC
            + acc.v1error_neg_trailer_q_active_fC
        );
        trial_v1error_signed_trailer_q_active_fC.push_back(
            acc.v1error_trailer_q_active_fC
            - acc.v1error_neg_trailer_q_active_fC
        );
        trial_v1_trailer_q_active_fC.push_back(
            acc.v1e_trailer_q_active_fC + acc.v1som_trailer_q_active_fC
        );
        semantic_hash = fnv1a_update_scalar(semantic_hash, group);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.rep);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.is_expected);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.leader_orientation);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.trailer_orientation);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.step);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.leader_channel);
        semantic_hash = fnv1a_update_scalar(semantic_hash, spec.trailer_channel);
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_index.back()
        );
        for (const auto count : acc.v1_trailer_channel_counts) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto& bin_counts : acc.v1_trailer_bin_channel_counts) {
            for (const auto count : bin_counts) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, count);
            }
        }
        for (const auto count : acc.v1_som_trailer_channel_counts) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto& bin_counts : acc.v1_som_trailer_bin_channel_counts) {
            for (const auto count : bin_counts) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, count);
            }
        }
        for (const auto count : acc.v1_error_trailer_channel_counts) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto& bin_counts : acc.v1_error_trailer_bin_channel_counts) {
            for (const auto count : bin_counts) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, count);
            }
        }
        for (const auto count : acc.v1_error_neg_trailer_channel_counts) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto& bin_counts : acc.v1_error_neg_trailer_bin_channel_counts) {
            for (const auto count : bin_counts) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, count);
            }
        }
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hctx_e_preprobe_total_counts.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hctx_e_trailer_total_counts.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hpred_e_preprobe_total_counts.back()
        );
        for (const auto count : trial_hpred_e_preprobe_channel_counts.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hpred_e_trailer_total_counts.back()
        );
        for (const auto count : trial_hpred_e_trailer_100ms_total_counts.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto& bin_counts :
             trial_hpred_e_trailer_100ms_channel_counts.back()) {
            for (const auto count : bin_counts) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, count);
            }
        }
        for (const auto count :
             trial_hpred_feedback_held_trailer_100ms_total_counts.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, count);
        }
        for (const auto& bin_counts :
             trial_hpred_feedback_held_trailer_100ms_channel_counts.back()) {
            for (const auto count : bin_counts) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, count);
            }
        }
        for (const auto weight :
             trial_hpred_feedback_normalized_trailer_100ms_total_weights.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, weight);
        }
        for (const auto& bin_weights :
             trial_hpred_feedback_normalized_trailer_100ms_channel_weights.back()) {
            for (const auto weight : bin_weights) {
                semantic_hash = fnv1a_update_scalar(semantic_hash, weight);
            }
        }
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hpred_feedback_normalized_preprobe_zero.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hpred_feedback_normalized_fallback_used.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_hpred_feedback_normalized_fallback_zero_template.back()
        );
        for (const auto signal :
             trial_v1_predicted_suppression_trailer_channel_signal_sum.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, signal);
        }
        for (const auto gain :
             trial_v1_predicted_suppression_trailer_channel_gain_mean.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, gain);
        }
        for (const auto value :
             trial_v1_predicted_suppression_trailer_raw_ie_before_sum.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, value);
        }
        for (const auto value :
             trial_v1_predicted_suppression_trailer_raw_ie_after_sum.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, value);
        }
        for (const auto value :
             trial_v1_predicted_suppression_trailer_raw_ie_delta_sum.back()) {
            semantic_hash = fnv1a_update_scalar(semantic_hash, value);
        }
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_v1e_trailer_q_active_fC.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_v1som_trailer_q_active_fC.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_v1error_trailer_q_active_fC.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_v1error_neg_trailer_q_active_fC.back()
        );
        semantic_hash = fnv1a_update_scalar(
            semantic_hash,
            trial_v1_trailer_q_active_fC.back()
        );
        const std::int64_t trailer = static_cast<std::int64_t>(acc.v1_trailer);
        const std::int64_t source = acc.source_total;
        semantic_hash = fnv1a_update_scalar(semantic_hash, trailer);
        semantic_hash = fnv1a_update_scalar(semantic_hash, source);
    };

    std::vector<RichterConditionSpec> expected_specs;
    expected_specs.reserve(static_cast<std::size_t>(args.reps_expected) * 6);
    for (int rep = 0; rep < args.reps_expected; ++rep) {
        for (int leader = 0; leader < 6; ++leader) {
            const int step = 1;
            const int trailer = (leader + step) % 6;
            expected_specs.push_back(RichterConditionSpec{
                args.seed + 1000003LL * rep + 101LL * leader + 17LL * step,
                rep,
                1,
                leader,
                trailer,
                step,
                orientation_channel(leader),
                orientation_channel(trailer),
            });
        }
    }
    const std::array<int, 4> unexpected_steps{{2, 3, 4, 5}};
    std::vector<RichterConditionSpec> unexpected_specs;
    unexpected_specs.reserve(static_cast<std::size_t>(args.reps_unexpected) * 24);
    for (int rep = 0; rep < args.reps_unexpected; ++rep) {
        for (int step : unexpected_steps) {
            for (int leader = 0; leader < 6; ++leader) {
                const int trailer = (leader + step) % 6;
                unexpected_specs.push_back(RichterConditionSpec{
                    args.seed + 7000001LL + 1000003LL * rep + 101LL * leader
                        + 17LL * step,
                    rep,
                    0,
                    leader,
                    trailer,
                    step,
                    orientation_channel(leader),
                    orientation_channel(trailer),
                });
            }
        }
    }

    std::vector<RichterConditionSpec> all_specs;
    all_specs.reserve(expected_specs.size() + unexpected_specs.size());
    all_specs.insert(all_specs.end(), expected_specs.begin(), expected_specs.end());
    all_specs.insert(all_specs.end(), unexpected_specs.begin(), unexpected_specs.end());
    std::vector<std::int32_t> feedback_replay_leader_templates =
        zero_leader_templates();
    if (args.feedback_replay_mode == "normalized"
        && args.feedback_replay_fallback == "leader") {
        feedback_replay_leader_templates = build_feedback_replay_leader_templates(
            topology,
            all_specs,
            args.execution_mode,
            args.grating_rate_hz,
            args.baseline_rate_hz,
            args.feedback_replay_target_per_bin,
            args.v1_som_to_e_scale,
            args.v1_som_divisive_scale,
            args.v1_direct_divisive_scale,
            args.v1_feedforward_divisive_scale,
            v1_feedforward_divisive_gate_source,
            v1_direct_divisive_gate_source,
            args.v1_predicted_suppression_scale,
            args.v1_predicted_suppression_neighbor_weight,
            v1_predicted_suppression_locus,
            args.v1_stim_sigma_deg
        );
    }
    const auto feedback_replay_leader_template_total_counts =
        leader_template_totals(feedback_replay_leader_templates);
    const auto feedback_replay_leader_template_channel_counts =
        leader_template_rows(feedback_replay_leader_templates);
    semantic_hash = fnv1a_update_vector(
        semantic_hash,
        feedback_replay_leader_templates
    );

    if (args.execution_mode == "gpu_only_production") {
        const auto expected_batch = run_native_richter_conditions_batched(
            topology,
            expected_specs,
            args.grating_rate_hz,
            args.baseline_rate_hz,
            args.feedback_replay_mode,
            args.feedback_replay_target_per_bin,
            args.feedback_replay_fallback,
            feedback_replay_leader_templates,
            args.v1_som_to_e_scale,
            args.v1_som_divisive_scale,
            args.v1_direct_divisive_scale,
            args.v1_feedforward_divisive_scale,
            v1_feedforward_divisive_gate_source,
            v1_direct_divisive_gate_source,
            args.v1_predicted_suppression_scale,
            args.v1_predicted_suppression_neighbor_weight,
            v1_predicted_suppression_locus,
            args.v1_stim_sigma_deg,
            v1_error_comparator_mode,
            args.v1_error_sensory_gain,
            args.v1_error_prediction_gain,
            args.v1_error_prediction_shift
        );
        for (std::size_t i = 0; i < expected_specs.size(); ++i) {
            accumulate_condition(expected, expected_batch[i]);
            record_trial(
                0,
                expected_specs[i],
                expected_batch[i]
            );
            ++expected_trials;
        }
        const auto unexpected_batch = run_native_richter_conditions_batched(
            topology,
            unexpected_specs,
            args.grating_rate_hz,
            args.baseline_rate_hz,
            args.feedback_replay_mode,
            args.feedback_replay_target_per_bin,
            args.feedback_replay_fallback,
            feedback_replay_leader_templates,
            args.v1_som_to_e_scale,
            args.v1_som_divisive_scale,
            args.v1_direct_divisive_scale,
            args.v1_feedforward_divisive_scale,
            v1_feedforward_divisive_gate_source,
            v1_direct_divisive_gate_source,
            args.v1_predicted_suppression_scale,
            args.v1_predicted_suppression_neighbor_weight,
            v1_predicted_suppression_locus,
            args.v1_stim_sigma_deg,
            v1_error_comparator_mode,
            args.v1_error_sensory_gain,
            args.v1_error_prediction_gain,
            args.v1_error_prediction_shift
        );
        for (std::size_t i = 0; i < unexpected_specs.size(); ++i) {
            accumulate_condition(unexpected, unexpected_batch[i]);
            record_trial(
                1,
                unexpected_specs[i],
                unexpected_batch[i]
            );
            ++unexpected_trials;
        }
    } else {
        for (const auto& spec : expected_specs) {
            const auto result = run_native_richter_condition(
                topology,
                args.execution_mode,
                spec.seed,
                spec.leader_channel,
                spec.trailer_channel,
                args.grating_rate_hz,
                args.baseline_rate_hz,
                args.feedback_replay_mode,
                args.feedback_replay_target_per_bin,
                args.feedback_replay_fallback,
                feedback_replay_leader_templates,
                args.v1_som_to_e_scale,
                args.v1_som_divisive_scale,
                args.v1_direct_divisive_scale,
                args.v1_feedforward_divisive_scale,
                v1_feedforward_divisive_gate_source,
                v1_direct_divisive_gate_source,
                args.v1_predicted_suppression_scale,
                args.v1_predicted_suppression_neighbor_weight,
                v1_predicted_suppression_locus,
                args.v1_stim_sigma_deg,
                v1_error_comparator_mode,
                args.v1_error_sensory_gain,
                args.v1_error_prediction_gain,
                args.v1_error_prediction_shift
            );
            ConditionAccum single;
            accumulate_condition(single, result, args.execution_mode);
            accumulate_condition(expected, result, args.execution_mode);
            record_trial(0, spec, single);
            ++expected_trials;
        }
        for (const auto& spec : unexpected_specs) {
            const auto result = run_native_richter_condition(
                topology,
                args.execution_mode,
                spec.seed,
                spec.leader_channel,
                spec.trailer_channel,
                args.grating_rate_hz,
                args.baseline_rate_hz,
                args.feedback_replay_mode,
                args.feedback_replay_target_per_bin,
                args.feedback_replay_fallback,
                feedback_replay_leader_templates,
                args.v1_som_to_e_scale,
                args.v1_som_divisive_scale,
                args.v1_direct_divisive_scale,
                args.v1_feedforward_divisive_scale,
                v1_feedforward_divisive_gate_source,
                v1_direct_divisive_gate_source,
                args.v1_predicted_suppression_scale,
                args.v1_predicted_suppression_neighbor_weight,
                v1_predicted_suppression_locus,
                args.v1_stim_sigma_deg,
                v1_error_comparator_mode,
                args.v1_error_sensory_gain,
                args.v1_error_prediction_gain,
                args.v1_error_prediction_shift
            );
            ConditionAccum single;
            accumulate_condition(single, result, args.execution_mode);
            accumulate_condition(unexpected, result, args.execution_mode);
            record_trial(1, spec, single);
            ++unexpected_trials;
        }
    }
    const auto run_t1 = std::chrono::steady_clock::now();

    auto per_trial = [](double value, std::int32_t trials) -> double {
        return trials > 0 ? value / static_cast<double>(trials) : 0.0;
    };
    auto population_rate = [](double count, std::int32_t trials, int n_cells, double seconds) -> double {
        if (trials <= 0 || n_cells <= 0 || seconds <= 0.0) {
            return 0.0;
        }
        return count / (static_cast<double>(trials) * static_cast<double>(n_cells) * seconds);
    };

    const double expected_v1_trailer_per_trial = per_trial(expected.v1_trailer, expected_trials);
    const double unexpected_v1_trailer_per_trial = per_trial(unexpected.v1_trailer, unexpected_trials);
    const double dampening = unexpected_v1_trailer_per_trial - expected_v1_trailer_per_trial;
    const double expected_v1_error_trailer_per_trial =
        per_trial(expected.v1_error_trailer, expected_trials);
    const double unexpected_v1_error_trailer_per_trial =
        per_trial(unexpected.v1_error_trailer, unexpected_trials);
    const double expected_v1_error_neg_trailer_per_trial =
        per_trial(expected.v1_error_neg_trailer, expected_trials);
    const double unexpected_v1_error_neg_trailer_per_trial =
        per_trial(unexpected.v1_error_neg_trailer, unexpected_trials);
    const double expected_v1_error_total_trailer_per_trial =
        expected_v1_error_trailer_per_trial
        + expected_v1_error_neg_trailer_per_trial;
    const double unexpected_v1_error_total_trailer_per_trial =
        unexpected_v1_error_trailer_per_trial
        + unexpected_v1_error_neg_trailer_per_trial;
    const double expected_v1_error_signed_trailer_per_trial =
        expected_v1_error_trailer_per_trial
        - expected_v1_error_neg_trailer_per_trial;
    const double unexpected_v1_error_signed_trailer_per_trial =
        unexpected_v1_error_trailer_per_trial
        - unexpected_v1_error_neg_trailer_per_trial;
    const double expected_v1e_trailer_q_active_fC_per_trial =
        per_trial(expected.v1e_trailer_q_active_fC, expected_trials);
    const double unexpected_v1e_trailer_q_active_fC_per_trial =
        per_trial(unexpected.v1e_trailer_q_active_fC, unexpected_trials);
    const double expected_v1som_trailer_q_active_fC_per_trial =
        per_trial(expected.v1som_trailer_q_active_fC, expected_trials);
    const double unexpected_v1som_trailer_q_active_fC_per_trial =
        per_trial(unexpected.v1som_trailer_q_active_fC, unexpected_trials);
    const double expected_v1error_trailer_q_active_fC_per_trial =
        per_trial(expected.v1error_trailer_q_active_fC, expected_trials);
    const double unexpected_v1error_trailer_q_active_fC_per_trial =
        per_trial(unexpected.v1error_trailer_q_active_fC, unexpected_trials);
    const double expected_v1error_neg_trailer_q_active_fC_per_trial =
        per_trial(expected.v1error_neg_trailer_q_active_fC, expected_trials);
    const double unexpected_v1error_neg_trailer_q_active_fC_per_trial =
        per_trial(unexpected.v1error_neg_trailer_q_active_fC, unexpected_trials);
    const double expected_v1error_total_trailer_q_active_fC_per_trial =
        expected_v1error_trailer_q_active_fC_per_trial
        + expected_v1error_neg_trailer_q_active_fC_per_trial;
    const double unexpected_v1error_total_trailer_q_active_fC_per_trial =
        unexpected_v1error_trailer_q_active_fC_per_trial
        + unexpected_v1error_neg_trailer_q_active_fC_per_trial;
    const double expected_v1error_signed_trailer_q_active_fC_per_trial =
        expected_v1error_trailer_q_active_fC_per_trial
        - expected_v1error_neg_trailer_q_active_fC_per_trial;
    const double unexpected_v1error_signed_trailer_q_active_fC_per_trial =
        unexpected_v1error_trailer_q_active_fC_per_trial
        - unexpected_v1error_neg_trailer_q_active_fC_per_trial;
    const double expected_v1_trailer_q_active_fC_per_trial =
        expected_v1e_trailer_q_active_fC_per_trial
        + expected_v1som_trailer_q_active_fC_per_trial;
    const double unexpected_v1_trailer_q_active_fC_per_trial =
        unexpected_v1e_trailer_q_active_fC_per_trial
        + unexpected_v1som_trailer_q_active_fC_per_trial;
    const std::size_t total_trials =
        static_cast<std::size_t>(expected_trials + unexpected_trials);
    if (trial_v1_trailer_channel_counts.size() != total_trials
        || trial_v1_som_trailer_channel_counts.size() != total_trials
        || trial_v1_error_trailer_channel_counts.size() != total_trials
        || trial_v1_error_neg_trailer_channel_counts.size() != total_trials
        || trial_v1_trailer_100ms_channel_counts.size() != total_trials
        || trial_v1_som_trailer_100ms_channel_counts.size() != total_trials
        || trial_v1_error_trailer_100ms_channel_counts.size() != total_trials
        || trial_v1_error_neg_trailer_100ms_channel_counts.size() != total_trials
        || trial_index.size() != total_trials
        || trial_schedule_index.size() != total_trials
        || trial_is_expected.size() != total_trials
        || trial_leader_orientation.size() != total_trials
        || trial_trailer_orientation.size() != total_trials
        || trial_leader_channel.size() != total_trials
        || trial_trailer_channel.size() != total_trials
        || trial_dtheta_step.size() != total_trials
        || trial_hctx_e_preprobe_total_counts.size() != total_trials
        || trial_hctx_e_trailer_total_counts.size() != total_trials
        || trial_hpred_e_preprobe_total_counts.size() != total_trials
        || trial_hpred_e_preprobe_channel_counts.size() != total_trials
        || trial_hpred_e_trailer_total_counts.size() != total_trials
        || trial_hpred_e_trailer_100ms_total_counts.size() != total_trials
        || trial_hpred_e_trailer_100ms_channel_counts.size() != total_trials
        || trial_hpred_feedback_held_trailer_100ms_total_counts.size()
            != total_trials
        || trial_hpred_feedback_held_trailer_100ms_channel_counts.size()
            != total_trials
        || trial_hpred_feedback_normalized_trailer_100ms_total_weights.size()
            != total_trials
        || trial_hpred_feedback_normalized_trailer_100ms_channel_weights.size()
            != total_trials
        || trial_hpred_feedback_normalized_preprobe_zero.size()
            != total_trials
        || trial_hpred_feedback_normalized_fallback_used.size()
            != total_trials
        || trial_hpred_feedback_normalized_fallback_zero_template.size()
            != total_trials
        || trial_v1_predicted_suppression_trailer_channel_signal_sum.size()
            != total_trials
        || trial_v1_predicted_suppression_trailer_channel_gain_mean.size()
            != total_trials
        || trial_v1_predicted_suppression_trailer_raw_ie_before_sum.size()
            != total_trials
        || trial_v1_predicted_suppression_trailer_raw_ie_after_sum.size()
            != total_trials
        || trial_v1_predicted_suppression_trailer_raw_ie_delta_sum.size()
            != total_trials
        || trial_v1e_trailer_q_active_fC.size() != total_trials
        || trial_v1som_trailer_q_active_fC.size() != total_trials
        || trial_v1error_trailer_q_active_fC.size() != total_trials
        || trial_v1error_neg_trailer_q_active_fC.size() != total_trials
        || trial_v1error_total_trailer_q_active_fC.size() != total_trials
        || trial_v1error_signed_trailer_q_active_fC.size() != total_trials
        || trial_v1_trailer_q_active_fC.size() != total_trials) {
        throw std::runtime_error("trial_data vector size mismatch");
    }
    auto selected_raw_ie_mean = [&](
        const std::vector<std::array<double, kV1Channels>>& values,
        int expected_flag,
        bool use_leader_channel
    ) -> double {
        double total = 0.0;
        std::int32_t n = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (trial_is_expected[i] != expected_flag) {
                continue;
            }
            const int channel = use_leader_channel
                ? trial_leader_channel[i]
                : trial_trailer_channel[i];
            total += values[i][static_cast<std::size_t>(channel)];
            ++n;
        }
        return n > 0 ? total / static_cast<double>(n) : 0.0;
    };
    auto all_channel_raw_ie_mean = [&](
        const std::vector<std::array<double, kV1Channels>>& values,
        int expected_flag
    ) -> double {
        double total = 0.0;
        std::int32_t n = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (trial_is_expected[i] != expected_flag) {
                continue;
            }
            total += std::accumulate(values[i].begin(), values[i].end(), 0.0);
            ++n;
        }
        return n > 0 ? total / static_cast<double>(n) : 0.0;
    };
    const double expected_predicted_channel_raw_ie_delta_sum_per_trial =
        selected_raw_ie_mean(
            trial_v1_predicted_suppression_trailer_raw_ie_delta_sum, 1, true
        );
    const double unexpected_predicted_channel_raw_ie_delta_sum_per_trial =
        selected_raw_ie_mean(
            trial_v1_predicted_suppression_trailer_raw_ie_delta_sum, 0, true
        );
    const double expected_actual_channel_raw_ie_delta_sum_per_trial =
        selected_raw_ie_mean(
            trial_v1_predicted_suppression_trailer_raw_ie_delta_sum, 1, false
        );
    const double unexpected_actual_channel_raw_ie_delta_sum_per_trial =
        selected_raw_ie_mean(
            trial_v1_predicted_suppression_trailer_raw_ie_delta_sum, 0, false
        );
    const double expected_all_channel_raw_ie_delta_sum_per_trial =
        all_channel_raw_ie_mean(
            trial_v1_predicted_suppression_trailer_raw_ie_delta_sum, 1
        );
    const double unexpected_all_channel_raw_ie_delta_sum_per_trial =
        all_channel_raw_ie_mean(
            trial_v1_predicted_suppression_trailer_raw_ie_delta_sum, 0
        );
    std::vector<std::array<double, kV1Channels>> trial_v1_trailer_channel_rates_hz;
    trial_v1_trailer_channel_rates_hz.reserve(total_trials);
    for (const auto& counts : trial_v1_trailer_channel_counts) {
        trial_v1_trailer_channel_rates_hz.push_back(
            v1_trailer_channel_rates_hz(counts)
        );
    }
    semantic_hash = fnv1a_update_scalar(semantic_hash, expected_trials);
    semantic_hash = fnv1a_update_scalar(semantic_hash, unexpected_trials);
    semantic_hash = fnv1a_update_scalar(semantic_hash, expected_v1_trailer_per_trial);
    semantic_hash = fnv1a_update_scalar(semantic_hash, unexpected_v1_trailer_per_trial);
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        expected_v1_trailer_q_active_fC_per_trial
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        unexpected_v1_trailer_q_active_fC_per_trial
    );
    const std::int32_t feedback_replay_preprobe_zero_trial_count =
        std::accumulate(
            trial_hpred_feedback_normalized_preprobe_zero.begin(),
            trial_hpred_feedback_normalized_preprobe_zero.end(),
            0
        );
    const std::int32_t feedback_replay_fallback_used_trial_count =
        std::accumulate(
            trial_hpred_feedback_normalized_fallback_used.begin(),
            trial_hpred_feedback_normalized_fallback_used.end(),
            0
        );
    const std::int32_t feedback_replay_fallback_zero_template_trial_count =
        std::accumulate(
            trial_hpred_feedback_normalized_fallback_zero_template.begin(),
            trial_hpred_feedback_normalized_fallback_zero_template.end(),
            0
        );
    const std::int32_t normalized_zero_total_bin_count_before_fallback =
        feedback_replay_preprobe_zero_trial_count * kTrailerBinCount;
    std::int32_t normalized_zero_total_bin_count_after_fallback = 0;
    for (const auto& trial_weights :
         trial_hpred_feedback_normalized_trailer_100ms_total_weights) {
        for (const double weight : trial_weights) {
            if (weight == 0.0) {
                ++normalized_zero_total_bin_count_after_fallback;
            }
        }
    }
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        feedback_replay_preprobe_zero_trial_count
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        feedback_replay_fallback_used_trial_count
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        feedback_replay_fallback_zero_template_trial_count
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        normalized_zero_total_bin_count_before_fallback
    );
    semantic_hash = fnv1a_update_scalar(
        semantic_hash,
        normalized_zero_total_bin_count_after_fallback
    );
    const std::string result_hash = hash_hex(semantic_hash);

    const auto write_t0 = std::chrono::steady_clock::now();
    if (!args.out_path.parent_path().empty()) {
        std::filesystem::create_directories(args.out_path.parent_path());
    }
    std::ofstream out(args.out_path);
    if (!out) {
        throw std::runtime_error("failed to open Richter result JSON: " + args.out_path.string());
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"artifact_kind\": \"native_richter_dampening_result\",\n";
    out << "  \"backend\": \"native_cxx_cuda_executable\",\n";
    out << "  \"execution_mode\": \"" << json_escape(args.execution_mode) << "\",\n";
    out << "  \"seed\": " << args.seed << ",\n";
    out << "  \"content_hash_fnv1a64\": \"" << result_hash << "\",\n";
    out << "  \"checkpoint\": {\"path\": \"" << json_escape(checkpoint.json_path.string())
        << "\", \"content_hash_fnv1a64\": \"" << checkpoint.content_hash
        << "\", \"W_ctx_pred_path\": \"" << json_escape(checkpoint.w_path.string())
        << "\", \"W_ctx_pred_fnv1a64\": \"" << checkpoint.w_hash
        << "\", \"has_learned_hpred_v1direct\": "
        << (checkpoint.has_learned_hpred_v1direct ? "true" : "false")
        << ", \"has_learned_hpred_v1som\": "
        << (checkpoint.has_learned_hpred_v1som ? "true" : "false");
    if (checkpoint.has_learned_hpred_v1direct) {
        out << ", \"W_hpred_v1direct_path\": \""
            << json_escape(checkpoint.learned_hpred_v1direct_path.string())
            << "\", \"W_hpred_v1direct_fnv1a64\": \""
            << checkpoint.learned_hpred_v1direct_hash << "\"";
    }
    if (checkpoint.has_learned_hpred_v1som) {
        out << ", \"W_hpred_v1som_path\": \""
            << json_escape(checkpoint.learned_hpred_v1som_path.string())
            << "\", \"W_hpred_v1som_fnv1a64\": \""
            << checkpoint.learned_hpred_v1som_hash << "\"";
    }
    out << "},\n";
    out << "  \"feedback\": {\"g_total\": " << args.feedback_g_total
        << ", \"r\": " << args.feedback_r
        << ", \"resolved_g_direct\": " << feedback.g_direct
        << ", \"resolved_g_som\": " << feedback.g_som
        << ", \"direct_source\": \"" << json_escape(args.feedback_direct_source)
        << "\""
        << ", \"direct_source_description\": \""
        << (args.feedback_direct_source == "fixed"
            ? "static_same_channel_hpred_to_v1e_apical_kernel"
            : (args.feedback_direct_source == "learned"
               ? "checkpointed_prediction_task_hpred_to_v1e_direct_apical_weights"
               : (args.feedback_direct_source == "learned-shifted"
                  ? "checkpointed_direct_weights_with_hpred_source_channel_rotated_by_plus_two_control"
                  : "direct_feedback_disabled_zero_weight_sentinel")))
        << "\""
        << ", \"som_source\": \"" << json_escape(args.feedback_som_source)
        << "\""
        << ", \"som_source_description\": \""
        << (args.feedback_som_source == "fixed"
            ? "static_center_surround_kernel"
            : (args.feedback_som_source == "learned"
               ? "checkpointed_prediction_task_hpred_to_v1som_weights"
               : (args.feedback_som_source == "learned-shifted"
                  ? "checkpointed_weights_with_hpred_source_channel_rotated_by_plus_two_control"
                  : "som_feedback_disabled_zero_weight_sentinel")))
        << "\""
        << ", \"som_kernel_center_weight\": "
        << args.feedback_som_center_weight
        << ", \"som_kernel_surround_d1_weight\": "
        << kFeedbackSomD1Weight
        << ", \"som_kernel_surround_d2_weight\": "
        << kFeedbackSomD2Weight
        << ", \"resolved_som_center_weight\": "
        << args.feedback_som_center_weight * feedback.g_som
        << ", \"resolved_som_surround_d1_weight\": "
        << kFeedbackSomD1Weight * feedback.g_som
        << ", \"resolved_som_surround_d2_weight\": "
        << kFeedbackSomD2Weight * feedback.g_som
        << ", \"v1_som_to_e_scale\": " << args.v1_som_to_e_scale
        << ", \"v1_som_divisive_scale\": " << args.v1_som_divisive_scale
        << ", \"v1_som_divisive_gate\": "
        << "\"same_channel_current_step_v1som_spike_count\""
        << ", \"v1_som_divisive_denominator\": "
        << "\"1 + scale * same_channel_current_step_v1som_spike_count\""
        << ", \"v1_direct_divisive_scale\": " << args.v1_direct_divisive_scale
        << ", \"v1_direct_divisive_gate_source\": \""
        << json_escape(args.v1_direct_divisive_gate_source)
        << "\""
        << ", \"v1_direct_divisive_gate\": "
        << "\""
        << v1_direct_divisive_gate_description(
            args.v1_direct_divisive_gate_source
        )
        << "\""
        << ", \"v1_direct_divisive_denominator\": "
        << "\""
        << v1_direct_divisive_denominator_description(
            args.v1_direct_divisive_gate_source
        )
        << "\""
        << ", \"v1_direct_divisive_target\": \"V1_E_apical_direct_i_ap_only\""
        << ", \"v1_feedforward_divisive_scale\": "
        << args.v1_feedforward_divisive_scale
        << ", \"v1_feedforward_divisive_gate_source\": \""
        << json_escape(args.v1_feedforward_divisive_gate_source)
        << "\""
        << ", \"v1_feedforward_divisive_gate\": "
        << "\""
        << v1_feedforward_divisive_gate_description(
            args.v1_feedforward_divisive_gate_source
        )
        << "\""
        << ", \"v1_feedforward_divisive_denominator\": "
        << "\""
        << v1_feedforward_divisive_denominator_description(
            args.v1_feedforward_divisive_gate_source
        )
        << "\""
        << ", \"v1_feedforward_divisive_target\": "
        << "\"V1_E_feedforward_somatic_effective_ie_only\""
        << ", \"v1_predicted_suppression_scale\": "
        << args.v1_predicted_suppression_scale
        << ", \"v1_predicted_suppression_neighbor_weight\": "
        << args.v1_predicted_suppression_neighbor_weight
        << ", \"v1_predicted_suppression_locus\": \""
        << json_escape(args.v1_predicted_suppression_locus)
        << "\""
        << ", \"v1_predicted_suppression_gate\": "
        << "\"same_channel_held_hpred_feedback_flags_plus_neighbor_weighted_pm1_during_trailer\""
        << ", \"v1_predicted_suppression_denominator\": "
        << "\"1 + scale * (same_channel_feedback_count + neighbor_weight * adjacent_feedback_count)\""
        << ", \"v1_predicted_suppression_target\": "
        << "\""
        << v1_predicted_suppression_target_description(
            args.v1_predicted_suppression_locus
        )
        << "\""
        << ", \"v1_error_comparator_mode\": \""
        << json_escape(args.v1_error_comparator_mode)
        << "\""
        << ", \"v1_error_comparator_description\": "
        << "\"separate_measurement_only_v1_error_population; fixed_symmetric uses ERR+ sensory_excitation_minus_held_hpred_prediction_inhibition; signed_normalized adds ERR- prediction_excitation_minus_sensory_inhibition using normalized_preprobe_hpred_shape; no_v1_feedback\""
        << ", \"v1_error_sensory_gain\": "
        << args.v1_error_sensory_gain
        << ", \"v1_error_prediction_gain\": "
        << args.v1_error_prediction_gain
        << ", \"v1_error_prediction_shift\": "
        << args.v1_error_prediction_shift
        << "},\n";
    out << "  \"feedback_replay\": {\"mode\": \""
        << json_escape(args.feedback_replay_mode)
        << "\", \"fallback_mode\": \""
        << json_escape(args.feedback_replay_fallback)
        << "\", \"normalized_target_per_100ms_bin\": "
        << args.feedback_replay_target_per_bin
        << ", \"target_source\": \"cli_or_default_rate100_observed_preprobe_total_per_trial_bin\", \"leader_template_total_counts\": ";
    write_vector_json(
        out,
        std::vector<std::int32_t>(
            feedback_replay_leader_template_total_counts.begin(),
            feedback_replay_leader_template_total_counts.end()
        )
    );
    out << ", \"leader_template_channel_counts\": ";
    write_fixed_matrix_json(
        out,
        std::vector<std::array<std::int32_t, kV1Channels>>(
            feedback_replay_leader_template_channel_counts.begin(),
            feedback_replay_leader_template_channel_counts.end()
        )
    );
    out << ", \"preprobe_zero_trial_count\": "
        << feedback_replay_preprobe_zero_trial_count
        << ", \"fallback_used_trial_count\": "
        << feedback_replay_fallback_used_trial_count
        << ", \"fallback_zero_template_trial_count\": "
        << feedback_replay_fallback_zero_template_trial_count
        << ", \"normalized_zero_total_bin_count_before_fallback\": "
        << normalized_zero_total_bin_count_before_fallback
        << ", \"normalized_zero_total_bin_count_after_fallback\": "
        << normalized_zero_total_bin_count_after_fallback << "},\n";
    out << "  \"schedule\": {\"leader_ms\": 500.0, \"preprobe_ms\": 100.0, \"trailer_ms\": 500.0, \"iti_ms\": 1500.0, \"dt_ms\": 0.1, \"orientation_channel_stride\": 2, \"expected_step\": 1, \"unexpected_steps\": [2,3,4,5], \"reps_expected\": "
        << args.reps_expected << ", \"reps_unexpected\": " << args.reps_unexpected
        << ", \"expected_trials\": " << expected_trials << ", \"unexpected_trials\": "
        << unexpected_trials << "},\n";
    out << "  \"stimulus\": {\"grating_rate_hz\": " << args.grating_rate_hz
        << ", \"baseline_rate_hz\": " << args.baseline_rate_hz
        << ", \"v1_stim_sigma_deg\": " << args.v1_stim_sigma_deg
        << "},\n";
    out << "  \"edge_counts\": {\"v1_stim_to_e\": " << topology.stim_pre.size()
        << ", \"v1_to_h_ctx\": " << topology.v1_to_h_pre.size()
        << ", \"ctx_to_pred\": " << topology.ctx_to_pred_pre.size()
        << ", \"fb_pred_to_v1e_apical\": " << topology.fb_direct_pre.size()
        << ", \"fb_pred_to_v1som\": " << topology.fb_som_pre.size() << "},\n";
    out << "  \"source_event_counts\": {\"expected_total\": " << expected.source_total
        << ", \"unexpected_total\": " << unexpected.source_total << "},\n";
    out << "  \"q_active_energy_fC\": {\n";
    out << "    \"definition\": \"integrated active charge; V1_E=sum(abs(I_e)+abs(I_i)+abs(I_ap_e))*dt_ms, V1_SOM=sum(abs(I_e)+abs(I_i))*dt_ms; pA*ms equals fC\",\n";
    out << "    \"expected\": {\"v1e_trailer_q_active_fC_per_trial\": "
        << expected_v1e_trailer_q_active_fC_per_trial
        << ", \"v1som_trailer_q_active_fC_per_trial\": "
        << expected_v1som_trailer_q_active_fC_per_trial
        << ", \"v1error_trailer_q_active_fC_per_trial\": "
        << expected_v1error_trailer_q_active_fC_per_trial
        << ", \"v1error_neg_trailer_q_active_fC_per_trial\": "
        << expected_v1error_neg_trailer_q_active_fC_per_trial
        << ", \"v1error_total_trailer_q_active_fC_per_trial\": "
        << expected_v1error_total_trailer_q_active_fC_per_trial
        << ", \"v1error_signed_trailer_q_active_fC_per_trial\": "
        << expected_v1error_signed_trailer_q_active_fC_per_trial
        << ", \"v1_trailer_q_active_fC_per_trial\": "
        << expected_v1_trailer_q_active_fC_per_trial << "},\n";
    out << "    \"unexpected\": {\"v1e_trailer_q_active_fC_per_trial\": "
        << unexpected_v1e_trailer_q_active_fC_per_trial
        << ", \"v1som_trailer_q_active_fC_per_trial\": "
        << unexpected_v1som_trailer_q_active_fC_per_trial
        << ", \"v1error_trailer_q_active_fC_per_trial\": "
        << unexpected_v1error_trailer_q_active_fC_per_trial
        << ", \"v1error_neg_trailer_q_active_fC_per_trial\": "
        << unexpected_v1error_neg_trailer_q_active_fC_per_trial
        << ", \"v1error_total_trailer_q_active_fC_per_trial\": "
        << unexpected_v1error_total_trailer_q_active_fC_per_trial
        << ", \"v1error_signed_trailer_q_active_fC_per_trial\": "
        << unexpected_v1error_signed_trailer_q_active_fC_per_trial
        << ", \"v1_trailer_q_active_fC_per_trial\": "
        << unexpected_v1_trailer_q_active_fC_per_trial << "}\n";
    out << "  },\n";
    out << "  \"raw_counts\": {\n";
    out << "    \"expected\": {\"v1_e\": {\"leader\": " << expected.v1_leader
        << ", \"preprobe\": " << expected.v1_preprobe << ", \"trailer\": " << expected.v1_trailer
        << "}, \"v1_error\": {\"trailer\": " << expected.v1_error_trailer
        << ", \"trailer_per_trial\": " << expected_v1_error_trailer_per_trial
        << "}, \"v1_error_neg\": {\"trailer\": "
        << expected.v1_error_neg_trailer
        << ", \"trailer_per_trial\": "
        << expected_v1_error_neg_trailer_per_trial
        << "}, \"v1_error_total\": {\"trailer_per_trial\": "
        << expected_v1_error_total_trailer_per_trial
        << "}, \"v1_error_signed\": {\"trailer_per_trial\": "
        << expected_v1_error_signed_trailer_per_trial
        << "}, \"hctx_e\": {\"preprobe\": " << expected.hctx_preprobe
        << ", \"trailer\": " << expected.hctx_trailer << "}, \"hpred_e\": {\"preprobe\": "
        << expected.hpred_preprobe << ", \"trailer\": " << expected.hpred_trailer << "}},\n";
    out << "    \"unexpected\": {\"v1_e\": {\"leader\": " << unexpected.v1_leader
        << ", \"preprobe\": " << unexpected.v1_preprobe << ", \"trailer\": " << unexpected.v1_trailer
        << "}, \"v1_error\": {\"trailer\": " << unexpected.v1_error_trailer
        << ", \"trailer_per_trial\": " << unexpected_v1_error_trailer_per_trial
        << "}, \"v1_error_neg\": {\"trailer\": "
        << unexpected.v1_error_neg_trailer
        << ", \"trailer_per_trial\": "
        << unexpected_v1_error_neg_trailer_per_trial
        << "}, \"v1_error_total\": {\"trailer_per_trial\": "
        << unexpected_v1_error_total_trailer_per_trial
        << "}, \"v1_error_signed\": {\"trailer_per_trial\": "
        << unexpected_v1_error_signed_trailer_per_trial
        << "}, \"hctx_e\": {\"preprobe\": " << unexpected.hctx_preprobe
        << ", \"trailer\": " << unexpected.hctx_trailer << "}, \"hpred_e\": {\"preprobe\": "
        << unexpected.hpred_preprobe << ", \"trailer\": " << unexpected.hpred_trailer << "}}\n";
    out << "  },\n";
    out << "  \"trial_data\": {\n";
    out << "    \"trial_index\": "; write_vector_json(out, trial_index); out << ",\n";
    out << "    \"schedule_trial_index\": "; write_vector_json(out, trial_schedule_index); out << ",\n";
    out << "    \"is_expected\": "; write_vector_json(out, trial_is_expected); out << ",\n";
    out << "    \"leader_orientation\": "; write_vector_json(out, trial_leader_orientation); out << ",\n";
    out << "    \"trailer_orientation\": "; write_vector_json(out, trial_trailer_orientation); out << ",\n";
    out << "    \"leader_channel\": "; write_vector_json(out, trial_leader_channel); out << ",\n";
    out << "    \"trailer_channel\": "; write_vector_json(out, trial_trailer_channel); out << ",\n";
    out << "    \"dtheta_step\": "; write_vector_json(out, trial_dtheta_step); out << ",\n";
    out << "    \"hctx_e_preprobe_total_counts\": ";
    write_vector_json(out, trial_hctx_e_preprobe_total_counts);
    out << ",\n";
    out << "    \"hctx_e_trailer_total_counts\": ";
    write_vector_json(out, trial_hctx_e_trailer_total_counts);
    out << ",\n";
    out << "    \"hpred_e_preprobe_total_counts\": ";
    write_vector_json(out, trial_hpred_e_preprobe_total_counts);
    out << ",\n";
    out << "    \"hpred_e_preprobe_channel_counts\": ";
    write_fixed_matrix_json(out, trial_hpred_e_preprobe_channel_counts);
    out << ",\n";
    out << "    \"hpred_e_trailer_total_counts\": ";
    write_vector_json(out, trial_hpred_e_trailer_total_counts);
    out << ",\n";
    out << "    \"hpred_e_trailer_100ms_total_counts\": ";
    write_fixed_matrix_json(out, trial_hpred_e_trailer_100ms_total_counts);
    out << ",\n";
    out << "    \"hpred_e_trailer_100ms_channel_counts\": ";
    write_fixed_cube_json(out, trial_hpred_e_trailer_100ms_channel_counts);
    out << ",\n";
    out << "    \"hpred_feedback_held_trailer_100ms_total_counts\": ";
    write_fixed_matrix_json(
        out,
        trial_hpred_feedback_held_trailer_100ms_total_counts
    );
    out << ",\n";
    out << "    \"hpred_feedback_held_trailer_100ms_channel_counts\": ";
    write_fixed_cube_json(
        out,
        trial_hpred_feedback_held_trailer_100ms_channel_counts
    );
    out << ",\n";
    out << "    \"hpred_feedback_normalized_trailer_100ms_total_weights\": ";
    write_fixed_matrix_json(
        out,
        trial_hpred_feedback_normalized_trailer_100ms_total_weights
    );
    out << ",\n";
    out << "    \"hpred_feedback_normalized_trailer_100ms_channel_weights\": ";
    write_fixed_cube_json(
        out,
        trial_hpred_feedback_normalized_trailer_100ms_channel_weights
    );
    out << ",\n";
    out << "    \"hpred_feedback_normalized_preprobe_zero\": ";
    write_vector_json(out, trial_hpred_feedback_normalized_preprobe_zero);
    out << ",\n";
    out << "    \"hpred_feedback_normalized_fallback_used\": ";
    write_vector_json(out, trial_hpred_feedback_normalized_fallback_used);
    out << ",\n";
    out << "    \"hpred_feedback_normalized_fallback_zero_template\": ";
    write_vector_json(
        out,
        trial_hpred_feedback_normalized_fallback_zero_template
    );
    out << ",\n";
    out << "    \"v1_predicted_suppression_trailer_channel_signal_sum\": ";
    write_fixed_matrix_json(
        out,
        trial_v1_predicted_suppression_trailer_channel_signal_sum
    );
    out << ",\n";
    out << "    \"v1_predicted_suppression_trailer_channel_gain_mean\": ";
    write_fixed_matrix_json(
        out,
        trial_v1_predicted_suppression_trailer_channel_gain_mean
    );
    out << ",\n";
    out << "    \"v1_predicted_suppression_trailer_raw_ie_before_sum\": ";
    write_fixed_matrix_json(
        out,
        trial_v1_predicted_suppression_trailer_raw_ie_before_sum
    );
    out << ",\n";
    out << "    \"v1_predicted_suppression_trailer_raw_ie_after_sum\": ";
    write_fixed_matrix_json(
        out,
        trial_v1_predicted_suppression_trailer_raw_ie_after_sum
    );
    out << ",\n";
    out << "    \"v1_predicted_suppression_trailer_raw_ie_delta_sum\": ";
    write_fixed_matrix_json(
        out,
        trial_v1_predicted_suppression_trailer_raw_ie_delta_sum
    );
    out << ",\n";
    out << "    \"v1_e_trailer_channel_counts\": ";
    write_fixed_matrix_json(out, trial_v1_trailer_channel_counts);
    out << ",\n";
    out << "    \"v1_e_trailer_100ms_channel_counts\": ";
    write_fixed_cube_json(out, trial_v1_trailer_100ms_channel_counts);
    out << ",\n";
    out << "    \"v1_som_trailer_channel_counts\": ";
    write_fixed_matrix_json(out, trial_v1_som_trailer_channel_counts);
    out << ",\n";
    out << "    \"v1_som_trailer_100ms_channel_counts\": ";
    write_fixed_cube_json(out, trial_v1_som_trailer_100ms_channel_counts);
    out << ",\n";
    out << "    \"v1_error_trailer_channel_counts\": ";
    write_fixed_matrix_json(out, trial_v1_error_trailer_channel_counts);
    out << ",\n";
    out << "    \"v1_error_trailer_100ms_channel_counts\": ";
    write_fixed_cube_json(out, trial_v1_error_trailer_100ms_channel_counts);
    out << ",\n";
    out << "    \"v1_error_neg_trailer_channel_counts\": ";
    write_fixed_matrix_json(out, trial_v1_error_neg_trailer_channel_counts);
    out << ",\n";
    out << "    \"v1_error_neg_trailer_100ms_channel_counts\": ";
    write_fixed_cube_json(out, trial_v1_error_neg_trailer_100ms_channel_counts);
    out << ",\n";
    out << "    \"v1e_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1e_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1som_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1som_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1error_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1error_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1error_neg_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1error_neg_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1error_total_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1error_total_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1error_signed_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1error_signed_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1_trailer_q_active_fC\": ";
    write_vector_json(out, trial_v1_trailer_q_active_fC);
    out << ",\n";
    out << "    \"v1_e_trailer_channel_rates_hz\": ";
    write_fixed_matrix_json(out, trial_v1_trailer_channel_rates_hz);
    out << "\n";
    out << "  },\n";
    out << "  \"rates_hz\": {\n";
    out << "    \"expected\": {\"v1_e_trailer_population\": " << population_rate(expected.v1_trailer, expected_trials, 192, 0.5)
        << ", \"hctx_e_preprobe_population\": " << population_rate(expected.hctx_preprobe, expected_trials, 192, 0.1)
        << ", \"hpred_e_trailer_population\": " << population_rate(expected.hpred_trailer, expected_trials, 192, 0.5) << "},\n";
    out << "    \"unexpected\": {\"v1_e_trailer_population\": " << population_rate(unexpected.v1_trailer, unexpected_trials, 192, 0.5)
        << ", \"hctx_e_preprobe_population\": " << population_rate(unexpected.hctx_preprobe, unexpected_trials, 192, 0.1)
        << ", \"hpred_e_trailer_population\": " << population_rate(unexpected.hpred_trailer, unexpected_trials, 192, 0.5) << "}\n";
    out << "  },\n";
    out << "  \"metrics\": {\"expected_v1_trailer_count_per_trial\": " << expected_v1_trailer_per_trial
        << ", \"unexpected_v1_trailer_count_per_trial\": " << unexpected_v1_trailer_per_trial
        << ", \"expected_v1e_trailer_q_active_fC_per_trial\": "
        << expected_v1e_trailer_q_active_fC_per_trial
        << ", \"unexpected_v1e_trailer_q_active_fC_per_trial\": "
        << unexpected_v1e_trailer_q_active_fC_per_trial
        << ", \"expected_v1som_trailer_q_active_fC_per_trial\": "
        << expected_v1som_trailer_q_active_fC_per_trial
        << ", \"unexpected_v1som_trailer_q_active_fC_per_trial\": "
        << unexpected_v1som_trailer_q_active_fC_per_trial
        << ", \"expected_v1_trailer_q_active_fC_per_trial\": "
        << expected_v1_trailer_q_active_fC_per_trial
        << ", \"unexpected_v1_trailer_q_active_fC_per_trial\": "
        << unexpected_v1_trailer_q_active_fC_per_trial
        << ", \"expected_predicted_channel_raw_ie_delta_sum_per_trial\": "
        << expected_predicted_channel_raw_ie_delta_sum_per_trial
        << ", \"unexpected_predicted_channel_raw_ie_delta_sum_per_trial\": "
        << unexpected_predicted_channel_raw_ie_delta_sum_per_trial
        << ", \"expected_actual_channel_raw_ie_delta_sum_per_trial\": "
        << expected_actual_channel_raw_ie_delta_sum_per_trial
        << ", \"unexpected_actual_channel_raw_ie_delta_sum_per_trial\": "
        << unexpected_actual_channel_raw_ie_delta_sum_per_trial
        << ", \"expected_all_channel_raw_ie_delta_sum_per_trial\": "
        << expected_all_channel_raw_ie_delta_sum_per_trial
        << ", \"unexpected_all_channel_raw_ie_delta_sum_per_trial\": "
        << unexpected_all_channel_raw_ie_delta_sum_per_trial
        << ", \"dampening_unexpected_minus_expected\": " << dampening
        << ", \"expected_dampening_sign\": \"" << (dampening > 0.0 ? "expected_less_than_unexpected" : (dampening < 0.0 ? "expected_greater_than_unexpected" : "equal"))
        << "\"},\n";
    out << "  \"device\": {\"backend_info\": \"" << json_escape(expectation_snn_cuda::backend_info()) << "\"},\n";
    const double run_wall = std::chrono::duration<double>(run_t1 - run_t0).count();
    const auto write_t1 = std::chrono::steady_clock::now();
    const double write_wall = std::chrono::duration<double>(write_t1 - write_t0).count();
    const double total_wall = std::chrono::duration<double>(write_t1 - total_t0).count();
    out << "  \"timing_seconds\": {\"run\": " << run_wall << ", \"write\": " << write_wall
        << ", \"total_excluding_build\": " << total_wall << "}\n";
    out << "}\n";

    std::cout << std::setprecision(12)
              << "command=richter-dampening\n"
              << "status=PASS\n"
              << "execution_mode=" << args.execution_mode << "\n"
              << "seed=" << args.seed << "\n"
              << "v1_stim_sigma_deg=" << args.v1_stim_sigma_deg << "\n"
              << "feedback_g_total=" << args.feedback_g_total << "\n"
              << "feedback_r=" << args.feedback_r << "\n"
              << "feedback_som_center_weight="
              << args.feedback_som_center_weight << "\n"
              << "feedback_direct_source=" << args.feedback_direct_source << "\n"
              << "feedback_som_source=" << args.feedback_som_source << "\n"
              << "v1_som_to_e_scale=" << args.v1_som_to_e_scale << "\n"
              << "v1_som_divisive_scale=" << args.v1_som_divisive_scale << "\n"
              << "v1_direct_divisive_scale=" << args.v1_direct_divisive_scale << "\n"
              << "v1_direct_divisive_gate_source="
              << args.v1_direct_divisive_gate_source << "\n"
              << "v1_feedforward_divisive_scale="
              << args.v1_feedforward_divisive_scale << "\n"
              << "v1_feedforward_divisive_gate_source="
              << args.v1_feedforward_divisive_gate_source << "\n"
              << "v1_predicted_suppression_scale="
              << args.v1_predicted_suppression_scale << "\n"
              << "v1_predicted_suppression_neighbor_weight="
              << args.v1_predicted_suppression_neighbor_weight << "\n"
              << "v1_predicted_suppression_locus="
              << args.v1_predicted_suppression_locus << "\n"
              << "feedback_resolved_g_direct=" << feedback.g_direct << "\n"
              << "feedback_resolved_g_som=" << feedback.g_som << "\n"
              << "feedback_resolved_som_center_weight="
              << args.feedback_som_center_weight * feedback.g_som << "\n"
              << "feedback_resolved_som_surround_d1_weight="
              << kFeedbackSomD1Weight * feedback.g_som << "\n"
              << "feedback_resolved_som_surround_d2_weight="
              << kFeedbackSomD2Weight * feedback.g_som << "\n"
              << "feedback_replay_mode=" << args.feedback_replay_mode << "\n"
              << "feedback_replay_fallback=" << args.feedback_replay_fallback << "\n"
              << "feedback_replay_target_per_100ms_bin="
              << args.feedback_replay_target_per_bin << "\n"
              << "feedback_replay_preprobe_zero_trial_count="
              << feedback_replay_preprobe_zero_trial_count << "\n"
              << "feedback_replay_fallback_used_trial_count="
              << feedback_replay_fallback_used_trial_count << "\n"
              << "feedback_replay_fallback_zero_template_trial_count="
              << feedback_replay_fallback_zero_template_trial_count << "\n"
              << "feedback_replay_normalized_zero_total_bin_count_before_fallback="
              << normalized_zero_total_bin_count_before_fallback << "\n"
              << "feedback_replay_normalized_zero_total_bin_count_after_fallback="
              << normalized_zero_total_bin_count_after_fallback << "\n"
              << "checkpoint_path=" << checkpoint.json_path << "\n"
              << "result_path=" << args.out_path << "\n"
              << "content_hash_fnv1a64=" << result_hash << "\n"
              << "expected_trials=" << expected_trials << "\n"
              << "unexpected_trials=" << unexpected_trials << "\n"
              << "expected_v1_trailer_count_per_trial=" << expected_v1_trailer_per_trial << "\n"
              << "unexpected_v1_trailer_count_per_trial=" << unexpected_v1_trailer_per_trial << "\n"
              << "expected_v1e_trailer_q_active_fC_per_trial="
              << expected_v1e_trailer_q_active_fC_per_trial << "\n"
              << "unexpected_v1e_trailer_q_active_fC_per_trial="
              << unexpected_v1e_trailer_q_active_fC_per_trial << "\n"
              << "expected_v1som_trailer_q_active_fC_per_trial="
              << expected_v1som_trailer_q_active_fC_per_trial << "\n"
              << "unexpected_v1som_trailer_q_active_fC_per_trial="
              << unexpected_v1som_trailer_q_active_fC_per_trial << "\n"
              << "expected_v1_trailer_q_active_fC_per_trial="
              << expected_v1_trailer_q_active_fC_per_trial << "\n"
              << "unexpected_v1_trailer_q_active_fC_per_trial="
              << unexpected_v1_trailer_q_active_fC_per_trial << "\n"
              << "expected_predicted_channel_raw_ie_delta_sum_per_trial="
              << expected_predicted_channel_raw_ie_delta_sum_per_trial << "\n"
              << "unexpected_predicted_channel_raw_ie_delta_sum_per_trial="
              << unexpected_predicted_channel_raw_ie_delta_sum_per_trial << "\n"
              << "expected_actual_channel_raw_ie_delta_sum_per_trial="
              << expected_actual_channel_raw_ie_delta_sum_per_trial << "\n"
              << "unexpected_actual_channel_raw_ie_delta_sum_per_trial="
              << unexpected_actual_channel_raw_ie_delta_sum_per_trial << "\n"
              << "dampening_unexpected_minus_expected=" << dampening << "\n"
              << "source_expected_total=" << expected.source_total << "\n"
              << "source_unexpected_total=" << unexpected.source_total << "\n"
              << "wall_run_seconds=" << run_wall << "\n"
              << "wall_total_excluding_build_seconds=" << total_wall << "\n";
    return 0;
}

int not_implemented(const std::string& command) {
    std::cerr << command << " is scaffolded in the native executable but not yet "
              << "implemented as a production C++/CUDA runtime path.\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.command == "help" || args.command == "--help" || args.command == "-h") {
            print_usage(std::cout);
            return 0;
        }
        if (args.command == "device-info") {
            return device_info();
        }
        if (args.command == "validate-fixture" || args.command == "self-test") {
            return validate_fixture(args);
        }
        if (args.command == "bench") {
            return bench(args);
        }
        if (args.command == "stage1-train") {
            return stage1_train(args);
        }
        if (args.command == "stage1-heldout-eval") {
            return stage1_heldout_eval(args);
        }
        if (args.command == "richter-dampening") {
            return richter_dampening(args);
        }
        if (args.command == "sensory-diagnostics") {
            return sensory_diagnostics(args);
        }
        throw std::runtime_error("unknown command: " + args.command);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        print_usage(std::cerr);
        return 1;
    }
}
