#include "modelSpec.h"
#include "runtime/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "v1Biology.h"
#include "v1TwoLayerConfig.h"

class V1LIF : public GeNN::NeuronModels::Base {
public:
    DECLARE_SNIPPET(V1LIF);

    SET_SIM_CODE(
        "if (RefracTime <= 0.0) {\n"
        "  scalar alpha = ((Isyn + Ioffset + Iext) * Rmembrane) + Vrest;\n"
        "  V = alpha - (ExpTC * (alpha - V));\n"
        "}\n"
        "else {\n"
        "  RefracTime -= dt;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= Vthresh");

    SET_RESET_CODE(
        "V = Vreset;\n"
        "RefracTime = TauRefrac;\n");

    SET_PARAMS({
        "C",
        "TauM",
        "Vrest",
        "Vreset",
        "Vthresh",
        "Ioffset",
        "TauRefrac"
    });

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const GeNN::ParamValues &pars, double dt) { return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rmembrane", [](const GeNN::ParamValues &pars, double) { return pars.at("TauM").cast<double>() / pars.at("C").cast<double>(); }}
    });

    SET_VARS({
        {"V", "scalar"},
        {"RefracTime", "scalar"},
        {"Iext", "scalar"}
    });

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_SNIPPET(V1LIF);

class HomeostaticInhibitory : public GeNN::WeightUpdateModels::Base {
public:
    DECLARE_SNIPPET(HomeostaticInhibitory);

    SET_PARAMS({
        "TauPre",
        "TauPost",
        "Eta",
        "TargetHz",
        "Wmin",
        "Wmax"
    });

    SET_DERIVED_PARAMS({
        {"PreDecay", [](const GeNN::ParamValues &pars, double dt) { return std::exp(-dt / pars.at("TauPre").cast<double>()); }},
        {"PostDecay", [](const GeNN::ParamValues &pars, double dt) { return std::exp(-dt / pars.at("TauPost").cast<double>()); }},
    });

    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});

    SET_PRE_DYNAMICS_CODE("preTrace *= PreDecay;\n");
    SET_POST_DYNAMICS_CODE("postTrace *= PostDecay;\n");

    SET_PRE_SPIKE_CODE("preTrace += 1.0;\n");
    SET_POST_SPIKE_CODE("postTrace += 1.0;\n");

    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "const scalar targetTrace = (TargetHz * TauPost) / 1000.0;\n"
        "const scalar newWeight = g - (Eta * (postTrace - targetTrace));\n"
        "g = fmin(Wmax, fmax(Wmin, newWeight));\n");

    SET_POST_SPIKE_SYN_CODE(
        "const scalar newWeight = g - (Eta * preTrace);\n"
        "g = fmin(Wmax, fmax(Wmin, newWeight));\n");
};
IMPLEMENT_SNIPPET(HomeostaticInhibitory);

class LocalPatch : public GeNN::InitSparseConnectivitySnippet::Base {
public:
    DECLARE_SNIPPET(LocalPatch);

    SET_ROW_BUILD_CODE(
        "const unsigned int preSite = id_pre / preNeuronsPerSite;\n"
        "const unsigned int preX = preSite % preSide;\n"
        "const unsigned int preY = preSite / preSide;\n"
        "for(int dy = -(int)radius; dy <= (int)radius; dy++) {\n"
        "    const int postY = (int)preY + dy;\n"
        "    if(postY < 0 || postY >= (int)postSide) {\n"
        "        continue;\n"
        "    }\n"
        "    for(int dx = -(int)radius; dx <= (int)radius; dx++) {\n"
        "        const int postX = (int)preX + dx;\n"
        "        if(postX < 0 || postX >= (int)postSide) {\n"
        "            continue;\n"
        "        }\n"
        "        const unsigned int postSite = ((unsigned int)postY * postSide) + (unsigned int)postX;\n"
        "        for(unsigned int postCell = 0; postCell < postNeuronsPerSite; postCell++) {\n"
        "            const unsigned int target = (postSite * postNeuronsPerSite) + postCell;\n"
        "            if(excludeSelf != 0 && target == id_pre) {\n"
        "                continue;\n"
        "            }\n"
        "            addSynapse(target);\n"
        "        }\n"
        "    }\n"
        "}\n");

    SET_PARAMS({
        {"preSide", "unsigned int"},
        {"preNeuronsPerSite", "unsigned int"},
        {"postSide", "unsigned int"},
        {"postNeuronsPerSite", "unsigned int"},
        {"radius", "unsigned int"},
        {"excludeSelf", "unsigned int"}
    });

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int postNeuronsPerSite = pars.at("postNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * postNeuronsPerSite;
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int preNeuronsPerSite = pars.at("preNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * preNeuronsPerSite;
        });
};
IMPLEMENT_SNIPPET(LocalPatch);

class LocalIntersitePatch : public GeNN::InitSparseConnectivitySnippet::Base {
public:
    DECLARE_SNIPPET(LocalIntersitePatch);

    SET_ROW_BUILD_CODE(
        "const unsigned int preSite = id_pre / preNeuronsPerSite;\n"
        "const unsigned int preX = preSite % preSide;\n"
        "const unsigned int preY = preSite / preSide;\n"
        "for(int dy = -(int)radius; dy <= (int)radius; dy++) {\n"
        "    const int postY = (int)preY + dy;\n"
        "    if(postY < 0 || postY >= (int)postSide) {\n"
        "        continue;\n"
        "    }\n"
        "    for(int dx = -(int)radius; dx <= (int)radius; dx++) {\n"
        "        const int postX = (int)preX + dx;\n"
        "        if(postX < 0 || postX >= (int)postSide) {\n"
        "            continue;\n"
        "        }\n"
        "        const unsigned int postSite = ((unsigned int)postY * postSide) + (unsigned int)postX;\n"
        "        if(postSite == preSite) {\n"
        "            continue;\n"
        "        }\n"
        "        for(unsigned int postCell = 0; postCell < postNeuronsPerSite; postCell++) {\n"
        "            addSynapse((postSite * postNeuronsPerSite) + postCell);\n"
        "        }\n"
        "    }\n"
        "}\n");

    SET_PARAMS({
        {"preSide", "unsigned int"},
        {"preNeuronsPerSite", "unsigned int"},
        {"postSide", "unsigned int"},
        {"postNeuronsPerSite", "unsigned int"},
        {"radius", "unsigned int"}
    });

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int postNeuronsPerSite = pars.at("postNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * postNeuronsPerSite;
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int preNeuronsPerSite = pars.at("preNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * preNeuronsPerSite;
        });
};
IMPLEMENT_SNIPPET(LocalIntersitePatch);

class SparseDistancePatch : public GeNN::InitSparseConnectivitySnippet::Base {
public:
    DECLARE_SNIPPET(SparseDistancePatch);

    SET_ROW_BUILD_CODE(
        "const unsigned int preSite = id_pre / preNeuronsPerSite;\n"
        "const unsigned int preX = preSite % preSide;\n"
        "const unsigned int preY = preSite / preSide;\n"
        "for(int dy = -(int)radius; dy <= (int)radius; dy++) {\n"
        "    const int postY = (int)preY + dy;\n"
        "    if(postY < 0 || postY >= (int)postSide) {\n"
        "        continue;\n"
        "    }\n"
        "    for(int dx = -(int)radius; dx <= (int)radius; dx++) {\n"
        "        const int postX = (int)preX + dx;\n"
        "        if(postX < 0 || postX >= (int)postSide) {\n"
        "            continue;\n"
        "        }\n"
        "        const scalar distanceSq = (scalar)((dx * dx) + (dy * dy));\n"
        "        const scalar probability = peakProbability * exp(-distanceSq / (2.0 * distanceSigmaSq));\n"
        "        const unsigned int postSite = ((unsigned int)postY * postSide) + (unsigned int)postX;\n"
        "        for(unsigned int postCell = 0; postCell < postNeuronsPerSite; postCell++) {\n"
        "            const unsigned int target = (postSite * postNeuronsPerSite) + postCell;\n"
        "            if(excludeSelf != 0 && target == id_pre) {\n"
        "                continue;\n"
        "            }\n"
        "            unsigned int hash = (id_pre + 1u) * 1103515245u;\n"
        "            hash ^= (target + 1u) * 12345u;\n"
        "            hash ^= (hash / 65536u);\n"
        "            const scalar sample = ((scalar)(hash & 0x00FFFFFFu)) / 16777216.0;\n"
        "            if(sample < probability) {\n"
        "                addSynapse(target);\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n");

    SET_PARAMS({
        {"preSide", "unsigned int"},
        {"preNeuronsPerSite", "unsigned int"},
        {"postSide", "unsigned int"},
        {"postNeuronsPerSite", "unsigned int"},
        {"radius", "unsigned int"},
        {"excludeSelf", "unsigned int"},
        {"peakProbability", "scalar"},
        {"distanceSigmaSq", "scalar"}
    });

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int postNeuronsPerSite = pars.at("postNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * postNeuronsPerSite;
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int preNeuronsPerSite = pars.at("preNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * preNeuronsPerSite;
        });
};
IMPLEMENT_SNIPPET(SparseDistancePatch);

class OrientationBiasedPatch : public GeNN::InitSparseConnectivitySnippet::Base {
public:
    DECLARE_SNIPPET(OrientationBiasedPatch);

    SET_ROW_BUILD_CODE(
        "const unsigned int preSite = id_pre / preNeuronsPerSite;\n"
        "const unsigned int preX = preSite % preSide;\n"
        "const unsigned int preY = preSite / preSide;\n"
        "const unsigned int neutralDensityMatch = (biasStrength == 0.0 && neutralDensityScale > 0.0) ? 1u : 0u;\n"
        "double preOri = 0.0;\n"
        "if(!neutralDensityMatch) {\n"
        "    const double preNormX = (((double)preX) + 0.5) / ((double)preSide);\n"
        "    const double preNormY = (((double)preY) + 0.5) / ((double)preSide);\n"
        "    const double preFieldX = sin(6.283185307179586 * preNormX) + (0.60 * cos(6.283185307179586 * preNormY)) + (0.35 * sin(6.283185307179586 * (preNormX + preNormY)));\n"
        "    const double preFieldY = cos(6.283185307179586 * preNormX) - (0.60 * sin(6.283185307179586 * preNormY)) + (0.35 * cos(6.283185307179586 * (preNormX - preNormY)));\n"
        "    preOri = 0.5 * atan2(preFieldY, preFieldX);\n"
        "    if(preOri < 0.0) {\n"
        "        preOri += 3.14159265358979323846;\n"
        "    }\n"
        "}\n"
        "for(int dy = -(int)radius; dy <= (int)radius; dy++) {\n"
        "    const int postY = (int)preY + dy;\n"
        "    if(postY < 0 || postY >= (int)postSide) {\n"
        "        continue;\n"
        "    }\n"
        "    for(int dx = -(int)radius; dx <= (int)radius; dx++) {\n"
        "        const int postX = (int)preX + dx;\n"
        "        if(postX < 0 || postX >= (int)postSide) {\n"
        "            continue;\n"
        "        }\n"
        "        const unsigned int postSite = ((unsigned int)postY * postSide) + (unsigned int)postX;\n"
        "        const double manhattanDistance = (double)(((dx < 0) ? -dx : dx) + ((dy < 0) ? -dy : dy));\n"
        "        double connectionProbability = 0.0;\n"
        "        if(neutralDensityMatch) {\n"
        "            connectionProbability = minProbability + ((1.0 - minProbability) * 0.5);\n"
        "            connectionProbability -= distancePenalty * manhattanDistance;\n"
        "            if(connectionProbability < minProbability) {\n"
        "                connectionProbability = minProbability;\n"
        "            }\n"
        "            connectionProbability *= neutralDensityScale;\n"
        "        }\n"
        "        else {\n"
        "            const double postNormX = (((double)postX) + 0.5) / ((double)postSide);\n"
        "            const double postNormY = (((double)postY) + 0.5) / ((double)postSide);\n"
        "            const double postFieldX = sin(6.283185307179586 * postNormX) + (0.60 * cos(6.283185307179586 * postNormY)) + (0.35 * sin(6.283185307179586 * (postNormX + postNormY)));\n"
        "            const double postFieldY = cos(6.283185307179586 * postNormX) - (0.60 * sin(6.283185307179586 * postNormY)) + (0.35 * cos(6.283185307179586 * (postNormX - postNormY)));\n"
        "            double postOri = 0.5 * atan2(postFieldY, postFieldX);\n"
        "            if(postOri < 0.0) {\n"
        "                postOri += 3.14159265358979323846;\n"
        "            }\n"
        "            double delta = fabs(preOri - postOri);\n"
        "            delta = fmin(delta, 3.14159265358979323846 - delta);\n"
        "            const double similarity = 0.5 * (1.0 + cos(2.0 * delta));\n"
        "            const double biasedSimilarity = ((1.0 - biasStrength) * 0.5) + (biasStrength * similarity);\n"
        "            connectionProbability = minProbability + ((1.0 - minProbability) * biasedSimilarity);\n"
        "            connectionProbability -= distancePenalty * manhattanDistance;\n"
        "        }\n"
        "        if(connectionProbability < minProbability) {\n"
        "            connectionProbability = minProbability;\n"
        "        }\n"
        "        if(connectionProbability > 1.0) {\n"
        "            connectionProbability = 1.0;\n"
        "        }\n"
        "        for(unsigned int postCell = 0; postCell < postNeuronsPerSite; postCell++) {\n"
        "            const unsigned int target = (postSite * postNeuronsPerSite) + postCell;\n"
        "            unsigned int hash = ((id_pre + 1u) * 747796405u) ^ ((target + 1u) * 2891336453u);\n"
        "            hash ^= (hash / 65536u);\n"
        "            hash *= 2246822519u;\n"
        "            hash ^= (hash / 8192u);\n"
        "            hash *= 3266489917u;\n"
        "            hash ^= (hash / 65536u);\n"
        "            const double unit = ((double)(hash & 0x00FFFFFFu)) * (1.0 / 16777216.0);\n"
        "            if(unit >= connectionProbability) {\n"
        "                continue;\n"
        "            }\n"
        "            addSynapse(target);\n"
        "        }\n"
        "    }\n"
        "}\n");

    SET_PARAMS({
        {"preSide", "unsigned int"},
        {"preNeuronsPerSite", "unsigned int"},
        {"postSide", "unsigned int"},
        {"postNeuronsPerSite", "unsigned int"},
        {"radius", "unsigned int"},
        {"minProbability", "scalar"},
        {"biasStrength", "scalar"},
        {"neutralDensityScale", "scalar"},
        {"distancePenalty", "scalar"}
    });

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int postNeuronsPerSite = pars.at("postNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * postNeuronsPerSite;
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const GeNN::ParamValues &pars) {
            const unsigned int radius = pars.at("radius").cast<unsigned int>();
            const unsigned int preNeuronsPerSite = pars.at("preNeuronsPerSite").cast<unsigned int>();
            const unsigned int patchWidth = (2u * radius) + 1u;
            return patchWidth * patchWidth * preNeuronsPerSite;
        });
};
IMPLEMENT_SNIPPET(OrientationBiasedPatch);

namespace {

constexpr char kDefaultOutputPrefix[] = "/scratch/proj/v1_snn_l4_l23/genn/v1_experiment";
constexpr unsigned int kDefaultOrientationCount = 12;
constexpr double kDefaultTrialMs = 250.0;
constexpr double kDefaultSettleMs = 50.0;
constexpr unsigned int kDefaultTrainingEpochs = 1;
constexpr unsigned int kDefaultRecurrentConsolidationEpochs = 3;
constexpr double kDefaultStdpAplus = 0.0001;
constexpr double kDefaultStdpAminus = 0.0000875;
constexpr unsigned int kDefaultL23EEPlasticityEnabled = 1;
constexpr double kDefaultL23EEStdpAplus = 0.000100;
constexpr double kDefaultL23EEStdpAminus = 0.000100;
constexpr unsigned int kDefaultL23PVHomeostaticEnabled = 1;
constexpr unsigned int kDefaultL23SOMHomeostaticEnabled = 1;
constexpr double kDefaultL23PVHomeostaticEta = 0.000020;
constexpr double kDefaultL23SOMHomeostaticEta = 0.000050;
constexpr double kDefaultL23PVHomeostaticTargetHz = 25.0;
constexpr double kDefaultL23SOMHomeostaticTargetHz = 5.0;
constexpr double kDefaultL23PVGate = 0.0;
constexpr double kDefaultL23SOMGate = 0.18;
constexpr double kDefaultL23VIPGate = 0.0;
constexpr double kDefaultL23SOMOutputScale = 1.0;
constexpr double kDefaultL23SOMContextOutputScale = 1.0;
constexpr double kDefaultL23PVContextOutputScale = 1.0;
constexpr double kDefaultL23EEContextOutputScale = 1.0;
constexpr double kDefaultCenterStimulusRadiusSites = 2.0;
constexpr double kDefaultBroadStimulusRadiusSites = 3.0;
constexpr char kDefaultSizeTuningRadiiSites[] = "0.5,1,2,3,4,6";
constexpr unsigned int kDefaultBlankRepeatCount = 4;
constexpr char kDefaultContrastSweepValues[] = "0.5,1.0";
constexpr unsigned int kDefaultRecurrentOnlyConsolidationEpochs = 18;
constexpr char kTrainingGratingModeLegacy[] = "legacy";
constexpr char kTrainingGratingModePhaseDrift[] = "phase_drift";
constexpr double kDefaultL4L23OrientationNeutralProbabilityScale = 1.27;
constexpr unsigned int kOrientationContextConditionCount = 5;
constexpr unsigned int kOrientationContextCenterOnly = 0;
constexpr unsigned int kOrientationContextSameSurround = 1;
constexpr unsigned int kOrientationContextOrthSurround = 2;
constexpr unsigned int kOrientationContextSurroundSameOnly = 3;
constexpr unsigned int kOrientationContextSurroundOrthOnly = 4;
constexpr double kL23ERecurrentPeakProbability = 0.12;
constexpr double kL23ERecurrentDistanceSigmaSq = 3.0;

constexpr double kStdpTauPlusMs = 20.0;
constexpr double kStdpTauMinusMs = 20.0;
constexpr double kL23EEStdpTauPlusMs = 60.0;
constexpr double kL23EEStdpTauMinusMs = 60.0;
constexpr double kStdpWeightMin = 0.0005;
constexpr double kStdpWeightMax = 0.020;
constexpr double kL23EEStdpWeightMin = 0.0010;
constexpr double kL23EEStdpWeightMax = 0.0100;
constexpr double kHomeostaticTraceTauMs = 20.0;
constexpr double kL23PVToL23EWeightMin = -0.0500;
constexpr double kL23PVToL23EWeightMax = -0.0020;
constexpr double kL23SOMToL23EWeightMin = -0.0400;
constexpr double kL23SOMToL23EWeightMax = -0.0010;

struct TrialWindow {
    double orientation_rad;
    double phase_rad;
    double start_ms;
    double measure_start_ms;
    double end_ms;
};

struct PopulationSiteMetrics {
    unsigned int site_id = 0;
    unsigned int x = 0;
    unsigned int y = 0;
    double map_pref_rad = 0.0;
    double measured_pref_rad = 0.0;
    double osi = 0.0;
    double mean_rate_hz = 0.0;
    std::vector<double> rates_hz;
};

struct CellTuningMetrics {
    unsigned int cell_id = 0;
    unsigned int site_id = 0;
    double site_pref_rad = 0.0;
    double measured_pref_rad = 0.0;
    double mean_rate_hz = 0.0;
    double peak_rate_hz = 0.0;
    double osi = 0.0;
    std::vector<double> rates_hz;
};

struct MultiPhaseCellTuningMetrics {
    unsigned int cell_id = 0;
    unsigned int site_id = 0;
    double site_pref_rad = 0.0;
    double best_orientation_rad = 0.0;
    double best_phase_rad = 0.0;
    double peak_rate_any_phase_hz = 0.0;
    double mean_rate_hz = 0.0;
    double phase_pooled_osi = 0.0;
    std::vector<double> phase_mean_rates_hz;
};

struct SweepResult {
    std::string label;
    std::vector<double> orientations_rad;
    std::vector<PopulationSiteMetrics> l4_sites;
    std::vector<PopulationSiteMetrics> l23_sites;
    double l4_median_osi = 0.0;
    double l23_median_osi = 0.0;
    double l4_median_map_error_deg = 0.0;
};

struct WeightStats {
    std::size_t count = 0;
    double min = 0.0;
    double mean = 0.0;
    double max = 0.0;
};

struct PopulationRateSummary {
    std::string name;
    double baseline_mean_rate_hz = 0.0;
    double post_mean_rate_hz = 0.0;
};

struct NamedWeightStats {
    std::string name;
    WeightStats before;
    WeightStats after;
};

struct L4IntersiteConfig {
    bool enabled = false;
    unsigned int radius = v1_genn::kL4IntersiteRadius;
    double weight_scale = v1_genn::kL4IntersiteWeightScale;
    double l4ee_scale = v1_genn::kL4IntersiteWeightScale;
    double l4e_to_l4pv_scale = v1_genn::kL4IntersiteWeightScale;
    double l4pv_to_l4e_scale = v1_genn::kL4IntersiteWeightScale;
};

struct TrainingGratingConfig {
    bool phase_drift_enabled = false;
    std::string mode = kTrainingGratingModeLegacy;
    unsigned int phase_count = 1;
    bool counterbalance_direction = false;
};

struct L4L23OrientationConfig {
    double bias_strength = v1_genn::kOrientationSoftBiasStrength;
    bool neutral_density_match_enabled = false;
    double neutral_probability_scale = kDefaultL4L23OrientationNeutralProbabilityScale;
};

struct L23EELognormalInitConfig {
    bool enabled = false;
    double sigma = 0.37;
};

struct OrientationContextAssayConfig {
    bool enabled = false;
    double center_radius_sites = kDefaultCenterStimulusRadiusSites;
    double broad_radius_sites = kDefaultBroadStimulusRadiusSites;
    double surround_inner_radius_sites = kDefaultCenterStimulusRadiusSites;
};

struct SensoryAssayConfig {
    bool enabled = false;
    unsigned int blank_repeat_count = 0;
    double contrast_radius_sites = kDefaultCenterStimulusRadiusSites;
    std::vector<double> contrasts;
};

struct ConnectivityStats {
    std::size_t edge_count = 0;
    double mean_distance_sites = 0.0;
    double max_distance_sites = 0.0;
    double same_site_fraction = 0.0;
    double beyond_radius_fraction = 0.0;
};

struct ContextValidationSummary {
    std::string condition;
    double l23e_mean_rate_hz = 0.0;
    double l23pv_mean_rate_hz = 0.0;
    double l23som_mean_rate_hz = 0.0;
};

struct RetinotopicContextMetrics {
    unsigned int validation_site_id = 0;
    PopulationSiteMetrics center_l23e;
    PopulationSiteMetrics center_l23pv;
    PopulationSiteMetrics center_l23som;
    PopulationSiteMetrics broad_l23e;
    PopulationSiteMetrics broad_l23pv;
    PopulationSiteMetrics broad_l23som;
};

struct RetinotopicSizeMetrics {
    unsigned int validation_site_id = 0;
    PopulationSiteMetrics l4e;
    PopulationSiteMetrics l23e;
    PopulationSiteMetrics l23pv;
    PopulationSiteMetrics l23som;
};

struct ValidationSiteConfig {
    std::vector<unsigned int> site_ids;
    std::vector<unsigned int> aperture_center_sites;
    bool include_validation_site_id = false;
};

struct ValidationTrialSet {
    unsigned int site_id = 0;
    unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max();
    std::vector<TrialWindow> center_trials;
    std::vector<TrialWindow> broad_trials;
    std::vector<TrialWindow> size_trials;
};

struct OrientationContextTrialSet {
    unsigned int validation_site_id = 0;
    unsigned int site_id = 0;
    unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max();
    double preferred_orientation_rad = 0.0;
    double orthogonal_orientation_rad = 0.0;
    std::array<TrialWindow, kOrientationContextConditionCount> trials{};
};

struct OrientationContextSiteMetrics {
    unsigned int validation_site_id = 0;
    unsigned int site_id = 0;
    unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max();
    double preferred_orientation_rad = 0.0;
    double orthogonal_orientation_rad = 0.0;
    std::array<double, kOrientationContextConditionCount> aperture_radius_sites{};
    std::array<double, kOrientationContextConditionCount> inner_radius_sites{};
    std::array<double, kOrientationContextConditionCount> l4e_rates_hz{};
    std::array<double, kOrientationContextConditionCount> l23e_rates_hz{};
    std::array<double, kOrientationContextConditionCount> l23pv_rates_hz{};
    std::array<double, kOrientationContextConditionCount> l23som_rates_hz{};
};

struct ContrastTrialRecord {
    unsigned int validation_site_id = 0;
    unsigned int site_id = 0;
    unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max();
    double contrast = 1.0;
    double orientation_rad = 0.0;
    double aperture_radius_sites = kDefaultCenterStimulusRadiusSites;
    TrialWindow trial{};
};

L4L23OrientationConfig getL4L23OrientationConfig();
L23EELognormalInitConfig getL23EELognormalInitConfig();
double getOrientationSoftBiasStrength();

GeNN::ParamValues makeLIFParameters(const v1_genn::LIFParameters &params)
{
    return {
        {"C", params.c},
        {"TauM", params.tau_m_ms},
        {"Vrest", params.v_rest_mv},
        {"Vreset", params.v_reset_mv},
        {"Vthresh", params.v_thresh_mv},
        {"Ioffset", params.i_offset_na},
        {"TauRefrac", params.tau_refrac_ms},
    };
}

GeNN::VarValues makeLIFVariables(const v1_genn::LIFParameters &params, const GeNN::InitVarSnippet::Init &external_drive)
{
    return {
        {"V", params.v_rest_mv},
        {"RefracTime", 0.0},
        {"Iext", external_drive},
    };
}

GeNN::ParamValues makePatchParameters(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius,
    bool exclude_self)
{
    return {
        {"preSide", v1_genn::kSheetSide},
        {"preNeuronsPerSite", pre_neurons_per_site},
        {"postSide", v1_genn::kSheetSide},
        {"postNeuronsPerSite", post_neurons_per_site},
        {"radius", radius},
        {"excludeSelf", exclude_self ? 1u : 0u},
    };
}

GeNN::ParamValues makeIntersitePatchParameters(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius)
{
    return {
        {"preSide", v1_genn::kSheetSide},
        {"preNeuronsPerSite", pre_neurons_per_site},
        {"postSide", v1_genn::kSheetSide},
        {"postNeuronsPerSite", post_neurons_per_site},
        {"radius", radius},
    };
}

GeNN::ParamValues makeOrientationBiasedPatchParameters(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius)
{
    const L4L23OrientationConfig config = getL4L23OrientationConfig();
    return {
        {"preSide", v1_genn::kSheetSide},
        {"preNeuronsPerSite", pre_neurons_per_site},
        {"postSide", v1_genn::kSheetSide},
        {"postNeuronsPerSite", post_neurons_per_site},
        {"radius", radius},
        {"minProbability", v1_genn::kOrientationSoftProbabilityFloor},
        {"biasStrength", config.bias_strength},
        {"neutralDensityScale", (config.neutral_density_match_enabled && config.bias_strength == 0.0)
            ? config.neutral_probability_scale : 0.0},
        {"distancePenalty", v1_genn::kOrientationDistancePenalty},
    };
}

GeNN::ParamValues makeSparseDistancePatchParameters(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius,
    bool exclude_self,
    double peak_probability,
    double distance_sigma_sq)
{
    return {
        {"preSide", v1_genn::kSheetSide},
        {"preNeuronsPerSite", pre_neurons_per_site},
        {"postSide", v1_genn::kSheetSide},
        {"postNeuronsPerSite", post_neurons_per_site},
        {"radius", radius},
        {"excludeSelf", exclude_self ? 1u : 0u},
        {"peakProbability", peak_probability},
        {"distanceSigmaSq", distance_sigma_sq},
    };
}

GeNN::ParamValues makeHomeostaticInhibitoryParameters(double target_hz, double wmin, double wmax)
{
    return {
        {"TauPre", kHomeostaticTraceTauMs},
        {"TauPost", kHomeostaticTraceTauMs},
        {"Eta", 0.0},
        {"TargetHz", target_hz},
        {"Wmin", wmin},
        {"Wmax", wmax},
    };
}

void addLocalProjection(
    GeNN::ModelSpec &model,
    const std::string &name,
    GeNN::NeuronGroup *source,
    GeNN::NeuronGroup *target,
    double weight,
    double tau_ms,
    const GeNN::ParamValues &patch_params)
{
    model.addSynapsePopulation(
        name,
        GeNN::SynapseMatrixType::SPARSE,
        source,
        target,
        GeNN::initWeightUpdate<GeNN::WeightUpdateModels::StaticPulse>({}, {{"g", weight}}),
        GeNN::initPostsynaptic<GeNN::PostsynapticModels::ExpCurr>({{"tau", tau_ms}}),
        GeNN::initConnectivity<LocalPatch>(patch_params));
}

void addLocalIntersiteProjection(
    GeNN::ModelSpec &model,
    const std::string &name,
    GeNN::NeuronGroup *source,
    GeNN::NeuronGroup *target,
    double weight,
    double tau_ms,
    const GeNN::ParamValues &patch_params)
{
    model.addSynapsePopulation(
        name,
        GeNN::SynapseMatrixType::SPARSE,
        source,
        target,
        GeNN::initWeightUpdate<GeNN::WeightUpdateModels::StaticPulse>({}, {{"g", weight}}),
        GeNN::initPostsynaptic<GeNN::PostsynapticModels::ExpCurr>({{"tau", tau_ms}}),
        GeNN::initConnectivity<LocalIntersitePatch>(patch_params));
}

GeNN::SynapseGroup *addPlasticLocalProjection(
    GeNN::ModelSpec &model,
    const std::string &name,
    GeNN::NeuronGroup *source,
    GeNN::NeuronGroup *target,
    double initial_weight,
    double tau_ms,
    double wmin,
    double wmax,
    const GeNN::ParamValues &patch_params)
{
    GeNN::SynapseGroup *synapse_group = model.addSynapsePopulation(
        name,
        GeNN::SynapseMatrixType::SPARSE,
        source,
        target,
        GeNN::initWeightUpdate<GeNN::WeightUpdateModels::STDP>(
            {
                {"tauPlus", kStdpTauPlusMs},
                {"tauMinus", kStdpTauMinusMs},
                {"Aplus", 0.0},
                {"Aminus", 0.0},
                {"Wmin", wmin},
                {"Wmax", wmax},
            },
            {{"g", initial_weight}}),
        GeNN::initPostsynaptic<GeNN::PostsynapticModels::ExpCurr>({{"tau", tau_ms}}),
        GeNN::initConnectivity<LocalPatch>(patch_params));
    synapse_group->setWUParamDynamic("Aplus", true);
    synapse_group->setWUParamDynamic("Aminus", true);
    return synapse_group;
}

GeNN::SynapseGroup *addPlasticSparseDistanceProjection(
    GeNN::ModelSpec &model,
    const std::string &name,
    GeNN::NeuronGroup *source,
    GeNN::NeuronGroup *target,
    double initial_weight,
    double tau_ms,
    double stdp_tau_plus_ms,
    double stdp_tau_minus_ms,
    double wmin,
    double wmax,
    const GeNN::ParamValues &patch_params)
{
    GeNN::SynapseGroup *synapse_group = model.addSynapsePopulation(
        name,
        GeNN::SynapseMatrixType::SPARSE,
        source,
        target,
        GeNN::initWeightUpdate<GeNN::WeightUpdateModels::STDP>(
            {
                {"tauPlus", stdp_tau_plus_ms},
                {"tauMinus", stdp_tau_minus_ms},
                {"Aplus", 0.0},
                {"Aminus", 0.0},
                {"Wmin", wmin},
                {"Wmax", wmax},
            },
            {{"g", initial_weight}}),
        GeNN::initPostsynaptic<GeNN::PostsynapticModels::ExpCurr>({{"tau", tau_ms}}),
        GeNN::initConnectivity<SparseDistancePatch>(patch_params));
    synapse_group->setWUParamDynamic("Aplus", true);
    synapse_group->setWUParamDynamic("Aminus", true);
    return synapse_group;
}

GeNN::SynapseGroup *addPlasticOrientationBiasedProjection(
    GeNN::ModelSpec &model,
    const std::string &name,
    GeNN::NeuronGroup *source,
    GeNN::NeuronGroup *target,
    double initial_weight,
    double tau_ms,
    const GeNN::ParamValues &patch_params)
{
    GeNN::SynapseGroup *synapse_group = model.addSynapsePopulation(
        name,
        GeNN::SynapseMatrixType::SPARSE,
        source,
        target,
        GeNN::initWeightUpdate<GeNN::WeightUpdateModels::STDP>(
            {
                {"tauPlus", kStdpTauPlusMs},
                {"tauMinus", kStdpTauMinusMs},
                {"Aplus", 0.0},
                {"Aminus", 0.0},
                {"Wmin", kStdpWeightMin},
                {"Wmax", kStdpWeightMax},
            },
            {{"g", initial_weight}}),
        GeNN::initPostsynaptic<GeNN::PostsynapticModels::ExpCurr>({{"tau", tau_ms}}),
        GeNN::initConnectivity<OrientationBiasedPatch>(patch_params));
    synapse_group->setWUParamDynamic("Aplus", true);
    synapse_group->setWUParamDynamic("Aminus", true);
    return synapse_group;
}

GeNN::SynapseGroup *addHomeostaticInhibitoryProjection(
    GeNN::ModelSpec &model,
    const std::string &name,
    GeNN::NeuronGroup *source,
    GeNN::NeuronGroup *target,
    double initial_weight,
    double tau_ms,
    const GeNN::ParamValues &weight_params,
    const GeNN::ParamValues &patch_params)
{
    GeNN::SynapseGroup *synapse_group = model.addSynapsePopulation(
        name,
        GeNN::SynapseMatrixType::SPARSE,
        source,
        target,
        GeNN::initWeightUpdate<HomeostaticInhibitory>(
            weight_params,
            {{"g", initial_weight}},
            {{"preTrace", 0.0}},
            {{"postTrace", 0.0}}),
        GeNN::initPostsynaptic<GeNN::PostsynapticModels::ExpCurr>({{"tau", tau_ms}}),
        GeNN::initConnectivity<LocalPatch>(patch_params));
    synapse_group->setWUParamDynamic("Eta", true);
    synapse_group->setWUParamDynamic("TargetHz", true);
    return synapse_group;
}

std::string getEnvOrDefault(const char *name, const char *default_value)
{
    const char *value = std::getenv(name);
    if(value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value;
}

double getEnvDoubleOrDefault(const char *name, double default_value)
{
    const char *value = std::getenv(name);
    if(value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if(end == value || *end != '\0') {
        throw std::runtime_error(std::string("Invalid numeric value for ") + name + ": " + value);
    }
    return parsed;
}

unsigned int getEnvUnsignedOrDefault(const char *name, unsigned int default_value)
{
    const char *value = std::getenv(name);
    if(value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if(end == value || *end != '\0' || parsed > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error(std::string("Invalid unsigned integer value for ") + name + ": " + value);
    }
    return static_cast<unsigned int>(parsed);
}

std::string trimWhitespace(const std::string &value)
{
    const std::string whitespace = " \t\r\n";
    const std::size_t first = value.find_first_not_of(whitespace);
    if(first == std::string::npos) {
        return "";
    }
    const std::size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1u);
}

std::vector<double> getEnvDoubleListOrDefault(const char *name, const char *default_value)
{
    const char *env_value = std::getenv(name);
    const std::string raw = (env_value == nullptr || env_value[0] == '\0') ? default_value : env_value;

    std::vector<double> values;
    std::stringstream stream(raw);
    std::string token;
    while(std::getline(stream, token, ',')) {
        const std::string trimmed = trimWhitespace(token);
        if(trimmed.empty()) {
            throw std::runtime_error(std::string("Invalid comma-separated numeric list for ") + name + ": " + raw);
        }

        char *end = nullptr;
        const double parsed = std::strtod(trimmed.c_str(), &end);
        if(end == trimmed.c_str() || *end != '\0' || !std::isfinite(parsed)) {
            throw std::runtime_error(std::string("Invalid comma-separated numeric list for ") + name + ": " + raw);
        }
        values.push_back(parsed);
    }

    if(values.empty()) {
        throw std::runtime_error(std::string("At least one value is required for ") + name + ".");
    }
    return values;
}

std::vector<unsigned int> getEnvUnsignedListOrEmpty(const char *name)
{
    const char *env_value = std::getenv(name);
    if(env_value == nullptr || env_value[0] == '\0') {
        return {};
    }

    std::vector<unsigned int> values;
    std::stringstream stream(env_value);
    std::string token;
    while(std::getline(stream, token, ',')) {
        const std::string trimmed = trimWhitespace(token);
        if(trimmed.empty()) {
            throw std::runtime_error(std::string("Invalid comma-separated unsigned integer list for ") + name + ": " + env_value);
        }

        char *end = nullptr;
        const unsigned long parsed = std::strtoul(trimmed.c_str(), &end, 10);
        if(end == trimmed.c_str() || *end != '\0' || parsed > std::numeric_limits<unsigned int>::max()) {
            throw std::runtime_error(std::string("Invalid comma-separated unsigned integer list for ") + name + ": " + env_value);
        }
        values.push_back(static_cast<unsigned int>(parsed));
    }

    if(values.empty()) {
        throw std::runtime_error(std::string("At least one value is required for ") + name + ".");
    }
    return values;
}

double getOrientationSoftBiasStrength()
{
    const char *strict_env = std::getenv("V1_L4_L23_ORIENTATION_BIAS_STRENGTH");
    const double strength = (strict_env != nullptr && strict_env[0] != '\0')
        ? getEnvDoubleOrDefault("V1_L4_L23_ORIENTATION_BIAS_STRENGTH", v1_genn::kOrientationSoftBiasStrength)
        : getEnvDoubleOrDefault("V1_FF_ORIENTATION_BIAS_STRENGTH", v1_genn::kOrientationSoftBiasStrength);
    if(strength < 0.0 || strength > 1.0) {
        throw std::runtime_error("V1_L4_L23_ORIENTATION_BIAS_STRENGTH must be in [0, 1].");
    }
    return strength;
}

L4L23OrientationConfig getL4L23OrientationConfig()
{
    L4L23OrientationConfig config;
    config.bias_strength = getOrientationSoftBiasStrength();
    config.neutral_density_match_enabled = getEnvUnsignedOrDefault(
        "V1_L4_L23_ORIENTATION_NEUTRAL_DENSITY_MATCH",
        0u) != 0u;
    config.neutral_probability_scale = getEnvDoubleOrDefault(
        "V1_L4_L23_ORIENTATION_NEUTRAL_PROBABILITY_SCALE",
        kDefaultL4L23OrientationNeutralProbabilityScale);

    if(config.neutral_probability_scale <= 0.0) {
        throw std::runtime_error("V1_L4_L23_ORIENTATION_NEUTRAL_PROBABILITY_SCALE must be positive.");
    }
    return config;
}

L23EELognormalInitConfig getL23EELognormalInitConfig()
{
    L23EELognormalInitConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_L23EE_LOGNORMAL_INIT", 0u) != 0u;
    config.sigma = getEnvDoubleOrDefault("V1_L23EE_LOGNORMAL_SIGMA", config.sigma);
    if(config.sigma < 0.0) {
        throw std::runtime_error("V1_L23EE_LOGNORMAL_SIGMA must be non-negative.");
    }
    return config;
}

L4IntersiteConfig getL4IntersiteConfig()
{
    L4IntersiteConfig config;
    config.enabled = getEnvUnsignedOrDefault(
        "V1_L4_INTERSITE_ENABLE",
        v1_genn::kL4IntersiteEnableDefault) != 0u;
    config.radius = getEnvUnsignedOrDefault(
        "V1_L4_INTERSITE_RADIUS",
        v1_genn::kL4IntersiteRadius);
    config.weight_scale = getEnvDoubleOrDefault(
        "V1_L4_INTERSITE_WEIGHT_SCALE",
        v1_genn::kL4IntersiteWeightScale);
    config.l4ee_scale = getEnvDoubleOrDefault(
        "V1_L4_INTERSITE_EE_SCALE",
        config.weight_scale);
    config.l4e_to_l4pv_scale = getEnvDoubleOrDefault(
        "V1_L4_INTERSITE_E_PV_SCALE",
        config.weight_scale);
    config.l4pv_to_l4e_scale = getEnvDoubleOrDefault(
        "V1_L4_INTERSITE_PV_E_SCALE",
        config.weight_scale);

    if(config.radius == 0u || config.radius >= v1_genn::kSheetSide) {
        throw std::runtime_error("V1_L4_INTERSITE_RADIUS must be in [1, V1_SHEET_SIDE).");
    }
    if(config.weight_scale < 0.0 || config.weight_scale > 1.0) {
        throw std::runtime_error("V1_L4_INTERSITE_WEIGHT_SCALE must be in [0, 1].");
    }
    if(config.l4ee_scale < 0.0 || config.l4ee_scale > 1.0
       || config.l4e_to_l4pv_scale < 0.0 || config.l4e_to_l4pv_scale > 1.0
       || config.l4pv_to_l4e_scale < 0.0 || config.l4pv_to_l4e_scale > 1.0) {
        throw std::runtime_error("L4 intersite per-projection scales must be in [0, 1].");
    }
    if(config.enabled
       && config.l4ee_scale == 0.0
       && config.l4e_to_l4pv_scale == 0.0
       && config.l4pv_to_l4e_scale == 0.0) {
        throw std::runtime_error("At least one L4 intersite projection scale must be positive when V1_L4_INTERSITE_ENABLE=1.");
    }
    return config;
}

TrainingGratingConfig getTrainingGratingConfig()
{
    TrainingGratingConfig config;
    config.mode = getEnvOrDefault("V1_TRAINING_GRATING_MODE", kTrainingGratingModeLegacy);
    const char *phase_count_env = std::getenv("V1_TRAINING_DRIFT_PHASE_COUNT");
    const bool explicit_phase_count = (phase_count_env != nullptr && std::string(phase_count_env).size() > 0u);
    config.phase_count = getEnvUnsignedOrDefault(
        "V1_TRAINING_DRIFT_PHASE_COUNT",
        config.mode == kTrainingGratingModePhaseDrift ? 4u : 1u);

    if(config.mode != kTrainingGratingModeLegacy && config.mode != kTrainingGratingModePhaseDrift) {
        throw std::runtime_error("V1_TRAINING_GRATING_MODE must be 'legacy' or 'phase_drift'.");
    }
    if(config.mode == kTrainingGratingModeLegacy && explicit_phase_count && config.phase_count > 1u) {
        config.mode = kTrainingGratingModePhaseDrift;
    }
    config.phase_drift_enabled = (config.mode == kTrainingGratingModePhaseDrift || config.phase_count > 1u);
    if(config.phase_drift_enabled && config.phase_count < 2u) {
        throw std::runtime_error("V1_TRAINING_DRIFT_PHASE_COUNT must be at least 2 when phase-drift training is enabled.");
    }
    if(!config.phase_drift_enabled) {
        config.phase_count = 1u;
    }
    config.counterbalance_direction = config.phase_drift_enabled;
    return config;
}

OrientationContextAssayConfig getOrientationContextAssayConfig(double default_broad_radius_sites)
{
    OrientationContextAssayConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_ORIENTATION_CONTEXT_ASSAY_ENABLE", 0u) != 0u;
    config.center_radius_sites = getEnvDoubleOrDefault(
        "V1_ORIENTATION_CONTEXT_CENTER_RADIUS_SITES",
        kDefaultCenterStimulusRadiusSites);
    config.broad_radius_sites = getEnvDoubleOrDefault(
        "V1_ORIENTATION_CONTEXT_BROAD_RADIUS_SITES",
        default_broad_radius_sites);
    config.surround_inner_radius_sites = getEnvDoubleOrDefault(
        "V1_ORIENTATION_CONTEXT_SURROUND_INNER_RADIUS_SITES",
        config.center_radius_sites);

    if(config.center_radius_sites <= 0.0) {
        throw std::runtime_error("V1_ORIENTATION_CONTEXT_CENTER_RADIUS_SITES must be positive.");
    }
    if(config.broad_radius_sites <= config.center_radius_sites) {
        throw std::runtime_error("V1_ORIENTATION_CONTEXT_BROAD_RADIUS_SITES must exceed the center radius.");
    }
    if(config.surround_inner_radius_sites < 0.0 || config.surround_inner_radius_sites >= config.broad_radius_sites) {
        throw std::runtime_error("V1_ORIENTATION_CONTEXT_SURROUND_INNER_RADIUS_SITES must be in [0, broad_radius).");
    }
    return config;
}

SensoryAssayConfig getSensoryAssayConfig()
{
    SensoryAssayConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_SENSORY_ASSAY_ENABLE", 0u) != 0u;
    config.blank_repeat_count = config.enabled
        ? getEnvUnsignedOrDefault("V1_BLANK_REPEAT_COUNT", kDefaultBlankRepeatCount)
        : 0u;
    config.contrast_radius_sites = getEnvDoubleOrDefault(
        "V1_CONTRAST_SWEEP_RADIUS_SITES",
        kDefaultCenterStimulusRadiusSites);
    config.contrasts = config.enabled
        ? getEnvDoubleListOrDefault("V1_CONTRAST_SWEEP_VALUES", kDefaultContrastSweepValues)
        : std::vector<double>();

    if(config.enabled && config.blank_repeat_count < kDefaultBlankRepeatCount) {
        throw std::runtime_error("V1_BLANK_REPEAT_COUNT must be at least 4 when V1_SENSORY_ASSAY_ENABLE=1.");
    }
    if(config.contrast_radius_sites <= 0.0) {
        throw std::runtime_error("V1_CONTRAST_SWEEP_RADIUS_SITES must be positive.");
    }
    for(double contrast : config.contrasts) {
        if(contrast <= 0.0 || !std::isfinite(contrast)) {
            throw std::runtime_error("V1_CONTRAST_SWEEP_VALUES entries must be finite and positive.");
        }
    }
    return config;
}

GeNN::NeuronGroup &requireNeuronGroup(GeNN::ModelSpec &model, const std::string &name)
{
    GeNN::NeuronGroup *group = model.findNeuronGroup(name);
    if(group == nullptr) {
        throw std::runtime_error("Unable to find neuron group: " + name);
    }
    return *group;
}

GeNN::SynapseGroup &requireSynapseGroup(GeNN::ModelSpec &model, const std::string &name)
{
    GeNN::SynapseGroup *group = model.findSynapseGroup(name);
    if(group == nullptr) {
        throw std::runtime_error("Unable to find synapse group: " + name);
    }
    return *group;
}

template <typename GroupT>
GeNN::Runtime::ArrayBase &requireArray(
    GeNN::Runtime::Runtime &runtime,
    const GroupT &group,
    const std::string &name)
{
    GeNN::Runtime::ArrayBase *array = runtime.getArray(group, name);
    if(array == nullptr) {
        throw std::runtime_error("Unable to find runtime array '" + name + "' for group '" + group.getName() + "'");
    }
    return *array;
}

double radiansToDegrees(double radians)
{
    return radians * 180.0 / v1_genn::kPi;
}

double positiveModuloDegrees(double degrees)
{
    double wrapped = std::fmod(degrees, 180.0);
    if(wrapped < 0.0) {
        wrapped += 180.0;
    }
    return wrapped;
}

double median(std::vector<double> values)
{
    if(values.empty()) {
        return 0.0;
    }

    const std::size_t midpoint = values.size() / 2u;
    std::nth_element(values.begin(), values.begin() + midpoint, values.end());
    double result = values[midpoint];
    if(values.size() % 2u == 0u) {
        const double upper = result;
        std::nth_element(values.begin(), values.begin() + midpoint - 1u, values.begin() + midpoint);
        result = 0.5 * (values[midpoint - 1u] + upper);
    }
    return result;
}

std::vector<double> makeSweepOrientations(unsigned int orientation_count)
{
    std::vector<double> orientations_rad;
    orientations_rad.reserve(orientation_count);
    for(unsigned int i = 0; i < orientation_count; i++) {
        orientations_rad.push_back((static_cast<double>(i) * v1_genn::kPi) / static_cast<double>(orientation_count));
    }
    return orientations_rad;
}

unsigned int durationToSteps(double duration_ms)
{
    const double exact_steps = duration_ms / v1_genn::kDtMs;
    const double rounded_steps = std::round(exact_steps);
    if(std::fabs(exact_steps - rounded_steps) > 1.0e-9) {
        std::ostringstream message;
        message << "Duration " << duration_ms << " ms is not an integer multiple of dt " << v1_genn::kDtMs << " ms.";
        throw std::runtime_error(message.str());
    }
    if(rounded_steps <= 0.0) {
        throw std::runtime_error("Duration must correspond to at least one simulation step.");
    }
    return static_cast<unsigned int>(rounded_steps);
}

void fillL4EDrive(
    std::vector<float> &drive,
    double orientation_rad,
    double phase_rad,
    double aperture_radius_sites = -1.0,
    unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max(),
    double aperture_inner_radius_sites = -1.0,
    double contrast = 1.0)
{
    drive.resize(v1_genn::kNumL4E);
    double center_x = (static_cast<double>(v1_genn::kSheetSide) - 1.0) * 0.5;
    double center_y = center_x;
    if(aperture_radius_sites > 0.0 && aperture_center_site != std::numeric_limits<unsigned int>::max()) {
        if(aperture_center_site >= v1_genn::kSiteCount) {
            throw std::runtime_error("Validation aperture center site is outside the sheet.");
        }
        const auto center_xy = v1_genn::siteIndexToXY(aperture_center_site);
        center_x = static_cast<double>(center_xy.first);
        center_y = static_cast<double>(center_xy.second);
    }
    for(unsigned int site = 0; site < v1_genn::kSiteCount; site++) {
        const auto xy = v1_genn::siteIndexToXY(site);
        double aperture = 1.0;
        if(aperture_radius_sites > 0.0) {
            const double dx = static_cast<double>(xy.first) - center_x;
            const double dy = static_cast<double>(xy.second) - center_y;
            const double radius = std::sqrt((dx * dx) + (dy * dy));
            aperture = (radius <= aperture_radius_sites) ? 1.0 : 0.0;
            if(aperture_inner_radius_sites >= 0.0 && radius <= aperture_inner_radius_sites) {
                aperture = 0.0;
            }
        }
        for(unsigned int neuron = 0; neuron < v1_genn::kL4EPerSite; neuron++) {
            const unsigned int index = (site * v1_genn::kL4EPerSite) + neuron;
            drive[index] = static_cast<float>(
                v1_genn::l4SimpleCellDrive(
                    xy.first,
                    xy.second,
                    neuron,
                    orientation_rad,
                    phase_rad) * aperture * contrast);
        }
    }
}

void fillL4ECenterSurroundDrive(
    std::vector<float> &drive,
    double center_orientation_rad,
    double surround_orientation_rad,
    double phase_rad,
    double center_radius_sites,
    double surround_outer_radius_sites,
    unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max(),
    double contrast = 1.0)
{
    if(center_radius_sites <= 0.0 || surround_outer_radius_sites <= center_radius_sites) {
        throw std::runtime_error("Center-surround validation drive requires outer radius greater than center radius.");
    }

    drive.resize(v1_genn::kNumL4E);
    double center_x = (static_cast<double>(v1_genn::kSheetSide) - 1.0) * 0.5;
    double center_y = center_x;
    if(aperture_center_site != std::numeric_limits<unsigned int>::max()) {
        if(aperture_center_site >= v1_genn::kSiteCount) {
            throw std::runtime_error("Validation aperture center site is outside the sheet.");
        }
        const auto center_xy = v1_genn::siteIndexToXY(aperture_center_site);
        center_x = static_cast<double>(center_xy.first);
        center_y = static_cast<double>(center_xy.second);
    }

    for(unsigned int site = 0; site < v1_genn::kSiteCount; site++) {
        const auto xy = v1_genn::siteIndexToXY(site);
        const double dx = static_cast<double>(xy.first) - center_x;
        const double dy = static_cast<double>(xy.second) - center_y;
        const double radius = std::sqrt((dx * dx) + (dy * dy));
        const bool in_center = (radius <= center_radius_sites);
        const bool in_surround = (!in_center && radius <= surround_outer_radius_sites);

        for(unsigned int neuron = 0; neuron < v1_genn::kL4EPerSite; neuron++) {
            const unsigned int index = (site * v1_genn::kL4EPerSite) + neuron;
            if(!in_center && !in_surround) {
                drive[index] = 0.0f;
                continue;
            }
            const double orientation_rad = in_center ? center_orientation_rad : surround_orientation_rad;
            drive[index] = static_cast<float>(
                v1_genn::l4SimpleCellDrive(
                    xy.first,
                    xy.second,
                    neuron,
                    orientation_rad,
                    phase_rad) * contrast);
        }
    }
}

void setConstantExternalCurrent(GeNN::Runtime::Runtime &runtime, const GeNN::NeuronGroup &group, double current_na)
{
    GeNN::Runtime::ArrayBase &array = requireArray(runtime, group, "Iext");
    std::fill(array.getHostPointer<float>(), array.getHostPointer<float>() + array.getCount(), static_cast<float>(current_na));
    array.pushToDevice();
}

double deterministicConnectionUnit(unsigned int pre_id, unsigned int post_id);
double softOrientationConnectionProbability(double similarity, unsigned int manhattan_distance, double bias_strength);
double orientationNeutralConnectionProbability(unsigned int manhattan_distance, double probability_scale);

std::vector<std::pair<unsigned int, unsigned int>> buildL4EToL23EConnectivity()
{
    std::vector<std::pair<unsigned int, unsigned int>> edges;
    edges.reserve(
        static_cast<std::size_t>(v1_genn::kNumL4E)
        * static_cast<std::size_t>(((2u * v1_genn::kFeedforwardRadius) + 1u) * ((2u * v1_genn::kFeedforwardRadius) + 1u))
        * static_cast<std::size_t>(v1_genn::kL23EPerSite));
    const L4L23OrientationConfig config = getL4L23OrientationConfig();
    const bool use_neutral_density_match =
        config.neutral_density_match_enabled && config.bias_strength == 0.0;

    for(unsigned int pre_id = 0; pre_id < v1_genn::kNumL4E; pre_id++) {
        const unsigned int pre_site = pre_id / v1_genn::kL4EPerSite;
        const unsigned int pre_x = pre_site % v1_genn::kSheetSide;
        const unsigned int pre_y = pre_site / v1_genn::kSheetSide;
        const double pre_orientation =
            use_neutral_density_match ? 0.0 : v1_genn::sitePreferredOrientationFromIndex(pre_site);

        for(int dy = -static_cast<int>(v1_genn::kFeedforwardRadius); dy <= static_cast<int>(v1_genn::kFeedforwardRadius); dy++) {
            const int post_y = static_cast<int>(pre_y) + dy;
            if(post_y < 0 || post_y >= static_cast<int>(v1_genn::kSheetSide)) {
                continue;
            }

            for(int dx = -static_cast<int>(v1_genn::kFeedforwardRadius); dx <= static_cast<int>(v1_genn::kFeedforwardRadius); dx++) {
                const int post_x = static_cast<int>(pre_x) + dx;
                if(post_x < 0 || post_x >= static_cast<int>(v1_genn::kSheetSide)) {
                    continue;
                }

                const unsigned int post_site =
                    (static_cast<unsigned int>(post_y) * v1_genn::kSheetSide)
                    + static_cast<unsigned int>(post_x);
                const unsigned int manhattan_distance = static_cast<unsigned int>(std::abs(dx) + std::abs(dy));
                double connection_probability = 0.0;
                if(use_neutral_density_match) {
                    connection_probability = orientationNeutralConnectionProbability(
                        manhattan_distance,
                        config.neutral_probability_scale);
                }
                else {
                    const double post_orientation = v1_genn::sitePreferredOrientationFromIndex(post_site);
                    const double delta = v1_genn::circularOrientationDifference(pre_orientation, post_orientation);
                    const double similarity = 0.5 * (1.0 + std::cos(2.0 * delta));
                    connection_probability = softOrientationConnectionProbability(
                        similarity,
                        manhattan_distance,
                        config.bias_strength);
                }

                for(unsigned int post_cell = 0; post_cell < v1_genn::kL23EPerSite; post_cell++) {
                    const unsigned int post_id = (post_site * v1_genn::kL23EPerSite) + post_cell;
                    if(deterministicConnectionUnit(pre_id, post_id) < connection_probability) {
                        edges.emplace_back(pre_id, post_id);
                    }
                }
            }
        }
    }

    return edges;
}

std::vector<std::pair<unsigned int, unsigned int>> buildLocalPatchConnectivity(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius,
    bool exclude_self)
{
    std::vector<std::pair<unsigned int, unsigned int>> edges;
    edges.reserve(
        static_cast<std::size_t>(v1_genn::kSiteCount)
        * static_cast<std::size_t>(pre_neurons_per_site)
        * static_cast<std::size_t>(((2u * radius) + 1u) * ((2u * radius) + 1u))
        * static_cast<std::size_t>(post_neurons_per_site));

    for(unsigned int pre_id = 0; pre_id < (v1_genn::kSiteCount * pre_neurons_per_site); pre_id++) {
        const unsigned int pre_site = pre_id / pre_neurons_per_site;
        const unsigned int pre_x = pre_site % v1_genn::kSheetSide;
        const unsigned int pre_y = pre_site / v1_genn::kSheetSide;

        for(int dy = -static_cast<int>(radius); dy <= static_cast<int>(radius); dy++) {
            const int post_y = static_cast<int>(pre_y) + dy;
            if(post_y < 0 || post_y >= static_cast<int>(v1_genn::kSheetSide)) {
                continue;
            }

            for(int dx = -static_cast<int>(radius); dx <= static_cast<int>(radius); dx++) {
                const int post_x = static_cast<int>(pre_x) + dx;
                if(post_x < 0 || post_x >= static_cast<int>(v1_genn::kSheetSide)) {
                    continue;
                }

                const unsigned int post_site =
                    (static_cast<unsigned int>(post_y) * v1_genn::kSheetSide)
                    + static_cast<unsigned int>(post_x);
                for(unsigned int post_cell = 0; post_cell < post_neurons_per_site; post_cell++) {
                    const unsigned int post_id = (post_site * post_neurons_per_site) + post_cell;
                    if(exclude_self && pre_neurons_per_site == post_neurons_per_site && post_id == pre_id) {
                        continue;
                    }
                    edges.emplace_back(pre_id, post_id);
                }
            }
        }
    }

    return edges;
}

std::vector<std::pair<unsigned int, unsigned int>> buildLocalIntersiteConnectivity(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius)
{
    std::vector<std::pair<unsigned int, unsigned int>> edges;
    edges.reserve(
        static_cast<std::size_t>(v1_genn::kSiteCount)
        * static_cast<std::size_t>(pre_neurons_per_site)
        * static_cast<std::size_t>(((2u * radius) + 1u) * ((2u * radius) + 1u))
        * static_cast<std::size_t>(post_neurons_per_site));

    for(unsigned int pre_id = 0; pre_id < (v1_genn::kSiteCount * pre_neurons_per_site); pre_id++) {
        const unsigned int pre_site = pre_id / pre_neurons_per_site;
        const unsigned int pre_x = pre_site % v1_genn::kSheetSide;
        const unsigned int pre_y = pre_site / v1_genn::kSheetSide;

        for(int dy = -static_cast<int>(radius); dy <= static_cast<int>(radius); dy++) {
            const int post_y = static_cast<int>(pre_y) + dy;
            if(post_y < 0 || post_y >= static_cast<int>(v1_genn::kSheetSide)) {
                continue;
            }

            for(int dx = -static_cast<int>(radius); dx <= static_cast<int>(radius); dx++) {
                const int post_x = static_cast<int>(pre_x) + dx;
                if(post_x < 0 || post_x >= static_cast<int>(v1_genn::kSheetSide)) {
                    continue;
                }

                const unsigned int post_site =
                    (static_cast<unsigned int>(post_y) * v1_genn::kSheetSide)
                    + static_cast<unsigned int>(post_x);
                if(post_site == pre_site) {
                    continue;
                }
                for(unsigned int post_cell = 0; post_cell < post_neurons_per_site; post_cell++) {
                    const unsigned int post_id = (post_site * post_neurons_per_site) + post_cell;
                    edges.emplace_back(pre_id, post_id);
                }
            }
        }
    }

    return edges;
}

ConnectivityStats summarizeConnectivity(
    const std::vector<std::pair<unsigned int, unsigned int>> &edges,
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius)
{
    ConnectivityStats stats;
    stats.edge_count = edges.size();
    if(edges.empty()) {
        return stats;
    }

    double distance_sum = 0.0;
    std::size_t same_site_count = 0;
    std::size_t beyond_radius_count = 0;
    for(const auto &edge : edges) {
        const unsigned int pre_site = edge.first / pre_neurons_per_site;
        const unsigned int post_site = edge.second / post_neurons_per_site;
        const auto pre_xy = v1_genn::siteIndexToXY(pre_site);
        const auto post_xy = v1_genn::siteIndexToXY(post_site);
        const double dx = static_cast<double>(pre_xy.first) - static_cast<double>(post_xy.first);
        const double dy = static_cast<double>(pre_xy.second) - static_cast<double>(post_xy.second);
        const double distance = std::sqrt((dx * dx) + (dy * dy));
        const double chebyshev_distance = std::max(std::fabs(dx), std::fabs(dy));
        distance_sum += distance;
        stats.max_distance_sites = std::max(stats.max_distance_sites, distance);
        if(pre_site == post_site) {
            same_site_count++;
        }
        if(chebyshev_distance > static_cast<double>(radius) + 1.0e-9) {
            beyond_radius_count++;
        }
    }

    stats.mean_distance_sites = distance_sum / static_cast<double>(edges.size());
    stats.same_site_fraction = static_cast<double>(same_site_count) / static_cast<double>(edges.size());
    stats.beyond_radius_fraction = static_cast<double>(beyond_radius_count) / static_cast<double>(edges.size());
    return stats;
}

double deterministicSparseSample(unsigned int pre_id, unsigned int post_id)
{
    std::uint32_t hash = static_cast<std::uint32_t>((pre_id + 1u) * 1103515245u);
    hash ^= static_cast<std::uint32_t>((post_id + 1u) * 12345u);
    hash ^= (hash >> 16);
    return static_cast<double>(hash & 0x00FFFFFFu) / 16777216.0;
}

std::vector<std::pair<unsigned int, unsigned int>> buildSparseDistanceConnectivity(
    unsigned int pre_neurons_per_site,
    unsigned int post_neurons_per_site,
    unsigned int radius,
    bool exclude_self,
    double peak_probability,
    double distance_sigma_sq)
{
    std::vector<std::pair<unsigned int, unsigned int>> edges;
    edges.reserve(
        static_cast<std::size_t>(v1_genn::kSiteCount)
        * static_cast<std::size_t>(pre_neurons_per_site)
        * static_cast<std::size_t>(32u));

    for(unsigned int pre_id = 0; pre_id < (v1_genn::kSiteCount * pre_neurons_per_site); pre_id++) {
        const unsigned int pre_site = pre_id / pre_neurons_per_site;
        const unsigned int pre_x = pre_site % v1_genn::kSheetSide;
        const unsigned int pre_y = pre_site / v1_genn::kSheetSide;

        for(int dy = -static_cast<int>(radius); dy <= static_cast<int>(radius); dy++) {
            const int post_y = static_cast<int>(pre_y) + dy;
            if(post_y < 0 || post_y >= static_cast<int>(v1_genn::kSheetSide)) {
                continue;
            }

            for(int dx = -static_cast<int>(radius); dx <= static_cast<int>(radius); dx++) {
                const int post_x = static_cast<int>(pre_x) + dx;
                if(post_x < 0 || post_x >= static_cast<int>(v1_genn::kSheetSide)) {
                    continue;
                }

                const double distance_sq = static_cast<double>((dx * dx) + (dy * dy));
                const double probability = peak_probability * std::exp(-distance_sq / (2.0 * distance_sigma_sq));
                const unsigned int post_site =
                    (static_cast<unsigned int>(post_y) * v1_genn::kSheetSide)
                    + static_cast<unsigned int>(post_x);
                for(unsigned int post_cell = 0; post_cell < post_neurons_per_site; post_cell++) {
                    const unsigned int post_id = (post_site * post_neurons_per_site) + post_cell;
                    if(exclude_self && pre_neurons_per_site == post_neurons_per_site && post_id == pre_id) {
                        continue;
                    }
                    if(deterministicSparseSample(pre_id, post_id) < probability) {
                        edges.emplace_back(pre_id, post_id);
                    }
                }
            }
        }
    }

    return edges;
}

std::vector<float> copyWeights(GeNN::Runtime::Runtime &runtime, GeNN::SynapseGroup &synapse_group)
{
    GeNN::Runtime::ArrayBase &weight_array = requireArray(runtime, synapse_group, "g");
    weight_array.pullFromDevice();
    const float *weights = weight_array.getHostPointer<float>();
    return std::vector<float>(weights, weights + weight_array.getCount());
}

void scaleSynapseWeights(GeNN::Runtime::Runtime &runtime, GeNN::SynapseGroup &synapse_group, double scale)
{
    GeNN::Runtime::ArrayBase &weight_array = requireArray(runtime, synapse_group, "g");
    weight_array.pullFromDevice();
    float *weights = weight_array.getHostPointer<float>();
    for(std::size_t i = 0; i < weight_array.getCount(); i++) {
        weights[i] = static_cast<float>(weights[i] * scale);
    }
    weight_array.pushToDevice();
}

WeightStats summarizeWeights(const std::vector<float> &weights)
{
    WeightStats stats;
    stats.count = weights.size();
    if(weights.empty()) {
        return stats;
    }

    auto minmax = std::minmax_element(weights.begin(), weights.end());
    stats.min = static_cast<double>(*minmax.first);
    stats.max = static_cast<double>(*minmax.second);
    stats.mean = std::accumulate(weights.begin(), weights.end(), 0.0) / static_cast<double>(weights.size());
    return stats;
}

double nonzeroWeightFraction(const std::vector<float> &weights)
{
    if(weights.empty()) {
        return 0.0;
    }
    const std::size_t nonzero_count = static_cast<std::size_t>(std::count_if(
        weights.begin(),
        weights.end(),
        [](float weight) { return weight != 0.0f; }));
    return static_cast<double>(nonzero_count) / static_cast<double>(weights.size());
}

double giniCoefficient(std::vector<double> values)
{
    if(values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    if(total <= 0.0) {
        return 0.0;
    }
    double weighted_sum = 0.0;
    for(std::size_t i = 0; i < values.size(); i++) {
        weighted_sum += static_cast<double>(i + 1u) * values[i];
    }
    const double count = static_cast<double>(values.size());
    return ((2.0 * weighted_sum) / (count * total)) - ((count + 1.0) / count);
}

double topMassShare(std::vector<double> values, double fraction)
{
    if(values.empty()) {
        return 0.0;
    }
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    if(total <= 0.0) {
        return 0.0;
    }
    std::sort(values.begin(), values.end(), std::greater<double>());
    const std::size_t top_count = std::max<std::size_t>(
        1u,
        static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(values.size()))));
    return std::accumulate(values.begin(), values.begin() + std::min(top_count, values.size()), 0.0) / total;
}

std::vector<double> positiveWeightValues(const std::vector<float> &weights)
{
    std::vector<double> values;
    values.reserve(weights.size());
    for(float weight : weights) {
        if(weight > 0.0f) {
            values.push_back(static_cast<double>(weight));
        }
    }
    return values;
}

double deterministicConnectionUnit(unsigned int pre_id, unsigned int post_id)
{
    std::uint32_t hash =
        ((static_cast<std::uint32_t>(pre_id) + 1u) * 747796405u)
        ^ ((static_cast<std::uint32_t>(post_id) + 1u) * 2891336453u);
    hash ^= hash >> 16;
    hash *= 2246822519u;
    hash ^= hash >> 13;
    hash *= 3266489917u;
    hash ^= hash >> 16;
    return static_cast<double>(hash & 0x00FFFFFFu) / 16777216.0;
}

double softOrientationConnectionProbability(double similarity, unsigned int manhattan_distance, double bias_strength)
{
    const double biased_similarity = ((1.0 - bias_strength) * 0.5) + (bias_strength * similarity);
    double probability =
        v1_genn::kOrientationSoftProbabilityFloor
        + ((1.0 - v1_genn::kOrientationSoftProbabilityFloor) * biased_similarity);
    probability -= v1_genn::kOrientationDistancePenalty * static_cast<double>(manhattan_distance);
    probability = std::max(v1_genn::kOrientationSoftProbabilityFloor, probability);
    return std::min(1.0, probability);
}

double orientationNeutralConnectionProbability(unsigned int manhattan_distance, double probability_scale)
{
    double probability =
        v1_genn::kOrientationSoftProbabilityFloor
        + ((1.0 - v1_genn::kOrientationSoftProbabilityFloor) * 0.5);
    probability -= v1_genn::kOrientationDistancePenalty * static_cast<double>(manhattan_distance);
    probability = std::max(v1_genn::kOrientationSoftProbabilityFloor, probability);
    probability *= probability_scale;
    probability = std::max(v1_genn::kOrientationSoftProbabilityFloor, probability);
    return std::min(1.0, probability);
}

double clippedL23EEWeight(double weight)
{
    return std::min(kL23EEStdpWeightMax, std::max(kL23EEStdpWeightMin, weight));
}

double deterministicLognormalFactor(unsigned int pre_id, unsigned int post_id, double sigma)
{
    if(sigma == 0.0) {
        return 1.0;
    }
    const double u1 = std::max(1.0e-12, deterministicConnectionUnit(pre_id, post_id));
    const double u2 = deterministicConnectionUnit(post_id, pre_id);
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * v1_genn::kPi * u2);
    return std::exp((-0.5 * sigma * sigma) + (sigma * z));
}

void applyL23EELognormalInitialWeights(
    GeNN::Runtime::Runtime &runtime,
    GeNN::SynapseGroup &synapse_group,
    const std::vector<std::pair<unsigned int, unsigned int>> &edges,
    const L23EELognormalInitConfig &config)
{
    if(!config.enabled) {
        return;
    }

    GeNN::Runtime::ArrayBase &weight_array = requireArray(runtime, synapse_group, "g");
    if(weight_array.getCount() % v1_genn::kNumL23E != 0u) {
        throw std::runtime_error("L23E->L23E lognormal init expected row-major sparse weight capacity.");
    }
    const std::size_t max_row_length = weight_array.getCount() / v1_genn::kNumL23E;
    weight_array.pullFromDevice();
    float *weights = weight_array.getHostPointer<float>();

    std::vector<std::size_t> active_slots;
    std::vector<double> active_values;
    active_slots.reserve(edges.size());
    active_values.reserve(edges.size());

    unsigned int previous_pre_id = std::numeric_limits<unsigned int>::max();
    std::size_t row_active_index = 0;
    for(const auto &edge : edges) {
        const unsigned int pre_id = edge.first;
        const unsigned int post_id = edge.second;
        if(pre_id != previous_pre_id) {
            previous_pre_id = pre_id;
            row_active_index = 0;
        }
        if(row_active_index >= max_row_length) {
            throw std::runtime_error("L23E->L23E lognormal init exceeded sparse row capacity.");
        }

        const std::size_t slot = (static_cast<std::size_t>(pre_id) * max_row_length) + row_active_index;
        row_active_index++;
        if(slot >= weight_array.getCount()) {
            throw std::runtime_error("L23E->L23E lognormal init slot is outside the sparse weight array.");
        }

        active_slots.push_back(slot);
        active_values.push_back(clippedL23EEWeight(
            v1_genn::kL23EEWeight * deterministicLognormalFactor(pre_id, post_id, config.sigma)));
    }
    if(active_values.empty()) {
        throw std::runtime_error("L23E->L23E lognormal init requires at least one active edge.");
    }

    for(std::size_t i = 0; i < active_slots.size(); i++) {
        weights[active_slots[i]] = static_cast<float>(active_values[i]);
    }
    weight_array.pushToDevice();
}

void writeWeightCsv(
    const std::string &path,
    const std::vector<float> &weights,
    const std::vector<std::pair<unsigned int, unsigned int>> &edges)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    if(weights.size() == edges.size()) {
        output << "synapse_index,pre_id,post_id,g\n";
        for(std::size_t i = 0; i < weights.size(); i++) {
            output << i << "," << edges[i].first << "," << edges[i].second << "," << weights[i] << "\n";
        }
    }
    else {
        output << "synapse_index,g\n";
        for(std::size_t i = 0; i < weights.size(); i++) {
            output << i << "," << weights[i] << "\n";
        }
    }
}

template <typename SpikeBatch>
std::vector<double> countSiteSpikesForTrials(
    const SpikeBatch &batch,
    const std::vector<TrialWindow> &trials,
    unsigned int neurons_per_site)
{
    std::vector<double> counts(static_cast<std::size_t>(trials.size()) * v1_genn::kSiteCount, 0.0);
    if(trials.empty()) {
        return counts;
    }

    const auto &spike_times = batch.first;
    const auto &spike_ids = batch.second;
    if(spike_times.size() != spike_ids.size()) {
        throw std::runtime_error("Recorded spike times and ids do not have matching lengths.");
    }

    std::size_t trial_index = 0;
    for(std::size_t i = 0; i < spike_times.size() && trial_index < trials.size(); i++) {
        const double spike_time = static_cast<double>(spike_times[i]);
        while(trial_index < trials.size() && spike_time >= trials[trial_index].end_ms) {
            trial_index++;
        }
        if(trial_index >= trials.size()) {
            break;
        }

        const TrialWindow &trial = trials[trial_index];
        if(spike_time < trial.measure_start_ms || spike_time >= trial.end_ms) {
            continue;
        }

        const unsigned int neuron_id = static_cast<unsigned int>(spike_ids[i]);
        const unsigned int site_id = neuron_id / neurons_per_site;
        counts[(trial_index * v1_genn::kSiteCount) + site_id] += 1.0;
    }

    return counts;
}

template <typename SpikeBatch>
std::vector<double> countNeuronSpikesForTrials(
    const SpikeBatch &batch,
    const std::vector<TrialWindow> &trials,
    unsigned int neuron_count)
{
    std::vector<double> counts(static_cast<std::size_t>(trials.size()) * neuron_count, 0.0);
    if(trials.empty()) {
        return counts;
    }

    const auto &spike_times = batch.first;
    const auto &spike_ids = batch.second;
    if(spike_times.size() != spike_ids.size()) {
        throw std::runtime_error("Recorded spike times and ids do not have matching lengths.");
    }

    std::size_t trial_index = 0;
    for(std::size_t i = 0; i < spike_times.size() && trial_index < trials.size(); i++) {
        const double spike_time = static_cast<double>(spike_times[i]);
        while(trial_index < trials.size() && spike_time >= trials[trial_index].end_ms) {
            trial_index++;
        }
        if(trial_index >= trials.size()) {
            break;
        }

        const TrialWindow &trial = trials[trial_index];
        if(spike_time < trial.measure_start_ms || spike_time >= trial.end_ms) {
            continue;
        }

        const unsigned int neuron_id = static_cast<unsigned int>(spike_ids[i]);
        if(neuron_id >= neuron_count) {
            throw std::runtime_error("Recorded spike id exceeds neuron count.");
        }
        counts[(trial_index * neuron_count) + neuron_id] += 1.0;
    }

    return counts;
}

template <typename SpikeBatch>
std::vector<double> countPopulationRatesForTrials(
    const SpikeBatch &batch,
    const std::vector<TrialWindow> &trials,
    unsigned int neuron_count)
{
    std::vector<double> counts(trials.size(), 0.0);
    if(trials.empty()) {
        return counts;
    }

    const auto &spike_times = batch.first;
    const auto &spike_ids = batch.second;
    if(spike_times.size() != spike_ids.size()) {
        throw std::runtime_error("Recorded spike times and ids do not have matching lengths.");
    }

    std::size_t trial_index = 0;
    for(std::size_t i = 0; i < spike_times.size() && trial_index < trials.size(); i++) {
        const double spike_time = static_cast<double>(spike_times[i]);
        while(trial_index < trials.size() && spike_time >= trials[trial_index].end_ms) {
            trial_index++;
        }
        if(trial_index >= trials.size()) {
            break;
        }

        const TrialWindow &trial = trials[trial_index];
        if(spike_time >= trial.measure_start_ms && spike_time < trial.end_ms) {
            counts[trial_index] += 1.0;
        }
    }

    std::vector<double> rates(trials.size(), 0.0);
    for(std::size_t trial_index = 0; trial_index < trials.size(); trial_index++) {
        const double measurement_duration_s = (trials[trial_index].end_ms - trials[trial_index].measure_start_ms) / 1000.0;
        rates[trial_index] = counts[trial_index] / (measurement_duration_s * static_cast<double>(neuron_count));
    }
    return rates;
}

double meanRate(const std::vector<double> &rates)
{
    if(rates.empty()) {
        return 0.0;
    }
    return std::accumulate(rates.begin(), rates.end(), 0.0) / static_cast<double>(rates.size());
}

std::vector<PopulationSiteMetrics> computeSiteMetrics(
    const std::vector<TrialWindow> &trials,
    const std::vector<double> &site_spike_counts,
    unsigned int neurons_per_site)
{
    if(trials.empty()) {
        return {};
    }

    const double measurement_duration_ms = trials.front().end_ms - trials.front().measure_start_ms;
    if(measurement_duration_ms <= 0.0) {
        throw std::runtime_error("Measurement window must be positive.");
    }
    const double measurement_duration_s = measurement_duration_ms / 1000.0;

    std::vector<PopulationSiteMetrics> metrics(v1_genn::kSiteCount);
    for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
        PopulationSiteMetrics metric;
        metric.site_id = site_id;
        const auto xy = v1_genn::siteIndexToXY(site_id);
        metric.x = xy.first;
        metric.y = xy.second;
        metric.map_pref_rad = v1_genn::sitePreferredOrientationFromIndex(site_id);
        metric.rates_hz.resize(trials.size(), 0.0);

        double vector_x = 0.0;
        double vector_y = 0.0;
        double total_rate = 0.0;
        for(std::size_t trial_index = 0; trial_index < trials.size(); trial_index++) {
            const double site_spikes = site_spike_counts[(trial_index * v1_genn::kSiteCount) + site_id];
            const double rate_hz = site_spikes / (measurement_duration_s * static_cast<double>(neurons_per_site));
            metric.rates_hz[trial_index] = rate_hz;
            total_rate += rate_hz;
            vector_x += rate_hz * std::cos(2.0 * trials[trial_index].orientation_rad);
            vector_y += rate_hz * std::sin(2.0 * trials[trial_index].orientation_rad);
        }

        metric.mean_rate_hz = total_rate / static_cast<double>(trials.size());
        if(total_rate > 0.0) {
            metric.osi = std::hypot(vector_x, vector_y) / total_rate;
            metric.measured_pref_rad = v1_genn::wrapOrientationRadians(0.5 * std::atan2(vector_y, vector_x));
        }
        else {
            metric.osi = 0.0;
            metric.measured_pref_rad = metric.map_pref_rad;
        }

        metrics[site_id] = metric;
    }

    return metrics;
}

std::vector<CellTuningMetrics> computeCellTuningMetrics(
    const std::vector<TrialWindow> &trials,
    const std::vector<double> &cell_spike_counts,
    unsigned int neuron_count,
    unsigned int neurons_per_site)
{
    if(trials.empty()) {
        return {};
    }

    const double measurement_duration_ms = trials.front().end_ms - trials.front().measure_start_ms;
    if(measurement_duration_ms <= 0.0) {
        throw std::runtime_error("Measurement window must be positive.");
    }
    const double measurement_duration_s = measurement_duration_ms / 1000.0;

    std::vector<CellTuningMetrics> metrics(neuron_count);
    for(unsigned int cell_id = 0; cell_id < neuron_count; cell_id++) {
        CellTuningMetrics metric;
        metric.cell_id = cell_id;
        metric.site_id = cell_id / neurons_per_site;
        metric.site_pref_rad = v1_genn::sitePreferredOrientationFromIndex(metric.site_id);
        metric.rates_hz.resize(trials.size(), 0.0);

        double vector_x = 0.0;
        double vector_y = 0.0;
        double total_rate = 0.0;
        for(std::size_t trial_index = 0; trial_index < trials.size(); trial_index++) {
            const double spikes = cell_spike_counts[(trial_index * neuron_count) + cell_id];
            const double rate_hz = spikes / measurement_duration_s;
            metric.rates_hz[trial_index] = rate_hz;
            total_rate += rate_hz;
            metric.peak_rate_hz = std::max(metric.peak_rate_hz, rate_hz);
            vector_x += rate_hz * std::cos(2.0 * trials[trial_index].orientation_rad);
            vector_y += rate_hz * std::sin(2.0 * trials[trial_index].orientation_rad);
        }

        metric.mean_rate_hz = total_rate / static_cast<double>(trials.size());
        if(total_rate > 0.0) {
            metric.osi = std::hypot(vector_x, vector_y) / total_rate;
            metric.measured_pref_rad = v1_genn::wrapOrientationRadians(0.5 * std::atan2(vector_y, vector_x));
        }
        else {
            metric.osi = 0.0;
            metric.measured_pref_rad = metric.site_pref_rad;
        }
        metrics[cell_id] = metric;
    }

    return metrics;
}

std::vector<MultiPhaseCellTuningMetrics> computeMultiPhaseCellTuningMetrics(
    const std::vector<TrialWindow> &trials,
    const std::vector<double> &cell_spike_counts,
    const std::vector<double> &orientations_rad,
    unsigned int phase_count,
    unsigned int neuron_count,
    unsigned int neurons_per_site)
{
    if(trials.empty()) {
        return {};
    }
    if(phase_count <= 1u) {
        throw std::runtime_error("Multiphase cell tuning requires phase_count > 1.");
    }
    if(trials.size() != (static_cast<std::size_t>(orientations_rad.size()) * static_cast<std::size_t>(phase_count))) {
        throw std::runtime_error("Multiphase cell tuning trials must be orientation x phase_count.");
    }

    const double measurement_duration_ms = trials.front().end_ms - trials.front().measure_start_ms;
    if(measurement_duration_ms <= 0.0) {
        throw std::runtime_error("Measurement window must be positive.");
    }
    const double measurement_duration_s = measurement_duration_ms / 1000.0;

    std::vector<MultiPhaseCellTuningMetrics> metrics(neuron_count);
    for(unsigned int cell_id = 0; cell_id < neuron_count; cell_id++) {
        MultiPhaseCellTuningMetrics metric;
        metric.cell_id = cell_id;
        metric.site_id = cell_id / neurons_per_site;
        metric.site_pref_rad = v1_genn::sitePreferredOrientationFromIndex(metric.site_id);
        metric.best_orientation_rad = metric.site_pref_rad;
        metric.phase_mean_rates_hz.resize(orientations_rad.size(), 0.0);

        double total_rate = 0.0;
        double vector_x = 0.0;
        double vector_y = 0.0;
        for(std::size_t orientation_index = 0; orientation_index < orientations_rad.size(); orientation_index++) {
            double orientation_rate_sum = 0.0;
            for(unsigned int phase_index = 0; phase_index < phase_count; phase_index++) {
                const std::size_t trial_index =
                    (orientation_index * static_cast<std::size_t>(phase_count)) + phase_index;
                const double spikes = cell_spike_counts[(trial_index * neuron_count) + cell_id];
                const double rate_hz = spikes / measurement_duration_s;
                orientation_rate_sum += rate_hz;
                total_rate += rate_hz;
                if(rate_hz > metric.peak_rate_any_phase_hz) {
                    metric.peak_rate_any_phase_hz = rate_hz;
                    metric.best_orientation_rad = trials[trial_index].orientation_rad;
                    metric.best_phase_rad = trials[trial_index].phase_rad;
                }
            }

            const double phase_mean_rate = orientation_rate_sum / static_cast<double>(phase_count);
            metric.phase_mean_rates_hz[orientation_index] = phase_mean_rate;
            vector_x += phase_mean_rate * std::cos(2.0 * orientations_rad[orientation_index]);
            vector_y += phase_mean_rate * std::sin(2.0 * orientations_rad[orientation_index]);
        }

        metric.mean_rate_hz = total_rate / static_cast<double>(trials.size());
        const double pooled_total_rate = std::accumulate(
            metric.phase_mean_rates_hz.begin(),
            metric.phase_mean_rates_hz.end(),
            0.0);
        if(pooled_total_rate > 0.0) {
            metric.phase_pooled_osi = std::hypot(vector_x, vector_y) / pooled_total_rate;
        }
        metrics[cell_id] = metric;
    }

    return metrics;
}

double responseCorrelation(const std::vector<double> &pre_rates, const std::vector<double> &post_rates)
{
    if(pre_rates.size() != post_rates.size()) {
        throw std::runtime_error("Response correlation requires aligned response vectors.");
    }
    if(pre_rates.size() < 2u) {
        return 0.0;
    }

    const double pre_mean = meanRate(pre_rates);
    const double post_mean = meanRate(post_rates);
    double covariance = 0.0;
    double pre_var = 0.0;
    double post_var = 0.0;
    for(std::size_t i = 0; i < pre_rates.size(); i++) {
        const double pre_centered = pre_rates[i] - pre_mean;
        const double post_centered = post_rates[i] - post_mean;
        covariance += pre_centered * post_centered;
        pre_var += pre_centered * pre_centered;
        post_var += post_centered * post_centered;
    }
    if(pre_var <= 0.0 || post_var <= 0.0) {
        return 0.0;
    }
    return covariance / std::sqrt(pre_var * post_var);
}

double computeMedianOSI(const std::vector<PopulationSiteMetrics> &metrics)
{
    std::vector<double> osis;
    osis.reserve(metrics.size());
    for(const PopulationSiteMetrics &metric : metrics) {
        osis.push_back(metric.osi);
    }
    return median(osis);
}

double computeMedianMapErrorDegrees(const std::vector<PopulationSiteMetrics> &metrics)
{
    std::vector<double> errors_deg;
    errors_deg.reserve(metrics.size());
    for(const PopulationSiteMetrics &metric : metrics) {
        if(metric.mean_rate_hz <= 0.0) {
            continue;
        }
        errors_deg.push_back(radiansToDegrees(
            v1_genn::circularOrientationDifference(metric.map_pref_rad, metric.measured_pref_rad)));
    }
    return median(errors_deg);
}

SweepResult buildSweepResult(
    const std::string &label,
    const std::vector<double> &orientations_rad,
    const std::vector<TrialWindow> &trials,
    const std::vector<double> &l4_site_spike_counts,
    const std::vector<double> &l23_site_spike_counts)
{
    SweepResult result;
    result.label = label;
    result.orientations_rad = orientations_rad;
    result.l4_sites = computeSiteMetrics(trials, l4_site_spike_counts, v1_genn::kL4EPerSite);
    result.l23_sites = computeSiteMetrics(trials, l23_site_spike_counts, v1_genn::kL23EPerSite);
    result.l4_median_osi = computeMedianOSI(result.l4_sites);
    result.l23_median_osi = computeMedianOSI(result.l23_sites);
    result.l4_median_map_error_deg = computeMedianMapErrorDegrees(result.l4_sites);
    return result;
}

void writePopulationSiteMetricsCsv(const std::string &path, const SweepResult &result, const std::vector<PopulationSiteMetrics> &metrics)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "site_id,x,y,map_pref_deg,measured_pref_deg,osi,mean_rate_hz";
    for(double orientation_rad : result.orientations_rad) {
        output << ",rate_" << static_cast<int>(std::lround(radiansToDegrees(orientation_rad))) << "deg_hz";
    }
    output << "\n";

    for(const PopulationSiteMetrics &metric : metrics) {
        output
            << metric.site_id << ","
            << metric.x << ","
            << metric.y << ","
            << positiveModuloDegrees(radiansToDegrees(metric.map_pref_rad)) << ","
            << positiveModuloDegrees(radiansToDegrees(metric.measured_pref_rad)) << ","
            << metric.osi << ","
            << metric.mean_rate_hz;
        for(double rate_hz : metric.rates_hz) {
            output << "," << rate_hz;
        }
        output << "\n";
    }
}

void writeL23ECellTuningCsv(
    const std::string &path,
    const std::vector<double> &orientations_rad,
    const std::vector<CellTuningMetrics> &metrics,
    double recurrent_output_scale = -1.0)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "cell_id,site_id,site_pref_deg,pref_deg";
    for(double orientation_rad : orientations_rad) {
        output << ",rate_" << static_cast<int>(std::lround(radiansToDegrees(orientation_rad))) << "deg_hz";
    }
    output << ",mean_rate_hz,peak_rate_hz,osi";
    if(recurrent_output_scale >= 0.0) {
        output << ",recurrent_output_scale";
    }
    output << "\n";

    for(const CellTuningMetrics &metric : metrics) {
        output << metric.cell_id << ","
               << metric.site_id << ","
               << positiveModuloDegrees(radiansToDegrees(metric.site_pref_rad)) << ","
               << positiveModuloDegrees(radiansToDegrees(metric.measured_pref_rad));
        for(double rate_hz : metric.rates_hz) {
            output << "," << rate_hz;
        }
        output << ","
               << metric.mean_rate_hz << ","
               << metric.peak_rate_hz << ","
               << metric.osi;
        if(recurrent_output_scale >= 0.0) {
            output << "," << recurrent_output_scale;
        }
        output << "\n";
    }
}

void writeL23ECellTuningMultiPhaseCsv(
    const std::string &path,
    const std::vector<double> &orientations_rad,
    const std::vector<MultiPhaseCellTuningMetrics> &metrics,
    unsigned int phase_count)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "cell_id,site_id,site_pref_deg,best_orientation_deg,best_phase_deg,phase_count,"
           << "peak_rate_any_phase_hz,mean_rate_hz,phase_pooled_osi";
    for(double orientation_rad : orientations_rad) {
        output << ",rate_" << static_cast<int>(std::lround(radiansToDegrees(orientation_rad))) << "deg_hz";
    }
    output << "\n";

    for(const MultiPhaseCellTuningMetrics &metric : metrics) {
        output << metric.cell_id << ","
               << metric.site_id << ","
               << positiveModuloDegrees(radiansToDegrees(metric.site_pref_rad)) << ","
               << positiveModuloDegrees(radiansToDegrees(metric.best_orientation_rad)) << ","
               << radiansToDegrees(metric.best_phase_rad) << ","
               << phase_count << ","
               << metric.peak_rate_any_phase_hz << ","
               << metric.mean_rate_hz << ","
               << metric.phase_pooled_osi;
        for(double rate_hz : metric.phase_mean_rates_hz) {
            output << "," << rate_hz;
        }
        output << "\n";
    }
}

void writeSubtypeRatesCsv(
    const std::string &path,
    const std::vector<double> &orientations_rad,
    const std::vector<TrialWindow> &baseline_trials,
    const std::vector<TrialWindow> &post_trials,
    const std::vector<std::pair<std::string, std::vector<double>>> &baseline_rates,
    const std::vector<std::pair<std::string, std::vector<double>>> &post_rates)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "phase,population,orientation_deg,rate_hz\n";
    auto writeRows = [&](const std::string &phase, const std::vector<TrialWindow> &trials, const std::vector<std::pair<std::string, std::vector<double>>> &rates) {
        for(const auto &population_rates : rates) {
            if(population_rates.second.size() != trials.size()) {
                throw std::runtime_error("Subtype rate vector length mismatch for " + population_rates.first);
            }
            for(std::size_t i = 0; i < trials.size(); i++) {
                const double orientation = (i < orientations_rad.size()) ? orientations_rad[i] : trials[i].orientation_rad;
                output << phase << ","
                       << population_rates.first << ","
                       << positiveModuloDegrees(radiansToDegrees(orientation)) << ","
                       << population_rates.second[i] << "\n";
            }
        }
    };

    writeRows("baseline", baseline_trials, baseline_rates);
    writeRows("post", post_trials, post_rates);
}

unsigned int getCenterSiteId()
{
    const unsigned int center = v1_genn::kSheetSide / 2u;
    return (center * v1_genn::kSheetSide) + center;
}

ValidationSiteConfig getValidationSiteConfig(
    const std::vector<double> &size_tuning_radii_sites,
    double broad_stimulus_radius_sites)
{
    ValidationSiteConfig config;
    const unsigned int center_site_id = getCenterSiteId();
    const char *explicit_site_env = std::getenv("V1_VALIDATION_SITE_IDS");
    if(explicit_site_env != nullptr && explicit_site_env[0] != '\0') {
        config.include_validation_site_id = true;
        const std::vector<unsigned int> explicit_site_ids = getEnvUnsignedListOrEmpty("V1_VALIDATION_SITE_IDS");
        for(unsigned int site_id : explicit_site_ids) {
            if(site_id >= v1_genn::kSiteCount) {
                throw std::runtime_error("V1_VALIDATION_SITE_IDS contains a site outside the sheet.");
            }
            if(std::find(config.site_ids.begin(), config.site_ids.end(), site_id) == config.site_ids.end()) {
                config.site_ids.push_back(site_id);
                config.aperture_center_sites.push_back(site_id);
            }
        }
        return config;
    }

    const unsigned int grid_side = getEnvUnsignedOrDefault("V1_VALIDATION_GRID_SIDE", 1u);
    if(grid_side <= 1u) {
        config.site_ids.push_back(center_site_id);
        // Preserve the legacy continuous sheet-center aperture by default.
        config.aperture_center_sites.push_back(std::numeric_limits<unsigned int>::max());
        return config;
    }
    if(grid_side > v1_genn::kSheetSide) {
        throw std::runtime_error("V1_VALIDATION_GRID_SIDE cannot exceed V1_SHEET_SIDE.");
    }
    config.include_validation_site_id = true;

    double max_radius_sites = std::max(kDefaultCenterStimulusRadiusSites, broad_stimulus_radius_sites);
    for(double radius_sites : size_tuning_radii_sites) {
        max_radius_sites = std::max(max_radius_sites, radius_sites);
    }
    const unsigned int requested_margin = static_cast<unsigned int>(std::ceil(max_radius_sites));
    const unsigned int max_margin = (v1_genn::kSheetSide - 1u) / 2u;
    const unsigned int margin = std::min(requested_margin, max_margin);
    const unsigned int low = margin;
    const unsigned int high = v1_genn::kSheetSide - 1u - margin;

    for(unsigned int grid_y = 0; grid_y < grid_side; grid_y++) {
        const double fy = static_cast<double>(grid_y) / static_cast<double>(grid_side - 1u);
        const unsigned int y = static_cast<unsigned int>(std::lround(
            static_cast<double>(low) + (fy * static_cast<double>(high - low))));
        for(unsigned int grid_x = 0; grid_x < grid_side; grid_x++) {
            const double fx = static_cast<double>(grid_x) / static_cast<double>(grid_side - 1u);
            const unsigned int x = static_cast<unsigned int>(std::lround(
                static_cast<double>(low) + (fx * static_cast<double>(high - low))));
            const unsigned int site_id = (y * v1_genn::kSheetSide) + x;
            if(std::find(config.site_ids.begin(), config.site_ids.end(), site_id) == config.site_ids.end()) {
                config.site_ids.push_back(site_id);
                config.aperture_center_sites.push_back(site_id);
            }
        }
    }
    return config;
}

void writeContextValidationCsv(
    const std::string &path,
    const std::vector<double> &orientations_rad,
    const std::vector<RetinotopicContextMetrics> &validation_metrics,
    bool include_validation_site_id,
    double som_output_scale)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "condition,population,site_id";
    if(include_validation_site_id) {
        output << ",validation_site_id";
    }
    output << ",som_output_scale,mean_rate_hz";
    for(double orientation_rad : orientations_rad) {
        output << ",rate_" << static_cast<int>(std::lround(radiansToDegrees(orientation_rad))) << "deg_hz";
    }
    output << "\n";

    auto writeRow = [&](const std::string &condition,
                        const std::string &population,
                        unsigned int validation_site_id,
                        const PopulationSiteMetrics &metrics) {
        output << condition << ","
               << population << ","
               << metrics.site_id;
        if(include_validation_site_id) {
            output << "," << validation_site_id;
        }
        output << ","
               << som_output_scale << ","
               << metrics.mean_rate_hz;
        for(double rate_hz : metrics.rates_hz) {
            output << "," << rate_hz;
        }
        output << "\n";
    };

    for(const RetinotopicContextMetrics &metrics : validation_metrics) {
        writeRow("center_only", "l23e", metrics.validation_site_id, metrics.center_l23e);
        writeRow("center_only", "l23pv", metrics.validation_site_id, metrics.center_l23pv);
        writeRow("center_only", "l23som", metrics.validation_site_id, metrics.center_l23som);
        writeRow("broad_field", "l23e", metrics.validation_site_id, metrics.broad_l23e);
        writeRow("broad_field", "l23pv", metrics.validation_site_id, metrics.broad_l23pv);
        writeRow("broad_field", "l23som", metrics.validation_site_id, metrics.broad_l23som);
    }
}

void writeSizeTuningCsv(
    const std::string &path,
    const std::vector<double> &radii_sites,
    const std::vector<double> &orientations_rad,
    const std::vector<RetinotopicSizeMetrics> &validation_metrics,
    bool include_validation_site_id,
    double som_output_scale)
{
    const std::size_t expected_count = radii_sites.size() * orientations_rad.size();
    for(const RetinotopicSizeMetrics &metrics : validation_metrics) {
        if(metrics.l4e.rates_hz.size() != expected_count
           || metrics.l23e.rates_hz.size() != expected_count
           || metrics.l23pv.rates_hz.size() != expected_count
           || metrics.l23som.rates_hz.size() != expected_count) {
            throw std::runtime_error("Size tuning rate vectors do not match radii/orientation grid.");
        }
    }

    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "radius_sites,population,site_id";
    if(include_validation_site_id) {
        output << ",validation_site_id";
    }
    output << ",som_output_scale,orientation_deg,rate_hz\n";

    auto writeRows = [&](const std::string &population,
                         unsigned int validation_site_id,
                         const PopulationSiteMetrics &metrics) {
        for(std::size_t radius_index = 0; radius_index < radii_sites.size(); radius_index++) {
            for(std::size_t orientation_index = 0; orientation_index < orientations_rad.size(); orientation_index++) {
                const std::size_t trial_index = (radius_index * orientations_rad.size()) + orientation_index;
                output << radii_sites[radius_index] << ","
                       << population << ","
                       << metrics.site_id;
                if(include_validation_site_id) {
                    output << "," << validation_site_id;
                }
                output << ","
                       << som_output_scale << ","
                       << positiveModuloDegrees(radiansToDegrees(orientations_rad[orientation_index])) << ","
                       << metrics.rates_hz[trial_index] << "\n";
            }
        }
    };

    for(const RetinotopicSizeMetrics &metrics : validation_metrics) {
        writeRows("l4e", metrics.validation_site_id, metrics.l4e);
        writeRows("l23e", metrics.validation_site_id, metrics.l23e);
        writeRows("l23pv", metrics.validation_site_id, metrics.l23pv);
        writeRows("l23som", metrics.validation_site_id, metrics.l23som);
    }
}

const char *orientationContextConditionName(unsigned int condition_index)
{
    switch(condition_index) {
    case kOrientationContextCenterOnly:
        return "center_only";
    case kOrientationContextSameSurround:
        return "same_surround";
    case kOrientationContextOrthSurround:
        return "orth_surround";
    case kOrientationContextSurroundSameOnly:
        return "surround_same_only";
    case kOrientationContextSurroundOrthOnly:
        return "surround_orth_only";
    default:
        throw std::runtime_error("Invalid orientation-context condition index.");
    }
}

double suppressionIndex(double center_rate_hz, double context_rate_hz)
{
    if(center_rate_hz <= 0.0) {
        return 0.0;
    }
    return (center_rate_hz - context_rate_hz) / center_rate_hz;
}

void writeOrientationContextAssayCsv(
    const std::string &path,
    const std::vector<OrientationContextSiteMetrics> &metrics_by_site,
    bool include_validation_site_id,
    double som_output_scale)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "condition,site_id";
    if(include_validation_site_id) {
        output << ",validation_site_id";
    }
    output << ",preferred_orientation_deg,stimulus_orientation_deg,orthogonal_orientation_deg"
           << ",aperture_radius_sites,inner_radius_sites,som_output_scale"
           << ",l4e_rate_hz,l23e_rate_hz,l23pv_rate_hz,l23som_rate_hz"
           << ",si_same_l4e,si_orth_l4e,osd_l4e"
           << ",si_same_l23e,si_orth_l23e,osd_l23e"
           << ",surround_same_l23e_ratio,surround_orth_l23e_ratio\n";

    for(const OrientationContextSiteMetrics &metrics : metrics_by_site) {
        const double center_l4e = metrics.l4e_rates_hz[kOrientationContextCenterOnly];
        const double same_l4e = metrics.l4e_rates_hz[kOrientationContextSameSurround];
        const double orth_l4e = metrics.l4e_rates_hz[kOrientationContextOrthSurround];
        const double si_same_l4e = suppressionIndex(center_l4e, same_l4e);
        const double si_orth_l4e = suppressionIndex(center_l4e, orth_l4e);
        const double osd_l4e = si_same_l4e - si_orth_l4e;

        const double center_l23e = metrics.l23e_rates_hz[kOrientationContextCenterOnly];
        const double same_l23e = metrics.l23e_rates_hz[kOrientationContextSameSurround];
        const double orth_l23e = metrics.l23e_rates_hz[kOrientationContextOrthSurround];
        const double surround_same_l23e = metrics.l23e_rates_hz[kOrientationContextSurroundSameOnly];
        const double surround_orth_l23e = metrics.l23e_rates_hz[kOrientationContextSurroundOrthOnly];
        const double si_same_l23e = suppressionIndex(center_l23e, same_l23e);
        const double si_orth_l23e = suppressionIndex(center_l23e, orth_l23e);
        const double osd_l23e = si_same_l23e - si_orth_l23e;
        const double surround_same_ratio = (center_l23e > 0.0) ? (surround_same_l23e / center_l23e) : 0.0;
        const double surround_orth_ratio = (center_l23e > 0.0) ? (surround_orth_l23e / center_l23e) : 0.0;

        for(unsigned int condition_index = 0; condition_index < kOrientationContextConditionCount; condition_index++) {
            const double stimulus_orientation = (
                condition_index == kOrientationContextOrthSurround
                || condition_index == kOrientationContextSurroundOrthOnly)
                ? metrics.orthogonal_orientation_rad
                : metrics.preferred_orientation_rad;
            output << orientationContextConditionName(condition_index) << ","
                   << metrics.site_id;
            if(include_validation_site_id) {
                output << "," << metrics.validation_site_id;
            }
            output << ","
                   << positiveModuloDegrees(radiansToDegrees(metrics.preferred_orientation_rad)) << ","
                   << positiveModuloDegrees(radiansToDegrees(stimulus_orientation)) << ","
                   << positiveModuloDegrees(radiansToDegrees(metrics.orthogonal_orientation_rad)) << ","
                   << metrics.aperture_radius_sites[condition_index] << ","
                   << metrics.inner_radius_sites[condition_index] << ","
                   << som_output_scale << ","
                   << metrics.l4e_rates_hz[condition_index] << ","
                   << metrics.l23e_rates_hz[condition_index] << ","
                   << metrics.l23pv_rates_hz[condition_index] << ","
                   << metrics.l23som_rates_hz[condition_index] << ","
                   << si_same_l4e << ","
                   << si_orth_l4e << ","
                   << osd_l4e << ","
                   << si_same_l23e << ","
                   << si_orth_l23e << ","
                   << osd_l23e << ","
                   << surround_same_ratio << ","
                   << surround_orth_ratio << "\n";
        }
    }
}

double siteRateFromCounts(
    const std::vector<double> &counts,
    const std::vector<TrialWindow> &trials,
    std::size_t trial_index,
    unsigned int site_id,
    unsigned int neurons_per_site)
{
    if(trial_index >= trials.size() || site_id >= v1_genn::kSiteCount) {
        throw std::runtime_error("Site rate lookup is outside the recorded trial/site grid.");
    }
    const std::size_t count_index = (trial_index * v1_genn::kSiteCount) + site_id;
    if(count_index >= counts.size()) {
        throw std::runtime_error("Site spike count vector is too small for requested trial/site.");
    }
    const TrialWindow &trial = trials[trial_index];
    const double measurement_duration_s = (trial.end_ms - trial.measure_start_ms) / 1000.0;
    if(measurement_duration_s <= 0.0) {
        throw std::runtime_error("Trial measurement duration must be positive.");
    }
    return counts[count_index] / (measurement_duration_s * static_cast<double>(neurons_per_site));
}

void writeBlankBaselineCsv(
    const std::string &path,
    const std::vector<TrialWindow> &blank_trials,
    const std::vector<double> &l4e_counts,
    const std::vector<double> &l23e_counts,
    const std::vector<double> &l23pv_counts,
    const std::vector<double> &l23som_counts)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "repeat_index,population,site_id,rate_hz\n";
    const auto writeRows = [&](const std::string &population,
                               const std::vector<double> &counts,
                               unsigned int neurons_per_site) {
        for(std::size_t trial_index = 0; trial_index < blank_trials.size(); trial_index++) {
            for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
                output << trial_index << ","
                       << population << ","
                       << site_id << ","
                       << siteRateFromCounts(counts, blank_trials, trial_index, site_id, neurons_per_site)
                       << "\n";
            }
        }
    };

    writeRows("l4e", l4e_counts, v1_genn::kL4EPerSite);
    writeRows("l23e", l23e_counts, v1_genn::kL23EPerSite);
    writeRows("l23pv", l23pv_counts, v1_genn::kL23PVPerSite);
    writeRows("l23som", l23som_counts, v1_genn::kL23SOMPerSite);
}

void writeContrastSweepCsv(
    const std::string &path,
    const std::vector<ContrastTrialRecord> &records,
    const std::vector<TrialWindow> &contrast_trials,
    const std::vector<double> &l4e_counts,
    const std::vector<double> &l23e_counts,
    const std::vector<double> &l23pv_counts,
    const std::vector<double> &l23som_counts)
{
    if(records.size() != contrast_trials.size()) {
        throw std::runtime_error("Contrast sweep records and trial windows do not align.");
    }

    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "contrast,site_id,validation_site_id,population,orientation_deg,aperture_radius_sites,rate_hz\n";
    const auto writeRow = [&](const ContrastTrialRecord &record,
                              std::size_t trial_index,
                              const std::string &population,
                              const std::vector<double> &counts,
                              unsigned int neurons_per_site) {
        output << record.contrast << ","
               << record.site_id << ","
               << record.validation_site_id << ","
               << population << ","
               << positiveModuloDegrees(radiansToDegrees(record.orientation_rad)) << ","
               << record.aperture_radius_sites << ","
               << siteRateFromCounts(counts, contrast_trials, trial_index, record.site_id, neurons_per_site)
               << "\n";
    };

    for(std::size_t trial_index = 0; trial_index < records.size(); trial_index++) {
        const ContrastTrialRecord &record = records[trial_index];
        writeRow(record, trial_index, "l4e", l4e_counts, v1_genn::kL4EPerSite);
        writeRow(record, trial_index, "l23e", l23e_counts, v1_genn::kL23EPerSite);
        writeRow(record, trial_index, "l23pv", l23pv_counts, v1_genn::kL23PVPerSite);
        writeRow(record, trial_index, "l23som", l23som_counts, v1_genn::kL23SOMPerSite);
    }
}

void writeMetricRow(std::ofstream &output, const std::string &metric, double value)
{
    output << metric << "," << value << "\n";
}

void writeL4IntersiteDiagnosticsCsv(
    const std::string &path,
    const L4IntersiteConfig &config,
    const std::vector<double> &radii_sites,
    const std::vector<RetinotopicSizeMetrics> &size_validation_metrics,
    const SweepResult &baseline,
    const SweepResult &post)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "metric,value\n";

    writeMetricRow(output, "enabled", config.enabled ? 1.0 : 0.0);
    writeMetricRow(output, "radius_sites", static_cast<double>(config.radius));
    writeMetricRow(output, "weight_scale", config.weight_scale);
    writeMetricRow(output, "l4ee_scale", config.l4ee_scale);
    writeMetricRow(output, "l4e_to_l4pv_scale", config.l4e_to_l4pv_scale);
    writeMetricRow(output, "l4pv_to_l4e_scale", config.l4pv_to_l4e_scale);
    writeMetricRow(output, "l4ee_base_weight", v1_genn::kL4EEWeight);
    writeMetricRow(output, "l4e_to_l4pv_base_weight", v1_genn::kL4EToPVWeight);
    writeMetricRow(output, "l4pv_to_l4e_base_weight", v1_genn::kL4PVToEWeight);
    writeMetricRow(output, "l4ee_effective_weight", config.enabled ? (v1_genn::kL4EEWeight * config.l4ee_scale) : 0.0);
    writeMetricRow(output, "l4e_to_l4pv_effective_weight", config.enabled ? (v1_genn::kL4EToPVWeight * config.l4e_to_l4pv_scale) : 0.0);
    writeMetricRow(output, "l4pv_to_l4e_effective_weight", config.enabled ? (v1_genn::kL4PVToEWeight * config.l4pv_to_l4e_scale) : 0.0);

    const auto l4ee_edges = config.enabled ? buildLocalIntersiteConnectivity(
        v1_genn::kL4EPerSite,
        v1_genn::kL4EPerSite,
        config.radius) : std::vector<std::pair<unsigned int, unsigned int>>{};
    const auto l4e_pv_edges = config.enabled ? buildLocalIntersiteConnectivity(
        v1_genn::kL4EPerSite,
        v1_genn::kL4PVPerSite,
        config.radius) : std::vector<std::pair<unsigned int, unsigned int>>{};
    const auto l4pv_e_edges = config.enabled ? buildLocalIntersiteConnectivity(
        v1_genn::kL4PVPerSite,
        v1_genn::kL4EPerSite,
        config.radius) : std::vector<std::pair<unsigned int, unsigned int>>{};

    const ConnectivityStats l4ee_stats = summarizeConnectivity(
        l4ee_edges,
        v1_genn::kL4EPerSite,
        v1_genn::kL4EPerSite,
        config.radius);
    const ConnectivityStats l4e_pv_stats = summarizeConnectivity(
        l4e_pv_edges,
        v1_genn::kL4EPerSite,
        v1_genn::kL4PVPerSite,
        config.radius);
    const ConnectivityStats l4pv_e_stats = summarizeConnectivity(
        l4pv_e_edges,
        v1_genn::kL4PVPerSite,
        v1_genn::kL4EPerSite,
        config.radius);

    auto writeStats = [&](const std::string &prefix, const ConnectivityStats &stats) {
        writeMetricRow(output, prefix + "_edge_count", static_cast<double>(stats.edge_count));
        writeMetricRow(output, prefix + "_mean_distance_sites", stats.mean_distance_sites);
        writeMetricRow(output, prefix + "_max_distance_sites", stats.max_distance_sites);
        writeMetricRow(output, prefix + "_same_site_fraction", stats.same_site_fraction);
        writeMetricRow(output, prefix + "_beyond_radius_fraction", stats.beyond_radius_fraction);
    };
    writeStats("l4ee", l4ee_stats);
    writeStats("l4e_to_l4pv", l4e_pv_stats);
    writeStats("l4pv_to_l4e", l4pv_e_stats);
    writeMetricRow(
        output,
        "max_projection_distance_sites",
        std::max(l4ee_stats.max_distance_sites, std::max(l4e_pv_stats.max_distance_sites, l4pv_e_stats.max_distance_sites)));
    writeMetricRow(
        output,
        "max_same_site_fraction",
        std::max(l4ee_stats.same_site_fraction, std::max(l4e_pv_stats.same_site_fraction, l4pv_e_stats.same_site_fraction)));
    writeMetricRow(
        output,
        "max_beyond_radius_fraction",
        std::max(l4ee_stats.beyond_radius_fraction, std::max(l4e_pv_stats.beyond_radius_fraction, l4pv_e_stats.beyond_radius_fraction)));

    writeMetricRow(output, "validation_site_count", static_cast<double>(size_validation_metrics.size()));
    if(!size_validation_metrics.empty() && !radii_sites.empty()) {
        std::vector<double> l4_rates_by_radius(radii_sites.size(), 0.0);
        for(const RetinotopicSizeMetrics &metrics : size_validation_metrics) {
            if(metrics.l4e.rates_hz.size() % radii_sites.size() != 0u) {
                throw std::runtime_error("L4 intersite diagnostics expected size tuning rates on radii/orientation grid.");
            }
            const std::size_t orientation_count = metrics.l4e.rates_hz.size() / radii_sites.size();
            if(orientation_count == 0u) {
                throw std::runtime_error("L4 intersite diagnostics requires at least one orientation.");
            }
            for(std::size_t radius_index = 0; radius_index < radii_sites.size(); radius_index++) {
                double site_radius_sum = 0.0;
                for(std::size_t orientation_index = 0; orientation_index < orientation_count; orientation_index++) {
                    site_radius_sum += metrics.l4e.rates_hz[(radius_index * orientation_count) + orientation_index];
                }
                l4_rates_by_radius[radius_index] += site_radius_sum
                    / (static_cast<double>(orientation_count) * static_cast<double>(size_validation_metrics.size()));
            }
        }

        const auto peak_iter = std::max_element(l4_rates_by_radius.begin(), l4_rates_by_radius.end());
        const std::size_t peak_index = static_cast<std::size_t>(peak_iter - l4_rates_by_radius.begin());
        const double peak_rate = *peak_iter;
        const double small_rate = l4_rates_by_radius.front();
        const double large_rate = l4_rates_by_radius.back();
        writeMetricRow(output, "l4_size_peak_radius_sites", radii_sites[peak_index]);
        writeMetricRow(output, "l4_size_peak_rate_hz", peak_rate);
        writeMetricRow(output, "l4_size_small_rate_hz", small_rate);
        writeMetricRow(output, "l4_size_large_rate_hz", large_rate);
        writeMetricRow(output, "l4_size_small_peak_ratio", peak_rate > 0.0 ? (small_rate / peak_rate) : 0.0);
        writeMetricRow(output, "l4_size_large_peak_ratio", peak_rate > 0.0 ? (large_rate / peak_rate) : 0.0);
    }
    writeMetricRow(output, "baseline_l4_median_osi", baseline.l4_median_osi);
    writeMetricRow(output, "post_l4_median_osi", post.l4_median_osi);
    writeMetricRow(output, "baseline_l4_map_error_deg_median", baseline.l4_median_map_error_deg);
    writeMetricRow(output, "post_l4_map_error_deg_median", post.l4_median_map_error_deg);
}

void writeL23EESpecificityCsv(
    const std::string &path,
    const std::vector<float> &weights_before,
    const std::vector<float> &weights_after,
    const std::vector<std::pair<unsigned int, unsigned int>> &edges,
    const std::vector<CellTuningMetrics> &l23e_cell_tuning)
{
    if(weights_before.size() != weights_after.size()) {
        throw std::runtime_error("L23E->L23E specificity export requires aligned before/after weight vectors.");
    }
    if(weights_after.size() % v1_genn::kNumL23E != 0u) {
        throw std::runtime_error("L23E->L23E specificity export expected row-major sparse weight capacity.");
    }
    if(l23e_cell_tuning.size() != v1_genn::kNumL23E) {
        throw std::runtime_error("L23E->L23E specificity export requires one tuning row per L23E cell.");
    }
    const std::size_t max_row_length = weights_after.size() / v1_genn::kNumL23E;

    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    output << std::fixed << std::setprecision(6);
    output << "synapse_index,pre_id,post_id,pre_site,post_site,distance_sites,"
           << "pre_pref_deg,post_pref_deg,delta_pref_deg,w_before,w_after,delta_w,"
           << "pre_peak_hz,post_peak_hz,response_corr\n";
    unsigned int previous_pre_id = std::numeric_limits<unsigned int>::max();
    std::size_t row_active_index = 0;
    for(const auto &edge : edges) {
        const unsigned int pre_id = edge.first;
        const unsigned int post_id = edge.second;
        if(pre_id != previous_pre_id) {
            previous_pre_id = pre_id;
            row_active_index = 0;
        }
        if(row_active_index >= max_row_length) {
            throw std::runtime_error("L23E->L23E specificity export exceeded sparse row capacity.");
        }

        const std::size_t synapse_index = (static_cast<std::size_t>(pre_id) * max_row_length) + row_active_index;
        row_active_index++;

        const double w_before = static_cast<double>(weights_before[synapse_index]);
        const double w_after = static_cast<double>(weights_after[synapse_index]);
        if(w_before == 0.0 && w_after == 0.0) {
            continue;
        }

        const unsigned int pre_site = pre_id / v1_genn::kL23EPerSite;
        const unsigned int post_site = post_id / v1_genn::kL23EPerSite;
        const auto pre_xy = v1_genn::siteIndexToXY(pre_site);
        const auto post_xy = v1_genn::siteIndexToXY(post_site);
        const double dx = static_cast<double>(pre_xy.first) - static_cast<double>(post_xy.first);
        const double dy = static_cast<double>(pre_xy.second) - static_cast<double>(post_xy.second);
        const double distance_sites = std::sqrt((dx * dx) + (dy * dy));
        const double pre_pref_rad = v1_genn::sitePreferredOrientationFromIndex(pre_site);
        const double post_pref_rad = v1_genn::sitePreferredOrientationFromIndex(post_site);
        const double delta_pref_rad = v1_genn::circularOrientationDifference(pre_pref_rad, post_pref_rad);
        const CellTuningMetrics &pre_tuning = l23e_cell_tuning.at(pre_id);
        const CellTuningMetrics &post_tuning = l23e_cell_tuning.at(post_id);
        const double corr = responseCorrelation(pre_tuning.rates_hz, post_tuning.rates_hz);

        output << synapse_index << ","
               << pre_id << ","
               << post_id << ","
               << pre_site << ","
               << post_site << ","
               << distance_sites << ","
               << positiveModuloDegrees(radiansToDegrees(pre_pref_rad)) << ","
               << positiveModuloDegrees(radiansToDegrees(post_pref_rad)) << ","
               << radiansToDegrees(delta_pref_rad) << ","
               << w_before << ","
               << w_after << ","
               << (w_after - w_before) << ","
               << pre_tuning.peak_rate_hz << ","
               << post_tuning.peak_rate_hz << ","
               << corr << "\n";
    }
}

void writeSummaryFiles(
    const std::string &output_prefix,
    const SweepResult &baseline,
    const SweepResult &post,
    const WeightStats &weights_before,
    const WeightStats &weights_after,
    const std::vector<NamedWeightStats> &additional_weight_stats,
    const std::vector<PopulationRateSummary> &subtype_rates,
    const std::vector<ContextValidationSummary> &context_validation,
    const TrainingGratingConfig &training_grating_config,
    double training_grating_phase_slot_ms,
    const L4L23OrientationConfig &l4_l23_orientation_config,
    std::size_t l4_l23_edge_count,
    double l4_l23_weights_before_nonzero_fraction,
    const L23EELognormalInitConfig &l23ee_lognormal_init_config,
    std::size_t l23ee_initial_active_count,
    double l23ee_initial_active_mean,
    double l23ee_initial_active_gini,
    double l23ee_initial_top10_mass_share,
    double l23pv_context_output_scale,
    const OrientationContextAssayConfig &orientation_context_assay_config,
    const SensoryAssayConfig &sensory_assay_config)
{
    const double l23_osi_delta = post.l23_median_osi - baseline.l23_median_osi;
    const bool feedforward_orientation_prior_enabled = (l4_l23_orientation_config.bias_strength > 0.0);
    const bool neutral_density_match_active =
        l4_l23_orientation_config.neutral_density_match_enabled
        && l4_l23_orientation_config.bias_strength == 0.0;

    std::ofstream csv((output_prefix + "_summary.csv").c_str());
    if(!csv) {
        throw std::runtime_error("Unable to open output file: " + output_prefix + "_summary.csv");
    }
    csv << std::fixed << std::setprecision(6);
    csv << "metric,value\n";
    csv << "baseline_l4_median_osi," << baseline.l4_median_osi << "\n";
    csv << "baseline_l23_median_osi," << baseline.l23_median_osi << "\n";
    csv << "post_l4_median_osi," << post.l4_median_osi << "\n";
    csv << "post_l23_median_osi," << post.l23_median_osi << "\n";
    csv << "baseline_l4_map_error_deg_median," << baseline.l4_median_map_error_deg << "\n";
    csv << "post_l4_map_error_deg_median," << post.l4_median_map_error_deg << "\n";
    csv << "l23_median_osi_delta," << l23_osi_delta << "\n";
    csv << "weights_before_count," << weights_before.count << "\n";
    csv << "weights_before_min," << weights_before.min << "\n";
    csv << "weights_before_mean," << weights_before.mean << "\n";
    csv << "weights_before_max," << weights_before.max << "\n";
    csv << "weights_after_count," << weights_after.count << "\n";
    csv << "weights_after_min," << weights_after.min << "\n";
    csv << "weights_after_mean," << weights_after.mean << "\n";
    csv << "weights_after_max," << weights_after.max << "\n";
    csv << "training_grating_mode_code," << (training_grating_config.phase_drift_enabled ? 1.0 : 0.0) << "\n";
    csv << "training_grating_phase_count," << training_grating_config.phase_count << "\n";
    csv << "training_grating_phase_slot_ms," << training_grating_phase_slot_ms << "\n";
    csv << "training_grating_counterbalance_enabled," << (training_grating_config.counterbalance_direction ? 1.0 : 0.0) << "\n";
    csv << "l4_l23_orientation_bias_strength," << l4_l23_orientation_config.bias_strength << "\n";
    csv << "l4_l23_feedforward_orientation_prior_enabled," << (feedforward_orientation_prior_enabled ? 1.0 : 0.0) << "\n";
    csv << "l4_l23_orientation_neutral_density_match_enabled,"
        << (l4_l23_orientation_config.neutral_density_match_enabled ? 1.0 : 0.0) << "\n";
    csv << "l4_l23_orientation_neutral_density_match_active,"
        << (neutral_density_match_active ? 1.0 : 0.0) << "\n";
    csv << "l4_l23_orientation_neutral_probability_scale,"
        << l4_l23_orientation_config.neutral_probability_scale << "\n";
    csv << "l4_l23_edge_count," << l4_l23_edge_count << "\n";
    csv << "l4_l23_weights_before_nonzero_fraction," << l4_l23_weights_before_nonzero_fraction << "\n";
    csv << "l4_l23_weights_before_mean_all_slots," << weights_before.mean << "\n";
    csv << "l23ee_lognormal_init_enabled," << (l23ee_lognormal_init_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "l23ee_lognormal_init_sigma," << l23ee_lognormal_init_config.sigma << "\n";
    csv << "l23ee_lognormal_init_target_mean," << v1_genn::kL23EEWeight << "\n";
    csv << "l23ee_lognormal_init_wmin," << kL23EEStdpWeightMin << "\n";
    csv << "l23ee_lognormal_init_wmax," << kL23EEStdpWeightMax << "\n";
    csv << "l23ee_initial_active_count," << l23ee_initial_active_count << "\n";
    csv << "l23ee_initial_active_mean," << l23ee_initial_active_mean << "\n";
    csv << "l23ee_initial_active_gini," << l23ee_initial_active_gini << "\n";
    csv << "l23ee_initial_top10_mass_share," << l23ee_initial_top10_mass_share << "\n";
    csv << "l23pv_context_output_scale," << l23pv_context_output_scale << "\n";
    csv << "l23pv_context_output_ablation_active," << (l23pv_context_output_scale != 1.0 ? 1.0 : 0.0) << "\n";
    csv << "inhibitory_orientation_rule_enabled,0.000000\n";
    csv << "orientation_context_assay_enabled," << (orientation_context_assay_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "orientation_context_center_radius_sites," << orientation_context_assay_config.center_radius_sites << "\n";
    csv << "orientation_context_broad_radius_sites," << orientation_context_assay_config.broad_radius_sites << "\n";
    csv << "orientation_context_surround_inner_radius_sites," << orientation_context_assay_config.surround_inner_radius_sites << "\n";
    csv << "orientation_context_annular_surround_only_enabled," << (orientation_context_assay_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "orientation_context_assay_orientation_source_code," << (orientation_context_assay_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "sensory_assay_enabled," << (sensory_assay_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "blank_repeat_count," << sensory_assay_config.blank_repeat_count << "\n";
    csv << "contrast_sweep_count," << sensory_assay_config.contrasts.size() << "\n";
    csv << "contrast_sweep_radius_sites," << sensory_assay_config.contrast_radius_sites << "\n";
    for(const NamedWeightStats &summary : additional_weight_stats) {
        csv << summary.name << "_weights_before_count," << summary.before.count << "\n";
        csv << summary.name << "_weights_before_min," << summary.before.min << "\n";
        csv << summary.name << "_weights_before_mean," << summary.before.mean << "\n";
        csv << summary.name << "_weights_before_max," << summary.before.max << "\n";
        csv << summary.name << "_weights_after_count," << summary.after.count << "\n";
        csv << summary.name << "_weights_after_min," << summary.after.min << "\n";
        csv << summary.name << "_weights_after_mean," << summary.after.mean << "\n";
        csv << summary.name << "_weights_after_max," << summary.after.max << "\n";
    }
    for(const PopulationRateSummary &summary : subtype_rates) {
        csv << "baseline_" << summary.name << "_mean_rate_hz," << summary.baseline_mean_rate_hz << "\n";
        csv << "post_" << summary.name << "_mean_rate_hz," << summary.post_mean_rate_hz << "\n";
        csv << summary.name << "_mean_rate_delta_hz," << (summary.post_mean_rate_hz - summary.baseline_mean_rate_hz) << "\n";
    }
    for(const ContextValidationSummary &summary : context_validation) {
        csv << summary.condition << "_central_l23e_mean_rate_hz," << summary.l23e_mean_rate_hz << "\n";
        csv << summary.condition << "_central_l23pv_mean_rate_hz," << summary.l23pv_mean_rate_hz << "\n";
        csv << summary.condition << "_central_l23som_mean_rate_hz," << summary.l23som_mean_rate_hz << "\n";
    }

    std::ofstream text((output_prefix + "_summary.txt").c_str());
    if(!text) {
        throw std::runtime_error("Unable to open output file: " + output_prefix + "_summary.txt");
    }
    text << std::fixed << std::setprecision(6);
    text << "baseline_l4_median_osi=" << baseline.l4_median_osi << "\n";
    text << "baseline_l23_median_osi=" << baseline.l23_median_osi << "\n";
    text << "post_l4_median_osi=" << post.l4_median_osi << "\n";
    text << "post_l23_median_osi=" << post.l23_median_osi << "\n";
    text << "baseline_l4_map_error_deg_median=" << baseline.l4_median_map_error_deg << "\n";
    text << "post_l4_map_error_deg_median=" << post.l4_median_map_error_deg << "\n";
    text << "l23_median_osi_delta=" << l23_osi_delta << "\n";
    text << "weights_before=count:" << weights_before.count
         << ",min:" << weights_before.min
         << ",mean:" << weights_before.mean
         << ",max:" << weights_before.max << "\n";
    text << "weights_after=count:" << weights_after.count
         << ",min:" << weights_after.min
         << ",mean:" << weights_after.mean
         << ",max:" << weights_after.max << "\n";
    text << "training_grating_mode=" << training_grating_config.mode << "\n";
    text << "training_grating_phase_count=" << training_grating_config.phase_count << "\n";
    text << "training_grating_phase_slot_ms=" << training_grating_phase_slot_ms << "\n";
    text << "training_grating_phase_order="
         << (training_grating_config.phase_drift_enabled
             ? "orientation_epoch_offset_bidirectional_counterbalanced"
             : "legacy_single_static_phase_per_trial")
         << "\n";
    text << "training_grating_counterbalance_enabled="
         << (training_grating_config.counterbalance_direction ? 1 : 0)
         << "\n";
    text << "l4_l23_orientation_bias_strength=" << l4_l23_orientation_config.bias_strength << "\n";
    text << "l4_l23_feedforward_orientation_prior_enabled="
         << (feedforward_orientation_prior_enabled ? 1 : 0) << "\n";
    text << "l4_l23_orientation_neutral_density_match_enabled="
         << (l4_l23_orientation_config.neutral_density_match_enabled ? 1 : 0) << "\n";
    text << "l4_l23_orientation_neutral_density_match_active="
         << (neutral_density_match_active ? 1 : 0) << "\n";
    text << "l4_l23_orientation_neutral_probability_scale="
         << l4_l23_orientation_config.neutral_probability_scale << "\n";
    text << "l4_l23_edge_count=" << l4_l23_edge_count << "\n";
    text << "l4_l23_weights_before_nonzero_fraction="
         << l4_l23_weights_before_nonzero_fraction << "\n";
    text << "l4_l23_weights_before_mean_all_slots=" << weights_before.mean << "\n";
    text << "l23ee_lognormal_init_enabled="
         << (l23ee_lognormal_init_config.enabled ? 1 : 0) << "\n";
    text << "l23ee_lognormal_init_sigma=" << l23ee_lognormal_init_config.sigma << "\n";
    text << "l23ee_lognormal_init_target_mean=" << v1_genn::kL23EEWeight << "\n";
    text << "l23ee_lognormal_init_wmin=" << kL23EEStdpWeightMin << "\n";
    text << "l23ee_lognormal_init_wmax=" << kL23EEStdpWeightMax << "\n";
    text << "l23ee_initial_active_count=" << l23ee_initial_active_count << "\n";
    text << "l23ee_initial_active_mean=" << l23ee_initial_active_mean << "\n";
    text << "l23ee_initial_active_gini=" << l23ee_initial_active_gini << "\n";
    text << "l23ee_initial_top10_mass_share=" << l23ee_initial_top10_mass_share << "\n";
    text << "l23pv_context_output_scale=" << l23pv_context_output_scale << "\n";
    text << "l23pv_context_output_ablation_active="
         << (l23pv_context_output_scale != 1.0 ? 1 : 0) << "\n";
    text << "inhibitory_orientation_rule_enabled=0\n";
    text << "orientation_context_assay_enabled="
         << (orientation_context_assay_config.enabled ? 1 : 0) << "\n";
    text << "orientation_context_protocol="
         << (orientation_context_assay_config.enabled
             ? "center_preferred_same_and_orthogonal_annular_surround"
             : "disabled")
         << "\n";
    text << "orientation_context_center_radius_sites="
         << orientation_context_assay_config.center_radius_sites << "\n";
    text << "orientation_context_broad_radius_sites="
         << orientation_context_assay_config.broad_radius_sites << "\n";
    text << "orientation_context_surround_inner_radius_sites="
         << orientation_context_assay_config.surround_inner_radius_sites << "\n";
    text << "orientation_context_assay_orientation_source="
         << (orientation_context_assay_config.enabled
             ? "site_map_preferred_orientation"
             : "disabled")
         << "\n";
    text << "sensory_assay_enabled="
         << (sensory_assay_config.enabled ? 1 : 0) << "\n";
    text << "blank_repeat_count=" << sensory_assay_config.blank_repeat_count << "\n";
    text << "contrast_sweep_count=" << sensory_assay_config.contrasts.size() << "\n";
    text << "contrast_sweep_radius_sites=" << sensory_assay_config.contrast_radius_sites << "\n";
    for(const NamedWeightStats &summary : additional_weight_stats) {
        text << summary.name << "_weights_before=count:" << summary.before.count
             << ",min:" << summary.before.min
             << ",mean:" << summary.before.mean
             << ",max:" << summary.before.max << "\n";
        text << summary.name << "_weights_after=count:" << summary.after.count
             << ",min:" << summary.after.min
             << ",mean:" << summary.after.mean
             << ",max:" << summary.after.max << "\n";
    }
    for(const PopulationRateSummary &summary : subtype_rates) {
        text << summary.name << "_baseline_mean_rate_hz=" << summary.baseline_mean_rate_hz << "\n";
        text << summary.name << "_post_mean_rate_hz=" << summary.post_mean_rate_hz << "\n";
        text << summary.name << "_mean_rate_delta_hz=" << (summary.post_mean_rate_hz - summary.baseline_mean_rate_hz) << "\n";
    }
    for(const ContextValidationSummary &summary : context_validation) {
        text << summary.condition << "_central_l23e_mean_rate_hz=" << summary.l23e_mean_rate_hz << "\n";
        text << summary.condition << "_central_l23pv_mean_rate_hz=" << summary.l23pv_mean_rate_hz << "\n";
        text << summary.condition << "_central_l23som_mean_rate_hz=" << summary.l23som_mean_rate_hz << "\n";
    }
}

}  // namespace

void modelDefinition(GeNN::ModelSpec &model)
{
    model.setDT(v1_genn::kDtMs);
    model.setName(v1_genn::kModelName);

    auto *l4e = model.addNeuronPopulation<V1LIF>(
        "L4E",
        v1_genn::kNumL4E,
        makeLIFParameters(v1_genn::kExcitatoryLIF),
        makeLIFVariables(v1_genn::kExcitatoryLIF, GeNN::uninitialisedVar()));

    auto *l4pv = model.addNeuronPopulation<V1LIF>(
        "L4PV",
        v1_genn::kNumL4PV,
        makeLIFParameters(v1_genn::kPVLIF),
        makeLIFVariables(v1_genn::kPVLIF, 0.0));

    auto *l4som = model.addNeuronPopulation<V1LIF>(
        "L4SOM",
        v1_genn::kNumL4SOM,
        makeLIFParameters(v1_genn::kSOMLIF),
        makeLIFVariables(v1_genn::kSOMLIF, 0.0));

    auto *l23e = model.addNeuronPopulation<V1LIF>(
        "L23E",
        v1_genn::kNumL23E,
        makeLIFParameters(v1_genn::kExcitatoryLIF),
        makeLIFVariables(v1_genn::kExcitatoryLIF, 0.0));

    auto *l23pv = model.addNeuronPopulation<V1LIF>(
        "L23PV",
        v1_genn::kNumL23PV,
        makeLIFParameters(v1_genn::kPVLIF),
        makeLIFVariables(v1_genn::kPVLIF, 0.0));

    auto *l23som = model.addNeuronPopulation<V1LIF>(
        "L23SOM",
        v1_genn::kNumL23SOM,
        makeLIFParameters(v1_genn::kSOMLIF),
        makeLIFVariables(v1_genn::kSOMLIF, 0.0));

    auto *l23vip = model.addNeuronPopulation<V1LIF>(
        "L23VIP",
        v1_genn::kNumL23VIP,
        makeLIFParameters(v1_genn::kVIPLIF),
        makeLIFVariables(v1_genn::kVIPLIF, 0.0));

    l4e->setSpikeRecordingEnabled(true);
    l4pv->setSpikeRecordingEnabled(true);
    l4som->setSpikeRecordingEnabled(true);
    l23e->setSpikeRecordingEnabled(true);
    l23pv->setSpikeRecordingEnabled(true);
    l23som->setSpikeRecordingEnabled(true);
    l23vip->setSpikeRecordingEnabled(true);

    const auto l4_ee_patch = makePatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL4EPerSite,
        v1_genn::kL4LocalRadius,
        true);
    const auto l4_e_pv_patch = makePatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL4PVPerSite,
        v1_genn::kL4LocalRadius,
        false);
    const auto l4_e_som_patch = makePatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL4SOMPerSite,
        v1_genn::kL4LocalRadius,
        false);
    const auto l4_pv_e_patch = makePatchParameters(
        v1_genn::kL4PVPerSite,
        v1_genn::kL4EPerSite,
        v1_genn::kL4LocalRadius,
        false);
    const auto l4_pv_pv_patch = makePatchParameters(
        v1_genn::kL4PVPerSite,
        v1_genn::kL4PVPerSite,
        v1_genn::kL4LocalRadius,
        true);
    const auto l4_som_e_patch = makePatchParameters(
        v1_genn::kL4SOMPerSite,
        v1_genn::kL4EPerSite,
        v1_genn::kL4LocalRadius,
        false);
    const auto l4_som_pv_patch = makePatchParameters(
        v1_genn::kL4SOMPerSite,
        v1_genn::kL4PVPerSite,
        v1_genn::kL4LocalRadius,
        false);
    const L4IntersiteConfig l4_intersite_config = getL4IntersiteConfig();
    const auto l4_ee_intersite_patch = makeIntersitePatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL4EPerSite,
        l4_intersite_config.radius);
    const auto l4_e_pv_intersite_patch = makeIntersitePatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL4PVPerSite,
        l4_intersite_config.radius);
    const auto l4_pv_e_intersite_patch = makeIntersitePatchParameters(
        v1_genn::kL4PVPerSite,
        v1_genn::kL4EPerSite,
        l4_intersite_config.radius);

    const auto ff_e_patch = makeOrientationBiasedPatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kFeedforwardRadius);
    const auto ff_i_patch = makePatchParameters(
        v1_genn::kL4EPerSite,
        v1_genn::kL23PVPerSite,
        v1_genn::kFeedforwardRadius,
        false);

    const auto l23_ee_patch = makeSparseDistancePatchParameters(
        v1_genn::kL23EPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kL23LocalRadius,
        true,
        kL23ERecurrentPeakProbability,
        kL23ERecurrentDistanceSigmaSq);
    const auto l23_e_pv_patch = makePatchParameters(
        v1_genn::kL23EPerSite,
        v1_genn::kL23PVPerSite,
        v1_genn::kL23LocalRadius,
        false);
    const auto l23_e_som_patch = makePatchParameters(
        v1_genn::kL23EPerSite,
        v1_genn::kL23SOMPerSite,
        v1_genn::kL23SOMInputRadius,
        false);
    const auto l23_e_vip_patch = makePatchParameters(
        v1_genn::kL23EPerSite,
        v1_genn::kL23VIPPerSite,
        v1_genn::kL23LocalRadius,
        false);
    const auto l23_pv_e_patch = makePatchParameters(
        v1_genn::kL23PVPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kL23LocalRadius,
        false);
    const auto l23_pv_pv_patch = makePatchParameters(
        v1_genn::kL23PVPerSite,
        v1_genn::kL23PVPerSite,
        v1_genn::kL23LocalRadius,
        true);
    const auto l23_som_e_patch = makePatchParameters(
        v1_genn::kL23SOMPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kL23SOMOutputRadius,
        false);
    const auto l23_som_pv_patch = makePatchParameters(
        v1_genn::kL23SOMPerSite,
        v1_genn::kL23PVPerSite,
        v1_genn::kL23SOMOutputRadius,
        false);
    const auto l23_som_vip_patch = makePatchParameters(
        v1_genn::kL23SOMPerSite,
        v1_genn::kL23VIPPerSite,
        v1_genn::kL23SOMOutputRadius,
        false);
    const auto l23_vip_som_patch = makePatchParameters(
        v1_genn::kL23VIPPerSite,
        v1_genn::kL23SOMPerSite,
        v1_genn::kL23LocalRadius,
        false);

    addLocalProjection(
        model,
        "L4E_to_L4E",
        l4e,
        l4e,
        v1_genn::kL4EEWeight,
        v1_genn::kExcTauSynMs,
        l4_ee_patch);
    addLocalProjection(
        model,
        "L4E_to_L4PV",
        l4e,
        l4pv,
        v1_genn::kL4EToPVWeight,
        v1_genn::kExcTauSynMs,
        l4_e_pv_patch);
    addLocalProjection(
        model,
        "L4E_to_L4SOM",
        l4e,
        l4som,
        v1_genn::kL4EToSOMWeight,
        v1_genn::kExcTauSynMs,
        l4_e_som_patch);
    addLocalProjection(
        model,
        "L4PV_to_L4E",
        l4pv,
        l4e,
        v1_genn::kL4PVToEWeight,
        v1_genn::kPVInhTauSynMs,
        l4_pv_e_patch);
    addLocalProjection(
        model,
        "L4PV_to_L4PV",
        l4pv,
        l4pv,
        v1_genn::kL4PVToPVWeight,
        v1_genn::kPVInhTauSynMs,
        l4_pv_pv_patch);
    addLocalProjection(
        model,
        "L4SOM_to_L4E",
        l4som,
        l4e,
        v1_genn::kL4SOMToEWeight,
        v1_genn::kSOMInhTauSynMs,
        l4_som_e_patch);
    addLocalProjection(
        model,
        "L4SOM_to_L4PV",
        l4som,
        l4pv,
        v1_genn::kL4SOMToPVWeight,
        v1_genn::kSOMInhTauSynMs,
        l4_som_pv_patch);
    if(l4_intersite_config.enabled) {
        addLocalIntersiteProjection(
            model,
            "L4E_to_L4E_intersite",
            l4e,
            l4e,
            v1_genn::kL4EEWeight * l4_intersite_config.l4ee_scale,
            v1_genn::kExcTauSynMs,
            l4_ee_intersite_patch);
        addLocalIntersiteProjection(
            model,
            "L4E_to_L4PV_intersite",
            l4e,
            l4pv,
            v1_genn::kL4EToPVWeight * l4_intersite_config.l4e_to_l4pv_scale,
            v1_genn::kExcTauSynMs,
            l4_e_pv_intersite_patch);
        addLocalIntersiteProjection(
            model,
            "L4PV_to_L4E_intersite",
            l4pv,
            l4e,
            v1_genn::kL4PVToEWeight * l4_intersite_config.l4pv_to_l4e_scale,
            v1_genn::kPVInhTauSynMs,
            l4_pv_e_intersite_patch);
    }

    // Plastic feedforward excitation is gated in simulate() by setting Aplus/Aminus.
    addPlasticOrientationBiasedProjection(
        model,
        "L4E_to_L23E",
        l4e,
        l23e,
        v1_genn::kL4EToL23EWeight,
        v1_genn::kExcTauSynMs,
        ff_e_patch);
    addLocalProjection(
        model,
        "L4E_to_L23PV",
        l4e,
        l23pv,
        v1_genn::kL4EToL23PVWeight,
        v1_genn::kExcTauSynMs,
        ff_i_patch);

    addPlasticSparseDistanceProjection(
        model,
        "L23E_to_L23E",
        l23e,
        l23e,
        v1_genn::kL23EEWeight,
        v1_genn::kExcTauSynMs,
        kL23EEStdpTauPlusMs,
        kL23EEStdpTauMinusMs,
        kL23EEStdpWeightMin,
        kL23EEStdpWeightMax,
        l23_ee_patch);
    addLocalProjection(
        model,
        "L23E_to_L23PV",
        l23e,
        l23pv,
        v1_genn::kL23EToPVWeight,
        v1_genn::kExcTauSynMs,
        l23_e_pv_patch);
    addLocalProjection(
        model,
        "L23E_to_L23SOM",
        l23e,
        l23som,
        v1_genn::kL23EToSOMWeight,
        v1_genn::kExcTauSynMs,
        l23_e_som_patch);
    addLocalProjection(
        model,
        "L23E_to_L23VIP",
        l23e,
        l23vip,
        v1_genn::kL23EToVIPWeight,
        v1_genn::kExcTauSynMs,
        l23_e_vip_patch);
    addHomeostaticInhibitoryProjection(
        model,
        "L23PV_to_L23E",
        l23pv,
        l23e,
        v1_genn::kL23PVToEWeight,
        v1_genn::kPVInhTauSynMs,
        makeHomeostaticInhibitoryParameters(
            kDefaultL23PVHomeostaticTargetHz,
            kL23PVToL23EWeightMin,
            kL23PVToL23EWeightMax),
        l23_pv_e_patch);
    addLocalProjection(
        model,
        "L23PV_to_L23PV",
        l23pv,
        l23pv,
        v1_genn::kL23PVToPVWeight,
        v1_genn::kPVInhTauSynMs,
        l23_pv_pv_patch);
    addHomeostaticInhibitoryProjection(
        model,
        "L23SOM_to_L23E",
        l23som,
        l23e,
        v1_genn::kL23SOMToEWeight,
        v1_genn::kSOMInhTauSynMs,
        makeHomeostaticInhibitoryParameters(
            kDefaultL23SOMHomeostaticTargetHz,
            kL23SOMToL23EWeightMin,
            kL23SOMToL23EWeightMax),
        l23_som_e_patch);
    addLocalProjection(
        model,
        "L23SOM_to_L23PV",
        l23som,
        l23pv,
        v1_genn::kL23SOMToPVWeight,
        v1_genn::kSOMInhTauSynMs,
        l23_som_pv_patch);
    addLocalProjection(
        model,
        "L23SOM_to_L23VIP",
        l23som,
        l23vip,
        v1_genn::kL23SOMToVIPWeight,
        v1_genn::kSOMInhTauSynMs,
        l23_som_vip_patch);
    addLocalProjection(
        model,
        "L23VIP_to_L23SOM",
        l23vip,
        l23som,
        v1_genn::kL23VIPToSOMWeight,
        v1_genn::kVIPInhTauSynMs,
        l23_vip_som_patch);
}

void simulate(GeNN::ModelSpec &model, GeNN::Runtime::Runtime &runtime)
{
    const unsigned int orientation_count = getEnvUnsignedOrDefault("V1_ORIENTATION_COUNT", kDefaultOrientationCount);
    const double trial_ms = getEnvDoubleOrDefault("V1_TRIAL_MS", kDefaultTrialMs);
    const double settle_ms = getEnvDoubleOrDefault("V1_SETTLE_MS", kDefaultSettleMs);
    const unsigned int training_epochs = getEnvUnsignedOrDefault("V1_TRAINING_EPOCHS", kDefaultTrainingEpochs);
    const unsigned int recurrent_consolidation_epochs = getEnvUnsignedOrDefault(
        "V1_RECURRENT_CONSOLIDATION_EPOCHS",
        kDefaultRecurrentConsolidationEpochs);
    const unsigned int recurrent_only_consolidation_epochs = getEnvUnsignedOrDefault(
        "V1_RECURRENT_ONLY_CONSOLIDATION_EPOCHS",
        kDefaultRecurrentOnlyConsolidationEpochs);
    const double stdp_aplus = getEnvDoubleOrDefault("V1_STDP_APLUS", kDefaultStdpAplus);
    const double stdp_aminus = getEnvDoubleOrDefault("V1_STDP_AMINUS", kDefaultStdpAminus);
    const bool l23ee_plasticity_enabled = (getEnvUnsignedOrDefault("V1_L23EE_STDP_ENABLE", kDefaultL23EEPlasticityEnabled) != 0u);
    const double l23ee_stdp_aplus = getEnvDoubleOrDefault("V1_L23EE_STDP_APLUS", kDefaultL23EEStdpAplus);
    const double l23ee_stdp_aminus = getEnvDoubleOrDefault("V1_L23EE_STDP_AMINUS", kDefaultL23EEStdpAminus);
    const bool l23pv_homeostatic_enabled = (getEnvUnsignedOrDefault("V1_L23PV_HOMEO_ENABLE", kDefaultL23PVHomeostaticEnabled) != 0u);
    const bool l23som_homeostatic_enabled = (getEnvUnsignedOrDefault("V1_L23SOM_HOMEO_ENABLE", kDefaultL23SOMHomeostaticEnabled) != 0u);
    const double l23pv_homeostatic_eta = getEnvDoubleOrDefault("V1_L23PV_HOMEO_ETA", kDefaultL23PVHomeostaticEta);
    const double l23som_homeostatic_eta = getEnvDoubleOrDefault("V1_L23SOM_HOMEO_ETA", kDefaultL23SOMHomeostaticEta);
    const double l23pv_homeostatic_target_hz = getEnvDoubleOrDefault("V1_L23PV_HOMEO_TARGET_HZ", kDefaultL23PVHomeostaticTargetHz);
    const double l23som_homeostatic_target_hz = getEnvDoubleOrDefault("V1_L23SOM_HOMEO_TARGET_HZ", kDefaultL23SOMHomeostaticTargetHz);
    const double l23pv_gate = getEnvDoubleOrDefault("V1_L23PV_GATE_NA", kDefaultL23PVGate);
    const double l23som_gate = getEnvDoubleOrDefault("V1_L23SOM_GATE_NA", kDefaultL23SOMGate);
    const double l23vip_gate = getEnvDoubleOrDefault("V1_L23VIP_GATE_NA", kDefaultL23VIPGate);
    const double l23som_output_scale = getEnvDoubleOrDefault("V1_L23SOM_OUTPUT_SCALE", kDefaultL23SOMOutputScale);
    const double l23som_context_output_scale = getEnvDoubleOrDefault(
        "V1_L23SOM_CONTEXT_OUTPUT_SCALE",
        kDefaultL23SOMContextOutputScale);
    const double l23pv_context_output_scale = getEnvDoubleOrDefault(
        "V1_L23PV_CONTEXT_OUTPUT_SCALE",
        kDefaultL23PVContextOutputScale);
    const double l23ee_context_output_scale = getEnvDoubleOrDefault(
        "V1_L23EE_CONTEXT_OUTPUT_SCALE",
        kDefaultL23EEContextOutputScale);
    const L4L23OrientationConfig l4_l23_orientation_config = getL4L23OrientationConfig();
    const L23EELognormalInitConfig l23ee_lognormal_init_config = getL23EELognormalInitConfig();
    const TrainingGratingConfig training_grating_config = getTrainingGratingConfig();
    const unsigned int cell_coverage_phase_count = getEnvUnsignedOrDefault(
        "V1_CELL_COVERAGE_PHASE_COUNT",
        1u);
    const double broad_stimulus_radius_sites = getEnvDoubleOrDefault(
        "V1_BROAD_STIMULUS_RADIUS_SITES",
        kDefaultBroadStimulusRadiusSites);
    const OrientationContextAssayConfig orientation_context_assay_config =
        getOrientationContextAssayConfig(broad_stimulus_radius_sites);
    const SensoryAssayConfig sensory_assay_config = getSensoryAssayConfig();
    const std::vector<double> size_tuning_radii_sites = getEnvDoubleListOrDefault(
        "V1_SIZE_TUNING_RADII_SITES",
        kDefaultSizeTuningRadiiSites);
    const std::string output_prefix = getEnvOrDefault("V1_OUTPUT_PREFIX", kDefaultOutputPrefix);

    if(orientation_count < 2u) {
        throw std::runtime_error("V1_ORIENTATION_COUNT must be at least 2.");
    }
    if(trial_ms <= 0.0) {
        throw std::runtime_error("V1_TRIAL_MS must be positive.");
    }
    if(settle_ms < 0.0 || settle_ms >= trial_ms) {
        throw std::runtime_error("V1_SETTLE_MS must be non-negative and smaller than V1_TRIAL_MS.");
    }
    if(stdp_aplus < 0.0 || stdp_aminus < 0.0) {
        throw std::runtime_error("V1_STDP_APLUS and V1_STDP_AMINUS must be non-negative.");
    }
    if(l23ee_stdp_aplus < 0.0 || l23ee_stdp_aminus < 0.0) {
        throw std::runtime_error("V1_L23EE_STDP_APLUS and V1_L23EE_STDP_AMINUS must be non-negative.");
    }
    if(l23pv_homeostatic_eta < 0.0 || l23som_homeostatic_eta < 0.0) {
        throw std::runtime_error("Inhibitory homeostatic etas must be non-negative.");
    }
    if(l23pv_homeostatic_target_hz < 0.0 || l23som_homeostatic_target_hz < 0.0) {
        throw std::runtime_error("Inhibitory homeostatic targets must be non-negative.");
    }
    if(l23som_output_scale < 0.0) {
        throw std::runtime_error("V1_L23SOM_OUTPUT_SCALE must be non-negative.");
    }
    if(l23som_context_output_scale < 0.0) {
        throw std::runtime_error("V1_L23SOM_CONTEXT_OUTPUT_SCALE must be non-negative.");
    }
    if(l23pv_context_output_scale < 0.0) {
        throw std::runtime_error("V1_L23PV_CONTEXT_OUTPUT_SCALE must be non-negative.");
    }
    if(l23ee_context_output_scale < 0.0) {
        throw std::runtime_error("V1_L23EE_CONTEXT_OUTPUT_SCALE must be non-negative.");
    }
    const bool multiphase_cell_coverage_enabled = (cell_coverage_phase_count > 1u);
    for(double radius_sites : size_tuning_radii_sites) {
        if(radius_sites <= 0.0) {
            throw std::runtime_error("V1_SIZE_TUNING_RADII_SITES values must be positive.");
        }
    }
    const L4IntersiteConfig l4_intersite_config = getL4IntersiteConfig();
    const ValidationSiteConfig validation_site_config = getValidationSiteConfig(
        size_tuning_radii_sites,
        broad_stimulus_radius_sites);
    if(validation_site_config.site_ids.empty()
       || validation_site_config.site_ids.size() != validation_site_config.aperture_center_sites.size()) {
        throw std::runtime_error("Validation site configuration is empty or malformed.");
    }

    GeNN::NeuronGroup &l4e = requireNeuronGroup(model, "L4E");
    GeNN::NeuronGroup &l4pv = requireNeuronGroup(model, "L4PV");
    GeNN::NeuronGroup &l4som = requireNeuronGroup(model, "L4SOM");
    GeNN::NeuronGroup &l23e = requireNeuronGroup(model, "L23E");
    GeNN::NeuronGroup &l23pv = requireNeuronGroup(model, "L23PV");
    GeNN::NeuronGroup &l23som = requireNeuronGroup(model, "L23SOM");
    GeNN::NeuronGroup &l23vip = requireNeuronGroup(model, "L23VIP");
    GeNN::SynapseGroup &l4e_to_l23e = requireSynapseGroup(model, "L4E_to_L23E");
    GeNN::SynapseGroup &l23e_to_l23e = requireSynapseGroup(model, "L23E_to_L23E");
    GeNN::SynapseGroup &l23pv_to_l23e = requireSynapseGroup(model, "L23PV_to_L23E");
    GeNN::SynapseGroup &l23pv_to_l23pv = requireSynapseGroup(model, "L23PV_to_L23PV");
    GeNN::SynapseGroup &l23som_to_l23e = requireSynapseGroup(model, "L23SOM_to_L23E");
    GeNN::SynapseGroup &l23som_to_l23pv = requireSynapseGroup(model, "L23SOM_to_L23PV");
    GeNN::SynapseGroup &l23som_to_l23vip = requireSynapseGroup(model, "L23SOM_to_L23VIP");

    const std::vector<double> orientations_rad = makeSweepOrientations(orientation_count);
    const unsigned int trial_steps = durationToSteps(trial_ms);
    const unsigned int effective_settle_steps = (settle_ms == 0.0) ? 0u : durationToSteps(settle_ms);
    if(effective_settle_steps >= trial_steps) {
        throw std::runtime_error("V1_SETTLE_MS must leave a positive measurement window.");
    }
    if(training_grating_config.phase_drift_enabled
       && (training_grating_config.phase_count > trial_steps || (trial_steps % training_grating_config.phase_count) != 0u)) {
        throw std::runtime_error(
            "V1_TRAINING_DRIFT_PHASE_COUNT must evenly divide the trial step count for within-trial phase stepping.");
    }
    const double training_grating_phase_slot_ms =
        training_grating_config.phase_drift_enabled
            ? (trial_ms / static_cast<double>(training_grating_config.phase_count))
            : trial_ms;

    const std::size_t sweep_count =
        static_cast<std::size_t>(training_epochs)
        + static_cast<std::size_t>(recurrent_consolidation_epochs)
        + static_cast<std::size_t>(recurrent_only_consolidation_epochs)
        + 2u
        + (2u * validation_site_config.site_ids.size())
        + 1u
        + (size_tuning_radii_sites.size() * validation_site_config.site_ids.size());
    const std::size_t multiphase_trial_count =
        multiphase_cell_coverage_enabled
            ? (static_cast<std::size_t>(orientation_count) * static_cast<std::size_t>(cell_coverage_phase_count))
            : 0u;
    const std::size_t orientation_context_trial_count =
        orientation_context_assay_config.enabled
            ? (static_cast<std::size_t>(validation_site_config.site_ids.size())
               * static_cast<std::size_t>(kOrientationContextConditionCount))
            : 0u;
    const std::size_t sensory_blank_trial_count =
        sensory_assay_config.enabled
            ? static_cast<std::size_t>(sensory_assay_config.blank_repeat_count)
            : 0u;
    const std::size_t sensory_contrast_trial_count =
        sensory_assay_config.enabled
            ? (static_cast<std::size_t>(validation_site_config.site_ids.size())
               * sensory_assay_config.contrasts.size())
            : 0u;
    const std::size_t total_trial_count =
        (static_cast<std::size_t>(orientation_count) * sweep_count)
        + multiphase_trial_count
        + orientation_context_trial_count
        + sensory_blank_trial_count
        + sensory_contrast_trial_count;
    const std::size_t total_recording_steps = total_trial_count * static_cast<std::size_t>(trial_steps);

    runtime.allocate(total_recording_steps);
    runtime.initialize();
    runtime.initializeSparse();

    const auto scaleSomToL23EOutput = [&](double scale) {
        scaleSynapseWeights(runtime, l23som_to_l23e, scale);
    };
    const auto scaleSomOutputs = [&](double scale) {
        scaleSomToL23EOutput(scale);
        scaleSynapseWeights(runtime, l23som_to_l23pv, scale);
        scaleSynapseWeights(runtime, l23som_to_l23vip, scale);
    };
    const auto scalePvOutputs = [&](double scale) {
        scaleSynapseWeights(runtime, l23pv_to_l23e, scale);
        scaleSynapseWeights(runtime, l23pv_to_l23pv, scale);
    };

    setConstantExternalCurrent(runtime, l4pv, 0.0);
    setConstantExternalCurrent(runtime, l4som, 0.0);
    setConstantExternalCurrent(runtime, l23pv, l23pv_gate);
    setConstantExternalCurrent(runtime, l23som, l23som_gate);
    setConstantExternalCurrent(runtime, l23vip, l23vip_gate);
    if(l23som_output_scale != 1.0) {
        scaleSomOutputs(l23som_output_scale);
    }

    GeNN::Runtime::ArrayBase &l4e_i_ext = requireArray(runtime, l4e, "Iext");
    float *l4e_i_ext_host = l4e_i_ext.getHostPointer<float>();

    const std::vector<std::pair<unsigned int, unsigned int>> ff_edges = buildL4EToL23EConnectivity();
    const std::vector<std::pair<unsigned int, unsigned int>> l23ee_edges = buildSparseDistanceConnectivity(
        v1_genn::kL23EPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kL23LocalRadius,
        true,
        kL23ERecurrentPeakProbability,
        kL23ERecurrentDistanceSigmaSq);
    const std::vector<std::pair<unsigned int, unsigned int>> l23pv_edges = buildLocalPatchConnectivity(
        v1_genn::kL23PVPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kL23LocalRadius,
        false);
    const std::vector<std::pair<unsigned int, unsigned int>> l23som_edges = buildLocalPatchConnectivity(
        v1_genn::kL23SOMPerSite,
        v1_genn::kL23EPerSite,
        v1_genn::kL23SOMOutputRadius,
        false);
    applyL23EELognormalInitialWeights(
        runtime,
        l23e_to_l23e,
        l23ee_edges,
        l23ee_lognormal_init_config);
    const std::vector<float> weights_before = copyWeights(runtime, l4e_to_l23e);
    const std::vector<float> l23ee_weights_before = copyWeights(runtime, l23e_to_l23e);
    const std::vector<float> l23pv_weights_before = copyWeights(runtime, l23pv_to_l23e);
    const std::vector<float> l23som_weights_before = copyWeights(runtime, l23som_to_l23e);

    std::vector<TrialWindow> baseline_trials;
    std::vector<TrialWindow> post_trials;
    std::vector<TrialWindow> multiphase_cell_coverage_trials;
    std::vector<TrialWindow> recurrence_context_trials;
    std::vector<TrialWindow> blank_baseline_trials;
    std::vector<TrialWindow> contrast_sweep_trials;
    std::vector<ContrastTrialRecord> contrast_sweep_records;
    std::vector<OrientationContextTrialSet> orientation_context_trials;
    baseline_trials.reserve(orientation_count);
    post_trials.reserve(orientation_count);
    multiphase_cell_coverage_trials.reserve(multiphase_trial_count);
    recurrence_context_trials.reserve(orientation_count);
    blank_baseline_trials.reserve(sensory_blank_trial_count);
    contrast_sweep_trials.reserve(sensory_contrast_trial_count);
    contrast_sweep_records.reserve(sensory_contrast_trial_count);
    orientation_context_trials.reserve(validation_site_config.site_ids.size());
    std::vector<ValidationTrialSet> validation_trials;
    validation_trials.reserve(validation_site_config.site_ids.size());
    for(std::size_t i = 0; i < validation_site_config.site_ids.size(); i++) {
        ValidationTrialSet trial_set;
        trial_set.site_id = validation_site_config.site_ids[i];
        trial_set.aperture_center_site = validation_site_config.aperture_center_sites[i];
        trial_set.center_trials.reserve(orientation_count);
        trial_set.broad_trials.reserve(orientation_count);
        trial_set.size_trials.reserve(
            static_cast<std::size_t>(orientation_count) * size_tuning_radii_sites.size());
        validation_trials.push_back(trial_set);
    }

    std::vector<float> l4e_drive;
    auto runSweep = [&](const std::string &label,
                        std::vector<TrialWindow> *measurement_trials,
                        bool feedforward_learning,
                        bool recurrent_learning,
                        bool inhibitory_learning,
                        unsigned int phase_cycle_offset,
                        double aperture_radius_sites,
                        unsigned int aperture_center_site = std::numeric_limits<unsigned int>::max()) {
        (void)label;
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", feedforward_learning ? stdp_aplus : 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", feedforward_learning ? stdp_aminus : 0.0);
        runtime.setDynamicParamValue(
            l23e_to_l23e,
            "Aplus",
            (recurrent_learning && l23ee_plasticity_enabled) ? l23ee_stdp_aplus : 0.0);
        runtime.setDynamicParamValue(
            l23e_to_l23e,
            "Aminus",
            (recurrent_learning && l23ee_plasticity_enabled) ? l23ee_stdp_aminus : 0.0);
        runtime.setDynamicParamValue(
            l23pv_to_l23e,
            "TargetHz",
            l23pv_homeostatic_target_hz);
        runtime.setDynamicParamValue(
            l23som_to_l23e,
            "TargetHz",
            l23som_homeostatic_target_hz);
        runtime.setDynamicParamValue(
            l23pv_to_l23e,
            "Eta",
            (inhibitory_learning && l23pv_homeostatic_enabled) ? l23pv_homeostatic_eta : 0.0);
        runtime.setDynamicParamValue(
            l23som_to_l23e,
            "Eta",
            (inhibitory_learning && l23som_homeostatic_enabled) ? l23som_homeostatic_eta : 0.0);

        const auto pushDrivePhase = [&](double orientation_rad, double phase_rad, double aperture_radius, unsigned int aperture_center) {
            fillL4EDrive(l4e_drive, orientation_rad, phase_rad, aperture_radius, aperture_center);
            std::copy(l4e_drive.begin(), l4e_drive.end(), l4e_i_ext_host);
            l4e_i_ext.pushToDevice();
        };
        const auto driftPhaseForSubslot = [&](unsigned int orientation_index, unsigned int subslot_index) {
            const unsigned int start_slot = (phase_cycle_offset + orientation_index) % training_grating_config.phase_count;
            const bool reverse_order = training_grating_config.counterbalance_direction && ((phase_cycle_offset % 2u) != 0u);
            const unsigned int phase_slot = reverse_order
                ? ((start_slot + training_grating_config.phase_count - (subslot_index % training_grating_config.phase_count))
                   % training_grating_config.phase_count)
                : ((start_slot + subslot_index) % training_grating_config.phase_count);
            return (2.0 * v1_genn::kPi * static_cast<double>(phase_slot))
                / static_cast<double>(training_grating_config.phase_count);
        };

        for(unsigned int orientation_index = 0; orientation_index < orientation_count; orientation_index++) {
            const double orientation_rad = orientations_rad[orientation_index];
            const bool plastic_exposure = feedforward_learning || recurrent_learning || inhibitory_learning;
            const bool phase_drift_trial = training_grating_config.phase_drift_enabled && plastic_exposure;
            const unsigned int phase_slot = plastic_exposure ? ((phase_cycle_offset + orientation_index) % 4u) : 0u;
            const double phase_rad = phase_drift_trial
                ? driftPhaseForSubslot(orientation_index, 0u)
                : (0.5 * v1_genn::kPi * static_cast<double>(phase_slot));
            if(!phase_drift_trial) {
                pushDrivePhase(orientation_rad, phase_rad, aperture_radius_sites, aperture_center_site);
            }

            const double trial_start_ms = runtime.getTime();
            if(measurement_trials != nullptr) {
                measurement_trials->push_back({
                    orientation_rad,
                    phase_rad,
                    trial_start_ms,
                    trial_start_ms + (static_cast<double>(effective_settle_steps) * v1_genn::kDtMs),
                    trial_start_ms + (static_cast<double>(trial_steps) * v1_genn::kDtMs),
                });
            }

            if(phase_drift_trial) {
                const unsigned int phase_slot_steps = trial_steps / training_grating_config.phase_count;
                unsigned int active_subslot = std::numeric_limits<unsigned int>::max();
                for(unsigned int step = 0; step < trial_steps; step++) {
                    const unsigned int subslot = std::min(
                        step / phase_slot_steps,
                        training_grating_config.phase_count - 1u);
                    if(subslot != active_subslot) {
                        active_subslot = subslot;
                        pushDrivePhase(
                            orientation_rad,
                            driftPhaseForSubslot(orientation_index, active_subslot),
                            aperture_radius_sites,
                            aperture_center_site);
                    }
                    runtime.stepTime();
                }
            }
            else {
                for(unsigned int step = 0; step < trial_steps; step++) {
                    runtime.stepTime();
                }
            }
        }
    };

    auto runMultiPhaseCellCoverageSweep = [&]() {
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
        runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

        for(unsigned int orientation_index = 0; orientation_index < orientation_count; orientation_index++) {
            const double orientation_rad = orientations_rad[orientation_index];
            for(unsigned int phase_index = 0; phase_index < cell_coverage_phase_count; phase_index++) {
                const double phase_rad =
                    (2.0 * v1_genn::kPi * static_cast<double>(phase_index))
                    / static_cast<double>(cell_coverage_phase_count);

                fillL4EDrive(l4e_drive, orientation_rad, phase_rad, -1.0);
                std::copy(l4e_drive.begin(), l4e_drive.end(), l4e_i_ext_host);
                l4e_i_ext.pushToDevice();

                const double trial_start_ms = runtime.getTime();
                multiphase_cell_coverage_trials.push_back({
                    orientation_rad,
                    phase_rad,
                    trial_start_ms,
                    trial_start_ms + (static_cast<double>(effective_settle_steps) * v1_genn::kDtMs),
                    trial_start_ms + (static_cast<double>(trial_steps) * v1_genn::kDtMs),
                });

                for(unsigned int step = 0; step < trial_steps; step++) {
                    runtime.stepTime();
                }
            }
        }
    };

    auto runHeldOutStimulusTrial = [&](double orientation_rad,
                                       double phase_rad,
                                       double aperture_radius_sites,
                                       unsigned int aperture_center_site,
                                       double aperture_inner_radius_sites) {
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
        runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

        fillL4EDrive(
            l4e_drive,
            orientation_rad,
            phase_rad,
            aperture_radius_sites,
            aperture_center_site,
            aperture_inner_radius_sites);
        std::copy(l4e_drive.begin(), l4e_drive.end(), l4e_i_ext_host);
        l4e_i_ext.pushToDevice();

        const double trial_start_ms = runtime.getTime();
        TrialWindow trial{
            orientation_rad,
            phase_rad,
            trial_start_ms,
            trial_start_ms + (static_cast<double>(effective_settle_steps) * v1_genn::kDtMs),
            trial_start_ms + (static_cast<double>(trial_steps) * v1_genn::kDtMs),
        };
        for(unsigned int step = 0; step < trial_steps; step++) {
            runtime.stepTime();
        }
        return trial;
    };

    auto runHeldOutCenterSurroundStimulusTrial = [&](double center_orientation_rad,
                                                     double surround_orientation_rad,
                                                     double phase_rad,
                                                     double center_radius_sites,
                                                     double surround_outer_radius_sites,
                                                     unsigned int aperture_center_site) {
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
        runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

        fillL4ECenterSurroundDrive(
            l4e_drive,
            center_orientation_rad,
            surround_orientation_rad,
            phase_rad,
            center_radius_sites,
            surround_outer_radius_sites,
            aperture_center_site);
        std::copy(l4e_drive.begin(), l4e_drive.end(), l4e_i_ext_host);
        l4e_i_ext.pushToDevice();

        const double trial_start_ms = runtime.getTime();
        TrialWindow trial{
            surround_orientation_rad,
            phase_rad,
            trial_start_ms,
            trial_start_ms + (static_cast<double>(effective_settle_steps) * v1_genn::kDtMs),
            trial_start_ms + (static_cast<double>(trial_steps) * v1_genn::kDtMs),
        };
        for(unsigned int step = 0; step < trial_steps; step++) {
            runtime.stepTime();
        }
        return trial;
    };

    auto runBlankBaselineTrial = [&]() {
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
        runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

        std::fill(l4e_i_ext_host, l4e_i_ext_host + l4e_i_ext.getCount(), 0.0f);
        l4e_i_ext.pushToDevice();

        const double trial_start_ms = runtime.getTime();
        TrialWindow trial{
            0.0,
            0.0,
            trial_start_ms,
            trial_start_ms + (static_cast<double>(effective_settle_steps) * v1_genn::kDtMs),
            trial_start_ms + (static_cast<double>(trial_steps) * v1_genn::kDtMs),
        };
        for(unsigned int step = 0; step < trial_steps; step++) {
            runtime.stepTime();
        }
        return trial;
    };

    auto runContrastStimulusTrial = [&](double orientation_rad,
                                        double contrast,
                                        double aperture_radius_sites,
                                        unsigned int aperture_center_site) {
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
        runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

        fillL4EDrive(
            l4e_drive,
            orientation_rad,
            0.0,
            aperture_radius_sites,
            aperture_center_site,
            -1.0,
            contrast);
        std::copy(l4e_drive.begin(), l4e_drive.end(), l4e_i_ext_host);
        l4e_i_ext.pushToDevice();

        const double trial_start_ms = runtime.getTime();
        TrialWindow trial{
            orientation_rad,
            0.0,
            trial_start_ms,
            trial_start_ms + (static_cast<double>(effective_settle_steps) * v1_genn::kDtMs),
            trial_start_ms + (static_cast<double>(trial_steps) * v1_genn::kDtMs),
        };
        for(unsigned int step = 0; step < trial_steps; step++) {
            runtime.stepTime();
        }
        return trial;
    };

    runSweep("baseline", &baseline_trials, false, false, false, 0u, -1.0);
    for(unsigned int epoch = 0; epoch < training_epochs; epoch++) {
        runSweep("training", nullptr, true, true, true, epoch, -1.0);
    }
    for(unsigned int epoch = 0; epoch < recurrent_consolidation_epochs; epoch++) {
        // Extra visual exposure lets sparse recurrent and inhibitory plasticity
        // integrate coactivity without further changing feedforward selectivity.
        runSweep("recurrent_consolidation", nullptr, false, true, true, training_epochs + epoch, -1.0);
    }
    for(unsigned int epoch = 0; epoch < recurrent_only_consolidation_epochs; epoch++) {
        runSweep(
            "recurrent_only_consolidation",
            nullptr,
            false,
            true,
            false,
            training_epochs + recurrent_consolidation_epochs + epoch,
            -1.0);
    }
    runSweep("post", &post_trials, false, false, false, 0u, -1.0);
    runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
    runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
    runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
    runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
    runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
    runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

    const std::vector<float> weights_after = copyWeights(runtime, l4e_to_l23e);
    const std::vector<float> l23ee_weights_after = copyWeights(runtime, l23e_to_l23e);
    const std::vector<float> l23pv_weights_after = copyWeights(runtime, l23pv_to_l23e);
    const std::vector<float> l23som_weights_after = copyWeights(runtime, l23som_to_l23e);

    if(multiphase_cell_coverage_enabled) {
        runMultiPhaseCellCoverageSweep();
    }

    if(l23som_context_output_scale != 1.0) {
        scaleSomToL23EOutput(l23som_context_output_scale);
    }
    if(l23pv_context_output_scale != 1.0) {
        scalePvOutputs(l23pv_context_output_scale);
    }

    if(sensory_assay_config.enabled) {
        for(unsigned int repeat = 0; repeat < sensory_assay_config.blank_repeat_count; repeat++) {
            (void)repeat;
            blank_baseline_trials.push_back(runBlankBaselineTrial());
        }
        for(std::size_t i = 0; i < validation_site_config.site_ids.size(); i++) {
            const unsigned int site_id = validation_site_config.site_ids[i];
            const unsigned int aperture_center_site = validation_site_config.aperture_center_sites[i];
            const double preferred_orientation_rad = v1_genn::sitePreferredOrientationFromIndex(site_id);
            for(double contrast : sensory_assay_config.contrasts) {
                ContrastTrialRecord record;
                record.validation_site_id = site_id;
                record.site_id = site_id;
                record.aperture_center_site = aperture_center_site;
                record.contrast = contrast;
                record.orientation_rad = preferred_orientation_rad;
                record.aperture_radius_sites = sensory_assay_config.contrast_radius_sites;
                record.trial = runContrastStimulusTrial(
                    preferred_orientation_rad,
                    contrast,
                    sensory_assay_config.contrast_radius_sites,
                    aperture_center_site);
                contrast_sweep_trials.push_back(record.trial);
                contrast_sweep_records.push_back(record);
            }
        }
    }

    for(ValidationTrialSet &trial_set : validation_trials) {
        runSweep(
            "center_validation",
            &trial_set.center_trials,
            false,
            false,
            false,
            0u,
            kDefaultCenterStimulusRadiusSites,
            trial_set.aperture_center_site);
        runSweep(
            "broad_validation",
            &trial_set.broad_trials,
            false,
            false,
            false,
            0u,
            broad_stimulus_radius_sites,
            trial_set.aperture_center_site);
        for(double radius_sites : size_tuning_radii_sites) {
            runSweep(
                "size_tuning",
                &trial_set.size_trials,
                false,
                false,
                false,
                0u,
                radius_sites,
                trial_set.aperture_center_site);
        }
    }
    if(orientation_context_assay_config.enabled) {
        for(std::size_t i = 0; i < validation_site_config.site_ids.size(); i++) {
            const unsigned int site_id = validation_site_config.site_ids[i];
            const unsigned int aperture_center_site = validation_site_config.aperture_center_sites[i];
            const double preferred_orientation_rad =
                v1_genn::sitePreferredOrientationFromIndex(site_id);
            const double orthogonal_orientation_rad =
                std::fmod(preferred_orientation_rad + (0.5 * v1_genn::kPi), v1_genn::kPi);

            OrientationContextTrialSet trial_set;
            trial_set.validation_site_id = site_id;
            trial_set.site_id = site_id;
            trial_set.aperture_center_site = aperture_center_site;
            trial_set.preferred_orientation_rad = preferred_orientation_rad;
            trial_set.orthogonal_orientation_rad = orthogonal_orientation_rad;
            trial_set.trials[kOrientationContextCenterOnly] = runHeldOutStimulusTrial(
                preferred_orientation_rad,
                0.0,
                orientation_context_assay_config.center_radius_sites,
                aperture_center_site,
                -1.0);
            trial_set.trials[kOrientationContextSameSurround] = runHeldOutCenterSurroundStimulusTrial(
                preferred_orientation_rad,
                preferred_orientation_rad,
                0.0,
                orientation_context_assay_config.center_radius_sites,
                orientation_context_assay_config.broad_radius_sites,
                aperture_center_site);
            trial_set.trials[kOrientationContextOrthSurround] = runHeldOutCenterSurroundStimulusTrial(
                preferred_orientation_rad,
                orthogonal_orientation_rad,
                0.0,
                orientation_context_assay_config.center_radius_sites,
                orientation_context_assay_config.broad_radius_sites,
                aperture_center_site);
            trial_set.trials[kOrientationContextSurroundSameOnly] = runHeldOutStimulusTrial(
                preferred_orientation_rad,
                0.0,
                orientation_context_assay_config.broad_radius_sites,
                aperture_center_site,
                orientation_context_assay_config.surround_inner_radius_sites);
            trial_set.trials[kOrientationContextSurroundOrthOnly] = runHeldOutStimulusTrial(
                orthogonal_orientation_rad,
                0.0,
                orientation_context_assay_config.broad_radius_sites,
                aperture_center_site,
                orientation_context_assay_config.surround_inner_radius_sites);
            orientation_context_trials.push_back(trial_set);
        }
    }
    if(l23ee_context_output_scale != 1.0) {
        scaleSynapseWeights(runtime, l23e_to_l23e, l23ee_context_output_scale);
    }
    runSweep("recurrence_context", &recurrence_context_trials, false, false, false, 0u, -1.0);

    runtime.pullRecordingBuffersFromDevice();
    const auto l4e_recordings = runtime.getRecordedSpikes(l4e);
    const auto l4pv_recordings = runtime.getRecordedSpikes(l4pv);
    const auto l4som_recordings = runtime.getRecordedSpikes(l4som);
    const auto l23e_recordings = runtime.getRecordedSpikes(l23e);
    const auto l23pv_recordings = runtime.getRecordedSpikes(l23pv);
    const auto l23som_recordings = runtime.getRecordedSpikes(l23som);
    const auto l23vip_recordings = runtime.getRecordedSpikes(l23vip);
    if(l4e_recordings.empty() || l4pv_recordings.empty() || l4som_recordings.empty()
       || l23e_recordings.empty() || l23pv_recordings.empty() || l23som_recordings.empty()
       || l23vip_recordings.empty()) {
        throw std::runtime_error("Expected recorded spikes for all recorded V1 populations.");
    }

    const std::vector<double> baseline_l4_counts =
        countSiteSpikesForTrials(l4e_recordings.at(0), baseline_trials, v1_genn::kL4EPerSite);
    const std::vector<double> baseline_l23_counts =
        countSiteSpikesForTrials(l23e_recordings.at(0), baseline_trials, v1_genn::kL23EPerSite);
    const std::vector<double> baseline_l23pv_site_counts =
        countSiteSpikesForTrials(l23pv_recordings.at(0), baseline_trials, v1_genn::kL23PVPerSite);
    const std::vector<double> baseline_l23som_site_counts =
        countSiteSpikesForTrials(l23som_recordings.at(0), baseline_trials, v1_genn::kL23SOMPerSite);
    const std::vector<double> baseline_l23vip_site_counts =
        countSiteSpikesForTrials(l23vip_recordings.at(0), baseline_trials, v1_genn::kL23VIPPerSite);
    const std::vector<double> post_l4_counts =
        countSiteSpikesForTrials(l4e_recordings.at(0), post_trials, v1_genn::kL4EPerSite);
    const std::vector<double> post_l23_counts =
        countSiteSpikesForTrials(l23e_recordings.at(0), post_trials, v1_genn::kL23EPerSite);
    const std::vector<double> post_l23_cell_counts =
        countNeuronSpikesForTrials(l23e_recordings.at(0), post_trials, v1_genn::kNumL23E);
    const std::vector<double> multiphase_l23_cell_counts =
        multiphase_cell_coverage_enabled
            ? countNeuronSpikesForTrials(l23e_recordings.at(0), multiphase_cell_coverage_trials, v1_genn::kNumL23E)
            : std::vector<double>();
    const std::vector<double> post_l23pv_site_counts =
        countSiteSpikesForTrials(l23pv_recordings.at(0), post_trials, v1_genn::kL23PVPerSite);
    const std::vector<double> post_l23som_site_counts =
        countSiteSpikesForTrials(l23som_recordings.at(0), post_trials, v1_genn::kL23SOMPerSite);
    const std::vector<double> post_l23vip_site_counts =
        countSiteSpikesForTrials(l23vip_recordings.at(0), post_trials, v1_genn::kL23VIPPerSite);
    const std::vector<double> recurrence_l23_cell_counts =
        countNeuronSpikesForTrials(l23e_recordings.at(0), recurrence_context_trials, v1_genn::kNumL23E);
    const std::vector<double> blank_l4e_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l4e_recordings.at(0), blank_baseline_trials, v1_genn::kL4EPerSite)
            : std::vector<double>();
    const std::vector<double> blank_l23e_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l23e_recordings.at(0), blank_baseline_trials, v1_genn::kL23EPerSite)
            : std::vector<double>();
    const std::vector<double> blank_l23pv_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l23pv_recordings.at(0), blank_baseline_trials, v1_genn::kL23PVPerSite)
            : std::vector<double>();
    const std::vector<double> blank_l23som_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l23som_recordings.at(0), blank_baseline_trials, v1_genn::kL23SOMPerSite)
            : std::vector<double>();
    const std::vector<double> contrast_l4e_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l4e_recordings.at(0), contrast_sweep_trials, v1_genn::kL4EPerSite)
            : std::vector<double>();
    const std::vector<double> contrast_l23e_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l23e_recordings.at(0), contrast_sweep_trials, v1_genn::kL23EPerSite)
            : std::vector<double>();
    const std::vector<double> contrast_l23pv_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l23pv_recordings.at(0), contrast_sweep_trials, v1_genn::kL23PVPerSite)
            : std::vector<double>();
    const std::vector<double> contrast_l23som_site_counts =
        sensory_assay_config.enabled
            ? countSiteSpikesForTrials(l23som_recordings.at(0), contrast_sweep_trials, v1_genn::kL23SOMPerSite)
            : std::vector<double>();

    const std::vector<std::pair<std::string, std::vector<double>>> baseline_subtype_rates{
        {"l4pv", countPopulationRatesForTrials(l4pv_recordings.at(0), baseline_trials, v1_genn::kNumL4PV)},
        {"l4som", countPopulationRatesForTrials(l4som_recordings.at(0), baseline_trials, v1_genn::kNumL4SOM)},
        {"l23pv", countPopulationRatesForTrials(l23pv_recordings.at(0), baseline_trials, v1_genn::kNumL23PV)},
        {"l23som", countPopulationRatesForTrials(l23som_recordings.at(0), baseline_trials, v1_genn::kNumL23SOM)},
        {"l23vip", countPopulationRatesForTrials(l23vip_recordings.at(0), baseline_trials, v1_genn::kNumL23VIP)},
    };
    const std::vector<std::pair<std::string, std::vector<double>>> post_subtype_rates{
        {"l4pv", countPopulationRatesForTrials(l4pv_recordings.at(0), post_trials, v1_genn::kNumL4PV)},
        {"l4som", countPopulationRatesForTrials(l4som_recordings.at(0), post_trials, v1_genn::kNumL4SOM)},
        {"l23pv", countPopulationRatesForTrials(l23pv_recordings.at(0), post_trials, v1_genn::kNumL23PV)},
        {"l23som", countPopulationRatesForTrials(l23som_recordings.at(0), post_trials, v1_genn::kNumL23SOM)},
        {"l23vip", countPopulationRatesForTrials(l23vip_recordings.at(0), post_trials, v1_genn::kNumL23VIP)},
    };

    std::vector<PopulationRateSummary> subtype_summaries;
    subtype_summaries.reserve(baseline_subtype_rates.size());
    for(std::size_t i = 0; i < baseline_subtype_rates.size(); i++) {
        subtype_summaries.push_back({
            baseline_subtype_rates[i].first,
            meanRate(baseline_subtype_rates[i].second),
            meanRate(post_subtype_rates[i].second),
        });
    }

    const SweepResult baseline = buildSweepResult(
        "baseline",
        orientations_rad,
        baseline_trials,
        baseline_l4_counts,
        baseline_l23_counts);
    const SweepResult post = buildSweepResult(
        "post",
        orientations_rad,
        post_trials,
        post_l4_counts,
        post_l23_counts);
    const std::vector<PopulationSiteMetrics> baseline_l23pv_sites =
        computeSiteMetrics(baseline_trials, baseline_l23pv_site_counts, v1_genn::kL23PVPerSite);
    const std::vector<PopulationSiteMetrics> baseline_l23som_sites =
        computeSiteMetrics(baseline_trials, baseline_l23som_site_counts, v1_genn::kL23SOMPerSite);
    const std::vector<PopulationSiteMetrics> baseline_l23vip_sites =
        computeSiteMetrics(baseline_trials, baseline_l23vip_site_counts, v1_genn::kL23VIPPerSite);
    const std::vector<PopulationSiteMetrics> post_l23pv_sites =
        computeSiteMetrics(post_trials, post_l23pv_site_counts, v1_genn::kL23PVPerSite);
    const std::vector<PopulationSiteMetrics> post_l23som_sites =
        computeSiteMetrics(post_trials, post_l23som_site_counts, v1_genn::kL23SOMPerSite);
    const std::vector<PopulationSiteMetrics> post_l23vip_sites =
        computeSiteMetrics(post_trials, post_l23vip_site_counts, v1_genn::kL23VIPPerSite);
    const std::vector<CellTuningMetrics> post_l23e_cell_tuning =
        computeCellTuningMetrics(post_trials, post_l23_cell_counts, v1_genn::kNumL23E, v1_genn::kL23EPerSite);
    const std::vector<MultiPhaseCellTuningMetrics> multiphase_l23e_cell_tuning =
        multiphase_cell_coverage_enabled
            ? computeMultiPhaseCellTuningMetrics(
                multiphase_cell_coverage_trials,
                multiphase_l23_cell_counts,
                orientations_rad,
                cell_coverage_phase_count,
                v1_genn::kNumL23E,
                v1_genn::kL23EPerSite)
            : std::vector<MultiPhaseCellTuningMetrics>();
    const std::vector<CellTuningMetrics> recurrence_l23e_cell_tuning =
        computeCellTuningMetrics(
            recurrence_context_trials,
            recurrence_l23_cell_counts,
            v1_genn::kNumL23E,
            v1_genn::kL23EPerSite);

    std::vector<OrientationContextSiteMetrics> orientation_context_metrics;
    orientation_context_metrics.reserve(orientation_context_trials.size());
    for(const OrientationContextTrialSet &trial_set : orientation_context_trials) {
        const std::vector<TrialWindow> trials(trial_set.trials.begin(), trial_set.trials.end());
        const std::vector<double> l4e_counts =
            countSiteSpikesForTrials(l4e_recordings.at(0), trials, v1_genn::kL4EPerSite);
        const std::vector<double> l23e_counts =
            countSiteSpikesForTrials(l23e_recordings.at(0), trials, v1_genn::kL23EPerSite);
        const std::vector<double> l23pv_counts =
            countSiteSpikesForTrials(l23pv_recordings.at(0), trials, v1_genn::kL23PVPerSite);
        const std::vector<double> l23som_counts =
            countSiteSpikesForTrials(l23som_recordings.at(0), trials, v1_genn::kL23SOMPerSite);
        const std::vector<PopulationSiteMetrics> l4e_sites =
            computeSiteMetrics(trials, l4e_counts, v1_genn::kL4EPerSite);
        const std::vector<PopulationSiteMetrics> l23e_sites =
            computeSiteMetrics(trials, l23e_counts, v1_genn::kL23EPerSite);
        const std::vector<PopulationSiteMetrics> l23pv_sites =
            computeSiteMetrics(trials, l23pv_counts, v1_genn::kL23PVPerSite);
        const std::vector<PopulationSiteMetrics> l23som_sites =
            computeSiteMetrics(trials, l23som_counts, v1_genn::kL23SOMPerSite);

        OrientationContextSiteMetrics metrics;
        metrics.validation_site_id = trial_set.validation_site_id;
        metrics.site_id = trial_set.site_id;
        metrics.aperture_center_site = trial_set.aperture_center_site;
        metrics.preferred_orientation_rad = trial_set.preferred_orientation_rad;
        metrics.orthogonal_orientation_rad = trial_set.orthogonal_orientation_rad;
        metrics.aperture_radius_sites[kOrientationContextCenterOnly] =
            orientation_context_assay_config.center_radius_sites;
        metrics.aperture_radius_sites[kOrientationContextSameSurround] =
            orientation_context_assay_config.broad_radius_sites;
        metrics.aperture_radius_sites[kOrientationContextOrthSurround] =
            orientation_context_assay_config.broad_radius_sites;
        metrics.aperture_radius_sites[kOrientationContextSurroundSameOnly] =
            orientation_context_assay_config.broad_radius_sites;
        metrics.aperture_radius_sites[kOrientationContextSurroundOrthOnly] =
            orientation_context_assay_config.broad_radius_sites;
        metrics.inner_radius_sites.fill(-1.0);
        metrics.inner_radius_sites[kOrientationContextSurroundSameOnly] =
            orientation_context_assay_config.surround_inner_radius_sites;
        metrics.inner_radius_sites[kOrientationContextSurroundOrthOnly] =
            orientation_context_assay_config.surround_inner_radius_sites;
        for(unsigned int condition_index = 0; condition_index < kOrientationContextConditionCount; condition_index++) {
            metrics.l4e_rates_hz[condition_index] =
                l4e_sites.at(trial_set.site_id).rates_hz.at(condition_index);
            metrics.l23e_rates_hz[condition_index] =
                l23e_sites.at(trial_set.site_id).rates_hz.at(condition_index);
            metrics.l23pv_rates_hz[condition_index] =
                l23pv_sites.at(trial_set.site_id).rates_hz.at(condition_index);
            metrics.l23som_rates_hz[condition_index] =
                l23som_sites.at(trial_set.site_id).rates_hz.at(condition_index);
        }
        orientation_context_metrics.push_back(metrics);
    }

    std::vector<RetinotopicContextMetrics> context_validation_metrics;
    std::vector<RetinotopicSizeMetrics> size_validation_metrics;
    context_validation_metrics.reserve(validation_trials.size());
    size_validation_metrics.reserve(validation_trials.size());
    for(const ValidationTrialSet &trial_set : validation_trials) {
        const std::vector<double> center_l23e_site_counts =
            countSiteSpikesForTrials(l23e_recordings.at(0), trial_set.center_trials, v1_genn::kL23EPerSite);
        const std::vector<double> center_l23pv_site_counts =
            countSiteSpikesForTrials(l23pv_recordings.at(0), trial_set.center_trials, v1_genn::kL23PVPerSite);
        const std::vector<double> center_l23som_site_counts =
            countSiteSpikesForTrials(l23som_recordings.at(0), trial_set.center_trials, v1_genn::kL23SOMPerSite);
        const std::vector<double> broad_l23e_site_counts =
            countSiteSpikesForTrials(l23e_recordings.at(0), trial_set.broad_trials, v1_genn::kL23EPerSite);
        const std::vector<double> broad_l23pv_site_counts =
            countSiteSpikesForTrials(l23pv_recordings.at(0), trial_set.broad_trials, v1_genn::kL23PVPerSite);
        const std::vector<double> broad_l23som_site_counts =
            countSiteSpikesForTrials(l23som_recordings.at(0), trial_set.broad_trials, v1_genn::kL23SOMPerSite);
        const std::vector<PopulationSiteMetrics> center_l23e_sites =
            computeSiteMetrics(trial_set.center_trials, center_l23e_site_counts, v1_genn::kL23EPerSite);
        const std::vector<PopulationSiteMetrics> center_l23pv_sites =
            computeSiteMetrics(trial_set.center_trials, center_l23pv_site_counts, v1_genn::kL23PVPerSite);
        const std::vector<PopulationSiteMetrics> center_l23som_sites =
            computeSiteMetrics(trial_set.center_trials, center_l23som_site_counts, v1_genn::kL23SOMPerSite);
        const std::vector<PopulationSiteMetrics> broad_l23e_sites =
            computeSiteMetrics(trial_set.broad_trials, broad_l23e_site_counts, v1_genn::kL23EPerSite);
        const std::vector<PopulationSiteMetrics> broad_l23pv_sites =
            computeSiteMetrics(trial_set.broad_trials, broad_l23pv_site_counts, v1_genn::kL23PVPerSite);
        const std::vector<PopulationSiteMetrics> broad_l23som_sites =
            computeSiteMetrics(trial_set.broad_trials, broad_l23som_site_counts, v1_genn::kL23SOMPerSite);

        context_validation_metrics.push_back({
            trial_set.site_id,
            center_l23e_sites.at(trial_set.site_id),
            center_l23pv_sites.at(trial_set.site_id),
            center_l23som_sites.at(trial_set.site_id),
            broad_l23e_sites.at(trial_set.site_id),
            broad_l23pv_sites.at(trial_set.site_id),
            broad_l23som_sites.at(trial_set.site_id),
        });

        const std::vector<double> size_l4e_site_counts =
            countSiteSpikesForTrials(l4e_recordings.at(0), trial_set.size_trials, v1_genn::kL4EPerSite);
        const std::vector<double> size_l23e_site_counts =
            countSiteSpikesForTrials(l23e_recordings.at(0), trial_set.size_trials, v1_genn::kL23EPerSite);
        const std::vector<double> size_l23pv_site_counts =
            countSiteSpikesForTrials(l23pv_recordings.at(0), trial_set.size_trials, v1_genn::kL23PVPerSite);
        const std::vector<double> size_l23som_site_counts =
            countSiteSpikesForTrials(l23som_recordings.at(0), trial_set.size_trials, v1_genn::kL23SOMPerSite);
        const std::vector<PopulationSiteMetrics> size_l4e_sites =
            computeSiteMetrics(trial_set.size_trials, size_l4e_site_counts, v1_genn::kL4EPerSite);
        const std::vector<PopulationSiteMetrics> size_l23e_sites =
            computeSiteMetrics(trial_set.size_trials, size_l23e_site_counts, v1_genn::kL23EPerSite);
        const std::vector<PopulationSiteMetrics> size_l23pv_sites =
            computeSiteMetrics(trial_set.size_trials, size_l23pv_site_counts, v1_genn::kL23PVPerSite);
        const std::vector<PopulationSiteMetrics> size_l23som_sites =
            computeSiteMetrics(trial_set.size_trials, size_l23som_site_counts, v1_genn::kL23SOMPerSite);

        size_validation_metrics.push_back({
            trial_set.site_id,
            size_l4e_sites.at(trial_set.site_id),
            size_l23e_sites.at(trial_set.site_id),
            size_l23pv_sites.at(trial_set.site_id),
            size_l23som_sites.at(trial_set.site_id),
        });
    }
    const RetinotopicContextMetrics &primary_context_validation = context_validation_metrics.front();

    const std::vector<NamedWeightStats> additional_weight_stats{
        {"l23e_to_l23e", summarizeWeights(l23ee_weights_before), summarizeWeights(l23ee_weights_after)},
        {"l23pv_to_l23e", summarizeWeights(l23pv_weights_before), summarizeWeights(l23pv_weights_after)},
        {"l23som_to_l23e", summarizeWeights(l23som_weights_before), summarizeWeights(l23som_weights_after)},
    };
    const std::vector<ContextValidationSummary> context_validation{
        {
            "center_only",
            primary_context_validation.center_l23e.mean_rate_hz,
            primary_context_validation.center_l23pv.mean_rate_hz,
            primary_context_validation.center_l23som.mean_rate_hz,
        },
        {
            "broad_field",
            primary_context_validation.broad_l23e.mean_rate_hz,
            primary_context_validation.broad_l23pv.mean_rate_hz,
            primary_context_validation.broad_l23som.mean_rate_hz,
        },
    };

    writePopulationSiteMetricsCsv(output_prefix + "_baseline_l4_sites.csv", baseline, baseline.l4_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_baseline_l23_sites.csv", baseline, baseline.l23_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_baseline_l23pv_sites.csv", baseline, baseline_l23pv_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_baseline_l23som_sites.csv", baseline, baseline_l23som_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_baseline_l23vip_sites.csv", baseline, baseline_l23vip_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_post_l4_sites.csv", post, post.l4_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_post_l23_sites.csv", post, post.l23_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_post_l23pv_sites.csv", post, post_l23pv_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_post_l23som_sites.csv", post, post_l23som_sites);
    writePopulationSiteMetricsCsv(output_prefix + "_post_l23vip_sites.csv", post, post_l23vip_sites);
    writeL23ECellTuningCsv(output_prefix + "_l23e_cell_tuning.csv", orientations_rad, post_l23e_cell_tuning);
    if(multiphase_cell_coverage_enabled) {
        writeL23ECellTuningMultiPhaseCsv(
            output_prefix + "_l23e_cell_tuning_multiphase.csv",
            orientations_rad,
            multiphase_l23e_cell_tuning,
            cell_coverage_phase_count);
    }
    writeL23ECellTuningCsv(
        output_prefix + "_l23e_recurrence_context_tuning.csv",
        orientations_rad,
        recurrence_l23e_cell_tuning,
        l23ee_context_output_scale);
    writeSubtypeRatesCsv(
        output_prefix + "_subtype_rates.csv",
        orientations_rad,
        baseline_trials,
        post_trials,
        baseline_subtype_rates,
        post_subtype_rates);
    writeContextValidationCsv(
        output_prefix + "_som_context_validation.csv",
        orientations_rad,
        context_validation_metrics,
        validation_site_config.include_validation_site_id,
        l23som_output_scale * l23som_context_output_scale);
    writeSizeTuningCsv(
        output_prefix + "_size_tuning.csv",
        size_tuning_radii_sites,
        orientations_rad,
        size_validation_metrics,
        validation_site_config.include_validation_site_id,
        l23som_output_scale * l23som_context_output_scale);
    if(orientation_context_assay_config.enabled) {
        writeOrientationContextAssayCsv(
            output_prefix + "_l23_orientation_context_suppression.csv",
            orientation_context_metrics,
            validation_site_config.include_validation_site_id,
            l23som_output_scale * l23som_context_output_scale);
    }
    if(sensory_assay_config.enabled) {
        writeBlankBaselineCsv(
            output_prefix + "_blank_baseline.csv",
            blank_baseline_trials,
            blank_l4e_site_counts,
            blank_l23e_site_counts,
            blank_l23pv_site_counts,
            blank_l23som_site_counts);
        writeContrastSweepCsv(
            output_prefix + "_contrast_sweep.csv",
            contrast_sweep_records,
            contrast_sweep_trials,
            contrast_l4e_site_counts,
            contrast_l23e_site_counts,
            contrast_l23pv_site_counts,
            contrast_l23som_site_counts);
    }
    writeL4IntersiteDiagnosticsCsv(
        output_prefix + "_l4_intersite_diagnostics.csv",
        l4_intersite_config,
        size_tuning_radii_sites,
        size_validation_metrics,
        baseline,
        post);

    writeWeightCsv(output_prefix + "_weights_before.csv", weights_before, ff_edges);
    writeWeightCsv(output_prefix + "_weights_after.csv", weights_after, ff_edges);
    writeWeightCsv(output_prefix + "_l23ee_weights_before.csv", l23ee_weights_before, l23ee_edges);
    writeWeightCsv(output_prefix + "_l23ee_weights_after.csv", l23ee_weights_after, l23ee_edges);
    writeWeightCsv(output_prefix + "_l23pv_to_l23e_weights_before.csv", l23pv_weights_before, l23pv_edges);
    writeWeightCsv(output_prefix + "_l23pv_to_l23e_weights_after.csv", l23pv_weights_after, l23pv_edges);
    writeWeightCsv(output_prefix + "_l23som_to_l23e_weights_before.csv", l23som_weights_before, l23som_edges);
    writeWeightCsv(output_prefix + "_l23som_to_l23e_weights_after.csv", l23som_weights_after, l23som_edges);
    writeL23EESpecificityCsv(
        output_prefix + "_l23ee_specificity.csv",
        l23ee_weights_before,
        l23ee_weights_after,
        l23ee_edges,
        post_l23e_cell_tuning);
    const std::vector<double> l23ee_initial_active_values = positiveWeightValues(l23ee_weights_before);
    const double l23ee_initial_active_mean = l23ee_initial_active_values.empty()
        ? 0.0
        : (std::accumulate(l23ee_initial_active_values.begin(), l23ee_initial_active_values.end(), 0.0)
           / static_cast<double>(l23ee_initial_active_values.size()));
    const double l23ee_initial_active_gini = giniCoefficient(l23ee_initial_active_values);
    const double l23ee_initial_top10_mass_share = topMassShare(l23ee_initial_active_values, 0.10);
    writeSummaryFiles(
        output_prefix,
        baseline,
        post,
        summarizeWeights(weights_before),
        summarizeWeights(weights_after),
        additional_weight_stats,
        subtype_summaries,
        context_validation,
        training_grating_config,
        training_grating_phase_slot_ms,
        l4_l23_orientation_config,
        ff_edges.size(),
        nonzeroWeightFraction(weights_before),
        l23ee_lognormal_init_config,
        l23ee_initial_active_values.size(),
        l23ee_initial_active_mean,
        l23ee_initial_active_gini,
        l23ee_initial_top10_mass_share,
        l23pv_context_output_scale,
        orientation_context_assay_config,
        sensory_assay_config);
}
