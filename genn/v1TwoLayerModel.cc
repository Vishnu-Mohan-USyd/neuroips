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
constexpr unsigned int kDefaultTrainingEpochs = 2;
constexpr unsigned int kDefaultRecurrentConsolidationEpochs = 3;
constexpr double kDefaultStdpAplus = 0.00012;
constexpr double kDefaultStdpAminus = 0.000105;
constexpr unsigned int kDefaultL23EEPlasticityEnabled = 1;
constexpr double kDefaultL23EEStdpAplus = 0.000100;
constexpr double kDefaultL23EEStdpAminus = 0.000100;
constexpr unsigned int kDefaultL23PVHomeostaticEnabled = 1;
constexpr unsigned int kDefaultL23SOMHomeostaticEnabled = 1;
constexpr double kDefaultL23PVHomeostaticEta = 0.000020;
constexpr double kDefaultL23SOMHomeostaticEta = 0.000050;
constexpr double kDefaultL23PVHomeostaticTargetHz = 25.0;
constexpr double kDefaultL23SOMHomeostaticTargetHz = 5.0;
constexpr double kDefaultL23PVGate = 0.01;
constexpr double kDefaultL23SOMGate = 0.18;
constexpr double kDefaultL23VIPGate = 0.0;
constexpr double kDefaultL23SOMOutputScale = 1.0;
constexpr double kDefaultL23SOMContextOutputScale = 1.0;
constexpr double kDefaultL23PVContextOutputScale = 1.0;
constexpr double kDefaultL23EEContextOutputScale = 1.0;
constexpr unsigned int kDefaultL23ESOMBroadRecruitmentRadius = 6;
constexpr double kDefaultL23ESOMBroadRecruitmentWeightScale = 0.054;
constexpr double kDefaultCenterStimulusRadiusSites = 2.0;
constexpr double kDefaultBroadStimulusRadiusSites = 3.0;
constexpr char kDefaultSizeTuningRadiiSites[] = "0.5,1,2,3,4,6";
constexpr unsigned int kDefaultBlankRepeatCount = 4;
constexpr char kDefaultContrastSweepValues[] = "0.5,1.0";
constexpr double kDefaultVideoFrameMs = 100.0;
constexpr double kDefaultVideoEventPreMs = 50.0;
constexpr double kDefaultVideoEventPostMs = 100.0;
constexpr double kDefaultVideoEventBinMs = 2.0;
constexpr double kDefaultVideoEventGrayCurrent = -1.0;
constexpr unsigned int kDefaultVideoEventRepeatCount = 1;
constexpr unsigned int kDefaultVideoEventControlCount = 4;
constexpr unsigned int kDefaultVideoConsolidationRepeatCount = 1;
constexpr double kDefaultVideoPVReliabilityOutputScale = 0.975;
constexpr double kDefaultVideoSOMReliabilityOutputScale = 0.90;
constexpr double kDefaultVideoFFReliabilityOutputScale = 1.10;
constexpr unsigned int kDefaultHVAPredictorTileSizeSites = 4;
constexpr unsigned int kDefaultHVAPredictorDelayFrames = 1;
constexpr double kDefaultHVAPredictorTraceTauFrames = 2.0;
constexpr double kDefaultHVAPredictorLearningRate = 0.005;
constexpr double kDefaultHVAPredictorEventLearningRate = 0.001;
constexpr double kDefaultHVAPredictorBiasLearningRate = 0.005;
constexpr double kDefaultHVAPredictorEventBiasLearningRate = 0.0;
constexpr double kDefaultHVAPredictorWeightDecay = 0.001;
constexpr double kDefaultHVAPredictorEventWeightDecay = 0.005;
constexpr double kDefaultHVAPredictorEventResidualGain = 0.5;
constexpr double kDefaultHVAPredictorRateScaleHz = 10.0;
constexpr double kDefaultHVAPredictorWeightClip = 2.0;
constexpr double kDefaultHVAPredictorHeldoutFraction = 0.25;
constexpr unsigned int kDefaultHVAPredictorLocalRadiusTiles = 1;
constexpr unsigned int kDefaultHVAPredictorTrainingEpochs = 5;
constexpr unsigned int kDefaultHVAPredictorEventWindowFrames = 3;
constexpr unsigned int kDefaultHVAPredictorTopKFutureWindowFrames = 2;
constexpr unsigned int kDefaultHVAPredictorTopK = 5;
constexpr double kDefaultHVAPredictorTopKLearningRate = 0.005;
constexpr double kDefaultHVAPredictorTopKWeightDecay = 0.001;
constexpr unsigned int kDefaultHVAPredictorTopKTargetSmoothRadiusTiles = 0;
constexpr unsigned int kDefaultHVAPredictorFeatureLagCount = 5;
constexpr unsigned int kDefaultHVAPredictorFeatureContextRadiusTiles = 1;
constexpr unsigned int kDefaultHVASequenceStateDim = 4;
constexpr double kDefaultHVASequenceStateLeak = 0.85;
constexpr double kDefaultHVASequenceStateInputScale = 0.35;
constexpr double kDefaultHVASequenceStateNeighborScale = 0.25;
constexpr double kDefaultHVAPredictorEventThresholdQuantile = 0.85;
// HVA tiles average sparse L23E activity, so the fixed floor must stay below
// typical tile-mean rates; the train-only quantile remains the primary event definition.
constexpr double kDefaultHVAPredictorEventThresholdMinHz = 0.05;
constexpr unsigned int kDefaultHVAPredictorEventMinTrainPositiveCount = 2;
constexpr unsigned int kHVAPredictorTraceChannelCount = 3;
constexpr unsigned int kHVAPredictorBaseFeatureChannelCount = 5;
constexpr unsigned int kHVAPredictorContextSummaryFeatureCount = 3;
constexpr unsigned int kHVAPredictorDirectionalContextFeatureCount = 6;
constexpr unsigned int kHVAPredictorRequiredTargetChannelCount = 1;
constexpr double kHVAPredictorFeatureStdFloor = 1.0e-3;
constexpr double kHVAPredictorEventRateFloor = 1.0e-4;
constexpr double kHVAPredictorFastTraceTauMs = 50.0;
constexpr double kHVAPredictorMediumTraceTauMs = 150.0;
constexpr double kHVAPredictorSlowTraceTauMs = 500.0;
constexpr unsigned int kDefaultRecurrentOnlyConsolidationEpochs = 27;
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

struct L23ESOMBroadRecruitmentConfig {
    bool enabled = false;
    unsigned int radius = kDefaultL23ESOMBroadRecruitmentRadius;
    double weight_scale = kDefaultL23ESOMBroadRecruitmentWeightScale;
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

struct VideoReplayConfig {
    bool enabled = false;
    std::string drive_path;
    unsigned int frame_count = 0;
    unsigned int max_frames = 0;
    unsigned int effective_frame_count = 0;
    unsigned int repeat_count = 1;
    double frame_ms = kDefaultVideoFrameMs;
};

struct VideoEventTimingConfig {
    bool enabled = false;
    unsigned int max_events = 0;
    unsigned int effective_event_count = 0;
    unsigned int repeat_count = kDefaultVideoEventRepeatCount;
    unsigned int gray_control_count = kDefaultVideoEventControlCount;
    unsigned int blank_control_count = kDefaultVideoEventControlCount;
    double pre_ms = kDefaultVideoEventPreMs;
    double post_ms = kDefaultVideoEventPostMs;
    double bin_ms = kDefaultVideoEventBinMs;
    double gray_current = kDefaultVideoEventGrayCurrent;
    bool gray_from_frame_mean = true;
};

struct VideoConsolidationConfig {
    bool requested = false;
    bool enabled = false;
    unsigned int repeat_count = kDefaultVideoConsolidationRepeatCount;
    unsigned int frame_start_index = 0;
    unsigned int frame_count = 0;
    unsigned int heldout_start_frame = 0;
    unsigned int heldout_excluded_frame_count = 0;
    bool l23ee_plasticity_enabled = true;
    bool inhibitory_homeostasis_enabled = true;
};

struct VideoPVReliabilityConfig {
    bool enabled = false;
    double output_scale = 1.0;
};

struct VideoSOMReliabilityConfig {
    bool enabled = false;
    double output_scale = 1.0;
};

struct VideoFFReliabilityConfig {
    bool enabled = false;
    double output_scale = 1.0;
};

struct HVAPredictorConfig {
    bool enabled = false;
    unsigned int tile_size_sites = kDefaultHVAPredictorTileSizeSites;
    unsigned int tile_grid_side = 0;
    unsigned int delay_frames = kDefaultHVAPredictorDelayFrames;
    double trace_tau_frames = kDefaultHVAPredictorTraceTauFrames;
    double learning_rate = kDefaultHVAPredictorLearningRate;
    double event_learning_rate = kDefaultHVAPredictorEventLearningRate;
    double bias_learning_rate = kDefaultHVAPredictorBiasLearningRate;
    double event_bias_learning_rate = kDefaultHVAPredictorEventBiasLearningRate;
    double weight_decay = kDefaultHVAPredictorWeightDecay;
    double event_weight_decay = kDefaultHVAPredictorEventWeightDecay;
    double event_residual_gain = kDefaultHVAPredictorEventResidualGain;
    double rate_scale_hz = kDefaultHVAPredictorRateScaleHz;
    double weight_clip = kDefaultHVAPredictorWeightClip;
    double heldout_fraction = kDefaultHVAPredictorHeldoutFraction;
    unsigned int local_radius_tiles = kDefaultHVAPredictorLocalRadiusTiles;
    unsigned int topk_local_radius_tiles = kDefaultHVAPredictorLocalRadiusTiles;
    unsigned int training_epochs = kDefaultHVAPredictorTrainingEpochs;
    unsigned int event_window_frames = kDefaultHVAPredictorEventWindowFrames;
    unsigned int topk_future_window_frames = kDefaultHVAPredictorTopKFutureWindowFrames;
    unsigned int topk_k = kDefaultHVAPredictorTopK;
    double topk_learning_rate = kDefaultHVAPredictorTopKLearningRate;
    double topk_weight_decay = kDefaultHVAPredictorTopKWeightDecay;
    unsigned int topk_target_smooth_radius_tiles = kDefaultHVAPredictorTopKTargetSmoothRadiusTiles;
    unsigned int feature_lag_count = kDefaultHVAPredictorFeatureLagCount;
    unsigned int feature_context_radius_tiles = kDefaultHVAPredictorFeatureContextRadiusTiles;
    bool directional_context_enabled = true;
    bool sequence_state_enabled = false;
    unsigned int sequence_state_dim = kDefaultHVASequenceStateDim;
    double sequence_state_leak = kDefaultHVASequenceStateLeak;
    double sequence_state_input_scale = kDefaultHVASequenceStateInputScale;
    double sequence_state_neighbor_scale = kDefaultHVASequenceStateNeighborScale;
    bool topk_repeat_avg_target_enabled = false;
    bool topk_frequency_balance_enabled = false;
    double topk_frequency_balance_floor = 0.01;
    double event_threshold_quantile = kDefaultHVAPredictorEventThresholdQuantile;
    double event_threshold_min_hz = kDefaultHVAPredictorEventThresholdMinHz;
    unsigned int event_min_train_positive_count = kDefaultHVAPredictorEventMinTrainPositiveCount;
};

bool hvaPredictorDirectionalContextActive(const HVAPredictorConfig &config)
{
    return config.directional_context_enabled && config.feature_context_radius_tiles > 0u;
}

bool hvaPredictorSequenceStateActive(const HVAPredictorConfig &config)
{
    return config.sequence_state_enabled && config.sequence_state_dim > 0u;
}

unsigned int hvaPredictorNonSequenceFeatureChannelCount(const HVAPredictorConfig &config)
{
    return kHVAPredictorBaseFeatureChannelCount
        + config.feature_lag_count
        + (kHVAPredictorContextSummaryFeatureCount * (config.feature_lag_count + 1u))
        + (hvaPredictorDirectionalContextActive(config)
               ? (kHVAPredictorDirectionalContextFeatureCount * (config.feature_lag_count + 1u))
               : 0u);
}

unsigned int hvaPredictorFeatureChannelCount(const HVAPredictorConfig &config)
{
    return hvaPredictorNonSequenceFeatureChannelCount(config)
        + (hvaPredictorSequenceStateActive(config) ? config.sequence_state_dim : 0u);
}

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

struct VideoFrameRecord {
    unsigned int repeat_index = 0;
    unsigned int frame_index = 0;
    TrialWindow trial{};
    double drive_min = 0.0;
    double drive_mean = 0.0;
    double drive_max = 0.0;
    double drive_std = 0.0;
};

struct VideoEventTimingRecord {
    std::string condition;
    unsigned int repeat_index = 0;
    unsigned int event_index = 0;
    unsigned int frame_index = 0;
    TrialWindow trial{};
    double event_start_ms = 0.0;
    double gray_current = 0.0;
    double drive_min = 0.0;
    double drive_mean = 0.0;
    double drive_max = 0.0;
    double drive_std = 0.0;
};

struct VideoConsolidationMetrics {
    bool enabled = false;
    unsigned int frame_start_index = 0;
    unsigned int frame_count = 0;
    unsigned int heldout_start_frame = 0;
    unsigned int heldout_excluded_frame_count = 0;
    unsigned int pre_eval_trial_count = 0;
    unsigned int consolidation_trial_count = 0;
    unsigned int post_eval_trial_count = 0;
    double pre_l23e_repeat_corr = 0.0;
    double post_l23e_repeat_corr = 0.0;
    double delta_l23e_repeat_corr = 0.0;
    double pre_l23e_repeat_top5_overlap = 0.0;
    double post_l23e_repeat_top5_overlap = 0.0;
    double delta_l23e_repeat_top5_overlap = 0.0;
    double l4_l23_weight_delta_max = 0.0;
    double l23ee_weight_delta_max = 0.0;
    double l23pv_weight_delta_max = 0.0;
    double l23som_weight_delta_max = 0.0;
};

struct HVAPredictorRateRow {
    unsigned int sample_index = 0;
    unsigned int repeat_index = 0;
    unsigned int frame_index = 0;
    unsigned int tile_id = 0;
    unsigned int tile_x = 0;
    unsigned int tile_y = 0;
    double l23e_rate_hz = 0.0;
    double state_norm = 0.0;
    double eligibility_trace = 0.0;
    double trace_fast = 0.0;
    double trace_medium = 0.0;
    double trace_slow = 0.0;
    double derivative = 0.0;
};

struct HVAPredictorPredictionRow {
    unsigned int prediction_index = 0;
    unsigned int repeat_index = 0;
    unsigned int frame_index = 0;
    unsigned int target_frame_index = 0;
    unsigned int target_channel_index = 0;
    std::string target_channel = "l23e";
    unsigned int tile_id = 0;
    unsigned int tile_x = 0;
    unsigned int tile_y = 0;
    std::string split = "train";
    bool learning_update_applied = false;
    double current_state_norm = 0.0;
    double target_state_norm = 0.0;
    double predicted_state_norm = 0.0;
    double target_residual_norm = 0.0;
    double predicted_residual_norm = 0.0;
    double target_residual_z = 0.0;
    double predicted_residual_z = 0.0;
    double train_residual_mean_norm = 0.0;
    double train_residual_std_norm = 1.0;
    double persistence_pred_state_norm = 0.0;
    double train_mean_pred_state_norm = 0.0;
    double no_learning_pred_state_norm = 0.0;
    double temporal_block_shift_pred_state_norm = 0.0;
    double spatial_tile_shuffle_pred_state_norm = 0.0;
    double target_rate_hz = 0.0;
    double predicted_rate_hz = 0.0;
    double error_rate_hz = 0.0;
    double event_window_target_state_norm = 0.0;
    double event_threshold_norm = 0.0;
    bool event_tile_selected = false;
    unsigned int target_event = 0;
    unsigned int single_frame_target_event = 0;
    double predicted_event_prob = 0.0;
    double persistence_event_prob = 0.0;
    double train_event_rate = 0.0;
    double no_learning_event_prob = 0.0;
    double temporal_block_shift_event_prob = 0.0;
    double spatial_tile_shuffle_event_prob = 0.0;
    double event_error = 0.0;
    double topk_target_value_norm = 0.0;
    bool topk_target = false;
    bool topk_sample_valid = false;
    double topk_model_score = 0.0;
    double topk_model_prob = 0.0;
    double topk_persistence_score = 0.0;
    double topk_train_frequency_score = 0.0;
    double topk_no_learning_score = 0.0;
    double topk_temporal_block_shift_score = 0.0;
    double topk_spatial_tile_shuffle_score = 0.0;
};

struct HVAPredictorEventTileRow {
    unsigned int target_channel_index = 0;
    std::string target_channel = "l23e";
    unsigned int tile_id = 0;
    unsigned int tile_x = 0;
    unsigned int tile_y = 0;
    double threshold_norm = 0.0;
    double threshold_hz = 0.0;
    unsigned int train_count = 0;
    unsigned int train_positive_count = 0;
    unsigned int train_negative_count = 0;
    unsigned int heldout_count = 0;
    unsigned int heldout_positive_count = 0;
    double train_positive_fraction = 0.0;
    double heldout_positive_fraction = 0.0;
    bool selected = false;
};

struct HVAPredictorResult {
    std::vector<HVAPredictorRateRow> rates;
    std::vector<HVAPredictorPredictionRow> predictions;
    std::vector<HVAPredictorEventTileRow> event_tiles;
    std::vector<std::string> target_channels;
    std::vector<bool> target_channel_required;
    std::vector<double> weights_before;
    std::vector<double> weights_after;
    std::vector<double> readout_weights_before;
    std::vector<double> readout_weights_after;
    std::vector<double> biases_after;
    std::vector<double> event_weights_before;
    std::vector<double> event_weights_after;
    std::vector<double> event_biases_after;
    std::vector<double> topk_weights_before;
    std::vector<double> topk_weights_after;
    std::vector<double> topk_biases_after;
    std::vector<std::pair<std::string, double>> metrics;
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

L23ESOMBroadRecruitmentConfig getL23ESOMBroadRecruitmentConfig()
{
    L23ESOMBroadRecruitmentConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_L23E_SOM_BROAD_RECRUIT_ENABLE", 1u) != 0u;
    config.radius = getEnvUnsignedOrDefault(
        "V1_L23E_SOM_BROAD_RECRUIT_RADIUS",
        kDefaultL23ESOMBroadRecruitmentRadius);
    config.weight_scale = getEnvDoubleOrDefault(
        "V1_L23E_SOM_BROAD_RECRUIT_WEIGHT_SCALE",
        kDefaultL23ESOMBroadRecruitmentWeightScale);
    if(config.radius == 0u || config.radius >= v1_genn::kSheetSide) {
        throw std::runtime_error("V1_L23E_SOM_BROAD_RECRUIT_RADIUS must be in [1, V1_SHEET_SIDE).");
    }
    if(config.weight_scale < 0.0 || config.weight_scale > 1.0) {
        throw std::runtime_error("V1_L23E_SOM_BROAD_RECRUIT_WEIGHT_SCALE must be in [0, 1].");
    }
    if(config.enabled && config.radius <= v1_genn::kL23SOMInputRadius) {
        throw std::runtime_error("V1_L23E_SOM_BROAD_RECRUIT_RADIUS must exceed kL23SOMInputRadius when enabled.");
    }
    if(config.enabled && config.weight_scale == 0.0) {
        throw std::runtime_error("V1_L23E_SOM_BROAD_RECRUIT_WEIGHT_SCALE must be positive when enabled.");
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

VideoReplayConfig getVideoReplayConfig()
{
    VideoReplayConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_VIDEO_REPLAY_ENABLE", 0u) != 0u;
    config.drive_path = getEnvOrDefault("V1_VIDEO_DRIVE_BIN", "");
    config.frame_count = getEnvUnsignedOrDefault("V1_VIDEO_FRAME_COUNT", 0u);
    config.max_frames = getEnvUnsignedOrDefault("V1_VIDEO_MAX_FRAMES", 0u);
    config.repeat_count = getEnvUnsignedOrDefault("V1_VIDEO_REPLAY_REPEAT_COUNT", 1u);
    config.frame_ms = getEnvDoubleOrDefault("V1_VIDEO_FRAME_MS", kDefaultVideoFrameMs);
    if(!config.enabled) {
        return config;
    }
    if(config.drive_path.empty()) {
        throw std::runtime_error("V1_VIDEO_DRIVE_BIN is required when V1_VIDEO_REPLAY_ENABLE=1.");
    }
    if(config.frame_count == 0u) {
        throw std::runtime_error("V1_VIDEO_FRAME_COUNT must be positive when V1_VIDEO_REPLAY_ENABLE=1.");
    }
    if(config.frame_ms <= 0.0 || !std::isfinite(config.frame_ms)) {
        throw std::runtime_error("V1_VIDEO_FRAME_MS must be finite and positive.");
    }
    if(config.repeat_count == 0u) {
        throw std::runtime_error("V1_VIDEO_REPLAY_REPEAT_COUNT must be at least 1.");
    }
    config.effective_frame_count = (config.max_frames > 0u)
        ? std::min(config.frame_count, config.max_frames)
        : config.frame_count;
    if(config.effective_frame_count == 0u) {
        throw std::runtime_error("V1_VIDEO_MAX_FRAMES selected zero frames.");
    }
    return config;
}

VideoPVReliabilityConfig getVideoPVReliabilityConfig(const VideoReplayConfig &video_config)
{
    VideoPVReliabilityConfig config;
    const bool enable_requested =
        getEnvUnsignedOrDefault("V1_VIDEO_PV_RELIABILITY_TUNING_ENABLE", 1u) != 0u;
    config.enabled = video_config.enabled && enable_requested;
    config.output_scale = config.enabled
        ? getEnvDoubleOrDefault(
            "V1_VIDEO_PV_RELIABILITY_OUTPUT_SCALE",
            kDefaultVideoPVReliabilityOutputScale)
        : 1.0;

    if(!config.enabled) {
        return config;
    }
    if(!video_config.enabled) {
        throw std::runtime_error(
            "V1 video PV reliability tuning requires V1_VIDEO_REPLAY_ENABLE=1.");
    }
    if(!std::isfinite(config.output_scale)
       || config.output_scale < 0.80
       || config.output_scale > 1.0) {
        throw std::runtime_error(
            "V1_VIDEO_PV_RELIABILITY_OUTPUT_SCALE must be finite and in [0.80, 1.0].");
    }
    return config;
}

VideoSOMReliabilityConfig getVideoSOMReliabilityConfig(const VideoReplayConfig &video_config)
{
    VideoSOMReliabilityConfig config;
    const bool enable_requested =
        getEnvUnsignedOrDefault("V1_VIDEO_SOM_RELIABILITY_TUNING_ENABLE", 1u) != 0u;
    config.enabled = video_config.enabled && enable_requested;
    config.output_scale = config.enabled
        ? getEnvDoubleOrDefault(
            "V1_VIDEO_SOM_RELIABILITY_OUTPUT_SCALE",
            kDefaultVideoSOMReliabilityOutputScale)
        : 1.0;

    if(!config.enabled) {
        return config;
    }
    if(!std::isfinite(config.output_scale)
       || config.output_scale < 0.80
       || config.output_scale > 1.0) {
        throw std::runtime_error(
            "V1_VIDEO_SOM_RELIABILITY_OUTPUT_SCALE must be finite and in [0.80, 1.0].");
    }
    return config;
}

VideoFFReliabilityConfig getVideoFFReliabilityConfig(const VideoReplayConfig &video_config)
{
    VideoFFReliabilityConfig config;
    const bool enable_requested =
        getEnvUnsignedOrDefault("V1_VIDEO_FF_RELIABILITY_TUNING_ENABLE", 1u) != 0u;
    config.enabled = video_config.enabled && enable_requested;
    config.output_scale = config.enabled
        ? getEnvDoubleOrDefault(
            "V1_VIDEO_L4E_L23E_OUTPUT_SCALE",
            kDefaultVideoFFReliabilityOutputScale)
        : 1.0;

    if(!config.enabled) {
        return config;
    }
    if(!std::isfinite(config.output_scale)
       || config.output_scale < 1.0
       || config.output_scale > 1.20) {
        throw std::runtime_error(
            "V1_VIDEO_L4E_L23E_OUTPUT_SCALE must be finite and in [1.0, 1.20].");
    }
    return config;
}

VideoEventTimingConfig getVideoEventTimingConfig(const VideoReplayConfig &video_config)
{
    VideoEventTimingConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_VIDEO_EVENT_TIMING_ENABLE", 0u) != 0u;
    config.max_events = getEnvUnsignedOrDefault("V1_VIDEO_EVENT_MAX_EVENTS", 0u);
    config.repeat_count = getEnvUnsignedOrDefault("V1_VIDEO_EVENT_REPEAT_COUNT", kDefaultVideoEventRepeatCount);
    config.gray_control_count = getEnvUnsignedOrDefault(
        "V1_VIDEO_EVENT_GRAY_CONTROL_COUNT",
        kDefaultVideoEventControlCount);
    config.blank_control_count = getEnvUnsignedOrDefault(
        "V1_VIDEO_EVENT_BLANK_CONTROL_COUNT",
        kDefaultVideoEventControlCount);
    config.pre_ms = getEnvDoubleOrDefault("V1_VIDEO_EVENT_PRE_MS", kDefaultVideoEventPreMs);
    config.post_ms = getEnvDoubleOrDefault("V1_VIDEO_EVENT_POST_MS", kDefaultVideoEventPostMs);
    config.bin_ms = getEnvDoubleOrDefault("V1_VIDEO_EVENT_BIN_MS", kDefaultVideoEventBinMs);
    config.gray_current = getEnvDoubleOrDefault("V1_VIDEO_EVENT_GRAY_CURRENT", kDefaultVideoEventGrayCurrent);
    config.gray_from_frame_mean = (config.gray_current < 0.0);

    if(!config.enabled) {
        return config;
    }
    if(!video_config.enabled) {
        throw std::runtime_error("V1_VIDEO_EVENT_TIMING_ENABLE=1 requires V1_VIDEO_REPLAY_ENABLE=1 and a video drive.");
    }
    if(config.repeat_count == 0u) {
        throw std::runtime_error("V1_VIDEO_EVENT_REPEAT_COUNT must be at least 1 when event timing is enabled.");
    }
    if(config.pre_ms <= 0.0 || config.post_ms <= 0.0) {
        throw std::runtime_error("V1_VIDEO_EVENT_PRE_MS and V1_VIDEO_EVENT_POST_MS must be positive.");
    }
    if(config.bin_ms < 1.0 || config.bin_ms > 2.0) {
        throw std::runtime_error("V1_VIDEO_EVENT_BIN_MS must be in [1, 2] ms.");
    }
    if(!config.gray_from_frame_mean && config.gray_current < 0.0) {
        throw std::runtime_error("V1_VIDEO_EVENT_GRAY_CURRENT must be non-negative, or negative to use frame-mean gray.");
    }

    config.effective_event_count = (config.max_events > 0u)
        ? std::min(video_config.effective_frame_count, config.max_events)
        : video_config.effective_frame_count;
    if(config.effective_event_count == 0u) {
        throw std::runtime_error("V1_VIDEO_EVENT_MAX_EVENTS selected zero events.");
    }
    config.gray_control_count = std::min(config.gray_control_count, config.effective_event_count);
    config.blank_control_count = std::min(config.blank_control_count, config.effective_event_count);
    return config;
}

HVAPredictorConfig getHVAPredictorConfig(const VideoReplayConfig &video_config)
{
    HVAPredictorConfig config;
    config.enabled = getEnvUnsignedOrDefault("V1_HVA_PREDICTOR_ENABLE", 0u) != 0u;
    config.tile_size_sites = getEnvUnsignedOrDefault(
        "V1_HVA_PREDICTOR_TILE_SIZE_SITES",
        kDefaultHVAPredictorTileSizeSites);
    const char *tile_grid_side_env = std::getenv("V1_HVA_PREDICTOR_TILE_GRID_SIDE");
    if(tile_grid_side_env != nullptr && tile_grid_side_env[0] != '\0') {
        config.tile_grid_side = getEnvUnsignedOrDefault("V1_HVA_PREDICTOR_TILE_GRID_SIDE", 0u);
    }
    else if(config.tile_size_sites > 0u) {
        config.tile_grid_side =
            (v1_genn::kSheetSide + config.tile_size_sites - 1u) / config.tile_size_sites;
    }
    config.delay_frames = getEnvUnsignedOrDefault(
        "V1_HVA_PREDICTOR_DELAY_FRAMES",
        kDefaultHVAPredictorDelayFrames);
    config.trace_tau_frames = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_TRACE_TAU_FRAMES",
        kDefaultHVAPredictorTraceTauFrames);
    const double legacy_learning_rate = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_LEARNING_RATE",
        kDefaultHVAPredictorLearningRate);
    config.learning_rate = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_RESIDUAL_LEARNING_RATE",
        legacy_learning_rate);
    config.event_learning_rate = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_EVENT_LEARNING_RATE",
        kDefaultHVAPredictorEventLearningRate);
    config.bias_learning_rate = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_BIAS_LEARNING_RATE",
        kDefaultHVAPredictorBiasLearningRate);
    config.event_bias_learning_rate = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_EVENT_BIAS_LEARNING_RATE",
        kDefaultHVAPredictorEventBiasLearningRate);
    config.weight_decay = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_WEIGHT_DECAY",
        kDefaultHVAPredictorWeightDecay);
    config.event_weight_decay = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_EVENT_WEIGHT_DECAY",
        kDefaultHVAPredictorEventWeightDecay);
    config.event_residual_gain = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_EVENT_RESIDUAL_GAIN",
        kDefaultHVAPredictorEventResidualGain);
    config.rate_scale_hz = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_RATE_SCALE_HZ",
        kDefaultHVAPredictorRateScaleHz);
    config.weight_clip = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_WEIGHT_CLIP",
        kDefaultHVAPredictorWeightClip);
    config.heldout_fraction = getEnvDoubleOrDefault(
        "V1_HVA_PREDICTOR_HELDOUT_FRACTION",
        kDefaultHVAPredictorHeldoutFraction);
    config.local_radius_tiles = getEnvUnsignedOrDefault(
        "V1_HVA_PREDICTOR_LOCAL_RADIUS_TILES",
        kDefaultHVAPredictorLocalRadiusTiles);
    config.topk_local_radius_tiles = getEnvUnsignedOrDefault(
        "V1_HVA_TOPK_LOCAL_RADIUS_TILES",
        config.local_radius_tiles);
    config.training_epochs = getEnvUnsignedOrDefault(
        "V1_HVA_PREDICTOR_EPOCHS",
        kDefaultHVAPredictorTrainingEpochs);
    config.event_window_frames = getEnvUnsignedOrDefault(
        "V1_HVA_EVENT_WINDOW_FRAMES",
        kDefaultHVAPredictorEventWindowFrames);
    config.topk_future_window_frames = getEnvUnsignedOrDefault(
        "V1_HVA_TOPK_FUTURE_WINDOW_FRAMES",
        kDefaultHVAPredictorTopKFutureWindowFrames);
    config.topk_k = getEnvUnsignedOrDefault(
        "V1_HVA_TOPK_K",
        kDefaultHVAPredictorTopK);
    config.topk_learning_rate = getEnvDoubleOrDefault(
        "V1_HVA_TOPK_LEARNING_RATE",
        kDefaultHVAPredictorTopKLearningRate);
    config.topk_weight_decay = getEnvDoubleOrDefault(
        "V1_HVA_TOPK_WEIGHT_DECAY",
        kDefaultHVAPredictorTopKWeightDecay);
    config.topk_target_smooth_radius_tiles = getEnvUnsignedOrDefault(
        "V1_HVA_TOPK_TARGET_SMOOTH_RADIUS_TILES",
        kDefaultHVAPredictorTopKTargetSmoothRadiusTiles);
    config.feature_lag_count = getEnvUnsignedOrDefault(
        "V1_HVA_FEATURE_LAG_COUNT",
        kDefaultHVAPredictorFeatureLagCount);
    config.feature_context_radius_tiles = getEnvUnsignedOrDefault(
        "V1_HVA_FEATURE_CONTEXT_RADIUS_TILES",
        kDefaultHVAPredictorFeatureContextRadiusTiles);
    config.directional_context_enabled =
        getEnvUnsignedOrDefault("V1_HVA_DIRECTIONAL_CONTEXT_ENABLE", 1u) != 0u;
    config.sequence_state_enabled =
        getEnvUnsignedOrDefault("V1_HVA_SEQUENCE_STATE_ENABLE", 0u) != 0u;
    config.sequence_state_dim = getEnvUnsignedOrDefault(
        "V1_HVA_SEQUENCE_STATE_DIM",
        kDefaultHVASequenceStateDim);
    config.sequence_state_leak = getEnvDoubleOrDefault(
        "V1_HVA_SEQUENCE_STATE_LEAK",
        kDefaultHVASequenceStateLeak);
    config.sequence_state_input_scale = getEnvDoubleOrDefault(
        "V1_HVA_SEQUENCE_STATE_INPUT_SCALE",
        kDefaultHVASequenceStateInputScale);
    config.sequence_state_neighbor_scale = getEnvDoubleOrDefault(
        "V1_HVA_SEQUENCE_STATE_NEIGHBOR_SCALE",
        kDefaultHVASequenceStateNeighborScale);
    config.topk_repeat_avg_target_enabled =
        getEnvUnsignedOrDefault("V1_HVA_TOPK_REPEAT_AVG_TARGET_ENABLE", 0u) != 0u;
    config.topk_frequency_balance_enabled =
        getEnvUnsignedOrDefault("V1_HVA_TOPK_FREQUENCY_BALANCE_ENABLE", 0u) != 0u;
    config.topk_frequency_balance_floor = getEnvDoubleOrDefault(
        "V1_HVA_TOPK_FREQUENCY_BALANCE_FLOOR",
        0.01);
    config.event_threshold_quantile = getEnvDoubleOrDefault(
        "V1_HVA_EVENT_THRESHOLD_QUANTILE",
        kDefaultHVAPredictorEventThresholdQuantile);
    config.event_threshold_min_hz = getEnvDoubleOrDefault(
        "V1_HVA_EVENT_THRESHOLD_MIN_HZ",
        kDefaultHVAPredictorEventThresholdMinHz);
    config.event_min_train_positive_count = getEnvUnsignedOrDefault(
        "V1_HVA_EVENT_MIN_TRAIN_POSITIVES",
        kDefaultHVAPredictorEventMinTrainPositiveCount);

    if(!config.enabled) {
        return config;
    }
    if(!video_config.enabled) {
        throw std::runtime_error("V1_HVA_PREDICTOR_ENABLE=1 requires V1_VIDEO_REPLAY_ENABLE=1.");
    }
    if(config.tile_size_sites == 0u || config.tile_size_sites > v1_genn::kSheetSide) {
        throw std::runtime_error("V1_HVA_PREDICTOR_TILE_SIZE_SITES must be in [1, V1_SHEET_SIDE].");
    }
    if(config.tile_grid_side == 0u || config.tile_grid_side > v1_genn::kSheetSide) {
        throw std::runtime_error("V1_HVA_PREDICTOR_TILE_GRID_SIDE must be in [1, V1_SHEET_SIDE], or set a valid tile size.");
    }
    if(config.delay_frames == 0u || config.delay_frames >= video_config.effective_frame_count) {
        throw std::runtime_error("V1_HVA_PREDICTOR_DELAY_FRAMES must be in [1, effective video frame count).");
    }
    if(config.trace_tau_frames <= 0.0 || !std::isfinite(config.trace_tau_frames)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_TRACE_TAU_FRAMES must be finite and positive.");
    }
    if(config.learning_rate < 0.0 || !std::isfinite(config.learning_rate)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_RESIDUAL_LEARNING_RATE must be finite and non-negative.");
    }
    if(config.event_learning_rate < 0.0 || !std::isfinite(config.event_learning_rate)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_EVENT_LEARNING_RATE must be finite and non-negative.");
    }
    if(config.bias_learning_rate < 0.0 || !std::isfinite(config.bias_learning_rate)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_BIAS_LEARNING_RATE must be finite and non-negative.");
    }
    if(config.event_bias_learning_rate < 0.0 || !std::isfinite(config.event_bias_learning_rate)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_EVENT_BIAS_LEARNING_RATE must be finite and non-negative.");
    }
    if(config.weight_decay < 0.0 || config.weight_decay >= 1.0 || !std::isfinite(config.weight_decay)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_WEIGHT_DECAY must be finite and in [0, 1).");
    }
    if(config.event_weight_decay < 0.0 || config.event_weight_decay >= 1.0 || !std::isfinite(config.event_weight_decay)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_EVENT_WEIGHT_DECAY must be finite and in [0, 1).");
    }
    if(config.event_residual_gain < 0.0 || !std::isfinite(config.event_residual_gain)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_EVENT_RESIDUAL_GAIN must be finite and non-negative.");
    }
    if(config.rate_scale_hz <= 0.0 || !std::isfinite(config.rate_scale_hz)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_RATE_SCALE_HZ must be finite and positive.");
    }
    if(config.weight_clip <= 0.0 || !std::isfinite(config.weight_clip)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_WEIGHT_CLIP must be finite and positive.");
    }
    if(config.heldout_fraction <= 0.0 || config.heldout_fraction >= 1.0 || !std::isfinite(config.heldout_fraction)) {
        throw std::runtime_error("V1_HVA_PREDICTOR_HELDOUT_FRACTION must be finite and in (0, 1).");
    }
    if(config.local_radius_tiles >= config.tile_grid_side) {
        throw std::runtime_error("V1_HVA_PREDICTOR_LOCAL_RADIUS_TILES must be smaller than tile grid side.");
    }
    if(config.topk_local_radius_tiles >= config.tile_grid_side) {
        throw std::runtime_error("V1_HVA_TOPK_LOCAL_RADIUS_TILES must be smaller than tile grid side.");
    }
    if(config.training_epochs == 0u) {
        throw std::runtime_error("V1_HVA_PREDICTOR_EPOCHS must be at least 1.");
    }
    if(config.event_window_frames == 0u) {
        throw std::runtime_error("V1_HVA_EVENT_WINDOW_FRAMES must be at least 1.");
    }
    if(config.topk_future_window_frames == 0u) {
        throw std::runtime_error("V1_HVA_TOPK_FUTURE_WINDOW_FRAMES must be at least 1.");
    }
    if(config.topk_k == 0u || config.topk_k > (config.tile_grid_side * config.tile_grid_side)) {
        throw std::runtime_error("V1_HVA_TOPK_K must be in [1, tile_count].");
    }
    if(config.topk_learning_rate < 0.0 || !std::isfinite(config.topk_learning_rate)) {
        throw std::runtime_error("V1_HVA_TOPK_LEARNING_RATE must be finite and non-negative.");
    }
    if(config.topk_weight_decay < 0.0 || config.topk_weight_decay >= 1.0 || !std::isfinite(config.topk_weight_decay)) {
        throw std::runtime_error("V1_HVA_TOPK_WEIGHT_DECAY must be finite and in [0, 1).");
    }
    if(config.topk_target_smooth_radius_tiles > 1u) {
        throw std::runtime_error("V1_HVA_TOPK_TARGET_SMOOTH_RADIUS_TILES currently supports only 0 or 1.");
    }
    if(config.topk_target_smooth_radius_tiles >= config.tile_grid_side) {
        throw std::runtime_error("V1_HVA_TOPK_TARGET_SMOOTH_RADIUS_TILES must be smaller than tile grid side.");
    }
    if(config.feature_lag_count > 64u) {
        throw std::runtime_error("V1_HVA_FEATURE_LAG_COUNT must be at most 64.");
    }
    if(config.feature_context_radius_tiles >= config.tile_grid_side) {
        throw std::runtime_error("V1_HVA_FEATURE_CONTEXT_RADIUS_TILES must be smaller than tile grid side.");
    }
    if(config.sequence_state_enabled && config.sequence_state_dim == 0u) {
        throw std::runtime_error("V1_HVA_SEQUENCE_STATE_DIM must be at least 1 when sequence state is enabled.");
    }
    if(config.sequence_state_dim > 32u) {
        throw std::runtime_error("V1_HVA_SEQUENCE_STATE_DIM must be at most 32.");
    }
    if(config.sequence_state_leak < 0.0 || config.sequence_state_leak > 1.0 || !std::isfinite(config.sequence_state_leak)) {
        throw std::runtime_error("V1_HVA_SEQUENCE_STATE_LEAK must be finite and in [0, 1].");
    }
    if(config.sequence_state_input_scale < 0.0 || !std::isfinite(config.sequence_state_input_scale)) {
        throw std::runtime_error("V1_HVA_SEQUENCE_STATE_INPUT_SCALE must be finite and non-negative.");
    }
    if(config.sequence_state_neighbor_scale < 0.0 || !std::isfinite(config.sequence_state_neighbor_scale)) {
        throw std::runtime_error("V1_HVA_SEQUENCE_STATE_NEIGHBOR_SCALE must be finite and non-negative.");
    }
    if(config.topk_frequency_balance_floor <= 0.0
       || config.topk_frequency_balance_floor > 1.0
       || !std::isfinite(config.topk_frequency_balance_floor)) {
        throw std::runtime_error("V1_HVA_TOPK_FREQUENCY_BALANCE_FLOOR must be finite and in (0, 1].");
    }
    if(config.event_threshold_quantile <= 0.0
       || config.event_threshold_quantile >= 1.0
       || !std::isfinite(config.event_threshold_quantile)) {
        throw std::runtime_error("V1_HVA_EVENT_THRESHOLD_QUANTILE must be finite and in (0, 1).");
    }
    if(config.event_threshold_min_hz < 0.0 || !std::isfinite(config.event_threshold_min_hz)) {
        throw std::runtime_error("V1_HVA_EVENT_THRESHOLD_MIN_HZ must be finite and non-negative.");
    }
    if(config.event_min_train_positive_count == 0u) {
        throw std::runtime_error("V1_HVA_EVENT_MIN_TRAIN_POSITIVES must be at least 1.");
    }
    return config;
}

unsigned int hvaPredictorHeldoutStartFrame(
    const VideoReplayConfig &video_config,
    const HVAPredictorConfig &config)
{
    if(video_config.effective_frame_count <= (config.delay_frames + 1u)) {
        throw std::runtime_error("HVA predictor requires enough video frames for train and held-out prediction windows.");
    }
    const unsigned int requested_heldout_frames = std::max(
        config.delay_frames + 1u,
        static_cast<unsigned int>(std::ceil(
            static_cast<double>(video_config.effective_frame_count) * config.heldout_fraction)));
    const unsigned int max_heldout_frames = video_config.effective_frame_count - config.delay_frames - 1u;
    const unsigned int heldout_frames = std::max(1u, std::min(requested_heldout_frames, max_heldout_frames));
    return video_config.effective_frame_count - heldout_frames;
}

VideoConsolidationConfig getVideoConsolidationConfig(
    const VideoReplayConfig &video_config,
    const HVAPredictorConfig &hva_predictor_config)
{
    VideoConsolidationConfig config;
    config.requested = getEnvUnsignedOrDefault("V1_VIDEO_CONSOLIDATION_ENABLE", 0u) != 0u;
    config.repeat_count = getEnvUnsignedOrDefault(
        "V1_VIDEO_CONSOLIDATION_REPEAT_COUNT",
        kDefaultVideoConsolidationRepeatCount);
    config.l23ee_plasticity_enabled =
        getEnvUnsignedOrDefault("V1_VIDEO_CONSOLIDATION_L23EE_STDP_ENABLE", 1u) != 0u;
    config.inhibitory_homeostasis_enabled =
        getEnvUnsignedOrDefault("V1_VIDEO_CONSOLIDATION_INHIBITORY_HOMEO_ENABLE", 1u) != 0u;

    if(!config.requested) {
        return config;
    }
    if(!video_config.enabled) {
        throw std::runtime_error("V1_VIDEO_CONSOLIDATION_ENABLE=1 requires V1_VIDEO_REPLAY_ENABLE=1.");
    }
    if(config.repeat_count == 0u) {
        throw std::runtime_error("V1_VIDEO_CONSOLIDATION_REPEAT_COUNT must be at least 1 when enabled.");
    }
    // Consolidation is explicitly a pre-HVA lower-V1 stage; control/somoff/recoff
    // jobs that do not train HVA skip it even if the env is inherited.
    if(!hva_predictor_config.enabled) {
        return config;
    }
    if(!config.l23ee_plasticity_enabled && !config.inhibitory_homeostasis_enabled) {
        throw std::runtime_error(
            "Video consolidation needs L23E->L23E plasticity and/or inhibitory homeostasis enabled.");
    }

    config.heldout_start_frame = hvaPredictorHeldoutStartFrame(video_config, hva_predictor_config);
    config.frame_start_index = 0u;
    config.frame_count = config.heldout_start_frame;
    config.heldout_excluded_frame_count = video_config.effective_frame_count - config.heldout_start_frame;
    if(config.frame_count == 0u || config.heldout_excluded_frame_count == 0u) {
        throw std::runtime_error("Video consolidation requires non-empty train and held-out frame blocks.");
    }

    config.enabled = true;
    return config;
}

std::vector<float> loadVideoDriveFrames(const VideoReplayConfig &config)
{
    if(!config.enabled) {
        return {};
    }

    const std::size_t frame_size = v1_genn::kNumL4E;
    const std::size_t value_count = static_cast<std::size_t>(config.effective_frame_count) * frame_size;
    std::vector<float> drive_frames(value_count, 0.0f);
    std::ifstream input(config.drive_path.c_str(), std::ios::binary);
    if(!input) {
        throw std::runtime_error("Unable to open V1_VIDEO_DRIVE_BIN: " + config.drive_path);
    }
    input.read(
        reinterpret_cast<char *>(drive_frames.data()),
        static_cast<std::streamsize>(value_count * sizeof(float)));
    if(input.gcount() != static_cast<std::streamsize>(value_count * sizeof(float))) {
        std::ostringstream message;
        message << "V1_VIDEO_DRIVE_BIN is too short for "
                << config.effective_frame_count << " frames x "
                << frame_size << " L4E values.";
        throw std::runtime_error(message.str());
    }
    return drive_frames;
}

VideoFrameRecord summarizeVideoDriveFrame(
    const std::vector<float> &drive_frames,
    unsigned int repeat_index,
    unsigned int frame_index,
    const TrialWindow &trial)
{
    const std::size_t frame_size = v1_genn::kNumL4E;
    const std::size_t offset = static_cast<std::size_t>(frame_index) * frame_size;
    if(offset + frame_size > drive_frames.size()) {
        throw std::runtime_error("Video drive frame index exceeds loaded drive data.");
    }

    VideoFrameRecord record;
    record.repeat_index = repeat_index;
    record.frame_index = frame_index;
    record.trial = trial;
    record.drive_min = std::numeric_limits<double>::infinity();
    record.drive_max = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sum_sq = 0.0;
    for(std::size_t i = 0; i < frame_size; i++) {
        const double value = static_cast<double>(drive_frames[offset + i]);
        record.drive_min = std::min(record.drive_min, value);
        record.drive_max = std::max(record.drive_max, value);
        sum += value;
        sum_sq += value * value;
    }
    record.drive_mean = sum / static_cast<double>(frame_size);
    const double variance = std::max(
        0.0,
        (sum_sq / static_cast<double>(frame_size)) - (record.drive_mean * record.drive_mean));
    record.drive_std = std::sqrt(variance);
    return record;
}

VideoEventTimingRecord makeVideoEventTimingRecord(
    const std::string &condition,
    const std::vector<float> &drive_frames,
    unsigned int repeat_index,
    unsigned int event_index,
    unsigned int frame_index,
    const TrialWindow &trial,
    double event_start_ms,
    double gray_current,
    bool use_frame_drive_stats)
{
    VideoEventTimingRecord record;
    record.condition = condition;
    record.repeat_index = repeat_index;
    record.event_index = event_index;
    record.frame_index = frame_index;
    record.trial = trial;
    record.event_start_ms = event_start_ms;
    record.gray_current = gray_current;
    if(use_frame_drive_stats) {
        const VideoFrameRecord frame_record = summarizeVideoDriveFrame(
            drive_frames,
            repeat_index,
            frame_index,
            trial);
        record.drive_min = frame_record.drive_min;
        record.drive_mean = frame_record.drive_mean;
        record.drive_max = frame_record.drive_max;
        record.drive_std = frame_record.drive_std;
    }
    else {
        record.drive_min = gray_current;
        record.drive_mean = gray_current;
        record.drive_max = gray_current;
        record.drive_std = 0.0;
    }
    return record;
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

void fillRuntimeArray(GeNN::Runtime::ArrayBase &array, double value)
{
    std::fill(array.getHostPointer<float>(), array.getHostPointer<float>() + array.getCount(), static_cast<float>(value));
    array.pushToDevice();
}

void resetNeuronTrialState(
    GeNN::Runtime::Runtime &runtime,
    const GeNN::NeuronGroup &group,
    const v1_genn::LIFParameters &params)
{
    fillRuntimeArray(requireArray(runtime, group, "V"), params.v_rest_mv);
    fillRuntimeArray(requireArray(runtime, group, "RefracTime"), 0.0);
}

void resetHomeostaticTraceState(
    GeNN::Runtime::Runtime &runtime,
    const GeNN::SynapseGroup &group)
{
    fillRuntimeArray(requireArray(runtime, group, "preTrace"), 0.0);
    fillRuntimeArray(requireArray(runtime, group, "postTrace"), 0.0);
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

void setSynapseWeights(
    GeNN::Runtime::Runtime &runtime,
    GeNN::SynapseGroup &synapse_group,
    const std::vector<float> &weights)
{
    GeNN::Runtime::ArrayBase &weight_array = requireArray(runtime, synapse_group, "g");
    if(weights.size() != weight_array.getCount()) {
        throw std::runtime_error("Synapse weight restore size does not match runtime array.");
    }
    float *runtime_weights = weight_array.getHostPointer<float>();
    std::copy(weights.begin(), weights.end(), runtime_weights);
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

double maxAbsDifference(const std::vector<float> &before, const std::vector<float> &after)
{
    if(before.size() != after.size()) {
        throw std::runtime_error("Weight delta requires vectors with matching sizes.");
    }
    double delta = 0.0;
    for(std::size_t i = 0; i < before.size(); i++) {
        delta = std::max(delta, std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i])));
    }
    return delta;
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

template <typename SpikeBatch>
std::vector<double> countPopulationSpikesForEventBins(
    const SpikeBatch &batch,
    const std::vector<VideoEventTimingRecord> &records,
    unsigned int bin_count,
    double bin_ms)
{
    std::vector<double> counts(static_cast<std::size_t>(records.size()) * bin_count, 0.0);
    if(records.empty()) {
        return counts;
    }

    const auto &spike_times = batch.first;
    const auto &spike_ids = batch.second;
    if(spike_times.size() != spike_ids.size()) {
        throw std::runtime_error("Recorded spike times and ids do not have matching lengths.");
    }

    std::size_t record_index = 0;
    for(std::size_t i = 0; i < spike_times.size() && record_index < records.size(); i++) {
        const double spike_time = static_cast<double>(spike_times[i]);
        while(record_index < records.size() && spike_time >= records[record_index].trial.end_ms) {
            record_index++;
        }
        if(record_index >= records.size()) {
            break;
        }

        const VideoEventTimingRecord &record = records[record_index];
        if(spike_time < record.trial.start_ms || spike_time >= record.trial.end_ms) {
            continue;
        }
        const double relative_to_trial_ms = spike_time - record.trial.start_ms;
        const unsigned int bin_index = static_cast<unsigned int>(std::floor(relative_to_trial_ms / bin_ms));
        if(bin_index < bin_count) {
            counts[(record_index * bin_count) + bin_index] += 1.0;
        }
    }
    return counts;
}

template <typename SpikeBatch>
std::vector<double> countSiteSpikesForEventBins(
    const SpikeBatch &batch,
    const std::vector<VideoEventTimingRecord> &records,
    const std::vector<unsigned int> &site_ids,
    unsigned int neurons_per_site,
    unsigned int bin_count,
    double bin_ms)
{
    std::vector<double> counts(
        static_cast<std::size_t>(records.size()) * bin_count * site_ids.size(),
        0.0);
    if(records.empty() || site_ids.empty()) {
        return counts;
    }

    std::vector<unsigned int> site_lookup(
        v1_genn::kSiteCount,
        std::numeric_limits<unsigned int>::max());
    for(unsigned int i = 0; i < site_ids.size(); i++) {
        if(site_ids[i] >= v1_genn::kSiteCount) {
            throw std::runtime_error("Video event site export id exceeds site count.");
        }
        site_lookup[site_ids[i]] = i;
    }

    const auto &spike_times = batch.first;
    const auto &spike_ids = batch.second;
    if(spike_times.size() != spike_ids.size()) {
        throw std::runtime_error("Recorded spike times and ids do not have matching lengths.");
    }

    std::size_t record_index = 0;
    for(std::size_t i = 0; i < spike_times.size() && record_index < records.size(); i++) {
        const double spike_time = static_cast<double>(spike_times[i]);
        while(record_index < records.size() && spike_time >= records[record_index].trial.end_ms) {
            record_index++;
        }
        if(record_index >= records.size()) {
            break;
        }

        const VideoEventTimingRecord &record = records[record_index];
        if(spike_time < record.trial.start_ms || spike_time >= record.trial.end_ms) {
            continue;
        }
        const unsigned int neuron_id = static_cast<unsigned int>(spike_ids[i]);
        const unsigned int site_id = neuron_id / neurons_per_site;
        if(site_id >= v1_genn::kSiteCount) {
            throw std::runtime_error("Recorded spike id maps outside the site grid.");
        }
        const unsigned int site_export_index = site_lookup[site_id];
        if(site_export_index == std::numeric_limits<unsigned int>::max()) {
            continue;
        }
        const double relative_to_trial_ms = spike_time - record.trial.start_ms;
        const unsigned int bin_index = static_cast<unsigned int>(std::floor(relative_to_trial_ms / bin_ms));
        if(bin_index < bin_count) {
            const std::size_t count_index =
                ((record_index * bin_count) + bin_index) * site_ids.size() + site_export_index;
            counts[count_index] += 1.0;
        }
    }
    return counts;
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

double standardDeviation(const std::vector<double> &values)
{
    if(values.size() < 2u) {
        return 0.0;
    }
    const double mean = meanRate(values);
    double sum_sq = 0.0;
    for(double value : values) {
        const double centered = value - mean;
        sum_sq += centered * centered;
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

double clippedValue(double value, double lower, double upper)
{
    return std::min(upper, std::max(lower, value));
}

double sumValues(const std::vector<double> &values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

double sumSquares(const std::vector<double> &values)
{
    double total = 0.0;
    for(double value : values) {
        total += value * value;
    }
    return total;
}

std::uint32_t quantizedVectorFingerprint32(const std::vector<double> &values)
{
    std::uint64_t hash = 1469598103934665603ull;
    for(double value : values) {
        const std::int64_t quantized = static_cast<std::int64_t>(std::llround(value * 1000000.0));
        std::uint64_t encoded = static_cast<std::uint64_t>(quantized);
        for(unsigned int byte_index = 0; byte_index < 8u; byte_index++) {
            hash ^= (encoded & 0xffu);
            hash *= 1099511628211ull;
            encoded >>= 8u;
        }
    }
    return static_cast<std::uint32_t>(hash & 0xffffffffu);
}

unsigned int hvaTileIdForSite(unsigned int site_id, unsigned int tile_grid_side)
{
    const auto xy = v1_genn::siteIndexToXY(site_id);
    const unsigned int tile_x = std::min(
        tile_grid_side - 1u,
        (xy.first * tile_grid_side) / v1_genn::kSheetSide);
    const unsigned int tile_y = std::min(
        tile_grid_side - 1u,
        (xy.second * tile_grid_side) / v1_genn::kSheetSide);
    return (tile_y * tile_grid_side) + tile_x;
}

std::vector<double> makeL23ETileRatesForVideoTrials(
    const std::vector<TrialWindow> &trials,
    const std::vector<double> &site_spike_counts,
    unsigned int tile_grid_side)
{
    const unsigned int tile_count = tile_grid_side * tile_grid_side;
    if(tile_grid_side == 0u || site_spike_counts.size() != (trials.size() * v1_genn::kSiteCount)) {
        throw std::runtime_error("Video consolidation tile-rate input dimensions are inconsistent.");
    }

    std::vector<unsigned int> sites_per_tile(tile_count, 0u);
    for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
        sites_per_tile[hvaTileIdForSite(site_id, tile_grid_side)]++;
    }

    std::vector<double> rates(trials.size() * static_cast<std::size_t>(tile_count), 0.0);
    for(std::size_t trial_index = 0; trial_index < trials.size(); trial_index++) {
        const double duration_s = (trials[trial_index].end_ms - trials[trial_index].measure_start_ms) / 1000.0;
        if(duration_s <= 0.0) {
            throw std::runtime_error("Video consolidation trial duration must be positive.");
        }
        for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
            const unsigned int tile_id = hvaTileIdForSite(site_id, tile_grid_side);
            const double spikes = site_spike_counts[
                (trial_index * static_cast<std::size_t>(v1_genn::kSiteCount)) + site_id];
            rates[(trial_index * tile_count) + tile_id] +=
                spikes / (duration_s * static_cast<double>(v1_genn::kL23EPerSite));
        }
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            if(sites_per_tile[tile_id] == 0u) {
                throw std::runtime_error("Video consolidation tile grid produced an empty tile.");
            }
            rates[(trial_index * tile_count) + tile_id] /= static_cast<double>(sites_per_tile[tile_id]);
        }
    }
    return rates;
}

std::vector<TrialWindow> selectVideoFrameBlockTrials(
    const std::vector<TrialWindow> &trials,
    unsigned int repeat_count,
    unsigned int source_frame_count,
    unsigned int frame_start_index,
    unsigned int frame_count)
{
    const std::size_t expected_size =
        static_cast<std::size_t>(repeat_count) * source_frame_count;
    if(trials.size() != expected_size || frame_start_index + frame_count > source_frame_count) {
        throw std::runtime_error("Video trial block selection received inconsistent dimensions.");
    }

    std::vector<TrialWindow> selected;
    selected.reserve(static_cast<std::size_t>(repeat_count) * frame_count);
    for(unsigned int repeat_index = 0; repeat_index < repeat_count; repeat_index++) {
        for(unsigned int frame_offset = 0; frame_offset < frame_count; frame_offset++) {
            const std::size_t source_index =
                (static_cast<std::size_t>(repeat_index) * source_frame_count)
                + frame_start_index
                + frame_offset;
            selected.push_back(trials[source_index]);
        }
    }
    return selected;
}

std::vector<double> selectVideoFrameBlockSiteCounts(
    const std::vector<double> &site_spike_counts,
    unsigned int repeat_count,
    unsigned int source_frame_count,
    unsigned int frame_start_index,
    unsigned int frame_count)
{
    const std::size_t expected_size =
        static_cast<std::size_t>(repeat_count) * source_frame_count * v1_genn::kSiteCount;
    if(site_spike_counts.size() != expected_size || frame_start_index + frame_count > source_frame_count) {
        throw std::runtime_error("Video site-count block selection received inconsistent dimensions.");
    }

    std::vector<double> selected;
    selected.reserve(static_cast<std::size_t>(repeat_count) * frame_count * v1_genn::kSiteCount);
    for(unsigned int repeat_index = 0; repeat_index < repeat_count; repeat_index++) {
        for(unsigned int frame_offset = 0; frame_offset < frame_count; frame_offset++) {
            const std::size_t source_index =
                (((static_cast<std::size_t>(repeat_index) * source_frame_count)
                  + frame_start_index
                  + frame_offset)
                 * v1_genn::kSiteCount);
            selected.insert(
                selected.end(),
                site_spike_counts.begin() + source_index,
                site_spike_counts.begin() + source_index + v1_genn::kSiteCount);
        }
    }
    return selected;
}

std::vector<unsigned int> topKTileIds(const std::vector<double> &values, unsigned int k)
{
    std::vector<unsigned int> ids(values.size(), 0u);
    std::iota(ids.begin(), ids.end(), 0u);
    std::sort(ids.begin(), ids.end(), [&](unsigned int left, unsigned int right) {
        if(values[left] == values[right]) {
            return left < right;
        }
        return values[left] > values[right];
    });
    ids.resize(std::min<std::size_t>(k, ids.size()));
    return ids;
}

double meanVideoRepeatCorrelation(
    const std::vector<double> &tile_rates,
    unsigned int repeat_count,
    unsigned int frame_count,
    unsigned int tile_count)
{
    if(repeat_count < 2u || frame_count == 0u || tile_count == 0u) {
        return 0.0;
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(repeat_count) * frame_count * tile_count;
    if(tile_rates.size() != expected_size) {
        throw std::runtime_error("Video repeat correlation received inconsistent tile-rate dimensions.");
    }

    std::vector<double> correlations;
    correlations.reserve(static_cast<std::size_t>(repeat_count) * repeat_count);
    for(unsigned int repeat_a = 0; repeat_a < repeat_count; repeat_a++) {
        for(unsigned int repeat_b = repeat_a + 1u; repeat_b < repeat_count; repeat_b++) {
            std::vector<double> rates_a;
            std::vector<double> rates_b;
            rates_a.reserve(static_cast<std::size_t>(frame_count) * tile_count);
            rates_b.reserve(static_cast<std::size_t>(frame_count) * tile_count);
            for(unsigned int frame_index = 0; frame_index < frame_count; frame_index++) {
                const std::size_t base_a =
                    ((static_cast<std::size_t>(repeat_a) * frame_count) + frame_index) * tile_count;
                const std::size_t base_b =
                    ((static_cast<std::size_t>(repeat_b) * frame_count) + frame_index) * tile_count;
                rates_a.insert(rates_a.end(), tile_rates.begin() + base_a, tile_rates.begin() + base_a + tile_count);
                rates_b.insert(rates_b.end(), tile_rates.begin() + base_b, tile_rates.begin() + base_b + tile_count);
            }
            correlations.push_back(responseCorrelation(rates_a, rates_b));
        }
    }
    return correlations.empty() ? 0.0 : meanRate(correlations);
}

double meanVideoRepeatTopKOverlap(
    const std::vector<double> &tile_rates,
    unsigned int repeat_count,
    unsigned int frame_count,
    unsigned int tile_count,
    unsigned int k)
{
    if(repeat_count < 2u || frame_count == 0u || tile_count == 0u || k == 0u) {
        return 0.0;
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(repeat_count) * frame_count * tile_count;
    if(tile_rates.size() != expected_size) {
        throw std::runtime_error("Video repeat top-k overlap received inconsistent tile-rate dimensions.");
    }

    double overlap_sum = 0.0;
    unsigned int comparison_count = 0u;
    std::vector<double> values_a(tile_count, 0.0);
    std::vector<double> values_b(tile_count, 0.0);
    for(unsigned int repeat_a = 0; repeat_a < repeat_count; repeat_a++) {
        for(unsigned int repeat_b = repeat_a + 1u; repeat_b < repeat_count; repeat_b++) {
            for(unsigned int frame_index = 0; frame_index < frame_count; frame_index++) {
                const std::size_t base_a =
                    ((static_cast<std::size_t>(repeat_a) * frame_count) + frame_index) * tile_count;
                const std::size_t base_b =
                    ((static_cast<std::size_t>(repeat_b) * frame_count) + frame_index) * tile_count;
                std::copy(tile_rates.begin() + base_a, tile_rates.begin() + base_a + tile_count, values_a.begin());
                std::copy(tile_rates.begin() + base_b, tile_rates.begin() + base_b + tile_count, values_b.begin());
                const std::vector<unsigned int> top_a = topKTileIds(values_a, k);
                const std::vector<unsigned int> top_b = topKTileIds(values_b, k);
                unsigned int shared_count = 0u;
                for(unsigned int tile_a : top_a) {
                    if(std::find(top_b.begin(), top_b.end(), tile_a) != top_b.end()) {
                        shared_count++;
                    }
                }
                overlap_sum += static_cast<double>(shared_count)
                    / static_cast<double>(std::max<std::size_t>(1u, top_a.size()));
                comparison_count++;
            }
        }
    }
    return comparison_count == 0u
        ? 0.0
        : (overlap_sum / static_cast<double>(comparison_count));
}

VideoConsolidationMetrics computeVideoConsolidationMetrics(
    const VideoConsolidationConfig &config,
    const VideoReplayConfig &video_config,
    const HVAPredictorConfig &hva_config,
    const std::vector<TrialWindow> &pre_trials,
    const std::vector<double> &pre_l23e_site_counts,
    const std::vector<TrialWindow> &post_trials,
    const std::vector<double> &post_l23e_site_counts,
    const std::vector<TrialWindow> &consolidation_trials,
    double l4_l23_weight_delta_max,
    double l23ee_weight_delta_max,
    double l23pv_weight_delta_max,
    double l23som_weight_delta_max)
{
    VideoConsolidationMetrics metrics;
    metrics.enabled = config.enabled;
    metrics.frame_start_index = config.frame_start_index;
    metrics.frame_count = config.frame_count;
    metrics.heldout_start_frame = config.heldout_start_frame;
    metrics.heldout_excluded_frame_count = config.heldout_excluded_frame_count;
    metrics.pre_eval_trial_count = static_cast<unsigned int>(pre_trials.size());
    metrics.consolidation_trial_count = static_cast<unsigned int>(consolidation_trials.size());
    metrics.l4_l23_weight_delta_max = l4_l23_weight_delta_max;
    metrics.l23ee_weight_delta_max = l23ee_weight_delta_max;
    metrics.l23pv_weight_delta_max = l23pv_weight_delta_max;
    metrics.l23som_weight_delta_max = l23som_weight_delta_max;
    if(!config.enabled) {
        return metrics;
    }
    if(config.frame_count == 0u) {
        throw std::runtime_error("Video consolidation metrics require a non-empty train-frame block.");
    }

    const std::vector<TrialWindow> post_eval_trials = selectVideoFrameBlockTrials(
        post_trials,
        video_config.repeat_count,
        video_config.effective_frame_count,
        config.frame_start_index,
        config.frame_count);
    const std::vector<double> post_eval_l23e_site_counts = selectVideoFrameBlockSiteCounts(
        post_l23e_site_counts,
        video_config.repeat_count,
        video_config.effective_frame_count,
        config.frame_start_index,
        config.frame_count);
    metrics.post_eval_trial_count = static_cast<unsigned int>(post_eval_trials.size());

    const unsigned int tile_count = hva_config.tile_grid_side * hva_config.tile_grid_side;
    const std::vector<double> pre_tile_rates = makeL23ETileRatesForVideoTrials(
        pre_trials,
        pre_l23e_site_counts,
        hva_config.tile_grid_side);
    const std::vector<double> post_tile_rates = makeL23ETileRatesForVideoTrials(
        post_eval_trials,
        post_eval_l23e_site_counts,
        hva_config.tile_grid_side);
    metrics.pre_l23e_repeat_corr = meanVideoRepeatCorrelation(
        pre_tile_rates,
        video_config.repeat_count,
        config.frame_count,
        tile_count);
    metrics.post_l23e_repeat_corr = meanVideoRepeatCorrelation(
        post_tile_rates,
        video_config.repeat_count,
        config.frame_count,
        tile_count);
    metrics.delta_l23e_repeat_corr =
        metrics.post_l23e_repeat_corr - metrics.pre_l23e_repeat_corr;
    metrics.pre_l23e_repeat_top5_overlap = meanVideoRepeatTopKOverlap(
        pre_tile_rates,
        video_config.repeat_count,
        config.frame_count,
        tile_count,
        5u);
    metrics.post_l23e_repeat_top5_overlap = meanVideoRepeatTopKOverlap(
        post_tile_rates,
        video_config.repeat_count,
        config.frame_count,
        tile_count,
        5u);
    metrics.delta_l23e_repeat_top5_overlap =
        metrics.post_l23e_repeat_top5_overlap - metrics.pre_l23e_repeat_top5_overlap;
    return metrics;
}

HVAPredictorResult trainHVAPredictorSidecar(
    const HVAPredictorConfig &config,
    const VideoReplayConfig &video_config,
    const std::vector<VideoFrameRecord> &frame_records,
    const std::vector<double> &l23e_site_spike_counts)
{
    HVAPredictorResult result;
    if(!config.enabled) {
        return result;
    }
    if(frame_records.size() != (static_cast<std::size_t>(video_config.effective_frame_count)
                                * static_cast<std::size_t>(video_config.repeat_count))) {
        throw std::runtime_error("HVA predictor expected one video frame record per replay presentation.");
    }
    const std::size_t expected_site_count_size =
        frame_records.size() * static_cast<std::size_t>(v1_genn::kSiteCount);
    if(l23e_site_spike_counts.size() != expected_site_count_size) {
        throw std::runtime_error("HVA predictor L23E site count vector has unexpected size.");
    }

    struct HVATargetChannelSpec {
        std::string name;
        const std::vector<double> *site_spike_counts = nullptr;
        unsigned int neurons_per_site = 1u;
        bool required = true;
    };
    std::vector<HVATargetChannelSpec> target_specs{
        {"l23e", &l23e_site_spike_counts, v1_genn::kL23EPerSite, true},
    };
    result.target_channels.reserve(target_specs.size());
    result.target_channel_required.reserve(target_specs.size());
    for(const HVATargetChannelSpec &target_spec : target_specs) {
        result.target_channels.push_back(target_spec.name);
        result.target_channel_required.push_back(target_spec.required);
    }

    const unsigned int tile_count = config.tile_grid_side * config.tile_grid_side;
    const unsigned int target_channel_count = static_cast<unsigned int>(target_specs.size());
    const unsigned int feature_channel_count = hvaPredictorFeatureChannelCount(config);
    const unsigned int non_sequence_feature_channel_count =
        hvaPredictorNonSequenceFeatureChannelCount(config);
    const bool sequence_state_active = hvaPredictorSequenceStateActive(config);
    const std::size_t pair_count = static_cast<std::size_t>(tile_count) * tile_count;
    const std::size_t target_pair_count =
        static_cast<std::size_t>(target_channel_count) * pair_count;
    const std::size_t readout_weight_count = target_pair_count * feature_channel_count;
    const double trace_decay = std::exp(-1.0 / config.trace_tau_frames);
    const std::array<double, kHVAPredictorTraceChannelCount> trace_tau_ms = {{
        kHVAPredictorFastTraceTauMs,
        kHVAPredictorMediumTraceTauMs,
        kHVAPredictorSlowTraceTauMs,
    }};
    std::array<double, kHVAPredictorTraceChannelCount> trace_tau_frames{};
    std::array<double, kHVAPredictorTraceChannelCount> trace_decays{};
    for(unsigned int channel = 0; channel < kHVAPredictorTraceChannelCount; channel++) {
        trace_tau_frames[channel] = std::max(1.0e-6, trace_tau_ms[channel] / video_config.frame_ms);
        trace_decays[channel] = std::exp(-1.0 / trace_tau_frames[channel]);
    }
    std::vector<unsigned int> sites_per_tile(tile_count, 0u);
    for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
        sites_per_tile[hvaTileIdForSite(site_id, config.tile_grid_side)]++;
    }

    const auto pairIndex = [&](unsigned int post_tile, unsigned int pre_tile) {
        return (static_cast<std::size_t>(post_tile) * tile_count) + pre_tile;
    };
    const auto targetTileIndex = [&](unsigned int target_channel, unsigned int tile_id) {
        return (static_cast<std::size_t>(target_channel) * tile_count) + tile_id;
    };
    const auto targetPairIndex = [&](unsigned int target_channel, unsigned int post_tile, unsigned int pre_tile) {
        return (static_cast<std::size_t>(target_channel) * pair_count) + pairIndex(post_tile, pre_tile);
    };
    const auto readoutIndex = [&](unsigned int target_channel,
                                  unsigned int post_tile,
                                  unsigned int pre_tile,
                                  unsigned int feature) {
        return (targetPairIndex(target_channel, post_tile, pre_tile) * feature_channel_count) + feature;
    };
    const auto manhattanDistance = [&](unsigned int pre_tile, unsigned int post_tile) {
        const int dx = static_cast<int>(pre_tile % config.tile_grid_side)
            - static_cast<int>(post_tile % config.tile_grid_side);
        const int dy = static_cast<int>(pre_tile / config.tile_grid_side)
            - static_cast<int>(post_tile / config.tile_grid_side);
        return static_cast<unsigned int>(std::abs(dx) + std::abs(dy));
    };
    const auto localReadoutEnabled = [&](unsigned int pre_tile, unsigned int post_tile) {
        return manhattanDistance(pre_tile, post_tile) <= config.local_radius_tiles;
    };
    const auto topKReadoutEnabled = [&](unsigned int pre_tile, unsigned int post_tile) {
        return manhattanDistance(pre_tile, post_tile) <= config.topk_local_radius_tiles;
    };

    const std::size_t sample_count = frame_records.size();
    const auto makeTileRates = [&](const std::vector<double> &site_spike_counts,
                                   unsigned int neurons_per_site) {
        std::vector<double> rates(sample_count * static_cast<std::size_t>(tile_count), 0.0);
        for(std::size_t sample_index = 0; sample_index < sample_count; sample_index++) {
            const double measurement_duration_s =
                (frame_records[sample_index].trial.end_ms - frame_records[sample_index].trial.measure_start_ms) / 1000.0;
            if(measurement_duration_s <= 0.0) {
                throw std::runtime_error("HVA predictor video replay trial duration must be positive.");
            }
            for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
                const unsigned int tile_id = hvaTileIdForSite(site_id, config.tile_grid_side);
                const double spikes = site_spike_counts[
                    (sample_index * static_cast<std::size_t>(v1_genn::kSiteCount)) + site_id];
                const double site_rate_hz =
                    spikes / (measurement_duration_s * static_cast<double>(neurons_per_site));
                rates[(sample_index * tile_count) + tile_id] += site_rate_hz;
            }
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                if(sites_per_tile[tile_id] == 0u) {
                    throw std::runtime_error("HVA predictor tile grid produced an empty tile.");
                }
                rates[(sample_index * tile_count) + tile_id] /=
                    static_cast<double>(sites_per_tile[tile_id]);
            }
        }
        return rates;
    };

    std::vector<double> input_tile_rates =
        makeTileRates(l23e_site_spike_counts, v1_genn::kL23EPerSite);
    std::vector<std::vector<double>> target_tile_rates;
    target_tile_rates.reserve(target_channel_count);
    for(const HVATargetChannelSpec &target_spec : target_specs) {
        target_tile_rates.push_back(makeTileRates(*target_spec.site_spike_counts, target_spec.neurons_per_site));
    }

    auto normalizedRate = [&](double rate_hz) {
        return clippedValue(rate_hz / config.rate_scale_hz, 0.0, 1.0);
    };
    const auto sigmoid = [](double value) {
        const double clipped = clippedValue(value, -60.0, 60.0);
        return 1.0 / (1.0 + std::exp(-clipped));
    };
    const auto logit = [](double probability) {
        const double p = clippedValue(
            probability,
            kHVAPredictorEventRateFloor,
            1.0 - kHVAPredictorEventRateFloor);
        return std::log(p / (1.0 - p));
    };
    const auto quantileValue = [](std::vector<double> values, double quantile) {
        if(values.empty()) {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        const double position = quantile * static_cast<double>(values.size() - 1u);
        const std::size_t lower_index = static_cast<std::size_t>(std::floor(position));
        const std::size_t upper_index = std::min(lower_index + 1u, values.size() - 1u);
        const double fraction = position - static_cast<double>(lower_index);
        return values[lower_index] + (fraction * (values[upper_index] - values[lower_index]));
    };
    const auto featureIndex = [&](std::size_t sample_index, unsigned int tile_id, unsigned int feature) {
        return ((sample_index * static_cast<std::size_t>(tile_count) + tile_id)
                * feature_channel_count)
            + feature;
    };
    std::vector<double> feature_series(
        sample_count * static_cast<std::size_t>(tile_count) * feature_channel_count,
        0.0);
    const auto inputStateAtLag = [&](unsigned int repeat_index,
                                     unsigned int frame_index,
                                     unsigned int tile_id,
                                     unsigned int lag) {
        if(frame_index < lag) {
            return 0.0;
        }
        const std::size_t lag_sample_index =
            (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count)
            + (frame_index - lag);
        return normalizedRate(input_tile_rates[(lag_sample_index * tile_count) + tile_id]);
    };
    const auto sequenceStateIndex = [&](std::size_t sample_index, unsigned int tile_id, unsigned int state_dim) {
        return ((sample_index * static_cast<std::size_t>(tile_count) + tile_id)
                * config.sequence_state_dim)
            + state_dim;
    };
    std::vector<double> sequence_state_series(
        sequence_state_active
            ? (sample_count * static_cast<std::size_t>(tile_count) * config.sequence_state_dim)
            : 0u,
        0.0);
    if(sequence_state_active) {
        for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
            std::vector<double> previous_hidden(
                static_cast<std::size_t>(tile_count) * config.sequence_state_dim,
                0.0);
            std::vector<double> next_hidden(previous_hidden.size(), 0.0);
            for(unsigned int frame_index = 0; frame_index < video_config.effective_frame_count; frame_index++) {
                const std::size_t sample_index =
                    (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame_index;
                for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                    const double current_state = inputStateAtLag(repeat_index, frame_index, tile_id, 0u);
                    const double lag1_state = inputStateAtLag(repeat_index, frame_index, tile_id, 1u);
                    const double derivative = current_state - lag1_state;
                    double neighbor_current_sum = 0.0;
                    unsigned int neighbor_current_count = 0u;
                    for(unsigned int other_tile = 0; other_tile < tile_count; other_tile++) {
                        if(other_tile == tile_id || manhattanDistance(other_tile, tile_id) > 1u) {
                            continue;
                        }
                        neighbor_current_sum += inputStateAtLag(repeat_index, frame_index, other_tile, 0u);
                        neighbor_current_count++;
                    }
                    const double neighbor_current_mean = neighbor_current_count > 0u
                        ? (neighbor_current_sum / static_cast<double>(neighbor_current_count))
                        : 0.0;
                    for(unsigned int state_dim = 0; state_dim < config.sequence_state_dim; state_dim++) {
                        double neighbor_hidden_sum = 0.0;
                        unsigned int neighbor_hidden_count = 0u;
                        for(unsigned int other_tile = 0; other_tile < tile_count; other_tile++) {
                            if(other_tile == tile_id || manhattanDistance(other_tile, tile_id) > 1u) {
                                continue;
                            }
                            neighbor_hidden_sum += previous_hidden[
                                (static_cast<std::size_t>(other_tile) * config.sequence_state_dim) + state_dim];
                            neighbor_hidden_count++;
                        }
                        const double neighbor_hidden_mean = neighbor_hidden_count > 0u
                            ? (neighbor_hidden_sum / static_cast<double>(neighbor_hidden_count))
                            : 0.0;
                        double local_basis = current_state;
                        switch(state_dim % 4u) {
                        case 0u:
                            local_basis = current_state;
                            break;
                        case 1u:
                            local_basis = lag1_state;
                            break;
                        case 2u:
                            local_basis = derivative;
                            break;
                        default:
                            local_basis = neighbor_current_mean;
                            break;
                        }
                        const std::size_t hidden_index =
                            (static_cast<std::size_t>(tile_id) * config.sequence_state_dim) + state_dim;
                        const double updated =
                            std::tanh(
                                (config.sequence_state_leak * previous_hidden[hidden_index])
                                + (config.sequence_state_input_scale * local_basis)
                                + (config.sequence_state_neighbor_scale * neighbor_hidden_mean));
                        next_hidden[hidden_index] = updated;
                        sequence_state_series[sequenceStateIndex(sample_index, tile_id, state_dim)] = updated;
                    }
                }
                previous_hidden.swap(next_hidden);
                std::fill(next_hidden.begin(), next_hidden.end(), 0.0);
            }
        }
    }
    for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
        std::vector<double> traces(
            static_cast<std::size_t>(tile_count) * kHVAPredictorTraceChannelCount,
            0.0);
        std::vector<double> previous_state(tile_count, 0.0);
        for(unsigned int frame_index = 0; frame_index < video_config.effective_frame_count; frame_index++) {
            const std::size_t sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame_index;
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                const double current_state =
                    normalizedRate(input_tile_rates[(sample_index * tile_count) + tile_id]);
                const double derivative =
                    (frame_index == 0u) ? 0.0 : (current_state - previous_state[tile_id]);
                feature_series[featureIndex(sample_index, tile_id, 0u)] = current_state;
                for(unsigned int channel = 0; channel < kHVAPredictorTraceChannelCount; channel++) {
                    const std::size_t trace_index =
                        (static_cast<std::size_t>(tile_id) * kHVAPredictorTraceChannelCount) + channel;
                    traces[trace_index] =
                        (trace_decays[channel] * traces[trace_index])
                        + ((1.0 - trace_decays[channel]) * current_state);
                    feature_series[featureIndex(sample_index, tile_id, 1u + channel)] =
                        traces[trace_index];
                }
                feature_series[featureIndex(sample_index, tile_id, 4u)] = derivative;
                previous_state[tile_id] = current_state;
                unsigned int feature_offset = kHVAPredictorBaseFeatureChannelCount;
                for(unsigned int lag = 1u; lag <= config.feature_lag_count; lag++) {
                    feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                        inputStateAtLag(repeat_index, frame_index, tile_id, lag);
                    feature_offset++;
                }
                for(unsigned int lag = 0u; lag <= config.feature_lag_count; lag++) {
                    double neighbor_sum = 0.0;
                    double neighbor_max = 0.0;
                    double neighbor_active_count = 0.0;
                    unsigned int neighbor_count = 0u;
                    double east_sum = 0.0;
                    double west_sum = 0.0;
                    double south_sum = 0.0;
                    double north_sum = 0.0;
                    double east_weight = 0.0;
                    double west_weight = 0.0;
                    double south_weight = 0.0;
                    double north_weight = 0.0;
                    const int tile_x = static_cast<int>(tile_id % config.tile_grid_side);
                    const int tile_y = static_cast<int>(tile_id / config.tile_grid_side);
                    for(unsigned int other_tile = 0; other_tile < tile_count; other_tile++) {
                        const unsigned int distance = manhattanDistance(other_tile, tile_id);
                        if(distance == 0u || distance > config.feature_context_radius_tiles) {
                            continue;
                        }
                        const double neighbor_state =
                            inputStateAtLag(repeat_index, frame_index, other_tile, lag);
                        neighbor_sum += neighbor_state;
                        neighbor_max = std::max(neighbor_max, neighbor_state);
                        neighbor_active_count += (neighbor_state > 0.0) ? 1.0 : 0.0;
                        neighbor_count++;
                        const int other_x = static_cast<int>(other_tile % config.tile_grid_side);
                        const int other_y = static_cast<int>(other_tile / config.tile_grid_side);
                        const int dx = other_x - tile_x;
                        const int dy = other_y - tile_y;
                        const double inv_distance = 1.0 / static_cast<double>(std::max(1u, distance));
                        const double east_component = std::max(0, dx) * inv_distance;
                        const double west_component = std::max(0, -dx) * inv_distance;
                        const double south_component = std::max(0, dy) * inv_distance;
                        const double north_component = std::max(0, -dy) * inv_distance;
                        east_sum += neighbor_state * east_component;
                        west_sum += neighbor_state * west_component;
                        south_sum += neighbor_state * south_component;
                        north_sum += neighbor_state * north_component;
                        east_weight += east_component;
                        west_weight += west_component;
                        south_weight += south_component;
                        north_weight += north_component;
                    }
                    const double neighbor_mean = (neighbor_count > 0u)
                        ? (neighbor_sum / static_cast<double>(neighbor_count))
                        : 0.0;
                    const double neighbor_active_fraction = (neighbor_count > 0u)
                        ? (neighbor_active_count / static_cast<double>(neighbor_count))
                        : 0.0;
                    const double east_mean = east_weight > 0.0 ? (east_sum / east_weight) : 0.0;
                    const double west_mean = west_weight > 0.0 ? (west_sum / west_weight) : 0.0;
                    const double south_mean = south_weight > 0.0 ? (south_sum / south_weight) : 0.0;
                    const double north_mean = north_weight > 0.0 ? (north_sum / north_weight) : 0.0;
                    feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                        neighbor_mean;
                    feature_offset++;
                    feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                        neighbor_max;
                    feature_offset++;
                    feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                        neighbor_active_fraction;
                    feature_offset++;
                    if(hvaPredictorDirectionalContextActive(config)) {
                        // Directional L23E-only context: causal neighbor pools and
                        // signed gradients help the top-k head infer local motion/
                        // displacement without future frames or non-L23 inputs.
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            east_mean;
                        feature_offset++;
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            west_mean;
                        feature_offset++;
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            south_mean;
                        feature_offset++;
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            north_mean;
                        feature_offset++;
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            east_mean - west_mean;
                        feature_offset++;
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            south_mean - north_mean;
                        feature_offset++;
                    }
                }
                if(sequence_state_active) {
                    for(unsigned int state_dim = 0; state_dim < config.sequence_state_dim; state_dim++) {
                        feature_series[featureIndex(sample_index, tile_id, feature_offset)] =
                            sequence_state_series[sequenceStateIndex(sample_index, tile_id, state_dim)];
                        feature_offset++;
                    }
                }
                if(feature_offset != feature_channel_count) {
                    throw std::runtime_error("HVA predictor feature layout mismatch.");
                }
            }
        }
    }
    const auto eventWindowTargetState = [&](unsigned int target_channel,
                                            unsigned int repeat_index,
                                            unsigned int start_frame,
                                            unsigned int tile_id) {
        const unsigned int end_frame = std::min(
            video_config.effective_frame_count,
            start_frame + config.event_window_frames);
        double window_max = 0.0;
        for(unsigned int frame = start_frame; frame < end_frame; frame++) {
            const std::size_t sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame;
            window_max = std::max(
                window_max,
                normalizedRate(target_tile_rates[target_channel][(sample_index * tile_count) + tile_id]));
        }
        return window_max;
    };
    const auto topKWindowTargetValues = [&](unsigned int repeat_index,
                                            unsigned int start_frame) {
        std::vector<double> values(tile_count, 0.0);
        const unsigned int end_frame = std::min(
            video_config.effective_frame_count,
            start_frame + config.topk_future_window_frames);
        const unsigned int frame_count = end_frame > start_frame ? (end_frame - start_frame) : 0u;
        if(frame_count == 0u) {
            return values;
        }
        for(unsigned int frame = start_frame; frame < end_frame; frame++) {
            const std::size_t sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame;
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                values[tile_id] += normalizedRate(target_tile_rates[0][(sample_index * tile_count) + tile_id]);
            }
        }
        for(double &value : values) {
            value /= static_cast<double>(frame_count);
        }
        return values;
    };
    const auto smoothTopKTargetValues = [&](const std::vector<double> &values) {
        if(config.topk_target_smooth_radius_tiles == 0u) {
            return values;
        }
        // Fixed radius-1 binomial target kernel; this denoises target mass only
        // and is never used as an input feature.
        constexpr double kernel[3][3] = {
            {1.0, 2.0, 1.0},
            {2.0, 4.0, 2.0},
            {1.0, 2.0, 1.0},
        };
        std::vector<double> smoothed(tile_count, 0.0);
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            const int tile_x = static_cast<int>(tile_id % config.tile_grid_side);
            const int tile_y = static_cast<int>(tile_id / config.tile_grid_side);
            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            for(int ky = 0; ky < 3; ky++) {
                const int source_y = tile_y + ky - 1;
                if(source_y < 0 || source_y >= static_cast<int>(config.tile_grid_side)) {
                    continue;
                }
                for(int kx = 0; kx < 3; kx++) {
                    const int source_x = tile_x + kx - 1;
                    if(source_x < 0 || source_x >= static_cast<int>(config.tile_grid_side)) {
                        continue;
                    }
                    const double weight = kernel[ky][kx];
                    const unsigned int source_tile =
                        (static_cast<unsigned int>(source_y) * config.tile_grid_side)
                        + static_cast<unsigned int>(source_x);
                    weighted_sum += weight * values[source_tile];
                    weight_sum += weight;
                }
            }
            smoothed[tile_id] = weight_sum > 0.0 ? (weighted_sum / weight_sum) : values[tile_id];
        }
        return smoothed;
    };
    const auto topKTargetMask = [&](const std::vector<double> &values) {
        std::vector<bool> mask(tile_count, false);
        std::vector<std::pair<double, unsigned int>> ranked;
        ranked.reserve(tile_count);
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            ranked.push_back({values[tile_id], tile_id});
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            if(a.first == b.first) {
                return a.second < b.second;
            }
            return a.first > b.first;
        });
        const unsigned int positive_count = std::min(config.topk_k, tile_count);
        for(unsigned int rank = 0; rank < positive_count; rank++) {
            mask[ranked[rank].second] = true;
        }
        return mask;
    };
    const auto topKTargetDistribution = [&](const std::vector<double> &values,
                                            const std::vector<double> *frequency_balance) {
        std::vector<double> distribution(tile_count, 0.0);
        const std::vector<bool> mask = topKTargetMask(values);
        double positive_sum = 0.0;
        unsigned int positive_count = 0u;
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            if(mask[tile_id]) {
                double target_mass = std::max(0.0, values[tile_id]);
                if(config.topk_frequency_balance_enabled && frequency_balance != nullptr) {
                    target_mass /= std::sqrt(std::max(
                        config.topk_frequency_balance_floor,
                        (*frequency_balance)[tile_id]));
                }
                positive_sum += target_mass;
                positive_count++;
            }
        }
        if(positive_count == 0u) {
            return distribution;
        }
        if(positive_sum <= 0.0 || !std::isfinite(positive_sum)) {
            const double uniform_mass = 1.0 / static_cast<double>(positive_count);
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                if(mask[tile_id]) {
                    distribution[tile_id] = uniform_mass;
                }
            }
            return distribution;
        }
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            if(mask[tile_id]) {
                double target_mass = std::max(0.0, values[tile_id]);
                if(config.topk_frequency_balance_enabled && frequency_balance != nullptr) {
                    target_mass /= std::sqrt(std::max(
                        config.topk_frequency_balance_floor,
                        (*frequency_balance)[tile_id]));
                }
                distribution[tile_id] = target_mass / positive_sum;
            }
        }
        return distribution;
    };
    const auto topKTargetValid = [](const std::vector<double> &values) {
        return !values.empty() && *std::max_element(values.begin(), values.end()) > 0.0;
    };
    const auto softmaxScores = [](const std::vector<double> &scores) {
        std::vector<double> probabilities(scores.size(), 0.0);
        if(scores.empty()) {
            return probabilities;
        }
        const double max_score = *std::max_element(scores.begin(), scores.end());
        double denominator = 0.0;
        for(std::size_t i = 0; i < scores.size(); i++) {
            probabilities[i] = std::exp(clippedValue(scores[i] - max_score, -60.0, 60.0));
            denominator += probabilities[i];
        }
        if(denominator <= 0.0 || !std::isfinite(denominator)) {
            const double uniform = 1.0 / static_cast<double>(scores.size());
            std::fill(probabilities.begin(), probabilities.end(), uniform);
            return probabilities;
        }
        for(double &probability : probabilities) {
            probability /= denominator;
        }
        return probabilities;
    };
    const auto sortedTopKTileIds = [&](const std::vector<double> &scores) {
        std::vector<std::pair<double, unsigned int>> ranked;
        ranked.reserve(scores.size());
        for(unsigned int tile_id = 0; tile_id < scores.size(); tile_id++) {
            ranked.push_back({scores[tile_id], tile_id});
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            if(a.first == b.first) {
                return a.second < b.second;
            }
            return a.first > b.first;
        });
        const unsigned int count = std::min(config.topk_k, static_cast<unsigned int>(ranked.size()));
        std::vector<unsigned int> ids;
        ids.reserve(count);
        for(unsigned int i = 0; i < count; i++) {
            ids.push_back(ranked[i].second);
        }
        return ids;
    };
    struct TopKRankStats {
        std::size_t count = 0u;
        double model_recall = 0.0;
        double persistence_recall = 0.0;
        double train_frequency_recall = 0.0;
        double no_learning_recall = 0.0;
        double time_shuffle_recall = 0.0;
        double spatial_shuffle_recall = 0.0;
        double model_ndcg = 0.0;
        double persistence_ndcg = 0.0;
        double train_frequency_ndcg = 0.0;
        double no_learning_ndcg = 0.0;
        double time_shuffle_ndcg = 0.0;
        double spatial_shuffle_ndcg = 0.0;
        double model_mrr = 0.0;
        double persistence_mrr = 0.0;
        double train_frequency_mrr = 0.0;
        double no_learning_mrr = 0.0;
        double time_shuffle_mrr = 0.0;
        double spatial_shuffle_mrr = 0.0;
    };
    struct TopKWeightedStats {
        std::size_t count = 0u;
        double model_ndcg = 0.0;
        double persistence_ndcg = 0.0;
        double train_frequency_ndcg = 0.0;
        double no_learning_ndcg = 0.0;
        double time_shuffle_ndcg = 0.0;
        double spatial_shuffle_ndcg = 0.0;
        double model_captured_mass = 0.0;
        double persistence_captured_mass = 0.0;
        double train_frequency_captured_mass = 0.0;
        double no_learning_captured_mass = 0.0;
        double time_shuffle_captured_mass = 0.0;
        double spatial_shuffle_captured_mass = 0.0;
    };
    const auto addTopKStats = [&](TopKRankStats &stats,
                                  const std::vector<bool> &target_mask,
                                  const std::vector<double> &model_scores,
                                  const std::vector<double> &persistence_scores,
                                  const std::vector<double> &train_frequency_scores,
                                  const std::vector<double> &no_learning_scores,
                                  const std::vector<double> &time_shuffle_scores,
                                  const std::vector<double> &spatial_shuffle_scores) {
        const unsigned int positive_count =
            static_cast<unsigned int>(std::count(target_mask.begin(), target_mask.end(), true));
        if(positive_count == 0u) {
            return;
        }
        const double ideal_dcg = [&]() {
            double value = 0.0;
            for(unsigned int rank = 0; rank < positive_count; rank++) {
                value += 1.0 / std::log2(static_cast<double>(rank) + 2.0);
            }
            return value;
        }();
        const auto scoreMetrics = [&](const std::vector<double> &scores) {
            const std::vector<unsigned int> ranked_ids = sortedTopKTileIds(scores);
            double hit_count = 0.0;
            double dcg = 0.0;
            double mrr = 0.0;
            for(unsigned int rank = 0; rank < ranked_ids.size(); rank++) {
                const unsigned int tile_id = ranked_ids[rank];
                if(target_mask[tile_id]) {
                    hit_count += 1.0;
                    dcg += 1.0 / std::log2(static_cast<double>(rank) + 2.0);
                    if(mrr == 0.0) {
                        mrr = 1.0 / (static_cast<double>(rank) + 1.0);
                    }
                }
            }
            return std::array<double, 3>{{
                hit_count / static_cast<double>(positive_count),
                ideal_dcg > 0.0 ? (dcg / ideal_dcg) : 0.0,
                mrr,
            }};
        };
        const auto model = scoreMetrics(model_scores);
        const auto persistence = scoreMetrics(persistence_scores);
        const auto train_frequency = scoreMetrics(train_frequency_scores);
        const auto no_learning = scoreMetrics(no_learning_scores);
        const auto time_shuffle = scoreMetrics(time_shuffle_scores);
        const auto spatial_shuffle = scoreMetrics(spatial_shuffle_scores);
        stats.model_recall += model[0];
        stats.persistence_recall += persistence[0];
        stats.train_frequency_recall += train_frequency[0];
        stats.no_learning_recall += no_learning[0];
        stats.time_shuffle_recall += time_shuffle[0];
        stats.spatial_shuffle_recall += spatial_shuffle[0];
        stats.model_ndcg += model[1];
        stats.persistence_ndcg += persistence[1];
        stats.train_frequency_ndcg += train_frequency[1];
        stats.no_learning_ndcg += no_learning[1];
        stats.time_shuffle_ndcg += time_shuffle[1];
        stats.spatial_shuffle_ndcg += spatial_shuffle[1];
        stats.model_mrr += model[2];
        stats.persistence_mrr += persistence[2];
        stats.train_frequency_mrr += train_frequency[2];
        stats.no_learning_mrr += no_learning[2];
        stats.time_shuffle_mrr += time_shuffle[2];
        stats.spatial_shuffle_mrr += spatial_shuffle[2];
        stats.count++;
    };
    const auto addTopKWeightedStats = [&](TopKWeightedStats &stats,
                                          const std::vector<double> &relevance,
                                          const std::vector<double> &model_scores,
                                          const std::vector<double> &persistence_scores,
                                          const std::vector<double> &train_frequency_scores,
                                          const std::vector<double> &no_learning_scores,
                                          const std::vector<double> &time_shuffle_scores,
                                          const std::vector<double> &spatial_shuffle_scores) {
        const unsigned int k = std::min(config.topk_k, tile_count);
        if(k == 0u || relevance.size() != tile_count) {
            return;
        }
        const double relevance_sum =
            std::accumulate(relevance.begin(), relevance.end(), 0.0);
        if(relevance_sum <= 0.0 || !std::isfinite(relevance_sum)) {
            return;
        }
        const std::vector<unsigned int> ideal_ids = sortedTopKTileIds(relevance);
        double ideal_dcg = 0.0;
        double ideal_mass = 0.0;
        for(unsigned int rank = 0; rank < k && rank < ideal_ids.size(); rank++) {
            const double value = std::max(0.0, relevance[ideal_ids[rank]]);
            ideal_dcg += value / std::log2(static_cast<double>(rank) + 2.0);
            ideal_mass += value;
        }
        if(ideal_dcg <= 0.0 || ideal_mass <= 0.0) {
            return;
        }
        const auto scoreMetrics = [&](const std::vector<double> &scores) {
            const std::vector<unsigned int> ranked_ids = sortedTopKTileIds(scores);
            double dcg = 0.0;
            double captured_mass = 0.0;
            for(unsigned int rank = 0; rank < k && rank < ranked_ids.size(); rank++) {
                const double value = std::max(0.0, relevance[ranked_ids[rank]]);
                dcg += value / std::log2(static_cast<double>(rank) + 2.0);
                captured_mass += value;
            }
            return std::array<double, 2>{{
                dcg / ideal_dcg,
                captured_mass / ideal_mass,
            }};
        };
        const auto model = scoreMetrics(model_scores);
        const auto persistence = scoreMetrics(persistence_scores);
        const auto train_frequency = scoreMetrics(train_frequency_scores);
        const auto no_learning = scoreMetrics(no_learning_scores);
        const auto time_shuffle = scoreMetrics(time_shuffle_scores);
        const auto spatial_shuffle = scoreMetrics(spatial_shuffle_scores);
        stats.model_ndcg += model[0];
        stats.persistence_ndcg += persistence[0];
        stats.train_frequency_ndcg += train_frequency[0];
        stats.no_learning_ndcg += no_learning[0];
        stats.time_shuffle_ndcg += time_shuffle[0];
        stats.spatial_shuffle_ndcg += spatial_shuffle[0];
        stats.model_captured_mass += model[1];
        stats.persistence_captured_mass += persistence[1];
        stats.train_frequency_captured_mass += train_frequency[1];
        stats.no_learning_captured_mass += no_learning[1];
        stats.time_shuffle_captured_mass += time_shuffle[1];
        stats.spatial_shuffle_captured_mass += spatial_shuffle[1];
        stats.count++;
    };

    const double site_count_sum_before = sumValues(l23e_site_spike_counts);
    const double site_count_sum_sq_before = sumSquares(l23e_site_spike_counts);
    const double tile_rate_sum_before = sumValues(input_tile_rates);
    const double tile_rate_sum_sq_before = sumSquares(input_tile_rates);
    const std::uint32_t site_count_fingerprint_before =
        quantizedVectorFingerprint32(l23e_site_spike_counts);
    const std::uint32_t tile_rate_fingerprint_before =
        quantizedVectorFingerprint32(input_tile_rates);

    const unsigned int future_target_horizon_frames =
        std::max(config.event_window_frames, config.topk_future_window_frames);
    const unsigned int heldout_start_frame = hvaPredictorHeldoutStartFrame(video_config, config);
    const unsigned int train_frame_count = heldout_start_frame;
    const auto predictionSplit = [&](unsigned int frame_index, unsigned int target_frame_index) {
        const unsigned int target_window_end_frame = std::min(
            video_config.effective_frame_count,
            target_frame_index + future_target_horizon_frames);
        if(target_frame_index < heldout_start_frame && target_window_end_frame <= heldout_start_frame) {
            return std::string("train");
        }
        if(frame_index >= heldout_start_frame) {
            return std::string("heldout");
        }
        return std::string("boundary_gap");
    };

    std::vector<double> feature_train_mean(feature_channel_count, 0.0);
    std::vector<double> feature_train_sq(feature_channel_count, 0.0);
    std::size_t feature_train_observation_count = 0u;
    for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
        for(unsigned int frame_index = 0; frame_index + config.delay_frames < video_config.effective_frame_count; frame_index++) {
            const unsigned int target_frame_index = frame_index + config.delay_frames;
            if(predictionSplit(frame_index, target_frame_index) != "train") {
                continue;
            }
            const std::size_t sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame_index;
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                    const double value = feature_series[featureIndex(sample_index, tile_id, feature)];
                    feature_train_mean[feature] += value;
                    feature_train_sq[feature] += value * value;
                }
                feature_train_observation_count++;
            }
        }
    }
    if(feature_train_observation_count == 0u) {
        throw std::runtime_error("HVA predictor feature standardization found no train observations.");
    }
    std::vector<double> feature_train_std(feature_channel_count, 1.0);
    unsigned int feature_std_floor_count = 0u;
    for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
        feature_train_mean[feature] /= static_cast<double>(feature_train_observation_count);
        const double mean_sq =
            feature_train_sq[feature] / static_cast<double>(feature_train_observation_count);
        const double variance = std::max(
            0.0,
            mean_sq - (feature_train_mean[feature] * feature_train_mean[feature]));
        const double raw_std = std::sqrt(variance);
        if(raw_std < kHVAPredictorFeatureStdFloor) {
            feature_train_std[feature] = 1.0;
            feature_std_floor_count++;
        }
        else {
            feature_train_std[feature] = raw_std;
        }
    }

    const std::size_t channel_tile_count =
        static_cast<std::size_t>(target_channel_count) * tile_count;
    std::vector<double> train_mean_target(channel_tile_count, 0.0);
    std::vector<double> train_residual_mean(channel_tile_count, 0.0);
    std::vector<double> train_residual_sq(channel_tile_count, 0.0);
    std::vector<unsigned int> train_target_count(channel_tile_count, 0u);
    std::vector<std::vector<double>> train_target_values(channel_tile_count);
    std::size_t train_prediction_count = 0u;
    std::size_t heldout_prediction_count = 0u;
    std::size_t boundary_gap_prediction_count = 0u;
    for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
        for(unsigned int frame_index = 0; frame_index + config.delay_frames < video_config.effective_frame_count; frame_index++) {
            const unsigned int target_frame_index = frame_index + config.delay_frames;
            const std::string split = predictionSplit(frame_index, target_frame_index);
            if(split == "boundary_gap") {
                boundary_gap_prediction_count += tile_count * target_channel_count;
                continue;
            }
            if(split == "heldout") {
                heldout_prediction_count += tile_count * target_channel_count;
                continue;
            }
            train_prediction_count += tile_count * target_channel_count;
            const std::size_t target_sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + target_frame_index;
            const std::size_t current_sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame_index;
            for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
                for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                    const std::size_t stats_index = targetTileIndex(target_channel, tile_id);
                    const double current =
                        normalizedRate(target_tile_rates[target_channel][(current_sample_index * tile_count) + tile_id]);
                    const double target =
                        normalizedRate(target_tile_rates[target_channel][(target_sample_index * tile_count) + tile_id]);
                    const double event_window_target =
                        eventWindowTargetState(target_channel, repeat_index, target_frame_index, tile_id);
                    const double residual = target - current;
                    train_mean_target[stats_index] += target;
                    train_residual_mean[stats_index] += residual;
                    train_residual_sq[stats_index] += residual * residual;
                    train_target_values[stats_index].push_back(event_window_target);
                    train_target_count[stats_index]++;
                }
            }
        }
    }
    if(train_prediction_count == 0u || heldout_prediction_count == 0u) {
        throw std::runtime_error("HVA predictor split produced empty train or held-out prediction set.");
    }
    std::vector<double> train_repeat_avg_topk_target_values(
        static_cast<std::size_t>(video_config.effective_frame_count) * tile_count,
        0.0);
    std::vector<unsigned int> train_repeat_avg_topk_target_counts(
        video_config.effective_frame_count,
        0u);
    if(config.topk_repeat_avg_target_enabled) {
        for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
            for(unsigned int frame_index = 0; frame_index + config.delay_frames < video_config.effective_frame_count; frame_index++) {
                const unsigned int target_frame_index = frame_index + config.delay_frames;
                if(predictionSplit(frame_index, target_frame_index) != "train") {
                    continue;
                }
                const std::vector<double> topk_target_values =
                    topKWindowTargetValues(repeat_index, target_frame_index);
                if(!topKTargetValid(topk_target_values)) {
                    continue;
                }
                for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                    train_repeat_avg_topk_target_values[
                        (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id]
                        += topk_target_values[tile_id];
                }
                train_repeat_avg_topk_target_counts[target_frame_index]++;
            }
        }
        for(unsigned int target_frame_index = 0; target_frame_index < video_config.effective_frame_count; target_frame_index++) {
            const unsigned int repeat_count = train_repeat_avg_topk_target_counts[target_frame_index];
            if(repeat_count == 0u) {
                continue;
            }
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                train_repeat_avg_topk_target_values[
                    (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id]
                    /= static_cast<double>(repeat_count);
            }
        }
        if(config.topk_target_smooth_radius_tiles > 0u) {
            for(unsigned int target_frame_index = 0; target_frame_index < video_config.effective_frame_count; target_frame_index++) {
                if(train_repeat_avg_topk_target_counts[target_frame_index] == 0u) {
                    continue;
                }
                std::vector<double> values(tile_count, 0.0);
                for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                    values[tile_id] = train_repeat_avg_topk_target_values[
                        (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id];
                }
                const std::vector<double> smoothed = smoothTopKTargetValues(values);
                for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                    train_repeat_avg_topk_target_values[
                        (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id] =
                        smoothed[tile_id];
                }
            }
        }
    }
    std::vector<double> eval_repeat_avg_topk_target_values(
        static_cast<std::size_t>(video_config.effective_frame_count) * tile_count,
        0.0);
    std::vector<unsigned int> eval_repeat_avg_topk_target_counts(
        video_config.effective_frame_count,
        0u);
    for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
        for(unsigned int frame_index = 0; frame_index + config.delay_frames < video_config.effective_frame_count; frame_index++) {
            const unsigned int target_frame_index = frame_index + config.delay_frames;
            if(predictionSplit(frame_index, target_frame_index) == "boundary_gap") {
                continue;
            }
            const std::vector<double> topk_target_values =
                topKWindowTargetValues(repeat_index, target_frame_index);
            if(!topKTargetValid(topk_target_values)) {
                continue;
            }
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                eval_repeat_avg_topk_target_values[
                    (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id]
                    += topk_target_values[tile_id];
            }
            eval_repeat_avg_topk_target_counts[target_frame_index]++;
        }
    }
    std::vector<double> eval_repeat_avg_smooth_topk_target_values =
        eval_repeat_avg_topk_target_values;
    for(unsigned int target_frame_index = 0; target_frame_index < video_config.effective_frame_count; target_frame_index++) {
        const unsigned int repeat_count = eval_repeat_avg_topk_target_counts[target_frame_index];
        if(repeat_count == 0u) {
            continue;
        }
        std::vector<double> values(tile_count, 0.0);
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            const std::size_t index =
                (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id;
            eval_repeat_avg_topk_target_values[index] /= static_cast<double>(repeat_count);
            values[tile_id] = eval_repeat_avg_topk_target_values[index];
        }
        const std::vector<double> smoothed = smoothTopKTargetValues(values);
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            eval_repeat_avg_smooth_topk_target_values[
                (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id] =
                smoothed[tile_id];
        }
    }
    const auto topKTrainingTargetValues = [&](unsigned int repeat_index,
                                              unsigned int target_frame_index) {
        if(config.topk_repeat_avg_target_enabled
           && target_frame_index < train_repeat_avg_topk_target_counts.size()
           && train_repeat_avg_topk_target_counts[target_frame_index] > 0u) {
            std::vector<double> values(tile_count, 0.0);
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                values[tile_id] = train_repeat_avg_topk_target_values[
                    (static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id];
            }
            return values;
        }
        return topKWindowTargetValues(repeat_index, target_frame_index);
    };
    const auto topKRepeatAvgEvalTargetValues = [&](unsigned int target_frame_index,
                                                   bool smoothed) {
        std::vector<double> values(tile_count, 0.0);
        if(target_frame_index >= eval_repeat_avg_topk_target_counts.size()
           || eval_repeat_avg_topk_target_counts[target_frame_index] == 0u) {
            return values;
        }
        const std::vector<double> &source = smoothed
            ? eval_repeat_avg_smooth_topk_target_values
            : eval_repeat_avg_topk_target_values;
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            values[tile_id] =
                source[(static_cast<std::size_t>(target_frame_index) * tile_count) + tile_id];
        }
        return values;
    };
    unsigned int train_repeat_avg_topk_target_frame_count = 0u;
    unsigned int train_repeat_avg_topk_target_sample_count = 0u;
    for(unsigned int target_frame_index = 0; target_frame_index < video_config.effective_frame_count; target_frame_index++) {
        if(train_repeat_avg_topk_target_counts[target_frame_index] > 0u) {
            train_repeat_avg_topk_target_frame_count++;
            train_repeat_avg_topk_target_sample_count += train_repeat_avg_topk_target_counts[target_frame_index];
        }
    }
    std::vector<unsigned int> train_topk_positive_count(tile_count, 0u);
    unsigned int train_topk_valid_sample_count = 0u;
    for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
        for(unsigned int frame_index = 0; frame_index + config.delay_frames < video_config.effective_frame_count; frame_index++) {
            const unsigned int target_frame_index = frame_index + config.delay_frames;
            if(predictionSplit(frame_index, target_frame_index) != "train") {
                continue;
            }
            const std::vector<double> topk_target_values =
                topKTrainingTargetValues(repeat_index, target_frame_index);
            if(!topKTargetValid(topk_target_values)) {
                continue;
            }
            const std::vector<bool> topk_mask = topKTargetMask(topk_target_values);
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                if(topk_mask[tile_id]) {
                    train_topk_positive_count[tile_id]++;
                }
            }
            train_topk_valid_sample_count++;
        }
    }
    std::vector<double> train_topk_frequency(tile_count, 1.0 / static_cast<double>(tile_count));
    if(train_topk_valid_sample_count > 0u) {
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            train_topk_frequency[tile_id] =
                static_cast<double>(train_topk_positive_count[tile_id])
                / static_cast<double>(train_topk_valid_sample_count);
        }
    }
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            const std::size_t stats_index = targetTileIndex(target_channel, tile_id);
            if(train_target_count[stats_index] == 0u) {
                throw std::runtime_error("HVA predictor train split produced an empty tile target set.");
            }
            train_mean_target[stats_index] /= static_cast<double>(train_target_count[stats_index]);
            train_residual_mean[stats_index] /= static_cast<double>(train_target_count[stats_index]);
        }
    }
    std::vector<double> train_residual_std(channel_tile_count, 1.0);
    std::vector<double> event_threshold_norm(channel_tile_count, 0.0);
    std::vector<double> train_event_rate(channel_tile_count, 0.0);
    std::vector<double> clipped_train_event_rate(channel_tile_count, 0.0);
    std::vector<unsigned int> train_event_positive_count(channel_tile_count, 0u);
    std::vector<unsigned int> train_event_negative_count(channel_tile_count, 0u);
    std::vector<unsigned int> heldout_event_count(channel_tile_count, 0u);
    std::vector<unsigned int> heldout_event_positive_count(channel_tile_count, 0u);
    std::vector<bool> event_tile_selected(channel_tile_count, false);
    const double event_threshold_min_norm =
        clippedValue(config.event_threshold_min_hz / config.rate_scale_hz, 0.0, 1.0);
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            const std::size_t stats_index = targetTileIndex(target_channel, tile_id);
            const double mean = train_residual_mean[stats_index];
            const double mean_sq =
                train_residual_sq[stats_index] / static_cast<double>(train_target_count[stats_index]);
            train_residual_std[stats_index] =
                std::max(1.0e-3, std::sqrt(std::max(0.0, mean_sq - (mean * mean))));
            event_threshold_norm[stats_index] = std::max(
                event_threshold_min_norm,
                quantileValue(train_target_values[stats_index], config.event_threshold_quantile));
            for(double target_value : train_target_values[stats_index]) {
                if(target_value >= event_threshold_norm[stats_index]) {
                    train_event_positive_count[stats_index]++;
                }
                else {
                    train_event_negative_count[stats_index]++;
                }
            }
            train_event_rate[stats_index] =
                static_cast<double>(train_event_positive_count[stats_index])
                / static_cast<double>(train_target_count[stats_index]);
            clipped_train_event_rate[stats_index] = clippedValue(
                train_event_rate[stats_index],
                kHVAPredictorEventRateFloor,
                1.0 - kHVAPredictorEventRateFloor);
            event_tile_selected[stats_index] =
                train_event_positive_count[stats_index] >= config.event_min_train_positive_count
                && train_event_negative_count[stats_index] >= config.event_min_train_positive_count;
        }
    }

    struct SplitStats {
        std::size_t count = 0u;
        double model_sq = 0.0;
        double residual_z_model_sq = 0.0;
        double persistence_sq = 0.0;
        double train_mean_sq = 0.0;
        double no_learning_sq = 0.0;
        double time_shuffle_sq = 0.0;
        double spatial_shuffle_sq = 0.0;
        double target_rate_sum = 0.0;
        double prediction_rate_sum = 0.0;
        double prediction_min = std::numeric_limits<double>::infinity();
        double prediction_max = -std::numeric_limits<double>::infinity();
        std::vector<double> targets;
        std::vector<double> predictions;

        void add(double target,
                 double prediction,
                 double target_residual_z,
                 double predicted_residual_z,
                 double persistence,
                 double train_mean,
                 double no_learning,
                 double temporal_block_shift_prediction,
                 double spatial_tile_shuffle_prediction,
                 double rate_scale_hz)
        {
            const double model_error = target - prediction;
            const double residual_z_error = target_residual_z - predicted_residual_z;
            const double persistence_error = target - persistence;
            const double train_mean_error = target - train_mean;
            const double no_learning_error = target - no_learning;
            const double time_shuffle_error = target - temporal_block_shift_prediction;
            const double spatial_shuffle_error = target - spatial_tile_shuffle_prediction;
            model_sq += model_error * model_error;
            residual_z_model_sq += residual_z_error * residual_z_error;
            persistence_sq += persistence_error * persistence_error;
            train_mean_sq += train_mean_error * train_mean_error;
            no_learning_sq += no_learning_error * no_learning_error;
            time_shuffle_sq += time_shuffle_error * time_shuffle_error;
            spatial_shuffle_sq += spatial_shuffle_error * spatial_shuffle_error;
            target_rate_sum += target * rate_scale_hz;
            prediction_rate_sum += prediction * rate_scale_hz;
            prediction_min = std::min(prediction_min, prediction);
            prediction_max = std::max(prediction_max, prediction);
            targets.push_back(target);
            predictions.push_back(prediction);
            count++;
        }
    };

    struct EventStats {
        std::size_t count = 0u;
        std::size_t positive_count = 0u;
        double model_brier = 0.0;
        double persistence_brier = 0.0;
        double train_mean_brier = 0.0;
        double no_learning_brier = 0.0;
        double time_shuffle_brier = 0.0;
        double spatial_shuffle_brier = 0.0;
        double model_logloss = 0.0;
        double persistence_logloss = 0.0;
        double train_mean_logloss = 0.0;
        double no_learning_logloss = 0.0;
        double time_shuffle_logloss = 0.0;
        double spatial_shuffle_logloss = 0.0;
        double target_sum = 0.0;
        double prediction_sum = 0.0;
        std::vector<double> targets;
        std::vector<double> predictions;
        std::vector<double> persistence_predictions;
        std::vector<double> train_mean_predictions;
        std::vector<double> no_learning_predictions;
        std::vector<double> time_shuffle_predictions;
        std::vector<double> spatial_shuffle_predictions;

        void add(unsigned int target_event,
                 double predicted_probability,
                 double persistence_probability,
                 double train_mean_probability,
                 double no_learning_probability,
                 double temporal_block_shift_probability,
                 double spatial_tile_shuffle_probability)
        {
            const auto logLoss = [](double target, double probability) {
                const double p = clippedValue(probability, 1.0e-6, 1.0 - 1.0e-6);
                return -((target * std::log(p)) + ((1.0 - target) * std::log(1.0 - p)));
            };
            const double target = static_cast<double>(target_event);
            const double model_error = target - predicted_probability;
            const double persistence_error = target - persistence_probability;
            const double train_mean_error = target - train_mean_probability;
            const double no_learning_error = target - no_learning_probability;
            const double time_shuffle_error = target - temporal_block_shift_probability;
            const double spatial_shuffle_error = target - spatial_tile_shuffle_probability;
            model_brier += model_error * model_error;
            persistence_brier += persistence_error * persistence_error;
            train_mean_brier += train_mean_error * train_mean_error;
            no_learning_brier += no_learning_error * no_learning_error;
            time_shuffle_brier += time_shuffle_error * time_shuffle_error;
            spatial_shuffle_brier += spatial_shuffle_error * spatial_shuffle_error;
            model_logloss += logLoss(target, predicted_probability);
            persistence_logloss += logLoss(target, persistence_probability);
            train_mean_logloss += logLoss(target, train_mean_probability);
            no_learning_logloss += logLoss(target, no_learning_probability);
            time_shuffle_logloss += logLoss(target, temporal_block_shift_probability);
            spatial_shuffle_logloss += logLoss(target, spatial_tile_shuffle_probability);
            target_sum += target;
            prediction_sum += predicted_probability;
            targets.push_back(target);
            predictions.push_back(predicted_probability);
            persistence_predictions.push_back(persistence_probability);
            train_mean_predictions.push_back(train_mean_probability);
            no_learning_predictions.push_back(no_learning_probability);
            time_shuffle_predictions.push_back(temporal_block_shift_probability);
            spatial_shuffle_predictions.push_back(spatial_tile_shuffle_probability);
            positive_count += (target_event != 0u) ? 1u : 0u;
            count++;
        }
    };

    result.weights_before.assign(target_pair_count, 0.0);
    result.weights_after.assign(target_pair_count, 0.0);
    result.readout_weights_before.assign(readout_weight_count, 0.0);
    result.readout_weights_after.assign(readout_weight_count, 0.0);
    result.biases_after.assign(channel_tile_count, 0.0);
    result.event_weights_before.assign(readout_weight_count, 0.0);
    result.event_weights_after.assign(readout_weight_count, 0.0);
    result.event_biases_after.assign(channel_tile_count, 0.0);
    result.topk_weights_before.assign(readout_weight_count, 0.0);
    result.topk_weights_after.assign(readout_weight_count, 0.0);
    result.topk_biases_after.assign(tile_count, 0.0);
    for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
        result.topk_biases_after[tile_id] = logit(clippedValue(
            train_topk_frequency[tile_id],
            kHVAPredictorEventRateFloor,
            1.0 - kHVAPredictorEventRateFloor));
    }
    for(std::size_t state_index = 0; state_index < channel_tile_count; state_index++) {
        result.event_biases_after[state_index] = logit(clipped_train_event_rate[state_index]);
    }
    result.rates.reserve(sample_count * static_cast<std::size_t>(tile_count));
    result.predictions.reserve(
        static_cast<std::size_t>(video_config.repeat_count)
        * static_cast<std::size_t>(video_config.effective_frame_count - config.delay_frames)
        * static_cast<std::size_t>(tile_count)
        * static_cast<std::size_t>(target_channel_count));

    SplitStats train_stats;
    SplitStats heldout_stats;
    std::vector<SplitStats> train_stats_by_channel(target_channel_count);
    std::vector<SplitStats> heldout_stats_by_channel(target_channel_count);
    EventStats train_event_stats_all;
    EventStats heldout_event_stats_all;
    EventStats train_event_stats_selected;
    EventStats heldout_event_stats_selected;
    EventStats train_single_frame_event_stats_all;
    EventStats heldout_single_frame_event_stats_all;
    EventStats train_single_frame_event_stats_selected;
    EventStats heldout_single_frame_event_stats_selected;
    TopKRankStats train_topk_stats;
    TopKRankStats heldout_topk_stats;
    TopKRankStats train_topk_repeat_avg_stats;
    TopKRankStats heldout_topk_repeat_avg_stats;
    TopKRankStats train_topk_repeat_avg_smooth_stats;
    TopKRankStats heldout_topk_repeat_avg_smooth_stats;
    TopKWeightedStats train_topk_repeat_avg_smooth_weighted_stats;
    TopKWeightedStats heldout_topk_repeat_avg_smooth_weighted_stats;
    std::vector<std::vector<double>> targets_by_channel_tile(channel_tile_count);
    std::vector<std::vector<double>> predictions_by_channel_tile(channel_tile_count);

    const auto rawFeatureValue = [&](std::size_t sample_index, unsigned int tile_id, unsigned int feature) {
        return feature_series[featureIndex(sample_index, tile_id, feature)];
    };
    const auto featureValue = [&](std::size_t sample_index, unsigned int tile_id, unsigned int feature) {
        return (rawFeatureValue(sample_index, tile_id, feature) - feature_train_mean[feature])
            / feature_train_std[feature];
    };
    const auto topKScore = [&](std::size_t sample_index, unsigned int post_tile) {
        double score = result.topk_biases_after[post_tile];
        for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
            if(!topKReadoutEnabled(pre_tile, post_tile)) {
                continue;
            }
            for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                score += result.topk_weights_after[readoutIndex(0u, post_tile, pre_tile, feature)]
                    * featureValue(sample_index, pre_tile, feature);
            }
        }
        return score;
    };
    const auto topKScores = [&](std::size_t sample_index) {
        std::vector<double> scores(tile_count, 0.0);
        for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
            scores[post_tile] = topKScore(sample_index, post_tile);
        }
        return scores;
    };
    for(unsigned int epoch = 0; epoch < config.training_epochs; epoch++) {
        for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
            for(unsigned int frame_index = 0; frame_index + config.delay_frames < video_config.effective_frame_count; frame_index++) {
                const unsigned int target_frame_index = frame_index + config.delay_frames;
                if(predictionSplit(frame_index, target_frame_index) != "train") {
                    continue;
                }
                const std::size_t sample_index =
                    (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame_index;
                const std::size_t target_sample_index =
                    (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + target_frame_index;
                const std::vector<double> topk_target_values =
                    topKTrainingTargetValues(repeat_index, target_frame_index);
                if(topKTargetValid(topk_target_values)) {
                    const std::vector<double> topk_target_distribution =
                        topKTargetDistribution(topk_target_values, &train_topk_frequency);
                    const std::vector<double> topk_scores = topKScores(sample_index);
                    const std::vector<double> topk_probabilities = softmaxScores(topk_scores);
                    for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
                        const double topk_error =
                            topk_target_distribution[post_tile] - topk_probabilities[post_tile];
                        result.topk_biases_after[post_tile] += config.topk_learning_rate * topk_error;
                        result.topk_biases_after[post_tile] =
                            clippedValue(result.topk_biases_after[post_tile], -config.weight_clip, config.weight_clip);
                        for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                            if(!topKReadoutEnabled(pre_tile, post_tile)) {
                                continue;
                            }
                            for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                                double &weight = result.topk_weights_after[
                                    readoutIndex(0u, post_tile, pre_tile, feature)];
                                weight =
                                    (weight * (1.0 - config.topk_weight_decay))
                                    + (config.topk_learning_rate
                                       * topk_error
                                       * featureValue(sample_index, pre_tile, feature));
                                weight = clippedValue(weight, -config.weight_clip, config.weight_clip);
                            }
                        }
                    }
                }
                for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
                    for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
                        const std::size_t state_index = targetTileIndex(target_channel, post_tile);
                        const double current_target_state =
                            normalizedRate(target_tile_rates[target_channel][(sample_index * tile_count) + post_tile]);
                        const double target_state =
                            normalizedRate(target_tile_rates[target_channel][(target_sample_index * tile_count) + post_tile]);
                        const double event_window_target =
                            eventWindowTargetState(target_channel, repeat_index, target_frame_index, post_tile);
                        const unsigned int target_event =
                            (event_window_target >= event_threshold_norm[state_index]) ? 1u : 0u;

                        double residual_norm_prediction = result.biases_after[state_index];
                        double event_logit = result.event_biases_after[state_index];
                        for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                            if(!localReadoutEnabled(pre_tile, post_tile)) {
                                continue;
                            }
                            for(unsigned int feature = 0; feature < non_sequence_feature_channel_count; feature++) {
                                const double feature_value = featureValue(sample_index, pre_tile, feature);
                                residual_norm_prediction += result.readout_weights_after[
                                    readoutIndex(target_channel, post_tile, pre_tile, feature)]
                                    * feature_value;
                                if(event_tile_selected[state_index]) {
                                    event_logit += result.event_weights_after[
                                        readoutIndex(target_channel, post_tile, pre_tile, feature)]
                                        * config.event_residual_gain
                                        * feature_value;
                                }
                            }
                        }

                        const double residual_error =
                            (target_state - current_target_state) - residual_norm_prediction;
                        const double predicted_event_probability = event_tile_selected[state_index]
                            ? sigmoid(event_logit)
                            : train_event_rate[state_index];
                        const double event_error =
                            static_cast<double>(target_event) - predicted_event_probability;
                        for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                            if(!localReadoutEnabled(pre_tile, post_tile)) {
                                continue;
                            }
                            for(unsigned int feature = 0; feature < non_sequence_feature_channel_count; feature++) {
                                const double feature_value = featureValue(sample_index, pre_tile, feature);
                                double &weight = result.readout_weights_after[
                                    readoutIndex(target_channel, post_tile, pre_tile, feature)];
                                weight =
                                    (weight * (1.0 - config.weight_decay))
                                    + (config.learning_rate * residual_error * feature_value);
                                weight = clippedValue(weight, -config.weight_clip, config.weight_clip);
                                if(event_tile_selected[state_index]) {
                                    double &event_weight = result.event_weights_after[
                                        readoutIndex(target_channel, post_tile, pre_tile, feature)];
                                    event_weight =
                                        (event_weight * (1.0 - config.event_weight_decay))
                                        + (config.event_learning_rate * event_error * config.event_residual_gain * feature_value);
                                    event_weight = clippedValue(event_weight, -config.weight_clip, config.weight_clip);
                                }
                            }
                        }
                        result.biases_after[state_index] += config.bias_learning_rate * residual_error;
                        result.biases_after[state_index] =
                            clippedValue(result.biases_after[state_index], -config.weight_clip, config.weight_clip);
                        if(event_tile_selected[state_index] && config.event_bias_learning_rate > 0.0) {
                            result.event_biases_after[state_index] += config.event_bias_learning_rate * event_error;
                            result.event_biases_after[state_index] =
                                clippedValue(result.event_biases_after[state_index], -config.weight_clip, config.weight_clip);
                        }
                    }
                }
            }
        }
    }

    unsigned int prediction_index = 0u;
    for(unsigned int repeat_index = 0; repeat_index < video_config.repeat_count; repeat_index++) {
        std::vector<double> previous_state(tile_count, 0.0);
        for(unsigned int frame_index = 0; frame_index < video_config.effective_frame_count; frame_index++) {
            const std::size_t sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + frame_index;
            std::vector<double> current_state(tile_count, 0.0);
            std::vector<double> derivative(tile_count, 0.0);
            std::vector<double> features(
                static_cast<std::size_t>(tile_count) * feature_channel_count,
                0.0);
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                    features[(tile_id * feature_channel_count) + feature] =
                        featureValue(sample_index, tile_id, feature);
                }
                current_state[tile_id] = rawFeatureValue(sample_index, tile_id, 0u);
                derivative[tile_id] = rawFeatureValue(sample_index, tile_id, 4u);
            }

            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                result.rates.push_back({
                    static_cast<unsigned int>(sample_index),
                    repeat_index,
                    frame_index,
                    tile_id,
                    tile_id % config.tile_grid_side,
                    tile_id / config.tile_grid_side,
                    input_tile_rates[(sample_index * tile_count) + tile_id],
                    current_state[tile_id],
                    rawFeatureValue(sample_index, tile_id, 2u),
                    rawFeatureValue(sample_index, tile_id, 1u),
                    rawFeatureValue(sample_index, tile_id, 2u),
                    rawFeatureValue(sample_index, tile_id, 3u),
                    derivative[tile_id],
                });
            }

            if(frame_index + config.delay_frames >= video_config.effective_frame_count) {
                previous_state.swap(current_state);
                continue;
            }
            const unsigned int target_frame_index = frame_index + config.delay_frames;
            const std::string split = predictionSplit(frame_index, target_frame_index);
            if(split == "boundary_gap") {
                previous_state.swap(current_state);
                continue;
            }
            const bool heldout = (split == "heldout");
            const std::size_t target_sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + target_frame_index;
            const unsigned int time_shuffle_frame =
                (train_frame_count > 0u) ? (target_frame_index % train_frame_count) : 0u;
            const std::size_t time_shuffle_sample_index =
                (static_cast<std::size_t>(repeat_index) * video_config.effective_frame_count) + time_shuffle_frame;
            std::vector<double> target_state(channel_tile_count, 0.0);
            std::vector<double> current_target_state(channel_tile_count, 0.0);
            std::vector<double> predicted_state(channel_tile_count, 0.0);
            std::vector<double> target_residual_norm(channel_tile_count, 0.0);
            std::vector<double> target_residual_z(channel_tile_count, 0.0);
            std::vector<double> predicted_residual_z(channel_tile_count, 0.0);
            std::vector<double> predicted_residual_norm(channel_tile_count, 0.0);
            std::vector<double> no_learning_state(channel_tile_count, 0.0);
            std::vector<double> temporal_block_shift_prediction(channel_tile_count, 0.0);
            std::vector<double> event_window_target_state(channel_tile_count, 0.0);
            std::vector<unsigned int> target_events(channel_tile_count, 0u);
            std::vector<unsigned int> single_frame_target_events(channel_tile_count, 0u);
            std::vector<double> predicted_event_prob(channel_tile_count, 0.0);
            std::vector<double> persistence_event_prob(channel_tile_count, 0.0);
            std::vector<double> no_learning_event_prob(channel_tile_count, 0.0);
            std::vector<double> temporal_block_shift_event_prob(channel_tile_count, 0.0);
            std::vector<double> spatial_tile_shuffle_event_prob(channel_tile_count, 0.0);
            const std::vector<double> topk_target_values =
                topKWindowTargetValues(repeat_index, target_frame_index);
            const bool topk_sample_valid = topKTargetValid(topk_target_values);
            const std::vector<bool> topk_target_mask = topKTargetMask(topk_target_values);
            const std::vector<double> topk_repeat_avg_target_values =
                topKRepeatAvgEvalTargetValues(target_frame_index, false);
            const bool topk_repeat_avg_sample_valid =
                topKTargetValid(topk_repeat_avg_target_values);
            const std::vector<bool> topk_repeat_avg_target_mask =
                topKTargetMask(topk_repeat_avg_target_values);
            const std::vector<double> topk_repeat_avg_smooth_target_values =
                topKRepeatAvgEvalTargetValues(target_frame_index, true);
            const bool topk_repeat_avg_smooth_sample_valid =
                topKTargetValid(topk_repeat_avg_smooth_target_values);
            const std::vector<bool> topk_repeat_avg_smooth_target_mask =
                topKTargetMask(topk_repeat_avg_smooth_target_values);
            const std::vector<double> topk_model_scores = topKScores(sample_index);
            const std::vector<double> topk_model_probabilities = softmaxScores(topk_model_scores);
            std::vector<double> topk_persistence_scores(tile_count, 0.0);
            std::vector<double> topk_time_shuffle_scores =
                topKWindowTargetValues(repeat_index, time_shuffle_frame);
            std::vector<double> topk_spatial_shuffle_scores(tile_count, 0.0);
            for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
                topk_persistence_scores[tile_id] = current_state[tile_id];
                const unsigned int spatial_shuffle_tile =
                    (tile_id + std::max(1u, tile_count / 2u)) % tile_count;
                topk_spatial_shuffle_scores[tile_id] = topk_target_values[spatial_shuffle_tile];
            }
            if(topk_sample_valid) {
                TopKRankStats &topk_stats = heldout ? heldout_topk_stats : train_topk_stats;
                addTopKStats(
                    topk_stats,
                    topk_target_mask,
                    topk_model_scores,
                    topk_persistence_scores,
                    train_topk_frequency,
                    train_topk_frequency,
                    topk_time_shuffle_scores,
                    topk_spatial_shuffle_scores);
            }
            if(topk_repeat_avg_sample_valid) {
                TopKRankStats &repeat_avg_stats =
                    heldout ? heldout_topk_repeat_avg_stats : train_topk_repeat_avg_stats;
                addTopKStats(
                    repeat_avg_stats,
                    topk_repeat_avg_target_mask,
                    topk_model_scores,
                    topk_persistence_scores,
                    train_topk_frequency,
                    train_topk_frequency,
                    topk_time_shuffle_scores,
                    topk_spatial_shuffle_scores);
            }
            if(topk_repeat_avg_smooth_sample_valid) {
                TopKRankStats &repeat_avg_smooth_stats = heldout
                    ? heldout_topk_repeat_avg_smooth_stats
                    : train_topk_repeat_avg_smooth_stats;
                TopKWeightedStats &repeat_avg_smooth_weighted_stats = heldout
                    ? heldout_topk_repeat_avg_smooth_weighted_stats
                    : train_topk_repeat_avg_smooth_weighted_stats;
                addTopKStats(
                    repeat_avg_smooth_stats,
                    topk_repeat_avg_smooth_target_mask,
                    topk_model_scores,
                    topk_persistence_scores,
                    train_topk_frequency,
                    train_topk_frequency,
                    topk_time_shuffle_scores,
                    topk_spatial_shuffle_scores);
                addTopKWeightedStats(
                    repeat_avg_smooth_weighted_stats,
                    topk_repeat_avg_smooth_target_values,
                    topk_model_scores,
                    topk_persistence_scores,
                    train_topk_frequency,
                    train_topk_frequency,
                    topk_time_shuffle_scores,
                    topk_spatial_shuffle_scores);
            }
            for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
                for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
                    const std::size_t state_index = targetTileIndex(target_channel, post_tile);
                    current_target_state[state_index] =
                        normalizedRate(target_tile_rates[target_channel][(sample_index * tile_count) + post_tile]);
                    target_state[state_index] =
                        normalizedRate(target_tile_rates[target_channel][(target_sample_index * tile_count) + post_tile]);
                    event_window_target_state[state_index] =
                        eventWindowTargetState(target_channel, repeat_index, target_frame_index, post_tile);
                    temporal_block_shift_prediction[state_index] =
                        normalizedRate(target_tile_rates[target_channel][(time_shuffle_sample_index * tile_count) + post_tile]);
                    target_events[state_index] =
                        (event_window_target_state[state_index] >= event_threshold_norm[state_index]) ? 1u : 0u;
                    single_frame_target_events[state_index] =
                        (target_state[state_index] >= event_threshold_norm[state_index]) ? 1u : 0u;
                    persistence_event_prob[state_index] =
                        (current_target_state[state_index] >= event_threshold_norm[state_index]) ? 1.0 : 0.0;
                    no_learning_event_prob[state_index] = train_event_rate[state_index];
                    temporal_block_shift_event_prob[state_index] =
                        (eventWindowTargetState(target_channel, repeat_index, time_shuffle_frame, post_tile)
                         >= event_threshold_norm[state_index]) ? 1.0 : 0.0;
                    target_residual_norm[state_index] =
                        target_state[state_index] - current_target_state[state_index];
                    target_residual_z[state_index] =
                        (target_residual_norm[state_index] - train_residual_mean[state_index])
                        / train_residual_std[state_index];
                    double residual_norm_prediction = result.biases_after[state_index];
                    for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                        if(!localReadoutEnabled(pre_tile, post_tile)) {
                            continue;
                        }
                        for(unsigned int feature = 0; feature < non_sequence_feature_channel_count; feature++) {
                            residual_norm_prediction += result.readout_weights_after[
                                readoutIndex(target_channel, post_tile, pre_tile, feature)]
                                * features[(pre_tile * feature_channel_count) + feature];
                        }
                    }
                    predicted_residual_norm[state_index] = residual_norm_prediction;
                    predicted_residual_z[state_index] =
                        (residual_norm_prediction - train_residual_mean[state_index])
                        / train_residual_std[state_index];
                    predicted_state[state_index] =
                        clippedValue(
                            current_target_state[state_index] + predicted_residual_norm[state_index],
                            0.0,
                            1.0);
                    no_learning_state[state_index] =
                        clippedValue(
                            current_target_state[state_index] + train_residual_mean[state_index],
                            0.0,
                            1.0);
                    if(event_tile_selected[state_index]) {
                        double event_logit = result.event_biases_after[state_index];
                        for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                            if(!localReadoutEnabled(pre_tile, post_tile)) {
                                continue;
                            }
                            for(unsigned int feature = 0; feature < non_sequence_feature_channel_count; feature++) {
                                event_logit += result.event_weights_after[
                                    readoutIndex(target_channel, post_tile, pre_tile, feature)]
                                    * config.event_residual_gain
                                    * features[(pre_tile * feature_channel_count) + feature];
                            }
                        }
                        predicted_event_prob[state_index] = sigmoid(event_logit);
                    }
                    else {
                        predicted_event_prob[state_index] = train_event_rate[state_index];
                    }
                }
            }

            for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
                for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
                    const std::size_t state_index = targetTileIndex(target_channel, post_tile);
                    const double target = target_state[state_index];
                    const double prediction = predicted_state[state_index];
                    const double current = current_target_state[state_index];
                    const double train_mean = train_mean_target[state_index];
                    const double no_learning = no_learning_state[state_index];
                    const double temporal_block_shift = temporal_block_shift_prediction[state_index];
                    const unsigned int spatial_shuffle_tile =
                        (post_tile + std::max(1u, tile_count / 2u)) % tile_count;
                    const std::size_t spatial_shuffle_index =
                        targetTileIndex(target_channel, spatial_shuffle_tile);
                    const double spatial_tile_shuffle = target_state[spatial_shuffle_index];
                    const double spatial_event_shuffle =
                        (event_window_target_state[spatial_shuffle_index] >= event_threshold_norm[spatial_shuffle_index]) ? 1.0 : 0.0;
                    const double spatial_single_frame_event_shuffle =
                        (target_state[spatial_shuffle_index] >= event_threshold_norm[spatial_shuffle_index]) ? 1.0 : 0.0;
                    spatial_tile_shuffle_event_prob[state_index] = spatial_event_shuffle;
                    const double model_error = target - prediction;

                    const double target_rate_hz = target * config.rate_scale_hz;
                    const double predicted_rate_hz = prediction * config.rate_scale_hz;
                    SplitStats &channel_stats = heldout
                        ? heldout_stats_by_channel[target_channel]
                        : train_stats_by_channel[target_channel];
                    channel_stats.add(
                        target,
                        prediction,
                        target_residual_z[state_index],
                        predicted_residual_z[state_index],
                        current,
                        train_mean,
                        no_learning,
                        temporal_block_shift,
                        spatial_tile_shuffle,
                        config.rate_scale_hz);
                    if(target_specs[target_channel].required) {
                        SplitStats &stats = heldout ? heldout_stats : train_stats;
                        stats.add(
                            target,
                            prediction,
                            target_residual_z[state_index],
                            predicted_residual_z[state_index],
                            current,
                            train_mean,
                            no_learning,
                            temporal_block_shift,
                            spatial_tile_shuffle,
                            config.rate_scale_hz);
                    }
                    if(heldout) {
                        targets_by_channel_tile[state_index].push_back(target);
                        predictions_by_channel_tile[state_index].push_back(prediction);
                        heldout_event_count[state_index]++;
                        if(target_events[state_index] != 0u) {
                            heldout_event_positive_count[state_index]++;
                        }
                    }
                    EventStats &event_stats_all = heldout ? heldout_event_stats_all : train_event_stats_all;
                    event_stats_all.add(
                        target_events[state_index],
                        predicted_event_prob[state_index],
                        persistence_event_prob[state_index],
                        train_event_rate[state_index],
                        no_learning_event_prob[state_index],
                        temporal_block_shift_event_prob[state_index],
                        spatial_event_shuffle);
                    if(event_tile_selected[state_index]) {
                        EventStats &event_stats_selected =
                            heldout ? heldout_event_stats_selected : train_event_stats_selected;
                        event_stats_selected.add(
                            target_events[state_index],
                            predicted_event_prob[state_index],
                            persistence_event_prob[state_index],
                            train_event_rate[state_index],
                            no_learning_event_prob[state_index],
                            temporal_block_shift_event_prob[state_index],
                            spatial_event_shuffle);
                    }
                    EventStats &single_frame_event_stats_all =
                        heldout ? heldout_single_frame_event_stats_all : train_single_frame_event_stats_all;
                    single_frame_event_stats_all.add(
                        single_frame_target_events[state_index],
                        predicted_event_prob[state_index],
                        persistence_event_prob[state_index],
                        train_event_rate[state_index],
                        no_learning_event_prob[state_index],
                        temporal_block_shift_event_prob[state_index],
                        spatial_single_frame_event_shuffle);
                    if(event_tile_selected[state_index]) {
                        EventStats &single_frame_event_stats_selected =
                            heldout
                                ? heldout_single_frame_event_stats_selected
                                : train_single_frame_event_stats_selected;
                        single_frame_event_stats_selected.add(
                            single_frame_target_events[state_index],
                            predicted_event_prob[state_index],
                            persistence_event_prob[state_index],
                            train_event_rate[state_index],
                            no_learning_event_prob[state_index],
                            temporal_block_shift_event_prob[state_index],
                            spatial_single_frame_event_shuffle);
                    }

                    result.predictions.push_back({
                        prediction_index++,
                        repeat_index,
                        frame_index,
                        target_frame_index,
                        target_channel,
                        target_specs[target_channel].name,
                        post_tile,
                        post_tile % config.tile_grid_side,
                        post_tile / config.tile_grid_side,
                        split,
                        false,
                        current,
                        target,
                        prediction,
                        target_residual_norm[state_index],
                        predicted_residual_norm[state_index],
                        target_residual_z[state_index],
                        predicted_residual_z[state_index],
                        train_residual_mean[state_index],
                        train_residual_std[state_index],
                        current,
                        train_mean,
                        no_learning,
                        temporal_block_shift,
                        spatial_tile_shuffle,
                        target_rate_hz,
                        predicted_rate_hz,
                        model_error * config.rate_scale_hz,
                        event_window_target_state[state_index],
                        event_threshold_norm[state_index],
                        event_tile_selected[state_index],
                        target_events[state_index],
                        single_frame_target_events[state_index],
                        predicted_event_prob[state_index],
                        persistence_event_prob[state_index],
                        train_event_rate[state_index],
                        no_learning_event_prob[state_index],
                        temporal_block_shift_event_prob[state_index],
                        spatial_tile_shuffle_event_prob[state_index],
                        static_cast<double>(target_events[state_index]) - predicted_event_prob[state_index],
                        topk_target_values[post_tile],
                        topk_target_mask[post_tile],
                        topk_sample_valid,
                        topk_model_scores[post_tile],
                        topk_model_probabilities[post_tile],
                        topk_persistence_scores[post_tile],
                        train_topk_frequency[post_tile],
                        train_topk_frequency[post_tile],
                        topk_time_shuffle_scores[post_tile],
                        topk_spatial_shuffle_scores[post_tile],
                    });
                }
            }
            previous_state.swap(current_state);
        }
    }

    result.event_tiles.reserve(channel_tile_count);
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            const std::size_t state_index = targetTileIndex(target_channel, tile_id);
            const double train_count_for_tile =
                static_cast<double>(train_target_count[state_index]);
            const double heldout_count_for_tile =
                static_cast<double>(heldout_event_count[state_index]);
            result.event_tiles.push_back({
                target_channel,
                target_specs[target_channel].name,
                tile_id,
                tile_id % config.tile_grid_side,
                tile_id / config.tile_grid_side,
                event_threshold_norm[state_index],
                event_threshold_norm[state_index] * config.rate_scale_hz,
                train_target_count[state_index],
                train_event_positive_count[state_index],
                train_event_negative_count[state_index],
                heldout_event_count[state_index],
                heldout_event_positive_count[state_index],
                train_count_for_tile > 0.0
                    ? (static_cast<double>(train_event_positive_count[state_index]) / train_count_for_tile)
                    : 0.0,
                heldout_count_for_tile > 0.0
                    ? (static_cast<double>(heldout_event_positive_count[state_index]) / heldout_count_for_tile)
                    : 0.0,
                event_tile_selected[state_index],
            });
        }
    }

    double active_pair_count = 0.0;
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
            for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                const std::size_t pair = targetPairIndex(target_channel, post_tile, pre_tile);
                double signed_sum = 0.0;
                for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                    signed_sum += result.readout_weights_after[
                        readoutIndex(target_channel, post_tile, pre_tile, feature)];
                }
                result.weights_after[pair] = signed_sum;
                if(localReadoutEnabled(pre_tile, post_tile)) {
                    active_pair_count += 1.0;
                }
            }
        }
    }

    const double prediction_count = static_cast<double>(result.predictions.size());
    const auto mse = [](double sum_sq, std::size_t count) {
        return count > 0u ? (sum_sq / static_cast<double>(count)) : 0.0;
    };
    const auto meanHz = [](double sum_hz, std::size_t count) {
        return count > 0u ? (sum_hz / static_cast<double>(count)) : 0.0;
    };
    double weight_l1 = 0.0;
    double weight_max_abs = 0.0;
    for(double weight : result.readout_weights_after) {
        weight_l1 += std::fabs(weight);
        weight_max_abs = std::max(weight_max_abs, std::fabs(weight));
    }
    double bias_l1 = 0.0;
    for(double bias : result.biases_after) {
        bias_l1 += std::fabs(bias);
    }
    double event_weight_l1 = 0.0;
    double event_weight_max_abs = 0.0;
    for(double weight : result.event_weights_after) {
        event_weight_l1 += std::fabs(weight);
        event_weight_max_abs = std::max(event_weight_max_abs, std::fabs(weight));
    }
    double event_bias_l1 = 0.0;
    for(double bias : result.event_biases_after) {
        event_bias_l1 += std::fabs(bias);
    }
    double topk_weight_l1 = 0.0;
    double topk_weight_max_abs = 0.0;
    for(double weight : result.topk_weights_after) {
        topk_weight_l1 += std::fabs(weight);
        topk_weight_max_abs = std::max(topk_weight_max_abs, std::fabs(weight));
    }
    double topk_bias_l1 = 0.0;
    for(double bias : result.topk_biases_after) {
        topk_bias_l1 += std::fabs(bias);
    }
    std::vector<double> topk_local_abs_weights;
    std::vector<double> topk_distant_abs_weights;
    std::vector<double> topk_diagonal_abs_weights;
    std::vector<double> topk_offdiagonal_abs_weights;
    unsigned int topk_local_nonzero_pair_count = 0u;
    unsigned int topk_distant_nonzero_pair_count = 0u;
    double topk_local_abs_weight_sum = 0.0;
    double topk_distant_abs_weight_sum = 0.0;
    double topk_distant_abs_weight_max = 0.0;
    for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
        const unsigned int post_x = post_tile % config.tile_grid_side;
        const unsigned int post_y = post_tile / config.tile_grid_side;
        for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
            const unsigned int pre_x = pre_tile % config.tile_grid_side;
            const unsigned int pre_y = pre_tile / config.tile_grid_side;
            double abs_weight = 0.0;
            for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                abs_weight += std::fabs(result.topk_weights_after[
                    readoutIndex(0u, post_tile, pre_tile, feature)]);
            }
            const unsigned int manhattan_distance =
                static_cast<unsigned int>(
                    std::abs(static_cast<int>(pre_x) - static_cast<int>(post_x))
                    + std::abs(static_cast<int>(pre_y) - static_cast<int>(post_y)));
            if(pre_tile == post_tile) {
                topk_diagonal_abs_weights.push_back(abs_weight);
            }
            else {
                topk_offdiagonal_abs_weights.push_back(abs_weight);
            }
            if(manhattan_distance <= config.topk_local_radius_tiles) {
                topk_local_abs_weights.push_back(abs_weight);
                topk_local_abs_weight_sum += abs_weight;
                if(abs_weight > 1.0e-12) {
                    topk_local_nonzero_pair_count++;
                }
            }
            else {
                topk_distant_abs_weights.push_back(abs_weight);
                topk_distant_abs_weight_sum += abs_weight;
                topk_distant_abs_weight_max = std::max(topk_distant_abs_weight_max, abs_weight);
                if(abs_weight > 1.0e-12) {
                    topk_distant_nonzero_pair_count++;
                }
            }
        }
    }
    struct FeatureGroupNorms {
        double current = 0.0;
        double trace = 0.0;
        double derivative = 0.0;
        double lag = 0.0;
        double context = 0.0;
        double sequence = 0.0;
        double total = 0.0;
    };
    const unsigned int lag_feature_begin = kHVAPredictorBaseFeatureChannelCount;
    const unsigned int context_feature_begin = lag_feature_begin + config.feature_lag_count;
    const auto addFeatureGroupNorm = [&](FeatureGroupNorms &norms,
                                         unsigned int feature,
                                         double abs_weight) {
        norms.total += abs_weight;
        if(feature == 0u) {
            norms.current += abs_weight;
        }
        else if(feature >= 1u && feature <= kHVAPredictorTraceChannelCount) {
            norms.trace += abs_weight;
        }
        else if(feature == 4u) {
            norms.derivative += abs_weight;
        }
        else if(feature >= lag_feature_begin && feature < context_feature_begin) {
            norms.lag += abs_weight;
        }
        else if(feature >= non_sequence_feature_channel_count) {
            norms.sequence += abs_weight;
        }
        else {
            norms.context += abs_weight;
        }
    };
    const auto computeFeatureGroupNorms = [&](const std::vector<double> &weights) {
        FeatureGroupNorms norms;
        const std::size_t pair_count_for_weights =
            weights.size() / static_cast<std::size_t>(feature_channel_count);
        for(std::size_t pair = 0; pair < pair_count_for_weights; pair++) {
            const std::size_t base = pair * static_cast<std::size_t>(feature_channel_count);
            for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                addFeatureGroupNorm(norms, feature, std::fabs(weights[base + feature]));
            }
        }
        return norms;
    };
    const FeatureGroupNorms residual_feature_norms =
        computeFeatureGroupNorms(result.readout_weights_after);
    const FeatureGroupNorms event_feature_norms =
        computeFeatureGroupNorms(result.event_weights_after);
    const FeatureGroupNorms topk_feature_norms =
        computeFeatureGroupNorms(result.topk_weights_after);

    std::vector<std::vector<double>> tile_correlations_by_channel(target_channel_count);
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        tile_correlations_by_channel[target_channel].reserve(tile_count);
        for(unsigned int tile_id = 0; tile_id < tile_count; tile_id++) {
            const std::size_t state_index = targetTileIndex(target_channel, tile_id);
            tile_correlations_by_channel[target_channel].push_back(responseCorrelation(
                predictions_by_channel_tile[state_index],
                targets_by_channel_tile[state_index]));
        }
    }
    const double mean_corr = meanRate(tile_correlations_by_channel.front());
    std::vector<double> local_abs_weights;
    std::vector<double> distant_abs_weights;
    std::vector<double> diagonal_abs_weights;
    std::vector<double> offdiagonal_abs_weights;
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
            const unsigned int post_x = post_tile % config.tile_grid_side;
            const unsigned int post_y = post_tile / config.tile_grid_side;
            for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                const unsigned int pre_x = pre_tile % config.tile_grid_side;
                const unsigned int pre_y = pre_tile / config.tile_grid_side;
                double abs_weight = 0.0;
                for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                    abs_weight += std::fabs(result.readout_weights_after[
                        readoutIndex(target_channel, post_tile, pre_tile, feature)]);
                }
                const unsigned int manhattan_distance =
                    static_cast<unsigned int>(
                        std::abs(static_cast<int>(pre_x) - static_cast<int>(post_x))
                        + std::abs(static_cast<int>(pre_y) - static_cast<int>(post_y)));
                if(pre_tile == post_tile) {
                    diagonal_abs_weights.push_back(abs_weight);
                }
                else {
                    offdiagonal_abs_weights.push_back(abs_weight);
                }
                if(manhattan_distance <= config.local_radius_tiles) {
                    local_abs_weights.push_back(abs_weight);
                }
                if(manhattan_distance > config.local_radius_tiles) {
                    distant_abs_weights.push_back(abs_weight);
                }
            }
        }
    }
    const double train_model_mse = mse(train_stats.model_sq, train_stats.count);
    const double train_residual_z_mse = mse(train_stats.residual_z_model_sq, train_stats.count);
    const double heldout_model_mse = mse(heldout_stats.model_sq, heldout_stats.count);
    const double heldout_residual_z_mse = mse(heldout_stats.residual_z_model_sq, heldout_stats.count);
    const double heldout_persistence_mse = mse(heldout_stats.persistence_sq, heldout_stats.count);
    const double heldout_train_mean_mse = mse(heldout_stats.train_mean_sq, heldout_stats.count);
    const double heldout_no_learning_mse = mse(heldout_stats.no_learning_sq, heldout_stats.count);
    const double heldout_time_shuffle_mse = mse(heldout_stats.time_shuffle_sq, heldout_stats.count);
    const double heldout_spatial_shuffle_mse = mse(heldout_stats.spatial_shuffle_sq, heldout_stats.count);
    const double site_count_sum_after = sumValues(l23e_site_spike_counts);
    const double site_count_sum_sq_after = sumSquares(l23e_site_spike_counts);
    const double tile_rate_sum_after = sumValues(input_tile_rates);
    const double tile_rate_sum_sq_after = sumSquares(input_tile_rates);
    const std::uint32_t site_count_fingerprint_after =
        quantizedVectorFingerprint32(l23e_site_spike_counts);
    const std::uint32_t tile_rate_fingerprint_after =
        quantizedVectorFingerprint32(input_tile_rates);
    std::vector<double> multitask_target_rates_flat;
    for(const std::vector<double> &rates : target_tile_rates) {
        multitask_target_rates_flat.insert(multitask_target_rates_flat.end(), rates.begin(), rates.end());
    }
    const double multitask_target_sum_before = sumValues(multitask_target_rates_flat);
    const double multitask_target_sum_sq_before = sumSquares(multitask_target_rates_flat);
    const std::uint32_t multitask_target_fingerprint_before =
        quantizedVectorFingerprint32(multitask_target_rates_flat);
    const double multitask_target_sum_after = sumValues(multitask_target_rates_flat);
    const double multitask_target_sum_sq_after = sumSquares(multitask_target_rates_flat);
    const std::uint32_t multitask_target_fingerprint_after =
        quantizedVectorFingerprint32(multitask_target_rates_flat);
    const double local_abs_weight_mean = meanRate(local_abs_weights);
    const double distant_abs_weight_mean = meanRate(distant_abs_weights);
    const double diagonal_abs_weight_mean = meanRate(diagonal_abs_weights);
    const double offdiagonal_abs_weight_mean = meanRate(offdiagonal_abs_weights);
    const double local_distant_abs_weight_ratio =
        distant_abs_weight_mean > 0.0
            ? (local_abs_weight_mean / distant_abs_weight_mean)
            : (local_abs_weight_mean > 0.0 ? 1.0e12 : 0.0);
    const double topk_local_abs_weight_mean = meanRate(topk_local_abs_weights);
    const double topk_distant_abs_weight_mean = meanRate(topk_distant_abs_weights);
    const double topk_diagonal_abs_weight_mean = meanRate(topk_diagonal_abs_weights);
    const double topk_offdiagonal_abs_weight_mean = meanRate(topk_offdiagonal_abs_weights);
    const double topk_active_readout_pair_fraction =
        static_cast<double>(topk_local_abs_weights.size())
        / static_cast<double>(std::max<std::size_t>(1u, pair_count));
    const double train_residual_std_min = *std::min_element(
        train_residual_std.begin(),
        train_residual_std.end());
    const double train_residual_std_median = median(train_residual_std);
    const double event_threshold_min_actual = *std::min_element(
        event_threshold_norm.begin(),
        event_threshold_norm.end());
    const double event_threshold_median = median(event_threshold_norm);
    const double event_threshold_max_actual = *std::max_element(
        event_threshold_norm.begin(),
        event_threshold_norm.end());
    const double event_train_rate_min = *std::min_element(
        train_event_rate.begin(),
        train_event_rate.end());
    const double event_train_rate_median = median(train_event_rate);
    const double event_train_rate_max = *std::max_element(
        train_event_rate.begin(),
        train_event_rate.end());
    std::vector<double> selected_train_event_rates;
    selected_train_event_rates.reserve(channel_tile_count);
    for(std::size_t state_index = 0; state_index < channel_tile_count; state_index++) {
        if(event_tile_selected[state_index]) {
            selected_train_event_rates.push_back(train_event_rate[state_index]);
        }
    }
    const double selected_event_train_rate_median =
        selected_train_event_rates.empty() ? 0.0 : median(selected_train_event_rates);
    const double event_bias_min = *std::min_element(
        result.event_biases_after.begin(),
        result.event_biases_after.end());
    const double event_bias_median = median(result.event_biases_after);
    const double event_bias_max = *std::max_element(
        result.event_biases_after.begin(),
        result.event_biases_after.end());
    const double feature_train_std_min = *std::min_element(
        feature_train_std.begin(),
        feature_train_std.end());
    const double feature_train_std_median = median(feature_train_std);
    const double feature_train_std_max = *std::max_element(
        feature_train_std.begin(),
        feature_train_std.end());
    const unsigned int event_selected_tile_count =
        static_cast<unsigned int>(std::count(event_tile_selected.begin(), event_tile_selected.end(), true));
    const auto eventBrier = [](double sum_sq, std::size_t count) {
        return count > 0u ? (sum_sq / static_cast<double>(count)) : 0.0;
    };
    const auto eventMeanLoss = [](double sum_loss, std::size_t count) {
        return count > 0u ? (sum_loss / static_cast<double>(count)) : 0.0;
    };
    const auto eventPositiveFraction = [](std::size_t positives, std::size_t count) {
        return count > 0u ? (static_cast<double>(positives) / static_cast<double>(count)) : 0.0;
    };
    const auto eventAuc = [](const std::vector<double> &scores, const std::vector<double> &targets) {
        if(scores.size() != targets.size() || scores.empty()) {
            return 0.5;
        }
        std::vector<std::pair<double, double>> ranked;
        ranked.reserve(scores.size());
        double positive_count = 0.0;
        for(std::size_t i = 0; i < scores.size(); i++) {
            ranked.push_back({scores[i], targets[i]});
            positive_count += targets[i] > 0.5 ? 1.0 : 0.0;
        }
        const double negative_count = static_cast<double>(scores.size()) - positive_count;
        if(positive_count <= 0.0 || negative_count <= 0.0) {
            return 0.5;
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });
        double positive_rank_sum = 0.0;
        std::size_t i = 0u;
        while(i < ranked.size()) {
            std::size_t j = i + 1u;
            while(j < ranked.size() && ranked[j].first == ranked[i].first) {
                j++;
            }
            const double average_rank =
                (static_cast<double>(i + 1u) + static_cast<double>(j)) * 0.5;
            for(std::size_t k = i; k < j; k++) {
                if(ranked[k].second > 0.5) {
                    positive_rank_sum += average_rank;
                }
            }
            i = j;
        }
        return (positive_rank_sum - ((positive_count * (positive_count + 1.0)) * 0.5))
            / (positive_count * negative_count);
    };
    const auto eventAuprc = [](const std::vector<double> &scores, const std::vector<double> &targets) {
        if(scores.size() != targets.size() || scores.empty()) {
            return 0.0;
        }
        std::vector<std::pair<double, double>> ranked;
        ranked.reserve(scores.size());
        double positive_count = 0.0;
        for(std::size_t i = 0; i < scores.size(); i++) {
            ranked.push_back({scores[i], targets[i]});
            positive_count += targets[i] > 0.5 ? 1.0 : 0.0;
        }
        if(positive_count <= 0.0) {
            return 0.0;
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        double true_positive = 0.0;
        double false_positive = 0.0;
        double previous_recall = 0.0;
        double area = 0.0;
        for(const auto &row : ranked) {
            if(row.second > 0.5) {
                true_positive += 1.0;
            }
            else {
                false_positive += 1.0;
            }
            const double recall = true_positive / positive_count;
            const double precision = true_positive / std::max(1.0, true_positive + false_positive);
            area += precision * (recall - previous_recall);
            previous_recall = recall;
        }
        return area;
    };
    const auto topKMean = [](double sum, std::size_t count) {
        return count > 0u ? (sum / static_cast<double>(count)) : 0.0;
    };
    const double heldout_topk_model_recall = topKMean(heldout_topk_stats.model_recall, heldout_topk_stats.count);
    const double heldout_topk_persistence_recall = topKMean(heldout_topk_stats.persistence_recall, heldout_topk_stats.count);
    const double heldout_topk_train_frequency_recall = topKMean(heldout_topk_stats.train_frequency_recall, heldout_topk_stats.count);
    const double heldout_topk_no_learning_recall = topKMean(heldout_topk_stats.no_learning_recall, heldout_topk_stats.count);
    const double heldout_topk_time_shuffle_recall = topKMean(heldout_topk_stats.time_shuffle_recall, heldout_topk_stats.count);
    const double heldout_topk_spatial_shuffle_recall = topKMean(heldout_topk_stats.spatial_shuffle_recall, heldout_topk_stats.count);
    const double heldout_topk_chance_recall =
        static_cast<double>(std::min(config.topk_k, tile_count)) / static_cast<double>(tile_count);
    const double heldout_topk_model_gain_vs_train_frequency =
        heldout_topk_model_recall - heldout_topk_train_frequency_recall;
    const double heldout_topk_time_gain_vs_train_frequency =
        heldout_topk_time_shuffle_recall - heldout_topk_train_frequency_recall;
    const double heldout_topk_spatial_gain_vs_train_frequency =
        heldout_topk_spatial_shuffle_recall - heldout_topk_train_frequency_recall;
    const double heldout_topk_time_retained_fraction =
        heldout_topk_model_gain_vs_train_frequency > 1.0e-12
            ? (heldout_topk_time_gain_vs_train_frequency / heldout_topk_model_gain_vs_train_frequency)
            : 1.0e12;
    const double heldout_topk_spatial_retained_fraction =
        heldout_topk_model_gain_vs_train_frequency > 1.0e-12
            ? (heldout_topk_spatial_gain_vs_train_frequency / heldout_topk_model_gain_vs_train_frequency)
            : 1.0e12;
    const double heldout_topk_repeat_avg_model_recall =
        topKMean(heldout_topk_repeat_avg_stats.model_recall, heldout_topk_repeat_avg_stats.count);
    const double heldout_topk_repeat_avg_persistence_recall =
        topKMean(heldout_topk_repeat_avg_stats.persistence_recall, heldout_topk_repeat_avg_stats.count);
    const double heldout_topk_repeat_avg_train_frequency_recall =
        topKMean(heldout_topk_repeat_avg_stats.train_frequency_recall, heldout_topk_repeat_avg_stats.count);
    const double heldout_topk_repeat_avg_smooth_model_recall =
        topKMean(heldout_topk_repeat_avg_smooth_stats.model_recall, heldout_topk_repeat_avg_smooth_stats.count);
    const double heldout_topk_repeat_avg_smooth_persistence_recall =
        topKMean(heldout_topk_repeat_avg_smooth_stats.persistence_recall, heldout_topk_repeat_avg_smooth_stats.count);
    const double heldout_topk_repeat_avg_smooth_train_frequency_recall =
        topKMean(heldout_topk_repeat_avg_smooth_stats.train_frequency_recall, heldout_topk_repeat_avg_smooth_stats.count);
    const double heldout_topk_repeat_avg_smooth_model_ndcg =
        topKMean(
            heldout_topk_repeat_avg_smooth_weighted_stats.model_ndcg,
            heldout_topk_repeat_avg_smooth_weighted_stats.count);
    const double heldout_topk_repeat_avg_smooth_persistence_ndcg =
        topKMean(
            heldout_topk_repeat_avg_smooth_weighted_stats.persistence_ndcg,
            heldout_topk_repeat_avg_smooth_weighted_stats.count);
    const double heldout_topk_repeat_avg_smooth_train_frequency_ndcg =
        topKMean(
            heldout_topk_repeat_avg_smooth_weighted_stats.train_frequency_ndcg,
            heldout_topk_repeat_avg_smooth_weighted_stats.count);
    const double heldout_topk_repeat_avg_smooth_model_captured_mass =
        topKMean(
            heldout_topk_repeat_avg_smooth_weighted_stats.model_captured_mass,
            heldout_topk_repeat_avg_smooth_weighted_stats.count);
    const double heldout_topk_repeat_avg_smooth_persistence_captured_mass =
        topKMean(
            heldout_topk_repeat_avg_smooth_weighted_stats.persistence_captured_mass,
            heldout_topk_repeat_avg_smooth_weighted_stats.count);
    const double heldout_topk_repeat_avg_smooth_train_frequency_captured_mass =
        topKMean(
            heldout_topk_repeat_avg_smooth_weighted_stats.train_frequency_captured_mass,
            heldout_topk_repeat_avg_smooth_weighted_stats.count);

    result.metrics = {
        {"enabled", config.enabled ? 1.0 : 0.0},
        {"host_side_learning", 1.0},
        {"prediction_target_mode_code", 3.0},
        {"residual_prediction_enabled", 1.0},
        {"l23e_residual_rate_head_enabled", 1.0},
        {"l23e_event_hazard_head_enabled", 1.0},
        {"l23e_event_window_hazard_head_enabled", 1.0},
        {"l23e_single_frame_event_report_only", 1.0},
        {"l23e_future_topk_head_enabled", 1.0},
        {"topk_objective_enabled", 1.0},
        {"topk_strength_weighted_target_enabled", 1.0},
        {"topk_repeat_avg_target_enabled", config.topk_repeat_avg_target_enabled ? 1.0 : 0.0},
        {"topk_repeat_avg_target_train_only", 1.0},
        {"topk_repeat_avg_target_frame_count", static_cast<double>(train_repeat_avg_topk_target_frame_count)},
        {"topk_repeat_avg_target_sample_count", static_cast<double>(train_repeat_avg_topk_target_sample_count)},
        {"topk_target_smooth_radius_tiles", static_cast<double>(config.topk_target_smooth_radius_tiles)},
        {"topk_target_smoothing_enabled", config.topk_target_smooth_radius_tiles > 0u ? 1.0 : 0.0},
        {"topk_target_smoothing_kernel_code", config.topk_target_smooth_radius_tiles > 0u ? 121242121.0 : 0.0},
        {"topk_target_smoothing_target_only", 1.0},
        {"topk_target_smoothing_input_feature_enabled", 0.0},
        {"topk_target_smoothing_train_repeat_avg_only", config.topk_repeat_avg_target_enabled ? 1.0 : 0.0},
        {"topk_target_smoothing_eval_repeat_avg_only", 1.0},
        {"topk_frequency_balance_enabled", config.topk_frequency_balance_enabled ? 1.0 : 0.0},
        {"topk_frequency_balance_train_only", 1.0},
        {"topk_frequency_balance_floor", config.topk_frequency_balance_floor},
        {"topk_target_channel_l23e_only", 1.0},
        {"topk_input_channel_l23e_only", 1.0},
        {"topk_feedback_enabled", 0.0},
        {"topk_tile_size_sites", static_cast<double>(config.tile_size_sites)},
        {"topk_tile_grid_side", static_cast<double>(config.tile_grid_side)},
        {"topk_tile_count", static_cast<double>(tile_count)},
        {"topk_k", static_cast<double>(config.topk_k)},
        {"topk_future_window_frames", static_cast<double>(config.topk_future_window_frames)},
        {"topk_future_window_ms", static_cast<double>(config.topk_future_window_frames) * video_config.frame_ms},
        {"topk_learning_rate", config.topk_learning_rate},
        {"topk_weight_decay", config.topk_weight_decay},
        {"topk_train_valid_sample_count", static_cast<double>(train_topk_stats.count)},
        {"topk_heldout_valid_sample_count", static_cast<double>(heldout_topk_stats.count)},
        {"topk_train_frequency_valid_sample_count", static_cast<double>(train_topk_valid_sample_count)},
        {"topk_heldout_model_recall_at_k", heldout_topk_model_recall},
        {"topk_heldout_persistence_recall_at_k", heldout_topk_persistence_recall},
        {"topk_heldout_train_frequency_recall_at_k", heldout_topk_train_frequency_recall},
        {"topk_heldout_no_learning_recall_at_k", heldout_topk_no_learning_recall},
        {"topk_heldout_temporal_block_shift_recall_at_k", heldout_topk_time_shuffle_recall},
        {"topk_heldout_spatial_tile_shuffle_recall_at_k", heldout_topk_spatial_shuffle_recall},
        {"topk_heldout_chance_recall_at_k", heldout_topk_chance_recall},
        {"topk_heldout_model_recall_vs_chance_ratio",
         heldout_topk_chance_recall > 0.0 ? (heldout_topk_model_recall / heldout_topk_chance_recall) : 0.0},
        {"topk_heldout_repeat_avg_valid_sample_count", static_cast<double>(heldout_topk_repeat_avg_stats.count)},
        {"topk_heldout_repeat_avg_model_recall_at_k", heldout_topk_repeat_avg_model_recall},
        {"topk_heldout_repeat_avg_persistence_recall_at_k", heldout_topk_repeat_avg_persistence_recall},
        {"topk_heldout_repeat_avg_train_frequency_recall_at_k", heldout_topk_repeat_avg_train_frequency_recall},
        {"topk_heldout_repeat_avg_no_learning_recall_at_k",
         topKMean(heldout_topk_repeat_avg_stats.no_learning_recall, heldout_topk_repeat_avg_stats.count)},
        {"topk_heldout_repeat_avg_temporal_block_shift_recall_at_k",
         topKMean(heldout_topk_repeat_avg_stats.time_shuffle_recall, heldout_topk_repeat_avg_stats.count)},
        {"topk_heldout_repeat_avg_spatial_tile_shuffle_recall_at_k",
         topKMean(heldout_topk_repeat_avg_stats.spatial_shuffle_recall, heldout_topk_repeat_avg_stats.count)},
        {"topk_heldout_repeat_avg_smooth_valid_sample_count",
         static_cast<double>(heldout_topk_repeat_avg_smooth_stats.count)},
        {"topk_heldout_repeat_avg_smooth_model_recall_at_k", heldout_topk_repeat_avg_smooth_model_recall},
        {"topk_heldout_repeat_avg_smooth_persistence_recall_at_k",
         heldout_topk_repeat_avg_smooth_persistence_recall},
        {"topk_heldout_repeat_avg_smooth_train_frequency_recall_at_k",
         heldout_topk_repeat_avg_smooth_train_frequency_recall},
        {"topk_heldout_repeat_avg_smooth_no_learning_recall_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_stats.no_learning_recall,
             heldout_topk_repeat_avg_smooth_stats.count)},
        {"topk_heldout_repeat_avg_smooth_temporal_block_shift_recall_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_stats.time_shuffle_recall,
             heldout_topk_repeat_avg_smooth_stats.count)},
        {"topk_heldout_repeat_avg_smooth_spatial_tile_shuffle_recall_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_stats.spatial_shuffle_recall,
             heldout_topk_repeat_avg_smooth_stats.count)},
        {"topk_heldout_repeat_avg_smooth_weighted_valid_sample_count",
         static_cast<double>(heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_repeat_avg_smooth_model_ndcg_at_k", heldout_topk_repeat_avg_smooth_model_ndcg},
        {"topk_heldout_repeat_avg_smooth_persistence_ndcg_at_k",
         heldout_topk_repeat_avg_smooth_persistence_ndcg},
        {"topk_heldout_repeat_avg_smooth_train_frequency_ndcg_at_k",
         heldout_topk_repeat_avg_smooth_train_frequency_ndcg},
        {"topk_heldout_repeat_avg_smooth_no_learning_ndcg_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_weighted_stats.no_learning_ndcg,
             heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_repeat_avg_smooth_temporal_block_shift_ndcg_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_weighted_stats.time_shuffle_ndcg,
             heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_repeat_avg_smooth_spatial_tile_shuffle_ndcg_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_weighted_stats.spatial_shuffle_ndcg,
             heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_repeat_avg_smooth_model_captured_ideal_mass_at_k",
         heldout_topk_repeat_avg_smooth_model_captured_mass},
        {"topk_heldout_repeat_avg_smooth_persistence_captured_ideal_mass_at_k",
         heldout_topk_repeat_avg_smooth_persistence_captured_mass},
        {"topk_heldout_repeat_avg_smooth_train_frequency_captured_ideal_mass_at_k",
         heldout_topk_repeat_avg_smooth_train_frequency_captured_mass},
        {"topk_heldout_repeat_avg_smooth_no_learning_captured_ideal_mass_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_weighted_stats.no_learning_captured_mass,
             heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_repeat_avg_smooth_temporal_block_shift_captured_ideal_mass_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_weighted_stats.time_shuffle_captured_mass,
             heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_repeat_avg_smooth_spatial_tile_shuffle_captured_ideal_mass_at_k",
         topKMean(
             heldout_topk_repeat_avg_smooth_weighted_stats.spatial_shuffle_captured_mass,
             heldout_topk_repeat_avg_smooth_weighted_stats.count)},
        {"topk_heldout_model_ndcg_at_k", topKMean(heldout_topk_stats.model_ndcg, heldout_topk_stats.count)},
        {"topk_heldout_persistence_ndcg_at_k", topKMean(heldout_topk_stats.persistence_ndcg, heldout_topk_stats.count)},
        {"topk_heldout_train_frequency_ndcg_at_k", topKMean(heldout_topk_stats.train_frequency_ndcg, heldout_topk_stats.count)},
        {"topk_heldout_no_learning_ndcg_at_k", topKMean(heldout_topk_stats.no_learning_ndcg, heldout_topk_stats.count)},
        {"topk_heldout_temporal_block_shift_ndcg_at_k", topKMean(heldout_topk_stats.time_shuffle_ndcg, heldout_topk_stats.count)},
        {"topk_heldout_spatial_tile_shuffle_ndcg_at_k", topKMean(heldout_topk_stats.spatial_shuffle_ndcg, heldout_topk_stats.count)},
        {"topk_heldout_model_mrr", topKMean(heldout_topk_stats.model_mrr, heldout_topk_stats.count)},
        {"topk_heldout_persistence_mrr", topKMean(heldout_topk_stats.persistence_mrr, heldout_topk_stats.count)},
        {"topk_heldout_train_frequency_mrr", topKMean(heldout_topk_stats.train_frequency_mrr, heldout_topk_stats.count)},
        {"topk_heldout_no_learning_mrr", topKMean(heldout_topk_stats.no_learning_mrr, heldout_topk_stats.count)},
        {"topk_heldout_temporal_block_shift_mrr", topKMean(heldout_topk_stats.time_shuffle_mrr, heldout_topk_stats.count)},
        {"topk_heldout_spatial_tile_shuffle_mrr", topKMean(heldout_topk_stats.spatial_shuffle_mrr, heldout_topk_stats.count)},
        {"topk_heldout_relative_improvement_vs_persistence",
         (heldout_topk_model_recall - heldout_topk_persistence_recall)
             / std::max(1.0e-12, heldout_topk_persistence_recall)},
        {"topk_heldout_relative_improvement_vs_train_frequency",
         heldout_topk_model_gain_vs_train_frequency
             / std::max(1.0e-12, heldout_topk_train_frequency_recall)},
        {"topk_heldout_relative_improvement_vs_no_learning",
         (heldout_topk_model_recall - heldout_topk_no_learning_recall)
             / std::max(1.0e-12, heldout_topk_no_learning_recall)},
        {"topk_heldout_model_gain_vs_train_frequency", heldout_topk_model_gain_vs_train_frequency},
        {"topk_heldout_temporal_block_shift_gain_vs_train_frequency", heldout_topk_time_gain_vs_train_frequency},
        {"topk_heldout_spatial_tile_shuffle_gain_vs_train_frequency", heldout_topk_spatial_gain_vs_train_frequency},
        {"topk_heldout_temporal_block_shift_retained_fraction", heldout_topk_time_retained_fraction},
        {"topk_heldout_spatial_tile_shuffle_retained_fraction", heldout_topk_spatial_retained_fraction},
        {"topk_weight_l1", topk_weight_l1},
        {"topk_weight_max_abs", topk_weight_max_abs},
        {"topk_bias_l1", topk_bias_l1},
        {"topk_weight_abs_current", topk_feature_norms.current},
        {"topk_weight_abs_trace", topk_feature_norms.trace},
        {"topk_weight_abs_derivative", topk_feature_norms.derivative},
        {"topk_weight_abs_lag", topk_feature_norms.lag},
        {"topk_weight_abs_context", topk_feature_norms.context},
        {"topk_weight_abs_sequence", topk_feature_norms.sequence},
        {"topk_weight_abs_group_total", topk_feature_norms.total},
        {"topk_local_readout_enabled", 1.0},
        {"topk_dense_all_to_all_readout_enabled", 0.0},
        {"topk_local_radius_tiles", static_cast<double>(config.topk_local_radius_tiles)},
        {"topk_active_readout_pair_fraction", topk_active_readout_pair_fraction},
        {"topk_local_pair_count", static_cast<double>(topk_local_abs_weights.size())},
        {"topk_distant_pair_count", static_cast<double>(topk_distant_abs_weights.size())},
        {"topk_local_nonzero_pair_count", static_cast<double>(topk_local_nonzero_pair_count)},
        {"topk_distant_nonzero_pair_count", static_cast<double>(topk_distant_nonzero_pair_count)},
        {"topk_local_abs_weight_sum", topk_local_abs_weight_sum},
        {"topk_distant_abs_weight_sum", topk_distant_abs_weight_sum},
        {"topk_distant_abs_weight_max", topk_distant_abs_weight_max},
        {"topk_local_abs_weight_mean", topk_local_abs_weight_mean},
        {"topk_distant_abs_weight_mean", topk_distant_abs_weight_mean},
        {"topk_diagonal_abs_weight_mean", topk_diagonal_abs_weight_mean},
        {"topk_offdiagonal_abs_weight_mean", topk_offdiagonal_abs_weight_mean},
        {"train_then_heldout_enabled", 1.0},
        {"evaluation_updates_enabled", 0.0},
        {"training_epoch_count", static_cast<double>(config.training_epochs)},
        {"training_update_count", static_cast<double>(config.training_epochs) * static_cast<double>(train_prediction_count)},
        {"feature_standardization_enabled", 1.0},
        {"feature_standardization_train_only", 1.0},
        {"feature_standardization_feature_count", static_cast<double>(feature_channel_count)},
        {"feature_standardization_train_observation_count", static_cast<double>(feature_train_observation_count)},
        {"feature_standardization_std_floor", kHVAPredictorFeatureStdFloor},
        {"feature_standardization_std_floor_count", static_cast<double>(feature_std_floor_count)},
        {"feature_standardization_std_min", feature_train_std_min},
        {"feature_standardization_std_median", feature_train_std_median},
        {"feature_standardization_std_max", feature_train_std_max},
        {"local_l2_weight_decay_enabled", config.weight_decay > 0.0 ? 1.0 : 0.0},
        {"local_l2_weight_decay", config.weight_decay},
        {"event_local_l2_weight_decay", config.event_weight_decay},
        {"posthoc_global_normalization_enabled", 0.0},
        {"event_window_frames", static_cast<double>(config.event_window_frames)},
        {"event_window_ms", static_cast<double>(config.event_window_frames) * video_config.frame_ms},
        {"event_window_target_mode_code", 1.0},
        {"event_hazard_train_only_threshold_enabled", 1.0},
        {"event_hazard_input_channel_l23e_only", 1.0},
        {"event_hazard_non_l23_target_enabled", 0.0},
        {"event_bias_initialized_from_train_base_rate", 1.0},
        {"event_base_rate_floor", kHVAPredictorEventRateFloor},
        {"event_residual_gain", config.event_residual_gain},
        {"learning_target_normalized_rate_residual_enabled", 1.0},
        {"learning_target_zscore_enabled", 0.0},
        {"train_only_normalization_enabled", 1.0},
        {"signed_residual_host_weights_enabled", 1.0},
        {"signed_weight_engineering_approximation", 1.0},
        {"local_readout_enabled", 1.0},
        {"dense_all_to_all_readout_enabled", 0.0},
        {"lower_v1_frozen", 1.0},
        {"hva_to_v1_connection_count", 0.0},
        {"hva_to_v1_current_enabled", 0.0},
        {"lower_v1_replay_site_count_sum_before", site_count_sum_before},
        {"lower_v1_replay_site_count_sum_after", site_count_sum_after},
        {"lower_v1_replay_site_count_sum_sq_before", site_count_sum_sq_before},
        {"lower_v1_replay_site_count_sum_sq_after", site_count_sum_sq_after},
        {"lower_v1_replay_site_count_fingerprint32_before", static_cast<double>(site_count_fingerprint_before)},
        {"lower_v1_replay_site_count_fingerprint32_after", static_cast<double>(site_count_fingerprint_after)},
        {"lower_v1_replay_tile_rate_sum_before", tile_rate_sum_before},
        {"lower_v1_replay_tile_rate_sum_after", tile_rate_sum_after},
        {"lower_v1_replay_tile_rate_sum_sq_before", tile_rate_sum_sq_before},
        {"lower_v1_replay_tile_rate_sum_sq_after", tile_rate_sum_sq_after},
        {"lower_v1_replay_tile_rate_fingerprint32_before", static_cast<double>(tile_rate_fingerprint_before)},
        {"lower_v1_replay_tile_rate_fingerprint32_after", static_cast<double>(tile_rate_fingerprint_after)},
        {"lower_v1_replay_multitask_target_sum_before", multitask_target_sum_before},
        {"lower_v1_replay_multitask_target_sum_after", multitask_target_sum_after},
        {"lower_v1_replay_multitask_target_sum_sq_before", multitask_target_sum_sq_before},
        {"lower_v1_replay_multitask_target_sum_sq_after", multitask_target_sum_sq_after},
        {"lower_v1_replay_multitask_target_fingerprint32_before", static_cast<double>(multitask_target_fingerprint_before)},
        {"lower_v1_replay_multitask_target_fingerprint32_after", static_cast<double>(multitask_target_fingerprint_after)},
        {"lower_v1_replay_multitask_target_fingerprint_equal", (multitask_target_fingerprint_before == multitask_target_fingerprint_after
                                                                 && multitask_target_sum_before == multitask_target_sum_after) ? 1.0 : 0.0},
        {"lower_v1_replay_fingerprint_equal", (site_count_fingerprint_before == site_count_fingerprint_after
                                                && tile_rate_fingerprint_before == tile_rate_fingerprint_after
                                                && site_count_sum_before == site_count_sum_after
                                                && tile_rate_sum_before == tile_rate_sum_after) ? 1.0 : 0.0},
        {"lower_v1_weight_delta_max_after_hva", 0.0},
        {"lower_v1_output_delta_max_after_hva", 0.0},
        {"v1_mutation_after_hva_enabled", 0.0},
        {"tile_grid_side", static_cast<double>(config.tile_grid_side)},
        {"tile_size_sites", static_cast<double>(config.tile_size_sites)},
        {"tile_count", static_cast<double>(tile_count)},
        {"target_channel_count", static_cast<double>(target_channel_count)},
        {"required_target_channel_count", static_cast<double>(kHVAPredictorRequiredTargetChannelCount)},
        {"l23e_target_channel_enabled", 1.0},
        {"l4e_target_channel_enabled", 0.0},
        {"l23pv_target_channel_enabled", 0.0},
        {"l23som_target_channel_enabled", 0.0},
        {"non_l23_required_target_channel_count", 0.0},
        {"non_l23_target_autoregressive_baseline_enabled", 0.0},
        {"input_channel_l23e_only", 1.0},
        {"input_channel_l4e_enabled", 0.0},
        {"input_channel_l23pv_enabled", 0.0},
        {"delay_frames", static_cast<double>(config.delay_frames)},
        {"trace_tau_frames", config.trace_tau_frames},
        {"trace_decay", trace_decay},
        {"feature_channel_count", static_cast<double>(feature_channel_count)},
        {"non_sequence_feature_channel_count", static_cast<double>(non_sequence_feature_channel_count)},
        {"base_feature_channel_count", static_cast<double>(kHVAPredictorBaseFeatureChannelCount)},
        {"lag_history_frame_count", static_cast<double>(config.feature_lag_count)},
        {"lag_history_ms", static_cast<double>(config.feature_lag_count) * video_config.frame_ms},
        {"lag_history_l23e_only", 1.0},
        {"lag_feature_future_lookahead_frames", 0.0},
        {"local_context_feature_enabled", config.feature_context_radius_tiles > 0u ? 1.0 : 0.0},
        {"local_context_radius_tiles", static_cast<double>(config.feature_context_radius_tiles)},
        {"local_context_summary_feature_count",
         static_cast<double>(kHVAPredictorContextSummaryFeatureCount * (config.feature_lag_count + 1u))},
        {"directional_context_feature_enabled", hvaPredictorDirectionalContextActive(config) ? 1.0 : 0.0},
        {"directional_context_radius_tiles",
         hvaPredictorDirectionalContextActive(config)
             ? static_cast<double>(config.feature_context_radius_tiles)
             : 0.0},
        {"directional_context_feature_count",
         hvaPredictorDirectionalContextActive(config)
             ? static_cast<double>(kHVAPredictorDirectionalContextFeatureCount * (config.feature_lag_count + 1u))
             : 0.0},
        {"directional_context_l23e_only", 1.0},
        {"directional_context_future_lookahead_frames", 0.0},
        {"sequence_state_enabled", sequence_state_active ? 1.0 : 0.0},
        {"sequence_state_dim", sequence_state_active ? static_cast<double>(config.sequence_state_dim) : 0.0},
        {"sequence_state_feature_count", sequence_state_active ? static_cast<double>(config.sequence_state_dim) : 0.0},
        {"sequence_state_leak", config.sequence_state_leak},
        {"sequence_state_input_scale", config.sequence_state_input_scale},
        {"sequence_state_neighbor_scale", config.sequence_state_neighbor_scale},
        {"sequence_state_neighbor_radius_tiles", sequence_state_active ? 1.0 : 0.0},
        {"sequence_state_l23e_only", 1.0},
        {"sequence_state_future_lookahead_frames", 0.0},
        {"topk_sequence_state_feature_enabled", sequence_state_active ? 1.0 : 0.0},
        {"residual_event_sequence_state_feature_enabled", 0.0},
        {"local_context_l23e_only", 1.0},
        {"feature_uses_non_l23_inputs", 0.0},
        {"feature_future_leakage_enabled", 0.0},
        {"trace_channel_count", static_cast<double>(kHVAPredictorTraceChannelCount)},
        {"trace_fast_tau_ms", trace_tau_ms[0]},
        {"trace_medium_tau_ms", trace_tau_ms[1]},
        {"trace_slow_tau_ms", trace_tau_ms[2]},
        {"trace_fast_tau_frames", trace_tau_frames[0]},
        {"trace_medium_tau_frames", trace_tau_frames[1]},
        {"trace_slow_tau_frames", trace_tau_frames[2]},
        {"derivative_feature_enabled", 1.0},
        {"past_only_feature_lookahead_frames", 0.0},
        {"event_threshold_quantile", config.event_threshold_quantile},
        {"event_threshold_min_hz", config.event_threshold_min_hz},
        {"event_threshold_min_norm", event_threshold_min_norm},
        {"event_min_train_positive_count", static_cast<double>(config.event_min_train_positive_count)},
        {"event_threshold_actual_min_norm", event_threshold_min_actual},
        {"event_threshold_median_norm", event_threshold_median},
        {"event_threshold_actual_max_norm", event_threshold_max_actual},
        {"event_train_rate_min", event_train_rate_min},
        {"event_train_rate_median", event_train_rate_median},
        {"event_train_rate_max", event_train_rate_max},
        {"event_selected_train_rate_median", selected_event_train_rate_median},
        {"event_bias_min", event_bias_min},
        {"event_bias_median", event_bias_median},
        {"event_bias_max", event_bias_max},
        {"event_selected_tile_count", static_cast<double>(event_selected_tile_count)},
        {"event_selected_tile_fraction", event_selected_tile_count / static_cast<double>(channel_tile_count)},
        {"learning_rate", config.learning_rate},
        {"residual_learning_rate", config.learning_rate},
        {"event_learning_rate", config.event_learning_rate},
        {"bias_learning_rate", config.bias_learning_rate},
        {"event_bias_learning_rate", config.event_bias_learning_rate},
        {"event_weight_decay", config.event_weight_decay},
        {"rate_scale_hz", config.rate_scale_hz},
        {"weight_clip", config.weight_clip},
        {"heldout_fraction", config.heldout_fraction},
        {"local_radius_tiles", static_cast<double>(config.local_radius_tiles)},
        {"active_readout_pair_fraction", active_pair_count / static_cast<double>(target_pair_count)},
        {"train_residual_std_min_norm", train_residual_std_min},
        {"train_residual_std_median_norm", train_residual_std_median},
        {"heldout_mode_code", 2.0},
        {"heldout_start_repeat", 0.0},
        {"heldout_start_frame", static_cast<double>(heldout_start_frame)},
        {"heldout_frame_count", static_cast<double>(video_config.effective_frame_count - heldout_start_frame)},
        {"train_frame_count", static_cast<double>(train_frame_count)},
        {"future_target_horizon_frames", static_cast<double>(future_target_horizon_frames)},
        {"topk_split_safety_horizon_frames", static_cast<double>(future_target_horizon_frames)},
        {"boundary_gap_prediction_count", static_cast<double>(boundary_gap_prediction_count)},
        {"sample_count", static_cast<double>(sample_count)},
        {"prediction_count", prediction_count},
        {"train_prediction_count", static_cast<double>(train_prediction_count)},
        {"heldout_prediction_count", static_cast<double>(heldout_prediction_count)},
        {"multitask_model_mse_norm", heldout_model_mse},
        {"multitask_model_raw_mse_norm", heldout_model_mse},
        {"multitask_model_residual_z_mse", heldout_residual_z_mse},
        {"multitask_zero_mse_norm", heldout_no_learning_mse},
        {"multitask_persistence_mse_norm", heldout_persistence_mse},
        {"multitask_train_model_mse_norm", train_model_mse},
        {"multitask_train_model_residual_z_mse", train_residual_z_mse},
        {"multitask_train_persistence_mse_norm", mse(train_stats.persistence_sq, train_stats.count)},
        {"multitask_train_train_mean_mse_norm", mse(train_stats.train_mean_sq, train_stats.count)},
        {"multitask_train_no_learning_mse_norm", mse(train_stats.no_learning_sq, train_stats.count)},
        {"multitask_heldout_model_mse_norm", heldout_model_mse},
        {"multitask_heldout_model_raw_mse_norm", heldout_model_mse},
        {"multitask_heldout_model_residual_z_mse", heldout_residual_z_mse},
        {"multitask_heldout_persistence_mse_norm", heldout_persistence_mse},
        {"multitask_heldout_train_mean_mse_norm", heldout_train_mean_mse},
        {"multitask_heldout_no_learning_mse_norm", heldout_no_learning_mse},
        {"multitask_heldout_temporal_block_shift_mse_norm", heldout_time_shuffle_mse},
        {"multitask_heldout_spatial_tile_shuffle_mse_norm", heldout_spatial_shuffle_mse},
        {"multitask_heldout_time_shuffle_mse_norm", heldout_time_shuffle_mse},
        {"multitask_heldout_spatial_shuffle_mse_norm", heldout_spatial_shuffle_mse},
        {"multitask_improvement_vs_zero_norm", heldout_no_learning_mse - heldout_model_mse},
        {"multitask_improvement_vs_persistence_norm", heldout_persistence_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_persistence_norm", heldout_persistence_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_train_mean_norm", heldout_train_mean_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_no_learning_norm", heldout_no_learning_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_temporal_block_shift_norm", heldout_time_shuffle_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_spatial_tile_shuffle_norm", heldout_spatial_shuffle_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_time_shuffle_norm", heldout_time_shuffle_mse - heldout_model_mse},
        {"multitask_heldout_improvement_vs_spatial_shuffle_norm", heldout_spatial_shuffle_mse - heldout_model_mse},
        {"multitask_relative_improvement_vs_zero", heldout_no_learning_mse > 0.0 ? ((heldout_no_learning_mse - heldout_model_mse) / heldout_no_learning_mse) : 0.0},
        {"multitask_relative_improvement_vs_persistence", heldout_persistence_mse > 0.0 ? ((heldout_persistence_mse - heldout_model_mse) / heldout_persistence_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_persistence", heldout_persistence_mse > 0.0 ? ((heldout_persistence_mse - heldout_model_mse) / heldout_persistence_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_train_mean", heldout_train_mean_mse > 0.0 ? ((heldout_train_mean_mse - heldout_model_mse) / heldout_train_mean_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_no_learning", heldout_no_learning_mse > 0.0 ? ((heldout_no_learning_mse - heldout_model_mse) / heldout_no_learning_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_temporal_block_shift", heldout_time_shuffle_mse > 0.0 ? ((heldout_time_shuffle_mse - heldout_model_mse) / heldout_time_shuffle_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_spatial_tile_shuffle", heldout_spatial_shuffle_mse > 0.0 ? ((heldout_spatial_shuffle_mse - heldout_model_mse) / heldout_spatial_shuffle_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_time_shuffle", heldout_time_shuffle_mse > 0.0 ? ((heldout_time_shuffle_mse - heldout_model_mse) / heldout_time_shuffle_mse) : 0.0},
        {"multitask_heldout_relative_improvement_vs_spatial_shuffle", heldout_spatial_shuffle_mse > 0.0 ? ((heldout_spatial_shuffle_mse - heldout_model_mse) / heldout_spatial_shuffle_mse) : 0.0},
        {"model_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].model_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"model_raw_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].model_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"model_residual_z_mse", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].residual_z_model_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"zero_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].no_learning_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"persistence_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].persistence_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_model_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].model_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_model_raw_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].model_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_model_residual_z_mse", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].residual_z_model_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_persistence_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].persistence_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_train_mean_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].train_mean_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_no_learning_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].no_learning_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_temporal_block_shift_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].time_shuffle_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"heldout_spatial_tile_shuffle_mse_norm", heldout_stats_by_channel[0].count > 0u ? (heldout_stats_by_channel[0].spatial_shuffle_sq / static_cast<double>(heldout_stats_by_channel[0].count)) : 0.0},
        {"target_mean_hz", meanHz(heldout_stats.target_rate_sum, heldout_stats.count)},
        {"prediction_mean_hz", meanHz(heldout_stats.prediction_rate_sum, heldout_stats.count)},
        {"target_std_norm", standardDeviation(heldout_stats.targets)},
        {"prediction_std_norm", standardDeviation(heldout_stats.predictions)},
        {"heldout_prediction_min_norm", heldout_stats.prediction_min},
        {"heldout_prediction_max_norm", heldout_stats.prediction_max},
        {"mean_tile_prediction_corr", mean_corr},
        {"median_tile_prediction_corr", median(tile_correlations_by_channel.front())},
        {"weight_l1", weight_l1},
        {"weight_max_abs", weight_max_abs},
        {"bias_l1", bias_l1},
        {"event_weight_l1", event_weight_l1},
        {"event_weight_max_abs", event_weight_max_abs},
        {"event_bias_l1", event_bias_l1},
        {"residual_weight_abs_current", residual_feature_norms.current},
        {"residual_weight_abs_trace", residual_feature_norms.trace},
        {"residual_weight_abs_derivative", residual_feature_norms.derivative},
        {"residual_weight_abs_lag", residual_feature_norms.lag},
        {"residual_weight_abs_context", residual_feature_norms.context},
        {"residual_weight_abs_sequence", residual_feature_norms.sequence},
        {"residual_weight_abs_group_total", residual_feature_norms.total},
        {"event_weight_abs_current", event_feature_norms.current},
        {"event_weight_abs_trace", event_feature_norms.trace},
        {"event_weight_abs_derivative", event_feature_norms.derivative},
        {"event_weight_abs_lag", event_feature_norms.lag},
        {"event_weight_abs_context", event_feature_norms.context},
        {"event_weight_abs_sequence", event_feature_norms.sequence},
        {"event_weight_abs_group_total", event_feature_norms.total},
        {"local_abs_weight_mean", local_abs_weight_mean},
        {"distant_abs_weight_mean", distant_abs_weight_mean},
        {"local_distant_abs_weight_ratio", local_distant_abs_weight_ratio},
        {"diagonal_abs_weight_mean", diagonal_abs_weight_mean},
        {"offdiagonal_abs_weight_mean", offdiagonal_abs_weight_mean},
    };

    const auto appendMetric = [&](const std::string &prefix, const std::string &name, double value) {
        result.metrics.push_back({prefix.empty() ? name : (prefix + "_" + name), value});
    };
    const auto appendEventMetrics = [&](const std::string &prefix, const EventStats &train, const EventStats &heldout) {
        const double heldout_model_brier = eventBrier(heldout.model_brier, heldout.count);
        const double heldout_persistence_brier = eventBrier(heldout.persistence_brier, heldout.count);
        const double heldout_train_mean_brier = eventBrier(heldout.train_mean_brier, heldout.count);
        const double heldout_no_learning_brier = eventBrier(heldout.no_learning_brier, heldout.count);
        const double heldout_time_shuffle_brier = eventBrier(heldout.time_shuffle_brier, heldout.count);
        const double heldout_spatial_shuffle_brier = eventBrier(heldout.spatial_shuffle_brier, heldout.count);
        const double heldout_model_logloss = eventMeanLoss(heldout.model_logloss, heldout.count);
        const double heldout_persistence_logloss = eventMeanLoss(heldout.persistence_logloss, heldout.count);
        const double heldout_train_mean_logloss = eventMeanLoss(heldout.train_mean_logloss, heldout.count);
        const double heldout_no_learning_logloss = eventMeanLoss(heldout.no_learning_logloss, heldout.count);
        const double heldout_time_shuffle_logloss = eventMeanLoss(heldout.time_shuffle_logloss, heldout.count);
        const double heldout_spatial_shuffle_logloss = eventMeanLoss(heldout.spatial_shuffle_logloss, heldout.count);
        const double heldout_model_auc = eventAuc(heldout.predictions, heldout.targets);
        const double heldout_persistence_auc = eventAuc(heldout.persistence_predictions, heldout.targets);
        const double heldout_train_mean_auc = eventAuc(heldout.train_mean_predictions, heldout.targets);
        const double heldout_no_learning_auc = eventAuc(heldout.no_learning_predictions, heldout.targets);
        const double heldout_time_shuffle_auc = eventAuc(heldout.time_shuffle_predictions, heldout.targets);
        const double heldout_spatial_shuffle_auc = eventAuc(heldout.spatial_shuffle_predictions, heldout.targets);
        const double heldout_model_auprc = eventAuprc(heldout.predictions, heldout.targets);
        const double heldout_persistence_auprc = eventAuprc(heldout.persistence_predictions, heldout.targets);
        const double heldout_train_mean_auprc = eventAuprc(heldout.train_mean_predictions, heldout.targets);
        const double heldout_no_learning_auprc = eventAuprc(heldout.no_learning_predictions, heldout.targets);
        const double heldout_time_shuffle_auprc = eventAuprc(heldout.time_shuffle_predictions, heldout.targets);
        const double heldout_spatial_shuffle_auprc = eventAuprc(heldout.spatial_shuffle_predictions, heldout.targets);
        appendMetric(prefix, "train_event_prediction_count", static_cast<double>(train.count));
        appendMetric(prefix, "train_event_positive_count", static_cast<double>(train.positive_count));
        appendMetric(prefix, "train_event_positive_fraction", eventPositiveFraction(train.positive_count, train.count));
        appendMetric(prefix, "heldout_event_prediction_count", static_cast<double>(heldout.count));
        appendMetric(prefix, "heldout_event_positive_count", static_cast<double>(heldout.positive_count));
        appendMetric(prefix, "heldout_event_positive_fraction", eventPositiveFraction(heldout.positive_count, heldout.count));
        appendMetric(prefix, "heldout_event_model_brier", heldout_model_brier);
        appendMetric(prefix, "heldout_event_persistence_brier", heldout_persistence_brier);
        appendMetric(prefix, "heldout_event_train_mean_brier", heldout_train_mean_brier);
        appendMetric(prefix, "heldout_event_no_learning_brier", heldout_no_learning_brier);
        appendMetric(prefix, "heldout_event_temporal_block_shift_brier", heldout_time_shuffle_brier);
        appendMetric(prefix, "heldout_event_spatial_tile_shuffle_brier", heldout_spatial_shuffle_brier);
        appendMetric(prefix, "heldout_event_model_logloss", heldout_model_logloss);
        appendMetric(prefix, "heldout_event_persistence_logloss", heldout_persistence_logloss);
        appendMetric(prefix, "heldout_event_train_mean_logloss", heldout_train_mean_logloss);
        appendMetric(prefix, "heldout_event_no_learning_logloss", heldout_no_learning_logloss);
        appendMetric(prefix, "heldout_event_temporal_block_shift_logloss", heldout_time_shuffle_logloss);
        appendMetric(prefix, "heldout_event_spatial_tile_shuffle_logloss", heldout_spatial_shuffle_logloss);
        appendMetric(prefix, "heldout_event_model_auc", heldout_model_auc);
        appendMetric(prefix, "heldout_event_persistence_auc", heldout_persistence_auc);
        appendMetric(prefix, "heldout_event_train_mean_auc", heldout_train_mean_auc);
        appendMetric(prefix, "heldout_event_no_learning_auc", heldout_no_learning_auc);
        appendMetric(prefix, "heldout_event_temporal_block_shift_auc", heldout_time_shuffle_auc);
        appendMetric(prefix, "heldout_event_spatial_tile_shuffle_auc", heldout_spatial_shuffle_auc);
        appendMetric(prefix, "heldout_event_model_auprc", heldout_model_auprc);
        appendMetric(prefix, "heldout_event_persistence_auprc", heldout_persistence_auprc);
        appendMetric(prefix, "heldout_event_train_mean_auprc", heldout_train_mean_auprc);
        appendMetric(prefix, "heldout_event_no_learning_auprc", heldout_no_learning_auprc);
        appendMetric(prefix, "heldout_event_temporal_block_shift_auprc", heldout_time_shuffle_auprc);
        appendMetric(prefix, "heldout_event_spatial_tile_shuffle_auprc", heldout_spatial_shuffle_auprc);
        appendMetric(prefix, "heldout_event_improvement_vs_persistence", heldout_persistence_brier - heldout_model_brier);
        appendMetric(prefix, "heldout_event_improvement_vs_train_mean", heldout_train_mean_brier - heldout_model_brier);
        appendMetric(prefix, "heldout_event_improvement_vs_no_learning", heldout_no_learning_brier - heldout_model_brier);
        appendMetric(prefix, "heldout_event_improvement_vs_temporal_block_shift", heldout_time_shuffle_brier - heldout_model_brier);
        appendMetric(prefix, "heldout_event_improvement_vs_spatial_tile_shuffle", heldout_spatial_shuffle_brier - heldout_model_brier);
        appendMetric(prefix, "heldout_event_logloss_improvement_vs_persistence", heldout_persistence_logloss - heldout_model_logloss);
        appendMetric(prefix, "heldout_event_logloss_improvement_vs_train_mean", heldout_train_mean_logloss - heldout_model_logloss);
        appendMetric(prefix, "heldout_event_logloss_improvement_vs_no_learning", heldout_no_learning_logloss - heldout_model_logloss);
        appendMetric(prefix, "heldout_event_logloss_improvement_vs_temporal_block_shift", heldout_time_shuffle_logloss - heldout_model_logloss);
        appendMetric(prefix, "heldout_event_logloss_improvement_vs_spatial_tile_shuffle", heldout_spatial_shuffle_logloss - heldout_model_logloss);
        appendMetric(prefix, "heldout_event_auc_improvement_vs_persistence", heldout_model_auc - heldout_persistence_auc);
        appendMetric(prefix, "heldout_event_auc_improvement_vs_train_mean", heldout_model_auc - heldout_train_mean_auc);
        appendMetric(prefix, "heldout_event_auc_improvement_vs_no_learning", heldout_model_auc - heldout_no_learning_auc);
        appendMetric(prefix, "heldout_event_auc_improvement_vs_temporal_block_shift", heldout_model_auc - heldout_time_shuffle_auc);
        appendMetric(prefix, "heldout_event_auc_improvement_vs_spatial_tile_shuffle", heldout_model_auc - heldout_spatial_shuffle_auc);
        appendMetric(prefix, "heldout_event_auprc_improvement_vs_persistence", heldout_model_auprc - heldout_persistence_auprc);
        appendMetric(prefix, "heldout_event_auprc_improvement_vs_train_mean", heldout_model_auprc - heldout_train_mean_auprc);
        appendMetric(prefix, "heldout_event_auprc_improvement_vs_no_learning", heldout_model_auprc - heldout_no_learning_auprc);
        appendMetric(prefix, "heldout_event_auprc_improvement_vs_temporal_block_shift", heldout_model_auprc - heldout_time_shuffle_auprc);
        appendMetric(prefix, "heldout_event_auprc_improvement_vs_spatial_tile_shuffle", heldout_model_auprc - heldout_spatial_shuffle_auprc);
        appendMetric(prefix, "heldout_event_relative_improvement_vs_persistence", heldout_persistence_brier > 0.0 ? ((heldout_persistence_brier - heldout_model_brier) / heldout_persistence_brier) : 0.0);
        appendMetric(prefix, "heldout_event_relative_improvement_vs_train_mean", heldout_train_mean_brier > 0.0 ? ((heldout_train_mean_brier - heldout_model_brier) / heldout_train_mean_brier) : 0.0);
        appendMetric(prefix, "heldout_event_relative_improvement_vs_no_learning", heldout_no_learning_brier > 0.0 ? ((heldout_no_learning_brier - heldout_model_brier) / heldout_no_learning_brier) : 0.0);
        appendMetric(prefix, "heldout_event_relative_improvement_vs_temporal_block_shift", heldout_time_shuffle_brier > 0.0 ? ((heldout_time_shuffle_brier - heldout_model_brier) / heldout_time_shuffle_brier) : 0.0);
        appendMetric(prefix, "heldout_event_relative_improvement_vs_spatial_tile_shuffle", heldout_spatial_shuffle_brier > 0.0 ? ((heldout_spatial_shuffle_brier - heldout_model_brier) / heldout_spatial_shuffle_brier) : 0.0);
        appendMetric(prefix, "heldout_event_prediction_mean", heldout.count > 0u ? (heldout.prediction_sum / static_cast<double>(heldout.count)) : 0.0);
        appendMetric(prefix, "heldout_event_target_mean", heldout.count > 0u ? (heldout.target_sum / static_cast<double>(heldout.count)) : 0.0);
        appendMetric(prefix, "heldout_event_prediction_corr", responseCorrelation(heldout.predictions, heldout.targets));
    };
    appendEventMetrics("l23e_event_all_tiles", train_event_stats_all, heldout_event_stats_all);
    appendEventMetrics("l23e_event_selected_tiles", train_event_stats_selected, heldout_event_stats_selected);
    appendEventMetrics(
        "l23e_single_frame_event_all_tiles",
        train_single_frame_event_stats_all,
        heldout_single_frame_event_stats_all);
    appendEventMetrics(
        "l23e_single_frame_event_selected_tiles",
        train_single_frame_event_stats_selected,
        heldout_single_frame_event_stats_selected);

    const auto appendChannelMetrics = [&](unsigned int target_channel, bool append_legacy_names) {
        const std::string &prefix = target_specs[target_channel].name;
        const SplitStats &train_channel_stats = train_stats_by_channel[target_channel];
        const SplitStats &heldout_channel_stats = heldout_stats_by_channel[target_channel];
        const double channel_train_model_mse = mse(train_channel_stats.model_sq, train_channel_stats.count);
        const double channel_train_residual_z_mse =
            mse(train_channel_stats.residual_z_model_sq, train_channel_stats.count);
        const double channel_heldout_model_mse = mse(heldout_channel_stats.model_sq, heldout_channel_stats.count);
        const double channel_heldout_residual_z_mse =
            mse(heldout_channel_stats.residual_z_model_sq, heldout_channel_stats.count);
        const double channel_heldout_persistence_mse =
            mse(heldout_channel_stats.persistence_sq, heldout_channel_stats.count);
        const double channel_heldout_train_mean_mse =
            mse(heldout_channel_stats.train_mean_sq, heldout_channel_stats.count);
        const double channel_heldout_no_learning_mse =
            mse(heldout_channel_stats.no_learning_sq, heldout_channel_stats.count);
        const double channel_heldout_time_shuffle_mse =
            mse(heldout_channel_stats.time_shuffle_sq, heldout_channel_stats.count);
        const double channel_heldout_spatial_shuffle_mse =
            mse(heldout_channel_stats.spatial_shuffle_sq, heldout_channel_stats.count);

        const auto appendForPrefix = [&](const std::string &metric_prefix) {
            appendMetric(metric_prefix, "target_channel_required", target_specs[target_channel].required ? 1.0 : 0.0);
            appendMetric(metric_prefix, "train_prediction_count", static_cast<double>(train_channel_stats.count));
            appendMetric(metric_prefix, "heldout_prediction_count", static_cast<double>(heldout_channel_stats.count));
            appendMetric(metric_prefix, "model_mse_norm", channel_heldout_model_mse);
            appendMetric(metric_prefix, "model_raw_mse_norm", channel_heldout_model_mse);
            appendMetric(metric_prefix, "model_residual_z_mse", channel_heldout_residual_z_mse);
            appendMetric(metric_prefix, "zero_mse_norm", channel_heldout_no_learning_mse);
            appendMetric(metric_prefix, "persistence_mse_norm", channel_heldout_persistence_mse);
            appendMetric(metric_prefix, "train_model_mse_norm", channel_train_model_mse);
            appendMetric(metric_prefix, "train_model_residual_z_mse", channel_train_residual_z_mse);
            appendMetric(metric_prefix, "train_persistence_mse_norm", mse(train_channel_stats.persistence_sq, train_channel_stats.count));
            appendMetric(metric_prefix, "train_train_mean_mse_norm", mse(train_channel_stats.train_mean_sq, train_channel_stats.count));
            appendMetric(metric_prefix, "train_no_learning_mse_norm", mse(train_channel_stats.no_learning_sq, train_channel_stats.count));
            appendMetric(metric_prefix, "heldout_model_mse_norm", channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_model_raw_mse_norm", channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_model_residual_z_mse", channel_heldout_residual_z_mse);
            appendMetric(metric_prefix, "heldout_persistence_mse_norm", channel_heldout_persistence_mse);
            appendMetric(metric_prefix, "heldout_train_mean_mse_norm", channel_heldout_train_mean_mse);
            appendMetric(metric_prefix, "heldout_no_learning_mse_norm", channel_heldout_no_learning_mse);
            appendMetric(metric_prefix, "heldout_temporal_block_shift_mse_norm", channel_heldout_time_shuffle_mse);
            appendMetric(metric_prefix, "heldout_spatial_tile_shuffle_mse_norm", channel_heldout_spatial_shuffle_mse);
            appendMetric(metric_prefix, "heldout_time_shuffle_mse_norm", channel_heldout_time_shuffle_mse);
            appendMetric(metric_prefix, "heldout_spatial_shuffle_mse_norm", channel_heldout_spatial_shuffle_mse);
            appendMetric(metric_prefix, "improvement_vs_zero_norm", channel_heldout_no_learning_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "improvement_vs_persistence_norm", channel_heldout_persistence_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_persistence_norm", channel_heldout_persistence_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_train_mean_norm", channel_heldout_train_mean_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_no_learning_norm", channel_heldout_no_learning_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_temporal_block_shift_norm", channel_heldout_time_shuffle_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_spatial_tile_shuffle_norm", channel_heldout_spatial_shuffle_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_time_shuffle_norm", channel_heldout_time_shuffle_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "heldout_improvement_vs_spatial_shuffle_norm", channel_heldout_spatial_shuffle_mse - channel_heldout_model_mse);
            appendMetric(metric_prefix, "relative_improvement_vs_zero", channel_heldout_no_learning_mse > 0.0 ? ((channel_heldout_no_learning_mse - channel_heldout_model_mse) / channel_heldout_no_learning_mse) : 0.0);
            appendMetric(metric_prefix, "relative_improvement_vs_persistence", channel_heldout_persistence_mse > 0.0 ? ((channel_heldout_persistence_mse - channel_heldout_model_mse) / channel_heldout_persistence_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_persistence", channel_heldout_persistence_mse > 0.0 ? ((channel_heldout_persistence_mse - channel_heldout_model_mse) / channel_heldout_persistence_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_train_mean", channel_heldout_train_mean_mse > 0.0 ? ((channel_heldout_train_mean_mse - channel_heldout_model_mse) / channel_heldout_train_mean_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_no_learning", channel_heldout_no_learning_mse > 0.0 ? ((channel_heldout_no_learning_mse - channel_heldout_model_mse) / channel_heldout_no_learning_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_temporal_block_shift", channel_heldout_time_shuffle_mse > 0.0 ? ((channel_heldout_time_shuffle_mse - channel_heldout_model_mse) / channel_heldout_time_shuffle_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_spatial_tile_shuffle", channel_heldout_spatial_shuffle_mse > 0.0 ? ((channel_heldout_spatial_shuffle_mse - channel_heldout_model_mse) / channel_heldout_spatial_shuffle_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_time_shuffle", channel_heldout_time_shuffle_mse > 0.0 ? ((channel_heldout_time_shuffle_mse - channel_heldout_model_mse) / channel_heldout_time_shuffle_mse) : 0.0);
            appendMetric(metric_prefix, "heldout_relative_improvement_vs_spatial_shuffle", channel_heldout_spatial_shuffle_mse > 0.0 ? ((channel_heldout_spatial_shuffle_mse - channel_heldout_model_mse) / channel_heldout_spatial_shuffle_mse) : 0.0);
            appendMetric(metric_prefix, "target_mean_hz", meanHz(heldout_channel_stats.target_rate_sum, heldout_channel_stats.count));
            appendMetric(metric_prefix, "prediction_mean_hz", meanHz(heldout_channel_stats.prediction_rate_sum, heldout_channel_stats.count));
            appendMetric(metric_prefix, "target_std_norm", standardDeviation(heldout_channel_stats.targets));
            appendMetric(metric_prefix, "prediction_std_norm", standardDeviation(heldout_channel_stats.predictions));
            appendMetric(metric_prefix, "heldout_prediction_min_norm", heldout_channel_stats.prediction_min);
            appendMetric(metric_prefix, "heldout_prediction_max_norm", heldout_channel_stats.prediction_max);
            appendMetric(metric_prefix, "mean_tile_prediction_corr", meanRate(tile_correlations_by_channel[target_channel]));
            appendMetric(metric_prefix, "median_tile_prediction_corr", median(tile_correlations_by_channel[target_channel]));
        };

        appendForPrefix(prefix);
        if(append_legacy_names) {
            appendForPrefix("");
        }
    };
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        appendChannelMetrics(target_channel, target_channel == 0u);
    }

    return result;
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

void writeVideoPopulationRatesCsv(
    const std::string &path,
    const std::vector<VideoFrameRecord> &records,
    const std::vector<double> &l4e_rates,
    const std::vector<double> &l23e_rates,
    const std::vector<double> &l23pv_rates,
    const std::vector<double> &l23som_rates)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "repeat_index,frame_index,population,rate_hz,frame_start_ms,frame_end_ms\n";
    const auto writeRows = [&](const std::string &population, const std::vector<double> &rates) {
        if(rates.size() != records.size()) {
            throw std::runtime_error("Video population rate vector does not match frame records.");
        }
        for(std::size_t i = 0; i < records.size(); i++) {
            output << records[i].repeat_index << ","
                   << records[i].frame_index << ","
                   << population << ","
                   << rates[i] << ","
                   << records[i].trial.measure_start_ms << ","
                   << records[i].trial.end_ms << "\n";
        }
    };
    writeRows("l4e", l4e_rates);
    writeRows("l23e", l23e_rates);
    writeRows("l23pv", l23pv_rates);
    writeRows("l23som", l23som_rates);
}

void writeVideoSiteRatesCsv(
    const std::string &path,
    const std::vector<VideoFrameRecord> &records,
    const std::vector<TrialWindow> &video_trials,
    const std::vector<double> &l4e_counts,
    const std::vector<double> &l23e_counts,
    const std::vector<double> &l23pv_counts,
    const std::vector<double> &l23som_counts)
{
    if(records.size() != video_trials.size()) {
        throw std::runtime_error("Video frame records and trial windows do not align.");
    }

    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "repeat_index,frame_index,population,site_id,rate_hz\n";
    const auto writeRows = [&](const std::string &population,
                               const std::vector<double> &counts,
                               unsigned int neurons_per_site) {
        for(std::size_t trial_index = 0; trial_index < records.size(); trial_index++) {
            for(unsigned int site_id = 0; site_id < v1_genn::kSiteCount; site_id++) {
                output << records[trial_index].repeat_index << ","
                       << records[trial_index].frame_index << ","
                       << population << ","
                       << site_id << ","
                       << siteRateFromCounts(counts, video_trials, trial_index, site_id, neurons_per_site)
                       << "\n";
            }
        }
    };
    writeRows("l4e", l4e_counts, v1_genn::kL4EPerSite);
    writeRows("l23e", l23e_counts, v1_genn::kL23EPerSite);
    writeRows("l23pv", l23pv_counts, v1_genn::kL23PVPerSite);
    writeRows("l23som", l23som_counts, v1_genn::kL23SOMPerSite);
}

void writeVideoFrameSummaryCsv(
    const std::string &path,
    const std::vector<VideoFrameRecord> &records,
    const std::vector<double> &l4e_rates,
    const std::vector<double> &l23e_rates,
    const std::vector<double> &l23pv_rates,
    const std::vector<double> &l23som_rates)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    if(l4e_rates.size() != records.size() || l23e_rates.size() != records.size()
       || l23pv_rates.size() != records.size() || l23som_rates.size() != records.size()) {
        throw std::runtime_error("Video frame summary rate vectors do not match frame records.");
    }

    output << std::fixed << std::setprecision(6);
    output << "repeat_index,frame_index,frame_start_ms,frame_end_ms"
           << ",l4e_rate_hz,l23e_rate_hz,l23pv_rate_hz,l23som_rate_hz"
           << ",l4e_drive_min,l4e_drive_mean,l4e_drive_max,l4e_drive_std\n";
    for(std::size_t i = 0; i < records.size(); i++) {
        output << records[i].repeat_index << ","
               << records[i].frame_index << ","
               << records[i].trial.measure_start_ms << ","
               << records[i].trial.end_ms << ","
               << l4e_rates[i] << ","
               << l23e_rates[i] << ","
               << l23pv_rates[i] << ","
               << l23som_rates[i] << ","
               << records[i].drive_min << ","
               << records[i].drive_mean << ","
               << records[i].drive_max << ","
               << records[i].drive_std << "\n";
    }
}

void writeVideoEventPopulationBinsCsv(
    const std::string &path,
    const std::vector<VideoEventTimingRecord> &records,
    unsigned int bin_count,
    double bin_ms,
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
    output << "condition,repeat_index,event_index,frame_index,population,bin_index"
           << ",bin_start_ms,bin_end_ms,rate_hz,spike_count,event_start_ms"
           << ",gray_current,l4e_drive_min,l4e_drive_mean,l4e_drive_max,l4e_drive_std\n";

    const auto writeRows = [&](const std::string &population,
                               const std::vector<double> &counts,
                               unsigned int neuron_count) {
        if(counts.size() != static_cast<std::size_t>(records.size()) * bin_count) {
            throw std::runtime_error("Video event population count vector has unexpected size.");
        }
        const double bin_duration_s = bin_ms / 1000.0;
        for(std::size_t record_index = 0; record_index < records.size(); record_index++) {
            const VideoEventTimingRecord &record = records[record_index];
            const double event_offset_ms = record.event_start_ms - record.trial.start_ms;
            for(unsigned int bin_index = 0; bin_index < bin_count; bin_index++) {
                const double relative_bin_start_ms =
                    (static_cast<double>(bin_index) * bin_ms) - event_offset_ms;
                const double spike_count = counts[(record_index * bin_count) + bin_index];
                const double rate_hz = spike_count / (bin_duration_s * static_cast<double>(neuron_count));
                output << record.condition << ","
                       << record.repeat_index << ","
                       << record.event_index << ","
                       << record.frame_index << ","
                       << population << ","
                       << bin_index << ","
                       << relative_bin_start_ms << ","
                       << (relative_bin_start_ms + bin_ms) << ","
                       << rate_hz << ","
                       << spike_count << ","
                       << record.event_start_ms << ","
                       << record.gray_current << ","
                       << record.drive_min << ","
                       << record.drive_mean << ","
                       << record.drive_max << ","
                       << record.drive_std << "\n";
            }
        }
    };

    writeRows("l4e", l4e_counts, v1_genn::kNumL4E);
    writeRows("l23e", l23e_counts, v1_genn::kNumL23E);
    writeRows("l23pv", l23pv_counts, v1_genn::kNumL23PV);
    writeRows("l23som", l23som_counts, v1_genn::kNumL23SOM);
}

void writeVideoEventSiteBinsCsv(
    const std::string &path,
    const std::vector<VideoEventTimingRecord> &records,
    const std::vector<unsigned int> &site_ids,
    unsigned int bin_count,
    double bin_ms,
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
    output << "condition,repeat_index,event_index,frame_index,population,site_id,bin_index"
           << ",bin_start_ms,bin_end_ms,rate_hz,spike_count,event_start_ms"
           << ",gray_current,l4e_drive_min,l4e_drive_mean,l4e_drive_max,l4e_drive_std\n";

    const auto writeRows = [&](const std::string &population,
                               const std::vector<double> &counts,
                               unsigned int neurons_per_site) {
        const std::size_t expected_size =
            static_cast<std::size_t>(records.size()) * bin_count * site_ids.size();
        if(counts.size() != expected_size) {
            throw std::runtime_error("Video event site count vector has unexpected size.");
        }
        const double bin_duration_s = bin_ms / 1000.0;
        for(std::size_t record_index = 0; record_index < records.size(); record_index++) {
            const VideoEventTimingRecord &record = records[record_index];
            const double event_offset_ms = record.event_start_ms - record.trial.start_ms;
            for(unsigned int bin_index = 0; bin_index < bin_count; bin_index++) {
                const double relative_bin_start_ms =
                    (static_cast<double>(bin_index) * bin_ms) - event_offset_ms;
                for(unsigned int site_export_index = 0; site_export_index < site_ids.size(); site_export_index++) {
                    const std::size_t count_index =
                        ((record_index * bin_count) + bin_index) * site_ids.size() + site_export_index;
                    const double spike_count = counts[count_index];
                    const double rate_hz = spike_count / (bin_duration_s * static_cast<double>(neurons_per_site));
                    output << record.condition << ","
                           << record.repeat_index << ","
                           << record.event_index << ","
                           << record.frame_index << ","
                           << population << ","
                           << site_ids[site_export_index] << ","
                           << bin_index << ","
                           << relative_bin_start_ms << ","
                           << (relative_bin_start_ms + bin_ms) << ","
                           << rate_hz << ","
                           << spike_count << ","
                           << record.event_start_ms << ","
                           << record.gray_current << ","
                           << record.drive_min << ","
                           << record.drive_mean << ","
                           << record.drive_max << ","
                           << record.drive_std << "\n";
                }
            }
        }
    };

    writeRows("l4e", l4e_counts, v1_genn::kL4EPerSite);
    writeRows("l23e", l23e_counts, v1_genn::kL23EPerSite);
    writeRows("l23pv", l23pv_counts, v1_genn::kL23PVPerSite);
    writeRows("l23som", l23som_counts, v1_genn::kL23SOMPerSite);
}

void writeHVAPredictorConfigCsv(
    const std::string &path,
    const HVAPredictorConfig &config,
    const HVAPredictorResult &result)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "metric,value\n";
    output << "enabled," << (config.enabled ? 1.0 : 0.0) << "\n";
    output << "mode_code,1.000000\n";
    output << "host_side_learning,1.000000\n";
    output << "eligibility_trace_enabled,1.000000\n";
    output << "delayed_prediction_error_enabled,1.000000\n";
    output << "lower_v1_frozen,1.000000\n";
    output << "hva_to_v1_connection_count,0.000000\n";
    output << "hva_to_v1_current_enabled,0.000000\n";
    output << "tile_size_sites," << config.tile_size_sites << "\n";
    output << "tile_grid_side," << config.tile_grid_side << "\n";
    output << "tile_count," << (config.tile_grid_side * config.tile_grid_side) << "\n";
    output << "delay_frames," << config.delay_frames << "\n";
    output << "trace_tau_frames," << config.trace_tau_frames << "\n";
    output << "learning_rate," << config.learning_rate << "\n";
    output << "residual_learning_rate," << config.learning_rate << "\n";
    output << "event_learning_rate," << config.event_learning_rate << "\n";
    output << "bias_learning_rate," << config.bias_learning_rate << "\n";
    output << "event_bias_learning_rate," << config.event_bias_learning_rate << "\n";
    output << "weight_decay," << config.weight_decay << "\n";
    output << "event_weight_decay," << config.event_weight_decay << "\n";
    output << "event_residual_gain," << config.event_residual_gain << "\n";
    output << "rate_scale_hz," << config.rate_scale_hz << "\n";
    output << "weight_clip," << config.weight_clip << "\n";
    output << "heldout_fraction," << config.heldout_fraction << "\n";
    output << "local_radius_tiles," << config.local_radius_tiles << "\n";
    output << "topk_local_radius_tiles," << config.topk_local_radius_tiles << "\n";
    output << "training_epochs," << config.training_epochs << "\n";
    output << "event_window_frames," << config.event_window_frames << "\n";
    output << "topk_future_window_frames," << config.topk_future_window_frames << "\n";
    output << "topk_k," << config.topk_k << "\n";
    output << "topk_learning_rate," << config.topk_learning_rate << "\n";
    output << "topk_weight_decay," << config.topk_weight_decay << "\n";
    output << "topk_target_smooth_radius_tiles,"
           << config.topk_target_smooth_radius_tiles << "\n";
    output << "feature_lag_count," << config.feature_lag_count << "\n";
    output << "feature_context_radius_tiles," << config.feature_context_radius_tiles << "\n";
    output << "directional_context_enabled," << (config.directional_context_enabled ? 1.0 : 0.0) << "\n";
    output << "sequence_state_enabled," << (config.sequence_state_enabled ? 1.0 : 0.0) << "\n";
    output << "sequence_state_dim," << config.sequence_state_dim << "\n";
    output << "sequence_state_leak," << config.sequence_state_leak << "\n";
    output << "sequence_state_input_scale," << config.sequence_state_input_scale << "\n";
    output << "sequence_state_neighbor_scale," << config.sequence_state_neighbor_scale << "\n";
    output << "topk_repeat_avg_target_enabled," << (config.topk_repeat_avg_target_enabled ? 1.0 : 0.0) << "\n";
    output << "topk_frequency_balance_enabled," << (config.topk_frequency_balance_enabled ? 1.0 : 0.0) << "\n";
    output << "topk_frequency_balance_floor," << config.topk_frequency_balance_floor << "\n";
    output << "event_threshold_quantile," << config.event_threshold_quantile << "\n";
    output << "event_threshold_min_hz," << config.event_threshold_min_hz << "\n";
    output << "event_min_train_positive_count," << config.event_min_train_positive_count << "\n";
    for(const auto &metric : result.metrics) {
        output << metric.first << "," << metric.second << "\n";
    }
}

void writeHVAPredictorRatesCsv(
    const std::string &path,
    const std::vector<HVAPredictorRateRow> &rows)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "sample_index,repeat_index,frame_index,tile_id,tile_x,tile_y,"
           << "l23e_rate_hz,state_norm,eligibility_trace,"
           << "trace_fast,trace_medium,trace_slow,derivative\n";
    for(const HVAPredictorRateRow &row : rows) {
        output << row.sample_index << ","
               << row.repeat_index << ","
               << row.frame_index << ","
               << row.tile_id << ","
               << row.tile_x << ","
               << row.tile_y << ","
               << row.l23e_rate_hz << ","
               << row.state_norm << ","
               << row.eligibility_trace << ","
               << row.trace_fast << ","
               << row.trace_medium << ","
               << row.trace_slow << ","
               << row.derivative << "\n";
    }
}

void writeHVAPredictorEventTilesCsv(
    const std::string &path,
    const std::vector<HVAPredictorEventTileRow> &rows)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "target_channel_index,target_channel,tile_id,tile_x,tile_y,"
           << "threshold_norm,threshold_hz,train_count,train_positive_count,"
           << "train_negative_count,heldout_count,heldout_positive_count,"
           << "train_positive_fraction,heldout_positive_fraction,selected\n";
    for(const HVAPredictorEventTileRow &row : rows) {
        output << row.target_channel_index << ","
               << row.target_channel << ","
               << row.tile_id << ","
               << row.tile_x << ","
               << row.tile_y << ","
               << row.threshold_norm << ","
               << row.threshold_hz << ","
               << row.train_count << ","
               << row.train_positive_count << ","
               << row.train_negative_count << ","
               << row.heldout_count << ","
               << row.heldout_positive_count << ","
               << row.train_positive_fraction << ","
               << row.heldout_positive_fraction << ","
               << (row.selected ? 1 : 0) << "\n";
    }
}

void writeHVAPredictorPredictionsCsv(
    const std::string &path,
    const std::vector<HVAPredictorPredictionRow> &rows)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "prediction_index,repeat_index,frame_index,target_frame_index,"
           << "target_channel_index,target_channel,tile_id,tile_x,tile_y,"
           << "split,learning_update_applied,"
           << "current_state_norm,target_state_norm,predicted_state_norm,"
           << "target_residual_norm,predicted_residual_norm,"
           << "target_residual_z,predicted_residual_z,"
           << "train_residual_mean_norm,train_residual_std_norm,"
           << "persistence_pred_state_norm,train_mean_pred_state_norm,no_learning_pred_state_norm,"
           << "temporal_block_shift_pred_state_norm,spatial_tile_shuffle_pred_state_norm,"
           << "target_rate_hz,predicted_rate_hz,error_rate_hz,abs_error_rate_hz,"
           << "event_window_target_state_norm,event_threshold_norm,event_tile_selected,"
           << "target_event,single_frame_target_event,predicted_event_prob,"
           << "persistence_event_prob,train_event_rate,no_learning_event_prob,"
           << "temporal_block_shift_event_prob,spatial_tile_shuffle_event_prob,event_error,"
           << "topk_target_value_norm,topk_target,topk_sample_valid,"
           << "topk_model_score,topk_model_prob,topk_persistence_score,"
           << "topk_train_frequency_score,topk_no_learning_score,"
           << "topk_temporal_block_shift_score,topk_spatial_tile_shuffle_score\n";
    for(const HVAPredictorPredictionRow &row : rows) {
        output << row.prediction_index << ","
               << row.repeat_index << ","
               << row.frame_index << ","
               << row.target_frame_index << ","
               << row.target_channel_index << ","
               << row.target_channel << ","
               << row.tile_id << ","
               << row.tile_x << ","
               << row.tile_y << ","
               << row.split << ","
               << (row.learning_update_applied ? 1 : 0) << ","
               << row.current_state_norm << ","
               << row.target_state_norm << ","
               << row.predicted_state_norm << ","
               << row.target_residual_norm << ","
               << row.predicted_residual_norm << ","
               << row.target_residual_z << ","
               << row.predicted_residual_z << ","
               << row.train_residual_mean_norm << ","
               << row.train_residual_std_norm << ","
               << row.persistence_pred_state_norm << ","
               << row.train_mean_pred_state_norm << ","
               << row.no_learning_pred_state_norm << ","
               << row.temporal_block_shift_pred_state_norm << ","
               << row.spatial_tile_shuffle_pred_state_norm << ","
               << row.target_rate_hz << ","
               << row.predicted_rate_hz << ","
               << row.error_rate_hz << ","
               << std::fabs(row.error_rate_hz) << ","
               << row.event_window_target_state_norm << ","
               << row.event_threshold_norm << ","
               << (row.event_tile_selected ? 1 : 0) << ","
               << row.target_event << ","
               << row.single_frame_target_event << ","
               << row.predicted_event_prob << ","
               << row.persistence_event_prob << ","
               << row.train_event_rate << ","
               << row.no_learning_event_prob << ","
               << row.temporal_block_shift_event_prob << ","
               << row.spatial_tile_shuffle_event_prob << ","
               << row.event_error << ","
               << row.topk_target_value_norm << ","
               << (row.topk_target ? 1 : 0) << ","
               << (row.topk_sample_valid ? 1 : 0) << ","
               << row.topk_model_score << ","
               << row.topk_model_prob << ","
               << row.topk_persistence_score << ","
               << row.topk_train_frequency_score << ","
               << row.topk_no_learning_score << ","
               << row.topk_temporal_block_shift_score << ","
               << row.topk_spatial_tile_shuffle_score << "\n";
    }
}

void writeHVAPredictorMetricsCsv(
    const std::string &path,
    const HVAPredictorResult &result)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "metric,value\n";
    for(const auto &metric : result.metrics) {
        output << metric.first << "," << metric.second << "\n";
    }
}

void writeHVAPredictorWeightsCsv(
    const std::string &path,
    const HVAPredictorConfig &config,
    const HVAPredictorResult &result)
{
    const unsigned int tile_count = config.tile_grid_side * config.tile_grid_side;
    const unsigned int target_channel_count = static_cast<unsigned int>(result.target_channels.size());
    const unsigned int feature_channel_count = hvaPredictorFeatureChannelCount(config);
    const std::size_t pair_count = static_cast<std::size_t>(tile_count) * tile_count;
    const std::size_t expected_pair_count = static_cast<std::size_t>(target_channel_count) * pair_count;
    if(result.weights_before.size() != result.weights_after.size()
       || result.weights_after.size() != expected_pair_count
       || result.readout_weights_after.size()
          != (expected_pair_count * feature_channel_count)
       || result.biases_after.size() != (static_cast<std::size_t>(target_channel_count) * tile_count)) {
        throw std::runtime_error("HVA predictor weight vectors have unexpected size.");
    }

    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "target_channel_index,target_channel,pre_tile_id,post_tile_id,"
           << "pre_tile_x,pre_tile_y,post_tile_x,post_tile_y,"
           << "distance_tiles,manhattan_distance_tiles,w_before,w_after,delta_w,"
           << "w_current_after,w_trace_fast_after,w_trace_medium_after,w_trace_slow_after,"
           << "w_derivative_after,abs_weight_sum_after,post_bias_after\n";
    for(unsigned int target_channel = 0; target_channel < target_channel_count; target_channel++) {
        for(unsigned int post_tile = 0; post_tile < tile_count; post_tile++) {
            for(unsigned int pre_tile = 0; pre_tile < tile_count; pre_tile++) {
                const std::size_t index =
                    (static_cast<std::size_t>(target_channel) * pair_count)
                    + (static_cast<std::size_t>(post_tile) * tile_count)
                    + pre_tile;
                const double w_before = result.weights_before[index];
                const double w_after = result.weights_after[index];
                const std::size_t readout_base = index * feature_channel_count;
                const double w_current = result.readout_weights_after[readout_base + 0u];
                const double w_trace_fast = result.readout_weights_after[readout_base + 1u];
                const double w_trace_medium = result.readout_weights_after[readout_base + 2u];
                const double w_trace_slow = result.readout_weights_after[readout_base + 3u];
                const double w_derivative = result.readout_weights_after[readout_base + 4u];
                double abs_weight_sum = 0.0;
                for(unsigned int feature = 0; feature < feature_channel_count; feature++) {
                    abs_weight_sum += std::fabs(result.readout_weights_after[readout_base + feature]);
                }
                const int dx = static_cast<int>(pre_tile % config.tile_grid_side)
                    - static_cast<int>(post_tile % config.tile_grid_side);
                const int dy = static_cast<int>(pre_tile / config.tile_grid_side)
                    - static_cast<int>(post_tile / config.tile_grid_side);
                const double distance_tiles = std::sqrt(static_cast<double>((dx * dx) + (dy * dy)));
                const unsigned int manhattan_distance =
                    static_cast<unsigned int>(std::abs(dx) + std::abs(dy));
                output << target_channel << ","
                       << result.target_channels[target_channel] << ","
                       << pre_tile << ","
                       << post_tile << ","
                       << (pre_tile % config.tile_grid_side) << ","
                       << (pre_tile / config.tile_grid_side) << ","
                       << (post_tile % config.tile_grid_side) << ","
                       << (post_tile / config.tile_grid_side) << ","
                       << distance_tiles << ","
                       << manhattan_distance << ","
                       << w_before << ","
                       << w_after << ","
                       << (w_after - w_before) << ","
                       << w_current << ","
                       << w_trace_fast << ","
                       << w_trace_medium << ","
                       << w_trace_slow << ","
                       << w_derivative << ","
                       << abs_weight_sum << ","
                       << result.biases_after.at((static_cast<std::size_t>(target_channel) * tile_count) + post_tile) << "\n";
            }
        }
    }
}

void writeMetricRow(std::ofstream &output, const std::string &metric, double value)
{
    output << metric << "," << value << "\n";
}

void writeVideoConsolidationMetricsCsv(
    const std::string &path,
    const VideoConsolidationConfig &config,
    const VideoConsolidationMetrics &metrics)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }
    output << std::fixed << std::setprecision(6);
    output << "metric,value\n";
    writeMetricRow(output, "requested", config.requested ? 1.0 : 0.0);
    writeMetricRow(output, "enabled", config.enabled ? 1.0 : 0.0);
    writeMetricRow(output, "repeat_count", static_cast<double>(config.repeat_count));
    writeMetricRow(output, "frame_start_index", static_cast<double>(config.frame_start_index));
    writeMetricRow(output, "frame_count", static_cast<double>(config.frame_count));
    writeMetricRow(output, "heldout_start_frame", static_cast<double>(config.heldout_start_frame));
    writeMetricRow(output, "heldout_excluded_frame_count", static_cast<double>(config.heldout_excluded_frame_count));
    writeMetricRow(output, "heldout_frames_used", 0.0);
    writeMetricRow(output, "present_frame_drive_only", 1.0);
    writeMetricRow(output, "future_frame_target_used", 0.0);
    writeMetricRow(output, "target_label_used", 0.0);
    writeMetricRow(output, "l23ee_plasticity_enabled", config.l23ee_plasticity_enabled ? 1.0 : 0.0);
    writeMetricRow(output, "inhibitory_homeostasis_enabled", config.inhibitory_homeostasis_enabled ? 1.0 : 0.0);
    writeMetricRow(output, "feedforward_l4_l23_plasticity_enabled", 0.0);
    writeMetricRow(output, "hva_feedback_enabled", 0.0);
    writeMetricRow(output, "pre_hva_stage", config.enabled ? 1.0 : 0.0);
    writeMetricRow(output, "pre_eval_trial_count", static_cast<double>(metrics.pre_eval_trial_count));
    writeMetricRow(output, "consolidation_trial_count", static_cast<double>(metrics.consolidation_trial_count));
    writeMetricRow(output, "post_eval_trial_count", static_cast<double>(metrics.post_eval_trial_count));
    writeMetricRow(output, "pre_l23e_repeat_corr", metrics.pre_l23e_repeat_corr);
    writeMetricRow(output, "post_l23e_repeat_corr", metrics.post_l23e_repeat_corr);
    writeMetricRow(output, "delta_l23e_repeat_corr", metrics.delta_l23e_repeat_corr);
    writeMetricRow(output, "pre_l23e_repeat_top5_overlap", metrics.pre_l23e_repeat_top5_overlap);
    writeMetricRow(output, "post_l23e_repeat_top5_overlap", metrics.post_l23e_repeat_top5_overlap);
    writeMetricRow(output, "delta_l23e_repeat_top5_overlap", metrics.delta_l23e_repeat_top5_overlap);
    writeMetricRow(output, "l4_l23_weight_delta_max", metrics.l4_l23_weight_delta_max);
    writeMetricRow(output, "l23ee_weight_delta_max", metrics.l23ee_weight_delta_max);
    writeMetricRow(output, "l23pv_weight_delta_max", metrics.l23pv_weight_delta_max);
    writeMetricRow(output, "l23som_weight_delta_max", metrics.l23som_weight_delta_max);
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
    const SensoryAssayConfig &sensory_assay_config,
    const VideoReplayConfig &video_replay_config,
    const VideoPVReliabilityConfig &video_pv_reliability_config,
    const VideoSOMReliabilityConfig &video_som_reliability_config,
    const VideoFFReliabilityConfig &video_ff_reliability_config,
    const VideoEventTimingConfig &video_event_timing_config,
    const VideoConsolidationConfig &video_consolidation_config,
    const VideoConsolidationMetrics &video_consolidation_metrics,
    const HVAPredictorConfig &hva_predictor_config,
    const HVAPredictorResult &hva_predictor_result,
    const L23ESOMBroadRecruitmentConfig &l23e_som_broad_recruitment_config)
{
    const double l23_osi_delta = post.l23_median_osi - baseline.l23_median_osi;
    const bool feedforward_orientation_prior_enabled = (l4_l23_orientation_config.bias_strength > 0.0);
    const bool neutral_density_match_active =
        l4_l23_orientation_config.neutral_density_match_enabled
        && l4_l23_orientation_config.bias_strength == 0.0;
    const double l23e_som_broad_estimated_total_extra_fraction =
        l23e_som_broad_recruitment_config.weight_scale
        * ((static_cast<double>(((2u * l23e_som_broad_recruitment_config.radius) + 1u)
                                * ((2u * l23e_som_broad_recruitment_config.radius) + 1u))
            - 1.0)
           / static_cast<double>(((2u * v1_genn::kL23SOMInputRadius) + 1u)
                                 * ((2u * v1_genn::kL23SOMInputRadius) + 1u)));

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
    csv << "video_replay_enabled," << (video_replay_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_frame_count," << video_replay_config.effective_frame_count << "\n";
    csv << "video_requested_frame_count," << video_replay_config.frame_count << "\n";
    csv << "video_max_frames," << video_replay_config.max_frames << "\n";
    csv << "video_repeat_count," << video_replay_config.repeat_count << "\n";
    csv << "video_presentation_count,"
        << (static_cast<std::size_t>(video_replay_config.effective_frame_count)
            * static_cast<std::size_t>(video_replay_config.repeat_count)) << "\n";
    csv << "video_frame_ms," << video_replay_config.frame_ms << "\n";
    csv << "video_feedback_disabled," << (video_replay_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_training_enabled," << (video_consolidation_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_pv_reliability_tuning_enabled,"
        << (video_pv_reliability_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_pv_reliability_output_scale,"
        << video_pv_reliability_config.output_scale << "\n";
    csv << "video_pv_reliability_l23pv_to_l23e_only,"
        << (video_pv_reliability_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_pv_reliability_som_modified,0.000000\n";
    csv << "video_pv_reliability_weight_density_modified,0.000000\n";
    csv << "video_pv_reliability_target_label_used,0.000000\n";
    csv << "video_pv_reliability_future_frame_used,0.000000\n";
    csv << "video_som_reliability_tuning_enabled,"
        << (video_som_reliability_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_som_reliability_output_scale,"
        << video_som_reliability_config.output_scale << "\n";
    csv << "video_som_reliability_l23som_to_l23e_only,"
        << (video_som_reliability_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_som_reliability_pv_modified,0.000000\n";
    csv << "video_som_reliability_som_to_som_modified,0.000000\n";
    csv << "video_som_reliability_weight_density_modified,0.000000\n";
    csv << "video_som_reliability_target_label_used,0.000000\n";
    csv << "video_som_reliability_future_frame_used,0.000000\n";
    csv << "video_ff_reliability_tuning_enabled,"
        << (video_ff_reliability_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_ff_reliability_l4e_l23e_output_scale,"
        << video_ff_reliability_config.output_scale << "\n";
    csv << "video_ff_reliability_l4e_to_l23e_only,"
        << (video_ff_reliability_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_ff_reliability_inhibitory_modified,0.000000\n";
    csv << "video_ff_reliability_weight_density_modified,0.000000\n";
    csv << "video_ff_reliability_target_label_used,0.000000\n";
    csv << "video_ff_reliability_future_frame_used,0.000000\n";
    csv << "video_event_timing_enabled," << (video_event_timing_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_event_frame_count," << video_event_timing_config.effective_event_count << "\n";
    csv << "video_event_repeat_count," << video_event_timing_config.repeat_count << "\n";
    csv << "video_event_gray_control_count," << video_event_timing_config.gray_control_count << "\n";
    csv << "video_event_blank_control_count," << video_event_timing_config.blank_control_count << "\n";
    csv << "video_event_pre_ms," << video_event_timing_config.pre_ms << "\n";
    csv << "video_event_post_ms," << video_event_timing_config.post_ms << "\n";
    csv << "video_event_bin_ms," << video_event_timing_config.bin_ms << "\n";
    csv << "video_event_gray_current," << video_event_timing_config.gray_current << "\n";
    csv << "video_event_gray_from_frame_mean,"
        << (video_event_timing_config.gray_from_frame_mean ? 1.0 : 0.0) << "\n";
    csv << "video_event_feedback_disabled," << (video_event_timing_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "video_event_training_enabled,0.000000\n";
    csv << "lower_v1_video_consolidation_requested,"
        << (video_consolidation_config.requested ? 1.0 : 0.0) << "\n";
    csv << "lower_v1_video_consolidation_enabled,"
        << (video_consolidation_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "lower_v1_video_consolidation_repeat_count,"
        << video_consolidation_config.repeat_count << "\n";
    csv << "lower_v1_video_consolidation_frame_start_index,"
        << video_consolidation_config.frame_start_index << "\n";
    csv << "lower_v1_video_consolidation_frame_count,"
        << video_consolidation_config.frame_count << "\n";
    csv << "lower_v1_video_consolidation_heldout_start_frame,"
        << video_consolidation_config.heldout_start_frame << "\n";
    csv << "lower_v1_video_consolidation_heldout_excluded_frame_count,"
        << video_consolidation_config.heldout_excluded_frame_count << "\n";
    csv << "lower_v1_video_consolidation_heldout_frames_used,0.000000\n";
    csv << "lower_v1_video_consolidation_present_frame_drive_only,1.000000\n";
    csv << "lower_v1_video_consolidation_future_frame_target_used,0.000000\n";
    csv << "lower_v1_video_consolidation_target_label_used,0.000000\n";
    csv << "lower_v1_video_consolidation_l23ee_plasticity_enabled,"
        << (video_consolidation_config.l23ee_plasticity_enabled ? 1.0 : 0.0) << "\n";
    csv << "lower_v1_video_consolidation_inhibitory_homeostasis_enabled,"
        << (video_consolidation_config.inhibitory_homeostasis_enabled ? 1.0 : 0.0) << "\n";
    csv << "lower_v1_video_consolidation_feedforward_l4_l23_plasticity_enabled,0.000000\n";
    csv << "lower_v1_video_consolidation_hva_feedback_enabled,0.000000\n";
    csv << "lower_v1_video_consolidation_pre_hva_stage,"
        << (video_consolidation_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "lower_v1_video_consolidation_pre_l23e_repeat_corr,"
        << video_consolidation_metrics.pre_l23e_repeat_corr << "\n";
    csv << "lower_v1_video_consolidation_post_l23e_repeat_corr,"
        << video_consolidation_metrics.post_l23e_repeat_corr << "\n";
    csv << "lower_v1_video_consolidation_delta_l23e_repeat_corr,"
        << video_consolidation_metrics.delta_l23e_repeat_corr << "\n";
    csv << "lower_v1_video_consolidation_pre_l23e_repeat_top5_overlap,"
        << video_consolidation_metrics.pre_l23e_repeat_top5_overlap << "\n";
    csv << "lower_v1_video_consolidation_post_l23e_repeat_top5_overlap,"
        << video_consolidation_metrics.post_l23e_repeat_top5_overlap << "\n";
    csv << "lower_v1_video_consolidation_delta_l23e_repeat_top5_overlap,"
        << video_consolidation_metrics.delta_l23e_repeat_top5_overlap << "\n";
    csv << "lower_v1_video_consolidation_l4_l23_weight_delta_max,"
        << video_consolidation_metrics.l4_l23_weight_delta_max << "\n";
    csv << "lower_v1_video_consolidation_l23ee_weight_delta_max,"
        << video_consolidation_metrics.l23ee_weight_delta_max << "\n";
    csv << "lower_v1_video_consolidation_l23pv_weight_delta_max,"
        << video_consolidation_metrics.l23pv_weight_delta_max << "\n";
    csv << "lower_v1_video_consolidation_l23som_weight_delta_max,"
        << video_consolidation_metrics.l23som_weight_delta_max << "\n";
    csv << "hva_predictor_enabled," << (hva_predictor_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "hva_predictor_host_side_learning," << (hva_predictor_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "hva_predictor_lower_v1_frozen,1.000000\n";
    csv << "hva_predictor_hva_to_v1_connection_count,0.000000\n";
    csv << "hva_predictor_hva_to_v1_current_enabled,0.000000\n";
    csv << "hva_predictor_tile_size_sites," << hva_predictor_config.tile_size_sites << "\n";
    csv << "hva_predictor_tile_grid_side," << hva_predictor_config.tile_grid_side << "\n";
    csv << "hva_predictor_tile_count,"
        << (hva_predictor_config.tile_grid_side * hva_predictor_config.tile_grid_side) << "\n";
    csv << "hva_predictor_delay_frames," << hva_predictor_config.delay_frames << "\n";
    csv << "hva_predictor_trace_tau_frames," << hva_predictor_config.trace_tau_frames << "\n";
    csv << "hva_predictor_learning_rate," << hva_predictor_config.learning_rate << "\n";
    csv << "hva_predictor_residual_learning_rate," << hva_predictor_config.learning_rate << "\n";
    csv << "hva_predictor_event_learning_rate," << hva_predictor_config.event_learning_rate << "\n";
    csv << "hva_predictor_bias_learning_rate," << hva_predictor_config.bias_learning_rate << "\n";
    csv << "hva_predictor_event_bias_learning_rate,"
        << hva_predictor_config.event_bias_learning_rate << "\n";
    csv << "hva_predictor_weight_decay," << hva_predictor_config.weight_decay << "\n";
    csv << "hva_predictor_event_weight_decay,"
        << hva_predictor_config.event_weight_decay << "\n";
    csv << "hva_predictor_event_residual_gain,"
        << hva_predictor_config.event_residual_gain << "\n";
    csv << "hva_predictor_rate_scale_hz," << hva_predictor_config.rate_scale_hz << "\n";
    csv << "hva_predictor_weight_clip," << hva_predictor_config.weight_clip << "\n";
    csv << "hva_predictor_heldout_fraction," << hva_predictor_config.heldout_fraction << "\n";
    csv << "hva_predictor_local_radius_tiles," << hva_predictor_config.local_radius_tiles << "\n";
    csv << "hva_predictor_topk_local_radius_tiles,"
        << hva_predictor_config.topk_local_radius_tiles << "\n";
    csv << "hva_predictor_training_epochs," << hva_predictor_config.training_epochs << "\n";
    csv << "hva_predictor_event_window_frames," << hva_predictor_config.event_window_frames << "\n";
    csv << "hva_predictor_topk_future_window_frames,"
        << hva_predictor_config.topk_future_window_frames << "\n";
    csv << "hva_predictor_topk_k," << hva_predictor_config.topk_k << "\n";
    csv << "hva_predictor_topk_learning_rate," << hva_predictor_config.topk_learning_rate << "\n";
    csv << "hva_predictor_topk_weight_decay," << hva_predictor_config.topk_weight_decay << "\n";
    csv << "hva_predictor_topk_target_smooth_radius_tiles,"
        << hva_predictor_config.topk_target_smooth_radius_tiles << "\n";
    csv << "hva_predictor_feature_lag_count," << hva_predictor_config.feature_lag_count << "\n";
    csv << "hva_predictor_feature_context_radius_tiles,"
        << hva_predictor_config.feature_context_radius_tiles << "\n";
    csv << "hva_predictor_directional_context_enabled,"
        << (hva_predictor_config.directional_context_enabled ? 1.0 : 0.0) << "\n";
    csv << "hva_predictor_sequence_state_enabled,"
        << (hva_predictor_config.sequence_state_enabled ? 1.0 : 0.0) << "\n";
    csv << "hva_predictor_sequence_state_dim,"
        << hva_predictor_config.sequence_state_dim << "\n";
    csv << "hva_predictor_sequence_state_leak,"
        << hva_predictor_config.sequence_state_leak << "\n";
    csv << "hva_predictor_sequence_state_input_scale,"
        << hva_predictor_config.sequence_state_input_scale << "\n";
    csv << "hva_predictor_sequence_state_neighbor_scale,"
        << hva_predictor_config.sequence_state_neighbor_scale << "\n";
    csv << "hva_predictor_topk_repeat_avg_target_enabled,"
        << (hva_predictor_config.topk_repeat_avg_target_enabled ? 1.0 : 0.0) << "\n";
    csv << "hva_predictor_topk_frequency_balance_enabled,"
        << (hva_predictor_config.topk_frequency_balance_enabled ? 1.0 : 0.0) << "\n";
    csv << "hva_predictor_topk_frequency_balance_floor,"
        << hva_predictor_config.topk_frequency_balance_floor << "\n";
    csv << "hva_predictor_event_threshold_quantile,"
        << hva_predictor_config.event_threshold_quantile << "\n";
    csv << "hva_predictor_event_threshold_min_hz,"
        << hva_predictor_config.event_threshold_min_hz << "\n";
    csv << "hva_predictor_event_min_train_positive_count,"
        << hva_predictor_config.event_min_train_positive_count << "\n";
    for(const auto &metric : hva_predictor_result.metrics) {
        csv << "hva_predictor_" << metric.first << "," << metric.second << "\n";
    }
    csv << "l23e_som_broad_recruitment_enabled,"
        << (l23e_som_broad_recruitment_config.enabled ? 1.0 : 0.0) << "\n";
    csv << "l23e_som_broad_recruitment_radius_sites,"
        << l23e_som_broad_recruitment_config.radius << "\n";
    csv << "l23e_som_broad_recruitment_weight_scale,"
        << l23e_som_broad_recruitment_config.weight_scale << "\n";
    csv << "l23e_som_broad_recruitment_weight,"
        << (v1_genn::kL23EToSOMWeight * l23e_som_broad_recruitment_config.weight_scale) << "\n";
    csv << "l23e_som_broad_recruitment_estimated_total_extra_fraction,"
        << l23e_som_broad_estimated_total_extra_fraction << "\n";
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
    text << "video_replay_enabled="
         << (video_replay_config.enabled ? 1 : 0) << "\n";
    text << "video_frame_count=" << video_replay_config.effective_frame_count << "\n";
    text << "video_requested_frame_count=" << video_replay_config.frame_count << "\n";
    text << "video_max_frames=" << video_replay_config.max_frames << "\n";
    text << "video_repeat_count=" << video_replay_config.repeat_count << "\n";
    text << "video_presentation_count="
         << (static_cast<std::size_t>(video_replay_config.effective_frame_count)
             * static_cast<std::size_t>(video_replay_config.repeat_count)) << "\n";
    text << "video_frame_ms=" << video_replay_config.frame_ms << "\n";
    text << "video_drive_path="
         << (video_replay_config.drive_path.empty() ? "disabled" : video_replay_config.drive_path)
         << "\n";
    text << "video_feedback_disabled="
         << (video_replay_config.enabled ? 1 : 0) << "\n";
    text << "video_training_enabled=" << (video_consolidation_config.enabled ? 1 : 0) << "\n";
    text << "video_pv_reliability_tuning_enabled="
         << (video_pv_reliability_config.enabled ? 1 : 0) << "\n";
    text << "video_pv_reliability_output_scale="
         << video_pv_reliability_config.output_scale << "\n";
    text << "video_pv_reliability_l23pv_to_l23e_only="
         << (video_pv_reliability_config.enabled ? 1 : 0) << "\n";
    text << "video_pv_reliability_som_modified=0\n";
    text << "video_pv_reliability_weight_density_modified=0\n";
    text << "video_pv_reliability_target_label_used=0\n";
    text << "video_pv_reliability_future_frame_used=0\n";
    text << "video_som_reliability_tuning_enabled="
         << (video_som_reliability_config.enabled ? 1 : 0) << "\n";
    text << "video_som_reliability_output_scale="
         << video_som_reliability_config.output_scale << "\n";
    text << "video_som_reliability_l23som_to_l23e_only="
         << (video_som_reliability_config.enabled ? 1 : 0) << "\n";
    text << "video_som_reliability_pv_modified=0\n";
    text << "video_som_reliability_som_to_som_modified=0\n";
    text << "video_som_reliability_weight_density_modified=0\n";
    text << "video_som_reliability_target_label_used=0\n";
    text << "video_som_reliability_future_frame_used=0\n";
    text << "video_ff_reliability_tuning_enabled="
         << (video_ff_reliability_config.enabled ? 1 : 0) << "\n";
    text << "video_ff_reliability_l4e_l23e_output_scale="
         << video_ff_reliability_config.output_scale << "\n";
    text << "video_ff_reliability_l4e_to_l23e_only="
         << (video_ff_reliability_config.enabled ? 1 : 0) << "\n";
    text << "video_ff_reliability_inhibitory_modified=0\n";
    text << "video_ff_reliability_weight_density_modified=0\n";
    text << "video_ff_reliability_target_label_used=0\n";
    text << "video_ff_reliability_future_frame_used=0\n";
    text << "video_event_timing_enabled="
         << (video_event_timing_config.enabled ? 1 : 0) << "\n";
    text << "video_event_frame_count=" << video_event_timing_config.effective_event_count << "\n";
    text << "video_event_repeat_count=" << video_event_timing_config.repeat_count << "\n";
    text << "video_event_gray_control_count=" << video_event_timing_config.gray_control_count << "\n";
    text << "video_event_blank_control_count=" << video_event_timing_config.blank_control_count << "\n";
    text << "video_event_pre_ms=" << video_event_timing_config.pre_ms << "\n";
    text << "video_event_post_ms=" << video_event_timing_config.post_ms << "\n";
    text << "video_event_bin_ms=" << video_event_timing_config.bin_ms << "\n";
    text << "video_event_gray_current=" << video_event_timing_config.gray_current << "\n";
    text << "video_event_gray_from_frame_mean="
         << (video_event_timing_config.gray_from_frame_mean ? 1 : 0) << "\n";
    text << "video_event_feedback_disabled="
         << (video_event_timing_config.enabled ? 1 : 0) << "\n";
    text << "video_event_training_enabled=0\n";
    text << "lower_v1_video_consolidation_requested="
         << (video_consolidation_config.requested ? 1 : 0) << "\n";
    text << "lower_v1_video_consolidation_enabled="
         << (video_consolidation_config.enabled ? 1 : 0) << "\n";
    text << "lower_v1_video_consolidation_repeat_count="
         << video_consolidation_config.repeat_count << "\n";
    text << "lower_v1_video_consolidation_frame_start_index="
         << video_consolidation_config.frame_start_index << "\n";
    text << "lower_v1_video_consolidation_frame_count="
         << video_consolidation_config.frame_count << "\n";
    text << "lower_v1_video_consolidation_heldout_start_frame="
         << video_consolidation_config.heldout_start_frame << "\n";
    text << "lower_v1_video_consolidation_heldout_excluded_frame_count="
         << video_consolidation_config.heldout_excluded_frame_count << "\n";
    text << "lower_v1_video_consolidation_heldout_frames_used=0\n";
    text << "lower_v1_video_consolidation_present_frame_drive_only=1\n";
    text << "lower_v1_video_consolidation_future_frame_target_used=0\n";
    text << "lower_v1_video_consolidation_target_label_used=0\n";
    text << "lower_v1_video_consolidation_pre_l23e_repeat_corr="
         << video_consolidation_metrics.pre_l23e_repeat_corr << "\n";
    text << "lower_v1_video_consolidation_post_l23e_repeat_corr="
         << video_consolidation_metrics.post_l23e_repeat_corr << "\n";
    text << "lower_v1_video_consolidation_pre_l23e_repeat_top5_overlap="
         << video_consolidation_metrics.pre_l23e_repeat_top5_overlap << "\n";
    text << "lower_v1_video_consolidation_post_l23e_repeat_top5_overlap="
         << video_consolidation_metrics.post_l23e_repeat_top5_overlap << "\n";
    text << "hva_predictor_enabled="
         << (hva_predictor_config.enabled ? 1 : 0) << "\n";
    text << "hva_predictor_mode="
         << (hva_predictor_config.enabled
             ? "host_side_l23e_tile_trace_delayed_error_predictor"
             : "disabled")
         << "\n";
    text << "hva_predictor_lower_v1_frozen=1\n";
    text << "hva_predictor_hva_to_v1_connection_count=0\n";
    text << "hva_predictor_hva_to_v1_current_enabled=0\n";
    text << "hva_predictor_tile_size_sites=" << hva_predictor_config.tile_size_sites << "\n";
    text << "hva_predictor_tile_grid_side=" << hva_predictor_config.tile_grid_side << "\n";
    text << "hva_predictor_delay_frames=" << hva_predictor_config.delay_frames << "\n";
    text << "hva_predictor_trace_tau_frames=" << hva_predictor_config.trace_tau_frames << "\n";
    text << "hva_predictor_learning_rate=" << hva_predictor_config.learning_rate << "\n";
    text << "hva_predictor_residual_learning_rate="
         << hva_predictor_config.learning_rate << "\n";
    text << "hva_predictor_event_learning_rate="
         << hva_predictor_config.event_learning_rate << "\n";
    text << "hva_predictor_bias_learning_rate=" << hva_predictor_config.bias_learning_rate << "\n";
    text << "hva_predictor_event_bias_learning_rate="
         << hva_predictor_config.event_bias_learning_rate << "\n";
    text << "hva_predictor_weight_decay=" << hva_predictor_config.weight_decay << "\n";
    text << "hva_predictor_event_weight_decay="
         << hva_predictor_config.event_weight_decay << "\n";
    text << "hva_predictor_event_residual_gain="
         << hva_predictor_config.event_residual_gain << "\n";
    text << "hva_predictor_rate_scale_hz=" << hva_predictor_config.rate_scale_hz << "\n";
    text << "hva_predictor_weight_clip=" << hva_predictor_config.weight_clip << "\n";
    text << "hva_predictor_heldout_fraction=" << hva_predictor_config.heldout_fraction << "\n";
    text << "hva_predictor_local_radius_tiles=" << hva_predictor_config.local_radius_tiles << "\n";
    text << "hva_predictor_topk_local_radius_tiles="
         << hva_predictor_config.topk_local_radius_tiles << "\n";
    text << "hva_predictor_training_epochs=" << hva_predictor_config.training_epochs << "\n";
    text << "hva_predictor_event_window_frames=" << hva_predictor_config.event_window_frames << "\n";
    text << "hva_predictor_topk_future_window_frames="
         << hva_predictor_config.topk_future_window_frames << "\n";
    text << "hva_predictor_topk_k=" << hva_predictor_config.topk_k << "\n";
    text << "hva_predictor_topk_learning_rate="
         << hva_predictor_config.topk_learning_rate << "\n";
    text << "hva_predictor_topk_weight_decay="
         << hva_predictor_config.topk_weight_decay << "\n";
    text << "hva_predictor_topk_target_smooth_radius_tiles="
         << hva_predictor_config.topk_target_smooth_radius_tiles << "\n";
    text << "hva_predictor_feature_lag_count="
         << hva_predictor_config.feature_lag_count << "\n";
    text << "hva_predictor_feature_context_radius_tiles="
         << hva_predictor_config.feature_context_radius_tiles << "\n";
    text << "hva_predictor_directional_context_enabled="
         << (hva_predictor_config.directional_context_enabled ? 1 : 0) << "\n";
    text << "hva_predictor_sequence_state_enabled="
         << (hva_predictor_config.sequence_state_enabled ? 1 : 0) << "\n";
    text << "hva_predictor_sequence_state_dim="
         << hva_predictor_config.sequence_state_dim << "\n";
    text << "hva_predictor_sequence_state_leak="
         << hva_predictor_config.sequence_state_leak << "\n";
    text << "hva_predictor_sequence_state_input_scale="
         << hva_predictor_config.sequence_state_input_scale << "\n";
    text << "hva_predictor_sequence_state_neighbor_scale="
         << hva_predictor_config.sequence_state_neighbor_scale << "\n";
    text << "hva_predictor_topk_repeat_avg_target_enabled="
         << (hva_predictor_config.topk_repeat_avg_target_enabled ? 1 : 0) << "\n";
    text << "hva_predictor_topk_frequency_balance_enabled="
         << (hva_predictor_config.topk_frequency_balance_enabled ? 1 : 0) << "\n";
    text << "hva_predictor_topk_frequency_balance_floor="
         << hva_predictor_config.topk_frequency_balance_floor << "\n";
    text << "hva_predictor_event_threshold_quantile="
         << hva_predictor_config.event_threshold_quantile << "\n";
    text << "hva_predictor_event_threshold_min_hz="
         << hva_predictor_config.event_threshold_min_hz << "\n";
    text << "hva_predictor_event_min_train_positive_count="
         << hva_predictor_config.event_min_train_positive_count << "\n";
    for(const auto &metric : hva_predictor_result.metrics) {
        text << "hva_predictor_" << metric.first << "=" << metric.second << "\n";
    }
    text << "l23e_som_broad_recruitment_enabled="
         << (l23e_som_broad_recruitment_config.enabled ? 1 : 0) << "\n";
    text << "l23e_som_broad_recruitment_radius_sites="
         << l23e_som_broad_recruitment_config.radius << "\n";
    text << "l23e_som_broad_recruitment_weight_scale="
         << l23e_som_broad_recruitment_config.weight_scale << "\n";
    text << "l23e_som_broad_recruitment_weight="
         << (v1_genn::kL23EToSOMWeight * l23e_som_broad_recruitment_config.weight_scale) << "\n";
    text << "l23e_som_broad_recruitment_estimated_total_extra_fraction="
         << l23e_som_broad_estimated_total_extra_fraction << "\n";
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
    const L23ESOMBroadRecruitmentConfig l23e_som_broad_recruitment_config =
        getL23ESOMBroadRecruitmentConfig();
    const auto l23_e_som_broad_recruitment_patch = makeIntersitePatchParameters(
        v1_genn::kL23EPerSite,
        v1_genn::kL23SOMPerSite,
        l23e_som_broad_recruitment_config.radius);
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
    if(l23e_som_broad_recruitment_config.enabled) {
        addLocalIntersiteProjection(
            model,
            "L23E_to_L23SOM_broad_recruitment",
            l23e,
            l23som,
            v1_genn::kL23EToSOMWeight * l23e_som_broad_recruitment_config.weight_scale,
            v1_genn::kExcTauSynMs,
            l23_e_som_broad_recruitment_patch);
    }
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
    const VideoReplayConfig video_replay_config = getVideoReplayConfig();
    const VideoPVReliabilityConfig video_pv_reliability_config =
        getVideoPVReliabilityConfig(video_replay_config);
    const VideoSOMReliabilityConfig video_som_reliability_config =
        getVideoSOMReliabilityConfig(video_replay_config);
    const VideoFFReliabilityConfig video_ff_reliability_config =
        getVideoFFReliabilityConfig(video_replay_config);
    const VideoEventTimingConfig video_event_timing_config =
        getVideoEventTimingConfig(video_replay_config);
    const HVAPredictorConfig hva_predictor_config =
        getHVAPredictorConfig(video_replay_config);
    const VideoConsolidationConfig video_consolidation_config =
        getVideoConsolidationConfig(video_replay_config, hva_predictor_config);
    const L23ESOMBroadRecruitmentConfig l23e_som_broad_recruitment_config =
        getL23ESOMBroadRecruitmentConfig();
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
    const unsigned int video_frame_steps = video_replay_config.enabled
        ? durationToSteps(video_replay_config.frame_ms)
        : 0u;
    const unsigned int video_event_pre_steps = video_event_timing_config.enabled
        ? durationToSteps(video_event_timing_config.pre_ms)
        : 0u;
    const unsigned int video_event_post_steps = video_event_timing_config.enabled
        ? durationToSteps(video_event_timing_config.post_ms)
        : 0u;
    const unsigned int video_event_bin_steps = video_event_timing_config.enabled
        ? durationToSteps(video_event_timing_config.bin_ms)
        : 0u;
    const unsigned int video_event_total_steps = video_event_pre_steps + video_event_post_steps;
    const unsigned int video_event_bin_count = video_event_timing_config.enabled
        ? (video_event_total_steps / video_event_bin_steps)
        : 0u;
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
    if(video_event_timing_config.enabled
       && ((video_event_total_steps % video_event_bin_steps) != 0u
           || (video_event_pre_steps % video_event_bin_steps) != 0u)) {
        throw std::runtime_error("Video event timing pre/post windows must align to V1_VIDEO_EVENT_BIN_MS.");
    }

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
    const std::size_t video_replay_trial_count =
        video_replay_config.enabled
            ? (static_cast<std::size_t>(video_replay_config.effective_frame_count)
               * static_cast<std::size_t>(video_replay_config.repeat_count))
            : 0u;
    const std::size_t video_pre_consolidation_trial_count =
        video_consolidation_config.enabled
            ? (static_cast<std::size_t>(video_consolidation_config.frame_count)
               * static_cast<std::size_t>(video_replay_config.repeat_count))
            : 0u;
    const std::size_t video_consolidation_trial_count =
        video_consolidation_config.enabled
            ? (static_cast<std::size_t>(video_consolidation_config.frame_count)
               * static_cast<std::size_t>(video_consolidation_config.repeat_count))
            : 0u;
    const std::size_t video_event_timing_trial_count =
        video_event_timing_config.enabled
            ? (static_cast<std::size_t>(video_event_timing_config.effective_event_count)
               + static_cast<std::size_t>(video_event_timing_config.gray_control_count)
               + static_cast<std::size_t>(video_event_timing_config.blank_control_count))
              * static_cast<std::size_t>(video_event_timing_config.repeat_count)
            : 0u;
    const std::size_t non_video_trial_count =
        (static_cast<std::size_t>(orientation_count) * sweep_count)
        + multiphase_trial_count
        + orientation_context_trial_count
        + sensory_blank_trial_count
        + sensory_contrast_trial_count;
    const std::size_t total_trial_count =
        non_video_trial_count
        + video_pre_consolidation_trial_count
        + video_consolidation_trial_count
        + video_replay_trial_count
        + video_event_timing_trial_count;
    (void)total_trial_count;
    const std::size_t total_recording_steps =
        (non_video_trial_count * static_cast<std::size_t>(trial_steps))
        + (video_pre_consolidation_trial_count * static_cast<std::size_t>(video_frame_steps))
        + (video_consolidation_trial_count * static_cast<std::size_t>(video_frame_steps))
        + (video_replay_trial_count * static_cast<std::size_t>(video_frame_steps))
        + (video_event_timing_trial_count * static_cast<std::size_t>(video_event_total_steps));
    const std::vector<float> video_drive_frames = loadVideoDriveFrames(video_replay_config);

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

    const auto resetVideoEventTrialState = [&]() {
        resetNeuronTrialState(runtime, l4e, v1_genn::kExcitatoryLIF);
        resetNeuronTrialState(runtime, l4pv, v1_genn::kPVLIF);
        resetNeuronTrialState(runtime, l4som, v1_genn::kSOMLIF);
        resetNeuronTrialState(runtime, l23e, v1_genn::kExcitatoryLIF);
        resetNeuronTrialState(runtime, l23pv, v1_genn::kPVLIF);
        resetNeuronTrialState(runtime, l23som, v1_genn::kSOMLIF);
        resetNeuronTrialState(runtime, l23vip, v1_genn::kVIPLIF);
        resetHomeostaticTraceState(runtime, l23pv_to_l23e);
        resetHomeostaticTraceState(runtime, l23som_to_l23e);
    };

    std::vector<TrialWindow> baseline_trials;
    std::vector<TrialWindow> post_trials;
    std::vector<TrialWindow> multiphase_cell_coverage_trials;
    std::vector<TrialWindow> recurrence_context_trials;
    std::vector<TrialWindow> blank_baseline_trials;
    std::vector<TrialWindow> contrast_sweep_trials;
    std::vector<ContrastTrialRecord> contrast_sweep_records;
    std::vector<TrialWindow> video_pre_consolidation_trials;
    std::vector<VideoFrameRecord> video_pre_consolidation_frame_records;
    std::vector<TrialWindow> video_consolidation_trials;
    std::vector<TrialWindow> video_replay_trials;
    std::vector<VideoFrameRecord> video_frame_records;
    std::vector<VideoEventTimingRecord> video_event_timing_records;
    std::vector<OrientationContextTrialSet> orientation_context_trials;
    baseline_trials.reserve(orientation_count);
    post_trials.reserve(orientation_count);
    multiphase_cell_coverage_trials.reserve(multiphase_trial_count);
    recurrence_context_trials.reserve(orientation_count);
    blank_baseline_trials.reserve(sensory_blank_trial_count);
    contrast_sweep_trials.reserve(sensory_contrast_trial_count);
    contrast_sweep_records.reserve(sensory_contrast_trial_count);
    video_pre_consolidation_trials.reserve(video_pre_consolidation_trial_count);
    video_pre_consolidation_frame_records.reserve(video_pre_consolidation_trial_count);
    video_consolidation_trials.reserve(video_consolidation_trial_count);
    video_replay_trials.reserve(video_replay_trial_count);
    video_frame_records.reserve(video_replay_trial_count);
    video_event_timing_records.reserve(video_event_timing_trial_count);
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

    auto runVideoBlock = [&](std::vector<TrialWindow> *trials,
                             std::vector<VideoFrameRecord> *records,
                             unsigned int repeat_count,
                             unsigned int frame_start_index,
                             unsigned int frame_count,
                             bool recurrent_learning,
                             bool inhibitory_learning) {
        if(!video_replay_config.enabled) {
            return;
        }
        if(l4e_i_ext.getCount() != v1_genn::kNumL4E) {
            throw std::runtime_error("L4E Iext size does not match video drive frame size.");
        }
        if(frame_count == 0u || frame_start_index + frame_count > video_replay_config.effective_frame_count) {
            throw std::runtime_error("Video block frame range is outside the loaded drive frames.");
        }
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(
            l23e_to_l23e,
            "Aplus",
            (recurrent_learning && l23ee_plasticity_enabled) ? l23ee_stdp_aplus : 0.0);
        runtime.setDynamicParamValue(
            l23e_to_l23e,
            "Aminus",
            (recurrent_learning && l23ee_plasticity_enabled) ? l23ee_stdp_aminus : 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "TargetHz", l23pv_homeostatic_target_hz);
        runtime.setDynamicParamValue(l23som_to_l23e, "TargetHz", l23som_homeostatic_target_hz);
        runtime.setDynamicParamValue(
            l23pv_to_l23e,
            "Eta",
            (inhibitory_learning && l23pv_homeostatic_enabled) ? l23pv_homeostatic_eta : 0.0);
        runtime.setDynamicParamValue(
            l23som_to_l23e,
            "Eta",
            (inhibitory_learning && l23som_homeostatic_enabled) ? l23som_homeostatic_eta : 0.0);

        const bool apply_pv_reliability_scale =
            video_pv_reliability_config.enabled
            && !recurrent_learning
            && !inhibitory_learning
            && video_pv_reliability_config.output_scale != 1.0;
        const bool apply_som_reliability_scale =
            video_som_reliability_config.enabled
            && !recurrent_learning
            && !inhibitory_learning
            && video_som_reliability_config.output_scale != 1.0;
        const bool apply_ff_reliability_scale =
            video_ff_reliability_config.enabled
            && !recurrent_learning
            && !inhibitory_learning
            && video_ff_reliability_config.output_scale != 1.0;
        std::vector<float> l23pv_to_l23e_weights_before_reliability_scale;
        if(apply_pv_reliability_scale) {
            // Non-plastic video replay can use a transient PV gain reduction without
            // changing learned weights or connection density.
            l23pv_to_l23e_weights_before_reliability_scale =
                copyWeights(runtime, l23pv_to_l23e);
            scaleSynapseWeights(
                runtime,
                l23pv_to_l23e,
                video_pv_reliability_config.output_scale);
        }
        std::vector<float> l23som_to_l23e_weights_before_reliability_scale;
        if(apply_som_reliability_scale) {
            // Keep the SOM reliability rebalance transient and pathway-specific.
            l23som_to_l23e_weights_before_reliability_scale =
                copyWeights(runtime, l23som_to_l23e);
            scaleSynapseWeights(
                runtime,
                l23som_to_l23e,
                video_som_reliability_config.output_scale);
        }
        std::vector<float> l4e_to_l23e_weights_before_reliability_scale;
        if(apply_ff_reliability_scale) {
            // Feedforward replay gain is uniform and transient; trained weights are restored.
            l4e_to_l23e_weights_before_reliability_scale =
                copyWeights(runtime, l4e_to_l23e);
            scaleSynapseWeights(
                runtime,
                l4e_to_l23e,
                video_ff_reliability_config.output_scale);
        }

        const std::size_t frame_size = v1_genn::kNumL4E;
        for(unsigned int repeat_index = 0; repeat_index < repeat_count; repeat_index++) {
            for(unsigned int frame_offset = 0; frame_offset < frame_count; frame_offset++) {
                const unsigned int frame_index = frame_start_index + frame_offset;
                const std::size_t offset = static_cast<std::size_t>(frame_index) * frame_size;
                std::copy(
                    video_drive_frames.data() + offset,
                    video_drive_frames.data() + offset + frame_size,
                    l4e_i_ext_host);
                l4e_i_ext.pushToDevice();

                const double trial_start_ms = runtime.getTime();
                TrialWindow trial{
                    0.0,
                    0.0,
                    trial_start_ms,
                    trial_start_ms,
                    trial_start_ms + (static_cast<double>(video_frame_steps) * v1_genn::kDtMs),
                };
                if(trials != nullptr) {
                    trials->push_back(trial);
                }
                if(records != nullptr) {
                    records->push_back(summarizeVideoDriveFrame(
                        video_drive_frames,
                        repeat_index,
                        frame_index,
                        trial));
                }
                for(unsigned int step = 0; step < video_frame_steps; step++) {
                    runtime.stepTime();
                }
            }
        }

        if(apply_pv_reliability_scale) {
            setSynapseWeights(
                runtime,
                l23pv_to_l23e,
                l23pv_to_l23e_weights_before_reliability_scale);
        }
        if(apply_som_reliability_scale) {
            setSynapseWeights(
                runtime,
                l23som_to_l23e,
                l23som_to_l23e_weights_before_reliability_scale);
        }
        if(apply_ff_reliability_scale) {
            setSynapseWeights(
                runtime,
                l4e_to_l23e,
                l4e_to_l23e_weights_before_reliability_scale);
        }
    };

    auto runVideoReplay = [&]() {
        runVideoBlock(
            &video_replay_trials,
            &video_frame_records,
            video_replay_config.repeat_count,
            0u,
            video_replay_config.effective_frame_count,
            false,
            false);
    };

    auto runVideoPreConsolidationReplay = [&]() {
        runVideoBlock(
            &video_pre_consolidation_trials,
            &video_pre_consolidation_frame_records,
            video_replay_config.repeat_count,
            video_consolidation_config.frame_start_index,
            video_consolidation_config.frame_count,
            false,
            false);
    };

    auto runVideoConsolidation = [&]() {
        runVideoBlock(
            &video_consolidation_trials,
            nullptr,
            video_consolidation_config.repeat_count,
            video_consolidation_config.frame_start_index,
            video_consolidation_config.frame_count,
            video_consolidation_config.l23ee_plasticity_enabled,
            video_consolidation_config.inhibitory_homeostasis_enabled);
    };

    auto runVideoEventTiming = [&]() {
        if(!video_event_timing_config.enabled) {
            return;
        }
        if(l4e_i_ext.getCount() != v1_genn::kNumL4E) {
            throw std::runtime_error("L4E Iext size does not match video event timing drive frame size.");
        }
        runtime.setDynamicParamValue(l4e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l4e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aplus", 0.0);
        runtime.setDynamicParamValue(l23e_to_l23e, "Aminus", 0.0);
        runtime.setDynamicParamValue(l23pv_to_l23e, "Eta", 0.0);
        runtime.setDynamicParamValue(l23som_to_l23e, "Eta", 0.0);

        const std::size_t frame_size = v1_genn::kNumL4E;
        const auto frameMean = [&](unsigned int frame_index) {
            const std::size_t offset = static_cast<std::size_t>(frame_index) * frame_size;
            if(offset + frame_size > video_drive_frames.size()) {
                throw std::runtime_error("Video event timing frame index exceeds loaded drive data.");
            }
            double sum = 0.0;
            for(std::size_t i = 0; i < frame_size; i++) {
                sum += static_cast<double>(video_drive_frames[offset + i]);
            }
            return sum / static_cast<double>(frame_size);
        };
        const auto fillConstantL4ECurrent = [&](double current) {
            std::fill(l4e_i_ext_host, l4e_i_ext_host + l4e_i_ext.getCount(), static_cast<float>(current));
            l4e_i_ext.pushToDevice();
        };
        const auto pushFrameDrive = [&](unsigned int frame_index) {
            const std::size_t offset = static_cast<std::size_t>(frame_index) * frame_size;
            if(offset + frame_size > video_drive_frames.size()) {
                throw std::runtime_error("Video event timing frame index exceeds loaded drive data.");
            }
            std::copy(
                video_drive_frames.data() + offset,
                video_drive_frames.data() + offset + frame_size,
                l4e_i_ext_host);
            l4e_i_ext.pushToDevice();
        };

        const auto runTimingTrial = [&](const std::string &condition,
                                        unsigned int repeat_index,
                                        unsigned int event_index,
                                        unsigned int frame_index,
                                        bool post_uses_frame,
                                        bool blank_control) {
            const double gray_current = video_event_timing_config.gray_from_frame_mean
                ? frameMean(frame_index)
                : video_event_timing_config.gray_current;
            fillConstantL4ECurrent(blank_control ? 0.0 : gray_current);
            if(video_consolidation_config.enabled) {
                resetVideoEventTrialState();
            }

            const double trial_start_ms = runtime.getTime();
            for(unsigned int step = 0; step < video_event_pre_steps; step++) {
                runtime.stepTime();
            }

            const double event_start_ms = runtime.getTime();
            if(post_uses_frame) {
                pushFrameDrive(frame_index);
            }
            else {
                fillConstantL4ECurrent(blank_control ? 0.0 : gray_current);
            }
            const TrialWindow trial{
                0.0,
                0.0,
                trial_start_ms,
                event_start_ms,
                event_start_ms + (static_cast<double>(video_event_post_steps) * v1_genn::kDtMs),
            };
            video_event_timing_records.push_back(makeVideoEventTimingRecord(
                condition,
                video_drive_frames,
                repeat_index,
                event_index,
                frame_index,
                trial,
                event_start_ms,
                blank_control ? 0.0 : gray_current,
                post_uses_frame));
            for(unsigned int step = 0; step < video_event_post_steps; step++) {
                runtime.stepTime();
            }
        };

        for(unsigned int repeat_index = 0; repeat_index < video_event_timing_config.repeat_count; repeat_index++) {
            for(unsigned int event_index = 0; event_index < video_event_timing_config.effective_event_count; event_index++) {
                runTimingTrial("event", repeat_index, event_index, event_index, true, false);
            }
            for(unsigned int event_index = 0; event_index < video_event_timing_config.gray_control_count; event_index++) {
                runTimingTrial("gray_control", repeat_index, event_index, event_index, false, false);
            }
            for(unsigned int event_index = 0; event_index < video_event_timing_config.blank_control_count; event_index++) {
                runTimingTrial("blank_control", repeat_index, event_index, event_index, false, true);
            }
        }
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
    if(video_consolidation_config.enabled) {
        runVideoPreConsolidationReplay();
        runVideoConsolidation();
    }
    runVideoReplay();
    std::vector<float> l4_l23_weights_after_video_consolidation;
    std::vector<float> l23ee_weights_after_video_consolidation;
    std::vector<float> l23pv_weights_after_video_consolidation;
    std::vector<float> l23som_weights_after_video_consolidation;
    if(video_consolidation_config.enabled) {
        l4_l23_weights_after_video_consolidation = copyWeights(runtime, l4e_to_l23e);
        l23ee_weights_after_video_consolidation = copyWeights(runtime, l23e_to_l23e);
        l23pv_weights_after_video_consolidation = copyWeights(runtime, l23pv_to_l23e);
        l23som_weights_after_video_consolidation = copyWeights(runtime, l23som_to_l23e);
    }
    runVideoEventTiming();

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
    const std::vector<double> video_pre_consolidation_l23e_site_counts =
        video_consolidation_config.enabled
            ? countSiteSpikesForTrials(l23e_recordings.at(0), video_pre_consolidation_trials, v1_genn::kL23EPerSite)
            : std::vector<double>();
    const std::vector<double> video_l4e_site_counts =
        video_replay_config.enabled
            ? countSiteSpikesForTrials(l4e_recordings.at(0), video_replay_trials, v1_genn::kL4EPerSite)
            : std::vector<double>();
    const std::vector<double> video_l23e_site_counts =
        video_replay_config.enabled
            ? countSiteSpikesForTrials(l23e_recordings.at(0), video_replay_trials, v1_genn::kL23EPerSite)
            : std::vector<double>();
    const std::vector<double> video_l23pv_site_counts =
        video_replay_config.enabled
            ? countSiteSpikesForTrials(l23pv_recordings.at(0), video_replay_trials, v1_genn::kL23PVPerSite)
            : std::vector<double>();
    const std::vector<double> video_l23som_site_counts =
        video_replay_config.enabled
            ? countSiteSpikesForTrials(l23som_recordings.at(0), video_replay_trials, v1_genn::kL23SOMPerSite)
            : std::vector<double>();
    const std::vector<double> video_l4e_population_rates =
        video_replay_config.enabled
            ? countPopulationRatesForTrials(l4e_recordings.at(0), video_replay_trials, v1_genn::kNumL4E)
            : std::vector<double>();
    const std::vector<double> video_l23e_population_rates =
        video_replay_config.enabled
            ? countPopulationRatesForTrials(l23e_recordings.at(0), video_replay_trials, v1_genn::kNumL23E)
            : std::vector<double>();
    const std::vector<double> video_l23pv_population_rates =
        video_replay_config.enabled
            ? countPopulationRatesForTrials(l23pv_recordings.at(0), video_replay_trials, v1_genn::kNumL23PV)
            : std::vector<double>();
    const std::vector<double> video_l23som_population_rates =
        video_replay_config.enabled
            ? countPopulationRatesForTrials(l23som_recordings.at(0), video_replay_trials, v1_genn::kNumL23SOM)
            : std::vector<double>();
    const double video_consolidation_l4_l23_weight_delta_max =
        video_consolidation_config.enabled
            ? maxAbsDifference(weights_after, l4_l23_weights_after_video_consolidation)
            : 0.0;
    const double video_consolidation_l23ee_weight_delta_max =
        video_consolidation_config.enabled
            ? maxAbsDifference(l23ee_weights_after, l23ee_weights_after_video_consolidation)
            : 0.0;
    const double video_consolidation_l23pv_weight_delta_max =
        video_consolidation_config.enabled
            ? maxAbsDifference(l23pv_weights_after, l23pv_weights_after_video_consolidation)
            : 0.0;
    const double video_consolidation_l23som_weight_delta_max =
        video_consolidation_config.enabled
            ? maxAbsDifference(l23som_weights_after, l23som_weights_after_video_consolidation)
            : 0.0;
    const VideoConsolidationMetrics video_consolidation_metrics = computeVideoConsolidationMetrics(
        video_consolidation_config,
        video_replay_config,
        hva_predictor_config,
        video_pre_consolidation_trials,
        video_pre_consolidation_l23e_site_counts,
        video_replay_trials,
        video_l23e_site_counts,
        video_consolidation_trials,
        video_consolidation_l4_l23_weight_delta_max,
        video_consolidation_l23ee_weight_delta_max,
        video_consolidation_l23pv_weight_delta_max,
        video_consolidation_l23som_weight_delta_max);
    const std::vector<double> video_event_l4e_population_bin_counts =
        video_event_timing_config.enabled
            ? countPopulationSpikesForEventBins(
                l4e_recordings.at(0),
                video_event_timing_records,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l23e_population_bin_counts =
        video_event_timing_config.enabled
            ? countPopulationSpikesForEventBins(
                l23e_recordings.at(0),
                video_event_timing_records,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l23pv_population_bin_counts =
        video_event_timing_config.enabled
            ? countPopulationSpikesForEventBins(
                l23pv_recordings.at(0),
                video_event_timing_records,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l23som_population_bin_counts =
        video_event_timing_config.enabled
            ? countPopulationSpikesForEventBins(
                l23som_recordings.at(0),
                video_event_timing_records,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l4e_site_bin_counts =
        video_event_timing_config.enabled
            ? countSiteSpikesForEventBins(
                l4e_recordings.at(0),
                video_event_timing_records,
                validation_site_config.site_ids,
                v1_genn::kL4EPerSite,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l23e_site_bin_counts =
        video_event_timing_config.enabled
            ? countSiteSpikesForEventBins(
                l23e_recordings.at(0),
                video_event_timing_records,
                validation_site_config.site_ids,
                v1_genn::kL23EPerSite,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l23pv_site_bin_counts =
        video_event_timing_config.enabled
            ? countSiteSpikesForEventBins(
                l23pv_recordings.at(0),
                video_event_timing_records,
                validation_site_config.site_ids,
                v1_genn::kL23PVPerSite,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();
    const std::vector<double> video_event_l23som_site_bin_counts =
        video_event_timing_config.enabled
            ? countSiteSpikesForEventBins(
                l23som_recordings.at(0),
                video_event_timing_records,
                validation_site_config.site_ids,
                v1_genn::kL23SOMPerSite,
                video_event_bin_count,
                video_event_timing_config.bin_ms)
            : std::vector<double>();

    const HVAPredictorResult hva_predictor_result = trainHVAPredictorSidecar(
        hva_predictor_config,
        video_replay_config,
        video_frame_records,
        video_l23e_site_counts);

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
    if(video_replay_config.enabled) {
        writeVideoPopulationRatesCsv(
            output_prefix + "_video_population_rates.csv",
            video_frame_records,
            video_l4e_population_rates,
            video_l23e_population_rates,
            video_l23pv_population_rates,
            video_l23som_population_rates);
        writeVideoSiteRatesCsv(
            output_prefix + "_video_site_rates.csv",
            video_frame_records,
            video_replay_trials,
            video_l4e_site_counts,
            video_l23e_site_counts,
            video_l23pv_site_counts,
            video_l23som_site_counts);
        writeVideoFrameSummaryCsv(
            output_prefix + "_video_frame_summary.csv",
            video_frame_records,
            video_l4e_population_rates,
            video_l23e_population_rates,
            video_l23pv_population_rates,
            video_l23som_population_rates);
    }
    if(video_consolidation_config.requested) {
        writeVideoConsolidationMetricsCsv(
            output_prefix + "_video_consolidation_metrics.csv",
            video_consolidation_config,
            video_consolidation_metrics);
    }
    if(video_event_timing_config.enabled) {
        writeVideoEventPopulationBinsCsv(
            output_prefix + "_video_event_population_bins.csv",
            video_event_timing_records,
            video_event_bin_count,
            video_event_timing_config.bin_ms,
            video_event_l4e_population_bin_counts,
            video_event_l23e_population_bin_counts,
            video_event_l23pv_population_bin_counts,
            video_event_l23som_population_bin_counts);
        writeVideoEventSiteBinsCsv(
            output_prefix + "_video_event_site_bins.csv",
            video_event_timing_records,
            validation_site_config.site_ids,
            video_event_bin_count,
            video_event_timing_config.bin_ms,
            video_event_l4e_site_bin_counts,
            video_event_l23e_site_bin_counts,
            video_event_l23pv_site_bin_counts,
            video_event_l23som_site_bin_counts);
    }
    if(hva_predictor_config.enabled) {
        writeHVAPredictorConfigCsv(
            output_prefix + "_hva_predictor_config.csv",
            hva_predictor_config,
            hva_predictor_result);
        writeHVAPredictorRatesCsv(
            output_prefix + "_hva_predictor_rates.csv",
            hva_predictor_result.rates);
        writeHVAPredictorEventTilesCsv(
            output_prefix + "_hva_predictor_event_tiles.csv",
            hva_predictor_result.event_tiles);
        writeHVAPredictorPredictionsCsv(
            output_prefix + "_hva_predictor_predictions.csv",
            hva_predictor_result.predictions);
        writeHVAPredictorMetricsCsv(
            output_prefix + "_hva_predictor_metrics.csv",
            hva_predictor_result);
        writeHVAPredictorWeightsCsv(
            output_prefix + "_hva_predictor_weights.csv",
            hva_predictor_config,
            hva_predictor_result);
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
        sensory_assay_config,
        video_replay_config,
        video_pv_reliability_config,
        video_som_reliability_config,
        video_ff_reliability_config,
        video_event_timing_config,
        video_consolidation_config,
        video_consolidation_metrics,
        hva_predictor_config,
        hva_predictor_result,
        l23e_som_broad_recruitment_config);
}
