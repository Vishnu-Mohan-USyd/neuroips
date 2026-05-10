#pragma once

namespace v1_genn {

constexpr char kModelName[] = "v1TwoLayerGenn";

#ifndef V1_SHEET_SIDE
#define V1_SHEET_SIDE 32
#endif

constexpr unsigned int kSheetSide = V1_SHEET_SIDE;
static_assert(kSheetSide >= 16, "V1 sheet must be large enough for distributed retinotopic audits.");
constexpr unsigned int kSiteCount = kSheetSide * kSheetSide;

constexpr unsigned int kL4EPerSite = 16;
constexpr unsigned int kL4PVPerSite = 3;
constexpr unsigned int kL4SOMPerSite = 1;
constexpr unsigned int kL23EPerSite = 16;
constexpr unsigned int kL23PVPerSite = 2;
constexpr unsigned int kL23SOMPerSite = 1;
constexpr unsigned int kL23VIPPerSite = 1;

constexpr unsigned int kL4IPerSite = kL4PVPerSite + kL4SOMPerSite;
constexpr unsigned int kL23IPerSite = kL23PVPerSite + kL23SOMPerSite + kL23VIPPerSite;

constexpr unsigned int kNumL4E = kSiteCount * kL4EPerSite;
constexpr unsigned int kNumL4PV = kSiteCount * kL4PVPerSite;
constexpr unsigned int kNumL4SOM = kSiteCount * kL4SOMPerSite;
constexpr unsigned int kNumL4I = kNumL4PV + kNumL4SOM;
constexpr unsigned int kNumL23E = kSiteCount * kL23EPerSite;
constexpr unsigned int kNumL23PV = kSiteCount * kL23PVPerSite;
constexpr unsigned int kNumL23SOM = kSiteCount * kL23SOMPerSite;
constexpr unsigned int kNumL23VIP = kSiteCount * kL23VIPPerSite;
constexpr unsigned int kNumL23I = kNumL23PV + kNumL23SOM + kNumL23VIP;

constexpr unsigned int kL4LocalRadius = 1;
constexpr unsigned int kL23LocalRadius = 2;
constexpr unsigned int kL23SOMInputRadius = 3;
constexpr unsigned int kL23SOMOutputRadius = 6;
constexpr unsigned int kFeedforwardRadius = 1;

#ifndef V1_L4_INTERSITE_ENABLE_DEFAULT
#define V1_L4_INTERSITE_ENABLE_DEFAULT 0
#endif
#ifndef V1_L4_INTERSITE_RADIUS
#define V1_L4_INTERSITE_RADIUS 2
#endif
#ifndef V1_L4_INTERSITE_WEIGHT_SCALE
#define V1_L4_INTERSITE_WEIGHT_SCALE 0.12
#endif

constexpr unsigned int kL4IntersiteEnableDefault = V1_L4_INTERSITE_ENABLE_DEFAULT;
constexpr unsigned int kL4IntersiteRadius = V1_L4_INTERSITE_RADIUS;
constexpr double kL4IntersiteWeightScale = V1_L4_INTERSITE_WEIGHT_SCALE;

constexpr double kDtMs = 0.1;
constexpr float kDefaultDurationMs = 250.0f;

struct LIFParameters {
    double c;
    double tau_m_ms;
    double v_rest_mv;
    double v_reset_mv;
    double v_thresh_mv;
    double i_offset_na;
    double tau_refrac_ms;
};

constexpr LIFParameters kExcitatoryLIF{
    0.25,   // C [nF]
    20.0,   // TauM [ms]
    -65.0,  // Vrest [mV]
    -60.0,  // Vreset [mV]
    -50.0,  // Vthresh [mV]
    0.0,    // Ioffset [nA]
    2.0     // TauRefrac [ms]
};

constexpr LIFParameters kPVLIF{
    0.20,   // C [nF]
    8.0,    // TauM [ms]
    -62.0,  // Vrest [mV]
    -55.0,  // Vreset [mV]
    -45.0,  // Vthresh [mV]
    0.0,    // Ioffset [nA]
    1.0     // TauRefrac [ms]
};

constexpr LIFParameters kSOMLIF{
    0.22,   // C [nF]
    18.0,   // TauM [ms]
    -62.0,  // Vrest [mV]
    -56.0,  // Vreset [mV]
    -47.0,  // Vthresh [mV]
    0.0,    // Ioffset [nA]
    2.0     // TauRefrac [ms]
};

constexpr LIFParameters kVIPLIF{
    0.20,   // C [nF]
    15.0,   // TauM [ms]
    -62.0,  // Vrest [mV]
    -55.0,  // Vreset [mV]
    -46.0,  // Vthresh [mV]
    0.0,    // Ioffset [nA]
    1.5     // TauRefrac [ms]
};

constexpr LIFParameters kInhibitoryLIF = kPVLIF;

constexpr double kExcTauSynMs = 5.0;
constexpr double kPVInhTauSynMs = 5.0;
constexpr double kSOMInhTauSynMs = 12.0;
constexpr double kVIPInhTauSynMs = 8.0;
constexpr double kInhTauSynMs = kPVInhTauSynMs;

constexpr double kL4EEWeight = 0.0050;
constexpr double kL4EToPVWeight = 0.0075;
constexpr double kL4EToSOMWeight = 0.0045;
constexpr double kL4PVToEWeight = -0.0175;
constexpr double kL4PVToPVWeight = -0.0150;
constexpr double kL4SOMToEWeight = -0.0060;
constexpr double kL4SOMToPVWeight = -0.0040;

constexpr double kL4EToL23EWeight = 0.0060;
constexpr double kL4EToL23PVWeight = 0.0065;

constexpr double kL23EEWeight = 0.0045;
constexpr double kL23EToPVWeight = 0.0070;
constexpr double kL23EToSOMWeight = 0.0055;
constexpr double kL23EToVIPWeight = 0.0035;
constexpr double kL23PVToEWeight = -0.01875;
constexpr double kL23PVToPVWeight = -0.0150;
constexpr double kL23SOMToEWeight = -0.0100;
constexpr double kL23SOMToPVWeight = -0.0060;
constexpr double kL23SOMToVIPWeight = -0.0060;
constexpr double kL23VIPToSOMWeight = -0.0100;

constexpr double kL4EIWeight = kL4EToPVWeight;
constexpr double kL4IEWeight = kL4PVToEWeight;
constexpr double kL4IIWeight = kL4PVToPVWeight;
constexpr double kL4EToL23IWeight = kL4EToL23PVWeight;
constexpr double kL23EIWeight = kL23EToPVWeight;
constexpr double kL23IEWeight = kL23PVToEWeight;
constexpr double kL23IIWeight = kL23PVToPVWeight;

}  // namespace v1_genn
