#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "v1TwoLayerConfig.h"

namespace v1_genn {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

constexpr double kOrientationSimilarityThreshold = 0.72;
constexpr double kOrientationDistancePenalty = 0.08;

constexpr double kDefaultStimulusPhaseDeg = 0.0;
constexpr double kDefaultStimulusContrast = 1.0;
constexpr double kDefaultDriveBias = 0.12;
constexpr double kDefaultDriveGain = 6.0;
constexpr double kDefaultSpatialFrequencyCyclesAcrossSheet = 2.5;
constexpr double kDefaultOrientationKappa = 2.0;

inline double wrapOrientationRadians(double theta)
{
    theta = std::fmod(theta, kPi);
    if(theta < 0.0) {
        theta += kPi;
    }
    return theta;
}

inline double circularOrientationDifference(double lhs, double rhs)
{
    const double delta = std::fabs(wrapOrientationRadians(lhs) - wrapOrientationRadians(rhs));
    return std::min(delta, kPi - delta);
}

inline double normalizedSheetCoordinate(unsigned int index, unsigned int side)
{
    return (static_cast<double>(index) + 0.5) / static_cast<double>(side);
}

inline double centeredSheetCoordinate(unsigned int index, unsigned int side)
{
    return (2.0 * normalizedSheetCoordinate(index, side)) - 1.0;
}

inline std::pair<unsigned int, unsigned int> siteIndexToXY(unsigned int site_index, unsigned int side = kSheetSide)
{
    return {site_index % side, site_index / side};
}

inline double sitePreferredOrientation(unsigned int site_x, unsigned int site_y, unsigned int side = kSheetSide)
{
    const double x = normalizedSheetCoordinate(site_x, side);
    const double y = normalizedSheetCoordinate(site_y, side);

    const double field_x =
        std::sin(kTwoPi * x)
        + (0.60 * std::cos(kTwoPi * y))
        + (0.35 * std::sin(kTwoPi * (x + y)));
    const double field_y =
        std::cos(kTwoPi * x)
        - (0.60 * std::sin(kTwoPi * y))
        + (0.35 * std::cos(kTwoPi * (x - y)));

    return wrapOrientationRadians(0.5 * std::atan2(field_y, field_x));
}

inline double sitePreferredOrientationFromIndex(unsigned int site_index, unsigned int side = kSheetSide)
{
    const auto [site_x, site_y] = siteIndexToXY(site_index, side);
    return sitePreferredOrientation(site_x, site_y, side);
}

inline double l4PreferredPhase(unsigned int neuron_within_site)
{
    constexpr std::array<double, 4> kBasePhases{0.0, 0.5 * kPi, kPi, 1.5 * kPi};
    return kBasePhases[neuron_within_site % kBasePhases.size()];
}

inline double orientationTuningGain(double preferred_orientation_rad, double stimulus_orientation_rad, double kappa)
{
    const double delta = circularOrientationDifference(preferred_orientation_rad, stimulus_orientation_rad);
    return std::exp(kappa * (std::cos(2.0 * delta) - 1.0));
}

inline double gratingAxisProjection(unsigned int site_x, unsigned int site_y, double stimulus_orientation_rad, unsigned int side = kSheetSide)
{
    const double x = centeredSheetCoordinate(site_x, side);
    const double y = centeredSheetCoordinate(site_y, side);
    return (x * std::cos(stimulus_orientation_rad)) + (y * std::sin(stimulus_orientation_rad));
}

inline double l4SimpleCellDrive(
    unsigned int site_x,
    unsigned int site_y,
    unsigned int neuron_within_site,
    double stimulus_orientation_rad,
    double stimulus_phase_rad,
    double contrast = kDefaultStimulusContrast,
    double spatial_frequency_cycles_across_sheet = kDefaultSpatialFrequencyCyclesAcrossSheet,
    double gain = kDefaultDriveGain,
    double bias = kDefaultDriveBias,
    double orientation_kappa = kDefaultOrientationKappa,
    unsigned int side = kSheetSide)
{
    const double preferred_orientation = sitePreferredOrientation(site_x, site_y, side);
    const double axis_projection = gratingAxisProjection(site_x, site_y, stimulus_orientation_rad, side);
    const double orientation_gain = orientationTuningGain(preferred_orientation, stimulus_orientation_rad, orientation_kappa);
    const double carrier = std::cos((kTwoPi * spatial_frequency_cycles_across_sheet * axis_projection) + stimulus_phase_rad + l4PreferredPhase(neuron_within_site));
    const double linear_response = contrast * orientation_gain * carrier;
    return bias + (gain * std::max(0.0, linear_response));
}

}  // namespace v1_genn
