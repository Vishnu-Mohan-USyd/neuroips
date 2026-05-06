#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "v1Biology.h"

namespace {

struct Options {
    double orientation_deg = 0.0;
    double phase_deg = v1_genn::kDefaultStimulusPhaseDeg;
    double contrast = v1_genn::kDefaultStimulusContrast;
    double spatial_frequency_cycles = v1_genn::kDefaultSpatialFrequencyCyclesAcrossSheet;
    double gain = v1_genn::kDefaultDriveGain;
    double bias = v1_genn::kDefaultDriveBias;
    double orientation_kappa = v1_genn::kDefaultOrientationKappa;
    std::string output_path;
};

void printUsage(const char *program)
{
    std::cerr
        << "Usage: " << program << " --orientation-deg FLOAT --output PATH [options]\n"
        << "Options:\n"
        << "  --phase-deg FLOAT              Stimulus phase in degrees (default "
        << v1_genn::kDefaultStimulusPhaseDeg << ")\n"
        << "  --contrast FLOAT               Stimulus contrast (default "
        << v1_genn::kDefaultStimulusContrast << ")\n"
        << "  --spatial-frequency FLOAT      Cycles across the sheet (default "
        << v1_genn::kDefaultSpatialFrequencyCyclesAcrossSheet << ")\n"
        << "  --gain FLOAT                   L4 drive gain (default "
        << v1_genn::kDefaultDriveGain << ")\n"
        << "  --bias FLOAT                   L4 drive bias (default "
        << v1_genn::kDefaultDriveBias << ")\n"
        << "  --orientation-kappa FLOAT      Orientation tuning sharpness (default "
        << v1_genn::kDefaultOrientationKappa << ")\n";
}

double parseDouble(const char *argument, const std::string &name)
{
    char *end = nullptr;
    const double value = std::strtod(argument, &end);
    if(end == argument || *end != '\0') {
        throw std::runtime_error("Invalid numeric value for " + name + ": " + argument);
    }
    return value;
}

Options parseOptions(int argc, char **argv)
{
    Options options;
    bool has_orientation = false;

    for(int i = 1; i < argc; i++) {
        const std::string argument(argv[i]);
        if(argument == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if(argument == "--orientation-deg" && (i + 1) < argc) {
            options.orientation_deg = parseDouble(argv[++i], argument);
            has_orientation = true;
            continue;
        }
        if(argument == "--phase-deg" && (i + 1) < argc) {
            options.phase_deg = parseDouble(argv[++i], argument);
            continue;
        }
        if(argument == "--contrast" && (i + 1) < argc) {
            options.contrast = parseDouble(argv[++i], argument);
            continue;
        }
        if(argument == "--spatial-frequency" && (i + 1) < argc) {
            options.spatial_frequency_cycles = parseDouble(argv[++i], argument);
            continue;
        }
        if(argument == "--gain" && (i + 1) < argc) {
            options.gain = parseDouble(argv[++i], argument);
            continue;
        }
        if(argument == "--bias" && (i + 1) < argc) {
            options.bias = parseDouble(argv[++i], argument);
            continue;
        }
        if(argument == "--orientation-kappa" && (i + 1) < argc) {
            options.orientation_kappa = parseDouble(argv[++i], argument);
            continue;
        }
        if(argument == "--output" && (i + 1) < argc) {
            options.output_path = argv[++i];
            continue;
        }

        throw std::runtime_error("Unrecognized or incomplete option: " + argument);
    }

    if(!has_orientation) {
        throw std::runtime_error("Missing required option: --orientation-deg");
    }
    if(options.output_path.empty()) {
        throw std::runtime_error("Missing required option: --output");
    }
    if(options.contrast < 0.0) {
        throw std::runtime_error("Contrast must be non-negative.");
    }
    if(options.spatial_frequency_cycles <= 0.0) {
        throw std::runtime_error("Spatial frequency must be positive.");
    }
    if(options.orientation_kappa < 0.0) {
        throw std::runtime_error("Orientation kappa must be non-negative.");
    }

    return options;
}

std::vector<double> generateDriveVector(const Options &options)
{
    std::vector<double> drive;
    drive.reserve(v1_genn::kNumL4E);

    const double stimulus_orientation_rad = options.orientation_deg * (v1_genn::kPi / 180.0);
    const double stimulus_phase_rad = options.phase_deg * (v1_genn::kPi / 180.0);

    for(unsigned int site_y = 0; site_y < v1_genn::kSheetSide; site_y++) {
        for(unsigned int site_x = 0; site_x < v1_genn::kSheetSide; site_x++) {
            for(unsigned int neuron_within_site = 0; neuron_within_site < v1_genn::kL4EPerSite; neuron_within_site++) {
                drive.push_back(v1_genn::l4SimpleCellDrive(
                    site_x,
                    site_y,
                    neuron_within_site,
                    stimulus_orientation_rad,
                    stimulus_phase_rad,
                    options.contrast,
                    options.spatial_frequency_cycles,
                    options.gain,
                    options.bias,
                    options.orientation_kappa));
            }
        }
    }

    return drive;
}

void writeWhitespaceVector(const std::string &path, const std::vector<double> &values)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    for(std::size_t i = 0; i < values.size(); i++) {
        if(i != 0) {
            output << ' ';
        }
        output << values[i];
    }
    output << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const Options options = parseOptions(argc, argv);
        const std::vector<double> drive = generateDriveVector(options);
        writeWhitespaceVector(options.output_path, drive);

        std::cout << "Wrote " << drive.size() << " L4E drive values to " << options.output_path << ".\n";
        return 0;
    }
    catch(const std::exception &error) {
        std::cerr << "v1GenerateL4Drive failed: " << error.what() << "\n";
        return 1;
    }
}
