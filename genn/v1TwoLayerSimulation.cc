#include "v1TwoLayerGenn_CODE/definitions.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "v1TwoLayerConfig.h"

namespace {

struct Options {
    float duration_ms = v1_genn::kDefaultDurationMs;
    std::string l4e_drive_path;
    std::string output_prefix = "v1_two_layer";
};

void printUsage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [--l4e-drive PATH] [--duration-ms FLOAT] [--output-prefix PREFIX]\n"
        << "  --l4e-drive PATH      Whitespace-delimited file with one static drive value per L4E neuron\n"
        << "  --duration-ms FLOAT   Simulated duration in milliseconds (default "
        << v1_genn::kDefaultDurationMs << ")\n"
        << "  --output-prefix PREF  Prefix for final-voltage CSV outputs\n";
}

Options parseOptions(int argc, char **argv)
{
    Options options;
    for(int i = 1; i < argc; i++) {
        const std::string arg(argv[i]);
        if(arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if(arg == "--l4e-drive" && (i + 1) < argc) {
            options.l4e_drive_path = argv[++i];
            continue;
        }
        if(arg == "--duration-ms" && (i + 1) < argc) {
            options.duration_ms = std::strtof(argv[++i], nullptr);
            continue;
        }
        if(arg == "--output-prefix" && (i + 1) < argc) {
            options.output_prefix = argv[++i];
            continue;
        }

        std::ostringstream message;
        message << "Unrecognized or incomplete option: " << arg;
        throw std::runtime_error(message.str());
    }

    if(options.duration_ms <= 0.0f) {
        throw std::runtime_error("Duration must be positive.");
    }
    return options;
}

std::vector<float> loadStaticDrive(const std::string &path, unsigned int expected_count)
{
    std::vector<float> values(expected_count, 0.0f);
    if(path.empty()) {
        std::cerr
            << "No --l4e-drive file provided; injecting zero L4 drive. "
            << "TODO: replace with exported Gabor drive for real runs.\n";
        return values;
    }

    std::ifstream input(path.c_str());
    if(!input) {
        throw std::runtime_error("Unable to open L4 drive file: " + path);
    }

    std::vector<float> loaded_values;
    float value = 0.0f;
    while(input >> value) {
        loaded_values.push_back(value);
    }

    if(loaded_values.size() != expected_count) {
        std::ostringstream message;
        message << "Expected " << expected_count << " L4E drive values but loaded "
                << loaded_values.size() << " from " << path;
        throw std::runtime_error(message.str());
    }

    return loaded_values;
}

void writeCsv(const std::string &path, const scalar *values, unsigned int count)
{
    std::ofstream output(path.c_str());
    if(!output) {
        throw std::runtime_error("Unable to open output file: " + path);
    }

    for(unsigned int i = 0; i < count; i++) {
        output << values[i] << "\n";
    }
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const Options options = parseOptions(argc, argv);
        const std::vector<float> l4e_drive = loadStaticDrive(options.l4e_drive_path, v1_genn::kNumL4E);

        allocateMem();
        initialize();

        for(unsigned int i = 0; i < v1_genn::kNumL4E; i++) {
            IextL4E[i] = l4e_drive[i];
        }

        initializeSparse();

        while(t < options.duration_ms) {
            stepTime();
        }

        pullL4EStateFromDevice();
        pullL4IStateFromDevice();
        pullL23EStateFromDevice();
        pullL23IStateFromDevice();

        writeCsv(options.output_prefix + "_l4e_final_v.csv", VL4E, v1_genn::kNumL4E);
        writeCsv(options.output_prefix + "_l4i_final_v.csv", VL4I, v1_genn::kNumL4I);
        writeCsv(options.output_prefix + "_l23e_final_v.csv", VL23E, v1_genn::kNumL23E);
        writeCsv(options.output_prefix + "_l23i_final_v.csv", VL23I, v1_genn::kNumL23I);

        std::cout << "Simulated " << t << " ms and wrote final voltages with prefix '"
                  << options.output_prefix << "'.\n";

        freeMem();
        return 0;
    }
    catch(const std::exception &error) {
        std::cerr << "v1TwoLayerGenn failed: " << error.what() << "\n";
        return 1;
    }
}
