#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cmath>

#include "GradientBoostingModel.hpp"

static void print_usage() {
    std::cout << "Usage: ./gb_engine <model_filepath> <feature_1> [feature_2] ... [feature_N]\n";
    std::cout << "Example: ./gb_engine ensemble.gb 0.1 -2.4 3.2 ...\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "[ERROR] Insufficient arguments.\n";
        print_usage();
        return 1;
    }

    std::string filepath = argv[1];

    try {
        GradientBoostingModel model(1);
        model.load_model(filepath);

        int expected_features = model.get_num_features();
        int provided_features = argc - 2;

        if (provided_features != expected_features) {
            std::cerr << "[ERROR] Model expects " << expected_features
                      << " features, but " << provided_features << " were provided.\n";
            return 1;
        }

        std::vector<uint8_t> binned_features;
        binned_features.reserve(static_cast<size_t>(expected_features));

        const float f_min = -10.0f;
        const float f_max = 10.0f;
        const float scale = 255.0f / (f_max - f_min);

        for (int i = 2; i < argc; ++i) {
            float val = std::stof(argv[i]);
            if (!std::isfinite(val)) {
                std::cerr << "[ERROR] Non-finite feature value provided at argument index " << (i - 1) << ".\n";
                return 1;
            }

            if (val < f_min) val = f_min;
            if (val > f_max) val = f_max;

            float mapped = (val - f_min) * scale;
            if (mapped < 0.0f) mapped = 0.0f;
            if (mapped > 255.0f) mapped = 255.0f;

            binned_features.push_back(static_cast<uint8_t>(mapped));
        }

        auto start = std::chrono::high_resolution_clock::now();
        float prediction = model.predict(binned_features.data());
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> exec_time = end - start;

        std::cout << "Prediction: " << prediction << "\n";
        std::cout << "ExecTimeMs: " << exec_time.count() << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
}