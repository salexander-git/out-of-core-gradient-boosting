#pragma once
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>

class OutOfCoreQuantizer {
private:
    std::ifstream file;
    size_t chunk_size;
    float feature_min;
    float feature_max;
    float scale;
    std::vector<float> buffer;
    std::vector<uint8_t> binned_data;

    uint8_t quantize_value(float val) const {
        if (!std::isfinite(val)) {
            throw std::runtime_error("non-finite feature value encountered during quantization");
        }
        if (val < feature_min) val = feature_min;
        if (val > feature_max) val = feature_max;

        float mapped = (val - feature_min) * scale;
        if (mapped < 0.0f) mapped = 0.0f;
        if (mapped > 255.0f) mapped = 255.0f;
        return static_cast<uint8_t>(mapped);
    }

public:
    OutOfCoreQuantizer(const std::string& filepath, size_t chunk_size_val, float f_min, float f_max)
        : chunk_size(chunk_size_val), feature_min(f_min), feature_max(f_max) {

        if (chunk_size == 0) {
            throw std::invalid_argument("chunk_size must be > 0");
        }
        if (!(feature_max > feature_min)) {
            throw std::invalid_argument("feature_max must be greater than feature_min");
        }

        file.open(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open data file: " + filepath);
        }

        scale = 255.0f / (feature_max - feature_min);
        buffer.resize(chunk_size);
        binned_data.resize(chunk_size);
    }

    size_t get_chunk_size() const {
        return chunk_size;
    }

    size_t process_next_chunk() {
        if (!file.is_open()) {
            throw std::runtime_error("quantizer file stream is not open");
        }
        if (file.eof()) return 0;

        const size_t max_floats_by_streamsize =
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float);
        if (chunk_size > max_floats_by_streamsize) {
            throw std::runtime_error("chunk_size exceeds supported stream read size");
        }

        file.read(reinterpret_cast<char*>(buffer.data()),
                  static_cast<std::streamsize>(chunk_size * sizeof(float)));

        std::streamsize raw_bytes = file.gcount();
        if (raw_bytes < 0) {
            throw std::runtime_error("negative byte count reported while reading");
        }

        if ((static_cast<size_t>(raw_bytes) % sizeof(float)) != 0) {
            throw std::runtime_error("input feature file ended with partial float data");
        }

        size_t floats_read = static_cast<size_t>(raw_bytes) / sizeof(float);
        if (floats_read == 0) return 0;

        for (size_t i = 0; i < floats_read; ++i) {
            binned_data[i] = quantize_value(buffer[i]);
        }
        return floats_read;
    }

    size_t process_next_chunk_into(uint8_t* out_buffer) {
        if (!out_buffer) {
            throw std::invalid_argument("process_next_chunk_into received a null out_buffer");
        }
        if (!file.is_open()) {
            throw std::runtime_error("quantizer file stream is not open");
        }
        if (file.eof()) return 0;

        const size_t max_floats_by_streamsize =
            static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float);
        if (chunk_size > max_floats_by_streamsize) {
            throw std::runtime_error("chunk_size exceeds supported stream read size");
        }

        file.read(reinterpret_cast<char*>(buffer.data()),
                  static_cast<std::streamsize>(chunk_size * sizeof(float)));

        std::streamsize raw_bytes = file.gcount();
        if (raw_bytes < 0) {
            throw std::runtime_error("negative byte count reported while reading");
        }

        if ((static_cast<size_t>(raw_bytes) % sizeof(float)) != 0) {
            throw std::runtime_error("input feature file ended with partial float data");
        }

        size_t floats_read = static_cast<size_t>(raw_bytes) / sizeof(float);
        if (floats_read == 0) return 0;

        for (size_t i = 0; i < floats_read; ++i) {
            out_buffer[i] = quantize_value(buffer[i]);
        }
        return floats_read;
    }

    const std::vector<uint8_t>& get_binned_data() const {
        return binned_data;
    }
};