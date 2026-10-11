#pragma once
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <limits>

class BinnedDataReader {
private:
    std::ifstream file;
    size_t chunk_size;

public:
    BinnedDataReader(const std::string& filepath, size_t chunk_size_val)
        : chunk_size(chunk_size_val) {
        if (chunk_size == 0) {
            throw std::invalid_argument("chunk_size must be > 0");
        }

        file.open(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open binned data file: " + filepath);
        }
    }

    size_t get_chunk_size() const {
        return chunk_size;
    }

    size_t read_next_chunk(uint8_t* out_buffer) {
        if (!out_buffer) {
            throw std::invalid_argument("read_next_chunk received a null out_buffer");
        }
        if (!file.is_open()) {
            throw std::runtime_error("binned data file stream is not open");
        }
        if (file.eof()) return 0;

        if (chunk_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("chunk_size exceeds supported stream read size");
        }

        file.read(reinterpret_cast<char*>(out_buffer), static_cast<std::streamsize>(chunk_size));
        size_t bytes_read = static_cast<size_t>(file.gcount());
        return bytes_read;
    }
};