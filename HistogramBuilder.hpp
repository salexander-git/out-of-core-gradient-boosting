#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

// Flat 256-bin histogram used by SplitEvaluator and finalize_level_splits()
struct alignas(64) ThreadLocalHistogram {
    uint32_t count[256] = {0};
    float sum_g[256] = {0.0f};
    float sum_h[256] = {0.0f};

    void reset() {
        for (int b = 0; b < 256; ++b) {
            count[b] = 0;
            sum_g[b] = 0.0f;
            sum_h[b] = 0.0f;
        }
    }

    void add_bin(uint8_t bin, float g, float h) {
        count[bin] += 1;
        sum_g[bin] += g;
        sum_h[bin] += h;
    }

    void merge_from(const ThreadLocalHistogram& other) {
        for (int b = 0; b < 256; ++b) {
            count[b] += other.count[b];
            sum_g[b] += other.sum_g[b];
            sum_h[b] += other.sum_h[b];
        }
    }
};

// Per-feature histogram used during parallel accumulation
struct FeatureHistogram {
    uint32_t count[256] = {0};
    float sum_g[256] = {0.0f};
    float sum_h[256] = {0.0f};

    void add_bin(uint8_t bin, float g, float h) {
        count[bin] += 1;
        sum_g[bin] += g;
        sum_h[bin] += h;
    }

    void merge_from(const FeatureHistogram& other) {
        for (int b = 0; b < 256; ++b) {
            count[b] += other.count[b];
            sum_g[b] += other.sum_g[b];
            sum_h[b] += other.sum_h[b];
        }
    }
};

struct NodeHistogramBlock {
    std::vector<FeatureHistogram> per_feature;

    NodeHistogramBlock() = default;
    explicit NodeHistogramBlock(int num_features) : per_feature(num_features) {}
};

struct ThreadHistogramState {
    std::vector<NodeHistogramBlock> per_node;

    ThreadHistogramState() = default;
    ThreadHistogramState(size_t num_nodes, int num_features) {
        per_node.reserve(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            per_node.emplace_back(num_features);
        }
    }
};