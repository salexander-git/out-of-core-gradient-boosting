#ifndef SPLIT_EVALUATOR_HPP
#define SPLIT_EVALUATOR_HPP

#include "HistogramBuilder.hpp"

// --- UPDATED STRUCT ---
struct SplitResult {
    int split_bin_idx;
    float weight_left;
    float weight_right;
    float gain;
};

class SplitEvaluator {
private:
    float lambda;
    float gamma;
    float min_child_weight;

    float calculate_weight(float sum_g, float sum_h) const {
        return -sum_g / (sum_h + lambda);
    }

    float calculate_score(float sum_g, float sum_h) const {
        return (sum_g * sum_g) / (sum_h + lambda);
    }

public:
    SplitEvaluator(float l = 1.0f, float g = 0.0f, float min_child_w = 1.0f)
        : lambda(l), gamma(g), min_child_weight(min_child_w) {}

    SplitResult evaluate_histogram(const ThreadLocalHistogram& hist) const {
        SplitResult best_split{};
        best_split.split_bin_idx = -1;
        best_split.weight_left = 0.0f;
        best_split.weight_right = 0.0f;
        best_split.gain = 0.0f;

        float total_g = 0.0f;
        float total_h = 0.0f;
        for (int i = 0; i < 256; ++i) {
            total_g += hist.sum_g[i];
            total_h += hist.sum_h[i];
        }

        if (total_h < min_child_weight) {
            return best_split;
        }

        float root_score = calculate_score(total_g, total_h);
        float current_best_gain = 0.0f;

        float left_g = 0.0f, left_h = 0.0f;

        for (int i = 0; i < 255; ++i) {
            left_g += hist.sum_g[i];
            left_h += hist.sum_h[i];

            float right_g = total_g - left_g;
            float right_h = total_h - left_h;

            if (left_h < min_child_weight || right_h < min_child_weight) continue;

            float left_score = calculate_score(left_g, left_h);
            float right_score = calculate_score(right_g, right_h);

            float gain = 0.5f * (left_score + right_score - root_score) - gamma;

            if (gain > current_best_gain) {
                current_best_gain = gain;
                best_split.split_bin_idx = i;
                best_split.weight_left = calculate_weight(left_g, left_h);
                best_split.weight_right = calculate_weight(right_g, right_h);
                best_split.gain = gain;
            }
        }

        return best_split;
    }
};

#endif