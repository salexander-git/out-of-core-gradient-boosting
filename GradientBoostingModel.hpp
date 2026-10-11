#pragma once
#include "Tree.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cmath>

class GradientBoostingModel {
private:
    int num_features;
    int max_depth;
    float min_child_weight;
    float lambda;
    float gamma;
    float learning_rate;
    float base_score;
    std::vector<GradientTree> trees;

    void validate_metadata_or_throw() const {
        if (num_features <= 0) {
            throw std::runtime_error("invalid ensemble num_features");
        }
        if (max_depth <= 0) {
            throw std::runtime_error("invalid ensemble max_depth");
        }
        if (min_child_weight < 0.0f) {
            throw std::runtime_error("invalid ensemble min_child_weight");
        }
        if (!std::isfinite(lambda) || !std::isfinite(gamma) ||
            !std::isfinite(learning_rate) || !std::isfinite(base_score)) {
            throw std::runtime_error("ensemble metadata contains non-finite values");
        }
        if (learning_rate <= 0.0f) {
            throw std::runtime_error("invalid ensemble learning_rate");
        }
    }

public:
    GradientBoostingModel(int features,
                          int depth = 6,
                          float min_weight = 1.0f,
                          float l = 1.0f,
                          float g = 0.0f,
                          float lr = 0.1f,
                          float base = 0.0f)
        : num_features(features),
          max_depth(depth),
          min_child_weight(min_weight),
          lambda(l),
          gamma(g),
          learning_rate(lr),
          base_score(base) {
        if (features <= 0) {
            throw std::invalid_argument("num_features must be > 0");
        }
        if (depth <= 0) {
            throw std::invalid_argument("max_depth must be > 0");
        }
        if (min_weight < 0.0f) {
            throw std::invalid_argument("min_child_weight must be >= 0");
        }
        if (lr <= 0.0f) {
            throw std::invalid_argument("learning_rate must be > 0");
        }
        if (!std::isfinite(l) || !std::isfinite(g) || !std::isfinite(lr) || !std::isfinite(base)) {
            throw std::invalid_argument("ensemble parameters must be finite");
        }
    }

    int get_num_features() const { return num_features; }
    int get_max_depth() const { return max_depth; }
    float get_min_child_weight() const { return min_child_weight; }
    float get_lambda() const { return lambda; }
    float get_gamma() const { return gamma; }
    float get_learning_rate() const { return learning_rate; }
    float get_base_score() const { return base_score; }
    size_t get_num_trees() const { return trees.size(); }
    bool is_initialized() const { return true; }

    void set_base_score(float base) {
        if (!std::isfinite(base)) {
            throw std::invalid_argument("base_score must be finite");
        }
        base_score = base;
    }

    void add_tree(const GradientTree& tree) {
        validate_metadata_or_throw();
        if (!tree.is_initialized()) {
            throw std::runtime_error("cannot add uninitialized tree to ensemble");
        }
        if (tree.get_num_features() != num_features) {
            throw std::runtime_error("tree num_features does not match ensemble num_features");
        }
        trees.push_back(tree);
    }

    const std::vector<GradientTree>& get_trees() const { return trees; }

    float predict_raw_sum(const uint8_t* feature_row) const {
        if (!feature_row) {
            throw std::invalid_argument("predict_raw_sum received a null feature_row");
        }
        validate_metadata_or_throw();

        float sum = 0.0f;
        for (const auto& tree : trees) {
            sum += tree.predict(feature_row);
        }
        return sum;
    }

    float predict(const uint8_t* feature_row) const {
        if (!feature_row) {
            throw std::invalid_argument("predict received a null feature_row");
        }
        validate_metadata_or_throw();

        float pred = base_score;
        for (const auto& tree : trees) {
            pred += learning_rate * tree.predict(feature_row);
        }
        return pred;
    }

    void predict_chunk(const uint8_t* binned_data, size_t num_rows, float* out_preds) const {
        if (!binned_data || !out_preds) {
            throw std::invalid_argument("predict_chunk received a null pointer");
        }
        validate_metadata_or_throw();

        for (size_t i = 0; i < num_rows; ++i) {
            const uint8_t* row = binned_data + i * static_cast<size_t>(num_features);
            out_preds[i] = predict(row);
        }
    }

    void save_model(const std::string& filepath) const {
        validate_metadata_or_throw();

        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open ensemble model file for writing: " + filepath);
        }

        const uint32_t magic = 0x47424531; // "GBE1"
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        out.write(reinterpret_cast<const char*>(&num_features), sizeof(num_features));
        out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
        out.write(reinterpret_cast<const char*>(&min_child_weight), sizeof(min_child_weight));
        out.write(reinterpret_cast<const char*>(&lambda), sizeof(lambda));
        out.write(reinterpret_cast<const char*>(&gamma), sizeof(gamma));
        out.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
        out.write(reinterpret_cast<const char*>(&base_score), sizeof(base_score));

        size_t num_trees = trees.size();
        out.write(reinterpret_cast<const char*>(&num_trees), sizeof(num_trees));

        for (const auto& tree : trees) {
            const auto& nodes = tree.get_nodes();
            size_t num_nodes = nodes.size();
            if (num_nodes == 0) {
                throw std::runtime_error("cannot save ensemble containing empty tree");
            }
            out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
            out.write(reinterpret_cast<const char*>(nodes.data()), num_nodes * sizeof(TreeNode));
        }

        if (!out) {
            throw std::runtime_error("failed while writing ensemble model file: " + filepath);
        }
    }

    void load_model(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open ensemble model file for reading: " + filepath);
        }

        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (!in || magic != 0x47424531) {
            throw std::runtime_error("invalid or unsupported ensemble model file: " + filepath);
        }

        in.read(reinterpret_cast<char*>(&num_features), sizeof(num_features));
        in.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
        in.read(reinterpret_cast<char*>(&min_child_weight), sizeof(min_child_weight));
        in.read(reinterpret_cast<char*>(&lambda), sizeof(lambda));
        in.read(reinterpret_cast<char*>(&gamma), sizeof(gamma));
        in.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
        in.read(reinterpret_cast<char*>(&base_score), sizeof(base_score));

        validate_metadata_or_throw();

        size_t num_trees = 0;
        in.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));
        if (!in) {
            throw std::runtime_error("failed reading ensemble tree count from: " + filepath);
        }

        trees.clear();
        trees.reserve(num_trees);

        for (size_t t = 0; t < num_trees; ++t) {
            size_t num_nodes = 0;
            in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
            if (!in || num_nodes == 0) {
                throw std::runtime_error("invalid tree node count in ensemble file: " + filepath);
            }

            std::vector<TreeNode> nodes(num_nodes);
            in.read(reinterpret_cast<char*>(nodes.data()), num_nodes * sizeof(TreeNode));
            if (!in) {
                throw std::runtime_error("failed while reading tree nodes from ensemble file: " + filepath);
            }

            GradientTree tree(num_features, max_depth, min_child_weight, lambda, gamma);
            tree.load_from_nodes(nodes);
            trees.push_back(tree);
        }
    }
};