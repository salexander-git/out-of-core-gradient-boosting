#pragma once
#include "HistogramBuilder.hpp"
#include "SplitEvaluator.hpp"
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <functional>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

struct TreeNode {
    bool is_leaf;
    float weight;
    int split_feature_idx;
    uint8_t split_bin;
    int left_child_idx;
    int right_child_idx;
};

class GradientTree {
private:
    int max_depth;
    float min_child_weight;
    float lambda;
    float gamma;
    int num_features;
    std::vector<TreeNode> nodes;
    std::vector<std::vector<FeatureHistogram>> active_histograms;

    int choose_effective_threads(size_t num_rows) const {
        #ifdef _OPENMP
            const int max_threads = omp_get_max_threads();

            const size_t work = num_rows * static_cast<size_t>(num_features);

            if (work < 1000000) return 1;
            if (work < 8000000) return std::min(2, max_threads);
            if (work < 25000000) return std::min(3, max_threads);
            return std::min(4, max_threads);
        #else
            (void)num_rows;
            return 1;
        #endif
    }

    void validate_loaded_nodes_or_throw(const std::vector<TreeNode>& candidate_nodes,
                                        int candidate_num_features) const {
        if (candidate_nodes.empty()) {
            throw std::runtime_error("cannot validate empty tree");
        }
        if (candidate_num_features <= 0) {
            throw std::runtime_error("invalid num_features in tree");
        }

        const int n = static_cast<int>(candidate_nodes.size());
        std::vector<uint8_t> state(static_cast<size_t>(n), 0); // 0=unvisited,1=visiting,2=done

        std::function<void(int)> dfs = [&](int idx) {
            if (idx < 0 || idx >= n) {
                throw std::runtime_error("corrupt tree: child index out of range");
            }
            uint8_t& s = state[static_cast<size_t>(idx)];
            if (s == 1) {
                throw std::runtime_error("corrupt tree: cycle detected");
            }
            if (s == 2) {
                return;
            }

            s = 1;
            const TreeNode& node = candidate_nodes[static_cast<size_t>(idx)];

            if (node.is_leaf) {
                if (node.left_child_idx != -1 || node.right_child_idx != -1) {
                    throw std::runtime_error("corrupt tree: leaf node has child pointers");
                }
            } else {
                if (node.split_feature_idx < 0 || node.split_feature_idx >= candidate_num_features) {
                    throw std::runtime_error("corrupt tree: invalid split_feature_idx");
                }
                if (node.left_child_idx < 0 || node.left_child_idx >= n) {
                    throw std::runtime_error("corrupt tree: invalid left_child_idx");
                }
                if (node.right_child_idx < 0 || node.right_child_idx >= n) {
                    throw std::runtime_error("corrupt tree: invalid right_child_idx");
                }
                if (node.left_child_idx == node.right_child_idx) {
                    throw std::runtime_error("corrupt tree: left and right children must differ");
                }

                dfs(node.left_child_idx);
                dfs(node.right_child_idx);
            }

            s = 2;
        };

        dfs(0);
    }

    void validate_current_tree_or_throw() const {
        validate_loaded_nodes_or_throw(nodes, num_features);
    }

public:
    GradientTree(int features, int depth = 6, float min_weight = 1.0f, float l = 1.0f, float g = 0.0f)
        : max_depth(depth), min_child_weight(min_weight), lambda(l), gamma(g), num_features(features) {
        if (features <= 0) {
            throw std::invalid_argument("num_features must be > 0");
        }
        if (depth <= 0) {
            throw std::invalid_argument("max_depth must be > 0");
        }
        if (min_weight < 0.0f) {
            throw std::invalid_argument("min_child_weight must be >= 0");
        }
    }

    int get_num_features() const { return num_features; }
    int get_max_depth() const { return max_depth; }
    float get_min_child_weight() const { return min_child_weight; }
    bool is_initialized() const { return !nodes.empty(); }

    void init_tree() {
        nodes.clear();
        TreeNode root{};
        root.is_leaf = true;
        root.weight = 0.0f;
        root.split_feature_idx = -1;
        root.split_bin = 0;
        root.left_child_idx = -1;
        root.right_child_idx = -1;
        nodes.push_back(root);
    }

    void start_new_level() {
        if (nodes.empty()) {
            throw std::runtime_error("init_tree() must be called before start_new_level()");
        }
        validate_current_tree_or_throw();
        active_histograms.assign(nodes.size(), std::vector<FeatureHistogram>(num_features));
    }

    void add_chunk(const uint8_t* binned_data, const float* gradients, const float* hessians, size_t num_rows) {
        if (!binned_data || !gradients || !hessians) {
            throw std::invalid_argument("add_chunk received a null pointer");
        }
        if (nodes.empty()) {
            throw std::runtime_error("init_tree() must be called before add_chunk()");
        }
        if (active_histograms.empty()) {
            throw std::runtime_error("start_new_level() must be called before add_chunk()");
        }

        validate_current_tree_or_throw();

        const size_t num_nodes = nodes.size();
        const int num_threads = choose_effective_threads(num_rows);

        if (num_threads == 1) {
            // Accumulate straight into the level's running histograms. The float sums then
            // follow row order alone, so splitting the rows into chunks cannot change them.
            for (size_t i = 0; i < num_rows; ++i) {
                int curr_idx = 0;

                while (!nodes[static_cast<size_t>(curr_idx)].is_leaf) {
                    const TreeNode& curr = nodes[static_cast<size_t>(curr_idx)];
                    const int f_idx = curr.split_feature_idx;
                    const uint8_t feature_val =
                        binned_data[i * static_cast<size_t>(num_features) + static_cast<size_t>(f_idx)];

                    curr_idx = (feature_val <= curr.split_bin) ? curr.left_child_idx : curr.right_child_idx;
                }

                auto& feature_blocks = active_histograms[static_cast<size_t>(curr_idx)];
                const float g = gradients[i];
                const float h = hessians[i];

                for (int f = 0; f < num_features; ++f) {
                    uint8_t bin = binned_data[i * static_cast<size_t>(num_features) + static_cast<size_t>(f)];
                    feature_blocks[static_cast<size_t>(f)].add_bin(bin, g, h);
                }
            }
            return;
        }

        std::vector<ThreadHistogramState> thread_states;
        thread_states.reserve(static_cast<size_t>(num_threads));
        for (int t = 0; t < num_threads; ++t) {
            thread_states.emplace_back(num_nodes, num_features);
        }

        {
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                auto& local_state = thread_states[static_cast<size_t>(tid)];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (size_t i = 0; i < num_rows; ++i) {
                    int curr_idx = 0;

                    while (!nodes[static_cast<size_t>(curr_idx)].is_leaf) {
                        const TreeNode& curr = nodes[static_cast<size_t>(curr_idx)];
                        const int f_idx = curr.split_feature_idx;
                        const uint8_t feature_val =
                            binned_data[i * static_cast<size_t>(num_features) + static_cast<size_t>(f_idx)];

                        curr_idx = (feature_val <= curr.split_bin) ? curr.left_child_idx : curr.right_child_idx;
                    }

                    auto& feature_blocks = local_state.per_node[static_cast<size_t>(curr_idx)].per_feature;
                    const float g = gradients[i];
                    const float h = hessians[i];

                    for (int f = 0; f < num_features; ++f) {
                        uint8_t bin = binned_data[i * static_cast<size_t>(num_features) + static_cast<size_t>(f)];
                        feature_blocks[static_cast<size_t>(f)].add_bin(bin, g, h);
                    }
                }
            }
        }

        const size_t total_pairs = num_nodes * static_cast<size_t>(num_features);

        if (total_pairs < 64) {
            for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
                for (int f = 0; f < num_features; ++f) {
                    for (int t = 0; t < num_threads; ++t) {
                        active_histograms[node_idx][static_cast<size_t>(f)].merge_from(
                            thread_states[static_cast<size_t>(t)].per_node[node_idx].per_feature[static_cast<size_t>(f)]
                        );
                    }
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
            for (size_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
                size_t node_idx = pair_idx / static_cast<size_t>(num_features);
                size_t feat_idx = pair_idx % static_cast<size_t>(num_features);

                auto& target = active_histograms[node_idx][feat_idx];
                for (int t = 0; t < num_threads; ++t) {
                    target.merge_from(thread_states[static_cast<size_t>(t)].per_node[node_idx].per_feature[feat_idx]);
                }
            }
        }
    }

    void finalize_level_splits() {
        if (nodes.empty()) {
            throw std::runtime_error("init_tree() must be called before finalize_level_splits()");
        }
        if (active_histograms.empty()) {
            throw std::runtime_error("start_new_level() must be called before finalize_level_splits()");
        }

        validate_current_tree_or_throw();

        SplitEvaluator evaluator(lambda, gamma, min_child_weight);
        size_t current_node_count = nodes.size();

        for (size_t i = 0; i < current_node_count; ++i) {
            if (!nodes[i].is_leaf) continue;

            float best_gain = 0.0f;
            int best_feature_idx = -1;
            SplitResult best_split{};
            best_split.split_bin_idx = -1;
            best_split.gain = 0.0f;
            best_split.weight_left = 0.0f;
            best_split.weight_right = 0.0f;

            for (int f = 0; f < num_features; ++f) {
                const auto& hist = active_histograms[i][static_cast<size_t>(f)];

                ThreadLocalHistogram flat_hist;
                for (int b = 0; b < 256; ++b) {
                    flat_hist.count[b] = hist.count[b];
                    flat_hist.sum_g[b] = hist.sum_g[b];
                    flat_hist.sum_h[b] = hist.sum_h[b];
                }

                float total_h = 0.0f;
                for (int b = 0; b < 256; ++b) total_h += hist.sum_h[b];
                if (total_h < min_child_weight) continue;

                SplitResult split = evaluator.evaluate_histogram(flat_hist);

                if (split.split_bin_idx != -1 && split.split_bin_idx < 256 && split.gain > best_gain) {
                    best_gain = split.gain;
                    best_feature_idx = f;
                    best_split = split;
                }
            }

            if (best_feature_idx != -1) {
                nodes[i].is_leaf = false;
                nodes[i].split_feature_idx = best_feature_idx;
                nodes[i].split_bin = static_cast<uint8_t>(best_split.split_bin_idx);

                TreeNode left_child{};
                left_child.is_leaf = true;
                left_child.weight = best_split.weight_left;
                left_child.split_feature_idx = -1;
                left_child.split_bin = 0;
                left_child.left_child_idx = -1;
                left_child.right_child_idx = -1;
                nodes[i].left_child_idx = static_cast<int>(nodes.size());
                nodes.push_back(left_child);

                TreeNode right_child{};
                right_child.is_leaf = true;
                right_child.weight = best_split.weight_right;
                right_child.split_feature_idx = -1;
                right_child.split_bin = 0;
                right_child.left_child_idx = -1;
                right_child.right_child_idx = -1;
                nodes[i].right_child_idx = static_cast<int>(nodes.size());
                nodes.push_back(right_child);
            }
        }

        active_histograms.clear();
        validate_current_tree_or_throw();
    }

    const std::vector<TreeNode>& get_nodes() const { return nodes; }

    void load_from_nodes(const std::vector<TreeNode>& new_nodes) {
        if (new_nodes.empty()) {
            throw std::runtime_error("cannot load tree from empty node list");
        }
        validate_loaded_nodes_or_throw(new_nodes, num_features);
        nodes = new_nodes;
        active_histograms.clear();
    }

    void save_model(const std::string& filepath) const {
        if (nodes.empty()) {
            throw std::runtime_error("cannot save an empty/uninitialized tree");
        }
        validate_current_tree_or_throw();

        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open model file for writing: " + filepath);
        }

        const uint32_t magic = 0x47425431; // "GBT1"
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&num_features), sizeof(int));

        size_t num_nodes = nodes.size();
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(nodes.data()), num_nodes * sizeof(TreeNode));

        if (!out) {
            throw std::runtime_error("failed while writing model file: " + filepath);
        }
    }

    void load_model(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open model file for reading: " + filepath);
        }

        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (!in || magic != 0x47425431) {
            throw std::runtime_error("invalid or unsupported model file: " + filepath);
        }

        int loaded_num_features = 0;
        in.read(reinterpret_cast<char*>(&loaded_num_features), sizeof(int));
        if (!in || loaded_num_features <= 0) {
            throw std::runtime_error("invalid num_features in model file: " + filepath);
        }

        size_t num_nodes = 0;
        in.read(reinterpret_cast<char*>(&num_nodes), sizeof(size_t));
        if (!in || num_nodes == 0) {
            throw std::runtime_error("invalid node count in model file: " + filepath);
        }

        std::vector<TreeNode> loaded_nodes(num_nodes);
        in.read(reinterpret_cast<char*>(loaded_nodes.data()), num_nodes * sizeof(TreeNode));
        if (!in) {
            throw std::runtime_error("failed while reading model nodes from: " + filepath);
        }

        validate_loaded_nodes_or_throw(loaded_nodes, loaded_num_features);

        num_features = loaded_num_features;
        nodes = std::move(loaded_nodes);
        active_histograms.clear();
    }

    float predict(const uint8_t* feature_row) const {
        if (!feature_row) {
            throw std::invalid_argument("predict received a null feature_row");
        }
        if (nodes.empty()) {
            throw std::runtime_error("cannot predict with an empty/uninitialized tree");
        }

        validate_current_tree_or_throw();

        int curr_idx = 0;
        while (!nodes[static_cast<size_t>(curr_idx)].is_leaf) {
            const TreeNode& curr = nodes[static_cast<size_t>(curr_idx)];
            const int f_idx = curr.split_feature_idx;

            curr_idx = (feature_row[f_idx] <= curr.split_bin) ? curr.left_child_idx : curr.right_child_idx;
        }
        return nodes[static_cast<size_t>(curr_idx)].weight;
    }

    void predict_chunk(const uint8_t* binned_data, size_t num_rows, float* out_preds) const {
        if (!binned_data || !out_preds) {
            throw std::invalid_argument("predict_chunk received a null pointer");
        }
        if (nodes.empty()) {
            throw std::runtime_error("cannot predict_chunk with an empty/uninitialized tree");
        }

        validate_current_tree_or_throw();

        for (size_t i = 0; i < num_rows; ++i) {
            out_preds[i] = predict(binned_data + i * static_cast<size_t>(num_features));
        }
    }
};