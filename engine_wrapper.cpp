#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "OutOfCoreQuantizer.hpp"
#include "BinnedDataReader.hpp"
#include "Tree.hpp"
#include "GradientBoostingModel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gb_core, m) {
    m.doc() = "Out-of-Core Gradient Boosting Engine";

    py::class_<OutOfCoreQuantizer>(m, "OutOfCoreQuantizer")
        .def(py::init<const std::string&, size_t, float, float>())
        .def("process_next_chunk", &OutOfCoreQuantizer::process_next_chunk)
        .def("get_binned_data", &OutOfCoreQuantizer::get_binned_data)
        .def("process_next_chunk_into",
             [](OutOfCoreQuantizer& q,
                py::array_t<uint8_t, py::array::c_style> array,
                size_t offset) {
                py::buffer_info buf = array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("process_next_chunk_into expects a 1D uint8 array");
                }
                if (buf.readonly) {
                    throw std::runtime_error("process_next_chunk_into requires a writable uint8 array");
                }
                if (offset > static_cast<size_t>(buf.size)) {
                    throw std::runtime_error("offset exceeds output array length");
                }

                const size_t available = static_cast<size_t>(buf.size) - offset;
                if (available < q.get_chunk_size()) {
                    throw std::runtime_error("output array does not have enough capacity for quantizer chunk_size");
                }

                uint8_t* ptr = static_cast<uint8_t*>(buf.ptr) + offset;
                return q.process_next_chunk_into(ptr);
             });

    py::class_<BinnedDataReader>(m, "BinnedDataReader")
        .def(py::init<const std::string&, size_t>())
        .def("read_next_chunk",
             [](BinnedDataReader& r,
                py::array_t<uint8_t, py::array::c_style> array,
                size_t offset) {
                py::buffer_info buf = array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("read_next_chunk expects a 1D uint8 array");
                }
                if (buf.readonly) {
                    throw std::runtime_error("read_next_chunk requires a writable uint8 array");
                }
                if (offset > static_cast<size_t>(buf.size)) {
                    throw std::runtime_error("offset exceeds output array length");
                }

                const size_t available = static_cast<size_t>(buf.size) - offset;
                if (available < r.get_chunk_size()) {
                    throw std::runtime_error("output array does not have enough capacity for reader chunk_size");
                }

                uint8_t* ptr = static_cast<uint8_t*>(buf.ptr) + offset;
                return r.read_next_chunk(ptr);
             });

    py::class_<TreeNode>(m, "TreeNode")
        .def_readonly("is_leaf", &TreeNode::is_leaf)
        .def_readonly("weight", &TreeNode::weight)
        .def_readonly("split_feature_idx", &TreeNode::split_feature_idx)
        .def_readonly("split_bin", &TreeNode::split_bin)
        .def_readonly("left_child_idx", &TreeNode::left_child_idx)
        .def_readonly("right_child_idx", &TreeNode::right_child_idx);

    py::class_<GradientTree>(m, "GradientTree")
        .def(py::init<int, int, float, float, float>(),
             py::arg("num_features"),
             py::arg("max_depth") = 6,
             py::arg("min_child_weight") = 1.0f,
             py::arg("lambda_val") = 1.0f,
             py::arg("gamma_val") = 0.0f)
        .def("get_num_features", &GradientTree::get_num_features)
        .def("get_max_depth", &GradientTree::get_max_depth)
        .def("get_min_child_weight", &GradientTree::get_min_child_weight)
        .def("is_initialized", &GradientTree::is_initialized)
        .def("init_tree", &GradientTree::init_tree)
        .def("start_new_level", &GradientTree::start_new_level)
        .def("finalize_level_splits", &GradientTree::finalize_level_splits)
        .def("add_chunk", [](GradientTree& tree,
                             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> binned_data,
                             py::array_t<float, py::array::c_style | py::array::forcecast> gradients,
                             py::array_t<float, py::array::c_style | py::array::forcecast> hessians) {
            py::buffer_info buf_b = binned_data.request();
            py::buffer_info buf_g = gradients.request();
            py::buffer_info buf_h = hessians.request();

            if (buf_b.ndim != 1 || buf_g.ndim != 1 || buf_h.ndim != 1) {
                throw std::runtime_error("add_chunk expects 1D arrays");
            }

            size_t num_rows = static_cast<size_t>(buf_g.shape[0]);
            if (static_cast<size_t>(buf_h.shape[0]) != num_rows) {
                throw std::runtime_error("gradients and hessians must have the same length");
            }

            size_t expected_binned = num_rows * static_cast<size_t>(tree.get_num_features());
            if (static_cast<size_t>(buf_b.shape[0]) != expected_binned) {
                throw std::runtime_error("binned_data length must equal num_rows * num_features");
            }

            tree.add_chunk(
                static_cast<const uint8_t*>(buf_b.ptr),
                static_cast<const float*>(buf_g.ptr),
                static_cast<const float*>(buf_h.ptr),
                num_rows
            );
        })
        .def("get_nodes", &GradientTree::get_nodes)
        .def("save_model", &GradientTree::save_model)
        .def("load_model", &GradientTree::load_model)
        .def("predict", [](const GradientTree& tree,
                           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> feature_row) {
            py::buffer_info buf = feature_row.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("predict expects a 1D uint8 feature row");
            }
            if (static_cast<int>(buf.shape[0]) != tree.get_num_features()) {
                throw std::runtime_error("feature row length does not match model num_features");
            }
            return tree.predict(static_cast<const uint8_t*>(buf.ptr));
        })
        .def("predict_chunk", [](const GradientTree& tree,
                                 py::array_t<uint8_t, py::array::c_style | py::array::forcecast> binned_data,
                                 py::array_t<float, py::array::c_style> out_preds) {
            py::buffer_info buf_b = binned_data.request();
            py::buffer_info buf_o = out_preds.request();

            if (buf_b.ndim != 1 || buf_o.ndim != 1) {
                throw std::runtime_error("predict_chunk expects 1D arrays");
            }
            if (buf_o.readonly) {
                throw std::runtime_error("predict_chunk requires a writable float32 output array");
            }

            size_t num_rows = static_cast<size_t>(buf_o.shape[0]);
            size_t expected_binned = num_rows * static_cast<size_t>(tree.get_num_features());
            if (static_cast<size_t>(buf_b.shape[0]) != expected_binned) {
                throw std::runtime_error("binned_data length must equal num_rows * num_features");
            }

            tree.predict_chunk(
                static_cast<const uint8_t*>(buf_b.ptr),
                num_rows,
                static_cast<float*>(buf_o.ptr)
            );
        });

    py::class_<GradientBoostingModel>(m, "GradientBoostingModel")
        .def(py::init<int, int, float, float, float, float, float>(),
             py::arg("num_features"),
             py::arg("max_depth") = 6,
             py::arg("min_child_weight") = 1.0f,
             py::arg("lambda_val") = 1.0f,
             py::arg("gamma_val") = 0.0f,
             py::arg("learning_rate") = 0.1f,
             py::arg("base_score") = 0.0f)
        .def("get_num_features", &GradientBoostingModel::get_num_features)
        .def("get_max_depth", &GradientBoostingModel::get_max_depth)
        .def("get_min_child_weight", &GradientBoostingModel::get_min_child_weight)
        .def("get_lambda", &GradientBoostingModel::get_lambda)
        .def("get_gamma", &GradientBoostingModel::get_gamma)
        .def("get_learning_rate", &GradientBoostingModel::get_learning_rate)
        .def("get_base_score", &GradientBoostingModel::get_base_score)
        .def("set_base_score", &GradientBoostingModel::set_base_score)
        .def("get_num_trees", &GradientBoostingModel::get_num_trees)
        .def("save_model", &GradientBoostingModel::save_model)
        .def("load_model", &GradientBoostingModel::load_model)
        .def("add_tree", &GradientBoostingModel::add_tree)
        .def("predict", [](const GradientBoostingModel& model,
                           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> feature_row) {
            py::buffer_info buf = feature_row.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("predict expects a 1D uint8 feature row");
            }
            if (static_cast<int>(buf.shape[0]) != model.get_num_features()) {
                throw std::runtime_error("feature row length does not match ensemble num_features");
            }
            return model.predict(static_cast<const uint8_t*>(buf.ptr));
        })
        .def("predict_raw_sum", [](const GradientBoostingModel& model,
                                   py::array_t<uint8_t, py::array::c_style | py::array::forcecast> feature_row) {
            py::buffer_info buf = feature_row.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("predict_raw_sum expects a 1D uint8 feature row");
            }
            if (static_cast<int>(buf.shape[0]) != model.get_num_features()) {
                throw std::runtime_error("feature row length does not match ensemble num_features");
            }
            return model.predict_raw_sum(static_cast<const uint8_t*>(buf.ptr));
        })
        .def("predict_chunk", [](const GradientBoostingModel& model,
                                 py::array_t<uint8_t, py::array::c_style | py::array::forcecast> binned_data,
                                 py::array_t<float, py::array::c_style> out_preds) {
            py::buffer_info buf_b = binned_data.request();
            py::buffer_info buf_o = out_preds.request();

            if (buf_b.ndim != 1 || buf_o.ndim != 1) {
                throw std::runtime_error("predict_chunk expects 1D arrays");
            }
            if (buf_o.readonly) {
                throw std::runtime_error("predict_chunk requires a writable float32 output array");
            }

            size_t num_rows = static_cast<size_t>(buf_o.shape[0]);
            size_t expected_binned = num_rows * static_cast<size_t>(model.get_num_features());
            if (static_cast<size_t>(buf_b.shape[0]) != expected_binned) {
                throw std::runtime_error("binned_data length must equal num_rows * num_features");
            }

            model.predict_chunk(
                static_cast<const uint8_t*>(buf_b.ptr),
                num_rows,
                static_cast<float*>(buf_o.ptr)
            );
        });
}