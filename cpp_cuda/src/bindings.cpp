#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "expectation_snn_cuda/manifest.hpp"

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size()
        && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::int64_t array_size_1d(const py::dict& arrays, const std::string& key) {
    if (!arrays.contains(py::str(key))) {
        throw std::runtime_error("missing manifest array: " + key);
    }
    py::array arr = py::cast<py::array>(arrays[py::str(key)]);
    if (arr.ndim() != 1) {
        throw std::runtime_error(
            "manifest array is not 1-D: " + key
            + " ndim=" + std::to_string(arr.ndim())
        );
    }
    return static_cast<std::int64_t>(arr.shape(0));
}

std::int32_t scalar_int32(const py::dict& arrays, const std::string& key) {
    if (!arrays.contains(py::str(key))) {
        throw std::runtime_error("missing manifest scalar: " + key);
    }
    py::array arr = py::cast<py::array>(arrays[py::str(key)]);
    if (arr.ndim() == 0) {
        return arr.attr("item")().cast<std::int32_t>();
    }
    if (arr.ndim() == 1 && arr.shape(0) == 1) {
        return arr.attr("item")(0).cast<std::int32_t>();
    }
    throw std::runtime_error("manifest scalar has unexpected shape: " + key);
}

double scalar_double(const py::dict& arrays, const std::string& key) {
    if (!arrays.contains(py::str(key))) {
        throw std::runtime_error("missing manifest scalar: " + key);
    }
    py::array arr = py::cast<py::array>(arrays[py::str(key)]);
    if (arr.ndim() == 0) {
        return arr.attr("item")().cast<double>();
    }
    if (arr.ndim() == 1 && arr.shape(0) == 1) {
        return arr.attr("item")(0).cast<double>();
    }
    throw std::runtime_error("manifest scalar has unexpected shape: " + key);
}

template <typename T>
std::vector<T> array_to_vector_1d(const py::dict& arrays, const std::string& key) {
    if (!arrays.contains(py::str(key))) {
        throw std::runtime_error("missing manifest array: " + key);
    }
    py::array_t<T, py::array::c_style | py::array::forcecast> arr =
        py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(
            arrays[py::str(key)]
        );
    if (arr.ndim() != 1) {
        throw std::runtime_error("manifest array is not 1-D: " + key);
    }
    const auto* ptr = arr.data();
    return std::vector<T>(ptr, ptr + arr.shape(0));
}

py::dict inspect_manifest_arrays(const py::dict& arrays) {
    const std::int32_t schema_version = scalar_int32(arrays, "schema_version");
    if (schema_version != 1) {
        throw std::runtime_error(
            "unsupported manifest schema_version="
            + std::to_string(schema_version)
        );
    }

    const std::vector<std::string> population_keys = {
        "pop_v1_stim_n", "pop_v1_e_n", "pop_v1_som_n", "pop_v1_pv_n",
        "pop_ctx_cue_n", "pop_ctx_e_n", "pop_ctx_inh_n",
        "pop_pred_cue_n", "pop_pred_e_n", "pop_pred_inh_n",
        "pop_direction_n",
    };
    py::dict population_sizes;
    for (const std::string& key : population_keys) {
        population_sizes[py::str(key)] = scalar_int32(arrays, key);
    }

    std::set<std::string> bank_names;
    for (const auto& item : arrays) {
        const std::string key = py::cast<std::string>(item.first);
        if (starts_with(key, "syn_") && ends_with(key, "_pre")) {
            bank_names.insert(key.substr(4, key.size() - 4 - 4));
        }
    }

    py::dict edge_counts;
    std::int64_t total_edges = 0;
    for (const std::string& bank : bank_names) {
        const std::string prefix = "syn_" + bank;
        const std::int64_t n_pre = array_size_1d(arrays, prefix + "_pre");
        const std::int64_t n_post = array_size_1d(arrays, prefix + "_post");
        const std::int64_t n_w = array_size_1d(arrays, prefix + "_w");
        if (n_pre != n_post || n_pre != n_w) {
            throw std::runtime_error(
                "manifest synapse bank shape mismatch: " + bank
                + " pre=" + std::to_string(n_pre)
                + " post=" + std::to_string(n_post)
                + " w=" + std::to_string(n_w)
            );
        }
        edge_counts[py::str(bank)] = n_pre;
        total_edges += n_pre;
    }

    py::list bank_names_out;
    for (const std::string& bank : bank_names) {
        bank_names_out.append(bank);
    }

    py::dict out;
    out["schema_version"] = schema_version;
    out["population_sizes"] = population_sizes;
    out["synapse_bank_count"] = static_cast<std::int64_t>(bank_names.size());
    out["synapse_bank_names"] = bank_names_out;
    out["edge_counts"] = edge_counts;
    out["total_edges"] = total_edges;
    return out;
}

py::array_t<double> vector_to_array_double(const std::vector<double>& values) {
    py::array_t<double> out(values.size());
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        buf(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
}

py::array_t<std::int32_t> vector_to_array_int32(
    const std::vector<std::int32_t>& values
) {
    py::array_t<std::int32_t> out(values.size());
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        buf(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
}

py::dict state_map_to_dict(
    const std::map<std::string, std::vector<double>>& state
) {
    py::dict out;
    for (const auto& [key, values] : state) {
        out[py::str(key)] = vector_to_array_double(values);
    }
    return out;
}

py::dict error_map_to_dict(const std::map<std::string, double>& errors) {
    py::dict out;
    for (const auto& [key, value] : errors) {
        out[py::str(key)] = value;
    }
    return out;
}

py::dict int_map_to_dict(const std::map<std::string, std::int32_t>& values) {
    py::dict out;
    for (const auto& [key, value] : values) {
        out[py::str(key)] = value;
    }
    return out;
}

py::dict int_vector_map_to_dict(
    const std::map<std::string, std::vector<std::int32_t>>& values
) {
    py::dict out;
    for (const auto& [key, value] : values) {
        out[py::str(key)] = vector_to_array_int32(value);
    }
    return out;
}

py::dict decay_result_to_dict(
    const expectation_snn_cuda::DecayPrimitiveResult& result
) {
    py::dict out;
    out["population"] = result.population;
    out["n_cells"] = result.n_cells;
    out["n_steps"] = result.n_steps;
    out["dt_ms"] = result.dt_ms;
    out["cpu_state"] = state_map_to_dict(result.cpu_state);
    out["cuda_state"] = state_map_to_dict(result.cuda_state);
    out["cpu_spike_counts"] = vector_to_array_int32(result.cpu_spike_counts);
    out["cuda_spike_counts"] = vector_to_array_int32(result.cuda_spike_counts);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["cpu_total_spikes"] = result.cpu_total_spikes;
    out["cuda_total_spikes"] = result.cuda_total_spikes;
    return out;
}

py::dict h_ring_dynamics_result_to_dict(
    const expectation_snn_cuda::HRingDynamicsTestResult& result
) {
    py::dict out;
    out["seed"] = result.seed;
    out["n_e"] = result.n_e;
    out["n_inh"] = result.n_inh;
    out["n_steps"] = result.n_steps;
    out["dt_ms"] = result.dt_ms;
    out["phase_steps"] = int_map_to_dict(result.phase_steps);
    out["metrics"] = error_map_to_dict(result.metrics);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["cpu_ctx_leader_counts"] =
        vector_to_array_int32(result.cpu_ctx_leader_counts);
    out["cuda_ctx_leader_counts"] =
        vector_to_array_int32(result.cuda_ctx_leader_counts);
    out["cpu_ctx_persistence_counts"] =
        vector_to_array_int32(result.cpu_ctx_persistence_counts);
    out["cuda_ctx_persistence_counts"] =
        vector_to_array_int32(result.cuda_ctx_persistence_counts);
    out["cpu_ctx_late_counts"] =
        vector_to_array_int32(result.cpu_ctx_late_counts);
    out["cuda_ctx_late_counts"] =
        vector_to_array_int32(result.cuda_ctx_late_counts);
    out["cpu_ctx_total_counts"] =
        vector_to_array_int32(result.cpu_ctx_total_counts);
    out["cuda_ctx_total_counts"] =
        vector_to_array_int32(result.cuda_ctx_total_counts);
    out["cpu_pred_leader_counts"] =
        vector_to_array_int32(result.cpu_pred_leader_counts);
    out["cuda_pred_leader_counts"] =
        vector_to_array_int32(result.cuda_pred_leader_counts);
    out["cpu_pred_pretrailer_counts"] =
        vector_to_array_int32(result.cpu_pred_pretrailer_counts);
    out["cuda_pred_pretrailer_counts"] =
        vector_to_array_int32(result.cuda_pred_pretrailer_counts);
    out["cpu_pred_trailer_counts"] =
        vector_to_array_int32(result.cpu_pred_trailer_counts);
    out["cuda_pred_trailer_counts"] =
        vector_to_array_int32(result.cuda_pred_trailer_counts);
    out["cpu_pred_total_counts"] =
        vector_to_array_int32(result.cpu_pred_total_counts);
    out["cuda_pred_total_counts"] =
        vector_to_array_int32(result.cuda_pred_total_counts);
    out["cpu_ctx_inh_total_counts"] =
        vector_to_array_int32(result.cpu_ctx_inh_total_counts);
    out["cuda_ctx_inh_total_counts"] =
        vector_to_array_int32(result.cuda_ctx_inh_total_counts);
    out["cpu_pred_inh_total_counts"] =
        vector_to_array_int32(result.cpu_pred_inh_total_counts);
    out["cuda_pred_inh_total_counts"] =
        vector_to_array_int32(result.cuda_pred_inh_total_counts);
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    return out;
}

py::dict scatter_result_to_dict(
    const expectation_snn_cuda::CsrScatterPrimitiveResult& result
) {
    py::dict out;
    out["bank_name"] = result.bank_name;
    out["pre_index"] = result.pre_index;
    out["n_edges"] = result.n_edges;
    out["n_pre"] = result.n_pre;
    out["n_target"] = result.n_target;
    out["n_edges_for_source"] = result.n_edges_for_source;
    out["drive_amp"] = result.drive_amp;
    out["cpu_target"] = vector_to_array_double(result.cpu_target);
    out["cuda_target"] = vector_to_array_double(result.cuda_target);
    out["max_abs_error"] = result.max_abs_error;
    return out;
}

py::dict event_ordering_result_to_dict(
    const expectation_snn_cuda::EventOrderingSliceResult& result
) {
    py::dict out;
    out["bank_name"] = result.bank_name;
    out["pre_index"] = result.pre_index;
    out["target_index"] = result.target_index;
    out["n_edges_for_source"] = result.n_edges_for_source;
    out["drive_amp"] = result.drive_amp;
    out["event_sum_to_target"] = result.event_sum_to_target;
    out["cpu_v_initial"] = result.cpu_v_initial;
    out["cuda_v_initial"] = result.cuda_v_initial;
    out["cpu_v_after_step0"] = result.cpu_v_after_step0;
    out["cuda_v_after_step0"] = result.cuda_v_after_step0;
    out["cpu_no_event_v_after_step0"] = result.cpu_no_event_v_after_step0;
    out["cpu_i_e_after_scatter"] = result.cpu_i_e_after_scatter;
    out["cuda_i_e_after_scatter"] = result.cuda_i_e_after_scatter;
    out["cpu_v_after_step1"] = result.cpu_v_after_step1;
    out["cuda_v_after_step1"] = result.cuda_v_after_step1;
    out["cpu_no_event_v_after_step1"] = result.cpu_no_event_v_after_step1;
    out["cpu_total_spikes"] = result.cpu_total_spikes;
    out["cuda_total_spikes"] = result.cuda_total_spikes;
    out["max_abs_error"] = result.max_abs_error;
    return out;
}

py::dict ctx_to_pred_count_result_to_dict(
    const expectation_snn_cuda::CtxToPredCountTestResult& result
) {
    py::dict out;
    out["bank_name"] = result.bank_name;
    out["pre_index"] = result.pre_index;
    out["target_index"] = result.target_index;
    out["n_steps"] = result.n_steps;
    out["window_start_step"] = result.window_start_step;
    out["window_end_step"] = result.window_end_step;
    out["n_edges_for_source"] = result.n_edges_for_source;
    out["drive_amp"] = result.drive_amp;
    out["event_sum_to_target"] = result.event_sum_to_target;
    out["event_steps"] = vector_to_array_int32(result.event_steps);
    out["cpu_counts"] = vector_to_array_int32(result.cpu_counts);
    out["cuda_counts"] = vector_to_array_int32(result.cuda_counts);
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["cpu_total_window_spikes"] = result.cpu_total_window_spikes;
    out["cuda_total_window_spikes"] = result.cuda_total_window_spikes;
    out["cpu_target_v_after_event_step"] = result.cpu_target_v_after_event_step;
    out["cuda_target_v_after_event_step"] = result.cuda_target_v_after_event_step;
    out["cpu_no_event_target_v_after_event_step"] =
        result.cpu_no_event_target_v_after_event_step;
    out["cpu_target_i_e_after_event_scatter"] =
        result.cpu_target_i_e_after_event_scatter;
    out["cuda_target_i_e_after_event_scatter"] =
        result.cuda_target_i_e_after_event_scatter;
    out["cpu_target_v_after_next_step"] = result.cpu_target_v_after_next_step;
    out["cuda_target_v_after_next_step"] = result.cuda_target_v_after_next_step;
    out["cpu_no_event_target_v_after_next_step"] =
        result.cpu_no_event_target_v_after_next_step;
    return out;
}

py::dict feedback_v1_count_result_to_dict(
    const expectation_snn_cuda::FeedbackV1CountTestResult& result
) {
    py::dict out;
    out["direct_bank_name"] = result.direct_bank_name;
    out["som_bank_name"] = result.som_bank_name;
    out["pre_index"] = result.pre_index;
    out["target_v1e_index"] = result.target_v1e_index;
    out["target_som_index"] = result.target_som_index;
    out["n_steps"] = result.n_steps;
    out["window_start_step"] = result.window_start_step;
    out["window_end_step"] = result.window_end_step;
    out["direct_edge_count"] = result.direct_edge_count;
    out["som_edge_count"] = result.som_edge_count;
    out["direct_edges_for_source"] = result.direct_edges_for_source;
    out["som_edges_for_source"] = result.som_edges_for_source;
    out["direct_drive_amp"] = result.direct_drive_amp;
    out["som_drive_amp"] = result.som_drive_amp;
    out["direct_event_sum_to_target"] = result.direct_event_sum_to_target;
    out["som_event_sum_to_target"] = result.som_event_sum_to_target;
    out["event_steps"] = vector_to_array_int32(result.event_steps);
    out["cpu_counts"] = vector_to_array_int32(result.cpu_counts);
    out["cuda_counts"] = vector_to_array_int32(result.cuda_counts);
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["cpu_total_window_spikes"] = result.cpu_total_window_spikes;
    out["cuda_total_window_spikes"] = result.cuda_total_window_spikes;
    out["cpu_v1e_soma_after_event_step"] =
        result.cpu_v1e_soma_after_event_step;
    out["cuda_v1e_soma_after_event_step"] =
        result.cuda_v1e_soma_after_event_step;
    out["cpu_no_event_v1e_soma_after_event_step"] =
        result.cpu_no_event_v1e_soma_after_event_step;
    out["cpu_v1e_i_ap_after_event_scatter"] =
        result.cpu_v1e_i_ap_after_event_scatter;
    out["cuda_v1e_i_ap_after_event_scatter"] =
        result.cuda_v1e_i_ap_after_event_scatter;
    out["cpu_v1som_i_e_after_event_scatter"] =
        result.cpu_v1som_i_e_after_event_scatter;
    out["cuda_v1som_i_e_after_event_scatter"] =
        result.cuda_v1som_i_e_after_event_scatter;
    out["cpu_v1e_ap_after_next_step"] = result.cpu_v1e_ap_after_next_step;
    out["cuda_v1e_ap_after_next_step"] = result.cuda_v1e_ap_after_next_step;
    out["cpu_no_event_v1e_ap_after_next_step"] =
        result.cpu_no_event_v1e_ap_after_next_step;
    out["cpu_v1som_v_after_next_step"] = result.cpu_v1som_v_after_next_step;
    out["cuda_v1som_v_after_next_step"] = result.cuda_v1som_v_after_next_step;
    out["cpu_no_event_v1som_v_after_next_step"] =
        result.cpu_no_event_v1som_v_after_next_step;
    out["cpu_v1e_soma_after_late_step"] =
        result.cpu_v1e_soma_after_late_step;
    out["cuda_v1e_soma_after_late_step"] =
        result.cuda_v1e_soma_after_late_step;
    out["cpu_no_event_v1e_soma_after_late_step"] =
        result.cpu_no_event_v1e_soma_after_late_step;
    return out;
}

py::dict v1_stim_feedforward_count_result_to_dict(
    const expectation_snn_cuda::V1StimFeedforwardCountTestResult& result
) {
    py::dict out;
    out["stim_bank_name"] = result.stim_bank_name;
    out["feedforward_bank_name"] = result.feedforward_bank_name;
    out["stim_pre_index"] = result.stim_pre_index;
    out["forced_v1e_index"] = result.forced_v1e_index;
    out["target_h_index"] = result.target_h_index;
    out["n_steps"] = result.n_steps;
    out["window_start_step"] = result.window_start_step;
    out["window_end_step"] = result.window_end_step;
    out["force_v1e_step"] = result.force_v1e_step;
    out["stim_edge_count"] = result.stim_edge_count;
    out["feedforward_edge_count"] = result.feedforward_edge_count;
    out["stim_edges_for_source"] = result.stim_edges_for_source;
    out["feedforward_edges_for_source"] = result.feedforward_edges_for_source;
    out["stim_drive_amp"] = result.stim_drive_amp;
    out["feedforward_drive_amp"] = result.feedforward_drive_amp;
    out["stim_event_sum_to_v1e_target"] = result.stim_event_sum_to_v1e_target;
    out["feedforward_event_sum_to_h_target"] =
        result.feedforward_event_sum_to_h_target;
    out["stim_event_steps"] = vector_to_array_int32(result.stim_event_steps);
    out["cpu_v1_counts"] = vector_to_array_int32(result.cpu_v1_counts);
    out["cuda_v1_counts"] = vector_to_array_int32(result.cuda_v1_counts);
    out["cpu_h_counts"] = vector_to_array_int32(result.cpu_h_counts);
    out["cuda_h_counts"] = vector_to_array_int32(result.cuda_h_counts);
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["cpu_total_v1_window_spikes"] = result.cpu_total_v1_window_spikes;
    out["cuda_total_v1_window_spikes"] = result.cuda_total_v1_window_spikes;
    out["cpu_total_h_window_spikes"] = result.cpu_total_h_window_spikes;
    out["cuda_total_h_window_spikes"] = result.cuda_total_h_window_spikes;
    out["cpu_v1e_soma_after_stim_step"] =
        result.cpu_v1e_soma_after_stim_step;
    out["cuda_v1e_soma_after_stim_step"] =
        result.cuda_v1e_soma_after_stim_step;
    out["cpu_no_stim_v1e_soma_after_stim_step"] =
        result.cpu_no_stim_v1e_soma_after_stim_step;
    out["cpu_v1e_i_e_after_stim_scatter"] =
        result.cpu_v1e_i_e_after_stim_scatter;
    out["cuda_v1e_i_e_after_stim_scatter"] =
        result.cuda_v1e_i_e_after_stim_scatter;
    out["cpu_v1e_soma_after_stim_next_step"] =
        result.cpu_v1e_soma_after_stim_next_step;
    out["cuda_v1e_soma_after_stim_next_step"] =
        result.cuda_v1e_soma_after_stim_next_step;
    out["cpu_no_stim_v1e_soma_after_stim_next_step"] =
        result.cpu_no_stim_v1e_soma_after_stim_next_step;
    out["cpu_h_v_after_force_v1e_step"] =
        result.cpu_h_v_after_force_v1e_step;
    out["cuda_h_v_after_force_v1e_step"] =
        result.cuda_h_v_after_force_v1e_step;
    out["cpu_no_ff_h_v_after_force_v1e_step"] =
        result.cpu_no_ff_h_v_after_force_v1e_step;
    out["cpu_h_i_e_after_ff_scatter"] =
        result.cpu_h_i_e_after_ff_scatter;
    out["cuda_h_i_e_after_ff_scatter"] =
        result.cuda_h_i_e_after_ff_scatter;
    out["cpu_h_v_after_ff_next_step"] = result.cpu_h_v_after_ff_next_step;
    out["cuda_h_v_after_ff_next_step"] = result.cuda_h_v_after_ff_next_step;
    out["cpu_no_ff_h_v_after_ff_next_step"] =
        result.cpu_no_ff_h_v_after_ff_next_step;
    return out;
}

py::dict closed_loop_count_result_to_dict(
    const expectation_snn_cuda::ClosedLoopDeterministicCountTestResult& result
) {
    py::dict out;
    out["n_steps"] = result.n_steps;
    out["window_start_step"] = result.window_start_step;
    out["window_end_step"] = result.window_end_step;
    out["stim_step"] = result.stim_step;
    out["v1_force_step"] = result.v1_force_step;
    out["hctx_force_step"] = result.hctx_force_step;
    out["hpred_force_step"] = result.hpred_force_step;
    out["stim_pre_index"] = result.stim_pre_index;
    out["v1e_index"] = result.v1e_index;
    out["hctx_index"] = result.hctx_index;
    out["hpred_index"] = result.hpred_index;
    out["feedback_v1e_index"] = result.feedback_v1e_index;
    out["feedback_som_index"] = result.feedback_som_index;
    out["edge_counts"] = int_map_to_dict(result.edge_counts);
    out["source_fanouts"] = int_map_to_dict(result.source_fanouts);
    out["drive_amps"] = error_map_to_dict(result.drive_amps);
    out["event_sums"] = error_map_to_dict(result.event_sums);
    out["cpu_v1_counts"] = vector_to_array_int32(result.cpu_v1_counts);
    out["cuda_v1_counts"] = vector_to_array_int32(result.cuda_v1_counts);
    out["cpu_hctx_counts"] = vector_to_array_int32(result.cpu_hctx_counts);
    out["cuda_hctx_counts"] = vector_to_array_int32(result.cuda_hctx_counts);
    out["cpu_hpred_counts"] = vector_to_array_int32(result.cpu_hpred_counts);
    out["cuda_hpred_counts"] = vector_to_array_int32(result.cuda_hpred_counts);
    out["cpu_total_v1_window_spikes"] = result.cpu_total_v1_window_spikes;
    out["cuda_total_v1_window_spikes"] = result.cuda_total_v1_window_spikes;
    out["cpu_total_hctx_window_spikes"] = result.cpu_total_hctx_window_spikes;
    out["cuda_total_hctx_window_spikes"] = result.cuda_total_hctx_window_spikes;
    out["cpu_total_hpred_window_spikes"] = result.cpu_total_hpred_window_spikes;
    out["cuda_total_hpred_window_spikes"] = result.cuda_total_hpred_window_spikes;
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["cpu_v1_soma_after_stim_step"] = result.cpu_v1_soma_after_stim_step;
    out["cuda_v1_soma_after_stim_step"] = result.cuda_v1_soma_after_stim_step;
    out["cpu_no_stim_v1_soma_after_stim_step"] =
        result.cpu_no_stim_v1_soma_after_stim_step;
    out["cpu_v1_i_e_after_stim_scatter"] =
        result.cpu_v1_i_e_after_stim_scatter;
    out["cuda_v1_i_e_after_stim_scatter"] =
        result.cuda_v1_i_e_after_stim_scatter;
    out["cpu_v1_soma_after_stim_next_step"] =
        result.cpu_v1_soma_after_stim_next_step;
    out["cuda_v1_soma_after_stim_next_step"] =
        result.cuda_v1_soma_after_stim_next_step;
    out["cpu_no_stim_v1_soma_after_stim_next_step"] =
        result.cpu_no_stim_v1_soma_after_stim_next_step;
    out["cpu_hctx_v_after_v1_step"] = result.cpu_hctx_v_after_v1_step;
    out["cuda_hctx_v_after_v1_step"] = result.cuda_hctx_v_after_v1_step;
    out["cpu_no_v1_hctx_v_after_v1_step"] =
        result.cpu_no_v1_hctx_v_after_v1_step;
    out["cpu_hctx_i_e_after_v1_scatter"] =
        result.cpu_hctx_i_e_after_v1_scatter;
    out["cuda_hctx_i_e_after_v1_scatter"] =
        result.cuda_hctx_i_e_after_v1_scatter;
    out["cpu_hctx_v_after_v1_next_step"] =
        result.cpu_hctx_v_after_v1_next_step;
    out["cuda_hctx_v_after_v1_next_step"] =
        result.cuda_hctx_v_after_v1_next_step;
    out["cpu_no_v1_hctx_v_after_v1_next_step"] =
        result.cpu_no_v1_hctx_v_after_v1_next_step;
    out["cpu_hpred_v_after_hctx_step"] =
        result.cpu_hpred_v_after_hctx_step;
    out["cuda_hpred_v_after_hctx_step"] =
        result.cuda_hpred_v_after_hctx_step;
    out["cpu_no_ctx_hpred_v_after_hctx_step"] =
        result.cpu_no_ctx_hpred_v_after_hctx_step;
    out["cpu_hpred_i_e_after_hctx_scatter"] =
        result.cpu_hpred_i_e_after_hctx_scatter;
    out["cuda_hpred_i_e_after_hctx_scatter"] =
        result.cuda_hpred_i_e_after_hctx_scatter;
    out["cpu_hpred_v_after_hctx_next_step"] =
        result.cpu_hpred_v_after_hctx_next_step;
    out["cuda_hpred_v_after_hctx_next_step"] =
        result.cuda_hpred_v_after_hctx_next_step;
    out["cpu_no_ctx_hpred_v_after_hctx_next_step"] =
        result.cpu_no_ctx_hpred_v_after_hctx_next_step;
    out["cpu_v1_soma_after_hpred_step"] =
        result.cpu_v1_soma_after_hpred_step;
    out["cuda_v1_soma_after_hpred_step"] =
        result.cuda_v1_soma_after_hpred_step;
    out["cpu_no_fb_v1_soma_after_hpred_step"] =
        result.cpu_no_fb_v1_soma_after_hpred_step;
    out["cpu_v1_i_ap_after_fb_scatter"] =
        result.cpu_v1_i_ap_after_fb_scatter;
    out["cuda_v1_i_ap_after_fb_scatter"] =
        result.cuda_v1_i_ap_after_fb_scatter;
    out["cpu_som_i_e_after_fb_scatter"] =
        result.cpu_som_i_e_after_fb_scatter;
    out["cuda_som_i_e_after_fb_scatter"] =
        result.cuda_som_i_e_after_fb_scatter;
    out["cpu_v1_ap_after_fb_next_step"] =
        result.cpu_v1_ap_after_fb_next_step;
    out["cuda_v1_ap_after_fb_next_step"] =
        result.cuda_v1_ap_after_fb_next_step;
    out["cpu_no_fb_v1_ap_after_fb_next_step"] =
        result.cpu_no_fb_v1_ap_after_fb_next_step;
    out["cpu_som_v_after_fb_next_step"] =
        result.cpu_som_v_after_fb_next_step;
    out["cuda_som_v_after_fb_next_step"] =
        result.cuda_som_v_after_fb_next_step;
    out["cpu_no_fb_som_v_after_fb_next_step"] =
        result.cpu_no_fb_som_v_after_fb_next_step;
    out["cpu_v1_soma_after_fb_late_step"] =
        result.cpu_v1_soma_after_fb_late_step;
    out["cuda_v1_soma_after_fb_late_step"] =
        result.cuda_v1_soma_after_fb_late_step;
    out["cpu_no_fb_v1_soma_after_fb_late_step"] =
        result.cpu_no_fb_v1_soma_after_fb_late_step;
    return out;
}

py::dict frozen_richter_deterministic_trial_result_to_dict(
    const expectation_snn_cuda::FrozenRichterDeterministicTrialResult& result
) {
    py::dict out;
    out["n_steps"] = result.n_steps;
    out["phase_steps"] = int_map_to_dict(result.phase_steps);
    out["edge_counts"] = int_map_to_dict(result.edge_counts);
    out["source_fanouts"] = int_map_to_dict(result.source_fanouts);
    out["source_event_counts"] = int_map_to_dict(result.source_event_counts);
    out["drive_amps"] = error_map_to_dict(result.drive_amps);
    out["event_sums"] = error_map_to_dict(result.event_sums);
    out["ordering_deltas"] = error_map_to_dict(result.ordering_deltas);
    out["expected_stim_pre_index"] = result.expected_stim_pre_index;
    out["unexpected_stim_pre_index"] = result.unexpected_stim_pre_index;
    out["stim_period_steps"] = result.stim_period_steps;
    out["v1e_index"] = result.v1e_index;
    out["hctx_index"] = result.hctx_index;
    out["hpred_index"] = result.hpred_index;
    out["feedback_v1e_index"] = result.feedback_v1e_index;
    out["feedback_som_index"] = result.feedback_som_index;
    out["cpu_raw_counts"] = int_vector_map_to_dict(result.cpu_raw_counts);
    out["cuda_raw_counts"] = int_vector_map_to_dict(result.cuda_raw_counts);
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    return out;
}

py::dict frozen_richter_seeded_source_result_to_dict(
    const expectation_snn_cuda::FrozenRichterSeededSourceResult& result
) {
    py::dict out;
    out["seed"] = result.seed;
    out["n_steps"] = result.n_steps;
    out["dt_ms"] = result.dt_ms;
    out["expected_channel"] = result.expected_channel;
    out["unexpected_channel"] = result.unexpected_channel;
    out["phase_steps"] = int_map_to_dict(result.phase_steps);
    out["edge_counts"] = int_map_to_dict(result.edge_counts);
    out["source_event_counts"] = int_map_to_dict(result.source_event_counts);
    out["rates_hz"] = error_map_to_dict(result.rates_hz);
    out["cpu_raw_counts"] = int_vector_map_to_dict(result.cpu_raw_counts);
    out["cuda_raw_counts"] = int_vector_map_to_dict(result.cuda_raw_counts);
    out["cpu_source_counts"] = int_vector_map_to_dict(result.cpu_source_counts);
    out["cuda_source_counts"] = int_vector_map_to_dict(result.cuda_source_counts);
    out["cpu_diagnostic_rates_hz"] =
        state_map_to_dict(result.cpu_diagnostic_rates_hz);
    out["cuda_diagnostic_rates_hz"] =
        state_map_to_dict(result.cuda_diagnostic_rates_hz);
    out["cpu_final_state"] = state_map_to_dict(result.cpu_final_state);
    out["cuda_final_state"] = state_map_to_dict(result.cuda_final_state);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    return out;
}

py::dict ctx_pred_plasticity_result_to_dict(
    const expectation_snn_cuda::CtxPredPlasticityTestResult& result
) {
    py::dict out;
    out["seed"] = result.seed;
    out["n_pre"] = result.n_pre;
    out["n_post"] = result.n_post;
    out["n_syn"] = result.n_syn;
    out["n_steps"] = result.n_steps;
    out["dt_ms"] = result.dt_ms;
    out["tau_coinc_ms"] = result.tau_coinc_ms;
    out["tau_elig_ms"] = result.tau_elig_ms;
    out["eta"] = result.eta;
    out["gamma"] = result.gamma;
    out["w_target"] = result.w_target;
    out["w_max"] = result.w_max;
    out["w_row_max"] = result.w_row_max;
    out["m_integral"] = result.m_integral;
    out["dt_trial_s"] = result.dt_trial_s;
    out["paired_pre"] = result.paired_pre;
    out["paired_post"] = result.paired_post;
    out["pre_rule_pre"] = result.pre_rule_pre;
    out["capped_pre"] = result.capped_pre;
    out["silent_pre"] = result.silent_pre;
    out["silent_post"] = result.silent_post;
    out["cpu_n_capped"] = result.cpu_n_capped;
    out["cuda_n_capped"] = result.cuda_n_capped;
    out["pre_event_steps"] = vector_to_array_int32(result.pre_event_steps);
    out["pre_event_cells"] = vector_to_array_int32(result.pre_event_cells);
    out["post_event_steps"] = vector_to_array_int32(result.post_event_steps);
    out["post_event_cells"] = vector_to_array_int32(result.post_event_cells);
    out["initial_w"] = vector_to_array_double(result.initial_w);
    out["cpu_w"] = vector_to_array_double(result.cpu_w);
    out["cuda_w"] = vector_to_array_double(result.cuda_w);
    out["cpu_elig_before_gate"] =
        vector_to_array_double(result.cpu_elig_before_gate);
    out["cuda_elig_before_gate"] =
        vector_to_array_double(result.cuda_elig_before_gate);
    out["cpu_elig_after_gate"] =
        vector_to_array_double(result.cpu_elig_after_gate);
    out["cuda_elig_after_gate"] =
        vector_to_array_double(result.cuda_elig_after_gate);
    out["cpu_xpre_after_gate"] =
        vector_to_array_double(result.cpu_xpre_after_gate);
    out["cuda_xpre_after_gate"] =
        vector_to_array_double(result.cuda_xpre_after_gate);
    out["cpu_xpost_after_gate"] =
        vector_to_array_double(result.cpu_xpost_after_gate);
    out["cuda_xpost_after_gate"] =
        vector_to_array_double(result.cuda_xpost_after_gate);
    out["cpu_row_sums"] = vector_to_array_double(result.cpu_row_sums);
    out["cuda_row_sums"] = vector_to_array_double(result.cuda_row_sums);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    return out;
}

py::dict ctx_pred_training_trial_slice_result_to_dict(
    const expectation_snn_cuda::CtxPredTrainingTrialSliceResult& result
) {
    py::dict out;
    out["seed"] = result.seed;
    out["n_pre"] = result.n_pre;
    out["n_post"] = result.n_post;
    out["n_syn"] = result.n_syn;
    out["n_steps"] = result.n_steps;
    out["dt_ms"] = result.dt_ms;
    out["phase_steps"] = int_map_to_dict(result.phase_steps);
    out["event_counts"] = int_map_to_dict(result.event_counts);
    out["gate_step"] = result.gate_step;
    out["leader_pre"] = result.leader_pre;
    out["boundary_pre"] = result.boundary_pre;
    out["trailer_post"] = result.trailer_post;
    out["late_trailer_post"] = result.late_trailer_post;
    out["capped_pre"] = result.capped_pre;
    out["silent_pre"] = result.silent_pre;
    out["silent_post"] = result.silent_post;
    out["cpu_n_capped"] = result.cpu_n_capped;
    out["cuda_n_capped"] = result.cuda_n_capped;
    out["hctx_pre_event_steps"] =
        vector_to_array_int32(result.hctx_pre_event_steps);
    out["hctx_pre_event_cells"] =
        vector_to_array_int32(result.hctx_pre_event_cells);
    out["hpred_post_event_steps"] =
        vector_to_array_int32(result.hpred_post_event_steps);
    out["hpred_post_event_cells"] =
        vector_to_array_int32(result.hpred_post_event_cells);
    out["initial_w_ctx_pred"] =
        vector_to_array_double(result.initial_w_ctx_pred);
    out["cpu_w_ctx_pred_final"] =
        vector_to_array_double(result.cpu_w_ctx_pred_final);
    out["cuda_w_ctx_pred_final"] =
        vector_to_array_double(result.cuda_w_ctx_pred_final);
    out["cpu_elig_before_gate"] =
        vector_to_array_double(result.cpu_elig_before_gate);
    out["cuda_elig_before_gate"] =
        vector_to_array_double(result.cuda_elig_before_gate);
    out["cpu_elig_after_iti"] =
        vector_to_array_double(result.cpu_elig_after_iti);
    out["cuda_elig_after_iti"] =
        vector_to_array_double(result.cuda_elig_after_iti);
    out["cpu_xpre_after_iti"] =
        vector_to_array_double(result.cpu_xpre_after_iti);
    out["cuda_xpre_after_iti"] =
        vector_to_array_double(result.cuda_xpre_after_iti);
    out["cpu_xpost_after_iti"] =
        vector_to_array_double(result.cpu_xpost_after_iti);
    out["cuda_xpost_after_iti"] =
        vector_to_array_double(result.cuda_xpost_after_iti);
    out["cpu_row_sums"] = vector_to_array_double(result.cpu_row_sums);
    out["cuda_row_sums"] = vector_to_array_double(result.cuda_row_sums);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    return out;
}

py::dict ctx_pred_tiny_trainer_result_to_dict(
    const expectation_snn_cuda::CtxPredTinyTrainerTestResult& result
) {
    py::dict out;
    out["seed"] = result.seed;
    out["schedule_variant"] = result.schedule_variant;
    out["n_trials"] = result.n_trials;
    out["n_pre"] = result.n_pre;
    out["n_post"] = result.n_post;
    out["n_syn"] = result.n_syn;
    out["h_ee_n_syn"] = result.h_ee_n_syn;
    out["n_steps"] = result.n_steps;
    out["trial_steps"] = result.trial_steps;
    out["dt_ms"] = result.dt_ms;
    out["phase_steps"] = int_map_to_dict(result.phase_steps);
    out["event_counts"] = int_map_to_dict(result.event_counts);
    out["gate_steps"] = vector_to_array_int32(result.gate_steps);
    out["trial_leader_pre_cells"] =
        vector_to_array_int32(result.trial_leader_pre_cells);
    out["trial_trailer_post_cells"] =
        vector_to_array_int32(result.trial_trailer_post_cells);
    out["hctx_pre_event_steps"] =
        vector_to_array_int32(result.hctx_pre_event_steps);
    out["hctx_pre_event_cells"] =
        vector_to_array_int32(result.hctx_pre_event_cells);
    out["hpred_post_event_steps"] =
        vector_to_array_int32(result.hpred_post_event_steps);
    out["hpred_post_event_cells"] =
        vector_to_array_int32(result.hpred_post_event_cells);
    out["initial_w_ctx_pred"] =
        vector_to_array_double(result.initial_w_ctx_pred);
    out["cpu_w_ctx_pred_final"] =
        vector_to_array_double(result.cpu_w_ctx_pred_final);
    out["cuda_w_ctx_pred_final"] =
        vector_to_array_double(result.cuda_w_ctx_pred_final);
    out["cpu_ctx_ee_w_final"] =
        vector_to_array_double(result.cpu_ctx_ee_w_final);
    out["cuda_ctx_ee_w_final"] =
        vector_to_array_double(result.cuda_ctx_ee_w_final);
    out["cpu_pred_ee_w_final"] =
        vector_to_array_double(result.cpu_pred_ee_w_final);
    out["cuda_pred_ee_w_final"] =
        vector_to_array_double(result.cuda_pred_ee_w_final);
    out["cpu_elig_after_training"] =
        vector_to_array_double(result.cpu_elig_after_training);
    out["cuda_elig_after_training"] =
        vector_to_array_double(result.cuda_elig_after_training);
    out["cpu_xpre_after_training"] =
        vector_to_array_double(result.cpu_xpre_after_training);
    out["cuda_xpre_after_training"] =
        vector_to_array_double(result.cuda_xpre_after_training);
    out["cpu_xpost_after_training"] =
        vector_to_array_double(result.cpu_xpost_after_training);
    out["cuda_xpost_after_training"] =
        vector_to_array_double(result.cuda_xpost_after_training);
    out["cpu_row_sums"] = vector_to_array_double(result.cpu_row_sums);
    out["cuda_row_sums"] = vector_to_array_double(result.cuda_row_sums);
    out["cpu_gate_w_before"] =
        vector_to_array_double(result.cpu_gate_w_before);
    out["cuda_gate_w_before"] =
        vector_to_array_double(result.cuda_gate_w_before);
    out["cpu_gate_w_after"] =
        vector_to_array_double(result.cpu_gate_w_after);
    out["cuda_gate_w_after"] =
        vector_to_array_double(result.cuda_gate_w_after);
    out["cpu_gate_dw_sum"] =
        vector_to_array_double(result.cpu_gate_dw_sum);
    out["cuda_gate_dw_sum"] =
        vector_to_array_double(result.cuda_gate_dw_sum);
    out["cpu_gate_elig_mean"] =
        vector_to_array_double(result.cpu_gate_elig_mean);
    out["cuda_gate_elig_mean"] =
        vector_to_array_double(result.cuda_gate_elig_mean);
    out["cpu_gate_elig_max"] =
        vector_to_array_double(result.cpu_gate_elig_max);
    out["cuda_gate_elig_max"] =
        vector_to_array_double(result.cuda_gate_elig_max);
    out["cpu_gate_row_sum_max"] =
        vector_to_array_double(result.cpu_gate_row_sum_max);
    out["cuda_gate_row_sum_max"] =
        vector_to_array_double(result.cuda_gate_row_sum_max);
    out["cpu_gate_n_capped"] =
        vector_to_array_int32(result.cpu_gate_n_capped);
    out["cuda_gate_n_capped"] =
        vector_to_array_int32(result.cuda_gate_n_capped);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    return out;
}

py::dict stage1_h_gate_dynamics_result_to_dict(
    const expectation_snn_cuda::Stage1HGateDynamicsResult& result
) {
    py::dict out;
    out["seed"] = result.seed;
    out["n_trials"] = result.n_trials;
    out["n_e"] = result.n_e;
    out["n_inh"] = result.n_inh;
    out["n_steps_per_trial"] = result.n_steps_per_trial;
    out["dt_ms"] = result.dt_ms;
    out["phase_steps"] = int_map_to_dict(result.phase_steps);
    out["metrics"] = error_map_to_dict(result.metrics);
    out["max_abs_error"] = error_map_to_dict(result.max_abs_error);
    out["leader_channels"] = vector_to_array_int32(result.leader_channels);
    out["trailer_channels"] = vector_to_array_int32(result.trailer_channels);
    out["cpu_ctx_persistence_ms_by_trial"] =
        vector_to_array_double(result.cpu_ctx_persistence_ms_by_trial);
    out["cuda_ctx_persistence_ms_by_trial"] =
        vector_to_array_double(result.cuda_ctx_persistence_ms_by_trial);
    out["cpu_pred_pretrailer_target_counts"] =
        vector_to_array_int32(result.cpu_pred_pretrailer_target_counts);
    out["cuda_pred_pretrailer_target_counts"] =
        vector_to_array_int32(result.cuda_pred_pretrailer_target_counts);
    out["cpu_ctx_total_counts"] =
        vector_to_array_int32(result.cpu_ctx_total_counts);
    out["cuda_ctx_total_counts"] =
        vector_to_array_int32(result.cuda_ctx_total_counts);
    out["cpu_pred_total_counts"] =
        vector_to_array_int32(result.cpu_pred_total_counts);
    out["cuda_pred_total_counts"] =
        vector_to_array_int32(result.cuda_pred_total_counts);
    out["cpu_ctx_inh_total_counts"] =
        vector_to_array_int32(result.cpu_ctx_inh_total_counts);
    out["cuda_ctx_inh_total_counts"] =
        vector_to_array_int32(result.cuda_ctx_inh_total_counts);
    out["cpu_pred_inh_total_counts"] =
        vector_to_array_int32(result.cpu_pred_inh_total_counts);
    out["cuda_pred_inh_total_counts"] =
        vector_to_array_int32(result.cuda_pred_inh_total_counts);
    return out;
}

}  // namespace

PYBIND11_MODULE(_native_cuda, m) {
    m.doc() = "Native CUDA skeleton for frozen ctx_pred Richter evaluation";

    m.def("backend_info", &expectation_snn_cuda::backend_info);
    m.def(
        "inspect_manifest_arrays",
        &inspect_manifest_arrays,
        "Inspect NumPy-loaded schema-v1 manifest arrays"
    );
    m.def(
        "run_decay_test",
        [](const std::string& population, std::int32_t n_steps, bool threshold_case) {
            return decay_result_to_dict(expectation_snn_cuda::run_decay_primitive(
                population, n_steps, threshold_case
            ));
        },
        py::arg("population"),
        py::arg("n_steps") = 10,
        py::arg("threshold_case") = false,
        "Run network-specific CPU/CUDA no-event neuron decay primitive"
    );
    m.def(
        "run_h_ring_dynamics_test",
        [](std::int64_t seed) {
            return h_ring_dynamics_result_to_dict(
                expectation_snn_cuda::run_h_ring_dynamics_test(seed)
            );
        },
        py::arg("seed") = 42,
        "Run bounded H_context/H_prediction recurrent/inhibitory dynamics primitive"
    );
    m.def(
        "run_csr_scatter_test",
        [](const py::dict& arrays, const std::string& bank_name, std::int32_t pre_index) {
            const std::string prefix = "syn_" + bank_name;
            return scatter_result_to_dict(expectation_snn_cuda::run_csr_scatter_primitive(
                bank_name,
                array_to_vector_1d<std::int32_t>(arrays, prefix + "_pre"),
                array_to_vector_1d<std::int32_t>(arrays, prefix + "_post"),
                array_to_vector_1d<double>(arrays, prefix + "_w"),
                scalar_double(arrays, prefix + "_drive_amp_pA"),
                pre_index
            ));
        },
        py::arg("arrays"),
        py::arg("bank_name"),
        py::arg("pre_index") = 0,
        "Run one-source-spike CSR scatter primitive for an exported bank"
    );
    m.def(
        "run_event_ordering_slice",
        [](const py::dict& arrays, const std::string& bank_name, std::int32_t pre_index) {
            const std::string prefix = "syn_" + bank_name;
            return event_ordering_result_to_dict(
                expectation_snn_cuda::run_event_ordering_slice(
                    bank_name,
                    array_to_vector_1d<std::int32_t>(arrays, prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, prefix + "_post"),
                    array_to_vector_1d<double>(arrays, prefix + "_w"),
                    scalar_double(arrays, prefix + "_drive_amp_pA"),
                    pre_index
                )
            );
        },
        py::arg("arrays"),
        py::arg("bank_name"),
        py::arg("pre_index") = 0,
        "Run deterministic one-event ordering slice against an exported bank"
    );
    m.def(
        "run_ctx_to_pred_count_test",
        [](const py::dict& arrays,
           const std::string& bank_name,
           std::int32_t pre_index,
           std::vector<std::int32_t> event_steps,
           std::int32_t n_steps,
           std::int32_t window_start_step,
           std::int32_t window_end_step) {
            const std::string prefix = "syn_" + bank_name;
            return ctx_to_pred_count_result_to_dict(
                expectation_snn_cuda::run_ctx_to_pred_count_test(
                    bank_name,
                    array_to_vector_1d<std::int32_t>(arrays, prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, prefix + "_post"),
                    array_to_vector_1d<double>(arrays, prefix + "_w"),
                    scalar_double(arrays, prefix + "_drive_amp_pA"),
                    pre_index,
                    event_steps,
                    n_steps,
                    window_start_step,
                    window_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("bank_name"),
        py::arg("pre_index") = 0,
        py::arg("event_steps") = std::vector<std::int32_t>{2, 3},
        py::arg("n_steps") = 20,
        py::arg("window_start_step") = 5,
        py::arg("window_end_step") = 10,
        "Run deterministic multi-step H_ctx -> H_pred count-window primitive"
    );
    m.def(
        "run_feedback_v1_count_test",
        [](const py::dict& arrays,
           const std::string& direct_bank_name,
           const std::string& som_bank_name,
           std::int32_t pre_index,
           std::vector<std::int32_t> event_steps,
           std::int32_t n_steps,
           std::int32_t window_start_step,
           std::int32_t window_end_step) {
            const std::string direct_prefix = "syn_" + direct_bank_name;
            const std::string som_prefix = "syn_" + som_bank_name;
            return feedback_v1_count_result_to_dict(
                expectation_snn_cuda::run_feedback_v1_count_test(
                    direct_bank_name,
                    array_to_vector_1d<std::int32_t>(arrays, direct_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, direct_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, direct_prefix + "_w"),
                    scalar_double(arrays, direct_prefix + "_drive_amp_pA"),
                    som_bank_name,
                    array_to_vector_1d<std::int32_t>(arrays, som_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, som_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, som_prefix + "_w"),
                    scalar_double(arrays, som_prefix + "_drive_amp_pA"),
                    pre_index,
                    event_steps,
                    n_steps,
                    window_start_step,
                    window_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("direct_bank_name") = "fb_pred_to_v1e_apical",
        py::arg("som_bank_name") = "fb_pred_to_v1som",
        py::arg("pre_index") = 7,
        py::arg("event_steps") = std::vector<std::int32_t>{2, 3},
        py::arg("n_steps") = 20,
        py::arg("window_start_step") = 5,
        py::arg("window_end_step") = 10,
        "Run deterministic multi-step H_pred -> V1 feedback count primitive"
    );
    m.def(
        "run_v1_stim_feedforward_count_test",
        [](const py::dict& arrays,
           const std::string& stim_bank_name,
           const std::string& feedforward_bank_name,
           std::int32_t stim_pre_index,
           std::vector<std::int32_t> stim_event_steps,
           std::int32_t force_v1e_step,
           std::int32_t n_steps,
           std::int32_t window_start_step,
           std::int32_t window_end_step) {
            const std::string stim_prefix = "syn_" + stim_bank_name;
            const std::string ff_prefix = "syn_" + feedforward_bank_name;
            return v1_stim_feedforward_count_result_to_dict(
                expectation_snn_cuda::run_v1_stim_feedforward_count_test(
                    stim_bank_name,
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, stim_prefix + "_w"),
                    scalar_double(arrays, stim_prefix + "_drive_amp_pA"),
                    feedforward_bank_name,
                    array_to_vector_1d<std::int32_t>(arrays, ff_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, ff_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, ff_prefix + "_w"),
                    scalar_double(arrays, ff_prefix + "_drive_amp_pA"),
                    stim_pre_index,
                    stim_event_steps,
                    force_v1e_step,
                    n_steps,
                    window_start_step,
                    window_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("stim_bank_name") = "v1_stim_to_e",
        py::arg("feedforward_bank_name") = "v1_to_h_ctx",
        py::arg("stim_pre_index") = 0,
        py::arg("stim_event_steps") = std::vector<std::int32_t>{2},
        py::arg("force_v1e_step") = 4,
        py::arg("n_steps") = 20,
        py::arg("window_start_step") = 4,
        py::arg("window_end_step") = 10,
        "Run deterministic V1 stimulus and V1->H feedforward count primitive"
    );
    m.def(
        "run_closed_loop_deterministic_count_test",
        [](const py::dict& arrays,
           std::int32_t stim_pre_index,
           std::int32_t stim_step,
           std::int32_t v1_force_step,
           std::int32_t hctx_force_step,
           std::int32_t hpred_force_step,
           std::int32_t n_steps,
           std::int32_t window_start_step,
           std::int32_t window_end_step) {
            const std::string stim_prefix = "syn_v1_stim_to_e";
            const std::string v1_to_h_prefix = "syn_v1_to_h_ctx";
            const std::string ctx_to_pred_prefix = "syn_ctx_to_pred";
            const std::string fb_direct_prefix = "syn_fb_pred_to_v1e_apical";
            const std::string fb_som_prefix = "syn_fb_pred_to_v1som";
            return closed_loop_count_result_to_dict(
                expectation_snn_cuda::run_closed_loop_deterministic_count_test(
                    "v1_stim_to_e",
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, stim_prefix + "_w"),
                    scalar_double(arrays, stim_prefix + "_drive_amp_pA"),
                    "v1_to_h_ctx",
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, v1_to_h_prefix + "_w"),
                    scalar_double(arrays, v1_to_h_prefix + "_drive_amp_pA"),
                    "ctx_to_pred",
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, ctx_to_pred_prefix + "_w"),
                    scalar_double(arrays, ctx_to_pred_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1e_apical",
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_direct_prefix + "_w"),
                    scalar_double(arrays, fb_direct_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1som",
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_som_prefix + "_w"),
                    scalar_double(arrays, fb_som_prefix + "_drive_amp_pA"),
                    stim_pre_index,
                    stim_step,
                    v1_force_step,
                    hctx_force_step,
                    hpred_force_step,
                    n_steps,
                    window_start_step,
                    window_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("stim_pre_index") = 0,
        py::arg("stim_step") = 2,
        py::arg("v1_force_step") = 4,
        py::arg("hctx_force_step") = 26,
        py::arg("hpred_force_step") = 28,
        py::arg("n_steps") = 35,
        py::arg("window_start_step") = 4,
        py::arg("window_end_step") = 32,
        "Run deterministic closed-loop V1/H_ctx/H_pred/V1 feedback count primitive"
    );
    m.def(
        "run_frozen_richter_deterministic_trial_test",
        [](const py::dict& arrays,
           std::int32_t expected_stim_pre_index,
           std::int32_t unexpected_stim_pre_index,
           std::int32_t stim_period_steps,
           std::int32_t n_steps,
           std::int32_t leader_start_step,
           std::int32_t leader_end_step,
           std::int32_t preprobe_start_step,
           std::int32_t preprobe_end_step,
           std::int32_t trailer_start_step,
           std::int32_t trailer_end_step,
           std::int32_t iti_start_step,
           std::int32_t iti_end_step) {
            const std::string stim_prefix = "syn_v1_stim_to_e";
            const std::string v1_to_h_prefix = "syn_v1_to_h_ctx";
            const std::string ctx_to_pred_prefix = "syn_ctx_to_pred";
            const std::string fb_direct_prefix = "syn_fb_pred_to_v1e_apical";
            const std::string fb_som_prefix = "syn_fb_pred_to_v1som";
            return frozen_richter_deterministic_trial_result_to_dict(
                expectation_snn_cuda::run_frozen_richter_deterministic_trial_test(
                    "v1_stim_to_e",
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, stim_prefix + "_w"),
                    scalar_double(arrays, stim_prefix + "_drive_amp_pA"),
                    "v1_to_h_ctx",
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, v1_to_h_prefix + "_w"),
                    scalar_double(arrays, v1_to_h_prefix + "_drive_amp_pA"),
                    "ctx_to_pred",
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, ctx_to_pred_prefix + "_w"),
                    scalar_double(arrays, ctx_to_pred_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1e_apical",
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_direct_prefix + "_w"),
                    scalar_double(arrays, fb_direct_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1som",
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_som_prefix + "_w"),
                    scalar_double(arrays, fb_som_prefix + "_drive_amp_pA"),
                    expected_stim_pre_index,
                    unexpected_stim_pre_index,
                    stim_period_steps,
                    n_steps,
                    leader_start_step,
                    leader_end_step,
                    preprobe_start_step,
                    preprobe_end_step,
                    trailer_start_step,
                    trailer_end_step,
                    iti_start_step,
                    iti_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("expected_stim_pre_index") = 0,
        py::arg("unexpected_stim_pre_index") = 20,
        py::arg("stim_period_steps") = 5,
        py::arg("n_steps") = 120,
        py::arg("leader_start_step") = 0,
        py::arg("leader_end_step") = 30,
        py::arg("preprobe_start_step") = 30,
        py::arg("preprobe_end_step") = 60,
        py::arg("trailer_start_step") = 60,
        py::arg("trailer_end_step") = 100,
        py::arg("iti_start_step") = 100,
        py::arg("iti_end_step") = 120,
        "Run deterministic bounded frozen-Richter trial scheduler primitive"
    );
    m.def(
        "run_frozen_richter_seeded_source_test",
        [](const py::dict& arrays,
           std::int64_t seed,
           std::int32_t expected_channel,
           std::int32_t unexpected_channel,
           double grating_rate_hz,
           double baseline_rate_hz,
           std::int32_t n_steps,
           std::int32_t leader_start_step,
           std::int32_t leader_end_step,
           std::int32_t preprobe_start_step,
           std::int32_t preprobe_end_step,
           std::int32_t trailer_start_step,
           std::int32_t trailer_end_step,
           std::int32_t iti_start_step,
           std::int32_t iti_end_step) {
            const std::string stim_prefix = "syn_v1_stim_to_e";
            const std::string v1_to_h_prefix = "syn_v1_to_h_ctx";
            const std::string ctx_to_pred_prefix = "syn_ctx_to_pred";
            const std::string fb_direct_prefix = "syn_fb_pred_to_v1e_apical";
            const std::string fb_som_prefix = "syn_fb_pred_to_v1som";
            return frozen_richter_seeded_source_result_to_dict(
                expectation_snn_cuda::run_frozen_richter_seeded_source_test(
                    "v1_stim_to_e",
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, stim_prefix + "_w"),
                    scalar_double(arrays, stim_prefix + "_drive_amp_pA"),
                    array_to_vector_1d<std::int32_t>(arrays, "v1_stim_channel"),
                    "v1_to_h_ctx",
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, v1_to_h_prefix + "_w"),
                    scalar_double(arrays, v1_to_h_prefix + "_drive_amp_pA"),
                    "ctx_to_pred",
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, ctx_to_pred_prefix + "_w"),
                    scalar_double(arrays, ctx_to_pred_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1e_apical",
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_direct_prefix + "_w"),
                    scalar_double(arrays, fb_direct_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1som",
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_som_prefix + "_w"),
                    scalar_double(arrays, fb_som_prefix + "_drive_amp_pA"),
                    seed,
                    expected_channel,
                    unexpected_channel,
                    grating_rate_hz,
                    baseline_rate_hz,
                    n_steps,
                    leader_start_step,
                    leader_end_step,
                    preprobe_start_step,
                    preprobe_end_step,
                    trailer_start_step,
                    trailer_end_step,
                    iti_start_step,
                    iti_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("seed") = 12345,
        py::arg("expected_channel") = 0,
        py::arg("unexpected_channel") = 1,
        py::arg("grating_rate_hz") = 500.0,
        py::arg("baseline_rate_hz") = 0.0,
        py::arg("n_steps") = 120,
        py::arg("leader_start_step") = 0,
        py::arg("leader_end_step") = 30,
        py::arg("preprobe_start_step") = 30,
        py::arg("preprobe_end_step") = 60,
        py::arg("trailer_start_step") = 60,
        py::arg("trailer_end_step") = 100,
        py::arg("iti_start_step") = 100,
        py::arg("iti_end_step") = 120,
        "Run seeded bounded frozen-Richter source-generation parity primitive"
    );
    m.def(
        "run_frozen_richter_controlled_source_test",
        [](const py::dict& arrays,
           std::vector<std::int32_t> event_steps,
           std::vector<std::int32_t> event_sources,
           std::int32_t expected_channel,
           std::int32_t unexpected_channel,
           std::int32_t n_steps,
           std::int32_t leader_start_step,
           std::int32_t leader_end_step,
           std::int32_t preprobe_start_step,
           std::int32_t preprobe_end_step,
           std::int32_t trailer_start_step,
           std::int32_t trailer_end_step,
           std::int32_t iti_start_step,
           std::int32_t iti_end_step) {
            const std::string stim_prefix = "syn_v1_stim_to_e";
            const std::string v1_to_h_prefix = "syn_v1_to_h_ctx";
            const std::string ctx_to_pred_prefix = "syn_ctx_to_pred";
            const std::string fb_direct_prefix = "syn_fb_pred_to_v1e_apical";
            const std::string fb_som_prefix = "syn_fb_pred_to_v1som";
            return frozen_richter_seeded_source_result_to_dict(
                expectation_snn_cuda::run_frozen_richter_controlled_source_test(
                    "v1_stim_to_e",
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, stim_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, stim_prefix + "_w"),
                    scalar_double(arrays, stim_prefix + "_drive_amp_pA"),
                    array_to_vector_1d<std::int32_t>(arrays, "v1_stim_channel"),
                    "v1_to_h_ctx",
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, v1_to_h_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, v1_to_h_prefix + "_w"),
                    scalar_double(arrays, v1_to_h_prefix + "_drive_amp_pA"),
                    "ctx_to_pred",
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, ctx_to_pred_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, ctx_to_pred_prefix + "_w"),
                    scalar_double(arrays, ctx_to_pred_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1e_apical",
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_direct_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_direct_prefix + "_w"),
                    scalar_double(arrays, fb_direct_prefix + "_drive_amp_pA"),
                    "fb_pred_to_v1som",
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_pre"),
                    array_to_vector_1d<std::int32_t>(arrays, fb_som_prefix + "_post"),
                    array_to_vector_1d<double>(arrays, fb_som_prefix + "_w"),
                    scalar_double(arrays, fb_som_prefix + "_drive_amp_pA"),
                    event_steps,
                    event_sources,
                    expected_channel,
                    unexpected_channel,
                    n_steps,
                    leader_start_step,
                    leader_end_step,
                    preprobe_start_step,
                    preprobe_end_step,
                    trailer_start_step,
                    trailer_end_step,
                    iti_start_step,
                    iti_end_step
                )
            );
        },
        py::arg("arrays"),
        py::arg("event_steps"),
        py::arg("event_sources"),
        py::arg("expected_channel") = 0,
        py::arg("unexpected_channel") = 1,
        py::arg("n_steps") = 120,
        py::arg("leader_start_step") = 0,
        py::arg("leader_end_step") = 30,
        py::arg("preprobe_start_step") = 30,
        py::arg("preprobe_end_step") = 60,
        py::arg("trailer_start_step") = 60,
        py::arg("trailer_end_step") = 100,
        py::arg("iti_start_step") = 100,
        py::arg("iti_end_step") = 120,
        "Run bounded frozen-Richter with explicit stimulus source events"
    );
    m.def(
        "run_ctx_pred_plasticity_test",
        [](std::int64_t seed, std::int32_t n_steps) {
            return ctx_pred_plasticity_result_to_dict(
                expectation_snn_cuda::run_ctx_pred_plasticity_test(
                    seed,
                    n_steps
                )
            );
        },
        py::arg("seed") = 42,
        py::arg("n_steps") = 640,
        "Run CPU/CUDA ctx_pred eligibility, delayed gate, clip, and row-cap primitive"
    );
    m.def(
        "run_ctx_pred_training_trial_slice_test",
        [](std::int64_t seed) {
            return ctx_pred_training_trial_slice_result_to_dict(
                expectation_snn_cuda::run_ctx_pred_training_trial_slice_test(seed)
            );
        },
        py::arg("seed") = 42,
        "Run controlled GPU-native Stage1 ctx_pred training trial-slice primitive"
    );
    m.def(
        "run_ctx_pred_tiny_trainer_test",
        [](std::int64_t seed, std::int32_t schedule_variant) {
            return ctx_pred_tiny_trainer_result_to_dict(
                expectation_snn_cuda::run_ctx_pred_tiny_trainer_test(
                    seed,
                    schedule_variant
                )
            );
        },
        py::arg("seed") = 42,
        py::arg("schedule_variant") = 0,
        "Run controlled multi-trial GPU-native Stage1 ctx_pred trainer primitive"
    );
    m.def(
        "run_ctx_pred_generated_schedule_test",
        [](
            std::int64_t seed,
            const std::vector<std::int32_t>& leader_pre_cells,
            const std::vector<std::int32_t>& trailer_post_cells
        ) {
            return ctx_pred_tiny_trainer_result_to_dict(
                expectation_snn_cuda::run_ctx_pred_generated_schedule_test(
                    seed,
                    leader_pre_cells,
                    trailer_post_cells
                )
            );
        },
        py::arg("seed"),
        py::arg("leader_pre_cells"),
        py::arg("trailer_post_cells"),
        "Run controlled GPU-native Stage1 trainer on generated trial-cell schedule"
    );
    m.def(
        "run_stage1_h_gate_dynamics_test",
        [](
            std::int64_t seed,
            const std::vector<std::int32_t>& leader_cells,
            const std::vector<std::int32_t>& trailer_cells,
            const std::vector<double>& w_ctx_pred
        ) {
            return stage1_h_gate_dynamics_result_to_dict(
                expectation_snn_cuda::run_stage1_h_gate_dynamics_test(
                    seed,
                    leader_cells,
                    trailer_cells,
                    w_ctx_pred
                )
            );
        },
        py::arg("seed"),
        py::arg("leader_cells"),
        py::arg("trailer_cells"),
        py::arg("w_ctx_pred"),
        "Run native H recurrent/inhibitory gate metrics over a Stage1 schedule"
    );
    m.def("inspect_manifest_path", [](const std::string& path) {
        const auto summary = expectation_snn_cuda::inspect_manifest_path(path);
        py::dict out;
        out["schema_version"] = summary.schema_version;
        out["synapse_bank_count"] = summary.synapse_bank_count;
        return out;
    });
}
