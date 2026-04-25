#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace expectation_snn_cuda {

struct ManifestSummary {
    std::int32_t schema_version;
    std::int64_t synapse_bank_count;
};

std::string backend_info();
ManifestSummary inspect_manifest_path(const std::string& path);

struct DecayPrimitiveResult {
    std::string population;
    std::int32_t n_cells;
    std::int32_t n_steps;
    double dt_ms;
    std::map<std::string, std::vector<double>> cpu_state;
    std::map<std::string, std::vector<double>> cuda_state;
    std::vector<std::int32_t> cpu_spike_counts;
    std::vector<std::int32_t> cuda_spike_counts;
    std::map<std::string, double> max_abs_error;
    std::int32_t cpu_total_spikes;
    std::int32_t cuda_total_spikes;
};

DecayPrimitiveResult run_decay_primitive(
    const std::string& population,
    std::int32_t n_steps,
    bool threshold_case
);

struct HRingDynamicsTestResult {
    std::int32_t seed;
    std::int32_t n_e;
    std::int32_t n_inh;
    std::int32_t n_steps;
    double dt_ms;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, double> metrics;
    std::map<std::string, double> max_abs_error;
    std::vector<std::int32_t> cpu_ctx_leader_counts;
    std::vector<std::int32_t> cuda_ctx_leader_counts;
    std::vector<std::int32_t> cpu_ctx_persistence_counts;
    std::vector<std::int32_t> cuda_ctx_persistence_counts;
    std::vector<std::int32_t> cpu_ctx_late_counts;
    std::vector<std::int32_t> cuda_ctx_late_counts;
    std::vector<std::int32_t> cpu_ctx_total_counts;
    std::vector<std::int32_t> cuda_ctx_total_counts;
    std::vector<std::int32_t> cpu_pred_leader_counts;
    std::vector<std::int32_t> cuda_pred_leader_counts;
    std::vector<std::int32_t> cpu_pred_pretrailer_counts;
    std::vector<std::int32_t> cuda_pred_pretrailer_counts;
    std::vector<std::int32_t> cpu_pred_trailer_counts;
    std::vector<std::int32_t> cuda_pred_trailer_counts;
    std::vector<std::int32_t> cpu_pred_total_counts;
    std::vector<std::int32_t> cuda_pred_total_counts;
    std::vector<std::int32_t> cpu_ctx_inh_total_counts;
    std::vector<std::int32_t> cuda_ctx_inh_total_counts;
    std::vector<std::int32_t> cpu_pred_inh_total_counts;
    std::vector<std::int32_t> cuda_pred_inh_total_counts;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
};

HRingDynamicsTestResult run_h_ring_dynamics_test(std::int64_t seed);

struct CsrScatterPrimitiveResult {
    std::string bank_name;
    std::int32_t pre_index;
    std::int32_t n_edges;
    std::int32_t n_pre;
    std::int32_t n_target;
    std::int32_t n_edges_for_source;
    double drive_amp;
    std::vector<double> cpu_target;
    std::vector<double> cuda_target;
    double max_abs_error;
};

CsrScatterPrimitiveResult run_csr_scatter_primitive(
    const std::string& bank_name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight,
    double drive_amp,
    std::int32_t pre_index
);

struct EventOrderingSliceResult {
    std::string bank_name;
    std::int32_t pre_index;
    std::int32_t target_index;
    std::int32_t n_edges_for_source;
    double drive_amp;
    double event_sum_to_target;
    double cpu_v_initial;
    double cuda_v_initial;
    double cpu_v_after_step0;
    double cuda_v_after_step0;
    double cpu_no_event_v_after_step0;
    double cpu_i_e_after_scatter;
    double cuda_i_e_after_scatter;
    double cpu_v_after_step1;
    double cuda_v_after_step1;
    double cpu_no_event_v_after_step1;
    std::int32_t cpu_total_spikes;
    std::int32_t cuda_total_spikes;
    double max_abs_error;
};

EventOrderingSliceResult run_event_ordering_slice(
    const std::string& bank_name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight,
    double drive_amp,
    std::int32_t pre_index
);

struct CtxToPredCountTestResult {
    std::string bank_name;
    std::int32_t pre_index;
    std::int32_t target_index;
    std::int32_t n_steps;
    std::int32_t window_start_step;
    std::int32_t window_end_step;
    std::int32_t n_edges_for_source;
    double drive_amp;
    double event_sum_to_target;
    std::vector<std::int32_t> event_steps;
    std::vector<std::int32_t> cpu_counts;
    std::vector<std::int32_t> cuda_counts;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
    std::map<std::string, double> max_abs_error;
    std::int32_t cpu_total_window_spikes;
    std::int32_t cuda_total_window_spikes;
    double cpu_target_v_after_event_step;
    double cuda_target_v_after_event_step;
    double cpu_no_event_target_v_after_event_step;
    double cpu_target_i_e_after_event_scatter;
    double cuda_target_i_e_after_event_scatter;
    double cpu_target_v_after_next_step;
    double cuda_target_v_after_next_step;
    double cpu_no_event_target_v_after_next_step;
};

CtxToPredCountTestResult run_ctx_to_pred_count_test(
    const std::string& bank_name,
    const std::vector<std::int32_t>& pre,
    const std::vector<std::int32_t>& post,
    const std::vector<double>& weight,
    double drive_amp,
    std::int32_t pre_index,
    const std::vector<std::int32_t>& event_steps,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
);

struct FeedbackV1CountTestResult {
    std::string direct_bank_name;
    std::string som_bank_name;
    std::int32_t pre_index;
    std::int32_t target_v1e_index;
    std::int32_t target_som_index;
    std::int32_t n_steps;
    std::int32_t window_start_step;
    std::int32_t window_end_step;
    std::int32_t direct_edge_count;
    std::int32_t som_edge_count;
    std::int32_t direct_edges_for_source;
    std::int32_t som_edges_for_source;
    double direct_drive_amp;
    double som_drive_amp;
    double direct_event_sum_to_target;
    double som_event_sum_to_target;
    std::vector<std::int32_t> event_steps;
    std::vector<std::int32_t> cpu_counts;
    std::vector<std::int32_t> cuda_counts;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
    std::map<std::string, double> max_abs_error;
    std::int32_t cpu_total_window_spikes;
    std::int32_t cuda_total_window_spikes;
    double cpu_v1e_soma_after_event_step;
    double cuda_v1e_soma_after_event_step;
    double cpu_no_event_v1e_soma_after_event_step;
    double cpu_v1e_i_ap_after_event_scatter;
    double cuda_v1e_i_ap_after_event_scatter;
    double cpu_v1som_i_e_after_event_scatter;
    double cuda_v1som_i_e_after_event_scatter;
    double cpu_v1e_ap_after_next_step;
    double cuda_v1e_ap_after_next_step;
    double cpu_no_event_v1e_ap_after_next_step;
    double cpu_v1som_v_after_next_step;
    double cuda_v1som_v_after_next_step;
    double cpu_no_event_v1som_v_after_next_step;
    double cpu_v1e_soma_after_late_step;
    double cuda_v1e_soma_after_late_step;
    double cpu_no_event_v1e_soma_after_late_step;
};

FeedbackV1CountTestResult run_feedback_v1_count_test(
    const std::string& direct_bank_name,
    const std::vector<std::int32_t>& direct_pre,
    const std::vector<std::int32_t>& direct_post,
    const std::vector<double>& direct_weight,
    double direct_drive_amp,
    const std::string& som_bank_name,
    const std::vector<std::int32_t>& som_pre,
    const std::vector<std::int32_t>& som_post,
    const std::vector<double>& som_weight,
    double som_drive_amp,
    std::int32_t pre_index,
    const std::vector<std::int32_t>& event_steps,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
);

struct V1StimFeedforwardCountTestResult {
    std::string stim_bank_name;
    std::string feedforward_bank_name;
    std::int32_t stim_pre_index;
    std::int32_t forced_v1e_index;
    std::int32_t target_h_index;
    std::int32_t n_steps;
    std::int32_t window_start_step;
    std::int32_t window_end_step;
    std::int32_t force_v1e_step;
    std::int32_t stim_edge_count;
    std::int32_t feedforward_edge_count;
    std::int32_t stim_edges_for_source;
    std::int32_t feedforward_edges_for_source;
    double stim_drive_amp;
    double feedforward_drive_amp;
    double stim_event_sum_to_v1e_target;
    double feedforward_event_sum_to_h_target;
    std::vector<std::int32_t> stim_event_steps;
    std::vector<std::int32_t> cpu_v1_counts;
    std::vector<std::int32_t> cuda_v1_counts;
    std::vector<std::int32_t> cpu_h_counts;
    std::vector<std::int32_t> cuda_h_counts;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
    std::map<std::string, double> max_abs_error;
    std::int32_t cpu_total_v1_window_spikes;
    std::int32_t cuda_total_v1_window_spikes;
    std::int32_t cpu_total_h_window_spikes;
    std::int32_t cuda_total_h_window_spikes;
    double cpu_v1e_soma_after_stim_step;
    double cuda_v1e_soma_after_stim_step;
    double cpu_no_stim_v1e_soma_after_stim_step;
    double cpu_v1e_i_e_after_stim_scatter;
    double cuda_v1e_i_e_after_stim_scatter;
    double cpu_v1e_soma_after_stim_next_step;
    double cuda_v1e_soma_after_stim_next_step;
    double cpu_no_stim_v1e_soma_after_stim_next_step;
    double cpu_h_v_after_force_v1e_step;
    double cuda_h_v_after_force_v1e_step;
    double cpu_no_ff_h_v_after_force_v1e_step;
    double cpu_h_i_e_after_ff_scatter;
    double cuda_h_i_e_after_ff_scatter;
    double cpu_h_v_after_ff_next_step;
    double cuda_h_v_after_ff_next_step;
    double cpu_no_ff_h_v_after_ff_next_step;
};

V1StimFeedforwardCountTestResult run_v1_stim_feedforward_count_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::string& feedforward_bank_name,
    const std::vector<std::int32_t>& feedforward_pre,
    const std::vector<std::int32_t>& feedforward_post,
    const std::vector<double>& feedforward_weight,
    double feedforward_drive_amp,
    std::int32_t stim_pre_index,
    const std::vector<std::int32_t>& stim_event_steps,
    std::int32_t force_v1e_step,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
);

struct ClosedLoopDeterministicCountTestResult {
    std::int32_t n_steps;
    std::int32_t window_start_step;
    std::int32_t window_end_step;
    std::int32_t stim_step;
    std::int32_t v1_force_step;
    std::int32_t hctx_force_step;
    std::int32_t hpred_force_step;
    std::int32_t stim_pre_index;
    std::int32_t v1e_index;
    std::int32_t hctx_index;
    std::int32_t hpred_index;
    std::int32_t feedback_v1e_index;
    std::int32_t feedback_som_index;
    std::map<std::string, std::int32_t> edge_counts;
    std::map<std::string, std::int32_t> source_fanouts;
    std::map<std::string, double> drive_amps;
    std::map<std::string, double> event_sums;
    std::vector<std::int32_t> cpu_v1_counts;
    std::vector<std::int32_t> cuda_v1_counts;
    std::vector<std::int32_t> cpu_hctx_counts;
    std::vector<std::int32_t> cuda_hctx_counts;
    std::vector<std::int32_t> cpu_hpred_counts;
    std::vector<std::int32_t> cuda_hpred_counts;
    std::int32_t cpu_total_v1_window_spikes;
    std::int32_t cuda_total_v1_window_spikes;
    std::int32_t cpu_total_hctx_window_spikes;
    std::int32_t cuda_total_hctx_window_spikes;
    std::int32_t cpu_total_hpred_window_spikes;
    std::int32_t cuda_total_hpred_window_spikes;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
    std::map<std::string, double> max_abs_error;
    double cpu_v1_soma_after_stim_step;
    double cuda_v1_soma_after_stim_step;
    double cpu_no_stim_v1_soma_after_stim_step;
    double cpu_v1_i_e_after_stim_scatter;
    double cuda_v1_i_e_after_stim_scatter;
    double cpu_v1_soma_after_stim_next_step;
    double cuda_v1_soma_after_stim_next_step;
    double cpu_no_stim_v1_soma_after_stim_next_step;
    double cpu_hctx_v_after_v1_step;
    double cuda_hctx_v_after_v1_step;
    double cpu_no_v1_hctx_v_after_v1_step;
    double cpu_hctx_i_e_after_v1_scatter;
    double cuda_hctx_i_e_after_v1_scatter;
    double cpu_hctx_v_after_v1_next_step;
    double cuda_hctx_v_after_v1_next_step;
    double cpu_no_v1_hctx_v_after_v1_next_step;
    double cpu_hpred_v_after_hctx_step;
    double cuda_hpred_v_after_hctx_step;
    double cpu_no_ctx_hpred_v_after_hctx_step;
    double cpu_hpred_i_e_after_hctx_scatter;
    double cuda_hpred_i_e_after_hctx_scatter;
    double cpu_hpred_v_after_hctx_next_step;
    double cuda_hpred_v_after_hctx_next_step;
    double cpu_no_ctx_hpred_v_after_hctx_next_step;
    double cpu_v1_soma_after_hpred_step;
    double cuda_v1_soma_after_hpred_step;
    double cpu_no_fb_v1_soma_after_hpred_step;
    double cpu_v1_i_ap_after_fb_scatter;
    double cuda_v1_i_ap_after_fb_scatter;
    double cpu_som_i_e_after_fb_scatter;
    double cuda_som_i_e_after_fb_scatter;
    double cpu_v1_ap_after_fb_next_step;
    double cuda_v1_ap_after_fb_next_step;
    double cpu_no_fb_v1_ap_after_fb_next_step;
    double cpu_som_v_after_fb_next_step;
    double cuda_som_v_after_fb_next_step;
    double cpu_no_fb_som_v_after_fb_next_step;
    double cpu_v1_soma_after_fb_late_step;
    double cuda_v1_soma_after_fb_late_step;
    double cpu_no_fb_v1_soma_after_fb_late_step;
};

ClosedLoopDeterministicCountTestResult run_closed_loop_deterministic_count_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    std::int32_t stim_pre_index,
    std::int32_t stim_step,
    std::int32_t v1_force_step,
    std::int32_t hctx_force_step,
    std::int32_t hpred_force_step,
    std::int32_t n_steps,
    std::int32_t window_start_step,
    std::int32_t window_end_step
);

struct FrozenRichterDeterministicTrialResult {
    std::int32_t n_steps;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, std::int32_t> edge_counts;
    std::map<std::string, std::int32_t> source_fanouts;
    std::map<std::string, std::int32_t> source_event_counts;
    std::map<std::string, double> drive_amps;
    std::map<std::string, double> event_sums;
    std::map<std::string, double> ordering_deltas;
    std::int32_t expected_stim_pre_index;
    std::int32_t unexpected_stim_pre_index;
    std::int32_t stim_period_steps;
    std::int32_t v1e_index;
    std::int32_t hctx_index;
    std::int32_t hpred_index;
    std::int32_t feedback_v1e_index;
    std::int32_t feedback_som_index;
    std::map<std::string, std::vector<std::int32_t>> cpu_raw_counts;
    std::map<std::string, std::vector<std::int32_t>> cuda_raw_counts;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
    std::map<std::string, double> max_abs_error;
};

FrozenRichterDeterministicTrialResult run_frozen_richter_deterministic_trial_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
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
    std::int32_t iti_end_step
);

struct FrozenRichterSeededSourceResult {
    std::int64_t seed;
    std::int32_t n_steps;
    double dt_ms;
    std::int32_t expected_channel;
    std::int32_t unexpected_channel;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, std::int32_t> edge_counts;
    std::map<std::string, std::int32_t> source_event_counts;
    std::map<std::string, double> rates_hz;
    std::map<std::string, std::vector<std::int32_t>> cpu_raw_counts;
    std::map<std::string, std::vector<std::int32_t>> cuda_raw_counts;
    std::map<std::string, std::vector<std::int32_t>> cpu_source_counts;
    std::map<std::string, std::vector<std::int32_t>> cuda_source_counts;
    std::vector<std::int32_t> cpu_v1_trailer_bin_channel_counts;
    std::vector<std::int32_t> cpu_v1_som_trailer_bin_channel_counts;
    std::vector<std::int32_t> cuda_v1_som_trailer_bin_channel_counts;
    std::vector<std::int32_t> cpu_v1_error_trailer_bin_channel_counts;
    std::vector<std::int32_t> cuda_v1_error_trailer_bin_channel_counts;
    std::vector<std::int32_t> cpu_v1_error_neg_trailer_bin_channel_counts;
    std::vector<std::int32_t> cuda_v1_error_neg_trailer_bin_channel_counts;
    std::vector<std::int32_t> cpu_hpred_preprobe_channel_counts;
    std::vector<std::int32_t> cpu_hpred_trailer_bin_total_counts;
    std::vector<std::int32_t> cpu_hpred_trailer_bin_channel_counts;
    std::vector<std::int32_t> cuda_hpred_preprobe_channel_counts;
    std::vector<std::int32_t> cpu_hpred_feedback_held_trailer_bin_total_counts;
    std::vector<std::int32_t> cpu_hpred_feedback_held_trailer_bin_channel_counts;
    std::vector<std::int32_t> cuda_hpred_feedback_held_trailer_bin_total_counts;
    std::vector<std::int32_t> cuda_hpred_feedback_held_trailer_bin_channel_counts;
    std::vector<double> cpu_hpred_feedback_normalized_trailer_bin_total_weights;
    std::vector<double> cpu_hpred_feedback_normalized_trailer_bin_channel_weights;
    std::vector<double> cuda_hpred_feedback_normalized_trailer_bin_total_weights;
    std::vector<double> cuda_hpred_feedback_normalized_trailer_bin_channel_weights;
    std::vector<std::int32_t> cpu_hpred_feedback_normalized_preprobe_zero;
    std::vector<std::int32_t> cpu_hpred_feedback_normalized_fallback_used;
    std::vector<std::int32_t> cpu_hpred_feedback_normalized_fallback_zero_template;
    std::vector<std::int32_t> cpu_hpred_feedback_normalized_leader_template_total_counts;
    std::vector<std::int32_t> cpu_hpred_feedback_normalized_leader_template_channel_counts;
    std::vector<std::int32_t> cuda_hpred_feedback_normalized_preprobe_zero;
    std::vector<std::int32_t> cuda_hpred_feedback_normalized_fallback_used;
    std::vector<std::int32_t> cuda_hpred_feedback_normalized_fallback_zero_template;
    std::vector<std::int32_t> cuda_hpred_feedback_normalized_leader_template_total_counts;
    std::vector<std::int32_t> cuda_hpred_feedback_normalized_leader_template_channel_counts;
    std::vector<double> cpu_v1_predicted_suppression_trailer_channel_signal_sum;
    std::vector<double> cpu_v1_predicted_suppression_trailer_channel_gain_sum;
    std::vector<double> cpu_v1_predicted_suppression_trailer_raw_ie_before_sum;
    std::vector<double> cpu_v1_predicted_suppression_trailer_raw_ie_after_sum;
    std::vector<double> cpu_v1_predicted_suppression_trailer_raw_ie_delta_sum;
    std::vector<double> cuda_v1_predicted_suppression_trailer_channel_signal_sum;
    std::vector<double> cuda_v1_predicted_suppression_trailer_channel_gain_sum;
    std::vector<double> cuda_v1_predicted_suppression_trailer_raw_ie_before_sum;
    std::vector<double> cuda_v1_predicted_suppression_trailer_raw_ie_after_sum;
    std::vector<double> cuda_v1_predicted_suppression_trailer_raw_ie_delta_sum;
    std::map<std::string, std::vector<double>> cpu_diagnostic_rates_hz;
    std::map<std::string, std::vector<double>> cuda_diagnostic_rates_hz;
    std::map<std::string, double> cpu_q_active_fC;
    std::map<std::string, double> cuda_q_active_fC;
    std::map<std::string, std::vector<double>> cpu_final_state;
    std::map<std::string, std::vector<double>> cuda_final_state;
    std::map<std::string, double> max_abs_error;
};

struct FrozenRichterSeededSourceBatchResult {
    std::int32_t n_conditions;
    std::vector<std::int32_t> v1_leader_total_counts;
    std::vector<std::int32_t> v1_preprobe_total_counts;
    std::vector<std::int32_t> v1_trailer_total_counts;
    std::vector<std::int32_t> v1_trailer_channel_counts;
    std::vector<std::int32_t> v1_trailer_bin_channel_counts;
    std::vector<std::int32_t> v1_som_trailer_total_counts;
    std::vector<std::int32_t> v1_som_trailer_channel_counts;
    std::vector<std::int32_t> v1_som_trailer_bin_channel_counts;
    std::vector<std::int32_t> v1_error_trailer_total_counts;
    std::vector<std::int32_t> v1_error_trailer_channel_counts;
    std::vector<std::int32_t> v1_error_trailer_bin_channel_counts;
    std::vector<std::int32_t> v1_error_neg_trailer_total_counts;
    std::vector<std::int32_t> v1_error_neg_trailer_channel_counts;
    std::vector<std::int32_t> v1_error_neg_trailer_bin_channel_counts;
    std::vector<double> v1e_q_active_fC_by_phase;
    std::vector<double> v1som_q_active_fC_by_phase;
    std::vector<double> v1error_q_active_fC_by_phase;
    std::vector<double> v1error_neg_q_active_fC_by_phase;
    std::vector<std::int32_t> hctx_preprobe_total_counts;
    std::vector<std::int32_t> hctx_trailer_total_counts;
    std::vector<std::int32_t> hpred_preprobe_total_counts;
    std::vector<std::int32_t> hpred_preprobe_channel_counts;
    std::vector<std::int32_t> hpred_trailer_total_counts;
    std::vector<std::int32_t> hpred_trailer_bin_total_counts;
    std::vector<std::int32_t> hpred_trailer_bin_channel_counts;
    std::vector<std::int32_t> hpred_feedback_held_trailer_bin_total_counts;
    std::vector<std::int32_t> hpred_feedback_held_trailer_bin_channel_counts;
    std::vector<double> hpred_feedback_normalized_trailer_bin_total_weights;
    std::vector<double> hpred_feedback_normalized_trailer_bin_channel_weights;
    std::vector<std::int32_t> hpred_feedback_normalized_preprobe_zero;
    std::vector<std::int32_t> hpred_feedback_normalized_fallback_used;
    std::vector<std::int32_t> hpred_feedback_normalized_fallback_zero_template;
    std::vector<std::int32_t> hpred_feedback_normalized_leader_template_total_counts;
    std::vector<std::int32_t> hpred_feedback_normalized_leader_template_channel_counts;
    std::vector<double> v1_predicted_suppression_trailer_channel_signal_sum;
    std::vector<double> v1_predicted_suppression_trailer_channel_gain_sum;
    std::vector<double> v1_predicted_suppression_trailer_raw_ie_before_sum;
    std::vector<double> v1_predicted_suppression_trailer_raw_ie_after_sum;
    std::vector<double> v1_predicted_suppression_trailer_raw_ie_delta_sum;
    std::vector<std::int32_t> source_total_counts;
};

FrozenRichterSeededSourceResult run_frozen_richter_seeded_source_cuda(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
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
    std::int32_t iti_end_step,
    const std::string& feedback_replay_mode = "raw",
    double feedback_replay_target_per_bin = 314.1666666666667,
    const std::string& feedback_replay_fallback_mode = "none",
    const std::vector<std::int32_t>& feedback_replay_leader_templates =
        std::vector<std::int32_t>(),
    double v1_som_to_e_scale = 1.0,
    double v1_som_divisive_scale = 0.0,
    double v1_direct_divisive_scale = 0.0,
    double v1_feedforward_divisive_scale = 0.0,
    std::int32_t v1_feedforward_divisive_gate_source_id = 0,
    std::int32_t v1_direct_divisive_gate_source_id = 0,
    double v1_predicted_suppression_scale = 0.0,
    double v1_predicted_suppression_neighbor_weight = 0.0,
    std::int32_t v1_predicted_suppression_locus_id = 0,
    double v1_stim_sigma_deg = 22.0,
    std::int32_t v1_error_comparator_mode_id = 0,
    double v1_error_sensory_gain = 1.0,
    double v1_error_prediction_gain = 1.0,
    std::int32_t v1_error_prediction_shift = 0
);

FrozenRichterSeededSourceBatchResult run_frozen_richter_seeded_source_cuda_batched(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    const std::vector<std::int64_t>& seeds,
    const std::vector<std::int32_t>& expected_channels,
    const std::vector<std::int32_t>& unexpected_channels,
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
    std::int32_t iti_end_step,
    const std::string& feedback_replay_mode = "raw",
    double feedback_replay_target_per_bin = 314.1666666666667,
    const std::string& feedback_replay_fallback_mode = "none",
    const std::vector<std::int32_t>& feedback_replay_leader_templates =
        std::vector<std::int32_t>(),
    double v1_som_to_e_scale = 1.0,
    double v1_som_divisive_scale = 0.0,
    double v1_direct_divisive_scale = 0.0,
    double v1_feedforward_divisive_scale = 0.0,
    std::int32_t v1_feedforward_divisive_gate_source_id = 0,
    std::int32_t v1_direct_divisive_gate_source_id = 0,
    double v1_predicted_suppression_scale = 0.0,
    double v1_predicted_suppression_neighbor_weight = 0.0,
    std::int32_t v1_predicted_suppression_locus_id = 0,
    double v1_stim_sigma_deg = 22.0,
    std::int32_t v1_error_comparator_mode_id = 0,
    double v1_error_sensory_gain = 1.0,
    double v1_error_prediction_gain = 1.0,
    std::int32_t v1_error_prediction_shift = 0
);

FrozenRichterSeededSourceResult run_frozen_richter_seeded_source_cpu(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
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
    std::int32_t iti_end_step,
    const std::string& feedback_replay_mode = "raw",
    double feedback_replay_target_per_bin = 314.1666666666667,
    const std::string& feedback_replay_fallback_mode = "none",
    const std::vector<std::int32_t>& feedback_replay_leader_templates =
        std::vector<std::int32_t>(),
    double v1_som_to_e_scale = 1.0,
    double v1_som_divisive_scale = 0.0,
    double v1_direct_divisive_scale = 0.0,
    double v1_feedforward_divisive_scale = 0.0,
    std::int32_t v1_feedforward_divisive_gate_source_id = 0,
    std::int32_t v1_direct_divisive_gate_source_id = 0,
    double v1_predicted_suppression_scale = 0.0,
    double v1_predicted_suppression_neighbor_weight = 0.0,
    std::int32_t v1_predicted_suppression_locus_id = 0,
    double v1_stim_sigma_deg = 22.0,
    std::int32_t v1_error_comparator_mode_id = 0,
    double v1_error_sensory_gain = 1.0,
    double v1_error_prediction_gain = 1.0,
    std::int32_t v1_error_prediction_shift = 0
);

FrozenRichterSeededSourceResult run_frozen_richter_seeded_source_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
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
    std::int32_t iti_end_step,
    std::int32_t v1_error_comparator_mode_id = 0,
    double v1_error_sensory_gain = 1.0,
    double v1_error_prediction_gain = 1.0,
    std::int32_t v1_error_prediction_shift = 0
);

FrozenRichterSeededSourceResult run_frozen_richter_controlled_source_test(
    const std::string& stim_bank_name,
    const std::vector<std::int32_t>& stim_pre,
    const std::vector<std::int32_t>& stim_post,
    const std::vector<double>& stim_weight,
    double stim_drive_amp,
    const std::vector<std::int32_t>& stim_channel,
    const std::string& v1_to_h_bank_name,
    const std::vector<std::int32_t>& v1_to_h_pre,
    const std::vector<std::int32_t>& v1_to_h_post,
    const std::vector<double>& v1_to_h_weight,
    double v1_to_h_drive_amp,
    const std::string& ctx_to_pred_bank_name,
    const std::vector<std::int32_t>& ctx_to_pred_pre,
    const std::vector<std::int32_t>& ctx_to_pred_post,
    const std::vector<double>& ctx_to_pred_weight,
    double ctx_to_pred_drive_amp,
    const std::string& feedback_direct_bank_name,
    const std::vector<std::int32_t>& feedback_direct_pre,
    const std::vector<std::int32_t>& feedback_direct_post,
    const std::vector<double>& feedback_direct_weight,
    double feedback_direct_drive_amp,
    const std::string& feedback_som_bank_name,
    const std::vector<std::int32_t>& feedback_som_pre,
    const std::vector<std::int32_t>& feedback_som_post,
    const std::vector<double>& feedback_som_weight,
    double feedback_som_drive_amp,
    const std::vector<std::int32_t>& event_steps,
    const std::vector<std::int32_t>& event_sources,
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
    std::int32_t iti_end_step
);

struct CtxPredPlasticityTestResult {
    std::int64_t seed;
    std::int32_t n_pre;
    std::int32_t n_post;
    std::int32_t n_syn;
    std::int32_t n_steps;
    double dt_ms;
    double tau_coinc_ms;
    double tau_elig_ms;
    double eta;
    double gamma;
    double w_target;
    double w_max;
    double w_row_max;
    double m_integral;
    double dt_trial_s;
    std::int32_t paired_pre;
    std::int32_t paired_post;
    std::int32_t pre_rule_pre;
    std::int32_t capped_pre;
    std::int32_t silent_pre;
    std::int32_t silent_post;
    std::int32_t cpu_n_capped;
    std::int32_t cuda_n_capped;
    std::vector<std::int32_t> pre_event_steps;
    std::vector<std::int32_t> pre_event_cells;
    std::vector<std::int32_t> post_event_steps;
    std::vector<std::int32_t> post_event_cells;
    std::vector<double> initial_w;
    std::vector<double> cpu_w;
    std::vector<double> cuda_w;
    std::vector<double> cpu_elig_before_gate;
    std::vector<double> cuda_elig_before_gate;
    std::vector<double> cpu_elig_after_gate;
    std::vector<double> cuda_elig_after_gate;
    std::vector<double> cpu_xpre_after_gate;
    std::vector<double> cuda_xpre_after_gate;
    std::vector<double> cpu_xpost_after_gate;
    std::vector<double> cuda_xpost_after_gate;
    std::vector<double> cpu_row_sums;
    std::vector<double> cuda_row_sums;
    std::map<std::string, double> max_abs_error;
};

CtxPredPlasticityTestResult run_ctx_pred_plasticity_test(
    std::int64_t seed,
    std::int32_t n_steps
);

struct CtxPredTrainingTrialSliceResult {
    std::int64_t seed;
    std::int32_t n_pre;
    std::int32_t n_post;
    std::int32_t n_syn;
    std::int32_t n_steps;
    double dt_ms;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, std::int32_t> event_counts;
    std::int32_t gate_step;
    std::int32_t leader_pre;
    std::int32_t boundary_pre;
    std::int32_t trailer_post;
    std::int32_t late_trailer_post;
    std::int32_t capped_pre;
    std::int32_t silent_pre;
    std::int32_t silent_post;
    std::int32_t cpu_n_capped;
    std::int32_t cuda_n_capped;
    std::vector<std::int32_t> hctx_pre_event_steps;
    std::vector<std::int32_t> hctx_pre_event_cells;
    std::vector<std::int32_t> hpred_post_event_steps;
    std::vector<std::int32_t> hpred_post_event_cells;
    std::vector<double> initial_w_ctx_pred;
    std::vector<double> cpu_w_ctx_pred_final;
    std::vector<double> cuda_w_ctx_pred_final;
    std::vector<double> cpu_elig_before_gate;
    std::vector<double> cuda_elig_before_gate;
    std::vector<double> cpu_elig_after_iti;
    std::vector<double> cuda_elig_after_iti;
    std::vector<double> cpu_xpre_after_iti;
    std::vector<double> cuda_xpre_after_iti;
    std::vector<double> cpu_xpost_after_iti;
    std::vector<double> cuda_xpost_after_iti;
    std::vector<double> cpu_row_sums;
    std::vector<double> cuda_row_sums;
    std::map<std::string, double> max_abs_error;
};

CtxPredTrainingTrialSliceResult run_ctx_pred_training_trial_slice_test(
    std::int64_t seed
);

struct CtxPredTinyTrainerTestResult {
    std::int64_t seed;
    std::int32_t schedule_variant;
    std::int32_t n_trials;
    std::int32_t n_pre;
    std::int32_t n_post;
    std::int32_t n_syn;
    std::int32_t h_ee_n_syn;
    std::int32_t n_steps;
    std::int32_t trial_steps;
    double dt_ms;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, std::int32_t> event_counts;
    std::vector<std::int32_t> gate_steps;
    std::vector<std::int32_t> trial_leader_pre_cells;
    std::vector<std::int32_t> trial_trailer_post_cells;
    std::vector<std::int32_t> hctx_pre_event_steps;
    std::vector<std::int32_t> hctx_pre_event_cells;
    std::vector<std::int32_t> hpred_post_event_steps;
    std::vector<std::int32_t> hpred_post_event_cells;
    std::vector<double> initial_w_ctx_pred;
    std::vector<double> cpu_w_ctx_pred_final;
    std::vector<double> cuda_w_ctx_pred_final;
    std::vector<double> cpu_ctx_ee_w_final;
    std::vector<double> cuda_ctx_ee_w_final;
    std::vector<double> cpu_pred_ee_w_final;
    std::vector<double> cuda_pred_ee_w_final;
    std::vector<double> cpu_elig_after_training;
    std::vector<double> cuda_elig_after_training;
    std::vector<double> cpu_xpre_after_training;
    std::vector<double> cuda_xpre_after_training;
    std::vector<double> cpu_xpost_after_training;
    std::vector<double> cuda_xpost_after_training;
    std::vector<double> cpu_row_sums;
    std::vector<double> cuda_row_sums;
    std::vector<double> cpu_gate_w_before;
    std::vector<double> cuda_gate_w_before;
    std::vector<double> cpu_gate_w_after;
    std::vector<double> cuda_gate_w_after;
    std::vector<double> cpu_gate_dw_sum;
    std::vector<double> cuda_gate_dw_sum;
    std::vector<double> cpu_gate_elig_mean;
    std::vector<double> cuda_gate_elig_mean;
    std::vector<double> cpu_gate_elig_max;
    std::vector<double> cuda_gate_elig_max;
    std::vector<double> cpu_gate_row_sum_max;
    std::vector<double> cuda_gate_row_sum_max;
    std::vector<std::int32_t> cpu_gate_n_capped;
    std::vector<std::int32_t> cuda_gate_n_capped;
    std::map<std::string, double> max_abs_error;
};

CtxPredTinyTrainerTestResult run_ctx_pred_tiny_trainer_test(
    std::int64_t seed,
    std::int32_t schedule_variant
);

CtxPredTinyTrainerTestResult run_ctx_pred_generated_schedule_test(
    std::int64_t seed,
    const std::vector<std::int32_t>& leader_pre_cells,
    const std::vector<std::int32_t>& trailer_post_cells
);

struct Stage1HGateDynamicsResult {
    std::int64_t seed;
    std::int32_t n_trials;
    std::int32_t n_e;
    std::int32_t n_inh;
    std::int32_t n_steps_per_trial;
    double dt_ms;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, double> metrics;
    std::map<std::string, double> max_abs_error;
    std::vector<std::int32_t> leader_channels;
    std::vector<std::int32_t> trailer_channels;
    std::vector<double> cpu_ctx_persistence_ms_by_trial;
    std::vector<double> cuda_ctx_persistence_ms_by_trial;
    std::vector<std::int32_t> cpu_pred_pretrailer_target_counts;
    std::vector<std::int32_t> cuda_pred_pretrailer_target_counts;
    std::vector<std::int32_t> cpu_pred_pretrailer_channel_counts;
    std::vector<std::int32_t> cuda_pred_pretrailer_channel_counts;
    std::vector<std::int32_t> cpu_pred_pretrailer_channel_counts_by_bin;
    std::vector<std::int32_t> cuda_pred_pretrailer_channel_counts_by_bin;
    std::vector<std::int32_t> cpu_ctx_total_counts;
    std::vector<std::int32_t> cuda_ctx_total_counts;
    std::vector<std::int32_t> cpu_pred_total_counts;
    std::vector<std::int32_t> cuda_pred_total_counts;
    std::vector<std::int32_t> cpu_ctx_inh_total_counts;
    std::vector<std::int32_t> cuda_ctx_inh_total_counts;
    std::vector<std::int32_t> cpu_pred_inh_total_counts;
    std::vector<std::int32_t> cuda_pred_inh_total_counts;
};

Stage1HGateDynamicsResult run_stage1_h_gate_dynamics_test(
    std::int64_t seed,
    const std::vector<std::int32_t>& leader_cells,
    const std::vector<std::int32_t>& trailer_cells,
    const std::vector<double>& w_ctx_pred
);

struct NativeStage1TrainResult {
    std::int64_t seed;
    std::string prediction_target;
    std::int32_t n_trials;
    std::int32_t n_pre;
    std::int32_t n_post;
    std::int32_t n_syn;
    std::int32_t n_steps;
    std::int32_t trial_steps;
    double dt_ms;
    std::map<std::string, std::int32_t> phase_steps;
    std::map<std::string, std::int32_t> event_counts;
    std::vector<double> w_ctx_pred_final;
    std::vector<double> row_sums;
    std::vector<double> gate_w_before;
    std::vector<double> gate_w_after;
    std::vector<double> gate_dw_sum;
    std::vector<double> gate_elig_mean;
    std::vector<double> gate_elig_max;
    std::vector<double> gate_row_sum_max;
    std::vector<std::int32_t> gate_n_capped;
    std::vector<double> w_hpred_v1direct_final;
    std::vector<double> w_hpred_v1direct_row_sums;
    std::vector<double> w_hpred_v1som_final;
    std::vector<double> w_hpred_v1som_row_sums;
};

NativeStage1TrainResult run_native_stage1_generated_train(
    std::int64_t seed,
    const std::vector<std::int32_t>& leader_cells,
    const std::vector<std::int32_t>& expected_trailer_cells,
    const std::string& prediction_target = "orientation_cell"
);

}  // namespace expectation_snn_cuda
