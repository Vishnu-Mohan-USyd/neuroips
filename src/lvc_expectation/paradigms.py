"""Synthetic paradigm generation for phase-1 expectation assays."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .geometry import OrientationGeometry
from .types import TrialBatch

TASK_MODES = {"orientation_relevant": 0, "orthogonal_relevant": 1}
PRESTIM_MODES = {"none": 0, "cue_only": 1, "context_only": 2, "omission": 3}
CONDITION_CODES = {"expected": 0, "unexpected": 1, "neutral": 2, "prestim": 3, "omission": 4}

CONDITION_EXPECTED = CONDITION_CODES["expected"]
CONDITION_UNEXPECTED = CONDITION_CODES["unexpected"]
CONDITION_NEUTRAL = CONDITION_CODES["neutral"]
PRESTIM_NONE = PRESTIM_MODES["none"]
PRESTIM_CUE_ONLY = PRESTIM_MODES["cue_only"]
PRESTIM_CONTEXT_ONLY = PRESTIM_MODES["context_only"]
PRESTIM_OMISSION = PRESTIM_MODES["omission"]
LOCAL_GLOBAL_PROBE_CONTROLLED_SOURCES = (0, 1, 2, 3, 4, 5)
LOCAL_GLOBAL_PROBE_CONTEXTS = (0, 1)
LOCAL_GLOBAL_PROBE_OFFSETS = (1, -1)
LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX = 1


@dataclass
class NeutralMatchReport:
    orientation_counts: dict[int, int]
    condition_counts: dict[str, int]
    context_counts: dict[int, int]
    task_counts: dict[str, int]
    prestim_counts: dict[str, int]
    omission_count: int
    orthogonal_event_count: int
    controlled_source_count: int
    transition_counts: dict[str, int]
    repetition_lags: list[int]
    run_lengths: list[int]


def _resolve_controlled_sources(
    geometry: OrientationGeometry,
    controlled_sources: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if controlled_sources is None:
        return torch.arange(geometry.n_orientations, dtype=torch.long)
    resolved = torch.as_tensor(controlled_sources, dtype=torch.long).flatten()
    if resolved.numel() == 0:
        raise ValueError("controlled_sources must contain at least one orientation index")
    if resolved.lt(0).any() or resolved.ge(geometry.n_orientations).any():
        raise ValueError("controlled_sources must lie within the orientation ring")
    return torch.unique(resolved, sorted=True)


def build_transition_matrices(
    geometry: OrientationGeometry,
    p_expected: float = 0.8,
    controlled_sources: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> dict[int, torch.Tensor]:
    if not 0.5 < p_expected < 1.0:
        raise ValueError("p_expected must be in (0.5, 1.0) for counterbalanced contexts")
    n = geometry.n_orientations
    controlled = _resolve_controlled_sources(geometry, controlled_sources)
    controlled_mask = torch.zeros(n, dtype=torch.bool)
    controlled_mask[controlled] = True
    matrices: dict[int, torch.Tensor] = {2: torch.full((n, n), 1.0 / n, dtype=torch.float32)}
    for context_id, preferred_shift in ((0, 1), (1, -1)):
        matrix = torch.zeros((n, n), dtype=torch.float32)
        for src in range(n):
            clockwise = geometry.wrap_index(src + 1)
            counterclockwise = geometry.wrap_index(src - 1)
            if controlled_mask[src]:
                preferred = geometry.wrap_index(src + preferred_shift)
                alternate = geometry.wrap_index(src - preferred_shift)
                matrix[src, preferred] = p_expected
                matrix[src, alternate] = 1.0 - p_expected
            else:
                matrix[src, clockwise] = 0.5
                matrix[src, counterclockwise] = 0.5
        matrices[context_id] = matrix
    return matrices


def _sample_task_mode(generator: torch.Generator) -> int:
    return int(torch.randint(low=0, high=len(TASK_MODES), size=(1,), generator=generator).item())


def _sample_prestim_mode(generator: torch.Generator) -> int:
    return int(torch.randint(low=0, high=len(PRESTIM_MODES), size=(1,), generator=generator).item())


def _run_lengths(sequence: torch.Tensor) -> list[int]:
    output: list[int] = []
    current = None
    current_len = 0
    for value in sequence.tolist():
        if value < 0:
            continue
        if current is None or current != value:
            if current_len:
                output.append(current_len)
            current = value
            current_len = 1
        else:
            current_len += 1
    if current_len:
        output.append(current_len)
    return output


def _repetition_lags(sequence: torch.Tensor) -> list[int]:
    last_seen: dict[int, int] = {}
    lags: list[int] = []
    visible_index = 0
    for value in sequence.tolist():
        if value < 0:
            continue
        if value in last_seen:
            lags.append(visible_index - last_seen[value])
        last_seen[value] = visible_index
        visible_index += 1
    return lags


def generate_trial_batch(
    config: ExperimentConfig,
    batch_size: int,
    seed: int,
    controlled_sources: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ambiguity_weights: torch.Tensor | None = None,
) -> TrialBatch:
    geometry = OrientationGeometry(config.geometry.n_orientations)
    resolved_controlled_sources = _resolve_controlled_sources(geometry, controlled_sources)
    controlled_source_mask = torch.zeros(geometry.n_orientations, dtype=torch.bool)
    controlled_source_mask[resolved_controlled_sources] = True
    transitions = build_transition_matrices(
        geometry,
        p_expected=config.sequence.expected_transition_probability,
        controlled_sources=resolved_controlled_sources,
    )
    generator = torch.Generator().manual_seed(seed)
    bsz, steps, n = batch_size, config.sequence.n_steps, geometry.n_orientations
    prestim_steps = config.sequence.prestim_steps

    orientations = torch.full((bsz, steps), -1, dtype=torch.long)
    blank_mask = torch.zeros((bsz, steps), dtype=torch.bool)
    expected_mask = torch.zeros((bsz, steps), dtype=torch.bool)
    context_ids = torch.randint(low=0, high=3, size=(bsz,), generator=generator)
    task_mode = torch.tensor([_sample_task_mode(generator) for _ in range(bsz)], dtype=torch.long)
    prestim_mode = torch.tensor([_sample_prestim_mode(generator) for _ in range(bsz)], dtype=torch.long)
    orthogonal_events = torch.bernoulli(
        torch.full((bsz, steps), config.sequence.orthogonal_event_probability),
        generator=generator,
    ).to(torch.long)
    condition_codes = torch.full((bsz, steps), CONDITION_CODES["prestim"], dtype=torch.long)
    expected_distribution = torch.zeros((bsz, steps, n), dtype=torch.float32)
    omission_targets = torch.full((bsz, steps), -1, dtype=torch.long)
    source_orientations = torch.full((bsz, steps), -1, dtype=torch.long)
    visible_step_index = torch.full((bsz, steps), -1, dtype=torch.long)
    repetition_lag = torch.full((bsz, steps), -1, dtype=torch.long)
    run_length = torch.full((bsz, steps), -1, dtype=torch.long)
    controlled_source_steps = torch.zeros((bsz, steps), dtype=torch.bool)

    for trial_idx in range(bsz):
        context_id = int(context_ids[trial_idx].item())
        matrix = transitions[context_id]
        first_orientation = int(torch.randint(low=0, high=n, size=(1,), generator=generator).item())
        first_step = prestim_steps if prestim_mode[trial_idx].item() in (PRESTIM_MODES["cue_only"], PRESTIM_MODES["context_only"]) else 0
        last_seen: dict[int, int] = {}
        visible_count = 0
        current_run_value = None
        current_run_length = 0
        if first_step > 0:
            blank_mask[trial_idx, :first_step] = True
            expected_distribution[trial_idx, :first_step, first_orientation] = 1.0
        orientations[trial_idx, first_step] = first_orientation
        condition_codes[trial_idx, first_step] = CONDITION_CODES["neutral"] if context_id == 2 else CONDITION_CODES["expected"]
        expected_distribution[trial_idx, first_step, first_orientation] = 1.0
        visible_step_index[trial_idx, first_step] = visible_count
        run_length[trial_idx, first_step] = 1
        last_seen[first_orientation] = visible_count
        visible_count += 1
        current_run_value = first_orientation
        current_run_length = 1
        previous_visible = first_orientation

        for step in range(first_step + 1, steps):
            probs = matrix[previous_visible]
            source_orientations[trial_idx, step] = previous_visible
            controlled_source_steps[trial_idx, step] = controlled_source_mask[previous_visible]
            sampled_orientation = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            expected_distribution[trial_idx, step] = probs
            is_omission = prestim_mode[trial_idx].item() == PRESTIM_MODES["omission"] and step == first_step + 1
            if is_omission:
                blank_mask[trial_idx, step] = True
                omission_targets[trial_idx, step] = sampled_orientation
                condition_codes[trial_idx, step] = CONDITION_CODES["omission"]
                previous_visible = sampled_orientation
                continue
            orientations[trial_idx, step] = sampled_orientation
            previous_visible = sampled_orientation
            visible_step_index[trial_idx, step] = visible_count
            if sampled_orientation in last_seen:
                repetition_lag[trial_idx, step] = visible_count - last_seen[sampled_orientation]
            last_seen[sampled_orientation] = visible_count
            visible_count += 1
            if current_run_value == sampled_orientation:
                current_run_length += 1
            else:
                current_run_value = sampled_orientation
                current_run_length = 1
            run_length[trial_idx, step] = current_run_length
            if context_id == 2:
                condition_codes[trial_idx, step] = CONDITION_CODES["neutral"]
            elif sampled_orientation == int(torch.argmax(probs).item()):
                expected_mask[trial_idx, step] = True
                condition_codes[trial_idx, step] = CONDITION_CODES["expected"]
            else:
                condition_codes[trial_idx, step] = CONDITION_CODES["unexpected"]

    metadata = {
        "condition_codes": condition_codes,
        "expected_distribution": expected_distribution,
        "omission_targets": omission_targets,
        "task_mode_names": {value: key for key, value in TASK_MODES.items()},
        "prestim_mode_names": {value: key for key, value in PRESTIM_MODES.items()},
        "condition_names": {value: key for key, value in CONDITION_CODES.items()},
        "generator_seed": int(seed),
        "controlled_sources": resolved_controlled_sources,
        "controlled_source_steps": controlled_source_steps,
        "source_orientations": source_orientations,
        "visible_step_index": visible_step_index,
        "repetition_lag": repetition_lag,
        "run_length": run_length,
        "ambiguity_weights": ambiguity_weights,
        "transition_matrices": transitions,
    }
    return TrialBatch(
        orientations=orientations,
        blank_mask=blank_mask,
        expected_mask=expected_mask,
        context_ids=context_ids,
        task_mode=task_mode,
        prestim_mode=prestim_mode,
        orthogonal_events=orthogonal_events,
        metadata=metadata,
    )


def _resolve_local_global_probe_sources(
    geometry: OrientationGeometry,
    probe_source_subset: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    default_sources = torch.tensor(LOCAL_GLOBAL_PROBE_CONTROLLED_SOURCES, dtype=torch.long)
    if geometry.n_orientations <= default_sources.max().item():
        raise ValueError("local-global probe requires an orientation ring that contains sources 0..5")
    if probe_source_subset is None:
        return default_sources
    resolved = _resolve_controlled_sources(geometry, probe_source_subset)
    default_mask = torch.zeros(geometry.n_orientations, dtype=torch.bool)
    default_mask[default_sources] = True
    if not default_mask[resolved].all():
        raise ValueError("local-global probe source subset must be drawn from [0, 1, 2, 3, 4, 5]")
    return resolved


def _resolve_local_global_probe_offsets(local_offset_family: int | None = None) -> tuple[int, ...]:
    if local_offset_family is None:
        return LOCAL_GLOBAL_PROBE_OFFSETS
    if local_offset_family not in LOCAL_GLOBAL_PROBE_OFFSETS:
        raise ValueError(f"local_offset_family must be one of {LOCAL_GLOBAL_PROBE_OFFSETS} when provided")
    return (int(local_offset_family),)


def generate_local_global_probe_batch(
    config: ExperimentConfig,
    seed: int,
    probe_source_subset: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    local_offset_family: int | None = None,
) -> TrialBatch:
    geometry = OrientationGeometry(config.geometry.n_orientations)
    resolved_probe_sources = _resolve_local_global_probe_sources(geometry, probe_source_subset)
    resolved_probe_offsets = _resolve_local_global_probe_offsets(local_offset_family)
    transitions = build_transition_matrices(
        geometry,
        p_expected=config.sequence.expected_transition_probability,
        controlled_sources=resolved_probe_sources,
    )
    steps = config.sequence.n_steps
    if steps <= LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX:
        raise ValueError("local-global probe requires at least two sequence steps")

    probe_specs: list[tuple[int, int, int, int, int, bool]] = []
    pair_id_lookup: dict[tuple[int, int], int] = {}
    pair_id = 0
    for source in resolved_probe_sources.tolist():
        for offset in LOCAL_GLOBAL_PROBE_OFFSETS:
            target = geometry.wrap_index(source + offset)
            pair_id_lookup[(source, target)] = pair_id
            if offset not in resolved_probe_offsets:
                pair_id += 1
                continue
            for context_id in LOCAL_GLOBAL_PROBE_CONTEXTS:
                expected_target = int(torch.argmax(transitions[context_id][source]).item())
                is_expected = target == expected_target
                probe_specs.append((source, target, offset, context_id, pair_id, is_expected))
            pair_id += 1

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(probe_specs), generator=generator)
    ordered_specs = [probe_specs[int(index)] for index in order.tolist()]

    bsz, n = len(ordered_specs), geometry.n_orientations
    orientations = torch.full((bsz, steps), -1, dtype=torch.long)
    blank_mask = torch.ones((bsz, steps), dtype=torch.bool)
    expected_mask = torch.zeros((bsz, steps), dtype=torch.bool)
    context_ids = torch.tensor([spec[3] for spec in ordered_specs], dtype=torch.long)
    task_mode = torch.full((bsz,), TASK_MODES["orientation_relevant"], dtype=torch.long)
    prestim_mode = torch.full((bsz,), PRESTIM_MODES["none"], dtype=torch.long)
    orthogonal_events = torch.zeros((bsz, steps), dtype=torch.long)
    condition_codes = torch.full((bsz, steps), CONDITION_CODES["prestim"], dtype=torch.long)
    expected_distribution = torch.zeros((bsz, steps, n), dtype=torch.float32)
    omission_targets = torch.full((bsz, steps), -1, dtype=torch.long)
    source_orientations = torch.full((bsz, steps), -1, dtype=torch.long)
    visible_step_index = torch.full((bsz, steps), -1, dtype=torch.long)
    repetition_lag = torch.full((bsz, steps), -1, dtype=torch.long)
    run_length = torch.full((bsz, steps), -1, dtype=torch.long)
    controlled_source_steps = torch.zeros((bsz, steps), dtype=torch.bool)

    probe_step_mask = torch.zeros((bsz, steps), dtype=torch.bool)
    probe_valid_mask = torch.zeros((bsz, steps), dtype=torch.bool)
    probe_source_orientation = torch.full((bsz,), -1, dtype=torch.long)
    probe_target_orientation = torch.full((bsz,), -1, dtype=torch.long)
    probe_local_offset_bins = torch.zeros((bsz,), dtype=torch.long)
    probe_pair_id = torch.full((bsz,), -1, dtype=torch.long)
    probe_global_expected_mask = torch.zeros((bsz, steps), dtype=torch.bool)
    probe_global_unexpected_mask = torch.zeros((bsz, steps), dtype=torch.bool)

    for trial_idx, (source, target, offset, context_id, current_pair_id, is_expected) in enumerate(ordered_specs):
        orientations[trial_idx, 0] = source
        orientations[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = target
        blank_mask[trial_idx, 0] = False
        blank_mask[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = False
        condition_codes[trial_idx, 0] = CONDITION_CODES["neutral"]
        condition_codes[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = (
            CONDITION_CODES["expected"] if is_expected else CONDITION_CODES["unexpected"]
        )
        expected_mask[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = is_expected
        expected_distribution[trial_idx, 0, source] = 1.0
        expected_distribution[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = transitions[context_id][source]
        source_orientations[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = source
        visible_step_index[trial_idx, 0] = 0
        visible_step_index[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX
        run_length[trial_idx, 0] = 1
        run_length[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = 2 if target == source else 1
        repetition_lag[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = 1 if target == source else -1
        controlled_source_steps[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = True

        probe_step_mask[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = True
        probe_valid_mask[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = True
        probe_source_orientation[trial_idx] = source
        probe_target_orientation[trial_idx] = target
        probe_local_offset_bins[trial_idx] = offset
        probe_pair_id[trial_idx] = current_pair_id
        if is_expected:
            probe_global_expected_mask[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = True
        else:
            probe_global_unexpected_mask[trial_idx, LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX] = True

    pair_context_counts: dict[int, dict[int, int]] = {}
    for trial_idx, context_id in enumerate(context_ids.tolist()):
        current_pair_id = int(probe_pair_id[trial_idx].item())
        if current_pair_id not in pair_context_counts:
            pair_context_counts[current_pair_id] = {0: 0, 1: 0}
        pair_context_counts[current_pair_id][int(context_id)] += 1

    pair_descriptors = {
        str(pair_id_lookup[(source, geometry.wrap_index(source + offset))]): {
            "source": int(source),
            "target": int(geometry.wrap_index(source + offset)),
            "local_offset_bins": int(offset),
            "symmetry_mate_pair_id": int(pair_id_lookup[(source, geometry.wrap_index(source - offset))]),
        }
        for source in resolved_probe_sources.tolist()
        for offset in LOCAL_GLOBAL_PROBE_OFFSETS
    }

    metadata = {
        "condition_codes": condition_codes,
        "expected_distribution": expected_distribution,
        "omission_targets": omission_targets,
        "task_mode_names": {value: key for key, value in TASK_MODES.items()},
        "prestim_mode_names": {value: key for key, value in PRESTIM_MODES.items()},
        "condition_names": {value: key for key, value in CONDITION_CODES.items()},
        "generator_seed": int(seed),
        "controlled_sources": resolved_probe_sources,
        "controlled_source_steps": controlled_source_steps,
        "source_orientations": source_orientations,
        "visible_step_index": visible_step_index,
        "repetition_lag": repetition_lag,
        "run_length": run_length,
        "transition_matrices": transitions,
        "probe_source_subset": resolved_probe_sources,
        "probe_step_mask": probe_step_mask,
        "probe_valid_mask": probe_valid_mask,
        "probe_source_orientation": probe_source_orientation,
        "probe_target_orientation": probe_target_orientation,
        "probe_local_offset_bins": probe_local_offset_bins,
        "probe_pair_id": probe_pair_id,
        "probe_global_expected_mask": probe_global_expected_mask,
        "probe_global_unexpected_mask": probe_global_unexpected_mask,
        "probe_visible_step_index": LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX,
        "probe_offset_family": (
            torch.tensor(int(local_offset_family), dtype=torch.long)
            if local_offset_family is not None
            else None
        ),
        "probe_report": {
            "fixed_probe_visible_step_index": LOCAL_GLOBAL_PROBE_VISIBLE_STEP_INDEX,
            "contexts_used": list(LOCAL_GLOBAL_PROBE_CONTEXTS),
            "controlled_sources": resolved_probe_sources.tolist(),
            "probe_source_subset": resolved_probe_sources.tolist(),
            "local_offset_family": int(local_offset_family) if local_offset_family is not None else None,
            "pair_context_counts": {str(key): value for key, value in pair_context_counts.items()},
            "pair_descriptors": pair_descriptors,
        },
    }
    return TrialBatch(
        orientations=orientations,
        blank_mask=blank_mask,
        expected_mask=expected_mask,
        context_ids=context_ids,
        task_mode=task_mode,
        prestim_mode=prestim_mode,
        orthogonal_events=orthogonal_events,
        metadata=metadata,
    )


class Phase1ParadigmGenerator:
    """Compatibility wrapper exposing the agreed phase-1 generator API."""

    def __init__(
        self,
        config: ExperimentConfig,
        controlled_sources: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.config = config
        self.controlled_sources = controlled_sources

    def generate_batch(
        self,
        batch_size: int,
        seed: int | None = None,
        ambiguity_weights: torch.Tensor | None = None,
    ) -> TrialBatch:
        resolved_seed = self.config.sequence.seed if seed is None else seed
        return generate_trial_batch(
            self.config,
            batch_size=batch_size,
            seed=resolved_seed,
            controlled_sources=self.controlled_sources,
            ambiguity_weights=ambiguity_weights,
        )

    def generate_local_global_probe_batch(
        self,
        seed: int | None = None,
        probe_source_subset: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        local_offset_family: int | None = None,
    ) -> TrialBatch:
        resolved_seed = self.config.sequence.seed if seed is None else seed
        return generate_local_global_probe_batch(
            self.config,
            seed=resolved_seed,
            probe_source_subset=probe_source_subset,
            local_offset_family=local_offset_family,
        )

    def build_neutral_match_report(self, batch: TrialBatch) -> NeutralMatchReport:
        return compute_neutral_match_report(batch)

    def run_nuisance_only_failure_test(self, batch: TrialBatch):
        from .sanity import run_nuisance_only_failure_test

        return run_nuisance_only_failure_test(batch)

    def run_predictive_structure_success_test(self, batch: TrialBatch):
        from .sanity import run_predictive_structure_success_test

        return run_predictive_structure_success_test(batch)


def compute_neutral_match_report(batch: TrialBatch) -> NeutralMatchReport:
    orientations = batch.orientations[batch.orientations.ge(0)].tolist()
    orientation_counts: dict[int, int] = {}
    for orientation in orientations:
        orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1

    condition_names = batch.metadata["condition_names"]
    condition_codes = batch.metadata["condition_codes"]
    condition_counts = {condition_names[idx]: int(condition_codes.eq(idx).sum().item()) for idx in condition_names}
    context_counts = {int(idx): int(batch.context_ids.eq(idx).sum().item()) for idx in range(3)}
    task_names = batch.metadata["task_mode_names"]
    task_counts = {task_names[idx]: int(batch.task_mode.eq(idx).sum().item()) for idx in task_names}
    prestim_names = batch.metadata["prestim_mode_names"]
    prestim_counts = {prestim_names[idx]: int(batch.prestim_mode.eq(idx).sum().item()) for idx in prestim_names}
    omission_count = int(batch.blank_mask.sum().item())
    orthogonal_event_count = int(batch.orthogonal_events.sum().item())
    controlled_source_count = int(batch.metadata["controlled_source_steps"].sum().item())

    transition_counts: dict[str, int] = {}
    repetition_lags: list[int] = []
    run_lengths: list[int] = []
    for sequence in batch.orientations:
        visible = sequence[sequence.ge(0)]
        if visible.numel() >= 2:
            for src, dst in zip(visible[:-1].tolist(), visible[1:].tolist()):
                key = f"{src}->{dst}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
        repetition_lags.extend(_repetition_lags(sequence))
        run_lengths.extend(_run_lengths(sequence))

    return NeutralMatchReport(
        orientation_counts=orientation_counts,
        condition_counts=condition_counts,
        context_counts=context_counts,
        task_counts=task_counts,
        prestim_counts=prestim_counts,
        omission_count=omission_count,
        orthogonal_event_count=orthogonal_event_count,
        controlled_source_count=controlled_source_count,
        transition_counts=transition_counts,
        repetition_lags=repetition_lags,
        run_lengths=run_lengths,
    )
