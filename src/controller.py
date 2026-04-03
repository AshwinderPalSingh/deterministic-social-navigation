from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

try:
    from .risk_field import StaticRiskProfile
except ImportError:
    from risk_field import StaticRiskProfile


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is not None:
        return _normalize(np.asarray(fallback, dtype=float))
    return np.zeros(2, dtype=float)


def _project_to_goal_cone(
    direction: np.ndarray,
    goal_direction: np.ndarray,
    min_alignment: float,
) -> np.ndarray:
    target = _normalize(direction, goal_direction)
    goal = _normalize(goal_direction, target)
    alignment = float(np.dot(target, goal))
    clamped_alignment = float(np.clip(min_alignment, -1.0, 1.0))
    if alignment >= clamped_alignment:
        return target.copy()

    lateral = target - alignment * goal
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-9:
        return goal.copy()

    max_lateral = float(np.sqrt(max(1.0 - clamped_alignment**2, 0.0)))
    projected = clamped_alignment * goal + max_lateral * (lateral / lateral_norm)
    return _normalize(projected, goal)


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    base = _normalize(vector)
    return np.array([-base[1], base[0]], dtype=float)


class FSMState(str, Enum):
    GOAL_SEEK = "GOAL_SEEK"
    STATIC_ESCAPE = "STATIC_ESCAPE"
    HUMAN_YIELD = "HUMAN_YIELD"
    HARD_STOP = "HARD_STOP"


@dataclass(frozen=True)
class HumanDecision:
    state: FSMState | None
    speed_scale: float
    hard_stop_imminent: bool
    tracked_index: int
    tracked_clearance: float
    tracked_ttc: float
    nearest_clearance: float
    interaction_level: float
    combined_interaction: float
    active_indices: list[int]
    interaction_strengths: list[float]
    interaction_distances: list[float]
    interaction_ttc: list[float]
    interaction_alignments: list[float]


@dataclass(frozen=True)
class StaticEscapeDecision:
    active: bool
    direction: np.ndarray
    escape_direction_memory: np.ndarray
    escape_commit_steps_remaining: int
    cooldown_steps: int = 0


def evaluate_human_speed_control(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    goal_dir: np.ndarray,
    humans: Sequence[object],
    robot_radius: float,
    previous_human_active: bool,
    previous_human_hard_stop: bool,
    min_speed_scale: float,
    stop_enter_clearance: float,
    stop_exit_clearance: float,
    yield_enter_clearance: float,
    yield_exit_clearance: float,
    hard_stop_ttc_enter: float,
    hard_stop_ttc_exit: float,
    yield_ttc_enter: float,
    yield_ttc_exit: float,
    interaction_speed_threshold: float,
    interaction_enter_threshold: float,
    interaction_exit_threshold: float,
    relevance_enter_clearance: float,
    relevance_exit_clearance: float,
    relevance_gain: float,
    interaction_beta: float,
    previous_interaction_level: float,
    previous_dominant_index: int,
    dominant_memory_active: bool,
    interaction_smoothing_alpha: float,
    interaction_current_blend: float,
    interaction_memory_blend: float,
    interaction_memory_gain: float,
    interaction_memory_floor_ratio: float,
    interaction_decay_rate: float,
    interaction_fast_decay: float,
    interaction_effective_cap: float,
    interaction_min_strength: float,
    interaction_max_active_humans: int,
    interaction_memory_enabled: bool,
    topk_filter: bool,
    weak_suppression: bool,
    top2_gap_threshold: float,
    dominant_strength_floor: float,
    persistence_active: bool,
    persistence_clearance_margin: float,
    persistence_ttc_margin: float,
) -> HumanDecision:
    smoothing_alpha = float(np.clip(interaction_smoothing_alpha, 0.0, 1.0))
    current_blend = float(np.clip(interaction_current_blend, 0.0, 1.0))
    memory_blend = float(np.clip(interaction_memory_blend, 0.0, 1.0))
    memory_gain = max(float(interaction_memory_gain), 0.0)
    memory_floor_ratio = float(np.clip(interaction_memory_floor_ratio, 0.0, 1.0))
    decay_rate = float(np.clip(interaction_decay_rate, 0.0, 1.0))
    fast_decay = float(np.clip(interaction_fast_decay, 0.0, 1.0))
    effective_cap = float(np.clip(interaction_effective_cap, 0.0, 1.0))
    min_strength = max(float(interaction_min_strength), 0.0)
    max_active_humans = max(int(interaction_max_active_humans), 1)
    memory_enabled = bool(interaction_memory_enabled)
    use_topk_filter = bool(topk_filter)
    weak_gate = bool(weak_suppression)
    nearest_clearance = float("inf")
    hard_stop_imminent = False
    tracked_index = -1
    tracked_clearance = float("inf")
    tracked_ttc = float("inf")
    tracked_score = 0.0
    interaction_strengths: list[float] = []
    interaction_distances: list[float] = []
    interaction_ttc: list[float] = []
    interaction_alignments: list[float] = []
    closing_speeds: list[float] = []
    previous_effective_level = previous_interaction_level if memory_enabled else 0.0
    previously_engaged = previous_human_active or (persistence_active and memory_enabled)
    relevance_clearance_threshold = (
        relevance_exit_clearance if previously_engaged else relevance_enter_clearance
    )
    relevance_ttc_threshold = yield_ttc_exit if previously_engaged else yield_ttc_enter
    interaction_threshold = (
        interaction_exit_threshold if previously_engaged else interaction_enter_threshold
    )
    closing_speed_reference = max(
        relevance_clearance_threshold / max(relevance_ttc_threshold, 1e-6),
        max(2.0 * interaction_speed_threshold, 0.25),
    )

    for index, human in enumerate(humans):
        relative_position = np.asarray(human.position, dtype=float) - position
        distance = float(np.linalg.norm(relative_position))
        clearance = max(distance - (robot_radius + float(human.radius)), 0.0)
        nearest_clearance = min(nearest_clearance, clearance)

        relative_velocity = np.asarray(human.velocity, dtype=float) - velocity
        closing_projection = float(np.dot(relative_position, relative_velocity))
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        approaching = closing_projection < 0.0
        closing_speed = max(-closing_projection / max(distance, 1e-6), 0.0) if approaching else 0.0
        ttc = float("inf")
        if approaching and relative_speed_sq > max(interaction_speed_threshold, 1e-3) ** 2:
            ttc = max(-closing_projection / (relative_speed_sq + 1e-6), 0.0)

        alignment = float(np.dot(_normalize(relative_position, goal_dir), goal_dir))
        alignment_gate = float(np.clip(0.5 * (alignment + 1.0), 0.0, 1.0))
        proximity_severity = 1.0 - float(
            np.clip(clearance / max(yield_enter_clearance, 1e-6), 0.0, 1.0)
        )
        ttc_severity = (
            0.0
            if not np.isfinite(ttc)
            else 1.0 - float(np.clip(ttc / max(yield_ttc_enter, 1e-6), 0.0, 1.0))
        )
        distance_relevance = 1.0 - float(
            np.clip(clearance / max(relevance_clearance_threshold, 1e-6), 0.0, 1.0)
        )
        closing_relevance = float(
            np.clip(closing_speed / closing_speed_reference, 0.0, 1.0)
            * (0.2 + 0.8 * distance_relevance)
            * alignment_gate
        )
        ttc_relevance = (
            0.0
            if not np.isfinite(ttc)
            else (
                1.0 - float(np.clip(ttc / max(relevance_ttc_threshold, 1e-6), 0.0, 1.0))
            )
            * (0.1 + 0.9 * distance_relevance)
            * alignment_gate
        )
        relevance = max(distance_relevance, closing_relevance, ttc_relevance)
        relevant = (
            clearance < relevance_clearance_threshold
            or closing_speed > 0.0
            or (np.isfinite(ttc) and ttc < relevance_ttc_threshold)
        )
        ttc_strength = ttc_severity * (0.1 + 0.9 * distance_relevance) * alignment_gate
        base_strength = max(proximity_severity, ttc_strength)
        interaction_strength = (
            float(np.clip(base_strength * (1.0 + relevance_gain * interaction_beta * relevance), 0.0, 1.0))
            if relevant
            else 0.0
        )
        if weak_gate and interaction_strength < min_strength:
            interaction_strength = 0.0
        interaction_strengths.append(interaction_strength)
        interaction_distances.append(clearance)
        interaction_ttc.append(ttc)
        interaction_alignments.append(alignment)
        closing_speeds.append(closing_speed)

    sorted_indices = sorted(
        range(len(interaction_strengths)),
        key=lambda index: (interaction_strengths[index], -interaction_distances[index]),
        reverse=True,
    )
    top_index = sorted_indices[0] if sorted_indices else -1
    second_index = sorted_indices[1] if len(sorted_indices) >= 2 else -1
    dominant_index = top_index
    if memory_enabled and dominant_memory_active and 0 <= previous_dominant_index < len(interaction_strengths):
        previous_strength = interaction_strengths[previous_dominant_index]
        top_strength = interaction_strengths[top_index] if top_index >= 0 else 0.0
        if previous_strength >= max(dominant_strength_floor, top_strength - top2_gap_threshold):
            dominant_index = previous_dominant_index

    active_indices: list[int] = []
    if use_topk_filter:
        if dominant_index >= 0 and interaction_strengths[dominant_index] > 0.0:
            active_indices.append(dominant_index)
            competing_index = second_index
            if competing_index == dominant_index:
                competing_index = sorted_indices[2] if len(sorted_indices) >= 3 else -1
            if (
                competing_index >= 0
                and interaction_strengths[competing_index] > 0.0
                and abs(interaction_strengths[dominant_index] - interaction_strengths[competing_index])
                <= top2_gap_threshold
            ):
                active_indices.append(competing_index)
        active_indices = active_indices[:max_active_humans]
    else:
        active_indices = [
            index
            for index in sorted_indices[:max_active_humans]
            if interaction_strengths[index] > 0.0
        ]
        if active_indices:
            dominant_index = active_indices[0]

    def combined_strength(indices: list[int]) -> float:
        if not indices:
            return 0.0
        residual = 1.0
        for index in indices:
            residual *= 1.0 - float(np.clip(interaction_strengths[index], 0.0, 1.0))
        return float(np.clip(1.0 - residual, 0.0, 1.0))

    selected_interaction = combined_strength(active_indices)
    if selected_interaction <= 1e-9 and top_index >= 0:
        selected_interaction = float(np.clip(interaction_strengths[top_index], 0.0, 1.0))
    current_interaction = float(np.clip(selected_interaction, 0.0, 1.0))
    combined_interaction = current_interaction
    if active_indices:
        tracked_index = dominant_index
        tracked_clearance = min(interaction_distances[index] for index in active_indices)
        finite_ttc = [interaction_ttc[index] for index in active_indices if np.isfinite(interaction_ttc[index])]
        tracked_ttc = min(finite_ttc) if finite_ttc else float("inf")
        tracked_score = max(interaction_strengths[index] for index in active_indices)
        tracked_closing_speed = max(closing_speeds[index] for index in active_indices)
    else:
        tracked_closing_speed = 0.0

    selected_clearance_severity = 0.0
    if np.isfinite(tracked_clearance):
        selected_clearance_severity = 1.0 - float(
            np.clip(tracked_clearance / max(yield_enter_clearance, 1e-6), 0.0, 1.0)
        )
    selected_ttc_severity = (
        0.0
        if not np.isfinite(tracked_ttc)
        else 1.0 - float(np.clip(tracked_ttc / max(yield_ttc_enter, 1e-6), 0.0, 1.0))
    )
    raw_interaction_level = current_interaction
    if memory_enabled:
        decay_factor = fast_decay if current_interaction < 0.1 else decay_rate
        decayed_memory = previous_effective_level * decay_factor
        memory_floor = memory_floor_ratio * current_interaction
        memory_target = float(
            np.clip(max(memory_floor, memory_gain * decayed_memory), 0.0, 1.0)
        )
        raw_interaction_level = float(
            np.clip(
                current_blend * current_interaction + memory_blend * memory_target,
                0.0,
                1.0,
            )
        )
    else:
        raw_interaction_level = current_interaction

    if raw_interaction_level >= previous_effective_level:
        rise_alpha = max(smoothing_alpha, 0.5)
        interaction_level = float(
            np.clip(
                rise_alpha * raw_interaction_level
                + (1.0 - rise_alpha) * previous_effective_level,
                0.0,
                effective_cap if memory_enabled else 1.0,
            )
        )
    else:
        fall_alpha = min(smoothing_alpha, 0.2)
        decay_factor = fast_decay if current_interaction < 0.1 else decay_rate
        blended_fall = (
            fall_alpha * raw_interaction_level
            + (1.0 - fall_alpha) * previous_effective_level
        )
        interaction_level = float(
            np.clip(
                max(raw_interaction_level, min(blended_fall, decay_factor * previous_effective_level)),
                0.0,
                effective_cap if memory_enabled else 1.0,
            )
        )
    if current_interaction < 0.05:
        interaction_level = 0.0

    stop_clearance_threshold = (
        stop_exit_clearance
        if previous_human_hard_stop
        else stop_enter_clearance
    )
    stop_ttc_threshold = (
        hard_stop_ttc_exit
        if previous_human_hard_stop
        else hard_stop_ttc_enter
    )
    yield_clearance_threshold = (
        yield_exit_clearance
        if previous_human_active
        else yield_enter_clearance
    )
    yield_ttc_threshold = (
        yield_ttc_exit
        if previous_human_active
        else yield_ttc_enter
    )

    state: FSMState | None = None
    speed_scale = 1.0
    clearance_guard = 1e-6
    hard_stop_floor = max(0.02, 0.25 * min_speed_scale)
    hard_clearance_floor = max(
        0.0,
        stop_enter_clearance - 0.5 * max(stop_exit_clearance - stop_enter_clearance, 1e-3),
    )
    hard_ttc_floor = max(
        0.0,
        hard_stop_ttc_enter - 0.5 * max(hard_stop_ttc_exit - hard_stop_ttc_enter, 0.2),
    )
    if np.isfinite(nearest_clearance) and nearest_clearance <= stop_clearance_threshold + clearance_guard:
        state = FSMState.HARD_STOP
        hard_stop_imminent = nearest_clearance <= hard_clearance_floor + clearance_guard
        if hard_stop_imminent:
            speed_scale = 0.0
        else:
            clearance_scale = float(
                np.clip(
                    (nearest_clearance - hard_clearance_floor)
                    / max(stop_clearance_threshold - hard_clearance_floor, 1e-6),
                    hard_stop_floor,
                    1.0,
                )
            )
            interaction_scale = float(
                np.clip(1.0 - max(combined_interaction, interaction_level), hard_stop_floor, 1.0)
            )
            speed_scale = min(clearance_scale, interaction_scale)
    elif (
        np.isfinite(tracked_ttc)
        and tracked_ttc < stop_ttc_threshold
        and tracked_clearance < max(yield_enter_clearance, stop_enter_clearance + 0.4)
    ):
        state = FSMState.HARD_STOP
        hard_stop_imminent = tracked_ttc <= hard_ttc_floor + clearance_guard
        if np.isfinite(tracked_clearance):
            hard_stop_imminent = hard_stop_imminent or tracked_clearance <= hard_clearance_floor + clearance_guard
        if hard_stop_imminent:
            speed_scale = 0.0
        else:
            clearance_scale = (
                1.0
                if not np.isfinite(tracked_clearance)
                else float(
                    np.clip(
                        tracked_clearance / max(stop_clearance_threshold, 1e-6),
                        hard_stop_floor,
                        1.0,
                    )
                )
            )
            ttc_scale = float(
                np.clip(
                    (tracked_ttc - hard_ttc_floor) / max(stop_ttc_threshold - hard_ttc_floor, 1e-6),
                    hard_stop_floor,
                    1.0,
                )
            )
            interaction_scale = float(
                np.clip(1.0 - max(combined_interaction, interaction_level), hard_stop_floor, 1.0)
            )
            speed_scale = min(clearance_scale, ttc_scale, interaction_scale)
    elif (
        tracked_index >= 0
        and (
            interaction_level >= interaction_threshold
            or (
                persistence_active and memory_enabled
                and not (
                    tracked_clearance > yield_exit_clearance + persistence_clearance_margin
                    or (
                        np.isfinite(tracked_ttc)
                        and tracked_ttc > yield_ttc_exit + persistence_ttc_margin
                    )
                    or current_interaction < 0.1
                )
            )
        )
        and (
            tracked_clearance < relevance_clearance_threshold
            or tracked_closing_speed > 0.0
            or (np.isfinite(tracked_ttc) and tracked_ttc < relevance_ttc_threshold)
            or interaction_level >= interaction_exit_threshold
        )
    ):
        state = FSMState.HUMAN_YIELD
        clearance_scale = float(
            np.clip(
                tracked_clearance / max(yield_clearance_threshold, 1e-6),
                min_speed_scale,
                1.0,
            )
        )
        ttc_scale = (
            1.0
            if not np.isfinite(tracked_ttc)
            else float(
                np.clip(
                    tracked_ttc / max(yield_ttc_threshold, 1e-6),
                    min_speed_scale,
                    1.0,
                )
            )
        )
        interaction_scale = float(
            np.clip(1.0 - max(combined_interaction, interaction_level), min_speed_scale, 1.0)
        )
        speed_scale = min(clearance_scale, ttc_scale, interaction_scale)
    if state not in {FSMState.HUMAN_YIELD, FSMState.HARD_STOP} and interaction_level < interaction_exit_threshold:
        interaction_level = 0.0

    return HumanDecision(
        state=state,
        speed_scale=speed_scale,
        hard_stop_imminent=hard_stop_imminent,
        tracked_index=tracked_index,
        tracked_clearance=tracked_clearance,
        tracked_ttc=tracked_ttc,
        nearest_clearance=nearest_clearance,
        interaction_level=interaction_level,
        combined_interaction=combined_interaction,
        active_indices=active_indices,
        interaction_strengths=interaction_strengths,
        interaction_distances=interaction_distances,
        interaction_ttc=interaction_ttc,
        interaction_alignments=interaction_alignments,
    )


def evaluate_static_escape(
    *,
    profile: StaticRiskProfile,
    goal_dir: np.ndarray,
    current_static_clearance: float,
    release_clearance: float,
    previous_state: str,
    previous_goal_distance: float,
    current_goal_distance: float,
    escape_cooldown_steps_remaining: int,
    escape_progress_window: Sequence[float],
    activation_threshold: float,
    exit_threshold: float,
    risk_slope_threshold: float,
    activation_radius: float,
    min_forward_dot: float,
    goal_regularization: float,
    goal_seek_weight: float,
    escape_direction_memory: np.ndarray,
    escape_commit_steps_remaining: int,
) -> StaticEscapeDecision:
    memory = np.asarray(escape_direction_memory, dtype=float).copy()
    timer = int(max(escape_commit_steps_remaining, 0))
    was_active = previous_state == FSMState.STATIC_ESCAPE.value
    current_static_clearance = float(current_static_clearance)
    release_clearance = float(release_clearance)
    progress = (
        float(previous_goal_distance) - float(current_goal_distance)
        if np.isfinite(float(previous_goal_distance)) and np.isfinite(float(current_goal_distance))
        else 0.0
    )
    avg_progress = (
        float(np.mean(np.asarray(list(escape_progress_window), dtype=float)))
        if escape_progress_window
        else progress
    )
    progress_stalled = avg_progress < 1e-3
    cooldown = int(max(escape_cooldown_steps_remaining, 0))
    if cooldown > 0:
        return StaticEscapeDecision(
            active=False,
            direction=goal_dir.copy(),
            escape_direction_memory=memory,
            escape_commit_steps_remaining=0,
            cooldown_steps=0,
        )
    within_activation_radius = profile.minimum_sampled_distance <= activation_radius
    goal_alignment = float(
        np.dot(
            _normalize(memory if np.linalg.norm(memory) > 1e-9 else goal_dir, goal_dir),
            goal_dir,
        )
    )
    progress_resumed = avg_progress > 1e-3
    clearance_released = current_static_clearance >= release_clearance - 1e-6
    exit_condition = (
        was_active
        and clearance_released
        and (
            progress_resumed
            or goal_alignment > 0.5
            or not within_activation_radius
        )
    )
    if exit_condition:
        return StaticEscapeDecision(
            active=False,
            direction=goal_dir.copy(),
            escape_direction_memory=goal_dir.copy(),
            escape_commit_steps_remaining=0,
            cooldown_steps=8 if was_active else 0,
        )
    stay_active = (
        was_active
        and current_static_clearance < release_clearance - 1e-6
        and (
            profile.current_risk >= exit_threshold
            or profile.future_max_risk >= exit_threshold
            or timer > 0
        )
    )
    activate = (
        cooldown <= 0
        and (
            stay_active
            or (
                current_static_clearance < release_clearance - 1e-6
                and
                within_activation_radius
                and profile.current_risk >= activation_threshold
                and profile.risk_slope >= risk_slope_threshold
            )
        )
    )
    if not activate:
        return StaticEscapeDecision(
            active=False,
            direction=goal_dir.copy(),
            escape_direction_memory=np.zeros(2, dtype=float),
            escape_commit_steps_remaining=0,
            cooldown_steps=0,
        )

    if (
        timer > 0
        and np.linalg.norm(memory) > 1e-9
        and current_static_clearance < release_clearance - 1e-6
        and not exit_condition
    ):
        goal_alignment = float(np.dot(_normalize(memory, goal_dir), goal_dir))
        if exit_condition or (
            goal_alignment > 0.75
            and (
                profile.minimum_sampled_distance > activation_radius
                or avg_progress > 1e-3
            )
        ):
            return StaticEscapeDecision(
                active=False,
                direction=goal_dir.copy(),
                escape_direction_memory=memory,
                escape_commit_steps_remaining=0,
                cooldown_steps=8,
            )
        timer -= 1
        return StaticEscapeDecision(
            active=True,
            direction=memory.copy(),
            escape_direction_memory=memory,
            escape_commit_steps_remaining=timer,
            cooldown_steps=0,
        )

    escape_dir = _normalize(-profile.trigger_gradient, goal_dir)
    lateral_dir = _perpendicular(escape_dir)
    if float(np.dot(lateral_dir, goal_dir)) < 0.0:
        lateral_dir *= -1.0
    if np.linalg.norm(memory) > 1e-9 and float(np.dot(lateral_dir, memory)) < 0.0:
        lateral_dir *= -1.0
    escape_alignment = float(np.dot(escape_dir, goal_dir))
    lateral_gain = 0.18 + 0.32 * max(0.0, -escape_alignment)
    escape_basis = _normalize(
        escape_dir + lateral_gain * lateral_dir,
        escape_dir,
    )
    goal_weight = max(float(goal_seek_weight), 0.0) * goal_regularization
    raw_dir = _normalize(
        escape_basis + goal_weight * goal_dir,
        escape_basis,
    )
    projected_dir = _project_to_goal_cone(raw_dir, goal_dir, min_forward_dot)
    proj_alignment = float(np.dot(projected_dir, goal_dir))
    raw_alignment = float(np.dot(raw_dir, goal_dir))
    forward_blocked = proj_alignment < 0.2 and raw_alignment < 0.2
    escape_mode = forward_blocked
    if escape_mode:
        candidates: list[np.ndarray] = []
        gradient = np.asarray(profile.trigger_gradient, dtype=float)
        if float(np.linalg.norm(gradient)) > 1e-6:
            candidates.append(_normalize(-gradient, goal_dir))
        tangent = _perpendicular(goal_dir)
        candidates.append(_normalize(tangent, goal_dir))
        candidates.append(_normalize(-tangent, goal_dir))
        candidates.append(_normalize(-goal_dir, goal_dir))

        best_dir = goal_dir.copy()
        best_score = (-1e9, -1e9, -1e9)
        for candidate in candidates:
            candidate_direction = _normalize(candidate, goal_dir)
            clearance_gain = -float(np.dot(profile.trigger_gradient, candidate_direction))
            goal_align = float(np.dot(candidate_direction, goal_dir))
            score = (
                clearance_gain,
                0.1 * goal_align,
                1.0,
            )
            if score > best_score:
                best_score = score
                best_dir = candidate_direction
        combined_dir = best_dir
    elif proj_alignment < 0.2 and raw_alignment > -0.5:
        combined_dir = raw_dir
    else:
        combined_dir = projected_dir
    if current_static_clearance >= release_clearance - 1e-6:
        combined_dir = goal_dir.copy()
    if (not escape_mode) and float(np.dot(combined_dir, goal_dir)) < -0.2:
        combined_dir = goal_dir.copy()
    goal_alignment = float(np.dot(combined_dir, goal_dir))
    if was_active and goal_alignment > 0.85 and (
        current_static_clearance >= release_clearance - 1e-6
        or
        profile.minimum_sampled_distance > activation_radius
        or avg_progress > 1e-3
    ):
        return StaticEscapeDecision(
            active=False,
            direction=goal_dir.copy(),
            escape_direction_memory=memory,
            escape_commit_steps_remaining=0,
            cooldown_steps=8,
        )
    memory_alignment = float(np.dot(_normalize(memory, goal_dir), goal_dir))
    memory_stale = (
        avg_progress < 1e-3
        and memory_alignment < 0.7
    )
    if (
        was_active
        and np.linalg.norm(memory) > 1e-9
        and current_static_clearance < release_clearance - 1e-6
        and not memory_stale
    ):
        return StaticEscapeDecision(
            active=True,
            direction=memory.copy(),
            escape_direction_memory=memory.copy(),
            escape_commit_steps_remaining=timer if timer > 0 else 6,
            cooldown_steps=0,
        )

    return StaticEscapeDecision(
        active=True,
        direction=combined_dir.copy(),
        escape_direction_memory=combined_dir.copy(),
        escape_commit_steps_remaining=7,
        cooldown_steps=0,
    )


def resolve_fsm_state(
    *,
    hard_stop_active: bool,
    human_yield_active: bool,
    static_escape_active: bool,
) -> FSMState:
    if hard_stop_active:
        return FSMState.HARD_STOP
    if human_yield_active:
        return FSMState.HUMAN_YIELD
    if static_escape_active:
        return FSMState.STATIC_ESCAPE
    return FSMState.GOAL_SEEK
