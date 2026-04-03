from __future__ import annotations

import argparse
from collections import deque
import csv
import copy
import heapq
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np

try:
    from .controller import (
        FSMState,
        StaticEscapeDecision,
        evaluate_human_speed_control,
        evaluate_static_escape,
        resolve_fsm_state,
    )
    from .metrics import compute_navigation_metrics, current_path_efficiency
    from .risk_field import sample_static_risk_profile
    from .robot import integrate_velocity_command, project_velocity_to_static_safe_set, scale_speed_to_safe_margin
except ImportError:
    from controller import (
        FSMState,
        StaticEscapeDecision,
        evaluate_human_speed_control,
        evaluate_static_escape,
        resolve_fsm_state,
    )
    from metrics import compute_navigation_metrics, current_path_efficiency
    from risk_field import sample_static_risk_profile
    from robot import integrate_velocity_command, project_velocity_to_static_safe_set, scale_speed_to_safe_margin

_PLOTTING_BACKEND = None
_PLOTTING_ERROR: RuntimeError | None = None


def _as_vector(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a 2D vector, got shape {array.shape}")
    return array


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def _validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _clamp_position(position: np.ndarray, world_size: np.ndarray) -> np.ndarray:
    return np.clip(position, [0.0, 0.0], world_size)


def _boundary_clearance(
    position: np.ndarray,
    world_size: np.ndarray,
    radius: float,
) -> float:
    point = _clamp_position(_as_vector(position, "position"), _as_vector(world_size, "world_size"))
    boundary_distance = min(
        float(point[0]),
        float(point[1]),
        float(world_size[0] - point[0]),
        float(world_size[1] - point[1]),
    )
    return float(boundary_distance - max(float(radius), 0.0))


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 1e-9:
        return vector / norm
    if fallback is not None:
        return _normalize(np.asarray(fallback, dtype=float))
    return np.zeros(2, dtype=float)


def _perpendicular(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.zeros(2, dtype=float)
    v = v / n
    return np.array([-v[1], v[0]], dtype=float)


@dataclass
class FailedBranch:
    point: np.ndarray
    guide_direction: np.ndarray
    guide_normal: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    radius: float = 1.4


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _signed_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    if np.linalg.norm(a) <= 1e-9 or np.linalg.norm(b) <= 1e-9:
        return 0.0
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = float(np.arccos(dot))
    cross = float(a[0] * b[1] - a[1] * b[0])
    if cross < 0.0:
        angle *= -1.0
    return angle


def _signed_distance_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    direction = line_end - line_start
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return 0.0
    offset = point - line_start
    return float((direction[0] * offset[1] - direction[1] * offset[0]) / length)


def _rotate_toward(
    previous_direction: np.ndarray,
    target_direction: np.ndarray,
    max_turn_angle: float,
    *,
    damping: float = 1.0,
) -> np.ndarray:
    previous = _normalize(previous_direction, target_direction)
    target = _normalize(target_direction, previous)
    turn_angle = _signed_angle_between(previous, target)
    if abs(turn_angle) <= 1e-9:
        return target.copy()
    limited_angle = float(
        np.clip(turn_angle, -abs(float(max_turn_angle)), abs(float(max_turn_angle)))
    )
    limited_angle *= float(np.clip(damping, 0.0, 1.0))
    if abs(limited_angle) < 0.01:
        return previous.copy()
    if abs(limited_angle) >= abs(turn_angle) - 1e-9:
        return target.copy()
    angle = limited_angle
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    return np.array(
        [
            cos_angle * previous[0] - sin_angle * previous[1],
            sin_angle * previous[0] + cos_angle * previous[1],
        ],
        dtype=float,
    )


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


def _compute_recovery_escape_direction(
    goal_direction: np.ndarray,
    risk_gradient: np.ndarray,
    previous_direction: np.ndarray,
    min_alignment: float = 0.05,
) -> np.ndarray:
    goal = _normalize(goal_direction, previous_direction)
    escape = _normalize(-np.asarray(risk_gradient, dtype=float), goal)
    lateral = np.array([-escape[1], escape[0]], dtype=float)
    if float(np.dot(lateral, previous_direction)) < 0.0:
        lateral *= -1.0
    if float(np.dot(lateral, goal)) < 0.0:
        lateral *= -1.0
    candidate = _normalize(0.7 * goal + 0.5 * escape + 0.25 * lateral, escape)
    return _project_to_goal_cone(candidate, goal, min_alignment)


def _compute_boundary_tangent_direction(
    goal_direction: np.ndarray,
    boundary_normal: np.ndarray,
    previous_direction: np.ndarray,
    fallback_direction: np.ndarray,
    min_alignment: float = 0.05,
) -> np.ndarray:
    goal = _normalize(goal_direction, fallback_direction)
    normal = _normalize(boundary_normal, fallback_direction)
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    if float(np.dot(tangent, goal)) < 0.0:
        tangent *= -1.0
    if abs(float(np.dot(tangent, goal))) <= 1e-6 and float(np.dot(tangent, previous_direction)) < 0.0:
        tangent *= -1.0
    candidate = _normalize(0.7 * goal + 0.3 * tangent, tangent)
    return _project_to_goal_cone(candidate, goal, min_alignment)


def _enforce_forward_progress(
    velocity: np.ndarray,
    goal_direction: np.ndarray,
    min_progress: float,
) -> np.ndarray:
    current_velocity = _as_vector(velocity, "velocity")
    goal = _normalize(goal_direction, current_velocity)
    speed = float(np.linalg.norm(current_velocity))
    if speed <= 1e-9:
        return current_velocity.copy()

    required_progress = float(np.clip(min_progress, 0.0, speed))
    current_progress = float(np.dot(current_velocity, goal))
    if current_progress >= required_progress - 1e-9:
        return current_velocity.copy()

    lateral = current_velocity - current_progress * goal
    lateral_norm = float(np.linalg.norm(lateral))
    max_lateral = float(np.sqrt(max(speed * speed - required_progress * required_progress, 0.0)))
    if lateral_norm > 1e-9 and max_lateral > 0.0:
        lateral = lateral * (max_lateral / lateral_norm)
    else:
        lateral = np.zeros(2, dtype=float)
    corrected_velocity = required_progress * goal + lateral
    corrected_speed = float(np.linalg.norm(corrected_velocity))
    if corrected_speed > speed > 1e-9:
        corrected_velocity *= speed / corrected_speed
    return corrected_velocity


def _project_to_forward_half_plane(
    velocity: np.ndarray,
    goal_direction: np.ndarray,
) -> np.ndarray:
    projected_velocity = _as_vector(velocity, "velocity").copy()
    forward = _normalize(goal_direction, projected_velocity)
    alignment = float(np.dot(projected_velocity, forward))
    if alignment < 0.0:
        lateral = projected_velocity - alignment * forward
        if np.linalg.norm(lateral) > 1e-6:
            projected_velocity = lateral
    return projected_velocity


def _static_clearance_with_boundary(
    position: np.ndarray,
    world_size: np.ndarray,
    obstacles: Sequence["Obstacle"],
    zones: Sequence["NoGoZone"],
) -> float:
    point = _clamp_position(_as_vector(position, "position"), world_size)
    clearance = min(
        float(point[0]),
        float(point[1]),
        float(world_size[0] - point[0]),
        float(world_size[1] - point[1]),
    )
    for obstacle in obstacles:
        clearance = min(clearance, float(obstacle.distance_to_surface(point)))
    for zone in zones:
        clearance = min(clearance, float(zone.signed_distance(point)))
    return float(clearance)


def _coerce_failed_branch_entries(
    failed_branches: Sequence[FailedBranch] | None,
    *,
    min_radius: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for index, branch in enumerate(failed_branches or []):
        branch_point = _as_vector(branch.point, f"failed_branch[{index}].point")
        branch_direction = _normalize(
            _as_vector(branch.guide_direction, f"failed_branch[{index}].guide_direction")
        )
        if np.linalg.norm(branch_direction) <= 1e-9:
            continue
        fallback_normal = _perpendicular(branch_direction)
        branch_normal = _normalize(
            _as_vector(branch.guide_normal, f"failed_branch[{index}].guide_normal"),
            fallback_normal,
        )
        if np.linalg.norm(branch_normal) <= 1e-9:
            continue
        branch_radius = max(float(branch.radius), float(min_radius))
        entries.append(
            (
                branch_point,
                branch_direction,
                branch_normal,
                branch_radius,
            )
        )
    return entries


def _inside_failed_branch_region(
    point: np.ndarray,
    *,
    branch_point: np.ndarray,
    branch_direction: np.ndarray,
    branch_normal: np.ndarray,
    branch_radius: float,
) -> bool:
    offset = point - branch_point
    d_parallel = float(np.dot(offset, branch_direction))
    if d_parallel <= 0.0:
        return False
    d_perp = abs(float(np.dot(offset, branch_normal)))
    return d_perp < branch_radius


def _polyline_arc_progress(
    position: np.ndarray,
    path_points: Sequence[np.ndarray],
    *,
    fallback_direction: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, float, np.ndarray]:
    points = [_as_vector(point, "path_point") for point in path_points]
    if len(points) < 2:
        fallback = _normalize(fallback_direction)
        return 0.0, 0.0, fallback, 0.0, points[0].copy() if points else np.zeros(2, dtype=float)

    best_distance = float("inf")
    best_progress = 0.0
    best_tangent = _normalize(points[1] - points[0], fallback_direction)
    best_projected_point = points[0].copy()
    total_length = 0.0
    cumulative = 0.0

    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            continue
        tangent = segment / segment_length
        projection = float(np.dot(position - start, tangent))
        projection = float(np.clip(projection, 0.0, segment_length))
        projected_point = start + projection * tangent
        distance = float(np.linalg.norm(position - projected_point))
        progress = cumulative + projection
        if distance < best_distance - 1e-9 or (
            abs(distance - best_distance) <= 1e-9 and progress > best_progress
        ):
            best_distance = distance
            best_progress = progress
            best_tangent = tangent.copy()
            best_projected_point = projected_point.copy()
        cumulative += segment_length
        total_length += segment_length

    return (
        best_progress,
        total_length,
        _normalize(best_tangent, fallback_direction),
        best_distance,
        best_projected_point,
    )


def _segment_static_clear(
    start: np.ndarray,
    end: np.ndarray,
    *,
    world_size: np.ndarray,
    obstacles: Sequence["Obstacle"],
    zones: Sequence["NoGoZone"],
    clearance_threshold: float,
    sample_spacing: float,
    failed_branches: Sequence[FailedBranch] | None = None,
) -> bool:
    start_point = _as_vector(start, "start")
    delta = _as_vector(end, "end") - start_point
    distance = float(np.linalg.norm(delta))
    failed_branch_entries = _coerce_failed_branch_entries(
        failed_branches,
        min_radius=max(sample_spacing, 1e-3),
    )
    if distance <= 1e-9:
        if _static_clearance_with_boundary(start_point, world_size, obstacles, zones) < clearance_threshold:
            return False
        return True
    segment_direction = _normalize(delta)
    steps = max(int(np.ceil(distance / max(sample_spacing, 1e-3))), 1)
    for step in range(steps + 1):
        alpha = float(step) / float(steps)
        point = start_point + alpha * delta
        if _static_clearance_with_boundary(point, world_size, obstacles, zones) < clearance_threshold:
            return False
        for branch_point, branch_direction, branch_normal, branch_radius in failed_branch_entries:
            if float(np.dot(segment_direction, branch_direction)) <= 0.6:
                continue
            if _inside_failed_branch_region(
                point,
                branch_point=branch_point,
                branch_direction=branch_direction,
                branch_normal=branch_normal,
                branch_radius=branch_radius,
            ):
                return False
    return True


def _build_static_guide_waypoints(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    world_size: np.ndarray,
    obstacles: Sequence["Obstacle"],
    zones: Sequence["NoGoZone"],
    clearance_threshold: float,
    grid_resolution: float = 0.2,
    failed_branches: Sequence[FailedBranch] | None = None,
) -> list[np.ndarray]:
    failed_branches = list(failed_branches or [])
    start = _clamp_position(_as_vector(start, "start"), world_size)
    goal = _clamp_position(_as_vector(goal, "goal"), world_size)
    if _segment_static_clear(
        start,
        goal,
        world_size=world_size,
        obstacles=obstacles,
        zones=zones,
        clearance_threshold=clearance_threshold,
        sample_spacing=0.5 * grid_resolution,
        failed_branches=failed_branches,
    ):
        return [goal.copy()]

    resolution = max(float(grid_resolution), 0.1)
    nx = max(int(np.ceil(world_size[0] / resolution)) + 1, 2)
    ny = max(int(np.ceil(world_size[1] / resolution)) + 1, 2)

    def _cell_point(ix: int, iy: int) -> np.ndarray:
        return np.array(
            [
                min(ix * resolution, float(world_size[0])),
                min(iy * resolution, float(world_size[1])),
            ],
            dtype=float,
        )

    failed_branch_entries = _coerce_failed_branch_entries(
        failed_branches,
        min_radius=0.5 * resolution,
    )

    def _branch_transition_blocked(candidate_point: np.ndarray, step_direction: np.ndarray) -> bool:
        if not failed_branch_entries:
            return False
        normalized_step = _normalize(step_direction)
        if np.linalg.norm(normalized_step) <= 1e-9:
            return False
        for branch_point, branch_direction, branch_normal, branch_radius in failed_branch_entries:
            if float(np.dot(normalized_step, branch_direction)) <= 0.6:
                continue
            if _inside_failed_branch_region(
                candidate_point,
                branch_point=branch_point,
                branch_direction=branch_direction,
                branch_normal=branch_normal,
                branch_radius=branch_radius,
            ):
                return True
        return False

    free = np.zeros((nx, ny), dtype=bool)
    free_points: list[tuple[int, int, np.ndarray]] = []
    for ix in range(nx):
        for iy in range(ny):
            point = _cell_point(ix, iy)
            if _static_clearance_with_boundary(point, world_size, obstacles, zones) >= clearance_threshold:
                free[ix, iy] = True
                free_points.append((ix, iy, point))

    if not free_points:
        return [goal.copy()]

    def _nearest_free_cell(target: np.ndarray) -> tuple[int, int]:
        return min(
            ((ix, iy) for ix, iy, _ in free_points),
            key=lambda index: float(np.linalg.norm(_cell_point(index[0], index[1]) - target)),
        )

    start_cell = _nearest_free_cell(start)
    goal_cell = _nearest_free_cell(goal)

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    cost_so_far: dict[tuple[int, int], float] = {start_cell: 0.0}
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_cell:
            break
        for dx, dy in neighbors:
            nx_i = current[0] + dx
            ny_i = current[1] + dy
            if nx_i < 0 or ny_i < 0 or nx_i >= nx or ny_i >= ny or not free[nx_i, ny_i]:
                continue
            current_point = _cell_point(current[0], current[1])
            neighbor_point = _cell_point(nx_i, ny_i)
            if _branch_transition_blocked(neighbor_point, neighbor_point - current_point):
                continue
            step_cost = float(np.hypot(dx, dy))
            new_cost = cost_so_far[current] + step_cost
            neighbor = (nx_i, ny_i)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor] - 1e-9:
                cost_so_far[neighbor] = new_cost
                heuristic = float(np.hypot(goal_cell[0] - nx_i, goal_cell[1] - ny_i))
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current

    if goal_cell not in came_from:
        if failed_branches:
            relaxed_failed_branches = failed_branches[1:]
            return _build_static_guide_waypoints(
                start=start,
                goal=goal,
                world_size=world_size,
                obstacles=obstacles,
                zones=zones,
                clearance_threshold=clearance_threshold,
                grid_resolution=resolution,
                failed_branches=relaxed_failed_branches,
            )
        return [goal.copy()]

    cell_path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal_cell
    while cursor is not None:
        cell_path.append(cursor)
        cursor = came_from[cursor]
    cell_path.reverse()

    path_points = [start.copy()]
    for cell in cell_path[1:-1]:
        path_points.append(_cell_point(*cell))
    path_points.append(goal.copy())

    simplified = [path_points[0]]
    anchor = 0
    while anchor < len(path_points) - 1:
        next_index = anchor + 1
        for candidate in range(len(path_points) - 1, anchor, -1):
            if _segment_static_clear(
                path_points[anchor],
                path_points[candidate],
                world_size=world_size,
                obstacles=obstacles,
                zones=zones,
                clearance_threshold=clearance_threshold,
                sample_spacing=0.5 * resolution,
                failed_branches=failed_branches,
            ):
                next_index = candidate
                break
        simplified.append(path_points[next_index])
        anchor = next_index

    return [point.copy() for point in simplified[1:]]


def _select_committed_detour_direction(
    *,
    position: np.ndarray,
    guide_direction: np.ndarray,
    static_gradient: np.ndarray,
    previous_direction: np.ndarray,
    escape_gain: float = 1.0,
    target_speed: float,
    min_progress_speed: float,
    dt: float,
    safety_distance: float,
    clearance_fn: Callable[[np.ndarray], float],
    dynamic_clearance_fn: Callable[[np.ndarray], float] | None = None,
    dynamic_safety_distance: float | None = None,
    failed_detours: Sequence[np.ndarray] | None = None,
    lookahead_risk_fn: Callable[[np.ndarray], float] | None = None,
    lookahead_step: float | None = None,
    progress_value: float = 0.0,
    progress_epsilon: float = 1e-3,
    progress_timer: float = 0.0,
    progress_timer_threshold: float = 1.0,
    detour_side: int = 0,
) -> tuple[np.ndarray | None, float]:
    base_direction = _normalize(guide_direction, guide_direction)
    previous = _normalize(previous_direction, base_direction)
    gradient = np.asarray(static_gradient, dtype=float)
    escape_gain = float(np.clip(escape_gain, 1.0, 2.0))
    candidates: list[np.ndarray] = []
    if float(np.linalg.norm(gradient)) > 1e-5:
        candidates.append(_normalize(-gradient, base_direction))
    tangent = _perpendicular(base_direction)
    candidates.append(_normalize(tangent, base_direction))
    candidates.append(_normalize(-tangent, base_direction))
    candidates.append(_normalize(-base_direction, base_direction))
    failed_directions = [
        _normalize(np.asarray(direction, dtype=float), base_direction)
        for direction in (failed_detours or [])
        if np.linalg.norm(np.asarray(direction, dtype=float)) > 1e-9
    ]
    current_clearance = float(clearance_fn(position))
    step_distance = max(float(lookahead_step) if lookahead_step is not None else target_speed * dt, 1e-6)
    soft_escape_boost = (
        float(progress_value) < float(progress_epsilon)
        and float(progress_timer) > float(progress_timer_threshold)
    )
    candidate_pool: list[tuple[np.ndarray, float, tuple[float, float, float]]] = []
    side = int(np.sign(detour_side))
    side_bias_dir = (
        _normalize(float(side) * tangent, base_direction)
        if side != 0 and np.linalg.norm(tangent) > 1e-9
        else np.zeros(2, dtype=float)
    )

    for candidate_direction in candidates:
        candidate_direction = _normalize(candidate_direction, base_direction)
        goal_alignment = float(np.dot(candidate_direction, base_direction))
        if goal_alignment < -0.85:
            continue
        side_alignment = (
            float(np.dot(candidate_direction, side_bias_dir))
            if side != 0 and np.linalg.norm(side_bias_dir) > 1e-9
            else 0.0
        )
        if side != 0 and side_alignment < -0.2:
            continue

        certified_velocity, _, _ = _certify_forward_progress_velocity(
            position=position,
            direction=candidate_direction,
            target_speed=target_speed,
            min_progress_speed=min_progress_speed,
            dt=dt,
            safety_distance=safety_distance,
            clearance_fn=clearance_fn,
            dynamic_clearance_fn=dynamic_clearance_fn,
            dynamic_safety_distance=dynamic_safety_distance,
        )
        speed = float(np.linalg.norm(certified_velocity))
        if speed <= 1e-9:
            continue
        next_position = position + step_distance * candidate_direction
        next_clearance = float(clearance_fn(next_position))
        if next_clearance < safety_distance + 1e-9:
            continue
        second_position = next_position + step_distance * candidate_direction
        second_clearance = float(clearance_fn(second_position))
        if second_clearance < safety_distance + 1e-9:
            continue
        clearance_gain = next_clearance - current_clearance
        risk1 = float(lookahead_risk_fn(next_position)) if lookahead_risk_fn is not None else 0.0
        risk2 = float(lookahead_risk_fn(second_position)) if lookahead_risk_fn is not None else 0.0
        max_future_risk = max(risk1, risk2)
        score_boost = 0.2 * clearance_gain if soft_escape_boost else 0.0
        score = (
            -max_future_risk + score_boost,
            0.15 * side_alignment + 0.1 * goal_alignment,
            0.05 * clearance_gain,
        )
        candidate_pool.append((candidate_direction.copy(), clearance_gain, score))

    if not candidate_pool:
        return None, 0.0

    def _is_bad(candidate_direction: np.ndarray) -> bool:
        return any(
            float(np.dot(candidate_direction, failed_direction)) > 0.9
            for failed_direction in failed_directions
        )

    filtered_candidates = [entry for entry in candidate_pool if not _is_bad(entry[0])]

    if filtered_candidates:
        best_direction, best_clearance_gain, _ = max(filtered_candidates, key=lambda entry: entry[2])
        return best_direction.copy(), float(best_clearance_gain)

    if failed_directions:
        best_direction, best_clearance_gain, _ = min(
            candidate_pool,
            key=lambda entry: max(
                float(np.dot(entry[0], failed_direction))
                for failed_direction in failed_directions
            ),
        )
        return best_direction.copy(), float(best_clearance_gain)

    best_direction, best_clearance_gain, _ = max(candidate_pool, key=lambda entry: entry[2])
    return best_direction.copy(), float(best_clearance_gain)


def _select_static_recovery_velocity(
    *,
    position: np.ndarray,
    goal_direction: np.ndarray,
    boundary_normal: np.ndarray,
    target_speed: float,
    dt: float,
    world_size: np.ndarray,
    safety_distance: float,
    current_clearance: float,
    clearance_fn: Callable[[np.ndarray], float],
    boundary_normal_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, float, float]:
    goal = _normalize(goal_direction, boundary_normal)
    normal = _normalize(boundary_normal, goal)
    left_tangent = np.array([-normal[1], normal[0]], dtype=float)
    right_tangent = -left_tangent
    candidates = (
        normal,
        left_tangent,
        right_tangent,
        -goal,
    )

    best_velocity = np.zeros(2, dtype=float)
    best_margin = current_clearance - safety_distance
    best_gain = 0.0
    best_score: tuple[float, float, float] | None = None

    for candidate in candidates:
        candidate_direction = _normalize(candidate, normal)
        candidate_projection = project_velocity_to_static_safe_set(
            position=position,
            velocity=candidate_direction * target_speed,
            dt=dt,
            world_size=world_size,
            safety_distance=safety_distance,
            clearance_fn=clearance_fn,
            boundary_normal_fn=boundary_normal_fn,
        )
        candidate_velocity = candidate_projection.velocity
        speed = float(np.linalg.norm(candidate_velocity))
        min_speed = 0.05
        if speed < min_speed:
            direction = _normalize(candidate_velocity, candidate_direction)
            candidate_velocity = direction * min_speed
            speed = min_speed
        if speed < 1e-6:
            continue

        next_clearance = float(clearance_fn(position + candidate_velocity * dt))
        if next_clearance < safety_distance + 1e-6:
            continue

        clearance_gain = next_clearance - current_clearance
        goal_alignment = float(np.dot(_normalize(candidate_velocity, candidate_direction), goal))
        score = (
            clearance_gain,
            0.1 * goal_alignment,
            speed,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_velocity = candidate_velocity.copy()
            best_margin = next_clearance - safety_distance
            best_gain = clearance_gain

    return best_velocity, best_margin, best_gain


def _select_progress_guarantee_velocity(
    *,
    position: np.ndarray,
    goal: np.ndarray,
    goal_direction: np.ndarray,
    boundary_normal: np.ndarray,
    target_speed: float,
    dt: float,
    world_size: np.ndarray,
    safety_distance: float,
    current_clearance: float,
    preferred_clearance: float,
    clearance_fn: Callable[[np.ndarray], float],
    boundary_normal_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dynamic_clearance_fn: Callable[[np.ndarray], float] | None = None,
    dynamic_safety_distance: float | None = None,
) -> tuple[np.ndarray, float, float]:
    goal_dir = _normalize(goal_direction, boundary_normal)
    normal = _normalize(boundary_normal, goal_dir)

    def _rotate(vector: np.ndarray, angle_rad: float) -> np.ndarray:
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        return np.array(
            [
                c * vector[0] - s * vector[1],
                s * vector[0] + c * vector[1],
            ],
            dtype=float,
        )

    candidate_directions = [
        goal_dir,
        _normalize(goal_dir + 0.35 * normal, goal_dir),
        _normalize(goal_dir - 0.35 * normal, goal_dir),
    ]
    for angle_deg in (20.0, -20.0, 40.0, -40.0, 60.0, -60.0):
        candidate_directions.append(_normalize(_rotate(goal_dir, float(np.deg2rad(angle_deg))), goal_dir))

    current_goal_distance = float(np.linalg.norm(goal - position))
    best_velocity = np.zeros(2, dtype=float)
    best_margin = current_clearance - safety_distance
    best_progress = 0.0
    evaluated: list[tuple[np.ndarray, float, float, float, float]] = []

    for candidate in candidate_directions:
        candidate_direction = _normalize(candidate, goal_dir)
        candidate_velocity = candidate_direction * target_speed
        if dynamic_clearance_fn is not None and dynamic_safety_distance is not None:
            dynamic_result = scale_speed_to_safe_margin(
                position=position,
                direction=candidate_direction,
                target_speed=target_speed,
                dt=dt,
                safety_distance=dynamic_safety_distance,
                clearance_fn=dynamic_clearance_fn,
            )
            candidate_velocity = dynamic_result.velocity
        if np.linalg.norm(candidate_velocity) <= 1e-9:
            continue
        candidate_projection = project_velocity_to_static_safe_set(
            position=position,
            velocity=candidate_velocity,
            dt=dt,
            world_size=world_size,
            safety_distance=safety_distance,
            clearance_fn=clearance_fn,
            boundary_normal_fn=boundary_normal_fn,
        )
        candidate_velocity = candidate_projection.velocity
        speed = float(np.linalg.norm(candidate_velocity))
        min_speed = 0.05
        if speed < min_speed:
            direction = _normalize(candidate_velocity, candidate_direction)
            candidate_velocity = direction * min_speed
            speed = min_speed
        if speed < 1e-6:
            continue

        next_position = position + candidate_velocity * dt
        next_clearance = float(clearance_fn(next_position))
        if next_clearance < safety_distance + 1e-6:
            continue
        if dynamic_clearance_fn is not None and dynamic_safety_distance is not None:
            next_dynamic_clearance = float(dynamic_clearance_fn(next_position))
            if next_dynamic_clearance < dynamic_safety_distance + 1e-6:
                continue

        next_goal_distance = float(np.linalg.norm(goal - next_position))
        goal_progress = current_goal_distance - next_goal_distance
        resulting_direction = _normalize(candidate_velocity, candidate_direction)
        alignment = float(np.dot(resulting_direction, goal_dir))
        clearance_gain = next_clearance - current_clearance
        evaluated.append(
            (
                candidate_velocity.copy(),
                next_clearance - safety_distance,
                goal_progress,
                alignment,
                clearance_gain,
            )
            )

    for min_alignment in (0.35, 0.2, 0.1, 0.01):
        best_score: tuple[float, float, float, float] | None = None
        for candidate_velocity, candidate_margin, goal_progress, alignment, clearance_gain in evaluated:
            if alignment < min_alignment:
                continue
            score = (
                float(goal_progress > 1e-4),
                goal_progress,
                alignment,
                clearance_gain if candidate_margin + safety_distance >= preferred_clearance - 1e-6 else candidate_margin,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_velocity = candidate_velocity.copy()
                best_margin = candidate_margin
                best_progress = goal_progress
        if best_score is not None:
            return best_velocity, best_margin, best_progress

    return best_velocity, best_margin, best_progress


def _scale_speed_to_dynamic_horizon_margin(
    *,
    position: np.ndarray,
    direction: np.ndarray,
    target_speed: float,
    dt: float,
    humans: Sequence["Human"],
    robot_radius: float,
    safety_distance: float,
    horizon_steps: int = 6,
    max_iterations: int = 14,
) -> tuple[np.ndarray, float]:
    direction = _normalize(direction)
    target_speed = max(float(target_speed), 0.0)
    if target_speed <= 1e-9 or np.linalg.norm(direction) <= 1e-9:
        return np.zeros(2, dtype=float), float("inf")
    if not humans:
        return direction * target_speed, float("inf")

    horizon = max(int(horizon_steps), 1)

    def _margin_for_speed(speed: float) -> tuple[bool, float]:
        min_margin = float("inf")
        for step in range(1, horizon + 1):
            step_time = float(step) * dt
            robot_position = position + direction * speed * step_time
            for human in humans:
                human_position = np.asarray(human.position, dtype=float) + step_time * np.asarray(human.velocity, dtype=float)
                clearance = float(np.linalg.norm(robot_position - human_position)) - (robot_radius + float(human.radius))
                min_margin = min(min_margin, clearance - safety_distance)
                if clearance < safety_distance - 1e-9:
                    return False, min_margin
        return True, min_margin

    full_safe, full_margin = _margin_for_speed(target_speed)
    if full_safe:
        return direction * target_speed, full_margin

    low = 0.0
    high = target_speed
    best_margin = float("-inf")
    for _ in range(max(int(max_iterations), 1)):
        mid = 0.5 * (low + high)
        safe, margin = _margin_for_speed(mid)
        if safe:
            low = mid
            best_margin = margin
        else:
            high = mid

    final_speed = low
    final_safe, final_margin = _margin_for_speed(final_speed)
    if not final_safe or final_speed <= 1e-9:
        return np.zeros(2, dtype=float), min(best_margin, final_margin)
    return direction * final_speed, final_margin


def _certify_forward_progress_velocity(
    *,
    position: np.ndarray,
    direction: np.ndarray,
    target_speed: float,
    min_progress_speed: float,
    dt: float,
    safety_distance: float,
    clearance_fn: Callable[[np.ndarray], float],
    dynamic_clearance_fn: Callable[[np.ndarray], float] | None = None,
    dynamic_safety_distance: float | None = None,
) -> tuple[np.ndarray, float, bool]:
    forward = _normalize(direction)
    requested_speed = max(float(target_speed), 0.0)
    progress_floor = max(float(min_progress_speed), 0.0)
    if requested_speed <= 1e-9 or np.linalg.norm(forward) <= 1e-9:
        return np.zeros(2, dtype=float), float("-inf"), False

    def _scaled_forward(speed: float) -> tuple[np.ndarray, float]:
        if speed <= 1e-9:
            return np.zeros(2, dtype=float), float("-inf")
        effective_speed = float(speed)
        dynamic_margin = float("inf")
        if dynamic_clearance_fn is not None and dynamic_safety_distance is not None:
            dynamic_projection = scale_speed_to_safe_margin(
                position=position,
                direction=forward,
                target_speed=effective_speed,
                dt=dt,
                safety_distance=dynamic_safety_distance,
                clearance_fn=dynamic_clearance_fn,
            )
            effective_speed = float(np.linalg.norm(dynamic_projection.velocity))
            dynamic_margin = dynamic_projection.clearance - safety_distance
        static_projection = scale_speed_to_safe_margin(
            position=position,
            direction=forward,
            target_speed=effective_speed,
            dt=dt,
            safety_distance=safety_distance,
            clearance_fn=clearance_fn,
        )
        return static_projection.velocity.copy(), min(
            static_projection.clearance - safety_distance,
            dynamic_margin,
        )

    certified_velocity, certified_margin = _scaled_forward(requested_speed)
    floor_feasible = False
    if progress_floor > 1e-9:
        floor_velocity, floor_margin = _scaled_forward(progress_floor)
        floor_feasible = np.linalg.norm(floor_velocity) >= progress_floor - 1e-6
        if floor_feasible and np.linalg.norm(certified_velocity) < progress_floor - 1e-6:
            certified_velocity = floor_velocity
            certified_margin = floor_margin
    return certified_velocity, certified_margin, floor_feasible


def _enforce_nonnegative_path_tangent_velocity(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    path_tangent: np.ndarray,
    dt: float,
    world_size: np.ndarray,
    safety_distance: float,
    clearance_fn: Callable[[np.ndarray], float],
    boundary_normal_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dynamic_clearance_fn: Callable[[np.ndarray], float] | None = None,
    dynamic_safety_distance: float | None = None,
    backward_tolerance: float = 1e-4,
) -> np.ndarray:
    candidate_velocity = _as_vector(velocity, "velocity").copy()
    tangent = _normalize(path_tangent, candidate_velocity)
    if np.linalg.norm(candidate_velocity) <= 1e-9 or np.linalg.norm(tangent) <= 1e-9:
        return candidate_velocity

    tangent_component = float(np.dot(candidate_velocity, tangent))
    if tangent_component >= -backward_tolerance:
        return candidate_velocity

    corrected_velocity = candidate_velocity - tangent_component * tangent
    corrected_speed = float(np.linalg.norm(corrected_velocity))
    if corrected_speed <= 1e-9:
        return np.zeros(2, dtype=float)

    corrected_direction = _normalize(corrected_velocity, tangent)
    if dynamic_clearance_fn is not None and dynamic_safety_distance is not None:
        dynamic_projection = scale_speed_to_safe_margin(
            position=position,
            direction=corrected_direction,
            target_speed=corrected_speed,
            dt=dt,
            safety_distance=dynamic_safety_distance,
            clearance_fn=dynamic_clearance_fn,
        )
        corrected_velocity = dynamic_projection.velocity.copy()
        corrected_speed = float(np.linalg.norm(corrected_velocity))
        if corrected_speed <= 1e-9:
            return np.zeros(2, dtype=float)
        corrected_direction = _normalize(corrected_velocity, tangent)

    static_projection = project_velocity_to_static_safe_set(
        position=position,
        velocity=corrected_direction * corrected_speed,
        dt=dt,
        world_size=world_size,
        safety_distance=safety_distance,
        clearance_fn=clearance_fn,
        boundary_normal_fn=boundary_normal_fn,
    )
    corrected_velocity = static_projection.velocity.copy()
    corrected_tangent_component = float(np.dot(corrected_velocity, tangent))
    if corrected_tangent_component < -backward_tolerance:
        corrected_velocity = corrected_velocity - corrected_tangent_component * tangent

    candidate_position = _clamp_position(position + corrected_velocity * dt, world_size)
    if float(clearance_fn(candidate_position)) < safety_distance - 1e-6:
        return np.zeros(2, dtype=float)
    if (
        dynamic_clearance_fn is not None
        and dynamic_safety_distance is not None
        and float(dynamic_clearance_fn(candidate_position)) < dynamic_safety_distance - 1e-6
    ):
        return np.zeros(2, dtype=float)
    return corrected_velocity


def _box_boundary_projection(
    position: np.ndarray,
    center: np.ndarray,
    half_size: np.ndarray,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    local = position - center
    clamped = np.clip(local, -half_size, half_size)
    if np.all(np.abs(local) <= half_size + 1e-9):
        face_clearance = half_size - np.abs(local)
        if face_clearance[0] <= face_clearance[1]:
            sign = 1.0 if local[0] >= 0.0 else -1.0
            if abs(local[0]) <= 1e-9 and fallback is not None and abs(float(fallback[0])) > 1e-9:
                sign = float(np.sign(fallback[0]))
            if sign == 0.0:
                sign = 1.0
            boundary_local = np.array(
                [sign * half_size[0], np.clip(local[1], -half_size[1], half_size[1])],
                dtype=float,
            )
            normal = np.array([sign, 0.0], dtype=float)
        else:
            sign = 1.0 if local[1] >= 0.0 else -1.0
            if abs(local[1]) <= 1e-9 and fallback is not None and abs(float(fallback[1])) > 1e-9:
                sign = float(np.sign(fallback[1]))
            if sign == 0.0:
                sign = 1.0
            boundary_local = np.array(
                [np.clip(local[0], -half_size[0], half_size[0]), sign * half_size[1]],
                dtype=float,
            )
            normal = np.array([0.0, sign], dtype=float)
        return center + boundary_local, normal

    boundary = center + clamped
    normal = _normalize(position - boundary, fallback)
    return boundary, normal


def _ensure_writable_matplotlib_config() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return

    config_dir = Path(__file__).resolve().parents[1] / ".mplconfig"
    config_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _format_float_sequence(values: Sequence[float]) -> str:
    if not values:
        return ""
    return ";".join("--" if not np.isfinite(value) else f"{float(value):0.3f}" for value in values)


def _parse_float_sequence(value: str) -> list[float]:
    if not value:
        return []
    parsed: list[float] = []
    for item in value.split(";"):
        item = item.strip()
        if not item or item == "--":
            parsed.append(float("inf"))
        else:
            parsed.append(float(item))
    return parsed


class NavLogger:
    def __init__(
        self,
        *,
        summary_interval: int = 0,
        detailed_threshold: float = 0.12,
        console_output: bool = True,
        safety_distance: float = 0.5,
    ) -> None:
        self.summary_interval = max(int(summary_interval), 0)
        self.detailed_threshold = _validate_nonnegative(detailed_threshold, "logger.detailed_threshold")
        self.console_output = bool(console_output)
        self.safety_distance = max(_validate_positive(safety_distance, "logger.safety_distance"), 0.5)
        self.path_start: np.ndarray | None = None
        self.path_goal: np.ndarray | None = None
        self.records: list[dict[str, float | int | str | bool]] = []
        self._detail_active = False
        self._recovery_times: list[float] = []
        self._latest_recovery_time = 0.0
        self._recovery_pending_since: float | None = None
        self._recovery_stable_frames = 0
        self._interaction_active = False

    def reset(self) -> None:
        self.records.clear()
        self._detail_active = False
        self._recovery_times = []
        self._latest_recovery_time = 0.0
        self._recovery_pending_since = None
        self._recovery_stable_frames = 0
        self._interaction_active = False

    def _update_recovery_metrics(self, record: dict[str, float | int | str | bool]) -> None:
        human_stop_reasons = {"human_safety", "escape_blocked", "goal_hold"}
        state = str(record.get("behavior_state", record.get("state", "")))
        stop_reason = str(record.get("stop_reason", ""))
        interaction_active = state == "HUMAN_YIELD" or (
            state == "HARD_STOP" and stop_reason in human_stop_reasons
        )
        stable_speed_scale = max(0.2, 0.3 * 0.9)
        stable_speed = 0.05
        stable_forward_motion = state in {"GOAL_SEEK", "STATIC_ESCAPE"} and (
            float(record.get("speed_scale", 0.0)) >= stable_speed_scale
            or float(record.get("speed", 0.0)) >= stable_speed
        )
        current_time = float(record["time"])

        if interaction_active:
            self._interaction_active = True
            self._recovery_pending_since = None
            self._recovery_stable_frames = 0
        elif self._interaction_active:
            self._interaction_active = False
            self._recovery_pending_since = current_time
            self._recovery_stable_frames = 0

        if self._recovery_pending_since is not None:
            if interaction_active:
                self._recovery_pending_since = None
                self._recovery_stable_frames = 0
            elif state == "goal":
                self._latest_recovery_time = max(current_time - self._recovery_pending_since, 0.0)
                self._recovery_times.append(self._latest_recovery_time)
                self._recovery_pending_since = None
                self._recovery_stable_frames = 0
            elif stable_forward_motion:
                self._recovery_stable_frames += 1
                if self._recovery_stable_frames >= 3:
                    self._latest_recovery_time = max(current_time - self._recovery_pending_since, 0.0)
                    self._recovery_times.append(self._latest_recovery_time)
                    self._recovery_pending_since = None
                    self._recovery_stable_frames = 0
            else:
                self._recovery_stable_frames = 0

        mean_recovery_time = float(np.mean(self._recovery_times)) if self._recovery_times else 0.0
        record["latest_recovery_time"] = float(self._latest_recovery_time)
        record["mean_recovery_time"] = float(mean_recovery_time)
        record["recovery_time"] = float(mean_recovery_time)

    def log(self, t: float, robot: Robot | None, metrics: dict[str, float | int | str | bool]) -> None:
        record = {
            "step": int(metrics["step"]),
            "time": float(t),
            "x": float(metrics["x"]),
            "y": float(metrics["y"]),
            "vx": float(metrics["vx"]),
            "vy": float(metrics["vy"]),
            "speed": float(metrics["speed"]),
            "speed_scale": float(metrics["speed_scale"]),
            "state": str(metrics["state"]),
            "behavior_state": str(metrics["behavior_state"]),
            "ttc": float(metrics["ttc"]),
            "clearance": float(metrics["clearance"]),
            "clr_interaction": float(metrics["clr_interaction"]),
            "clr_global": float(metrics["clr_global"]),
            "global_clearance": float(metrics["global_clearance"]),
            "clr_rate": float(metrics["clr_rate"]),
            "interacting_human_id": int(metrics["interacting_human_id"]),
            "primary_human_id": int(metrics["primary_human_id"]),
            "interaction_current": float(metrics["interaction_current"]),
            "interaction_memory": float(metrics["interaction_memory"]),
            "interaction_effective": float(metrics["interaction_effective"]),
            "interaction_level": float(metrics["interaction_level"]),
            "top_1_interaction_strength": float(metrics["top_1_interaction_strength"]),
            "top_2_interaction_strength": float(metrics["top_2_interaction_strength"]),
            "interaction_strength_gap": float(metrics["interaction_strength_gap"]),
            "multi_dominant_interaction": bool(metrics["multi_dominant_interaction"]),
            "num_active_interactions": int(metrics["num_active_interactions"]),
            "number_of_active_humans": int(metrics["number_of_active_humans"]),
            "active_humans": int(metrics["active_humans"]),
            "min_distance_to_humans": float(metrics["min_distance_to_humans"]),
            "min_clearance": float(metrics["min_clearance"]),
            "global_min_clearance": float(metrics["global_min_clearance"]),
            "interaction_switch_count": int(metrics["interaction_switch_count"]),
            "interaction_switch_timestamps": str(metrics["interaction_switch_timestamps"]),
            "per_human_min_clearances": str(metrics["per_human_min_clearances"]),
            "human_interaction_distances": str(metrics["human_interaction_distances"]),
            "human_interaction_ttc": str(metrics["human_interaction_ttc"]),
            "human_interaction_alignments": str(metrics["human_interaction_alignments"]),
            "human_interaction_scores": str(metrics["human_interaction_scores"]),
            "lateral_deviation": float(metrics["lateral_deviation"]),
            "heading_change": float(metrics["heading_change"]),
            "raw_curvature": float(metrics["raw_curvature"]),
            "curvature": float(metrics["curvature"]),
            "stop_reason": str(metrics["stop_reason"]),
            "safety_margin": float(metrics["safety_margin"]),
            "risk_slope": float(metrics["risk_slope"]),
            "path_efficiency": float(metrics["path_efficiency"]),
            "recovery_time": float(metrics["recovery_time"]),
            "invariant_recovery_active": bool(metrics["invariant_recovery_active"]),
            "invariant_recovery_count": int(metrics["invariant_recovery_count"]),
            "stagnation_active": bool(metrics["stagnation_active"]),
            "avg_goal_progress": float(metrics["avg_goal_progress"]),
            "global_progress": float(metrics["global_progress"]),
            "guide_progress_max": float(metrics["guide_progress_max"]),
            "failed_branch_count": int(metrics["failed_branch_count"]),
            "goal_distance": float(metrics["goal_distance"]),
            "detail_active": bool(metrics["detail_active"]),
        }
        self._update_recovery_metrics(record)
        self.records.append(record)

        if not self.console_output:
            return

        step_index = len(self.records) - 1
        if self.summary_interval > 0 and step_index % self.summary_interval == 0:
            print(self._format_summary(record))

        detail_active = bool(record["detail_active"])
        if detail_active and not self._detail_active:
            print(self._format_detail(record, prefix="detail_on"))
        elif not detail_active and self._detail_active:
            print(self._format_detail(record, prefix="detail_off"))
        self._detail_active = detail_active

    def summary(self, step_interval: int | None = None) -> None:
        if not self.records:
            return
        interval = self.summary_interval if step_interval is None else max(int(step_interval), 1)
        for index in range(0, len(self.records), interval):
            print(self._format_summary(self.records[index]))

    def save(self, path: str = "logs.csv") -> None:
        if not self.records:
            return
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.records[0].keys())
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def validation_metrics(self) -> dict[str, float]:
        if not self.records:
            return {}
        metrics = compute_navigation_metrics(
            self.records,
            safety_distance=self.safety_distance,
            path_start=self.path_start,
            path_goal=self.path_goal,
        )

        finite_human_clearances = [
            float(record["min_distance_to_humans"])
            for record in self.records
            if np.isfinite(float(record["min_distance_to_humans"]))
        ]
        min_human_clearance = (
            float(np.min(finite_human_clearances)) if finite_human_clearances else float("inf")
        )
        finite_interaction_clearances = [
            float(record["clr_interaction"])
            for record in self.records
            if np.isfinite(float(record["clr_interaction"]))
        ]
        min_interaction_clearance = (
            float(np.min(finite_interaction_clearances))
            if finite_interaction_clearances
            else float("inf")
        )
        per_human_min_clearances = _parse_float_sequence(
            str(self.records[-1]["per_human_min_clearances"])
        )
        finite_per_human_clearances = [
            clearance for clearance in per_human_min_clearances if np.isfinite(clearance)
        ]
        metrics.update(
            {
                "minimum_clearance": float(
                    np.min(
                        [
                            float(record.get("raw_min_clearance", record.get("min_clearance", float("inf"))))
                            for record in self.records
                            if np.isfinite(
                                float(record.get("raw_min_clearance", record.get("min_clearance", float("inf"))))
                            )
                        ]
                    )
                )
                if any(
                    np.isfinite(
                        float(record.get("raw_min_clearance", record.get("min_clearance", float("inf"))))
                    )
                    for record in self.records
                )
                else float("inf"),
                "min_human_clearance": min_human_clearance,
                "min_interaction_clearance": min_interaction_clearance,
                "min_per_human_clearance": float(np.min(finite_per_human_clearances))
                if finite_per_human_clearances
                else float("inf"),
                "mean_per_human_clearance": float(np.mean(finite_per_human_clearances))
                if finite_per_human_clearances
                else float("inf"),
                "mean_abs_heading_change": float(
                    np.mean([abs(float(record["heading_change"])) for record in self.records])
                ),
            }
        )
        metrics["min_physical_clearance"] = metrics["minimum_clearance"]
        metrics["minimum_clearance_raw"] = metrics["minimum_clearance"]
        return metrics

    def _format_summary(self, record: dict[str, float | int | str | bool]) -> str:
        ttc = float(record["ttc"])
        interaction_clearance = float(record["clr_interaction"])
        global_clearance = float(record["clr_global"])
        ttc_text = "--" if not np.isfinite(ttc) else f"{ttc:0.2f}"
        interaction_clearance_text = (
            "--" if not np.isfinite(interaction_clearance) else f"{interaction_clearance:0.2f}"
        )
        global_clearance_text = (
            "--" if not np.isfinite(global_clearance) else f"{global_clearance:0.2f}"
        )
        human_id = int(record["interacting_human_id"])
        human_text = "--" if human_id < 0 else str(human_id)
        return (
            f"t={float(record['time']):0.1f} | spd={float(record['speed_scale']):0.2f} | "
            f"int={float(record['interaction_effective']):0.2f} | "
            f"sm={float(record['safety_margin']):0.2f} | "
            f"rs={float(record['risk_slope']):0.2f} | "
            f"mem={float(record['interaction_memory']):0.2f} | "
            f"clr_i={interaction_clearance_text} | clr_g={global_clearance_text} | "
            f"hid={human_text} | act={int(record['number_of_active_humans'])} | "
            f"sw={int(record['interaction_switch_count'])} | ttc={ttc_text} | state={record['state']}"
        )

    def _format_detail(
        self, record: dict[str, float | int | str | bool], *, prefix: str
    ) -> str:
        human_id = int(record["interacting_human_id"])
        human_text = "--" if human_id < 0 else str(human_id)
        return (
            f"{prefix}: t={float(record['time']):0.1f} | "
            f"active={int(record['num_active_interactions'])} | "
            f"hid={human_text} | "
            f"sw={int(record['interaction_switch_count'])} | "
            f"int_cur={float(record['interaction_current']):0.3f} | "
            f"int_eff={float(record['interaction_effective']):0.3f} | "
            f"top2={float(record['top_1_interaction_strength']):0.3f}/"
            f"{float(record['top_2_interaction_strength']):0.3f} | "
            f"gap={float(record['interaction_strength_gap']):0.3f} | "
            f"clr_i={float(record['clr_interaction']):0.3f} | "
            f"clr_g={float(record['clr_global']):0.3f} | "
            f"dclr={float(record['clr_rate']):0.3f} | "
            f"sm={float(record['safety_margin']):0.3f} | "
            f"rs={float(record['risk_slope']):0.3f} | "
            f"dev={float(record['lateral_deviation']):0.3f} | "
            f"dtheta={float(record['heading_change']):0.3f}"
        )


@dataclass
class Human:
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray | None = None
    waypoints: tuple[np.ndarray, ...] | None = None
    loop_waypoints: bool = True
    radius: float = 0.18
    preferred_speed: float | None = None
    noise_std: float = 0.05
    velocity_smoothing: float = 0.8
    goal_tolerance: float = 0.35
    max_acceleration: float = 0.5
    preferred_speed_min: float = 0.6
    preferred_speed_max: float = 1.4
    rng_seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)
    _waypoint_index: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "human.position")
        self.velocity = _as_vector(self.velocity, "human.velocity")
        self._rng = np.random.default_rng(self.rng_seed)
        if self.goal is not None:
            self.goal = _as_vector(self.goal, "human.goal")
        if self.waypoints is not None:
            waypoints = tuple(_as_vector(waypoint, "human.waypoints") for waypoint in self.waypoints)
            if not waypoints:
                raise ValueError("human.waypoints must contain at least one waypoint")
            self.waypoints = waypoints
            if self.goal is None:
                self.goal = waypoints[0].copy()
                self._waypoint_index = 0
            else:
                self._waypoint_index = int(
                    np.argmin([np.linalg.norm(self.goal - waypoint) for waypoint in waypoints])
                )
                self.goal = waypoints[self._waypoint_index].copy()
        self.radius = _validate_positive(self.radius, "human.radius")
        self.preferred_speed_min = _validate_positive(
            self.preferred_speed_min, "human.preferred_speed_min"
        )
        self.preferred_speed_max = _validate_positive(
            self.preferred_speed_max, "human.preferred_speed_max"
        )
        if self.preferred_speed_max < self.preferred_speed_min:
            raise ValueError(
                "human.preferred_speed_max must be greater than or equal to human.preferred_speed_min"
            )
        if (
            self.preferred_speed is None
            or float(self.preferred_speed) < self.preferred_speed_min
            or float(self.preferred_speed) > self.preferred_speed_max
        ):
            self.preferred_speed = float(
                self._rng.uniform(self.preferred_speed_min, self.preferred_speed_max)
            )
        self.preferred_speed = _validate_positive(self.preferred_speed, "human.preferred_speed")
        self.noise_std = _validate_nonnegative(self.noise_std, "human.noise_std")
        self.velocity_smoothing = _validate_unit_interval(
            self.velocity_smoothing, "human.velocity_smoothing"
        )
        self.goal_tolerance = _validate_positive(self.goal_tolerance, "human.goal_tolerance")
        self.max_acceleration = _validate_positive(self.max_acceleration, "human.max_acceleration")
        initial_direction = _normalize(
            self.velocity,
            (self.goal - self.position) if self.goal is not None else np.array([1.0, 0.0], dtype=float),
        )
        if np.linalg.norm(initial_direction) > 1e-9:
            self.velocity = self.preferred_speed * initial_direction

    def get_rng_state(self) -> dict:
        return copy.deepcopy(self._rng.bit_generator.state)

    def set_rng_state(self, state: dict) -> None:
        self._rng.bit_generator.state = copy.deepcopy(state)

    def _sample_goal(self, world_size: np.ndarray) -> np.ndarray:
        margin = 0.12 * float(np.min(world_size))
        lower = np.array([margin, margin], dtype=float)
        upper = np.maximum(world_size - margin, lower + 1e-3)
        min_distance = max(self.goal_tolerance * 2.0, 0.25 * float(np.min(world_size)))
        for _ in range(12):
            candidate = self._rng.uniform(lower, upper)
            if np.linalg.norm(candidate - self.position) >= min_distance:
                return candidate
        return self._rng.uniform(lower, upper)

    def _advance_goal(self, world_size: np.ndarray) -> None:
        if self.waypoints:
            next_index = self._waypoint_index + 1
            if self.loop_waypoints:
                next_index %= len(self.waypoints)
            else:
                next_index = min(next_index, len(self.waypoints) - 1)
            self._waypoint_index = int(next_index)
            self.goal = self.waypoints[self._waypoint_index].copy()
            return
        self.goal = self._sample_goal(world_size)

    def update(
        self,
        dt: float,
        world_size: np.ndarray,
        *,
        blocking_position: np.ndarray | None = None,
        blocking_radius: float = 0.0,
        blocking_clearance: float = 0.0,
    ) -> None:
        dt = _validate_positive(dt, "dt")
        if self.goal is not None:
            goal_vector = self.goal - self.position
            if np.linalg.norm(goal_vector) <= self.goal_tolerance:
                self._advance_goal(world_size)
                goal_vector = self.goal - self.position

            desired_dir = _normalize(goal_vector, _normalize(self.velocity, goal_vector))
            noise_angle = float(self._rng.normal(0.0, self.noise_std))
            cos_angle = float(np.cos(noise_angle))
            sin_angle = float(np.sin(noise_angle))
            noisy_dir = np.array(
                [
                    cos_angle * desired_dir[0] - sin_angle * desired_dir[1],
                    sin_angle * desired_dir[0] + cos_angle * desired_dir[1],
                ],
                dtype=float,
            )
            desired_velocity = self.preferred_speed * _normalize(noisy_dir, desired_dir)
            delta_velocity = desired_velocity - self.velocity
            max_delta_speed = self.max_acceleration * dt
            delta_speed = float(np.linalg.norm(delta_velocity))
            if delta_speed > max_delta_speed > 0.0:
                delta_velocity *= max_delta_speed / delta_speed
            acceleration_limited_velocity = self.velocity + delta_velocity
            self.velocity = self.velocity_smoothing * self.velocity + (
                1.0 - self.velocity_smoothing
            ) * acceleration_limited_velocity
            speed = float(np.linalg.norm(self.velocity))
            if speed > self.preferred_speed > 1e-9:
                self.velocity *= self.preferred_speed / speed

        if blocking_position is not None:
            blocking_center = _as_vector(blocking_position, "blocking_position")
            blocking_radius = _validate_nonnegative(blocking_radius, "blocking_radius")
            blocking_clearance = _validate_nonnegative(blocking_clearance, "blocking_clearance")
            speed = float(np.linalg.norm(self.velocity))
            if speed > 1e-9:
                direction = _normalize(self.velocity)
                blocking_projection = scale_speed_to_safe_margin(
                    position=self.position,
                    direction=direction,
                    target_speed=speed,
                    dt=dt,
                    safety_distance=blocking_clearance,
                    clearance_fn=lambda candidate_position: np.linalg.norm(
                        candidate_position - blocking_center
                    )
                    - (self.radius + blocking_radius),
                )
                if (
                    self.goal is not None
                    and blocking_projection.speed_scale < 0.25
                    and np.linalg.norm(self.goal - self.position) > self.goal_tolerance
                ):
                    goal_dir = _normalize(self.goal - self.position, direction)
                    block_dir = _normalize(blocking_center - self.position, goal_dir)
                    if float(np.dot(goal_dir, block_dir)) > 0.1:
                        left_tangent = np.array([-block_dir[1], block_dir[0]], dtype=float)
                        right_tangent = -left_tangent
                        tangent_candidates = [left_tangent, right_tangent]
                        tangent_projections = [
                            scale_speed_to_safe_margin(
                                position=self.position,
                                direction=tangent,
                                target_speed=self.preferred_speed,
                                dt=dt,
                                safety_distance=blocking_clearance,
                                clearance_fn=lambda candidate_position: np.linalg.norm(
                                    candidate_position - blocking_center
                                )
                                - (self.radius + blocking_radius),
                            )
                            for tangent in tangent_candidates
                        ]
                        best_index = max(
                            range(len(tangent_candidates)),
                            key=lambda index: (
                                np.linalg.norm(tangent_projections[index].velocity),
                                float(np.dot(tangent_candidates[index], goal_dir)),
                            ),
                        )
                        best_projection = tangent_projections[best_index]
                        if np.linalg.norm(best_projection.velocity) > 1e-6:
                            self.velocity = best_projection.velocity.copy()
                        else:
                            self.velocity = blocking_projection.velocity.copy()
                    else:
                        self.velocity = blocking_projection.velocity.copy()
                else:
                    self.velocity = blocking_projection.velocity.copy()

        next_position = self.position + self.velocity * dt

        for axis in range(2):
            if next_position[axis] < 0.0:
                next_position[axis] = -next_position[axis]
                self.velocity[axis] *= -1.0
            elif next_position[axis] > world_size[axis]:
                next_position[axis] = 2.0 * world_size[axis] - next_position[axis]
                self.velocity[axis] *= -1.0

        if blocking_position is not None:
            minimum_distance = blocking_radius + self.radius + blocking_clearance
            offset = next_position - blocking_center
            distance = float(np.linalg.norm(offset))
            if distance < minimum_distance - 1e-6:
                fallback_direction = _normalize(
                    self.position - blocking_center,
                    self.velocity if np.linalg.norm(self.velocity) > 1e-9 else np.array([1.0, 0.0], dtype=float),
                )
                outward = _normalize(offset, fallback_direction)
                next_position = blocking_center + outward * minimum_distance
                corrected_velocity = (next_position - self.position) / dt
                corrected_speed = float(np.linalg.norm(corrected_velocity))
                if corrected_speed > self.preferred_speed > 1e-9:
                    corrected_velocity *= self.preferred_speed / corrected_speed
                self.velocity = corrected_velocity

        self.position = _clamp_position(next_position, world_size)


@dataclass
class Obstacle:
    kind: str
    center: np.ndarray
    size: np.ndarray | float

    def __post_init__(self) -> None:
        self.center = _as_vector(self.center, "obstacle.center")
        if self.kind == "circle":
            self.size = _validate_positive(float(self.size), "obstacle.size")
        elif self.kind == "rectangle":
            size = _as_vector(self.size, "obstacle.size")
            if np.any(size <= 0.0):
                raise ValueError("rectangle obstacle size must be positive in both dimensions")
            self.size = size
        else:
            raise ValueError(f"unsupported obstacle kind: {self.kind}")

    def distance_to_surface(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        if self.kind == "circle":
            return max(np.linalg.norm(position - self.center) - float(self.size), 0.0)

        half_size = np.asarray(self.size, dtype=float) / 2.0
        delta = np.abs(position - self.center) - half_size
        outside = np.maximum(delta, 0.0)
        return float(np.linalg.norm(outside))

    def distance_grid(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        if self.kind == "circle":
            radius = float(self.size)
            distance = np.hypot(grid_x - self.center[0], grid_y - self.center[1]) - radius
            return np.maximum(distance, 0.0)

        half_size = np.asarray(self.size, dtype=float) / 2.0
        dx = np.abs(grid_x - self.center[0]) - half_size[0]
        dy = np.abs(grid_y - self.center[1]) - half_size[1]
        return np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))

    def closest_point(self, position: np.ndarray) -> np.ndarray:
        position = _as_vector(position, "position")
        if self.kind == "circle":
            delta = position - self.center
            distance = np.linalg.norm(delta)
            if distance < 1e-9:
                return self.center + np.array([float(self.size), 0.0], dtype=float)
            return self.center + (delta / distance) * float(self.size)

        half_size = np.asarray(self.size, dtype=float) / 2.0
        lower = self.center - half_size
        upper = self.center + half_size
        return np.clip(position, lower, upper)

    def surface_projection(
        self,
        position: np.ndarray,
        fallback: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = _as_vector(position, "position")
        if self.kind == "circle":
            normal = _normalize(position - self.center, fallback)
            boundary = self.center + normal * float(self.size)
            return boundary, normal

        half_size = np.asarray(self.size, dtype=float) / 2.0
        return _box_boundary_projection(position, self.center, half_size, fallback)

    def project_point_outside(
        self,
        position: np.ndarray,
        clearance: float,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        clearance = _validate_nonnegative(clearance, "clearance")
        boundary, normal = self.surface_projection(position, fallback)
        return boundary + normal * clearance

    def create_patch(self) -> patches.Patch:
        import matplotlib.patches as patches

        if self.kind == "circle":
            return patches.Circle(self.center, radius=float(self.size), color="0.55", alpha=0.95)

        size = np.asarray(self.size, dtype=float)
        lower_left = self.center - size / 2.0
        return patches.Rectangle(lower_left, size[0], size[1], color="0.55", alpha=0.95)


@dataclass
class NoGoZone:
    kind: str
    center: np.ndarray
    size: np.ndarray | float

    def __post_init__(self) -> None:
        self.center = _as_vector(self.center, "zone.center")
        if self.kind == "circle":
            self.size = _validate_positive(float(self.size), "zone.size")
        elif self.kind == "rectangle":
            size = _as_vector(self.size, "zone.size")
            if np.any(size <= 0.0):
                raise ValueError("rectangle zone size must be positive in both dimensions")
            self.size = size
        else:
            raise ValueError(f"unsupported zone kind: {self.kind}")

    def signed_distance(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        if self.kind == "circle":
            return float(np.linalg.norm(position - self.center) - float(self.size))

        half_size = np.asarray(self.size, dtype=float) / 2.0
        delta = position - self.center
        q = np.abs(delta) - half_size
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(q[0], q[1]), 0.0)
        return float(outside + inside)

    def signed_distance_grid(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        if self.kind == "circle":
            return np.hypot(grid_x - self.center[0], grid_y - self.center[1]) - float(self.size)

        half_size = np.asarray(self.size, dtype=float) / 2.0
        dx = grid_x - self.center[0]
        dy = grid_y - self.center[1]
        qx = np.abs(dx) - half_size[0]
        qy = np.abs(dy) - half_size[1]
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside

    def boundary_projection(
        self,
        position: np.ndarray,
        fallback: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = _as_vector(position, "position")
        if self.kind == "circle":
            normal = _normalize(position - self.center, fallback)
            boundary = self.center + normal * float(self.size)
            return boundary, normal

        half_size = np.asarray(self.size, dtype=float) / 2.0
        return _box_boundary_projection(position, self.center, half_size, fallback)

    def project_point_outside(
        self,
        position: np.ndarray,
        clearance: float,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        clearance = _validate_nonnegative(clearance, "clearance")
        boundary, normal = self.boundary_projection(position, fallback)
        return boundary + normal * clearance

    def create_patch(self):
        import matplotlib.patches as patches

        if self.kind == "circle":
            return patches.Circle(
                self.center,
                radius=float(self.size),
                facecolor="#6f1230",
                edgecolor="#3f0819",
                alpha=0.22,
                linewidth=1.4,
                linestyle="--",
            )

        size = np.asarray(self.size, dtype=float)
        lower_left = self.center - size / 2.0
        return patches.Rectangle(
            lower_left,
            size[0],
            size[1],
            facecolor="#6f1230",
            edgecolor="#3f0819",
            alpha=0.22,
            linewidth=1.4,
            linestyle="--",
        )


@dataclass
class RiskField:
    humans: Sequence[Human]
    obstacles: Sequence[Obstacle]
    zones: Sequence[NoGoZone]
    goal: np.ndarray
    world_size: np.ndarray
    alpha: float = 2.2
    w_h: float = 5.0
    w_social: float = 2.2
    w_o: float = 8.0
    w_zone: float = 18.0
    w_g: float = 1.5
    sigma_parallel: float = 2.8
    sigma_perp: float = 1.3
    social_sigma_perp: float = 1.9
    sigma_obs: float = 1.3
    sigma_zone: float = 0.8
    sigma_goal: float = 2.0
    zone_inside_gain: float = 6.0
    epsilon: float = 1e-3
    stationary_sigma: float | None = None
    social_front_scale: float = 1.35
    social_rear_scale: float = 0.6
    normalize_gradient: bool = True
    gradient_clip: float = 1.0
    prediction_horizon: float = 3.0
    prediction_dt: float = 0.3
    lambda_decay: float = 1.5
    max_prediction_distance: float = 5.0
    sigma_parallel_growth: float = 0.2
    interaction_tau: float = 1.0
    multi_human_aggregation: bool = True

    def __post_init__(self) -> None:
        self.humans = list(self.humans)
        self.obstacles = list(self.obstacles)
        self.zones = list(self.zones)
        self.goal = _as_vector(self.goal, "risk_field.goal")
        self.world_size = _as_vector(self.world_size, "risk_field.world_size")
        if np.any(self.world_size <= 0.0):
            raise ValueError("risk_field.world_size must be positive in both dimensions")

        self.alpha = _validate_positive(self.alpha, "risk_field.alpha")
        self.w_h = _validate_nonnegative(self.w_h, "risk_field.w_h")
        self.w_social = _validate_nonnegative(self.w_social, "risk_field.w_social")
        self.w_o = _validate_nonnegative(self.w_o, "risk_field.w_o")
        self.w_zone = _validate_nonnegative(self.w_zone, "risk_field.w_zone")
        self.w_g = _validate_nonnegative(self.w_g, "risk_field.w_g")
        self.sigma_parallel = _validate_positive(
            self.sigma_parallel, "risk_field.sigma_parallel"
        )
        self.sigma_perp = _validate_positive(self.sigma_perp, "risk_field.sigma_perp")
        self.social_sigma_perp = _validate_positive(
            self.social_sigma_perp, "risk_field.social_sigma_perp"
        )
        self.sigma_obs = _validate_positive(self.sigma_obs, "risk_field.sigma_obs")
        self.sigma_zone = _validate_positive(self.sigma_zone, "risk_field.sigma_zone")
        self.sigma_goal = _validate_positive(self.sigma_goal, "risk_field.sigma_goal")
        self.zone_inside_gain = _validate_nonnegative(
            self.zone_inside_gain, "risk_field.zone_inside_gain"
        )
        self.epsilon = _validate_positive(self.epsilon, "risk_field.epsilon")
        if self.stationary_sigma is None:
            self.stationary_sigma = 0.5 * (self.sigma_parallel + self.sigma_perp)
        self.stationary_sigma = _validate_positive(
            self.stationary_sigma, "risk_field.stationary_sigma"
        )
        self.gradient_clip = _validate_positive(
            self.gradient_clip, "risk_field.gradient_clip"
        )
        self.prediction_horizon = _validate_positive(
            self.prediction_horizon, "risk_field.prediction_horizon"
        )
        self.prediction_dt = _validate_positive(self.prediction_dt, "risk_field.prediction_dt")
        self.lambda_decay = _validate_nonnegative(self.lambda_decay, "risk_field.lambda_decay")
        self.max_prediction_distance = _validate_positive(
            self.max_prediction_distance, "risk_field.max_prediction_distance"
        )
        self.sigma_parallel_growth = _validate_nonnegative(
            self.sigma_parallel_growth, "risk_field.sigma_parallel_growth"
        )
        self.social_front_scale = _validate_positive(
            self.social_front_scale, "risk_field.social_front_scale"
        )
        self.social_rear_scale = _validate_positive(
            self.social_rear_scale, "risk_field.social_rear_scale"
        )
        self.interaction_tau = _validate_positive(
            self.interaction_tau, "risk_field.interaction_tau"
        )
        self.multi_human_aggregation = bool(self.multi_human_aggregation)

        self._sigma_parallel_sq = self.sigma_parallel**2
        self._sigma_perp_sq = self.sigma_perp**2
        self._social_sigma_perp_sq = self.social_sigma_perp**2
        self._sigma_obs_sq = self.sigma_obs**2
        self._sigma_zone_sq = self.sigma_zone**2
        self._sigma_goal_sq = self.sigma_goal**2
        self._stationary_sigma_sq = self.stationary_sigma**2
        self._social_stationary_sigma_sq = max(self.stationary_sigma, self.social_sigma_perp) ** 2
        self._interaction_tau_denom = 2.0 * (self.interaction_tau**2)
        prediction_steps = int(np.floor(self.prediction_horizon / self.prediction_dt))
        self._prediction_times = np.arange(prediction_steps + 1, dtype=float) * self.prediction_dt
        self._prediction_weights = np.exp(-self.lambda_decay * self._prediction_times)
        self._prediction_weights /= np.sum(self._prediction_weights)
        self._prediction_distance_sq = self.max_prediction_distance**2
        self._moving_parallel_sigma_sq = (
            self.sigma_parallel + self.sigma_parallel_growth * self._prediction_times
        ) ** 2
        self._social_front_parallel_sigma_sq = (
            self.social_front_scale
            * (self.sigma_parallel + self.sigma_parallel_growth * self._prediction_times)
        ) ** 2
        social_rear_sigma = max(self.stationary_sigma * self.social_rear_scale, 1e-3)
        self._social_rear_parallel_sigma_sq = np.full_like(
            self._prediction_times, social_rear_sigma**2, dtype=float
        )
        self._prediction_cache: list[dict[str, np.ndarray | bool]] = []
        self._prediction_cache_valid = False
        self._robot_reference_position = self.goal.copy()
        self._robot_reference_speed = 1.0

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = _clamp_position(_as_vector(goal, "goal"), self.world_size)

    def set_robot_state(self, position: np.ndarray, speed: float) -> None:
        self._robot_reference_position = _clamp_position(
            _as_vector(position, "robot_position"), self.world_size
        )
        self._robot_reference_speed = max(float(speed), 1e-6)

    def invalidate_cache(self) -> None:
        self._prediction_cache_valid = False

    def get_predicted_trajectories(self) -> list[np.ndarray]:
        self._ensure_prediction_cache()
        return [entry["positions"].copy() for entry in self._prediction_cache]

    def compute_risk(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        risk = self.compute_hazard_risk(position)
        if self.w_g > 0.0:
            risk += self._goal_risk(position)

        return float(risk)

    def compute_hazard_risk(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        risk = self._predictive_human_risk(position)
        for obstacle in self.obstacles:
            risk += self._obstacle_risk(position, obstacle)
        for zone in self.zones:
            risk += self._zone_risk(position, zone)
        return float(risk)

    def compute_dynamic_risk(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        return float(self._predictive_human_risk(position))

    def compute_static_risk(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        risk = 0.0
        for obstacle in self.obstacles:
            risk += self._obstacle_risk(position, obstacle)
        for zone in self.zones:
            risk += self._zone_risk(position, zone)
        return float(risk)

    def nearest_static_distance(self, position: np.ndarray) -> float:
        return float(
            min(
                self.nearest_obstacle_distance(position),
                self.nearest_zone_distance(position),
            )
        )

    def compute_static_gradient(self, position: np.ndarray) -> np.ndarray:
        position = _clamp_position(_as_vector(position, "position"), self.world_size)
        epsilon = self.epsilon

        x_plus = _clamp_position(position + np.array([epsilon, 0.0]), self.world_size)
        x_minus = _clamp_position(position - np.array([epsilon, 0.0]), self.world_size)
        y_plus = _clamp_position(position + np.array([0.0, epsilon]), self.world_size)
        y_minus = _clamp_position(position - np.array([0.0, epsilon]), self.world_size)

        dx = x_plus[0] - x_minus[0]
        dy = y_plus[1] - y_minus[1]
        grad_x = 0.0 if dx < 1e-12 else (self.compute_static_risk(x_plus) - self.compute_static_risk(x_minus)) / dx
        grad_y = 0.0 if dy < 1e-12 else (self.compute_static_risk(y_plus) - self.compute_static_risk(y_minus)) / dy

        gradient = np.array([grad_x, grad_y], dtype=float)
        norm = np.linalg.norm(gradient)
        if norm > self.gradient_clip:
            gradient *= self.gradient_clip / norm
        return gradient

    def hazard_breakdown(self, position: np.ndarray) -> dict[str, list[float]]:
        position = _as_vector(position, "position")
        self._ensure_prediction_cache()
        robot_time = np.linalg.norm(position - self._robot_reference_position) / self._robot_reference_speed
        interaction = np.exp(
            -((self._prediction_times - robot_time) ** 2) / self._interaction_tau_denom
        )
        return {
            "humans": [
                self._single_predictive_human_risk(position, entry, interaction)
                for entry in self._prediction_cache
            ],
            "obstacles": [self._obstacle_risk(position, obstacle) for obstacle in self.obstacles],
            "zones": [self._zone_risk(position, zone) for zone in self.zones],
        }

    def nearest_human_distance(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        if not self.humans:
            return float("inf")
        return float(
            min(max(np.linalg.norm(position - human.position) - human.radius, 0.0) for human in self.humans)
        )

    def nearest_obstacle_distance(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        if not self.obstacles:
            return float("inf")
        return float(min(obstacle.distance_to_surface(position) for obstacle in self.obstacles))

    def nearest_zone_distance(self, position: np.ndarray) -> float:
        position = _as_vector(position, "position")
        if not self.zones:
            return float("inf")
        return float(min(max(zone.signed_distance(position), 0.0) for zone in self.zones))

    def nearest_hazard_distance(self, position: np.ndarray) -> float:
        return float(
            min(
                self.nearest_human_distance(position),
                self.nearest_obstacle_distance(position),
                self.nearest_zone_distance(position),
            )
        )

    def compute_gradient(self, position: np.ndarray) -> np.ndarray:
        position = _clamp_position(_as_vector(position, "position"), self.world_size)
        epsilon = self.epsilon

        x_plus = _clamp_position(position + np.array([epsilon, 0.0]), self.world_size)
        x_minus = _clamp_position(position - np.array([epsilon, 0.0]), self.world_size)
        y_plus = _clamp_position(position + np.array([0.0, epsilon]), self.world_size)
        y_minus = _clamp_position(position - np.array([0.0, epsilon]), self.world_size)

        dx = x_plus[0] - x_minus[0]
        dy = y_plus[1] - y_minus[1]
        grad_x = 0.0 if dx < 1e-12 else (self.compute_risk(x_plus) - self.compute_risk(x_minus)) / dx
        grad_y = 0.0 if dy < 1e-12 else (self.compute_risk(y_plus) - self.compute_risk(y_minus)) / dy

        gradient = np.array([grad_x, grad_y], dtype=float)
        norm = np.linalg.norm(gradient)
        if norm > self.gradient_clip:
            gradient *= self.gradient_clip / norm
        return gradient

    def make_grid(self, resolution: int | Sequence[int] = 120) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(resolution, int):
            nx = ny = int(resolution)
        else:
            size = np.asarray(resolution, dtype=int)
            if size.shape != (2,):
                raise ValueError("resolution must be an int or a length-2 sequence")
            nx, ny = int(size[0]), int(size[1])
        if nx < 2 or ny < 2:
            raise ValueError("resolution must be at least 2 in each dimension")

        xs = np.linspace(0.0, self.world_size[0], nx)
        ys = np.linspace(0.0, self.world_size[1], ny)
        return np.meshgrid(xs, ys)

    def compute_hazard_grid(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        risk = self._predictive_human_risk_grid(grid_x, grid_y)
        for obstacle in self.obstacles:
            risk += self._obstacle_risk_grid(grid_x, grid_y, obstacle)
        for zone in self.zones:
            risk += self._zone_risk_grid(grid_x, grid_y, zone)
        return risk

    def compute_grid_layers(self, grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hazard_risk = self.compute_hazard_grid(grid_x, grid_y)
        total_risk = hazard_risk.copy()
        if self.w_g > 0.0:
            total_risk += self._goal_risk_grid(grid_x, grid_y)

        return hazard_risk, total_risk

    def compute_grid(
        self, resolution: int | Sequence[int] = 120
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_x, grid_y = self.make_grid(resolution)
        _, risk = self.compute_grid_layers(grid_x, grid_y)

        return grid_x, grid_y, risk

    def human_visual_risk_grid(self, grid_x: np.ndarray, grid_y: np.ndarray, human: Human) -> np.ndarray:
        dx = grid_x - human.position[0]
        dy = grid_y - human.position[1]
        distance_sq = dx**2 + dy**2
        speed = float(np.linalg.norm(human.velocity))
        if speed <= 1e-6:
            return np.exp(-(distance_sq / self._stationary_sigma_sq))

        heading = human.velocity / speed
        d_parallel = dx * heading[0] + dy * heading[1]
        d_perp_sq = np.maximum(distance_sq - d_parallel**2, 0.0)
        sigma_front = max(self.social_front_scale * self.sigma_parallel, 1.05 * self.sigma_perp)
        sigma_rear = min(self.stationary_sigma * self.social_rear_scale, 0.9 * self.sigma_perp)
        sigma_rear = max(sigma_rear, 0.45 * self.sigma_perp)
        sigma_parallel_sq = np.where(d_parallel >= 0.0, sigma_front**2, sigma_rear**2)
        return np.exp(-(d_parallel**2 / sigma_parallel_sq + d_perp_sq / self._sigma_perp_sq))

    def human_visual_risk_grid_all(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        if not self.humans:
            return np.zeros_like(grid_x)

        positions = np.asarray([human.position for human in self.humans], dtype=float)
        velocities = np.asarray([human.velocity for human in self.humans], dtype=float)
        dx = grid_x[None, :, :] - positions[:, 0][:, None, None]
        dy = grid_y[None, :, :] - positions[:, 1][:, None, None]
        distance_sq = dx**2 + dy**2
        speed = np.linalg.norm(velocities, axis=1)
        moving = speed > 1e-6
        risk = np.exp(-(distance_sq / self._stationary_sigma_sq))
        if np.any(moving):
            headings = velocities[moving] / speed[moving, None]
            moving_dx = dx[moving]
            moving_dy = dy[moving]
            moving_distance_sq = distance_sq[moving]
            d_parallel = (
                moving_dx * headings[:, 0][:, None, None]
                + moving_dy * headings[:, 1][:, None, None]
            )
            d_perp_sq = np.maximum(moving_distance_sq - d_parallel**2, 0.0)
            sigma_front = max(self.social_front_scale * self.sigma_parallel, 1.05 * self.sigma_perp)
            sigma_rear = min(self.stationary_sigma * self.social_rear_scale, 0.9 * self.sigma_perp)
            sigma_rear = max(sigma_rear, 0.45 * self.sigma_perp)
            sigma_parallel_sq = np.where(d_parallel >= 0.0, sigma_front**2, sigma_rear**2)
            risk[moving] = np.exp(
                -(d_parallel**2 / sigma_parallel_sq + d_perp_sq / self._sigma_perp_sq)
            )
        return np.max(risk, axis=0)

    def obstacle_visual_risk_grid(
        self, grid_x: np.ndarray, grid_y: np.ndarray, obstacle: Obstacle
    ) -> np.ndarray:
        distance = obstacle.distance_grid(grid_x, grid_y)
        return np.exp(-(distance**2) / self._sigma_obs_sq)

    def obstacle_visual_risk_grid_all(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        if not self.obstacles:
            return np.zeros_like(grid_x)
        layers = [self.obstacle_visual_risk_grid(grid_x, grid_y, obstacle) for obstacle in self.obstacles]
        return np.maximum.reduce(layers)

    def zone_visual_risk_grid(
        self, grid_x: np.ndarray, grid_y: np.ndarray, zone: NoGoZone
    ) -> np.ndarray:
        signed_distance = zone.signed_distance_grid(grid_x, grid_y)
        outside_risk = np.exp(-((np.maximum(signed_distance, 0.0) ** 2) / self._sigma_zone_sq))
        return np.where(signed_distance <= 0.0, 1.0, outside_risk)

    def zone_visual_risk_grid_all(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        if not self.zones:
            return np.zeros_like(grid_x)
        layers = [self.zone_visual_risk_grid(grid_x, grid_y, zone) for zone in self.zones]
        return np.maximum.reduce(layers)

    def _ensure_prediction_cache(self) -> None:
        if self._prediction_cache_valid:
            return

        cache: list[dict[str, np.ndarray | bool]] = []
        for human in self.humans:
            predicted_positions = human.position[None, :] + self._prediction_times[:, None] * human.velocity[None, :]
            predicted_positions = np.clip(predicted_positions, [0.0, 0.0], self.world_size)
            speed = np.linalg.norm(human.velocity)

            if speed < 1e-6:
                cache.append(
                    {
                        "positions": predicted_positions,
                        "moving": False,
                        "heading": np.zeros(2, dtype=float),
                        "sigma_parallel_sq": np.full_like(
                            self._prediction_times, self._stationary_sigma_sq, dtype=float
                        ),
                    }
                )
                continue

            cache.append(
                {
                    "positions": predicted_positions,
                    "moving": True,
                    "heading": human.velocity / speed,
                    "sigma_parallel_sq": self._moving_parallel_sigma_sq.copy(),
                }
            )

        self._prediction_cache = cache
        self._prediction_cache_valid = True

    def _single_predictive_human_risk(
        self,
        position: np.ndarray,
        entry: dict[str, np.ndarray | bool],
        interaction: np.ndarray,
    ) -> float:
        predicted_positions = np.asarray(entry["positions"], dtype=float)
        delta = position[None, :] - predicted_positions
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        active = distance_sq <= self._prediction_distance_sq
        if not np.any(active):
            return 0.0

        if bool(entry["moving"]):
            heading = np.asarray(entry["heading"], dtype=float)
            sigma_parallel_sq = np.asarray(entry["sigma_parallel_sq"], dtype=float)
            d_parallel = delta @ heading
            d_perp_sq = np.maximum(distance_sq - d_parallel**2, 0.0)
            collision_exponent = -0.5 * (
                d_parallel**2 / sigma_parallel_sq + d_perp_sq / self._sigma_perp_sq
            )
            social_parallel_sq = np.where(
                d_parallel >= 0.0,
                self._social_front_parallel_sigma_sq,
                self._social_rear_parallel_sigma_sq,
            )
            social_exponent = -0.5 * (
                d_parallel**2 / social_parallel_sq + d_perp_sq / self._social_sigma_perp_sq
            )
            base_weights = np.where(active, self._prediction_weights, 0.0)
            base_weight_sum = float(np.sum(base_weights))
            if base_weight_sum <= 1e-12:
                return 0.0

            weights = np.where(active, self._prediction_weights * interaction, 0.0)
            total_weight = float(np.sum(weights))
            if total_weight <= 1e-12:
                return 0.0

            normalized_weights = weights / total_weight
            relevance = total_weight / base_weight_sum
        else:
            collision_exponent = -0.5 * (distance_sq / self._stationary_sigma_sq)
            social_exponent = -0.5 * (distance_sq / self._social_stationary_sigma_sq)
            normalized_weights = np.where(active, self._prediction_weights, 0.0)
            normalized_weights /= float(np.sum(normalized_weights))
            relevance = 1.0

        collision_contribution = np.where(active, np.exp(collision_exponent), 0.0)
        human_risk = self.w_h * relevance * float(np.sum(normalized_weights * collision_contribution))
        if self.w_social > 0.0:
            social_contribution = np.where(active, np.exp(social_exponent), 0.0)
            human_risk += self.w_social * relevance * float(
                np.sum(normalized_weights * social_contribution)
            )
        return human_risk

    def _predictive_human_risk(self, position: np.ndarray) -> float:
        self._ensure_prediction_cache()
        position = _as_vector(position, "position")
        robot_time = np.linalg.norm(position - self._robot_reference_position) / self._robot_reference_speed
        interaction = np.exp(
            -((self._prediction_times - robot_time) ** 2) / self._interaction_tau_denom
        )
        risk = 0.0

        for entry in self._prediction_cache:
            human_risk = self._single_predictive_human_risk(position, entry, interaction)
            if self.multi_human_aggregation:
                risk += human_risk
            else:
                risk = max(risk, human_risk)

        return risk

    def _predictive_human_risk_grid(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        self._ensure_prediction_cache()
        if not self._prediction_cache:
            return np.zeros_like(grid_x)

        robot_time = (
            np.hypot(
                grid_x - self._robot_reference_position[0],
                grid_y - self._robot_reference_position[1],
            )
            / self._robot_reference_speed
        )
        interaction = np.exp(
            -((self._prediction_times[:, None, None] - robot_time[None, :, :]) ** 2) / self._interaction_tau_denom
        )
        predicted_positions = np.stack(
            [np.asarray(entry["positions"], dtype=float) for entry in self._prediction_cache],
            axis=0,
        )
        moving = np.asarray([bool(entry["moving"]) for entry in self._prediction_cache], dtype=bool)
        dx = grid_x[None, None, :, :] - predicted_positions[:, :, 0][:, :, None, None]
        dy = grid_y[None, None, :, :] - predicted_positions[:, :, 1][:, :, None, None]
        distance_sq = dx**2 + dy**2
        active = distance_sq <= self._prediction_distance_sq
        prediction_weights = self._prediction_weights[:, None, None]
        interaction_weights = prediction_weights * interaction
        human_risk = np.zeros((len(self._prediction_cache),) + grid_x.shape, dtype=float)

        if np.any(moving):
            moving_dx = dx[moving]
            moving_dy = dy[moving]
            moving_distance_sq = distance_sq[moving]
            moving_active = active[moving]
            headings = np.stack(
                [np.asarray(entry["heading"], dtype=float) for entry in self._prediction_cache if bool(entry["moving"])],
                axis=0,
            )
            sigma_parallel_sq = np.stack(
                [
                    np.asarray(entry["sigma_parallel_sq"], dtype=float)
                    for entry in self._prediction_cache
                    if bool(entry["moving"])
                ],
                axis=0,
            )[:, :, None, None]
            d_parallel = (
                moving_dx * headings[:, None, 0][:, :, None, None]
                + moving_dy * headings[:, None, 1][:, :, None, None]
            )
            d_perp_sq = np.maximum(moving_distance_sq - d_parallel**2, 0.0)
            collision_exponent = -0.5 * (
                d_parallel**2 / sigma_parallel_sq + d_perp_sq / self._sigma_perp_sq
            )
            social_parallel_sq = np.where(
                d_parallel >= 0.0,
                self._social_front_parallel_sigma_sq[None, :, None, None],
                self._social_rear_parallel_sigma_sq[None, :, None, None],
            )
            social_exponent = -0.5 * (
                d_parallel**2 / social_parallel_sq + d_perp_sq / self._social_sigma_perp_sq
            )
            base_weights = np.where(moving_active, prediction_weights[None], 0.0)
            base_weight_sum = np.sum(base_weights, axis=1, keepdims=True)
            weights = np.where(moving_active, interaction_weights[None], 0.0)
            weight_sum = np.sum(weights, axis=1, keepdims=True)
            normalized_weights = np.divide(
                weights,
                weight_sum,
                out=np.zeros_like(weights),
                where=weight_sum > 1e-12,
            )
            relevance = np.divide(
                weight_sum,
                base_weight_sum,
                out=np.zeros_like(weight_sum),
                where=base_weight_sum > 1e-12,
            )[:, 0]
            collision_contribution = np.where(moving_active, np.exp(collision_exponent), 0.0)
            moving_human_risk = self.w_h * relevance * np.sum(
                normalized_weights * collision_contribution,
                axis=1,
            )
            if self.w_social > 0.0:
                social_contribution = np.where(moving_active, np.exp(social_exponent), 0.0)
                moving_human_risk += self.w_social * relevance * np.sum(
                    normalized_weights * social_contribution,
                    axis=1,
                )
            human_risk[moving] = moving_human_risk

        if np.any(~moving):
            stationary_distance_sq = distance_sq[~moving]
            stationary_active = active[~moving]
            collision_exponent = -0.5 * (stationary_distance_sq / self._stationary_sigma_sq)
            social_exponent = -0.5 * (stationary_distance_sq / self._social_stationary_sigma_sq)
            weights = np.where(stationary_active, prediction_weights[None], 0.0)
            weight_sum = np.sum(weights, axis=1, keepdims=True)
            normalized_weights = np.divide(
                weights,
                weight_sum,
                out=np.zeros_like(weights),
                where=weight_sum > 1e-12,
            )
            collision_contribution = np.where(stationary_active, np.exp(collision_exponent), 0.0)
            stationary_human_risk = self.w_h * np.sum(
                normalized_weights * collision_contribution,
                axis=1,
            )
            if self.w_social > 0.0:
                social_contribution = np.where(stationary_active, np.exp(social_exponent), 0.0)
                stationary_human_risk += self.w_social * np.sum(
                    normalized_weights * social_contribution,
                    axis=1,
                )
            human_risk[~moving] = stationary_human_risk

        if self.multi_human_aggregation:
            return np.sum(human_risk, axis=0)
        return np.max(human_risk, axis=0)

    def _obstacle_risk(self, position: np.ndarray, obstacle: Obstacle) -> float:
        distance = obstacle.distance_to_surface(position)
        return self.w_o * np.exp(-0.5 * (distance**2) / self._sigma_obs_sq)

    def _obstacle_risk_grid(
        self, grid_x: np.ndarray, grid_y: np.ndarray, obstacle: Obstacle
    ) -> np.ndarray:
        distance = obstacle.distance_grid(grid_x, grid_y)
        return self.w_o * np.exp(-0.5 * (distance**2) / self._sigma_obs_sq)

    def _zone_risk(self, position: np.ndarray, zone: NoGoZone) -> float:
        signed_distance = zone.signed_distance(position)
        if signed_distance <= 0.0:
            inside_depth = -signed_distance
            return self.w_zone * (1.0 + self.zone_inside_gain * (inside_depth**2) / self._sigma_zone_sq)
        return self.w_zone * np.exp(-0.5 * (signed_distance**2) / self._sigma_zone_sq)

    def _zone_risk_grid(
        self, grid_x: np.ndarray, grid_y: np.ndarray, zone: NoGoZone
    ) -> np.ndarray:
        signed_distance = zone.signed_distance_grid(grid_x, grid_y)
        outside_risk = self.w_zone * np.exp(
            -0.5 * (np.maximum(signed_distance, 0.0) ** 2) / self._sigma_zone_sq
        )
        inside_depth_sq = np.maximum(-signed_distance, 0.0) ** 2
        inside_risk = self.w_zone * (
            1.0 + self.zone_inside_gain * inside_depth_sq / self._sigma_zone_sq
        )
        return np.where(signed_distance <= 0.0, inside_risk, outside_risk)

    def _goal_risk(self, position: np.ndarray) -> float:
        delta = position - self.goal
        distance_sq = float(np.dot(delta, delta))
        return -self.w_g * np.exp(-0.5 * distance_sq / self._sigma_goal_sq)

    def _goal_risk_grid(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        distance_sq = (grid_x - self.goal[0]) ** 2 + (grid_y - self.goal[1]) ** 2
        return -self.w_g * np.exp(-0.5 * distance_sq / self._sigma_goal_sq)


@dataclass
class TTCInteraction:
    human_index: int
    ttc: float
    clearance: float
    lateral_offset: float
    forward_alignment: float
    predicted_time: float
    strength: float
    lateral_direction: np.ndarray
    interaction_vector: np.ndarray


@dataclass
class PostPassContext:
    human_index: int
    committed_side: float
    longitudinal_axis: np.ndarray
    side_direction: np.ndarray
    reference_relative_position: np.ndarray
    last_relative_position: np.ndarray
    previous_clearance: float


@dataclass
class Robot:
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    speed: float = 1.0
    radius: float = 0.2
    goal_tolerance: float = 0.15
    goal_slowdown_distance: float = 1.5
    momentum: float = 0.8
    grad_smoothing: float = 0.8
    max_turn_angle: float = float(np.deg2rad(25.0))
    min_forward_dot: float = 0.2
    goal_bias: float = 0.2
    grad_scale_k: float = 0.35
    safe_distance: float = 1.2
    safety_distance: float = 0.5
    safety_hysteresis: float = 0.08
    risk_threshold: float = 0.75
    high_risk_gain: float = 2.4
    repulsion_gain: float = 1.7
    risk_blend_sharpness: float = 10.0
    repulsion_zone_scale: float = 2.0
    barrier_scale: float = 0.5
    barrier_gain: float = 4.0
    hard_min_clearance: float = 0.7
    close_avoidance_gain: float = 2.8
    close_risk_beta: float = 2.0
    close_gradient_gamma: float = 0.9
    emergency_brake_clearance: float = 0.5
    emergency_brake_speed: float = 0.15
    safety_projection_margin: float = 0.25
    safety_envelope_buffer: float = 0.5
    min_clearance: float = 0.35
    grad_epsilon: float = 0.05
    direction_epsilon: float = 0.05
    velocity_smoothing: float = 0.8
    alpha_damping: float = 0.95
    ttc_mid: float = 2.0
    ttc_slope: float = 0.5
    ttc_stop: float = 1.0
    ttc_resume: float = 1.4
    safe_clearance: float = 0.95
    risk_speed_gain: float = 0.32
    min_speed_scale: float = 0.15
    goal_ttc_disable_distance: float = 0.35
    ttc_human_speed_threshold: float = 0.05
    interaction_forward_threshold: float = 0.15
    lateral_safe_distance: float = 1.2
    lateral_block_threshold: float = 0.55
    ttc_attention_threshold: float = 6.5
    interaction_beta: float = 0.6
    interaction_prediction_dt: float = 0.2
    static_escape_lookahead_time: float = 0.6
    static_escape_lookahead_samples: int = 5
    clearance_push_gain: float = 5.2
    comfort_clearance: float = 1.15
    interaction_persistence_tau: float = 0.75
    interaction_memory_gain: float = 1.8
    interaction_memory_floor_ratio: float = 0.6
    interaction_current_blend: float = 0.4
    interaction_memory_blend: float = 0.6
    interaction_strength_sharpness: float = 6.0
    interaction_min_strength: float = 0.035
    interaction_distance_sigma: float = 2.2
    interaction_ttc_sigma: float = 3.0
    interaction_max_active_humans: int = 2
    primary_interaction_weight: float = 2.1
    interaction_force_memory_decay: float = 0.85
    interaction_force_memory_gain: float = 1.35
    side_commitment_decay: float = 0.95
    side_commitment_gain: float = 1.4
    interaction_recovery_threshold: float = 0.02
    speed_interaction_gain: float = 1.05
    speed_scale_smoothing: float = 0.4
    yield_score_enter: float = 0.45
    yield_score_exit: float = 0.25
    interaction_switch_margin: float = 0.35
    interaction_conflict_margin: float = 0.15
    interaction_conflict_beta_scale: float = 0.6
    recovery_speed_bias: float = 0.9
    interaction_enter_clearance: float = 2.5
    interaction_exit_clearance: float = 3.0
    interaction_exit_threshold: float = 0.08
    interaction_trigger_threshold_active: float = 0.08
    interaction_trigger_threshold_inactive: float = 0.18
    interaction_signal_alpha: float = 0.4
    interaction_deadband: float = 0.05
    interaction_min_hold_time: float = 0.5
    interaction_rearm_time: float = 0.3
    interaction_decay_rate: float = 0.8
    interaction_effective_gamma: float = 1.2
    interaction_effective_cap: float = 0.85
    interaction_fast_decay: float = 0.75
    interaction_reset_clearance: float = 4.0
    interaction_gap_relaxation_threshold: float = 0.02
    interaction_ignore_secondary_gap: float = 0.015
    interaction_recovery_gain: float = 2.0
    interaction_far_clearance: float = 4.0
    risk_distance_scale: float = 10.0
    lateral_gradient_scale: float = 0.5
    progress_speed_floor: float = 0.25
    interaction_direction_relax_factor: float = 0.5
    interaction_goal_commit_clearance: float = 3.0
    interaction_speed_confidence_clearance: float = 3.5
    post_interaction_speed_boost: float = 0.15
    post_interaction_relief_time: float = 0.5
    post_interaction_risk_scale: float = 0.65
    goal_commit_gain: float = 1.2
    goal_seek_weight: float = 1.8
    deviation_relax_factor: float = 0.3
    progress_enforcement_threshold: float = 0.1
    progress_goal_boost_gain: float = 0.4
    progress_sigmoid_slope: float = 5.0
    weak_interaction_disable_threshold: float = 0.12
    goal_stabilization_distance: float = 2.0
    heading_rate_limit: float = 0.15
    recovery_goal_scale: float = 1.2
    recovery_lateral_scale: float = 0.7
    hard_turn_limit: float = 0.3
    curvature_damping: float = 0.5
    post_pass_speed_boost: float = 0.9
    post_pass_interaction_scale: float = 0.5
    weak_interaction_goal_threshold: float = 0.06
    weak_interaction_goal_blend_gain: float = 0.55
    conflict_goal_blend_gain: float = 0.2
    stop_clearance_threshold: float = 0.6
    approach_stop_clearance: float = 1.0
    stop_resume_clearance: float = 1.2
    stop_resume_hold_time: float = 0.7
    interaction_shutdown_clearance: float = 1.2
    approach_clearance_threshold: float = 1.2
    approach_speed_gain: float = 0.7
    rear_approach_speed_cap: float = 0.6
    weak_suppression: bool = True
    interaction_memory_enabled: bool = True
    topk_filter: bool = True
    guide_planner_enabled: bool = True
    failed_branch_memory_enabled: bool = True
    execution_progress_invariant_enabled: bool = True
    interaction_mode_active: bool = False
    stop_mode_active: bool = False
    behavior_state: str = FSMState.GOAL_SEEK.value
    behavior_speed_scale: float = 1.0
    ttc_min: float = float("inf")
    clearance_min: float = float("inf")
    interaction_clearance: float = float("inf")
    global_clearance: float = float("inf")
    physical_clearance: float = float("inf")
    interaction_clearance_rate: float = 0.0
    interacting_human_id: int = -1
    interaction_level_current: float = 0.0
    interaction_level: float = 0.0
    interaction_hold_timer: float = 0.0
    interaction_active_duration: float = 0.0
    interaction_rearm_timer: float = 0.0
    interaction_deadlock_timer: float = 0.0
    interaction_release_timer: float = 0.0
    interaction_release_direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    interaction_release_projection_anchor: float = 0.0
    interaction_release_path_distance_anchor: float = 0.0
    interaction_release_anchor_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    interaction_release_progress: float = 0.0
    interaction_release_required_progress: float = 0.0
    post_interaction_relief_timer: float = 0.0
    stop_resume_timer: float = 0.0
    stagnation_counter: int = 0
    stop_mode_reason: str = "none"
    active_interactions: List[TTCInteraction] = field(default_factory=list)
    trail: List[np.ndarray] = field(default_factory=list)
    prev_grad: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    prev_direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    interaction_force_memory: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    committed_side_direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    escape_direction_memory: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    escape_commit_steps_remaining: int = 0
    escape_progress_window: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    goal_progress_window: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    goal_distance_window: deque[float] = field(default_factory=lambda: deque(maxlen=24))
    progress_threshold: float = 0.08
    progress_variance_threshold: float = 0.02
    progress_regression_tolerance: float = 0.02
    escape_commit_steps: int = 28
    escape_flip_lock_steps: int = 6
    escape_retrigger_window_steps: int = 240
    escape_retrigger_threshold: int = 2
    escape_gain: float = 1.0
    escape_activation_steps: deque[int] = field(default_factory=lambda: deque(maxlen=8))
    control_step_counter: int = 0
    static_guide_waypoints: List[np.ndarray] = field(default_factory=list)
    static_guide_index: int = 0
    static_guide_anchor_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    static_guide_anchor_progress: float = 0.0
    guide_progress_max: float = 0.0
    guide_progress_regression_tolerance: float = 1e-4
    detour_active: bool = False
    detour_direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    detour_steps_remaining: int = 0
    detour_anchor_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    detour_anchor_distance: float = float("inf")
    detour_last_progress: float = 0.0
    current_branch_point: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    current_branch_guide_direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    detour_lock_direction: np.ndarray | None = None
    detour_lock_steps: int = 0
    detour_side: int = 0
    prev_goal_dist: float = float("inf")
    last_flip_step: int = -1000
    detour_side_flip_count: int = 0
    failed_branches: list[FailedBranch] = field(default_factory=list)
    branch_rebuild_required: bool = False
    failed_detours: list[np.ndarray] = field(default_factory=list)
    last_detour_direction: np.ndarray | None = None
    interaction_level_memory: float = 0.0
    dominant_human_memory_id: int = -1
    post_pass_context: PostPassContext | None = None
    previous_interaction_clearance: float = float("inf")
    previous_interacting_human_id: int = -1
    interaction_switch_count: int = 0
    per_human_min_clearances: list[float] = field(default_factory=list)
    global_min_clearance: float = float("inf")
    minimum_physical_clearance: float = float("inf")
    previous_total_risk: float = 0.0
    previous_rule_clearance: float = float("inf")
    safety_margin: float = float("inf")
    risk_slope: float = 0.0
    latest_recovery_time: float = 0.0
    recovery_timer_active: bool = False
    recovery_timer_elapsed: float = 0.0
    static_escape_duration: float = 0.0
    static_escape_cooldown: int = 0
    deadlock_timer: float = 0.0
    deadlock_recovery_timer: float = 0.0
    progress_stall_timer: float = 0.0
    previous_goal_distance: float = float("inf")
    top_1_interaction_strength: float = 0.0
    top_2_interaction_strength: float = 0.0
    interaction_strength_gap: float = 0.0
    multi_dominant_interaction: bool = False
    human_interaction_distances: list[float] = field(default_factory=list)
    human_interaction_ttc: list[float] = field(default_factory=list)
    human_interaction_alignments: list[float] = field(default_factory=list)
    human_interaction_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "robot.position")
        self.velocity = _as_vector(self.velocity, "robot.velocity")
        self.goal = _as_vector(self.goal, "robot.goal")
        self.speed = _validate_positive(self.speed, "robot.speed")
        self.radius = _validate_positive(self.radius, "robot.radius")
        self.goal_tolerance = _validate_positive(self.goal_tolerance, "robot.goal_tolerance")
        self.goal_slowdown_distance = _validate_positive(
            self.goal_slowdown_distance, "robot.goal_slowdown_distance"
        )
        self.momentum = _validate_unit_interval(self.momentum, "robot.momentum")
        self.grad_smoothing = _validate_unit_interval(
            self.grad_smoothing, "robot.grad_smoothing"
        )
        self.max_turn_angle = _validate_positive(self.max_turn_angle, "robot.max_turn_angle")
        self.min_forward_dot = _validate_unit_interval(
            self.min_forward_dot, "robot.min_forward_dot"
        )
        self.goal_bias = _validate_nonnegative(self.goal_bias, "robot.goal_bias")
        self.grad_scale_k = _validate_positive(self.grad_scale_k, "robot.grad_scale_k")
        self.safe_distance = _validate_positive(self.safe_distance, "robot.safe_distance")
        self.safety_distance = max(
            _validate_positive(self.safety_distance, "robot.safety_distance"),
            0.5,
        )
        self.safety_hysteresis = _validate_nonnegative(
            self.safety_hysteresis, "robot.safety_hysteresis"
        )
        self.risk_threshold = _validate_nonnegative(self.risk_threshold, "robot.risk_threshold")
        self.high_risk_gain = _validate_positive(self.high_risk_gain, "robot.high_risk_gain")
        self.repulsion_gain = _validate_nonnegative(self.repulsion_gain, "robot.repulsion_gain")
        self.risk_blend_sharpness = _validate_positive(
            self.risk_blend_sharpness, "robot.risk_blend_sharpness"
        )
        self.repulsion_zone_scale = _validate_positive(
            self.repulsion_zone_scale, "robot.repulsion_zone_scale"
        )
        self.barrier_scale = _validate_positive(self.barrier_scale, "robot.barrier_scale")
        self.barrier_gain = _validate_nonnegative(self.barrier_gain, "robot.barrier_gain")
        self.hard_min_clearance = _validate_positive(
            self.hard_min_clearance, "robot.hard_min_clearance"
        )
        self.close_avoidance_gain = _validate_nonnegative(
            self.close_avoidance_gain, "robot.close_avoidance_gain"
        )
        self.close_risk_beta = _validate_nonnegative(
            self.close_risk_beta, "robot.close_risk_beta"
        )
        self.close_gradient_gamma = _validate_nonnegative(
            self.close_gradient_gamma, "robot.close_gradient_gamma"
        )
        self.emergency_brake_clearance = _validate_nonnegative(
            self.emergency_brake_clearance, "robot.emergency_brake_clearance"
        )
        self.emergency_brake_speed = _validate_nonnegative(
            self.emergency_brake_speed, "robot.emergency_brake_speed"
        )
        self.safety_projection_margin = _validate_nonnegative(
            self.safety_projection_margin, "robot.safety_projection_margin"
        )
        self.safety_envelope_buffer = _validate_positive(
            self.safety_envelope_buffer, "robot.safety_envelope_buffer"
        )
        self.min_clearance = _validate_positive(self.min_clearance, "robot.min_clearance")
        self.grad_epsilon = _validate_nonnegative(self.grad_epsilon, "robot.grad_epsilon")
        self.direction_epsilon = _validate_nonnegative(
            self.direction_epsilon, "robot.direction_epsilon"
        )
        self.velocity_smoothing = _validate_unit_interval(
            self.velocity_smoothing, "robot.velocity_smoothing"
        )
        self.alpha_damping = _validate_unit_interval(self.alpha_damping, "robot.alpha_damping")
        self.ttc_mid = _validate_positive(self.ttc_mid, "robot.ttc_mid")
        self.ttc_slope = _validate_positive(self.ttc_slope, "robot.ttc_slope")
        self.ttc_stop = _validate_positive(self.ttc_stop, "robot.ttc_stop")
        self.ttc_resume = _validate_positive(self.ttc_resume, "robot.ttc_resume")
        if self.ttc_resume <= self.ttc_stop:
            raise ValueError("robot.ttc_resume must be greater than robot.ttc_stop")
        self.safe_clearance = _validate_positive(self.safe_clearance, "robot.safe_clearance")
        self.risk_speed_gain = _validate_nonnegative(self.risk_speed_gain, "robot.risk_speed_gain")
        self.min_speed_scale = _validate_unit_interval(self.min_speed_scale, "robot.min_speed_scale")
        self.goal_ttc_disable_distance = _validate_positive(
            self.goal_ttc_disable_distance, "robot.goal_ttc_disable_distance"
        )
        self.ttc_human_speed_threshold = _validate_nonnegative(
            self.ttc_human_speed_threshold, "robot.ttc_human_speed_threshold"
        )
        self.interaction_forward_threshold = _validate_unit_interval(
            self.interaction_forward_threshold, "robot.interaction_forward_threshold"
        )
        self.lateral_safe_distance = _validate_positive(
            self.lateral_safe_distance, "robot.lateral_safe_distance"
        )
        self.lateral_block_threshold = _validate_positive(
            self.lateral_block_threshold, "robot.lateral_block_threshold"
        )
        self.ttc_attention_threshold = _validate_positive(
            self.ttc_attention_threshold, "robot.ttc_attention_threshold"
        )
        self.interaction_beta = _validate_positive(self.interaction_beta, "robot.interaction_beta")
        self.interaction_prediction_dt = _validate_positive(
            self.interaction_prediction_dt, "robot.interaction_prediction_dt"
        )
        self.static_escape_lookahead_time = _validate_positive(
            self.static_escape_lookahead_time, "robot.static_escape_lookahead_time"
        )
        self.static_escape_lookahead_samples = max(int(self.static_escape_lookahead_samples), 1)
        self.clearance_push_gain = _validate_nonnegative(
            self.clearance_push_gain, "robot.clearance_push_gain"
        )
        self.comfort_clearance = _validate_positive(
            self.comfort_clearance, "robot.comfort_clearance"
        )
        self.interaction_persistence_tau = _validate_positive(
            self.interaction_persistence_tau, "robot.interaction_persistence_tau"
        )
        self.interaction_memory_gain = _validate_nonnegative(
            self.interaction_memory_gain, "robot.interaction_memory_gain"
        )
        self.interaction_memory_floor_ratio = _validate_unit_interval(
            self.interaction_memory_floor_ratio, "robot.interaction_memory_floor_ratio"
        )
        self.interaction_current_blend = _validate_unit_interval(
            self.interaction_current_blend, "robot.interaction_current_blend"
        )
        self.interaction_memory_blend = _validate_unit_interval(
            self.interaction_memory_blend, "robot.interaction_memory_blend"
        )
        if abs(
            (self.interaction_current_blend + self.interaction_memory_blend) - 1.0
        ) > 1e-6:
            raise ValueError(
                "robot.interaction_current_blend and robot.interaction_memory_blend must sum to 1"
            )
        self.interaction_strength_sharpness = _validate_positive(
            self.interaction_strength_sharpness, "robot.interaction_strength_sharpness"
        )
        self.interaction_min_strength = _validate_nonnegative(
            self.interaction_min_strength, "robot.interaction_min_strength"
        )
        self.interaction_distance_sigma = _validate_positive(
            self.interaction_distance_sigma, "robot.interaction_distance_sigma"
        )
        self.interaction_ttc_sigma = _validate_positive(
            self.interaction_ttc_sigma, "robot.interaction_ttc_sigma"
        )
        self.interaction_max_active_humans = max(int(self.interaction_max_active_humans), 1)
        self.primary_interaction_weight = _validate_positive(
            self.primary_interaction_weight, "robot.primary_interaction_weight"
        )
        self.interaction_force_memory_decay = _validate_unit_interval(
            self.interaction_force_memory_decay, "robot.interaction_force_memory_decay"
        )
        self.interaction_force_memory_gain = _validate_nonnegative(
            self.interaction_force_memory_gain, "robot.interaction_force_memory_gain"
        )
        self.side_commitment_decay = _validate_unit_interval(
            self.side_commitment_decay, "robot.side_commitment_decay"
        )
        self.side_commitment_gain = _validate_nonnegative(
            self.side_commitment_gain, "robot.side_commitment_gain"
        )
        self.interaction_recovery_threshold = _validate_nonnegative(
            self.interaction_recovery_threshold, "robot.interaction_recovery_threshold"
        )
        self.speed_interaction_gain = _validate_nonnegative(
            self.speed_interaction_gain, "robot.speed_interaction_gain"
        )
        self.speed_scale_smoothing = _validate_unit_interval(
            self.speed_scale_smoothing, "robot.speed_scale_smoothing"
        )
        self.yield_score_enter = _validate_unit_interval(
            self.yield_score_enter, "robot.yield_score_enter"
        )
        self.yield_score_exit = _validate_unit_interval(
            self.yield_score_exit, "robot.yield_score_exit"
        )
        if self.yield_score_exit >= self.yield_score_enter:
            raise ValueError("robot.yield_score_exit must be smaller than robot.yield_score_enter")
        self.interaction_switch_margin = _validate_unit_interval(
            self.interaction_switch_margin, "robot.interaction_switch_margin"
        )
        self.interaction_conflict_margin = _validate_unit_interval(
            self.interaction_conflict_margin, "robot.interaction_conflict_margin"
        )
        self.interaction_conflict_beta_scale = _validate_unit_interval(
            self.interaction_conflict_beta_scale, "robot.interaction_conflict_beta_scale"
        )
        self.recovery_speed_bias = _validate_unit_interval(
            self.recovery_speed_bias, "robot.recovery_speed_bias"
        )
        self.interaction_enter_clearance = _validate_positive(
            self.interaction_enter_clearance, "robot.interaction_enter_clearance"
        )
        self.interaction_exit_clearance = _validate_positive(
            self.interaction_exit_clearance, "robot.interaction_exit_clearance"
        )
        if self.interaction_exit_clearance <= self.interaction_enter_clearance:
            raise ValueError(
                "robot.interaction_exit_clearance must be greater than robot.interaction_enter_clearance"
            )
        self.interaction_exit_threshold = _validate_nonnegative(
            self.interaction_exit_threshold, "robot.interaction_exit_threshold"
        )
        self.interaction_trigger_threshold_active = _validate_nonnegative(
            self.interaction_trigger_threshold_active, "robot.interaction_trigger_threshold_active"
        )
        self.interaction_trigger_threshold_inactive = _validate_nonnegative(
            self.interaction_trigger_threshold_inactive, "robot.interaction_trigger_threshold_inactive"
        )
        self.interaction_signal_alpha = _validate_unit_interval(
            self.interaction_signal_alpha, "robot.interaction_signal_alpha"
        )
        self.interaction_deadband = _validate_nonnegative(
            self.interaction_deadband, "robot.interaction_deadband"
        )
        self.interaction_min_hold_time = _validate_nonnegative(
            self.interaction_min_hold_time, "robot.interaction_min_hold_time"
        )
        self.interaction_rearm_time = _validate_nonnegative(
            self.interaction_rearm_time, "robot.interaction_rearm_time"
        )
        self.interaction_deadlock_timer = _validate_nonnegative(
            self.interaction_deadlock_timer, "robot.interaction_deadlock_timer"
        )
        self.interaction_release_timer = _validate_nonnegative(
            self.interaction_release_timer, "robot.interaction_release_timer"
        )
        self.interaction_release_direction = _as_vector(
            self.interaction_release_direction, "robot.interaction_release_direction"
        )
        self.interaction_release_projection_anchor = _validate_nonnegative(
            self.interaction_release_projection_anchor,
            "robot.interaction_release_projection_anchor",
        )
        self.interaction_release_path_distance_anchor = _validate_nonnegative(
            self.interaction_release_path_distance_anchor,
            "robot.interaction_release_path_distance_anchor",
        )
        self.interaction_release_anchor_position = _as_vector(
            self.interaction_release_anchor_position, "robot.interaction_release_anchor_position"
        )
        self.interaction_release_progress = _validate_nonnegative(
            self.interaction_release_progress, "robot.interaction_release_progress"
        )
        self.interaction_release_required_progress = _validate_nonnegative(
            self.interaction_release_required_progress, "robot.interaction_release_required_progress"
        )
        self.interaction_decay_rate = _validate_unit_interval(
            self.interaction_decay_rate, "robot.interaction_decay_rate"
        )
        self.interaction_effective_gamma = _validate_positive(
            self.interaction_effective_gamma, "robot.interaction_effective_gamma"
        )
        self.interaction_effective_cap = _validate_unit_interval(
            self.interaction_effective_cap, "robot.interaction_effective_cap"
        )
        self.interaction_fast_decay = _validate_unit_interval(
            self.interaction_fast_decay, "robot.interaction_fast_decay"
        )
        self.interaction_reset_clearance = _validate_positive(
            self.interaction_reset_clearance, "robot.interaction_reset_clearance"
        )
        self.interaction_gap_relaxation_threshold = _validate_positive(
            self.interaction_gap_relaxation_threshold, "robot.interaction_gap_relaxation_threshold"
        )
        self.interaction_ignore_secondary_gap = _validate_positive(
            self.interaction_ignore_secondary_gap, "robot.interaction_ignore_secondary_gap"
        )
        self.interaction_recovery_gain = _validate_nonnegative(
            self.interaction_recovery_gain, "robot.interaction_recovery_gain"
        )
        self.interaction_far_clearance = _validate_positive(
            self.interaction_far_clearance, "robot.interaction_far_clearance"
        )
        self.risk_distance_scale = _validate_positive(
            self.risk_distance_scale, "robot.risk_distance_scale"
        )
        self.lateral_gradient_scale = _validate_unit_interval(
            self.lateral_gradient_scale, "robot.lateral_gradient_scale"
        )
        self.progress_speed_floor = _validate_nonnegative(
            self.progress_speed_floor, "robot.progress_speed_floor"
        )
        self.interaction_direction_relax_factor = _validate_unit_interval(
            self.interaction_direction_relax_factor, "robot.interaction_direction_relax_factor"
        )
        self.interaction_goal_commit_clearance = _validate_positive(
            self.interaction_goal_commit_clearance, "robot.interaction_goal_commit_clearance"
        )
        self.interaction_speed_confidence_clearance = _validate_positive(
            self.interaction_speed_confidence_clearance, "robot.interaction_speed_confidence_clearance"
        )
        self.post_interaction_speed_boost = _validate_nonnegative(
            self.post_interaction_speed_boost, "robot.post_interaction_speed_boost"
        )
        self.post_interaction_relief_time = _validate_nonnegative(
            self.post_interaction_relief_time, "robot.post_interaction_relief_time"
        )
        self.post_interaction_risk_scale = _validate_unit_interval(
            self.post_interaction_risk_scale, "robot.post_interaction_risk_scale"
        )
        self.goal_commit_gain = _validate_positive(
            self.goal_commit_gain, "robot.goal_commit_gain"
        )
        self.goal_seek_weight = _validate_positive(
            self.goal_seek_weight, "robot.goal_seek_weight"
        )
        self.deviation_relax_factor = _validate_unit_interval(
            self.deviation_relax_factor, "robot.deviation_relax_factor"
        )
        self.progress_enforcement_threshold = _validate_nonnegative(
            self.progress_enforcement_threshold, "robot.progress_enforcement_threshold"
        )
        self.progress_goal_boost_gain = _validate_nonnegative(
            self.progress_goal_boost_gain, "robot.progress_goal_boost_gain"
        )
        self.progress_sigmoid_slope = _validate_positive(
            self.progress_sigmoid_slope, "robot.progress_sigmoid_slope"
        )
        self.weak_interaction_disable_threshold = _validate_nonnegative(
            self.weak_interaction_disable_threshold, "robot.weak_interaction_disable_threshold"
        )
        self.goal_stabilization_distance = _validate_positive(
            self.goal_stabilization_distance, "robot.goal_stabilization_distance"
        )
        self.progress_threshold = _validate_nonnegative(
            self.progress_threshold, "robot.progress_threshold"
        )
        self.progress_variance_threshold = _validate_nonnegative(
            self.progress_variance_threshold, "robot.progress_variance_threshold"
        )
        self.progress_regression_tolerance = _validate_nonnegative(
            self.progress_regression_tolerance, "robot.progress_regression_tolerance"
        )
        self.escape_commit_steps = max(int(self.escape_commit_steps), 1)
        self.escape_flip_lock_steps = max(int(self.escape_flip_lock_steps), 1)
        self.escape_retrigger_window_steps = max(int(self.escape_retrigger_window_steps), 1)
        self.escape_retrigger_threshold = max(int(self.escape_retrigger_threshold), 1)
        self.escape_gain = float(np.clip(self.escape_gain, 1.0, 2.0))
        self.heading_rate_limit = _validate_positive(
            self.heading_rate_limit, "robot.heading_rate_limit"
        )
        self.recovery_goal_scale = _validate_positive(
            self.recovery_goal_scale, "robot.recovery_goal_scale"
        )
        self.recovery_lateral_scale = _validate_unit_interval(
            self.recovery_lateral_scale, "robot.recovery_lateral_scale"
        )
        self.hard_turn_limit = _validate_positive(
            self.hard_turn_limit, "robot.hard_turn_limit"
        )
        self.curvature_damping = _validate_unit_interval(
            self.curvature_damping, "robot.curvature_damping"
        )
        self.post_pass_speed_boost = _validate_unit_interval(
            self.post_pass_speed_boost, "robot.post_pass_speed_boost"
        )
        self.post_pass_interaction_scale = _validate_unit_interval(
            self.post_pass_interaction_scale, "robot.post_pass_interaction_scale"
        )
        self.weak_interaction_goal_threshold = _validate_nonnegative(
            self.weak_interaction_goal_threshold, "robot.weak_interaction_goal_threshold"
        )
        self.weak_interaction_goal_blend_gain = _validate_nonnegative(
            self.weak_interaction_goal_blend_gain, "robot.weak_interaction_goal_blend_gain"
        )
        self.conflict_goal_blend_gain = _validate_nonnegative(
            self.conflict_goal_blend_gain, "robot.conflict_goal_blend_gain"
        )
        self.stop_clearance_threshold = _validate_nonnegative(
            self.stop_clearance_threshold, "robot.stop_clearance_threshold"
        )
        self.approach_stop_clearance = _validate_positive(
            self.approach_stop_clearance, "robot.approach_stop_clearance"
        )
        self.stop_resume_clearance = _validate_positive(
            self.stop_resume_clearance, "robot.stop_resume_clearance"
        )
        if self.approach_stop_clearance < self.stop_clearance_threshold:
            raise ValueError(
                "robot.approach_stop_clearance must be at least robot.stop_clearance_threshold"
            )
        if self.stop_resume_clearance <= self.stop_clearance_threshold:
            raise ValueError(
                "robot.stop_resume_clearance must be greater than robot.stop_clearance_threshold"
            )
        self.stop_resume_hold_time = _validate_nonnegative(
            self.stop_resume_hold_time, "robot.stop_resume_hold_time"
        )
        self.interaction_shutdown_clearance = _validate_positive(
            self.interaction_shutdown_clearance, "robot.interaction_shutdown_clearance"
        )
        self.approach_clearance_threshold = _validate_positive(
            self.approach_clearance_threshold, "robot.approach_clearance_threshold"
        )
        self.approach_speed_gain = _validate_nonnegative(
            self.approach_speed_gain, "robot.approach_speed_gain"
        )
        self.rear_approach_speed_cap = _validate_unit_interval(
            self.rear_approach_speed_cap, "robot.rear_approach_speed_cap"
        )
        self.weak_suppression = bool(self.weak_suppression)
        self.interaction_memory_enabled = bool(self.interaction_memory_enabled)
        self.topk_filter = bool(self.topk_filter)
        self.guide_planner_enabled = bool(self.guide_planner_enabled)
        self.failed_branch_memory_enabled = bool(self.failed_branch_memory_enabled)
        self.execution_progress_invariant_enabled = bool(self.execution_progress_invariant_enabled)
        self.interaction_mode_active = bool(self.interaction_mode_active)
        self.stop_mode_active = bool(self.stop_mode_active)
        self.prev_grad = _as_vector(self.prev_grad, "robot.prev_grad")
        self.prev_direction = _as_vector(self.prev_direction, "robot.prev_direction")
        self.interaction_force_memory = _as_vector(
            self.interaction_force_memory, "robot.interaction_force_memory"
        )
        self.committed_side_direction = _as_vector(
            self.committed_side_direction, "robot.committed_side_direction"
        )
        self.escape_direction_memory = _as_vector(
            self.escape_direction_memory, "robot.escape_direction_memory"
        )
        self.detour_direction = _as_vector(self.detour_direction, "robot.detour_direction")
        self.detour_anchor_position = _as_vector(self.detour_anchor_position, "robot.detour_anchor_position")
        self.current_branch_point = _as_vector(self.current_branch_point, "robot.current_branch_point")
        self.current_branch_guide_direction = _as_vector(
            self.current_branch_guide_direction,
            "robot.current_branch_guide_direction",
        )
        self.detour_lock_direction = (
            _as_vector(self.detour_lock_direction, "robot.detour_lock_direction")
            if self.detour_lock_direction is not None
            else None
        )
        self.detour_lock_steps = max(int(self.detour_lock_steps), 0)
        self.detour_side = int(np.clip(np.sign(float(self.detour_side)), -1, 1))
        self.prev_goal_dist = float(self.prev_goal_dist)
        self.last_flip_step = int(self.last_flip_step)
        self.detour_side_flip_count = max(int(self.detour_side_flip_count), 0)
        self.failed_branches = [
            FailedBranch(
                point=_as_vector(branch.point, f"robot.failed_branches[{index}].point"),
                guide_direction=_as_vector(
                    branch.guide_direction,
                    f"robot.failed_branches[{index}].guide_direction",
                ),
                guide_normal=_as_vector(
                    branch.guide_normal,
                    f"robot.failed_branches[{index}].guide_normal",
                ),
                radius=_validate_positive(branch.radius, f"robot.failed_branches[{index}].radius"),
            )
            for index, branch in enumerate(self.failed_branches)
        ]
        self.branch_rebuild_required = bool(self.branch_rebuild_required)
        self.failed_detours = [
            _as_vector(direction, "robot.failed_detours")
            for direction in self.failed_detours
        ]
        self.last_detour_direction = (
            _as_vector(self.last_detour_direction, "robot.last_detour_direction")
            if self.last_detour_direction is not None
            else None
        )
        self.static_guide_waypoints = [
            _as_vector(waypoint, "robot.static_guide_waypoints") for waypoint in self.static_guide_waypoints
        ]
        self.static_guide_anchor_position = _as_vector(
            self.static_guide_anchor_position,
            "robot.static_guide_anchor_position",
        )
        self.static_guide_anchor_progress = _validate_nonnegative(
            self.static_guide_anchor_progress,
            "robot.static_guide_anchor_progress",
        )
        self.guide_progress_max = _validate_nonnegative(
            self.guide_progress_max,
            "robot.guide_progress_max",
        )
        self.guide_progress_regression_tolerance = _validate_nonnegative(
            self.guide_progress_regression_tolerance,
            "robot.guide_progress_regression_tolerance",
        )
        self.guide_progress_max = max(
            self.guide_progress_max,
            self.static_guide_anchor_progress,
        )
        self.reset_control_state()
        self.trail = [self.position.copy()]

    def reset_control_state(self) -> None:
        self.prev_grad = np.zeros(2, dtype=float)
        self.prev_direction = _normalize(self.velocity, self.goal - self.position)
        self.interaction_force_memory = np.zeros(2, dtype=float)
        self.committed_side_direction = np.zeros(2, dtype=float)
        self.escape_direction_memory = np.zeros(2, dtype=float)
        self.escape_commit_steps_remaining = 0
        self.escape_progress_window = deque(maxlen=10)
        self.goal_progress_window = deque(maxlen=10)
        self.goal_distance_window = deque(maxlen=24)
        self.escape_activation_steps = deque(maxlen=8)
        self.control_step_counter = 0
        self.escape_gain = 1.0
        self.static_guide_index = 0
        self.static_guide_anchor_position = self.position.copy()
        self.detour_active = False
        self.detour_direction[:] = 0.0
        self.detour_steps_remaining = 0
        self.detour_anchor_position = self.position.copy()
        self.detour_anchor_distance = float(np.linalg.norm(self.goal - self.position))
        self.detour_last_progress = 0.0
        self.current_branch_point = self.position.copy()
        self.current_branch_guide_direction[:] = 0.0
        self.detour_lock_direction = None
        self.detour_lock_steps = 0
        self.detour_side = 0
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.position))
        self.last_flip_step = -1000
        self.detour_side_flip_count = 0
        self.failed_branches = []
        self.branch_rebuild_required = False
        self.failed_detours = []
        self.last_detour_direction = None
        self.interaction_level_memory = 0.0
        self.dominant_human_memory_id = -1
        self.interaction_hold_timer = 0.0
        self.interaction_active_duration = 0.0
        self.interaction_rearm_timer = 0.0
        self.interaction_deadlock_timer = 0.0
        self.interaction_release_timer = 0.0
        self.interaction_release_direction[:] = 0.0
        self.interaction_release_projection_anchor = 0.0
        self.interaction_release_path_distance_anchor = 0.0
        self.interaction_release_anchor_position = self.position.copy()
        self.interaction_release_progress = 0.0
        self.interaction_release_required_progress = 0.0
        self.post_interaction_relief_timer = 0.0
        self.stop_resume_timer = 0.0
        self.stop_mode_reason = "none"
        self.post_pass_context = None
        self.interaction_mode_active = False
        self.stop_mode_active = False
        self.behavior_state = FSMState.GOAL_SEEK.value
        self.behavior_speed_scale = 1.0
        self.ttc_min = float("inf")
        self.clearance_min = float("inf")
        self.interaction_clearance = float("inf")
        self.global_clearance = float("inf")
        self.physical_clearance = float("inf")
        self.interaction_clearance_rate = 0.0
        self.interacting_human_id = -1
        self.previous_interaction_clearance = float("inf")
        self.previous_interacting_human_id = -1
        self.interaction_switch_count = 0
        self.per_human_min_clearances = []
        self.global_min_clearance = float("inf")
        self.minimum_physical_clearance = float("inf")
        self.previous_total_risk = 0.0
        self.previous_rule_clearance = float("inf")
        self.safety_margin = float("inf")
        self.risk_slope = 0.0
        self.latest_recovery_time = 0.0
        self.recovery_timer_active = False
        self.recovery_timer_elapsed = 0.0
        self.static_escape_duration = 0.0
        self.static_escape_cooldown = 0
        self.deadlock_timer = 0.0
        self.deadlock_recovery_timer = 0.0
        self.progress_stall_timer = 0.0
        self.stagnation_counter = 0
        self.previous_goal_distance = float(np.linalg.norm(self.goal - self.position))
        self.goal_distance_window.append(self.previous_goal_distance)
        self.static_guide_anchor_progress = 0.0
        self.guide_progress_max = 0.0
        self.top_1_interaction_strength = 0.0
        self.top_2_interaction_strength = 0.0
        self.interaction_strength_gap = 0.0
        self.multi_dominant_interaction = False
        self.human_interaction_distances = []
        self.human_interaction_ttc = []
        self.human_interaction_alignments = []
        self.human_interaction_scores = []
        self.interaction_level_current = 0.0
        self.interaction_level = 0.0
        self.interaction_hold_timer = 0.0
        self.interaction_active_duration = 0.0
        self.interaction_rearm_timer = 0.0
        self.interaction_deadlock_timer = 0.0
        self.interaction_release_timer = 0.0
        self.interaction_release_direction[:] = 0.0
        self.interaction_release_projection_anchor = 0.0
        self.interaction_release_path_distance_anchor = 0.0
        self.interaction_release_anchor_position = self.position.copy()
        self.interaction_release_progress = 0.0
        self.interaction_release_required_progress = 0.0
        self.post_interaction_relief_timer = 0.0
        self.active_interactions = []

    def _build_post_pass_context(
        self,
        interaction: TTCInteraction,
        humans: Sequence[Human],
        fallback_axis: np.ndarray,
    ) -> PostPassContext:
        human = humans[interaction.human_index]
        longitudinal_axis = _normalize(human.velocity, fallback_axis)
        lateral_axis = np.array([-longitudinal_axis[1], longitudinal_axis[0]], dtype=float)
        preferred_side = interaction.lateral_direction
        if np.linalg.norm(self.committed_side_direction) > 1e-9:
            preferred_side = self.committed_side_direction
        committed_side = float(np.sign(np.dot(preferred_side, lateral_axis)))
        if committed_side == 0.0:
            committed_side = 1.0
        side_direction = committed_side * lateral_axis
        relative_position = self.position - human.position
        clearance = max(np.linalg.norm(relative_position) - (self.radius + human.radius), 0.0)
        return PostPassContext(
            human_index=interaction.human_index,
            committed_side=committed_side,
            longitudinal_axis=longitudinal_axis.copy(),
            side_direction=side_direction.copy(),
            reference_relative_position=relative_position.copy(),
            last_relative_position=relative_position.copy(),
            previous_clearance=clearance,
        )

    def _compute_post_pass_force(
        self,
        humans: Sequence[Human],
    ) -> tuple[np.ndarray, bool, bool]:
        if self.post_pass_context is None:
            return np.zeros(2, dtype=float), False, False

        human_index = self.post_pass_context.human_index
        if human_index < 0 or human_index >= len(humans):
            self.post_pass_context = None
            return np.zeros(2, dtype=float), False, False

        context = self.post_pass_context
        human = humans[human_index]
        relative_position = self.position - human.position
        clearance = max(np.linalg.norm(relative_position) - (self.radius + human.radius), 0.0)
        longitudinal_now = float(np.dot(relative_position, context.longitudinal_axis))
        longitudinal_ref = float(np.dot(context.reference_relative_position, context.longitudinal_axis))
        passed_human = longitudinal_now * longitudinal_ref <= 0.0
        clearance_non_decreasing = clearance >= context.previous_clearance - 1e-6
        geometric_release = (
            passed_human
            and clearance > self.comfort_clearance
            and clearance_non_decreasing
        )

        context.last_relative_position = relative_position.copy()
        context.previous_clearance = clearance
        if geometric_release:
            self.post_pass_context = None
            return np.zeros(2, dtype=float), True, clearance_non_decreasing

        comfort_scale = max(self.comfort_clearance, 1e-3)
        post_force = self.side_commitment_gain * np.exp(-((clearance / comfort_scale) ** 2))
        return post_force * context.side_direction, False, clearance_non_decreasing

    def _compute_actual_clearance(self, humans: Sequence[Human]) -> float:
        if not humans:
            return float("inf")
        return float(
            min(
                max(np.linalg.norm(self.position - human.position) - (self.radius + human.radius), 0.0)
                for human in humans
            )
        )

    def _compute_human_clearance(self, humans: Sequence[Human], human_index: int) -> float:
        if human_index < 0 or human_index >= len(humans):
            return float("inf")
        human = humans[human_index]
        return max(np.linalg.norm(self.position - human.position) - (self.radius + human.radius), 0.0)

    def _compute_all_human_clearances(self, humans: Sequence[Human]) -> list[float]:
        return [self._compute_human_clearance(humans, index) for index in range(len(humans))]

    def _compute_min_physical_clearance_at(
        self,
        position: np.ndarray,
        humans: Sequence[Human],
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone] = (),
    ) -> float:
        min_clearance = float("inf")
        for human in humans:
            clearance = np.linalg.norm(position - human.position) - (self.radius + human.radius)
            min_clearance = min(min_clearance, clearance)
        for obstacle in obstacles:
            clearance = obstacle.distance_to_surface(position) - self.radius
            min_clearance = min(min_clearance, clearance)
        for zone in zones:
            clearance = zone.signed_distance(position) - self.radius
            min_clearance = min(min_clearance, clearance)
        return float(min_clearance)

    def _compute_min_static_clearance_at(
        self,
        position: np.ndarray,
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone] = (),
    ) -> float:
        return self._compute_min_physical_clearance_at(position, [], obstacles, zones)

    def _compute_environment_clearance_at(
        self,
        position: np.ndarray,
        world_size: np.ndarray,
        humans: Sequence[Human],
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone] = (),
    ) -> float:
        physical_clearance = self._compute_min_physical_clearance_at(
            position,
            humans,
            obstacles,
            zones,
        )
        boundary_clearance = _boundary_clearance(position, world_size, self.radius)
        return float(min(physical_clearance, boundary_clearance))

    def _static_boundary_normal(
        self,
        position: np.ndarray,
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone],
        fallback_direction: np.ndarray,
    ) -> np.ndarray:
        best_clearance = float("inf")
        best_normal = _normalize(fallback_direction, self.goal - self.position)
        for obstacle in obstacles:
            _, normal = obstacle.surface_projection(position, fallback_direction)
            clearance = obstacle.distance_to_surface(position) - self.radius
            if clearance < best_clearance:
                best_clearance = clearance
                best_normal = normal
        for zone in zones:
            _, normal = zone.boundary_projection(position, fallback_direction)
            clearance = zone.signed_distance(position) - self.radius
            if clearance < best_clearance:
                best_clearance = clearance
                best_normal = normal
        return _normalize(best_normal, fallback_direction)

    def _compute_safety_response(
        self,
        humans: Sequence[Human],
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone],
        fallback_direction: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        eps = 1e-6
        min_clearance = float("inf")
        speed_scale = 1.0
        close_push = np.zeros(2, dtype=float)
        envelope_buffer = max(self.safety_envelope_buffer, eps)

        def apply_response(clearance: float, away_dir: np.ndarray) -> None:
            nonlocal min_clearance, speed_scale, close_push

            min_clearance = min(min_clearance, clearance)
            envelope_clearance = clearance - self.safety_projection_margin
            if envelope_clearance < envelope_buffer:
                speed_scale = min(
                    speed_scale,
                    float(np.clip(envelope_clearance / envelope_buffer, 0.1, 1.0)),
                )
            if clearance >= self.hard_min_clearance:
                return

            d = max(clearance, 0.0)
            speed_scale = min(speed_scale, float(np.clip(d / self.hard_min_clearance, 0.0, 1.0)) ** 2)
            push_strength = min(self.close_avoidance_gain / (d + eps), 8.0)
            close_push += push_strength * away_dir

        for human in humans:
            delta = self.position - human.position
            away_dir = _normalize(delta, fallback_direction)
            clearance = float(np.linalg.norm(delta)) - (self.radius + human.radius)
            apply_response(clearance, away_dir)

        for obstacle in obstacles:
            _, away_dir = obstacle.surface_projection(self.position, fallback_direction)
            clearance = obstacle.distance_to_surface(self.position) - self.radius
            apply_response(clearance, away_dir)

        for zone in zones:
            _, away_dir = zone.boundary_projection(self.position, fallback_direction)
            clearance = zone.signed_distance(self.position) - self.radius
            apply_response(clearance, away_dir)

        return min_clearance, speed_scale, close_push

    def _project_safe_position(
        self,
        target_position: np.ndarray,
        world_size: np.ndarray,
        humans: Sequence[Human],
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone] = (),
        *,
        fallback_direction: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        safe_position = _clamp_position(_as_vector(target_position, "target_position"), world_size).copy()
        fallback = _normalize(
            self.prev_direction if fallback_direction is None else fallback_direction,
            self.goal - self.position,
        )
        margin = max(self.safety_projection_margin, self.safety_distance)
        obstacle_clearance = self.radius + margin
        target_clearance = margin
        start_clearance = self._compute_min_physical_clearance_at(
            self.position,
            humans,
            obstacles,
            zones,
        )

        for _ in range(8):
            adjusted = False

            for human in humans:
                safe_radius = human.radius + self.radius + margin
                delta = safe_position - human.position
                distance = float(np.linalg.norm(delta))
                if distance >= safe_radius - 1e-9:
                    continue
                away_dir = _normalize(delta, fallback)
                safe_position = human.position + away_dir * safe_radius
                adjusted = True

            for obstacle in obstacles:
                if obstacle.distance_to_surface(safe_position) >= obstacle_clearance - 1e-9:
                    continue
                safe_position = obstacle.project_point_outside(
                    safe_position,
                    obstacle_clearance,
                    fallback=fallback,
                )
                adjusted = True

            for zone in zones:
                if zone.signed_distance(safe_position) >= obstacle_clearance - 1e-9:
                    continue
                safe_position = zone.project_point_outside(
                    safe_position,
                    obstacle_clearance,
                    fallback=fallback,
                )
                adjusted = True

            clamped_position = _clamp_position(safe_position, world_size)
            if not np.allclose(clamped_position, safe_position, atol=1e-9):
                adjusted = True
            safe_position = clamped_position
            if not adjusted:
                break

        safe_clearance = self._compute_min_physical_clearance_at(
            safe_position,
            humans,
            obstacles,
            zones,
        )
        if safe_clearance < target_clearance - 1e-9 and start_clearance >= target_clearance - 1e-9:
            displacement = safe_position - self.position
            low = 0.0
            high = 1.0
            for _ in range(18):
                mid = 0.5 * (low + high)
                candidate = self.position + mid * displacement
                clearance = self._compute_min_physical_clearance_at(
                    candidate,
                    humans,
                    obstacles,
                    zones,
                )
                if clearance >= target_clearance - 1e-9:
                    low = mid
                else:
                    high = mid
            safe_position = self.position + low * displacement
            safe_clearance = self._compute_min_physical_clearance_at(
                safe_position,
                humans,
                obstacles,
                zones,
            )

        return safe_position, safe_clearance

    def _project_safe_velocity(
        self,
        velocity: np.ndarray,
        dt: float,
        world_size: np.ndarray,
        humans: Sequence[Human],
        obstacles: Sequence[Obstacle],
        zones: Sequence[NoGoZone] = (),
    ) -> tuple[np.ndarray, float]:
        predicted_position = self.position + velocity * dt
        safe_position, safe_clearance = self._project_safe_position(
            predicted_position,
            world_size,
            humans,
            obstacles,
            zones,
            fallback_direction=velocity,
        )
        safe_velocity = (safe_position - self.position) / dt
        return safe_velocity, safe_clearance

    def _update_per_human_min_clearances(self, current_clearances: Sequence[float]) -> None:
        if len(self.per_human_min_clearances) < len(current_clearances):
            self.per_human_min_clearances.extend(
                [float("inf")] * (len(current_clearances) - len(self.per_human_min_clearances))
            )
        for index, clearance in enumerate(current_clearances):
            self.per_human_min_clearances[index] = min(self.per_human_min_clearances[index], clearance)

    def _select_primary_interaction(
        self,
        active_interactions: Sequence[TTCInteraction],
    ) -> TTCInteraction | None:
        if not active_interactions:
            return None
        strongest = max(active_interactions, key=lambda interaction: interaction.strength)
        if self.interaction_mode_active and self.interacting_human_id >= 0:
            for interaction in active_interactions:
                if interaction.human_index == self.interacting_human_id:
                    return interaction
        if self.interacting_human_id >= 0:
            for interaction in active_interactions:
                if interaction.human_index != self.interacting_human_id:
                    continue
                strength_gap = strongest.strength - interaction.strength
                if strength_gap <= self.interaction_switch_margin * max(strongest.strength, 1e-6):
                    return interaction
                break
        return strongest

    def _compute_ttc_metrics(
        self, humans: Sequence[Human], robot_direction: np.ndarray
    ) -> tuple[float, float, float, float, List[TTCInteraction]]:
        ttc_min = float("inf")
        clearance_min = float("inf")
        critical_clearance = float("inf")
        critical_lateral_offset = float("inf")
        active_interactions: list[TTCInteraction] = []
        robot_direction = _normalize(robot_direction, self.goal - self.position)
        interaction_horizon = max(self.ttc_attention_threshold, self.ttc_resume, 5.0)
        predicted_speed = max(np.linalg.norm(self.velocity), self.speed * self.min_speed_scale, 1e-3)
        robot_prediction_velocity = robot_direction * predicted_speed
        goal_vector = self.goal - self.position
        heading_lateral = np.array([-robot_direction[1], robot_direction[0]], dtype=float)
        corridor_width = max(self.lateral_safe_distance, self.comfort_clearance + self.radius)
        sharpness = self.interaction_strength_sharpness
        safe_time_scale = max(self.ttc_attention_threshold, 1e-3)
        activation_distance = max(
            self.interaction_enter_clearance,
            2.0 * self.comfort_clearance,
            1.5 * self.safe_distance,
        )
        speed_floor_sq = max(self.ttc_human_speed_threshold, 1e-3) ** 2
        interaction_distances: list[float] = []
        interaction_ttc_values: list[float] = []
        interaction_alignments: list[float] = []
        interaction_scores: list[float] = []

        for index, human in enumerate(humans):
            rel_pos = human.position - self.position
            distance = float(np.linalg.norm(rel_pos))
            current_clearance = max(distance - (self.radius + human.radius), 0.0)
            if distance <= 1e-9:
                clearance = 0.0
                ttc = 0.0
                clearance_min = min(clearance_min, clearance)
                interaction_distances.append(0.0)
                interaction_ttc_values.append(0.0)
                interaction_alignments.append(1.0)
                interaction_scores.append(1.0)
                if ttc < ttc_min:
                    ttc_min = ttc
                    critical_clearance = clearance
                    critical_lateral_offset = 0.0
                continue

            rel_dir = rel_pos / distance
            forward_alignment = float(np.dot(rel_dir, robot_direction))
            projection = float(np.dot(rel_pos, robot_direction))
            lateral_vec = rel_pos - projection * robot_direction
            lateral_offset = float(np.linalg.norm(lateral_vec))
            rel_vel = human.velocity - robot_prediction_velocity
            rel_speed_sq = float(np.dot(rel_vel, rel_vel))
            rel_motion = float(np.dot(rel_pos, rel_vel))
            closing_speed = max(-rel_motion / (distance + 1e-6), 0.0)
            has_ttc = rel_motion < 0.0
            closest_time = (
                -rel_motion / (max(rel_speed_sq, speed_floor_sq) + 1e-6) if has_ttc else float("inf")
            )
            ttc = closest_time if has_ttc else float("inf")
            predicted_time = float(np.clip(ttc, 0.0, interaction_horizon)) if np.isfinite(ttc) else interaction_horizon
            closest_rel = rel_pos + predicted_time * rel_vel if np.isfinite(ttc) else rel_pos
            closest_distance = float(np.linalg.norm(closest_rel))
            predicted_clearance = max(closest_distance - (self.radius + human.radius), 0.0)
            predicted_projection = float(np.dot(closest_rel, robot_direction))
            predicted_lateral = closest_rel - predicted_projection * robot_direction
            predicted_lateral_offset = float(np.linalg.norm(predicted_lateral))
            clearance_min = min(clearance_min, predicted_clearance)
            within_activation_radius = current_clearance <= activation_distance
            within_ttc_horizon = np.isfinite(ttc) and ttc <= interaction_horizon
            if not within_activation_radius and not within_ttc_horizon:
                interaction_distances.append(current_clearance)
                interaction_ttc_values.append(ttc)
                interaction_alignments.append(forward_alignment)
                interaction_scores.append(0.0)
                continue

            forward_metric = max(
                forward_alignment,
                predicted_projection / (closest_distance + 1e-6) if closest_distance > 1e-9 else 1.0,
            )
            forward_score = 1.0 / (
                1.0 + np.exp(-sharpness * (forward_metric - self.interaction_forward_threshold))
            )
            ttc_score = (
                np.exp(-predicted_time / self.interaction_ttc_sigma) if within_ttc_horizon else 0.0
            )
            proximity_score = np.exp(-current_clearance / self.interaction_distance_sigma)
            activation_score = (
                np.exp(-current_clearance / activation_distance) if within_activation_radius else 0.0
            )
            alignment_weight = 0.45 + 0.55 * np.clip(0.5 * (forward_metric + 1.0), 0.0, 1.0)
            lateral_weight = 0.5 + 0.5 * np.exp(
                -0.5 * (min(lateral_offset, predicted_lateral_offset) / corridor_width) ** 2
            )
            clearance_weight = 0.5 + 0.5 * np.exp(
                -predicted_clearance / max(self.comfort_clearance, 1e-3)
            )
            ttc_interaction_score = float(
                np.sqrt(max(proximity_score * ttc_score, 0.0))
                * alignment_weight
                * lateral_weight
                * clearance_weight
            )
            distance_fallback_score = float(
                activation_score
                * (0.65 + 0.35 * proximity_score)
                * (0.4 + 0.6 * forward_score)
                * lateral_weight
                * clearance_weight
            )
            if within_ttc_horizon:
                interaction_strength = max(ttc_interaction_score, 0.55 * distance_fallback_score)
            else:
                interaction_strength = 0.55 * distance_fallback_score
            if within_activation_radius:
                interaction_strength = max(
                    interaction_strength,
                    self.interaction_min_strength + 0.04 * activation_score,
                )
            interaction_distances.append(current_clearance)
            interaction_ttc_values.append(ttc)
            interaction_alignments.append(forward_alignment)
            interaction_scores.append(interaction_strength)
            if interaction_strength < self.interaction_min_strength:
                continue

            side_sign = np.sign(robot_direction[0] * closest_rel[1] - robot_direction[1] * closest_rel[0])
            if side_sign == 0.0:
                side_sign = np.sign(
                    robot_direction[0] * human.velocity[1] - robot_direction[1] * human.velocity[0]
                )
            if side_sign == 0.0:
                side_sign = 1.0

            lateral_direction = -side_sign * heading_lateral
            lateral_direction = _normalize(lateral_direction, heading_lateral)
            if np.dot(lateral_direction, goal_vector) < -1e-6:
                lateral_direction *= -1.0

            interaction_vector = interaction_strength * lateral_direction
            active_interactions.append(
                TTCInteraction(
                    human_index=index,
                    ttc=predicted_time,
                    clearance=predicted_clearance,
                    lateral_offset=min(lateral_offset, predicted_lateral_offset),
                    forward_alignment=forward_alignment,
                    predicted_time=predicted_time,
                    strength=interaction_strength,
                    lateral_direction=lateral_direction,
                    interaction_vector=interaction_vector,
                )
            )
            if predicted_time < ttc_min:
                ttc_min = predicted_time
                critical_clearance = predicted_clearance
                critical_lateral_offset = min(lateral_offset, predicted_lateral_offset)

        self.human_interaction_distances = interaction_distances
        self.human_interaction_ttc = interaction_ttc_values
        self.human_interaction_alignments = interaction_alignments
        self.human_interaction_scores = interaction_scores
        return ttc_min, clearance_min, critical_clearance, critical_lateral_offset, active_interactions

    def _compute_human_approach_state(
        self,
        humans: Sequence[Human],
        heading_direction: np.ndarray,
    ) -> tuple[float, float, float, bool, float]:
        if not humans:
            return float("inf"), float("inf"), 0.0, False, 0.0

        human_positions = np.asarray([human.position for human in humans], dtype=float)
        human_velocities = np.asarray([human.velocity for human in humans], dtype=float)
        human_radii = np.asarray([human.radius for human in humans], dtype=float)
        rel_pos = human_positions - self.position
        distances = np.linalg.norm(rel_pos, axis=1)
        clearances = np.maximum(distances - (self.radius + human_radii), 0.0)
        rel_vel = human_velocities - self.velocity[None, :]
        closing_projection = np.einsum("ij,ij->i", rel_vel, rel_pos)
        approaching_mask = closing_projection < 0.0
        closing_speed = np.where(
            approaching_mask,
            -closing_projection / (distances + 1e-6),
            0.0,
        )
        heading = _normalize(heading_direction, self.goal - self.position)
        behind_mask = np.einsum("ij,j->i", rel_pos, heading) < 0.0

        nearest_human_clearance = float(np.min(clearances))
        if np.any(approaching_mask):
            approach_indices = np.flatnonzero(approaching_mask)
            nearest_approach_index = approach_indices[np.argmin(clearances[approach_indices])]
            nearest_approach_clearance = float(clearances[nearest_approach_index])
            nearest_approach_speed = float(closing_speed[nearest_approach_index])
        else:
            nearest_approach_clearance = float("inf")
            nearest_approach_speed = 0.0

        rear_approach_mask = approaching_mask & behind_mask
        rear_approaching = bool(np.any(rear_approach_mask))
        rear_approach_speed = (
            float(np.max(closing_speed[rear_approach_mask])) if rear_approaching else 0.0
        )
        return (
            nearest_human_clearance,
            nearest_approach_clearance,
            nearest_approach_speed,
            rear_approaching,
            rear_approach_speed,
        )

    def _enter_goal_state(self, goal_dir: np.ndarray) -> None:
        preserved_guide_progress = max(
            self.guide_progress_max,
            self.static_guide_anchor_progress,
        )
        self.position = self.goal.copy()
        self.velocity[:] = 0.0
        self.prev_grad[:] = 0.0
        self.prev_direction = goal_dir.copy()
        self.interaction_force_memory[:] = 0.0
        self.committed_side_direction[:] = 0.0
        self.escape_direction_memory[:] = 0.0
        self.escape_commit_steps_remaining = 0
        self.escape_progress_window.clear()
        self.goal_progress_window.clear()
        self.goal_distance_window.clear()
        self.goal_distance_window.append(0.0)
        self.escape_activation_steps.clear()
        self.escape_gain = 1.0
        self.control_step_counter = 0
        self.interaction_level_memory = 0.0
        self.dominant_human_memory_id = -1
        self.interaction_hold_timer = 0.0
        self.interaction_active_duration = 0.0
        self.interaction_rearm_timer = 0.0
        self.interaction_deadlock_timer = 0.0
        self.interaction_release_timer = 0.0
        self.interaction_release_direction[:] = 0.0
        self.interaction_release_projection_anchor = 0.0
        self.interaction_release_path_distance_anchor = 0.0
        self.interaction_release_anchor_position = self.position.copy()
        self.interaction_release_progress = 0.0
        self.interaction_release_required_progress = 0.0
        self.post_interaction_relief_timer = 0.0
        self.stop_resume_timer = 0.0
        self.post_pass_context = None
        self.stop_mode_active = False
        self.stop_mode_reason = "none"
        self.detour_active = False
        self.detour_direction[:] = 0.0
        self.detour_steps_remaining = 0
        self.detour_anchor_position = self.position.copy()
        self.detour_anchor_distance = 0.0
        self.detour_last_progress = 0.0
        self.current_branch_point = self.position.copy()
        self.current_branch_guide_direction[:] = 0.0
        self.detour_lock_direction = None
        self.detour_lock_steps = 0
        self.detour_side = 0
        self.prev_goal_dist = 0.0
        self.last_flip_step = -1000
        self.failed_branches = []
        self.branch_rebuild_required = False
        self.failed_detours = []
        self.last_detour_direction = None
        self.deadlock_timer = 0.0
        self.deadlock_recovery_timer = 0.0
        self.progress_stall_timer = 0.0
        self.stagnation_counter = 0
        self.previous_goal_distance = 0.0
        self.detour_side_flip_count = 0
        self.behavior_state = "goal"
        self.behavior_speed_scale = 0.0
        self.ttc_min = float("inf")
        self.clearance_min = float("inf")
        self.interaction_clearance = float("inf")
        self.global_clearance = float("inf")
        self.interaction_clearance_rate = 0.0
        self.interacting_human_id = -1
        self.previous_interaction_clearance = float("inf")
        self.previous_interacting_human_id = -1
        self.safety_margin = float("inf")
        self.risk_slope = 0.0
        self.top_1_interaction_strength = 0.0
        self.top_2_interaction_strength = 0.0
        self.interaction_strength_gap = 0.0
        self.multi_dominant_interaction = False
        self.human_interaction_distances = []
        self.human_interaction_ttc = []
        self.human_interaction_alignments = []
        self.human_interaction_scores = []
        self.interaction_level_current = 0.0
        self.interaction_level = 0.0
        self.latest_recovery_time = 0.0
        self.recovery_timer_active = False
        self.recovery_timer_elapsed = 0.0
        self.static_escape_duration = 0.0
        self.static_escape_cooldown = 0
        self.interaction_mode_active = False
        self.active_interactions = []
        self.stagnation_counter = 0
        self.static_guide_anchor_position = self.position.copy()
        self.static_guide_anchor_progress = preserved_guide_progress
        self.guide_progress_max = preserved_guide_progress

    def _clear_detour_mode(self) -> None:
        self.detour_active = False
        self.detour_direction[:] = 0.0
        self.detour_steps_remaining = 0
        self.detour_anchor_position = self.position.copy()
        self.detour_anchor_distance = float(np.linalg.norm(self.goal - self.position))
        self.detour_last_progress = 0.0
        self.detour_lock_direction = None
        self.detour_lock_steps = 0

    def _clear_interaction_release(self) -> None:
        self.interaction_release_timer = 0.0
        self.interaction_release_direction[:] = 0.0
        self.interaction_release_projection_anchor = 0.0
        self.interaction_release_path_distance_anchor = 0.0
        self.interaction_release_anchor_position = self.position.copy()
        self.interaction_release_progress = 0.0
        self.interaction_release_required_progress = 0.0

    def _remember_failed_branch(self, point: np.ndarray, guide_direction: np.ndarray, radius: float = 1.4) -> None:
        if not self.failed_branch_memory_enabled:
            return
        branch_point = _as_vector(point, "failed_branch.point").copy()
        branch_direction = _normalize(guide_direction, self.prev_direction)
        if np.linalg.norm(branch_direction) <= 1e-9:
            return
        branch_normal = _perpendicular(branch_direction)
        if np.linalg.norm(branch_normal) <= 1e-9:
            return
        branch_radius = float(np.clip(radius, 1.0, 1.8))
        for index, branch in enumerate(self.failed_branches):
            same_point = float(np.linalg.norm(branch.point - branch_point)) < 0.75 * branch_radius
            same_direction = float(
                np.dot(_normalize(branch.guide_direction, branch_direction), branch_direction)
            ) > 0.85
            if same_point and same_direction:
                self.failed_branches[index] = FailedBranch(
                    point=0.5 * (branch.point + branch_point),
                    guide_direction=branch_direction.copy(),
                    guide_normal=branch_normal.copy(),
                    radius=max(float(branch.radius), branch_radius),
                )
                return
        self.failed_branches.append(
            FailedBranch(
                point=branch_point,
                guide_direction=branch_direction.copy(),
                guide_normal=branch_normal.copy(),
                radius=branch_radius,
            )
        )
        if len(self.failed_branches) > 6:
            self.failed_branches = self.failed_branches[-6:]

    def _current_static_guide_target(self, *, advance: bool = True) -> np.ndarray:
        if not self.guide_planner_enabled:
            return self.goal.copy()
        if not self.static_guide_waypoints:
            return self.goal.copy()

        advance_radius = max(self.goal_tolerance, 0.35 * self.goal_stabilization_distance)
        while advance and self.static_guide_index < len(self.static_guide_waypoints):
            target = self.static_guide_waypoints[self.static_guide_index]
            if np.linalg.norm(target - self.position) <= advance_radius:
                self.static_guide_index += 1
            else:
                break

        if self.static_guide_index >= len(self.static_guide_waypoints):
            return self.goal.copy()

        return self.static_guide_waypoints[self.static_guide_index].copy()

    def _current_static_guide_direction(self, goal_dir: np.ndarray, *, advance: bool = True) -> np.ndarray:
        guide_fallback = _normalize(goal_dir, self.prev_direction)
        if not self.guide_planner_enabled:
            return guide_fallback
        target = self._current_static_guide_target(advance=advance)
        guide_direction = _normalize(target - self.position, guide_fallback)
        if np.linalg.norm(guide_direction) <= 1e-9:
            return guide_fallback
        return guide_direction

    def _guide_path_state_at(
        self,
        position: np.ndarray,
        goal_dir: np.ndarray,
    ) -> tuple[float, float, np.ndarray, float, np.ndarray]:
        fallback_direction = self._current_static_guide_direction(goal_dir, advance=False)
        guide_points: list[np.ndarray] = [self.static_guide_anchor_position.copy()]
        if self.static_guide_waypoints:
            guide_points.extend(waypoint.copy() for waypoint in self.static_guide_waypoints)
        else:
            guide_points.append(self.goal.copy())
        return _polyline_arc_progress(
            position,
            guide_points,
            fallback_direction=fallback_direction,
        )

    def _current_guide_path_state(
        self,
        goal_dir: np.ndarray,
    ) -> tuple[float, float, np.ndarray, float, np.ndarray]:
        return self._guide_path_state_at(self.position, goal_dir)

    def _current_global_guide_path_state(
        self,
        goal_dir: np.ndarray,
    ) -> tuple[float, float, np.ndarray, float, np.ndarray]:
        local_progress, local_total, path_tangent, path_distance, projection_point = (
            self._guide_path_state_at(self.position, goal_dir)
        )
        global_progress = self.static_guide_anchor_progress + local_progress
        global_total = self.static_guide_anchor_progress + local_total
        return (
            global_progress,
            global_total,
            path_tangent,
            path_distance,
            projection_point,
        )

    def _global_guide_path_state_at(
        self,
        position: np.ndarray,
        goal_dir: np.ndarray,
    ) -> tuple[float, float, np.ndarray, float, np.ndarray]:
        local_progress, local_total, path_tangent, path_distance, projection_point = (
            self._guide_path_state_at(position, goal_dir)
        )
        global_progress = self.static_guide_anchor_progress + local_progress
        global_total = self.static_guide_anchor_progress + local_total
        return (
            global_progress,
            global_total,
            path_tangent,
            path_distance,
            projection_point,
        )

    def _path_reentry_epsilon(self, dt: float) -> float:
        return max(
            0.5 * self.radius,
            self.speed * self.min_speed_scale * dt,
            1e-3,
        )

    def _rebuild_static_guide_from_current(
        self,
        *,
        world_size: np.ndarray,
        obstacles: Sequence["Obstacle"],
        zones: Sequence["NoGoZone"],
    ) -> None:
        if not self.guide_planner_enabled:
            current_global_progress = max(
                self.guide_progress_max,
                self.static_guide_anchor_progress,
            )
            self.static_guide_waypoints = []
            self.static_guide_index = 0
            self.static_guide_anchor_position = self.position.copy()
            self.static_guide_anchor_progress = current_global_progress
            self.guide_progress_max = max(self.guide_progress_max, current_global_progress)
            self.branch_rebuild_required = False
            return
        current_global_progress = max(
            self.guide_progress_max,
            self.static_guide_anchor_progress,
        )
        if self.static_guide_waypoints:
            current_goal_dir = _normalize(self.goal - self.position, self.prev_direction)
            current_local_progress, _, _, _, _ = self._current_guide_path_state(current_goal_dir)
            current_global_progress = max(
                current_global_progress,
                self.static_guide_anchor_progress + current_local_progress,
            )
        guide_clearance_levels = [
            self.radius + self.safety_distance + 0.5 * self.safety_hysteresis,
            self.radius + self.safety_distance,
        ]
        selected_waypoints: list[np.ndarray] | None = None
        for guide_clearance in guide_clearance_levels:
            candidate_waypoints = _build_static_guide_waypoints(
                start=self.position,
                goal=self.goal,
                world_size=world_size,
                obstacles=obstacles,
                zones=zones,
                clearance_threshold=guide_clearance,
                failed_branches=self.failed_branches,
            )
            path_cursor = self.position.copy()
            path_feasible = True
            for waypoint in candidate_waypoints:
                if not _segment_static_clear(
                    path_cursor,
                    waypoint,
                    world_size=world_size,
                    obstacles=obstacles,
                    zones=zones,
                    clearance_threshold=guide_clearance,
                    sample_spacing=0.1,
                    failed_branches=self.failed_branches,
                ):
                    path_feasible = False
                    break
                path_cursor = waypoint
            if path_feasible:
                selected_waypoints = candidate_waypoints
                break
        self.static_guide_waypoints = (
            selected_waypoints
            if selected_waypoints is not None
            else [self.goal.copy()]
        )
        self.static_guide_index = 0
        self.static_guide_anchor_position = self.position.copy()
        self.static_guide_anchor_progress = current_global_progress
        self.guide_progress_max = max(self.guide_progress_max, current_global_progress)

    def _current_guided_direction(self, goal_dir: np.ndarray) -> np.ndarray:
        static_guide_direction = self._current_static_guide_direction(
            goal_dir,
            advance=not self.detour_active,
        )
        if self.detour_lock_steps > 0 and self.detour_lock_direction is not None:
            locked_direction = _normalize(self.detour_lock_direction, static_guide_direction)
            if np.linalg.norm(locked_direction) > 1e-9:
                self.detour_lock_direction = locked_direction.copy()
                self.detour_lock_steps -= 1
                return locked_direction
            self.detour_lock_direction = None
            self.detour_lock_steps = 0
        elif self.detour_lock_steps <= 0:
            self.detour_lock_direction = None
        if (
            not self.detour_active
            or np.linalg.norm(self.detour_direction) <= 1e-9
        ):
            self._clear_detour_mode()
            return static_guide_direction
        if self.detour_steps_remaining <= 0:
            current_goal_distance = float(np.linalg.norm(self.goal - self.position))
            detour_progress = self.detour_anchor_distance - current_goal_distance
            self.detour_last_progress = float(detour_progress)
            if detour_progress < 0.1 and self.last_detour_direction is not None:
                self._remember_failed_branch(
                    self.current_branch_point,
                    self.current_branch_guide_direction,
                )
                self.branch_rebuild_required = True
                failed = _normalize(self.last_detour_direction, static_guide_direction)
                if np.linalg.norm(failed) > 1e-9:
                    self.failed_detours.append(failed.copy())
                    if len(self.failed_detours) > 3:
                        self.failed_detours = self.failed_detours[-3:]
            self._clear_detour_mode()
            return static_guide_direction

        guidance_direction = _normalize(self.detour_direction, static_guide_direction)
        if np.linalg.norm(guidance_direction) <= 1e-9:
            self._clear_detour_mode()
            return static_guide_direction
        return guidance_direction

    def _update_behavior_driven(
        self,
        dt: float,
        world_size: np.ndarray,
        risk_field: RiskField,
        goal_dir: np.ndarray,
        goal_distance: float,
    ) -> None:
        humans = risk_field.humans
        obstacles = risk_field.obstacles
        zones = risk_field.zones
        if self.branch_rebuild_required:
            self._rebuild_static_guide_from_current(
                world_size=world_size,
                obstacles=obstacles,
                zones=zones,
            )
            self.branch_rebuild_required = False
        previous_state = self.behavior_state
        previous_stop_reason = self.stop_mode_reason
        if previous_state != FSMState.STATIC_ESCAPE.value:
            self.detour_side = 0
        static_guide_target = self._current_static_guide_target()
        static_guide_dir = self._current_static_guide_direction(goal_dir)
        (
            guide_path_progress,
            guide_path_total_length,
            guide_path_tangent,
            guide_path_distance,
            guide_path_projection_point,
        ) = self._current_global_guide_path_state(goal_dir)
        path_reentry_epsilon = self._path_reentry_epsilon(dt)
        off_path_active = guide_path_distance > path_reentry_epsilon
        path_reentry_vector = guide_path_projection_point - self.position
        path_reentry_lateral = (
            path_reentry_vector
            - float(np.dot(path_reentry_vector, guide_path_tangent)) * guide_path_tangent
        )
        path_reentry_direction = _normalize(path_reentry_lateral, guide_path_tangent)
        previous_goal_distance = self.previous_goal_distance
        if self.static_escape_cooldown > 0:
            self.static_escape_cooldown -= 1
        if self.detour_steps_remaining > 0:
            self.detour_steps_remaining -= 1
        progress_delta = (
            previous_goal_distance - goal_distance
            if np.isfinite(previous_goal_distance)
            else 0.0
        )
        if previous_state == FSMState.STATIC_ESCAPE.value:
            self.escape_progress_window.append(float(progress_delta))
        else:
            self.escape_progress_window.clear()
        low_speed_threshold = 0.05
        deadlock_detection_time = 1.0
        deadlock_recovery_time = max(
            self.post_interaction_relief_time,
            2.0 * self.static_escape_lookahead_time,
        )
        progress_stall_time = 0.8
        progress_epsilon = max(1e-3, 0.05 * dt)
        self.control_step_counter += 1
        recent_avg_progress = (
            float(np.mean(np.asarray(self.goal_progress_window, dtype=float)))
            if self.goal_progress_window
            else progress_delta
        )
        recent_progress_stalled = (
            len(self.goal_progress_window) >= self.goal_progress_window.maxlen
            and recent_avg_progress < progress_epsilon
        )
        distance_window = self.goal_distance_window
        distance_array = np.asarray(distance_window, dtype=float) if distance_window else np.empty(0, dtype=float)
        window_progress_delta = (
            float(distance_array[0] - distance_array[-1])
            if distance_array.size >= 2
            else 0.0
        )
        window_progress_variance = (
            float(np.var(distance_array))
            if distance_array.size >= 2
            else 0.0
        )
        meaningful_progress = (
            distance_array.size >= distance_window.maxlen
            and window_progress_delta > self.progress_threshold
        )
        low_variance_plateau = window_progress_variance < self.progress_variance_threshold
        false_progress = (
            distance_array.size >= distance_window.maxlen
            and (
                window_progress_delta < self.progress_threshold
                or ((not meaningful_progress) and low_variance_plateau)
            )
        )
        best_recent_goal_distance = (
            float(np.min(distance_array))
            if distance_window
            else goal_distance
        )
        progress_regressing = (
            len(distance_window) >= distance_window.maxlen
            and goal_distance > best_recent_goal_distance + self.progress_regression_tolerance
        )
        escape_trigger_active = (
            false_progress
            or recent_progress_stalled
            or progress_regressing
        )
        guide_dir = self._current_guided_direction(goal_dir)
        previous_direction = _normalize(self.prev_direction, guide_dir)
        start_position = self.position.copy()
        settle_radius = max(2.0 * self.goal_tolerance, 0.5 * self.goal_stabilization_distance)
        goal_capture_radius = max(self.goal_tolerance, 0.6 * self.goal_stabilization_distance)

        current_human_clearances = self._compute_all_human_clearances(humans)
        self._update_per_human_min_clearances(current_human_clearances)
        self.global_clearance = min(current_human_clearances) if current_human_clearances else float("inf")
        self.global_min_clearance = min(self.global_min_clearance, self.global_clearance)

        reference_speed = max(np.linalg.norm(self.velocity), self.speed * self.min_speed_scale, 1e-3)
        reference_velocity = (
            self.velocity.copy()
            if np.linalg.norm(self.velocity) > 1e-9
            else guide_dir * reference_speed
        )
        static_activation_radius = max(
            self.safe_distance + 0.25,
            self.safety_distance + self.safety_envelope_buffer + 0.25,
        )
        static_gradient_decay_sigma = max(
            0.75 * static_activation_radius,
            self.safety_distance,
        )
        risk_field.set_robot_state(self.position, reference_speed)
        static_profile = sample_static_risk_profile(
            risk_field,
            self.position,
            reference_velocity,
            lookahead_time=self.static_escape_lookahead_time,
            sample_count=self.static_escape_lookahead_samples,
            activation_radius=static_activation_radius,
            gradient_decay_sigma=static_gradient_decay_sigma,
        )
        current_risk = static_profile.current_risk + risk_field.compute_dynamic_risk(self.position)
        self.prev_grad = risk_field.compute_static_gradient(self.position)
        self.risk_slope = static_profile.risk_slope
        detour_lock_release_safe_threshold = float(
            max(0.65 * self.risk_threshold, self.grad_epsilon)
        )
        if self.detour_lock_steps > 0 and self.detour_lock_direction is not None:
            lock_direction = _normalize(self.detour_lock_direction, guide_dir)
            if (
                current_risk < detour_lock_release_safe_threshold
                and float(np.dot(lock_direction, goal_dir)) > 0.8
            ):
                self.detour_lock_steps = 0
                self.detour_lock_direction = None
                guide_dir = self._current_guided_direction(goal_dir)
                previous_direction = _normalize(self.prev_direction, guide_dir)

        current_boundary_clearance = _boundary_clearance(self.position, world_size, self.radius)
        current_static_clearance = min(
            self._compute_min_static_clearance_at(
                self.position,
                obstacles,
                zones,
            ),
            current_boundary_clearance,
        )
        static_release_clearance = self.safe_clearance + 0.05
        current_physical_clearance = self._compute_environment_clearance_at(
            self.position,
            world_size,
            humans,
            obstacles,
            zones,
        )
        force_recovery = False
        if current_static_clearance < self.safety_distance - 1e-6:
            recovery_gradient = risk_field.compute_static_gradient(self.position)
            recovery_direction = _normalize(-recovery_gradient, goal_dir)
            guide_dir = recovery_direction
            previous_direction = _normalize(self.prev_direction, guide_dir)
            force_recovery = True
        self.physical_clearance = current_physical_clearance
        self.minimum_physical_clearance = min(
            self.minimum_physical_clearance,
            current_physical_clearance,
        )
        previous_interaction_current = self.interaction_level_current
        stagnation_steps_threshold = 10
        stagnation_active = self.stagnation_counter >= stagnation_steps_threshold
        effective_min_forward_dot = (
            -0.2
            if (
                false_progress
                or recent_progress_stalled
                or stagnation_active
                or progress_regressing
            )
            else self.min_forward_dot
        )

        previous_human_hard_stop = (
            previous_state == FSMState.HARD_STOP.value
            and previous_stop_reason in {"human_safety", "escape_blocked"}
        )
        previous_human_interaction = previous_human_hard_stop or previous_state == FSMState.HUMAN_YIELD.value
        previous_human_active = previous_human_hard_stop or previous_state == FSMState.HUMAN_YIELD.value
        interaction_persistence_active = self.interaction_memory_enabled and self.interaction_hold_timer > 1e-9
        interaction_release_active = (
            (
                self.interaction_release_timer > 1e-9
                or self.interaction_release_required_progress > self.interaction_release_progress + 1e-6
            )
            and np.linalg.norm(self.interaction_release_direction) > 1e-9
        )
        release_reference_dir = _normalize(
            path_reentry_direction if off_path_active else guide_path_tangent,
            guide_dir,
        )
        if interaction_release_active and (
            current_static_clearance < self.safety_distance - 1e-6
            or current_physical_clearance < self.safety_distance - 1e-6
        ):
            self._clear_interaction_release()
            interaction_release_active = False
        if interaction_release_active:
            if np.linalg.norm(release_reference_dir) <= 1e-9:
                self._clear_interaction_release()
                interaction_release_active = False
            else:
                self.interaction_release_direction = release_reference_dir.copy()
        dominant_memory_active = (
            self.interaction_memory_enabled
            and
            self.interaction_rearm_timer > 1e-9
            and self.dominant_human_memory_id >= 0
            and self.interaction_level_current >= 0.2
        )
        previous_interaction_duration = self.interaction_active_duration
        previous_dominant_index = (
            self.dominant_human_memory_id
            if dominant_memory_active
            else self.interacting_human_id
        )
        effective_goal_dir = (
            release_reference_dir.copy()
            if interaction_release_active
            else (guide_dir if (self.detour_active or force_recovery) else goal_dir)
        )
        human_decision = evaluate_human_speed_control(
            position=self.position,
            velocity=self.velocity,
            goal_dir=effective_goal_dir,
            humans=humans,
            robot_radius=self.radius,
            previous_human_active=previous_human_active,
            previous_human_hard_stop=previous_human_hard_stop,
            min_speed_scale=self.min_speed_scale,
            stop_enter_clearance=self.safety_distance,
            stop_exit_clearance=self.safety_distance + self.safety_hysteresis,
            yield_enter_clearance=max(self.safe_clearance, self.safety_distance + self.safety_hysteresis),
            yield_exit_clearance=max(self.safe_clearance, self.safety_distance + 2.0 * self.safety_hysteresis),
            hard_stop_ttc_enter=self.ttc_stop,
            hard_stop_ttc_exit=self.ttc_resume,
            yield_ttc_enter=max(self.ttc_stop + 0.35, 0.85 * self.ttc_mid),
            yield_ttc_exit=max(self.ttc_resume, 0.9 * self.ttc_mid),
            interaction_speed_threshold=self.ttc_human_speed_threshold,
            interaction_enter_threshold=self.yield_score_enter,
            interaction_exit_threshold=self.yield_score_exit,
            relevance_enter_clearance=self.interaction_enter_clearance,
            relevance_exit_clearance=self.interaction_exit_clearance,
            relevance_gain=self.speed_interaction_gain,
            interaction_beta=self.interaction_beta,
            previous_interaction_level=self.interaction_level_memory,
            previous_dominant_index=previous_dominant_index,
            dominant_memory_active=dominant_memory_active,
            interaction_smoothing_alpha=self.interaction_signal_alpha,
            interaction_current_blend=self.interaction_current_blend,
            interaction_memory_blend=self.interaction_memory_blend,
            interaction_memory_gain=self.interaction_memory_gain * self.interaction_force_memory_gain,
            interaction_memory_floor_ratio=self.interaction_memory_floor_ratio,
            interaction_decay_rate=self.interaction_decay_rate * self.interaction_force_memory_decay,
            interaction_fast_decay=self.interaction_fast_decay * self.interaction_force_memory_decay,
            interaction_effective_cap=self.interaction_effective_cap,
            interaction_min_strength=self.interaction_min_strength,
            interaction_max_active_humans=self.interaction_max_active_humans,
            interaction_memory_enabled=self.interaction_memory_enabled,
            topk_filter=self.topk_filter,
            weak_suppression=self.weak_suppression,
            top2_gap_threshold=self.interaction_conflict_margin,
            dominant_strength_floor=self.yield_score_exit,
            persistence_active=interaction_persistence_active,
            persistence_clearance_margin=0.1 + 0.05 * self.interaction_persistence_tau,
            persistence_ttc_margin=0.15 + 0.1 * self.interaction_persistence_tau,
        )
        self.human_interaction_scores = human_decision.interaction_strengths
        self.human_interaction_distances = human_decision.interaction_distances
        self.human_interaction_ttc = human_decision.interaction_ttc
        self.human_interaction_alignments = human_decision.interaction_alignments
        sorted_strengths = sorted(human_decision.interaction_strengths, reverse=True)
        self.top_1_interaction_strength = sorted_strengths[0] if sorted_strengths else 0.0
        self.top_2_interaction_strength = sorted_strengths[1] if len(sorted_strengths) >= 2 else 0.0
        self.interaction_strength_gap = (
            self.top_1_interaction_strength - self.top_2_interaction_strength
            if self.top_1_interaction_strength > 0.0
            else 0.0
        )
        self.multi_dominant_interaction = (
            len(sorted_strengths) >= 2
            and self.interaction_strength_gap < self.interaction_conflict_margin
        )
        tracked_human_index = human_decision.tracked_index
        tracked_interaction_clearance = human_decision.tracked_clearance
        self.ttc_min = human_decision.tracked_ttc
        low_motion_without_humans = (
            np.linalg.norm(self.velocity) < low_speed_threshold
            and human_decision.combined_interaction < 0.1
            and goal_distance > self.goal_tolerance
        )
        if low_motion_without_humans:
            self.deadlock_timer += dt
        else:
            self.deadlock_timer = max(self.deadlock_timer - 2.0 * dt, 0.0)
        if (
            goal_distance > self.goal_tolerance
            and human_decision.combined_interaction < 0.1
            and recent_progress_stalled
        ):
            self.progress_stall_timer += dt
        else:
            self.progress_stall_timer = max(self.progress_stall_timer - dt, 0.0)
        if (
            self.deadlock_recovery_timer > 1e-9
            and human_decision.combined_interaction < 0.1
            and current_static_clearance < self.safe_clearance
        ):
            self.deadlock_recovery_timer = deadlock_recovery_time
        else:
            self.deadlock_recovery_timer = max(self.deadlock_recovery_timer - dt, 0.0)
        if (
            self.deadlock_recovery_timer <= 1e-9
            and (
                self.deadlock_timer >= deadlock_detection_time
                or recent_progress_stalled
            )
        ):
            self.deadlock_recovery_timer = deadlock_recovery_time
        deadlock_recovery_active = self.deadlock_recovery_timer > 1e-9

        if (
            self.interacting_human_id >= 0
            and tracked_human_index >= 0
            and tracked_human_index != self.interacting_human_id
        ):
            self.interaction_switch_count += 1

        active_human_interaction = (
            human_decision.active_indices
            and human_decision.interaction_level >= self.yield_score_exit
        )
        if active_human_interaction:
            self.active_interactions = []
            for index in human_decision.active_indices:
                human_position = humans[index].position
                forward_alignment = float(
                    np.dot(_normalize(human_position - self.position, effective_goal_dir), effective_goal_dir)
                )
                self.active_interactions.append(
                    TTCInteraction(
                        human_index=index,
                        ttc=human_decision.interaction_ttc[index],
                        clearance=human_decision.interaction_distances[index],
                        lateral_offset=0.0,
                        forward_alignment=forward_alignment,
                        predicted_time=human_decision.interaction_ttc[index],
                        strength=human_decision.interaction_strengths[index],
                        lateral_direction=np.zeros(2, dtype=float),
                        interaction_vector=np.zeros(2, dtype=float),
                    )
                )
            self.interacting_human_id = tracked_human_index
            self.interaction_clearance = tracked_interaction_clearance
        else:
            self.active_interactions = []
            self.interacting_human_id = -1
            self.interaction_clearance = float("inf")

        if (
            self.interacting_human_id >= 0
            and self.interacting_human_id == self.previous_interacting_human_id
            and np.isfinite(self.interaction_clearance)
            and np.isfinite(self.previous_interaction_clearance)
        ):
            self.interaction_clearance_rate = (
                self.interaction_clearance - self.previous_interaction_clearance
            ) / dt
        else:
            self.interaction_clearance_rate = 0.0
        if self.interacting_human_id >= 0 and np.isfinite(self.interaction_clearance):
            self.previous_interacting_human_id = self.interacting_human_id
            self.previous_interaction_clearance = self.interaction_clearance
        else:
            self.previous_interacting_human_id = -1
            self.previous_interaction_clearance = float("inf")

        if (
            tracked_human_index >= 0
            and self.interaction_memory_enabled
            and human_decision.interaction_level >= self.yield_score_exit
            and human_decision.combined_interaction >= max(self.yield_score_exit, 0.2)
        ):
            self.dominant_human_memory_id = tracked_human_index
            self.interaction_rearm_timer = self.interaction_rearm_time
        else:
            self.interaction_rearm_timer = max(self.interaction_rearm_timer - dt, 0.0)
            if self.interaction_rearm_timer <= 1e-9:
                self.dominant_human_memory_id = -1

        self.interaction_level_current = human_decision.combined_interaction
        self.interaction_level = human_decision.interaction_level
        self.interaction_level_memory = human_decision.interaction_level
        if self.interaction_level_current < self.interaction_deadband:
            self.interaction_level_current = 0.0
        if self.interaction_level < self.interaction_deadband:
            self.interaction_level = 0.0
        if self.interaction_level_memory < self.interaction_deadband:
            self.interaction_level_memory = 0.0
        self.interaction_mode_active = human_decision.state in {FSMState.HUMAN_YIELD, FSMState.HARD_STOP}
        self.interaction_force_memory[:] = 0.0
        self.committed_side_direction[:] = 0.0
        self.post_pass_context = None

        static_escape_decision = evaluate_static_escape(
            profile=static_profile,
            goal_dir=guide_dir,
            current_static_clearance=current_static_clearance,
            release_clearance=static_release_clearance,
            previous_state=previous_state,
            previous_goal_distance=previous_goal_distance,
            current_goal_distance=goal_distance,
            escape_cooldown_steps_remaining=self.static_escape_cooldown,
            escape_progress_window=self.escape_progress_window,
            activation_threshold=self.risk_threshold,
            exit_threshold=max(0.65 * self.risk_threshold, self.grad_epsilon),
            risk_slope_threshold=0.8 * self.risk_threshold / max(self.static_escape_lookahead_time, 1e-6),
            activation_radius=static_activation_radius,
            min_forward_dot=effective_min_forward_dot,
            goal_regularization=max(
                effective_min_forward_dot,
                0.1
                + 0.2
                * float(
                    np.clip(
                        (current_static_clearance - self.safety_distance)
                        / max(self.safe_clearance - self.safety_distance, 1e-6),
                        0.0,
                        1.0,
                    )
                ),
            ),
            goal_seek_weight=self.goal_seek_weight,
            escape_direction_memory=self.escape_direction_memory,
            escape_commit_steps_remaining=self.escape_commit_steps_remaining,
        )
        suppress_static_escape = (
            current_static_clearance >= static_release_clearance - 1e-6
            and human_decision.combined_interaction < 0.1
            and progress_delta > progress_epsilon
        )
        if suppress_static_escape:
            static_escape_decision = evaluate_static_escape(
                profile=static_profile,
                goal_dir=guide_dir,
                current_static_clearance=current_static_clearance,
                release_clearance=static_release_clearance,
                previous_state=FSMState.GOAL_SEEK.value,
                previous_goal_distance=previous_goal_distance,
                current_goal_distance=goal_distance,
                escape_cooldown_steps_remaining=self.static_escape_cooldown,
                escape_progress_window=self.escape_progress_window,
                activation_threshold=float("inf"),
                exit_threshold=max(0.65 * self.risk_threshold, self.grad_epsilon),
                risk_slope_threshold=float("inf"),
                activation_radius=static_activation_radius,
                min_forward_dot=effective_min_forward_dot,
                goal_regularization=max(
                    effective_min_forward_dot,
                    0.1
                    + 0.2
                    * float(
                        np.clip(
                            (current_static_clearance - self.safety_distance)
                            / max(self.safe_clearance - self.safety_distance, 1e-6),
                            0.0,
                            1.0,
                        )
                    ),
                ),
                goal_seek_weight=self.goal_seek_weight,
                escape_direction_memory=np.zeros(2, dtype=float),
                escape_commit_steps_remaining=0,
            )
        self.escape_direction_memory = static_escape_decision.escape_direction_memory.copy()
        self.escape_commit_steps_remaining = static_escape_decision.escape_commit_steps_remaining
        self.static_escape_cooldown = max(
            self.static_escape_cooldown,
            static_escape_decision.cooldown_steps,
        )

        blocked_escape_by_human = False
        if static_escape_decision.active and tracked_human_index >= 0 and human_decision.state is not None:
            human_direction = _normalize(
                humans[tracked_human_index].position - self.position,
                static_escape_decision.direction,
            )
            blocked_escape_by_human = (
                float(np.dot(human_direction, static_escape_decision.direction)) > self.interaction_forward_threshold
                and tracked_interaction_clearance < max(self.safe_clearance, self.safety_distance + self.safety_hysteresis)
            )

        static_stop_threshold = self.safety_distance
        if previous_state == FSMState.HARD_STOP.value and previous_stop_reason == "static_safety":
            static_stop_threshold += self.safety_hysteresis
        hard_static_violation = current_static_clearance < (static_stop_threshold - 1e-6)
        hard_stop_active = hard_static_violation or human_decision.state == FSMState.HARD_STOP or blocked_escape_by_human
        next_state = resolve_fsm_state(
            hard_stop_active=hard_stop_active,
            human_yield_active=human_decision.state == FSMState.HUMAN_YIELD,
            static_escape_active=static_escape_decision.active,
        )

        next_stop_reason = "none"
        if next_state == FSMState.HARD_STOP:
            if blocked_escape_by_human:
                next_stop_reason = "escape_blocked"
            elif human_decision.state == FSMState.HARD_STOP:
                next_stop_reason = "human_safety"
            else:
                next_stop_reason = "static_safety"

        release_clearance_threshold = (
            max(self.safe_clearance, self.safety_distance + 2.0 * self.safety_hysteresis) + 0.2
        )
        release_ttc_threshold = max(self.ttc_resume + 0.2, 1.6)
        release_clearance_value = (
            tracked_interaction_clearance
            if np.isfinite(tracked_interaction_clearance)
            else human_decision.nearest_clearance
        )
        human_release_safe = (
            release_clearance_value > release_clearance_threshold
            or (
                np.isfinite(human_decision.tracked_ttc)
                and human_decision.tracked_ttc > release_ttc_threshold
            )
            or human_decision.combined_interaction < 0.1
        )
        interaction_hard_reset = (
            not human_decision.active_indices
            and human_decision.combined_interaction <= self.interaction_recovery_threshold
            and (
                not np.isfinite(human_decision.tracked_ttc)
                or human_decision.tracked_ttc > release_ttc_threshold
            )
            and human_decision.nearest_clearance > max(
                release_clearance_threshold,
                self.interaction_reset_clearance,
            )
        )
        interaction_risk_not_worsening = (
            self.interaction_clearance_rate >= -0.02
            or human_decision.combined_interaction <= previous_interaction_current + 0.03
        )
        max_interaction_hold_time = 3.0
        force_gradual_release = previous_interaction_duration > max_interaction_hold_time and (
            interaction_risk_not_worsening
            and (
                human_release_safe
                or human_decision.combined_interaction < max(self.yield_score_enter, 0.35)
            )
        )
        hard_stop_clearance_floor = self.safety_distance - 0.5 * self.safety_hysteresis
        hard_stop_imminent = (
            next_state == FSMState.HARD_STOP
            and (
                human_decision.hard_stop_imminent
                or (
                    next_stop_reason in {"human_safety", "escape_blocked"}
                    and np.isfinite(human_decision.tracked_ttc)
                    and human_decision.tracked_ttc <= self.ttc_stop + 1e-6
                )
                or (
                    next_stop_reason == "static_safety"
                    and current_static_clearance < (self.safety_distance - 1e-6)
                )
            )
        )
        interaction_deadlock_detection_time = max(deadlock_detection_time, 2.0 * self.interaction_min_hold_time)
        low_effective_human_progress = (
            np.linalg.norm(self.velocity) < low_speed_threshold
            or progress_delta < progress_epsilon
        )
        safe_human_deadlock = (
            goal_distance > self.goal_tolerance
            and low_effective_human_progress
            and human_decision.combined_interaction >= self.interaction_recovery_threshold
            and (
                next_state in {FSMState.HUMAN_YIELD, FSMState.HARD_STOP}
                or previous_human_interaction
            )
            and not hard_stop_imminent
            and current_static_clearance >= self.safety_distance - 1e-6
            and current_physical_clearance >= self.safety_distance - 1e-6
            and (
                not np.isfinite(human_decision.tracked_ttc)
                or human_decision.tracked_ttc > release_ttc_threshold
            )
        )
        if safe_human_deadlock:
            self.interaction_deadlock_timer += dt
        else:
            self.interaction_deadlock_timer = max(self.interaction_deadlock_timer - dt, 0.0)
        interaction_deadlock_active = self.interaction_deadlock_timer >= interaction_deadlock_detection_time
        if (
            interaction_release_active
            and (
                current_static_clearance < self.safety_distance - 1e-6
                or current_physical_clearance < self.safety_distance - 1e-6
                or hard_stop_imminent
            )
        ):
            self._clear_interaction_release()
            interaction_release_active = False
        escape_trigger_active = escape_trigger_active or interaction_deadlock_active
        progress_mode_clearance = min(current_static_clearance, current_physical_clearance)
        progress_mode_entry_clearance = (
            self.safety_distance + self.safety_hysteresis
            if humans
            else self.safety_distance + 1e-4
        )
        constraint_mode_active = (
            progress_mode_clearance < (progress_mode_entry_clearance - 1e-6)
            or hard_stop_imminent
        )
        safe_progress_override = (
            goal_distance > self.goal_tolerance
            and not constraint_mode_active
        )
        if safe_progress_override:
            static_escape_decision = StaticEscapeDecision(
                active=False,
                direction=guide_dir.copy(),
                escape_direction_memory=np.zeros(2, dtype=float),
                escape_commit_steps_remaining=0,
                cooldown_steps=8,
            )
            self.escape_direction_memory[:] = 0.0
            self.escape_commit_steps_remaining = 0
            self.static_escape_cooldown = max(self.static_escape_cooldown, 8)
            self.deadlock_recovery_timer = 0.0
            self.static_escape_duration = 0.0
            next_state = FSMState.GOAL_SEEK
            next_stop_reason = "none"
        if (
            escape_trigger_active
            and goal_distance > self.goal_tolerance
            and current_static_clearance >= self.safety_distance - 1e-6
            and not hard_stop_imminent
        ):
            next_state = FSMState.STATIC_ESCAPE
            next_stop_reason = "none"
        if (
            next_state == FSMState.HARD_STOP
            and next_stop_reason == "static_safety"
            and not hard_stop_imminent
        ):
            next_state = FSMState.STATIC_ESCAPE
            next_stop_reason = "none"
        if (
            next_state == FSMState.HARD_STOP
            and next_stop_reason == "human_safety"
            and not hard_stop_imminent
            and (
                not np.isfinite(human_decision.tracked_ttc)
                or human_decision.tracked_ttc > self.ttc_stop
            )
        ):
            next_state = FSMState.HUMAN_YIELD
            next_stop_reason = "none"
        if (
            interaction_release_active
            and goal_distance > self.goal_tolerance
            and current_static_clearance >= self.safety_distance - 1e-6
            and current_physical_clearance >= self.safety_distance - 1e-6
            and not hard_stop_imminent
        ):
            next_state = FSMState.STATIC_ESCAPE if self.detour_active else FSMState.GOAL_SEEK
            next_stop_reason = "none"
        if (
            deadlock_recovery_active
            and not safe_progress_override
            and self.static_escape_cooldown <= 0
            and human_decision.combined_interaction < 0.1
            and next_state in {FSMState.GOAL_SEEK, FSMState.HARD_STOP}
            and not hard_stop_imminent
            and (
                current_static_clearance < static_release_clearance - 1e-6
                or static_escape_decision.active
            )
        ):
            next_state = FSMState.STATIC_ESCAPE
            next_stop_reason = "none"
        if (
            previous_state == FSMState.STATIC_ESCAPE.value
            and self.stagnation_counter >= stagnation_steps_threshold
            and human_decision.combined_interaction < 0.1
            and current_static_clearance >= self.safe_clearance
        ):
            next_state = FSMState.GOAL_SEEK
            next_stop_reason = "none"
        if (
            previous_state == FSMState.HARD_STOP.value
            and self.stop_resume_timer > 1e-9
            and next_state != FSMState.HARD_STOP
            and hard_stop_imminent
            and not interaction_release_active
        ):
            next_state = FSMState.HARD_STOP
            next_stop_reason = previous_stop_reason if previous_stop_reason != "none" else "human_safety"
        elif (
            next_state == FSMState.HARD_STOP
            and next_stop_reason == "human_safety"
            and force_gradual_release
        ):
            next_state = (
                FSMState.HUMAN_YIELD
                if human_decision.combined_interaction >= 0.1
                else (FSMState.STATIC_ESCAPE if static_escape_decision.active else FSMState.GOAL_SEEK)
            )
            next_stop_reason = "none"
        elif (
            previous_state == FSMState.HUMAN_YIELD.value
            and self.interaction_hold_timer > 1e-9
            and next_state in {FSMState.GOAL_SEEK, FSMState.STATIC_ESCAPE}
            and human_decision.combined_interaction >= 0.1
            and not force_gradual_release
        ):
            next_state = FSMState.HUMAN_YIELD
            next_stop_reason = "none"
        elif (
            next_state == FSMState.HUMAN_YIELD
            and force_gradual_release
            and human_decision.combined_interaction < self.yield_score_exit
        ):
            next_state = FSMState.STATIC_ESCAPE if static_escape_decision.active else FSMState.GOAL_SEEK
            next_stop_reason = "none"
        if next_state == FSMState.STATIC_ESCAPE:
            if previous_state == FSMState.STATIC_ESCAPE.value:
                self.static_escape_duration += dt
            else:
                self.static_escape_duration = dt
        else:
            self.static_escape_duration = 0.0
        escape_timeout_active = (
            next_state == FSMState.STATIC_ESCAPE
            and self.static_escape_duration >= max(3.0, 6.0 * self.static_escape_lookahead_time)
            and human_decision.combined_interaction < 0.1
        )
        if (
            escape_timeout_active
            and current_static_clearance >= self.safety_distance - 1e-6
            and static_profile.future_max_risk < max(self.risk_threshold, self.grad_epsilon)
        ):
            next_state = FSMState.GOAL_SEEK
            next_stop_reason = "none"
            self.static_escape_duration = 0.0
        if force_recovery:
            next_state = FSMState.GOAL_SEEK
            next_stop_reason = "none"
            hard_stop_imminent = False
            static_escape_decision = StaticEscapeDecision(
                active=False,
                direction=guide_dir.copy(),
                escape_direction_memory=np.zeros(2, dtype=float),
                escape_commit_steps_remaining=0,
                cooldown_steps=0,
            )
            self.escape_direction_memory[:] = 0.0
            self.escape_commit_steps_remaining = 0
            self.stop_resume_timer = 0.0
            self.static_escape_duration = 0.0

        self.stop_mode_reason = next_stop_reason
        self.stop_mode_active = next_state == FSMState.HARD_STOP
        human_interaction_state = next_state == FSMState.HUMAN_YIELD or (
            next_state == FSMState.HARD_STOP
            and self.stop_mode_reason in {"human_safety", "escape_blocked", "goal_hold"}
        )
        if not human_interaction_state:
            released_interaction = human_decision.combined_interaction
            if released_interaction < 0.1:
                released_interaction *= 0.6
            self.interaction_level = min(self.interaction_level, released_interaction)
            self.interaction_level_memory = self.interaction_level
            if interaction_hard_reset:
                self.interaction_level_current = 0.0
                self.interaction_level = 0.0
                self.interaction_level_memory = 0.0
                self.interaction_hold_timer = 0.0
                self.interaction_active_duration = 0.0
                self.interaction_rearm_timer = 0.0
                self.dominant_human_memory_id = -1
                self.active_interactions = []
                self.interacting_human_id = -1
                self.interaction_clearance = float("inf")
                self.interaction_clearance_rate = 0.0
                self.previous_interaction_clearance = float("inf")
                self.previous_interacting_human_id = -1
        if human_interaction_state:
            self.interaction_hold_timer = self.interaction_min_hold_time
        else:
            self.interaction_hold_timer = max(self.interaction_hold_timer - dt, 0.0)
        if human_interaction_state:
            self.interaction_active_duration = previous_interaction_duration + dt
        else:
            self.interaction_active_duration = 0.0
        if next_state == FSMState.HARD_STOP:
            if previous_state != FSMState.HARD_STOP.value:
                self.stop_resume_timer = self.stop_resume_hold_time
            else:
                self.stop_resume_timer = max(self.stop_resume_timer - dt, 0.0)
        else:
            self.stop_resume_timer = 0.0

        progress_recovery_active = (
            not safe_progress_override
            and
            human_decision.combined_interaction < 0.1
            and (
                deadlock_recovery_active
                or recent_progress_stalled
                or escape_timeout_active
            )
        )
        recovery_direction = _compute_recovery_escape_direction(
            guide_dir,
            static_profile.trigger_gradient,
            previous_direction,
            min_alignment=effective_min_forward_dot,
        )
        boundary_normal = self._static_boundary_normal(
            self.position,
            obstacles,
            zones,
            previous_direction,
        )
        boundary_escape_direction = _project_to_goal_cone(
            _normalize(
                0.4 * guide_dir + 0.6 * boundary_normal,
                boundary_normal,
            ),
            guide_dir,
            effective_min_forward_dot,
        )
        boundary_tangent_direction = _compute_boundary_tangent_direction(
            guide_dir,
            boundary_normal,
            previous_direction,
            recovery_direction,
            min_alignment=effective_min_forward_dot,
        )
        if next_state in {FSMState.STATIC_ESCAPE, FSMState.HUMAN_YIELD} and static_escape_decision.active:
            direction_candidate = static_escape_decision.direction.copy()
        else:
            direction_candidate = guide_dir.copy()
        if (
            next_state == FSMState.HARD_STOP
            and not hard_stop_imminent
            and next_stop_reason in {"static_safety", "escape_blocked"}
        ):
            direction_candidate = _normalize(
                0.5 * direction_candidate + 0.5 * boundary_escape_direction,
                boundary_escape_direction,
            )
        elif progress_recovery_active:
            direction_candidate = _normalize(
                0.35 * direction_candidate
                + 0.4 * boundary_escape_direction
                + 0.25 * recovery_direction,
                direction_candidate,
            )

        if next_state == FSMState.HARD_STOP and hard_stop_imminent:
            direction = previous_direction.copy()
        else:
            direction_blend = min(
                self.momentum,
                0.4 if next_state in {FSMState.STATIC_ESCAPE, FSMState.HARD_STOP} else 0.6,
            )
            smoothed_direction = _normalize(
                direction_blend * previous_direction + (1.0 - direction_blend) * direction_candidate,
                direction_candidate,
            )
            if abs(_signed_angle_between(previous_direction, smoothed_direction)) > float(np.deg2rad(12.0)):
                smoothed_direction = _normalize(
                    0.65 * previous_direction + 0.35 * smoothed_direction,
                    smoothed_direction,
                )
            direction = _rotate_toward(
                previous_direction,
                smoothed_direction,
                min(
                    self.max_turn_angle,
                    self.hard_turn_limit,
                    self.heading_rate_limit / max(self.curvature_damping, 1e-6),
                ),
                damping=self.curvature_damping,
            )
            if next_state in {FSMState.STATIC_ESCAPE, FSMState.HUMAN_YIELD, FSMState.HARD_STOP}:
                direction = _project_to_goal_cone(direction, guide_dir, effective_min_forward_dot)

        commanded_speed_limit = self.speed
        if goal_distance <= settle_radius:
            commanded_speed_limit = min(commanded_speed_limit, goal_distance / dt)

        hard_stop_speed_floor = max(0.02, 0.25 * self.min_speed_scale)
        if next_state == FSMState.HARD_STOP:
            if hard_stop_imminent:
                target_speed = 0.0
            elif next_stop_reason in {"human_safety", "escape_blocked"}:
                target_speed = commanded_speed_limit * max(
                    hard_stop_speed_floor,
                    min(human_decision.speed_scale, 0.18),
                )
            else:
                static_speed_scale = float(
                    np.clip(
                        (current_static_clearance - hard_stop_clearance_floor)
                        / max(self.safe_clearance - hard_stop_clearance_floor, 1e-6),
                        hard_stop_speed_floor,
                        0.25,
                    )
                )
                target_speed = commanded_speed_limit * static_speed_scale
        elif next_state == FSMState.HUMAN_YIELD:
            target_speed = commanded_speed_limit * human_decision.speed_scale
        else:
            target_speed = commanded_speed_limit
        if next_state in {FSMState.STATIC_ESCAPE, FSMState.HARD_STOP} and target_speed > 0.0:
            risk_gradient_scale = float(
                np.clip(
                    1.0 / (
                        1.0
                        + 0.35 * np.linalg.norm(static_profile.trigger_gradient)
                        + 0.2 * max(static_profile.future_max_risk, static_profile.current_risk)
                    ),
                    hard_stop_speed_floor if next_state == FSMState.HARD_STOP else self.min_speed_scale,
                    1.0,
                )
            )
            target_speed *= risk_gradient_scale

        raw_speed_scale = (
            0.0 if commanded_speed_limit <= 1e-9 else float(np.clip(target_speed / commanded_speed_limit, 0.0, 1.0))
        )
        previous_speed_scale = float(np.clip(self.behavior_speed_scale, 0.0, 1.0))
        relief_boost_active = (
            self.post_interaction_relief_timer > 1e-9
            or (previous_human_interaction and not human_interaction_state)
        )
        recovery_boost_rate = (
            max(self.interaction_recovery_gain, 2.0)
            if relief_boost_active
            else self.interaction_recovery_gain
        )
        if next_state == FSMState.HARD_STOP and hard_stop_imminent:
            filtered_speed_scale = 0.0
        elif raw_speed_scale <= previous_speed_scale:
            filtered_speed_scale = max(
                raw_speed_scale,
                self.speed_scale_smoothing * raw_speed_scale
                + (1.0 - self.speed_scale_smoothing) * previous_speed_scale,
            )
        else:
            filtered_speed_scale = min(
                raw_speed_scale,
                previous_speed_scale + recovery_boost_rate * dt,
            )
        if relief_boost_active and next_state in {FSMState.GOAL_SEEK, FSMState.STATIC_ESCAPE}:
            filtered_speed_scale = min(
                raw_speed_scale,
                filtered_speed_scale + max(self.interaction_recovery_gain, 2.0) * dt,
            )
        target_speed = commanded_speed_limit * filtered_speed_scale

        if (
            next_state in {FSMState.GOAL_SEEK, FSMState.STATIC_ESCAPE}
            and target_speed > 0.0
            and goal_distance > self.goal_tolerance
        ):
            target_speed = max(target_speed, min(0.05 * self.speed, commanded_speed_limit))

        dynamic_safety_margin = float("inf")
        dynamic_clearance_buffer = (
            self.safety_hysteresis
            if hard_stop_imminent
            else 0.5 * self.safety_hysteresis
        )
        if self.detour_active:
            dynamic_safety_clearance = self.safety_distance + 1e-4
        else:
            if (
                next_state in {FSMState.GOAL_SEEK, FSMState.STATIC_ESCAPE}
                and human_decision.combined_interaction < 0.1
            ):
                dynamic_clearance_buffer *= 0.5
            dynamic_safety_clearance = self.safety_distance + dynamic_clearance_buffer
        def projection_clearance_with_boundary(candidate_position: np.ndarray) -> float:
            return self._compute_environment_clearance_at(
                candidate_position,
                world_size,
                humans,
                obstacles,
                zones,
            )

        dynamic_clearance_fn: Callable[[np.ndarray], float] | None = None
        if target_speed > 1e-9 and humans:
            predicted_human_positions = [human.position + dt * human.velocity for human in humans]

            def dynamic_clearance_fn(candidate_position: np.ndarray) -> float:
                min_clearance = float("inf")
                for predicted_position, human in zip(predicted_human_positions, humans):
                    clearance = np.linalg.norm(candidate_position - predicted_position) - (
                        self.radius + human.radius
                    )
                    min_clearance = min(min_clearance, clearance)
                return float(min_clearance)

            dynamic_projection = scale_speed_to_safe_margin(
                position=self.position,
                direction=direction,
                target_speed=target_speed,
                dt=dt,
                safety_distance=dynamic_safety_clearance,
                clearance_fn=dynamic_clearance_fn,
            )
            target_speed = float(np.linalg.norm(dynamic_projection.velocity))
            dynamic_safety_margin = dynamic_projection.clearance - self.safety_distance

        static_projection_safety_distance = self.safety_distance
        if self.detour_active:
            static_projection_safety_distance = max(
                static_projection_safety_distance,
                self.safety_distance + 1e-4,
            )
        if force_recovery and current_static_clearance < self.safety_distance - 1e-6:
            static_projection_safety_distance = min(
                self.safety_distance,
                current_static_clearance + 1e-3,
            )
        tracking_goal_dir = (
            release_reference_dir.copy()
            if interaction_release_active
            else (
                guide_dir.copy()
                if self.detour_active
                or (
                    next_state in {FSMState.GOAL_SEEK, FSMState.STATIC_ESCAPE}
                    and human_decision.combined_interaction < 0.1
                )
                else goal_dir.copy()
            )
        )
        human_stagnation_release = (
            goal_distance > self.goal_tolerance
            and (
                recent_progress_stalled
                or interaction_deadlock_active
                or interaction_release_active
            )
            and bool(humans)
            and dynamic_clearance_fn is not None
            and next_state in {FSMState.GOAL_SEEK, FSMState.HUMAN_YIELD, FSMState.HARD_STOP, FSMState.STATIC_ESCAPE}
            and not hard_stop_imminent
            and current_static_clearance >= self.safety_distance - 1e-6
            and current_physical_clearance >= self.safety_distance - 1e-6
            and (
                human_decision.combined_interaction >= self.interaction_recovery_threshold
                or next_state in {FSMState.HUMAN_YIELD, FSMState.HARD_STOP, FSMState.STATIC_ESCAPE}
            )
        )

        if next_state == FSMState.HARD_STOP and hard_stop_imminent:
            safe_velocity = np.zeros(2, dtype=float)
            static_projection_margin = current_static_clearance - self.safety_distance
            static_recovery_motion = False
            progress_mode_motion = False
        else:
            static_recovery_motion = False
            progress_mode_motion = False
            commanded_velocity = direction * target_speed
            static_projection = project_velocity_to_static_safe_set(
                position=self.position,
                velocity=commanded_velocity,
                dt=dt,
                world_size=world_size,
                safety_distance=static_projection_safety_distance,
                clearance_fn=projection_clearance_with_boundary,
                boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                    candidate_position,
                    obstacles,
                    zones,
                    fallback_velocity,
                ),
            )
            safe_velocity = static_projection.velocity
            static_projection_margin = static_projection.safety_margin
            safe_velocity = _project_to_forward_half_plane(safe_velocity, tracking_goal_dir)
            forward_progress_epsilon = min(max(0.05 * self.speed, 1e-3), np.linalg.norm(safe_velocity))
            if (
                np.linalg.norm(safe_velocity) > 1e-9
                and float(np.dot(safe_velocity, tracking_goal_dir)) < forward_progress_epsilon
            ):
                if static_projection_margin > self.safety_hysteresis:
                    safe_velocity = _enforce_forward_progress(
                        safe_velocity,
                        tracking_goal_dir,
                        forward_progress_epsilon,
                    )
                else:
                    safe_velocity[:] = 0.0
            stalled_static_escape = (
                human_decision.combined_interaction < 0.1
                and current_static_clearance <= max(self.safe_clearance, static_activation_radius)
                and (
                    np.linalg.norm(safe_velocity) <= 1e-9
                    or progress_recovery_active
                )
            )
            static_deadlock_recovery = (
                not safe_progress_override
                and
                next_state in {FSMState.STATIC_ESCAPE, FSMState.GOAL_SEEK}
                and human_decision.combined_interaction < 0.1
                and (
                    (
                        next_state == FSMState.STATIC_ESCAPE
                        and (
                            (
                                np.linalg.norm(safe_velocity) < 1e-3
                                and recent_progress_stalled
                            )
                            or (
                                deadlock_recovery_active
                                and current_static_clearance < self.safe_clearance
                                and recent_progress_stalled
                            )
                        )
                    )
                    or (
                        next_state == FSMState.GOAL_SEEK
                        and recent_progress_stalled
                        and suppress_static_escape
                        and deadlock_recovery_active
                    )
                )
            )
            if (
                target_speed > 1e-9
                and (
                    stalled_static_escape
                    or (
                        next_state in {FSMState.GOAL_SEEK, FSMState.HUMAN_YIELD}
                        and np.linalg.norm(safe_velocity) <= 1e-9
                        and static_profile.future_max_risk >= max(0.65 * self.risk_threshold, self.grad_epsilon)
                    )
                )
            ):
                fallback_escape_dir = _normalize(
                    0.35 * previous_direction
                    + 0.4 * boundary_escape_direction
                    + 0.15 * recovery_direction
                    + 0.1 * boundary_tangent_direction,
                    boundary_escape_direction,
                )
                fallback_escape_dir = _project_to_goal_cone(
                    fallback_escape_dir,
                    guide_dir,
                    effective_min_forward_dot,
                )
                fallback_projection = project_velocity_to_static_safe_set(
                    position=self.position,
                    velocity=fallback_escape_dir * max(target_speed, min(0.18 * self.speed, commanded_speed_limit)),
                    dt=dt,
                    world_size=world_size,
                    safety_distance=static_projection_safety_distance,
                    clearance_fn=projection_clearance_with_boundary,
                    boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                        candidate_position,
                        obstacles,
                        zones,
                        fallback_velocity,
                    ),
                )
                fallback_velocity = _project_to_forward_half_plane(
                    fallback_projection.velocity,
                    tracking_goal_dir,
                )
                if np.linalg.norm(fallback_velocity) > 1e-9:
                    safe_velocity = fallback_velocity
                    static_projection_margin = fallback_projection.safety_margin
                elif stalled_static_escape:
                    tangent_speed = min(
                        max(target_speed, 0.12 * self.speed),
                        commanded_speed_limit,
                    )
                    tangent_velocity = _project_to_forward_half_plane(
                        boundary_escape_direction * tangent_speed,
                        tracking_goal_dir,
                    )
                    tangent_position = self.position + tangent_velocity * dt
                    tangent_clearance = projection_clearance_with_boundary(tangent_position)
                    if tangent_clearance >= self.safety_distance - 5e-7:
                        safe_velocity = tangent_velocity
                        static_projection_margin = tangent_clearance - self.safety_distance
            if safe_progress_override:
                progress_min_speed = min(
                    max(0.05 * self.speed, 0.05),
                    commanded_speed_limit,
                )
                progress_speed = min(
                    max(target_speed, progress_min_speed),
                    commanded_speed_limit,
                )
                progress_velocity, progress_margin, progress_floor_feasible = _certify_forward_progress_velocity(
                    position=self.position,
                    direction=tracking_goal_dir,
                    target_speed=progress_speed,
                    min_progress_speed=progress_min_speed,
                    dt=dt,
                    safety_distance=static_projection_safety_distance,
                    clearance_fn=projection_clearance_with_boundary,
                    dynamic_clearance_fn=dynamic_clearance_fn,
                    dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                )
                progress_gain = float(np.dot(progress_velocity, tracking_goal_dir) * dt)
                goal_forward_velocity, _, goal_floor_feasible = _certify_forward_progress_velocity(
                    position=self.position,
                    direction=static_guide_dir,
                    target_speed=progress_speed,
                    min_progress_speed=progress_min_speed,
                    dt=dt,
                    safety_distance=static_projection_safety_distance,
                    clearance_fn=projection_clearance_with_boundary,
                    dynamic_clearance_fn=dynamic_clearance_fn,
                    dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                )
                escape_trigger = (
                    false_progress
                    or recent_progress_stalled
                    or progress_regressing
                    or interaction_deadlock_active
                )
                if (
                    escape_trigger
                    and not self.detour_active
                ):
                    if self.detour_side == 0:
                        tangent = _perpendicular(static_guide_dir)
                        outward = _normalize(-static_profile.trigger_gradient, static_guide_dir)
                        self.detour_side = 1 if float(np.dot(tangent, outward)) > 0.0 else -1
                    recent_escape_triggers = sum(
                        1
                        for trigger_step in self.escape_activation_steps
                        if self.control_step_counter - trigger_step <= self.escape_retrigger_window_steps
                    )
                    if recent_escape_triggers >= self.escape_retrigger_threshold:
                        self.escape_gain = min(2.0, self.escape_gain + 0.2)
                    else:
                        self.escape_gain = max(1.0, self.escape_gain - 0.05)
                    detour_direction, detour_clearance_gain = _select_committed_detour_direction(
                        position=self.position,
                        guide_direction=static_guide_dir,
                        static_gradient=static_profile.trigger_gradient,
                        previous_direction=previous_direction,
                        escape_gain=self.escape_gain,
                        target_speed=progress_speed,
                        min_progress_speed=progress_min_speed,
                        dt=dt,
                        safety_distance=static_projection_safety_distance,
                        clearance_fn=projection_clearance_with_boundary,
                        dynamic_clearance_fn=dynamic_clearance_fn,
                        dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                        failed_detours=self.failed_detours,
                        lookahead_risk_fn=lambda candidate_position: risk_field.compute_hazard_risk(
                            candidate_position
                        ),
                        lookahead_step=self.speed * dt,
                        progress_value=progress_delta,
                        progress_epsilon=progress_epsilon,
                        progress_timer=self.progress_stall_timer,
                        progress_timer_threshold=progress_stall_time,
                        detour_side=self.detour_side,
                    )
                    if detour_direction is not None and detour_clearance_gain > -1e-6:
                        detour_direction = _normalize(detour_direction, static_guide_dir)
                        if float(np.dot(detour_direction, static_guide_dir)) <= -0.8:
                            tangent = _perpendicular(static_guide_dir)
                            if float(np.dot(tangent, previous_direction)) < 0.0:
                                tangent *= -1.0
                            detour_direction = _normalize(tangent, static_guide_dir)
                        if float(np.dot(detour_direction, previous_direction)) < -0.3:
                            detour_direction = _normalize(previous_direction, static_guide_dir)

                        commit_steps = max(self.goal_distance_window.maxlen, self.escape_commit_steps)
                        commit_steps = max(
                            int(np.ceil(commit_steps * self.escape_gain)),
                            self.escape_flip_lock_steps,
                        )
                        self.detour_active = True
                        self.detour_direction = detour_direction.copy()
                        self.detour_steps_remaining = commit_steps
                        self.detour_anchor_position = self.position.copy()
                        self.detour_anchor_distance = goal_distance
                        self.detour_last_progress = 0.0
                        self.current_branch_point = static_guide_target.copy()
                        self.current_branch_guide_direction = static_guide_dir.copy()
                        self.detour_lock_direction = detour_direction.copy()
                        self.detour_lock_steps = 8
                        self.last_detour_direction = detour_direction.copy()
                        self.escape_activation_steps.append(self.control_step_counter)
                        if interaction_release_active:
                            self._clear_interaction_release()
                            interaction_release_active = False
                        guide_dir = detour_direction.copy()
                        previous_direction = _normalize(self.prev_direction, guide_dir)
                        tracking_goal_dir = detour_direction.copy()
                        progress_velocity, progress_margin, progress_floor_feasible = _certify_forward_progress_velocity(
                            position=self.position,
                            direction=tracking_goal_dir,
                            target_speed=progress_speed,
                            min_progress_speed=progress_min_speed,
                            dt=dt,
                            safety_distance=static_projection_safety_distance,
                            clearance_fn=projection_clearance_with_boundary,
                            dynamic_clearance_fn=dynamic_clearance_fn,
                            dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                        )
                        progress_gain = float(np.dot(progress_velocity, tracking_goal_dir) * dt)

                if np.linalg.norm(progress_velocity) < progress_min_speed - 1e-6:
                    candidate_velocity, candidate_margin, candidate_gain = _select_progress_guarantee_velocity(
                        position=self.position,
                        goal=self.goal,
                        goal_direction=tracking_goal_dir,
                        boundary_normal=boundary_normal,
                        target_speed=progress_speed,
                        dt=dt,
                        world_size=world_size,
                        safety_distance=static_projection_safety_distance,
                        current_clearance=current_static_clearance,
                        preferred_clearance=self.safe_clearance,
                        clearance_fn=projection_clearance_with_boundary,
                        boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                            candidate_position,
                            obstacles,
                            zones,
                            fallback_velocity,
                        ),
                        dynamic_clearance_fn=dynamic_clearance_fn,
                        dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                    )
                    candidate_speed = float(np.linalg.norm(candidate_velocity))
                    candidate_alignment = float(
                        np.dot(_normalize(candidate_velocity, tracking_goal_dir), tracking_goal_dir)
                    )
                    if (
                        candidate_speed > 1e-9
                        and candidate_alignment >= max(self.min_forward_dot, 0.2) - 1e-6
                        and candidate_gain > 1e-4
                    ):
                        progress_velocity = candidate_velocity
                        progress_margin = candidate_margin
                        progress_gain = candidate_gain

                if recent_progress_stalled and progress_floor_feasible:
                    enforced_velocity, enforced_margin, _ = _certify_forward_progress_velocity(
                        position=self.position,
                        direction=tracking_goal_dir,
                        target_speed=progress_speed,
                        min_progress_speed=progress_min_speed,
                        dt=dt,
                        safety_distance=static_projection_safety_distance,
                        clearance_fn=projection_clearance_with_boundary,
                        dynamic_clearance_fn=dynamic_clearance_fn,
                        dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                    )
                    if np.linalg.norm(enforced_velocity) >= progress_min_speed - 1e-6:
                        progress_velocity = enforced_velocity
                        progress_margin = enforced_margin
                        progress_gain = float(np.dot(progress_velocity, tracking_goal_dir) * dt)

                progress_speed_actual = float(np.linalg.norm(progress_velocity))
                progress_alignment = float(
                    np.dot(_normalize(progress_velocity, tracking_goal_dir), tracking_goal_dir)
                ) if progress_speed_actual > 1e-9 else 0.0
                if progress_speed_actual > 1e-9 and progress_alignment < max(self.min_forward_dot, 0.2):
                    forced_speed = max(float(np.dot(progress_velocity, tracking_goal_dir)), progress_min_speed)
                    corrected_velocity, corrected_margin, _ = _certify_forward_progress_velocity(
                        position=self.position,
                        direction=tracking_goal_dir,
                        target_speed=forced_speed,
                        min_progress_speed=progress_min_speed,
                        dt=dt,
                        safety_distance=static_projection_safety_distance,
                        clearance_fn=projection_clearance_with_boundary,
                        dynamic_clearance_fn=dynamic_clearance_fn,
                        dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                    )
                    progress_velocity = corrected_velocity
                    progress_margin = corrected_margin

                if np.linalg.norm(progress_velocity) < progress_min_speed - 1e-6:
                    forced_velocity, forced_margin, forced_floor_feasible = _certify_forward_progress_velocity(
                        position=self.position,
                        direction=tracking_goal_dir,
                        target_speed=progress_min_speed,
                        min_progress_speed=progress_min_speed,
                        dt=dt,
                        safety_distance=static_projection_safety_distance,
                        clearance_fn=projection_clearance_with_boundary,
                        dynamic_clearance_fn=dynamic_clearance_fn,
                        dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                    )
                    if forced_floor_feasible and np.linalg.norm(forced_velocity) > 1e-9:
                        progress_velocity = forced_velocity
                        progress_margin = forced_margin

                if np.linalg.norm(progress_velocity) > 1e-9 and float(np.dot(progress_velocity, tracking_goal_dir)) > 1e-9:
                    safe_velocity = progress_velocity
                    static_projection_margin = progress_margin
                    progress_mode_motion = True
            if human_stagnation_release:
                release_min_speed = min(
                    max(0.05 * self.speed, 0.05),
                    commanded_speed_limit,
                )
                release_target_speed = min(
                    max(target_speed, self.progress_speed_floor * self.speed),
                    commanded_speed_limit,
                )
                release_velocity, release_margin, release_floor_feasible = _certify_forward_progress_velocity(
                    position=self.position,
                    direction=release_reference_dir,
                    target_speed=release_target_speed,
                    min_progress_speed=release_min_speed,
                    dt=dt,
                    safety_distance=static_projection_safety_distance,
                    clearance_fn=projection_clearance_with_boundary,
                    dynamic_clearance_fn=dynamic_clearance_fn,
                    dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                )
                if release_floor_feasible:
                    release_alignment = float(
                        np.dot(_normalize(release_velocity, release_reference_dir), release_reference_dir)
                    )
                    if (
                        np.linalg.norm(release_velocity) >= release_min_speed - 1e-6
                        and release_alignment > 1e-6
                    ):
                        release_hold_time = max(1.0, self.interaction_min_hold_time)
                        release_direction = release_reference_dir.copy()
                        release_projection_anchor = guide_path_progress
                        release_path_distance_anchor = guide_path_distance
                        if off_path_active:
                            release_progress_target = max(
                                guide_path_distance - path_reentry_epsilon,
                                1e-6,
                            )
                        else:
                            release_projection_speed = max(
                                release_min_speed,
                                float(np.dot(release_velocity, release_direction)),
                            )
                            release_progress_target = max(
                                0.04,
                                2.0 * release_projection_speed * dt,
                            )
                        if not interaction_release_active:
                            self.interaction_release_direction = release_direction.copy()
                            self.interaction_release_projection_anchor = release_projection_anchor
                            self.interaction_release_path_distance_anchor = release_path_distance_anchor
                            self.interaction_release_anchor_position = self.position.copy()
                            self.interaction_release_progress = 0.0
                            self.interaction_release_required_progress = release_progress_target
                            interaction_release_active = True
                        else:
                            self.interaction_release_direction = release_direction.copy()
                        self.interaction_release_timer = max(
                            self.interaction_release_timer,
                            release_hold_time,
                        )
                        safe_velocity = release_velocity
                        static_projection_margin = release_margin
                        progress_mode_motion = True
                        next_state = (
                            FSMState.STATIC_ESCAPE
                            if self.detour_active or interaction_deadlock_active or interaction_release_active
                            else FSMState.GOAL_SEEK
                        )
                        next_stop_reason = "none"
                        self.stop_mode_reason = "none"
                        self.stop_mode_active = False
                        self.interaction_hold_timer = 0.0
                        self.stop_resume_timer = 0.0
            if static_deadlock_recovery:
                recovery_speed = min(
                    max(target_speed, self.progress_speed_floor * self.speed),
                    commanded_speed_limit,
                )
                recovery_velocity, recovery_margin, recovery_clearance_gain = _select_static_recovery_velocity(
                    position=self.position,
                    goal_direction=tracking_goal_dir,
                    boundary_normal=boundary_normal,
                    target_speed=recovery_speed,
                    dt=dt,
                    world_size=world_size,
                    safety_distance=static_projection_safety_distance,
                    current_clearance=current_static_clearance,
                    clearance_fn=projection_clearance_with_boundary,
                    boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                        candidate_position,
                        obstacles,
                        zones,
                        fallback_velocity,
                    ),
                )
                if np.linalg.norm(recovery_velocity) > 1e-9:
                    safe_velocity = recovery_velocity
                    static_projection_margin = recovery_margin
                    static_recovery_motion = True

            if interaction_release_active and not hard_stop_imminent:
                release_hold_min_speed = min(
                    max(0.05 * self.speed, 0.05),
                    commanded_speed_limit,
                )
                release_hold_target_speed = min(
                    max(target_speed, release_hold_min_speed),
                    commanded_speed_limit,
                )
                held_release_velocity, held_release_margin, held_release_feasible = _certify_forward_progress_velocity(
                    position=self.position,
                    direction=release_reference_dir,
                    target_speed=release_hold_target_speed,
                    min_progress_speed=release_hold_min_speed,
                    dt=dt,
                    safety_distance=static_projection_safety_distance,
                    clearance_fn=projection_clearance_with_boundary,
                    dynamic_clearance_fn=dynamic_clearance_fn,
                    dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                )
                if (
                    held_release_feasible
                    and np.linalg.norm(held_release_velocity) >= release_hold_min_speed - 1e-6
                    and float(np.dot(held_release_velocity, release_reference_dir)) > 1e-6
                ):
                    safe_velocity = held_release_velocity
                    static_projection_margin = held_release_margin
                    progress_mode_motion = True
                    static_recovery_motion = False
                    next_state = (
                        FSMState.STATIC_ESCAPE
                        if self.detour_active or interaction_deadlock_active or interaction_release_active
                        else FSMState.GOAL_SEEK
                    )
                    next_stop_reason = "none"
                    self.stop_mode_reason = "none"
                    self.stop_mode_active = False
                    self.interaction_hold_timer = 0.0
                    self.stop_resume_timer = 0.0

        if self.detour_active or static_recovery_motion or progress_mode_motion:
            integration_smoothing = 0.0
        else:
            integration_smoothing = (
                min(self.velocity_smoothing, 0.25)
                if relief_boost_active or progress_recovery_active
                else self.velocity_smoothing
            )
        integration_command = safe_velocity.copy()
        if integration_smoothing > 1e-9 and next_state != FSMState.HARD_STOP:
            pre_projection_command = (
                integration_smoothing * self.velocity
                + (1.0 - integration_smoothing) * safe_velocity
            )
            integration_projection = project_velocity_to_static_safe_set(
                position=self.position,
                velocity=pre_projection_command,
                dt=dt,
                world_size=world_size,
                safety_distance=static_projection_safety_distance,
                clearance_fn=projection_clearance_with_boundary,
                boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                    candidate_position,
                    obstacles,
                    zones,
                    fallback_velocity,
                ),
            )
            integration_command = integration_projection.velocity
            integration_smoothing = 0.0
        backward_tangent_tolerance = max(1e-4, 1e-3 * dt)
        next_position, next_velocity = integrate_velocity_command(
            position=self.position,
            current_velocity=self.velocity,
            commanded_velocity=integration_command,
            dt=dt,
            world_size=world_size,
            velocity_smoothing=integration_smoothing,
            max_speed=self.speed,
        )
        if off_path_active and next_state != FSMState.HARD_STOP:
            corrected_executed_velocity = _enforce_nonnegative_path_tangent_velocity(
                position=self.position,
                velocity=next_velocity,
                path_tangent=guide_path_tangent,
                dt=dt,
                world_size=world_size,
                safety_distance=static_projection_safety_distance,
                clearance_fn=projection_clearance_with_boundary,
                boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                    candidate_position,
                    obstacles,
                    zones,
                    fallback_velocity,
                ),
                dynamic_clearance_fn=dynamic_clearance_fn,
                dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                backward_tolerance=backward_tangent_tolerance,
            )
            if np.linalg.norm(corrected_executed_velocity - next_velocity) > 1e-9:
                next_position, next_velocity = integrate_velocity_command(
                    position=self.position,
                    current_velocity=self.velocity,
                    commanded_velocity=corrected_executed_velocity,
                    dt=dt,
                    world_size=world_size,
                    velocity_smoothing=0.0,
                    max_speed=self.speed,
                )
                final_tangent_component = float(np.dot(next_velocity, guide_path_tangent))
                if final_tangent_component < -backward_tangent_tolerance:
                    next_velocity = next_velocity - final_tangent_component * guide_path_tangent
                    next_position = _clamp_position(self.position + next_velocity * dt, world_size)
                    if (
                        projection_clearance_with_boundary(next_position) < static_projection_safety_distance - 1e-6
                        or (
                            dynamic_clearance_fn is not None
                            and dynamic_safety_clearance is not None
                            and float(dynamic_clearance_fn(next_position)) < dynamic_safety_clearance - 1e-6
                        )
                    ):
                        next_position = self.position.copy()
                        next_velocity[:] = 0.0
        if next_state != FSMState.HARD_STOP:
            next_goal_dir = _normalize(self.goal - next_position, guide_path_tangent)
            next_global_progress, _, next_path_tangent, _, _ = self._global_guide_path_state_at(
                next_position,
                next_goal_dir,
            )
            if (
                self.execution_progress_invariant_enabled
                and next_global_progress < self.guide_progress_max - self.guide_progress_regression_tolerance
            ):
                corrected_global_velocity = _enforce_nonnegative_path_tangent_velocity(
                    position=self.position,
                    velocity=next_velocity,
                    path_tangent=next_path_tangent,
                    dt=dt,
                    world_size=world_size,
                    safety_distance=static_projection_safety_distance,
                    clearance_fn=projection_clearance_with_boundary,
                    boundary_normal_fn=lambda candidate_position, fallback_velocity: self._static_boundary_normal(
                        candidate_position,
                        obstacles,
                        zones,
                        fallback_velocity,
                    ),
                    dynamic_clearance_fn=dynamic_clearance_fn,
                    dynamic_safety_distance=dynamic_safety_clearance if dynamic_clearance_fn is not None else None,
                    backward_tolerance=self.guide_progress_regression_tolerance,
                )
                if np.linalg.norm(corrected_global_velocity - next_velocity) > 1e-9:
                    next_position, next_velocity = integrate_velocity_command(
                        position=self.position,
                        current_velocity=self.velocity,
                        commanded_velocity=corrected_global_velocity,
                        dt=dt,
                        world_size=world_size,
                        velocity_smoothing=0.0,
                        max_speed=self.speed,
                    )
                    next_goal_dir = _normalize(self.goal - next_position, next_path_tangent)
                    next_global_progress, _, _, _, _ = self._global_guide_path_state_at(
                        next_position,
                        next_goal_dir,
                    )
                if (
                    self.execution_progress_invariant_enabled
                    and next_global_progress < self.guide_progress_max - self.guide_progress_regression_tolerance
                ):
                    next_position = self.position.copy()
                    next_velocity[:] = 0.0
        if next_state == FSMState.HARD_STOP and hard_stop_imminent:
            next_position = self.position.copy()
            next_velocity[:] = 0.0
        integrated_clearance = projection_clearance_with_boundary(next_position)
        if (
            current_static_clearance >= self.safety_distance - 1e-6
            and next_state != FSMState.HARD_STOP
            and integrated_clearance < self.safety_distance - 1e-6
        ):
            next_position = self.position.copy()
            next_velocity[:] = 0.0
            integrated_clearance = projection_clearance_with_boundary(next_position)

        self.position = next_position
        self.velocity = (
            next_velocity.copy()
            if static_recovery_motion or progress_mode_motion
            else _project_to_forward_half_plane(next_velocity, tracking_goal_dir)
        )
        applied_speed = float(np.linalg.norm(self.velocity))
        if applied_speed > self.speed > 1e-9:
            self.velocity *= self.speed / applied_speed
            applied_speed = self.speed

        self.behavior_state = next_state.value
        if self.behavior_state != FSMState.STATIC_ESCAPE.value:
            self.detour_side = 0
        self.behavior_speed_scale = float(np.clip(applied_speed / max(self.speed, 1e-9), 0.0, 1.0))
        self.prev_direction = _normalize(self.velocity, direction).copy()
        recovery_speed_ratio = float(np.clip(applied_speed / max(commanded_speed_limit, 1e-9), 0.0, 1.0))
        new_goal_distance = float(np.linalg.norm(self.goal - self.position))
        goal_distance_progress = (
            self.prev_goal_dist - new_goal_distance
            if np.isfinite(self.prev_goal_dist)
            else 0.0
        )
        updated_static_clearance = self._compute_min_static_clearance_at(
            self.position,
            obstacles,
            zones,
        )
        progress = float(np.dot(self.position - start_position, goal_dir))
        executed_goal_progress = float(goal_distance - new_goal_distance)
        self.goal_progress_window.append(executed_goal_progress)
        if (
            self.behavior_state == FSMState.STATIC_ESCAPE.value
            and self.detour_side != 0
        ):
            if goal_distance_progress < 0.01:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            if (
                self.stagnation_counter > 25
                and (self.control_step_counter - self.last_flip_step) > 50
            ):
                self._remember_failed_branch(
                    self.current_branch_point,
                    self.current_branch_guide_direction,
                )
                self.branch_rebuild_required = True
                failed_direction_source = (
                    self.detour_direction
                    if np.linalg.norm(self.detour_direction) > 1e-9
                    else (
                        self.last_detour_direction
                        if self.last_detour_direction is not None
                        else self.prev_direction
                    )
                )
                failed_direction = _normalize(failed_direction_source, self.prev_direction)
                if np.linalg.norm(failed_direction) > 1e-9:
                    self.failed_detours.append(failed_direction.copy())
                    if len(self.failed_detours) > 3:
                        self.failed_detours = self.failed_detours[-3:]
                flipped_side = -self.detour_side
                self._clear_detour_mode()
                self.detour_side = flipped_side
                self.stagnation_counter = 0
                self.last_flip_step = self.control_step_counter
                self.detour_side_flip_count += 1
        elif (
            human_decision.combined_interaction < 0.1
            and (previous_state == FSMState.STATIC_ESCAPE.value or self.behavior_state == FSMState.STATIC_ESCAPE.value)
        ):
            if progress < 1e-4:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0
        self.prev_goal_dist = new_goal_distance
        if self.interaction_release_timer > 1e-9:
            self.interaction_release_timer = max(self.interaction_release_timer - dt, 0.0)
        if (
            self.interaction_release_required_progress > 1e-9
            and np.linalg.norm(self.interaction_release_direction) > 1e-9
        ):
            current_path_progress, _, _, current_path_distance, _ = self._current_global_guide_path_state(
                _normalize(self.goal - self.position, self.prev_direction)
            )
            path_reentry_epsilon = self._path_reentry_epsilon(dt)
            if self.interaction_release_path_distance_anchor > path_reentry_epsilon + 1e-6:
                self.interaction_release_progress = max(
                    0.0,
                    self.interaction_release_path_distance_anchor - current_path_distance,
                )
                if (
                    current_path_distance <= path_reentry_epsilon + 1e-6
                    or self.interaction_release_progress >= self.interaction_release_required_progress - 1e-6
                ):
                    self.interaction_release_path_distance_anchor = 0.0
                    self.interaction_release_projection_anchor = current_path_progress
                    self.interaction_release_progress = 0.0
                    self.interaction_release_required_progress = max(
                        0.04,
                        2.0 * self.speed * self.min_speed_scale * dt,
                    )
                    self.interaction_release_timer = max(
                        self.interaction_release_timer,
                        self.interaction_min_hold_time,
                    )
            else:
                self.interaction_release_progress = max(
                    0.0,
                    current_path_progress - self.interaction_release_projection_anchor,
                )
                if (
                    self.interaction_release_progress >= self.interaction_release_required_progress - 1e-6
                    and self.interaction_release_timer <= 1e-9
                ):
                    self._clear_interaction_release()
        else:
            self.interaction_release_progress = 0.0
        current_global_progress, _, _, _, _ = self._current_global_guide_path_state(
            _normalize(self.goal - self.position, self.prev_direction)
        )
        self.guide_progress_max = max(self.guide_progress_max, current_global_progress)
        if self.deadlock_recovery_timer > 1e-9 and updated_static_clearance >= self.safe_clearance:
            self.deadlock_recovery_timer = 0.0

        was_human_interaction = previous_state == FSMState.HUMAN_YIELD.value or (
            previous_state == FSMState.HARD_STOP.value
            and previous_stop_reason in {"human_safety", "escape_blocked", "goal_hold"}
        )
        is_human_interaction = self.behavior_state == FSMState.HUMAN_YIELD.value or (
            self.behavior_state == FSMState.HARD_STOP.value
            and self.stop_mode_reason in {"human_safety", "escape_blocked", "goal_hold"}
        )
        self.interaction_mode_active = is_human_interaction
        if was_human_interaction and not is_human_interaction:
            self.post_interaction_relief_timer = self.post_interaction_relief_time
            self.recovery_timer_active = True
            self.recovery_timer_elapsed = 0.0
        elif is_human_interaction:
            self.post_interaction_relief_timer = 0.0
            self.recovery_timer_active = False
            self.recovery_timer_elapsed = 0.0
        else:
            self.post_interaction_relief_timer = max(self.post_interaction_relief_timer - dt, 0.0)
            if self.recovery_timer_active:
                self.recovery_timer_elapsed += dt
                if recovery_speed_ratio >= 0.9:
                    self.latest_recovery_time = self.recovery_timer_elapsed
                    self.recovery_timer_active = False

        boundary_clearance = _boundary_clearance(self.position, world_size, self.radius)
        goal_capture_clear = (
            min(
                self._compute_min_physical_clearance_at(
                    self.position,
                    humans,
                    obstacles,
                    zones,
                ),
                boundary_clearance,
            ) >= self.safety_distance - 1e-6
            and self._compute_min_static_clearance_at(
                self.position,
                obstacles,
                zones,
            ) >= self.safety_distance - 1e-6
        )
        if np.linalg.norm(self.goal - self.position) <= goal_capture_radius and goal_capture_clear:
            self._enter_goal_state(goal_dir)
        else:
            self.physical_clearance = min(
                self._compute_min_physical_clearance_at(
                    self.position,
                    humans,
                    obstacles,
                    zones,
                ),
                boundary_clearance,
            )
            self.minimum_physical_clearance = min(
                self.minimum_physical_clearance,
                self.physical_clearance,
            )
            self.previous_goal_distance = goal_distance
            self.goal_distance_window.append(new_goal_distance)

        final_static_clearance = min(
            self._compute_min_static_clearance_at(
                self.position,
                obstacles,
                zones,
            ),
            boundary_clearance,
        )
        final_dynamic_clearance = self._compute_min_physical_clearance_at(
            self.position,
            humans,
            [],
            (),
        )
        final_dynamic_clearance = min(final_dynamic_clearance, boundary_clearance)
        self.safety_margin = min(final_static_clearance, final_dynamic_clearance) - self.safety_distance
        self.clearance_min = min(self.global_clearance, final_static_clearance)
        self.previous_total_risk = float(current_risk)
        rule_clearances = [
            value
            for value in (
                tracked_interaction_clearance,
                final_static_clearance,
                self.global_clearance,
            )
            if np.isfinite(value)
        ]
        self.previous_rule_clearance = min(rule_clearances) if rule_clearances else float("inf")
        self.safety_margin = min(self.safety_margin, static_projection_margin, dynamic_safety_margin)
        self.trail.append(self.position.copy())

    def update(self, dt: float, world_size: np.ndarray, risk_field: RiskField | None = None) -> None:
        dt = _validate_positive(dt, "dt")
        goal_vector = self.goal - self.position
        goal_distance = np.linalg.norm(goal_vector)

        if goal_distance <= self.goal_tolerance:
            goal_hold_blocked = False
            if risk_field is not None:
                goal_hold_blocked = (
                    risk_field.compute_hazard_risk(self.position) > max(0.5 * self.risk_threshold, 0.15)
                    or self._compute_min_physical_clearance_at(
                        self.position,
                        risk_field.humans,
                        risk_field.obstacles,
                        risk_field.zones,
                    ) < (self.safety_distance - 1e-6)
                    or risk_field.nearest_hazard_distance(self.position) < self.safe_distance
                )
            if goal_hold_blocked:
                self.velocity[:] = 0.0
                self.stop_mode_active = True
                self.stop_mode_reason = "goal_hold"
                self.stop_resume_timer = 0.0
                self.behavior_state = FSMState.HARD_STOP.value
                self.behavior_speed_scale = 0.0
                self.ttc_min = float("inf")
                self.interaction_clearance_rate = 0.0
                self.trail.append(self.position.copy())
                return
            self._enter_goal_state(_normalize(goal_vector, self.prev_direction))
            self.trail.append(self.position.copy())
            return

        goal_dir = goal_vector / goal_distance
        if risk_field is not None:
            self._update_behavior_driven(dt, world_size, risk_field, goal_dir, goal_distance)
            return
        settle_radius = max(2.0 * self.goal_tolerance, 0.5 * self.goal_stabilization_distance)
        goal_capture_radius = max(self.goal_tolerance, 0.6 * self.goal_stabilization_distance)
        direction = _normalize(
            self.momentum * self.prev_direction + (1.0 - self.momentum) * goal_dir,
            goal_dir,
        )
        direction = _rotate_toward(
            _normalize(self.prev_direction, goal_dir),
            direction,
            min(
                self.max_turn_angle,
                self.hard_turn_limit,
                self.heading_rate_limit / max(self.curvature_damping, 1e-6),
            ),
            damping=self.curvature_damping,
        )

        commanded_speed_limit = self.speed
        if goal_distance <= settle_radius:
            commanded_speed_limit = min(commanded_speed_limit, goal_distance / dt)
        target_speed = commanded_speed_limit
        if target_speed > 0.0 and goal_distance > self.goal_tolerance:
            target_speed = max(target_speed, min(0.05 * self.speed, commanded_speed_limit))

        next_position, next_velocity = integrate_velocity_command(
            position=self.position,
            current_velocity=self.velocity,
            commanded_velocity=direction * target_speed,
            dt=dt,
            world_size=world_size,
            velocity_smoothing=self.velocity_smoothing,
            max_speed=self.speed,
        )
        self.position = next_position
        self.velocity = _project_to_forward_half_plane(next_velocity, goal_dir)
        self.behavior_state = FSMState.GOAL_SEEK.value
        self.behavior_speed_scale = float(np.clip(np.linalg.norm(self.velocity) / max(self.speed, 1e-9), 0.0, 1.0))
        self.prev_direction = _normalize(self.velocity, direction).copy()
        self.safety_margin = float("inf")
        self.risk_slope = 0.0
        self.clearance_min = float("inf")
        self.global_clearance = float("inf")
        self.interaction_clearance = float("inf")
        self.interaction_clearance_rate = 0.0
        self.interaction_level_current = 0.0
        self.interaction_level = 0.0
        self.interaction_level_memory = 0.0
        self.dominant_human_memory_id = -1
        self.interaction_hold_timer = 0.0
        self.interaction_active_duration = 0.0
        self.interaction_rearm_timer = 0.0
        self.stop_resume_timer = 0.0
        self.interaction_mode_active = False
        self.active_interactions = []
        self.stop_mode_active = False
        self.stop_mode_reason = "none"
        self.deadlock_timer = 0.0
        self.deadlock_recovery_timer = 0.0
        self.progress_stall_timer = 0.0
        self.static_escape_duration = 0.0
        self.static_escape_cooldown = 0
        self.previous_goal_distance = goal_distance
        self.goal_distance_window.append(float(np.linalg.norm(self.goal - self.position)))
        self.ttc_min = float("inf")
        if np.linalg.norm(self.goal - self.position) <= goal_capture_radius:
            self._enter_goal_state(goal_dir)
        self.trail.append(self.position.copy())


class Environment:
    def __init__(
        self,
        world_size: Sequence[float],
        robot: Robot,
        humans: Iterable[Human],
        obstacles: Iterable[Obstacle],
        zones: Iterable[NoGoZone],
        risk_field: RiskField,
        dt: float = 0.1,
        show_risk: bool = True,
        show_entity_risk_fields: bool = True,
        show_debug: bool = True,
        real_time: bool = False,
        render_skip: int = 3,
        risk_resolution: int = 60,
        risk_update_interval: int = 5,
        record_human_trails: bool = False,
        scenario_name: str = "custom",
        logger: NavLogger | None = None,
    ) -> None:
        self.world_size = _as_vector(world_size, "world_size")
        if np.any(self.world_size <= 0.0):
            raise ValueError("world_size must be positive in both dimensions")

        self.dt = _validate_positive(dt, "dt")
        self.robot = robot
        self.humans = list(humans)
        self.obstacles = list(obstacles)
        self.zones = list(zones)
        self.risk_field = risk_field
        self.show_risk = bool(show_risk)
        self.show_entity_risk_fields = bool(show_entity_risk_fields)
        self.show_debug = bool(show_debug)
        self.real_time = bool(real_time)
        self.render_skip = int(render_skip)
        self.risk_resolution = int(risk_resolution)
        self.risk_update_interval = int(risk_update_interval)
        self.record_human_trails = bool(record_human_trails)
        self.scenario_name = str(scenario_name)
        self.logger = logger
        if self.render_skip < 1:
            raise ValueError("render_skip must be at least 1")
        if self.risk_resolution < 2:
            raise ValueError("risk_resolution must be at least 2")
        if self.risk_resolution > 80:
            raise ValueError("risk_resolution must not exceed 80")
        if self.risk_update_interval < 1:
            raise ValueError("risk_update_interval must be at least 1")

        self.time = 0.0
        self.frame_count = 0
        self.paused = False
        self.playback_speed_factor = 1.0
        self._risk_cache: np.ndarray | None = None
        self._hazard_risk_cache: np.ndarray | None = None
        self._risk_dirty = True
        self._interaction_switch_timestamps: list[float] = []
        self._last_interaction_switch_count = 0
        self._path_length = 0.0
        self.invariant_recovery_count = 0
        self._invariant_recovery_active = False
        self._stagnation_active = False
        self._avg_goal_progress = 0.0
        self._invariant_recovery_direction = np.zeros(2, dtype=float)
        self._invariant_recovery_steps_remaining = 0

        self._initial_robot_state = (
            self.robot.position.copy(),
            self.robot.velocity.copy(),
            self.robot.goal.copy(),
        )
        self._reference_path_start = self.robot.position.copy()
        self._reference_path_goal = self.robot.goal.copy()
        self._initial_human_states = [
            (
                human.position.copy(),
                human.velocity.copy(),
                None if human.goal is None else human.goal.copy(),
                human._waypoint_index,
                human.get_rng_state(),
            )
            for human in self.humans
        ]

        self.robot.position = _clamp_position(self.robot.position, self.world_size)
        self.robot.goal = _clamp_position(self.robot.goal, self.world_size)
        for human in self.humans:
            human.position = _clamp_position(human.position, self.world_size)
            if human.goal is not None:
                human.goal = _clamp_position(human.goal, self.world_size)
        self.human_trails = (
            [[human.position.copy()] for human in self.humans] if self.record_human_trails else []
        )

        self.risk_field.world_size = self.world_size.copy()
        self.risk_field.set_goal(self.robot.goal)
        self.risk_field.set_robot_state(
            self.robot.position,
            max(np.linalg.norm(self.robot.velocity), self.robot.speed * self.robot.min_speed_scale, 1e-3),
        )
        self.risk_field.invalidate_cache()
        if self.logger is not None:
            self.logger.reset()
            self.logger.safety_distance = self.robot.safety_distance
            self.logger.path_start = self._reference_path_start.copy()
            self.logger.path_goal = self._reference_path_goal.copy()

    def _min_human_clearance(self) -> float:
        if not self.humans:
            return float("inf")
        return float(
            min(
                np.linalg.norm(self.robot.position - human.position) - (self.robot.radius + human.radius)
                for human in self.humans
            )
        )

    def _collect_log_metrics(
        self,
        previous_heading: np.ndarray,
        previous_position: np.ndarray,
    ) -> dict[str, float | int | str | bool]:
        current_heading = _normalize(self.robot.velocity, self.robot.prev_direction)
        current_goal_dir = _normalize(self.robot.goal - self.robot.position, self.robot.prev_direction)
        current_global_progress, _, _, _, _ = self.robot._current_global_guide_path_state(current_goal_dir)
        heading_change = _signed_angle_between(previous_heading, current_heading)
        step_distance = float(np.linalg.norm(self.robot.position - previous_position))
        speed = float(np.linalg.norm(self.robot.velocity))
        curvature_distance_floor = max(0.05, 0.3 * self.dt * max(self.robot.speed, 1e-6))
        if step_distance > 1e-6 and speed > 1e-6:
            raw_curvature = float(heading_change / max(step_distance, 1e-6))
        else:
            raw_curvature = 0.0
        low_motion_curvature = (
            step_distance < curvature_distance_floor
            or speed < 0.15
        )
        if low_motion_curvature:
            curvature = 0.0
        else:
            curvature = float(
                np.clip(heading_change / max(step_distance, curvature_distance_floor), -5.0, 5.0)
            )
        lateral_deviation = _signed_distance_to_line(
            self.robot.position, self._reference_path_start, self._reference_path_goal
        )
        interaction_effective = float(self.robot.interaction_level)
        min_distance_to_humans = self._min_human_clearance()
        physical_min_clearance = self.robot._compute_environment_clearance_at(
            self.robot.position,
            self.world_size,
            self.humans,
            self.obstacles,
            self.zones,
        )
        boundary_clearance = _boundary_clearance(self.robot.position, self.world_size, self.robot.radius)
        return {
            "step": self.frame_count,
            "x": self.robot.position[0],
            "y": self.robot.position[1],
            "vx": self.robot.velocity[0],
            "vy": self.robot.velocity[1],
            "speed": speed,
            "speed_scale": self.robot.behavior_speed_scale,
            "state": self.robot.behavior_state,
            "behavior_state": self.robot.behavior_state,
            "ttc": self.robot.ttc_min,
            "clearance": self.robot.interaction_clearance,
            "clr_interaction": self.robot.interaction_clearance,
            "clr_global": self.robot.global_clearance,
            "global_clearance": self.robot.global_clearance,
            "clr_rate": self.robot.interaction_clearance_rate,
            "interacting_human_id": self.robot.interacting_human_id,
            "primary_human_id": self.robot.interacting_human_id,
            "interaction_current": self.robot.interaction_level_current,
            "interaction_memory": self.robot.interaction_level_memory,
            "interaction_effective": interaction_effective,
            "interaction_level": interaction_effective,
            "top_1_interaction_strength": self.robot.top_1_interaction_strength,
            "top_2_interaction_strength": self.robot.top_2_interaction_strength,
            "interaction_strength_gap": self.robot.interaction_strength_gap,
            "multi_dominant_interaction": self.robot.multi_dominant_interaction,
            "num_active_interactions": len(self.robot.active_interactions),
            "number_of_active_humans": len(self.robot.active_interactions),
            "active_humans": len(self.robot.active_interactions),
            "min_distance_to_humans": min_distance_to_humans,
            "min_clearance": physical_min_clearance,
            "raw_min_clearance": physical_min_clearance,
            "boundary_clearance": boundary_clearance,
            "global_min_clearance": self.robot.global_min_clearance,
            "interaction_switch_count": self.robot.interaction_switch_count,
            "interaction_switch_timestamps": _format_float_sequence(self._interaction_switch_timestamps),
            "per_human_min_clearances": _format_float_sequence(self.robot.per_human_min_clearances),
            "human_interaction_distances": _format_float_sequence(self.robot.human_interaction_distances),
            "human_interaction_ttc": _format_float_sequence(self.robot.human_interaction_ttc),
            "human_interaction_alignments": _format_float_sequence(self.robot.human_interaction_alignments),
            "human_interaction_scores": _format_float_sequence(self.robot.human_interaction_scores),
            "lateral_deviation": lateral_deviation,
            "heading_change": heading_change,
            "raw_curvature": raw_curvature,
            "curvature": curvature,
            "stop_reason": self.robot.stop_mode_reason,
            "safety_margin": self.robot.safety_margin,
            "risk_slope": self.robot.risk_slope,
            "path_efficiency": current_path_efficiency(
                self._path_length,
                self._reference_path_start,
                self._reference_path_goal,
            ),
            "recovery_time": self.robot.latest_recovery_time,
            "invariant_recovery_active": self._invariant_recovery_active,
            "invariant_recovery_count": self.invariant_recovery_count,
            "stagnation_active": self._stagnation_active,
            "avg_goal_progress": self._avg_goal_progress,
            "global_progress": current_global_progress,
            "guide_progress_max": self.robot.guide_progress_max,
            "failed_branch_count": len(self.robot.failed_branches),
            "goal_distance": float(np.linalg.norm(self.robot.goal - self.robot.position)),
            "detail_active": interaction_effective >= (
                self.logger.detailed_threshold if self.logger is not None else 0.0
            ),
        }

    def enforce_global_progress_post_environment(
        self,
        *,
        baseline_position: np.ndarray,
        baseline_velocity: np.ndarray,
        baseline_direction: np.ndarray,
        previous_position: np.ndarray,
        previous_direction: np.ndarray,
        reference_progress_max: float,
        current_clearance: float,
        current_static_clearance: float,
    ) -> tuple[float, float, bool]:
        if not self.robot.execution_progress_invariant_enabled:
            return current_clearance, current_static_clearance, False
        guide_floor = reference_progress_max - self.robot.guide_progress_regression_tolerance
        current_goal_dir = _normalize(self.robot.goal - self.robot.position, self.robot.prev_direction)
        current_global_progress, _, _, _, _ = self.robot._current_global_guide_path_state(current_goal_dir)
        if (
            np.isfinite(current_global_progress)
            and current_global_progress >= guide_floor - 1e-9
        ):
            return current_clearance, current_static_clearance, False

        def _candidate_state(
            position: np.ndarray,
            direction: np.ndarray,
        ) -> tuple[float, float, float]:
            candidate_direction = _normalize(direction, self.robot.prev_direction)
            candidate_goal_dir = _normalize(self.robot.goal - position, candidate_direction)
            candidate_global_progress, _, _, _, _ = self.robot._global_guide_path_state_at(
                position,
                candidate_goal_dir,
            )
            candidate_clearance = self.robot._compute_environment_clearance_at(
                position,
                self.world_size,
                self.humans,
                self.obstacles,
                self.zones,
            )
            candidate_static_clearance = min(
                self.robot._compute_min_static_clearance_at(
                    position,
                    self.obstacles,
                    self.zones,
                ),
                _boundary_clearance(position, self.world_size, self.robot.radius),
            )
            return candidate_global_progress, candidate_clearance, candidate_static_clearance

        baseline_global_progress, baseline_clearance, baseline_static_clearance = _candidate_state(
            baseline_position,
            baseline_direction,
        )
        if (
            np.isfinite(baseline_global_progress)
            and baseline_global_progress >= guide_floor - 1e-9
            and baseline_clearance >= self.robot.safety_distance - 1e-9
            and baseline_static_clearance >= self.robot.safety_distance - 1e-9
        ):
            self.robot.position = baseline_position.copy()
            self.robot.velocity = baseline_velocity.copy()
            self.robot.prev_direction = _normalize(baseline_direction, self.robot.prev_direction)
            if self.robot.trail:
                self.robot.trail[-1] = self.robot.position.copy()
            return baseline_clearance, baseline_static_clearance, True

        rollback_position = previous_position.copy()
        rollback_direction = _normalize(previous_direction, self.robot.prev_direction)
        rollback_global_progress, rollback_clearance, rollback_static_clearance = _candidate_state(
            rollback_position,
            rollback_direction,
        )
        if (
            np.isfinite(rollback_global_progress)
            and rollback_global_progress >= guide_floor - 1e-9
            and rollback_clearance >= self.robot.safety_distance - 1e-9
            and rollback_static_clearance >= self.robot.safety_distance - 1e-9
        ):
            self.robot.position = rollback_position
            self.robot.velocity[:] = 0.0
            self.robot.prev_direction = rollback_direction
            if self.robot.trail:
                self.robot.trail[-1] = self.robot.position.copy()
            return rollback_clearance, rollback_static_clearance, True

        return current_clearance, current_static_clearance, False

    def step(self) -> None:
        if self.robot.behavior_state == "goal":
            return
        previous_heading = _normalize(self.robot.velocity, self.robot.prev_direction)
        previous_position = self.robot.position.copy()
        previous_guide_progress_max = float(self.robot.guide_progress_max)
        start_step_clearance = self.robot._compute_environment_clearance_at(
            self.robot.position,
            self.world_size,
            self.humans,
            self.obstacles,
            self.zones,
        )
        self.robot.update(self.dt, self.world_size, self.risk_field)
        post_update_position = self.robot.position.copy()
        post_update_velocity = self.robot.velocity.copy()
        post_update_direction = self.robot.prev_direction.copy()
        for human in self.humans:
            human.update(
                self.dt,
                self.world_size,
                blocking_position=self.robot.position,
                blocking_radius=self.robot.radius,
                blocking_clearance=self.robot.safety_distance,
            )
        corrected_static_clearance = min(
            self.robot._compute_min_static_clearance_at(
                self.robot.position,
                self.obstacles,
                self.zones,
            ),
            _boundary_clearance(self.robot.position, self.world_size, self.robot.radius),
        )
        corrected_clearance = self.robot._compute_environment_clearance_at(
            self.robot.position,
            self.world_size,
            self.humans,
            self.obstacles,
            self.zones,
        )

        progress_window = self.robot.goal_progress_window
        if progress_window and len(progress_window) >= progress_window.maxlen:
            avg_progress = float(np.mean(np.asarray(progress_window, dtype=float)))
        else:
            avg_progress = float("inf")
        current_goal_distance = float(np.linalg.norm(self.robot.goal - self.robot.position))
        distance_window = self.robot.goal_distance_window
        distance_array = np.asarray(distance_window, dtype=float) if distance_window else np.empty(0, dtype=float)
        window_progress_delta = (
            float(distance_array[0] - distance_array[-1])
            if distance_array.size >= 2
            else 0.0
        )
        window_progress_variance = (
            float(np.var(distance_array))
            if distance_array.size >= 2
            else 0.0
        )
        meaningful_progress = (
            distance_array.size >= distance_window.maxlen
            and window_progress_delta > self.robot.progress_threshold
        )
        low_variance_plateau = window_progress_variance < self.robot.progress_variance_threshold
        false_progress = (
            distance_array.size >= distance_window.maxlen
            and (
                window_progress_delta < self.robot.progress_threshold
                or ((not meaningful_progress) and low_variance_plateau)
            )
        )
        best_recent_goal_distance = (
            float(np.min(distance_array))
            if distance_window
            else current_goal_distance
        )
        progress_regressing = (
            len(distance_window) >= distance_window.maxlen
            and current_goal_distance > best_recent_goal_distance + self.robot.progress_regression_tolerance
        )
        stagnation = (
            progress_window.maxlen > 0
            and len(progress_window) >= progress_window.maxlen
            and avg_progress < 1e-3
        )
        self._stagnation_active = (
            false_progress
            or stagnation
            or progress_regressing
        )
        self._avg_goal_progress = avg_progress if np.isfinite(avg_progress) else 0.0
        self._invariant_recovery_active = False

        recovery_needed = (
            corrected_clearance < self.robot.safety_distance - 1e-9
            or (
                self.robot.execution_progress_invariant_enabled
                and (false_progress or stagnation or progress_regressing)
                and not self.robot.detour_active
            )
        )
        if recovery_needed:
            recovery_velocity = self.robot.velocity.copy()
            if false_progress or stagnation or progress_regressing:
                goal_dir = _normalize(self.robot.goal - self.robot.position, self.robot.prev_direction)
                min_escape_speed = 0.06
                if (
                    self._invariant_recovery_steps_remaining > 0
                    and np.linalg.norm(self._invariant_recovery_direction) > 1e-9
                ):
                    recovery_direction = self._invariant_recovery_direction.copy()
                    self._invariant_recovery_steps_remaining -= 1
                else:
                    recovery_direction, _ = _select_committed_detour_direction(
                        position=self.robot.position,
                        guide_direction=goal_dir,
                        static_gradient=self.risk_field.compute_static_gradient(self.robot.position),
                        previous_direction=self.robot.prev_direction,
                        escape_gain=self.robot.escape_gain,
                        target_speed=min_escape_speed,
                        min_progress_speed=min_escape_speed,
                        dt=self.dt,
                        safety_distance=self.robot.safety_distance,
                        clearance_fn=lambda candidate_position: self.robot._compute_environment_clearance_at(
                            candidate_position,
                            self.world_size,
                            self.humans,
                            self.obstacles,
                            self.zones,
                        ),
                        lookahead_risk_fn=lambda candidate_position: self.risk_field.compute_hazard_risk(
                            candidate_position
                        ),
                        lookahead_step=self.robot.speed * self.dt,
                        progress_value=avg_progress,
                        progress_epsilon=1e-3,
                        progress_timer=self.robot.progress_stall_timer,
                        progress_timer_threshold=0.8,
                        detour_side=self.robot.detour_side,
                    )
                    if recovery_direction is None:
                        tangent = _perpendicular(goal_dir)
                        if float(np.dot(tangent, self.robot.prev_direction)) < 0.0:
                            tangent *= -1.0
                        recovery_direction = _normalize(tangent, goal_dir)
                    if float(np.dot(recovery_direction, goal_dir)) <= -0.8:
                        tangent = _perpendicular(goal_dir)
                        if float(np.dot(tangent, self.robot.prev_direction)) < 0.0:
                            tangent *= -1.0
                        recovery_direction = _normalize(tangent, goal_dir)
                    if float(np.dot(recovery_direction, self.robot.prev_direction)) < -0.3:
                        recovery_direction = _normalize(self.robot.prev_direction, goal_dir)
                    self._invariant_recovery_direction = recovery_direction.copy()
                    self._invariant_recovery_steps_remaining = max(
                        self.robot.escape_commit_steps,
                        self.robot.escape_flip_lock_steps,
                        12,
                    )
                recovery_velocity = recovery_direction * min_escape_speed
            else:
                self._invariant_recovery_direction[:] = 0.0
                self._invariant_recovery_steps_remaining = 0

            projected_position, projected_static_clearance = self.robot._project_safe_position(
                self.robot.position + recovery_velocity * self.dt,
                self.world_size,
                self.humans,
                self.obstacles,
                self.zones,
                fallback_direction=_normalize(recovery_velocity, self.robot.prev_direction),
            )
            projected_velocity = (projected_position - self.robot.position) / max(self.dt, 1e-6)
            projected_clearance = self.robot._compute_environment_clearance_at(
                self.robot.position + projected_velocity * self.dt,
                self.world_size,
                self.humans,
                self.obstacles,
                self.zones,
            )
            if projected_clearance >= self.robot.safety_distance - 1e-9:
                displacement = projected_position - self.robot.position
                self.robot.position = projected_position
                self.robot.velocity = projected_velocity
                self.robot.prev_direction = _normalize(self.robot.velocity, self.robot.prev_direction)
                if np.linalg.norm(displacement) > 1e-9 and self.robot.trail:
                    self.robot.trail[-1] = self.robot.position.copy()
                corrected_clearance = projected_clearance
                corrected_static_clearance = min(
                    projected_static_clearance,
                    _boundary_clearance(self.robot.position, self.world_size, self.robot.radius),
                )
                self._invariant_recovery_active = True
                self.invariant_recovery_count += 1
            else:
                hold_clearance = self.robot._compute_environment_clearance_at(
                    self.robot.position,
                    self.world_size,
                    self.humans,
                    self.obstacles,
                    self.zones,
                )
                if hold_clearance >= self.robot.safety_distance - 1e-9:
                    self.robot.velocity[:] = 0.0
                    corrected_clearance = hold_clearance
                    corrected_static_clearance = min(
                        self.robot._compute_min_static_clearance_at(
                            self.robot.position,
                            self.obstacles,
                            self.zones,
                        ),
                        _boundary_clearance(self.robot.position, self.world_size, self.robot.radius),
                    )
                    self._invariant_recovery_active = True
                    self.invariant_recovery_count += 1
        if not (false_progress or stagnation or progress_regressing):
            self._invariant_recovery_direction[:] = 0.0
            self._invariant_recovery_steps_remaining = 0
        if (
            start_step_clearance >= self.robot.safety_distance - 1e-6
            and corrected_clearance < self.robot.safety_distance - 1e-6
        ):
            self.robot.position = previous_position.copy()
            self.robot.velocity[:] = 0.0
            corrected_clearance = self.robot._compute_environment_clearance_at(
                self.robot.position,
                self.world_size,
                self.humans,
                self.obstacles,
                self.zones,
            )
            corrected_static_clearance = min(
                self.robot._compute_min_static_clearance_at(
                    self.robot.position,
                    self.obstacles,
                    self.zones,
                ),
                _boundary_clearance(self.robot.position, self.world_size, self.robot.radius),
            )
            self._invariant_recovery_active = True
            self.invariant_recovery_count += 1
        corrected_clearance, corrected_static_clearance, progress_rollback = (
            self.enforce_global_progress_post_environment(
                baseline_position=post_update_position,
                baseline_velocity=post_update_velocity,
                baseline_direction=post_update_direction,
                previous_position=previous_position,
                previous_direction=previous_heading,
                reference_progress_max=previous_guide_progress_max,
                current_clearance=corrected_clearance,
                current_static_clearance=corrected_static_clearance,
            )
        )
        if progress_rollback:
            self._invariant_recovery_active = False
        final_goal_dir = _normalize(self.robot.goal - self.robot.position, self.robot.prev_direction)
        final_global_progress, _, _, _, _ = self.robot._current_global_guide_path_state(final_goal_dir)
        if np.isfinite(final_global_progress):
            self.robot.guide_progress_max = max(previous_guide_progress_max, final_global_progress)
        else:
            self.robot.guide_progress_max = previous_guide_progress_max
        self._path_length += float(np.linalg.norm(self.robot.position - previous_position))
        self.robot.physical_clearance = corrected_clearance
        self.robot.minimum_physical_clearance = min(
            self.robot.minimum_physical_clearance,
            corrected_clearance,
        )
        self.robot.safety_margin = min(
            self.robot.safety_margin,
            corrected_clearance - self.robot.safety_distance,
            corrected_static_clearance - self.robot.safety_distance,
        )
        if self.record_human_trails:
            for trail, human in zip(self.human_trails, self.humans):
                trail.append(human.position.copy())
        self.risk_field.set_robot_state(
            self.robot.position,
            max(np.linalg.norm(self.robot.velocity), self.robot.speed * self.robot.min_speed_scale, 1e-3),
        )
        self.risk_field.invalidate_cache()
        self.time += self.dt
        self.frame_count += 1
        if self.robot.interaction_switch_count > self._last_interaction_switch_count:
            self._interaction_switch_timestamps.append(self.time)
            self._last_interaction_switch_count = self.robot.interaction_switch_count
        self._risk_dirty = True
        if self.logger is not None:
            self.logger.log(self.time, self.robot, self._collect_log_metrics(previous_heading, previous_position))

    def run(self, max_steps: int | None = None) -> None:
        completed_steps = 0
        while self.robot.behavior_state != "goal":
            if max_steps is not None and completed_steps >= max_steps:
                break
            self.step()
            completed_steps += 1

    def reset(self) -> None:
        position, velocity, goal = self._initial_robot_state
        self.robot.position = position.copy()
        self.robot.velocity = velocity.copy()
        self.robot.goal = goal.copy()
        self.robot.trail = [self.robot.position.copy()]

        for human, (human_position, human_velocity, human_goal, human_waypoint_index, human_rng_state) in zip(
            self.humans, self._initial_human_states
        ):
            human.position = human_position.copy()
            human.velocity = human_velocity.copy()
            human.goal = None if human_goal is None else human_goal.copy()
            if human.goal is not None:
                human.goal = _clamp_position(human.goal, self.world_size)
            human._waypoint_index = int(human_waypoint_index)
            human.set_rng_state(human_rng_state)

        self.robot.reset_control_state()
        self.risk_field.set_goal(self.robot.goal)
        self.risk_field.set_robot_state(
            self.robot.position,
            max(np.linalg.norm(self.robot.velocity), self.robot.speed * self.robot.min_speed_scale, 1e-3),
        )
        self.risk_field.invalidate_cache()
        self.time = 0.0
        self.frame_count = 0
        self._risk_cache = None
        self._hazard_risk_cache = None
        self._risk_dirty = True
        self._interaction_switch_timestamps = []
        self._last_interaction_switch_count = 0
        self._path_length = 0.0
        self.invariant_recovery_count = 0
        self._invariant_recovery_active = False
        self._stagnation_active = False
        self._avg_goal_progress = 0.0
        self._invariant_recovery_direction = np.zeros(2, dtype=float)
        self._invariant_recovery_steps_remaining = 0
        self.human_trails = (
            [[human.position.copy()] for human in self.humans] if self.record_human_trails else []
        )
        if self.logger is not None:
            self.logger.reset()
            self.logger.safety_distance = self.robot.safety_distance
            self.logger.path_start = self._reference_path_start.copy()
            self.logger.path_goal = self._reference_path_goal.copy()

    def save_logs(self, path: str = "logs.csv") -> None:
        if self.logger is not None:
            self.logger.save(path)

    def print_log_summary(self, step_interval: int | None = None) -> None:
        if self.logger is not None:
            self.logger.summary(step_interval)

    def validation_metrics(self) -> dict[str, float]:
        if self.logger is None:
            return {}
        return self.logger.validation_metrics()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def toggle_risk(self) -> None:
        self.show_risk = not self.show_risk
        self._risk_dirty = True

    def toggle_entity_risk_fields(self) -> None:
        self.show_entity_risk_fields = not self.show_entity_risk_fields

    def toggle_debug(self) -> None:
        self.show_debug = not self.show_debug

    def _apply_playback_speed(self) -> None:
        animation_obj = getattr(self, "_animation", None)
        event_source = getattr(animation_obj, "event_source", None)
        if event_source is None:
            return

        base_interval_ms = self.dt * 1000.0 * self.render_skip if self.real_time else 1.0
        event_source.interval = max(int(round(base_interval_ms / self.playback_speed_factor)), 1)

    def adjust_playback_speed(self, *, faster: bool) -> None:
        if faster:
            self.playback_speed_factor = min(4.0, self.playback_speed_factor * 1.25)
        else:
            self.playback_speed_factor = max(0.25, self.playback_speed_factor / 1.25)
        self._apply_playback_speed()

    def animate(self) -> None:
        _ensure_writable_matplotlib_config()
        try:
            import matplotlib.animation as animation
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "matplotlib is required to animate the simulation. Install dependencies in a "
                "virtual environment, for example: python3 -m venv .venv && .venv/bin/pip "
                "install numpy matplotlib"
            ) from exc

        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(
                f"Interaction-Aware Predictive Risk Navigation - {self.scenario_name}"
            )

        ax.set_xlim(0.0, self.world_size[0])
        ax.set_ylim(0.0, self.world_size[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Interaction-Aware Predictive Social Navigation - {self.scenario_name}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        grid_x, grid_y = self.risk_field.make_grid(self.risk_resolution)
        initial_hazard_risk, initial_risk = self.risk_field.compute_grid_layers(grid_x, grid_y)
        self._hazard_risk_cache = initial_hazard_risk
        self._risk_cache = initial_risk
        field_extent = (0.0, self.world_size[0], 0.0, self.world_size[1])
        risk_image = ax.imshow(
            initial_risk,
            extent=field_extent,
            origin="lower",
            cmap="magma",
            alpha=0.22,
            interpolation="bilinear",
            zorder=0,
            visible=self.show_risk,
        )
        risk_min = float(np.min(initial_risk))
        risk_max = float(np.max(initial_risk))
        if risk_max - risk_min < 1e-9:
            risk_max = risk_min + 1e-9
        risk_image.set_clim(risk_min, risk_max)

        def field_to_rgba(
            field: np.ndarray,
            high_rgb: tuple[float, float, float],
            mid_rgb: tuple[float, float, float],
            *,
            max_alpha: float,
        ) -> np.ndarray:
            clipped = np.clip(field, 0.0, 1.0)
            mix = np.clip((clipped - 0.45) / 0.55, 0.0, 1.0)[..., None]
            high = np.asarray(high_rgb, dtype=float)
            mid = np.asarray(mid_rgb, dtype=float)
            rgba = np.zeros(clipped.shape + (4,), dtype=float)
            rgba[..., :3] = (1.0 - mix) * mid + mix * high
            rgba[..., 3] = max_alpha * np.clip((clipped - 0.18) / 0.82, 0.0, 1.0)
            return rgba

        def alpha_composite(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
            result = np.zeros_like(base)
            base_alpha = base[..., 3]
            overlay_alpha = overlay[..., 3]
            out_alpha = overlay_alpha + base_alpha * (1.0 - overlay_alpha)
            numerator = (
                overlay[..., :3] * overlay_alpha[..., None]
                + base[..., :3] * base_alpha[..., None] * (1.0 - overlay_alpha[..., None])
            )
            np.divide(
                numerator,
                out_alpha[..., None],
                out=result[..., :3],
                where=out_alpha[..., None] > 1e-9,
            )
            result[..., 3] = out_alpha
            return result

        human_high = (0.86, 0.12, 0.14)
        human_mid = (1.0, 0.55, 0.0)
        obstacle_high = (0.10, 0.35, 0.90)
        obstacle_mid = (0.0, 0.85, 0.95)
        zone_high = (0.45, 0.12, 0.72)
        zone_mid = (0.95, 0.10, 0.70)
        static_entity_rgba = np.zeros(grid_x.shape + (4,), dtype=float)
        static_entity_rgba = alpha_composite(
            static_entity_rgba,
            field_to_rgba(
                self.risk_field.zone_visual_risk_grid_all(grid_x, grid_y),
                zone_high,
                zone_mid,
                max_alpha=0.44,
            ),
        )
        static_entity_rgba = alpha_composite(
            static_entity_rgba,
            field_to_rgba(
                self.risk_field.obstacle_visual_risk_grid_all(grid_x, grid_y),
                obstacle_high,
                obstacle_mid,
                max_alpha=0.42,
            ),
        )
        entity_field_image = ax.imshow(
            static_entity_rgba,
            extent=field_extent,
            origin="lower",
            interpolation="bilinear",
            zorder=1.35,
            visible=self.show_entity_risk_fields,
        )
        entity_field_state = {"frame": -1}
        risk_overlay_state = {"last_frame": -self.risk_update_interval}
        nearby_radius = max(
            1.25 * self.robot.safe_distance,
            self.risk_field.sigma_parallel,
            self.risk_field.sigma_obs,
            self.risk_field.sigma_zone,
        )
        nearby_radius_sq = nearby_radius**2

        for index, obstacle in enumerate(self.obstacles):
            patch = obstacle.create_patch()
            if index == 0:
                patch.set_label("Obstacle")
            patch.set_zorder(4)
            ax.add_patch(patch)
        for index, zone in enumerate(self.zones):
            patch = zone.create_patch()
            if index == 0:
                patch.set_label("No-go zone")
            patch.set_zorder(4)
            ax.add_patch(patch)

        ax.scatter(
            self.robot.goal[0],
            self.robot.goal[1],
            s=180,
            c="tab:green",
            marker="*",
            label="Goal",
            zorder=4,
        )
        robot_artist = ax.scatter(
            self.robot.position[0],
            self.robot.position[1],
            s=160,
            c="tab:blue",
            edgecolors="black",
            linewidths=0.5,
            label="Robot",
            zorder=5,
        )
        human_artist = ax.scatter(
            [human.position[0] for human in self.humans],
            [human.position[1] for human in self.humans],
            s=120,
            c="tab:red",
            edgecolors="black",
            linewidths=0.5,
            label="Humans",
            zorder=5,
        )
        primary_human_artist = ax.scatter(
            [],
            [],
            s=260,
            facecolors="none",
            edgecolors="gold",
            linewidths=2.2,
            label="Primary human",
            zorder=8,
        )
        human_id_texts = [
            ax.text(
                0.0,
                0.0,
                str(index),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "tab:red", "alpha": 0.75, "edgecolor": "none", "pad": 0.2},
                visible=False,
                zorder=8,
            )
            for index in range(len(self.humans))
        ]
        prediction_lines = [
            ax.plot(
                [],
                [],
                linestyle="--",
                linewidth=1.0,
                color="tab:red",
                alpha=0.25,
                zorder=2,
            )[0]
            for _ in self.humans
        ]
        interaction_lines = [
            ax.plot(
                [],
                [],
                color="gold",
                linewidth=1.6,
                alpha=0.85,
                zorder=7,
            )[0]
            for _ in self.humans
        ]

        robot_trail, = ax.plot([], [], color="tab:blue", linewidth=1.6, alpha=0.8, zorder=3)
        robot_velocity = ax.quiver(
            [self.robot.position[0]],
            [self.robot.position[1]],
            [self.robot.velocity[0]],
            [self.robot.velocity[1]],
            color="tab:blue",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            zorder=6,
        )
        human_velocity = ax.quiver(
            [human.position[0] for human in self.humans],
            [human.position[1] for human in self.humans],
            [human.velocity[0] for human in self.humans],
            [human.velocity[1] for human in self.humans],
            color="tab:red",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            zorder=6,
        )
        debug_range = max(
            2.5,
            self.robot.speed * max(self.robot.ttc_attention_threshold, self.risk_field.prediction_horizon),
        )
        cone_half_angle = float(
            np.degrees(np.arccos(np.clip(self.robot.interaction_forward_threshold, -1.0, 1.0)))
        )
        forward_cone = mpatches.Wedge(
            center=tuple(self.robot.position),
            r=debug_range,
            theta1=-cone_half_angle,
            theta2=cone_half_angle,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.08,
            linewidth=1.0,
            linestyle=":",
            zorder=1,
        )
        forward_cone.set_visible(self.show_debug)
        lateral_band = mpatches.Polygon(
            np.zeros((4, 2), dtype=float),
            closed=True,
            facecolor="tab:cyan",
            edgecolor="tab:cyan",
            alpha=0.08,
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        lateral_band.set_visible(self.show_debug)
        ax.add_patch(forward_cone)
        ax.add_patch(lateral_band)
        ttc_active_artist = ax.scatter(
            [],
            [],
            s=220,
            facecolors="none",
            edgecolors="gold",
            linewidths=1.8,
            label="TTC-active",
            zorder=7,
        )
        ttc_active_artist.set_visible(self.show_debug)
        ttc_texts = [
            ax.text(
                0.0,
                0.0,
                "",
                color="goldenrod",
                fontsize=8,
                ha="left",
                va="bottom",
                visible=False,
                zorder=8,
            )
            for _ in self.humans
        ]
        status_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            zorder=7,
        )

        ax.legend(loc="upper right")

        def refresh_risk_overlay(force: bool = False) -> None:
            if not self.show_risk and not force:
                risk_image.set_visible(False)
                return
            if (
                force
                or self._risk_cache is None
                or self._hazard_risk_cache is None
                or (
                    self._risk_dirty
                    and self.frame_count - risk_overlay_state["last_frame"] >= self.risk_update_interval
                )
            ):
                hazard_grid, risk_grid = self.risk_field.compute_grid_layers(grid_x, grid_y)
                self._hazard_risk_cache = hazard_grid
                self._risk_cache = risk_grid
                self._risk_dirty = False
                risk_overlay_state["last_frame"] = self.frame_count
                risk_image.set_data(risk_grid)
                risk_min = float(np.min(risk_grid))
                risk_max = float(np.max(risk_grid))
                if risk_max - risk_min < 1e-9:
                    risk_max = risk_min + 1e-9
                risk_image.set_clim(risk_min, risk_max)

            risk_image.set_visible(self.show_risk)

        def refresh_entity_risk_fields(force: bool = False) -> None:
            if not self.show_entity_risk_fields:
                entity_field_image.set_visible(False)
                return

            if not force and entity_field_state["frame"] == self.frame_count:
                entity_field_image.set_visible(True)
                return

            composite_rgba = static_entity_rgba
            if self.humans:
                human_rgba = field_to_rgba(
                    self.risk_field.human_visual_risk_grid_all(grid_x, grid_y),
                    human_high,
                    human_mid,
                    max_alpha=0.48,
                )
                composite_rgba = alpha_composite(static_entity_rgba, human_rgba)
            entity_field_image.set_data(composite_rgba)
            entity_field_image.set_visible(True)
            entity_field_state["frame"] = self.frame_count

        def compute_debug_risk_metrics() -> tuple[float, float, str]:
            breakdown = self.risk_field.hazard_breakdown(self.robot.position)
            source_label = "none"
            source_value = 0.0
            for prefix, values in (
                ("human", breakdown["humans"]),
                ("obstacle", breakdown["obstacles"]),
                ("zone", breakdown["zones"]),
            ):
                for index, value in enumerate(values):
                    if value > source_value:
                        source_value = value
                        source_label = f"{prefix} {index}"

            robot_risk = float(sum(sum(values) for values in breakdown.values()))
            if self._hazard_risk_cache is None:
                return robot_risk, robot_risk, source_label

            local_distance_sq = (
                (grid_x - self.robot.position[0]) ** 2 + (grid_y - self.robot.position[1]) ** 2
            )
            nearby_mask = local_distance_sq <= nearby_radius_sq
            if not np.any(nearby_mask):
                return robot_risk, robot_risk, source_label

            nearby_risk = float(np.max(self._hazard_risk_cache[nearby_mask]))
            return robot_risk, nearby_risk, source_label

        def render_scene(force: bool = False):
            refresh_risk_overlay(force=force)
            refresh_entity_risk_fields(force=force)

            robot_artist.set_offsets(self.robot.position[None, :])
            human_positions = np.array([human.position for human in self.humans], dtype=float)
            if human_positions.size == 0:
                human_positions = np.empty((0, 2), dtype=float)
            human_artist.set_offsets(human_positions)
            primary_index = self.robot.interacting_human_id
            if 0 <= primary_index < len(self.humans):
                primary_human_artist.set_offsets(human_positions[primary_index][None, :])
                primary_human_artist.set_visible(True)
            else:
                primary_human_artist.set_offsets(np.empty((0, 2), dtype=float))
                primary_human_artist.set_visible(False)
            for index, label in enumerate(human_id_texts):
                if self.show_debug and index < len(self.humans):
                    label.set_position(self.humans[index].position + np.array([0.0, 0.18], dtype=float))
                    label.set_visible(True)
                else:
                    label.set_visible(False)

            trail = np.array(self.robot.trail, dtype=float)
            robot_trail.set_data(trail[:, 0], trail[:, 1])

            predicted_paths = self.risk_field.get_predicted_trajectories()
            for line, path in zip(prediction_lines, predicted_paths):
                line.set_data(path[:, 0], path[:, 1])

            robot_velocity.set_offsets(self.robot.position[None, :])
            robot_velocity.set_UVC([self.robot.velocity[0]], [self.robot.velocity[1]])
            human_velocity.set_offsets(human_positions)
            human_velocity.set_UVC(
                [human.velocity[0] for human in self.humans],
                [human.velocity[1] for human in self.humans],
            )

            debug_direction = _normalize(self.robot.prev_direction, self.robot.goal - self.robot.position)
            heading_angle = float(np.degrees(np.arctan2(debug_direction[1], debug_direction[0])))
            forward_cone.set_center(tuple(self.robot.position))
            forward_cone.set_radius(debug_range)
            forward_cone.theta1 = heading_angle - cone_half_angle
            forward_cone.theta2 = heading_angle + cone_half_angle
            forward_cone.set_visible(self.show_debug)

            perpendicular = np.array([-debug_direction[1], debug_direction[0]], dtype=float)
            band_start = self.robot.position
            band_end = self.robot.position + debug_direction * debug_range
            band_half_width = self.robot.lateral_safe_distance
            band_points = np.vstack(
                [
                    band_start + perpendicular * band_half_width,
                    band_end + perpendicular * band_half_width,
                    band_end - perpendicular * band_half_width,
                    band_start - perpendicular * band_half_width,
                ]
            )
            lateral_band.set_xy(band_points)
            lateral_band.set_visible(self.show_debug)

            active_indices = [interaction.human_index for interaction in self.robot.active_interactions]
            if active_indices:
                ttc_active_artist.set_offsets(human_positions[active_indices])
            else:
                ttc_active_artist.set_offsets(np.empty((0, 2), dtype=float))
            ttc_active_artist.set_visible(self.show_debug and bool(active_indices))

            for text in ttc_texts:
                text.set_visible(False)
            for line in interaction_lines:
                line.set_data([], [])
                line.set_visible(False)
            if self.show_debug:
                for interaction in self.robot.active_interactions:
                    human_position = self.humans[interaction.human_index].position
                    label = ttc_texts[interaction.human_index]
                    label.set_position(human_position + np.array([0.1, 0.1], dtype=float))
                    label.set_text(f"{interaction.ttc:.1f}s")
                    label.set_visible(True)

                    line = interaction_lines[interaction.human_index]
                    line_start = self.robot.position
                    line_end = line_start + 0.9 * interaction.interaction_vector
                    line.set_data(
                        [line_start[0], line_end[0]],
                        [line_start[1], line_end[1]],
                    )
                    line.set_visible(True)

            state = "paused" if self.paused else "running"
            risk_state = "on" if self.show_risk else "off"
            field_state = "on" if self.show_entity_risk_fields else "off"
            debug_state = "on" if self.show_debug else "off"
            ttc_text = "--" if not np.isfinite(self.robot.ttc_min) else f"{self.robot.ttc_min:3.1f}s"
            interaction_clearance_text = (
                "--"
                if not np.isfinite(self.robot.interaction_clearance)
                else f"{self.robot.interaction_clearance:3.1f}m"
            )
            global_clearance_text = (
                "--"
                if not np.isfinite(self.robot.global_clearance)
                else f"{self.robot.global_clearance:3.1f}m"
            )
            human_id_text = "--" if self.robot.interacting_human_id < 0 else str(self.robot.interacting_human_id)
            robot_risk, nearby_risk, dominant_source = compute_debug_risk_metrics()
            status_text.set_text(
                f"scenario = {self.scenario_name}\nt = {self.time:4.1f} s\nstate = {state}\n"
                f"risk = {risk_state} | fields = {field_state} | debug = {debug_state}\n"
                f"play = x{self.playback_speed_factor:0.2f} | rt = {'on' if self.real_time else 'off'} | "
                f"skip = {self.render_skip}\n"
                f"robot = {self.robot.behavior_state} ({self.robot.behavior_speed_scale:0.2f}x)\n"
                f"ttc = {ttc_text} | clr_i = {interaction_clearance_text} | "
                f"clr_g = {global_clearance_text} | hid = {human_id_text}\n"
                f"active_h = {len(active_indices)} | int = {self.robot.interaction_level:0.2f}\n"
                f"risk_r = {robot_risk:0.2f} | local_max = {nearby_risk:0.2f}\n"
                f"source = {dominant_source}\n"
                f"top2 = {self.robot.top_1_interaction_strength:0.2f}/"
                f"{self.robot.top_2_interaction_strength:0.2f} | "
                f"gap = {self.robot.interaction_strength_gap:0.2f}\n"
                f"pred = {self.risk_field.prediction_horizon:.1f}s horizon\n"
                "keys: space pause | r reset | [ slower | ] faster | v risk | f fields | d debug"
            )
            return (
                risk_image,
                entity_field_image,
                robot_artist,
                human_artist,
                primary_human_artist,
                forward_cone,
                lateral_band,
                ttc_active_artist,
                *prediction_lines,
                *interaction_lines,
                robot_trail,
                robot_velocity,
                human_velocity,
                status_text,
                *human_id_texts,
                *ttc_texts,
            )

        def update(_frame: int):
            if not self.paused:
                for _ in range(self.render_skip):
                    if self.robot.behavior_state == "goal":
                        break
                    self.step()

            return render_scene()

        def on_key_press(event) -> None:
            if event.key == " ":
                self.toggle_pause()
            elif event.key == "r":
                self.reset()
            elif event.key == "[":
                self.adjust_playback_speed(faster=False)
            elif event.key == "]":
                self.adjust_playback_speed(faster=True)
            elif event.key == "v":
                self.toggle_risk()
            elif event.key == "f":
                self.toggle_entity_risk_fields()
            elif event.key == "d":
                self.toggle_debug()
            else:
                return

            render_scene(force=True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key_press)
        render_scene(force=True)
        self._animation = animation.FuncAnimation(
            fig,
            update,
            interval=1,
            blit=False,
            cache_frame_data=False,
        )
        self._apply_playback_speed()
        plt.show()


def _make_goal_human(
    position: Sequence[float],
    goal: Sequence[float],
    preferred_speed: float,
    *,
    noise_std: float,
    velocity_smoothing: float,
    goal_tolerance: float,
    rng_seed: int,
    radius: float = 0.18,
    loop_between_goals: bool = True,
) -> Human:
    position_array = np.asarray(position, dtype=float)
    goal_array = np.asarray(goal, dtype=float)
    velocity = preferred_speed * _normalize(goal_array - position_array, goal_array - position_array)
    return Human(
        position=position_array,
        velocity=velocity,
        goal=goal_array,
        waypoints=(goal_array.copy(), position_array.copy()) if loop_between_goals else None,
        radius=radius,
        preferred_speed=preferred_speed,
        noise_std=noise_std,
        velocity_smoothing=velocity_smoothing,
        goal_tolerance=goal_tolerance,
        rng_seed=rng_seed,
    )


def _sample_group_preferred_speed(rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(1.2, 0.2), 0.75, 1.4))


def _make_goal_human_group(
    start_center: Sequence[float],
    goal_center: Sequence[float],
    count: int,
    *,
    group_seed: int,
    noise_std: float,
    velocity_smoothing: float,
    goal_tolerance: float,
    lateral_spacing: float = 0.32,
    longitudinal_spacing: float = 0.18,
    radius: float = 0.18,
) -> list[Human]:
    start_array = np.asarray(start_center, dtype=float)
    goal_array = np.asarray(goal_center, dtype=float)
    count = max(int(count), 1)
    direction = _normalize(goal_array - start_array, np.array([1.0, 0.0], dtype=float))
    lateral_dir = np.array([-direction[1], direction[0]], dtype=float)
    rng = np.random.default_rng(int(group_seed))
    base_speed = _sample_group_preferred_speed(rng)
    group_noise = max(float(noise_std), 0.015)
    humans: list[Human] = []
    centered_indices = np.arange(count, dtype=float) - 0.5 * float(count - 1)

    for index, centered_index in enumerate(centered_indices):
        lateral_offset = centered_index * lateral_spacing
        longitudinal_offset = 0.0
        if count > 1:
            longitudinal_offset = (0.5 if index % 2 else -0.5) * longitudinal_spacing
        offset = lateral_offset * lateral_dir + longitudinal_offset * direction
        preferred_speed = float(np.clip(base_speed + rng.normal(0.0, 0.05), 0.75, 1.4))
        humans.append(
            _make_goal_human(
                start_array + offset,
                goal_array + offset,
                preferred_speed,
                noise_std=group_noise,
                velocity_smoothing=velocity_smoothing,
                goal_tolerance=goal_tolerance,
                rng_seed=int(group_seed + 37 * (index + 1)),
                radius=radius,
            )
        )
    return humans


def build_scenario(
    type: str,
    *,
    world_size: float = 10.0,
    human_noise_std: float = 0.05,
    human_velocity_smoothing: float = 0.8,
    human_goal_tolerance: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[Human], list[Obstacle], list[NoGoZone]]:
    size = _validate_positive(world_size, "world_size")
    scale = size / 10.0

    def point(x: float, y: float) -> np.ndarray:
        return np.array([x, y], dtype=float) * scale

    scenario = type.lower()
    if scenario in {"crossing", "two-crossing", "two_crossing", "crossing_flow", "crossing-flow"}:
        scenario = "crossing_flow"
    elif scenario in {"multi_human_crossing", "multi-human-crossing"}:
        scenario = "crossing_flow"
    elif scenario in {"crossing_humans", "crossing-humans"}:
        scenario = "crossing_flow"
    elif scenario in {"head_on_interaction", "head-on-interaction"}:
        scenario = "head_on"
    elif scenario in {"mixed_scenario", "mixed-scenario"}:
        scenario = "mixed_scenario"
    elif scenario in {"narrow_passage", "narrow-passage"}:
        scenario = "narrow_passage"
    elif scenario in {"corridor_trap", "corridor-trap", "double_corridor_trap", "double-corridor-trap"}:
        scenario = "corridor_trap"
    elif scenario in {"random_crowd", "random-crowd"}:
        scenario = "random_crowd"
    elif scenario in {"structured_crowd", "structured-crowd"}:
        scenario = "structured_crowd"
    elif scenario in {"stress_crowd", "stress-crowd", "adversarial_crowd", "adversarial-crowd"}:
        scenario = "stress_crowd"
    elif scenario in {"permanent_blocking", "permanent-blocking", "adversarial_blocking", "adversarial-blocking"}:
        scenario = "permanent_blocking"
    seed_offset = int(seed)

    def seeded(local_seed: int) -> int:
        return seed_offset + int(local_seed)

    robot_start = point(1.0, 5.0)
    robot_goal = point(9.0, 5.0)
    crossing_extent = 0.36 * size
    corridor_bias = 0.12 * size
    humans: list[Human]
    obstacles: list[Obstacle] = []
    zones: list[NoGoZone] = []

    def make_path_group_builders(
        path_start: np.ndarray,
        path_goal: np.ndarray,
    ) -> tuple[
        Callable[[float, float], np.ndarray],
        Callable[[float, float, int], list[Human]],
        Callable[[float, float, int], list[Human]],
        Callable[[float, float, float, float, int], list[Human]],
    ]:
        path_vector = path_goal - path_start
        path_dir = _normalize(path_vector, np.array([1.0, 0.0], dtype=float))
        path_normal = np.array([-path_dir[1], path_dir[0]], dtype=float)

        def corridor_point(progress: float, lateral_offset: float = 0.0) -> np.ndarray:
            clamped_progress = float(np.clip(progress, 0.0, 1.0))
            return (
                path_start
                + clamped_progress * path_vector
                + lateral_offset * path_normal
            )

        def crossing_group(
            progress: float,
            direction_sign: float,
            count: int,
            *,
            corridor_offset: float = 0.0,
            group_seed: int,
        ) -> list[Human]:
            center = corridor_point(progress, corridor_offset)
            start_center = center + direction_sign * crossing_extent * path_normal
            goal_center = center - direction_sign * crossing_extent * path_normal
            return _make_goal_human_group(
                start_center,
                goal_center,
                count,
                group_seed=seeded(group_seed),
                noise_std=human_noise_std,
                velocity_smoothing=human_velocity_smoothing,
                goal_tolerance=human_goal_tolerance,
                lateral_spacing=0.28 * scale,
                longitudinal_spacing=0.22 * scale,
            )

        def head_on_group(
            start_progress: float,
            goal_progress: float,
            count: int,
            *,
            lateral_offset: float = 0.0,
            goal_lateral_offset: float | None = None,
            group_seed: int,
        ) -> list[Human]:
            if goal_lateral_offset is None:
                goal_lateral_offset = lateral_offset
            return _make_goal_human_group(
                corridor_point(start_progress, lateral_offset),
                corridor_point(goal_progress, goal_lateral_offset),
                count,
                group_seed=seeded(group_seed),
                noise_std=human_noise_std,
                velocity_smoothing=human_velocity_smoothing,
                goal_tolerance=human_goal_tolerance,
                lateral_spacing=0.26 * scale,
                longitudinal_spacing=0.24 * scale,
            )

        def diagonal_group(
            start_progress: float,
            start_lateral: float,
            goal_progress: float,
            goal_lateral: float,
            count: int,
            *,
            group_seed: int,
        ) -> list[Human]:
            return _make_goal_human_group(
                corridor_point(start_progress, start_lateral),
                corridor_point(goal_progress, goal_lateral),
                count,
                group_seed=seeded(group_seed),
                noise_std=human_noise_std,
                velocity_smoothing=human_velocity_smoothing,
                goal_tolerance=human_goal_tolerance,
                lateral_spacing=0.24 * scale,
                longitudinal_spacing=0.18 * scale,
            )

        return corridor_point, crossing_group, head_on_group, diagonal_group

    corridor_point, crossing_group, head_on_group, diagonal_group = make_path_group_builders(
        robot_start,
        robot_goal,
    )

    if scenario == "empty":
        humans = []
    elif scenario == "zone_only":
        humans = []
        zones = [
            NoGoZone(kind="rectangle", center=point(5.3, 4.0), size=np.array([1.5, 2.0]) * scale),
            NoGoZone(kind="rectangle", center=point(7.3, 3.1), size=np.array([1.0, 1.4]) * scale),
        ]
    elif scenario == "head_on":
        humans = []
        humans.extend(head_on_group(0.88, 0.06, 3, lateral_offset=0.00 * scale, group_seed=7))
        humans.extend(head_on_group(0.83, 0.10, 2, lateral_offset=0.55 * scale, group_seed=17))
        humans.extend(head_on_group(0.79, 0.12, 2, lateral_offset=-0.55 * scale, group_seed=27))
    elif scenario == "diagonal":
        humans = [
            _make_goal_human(
                point(5.8, 8.0),
                point(7.9, 3.6),
                0.75,
                noise_std=0.0,
                velocity_smoothing=human_velocity_smoothing,
                goal_tolerance=human_goal_tolerance,
                rng_seed=seeded(11),
            )
        ]
    elif scenario == "blocking":
        humans = [
            _make_goal_human(
                point(5.0, 5.0),
                point(4.1, 5.0),
                0.12,
                noise_std=0.005,
                velocity_smoothing=0.8,
                goal_tolerance=human_goal_tolerance,
                rng_seed=seeded(13),
            )
        ]
    elif scenario == "permanent_blocking":
        robot_start = point(1.0, 5.0)
        robot_goal = point(9.0, 5.0)
        humans = []
        obstacles = [
            Obstacle(kind="rectangle", center=point(5.0, 5.0), size=np.array([0.9, 10.0]) * scale),
        ]
        zones = []
    elif scenario == "overtake":
        humans = [
            _make_goal_human(
                point(4.0, 5.7),
                point(8.0, 6.0),
                0.45,
                noise_std=0.0,
                velocity_smoothing=human_velocity_smoothing,
                goal_tolerance=human_goal_tolerance,
                rng_seed=seeded(17),
            )
        ]
    elif scenario == "dense":
        humans = []
        humans.extend(crossing_group(0.38, 1.0, 3, corridor_offset=-0.25 * scale, group_seed=19))
        humans.extend(crossing_group(0.62, -1.0, 3, corridor_offset=0.28 * scale, group_seed=23))
        humans.extend(head_on_group(0.86, 0.14, 2, lateral_offset=0.12 * scale, group_seed=29))
        obstacles = [
            Obstacle(kind="circle", center=point(3.0, 2.4), size=0.45 * scale),
            Obstacle(kind="rectangle", center=point(7.9, 7.0), size=np.array([0.8, 0.7]) * scale),
        ]
    elif scenario == "crossing_flow":
        humans = []
        humans.extend(crossing_group(0.32, 1.0, 2, corridor_offset=-0.32 * scale, group_seed=43))
        humans.extend(crossing_group(0.70, -1.0, 3, corridor_offset=0.28 * scale, group_seed=47))
    elif scenario == "mixed_crowd":
        humans = []
        humans.extend(crossing_group(0.36, 1.0, 3, corridor_offset=-corridor_bias, group_seed=53))
        humans.extend(crossing_group(0.66, -1.0, 2, corridor_offset=0.8 * corridor_bias, group_seed=59))
        humans.extend(diagonal_group(0.88, 0.55 * scale, 0.18, -0.35 * scale, 2, group_seed=61))
    elif scenario in {"obstacle_mix", "mixed_scenario"}:
        humans = []
        humans.extend(
            head_on_group(
                0.84,
                0.12,
                2,
                lateral_offset=0.58 * scale,
                goal_lateral_offset=0.46 * scale,
                group_seed=71,
            )
        )
        humans.extend(crossing_group(0.34, 1.0, 2, corridor_offset=-0.62 * scale, group_seed=73))
        humans.extend(diagonal_group(0.76, 0.90 * scale, 0.38, -0.18 * scale, 1, group_seed=79))
        obstacles = [
            Obstacle(kind="circle", center=point(4.7, 3.9), size=0.6 * scale),
            Obstacle(kind="rectangle", center=point(7.5, 8.0), size=np.array([0.8, 0.6]) * scale),
        ]
        zones = []
    elif scenario == "narrow_passage":
        humans = []
        obstacles = [
            Obstacle(kind="rectangle", center=point(4.9, 2.2), size=np.array([4.4, 2.8]) * scale),
            Obstacle(kind="rectangle", center=point(4.9, 7.8), size=np.array([4.4, 2.8]) * scale),
        ]
        humans.extend(head_on_group(0.88, 0.18, 3, lateral_offset=0.00 * scale, group_seed=101))
        humans.extend(head_on_group(0.22, 0.80, 2, lateral_offset=-0.26 * scale, group_seed=109))
    elif scenario == "corridor_trap":
        robot_start = point(1.0, 4.2)
        robot_goal = point(9.0, 4.2)
        humans = [
            Human(
                position=point(5.85, 1.48),
                velocity=np.zeros(2, dtype=float),
                waypoints=(
                    point(5.85, 1.48),
                    point(6.00, 1.48),
                ),
                loop_waypoints=True,
                preferred_speed=0.05,
                noise_std=0.0,
                velocity_smoothing=0.92,
                goal_tolerance=0.06 * scale,
                rng_seed=seeded(131),
                radius=0.34,
            ),
            Human(
                position=point(5.95, 1.92),
                velocity=np.zeros(2, dtype=float),
                waypoints=(
                    point(5.95, 1.92),
                    point(6.10, 1.92),
                ),
                loop_waypoints=True,
                preferred_speed=0.05,
                noise_std=0.0,
                velocity_smoothing=0.92,
                goal_tolerance=0.06 * scale,
                rng_seed=seeded(137),
                radius=0.34,
            ),
            Human(
                position=point(6.05, 2.30),
                velocity=np.zeros(2, dtype=float),
                waypoints=(
                    point(6.05, 2.30),
                    point(6.20, 2.30),
                ),
                loop_waypoints=True,
                preferred_speed=0.05,
                noise_std=0.0,
                velocity_smoothing=0.92,
                goal_tolerance=0.06 * scale,
                rng_seed=seeded(149),
                radius=0.34,
            ),
        ]
        obstacles = [
            Obstacle(kind="rectangle", center=point(4.8, 6.4), size=np.array([1.8, 2.0]) * scale),
            Obstacle(kind="rectangle", center=point(4.8, 3.4), size=np.array([1.8, 2.0]) * scale),
            Obstacle(kind="rectangle", center=point(7.1, 8.9), size=np.array([1.8, 0.7]) * scale),
        ]
        zones = []
    elif scenario == "random_crowd":
        rng = np.random.default_rng(seeded(151))
        robot_start = point(1.0, 5.0)
        robot_goal = point(9.0, 5.0)
        humans = []
        for index in range(8):
            start_progress = float(rng.uniform(0.18, 0.86))
            goal_progress = float(np.clip(1.0 - start_progress + rng.normal(0.0, 0.08), 0.10, 0.92))
            start_lateral = float(rng.uniform(-0.95, 0.95)) * scale
            goal_lateral = float(rng.uniform(-0.95, 0.95)) * scale
            direction_sign = 1.0 if index % 2 == 0 else -1.0
            humans.extend(
                crossing_group(
                    start_progress,
                    direction_sign,
                    1,
                    corridor_offset=start_lateral,
                    group_seed=seeded(163 + 7 * index),
                )
            )
            humans.extend(
                diagonal_group(
                    start_progress,
                    start_lateral,
                    goal_progress,
                    goal_lateral,
                    1,
                    group_seed=seeded(227 + 11 * index),
                )
            )
        obstacles = [
            Obstacle(kind="circle", center=point(4.4, 5.1), size=0.45 * scale),
            Obstacle(kind="rectangle", center=point(6.8, 7.0), size=np.array([0.9, 0.7]) * scale),
        ]
        zones = []
    elif scenario == "structured_crowd":
        rng = np.random.default_rng(seeded(251))
        robot_start = point(1.0, 5.0)
        robot_goal = point(9.0, 5.0)
        humans = []
        crossing_specs = (
            (0.24, 1.0, 2),
            (0.38, -1.0, 2),
            (0.52, 1.0, 3),
            (0.66, -1.0, 2),
            (0.80, 1.0, 2),
        )
        for index, (progress, direction_sign, count) in enumerate(crossing_specs):
            humans.extend(
                crossing_group(
                    float(np.clip(progress + rng.normal(0.0, 0.015), 0.14, 0.90)),
                    direction_sign,
                    count,
                    corridor_offset=float(rng.normal(0.0, 0.22)) * scale,
                    group_seed=seeded(263 + 13 * index),
                )
            )
        for index in range(5):
            start_progress = float(rng.uniform(0.18, 0.84))
            goal_progress = float(np.clip(1.0 - start_progress + rng.normal(0.0, 0.06), 0.12, 0.90))
            start_lateral = float(rng.uniform(-0.75, 0.75)) * scale
            goal_lateral = float(np.clip(-start_lateral + rng.normal(0.0, 0.18), -0.90, 0.90)) * scale
            humans.extend(
                diagonal_group(
                    start_progress,
                    start_lateral,
                    goal_progress,
                    goal_lateral,
                    1,
                    group_seed=seeded(341 + 17 * index),
                )
            )
        obstacles = [
            Obstacle(kind="circle", center=point(4.2, 4.0), size=0.40 * scale),
            Obstacle(kind="rectangle", center=point(6.6, 6.5), size=np.array([0.8, 0.6]) * scale),
        ]
        zones = []
    elif scenario == "stress_crowd":
        rng = np.random.default_rng(seeded(401))
        robot_start = point(1.0, 5.0)
        robot_goal = point(9.0, 5.0)
        humans = []
        obstacles = [
            Obstacle(kind="rectangle", center=point(4.9, 2.15), size=np.array([4.6, 2.5]) * scale),
            Obstacle(kind="rectangle", center=point(4.9, 7.85), size=np.array([4.6, 2.5]) * scale),
            Obstacle(kind="circle", center=point(6.1, 5.0), size=0.34 * scale),
        ]
        for index in range(4):
            humans.extend(
                head_on_group(
                    0.86 - 0.03 * index + float(rng.normal(0.0, 0.015)),
                    0.14 + 0.02 * index + float(rng.normal(0.0, 0.015)),
                    1,
                    lateral_offset=float(rng.normal(0.0, 0.12)) * scale,
                    goal_lateral_offset=float(rng.normal(0.0, 0.12)) * scale,
                    group_seed=seeded(419 + 11 * index),
                )
            )
        for index in range(6):
            humans.extend(
                crossing_group(
                    float(np.clip(0.34 + 0.08 * index + rng.normal(0.0, 0.018), 0.18, 0.86)),
                    1.0 if index % 2 == 0 else -1.0,
                    1,
                    corridor_offset=float(rng.normal(0.0, 0.12)) * scale,
                    group_seed=seeded(487 + 19 * index),
                )
            )
        for index in range(3):
            humans.append(
                Human(
                    position=point(5.55 + 0.22 * index, 5.0 + 0.16 * ((index % 2) - 0.5)),
                    velocity=np.zeros(2, dtype=float),
                    waypoints=(
                        point(5.45 + 0.22 * index, 4.72),
                        point(5.45 + 0.22 * index, 5.28),
                    ),
                    loop_waypoints=True,
                    preferred_speed=0.08,
                    noise_std=0.0,
                    velocity_smoothing=0.92,
                    goal_tolerance=0.05 * scale,
                    rng_seed=seeded(563 + 23 * index),
                    radius=0.34,
                )
            )
        zones = []
    elif scenario == "demo":
        demo_start = point(1.0, 1.0)
        demo_goal = point(8.6, 8.2)
        _, demo_crossing_group, demo_head_on_group, demo_diagonal_group = make_path_group_builders(
            demo_start,
            demo_goal,
        )
        humans = []
        humans.extend(demo_head_on_group(0.82, 0.12, 2, lateral_offset=0.32 * scale, group_seed=31))
        humans.extend(demo_crossing_group(0.42, 1.0, 2, corridor_offset=-0.34 * scale, group_seed=37))
        humans.extend(demo_diagonal_group(0.82, 0.62 * scale, 0.20, -0.62 * scale, 1, group_seed=41))
        obstacles = [
            Obstacle(kind="circle", center=point(3.7, 4.6), size=0.60 * scale),
            Obstacle(kind="rectangle", center=point(6.9, 6.7), size=np.array([1.0, 0.8]) * scale),
        ]
        zones = [
            NoGoZone(kind="circle", center=point(5.5, 3.1), size=0.35 * scale),
        ]
        robot_start = demo_start
        robot_goal = demo_goal
    else:
        raise ValueError(
            "unsupported scenario type: "
            f"{type}. Choose from empty, head_on, diagonal, blocking, overtake, dense, "
            "crossing_flow, two_crossing, mixed_crowd, multi_human_crossing, obstacle_mix, "
            "mixed_scenario, head_on_interaction, narrow_passage, corridor_trap, random_crowd, "
            "structured_crowd, stress_crowd, zone_only, demo."
        )

    return robot_start, robot_goal, humans, obstacles, zones


def build_demo_environment(
    world_size: float = 10.0,
    dt: float = 0.1,
    *,
    scenario: str = "demo",
    ablation_config: dict[str, bool] | None = None,
    alpha: float = 2.2,
    w_h: float = 5.0,
    w_social: float = 2.2,
    w_o: float = 8.0,
    w_g: float = 1.5,
    sigma_parallel: float = 2.8,
    sigma_perp: float = 1.3,
    social_sigma_perp: float = 1.9,
    sigma_obs: float = 1.3,
    sigma_zone: float = 0.8,
    sigma_goal: float = 2.0,
    zone_weight: float = 18.0,
    zone_inside_gain: float = 6.0,
    epsilon: float = 1e-3,
    normalize_gradient: bool = True,
    gradient_clip: float = 1.0,
    max_grad: float | None = None,
    prediction_horizon: float = 3.0,
    prediction_dt: float = 0.3,
    lambda_decay: float = 1.5,
    max_prediction_distance: float = 5.0,
    sigma_parallel_growth: float = 0.2,
    social_front_scale: float = 1.35,
    social_rear_scale: float = 0.6,
    tau: float = 1.0,
    d0: float = 1.5,
    momentum: float = 0.8,
    grad_smoothing: float = 0.8,
    max_turn_angle: float = float(np.deg2rad(25.0)),
    min_forward_dot: float = 0.2,
    beta: float = 0.2,
    grad_scale_k: float = 0.35,
    safe_dist: float = 1.2,
    risk_threshold: float = 0.75,
    high_risk_gain: float = 2.4,
    repulsion_gain: float = 1.7,
    risk_blend_sharpness: float = 10.0,
    repulsion_zone_scale: float = 2.0,
    barrier_scale: float = 0.5,
    barrier_gain: float = 4.0,
    min_clearance: float = 0.35,
    grad_epsilon: float = 0.05,
    direction_epsilon: float = 0.05,
    velocity_smoothing: float = 0.8,
    alpha_damping: float = 0.95,
    ttc_mid: float = 2.0,
    ttc_slope: float = 0.5,
    ttc_stop: float = 1.0,
    ttc_resume: float = 1.4,
    safe_clearance: float = 0.95,
    risk_speed_gain: float = 0.32,
    min_speed_scale: float = 0.15,
    goal_ttc_disable_distance: float = 0.35,
    ttc_human_speed_threshold: float = 0.05,
    interaction_forward_threshold: float = 0.15,
    lateral_safe_distance: float = 1.2,
    lateral_block_threshold: float = 0.55,
    ttc_attention_threshold: float = 6.5,
    interaction_beta: float = 0.6,
    interaction_prediction_dt: float = 0.2,
    clearance_push_gain: float = 5.2,
    comfort_clearance: float = 1.15,
    interaction_persistence_tau: float = 0.75,
    interaction_memory_gain: float = 1.8,
    interaction_memory_floor_ratio: float = 0.6,
    interaction_current_blend: float = 0.4,
    interaction_memory_blend: float = 0.6,
    interaction_strength_sharpness: float = 6.0,
    interaction_min_strength: float = 0.035,
    interaction_force_memory_decay: float = 0.85,
    interaction_force_memory_gain: float = 1.35,
    side_commitment_decay: float = 0.95,
    side_commitment_gain: float = 1.4,
    interaction_recovery_threshold: float = 0.02,
    speed_interaction_gain: float = 1.05,
    speed_scale_smoothing: float = 0.4,
    yield_score_enter: float = 0.45,
    yield_score_exit: float = 0.25,
    interaction_switch_margin: float = 0.35,
    interaction_conflict_margin: float = 0.15,
    interaction_conflict_beta_scale: float = 0.6,
    recovery_speed_bias: float = 0.9,
    human_noise_std: float = 0.05,
    human_velocity_smoothing: float = 0.8,
    human_goal_tolerance: float = 0.35,
    seed: int = 0,
    show_risk: bool = True,
    show_debug: bool = True,
    real_time: bool = False,
    render_skip: int = 3,
    risk_resolution: int = 60,
    risk_update_interval: int = 5,
    record_human_trails: bool = False,
    log_summary_interval: int = 0,
    log_detailed_threshold: float = 0.12,
    log_console_output: bool = True,
) -> Environment:
    size = _validate_positive(world_size, "world_size")
    world = np.array([size, size], dtype=float)
    if max_grad is None:
        max_grad = gradient_clip
    ablation_config = ablation_config or {}
    weak_suppression = bool(ablation_config.get("weak_suppression", True))
    interaction_memory_enabled = bool(ablation_config.get("interaction_memory", True))
    topk_filter = bool(ablation_config.get("topk_filter", True))
    multi_human_aggregation = bool(ablation_config.get("multi_human", True))
    guide_planner_enabled = bool(ablation_config.get("guide_planner", True))
    failed_branch_memory_enabled = bool(ablation_config.get("failed_branch_memory", True))
    execution_progress_invariant_enabled = bool(ablation_config.get("execution_progress_invariant", True))

    robot_start, robot_goal, humans, obstacles, zones = build_scenario(
        scenario,
        world_size=size,
        human_noise_std=human_noise_std,
        human_velocity_smoothing=human_velocity_smoothing,
        human_goal_tolerance=human_goal_tolerance,
        seed=seed,
    )

    robot = Robot(
        position=robot_start,
        velocity=np.zeros(2, dtype=float),
        goal=robot_goal,
        speed=1.0,
        goal_slowdown_distance=d0,
        momentum=momentum,
        grad_smoothing=grad_smoothing,
        max_turn_angle=max_turn_angle,
        min_forward_dot=min_forward_dot,
        goal_bias=beta,
        grad_scale_k=grad_scale_k,
        safe_distance=safe_dist,
        risk_threshold=risk_threshold,
        high_risk_gain=high_risk_gain,
        repulsion_gain=repulsion_gain,
        risk_blend_sharpness=risk_blend_sharpness,
        repulsion_zone_scale=repulsion_zone_scale,
        barrier_scale=barrier_scale,
        barrier_gain=barrier_gain,
        min_clearance=min_clearance,
        grad_epsilon=grad_epsilon,
        direction_epsilon=direction_epsilon,
        velocity_smoothing=velocity_smoothing,
        alpha_damping=alpha_damping,
        ttc_mid=ttc_mid,
        ttc_slope=ttc_slope,
        ttc_stop=ttc_stop,
        ttc_resume=ttc_resume,
        safe_clearance=safe_clearance,
        risk_speed_gain=risk_speed_gain,
        min_speed_scale=min_speed_scale,
        goal_ttc_disable_distance=goal_ttc_disable_distance,
        ttc_human_speed_threshold=ttc_human_speed_threshold,
        interaction_forward_threshold=interaction_forward_threshold,
        lateral_safe_distance=lateral_safe_distance,
        lateral_block_threshold=lateral_block_threshold,
        ttc_attention_threshold=ttc_attention_threshold,
        interaction_beta=interaction_beta,
        interaction_prediction_dt=interaction_prediction_dt,
        clearance_push_gain=clearance_push_gain,
        comfort_clearance=comfort_clearance,
        interaction_persistence_tau=interaction_persistence_tau,
        interaction_memory_gain=interaction_memory_gain,
        interaction_memory_floor_ratio=interaction_memory_floor_ratio,
        interaction_current_blend=interaction_current_blend,
        interaction_memory_blend=interaction_memory_blend,
        interaction_strength_sharpness=interaction_strength_sharpness,
        interaction_min_strength=interaction_min_strength,
        interaction_force_memory_decay=interaction_force_memory_decay,
        interaction_force_memory_gain=interaction_force_memory_gain,
        side_commitment_decay=side_commitment_decay,
        side_commitment_gain=side_commitment_gain,
        interaction_recovery_threshold=interaction_recovery_threshold,
        speed_interaction_gain=speed_interaction_gain,
        speed_scale_smoothing=speed_scale_smoothing,
        yield_score_enter=yield_score_enter,
        yield_score_exit=yield_score_exit,
        interaction_switch_margin=interaction_switch_margin,
        interaction_conflict_margin=interaction_conflict_margin,
        interaction_conflict_beta_scale=interaction_conflict_beta_scale,
        recovery_speed_bias=recovery_speed_bias,
        weak_suppression=weak_suppression,
        interaction_memory_enabled=interaction_memory_enabled,
        topk_filter=topk_filter,
        guide_planner_enabled=guide_planner_enabled,
        failed_branch_memory_enabled=failed_branch_memory_enabled,
        execution_progress_invariant_enabled=execution_progress_invariant_enabled,
    )
    guide_clearance = robot.radius + robot.safety_distance + 0.5 * robot.safety_hysteresis
    if guide_planner_enabled:
        robot.static_guide_waypoints = _build_static_guide_waypoints(
            start=robot_start,
            goal=robot_goal,
            world_size=world,
            obstacles=obstacles,
            zones=zones,
            clearance_threshold=guide_clearance,
        )
    else:
        robot.static_guide_waypoints = []
    robot.static_guide_index = 0
    robot.static_guide_anchor_position = robot.position.copy()
    robot.static_guide_anchor_progress = 0.0
    robot.guide_progress_max = 0.0
    risk_field = RiskField(
        humans=humans,
        obstacles=obstacles,
        zones=zones,
        goal=robot.goal.copy(),
        world_size=world,
        alpha=alpha,
        w_h=w_h,
        w_social=w_social,
        w_o=w_o,
        w_zone=zone_weight,
        w_g=w_g,
        sigma_parallel=sigma_parallel,
        sigma_perp=sigma_perp,
        social_sigma_perp=social_sigma_perp,
        sigma_obs=sigma_obs,
        sigma_zone=sigma_zone,
        sigma_goal=sigma_goal,
        zone_inside_gain=zone_inside_gain,
        epsilon=epsilon,
        normalize_gradient=normalize_gradient,
        gradient_clip=max_grad,
        prediction_horizon=prediction_horizon,
        prediction_dt=prediction_dt,
        lambda_decay=lambda_decay,
        max_prediction_distance=max_prediction_distance,
        sigma_parallel_growth=sigma_parallel_growth,
        social_front_scale=social_front_scale,
        social_rear_scale=social_rear_scale,
        interaction_tau=tau,
        multi_human_aggregation=multi_human_aggregation,
    )
    logger = NavLogger(
        summary_interval=log_summary_interval,
        detailed_threshold=log_detailed_threshold,
        console_output=log_console_output,
        safety_distance=robot.safety_distance,
    )
    return Environment(
        world_size=world,
        robot=robot,
        humans=humans,
        obstacles=obstacles,
        zones=zones,
        risk_field=risk_field,
        dt=dt,
        show_risk=show_risk,
        show_debug=show_debug,
        real_time=real_time,
        render_skip=render_skip,
        risk_resolution=risk_resolution,
        risk_update_interval=risk_update_interval,
        record_human_trails=record_human_trails,
        scenario_name=scenario,
        logger=logger,
    )


ABlationConfig = dict[str, bool]

FULL_ABLATION_CONFIG: ABlationConfig = {
    "weak_suppression": True,
    "interaction_memory": True,
    "topk_filter": True,
    "multi_human": True,
    "guide_planner": True,
    "failed_branch_memory": True,
    "execution_progress_invariant": True,
}

ABLATION_PRESETS: dict[str, tuple[str, ABlationConfig]] = {
    "full": ("FULL_SYSTEM", FULL_ABLATION_CONFIG),
    "no_weak": (
        "NO_WEAK_SUPPRESSION",
        {
            "weak_suppression": False,
            "interaction_memory": True,
            "topk_filter": True,
            "multi_human": True,
            "guide_planner": True,
            "failed_branch_memory": True,
            "execution_progress_invariant": True,
        },
    ),
    "no_memory": (
        "NO_INTERACTION_MEMORY",
        {
            "weak_suppression": True,
            "interaction_memory": False,
            "topk_filter": True,
            "multi_human": True,
            "guide_planner": True,
            "failed_branch_memory": True,
            "execution_progress_invariant": True,
        },
    ),
    "no_topk": (
        "NO_TOPK_FILTER",
        {
            "weak_suppression": True,
            "interaction_memory": True,
            "topk_filter": False,
            "multi_human": True,
            "guide_planner": True,
            "failed_branch_memory": True,
            "execution_progress_invariant": True,
        },
    ),
    "no_multi": (
        "NO_MULTI_HUMAN_AGGREGATION",
        {
            "weak_suppression": True,
            "interaction_memory": True,
            "topk_filter": True,
            "multi_human": False,
            "guide_planner": True,
            "failed_branch_memory": True,
            "execution_progress_invariant": True,
        },
    ),
    "no_invariant": (
        "NO_INVARIANT_ENFORCEMENT",
        {
            "weak_suppression": True,
            "interaction_memory": True,
            "topk_filter": True,
            "multi_human": True,
            "guide_planner": True,
            "failed_branch_memory": True,
            "execution_progress_invariant": False,
        },
    ),
    "reactive_baseline": (
        "REACTIVE_BASELINE",
        {
            "weak_suppression": False,
            "interaction_memory": False,
            "topk_filter": False,
            "multi_human": False,
            "guide_planner": False,
            "failed_branch_memory": False,
            "execution_progress_invariant": False,
        },
    ),
}


@dataclass
class ExperimentConfig:
    name: str
    scenario: str
    seed: int = 0
    max_steps: int = 400
    ablation_config: ABlationConfig = field(default_factory=lambda: copy.deepcopy(FULL_ABLATION_CONFIG))
    environment_kwargs: dict[str, object] = field(default_factory=dict)


def _sanitize_path_component(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def _write_csv_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        selected_fields = list(fieldnames or [])
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected_fields)
            if selected_fields:
                writer.writeheader()
        return

    selected_fields = list(fieldnames or rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _serialize_environment_metadata(
    environment: Environment,
    *,
    config_name: str,
    scenario: str,
    trial_index: int,
    seed: int,
    log_path: Path,
) -> dict[str, object]:
    return {
        "config": config_name,
        "scenario": scenario,
        "trial": int(trial_index),
        "seed": int(seed),
        "log_path": str(log_path),
        "world_size": [float(value) for value in environment.world_size],
        "robot_start": [float(value) for value in environment._reference_path_start],
        "robot_goal": [float(value) for value in environment._reference_path_goal],
        "safety_distance": float(environment.robot.safety_distance),
        "obstacles": [
            {
                "kind": obstacle.kind,
                "center": [float(value) for value in obstacle.center],
                "size": (
                    float(obstacle.size)
                    if obstacle.kind == "circle"
                    else [float(value) for value in np.asarray(obstacle.size, dtype=float)]
                ),
            }
            for obstacle in environment.obstacles
        ],
        "zones": [
            {
                "kind": zone.kind,
                "center": [float(value) for value in zone.center],
                "size": (
                    float(zone.size)
                    if zone.kind == "circle"
                    else [float(value) for value in np.asarray(zone.size, dtype=float)]
                ),
            }
            for zone in environment.zones
        ],
        "humans": [
            {
                "initial_position": [float(value) for value in state[0]],
                "initial_velocity": [float(value) for value in state[1]],
                "initial_goal": None if state[2] is None else [float(value) for value in state[2]],
                "radius": float(human.radius),
            }
            for state, human in zip(environment._initial_human_states, environment.humans)
        ],
        "failed_branches": [
            {
                "point": [float(value) for value in branch.point],
                "guide_direction": [float(value) for value in branch.guide_direction],
                "guide_normal": [float(value) for value in branch.guide_normal],
                "radius": float(branch.radius),
            }
            for branch in environment.robot.failed_branches
        ],
    }


def _load_run_log(log_path: Path) -> dict[str, np.ndarray | list[str]]:
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def float_series(name: str, fallback: str | None = None) -> np.ndarray:
        key = name if rows and name in rows[0] else fallback
        if key is None:
            return np.array([], dtype=float)
        values = []
        for row in rows:
            raw = row.get(key, "")
            values.append(float(raw) if raw not in {"", "--"} else float("nan"))
        return np.asarray(values, dtype=float)

    def int_series(name: str, fallback: str | None = None) -> np.ndarray:
        key = name if rows and name in rows[0] else fallback
        if key is None:
            return np.array([], dtype=int)
        values = []
        for row in rows:
            raw = row.get(key, "")
            values.append(int(raw) if raw not in {"", "--"} else -1)
        return np.asarray(values, dtype=int)

    state_key = "behavior_state" if rows and "behavior_state" in rows[0] else "state"
    return {
        "time": float_series("time"),
        "x": float_series("x"),
        "y": float_series("y"),
        "speed": float_series("speed", "speed_scale"),
        "interaction_level": float_series("interaction_level", "interaction_effective"),
        "active_humans": int_series("active_humans", "number_of_active_humans"),
        "primary_human_id": int_series("primary_human_id", "interacting_human_id"),
        "min_clearance": float_series("min_clearance", "min_distance_to_humans"),
        "global_clearance": float_series("global_clearance", "clr_global"),
        "curvature": float_series("curvature"),
        "global_progress": float_series("global_progress"),
        "guide_progress_max": float_series("guide_progress_max"),
        "failed_branch_count": int_series("failed_branch_count"),
        "goal_distance": float_series("goal_distance"),
        "behavior_state": [row.get(state_key, "") for row in rows],
    }


def _prepare_plotting_backend():
    global _PLOTTING_BACKEND, _PLOTTING_ERROR

    if _PLOTTING_BACKEND is not None:
        return _PLOTTING_BACKEND
    if _PLOTTING_ERROR is not None:
        raise _PLOTTING_ERROR

    _ensure_writable_matplotlib_config()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        _PLOTTING_ERROR = RuntimeError(
            "matplotlib is required to generate experiment plots. Install dependencies in a "
            "virtual environment, for example: python3 -m venv .venv && .venv/bin/pip "
            "install numpy matplotlib"
        )
        raise _PLOTTING_ERROR from exc

    _PLOTTING_BACKEND = plt
    return _PLOTTING_BACKEND


def _save_figure(figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")


def _plot_clearance_profile(log_data: dict[str, np.ndarray | list[str]], output_stem: Path, title: str) -> None:
    plt = _prepare_plotting_backend()
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    time = np.asarray(log_data["time"], dtype=float)
    clearance = np.asarray(log_data["min_clearance"], dtype=float)
    valid = np.isfinite(time) & np.isfinite(clearance)
    if np.any(valid):
        axis.plot(time[valid], clearance[valid], color="#005f73", linewidth=2.0, label="Clearance")
        valid_indices = np.flatnonzero(valid)
        min_index = valid_indices[int(np.argmin(clearance[valid]))]
        axis.scatter(
            [time[min_index]],
            [clearance[min_index]],
            color="#ae2012",
            s=45,
            zorder=3,
            label=f"Minimum = {clearance[min_index]:0.2f} m",
        )
    else:
        axis.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Minimum Clearance [m]")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if np.any(valid):
        axis.legend(loc="best")
    _save_figure(figure, output_stem)
    plt.close(figure)


def _plot_interaction_profile(
    log_data: dict[str, np.ndarray | list[str]],
    output_stem: Path,
    title: str,
    activation_threshold: float = 0.05,
) -> None:
    plt = _prepare_plotting_backend()
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    time = np.asarray(log_data["time"], dtype=float)
    interaction = np.asarray(log_data["interaction_level"], dtype=float)
    valid = np.isfinite(time) & np.isfinite(interaction)
    if np.any(valid):
        axis.plot(time[valid], interaction[valid], color="#ca6702", linewidth=2.0, label="Interaction")
        active = valid & (interaction > activation_threshold)
        if np.any(active):
            axis.fill_between(
                time,
                0.0,
                interaction,
                where=active,
                color="#ee9b00",
                alpha=0.25,
                label="Activation period",
            )
    else:
        axis.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Interaction Level")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if np.any(valid):
        axis.legend(loc="best")
    _save_figure(figure, output_stem)
    plt.close(figure)


def _plot_speed_profile(log_data: dict[str, np.ndarray | list[str]], output_stem: Path, title: str) -> None:
    plt = _prepare_plotting_backend()
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    time = np.asarray(log_data["time"], dtype=float)
    speed = np.asarray(log_data["speed"], dtype=float)
    valid = np.isfinite(time) & np.isfinite(speed)
    if np.any(valid):
        axis.plot(time[valid], speed[valid], color="#1d3557", linewidth=2.0, label="Speed")
        axis.axhline(
            float(np.max(speed[valid])),
            color="0.45",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="Peak speed",
        )
    else:
        axis.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Speed [m/s]")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if np.any(valid):
        axis.legend(loc="best")
    _save_figure(figure, output_stem)
    plt.close(figure)


def _plot_curvature_profile(log_data: dict[str, np.ndarray | list[str]], output_stem: Path, title: str) -> None:
    plt = _prepare_plotting_backend()
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    time = np.asarray(log_data["time"], dtype=float)
    curvature = np.asarray(log_data["curvature"], dtype=float)
    valid = np.isfinite(time) & np.isfinite(curvature)
    if np.any(valid):
        clipped = np.clip(curvature[valid], -5.0, 5.0)
        axis.plot(time[valid], clipped, color="#6a4c93", linewidth=1.8, label="Curvature")
        axis.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
    else:
        axis.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=axis.transAxes)
    axis.set_title(title)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Curvature [rad/m]")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    if np.any(valid):
        axis.legend(loc="best")
    _save_figure(figure, output_stem)
    plt.close(figure)


def _plot_trajectory(environment: Environment, output_stem: Path, title: str) -> None:
    plt = _prepare_plotting_backend()
    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_xlim(0.0, environment.world_size[0])
    axis.set_ylim(0.0, environment.world_size[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    for obstacle in environment.obstacles:
        axis.add_patch(obstacle.create_patch())
    for zone in environment.zones:
        axis.add_patch(zone.create_patch())

    robot_trail = np.asarray(environment.robot.trail, dtype=float)
    if robot_trail.size > 0:
        axis.plot(robot_trail[:, 0], robot_trail[:, 1], color="tab:blue", linewidth=2.2, label="Robot")
        axis.scatter(robot_trail[0, 0], robot_trail[0, 1], color="tab:blue", s=45, marker="o", label="Start")

    axis.scatter(
        environment.robot.goal[0],
        environment.robot.goal[1],
        color="tab:green",
        s=180,
        marker="*",
        label="Goal",
        zorder=5,
    )

    for index, trail in enumerate(environment.human_trails):
        human_trail = np.asarray(trail, dtype=float)
        if human_trail.size == 0:
            continue
        label = "Human trajectory" if index == 0 else None
        axis.plot(
            human_trail[:, 0],
            human_trail[:, 1],
            color="tab:red",
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            label=label,
        )

    axis.legend(loc="best")
    _save_figure(figure, output_stem)
    plt.close(figure)


def _generate_run_plots(environment: Environment, log_path: Path, output_dir: Path, title_prefix: str) -> None:
    log_data = _load_run_log(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_clearance_profile(log_data, output_dir / "clearance", f"{title_prefix} Clearance vs Time")
    _plot_interaction_profile(log_data, output_dir / "interaction", f"{title_prefix} Interaction vs Time")
    _plot_speed_profile(log_data, output_dir / "speed", f"{title_prefix} Speed Profile")
    _plot_curvature_profile(log_data, output_dir / "curvature", f"{title_prefix} Curvature vs Time")
    _plot_trajectory(environment, output_dir / "trajectory", f"{title_prefix} Trajectory")


def _safe_mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        non_nan = array[~np.isnan(array)]
        if non_nan.size > 0 and np.all(np.isposinf(non_nan)):
            return float("inf"), 0.0
        if non_nan.size > 0 and np.all(np.isneginf(non_nan)):
            return float("-inf"), 0.0
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _compute_run_summary(
    environment: Environment,
    *,
    config_name: str,
    scenario: str,
    trial_index: int,
    seed: int,
    log_path: Path,
) -> dict[str, object]:
    metrics = environment.validation_metrics()
    min_clr = float(metrics.get("minimum_clearance", float("inf")))
    avg_interaction_duration = float(metrics.get("average_interaction_duration", 0.0))
    recovery_time = float(metrics.get("recovery_time", 0.0))
    path_efficiency = float(metrics.get("path_efficiency", float("inf")))
    raw_curvature_mean = float(metrics.get("raw_curvature_mean", metrics.get("mean_abs_curvature", 0.0)))
    raw_curvature_max = float(metrics.get("raw_curvature_max", metrics.get("max_abs_curvature", 0.0)))
    smoothness_mean = float(metrics.get("smoothness_mean", 0.0))
    smoothness_max = float(metrics.get("smoothness_max", 0.0))
    safety_violations = float(metrics.get("safety_violations", 0.0))
    unresolved_recoveries = float(metrics.get("unresolved_recoveries", 0.0))
    success = int(environment.robot.behavior_state == "goal")
    collision = int(safety_violations > 0.0 or min_clr < environment.robot.safety_distance - 1e-4)
    time_to_goal = float(environment.time) if success else float("nan")
    return {
        "config": config_name,
        "scenario": scenario,
        "trial": int(trial_index),
        "seed": int(seed),
        "log_path": str(log_path),
        "min_clr": min_clr,
        "avg_interaction_time": avg_interaction_duration,
        "avg_interaction_duration": avg_interaction_duration,
        "recovery_time": recovery_time,
        "path_efficiency": path_efficiency,
        "curvature": raw_curvature_mean,
        "raw_curvature_mean": raw_curvature_mean,
        "raw_curvature_max": raw_curvature_max,
        "smoothness_mean": smoothness_mean,
        "smoothness_max": smoothness_max,
        "safety_violations": safety_violations,
        "unresolved_recoveries": unresolved_recoveries,
        "collision": collision,
        "success": success,
        "time_to_goal": time_to_goal,
        "failed_branch_count": int(len(environment.robot.failed_branches)),
        "invariant_recovery_count": int(environment.invariant_recovery_count),
        "steps": int(environment.frame_count),
        "sim_time": float(environment.time),
    }


def _aggregate_run_summaries(run_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in run_rows:
        grouped.setdefault(str(row["config"]), []).append(dict(row))

    summary_rows: list[dict[str, object]] = []
    for config_name, rows in grouped.items():
        min_mean, min_std = _safe_mean_std([float(row["min_clr"]) for row in rows])
        collision_rate, _ = _safe_mean_std([float(row["collision"]) for row in rows])
        int_mean, int_std = _safe_mean_std([float(row["avg_interaction_time"]) for row in rows])
        recovery_mean, recovery_std = _safe_mean_std([float(row["recovery_time"]) for row in rows])
        unresolved_mean, unresolved_std = _safe_mean_std([float(row["unresolved_recoveries"]) for row in rows])
        path_eff_mean, path_eff_std = _safe_mean_std([float(row["path_efficiency"]) for row in rows])
        curvature_mean, curvature_std = _safe_mean_std([float(row["curvature"]) for row in rows])
        smooth_mean_mean, smooth_mean_std = _safe_mean_std([float(row["smoothness_mean"]) for row in rows])
        raw_curvature_max_mean, raw_curvature_max_std = _safe_mean_std(
            [float(row["raw_curvature_max"]) for row in rows]
        )
        smooth_max_mean, smooth_max_std = _safe_mean_std([float(row["smoothness_max"]) for row in rows])
        safety_mean, safety_std = _safe_mean_std([float(row["safety_violations"]) for row in rows])
        time_to_goal_mean, time_to_goal_std = _safe_mean_std([float(row["time_to_goal"]) for row in rows])
        steps_mean, steps_std = _safe_mean_std([float(row["steps"]) for row in rows])
        failed_branch_mean, failed_branch_std = _safe_mean_std([float(row["failed_branch_count"]) for row in rows])
        invariant_recovery_mean, invariant_recovery_std = _safe_mean_std(
            [float(row["invariant_recovery_count"]) for row in rows]
        )
        success_rate, _ = _safe_mean_std([float(row["success"]) for row in rows])
        summary_rows.append(
            {
                "config": config_name,
                "scenario": rows[0]["scenario"],
                "num_trials": len(rows),
                "min_clr_mean": min_mean,
                "min_clr_std": min_std,
                "collision_rate": collision_rate,
                "avg_interaction_time_mean": int_mean,
                "avg_interaction_time_std": int_std,
                "recovery_time_mean": recovery_mean,
                "recovery_time_std": recovery_std,
                "unresolved_recoveries_mean": unresolved_mean,
                "unresolved_recoveries_std": unresolved_std,
                "path_efficiency_mean": path_eff_mean,
                "path_efficiency_std": path_eff_std,
                "curvature_mean": curvature_mean,
                "curvature_std": curvature_std,
                "raw_curvature_max_mean": raw_curvature_max_mean,
                "raw_curvature_max_std": raw_curvature_max_std,
                "smoothness_mean_mean": smooth_mean_mean,
                "smoothness_mean_std": smooth_mean_std,
                "smoothness_max_mean": smooth_max_mean,
                "smoothness_max_std": smooth_max_std,
                "safety_violations_mean": safety_mean,
                "safety_violations_std": safety_std,
                "time_to_goal_mean": time_to_goal_mean,
                "time_to_goal_std": time_to_goal_std,
                "steps_mean": steps_mean,
                "steps_std": steps_std,
                "failed_branch_count_mean": failed_branch_mean,
                "failed_branch_count_std": failed_branch_std,
                "invariant_recovery_count_mean": invariant_recovery_mean,
                "invariant_recovery_count_std": invariant_recovery_std,
                "success_rate": success_rate,
            }
        )

    return summary_rows


def _plot_summary_comparison(summary_rows: Sequence[dict[str, object]], output_stem: Path) -> None:
    if len(summary_rows) <= 1:
        return

    plt = _prepare_plotting_backend()
    labels = [str(row["config"]) for row in summary_rows]
    x = np.arange(len(labels), dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    metric_specs = [
        ("min_clr_mean", "min_clr_std", "Min Clearance [m]"),
        ("avg_interaction_time_mean", "avg_interaction_time_std", "Interaction Duration [s]"),
        ("recovery_time_mean", "recovery_time_std", "Recovery Time [s]"),
        ("path_efficiency_mean", "path_efficiency_std", "Path Efficiency"),
        ("smoothness_mean_mean", "smoothness_mean_std", "Mean Curvature"),
        ("safety_violations_mean", "safety_violations_std", "Safety Violations"),
    ]
    colors = ["#005f73", "#ca6702", "#6a4c93", "#2a9d8f", "#8c564b", "#bc6c25"]

    for axis, color, (mean_key, std_key, ylabel) in zip(axes.flat, colors, metric_specs):
        means = np.asarray([float(row[mean_key]) for row in summary_rows], dtype=float)
        if std_key is None:
            errors = None
        else:
            errors = np.asarray([float(row[std_key]) for row in summary_rows], dtype=float)
        axis.bar(x, means, yerr=errors, capsize=4, color=color, alpha=0.85)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    figure.suptitle("Experiment Summary Comparison")
    _save_figure(figure, output_stem)
    plt.close(figure)


def build_experiment_configs(
    scenario: str,
    *,
    ablation: str = "full",
    seed: int = 0,
    max_steps: int = 400,
    environment_kwargs: dict[str, object] | None = None,
) -> list[ExperimentConfig]:
    ablation_key = str(ablation).lower()
    if ablation_key == "all":
        selected_keys = ["full", "no_weak", "no_memory", "no_topk", "no_multi"]
    else:
        if ablation_key not in ABLATION_PRESETS:
            raise ValueError(f"unsupported ablation type: {ablation}")
        selected_keys = [ablation_key]

    configs: list[ExperimentConfig] = []
    for key in selected_keys:
        label, config = ABLATION_PRESETS[key]
        configs.append(
            ExperimentConfig(
                name=label,
                scenario=scenario,
                seed=int(seed),
                max_steps=max(int(max_steps), 1),
                ablation_config=copy.deepcopy(config),
                environment_kwargs=copy.deepcopy(environment_kwargs or {}),
            )
        )
    return configs


def run_experiments(
    config_list: Sequence[ExperimentConfig],
    num_trials: int,
    *,
    output_dir: str | Path = "results/experiments",
    generate_plots: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root_dir = Path(output_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    plotting_notice: str | None = None

    if generate_plots:
        try:
            _prepare_plotting_backend()
        except RuntimeError as exc:
            generate_plots = False
            plotting_notice = str(exc)
            (root_dir / "plots_skipped.txt").write_text(f"{plotting_notice}\n", encoding="utf-8")

    for config in config_list:
        config_dir = root_dir / _sanitize_path_component(config.name)
        config_dir.mkdir(parents=True, exist_ok=True)
        for trial_index in range(max(int(num_trials), 1)):
            trial_seed = int(config.seed) + trial_index
            trial_dir = config_dir / f"trial_{trial_index:03d}_seed_{trial_seed}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            batch_environment_kwargs = copy.deepcopy(config.environment_kwargs)
            for key in (
                "show_risk",
                "show_debug",
                "real_time",
                "log_console_output",
                "record_human_trails",
                "scenario",
                "seed",
                "ablation_config",
            ):
                batch_environment_kwargs.pop(key, None)

            environment = build_demo_environment(
                scenario=config.scenario,
                seed=trial_seed,
                ablation_config=copy.deepcopy(config.ablation_config),
                show_risk=False,
                show_debug=False,
                real_time=False,
                log_console_output=False,
                record_human_trails=generate_plots,
                **batch_environment_kwargs,
            )
            environment.run(max_steps=config.max_steps)

            log_path = trial_dir / "timestep_log.csv"
            environment.save_logs(str(log_path))
            _write_json(
                trial_dir / "trial_metadata.json",
                _serialize_environment_metadata(
                    environment,
                    config_name=config.name,
                    scenario=config.scenario,
                    trial_index=trial_index,
                    seed=trial_seed,
                    log_path=log_path,
                ),
            )

            run_summary = _compute_run_summary(
                environment,
                config_name=config.name,
                scenario=config.scenario,
                trial_index=trial_index,
                seed=trial_seed,
                log_path=log_path,
            )
            _write_csv_rows(trial_dir / "run_summary.csv", [run_summary])
            run_rows.append(run_summary)

            if generate_plots:
                _generate_run_plots(
                    environment,
                    log_path,
                    trial_dir / "plots",
                    f"{config.name} | trial {trial_index:03d}",
                )

    summary_rows = _aggregate_run_summaries(run_rows)
    _write_csv_rows(root_dir / "run_results.csv", run_rows)
    _write_csv_rows(root_dir / "summary_results.csv", summary_rows)
    if generate_plots:
        _plot_summary_comparison(summary_rows, root_dir / "summary_comparison")
    elif plotting_notice:
        print(f"plots skipped | {plotting_notice}")
    return run_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal 2D social navigation simulator with an interaction-aware predictive risk field."
    )
    parser.add_argument("--world-size", type=float, default=10.0, help="World width/height in meters.")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep in seconds.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="demo",
        choices=[
            "crossing",
            "crossing_flow",
            "empty",
            "head_on",
            "head_on_interaction",
            "diagonal",
            "blocking",
            "overtake",
            "dense",
            "two_crossing",
            "mixed_crowd",
            "multi_human_crossing",
            "obstacle_mix",
            "mixed_scenario",
            "narrow_passage",
            "corridor_trap",
            "random_crowd",
            "structured_crowd",
            "stress_crowd",
            "permanent_blocking",
            "zone_only",
            "demo",
        ],
        help="Interaction scenario to load.",
    )
    parser.add_argument("--alpha", type=float, default=2.2, help="Risk gradient influence weight.")
    parser.add_argument("--human-weight", type=float, default=5.0, help="Human risk weight.")
    parser.add_argument(
        "--social-weight",
        type=float,
        default=2.2,
        help="Additional social comfort weight around humans.",
    )
    parser.add_argument("--obstacle-weight", type=float, default=8.0, help="Obstacle risk weight.")
    parser.add_argument("--goal-weight", type=float, default=1.5, help="Goal attraction weight.")
    parser.add_argument(
        "--sigma-parallel",
        type=float,
        default=2.8,
        help="Human risk spread along velocity direction.",
    )
    parser.add_argument(
        "--sigma-perp",
        type=float,
        default=1.3,
        help="Human risk spread perpendicular to velocity direction.",
    )
    parser.add_argument(
        "--social-sigma-perp",
        type=float,
        default=1.9,
        help="Lateral spread of the social-comfort human risk term.",
    )
    parser.add_argument("--sigma-obs", type=float, default=1.3, help="Obstacle risk spread.")
    parser.add_argument("--sigma-zone", type=float, default=0.8, help="No-go zone risk spread.")
    parser.add_argument("--sigma-goal", type=float, default=2.0, help="Goal attraction spread.")
    parser.add_argument("--zone-weight", type=float, default=18.0, help="No-go zone penalty weight.")
    parser.add_argument(
        "--zone-inside-gain",
        type=float,
        default=6.0,
        help="Additional penalty growth applied inside a no-go zone.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Finite-difference step.")
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Time-to-interaction weighting width in seconds.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=float,
        default=3.0,
        help="Human prediction horizon in seconds.",
    )
    parser.add_argument(
        "--prediction-dt",
        type=float,
        default=0.3,
        help="Prediction timestep in seconds.",
    )
    parser.add_argument(
        "--lambda-decay",
        type=float,
        default=1.5,
        help="Exponential decay applied to future human risk.",
    )
    parser.add_argument(
        "--max-prediction-distance",
        type=float,
        default=5.0,
        help="Skip predicted human contributions farther than this distance.",
    )
    parser.add_argument(
        "--sigma-parallel-growth",
        type=float,
        default=0.2,
        help="Increase in parallel uncertainty per second into the future.",
    )
    parser.add_argument(
        "--social-front-scale",
        type=float,
        default=1.35,
        help="Scale factor that extends human social risk in the forward direction.",
    )
    parser.add_argument(
        "--social-rear-scale",
        type=float,
        default=0.6,
        help="Scale factor applied to the rear social-risk extent behind humans.",
    )
    parser.add_argument(
        "--max-grad",
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm after clipping.",
    )
    parser.add_argument(
        "--disable-gradient-normalization",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--d0",
        type=float,
        default=1.5,
        help="Distance over which risk influence tapers near the goal.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.8,
        help="Directional momentum for smoothing heading changes.",
    )
    parser.add_argument(
        "--grad-smoothing",
        type=float,
        default=0.8,
        help="Exponential smoothing factor applied to the risk gradient.",
    )
    parser.add_argument(
        "--max-turn-angle",
        type=float,
        default=float(np.deg2rad(25.0)),
        help="Maximum allowed heading change per step, in radians.",
    )
    parser.add_argument(
        "--min-forward-dot",
        type=float,
        default=0.2,
        help="Minimum allowed alignment with the goal direction.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Small additive goal bias applied after avoidance shaping.",
    )
    parser.add_argument(
        "--grad-scale-k",
        type=float,
        default=0.35,
        help="Gradient magnitude saturation constant for avoidance scaling.",
    )
    parser.add_argument(
        "--safe-dist",
        type=float,
        default=1.2,
        help="Distance scale for the comfort buffer amplification.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.75,
        help="Hazard risk level above which avoidance is amplified.",
    )
    parser.add_argument(
        "--high-risk-gain",
        type=float,
        default=2.4,
        help="Multiplier applied when the robot enters elevated-risk regions.",
    )
    parser.add_argument(
        "--repulsion-gain",
        type=float,
        default=1.7,
        help="Short-range repulsion gain used to preserve a local comfort buffer.",
    )
    parser.add_argument(
        "--risk-blend-sharpness",
        type=float,
        default=10.0,
        help="Sharpness of the switch from projected to full-gradient avoidance.",
    )
    parser.add_argument(
        "--repulsion-zone-scale",
        type=float,
        default=2.0,
        help="Repulsion activation zone as a multiple of safe distance.",
    )
    parser.add_argument(
        "--barrier-scale",
        type=float,
        default=0.5,
        help="Decay scale of the exponential safety barrier.",
    )
    parser.add_argument(
        "--barrier-gain",
        type=float,
        default=4.0,
        help="Gain applied to the exponential safety barrier term.",
    )
    parser.add_argument(
        "--min-clearance",
        type=float,
        default=0.35,
        help="Emergency clearance threshold for repulsion-only override.",
    )
    parser.add_argument(
        "--grad-epsilon",
        type=float,
        default=0.05,
        help="Ignore effective gradients smaller than this threshold.",
    )
    parser.add_argument(
        "--direction-epsilon",
        type=float,
        default=0.05,
        help="Ignore direction updates smaller than this threshold.",
    )
    parser.add_argument(
        "--velocity-smoothing",
        type=float,
        default=0.8,
        help="Velocity low-pass filter coefficient.",
    )
    parser.add_argument(
        "--alpha-damping",
        type=float,
        default=0.95,
        help="Additional damping factor applied to the avoidance gain.",
    )
    parser.add_argument(
        "--ttc-mid",
        type=float,
        default=2.0,
        help="TTC value at which the robot transitions from nominal speed to slowing.",
    )
    parser.add_argument(
        "--ttc-slope",
        type=float,
        default=0.5,
        help="Slope of the sigmoid used for TTC-based speed scaling.",
    )
    parser.add_argument(
        "--ttc-stop",
        type=float,
        default=1.0,
        help="Emergency stop threshold on minimum TTC.",
    )
    parser.add_argument(
        "--ttc-resume",
        type=float,
        default=1.4,
        help="Resume threshold used for yield hysteresis.",
    )
    parser.add_argument(
        "--safe-clearance",
        type=float,
        default=0.95,
        help="Clearance threshold used together with TTC to trigger yielding.",
    )
    parser.add_argument(
        "--risk-speed-gain",
        type=float,
        default=0.32,
        help="Exponential speed attenuation applied as local hazard risk increases.",
    )
    parser.add_argument(
        "--min-speed-scale",
        type=float,
        default=0.15,
        help="Minimum commanded speed scale outside yield mode.",
    )
    parser.add_argument(
        "--goal-ttc-disable-distance",
        type=float,
        default=0.35,
        help="Disable TTC-based yielding when the robot is within this distance of the goal.",
    )
    parser.add_argument(
        "--ttc-human-speed-threshold",
        type=float,
        default=0.05,
        help="Ignore TTC interactions whose relative speed falls below this threshold.",
    )
    parser.add_argument(
        "--interaction-forward-threshold",
        type=float,
        default=0.15,
        help="Minimum forward-cone alignment required before a human affects TTC.",
    )
    parser.add_argument(
        "--lateral-safe-distance",
        type=float,
        default=1.2,
        help="Ignore TTC interactions whose lateral offset exceeds this distance.",
    )
    parser.add_argument(
        "--lateral-block-threshold",
        type=float,
        default=0.55,
        help="Only allow yield behavior when the critical interaction is this laterally centered.",
    )
    parser.add_argument(
        "--ttc-attention-threshold",
        type=float,
        default=6.5,
        help="Only slow down for TTC interactions inside this time horizon.",
    )
    parser.add_argument(
        "--interaction-beta",
        type=float,
        default=0.6,
        help="Gain applied to the interaction-driven lateral steering force.",
    )
    parser.add_argument(
        "--interaction-prediction-dt",
        type=float,
        default=0.2,
        help="Prediction step used for TTC interaction geometry.",
    )
    parser.add_argument(
        "--clearance-push-gain",
        type=float,
        default=5.2,
        help="Additional lateral push gain when predicted clearance drops below the safety margin.",
    )
    parser.add_argument(
        "--comfort-clearance",
        type=float,
        default=1.15,
        help="Desired passing clearance used for social-distance preservation.",
    )
    parser.add_argument(
        "--interaction-persistence-tau",
        type=float,
        default=0.75,
        help="Time constant of the decaying interaction memory after TTC weakens.",
    )
    parser.add_argument(
        "--interaction-memory-gain",
        type=float,
        default=1.8,
        help="Gain applied to the blended interaction-memory term.",
    )
    parser.add_argument(
        "--interaction-memory-floor-ratio",
        type=float,
        default=0.6,
        help="Minimum fraction of current interaction preserved in memory.",
    )
    parser.add_argument(
        "--interaction-current-blend",
        type=float,
        default=0.4,
        help="Blend weight applied to the current interaction signal.",
    )
    parser.add_argument(
        "--interaction-memory-blend",
        type=float,
        default=0.6,
        help="Blend weight applied to the memory interaction signal.",
    )
    parser.add_argument(
        "--interaction-strength-sharpness",
        type=float,
        default=6.0,
        help="Sharpness used in the continuous interaction-strength gate.",
    )
    parser.add_argument(
        "--interaction-min-strength",
        type=float,
        default=0.035,
        help="Ignore interactions weaker than this threshold.",
    )
    parser.add_argument(
        "--interaction-force-memory-decay",
        type=float,
        default=0.85,
        help="Fixed decay factor for directional interaction-force memory.",
    )
    parser.add_argument(
        "--interaction-force-memory-gain",
        type=float,
        default=1.35,
        help="Gain applied to the persistent directional interaction memory.",
    )
    parser.add_argument(
        "--side-commitment-decay",
        type=float,
        default=0.95,
        help="Decay applied to the committed passing-side direction.",
    )
    parser.add_argument(
        "--side-commitment-gain",
        type=float,
        default=1.4,
        help="Gain applied to the committed passing-side direction.",
    )
    parser.add_argument(
        "--interaction-recovery-threshold",
        type=float,
        default=0.02,
        help="Return to full-speed goal seeking only below this effective interaction level.",
    )
    parser.add_argument(
        "--speed-interaction-gain",
        type=float,
        default=1.05,
        help="How strongly continuous interaction level reduces speed.",
    )
    parser.add_argument(
        "--speed-scale-smoothing",
        type=float,
        default=0.4,
        help="Low-pass filter coefficient for behavior-level speed scaling.",
    )
    parser.add_argument(
        "--interaction-switch-margin",
        type=float,
        default=0.35,
        help="Only switch the primary interacting human when the new candidate exceeds the current one by this relative margin.",
    )
    parser.add_argument(
        "--interaction-conflict-margin",
        type=float,
        default=0.15,
        help="Treat the top two interactions as competing when they fall within this relative margin.",
    )
    parser.add_argument(
        "--interaction-conflict-beta-scale",
        type=float,
        default=0.6,
        help="Scale interaction steering down by this factor when competing interactions are detected.",
    )
    parser.add_argument(
        "--recovery-speed-bias",
        type=float,
        default=0.9,
        help="Extra smooth bias toward full speed once the effective interaction drops below the recovery threshold.",
    )
    parser.add_argument(
        "--yield-score-enter",
        type=float,
        default=0.45,
        help="Continuous yield score needed to enter yield mode.",
    )
    parser.add_argument(
        "--yield-score-exit",
        type=float,
        default=0.25,
        help="Continuous yield score below which yield mode is released.",
    )
    parser.add_argument(
        "--human-noise-std",
        type=float,
        default=0.05,
        help="Standard deviation of heading noise in human goal-directed motion (radians).",
    )
    parser.add_argument(
        "--human-velocity-smoothing",
        type=float,
        default=0.8,
        help="Low-pass filter coefficient for human goal-driven velocity updates.",
    )
    parser.add_argument(
        "--human-goal-tolerance",
        type=float,
        default=0.35,
        help="Distance at which a human samples a new goal.",
    )
    parser.add_argument(
        "--risk-resolution",
        type=int,
        default=60,
        help="Grid resolution used for the risk overlay.",
    )
    parser.add_argument(
        "--risk-update-interval",
        type=int,
        default=5,
        help="Recompute the risk overlay every N animation frames.",
    )
    parser.add_argument(
        "--render-skip",
        type=int,
        default=3,
        help="Number of simulation steps to advance between draw calls.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Throttle animation to simulation time instead of running at maximum speed.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run the simulation headless without matplotlib for maximum throughput.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of headless runs to execute per configuration.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum simulation steps per run in experiment mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for reproducible experiments.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_weak", "no_memory", "no_topk", "no_multi", "no_invariant", "reactive_baseline", "all"],
        help="Experiment variant to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Output directory used for experiment logs, plots, and summaries.",
    )
    parser.add_argument(
        "--log-summary-interval",
        type=int,
        default=0,
        help="Print a compact console summary every N simulation steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--log-detailed-threshold",
        type=float,
        default=0.12,
        help="Trigger detailed interaction logging when effective interaction exceeds this threshold.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional CSV path for structured timestep logs.",
    )
    parser.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Disable console logging while keeping structured logs in memory.",
    )
    parser.add_argument(
        "--hide-risk",
        action="store_true",
        help="Start with the risk overlay hidden. Press v to toggle it.",
    )
    parser.add_argument(
        "--hide-debug",
        action="store_true",
        help="Start with TTC debug visuals hidden. Press d to toggle them.",
    )
    return parser.parse_args()


def _environment_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "world_size": args.world_size,
        "dt": args.dt,
        "alpha": args.alpha,
        "w_h": args.human_weight,
        "w_social": args.social_weight,
        "w_o": args.obstacle_weight,
        "w_g": args.goal_weight,
        "sigma_parallel": args.sigma_parallel,
        "sigma_perp": args.sigma_perp,
        "social_sigma_perp": args.social_sigma_perp,
        "sigma_obs": args.sigma_obs,
        "sigma_zone": args.sigma_zone,
        "sigma_goal": args.sigma_goal,
        "zone_weight": args.zone_weight,
        "zone_inside_gain": args.zone_inside_gain,
        "epsilon": args.epsilon,
        "normalize_gradient": not args.disable_gradient_normalization,
        "max_grad": args.max_grad,
        "prediction_horizon": args.prediction_horizon,
        "prediction_dt": args.prediction_dt,
        "lambda_decay": args.lambda_decay,
        "max_prediction_distance": args.max_prediction_distance,
        "sigma_parallel_growth": args.sigma_parallel_growth,
        "social_front_scale": args.social_front_scale,
        "social_rear_scale": args.social_rear_scale,
        "tau": args.tau,
        "d0": args.d0,
        "momentum": args.momentum,
        "grad_smoothing": args.grad_smoothing,
        "max_turn_angle": args.max_turn_angle,
        "min_forward_dot": args.min_forward_dot,
        "beta": args.beta,
        "grad_scale_k": args.grad_scale_k,
        "safe_dist": args.safe_dist,
        "risk_threshold": args.risk_threshold,
        "high_risk_gain": args.high_risk_gain,
        "repulsion_gain": args.repulsion_gain,
        "risk_blend_sharpness": args.risk_blend_sharpness,
        "repulsion_zone_scale": args.repulsion_zone_scale,
        "barrier_scale": args.barrier_scale,
        "barrier_gain": args.barrier_gain,
        "min_clearance": args.min_clearance,
        "grad_epsilon": args.grad_epsilon,
        "direction_epsilon": args.direction_epsilon,
        "velocity_smoothing": args.velocity_smoothing,
        "alpha_damping": args.alpha_damping,
        "ttc_mid": args.ttc_mid,
        "ttc_slope": args.ttc_slope,
        "ttc_stop": args.ttc_stop,
        "ttc_resume": args.ttc_resume,
        "safe_clearance": args.safe_clearance,
        "risk_speed_gain": args.risk_speed_gain,
        "min_speed_scale": args.min_speed_scale,
        "goal_ttc_disable_distance": args.goal_ttc_disable_distance,
        "ttc_human_speed_threshold": args.ttc_human_speed_threshold,
        "interaction_forward_threshold": args.interaction_forward_threshold,
        "lateral_safe_distance": args.lateral_safe_distance,
        "lateral_block_threshold": args.lateral_block_threshold,
        "ttc_attention_threshold": args.ttc_attention_threshold,
        "interaction_beta": args.interaction_beta,
        "interaction_prediction_dt": args.interaction_prediction_dt,
        "clearance_push_gain": args.clearance_push_gain,
        "comfort_clearance": args.comfort_clearance,
        "interaction_persistence_tau": args.interaction_persistence_tau,
        "interaction_memory_gain": args.interaction_memory_gain,
        "interaction_memory_floor_ratio": args.interaction_memory_floor_ratio,
        "interaction_current_blend": args.interaction_current_blend,
        "interaction_memory_blend": args.interaction_memory_blend,
        "interaction_strength_sharpness": args.interaction_strength_sharpness,
        "interaction_min_strength": args.interaction_min_strength,
        "interaction_force_memory_decay": args.interaction_force_memory_decay,
        "interaction_force_memory_gain": args.interaction_force_memory_gain,
        "side_commitment_decay": args.side_commitment_decay,
        "side_commitment_gain": args.side_commitment_gain,
        "interaction_recovery_threshold": args.interaction_recovery_threshold,
        "speed_interaction_gain": args.speed_interaction_gain,
        "speed_scale_smoothing": args.speed_scale_smoothing,
        "yield_score_enter": args.yield_score_enter,
        "yield_score_exit": args.yield_score_exit,
        "interaction_switch_margin": args.interaction_switch_margin,
        "interaction_conflict_margin": args.interaction_conflict_margin,
        "interaction_conflict_beta_scale": args.interaction_conflict_beta_scale,
        "recovery_speed_bias": args.recovery_speed_bias,
        "human_noise_std": args.human_noise_std,
        "human_velocity_smoothing": args.human_velocity_smoothing,
        "human_goal_tolerance": args.human_goal_tolerance,
        "show_risk": not args.hide_risk,
        "show_debug": not args.hide_debug,
        "real_time": args.real_time,
        "render_skip": args.render_skip,
        "risk_resolution": args.risk_resolution,
        "risk_update_interval": args.risk_update_interval,
        "log_summary_interval": args.log_summary_interval,
        "log_detailed_threshold": args.log_detailed_threshold,
        "log_console_output": not args.quiet_logs,
    }


def main() -> None:
    args = parse_args()
    environment_kwargs = _environment_kwargs_from_args(args)
    experiment_mode = args.no_render or args.num_runs > 1 or args.ablation == "all"

    if experiment_mode:
        experiment_name = (
            f"{args.scenario}_{args.ablation}_seed{args.seed}_runs{max(int(args.num_runs), 1)}"
        )
        experiment_root = Path(args.output_dir).expanduser().resolve() / _sanitize_path_component(
            experiment_name
        )
        configs = build_experiment_configs(
            args.scenario,
            ablation=args.ablation,
            seed=args.seed,
            max_steps=args.max_steps,
            environment_kwargs=environment_kwargs,
        )
        _, summary_rows = run_experiments(
            configs,
            max(int(args.num_runs), 1),
            output_dir=experiment_root,
            generate_plots=True,
        )
        print(f"Experiment outputs saved to: {experiment_root}")
        print(f"Per-run results saved to: {experiment_root / 'run_results.csv'}")
        print(f"Summary results saved to: {experiment_root / 'summary_results.csv'}")
        if summary_rows:
            for row in summary_rows:
                print(
                    "summary | "
                    f"{row['config']} | "
                    f"min_clr={float(row['min_clr_mean']):.3f} +/- {float(row['min_clr_std']):.3f} | "
                    f"success={float(row['success_rate']):.2f} | "
                    f"curv={float(row['curvature_mean']):.3f}"
                )
        return

    ablation_name, ablation_config = ABLATION_PRESETS[args.ablation]
    environment = build_demo_environment(
        scenario=args.scenario,
        seed=args.seed,
        ablation_config=copy.deepcopy(ablation_config),
        **environment_kwargs,
    )
    environment.animate()
    log_path = args.log_file if args.log_file else "logs/run_latest.csv"
    environment.save_logs(log_path)
    print(f"Logs saved to: {log_path}")
    metrics = environment.validation_metrics()
    if metrics:
        print(
            "validation | "
            f"config={ablation_name} | "
            f"min_clr={metrics.get('min_physical_clearance', metrics['min_human_clearance']):.3f} | "
            f"avg_int={metrics['average_interaction_duration']:.3f}s | "
            f"recovery={metrics['average_speed_recovery_time']:.3f}s | "
            f"curv={metrics['mean_abs_curvature']:.3f}"
        )


if __name__ == "__main__":
    main()
