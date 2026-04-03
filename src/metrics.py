from __future__ import annotations

from typing import Sequence

import numpy as np


def current_path_efficiency(
    path_length: float,
    start: np.ndarray,
    goal: np.ndarray,
) -> float:
    shortest_path = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)))
    if shortest_path <= 1e-9:
        return 1.0
    return float(path_length / shortest_path)


def _compute_path_length(records: Sequence[dict[str, float | int | str | bool]]) -> float:
    if len(records) < 2:
        return 0.0
    points = np.array(
        [[float(record["x"]), float(record["y"])] for record in records],
        dtype=float,
    )
    segments = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(segments, axis=1)))


def _finite_min(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.min(finite)) if finite else float("inf")


def _finite_max(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.max(finite)) if finite else 0.0


def _compute_recovery_times(
    records: Sequence[dict[str, float | int | str | bool]],
    *,
    nominal_speed_ratio: float,
) -> tuple[list[float], int]:
    human_stop_reasons = {"human_safety", "escape_blocked", "goal_hold"}
    recovery_times: list[float] = []
    unresolved = 0
    episode_start: int | None = None
    def is_interaction_active(record: dict[str, float | int | str | bool]) -> bool:
        state = str(record.get("behavior_state", record.get("state", "")))
        stop_reason = str(record.get("stop_reason", ""))
        if state == "HUMAN_YIELD":
            return True
        if state == "HARD_STOP" and stop_reason in human_stop_reasons:
            return True
        return False

    for index, record in enumerate(records):
        active = is_interaction_active(record)
        if active and episode_start is None:
            episode_start = index
            continue
        if active or episode_start is None:
            continue

        # Recovery is defined on the executed system as the first exit from the
        # interaction-absorbing subset, not three additional stable frames.
        episode_end_time = float(records[index - 1]["time"])
        recovery_times.append(max(0.0, float(record["time"]) - episode_end_time))
        episode_start = None

    if episode_start is not None:
        unresolved += 1
    return recovery_times, unresolved


def compute_navigation_metrics(
    records: Sequence[dict[str, float | int | str | bool]],
    *,
    safety_distance: float,
    path_start: np.ndarray | None = None,
    path_goal: np.ndarray | None = None,
    nominal_speed_ratio: float = 0.9,
) -> dict[str, float]:
    if not records:
        return {}

    minimum_clearance = _finite_min(
        [float(record.get("min_clearance", float("inf"))) for record in records]
    )
    interaction_durations: list[float] = []
    active_start_index: int | None = None
    for index, record in enumerate(records):
        state = str(record.get("behavior_state", record.get("state", "")))
        stop_reason = str(record.get("stop_reason", ""))
        interaction_active = state == "HUMAN_YIELD" or (
            state == "HARD_STOP" and stop_reason in {"human_safety", "escape_blocked", "goal_hold"}
        )
        if interaction_active and active_start_index is None:
            active_start_index = index
        elif not interaction_active and active_start_index is not None:
            interaction_durations.append(
                float(records[index - 1]["time"]) - float(records[active_start_index]["time"])
            )
            active_start_index = None
    if active_start_index is not None:
        interaction_durations.append(
            float(records[-1]["time"]) - float(records[active_start_index]["time"])
        )

    path_length = _compute_path_length(records)
    if path_start is None:
        path_start = np.array([float(records[0]["x"]), float(records[0]["y"])], dtype=float)
    if path_goal is None:
        path_goal = np.array([float(records[-1]["x"]), float(records[-1]["y"])], dtype=float)
    path_efficiency = current_path_efficiency(path_length, path_start, path_goal)

    filtered_curvatures = np.array(
        [abs(float(record.get("curvature", 0.0))) for record in records],
        dtype=float,
    )
    mean_curvature = float(np.mean(filtered_curvatures)) if filtered_curvatures.size else 0.0
    max_curvature = float(np.max(filtered_curvatures)) if filtered_curvatures.size else 0.0
    raw_curvatures = np.array(
        [
            abs(float(record.get("raw_curvature", record.get("curvature", 0.0))))
            for record in records
        ],
        dtype=float,
    )
    raw_curvature_mean = float(np.mean(raw_curvatures)) if raw_curvatures.size else 0.0
    raw_curvature_max = _finite_max(raw_curvatures.tolist()) if raw_curvatures.size else 0.0
    safety_violations = float(
        sum(
            float(record.get("min_clearance", float("inf"))) < (float(safety_distance) - 1e-4)
            for record in records
        )
    )
    recovery_times, unresolved_recoveries = _compute_recovery_times(
        records,
        nominal_speed_ratio=nominal_speed_ratio,
    )
    latest_recovery_time = recovery_times[-1] if recovery_times else 0.0

    return {
        "minimum_clearance": minimum_clearance,
        "min_physical_clearance": minimum_clearance,
        "min_global_clearance": minimum_clearance,
        "average_interaction_duration": float(np.mean(interaction_durations))
        if interaction_durations
        else 0.0,
        "average_speed_recovery_time": float(np.mean(recovery_times)) if recovery_times else 0.0,
        "recovery_time": float(np.mean(recovery_times)) if recovery_times else 0.0,
        "mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else 0.0,
        "latest_recovery_time": float(latest_recovery_time),
        "path_efficiency": path_efficiency,
        "path_length": path_length,
        "mean_abs_curvature": raw_curvature_mean,
        "max_abs_curvature": raw_curvature_max,
        "raw_curvature_mean": raw_curvature_mean,
        "raw_curvature_max": raw_curvature_max,
        "smoothness_mean": mean_curvature,
        "smoothness_max": max_curvature,
        "safety_violations": safety_violations,
        "per_human_safety_violations": safety_violations,
        "safety_distance": float(safety_distance),
        "unresolved_recoveries": float(unresolved_recoveries),
        "interaction_switch_count": float(records[-1].get("interaction_switch_count", 0.0)),
    }
