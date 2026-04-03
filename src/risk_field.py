from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _as_vector(value: np.ndarray | tuple[float, float] | list[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a 2D vector, got shape {array.shape}")
    return array


def _clamp_position(position: np.ndarray, world_size: np.ndarray) -> np.ndarray:
    return np.clip(position, [0.0, 0.0], world_size)


@dataclass(frozen=True)
class StaticRiskProfile:
    current_risk: float
    future_max_risk: float
    risk_slope: float
    nearest_static_distance: float
    minimum_sampled_distance: float
    trigger_position: np.ndarray
    trigger_gradient: np.ndarray
    trigger_distance: float


class StaticRiskField(Protocol):
    world_size: np.ndarray

    def compute_static_risk(self, position: np.ndarray) -> float:
        ...

    def compute_static_gradient(self, position: np.ndarray) -> np.ndarray:
        ...

    def nearest_static_distance(self, position: np.ndarray) -> float:
        ...


def sample_static_risk_profile(
    risk_field: StaticRiskField,
    position: np.ndarray,
    reference_velocity: np.ndarray,
    *,
    lookahead_time: float,
    sample_count: int,
    activation_radius: float,
    gradient_decay_sigma: float,
) -> StaticRiskProfile:
    position = _as_vector(position, "position")
    world_size = _as_vector(risk_field.world_size, "risk_field.world_size")
    reference_velocity = _as_vector(reference_velocity, "reference_velocity")
    sample_count = max(int(sample_count), 1)
    lookahead_time = max(float(lookahead_time), 0.0)
    activation_radius = max(float(activation_radius), 0.0)
    gradient_decay_sigma = max(float(gradient_decay_sigma), 1e-6)

    times = np.linspace(0.0, lookahead_time, sample_count + 1, dtype=float)
    sampled_positions = np.array(
        [
            _clamp_position(position + time * reference_velocity, world_size)
            for time in times
        ],
        dtype=float,
    )
    risks = np.array(
        [float(risk_field.compute_static_risk(sample_position)) for sample_position in sampled_positions],
        dtype=float,
    )
    sampled_distances = np.array(
        [
            float(risk_field.nearest_static_distance(sample_position))
            for sample_position in sampled_positions
        ],
        dtype=float,
    )
    distance_offsets = np.maximum(sampled_distances - activation_radius, 0.0)
    sample_weights = np.exp(-(distance_offsets**2) / (gradient_decay_sigma**2))
    weighted_risks = risks * sample_weights
    current_risk = float(weighted_risks[0])
    max_index = int(np.argmax(weighted_risks))
    future_max_risk = float(np.max(weighted_risks))

    if len(times) >= 2 and lookahead_time > 0.0:
        delta_time = np.diff(times)
        delta_risk = np.diff(weighted_risks)
        positive_slope = np.divide(
            delta_risk,
            np.maximum(delta_time, 1e-9),
            out=np.zeros_like(delta_risk),
            where=delta_time > 0.0,
        )
        risk_slope = float(np.max(np.maximum(positive_slope, 0.0)))
    else:
        risk_slope = 0.0

    trigger_position = sampled_positions[max_index].copy()
    trigger_gradient = risk_field.compute_static_gradient(trigger_position) * sample_weights[max_index]
    nearest_static_distance = float(risk_field.nearest_static_distance(position))
    minimum_sampled_distance = float(np.min(sampled_distances)) if sampled_distances.size else float("inf")
    trigger_distance = float(sampled_distances[max_index]) if sampled_distances.size else float("inf")
    return StaticRiskProfile(
        current_risk=current_risk,
        future_max_risk=future_max_risk,
        risk_slope=risk_slope,
        nearest_static_distance=nearest_static_distance,
        minimum_sampled_distance=minimum_sampled_distance,
        trigger_position=trigger_position,
        trigger_gradient=trigger_gradient,
        trigger_distance=trigger_distance,
    )
