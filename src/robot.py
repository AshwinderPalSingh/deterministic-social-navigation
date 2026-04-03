from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is not None:
        return _normalize(np.asarray(fallback, dtype=float))
    return np.zeros(2, dtype=float)


def _clamp_position(position: np.ndarray, world_size: np.ndarray) -> np.ndarray:
    return np.clip(position, [0.0, 0.0], world_size)


@dataclass(frozen=True)
class SafetyProjectionResult:
    velocity: np.ndarray
    clearance: float
    safety_margin: float
    projected: bool
    speed_scale: float


def scale_speed_to_safe_margin(
    *,
    position: np.ndarray,
    direction: np.ndarray,
    target_speed: float,
    dt: float,
    safety_distance: float,
    clearance_fn: Callable[[np.ndarray], float],
    max_iterations: int = 18,
) -> SafetyProjectionResult:
    direction = _normalize(direction)
    target_speed = max(float(target_speed), 0.0)
    safe_margin_eps = 1e-4
    if target_speed <= 1e-9 or np.linalg.norm(direction) <= 1e-9:
        clearance = float(clearance_fn(position))
        return SafetyProjectionResult(
            velocity=np.zeros(2, dtype=float),
            clearance=clearance,
            safety_margin=clearance - safety_distance,
            projected=False,
            speed_scale=0.0,
        )

    full_velocity = direction * target_speed
    full_position = position + full_velocity * dt
    full_clearance = float(clearance_fn(full_position))
    if full_clearance >= safety_distance + 1e-9:
        safe_scale = 1.0
        if full_clearance <= safety_distance + safe_margin_eps:
            safe_scale = 1.0 - safe_margin_eps
        candidate_velocity = safe_scale * full_velocity
        candidate_position = position + candidate_velocity * dt
        candidate_clearance = float(clearance_fn(candidate_position))
        return SafetyProjectionResult(
            velocity=candidate_velocity,
            clearance=candidate_clearance,
            safety_margin=candidate_clearance - safety_distance,
            projected=safe_scale < 1.0 - 1e-12,
            speed_scale=safe_scale,
        )

    low = 0.0
    high = 1.0
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        candidate_position = position + mid * full_velocity * dt
        clearance = float(clearance_fn(candidate_position))
        if clearance >= safety_distance:
            low = mid
        else:
            high = mid
    safe_scale = float(np.clip(low * (1.0 - safe_margin_eps), 0.0, 1.0))
    candidate_velocity = safe_scale * full_velocity
    candidate_position = position + candidate_velocity * dt
    clearance = float(clearance_fn(candidate_position))
    return SafetyProjectionResult(
        velocity=candidate_velocity,
        clearance=clearance,
        safety_margin=clearance - safety_distance,
        projected=True,
        speed_scale=safe_scale,
    )


def project_velocity_to_static_safe_set(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    world_size: np.ndarray,
    safety_distance: float,
    clearance_fn: Callable[[np.ndarray], float],
    boundary_normal_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> SafetyProjectionResult:
    velocity = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-9:
        clearance = float(clearance_fn(position))
        return SafetyProjectionResult(
            velocity=np.zeros(2, dtype=float),
            clearance=clearance,
            safety_margin=clearance - safety_distance,
            projected=False,
            speed_scale=0.0,
        )

    def _enforce_projection_clearance_floor(
        result: SafetyProjectionResult,
        *,
        fallback_velocity: np.ndarray,
    ) -> SafetyProjectionResult:
        candidate_velocity = np.asarray(result.velocity, dtype=float)
        next_position = _clamp_position(position + candidate_velocity * dt, world_size)
        clearance = float(clearance_fn(next_position))
        if clearance >= safety_distance:
            candidate_speed = float(np.linalg.norm(candidate_velocity))
            return SafetyProjectionResult(
                velocity=candidate_velocity.copy(),
                clearance=clearance,
                safety_margin=clearance - safety_distance,
                projected=result.projected,
                speed_scale=candidate_speed / max(speed, 1e-6),
            )

        hold_clearance = float(clearance_fn(position))
        if hold_clearance >= safety_distance:
            corrected_velocity = np.zeros(2, dtype=float)
            corrected_clearance = hold_clearance
        else:
            corrected_velocity = candidate_velocity.copy()
            corrected_clearance = clearance
        corrected_speed = float(np.linalg.norm(corrected_velocity))
        return SafetyProjectionResult(
            velocity=corrected_velocity,
            clearance=corrected_clearance,
            safety_margin=corrected_clearance - safety_distance,
            projected=True,
            speed_scale=corrected_speed / max(speed, 1e-6),
        )

    direction = velocity / speed
    scaled_result = scale_speed_to_safe_margin(
        position=position,
        direction=direction,
        target_speed=speed,
        dt=dt,
        safety_distance=safety_distance,
        clearance_fn=clearance_fn,
    )
    if not scaled_result.projected:
        return _enforce_projection_clearance_floor(
            scaled_result,
            fallback_velocity=velocity,
        )

    candidate_position = _clamp_position(position + velocity * dt, world_size)
    outward_normal = _normalize(boundary_normal_fn(candidate_position, velocity), direction)
    inward_component = min(float(np.dot(velocity, outward_normal)), 0.0)
    projected_velocity = velocity - inward_component * outward_normal
    projected_speed = float(np.linalg.norm(projected_velocity))
    if projected_speed <= 1e-9:
        outward_result = scale_speed_to_safe_margin(
            position=position,
            direction=outward_normal,
            target_speed=speed,
            dt=dt,
            safety_distance=safety_distance,
            clearance_fn=clearance_fn,
        )
        if np.linalg.norm(outward_result.velocity) > 1e-9 or outward_result.clearance >= scaled_result.clearance:
            return _enforce_projection_clearance_floor(
                outward_result,
                fallback_velocity=velocity,
            )
        return _enforce_projection_clearance_floor(
            scaled_result,
            fallback_velocity=velocity,
        )

    projected_direction = _normalize(projected_velocity, direction)
    projected_result = scale_speed_to_safe_margin(
        position=position,
        direction=projected_direction,
        target_speed=projected_speed,
        dt=dt,
        safety_distance=safety_distance,
        clearance_fn=clearance_fn,
    )
    if projected_result.clearance + 1e-9 < safety_distance:
        return _enforce_projection_clearance_floor(
            scaled_result,
            fallback_velocity=velocity,
        )
    projected_velocity = projected_result.velocity.copy()
    projected_speed = float(np.linalg.norm(projected_velocity))
    min_escape_speed = 0.05
    if projected_speed < min_escape_speed:
        fallback_direction = _normalize(projected_velocity, projected_direction)
        boosted_velocity = fallback_direction * min_escape_speed
        boosted_position = position + boosted_velocity * dt
        boosted_clearance = float(clearance_fn(boosted_position))
        if boosted_clearance >= safety_distance:
            projected_result = SafetyProjectionResult(
                velocity=boosted_velocity,
                clearance=boosted_clearance,
                safety_margin=boosted_clearance - safety_distance,
                projected=True,
                speed_scale=min_escape_speed / max(speed, 1e-6),
            )

    violation_position = position + velocity * dt
    violation_clearance = float(clearance_fn(violation_position))
    violation_ratio = float(
        np.clip((safety_distance - violation_clearance) / max(safety_distance, 1e-6), 0.0, 1.0)
    )
    blended_velocity = (1.0 - violation_ratio) * scaled_result.velocity + violation_ratio * projected_result.velocity
    blended_speed = float(np.linalg.norm(blended_velocity))
    if blended_speed <= 1e-9:
        return _enforce_projection_clearance_floor(
            projected_result,
            fallback_velocity=velocity,
        )

    blended_result = scale_speed_to_safe_margin(
        position=position,
        direction=_normalize(blended_velocity, projected_direction),
        target_speed=blended_speed,
        dt=dt,
        safety_distance=safety_distance,
        clearance_fn=clearance_fn,
    )
    if np.linalg.norm(blended_result.velocity) >= np.linalg.norm(projected_result.velocity) - 1e-9:
        return _enforce_projection_clearance_floor(
            blended_result,
            fallback_velocity=velocity,
        )
    if np.linalg.norm(projected_result.velocity) <= 1e-9:
        outward_result = scale_speed_to_safe_margin(
            position=position,
            direction=outward_normal,
            target_speed=speed,
            dt=dt,
            safety_distance=safety_distance,
            clearance_fn=clearance_fn,
        )
        if np.linalg.norm(outward_result.velocity) > 1e-9 or outward_result.clearance >= projected_result.clearance:
            return _enforce_projection_clearance_floor(
                outward_result,
                fallback_velocity=velocity,
            )
    return _enforce_projection_clearance_floor(
        projected_result,
        fallback_velocity=velocity,
    )


def integrate_velocity_command(
    *,
    position: np.ndarray,
    current_velocity: np.ndarray,
    commanded_velocity: np.ndarray,
    dt: float,
    world_size: np.ndarray,
    velocity_smoothing: float,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = velocity_smoothing * current_velocity + (1.0 - velocity_smoothing) * commanded_velocity
    speed = float(np.linalg.norm(velocity))
    if speed > max_speed > 1e-9:
        velocity *= max_speed / speed
    next_position = _clamp_position(position + velocity * dt, world_size)
    if next_position[0] != position[0] + velocity[0] * dt:
        velocity[0] = 0.0
    if next_position[1] != position[1] + velocity[1] * dt:
        velocity[1] = 0.0
    return next_position, velocity
