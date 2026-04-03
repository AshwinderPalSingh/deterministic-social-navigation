from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "paper_eval"


def _ensure_writable_matplotlib_config() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    config_dir = REPO_ROOT / ".mplconfig"
    config_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)


def _load_plotting() -> tuple[Any, Any, Any, Any]:
    _ensure_writable_matplotlib_config()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for figure generation") from exc
    return plt, Rectangle, Circle, FancyBboxPatch, Polygon


def _trial_paths(
    trial_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    trial_root = Path(trial_dir).expanduser().resolve()
    log_path = trial_root / "timestep_log.csv"
    metadata_path = trial_root / "trial_metadata.json"
    if not log_path.exists():
        raise FileNotFoundError(f"missing timestep log: {log_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")
    return log_path, metadata_path


def _float_or_nan(value: str | None) -> float:
    if value in {None, "", "--"}:
        return float("nan")
    return float(value)


def _int_or_default(value: str | None, default: int = 0) -> int:
    if value in {None, "", "--"}:
        return default
    return int(float(value))


def _load_log_series(log_path: Path) -> dict[str, np.ndarray]:
    with log_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {
            "step": np.array([], dtype=int),
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "global_progress": np.array([], dtype=float),
            "guide_progress_max": np.array([], dtype=float),
            "failed_branch_count": np.array([], dtype=int),
        }

    def float_series(name: str) -> np.ndarray:
        return np.asarray([_float_or_nan(row.get(name)) for row in rows], dtype=float)

    def int_series(name: str) -> np.ndarray:
        return np.asarray([_int_or_default(row.get(name), default=0) for row in rows], dtype=int)

    return {
        "step": int_series("step"),
        "x": float_series("x"),
        "y": float_series("y"),
        "global_progress": float_series("global_progress"),
        "guide_progress_max": float_series("guide_progress_max"),
        "failed_branch_count": int_series("failed_branch_count"),
    }


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _default_trial_dir() -> Path:
    combined_path = DEFAULT_RESULTS_DIR / "combined_run_results.csv"
    if combined_path.exists():
        with combined_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        candidates: list[dict[str, str]] = []
        for row in rows:
            if row.get("paper_scenario") != "corridor_trap":
                continue
            if row.get("controller") not in {"proposed", "full_system"}:
                continue
            if str(row.get("success", "")).strip().lower() not in {"1", "1.0", "true", "yes"}:
                continue
            if not row.get("log_path"):
                continue
            candidates.append(row)
        if candidates:
            candidates.sort(
                key=lambda row: (
                    _float_or_nan(row.get("invariant_recovery_count")),
                    _float_or_nan(row.get("steps")),
                    _float_or_nan(row.get("seed")),
                )
            )
            return Path(candidates[0]["log_path"]).expanduser().resolve().parent

    fallback_root = DEFAULT_RESULTS_DIR / "corridor_trap" / "full_system"
    fallback_trials = sorted(fallback_root.glob("trial_*_seed_*"))
    if fallback_trials:
        return fallback_trials[0].resolve()
    raise FileNotFoundError(
        "No default logged trial was found. Run `python evaluation/paper_validation.py` first or pass --trial-dir."
    )


def _axis_padding(x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    return 0.08 * max(width, height)


def _failed_branch_split_index(failed_branch_count: np.ndarray) -> int | None:
    if failed_branch_count.size <= 1:
        return None
    deltas = np.diff(failed_branch_count)
    indices = np.nonzero(deltas > 0)[0]
    if indices.size == 0:
        return None
    return int(indices[0] + 1)


def _path_label_point(path_x: np.ndarray, path_y: np.ndarray, fraction: float, dy: float = 0.0) -> tuple[float, float] | None:
    if path_x.size == 0:
        return None
    index = min(path_x.size - 1, max(0, int(round(fraction * (path_x.size - 1)))))
    return float(path_x[index]), float(path_y[index] + dy)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    width = max(int(window), 1)
    if width % 2 == 0:
        width += 1
    width = min(width, values.size if values.size % 2 == 1 else max(values.size - 1, 1))
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _segment_point_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    alpha = float(np.dot(point - start, direction) / length_sq)
    alpha = min(1.0, max(0.0, alpha))
    closest = start + alpha * direction
    return float(np.linalg.norm(point - closest))


def _point_in_obstacle(point: np.ndarray, obstacle: dict[str, Any], margin: float = 0.0) -> bool:
    center = np.asarray(obstacle["center"], dtype=float)
    if obstacle["kind"] == "circle":
        radius = float(obstacle["size"]) + margin
        return float(np.linalg.norm(point - center)) <= radius
    size = np.asarray(obstacle["size"], dtype=float)
    lower_left = center - 0.5 * size - margin
    upper_right = center + 0.5 * size + margin
    return bool(
        lower_left[0] <= point[0] <= upper_right[0]
        and lower_left[1] <= point[1] <= upper_right[1]
    )


def _segment_hits_rectangle(
    start: np.ndarray,
    end: np.ndarray,
    lower_left: np.ndarray,
    upper_right: np.ndarray,
) -> bool:
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= 1e-12:
            if start[axis] < lower_left[axis] or start[axis] > upper_right[axis]:
                return False
            continue
        inv = 1.0 / float(direction[axis])
        t1 = float((lower_left[axis] - start[axis]) * inv)
        t2 = float((upper_right[axis] - start[axis]) * inv)
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def _segment_hits_obstacle(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: dict[str, Any],
    margin: float = 0.0,
) -> bool:
    if _point_in_obstacle(start, obstacle, margin=margin) or _point_in_obstacle(end, obstacle, margin=margin):
        return True
    center = np.asarray(obstacle["center"], dtype=float)
    if obstacle["kind"] == "circle":
        radius = float(obstacle["size"]) + margin
        return _segment_point_distance(center, start, end) <= radius
    size = np.asarray(obstacle["size"], dtype=float)
    lower_left = center - 0.5 * size - margin
    upper_right = center + 0.5 * size + margin
    return _segment_hits_rectangle(start, end, lower_left, upper_right)


def _segment_hits_any_obstacle(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Sequence[dict[str, Any]],
    margin: float = 0.0,
) -> bool:
    return any(_segment_hits_obstacle(start, end, obstacle, margin=margin) for obstacle in obstacles)


def _refine_sample_indices(
    indices: Sequence[int],
    points: np.ndarray,
    obstacles: Sequence[dict[str, Any]],
    margin: float,
) -> list[int]:
    if len(indices) <= 1:
        return list(indices)
    refined: list[int] = [int(indices[0])]
    for index in indices[1:]:
        start_idx = refined[-1]
        end_idx = int(index)
        stack: list[tuple[int, int]] = [(start_idx, end_idx)]
        while stack:
            left, right = stack.pop()
            start = points[left]
            end = points[right]
            if right - left <= 1 or not _segment_hits_any_obstacle(start, end, obstacles, margin=margin):
                if refined[-1] != left:
                    refined.append(left)
                if refined[-1] != right:
                    refined.append(right)
                continue
            mid = left + max(1, (right - left) // 2)
            if mid >= right:
                if refined[-1] != right:
                    refined.append(right)
                continue
            stack.append((mid, right))
            stack.append((left, mid))
    return refined


def _render_path_points(
    path_x: np.ndarray,
    path_y: np.ndarray,
    obstacles: Sequence[dict[str, Any]],
    *,
    window: int = 5,
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if path_x.size <= 2 or path_y.size <= 2:
        return path_x.copy(), path_y.copy()

    smoothed_x = _moving_average(path_x, window)
    smoothed_y = _moving_average(path_y, window)
    smoothed_x[0] = path_x[0]
    smoothed_y[0] = path_y[0]
    smoothed_x[-1] = path_x[-1]
    smoothed_y[-1] = path_y[-1]

    smoothed_points = np.column_stack((smoothed_x, smoothed_y))
    raw_points = np.column_stack((path_x, path_y))
    margin = 1e-3
    for index, point in enumerate(smoothed_points):
        if any(_point_in_obstacle(point, obstacle, margin=margin) for obstacle in obstacles):
            smoothed_points[index] = raw_points[index]

    step = max(int(stride), 1)
    sample_indices = list(range(0, smoothed_points.shape[0], step))
    if sample_indices[-1] != smoothed_points.shape[0] - 1:
        sample_indices.append(smoothed_points.shape[0] - 1)
    sample_indices = _refine_sample_indices(sample_indices, smoothed_points, obstacles, margin=margin)
    sampled = smoothed_points[np.asarray(sample_indices, dtype=int)]
    return sampled[:, 0], sampled[:, 1]


def _arrow_segment_indices(point_count: int, arrow_count: int) -> list[int]:
    segment_count = point_count - 1
    if segment_count <= 0 or arrow_count <= 0:
        return []
    if segment_count <= arrow_count:
        return list(range(segment_count))
    step = segment_count / float(arrow_count + 1)
    return sorted(
        min(segment_count - 1, max(0, int(round((index + 1) * step))))
        for index in range(arrow_count)
    )


def _transition_indices(states: list[str]) -> list[int]:
    if not states:
        return []
    indices = [0]
    for index in range(1, len(states)):
        if states[index] != states[index - 1]:
            indices.append(index)
    if indices[-1] != len(states):
        indices.append(len(states))
    return indices


def _progress_states(values: np.ndarray) -> list[str]:
    if values.size < 2:
        return []
    value_span = float(np.nanmax(values) - np.nanmin(values)) if np.any(np.isfinite(values)) else 0.0
    tolerance = max(1e-6, 1e-3 * max(1.0, value_span))
    states: list[str] = []
    for index in range(values.size - 1):
        if not np.isfinite(values[index]) or not np.isfinite(values[index + 1]):
            states.append("INVALID")
            continue
        delta = float(values[index + 1] - values[index])
        if abs(delta) <= tolerance:
            states.append("WAIT")
        elif delta > tolerance:
            states.append("PROGRESS")
        else:
            states.append("REGRESS")
    return states


def _branch_region_polygon(branch: dict[str, Any]) -> np.ndarray:
    point = np.asarray(branch["point"], dtype=float)
    direction = np.asarray(branch["guide_direction"], dtype=float)
    normal = np.asarray(branch["guide_normal"], dtype=float)
    radius = float(branch["radius"])
    direction_norm = float(np.linalg.norm(direction))
    normal_norm = float(np.linalg.norm(normal))
    if direction_norm <= 1e-9 or normal_norm <= 1e-9:
        return np.empty((0, 2), dtype=float)
    direction = direction / direction_norm
    normal = normal / normal_norm
    forward = 2.4 * radius
    lateral = 0.9 * radius
    start = point - 0.2 * radius * direction
    end = point + forward * direction
    return np.vstack(
        [
            start + lateral * normal,
            end + lateral * normal,
            end - lateral * normal,
            start - lateral * normal,
        ]
    )


def plot_trajectory(
    trial_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] = "corridor_trap.png",
    *,
    show_initial_humans: bool = False,
    arrow_count: int = 1,
    dpi: int = 300,
) -> Path:
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    log_path, metadata_path = _trial_paths(trial_dir)
    series = _load_log_series(log_path)
    metadata = _load_metadata(metadata_path)
    x = series["x"]
    y = series["y"]
    if x.size == 0 or y.size == 0:
        raise ValueError(f"no execution states found in {log_path}")

    plt, Rectangle, Circle, _, Polygon = _load_plotting()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("white")

    for obstacle in metadata.get("obstacles", []):
        center = np.asarray(obstacle["center"], dtype=float)
        if obstacle["kind"] == "circle":
            axis.add_patch(
                Circle(
                    center,
                    radius=float(obstacle["size"]),
                    facecolor="#b0b0b0",
                    edgecolor="#555555",
                    linewidth=1.5,
                    zorder=1,
                )
            )
        else:
            size = np.asarray(obstacle["size"], dtype=float)
            lower_left = center - 0.5 * size
            axis.add_patch(
                Rectangle(
                    lower_left,
                    float(size[0]),
                    float(size[1]),
                    facecolor="#b0b0b0",
                    edgecolor="#555555",
                    linewidth=1.5,
                    zorder=1,
                )
            )

    for zone in metadata.get("zones", []):
        center = np.asarray(zone["center"], dtype=float)
        if zone["kind"] == "circle":
            axis.add_patch(
                Circle(
                    center,
                    radius=float(zone["size"]),
                    facecolor="#f6d6a8",
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1.2,
                )
            )
        else:
            size = np.asarray(zone["size"], dtype=float)
            lower_left = center - 0.5 * size
            axis.add_patch(
                Rectangle(
                    lower_left,
                    float(size[0]),
                    float(size[1]),
                    facecolor="#f6d6a8",
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1.2,
                )
            )

    branch_polygons = []
    for branch in metadata.get("failed_branches", []):
        polygon = _branch_region_polygon(branch)
        if polygon.size == 0:
            continue
        branch_polygons.append(polygon)
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="#f28e2b",
                edgecolor="none",
                alpha=0.14,
                zorder=1.4,
            )
        )

    if show_initial_humans:
        human_points = [human["initial_position"] for human in metadata.get("humans", [])]
        if human_points:
            humans = np.asarray(human_points, dtype=float)
            axis.scatter(
                humans[:, 0],
                humans[:, 1],
                s=12.0,
                c="#4c78a8",
                alpha=0.8,
                marker="o",
                linewidths=0.0,
                zorder=2,
            )

    split_index = _failed_branch_split_index(series["failed_branch_count"])
    if split_index is not None and split_index > 1:
        failed_x, failed_y = _render_path_points(
            x[: split_index + 1],
            y[: split_index + 1],
            metadata.get("obstacles", []),
        )
        axis.plot(
            failed_x,
            failed_y,
            color="#8c8c8c",
            linewidth=2.2,
            linestyle="--",
            solid_capstyle="round",
            dash_capstyle="round",
            dash_joinstyle="round",
            zorder=2.5,
            antialiased=True,
        )
        final_x, final_y = _render_path_points(
            x[split_index:],
            y[split_index:],
            metadata.get("obstacles", []),
        )
    else:
        final_x, final_y = _render_path_points(x, y, metadata.get("obstacles", []))

    axis.plot(
        final_x,
        final_y,
        color="#d62728",
        linewidth=2.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
        antialiased=True,
    )

    for index in _arrow_segment_indices(final_x.size, arrow_count):
        axis.annotate(
            "",
            xy=(float(final_x[index + 1]), float(final_y[index + 1])),
            xytext=(float(final_x[index]), float(final_y[index])),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#d62728",
                lw=1.0,
                shrinkA=0.0,
                shrinkB=0.0,
                mutation_scale=8.0,
            ),
            zorder=3.2,
        )

    robot_start = np.asarray(metadata["robot_start"], dtype=float)
    robot_goal = np.asarray(metadata["robot_goal"], dtype=float)
    axis.scatter(
        [robot_start[0]],
        [robot_start[1]],
        s=78.0,
        c="#2ca02c",
        edgecolors="black",
        linewidths=0.6,
        marker="o",
        zorder=4,
    )
    axis.scatter(
        [robot_goal[0]],
        [robot_goal[1]],
        s=160.0,
        c="#ffbf00",
        edgecolors="black",
        linewidths=0.6,
        marker="*",
        zorder=4,
    )

    text_box = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
    if split_index is not None and split_index > 1:
        failed_label = _path_label_point(x[: split_index + 1], y[: split_index + 1], 0.55, dy=-0.24)
        if failed_label is not None:
            axis.text(
                failed_label[0],
                failed_label[1],
                "Abandoned prefix",
                fontsize=9,
                color="#444444",
                ha="center",
                va="top",
                bbox=text_box,
                zorder=5,
            )
    if branch_polygons:
        polygon = branch_polygons[0]
        centroid = np.mean(polygon, axis=0)
        axis.text(
            float(centroid[0]),
            float(np.max(polygon[:, 1]) + 0.16),
            "Excluded region",
            fontsize=9,
            color="#a05a12",
            ha="center",
            va="bottom",
            bbox=text_box,
            zorder=5,
        )

    x_min = min(float(np.nanmin(x)), float(robot_start[0]), float(robot_goal[0]))
    x_max = max(float(np.nanmax(x)), float(robot_start[0]), float(robot_goal[0]))
    y_min = min(float(np.nanmin(y)), float(robot_start[1]), float(robot_goal[1]))
    y_max = max(float(np.nanmax(y)), float(robot_start[1]), float(robot_goal[1]))
    for obstacle in metadata.get("obstacles", []):
        center = np.asarray(obstacle["center"], dtype=float)
        if obstacle["kind"] == "circle":
            radius = float(obstacle["size"])
            x_min = min(x_min, float(center[0] - radius))
            x_max = max(x_max, float(center[0] + radius))
            y_min = min(y_min, float(center[1] - radius))
            y_max = max(y_max, float(center[1] + radius))
        else:
            size = np.asarray(obstacle["size"], dtype=float)
            lower_left = center - 0.5 * size
            upper_right = center + 0.5 * size
            x_min = min(x_min, float(lower_left[0]))
            x_max = max(x_max, float(upper_right[0]))
            y_min = min(y_min, float(lower_left[1]))
            y_max = max(y_max, float(upper_right[1]))
    padding = _axis_padding(x_min, x_max, y_min, y_max)
    axis.set_xlim(x_min - padding, x_max + padding)
    axis.set_ylim(y_min - padding, y_max + padding)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_progress(
    trial_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] = "progress_plot.png",
    *,
    dpi: int = 300,
) -> Path:
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    log_path, _ = _trial_paths(trial_dir)
    series = _load_log_series(log_path)
    step_values = series["step"]
    progress_values = series["global_progress"]
    frontier_values = series["guide_progress_max"]
    if step_values.size == 0 or progress_values.size == 0 or frontier_values.size == 0:
        raise ValueError(f"missing progress fields in {log_path}")

    plt, _, _, _, _ = _load_plotting()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.6, 3.4))
    axis.set_facecolor("white")
    mark_stride = max(1, int(math.ceil(step_values.size / 10.0)))
    axis.plot(
        step_values,
        progress_values,
        color="#3b6c99",
        linewidth=2.9,
        solid_capstyle="round",
        label=r"$s_{\mathrm{g}}(k)$",
        marker="o",
        markersize=2.8,
        markevery=mark_stride,
        markerfacecolor="white",
        markeredgewidth=0.7,
        zorder=2,
    )
    axis.plot(
        step_values,
        frontier_values,
        color="#9c755f",
        linewidth=2.8,
        linestyle="--",
        solid_capstyle="round",
        label=r"$s_{\max}(k)$",
        marker="s",
        markersize=2.6,
        markevery=mark_stride,
        markerfacecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )

    states = _progress_states(frontier_values)
    transitions = _transition_indices(states)
    marker_indices: list[int] = []
    wait_done = False
    progress_done = False
    span = max(1.0, float(np.nanmax(frontier_values) - np.nanmin(frontier_values)))
    for start, end in zip(transitions[:-1], transitions[1:]):
        state = states[start]
        midpoint = start + max(0, (end - start - 1) // 2)
        marker_indices.append(start)
        if state == "WAIT" and not wait_done:
            axis.text(
                float(step_values[midpoint]),
                float(frontier_values[midpoint] + 0.04 * span),
                "WAIT",
                fontsize=9,
                color="#555555",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
                zorder=4,
            )
            wait_done = True
        elif state == "PROGRESS" and not progress_done:
            axis.text(
                float(step_values[midpoint]),
                float(frontier_values[midpoint] + 0.06 * span),
                "PROGRESS",
                fontsize=9,
                color="#a05a12",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
                zorder=4,
            )
            progress_done = True
    marker_indices.append(int(step_values.size - 1))
    marker_indices = sorted(set(index for index in marker_indices if 0 <= index < step_values.size))
    if marker_indices:
        axis.scatter(
            [float(step_values[index]) for index in marker_indices],
            [float(frontier_values[index]) for index in marker_indices],
            s=18.0,
            color="#9c755f",
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
    axis.set_xlabel("Step", fontsize=10)
    axis.set_ylabel("Progress", fontsize=10)
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.tick_params(axis="both", labelsize=9)
    axis.margins(x=0.03, y=0.12)
    axis.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.8)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_pipeline_diagram(
    output_path: str | os.PathLike[str] = "pipeline.png",
    *,
    dpi: int = 300,
) -> Path:
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    plt, _, _, FancyBboxPatch, _ = _load_plotting()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.6, 2.3))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    labels = [
        "Guide Planner",
        "Controller",
        "Safety Projection",
        "Constraint-Preserving\nSmoothing",
    ]
    x_positions = [0.05, 0.29, 0.53, 0.77]
    box_width = 0.16
    box_height = 0.28
    y_base = 0.26

    bar_x = x_positions[0]
    bar_y = 0.72
    bar_width = (x_positions[-1] + box_width) - x_positions[0]
    bar_height = 0.14
    invariant_bar = FancyBboxPatch(
        (bar_x, bar_y),
        bar_width,
        bar_height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#4f4f4f",
        facecolor="#e3e8f3",
    )
    axis.add_patch(invariant_bar)
    axis.text(
        bar_x + 0.5 * bar_width,
        bar_y + 0.5 * bar_height,
        "Progress Invariant",
        ha="center",
        va="center",
        fontsize=9,
        color="#222222",
    )

    for index, (label, x_base) in enumerate(zip(labels, x_positions)):
        box = FancyBboxPatch(
            (x_base, y_base),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="#4f4f4f",
            facecolor="#ececec",
        )
        axis.add_patch(box)
        axis.text(
            x_base + 0.5 * box_width,
            y_base + 0.5 * box_height,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#222222",
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(x_positions[index + 1] - 0.015, y_base + 0.5 * box_height),
                xytext=(x_base + box_width + 0.015, y_base + 0.5 * box_height),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#555555",
                    lw=1.2,
                    mutation_scale=11.0,
                    shrinkA=0.0,
                    shrinkB=0.0,
                ),
            )

    for constrained_index in (1, 2):
        x_base = x_positions[constrained_index]
        axis.annotate(
            "",
            xy=(x_base + 0.5 * box_width, y_base + box_height + 0.01),
            xytext=(x_base + 0.5 * box_width, bar_y),
            arrowprops=dict(
                arrowstyle="->",
                color="#555555",
                lw=1.2,
                mutation_scale=11.0,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
        )

    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def generate_paper_figures(
    trial_dir: str | os.PathLike[str],
    *,
    trajectory_output: str | os.PathLike[str] = FIGURES_DIR / "corridor_trap.png",
    progress_output: str | os.PathLike[str] = FIGURES_DIR / "progress_plot.png",
    pipeline_output: str | os.PathLike[str] = FIGURES_DIR / "pipeline.png",
) -> tuple[Path, Path, Path]:
    trajectory_path = plot_trajectory(trial_dir, trajectory_output)
    progress_path = plot_progress(trial_dir, progress_output)
    pipeline_path = plot_pipeline_diagram(pipeline_output)
    return trajectory_path, progress_path, pipeline_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from a logged experiment trial.")
    parser.add_argument(
        "--trial-dir",
        type=str,
        default="",
        help="Trial directory containing timestep_log.csv and trial_metadata.json.",
    )
    parser.add_argument("--trajectory-output", type=str, default=str(FIGURES_DIR / "corridor_trap.png"))
    parser.add_argument("--progress-output", type=str, default=str(FIGURES_DIR / "progress_plot.png"))
    parser.add_argument("--pipeline-output", type=str, default=str(FIGURES_DIR / "pipeline.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_dir = args.trial_dir if args.trial_dir else _default_trial_dir()
    generate_paper_figures(
        trial_dir,
        trajectory_output=args.trajectory_output,
        progress_output=args.progress_output,
        pipeline_output=args.pipeline_output,
    )


if __name__ == "__main__":
    main()
