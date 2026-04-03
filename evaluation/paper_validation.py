from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phase3 import build_experiment_configs, run_experiments
from src.trajectory_visualization import generate_paper_figures, plot_trajectory


MAIN_SCENARIO_SPECS = (
    {
        "paper_name": "corridor_trap",
        "scenario": "corridor_trap",
        "max_steps": 1200,
        "scenario_class": "controlled",
        "include_baseline": True,
    },
    {
        "paper_name": "crossing_humans",
        "scenario": "crossing_flow",
        "max_steps": 800,
        "scenario_class": "controlled",
        "include_baseline": True,
    },
    {
        "paper_name": "random_crowd",
        "scenario": "random_crowd",
        "max_steps": 800,
        "scenario_class": "controlled",
        "include_baseline": True,
    },
    {
        "paper_name": "structured_crowd",
        "scenario": "structured_crowd",
        "max_steps": 900,
        "scenario_class": "controlled",
        "include_baseline": False,
    },
)

STRESS_SCENARIO_SPECS = (
    {
        "paper_name": "stress_crowd",
        "scenario": "stress_crowd",
        "max_steps": 650,
        "num_runs": 50,
        "scenario_class": "stress",
        "include_baseline": False,
    },
)

DIAGNOSTIC_SCENARIO_SPECS = (
    {
        "paper_name": "permanent_blocking",
        "scenario": "permanent_blocking",
        "max_steps": 800,
        "num_runs": 10,
        "scenario_class": "diagnostic",
        "include_baseline": False,
    },
)

SCENARIO_LABELS = {
    "corridor_trap": "Corridor Trap",
    "crossing_humans": "Crossing Humans",
    "random_crowd": "Random Crowd",
    "structured_crowd": "Structured Crowd",
    "stress_crowd": "Stress Crowd*",
    "permanent_blocking": "Permanent Blocking**",
}

CONTROLLED_SCENARIO_ORDER = (
    "corridor_trap",
    "crossing_humans",
    "random_crowd",
    "structured_crowd",
)
BASELINE_SCENARIO_ORDER = (
    "corridor_trap",
    "crossing_humans",
    "random_crowd",
)

DT_SECONDS = 0.1
PROPOSED_CONTROLLER = "proposed"
BASELINE_CONTROLLER = "reactive_baseline"
NO_INVARIANT_CONTROLLER = "no_invariant"
CONTROLLER_LABELS = {
    PROPOSED_CONTROLLER: "Proposed",
    BASELINE_CONTROLLER: "Reactive baseline",
    NO_INVARIANT_CONTROLLER: "No invariant",
}


def _resolve_output_root(output_dir: str | Path) -> Path:
    output_path = Path(output_dir).expanduser()
    return (REPO_ROOT / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()


def _write_csv_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_plotting() -> Any:
    if not os.environ.get("MPLCONFIGDIR"):
        config_dir = REPO_ROOT / ".mplconfig"
        config_dir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(config_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for paper figure generation") from exc
    return plt


def _float_or_nan(value: str | None) -> float:
    if value in {None, "", "nan", "NaN", "--"}:
        return float("nan")
    return float(value)


def _bool_value(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes"}


def _finite_values(rows: Iterable[dict[str, object] | dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_or_nan(str(row.get(key)) if row.get(key) is not None else None)
        if math.isfinite(value):
            values.append(value)
    return values


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return mean, math.sqrt(variance)


def _percentiles(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    sorted_values = sorted(values)

    def percentile(level: float) -> float:
        index = (len(sorted_values) - 1) * level
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return sorted_values[lower]
        mix = index - lower
        return sorted_values[lower] * (1.0 - mix) + sorted_values[upper] * mix

    return percentile(0.25), percentile(0.50), percentile(0.75)


def _read_log_rows(log_path: Path) -> list[dict[str, str]]:
    if not log_path.exists():
        return []
    with log_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _wait_metrics_from_log(log_path: Path, dt: float = DT_SECONDS) -> tuple[float, float]:
    rows = _read_log_rows(log_path)
    if len(rows) < 2:
        return 0.0, 0.0
    progress = [_float_or_nan(row.get("global_progress")) for row in rows]
    tolerance = 1e-6
    current_wait_steps = 0
    best_wait_steps = 0
    total_wait_steps = 0
    for current_value, next_value in zip(progress, progress[1:]):
        if math.isfinite(current_value) and math.isfinite(next_value) and abs(next_value - current_value) <= tolerance:
            current_wait_steps += 1
            total_wait_steps += 1
            best_wait_steps = max(best_wait_steps, current_wait_steps)
        else:
            current_wait_steps = 0
    return best_wait_steps * dt, total_wait_steps * dt


def _stagnation_metrics_from_log(log_path: Path, dt: float = DT_SECONDS) -> tuple[float, float]:
    rows = _read_log_rows(log_path)
    if len(rows) < 2:
        return 0.0, 0.0
    position_tolerance = 1e-3
    progress_tolerance = 1e-6
    current_steps = 0
    best_steps = 0
    total_steps = 0
    for current_row, next_row in zip(rows, rows[1:]):
        current_x = _float_or_nan(current_row.get("x"))
        current_y = _float_or_nan(current_row.get("y"))
        next_x = _float_or_nan(next_row.get("x"))
        next_y = _float_or_nan(next_row.get("y"))
        current_progress = _float_or_nan(current_row.get("global_progress"))
        next_progress = _float_or_nan(next_row.get("global_progress"))
        if not all(math.isfinite(value) for value in (current_x, current_y, next_x, next_y, current_progress, next_progress)):
            current_steps = 0
            continue
        step_distance = math.hypot(next_x - current_x, next_y - current_y)
        progress_delta = next_progress - current_progress
        if step_distance <= position_tolerance and abs(progress_delta) <= progress_tolerance:
            current_steps += 1
            total_steps += 1
            best_steps = max(best_steps, current_steps)
        else:
            current_steps = 0
    return best_steps * dt, total_steps * dt


def _regression_metrics_from_log(log_path: Path) -> tuple[int, int]:
    rows = _read_log_rows(log_path)
    if not rows:
        return 0, 0
    tolerance = 1e-6
    regression_steps = 0
    regression_events = 0
    active = False
    for row in rows:
        progress = _float_or_nan(row.get("global_progress"))
        frontier = _float_or_nan(row.get("guide_progress_max"))
        if math.isfinite(progress) and math.isfinite(frontier) and progress < frontier - tolerance:
            regression_steps += 1
            if not active:
                regression_events += 1
                active = True
        else:
            active = False
    return regression_events, regression_steps


def _failure_type(row: dict[str, object]) -> str:
    if _bool_value(row.get("success", 0)):
        return "none"
    if float(row.get("unresolved_recoveries", 0.0)) > 0.0:
        return "infeasible_motion"
    max_stagnation_time = float(row.get("max_stagnation_time", 0.0))
    regression_events = int(float(row.get("regression_events", 0.0)))
    sim_time = float(row.get("sim_time", 0.0))
    deadlock_threshold = max(10.0, 0.25 * sim_time)
    if max_stagnation_time >= deadlock_threshold or regression_events > 0:
        return "deadlock"
    return "timeout"


def _enrich_run_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item.pop("failure_reason", None)
        log_path = Path(str(item["log_path"])).expanduser().resolve()
        max_wait_time, total_wait_time = _wait_metrics_from_log(log_path)
        max_stagnation_time, total_stagnation_time = _stagnation_metrics_from_log(log_path)
        regression_events, regression_steps = _regression_metrics_from_log(log_path)
        item["max_wait_time"] = max_wait_time
        item["total_wait_time"] = total_wait_time
        item["max_stagnation_time"] = max_stagnation_time
        item["total_stagnation_time"] = total_stagnation_time
        item["regression_events"] = regression_events
        item["regression_steps"] = regression_steps
        item["failure_type"] = _failure_type(item)
        enriched.append(item)
    return enriched


def _scenario_summary(
    rows: Sequence[dict[str, object]],
    *,
    paper_name: str,
    scenario_class: str,
    controller: str,
) -> dict[str, object]:
    success_values = _finite_values(rows, "success")
    collision_values = _finite_values(rows, "collision")
    unresolved_values = _finite_values(rows, "unresolved_recoveries")
    time_values = _finite_values(rows, "time_to_goal")
    clearance_values = _finite_values(rows, "min_clr")
    wait_values = _finite_values(rows, "max_wait_time")
    stagnation_values = _finite_values(rows, "max_stagnation_time")
    invariant_values = _finite_values(rows, "invariant_recovery_count")
    regression_values = _finite_values(rows, "regression_events")
    failed_branch_values = _finite_values(rows, "failed_branch_count")

    success_rate = _mean_std(success_values)[0]
    collision_rate = _mean_std(collision_values)[0]
    failure_rate = 1.0 - success_rate if math.isfinite(success_rate) else float("nan")
    unresolved_mean, unresolved_std = _mean_std(unresolved_values)
    time_mean, time_std = _mean_std(time_values)
    time_p25, time_p50, time_p75 = _percentiles(time_values)
    clearance_mean, clearance_std = _mean_std(clearance_values)
    clearance_p25, clearance_p50, clearance_p75 = _percentiles(clearance_values)
    wait_mean, wait_std = _mean_std(wait_values)
    wait_p25, wait_p50, wait_p75 = _percentiles(wait_values)
    stagnation_mean, stagnation_std = _mean_std(stagnation_values)
    stagnation_p25, stagnation_p50, stagnation_p75 = _percentiles(stagnation_values)
    invariant_mean, invariant_std = _mean_std(invariant_values)
    invariant_p25, invariant_p50, invariant_p75 = _percentiles(invariant_values)
    regression_mean, regression_std = _mean_std(regression_values)
    regression_p25, regression_p50, regression_p75 = _percentiles(regression_values)
    failed_branch_mean, failed_branch_std = _mean_std(failed_branch_values)

    failure_counts: dict[str, int] = {}
    for row in rows:
        failure_type = str(row.get("failure_type", "unknown"))
        if failure_type == "none":
            continue
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

    dominant_failure = max(failure_counts.items(), key=lambda item: item[1])[0] if failure_counts else "none"
    return {
        "paper_scenario": paper_name,
        "scenario_class": scenario_class,
        "controller": controller,
        "scenario": str(rows[0]["scenario"]) if rows else paper_name,
        "num_trials": len(rows),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "failure_rate": failure_rate,
        "time_to_goal_mean": time_mean,
        "time_to_goal_std": time_std,
        "time_to_goal_p25": time_p25,
        "time_to_goal_p50": time_p50,
        "time_to_goal_p75": time_p75,
        "min_clr_mean": clearance_mean,
        "min_clr_std": clearance_std,
        "min_clr_p25": clearance_p25,
        "min_clr_p50": clearance_p50,
        "min_clr_p75": clearance_p75,
        "unresolved_recoveries_mean": unresolved_mean,
        "unresolved_recoveries_std": unresolved_std,
        "max_wait_time_mean": wait_mean,
        "max_wait_time_std": wait_std,
        "max_wait_time_p25": wait_p25,
        "max_wait_time_p50": wait_p50,
        "max_wait_time_p75": wait_p75,
        "max_stagnation_time_mean": stagnation_mean,
        "max_stagnation_time_std": stagnation_std,
        "max_stagnation_time_p25": stagnation_p25,
        "max_stagnation_time_p50": stagnation_p50,
        "max_stagnation_time_p75": stagnation_p75,
        "invariant_recovery_count_mean": invariant_mean,
        "invariant_recovery_count_std": invariant_std,
        "invariant_recovery_count_p25": invariant_p25,
        "invariant_recovery_count_p50": invariant_p50,
        "invariant_recovery_count_p75": invariant_p75,
        "regression_events_mean": regression_mean,
        "regression_events_std": regression_std,
        "regression_events_p25": regression_p25,
        "regression_events_p50": regression_p50,
        "regression_events_p75": regression_p75,
        "failed_branch_count_mean": failed_branch_mean,
        "failed_branch_count_std": failed_branch_std,
        "dominant_failure_mode": dominant_failure,
    }


def _summarize_runs(run_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["paper_scenario"]), str(row["scenario_class"]), str(row["controller"]))
        grouped.setdefault(key, []).append(dict(row))
    summary_rows = [
        _scenario_summary(
            rows,
            paper_name=paper_name,
            scenario_class=scenario_class,
            controller=controller,
        )
        for (paper_name, scenario_class, controller), rows in grouped.items()
    ]
    class_order = {"controlled": 0, "stress": 1, "diagnostic": 2, "ablation": 3}
    controller_order = {PROPOSED_CONTROLLER: 0, BASELINE_CONTROLLER: 1, NO_INVARIANT_CONTROLLER: 2}
    summary_rows.sort(
        key=lambda row: (
            class_order.get(str(row["scenario_class"]), 99),
            CONTROLLED_SCENARIO_ORDER.index(str(row["paper_scenario"]))
            if str(row["paper_scenario"]) in CONTROLLED_SCENARIO_ORDER
            else 99,
            str(row["paper_scenario"]),
            controller_order.get(str(row["controller"]), 99),
        )
    )
    return summary_rows


def run_paper_evaluation(
    *,
    num_runs: int = 50,
    diagnostic_runs: int = 10,
    base_seed: int = 0,
    output_dir: str | Path = "results/paper_eval",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root_dir = _resolve_output_root(output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    combined_runs: list[dict[str, object]] = []
    existing_rows = _read_csv_rows(root_dir / "combined_run_results.csv")
    all_specs = MAIN_SCENARIO_SPECS + STRESS_SCENARIO_SPECS + DIAGNOSTIC_SCENARIO_SPECS

    for spec in all_specs:
        scenario_dir = root_dir / spec["paper_name"]
        scenario_runs = int(
            spec.get(
                "num_runs",
                num_runs if spec["scenario_class"] in {"controlled", "stress"} else diagnostic_runs,
            )
        )
        environment_kwargs = {
            "show_risk": False,
            "show_debug": False,
            "real_time": False,
            "log_console_output": False,
        }
        environment_kwargs.update(dict(spec.get("environment_kwargs", {})))

        controller_specs = [(PROPOSED_CONTROLLER, "full", spec["scenario_class"])]
        if bool(spec.get("include_baseline", False)):
            controller_specs.append((BASELINE_CONTROLLER, "reactive_baseline", spec["scenario_class"]))
        if spec["scenario_class"] in {"controlled", "stress"}:
            controller_specs.append((NO_INVARIANT_CONTROLLER, "no_invariant", "ablation"))

        for controller_name, ablation_name, scenario_class in controller_specs:
            existing_subset = [
                dict(row)
                for row in existing_rows
                if row.get("paper_scenario") == spec["paper_name"]
                and row.get("scenario_class") == scenario_class
                and row.get("controller") == controller_name
                and Path(str(row.get("log_path", ""))).expanduser().exists()
            ]
            existing_subset.sort(key=lambda row: (int(float(row.get("trial", 0.0))), int(float(row.get("seed", 0.0)))))
            if len(existing_subset) >= scenario_runs:
                combined_runs.extend(existing_subset[:scenario_runs])
                continue

            configs = build_experiment_configs(
                str(spec["scenario"]),
                ablation=ablation_name,
                seed=base_seed,
                max_steps=int(spec["max_steps"]),
                environment_kwargs=environment_kwargs,
            )
            run_rows, _ = run_experiments(
                configs,
                int(scenario_runs),
                output_dir=scenario_dir,
                generate_plots=False,
            )
            for row in run_rows:
                item = dict(row)
                item["paper_scenario"] = spec["paper_name"]
                item["scenario_class"] = scenario_class
                item["controller"] = controller_name
                combined_runs.append(item)

    enriched_runs = _enrich_run_rows(combined_runs)
    summary_rows = _summarize_runs(enriched_runs)
    failure_rows = [
        row
        for row in enriched_runs
        if str(row.get("failure_type", "none")) != "none"
    ]
    failure_rows.sort(
        key=lambda row: (
            0 if str(row.get("paper_scenario")) == "stress_crowd" else 1,
            -float(row.get("max_wait_time", 0.0)),
            -float(row.get("max_stagnation_time", 0.0)),
            str(row.get("paper_scenario")),
            int(float(row.get("seed", 0.0))),
        )
    )
    _write_csv_rows(root_dir / "combined_run_results.csv", enriched_runs)
    _write_csv_rows(root_dir / "combined_summary_results.csv", summary_rows)
    _write_csv_rows(root_dir / "failure_cases.csv", failure_rows)
    return enriched_runs, summary_rows


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def _format_mean_std(mean: float, std: float, digits: int = 2) -> str:
    return f"${mean:.{digits}f}\\pm{std:.{digits}f}$"


def _format_iqr(p25: float, p50: float, p75: float, digits: int = 2) -> str:
    return f"${p50:.{digits}f}\\,[{p25:.{digits}f},{p75:.{digits}f}]$"


def _clearance_digits(std: float) -> int:
    if not math.isfinite(std):
        return 3
    magnitude = abs(std)
    if magnitude >= 1e-3:
        return 3
    if magnitude >= 1e-5:
        return 4
    return 6


def _label_for(paper_name: str) -> str:
    return SCENARIO_LABELS.get(paper_name, paper_name.replace("_", " ").title())


def generate_table_fragments(
    *,
    output_dir: str | Path = "results/paper_eval",
) -> tuple[Path, Path, Path, Path, Path]:
    root_dir = _resolve_output_root(output_dir)
    rows = _read_csv_rows(root_dir / "combined_summary_results.csv")
    if not rows:
        raise FileNotFoundError("combined_summary_results.csv not found; run evaluation first")

    controlled_rows = [
        row
        for row in rows
        if row.get("scenario_class") == "controlled" and row.get("controller", PROPOSED_CONTROLLER) == PROPOSED_CONTROLLER
    ]
    noncontrolled_rows = [
        row
        for row in rows
        if row.get("scenario_class") in {"stress", "diagnostic"}
        and row.get("controller", PROPOSED_CONTROLLER) == PROPOSED_CONTROLLER
    ]
    baseline_rows = [
        row
        for row in rows
        if row.get("scenario_class") == "controlled"
        and row.get("paper_scenario") in BASELINE_SCENARIO_ORDER
        and row.get("controller") in {PROPOSED_CONTROLLER, BASELINE_CONTROLLER}
    ]
    ablation_rows = [
        row
        for row in rows
        if row.get("scenario_class") == "ablation" and row.get("controller") == NO_INVARIANT_CONTROLLER
    ]

    main_table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Scenario & Success & Collision & $T_{\\mathrm{goal}}$ [s] & Min. clr. [m] & Unresolved recoveries \\\\",
        "\\midrule",
    ]
    distribution_table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Scenario & $T_{\\mathrm{goal}}$ p50 [p25,p75] & Min. clr. p50 [p25,p75] & Max wait [s] p50 [p25,p75] & Inv. rec. p50 [p25,p75] \\\\",
        "\\midrule",
    ]
    failure_table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Scenario & Failure & Collision & Unresolved recoveries & Max wait p75 [s] & Dominant mode \\\\",
        "\\midrule",
    ]
    baseline_table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "Scenario & Controller & Success & Collision & $T_{\\mathrm{goal}}$ [s] & Min. clr. [m] \\\\",
        "\\midrule",
    ]
    ablation_table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Scenario & Failure & Regression events p50 [p25,p75] & Max wait [s] p50 [p25,p75] & Dominant mode \\\\",
        "\\midrule",
    ]

    controlled_rows.sort(key=lambda row: CONTROLLED_SCENARIO_ORDER.index(str(row["paper_scenario"])))
    for row in controlled_rows:
        paper_name = str(row["paper_scenario"])
        label = _label_for(paper_name)
        clr_digits = _clearance_digits(float(row["min_clr_std"]))
        main_table_lines.append(
            f"{label} & {_format_percent(float(row['success_rate']))} & "
            f"{_format_percent(float(row['collision_rate']))} & "
            f"{_format_mean_std(float(row['time_to_goal_mean']), float(row['time_to_goal_std']))} & "
            f"{_format_mean_std(float(row['min_clr_mean']), float(row['min_clr_std']), digits=clr_digits)} & "
            f"{_format_mean_std(float(row['unresolved_recoveries_mean']), float(row['unresolved_recoveries_std']))} \\\\"
        )
        distribution_table_lines.append(
            f"{label} & "
            f"{_format_iqr(float(row['time_to_goal_p25']), float(row['time_to_goal_p50']), float(row['time_to_goal_p75']))} & "
            f"{_format_iqr(float(row['min_clr_p25']), float(row['min_clr_p50']), float(row['min_clr_p75']), digits=clr_digits)} & "
            f"{_format_iqr(float(row['max_wait_time_p25']), float(row['max_wait_time_p50']), float(row['max_wait_time_p75']))} & "
            f"{_format_iqr(float(row['invariant_recovery_count_p25']), float(row['invariant_recovery_count_p50']), float(row['invariant_recovery_count_p75']), digits=0)} \\\\"
        )
        failure_table_lines.append(
            f"{label} & {_format_percent(float(row['failure_rate']))} & "
            f"{_format_percent(float(row['collision_rate']))} & "
            f"{_format_mean_std(float(row['unresolved_recoveries_mean']), float(row['unresolved_recoveries_std']))} & "
            f"{float(row['max_wait_time_p75']):.1f} & {str(row['dominant_failure_mode']).replace('_', ' ')} \\\\"
        )

    class_order = {"stress": 0, "diagnostic": 1}
    noncontrolled_rows.sort(
        key=lambda row: (
            class_order.get(str(row["scenario_class"]), 99),
            str(row["paper_scenario"]),
        )
    )
    for row in noncontrolled_rows:
        paper_name = str(row["paper_scenario"])
        label = _label_for(paper_name)
        failure_table_lines.append(
            f"{label} & {_format_percent(float(row['failure_rate']))} & "
            f"{_format_percent(float(row['collision_rate']))} & "
            f"{_format_mean_std(float(row['unresolved_recoveries_mean']), float(row['unresolved_recoveries_std']))} & "
            f"{float(row['max_wait_time_p75']):.1f} & {str(row['dominant_failure_mode']).replace('_', ' ')} \\\\"
        )

    for paper_name in BASELINE_SCENARIO_ORDER:
        scenario_rows = [row for row in baseline_rows if str(row["paper_scenario"]) == paper_name]
        scenario_rows.sort(
            key=lambda row: 0 if str(row.get("controller", PROPOSED_CONTROLLER)) == PROPOSED_CONTROLLER else 1
        )
        for row in scenario_rows:
            label = _label_for(paper_name)
            controller_key = str(row.get("controller", PROPOSED_CONTROLLER))
            time_mean = _float_or_nan(str(row.get("time_to_goal_mean")))
            time_std = _float_or_nan(str(row.get("time_to_goal_std")))
            clr_mean = _float_or_nan(str(row.get("min_clr_mean")))
            clr_std = _float_or_nan(str(row.get("min_clr_std")))
            clr_digits = _clearance_digits(clr_std)
            time_cell = _format_mean_std(time_mean, time_std) if math.isfinite(time_mean) and math.isfinite(time_std) else "--"
            clr_cell = _format_mean_std(clr_mean, clr_std, digits=clr_digits) if math.isfinite(clr_mean) and math.isfinite(clr_std) else "--"
            baseline_table_lines.append(
                f"{label} & {CONTROLLER_LABELS.get(controller_key, controller_key)} & "
                f"{_format_percent(float(row['success_rate']))} & "
                f"{_format_percent(float(row['collision_rate']))} & "
                f"{time_cell} & {clr_cell} \\\\"
            )

    ablation_rows.sort(
        key=lambda row: (
            CONTROLLED_SCENARIO_ORDER.index(str(row["paper_scenario"]))
            if str(row["paper_scenario"]) in CONTROLLED_SCENARIO_ORDER
            else 99,
            str(row["paper_scenario"]),
        )
    )
    for row in ablation_rows:
        paper_name = str(row["paper_scenario"])
        label = _label_for(paper_name)
        ablation_table_lines.append(
            f"{label} & {_format_percent(float(row['failure_rate']))} & "
            f"{_format_iqr(float(row['regression_events_p25']), float(row['regression_events_p50']), float(row['regression_events_p75']), digits=0)} & "
            f"{_format_iqr(float(row['max_wait_time_p25']), float(row['max_wait_time_p50']), float(row['max_wait_time_p75']))} & "
            f"{str(row['dominant_failure_mode']).replace('_', ' ')} \\\\"
        )

    main_table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Controlled-scenario metrics generated directly from logged run summaries. Mean and standard deviation are reported over 50 seeds per scenario.}",
            "\\label{tab:main}",
            "\\end{table}",
        ]
    )
    distribution_table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Distribution-aware summary from logged runs. Entries report median with interquartile range; invariant-recovery counts expose how often runtime rollback and correction were activated.}",
            "\\label{tab:distributions}",
            "\\end{table}",
        ]
    )
    failure_table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Failure-oriented diagnostics derived from logged runs. The starred stress row is a logged adversarial crowd suite used to expose safe failure; the double-starred permanent-blocking row is an out-of-regime diagnostic used to illustrate behavior when admissible-dynamics assumptions are violated.}",
            "\\label{tab:failures}",
            "\\end{table}",
        ]
    )
    baseline_table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Comparison against a reactive projected baseline that retains runtime safety projection but disables guide-planner memory and post-correction progress enforcement. Time-to-goal is reported over successful runs only.}",
            "\\label{tab:baseline}",
            "\\end{table}",
        ]
    )
    ablation_table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Invariant-off ablation generated from logged runs. Regression events count episodes in which executed progress falls below the previously attained frontier.}",
            "\\label{tab:noinvariant}",
            "\\end{table}",
        ]
    )

    main_table_path = root_dir / "main_results_table.tex"
    distribution_table_path = root_dir / "distribution_results_table.tex"
    failure_table_path = root_dir / "failure_analysis_table.tex"
    baseline_table_path = root_dir / "baseline_results_table.tex"
    ablation_table_path = root_dir / "invariant_ablation_table.tex"
    main_table_path.write_text("\n".join(main_table_lines) + "\n", encoding="utf-8")
    distribution_table_path.write_text("\n".join(distribution_table_lines) + "\n", encoding="utf-8")
    failure_table_path.write_text("\n".join(failure_table_lines) + "\n", encoding="utf-8")
    baseline_table_path.write_text("\n".join(baseline_table_lines) + "\n", encoding="utf-8")
    ablation_table_path.write_text("\n".join(ablation_table_lines) + "\n", encoding="utf-8")
    return main_table_path, distribution_table_path, failure_table_path, baseline_table_path, ablation_table_path


def select_representative_trial(
    *,
    output_dir: str | Path = "results/paper_eval",
    paper_scenario: str = "corridor_trap",
) -> Path:
    root_dir = _resolve_output_root(output_dir)
    run_rows = _read_csv_rows(root_dir / "combined_run_results.csv")
    candidates = [
        row
        for row in run_rows
        if row.get("paper_scenario") == paper_scenario
        and row.get("controller", PROPOSED_CONTROLLER) == PROPOSED_CONTROLLER
        and _bool_value(row.get("success", 0))
    ]
    if not candidates:
        raise FileNotFoundError(f"no successful trial found for {paper_scenario}")
    candidates.sort(
        key=lambda row: (
            int(float(row.get("invariant_recovery_count", 0.0))),
            float(row["steps"]),
            int(float(row["seed"])),
        )
    )
    return Path(candidates[0]["log_path"]).expanduser().resolve().parent


def select_failure_trial(
    *,
    output_dir: str | Path = "results/paper_eval",
    paper_scenario: str = "stress_crowd",
) -> Path:
    root_dir = _resolve_output_root(output_dir)
    rows = _read_csv_rows(root_dir / "failure_cases.csv")
    preferred = [
        row
        for row in rows
        if row.get("paper_scenario") == paper_scenario and row.get("controller") == PROPOSED_CONTROLLER
    ]
    fallback = [
        row
        for row in rows
        if row.get("controller") == PROPOSED_CONTROLLER
    ]
    candidates = preferred or fallback
    if not candidates:
        raise FileNotFoundError("no failure cases found")
    candidates.sort(
        key=lambda row: (
            -float(row.get("max_wait_time", 0.0)),
            -float(row.get("max_stagnation_time", 0.0)),
            str(row.get("paper_scenario")),
            int(float(row.get("seed", 0.0))),
        )
    )
    return Path(candidates[0]["log_path"]).expanduser().resolve().parent


def _histogram_rows(output_path: Path, title: str, xlabel: str, datasets: Sequence[tuple[str, Sequence[float], str]]) -> Path:
    plt = _load_plotting()
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    axis.set_facecolor("white")
    for label, values, color in datasets:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            continue
        axis.hist(
            finite,
            bins=min(12, max(5, int(math.sqrt(len(finite))))),
            alpha=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=label,
        )
    axis.set_xlabel(xlabel, fontsize=10)
    axis.set_ylabel("Count", fontsize=10)
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8, loc="best")
    axis.set_title(title, fontsize=11, pad=6.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_distribution_plots(
    *,
    output_dir: str | Path = "results/paper_eval",
    time_output: str | os.PathLike[str] = "time_hist.png",
    clearance_output: str | os.PathLike[str] = "clearance_hist.png",
    recovery_output: str | os.PathLike[str] = "recovery_hist.png",
) -> tuple[Path, Path, Path]:
    root_dir = _resolve_output_root(output_dir)
    rows = _read_csv_rows(root_dir / "combined_run_results.csv")
    if not rows:
        raise FileNotFoundError("combined_run_results.csv not found; run evaluation first")

    colors = {
        "corridor_trap": "#d62728",
        "crossing_humans": "#4c78a8",
        "random_crowd": "#f28e2b",
        "structured_crowd": "#59a14f",
    }

    def datasets(metric: str) -> list[tuple[str, Sequence[float], str]]:
        items: list[tuple[str, Sequence[float], str]] = []
        for paper_name in CONTROLLED_SCENARIO_ORDER:
            subset = [
                row
                for row in rows
                if row.get("paper_scenario") == paper_name
                and row.get("controller", PROPOSED_CONTROLLER) == PROPOSED_CONTROLLER
                and row.get("scenario_class") == "controlled"
            ]
            items.append(
                (
                    _label_for(paper_name).replace("*", ""),
                    _finite_values(subset, metric),
                    colors[paper_name],
                )
            )
        return items

    time_path = _histogram_rows(
        Path(time_output).expanduser().resolve(),
        "Time-to-Goal Distribution",
        "Time to goal [s]",
        datasets("time_to_goal"),
    )
    clearance_path = _histogram_rows(
        Path(clearance_output).expanduser().resolve(),
        "Minimum-Clearance Distribution",
        "Minimum clearance [m]",
        datasets("min_clr"),
    )
    recovery_path = _histogram_rows(
        Path(recovery_output).expanduser().resolve(),
        "Invariant-Recovery Distribution",
        "Invariant recovery count",
        datasets("invariant_recovery_count"),
    )
    return time_path, clearance_path, recovery_path


def generate_paper_assets(
    *,
    output_dir: str | Path = "results/paper_eval",
    representative_scenario: str = "corridor_trap",
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    root_dir = _resolve_output_root(output_dir)
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = select_representative_trial(output_dir=root_dir, paper_scenario=representative_scenario)
    failure_trial_dir = select_failure_trial(output_dir=root_dir, paper_scenario="stress_crowd")
    trajectory_path, progress_path, pipeline_path = generate_paper_figures(
        trial_dir,
        trajectory_output=figures_dir / "corridor_trap.png",
        progress_output=figures_dir / "progress_plot.png",
        pipeline_output=figures_dir / "pipeline.png",
    )
    failure_case_path = plot_trajectory(
        failure_trial_dir,
        figures_dir / "failure_case.png",
    )
    time_hist_path, clearance_hist_path, recovery_hist_path = generate_distribution_plots(
        output_dir=root_dir,
        time_output=figures_dir / "time_hist.png",
        clearance_output=figures_dir / "clearance_hist.png",
        recovery_output=figures_dir / "recovery_hist.png",
    )
    (
        main_table_path,
        distribution_table_path,
        failure_table_path,
        baseline_table_path,
        ablation_table_path,
    ) = generate_table_fragments(output_dir=root_dir)
    return (
        trajectory_path,
        progress_path,
        pipeline_path,
        failure_case_path,
        time_hist_path,
        clearance_hist_path,
        recovery_hist_path,
        main_table_path,
        distribution_table_path,
        failure_table_path,
        baseline_table_path,
        ablation_table_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible paper evaluation suite and generate paper assets.")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of seeds to evaluate per controlled scenario.")
    parser.add_argument("--diagnostic-runs", type=int, default=10, help="Number of seeds for out-of-regime diagnostics.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the evaluation suite.")
    parser.add_argument("--output-dir", type=str, default="results/paper_eval", help="Evaluation output root.")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the experiment sweep and only regenerate tables/figures from existing results.",
    )
    parser.add_argument(
        "--representative-scenario",
        type=str,
        default="corridor_trap",
        choices=[spec["paper_name"] for spec in MAIN_SCENARIO_SPECS],
        help="Scenario used for the representative qualitative figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation and only run the evaluation sweep / table generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_eval:
        run_paper_evaluation(
            num_runs=max(int(args.num_runs), 1),
            diagnostic_runs=max(int(args.diagnostic_runs), 1),
            base_seed=int(args.seed),
            output_dir=args.output_dir,
        )
    if args.skip_figures:
        generate_table_fragments(output_dir=args.output_dir)
        return
    generate_paper_assets(
        output_dir=args.output_dir,
        representative_scenario=args.representative_scenario,
    )


if __name__ == "__main__":
    main()
