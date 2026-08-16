"""Headless physical experiment for comparing Nova5 controller settings.

The script deliberately changes only runtime constants.  It does not alter the
controller implementation or relax any grasp/collision acceptance rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_sorting_demo as demo_module


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "nova5" / "nova5_sorting_line.xml"


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    tracking_period_s: float
    max_iterations: int
    warning_margin_m: float
    single_arm: bool = False


VARIANTS = (
    ExperimentVariant("dual_baseline", 0.08, 24, 0.10),
    ExperimentVariant("dual_dense_tracking", 0.04, 24, 0.10),
    ExperimentVariant("dual_high_qp_iterations", 0.08, 48, 0.10),
    ExperimentVariant("dual_reduced_warning", 0.08, 24, 0.05),
    ExperimentVariant("single_arm_isolation", 0.08, 24, 0.10, single_arm=True),
)


def run_trial(variant: ExperimentVariant, seed: int, duration_s: float) -> dict[str, Any]:
    """Run one deterministic MuJoCo replay and retain compact diagnostic counts."""
    original = {
        "tracking_period": demo_module.TRACKING_IK_PERIOD_S,
        "max_iterations": demo_module.TRACKING_IK_MAX_ITERATIONS,
        "single_arm": demo_module.SINGLE_ARM_VALIDATION_MODE,
    }
    try:
        demo_module.TRACKING_IK_PERIOD_S = variant.tracking_period_s
        demo_module.TRACKING_IK_MAX_ITERATIONS = variant.max_iterations
        demo_module.SINGLE_ARM_VALIDATION_MODE = variant.single_arm
        demo = demo_module.SortingDemo(MODEL, seed)
        demo.parameters.warning_margin_m = variant.warning_margin_m
        demo._apply_warning_margin()

        event_counts: Counter[str] = Counter()
        important_events: list[dict[str, Any]] = []

        def collect(event: str, **fields: object) -> None:
            event_counts[event] += 1
            if event in {"grasp", "release", "place", "missed", "safety_stop", "joint_plan", "joint_defer"}:
                important_events.append({"time_s": round(float(demo.data.time), 3), "event": event, **fields})

        demo._log = collect  # type: ignore[method-assign]
        started = time.perf_counter()
        demo.run_headless(duration_s)
        wall_s = time.perf_counter() - started
        outcomes = demo.arm_outcomes
        return {
            "variant": variant.name,
            "seed": seed,
            "duration_s": duration_s,
            "wall_s": round(wall_s, 3),
            "placed": len(demo.placed),
            "missed": len(demo.missed),
            "spawned": len(demo.spawned),
            "paused": demo.paused,
            "grasp_count": event_counts["grasp"],
            "joint_plan_count": event_counts["joint_plan"],
            "joint_defer_count": event_counts["joint_defer"],
            "path_abort_count": event_counts["path_abort"],
            "safety_recover_count": event_counts["safety_recover"],
            "safety_stop_count": event_counts["safety_stop"],
            "ik_path_reject_count": event_counts["ik_path_reject"],
            "grasp_wait_count": event_counts["grasp_wait"],
            "arm_a_attempts": outcomes[demo_module.ArmId.A]["attempts"],
            "arm_b_attempts": outcomes[demo_module.ArmId.B]["attempts"],
            "arm_a_placed": outcomes[demo_module.ArmId.A]["placed"],
            "arm_b_placed": outcomes[demo_module.ArmId.B]["placed"],
            "events": important_events,
        }
    except Exception as error:  # Keep a failed seed visible in the study.
        return {
            "variant": variant.name,
            "seed": seed,
            "duration_s": duration_s,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        demo_module.TRACKING_IK_PERIOD_S = original["tracking_period"]
        demo_module.TRACKING_IK_MAX_ITERATIONS = original["max_iterations"]
        demo_module.SINGLE_ARM_VALIDATION_MODE = original["single_arm"]


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        group = [record for record in records if record["variant"] == variant.name]
        completed = [record for record in group if "error" not in record]
        count = max(1, len(completed))
        summary.append(
            {
                "variant": variant.name,
                "trials": len(group),
                "errors": len(group) - len(completed),
                "mean_placed": round(sum(record.get("placed", 0) for record in completed) / count, 3),
                "mean_missed": round(sum(record.get("missed", 0) for record in completed) / count, 3),
                "mean_grasps": round(sum(record.get("grasp_count", 0) for record in completed) / count, 3),
                "joint_plan_rate": round(sum(record.get("joint_plan_count", 0) > 0 for record in completed) / count, 3),
                "mean_joint_defers": round(sum(record.get("joint_defer_count", 0) for record in completed) / count, 3),
                "mean_path_aborts": round(sum(record.get("path_abort_count", 0) for record in completed) / count, 3),
                "mean_safety_recovers": round(sum(record.get("safety_recover_count", 0) for record in completed) / count, 3),
                "mean_ik_path_rejects": round(sum(record.get("ik_path_reject_count", 0) for record in completed) / count, 3),
                "mean_wall_s": round(sum(record.get("wall_s", 0.0) for record in completed) / count, 3),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 100-replay physical Nova5 study.")
    parser.add_argument("--seeds-per-variant", type=int, default=20)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "dual_arm_experiment")
    args = parser.parse_args()
    if args.seeds_per_variant < 1 or args.duration <= 0:
        raise SystemExit("--seeds-per-variant and --duration must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = len(VARIANTS) * args.seeds_per_variant
    started = time.perf_counter()
    for variant in VARIANTS:
        for seed in range(args.seeds_per_variant):
            record = run_trial(variant, seed, args.duration)
            records.append(record)
            print(f"[{len(records):03d}/{total}] {variant.name} seed={seed} placed={record.get('placed')} missed={record.get('missed')} wall={record.get('wall_s')}")

    summary = summarize(records)
    (args.output_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {
        "variants": [asdict(variant) for variant in VARIANTS],
        "trials": total,
        "duration_s": args.duration,
        "wall_minutes": round((time.perf_counter() - started) / 60.0, 3),
        "summary": summary,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
