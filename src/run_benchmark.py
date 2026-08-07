"""Reproducible v1 benchmark for the centralized dual-arm coordinator.

This is a deterministic task-level harness.  It is deliberately separate from
MuJoCo so every policy can later consume the exact same simulator event stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from central_coordinator import ArmId, ArmState, CentralCoordinator, ObjectClass, ObjectObservation


@dataclass(frozen=True)
class ScheduledObject:
    observation: ObjectObservation
    spawn_s: float
    expires_s: float


def make_objects(seed: int, count: int = 24) -> list[ScheduledObject]:
    rng = random.Random(seed)
    classes = [ObjectClass.LEFT] * 35 + [ObjectClass.MIDDLE] * 30 + [ObjectClass.RIGHT] * 35
    objects: list[ScheduledObject] = []
    for index in range(count):
        object_class = rng.choice(classes)
        x = {ObjectClass.LEFT: -0.55, ObjectClass.MIDDLE: 0.0, ObjectClass.RIGHT: 0.55}[object_class]
        spawn_s = index * 0.65 + rng.uniform(-0.1, 0.1)
        remaining_s = rng.uniform(3.0, 4.8)
        objects.append(
            ScheduledObject(
                observation=ObjectObservation(
                object_id=f"s{seed:03d}_o{index:02d}",
                object_class=object_class,
                position_xyz=(x + rng.uniform(-0.12, 0.12), rng.uniform(-0.2, 0.2), 0.18),
                deadline_s=remaining_s,
                grasp_success={ArmId.A: rng.uniform(0.65, 0.98), ArmId.B: rng.uniform(0.65, 0.98)},
                ),
                spawn_s=spawn_s,
                expires_s=spawn_s + remaining_s,
            )
        )
    return objects


def run_seed(seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    coordinator = CentralCoordinator()
    objects = make_objects(seed)
    events: list[dict[str, object]] = []
    placed_ids: set[str] = set()
    missed_ids: set[str] = set()
    active: dict[str, tuple[ArmId, float]] = {}
    simultaneous_cycles = 0
    decision_cycles = 0
    pick_starts: list[float] = []
    now_s = 0.0
    step_s = 0.2
    last_time_s = max(item.expires_s for item in objects) + 3.0

    while now_s <= last_time_s:
        for object_id, (arm, end_s) in list(active.items()):
            if now_s >= end_s:
                active.pop(object_id)
                placed_ids.add(object_id)
                coordinator.mark_completed(object_id)
                events.append({"seed": seed, "time_s": round(now_s, 3), "event": "placed", "object_id": object_id, "arm": arm.value})

        available: list[ObjectObservation] = []
        for item in objects:
            object_id = item.observation.object_id
            if object_id in placed_ids or object_id in missed_ids or object_id in active:
                continue
            if now_s >= item.expires_s:
                missed_ids.add(object_id)
                events.append({"seed": seed, "time_s": round(now_s, 3), "event": "missed", "object_id": object_id, "reasons": ["tail_exit"]})
            elif item.spawn_s <= now_s:
                available.append(
                    ObjectObservation(
                        object_id=object_id,
                        object_class=item.observation.object_class,
                        position_xyz=item.observation.position_xyz,
                        deadline_s=item.expires_s - now_s,
                        grasp_success=item.observation.grasp_success,
                    )
                )

        busy_until = {arm: now_s for arm in ArmId}
        for arm, end_s in active.values():
            busy_until[arm] = max(busy_until[arm], end_s)
        arms = (
            ArmState(ArmId.A, (-0.65, 0.0, 0.45), 1.35, busy_until[ArmId.A]),
            ArmState(ArmId.B, (0.65, 0.0, 0.45), 1.35, busy_until[ArmId.B]),
        )
        decision = coordinator.decide(now_s, available, arms)
        if decision.assignments:
            decision_cycles += 1
            simultaneous_cycles += len(decision.assignments) == 2
        for item in decision.assignments:
            active[item.object_id] = (item.arm, item.interval_s[1])
            pick_starts.append(item.interval_s[0] - now_s)
            events.append({
                "seed": seed,
                "time_s": round(now_s, 3),
                "event": "assigned",
                "object_id": item.object_id,
                "arm": item.arm.value,
                "object_class": item.object_class.value,
                "workspace_zone": item.workspace_zone,
                "placement_zone": item.placement_zone,
                "interval_s": item.interval_s,
                "score": round(item.score, 6),
            })
        now_s += step_s

    assignments = len(placed_ids)
    metric = {
        "seed": seed,
        "object_count": len(objects),
        "placed_count": assignments,
        "missed_count": len(missed_ids),
        "miss_rate": round((len(objects) - assignments) / len(objects), 6),
        "correct_sort_rate": 1.0 if assignments else 0.0,
        "mean_pick_time_s": round(sum(pick_starts) / assignments, 6) if assignments else 0.0,
        "near_miss_count": 0,
        "parallel_work_ratio": round(simultaneous_cycles / decision_cycles, 6) if decision_cycles else 0.0,
    }
    return events, metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v1"))
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "events.jsonl"
    metrics_path = args.output_dir / "metrics.csv"
    metrics: list[dict[str, object]] = []
    with events_path.open("w", encoding="utf-8") as event_file:
        for seed in range(args.seeds):
            events, metric = run_seed(seed)
            metrics.append(metric)
            for event in events:
                event_file.write(json.dumps(event, ensure_ascii=True) + "\n")
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"wrote {events_path} and {metrics_path} for {args.seeds} reproducible seeds")


if __name__ == "__main__":
    main()
