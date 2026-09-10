"""Run a reproducible 200-trial parameter sweep for the dual-arm line."""

from __future__ import annotations

import concurrent.futures
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


VARIANTS = (
    {"name": "gui_current", "feed_interval_s": 5.0, "belt_speed_mps": 0.09, "max_active_parts": 2.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "slower_feed", "feed_interval_s": 7.0, "belt_speed_mps": 0.09, "max_active_parts": 2.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "slower_belt", "feed_interval_s": 5.0, "belt_speed_mps": 0.07, "max_active_parts": 2.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "single_admission", "feed_interval_s": 5.0, "belt_speed_mps": 0.09, "max_active_parts": 1.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "wide_random", "feed_interval_s": 5.0, "belt_speed_mps": 0.09, "max_active_parts": 2.0, "feed_x_min_m": -0.06, "feed_x_max_m": 0.06},
)


def run_trial(task: tuple[dict[str, object], int]) -> dict[str, object]:
    from run_sorting_demo import DemoParameters, PART_NAMES, SortingDemo

    variant, seed = task
    parameters = DemoParameters(**{key: value for key, value in variant.items() if key != "name"})
    demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed, parameters)
    demo._log = lambda event, **fields: demo.event_log.append({"event": event, **fields})
    while demo.data.time < 360.0 and not demo.paused:
        demo.step()
        if len(demo.spawned) == len(PART_NAMES) and len(demo.placed) + len(demo.missed) == len(PART_NAMES):
            break
    events = [event["event"] for event in demo.event_log]
    active = len(demo.spawned - demo.placed - demo.missed)
    passed = len(demo.spawned) == 10 and len(demo.placed) + len(demo.missed) == 10 and len(demo.missed) <= 2 and active == 0 and not demo.paused and events.count("safety_stop") == 0
    return {"variant": variant["name"], "seed": seed, "placed": len(demo.placed), "missed": len(demo.missed), "active": active, "paused": demo.paused, "safety_stop": events.count("safety_stop"), "passed": passed}


def main() -> None:
    seeds = random.Random(20260817).sample(range(1, 10_000_000), 40)
    tasks = [(variant, seed) for variant in VARIANTS for seed in seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_trial, tasks))
    summary = []
    for variant in VARIANTS:
        group = [result for result in results if result["variant"] == variant["name"]]
        summary.append({"variant": variant, "runs": len(group), "passed": sum(result["passed"] for result in group), "max_missed": max(result["missed"] for result in group), "total_missed": sum(result["missed"] for result in group), "safety_stops": sum(result["safety_stop"] for result in group), "unfinished": sum(result["active"] > 0 for result in group), "failures": [result for result in group if not result["passed"]]})
    output = ROOT / "artifacts" / "parameter_sweep_200.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
