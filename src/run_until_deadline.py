"""Run rotating MuJoCo scenarios until the local 08:30 deadline."""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, time as clock_time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SCENARIOS = (
    {"name": "dual_nominal", "max_active_parts": 2.0, "belt_speed_mps": 0.09, "feed_interval_s": 5.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "dual_slow_feed", "max_active_parts": 2.0, "belt_speed_mps": 0.09, "feed_interval_s": 7.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "dual_fast_belt", "max_active_parts": 2.0, "belt_speed_mps": 0.12, "feed_interval_s": 5.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "dual_capacity_three", "max_active_parts": 3.0, "belt_speed_mps": 0.09, "feed_interval_s": 5.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
    {"name": "single_arm_baseline", "max_active_parts": 1.0, "belt_speed_mps": 0.09, "feed_interval_s": 5.0, "feed_x_min_m": -0.03, "feed_x_max_m": 0.03},
)


def run_trial(scenario: dict[str, object], seed: int) -> dict[str, object]:
    from run_sorting_demo import DemoParameters, PART_NAMES, SortingDemo

    params = DemoParameters(**{key: value for key, value in scenario.items() if key != "name"})
    demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed, params)
    demo._log = lambda event, **fields: demo.event_log.append({"event": event, **fields})
    max_parallel = 0
    while demo.data.time < 120.0 and not demo.paused:
        demo.step()
        max_parallel = max(max_parallel, len(demo.missions))
        if len(demo.spawned) == len(PART_NAMES) and len(demo.placed) + len(demo.missed) == len(PART_NAMES):
            break
    events = [item["event"] for item in demo.event_log]
    return {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": scenario["name"],
        "seed": seed,
        "sim_time_s": round(float(demo.data.time), 2),
        "spawned": len(demo.spawned),
        "placed": len(demo.placed),
        "missed": len(demo.missed),
        "active": len(demo.spawned - demo.placed - demo.missed),
        "max_parallel_missions": max_parallel,
        "paused": demo.paused,
        "safety_stops": events.count("safety_stop"),
        "grasps": events.count("grasp"),
        "releases": events.count("release"),
        "miss_reasons": [item.get("reason", "unknown") for item in demo.event_log if item["event"] == "missed"],
    }


def main() -> None:
    deadline = clock_time(8, 30)
    output = ROOT / "artifacts" / "until_0830.jsonl"
    output.parent.mkdir(exist_ok=True)
    rng = random.Random(20260817)
    trial_count = 0
    while datetime.now().time() < deadline:
        scenario = SCENARIOS[trial_count % len(SCENARIOS)]
        result = run_trial(scenario, rng.randrange(1, 10_000_000))
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        trial_count += 1
        if trial_count % 5 == 0:
            print(json.dumps({"trials": trial_count, "last": result}, ensure_ascii=False), flush=True)
        time.sleep(0.2)
    print(json.dumps({"finished": datetime.now().isoformat(timespec="seconds"), "trials": trial_count}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
