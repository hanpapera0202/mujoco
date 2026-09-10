"""Repeatable multiprocess stability benchmark for the single-item feeder."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))


def run_trial(seed: int, duration_s: float) -> dict[str, int | float | bool]:
    """Run one isolated MuJoCo process and return only measurable outcomes."""
    from run_sorting_demo import PART_NAMES, SortingDemo

    demo = SortingDemo(PROJECT_ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed)

    def quiet_log(event: str, **fields: object) -> None:
        demo.event_log.append({"event": event, **fields})

    demo._log = quiet_log
    while demo.data.time < duration_s and not demo.paused:
        demo.step()
        if len(demo.spawned) == len(PART_NAMES) and len(demo.placed) + len(demo.missed) == len(PART_NAMES):
            break

    events = [event["event"] for event in demo.event_log]
    active = len(demo.spawned - demo.placed - demo.missed)
    return {
        "seed": seed,
        "time_s": round(float(demo.data.time), 2),
        "spawned": len(demo.spawned),
        "placed": len(demo.placed),
        "missed": len(demo.missed),
        "active": active,
        "paused": demo.paused,
        "safety_stop": events.count("safety_stop"),
        "path_abort": events.count("path_abort"),
        "miss_reasons": [
            event.get("reason", "unknown")
            for event in demo.event_log
            if event["event"] == "missed"
        ],
    }


def is_pass(result: dict[str, int | float | bool]) -> bool:
    return (
        result["spawned"] == 10
        and result["placed"] + result["missed"] == 10
        and result["missed"] <= 2
        and result["active"] == 0
        and not result["paused"]
        and result["safety_stop"] == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ten-item single-feeder MuJoCo stability benchmark.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument("--seed", type=int, default=20260816, help="Seed used to generate trial seeds.")
    parser.add_argument("--trial-seeds", help="Comma-separated explicit scenario seeds.")
    parser.add_argument("--output", type=Path, help="Optional JSON result file for long unattended runs.")
    args = parser.parse_args()
    trial_seeds = (
        [int(value) for value in args.trial_seeds.split(",") if value.strip()]
        if args.trial_seeds
        else random.Random(args.seed).sample(range(1, 10_000_000), args.runs)
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_trial, trial_seeds, [args.duration] * args.runs))
    failures = [result for result in results if not is_pass(result)]
    summary = {
        "runs": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "max_missed": max(result["missed"] for result in results),
        "total_missed": sum(result["missed"] for result in results),
        "safety_stops": sum(result["safety_stop"] for result in results),
        "unfinished": sum(result["active"] > 0 or result["spawned"] < 10 for result in results),
        "failures": failures,
    }
    payload = json.dumps(summary, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
