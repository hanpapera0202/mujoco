"""Run the Nova5 sorting line with a friction-driven circulating conveyor."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCY_DIR = SCRIPT_DIR / ".deps"
if DEPENDENCY_DIR.exists():
    sys.path.insert(0, str(DEPENDENCY_DIR))

import mujoco
import mujoco.viewer


SEGMENT_LENGTH_M = 0.74
SEGMENT_COUNT = 8
# The active conveyor path is shorter than eight plate lengths. This makes
# adjacent plates overlap slightly, removing head-end gaps during circulation.
CONVEYOR_LOOP_LENGTH_M = 5.18
UPSTREAM_CENTER_Y_M = 2.22
BELT_SPEED_MPS = 0.18
TAIL_EXIT_Y_M = -2.25
SPAWN_INTERVAL_S = 1.0
FEEDER_CLEAR_Y_M = 1.60
MAX_ACTIVE_PARTS = 8
PART_NAMES = tuple(f"part_{index:02d}" for index in range(1, 15))


def joint_qpos_address(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = model.body_jntadr[body_id]
    return model.jnt_qposadr[joint_id]


def joint_dof_address(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = model.body_jntadr[body_id]
    return model.jnt_dofadr[joint_id]


def place_part(data: mujoco.MjData, qpos_address: int, rng: random.Random) -> None:
    """Place one free object above the sloped feeder, never directly on the belt."""
    yaw = rng.uniform(-math.pi, math.pi)
    data.qpos[qpos_address : qpos_address + 7] = (
        rng.uniform(-0.20, 0.20),
        rng.uniform(2.20, 2.32),
        0.490,
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    )
    data.qvel[qpos_address : qpos_address + 6] = 0.0


def park_part(data: mujoco.MjData, qpos_address: int, parking_index: int) -> None:
    """Keep queued parts outside the camera and contact area, above the floor plane."""
    data.qpos[qpos_address : qpos_address + 7] = (8.0 + parking_index * 0.20, 8.0, 0.155, 1.0, 0.0, 0.0, 0.0)
    data.qvel[qpos_address : qpos_address + 6] = 0.0


def update_belt(
    data: mujoco.MjData,
    segment_qpos_addresses: list[int],
    segment_dof_addresses: list[int],
) -> None:
    """Translate each belt segment. Teleports occur above the feeder, never at the tail."""
    travelled = (BELT_SPEED_MPS * data.time) % CONVEYOR_LOOP_LENGTH_M
    phase_pitch = CONVEYOR_LOOP_LENGTH_M / SEGMENT_COUNT
    for index, (qpos_address, dof_address) in enumerate(zip(segment_qpos_addresses, segment_dof_addresses)):
        center_y = UPSTREAM_CENTER_Y_M - ((index * phase_pitch + travelled) % CONVEYOR_LOOP_LENGTH_M)
        data.qpos[qpos_address] = center_y
        data.qvel[dof_address] = -BELT_SPEED_MPS


def feeder_is_clear(data: mujoco.MjData, active_parts: set[str], part_qpos_addresses: dict[str, int]) -> bool:
    """Prevent a new part entering before the previous part clears the chute."""
    return all(data.qpos[part_qpos_addresses[name] + 1] < FEEDER_CLEAR_Y_M for name in active_parts)


def run(model_path: Path, seed: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    rng = random.Random(seed)

    segment_qpos_addresses = [
        joint_qpos_address(model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)
    ]
    segment_dof_addresses = [
        joint_dof_address(model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)
    ]
    part_qpos_addresses = {name: joint_qpos_address(model, name) for name in PART_NAMES}
    parking_indices = {name: index for index, name in enumerate(PART_NAMES)}
    active_parts: set[str] = set()
    waiting_parts = list(PART_NAMES)
    next_spawn_time = 0.2

    for part_name, qpos_address in part_qpos_addresses.items():
        park_part(data, qpos_address, parking_indices[part_name])
    update_belt(data, segment_qpos_addresses, segment_dof_addresses)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_wall_time = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            target_time = data.time + min(now - last_wall_time, 0.05)
            last_wall_time = now

            while data.time < target_time:
                update_belt(data, segment_qpos_addresses, segment_dof_addresses)

                if (
                    waiting_parts
                    and len(active_parts) < MAX_ACTIVE_PARTS
                    and data.time >= next_spawn_time
                    and feeder_is_clear(data, active_parts, part_qpos_addresses)
                ):
                    part_name = waiting_parts.pop(0)
                    place_part(data, part_qpos_addresses[part_name], rng)
                    active_parts.add(part_name)
                    next_spawn_time = data.time + SPAWN_INTERVAL_S

                for part_name in tuple(active_parts):
                    qpos_address = part_qpos_addresses[part_name]
                    x, y, z = data.qpos[qpos_address : qpos_address + 3]
                    if y < TAIL_EXIT_Y_M or z < -0.25 or abs(x) > 1.8:
                        park_part(data, qpos_address, parking_indices[part_name])
                        active_parts.remove(part_name)
                        waiting_parts.append(part_name)

                mujoco.mj_step(model, data)

            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Nova5 dual-arm physical sorting line.")
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR.parent / "models" / "nova5" / "nova5_sorting_line.xml",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random feeder seed.")
    args = parser.parse_args()
    run(args.model.resolve(), args.seed)


if __name__ == "__main__":
    main()
