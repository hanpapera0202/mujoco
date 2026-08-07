"""Single-seed dual-Nova5 sorting demonstration driven by the central coordinator.

The demo uses position IK for arm motion and a kinematic attachment after the
gripper closes.  It is intentionally a stable integration demo; pure contact
grasping is a later physics-validation stage.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from central_coordinator import ArmId, ArmState, CentralCoordinator, ObjectClass, ObjectObservation
from run_sorting_line import (
    BELT_SPEED_MPS,
    CONVEYOR_LOOP_LENGTH_M,
    PART_NAMES,
    SEGMENT_COUNT,
    TAIL_EXIT_Y_M,
    UPSTREAM_CENTER_Y_M,
    joint_dof_address,
    joint_qpos_address,
    park_part,
    update_belt,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEPENDENCY_DIR = SCRIPT_DIR / ".deps"
if DEPENDENCY_DIR.exists():
    sys.path.insert(0, str(DEPENDENCY_DIR))

import mujoco
import mujoco.viewer


CONTROL_STEP_S = 0.002
SCHEDULER_PERIOD_S = 0.25
PICK_HEIGHT_M = 0.20
PREGRASP_HEIGHT_M = 0.42
BIN_APPROACH_HEIGHT_M = 0.46
BIN_DROP_HEIGHT_M = 0.19


@dataclass(frozen=True)
class DemoItem:
    part_name: str
    object_class: ObjectClass
    spawn_time_s: float
    spawn_xyz: tuple[float, float, float]
    deadline_s: float


@dataclass
class ArmMission:
    arm: ArmId
    object_id: str
    placement_zone: str
    keyframes: list[tuple[str, float, np.ndarray, float]]
    keyframe_index: int = 0
    keyframe_started_s: float = 0.0
    attached: bool = False

    @property
    def done(self) -> bool:
        return self.keyframe_index >= len(self.keyframes)


class ArmKinematics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm: ArmId) -> None:
        prefix = arm.value
        self.model = model
        self.data = data
        self.arm = arm
        self.joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint{index}") for index in range(1, 7)
        ])
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        self.tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_tool0")
        self.finger_qpos_addresses = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{side}_finger_slide")]
            for side in ("left", "right")
        ])
        self.home_qpos = data.qpos[self.qpos_addresses].copy()

    def tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site_id].copy()

    def solve_position_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray) -> np.ndarray:
        """Damped least-squares position IK, bounded by each Nova5 joint range."""
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = start_qpos
        for _ in range(260):
            mujoco.mj_forward(self.model, self.data)
            error = target_xyz - self.tool_position()
            if np.linalg.norm(error) < 0.012:
                break
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.tool_site_id)
            selected = jacobian[:, self.dof_addresses]
            step = selected.T @ np.linalg.solve(selected @ selected.T + 0.035 * np.eye(3), error)
            step *= min(1.0, 0.16 / max(np.linalg.norm(step), 1e-9))
            updated = self.data.qpos[self.qpos_addresses] + step
            for index, joint_id in enumerate(self.joint_ids):
                low, high = self.model.jnt_range[joint_id]
                updated[index] = np.clip(updated[index], low + 0.02, high - 0.02)
            self.data.qpos[self.qpos_addresses] = updated
        solution = self.data.qpos[self.qpos_addresses].copy()
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return solution

    def set_joint_pose(self, qpos: np.ndarray, gripper_opening: float) -> None:
        self.data.qpos[self.qpos_addresses] = qpos
        self.data.qvel[self.dof_addresses] = 0.0
        self.data.qpos[self.finger_qpos_addresses] = gripper_opening


def place_part(data: mujoco.MjData, qpos_address: int, xyz: tuple[float, float, float]) -> None:
    data.qpos[qpos_address : qpos_address + 7] = (*xyz, 1.0, 0.0, 0.0, 0.0)
    data.qvel[qpos_address : qpos_address + 6] = 0.0


def make_demo_items(seed: int) -> list[DemoItem]:
    """Small, repeatable workload: a contested middle part plus exclusive work."""
    rng = random.Random(seed)
    jitter = lambda amount: rng.uniform(-amount, amount)
    return [
        DemoItem("part_01", ObjectClass.MIDDLE, 0.2, (0.0 + jitter(0.03), 1.02, 0.13), 4.35),
        DemoItem("part_02", ObjectClass.RIGHT, 0.2, (0.18 + jitter(0.03), 0.92, 0.13), 4.75),
        DemoItem("part_03", ObjectClass.LEFT, 0.2, (-0.18 + jitter(0.03), 1.22, 0.13), 4.85),
        DemoItem("part_04", ObjectClass.MIDDLE, 7.0, (0.0 + jitter(0.03), 1.38, 0.13), 4.55),
    ]


def interpolate(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    ratio = min(1.0, max(0.0, ratio))
    return first + (second - first) * ratio


class SortingDemo:
    def __init__(self, model_path: Path, seed: int) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = CONTROL_STEP_S
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self.items = make_demo_items(seed)
        self.by_name = {item.part_name: item for item in self.items}
        self.qpos_addresses = {name: joint_qpos_address(self.model, name) for name in PART_NAMES}
        self.segment_qpos_addresses = [joint_qpos_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.segment_dof_addresses = [joint_dof_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.kinematics = {arm: ArmKinematics(self.model, self.data, arm) for arm in ArmId}
        self.coordinator = CentralCoordinator(pick_speed_mps=0.55, fixed_cycle_s=1.1)
        self.missions: dict[ArmId, ArmMission] = {}
        self.spawned: set[str] = set()
        self.placed: set[str] = set()
        self.missed: set[str] = set()
        self.locked_positions: dict[str, np.ndarray] = {}
        self.last_schedule_s = -SCHEDULER_PERIOD_S
        self.output_offsets = {"left_bin": 0, "right_bin": 0}
        for index, name in enumerate(PART_NAMES):
            park_part(self.data, self.qpos_addresses[name], index)
        update_belt(self.data, self.segment_qpos_addresses, self.segment_dof_addresses)
        mujoco.mj_forward(self.model, self.data)

    def _tool_arm_states(self) -> tuple[ArmState, ArmState]:
        return tuple(
            ArmState(arm, tuple(self.kinematics[arm].tool_position()), 1.55, 999.0 if arm in self.missions else 0.0)
            for arm in ArmId
        )

    def _available_observations(self) -> list[ObjectObservation]:
        observations: list[ObjectObservation] = []
        for name, item in self.by_name.items():
            if name not in self.spawned or name in self.placed or name in self.missed:
                continue
            if any(mission.object_id == name for mission in self.missions.values()):
                continue
            xyz = self.data.qpos[self.qpos_addresses[name] : self.qpos_addresses[name] + 3]
            if xyz[1] < TAIL_EXIT_Y_M:
                self.missed.add(name)
                print(f"[{self.data.time:5.2f}s] MISSED {name}: tail_exit")
                continue
            remaining = min(item.deadline_s, max(0.1, (xyz[1] - TAIL_EXIT_Y_M) / BELT_SPEED_MPS))
            observations.append(ObjectObservation(name, item.object_class, tuple(xyz), remaining, {ArmId.A: 0.93, ArmId.B: 0.93}))
        return observations

    def _plan_mission(self, arm: ArmId, object_id: str, placement_zone: str) -> ArmMission:
        kin = self.kinematics[arm]
        qpos_address = self.qpos_addresses[object_id]
        pick_xyz = self.data.qpos[qpos_address : qpos_address + 3].copy()
        pick_xyz[2] = PICK_HEIGHT_M
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        drop_site = "left_bin_drop" if placement_zone == "left_bin" else "right_bin_drop"
        drop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, drop_site)
        drop = self.data.site_xpos[drop_site_id].copy()
        column = self.output_offsets[placement_zone] % 2
        row = self.output_offsets[placement_zone] // 2
        self.output_offsets[placement_zone] += 1
        drop += np.array(((column - 0.5) * 0.10, (row - 0.5) * 0.08, BIN_DROP_HEIGHT_M))
        bin_approach = drop.copy()
        bin_approach[2] = BIN_APPROACH_HEIGHT_M

        q_home = kin.home_qpos
        q_pregrasp = kin.solve_position_ik(pregrasp, q_home)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp)
        q_bin_approach = kin.solve_position_ik(bin_approach, q_pregrasp)
        q_drop = kin.solve_position_ik(drop, q_bin_approach)
        return ArmMission(
            arm,
            object_id,
            placement_zone,
            [
                ("approach", 1.15, q_pregrasp, 0.035),
                ("descend", 0.70, q_pick, 0.035),
                ("close", 0.40, q_pick, 0.0),
                ("lift", 0.70, q_pregrasp, 0.0),
                ("to_bin", 1.30, q_bin_approach, 0.0),
                ("lower", 0.65, q_drop, 0.0),
                ("open", 0.40, q_drop, 0.035),
                ("retreat", 0.65, q_bin_approach, 0.035),
                ("home", 1.00, q_home, 0.035),
            ],
            keyframe_started_s=self.data.time,
        )

    def _schedule(self) -> None:
        if self.data.time - self.last_schedule_s < SCHEDULER_PERIOD_S:
            return
        self.last_schedule_s = self.data.time
        decision = self.coordinator.decide(self.data.time, self._available_observations(), self._tool_arm_states())
        for assignment in decision.assignments:
            mission = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
            self.missions[assignment.arm] = mission
            print(
                f"[{self.data.time:5.2f}s] ASSIGN {assignment.object_id} ({assignment.object_class.value}) "
                f"-> Robot {assignment.arm.value}, {assignment.placement_zone}"
            )

    def _update_missions(self) -> None:
        for arm, mission in list(self.missions.items()):
            kin = self.kinematics[arm]
            stage, duration, target, opening = mission.keyframes[mission.keyframe_index]
            elapsed = self.data.time - mission.keyframe_started_s
            current = self.data.qpos[kin.qpos_addresses].copy()
            kin.set_joint_pose(interpolate(current, target, min(1.0, CONTROL_STEP_S / max(duration - elapsed, CONTROL_STEP_S))), opening)
            if stage == "lift" and not mission.attached:
                mission.attached = True
                print(f"[{self.data.time:5.2f}s] GRASP {mission.object_id} by Robot {arm.value}")
            if mission.attached:
                object_address = self.qpos_addresses[mission.object_id]
                tool_xyz = kin.tool_position()
                self.data.qpos[object_address : object_address + 7] = (*tool_xyz, 1.0, 0.0, 0.0, 0.0)
                self.data.qvel[object_address : object_address + 6] = 0.0
            if elapsed < duration:
                continue
            if stage == "open" and mission.attached:
                mission.attached = False
                object_address = self.qpos_addresses[mission.object_id]
                tool_xyz = kin.tool_position()
                self.locked_positions[mission.object_id] = tool_xyz.copy()
                self.data.qpos[object_address : object_address + 7] = (*tool_xyz, 1.0, 0.0, 0.0, 0.0)
                print(f"[{self.data.time:5.2f}s] PLACE {mission.object_id} in {mission.placement_zone}")
            mission.keyframe_index += 1
            mission.keyframe_started_s = self.data.time
            if mission.done:
                self.missions.pop(arm)
                self.placed.add(mission.object_id)
                self.coordinator.mark_completed(mission.object_id)

    def step(self) -> None:
        update_belt(self.data, self.segment_qpos_addresses, self.segment_dof_addresses)
        for item in self.items:
            if item.part_name not in self.spawned and self.data.time >= item.spawn_time_s:
                place_part(self.data, self.qpos_addresses[item.part_name], item.spawn_xyz)
                self.spawned.add(item.part_name)
                print(f"[{self.data.time:5.2f}s] INFEED {item.part_name} ({item.object_class.value})")
        self._schedule()
        self._update_missions()
        for name, xyz in self.locked_positions.items():
            address = self.qpos_addresses[name]
            self.data.qpos[address : address + 7] = (*xyz, 1.0, 0.0, 0.0, 0.0)
            self.data.qvel[address : address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def run_headless(self, duration_s: float) -> None:
        while self.data.time < duration_s:
            self.step()
        print(f"finished: placed={sorted(self.placed)} missed={sorted(self.missed)}")

    def run_viewer(self, duration_s: float) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last_wall_time = time.perf_counter()
            while viewer.is_running():
                now = time.perf_counter()
                target_time = self.data.time + min(now - last_wall_time, 0.05)
                last_wall_time = now
                while self.data.time < target_time and self.data.time < duration_s:
                    self.step()
                viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the centralized dual-Nova5 picking demonstration.")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR.parent / "models" / "nova5" / "nova5_sorting_line.xml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--headless", action="store_true", help="Run the scenario without opening the MuJoCo viewer.")
    args = parser.parse_args()
    demo = SortingDemo(args.model.resolve(), args.seed)
    if args.headless:
        demo.run_headless(args.duration)
    else:
        demo.run_viewer(args.duration)


if __name__ == "__main__":
    main()
