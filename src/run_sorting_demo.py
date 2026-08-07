"""Single-seed dual-Nova5 sorting demonstration driven by the central coordinator.

All arm motion is driven through MuJoCo position control. A part is counted as
grasped only after a finger pad reports a physical MuJoCo contact.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass
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
GRASP_XY_TOLERANCE_M = 0.055


@dataclass
class DemoParameters:
    horizon_s: float = 8.0
    parallel_bonus: float = 2.0
    pick_speed_mps: float = 0.55
    fixed_cycle_s: float = 1.1
    urgency_weight: float = 3.0
    success_weight: float = 2.0
    travel_weight: float = 0.25
    belt_speed_mps: float = 0.24
    simulation_speed: float = 1.0


CSPR_ALGORITHM_ID = "cspr"
CSPR_ALGORITHM_NAME = "CSPR - Centralized Spatiotemporal Reservation"
# Tool x: jaw closing direction, y: vertical finger length, z: conveyor approach.
GRASP_XMAT = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))


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
    intercept_close_s: float
    keyframe_index: int = 0
    keyframe_started_s: float = 0.0
    next_replan_s: float = 0.0
    last_safety_hold_s: float = -1.0
    last_safe_qpos: np.ndarray | None = None
    grasped: bool = False
    grasp_equality_id: int | None = None
    failed: bool = False

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
        self.grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_grasp_zone")
        self.finger_qpos_addresses = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{side}_finger_slide")]
            for side in ("left", "right")
        ])
        self.position_actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_joint{index}_position")
            for index in range(1, 7)
        ])
        self.finger_actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_{side}_finger_position")
            for side in ("left", "right")
        ])
        self.finger_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_{side}_finger_pad")
            for side in ("left", "right")
        }
        self.home_qpos = data.qpos[self.qpos_addresses].copy()

    def tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site_id].copy()

    def grasp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.grasp_site_id].copy()

    def solve_position_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray) -> np.ndarray:
        """Damped 6D IK with a fixed, conveyor-facing parallel-gripper pose."""
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = start_qpos
        for _ in range(360):
            mujoco.mj_forward(self.model, self.data)
            current_xmat = self.data.site_xmat[self.grasp_site_id].reshape(3, 3)
            position_error = target_xyz - self.grasp_position()
            rotation_error = 0.5 * sum(np.cross(current_xmat[:, index], GRASP_XMAT[:, index]) for index in range(3))
            error = np.concatenate((position_error, 0.28 * rotation_error))
            if np.linalg.norm(position_error) < 0.012 and np.linalg.norm(rotation_error) < 0.05:
                break
            position_jacobian = np.zeros((3, self.model.nv))
            rotation_jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, position_jacobian, rotation_jacobian, self.grasp_site_id)
            selected = np.vstack((position_jacobian[:, self.dof_addresses], 0.28 * rotation_jacobian[:, self.dof_addresses]))
            step = selected.T @ np.linalg.solve(selected @ selected.T + 0.045 * np.eye(6), error)
            step *= min(1.0, 0.11 / max(np.linalg.norm(step), 1e-9))
            updated = self.data.qpos[self.qpos_addresses] + step
            for index, joint_id in enumerate(self.joint_ids):
                low, high = self.model.jnt_range[joint_id]
                updated[index] = np.clip(updated[index], low + 0.02, high - 0.02)
            self.data.qpos[self.qpos_addresses] = updated
        solution = self.data.qpos[self.qpos_addresses].copy()
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return solution

    def command_joint_pose(self, qpos: np.ndarray, gripper_opening: float) -> None:
        # The arm trajectory is position-controlled for deterministic replay.
        # The free part is never repositioned here: it can move only by contact.
        self.data.qpos[self.qpos_addresses] = qpos
        self.data.qvel[self.dof_addresses] = 0.0
        self.data.qpos[self.finger_qpos_addresses] = gripper_opening
        self.data.ctrl[self.position_actuator_ids] = qpos
        self.data.ctrl[self.finger_actuator_ids] = gripper_opening


def place_part(data: mujoco.MjData, qpos_address: int, xyz: tuple[float, float, float]) -> None:
    data.qpos[qpos_address : qpos_address + 7] = (*xyz, 1.0, 0.0, 0.0, 0.0)
    data.qvel[qpos_address : qpos_address + 6] = 0.0


def make_demo_items(seed: int) -> list[DemoItem]:
    """Ten deterministic moving parts with mixed exclusive and shared work."""
    rng = random.Random(seed)
    classes = (ObjectClass.RIGHT, ObjectClass.LEFT, ObjectClass.MIDDLE, ObjectClass.RIGHT, ObjectClass.LEFT, ObjectClass.MIDDLE, ObjectClass.RIGHT, ObjectClass.LEFT, ObjectClass.MIDDLE, ObjectClass.RIGHT)
    items: list[DemoItem] = []
    for index, object_class in enumerate(classes, start=1):
        center_x = {ObjectClass.LEFT: -0.12, ObjectClass.MIDDLE: 0.0, ObjectClass.RIGHT: 0.12}[object_class]
        spawn_time_s = 0.2 if index <= 2 else 0.2 + (index - 2) * 0.55
        items.append(DemoItem(f"part_{index:02d}", object_class, spawn_time_s, (center_x + rng.uniform(-0.018, 0.018), 1.20, 0.13), 8.0))
    return items


def interpolate(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    ratio = min(1.0, max(0.0, ratio))
    return first + (second - first) * ratio


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return np.array((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw))


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return quaternion_multiply(quaternion_multiply(quaternion, np.array((0.0, *vector))), np.array((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])))[1:]


class SortingDemo:
    def __init__(self, model_path: Path, seed: int, parameters: DemoParameters | None = None) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = CONTROL_STEP_S
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self.algorithm_id = CSPR_ALGORITHM_ID
        self.parameters = parameters or DemoParameters()
        self.state_lock = threading.RLock()
        self.reset_requested = threading.Event()
        self.paused = False
        self.latest_decision: dict[str, object] = {"assignments": [], "rejected": {}}
        self.last_preflight: dict[str, object] = {"status": "pending", "reason": "waiting_for_task"}
        self.event_log: list[dict[str, object]] = []
        self._reset_state()

    def _reset_state(self) -> None:
        """Restore the exact seed scenario without replacing the viewer's MjData."""
        mujoco.mj_resetData(self.model, self.data)
        self.items = make_demo_items(self.seed)
        self.by_name = {item.part_name: item for item in self.items}
        self.qpos_addresses = {name: joint_qpos_address(self.model, name) for name in PART_NAMES}
        self.part_dof_addresses = {name: joint_dof_address(self.model, name) for name in PART_NAMES}
        self.segment_qpos_addresses = [joint_qpos_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.segment_dof_addresses = [joint_dof_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.kinematics = {arm: ArmKinematics(self.model, self.data, arm) for arm in ArmId}
        self.grasp_equality_ids = {
            (arm, part): mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{arm.value}_{part}_grasp")
            for arm in ArmId for part in (f"part_{index:02d}" for index in range(1, 11))
        }
        self.data.eq_active[:] = False
        for kin in self.kinematics.values():
            kin.command_joint_pose(kin.home_qpos, 0.035)
        self.coordinator = CentralCoordinator(
            pick_speed_mps=self.parameters.pick_speed_mps,
            fixed_cycle_s=self.parameters.fixed_cycle_s,
            horizon_s=self.parameters.horizon_s,
            parallel_bonus=self.parameters.parallel_bonus,
            urgency_weight=self.parameters.urgency_weight,
            success_weight=self.parameters.success_weight,
            travel_weight=self.parameters.travel_weight,
        )
        self.missions: dict[ArmId, ArmMission] = {}
        self.deferred_assignments = {}
        self.spawned: set[str] = set()
        self.placed: set[str] = set()
        self.missed: set[str] = set()
        self.last_schedule_s = -SCHEDULER_PERIOD_S
        self.output_offsets = {"left_bin": 0, "right_bin": 0}
        self.latest_decision = {"assignments": [], "rejected": {}}
        self.last_preflight = {"status": "pending", "reason": "waiting_for_task"}
        self.event_log = []
        for index, name in enumerate(PART_NAMES):
            park_part(self.data, self.qpos_addresses[name], index)
        self._update_belt()
        mujoco.mj_forward(self.model, self.data)

    def request_reset(self) -> None:
        self.reset_requested.set()

    def set_paused(self, paused: bool) -> None:
        with self.state_lock:
            self.paused = paused

    def update_settings(self, values: dict[str, object]) -> None:
        """Apply validated dashboard settings at the next deterministic restart."""
        with self.state_lock:
            if "algorithm" in values:
                algorithm_id = str(values["algorithm"])
                if algorithm_id != CSPR_ALGORITHM_ID:
                    raise ValueError("This algorithm is reserved for a future implementation")
                self.algorithm_id = algorithm_id
            if "seed" in values:
                self.seed = int(values["seed"])
            for field_name in asdict(self.parameters):
                if field_name in values:
                    value = float(values[field_name])
                    if value <= 0.0:
                        raise ValueError(f"{field_name} must be positive")
                    setattr(self.parameters, field_name, value)
        self.request_reset()

    def reset_if_requested(self) -> bool:
        if not self.reset_requested.is_set():
            return False
        self.reset_requested.clear()
        with self.state_lock:
            self._reset_state()
        print(f"[reset] Replayed seed {self.seed}")
        return True

    def snapshot(self) -> dict[str, object]:
        with self.state_lock:
            missions = {
                arm.value: {
                    "object_id": mission.object_id,
                    "stage": mission.keyframes[mission.keyframe_index][0],
                    "placement_zone": mission.placement_zone,
                }
                for arm, mission in self.missions.items()
                if not mission.done
            }
            return {
                "seed": self.seed,
                "algorithm": {"id": self.algorithm_id, "name": CSPR_ALGORITHM_NAME},
                "time_s": round(float(self.data.time), 3),
                "paused": self.paused,
                "parameters": asdict(self.parameters),
                "counts": {"spawned": len(self.spawned), "placed": len(self.placed), "missed": len(self.missed)},
                "missions": missions,
                "deferred": [item.object_id for item in self.deferred_assignments.values()],
                "decision": self.latest_decision,
                "preflight": self.last_preflight,
                "events": self.event_log[-12:],
            }

    def _log(self, event: str, **fields: object) -> None:
        entry = {"time_s": round(float(self.data.time), 3), "event": event, **fields}
        self.event_log.append(entry)
        print(f"[{self.data.time:5.2f}s] {event.upper()} " + " ".join(f"{key}={value}" for key, value in fields.items()))

    def _inter_arm_contacts(self) -> list[tuple[str, str]]:
        contacts: list[tuple[str, str]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or str(contact.geom1)
            second = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or str(contact.geom2)
            if (first.startswith("A_") and second.startswith("B_")) or (first.startswith("B_") and second.startswith("A_")):
                contacts.append((first, second))
        return contacts

    def _geom_description(self, geom_id: int) -> str:
        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name:
            return geom_name
        body_id = self.model.geom_bodyid[geom_id]
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(geom_id)

    @staticmethod
    def _arm_for_description(description: str) -> ArmId | None:
        if description.startswith("A_"):
            return ArmId.A
        if description.startswith("B_"):
            return ArmId.B
        return None

    def _forbidden_contacts(self, data: mujoco.MjData) -> list[tuple[str, str]]:
        """Return arm-arm and arm-environment contacts; object contacts are allowed."""
        forbidden: list[tuple[str, str]] = []
        for index in range(data.ncon):
            contact = data.contact[index]
            first = self._geom_description(contact.geom1)
            second = self._geom_description(contact.geom2)
            first_arm = self._arm_for_description(first)
            second_arm = self._arm_for_description(second)
            if first_arm is not None and second_arm is not None and first_arm is not second_arm:
                forbidden.append((first, second))
            elif first_arm is not None and not second.startswith("part_"):
                forbidden.append((first, second))
            elif second_arm is not None and not first.startswith("part_"):
                forbidden.append((first, second))
        return forbidden

    def _preflight_mission(self, mission: ArmMission) -> tuple[bool, str]:
        """Sample an IK path before it is admitted to the physics simulation."""
        kin = self.kinematics[mission.arm]
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        start = self.data.qpos[kin.qpos_addresses].copy()
        targets = [frame[2] for frame in mission.keyframes]
        previous = start
        for target in targets:
            for ratio in np.linspace(0.2, 1.0, 5):
                trial.qpos[kin.qpos_addresses] = interpolate(previous, target, float(ratio))
                mujoco.mj_forward(self.model, trial)
                contacts = self._forbidden_contacts(trial)
                if contacts:
                    return False, f"{contacts[0][0]} / {contacts[0][1]}"
            previous = target
        return True, "clear"

    def _pose_is_safe(self, arm: ArmId, qpos: np.ndarray, gripper_opening: float | None = None) -> bool:
        kin = self.kinematics[arm]
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        trial.qpos[kin.qpos_addresses] = qpos
        if gripper_opening is not None:
            trial.qpos[kin.finger_qpos_addresses] = gripper_opening
        mujoco.mj_forward(self.model, trial)
        return not self._forbidden_contacts(trial)

    def _update_belt(self) -> None:
        travelled = (self.parameters.belt_speed_mps * self.data.time) % CONVEYOR_LOOP_LENGTH_M
        phase_pitch = CONVEYOR_LOOP_LENGTH_M / SEGMENT_COUNT
        for index, (qpos_address, dof_address) in enumerate(zip(self.segment_qpos_addresses, self.segment_dof_addresses)):
            self.data.qpos[qpos_address] = UPSTREAM_CENTER_Y_M - ((index * phase_pitch + travelled) % CONVEYOR_LOOP_LENGTH_M)
            self.data.qvel[dof_address] = -self.parameters.belt_speed_mps

    def _tool_arm_states(self) -> tuple[ArmState, ArmState]:
        return tuple(
            ArmState(arm, tuple(self.kinematics[arm].tool_position()), 1.55, 999.0 if arm in self.missions or arm in self.deferred_assignments else 0.0)
            for arm in ArmId
        )

    def _available_observations(self) -> list[ObjectObservation]:
        observations: list[ObjectObservation] = []
        for name, item in self.by_name.items():
            if name not in self.spawned or name in self.placed or name in self.missed:
                continue
            if self._object_is_claimed(name):
                continue
            xyz = self.data.qpos[self.qpos_addresses[name] : self.qpos_addresses[name] + 3]
            if xyz[1] < TAIL_EXIT_Y_M:
                self.missed.add(name)
                self._log("missed", object_id=name, reason="tail_exit")
                continue
            downstream_speed = max(0.03, -float(self.data.qvel[self.part_dof_addresses[name] + 1]))
            remaining = min(item.deadline_s, max(0.1, (xyz[1] - TAIL_EXIT_Y_M) / downstream_speed))
            observations.append(ObjectObservation(name, item.object_class, tuple(xyz), remaining, {ArmId.A: 0.93, ArmId.B: 0.93}))
        return observations

    def _object_is_claimed(self, object_id: str) -> bool:
        return any(mission.object_id == object_id for mission in self.missions.values()) or any(
            assignment.object_id == object_id for assignment in self.deferred_assignments.values()
        )

    def _predict_part_position(self, object_id: str, horizon_s: float) -> np.ndarray:
        """Predict interception from MuJoCo's current free-body velocity."""
        xyz = self.data.qpos[self.qpos_addresses[object_id] : self.qpos_addresses[object_id] + 3].copy()
        velocity = self.data.qvel[self.part_dof_addresses[object_id] : self.part_dof_addresses[object_id] + 3].copy()
        if abs(velocity[1]) < 0.03:
            velocity[1] = -self.parameters.belt_speed_mps
        return xyz + velocity * max(0.0, horizon_s)

    def _plan_mission(self, arm: ArmId, object_id: str, placement_zone: str) -> ArmMission:
        kin = self.kinematics[arm]
        time_to_close_s = 1.15 + 0.70 + 0.40
        intercept_close_s = self.data.time + time_to_close_s
        pick_xyz = self._predict_part_position(object_id, time_to_close_s)
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
            intercept_close_s,
            keyframe_started_s=self.data.time,
            next_replan_s=self.data.time + 0.12,
            last_safe_qpos=self.data.qpos[kin.qpos_addresses].copy(),
        )

    def _refresh_intercept(self, mission: ArmMission) -> None:
        if mission.failed or self.data.time < mission.next_replan_s or mission.keyframe_index > 1:
            return
        kin = self.kinematics[mission.arm]
        pick_xyz = self._predict_part_position(mission.object_id, mission.intercept_close_s - self.data.time)
        pick_xyz[2] = PICK_HEIGHT_M
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        start = self.data.qpos[kin.qpos_addresses].copy()
        q_pregrasp = kin.solve_position_ik(pregrasp, start)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp)
        if not self._pose_is_safe(mission.arm, q_pregrasp) or not self._pose_is_safe(mission.arm, q_pick):
            mission.next_replan_s = self.data.time + 0.12
            return
        mission.keyframes[0] = ("approach", 1.15, q_pregrasp, 0.035)
        mission.keyframes[1] = ("descend", 0.70, q_pick, 0.035)
        mission.keyframes[2] = ("close", 0.40, q_pick, 0.0)
        mission.keyframes[3] = ("lift", 0.70, q_pregrasp, 0.0)
        mission.next_replan_s = self.data.time + 0.12

    def _schedule(self) -> None:
        if self.data.time - self.last_schedule_s < SCHEDULER_PERIOD_S:
            return
        self.last_schedule_s = self.data.time
        self._start_safe_deferred_assignments()
        decision = self.coordinator.decide(self.data.time, self._available_observations(), self._tool_arm_states())
        if decision.assignments:
            self.latest_decision = {
                "assignments": [
                    {
                        "object_id": item.object_id,
                        "arm": item.arm.value,
                        "class": item.object_class.value,
                        "zone": item.workspace_zone,
                        "placement": item.placement_zone,
                        "interval_s": [round(value, 3) for value in item.interval_s],
                        "score": round(item.score, 4),
                    }
                    for item in decision.assignments
                ],
                "rejected": decision.rejected,
            }
        for assignment in decision.assignments:
            if self._object_is_claimed(assignment.object_id):
                continue
            if self._may_enter_assignment(assignment):
                self._start_assignment(assignment)
            else:
                self.deferred_assignments[assignment.arm] = assignment
                self._log("reserve_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="central_corridor")

    def _may_enter_assignment(self, assignment) -> bool:
        if assignment.object_class is not ObjectClass.MIDDLE:
            return True
        # A middle task is admitted only after the other arm fully retreats.
        # This deliberately sacrifices one overlap window to give a hard
        # safety boundary for the first physical-contact benchmark.
        return all(arm is assignment.arm for arm in self.missions)

    def _start_safe_deferred_assignments(self) -> None:
        for arm, assignment in list(self.deferred_assignments.items()):
            if arm not in self.missions and self._may_enter_assignment(assignment):
                self.deferred_assignments.pop(arm)
                self._start_assignment(assignment)

    def _start_assignment(self, assignment) -> None:
        mission = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
        safe, reason = self._preflight_mission(mission)
        self.last_preflight = {"status": "clear" if safe else "deferred", "object_id": assignment.object_id, "arm": assignment.arm.value, "reason": reason}
        if not safe:
            self.deferred_assignments[assignment.arm] = assignment
            self._log("reserve_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="path_collision", contact=reason)
            return
        self.missions[assignment.arm] = mission
        self._log("assign", object_id=assignment.object_id, arm=assignment.arm.value, placement=assignment.placement_zone)

    def _update_missions(self) -> None:
        for arm, mission in list(self.missions.items()):
            kin = self.kinematics[arm]
            self._refresh_intercept(mission)
            stage, duration, target, opening = mission.keyframes[mission.keyframe_index]
            elapsed = self.data.time - mission.keyframe_started_s
            current = self.data.qpos[kin.qpos_addresses].copy()
            require_clearance = stage != "close"
            if require_clearance and not self._pose_is_safe(arm, current, opening):
                if mission.last_safe_qpos is not None:
                    kin.command_joint_pose(mission.last_safe_qpos, opening)
                mission.keyframe_started_s += CONTROL_STEP_S
                continue
            next_qpos = interpolate(current, target, min(1.0, CONTROL_STEP_S / max(duration - elapsed, CONTROL_STEP_S)))
            if require_clearance and not self._pose_is_safe(arm, next_qpos, opening):
                mission.keyframe_started_s += CONTROL_STEP_S
                if self.data.time - mission.last_safety_hold_s >= 0.5:
                    mission.last_safety_hold_s = self.data.time
                    self._log("reserve_wait", object_id=mission.object_id, arm=arm.value, reason="step_collision_guard")
                continue
            kin.command_joint_pose(next_qpos, opening)
            mission.last_safe_qpos = next_qpos.copy()
            if stage == "close" and elapsed >= duration and not mission.grasped:
                if not self._confirm_grasp(arm, mission):
                    continue
            if elapsed < duration:
                continue
            if stage == "open" and mission.grasped:
                self.data.eq_active[mission.grasp_equality_id] = False
                self._log("release", object_id=mission.object_id, placement=mission.placement_zone)
            mission.keyframe_index += 1
            mission.keyframe_started_s = self.data.time
            if mission.done:
                self.missions.pop(arm)
                if not mission.failed and self._part_is_in_target_bin(mission.object_id, mission.placement_zone):
                    self.placed.add(mission.object_id)
                    self._log("place", object_id=mission.object_id, placement=mission.placement_zone)
                elif not mission.failed:
                    self.missed.add(mission.object_id)
                    self._log("missed", object_id=mission.object_id, reason="placement_not_verified")
                self.coordinator.mark_completed(mission.object_id)

    def _confirm_grasp(self, arm: ArmId, mission: ArmMission) -> bool:
        """Accept a grasp only after a physical finger-pad contact is reported."""
        kin = self.kinematics[arm]
        part_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mission.object_id)
        part_geoms = set(range(self.model.body_geomadr[part_body], self.model.body_geomadr[part_body] + self.model.body_geomnum[part_body]))
        touching_fingers: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if pair.intersection(part_geoms):
                touching_fingers.update(kin.finger_geom_ids.intersection(pair))
        if touching_fingers:
            mission.grasped = True
            mission.grasp_equality_id = self._activate_grasp_constraint(arm, mission.object_id)
            self._log("grasp", object_id=mission.object_id, arm=arm.value, contact="finger_physical", finger_count=len(touching_fingers), grasp_constraint="active")
            return True
        self.missed.add(mission.object_id)
        self.coordinator.mark_completed(mission.object_id)
        self._log(
            "missed",
            object_id=mission.object_id,
            reason="no_finger_contact",
        )
        mission.failed = True
        mission.keyframes = [("recover", 0.6, self.data.qpos[kin.qpos_addresses].copy(), 0.035), ("home", 1.0, kin.home_qpos, 0.035)]
        mission.keyframe_index = 0
        mission.keyframe_started_s = self.data.time
        return False

    def _activate_grasp_constraint(self, arm: ArmId, object_id: str) -> int:
        """Hold the current contact pose with a MuJoCo weld; never teleport the part."""
        equality_id = self.grasp_equality_ids[(arm, object_id)]
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm.value}_gripper")
        part_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        gripper_pos = self.data.xpos[gripper_id]
        gripper_quat = self.data.xquat[gripper_id]
        part_pos = self.data.xpos[part_id]
        part_quat = self.data.xquat[part_id]
        inverse_gripper = np.array((gripper_quat[0], -gripper_quat[1], -gripper_quat[2], -gripper_quat[3]))
        self.model.eq_data[equality_id, :3] = quaternion_rotate(inverse_gripper, part_pos - gripper_pos)
        self.model.eq_data[equality_id, 3:7] = quaternion_multiply(inverse_gripper, part_quat)
        self.data.eq_active[equality_id] = True
        return equality_id

    def _part_is_in_target_bin(self, object_id: str, placement_zone: str) -> bool:
        drop_site = "left_bin_drop" if placement_zone == "left_bin" else "right_bin_drop"
        drop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, drop_site)
        part_xyz = self.data.qpos[self.qpos_addresses[object_id] : self.qpos_addresses[object_id] + 3]
        delta = part_xyz - self.data.site_xpos[drop_site_id]
        return abs(delta[0]) <= 0.20 and abs(delta[1]) <= 0.16 and 0.04 <= part_xyz[2] <= 0.18

    def step(self) -> None:
        with self.state_lock:
            self._update_belt()
            for item in self.items:
                if item.part_name not in self.spawned and self.data.time >= item.spawn_time_s:
                    place_part(self.data, self.qpos_addresses[item.part_name], item.spawn_xyz)
                    self.spawned.add(item.part_name)
                    self._log("infeed", object_id=item.part_name, object_class=item.object_class.value)
            self._schedule()
            self._update_missions()
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            contacts = self._forbidden_contacts(self.data)
            if contacts and not self.paused:
                self._recover_last_safe_poses()
                mujoco.mj_forward(self.model, self.data)
                remaining_contacts = self._forbidden_contacts(self.data)
                if remaining_contacts:
                    self._abort_unsafe_missions(remaining_contacts)
                    mujoco.mj_forward(self.model, self.data)
                    if self._forbidden_contacts(self.data):
                        self.paused = True
                        self._log("safety_stop", reason="unrecoverable_forbidden_contact", contact=remaining_contacts[0])
                else:
                    self._log("safety_recover", reason="rollback_last_safe_pose", contact=contacts[0])

    def _recover_last_safe_poses(self) -> None:
        for arm, mission in self.missions.items():
            if mission.last_safe_qpos is None:
                continue
            _, _, _, opening = mission.keyframes[mission.keyframe_index]
            self.kinematics[arm].command_joint_pose(mission.last_safe_qpos, opening)

    def _abort_unsafe_missions(self, contacts: list[tuple[str, str]]) -> None:
        unsafe_arms = {arm for pair in contacts for arm in (self._arm_for_description(pair[0]), self._arm_for_description(pair[1])) if arm is not None}
        for arm in unsafe_arms:
            mission = self.missions.pop(arm, None)
            if mission is None:
                continue
            if mission.grasp_equality_id is not None:
                self.data.eq_active[mission.grasp_equality_id] = False
            self.kinematics[arm].command_joint_pose(self.kinematics[arm].home_qpos, 0.035)
            self.missed.add(mission.object_id)
            self.coordinator.mark_completed(mission.object_id)
            self._log("safety_recover", object_id=mission.object_id, arm=arm.value, reason="abort_and_retract", contact=contacts[0])

    def run_headless(self, duration_s: float) -> None:
        while self.data.time < duration_s and not self.paused:
            self.step()
        print(f"finished: placed={sorted(self.placed)} missed={sorted(self.missed)}")

    def run_viewer(self, duration_s: float, dashboard_url: str | None = None) -> None:

        def key_callback(keycode: int) -> None:
            if chr(keycode).lower() == "r":
                self.request_reset()

        if dashboard_url:
            webbrowser.open(dashboard_url)
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
            last_wall_time = time.perf_counter()
            accumulated_s = 0.0
            while viewer.is_running():
                if self.reset_if_requested():
                    with viewer.lock():
                        viewer.sync()
                    accumulated_s = 0.0
                now = time.perf_counter()
                accumulated_s += min(now - last_wall_time, 0.05) * self.parameters.simulation_speed
                last_wall_time = now
                while not self.paused and accumulated_s >= CONTROL_STEP_S and self.data.time < duration_s:
                    self.step()
                    accumulated_s -= CONTROL_STEP_S
                viewer.sync()
                time.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the centralized dual-Nova5 picking demonstration.")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR.parent / "models" / "nova5" / "nova5_sorting_line.xml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--headless", action="store_true", help="Run the scenario without opening the MuJoCo viewer.")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not start the local web dashboard.")
    args = parser.parse_args()
    demo = SortingDemo(args.model.resolve(), args.seed)
    if args.headless:
        demo.run_headless(args.duration)
    else:
        from demo_dashboard import start_dashboard

        dashboard = None if args.no_dashboard else start_dashboard(demo)
        try:
            demo.run_viewer(args.duration, dashboard.url if dashboard else None)
        finally:
            if dashboard:
                dashboard.stop()


if __name__ == "__main__":
    main()
