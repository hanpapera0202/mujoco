"""Single-seed dual-Nova5 sorting demonstration driven by the central coordinator.

All arm motion is driven through MuJoCo position control. A part is counted as
grasped only after a finger pad reports a physical MuJoCo contact.
"""

from __future__ import annotations

import argparse
import ctypes
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
MIN_PICK_HEIGHT_M = 0.135
PREGRASP_HEIGHT_M = 0.42
BIN_APPROACH_HEIGHT_M = 0.46
BIN_DROP_HEIGHT_M = 0.30
GRASP_XY_TOLERANCE_M = 0.055
GRIP_OPEN_M = 0.035
GRIP_CLOSED_M = 0.0
# The grasp-zone site is centred between the two finger pads.  Dynamic belt
# prediction therefore targets the measured part centre directly.
GRASP_ALIGNMENT_OFFSET_M = {
    ArmId.A: np.zeros(3),
    ArmId.B: np.zeros(3),
}


@dataclass
class DemoParameters:
    horizon_s: float = 30.0
    parallel_bonus: float = 2.0
    pick_speed_mps: float = 0.55
    # This starts from the measured v0.2 executor cycle, then is updated by
    # completed missions.  It must include approach, grasp, placement, return.
    fixed_cycle_s: float = 7.0
    urgency_weight: float = 3.0
    success_weight: float = 2.0
    travel_weight: float = 0.25
    belt_speed_mps: float = 0.12
    feed_interval_s: float = 7.5
    max_active_parts: float = 2.0
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
    failed: bool = False
    assigned_at_s: float = 0.0
    trajectory: list[tuple[float, np.ndarray, float]] | None = None
    stage_start_qpos: np.ndarray | None = None
    release_started: bool = False

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

    def command_joint_pose(self, qpos: np.ndarray, gripper_opening: float, *, initialize_fingers: bool = False) -> None:
        # Only reset may initialize state.  Runtime motion is generated by the
        # MuJoCo actuators so finger contact has a real surface velocity and
        # can physically carry a free body.
        if initialize_fingers:
            self.data.qpos[self.qpos_addresses] = qpos
            self.data.qvel[self.dof_addresses] = 0.0
            self.data.qpos[self.finger_qpos_addresses] = gripper_opening
        self.data.ctrl[self.position_actuator_ids] = qpos
        self.data.ctrl[self.finger_actuator_ids] = gripper_opening

    def set_pad_adhesion(self, force_n: float) -> None:
        self.model.geom_adhesion[list(self.finger_geom_ids)] = max(0.0, force_n)



def place_part(data: mujoco.MjData, qpos_address: int, xyz: tuple[float, float, float]) -> None:
    data.qpos[qpos_address : qpos_address + 7] = (*xyz, 1.0, 0.0, 0.0, 0.0)
    data.qvel[qpos_address : qpos_address + 6] = 0.0


def make_demo_items(seed: int, feed_interval_s: float) -> list[DemoItem]:
    """Ten deterministic moving parts controlled as shared work."""
    rng = random.Random(seed)
    classes = (ObjectClass.MIDDLE,) * 10
    items: list[DemoItem] = []
    for index, object_class in enumerate(classes, start=1):
        # All parts enter the shared central lane.  The coordinator chooses an
        # arm from predicted cost, deadline and balanced assignment history.
        center_x = 0.0
        # Physical feed is paced below the measured service rate. Concurrent
        # MIDDLE allocation remains covered independently by coordinator tests.
        spawn_time_s = 0.2 if index == 1 else 10.0 + (index - 2) * feed_interval_s
        spawn_y = 2.30 if index == 2 else 1.20
        spawn_z = 0.16 if index == 8 else 0.13
        items.append(DemoItem(f"part_{index:02d}", object_class, spawn_time_s, (center_x + rng.uniform(-0.018, 0.018), spawn_y, spawn_z), 30.0))
    return items


def interpolate(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    ratio = min(1.0, max(0.0, ratio))
    return first + (second - first) * ratio


def smoothstep(ratio: float) -> float:
    ratio = min(1.0, max(0.0, ratio))
    return ratio * ratio * (3.0 - 2.0 * ratio)


class SortingDemo:
    def __init__(self, model_path: Path, seed: int, parameters: DemoParameters | None = None) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = CONTROL_STEP_S
        # Industrial arms compensate their own link weight.  Apply the same
        # assumption here and add damping for stable position-servo tracking.
        for body_id in range(1, self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.startswith(("A_", "B_")):
                self.model.body_gravcomp[body_id] = 1.0
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self.algorithm_id = CSPR_ALGORITHM_ID
        self.parameters = parameters or DemoParameters()
        self.state_lock = threading.RLock()
        self.reset_requested = threading.Event()
        self.viewer_open_requested = threading.Event()
        self.viewer_active = False
        self.viewer_launching = False
        self.paused = False
        self.latest_decision: dict[str, object] = {"assignments": [], "rejected": {}}
        self.last_preflight: dict[str, object] = {"status": "pending", "reason": "waiting_for_task"}
        self.event_log: list[dict[str, object]] = []
        self._reset_state()

    def _reset_state(self) -> None:
        """Restore the exact seed scenario without replacing the viewer's MjData."""
        mujoco.mj_resetData(self.model, self.data)
        self.items = make_demo_items(self.seed, self.parameters.feed_interval_s)
        self.by_name = {item.part_name: item for item in self.items}
        self.qpos_addresses = {name: joint_qpos_address(self.model, name) for name in PART_NAMES}
        self.part_dof_addresses = {name: joint_dof_address(self.model, name) for name in PART_NAMES}
        self.warning_envelope_ids = {
            arm: tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{arm.value}_{region}_warning")
                for region in ("upper_arm", "forearm", "gripper")
            )
            for arm in ArmId
        }
        self.segment_qpos_addresses = [joint_qpos_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.segment_dof_addresses = [joint_dof_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.kinematics = {arm: ArmKinematics(self.model, self.data, arm) for arm in ArmId}
        for kin in self.kinematics.values():
            self.model.dof_damping[kin.dof_addresses] = np.array((14.0, 18.0, 16.0, 9.0, 8.0, 6.0))
        # Runtime model-array edits become active only after MuJoCo rebuilds
        # its derived constants.  Without this call, body_gravcomp remains
        # numerically configured but produces zero generalized force.
        mujoco.mj_setConst(self.model, self.data)
        for kin in self.kinematics.values():
            kin.command_joint_pose(kin.home_qpos, GRIP_OPEN_M, initialize_fingers=True)
            kin.set_pad_adhesion(0.0)
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
        self.next_feed_s = 0.2
        self.output_offsets = {"left_bin": 0, "right_bin": 0}
        self.latest_decision = {"assignments": [], "rejected": {}}
        self.last_preflight = {"status": "pending", "reason": "waiting_for_task"}
        self.event_log = []
        self.arm_outcomes = {arm: {"attempts": 0, "grasped": 0, "placed": 0, "cycle_s": self.parameters.fixed_cycle_s} for arm in ArmId}
        for index, name in enumerate(PART_NAMES):
            park_part(self.data, self.qpos_addresses[name], index)
        self._update_belt()
        mujoco.mj_forward(self.model, self.data)

    def request_reset(self) -> None:
        self.reset_requested.set()

    def request_viewer_open(self) -> str:
        """Open a closed viewer or bring the active MuJoCo window forward."""
        with self.state_lock:
            already_open = self.viewer_active or self.viewer_launching
            if not already_open:
                self.viewer_open_requested.set()
                return "requested"
        return "focused" if self._focus_mujoco_window() else "already_open"

    @staticmethod
    def _focus_mujoco_window() -> bool:
        if sys.platform != "win32":
            return False
        user32 = ctypes.windll.user32
        matches: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @callback_type
        def find_window(handle: int, _parameter: int) -> bool:
            length = user32.GetWindowTextLengthW(handle)
            if length <= 0 or not user32.IsWindowVisible(handle):
                return True
            title = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(handle, title, length + 1)
            if "mujoco" in title.value.lower():
                matches.append(handle)
                return False
            return True

        user32.EnumWindows(find_window, 0)
        if not matches:
            return False
        user32.ShowWindow(matches[0], 9)
        return bool(user32.SetForegroundWindow(matches[0]))

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
                "viewer": {
                    "active": self.viewer_active,
                    "launching": self.viewer_launching,
                    "requested": self.viewer_open_requested.is_set(),
                },
                "parameters": asdict(self.parameters),
                "counts": {"spawned": len(self.spawned), "placed": len(self.placed), "missed": len(self.missed)},
                "feedback": {
                    "active_parts": len(self.spawned - self.placed - self.missed),
                    "cycle_estimate_s": round(float(self.coordinator.fixed_cycle_s), 2),
                    "arms": {arm.value: {key: round(float(value), 2) for key, value in outcome.items()} for arm, outcome in self.arm_outcomes.items()},
                },
                "missions": missions,
                "deferred": [item.object_id for item in self.deferred_assignments.values()],
                "decision": self.latest_decision,
                "preflight": self.last_preflight,
                "safety": {
                    "warning_margin_m": 0.10,
                    "mode": "concurrent_mission_preflight",
                },
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

    def _warning_boxes_overlap(self, data: mujoco.MjData, first_id: int, second_id: int) -> bool:
        """Test two oriented warning boxes with MuJoCo's native distance query."""
        return mujoco.mj_geomDistance(self.model, data, first_id, second_id, 10.0, None) <= 0.0

    def _warning_envelope_overlaps(self, data: mujoco.MjData) -> list[tuple[str, str]]:
        """Detect A/B overlap using three expanded safety boxes per arm."""
        overlaps: list[tuple[str, str]] = []
        for first_id in self.warning_envelope_ids[ArmId.A]:
            for second_id in self.warning_envelope_ids[ArmId.B]:
                if self._warning_boxes_overlap(data, first_id, second_id):
                    overlaps.append((self._geom_description(first_id), self._geom_description(second_id)))
        return overlaps

    def _preflight_mission(self, mission: ArmMission) -> tuple[bool, str]:
        """Check both arms on one time axis before admitting a mission."""
        kin = self.kinematics[mission.arm]
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        for at_s, candidate_qpos, opening in mission.trajectory or []:
            trial.qpos[:] = self.data.qpos
            trial.qpos[kin.qpos_addresses] = candidate_qpos
            trial.qpos[kin.finger_qpos_addresses] = opening
            # Existing missions contribute their predicted joint state at the
            # same instant, instead of being treated as a static obstacle.
            for other_arm, other_mission in self.missions.items():
                if other_arm is mission.arm:
                    continue
                other_qpos, other_opening = self._mission_pose_at(other_mission, at_s)
                other_kin = self.kinematics[other_arm]
                trial.qpos[other_kin.qpos_addresses] = other_qpos
                trial.qpos[other_kin.finger_qpos_addresses] = other_opening
            mujoco.mj_forward(self.model, trial)
            contacts = self._forbidden_contacts(trial)
            if contacts:
                return False, f"time={at_s:.2f}: {contacts[0][0]} / {contacts[0][1]}"
            if any(other_arm is not mission.arm for other_arm in self.missions):
                envelope_overlaps = self._warning_envelope_overlaps(trial)
                if envelope_overlaps:
                    return False, f"time={at_s:.2f}: safety_envelope {envelope_overlaps[0][0]} / {envelope_overlaps[0][1]}"
        return True, "clear"

    def _mission_pose_at(self, mission: ArmMission, at_s: float) -> tuple[np.ndarray, float]:
        """Return the executor's reserved pose for an absolute simulation time."""
        trajectory = mission.trajectory or []
        if not trajectory:
            kin = self.kinematics[mission.arm]
            return self.data.qpos[kin.qpos_addresses].copy(), GRIP_OPEN_M
        for sample_time, qpos, opening in trajectory:
            if sample_time >= at_s:
                return qpos, opening
        return trajectory[-1][1], trajectory[-1][2]

    def _build_trajectory(self, arm: ArmId, keyframes: list[tuple[str, float, np.ndarray, float]]) -> list[tuple[float, np.ndarray, float]]:
        kin = self.kinematics[arm]
        previous = self.data.qpos[kin.qpos_addresses].copy()
        at_s = float(self.data.time)
        samples: list[tuple[float, np.ndarray, float]] = []
        for _, duration, target, opening in keyframes:
            for ratio in np.linspace(0.2, 1.0, 5):
                samples.append((at_s + duration * float(ratio), interpolate(previous, target, float(ratio)), opening))
            at_s += duration
            previous = target
        return samples

    def _pose_is_safe(self, arm: ArmId, qpos: np.ndarray, gripper_opening: float | None = None) -> bool:
        kin = self.kinematics[arm]
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        trial.qpos[kin.qpos_addresses] = qpos
        if gripper_opening is not None:
            trial.qpos[kin.finger_qpos_addresses] = gripper_opening
        mujoco.mj_forward(self.model, trial)
        if self._forbidden_contacts(trial):
            return False
        concurrent_mission = any(other_arm is not arm for other_arm in self.missions)
        return not concurrent_mission or not self._warning_envelope_overlaps(trial)

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
            grasp_probability = {
                arm: (outcome["grasped"] + 1.0) / (outcome["attempts"] + 2.0)
                for arm, outcome in self.arm_outcomes.items()
            }
            observations.append(ObjectObservation(name, item.object_class, tuple(xyz), remaining, grasp_probability))
        return observations

    def _object_is_claimed(self, object_id: str) -> bool:
        return any(mission.object_id == object_id for mission in self.missions.values()) or any(
            assignment.object_id == object_id for assignment in self.deferred_assignments.values()
        )

    def _predict_part_position(self, object_id: str, horizon_s: float) -> np.ndarray:
        """Predict interception from MuJoCo's current free-body velocity."""
        xyz = self.data.qpos[self.qpos_addresses[object_id] : self.qpos_addresses[object_id] + 3].copy()
        velocity = self.data.qvel[self.part_dof_addresses[object_id] : self.part_dof_addresses[object_id] + 3].copy()
        # The belt command is the deterministic longitudinal reference.  A
        # freshly spawned body's instantaneous contact velocity still ramps
        # up and otherwise introduces centimetres of intercept bias.
        velocity[1] = -self.parameters.belt_speed_mps
        return xyz + velocity * max(0.0, horizon_s)

    @staticmethod
    def _grasp_target(arm: ArmId, part_xyz: np.ndarray) -> np.ndarray:
        return part_xyz + GRASP_ALIGNMENT_OFFSET_M[arm]

    def _plan_mission(self, arm: ArmId, object_id: str, placement_zone: str) -> ArmMission:
        kin = self.kinematics[arm]
        time_to_close_s = 2.40 + 4.00 + 0.80
        intercept_close_s = self.data.time + time_to_close_s
        pick_xyz = self._grasp_target(arm, self._predict_part_position(object_id, time_to_close_s))
        pick_xyz[2] = max(MIN_PICK_HEIGHT_M, pick_xyz[2])
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        drop_site = "left_bin_drop" if placement_zone == "left_bin" else "right_bin_drop"
        drop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, drop_site)
        drop = self.data.site_xpos[drop_site_id].copy()
        column = self.output_offsets[placement_zone] % 2
        row = self.output_offsets[placement_zone] // 2
        self.output_offsets[placement_zone] += 1
        # Six separated tray slots.  The old 10 x 8 cm pattern let settling
        # parts overlap the next lowering path and knock a held part loose.
        drop[:2] += np.array(((column - 0.5) * 0.16, (row - 1.0) * 0.14))
        drop[2] = BIN_DROP_HEIGHT_M
        bin_approach = drop.copy()
        bin_approach[2] = BIN_APPROACH_HEIGHT_M

        q_home = kin.home_qpos
        q_pregrasp = kin.solve_position_ik(pregrasp, q_home)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp)
        q_bin_approach = kin.solve_position_ik(bin_approach, q_pregrasp)
        q_drop = kin.solve_position_ik(drop, q_bin_approach)
        keyframes = [
                ("approach", 2.40, q_pregrasp, GRIP_OPEN_M),
                ("descend", 4.00, q_pick, GRIP_OPEN_M),
                # Keep force applied long enough for a bilateral pinch to
                # settle before lifting.
                ("close", 0.80, q_pick, GRIP_CLOSED_M),
                ("lift", 3.00, q_pregrasp, GRIP_CLOSED_M),
                ("to_bin", 2.60, q_bin_approach, GRIP_CLOSED_M),
                ("lower", 1.20, q_drop, GRIP_CLOSED_M),
                ("open", 0.60, q_drop, GRIP_OPEN_M),
                ("settle", 0.80, q_drop, GRIP_OPEN_M),
                ("retreat", 1.20, q_bin_approach, GRIP_OPEN_M),
                ("home", 1.80, q_home, GRIP_OPEN_M),
            ]
        return ArmMission(
            arm,
            object_id,
            placement_zone,
            keyframes,
            intercept_close_s,
            keyframe_started_s=self.data.time,
            next_replan_s=self.data.time + 0.25,
            last_safe_qpos=self.data.qpos[kin.qpos_addresses].copy(),
            assigned_at_s=float(self.data.time),
            trajectory=self._build_trajectory(arm, keyframes),
            stage_start_qpos=self.data.qpos[kin.qpos_addresses].copy(),
        )

    def _refresh_intercept(self, mission: ArmMission) -> None:
        if mission.failed or self.data.time < mission.next_replan_s or mission.keyframe_index > 1:
            return
        kin = self.kinematics[mission.arm]
        pick_xyz = self._grasp_target(mission.arm, self._predict_part_position(mission.object_id, mission.intercept_close_s - self.data.time))
        pick_xyz[2] = max(MIN_PICK_HEIGHT_M, pick_xyz[2])
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        start = self.data.qpos[kin.qpos_addresses].copy()
        q_pregrasp = kin.solve_position_ik(pregrasp, start)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp)
        if not self._pose_is_safe(mission.arm, q_pregrasp) or not self._pose_is_safe(mission.arm, q_pick):
            mission.next_replan_s = self.data.time + 0.12
            return
        mission.keyframes[0] = ("approach", 2.40, q_pregrasp, GRIP_OPEN_M)
        mission.keyframes[1] = ("descend", 4.00, q_pick, GRIP_OPEN_M)
        mission.keyframes[2] = ("close", 0.80, q_pick, GRIP_CLOSED_M)
        mission.keyframes[3] = ("lift", 3.00, q_pregrasp, GRIP_CLOSED_M)
        mission.trajectory = self._build_trajectory(mission.arm, mission.keyframes)
        mission.next_replan_s = self.data.time + 0.25

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
                        "interval_s": [round(float(value), 3) for value in item.interval_s],
                        "score": round(float(item.score), 4),
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
        # Shared work is not serialized by a LEFT/RIGHT/MIDDLE label.  The
        # synchronized preflight and 10 cm envelopes decide whether both
        # equal-peer arms can proceed.
        return True

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
            if "safety_envelope" in reason:
                self.deferred_assignments[assignment.arm] = assignment
                self._log("reserve_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="path_collision", contact=reason)
            else:
                # A fixed obstacle will not become feasible by waiting. Release
                # the commitment so the next centralized cycle can try the
                # equal-peer arm with its updated assignment count.
                self.coordinator.mark_completed(assignment.object_id)
                self._log("screen_reject", object_id=assignment.object_id, arm=assignment.arm.value, reason="fixed_path_collision", contact=reason)
            return
        self.missions[assignment.arm] = mission
        self.arm_outcomes[assignment.arm]["attempts"] += 1
        self._log("assign", object_id=assignment.object_id, arm=assignment.arm.value, placement=assignment.placement_zone)

    def _update_missions(self) -> None:
        for arm, mission in list(self.missions.items()):
            # Failed simultaneous grips leave both wrists near the conveyor.
            # Release their recovery reservation in a deterministic order so
            # two retreat paths never begin from the same narrow corridor.
            if mission.failed and arm is ArmId.B and ArmId.A in self.missions and self.missions[ArmId.A].failed:
                continue
            kin = self.kinematics[arm]
            # The belt velocity is deterministic in this benchmark.  Keep the
            # admitted intercept fixed so online IK cannot invalidate the
            # collision-checked trajectory while both arms are moving.
            stage, duration, target, opening = mission.keyframes[mission.keyframe_index]
            elapsed = self.data.time - mission.keyframe_started_s
            current = self.data.qpos[kin.qpos_addresses].copy()
            if stage == "close":
                commanded_opening = GRIP_OPEN_M * (1.0 - smoothstep(elapsed / duration))
            elif stage == "open":
                commanded_opening = GRIP_OPEN_M * smoothstep(elapsed / duration)
            else:
                commanded_opening = opening
            if stage == "open" and not mission.release_started:
                touching_fingers = self._touching_fingers(arm, mission.object_id)
                if touching_fingers != kin.finger_geom_ids:
                    self._fail_grasp(arm, mission, "grip_lost_before_release", touching_fingers)
                    continue
                self._log(
                    "release",
                    object_id=mission.object_id,
                    placement=mission.placement_zone,
                    finger_count=len(touching_fingers),
                    part_xyz=np.round(self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3], 3).tolist(),
                    grasp_xyz=np.round(kin.grasp_position(), 3).tolist(),
                )
                mission.release_started = True
            if stage == "open":
                kin.set_pad_adhesion(0.0)
            require_clearance = stage != "close"
            if require_clearance and not self._pose_is_safe(arm, current, commanded_opening):
                if mission.last_safe_qpos is not None:
                    kin.command_joint_pose(mission.last_safe_qpos, commanded_opening)
                mission.keyframe_started_s += CONTROL_STEP_S
                continue
            stage_start = mission.stage_start_qpos if mission.stage_start_qpos is not None else current
            commanded_qpos = interpolate(stage_start, target, smoothstep(elapsed / max(duration, CONTROL_STEP_S)))
            if require_clearance and not self._pose_is_safe(arm, commanded_qpos, commanded_opening):
                mission.keyframe_started_s += CONTROL_STEP_S
                if self.data.time - mission.last_safety_hold_s >= 0.5:
                    mission.last_safety_hold_s = self.data.time
                    self._log("reserve_wait", object_id=mission.object_id, arm=arm.value, reason="step_collision_guard")
                continue
            kin.command_joint_pose(commanded_qpos, commanded_opening)
            mission.last_safe_qpos = current.copy()
            joint_error = float(np.max(np.abs(target - current)))
            joint_speed = float(np.max(np.abs(self.data.qvel[kin.dof_addresses])))
            stage_reached = elapsed >= duration and joint_error <= 0.08 and joint_speed <= 0.12
            if stage == "close" and stage_reached and not mission.grasped:
                if not self._confirm_grasp(arm, mission):
                    continue
                kin.set_pad_adhesion(20.0)
            if not stage_reached:
                continue
            mission.keyframe_index += 1
            mission.keyframe_started_s = self.data.time
            # Preserve the previous actuator target across a stage boundary.
            # Starting the next interpolation from the lagging measured pose
            # would briefly unload the servo and make the wrist dip.
            mission.stage_start_qpos = target.copy()
            if mission.done:
                kin.set_pad_adhesion(0.0)
                self.missions.pop(arm)
                if not mission.failed and self._part_is_in_target_bin(mission.object_id, mission.placement_zone):
                    self.placed.add(mission.object_id)
                    self.arm_outcomes[arm]["placed"] += 1
                    self._log("place", object_id=mission.object_id, placement=mission.placement_zone)
                elif not mission.failed:
                    self.missed.add(mission.object_id)
                    self._log("missed", object_id=mission.object_id, reason="placement_not_verified", part_xyz=np.round(self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3], 3).tolist())
                self.coordinator.mark_completed(mission.object_id)
                measured_cycle_s = self.data.time - mission.assigned_at_s
                arm_feedback = self.arm_outcomes[arm]
                arm_feedback["cycle_s"] = 0.8 * arm_feedback["cycle_s"] + 0.2 * measured_cycle_s
                self.coordinator.fixed_cycle_s = float(np.mean([item["cycle_s"] for item in self.arm_outcomes.values()]))
                self._log("cycle_feedback", arm=arm.value, cycle_s=round(measured_cycle_s, 2), estimate_s=round(self.coordinator.fixed_cycle_s, 2))

    def _confirm_grasp(self, arm: ArmId, mission: ArmMission) -> bool:
        """Accept a grasp only after a physical finger-pad contact is reported."""
        kin = self.kinematics[arm]
        touching_fingers = self._touching_fingers(arm, mission.object_id)
        if touching_fingers == kin.finger_geom_ids:
            mission.grasped = True
            self.arm_outcomes[arm]["grasped"] += 1
            self._log("grasp", object_id=mission.object_id, arm=arm.value, contact="bilateral_finger_physical", finger_count=len(touching_fingers), grasp_constraint="none")
            return True
        self._fail_grasp(arm, mission, "no_bilateral_finger_contact", touching_fingers)
        return False

    def _touching_fingers(self, arm: ArmId, object_id: str) -> set[int]:
        kin = self.kinematics[arm]
        part_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        part_geoms = set(range(self.model.body_geomadr[part_body], self.model.body_geomadr[part_body] + self.model.body_geomnum[part_body]))
        touching_fingers: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if pair.intersection(part_geoms):
                touching_fingers.update(kin.finger_geom_ids.intersection(pair))
        return touching_fingers

    def _fail_grasp(self, arm: ArmId, mission: ArmMission, reason: str, touching_fingers: set[int]) -> None:
        kin = self.kinematics[arm]
        self.missed.add(mission.object_id)
        self.coordinator.mark_completed(mission.object_id)
        self._log(
            "missed",
            object_id=mission.object_id,
            reason=reason,
            finger_count=len(touching_fingers),
            grasp_error_m=round(float(np.linalg.norm(self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3] - kin.grasp_position())), 3),
            grasp_error_xyz=np.round(self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3] - kin.grasp_position(), 3).tolist(),
        )
        mission.failed = True
        kin.set_pad_adhesion(0.0)
        mission.keyframes = [("recover", 0.6, self.data.qpos[kin.qpos_addresses].copy(), GRIP_OPEN_M), ("home", 1.0, kin.home_qpos, GRIP_OPEN_M)]
        mission.keyframe_index = 0
        mission.keyframe_started_s = self.data.time
        mission.stage_start_qpos = self.data.qpos[kin.qpos_addresses].copy()

    def _part_is_in_target_bin(self, object_id: str, placement_zone: str) -> bool:
        drop_site = "left_bin_drop" if placement_zone == "left_bin" else "right_bin_drop"
        floor_geom = "left_tray_floor" if placement_zone == "left_bin" else "right_tray_floor"
        drop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, drop_site)
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom)
        part_xyz = self.data.qpos[self.qpos_addresses[object_id] : self.qpos_addresses[object_id] + 3]
        delta = part_xyz - self.data.site_xpos[drop_site_id]
        half_x, half_y = self.model.geom_size[floor_geom_id, :2] - 0.02
        return abs(delta[0]) <= half_x and abs(delta[1]) <= half_y and 0.04 <= part_xyz[2] <= 0.18

    def step(self) -> None:
        with self.state_lock:
            self._update_belt()
            active_parts = len(self.spawned - self.placed - self.missed)
            for item in self.items:
                if item.part_name not in self.spawned and self.data.time >= item.spawn_time_s:
                    # Preserve the initial A/B benchmark pair.  Afterwards,
                    # release at most one waiting part per configured feed
                    # interval, even when capacity becomes available late.
                    initial_pair = len(self.spawned) < 2
                    if not initial_pair and (
                        active_parts >= int(self.parameters.max_active_parts)
                        or self.data.time < self.next_feed_s
                        or not self._has_available_handler(item)
                    ):
                        break
                    place_part(self.data, self.qpos_addresses[item.part_name], item.spawn_xyz)
                    self.spawned.add(item.part_name)
                    active_parts += 1
                    self._log("infeed", object_id=item.part_name, object_class=item.object_class.value)
                    if len(self.spawned) == 2:
                        self.next_feed_s = self.data.time + self.parameters.feed_interval_s
                    elif not initial_pair:
                        self.next_feed_s = self.data.time + self.parameters.feed_interval_s
                        break
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

    def _has_available_handler(self, item: DemoItem) -> bool:
        unavailable = set(self.missions) | set(self.deferred_assignments)
        if item.object_class is ObjectClass.LEFT:
            return ArmId.A not in unavailable
        if item.object_class is ObjectClass.RIGHT:
            return ArmId.B not in unavailable
        return any(arm not in unavailable for arm in ArmId)

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
            self.kinematics[arm].command_joint_pose(self.kinematics[arm].home_qpos, GRIP_OPEN_M)
            self.kinematics[arm].set_pad_adhesion(0.0)
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
        self.viewer_open_requested.set()
        while True:
            self.viewer_open_requested.wait()
            self.viewer_open_requested.clear()
            with self.state_lock:
                self.viewer_launching = True
            try:
                with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
                    with self.state_lock:
                        self.viewer_launching = False
                        self.viewer_active = True
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
            finally:
                with self.state_lock:
                    self.viewer_active = False
                    self.viewer_launching = False


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
