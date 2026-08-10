"""Single-seed dual-Nova5 sorting demonstration driven by the central coordinator.

All arm motion is driven through MuJoCo position control. A part is counted as
grasped only after a finger pad reports a physical MuJoCo contact.
"""

from __future__ import annotations

import argparse
import copy
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
from bayesian_joint_planner import BayesianJointGame, JointStrategyEvidence
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
TRACKING_IK_PERIOD_S = 0.08
TRACKING_LEAD_S = 0.10
TRACKING_IK_MAX_ITERATIONS = 12
TRACKING_IK_MIN_TARGET_DELTA_M = 0.008
SAFETY_CHECK_PERIOD_S = 0.02
MAX_VIEWER_SUBSTEPS = 8
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
    warning_margin_m: float = 0.10
    feed_batch_size: float = 2.0
    feed_x_min_m: float = -0.15
    feed_x_max_m: float = 0.15
    feed_y_min_m: float = 1.15
    feed_y_max_m: float = 1.25


CSPR_ALGORITHM_ID = "bc_jsp"
CSPR_ALGORITHM_NAME = "BC-JSP - Bayesian Centralized Joint Strategy Planner"
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
    route_variant: str = "direct"
    joint_strategy: tuple[str, str] | None = None
    last_pick_xyz: np.ndarray | None = None
    tracking_updates: int = 0
    next_safety_check_s: float = 0.0
    preparation_only: bool = False
    handoff_assignment: object | None = None
    handoff_lead_assignment: object | None = None
    lead_started: bool = False
    handoff_target_qpos: np.ndarray | None = None
    preparation_complete: bool = False

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

    def solve_position_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray, *, max_iterations: int = 360) -> np.ndarray:
        """Damped 6D IK with a fixed, conveyor-facing parallel-gripper pose."""
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = start_qpos
        for _ in range(max_iterations):
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


def make_demo_items(
    seed: int,
    feed_interval_s: float,
    feed_batch_size: int = 2,
    feed_x_range_m: tuple[float, float] = (-0.15, 0.15),
    feed_y_range_m: tuple[float, float] = (1.15, 1.25),
) -> list[DemoItem]:
    """Ten deterministic moving parts controlled as shared work."""
    rng = random.Random(seed)
    classes = (ObjectClass.MIDDLE,) * 10
    items: list[DemoItem] = []
    for index, object_class in enumerate(classes, start=1):
        # All parts enter the shared central lane.  The coordinator chooses an
        # arm from predicted cost, deadline and balanced assignment history.
        x_min, x_max = feed_x_range_m
        y_min, y_max = feed_y_range_m
        middle_x = 0.5 * (x_min + x_max)
        batch_size = max(1, int(feed_batch_size))
        slot = (index - 1) % batch_size
        if batch_size == 1:
            center_x = rng.uniform(x_min, x_max)
        elif slot % 2 == 0:
            center_x = rng.uniform(x_min, middle_x - 0.02)
        else:
            center_x = rng.uniform(middle_x + 0.02, x_max)
        # Physical feed is paced below the measured service rate. Concurrent
        # MIDDLE allocation remains covered independently by coordinator tests.
        spawn_time_s = 0.2 + ((index - 1) // batch_size) * feed_interval_s
        # Every part is created on the physical head segment. With short GUI
        # feed intervals this lets the rolling horizon observe two moving
        # objects together instead of parking part 02 outside both workspaces.
        spawn_y = rng.uniform(y_min, y_max)
        spawn_z = 0.16 if index == 8 else 0.13
        items.append(DemoItem(f"part_{index:02d}", object_class, spawn_time_s, (center_x, spawn_y, spawn_z), 30.0))
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
        self.latest_joint_plan: dict[str, object] = {"status": "pending", "evaluated": 0}
        self.event_log: list[dict[str, object]] = []
        self._reset_state()

    def _reset_state(self) -> None:
        """Restore the exact seed scenario without replacing the viewer's MjData."""
        mujoco.mj_resetData(self.model, self.data)
        self.items = make_demo_items(
            self.seed,
            self.parameters.feed_interval_s,
            int(self.parameters.feed_batch_size),
            (self.parameters.feed_x_min_m, self.parameters.feed_x_max_m),
            (self.parameters.feed_y_min_m, self.parameters.feed_y_max_m),
        )
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
        self._apply_warning_margin()
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
        self.handoff_leads = {}
        self.spawned: set[str] = set()
        self.placed: set[str] = set()
        self.missed: set[str] = set()
        self.last_schedule_s = -SCHEDULER_PERIOD_S
        self.next_feed_s = 0.2
        self.output_offsets = {"left_bin": 0, "right_bin": 0}
        self.latest_decision = {"assignments": [], "rejected": {}}
        self.last_preflight = {"status": "pending", "reason": "waiting_for_task"}
        self.latest_joint_plan = {"status": "pending", "evaluated": 0}
        # Preserve learned route beliefs across GUI replay/settings changes;
        # constructing a new SortingDemo still starts from the documented prior.
        if not hasattr(self, "bayesian_game"):
            self.bayesian_game = BayesianJointGame()
        self.joint_result_buffer: dict[tuple[str, str], list[bool]] = {}
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
                if algorithm_id not in (CSPR_ALGORITHM_ID, "cspr"):
                    raise ValueError("This algorithm is reserved for a future implementation")
                self.algorithm_id = CSPR_ALGORITHM_ID
            if "seed" in values:
                self.seed = int(values["seed"])
            candidate_parameters = asdict(self.parameters)
            for field_name in candidate_parameters:
                if field_name in values:
                    value = float(values[field_name])
                    if field_name not in ("feed_x_min_m", "feed_x_max_m") and value <= 0.0:
                        raise ValueError(f"{field_name} must be positive")
                    if field_name == "warning_margin_m" and not 0.02 <= value <= 0.30:
                        raise ValueError("warning_margin_m must be between 0.02 and 0.30")
                    if field_name == "feed_batch_size" and (value > 10 or not value.is_integer()):
                        raise ValueError("feed_batch_size must be an integer between 1 and 10")
                    candidate_parameters[field_name] = value
            if not -0.18 <= candidate_parameters["feed_x_min_m"] < candidate_parameters["feed_x_max_m"] <= 0.18:
                raise ValueError("feed X range must stay within -0.18..0.18 m")
            if candidate_parameters["feed_x_max_m"] - candidate_parameters["feed_x_min_m"] < 0.08:
                raise ValueError("feed X range must be at least 0.08 m wide")
            if not 0.90 <= candidate_parameters["feed_y_min_m"] < candidate_parameters["feed_y_max_m"] <= 1.40:
                raise ValueError("feed Y range must stay within 0.90..1.40 m")
            for field_name, value in candidate_parameters.items():
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
                    "route": mission.route_variant,
                    "tracking_updates": mission.tracking_updates,
                    "preparation_only": mission.preparation_only,
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
                "performance": {
                    "control_hz": round(1.0 / CONTROL_STEP_S, 1),
                    "tracking_ik_hz": round(1.0 / TRACKING_IK_PERIOD_S, 1),
                    "safety_prediction_hz": round(1.0 / SAFETY_CHECK_PERIOD_S, 1),
                    "max_viewer_substeps": MAX_VIEWER_SUBSTEPS,
                },
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
                "joint_plan": self.latest_joint_plan,
                "safety": {
                    "warning_margin_m": round(float(self.parameters.warning_margin_m), 3),
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

    def _apply_warning_margin(self) -> None:
        """Resize visual warning boxes without changing physical collision geometry."""
        margin = self.parameters.warning_margin_m
        for arm in ArmId:
            for region in ("upper_arm", "forearm", "gripper"):
                collision_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{arm.value}_{region}_collision")
                warning_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{arm.value}_{region}_warning")
                self.model.geom_size[warning_id] = self.model.geom_size[collision_id] + margin

    def _warning_envelope_overlaps(self, data: mujoco.MjData) -> list[tuple[str, str]]:
        """Detect A/B overlap using three expanded safety boxes per arm."""
        overlaps: list[tuple[str, str]] = []
        for first_id in self.warning_envelope_ids[ArmId.A]:
            for second_id in self.warning_envelope_ids[ArmId.B]:
                if self._warning_boxes_overlap(data, first_id, second_id):
                    overlaps.append((self._geom_description(first_id), self._geom_description(second_id)))
        return overlaps

    def _preflight_mission(self, mission: ArmMission, enforce_warning: bool = True) -> tuple[bool, str]:
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
            if enforce_warning and any(other_arm is not mission.arm for other_arm in self.missions):
                envelope_overlaps = self._warning_envelope_overlaps(trial)
                if envelope_overlaps:
                    return False, f"time={at_s:.2f}: safety_envelope {envelope_overlaps[0][0]} / {envelope_overlaps[0][1]}"
        return True, "clear"

    def _preflight_joint_pair(self, first: ArmMission, second: ArmMission) -> tuple[bool, str]:
        """Validate both predicted trajectories on one dense simulation clock."""
        trial = mujoco.MjData(self.model)
        start_s = min(first.trajectory[0][0], second.trajectory[0][0])
        end_s = max(first.trajectory[-1][0], second.trajectory[-1][0])
        for at_s in np.arange(start_s, end_s + 0.001, 0.10):
            trial.qpos[:] = self.data.qpos
            for mission in (first, second):
                qpos, opening = self._mission_pose_at(mission, float(at_s))
                kin = self.kinematics[mission.arm]
                trial.qpos[kin.qpos_addresses] = qpos
                trial.qpos[kin.finger_qpos_addresses] = opening
            mujoco.mj_forward(self.model, trial)
            contacts = self._forbidden_contacts(trial)
            if contacts:
                return False, f"time={at_s:.2f}: {contacts[0][0]} / {contacts[0][1]}"
            overlaps = self._warning_envelope_overlaps(trial)
            if overlaps:
                return False, f"time={at_s:.2f}: safety_envelope {overlaps[0][0]} / {overlaps[0][1]}"
        return True, "clear"

    def _mission_pose_at(self, mission: ArmMission, at_s: float) -> tuple[np.ndarray, float]:
        """Return the executor's reserved pose for an absolute simulation time."""
        trajectory = mission.trajectory or []
        if not trajectory:
            kin = self.kinematics[mission.arm]
            return self.data.qpos[kin.qpos_addresses].copy(), GRIP_OPEN_M
        if at_s <= trajectory[0][0]:
            return trajectory[0][1], trajectory[0][2]
        for previous, following in zip(trajectory, trajectory[1:]):
            if following[0] >= at_s:
                ratio = (at_s - previous[0]) / max(following[0] - previous[0], CONTROL_STEP_S)
                return interpolate(previous[1], following[1], ratio), float(previous[2] + (following[2] - previous[2]) * ratio)
        return trajectory[-1][1], trajectory[-1][2]

    def _build_trajectory(self, arm: ArmId, keyframes: list[tuple[str, float, np.ndarray, float]]) -> list[tuple[float, np.ndarray, float]]:
        kin = self.kinematics[arm]
        previous = self.data.qpos[kin.qpos_addresses].copy()
        at_s = float(self.data.time)
        opening = float(np.mean(self.data.qpos[kin.finger_qpos_addresses]))
        samples: list[tuple[float, np.ndarray, float]] = [(at_s, previous.copy(), opening)]
        for _, duration, target, opening in keyframes:
            for ratio in np.linspace(0.2, 1.0, 5):
                samples.append((at_s + duration * float(ratio), interpolate(previous, target, float(ratio)), opening))
            at_s += duration
            previous = target
        return samples

    def _route_candidates(self, mission: ArmMission) -> list[ArmMission]:
        """Generate three equal-peer route actions for the Bayesian game."""
        variants: list[ArmMission] = []
        escape_qpos = {
            ArmId.A: np.array((-2.1728, -0.2902, -1.8000, 2.5770, 4.0202, -6.1516)),
            ArmId.B: np.array((3.3961, 1.7425, -1.3483, 0.0000, 1.9187, 0.0000)),
        }[mission.arm]
        for name, duration_scale, escape_duration in (
            ("direct", 1.00, 0.0),
            ("balanced", 1.12, 2.4),
            ("outer", 1.28, 3.2),
        ):
            candidate = copy.deepcopy(mission)
            candidate.route_variant = name
            task_frames = [(stage, duration * duration_scale, qpos.copy(), opening) for stage, duration, qpos, opening in mission.keyframes]
            if escape_duration:
                task_frames.insert(0, (f"{name}_escape", escape_duration, escape_qpos.copy(), GRIP_OPEN_M))
            candidate.keyframes = task_frames
            candidate.keyframe_index = 0
            candidate.keyframe_started_s = float(self.data.time)
            candidate.trajectory = self._build_trajectory(candidate.arm, candidate.keyframes)
            candidate.stage_start_qpos = self.data.qpos[self.kinematics[candidate.arm].qpos_addresses].copy()
            variants.append(candidate)
        return variants

    def _joint_evidence(self, first: ArmMission, second: ArmMission) -> JointStrategyEvidence:
        safe, reason = self._preflight_joint_pair(first, second)
        start_s = min(first.trajectory[0][0], second.trajectory[0][0])
        end_s = max(first.trajectory[-1][0], second.trajectory[-1][0])
        times = np.arange(start_s, end_s + 0.001, 0.10)
        previous: dict[ArmId, np.ndarray] = {}
        moving_together = 0
        path_length = 0.0
        for at_s in times:
            moving: dict[ArmId, bool] = {}
            for mission in (first, second):
                qpos, _ = self._mission_pose_at(mission, float(at_s))
                old = previous.get(mission.arm, qpos)
                delta = float(np.linalg.norm(qpos - old))
                path_length += delta
                moving[mission.arm] = delta > 1e-4
                previous[mission.arm] = qpos
            moving_together += int(all(moving.values()))
        simultaneous_ratio = moving_together / max(1, len(times) - 1)
        grasp = {
            arm: (self.arm_outcomes[arm]["grasped"] + 1.0) / (self.arm_outcomes[arm]["attempts"] + 2.0)
            for arm in ArmId
        }
        by_arm = {first.arm: first, second.arm: second}
        return JointStrategyEvidence(
            route_a=by_arm[ArmId.A].route_variant,
            route_b=by_arm[ArmId.B].route_variant,
            collision_free=safe,
            makespan_s=end_s - start_s,
            simultaneous_ratio=simultaneous_ratio,
            path_length_rad=path_length,
            grasp_probability_a=grasp[ArmId.A],
            grasp_probability_b=grasp[ArmId.B],
            rejection_reason="" if safe else reason,
        )

    def _pose_is_safe(self, arm: ArmId, qpos: np.ndarray, gripper_opening: float | None = None) -> bool:
        """Check immediate physical safety; warning envelopes are admission constraints.

        Reapplying the 10 cm planning envelope independently to each servo step
        can freeze both arms in mutually blocking poses even after their shared
        timeline passed preflight. Physical contacts remain checked here and
        globally after every MuJoCo step.
        """
        kin = self.kinematics[arm]
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        trial.qpos[kin.qpos_addresses] = qpos
        if gripper_opening is not None:
            trial.qpos[kin.finger_qpos_addresses] = gripper_opening
        mujoco.mj_forward(self.model, trial)
        if self._forbidden_contacts(trial):
            return False
        return True

    def _update_belt(self) -> None:
        travelled = (self.parameters.belt_speed_mps * self.data.time) % CONVEYOR_LOOP_LENGTH_M
        phase_pitch = CONVEYOR_LOOP_LENGTH_M / SEGMENT_COUNT
        for index, (qpos_address, dof_address) in enumerate(zip(self.segment_qpos_addresses, self.segment_dof_addresses)):
            self.data.qpos[qpos_address] = UPSTREAM_CENTER_Y_M - ((index * phase_pitch + travelled) % CONVEYOR_LOOP_LENGTH_M)
            self.data.qvel[dof_address] = -self.parameters.belt_speed_mps

    def _tool_arm_states(self) -> tuple[ArmState, ArmState]:
        return tuple(
            # The screening radius includes the vertical approach posture; the
            # subsequent IK and MuJoCo collision checks remain authoritative.
            ArmState(arm, tuple(self.kinematics[arm].tool_position()), 1.80, 999.0 if arm in self.missions or arm in self.deferred_assignments else 0.0)
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
        prepare_s, track_s, descend_s, close_s = 1.40, 1.60, 3.40, 0.80
        time_to_close_s = prepare_s + track_s + descend_s + close_s
        intercept_close_s = self.data.time + time_to_close_s
        prepare_xyz = self._grasp_target(arm, self._predict_part_position(object_id, prepare_s))
        prepare_xyz[2] = 0.56
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
        q_prepare = kin.solve_position_ik(prepare_xyz, q_home)
        q_pregrasp = kin.solve_position_ik(pregrasp, q_prepare)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp)
        q_bin_approach = kin.solve_position_ik(bin_approach, q_pregrasp)
        q_drop = kin.solve_position_ik(drop, q_bin_approach)
        keyframes = [
                ("prepare", prepare_s, q_prepare, GRIP_OPEN_M),
                ("track", track_s, q_pregrasp, GRIP_OPEN_M),
                ("descend", descend_s, q_pick, GRIP_OPEN_M),
                # Keep force applied long enough for a bilateral pinch to
                # settle before lifting.
                ("close", close_s, q_pick, GRIP_CLOSED_M),
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
            last_pick_xyz=pick_xyz.copy(),
        )

    def _refresh_intercept(self, mission: ArmMission) -> None:
        close_index = next((index for index, frame in enumerate(mission.keyframes) if frame[0] == "close"), -1)
        if mission.failed or self.data.time < mission.next_replan_s or close_index < 0 or mission.keyframe_index > close_index:
            return
        kin = self.kinematics[mission.arm]
        current_stage = mission.keyframes[mission.keyframe_index][0]
        if current_stage not in ("prepare", "track", "descend", "close"):
            return

        # Follow the observed conveyor body instead of repeatedly aiming at one
        # old intercept. A small lead compensates actuator and IK latency.
        pick_xyz = self._grasp_target(
            mission.arm,
            self._predict_part_position(mission.object_id, TRACKING_LEAD_S),
        )
        pick_xyz[2] = max(MIN_PICK_HEIGHT_M, pick_xyz[2])
        if mission.last_pick_xyz is not None and np.linalg.norm(pick_xyz - mission.last_pick_xyz) < TRACKING_IK_MIN_TARGET_DELTA_M:
            mission.next_replan_s = self.data.time + TRACKING_IK_PERIOD_S
            return
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        start = self.data.qpos[kin.qpos_addresses].copy()
        q_pregrasp = kin.solve_position_ik(pregrasp, start, max_iterations=TRACKING_IK_MAX_ITERATIONS)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp, max_iterations=TRACKING_IK_MAX_ITERATIONS)
        if not self._pose_is_safe(mission.arm, q_pregrasp) or not self._pose_is_safe(mission.arm, q_pick):
            mission.next_replan_s = self.data.time + TRACKING_IK_PERIOD_S
            return
        for index, (stage, duration, target, opening) in enumerate(mission.keyframes):
            if stage == "prepare" and index == mission.keyframe_index:
                prepare_xyz = pick_xyz.copy()
                prepare_xyz[2] = 0.56
                q_prepare = kin.solve_position_ik(prepare_xyz, start, max_iterations=TRACKING_IK_MAX_ITERATIONS)
                mission.keyframes[index] = (stage, duration, q_prepare, opening)
            elif stage == "track":
                mission.keyframes[index] = (stage, duration, q_pregrasp, opening)
            elif stage in ("descend", "close"):
                mission.keyframes[index] = (stage, duration, q_pick, opening)
            elif stage == "lift":
                mission.keyframes[index] = (stage, duration, q_pregrasp, opening)
        mission.trajectory = self._build_trajectory(mission.arm, mission.keyframes[mission.keyframe_index :])
        mission.last_pick_xyz = pick_xyz.copy()
        mission.tracking_updates += 1
        mission.next_replan_s = self.data.time + TRACKING_IK_PERIOD_S

    def _schedule(self) -> None:
        if self.data.time - self.last_schedule_s < SCHEDULER_PERIOD_S:
            return
        self.last_schedule_s = self.data.time
        self._start_safe_deferred_assignments()
        self._activate_prepared_handoffs()
        if any(
            mission.preparation_only
            and mission.handoff_lead_assignment is not None
            and not mission.lead_started
            for mission in self.missions.values()
        ):
            self.latest_decision = {
                "assignments": [],
                "rejected": {},
                "status": "preparing_handoff_lead",
            }
            return
        # A committed moving part must not be starved by repeatedly assigning
        # newer arrivals to the peer arm. Retry the reservation until it starts
        # or exits before creating another commitment.
        if self.deferred_assignments:
            self.latest_decision = {
                "assignments": [],
                "rejected": {},
                "status": "prioritizing_deferred_dynamic_object",
            }
            return
        observations = self._available_observations()
        # Avoid a Braess-like local greedy commitment: when the feed is faster
        # than one service cycle, retain a lone object briefly so the imminent
        # peer can enter the 3 x 3 joint-strategy game with it.
        future_arrivals = [item.spawn_time_s for item in self.items if item.part_name not in self.spawned]
        should_batch = (
            not self.missions
            and not self.deferred_assignments
            and len(observations) == 1
            and self.parameters.feed_interval_s < self.coordinator.fixed_cycle_s
            and future_arrivals
            and min(future_arrivals) - self.data.time <= self.parameters.feed_interval_s + SCHEDULER_PERIOD_S
        )
        if should_batch:
            self.latest_decision = {"assignments": [], "rejected": {}, "status": "batching_for_joint_game"}
            return
        decision = self.coordinator.decide(self.data.time, observations, self._tool_arm_states())
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
        available = [item for item in decision.assignments if not self._object_is_claimed(item.object_id)]
        if len(available) == 2 and all(item.arm not in self.missions for item in available):
            self._start_joint_assignments(available)
            return
        for assignment in available:
            if self._object_is_claimed(assignment.object_id):
                continue
            if self._may_enter_assignment(assignment):
                self._start_assignment(assignment)
            else:
                self.deferred_assignments[assignment.arm] = assignment
                self._log("reserve_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="central_corridor")

    def _start_joint_assignments(self, assignments) -> None:
        """Evaluate the 3 x 3 Bayesian joint-strategy game and start both arms."""
        bases = [self._plan_mission(item.arm, item.object_id, item.placement_zone) for item in assignments]
        candidates = {mission.arm: self._route_candidates(mission) for mission in bases}
        evidence: list[JointStrategyEvidence] = []
        mission_pairs: dict[tuple[str, str], tuple[ArmMission, ArmMission]] = {}
        for first in candidates[ArmId.A]:
            for second in candidates[ArmId.B]:
                item = self._joint_evidence(first, second)
                evidence.append(item)
                mission_pairs[(item.route_a, item.route_b)] = (first, second)
        selected = self.bayesian_game.choose(evidence)
        rejected = sum(not item.collision_free for item in evidence)
        if selected is None:
            self.latest_joint_plan = {
                "status": "no_safe_joint_strategy",
                "evaluated": len(evidence),
                "rejected": rejected,
                "reason": evidence[0].rejection_reason if evidence else "no_candidates",
            }
            # All nine joint actions are unsafe. Admit one screened mission so
            # the line still makes progress and defer its equal peer.
            first, second = assignments
            self.deferred_assignments[second.arm] = second
            # Bootstrap through the peer's safe standby pose. The lead arm
            # stays at its measured pose until the shared corridor is clear.
            self.handoff_leads[second.arm] = first
            self._log("joint_defer", evaluated=len(evidence), reason="all_joint_routes_unsafe")
            return

        chosen = selected.evidence
        pair = mission_pairs[(chosen.route_a, chosen.route_b)]
        strategy = (chosen.route_a, chosen.route_b)
        for mission in pair:
            mission.joint_strategy = strategy
            self.missions[mission.arm] = mission
            self.arm_outcomes[mission.arm]["attempts"] += 1
            self._log(
                "assign",
                object_id=mission.object_id,
                arm=mission.arm.value,
                placement=mission.placement_zone,
                route=mission.route_variant,
            )
        self.last_preflight = {"status": "clear", "reason": "joint_trajectory_clear"}
        self.latest_joint_plan = {
            "status": "selected",
            "evaluated": len(evidence),
            "rejected": rejected,
            "route_a": chosen.route_a,
            "route_b": chosen.route_b,
            "completion_probability": round(selected.completion_probability, 4),
            "collision_probability": round(selected.collision_probability, 4),
            "expected_utility": round(selected.expected_utility, 3),
            "makespan_s": round(chosen.makespan_s, 3),
            "simultaneous_ratio": round(chosen.simultaneous_ratio, 4),
            "path_length_rad": round(chosen.path_length_rad, 3),
        }
        self._log("joint_plan", **self.latest_joint_plan)

    def _may_enter_assignment(self, assignment) -> bool:
        # Shared work is not serialized by a LEFT/RIGHT/MIDDLE label.  The
        # synchronized preflight and 10 cm envelopes decide whether both
        # equal-peer arms can proceed.
        return True

    def _start_safe_deferred_assignments(self) -> None:
        for arm, assignment in list(self.deferred_assignments.items()):
            if any(
                mission.preparation_only
                and mission.handoff_lead_assignment is not None
                and mission.handoff_lead_assignment.object_id == assignment.object_id
                and not mission.lead_started
                for mission in self.missions.values()
            ):
                continue
            part_y = float(self.data.qpos[self.qpos_addresses[assignment.object_id] + 1])
            if part_y < TAIL_EXIT_Y_M:
                self.deferred_assignments.pop(arm)
                self.missed.add(assignment.object_id)
                self.coordinator.mark_completed(assignment.object_id)
                self._log("missed", object_id=assignment.object_id, reason="tail_exit_while_deferred")
                continue
            if arm not in self.missions and arm in self.handoff_leads:
                if self._start_handoff_preparation(assignment):
                    self.deferred_assignments.pop(arm)
            elif arm not in self.missions and any(other_arm is not arm for other_arm in self.missions):
                if self._start_handoff_preparation(assignment):
                    self.deferred_assignments.pop(arm)
            elif arm not in self.missions and self._may_enter_assignment(assignment):
                self.deferred_assignments.pop(arm)
                self._start_assignment(assignment)

    def _activate_prepared_handoffs(self) -> None:
        """Convert a safe standby motion into a full mission after the peer clears."""
        for arm, mission in list(self.missions.items()):
            if not mission.preparation_only or not mission.preparation_complete:
                continue
            if mission.handoff_lead_assignment is not None and not mission.lead_started:
                if self._start_assignment(mission.handoff_lead_assignment):
                    mission.lead_started = True
                    self._start_handoff_creep(mission)
                    self._log("handoff_lead_start", object_id=mission.handoff_lead_assignment.object_id, arm=arm.value)
                else:
                    continue
        if len(self.missions) < 2:
            for arm, mission in list(self.missions.items()):
                if not mission.preparation_only or mission.handoff_assignment is None or not mission.preparation_complete:
                    continue
                if mission.handoff_lead_assignment is not None and not mission.lead_started:
                    continue
                assignment = mission.handoff_assignment
                self.missions.pop(arm)
                self._log("handoff_ready", object_id=assignment.object_id, arm=arm.value)
                self._start_assignment(assignment)

    def _start_handoff_preparation(self, assignment) -> bool:
        """Move a deferred peer to the closest safe handoff pose."""
        blockers = [mission for mission in self.missions.values() if mission.arm is not assignment.arm]
        if blockers:
            blocker = blockers[0]
        else:
            # Bootstrap mode: screen against the other arm's current static
            # posture before admitting the lead assignment.
            other_arm = ArmId.B if assignment.arm is ArmId.A else ArmId.A
            other_kin = self.kinematics[other_arm]
            qpos = self.data.qpos[other_kin.qpos_addresses].copy()
            blocker = ArmMission(
                other_arm,
                "__static_peer__",
                "",
                [("static", 999.0, qpos, GRIP_OPEN_M)],
                self.data.time,
                trajectory=[(self.data.time, qpos.copy(), GRIP_OPEN_M)],
                stage_start_qpos=qpos.copy(),
            )
        base = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
        kin = self.kinematics[assignment.arm]
        escape_qpos = {
            ArmId.A: np.array((-2.1728, -0.2902, -1.8000, 2.5770, 4.0202, -6.1516)),
            ArmId.B: np.array((3.3961, 1.7425, -1.3483, 0.0000, 1.9187, 0.0000)),
        }[assignment.arm]
        rng = np.random.default_rng(self.seed + int(self.data.time * 1000.0) + (1 if assignment.arm is ArmId.A else 2))
        ranges = np.array([self.model.jnt_range[joint_id] for joint_id in kin.joint_ids])
        candidates = [escape_qpos]
        candidates.extend(rng.uniform(ranges[:, 0] + 0.05, ranges[:, 1] - 0.05) for _ in range(96))

        safe_qpos = None
        for qpos in candidates:
            probe = copy.deepcopy(base)
            probe.preparation_only = True
            probe.keyframes = [("handoff_escape", 4.0, qpos.copy(), GRIP_OPEN_M)]
            probe.trajectory = self._build_trajectory(probe.arm, probe.keyframes)
            probe.stage_start_qpos = self.data.qpos[kin.qpos_addresses].copy()
            safe, _ = self._preflight_joint_pair(probe, blocker)
            if safe:
                safe_qpos = qpos.copy()
                break
        if safe_qpos is None:
            self._log("handoff_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="no_safe_standby")
            return False

        # Find the furthest approach posture that remains safe over the peer's
        # complete future trajectory. This is the geometric handoff frontier.
        ready_qpos = safe_qpos.copy()
        for alpha in np.linspace(0.1, 1.0, 10):
            qpos = interpolate(safe_qpos, base.keyframes[1][2], float(alpha))
            probe = copy.deepcopy(base)
            probe.preparation_only = True
            readiness = self.bayesian_game.belief_for("outer", "direct").mean
            approach_duration = 0.8 + 1.6 * (1.0 - readiness)
            probe.keyframes = [
                ("handoff_escape", 4.0, safe_qpos.copy(), GRIP_OPEN_M),
                ("handoff_ready", approach_duration, qpos.copy(), GRIP_OPEN_M),
            ]
            probe.trajectory = self._build_trajectory(probe.arm, probe.keyframes)
            probe.stage_start_qpos = self.data.qpos[kin.qpos_addresses].copy()
            safe, _ = self._preflight_joint_pair(probe, blocker)
            if safe:
                ready_qpos = qpos.copy()
            else:
                break
        base.preparation_only = True
        base.handoff_assignment = assignment
        base.handoff_lead_assignment = self.handoff_leads.pop(assignment.arm, None)
        base.handoff_target_qpos = base.keyframes[1][2].copy()
        base.route_variant = "handoff_prepare"
        readiness = self.bayesian_game.belief_for("outer", "direct").mean
        approach_duration = 0.8 + 1.6 * (1.0 - readiness)
        base.keyframes = [
            ("handoff_escape", 4.0, safe_qpos, GRIP_OPEN_M),
            ("handoff_ready", approach_duration, ready_qpos, GRIP_OPEN_M),
        ]
        base.keyframe_index = 0
        base.keyframe_started_s = self.data.time
        base.trajectory = self._build_trajectory(base.arm, base.keyframes)
        base.stage_start_qpos = self.data.qpos[kin.qpos_addresses].copy()
        self.missions[assignment.arm] = base
        self._log("handoff_prepare", object_id=assignment.object_id, arm=assignment.arm.value, readiness=round(readiness, 3))
        return True

    def _start_handoff_creep(self, standby: ArmMission) -> None:
        """Try a slow, screened approach while the lead arm performs its pick."""
        if standby.handoff_target_qpos is None or standby.arm not in self.missions:
            return
        kin = self.kinematics[standby.arm]
        current = self.data.qpos[kin.qpos_addresses].copy()
        readiness = self.bayesian_game.belief_for("outer", "direct").mean
        duration = 2.0 + 3.0 * (1.0 - readiness)
        probe = copy.deepcopy(standby)
        probe.preparation_complete = False
        probe.keyframes = [("handoff_creep", duration, standby.handoff_target_qpos.copy(), GRIP_OPEN_M)]
        probe.keyframe_index = 0
        probe.keyframe_started_s = self.data.time
        probe.trajectory = self._build_trajectory(probe.arm, probe.keyframes)
        probe.stage_start_qpos = current.copy()
        lead = self.missions.get(standby.handoff_lead_assignment.arm) if standby.handoff_lead_assignment is not None else None
        if lead is None:
            return
        safe, _ = self._preflight_joint_pair(lead, probe)
        if not safe:
            self._log("handoff_creep_hold", object_id=standby.object_id, arm=standby.arm.value, reason="peer_path_screen")
            return
        standby.keyframes = probe.keyframes
        standby.keyframe_index = 0
        standby.keyframe_started_s = self.data.time
        standby.trajectory = probe.trajectory
        standby.stage_start_qpos = current.copy()
        standby.preparation_complete = False
        self._log("handoff_creep", object_id=standby.object_id, arm=standby.arm.value, duration=round(duration, 2))

    def _start_assignment(self, assignment) -> bool:
        mission = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
        safe, reason = self._preflight_mission(mission)
        if not safe and "safety_envelope" in reason:
            # A prepared peer changes the admissible corridor. Re-evaluate the
            # same assignment through the Bayesian route candidates instead
            # of treating the first direct IK path as the only possibility.
            for candidate in self._route_candidates(mission):
                candidate_safe, candidate_reason = self._preflight_mission(candidate)
                if candidate_safe:
                    mission, safe, reason = candidate, True, "clear_alternate_route"
                    break
        self.last_preflight = {"status": "clear" if safe else "deferred", "object_id": assignment.object_id, "arm": assignment.arm.value, "reason": reason}
        if not safe:
            if "safety_envelope" in reason:
                self.deferred_assignments[assignment.arm] = assignment
                self._log("reserve_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="path_collision", contact=reason)
            else:
                # A fixed obstacle will not become feasible by waiting. Release
                # the commitment so the next centralized cycle can try the
                # equal-peer arm with its updated assignment count.
                self._release_unstarted_assignment(assignment)
                self._log("screen_reject", object_id=assignment.object_id, arm=assignment.arm.value, reason="fixed_path_collision", contact=reason)
            return False
        self.missions[assignment.arm] = mission
        self.arm_outcomes[assignment.arm]["attempts"] += 1
        self._log("assign", object_id=assignment.object_id, arm=assignment.arm.value, placement=assignment.placement_zone)
        return True

    def _release_unstarted_assignment(self, assignment) -> None:
        """Undo accounting for an executor-rejected high-level commitment."""
        self.coordinator.mark_completed(assignment.object_id)
        self.coordinator.assignment_counts[assignment.arm] = max(
            0,
            self.coordinator.assignment_counts[assignment.arm] - 1,
        )

    def _update_missions(self) -> None:
        for arm, mission in list(self.missions.items()):
            # Failed simultaneous grips leave both wrists near the conveyor.
            # Release their recovery reservation in a deterministic order so
            # two retreat paths never begin from the same narrow corridor.
            if mission.failed and arm is ArmId.B and ArmId.A in self.missions and self.missions[ArmId.A].failed:
                continue
            kin = self.kinematics[arm]
            if mission.preparation_only and mission.preparation_complete:
                # Hold the last safe standby pose.  This is a real executor
                # state, not a completed pick, so the object remains claimed
                # until the peer clears and the handoff is activated.
                target = mission.keyframes[-1][2]
                kin.command_joint_pose(target, GRIP_OPEN_M)
                mission.last_safe_qpos = target.copy()
                continue
            # Safety guards may stretch a stage while the object keeps moving.
            # Recompute the interception from the remaining close time so a
            # delayed arm does not close at a stale conveyor coordinate.
            self._refresh_intercept(mission)
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
            safety_check_due = self.data.time >= mission.next_safety_check_s
            if safety_check_due:
                mission.next_safety_check_s = self.data.time + SAFETY_CHECK_PERIOD_S
            if require_clearance and safety_check_due and not self._pose_is_safe(arm, current, commanded_opening):
                if mission.last_safe_qpos is not None:
                    kin.command_joint_pose(mission.last_safe_qpos, commanded_opening)
                mission.keyframe_started_s += CONTROL_STEP_S
                mission.next_safety_check_s = self.data.time + CONTROL_STEP_S
                continue
            stage_start = mission.stage_start_qpos if mission.stage_start_qpos is not None else current
            commanded_qpos = interpolate(stage_start, target, smoothstep(elapsed / max(duration, CONTROL_STEP_S)))
            if require_clearance and safety_check_due and not self._pose_is_safe(arm, commanded_qpos, commanded_opening):
                mission.keyframe_started_s += CONTROL_STEP_S
                mission.next_safety_check_s = self.data.time + CONTROL_STEP_S
                if self.data.time - mission.last_safety_hold_s >= 0.5:
                    mission.last_safety_hold_s = self.data.time
                    self._log("reserve_wait", object_id=mission.object_id, arm=arm.value, reason="step_collision_guard")
                continue
            kin.command_joint_pose(commanded_qpos, commanded_opening)
            mission.last_safe_qpos = current.copy()
            joint_error = float(np.max(np.abs(target - current)))
            joint_speed = float(np.max(np.abs(self.data.qvel[kin.dof_addresses])))
            conveyor_tracking_stage = stage in ("prepare", "track", "descend", "close")
            stage_reached = elapsed >= duration and joint_error <= 0.08 and (
                conveyor_tracking_stage or joint_speed <= 0.12
            )
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
            if mission.preparation_only and mission.done:
                mission.preparation_complete = True
                mission.keyframe_index = len(mission.keyframes) - 1
                mission.keyframe_started_s = self.data.time
                mission.next_replan_s = float("inf")
                mission.stage_start_qpos = target.copy()
                self._log("handoff_standby", object_id=mission.object_id, arm=arm.value)
                continue
            if mission.done:
                kin.set_pad_adhesion(0.0)
                self.missions.pop(arm)
                placement_success = not mission.failed and self._part_is_in_target_bin(mission.object_id, mission.placement_zone)
                if placement_success:
                    self.placed.add(mission.object_id)
                    self.arm_outcomes[arm]["placed"] += 1
                    self._log("place", object_id=mission.object_id, placement=mission.placement_zone)
                elif not mission.failed:
                    self.missed.add(mission.object_id)
                    self._log("missed", object_id=mission.object_id, reason="placement_not_verified", part_xyz=np.round(self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3], 3).tolist())
                self.coordinator.mark_completed(mission.object_id)
                self._record_joint_outcome(mission, placement_success)
                measured_cycle_s = self.data.time - mission.assigned_at_s
                arm_feedback = self.arm_outcomes[arm]
                arm_feedback["cycle_s"] = 0.8 * arm_feedback["cycle_s"] + 0.2 * measured_cycle_s
                self.coordinator.fixed_cycle_s = float(np.mean([item["cycle_s"] for item in self.arm_outcomes.values()]))
                self._log("cycle_feedback", arm=arm.value, cycle_s=round(measured_cycle_s, 2), estimate_s=round(self.coordinator.fixed_cycle_s, 2))

    def _record_joint_outcome(self, mission: ArmMission, success: bool) -> None:
        if mission.joint_strategy is None:
            return
        outcomes = self.joint_result_buffer.setdefault(mission.joint_strategy, [])
        outcomes.append(success)
        if len(outcomes) < 2:
            return
        joint_success = all(outcomes)
        self.bayesian_game.update(*mission.joint_strategy, joint_success)
        posterior = self.bayesian_game.belief_for(*mission.joint_strategy).mean
        self._log(
            "bayes_update",
            route_a=mission.joint_strategy[0],
            route_b=mission.joint_strategy[1],
            success=joint_success,
            posterior=round(posterior, 4),
        )
        self.joint_result_buffer.pop(mission.joint_strategy, None)

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
                    # Items with the same scheduled timestamp form one batch.
                    # Capacity remains authoritative if an older batch has not
                    # yet cleared the physical line.
                    if active_parts >= int(self.parameters.max_active_parts) or not self._has_available_handler(item):
                        break
                    place_part(self.data, self.qpos_addresses[item.part_name], item.spawn_xyz)
                    self.spawned.add(item.part_name)
                    active_parts += 1
                    self._log("infeed", object_id=item.part_name, object_class=item.object_class.value)
            self._schedule()
            self._update_missions()
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
                        substeps = 0
                        while not self.paused and accumulated_s >= CONTROL_STEP_S and self.data.time < duration_s and substeps < MAX_VIEWER_SUBSTEPS:
                            self.step()
                            accumulated_s -= CONTROL_STEP_S
                            substeps += 1
                        if substeps == MAX_VIEWER_SUBSTEPS:
                            accumulated_s = min(accumulated_s, CONTROL_STEP_S * MAX_VIEWER_SUBSTEPS)
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
