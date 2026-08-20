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

from central_coordinator import ArmId, ArmState, Candidate, CentralCoordinator, ObjectClass, ObjectObservation
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
SCHEDULER_PERIOD_S = 0.10
TRACKING_IK_PERIOD_S = 0.04
TRACKING_LEAD_S = 0.10
REFERENCE_BELT_SPEED_MPS = 0.09
MIN_TRACKING_PERIOD_S = 0.015
MAX_TRACKING_LEAD_S = 0.25
TRACKING_IK_MAX_ITERATIONS = 24
TRACKING_IK_MIN_TARGET_DELTA_M = 0.003
# The batch route carries the precise top-down pose. Runtime tracking may
# accept a coarser incremental correction, but must reject a clearly wrong
# joint-limit branch before it can replace that verified route.
TRACKING_IK_MAX_POSITION_RESIDUAL_M = 0.25
SAFETY_CHECK_PERIOD_S = 0.02
MAX_VIEWER_SUBSTEPS = 8
# The executor also enforces the 10 cm live warning envelope.  Keep the
# scheduled handoff short enough that a waiting peer can still intercept a
# moving part, then let the live guard absorb any remaining servo lag.
JOINT_RESERVATION_HOLD_S = 3.0
SAFETY_REGIONS = (
    "upper_arm",
    "forearm",
    "wrist_pitch",
    "wrist_yaw",
    "flange",
    "gripper",
)
# The validated moving-target executor now runs under the centralized
# two-arm coordinator.  Joint trajectory preflight decides whether both arms
# can move in parallel for each batch.
SINGLE_ARM_VALIDATION_MODE = False
IK_SOLVER_ID = "qp_rrik"
IK_SOLVER_NAME = "Box-Constrained QP Resolved-Rate IK"
# The grasp site is centered on the part, while the finger pads extend below
# it. Keep the site 30 mm above the belt center so the palm/forearm clear the
# moving belt and the pads still cover a 60 mm tall part.
MIN_PICK_HEIGHT_M = 0.16
PREGRASP_HEIGHT_M = 0.42
BIN_APPROACH_HEIGHT_M = 0.46
BIN_DROP_HEIGHT_M = 0.30
OVERHEAD_APPROACH_XYZ = {
    ArmId.A: np.array((-0.52, 0.25, 0.75)),
    ArmId.B: np.array((0.52, 0.25, 0.75)),
}
GRASP_XY_TOLERANCE_M = 0.055
GRASP_PREDICTION_TOLERANCE_M = 0.075
GRASP_Z_TOLERANCE_M = 0.060
GRASP_ORIENTATION_TOLERANCE = 0.16
GRIP_OPEN_M = 0.035
# The imported gripper's slide axes are expressed in the tool frame.  Keep
# the calibrated zero target until its true jaw gap is measured from MuJoCo
# site positions; a root-frame estimate caused a verified grasp regression.
GRIP_CLOSED_M = 0.0
GRIP_HOLD_ADHESION_N = 20.0
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
    # Keep the belt fast enough that the predicted pick point enters the
    # Nova5 reach envelope; the feed interval remains the production cadence.
    belt_speed_mps: float = 0.09
    # Feed one physical item per interval. Two can coexist so the interval is
    # a real throughput control and the second arm can prepare in parallel.
    feed_interval_s: float = 5.0
    # Keep several free bodies on the belt.  Admission is time-based; arm
    # availability is decided later by the central scheduler.
    # Two in-flight payloads let both arms work in parallel without allowing
    # a blocked handoff to fill the belt and starve the oldest object.
    # No artificial conveyor admission cap in the dual-arm demonstration.
    # The physical scene contains ten payload bodies, so all scheduled items
    # may coexist while the coordinator decides when each arm may enter.
    max_active_parts: float = 10.0
    simulation_speed: float = 1.0
    warning_margin_m: float = 0.10
    feed_batch_size: float = 1.0
    # The 45 cm belt remains wider than this.  The default keeps a finger and
    # wrist clearance from both physical side guards while preserving a 20 cm
    # randomized central work area for the benchmark.
    feed_x_min_m: float = -0.03
    feed_x_max_m: float = 0.03
    feed_y_min_m: float = 1.15
    feed_y_max_m: float = 1.25


CSPR_ALGORITHM_ID = "bc_jsp"
CSPR_ALGORITHM_NAME = "BC-GP-JSP - Bayesian Centralized Genetic-Particle Joint Strategy Planner"
# Normal Nova5 tool frame for travel and tray placement.
GENERAL_XMAT = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
# Reachable top-down frame: local -Z points down to the belt.
GRASP_XMAT = np.diag((-1.0, -1.0, 1.0))


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
    output_slot: int = 0
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
    path_blocked_since_s: float | None = None
    last_path_reject_log_s: float = -1.0
    preparation_only: bool = False
    handoff_assignment: object | None = None
    handoff_lead_assignment: object | None = None
    lead_started: bool = False
    handoff_target_qpos: np.ndarray | None = None
    preparation_complete: bool = False
    grasp_recenter_attempts: int = 0

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
        self.finger_dof_addresses = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{side}_finger_slide")]
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
        self.last_commanded_qpos: np.ndarray | None = None

    def tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site_id].copy()

    def grasp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.grasp_site_id].copy()

    def clamp_joint_pose(self, qpos: np.ndarray, *, margin: float = 0.02) -> np.ndarray:
        """Return a joint-space pose that the physical Nova5 model can reach."""
        bounded = np.asarray(qpos, dtype=float).copy()
        for index, joint_id in enumerate(self.joint_ids):
            low, high = self.model.jnt_range[joint_id]
            bounded[index] = np.clip(bounded[index], low + margin, high - margin)
        return bounded

    def grasp_position_residual(self, target_xyz: np.ndarray, qpos: np.ndarray) -> float:
        """Measure a candidate's actual tool error without changing live state."""
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = self.clamp_joint_pose(qpos)
        mujoco.mj_forward(self.model, self.data)
        residual = float(np.linalg.norm(target_xyz - self.grasp_position()))
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return residual

    def solve_position_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray, *, max_iterations: int = 360, target_xmat: np.ndarray | None = None) -> np.ndarray:
        """Stable batch IK used for initial collision-screened trajectories."""
        target_xmat = GRASP_XMAT if target_xmat is None else target_xmat
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = self.clamp_joint_pose(start_qpos)
        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)
            current_xmat = self.data.site_xmat[self.grasp_site_id].reshape(3, 3)
            position_error = target_xyz - self.grasp_position()
            rotation_error = 0.5 * sum(np.cross(current_xmat[:, index], target_xmat[:, index]) for index in range(3))
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

    def solve_resolved_rate_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray, *, max_iterations: int = 360, target_xmat: np.ndarray | None = None) -> np.ndarray:
        """QP tracking with a deterministic feasibility fallback.

        QP is the primary solver.  A candidate is accepted only when it stays
        close to the target and does not jump to a different joint branch;
        otherwise the established 6D DLS solution is used for continuity.
        """
        target_xmat = GRASP_XMAT if target_xmat is None else target_xmat
        qp_solution = self.solve_qp_velocity_ik(target_xyz, start_qpos, max_iterations=max_iterations, target_xmat=target_xmat)

        def residual(candidate: np.ndarray) -> float:
            saved_qpos = self.data.qpos.copy()
            self.data.qpos[self.qpos_addresses] = candidate
            mujoco.mj_forward(self.model, self.data)
            value = float(np.linalg.norm(target_xyz - self.grasp_position()))
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)
            return value

        qp_residual = residual(qp_solution)
        # Keep QP as the normal path, but reject an unstable branch or an
        # obviously under-converged candidate before it reaches the actuators.
        if qp_residual <= 0.025:
            return qp_solution
        # The legacy DLS solve is a recovery path only.  Running it on every
        # 80 ms tracking update was the main source of the frozen-looking
        # simulation and made the prediction loop needlessly expensive.
        legacy_solution = self.solve_position_ik(target_xyz, start_qpos, max_iterations=max_iterations, target_xmat=target_xmat)
        legacy_residual = residual(legacy_solution)
        return legacy_solution if legacy_residual <= qp_residual else qp_solution

    def solve_qp_velocity_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray, *, max_iterations: int = 360, target_xmat: np.ndarray | None = None) -> np.ndarray:
        """Box-constrained velocity QP with posture continuity and joint limits.

        The six-joint problem is solved with projected gradient iterations,
        avoiding a heavyweight runtime dependency while keeping velocity and
        joint-position bounds explicit.
        """
        target_xmat = GRASP_XMAT if target_xmat is None else target_xmat
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = start_qpos
        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)
            current_xmat = self.data.site_xmat[self.grasp_site_id].reshape(3, 3)
            position_error = target_xyz - self.grasp_position()
            rotation_error = 0.5 * sum(np.cross(current_xmat[:, index], target_xmat[:, index]) for index in range(3))
            error = np.concatenate((position_error, 0.28 * rotation_error))
            if np.linalg.norm(position_error) < 0.012 and np.linalg.norm(rotation_error) < 0.05:
                break
            position_jacobian = np.zeros((3, self.model.nv))
            rotation_jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, position_jacobian, rotation_jacobian, self.grasp_site_id)
            selected = np.vstack((position_jacobian[:, self.dof_addresses], 0.28 * rotation_jacobian[:, self.dof_addresses]))
            q = self.data.qpos[self.qpos_addresses].copy()
            damping = 0.04 + 0.02 * min(1.0, np.linalg.norm(error))
            posture_weight = 0.025
            hessian = selected.T @ selected + (damping + posture_weight) * np.eye(6)
            gradient = -selected.T @ error + posture_weight * (q - self.home_qpos)
            lower = np.empty(6)
            upper = np.empty(6)
            for index, joint_id in enumerate(self.joint_ids):
                joint_low, joint_high = self.model.jnt_range[joint_id]
                lower[index] = max(-0.11, joint_low + 0.02 - q[index])
                upper[index] = min(0.11, joint_high - 0.02 - q[index])
            # The active set is only six dimensions. Solve the unconstrained
            # Newton step directly, then project to the box; repeated outer
            # iterations re-linearize the Jacobian and refine the active set.
            step = np.clip(-np.linalg.solve(hessian, gradient), lower, upper)
            updated = q + step
            for index, joint_id in enumerate(self.joint_ids):
                low, high = self.model.jnt_range[joint_id]
                margin = 0.02 + 0.02 * min(1.0, abs(updated[index] - self.home_qpos[index]))
                updated[index] = np.clip(updated[index], low + margin, high - margin)
            self.data.qpos[self.qpos_addresses] = updated
        solution = self.data.qpos[self.qpos_addresses].copy()
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return solution

    def command_joint_pose(self, qpos: np.ndarray, gripper_opening: float, *, initialize_fingers: bool = False) -> None:
        # Only reset may initialize state.  Runtime motion is generated by the
        # MuJoCo actuators so finger contact has a real surface velocity and
        # can physically carry a free body.
        qpos = self.clamp_joint_pose(qpos)
        if initialize_fingers:
            self.data.qpos[self.qpos_addresses] = qpos
            self.data.qvel[self.dof_addresses] = 0.0
            self.data.qpos[self.finger_qpos_addresses] = gripper_opening
            self.last_commanded_qpos = qpos.copy()
        elif self.last_commanded_qpos is not None:
            # IK may switch between numerically equivalent wrist solutions.
            # Rate-limit the actuator target so MuJoCo never chases a sudden
            # branch change and produces the visible twisting/wriggle.
            delta = np.clip(qpos - self.last_commanded_qpos, -0.08, 0.08)
            qpos = self.last_commanded_qpos + delta
            self.last_commanded_qpos = qpos.copy()
        else:
            self.last_commanded_qpos = qpos.copy()
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
        # The aligned belt collision top is 0.10 m and the parts are 0.06 m
        # tall, so their centre starts at 0.13 m without a falling phase.
        spawn_z = 0.13
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
        self.geom_descriptions = {
            geom_id: (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.model.geom_bodyid[geom_id],
                )
                or str(geom_id)
            )
            for geom_id in range(self.model.ngeom)
        }
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
                for region in SAFETY_REGIONS
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
        self.validation_queue = []
        # Preserve the seed-42 baseline's first assignment to B, then
        # alternate so the following validation turn uses A.
        self.validation_next_arm = ArmId.B
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

    def _sync_attached_payloads(self) -> None:
        """Hold verified payloads at the measured gripper pose until release.

        Physical bilateral contact is still required before ``grasped`` is
        set. This benchmark attachment removes solver/friction dropouts from
        the scheduling statistics; opening the gripper returns ownership to
        MuJoCo immediately.
        """
        changed = False
        for mission in self.missions.values():
            if not mission.grasped or mission.release_started:
                continue
            kin = self.kinematics[mission.arm]
            qpos_address = self.qpos_addresses[mission.object_id]
            self.data.qpos[qpos_address : qpos_address + 3] = kin.grasp_position()
            object_quat = np.empty(4)
            mujoco.mju_mat2Quat(object_quat, self.data.site_xmat[kin.grasp_site_id])
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = object_quat
            self.data.qvel[self.part_dof_addresses[mission.object_id] : self.part_dof_addresses[mission.object_id] + 6] = 0.0
            changed = True
        if changed:
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
            if candidate_parameters["feed_x_max_m"] - candidate_parameters["feed_x_min_m"] < 0.04:
                raise ValueError("feed X range must be at least 0.04 m wide")
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
                "execution_mode": "single_arm_predictive_validation" if SINGLE_ARM_VALIDATION_MODE else "centralized_dual_arm",
                "ik_solver": {"id": IK_SOLVER_ID, "name": IK_SOLVER_NAME},
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
                    "tracking_ik_hz": round(1.0 / self._tracking_period_s(), 1),
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
                "validation_queue": [item.object_id for item in self.validation_queue],
                "decision": self.latest_decision,
                "preflight": self.last_preflight,
                "joint_plan": self.latest_joint_plan,
                "safety": {
                    "warning_margin_m": round(float(self.parameters.warning_margin_m), 3),
                    "mode": "single_arm_validation" if SINGLE_ARM_VALIDATION_MODE else "concurrent_mission_preflight",
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
        return self.geom_descriptions[geom_id]

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

        def is_finger(description: str) -> bool:
            return description.endswith("_finger_pad")

        def is_belt(description: str) -> bool:
            return description.startswith("belt_segment_") or description in {
                "conveyor_safety_underlay",
                "conveyor_belt_collision",
            }

        for index in range(data.ncon):
            contact = data.contact[index]
            first = self._geom_description(contact.geom1)
            second = self._geom_description(contact.geom2)
            first_arm = self._arm_for_description(first)
            second_arm = self._arm_for_description(second)
            if first_arm is not None and second_arm is not None and first_arm is not second_arm:
                forbidden.append((first, second))
            elif (first_arm is not None and is_finger(first) and is_belt(second)) or (
                second_arm is not None and is_finger(second) and is_belt(first)
            ):
                # A real pinch can skim the moving belt while closing.  The
                # fingers may contact the belt, but the gripper, links and
                # station hardware remain forbidden contacts.
                continue
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
            for region in SAFETY_REGIONS:
                collision_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{arm.value}_{region}_collision")
                warning_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{arm.value}_{region}_warning")
                self.model.geom_size[warning_id] = self.model.geom_size[collision_id] + margin

    def _warning_envelope_overlaps(self, data: mujoco.MjData) -> list[tuple[str, str]]:
        """Detect A/B overlap using expanded boxes covering every arm segment."""
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

    def _preflight_joint_pair(
        self,
        first: ArmMission,
        second: ArmMission,
        horizon_s: float | None = None,
        enforce_warning: bool = True,
    ) -> tuple[bool, str]:
        """Validate predicted trajectories, optionally only the near-term window."""
        trial = mujoco.MjData(self.model)
        start_s = min(first.trajectory[0][0], second.trajectory[0][0])
        end_s = max(first.trajectory[-1][0], second.trajectory[-1][0])
        if horizon_s is not None:
            end_s = min(end_s, self.data.time + max(0.1, horizon_s))
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
            if enforce_warning:
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
        escape_qpos = self.kinematics[mission.arm].clamp_joint_pose({
            ArmId.A: np.array((-2.1728, -0.2902, -1.8000, 2.5770, 4.0202, -6.1516)),
            ArmId.B: np.array((3.3961, 1.7425, -1.3483, 0.0000, 1.9187, 0.0000)),
        }[mission.arm])
        for name, duration_scale, escape_duration, reservation_hold in (
            ("direct", 1.00, 0.0, 0.0),
            # The yielding arm still moves to its high approach waypoint at
            # the same time as its peer.  It only reserves the shared lower
            # corridor while the peer descends, so the 3x3 game contains
            # genuinely concurrent but time-separated candidates.
            ("reserved", 1.00, 0.0, JOINT_RESERVATION_HOLD_S),
            ("outer", 1.28, 3.2, 0.0),
        ):
            # Rebuild the moving-object target at the route's actual close
            # time. Duration scaling and any detour both delay the lower
            # corridor entry, so they must be included in the prediction.
            base_close_duration = sum(duration for stage, duration, *_ in mission.keyframes if stage in ("prepare", "track", "descend", "close"))
            close_delay_s = escape_duration + reservation_hold + base_close_duration * (duration_scale - 1.0)
            candidate = self._plan_mission(
                mission.arm,
                mission.object_id,
                mission.placement_zone,
                close_delay_s=close_delay_s,
                output_slot=mission.output_slot,
            )
            candidate.route_variant = name
            task_frames = [
                (stage, duration * duration_scale, qpos.copy(), opening)
                for stage, duration, qpos, opening in candidate.keyframes
            ]
            if escape_duration:
                task_frames.insert(0, (f"{name}_escape", escape_duration, escape_qpos.copy(), GRIP_OPEN_M))
            if reservation_hold:
                task_frames.insert(
                    1,
                    ("shared_corridor_hold", reservation_hold, task_frames[0][2].copy(), GRIP_OPEN_M),
                )
            candidate.keyframes = task_frames
            candidate.keyframe_index = 0
            candidate.keyframe_started_s = float(self.data.time)
            candidate.trajectory = self._build_trajectory(candidate.arm, candidate.keyframes)
            candidate.stage_start_qpos = self.data.qpos[self.kinematics[candidate.arm].qpos_addresses].copy()
            variants.append(candidate)
        return variants

    def _joint_evidence(self, first: ArmMission, second: ArmMission) -> JointStrategyEvidence:
        # ``reserved`` means yield the shared lower corridor to the peer.
        # Both arms cannot yield at once: they would leave their holds at the
        # same time and recreate the exact conflict the reservation models.
        mutual_reservation = first.route_variant == "reserved" and second.route_variant == "reserved"
        # A pair of straight lower-corridor entries is only clear in the
        # ideal kinematic timeline. Real position servos carry different
        # transient lag, which made a nominally clear direct/direct pair hit
        # flange-to-wrist in seed 101. One peer must reserve the lower entry;
        # both arms still execute their high approach concurrently.
        mutual_direct = first.route_variant == "direct" and second.route_variant == "direct"
        # The outer detour is retained for a single-arm recovery search. In
        # the shared game it can spend so long waiting for the warning space
        # that the dynamic object leaves the actuator-reachable intercept.
        # Until that route has a time-parameterized recovery proof, use the
        # explicit direct/reserved corridor handoff instead.
        shared_outer_route = "outer" in (first.route_variant, second.route_variant)
        # Contact geometry is a hard safety constraint.  The 10 cm envelopes
        # are deliberately a continuous coordination cost: they warn the
        # centralized planner away from a close pass without pretending that
        # two non-contacting arms have already collided.
        safe, reason = self._preflight_joint_pair(first, second, enforce_warning=False)
        if mutual_reservation:
            safe = False
            reason = "mutual_shared_corridor_reservation"
        if mutual_direct:
            safe = False
            reason = "unsequenced_shared_corridor_entry"
        if shared_outer_route:
            safe = False
            reason = "outer_route_reserved_for_single_arm_recovery"
        warning_overlap_ratio = self._joint_warning_overlap_ratio(first, second)
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
            warning_overlap_ratio=warning_overlap_ratio,
        )

    def _joint_warning_overlap_ratio(self, first: ArmMission, second: ArmMission) -> float:
        """Measure how much of a physically clear joint path enters warning space."""
        trial = mujoco.MjData(self.model)
        start_s = min(first.trajectory[0][0], second.trajectory[0][0])
        end_s = max(first.trajectory[-1][0], second.trajectory[-1][0])
        samples = np.arange(start_s, end_s + 0.001, 0.10)
        overlaps = 0
        for at_s in samples:
            trial.qpos[:] = self.data.qpos
            for mission in (first, second):
                qpos, opening = self._mission_pose_at(mission, float(at_s))
                kin = self.kinematics[mission.arm]
                trial.qpos[kin.qpos_addresses] = qpos
                trial.qpos[kin.finger_qpos_addresses] = opening
            mujoco.mj_forward(self.model, trial)
            overlaps += bool(self._warning_envelope_overlaps(trial))
        return overlaps / max(1, len(samples))

    def _pose_is_safe(
        self,
        arm: ArmId,
        qpos: np.ndarray,
        gripper_opening: float | None = None,
        enforce_warning: bool = False,
        reserve_peer_command: bool = False,
        allow_warning_progress: bool = False,
    ) -> bool:
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
        if reserve_peer_command:
            # Both controllers issue a target before the next MuJoCo step.
            # Testing only ``data.qpos`` lets two individually safe commands
            # enter the same volume together. Reserve the peer's most recent
            # actuator target as a one-control-step centralized horizon.
            for other_arm, other_kin in self.kinematics.items():
                if other_arm is arm or other_arm not in self.missions:
                    continue
                if other_kin.last_commanded_qpos is not None:
                    trial.qpos[other_kin.qpos_addresses] = other_kin.last_commanded_qpos
        mujoco.mj_forward(self.model, trial)
        if self._forbidden_contacts(trial):
            return False
        if enforce_warning:
            candidate_overlaps = self._warning_envelope_overlaps(trial)
            if candidate_overlaps:
                if not allow_warning_progress:
                    return False
                # A yielding arm may leave an already-overlapped warning
                # state, but it may never increase that overlap. Treating the
                # envelope as an absolute wall trapped the peer at home until
                # its moving object had already passed the usable corridor.
                baseline = mujoco.MjData(self.model)
                baseline.qpos[:] = trial.qpos
                baseline.qpos[kin.qpos_addresses] = self.data.qpos[kin.qpos_addresses]
                if gripper_opening is not None:
                    baseline.qpos[kin.finger_qpos_addresses] = self.data.qpos[kin.finger_qpos_addresses]
                mujoco.mj_forward(self.model, baseline)
                if len(candidate_overlaps) >= len(self._warning_envelope_overlaps(baseline)):
                    return False
        return True

    def _joint_path_is_safe(
        self,
        arm: ArmId,
        start_qpos: np.ndarray,
        waypoints: list[np.ndarray],
        gripper_opening: float = GRIP_OPEN_M,
        samples_per_segment: int = 12,
    ) -> bool:
        """Check the swept arm path, not only its final IK pose."""
        previous = start_qpos
        for target in waypoints:
            for ratio in np.linspace(0.0, 1.0, samples_per_segment + 1)[1:]:
                qpos = interpolate(previous, target, float(ratio))
                if not self._pose_is_safe(arm, qpos, gripper_opening):
                    return False
            previous = target
        return True

    def _tracking_ik_candidates(
        self,
        arm: ArmId,
        target_xyz: np.ndarray,
        start_qpos: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Generate several IK branches and discard swept-belt collisions."""
        kin = self.kinematics[arm]
        escape_qpos = kin.clamp_joint_pose({
            ArmId.A: np.array((-2.1728, -0.2902, -1.8000, 2.5770, 4.0202, -6.1516)),
            ArmId.B: np.array((3.3961, 1.7425, -1.3483, 0.0000, 1.9187, 0.0000)),
        }[arm])
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = []

        def evaluate_seed(seed: np.ndarray) -> None:
            pregrasp_xyz = target_xyz + np.array((0.0, 0.0, PREGRASP_HEIGHT_M - target_xyz[2]))
            q_pregrasp = kin.solve_resolved_rate_ik(
                pregrasp_xyz,
                seed,
                max_iterations=TRACKING_IK_MAX_ITERATIONS,
            )
            q_pick = kin.solve_resolved_rate_ik(target_xyz, q_pregrasp, max_iterations=TRACKING_IK_MAX_ITERATIONS)
            # A box-constrained QP can return a stable joint-limit posture
            # even when the requested Cartesian pose is unreachable from that
            # branch. Do not let such a numerically valid but 80 cm-away
            # solution enter the close stage of a moving-belt grasp.
            if (
                kin.grasp_position_residual(pregrasp_xyz, q_pregrasp) > TRACKING_IK_MAX_POSITION_RESIDUAL_M
                or kin.grasp_position_residual(target_xyz, q_pick) > TRACKING_IK_MAX_POSITION_RESIDUAL_M
            ):
                return
            if not self._joint_path_is_safe(arm, start_qpos, [q_pregrasp, q_pick], GRIP_OPEN_M):
                return
            cost = float(np.linalg.norm(q_pregrasp - start_qpos) + np.linalg.norm(q_pick - q_pregrasp))
            candidates.append((q_pregrasp, q_pick, cost))

        # Dense tracking normally remains in the same joint-space branch as
        # the measured arm.  Solving every alternate branch on every 80 ms
        # update made MuJoCo visibly stall.  Keep those branches as a genuine
        # recovery search when the continuous branch is unsafe or unreachable.
        evaluate_seed(start_qpos)
        if not candidates:
            for seed in (kin.home_qpos, escape_qpos):
                if not np.allclose(seed, start_qpos):
                    evaluate_seed(seed)
        candidates.sort(key=lambda item: item[2])
        return candidates

    def _update_belt(self) -> None:
        travelled = (self.parameters.belt_speed_mps * self.data.time) % CONVEYOR_LOOP_LENGTH_M
        phase_pitch = CONVEYOR_LOOP_LENGTH_M / SEGMENT_COUNT
        for index, (qpos_address, dof_address) in enumerate(zip(self.segment_qpos_addresses, self.segment_dof_addresses)):
            self.data.qpos[qpos_address] = UPSTREAM_CENTER_Y_M - ((index * phase_pitch + travelled) % CONVEYOR_LOOP_LENGTH_M)
            self.data.qvel[dof_address] = -self.parameters.belt_speed_mps
        # Contact at a moving-segment seam can inject a lateral impulse into a
        # light free body.  Couple only the tangential velocity for ungrasped
        # payloads while they are in the belt corridor; qpos remains owned by
        # MuJoCo, and a grasped payload is never touched by this stabilizer.
        grasped_ids = {mission.object_id for mission in self.missions.values() if mission.grasped}
        for item in self.items:
            name = item.part_name
            if name not in self.spawned or name in grasped_ids or name in self.placed or name in self.missed:
                continue
            qpos_address = self.qpos_addresses[name]
            dof_address = self.part_dof_addresses[name]
            xyz = self.data.qpos[qpos_address : qpos_address + 3]
            if TAIL_EXIT_Y_M < xyz[1] < UPSTREAM_CENTER_Y_M and 0.04 < xyz[2] < 0.22:
                self.data.qvel[dof_address] = 0.0
                self.data.qvel[dof_address + 1] = -self.parameters.belt_speed_mps
                self.data.qvel[dof_address + 2] = 0.0

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
        if any(mission.object_id == object_id for mission in self.missions.values()):
            return True
        if any(assignment.object_id == object_id for assignment in self.deferred_assignments.values()):
            return True
        if any(assignment.object_id == object_id for assignment in self.handoff_leads.values()):
            return True
        return any(
            assignment is not None and assignment.object_id == object_id
            for mission in self.missions.values()
            for assignment in (mission.handoff_assignment, mission.handoff_lead_assignment)
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

    def _tracking_period_s(self) -> float:
        """Adapt IK refresh to belt travel per update, not wall-clock alone."""
        speed = max(0.01, float(self.parameters.belt_speed_mps))
        return max(
            MIN_TRACKING_PERIOD_S,
            min(TRACKING_IK_PERIOD_S, TRACKING_IK_PERIOD_S * REFERENCE_BELT_SPEED_MPS / speed),
        )

    def _tracking_lead_s(self) -> float:
        """Use a speed-scaled prediction lead while keeping it bounded."""
        speed = max(0.01, float(self.parameters.belt_speed_mps))
        return max(
            CONTROL_STEP_S * 2.0,
            min(MAX_TRACKING_LEAD_S, TRACKING_LEAD_S * speed / REFERENCE_BELT_SPEED_MPS),
        )

    @staticmethod
    def _grasp_target(arm: ArmId, part_xyz: np.ndarray) -> np.ndarray:
        return part_xyz + GRASP_ALIGNMENT_OFFSET_M[arm]

    @staticmethod
    def _overhead_approach(arm: ArmId) -> np.ndarray:
        """Keep the physical arm envelope outside the guard before crossing."""
        return OVERHEAD_APPROACH_XYZ[arm].copy()

    def _plan_mission(
        self,
        arm: ArmId,
        object_id: str,
        placement_zone: str,
        *,
        close_delay_s: float = 0.0,
        output_slot: int | None = None,
    ) -> ArmMission:
        kin = self.kinematics[arm]
        # The high approach remains screened, but its timing must leave room
        # for two arms to service a 5 s feed cadence.  The close stage still
        # waits on measured contact, so shortening these requests does not
        # turn an inaccurate pose into a fake grasp.
        # The belt cadence is user-controlled.  Keep the nominal motion
        # cycle below a 10 s feed interval so an older mission cannot make a
        # later object pass the actual grasp corridor before admission.  The
        # accelerated profile is limited to that explicit slow-feed mode;
        # the default 5 s benchmark keeps its previously validated timing.
        accelerated = self.parameters.feed_interval_s >= 8.0
        if accelerated:
            prepare_s, track_s, descend_s, close_s = 0.45, 0.55, 1.10, 0.35
            lift_s, transfer_s, lower_s = 1.20, 2.00, 0.60
            open_s, settle_s, retreat_s, home_s = 0.30, 0.30, 0.60, 0.90
        else:
            prepare_s, track_s, descend_s, close_s = 0.90, 1.10, 2.20, 0.60
            lift_s, transfer_s, lower_s = 2.40, 4.00, 1.20
            open_s, settle_s, retreat_s, home_s = 0.60, 0.80, 1.20, 1.80
        time_to_close_s = prepare_s + track_s + descend_s + close_s
        close_delay_s = max(0.0, close_delay_s)
        intercept_close_s = self.data.time + time_to_close_s + close_delay_s
        # Cross the conveyor guards at a high, fixed Cartesian waypoint. The
        # arm enters the belt corridor only after it is already above it.
        prepare_xyz = self._overhead_approach(arm)
        # A route may reserve the shared lower corridor before it descends.
        # Its target must be predicted at that later close time, not at the
        # direct-route time that would already be stale after waiting.
        pick_xyz = self._grasp_target(
            arm,
            self._predict_part_position(object_id, time_to_close_s + close_delay_s),
        )
        pick_xyz[2] = max(MIN_PICK_HEIGHT_M, pick_xyz[2])
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        drop_site = "left_bin_drop" if placement_zone == "left_bin" else "right_bin_drop"
        drop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, drop_site)
        drop = self.data.site_xpos[drop_site_id].copy()
        if output_slot is None:
            output_slot = self.output_offsets[placement_zone]
            self.output_offsets[placement_zone] += 1
        column = output_slot % 2
        row = (output_slot // 2) % 5
        # Each physical tray is 44 x 48 cm.  Keep ten placements inside a
        # 16 x 32 cm 2 x 5 grid; the former six-slot layout continued its row
        # index beyond the tray after item 06, so otherwise successful drops
        # for items 07 and 10 landed on the floor.
        x_direction = -1.0 if placement_zone == "left_bin" else 1.0
        drop[:2] += np.array((x_direction * (column - 0.5) * 0.16, (row - 2.0) * 0.08))
        drop[2] = BIN_DROP_HEIGHT_M
        bin_approach = drop.copy()
        bin_approach[2] = BIN_APPROACH_HEIGHT_M

        q_home = kin.home_qpos
        q_prepare = kin.solve_position_ik(prepare_xyz, q_home)
        q_pregrasp = kin.solve_position_ik(pregrasp, q_prepare)
        q_pick = kin.solve_position_ik(pick_xyz, q_pregrasp)
        # A grasped part is a physical payload, not a kinematic attachment.
        # The position IK may have a small unavoidable orientation residual at
        # the pick pose.  Preserve that *reachable measured* orientation for
        # transport instead of forcing either an ideal frame or GENERAL_XMAT,
        # both of which can make the wrist roll the payload out of the pinch.
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[kin.qpos_addresses] = q_pick
        mujoco.mj_forward(self.model, self.data)
        carry_xmat = self.data.site_xmat[kin.grasp_site_id].reshape(3, 3).copy()
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        # The carry pose sits in a different IK basin from the near-belt pick
        # pose.  Starting its batch solve from the known collision-screened
        # home branch avoids a joint-limit local minimum; the complete sweep
        # is still screened before the central planner accepts it.
        q_bin_approach = kin.solve_position_ik(bin_approach, kin.home_qpos, target_xmat=carry_xmat)
        # The tray floor lies below the wrist's collision-free orientation
        # workspace. Release from the reachable approach height and let the
        # free body settle under gravity rather than forcing a wrist flip.
        q_drop = q_bin_approach.copy()
        keyframes = [
                ("prepare", prepare_s, q_prepare, GRIP_OPEN_M),
                ("track", track_s, q_pregrasp, GRIP_OPEN_M),
                ("descend", descend_s, q_pick, GRIP_OPEN_M),
                # Keep force applied long enough for a bilateral pinch to
                # settle before lifting.
                ("close", close_s, q_pick, GRIP_CLOSED_M),
                ("lift", lift_s, q_pregrasp, GRIP_CLOSED_M),
                # Let the real position servo finish the horizontal transfer
                # before the release-window check; the lift remains fast.
                ("to_bin", transfer_s, q_bin_approach, GRIP_CLOSED_M),
                ("lower", lower_s, q_drop, GRIP_CLOSED_M),
                ("open", open_s, q_drop, GRIP_OPEN_M),
                ("settle", settle_s, q_drop, GRIP_OPEN_M),
                ("retreat", retreat_s, q_bin_approach, GRIP_OPEN_M),
                ("home", home_s, q_home, GRIP_OPEN_M),
            ]
        return ArmMission(
            arm,
            object_id,
            placement_zone,
            keyframes,
            intercept_close_s,
            output_slot=output_slot,
            keyframe_started_s=self.data.time,
            next_replan_s=self.data.time + self._tracking_period_s(),
            last_safe_qpos=self.data.qpos[kin.qpos_addresses].copy(),
            assigned_at_s=float(self.data.time),
            trajectory=self._build_trajectory(arm, keyframes),
            stage_start_qpos=self.data.qpos[kin.qpos_addresses].copy(),
            last_pick_xyz=pick_xyz.copy(),
        )

    def _hold_pregrasp_targets(self, mission: ArmMission, hold_qpos: np.ndarray) -> None:
        """Prevent a rejected live replan from executing its stale target."""
        for index, (stage, duration, target, opening) in enumerate(mission.keyframes):
            if index >= mission.keyframe_index and stage in ("track", "descend", "close", "lift"):
                mission.keyframes[index] = (stage, duration, hold_qpos.copy(), opening)
        mission.trajectory = self._build_trajectory(
            mission.arm,
            mission.keyframes[mission.keyframe_index :],
        )

    def _refresh_intercept(self, mission: ArmMission) -> None:
        close_index = next((index for index, frame in enumerate(mission.keyframes) if frame[0] == "close"), -1)
        if mission.failed or self.data.time < mission.next_replan_s or close_index < 0 or mission.keyframe_index > close_index:
            return
        kin = self.kinematics[mission.arm]
        current_stage = mission.keyframes[mission.keyframe_index][0]
        # The final descent/close segment is already predicted against the
        # belt velocity. Re-solving it while the fingers are entering the
        # grasp window can reject the very path that would make contact and
        # leave the arm hovering until the object passes. Dense IK remains
        # active in the high approach/track stages and as a guarded close
        # correction while the moving target is still outside the window.
        if current_stage not in ("prepare", "track", "close"):
            return
        # The overhead approach is a committed collision-screened segment.
        # Do not replace it with a direct current->pick IK path while it is
        # still in progress; that was the source of the guard collision.
        if current_stage == "prepare":
            mission.next_replan_s = self.data.time + self._tracking_period_s()
            return

        # Predict the part at the remaining time until the close stage, not
        # merely one controller period ahead. A 100 ms target while the arm is
        # still 5 seconds from closing overwrites a valid downstream intercept
        # with an upstream, frequently unreachable IK request.
        elapsed = max(0.0, float(self.data.time - mission.keyframe_started_s))
        _, current_duration, *_ = mission.keyframes[mission.keyframe_index]
        remaining_to_close_s = max(0.0, current_duration - elapsed)
        # The Cartesian pick must align at *entry* to ``close``. The close
        # frame's duration is finger travel, not additional conveyor lead.
        remaining_to_close_s += sum(frame[1] for frame in mission.keyframes[mission.keyframe_index + 1 : close_index])
        prediction_horizon_s = max(self._tracking_lead_s(), remaining_to_close_s)
        mission.intercept_close_s = float(self.data.time + remaining_to_close_s)
        pick_xyz = self._grasp_target(
            mission.arm,
            self._predict_part_position(mission.object_id, prediction_horizon_s),
        )
        pick_xyz[2] = max(MIN_PICK_HEIGHT_M, pick_xyz[2])
        if mission.last_pick_xyz is not None and np.linalg.norm(pick_xyz - mission.last_pick_xyz) < TRACKING_IK_MIN_TARGET_DELTA_M:
            mission.next_replan_s = self.data.time + self._tracking_period_s()
            return
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        start = self.data.qpos[kin.qpos_addresses].copy()
        candidates = self._tracking_ik_candidates(mission.arm, pick_xyz, start)
        if not candidates:
            if current_stage == "close":
                # At the final grasp window the arm is already in the lower
                # corridor. A full swept-path search can reject every tiny
                # correction because the standby peer's warning box overlaps
                # the historical route. Accept only a physically safe
                # endpoint correction here, then let measured finger contact
                # decide whether the grasp is real.
                q_pick = kin.solve_resolved_rate_ik(
                    pick_xyz,
                    start,
                    max_iterations=TRACKING_IK_MAX_ITERATIONS,
                )
                if self._pose_is_safe(
                    mission.arm,
                    q_pick,
                    GRIP_OPEN_M,
                    reserve_peer_command=True,
                ):
                    for index, (stage, duration, target, opening) in enumerate(mission.keyframes):
                        if stage == "close":
                            mission.keyframes[index] = (stage, duration, q_pick, opening)
                            break
                    mission.last_pick_xyz = pick_xyz.copy()
                    mission.next_replan_s = self.data.time + self._tracking_period_s()
                    return
            if mission.path_blocked_since_s is None:
                mission.path_blocked_since_s = float(self.data.time)
            mission.next_replan_s = self.data.time + self._tracking_period_s()
            if self.data.time - mission.last_path_reject_log_s >= 0.5:
                mission.last_path_reject_log_s = float(self.data.time)
                self._log("ik_path_reject", object_id=mission.object_id, arm=mission.arm.value, reason="swept_path_collision")
            # A rejected dynamic path must never leave the previous target in
            # the executor.  Hold the last physically screened pose until a
            # new branch is found; otherwise a stale target can drive the
            # palm through a conveyor guard between two IK updates.
            hold_qpos = mission.last_safe_qpos.copy() if mission.last_safe_qpos is not None else start
            self._hold_pregrasp_targets(mission, hold_qpos)
            # The old executor held the last pose but kept its stage clock
            # advancing.  It could therefore enter descend/close with a
            # rejected trajectory and repeatedly drive into the same guard.
            mission.keyframe_started_s += self._tracking_period_s()
            return
        q_pregrasp, q_pick, _ = candidates[0]
        updated = copy.deepcopy(mission)
        for index, (stage, duration, target, opening) in enumerate(updated.keyframes):
            if stage == "prepare" and index == mission.keyframe_index:
                prepare_xyz = self._overhead_approach(mission.arm)
                q_prepare = kin.solve_resolved_rate_ik(prepare_xyz, start, max_iterations=TRACKING_IK_MAX_ITERATIONS)
                updated.keyframes[index] = (stage, duration, q_prepare, opening)
            elif stage == "track":
                updated.keyframes[index] = (stage, duration, q_pregrasp, opening)
            elif stage in ("descend", "close"):
                updated.keyframes[index] = (stage, duration, q_pick, opening)
            elif stage == "lift":
                updated.keyframes[index] = (stage, duration, q_pregrasp, opening)
        updated.trajectory = self._build_trajectory(updated.arm, updated.keyframes)
        if mission.joint_strategy is not None:
            jointly_safe, reason = self._preflight_mission(updated, enforce_warning=False)
            if not jointly_safe:
                mission.next_replan_s = self.data.time + self._tracking_period_s()
                mission.keyframe_started_s += self._tracking_period_s()
                hold_qpos = mission.last_safe_qpos.copy() if mission.last_safe_qpos is not None else start
                self._hold_pregrasp_targets(mission, hold_qpos)
                # A blocked moving-target branch can be evaluated every
                # 80 ms.  Keep the evidence in state, but do not flood the
                # dashboard with identical rejection messages.
                if self.data.time - mission.last_path_reject_log_s >= 0.5:
                    mission.last_path_reject_log_s = float(self.data.time)
                    self._log(
                        "joint_replan_reject",
                        object_id=mission.object_id,
                        arm=mission.arm.value,
                        reason=reason,
                    )
                return
        mission.path_blocked_since_s = None
        mission.keyframes = updated.keyframes
        mission.trajectory = updated.trajectory
        mission.last_pick_xyz = pick_xyz.copy()
        mission.tracking_updates += 1
        mission.next_replan_s = self.data.time + self._tracking_period_s()

    def _grasp_window_ready(self, arm: ArmId, object_id: str) -> bool:
        """Require both measured proximity and belt-speed prediction agreement."""
        kin = self.kinematics[arm]
        object_xyz = self.data.qpos[self.qpos_addresses[object_id] : self.qpos_addresses[object_id] + 3]
        grasp_xyz = kin.grasp_position()
        predicted_xyz = self._predict_part_position(object_id, self._tracking_lead_s())
        measured_xy_error = float(np.linalg.norm(object_xyz[:2] - grasp_xyz[:2]))
        predicted_xy_error = float(np.linalg.norm(predicted_xyz[:2] - grasp_xyz[:2]))
        measured_z_error = abs(float(object_xyz[2] - grasp_xyz[2]))
        return (
            measured_xy_error <= GRASP_XY_TOLERANCE_M
            and predicted_xy_error <= GRASP_PREDICTION_TOLERANCE_M
            and measured_z_error <= GRASP_Z_TOLERANCE_M
        )

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
        # The deadline is the physical tail, not the order in which the
        # feeder happened to create the objects.  Always expose the most
        # urgent intercept first so a deferred task cannot be starved by a
        # newer arrival.
        observations.sort(key=lambda item: (item.deadline_s, item.object_id))
        # Avoid a Braess-like local greedy commitment: when the feed is faster
        # than one service cycle, retain a lone object briefly so the imminent
        # peer can enter the 3 x 3 joint-strategy game with it.
        future_arrivals = [item.spawn_time_s for item in self.items if item.part_name not in self.spawned]
        should_batch = (
            not SINGLE_ARM_VALIDATION_MODE
            and int(self.parameters.feed_batch_size) > 1
            # With a one-item admission limit, the next item intentionally
            # remains at the feeder until this one clears. Waiting here for a
            # peer would therefore leave the only physical item unassigned
            # until it reaches the tail.
            and int(self.parameters.max_active_parts) > 1
            and not self.missions
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
        if int(self.parameters.max_active_parts) == 1 and available:
            # The stability benchmark deliberately admits one moving object
            # at a time. Keep its two equal arms balanced by completed work
            # history, instead of allowing a small instantaneous distance
            # advantage to fill one tray and leave the other arm untested.
            assignment = available[0]
            minimum_attempts = min(self.arm_outcomes[arm]["attempts"] for arm in ArmId)
            least_used = [arm for arm in ArmId if self.arm_outcomes[arm]["attempts"] == minimum_attempts]
            selected_arm = assignment.arm if assignment.arm in least_used else least_used[0]
            if selected_arm is not assignment.arm:
                self.coordinator.assignment_counts[assignment.arm] = max(
                    0,
                    self.coordinator.assignment_counts[assignment.arm] - 1,
                )
                self.coordinator.assignment_counts[selected_arm] += 1
                available[0] = Candidate(
                    selected_arm,
                    assignment.object_id,
                    assignment.object_class,
                    assignment.workspace_zone,
                    "left_bin" if selected_arm is ArmId.A else "right_bin",
                    assignment.interval_s,
                    assignment.score,
                )
        if SINGLE_ARM_VALIDATION_MODE:
            # Keep one executor active so the belt-speed intercept can be
            # judged without joint-route or handoff effects.  The same path
            # is used for A and B; the next arm gets its turn after release.
            if self.missions:
                self.latest_decision = {
                    "assignments": [],
                    "rejected": {},
                    "status": "single_arm_validation_busy",
                }
                return
            candidates = list(self.validation_queue)
            self.validation_queue.clear()
            candidates.extend(available)
            for index, assignment in enumerate(candidates):
                if assignment.arm is not self.validation_next_arm and assignment.object_class is ObjectClass.MIDDLE:
                    # The coordinator may prefer the same arm repeatedly by
                    # cost.  Alternate the baseline executor explicitly so
                    # both physical arms are validated with identical logic.
                    original_arm = assignment.arm
                    assignment = Candidate(
                        self.validation_next_arm,
                        assignment.object_id,
                        assignment.object_class,
                        assignment.workspace_zone,
                        "left_bin" if self.validation_next_arm is ArmId.A else "right_bin",
                        assignment.interval_s,
                        assignment.score,
                    )
                    self.coordinator.assignment_counts[original_arm] = max(
                        0,
                        self.coordinator.assignment_counts[original_arm] - 1,
                    )
                    self.coordinator.assignment_counts[assignment.arm] += 1
                part_y = float(self.data.qpos[self.qpos_addresses[assignment.object_id] + 1])
                if part_y < TAIL_EXIT_Y_M:
                    self.missed.add(assignment.object_id)
                    self.coordinator.mark_completed(assignment.object_id)
                    self._log("missed", object_id=assignment.object_id, reason="tail_exit_while_validation_queued")
                    continue
                if self._start_assignment(assignment):
                    self.validation_queue.extend(candidates[index + 1 :])
                    self.validation_next_arm = ArmId.B if assignment.arm is ArmId.A else ArmId.A
                    self.latest_decision["status"] = "single_arm_validation"
                    return
            self.validation_queue.clear()
            return
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
            # A joint route can be unsafe even when each arm's own route is
            # physically clear.  Do not deadlock both equal peers in that
            # case: start the first assignment that passes its single-arm
            # physical preflight, then prepare the peer in parallel.  The
            # peer will only leave standby after the lead arm's live path is
            # physically clear, preserving the no-collision guarantee while
            # allowing the line to make progress.
            first, second = assignments
            lead_assignment = None
            deferred_assignment = None
            for candidate_lead, candidate_peer in ((first, second), (second, first)):
                probe = self._plan_mission(
                    candidate_lead.arm,
                    candidate_lead.object_id,
                    candidate_lead.placement_zone,
                )
                safe, _ = self._preflight_mission(probe)
                if safe:
                    lead_assignment = candidate_lead
                    deferred_assignment = candidate_peer
                    self.missions[candidate_lead.arm] = probe
                    self.arm_outcomes[candidate_lead.arm]["attempts"] += 1
                    self._log(
                        "assign",
                        object_id=candidate_lead.object_id,
                        arm=candidate_lead.arm.value,
                        placement=candidate_lead.placement_zone,
                        route="single_arm_fallback",
                    )
                    break
            if lead_assignment is None:
                lead_assignment, deferred_assignment = first, second
            self.deferred_assignments[deferred_assignment.arm] = deferred_assignment
            # Bootstrap through the peer's safe standby pose.  It can move
            # while the lead performs its own pick, but never enters a
            # physically colliding handoff path.
            self.handoff_leads[deferred_assignment.arm] = lead_assignment
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

    def _must_yield_warning_space(self, mission: ArmMission) -> bool:
        """Choose one equal peer to yield the 10 cm warning envelope.

        The route game grants the direct lower-corridor entry to one arm and
        labels the other arm ``reserved`` or ``outer``.  At execution time,
        position-servo lag can differ from the ideal preflight timeline; the
        yielding peer therefore treats warning overlap as a temporary hold.
        This remains a centralized decision, not a fixed LEFT/RIGHT rule.
        """
        peer = next((item for arm, item in self.missions.items() if arm is not mission.arm), None)
        if peer is None or mission.joint_strategy is None:
            return False
        peer_stage = peer.keyframes[min(peer.keyframe_index, len(peer.keyframes) - 1)][0]
        # Once the priority arm has a verified bilateral grasp and is lifting
        # the payload away from the belt, the lower shared corridor is being
        # released. Keeping the soft warning barrier active through its full
        # bin-transfer serialized both arms and made the waiting object's
        # intercept expire. Physical contacts remain a hard per-step guard.
        if peer.grasped and peer_stage in {"lift", "to_bin", "lower", "open", "settle", "retreat", "home"}:
            return False
        if mission.route_variant == "direct" and peer.route_variant != "direct":
            return False
        if mission.route_variant != "direct" and peer.route_variant == "direct":
            return True
        # Ties are rare because direct/direct and reserved/reserved are
        # excluded by the joint game. Keep the initial B-side tie break
        # deterministic rather than allowing a simultaneous warning entry.
        return mission.arm is ArmId.B

    def _start_safe_deferred_assignments(self) -> None:
        for arm, assignment in list(self.deferred_assignments.items()):
            handoff_owner = next(
                (
                    mission
                    for mission in self.missions.values()
                    if mission.preparation_only
                    and mission.handoff_lead_assignment is not None
                    and mission.handoff_lead_assignment.object_id == assignment.object_id
                ),
                None,
            )
            if handoff_owner:
                if handoff_owner.lead_started:
                    self.deferred_assignments.pop(arm)
                    self.coordinator.assignment_counts[arm] = max(
                        0,
                        self.coordinator.assignment_counts[arm] - 1,
                    )
                    self._log("handoff_claim", object_id=assignment.object_id, arm=arm.value)
                continue
            part_y = float(self.data.qpos[self.qpos_addresses[assignment.object_id] + 1])
            if part_y < TAIL_EXIT_Y_M:
                self.deferred_assignments.pop(arm)
                self.missed.add(assignment.object_id)
                self.coordinator.mark_completed(assignment.object_id)
                self._log("missed", object_id=assignment.object_id, reason="tail_exit_while_deferred")
                continue
            # A reservation is useful only while its original arm can still
            # reach the moving intercept.  Near the tail, give the other
            # equal-peer arm one chance to take the same object if it is
            # idle.  This changes only the centralized assignment; physical
            # preflight and the live contact guard still decide admission.
            remaining_s = (part_y - TAIL_EXIT_Y_M) / max(0.03, self.parameters.belt_speed_mps)
            peer = ArmId.B if arm is ArmId.A else ArmId.A
            if remaining_s <= max(4.0, self.parameters.fixed_cycle_s * 0.75) and peer not in self.missions:
                transferred = Candidate(
                    peer,
                    assignment.object_id,
                    assignment.object_class,
                    assignment.workspace_zone,
                    "left_bin" if peer is ArmId.A else "right_bin",
                    assignment.interval_s,
                    assignment.score,
                )
                old_count = self.coordinator.assignment_counts[arm]
                self.coordinator.assignment_counts[arm] = max(0, old_count - 1)
                self.coordinator.assignment_counts[peer] += 1
                self.deferred_assignments.pop(arm)
                if self._start_assignment(transferred):
                    self._log("deferred_reassign", object_id=assignment.object_id, from_arm=arm.value, to_arm=peer.value)
                    continue
                # If the peer could not pass its own screen, keep the
                # original reservation alive for the next scheduler tick.
                self.coordinator.assignment_counts[peer] = max(0, self.coordinator.assignment_counts[peer] - 1)
                self.coordinator.assignment_counts[arm] += 1
                self.deferred_assignments[arm] = assignment
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
                lead_probe = self._plan_mission(
                    mission.handoff_lead_assignment.arm,
                    mission.handoff_lead_assignment.object_id,
                    mission.handoff_lead_assignment.placement_zone,
                )
                physically_safe, contact_reason = self._preflight_joint_pair(
                    lead_probe,
                    mission,
                    enforce_warning=False,
                )
                if not physically_safe:
                    if self.data.time - mission.last_safety_hold_s >= 0.5:
                        mission.last_safety_hold_s = self.data.time
                        self._log(
                            "handoff_lead_wait",
                            object_id=mission.handoff_lead_assignment.object_id,
                            arm=arm.value,
                            reason="physical_path_collision",
                            contact=contact_reason,
                        )
                    continue
                lead_assignment = mission.handoff_lead_assignment
                # In the single-arm fallback the lead may already be running
                # while the peer is preparing.  Replacing that mission here
                # would create a second executor for the same object and can
                # make a valid grasp look like a later grip loss.
                existing_lead = self.missions.get(lead_assignment.arm)
                if existing_lead is not None and existing_lead.object_id == lead_assignment.object_id:
                    mission.lead_started = True
                    self._log("handoff_lead_start", object_id=lead_assignment.object_id, arm=lead_assignment.arm.value)
                elif self._start_assignment(lead_assignment):
                    mission.lead_started = True
                    self._log("handoff_lead_start", object_id=lead_assignment.object_id, arm=lead_assignment.arm.value)
                else:
                    continue
        # Recovery can leave both members of a newly injected batch in
        # ``handoff_prepare`` without a predeclared lead.  The former logic
        # only promoted a standby once there was fewer than two missions,
        # so two valid standby arms waited for one another until both moving
        # objects passed the tail.  Break that symmetric deadlock centrally:
        # promote the object with less conveyor time remaining, but only after
        # its full path is physically clear against the peer's measured
        # standby posture.  This is a temporary sequence, not a fixed A/B
        # priority, and the peer is promoted immediately after clearance.
        prepared_without_lead = [
            mission
            for mission in self.missions.values()
            if (
                mission.preparation_only
                and mission.preparation_complete
                and mission.handoff_assignment is not None
                and mission.handoff_lead_assignment is None
            )
        ]
        if len(prepared_without_lead) >= 2:
            def remaining_conveyor_time(mission: ArmMission) -> float:
                part_y = float(self.data.qpos[self.qpos_addresses[mission.object_id] + 1])
                downstream_speed = max(0.03, -float(self.data.qvel[self.part_dof_addresses[mission.object_id] + 1]))
                return max(0.0, (part_y - TAIL_EXIT_Y_M) / downstream_speed)

            leader = min(prepared_without_lead, key=remaining_conveyor_time)
            assignment = leader.handoff_assignment
            probe = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
            safe, reason = self._preflight_mission(probe, enforce_warning=False)
            if safe:
                self.missions[leader.arm] = probe
                self.arm_outcomes[leader.arm]["attempts"] += 1
                self._log(
                    "handoff_deadlock_promote",
                    object_id=assignment.object_id,
                    arm=assignment.arm.value,
                    reason="least_remaining_conveyor_time",
                )
            elif self.data.time - leader.last_safety_hold_s >= 0.5:
                leader.last_safety_hold_s = self.data.time
                self._log(
                    "handoff_deadlock_wait",
                    object_id=assignment.object_id,
                    arm=assignment.arm.value,
                    reason=reason,
                )
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
        escape_qpos = kin.clamp_joint_pose({
            ArmId.A: np.array((-2.1728, -0.2902, -1.8000, 2.5770, 4.0202, -6.1516)),
            ArmId.B: np.array((3.3961, 1.7425, -1.3483, 0.0000, 1.9187, 0.0000)),
        }[assignment.arm])
        rng = np.random.default_rng(self.seed + int(self.data.time * 1000.0) + (1 if assignment.arm is ArmId.A else 2))
        ranges = np.array([self.model.jnt_range[joint_id] for joint_id in kin.joint_ids])
        # The fixed outer high approach is a deterministic, physically clear
        # preparation route while the peer starts a shared-middle pick.  Keep
        # the sampled poses as fallbacks for unusual future geometries.
        # Home is the only universally validated standby pose.  The former
        # random-first search could select a mathematically collision-free
        # pose whose actuator lag later drove both grippers into one another.
        # Try the deterministic home posture first, then the screened outer
        # poses as explicit fallbacks.
        # A standby pose must be visibly and functionally prepared.  Trying
        # home first made the first collision-free candidate win, so the peer
        # often remained parked even though a high outer approach was safe.
        # Prefer the overhead approach, then the outer escape, and use home
        # only as the final geometrical fallback.
        candidates = [base.keyframes[0][2].copy(), escape_qpos, kin.home_qpos.copy()]
        candidates.extend(rng.uniform(ranges[:, 0] + 0.05, ranges[:, 1] - 0.05) for _ in range(96))

        safe_qpos = None
        for qpos in candidates:
            probe = copy.deepcopy(base)
            probe.preparation_only = True
            probe.keyframes = [("handoff_escape", 4.0, qpos.copy(), GRIP_OPEN_M)]
            probe.trajectory = self._build_trajectory(probe.arm, probe.keyframes)
            probe.stage_start_qpos = self.data.qpos[kin.qpos_addresses].copy()
            # The 10 cm envelopes are a warning zone.  At this stage an
            # actual physical contact remains a hard rejection, while a
            # warning-only overlap may proceed to the outer standby pose.
            safe, _ = self._preflight_joint_pair(probe, blocker, enforce_warning=False)
            if safe:
                safe_qpos = qpos.copy()
                break
        if safe_qpos is None:
            self._log("handoff_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="no_safe_standby")
            return False

        # Do not move the standby arm into the low pre-grasp pose yet.  That
        # pose is only safe after the lead has secured its payload; entering it
        # during preparation was the source of forearm-to-gripper contacts.
        # The actual probabilistic creep is screened later by
        # ``_start_handoff_creep`` after the lead grasp event.
        ready_qpos = safe_qpos.copy()
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
        """Begin a short-horizon, probabilistically timed approach in parallel."""
        if standby.handoff_target_qpos is None or standby.arm not in self.missions:
            return
        kin = self.kinematics[standby.arm]
        current = self.data.qpos[kin.qpos_addresses].copy()
        readiness = self.bayesian_game.belief_for("outer", "direct").mean
        duration = 2.0 + 3.0 * (1.0 - readiness)
        pick_xyz = self._grasp_target(
            standby.arm,
            self._predict_part_position(object_id=standby.object_id, horizon_s=duration + self._tracking_lead_s()),
        )
        pick_xyz[2] = PREGRASP_HEIGHT_M
        creep_qpos = kin.solve_position_ik(
            pick_xyz,
            current,
            max_iterations=TRACKING_IK_MAX_ITERATIONS,
        )
        lead = self.missions.get(standby.handoff_lead_assignment.arm) if standby.handoff_lead_assignment is not None else None
        if lead is None:
            return
        selected_probe = None
        # Preserve parallel motion while selecting the furthest physically
        # safe point on the approach path. Warning-box overlap is advisory in
        # this phase; actual MuJoCo geometry contact remains hard forbidden.
        for alpha in np.linspace(0.2, 1.0, 5):
            probe = copy.deepcopy(standby)
            probe.preparation_complete = False
            target = interpolate(current, creep_qpos, float(alpha))
            probe.keyframes = [("handoff_creep", duration, target, GRIP_OPEN_M)]
            probe.keyframe_index = 0
            probe.keyframe_started_s = self.data.time
            probe.trajectory = self._build_trajectory(probe.arm, probe.keyframes)
            probe.stage_start_qpos = current.copy()
            safe, _ = self._preflight_joint_pair(lead, probe, horizon_s=duration, enforce_warning=False)
            if safe:
                selected_probe = probe
            else:
                break
        if selected_probe is None:
            self._log("handoff_creep_hold", object_id=standby.object_id, arm=standby.arm.value, reason="peer_path_screen")
            return
        standby.keyframes = selected_probe.keyframes
        standby.keyframe_index = 0
        standby.keyframe_started_s = self.data.time
        standby.trajectory = selected_probe.trajectory
        standby.stage_start_qpos = current.copy()
        standby.preparation_complete = False
        self._log("handoff_creep", object_id=standby.object_id, arm=standby.arm.value, duration=round(duration, 2))

    def _start_assignment(self, assignment) -> bool:
        mission = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
        safe, reason = self._preflight_mission(mission)
        if not safe:
            # A final-pose failure is not enough to discard the task. Test all
            # route candidates and keep only a complete collision-free sweep.
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
            # A reservation, outer detour or safety hold must never keep an
            # already departed part claimed. Deadline expiry applies to every
            # pre-grasp stage, not only to the final close command.
            if not mission.grasped and not mission.failed:
                part_y = float(self.data.qpos[self.qpos_addresses[mission.object_id] + 1])
                if part_y < TAIL_EXIT_Y_M:
                    self._fail_grasp(arm, mission, "tail_exit_before_grasp", set())
                    continue
            if mission.preparation_only and mission.preparation_complete:
                # Keep the peer in its screened high standby pose while the
                # lead approaches.  Begin the short handoff creep only after
                # the lead has a verified physical grasp; this preserves
                # parallel preparation without sending both grippers into the
                # same low corridor at once.
                lead = None
                if mission.handoff_lead_assignment is not None:
                    lead = self.missions.get(mission.handoff_lead_assignment.arm)
                if (
                    mission.lead_started
                    and lead is not None
                    and lead.grasped
                    and mission.keyframes[-1][0] == "handoff_ready"
                ):
                    self._start_handoff_creep(mission)
                    continue
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
            # Do not close at a stale waypoint.  The object is moving with
            # the belt, so hold the close stage open until the measured pose
            # and the velocity-predicted pose both enter the grasp window.
            if stage == "close" and not self._grasp_window_ready(arm, mission.object_id):
                kin.command_joint_pose(target, GRIP_OPEN_M)
                mission.keyframe_started_s += CONTROL_STEP_S
                if self.data.time - mission.last_safety_hold_s >= 0.5:
                    mission.last_safety_hold_s = self.data.time
                    self._log("grasp_wait", object_id=mission.object_id, arm=arm.value, reason="outside_predicted_grasp_window")
                continue
            # Once a verified payload is horizontally inside its tray, there
            # is no reason to spend another lower/settle cycle before opening
            # the fingers.  Release from the approach height is a physical
            # drop into the tray; MuJoCo still decides whether it lands.
            payload_release_stable = float(np.max(np.abs(self.data.qvel[kin.dof_addresses]))) <= 0.16
            if (
                mission.grasped
                and stage in {"to_bin", "lower"}
                and payload_release_stable
                and self._part_xy_in_target_bin(mission.object_id, mission.placement_zone)
            ):
                open_index = next(
                    (index for index, frame in enumerate(mission.keyframes) if frame[0] == "open"),
                    mission.keyframe_index,
                )
                if open_index > mission.keyframe_index:
                    mission.keyframe_index = open_index
                    mission.keyframe_started_s = self.data.time
                    mission.stage_start_qpos = current.copy()
                    stage, duration, target, opening = mission.keyframes[open_index]
                    elapsed = 0.0
                    self._log("early_release_window", object_id=mission.object_id, placement=mission.placement_zone)
            if stage == "open" and not mission.release_started:
                touching_fingers = self._touching_fingers(arm, mission.object_id)
                part_xyz = self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3]
                tool_delta = part_xyz - kin.grasp_position()
                if float(part_xyz[2]) < 0.04 or float(np.linalg.norm(tool_delta)) > 0.18:
                    self._fail_grasp(arm, mission, "payload_detached_before_release", touching_fingers)
                    continue
                # A part can already be supported by the tray floor just
                # below the fingers. Contact may then transfer from a finger
                # pad to the tray a step before opening, which is a valid
                # release state rather than a dropped payload.
                tray_supported = (
                    float(np.linalg.norm(tool_delta[:2])) <= 0.05
                    and -0.12 <= float(tool_delta[2]) <= -0.035
                )
                settled_in_bin = self._part_is_in_target_bin(mission.object_id, mission.placement_zone) or tray_supported
                drop_window = self._part_xy_in_target_bin(mission.object_id, mission.placement_zone)
                if not settled_in_bin and (not drop_window or not payload_release_stable):
                    # Keep the payload physically pinched until the measured
                    # payload, not merely the commanded wrist, is over the
                    # tray.  Opening here was the direct cause of objects
                    # being released beside the bin after a servo lag.
                    kin.command_joint_pose(target, GRIP_CLOSED_M)
                    kin.set_pad_adhesion(GRIP_HOLD_ADHESION_N)
                    mission.keyframe_started_s += CONTROL_STEP_S
                    if self.data.time - mission.last_safety_hold_s >= 0.5:
                        mission.last_safety_hold_s = self.data.time
                        self._log("release_wait", object_id=mission.object_id, reason="payload_not_over_target_bin")
                    continue
                if touching_fingers != kin.finger_geom_ids and not settled_in_bin and not drop_window:
                    self._fail_grasp(arm, mission, "grip_lost_before_release", touching_fingers)
                    continue
                self._log(
                    "release",
                    object_id=mission.object_id,
                    placement=mission.placement_zone,
                    finger_count=len(touching_fingers),
                    release_mode=(
                        "bilateral_contact"
                        if touching_fingers == kin.finger_geom_ids
                        else ("tray_supported" if settled_in_bin else "drop_window")
                    ),
                    part_xyz=np.round(self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3], 3).tolist(),
                    grasp_xyz=np.round(kin.grasp_position(), 3).tolist(),
                )
                mission.release_started = True
            if stage == "open":
                kin.set_pad_adhesion(0.0)
            require_clearance = stage != "close"
            handoff_warning_guard = mission.preparation_only and stage == "handoff_creep"
            # A 10 cm arm envelope is meaningful near the belt, where both
            # manipulators share volume.  Applying it to the high outer
            # escape/prepare stages trapped a yielding arm at its home pose
            # and made it miss the moving part before it could prepare.
            lower_corridor_stage = stage in {"track", "descend", "close", "lift"}
            warning_guard = handoff_warning_guard or (
                lower_corridor_stage and self._must_yield_warning_space(mission)
            )
            safety_check_due = self.data.time >= mission.next_safety_check_s
            if safety_check_due:
                mission.next_safety_check_s = self.data.time + SAFETY_CHECK_PERIOD_S
            if require_clearance and safety_check_due and not self._pose_is_safe(
                arm,
                current,
                commanded_opening,
                warning_guard,
                reserve_peer_command=True,
                allow_warning_progress=warning_guard,
            ):
                if mission.last_safe_qpos is not None:
                    kin.command_joint_pose(mission.last_safe_qpos, commanded_opening)
                mission.keyframe_started_s += CONTROL_STEP_S
                mission.next_safety_check_s = self.data.time + CONTROL_STEP_S
                continue
            stage_start = mission.stage_start_qpos if mission.stage_start_qpos is not None else current
            commanded_qpos = interpolate(stage_start, target, smoothstep(elapsed / max(duration, CONTROL_STEP_S)))
            if require_clearance and safety_check_due and not self._pose_is_safe(
                arm,
                commanded_qpos,
                commanded_opening,
                warning_guard,
                reserve_peer_command=True,
                allow_warning_progress=warning_guard,
            ):
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
                    # Contact can arrive on adjacent solver steps because the
                    # two slide actuators do not settle identically.  Keep the
                    # jaws physically closed for a short confirmation window
                    # instead of declaring a miss on the first single-pad
                    # observation.
                    if elapsed < duration + 0.80:
                        mission.keyframe_started_s += CONTROL_STEP_S
                        continue
                    touching_fingers = self._touching_fingers(arm, mission.object_id)
                    reason = "no_bilateral_finger_contact" if touching_fingers != kin.finger_geom_ids else "grasp_pose_error"
                    self._fail_grasp(arm, mission, reason, touching_fingers)
                    continue
                # Adhesion is enabled only after bilateral MuJoCo contact has
                # been verified. It remains a force-based free-body grasp,
                # rather than directly writing the part pose or adding a
                # kinematic teleport constraint.
                kin.set_pad_adhesion(GRIP_HOLD_ADHESION_N)
            if not stage_reached:
                continue
            mission.keyframe_index += 1
            mission.keyframe_started_s = self.data.time
            # A target is only a request to MuJoCo's actuators. Starting the
            # next segment from that request while the real arm still lags
            # causes a discontinuous catch-up jump, which can cut across the
            # peer arm's reserved corridor. Always continue from measurement.
            mission.stage_start_qpos = self.data.qpos[kin.qpos_addresses].copy()
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
        object_xyz = self.data.qpos[self.qpos_addresses[mission.object_id] : self.qpos_addresses[mission.object_id] + 3]
        position_error = float(np.linalg.norm(object_xyz - kin.grasp_position()))
        current_xmat = self.data.site_xmat[kin.grasp_site_id].reshape(3, 3)
        orientation_error = float(
            np.linalg.norm(
                0.5 * sum(np.cross(current_xmat[:, index], GRASP_XMAT[:, index]) for index in range(3))
            )
        )
        if (
            touching_fingers == kin.finger_geom_ids
            and position_error <= GRASP_XY_TOLERANCE_M
            and orientation_error <= GRASP_ORIENTATION_TOLERANCE
        ):
            mission.grasped = True
            self.arm_outcomes[arm]["grasped"] += 1
            self._log(
                "grasp",
                object_id=mission.object_id,
                arm=arm.value,
                contact="bilateral_finger_physical",
                finger_count=len(touching_fingers),
                grasp_constraint="verified_transport_attachment",
                position_error_m=round(position_error, 4),
                orientation_error=round(orientation_error, 4),
            )
            return True
        if touching_fingers != kin.finger_geom_ids and mission.grasp_recenter_attempts < 1:
            # A single-pad edge touch is not a grasp yet.  Use the measured
            # payload centre for one guarded close-stage recentering attempt;
            # the next solver step must still report bilateral contact before
            # the mission can enter lift.
            target = self._grasp_target(arm, object_xyz)
            current = self.data.qpos[kin.qpos_addresses].copy()
            candidate = kin.solve_resolved_rate_ik(
                target,
                current,
                max_iterations=12,
                target_xmat=GRASP_XMAT,
            )
            if (
                kin.grasp_position_residual(target, candidate) <= 0.060
                and self._pose_is_safe(arm, candidate, GRIP_OPEN_M, reserve_peer_command=True)
            ):
                for index, (stage, duration, frame_target, opening) in enumerate(mission.keyframes):
                    if index == mission.keyframe_index and stage == "close":
                        mission.keyframes[index] = (stage, duration, candidate, opening)
                        mission.stage_start_qpos = current
                        mission.keyframe_started_s = self.data.time
                        mission.last_pick_xyz = target.copy()
                        mission.grasp_recenter_attempts += 1
                        self._log("grasp_recenter", object_id=mission.object_id, arm=arm.value)
                        break
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
        return self._part_xy_in_target_bin(object_id, placement_zone) and 0.04 <= part_xyz[2] <= 0.18

    def _part_xy_in_target_bin(self, object_id: str, placement_zone: str) -> bool:
        """Return whether a payload is over the usable interior of its tray."""
        drop_site = "left_bin_drop" if placement_zone == "left_bin" else "right_bin_drop"
        floor_geom = "left_tray_floor" if placement_zone == "left_bin" else "right_tray_floor"
        drop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, drop_site)
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom)
        part_xyz = self.data.qpos[self.qpos_addresses[object_id] : self.qpos_addresses[object_id] + 3]
        delta = part_xyz - self.data.site_xpos[drop_site_id]
        # Leave a landing margin for the free body after release.  Releasing
        # at the tray's geometric edge made the payload bounce out even when
        # its centre technically overlapped the floor geom.
        # The floor geom already represents the physical tray interior.  The
        # previous 14 cm X inset rejected parts that had visibly landed on the
        # tray, especially after the wrist IK residual shifted a drop by a few
        # centimetres.  Keep a small edge margin for bounce, but do not shrink
        # the usable tray to a narrow mathematical centre strip.
        half_x, half_y = self.model.geom_size[floor_geom_id, :2] - np.array((0.08, 0.03))
        return abs(float(delta[0])) <= half_x and abs(float(delta[1])) <= half_y

    def step(self) -> None:
        with self.state_lock:
            self._update_belt()
            active_parts = len(self.spawned - self.placed - self.missed)
            effective_capacity = 1 if SINGLE_ARM_VALIDATION_MODE else len(self.items)
            pending = [
                item
                for item in self.items
                if item.part_name not in self.spawned and self.data.time >= item.spawn_time_s
            ]
            if pending:
                # A scheduled group is an atomic workset for the centralized
                # joint planner. Previously a capacity of one could inject
                # only half of a two-object group, then make the other object
                # wait until its original intercept had already passed.
                batch_time_s = min(item.spawn_time_s for item in pending)
                batch = [item for item in pending if abs(item.spawn_time_s - batch_time_s) < 1e-9]
                enough_capacity = active_parts + len(batch) <= effective_capacity
                # Infeed is a conveyor event, not a robot admission event.
                # Handler availability belongs to the centralized scheduler;
                # coupling it here made ``feed_interval_s`` silently wait for
                # a completed grasp.  Only physical active-part capacity may
                # defer the scheduled feed.
                if enough_capacity:
                    for item in batch:
                        place_part(self.data, self.qpos_addresses[item.part_name], item.spawn_xyz)
                        self.spawned.add(item.part_name)
                        active_parts += 1
                        self._log("infeed", object_id=item.part_name, object_class=item.object_class.value)
            self._schedule()
            self._update_missions()
            mujoco.mj_step(self.model, self.data)
            self._sync_attached_payloads()
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
                    self._quarantine_recovered_paths(contacts)

    def _has_available_handler(self, item: DemoItem) -> bool:
        # Feeding is independent from assignment.  A MIDDLE object must be
        # allowed to enter the belt even while both arms are executing; the
        # centralized coordinator owns the later admission/claim decision.
        return True

    def _recover_last_safe_poses(self) -> None:
        for arm, mission in self.missions.items():
            if mission.last_safe_qpos is None:
                continue
            _, _, _, opening = mission.keyframes[mission.keyframe_index]
            kin = self.kinematics[arm]
            # Position-control targets alone do not remove an already-created
            # contact: MuJoCo keeps the penetrated qpos until dynamics resolve
            # it. Restore the last screened state before the next forward pass
            # so a recoverable contact cannot become a permanent safety stop.
            self.data.qpos[kin.qpos_addresses] = mission.last_safe_qpos
            self.data.qvel[kin.dof_addresses] = 0.0
            self.data.qpos[kin.finger_qpos_addresses] = opening
            self.data.qvel[kin.finger_dof_addresses] = 0.0
            kin.command_joint_pose(mission.last_safe_qpos, opening)

    def _abort_unsafe_missions(self, contacts: list[tuple[str, str]]) -> None:
        unsafe_arms = {arm for pair in contacts for arm in (self._arm_for_description(pair[0]), self._arm_for_description(pair[1])) if arm is not None}
        for arm in unsafe_arms:
            mission = self.missions.pop(arm, None)
            if mission is None:
                continue
            kin = self.kinematics[arm]
            # A controller target alone leaves the current penetration in the
            # state vector until physics can resolve it.  Directly restore a
            # screened home state before the final forward pass so recovery
            # cannot become a permanent GUI pause.
            self.data.qpos[kin.qpos_addresses] = kin.home_qpos
            self.data.qvel[kin.dof_addresses] = 0.0
            self.data.qpos[kin.finger_qpos_addresses] = GRIP_OPEN_M
            self.data.qvel[kin.finger_dof_addresses] = 0.0
            kin.command_joint_pose(kin.home_qpos, GRIP_OPEN_M)
            kin.set_pad_adhesion(0.0)
            self.missed.add(mission.object_id)
            self.coordinator.mark_completed(mission.object_id)
            self._log("safety_recover", object_id=mission.object_id, arm=arm.value, reason="abort_and_retract", contact=contacts[0])

    def _quarantine_recovered_paths(self, contacts: list[tuple[str, str]]) -> None:
        """Retire a command path after physical rollback instead of replaying it.

        A successful rollback proves that the last safe pose is usable; it does
        not prove that the mission's next target is usable.  Keeping the old
        target caused an infinite guard-contact/recover loop in the viewer.
        """
        unsafe_arms = {
            arm
            for first, second in contacts
            for arm in (self._arm_for_description(first), self._arm_for_description(second))
            if arm is not None
        }
        for arm in unsafe_arms:
            mission = self.missions.pop(arm, None)
            if mission is None:
                continue
            kin = self.kinematics[arm]
            kin.command_joint_pose(kin.home_qpos, GRIP_OPEN_M)
            kin.set_pad_adhesion(0.0)
            self.missed.add(mission.object_id)
            self.coordinator.mark_completed(mission.object_id)
            # A standby peer must not wait forever for a lead path that has
            # just been revoked.  It may complete its own already-reserved
            # task when its preparation stage finishes.
            for peer in self.missions.values():
                lead = peer.handoff_lead_assignment
                if lead is not None and lead.object_id == mission.object_id:
                    peer.handoff_lead_assignment = None
                    peer.lead_started = True
            self._log(
                "path_abort",
                object_id=mission.object_id,
                arm=arm.value,
                reason="recovered_forbidden_contact",
                contact=contacts[0],
            )

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
