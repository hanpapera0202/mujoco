"""Single-seed dual-Nova5 sorting demonstration driven by the central coordinator.

The demo uses position IK for arm motion and a kinematic attachment after the
gripper closes.  It is intentionally a stable integration demo; pure contact
grasping is a later physics-validation stage.
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
PICK_HEIGHT_M = 0.23
PREGRASP_HEIGHT_M = 0.42
BIN_APPROACH_HEIGHT_M = 0.46
BIN_DROP_HEIGHT_M = 0.19
GRASP_XY_TOLERANCE_M = 0.055
EXPECTED_OBJECT_BELOW_GRASP_M = 0.110
GRASP_Z_TOLERANCE_M = 0.030
CENTRAL_CORRIDOR_BLOCKING_STAGES = {"approach", "descend", "close", "lift"}


@dataclass
class DemoParameters:
    horizon_s: float = 5.0
    parallel_bonus: float = 2.0
    pick_speed_mps: float = 0.55
    fixed_cycle_s: float = 1.1
    urgency_weight: float = 3.0
    success_weight: float = 2.0
    travel_weight: float = 0.25
    belt_speed_mps: float = BELT_SPEED_MPS
    simulation_speed: float = 1.0


CSPR_ALGORITHM_ID = "cspr"
CSPR_ALGORITHM_NAME = "CSPR - Centralized Spatiotemporal Reservation"


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
    attachment_local_xyz: np.ndarray | None = None
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
        self.home_qpos = data.qpos[self.qpos_addresses].copy()

    def tool_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site_id].copy()

    def grasp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.grasp_site_id].copy()

    def solve_position_ik(self, target_xyz: np.ndarray, start_qpos: np.ndarray) -> np.ndarray:
        """Damped least-squares position IK, bounded by each Nova5 joint range."""
        saved_qpos = self.data.qpos.copy()
        self.data.qpos[self.qpos_addresses] = start_qpos
        for _ in range(260):
            mujoco.mj_forward(self.model, self.data)
            error = target_xyz - self.grasp_position()
            if np.linalg.norm(error) < 0.012:
                break
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.grasp_site_id)
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
        DemoItem("part_02", ObjectClass.RIGHT, 0.2, (0.28 + jitter(0.03), 0.92, 0.13), 4.75),
        DemoItem("part_03", ObjectClass.LEFT, 0.2, (-0.28 + jitter(0.03), 1.22, 0.13), 4.85),
    ]


def interpolate(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    ratio = min(1.0, max(0.0, ratio))
    return first + (second - first) * ratio


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
        self.event_log: list[dict[str, object]] = []
        self._reset_state()

    def _reset_state(self) -> None:
        """Restore the exact seed scenario without replacing the viewer's MjData."""
        mujoco.mj_resetData(self.model, self.data)
        self.items = make_demo_items(self.seed)
        self.by_name = {item.part_name: item for item in self.items}
        self.qpos_addresses = {name: joint_qpos_address(self.model, name) for name in PART_NAMES}
        self.segment_qpos_addresses = [joint_qpos_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.segment_dof_addresses = [joint_dof_address(self.model, f"belt_segment_{index:02d}") for index in range(1, SEGMENT_COUNT + 1)]
        self.kinematics = {arm: ArmKinematics(self.model, self.data, arm) for arm in ArmId}
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
        self.locked_positions: dict[str, np.ndarray] = {}
        self.last_schedule_s = -SCHEDULER_PERIOD_S
        self.output_offsets = {"left_bin": 0, "right_bin": 0}
        self.latest_decision = {"assignments": [], "rejected": {}}
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
            if any(mission.object_id == name for mission in self.missions.values()):
                continue
            xyz = self.data.qpos[self.qpos_addresses[name] : self.qpos_addresses[name] + 3]
            if xyz[1] < TAIL_EXIT_Y_M:
                self.missed.add(name)
                self._log("missed", object_id=name, reason="tail_exit")
                continue
            remaining = min(item.deadline_s, max(0.1, (xyz[1] - TAIL_EXIT_Y_M) / self.parameters.belt_speed_mps))
            observations.append(ObjectObservation(name, item.object_class, tuple(xyz), remaining, {ArmId.A: 0.93, ArmId.B: 0.93}))
        return observations

    def _plan_mission(self, arm: ArmId, object_id: str, placement_zone: str) -> ArmMission:
        kin = self.kinematics[arm]
        qpos_address = self.qpos_addresses[object_id]
        pick_xyz = self.data.qpos[qpos_address : qpos_address + 3].copy()
        # The part continues moving while the arm approaches, descends, and closes.
        # Aim at this predicted belt position instead of the stale observation.
        time_to_close_s = 1.15 + 0.70 + 0.40
        pick_xyz[1] -= self.parameters.belt_speed_mps * time_to_close_s
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
            if self._may_enter_assignment(assignment):
                self._start_assignment(assignment)
            else:
                self.deferred_assignments[assignment.arm] = assignment
                self._log("reserve_wait", object_id=assignment.object_id, arm=assignment.arm.value, reason="central_corridor")

    def _may_enter_assignment(self, assignment) -> bool:
        if assignment.object_class is not ObjectClass.MIDDLE:
            return True
        return all(
            mission.keyframes[mission.keyframe_index][0] not in CENTRAL_CORRIDOR_BLOCKING_STAGES
            for arm, mission in self.missions.items()
            if arm is not assignment.arm
        )

    def _start_safe_deferred_assignments(self) -> None:
        for arm, assignment in list(self.deferred_assignments.items()):
            if arm not in self.missions and self._may_enter_assignment(assignment):
                self.deferred_assignments.pop(arm)
                self._start_assignment(assignment)

    def _start_assignment(self, assignment) -> None:
        mission = self._plan_mission(assignment.arm, assignment.object_id, assignment.placement_zone)
        self.missions[assignment.arm] = mission
        self._log("assign", object_id=assignment.object_id, arm=assignment.arm.value, placement=assignment.placement_zone)

    def _update_missions(self) -> None:
        for arm, mission in list(self.missions.items()):
            kin = self.kinematics[arm]
            stage, duration, target, opening = mission.keyframes[mission.keyframe_index]
            elapsed = self.data.time - mission.keyframe_started_s
            current = self.data.qpos[kin.qpos_addresses].copy()
            kin.set_joint_pose(interpolate(current, target, min(1.0, CONTROL_STEP_S / max(duration - elapsed, CONTROL_STEP_S))), opening)
            if stage == "close" and elapsed >= duration and not mission.attached:
                if not self._confirm_grasp(arm, mission):
                    continue
            if mission.attached:
                object_address = self.qpos_addresses[mission.object_id]
                grasp_xyz = kin.grasp_position()
                grasp_xmat = self.data.site_xmat[kin.grasp_site_id].reshape(3, 3)
                object_xyz = grasp_xyz + grasp_xmat @ mission.attachment_local_xyz
                self.data.qpos[object_address : object_address + 7] = (*object_xyz, 1.0, 0.0, 0.0, 0.0)
                self.data.qvel[object_address : object_address + 6] = 0.0
            if elapsed < duration:
                continue
            if stage == "open" and mission.attached:
                mission.attached = False
                object_address = self.qpos_addresses[mission.object_id]
                grasp_xyz = kin.grasp_position()
                self.locked_positions[mission.object_id] = grasp_xyz.copy()
                self.data.qpos[object_address : object_address + 7] = (*grasp_xyz, 1.0, 0.0, 0.0, 0.0)
                self._log("place", object_id=mission.object_id, placement=mission.placement_zone)
            mission.keyframe_index += 1
            mission.keyframe_started_s = self.data.time
            if mission.done:
                self.missions.pop(arm)
                if not mission.failed:
                    self.placed.add(mission.object_id)
                self.coordinator.mark_completed(mission.object_id)

    def _confirm_grasp(self, arm: ArmId, mission: ArmMission) -> bool:
        """Attach only when the closing gripper physically reaches the free part."""
        kin = self.kinematics[arm]
        object_address = self.qpos_addresses[mission.object_id]
        object_xyz = self.data.qpos[object_address : object_address + 3].copy()
        grasp_xyz = kin.grasp_position()
        delta = object_xyz - grasp_xyz
        xy_error = float(np.linalg.norm(delta[:2]))
        z_alignment_error = abs(float(delta[2]) + EXPECTED_OBJECT_BELOW_GRASP_M)
        if xy_error <= GRASP_XY_TOLERANCE_M and z_alignment_error <= GRASP_Z_TOLERANCE_M:
            grasp_xmat = self.data.site_xmat[kin.grasp_site_id].reshape(3, 3)
            mission.attachment_local_xyz = grasp_xmat.T @ delta
            mission.attached = True
            self._log("grasp", object_id=mission.object_id, arm=arm.value, xy_error_m=round(xy_error, 4))
            return True
        self.missed.add(mission.object_id)
        self.coordinator.mark_completed(mission.object_id)
        self._log(
            "missed",
            object_id=mission.object_id,
            reason="grasp_alignment",
            xy_error_m=round(xy_error, 4),
            z_alignment_error_m=round(z_alignment_error, 4),
        )
        mission.failed = True
        mission.keyframes = [("recover", 0.6, self.data.qpos[kin.qpos_addresses].copy(), 0.035), ("home", 1.0, kin.home_qpos, 0.035)]
        mission.keyframe_index = 0
        mission.keyframe_started_s = self.data.time
        return False

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
            for name, xyz in self.locked_positions.items():
                address = self.qpos_addresses[name]
                self.data.qpos[address : address + 7] = (*xyz, 1.0, 0.0, 0.0, 0.0)
                self.data.qvel[address : address + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            contacts = self._inter_arm_contacts()
            if contacts and not self.paused:
                self.paused = True
                self._log("safety_stop", reason="inter_arm_contact", contact=contacts[0])

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
