"""Single-Nova5 physical conveyor-line demonstration.

This is intentionally independent from the dual-arm coordinator.  It keeps
the same moving belt and free-body parts, but gives one A arm all assignments.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

from central_coordinator import ArmId
from run_sorting_demo import (
    DemoParameters,
    GRIP_CLOSED_M,
    GRIP_OPEN_M,
    MIN_PICK_HEIGHT_M,
    PREGRASP_HEIGHT_M,
    ArmKinematics,
    make_demo_items,
    place_part,
)
from run_sorting_line import (
    CONVEYOR_LOOP_LENGTH_M,
    PART_NAMES,
    SEGMENT_COUNT,
    TAIL_EXIT_Y_M,
    UPSTREAM_CENTER_Y_M,
    joint_dof_address,
    joint_qpos_address,
    park_part,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "nova5" / "nova5_single_arm_sorting_line.xml"
CONTROL_STEP_S = 0.002
BELT_SPEED_MPS = 0.12
FEED_INTERVAL_S = 7.5


class SingleArmDemo:
    def __init__(self, seed: int) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.model.opt.timestep = CONTROL_STEP_S
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self.parameters = DemoParameters(feed_interval_s=FEED_INTERVAL_S, max_active_parts=1.0, feed_batch_size=1.0)
        self.belt_speed_mps = BELT_SPEED_MPS
        self.feed_interval_s = FEED_INTERVAL_S
        self.simulation_speed = 1.0
        self.paused = False
        self.reset_requested = threading.Event()
        self.viewer_active = False
        self.viewer_launching = False
        self.kin = ArmKinematics(self.model, self.data, arm=ArmId.A)
        self.items = make_demo_items(seed, self.feed_interval_s, 1, (-0.15, 0.15), (1.15, 1.25))
        self.qpos = {name: joint_qpos_address(self.model, name) for name in PART_NAMES}
        self.qvel = {name: joint_dof_address(self.model, name) for name in PART_NAMES}
        self.segment_qpos = [joint_qpos_address(self.model, f"belt_segment_{i:02d}") for i in range(1, SEGMENT_COUNT + 1)]
        self.segment_qvel = [joint_dof_address(self.model, f"belt_segment_{i:02d}") for i in range(1, SEGMENT_COUNT + 1)]
        self.spawned: set[str] = set()
        self.placed: set[str] = set()
        self.missed: set[str] = set()
        self.mission: dict[str, object] | None = None
        self.output_slot = 0
        for name in PART_NAMES:
            park_part(self.data, self.qpos[name], PART_NAMES.index(name))
        self.kin.command_joint_pose(self.kin.home_qpos, GRIP_OPEN_M, initialize_fingers=True)
        self.kin.set_pad_adhesion(0.0)
        self._update_belt()
        mujoco.mj_forward(self.model, self.data)

    def request_reset(self) -> None:
        self.reset_requested.set()

    def reset_if_requested(self) -> bool:
        if not self.reset_requested.is_set():
            return False
        self.reset_requested.clear()
        mujoco.mj_resetData(self.model, self.data)
        self.spawned.clear()
        self.placed.clear()
        self.missed.clear()
        self.mission = None
        self.output_slot = 0
        self.items = make_demo_items(self.seed, self.feed_interval_s, 1, (-0.15, 0.15), (1.15, 1.25))
        for name in PART_NAMES:
            park_part(self.data, self.qpos[name], PART_NAMES.index(name))
        self.kin.command_joint_pose(self.kin.home_qpos, GRIP_OPEN_M, initialize_fingers=True)
        self.kin.set_pad_adhesion(0.0)
        self._update_belt()
        mujoco.mj_forward(self.model, self.data)
        print(f"[reset] single-arm seed={self.seed}")
        return True

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def request_viewer_open(self) -> str:
        return "focused" if self.viewer_active else "already_open"

    def update_settings(self, values: dict[str, object]) -> None:
        if "seed" in values:
            self.seed = int(values["seed"])
        if "belt_speed_mps" in values:
            self.belt_speed_mps = max(0.01, float(values["belt_speed_mps"]))
        if "feed_interval_s" in values:
            self.feed_interval_s = max(0.5, float(values["feed_interval_s"]))
        if "simulation_speed" in values:
            self.simulation_speed = max(0.1, min(4.0, float(values["simulation_speed"])))
        self.parameters.belt_speed_mps = self.belt_speed_mps
        self.parameters.feed_interval_s = self.feed_interval_s
        self.parameters.simulation_speed = self.simulation_speed
        self.request_reset()

    def snapshot(self) -> dict[str, object]:
        mission = {}
        if self.mission is not None:
            frames = self.mission["frames"]
            mission = {
                "A": {
                    "object_id": self.mission["object_id"],
                    "stage": frames[int(self.mission["index"])][0],
                    "placement_zone": "left_bin",
                    "route": "single_arm",
                    "tracking_updates": 0,
                    "preparation_only": False,
                }
            }
        return {
            "seed": self.seed,
            "mode": "single_arm",
            "algorithm": {"id": "single_arm_executor", "name": "Single-Arm Conveyor Executor"},
            "ik_solver": {"id": "qp_rrik", "name": "Box-Constrained QP Resolved-Rate IK"},
            "time_s": round(float(self.data.time), 3),
            "paused": self.paused,
            "viewer": {"active": self.viewer_active, "launching": self.viewer_launching, "requested": False},
            "parameters": asdict(self.parameters),
            "performance": {"control_hz": 500.0, "tracking_ik_hz": 0.0, "safety_prediction_hz": 0.0, "max_viewer_substeps": 8},
            "counts": {"spawned": len(self.spawned), "placed": len(self.placed), "missed": len(self.missed)},
            "feedback": {"active_parts": len(self.spawned - self.placed - self.missed), "cycle_estimate_s": 0.0, "arms": {"A": {"attempts": 1 if self.mission else 0, "grasped": 1 if self.mission and self.mission["grasped"] else 0, "placed": len(self.placed), "cycle_s": 0.0}}},
            "missions": mission,
            "deferred": [],
            "decision": {"assignments": [], "rejected": {}, "status": "single_arm"},
            "preflight": {"status": "clear", "reason": "single_arm_workspace"},
            "joint_plan": {"status": "single_arm"},
            "safety": {"warning_margin_m": 0.10, "mode": "single_arm_physical_contacts"},
            "events": [],
        }

    def _update_belt(self) -> None:
        travelled = (self.belt_speed_mps * self.data.time) % CONVEYOR_LOOP_LENGTH_M
        pitch = CONVEYOR_LOOP_LENGTH_M / SEGMENT_COUNT
        for index, (qpos_address, qvel_address) in enumerate(zip(self.segment_qpos, self.segment_qvel)):
            self.data.qpos[qpos_address] = UPSTREAM_CENTER_Y_M - ((index * pitch + travelled) % CONVEYOR_LOOP_LENGTH_M)
            self.data.qvel[qvel_address] = -self.belt_speed_mps

    def _predict(self, object_id: str, horizon: float) -> np.ndarray:
        xyz = self.data.qpos[self.qpos[object_id] : self.qpos[object_id] + 3].copy()
        xyz[1] += -self.belt_speed_mps * horizon
        return xyz

    def _plan(self, object_id: str) -> None:
        prepare_s, track_s, descend_s, close_s = 1.2, 1.0, 2.4, 0.7
        close_at = prepare_s + track_s + descend_s + close_s
        prepare_xyz = self._predict(object_id, prepare_s)
        prepare_xyz[2] = 0.56
        pick_xyz = self._predict(object_id, close_at)
        pick_xyz[2] = max(MIN_PICK_HEIGHT_M, pick_xyz[2])
        pregrasp = pick_xyz.copy()
        pregrasp[2] = PREGRASP_HEIGHT_M
        drop_site = self.model.site("left_bin_drop").id
        drop = self.data.site_xpos[drop_site].copy()
        column = self.output_slot % 2
        row = self.output_slot // 2
        self.output_slot += 1
        drop[:2] += ((column - 0.5) * 0.16, (row - 1.0) * 0.14)
        drop[2] = 0.30
        approach = drop.copy()
        approach[2] = 0.46
        q_home = self.kin.home_qpos.copy()
        q_prepare = self.kin.solve_position_ik(prepare_xyz, q_home)
        q_pregrasp = self.kin.solve_position_ik(pregrasp, q_prepare)
        q_pick = self.kin.solve_position_ik(pick_xyz, q_pregrasp)
        q_approach = self.kin.solve_position_ik(approach, q_pregrasp)
        q_drop = self.kin.solve_position_ik(drop, q_approach)
        self.mission = {
            "object_id": object_id,
            "frames": [
                ("prepare", prepare_s, q_prepare, GRIP_OPEN_M),
                ("track", track_s, q_pregrasp, GRIP_OPEN_M),
                ("descend", descend_s, q_pick, GRIP_OPEN_M),
                ("close", close_s, q_pick, GRIP_CLOSED_M),
                ("lift", 2.2, q_pregrasp, GRIP_CLOSED_M),
                ("to_bin", 2.0, q_approach, GRIP_CLOSED_M),
                ("lower", 0.9, q_drop, GRIP_CLOSED_M),
                ("open", 0.5, q_drop, GRIP_OPEN_M),
                ("home", 1.2, q_home, GRIP_OPEN_M),
            ],
            "index": 0,
            "started": self.data.time,
            "stage_started": self.data.time,
            "stage_start_q": self.data.qpos[self.kin.qpos_addresses].copy(),
            "grasped": False,
        }
        print(f"[{self.data.time:5.2f}s] ASSIGN {object_id}")

    def _update_mission(self) -> None:
        if self.mission is None:
            return
        mission = self.mission
        object_id = str(mission["object_id"])
        frames = mission["frames"]
        index = int(mission["index"])
        stage, duration, target, opening = frames[index]
        elapsed = self.data.time - float(mission["stage_started"])
        ratio = min(1.0, max(0.0, elapsed / max(float(duration), CONTROL_STEP_S)))
        smooth = ratio * ratio * (3.0 - 2.0 * ratio)
        commanded_opening = GRIP_OPEN_M * (1.0 - smooth) if stage == "close" else opening
        # Interpolate from the fixed stage-start pose.  Reusing the current
        # pose here reapplies the same displacement every control tick and
        # causes the visible creeping/wrist rotation.
        stage_start = np.asarray(mission["stage_start_q"])
        commanded = stage_start + (np.asarray(target) - stage_start) * smooth
        self.kin.command_joint_pose(commanded, commanded_opening)
        if stage == "close" and elapsed >= float(duration) and not mission["grasped"]:
            touching = self._touching_fingers(object_id)
            if touching == self.kin.finger_geom_ids:
                mission["grasped"] = True
                self.kin.set_pad_adhesion(20.0)
                print(f"[{self.data.time:5.2f}s] GRASP {object_id}")
            else:
                self.missed.add(object_id)
                self.mission = None
                print(f"[{self.data.time:5.2f}s] MISSED {object_id}")
                return
        if elapsed < float(duration):
            return
        if stage == "open":
            self.kin.set_pad_adhesion(0.0)
            self.placed.add(object_id)
            print(f"[{self.data.time:5.2f}s] PLACE {object_id}")
        if index + 1 >= len(frames):
            self.mission = None
            return
        mission["index"] = index + 1
        mission["stage_started"] = self.data.time
        mission["stage_start_q"] = np.asarray(target).copy()

    def _touching_fingers(self, object_id: str) -> set[int]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        first = self.model.body_geomadr[body_id]
        parts = set(range(first, first + self.model.body_geomnum[body_id]))
        touching: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {contact.geom1, contact.geom2}.intersection(parts):
                touching.update(self.kin.finger_geom_ids.intersection({contact.geom1, contact.geom2}))
        return touching

    def step(self) -> None:
        self._update_belt()
        for item in self.items:
            if item.part_name not in self.spawned and self.data.time >= item.spawn_time_s:
                place_part(self.data, self.qpos[item.part_name], item.spawn_xyz)
                self.spawned.add(item.part_name)
                print(f"[{self.data.time:5.2f}s] INFEED {item.part_name}")
        if self.mission is None:
            available = [item for item in self.items if item.part_name in self.spawned and item.part_name not in self.placed and item.part_name not in self.missed]
            available = [item for item in available if self.data.qpos[self.qpos[item.part_name] + 1] > TAIL_EXIT_Y_M]
            if available:
                self._plan(available[0].part_name)
        self._update_mission()
        mujoco.mj_step(self.model, self.data)

    def run(self, duration: float, dashboard_url: str | None = None) -> None:
        if dashboard_url:
            webbrowser.open(dashboard_url)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last = time.perf_counter()
            accumulated = 0.0
            self.viewer_active = True
            while viewer.is_running() and self.data.time < duration:
                self.reset_if_requested()
                now = time.perf_counter()
                accumulated += min(now - last, 0.05) * self.simulation_speed
                last = now
                for _ in range(8):
                    if self.paused or accumulated < CONTROL_STEP_S:
                        break
                    self.step()
                    accumulated -= CONTROL_STEP_S
                viewer.sync()
                time.sleep(0.001)
            self.viewer_active = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-arm Nova5 sorting-line demonstration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    from demo_dashboard import start_dashboard

    demo = SingleArmDemo(args.seed)
    dashboard = start_dashboard(demo)
    try:
        demo.run(args.duration, dashboard.url)
    finally:
        dashboard.stop()


if __name__ == "__main__":
    main()
