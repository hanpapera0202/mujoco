from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np

from central_coordinator import ArmId, Candidate, ObjectClass
from run_sorting_demo import JOINT_RESERVATION_HOLD_S, SortingDemo, place_part


class DynamicInterceptTests(unittest.TestCase):
    def test_arm_a_physically_grasps_a_moving_middle_object(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        item = demo.items[0]
        place_part(demo.data, demo.qpos_addresses[item.part_name], item.spawn_xyz)
        demo.spawned.add(item.part_name)
        mujoco.mj_forward(demo.model, demo.data)
        assignment = Candidate(
            ArmId.A,
            item.part_name,
            ObjectClass.MIDDLE,
            "shared_middle",
            "left_bin",
            (0.0, 10.0),
            1.0,
        )
        with redirect_stdout(StringIO()):
            demo._start_assignment(assignment)
            stages = [stage for stage, *_ in demo.missions[ArmId.A].keyframes]
            self.assertEqual(stages[:4], ["prepare", "track", "descend", "close"])
            while demo.data.time < 8.0 and not demo.paused:
                demo._update_belt()
                demo._update_missions()
                mujoco.mj_forward(demo.model, demo.data)
                mujoco.mj_step(demo.model, demo.data)
        grasps = [event for event in demo.event_log if event["event"] == "grasp"]
        self.assertEqual([(event["arm"], event["finger_count"]) for event in grasps], [("A", 2)])
        # Remaining-time prediction only changes the route when the moving
        # target materially shifts, rather than overwriting it every 80 ms.
        self.assertGreaterEqual(demo.missions[ArmId.A].tracking_updates, 5)

    def test_executor_rejection_rolls_back_assignment_fairness(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        assignment = Candidate(ArmId.A, "part_01", ObjectClass.MIDDLE, "shared_middle", "left_bin", (0.0, 10.0), 1.0)
        demo.coordinator.assignment_counts[ArmId.A] = 1
        demo._release_unstarted_assignment(assignment)
        self.assertEqual(demo.coordinator.assignment_counts[ArmId.A], 0)

    def test_recovered_guard_contact_retires_the_old_path(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        mission = demo._plan_mission(ArmId.A, "part_01", "left_bin")
        demo.missions[ArmId.A] = mission
        with redirect_stdout(StringIO()):
            demo._quarantine_recovered_paths([("conveyor_right_guard", "A_right_finger_pad")])
        self.assertNotIn(ArmId.A, demo.missions)
        self.assertIn("part_01", demo.missed)
        self.assertEqual(demo.event_log[-1]["event"], "path_abort")

    def test_joint_plan_rejects_an_unsafe_live_replan_before_contact(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        with redirect_stdout(StringIO()):
            while demo.data.time < 12.0 and not demo.paused:
                demo.step()
        events = [event["event"] for event in demo.event_log]
        self.assertIn("joint_plan", events)
        self.assertNotIn("safety_stop", events)

    def test_reachable_output_transfer_preserves_the_verified_pick_orientation(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        for arm, object_id, placement in (
            (ArmId.A, "part_01", "left_bin"),
            (ArmId.B, "part_02", "right_bin"),
        ):
            mission = demo._plan_mission(arm, object_id, placement)
            targets = {stage: qpos for stage, _, qpos, _ in mission.keyframes}
            kin = demo.kinematics[arm]
            orientations = {}
            for stage in ("close", "to_bin"):
                demo.data.qpos[kin.qpos_addresses] = targets[stage]
                mujoco.mj_forward(demo.model, demo.data)
                orientations[stage] = demo.data.site_xmat[kin.grasp_site_id].reshape(3, 3).copy()
            self.assertLess(np.linalg.norm(orientations["to_bin"] - orientations["close"]), 0.05)
            np.testing.assert_allclose(targets["lower"], targets["to_bin"])

    def test_tail_exit_during_any_pregrasp_stage_releases_the_stale_assignment(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        base = demo._plan_mission(ArmId.A, "part_01", "left_bin")
        mission = next(route for route in demo._route_candidates(base) if route.route_variant == "outer")
        self.assertEqual(mission.keyframes[0][0], "outer_escape")
        mission.keyframe_started_s = demo.data.time
        mission.stage_start_qpos = demo.data.qpos[demo.kinematics[ArmId.A].qpos_addresses].copy()
        demo.missions[ArmId.A] = mission
        demo.data.qpos[demo.qpos_addresses["part_01"] + 1] = -3.0
        mujoco.mj_forward(demo.model, demo.data)
        with redirect_stdout(StringIO()):
            demo._update_missions()
            demo._update_missions()
        self.assertIn("part_01", demo.missed)
        self.assertTrue(mission.failed)
        self.assertEqual(
            [event["event"] for event in demo.event_log].count("missed"),
            1,
        )

    def test_outer_escape_pose_is_within_the_physical_joint_limits(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        for arm, object_id, placement in (
            (ArmId.A, "part_01", "left_bin"),
            (ArmId.B, "part_02", "right_bin"),
        ):
            mission = demo._plan_mission(arm, object_id, placement)
            outer = next(route for route in demo._route_candidates(mission) if route.route_variant == "outer")
            target = outer.keyframes[0][2]
            joint_ranges = np.array([demo.model.jnt_range[joint_id] for joint_id in demo.kinematics[arm].joint_ids])
            self.assertTrue(np.all(target >= joint_ranges[:, 0] + 0.019))
            self.assertTrue(np.all(target <= joint_ranges[:, 1] - 0.019))

    def test_reserved_route_repredicts_the_moving_part_without_consuming_another_bin_slot(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        mission = demo._plan_mission(ArmId.A, "part_01", "left_bin")
        offsets_before = dict(demo.output_offsets)
        routes = {route.route_variant: route for route in demo._route_candidates(mission)}
        direct = routes["direct"]
        reserved = routes["reserved"]
        direct_pick = next(qpos for stage, _, qpos, _ in direct.keyframes if stage == "descend")
        reserved_pick = next(qpos for stage, _, qpos, _ in reserved.keyframes if stage == "descend")
        self.assertGreater(np.linalg.norm(reserved_pick - direct_pick), 0.02)
        self.assertAlmostEqual(reserved.intercept_close_s - direct.intercept_close_s, JOINT_RESERVATION_HOLD_S)
        self.assertEqual(demo.output_offsets, offsets_before)
        self.assertEqual(reserved.output_slot, mission.output_slot)

    def test_mutual_corridor_reservation_is_rejected_by_the_joint_game(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        first = demo._plan_mission(ArmId.A, "part_01", "left_bin")
        second = demo._plan_mission(ArmId.B, "part_02", "right_bin")
        routes_a = {route.route_variant: route for route in demo._route_candidates(first)}
        routes_b = {route.route_variant: route for route in demo._route_candidates(second)}
        evidence = demo._joint_evidence(routes_a["reserved"], routes_b["reserved"])
        self.assertFalse(evidence.collision_free)
        self.assertEqual(evidence.rejection_reason, "mutual_shared_corridor_reservation")

    def test_unsequenced_direct_entries_are_rejected_by_the_joint_game(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=101)
        first = demo._plan_mission(ArmId.A, "part_01", "left_bin")
        second = demo._plan_mission(ArmId.B, "part_02", "right_bin")
        routes_a = {route.route_variant: route for route in demo._route_candidates(first)}
        routes_b = {route.route_variant: route for route in demo._route_candidates(second)}
        evidence = demo._joint_evidence(routes_a["direct"], routes_b["direct"])
        self.assertFalse(evidence.collision_free)
        self.assertEqual(evidence.rejection_reason, "unsequenced_shared_corridor_entry")

    def test_seed_101_places_the_initial_pair_without_a_safety_stop(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=101)
        with redirect_stdout(StringIO()):
            while demo.data.time < 30.0 and not demo.paused:
                demo.step()
        self.assertFalse(demo.paused)
        self.assertTrue({"part_01", "part_02"}.issubset(demo.placed))
        events = [event["event"] for event in demo.event_log]
        self.assertNotIn("safety_stop", events)
        self.assertNotIn("path_abort", events)

    def test_deferred_peer_reaches_safe_standby_without_marking_missed(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        item = demo.items[1]
        place_part(demo.data, demo.qpos_addresses[item.part_name], item.spawn_xyz)
        demo.spawned.add(item.part_name)
        mujoco.mj_forward(demo.model, demo.data)
        assignment = Candidate(
            ArmId.B,
            item.part_name,
            ObjectClass.MIDDLE,
            "shared_middle",
            "right_bin",
            (0.0, 10.0),
            1.0,
        )
        with redirect_stdout(StringIO()):
            self.assertTrue(demo._start_handoff_preparation(assignment))
            while demo.data.time < 5.8:
                demo._update_belt()
                demo._update_missions()
                mujoco.mj_forward(demo.model, demo.data)
                mujoco.mj_step(demo.model, demo.data)
        mission = demo.missions[ArmId.B]
        self.assertTrue(mission.preparation_only)
        self.assertTrue(mission.preparation_complete)
        self.assertEqual(demo.missed, set())
        self.assertEqual(mission.keyframes[mission.keyframe_index][0], "handoff_ready")


if __name__ == "__main__":
    unittest.main()
