from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from central_coordinator import ArmId, ObjectClass
from run_sorting_demo import SortingDemo, make_demo_items


class NarrowMiddleScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "models" / "nova5" / "nova5_sorting_line.xml"
        cls.demo = SortingDemo(model_path, seed=42)

    def test_conveyor_effective_width_is_45_cm(self):
        geom_id = self.demo.model.geom("belt_segment_01_geom").id
        self.assertAlmostEqual(float(self.demo.model.geom_size[geom_id, 0]) * 2.0, 0.45)

    def test_all_seed_objects_are_shared_middle_work(self):
        self.assertEqual(len(self.demo.items), 10)
        self.assertTrue(all(item.object_class is ObjectClass.MIDDLE for item in self.demo.items))
        first, second = self.demo.items[:2]
        self.assertEqual(first.spawn_time_s, second.spawn_time_s)
        self.assertAlmostEqual(first.spawn_xyz[1], second.spawn_xyz[1])

    def test_feed_interval_controls_every_scheduled_release(self):
        release_times = [item.spawn_time_s for item in self.demo.items]
        expected = [0.2 + (index // int(self.demo.parameters.feed_batch_size)) * self.demo.parameters.feed_interval_s for index in range(10)]
        np.testing.assert_allclose(release_times, expected)

    def test_default_feed_releases_two_items_per_batch(self):
        release_times = [item.spawn_time_s for item in self.demo.items]
        self.assertEqual(release_times[0], release_times[1])
        self.assertEqual(release_times[2], release_times[3])
        self.assertEqual(int(self.demo.parameters.feed_batch_size), 2)

    def test_seeded_feed_uses_both_sides_of_the_middle_lane(self):
        lateral = [item.spawn_xyz[0] for item in self.demo.items]
        self.assertLess(lateral[0], -0.04)
        self.assertGreater(lateral[1], 0.04)
        self.assertTrue(all(0.04 <= abs(value) <= self.demo.parameters.feed_lateral_spread_m for value in lateral))
        replay = make_demo_items(
            self.demo.seed,
            self.demo.parameters.feed_interval_s,
            int(self.demo.parameters.feed_batch_size),
            self.demo.parameters.feed_lateral_spread_m,
        )
        np.testing.assert_allclose([item.spawn_xyz for item in replay], [item.spawn_xyz for item in self.demo.items])

    def test_robot_bases_are_closer_to_narrow_line(self):
        first = self.demo.model.body_pos[self.demo.model.body("robot_A_base").id]
        second = self.demo.model.body_pos[self.demo.model.body("robot_B_base").id]
        np.testing.assert_allclose(first, (-0.58, -0.08, 0.0))
        np.testing.assert_allclose(second, (0.58, -0.08, 0.0))

    def test_each_warning_box_expands_its_collision_box_by_10_cm(self):
        for arm in ArmId:
            for region in ("upper_arm", "forearm", "gripper"):
                collision = self.demo.model.geom(f"{arm.value}_{region}_collision").size
                warning = self.demo.model.geom(f"{arm.value}_{region}_warning").size
                np.testing.assert_allclose(warning - collision, (self.demo.parameters.warning_margin_m,) * 3)

    def test_warning_margin_can_be_changed_without_resizing_physical_boxes(self):
        model_path = ROOT / "models" / "nova5" / "nova5_sorting_line.xml"
        demo = SortingDemo(model_path, seed=42)
        collision_before = demo.model.geom("A_forearm_collision").size.copy()
        demo.update_settings({"warning_margin_m": 0.05})
        self.assertTrue(demo.reset_if_requested())
        np.testing.assert_allclose(demo.model.geom("A_forearm_collision").size, collision_before)
        np.testing.assert_allclose(
            demo.model.geom("A_forearm_warning").size - collision_before,
            (0.05, 0.05, 0.05),
        )

    def test_home_pose_has_no_warning_envelope_overlap(self):
        self.assertEqual(self.demo._warning_envelope_overlaps(self.demo.data), [])


if __name__ == "__main__":
    unittest.main()
