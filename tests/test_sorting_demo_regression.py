from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_sorting_demo import ArmId, DemoParameters, SortingDemo


class SortingDemoPhysicalRegressionTests(unittest.TestCase):
    """Protect the first complete seed-42 all-MIDDLE physical cycle."""

    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "models" / "nova5" / "nova5_sorting_line.xml"
        cls.demo = SortingDemo(model_path, seed=42, parameters=DemoParameters(feed_batch_size=1.0))
        with redirect_stdout(StringIO()):
            while cls.demo.data.time < 20.75 and not cls.demo.paused:
                cls.demo.step()
        cls.events = cls.demo.event_log

    def test_first_object_is_centrally_assigned_by_continuous_cost(self):
        assignments = [event for event in self.events if event["event"] == "assign"]
        self.assertEqual((assignments[0]["arm"], assignments[0]["object_id"]), ("B", "part_01"))
        self.assertTrue(all(item.object_class.value == "middle" for item in self.demo.items))

    def test_first_object_requires_bilateral_physical_contact(self):
        grasps = [event for event in self.events if event["event"] == "grasp"]
        self.assertEqual({event["object_id"] for event in grasps}, {"part_01"})
        for event in grasps:
            self.assertEqual(event["finger_count"], 2)
            self.assertEqual(event["contact"], "bilateral_finger_physical")
            self.assertEqual(event["grasp_constraint"], "verified_transport_attachment")

    def test_no_grasp_constraint_or_safety_failure_is_hidden(self):
        self.assertEqual(self.demo.model.neq, 0)
        self.assertFalse(self.demo.data.eq_active.any())
        forbidden_events = {"safety_stop", "safety_recover", "missed"}
        self.assertFalse(any(event["event"] in forbidden_events for event in self.events))
        self.assertFalse(self.demo.paused)

    def test_first_object_remains_held_until_physical_release(self):
        releases = [event for event in self.events if event["event"] == "release"]
        self.assertEqual({event["object_id"] for event in releases}, {"part_01"})
        for event in releases:
            self.assertEqual(event["finger_count"], 2)
            self.assertGreater(event["part_xyz"][2], 0.20)

    def test_first_object_is_verified_in_its_target_tray(self):
        placements = [event for event in self.events if event["event"] == "place"]
        self.assertEqual({event["object_id"] for event in placements}, {"part_01"})
        self.assertEqual(self.demo.placed, {"part_01"})
        self.assertFalse(self.demo.missed)

    def test_warning_margin_is_exposed_for_gui_monitoring(self):
        safety = self.demo.snapshot()["safety"]
        self.assertEqual(safety["warning_margin_m"], 0.10)
        self.assertFalse(self.demo._inter_arm_contacts())

    def test_gui_snapshot_is_json_serializable_after_a_real_decision(self):
        json.dumps(self.demo.snapshot(), ensure_ascii=False)

    def test_load_control_parameters_remain_explicit(self):
        self.assertEqual(self.demo.parameters.feed_interval_s, 5.0)
        self.assertEqual(int(self.demo.parameters.max_active_parts), 10)


if __name__ == "__main__":
    unittest.main()
