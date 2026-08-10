from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_sorting_demo import ArmId, SortingDemo


class SortingDemoPhysicalRegressionTests(unittest.TestCase):
    """Protect the seed-42 bilateral grasp demonstrated by v0.3.0."""

    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "models" / "nova5" / "nova5_sorting_line.xml"
        cls.demo = SortingDemo(model_path, seed=42)
        with redirect_stdout(StringIO()):
            while cls.demo.data.time < 4.15 and not cls.demo.paused:
                cls.demo.step()
        cls.events = cls.demo.event_log

    def test_first_pair_is_assigned_to_equal_peer_arms_concurrently(self):
        assignments = [event for event in self.events if event["event"] == "assign"]
        first_pair = {(event["arm"], event["object_id"]) for event in assignments}
        self.assertEqual(first_pair, {("A", "part_02"), ("B", "part_01")})
        self.assertAlmostEqual(assignments[0]["time_s"], assignments[1]["time_s"], places=3)

    def test_first_pair_requires_bilateral_physical_contact(self):
        grasps = [event for event in self.events if event["event"] == "grasp"]
        self.assertEqual({event["object_id"] for event in grasps}, {"part_01", "part_02"})
        for event in grasps:
            self.assertEqual(event["finger_count"], 2)
            self.assertEqual(event["contact"], "bilateral_finger_physical")
            self.assertEqual(event["grasp_constraint"], "none")

    def test_no_grasp_constraint_or_safety_failure_is_hidden(self):
        self.assertFalse(self.demo.data.eq_active.any())
        forbidden_events = {"safety_stop", "missed"}
        self.assertFalse(any(event["event"] in forbidden_events for event in self.events))
        self.assertFalse(self.demo.paused)

    def test_load_control_parameters_remain_explicit(self):
        self.assertGreaterEqual(self.demo.parameters.feed_interval_s, self.demo.parameters.fixed_cycle_s)
        self.assertEqual(int(self.demo.parameters.max_active_parts), 2)


if __name__ == "__main__":
    unittest.main()

