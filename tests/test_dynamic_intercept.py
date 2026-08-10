from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco

from central_coordinator import ArmId, Candidate, ObjectClass
from run_sorting_demo import SortingDemo, place_part


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

    def test_executor_rejection_rolls_back_assignment_fairness(self):
        demo = SortingDemo(ROOT / "models" / "nova5" / "nova5_sorting_line.xml", seed=42)
        assignment = Candidate(ArmId.A, "part_01", ObjectClass.MIDDLE, "shared_middle", "left_bin", (0.0, 10.0), 1.0)
        demo.coordinator.assignment_counts[ArmId.A] = 1
        demo._release_unstarted_assignment(assignment)
        self.assertEqual(demo.coordinator.assignment_counts[ArmId.A], 0)


if __name__ == "__main__":
    unittest.main()
