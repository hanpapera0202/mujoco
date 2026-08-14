from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco

from run_single_arm_demo import MODEL, SingleArmDemo


class SingleArmModelTests(unittest.TestCase):
    def test_model_contains_one_arm_and_complete_line(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.assertEqual(model.nu, 8)
        self.assertGreaterEqual(model.nbody, 30)
        self.assertIsNotNone(model.body("robot_A_base"))
        self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_B_base"), -1)
        for index in range(1, 9):
            self.assertIsNotNone(model.body(f"belt_segment_{index:02d}"))
        self.assertIsNotNone(model.site("left_bin_drop"))

    def test_single_arm_demo_advances_dynamic_conveyor(self):
        demo = SingleArmDemo(seed=42)
        initial = float(demo.data.qpos[demo.qpos["part_01"] + 1])
        for _ in range(150):
            demo.step()
        self.assertGreater(len(demo.spawned), 0)
        self.assertNotEqual(float(demo.data.qpos[demo.qpos["part_01"] + 1]), initial)


if __name__ == "__main__":
    unittest.main()
