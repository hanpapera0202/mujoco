import unittest
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_chain_kinematics import (
    ClosedChainKinematics,
    DualArmLowLevelLayer,
    estimate_rigid_transform,
    identity_transform,
    invert_transform,
    joint_smoothness_cost,
    make_transform,
    pose_error,
    quintic_time_scaling,
)


class ClosedChainKinematicsTests(unittest.TestCase):
    def test_three_point_calibration_recovers_a_rigid_transform(self):
        source = np.array(((0.0, 0.0, 0.0), (0.4, 0.1, 0.2), (-0.1, 0.6, 0.3), (0.2, -0.2, 0.8)))
        angle = 0.37
        rotation = np.array(
            ((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
        )
        expected = make_transform(rotation, (1.2, -0.4, 0.7))
        target = (expected[:3, :3] @ source.T).T + expected[:3, 3]
        actual = estimate_rigid_transform(source, target)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_collinear_calibration_points_are_rejected(self):
        points = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            estimate_rigid_transform(points, points)

    def test_transform_inverse_and_peer_pose_close_the_chain(self):
        base = make_transform(translation=(0.8, -0.2, 0.1))
        tool_a = make_transform(translation=(0.0, 0.0, 0.15))
        tool_b = make_transform(translation=(0.2, 0.0, 0.15))
        chain = ClosedChainKinematics(base, tool_a, tool_b)
        tool_a_world = make_transform(translation=(0.4, 0.3, 0.8))
        tool_b_world = chain.peer_tool_pose(tool_a_world)
        self.assertEqual(pose_error(tool_b_world, chain.peer_tool_pose(tool_a_world)), (0.0, 0.0))
        np.testing.assert_allclose(invert_transform(invert_transform(base)), base)
        np.testing.assert_allclose(invert_transform(base) @ base, identity_transform())

    def test_quintic_scaling_has_fixed_endpoints_and_is_monotonic(self):
        values = np.array([quintic_time_scaling(x) for x in np.linspace(0.0, 1.0, 101)])
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertTrue(np.all(np.diff(values) >= 0.0))

    def test_smooth_path_cost_is_lower_than_a_jagged_path(self):
        smooth = np.linspace(0.0, 1.0, 12)[:, None] * np.ones((1, 6))
        jagged = smooth.copy()
        jagged[4:8] += np.array((0.5, -0.4, 0.3, -0.2, 0.1, -0.1))
        self.assertLess(joint_smoothness_cost(smooth, 0.1), joint_smoothness_cost(jagged, 0.1))

    def test_low_level_layer_calibrates_and_reports_independent_mode(self):
        points_a = np.array(((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)))
        points_b = points_a - np.array((0.5, -0.2, 0.1))
        layer = DualArmLowLevelLayer()
        self.assertAlmostEqual(layer.calibrate(points_a, points_b), 0.0, places=10)
        self.assertEqual(layer.snapshot()["mode"], "independent_pick")
        with self.assertRaises(RuntimeError):
            layer.peer_target(identity_transform())

    def test_low_level_layer_exposes_a_synchronized_c2_path(self):
        layer = DualArmLowLevelLayer()
        path = layer.synchronized_path(identity_transform(), make_transform(translation=(1.0, 0.0, 0.0)), 1.0, 0.1)
        self.assertEqual(len(path), 11)
        np.testing.assert_allclose(path[0][1], identity_transform())
        np.testing.assert_allclose(path[-1][1][:3, 3], (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
