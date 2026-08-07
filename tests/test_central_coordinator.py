from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from central_coordinator import ArmId, ArmState, CentralCoordinator, ObjectClass, ObjectObservation


ARMS = (
    ArmState(ArmId.A, (-0.6, 0.0, 0.4), 1.5),
    ArmState(ArmId.B, (0.6, 0.0, 0.4), 1.5),
)


def obj(name, object_class, x, deadline=4.0):
    return ObjectObservation(name, object_class, (x, 0.0, 0.2), deadline, {ArmId.A: 0.9, ArmId.B: 0.9})


class CentralCoordinatorTests(unittest.TestCase):
    def test_exclusive_objects_keep_arms_parallel(self):
        decision = CentralCoordinator().decide(0.0, (obj("left", ObjectClass.LEFT, -0.5), obj("right", ObjectClass.RIGHT, 0.5)), ARMS)
        self.assertEqual({(item.arm, item.object_id) for item in decision.assignments}, {(ArmId.A, "left"), (ArmId.B, "right")})

    def test_middle_uses_one_arm_but_other_arm_can_take_exclusive_work(self):
        decision = CentralCoordinator().decide(0.0, (obj("middle", ObjectClass.MIDDLE, 0.0), obj("right", ObjectClass.RIGHT, 0.5)), ARMS)
        self.assertEqual(len(decision.assignments), 2)
        self.assertEqual({item.arm for item in decision.assignments}, {ArmId.A, ArmId.B})
        self.assertEqual({item.object_id for item in decision.assignments}, {"middle", "right"})

    def test_shared_middle_reservation_is_committed(self):
        coordinator = CentralCoordinator()
        middle = obj("middle", ObjectClass.MIDDLE, 0.0)
        first = coordinator.decide(0.0, (middle,), ARMS)
        self.assertEqual(len(first.assignments), 1)
        second = coordinator.decide(0.1, (middle,), ARMS)
        self.assertEqual(second.assignments, [])
        self.assertEqual(second.rejected["middle"], ["already_committed"])

    def test_cross_zone_assignment_is_rejected(self):
        decision = CentralCoordinator().decide(0.0, (obj("left", ObjectClass.LEFT, -0.5),), (ARMS[1],))
        self.assertEqual(decision.assignments, [])
        self.assertIn("exclusive_zone", decision.rejected["left"][0])


if __name__ == "__main__":
    unittest.main()
