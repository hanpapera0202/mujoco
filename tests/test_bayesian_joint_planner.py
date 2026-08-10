import unittest

from src.bayesian_joint_planner import BayesianJointGame, JointStrategyEvidence


def evidence(route_a="direct", route_b="direct", **changes):
    values = dict(
        route_a=route_a,
        route_b=route_b,
        collision_free=True,
        makespan_s=20.0,
        simultaneous_ratio=0.7,
        path_length_rad=10.0,
        grasp_probability_a=0.8,
        grasp_probability_b=0.8,
    )
    values.update(changes)
    return JointStrategyEvidence(**values)


class BayesianJointPlannerTests(unittest.TestCase):
    def test_deterministic_collision_is_a_hard_rejection(self):
        game = BayesianJointGame()
        selected = game.choose([
            evidence(collision_free=False, makespan_s=1.0),
            evidence("balanced", "outer", makespan_s=30.0),
        ])
        self.assertEqual((selected.evidence.route_a, selected.evidence.route_b), ("balanced", "outer"))

    def test_global_utility_can_reject_both_local_shortest_routes(self):
        game = BayesianJointGame()
        direct = evidence(simultaneous_ratio=0.0, grasp_probability_a=0.55, grasp_probability_b=0.55)
        coordinated = evidence("balanced", "balanced", makespan_s=22.0, simultaneous_ratio=0.9)
        selected = game.choose([direct, coordinated])
        self.assertEqual((selected.evidence.route_a, selected.evidence.route_b), ("balanced", "balanced"))

    def test_beta_bernoulli_feedback_changes_the_posterior(self):
        game = BayesianJointGame()
        before = game.belief_for("direct", "outer").mean
        game.update("direct", "outer", False)
        self.assertLess(game.belief_for("direct", "outer").mean, before)


if __name__ == "__main__":
    unittest.main()
