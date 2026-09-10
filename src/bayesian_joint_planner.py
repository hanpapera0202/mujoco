"""Explainable Bayesian game for centralized dual-arm route selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetaBelief:
    alpha: float = 2.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0


@dataclass(frozen=True)
class JointStrategyEvidence:
    route_a: str
    route_b: str
    collision_free: bool
    makespan_s: float
    simultaneous_ratio: float
    path_length_rad: float
    grasp_probability_a: float
    grasp_probability_b: float
    rejection_reason: str = ""
    warning_overlap_ratio: float = 0.0
    smoothness_cost: float = 0.0


@dataclass(frozen=True)
class JointStrategyScore:
    evidence: JointStrategyEvidence
    completion_probability: float
    collision_probability: float
    expected_utility: float


class BayesianJointGame:
    """Score joint routes under uncertain grasp and route-safety outcomes.

    Deterministic MuJoCo collisions remain hard constraints. Beta-Bernoulli
    beliefs capture uncertainty left after finite trajectory sampling.
    """

    def __init__(self) -> None:
        self.route_safety: dict[tuple[str, str], BetaBelief] = {}

    def belief_for(self, route_a: str, route_b: str) -> BetaBelief:
        return self.route_safety.setdefault((route_a, route_b), BetaBelief(9.0, 1.0))

    def score(self, evidence: JointStrategyEvidence) -> JointStrategyScore:
        safety_probability = self.belief_for(evidence.route_a, evidence.route_b).mean
        if not evidence.collision_free:
            safety_probability = 0.0
        completion_probability = (
            safety_probability * evidence.grasp_probability_a * evidence.grasp_probability_b
        )
        collision_probability = 1.0 - safety_probability
        # Completion dominates makespan, which dominates concurrency and path
        # length. The weights are exposed here so later experiments can tune
        # the utility without changing the executor.
        expected_utility = (
            1000.0 * completion_probability
            - 12.0 * evidence.makespan_s
            + 30.0 * evidence.simultaneous_ratio
            - 0.25 * evidence.path_length_rad
            - 400.0 * collision_probability
            - 180.0 * evidence.warning_overlap_ratio
            - 0.10 * evidence.smoothness_cost
        )
        return JointStrategyScore(evidence, completion_probability, collision_probability, expected_utility)

    def choose(self, candidates: list[JointStrategyEvidence]) -> JointStrategyScore | None:
        feasible = [self.score(item) for item in candidates if item.collision_free]
        return max(feasible, key=lambda item: item.expected_utility, default=None)

    def update(self, route_a: str, route_b: str, success: bool) -> None:
        self.belief_for(route_a, route_b).update(success)
