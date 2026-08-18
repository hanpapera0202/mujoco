"""Small bounded GA/PSO planner for centralized dual-arm assignment.

The planner operates only when the coordinator is making a new assignment.
It never runs inside MuJoCo's 500 Hz control loop.  Hard collision and
reservation checks remain outside this module and are always authoritative.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class EvolutionaryPlan:
    indices: tuple[int, ...]
    ga_generations: int
    pso_iterations: int
    pso_parallel_weight: float


class EvolutionaryPlanner:
    """Bounded genetic selection with a small PSO weight refinement."""

    def __init__(self, seed: int = 17, population_size: int = 18, generations: int = 8, pso_particles: int = 8) -> None:
        self.rng = random.Random(seed)
        self.population_size = population_size
        self.generations = generations
        self.pso_particles = pso_particles

    def choose(
        self,
        candidates: Sequence[object],
        *,
        parallel_bonus: float,
        assignment_counts: dict[object, int],
        pair_conflicts: Callable[[object, object], bool],
        reservation_conflicts: Callable[[object], bool],
    ) -> EvolutionaryPlan:
        options: list[tuple[int, ...]] = [()]
        for index, candidate in enumerate(candidates):
            if not reservation_conflicts(candidate):
                options.append((index,))
        for first in range(len(candidates)):
            for second in range(first + 1, len(candidates)):
                left, right = candidates[first], candidates[second]
                if left.arm == right.arm or left.object_id == right.object_id:
                    continue
                if reservation_conflicts(left) or reservation_conflicts(right) or pair_conflicts(left, right):
                    continue
                options.append((first, second))
        if len(options) == 1:
            return EvolutionaryPlan((), 0, 0, parallel_bonus)

        def fitness(option: tuple[int, ...], parallel_weight: float) -> float:
            selected = [candidates[index] for index in option]
            score = sum(float(item.score) for item in selected)
            if len(selected) == 2:
                score += parallel_weight
            score -= 0.15 * sum(assignment_counts.get(item.arm, 0) for item in selected)
            # Stable first-choice tie break; the fairness term makes the next
            # equal tie prefer the other arm.
            score -= 0.001 * sum(1 for item in selected if getattr(item.arm, "value", item.arm) == "B")
            return score

        population = [self.rng.choice(options) for _ in range(self.population_size)]
        for _ in range(self.generations):
            population.sort(key=lambda item: fitness(item, parallel_bonus), reverse=True)
            next_population = population[: max(2, self.population_size // 3)]
            while len(next_population) < self.population_size:
                first = self.rng.choice(population[: max(2, self.population_size // 2)])
                second = self.rng.choice(population[: max(2, self.population_size // 2)])
                child = self.rng.choice((first, second))
                if self.rng.random() < 0.25:
                    child = self.rng.choice(options)
                next_population.append(child)
            population = next_population

        # PSO refines the continuous priority on parallel motion. The chosen
        # assignment is still an option from the hard-screened GA pool.
        particles = [max(0.0, parallel_bonus + self.rng.uniform(-1.0, 1.0)) for _ in range(self.pso_particles)]
        velocities = [0.0] * len(particles)
        best_weight = parallel_bonus
        best_option = max(population, key=lambda item: fitness(item, best_weight))
        best_value = fitness(best_option, best_weight)
        personal = [(weight, weight, best_value) for weight in particles]
        for iteration in range(8):
            for index, weight in enumerate(particles):
                option = max(options, key=lambda item: fitness(item, weight))
                value = fitness(option, weight)
                if value > personal[index][2]:
                    personal[index] = (weight, weight, value)
                if value > best_value:
                    best_weight, best_option, best_value = weight, option, value
            for index, weight in enumerate(particles):
                _, personal_best, _ = personal[index]
                velocities[index] = 0.55 * velocities[index] + 0.9 * self.rng.random() * (personal_best - weight) + 0.9 * self.rng.random() * (best_weight - weight)
                particles[index] = min(4.0, max(0.0, weight + velocities[index]))
        return EvolutionaryPlan(tuple(best_option), self.generations, iteration + 1, round(best_weight, 4))
