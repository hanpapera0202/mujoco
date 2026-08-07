"""Centralized task allocation for a dual-Nova5 conveyor sorting line.

This module intentionally owns high-level coordination only.  A MuJoCo/IK
executor receives the selected assignments; it reports completion separately.
The coordinator therefore stays deterministic, cheap to evaluate, and suitable
for comparing assignment policies using exactly the same low-level executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from math import dist
from typing import Iterable


class ArmId(str, Enum):
    A = "A"
    B = "B"


class ObjectClass(str, Enum):
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class ObjectState(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    PICKED = "picked"
    PLACED = "placed"
    MISSED = "missed"


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    object_class: ObjectClass
    position_xyz: tuple[float, float, float]
    deadline_s: float
    grasp_success: dict[ArmId, float]
    state: ObjectState = ObjectState.AVAILABLE


@dataclass(frozen=True)
class ArmState:
    arm: ArmId
    tool_xyz: tuple[float, float, float]
    max_reach_m: float
    busy_until_s: float = 0.0


@dataclass(frozen=True)
class Candidate:
    arm: ArmId
    object_id: str
    object_class: ObjectClass
    workspace_zone: str
    placement_zone: str
    interval_s: tuple[float, float]
    score: float


@dataclass(frozen=True)
class Reservation:
    arm: ArmId
    object_id: str
    zone: str
    interval_s: tuple[float, float]


@dataclass
class Decision:
    assignments: list[Candidate] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)


class CentralCoordinator:
    """Hard-screen candidates then globally choose a parallel-safe assignment.

    LEFT objects are exclusive to A and RIGHT objects are exclusive to B in v1.
    MIDDLE objects are shared: assigning one to A routes it to ``left_bin``;
    assigning one to B routes it to ``right_bin``.  Reservations are commitments:
    they are never reassigned by later calls to :meth:`decide`.
    """

    def __init__(
        self,
        pick_speed_mps: float = 0.45,
        fixed_cycle_s: float = 1.2,
        horizon_s: float = 5.0,
        parallel_bonus: float = 2.0,
        urgency_weight: float = 3.0,
        success_weight: float = 2.0,
        travel_weight: float = 0.25,
    ) -> None:
        self.pick_speed_mps = pick_speed_mps
        self.fixed_cycle_s = fixed_cycle_s
        self.horizon_s = horizon_s
        self.parallel_bonus = parallel_bonus
        self.urgency_weight = urgency_weight
        self.success_weight = success_weight
        self.travel_weight = travel_weight
        self.reservations: list[Reservation] = []

    def decide(
        self,
        now_s: float,
        objects: Iterable[ObjectObservation],
        arms: Iterable[ArmState],
    ) -> Decision:
        """Return at most one committed assignment per free arm.

        Screening is O(objects * arms).  The final global choice enumerates only
        single candidates and A/B pairs, so it stays small and deterministic.
        """
        self._remove_expired_reservations(now_s)
        arm_states = {arm.arm: arm for arm in arms}
        reserved_object_ids = {reservation.object_id for reservation in self.reservations}
        candidates: list[Candidate] = []
        rejected: dict[str, list[str]] = {}

        for obj in objects:
            reasons = self._object_rejection_reasons(obj, reserved_object_ids)
            if reasons:
                rejected[obj.object_id] = reasons
                continue

            feasible = 0
            for arm in arm_states.values():
                candidate, reason = self._screen_pair(now_s, obj, arm)
                if candidate is None:
                    rejected.setdefault(obj.object_id, []).append(f"{arm.arm}:{reason}")
                    continue
                candidates.append(candidate)
                feasible += 1
            if feasible == 0:
                rejected.setdefault(obj.object_id, []).append("no_feasible_arm")

        assignments = self._choose_global_assignment(candidates)
        self.reservations.extend(
            Reservation(item.arm, item.object_id, item.workspace_zone, item.interval_s) for item in assignments
        )
        return Decision(assignments, rejected)

    def mark_completed(self, object_id: str) -> None:
        """Release a completed commitment after the executor reports placement."""
        self.reservations = [item for item in self.reservations if item.object_id != object_id]

    def _object_rejection_reasons(self, obj: ObjectObservation, reserved_object_ids: set[str]) -> list[str]:
        if obj.object_id in reserved_object_ids:
            return ["already_committed"]
        if obj.state is not ObjectState.AVAILABLE:
            return [f"state={obj.state.value}"]
        if obj.deadline_s <= 0.0:
            return ["deadline_expired"]
        if obj.deadline_s > self.horizon_s:
            return ["outside_rolling_horizon"]
        return []

    def _screen_pair(self, now_s: float, obj: ObjectObservation, arm: ArmState) -> tuple[Candidate | None, str]:
        if not self._arm_may_handle(arm.arm, obj.object_class):
            return None, "exclusive_zone"
        if arm.busy_until_s > now_s:
            return None, "arm_busy"

        travel_m = dist(arm.tool_xyz, obj.position_xyz)
        if travel_m > arm.max_reach_m:
            return None, "out_of_reach"

        travel_s = travel_m / self.pick_speed_mps
        eta_s = travel_s + self.fixed_cycle_s
        if eta_s >= obj.deadline_s:
            return None, "deadline_infeasible"

        success = obj.grasp_success.get(arm.arm, 0.0)
        if success <= 0.0:
            return None, "no_grasp_candidate"

        urgency = 1.0 / max(obj.deadline_s, 0.05)
        score = self.urgency_weight * urgency + self.success_weight * success - self.travel_weight * travel_m
        return Candidate(
            arm=arm.arm,
            object_id=obj.object_id,
            object_class=obj.object_class,
            workspace_zone=self._workspace_zone(obj.object_class),
            placement_zone="left_bin" if arm.arm is ArmId.A else "right_bin",
            interval_s=(now_s + travel_s, now_s + eta_s),
            score=score,
        ), ""

    @staticmethod
    def _arm_may_handle(arm: ArmId, object_class: ObjectClass) -> bool:
        return object_class is ObjectClass.MIDDLE or (
            object_class is ObjectClass.LEFT and arm is ArmId.A
        ) or (object_class is ObjectClass.RIGHT and arm is ArmId.B)

    @staticmethod
    def _workspace_zone(object_class: ObjectClass) -> str:
        return "shared_middle" if object_class is ObjectClass.MIDDLE else f"exclusive_{object_class.value}"

    def _choose_global_assignment(self, candidates: list[Candidate]) -> list[Candidate]:
        feasible_sets: list[tuple[Candidate, ...]] = [()]
        feasible_sets.extend((candidate,) for candidate in candidates if not self._reservation_conflicts(candidate))
        for first, second in combinations(candidates, 2):
            pair = (first, second)
            if first.arm is second.arm or first.object_id == second.object_id:
                continue
            if self._reservation_conflicts(first) or self._reservation_conflicts(second):
                continue
            if self._pair_conflicts(first, second):
                continue
            feasible_sets.append(pair)

        def objective(items: tuple[Candidate, ...]) -> tuple[float, int, float]:
            # Prefer two useful, simultaneous motions before minor score gains.
            parallel = self.parallel_bonus if len(items) == 2 else 0.0
            score = sum(item.score for item in items) + parallel
            earliest_deadline_proxy = -sum(item.interval_s[1] for item in items)
            return score, len(items), earliest_deadline_proxy

        return list(max(feasible_sets, key=objective))

    def _pair_conflicts(self, first: Candidate, second: Candidate) -> bool:
        return first.workspace_zone == second.workspace_zone and self._intervals_overlap(first.interval_s, second.interval_s)

    def _reservation_conflicts(self, candidate: Candidate) -> bool:
        return any(
            existing.zone == candidate.workspace_zone
            and self._intervals_overlap(existing.interval_s, candidate.interval_s)
            for existing in self.reservations
        )

    def _remove_expired_reservations(self, now_s: float) -> None:
        self.reservations = [item for item in self.reservations if item.interval_s[1] > now_s]

    @staticmethod
    def _intervals_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
        return max(first[0], second[0]) < min(first[1], second[1])
