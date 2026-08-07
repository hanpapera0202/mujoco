"""Centralized deadline-aware coordinator for the dual-Nova5 sorting line.

The coordinator is deliberately independent from inverse kinematics. It filters
unsafe or infeasible arm-object pairs, then reserves shared zones in time. A and
B are peers: both are selected by one global assignment pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import dist
from typing import Iterable


class ArmId(str, Enum):
    A = "A"
    B = "B"


class ObjectState(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    PICKED = "picked"
    PLACED = "placed"
    MISSED = "missed"


@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    position_xyz: tuple[float, float, float]
    deadline_s: float
    grasp_success: dict[ArmId, float]
    shared_zone: str
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
    zone: str
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
    """Fast screening plus global, time-aware assignment and reservation."""

    def __init__(self, pick_speed_mps: float = 0.45, fixed_cycle_s: float = 1.2) -> None:
        self.pick_speed_mps = pick_speed_mps
        self.fixed_cycle_s = fixed_cycle_s
        self.reservations: list[Reservation] = []

    def decide(
        self,
        now_s: float,
        objects: Iterable[ObjectObservation],
        arms: Iterable[ArmState],
    ) -> Decision:
        """Assign feasible objects globally; both arms can be assigned together."""
        self.reservations = [reservation for reservation in self.reservations if reservation.interval_s[1] > now_s]
        arm_states = tuple(arms)
        candidates: list[Candidate] = []
        rejected: dict[str, list[str]] = {}

        for obj in objects:
            reasons = self._object_rejection_reasons(obj)
            if reasons:
                rejected[obj.object_id] = reasons
                continue

            feasible_for_object = 0
            for arm in arm_states:
                candidate, reason = self._screen_pair(now_s, obj, arm)
                if candidate is None:
                    rejected.setdefault(obj.object_id, []).append(f"{arm.arm}:{reason}")
                    continue
                candidates.append(candidate)
                feasible_for_object += 1
            if feasible_for_object == 0:
                rejected.setdefault(obj.object_id, []).append("no_feasible_arm")

        return Decision(self._reserve_best(candidates), rejected)

    def _object_rejection_reasons(self, obj: ObjectObservation) -> list[str]:
        if obj.state is not ObjectState.AVAILABLE:
            return [f"state={obj.state.value}"]
        if obj.deadline_s <= 0.0:
            return ["deadline_expired"]
        return []

    def _screen_pair(self, now_s: float, obj: ObjectObservation, arm: ArmState) -> tuple[Candidate | None, str]:
        if arm.busy_until_s > now_s:
            return None, "arm_busy"

        travel_m = dist(arm.tool_xyz, obj.position_xyz)
        if travel_m > arm.max_reach_m:
            return None, "out_of_reach"

        eta_s = travel_m / self.pick_speed_mps + self.fixed_cycle_s
        if eta_s >= obj.deadline_s:
            return None, "deadline_infeasible"

        success = obj.grasp_success.get(arm.arm, 0.0)
        if success <= 0.0:
            return None, "no_grasp_candidate"

        interval = (now_s + travel_m / self.pick_speed_mps, now_s + eta_s)
        urgency = 1.0 / max(obj.deadline_s, 0.05)
        score = 3.0 * urgency + 2.0 * success - 0.25 * travel_m
        return Candidate(arm.arm, obj.object_id, obj.shared_zone, interval, score), ""

    def _reserve_best(self, candidates: list[Candidate]) -> list[Candidate]:
        assignments: list[Candidate] = []
        used_arms: set[ArmId] = set()
        used_objects: set[str] = set()

        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if candidate.arm in used_arms or candidate.object_id in used_objects:
                continue
            if self._reservation_conflicts(candidate):
                continue
            reservation = Reservation(candidate.arm, candidate.object_id, candidate.zone, candidate.interval_s)
            self.reservations.append(reservation)
            assignments.append(candidate)
            used_arms.add(candidate.arm)
            used_objects.add(candidate.object_id)
        return assignments

    def _reservation_conflicts(self, candidate: Candidate) -> bool:
        for existing in self.reservations:
            if existing.zone != candidate.zone:
                continue
            start = max(existing.interval_s[0], candidate.interval_s[0])
            end = min(existing.interval_s[1], candidate.interval_s[1])
            if start < end:
                return True
        return False
