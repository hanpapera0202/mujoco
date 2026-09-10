"""Closed-chain coordination primitives derived from the supplied paper.

The normal conveyor task gives each peer arm a different object.  In that
mode the closed-chain constraint must stay optional: enforcing one shared
rigid-body pose would be physically incorrect.  This module therefore keeps
the paper's reusable pieces independent from the executor:

* three-point rigid-frame calibration;
* SE(3) pose composition and residuals for a shared workpiece;
* quintic time scaling and a joint-trajectory smoothness metric.

All transforms use column vectors and map the frame named on the right into
the frame named on the left, e.g. ``T_A_B @ p_B`` gives ``p_A``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ArrayLike = np.ndarray | list[list[float]] | list[tuple[float, float, float]]


def identity_transform() -> np.ndarray:
    """Return a new 4x4 identity transform."""

    return np.eye(4, dtype=float)


def make_transform(rotation: ArrayLike | None = None, translation: ArrayLike | None = None) -> np.ndarray:
    """Build a homogeneous transform from a rotation and translation."""

    result = identity_transform()
    if rotation is not None:
        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        result[:3, :3] = matrix
    if translation is not None:
        vector = np.asarray(translation, dtype=float).reshape(-1)
        if vector.shape != (3,):
            raise ValueError("translation must have three elements")
        result[:3, 3] = vector
    return result


def invert_transform(transform: ArrayLike) -> np.ndarray:
    """Invert an SE(3) transform without a general 4x4 inverse."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    rotation = matrix[:3, :3]
    inverse = identity_transform()
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def estimate_rigid_transform(source_points: ArrayLike, target_points: ArrayLike) -> np.ndarray:
    """Estimate ``T_target_source`` from three or more corresponding points.

    This is the three-point calibration step used by the paper, generalized to
    more points so real calibration can reject measurement noise.  The Kabsch
    solution is translation-invariant and enforces a proper rotation.
    """

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape (N, 3)")
    if source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError("at least three 3D point pairs are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    if np.linalg.matrix_rank(source_zero) < 2 or np.linalg.matrix_rank(target_zero) < 2:
        raise ValueError("calibration points must not be collinear")
    covariance = target_zero.T @ source_zero
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transposed
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def rotation_error_rad(actual: ArrayLike, expected: ArrayLike) -> float:
    """Return the geodesic angle between two rotation matrices."""

    actual_matrix = np.asarray(actual, dtype=float)
    expected_matrix = np.asarray(expected, dtype=float)
    if actual_matrix.shape != (3, 3) or expected_matrix.shape != (3, 3):
        raise ValueError("rotation matrices must have shape (3, 3)")
    relative = actual_matrix @ expected_matrix.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def pose_error(actual: ArrayLike, expected: ArrayLike) -> tuple[float, float]:
    """Return ``(translation_error_m, rotation_error_rad)`` for two poses."""

    actual_matrix = np.asarray(actual, dtype=float)
    expected_matrix = np.asarray(expected, dtype=float)
    if actual_matrix.shape != (4, 4) or expected_matrix.shape != (4, 4):
        raise ValueError("poses must have shape (4, 4)")
    translation = float(np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3]))
    rotation = rotation_error_rad(actual_matrix[:3, :3], expected_matrix[:3, :3])
    return translation, rotation


@dataclass
class ClosedChainKinematics:
    """Optional shared-workpiece relationship for two centrally planned arms.

    ``tool_a_in_object`` is ``T_object_tool_a`` and
    ``tool_b_in_object`` is ``T_object_tool_b``.  Given arm A's world pose,
    :meth:`peer_tool_pose` derives the world pose arm B must maintain for the
    same rigid object.  No master/slave scheduling is implied; a central
    planner may use the result for either peer and still solve both paths
    simultaneously.
    """

    base_b_in_a: np.ndarray = field(default_factory=identity_transform)
    tool_a_in_object: np.ndarray = field(default_factory=identity_transform)
    tool_b_in_object: np.ndarray = field(default_factory=identity_transform)

    def __post_init__(self) -> None:
        for name in ("base_b_in_a", "tool_a_in_object", "tool_b_in_object"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (4, 4):
                raise ValueError(f"{name} must have shape (4, 4)")
            setattr(self, name, value.copy())

    def peer_tool_pose(self, tool_a_world: ArrayLike) -> np.ndarray:
        """Derive arm B's world pose when arm A holds the shared object.

        The base relationship is intentionally not needed here because the
        method operates in world coordinates.  It remains part of the model
        for converting base-frame measurements and for future real-robot
        execution.
        """

        object_world = np.asarray(tool_a_world, dtype=float) @ self.tool_a_in_object
        return object_world @ invert_transform(self.tool_b_in_object)

    def relative_base_pose(self) -> np.ndarray:
        """Return the calibrated ``T_A_B`` mapping base-B coordinates to A."""

        return self.base_b_in_a.copy()

    def residual(self, tool_a_world: ArrayLike, tool_b_world: ArrayLike) -> tuple[float, float]:
        """Measure how far two observed tools are from the closed-chain pose."""

        expected_b = self.peer_tool_pose(tool_a_world)
        return pose_error(tool_b_world, expected_b)


def quintic_time_scaling(ratio: float) -> float:
    """C2-continuous 0-to-1 time scaling with zero endpoint speed/acceleration."""

    s = float(np.clip(ratio, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def joint_smoothness_cost(samples: ArrayLike, timestep_s: float) -> float:
    """Measure joint velocity/acceleration roughness for a sampled path.

    The cost is dimensionless after averaging across joints.  It is intended
    as a soft planner preference; collision and joint-limit violations remain
    hard rejections in the existing coordinator.
    """

    trajectory = np.asarray(samples, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError("samples must have shape (N, joints)")
    if timestep_s <= 0.0:
        raise ValueError("timestep_s must be positive")
    if trajectory.shape[0] < 3:
        return 0.0
    velocity = np.diff(trajectory, axis=0) / timestep_s
    acceleration = np.diff(velocity, axis=0) / timestep_s
    velocity_term = float(np.mean(np.linalg.norm(velocity, axis=1)))
    acceleration_term = float(np.mean(np.linalg.norm(acceleration, axis=1)))
    return velocity_term + 0.25 * acceleration_term


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to an ``(w, x, y, z)`` quaternion."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return np.array(
            (0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale)
        )
    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    next_index = (index + 1) % 3
    last_index = (index + 2) % 3
    scale = 2.0 * np.sqrt(max(1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index], 1e-15))
    quaternion = np.zeros(4, dtype=float)
    quaternion[index + 1] = 0.25 * scale
    quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
    quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / scale
    quaternion[last_index + 1] = (rotation[last_index, index] + rotation[index, last_index]) / scale
    return quaternion


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def interpolate_transform(start: ArrayLike, end: ArrayLike, ratio: float) -> np.ndarray:
    """Interpolate translation and rotation using quintic time scaling."""

    first = np.asarray(start, dtype=float)
    second = np.asarray(end, dtype=float)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("poses must have shape (4, 4)")
    s = quintic_time_scaling(ratio)
    first_quaternion = _rotation_to_quaternion(first[:3, :3])
    second_quaternion = _rotation_to_quaternion(second[:3, :3])
    if np.dot(first_quaternion, second_quaternion) < 0.0:
        second_quaternion = -second_quaternion
    dot = float(np.clip(np.dot(first_quaternion, second_quaternion), -1.0, 1.0))
    if dot > 0.9995:
        quaternion = first_quaternion + s * (second_quaternion - first_quaternion)
    else:
        angle = np.arccos(dot)
        quaternion = (
            np.sin((1.0 - s) * angle) * first_quaternion
            + np.sin(s * angle) * second_quaternion
        ) / np.sin(angle)
    return make_transform(
        _quaternion_to_rotation(quaternion),
        (1.0 - s) * first[:3, 3] + s * second[:3, 3],
    )


def sample_pose_path(start: ArrayLike, end: ArrayLike, duration_s: float, timestep_s: float) -> list[tuple[float, np.ndarray]]:
    """Discretize a synchronized low-level pose path with C2 endpoint timing."""

    if duration_s <= 0.0 or timestep_s <= 0.0:
        raise ValueError("duration_s and timestep_s must be positive")
    count = max(1, int(np.ceil(duration_s / timestep_s)))
    times = np.linspace(0.0, float(duration_s), count + 1)
    return [(float(time), interpolate_transform(start, end, time / duration_s)) for time in times]


@dataclass
class DualArmLowLevelLayer:
    """Paper-inspired low-level interface below the central task planner.

    ``independent_pick`` is the normal conveyor mode. ``closed_chain`` is for
    a shared workpiece or handoff and makes :meth:`peer_target` available.
    Both modes still return trajectories for the central collision/QP layer;
    this class never grants one arm scheduling priority over the other.
    """

    mode: str = "independent_pick"
    chain: ClosedChainKinematics = field(default_factory=ClosedChainKinematics)
    calibration_point_count: int = 0
    calibration_rms_m: float | None = None

    def set_mode(self, mode: str) -> None:
        if mode not in {"independent_pick", "closed_chain"}:
            raise ValueError("mode must be independent_pick or closed_chain")
        self.mode = mode

    def calibrate(self, points_in_a: ArrayLike, points_in_b: ArrayLike) -> float:
        """Calibrate ``T_A_B`` and return the point-fit RMS error in metres."""

        points_a = np.asarray(points_in_a, dtype=float)
        points_b = np.asarray(points_in_b, dtype=float)
        transform = estimate_rigid_transform(points_b, points_a)
        predicted_a = (transform[:3, :3] @ points_b.T).T + transform[:3, 3]
        errors = np.linalg.norm(predicted_a - points_a, axis=1)
        self.chain.base_b_in_a = transform
        self.calibration_point_count = int(points_a.shape[0])
        self.calibration_rms_m = float(np.sqrt(np.mean(errors**2)))
        return self.calibration_rms_m

    def peer_target(self, tool_a_world: ArrayLike) -> np.ndarray:
        """Return the B target only when the shared-workpiece mode is active."""

        if self.mode != "closed_chain":
            raise RuntimeError("peer_target requires closed_chain mode")
        return self.chain.peer_tool_pose(tool_a_world)

    def synchronized_path(self, start: ArrayLike, end: ArrayLike, duration_s: float, timestep_s: float) -> list[tuple[float, np.ndarray]]:
        """Create the low-level discrete path consumed by a joint controller."""

        return sample_pose_path(start, end, duration_s, timestep_s)

    def snapshot(self) -> dict[str, object]:
        """Return GUI/log-friendly low-level state."""

        return {
            "architecture": "paper_closed_chain_low_level",
            "mode": self.mode,
            "calibration_points": self.calibration_point_count,
            "calibration_rms_m": None if self.calibration_rms_m is None else round(self.calibration_rms_m, 9),
            "peer_target_enabled": self.mode == "closed_chain",
        }
