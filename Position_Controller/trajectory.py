"""Deterministic takeoff, 200 m trajectory, and landing logic."""

import math
from dataclasses import dataclass

try:
    from .VehicleState import VehicleState
except ImportError:  # Allows direct execution from Position_Controller.
    from VehicleState import VehicleState


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def limit_xy(vx: float, vy: float, limit: float) -> tuple[float, float]:
    speed = math.hypot(vx, vy)
    scale = min(1.0, limit / speed) if speed else 1.0
    return vx * scale, vy * scale


@dataclass(frozen=True)
class Config:
    connection: str = "udp:127.0.0.1:14540"
    hz: float = 10.0
    timing_tolerance: float = 0.10
    distance_m: float = 200.0
    cruise_m_s: float = 5.0
    takeoff_m: float = 8.0
    slow_below_m: float = 5.0
    slow_descent_m_s: float = 0.1
    land_handoff_m: float = 0.15
    fast_descent_m_s: float = 0.5
    acceleration_m_s2: float = 1.0
    position_kp: float = 0.8
    altitude_kp: float = 1.0
    takeoff_speed_m_s: float = 0.5
    position_tolerance_m: float = 0.5
    speed_tolerance_m_s: float = 0.3
    telemetry_timeout_s: float = 0.5
    mission_timeout_s: float = 240.0

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    def validate(self) -> None:
        for name, value in vars(self).items():
            if name != "connection" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.timing_tolerance < 1:
            raise ValueError("timing_tolerance must be between 0 and 1")
        if not self.slow_descent_m_s < self.fast_descent_m_s:
            raise ValueError("slow descent must be slower than fast descent")
        if not 0 < self.land_handoff_m < self.slow_below_m:
            raise ValueError("land handoff must be inside the slow-descent zone")
        if self.takeoff_m <= self.slow_below_m:
            raise ValueError("takeoff height must exceed the slow-descent height")


@dataclass(frozen=True)
class Command:
    vx: float
    vy: float
    vz: float
    yaw: float
    target_x: float
    target_y: float
    target_z: float


class Mission:
    """Mission state machine advanced by one fixed 0.1 s step per call."""

    def __init__(self, config: Config, initial: VehicleState, heading_deg: float):
        config.validate()
        self.c = config
        self.start_x, self.start_y, self.ground_z = initial.x, initial.y, initial.z
        self.yaw = initial.yaw
        heading = math.radians(heading_deg)
        self.north, self.east = math.cos(heading), math.sin(heading)
        self.end_x = self.start_x + self.north * config.distance_m
        self.end_y = self.start_y + self.east * config.distance_m
        self.progress = 0.0
        self.reference_speed = 0.0
        self.phase = "TAKEOFF"

    def agl(self, state: VehicleState) -> float:
        return self.ground_z - state.z

    def hold_xy(
        self, state: VehicleState, x: float, y: float, cap: float
    ) -> tuple[float, float]:
        return limit_xy(
            self.c.position_kp * (x - state.x),
            self.c.position_kp * (y - state.y),
            cap,
        )

    def step(self, state: VehicleState) -> Command:
        target_z = self.ground_z - self.c.takeoff_m

        if self.phase == "TAKEOFF":
            vx, vy = self.hold_xy(state, self.start_x, self.start_y, 1.0)
            vz = clamp(
                self.c.altitude_kp * (target_z - state.z),
                -self.c.takeoff_speed_m_s,
                self.c.takeoff_speed_m_s,
            )
            if abs(target_z - state.z) <= 0.2 and abs(state.vz) <= self.c.speed_tolerance_m_s:
                self.phase = "TRAJECTORY"
            return Command(vx, vy, vz, self.yaw, self.start_x, self.start_y, target_z)

        if self.phase == "TRAJECTORY":
            remaining = max(0.0, self.c.distance_m - self.progress)
            lookahead = self.reference_speed * self.c.dt
            stopping_speed = math.sqrt(
                2 * self.c.acceleration_m_s2 * max(0.0, remaining - lookahead)
            )
            goal_speed = min(self.c.cruise_m_s, stopping_speed)
            change = self.c.acceleration_m_s2 * self.c.dt
            next_speed = self.reference_speed + clamp(
                goal_speed - self.reference_speed, -change, change
            )
            step_distance = 0.5 * (self.reference_speed + next_speed) * self.c.dt
            if step_distance >= remaining:
                self.progress, next_speed = self.c.distance_m, 0.0
            else:
                self.progress += step_distance
            self.reference_speed = next_speed
            target_x = self.start_x + self.north * self.progress
            target_y = self.start_y + self.east * self.progress
            vx = self.north * next_speed + self.c.position_kp * (target_x - state.x)
            vy = self.east * next_speed + self.c.position_kp * (target_y - state.y)
            vx, vy = limit_xy(vx, vy, self.c.cruise_m_s)
            vz = clamp(
                self.c.altitude_kp * (target_z - state.z),
                -self.c.takeoff_speed_m_s,
                self.c.takeoff_speed_m_s,
            )
            error = math.hypot(self.end_x - state.x, self.end_y - state.y)
            speed = math.hypot(state.vx, state.vy)
            if (
                self.progress >= self.c.distance_m
                and error <= self.c.position_tolerance_m
                and speed <= self.c.speed_tolerance_m_s
            ):
                self.phase = "LAND_FAST"
            return Command(vx, vy, vz, self.yaw, target_x, target_y, target_z)

        vx, vy = self.hold_xy(state, self.end_x, self.end_y, 1.0)
        if self.phase == "LAND_FAST" and self.agl(state) <= self.c.slow_below_m:
            self.phase = "LAND_SLOW"
        if self.phase == "LAND_SLOW" and self.agl(state) <= self.c.land_handoff_m:
            self.phase = "LAND_HANDOFF"
        if state.landed == 1:
            self.phase = "DONE"
        if self.phase in ("LAND_HANDOFF", "DONE"):
            vz = 0.0
        elif self.phase == "LAND_SLOW":
            vz = self.c.slow_descent_m_s
        else:
            vz = self.c.fast_descent_m_s
        return Command(vx, vy, vz, self.yaw, self.end_x, self.end_y, self.ground_z)
