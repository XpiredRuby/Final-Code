#!/usr/bin/env python3
"""Simple 10 Hz PX4 SITL mission based on Vishnu's controller."""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from pymavlink import mavutil
except ImportError:  # Unit tests only exercise the mission logic.
    mavutil = None


def decode_px4_mode(message) -> str:
    """Decode the PX4 custom mode without relying on changing base-mode flags."""
    main_mode = (message.custom_mode >> 16) & 0xFF
    sub_mode = (message.custom_mode >> 24) & 0xFF
    if main_mode == 6:
        return "OFFBOARD"
    if main_mode == 4 and sub_mode == 6:
        return "LAND"
    return mavutil.mode_string_v10(message)


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


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    mode: str = "UNKNOWN"
    armed: bool = False
    landed: int | None = None
    position_at: float = 0.0
    attitude_at: float = 0.0
    heartbeat_at: float = 0.0
    landed_at: float = 0.0

    def fresh(self, now: float, fast_timeout: float) -> bool:
        fast = (self.position_at, self.attitude_at)
        slow = (self.heartbeat_at, self.landed_at)
        return (
            all(stamp > 0 and 0 <= now - stamp <= fast_timeout for stamp in fast)
            and all(stamp > 0 and 0 <= now - stamp <= 1.5 for stamp in slow)
        )


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
    """Deterministic mission logic. Every step uses exactly 0.1 seconds."""

    def __init__(self, config: Config, initial: State, heading_deg: float):
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

    def agl(self, state: State) -> float:
        return self.ground_z - state.z

    def hold_xy(self, state: State, x: float, y: float, cap: float) -> tuple[float, float]:
        return limit_xy(
            self.c.position_kp * (x - state.x),
            self.c.position_kp * (y - state.y),
            cap,
        )

    def step(self, state: State) -> Command:
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
            braking_distance = max(0.0, remaining - lookahead)
            stopping_speed = math.sqrt(2 * self.c.acceleration_m_s2 * braking_distance)
            goal_speed = min(self.c.cruise_m_s, stopping_speed)
            change = self.c.acceleration_m_s2 * self.c.dt
            next_speed = self.reference_speed + clamp(
                goal_speed - self.reference_speed, -change, change
            )
            step_distance = 0.5 * (self.reference_speed + next_speed) * self.c.dt
            if step_distance >= remaining:
                self.progress = self.c.distance_m
                next_speed = 0.0
            else:
                self.progress += step_distance
            self.reference_speed = next_speed
            target_x = self.start_x + self.north * self.progress
            target_y = self.start_y + self.east * self.progress
            vx = self.north * self.reference_speed + self.c.position_kp * (target_x - state.x)
            vy = self.east * self.reference_speed + self.c.position_kp * (target_y - state.y)
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
        vz = 0.0 if self.phase in ("LAND_HANDOFF", "DONE") else (
            self.c.slow_descent_m_s if self.phase == "LAND_SLOW" else self.c.fast_descent_m_s
        )
        return Command(vx, vy, vz, self.yaw, self.end_x, self.end_y, self.ground_z)


class PX4:
    def __init__(self, config: Config, heading_deg: float, log_path: Path):
        if mavutil is None:
            raise RuntimeError("Run: python -m pip install -r requirements.txt")
        self.c, self.heading_deg, self.log_path = config, heading_deg, log_path
        self.state, self.lock = State(), threading.Lock()
        self.master = self.thread = None
        self.running = False

    def snapshot(self) -> State:
        with self.lock:
            return State(**vars(self.state))

    def connect(self) -> None:
        print(f"Connecting to {self.c.connection}")
        self.master = mavutil.mavlink_connection(self.c.connection)
        if self.master.wait_heartbeat(timeout=10) is None:
            raise TimeoutError("No PX4 heartbeat")
        self.running = True
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()
        for message_id in (32, 30, 245):  # position, attitude, extended state
            self.command_long(511, message_id, int(self.c.dt * 1_000_000))

    def receive(self) -> None:
        while self.running:
            message = self.master.recv_match(blocking=True, timeout=0.2)
            if message is None:
                continue
            now, kind = time.monotonic(), message.get_type()
            with self.lock:
                if kind == "LOCAL_POSITION_NED":
                    self.state.x, self.state.y, self.state.z = message.x, message.y, message.z
                    self.state.vx, self.state.vy, self.state.vz = message.vx, message.vy, message.vz
                    self.state.position_at = now
                elif kind == "ATTITUDE":
                    self.state.yaw, self.state.attitude_at = message.yaw, now
                elif kind == "HEARTBEAT":
                    self.state.mode = decode_px4_mode(message)
                    self.state.armed = bool(message.base_mode & 128)
                    self.state.heartbeat_at = now
                elif kind == "EXTENDED_SYS_STATE":
                    self.state.landed, self.state.landed_at = message.landed_state, now

    def command_long(self, command: int, *params: float) -> None:
        values = list(params) + [0.0] * (7 - len(params))
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component, command, 0, *values
        )

    def send(self, command: Command) -> None:
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            2503,  # velocity and yaw only
            0, 0, 0,
            command.vx, command.vy, command.vz,
            0, 0, 0,
            command.yaw, 0,
        )

    def set_mode(self, name: str, keepalive: Command) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.send(keepalive)
            self.master.set_mode(name)
            time.sleep(self.c.dt)
            if name in self.snapshot().mode.upper():
                return
        raise TimeoutError(f"PX4 did not enter {name}")

    def wait_ready(self) -> State:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state, now = self.snapshot(), time.monotonic()
            if state.fresh(now, self.c.telemetry_timeout_s):
                if state.armed or state.landed != 1:
                    raise RuntimeError("Initialization requires disarmed ON_GROUND state")
                return state
            time.sleep(0.05)
        raise TimeoutError("Fresh initialization data not received")

    def arm(self, keepalive: Command) -> None:
        state = self.snapshot()
        if state.armed or state.landed != 1:
            raise RuntimeError("Refusing unsafe arm request")
        self.command_long(400, 1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.send(keepalive)
            if self.snapshot().armed:
                return
            time.sleep(0.05)
        raise TimeoutError("PX4 did not arm")

    def disarm(self) -> None:
        state = self.snapshot()
        if state.landed != 1:
            raise RuntimeError("Refusing to disarm before ON_GROUND")
        if state.armed:
            self.command_long(400, 0)

    def run(self) -> None:
        self.connect()
        initial = self.wait_ready()
        mission = Mission(self.c, initial, self.heading_deg)
        zero = Command(0, 0, 0, initial.yaw, initial.x, initial.y, initial.z)
        for _ in range(round(2 / self.c.dt)):
            self.send(zero)
            time.sleep(self.c.dt)
        self.set_mode("OFFBOARD", zero)
        self.arm(zero)

        samples = violations = late_streak = 0
        low = self.c.dt * (1 - self.c.timing_tolerance)
        high = self.c.dt * (1 + self.c.timing_tolerance)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("time", "actual_dt", "phase", "x", "y", "z", "vx", "vy", "vz",
                 "landed", "target_x", "target_y", "target_z", "cmd_vx", "cmd_vy",
                 "cmd_vz")
            )
            start = previous = next_tick = time.monotonic()
            while mission.phase != "DONE":
                now, state = time.monotonic(), self.snapshot()
                actual_dt = self.c.dt if samples == 0 else now - previous
                previous = now
                if not state.fresh(now, self.c.telemetry_timeout_s):
                    raise RuntimeError("Telemetry became stale")
                mode_ok = "OFFBOARD" in state.mode.upper()
                land_handoff_ok = mission.phase == "LAND_HANDOFF" and (
                    "LAND" in state.mode.upper() or state.landed == 1
                )
                if not (mode_ok or land_handoff_ok):
                    raise RuntimeError("PX4 left OFFBOARD")
                if not state.armed and state.landed != 1:
                    raise RuntimeError("Vehicle disarmed in flight")
                if now - start > self.c.mission_timeout_s:
                    raise TimeoutError("Mission timeout")
                on_time = low - 1e-12 <= actual_dt <= high + 1e-12
                late_streak = 0 if on_time else late_streak + 1
                violations += not on_time
                if late_streak >= 3:
                    raise RuntimeError("10 Hz timing left its 10% band three times")

                command = mission.step(state)  # Always advances by exactly 0.1 s.
                self.send(command)
                if mission.phase == "LAND_HANDOFF" and "LAND" not in state.mode.upper():
                    self.master.set_mode("LAND")
                writer.writerow(
                    (now - start, actual_dt, mission.phase, state.x, state.y, state.z,
                     state.vx, state.vy, state.vz, state.landed,
                     command.target_x, command.target_y, command.target_z,
                     command.vx, command.vy, command.vz)
                )
                samples += 1
                if samples % 10 == 0:
                    stream.flush()
                    print(
                        f"{mission.phase:10s} x={state.x:7.2f} "
                        f"AGL={mission.agl(state):5.2f} vz={command.vz:4.2f}"
                    )
                next_tick += self.c.dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        self.send(zero)
        self.disarm()
        print(f"Mission complete. Timing violations: {violations}/{samples}")

    def failure_land(self) -> None:
        if self.master and self.snapshot().armed:
            print("Failure detected. Handing control to PX4 AUTO.LAND.")
            for _ in range(20):
                self.master.set_mode("LAND")
                if "LAND" in self.snapshot().mode.upper():
                    break
                time.sleep(0.1)

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple PX4 SITL mission")
    parser.add_argument("--connection", default="udp:127.0.0.1:14540")
    parser.add_argument("--heading-deg", type=float, default=0.0, help="0=north, 90=east")
    parser.add_argument("--log", type=Path, default=Path("flight_log.csv"))
    args = parser.parse_args()
    px4 = PX4(Config(connection=args.connection), args.heading_deg, args.log)
    try:
        px4.run()
    except BaseException:
        px4.failure_land()
        raise
    finally:
        px4.stop()


if __name__ == "__main__":
    main()
