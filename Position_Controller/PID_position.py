#!/usr/bin/env python3
"""Condensed 10 Hz PX4 position controller derived from Vishnu's layout."""

import argparse
import csv
import threading
import time
from pathlib import Path

try:
    from pymavlink import mavutil
except ImportError:  # Unit tests only exercise the mission logic.
    mavutil = None

try:
    from .VehicleState import VehicleState, decode_px4_mode
    from .trajectory import Command, Config, Mission
except ImportError:  # Allows: python Position_Controller/PID_position.py
    from VehicleState import VehicleState, decode_px4_mode
    from trajectory import Command, Config, Mission


class PositionController:
    def __init__(self, config: Config, heading_deg: float, log_path: Path):
        if mavutil is None:
            raise RuntimeError("Run: python -m pip install -r requirements.txt")
        self.c, self.heading_deg, self.log_path = config, heading_deg, log_path
        self.state, self.lock = VehicleState(), threading.Lock()
        self.master = self.thread = None
        self.running = False

    def snapshot(self) -> VehicleState:
        with self.lock:
            return VehicleState(**vars(self.state))

    def command_long(self, command: int, *params: float) -> None:
        values = list(params) + [0.0] * (7 - len(params))
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            command,
            0,
            *values,
        )

    def connect(self) -> None:
        print(f"Connecting to {self.c.connection}")
        self.master = mavutil.mavlink_connection(self.c.connection)
        if self.master.wait_heartbeat(timeout=10) is None:
            raise TimeoutError("No PX4 heartbeat")
        self.running = True
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()
        for message_id in (32, 30, 245):
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

    def send(self, command: Command) -> None:
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            2503,  # Use velocity and yaw fields only.
            0,
            0,
            0,
            command.vx,
            command.vy,
            command.vz,
            0,
            0,
            0,
            command.yaw,
            0,
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

    def wait_ready(self) -> VehicleState:
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
                (
                    "time",
                    "actual_dt",
                    "phase",
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "landed",
                    "target_x",
                    "target_y",
                    "target_z",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_vz",
                )
            )
            start = previous = next_tick = time.monotonic()
            while mission.phase != "DONE":
                now, state = time.monotonic(), self.snapshot()
                actual_dt = self.c.dt if samples == 0 else now - previous
                previous = now
                if not state.fresh(now, self.c.telemetry_timeout_s):
                    raise RuntimeError("Telemetry became stale")
                mode_ok = "OFFBOARD" in state.mode.upper()
                landing_ok = mission.phase == "LAND_HANDOFF" and (
                    "LAND" in state.mode.upper() or state.landed == 1
                )
                if not (mode_ok or landing_ok):
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

                command = mission.step(state)
                self.send(command)
                if mission.phase == "LAND_HANDOFF" and "LAND" not in state.mode.upper():
                    self.master.set_mode("LAND")
                writer.writerow(
                    (
                        now - start,
                        actual_dt,
                        mission.phase,
                        state.x,
                        state.y,
                        state.z,
                        state.vx,
                        state.vy,
                        state.vz,
                        state.landed,
                        command.target_x,
                        command.target_y,
                        command.target_z,
                        command.vx,
                        command.vy,
                        command.vz,
                    )
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
    controller = PositionController(Config(connection=args.connection), args.heading_deg, args.log)
    try:
        controller.run()
    except BaseException:
        controller.failure_land()
        raise
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
