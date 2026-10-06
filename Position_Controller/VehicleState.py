"""Latest vehicle state received from PX4."""

from dataclasses import dataclass

try:
    from pymavlink import mavutil
except ImportError:  # Unit tests do not require a PX4 connection.
    mavutil = None


def decode_px4_mode(message) -> str:
    """Decode the PX4 custom mode without relying on base-mode flags."""
    main_mode = (message.custom_mode >> 16) & 0xFF
    sub_mode = (message.custom_mode >> 24) & 0xFF
    if main_mode == 6:
        return "OFFBOARD"
    if main_mode == 4 and sub_mode == 6:
        return "LAND"
    return mavutil.mode_string_v10(message) if mavutil else "UNKNOWN"


@dataclass
class VehicleState:
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
