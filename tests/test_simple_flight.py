import math
import unittest
from types import SimpleNamespace

from simple_flight import Config, Mission, State, decode_px4_mode


class SimpleFlightTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.state = State(z=0.0, yaw=0.2, landed=1)

    def test_confirmed_values(self):
        self.assertEqual(self.config.hz, 10.0)
        self.assertEqual(self.config.dt, 0.1)
        self.assertEqual(self.config.distance_m, 200.0)
        self.assertEqual(self.config.cruise_m_s, 5.0)
        self.assertEqual(self.config.slow_below_m, 5.0)
        self.assertEqual(self.config.slow_descent_m_s, 0.1)

    def test_timing_band_is_plus_or_minus_ten_percent(self):
        low = self.config.dt * (1 - self.config.timing_tolerance)
        high = self.config.dt * (1 + self.config.timing_tolerance)
        self.assertAlmostEqual(low, 0.09)
        self.assertAlmostEqual(high, 0.11)

    def test_trajectory_is_exactly_200_m_and_never_exceeds_5_m_s(self):
        mission = Mission(self.config, self.state, heading_deg=0.0)
        mission.phase = "TRAJECTORY"
        state = State(z=-8.0, landed=2)
        previous_speed = 0.0
        for _ in range(3000):
            command = mission.step(state)
            self.assertLessEqual(mission.reference_speed, 5.0)
            self.assertLessEqual(
                abs(mission.reference_speed - previous_speed),
                self.config.acceleration_m_s2 * self.config.dt + 1e-12,
            )
            previous_speed = mission.reference_speed
            state.x = command.target_x
            state.vx = mission.reference_speed
            if mission.progress == 200.0:
                break
        self.assertEqual(mission.progress, 200.0)

    def test_heading_rotates_the_200_m_endpoint(self):
        mission = Mission(self.config, self.state, heading_deg=90.0)
        self.assertAlmostEqual(mission.end_x, 0.0, places=7)
        self.assertAlmostEqual(mission.end_y, 200.0, places=7)

    def test_descent_slows_at_five_metres_agl(self):
        mission = Mission(self.config, self.state, heading_deg=0.0)
        mission.phase = "LAND_FAST"
        self.assertEqual(mission.step(State(x=200.0, z=-5.01, landed=2)).vz, 0.5)
        self.assertEqual(mission.step(State(x=200.0, z=-5.0, landed=2)).vz, 0.1)
        self.assertEqual(mission.phase, "LAND_SLOW")

    def test_touchdown_hands_control_to_px4_land(self):
        mission = Mission(self.config, self.state, heading_deg=0.0)
        mission.phase = "LAND_SLOW"
        command = mission.step(State(x=200.0, z=-0.15, landed=2))
        self.assertEqual(mission.phase, "LAND_HANDOFF")
        self.assertEqual(command.vz, 0.0)

    def test_landing_uses_known_initial_surface_height(self):
        initial = State(x=10.0, y=20.0, z=3.0, yaw=math.pi / 2, landed=1)
        mission = Mission(self.config, initial, heading_deg=0.0)
        self.assertEqual(mission.agl(State(z=-2.0)), 5.0)

    def test_px4_mode_decoder_handles_current_heartbeat_flags(self):
        offboard = SimpleNamespace(custom_mode=6 << 16)
        land = SimpleNamespace(custom_mode=(6 << 24) | (4 << 16))
        self.assertEqual(decode_px4_mode(offboard), "OFFBOARD")
        self.assertEqual(decode_px4_mode(land), "LAND")


if __name__ == "__main__":
    unittest.main()
