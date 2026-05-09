import numpy as np


class Solution:

    def __init__(self, game):
        # This reference to the game object may only be used to call its public methods.
        self._game = game
        self._prev_steer = 0.0
        self._prev_throttle = 0.0

    @property
    def config(self):
        # Keep default simulator settings; no extra sensors are required for the first baseline.
        return {}

    def do_iteration(self, simulator_output, user_input=None):
        observation = simulator_output.get("observation")
        info = simulator_output.get("info", {})

        if observation is None:
            return user_input or [0.0, 0.0]

        action = self._compute_action(observation, info)
        return action

    def _compute_action(self, observation, info):
        obs = np.asarray(observation, dtype=np.float32)

        steer = self._lane_steering(obs)
        throttle = self._speed_control(obs, info)

        steer = self._smooth(steer, self._prev_steer, alpha=0.18)
        throttle = self._smooth(throttle, self._prev_throttle, alpha=0.22)

        self._prev_steer = steer
        self._prev_throttle = throttle

        return [float(np.clip(steer, -1.0, 1.0)), float(np.clip(throttle, -1.0, 1.0))]

    def _lane_steering(self, obs):
        # Default observation uses 19 state dims + 240 lidar rays.
        # The lane offset is one of the early state values and is expected to be centered around 0.5.
        if obs.shape[0] >= 9:
            lane_offset = float(obs[8])
        else:
            lane_offset = 0.5

        # Positive offset means the car is to the right of lane center, negative to the left.
        error = lane_offset - 0.5
        steer = -2.0 * error

        # Make steering gentler for small deviations.
        return steer * min(1.0, max(0.3, abs(error) * 2.5))

    def _speed_control(self, obs, info):
        if info.get("crash", False) or info.get("out_of_road", False):
            return -1.0

        speed = self._extract_speed(obs)
        desired_speed = 0.45

        if self._obstacle_ahead(obs):
            # If there is something close in front, brake.
            return -0.8

        throttle = (desired_speed - speed) * 0.9
        return float(np.clip(throttle, -1.0, 1.0))

    def _extract_speed(self, obs):
        # Speed is part of the state observation and is normalized to [0, 1].
        if obs.shape[0] >= 4:
            return float(obs[3])
        return 0.0

    def _obstacle_ahead(self, obs):
        if obs.shape[0] < 240:
            return False

        lidar = obs[-240:]
        # The lidar vector starts at the vehicle head and goes clockwise.
        # Use the narrow front sector of +/- 15 rays.
        center = 0
        sector = np.concatenate((lidar[-15:], lidar[:15]))
        return float(np.min(sector)) < 0.28

    def _smooth(self, target, previous, alpha):
        return previous + alpha * (target - previous)
