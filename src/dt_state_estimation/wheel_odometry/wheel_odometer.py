import time
import logging
from threading import Semaphore
from typing import Tuple, Optional

import numpy as np

from dt_state_estimation.wheel_odometry.types import \
    IWheelOdometer, \
    Pose2DEstimate, \
    Velocity2DEstimate
from dt_state_estimation.wheel_odometry.utils import angle_clamp


class WheelOdometer(IWheelOdometer):
    """
    Performs odometry estimation using data from wheel encoders (aka deadreckoning).
    """

    def __init__(self, ticks_per_meter: float, wheel_baseline: float):
        super(WheelOdometer, self).__init__(ticks_per_meter, wheel_baseline)
        # current pose
        self._pose = Pose2DEstimate(0, 0, 0, 0)
        # linear/angular velocity
        self._velocity = Velocity2DEstimate(0, 0, 0)
        # temporary data
        self._left_ticks_last = None
        self._right_ticks_last = None
        self._timestamp_last = None
        self._has_estimate = None
        # others
        self._lock = Semaphore()
        self._logger = logging.getLogger("WheelOdometer")

    def initialize(self):
        pass

    def update(self, left_ticks: int, right_ticks: int, timestamp: float = None):
        if self._timestamp_last is None:
            # move cursor forward
            self._left_ticks_last = left_ticks
            self._right_ticks_last = right_ticks
            self._timestamp_last = timestamp
            return

        # timestamp is NOW if not given
        timestamp = timestamp or time.time()

        # compute delta_t between this reading and the previous
        dt = timestamp - self._timestamp_last

        # compute the motion of left and right wheels in number of ticks
        left_delta_ticks = left_ticks - self._left_ticks_last
        right_delta_ticks = right_ticks - self._right_ticks_last

        # compute the motion of left and right wheels in meters traveled
        left_distance = left_delta_ticks / self.ticks_per_meter
        right_distance = right_delta_ticks / self.ticks_per_meter

        # displacement in body-relative x-direction (assuming differential drive)
        delta_x = (left_distance + right_distance) / 2

        # change in heading
        delta_theta = (right_distance - left_distance) / self.wheel_baseline

        if dt < 1e-6:
            self._logger.warning(f"Time between readings ({dt:.5f}) is too small. Ignoring")
            return

        # update internal state
        with self._lock:
            # linear and angular velocities
            self._velocity.v = delta_x / dt
            self._velocity.w = delta_theta / dt

            self._logger.debug(
                f"Time = {timestamp} s; Dt = {dt:.5f} s;\n "
                f"\tLeft wheel: {left_ticks} ticks; {left_distance} meters;\n "
                f"\tRight wheel: {right_ticks} ticks; {right_distance} meters;\n "
                f"\tv: {self._velocity.v:.4f} m/s;\n "
                f"\tw: {np.rad2deg(self._velocity.w)} deg/s;"
            )

            print(delta_x, self._pose.theta, np.cos(self._pose.theta), np.sin(self._pose.theta))

            # update pose
            self._pose.theta = angle_clamp(self._pose.theta + delta_theta)
            self._pose.x = self._pose.x + delta_x * np.cos(self._pose.theta)
            self._pose.y = self._pose.y + delta_x * np.sin(self._pose.theta)

            # move cursor forward
            self._left_ticks_last = left_ticks
            self._right_ticks_last = right_ticks
            self._timestamp_last = timestamp
            self._has_estimate = True

    def get_estimate(self) -> Tuple[Optional[Pose2DEstimate], Optional[Velocity2DEstimate]]:
        if not self._has_estimate:
            return None, None
        with self._lock:
            return self._pose.copy(), self._velocity.copy()
