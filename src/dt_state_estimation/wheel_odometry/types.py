import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Tuple


@dataclasses.dataclass
class PoseEstimate:
    x: float
    y: float
    z: float
    q: float

    def copy(self) -> 'PoseEstimate':
        return type(self).__init__(**dataclasses.asdict(self))


@dataclasses.dataclass
class VelocityEstimate:
    v: float
    w: float

    def copy(self) -> 'VelocityEstimate':
        return type(self).__init__(**dataclasses.asdict(self))


class IWheelOdometer(metaclass=ABCMeta):

    def __init__(self,
                 encoder_stale_dt: float,
                 ticks_per_meter: float,
                 wheel_base: float
                 ):
        """
        Performs odometry estimation using data from wheel encoders (aka deadreckoning).

        Args:
            # parameters for deadreckoning_node

            encoder_stale_dt (:obj:`float`):    Max allowable age of last encoder message,
                                                after which linear and angular velocities
                                                are set to zero

            ticks_per_meter (:obj:`int`):       Should be specified elsewhere

            wheel_base (:obj:`float`):          Lateral distance between the center of the
                                                two wheels (in meters)

        """
        # store parameters
        self.encoder_stale_dt: float = encoder_stale_dt
        self.ticks_per_meter: float = ticks_per_meter
        self.wheel_base: float = wheel_base

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def update(self, left_ticks: int, right_ticks: int, time: float = None):
        """
        Args:
            left_ticks `int`:  Number of ticks counted so far by the left wheel encoder.
            right_ticks `int`:  Number of ticks counted so far by the right wheel encoder.
            time `float`:       Time in seconds

        """

    @abstractmethod
    def get_estimate(self) -> Tuple[PoseEstimate, VelocityEstimate]:
        """
        Returns a pose estimate and a velocity estimate based on the data processed so far.
        """


__all__ = [
    "IWheelOdometer",
    "PoseEstimate",
    "VelocityEstimate"
]
