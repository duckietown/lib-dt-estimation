import dataclasses
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Tuple

import numpy as np


class LaneFilterStatus(Enum):
    UNKNOWN = "UNKNOWN"
    LOST = "LOST"
    GOOD = "GOOD"
    STRUGGLING = "STRUGGLING"


class SegmentColor(Enum):
    WHITE = "white"
    YELLOW = "yellow"
    RED = "red"


@dataclasses.dataclass
class SegmentPoint:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclasses.dataclass
class Segment:
    points: List[SegmentPoint]
    color: SegmentColor


class ILaneFilter(metaclass=ABCMeta):

    def __init__(self,
                 mean_d_0: float = 0,
                 mean_phi_0: float = 0,
                 sigma_d_0: float = 0.1,
                 sigma_phi_0: float = 0.1,
                 delta_d: float = 0.02,
                 delta_phi: float = np.deg2rad(5),
                 d_max: float = 0.3,
                 d_min: float = -0.15,
                 phi_min: float = -np.deg2rad(85),
                 phi_max: float = np.deg2rad(85),
                 cov_v: float = 0.5,
                 linewidth_white: float = 0.05,
                 linewidth_yellow: float = 0.025,
                 lanewidth: float = 0.23,
                 min_max: float = 0.1,
                 sigma_d_mask: float = 1.0,
                 sigma_phi_mask: float = 2.0,
                 curvature_res: float = 0,
                 range_min: float = 0.2,
                 range_est: float = 0.33,
                 range_max: float = 0.6,
                 curvature_right: float = -0.054,
                 curvature_left: float = 0.025,
                 ):
        # store parameters
        self.mean_d_0: float = mean_d_0
        self.mean_phi_0: float = mean_phi_0
        self.sigma_d_0: float = sigma_d_0
        self.sigma_phi_0: float = sigma_phi_0
        self.delta_d: float = delta_d
        self.delta_phi: float = delta_phi
        self.d_max: float = d_max
        self.d_min: float = d_min
        self.phi_min: float = phi_min
        self.phi_max: float = phi_max
        self.cov_v: float = cov_v
        self.linewidth_white: float = linewidth_white
        self.linewidth_yellow: float = linewidth_yellow
        self.lanewidth: float = lanewidth
        self.min_max: float = min_max
        self.sigma_d_mask: float = sigma_d_mask
        self.sigma_phi_mask: float = sigma_phi_mask
        self.curvature_res: float = curvature_res
        self.range_min: float = range_min
        self.range_est: float = range_est
        self.range_max: float = range_max
        self.curvature_right: float = curvature_right
        self.curvature_left: float = curvature_left
        # public objects
        self.d, self.phi = \
            np.mgrid[
                self.d_min: self.d_max: self.delta_d,
                self.phi_min: self.phi_max: self.delta_phi
            ]

        self.d_pcolor, self.phi_pcolor = \
            np.mgrid[
                self.d_min: (self.d_max + self.delta_d): self.delta_d,
                self.phi_min: (self.phi_max + self.delta_phi): self.delta_phi,
            ]

        self.belief = np.empty(self.d.shape)

        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.cov_mask = [self.sigma_d_mask, self.sigma_phi_mask]
        # private objects
        self._status: LaneFilterStatus = LaneFilterStatus.UNKNOWN

    @property
    def status(self) -> LaneFilterStatus:
        return self._status

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self, dt, v, w):
        pass

    @abstractmethod
    def update(self, segment_list):
        """
        segment list: a list of Segment objects
        """

    @abstractmethod
    def get_estimate(self) -> Tuple[float, float]:
        """
        Returns a tuple of two estimated values, `d` (lateral offset) and
        `phi` (heading offset).
        """

    @abstractmethod
    def get_entropy(self) -> float:
        pass

    @abstractmethod
    def get_max(self) -> float:
        pass

    @abstractmethod
    def get_inlier_segments(self, segments: List[Segment], d_max, phi_max):
        pass


__all__ = [
    "LaneFilterStatus",
    "ILaneFilter",
    "Segment",
    "SegmentColor",
    "SegmentPoint"
]
