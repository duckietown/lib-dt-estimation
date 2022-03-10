from math import floor, sqrt
from typing import List

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy, multivariate_normal

from . import ILaneFilter

from .types import Segment, SegmentColor


class LaneFilterHistogram(ILaneFilter):
    """Generates an estimate of the lane pose.

    Creates and maintain a histogram grid filter to estimate the lane pose.

    Lane pose is defined as the tuple (`d`, `phi`) : lateral deviation and angulare deviation
    from the center of the lane.

    - Predict step : Uses the estimated linear and angular velocities to predict the change in
    the lane pose.

    - Update Step : The filter receives a segment list. For each segment, it extracts the
    corresponding lane pose "votes", and adds it to the corresponding part of the histogram.

    Best estimate correspond to the slot of the histogram with the highest voted value.
    """

    def __init__(self, *args, **kwargs):
        super(LaneFilterHistogram, self).__init__(*args, **kwargs)

        # Additional variables
        self.red_to_white = False
        self.use_yellow = True
        self.range_est_min = 0
        self.filtered_segments = []
        # initialize
        self.initialize()

    def initialize(self):
        pos = np.empty(self.d.shape + (2,))
        pos[:, :, 0] = self.d
        pos[:, :, 1] = self.phi
        RV = multivariate_normal(self.mean_0, self.cov_0)
        self.belief = RV.pdf(pos)

    def get_entropy(self):
        belief = self.belief
        s = entropy(belief.flatten())
        return s

    def predict(self, delta_t: float, v: float, w: float):
        d_t = self.d + v * delta_t * np.sin(self.phi)
        phi_t = self.phi + w * delta_t

        p_belief = np.zeros(self.belief.shape)

        # there has got to be a better/cleaner way to do this - just applying the process model to
        # translate each cell value
        for i in range(self.belief.shape[0]):
            for j in range(self.belief.shape[1]):
                if self.belief[i, j] > 0:
                    if (
                            d_t[i, j] > self.d_max
                            or d_t[i, j] < self.d_min
                            or phi_t[i, j] < self.phi_min
                            or phi_t[i, j] > self.phi_max
                    ):
                        continue

                    i_new = int(floor((d_t[i, j] - self.d_min) / self.delta_d))
                    j_new = int(floor((phi_t[i, j] - self.phi_min) / self.delta_phi))

                    p_belief[i_new, j_new] += self.belief[i, j]

        s_belief = np.zeros(self.belief.shape)
        gaussian_filter(p_belief, self.cov_mask, output=s_belief, mode="constant")

        if np.sum(s_belief) == 0:
            return
        self.belief = s_belief / np.sum(s_belief)

    def update(self, segments: List[Segment]):
        # prepare the segments for each belief array
        segmentsArray = self._prepare_segments(segments)
        # generate all belief arrays

        measurement_likelihood = self._generate_measurement_likelihood(segmentsArray)

        if measurement_likelihood is not None:
            self.belief = np.multiply(self.belief, measurement_likelihood)
            if np.sum(self.belief) == 0:
                self.belief = measurement_likelihood
            else:
                self.belief /= np.sum(self.belief)

    def get_estimate(self):
        maxids = np.unravel_index(self.belief.argmax(), self.belief.shape)
        d_max = self.d_min + (maxids[0] + 0.5) * self.delta_d
        phi_max = self.phi_min + (maxids[1] + 0.5) * self.delta_phi
        return d_max, phi_max

    def get_max(self):
        return self.belief.max()

    def get_inlier_segments(self, segments: List[Segment], d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self._generate_vote(segment)
            if abs(d_s - d_max) < self.delta_d and abs(phi_s - phi_max) < self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    def _generate_measurement_likelihood(self, segments: List[Segment]):
        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(self.d.shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self._generate_vote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] += 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None
        measurement_likelihood /= np.sum(measurement_likelihood)
        return measurement_likelihood

    # generate a vote for one segment
    def _generate_vote(self, segment: Segment):
        p1 = segment.points[0].as_array()
        p2 = segment.points[1].as_array()
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if l1 < 0:
            l1 = -l1
        if l2 < 0:
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == SegmentColor.WHITE:  # right lane is white
            if p1[0] > p2[0]:  # right edge of white lane
                d_i -= self.linewidth_white
            else:  # left edge of white lane

                d_i = -d_i

                phi_i = -phi_i
            d_i -= self.lanewidth / 2

        elif segment.color == SegmentColor.YELLOW:  # left lane is yellow
            if p2[0] > p1[0]:  # left edge of yellow lane
                d_i -= self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    @staticmethod
    def _get_segment_distance(segment: Segment):
        """
        Gets the distance from the center of the robot to the center point of the given segment.

        Args:
            segment:

        Returns:

        """
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c ** 2 + y_c ** 2)

    def _prepare_segments(self, segments: List[Segment]):
        """
        Prepares the segments for the creation of the belief arrays.

        Args:
            segments:

        Returns:

        """
        filtered_segments = []
        for segment in segments:
            # Optional transform from RED to WHITE
            if self.red_to_white and segment.color == SegmentColor.RED:
                segment.color = SegmentColor.WHITE

            # Optional filtering out YELLOW
            if not self.use_yellow and segment.color == SegmentColor.YELLOW:
                continue

            # we don't care about RED ones for now
            if segment.color != SegmentColor.WHITE and segment.color != SegmentColor.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            # only consider points in a certain range from the robot for the position estimation
            point_range = self._get_segment_distance(segment)
            if self.range_est > point_range > self.range_est_min:
                filtered_segments.append(segment)

        return filtered_segments


__all__ = ["LaneFilterHistogram"]
