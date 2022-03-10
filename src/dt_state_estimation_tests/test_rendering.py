import numpy as np

from dt_state_estimation.lane_filter import LaneFilterHistogram
from dt_state_estimation.lane_filter.types import Segment, SegmentPoint, SegmentColor

from dt_state_estimation.lane_filter.rendering import plot_belief

segment_length: float = 0.05
lane_width: float = 0.225
half_lane: float = lane_width * 0.5


def _perform_test(lateral_shift: float, rotation: float, distance: float = 0.0,
                  delta_d: float = 0.02, delta_phi: float = 0.1, **kwargs):
    # create filter
    filter = LaneFilterHistogram(delta_d=delta_d, delta_phi=delta_phi, **kwargs)
    # construct two segments
    white_segment = Segment(
        points=[
            SegmentPoint(distance, lateral_shift - half_lane),
            SegmentPoint(distance + segment_length, lateral_shift - half_lane),
        ],
        color=SegmentColor.WHITE
    )
    yellow_segment = Segment(
        points=[
            SegmentPoint(distance + segment_length, half_lane + lateral_shift),
            SegmentPoint(distance, half_lane + lateral_shift),
        ],
        color=SegmentColor.YELLOW
    )
    # apply update
    filter.update([white_segment, yellow_segment])
    d_hat, phi_hat = filter.get_estimate()

    # plot belief
    bgr = plot_belief(filter, filter.belief, phi_hat, d_hat)
    print(bgr.shape)


def test_2_segments_centered():
    _perform_test(0, 0)
