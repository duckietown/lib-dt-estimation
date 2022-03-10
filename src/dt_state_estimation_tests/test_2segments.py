import numpy as np

from dt_state_estimation.lane_filter import LaneFilterHistogram
from dt_state_estimation.lane_filter.types import Segment, SegmentPoint, SegmentColor

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

    print("Lateral shift introduced/estimated: ", lateral_shift, d_hat)
    print("Rotation introduced/estimated: ", lateral_shift, phi_hat)

    # error must be within 105% of filter's resolution (extra 5% accounts for numeric precision)
    assert d_hat + lateral_shift <= delta_d * 1.05
    assert phi_hat + rotation <= delta_phi * 1.05


def test_2_segments_centered():
    _perform_test(0, 0)


def test_2_segments_increasing_distances():
    for distance in np.arange(0, 5, 0.25):
        _perform_test(0, 0, distance=distance)


def test_2_segments_lateral_shift():
    max_shift = 0.15
    for lateral_shift in np.arange(-max_shift, max_shift, 0.01):
        _perform_test(lateral_shift, 0)
