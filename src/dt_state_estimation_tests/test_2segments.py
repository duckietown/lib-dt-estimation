import numpy as np

from dt_state_estimation.lane_filter import LaneFilterHistogram
from dt_state_estimation.lane_filter.types import Segment, SegmentPoint, SegmentColor

segment_length: float = 0.05
lane_width: float = 0.225
half_lane: float = lane_width * 0.5


def _perform_test(lateral_shift: float, rotation: float, distance: float = 0.0,
                  delta_d: float = 0.02, delta_phi: float = 0.1, **kwargs):
    """
    Args:
        lateral_shift `float`:  A positive number shifts the segments to the robot's right,
                                effectively pushing the pose towards the left curb.
                                A negative number shifts the segments to the robot's left,
                                effectively pushing the pose towards the yellow line.

        rotation `float`:       A positive rotation indicates an angle going from the robot's
                                x-axis towards its y-axis.
                                Positive rotations here simulate left curves, where the white
                                markings appear in front of the robot, only white segments have
                                a vote in this case.
                                Negative rotations instead simulate right curves, where the yellow
                                markings appear in front of the robot, only yellow segments have
                                a vote in this case.
    """
    # create filter
    filter = LaneFilterHistogram(delta_d=delta_d, delta_phi=delta_phi, **kwargs)

    # simple 2D rotation, used to rotate the segments about the robot's origin
    R = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 1]
    ])

    # white segment (controls left curves, which is a positive `rotation` above)
    p0 = np.array([distance, lateral_shift - half_lane, 1])
    p1 = np.array([distance + segment_length, lateral_shift - half_lane, 1])

    p0 = np.dot(R, p0)
    p1 = np.dot(R, p1)

    white_segment = Segment(
        points=[
            SegmentPoint(p0[0], p0[1]),
            SegmentPoint(p1[0], p1[1]),
        ],
        color=SegmentColor.WHITE
    )

    # yellow segment (controls right curves, which is a negative `rotation` above)
    p0 = np.array([distance, lateral_shift + half_lane, 1])
    p1 = np.array([distance + segment_length, lateral_shift + half_lane, 1])

    p0 = np.dot(R, p0)
    p1 = np.dot(R, p1)

    yellow_segment = Segment(
        points=[
            SegmentPoint(p1[0], p1[1]),
            SegmentPoint(p0[0], p0[1]),
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


def test_2_segments_rotation():
    max_rotation_deg = 85
    filter_resolution_deg = 5.0
    for rotation_deg in np.arange(-max_rotation_deg, max_rotation_deg, filter_resolution_deg):
        rotation_rad = np.deg2rad(rotation_deg)
        _perform_test(0, rotation_rad)
