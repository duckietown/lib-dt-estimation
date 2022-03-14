import numpy as np

from dt_state_estimation.wheel_odometry import WheelOdometer

# distance between wheels
wheel_baseline: float = 0.1
# ticks per revolution
encoder_resolution: int = 100
# wheel radius
wheel_radius: float = 0.025

wheel_length = 2 * np.pi * wheel_radius
ticks_per_meter = encoder_resolution / wheel_length


def _odometer():
    odometer = WheelOdometer(ticks_per_meter, wheel_baseline)
    odometer.update(0, 0, timestamp=0.0)
    # noinspection PyProtectedMember
    # odometer._logger.setLevel(logging.DEBUG)
    return odometer


def test_no_data():
    odometer = WheelOdometer(ticks_per_meter, wheel_baseline)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    assert pose is None
    assert velocity is None


def test_no_motion():
    odometer = _odometer()
    # first reading: the odometer should have NO estimate
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    assert pose is None
    assert velocity is None
    # second reading: the odometer should have an estimate but this should be zero (no motion)
    odometer.update(0, 0, timestamp=1.0)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    assert pose.x == 0 and pose.y == 0 and pose.theta == 0
    assert velocity.v == 0 and velocity.w == 0


def test_1_wheel_rotation_forward():
    odometer = _odometer()
    odometer.update(encoder_resolution, encoder_resolution, timestamp=1.0)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    assert equal(pose.x, wheel_length) and pose.y == 0 and pose.theta == 0
    assert equal(velocity.v, wheel_length / 1.0) and velocity.w == 0


def test_1_meter_forward():
    odometer = _odometer()
    odometer.update(ticks_per_meter, ticks_per_meter, timestamp=1.0)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    assert equal(pose.x, 1.0) and pose.y == 0 and pose.theta == 0
    assert equal(velocity.v, 1.0) and velocity.w == 0


def test_1_full_wheels_rotation_left():
    odometer = _odometer()
    odometer.update(-encoder_resolution, encoder_resolution, timestamp=1.0)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    # rotating both wheels in opposite direction for a full rotation, rotates the chassis by 180deg
    assert pose.x == 0 and pose.y == 0 and equal(pose.theta, np.pi)
    assert velocity.v == 0 and equal(velocity.w, np.pi / 1.0)


def test_1_full_wheels_rotation_right():
    odometer = _odometer()
    odometer.update(encoder_resolution, -encoder_resolution, timestamp=1.0)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    # rotating both wheels in opposite direction for a full rotation, rotates the chassis by 180deg
    assert pose.x == 0 and pose.y == 0 and equal(pose.theta, -np.pi)
    assert velocity.v == 0 and equal(velocity.w, -np.pi / 1.0)


def test_1_full_circle_left():
    odometer = _odometer()
    circle_length = 2 * np.pi * wheel_baseline
    ticks_needed = ticks_per_meter * circle_length
    # we need to update the odometer at least 4 times in a full circle to make sure that the
    # tangential velocities at each quadrant cancel each other out.
    num_updates = 4
    for t, ticks in enumerate(np.linspace(0, ticks_needed, num_updates + 1)):
        odometer.update(0, ticks, timestamp=t + 1)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    # full circle (left) rotation, no change in position, 360deg change in orientation
    assert equal(pose.x, 0)
    assert equal(pose.y, 0)
    assert equal(pose.theta, 2 * np.pi)
    # angular rotation is simply `2pi / time`, linear is the tangential to the circle
    assert equal(velocity.w, 2 * np.pi / num_updates)
    assert equal(velocity.v, velocity.w * wheel_baseline / 2)


def test_1_full_circle_right():
    odometer = _odometer()
    circle_length = 2 * np.pi * wheel_baseline
    ticks_needed = ticks_per_meter * circle_length
    # we need to update the odometer at least 4 times in a full circle to make sure that the
    # tangential velocities at each quadrant cancel each other out.
    num_updates = 4
    for t, ticks in enumerate(np.linspace(0, ticks_needed, num_updates + 1)):
        odometer.update(ticks, 0, timestamp=t + 1)
    pose, velocity = odometer.get_estimate()
    print(pose, velocity)
    # full circle (right) rotation, no change in position, -360deg change in orientation
    assert equal(pose.x, 0)
    assert equal(pose.y, 0)
    assert equal(pose.theta, -2 * np.pi)
    # angular rotation is simply `-2pi / time`, linear is the tangential to the circle
    assert equal(velocity.w, -2 * np.pi / num_updates)
    assert equal(velocity.v, -velocity.w * wheel_baseline / 2)


def equal(a, b):
    return np.allclose(a, b)
