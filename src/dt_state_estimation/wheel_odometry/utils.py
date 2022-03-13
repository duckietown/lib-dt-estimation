import math


def angle_clamp(theta):
    if theta > 2 * math.pi:
        return theta - 2 * math.pi
    elif theta < -2 * math.pi:
        return theta + 2 * math.pi
    else:
        return theta
