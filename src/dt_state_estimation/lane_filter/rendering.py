import io
import os

import numpy as np
import numpy.ma as ma
from reprep.graphics.filter_scale import scale

import cv2
import matplotlib.pyplot as plt

from dt_state_estimation.lane_filter import ILaneFilter
from typing import Tuple


BGRImage = np.ndarray
assets_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
background_image_fpath: str = os.path.join(assets_dir, "lane_filter_grid.png")
background_image: BGRImage = cv2.imread(background_image_fpath)


def plot_d_phi(d: float, phi: float, size: Tuple[int, int] = (-1, -1)):
    """
    Generates a debug image with the estimated pose of the robot drawn.

    Args:
        d (:obj:`float`): Estimated `d`
        phi (:obj:`float`): Estimated `phi`
        size (:obj:`tuple`): Size of the image to draw

    Returns:
        :obj:`numpy array`: an OpenCV image

    """
    size_x, size_y = size
    # 36cm is the total width depicted in the background image
    total_width_meters = 0.36
    # in the image, the origin is at 23.7cm from the left border
    origin_meters = 0.237

    # the axis of the rendered image and the axis of the robot are flipped
    d *= -1.0
    phi *= -1.0

    # start with a fresh copy of the background
    image = background_image.copy()

    # resize if needed
    if size_x > 0 and size_y > 0:
        image = cv2.resize(image, size)
    size_y, size_x, _ = image.shape

    # compute origin
    pixel_per_meter = size_x / total_width_meters
    origin_x_px = origin_meters * pixel_per_meter
    origin_y_px = size_y * 0.5

    # compute robot's location
    x = int(origin_x_px + d * pixel_per_meter)
    y = int(origin_y_px)

    # draw location
    cv2.circle(image, (x, y), 14, (0, 0, 200), 3)

    # simple 2D rotation, used to rotate the heading indicator
    R = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    # compute heading indicator line coordinates
    heading_bar = np.array([[0, 0, 1], [0, -40, 1]])
    heading_bar = np.dot(R, heading_bar.T).T[:, :2].astype(int) + [x, y]

    # draw heading
    cv2.line(image, tuple(heading_bar[0]), tuple(heading_bar[1]), (0, 0, 200), 5)
    # ---
    return image


def plot_belief(
        filter: ILaneFilter,
        bgcolor=(0, 204, 255),
        dpi=150,
        other_phi=None,
        other_d=None,
) -> BGRImage:
    """Returns a BGR image"""

    # get estimate
    d, phi = filter.get_estimate()

    bgcolor = tuple(x / 255.0 for x in bgcolor)
    figure = plt.figure(facecolor=bgcolor)

    f_d = lambda x: 100 * x
    f_phi = np.rad2deg
    # Where are the lanes?
    lane_width = filter.lanewidth
    d_max = filter.d_max
    d_min = filter.d_min
    phi_max = filter.phi_max
    phi_min = filter.phi_min
    delta_d = filter.delta_d
    delta_phi = filter.delta_phi

    # note transpose
    belief = filter.belief.copy()
    zeros = belief == 0

    belief[zeros] = np.nan

    belief_image = scale(belief, min_value=0)

    x = f_d(filter.d_pcolor)
    y = f_phi(filter.phi_pcolor)

    z = belief_image[:, :, 0]  # just R component
    z = ma.masked_array(z, zeros)

    plt.pcolor(x, y, np.ones(z.shape), cmap="Pastel1")

    plt.pcolor(x, y, z, cmap="gray")

    if other_phi is not None:
        for _phi, _d in zip(other_phi, other_d):
            plt.plot(
                f_d(_d),
                f_phi(_phi),
                "go",
                markersize=15,
                markeredgecolor="none",
                markeredgewidth=3,
                markerfacecolor="blue",
            )

    plt.plot(
        f_d(d),
        f_phi(phi),
        "go",
        markersize=20,
        markeredgecolor="magenta",
        markeredgewidth=3,
        markerfacecolor="none",
    )

    plt.plot(
        f_d(d),
        f_phi(phi),
        "o",
        markersize=2,
        markeredgecolor="none",
        markeredgewidth=0,
        markerfacecolor="magenta",
    )

    W = f_d(lane_width / 2)
    width_white = f_d(filter.linewidth_white)
    width_yellow = f_d(filter.linewidth_yellow)

    plt.plot([-W, -W], [f_phi(phi_min), f_phi(phi_max)], "w-")
    plt.plot([-W - width_white, -W - width_white], [f_phi(phi_min), f_phi(phi_max)], "k-")
    plt.plot([0, 0], [f_phi(phi_min), f_phi(phi_max)], "k-")
    plt.plot([+W, +W], [f_phi(phi_min), f_phi(phi_max)], "-", color="yellow")
    plt.plot([+W + width_yellow, +W + width_yellow], [f_phi(phi_min), f_phi(phi_max)], "-",
             color="yellow")
    s = ""
    s += f"status = {filter.status.value}"
    s += f"\nphi = {f_phi(phi):.1f} deg"
    s += f"\nd = {f_d(d):.1f} cm"
    s += f"\nentropy = {filter.get_entropy():.4f}"
    s += f"\nmax = {belief.max():.4f}"
    s += f"\nmin = {belief.min():.4f}"

    if other_phi is not None:
        s += "\n Other answers:"
        for _phi, _d in zip(other_phi, other_d):
            s += f"\nphi = {f_phi(_phi):.1f} deg"
            s += f"\nd = {f_d(_d):.1f} cm"

    y = f_phi(phi_max) - 10
    args = dict(rotation=-90, color="white")
    annotate = True
    if annotate:
        plt.annotate(s, xy=(0.05, 0.99), xycoords="figure fraction")
        plt.annotate("in middle of right lane", xy=(0, y), **args)
        plt.annotate("on right white tape", xy=(-W, y), **args)
        plt.annotate("on left yellow tape", xy=(+W, y), **args)
        plt.annotate("in other lane", xy=(+W * 1.3, y), **args)

    plt.axis([f_d(d_min), f_d(d_max), f_phi(phi_min), f_phi(phi_max)])

    plt.ylabel(f"phi: orientation (deg); cell = {f_phi(delta_phi):.1f} deg")
    plt.xlabel(f"d: distance from center line (cm); cell = {f_d(delta_d):.1f} cm")

    plt.gca().invert_xaxis()

    bgr = _plt_to_bgr(figure, dpi)
    plt.close()
    return bgr


def _plt_to_bgr(figure, dpi) -> BGRImage:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0.15,
                transparent=True, facecolor=figure.get_facecolor())
    buf.seek(0)
    png = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    return cv2.imdecode(png, cv2.IMREAD_COLOR)


__all__ = [
    "plot_belief",
    "plot_d_phi"
]
