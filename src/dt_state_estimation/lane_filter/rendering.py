import io

import numpy as np
import numpy.ma as ma
from reprep.graphics.filter_scale import scale

import cv2
import matplotlib.pyplot as plt

from dt_state_estimation.lane_filter import ILaneFilter

BGRImage = np.ndarray


def plot_belief(
        lane_filter: ILaneFilter,
        belief,
        phi,
        d,
        bgcolor=(0, 204, 255),
        dpi=150,
        other_phi=None,
        other_d=None,
) -> BGRImage:
    """Returns a BGR image"""
    bgcolor = tuple(x / 255.0 for x in bgcolor)
    figure = plt.figure(facecolor=bgcolor)

    f_d = lambda x: 100 * x
    f_phi = np.rad2deg
    # Where are the lanes?
    lane_width = lane_filter.lanewidth
    d_max = lane_filter.d_max
    d_min = lane_filter.d_min
    phi_max = lane_filter.phi_max
    phi_min = lane_filter.phi_min
    delta_d = lane_filter.delta_d
    delta_phi = lane_filter.delta_phi

    # note transpose
    belief = belief.copy()
    zeros = belief == 0

    belief[zeros] = np.nan

    belief_image = scale(belief, min_value=0)

    x = f_d(lane_filter.d_pcolor)
    y = f_phi(lane_filter.phi_pcolor)

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
    width_white = f_d(lane_filter.linewidth_white)
    width_yellow = f_d(lane_filter.linewidth_yellow)

    plt.plot([-W, -W], [f_phi(phi_min), f_phi(phi_max)], "w-")
    plt.plot([-W - width_white, -W - width_white], [f_phi(phi_min), f_phi(phi_max)], "k-")
    plt.plot([0, 0], [f_phi(phi_min), f_phi(phi_max)], "k-")
    plt.plot([+W, +W], [f_phi(phi_min), f_phi(phi_max)], "-", color="yellow")
    plt.plot([+W + width_yellow, +W + width_yellow], [f_phi(phi_min), f_phi(phi_max)], "-",
             color="yellow")
    s = ""
    s += f"status = {lane_filter.status.value}"
    s += f"\nphi = {f_phi(phi):.1f} deg"
    s += f"\nd = {f_d(d):.1f} cm"
    s += f"\nentropy = {lane_filter.get_entropy():.4f}"
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


__all__ = ["plot_belief"]
