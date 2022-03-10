import tempfile

import numpy as np
import numpy.ma as ma
from reprep.graphics.filter_scale import scale

from matplotlib import pylab
import cv2

from dt_state_estimation.lane_filter import ILaneFilter

BGRImage = np.ndarray

DPI = 120


def plot_belief(
        lane_filter: ILaneFilter,
        belief,
        phi,
        d,
        bgcolor=(0, 204, 255),
        other_phi=None,
        other_d=None,
) -> BGRImage:
    """Returns a BGR image"""
    bgcolor = tuple(x / 255.0 for x in bgcolor)
    figure = pylab.figure(facecolor=bgcolor)

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

    pylab.pcolor(x, y, np.ones(z.shape), cmap="Pastel1")

    pylab.pcolor(x, y, z, cmap="gray")

    if other_phi is not None:
        for _phi, _d in zip(other_phi, other_d):
            pylab.plot(
                f_d(_d),
                f_phi(_phi),
                "go",
                markersize=15,
                markeredgecolor="none",
                markeredgewidth=3,
                markerfacecolor="blue",
            )

    pylab.plot(
        f_d(d),
        f_phi(phi),
        "go",
        markersize=20,
        markeredgecolor="magenta",
        markeredgewidth=3,
        markerfacecolor="none",
    )

    pylab.plot(
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

    pylab.plot([-W, -W], [f_phi(phi_min), f_phi(phi_max)], "k-")
    pylab.plot([-W - width_white, -W - width_white], [f_phi(phi_min), f_phi(phi_max)], "k-")
    pylab.plot([0, 0], [f_phi(phi_min), f_phi(phi_max)], "k-")
    pylab.plot([+W, +W], [f_phi(phi_min), f_phi(phi_max)], "y-")
    pylab.plot([+W + width_yellow, +W + width_yellow], [f_phi(phi_min), f_phi(phi_max)], "y-")
    s = ""
    s += f"status = {lane_filter.status}"
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
        pylab.annotate(s, xy=(0.7, 0.35), xycoords="figure fraction")
        pylab.annotate("in middle of right lane", xy=(0, y), **args)
        pylab.annotate("on right white tape", xy=(-W, y), **args)
        pylab.annotate("on left yellow tape", xy=(+W, y), **args)
        pylab.annotate("in other lane", xy=(+W * 1.3, y), **args)

    pylab.axis([f_d(d_min), f_d(d_max), f_phi(phi_min), f_phi(phi_max)])

    pylab.ylabel(f"phi: orientation (deg); cell = {f_phi(delta_phi):.1f} deg")
    pylab.xlabel(f"d: distance from center line (cm); cell = {f_d(delta_d):.1f} cm")

    return _pylab_to_bgr(figure)


def _pylab_to_bgr(figure) -> BGRImage:
    temp_file = tempfile.NamedTemporaryFile(suffix=".png")
    # save image to temporary file
    pylab.savefig(temp_file.name, dpi=DPI, bbox_inches='tight', pad_inches=0.01,
                  transparent=True, facecolor=figure.get_facecolor())
    # read temporary file back
    with open(temp_file.name, "rb") as f:
        png_data = f.read()
    # close (thus clean) temporary file
    temp_file.close()
    # convert PNG to BGR
    bgr = _bgr_from_png(png_data)
    # clean up and return the BGR image
    pylab.close()
    return bgr


def _bgr_from_png(png):
    """ Returns an OpenCV BGR image from PNG bytes """
    s = np.fromstring(png, np.uint8)
    bgr = cv2.imdecode(s, cv2.IMREAD_COLOR)
    if bgr is None:
        msg = 'Could not decode image (cv2.imdecode returned None). '
        msg += 'This is usual a sign of data corruption.'
        raise ValueError(msg)
    return bgr


__all__ = ["plot_belief"]
