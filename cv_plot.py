"""
contains utilities for plotting, including the radar plots
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def cv_plot_fig(row,image_mol,turn_imshow_axis_off=True,w_pad=-2.5,
                figsize=(7, 3),width_ratios =None,**kw_imshow):
    """

    :param row: molecular properties
    :param image_mol: image of the molecule
    :param turn_imshow_axis_off: if true, turn imshow axis off
    :param kw_imshow: passed to plt.imshow()
    :return: matplotlib figure
    """
    if width_ratios is None:
        width_ratios = [1, 3, 1]
    name_vals_bad = [
        ["Lipinski", row['Lipinski violations'] / 5],
        ["Deremits", row['Lilly demerits'] / 100],
        ["Mol. Wt", row["Molecular weight"] / 1000],
        ["Alerts", row["Total alert count"] / 4],
    ]
    name_vals_good = [
        ["CNS MPO", row["cns_mpo"] / 6],
        ["QED", row["QED"]],
        ["-cLogS", -row["log_s"]],
        ["-cLogD", -row["log_d"]],
    ]

    N = len(name_vals_bad)
    theta = radar_factory(N, frame='polygon')

    fig, axs = plt.subplots(figsize=figsize, subplot_kw=dict(projection="radar"),
                            nrows=1, ncols=3, width_ratios=width_ratios)
    axs[1].remove()
    axs[1] = fig.add_subplot(1, 3, 2)

    ax_bad, ax_im, ax_good = axs

    fig.subplots_adjust(hspace=-0.0)
    # Plot the four cases from the example data on separate Axes
    for ax, r_name__r_data, color in zip([ax_bad, ax_good],
                                         [name_vals_bad, name_vals_good],
                                         ["crimson", "forestgreen"]):
        r_name = [r[0] for r in r_name__r_data]
        r_data = [r[1] for r in r_name__r_data]
        ax.set_rgrids([])
        ax.plot(theta, r_data, color=color)
        ax.fill(theta, r_data, facecolor=color, alpha=0.25,
                label='_nolegend_')
        ax.set_varlabels(r_name, fontsize=10)
        # Adjust tick padding
        ax.tick_params(axis='x', pad=-6,
                       color=color)  # Adjust padding for radial ticks
        alignments = ["center", "right", "center", "left"]
        for ha, t in zip(alignments, ax.get_xticklabels()):
            plt.setp(t, horizontalalignment=ha)
    ax_im.imshow(image_mol, interpolation='spline36', **kw_imshow)
    if turn_imshow_axis_off:
        ax_im.axis('off')
        ax_im.set_facecolor("none")
    plt.tight_layout(w_pad=w_pad)
    return fig

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        """
        Defines radar transform from polar
        """
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        """
        Defines/registers radar axes
        """
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            """

            :param line: line to close
            :return: None
            """
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels,**kw):
            """

            :param labels: for the thetas
            :param kw: passed to set_thetagrids
            :return: nothing
            """
            self.set_thetagrids(np.degrees(theta), labels,**kw)

        def _gen_axes_patch(self):
            """

            :return: Circle or  RegularPolygon object
            """
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown value for {frame}")

        def _gen_axes_spines(self):
            """

            :return: spines for axes
            """
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError(f"Unknown value for {frame}")

    register_projection(RadarAxes)
    return theta
