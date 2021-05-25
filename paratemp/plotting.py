"""
Module for plotting functions and utilities.


"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2018.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2018 Thomas J. Heavey IV                                   #
#                                                                      #
# Licensed under the Apache License, Version 2.0 (the "License");      #
# you may not use this file except in compliance with the License.     #
# You may obtain a copy of the License at                              #
#                                                                      #
#    http://www.apache.org/licenses/LICENSE-2.0                        #
#                                                                      #
# Unless required by applicable law or agreed to in writing, software  #
# distributed under the License is distributed on an "AS IS" BASIS,    #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or      #
# implied.                                                             #
# See the License for the specific language governing permissions and  #
# limitations under the License.                                       #
#                                                                      #
########################################################################

from __future__ import absolute_import, division, print_function

import math

import matplotlib as mpl
from matplotlib import pyplot as plt
from six.moves import range

from .utils import calc_fes_1d, _parse_ax_input
from .exceptions import InputError


__all__ = ["fes_array_3_legend", "plot_dist_array", "fes_1d"]


def fes_1d(
    x,
    temp,
    ax=None,
    bins=None,
    xlabel=r"distance / $\mathrm{\AA}$",
    data=None,
    **kwargs
):
    """
    Make FES of some time series data

    :type x: Iterable or str
    :param x: Data to form the FES from.
        If a string is given, the data will be taken from `data[x]` and `data`
        must also be given.

    :param float temp: Temperature for Boltzmann weighting
        calculation.

    :type bins: int or Sequence[int or float] or str
    :param bins: Default: None. The bins argument to be passed to
        np.histogram

    :param str xlabel: Default: 'distance / $\\mathrm{\\AA}$'. The label for
        the x axis.

    :type ax: matplotlib.axes.Axes
    :param ax: Default: None. The axes objects on which to make the plots.
        If None is supplied, new axes objects will be created.

    :param data: Default: None.
        If given, this must be an object that can be indexed by `x` to give
        the series from which the FES should be made.

        For example, these are equivalent:

        >>> fes_1d(data[x], temp)

        >>> fes_1d(x, temp, data=data)

    :param kwargs: keyword arguments to pass to the plotter

    :rtype: Tuple(np.ndarray, np.ndarray, matplotlib.lines.Line2D,
        matplotlib.figure.Figure, matplotlib.axes.Axes)

    :return: The delta G values, the bin centers, the lines object, the
        figure and the axes
    """
    if data is not None:
        _x = data[x]
    else:
        _x = x
    _fig, _ax = _parse_ax_input(ax)
    delta_g, bin_mids = calc_fes_1d(_x, temp=temp, bins=bins)
    lines = _ax.plot(bin_mids, delta_g, **kwargs)
    _ax.set_ylabel(r"$\Delta G$ / (kcal / mol)")
    _ax.set_xlabel(xlabel)
    return delta_g, bin_mids, lines, _fig, _ax


def fes_array_3_legend(data, temp, labels=None, axes=None, bins=None, **kwargs):
    """

    :param pd.DataFrame data: A dataframe with the data to be transformed
        into the FES. It needs to have at least three columns. If `labels` is
        given, these must correspond to columns in `data`. If `labels` is
        None, the first three columns of `data` will be used with their column
        names as the labels.
    :param float temp: Temperature at which to calculate the free energy
        surface. This should be the temperature at which the simulation was run.
    :param Iterable[str] labels: An iterable of at least length three or None.
        If this is not None, the first three elements of it will be used as
        column names to pick with data to plot from `data`.
        If it is None, the first three column names from `data` will be used.
        These values will also be used as labels for the legend.
    :param np.array(matplotlib.axes.Axes) axes: A set of axes on which to
        make the FESes and the legend. If this is given, it must support up to
        axes.flat[3].
        If None, a new figure with 2x2 axes will be created.
    :type bins: int or Sequence[int or float] or str
    :param bins: Default: None. The bins argument to be passed to
        np.histogram
    :param kwargs: keyword arguments to pass to the plot function
    :rtype: Tuple(List(np.ndarray), List(np.ndarray),
        List(matplotlib.lines.Line2D), matplotlib.figure.Figure,
        matplotlib.axes.Axes)
    :return: The delta G values, the bin centers, the lines objects, the
        figure and the axes
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    else:
        try:
            fig = axes.flat[3].figure
        except (IndexError, TypeError):
            raise InputError(
                "axes={}".format(axes),
                "Input axes must be " "able to plot at least four things",
            )
        except AttributeError:
            try:
                fig = axes[3].figure
            except IndexError:
                raise InputError(
                    "axes={}".format(axes),
                    "Input axes must " "be able to plot at least four things",
                )
    if labels is None:
        _labels = data.columns[:3]
    elif len(labels) >= 3:
        _labels = labels[:3]
    else:
        raise InputError(labels, "len(labels) must be >= 3 if not None")
    delta_gs = []
    bin_data = []
    handles = []
    # Use whatever the default colors for the system are
    # TODO find a more elegant way to do this
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, key in enumerate(_labels):
        delta_g, bin_mids = calc_fes_1d(data[key], temp=temp, bins=bins)
        delta_gs.append(delta_g)
        bin_data.append(bin_mids)
        ax = axes.flat[i]
        (line,) = ax.plot(bin_mids, delta_g, colors[i], **kwargs)
        handles.append(line)
        ax.set_ylabel(r"$\Delta G$ / (kcal / mol)")
        ax.set_xlabel(r"distance / $\mathrm{\AA}$")
    axes.flat[3].axis("off")
    axes.flat[3].legend(handles, _labels, loc="center")
    return delta_gs, bin_data, handles, fig, axes


def plot_dist_array(
    array, index_offset=1, num_data_rows=None, n_rows=None, n_cols=None
):
    """
    Puts each row of array in a different axes of a figure. Return figure.

    :param array:
    :param index_offset:
    :param num_data_rows:
    :param n_rows:
    :param n_cols:
    :return:
    """
    if not num_data_rows:
        num_data_rows = array.shape[1] - index_offset
    if n_rows is None and n_cols is None:
        n_rows = int(math.ceil(math.sqrt(float(num_data_rows))))
        n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(num_data_rows):
        ax = axes.flat[i]
        ax.plot(array[:, 0], array[:, i + index_offset])
    return fig
