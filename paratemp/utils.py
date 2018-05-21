"""
Module for random functions and such such as parsing or small calculations


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

import numpy as np
from matplotlib import pyplot as plt

from .constants import r
from .tools import running_mean


__all__ = ['calc_fes_2d', 'calc_fes_1d']


def _parse_bin_input(bins):
    if bins is None:
        return dict()
    return dict(bins=bins)


def calc_fes_2d(x, y, temp, bins=None):
    d_bins = _parse_bin_input(bins)
    counts, xedges, yedges = np.histogram2d(x, y, **d_bins)
    probs = np.array([[i / counts.max() for i in j] for j in counts]) \
        + 1e-40
    delta_g = np.array([[-r * temp * np.log(p) for p in j] for j in probs])
    xmids, ymids = running_mean(xedges), running_mean(yedges)
    return delta_g, xmids, ymids


def calc_fes_1d(data, temp, bins=None):
    d_bins = _parse_bin_input(bins)
    n, _bins = np.histogram(data, **d_bins)
    n = [float(j) for j in n]
    # TODO find better way to account for zeros here rather than
    # just adding a small amount to each.
    prob = np.array([j / max(n) for j in n]) + 1e-40
    delta_g = np.array([-r * temp * np.log(p) for p in prob])
    bin_mids = running_mean(_bins, 2)
    return delta_g, bin_mids


def _parse_ax_input(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _parse_z_bin_input(bins, zfinal, zrange):
    if bins is None:
        try:
            float(zrange)
            _zrange = [0, zrange, 11]
        except TypeError:
            dict_zrange = {1: [0, zrange[0], 11],
                           2: list(zrange) + [11]}
            _zrange = dict_zrange.get(len(zrange), zrange)
        _bins = np.append(np.linspace(*_zrange), [zfinal])
        vmax = _zrange[1]
    else:
        _bins = bins
        vmax = list(bins)[-1]
    return _bins, vmax
