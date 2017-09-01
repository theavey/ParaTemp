#! /usr/bin/env python

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2017.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017 Thomas J. Heavey IV                                   #
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

from __future__ import absolute_import

import MDAnalysis as MDa
# import mdtraj as md  # Think I'm going with MDAnalysis instead
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable

from .exceptions import InputError


# TODO move all import statements to the beginning (out of functions)


class Taddol(MDa.Universe):
    """"""

    def __init__(self, *args, **kwargs):
        """

        :param verbosity: Setting whether to print details. If in the future
        more levels of verbosity are desired, this may be changed to an int.
        Default: 1
        :type verbosity: int or bool
        :param oc_cutoffs: Cutoffs of O-O distance for determining open/closed
        TADDOL configurations. Default: ((1.0, 3.25), (3.75, 10.0))
        :type oc_cutoffs: Iterable(Iterable(float, float), Iterable(float, float))
        :param args:
        :param kwargs:
        """
        # self.univ = (line below): I'm not sure if this is needed or if this
        # just automatically inherits everything
        # Maybe use the super() command? need to learn more about this
        self._verbosity = kwargs.pop('verbosity', 1)
        self._oc_cutoffs = kwargs.pop('oc_cutoffs', ((1.0, 3.25), (3.75, 10.0)))
        super(Taddol, self).__init__(*args, **kwargs)
        self._data = pd.DataFrame(np.arange(0, self.trajectory.totaltime, self.trajectory.dt),
                                  columns=['Time'])
        self._num_frames = self.trajectory.n_frames
        self.counts_hist_ox_dists = None
        self._cv_hist_data = {}

    @property
    def data(self):
        """
        The pandas dataframe that is the backend to much of the added functions

        :return: the distances and properties for this trajectory
        :rtype: pd.DataFrame
        """
        return self._data

    @property
    def ox_dists(self):
        """
        oxygen distances property

        :return:
        """
        try:
            self._data['O-O']
        except KeyError:
            if self._verbosity:
                print('Calculating oxygen distances...\n'
                      'This may take a few minutes.')
            self._calc_ox_dists()
        # might want to (optionally) return the time column here too
        # though, as a @property, this can't take arguments, so it would need
        # to be some variable in the class
        return self._data.filter(('O-O', 'O(l)-Cy', 'O(r)-Cy'))

    def _calc_ox_dists(self):
        """
        Calculate the three oxygen-related distances.

        :return:
        """
        # TODO Find a way to make this atom-ordering independent
        # For example, this will break if TADDOL is not the first molecule
        # listed.
        # aoxr aoxl aoxr
        first_group = self.select_atoms('bynum 7 9', 'bynum 7')
        # aoxl cyclon cyclon
        second_group = self.select_atoms('bynum 9 13', 'bynum 13')
        ox_dists = np.zeros((self._num_frames, 3))
        for i, frame in enumerate(self.trajectory):
            MDa.lib.distances.calc_bonds(first_group.positions,
                                         second_group.positions,
                                         box=self.dimensions,
                                         result=ox_dists[i])
        self._data['O-O'] = ox_dists[:, 0]
        self._data['O(l)-Cy'] = ox_dists[:, 1]
        self._data['O(r)-Cy'] = ox_dists[:, 2]

    @property
    def pi_dists(self):
        """
        pi distances property

        :return:
        """
        try:
            self._data['pi-0']
        except KeyError:
            if self._verbosity:
                print('Calculating pi distances...\n'
                      'This may take a few minutes.')
            self._calc_pi_dists()
        return self._data.filter(['pi-'+str(i) for i in range(16)])

    def _calc_pi_dists(self):
        """
        Calculate the 16 TADDOL pi dists.

        :return:
        """
        raise NotImplementedError('calculating pi distances has not been '
                                  'implemented yet. Try again later.')

    def _calc_counts_hist_ox_dists(self):
        """

        :return:
        """
        if self.counts_hist_ox_dists is None:
            if self.ox_dists is None:
                self._calc_ox_dists()
            # todo write this as below
            pass
        else:
            print('ox histogram counts already calculated and saved in '
                  'self.counts_hist_ox_dists\nNot recalculating.\n'
                  'To recalculate, set self.counts_hist_ox_dists to None and '
                  'rerun this function')

    @property
    def open_ox_dists(self):
        """
        oxygen distances in a open TADDOL configuration

        :return:
        """
        try:
            self._data['open_TAD']
        except KeyError:
            if self._verbosity:
                print('Finding open/closed configurations...')
            self._calc_open_closed()
        return self._data[self._data['open_TAD']].filter(
            ('O-O', 'O(l)-Cy', 'O(r)-Cy'))

    @property
    def closed_ox_dists(self):
        """
        oxygen distances in a closed TADDOL configuration

        :return:
        """
        try:
            self._data['closed_TAD']
        except KeyError:
            if self._verbosity:
                print('Finding open/closed configurations...')
            self._calc_open_closed()
        return self._data[self._data['closed_TAD']].filter(
            ('O-O', 'O(l)-Cy', 'O(r)-Cy'))

    @property
    def oc_cutoffs(self):
        """
        Cutoffs for O-O distance for determining open/closed TADDOL configs

        :return:
        """
        return self._oc_cutoffs

    @oc_cutoffs.setter
    def oc_cutoffs(self, value):
        try:
            [[value[i][j] for i in range(2)] for j in range(2)]
        except (TypeError, IndexError):
            raise TypeError('cutoffs must be an iterable of shape (2, 2)')

    def _calc_open_closed(self):
        """
        Select the coordinates for open vs. closed TADDOL

        :return:
        """
        # I'm not sure this function is necessary. These queries might be
        # really fast already.
        cutoffs = self.oc_cutoffs
        cut_closed = cutoffs[0]
        cut_open = cutoffs[1]
        self._data['closed_TAD'] = self.ox_dists['O-O'].apply(
            lambda x: cut_closed[0] <= x <= cut_closed[1])
        self._data['open_TAD'] = self.ox_dists['O-O'].apply(
            lambda x: cut_open[0] <= x <= cut_open[1])

    @property
    def cv1_dists(self):
        """
        Distances for CV1 during the trajectory

        :return: CV1 distances
        :rtype: pd.Series
        """
        try:
            self._data['CV1']
        except KeyError:
            print('Calculating CV values...\n'
                  'This may take a few minutes.')
            self._calc_cvs()
        return self._data['CV1']

    @property
    def cv2_dists(self):
        """
        Distances for CV2 during the trajectory

        :return: CV2 distances
        :rtype: pd.Series
        """
        try:
            self._data['CV2']
        except KeyError:
            print('Calculating CV values...\n'
                  'This may take a few minutes.')
            self._calc_cvs()
        return self._data['CV2']

    def _calc_cvs(self):
        """
        Calculate the CV values

        :return: None
        :rtype: None
        """
        # TODO generalize the atom selections, likely with a class variable
        first_group = self.select_atoms('bynum 160', 'bynum 133')
        second_group = self.select_atoms('bynum 9', 'bynum 8')
        cv_dists = np.zeros((self._num_frames, 2))
        for i, frame in enumerate(self.trajectory):
            MDa.lib.distances.calc_bonds(first_group.positions,
                                         second_group.positions,
                                         box=self.dimensions,
                                         result=cv_dists[i])
        self._data['CV1'] = cv_dists[:, 0]
        self._data['CV2'] = cv_dists[:, 1]

    def hist_2d_cvs(self, x=None, y=None, return_fig=True, **kwargs):
        """"""
        # TODO make the constants here arguments
        # TODO make this optionally save figure
        if x is None:
            x = self.cv1_dists
        if y is None:
            y = self.cv2_dists
        fig, ax = plt.subplots()
        counts, xedges, yedges = ax.hist2d(x, y,
                                           32, **kwargs)[:3]
        self._cv_hist_data['counts'] = counts
        self._cv_hist_data['xedges'] = xedges
        self._cv_hist_data['yedges'] = yedges
        ax.axis((1.5, 10, 1.5, 10))
        ax.set_xlabel('CV 2')
        ax.set_ylabel('CV 1')
        ax.set_aspect('equal', 'box-forced')
        fig.tight_layout()
        if return_fig:
            return fig

    def fes_2d_cvs(self, x=None, y=None, temp=205., **kwargs):
        """"""
        # TODO make the constants here arguments
        # TODO make this optionally save figure
        # TODO check on cv1 vs. cv2 for x / y
        if x is None:
            x = self.cv1_dists
        if y is None:
            y = self.cv2_dists
        try:
            counts = self._cv_hist_data['counts']
            xedges = self._cv_hist_data['xedges']
            yedges = self._cv_hist_data['yedges']
        except KeyError:
            counts, xedges, yedges = np.histogram2d(x, y, 32)
            self._cv_hist_data['counts'] = counts
            self._cv_hist_data['xedges'] = xedges
            self._cv_hist_data['yedges'] = yedges
        probs = np.array([[i / counts.max() for i in j] for j in counts]) \
            + 1e-40
        r = 0.0019872  # kcal_th/(K mol)
        delta_g = np.array([[-r * temp * np.log(p) for p in j] for j in probs])
        fig, ax = plt.subplots()
        contours = ax.contourf(xedges[:-1], yedges[:-1], delta_g.transpose(),
                               np.append(np.linspace(0, 20, 11), [40]),
                               vmax=20, **kwargs)
        ax.axis((1.5, 10, 1.5, 10))
        ax.set_xlabel('CV 2')
        ax.set_ylabel('CV 1')
        ax.set_aspect('equal', 'box-forced')
        fig.colorbar(contours, label='kcal / mol')
        fig.tight_layout()
        return fig

    def plot_ox_dists(self, save=False, save_format='png',
                      save_base_name='ox-dists',
                      display=True, **kwargs):
        """
        Plot the three oxygen-related distances.

        :param save:
        :param save_format:
        :param save_base_name:
        :param display:
        :param kwargs:
        :return:
        """
        if self.ox_dists is None:
            self._calc_ox_dists()
        pass  # TODO write this based on below

    def hist_ox_dists(self, n_bins=10, save=False, save_format='pdf',
                      save_base_name='ox-dists-hist',
                      display=True, separate=True, **kwargs):
        """
        Make histogram of alcoholic O distances in TADDOL trajectory

        :param n_bins:
        :param save:
        :param save_format:
        :param save_base_name:
        :param display:
        :param separate:
        :param kwargs:
        :return:
        """
        if self.ox_dists is None:
            self._calc_ox_dists()
        # Save the histogram figures and/or data for making the FESs
        # or better yet, use separate calc hist data function and just plot it
        pass  # TODO write this based on below

    def fes_ox_dists(self, temp=791., save=False,
                     save_format='pdf',
                     save_base_name='ox-dists-fes',
                     display=True, **kwargs):
        """

        :param temp:
        :param save:
        :param save_format:
        :param save_base_name:
        :param display:
        :param kwargs: keyword arguments to pass to the plotter
        :return:
        """
        if self.counts_hist_ox_dists is None:
            self.calc_counts_hist_ox_dists()
        pass  # TODO write this based on below


def get_taddol_selections(universe, univ_in_dict=True):
    """Returns a dict of AtomSelections from the given universe"""
    d_out = dict()
    if univ_in_dict:
        d_out['universe'] = universe
    d_out["phenrtt"] = universe.select_atoms('bynum 92 94')
    d_out["phenrtb"] = universe.select_atoms('bynum 82 87')
    d_out["phenrbt"] = universe.select_atoms('bynum 69 71')
    d_out["phenrbb"] = universe.select_atoms('bynum 59 64')
    d_out["phenltt"] = universe.select_atoms('bynum 115 117')
    d_out["phenltb"] = universe.select_atoms('bynum 105 110')
    d_out["phenlbt"] = universe.select_atoms('bynum 36 41')
    d_out["phenlbb"] = universe.select_atoms('bynum 46 48')
    d_out["quatl"] = universe.select_atoms('bynum 6')
    d_out["quatr"] = universe.select_atoms('bynum 1')
    d_out["chirl"] = universe.select_atoms('bynum 4')
    d_out["chirr"] = universe.select_atoms('bynum 2')
    d_out["cyclon"] = universe.select_atoms('bynum 13')
    d_out["cyclof"] = universe.select_atoms('bynum 22')
    d_out["aoxl"] = universe.select_atoms('bynum 9')
    d_out["aoxr"] = universe.select_atoms('bynum 7')
    return d_out


def get_dist(a, b, box=None):
    """Calculate the distance between AtomGroups a and b.

    If a box is provided, this will use the builtin MDAnalysis function to
    account for periodic boundary conditions."""
    if box is not None:
        coordinates = (np.array([atom.centroid()]) for atom in (a, b))
        return MDa.lib.distances.calc_bonds(*coordinates,
                                            box=box)
    else:
        from numpy.linalg import norm
        return norm(a.centroid() - b.centroid())


def get_dist_dict(dictionary, a, b, box=None):
    """Calculate distance using dict of AtomSelections"""
    return get_dist(dictionary[a], dictionary[b], box=box)


def get_angle(a, b, c, units='rad'):
    """Calculate the angle between ba and bc for AtomGroups a, b, c"""
    # TODO look at using the MDAnalysis builtin function
    from numpy import arccos, rad2deg, dot
    from numpy.linalg import norm
    b_center = b.centroid()
    ba = a.centroid() - b_center
    bc = c.centroid() - b_center
    angle = arccos(dot(ba, bc)/(norm(ba)*norm(bc)))
    if units == 'rad':
        return angle
    elif units == 'deg':
        return rad2deg(angle)
    else:
        raise InputError(units,
                         'Unrecognized units: '
                         'the two recognized units are rad and deg.')


def get_angle_dict(dictionary, a, b, c, units='rad'):
    """Calculate angle using dict of AtomSelections"""
    return get_angle(dictionary[a], dictionary[b], dictionary[c],
                     units=units)


def get_dihedral(a, b, c, d, units='rad'):
    """Calculate the angle between abc and bcd for AtomGroups a,b,c,d

    Based on formula given in
    https://en.wikipedia.org/wiki/Dihedral_angle"""
    # TODO look at using the MDAnalysis builtin function
    from numpy import cross, arctan2, dot, rad2deg
    from numpy.linalg import norm
    ba = a.centroid() - b.centroid()
    bc = b.centroid() - c.centroid()
    dc = d.centroid() - c.centroid()
    angle = arctan2(dot(cross(cross(ba, bc), cross(bc, dc)), bc) /
                    norm(bc), dot(cross(ba, bc), cross(bc, dc)))
    if units == 'rad':
        return angle
    elif units == 'deg':
        return rad2deg(angle)
    else:
        raise InputError(units,
                         'Unrecognized units: '
                         'the two recognized units are rad and deg.')


def get_dihedral_dict(dictionary, a, b, c, d, units='rad'):
    """Calculate dihedral using dict of AtomSelections"""
    return get_dihedral(dictionary[a], dictionary[b],
                        dictionary[c], dictionary[d],
                        units=units)


def get_taddol_ox_dists(universe, sel_dict=False):
    """Get array of oxygen distances in TADDOL trajectory"""
    from numpy import array
    if not sel_dict:
        sel_dict = get_taddol_selections(universe)
    output = []
    for frame in universe.trajectory:
        box = universe.dimensions
        output.append((universe.trajectory.time,
                       get_dist_dict(sel_dict, 'aoxl', 'aoxr', box=box),
                       get_dist_dict(sel_dict, 'aoxl', 'cyclon', box=box),
                       get_dist_dict(sel_dict, 'aoxr', 'cyclon', box=box)))
    return array(output)


def make_plot_taddol_ox_dists(data, save=False, save_format='pdf',
                              save_base_name='ox_dists',
                              display=True):
    """Make plot of alcoholic O distances in TADDOL trajectory"""
    from matplotlib.pyplot import subplots
    fig, axes = subplots()
    axes.plot(data[:, 0], data[:, 1], label='O-O')
    axes.plot(data[:, 0], data[:, 2], label='O(l)-Cy')
    axes.plot(data[:, 0], data[:, 3], label='O(r)-Cy')
    axes.legend()
    axes.set_xlabel('time / ps')
    axes.set_ylabel('distance / $\mathrm{\AA}$')
    if save:
        fig.savefig(save_base_name+save_format)
    if display:
        return fig
    else:
        return None


def make_hist_taddol_ox_dists(data, n_bins=10, save=False, save_format='pdf',
                              save_base_name='ox_dists_hist',
                              display=True, separate=False):
    """Make histogram of alcoholic O distances in TADDOL trajectory"""
    from matplotlib.pyplot import subplots
    legend_entries = ['O-O', 'O(l)-Cy', 'O(r)-Cy']
    if separate:
        fig, axes = subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        handles = []
        # Use whatever the default colors for the system are
        # TODO find a more elegant way to do this
        colors = mpl.rcParams['axes.prop_cycle'].by_key().values()[0]
        for i in range(3):
            ax = axes.flat[i]
            n, bins, patches = ax.hist(data[:, 1 + i], n_bins,
                                       label=legend_entries[i],
                                       facecolor=colors[i])
            handles.append(patches[0])
            ax.set_xlabel(r'distance / $\mathrm{\AA}$')
            ax.set_ylabel('frequency')
        axes.flat[3].axis('off')
        axes.flat[3].legend(handles, legend_entries, loc='center')
    else:
        fig, ax = subplots()
        ax.hist(data[:, 1:], n_bins, histtype='stepfilled')
        ax.set_xlabel(r'distance / $\mathrm{\AA}$')
        ax.set_ylabel('frequency')
        ax.legend(legend_entries)
    if save:
        fig.savefig(save_base_name+save_format)
    if display:
        return fig
    else:
        return None


def get_taddol_pi_dists(universe, sel_dict=False):
    """Get array of phenanthryl distances in TADDOL trajectory"""
    from numpy import array
    if not sel_dict:
        sel_dict = get_taddol_selections(universe)
    output = []
    # For brevity, I redefine these:
    gdd = get_dist_dict
    sd = sel_dict
    for frame in universe.trajectory:
        output.append((universe.trajectory.time,
                       gdd(sd, 'phenrtt', 'phenltt'),
                       gdd(sd, 'phenrtt', 'phenltb'),
                       gdd(sd, 'phenrtt', 'phenlbt'),
                       gdd(sd, 'phenrtt', 'phenlbb'),
                       gdd(sd, 'phenrtb', 'phenltt'),
                       gdd(sd, 'phenrtb', 'phenltb'),
                       gdd(sd, 'phenrtb', 'phenlbt'),
                       gdd(sd, 'phenrtb', 'phenlbb'),
                       gdd(sd, 'phenrbt', 'phenltt'),
                       gdd(sd, 'phenrbt', 'phenltb'),
                       gdd(sd, 'phenrbt', 'phenlbt'),
                       gdd(sd, 'phenrbt', 'phenlbb'),
                       gdd(sd, 'phenrbb', 'phenltt'),
                       gdd(sd, 'phenrbb', 'phenltb'),
                       gdd(sd, 'phenrbb', 'phenlbt'),
                       gdd(sd, 'phenrbb', 'phenlbb')))
    return array(output)


def plot_dist_array(array, index_offset=1, num_data_rows=False,
                    n_rows=False, n_cols=False):
    """
    Puts each row of array in a different axes of a figure. Returns the figure.

    :param array:
    :param index_offset:
    :param num_data_rows:
    :param n_rows:
    :param n_cols:
    :return:
    """
    if not num_data_rows:
        num_data_rows = array.shape[1] - index_offset
    from math import sqrt, ceil
    if n_rows == n_cols == False:
        n_rows = int(ceil(sqrt(float(num_data_rows))))
        n_cols = n_rows
    from matplotlib.pyplot import subplots
    fig, axes = subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(num_data_rows):
        ax = axes.flat[i]
        ax.plot(array[:, 0], array[:, i+index_offset])
    return fig


def make_taddol_pi_dist_array(dists, save=False, save_format='pdf',
                              save_base_name='pi_dists',
                              display=True):
    """Plot array of pi distances in TADDOL trajectory"""
    fig = plot_dist_array(dists)
    [ax.get_xaxis().set_ticks([]) for ax in fig.axes]
    fig.text(0.05, 0.585, 'distance / $\mathrm{\AA}$', ha='center',
             rotation='vertical')
    fig.text(0.513, 0.08, 'time', ha='center')
    if save:
        fig.savefig(save_base_name+save_format)
    if display:
        return fig
    else:
        return None


def make_fes_taddol_ox_dist(dists, temp=791., save=False,
                            save_format='pdf',
                            save_base_name='ox_dists_fes',
                            display=True, **kwargs):
    """Plot the relative free energy surface of O distances in TADDOL"""
    from matplotlib.pyplot import subplots
    from numpy import log, array, histogram
    r = 0.0019872  # kcal_th/(K mol)
    delta_gs = []
    fig, axes = subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    handles = []
    # Use whatever the default colors for the system are
    # TODO find a more elegant way to do this
    colors = mpl.rcParams['axes.prop_cycle'].by_key().values()[0]
    for i in range(3):
        n, bins = histogram(dists[:, 1+i])
        n = [float(j) for j in n]
        # TODO find better way to account for zeros here rather than
        # just adding a small amount to each.
        prob = array([j / max(n) for j in n]) + 1e-10
        delta_g = array([-r * temp * log(p) for p in prob])
        delta_gs.append(delta_g)
        ax = axes.flat[i]
        line, = ax.plot(bins[:-1], delta_g, colors[i], **kwargs)
        handles.append(line)
        ax.set_ylabel(r'$\Delta G$ / (kcal / mol)')
        ax.set_xlabel(r'distance / $\mathrm{\AA}$')
    axes.flat[3].axis('off')
    axes.flat[3].legend(handles, ['O-O', 'O(l)-Cy', 'O(r)-Cy'],
                        loc='center')
    if save:
        fig.savefig(save_base_name+save_format)
    if display:
        return fig
    else:
        return None


def select_open_closed_dists(dists, cutoffs=((1.0, 3.25),
                                             (3.75, 10.0))):
    """
    Select the coordinates for open vs. closed TADDOL

    :param dists:
    :param cutoffs:
    :return:
    """
    cut_closed = cutoffs[0]
    cut_open = cutoffs[1]
    set_open = []
    set_closed = []
    for ts in dists:
        if cut_open[0] <= ts[1] <= cut_open[1]:
            set_open.append(ts)
        if cut_closed[0] <= ts[1] <= cut_closed[1]:
            set_closed.append(ts)
    from pandas import DataFrame
    columns = ['Time', 'O-O', 'Ol-Cy', 'Or-Cy']
    return DataFrame(set_open, columns=columns), \
        DataFrame(set_closed, columns=columns)
