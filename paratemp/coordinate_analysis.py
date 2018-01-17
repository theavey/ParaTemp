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

from __future__ import absolute_import, division, print_function
from six.moves import range

import os.path
from warnings import warn

import MDAnalysis as MDa
import matplotlib as mpl
import matplotlib.pyplot as plt
# import mdtraj as md  # Think I'm going with MDAnalysis instead
import numpy as np
import pandas as pd
from typing import Iterable, Tuple

from . import exceptions
from .exceptions import InputError


# TODO move all import statements to the beginning (out of functions)


class Universe(MDa.Universe):

    def __init__(self, *args, **kwargs):
        """

        :type verbosity: int or bool
        :param verbosity: Default: 1. Setting whether to print details. If in
            the future more levels of verbosity are desired, this may be
            changed to an int.

        :param float temp: Default: None. Temperature of this simulation to be
            used for calculating free energy surfaces with weighted histograms.
        :param args:
        :param kwargs:
        """
        # self.univ = (line below): I'm not sure if this is needed or if this
        # just automatically inherits everything
        # Maybe use the super() command? need to learn more about this
        self._verbosity = kwargs.pop('verbosity', 1)
        self._temperature = kwargs.pop('temp', None)
        super(Universe, self).__init__(*args, **kwargs)
        self._num_frames = self.trajectory.n_frames
        self._last_time = self.trajectory.totaltime
        self._data = self._init_dataframe()
        # dict of distance definitions
        self._dict_dist_defs = {}
        self._dict_dihed_defs = {}

    def _init_dataframe(self):
        """
        Initialize a pandas.DataFrame with Times column.

        This uses self._last_time as the final time and self._num_frames as
        the number of rows to put into the DataFrame to be returned.
        This uses np.linspace to make the evenly spaced times that should
        match with the times in the trajectory file.

        :return: a DataFrame with one column of Times
        :rtype: pd.DataFrame
        """
        return pd.DataFrame(np.linspace(0, self._last_time,
                                        num=self._num_frames),
                            columns=['Time'])

    def save_data(self, filename=None, overwrite=False):
        """
        Save calculated data to disk

        :param str filename: Filename to save the data as. Defaults to the
            name of the trajectory with a '.h5' extension.

        :param bool overwrite: Whether to overwrite existing data on disk.
            If it's True, it will completely overwrite the existing data store.
            If it's False, but a store for this time already exists, only new
            columns in self.data will be added to the store, and no data will be
            overwritten.

        :return: None
        """
        if filename is None:
            filename = os.path.splitext(self.trajectory.filename)[0] + '.h5'
        with pd.HDFStore(filename) as store:
            time = 'time_' + str(int(self._last_time/1000)) + 'ns'
            # TODO use self.final_time_str
            if overwrite or ('/'+time not in store.keys()):
                store[time] = self._data
            else:
                store_cols = store.get_node(time).axis0.read()
                set_diff_cols = set(self._data.columns).difference(store_cols)
                if not set_diff_cols:
                    if self._verbosity:
                        print('No data added to {}[{}]'.format(filename, time))
                    return
                store_df = store[time]  # seems this has to be done to add cols
                # see https://stackoverflow.com/questions/15939603/
                # append-new-columns-to-hdfstore-with-pandas
                for col in set_diff_cols:
                    store_df[col] = self._data[col]
                store[time] = store_df
        if self._verbosity:
            print('Saved data to {}[{}]'.format(filename, time))

    def read_data(self, filename=None, ignore_no_data=False):
        """
        Read calculated data from disk

        This will read the data from disk and add it to self.data. Any
        existing data will not be overwritten.

        :param str filename: Filename from which to read the data.
            Defaults to the name of the trajectory with a '.h5' extension.
        :param bool ignore_no_data: Default: False. If True, not having data
            in the file will not raise an error.
        :return: None
        :raises: IOError
        """
        if filename is None:
            filename = os.path.splitext(self.trajectory.filename)[0] + '.h5'
        with pd.HDFStore(filename) as store:
            time = 'time_' + str(int(self._last_time/1000)) + 'ns'
            # TODO use self.final_time_str
            try:
                read_df = store[time]
                keys_to_read = set(read_df.columns).difference(
                    self._data.columns)
            except KeyError:
                keys_to_read = []
                read_df = pd.DataFrame()  # not necessary; stops IDE complaint
                if not ignore_no_data:
                    raise IOError('This data does not exist!\n{}[{}]'.format(
                        filename, time))
                else:
                    if self._verbosity:
                        print('No data to read in '
                              '{}[{}]'.format(filename, time))
        for key in keys_to_read:
            self._data[key] = read_df[key]

    def calculate_distances(self, recalculate=False, ignore_file_change=False,
                            *args, **kwargs):
        """
        Calculate distances by iterating through the trajectory

        :param recalculate: Default: False. If True, all requested
            distances will be calculated.
            If False, the intersection of the set of requested distance names
            and the existing column titles in self.data will be removed from the
            information to be calculated.
        :param ignore_file_change: Default: False. If True, if the file has
            changed since object instantiation, no error will be raised and
            only information through the last frame when the object was
            instantiated will be calculated. If self._verbosity, the fact that
            the file has changed will be printed.
            If False, if the length of the trajectory has changed,
            FileChangedError will be raised.
        :param args:
        :param kwargs:
        :return: None
        :raises: exceptions.FileChangedError, SyntaxError, NotImplementedError
        """
        # TODO document this function
        # TODO find a way to take keyword type args with non-valid python
        # identifiers (e.g., "O-O").
        # Make empty atom selections to be appended to:
        first_group = self.select_atoms('protein and not protein')
        second_group = self.select_atoms('protein and not protein')
        column_names = []
        if len(args) == 0 and len(kwargs) == 0:
            args = ['all']
        if len(args) != 0:
            try:
                args = [arg.lower() for arg in args]
            except AttributeError:
                raise SyntaxError('All positional arguments must be strings')
            bad_args = []
            for arg in args:
                try:
                    temp_dict = self._dict_dist_defs[arg]
                    temp_dict.update(kwargs)
                    kwargs = temp_dict.copy()
                except KeyError:
                    bad_args.append(arg)
            if len(bad_args) != 0:
                warn('The following positional arguments were given but not '
                     'recognized: ' + str(bad_args) + '\nThey will be '
                     'ignored.')
        if not recalculate:
            set_existing_data = set(self.data.columns)
            set_new_data = set(kwargs.keys())
            set_overlap = set_existing_data.intersection(set_new_data)
            for col in set_overlap:
                del kwargs[col]
        if len(kwargs) != 0:
            for key in kwargs:
                try:
                    atoms = kwargs[key].split()
                except AttributeError:
                    # assume it is iterable as is
                    atoms = kwargs[key]
                if len(atoms) != 2:
                    raise SyntaxError('This input should split to two atom '
                                      'indices: {}'.format(kwargs[key]))
                try:
                    [int(atom) for atom in atoms]
                except ValueError:
                    raise NotImplementedError('Only selection by atom index is'
                                              ' currently supported.\nAt your '
                                              'own risk you can try assigning '
                                              'to self._data[{}].'.format(key))
                first_group += self.select_atoms('bynum '+str(atoms[0]))
                second_group += self.select_atoms('bynum '+str(atoms[1]))
                column_names += [key]
        else:
            if self._verbosity:
                print('Nothing (new) to calculate here.')
        n1 = first_group.n_atoms
        n2 = second_group.n_atoms
        nc = len(column_names)
        if not nc == n1 == n2:
            raise SyntaxError('Different numbers of atom selections or number'
                              ' of column labels '
                              '({}, {}, and {}, respectively).'.format(n1,
                                                                       n2,
                                                                       nc) +
                              '\nThis should not happen.')
        if self._num_frames != self.trajectory.n_frames:
            if self._verbosity:
                print('Current trajectory has {} frames, '.format(
                    self.trajectory.n_frames) +
                      'but this object was instantiated with ' 
                      '{} frames.'.format(self._num_frames))
            if not ignore_file_change:
                raise exceptions.FileChangedError()
        dists = np.zeros((self._num_frames, n1))
        for i in range(self._num_frames):
            self.trajectory[i]
            MDa.lib.distances.calc_bonds(first_group.positions,
                                         second_group.positions,
                                         box=self.dimensions,
                                         result=dists[i])
        for i, column in enumerate(column_names):
            self._data[column] = dists[:, i]

    def calculate_dihedrals(self, *args, **kwargs):
        """"""
        # todo there should be a way to generalize the "calculate" functions
        # use this function
        # http://www.mdanalysis.org/docs/documentation_pages/lib/distances.html
        # Make empty atom selections to be appended to:
        groups = [self.select_atoms('protein and not protein')] * 4
        column_names = []
        if len(args) == 0 and len(kwargs) == 0:
            raise InputError('calculate_dihedrals', 'No arguments given; '
                                                    'nothing to calculate')
        if len(args) != 0:
            try:
                args = [arg.lower() for arg in args]
            except AttributeError:
                raise SyntaxError('All positional arguments must be strings')
            bad_args = []
            for arg in args:
                try:
                    temp_dict = self._dict_dihed_defs[arg]
                    temp_dict.update(kwargs)
                    kwargs = temp_dict.copy()
                except KeyError:
                    bad_args.append(arg)
            if len(bad_args) != 0:
                warn('The following positional arguments were given but not '
                     'recognized: ' + str(bad_args) + '\nThey will be '
                                                      'ignored.')
        if len(kwargs) != 0:
            for key in kwargs:
                try:
                    atoms = kwargs[key].split()
                except AttributeError:
                    # assume it is iterable as is
                    atoms = kwargs[key]
                if len(atoms) != 4:
                    raise SyntaxError('This input should split to four atom '
                                      'indices: {}'.format(kwargs[key]))
                try:
                    [int(atom) for atom in atoms]
                except ValueError:
                    raise NotImplementedError('Only selection by atom index is'
                                              ' currently supported.\nAt your '
                                              'own risk you can try assigning '
                                              'to self._data[{}].'.format(key))
                for i in range(4):
                    groups[i] += self.select_atoms('bynum ' + str(atoms[i]))
                column_names += [key]
        n_atoms = [x.n_atoms for x in groups]
        nc = len(column_names)
        if not (n_atoms.count(n_atoms[0]) == len(n_atoms)
                and n_atoms[0] == nc):
            raise SyntaxError('Different number of column labels or atom '
                              'selections ({}, {}, {}, {} and {}, '
                              'respectively).'.format(nc, *n_atoms) +
                              '\nThis should not happen.')
        if self._num_frames != self.trajectory.n_frames:
            raise exceptions.FileChangedError()
        diheds = np.zeros((self._num_frames, nc))
        for i, frame in enumerate(self.trajectory):
            MDa.lib.distances.calc_dihedrals(groups[0].positions,
                                             groups[1].positions,
                                             groups[2].positions,
                                             groups[3].positions,
                                             box=self.dimensions,
                                             result=diheds[i])
        for i, column in enumerate(column_names):
            self._data[column] = diheds[:, i]

    def update_num_frames(self, silent=False):
        """
        Update number of frames and last time from trajectory file

        :param bool silent: Default: False. If True, nothing will be printed.
        :return: None
        """
        num_frames = self.trajectory.n_frames
        if num_frames != self._num_frames:
            if self._verbosity and not silent:
                print('Updating num of frames from {} to {}'.format(
                    self._num_frames, num_frames) +
                      '\nand the final time.')
            self._num_frames = num_frames
            self._last_time = self.trajectory.totaltime

    def update_data_len(self, update_time=True, silent=False):
        """
        Update the times and length of self.data based on trajectory file

        :param update_time: Default: True. If True, self.update_num_frames
            will be used to find the new length and final time of the
            trajectory file. If False, these will just be read from the instance
            variables and not updated based on the trajectory file.
        :param silent: Default: False. If True, nothing will be printed.
        :return: None
        """
        if update_time:
            self.update_num_frames(silent=True)
        if self._data['Time'].iat[-1] != self._last_time:
            old_len = len(self.data)
            new_times_df = self._init_dataframe()
            self._data = self._data.join(new_times_df.set_index('Time'),
                                         on='Time')
            if self._verbosity and not silent:
                print('Updating data from '
                      '{} frames to {} frames'.format(old_len, len(self._data)))
        else:
            if self._verbosity and not silent:
                print('No need to update self.data')

    @property
    def data(self):
        """
        The pd.DataFrame that is the backend to much of the added functions

        :return: the distances and properties for this trajectory
        :rtype: pd.DataFrame
        """
        # TODO might be able to be clever here and catch key errors and
        # and then calculate as needed
        # An easier solution is just to add a calc funtion that parsers things
        # I need calculated without needing to return them.
        # I doubt I could do the clever thing without subclassing DataFrame,
        # and I'm not sure I want to mess with their item access stuff.
        return self._data

    @property
    def final_time_str(self):
        """"""
        ps, ns, us, ms = (1, 'ps'), (1e3, 'ns'), (1e6, 'us'), (1e9, 'ms')
        time_dict = {1: ps, 2: ps, 3: ps, 4: ns, 5: ns, 6: ns, 7: us, 8: us,
                     9: us}
        f_time = str(int(self._last_time))
        try:
            power, exten = time_dict[len(f_time)]
        except KeyError:
            power, exten = ms
        return str(int(self._last_time/power)) + exten

    @staticmethod
    def _running_mean(x, n=2):
        """
        Calculate running mean over an iterable

        Taken from https://stackoverflow.com/a/22621523/3961920

        :param Iterable x: List over which to calculate the mean.
        :param int n: Default: 2. Width for the means.
        :return: Array of the running mean values.
        :rtype: np.ndarray
        """
        return np.convolve(x, np.ones((n,)) / n, mode='valid')

    def fes_2d(self, x, y, temp=None, ax=None, bins=None,
               zrange=(0, 20, 11), zfinal=40, n_bins=32, transpose=False,
               xlabel='x', ylabel='y', scale=True,
               **kwargs):
        """
        plot FES in 2D along defined values

        :param x: Value along x axis to plot. If a string is given, the data
            will be taken from self.data[x].
        :type x: Iterable or str
        :param y: Value along y axis to plot. If a string is given, the data
            will be taken from self.data[y].
        :type y: Iterable or str
        :param float temp: Default: None. Temperature for Boltzmann weighting
            calculation.
            If None is provided, the temperature will be taken from
            self._temperature
        :param matplotlib.axes.Axes ax: Default: None. Axes on which to make
            the FES. If None, a new axes and figure will be created.
        :param Iterable bins: Default: None. The bins to be used for the z
            ranges. If this is not None, zrange and zfinal are ignored.
        :param zrange: Default: (0, 20, 11). Input to np.linspace for
            determining contour levels. If a float-like is given, it will be set
            as the max with 11+1 bins. If a len=2 list-like is given, it will be
            used as the min and max with 11+1 bins. Otherwise, the input will
            be used as-is for input to np.linspace.
        :type zrange: Iterable or Float
        :param zfinal: Default: 40. Energy at which to stop coloring the FES.
            Anything above this energy will appear as white.
        :param int n_bins: Default: 32. Number of bins in x and y for
            histogramming.
        :param bool transpose: Default: False. Whether to transpose the data
            and axes such that the input x will be along the y axis and the
            inverse. Note, this also makes the xlabel on the y-axis and the
            inverse.
        :param str xlabel: Default: 'x'. Label for x-axis (or y-axis if
            transpose=True).
        :param str ylabel: Default: 'y'. Label for y-axis (or x-axis if
            transpose=True).
        :param bool scale: Default: True. Include a colorbar scale in the
            figure of the axes.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: The delta G values, the bin centers, the contours, the figure,
            and the axes
        :rtype: Tuple(np.ndarray, Tuple(np.ndarray, np.ndarray),
            matplotlib.contour.QuadContourSet, matplotlib.figure.Figure,
            matplotlib.axes.Axes)
        """
        if temp is None:
            _temp = self._temperature
        else:
            _temp = temp
        if _temp is None:
            raise ValueError('The temperature must be defined to calculate an'
                             ' FES')
        # TODO make the constants here arguments
        if type(x) is str:
            try:
                _x = self.data[x]
            except KeyError:
                raise InputError(x, 'input as a str must be an existing'
                                    'key for the data in this object')
        else:
            _x = x
        if type(y) is str:
            try:
                _y = self.data[y]
            except KeyError:
                raise InputError(y, 'input as a str must be an existing'
                                    'key for the data in this object')
        else:
            _y = y
        counts, xedges, yedges = np.histogram2d(_x, _y, n_bins)
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
        probs = np.array([[i / counts.max() for i in j] for j in counts]) \
            + 1e-40
        r = 0.0019872  # kcal_th/(K mol)
        delta_g = np.array([[-r * _temp * np.log(p) for p in j] for j in probs])
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        xmids, ymids = self._running_mean(xedges), self._running_mean(yedges)
        if not transpose:
            # This is because np.histogram2d returns the counts oddly
            delta_g = delta_g.transpose()
            _xlabel, _ylabel = xlabel, ylabel
        else:
            xmids, ymids = ymids, xmids
            _xlabel, _ylabel = ylabel, xlabel
        contours = ax.contourf(xmids, ymids, delta_g,
                               _bins, vmax=vmax, **kwargs)
        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabel)
        ax.set_aspect('equal', 'box-forced')
        if scale:
            fig.colorbar(contours, label='kcal / mol')
            fig.tight_layout()
        return delta_g, (xmids, ymids), contours, fig, ax

    def fes_1d(self, data, temp=None, xlabel=r'distance / $\mathrm{\AA}$',
               ax=None, **kwargs):
        """
        Make FES of some time series data

        :type data: Iterable or str
        :param data: Data to form the FES from. If a string is given, the data
            will be taken from self.data[data].

        :param float temp: Default: None. Temperature for Boltzmann weighting
            calculation.
            If None is provided, the temperature will be taken from
            self._temperature

        :param str xlabel: Default: 'distance / $\mathrm{\AA}$'. The label for
            the x axis.

        :type ax: matplotlib.axes.Axes
        :param ax: Default: None. The axes objects on which to make the plots.
            If None is supplied, new axes objects will be created.

        :param kwargs: keyword arguments to pass to the plotter

        :rtype: Tuple(np.ndarray, np.ndarray, matplotlib.lines.Line2D,
            matplotlib.figure.Figure, matplotlib.axes.Axes)

        :return: The delta G values, the bin centers, the lines object, the
            figure and the axes
        """
        if temp is None:
            _temp = self._temperature
        else:
            _temp = temp
        if _temp is None:
            raise ValueError('The temperature must be defined to calculate an'
                             ' FES')
        if type(data) is str:
            try:
                _data = self.data[data]
            except KeyError:
                raise InputError(data, 'input as a str must be an existing'
                                       'key for the data in this object')
        else:
            _data = data
        if ax is None:
            _fig, _ax = plt.subplots()
        else:
            _ax = ax
            _fig = ax.figure
        r = 0.0019872  # kcal_th/(K mol)
        n, bins = np.histogram(_data)
        n = [float(j) for j in n]
        # TODO find better way to account for zeros here rather than
        # just adding a small amount to each.
        prob = np.array([j / max(n) for j in n]) + 1e-40
        delta_g = np.array([-r * _temp * np.log(p) for p in prob])
        bin_mids = self._running_mean(bins, 2)
        lines = _ax.plot(bin_mids, delta_g, **kwargs)
        _ax.set_ylabel(r'$\Delta G$ / (kcal / mol)')
        _ax.set_xlabel(xlabel)
        return delta_g, bin_mids, lines, _fig, _ax


class Taddol(Universe):
    """"""

    def __init__(self, *args, **kwargs):
        """

        :param verbosity: Setting whether to print details. If in the
            future more levels of verbosity are desired, this may be
            changed to an int.
            Default: 1
        :type verbosity: int or bool
        :param oc_cutoffs: Cutoffs of O-O distance for determining
            open/closed TADDOL configurations. Default: ((1.0, 3.25),
                                                     (3.75, 10.0))
        :type oc_cutoffs: Iterable(Iterable(float, float),
                                   Iterable(float, float))
        :param args:
        :param kwargs:
        """
        self._oc_cutoffs = kwargs.pop('oc_cutoffs',
                                      ((1.0, 3.25), (3.75, 10.0)))
        super(Taddol, self).__init__(*args, **kwargs)
        self._cv_hist_data = {}
        # TODO add temp argument and pass to FES functions
        # dict of distance definitions
        # TODO Find a way to make this atom-ordering independent
        # For example, this will break if TADDOL is not the first molecule
        # listed.
        self._dict_dist_defs = {'ox': {'O-O': (7, 9),
                                       'O(l)-Cy': (9, 13),
                                       'O(r)-Cy': (7, 13)},
                                'cv': {'CV1': (160, 9),
                                       'CV2': (133, 8)}}
        self._dict_dihed_defs = {}

    def calculate_distances(self, *args, **kwargs):
        """"""
        # TODO document this function
        # TODO find a way to take keyword type args with non-valid python
        # identifiers (e.g., "O-O").
        # Make empty atom selections to be appended to:
        first_group = self.select_atoms('protein and not protein')
        second_group = self.select_atoms('protein and not protein')
        column_names = []
        if len(args) == 0 and len(kwargs) == 0:
            args = ['all']
        if len(args) != 0:
            try:
                args = [arg.lower() for arg in args]
            except AttributeError:
                raise SyntaxError('All positional arguments must be strings')
            if 'pi' in args:
                args.remove('pi')
                warn('pi distances have not yet been implemented and will not'
                     ' be calculated.')
            if 'all' in args:
                args.remove('all')
                print('"all" given or implied, calculating distances for '
                      'oxygens and CVs')
                args.append('ox')
                args.append('cv')
            bad_args = []
            for arg in args:
                try:
                    temp_dict = self._dict_dist_defs[arg]
                    temp_dict.update(kwargs)
                    kwargs = temp_dict.copy()
                except KeyError:
                    bad_args.append(arg)
            if len(bad_args) != 0:
                warn('The following positional arguments were given but not '
                     'recognized: ' + str(bad_args) + '\nThey will be '
                     'ignored.')
        if len(kwargs) != 0:
            for key in kwargs:
                try:
                    atoms = kwargs[key].split()
                except AttributeError:
                    # assume it is iterable as is
                    atoms = kwargs[key]
                if len(atoms) != 2:
                    raise SyntaxError('This input should split to two atom '
                                      'indices: {}'.format(kwargs[key]))
                try:
                    [int(atom) for atom in atoms]
                except ValueError:
                    raise NotImplementedError('Only selection by atom index is'
                                              ' currently supported.\nAt your '
                                              'own risk you can try assigning '
                                              'to self._data[{}].'.format(key))
                first_group += self.select_atoms('bynum '+str(atoms[0]))
                second_group += self.select_atoms('bynum '+str(atoms[1]))
                column_names += [key]
        n1 = first_group.n_atoms
        n2 = second_group.n_atoms
        nc = len(column_names)
        if not nc == n1 == n2:
            raise SyntaxError('Different numbers of atom selections or number'
                              'of column labels '
                              '({}, {}, and {}, respectively).'.format(n1,
                                                                       n2,
                                                                       nc) +
                              '\nThis should not happen.')
        if self._num_frames != self.trajectory.n_frames:
            raise exceptions.FileChangedError()
        dists = np.zeros((self._num_frames, n1))
        for i, frame in enumerate(self.trajectory):
            MDa.lib.distances.calc_bonds(first_group.positions,
                                         second_group.positions,
                                         box=self.dimensions,
                                         result=dists[i])
        for i, column in enumerate(column_names):
            self._data[column] = dists[:, i]

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
            self.calculate_distances('ox')
        # might want to (optionally) return the time column here too
        # though, as a @property, this can't take arguments, so it would need
        # to be some variable in the class
        return self._data.filter(('O-O', 'O(l)-Cy', 'O(r)-Cy'))

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
            self.calculate_distances('pi')
        return self._data.filter(['pi-'+str(i) for i in range(16)])

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
            self.calc_open_closed()
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
            self.calc_open_closed()
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
            [[float(value[i][j]) for i in range(2)] for j in range(2)]
        except (TypeError, IndexError):
            raise TypeError('cutoffs must be an iterable of shape (2, 2)')
        except ValueError:
            raise SyntaxError('These values must be able to be cast as floats')
        self._oc_cutoffs = value

    def calc_open_closed(self):
        """
        Select the coordinates for open vs. closed TADDOL

        :return:
        """
        # I'm not sure this function is necessary. These queries might be
        # really fast already. I think it's nice to have to use the default
        # cutoffs and such.
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
            self.calculate_distances('cv')
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
            self.calculate_distances('cv')
        return self._data['CV2']

    def hist_2d_cvs(self, x=None, y=None, return_fig=True, ax=None, **kwargs):
        """"""
        # TODO make the constants here arguments
        # TODO make this optionally save figure
        if x is None:
            x = self.cv1_dists
        if y is None:
            y = self.cv2_dists
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
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

    def fes_2d_cvs(self, x=None, y=None, temp=205., ax=None, bins=None,
                   zrange=(0, 20, 11), zfinal=40, n_bins=32, transpose=False,
                   xlabel='CV 1', ylabel='CV 2',
                   **kwargs):
        """
        plot FES in 2D along defined CVs

        :param Iterable x: Default: self.cv1_dists. Length component to plot
            along x axis.
        :param Iterable y: Default: self.cv2_dists. Length component to plot
            along y axis.
        :param float temp: Default: 205. Temperature for Boltzmann weighting
            calculation.
        :param atplotlib.axes.Axes ax: Default: None. Axes on which to make the
            FES. If None, a new axes and figure will be created.
        :param Iterable bins: Default: None. The bins to be used for the z
            ranges. If this is not None, zrange and zfinal are ignored.
        :param zrange: Default: (0, 20, 11). Input to np.linspace for
            determining contour levels. If a float-like is given, it will be
            set as the max with 11+1 bins. If a len=2 list-like is given,
            it will be used as the min and max with 11+1 bins. Otherwise,
            the input will be used as-is for input to np.linspace.
        :type zrange: Iterable or Float
        :param zfinal: Default: 40. Energy at which to stop coloring the FES.
            Anything above this energy will appear as white.
        :param int n_bins: Default: 32. Number of bins in x and y for
            histogramming.
        :param bool transpose: Default: False. Whether to transpose the data
            and axes such that the input x will be along the y axis and the
            inverse. Note, this also makes the xlabel on the y-axis and the
            inverse.
        :param str xlabel: Default: 'CV 1'. Label for x-axis (or y-axis if
            transpose=True).
        :param str ylabel: Default: 'CV 2'. Label for y-axis (or x-axis if
            transpose=True).
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: The figure of the FES.
        :rtype: matplotlib.figure.Figure
        """
        # TODO make the constants here arguments
        # TODO make this optionally save figure
        if x is None and y is None:
            x = self.cv1_dists
            y = self.cv2_dists
            try:
                counts = self._cv_hist_data['counts']
                xedges = self._cv_hist_data['xedges']
                yedges = self._cv_hist_data['yedges']
            except KeyError:
                counts, xedges, yedges = np.histogram2d(x, y, n_bins)
                self._cv_hist_data['counts'] = counts
                self._cv_hist_data['xedges'] = xedges
                self._cv_hist_data['yedges'] = yedges
        else:
            if x is None:
                x = self.cv1_dists
            if y is None:
                y = self.cv2_dists
            counts, xedges, yedges = np.histogram2d(x, y, n_bins)
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
            vmax = bins[-1]
        probs = np.array([[i / counts.max() for i in j] for j in counts]) \
            + 1e-40
        r = 0.0019872  # kcal_th/(K mol)
        delta_g = np.array([[-r * temp * np.log(p) for p in j] for j in probs])
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        xmids, ymids = self._running_mean(xedges), self._running_mean(yedges)
        if not transpose:
            # This is because np.histogram2d returns the counts oddly
            delta_g = delta_g.transpose()
            _xlabel, _ylabel = xlabel, ylabel
        else:
            xmids, ymids = ymids, xmids
            _xlabel, _ylabel = ylabel, xlabel
        contours = ax.contourf(xmids, ymids, delta_g,
                               _bins, vmax=vmax, **kwargs)
        ax.axis((1.5, 10, 1.5, 10))
        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabel)
        ax.set_aspect('equal', 'box-forced')
        fig.colorbar(contours, label='kcal / mol')
        fig.tight_layout()
        return fig

    def plot_ox_dists(self, save=False, save_format='png',
                      save_base_name='ox-dists',
                      display=True, ax=None, **kwargs):
        """
        Plot the three oxygen-related distances.

        :param bool save: Default: False. Save the figure to disk.
        :param str save_format: Default: 'png'. Format in which to save the
            figure.
        :param str save_base_name: Default: 'ox-dists'. Name for the saved
            figure file.
        :param bool display: Default: True. Return the figure, otherwise
            return None.
        :param matplotlib.axes.Axes ax: Default: None. The axes object on
            which to make the plots. If None is supplied, a new axes object will
            be created.
        :param dict kwargs: Keywords to pass to the plotting function.
        :return: The figure of oxygen distances or None.
        """
        ox_dists = self.ox_dists
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(self._data['Time'], ox_dists['O-O'], label='O-O', **kwargs)
        ax.plot(self._data['Time'], ox_dists['O(l)-Cy'], label='O(l)-Cy',
                **kwargs)
        ax.plot(self._data['Time'], ox_dists['O(r)-Cy'], label='O(r)-Cy',
                **kwargs)
        ax.legend()
        ax.set_xlabel('time / ps')
        ax.set_ylabel('distance / $\mathrm{\AA}$')
        if save:
            fig.savefig(save_base_name + save_format)
        if display:
            return fig
        else:
            return None

    def hist_ox_dists(self, data=None, n_bins=10, save=False,
                      save_format='pdf', save_base_name='ox-dists-hist',
                      display=True, axes=None, **kwargs):
        """
        Make histogram of alcoholic O distances in TADDOL trajectory

        :param data: Default: self.ox_dists. Data to form the histogram from.
        :type data: pd.DataFrame
        :param int n_bins: Default: 10. Number of bins for histograms.
        :param bool save: Default: False. Save the figure to disk.
        :param str save_format: Default: 'pdf'. Format in which to save the
            figure.
        :param str save_base_name: Default: 'ox-dists-hist'. Name for the saved
            figure.
        :param bool display: Default: True. Return the figure from the function
            otherwise return None.
        :param axes: Default: None. The axes objects on
            which to make the plots. If None is supplied, new axes objects will
            be created.
        :param dict kwargs: Keyword arguments to pass to the plotting function.
        :return: The figure of histograms of oxygen distances.
        """
        try:
            data['O-O']
        except KeyError:
            raise InputError(data, 'data must be a pd.DataFrame like object '
                                   'with item O-O, O(l)-Cy, and O(r)-Cy.')
        except TypeError:
            if self._verbosity:
                print('Using default data: self.ox_dists.')
            data = self.ox_dists
        if axes is None:
            fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,
                                     sharex=True)
        else:
            try:
                fig = axes.flat[3].figure
            except (IndexError, TypeError):
                raise InputError('axes={}'.format(axes), 'Input axes must be '
                                 'able to plot at least four things')
            except AttributeError:
                try:
                    fig = axes[3].figure
                except IndexError:
                    raise InputError('axes={}'.format(axes), 'Input axes must '
                                     'be able to plot at least four things')
        handles = []
        # Use whatever the default colors for the system are
        # TODO find a more elegant way to do this
        colors = mpl.rcParams['axes.prop_cycle'].by_key().values()[0]
        for i, key in enumerate(('O-O', 'O(l)-Cy', 'O(r)-Cy')):
            n, bins = np.histogram(data[key], n_bins)
            ax = axes.flat[i]
            line, = ax.plot(bins[:-1], n, colors[i], **kwargs)
            handles.append(line)
            ax.set_ylabel(r'count')
            ax.set_xlabel(r'distance / $\mathrm{\AA}$')
        axes.flat[3].axis('off')
        axes.flat[3].legend(handles, ['O-O', 'O(l)-Cy', 'O(r)-Cy'],
                            loc='center')
        if save:
            fig.savefig(save_base_name + save_format)
        if display:
            return fig
        else:
            return None

    def fes_ox_dists(self, data=None, temp=791., save=False,
                     save_format='pdf',
                     save_base_name='ox-dists-fes',
                     display=True, axes=None, **kwargs):
        """
        Make FESs of the oxygen distances of a TADDOL from histogram data

        :param data: Default: self.ox_dists. Data to form the FES from.
        :type data: pd.DataFrame
        :param float temp: Default: 791 K. Temperature of the trajectory used
            to calculate the free energy.
        :param bool save: Default: False. Whether to save the FESs to disk.
        :param str save_format: Default: 'pdf'. Format in which to save the
            figure.
        :param str save_base_name: Default: 'ox-dists-fes'. Name of the saved
            figure.
        :param bool display: Default: True. Whether to return the figure after
            producing it.
        :param axes: Default: None. The axes objects on
            which to make the plots. If None is supplied, new axes objects will
            be created.
        :param kwargs: keyword arguments to pass to the plotter
        :return:
        """
        try:
            data['O-O']
        except KeyError:
            raise InputError(data, 'data must be a pd.DataFrame like object '
                                   'with item O-O, O(l)-Cy, and O(r)-Cy.')
        except TypeError:
            if self._verbosity:
                print('Using default data: self.ox_dists.')
            data = self.ox_dists
        if axes is None:
            fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,
                                     sharex=True)
        else:
            try:
                fig = axes.flat[3].figure
            except (IndexError, TypeError):
                raise InputError('axes={}'.format(axes), 'Input axes must be '
                                 'able to plot at least four things')
            except AttributeError:
                try:
                    fig = axes[3].figure
                except IndexError:
                    raise InputError('axes={}'.format(axes), 'Input axes must '
                                     'be able to plot at least four things')
        r = 0.0019872  # kcal_th/(K mol)
        delta_gs = []
        handles = []
        # Use whatever the default colors for the system are
        # TODO find a more elegant way to do this
        colors = mpl.rcParams['axes.prop_cycle'].by_key().values()[0]
        for i, key in enumerate(('O-O', 'O(l)-Cy', 'O(r)-Cy')):
            n, bins = np.histogram(data[key])
            # TODO find better way to account for zeros here rather than
            # just adding a small amount to each.
            prob = n.astype(float) / n.max() + 1e-20
            delta_g = -r * temp * np.log(prob)
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
            fig.savefig(save_base_name + save_format)
        if display:
            return fig
        else:
            return None


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


def plot_dist_array(array, index_offset=1, num_data_rows=None,
                    n_rows=None, n_cols=None):
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
    from math import sqrt, ceil
    if n_rows is None and n_cols is None:
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
