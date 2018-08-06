#! /usr/bin/env python

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2018.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017, 2018 Thomas J. Heavey IV                             #
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
from typing import Iterable, Sequence

from paratemp.plotting import fes_array_3_legend, plot_dist_array, fes_1d
from paratemp.utils import calc_fes_2d, calc_fes_1d, _parse_ax_input, \
    _parse_z_bin_input
from .exceptions import InputError, FileChangedError


__all__ = ['Universe', 'Taddol']


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
        self.temperature = kwargs.pop('temp', None)
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
                store_cols = store.get_node(time).axis0.read().astype(str)
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
                    return
        for key in keys_to_read:
            self._data[key] = read_df[key]

    def calculate_distances(self, recalculate=False, ignore_file_change=False,
                            read_data=True, save_data=True,
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
        :param bool read_data: Default: True.
            If True, :func:`read_data` will be used to read any data in the
            default file with `ignore_no_data=True`.
        :param bool save_data: Default: True.
            If True, :func:`save_data` will be used to save the calculated
            distances to the default file.
            Nothing will be saved if there is nothing new to calculate.
        :param args:
        :param kwargs:
        :return: None
        :raises: FileChangedError
        :raises: SyntaxError
        :raises: NotImplementedError
        """
        # TODO document this function
        # TODO find a way to take keyword type args with non-valid python
        # identifiers (e.g., "O-O").
        if read_data:
            v = self._verbosity
            self._verbosity = False
            self.read_data(ignore_no_data=True)
            self._verbosity = v
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
            return
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
                raise FileChangedError()
        dists = np.zeros((self._num_frames, n1))
        for i in range(self._num_frames):
            self.trajectory[i]
            MDa.lib.distances.calc_bonds(first_group.positions,
                                         second_group.positions,
                                         box=self.dimensions,
                                         result=dists[i])
        for i, column in enumerate(column_names):
            self._data[column] = dists[:, i]
        if save_data:
            self.save_data()

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
            raise FileChangedError()
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

    def select_frames(self, criteria, name):
        d = dict()
        for key in criteria:
            d[key+'_min'] = self.data[key] > criteria[key][0]
            d[key+'_max'] = self.data[key] < criteria[key][1]
        self._data[name] = pd.DataFrame(d).all(axis=1)
        if self._verbosity:
            num = len(self.data[self.data[name]])
            plural = 's' if num != 1 else ''
            print('These criteria include {} frame{}'.format(num, plural))
        return self.data.index[self.data[name]]


    def fes_1d(self, data, bins=None, temp=None,
               xlabel=r'distance / $\mathrm{\AA}$', ax=None, **kwargs):
        """
        Make FES of some time series data

        :type data: Iterable or str
        :param data: Data to form the FES from. If a string is given, the data
            will be taken from self.data[data].

        :param float temp: Default: None. Temperature for Boltzmann weighting
            calculation.
            If None is provided, the temperature will be taken from
            self.temperature

        :type bins: int or Sequence[int or float] or str
        :param bins: Default: None. The bins argument to be passed to
            np.histogram

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
        _temp = self._parse_temp_input(temp)
        _data = self._parse_data_input(data)
        return fes_1d(x=_data, temp=_temp, ax=ax, bins=bins, xlabel=xlabel,
                      **kwargs)

    def fes_2d(self, x, y, temp=None, ax=None, bins=None,
               zrange=(0, 20, 11), zfinal=40, n_bins=32, transpose=False,
               xlabel='x', ylabel='y', scale=True, square=True,
               **kwargs):
        """
        plot FES in 2D along defined values

        :type x: Iterable or str
        :param x: Value along x axis to plot. If a string is given, the data
            will be taken from self.data[x].
        :type y: Iterable or str
        :param y: Value along y axis to plot. If a string is given, the data
            will be taken from self.data[y].
        :param float temp: Default: None. Temperature for Boltzmann weighting
            calculation.
            If None is provided, the temperature will be taken from
            self.temperature
        :param matplotlib.axes.Axes ax: Default: None. Axes on which to make
            the FES. If None, a new axes and figure will be created.
        :param Iterable bins: Default: None. The bins to be used for the z
            ranges. If this is not None, zrange and zfinal are ignored.
        :type zrange: Iterable or float
        :param zrange: Default: (0, 20, 11). Input to np.linspace for
            determining contour levels. If a float-like is given, it will be set
            as the max with 11+1 bins. If a len=2 list-like is given, it will be
            used as the min and max with 11+1 bins. Otherwise, the input will
            be used as-is for input to np.linspace.
        :param zfinal: Default: 40. Energy at which to stop coloring the FES.
            Anything above this energy will appear as white.
        :type n_bins: int or (int, int) or (int, np.ndarray) or (np.ndarray,
            int) or (np.ndarray, np.ndarray)
        :param n_bins: Default: 32. Number of bins in x and y for
            histogramming. This uses np.histogram2d which is fairly flexible
            in how the bins can be specified. See `their documentation
            <https://docs.scipy.org/doc/numpy/reference/generated/numpy
            .histogram2d.html>`.
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
        :param bool square: Default: True.
            If True, the plot will be made square with `ax.set_aspect(
            'equal', 'box')`.
            If False, the aspect ratio will be the default value.
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: The delta G values, the bin centers, the contours, the figure,
            and the axes
        :rtype: tuple[np.ndarray, tuple[np.ndarray, np.ndarray],
            matplotlib.contour.QuadContourSet, matplotlib.figure.Figure,
            matplotlib.axes.Axes]
        """
        _temp = self._parse_temp_input(temp)
        _x = self._parse_data_input(x)
        _y = self._parse_data_input(y)
        _bins, vmax = _parse_z_bin_input(bins, zfinal, zrange)
        delta_g, xmids, ymids = calc_fes_2d(_x, _y, temp=_temp, bins=n_bins)
        fig, ax = _parse_ax_input(ax)
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
        if square:
            ax.set_aspect('equal', 'box')
        if scale:
            fig.colorbar(contours, label='kcal / mol')
        fig.tight_layout()
        return delta_g, (xmids, ymids), contours, fig, ax

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

    def _parse_temp_input(self, temp):
        if temp is None:
            _temp = self.temperature
        else:
            _temp = temp
        if _temp is None:
            raise ValueError('The temperature must be defined to calculate an'
                             ' FES')
        return _temp

    def _parse_data_input(self, x):
        if type(x) is str:
            try:
                return self.data[x]
            except KeyError:
                raise InputError(x, 'input as a str must be an existing '
                                    'key for the data in this object')
        else:
            return x


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
            raise FileChangedError()
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
        ax.axis((1.5, 10, 1.5, 10))
        ax.set_xlabel('CV 2')
        ax.set_ylabel('CV 1')
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        if return_fig:
            return fig
        else:
            return counts, xedges, yedges, fig, ax

    def fes_2d_cvs(self, x=None, y=None, temp=205.,
                   xlabel='CV 1', ylabel='CV 2',
                   **kwargs):
        """
        plot FES in 2D along defined CVs

        See also documentation for :func:`Universe.fes_2d`.

        :param Iterable x: Default: self.cv1_dists. Length component to plot
            along x axis.
        :param Iterable y: Default: self.cv2_dists. Length component to plot
            along y axis.
        :param float temp: Default: 205. Temperature for Boltzmann weighting
            calculation.
        :param str xlabel: Default: 'CV 1'. Label for x-axis (or y-axis if
            transpose=True).
        :param str ylabel: Default: 'CV 2'. Label for y-axis (or x-axis if
            transpose=True).
        :param kwargs: Keyword arguments to pass to the plotting function.
        :return: The figure of the FES.
        :rtype: matplotlib.figure.Figure
        """
        if x is None:
            x = self.cv1_dists
        if y is None:
            y = self.cv2_dists
        fig = self.fes_2d(x=x, y=y, temp=temp, xlabel=xlabel, ylabel=ylabel,
                          **kwargs)[-2]
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

    def fes_ox_dists(self, data=None, temp=791., bins=None, save=False,
                     save_format='pdf',
                     save_base_name='ox-dists-fes',
                     display=True, axes=None, **kwargs):
        """
        Make FESs of the oxygen distances of a TADDOL from histogram data

        :param data: Default: self.ox_dists. Data to form the FES from.
        :type data: pd.DataFrame
        :param float temp: Default: 791 K. Temperature of the trajectory used
            to calculate the free energy.
        :type bins: int or Sequence[int or float] or str
        :param bins: Default: None. The bins argument to be passed to
            np.histogram
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
                                   'with items O-O, O(l)-Cy, and O(r)-Cy.')
        except TypeError:
            if self._verbosity:
                print('Using default data: self.ox_dists.')
            data = self.ox_dists
        fig = fes_array_3_legend(data, temp=temp, labels=('O-O', 'O(l)-Cy',
                                                          'O(r)-Cy'),
                                 axes=axes, bins=bins, **kwargs)[3]
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
    warn('get_dist will soon be deprecated. Use '
         'Universe.calculate_distances', DeprecationWarning)
    if box is not None:
        coordinates = (np.array([atom.centroid()]) for atom in (a, b))
        return MDa.lib.distances.calc_bonds(*coordinates,
                                            box=box)
    else:
        return np.linalg.norm(a.centroid() - b.centroid())


def get_dist_dict(dictionary, a, b, box=None):
    """Calculate distance using dict of AtomSelections"""
    warn('get_dist_dict will soon be deprecated. Use '
         'Universe.calculate_distances', DeprecationWarning)
    return get_dist(dictionary[a], dictionary[b], box=box)


def get_angle(a, b, c, units='rad'):
    """Calculate the angle between ba and bc for AtomGroups a, b, c"""
    warn('get_angle will soon be deprecated. Implement '
         'Universe.calculate_angles', DeprecationWarning)
    # TODO look at using the MDAnalysis builtin function
    b_center = b.centroid()
    ba = a.centroid() - b_center
    bc = c.centroid() - b_center
    angle = np.arccos(np.dot(ba, bc) /
                      (np.linalg.norm(ba) * np.linalg.norm(bc)))
    if units == 'rad':
        return angle
    elif units == 'deg':
        return np.rad2deg(angle)
    else:
        raise InputError(units,
                         'Unrecognized units: '
                         'the two recognized units are rad and deg.')


def get_angle_dict(dictionary, a, b, c, units='rad'):
    """Calculate angle using dict of AtomSelections"""
    warn('get_angle_dict will soon be deprecated. Implement '
         'Universe.calculate_angles', DeprecationWarning)
    return get_angle(dictionary[a], dictionary[b], dictionary[c],
                     units=units)


def get_dihedral(a, b, c, d, units='rad'):
    """Calculate the angle between abc and bcd for AtomGroups a,b,c,d

    Based on formula given in
    https://en.wikipedia.org/wiki/Dihedral_angle"""
    warn('get_dihedral will soon be deprecated. Use '
         'Universe.calculate_dihedrals', DeprecationWarning)
    # TODO look at using the MDAnalysis builtin function
    ba = a.centroid() - b.centroid()
    bc = b.centroid() - c.centroid()
    dc = d.centroid() - c.centroid()
    angle = np.arctan2(
        np.dot(np.cross(np.cross(ba, bc), np.cross(bc, dc)), bc
               ) / np.linalg.norm(bc),
        np.dot(np.cross(ba, bc), np.cross(bc, dc)))
    if units == 'rad':
        return angle
    elif units == 'deg':
        return np.rad2deg(angle)
    else:
        raise InputError(units,
                         'Unrecognized units: '
                         'the two recognized units are rad and deg.')


def get_dihedral_dict(dictionary, a, b, c, d, units='rad'):
    """Calculate dihedral using dict of AtomSelections"""
    warn('get_dihedral_dict will soon be deprecated. Use '
         'Universe.calculate_dihedrals', DeprecationWarning)
    return get_dihedral(dictionary[a], dictionary[b],
                        dictionary[c], dictionary[d],
                        units=units)


def get_taddol_ox_dists(universe, sel_dict=False):
    """Get array of oxygen distances in TADDOL trajectory"""
    warn('get_taddol_ox_dists will soon be deprecated. Use Taddol.ox_dists',
         DeprecationWarning)
    if not sel_dict:
        sel_dict = get_taddol_selections(universe)
    output = []
    for frame in universe.trajectory:
        box = universe.dimensions
        output.append((universe.trajectory.time,
                       get_dist_dict(sel_dict, 'aoxl', 'aoxr', box=box),
                       get_dist_dict(sel_dict, 'aoxl', 'cyclon', box=box),
                       get_dist_dict(sel_dict, 'aoxr', 'cyclon', box=box)))
    return np.array(output)


def make_plot_taddol_ox_dists(data, save=False, save_format='pdf',
                              save_base_name='ox_dists',
                              display=True):
    """Make plot of alcoholic O distances in TADDOL trajectory"""
    warn('make_plot_taddol_ox_dists will soon be deprecated. Use '
         'Taddol.plot_ox_dists',
         DeprecationWarning)
    fig, axes = plt.subplots()
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
    warn('make_hist_taddol_ox_dists will soon be deprecated. Use '
         'Taddol.hist_ox_dists',
         DeprecationWarning)
    legend_entries = ['O-O', 'O(l)-Cy', 'O(r)-Cy']
    if separate:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
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
        fig, ax = plt.subplots()
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
    warn('get_taddol_pi_dists will soon be deprecated. Use '
         'Taddol.pi_dists',
         DeprecationWarning)
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
    return np.array(output)


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


def make_fes_taddol_ox_dist(dists, temp=791., bins=None, save=False,
                            save_format='pdf',
                            save_base_name='ox_dists_fes',
                            display=True, **kwargs):
    """Plot the relative free energy surface of O distances in TADDOL"""
    warn('make_fes_taddol_ox_dist will soon be deprecated. Use '
         'Taddol.fes_ox_dists',
         DeprecationWarning)
    delta_gs = []
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    handles = []
    # Use whatever the default colors for the system are
    # TODO find a more elegant way to do this
    colors = mpl.rcParams['axes.prop_cycle'].by_key().values()[0]
    for i in range(3):
        delta_g, bin_mids = calc_fes_1d(dists[:, 1 + i], temp=temp, bins=bins)
        delta_gs.append(delta_g)
        ax = axes.flat[i]
        line, = ax.plot(bin_mids, delta_g, colors[i], **kwargs)
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
    warn('select_open_closed_dists will soon be deprecated. Use '
         'Taddol.calc_open_closed',
         DeprecationWarning)
    cut_closed = cutoffs[0]
    cut_open = cutoffs[1]
    set_open = []
    set_closed = []
    for ts in dists:
        if cut_open[0] <= ts[1] <= cut_open[1]:
            set_open.append(ts)
        if cut_closed[0] <= ts[1] <= cut_closed[1]:
            set_closed.append(ts)
    columns = ['Time', 'O-O', 'Ol-Cy', 'Or-Cy']
    return pd.DataFrame(set_open, columns=columns), \
        pd.DataFrame(set_closed, columns=columns)
