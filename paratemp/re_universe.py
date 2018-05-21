"""
Contains a class for containing multiple replica Universes in a single object.

This module defines a class that will contain Universes for each replica in a
replica exchange simulation.
While each replica can be defined individually, this adds a convenient
container for logically similar trajectories and allows easy iteration over
and access to the individual replica Universes.
"""

########################################################################
#                                                                      #
# This module was written by Thomas Heavey in 2018.                    #
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


from __future__ import absolute_import, print_function

import collections
import errno
import glob
import numpy as np
import os
import six
from typing import Iterable

from .tools import find_nearest_idx
from .coordinate_analysis import Universe
from . import get_temperatures, exceptions


__all__ = ['REUniverse']


class REUniverse(collections.Sequence):
    """
    A class for working with MDAnalysis Universes from replica exchange sims.

    This class creates Universe objects for different simulations that all
    have the same topology (atoms, bonds, etc.), but different temperatures
    (not strictly required, though a 'temperature' needs to be specified for
    each).
    This class itself does not add much on top of
    :class:`~paratemp.coordinate_analysis.Universe` other than iteration over
    the replicas and creating them starting all from the same topology with
    different trajectories.

    An instance of this class can be indexed (with '[]'s) with either ints
    (to get the Universe with that index) or with strings that can be
    converted to floats to get the replica with the temperature nearest to
    that value.

    >>> reu = REUniverse('test.gro', 'simulation_folder', trajs=['cold.xtc', \
    'warm.xtc'], temps=[100, 200])
    <REUniverse with 2 replicas>
    >>> reu[0].temperature
    100.0
    >>> reu['75'].temperature
    100.0
    >>> print([u.temperature for u in reu])
    [100.0, 200.0]
    >>> len(reu)
    2
    >>> reu.keys()
    ('100.0', '200.0')

    """

    def __init__(self, topology, base_folder,
                 trajs=None, traj_glob='*.xtc',
                 temps='TOPO/temperatures.dat'):
        """
        Instatiate a replica exchange universe for a set of replica trajectories

        :param str topology: Name of the topology file (such as a .gro or
            .pdb file).
        :param str base_folder: Name of the folder in which to look for the
            trajectories, topology, temperature file, and others.
        :param Iterable[str] trajs: List of the trajectory files. If this
            is None, traj_glob will be used instead.
            The files can be listed either relative to the current directory
            (checked first) or relative to `base_folder`.

            **Be aware**: the order of these does not matter because they will
            be sorted alphanumerically and then by length. This should be
            fine if all the trajectory names only differ by the value of
            some index, but in other cases, this could cause issues with
            unexpected ordering or incorrect matching of temperatures to
            trajectories.
        :param str traj_glob: If `trajs` is None, this string will be glob
            expanded to find the trajectories.
        :type temps: str or Iterable
        :param temps: Temperatures or path to file with temperatures of the
            replicas in the simulations.
            If a string is provided, it is assumed to be a path relative to
            the current directory or `base_folder`.
            Otherwise, it is assumed to be an iterable of values that can be
            cast to floats.
        """
        self.base_folder = os.path.abspath(base_folder)
        self._top = self._fn(topology)
        self._trajs = self._get_trajs(trajs, traj_glob)
        self._temps = self._get_temps(temps)
        if len(self._temps) != len(self._trajs):
            raise ValueError(
                'len of temps ({}) not same'.format(len(self._temps)) +
                ' as len of trajs ({})'.format(len(self._trajs)))
        self._trajs.sort()
        self._trajs.sort(key=len)
        # TODO find more sure way to match temp to trajectory
        self.universes = np.array(
            [Universe(self._top, t, temp=self._temps[i])
             for i, t in enumerate(self._trajs)])

    def _fn(self, path):
        """
        Return absolute path to file relative to either here or base_folder

        :param str path: (Relative path and) file to look for in current
            directory or relative to `base_folder`. Current directory is
            checked first and the absolute path to that is returned if it is
            found there. Otherwise, the file is searched for relative to
            `base_folder`. If it's not found there either, FileNotFoundError
            is raised
        :return: Absolute path to path
        :rtype: str

        :raises: OSError if the file is not found relative to current
            directory or `base_folder`.
        """
        if os.path.isfile(path):
            return os.path.abspath(path)
        elif os.path.isfile(os.path.join(self.base_folder, path)):
            return os.path.abspath(os.path.join(self.base_folder, path))
        else:
            raise OSError(errno.ENOENT,
                          '{} not found here or under base_folder'.format(path))

    def _get_temps(self, temps):
        """
        Get the temperatures for the set of replicas

        :type temps: str or Iterable
        :param temps: Either the path to the file with the temperatures or a
            list-like of the temperatures.

            If a string is given, it will be processed as being absolute,
            relative to current dir., or relative to base_folder. This uses
            :func:`paratemp.tools.get_temperatures` to read the file.

            If the input is not string-like, it will be converted to a
            :func:`np.ndarray` of floats.

        :rtype: np.ndarray
        :return: The temperatures as floats
        """
        if isinstance(temps, six.string_types):
            return get_temperatures(self._fn(temps))
        else:
            return np.array([float(t) for t in temps])

    def _get_trajs(self, trajs, traj_glob):
        """
        Get paths to trajectory files

        This method will first see if trajs is not None.
        If it is None, it will try to glob expand in the current path then
        from base_folder if the first glob returns an empty list.

        If none of these works, FileNotFoundError will be raised.

        For the first one that works, the list of files will be expanded to
        absolute paths and returned as a list of strings of the paths.

        :param Iterable trajs: If this is not None, this will be taken as a
            list of paths to the trajectory files. The order here does
            not matter because they will be sorted in
            :meth:`~paratemp.re_universe.REUniverse.__init__` (as it is
            currently implemented).
        :param str traj_glob: A string which can be glob expanded to give
            the trajectory files (and only the relevant trajectory files).
        :rtype: list
        :return: A list of the absolute paths to the trajectories.
        :raises: OSError if trajs is None and glob expansion both here and
            in base_folder give empty lists.
        """
        if trajs is not None:
            return [self._fn(t) for t in trajs]
        elif isinstance(traj_glob, six.string_types):
            g1 = glob.glob(traj_glob)
            g2 = glob.glob(os.path.join(self.base_folder, traj_glob))
            if g1:
                return [self._fn(t) for t in g1]
            elif g2:
                return [self._fn(t) for t in g2]
            else:
                raise OSError(errno.ENOENT,
                              '{} did not seem to lead'.format(traj_glob) +
                              ' to any files here or under base_folder')
        else:
            raise exceptions.InputError((trajs, traj_glob),
                                        'use trajs or traj_glob '
                                        'to find trajectory files')

    def __getitem__(self, i):
        """
        Get one of the replica universes by index or temperature

        If `i` is an int, then this will return the Universe with `i` as its
        index.
        If `i` is a string, it is assumed to be a float, and the replica
        Universe with the temperature absolutely nearest to `i` will be
        returned.

        :type i: int or str
        :param i: The index of the universe to be returned or a string of the
            temperature closest to the temp or the Universe to be returned.
        :rtype: Universe
        :return: The Universe indicated by `i`.
        """
        if isinstance(i, int):
            if i > len(self) - 1:
                raise IndexError(
                    'index {} is '.format(i) +
                    'larger than the number of replicas present')
            return self.universes[i]
        if isinstance(i, six.string_types) or isinstance(i, float):
            return self.universes[find_nearest_idx(self._temps, float(i))]

    def __len__(self):
        """
        Return the number of replicas in the object.

        :rtype: int
        :return: The number of replicas
        """
        return len(self.universes)

    def __repr__(self):
        return '<REUniverse with {} replicas>'.format(len(self))

    def keys(self):
        """
        Return the temperatures of the replicas (as can be used for indexing)

        As currently implemented, these are rounded to the nearest tenths
        place, but this may change in the future.

        :rtype: tuple[str]
        :return: The temperatures of the replicas
        """
        # TODO possibly change precision based on spread of temperatures
        # (tenths might not always be precise enough for large systems)
        return ('{:.1f}'.format(t) for t in self._temps)

    def values(self):
        """
        Return the universes.

        :rtype: np.ndarray[Universe]
        :return: The numpy array of the Universe objects
        """
        return self.universes

    def items(self):
        """
        Return a list of key-value pairs

        :rtype: list[tuple[str, Universe]]
        :return: A list of key-value pair tuples

        """
        return zip(self.keys(), self.values())
