"""This defines classes for working with geometry file formats"""

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

import re
import numpy as np
from numpy.linalg import norm
from .exceptions import UnknownEnergyError, InputError
# TODO add tests for these


__all__ = ['rotation_matrix', 'Vector', 'XYZ', 'COM']



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    copied from
    https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis * np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


class Vector(np.ndarray):

    def __new__(cls, *xyz):
        if len(xyz) != 3:
            try:
                xyz = xyz[0]
            except IndexError:
                raise InputError(xyz, '3 values are required to make a vector')
        if len(xyz) != 3:
            raise InputError(xyz, 'Length of vector must be 3.')
        obj = super(Vector, cls).__new__(cls, shape=(3,),
                                         buffer=np.array(xyz, dtype=float))
        return obj

    def cross(self, vec):
        return np.cross(self, vec)

    def diff_angle(self, vec):
        return np.arccos(self.dot(vec) /
                         (norm(self) * norm(vec)))

    def rotate(self, axis, angle):
        r_mat = rotation_matrix(axis, angle)
        return np.dot(r_mat, self)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @property
    def mag(self):
        return norm(self)


class XYZ(object):
    def __init__(self, f_name):
        self.file = f_name
        with open(f_name, 'r') as f_file:
            f_lines = f_file.readlines()
        self._header = f_lines[0:2]
        if 'Energy' in self._header[1]:
            energy_match = re.search('(?:Energy:\s+)(-\d+\.\d+)',
                                     self._header[1])
            self._energy = float(energy_match.group(1))
        else:
            self._energy = None
        self._original_energy = self._energy
        data = [line.split() for line in f_lines[2:]]
        self.atoms = [atom[0] for atom in data]
        self.coords = [Vector([float(coord) for coord in atom[1:4]]) for
                       atom in data]

    def center_on(self, index):
        center = self.coords[index]
        self.coords = [coord - center for coord in self.coords]

    def rotate_to_x_axis_on(self, index):
        vec_x = Vector(1, 0, 0)
        angle = self.coords[index].diff_angle(vec_x)
        axis = self.coords[index].cross(vec_x)
        self.coords = [coord.rotate(angle, axis) for coord in self.coords]

    def center_and_rotate_on(self, index1, index2):
        self.center_on(index1)
        self.rotate_to_x_axis_on(index2)

    def __str__(self):
        f_string = ('   {0: <10s} {1.x: > 10.5f} {1.y: > 10.5f} '
                    '{1.z: > 10.5f}\n')
        output_list = list(self._header)
        output_list += [f_string.format(self.atoms[i], self.coords[i]) for i
                        in range(len(self.atoms))]
        return ''.join(output_list)

    @property
    def n_atoms(self):
        _n_atoms = len(self.atoms)
        _n_coords = len(self.coords)
        if _n_atoms != _n_coords:
            print('!!n atoms != n coords!! ({} != {})'.format(_n_atoms,
                                                              _n_coords))
        else:
            return _n_atoms

    @property
    def energy(self):
        if self._energy is None:
            raise UnknownEnergyError()
        return self._energy

    @property
    def original_energy(self):
        return self._original_energy

    def replace_coords(self, arg):
        if isinstance(arg, str):
            self.coords = XYZ(arg).coords.copy()
        else:
            self.coords = arg.coords.copy()
        self._energy = None  # Moved atoms, don't know energy

    def move_subset(self, movement, indices):
        if not isinstance(movement, Vector):
            movement = Vector(*movement)
        for index in indices:
            self.coords[index] = self.coords[index] + movement
        self._energy = None  # Moved atoms, don't know energy

    def write(self, f_name):
        with open(f_name, 'w') as f_file:
            f_file.write(str(self))

    def average_loc(self, *args):
        if len(args) == 1:  # if an Iterable is passed in
            args = args[0]
        total_vec = Vector(0, 0, 0)
        for i in args:
            total_vec = total_vec + self.coords[i]
        return total_vec / len(args)


class COM(XYZ):
    def __init__(self, f_name):
        self.file = f_name
        self._header = []
        self._title = []
        self._cm = []
        self.atoms = []
        self.coords = []
        self._footer = []
        with open(f_name, 'r') as f_file:
            f_lines = f_file.readlines()
        self._parser(f_lines)

    def _parser(self, lines):
        section = 'header'
        data = []
        for line in lines:
            if section == 'header':
                self._header.append(line)
                if line.strip() == '':
                    section = 'title'
                continue
            elif section == 'title':
                self._title.append(line)
                if line.strip() == '':
                    section = 'charge_mult'
                continue
            elif section == 'charge_mult':
                self._cm.append(line)
                section = 'geom'
                continue
            elif section == 'geom':
                if line.strip() == '':
                    section = 'opt_input'
                    continue
                data.append(line.split())
                continue
            elif section == 'opt_input':
                self._footer.append(line)
                continue
        self.atoms = [atom[0] for atom in data]
        self.coords = [Vector([float(coord) for
                               coord in atom[1:4]]) for atom in data]

    def __str__(self):
        f_string = ('   {0: <10s} {1.x: > 10.5f} {1.y: > 10.5f} '
                    '{1.z: > 10.5f}\n')
        output_list = list(self._header)
        output_list += list(self._title)
        output_list += list(self._cm)
        output_list += [f_string.format(self.atoms[i], self.coords[i]) for i
                        in range(len(self.atoms))]
        output_list += ['\n']
        output_list += list(self._footer)
        return ''.join(output_list)
