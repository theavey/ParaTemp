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

import re
from vpython import vector
import vpython.cyvector as cyvector
# TODO figure out this import above
from ParaTemp.exceptions import UnknownEnergyError
# TODO add tests for these


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
        self.coords = [vector(*[float(coord) for coord in atom[1:4]]) for
                       atom in data]

    def center_on(self, index):
        center = self.coords[index]
        self.coords = [coord - center for coord in self.coords]

    def rotate_to_x_axis_on(self, index):
        vec_x = vector(1, 0, 0)
        angle = self.coords[index].diff_angle(vec_x)
        axis = self.coords[index].cross(vec_x)
        self.coords = [coord.rotate(angle, axis) for coord in self.coords]

    def center_and_rotate_on(self, index1, index2):
        self.center_on(index1)
        self.rotate_to_x_axis_on(index2)

    def __str__(self):
        f_string = ('   {0: <10s} {1.x: > 10.5f} {1.y: > 10.5f} '
                    '{1.z: > 10.5f}\n')
        output_list = self._header.copy()
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
        if type(arg) is str:
            self.coords = XYZ(arg).coords.copy()
        else:
            self.coords = arg.coords.copy()
        self._energy = None  # Moved atoms, don't know energy

    def move_subset(self, movement, indicies):
        if type(movement) is not cyvector.vector:
            movement = vector(*movement)
        for index in indicies:
            self.coords[index] = self.coords[index] + movement
        self._energy = None  # Moved atoms, don't know energy

    def write(self, f_name):
        with open(f_name, 'w') as f_file:
            f_file.write(str(self))

    def average_loc(self, *args):
        if len(args) == 1:  # if an Iterable is passed in
            args = args[0]
        total_vec = vector(0, 0, 0)
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
        self.coords = [vector(*[float(coord) for
                                   coord in atom[1:4]]) for atom in data]

    def __str__(self):
        f_string = ('   {0: <10s} {1.x: > 10.5f} {1.y: > 10.5f} '
                    '{1.z: > 10.5f}\n')
        output_list = self._header.copy()
        output_list += self._title.copy()
        output_list += self._cm.copy()
        output_list += [f_string.format(self.atoms[i], self.coords[i]) for i
                        in range(len(self.atoms))]
        output_list += ['\n']
        output_list += self._footer.copy()
        return ''.join(output_list)
