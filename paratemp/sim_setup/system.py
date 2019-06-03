"""This contains code for setting up a combination of molecules for MD calcs"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2019.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2019 Thomas J. Heavey IV                                   #
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

import logging
from pathlib import Path
from typing import Dict

import parmed

from . import Molecule


__all__ = ['System']


GroTopFile = parmed.gromacs.GromacsTopologyFile
ParmedRes = parmed.topologyobjects.Residue


log = logging.getLogger(__name__)
if not log.hasHandlers():
    level = logging.INFO
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


class System(object):

    def __init__(self, *args: Molecule,
                 name: str = 'default',
                 shift: bool = True,
                 spacing: float = 2.0,
                 include_gbsa: bool = True):
        log.debug('Initializing System with {} Molecules'.format(len(args)))
        self._name = name
        for arg in args:
            if not isinstance(arg, Molecule):
                raise TypeError(
                    'positional arguments must of type Molecule; given '
                    '{}'.format(type(arg)))
        self._directory = Path(self._name).resolve()
        self._directory.mkdir()
        ptop = args[0].topology.copy()  # type: GroTopFile
        for mol in args[1:]:
            ptop += mol.topology
        self._ptop = ptop
        if shift:
            self._shift_to_nonoverlapping(spacing)
        top_path = self._directory / '{}.top'.format(self._name)
        ptop.write(str(top_path))
        if include_gbsa:
            self._add_gbsa_include(top_path)
        ptop.save(str(self._directory / 'rough_{}.gro'.format(self._name)))
        log.info('Wrote combined topology and geometry files in {}'.format(
            self._directory))

    @staticmethod
    def _get_res_max_z(res: ParmedRes) -> float:
        """Return the max z coordinate for any atom in the given residue

        Picked the z-axis because if aligned along moments of inertia,
        z is likely to be the shortest, keeping the box required smaller."""
        return max((a.xz for a in res.atoms))

    @staticmethod
    def _get_all_res_max_x(ptop: GroTopFile) -> Dict[ParmedRes, float]:
        return {res: System._get_res_max_z(res) for res in ptop.residues}

    def _shift_to_nonoverlapping(self, spacing: float = 2.0):
        log.info('Shifting molecules in z direction to prevent overlap')
        z_maxes = self._get_all_res_max_x(self._ptop)
        shifts, shift = dict(), 0
        for res in z_maxes:
            shifts[res] = shift
            shift += spacing + z_maxes[res]
        for atom in self._ptop.atoms:
            atom.xz += shifts[atom.residue]

    @staticmethod
    def _add_gbsa_include(path: Path):
        log.info('Adding lines to include implicit solvation parameters')
        to_add = ('; Include parameters for implicit solvation\n'
                  '#include '
                  '/projectnb/nonadmd/theavey/GROMACS-basics/gbsa_all.itp\n\n')
        temp_path = path.with_suffix(path.suffix + '.temp')
        lines = path.read_text().splitlines(keepends=True)
        with temp_path.open('w') as temp_file:
            post_defaults, done = False, False
            for line in lines:
                if done:
                    pass
                elif post_defaults:
                    if not line.strip():
                        line += to_add
                        done = True
                elif line == '[ defaults ]\n':
                    post_defaults = True
                temp_file.write(line)
        bak_path = path.with_suffix(path.suffix + '.bak0')
        if bak_path.exists():
            bak_path = bak_path.with_suffix('.bak' +
                                            str(int(str(bak_path)[-1])+1))
        path.rename(bak_path)
        temp_path.rename(path)
