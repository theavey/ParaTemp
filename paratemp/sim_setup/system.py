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
import re
from typing import Dict

import parmed
import pkg_resources

from . import Molecule


__all__ = ['System']


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


GroTopFile = parmed.gromacs.GromacsTopologyFile
ParmedRes = parmed.topologyobjects.Residue


gbsa_itp = pkg_resources.resource_string(__name__,
                                         'SimpleSim_data/gbsa_all.itp')


def get_gbsa_itp(directory: Path):
    to_path = directory / 'gbsa_all.itp'
    to_path.write_bytes(gbsa_itp)
    return to_path.resolve()


class System(object):
    """
    The System class is intended to combine several Molecules.

    It will add the molecules together into one box and optionally shift the
    molecules in the z direction to make sure the molecules are no longer
    overlapping.
    It can also optionally add GBSA implicit solvation parameters based on
    Amber03.

    :param args: molecules to combine into this object
    :param name: name of the system
    :param shift: If True, the molecules will be moved to be non-overlapping
    :param spacing: distance to put between the molecules (in angstroms)
    :param include_gbsa: If True, GBSA parameters for implicit solvation will
        be included in the topology file
    :param box_length: Length of cubic box (in angstroms)
    """

    def __init__(self, *args: Molecule,
                 name: str = 'default',
                 shift: bool = True,
                 spacing: float = 2.0,
                 include_gbsa: bool = True,
                 box_length: float = 25.0):
        log.debug('Initializing System with {} Molecules'.format(len(args)))
        self._name = name
        for arg in args:
            if not isinstance(arg, Molecule):
                raise TypeError(
                    'positional arguments must of type Molecule; given '
                    '{}'.format(type(arg)))
        self.n_molecules = len(args)
        self._directory = Path(self._name).resolve()
        self._directory.mkdir()
        ptop = args[0].topology.copy(GroTopFile)  # type: GroTopFile
        for mol in args[1:]:
            ptop += mol.topology
        self._ptop = ptop
        if shift:
            self._shift_to_nonoverlapping(spacing)
        if box_length:
            ptop.box = [box_length, box_length, box_length, 90.0, 90.0, 90.0]
        self.atom_types = set(a.type for a in ptop.atoms)
        top_path = self._directory / '{}.top'.format(self._name)
        ptop.write(str(top_path))
        if include_gbsa:
            self._add_gbsa_include(top_path)
        self.top_path = top_path
        gro_path = self._directory / 'rough_{}.gro'.format(self._name)
        ptop.save(str(gro_path))
        self.gro_path = gro_path
        log.info('Wrote combined topology and geometry files in {}'.format(
            self._directory))

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def name(self) -> str:
        return self._name

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

    def _add_gbsa_include(self, path: Path):
        log.info('Adding lines to include implicit solvation parameters')
        directory = path.parent
        path_gbsa_all_itp = get_gbsa_itp(directory)
        if not path_gbsa_all_itp.is_file():
            raise FileNotFoundError('Could not create or find "gbsa_all.itp"')
        path_gbsa_itp = path_gbsa_all_itp.with_name('gbsa.itp')
        lines = ['[ implicit_genborn_params ]\n',
                 '; atype      sar      st     pi       gbr       hct\n']
        param_dict = self._make_gbsa_dict()
        try:
            lines += [param_dict[at] for at in self.atom_types]
        except KeyError as e:
            message = 'No GBSA parameters for atomtype "{}"'.format(
                e.args[0])
            log.error(message)
            raise KeyError(e.args[0], message)
        path_gbsa_itp.write_text(''.join(lines))
        to_add = ('; Include parameters for implicit solvation\n'
                  '#include "{}"\n\n'.format(path_gbsa_itp))
        temp_path = path.with_suffix(path.suffix + '.temp')
        lines = path.read_text().splitlines(keepends=True)
        with temp_path.open('w') as temp_file:
            post_atomtypes, done = False, False
            for line in lines:
                if done:
                    pass
                elif post_atomtypes:
                    if not line.strip():
                        line += to_add
                        done = True
                elif line.strip() == '[ atomtypes ]':
                    post_atomtypes = True
                temp_file.write(line)
        bak_path = path.with_suffix(path.suffix + '.bak0')
        if bak_path.exists():
            bak_path = bak_path.with_suffix('.bak' +
                                            str(int(str(bak_path)[-1])+1))
        path.rename(bak_path)
        temp_path.rename(path)

    @staticmethod
    def _make_gbsa_dict() -> dict:
        """
        get dict of atom types to GBSA parameter lines

        Uses :variable:`gbsa_itp` as the input file path to search for
        parameters.

        :return: dict of atom types to GBSA parameter lines
        """
        log.info('Finding implicit solvation parameters '
                 'from gbsa_all.itp')
        in_gbsa_params_sec = False
        param_dict = dict()
        for line in str(gbsa_itp, 'utf-8').splitlines(keepends=True):
            _line = line.split(sep=';', maxsplit=1)[0].strip()
            if not _line:
                continue
            match = re.search(r'\[\s+(\w+)\s+\]', _line)
            if match:
                if in_gbsa_params_sec:
                    break  # reached next section
                if match.group(1) == 'implicit_genborn_params':
                    in_gbsa_params_sec = True
                continue
            elif not in_gbsa_params_sec:
                continue
            param_dict[_line.split()[0]] = line
        if not param_dict:
            raise ValueError('No GBSA parameters found in gbsa_all.itp')
        return param_dict

    def __repr__(self):
        return '<{} System from {} Molecules>'.format(self.name,
                                                      self.n_molecules)
