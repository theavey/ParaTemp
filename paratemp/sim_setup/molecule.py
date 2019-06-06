"""This contains code for setting up a molecule for MD calcs"""

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

from io import TextIOBase
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Union, Dict, Any

import parmed
import pkg_resources

from ..tools import cd


__all__ = ['Molecule', 'make_mol_inputs']


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


def make_mol_inputs() -> Dict[str, Any]:
    geometry = None
    while geometry is None:
        geometry = Path(input('Path to geometry input (as a PDB, '
                              'MDL, or mol2): '))
        if not geometry.exists():
            print('That file does not exist. Try again...')
            geometry = None
    charge = None
    while charge is None:
        try:
            charge = int(input('What is the charge on this molecule? '))
        except ValueError:
            print('That was not a valid integer. Try again...')
    name = None
    while name is None:
        name = input('What is the name of this molecule? (try to avoid '
                     'special characters)')
    resname = None
    while resname is None:
        resname = input('What should be the residue name (normally a unique '
                        'three character string)?')
        resname = resname.strip()
        if 0 == len(resname) or len(resname) > 4:
            print('Try a shorter or longer string...')
            resname = None
    return dict(geometry=geometry, charge=charge,
                name=name, resname=resname)


class Molecule(object):
    """
    Molecule class will make a GAFF-parameterized Structure from an input
    """

    def __init__(self, geometry: Union[str, Path],
                 charge: int = 0,
                 name: str = None,
                 resname: str = 'MOL',):
        log.debug('Initializing Molecule with {}'.format(geometry))
        self._input_geo_path = Path(geometry)
        self._name = self._input_geo_path.stem if name is None else name
        self.resname = resname
        self._directory = Path(self._name).resolve()
        self._directory.mkdir(exist_ok=True)
        shutil.copy(self._input_geo_path, self._directory)
        self.charge = int(charge)
        self._parameterized = False
        self._gro = None
        self._top = None
        self._ptop = None

    @classmethod
    def from_make_mol_inputs(cls, mol_inputs):
        return cls(**mol_inputs)

    @classmethod
    def assisted(cls):
        return cls(**make_mol_inputs())

    def parameterize(self):
        # could take keywords for FF
        # could use charges from QM calc
        # TODO convert from whatever to PDB, MDL, or MOL2
        log.debug('Parameterizing {} with acpype'.format(self._name))
        env_to_load = self._get_amber_env()
        cl = shlex.split('acpype.py -i {} '
                         '-o gmx '
                         '-n {} '
                         '-b {} '.format(
                            self._input_geo_path.resolve(),
                            self.charge,
                            self._name))
        log.warning('Running acpype.py; this may take a few minutes')
        proc = self._run_in_dir(cl, env=env_to_load)
        log.info('acpype said:\n {}'.format(proc.stdout))
        proc.check_returncode()
        ac_dir = self._directory / '{}.acpype'.format(self._name)
        gro = ac_dir / '{}_GMX.gro'.format(self._name)
        top = ac_dir / '{}_GMX.top'.format(self._name)
        if not gro.is_file() or not top.is_file():
            mes = 'gro or top file not created in {}'.format(ac_dir)
            log.error(mes)
            raise FileNotFoundError(mes)
        self._gro = gro
        self._top = top
        ptop = parmed.gromacs.GromacsTopologyFile(str(top),
                                                  xyz=str(gro))
        self._ptop = ptop
        for res in ptop.residues:
            res.name = self.resname
        ptop.write(str(self._directory / '{}.top'.format(self._name)))
        ptop.save(str(self._directory / '{}.gro'.format(self._name)))
        log.info('Wrote top and gro files in {}'.format(self._directory))
        self._parameterized = True

    @staticmethod
    def _get_amber_env() -> Dict[str, str]:
        log.info('Using special environment variables for Amber executables')
        amber_env_stream = pkg_resources.resource_stream(
            __name__, 'SimpleSim_data/amber_env.json')  # type: TextIOBase
        amber_env = json.load(amber_env_stream)
        curr_env = dict(os.environ)
        curr_env.update(amber_env)
        conda_prefix = Path(curr_env['CONDA_PREFIX'])
        conda_bin = conda_prefix / 'bin'
        curr_env['PATH'] += os.pathsep + str(conda_bin)
        return curr_env

    @property
    def topology(self) -> parmed.Structure:
        return self._ptop

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def name(self) -> str:
        return self._name

    def _run_in_dir(self, cl, **kwargs) -> subprocess.CompletedProcess:
        with cd(self._directory):
            proc = subprocess.run(cl, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True,
                                  **kwargs)
        return proc

    def __repr__(self):
        return '<{} Molecule; parameterized: {}>'.format(self.name,
                                                          self._parameterized)
