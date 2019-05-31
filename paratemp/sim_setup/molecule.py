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

import logging
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Union

import parmed

from ..tools import cd


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


class Molecule(object):

    def __init__(self, geometry: Union[str, Path],
                 charge: int = 0,
                 name: str = None,
                 resname: str = 'MOL',):
        log.debug(f'Initializing Molecule with {geometry}')
        self._input_geo_path = Path(geometry)
        self._name = self._input_geo_path.stem if name is None else name
        self.resname = resname
        self._directory = Path(self._name).resolve()
        self._directory.mkdir(exist_ok=True)
        shutil.copy(self._input_geo_path, self._directory)
        self.charge = charge
        self._gro = None
        self._top = None
        self._ptop = None

    def parameterize(self):
        # could take keywords for FF
        # could use charges from QM calc
        # TODO convert from whatever to PDB, MDL, or MOL2
        log.debug(f'Parameterizing {self._name} with acpype')
        cl = shlex.split(f'acpype.py -i {self._input_geo_path.resolve()} '
                         f'-o gmx '
                         f'-n {self.charge} '
                         f'-b {self._name} ')
        log.warning('Running acpype.py; this may take a few minutes')
        proc = self._run_in_dir(cl)
        log.info(f'acpype said:\n {proc.stdout}')
        proc.check_returncode()
        ac_dir = self._directory / f'{self._name}.acpype'
        gro = ac_dir / f'{self._name}_GMX.gro'
        top = ac_dir / f'{self._name}_GMX.top'
        if not gro.is_file() or not top.is_file():
            mes = f'gro or top file not created in {ac_dir}'
            log.error(mes)
            raise FileNotFoundError(mes)
        self._gro = gro
        self._top = top
        ptop = parmed.gromacs.GromacsTopologyFile(str(top),
                                                  xyz=str(gro))
        self._ptop = ptop
        for res in ptop.residues:
            res.name = self.resname
        ptop.write(str(self._directory / f'{self._name}.top'))
        ptop.save(str(self._directory / f'{self._name}.gro'))
        log.info(f'Wrote top and gro files in {self._directory}')

    def _run_in_dir(self, cl) -> subprocess.CompletedProcess:
        with cd(self._directory):
            proc = subprocess.run(cl, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True)
        return proc
