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

import parmed

from . import Molecule


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
                 name: str = 'default'):
        log.debug(f'Initializing System with {len(args)} Molecules')
        self._name = name
        for arg in args:
            if not isinstance(arg, Molecule):
                raise TypeError(
                    f'positional arguments must of type Molecule; given '
                    f'{type(arg)}')
        self._directory = Path(self._name).resolve()
        self._directory.mkdir()
        ptop: parmed.gromacs.GromacsTopologyFile = args[0].topology.copy()
        for mol in args[1:]:
            ptop += mol.topology
        self._ptop = ptop
        ptop.write(str(self._directory / f'{self._name}.top'))
        ptop.save(str(self._directory / f'rough_{self._name}.gro'))

    # TODO make non-overlapping
    # TODO add GBSA atomic parameters
