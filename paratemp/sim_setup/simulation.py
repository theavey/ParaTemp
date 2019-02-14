"""This file defines a class useful for setting up simulations"""

########################################################################
#                                                                      #
# This test was written by Thomas Heavey in 2019.                      #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017-19 Thomas J. Heavey IV                                #
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

from collections import OrderedDict
import gromacs
import pathlib
import re
import sys

from ..tools import cd


__all__ = ['Simulation']


if sys.version_info >= (3, 6):
    def resolve_path(path):
        return pathlib.Path(path).resolve()
else:
    def resolve_path(path):
        try:
            return pathlib.Path(path).resolve()
        except FileNotFoundError:
            pass
        path = pathlib.Path(path)
        if path.is_absolute():
            return path
        return pathlib.Path.cwd().joinpath(path)


class Simulation(object):
    """
    A class for setting up and running GROMACS simulations
    """

    _fp = staticmethod(resolve_path)

    def __init__(self, name, gro, top, base_folder='.', mdps=None):
        self.name = name
        self.top = self._fp(top)
        self.geometries = OrderedDict(initial=self._fp(gro))
        self.base_folder = self._fp(base_folder)
        self.folders = dict(base=self.base_folder)
        self.mdps = dict() if mdps is None else mdps
        self.tprs = dict()
        self.deffnms = dict()
        self.outputs = dict()
        for mdp in self.mdps:
            setattr(self, mdp, self._make_step_method(mdp))
            self.mdps[mdp] = self._fp(mdps[mdp])

    @property
    def last_geometry(self) -> pathlib.Path:
        """
        The path to the output geometry from the most recent simulation

        :return: The path to the last output geometry
        """
        return next(reversed(self.geometries.items()))[1]

    @property
    def _next_folder_index(self) -> int:
        """
        Index for next folder to be created

        Note, this will not work if there are 99 or more folders.
        Folders should be of the form '01-minimize-benzene'
        :return: next folder index
        :rtype: int
        """
        folders = [d.name for d in self.base_folder.iterdir() if d.is_dir()]
        nums = [int(d[:2]) for d in folders if re.match(r'\d{2}-\w+-\w+', d)]
        nums.sort()
        return nums[-1]+1 if nums else 1

    def _make_step_method(self, step_name):
        """
        Make a function that runs a GROMACS "step" (minimization, equil, etc.)
        :param step_name:
        :return:
        """
        def func(geometry=None):
            geometry = self.last_geometry if geometry is None else geometry
            folder_index = self._next_folder_index
            folder = self.base_folder / '{:0>2}-{}-{}'.format(folder_index,
                                                              step_name,
                                                              self.name)
            folder.mkdir()
            self.folders[step_name] = folder
            with cd(folder):
                tpr = self._compile_tpr(step_name, geometry)
                self._run_mdrun(step_name, tpr)
            return folder
        return func

    def _compile_tpr(self, step_name, geometry=None, trajectory=None):
        geometry = self.last_geometry if geometry is None else geometry
        tpr = '{}-{}.tpr'.format(self.name, step_name)
        p_tpr = self._fp(tpr)
        self.tprs[step_name] = p_tpr
        rc, output, junk = gromacs.grompp(c=geometry,
                                          p=self.top,
                                          f=self.mdps[step_name],
                                          o=tpr,
                                          t=trajectory,
                                          stdout=False)
        # Doesn't capture output if failed?
        self.outputs['compile_{}'.format(step_name)] = output
        return p_tpr

    def _run_mdrun(self, step_name, tpr=None):
        tpr = self.tprs[step_name] if tpr is None else tpr
        deffnm = '{}-{}-out'.format(self.name, step_name)
        p_deffnm = self._fp(deffnm)
        self.deffnms[step_name] = p_deffnm
        rc, output, junk = gromacs.mdrun(s=tpr, deffnm=deffnm, stdout=False)
        # Doesn't capture output if failed?
        self.outputs['run_{}'.format(step_name)] = output
        gro = p_deffnm.with_suffix('.gro')
        self.geometries[step_name] = gro
        return gro
