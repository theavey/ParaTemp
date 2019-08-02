"""This contains code for setting up parallel tempering calcs"""

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

import pathlib

from . import para_temp_setup
from .simulation import Simulation, SimpleSimulation, GenPath
from ..tools import cd


class PTSimulation(Simulation):

    def __init__(self, *args, template_mdp: str = 'templatemdp.txt', **kwargs):
        super(PTSimulation, self).__init__(*args, **kwargs)
        self.template_mdp = template_mdp

    def production_pt(self,
                      start_temp: float, scaling_exponent: float,
                      number: int = 16,
                      temps_file: str = 'temperatures.dat',
                      geometry: GenPath = None,
                      max_warn: int = 0,
                      ) -> pathlib.Path:
        """
        Compile TPRs and run a parallel tempering calculation

        :param float start_temp: starting (lowest) temperature, in Kelvin
        :param float scaling_exponent: exponent by which to scale the temperatures.
            The temperature will be :math:`T_0 e^{j s}` where :math:`T_0` is the
            ``start_temp``, :math:`s` is the ``scaling_exponent``, and :math:`j`
            is the index of the replica between 0 and (``number``-1).
        :param int number: number of replicas/walkers
        :param str temps_file: name of file in which to store temperatures
        :param geometry: Path to the geometry to be used as input. If None,
            :attr:`last_geometry` will be used.
        :type max_warn: int or str
        :param max_warn: maximum number of warnings to ignore. str is applied to
            this argument, so type shouldn't matter significantly.
        :return: The Path to the output geometry
        """
        # TODO make this (and compile_tprs) use gromacswrapper
        step_name = 'production_pt'
        folder, geometry = self._setup_for_step(geometry, step_name)
        with cd(folder):
            with cd('tprs'):
                tpr_base = para_temp_setup.compile_tprs(
                    start_temp, scaling_exponent, self.template_mdp,
                    number=number, base_name='{}-pt'.format(self.name),
                    topology=str(self.top), structure=str(geometry),
                    index=None,  # TODO allow using an index file
                    temps_file=temps_file, maxwarn=max_warn,
                    grompp_exe='gmx_mpi grompp')
                # TODO capture stdout/stderr in a consistent manner
                output = open('gromacs_compile_output.log', 'r').read()
                self.outputs['compile_{}'.format(step_name)] = output
            self.tprs[step_name] = tpr_base
            self._run_mdrun(step_name, tpr_base)
        return folder


class SimplePTSimulation(SimpleSimulation):

    def __init__(self, *args, **kwargs):
        super(SimplePTSimulation, self).__init__(*args, **kwargs)
        self._SimClass = PTSimulation

    def make_simulation(self, solvent_model: str = 'rf',
                        mdps: dict = None, **kwargs):
        super(SimplePTSimulation, self).make_simulation(
            solvent_model, mdps,
            template_mdp=str(self._path_mdps_dir /
                             'template-mdp-{}.txt'.format(solvent_model)),
            **kwargs
        )
