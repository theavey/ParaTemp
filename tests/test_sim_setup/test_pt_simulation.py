"""This contains a set of tests for setting up parallel tempering calcs"""

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

import pytest

from .test_simulation import TestSimulation


class TestPTSimulation(TestSimulation):

    @pytest.fixture
    def sim_with_dir(self, pt_blank_dir):
        from paratemp.sim_setup import PTSimulation
        gro = pt_blank_dir / 'PT-out0.gro'
        top = pt_blank_dir / 'spc-and-methanol.top'
        sim = PTSimulation(name='sim_fixture',
                           gro=str(gro), top=str(top),
                           base_folder=str(pt_blank_dir),
                           mdps=self.mdps)
        return sim, pt_blank_dir
