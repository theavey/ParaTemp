"""This contains a set of tests for paratemp.coordinate_analysis"""

########################################################################
#                                                                      #
# This test was written by Thomas Heavey in 2018.                      #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017-18 Thomas J. Heavey IV                                #
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

from paratemp.tools import cd
import re


def test_find_energies(pt_run_dir):
    # Doesn't currently test:
    #    content of the outputs
    #    what happens if they already exist
    from paratemp.energy_histo import find_energies
    with cd(pt_run_dir):
        l_xvgs = find_energies()
    for xvg in l_xvgs:
        assert pt_run_dir.join(xvg).exists()
        assert re.match(r'energy[01].xvg', xvg)
    assert len(l_xvgs) == 2


def test_make_indices(pt_run_dir):
    # Doesn't currently test:
    #    content of the outputs
    #    what happens if they already exist
    from paratemp.energy_histo import make_indices
    with cd(pt_run_dir):
        make_indices('PT-out0.log')
    assert pt_run_dir.join('replica_temp.xvg').exists()
    assert pt_run_dir.join('replica_index.xvg').exists()
    assert pt_run_dir.join('demux.pl.log').exists()
