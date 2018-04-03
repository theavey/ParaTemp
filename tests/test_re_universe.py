"""This contains a set of tests for paratemp.re_universe"""

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

import py
import pytest


class TestREUniverse(object):

    @pytest.fixture
    def reu(self):
        from paratemp.re_universe import REUniverse
        dir = py.path.local('tests/test-data/spc-and-methanol-run')
        with dir.as_cwd():
            reu = REUniverse('TOPO/nvt0.tpr',
                             base_folder='.', traj_glob='PT-out*.trr')
        return reu

    def test_reu_len(self, reu):
        assert len(reu) == 2
