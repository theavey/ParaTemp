"""This contains a set of tests for ParaTemp.geometries"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2017.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017 Thomas J. Heavey IV                                   #
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

import pytest


class TestXYZ(object):

    @pytest.fixture
    def xyz(self):
        from ..ParaTemp.geometries import XYZ
        return XYZ('tests/test-data/stil-3htmf.xyz')

    def test_n_atoms(self, xyz):
        assert xyz.n_atoms == 66

    def test_energy(self, xyz):
        assert xyz.energy == -1058630.8496721
