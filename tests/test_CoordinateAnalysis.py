
"""This contains a set of tests for ParaTemp.CoordinateAnalysis"""

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
import numpy as np
import matplotlib
matplotlib.use('agg')

from ..ParaTemp import CoordinateAnalysis as ca


def test_matplotlib_testing_backend():
    assert matplotlib.get_backend() == 'agg'


def test_running_mean():
    tl = [0, 2, 4]
    assert (ca.Taddol._running_mean(tl) == [1, 3]).all()


class TestXTCUniverse(object):

    Univ = ca.Universe('tests/test-data/spc2.gro', 'tests/test-data/t-spc2-traj.xtc')
    delta_g_ref = np.load('tests/ref-data/spc2-fes1d-delta-gs.npy')
    bins_ref = np.load('tests/ref-data/spc2-fes1d-bins.npy')

    def test_distance(self):
        self.Univ.calculate_distances(a='4 5')
        assert self.Univ.data['a'][0] == pytest.approx(self.Univ.data['a'][1])

    def test_fes_1d_data_str(self):
        delta_g_str, bins_str, lines_str, fig_str, ax_str = \
            self.Univ.fes_1d('a')
        assert (delta_g_str == self.delta_g_ref).all()
        assert (bins_str == self.bins_ref).all()

    def test_fes_1d_data_data(self):
        delta_g_data, bins_data, lines_data, fig_data, ax_data = \
            self.Univ.fes_1d(self.Univ.data['a'])
        assert (delta_g_data == self.delta_g_ref).all()
        assert (bins_data == self.bins_ref).all()

# TODO add further Universe tests
#       fes_2d
#       save_data
#       read_data
#       calculate_dihedrals
#       figure from fes_1d
#       figure from fes_2d
