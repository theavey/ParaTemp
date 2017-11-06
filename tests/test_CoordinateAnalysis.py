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


def test_matplotlib_testing_backend():
    # Travis should fail if this isn't true, but hopefully this makes it
    # clearer as to why it failed.
    assert matplotlib.get_backend() == 'agg'


def test_running_mean():
    from ..paratemp import coordinate_analysis as ca
    tl = [0, 2, 4]
    assert (ca.Taddol._running_mean(tl) == [1, 3]).all()


class TestXTCUniverse(object):

    @pytest.fixture
    def univ(self):
        from ..paratemp import coordinate_analysis as ca
        _univ = ca.Universe('tests/test-data/spc2.gro',
                            'tests/test-data/t-spc2-traj.xtc',
                            temp=205.)
        return _univ

    @pytest.fixture
    def univ_w_a(self, univ):
        univ.calculate_distances(a='4 5')
        return univ

    @pytest.fixture
    def ref_a_dists(self):
        import pandas
        return pandas.read_csv('tests/ref-data/spc2-a-dists.csv',
                               index_col=0)

    @pytest.fixture
    def ref_delta_g(self):
        return np.load('tests/ref-data/spc2-fes1d-delta-gs.npy')

    @pytest.fixture
    def ref_bins(self):
        return np.load('tests/ref-data/spc2-fes1d-bins.npy')

    def test_distance_str(self, univ, ref_a_dists):
        univ.calculate_distances(a='4 5')
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_distance_list_int(self, univ, ref_a_dists):
        univ.calculate_distances(a=[4, 5])
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_distance_list_str(self, univ, ref_a_dists):
        univ.calculate_distances(a=['4', '5'])
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_calculate_distance_no_recalc(self, univ_w_a, capsys):
        univ_w_a.calculate_distances(a=[4, 5])
        out, err = capsys.readouterr()
        assert out == 'Nothing (new) to calculate here.\n'

    def test_fes_1d_data_str(self, univ_w_a, ref_delta_g, ref_bins):
        """
        :type univ_w_a: ParaTemp.CoordinateAnalysis.Universe
        :type ref_delta_g: pandas.DataFrame
        :type ref_bins: pandas.DataFrame
        """
        delta_g_str, bins_str, lines_str, fig_str, ax_str = \
            univ_w_a.fes_1d('a')
        assert (delta_g_str == ref_delta_g).all()
        assert (bins_str == ref_bins).all()

    def test_fes_1d_data_data(self, univ_w_a, ref_delta_g, ref_bins):
        """
        :type univ_w_a: ParaTemp.CoordinateAnalysis.Universe
        :type ref_delta_g: pandas.DataFrame
        :type ref_bins: pandas.DataFrame
        """
        delta_g_data, bins_data, lines_data, fig_data, ax_data = \
            univ_w_a.fes_1d(univ_w_a.data['a'])
        assert (delta_g_data == ref_delta_g).all()
        assert (bins_data == ref_bins).all()

# TODO add further Universe tests
#       fes_2d
#       save_data
#       read_data
#       calculate_dihedrals
#       figure from fes_1d
#       figure from fes_2d
