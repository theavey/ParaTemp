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

import matplotlib
import numpy as np
import pandas as pd
import py
import pytest


matplotlib.use('agg')


def test_matplotlib_testing_backend():
    # Travis should fail if this isn't true, but hopefully this makes it
    # clearer as to why it failed.
    assert matplotlib.get_backend() == 'agg'


class TestXTCUniverse(object):

    @pytest.fixture
    def univ(self, tmpdir):
        from paratemp import coordinate_analysis as ca
        gro = py.path.local('tests/test-data/spc2.gro')
        traj = py.path.local('tests/test-data/t-spc2-traj.xtc')
        gro.copy(tmpdir)
        traj.copy(tmpdir)
        with tmpdir.as_cwd():
            _univ = ca.Universe(gro.basename,
                                traj.basename,
                                temp=205.)
        return _univ

    @pytest.fixture
    def univ_w_a(self, univ):
        univ.calculate_distances(a='4 5',
                                 read_data=False, save_data=False)
        return univ

    @pytest.fixture
    def univ_pbc(self, tmpdir):
        from paratemp import coordinate_analysis as ca
        gro = py.path.local('tests/test-data/spc2.gro')
        traj = py.path.local('tests/test-data/spc2-traj-pbc.xtc')
        gro.copy(tmpdir)
        traj.copy(tmpdir)
        with tmpdir.as_cwd():
            _univ = ca.Universe(gro.basename,
                                traj.basename,
                                temp=205.)
        return _univ

    @pytest.fixture
    def ref_a_dists(self):
        import pandas
        return pandas.read_csv('tests/ref-data/spc2-a-dists.csv',
                               index_col=0)

    @pytest.fixture
    def ref_a_pbc_dists(self):
        import pandas
        return pandas.read_csv('tests/ref-data/spc2-a-pbc-dists.csv',
                               index_col=0)

    @pytest.fixture
    def ref_delta_g(self):
        return np.load('tests/ref-data/spc2-fes1d-delta-gs.npy')

    @pytest.fixture
    def ref_bins(self):
        return np.load('tests/ref-data/spc2-fes1d-bins.npy')

    @pytest.fixture
    def ref_delta_g_20(self):
        """Created using calc_fes_1d with temp=205. and bins=20.
        Saved with np.save('spc2-fes1d-delta-gs-20.npy', dg20,
        allow_pickle=False)."""
        return np.load('tests/ref-data/spc2-fes1d-delta-gs-20.npy')

    @pytest.fixture
    def ref_bins_20(self):
        return np.load('tests/ref-data/spc2-fes1d-bins-20.npy')

    def test_distance_str(self, univ, ref_a_dists):
        univ.calculate_distances(a='4 5')
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_distance_list_int(self, univ, ref_a_dists):
        univ.calculate_distances(a=[4, 5])
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_distance_list_str(self, univ, ref_a_dists):
        univ.calculate_distances(a=['4', '5'])
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_calculate_distances_no_recalc(self, univ_w_a, capsys):
        univ_w_a.calculate_distances(a=[4, 5])
        out, err = capsys.readouterr()
        assert out == 'Nothing (new) to calculate here.\n'

    def test_calculate_distances_yes_recalc(self, univ_w_a):
        """
        :type univ_w_a: paratemp.coordinate_analysis.Universe
        """
        univ_w_a.calculate_distances(a='5 5', recalculate=True)
        assert (np.array([0., 0.]) == univ_w_a.data['a']).all()

    def test_distance_pbc(self, univ_pbc, ref_a_pbc_dists):
        univ_pbc.calculate_distances(a='4 5')
        assert np.isclose(ref_a_pbc_dists['a'], univ_pbc.data['a']).all()

    def test_calc_fes_1d(self, univ_w_a, ref_delta_g, ref_bins, ref_delta_g_20,
                         ref_bins_20):
        """

        While not technically a test of Universe, the useful fixtures are
        all already defined here.

        :type univ_w_a: paratemp.coordinate_analysis.Universe
        :type ref_delta_g: np.ndarray
        :type ref_bins: np.ndarray
        :type ref_delta_g_20: np.ndarray
        :type ref_bins_20: np.ndarray
        """
        from paratemp.utils import calc_fes_1d
        delta_g_data, bins_data = calc_fes_1d(univ_w_a.data['a'], temp=205.,
                                              bins=None)
        assert np.allclose(delta_g_data, ref_delta_g)
        assert np.allclose(bins_data, ref_bins)
        delta_g_20, bins_20 = calc_fes_1d(univ_w_a.data['a'], temp=205.,
                                          bins=20)
        assert np.allclose(delta_g_20, ref_delta_g_20)
        assert np.allclose(bins_20, ref_bins_20)

    def test_fes_1d_data_str(self, univ_w_a, ref_delta_g, ref_bins):
        """
        :type univ_w_a: paratemp.coordinate_analysis.Universe
        :type ref_delta_g: np.ndarray
        :type ref_bins: np.ndarray
        """
        delta_g_str, bins_str, lines_str, fig_str, ax_str = \
            univ_w_a.fes_1d('a')
        assert np.allclose(delta_g_str, ref_delta_g)
        assert np.allclose(bins_str, ref_bins)

    def test_fes_1d_data_data(self, univ_w_a, ref_delta_g, ref_bins):
        """
        :type univ_w_a: paratemp.coordinate_analysis.Universe
        :type ref_delta_g: np.ndarray
        :type ref_bins: np.ndarray
        """
        delta_g_data, bins_data, lines_data, fig_data, ax_data = \
            univ_w_a.fes_1d(univ_w_a.data['a'])
        assert np.allclose(delta_g_data, ref_delta_g)
        assert np.allclose(bins_data, ref_bins)

    def test_final_time_str(self, univ):
        assert univ.final_time_str == '2ps'
        univ._last_time = 1001.0
        assert univ.final_time_str == '1ns'
        univ._last_time = 32111222.12
        assert univ.final_time_str == '32us'
        univ._last_time = 5.1e12
        assert univ.final_time_str == '5100ms'

    def test_save_data(self, univ_w_a, tmpdir, capsys):
        time = 'time_' + str(int(univ_w_a._last_time / 1000)) + 'ns'
        f_name = univ_w_a.trajectory.filename.replace('xtc', 'h5')
        with tmpdir.as_cwd():
            univ_w_a.save_data()
            out, err = capsys.readouterr()
            assert tmpdir.join(f_name).exists()
            with pd.HDFStore(f_name) as store:
                df = store[time]
        assert out == 'Saved data to {f_name}[{time}]\n'.format(
            f_name=f_name, time=time)
        assert np.allclose(df, univ_w_a.data)

    def test_save_data_no_new(self, univ_w_a, tmpdir, capsys):
        time = 'time_' + str(int(univ_w_a._last_time / 1000)) + 'ns'
        f_name = univ_w_a.trajectory.filename.replace('xtc', 'h5')
        with tmpdir.as_cwd():
            univ_w_a.save_data()
            capsys.readouterr()
            univ_w_a.save_data()
            out, err = capsys.readouterr()
            assert tmpdir.join(f_name).exists()
            with pd.HDFStore(f_name) as store:
                df = store[time]
        assert out == 'No data added to {f_name}[{time}]\n'.format(
            f_name=f_name, time=time)
        assert np.allclose(df, univ_w_a.data)

    def test_save_data_add_new(self, univ, univ_w_a, tmpdir, capsys):
        time = 'time_' + str(int(univ_w_a._last_time / 1000)) + 'ns'
        f_name = univ_w_a.trajectory.filename.replace('xtc', 'h5')
        with tmpdir.as_cwd():
            univ_w_a.save_data()
            capsys.readouterr()
            univ.calculate_distances(b='4 5', save_data=False)
            univ.save_data()
            out, err = capsys.readouterr()
        assert out == 'Saved data to {f_name}[{time}]\n'.format(
            f_name=f_name, time=time)

    def test_read_data(self, univ, univ_w_a, tmpdir, capsys):
        """
        :type univ_w_a: paratemp.Universe
        :type univ: paratemp.Universe
        """
        with tmpdir.as_cwd():
            univ_w_a.save_data()
            capsys.readouterr()  # just so it doesn't print
            univ.read_data()
        assert (univ_w_a.data == univ.data).all().all()

    def test_calculate_distances_save(self, univ, tmpdir, capsys):
        """
        :type univ: paratemp.Universe
        """
        time = 'time_' + str(int(univ._last_time / 1000)) + 'ns'
        f_name = univ.trajectory.filename.replace('xtc', 'h5')
        with tmpdir.as_cwd():
            univ.calculate_distances(a='4 5')
            out, err = capsys.readouterr()
            assert tmpdir.join(f_name).exists()
            with pd.HDFStore(f_name) as store:
                df = store[time]
        assert out == 'Saved data to {f_name}[{time}]\n'.format(
            f_name=f_name, time=time)
        assert np.allclose(df, univ.data)

# TODO add further Universe tests
#       ignore_file_change=True
#       fes_2d
#       save_data
#       read_data
#       calculate_dihedrals
#       figure from fes_1d
#       figure from fes_2d
