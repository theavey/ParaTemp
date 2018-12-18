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
    def ref_a_pbc_dists(self):
        import pandas
        return pandas.read_csv('tests/ref-data/spc2-a-pbc-dists.csv',
                               index_col=0)

    def test_distance_str(self, univ, ref_a_dists):
        univ.calculate_distances(a='4 5',
                                 read_data=False, save_data=False)
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_distance_list_int(self, univ, ref_a_dists):
        univ.calculate_distances(a=[4, 5],
                                 read_data=False, save_data=False)
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_distance_list_str(self, univ, ref_a_dists):
        univ.calculate_distances(a=['4', '5'],
                                 read_data=False, save_data=False)
        assert np.isclose(ref_a_dists, univ.data['a']).all()

    def test_calculate_distances_no_recalc(self, univ_w_a, capsys):
        univ_w_a.calculate_distances(a=[4, 5],
                                     read_data=False, save_data=False)
        out, err = capsys.readouterr()
        assert out == 'Nothing (new) to calculate here.\n'

    def test_calculate_distances_yes_recalc(self, univ_w_a):
        """
        :type univ_w_a: paratemp.coordinate_analysis.Universe
        """
        univ_w_a.calculate_distances(a='5 5', recalculate=True,
                                     read_data=False, save_data=False)
        assert (np.array([0., 0.]) == univ_w_a.data['a']).all()

    def test_distance_pbc(self, univ_pbc, ref_a_pbc_dists):
        univ_pbc.calculate_distances(a='4 5',
                                     read_data=False, save_data=False)
        assert np.isclose(ref_a_pbc_dists['a'], univ_pbc.data['a']).all()

    def test_distances_com(self, univ, ref_g_dists):
        univ.calculate_distances(
            read_data=False, save_data=False,
            g=((1, 2), (3, 4)))
        assert np.isclose(ref_g_dists, univ.data).all()

    def test_calculate_distance_raises(self, univ):
        with pytest.raises(SyntaxError):
            univ.calculate_distances(1, read_data=False, save_data=False)
        with pytest.raises(SyntaxError):
            univ.calculate_distances(a=['0', '5'],
                                     read_data=False, save_data=False)
        with pytest.raises(SyntaxError):
            univ.calculate_distances(a=['1', '2', '5'],
                                     read_data=False, save_data=False)
        with pytest.raises(NotImplementedError):
            univ.calculate_distances(a=['fail', 'here'],
                                     read_data=False, save_data=False)

    def test_calculate_distance_warns(self, univ):
        with pytest.warns(UserWarning,
                          match='following positional arguments were given'):
            univ.calculate_distances('fail', read_data=False, save_data=False)

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

    def test_read_data_no_data(self, univ, tmpdir, capsys):
        """
        :type univ: paratemp.Universe
        """
        time = 'time_' + str(int(univ._last_time / 1000)) + 'ns'
        f_name = univ.trajectory.filename.replace('xtc', 'h5')
        with tmpdir.as_cwd():
            with pytest.raises(IOError, message='This data does not exist!\n'
                                                '{}[{}]\n'.format(f_name,
                                                                  time)):
                univ.read_data()
            univ.read_data(ignore_no_data=True)
            out, err = capsys.readouterr()
        assert out == 'No data to read in {}[{}]\n'.format(f_name, time)

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

    def test_calculate_distances_read(self, univ_w_a, tmpdir, capsys):
        """
        :type univ_w_a: paratemp.Universe
        """
        with tmpdir.as_cwd():
            univ_w_a.save_data()
            capsys.readouterr()
            univ_w_a._data = univ_w_a._init_dataframe()
            univ_w_a.calculate_distances(a='4 5')
            out, err = capsys.readouterr()
        assert out == 'Nothing (new) to calculate here.\n'

    def test_select_frames(self, univ_pbc, capsys):
        u = univ_pbc
        u.calculate_distances(a='4 5',
                              read_data=False, save_data=False)
        frames = u.select_frames({'a': (0.1, 0.75)}, 'short')
        out, err = capsys.readouterr()
        assert out == 'These criteria include 1 frame\n'
        assert (u.data['short'] == [False, True]).all()
        assert (frames == [1]).all()

    def test_update_num_frames(self, univ, capsys):
        old_lt, old_nf = univ._last_time, univ._num_frames
        univ.load_new(['tests/test-data/t-spc2-traj.xtc',
                       'tests/test-data/spc2-traj-pbc.xtc'])
        univ.update_num_frames()
        out, err = capsys.readouterr()
        assert old_lt != univ._last_time
        assert old_nf != univ._num_frames
        assert out == 'Updating num of frames from {} to {}'.format(
            old_nf, univ._num_frames) + '\nand the final time.\n'


# interface to calculate_distances is not the same, currently
@pytest.mark.xfail(run=False)
class TestXTCTaddol(TestXTCUniverse):

    @pytest.fixture
    def univ(self, tmpdir):
        from paratemp import coordinate_analysis as ca
        gro = py.path.local('tests/test-data/spc2.gro')
        traj = py.path.local('tests/test-data/t-spc2-traj.xtc')
        gro.copy(tmpdir)
        traj.copy(tmpdir)
        with tmpdir.as_cwd():
            _univ = ca.Taddol(gro.basename,
                              traj.basename,
                              temp=205.)
        return _univ

    @pytest.fixture
    def univ_pbc(self, tmpdir):
        from paratemp import coordinate_analysis as ca
        gro = py.path.local('tests/test-data/spc2.gro')
        traj = py.path.local('tests/test-data/spc2-traj-pbc.xtc')
        gro.copy(tmpdir)
        traj.copy(tmpdir)
        with tmpdir.as_cwd():
            _univ = ca.Taddol(gro.basename,
                              traj.basename,
                              temp=205.)
        return _univ


# TODO add further Universe tests
#       ignore_file_change=True
#       fes_2d
#       calculate_dihedrals
#       figure from fes_1d
#       figure from fes_2d
