"""This contains a set of tests for ParaTemp.sim_setup"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2018.                    #
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

import os
import py
import pytest
import re


def test_job_info_from_qsub():
    from paratemp.sim_setup import _job_info_from_qsub
    job_info = _job_info_from_qsub('Your job 2306551 ("PT-NTD-CG") '
                                   'has been submitted')
    assert job_info == ('2306551', 'PT-NTD-CG', '2306551 ("PT-NTD-CG")')


class TestUpdateNum(object):
    @pytest.fixture
    def match_10(self):
        return re.search(r'([,=])(\d+)', '=10')

    @pytest.fixture
    def match_text(self):
        return re.search(r'([,=])(\w+)', '=text')

    @pytest.fixture
    def match_float(self):
        return re.search(r'([,=])(\d+\.\d+)', '=2.1')

    @pytest.fixture
    def match_bad_few(self):
        return re.search(r'([,=])', '=10')

    @pytest.fixture
    def match_bad_many(self):
        return re.search(r'([,=])(\d+)(\d+)', '=10')

    @pytest.fixture
    def rd1030(self):
        return {10: 30}

    def test_update_num(self, match_10, rd1030):
        from paratemp.sim_setup import _update_num
        assert '=30' == _update_num(match_10, shift=10, cat_repl_dict=rd1030)
        assert '=1' == _update_num(match_10, shift=9, cat_repl_dict=dict())
        assert '=1' == _update_num(match_10, shift=9, cat_repl_dict=rd1030)

    def test_update_num_raises(self, match_10, match_text, match_float,
                               match_bad_few, match_bad_many):
        from paratemp.sim_setup import _update_num
        with pytest.raises(KeyError):
            _update_num(match_10, shift=10, cat_repl_dict=dict())
        with pytest.raises(ValueError,
                           match='cannot be converted to a valid int'):
            _update_num(match_text, cat_repl_dict=dict())
        with pytest.raises(ValueError,
                           match='cannot be converted to a valid int'):
            _update_num(match_float, cat_repl_dict=dict())
        with pytest.raises(ValueError, match='unpack'):
            _update_num(match_bad_few, cat_repl_dict=dict())
        with pytest.raises(ValueError, match='too many.*unpack'):
            _update_num(match_bad_many, cat_repl_dict=dict())


@pytest.fixture
def n_top_dc():
    path = 'tests/test-data/ptad-cin-cg.top'
    b_path = os.path.join(os.path.dirname(path),
                          'unequal-'+os.path.basename(path))
    yield os.path.abspath(path)
    # If a backup of the original was made, copy the backup over the updated
    # version:
    if os.path.isfile(b_path):
        os.rename(b_path, path)


@pytest.fixture
def folder_dc(n_top_dc):
    return os.path.dirname(n_top_dc)


class TestGetSolvCountTop(object):

    def test_get_solv_count_top(self, n_top_dc, folder_dc):
        from paratemp.sim_setup import get_solv_count_top
        # Test giving the file name as input
        assert get_solv_count_top(n_top_dc) == 361
        # Test giving only the containing folder as input
        assert get_solv_count_top(folder=folder_dc) == 361


class TestSetSolvCountTop(object):

    def test_set_solv_count_top_n(self, n_top_dc):
        from paratemp.sim_setup import set_solv_count_top, get_solv_count_top
        set_solv_count_top(n_top_dc, s_count=100)
        assert get_solv_count_top(n_top_dc) == 100

    def test_set_solv_count_top_folder(self, folder_dc, n_top_dc):
        from paratemp.sim_setup import set_solv_count_top, get_solv_count_top
        set_solv_count_top(folder=folder_dc, s_count=50)
        assert get_solv_count_top(n_top_dc) == 50


class TestMakeGROMACSSubScript(object):

    @pytest.fixture
    def temp_path_str(self, tmpdir):
        """

        :param py.path.local tmpdir: temporary directory builtin fixture
        :return:
        """
        path = tmpdir.join('test-str.sub')
        yield str(path)
        if path.check():
            path.remove()

    @pytest.fixture
    def temp_path(self, tmpdir):
        """

        :param py.path.local tmpdir: temporary directory builtin fixture
        :return:
        """
        path = tmpdir.join('test.sub')
        yield path
        if path.check():
            path.remove()

    def test_make_sge_line(self):
        from paratemp.sim_setup import _make_sge_line
        assert _make_sge_line('l', 5) == '#$ -l 5'
        assert _make_sge_line('test', 'str') == '#$ -test str'

    def test_raises_error(self, temp_path):
        from paratemp.sim_setup import make_gromacs_sub_script
        make_gromacs_sub_script(temp_path)
        with pytest.raises(OSError):
            make_gromacs_sub_script(temp_path)
        make_gromacs_sub_script(temp_path, overwrite=True)
        with pytest.raises(ValueError):
            make_gromacs_sub_script(temp_path, cores=17, tpn=16, overwrite=True)

    def test_str_vs_localpath(self, temp_path, temp_path_str):
        from paratemp.sim_setup import make_gromacs_sub_script
        ts1 = make_gromacs_sub_script(temp_path)
        ts2 = make_gromacs_sub_script(temp_path_str)
        assert ts1.readlines()  # ensure it's not empty
        assert ts1.readlines() == ts2.readlines()

    def test_get_mdrun_line(self):
        from paratemp.sim_setup import _get_mdrun_line
        kwargs = dict(checkpoint='PT-out', deffnm='PT-out',
                      multi=True, nsims=4,
                      other_mdrun=None,
                      plumed='plumed.dat', replex=1000, tpr='TOPO/npt')
        line = _get_mdrun_line(**kwargs)
        ref_line = ('mpirun -n $NSIMS -loadbalance -x OMP_NUM_THREADS ' 
                    'mdrun_mpi -s TOPO/npt -deffnm PT-out -plumed ' 
                    'plumed.dat -multi 4 -replex 1000 -cpi PT-out ')
        assert line == ref_line
        kwargs['multi'] = 4
        kwargs['nsims'] = 'wrong'
        line = _get_mdrun_line(**kwargs)
        assert line == ref_line
        other = '-nsteps 500'
        kwargs['other_mdrun'] = other
        line = _get_mdrun_line(**kwargs)
        assert line == ref_line + other

    def test_get_sge_basic_lines(self):
        from paratemp.sim_setup import _get_sge_basic_lines
        ref_lines = ['#!/bin/bash -l\n',
                     '#$ -l h_rt=12:00:00',
                     '#$ -N job_name',
                     '#$ -o error.log',
                     '#$ -pe mpi_16_tasks_per_node 32']
        kwargs = dict(cores=32, log='error.log', name='job_name',
                      time='12:00:00', tpn=16)
        lines = _get_sge_basic_lines(**kwargs)
        assert lines == ref_lines
        kwargs['tpn'] = '16'
        kwargs['cores'] = '32'
        lines = _get_sge_basic_lines(**kwargs)
        assert lines == ref_lines
        kwargs['cores'] = '31'
        with pytest.raises(ValueError):
            _get_sge_basic_lines(**kwargs)

    def test_make_gromacs_sub_script(self, temp_path):
        from paratemp.sim_setup import make_gromacs_sub_script
        kwargs = dict(checkpoint='PT-out', deffnm='PT-out',
                      multi=True, nsims=4, other_mdrun=None,
                      plumed='plumed.dat', replex=1000, tpr='TOPO/npt',
                      cores=32, log='error.log', name='job_name',
                      time='12:00:00', tpn=16)
        ref_file = py.path.local('tests/ref-data/sub-script-ref.sub')
        test_file = make_gromacs_sub_script(temp_path, **kwargs)
        assert ref_file.readlines() == test_file.readlines()
