"""This contains a set of tests for paratemp.para_temp_setup"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2018.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2018 Thomas J. Heavey IV                                   #
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

import distutils.spawn
import errno
import os
import py
import pytest
import shutil


n_gro, n_top, n_template, n_ndx = ('spc-and-methanol.gro',
                                   'spc-and-methanol.top',
                                   'templatemdp.txt',
                                   'index.ndx')
n_gro_o1, n_gro_o2 = 'PT-out0.gro', 'PT-out1.gro'


@pytest.fixture
def grompp():
    if distutils.spawn.find_executable('gmx'):
        return 'gmx grompp'
    if distutils.spawn.find_executable('gmx_mpi'):
        return 'gmx_mpi grompp'
    elif distutils.spawn.find_executable('grompp'):
        return 'grompp'
    else:
        raise OSError(errno.ENOENT, 'No GROMACS executable found')


class TestCompileTPRs(object):

    @pytest.fixture
    def pt_dir_blank(self, tmpdir):
        dir_from = py.path.local('tests/test-data/spc-and-methanol')
        files_from = dir_from.listdir()
        for f in files_from:
            f.copy(tmpdir)
        return tmpdir

    def test_pt_dir_blank(self, pt_dir_blank):
        files_present = {f.basename for f in pt_dir_blank.listdir()}
        must_contain = {n_top, n_gro, n_template, n_ndx, n_gro_o1, n_gro_o2}
        assert must_contain - files_present == set()

    def test_basic(self, pt_dir_blank, grompp):
        """

        :param py.path.local pt_dir_blank:
        :return:
        """
        from paratemp.sim_setup import compile_tprs
        from paratemp.tools import get_temperatures
        dir_topo = pt_dir_blank.mkdir('TOPO')
        number = 2
        with dir_topo.as_cwd():
            compile_tprs(start_temp=298, scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         structure='../'+n_gro,
                         base_name='nvt',
                         grompp_exe=grompp)
        assert dir_topo.check()
        for i in range(number):
            assert dir_topo.join('nvt{}.tpr'.format(i)).check()
        assert get_temperatures(
            str(dir_topo.join('temperatures.dat'))).shape == (2,)

    def test_multi_structure(self, pt_dir_blank, grompp):
        from paratemp.sim_setup import compile_tprs
        from paratemp.tools import get_temperatures
        dir_topo = pt_dir_blank.mkdir('TOPO')
        number = 2
        with dir_topo.as_cwd():
            compile_tprs(start_temp=298,  scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         multi_structure=True,
                         structure='../PT-out',
                         base_name='nvt',
                         grompp_exe=grompp)
        assert dir_topo.check()
        for i in range(number):
            assert dir_topo.join('nvt{}.tpr'.format(i)).check()
        assert get_temperatures(
            str(dir_topo.join('temperatures.dat'))).shape == (2,)

    def test_raises_os_error(self, pt_dir_blank, grompp):
        from paratemp.sim_setup import compile_tprs
        dir_topo = pt_dir_blank.mkdir('TOPO')
        number = 2
        with dir_topo.as_cwd(), pytest.raises(
                OSError, match='Incorrect number of structure files found'):
            compile_tprs(start_temp=298,  scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         multi_structure=True,
                         structure='../',
                         base_name='nvt',
                         grompp_exe=grompp)
        with dir_topo.as_cwd(), pytest.raises(
                OSError, match='No structure file found'):
            compile_tprs(start_temp=298,  scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         structure='../not-here.gro',
                         base_name='nvt',
                         grompp_exe=grompp)
        with dir_topo.as_cwd(), pytest.raises(
                OSError, match='No topology file found'):
            compile_tprs(start_temp=298,  scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         structure='../'+n_gro,
                         topology='../not-here.top',
                         base_name='nvt',
                         grompp_exe=grompp)

    def test_raises_runtime_error(self, pt_dir_blank, grompp):
        from paratemp.sim_setup import compile_tprs
        dir_topo = pt_dir_blank.mkdir('TOPO')
        number = 2
        with dir_topo.as_cwd(), pytest.raises(RuntimeError):
            compile_tprs(start_temp=298,  scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         structure='../*top',
                         base_name='nvt',
                         grompp_exe=grompp)

    def test_warns(self, pt_dir_blank, grompp):
        from paratemp.sim_setup import compile_tprs
        from paratemp.tools import get_temperatures
        dir_topo = pt_dir_blank.mkdir('TOPO')
        number = 2
        with dir_topo.as_cwd(), pytest.warns(
                UserWarning, match=r'Found \d+ structure files'):
            compile_tprs(start_temp=298,  scaling_exponent=0.025,
                         number=number,
                         template='../'+n_template,
                         structure='../*.gro',
                         base_name='nvt',
                         grompp_exe=grompp)
        assert dir_topo.check()
        for i in range(number):
            assert dir_topo.join('nvt{}.tpr'.format(i)).check()
        assert get_temperatures(
            str(dir_topo.join('temperatures.dat'))).shape == (2,)


class TestAddCptToSubScript(object):

    @pytest.fixture
    def sub_script_path(self):
        path = 'tests/test-data/gromacs-start-job.sub'
        b_path = os.path.join(os.path.dirname(path),
                              'temp-submission-script.bak')
        b2_path = os.path.join(os.path.dirname(path),
                               'backup.bak')
        shutil.copy(path, b2_path)
        yield os.path.abspath(path)
        # If a backup of the original was made, copy the backup over the updated
        # version:
        if os.path.isfile(b_path):
            os.rename(b_path, path)
        else:
            os.rename(b2_path, path)

    @pytest.fixture
    def sub_script_path_cpt(self):
        path = 'tests/test-data/gromacs-start-job-cpt.sub'
        b_path = os.path.join(os.path.dirname(path),
                              'temp-submission-script.bak')
        b2_path = os.path.join(os.path.dirname(path),
                               'backup.bak')
        shutil.copy(path, b2_path)
        yield os.path.abspath(path)
        # If a backup of the original was made, copy the backup over the updated
        # version:
        if os.path.isfile(b_path):
            os.rename(b_path, path)
        else:
            os.rename(b2_path, path)

    def test_adding_to_script(self, sub_script_path):
        orig_lines = open(sub_script_path, 'r').readlines()
        from paratemp.sim_setup.para_temp_setup import \
            _add_cpt_to_sub_script as acpt
        acpt(sub_script_path, 'checkpoint_test')
        new_lines = open(sub_script_path, 'r').readlines()
        for line in new_lines:
            if 'mpirun' in line:
                md_line = line
                break
        else:
            raise ValueError('Could not find "mpirun" line')
        assert 'checkpoint_test' in md_line
        assert len(set(new_lines) - set(orig_lines)) == 1

    def test_no_change(self, sub_script_path_cpt):
        orig_lines = open(sub_script_path_cpt, 'r').readlines()
        from paratemp.sim_setup.para_temp_setup import \
            _add_cpt_to_sub_script as acpt
        acpt(sub_script_path_cpt, 'checkpoint_test')
        new_lines = open(sub_script_path_cpt, 'r').readlines()
        for line in new_lines:
            if 'mpirun' in line:
                md_line = line
                break
        else:
            raise ValueError('Could not find "mpirun" line')
        assert 'checkpoint_test' not in md_line
        assert not len(set(new_lines) - set(orig_lines))

    def test_comment_line(self, sub_script_path):
        orig_lines = open(sub_script_path, 'r').readlines()
        from paratemp.sim_setup.para_temp_setup import \
            _add_cpt_to_sub_script as acpt
        acpt(sub_script_path, 'checkpoint_test')
        new_lines = open(sub_script_path, 'r').readlines()
        for line in new_lines:
            if 'comment' in line:
                new_comm_line = line
                break
        else:
            raise ValueError('Could not find "comment" line')
        for line in orig_lines:
            if 'comment' in line:
                orig_comm_line = line
                break
        else:
            raise ValueError('Could not find "comment" line')
        assert orig_comm_line == new_comm_line

    def test_raises_value_error(self, tmpdir):
        from paratemp.sim_setup.para_temp_setup import \
            _add_cpt_to_sub_script as acpt
        test_sub = tmpdir.join('test.sub').ensure()
        with pytest.raises(ValueError, match='Could not find GROMACS mdrun'):
            acpt(str(test_sub), 'checkpoint_test')


class TestFindCPTBase(object):

    @pytest.fixture
    def test_dir(self):
        lp = py.path.local('tests/test-data/spc-and-methanol-run')
        if lp.check():
            return lp
        else:
            raise OSError(errno.ENOENT, 'run spc-and-methanol dir not found')

    def test_works(self, test_dir):
        from paratemp.sim_setup.para_temp_setup import _find_cpt_base
        cpt_base = _find_cpt_base(str(test_dir)+'/')
        assert cpt_base == str(test_dir.join('PT-out'))

    def test_raises_value_error(self, tmpdir):
        from paratemp.sim_setup.para_temp_setup import _find_cpt_base
        with pytest.raises(ValueError):
            _find_cpt_base(str(tmpdir))
