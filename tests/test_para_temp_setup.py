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

import os
import py
import pytest


n_gro, n_top, n_template, n_ndx = ('spc-and-methanol.gro',
                                   'spc-and-methanol.top',
                                   'templatemdp.txt',
                                   'index.ndx')


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
        must_contain = {n_top, n_gro, n_template, n_ndx}
        assert must_contain - files_present == set()

    def test_compile_tprs(self, pt_dir_blank):
        """

        :param py.path.local pt_dir_blank:
        :return:
        """
        from paratemp.para_temp_setup import compile_tprs
        with pt_dir_blank.as_cwd():
            compile_tprs(start_temp=298, number=2,
                         topology=n_top,
                         template=n_template,
                         structure=n_gro,
                         index=n_ndx,
                         base_name='nvt',
                         gromacs_exe='gmx')
        assert pt_dir_blank.ensure_dir('TOPO')
