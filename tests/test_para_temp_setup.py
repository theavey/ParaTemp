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


@pytest.fixture
def pt_dir(tmpdir):
    dir_from = py.path.local('tests/test-data/spc-and-methanol')
    files_from = dir_from.listdir()
    for f in files_from:
        f.copy(tmpdir)
    return tmpdir


def test_pt_dir(pt_dir):
    files_present = {f.basename for f in pt_dir.listdir()}
    must_contain = {'spc-and-methanol.gro',
                    'spc-and-methanol.top',
                    'templatemdp.txt'}
    assert must_contain - files_present == set()

class TestCompileTPRs(object):

    def
