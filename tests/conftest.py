"""This contains a set of fixtures and such for tests"""

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

import pathlib
import shutil

import gromacs
import numpy as np
import pkg_resources
import pytest

gromacs.config.setup()


@pytest.fixture
def path_ref_data():
    return pathlib.Path(pkg_resources.resource_filename(__name__, 'ref-data'))


@pytest.fixture
def path_test_data():
    return pathlib.Path(pkg_resources.resource_filename(__name__, 'test-data'))


@pytest.fixture
def ref_a_dists(path_ref_data):
    import pandas
    return pandas.read_csv(path_ref_data / 'spc2-a-dists.csv',
                           names=['a'], index_col=0)


@pytest.fixture
def ref_g_dists(path_ref_data):
    import numpy
    return numpy.load(path_ref_data / 'spc2-g-dists.npy')


@pytest.fixture
def ref_delta_g(path_ref_data):
    return np.load(path_ref_data / 'spc2-fes1d-delta-gs.npy')


@pytest.fixture
def ref_bins(path_ref_data):
    return np.load(path_ref_data / 'spc2-fes1d-bins.npy')


@pytest.fixture
def ref_delta_g_20(path_ref_data):
    """Created using calc_fes_1d with temp=205. and bins=20.
    Saved with np.save('spc2-fes1d-delta-gs-20.npy', dg20,
    allow_pickle=False)."""
    return np.load(path_ref_data / 'spc2-fes1d-delta-gs-20.npy')


@pytest.fixture
def ref_bins_20(path_ref_data):
    return np.load(path_ref_data / 'spc2-fes1d-bins-20.npy')


@pytest.fixture
def pt_blank_dir(tmp_path: pathlib.PosixPath, path_test_data):
    dir_from = path_test_data / 'spc-and-methanol'
    tmp_path = tmp_path.joinpath('spc-and-methanol')
    # str needed for Python 3.5
    shutil.copytree(str(dir_from), str(tmp_path))
    return tmp_path


@pytest.fixture
def pt_run_dir(tmp_path: pathlib.PosixPath, path_test_data):
    dir_from = path_test_data / 'spc-and-methanol-run'
    tmp_path = tmp_path.joinpath('spc-and-methanol-run')
    # str needed for Python 3.5
    shutil.copytree(str(dir_from), str(tmp_path))
    return tmp_path
