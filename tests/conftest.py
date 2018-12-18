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

import numpy as np
import pytest


@pytest.fixture
def ref_a_dists():
    import pandas
    return pandas.read_csv('tests/ref-data/spc2-a-dists.csv',
                           names=['a'], index_col=0)


@pytest.fixture
def ref_g_dists():
    import numpy
    return numpy.load('tests/ref-data/spc2-g-dists.npy')


@pytest.fixture
def ref_delta_g():
    return np.load('tests/ref-data/spc2-fes1d-delta-gs.npy')


@pytest.fixture
def ref_bins():
    return np.load('tests/ref-data/spc2-fes1d-bins.npy')


@pytest.fixture
def ref_delta_g_20():
    """Created using calc_fes_1d with temp=205. and bins=20.
    Saved with np.save('spc2-fes1d-delta-gs-20.npy', dg20,
    allow_pickle=False)."""
    return np.load('tests/ref-data/spc2-fes1d-delta-gs-20.npy')


@pytest.fixture
def ref_bins_20():
    return np.load('tests/ref-data/spc2-fes1d-bins-20.npy')
