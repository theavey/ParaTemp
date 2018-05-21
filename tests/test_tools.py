"""This contains a set of tests for ParaTemp.tools"""

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

import pytest
import numpy as np


@pytest.fixture
def ref_temps():
    return np.load('tests/ref-data/temperatures.npy')


def test_get_temps(ref_temps):
    from paratemp import get_temperatures
    assert (get_temperatures('tests/test-data/temperatures.dat')
            == ref_temps).all()
    assert (get_temperatures('tests/test-data/temperatures-new.dat')
            == ref_temps).all()


def test_find_nearest_idx(ref_temps):
    from paratemp.tools import find_nearest_idx
    assert find_nearest_idx(ref_temps, 0) == 0
    assert find_nearest_idx(ref_temps, 500.) == 15
    assert find_nearest_idx(ref_temps, 221.) == 1


def test_running_mean():
    from paratemp.tools import running_mean
    tl = [0, 2, 4]
    assert (running_mean(tl) == [1, 3]).all()