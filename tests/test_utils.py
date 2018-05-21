"""This contains a set of tests for paratemp.utils"""

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

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


def test_parse_ax_input():
    from paratemp.utils import _parse_ax_input as pai
    fig, ax = plt.subplots()
    f, a = pai(None)
    assert a != ax
    assert isinstance(f, Figure)
    assert isinstance(a, Axes)
    f, a = pai(ax)
    assert a == ax
    assert isinstance(a, Axes)
    assert isinstance(f, Figure)


def test_parse_bin_input():
    from paratemp.utils import _parse_bin_input as pbi
    d_ref = dict(bins='test')
    d = pbi(None)
    assert isinstance(d, dict)
    assert not d
    d = pbi('test')
    assert isinstance(d, dict)
    assert d
    assert d == d_ref


def test_parse_z_bin_input():
    from paratemp.utils import _parse_z_bin_input as pzbi
    b, v = pzbi(range(5), 'a', 'c')
    assert b == range(5)
    assert v == 4
    b, v = pzbi(None, 42, [5])
    assert (b == np.append(np.linspace(0, 5, 11), [42])).all()
    assert v == 5
    b, v = pzbi(None, 42, 5)
    assert (b == np.append(np.linspace(0, 5, 11), [42])).all()
    assert v == 5
    b, v = pzbi(None, 42, [1, 5])
    assert (b == np.append(np.linspace(1, 5, 11), [42])).all()
    assert v == 5
