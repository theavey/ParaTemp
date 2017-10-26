"""This contains a set of tests for ParaTemp.geometries"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2017.                    #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017 Thomas J. Heavey IV                                   #
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


class TestXYZ(object):

    @pytest.fixture
    def xyz(self):
        from ..ParaTemp.geometries import XYZ
        return XYZ('tests/test-data/stil-3htmf.xyz')

    def test_n_atoms(self, xyz):
        assert xyz.n_atoms == 66

    def test_energy(self, xyz):
        assert xyz.energy == -1058630.8496721


class TestVector(object):

    @pytest.fixture
    def pi(self):
        from numpy import pi
        return pi

    @pytest.fixture
    def x_axis_int_list(self):
        from ..ParaTemp.geometries import Vector
        return Vector([1, 0, 0])

    @pytest.fixture
    def x_axis_float_list(self):
        from ..ParaTemp.geometries import Vector
        return Vector([1., 0., 0.])

    @pytest.fixture
    def x_axis_int(self):
        from ..ParaTemp.geometries import Vector
        return Vector(1, 0, 0)

    @pytest.fixture
    def y_axis(self):
        from ..ParaTemp.geometries import Vector
        return Vector(0, 1, 0)

    @pytest.fixture
    def z_axis(self):
        from ..ParaTemp.geometries import Vector
        return Vector(0, 0, 1)

    def test_input_int_float(self, x_axis_int_list, x_axis_float_list):
        assert (x_axis_float_list == x_axis_int_list).all()

    def test_input_list(self, x_axis_int, x_axis_int_list):
        assert (x_axis_int_list == x_axis_int).all()

    def test_rotate_x_to_y(self, x_axis_int, y_axis, pi):
        assert np.isclose(x_axis_int.rotate([0, 0, 1], pi/2), y_axis).all()

    def test_diff_angle(self, x_axis_int, y_axis, pi):
        assert x_axis_int.diff_angle(y_axis) == pi/2

    def test_cross(self, x_axis_int, y_axis, z_axis):
        assert (x_axis_int.cross(y_axis) == z_axis).all()

    def test_x(self, z_axis):
        assert z_axis.x == 0.

    def test_y(self, z_axis):
        assert z_axis.y == 0.

    def test_z(self, z_axis):
        assert z_axis.z == 1.
