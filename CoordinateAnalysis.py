#! /usr/bin/env python

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
from .exceptions import InputError


def get_dist(a, b):
    """Calculate the distance between AtomGroups a and b"""
    from numpy.linalg import norm
    return norm(a.centroid() - b.centroid())


def get_angle(a, b, c, units='rad'):
    """Calculate the angle between ba and bc for AtomGroups a, b, c"""
    from numpy import arccos, rad2deg, dot
    from numpy.linalg import norm
    b_center = b.centroid()
    ba = a.centroid() - b_center
    bc = c.centroid() - b_center
    angle = arccos(dot(ba, bc)/(norm(ba)*norm(bc)))
    if units == 'rad':
        return angle
    elif units == 'deg':
        return rad2deg(angle)
    else:
        raise InputError(units,
                         'Unrecognized units. '
                         'The two recognized units are rad and deg.')


def get_dihedral(a, b, c, d, units='rad'):
    """Calculate the angle between abc and bcd for AtomGroups a,b,c,d

    Based on formula given in
    https://en.wikipedia.org/wiki/Dihedral_angle"""
    from numpy import cross, arctan2, dot, rad2deg
    from numpy.linalg import norm
    ba = a.centroid() - b.centroid()
    bc = b.centroid() - c.centroid()
    dc = d.centroid() - c.centroid()
    angle = arctan2(dot(cross(cross(ba, bc), cross(bc, dc)), bc) /
                    norm(bc),
                    dot(cross(ba, bc), cross(bc, dc)))
    if units == 'rad':
        return angle
    elif units == 'deg':
        return rad2deg(angle)
    else:
        raise InputError(units,
                         'Unrecognized units. '
                         'The two recognized units are rad and deg.')
