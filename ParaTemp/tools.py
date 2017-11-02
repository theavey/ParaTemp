"""
Common tools for general use, largely file/path management

"""

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


import os
import shutil
from contextlib import contextmanager
import numpy as np


@contextmanager
def cd(new_dir):
    prev_dir = os.getcwd()
    os.chdir(os.path.expanduser(new_dir))
    try:
        yield
    finally:
        os.chdir(prev_dir)


def copy_no_overwrite(src, dst, silent=False):
    if os.path.exists(dst):
        if silent:
            return dst
        else:
            raise OSError(17, 'File already exists', dst)
    else:
        return shutil.copy(src, dst)


def get_temperatures(filename='TOPO/temperatures.dat'):
    """
    Get temperatures of replicas from sim. setup with para_temp_setup

    :param filename: The location of the file with the temperatures.
    :return: list of temperatures
    :rtype: numpy.ndarray
    """
    with open(filename, 'r') as t_file:
        temps = list(t_file.read()[1:-2].split(', '))
    return np.array([float(temp) for temp in temps])


def all_elements_same(in_list):
    """Check if all list elements the same.

    all_elements_same is a quick function to see if all elements of a list
    are the same. Based on http://stackoverflow.com/a/3844948/3961920
    If they're all the same, returns True, otherwise returns False."""
    return in_list.count(in_list[0]) == len(in_list)