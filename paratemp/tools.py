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
    exists = False
    if os.path.isdir(src):
        raise OSError(21, 'Is a directory: {}'.format(src))
    elif os.path.isdir(dst):
        if os.path.isfile(os.path.join(dst, os.path.basename(src))):
            exists = True
    elif os.path.isfile(dst):
        exists = True
    if exists:
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


class _BlankStream(object):
    """
    A class for use when not actually wanting to write to a file.
    """
    def write(self, string):
        pass

    def fileno(self):
        return 0  # Not sure if this works. Maybe None would be better

    def flush(self):
        pass


def _replace_string_in_file(old_str, new_str, file_name,
                            log_stream=_BlankStream()):
    """
    Replace a specified string possibly in each line of a file.

    The file will be copied with the extension '.bak' before edited, and this
    copy operation will not overwrite an existing file.

    This is intended for use in replaced tpr names in a submission script, but
    it is not only specific to that use.
    :param str old_str: String to be replaced.
    :param str new_str: String to be inserted.
    :param str file_name: Name of the file to be edited.
    :param log_stream: Default: _BlankStream(). The file stream to which to log
    information. The default will just not log anything.
    :type log_stream: _BlankStream or BinaryIO
    :return: None
    """
    log_stream.write('Editing '
                     '{} for new string "{}"\n'.format(file_name,
                                                       new_str))
    log_stream.write('Copying file as backup to '
                     '{}\n'.format(file_name + '.bak'))
    log_stream.flush()
    copy_no_overwrite(file_name, file_name + '.bak')
    with open(file_name + '.bak', 'r') as old_f, open(file_name, 'w') as new_f:
        for line in old_f:
            line = line.replace(old_str, new_str)
            new_f.write(line)