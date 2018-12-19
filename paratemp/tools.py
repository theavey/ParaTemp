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


from contextlib import contextmanager
import errno
import numpy as np
import os
import shutil
import sys


__all__ = ['cd', 'copy_no_overwrite', 'get_temperatures', 'all_elements_same',
           'find_nearest_idx', 'running_mean']

if sys.version_info >= (3, 6):
    @contextmanager
    def cd(new_dir):
        prev_dir = os.getcwd()
        os.chdir(os.path.expanduser(new_dir))
        try:
            yield
        finally:
            os.chdir(prev_dir)
else:
    @contextmanager
    def cd(new_dir):
        new_dir = str(new_dir)
        prev_dir = os.getcwd()
        os.chdir(os.path.expanduser(new_dir))
        try:
            yield
        finally:
            os.chdir(prev_dir)


def copy_no_overwrite(src, dst, silent=False):
    exists = False
    if os.path.isdir(src):
        raise OSError(errno.EISDIR, 'Is a directory: {}'.format(src))
    elif os.path.isdir(dst):
        if os.path.isfile(os.path.join(dst, os.path.basename(src))):
            exists = True
    elif os.path.isfile(dst):
        exists = True
    if exists:
        if silent:
            return dst
        else:
            raise OSError(errno.EEXIST, 'File already exists', dst)
    else:
        return shutil.copy(src, dst)


def get_temperatures(filename='TOPO/temperatures.dat'):
    """
    Get temperatures of replicas from sim. setup with para_temp_setup

    :param str filename: The location of the file with the temperatures.
    :return: list of temperatures
    :rtype: numpy.ndarray
    """
    try:
        return np.loadtxt(filename)
    except ValueError:
        pass
    with open(filename, 'r') as t_file:
        temps = list(t_file.read()[1:-2].split(', '))
    return np.array([float(temp) for temp in temps])


def all_elements_same(in_list):
    """Check if all list elements the same.

    all_elements_same is a quick function to see if all elements of a list
    are the same. Based on http://stackoverflow.com/a/3844948/3961920
    If they're all the same, returns True, otherwise returns False.

    :param list in_list: List to check for equality of all elements.
    :rtype: bool
    :return: True if all elements the same; False otherwise."""
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


def find_nearest_idx(array, value):
    """
    Find index of value nearest to value in an array

    :param np.ndarray array: Array of values in which to look
    :param float value: Value for which the index of the closest value in
        `array` is desired.
    :rtype: int
    :return: The index of the item in `array` nearest to `value`
    """
    return (np.abs(array - value)).argmin()


def running_mean(x, n=2):
    """
    Calculate running mean over an iterable

    Taken from https://stackoverflow.com/a/22621523/3961920

    :param Iterable x: List over which to calculate the mean.
    :param int n: Default: 2. Width for the means.
    :return: Array of the running mean values.
    :rtype: np.ndarray
    """
    if len(x) != 0:
        return np.convolve(x, np.ones((n,)) / n, mode='valid')
    else:
        raise ValueError('x cannot be empty')
