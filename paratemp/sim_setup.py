"""
A set of functions for setuping up GROMACS simulations
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


import glob
import os
import re
import shutil
import subprocess

from .tools import _BlankStream, _replace_string_in_file
from .exceptions import InputError
from .tools import cd, copy_no_overwrite


def get_gro_files(trr_base='npt_PT_out', tpr_base='TOPO/npt',
                  time=200000):
    """
    Get a single frame from TRR as GRO file for several trajectories

    :param trr_base:
    :param tpr_base:
    :param time:
    :return:
    """
    from glob import glob
    trr_files = glob(trr_base+'*.trr')
    trr_files.sort()
    trr_files.sort(key=len)
    tpr_files = glob(tpr_base + '*.tpr')
    tpr_files.sort()
    tpr_files.sort(key=len)
    from gromacs.tools import Trjconv_mpi
    for i, trr_file in enumerate(trr_files):
        out_file = trr_file.replace('trr', 'gro')
        Trjconv_mpi(s=tpr_files[i], f=trr_file, o=out_file, dump=time,
                    input='0')()


def get_n_solvent(folder, solvent='DCM'):
    """
    Find the number of solvent molecules of given type in topology file.

    :param str folder: The folder in which to look for a file ending in '.top'.
    :param str solvent: Default: 'DCM'
    :return: The number of solvent molecules.
    :rtype: int
    """
    re_n_solv = re.compile('(?:^\s*{}\s+)(\d+)'.format(solvent))
    with cd(folder):
        f_top = glob.glob('*.top')
        if len(f_top) != 1:
            raise ValueError('Found {} .top files in {}\nOnly can deal with '
                             '1'.format(len(f_top), folder))
        else:
            f_top = f_top[0]
        with open(f_top, 'r') as file_top:
            for line in file_top:
                solv_match = re_n_solv.search(line)
                if solv_match:
                    return int(solv_match.group(1))
            else:
                # Not the right error, but fine for now
                raise ValueError("Didn't find n_solv in {}".format(folder))


def copy_topology(f_from, f_to, overwrite=False):
    os.makedirs(f_to, exist_ok=True)
    to_copy = glob.glob(f_from+'/*.top')
    to_copy += glob.glob(f_from+'/*.itp')
    for path in to_copy:
        copy_no_overwrite(path, f_to, silent=overwrite)


def _submit_script(script_name, log_stream=_BlankStream()):
    """
    Submit an existing submission script to qsub and return job information

    :param str script_name: Name of the script file.
    :param log_stream: Default: _BlankStream(). The file stream to which to log
    information. The default will just not log anything.
    :type log_stream: _BlankStream or BinaryIO
    :return: the job information as output by _job_info_from_qsub
    """
    cl = ['qsub', script_name]
    proc = subprocess.Popen(cl, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    output = proc.communicate()[0]
    log_stream.write(output)
    log_stream.flush()
    if proc.returncode != 0:
        print(output)
        raise subprocess.CalledProcessError(proc.returncode, ' '.join(cl))
    return _job_info_from_qsub(output)


def _job_info_from_qsub(output):
    """
    Get job information from the return from qsub

    :param str output: the line returned from qsub
    :return: the job number, the job name, and the job number and name as in the
    given string
    :rtype: Tuple(str, str, str)
    """
    match = re.search(r'(\d+)\s\("(\w.*)"\)', output)
    if not match:
        raise ValueError('Output from qsub was not able to be parsed: \n'
                         '    {}'.format(output))
    return match.group(1), match.group(2), match.group(0)
