#! /usr/bin/env python3

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2016-17.                 #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2016 Thomas J. Heavey IV                                   #
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

# This is written to work with python 3 because it should be good to
# be working on the newest version of python.

import os

import re
import glob
import subprocess

from .tools import cd, copy_no_overwrite
from .exceptions import InputError


def compile_tprs(template='templatemdp.txt', start_temp=205., number=16,
                 scaling_exponent=0.025, base_name='npt',
                 topology='../*top', multi_structure=False,
                 structure='../*gro', index='../index.ndx',
                 temps_file='temperatures.dat', maxwarn='0'):
    """
    Compile TPR files for REMD run with GROMACS

    :param template: name of template mdp file
    :param start_temp: starting (lowest) temperature
    :param number: number of replicas/walkers
    :param scaling_exponent: exponent by which to scale the temperatures
    :param base_name: base name for output mdp and tpr files
    :param topology: name of topology file
    :param multi_structure: bool, multiple (different) structure files
    (uses glob expansion on the input structure base name)
    :param structure: (base) name of structure file(s)
    :param index: name of index file
    :param temps_file: name of file in which to store temperatures
    :param maxwarn: maximum number of warnings to ignore. str is applied to
    this argument, so type shouldn't matter significantly.
    :type maxwarn: int or str
    :return: None
    """
    # if args.multi_structure:
    from glob import glob
    structures = glob(structure+'*.gro')
    structures.sort()
    structures.sort(key=len)
    temps = []
    error = False
    from math import exp
    for i in range(number):
        mdp_name = base_name + str(i) + '.mdp'
        temp = start_temp * exp(i * scaling_exponent)
        temps += [temp]
        if multi_structure:
            structure = structures[i]
        with open(template, 'r') as f_template, \
                open(mdp_name, 'w') as out_file:
            for line in f_template:
                if 'TempGoesHere' in line:
                    line = line.replace('TempGoesHere', str(temp))
                out_file.write(line)
        command_line = ['grompp_mpi',
                        '-f', mdp_name,
                        '-p', topology,
                        '-c', structure,
                        '-n', index,
                        '-o', mdp_name.replace('mdp', 'tpr'),
                        '-maxwarn', str(maxwarn)]
        with open('gromacs_compile_output.log', 'a') as log_file:
            from subprocess import Popen, PIPE, STDOUT
            proc = Popen(command_line,
                         stdout=PIPE, bufsize=1,
                         stderr=STDOUT,
                         universal_newlines=True)
            for line in proc.stdout:
                if error is True:  # Catch the next line after the error
                    error = line
                if ('Fatal error' in line or
                        'File input/output error' in line):
                    error = True  # Deal with this after writing log file
                log_file.write(line)
        if error:
            raise RuntimeError(error)
    with open(temps_file, 'w') as temps_out:
        temps_out.write(str(temps))
        temps_out.write('\n')


if __name__ == "__main__":
    from argparse import ArgumentParser

    __version__ = '0.1.3'

    parser = ArgumentParser(description='A script to help setup parallel '
                                        'tempering jobs in GROMACS with '
                                        'PLUMED')
    parser.add_argument('-l', '--template', default='templatemdp.txt',
                        help='name of template file')
    parser.add_argument('-s', '--start_temp', default=205,
                        help='starting (lowest) temperature')
    parser.add_argument('-n', '--number', default=16,
                        help='number of replicates')
    parser.add_argument('-e', '--scaling_exponent', default=0.025,
                        help='exponent by which to scale temps')
    parser.add_argument('-b', '--base_name', default='npt',
                        help='base name for output mdp and tpr files')
    parser.add_argument('-p', '--topology',
                        default='../*.top',
                        help='name of topology file (.top)')
    parser.add_argument('-m', '--multi_structure', dest='multi_structure',
                        action='store_true',
                        help='Use multiple starting structure files')
    parser.set_defaults(multi_structure=False)
    parser.add_argument('-c', '--structure', default='../*.gro',
                        help='structure file or basename (.gro) ')
    parser.add_argument('--index', default='../index.ndx',
                        help='index files')
    parser.add_argument('-t', '--temps_file', default='temperatures.dat',
                        help='name of file with list of temperatures')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v{}'.format(__version__))
    args = parser.parse_args()

    # TODO see if argparse can do any type checking
    compile_tprs(
        template=args.template,
        start_temp=float(args.start_temp),
        number=int(args.number),
        scaling_exponent=float(args.scaling_exponent),
        base_name=args.base_name,
        topology=args.topology,
        multi_structure=args.multi_structure,
        structure=args.structure,
        index=args.index,
        temps_file=args.temps_file
    )


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


class _BlankStream(object):
    """
    A class for use when not actually wanting to write to a file.
    """
    def write(self, string):
        pass

    def fileno(self):
        return 0  # Not sure if this works. Maybe None would be better


def extend_tprs(base_name, time, sub_script=None, submit=False,
                extend_infix='-extend', verbose=True, log='extend-tprs.log'):
    """
    Extend a set of tpr files

    :param str base_name: Base of the tpr files. This should return the file
    names when globbed with '*.tpr' appended to this base name. Also, this will
    cause issues when adding the infix if the file name doesn't fit the pattern
    of '{base_name}{number}.tpr'.
    :param time: Amount of time in picoseconds by which to extend the job. This
    will be cast to a string, so an int or string should be fine (not sure if
    floats are okay).
    :type time: str or int
    :param str sub_script: Default: None. Name of the submission script. If
    given, the script will be edited to match the new name of the extended tpr
    files.
    :param bool submit: Default: False. If true, the job will be submitted to
    the queuing system.
    :param str extend_infix: Default: '-extend'. str to put into the name of the
    extended tpr files after the base_name and before the '[number].tpr'.
    :param bool verbose: Default: True. If True, a lot more status information
    will be printed.
    :param str log: Name of file to which to log information on this process and
    output from GROMACS tools.
    :return: None
    """
    re_split_name = re.compile(r'({})(\d+\.tpr)'.format(base_name))
    _time = str(time)
    with open(log, 'a') as _log:
        tpr_names = glob.glob(base_name+'*.tpr')
        if len(tpr_names) < 1:
            raise InputError(base_name,
                             'no files found for {}'.format(base_name+'*.tpr'))
        if verbose:
            print('Extending {} tpr files'.format(len(tpr_names)))
        for tpr_name in tpr_names:
            tpr_groups = re_split_name.match(tpr_name)
            new_tpr_name = tpr_groups.group(1)+extend_infix+tpr_groups.group(2)
            _extend_tpr(tpr_name, new_tpr_name, _time, _log)
        if verbose:
            print(' '*4+'Done extending tpr files.')
        if sub_script is not None:
            if verbose:
                print('Editing '
                      '{} for new tpr names with {}'.format(sub_script,
                                                            extend_infix))
            _replace_string_in_file(base_name, base_name + extend_infix,
                                    sub_script, _log)
        if submit:
            if verbose:
                print('Submitting job...')
        job_info = _submit_script(sub_script, _log)
        if verbose:
            print('Job number {} has been submitted.'.format(job_info[2]))


def _extend_tpr(old_name, new_name, time, log_stream=_BlankStream()):
    """
    Extend a tpr file with GROMACS convert-tpr and write to log file

    :param str old_name: Name of the old tpr file to be extended
    :param str new_name: Name for the new extended tpr file
    :param str time: Amount of time in picoseconds to extend the job
    :param log_stream: Default: _BlankStream(). The file stream to which to log
    information. The default will just not log anything.
    :type log_stream: _BlankStream or BinaryIO
    :return:
    """
    log_stream.write('Extending {} as {}'.format(old_name, new_name))
    cl = ['gmx_mpi', 'convert-tpr',
          '-s', old_name,
          '-o', new_name,
          '-extend', time]
    return_code = subprocess.call(cl, stderr=log_stream, stdout=log_stream,
                                  universal_newlines=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ' '.join(cl))


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
                     '{} for new string "{}"'.format(file_name,
                                                     new_str))
    log_stream.write('Copying file as backup to '
                     '{}'.format(file_name + '.bak'))
    copy_no_overwrite(file_name, file_name + '.bak')
    with open(file_name + '.bak', 'r') as old_f, open(file_name, 'w') as new_f:
        for line in old_f:
            line = line.replace(old_str, new_str)
            new_f.write(line)


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
    return match.group(1), match.group(2), match.group(0)
