#! /usr/bin/env python3

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2016-17.                 #
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

# This is written to work with python 3 because it should be good to
# be working on the newest version of python.

import errno
import glob
from math import exp
import os
import re
import shlex
import shutil
import subprocess
from subprocess import Popen, PIPE, STDOUT
from warnings import warn

from .sim_setup import _submit_script
from .tools import _BlankStream, _replace_string_in_file
from .exceptions import InputError
from .tools import cd
from ._version import get_versions


def compile_tprs(template='templatemdp.txt', start_temp=205., number=16,
                 scaling_exponent=0.025, base_name='npt',
                 topology='../*top', multi_structure=False,
                 structure='../*gro', index='../index.ndx',
                 temps_file='temperatures.dat', maxwarn='0',
                 grompp_exe='gmx_mpi grompp'):
    """
    Compile TPR files for multi-temperature run with GROMACS

    This works in the current directory.

    With exponential temperature spacing, this is mostly useful for compiling
    TPR files for replica exchange dynamics.

    Directly, this function will write two files in the current directory:
    'gromacs_compile_output.log' and ``temps_file`` (by default,
    'temperatures.dat'). These are the stdout and stderr from grompp and the
    temperatures of the simulations, respectively.

    :param str template: name of template mdp file
    :param float start_temp: starting (lowest) temperature
    :param int number: number of replicas/walkers
    :param float scaling_exponent: exponent by which to scale the temperatures
    :param str base_name: base name for output mdp and tpr files
    :param str topology: name of topology file
    :param bool multi_structure: multiple (different) structure files
        (uses glob expansion on the input structure base name)
    :param str structure: (base) name of structure file(s)
    :param str index: name of index file
    :param str temps_file: name of file in which to store temperatures
    :type maxwarn: int or str
    :param maxwarn: maximum number of warnings to ignore. str is applied to
        this argument, so type shouldn't matter significantly.
    :param str grompp_exe: The name of the GROMACS executable. This is often
        just `gmx grompp`, but on some systems the MPI-compiled version may be
        `gmx_mpi grompp`, as is true on my system.
        On older versions of GROMACS, it may be something like `grompp` alone.
    :return: None
    :raises OSError: If structure or topology files not found (or not enough
        found if ``multi_structure=True``).
    """
    if multi_structure:
        structures = glob.glob(structure+'*.gro')
        structures.sort()
        structures.sort(key=len)
        if len(structures) != number:
            raise OSError(
                errno.ENOENT, 'Incorrect number of structure files found.\n'
                              'Found {}, needed {}.'.format(len(structures),
                                                            number))
        _structure = structures[number]  # just to prevent IDE warning
    else:
        structures = glob.glob(structure)
        if len(structures) > 1:
            _structure = structures[0]
            warn('Found {} structure files, '
                 'using {}'.format(len(structures), _structure))
        elif len(structures) == 0:
            raise OSError(errno.ENOENT, 'No structure file found.')
        else:
            _structure = structures[0]
    try:
        _topology = glob.glob(topology)[0]
    except IndexError:
        raise OSError(errno.ENOENT, 'No topology file found.')
    temps = []
    error = False
    for i in range(number):
        mdp_name = base_name + str(i) + '.mdp'
        temp = start_temp * exp(i * scaling_exponent)
        temps += [temp]
        if multi_structure:
            _structure = structures[i]
        with open(template, 'r') as f_template, \
                open(mdp_name, 'w') as out_file:
            for line in f_template:
                if 'TempGoesHere' in line:
                    line = line.replace('TempGoesHere', str(temp))
                out_file.write(line)
        command_line = shlex.split(grompp_exe)
        command_line += ['-f', mdp_name,
                         '-p', _topology,
                         '-c', _structure,
                         '-n', index,
                         '-o', mdp_name.replace('mdp', 'tpr'),
                         '-maxwarn', str(maxwarn)]
        with open('gromacs_compile_output.log', 'a') as log_file:
            proc = Popen(command_line,
                         stdout=PIPE, bufsize=1,
                         stderr=STDOUT,
                         universal_newlines=True)
            stdout = proc.communicate()[0]
            for line in stdout.splitlines():
                if error is True:  # Catch the next line after the error
                    error = line
                elif error:  # If error is not True but is set to string
                    pass
                elif ('Fatal error' in line or
                        'File input/output error' in line or
                        'Error in user input' in line):
                    error = True  # Deal with this after writing log file
                log_file.write(line+'\n')
        if error or proc.returncode != 0:
            error = error if error else 'Unknown error. Check log file.'
            raise RuntimeError(error, 'returncode: {}'.format(proc.returncode))
    with open(temps_file, 'w') as temps_out:
        temps_out.write(str(temps))
        temps_out.write('\n')


if __name__ == "__main__":
    from argparse import ArgumentParser

    __version__ = get_versions()['version']

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


def extend_tprs(base_name, time, working_dir=None, sub_script=None,
                submit=False, extend_infix='-extend', first_extension=True,
                cpt_base='npt', verbose=True,
                log='extend-tprs.log'):
    """
    Extend a set of tpr files

    :param str base_name: Base of the tpr files. This should return the file
        names when globbed with '\*.tpr' appended to this base name. Also, this
        will cause issues when adding the infix if the file name doesn't fit
        the pattern of '{base_name}{number}.tpr'.
    :param time: Amount of time in picoseconds by which to extend the job. This
        will be cast to a string, so an int, string, or float should be fine.
    :type time: str or int or float
    :param str working_dir: Default: None. If given, this directory will be
        changed into and work will continue there.
        If working_dir is None, the working dir will be taken to be the
        directory one directory above the location given in base_name.
    :param str sub_script: Default: None. Name of the submission script. If
        given, the script will be edited to match the new name of the extended
        tpr files.
        sub_script can be given as an absolute path or relative to current
        directory (first priority) or relative to working_dir (checked second).
    :param bool submit: Default: False. If true, the job will be submitted to
        the queuing system.
    :param str extend_infix: Default: '-extend'. str to put into the name of the
        extended tpr files after the base_name and before the '[number].tpr'.
    :param bool first_extension: Default: True. If True, '-cpi {checkpoint
        base name}' will be added to the submission script so that it becomes a
        run continuation.
    :param str cpt_base: Default: 'npt'. The first part of the name of the
        checkpoint files that will end in '{number}.cpt'. The full checkpoint
        base_name will be found using
        :func:`~paratemp.para_temp_setup._find_cpt_base`.
    :param bool verbose: Default: True. If True, a lot more status information
        will be printed.
    :param str log: Default: 'extend-tprs.log'. Name of file to which to log
        information on this process and output from GROMACS tools.
    :return: None
    """
    _tpr_dir, _rel_base_name = os.path.split(os.path.abspath(base_name))
    if working_dir is None:
        _working_dir = os.path.abspath(_tpr_dir+'/../')
    else:
        _working_dir = working_dir
    if sub_script is not None:
        second_poss = os.path.abspath(os.path.join(_working_dir, sub_script))
        if os.path.isfile(sub_script):
            _sub_script = os.path.abspath(sub_script)
        elif os.path.isfile(second_poss):
            _sub_script = second_poss
        else:
            raise OSError(errno.ENOENT, 'Submit script not found relative to '
                                        'here or working_dir.')
    else:
        _sub_script = None  # Only needed so the IDE stops bothering me
    _time = str(time)
    re_split_name = re.compile(r'({})(\d+\.tpr)'.format(_rel_base_name))
    with cd(_working_dir), open(log, 'a') as _log:
        with cd(_tpr_dir):
            tpr_names = glob.glob(_rel_base_name+'*.tpr')
            if len(tpr_names) < 1:
                raise InputError(base_name, 'no files found for {}'.format(
                                     base_name+'*.tpr'))
            if verbose:
                print('Extending {} tpr files'.format(len(tpr_names)))
            for tpr_name in tpr_names:
                tpr_groups = re_split_name.match(tpr_name)
                new_tpr_name = (tpr_groups.group(1) + extend_infix +
                                tpr_groups.group(2))
                _extend_tpr(tpr_name, new_tpr_name, _time, _log)
            if verbose:
                print(' '*4 + 'Done extending tpr files.')
        if sub_script is not None:
            _sub_script = os.path.relpath(_sub_script)
            if verbose:
                print('Editing '
                      '{} for new tpr names with {}'.format(_sub_script,
                                                            extend_infix))
            _replace_string_in_file(_rel_base_name + ' ', _rel_base_name +
                                    extend_infix + ' ', _sub_script, _log)
            if first_extension:
                _cpt_base = _find_cpt_base(cpt_base)
                _add_cpt_to_sub_script(_sub_script, _cpt_base, _log)
            if submit:
                if verbose:
                    print('Submitting job...')
                job_info = _submit_script(_sub_script, _log)
                if verbose:
                    print('Job number {} has been submitted.'.format(
                        job_info[2]))
        elif submit:
            print('Job not submitted because no submission script name was '
                  'provided.')


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
    log_stream.write('Extending {} as {}\n'.format(old_name, new_name))
    log_stream.flush()
    cl = ['gmx_mpi', 'convert-tpr',
          '-s', old_name,
          '-o', new_name,
          '-extend', time]
    return_code = subprocess.call(cl, stderr=log_stream, stdout=log_stream,
                                  universal_newlines=True)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ' '.join(cl))


def _find_cpt_base(cpt_base):
    """
    Find checkpoint file base name in current directory

    :param str cpt_base: Start of checkpoint file name that ends with a
        number of one to three digits followed by '.cpt'
    :return: The base name of the checkpoint files (everything but the number
        and ".cpt")
    :rtype: str
    """
    possible_matches = glob.glob(cpt_base+'*.cpt')
    for f_name in possible_matches:
        match = re.match(r'({}.*?)\d{}\.cpt'.format(cpt_base, '{1,3}'), f_name)
        if match:
            return match.group(1)
    raise ValueError('No checkpoint file name found based on the base '
                     'name {}.'.format(cpt_base))


def _add_cpt_to_sub_script(sub_script, cpt_base, log_stream=_BlankStream(),
                           temp_bak_name='temp-submission-script.bak'):
    """
    Add a checkpoint file to GROMACS submission line

    Note, only one replacement will be done here.

    :param str sub_script: Name of the submission script
    :param str cpt_base: Base name of the checkpoint file(s) to pass to GROMACS
    :type log_stream: _BlankStream or BinaryIO
    :param log_stream: Default: _BlankStream(). The file stream to which to log
        information. The default will just not log anything.
    :param str temp_bak_name: Name for a temporary backup of the submission
        script in case things go wrong. Assuming no exceptions are raise,
        this will be deleted after the new file is written.
    :return: None
    """
    re_mdrun_line = re.compile('mdrun_mpi|gmx_mpi\s+mdrun|gmx\s+mdrun_mpi')
    log_stream.write('Adding "-cpi {}" to {}\n'.format(cpt_base, sub_script))
    log_stream.flush()
    with open(sub_script, 'r') as f_in:
        lines_in = f_in.readlines()
    with open(temp_bak_name, 'w') as temp_out:
        [temp_out.write(line) for line in lines_in]
    with open(sub_script, 'w') as f_out:
        changed = False
        for line in lines_in:
            line_list = line.split('#', 1)
            line = line_list[0]
            if not changed:
                match = re_mdrun_line.search(line)
                if match:
                    if '-cpi ' not in line:
                        line = line.replace('\n', ' ') + '-cpi {}\n'.format(
                            cpt_base)
                    changed = True
            if len(line_list) > 1:
                line = '#'.join(line_list)
            f_out.write(line)
        else:
            os.remove(temp_bak_name)
    if not changed:
        raise ValueError('Could not find GROMACS mdrun line in submission '
                         'script, so the checkpoint file ("-cpi ...") was not '
                         'added to it.')


def cleanup_bad_gromacs_restart(out_base, working_dir='./', list_files=True,
                                replace_files=False, verbose=True):
    """
    Replace "new" files with GROMACS backed-up files after messed-up restart

    No timestamps are accounted for, and this is purely based on the file
    names and the default way GROMACS backs up files it would have otherwise
    replaced.

    :param str out_base: Base name for output files, likely the same as the
        '-deffnm' argument.
    :param str working_dir: Directory in which to look and do these replacements
    :param str list_files: If true, matched and unmatched files will all be
        printed
    :param bool replace_files: If true, the backed-up files will be moved to
        overwrite the "new" files.
    :param bool verbose: If true, more file counts and such will be printed.
    :return: None
    """
    with cd(working_dir):
        good_files = glob.glob('#'+out_base+'*')
        bad_files = glob.glob(out_base+'*')
        good_files.sort()
        bad_files.sort()
        if verbose:
            print('Found {} "bad" and {} "good" files.'.format(len(
                bad_files), len(good_files)))
        match_dict = dict()
        unmatched_good = list()
        unmatched_bad = list(bad_files)
        for g_name in good_files:
            poss_bad_name = g_name[1:-3]  # remove # from start and end and .1
            if poss_bad_name in bad_files:
                unmatched_bad.remove(poss_bad_name)
                match_dict[poss_bad_name] = g_name
            else:
                unmatched_good.append(g_name)
        if verbose:
            print('Total of {} matched files.'.format(len(match_dict)))
        if len(unmatched_good) + len(unmatched_bad) != 0 and verbose:
            print('Unmatched file counts:\n    '
                  'good:{:>3}\n    bad:{:>3}'.format(len(unmatched_good),
                                                     len(unmatched_bad)))
        elif verbose:
            print('No unmatched files.')
        if list_files:
            if len(unmatched_good) != 0:
                print('Unmatched "good" files:')
                for g_name in unmatched_good:
                    print('    {}'.format(g_name))
            else:
                print('No unmatched "good" files')
            if len(unmatched_bad) != 0:
                print('Unmatched "bad" files:')
                for b_name in unmatched_bad:
                    print('    {}'.format(b_name))
            else:
                print('No unmatched "bad" files')
            if len(match_dict) > 0:
                print('Matched files:\n')
                print('-'*63)
                print('{:^30} | {:^30}'.format('good', 'bad'))
                print('-'*63)
                for key in sorted(match_dict):
                    print('{:>30} | {:>30}'.format(match_dict[key], key))
                print('-'*63)
            else:
                print('No matched files!!')
        if replace_files:
            if verbose:
                print('Now replacing "bad" with matched "good" files.')
            for b_name in match_dict:
                shutil.move(match_dict[b_name], b_name)
            if verbose:
                print('Done replacing files.')
