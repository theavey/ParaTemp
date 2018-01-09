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
import subprocess
from typing import Callable, Iterable, Match
import warnings

from .tools import _BlankStream
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

    Note, this function is being deprecated in favor of the more general
    function get_solv_count_top, which takes the strengths of this function
    while also allowing for specification of an exact top file.

    :param str folder: The folder in which to look for a file ending in '.top'.
    :param str solvent: Default: 'DCM'
    :return: The number of solvent molecules.
    :rtype: int
    """
    warnings.warn('This function is deprecated. Please use '
                  'get_solv_count_top', DeprecationWarning)
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
            # Not the right error, but fine for now
            raise ValueError("Didn't find n_solv in {}".format(folder))


def get_solv_count_top(n_top=None, folder=None, res_name='DCM'):
    """
    Find the number of solvent molecules of given residue in topology file.

    All lines that start with a semicolon or whitespace followed by a
    semicolon will be automatically ignored (semicolons are the comment
    characters in topology files).

    :param str n_top: Default: None. Name (and path) of the topology file. If
        None, folder will be used, but this argument takes priority.
    :param str folder: Default: None. If n_top is not provided, this is the
        folder that will be searched for a file ending in '.top' to be searched.
    :param str res_name: Default: 'DCM'. Name of the residue to look for in
        the topology file. This is case insensitive (the re.IGNORECASE flag
        is used).
    :return: The found number of solvent molecules in the topology file.
    :rtype: int
    :raises ValueError: This is raised if more than one topology is found in
        the given folder.
    :raises RuntimeError: This is raised if the regex is unable to find the
        line with the solvent count. This could also be raised if it cannot find
        the molecules section.
    """
    re_n_solv = re.compile(r'(?:^\s*{}\s+)(\d+)'.format(res_name),
                           flags=re.IGNORECASE)
    n_top = _get_n_top(n_top, folder)
    with open(n_top, 'r') as in_top:
        mol_section = False
        for line in in_top:
            if line.strip().startswith(';'):
                pass
            elif not mol_section:
                if re.search(r'\[\s*molecules\s*\]', line,
                             flags=re.IGNORECASE):
                    mol_section = True
            else:
                solv_match = re_n_solv.search(line)
                if solv_match:
                    return int(solv_match.group(1))
        # Not the right error, but fine for now
        raise RuntimeError('Did not find a line with the solvent count in '
                           '{}'.format(n_top))


def _get_n_top(n_top, folder):
    """
    Get path and name of topology file

    :param str n_top: None or path and file name of topology file.
    :param str folder: None or folder containing one topology file.
    :return: path to the topology file
    :rtype: str
    :raises ValueError: This is raised if more than one topology is found in
        the given folder.
    """
    if n_top is None:
        if folder is None:
            raise InputError('None', 'Either folder or n_top must be '
                             'specified')
        with cd(folder):
            n_top = glob.glob('*.top')
            if len(n_top) != 1:
                raise ValueError(
                    'Found {} .top files in {}\n'.format(len(n_top), folder) +
                    'Only can deal with 1')
            else:
                n_top = os.path.abspath(n_top[0])
    return n_top


def set_solv_count_top(n_top=None, folder=None, s_count=0,
                       res_name='DCM', prepend='unequal-', verbose=True):
    """
    Set the number of solvent molecules for a given residue in topology file.

    If the count of the solvent as found with get_solv_count_top is already
    equal to s_count, nothing will changed or copied.

    All lines that start with a semicolon or whitespace followed by a
    semicolon will be automatically ignored (semicolons are the comment
    characters in topology files).

    :param str n_top: Default: None. Name (and path) of the topology file. If
        None, folder will be used, but this argument takes priority.
    :param str folder: Default: None. If n_top is not provided, this is the
        folder that will be searched for a file ending in '.top' to be used.
    :param int s_count: Default: 0. The count of solvent molecules to be set
        in the topology file.
    :param str res_name: Default: 'DCM'. Name of the residue to look for in
        the topology file. This is case insensitive (this and the line will
        be made lower case when searching for it).
    :param str prepend: Default: 'unequal-'. The string to prepend to the
        topology file name when copying it (to keep a copy of the original).
    :param bool verbose: Default: True. If True, messages will be printed if
        no changes need to be made or after the changes have successfully been
        made.
    :return: None
    :raises RuntimeError: This is raised if the it is unable to find the
        line with the solvent count. This could also be raised if it cannot find
        the molecules section.
    """
    n_top = _get_n_top(n_top, folder)
    if s_count == get_solv_count_top(n_top=n_top, res_name=res_name):
        if verbose:
            print('Solvent count in {} already set at {}'.format(
                        os.path.relpath(n_top), str(s_count)) +
                  '\nNot copying or changing file.')
        return None
    bak_name = os.path.join(os.path.dirname(n_top),
                            prepend+os.path.basename(n_top))
    copy_no_overwrite(n_top, bak_name)
    with open(n_top, 'r') as in_top:
        lines = in_top.readlines()
    with open(n_top, 'w') as out_top:
        mol_section = False
        done = False
        for line in lines:
            if line.strip().startswith(';'):
                pass
            elif not mol_section:
                if re.search(r'\[\s*molecules\s*\]', line,
                             flags=re.IGNORECASE):
                    mol_section = True
            elif not done and res_name.lower() in line.lower():
                line = re.sub(r'\d+', str(s_count), line)
                done = True
            out_top.write(line)
    if not done:
        # Not the right error, but fine for now
        raise RuntimeError('Did not find a line with the solvent count'
                           ' in {}'.format(n_top))
    elif verbose:
        print('Solvent count in {} set at {}'.format(
                    os.path.relpath(n_top), str(s_count)) +
              '\nOriginal copied to {}.'.format(
                    os.path.relpath(bak_name)
              ))
    return None


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


d_cgenff_ptad_repls = {1: 63, 9: 69, 8: 72, 120: 182}


def _update_num(match, shift=120,
                cat_repl_dict=d_cgenff_ptad_repls):
    """
    Return a string with an updated number based on atom-index changes

    For a similar function to be used in the case of a changing catalyst that is
    still going to be the first molecule, the `shift' functionality will need to
    be updated/changed (or a different function can be defined and passed to
    update_plumed_input).

    :param Match match: Match object with two groups for the pre-string and
        number to be changed. The first group is returned as-is. The second
        group is the number that needs to be changed based on the rules.
    :param int shift: Default: 120. Number by which to shift the reactants.
        In most use cases, the catalyst was the first molecule, and gets moved
        to after the reactants. If it gets moved to after the reactants,
        the is the number of atoms in the original catalyst.
    :param dict[int, int] cat_repl_dict: Default: d_cgenff_ptad_repls. dict
        of atom index replacements to do for the catalyst.
    :rtype: str
    :return: pre-string combined with the new atom index
    """
    pre, s = match.groups()
    try:
        n = int(s)
    except ValueError:
        raise ValueError('"{}" cannot be converted to a valid int'.format(s))
    if n < shift+1:
        out = cat_repl_dict[n]
    else:
        out = n - shift
    return pre + str(out)


c_line_keywords = {'WHOLEMOLECULES', 'c1:', 'c2:',
                   'g1:', 'g2:', 'g3:', 'g4:',
                   'dm1:', 'dm2:'}
d_line_keywords = {'tr5:', 'tr6:', 'FILE=COLVAR'}
d_equil_repls = {'dm2:': ['72', '71'],
                 'dm1:': ['40', '12']}


def update_plumed_input(n_plu_in, n_plu_out,
                        change_keys=c_line_keywords,
                        num_updater=_update_num,
                        delete_keys=d_line_keywords,
                        equil=False,
                        equil_changes=d_equil_repls):
    """
    Write a changed PLUMED input based on a previous PLUMED input

    :param str n_plu_in: Name of PLUMED source file to be used as a template.
        This file is not changed.
    :param str n_plu_out: Name of the PLUMED file to be written. This file
        will be overwritten if it already exists.
    :type change_keys: set[str] or Iterable[str]
    :param change_keys: Keywords to look for in lines that should be changed.
        Note, these need to be found space separated in the lines to be changed.
    :type num_updater: Callable[[Match], str]
    :param num_updater: Default: _update_num. A function that takes a
        re.MatchObject and returns a str. The match object will have two (
        three including the full match) groups. The first will be a
        pre-string that should be returned as-is, and the second will be a
        string of an int that should be changed based on how the atom indices
        have changed.
    :type delete_keys: set[str] or Iterable[str]
    :param delete_keys: Keywords to look for in lines that should be deleted.
        Note, these need to be found space separated in the lines to be deleted.
    :param bool equil: Default: False. If True, other specified changes can be
        made to produce PLUMED input suitable for equilibration before a
        production run
    :param dict equil_changes: dict with changes to make for equilibration
        PLUMED input. Note, the keys in this dict must match an appropriate
        key in change_keys for this to work currently.
    :return: None
    """
    with open(n_plu_in, 'r') as from_file, \
            open(n_plu_out, 'w') as to_file:
        for line in from_file:
            c_key_match = set(line.split()) & set(change_keys)
            if c_key_match:
                line = re.sub(r'([=,-])(\d+)', num_updater, line)
                if equil:
                    if len(c_key_match) > 1:
                        raise KeyError('More than one keyword matched in '
                                       'line: {}'.format(c_key_match))
                    key = c_key_match.pop()
                    if key in equil_changes.keys():
                        line = line.replace(*equil_changes[key])
            elif set(line.split()) & set(delete_keys):
                line = ''
            elif equil and line.startswith('UPPER_WALLS'):
                # soften the upper walls
                line = line.replace('150.0,150.0 EXP=2,2', '75.0,75.0 EXP=1,1')
                # pull them a little closer to make sure they're within the
                # walls for the production simulation
                line = line.replace('AT=12.0,12.0', 'AT=10.5,10.5')
            to_file.write(line)
