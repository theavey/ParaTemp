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


def compile_tprs(template='templatemdp.txt', start_temp=205., number=16,
                 scaling_exponent=0.025, base_name='npt',
                 topology='../*top', multi_structure=False,
                 structure='../*gro', index='../index.ndx',
                 temps_file='temperatures.dat'):
    """
    Compile TPR files for REMD run with GROMACS

    :param template:
    :param start_temp:
    :param number:
    :param scaling_exponent:
    :param base_name:
    :param topology:
    :param multi_structure:
    :param structure:
    :param index:
    :param temps_file:
    :return:
    """
    # if args.multi_structure:
    from glob import glob
    structures = glob(structure+'*.gro')
    structures.sort()
    structures.sort(key=len)
    temps = []
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
                        '-maxwarn', '2']
        with open('gromacs_compile_output.log', 'w') as log_file:
            from subprocess import Popen, PIPE
            proc = Popen(command_line,
                         stdout=PIPE, bufsize=1,
                         universal_newlines=True)
            for line in proc.stdout:
                log_file.write(line)
    with open(temps_file, 'w') as temps_out:
        temps_out.write(str(temps))
        temps_out.write('\n')


if __name__ == "__main__":
    import argparse

    __version__ = '0.1.3'

    parser = argparse.ArgumentParser(description='A script to help setup parallel '
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
                        default='../taddol_3htmf_stilbene_em.top',
                        help='name of topology file (.top)')
    parser.add_argument('-m', '--multi_structure', default=False,
                        help='Use multiple starting structure files')
    parser.add_argument('-c', '--structure', default='../major_endo.gro',
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
