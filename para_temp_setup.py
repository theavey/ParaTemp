#! /usr/bin/env python3

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2016.                    #
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


if __name__ == "__main__":
    import glob
    import argparse
    import math
    import subprocess
    import readline
    import os
    import shutil
    from datetime import datetime

    __version__ = '0.1.2'

    parser = argparse.ArgumentParser(description='A script to help setup parallel'
                                                 'tempering jobs in GROMACS with'
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
    # todo accept different structure files (a series/list of them)
    parser.add_argument('-c', '--structure', default='../major_endo.gro',
                        help='structure file (.gro) ')
    parser.add_argument('--index', default='../index.ndx',
                        help='index files')
    parser.add_argument('-t', '--temps_file', default='temperatures.dat',
                        help='name of file with list of temperatures')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v{}'.format(__version__))
    args = parser.parse_args()

    number = int(args.number)
    start_temp = float(args.start_temp)
    scaling_exponent = float(args.scaling_exponent)

    # todo make this a function and call it
    temps = []
    for i in range(number):
        mdp_name = args.base_name + str(i) + '.mdp'
        temp = start_temp * math.exp(i * scaling_exponent)
        temps += [temp]
        with open(args.template, 'r') as template, \
                open(mdp_name, 'w') as out_file:
            for line in template:
                if 'TempGoesHere' in line:
                    line = line.replace('TempGoesHere', str(temp))
                out_file.write(line)
        command_line = ['grompp_mpi',
                        '-f', mdp_name,
                        '-p', args.topology,
                        '-c', args.structure,
                        '-n', args.index,
                        '-o', mdp_name.replace('mdp', 'tpr'),
                        '-maxwarn', '2']
        with open('gromacs_compile_output.log', 'w') as log_file:
            with subprocess.Popen(command_line,
                                  stdout=subprocess.PIPE, bufsize=1,
                                  universal_newlines=True) as proc:
                for line in proc.stdout:
                    log_file.write(line)
    with open(args.temps_file, 'w') as temps_out:
        temps_out.write(temps)
