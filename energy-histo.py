#! /usr/bin/env python

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

import argparse
import glob
import gromacs.formats
import gromacs.tools
import re
import matplotlib.pyplot as plt
import os

__version__ = '0.0.1'

# todo add argument and code for way to save to a file instead of viewing
# todo make this a function and only call it if the script is called alone
# todo take arguments to change the plotting
parser = argparse.ArgumentParser(description='A script to plot energy '
                                             'histograms from a GROMACS '
                                             'parallel tempering simulation.')
parser.add_argument('--version', action='version',
                    version='%(prog)s v{}'.format(__version__))
args = parser.parse_args()


# Find .edr files in this directory and make .xvg files for each
def find_energies():
    energy_files = glob.glob('*[0-9].edr')
    output_files = []
    for file_name in energy_files:
        output_name = ('energy' + re.search('[0-9]*(?=\.edr)',
                                            file_name).group(0) + '.xvg')
        if not os.path.isfile(output_name):
            gromacs.tools.G_energy(f=file_name, o=output_name, input="13")()
        output_files += [output_name]
    return output_files


def import_energies(output_files):
    imported_data = []
    for file_name in output_files:
        xvg_file = gromacs.formats.XVG(filename=file_name)
        imported_data += [xvg_file.array[1]]
    return imported_data

output_files = find_energies()
imported_data = import_energies(output_files)

plt.hist(imported_data, 50, histtype='stepfilled')

plt.show()
