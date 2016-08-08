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
import numpy as np

__version__ = '0.0.1'

# todo add argument and code for way to save to a file instead of viewing
# todo take arguments to change the plotting
parser = argparse.ArgumentParser(description='A script to plot energy '
                                             'histograms from a GROMACS '
                                             'parallel tempering simulation.')
parser.add_argument('--version', action='version',
                    version='%(prog)s v{}'.format(__version__))
args = parser.parse_args()


def find_energies():
    """find_energies() is a function that finds all files in the current
    directory that end in a numeral followed by '.edr'. For each of these
    files, it checks if a file named energy(same number).xvg exists, and if
    not, creates it by calling the GROMACS tool gmx energy where input='13'
    corresponds to the total energy at each time."""
    energy_files = glob.glob('*[0-9].edr')
    output_files = []
    for file_name in energy_files:
        output_name = ('energy' + re.search('[0-9]*(?=\.edr)',
                                            file_name).group(0) + '.xvg')
        if not os.path.isfile(output_name):
            gromacs.tools.Energy(f=file_name, o=output_name, input="13")()
        output_files += [output_name]
    output_files.sort()
    output_files.sort(key=len)
    return output_files


def import_energies(output_files):
    """import_energies(file_list) takes a list of .xvg files in the current
    directory, imports their second columns (likely a list of energies at
    consecutive time steps), and returns that as a list of arrays."""
    imported_data = []
    for file_name in output_files:
        xvg_file = gromacs.formats.XVG(filename=file_name)
        imported_data += [xvg_file.array[1]]
    return imported_data


# Run this only if called from the command line
if __name__ == "__main__":
    out_files = find_energies()
    all_data = import_energies(out_files)

    plt.hist(all_data, 50, histtype='stepfilled')

    plt.show()


def combine_energy_files(basename='energy', files=False):
    """combine_energy_files(basename='energy', files=False) is a function that
    combines a set of .xvg files writes a combined .xvg file.
    'basename' is the first part of the name for the xvg files to be combined and
    should differentiate these from any other *.xvg files in the folder.
    Alternatively, the list of files (in the desired order) can be passed in
    with the keyword 'files'."""
    if not files:
        files = glob.glob(basename + '*.xvg')
        files.sort()
        files.sort(key=len)
    data = [gromacs.formats.XVG(filename=files[0]).array[0]]
    data += import_energies(files)
    data = np.array(data)
    gromacs.formats.XVG(array=data).write(filename=basename+'_comb.xvg')


def deconvolve_energies(energyfile='energy_comb.xvg',
                        indexfile='replica_index.xvg'):
    """deconvolve_energies(energyfile='energy_comb.xvg',
    indexfile='replica_index.xvg') is a function that takes an xvg files that is
    has n columns of energies likely from a replica exchange simulation where each
    replica remains at a constant temperature (as GROMACS does) and using the n
    data columns of an index xvg file returns an array of the energies where each
    row is now from one 'walker' (continuous coordinates taken by sampling various
    temperatures or other replica conditions)."""
    energies_indexed = gromacs.formats.XVG(filename=energyfile).array
    indices_indexed = gromacs.formats.XVG(filename=indexfile).array.astype(int)
    # todo check for relative start/end points
    # todo fix indices/energies in a more general way
    # todo change range to fit more generally
    ratio = indices_indexed.shape[1] / energies_indexed.shape[1]
    if ratio < 1.0:
        print("I'm not sure how to handle the ratio {} yet".format(ratio))
        return(IndexError)
    ratio = int(ratio)
    extra = np.mod(energies_indexed.shape[1], ratio)
    length_e = energies_indexed.shape[1] - extra
    length_i = indices_indexed.shape[1] / ratio
    if length_e != length_i:
        print("length of energies, {}, != length of indices, "
              "{}!".format(length_e, length_i))
        return(IndexError)
    deconvolved_energies = energies_indexed[1:, :-extra][indices_indexed[1:, ::ratio],
                                                         np.arange(length_e)]
    return deconvolved_energies


