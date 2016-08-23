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

import glob
import gromacs.formats
import gromacs.tools
import re
import matplotlib.pyplot as plt
import os
import numpy as np

__version__ = '0.0.2'


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
    import argparse
    # todo add argument and code for way to save to a file instead of viewing
    # todo take arguments to change the plotting
    parser = argparse.ArgumentParser(description='A script to plot energy '
                                                 'histograms from a GROMACS '
                                                 'parallel tempering simulation.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v{}'.format(__version__))
    args = parser.parse_args()

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
                        indexfile='replica_temp.xvg'):
    """deconvolve_energies(energyfile='energy_comb.xvg',
    indexfile='replica_temp.xvg') is a function that takes an xvg files that is
    has n columns of energies likely from a replica exchange simulation where each
    replica remains at a constant temperature (as GROMACS does) and using the n
    data columns of an index xvg file returns an array of the energies where each
    row is now from one 'walker' (continuous coordinates taken by sampling various
    temperatures or other replica conditions)."""
    energies_indexed = gromacs.formats.XVG(filename=energyfile).array
    indices_indexed = gromacs.formats.XVG(filename=indexfile).array.astype(int)
    # todo check for relative start/end points automatically
    length_e = energies_indexed.shape[1]
    length_i = indices_indexed.shape[1]
    ratio = float(length_e) / float(length_i)
    approx_ratio = int(round(ratio))
    if ratio == 1.0:
        deconvolved_energies = energies_indexed[1:][
            indices_indexed[1:],
            np.arange(length_i)]
    elif ratio > 1:
        if approx_ratio == ratio:
            extra_e = 0
            extra_i = 0
        elif approx_ratio > ratio:
            extra_i = length_i - length_e / approx_ratio
            extra_e = np.mod(length_e, length_i-extra_i)
        elif approx_ratio < ratio:
            extra_e = np.mod(length_e, length_i)
            extra_i = 0
        else:
            raise ImportError('ratio: {}, approx ratio: {}'.format(ratio, approx_ratio))
        deconvolved_energies = energies_indexed[1:, :length_e-extra_e:approx_ratio][
            indices_indexed[1:, :length_i-extra_i],
            np.arange((length_i-extra_i))]
        e_times = [energies_indexed[0, :length_e-extra_e:approx_ratio][0],
                   energies_indexed[0, :length_e-extra_e:approx_ratio][-1]]
        i_times = [indices_indexed[0, :length_i-extra_i][0],
                   indices_indexed[0, :length_i-extra_i][-1]]
    elif ratio < 1:
        print('likely undersampling energies because energy / indices ratio is '
              '{}'.format(ratio))
        ratio = 1 / ratio
        approx_ratio = int(round(ratio))
        if approx_ratio == ratio:
            extra_e = 0
            extra_i = 0
        # Not so sure about this...
        elif approx_ratio > ratio:
            extra_e = length_e - length_i / approx_ratio
            extra_i = np.mod(length_i, length_e-extra_e)
        elif approx_ratio < ratio:
            extra_i = np.mod(length_i, length_e)
            extra_e = 0
        else:
            raise ImportError('ratio: {}, approx ratio: {}'.format(ratio, approx_ratio))
        deconvolved_energies = energies_indexed[1:, :length_e-extra_e][
            indices_indexed[1:, :length_i-extra_i:approx_ratio],
            np.arange((length_i-extra_i)/approx_ratio)]
        e_times = [energies_indexed[0, :length_e-extra_e][0],
                   energies_indexed[0, :length_e-extra_e][-1]]
        i_times = [indices_indexed[0, :length_i-extra_i:approx_ratio][0],
                   indices_indexed[0, :length_i-extra_i:approx_ratio][-1]]
    else:
        print('length of energy file is {}'.format(length_e))
        print('length of index file is {}'.format(length_i))
        raise ImportError('Not sure how to handle those values')
    # todo maybe error check is a more pythonic manner with a try/except loop
    # if length_e != length_i:
    #     print("length of energies, {}, != length of indices, "
    #           "{}!".format(length_e, length_i))
    #     raise IndexError('lengths not equals')
    print('energies start: {}; end: {}'.format(e_times[0], e_times[1]))
    print('indices start: {}; end: {}'.format(i_times[0], i_times[1]))
    print('These values should be about the same if this is working properly')
    return deconvolved_energies


def plot_array(array, index_offset=0, num_replicas=False, n_rows=False, n_cols=False):
    """plot_array(array, index_offset=0, num_replicas=16, n_rows=False, n_cols=False)
    will put each column of array in a different axes of a figure and then return
    the figure."""
    if not num_replicas:
        num_replicas = array.shape[0] - index_offset
    from math import sqrt, ceil
    if n_rows == n_cols == False:
        n_rows = int(ceil(sqrt(float(num_replicas))))
        n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(num_replicas):
        ax = axes.flat[i]
        ax.plot(array[i+index_offset])
    return fig


def hist_array(array, index_offset=0, num_replicas=False, n_rows=False, n_cols=False,
               n_bins=10):
    """hist_array(array, index_offset=0, num_replicas=16, n_rows=False, n_cols=False,
    n_bins=10) will put each column of array in a different axes of a figure and
    then return the figure."""
    if not num_replicas:
        num_replicas = array.shape[0] - index_offset
    from math import sqrt, ceil
    if n_rows == n_cols == False:
        n_rows = int(ceil(sqrt(float(num_replicas))))
        n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(num_replicas):
        ax = axes.flat[i]
        ax.hist(array[i+index_offset], n_bins)
    return fig


def solute_trr(trr_base_name='npt_PT_out', tpr_base_name='TOPO/npt',
               output_base_name='solute', index='index.ndx', demux=True):
    """solute_trr takes file base names as input, creates a separate trr file for each
    trajectory that only includes the solutes, and then returns a list of the names of
    the created files.
    If demux=True, it will first use trjcat to deconvolve the walker trajectories (to
    get continuous coordinate files as opposed to continuous temperature).
    This uses gromacswrapper to call trjconv and possibly trjcat."""
    trr_files = glob.glob(trr_base_name + '*.trr')
    trr_files.sort()
    trr_files.sort(key=len)
    tpr_files = glob.glob(tpr_base_name + '*.tpr')
    tpr_files.sort()
    tpr_files.sort(key=len)
    output_files = []
    # This seems to be quite terribly slow as it is implemented.
    # The only ways I can think of trying to make it faster are rewriting this whole
    # thing myself, which would be very slow, or possibly something else I can't
    # think of now.
    if demux:
        for (i, trr_name) in enumerate(trr_files):
            number_match = re.search('(?:'+trr_base_name+')(\d+)(?:\.trr)', trr_name)
            number = number_match.group(1)
            out_file = output_base_name + number + '.trr'
            output_files.append(out_file)
            gromacs.tools.Trjconv(s=tpr_files[i], f=trr_name, o=out_file,
                                  n=index, input='CHR')()
        d_trr_base_name = 'deconv' + trr_base_name
        prev_deconv_files = glob.glob(d_trr_base_name+'*.trr')
        if len(prev_deconv_files) == len(trr_files):
            print('Likely already deconvolved trajectories, skipping that step')
        else:
            gromacs.tools.Trjcat(f=trr_files, o='demuxed.trr',
                                 demux='replica_index.xvg')()
            trr_files = glob.glob('*demuxed.trr')
            trr_files.sort()
            trr_files.sort(key=len)
            for (i, trr_file) in enumerate(trr_files):
                number = trr_file.split('_')[0]
                new_name = d_trr_base_name + number + '.trr'
                os.rename(trr_file, new_name)
                trr_files[i] = new_name
        trr_base_name = d_trr_base_name
    if len(trr_files) != len(tpr_files):
        raise IndexError('Number of trr and tpr files not equal: '
                         '{} and {}'.format(len(trr_files), len(tpr_files)))
    for (i, trr_name) in enumerate(trr_files):
        number_match = re.search('(?:'+trr_base_name+')(\d+)(?:\.trr)', trr_name)
        number = number_match.group(1)
        out_file = output_base_name + number + '.trr'
        output_files.append(out_file)
        gromacs.tools.Trjconv(s=tpr_files[i], pbc='mol', f=trr_name, o=out_file,
                              n=index, center=True, input=('CHR', 'CHR'))()
    return output_files
