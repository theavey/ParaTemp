#! /usr/bin/env python

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


def get_energies(in_base_name='npt_PT_out'):
    """Import the energies of GROMACS REMD trajectories"""
    from panedr import edr_to_df
    from glob import glob
    from re import match
    from pandas import Panel
    in_files = glob(in_base_name+'*.edr')
    in_files.sort()
    in_files.sort(key=len)
    dfs = {}
    for edr_file in in_files:
        number = int(match('\w+?(\d+)\.edr', edr_file).group(1))
        df = edr_to_df(edr_file)
        dfs[number] = df
    return Panel(dfs)


def make_energy_component_plots(panel, component, save=False,
                                save_format='.png',
                                save_base_name='energy_component',
                                display=True):
    """Plot an energy component from a Panel of energy DataFrames"""
    num_traj = len(panel)
    from math import sqrt, ceil
    n_rows = int(ceil(sqrt(float(num_traj))))
    n_cols = n_rows
    from matplotlib.pyplot import subplots
    fig, axes = subplots(ncols=n_cols, nrows=n_rows, sharex=True,
                         sharey=True)
    for i in range(num_traj):
        ax = axes.flat[i]
        ax.plot(panel[i][component])
    [ax.get_xaxis().set_ticks([]) for ax in fig.axes]
    fig.text(0.513, 0.08, 'time', ha='center')
    # These y-axis units are right for (all?) the energy components,
    # but not the pressures and such also available.
    fig.text(0.05, 0.585, 'energy / (kJ / mol)', ha='center',
             rotation='vertical')
    if save:
        fig.savefig(save_base_name+component+save_format)
    if display:
        return fig
    else:
        return None
