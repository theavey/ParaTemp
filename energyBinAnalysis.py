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
    # TODO add option to only plot some?
    # TODO add option to plot multiple energy components either
    # separately or together
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
    fig.text(0.035, 0.62, 'energy / (kJ / mol)', ha='center',
             rotation='vertical')
    if save:
        fig.savefig(save_base_name+component+save_format)
    if display:
        return fig
    else:
        return None


def select_open_closed_energies(panel, set_open, set_closed,
                                frame_index=15):
    """Select the energies for open vs. closed TADDOL configurations"""
    df = panel[frame_index]
    from pandas import merge
    energies_open = merge(df, set_open, on='Time', how='inner')
    energies_closed = merge(df, set_closed, on='Time', how='inner')
    return energies_open, energies_closed


def make_hist_o_v_c_energy_components(eners_open, eners_closed,
                                      save=False,
                                      save_format='.pdf',
                                      save_base_name='o_v_c_hist_',
                                      display=True
                                      ):
    """Hist the energy components for open v closed for 1 replica"""
    e_columns = eners_closed.columns[1:16]
    e_c_figs = []
    for col in e_columns:
        from matplotlib.pyplot import subplots
        fig, ax = subplots()
        mean_open = eners_open[col].mean()
        mean_closed = eners_closed[col].mean()
        n_open, bins, patches = ax.hist(eners_open[col],
                                        normed=True, label='open',
                                        facecolor='white')
        n_closed, bins, patches = ax.hist(eners_closed[col],
                                          normed=True, label='closed',
                                          facecolor='grey')
        max_n = 1.2 * max(max(n_open), max(n_closed))
        ax.plot((mean_open, mean_open), (0, max_n), 'k--')
        ax.plot((mean_closed, mean_closed), (0, max_n), 'k-')
        ax.set_ylim([0, max_n])
        ax.legend()
        ax.set_title(col)
        e_c_figs.append(fig)
        if save:
            fig.savefig(save_base_name+col+save_format)
    if display:
        return e_c_figs
    else:
        return None
