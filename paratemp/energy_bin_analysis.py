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


import glob
import pandas as pd
import panedr
import re


def get_energies(in_base_name: str = 'npt_PT_out') -> pd.Panel:
    """Import the energies of GROMACS REMD trajectories.

    :param in_base_name: The base name for the output energy files
    :return: The Panel of all the time-step energies
    :rtype: pd.Panel
    """
    in_files = glob.glob(in_base_name+'*.edr')
    in_files.sort()
    in_files.sort(key=len)
    dfs = dict()
    for edr_file in in_files:
        try:
            number = int(re.match(r'.+?(\d+)\.edr', edr_file).group(1))
        except AttributeError:
            raise ValueError('Unable to parse edr file name '
                             '"{}"'.format(edr_file))
        df = panedr.edr_to_df(edr_file)
        dfs[number] = df
    return pd.Panel(dfs).rename_axis('replica')


def make_energy_component_plots(panel, component, save=False,
                                save_format='.png',
                                save_base_name='energy_component_',
                                display=True):
    """Plot an energy component from a Panel of energy DataFrames.

    :param panel:
    :param component:
    :param save:
    :param save_format:
    :param save_base_name:
    :param display:
    :return:
    """
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
    fig.tight_layout()
    if save:
        fig.savefig(save_base_name+component+save_format)
    if display:
        return fig
    else:
        return None


def select_open_closed_energies(panel, set_open, set_closed,
                                frame_index=15):
    """Select the energies for open vs. closed TADDOL configurations.

    :param panel: a pd.Panel returned from get_energies
    :param set_open:
    :param set_closed:
    :param frame_index:
    :return:
    """
    df = panel[frame_index]
    from pandas import merge
    energies_open = merge(df, set_open, on='Time', how='inner')
    energies_closed = merge(df, set_closed, on='Time', how='inner')
    return energies_open, energies_closed


def make_hist_o_v_c_energy_components(eners_open, eners_closed,
                                      save=False,
                                      save_format='.pdf',
                                      save_base_name='o_v_c_hist_',
                                      display=True,
                                      subplot=False
                                      ):
    """Hist the energy components for open v closed for 1 replica.

    :param eners_open:
    :param eners_closed:
    :param save:
    :param save_format:
    :param save_base_name:
    :param display:
    :param subplot:
    :return:
    """
    e_columns = eners_closed.columns[1:16]
    from matplotlib.pyplot import subplots
    fig, axes = subplots(nrows=5, ncols=3, figsize=(17, 22),
                         gridspec_kw={'left': None, 'right': None,
                                      'top': None, 'bottom': None})
    e_c_figs = []
    for i, col in enumerate(e_columns):
        if subplot:
            ax = axes.flat[i]
        else:
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
        if not subplot:
            e_c_figs.append(fig)
            if save:
                fig.tight_layout()
                fig.savefig(save_base_name+col+save_format)
    if save and subplot:
        fig.tight_layout()
        fig.savefig(save_base_name+save_format)
    if display:
        if subplot:
            fig.tight_layout()
            return fig
        else:
            return e_c_figs
    else:
        return None


def deconvolve_energies(energies_panel, index='replica_temp.xvg'):
    """Return the energies of walkers from REMD simulations.

    This assumes a near-integer ratio of number of energies to indexes
    or near-integer inverse of that. If it's like 3/2 or 5/3 (either
    way) it won't throw an error, but also won't give meaningful
    results.
    :param energies_panel:
    :param index:
    :return:
    """
    from gromacs.fileformats import XVG
    indexer = XVG(filename=index).array
    i_all_times = indexer[0]
    indexer = indexer[1:].astype(int)
    # Assuming all replicas have the same times, though I don't know
    # why it would be otherwise.
    from numpy import array
    e_all_times = array(energies_panel[0]['Time'])
    e_len = len(e_all_times)
    i_len = len(i_all_times)
    ratio = float(e_len) / float(i_len)
    approx_ratio = int(round(ratio))

    if ratio > 1:
        from numpy import repeat
        indexer = repeat(indexer, approx_ratio, axis=1)
        i_all_times = repeat(i_all_times, approx_ratio, axis=0)
        i_len = len(i_all_times)
        ratio = float(e_len) / float(i_len)
        approx_ratio = int(round(ratio))

    from numpy import mod
    if ratio == 1.0:
        e_end = i_end = e_len
        e_freq = i_freq = 1

    elif ratio > 1:
        e_freq = approx_ratio
        i_freq = 1
        if approx_ratio == ratio:
            e_end = e_len
            i_end = i_len
        elif approx_ratio > ratio:
            # Note: because these are ints, it's essentially already
            # using a floor function.
            i_end = e_len / approx_ratio
            e_end = e_len - mod(e_len, i_end)
        elif approx_ratio < ratio:
            e_end = e_len - mod(e_len, i_len)
            i_end = i_len
        else:
            raise ImportError('ratio: {}, '.format(ratio) +
                              'approx ratio: {}'.format(approx_ratio))

    elif ratio < 1:
        print('likely undersampling energies because energy / indices '
              'ratio is {}'.format(ratio))
        ratio = 1 / ratio
        approx_ratio = int(round(ratio))
        e_freq = 1
        i_freq = approx_ratio
        if approx_ratio == ratio:
            e_end = e_len
            i_end = i_len
        elif approx_ratio > ratio:
            e_end = i_len / approx_ratio
            i_end = i_len - mod(i_len, e_end)
        elif approx_ratio < ratio:
            e_end = e_len
            i_end = i_len - mod(i_len, e_len)
        else:
            raise ImportError('ratio: {}, '.format(ratio) +
                              'approx ratio: {}'.format(approx_ratio))

    else:
        print('length of energy file is {}'.format(e_len))
        print('length of index file is {}'.format(i_len))
        raise ImportError('Not sure how to handle those values')

    e_times = (e_all_times[0],
               e_all_times[:e_end:e_freq][-1])
    i_times = (i_all_times[0],
               i_all_times[:i_end:i_freq][-1])
    if not (float(e_times[0]) == float(i_times[0]) and
            float(e_times[1]) == float(i_times[1])):
        print('energies start: {}; end: {}'.format(e_times[0],
                                                   e_times[1]))
        print('indices start: {}; end: {}'.format(i_times[0],
                                                  i_times[1]))
        print('These values should be about the same if this is working'
              ' properly')

    from numpy import arange
    energies_array = array(energies_panel)[:, :e_end:e_freq][
        indexer[:, :i_end:i_freq], arange(i_end/i_freq)]
    from pandas import Panel
    # todo remove this deprecated panel def below
    return Panel(energies_array,
                 items=energies_panel.items.set_names('walker'),
                 major_axis=energies_panel.major_axis[:e_end:e_freq],
                 minor_axis=energies_panel.minor_axis)


def plot_convergence():
    # todo define this function w/ doc string
    pass
