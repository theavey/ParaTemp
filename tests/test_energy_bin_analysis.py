"""This contains a set of tests for paratemp.coordinate_analysis"""

########################################################################
#                                                                      #
# This test was written by Thomas Heavey in 2018.                      #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017-18 Thomas J. Heavey IV                                #
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

from __future__ import absolute_import

import pandas as pd
import pathlib
import pytest
from matplotlib.figure import Figure

from paratemp.tools import cd


def test_get_energies(pt_run_dir):
    # Doesn't currently test:
    #    content of the outputs
    from paratemp.energy_bin_analysis import get_energies
    with cd(pt_run_dir):
        mi_df = get_energies('PT-out')
    assert len(mi_df.index.levels[0]) == 2
    assert isinstance(mi_df, pd.DataFrame)
    assert isinstance(mi_df.index, pd.MultiIndex)


@pytest.fixture
def energies_df(pt_run_dir):
    from paratemp.energy_bin_analysis import get_energies
    with cd(pt_run_dir):
        mi_df = get_energies('PT-out')
    return mi_df


def test_make_energy_component_plots(energies_df):
    from paratemp.energy_bin_analysis import make_energy_component_plots
    fig = make_energy_component_plots(energies_df, 'Pressure', display=True)
    assert isinstance(fig, Figure)
    fig = make_energy_component_plots(energies_df, 'Pressure', display=False)
    assert fig is None


@pytest.fixture
def replica_temp_path(pt_run_dir: pathlib.PosixPath):
    # Doesn't currently test:
    #    content of the outputs
    #    what happens if they already exist
    from paratemp.energy_histo import make_indices
    with cd(pt_run_dir):
        make_indices('PT-out0.log')
        return pt_run_dir / 'replica_temp.xvg'


class TestDeconvolveEnergies(object):

    def test_function_runs(self, energies_df, replica_temp_path):
        from paratemp.energy_bin_analysis import deconvolve_energies
        df = deconvolve_energies(energies_df, index=str(replica_temp_path))
        assert isinstance(df, pd.DataFrame)
