"""This contains a set of tests for plotting functions"""

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


import numpy as np


def test_fes_1d_data_str(ref_a_dists, ref_delta_g, ref_bins):
    """
    :type ref_a_dists: pandas.DataFrame
    :type ref_delta_g: np.ndarray
    :type ref_bins: np.ndarray
    """
    from paratemp.plotting import fes_1d
    delta_g_str, bins_str, lines_str, fig_str, ax_str = \
        fes_1d('a', temp=205., data=ref_a_dists)
    assert np.allclose(delta_g_str, ref_delta_g)
    assert np.allclose(bins_str, ref_bins)


def test_fes_1d_data_data(ref_a_dists, ref_delta_g, ref_bins):
    """
    :type ref_a_dists: pandas.DataFrame
    :type ref_delta_g: np.ndarray
    :type ref_bins: np.ndarray
    """
    from paratemp.plotting import fes_1d
    delta_g_data, bins_data, lines_data, fig_data, ax_data = \
        fes_1d(ref_a_dists['a'], temp=205.)
    assert np.allclose(delta_g_data, ref_delta_g)
    assert np.allclose(bins_data, ref_bins)
