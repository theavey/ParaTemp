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
    dfs = {}
    for edr_file in in_files:
        number = match('\w+?(\d+)\.edr', edr_file).group(1)
        df = edr_to_df(edr_file)
        dfs[number] = df
    return Panel(dfs)
