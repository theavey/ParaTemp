"""This contains a set of tests for ParaTemp.para_temp_setup"""

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

from __future__ import absolute_import

import pytest


def test_job_info_from_qsub():
    from ..paratemp.para_temp_setup import _job_info_from_qsub
    job_info = _job_info_from_qsub('Your job 2306551 ("PT-NTD-CG") '
                                   'has been submitted')
    assert job_info == ('2306551', 'PT-NTD-CG', '2306551 ("PT-NTD-CG")')
