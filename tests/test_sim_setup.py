"""This contains a set of tests for ParaTemp.sim_setup"""

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

import re
import pytest


def test_job_info_from_qsub():
    from ..paratemp.sim_setup import _job_info_from_qsub
    job_info = _job_info_from_qsub('Your job 2306551 ("PT-NTD-CG") '
                                   'has been submitted')
    assert job_info == ('2306551', 'PT-NTD-CG', '2306551 ("PT-NTD-CG")')


class TestUpdateNum(object):
    @pytest.fixture
    def match_10(self):
        return re.search(r'([,=])(\d+)', '=10')

    @pytest.fixture
    def match_text(self):
        return re.search(r'([,=])(\w+)', '=text')

    @pytest.fixture
    def match_float(self):
        return re.search(r'([,=])(\d+\.\d+)', '=2.1')

    @pytest.fixture
    def match_bad_few(self):
        return re.search(r'([,=])', '=10')

    @pytest.fixture
    def match_bad_many(self):
        return re.search(r'([,=])(\d+)(\d+)', '=10')

    @pytest.fixture
    def rd1030(self):
        return {10: 30}

    def test_update_num(self, match_10, rd1030):
        from ..paratemp.sim_setup import _update_num
        assert '=30' == _update_num(match_10, shift=10, cat_repl_dict=rd1030)
        assert '=1' == _update_num(match_10, shift=9)
        assert '=1' == _update_num(match_10, shift=9, cat_repl_dict=rd1030)

    def test_update_num_raises(self, match_10, match_text, match_float,
                               match_bad_few, match_bad_many):
        from ..paratemp.sim_setup import _update_num
        with pytest.raises(KeyError):
            _update_num(match_10, shift=10, cat_repl_dict=dict())
        with pytest.raises(ValueError,
                           match='cannot be converted to a valid int'):
            _update_num(match_text)
        with pytest.raises(ValueError,
                           match='cannot be converted to a valid int'):
            _update_num(match_float)
        with pytest.raises(ValueError, match='unpack'):
            _update_num(match_bad_few)
        with pytest.raises(ValueError, match='too many.*unpack'):
            _update_num(match_bad_many)
