"""This contains a set of tests for paratemp.exceptions"""

########################################################################
#                                                                      #
# This script was written by Thomas Heavey in 2018.                    #
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

import pytest


def test_input_error():
    from paratemp.exceptions import InputError
    with pytest.raises(InputError):
        e = InputError('bad input', 'message about it')
        print(e)
        raise e


def test_file_changed_error():
    from paratemp.exceptions import FileChangedError
    with pytest.raises(FileChangedError):
        e = FileChangedError('this is an extra message')
        print(e)
        raise e


def test_unknown_energy_error():
    from paratemp.exceptions import UnknownEnergyError
    with pytest.raises(UnknownEnergyError):
        e = UnknownEnergyError()
        print(e)
        raise e
    with pytest.raises(UnknownEnergyError):
        e = UnknownEnergyError('some other message')
        print(e)
        raise e
