########################################################################
#                                                                      #
# This package was written by Thomas Heavey in 2018.                   #
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

from .tools import copy_no_overwrite, cd, get_temperatures
from . import coordinate_analysis
from .coordinate_analysis import Universe
from . import re_universe
from .re_universe import REUniverse
from ._version import get_versions
from . import plotting

del absolute_import


__version__ = get_versions()["version"]
del get_versions

__author__ = ["Thomas Heavey", "Jacob Marasco"]
