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

# This file defines exceptions that are useful for the ParaTemp package


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

    def __str__(self):
        output = 'Incorrect input "{}". {}'.format(self.expr, self.msg)
        return repr(output)


class FileChangedError(Exception):
    """
    Exception raised for error when file changed between initialization
    and when it is read from disk again later.

    """

    def __init__(self, msg=None):
        self.msg = msg

    def __str__(self):
        output = ('Number of frames in trajectory does not match the '
                  'number when this was initialized.\nTry '
                  'reinitializing the object or rewrite the code to '
                  'deal with this in a smarter way.')
        if self.msg is not None:
            output += '\n' + self.msg
            return repr(output)


class UnknownEnergyError(Exception):
    """
    Exception raised when an object does not know it's energy, but it's queried

    """

    def __init__(self, msg=None):
        self.msg = msg

    def __str__(self):
        standard_response = ("The energy is unknown either because it wasn't "
                             "in the original file or the coordinates have "
                             "changed.\nCould try XYZ.original_energy")
        if self.msg is None:
            return repr(standard_response)
        else:
            return repr(self.msg)
