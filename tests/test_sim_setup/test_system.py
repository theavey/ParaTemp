"""
tests for the system module

"""

import pathlib

import pytest

from paratemp import cd


class TestSystem(object):

    @pytest.mark.parametrize('shift', [True, False])
    @pytest.mark.parametrize('box_length', [0, 0.0, 1, None])
    @pytest.mark.parametrize('include_gbsa', [True, False])
    @pytest.mark.parametrize('n_args', [1, 2])
    def test_init(self, molecule_w_params, shift, box_length,
                  include_gbsa, n_args):
        from paratemp.sim_setup import System
        mol, tmp_path = molecule_w_params
        with cd(tmp_path):
            system = System(*([mol]*n_args),
                            shift=shift,
                            box_length=box_length,
                            include_gbsa=include_gbsa)
            assert isinstance(system, System)
            assert system.n_molecules == n_args
            assert (hasattr(system, 'directory') and
                    isinstance(system.directory, pathlib.Path))
            assert (hasattr(system, 'name') and
                    isinstance(system.name, str))
            assert repr(system) == ('<default System from {} '
                                    'Molecules>'.format(n_args))
