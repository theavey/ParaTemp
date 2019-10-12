"""
fixtures and setup for testing the sim_setup subpackage

"""

import pytest


@pytest.fixture(scope='function')  # the created directories should be unique
def molecule(path_test_data, tmp_path):
    from paratemp import cd
    from paratemp.sim_setup import Molecule
    path_gro = path_test_data / 'water.mol2'
    # Note: this instantiation will make a new directory!
    with cd(tmp_path):
        mol = Molecule(path_gro)
    return mol, tmp_path


@pytest.fixture
def molecule_w_params(molecule):
    mol, tmp_path = molecule
    mol.parameterize()
    return mol, tmp_path
