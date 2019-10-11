"""
fixtures and setup for testing the sim_setup subpackage

"""

import pytest


@pytest.fixture
def molecule():
    from paratemp.sim_setup import molecule

