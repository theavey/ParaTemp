"""This contains a set of tests for paratemp.sim_setup.Simulation"""

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

import pathlib
import pytest

from paratemp.tools import cd


class TestSimulation(object):

    @pytest.mark.xfail
    def test_runs(self, pt_blank_dir):
        from paratemp.sim_setup import Simulation
        gro = pt_blank_dir / 'PT-out0.gro'
        top = pt_blank_dir / 'spc-and-methanol.top'
        sim = Simulation(name='test_sim',
                         gro=gro, top=top, base_folder=pt_blank_dir)
        assert isinstance(sim, Simulation)

    min_mdp = 'examples/sample-mdps/minim.mdp'
    equil_mdp = 'examples/sample-mdps/equil.mdp'
    prod_mdp = 'examples/sample-mdps/prod.mdp'
    mdps = dict(minimize=min_mdp, equilibrate=equil_mdp,
                production=prod_mdp)

    @pytest.fixture
    def sim_with_dir(self, pt_blank_dir):
        from paratemp.sim_setup import Simulation
        gro = pt_blank_dir / 'PT-out0.gro'
        top = pt_blank_dir / 'spc-and-methanol.top'
        sim = Simulation(name='sim_fixture',
                         gro=gro, top=top, base_folder=pt_blank_dir,
                         mdps=self.mdps)
        return sim, pt_blank_dir

    @pytest.fixture
    def sim(self, sim_with_dir):
        return sim_with_dir[0]

    attrs = {'name': str,
             'top': pathlib.Path,
             'base_folder': pathlib.Path,
             'mdps': dict,
             'tprs': dict,
             'deffnms': dict,
             'outputs': dict,
             'geometries': dict,
             'folders': dict,
             }

    @pytest.mark.xfail
    def test_attrs_exist_and_type(self, sim):
        for attr in self.attrs:
            assert hasattr(sim, attr)
            dtype = self.attrs[attr]
            assert isinstance(getattr(sim, attr), dtype)

    @pytest.mark.xfail
    def test_methods_exist_and_callable(self, sim):
        for step in self.mdps:
            assert hasattr(sim, step)
            assert callable(getattr(sim, step))

    @pytest.mark.xfail
    def test_fp(self, sim):
        sample_file = 'tests/__init__.py'
        fp: pathlib.Path = sim._fp(sample_file)
        assert fp.exists()
        assert fp.is_absolute()
        assert fp.is_file()
        assert fp.samefile(sample_file)

    @pytest.mark.xfail
    def test_last_geom(self, sim):
        gro: pathlib.Path = sim.last_geometry
        assert isinstance(gro, pathlib.Path)
        assert gro.suffix == '.gro'
        assert gro.is_absolute()
        assert gro.is_file()

    @pytest.mark.xfail
    def test_compile_tpr(self, sim_with_dir):
        sim, path = sim_with_dir
        step = 'minimize'
        min_path: pathlib.Path = path / step
        min_path.mkdir()
        with cd(min_path):
            tpr = sim._compile_tpr(step_name=step)
            mdout = pathlib.Path('mdout.mdp').resolve()
        assert isinstance(tpr, pathlib.Path)
        assert tpr.exists()
        assert tpr.is_absolute()
        assert tpr.is_file()
        assert tpr.suffix == '.tpr'
        assert mdout.exists()
        d_tpr = sim.tprs[step]
        assert tpr.samefile(d_tpr)
        assert isinstance(sim.outputs['compile_{}'.format(step)], str)

    @pytest.fixture
    def sim_with_tpr(self, sim_with_dir):
        sim, path = sim_with_dir
        step = 'minimize'
        min_path: pathlib.Path = path / step
        min_path.mkdir()
        with cd(min_path):
            tpr = sim._compile_tpr(step_name=step)
        return sim, min_path, step

    @pytest.mark.xfail
    def test_run_mdrun(self, sim_with_tpr):
        sim, path, step = sim_with_tpr
        with cd(path):
            gro = sim._run_mdrun(step_name=step)
        assert isinstance(gro, pathlib.Path)
        assert gro.exists()
        assert gro.is_absolute()
        assert gro.is_file()
        assert gro.suffix == '.gro'
        assert gro.samefile(sim.last_geometry)
        assert isinstance(sim.deffnms[step], pathlib.Path)
        assert isinstance(sim.outputs['run_{}'.format(step)], str)

    @pytest.mark.xfail
    @pytest.mark.parametrize('step', list(mdps.keys()))
    def test_step_methods(self, sim, step):
        method = getattr(sim, step)
        step_dir = method()
        assert step in step_dir.name
        assert step_dir.exists()
        assert step_dir.is_dir()
        d_step_dir = sim.folders[step]
        assert isinstance(d_step_dir, pathlib.Path)
        assert step_dir.samefile(d_step_dir)
