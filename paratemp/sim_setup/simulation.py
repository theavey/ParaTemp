"""This file defines a class useful for setting up simulations"""

########################################################################
#                                                                      #
# This test was written by Thomas Heavey in 2019.                      #
#        theavey@bu.edu     thomasjheavey@gmail.com                    #
#                                                                      #
# Copyright 2017-19 Thomas J. Heavey IV                                #
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

from collections import OrderedDict
import errno
import logging
import pathlib
import pickle
import re
import sys
import typing

import gromacs
import pkg_resources

from .molecule import Molecule
from .system import System
from ..tools import cd


__all__ = ['Simulation', 'SimpleSimulation']


log = logging.getLogger(__name__)
if not log.hasHandlers():
    level = logging.INFO
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


GenPath = typing.Union[pathlib.Path, str]


if sys.version_info >= (3, 6):
    def resolve_path(path):
        return pathlib.Path(path).resolve()
else:
    def resolve_path(path):
        try:
            return pathlib.Path(path).resolve()
        except FileNotFoundError:
            pass
        path = pathlib.Path(path)
        if path.is_absolute():
            return path
        return pathlib.Path.cwd().joinpath(path)


class Simulation(object):
    """
    A class for setting up and running GROMACS simulations
    """

    _fp = staticmethod(resolve_path)

    def __init__(self, name: str, gro: GenPath,
                 top: GenPath, base_folder: GenPath = '.',
                 mdps: dict = None):
        self.name = name
        self.top = self._fp(top)
        self.geometries = OrderedDict(initial=self._fp(gro))
        self.base_folder = self._fp(base_folder)
        self.directories = dict(base=self.base_folder)
        self.mdps = dict() if mdps is None else mdps
        self.tprs = dict()
        self.deffnms = dict()
        self.outputs = dict()
        for mdp in self.mdps:
            setattr(self, mdp, self._make_step_method(mdp))
            self.mdps[mdp] = self._fp(mdps[mdp])

    @property
    def last_geometry(self) -> pathlib.Path:
        """
        The path to the output geometry from the most recent simulation

        :return: The path to the last output geometry
        """
        return next(reversed(self.geometries.items()))[1]

    @property
    def _next_folder_index(self) -> int:
        """
        Index for next folder to be created

        Note, this will not work if there are 99 or more folders.
        Folders should be of the form '01-minimize-benzene'
        :return: next folder index
        :rtype: int
        """
        folders = [d.name for d in self.base_folder.iterdir() if d.is_dir()]
        nums = [int(d[:2]) for d in folders if re.match(r'\d{2}-\w+-\w+', d)]
        nums.sort()
        return nums[-1]+1 if nums else 1

    def _make_step_method(self, step_name: str) -> typing.Callable:
        """
        Make a function that runs a GROMACS "step" (minimization, equil, etc.)

        :param str step_name: Name of the step. This should be a valid key to
            `mdps` dict and will be the name of the method to which this
            function is mapped.
        :return: A function to run the step specified by the mdp
        :rtype: typing.Callable
        """
        def func(geometry=None):
            geometry = self.last_geometry if geometry is None else geometry
            folder_index = self._next_folder_index
            folder = self.base_folder / '{:0>2}-{}-{}'.format(folder_index,
                                                              step_name,
                                                              self.name)
            folder.mkdir()
            self.directories[step_name] = folder
            with cd(folder):
                tpr = self._compile_tpr(step_name, geometry)
                self._run_mdrun(step_name, tpr)
            return folder
        return func

    def _compile_tpr(self, step_name: str,
                     geometry: GenPath = None,
                     trajectory: GenPath = None
                     ) -> pathlib.Path:
        """
        Make a tpr file for the chosen step_name and associated mdp file

        :param step_name: Key for the mdp file from the dict mdps
        :param geometry: Path to the geometry to be used as input. If None,
            :attr:`last_geometry` will be used.
        :param trajectory: Path to a trajectory file from which to take the
            input geometry. This is useful when a full precision geometry is
            needed as input and a trr file can be used. If None,
            no trajectory will be given to grompp.
        :return: The Path to the tpr file
        """
        geometry = self.last_geometry if geometry is None else geometry
        tpr = '{}-{}.tpr'.format(self.name, step_name)
        p_tpr = self._fp(tpr)
        self.tprs[step_name] = p_tpr
        if hasattr(gromacs, 'grompp'):
            grompp_func = gromacs.grompp
        elif hasattr(gromacs, 'grompp_mpi'):
            grompp_func = gromacs.grompp_mpi
        else:
            raise OSError(errno.ENOENT, 'Could not find grompp executable '
                                        'using gromacswrapper package')
        rc, output, junk = grompp_func(c=geometry,
                                       p=self.top,
                                       f=self.mdps[step_name],
                                       o=tpr,
                                       t=trajectory,
                                       stdout=False)
        # Doesn't capture output if failed?
        self.outputs['compile_{}'.format(step_name)] = output
        return p_tpr

    def _run_mdrun(self, step_name: str, tpr: GenPath = None
                   ) -> pathlib.Path:
        """
        Run mdrun with the given step_name or explicitly given tpr file.

        :param step_name: The name of this step
        :param tpr: Path to the tpr file. If None, the tpr will be found
            from the dict :attr:`tprs` with the key being `step_name`
        :return: The Path to the output geometry
        """
        tpr = self.tprs[step_name] if tpr is None else tpr
        deffnm = '{}-{}-out'.format(self.name, step_name)
        p_deffnm = self._fp(deffnm)
        self.deffnms[step_name] = p_deffnm
        if hasattr(gromacs, 'mdrun'):
            mdrun_func = gromacs.mdrun
        elif hasattr(gromacs, 'mdrun_mpi'):
            mdrun_func = gromacs.mdrun_mpi
        else:
            raise OSError(errno.ENOENT, 'Could not find mdrun executable '
                                        'using gromacswrapper package')
        rc, output, junk = mdrun_func(s=tpr, deffnm=deffnm, stdout=False)
        # Doesn't capture output if failed?
        self.outputs['run_{}'.format(step_name)] = output
        gro = p_deffnm.with_suffix('.gro')
        self.geometries[step_name] = gro
        return gro


_type_mol_inputs = typing.Union[str, typing.List[typing.Union[dict,
                                                              Molecule]]]


def get_mdps_folder() -> pathlib.Path:
    directory = pkg_resources.resource_filename(
        __name__, 'SimpleSim_data/mdps')
    return pathlib.Path(directory)


class SimpleSimulation(object):
    """
    SimpleSimulation can be used to easily setup a Simulation with many defaults
    """

    _path_mdps_dir = get_mdps_folder()
    _default_mdps_gbsa = {'minimize':
                              str(_path_mdps_dir / 'minim-gbsa.mdp'),
                          'equilibrate':
                              str(_path_mdps_dir / 'equil-gbsa.mdp'),
                          'production':
                              str(_path_mdps_dir / 'production-gbsa.mdp')}
    _default_mdps_rf = {'minimize':
                            str(_path_mdps_dir / 'minim-rf.mdp'),
                        'equilibrate':
                            str(_path_mdps_dir / 'equil-rf.mdp'),
                        'production':
                            str(_path_mdps_dir / 'production-rf.mdp')}

    def __init__(self, name: str,
                 mol_inputs: _type_mol_inputs = 'ask',
                 solvent_dielectric: float = 9.1  # DCM
                 ):
        log.info('Instantiating a SimpleSimulation named {}'.format(name))
        self.name = name
        self.molecules = list()  # type: typing.List[Molecule]
        self.directories = dict()  # type: typing.Dict[str, pathlib.Path]
        self._process_mol_inputs(mol_inputs)
        self.n_molecules = len(self.molecules)
        self._dielectric = solvent_dielectric
        self._steps = dict(parameterized=False,
                           combined=False,
                           simulation_created=False)
        self.system = None  # type: System
        self._SimClass = Simulation
        self.simulation = None  # type: Simulation

    def _process_mol_inputs(self, mol_inputs):
        if mol_inputs == 'ask':
            more = True
            while more:
                self.molecules.append(Molecule.assisted())
                more = (True if 'y' in input('Any more molecules? [yn]').lower()
                        else False)
        elif isinstance(mol_inputs, typing.Sequence):
            if isinstance(mol_inputs[0], Molecule):
                self.molecules = mol_inputs
            else:
                for mol in mol_inputs:
                    self.molecules.append(Molecule.from_make_mol_inputs(mol))
        elif isinstance(mol_inputs, Molecule):
            self.molecules = [mol_inputs]
        else:
            try:
                self.molecules = Molecule.from_make_mol_inputs(mol_inputs)
            except KeyError:  # maybe other Errors?
                raise ValueError('Unrecognized input: {}'.format(mol_inputs))
        dirs = {'molecule_{}'.format(mol.name): mol.directory for mol
                in self.molecules}
        self.directories.update(dirs)

    def parameterize(self):
        """
        Parameterize all Molecules in this SimpleSimulation

        :return: None
        """
        log.info('Parameterizing the {} Molecules'.format(len(self.molecules)))
        # TODO optionally include position restraints?
        # was necessary before otherwise they just flew apart
        # low dielectric might make it less necessary
        # especially if they're oppositely charged, but not regularly the case
        for mol in self.molecules:
            mol.parameterize()
        self._steps['parameterized'] = True

    def combine(self, box_length: float = None, include_gbsa: bool = False):
        """
        Combine all molecules into a given System

        :param box_length: side length of the periodic cube to use
        :param include_gbsa: If True, GBSA parameters will be added to the
        topology file
        :return: None
        """
        log.info('Combining the {} Molecules into a single System'.format(
            len(self.molecules)))
        if box_length is not None:
            d_box_length = {'box_length': box_length}
        else:
            d_box_length = dict()
        self.system = System(*self.molecules,
                             name=self.name,
                             shift=True,
                             spacing=2.0,
                             include_gbsa=include_gbsa,
                             **d_box_length)
        self.directories['system'] = self.system.directory
        self._steps['combined'] = True

    def make_simulation(self, solvent_model: str = 'rf',
                        mdps: dict = None):
        """
        Make a Simulation object from the System

        :param str solvent_model: type of solvent model to use. This currently
            knows how to handle 'rf' for reaction field or 'gbsa' for the
            Generalized Born Solvent model (crashes tested versions of
            GROMACS > ~5.0).
            This will determine which default set of mdp files to use.
        :param dict mdps: dict of step names to strings of path to existing
            mdp files
        :return: None
        """
        log.info('Creating a Simulation object from the {} '
                 'System object'.format(self.system.name))
        self.directories['simulation_base'] = self.system.directory
        solvent_model_dict = {'rf': self._default_mdps_rf,
                              'gbsa': self._default_mdps_gbsa}
        _mdps = solvent_model_dict[solvent_model.lower()].copy()
        if mdps is not None:
            _mdps.update(mdps)
        _mdps = self._insert_dielectric(_mdps)
        self.simulation = self._SimClass(
            name=self.name,
            gro=self.system.gro_path,
            top=self.system.top_path,
            base_folder=self.system.directory,
            mdps=_mdps
        )
        self._steps['simulation_created'] = True

    def _insert_dielectric(self, mdps: dict) -> typing.Dict[str, str]:
        """
        Use Python format ({}) to put dielectric constant into given mdp files

        :param dict mdps: dict of step names to strings of path to existing
            mdp files
        :return: dict of step names to strings ot paths to edited mdp files
            (now in a folder specific to this simulation)
        """
        _dir = self.directories['simulation_base']
        d_out = dict()
        for key in mdps:
            old_path = pathlib.Path(mdps[key])
            new_path = _dir / old_path.name
            text = old_path.read_text()
            text = text.format(dielectric=self._dielectric)
            new_path.write_text(text)
            log.info('wrote {} mdp with dielectric replaced to {}'.format(
                key, new_path))
            d_out[key] = str(new_path)
        return d_out

    def save(self):
        path = pathlib.Path('{}.pkl'.format(self.name))
        if self._steps['simulation_created']:
            # This doesn't work...
            # TODO find a way around this (just don't save Sim?)
            raise AttributeError('Cannot save SimpleSimulation after making '
                                 'simulation')
        pickle.dump(self, path.open('wb'))
        log.info('Saved SimpleSimulation to {}'.format(path))

    @classmethod
    def load(cls, name: str):
        path = pathlib.Path('{}.pkl'.format(name))
        if not path.exists():
            raise FileNotFoundError('Could not find save file for this name: '
                                    '{}'.format(path))
        ssim = pickle.load(path.open('rb'))
        if not isinstance(ssim, cls):
            raise TypeError('The loaded pickle file ({}) is not of the '
                            'correct type: {}'.format(path, cls))
        return ssim

    def __repr__(self):
        return ('<{} SimpleSimulation with {} Molecules; params: {}; '
                'combined: {}, sim made: '
                '{}>'.format(self.name,
                             self.n_molecules,
                             self._steps['parameterized'],
                             self._steps['combined'],
                             self._steps['simulation_created']))

