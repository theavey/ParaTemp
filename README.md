[![Build Status](https://travis-ci.org/theavey/ParaTemp.svg?branch=master)](https://travis-ci.org/theavey/ParaTemp)
[![Coverage Status](https://coveralls.io/repos/github/theavey/ParaTemp/badge.svg?branch=master)](https://coveralls.io/github/theavey/ParaTemp?branch=master)
[![Code Health](https://landscape.io/github/theavey/ParaTemp/master/landscape.svg?style=flat)](https://landscape.io/github/theavey/ParaTemp/master)
[![DOI](https://zenodo.org/badge/64339257.svg)](https://zenodo.org/badge/latestdoi/64339257)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat)](https://github.com/theavey/ParaTemp/blob/master/LICENSE)

[![ParaTemp Logo](https://raw.githubusercontent.com/theavey/ParaTemp/master/graphics/logo.png)](https://github.com/theavey/ParaTemp)

__*A Python package for setup and analysis of parallel tempering and replica exchange molecular dynamics*__

This is a package with many tools, functions, and some classes for
analyzing and running molecular dynamics trajectories.
It is specifically designed for working with Replica Exchange Molecular
Dynamics (Parallel Tempering, when the replicas are at different
temperatures).
It is, in-part, specialized for analyzing trajectories of TADDOL-catalyzed
reactions, though most components are general.


## Installing

To install, run:
```
git clone https://github.com/theavey/ParaTemp.git
cd ParaTemp
pip install -r requirements.txt
python setup.py install
```
or this should work using conda (based on [this gist](
https://gist.github.com/luiscape/19d2d73a8c7b59411a2fb73a697f5ed4)):
```
git clone https://github.com/theavey/ParaTemp.git
cd ParaTemp
conda install --yes --file requirements.txt
python setup.py install
```


## Notes

All simulations have so far been from GROMACS, but with the powerful
generality of [MDAnalysis](https://www.mdanalysis.org/), that should not
be a particular constraint for using the analysis components of this package.

This package depends on MDAnalysis, NumPy, pandas, panedr, and
gromacswrapper.
The dependencies are listed in [requirements.txt](./requirements.txt).
Most if not all should be installable with conda.

With recent updates to MDAnalysis, this should now be compatible with Python
2.7 and Python 3.4+.


