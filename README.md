[![Build Status](https://travis-ci.org/theavey/ParaTemp.svg?branch=master)](https://travis-ci.org/theavey/ParaTemp)
[![Code Health](https://landscape.io/github/theavey/ParaTemp/master/landscape.svg?style=flat)](https://landscape.io/github/theavey/ParaTemp/master)
[![DOI](https://zenodo.org/badge/64339257.svg)](https://zenodo.org/badge/latestdoi/64339257)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat)](https://github.com/theavey/ParaTemp/blob/master/LICENSE)

This is a package with many tools, functions, and some classes for
analyzing and running molecular dynamics trajectories.
It is specifically designed for working with Replica Exchange Molecular
Dynamics (Parallel Tempering, when the replicas are at different
temperatures).
It is in part specialized for analyzing trajectories of TADDOL-catalyzed
reactions, though I hope to generalize it more.

All simulations have so far been from GROMACS, but with the powerful
generality of [MDAnalysis](https://www.mdanalysis.org/), that should not
be a particular constraint for using this software.

This package depends on MDAnalysis, NumPy, pandas, panedr, and
gromacswrapper.
The dependencies are listed in [requirements.txt](./requirements.txt).
Most if not all should be installable with conda.

To install, run:
```
git clone https://github.com/theavey/ParaTemp.git
cd ParaTemp
pip install -r requirements.txt
python setup.py install
```
