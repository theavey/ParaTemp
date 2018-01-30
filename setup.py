from setuptools import setup
import versioneer

setup(
    name='ParaTemp',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['paratemp'],
    scripts=['check-convergence.py'],
    url='https://github.com/theavey/ParaTemp',
    license='Apache License 2.0',
    author='Thomas Heavey',
    author_email='thomasjheavey@gmail.com',
    description='Scripts for molecular dynamics analysis and parallel '
                'tempering in GROMACS'
)
