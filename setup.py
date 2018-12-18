from setuptools import setup
import versioneer

setup(
    name='ParaTemp',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['paratemp'],
    scripts=['scripts/check-convergence.py'],
    url='https://github.com/theavey/ParaTemp',
    license='Apache License 2.0',
    author='Thomas Heavey',
    author_email='thomasjheavey@gmail.com',
    description='Scripts for molecular dynamics analysis and parallel '
                'tempering in GROMACS',
    install_requires=[
        'MDAnalysis>=0.17.0',
        'pandas',
        'numpy',
        'matplotlib',
        'panedr',
        'gromacswrapper>=0.7',
        'tables',
        'typing',
        'scipy',
        'six',
        'py',
    ],
    extras_require={
            'docs': [
                'sphinx',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'coveralls',
                'py',
            ],
    },
    tests_require=[
        'pytest>=3.9',
        'pytest-cov',
        'coveralls',
        'py',
    ],
    classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
    ],
    zip_safe=True,
)
