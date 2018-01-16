"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import subprocess
import re

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

cp = subprocess.run(
    ["git", "describe", "--always"],
    stdout=subprocess.PIPE,
    check=True)
so = cp.stdout 

version = so.strip().decode("ascii").lstrip("v").replace("-",
    "+dev", 1).replace("-", ".")

setup(
    name='FCDR_HIRS',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='Library and scripts for HIRS FCDR analysis and production',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/FIDUCEO/FCDR_HIRS',

    # Author details
    author='Gerrit Holl',
    author_email='g.holl@reading.ac.uk',

    # Choose your license
    license='GPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
    ],

    # What does your project relate to?
    keywords="FIDUCEO HIRS radiometer metrology climate",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["numpy>=1.13",
                      "scipy>=1.0",
                      "matplotlib>=2.0",
                      "numexpr>=2.6",
                      "typhon>=0.5.0",
                      "pyatmlab>=0.1.1",
                      "progressbar2>=3.10",
                      "netCDF4>=1.2",
                      "pandas>=0.21",
                      "xarray>=0.10",
                      "seaborn>=0.7",
                      "sympy==1.0", # see #151
                      "pint>=0.8",
                      "fcdr-tools>=1.1.1",
                      "joblib>=0.11",
                      "cartopy"],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # FIXME: include SRFs here
#    package_data={
#        'sample': ['package_data.dat'],
#    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
#    data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'plot_hirs_field_timeseries=FCDR_HIRS.analysis.timeseries:main',
            "inspect_hirs_matchups=FCDR_HIRS.analysis.inspect_hirs_matchups:main",
            "map_hirs_field=FCDR_HIRS.analysis.map:main",
            "plot_hirs_field_matrix=FCDR_HIRS.analysis.fieldmat:main",
            "plot_hirs_calibcounts_per_scanpos=FCDR_HIRS.analysis.calibcounts_stats_per_scanpos:main",
            "plot_hirs_test_rself=FCDR_HIRS.analysis.test_rself:main",
            "calc_sensitivity_params=FCDR_HIRS.analysis.sensitivities:main",
            "combine_hirs_hirs_matchups=FCDR_HIRS.processing.combine_matchups:combine_hirs",
            "combine_hirs_iasi_matchups=FCDR_HIRS.processing.combine_matchups:combine_iasi",
            "generate_fcdr=FCDR_HIRS.processing.generate_fcdr:main",
            "convert_hirs_srfs=FCDR_HIRS.analysis.convert_srfs_with_shift:main",
            "plot_hirs_fcdr=FCDR_HIRS.analysis.monitor_fcdr:main",
            "summarise_hirs_fcdr=FCDR_HIRS.analysis.summarise_fcdr:summarise",
            "determine_hirs_latlon_compression_ratio=FCDR_HIRS.analysis.determine_latlon_compression_ratio:main",
            "determine_hirs_unc_storage=FCDR_HIRS.analysis.determine_optimal_uncertainty_format:main",
            "plot_hirs_flags=FCDR_HIRS.analysis.plot_flags:main",
            "hirs_info_content=FCDR_HIRS.analysis.corrmat_info_content:main",
            "hirs_logfile_analysis=FCDR_HIRS.analysis.logfile_analysis:main",
            "hirs_orbit_map=FCDR_HIRS.analysis.map_single_orbit:main",
            "write_hirs_harm_meta=FCDR_HIRS.analysis.write_harm_meta:main"
        ],
    },
)
