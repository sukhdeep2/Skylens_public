#!/usr/bin/env python
"""
package for 3X2 analysis
http://opensource.org/licenses/MIT
"""
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='skylens',
    version='0.1',
    description='3X2 theory+covariance',
    url='',
    author='Sukhdeep Singh, Hung-Jin Huang, Yin Li',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['skylens'],
    package_dir = {'skylens': 'skylens'},
    install_requires=['scipy', 'numpy', 'camb', 'zarr<=2.3.1', 'sympy', 'sparse', 'dask','distributed','astropy','healpy','dask[bag]','pandas','pyyaml'],
    python_requires='>=3.6',
)
