# SkyLens

This is a public version of the code used for calculations presented in [Singh 2021](https://arxiv.org/abs/2105.04548). 

Dependencies

1. Dask: To enable parallel computation using graphs.
2. Boltzmann code to compute matter Power spectra, P(k,z): Camb, Class and CCL are supported. There are wrapper functions in power_spectra.py. Default is Camb. In passing pk_params dictionary, you can choose the function to call and compute power spectra.
3. sympy: To help with computation of wigner_3j matrices. 
4. sparse, zarr: This is used to effciently store and read some large wigner_3j matrices, which are computed only once.
5. Healpy: For window related calcuations, we use healpy maps (window inputs are healpy maps).
6. Astropy, numpy, scipy.

## Usage

Skylens can be installed by cloning the repository and then running: python setup.py install.

Please see the notebooks in imaster_notebooks directory for examples of various calculations. 

For certain calculations, wigner_3j matrices are required. These are read from files and such files can be generated using Gen_wig_m0.py and Gen_wig_m2.py provided within the skylens directory.

Code is still under active development, apologies for the tardiness. If you need any help in running the code, please feel free to send me (Sukhdeep) an email. I am also available on Slack if you are part of LSST-DESC.

## License
This program comes with no gurarantees/warranties and is solely provided in the hope that it maybe useful. If you use this code, please cite the Singh 2021 paper. 
You are free copy and modify the code as desired as long a you provide appropriate citation to the original paper.
