## General modelchain documentation ##

See [`CHAIN MANUAL`](https://github.com/TNO/SHRA-Groningen-seismic-source-model/blob/main/CHAIN_MANUAL.md) in the [`SHRA-Groningen-seismic-source-model`](https://github.com/TNO/SHRA-Groningen-seismic-source-model/) repository.

## Hazard and risk models documentation ##

 The **recommended** approach is to clone this repository through [git](https://git-scm.com/) by running:  
  `git clone --recurse-submodules --remote-submodules <repository-address>` <br>
  When using this approach, the `chaintools` repository is automatically included in the correct manner and there is no  need to obtain it separately.
  
  Alternatively, a copy of the code can be obtained by using the 'download' button. 
  In this case, the 'chaintools' folder is downloaded as an empty folder. This folder should be 
  replaced with the full [`chaintools`](https://github.com/TNO/SHRA-Groningen-chaintools) repository which has to be downloaded separately (this means that
  the project should contain the folder `chaintools/chaintools`, the lower of which contains an `__init__.py` file).

### Setting up virtual Python environments ###

To run the code, the user needs to set up a Python environment. 
We highly recommend using [mamba](https://github.com/conda-forge/miniforge) or 
[conda](https://docs.conda.io/projects/miniconda/en/latest/) as your package manager, as this ensured that any 
required binaries are taken care of (this is not the case for the default Python package manager pip).
Mamba and Conda also ensure that Python is available on the system, if this was not already the case.

The repository contains an `environment.yml` file which can be used to set up a Python environment which contains all the relevant packages (and their correct versions) required to run 
the code as intended. 

To set up the virtual environment, the following command is run (`conda` and `mamba` may be used interchangably): <br>
`mamba env create --f <path_to_environment.yml>` <br>

> The provided `environment.yml` files has references to the exact versions of packages used by the developers, 
which are not available under operating systems other than _Ubuntu 20.04.6 LTS_. In these cases, the less comprehensive 
`environment_light.yml` can be used instead.  However, it should be stressed that this has not been tested extensively 
and may require some custom solutions.

This creates a virtual environment with the name `hazard_risk_models`.

### Obtaining required inputfiles ###

The datafiles that are required as input into the modelchain are published at [Zenodo](https://doi.org/10.5281/zenodo.10245813)

### Running the hazard and risk models ###

First activate the python environment: <br>
`mamba activate hazard_risk_models` <br>

Then, the work is divided in two parts. First, a number of preparatory calculations need to be performed. These calculations
depend on the models and model versions available, but not on the actual realization of the seismic source model. Second, the
hazard and risk calculations for a specific realization of the seismic source model can be performed. <br>

Note that the preparatory calculations can take a lot of computation time and memory. At TNO the codes are run on a server with 256 GB of working memory. Results on more limited hardware may vary. This has not been tested. The memory and processing time required can be limited by choosing modest discretization settings and/or calculating a limited subset of, for example, site response zones. <br>

Also note that calculating and storing preporatory calculations for the full
logic tree expansion requires a lot of disk space, in the order of several 
terabytes. Also this can be limited by choosing modest discretization settings
and/or calculating a limited subset of, for example, site response zones. The
final hazard and risk integrators at this moment only provide mean hazard and
mean risk and therefore do not rely on the full logic tree expansion of the
preparatory calculations.<br>

Run the preparatory calculations: <br>
`python parse_input.py hr_config.yml` <br>
`python gmm_tables.py hr_config.yml` <br>
`python fcm_tables.py hr_config.yml` <br> 
`python rupture_prep.py hr_config.yml`<br>
`python exposure_prep.py hr_config.yml`<br>
`python hazard_prep.py hr_config.yml`<br>
`python im_prep.py hr_config.yml`<br>
`python risk_prep.py hr_config.yml`<br>

Run the hazard and risk calculations: <br>
`python source_integrator.py hr_config.yml`<br>
`python hazard_integrator.py hr_config.yml`<br>
`python risk_integrator.py hr_config.yml`<br>

An example `hr_config.yml` file is provided in the [demo](/demo) folder. <br>
For more details see the [`CHAIN MANUAL`](https://github.com/TNO/SHRA-Groningen-seismic-source-model/blob/main/CHAIN_MANUAL.md) in the [`SHRA-Groningen-seismic-source-model`](https://github.com/TNO/SHRA-Groningen-seismic-source-model/) repository.

## License ##
Licensed under the [EUPL](/LICENSE)

Copyright (C) 2023 TNO
