# SECTION 1: GENERIC SETTINGS

# SECTION 1.1: PATHS to FILES and DIRECTORIES
paths:
  # base paths
  base_path: &base_path /data/os_psdra/run_demo/

  # external input files and directories
  gmm_input_path: &gmm_input_path [*base_path, res]
  fcm_input_path: &fcm_input_path [*base_path, res]

  forecast_file: &forecast_file [*base_path, ssm_forecast.h5]
  zonation_file: &zonation_file [*base_path, Geological_zones_V6.zip]

  # internal files: preparatory 
  config_file: &config_file [*base_path, config.zarr]
  tables_file: &tables_file [*base_path, tables.zarr]
  hazard_prep_file: &hazard_prep_file [*base_path, hazard_prep.zarr]
  risk_prep_file: &risk_prep_file [*base_path, risk_prep.zarr]
  exposure_prep_file: &exposure_prep_file [*base_path, exposure_prep.zarr]
  rupture_prep_file: &rupture_prep_file [*base_path, rupture_prep.zarr]

  # internal files: operational
  source_distribution_file:
    &source_distribution_file [*base_path, source_distribution.zarr]
  hazard_risk_file: &hazard_risk_file [*base_path, hazard_risk.zarr]

# SECTION 1.2: MODEL VERSIONS
model_versions:
  gmm_version: &gmm_version GMM-V6 # or GMM-V7
  fcm_version: &fcm_version FCM-V7 # or FCM-TNO2020

# SECTION 1.3: DATA STORES
# We recognize the following data types:
# - xarray_dataarray: a single xarray dataarray
# - xarray_dataset: a single xarray dataset consisting of multiple dataarrays
# - xarray_datatree: a tree of xarray datasets
#
# The data types are independent of the file format / structure.
# The file format is currently determined by the file extension, e.g.,
# - .h5: HDF5 file
# - .zarr: Zarr data store (actually a directory)
# We advise to use .zarr exclusively for the data stores, as some of the
# modules depend on its append functionality.

data_stores:
  config_tree: &config_tree
    type: xarray_datatree
    path: *config_file

  gmm_config: &gmm_config
    type: xarray_dataset
    path: *config_file
    group: *gmm_version

  fcm_config: &fcm_config
    type: xarray_dataset
    path: *config_file
    group: *fcm_version

  gmm_tables: &gmm_tables
    type: xarray_dataset
    path: *tables_file
    group: *gmm_version

  fcm_tables: &fcm_tables
    type: xarray_dataset
    path: *tables_file
    group: *fcm_version

  hazard_prep: &hazard_prep
    type: xarray_dataset
    path: *hazard_prep_file
    group: *gmm_version

  im_prep: &im_prep
    type: xarray_dataset
    path: *hazard_prep_file
    group: [*gmm_version, FCM_hazard]

  risk_prep: &risk_prep
    type: xarray_dataset
    path: *risk_prep_file
    group: [*gmm_version, *fcm_version]

  exposure_grid: &exposure_grid
    type: xarray_dataset
    path: *exposure_prep_file
    group: exposure_grid

  rupture_prep: &rupture_prep
    type: xarray_dataset
    path: *rupture_prep_file

  source_distribution: &source_distribution
    type: xarray_dataset
    path: *source_distribution_file

  forecast: &forecast
    type: xarray_dataarray
    path: *forecast_file
    group: forecast/forecast

  risk: &risk
    type: xarray_dataset
    path: *hazard_risk_file
    group: [*gmm_version, risk, *fcm_version]

  hazard: &hazard
    type: xarray_dataset
    path: *hazard_risk_file
    group: [*gmm_version, hazard]

# sampling settings for various dimensions
# this may include coordinate settings
dimensions:
  magnitude: &magnitude # check consistency with SSM
    length: 51
    interval: [1.5, 6.5]
  distance_hypocenter: &distance_hypocenter
    length: 50 # reduce to speed up
    sequence_spacing: log
    interval: [3.0, 70.0]
    units: km
  distance_rupture: &distance_rupture
    length: 50 # reduce to speed up
    sequence_spacing: log
    interval: [3.0, 70.0]
    units: km
  gm_reference: &gm_reference # allow for other ranges
    length: 100 # reduce to speed up
  gm_surface: &gm_surface
    length: 100 # reduce to speed up

# supplementary coordinates if more than one is needed for a dimension
coordinates: &coordinates
  SA_g_reference: &SA_g_reference
    dim: gm_reference
    sequence_spacing: log
    interval: [1.0e-5, 10.0]
    units: g
  SA_reference:
    <<: *SA_g_reference
    multiplier: 981.0 # unit conversion from g to cm/s2
    units: "cm/s2"
  SA_g_surface: &SA_g_surface
    <<: *SA_g_reference
    dim: gm_surface
  SA_surface: &SA_surface
    <<: *SA_g_surface
    multiplier: 981.0 # unit conversion from g to cm/s2
    units: "cm/s2"

# SECTION 2: MODULE SPECIFIC SETTINGS
modules:
  parse_input:
    gmm_path: *gmm_input_path
    fcm_path: *fcm_input_path
    data_sinks:
      gmm_config: *config_tree
      fcm_config: *config_tree

  gmm_tables:
    data_sources:
      gmm_config: *gmm_config
    data_sinks:
      gmm_tables: *gmm_tables
    dimensions:
      magnitude: *magnitude
      distance_rupture: *distance_rupture
      gm_reference: *gm_reference
      gm_surface: *gm_surface
    coordinates: *coordinates
    chunks:
      zone: 1
      magnitude: 10
      distance_rupture: 10
    # uncomment following to reduce the number of 
    # site response zones by a factor of, e.g., 40
    # this may help to test the code
    # thinnings:
    #   zone: 40
    # alternatively, one may choose to select specific zones
    # filters:
    #   zone: ["110","808"]

  fcm_tables:
    data_sources:
      fcm_config: *fcm_config
    data_sinks:
      fcm_tables: *fcm_tables
    dimensions:
      gm_surface: *gm_surface
    coordinates:
      SA_surface: *SA_surface
      SA_g_surface: *SA_g_surface

  hazard_prep:
    data_sources:
      gmm_tables: *gmm_tables
    data_sinks:
      hazard_prep: *hazard_prep
    full_logictree: false 
    n_workers: 8
    chunks:
      zone: 1
      IM: 10
      distance_rupture: 10
      magnitude: 10

  im_prep:
    data_sources:
      fcm_config: *fcm_config
      gmm_config: *gmm_config
      gmm_tables: *gmm_tables
    data_sinks:
      im_prep: *im_prep
    n_workers: 4
    n_sample: 1_000
    n_batch: 5
    full_logictree: false
    rng_seed: 42
    chunks:
      zone: 1
      magnitude: 20
      distance_rupture: 20
    s2s_mode: default # default: determined by GMM version; alternatives: epistemic (V7), aleatory (V5/6)
    s2s_p2p_mode: consistent # s2s_p2p_mode: zero (NAM/SodM), consistent (TNO), full (implicit in V7)

  risk_prep:
    data_sources:
      im_prep: *im_prep
      fcm_tables: *fcm_tables
    data_sinks:
      risk_prep: *risk_prep
    n_workers: 8
    full_logictree: false
    chunks:
      IM_FCM: 1
      zone: 1
      magnitude: 20
      distance_rupture: 20

  exposure_prep:
    data_sinks:
      exposure_grid: *exposure_grid
    zonation_file: *zonation_file
    grid_crs: EPSG:28992
    grid_spacing: 1000.

  rupture_prep:
    data_sinks:
      rupture_prep: *rupture_prep
    dimensions:
      magnitude: *magnitude
      distance_hypocenter: *distance_hypocenter
      distance_rupture: *distance_rupture
      azimuth:
        length: 91
        units: degrees
    n_workers: 8
    n_sample: 10_000
    rng_seed: 42

  source_integrator:
    data_sources:
      rupture_prep: *rupture_prep
      forecast: *forecast
      exposure_grid: *exposure_grid
    full_logictree: false
    data_sinks:
      source_distribution: *source_distribution
    rupture_azimuth: -30.
    n_workers: 8

  hazard_integrator:
    data_sources:
      source_distribution: *source_distribution
      hazard_prep: *hazard_prep
      exposure_grid: *exposure_grid
    data_sinks:
      hazard: *hazard
    return_periods: [475., 2475.]

  risk_integrator:
    data_sources:
      source_distribution: *source_distribution
      risk_prep: *risk_prep
      exposure_grid: *exposure_grid
    data_sinks:
      risk: *risk
