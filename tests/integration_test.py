import os
import xarray as xr
import tempfile
import yaml

from unittest import TestCase
from rupture_prep import main as rupture_prep_main
from gmm_tables import main as gmm_tables_main
from fcm_tables import main as fcm_tables_main
from hazard_prep import main as hazard_prep_main
from im_prep import main as im_prep_main
from risk_prep import main as risk_prep_main
from exposure_prep import main as exposure_prep_main
from source_integrator import main as source_integrator_main
from hazard_integrator import main as hazard_integrator_main
from risk_integrator import main as risk_integrator_main
from chaintools.chaintools import tools_configuration as cfg

class IntegrationTests(TestCase):

    test_config_path_v7 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_config_testing_gmmv7.yml')
    test_config_path_v6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_config_testing_gmmv6.yml')

    test_config_path_fcm_v7 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'example_config_testing_gmmv7.yml')
    test_config_path_fcm_TNO2020 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'example_config_testing_fcmTNO2020.yml')

    test_config_path_hazard_integrator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'example_config_testing_hazard_integrator.yml')

    def test_rupture_prep(self):
        # assert
        # ------
        module = 'rupture_prep'
        module_config = cfg.configure([self.test_config_path_v7], module)
        expected_outcome_file = './tests/res/rupture_prep_testing.zarr'
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'rupture_prep.zarr']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][module]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            rupture_prep_main(['',temp_config_path])

            # assert
            # ------
            testing_outcome = xr.open_zarr(outcome_path)
            expected_outcome = xr.open_zarr(expected_outcome_file)
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def test_source_integrator(self):
        # assert
        # ------
        module = 'source_integrator'
        module_config = cfg.configure([self.test_config_path_v7], module)
        expected_outcome_file = './tests/res/source_distribution_testing.h5'
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'source_distribution.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks']['source_distribution']['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            source_integrator_main(['', temp_config_path])

            # assert
            # ------
            testing_outcome = xr.load_dataset(outcome_path, engine='h5netcdf')
            expected_outcome = xr.load_dataset(expected_outcome_file, engine='h5netcdf')
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def test_gmmv6_tables(self):
        """
        Integration test for the workflow to generate ground motion distribution parameter tables based on input the
        variables. Tests for 2 zones only, with a coarse parameter grid. Tests for GMM-V6
        """
        self.gmm_tables_function(gmm_version='GMM_V6')

    def test_gmmv7_tables(self):
        """
        Integration test for the workflow to generate ground motion distribution parameter tables based on input the
        variables. Tests for 2 zones only, with a coarse parameter grid. Tests for GMM-V7.
        """
        self.gmm_tables_function(gmm_version='GMM_V7')

    def test_hazard_prep_gmmv6(self):
        """
        Integration test for the workflow to generate the hazard prep (conditional probabilities of exceedance). Tests
        for 2 zones only, with a coarse parameter grid. Tests for GMM-V6.
        """
        self.hazard_prep_function(gmm_version='GMM_V6')

    def test_hazard_prep_gmmv7(self):
        """
        Integration test for the workflow to generate the hazard prep (conditional probabilities of exceedance). Tests
        for 2 zones only, with a coarse parameter grid. Tests for GMM-V7.
        """
        self.hazard_prep_function(gmm_version='GMM_V7')

    def test_im_prep_gmmv6(self):
        """

        """
        self.im_prep_function(gmm_version='GMM_V6')

    def test_im_prep_gmmv7(self):
        """

        """
        self.im_prep_function(gmm_version='GMM_V7')

    def test_fcm_tables_fcmV7(self):
        """
        Integration test for the workflow to generate tables of fragity and consequence model parameters for FCM V7
        """
        self.fcm_tables_function(fcm_version='FCM_V7')

    def test_fcm_tables_fcmTNO2020(self):
        """
        Integration test for the workflow to generate tables of fragity and consequence model parameters for FCM V7
        """
        self.fcm_tables_function(fcm_version='FCM_TNO2020')

    def test_risk_prep_gmmv6_fcmv7(self):
        """

        """
        self.risk_prep_function(gmm_version='GMM_V6')

    def test_risk_prep_gmmv7_fcmv7(self):
        """

        """
        self.risk_prep_function(gmm_version='GMM_V7')

    def test_risk_prep_gmmv7_fcmTNO2020(self):
        """

        """
        self.risk_prep_function(gmm_version='GMM_V7', fcm_version='FCM_TNO2020')

    def test_hazard_integrator(self):
        """

        """
        # arrange
        # -------
        module = 'hazard_integrator'
        module_config = cfg.configure([self.test_config_path_hazard_integrator], module)
        expected_outcome_file = './tests/res/hazard_risk_testing.h5'
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'hazard_risk_testing.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks']['hazard']['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            hazard_integrator_main(['', temp_config_path])

            # assert
            # ------
            testing_outcome = xr.open_dataset(outcome_path, group='hazard/GMM-V7', engine='h5netcdf')
            expected_outcome = xr.open_dataset(expected_outcome_file, group='hazard/GMM-V7', engine='h5netcdf')
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def test_exposure_prep(self):
        """

        """
        # arrange
        # -------
        module = 'exposure_prep'
        sink_1 = 'exposure_grid'
        sink_2 = 'exposure_database'
        module_config = cfg.configure([self.test_config_path_v7], module)
        expected_outcome_file = './tests/res/exposure_prep_testing.h5'
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'exposure_prep.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][sink_1]['path'] = testing_outcome_file
            module_config['data_sinks'][sink_2]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            exposure_prep_main(['', temp_config_path])

            # assert
            # ------
            testing_outcome = xr.open_dataset(outcome_path, group=sink_1, engine='h5netcdf')
            expected_outcome = xr.open_dataset(expected_outcome_file, group=sink_1, engine='h5netcdf')
            xr.testing.assert_allclose(expected_outcome, testing_outcome)
            testing_outcome = xr.open_dataset(outcome_path, group=sink_2, engine='h5netcdf')
            expected_outcome = xr.open_dataset(expected_outcome_file, group=sink_2, engine='h5netcdf')
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def fcm_tables_function(self, fcm_version: str):
        """
        Integration test for the workflow to generate tables of fragity and consequence model parameters conditional
        on surface ground motions, as specified in the configuration file
        """

        module = 'fcm_tables'
        if fcm_version == 'FCM_TNO2020':
            module_config = cfg.configure([self.test_config_path_fcm_TNO2020], module)
        elif fcm_version == 'FCM_V7':
            module_config = cfg.configure([self.test_config_path_fcm_v7], module)
        else:
            raise UserWarning('Requested unknown version of FCM for testing')

        testing_group = module_config['data_sinks'][module]['group']
        expected_outcome_file = './tests/res/FCM_tables_testing.h5'
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'FCM_tables.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][module]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            fcm_tables_main(['', temp_config_path])

            # assert
            # ------
            testing_outcome = xr.open_dataset(outcome_path, group=testing_group, engine='h5netcdf')
            expected_outcome = xr.open_dataset(expected_outcome_file, group=testing_group, engine='h5netcdf')
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def gmm_tables_function(self, gmm_version: str):
        """
        Integration test for the workflow to generate ground motion distribution parameter tables based on input the
        variables.
        """
        # arrange
        # -------
        module = 'gmm_tables'
        if gmm_version == 'GMM_V6':
            module_config = cfg.configure([self.test_config_path_v6], module)
            expected_outcome_file = './tests/res/GMM_tables_testing_gmmv6.h5'
        elif gmm_version == 'GMM_V7':
            module_config = cfg.configure([self.test_config_path_v7], module)
            expected_outcome_file = './tests/res/GMM_tables_testing_gmmv7.h5'
        else:
            raise UserWarning('Requested unknown version of GMM')

        testing_group = module_config['data_sinks'][module]['group']
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'GMM_tables.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][module]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            gmm_tables_main(['', temp_config_path])

            # assert
            # ------
            testing_outcome = xr.open_dataset(outcome_path, group=testing_group, engine='h5netcdf')
            expected_outcome = xr.open_dataset(expected_outcome_file, group=testing_group, engine='h5netcdf')
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def hazard_prep_function(self, gmm_version: str):
        """
        Integration test for the workflow to generate the hazard prep (conditional probabilities of exceedance). Tests
        for 2 zones only, with a coarse parameter grid.
        """
        # arrange
        # -------
        module = 'hazard_prep'
        if gmm_version == 'GMM_V6':
            module_config = cfg.configure([self.test_config_path_v6], module)
            expected_outcome_file = './tests/res/hazard_prep_testing_gmmv6.h5'
        elif gmm_version == 'GMM_V7':
            module_config = cfg.configure([self.test_config_path_v7], module)
            expected_outcome_file = './tests/res/hazard_prep_testing_gmmv7.h5'
        else:
            raise UserWarning('Requested unknown version of GMM')

        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'hazard_prep.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][module]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            hazard_prep_main(['', temp_config_path])

            # assert
            # ------
            # testing_outcome = xr.open_dataset(outcome_path, engine='h5netcdf')
            # expected_outcome = xr.open_dataset(expected_outcome_file, engine='h5netcdf')
            # xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def im_prep_function(self, gmm_version: str):
        """

        """
        # arrange
        # -------
        module = 'im_prep'
        if gmm_version == 'GMM_V6':
            module_config = cfg.configure([self.test_config_path_v6], module)
            expected_outcome_file = './tests/res/im_prep_testing_gmmv6.h5'
        elif gmm_version == 'GMM_V7':
            module_config = cfg.configure([self.test_config_path_v7], module)
            expected_outcome_file = './tests/res/im_prep_testing_gmmv7.h5'
        else:
            raise UserWarning('Requested unknown version of GMM')

        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'im_prep.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][module]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            im_prep_main(['',temp_config_path])

            # assert
            # ------
            testing_outcome = xr.open_dataset(outcome_path, engine="h5netcdf")
            expected_outcome = xr.open_dataset(expected_outcome_file, engine="h5netcdf")
            testing_outcome = testing_outcome.transpose(*expected_outcome.dims)
            expected_outcome = expected_outcome.transpose(*expected_outcome.dims)
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    def risk_prep_function(self, gmm_version: str, fcm_version='FCM_V7'):
        """

        """

        # arrange
        # -------
        module = 'risk_prep'
        if gmm_version == 'GMM_V6':
            module_config = cfg.configure([self.test_config_path_v6], module)
            expected_outcome_file = './tests/res/risk_prep_testing_gmmv6.h5'
        elif gmm_version == 'GMM_V7' and fcm_version == 'FCM_V7':
            module_config = cfg.configure([self.test_config_path_v7], module)
            expected_outcome_file = './tests/res/risk_prep_testing_gmmv7.h5'
        elif gmm_version == 'GMM_V7' and fcm_version == 'FCM_TNO2020':
            module_config = cfg.configure([self.test_config_path_fcm_TNO2020], module)
            expected_outcome_file = './tests/res/risk_prep_testing_gmmv7_fcmTNO2020.h5'
        else:
            raise UserWarning('Requested unknown version of GMM')

        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            testing_outcome_file = [test_out_dir, 'risk_prep.h5']
            outcome_path = os.path.join(testing_outcome_file[0], testing_outcome_file[1])
            module_config['data_sinks'][module]['path'] = testing_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            risk_prep_main(['', temp_config_path])

            # assert
            # ------
            testing_outcome = xr.load_dataset(outcome_path, engine="h5netcdf")
            expected_outcome = xr.load_dataset(expected_outcome_file, engine="h5netcdf")
            xr.testing.assert_allclose(expected_outcome, testing_outcome)

    @staticmethod
    def build_config_file_for_module(config: dict, dir: str, module: str, config_name: str= 'temp_config.yml') -> str:
        """
        Write and save a new configuration .yaml file for a specific module.
        :param config: Dictionary with configuration for a specific module
        :param dir: Directory where the configuration files will be stored
        :param module: Name of the module
        :param config_name: Optional, name of the configuration file.
        :return: Returns the path to the new configuration file.
        """

        temp_module_config = {'modules': {module: config}}
        temp_yml_path = os.path.join(dir, config_name)
        with open(temp_yml_path, 'w') as f:
            yaml.dump(temp_module_config, f)

        return temp_yml_path
