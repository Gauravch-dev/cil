"""Independent original-model parity plus invalid-input and category coverage."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from riskradar_predictor import DEFAULTS,EXAMPLES,FEATURES,artifacts,build_features,category_choices,predict,risk_band


class RiskRadarPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(StringIO()):import api
        api.model.set_params(device='cpu',n_jobs=2)
        cls.reference=api

    def test_original_example_outputs_preserved(self):
        expected={'Typical':(.0779,'LOW'),'Drier':(.0038,'LOW'),'Wetter':(.784,'HIGH')}
        for name,values in EXAMPLES.items():
            result=predict(values)
            self.assertEqual((result['flood_risk_score'],result['risk_level']),expected[name])

    def test_features_and_scores_match_independent_original_api(self):
        rng=np.random.default_rng(20261025)
        for _ in range(40):
            annual=float(rng.uniform(0,6000));wet=float(rng.uniform(annual/12,annual))
            values={**DEFAULTS,'annual_precipitation':annual,'precipitation_of_wettest_month':wet,
                'precipitation_seasonality':float(rng.uniform(0,250)),
                'climate_type':int(rng.integers(0,6)),'landcover_type':int(rng.integers(0,11)),
                'soil_type':int(rng.integers(0,25))}
            raw=self.reference.FloodInput(**values)
            np.testing.assert_array_equal(build_features(values),self.reference.build_features(raw))
            original=self.reference.predict(raw)
            self.assertEqual(predict(values),{key:original[key] for key in ['flood_risk_score','risk_level']})

    def test_source_encoder_vocabulary_is_fully_available(self):
        self.assertEqual([len(category_choices()[key]) for key in ['climate_type','landcover_type','soil_type']],[6,11,25])
        for key,choices in category_choices().items():
            self.assertEqual(choices[-1],'Unspecified' if key!='climate_type' else 'Warm Temperate')
            self.assertTrue(np.isfinite(build_features({**DEFAULTS,key:len(choices)-1})).all())

    def test_no_arbitrary_3000_mm_or_100_percent_input_cap(self):
        features=build_features({**DEFAULTS,'annual_precipitation':5000.,'precipitation_of_wettest_month':1500.,'precipitation_seasonality':150.})
        self.assertEqual(features[0,FEATURES.index('annual_precipitation')],5000.)

    def test_negative_rain_and_terrain_are_rejected(self):
        for key in set(DEFAULTS)-{'annual_mean_temperature','climate_type','landcover_type','soil_type'}:
            with self.assertRaises(ValueError,msg=key):predict({**DEFAULTS,key:-1})

    def test_nonfinite_and_missing_values_are_rejected(self):
        for bad in [None,np.nan,np.inf,-np.inf,True,'not a number']:
            with self.assertRaises(ValueError):predict({**DEFAULTS,'annual_precipitation':bad})
        with self.assertRaises(ValueError):predict({})

    def test_rainfall_period_consistency_is_checked(self):
        for values in [{'annual_precipitation':100.,'precipitation_of_wettest_month':300.},
                       {'annual_precipitation':2000.,'precipitation_of_wettest_month':100.}]:
            with self.assertRaises(ValueError):predict({**DEFAULTS,**values})

    def test_dry_and_fractional_conditions_are_supported(self):
        for annual,wet in [(0.,0.),(10.5,2.5)]:
            self.assertIn(predict({**DEFAULTS,'annual_precipitation':annual,'precipitation_of_wettest_month':wet})['risk_level'],['LOW','MEDIUM','HIGH'])

    def test_invalid_category_codes_are_rejected(self):
        for key,count in [('climate_type',6),('landcover_type',11),('soil_type',25)]:
            for value in [-1,count,1.5,True]:
                with self.assertRaises(ValueError):predict({**DEFAULTS,key:value})

    def test_runoff_and_temperature_units_are_checked(self):
        for key,value in [('curve_number_amcii',101.),('annual_mean_temperature',100.)]:
            with self.assertRaises(ValueError):predict({**DEFAULTS,key:value})

    def test_original_risk_band_boundaries(self):
        self.assertEqual([risk_band(v) for v in [0,.4,.40001,.7,.70001,1]],['LOW','LOW','MEDIUM','MEDIUM','HIGH','HIGH'])


if __name__=='__main__':unittest.main()
