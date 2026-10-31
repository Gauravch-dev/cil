import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from river_forecast import clean_station,causal_features,supervised_rows,stage_category,FEATURES


def fixture(hours=150):
    times=pd.date_range('2023-07-01 08:00',periods=hours,freq='h')
    return pd.DataFrame({'Data Acquisition Time':times.strftime('%d-%m-%Y %H:%M'),
                         'River Water Level Manual Hourly (meter)':525+np.arange(hours)*.01})


class RiverForecastTests(unittest.TestCase):
    def test_future_mutations_do_not_change_features(self):
        raw=fixture();issue=pd.Timestamp('2023-07-05 08:00')
        first,_=clean_station(raw,'kurundwad')
        altered=raw.copy();altered.loc[97:,'River Water Level Manual Hourly (meter)']+=2
        second,_=clean_station(altered,'kurundwad')
        a=causal_features(first,'kurundwad').loc[:issue,FEATURES]
        b=causal_features(second,'kurundwad').loc[:issue,FEATURES]
        pd.testing.assert_frame_equal(a,b)

    def test_later_reading_in_same_hour_cannot_enter_earlier_features(self):
        raw=fixture();extra=pd.DataFrame({'Data Acquisition Time':['05-07-2023 08:30'],
                                          'River Water Level Manual Hourly (meter)':[530.]})
        a,_=clean_station(raw,'kurundwad');b,_=clean_station(pd.concat([raw,extra]),'kurundwad')
        self.assertEqual(a.loc['2023-07-05 08:00','level'],b.loc['2023-07-05 08:00','level'])

    def test_missing_future_reading_is_unknown_not_interpolated_label(self):
        raw=fixture();raw=raw.drop(index=100)
        h,_=clean_station(raw,'kurundwad');rows=supervised_rows(h,'kurundwad')
        self.assertTrue(rows.loc['2023-07-05 08:00','input_eligible'])
        self.assertTrue(pd.isna(rows.loc['2023-07-05 08:00','target_peak']))

    def test_target_excludes_current_and_includes_next_24_hours(self):
        h,_=clean_station(fixture(),'kurundwad');rows=supervised_rows(h,'kurundwad')
        self.assertAlmostEqual(rows.loc['2023-07-05 08:00','target_peak'],526.2)

    def test_official_threshold_boundaries(self):
        self.assertEqual(stage_category(535.639,'kurundwad'),'Normal')
        self.assertEqual(stage_category(535.64,'kurundwad'),'Warning')
        self.assertEqual(stage_category(537.16,'kurundwad'),'Danger')
        with self.assertRaises(ValueError):stage_category(float('nan'),'kurundwad')

    def test_conflicting_same_timestamp_values_are_not_averaged(self):
        raw=fixture();extra=raw.iloc[[96]].copy();extra['River Water Level Manual Hourly (meter)']+=1
        h,audit=clean_station(pd.concat([raw,extra]),'kurundwad')
        self.assertEqual(audit['conflicting_timestamps'],1)
        self.assertTrue(pd.isna(h.loc['2023-07-05 08:00','level']))

    def test_long_stale_readings_do_not_become_valid_history(self):
        raw=fixture();raw['River Water Level Manual Hourly (meter)']=525.
        h,audit=clean_station(raw,'kurundwad')
        self.assertGreater(audit['stale_after_72h'],0)
        self.assertFalse(causal_features(h,'kurundwad').loc['2023-07-05 08:00','input_eligible'])

    def test_feature_list_excludes_label_sources(self):
        self.assertFalse(set(FEATURES)&{'target_peak','target_change','label_end','future_observed_hours','observed_max'})


if __name__=='__main__':unittest.main()
