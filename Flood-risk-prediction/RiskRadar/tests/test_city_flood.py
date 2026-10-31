"""Causal inputs, physical target meaning and manual-input boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from city_flood import feature_values,prepare_dataset,rainfall_reports,FEATURES,WARNING,predict_manual


class CityFloodTests(unittest.TestCase):
    def observations(self):
        index=pd.date_range('2020-07-01',periods=100,freq='h')
        return pd.DataFrame({'level':np.linspace(532,534,len(index)),
            'observed_max':np.linspace(532,534,len(index))},index=index)

    def empty_rain(self):return pd.DataFrame({'time':pd.to_datetime([]),'recent_rain_mm':pd.Series(dtype=float)})

    def test_unknown_rain_does_not_become_zero(self):
        values=feature_values('2025-07-01',533,532.8,532)
        self.assertTrue(pd.isna(values.recent_rain_mm.iloc[0]));self.assertEqual(values.rain_available.iloc[0],0)

    def test_measured_zero_rain_is_distinct_from_missing(self):
        values=feature_values('2025-07-01',533,532.8,532,0)
        self.assertEqual(values.recent_rain_mm.iloc[0],0);self.assertEqual(values.rain_available.iloc[0],1)

    def test_invalid_manual_inputs_rejected(self):
        for current,rain in [(np.nan,0),(100,0),(533,-1),(533,np.inf)]:
            with self.assertRaises(ValueError):feature_values('2025-07-01',current,532,531,rain)

    def test_rain_published_after_issue_cannot_enter_features(self):
        rain=pd.DataFrame({'time':pd.to_datetime(['2020-07-02 08:30']),'recent_rain_mm':[90.]})
        data=prepare_dataset(self.observations(),rain)
        row=data.loc[data.time.eq('2020-07-02 08:00')].iloc[0]
        self.assertTrue(pd.isna(row.recent_rain_mm));self.assertEqual(row.rain_available,0)

    def test_stale_rain_is_unknown(self):
        rain=pd.DataFrame({'time':pd.to_datetime(['2020-07-01 00:00']),'recent_rain_mm':[90.]})
        data=prepare_dataset(self.observations(),rain)
        self.assertTrue(data.loc[data.time.eq('2020-07-02 08:00'),'recent_rain_mm'].isna().all())

    def test_future_peak_defines_label_without_becoming_input(self):
        hourly=self.observations();hourly.loc['2020-07-02 12:00','observed_max']=WARNING+.1
        data=prepare_dataset(hourly,self.empty_rain())
        row=data.loc[data.time.eq('2020-07-02 08:00')].iloc[0]
        self.assertEqual(row.target,1);self.assertLess(row.current_level,WARNING)
        self.assertFalse(set(FEATURES)&{'target','target_peak','label_end','future_observed_hours'})

    def test_incomplete_future_is_not_a_negative_example(self):
        hourly=self.observations();hourly.loc['2020-07-02 12:00','observed_max']=np.nan
        data=prepare_dataset(hourly,self.empty_rain())
        self.assertFalse(data.time.eq('2020-07-02 08:00').any())

    def test_future_changes_cannot_change_issue_inputs(self):
        a=self.observations();b=a.copy();b.loc['2020-07-02 09:00':,['level','observed_max']]=540
        x=prepare_dataset(a,self.empty_rain());y=prepare_dataset(b,self.empty_rain())
        pd.testing.assert_frame_equal(x.loc[x.time.eq('2020-07-02 08:00'),FEATURES].reset_index(drop=True),
            y.loc[y.time.eq('2020-07-02 08:00'),FEATURES].reset_index(drop=True))

    def test_unsupported_observation_datum_not_accepted(self):
        with self.assertRaises(ValueError):predict_manual('2026-07-01',10,9,8,bundle={})


if __name__=='__main__':unittest.main()
