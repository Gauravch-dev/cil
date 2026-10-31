"""Guards for genuine rainfall targets, causal inputs and indicator semantics."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from rainfall_risk import features,category,historical_indicators,predict,NUMERIC
from rainfall_flood import gauge_risk_summary


class RainfallRiskTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'rain_mm':np.arange(35,dtype=float)%8},
                            index=pd.date_range('2021-07-01',periods=35))

    def test_future_rain_does_not_change_earlier_features(self):
        original=self.frame();changed=original.copy();changed.loc[changed.index[25]:,'rain_mm']=999.
        pd.testing.assert_frame_equal(features(original).iloc[:25],features(changed).iloc[:25])

    def test_fourteen_day_window_is_required(self):
        frame=self.frame();bank=features(frame)
        self.assertTrue(bank.iloc[12].isna().any());self.assertFalse(bank.iloc[13].isna().any())

    def test_missing_rain_is_not_imputed_as_dry(self):
        frame=self.frame();frame.iloc[-3,0]=np.nan
        self.assertTrue(features(frame).iloc[-1].isna().any())

    def test_thresholds_and_dry_day_boundary(self):
        np.testing.assert_array_equal(category([0,9.9,10,19.9,20],{'medium':10,'high':20}),[0,0,1,1,2])
        np.testing.assert_array_equal(category([0,1],{'medium':0,'high':0}),[0,2])

    def test_negative_or_incomplete_simulations_are_rejected(self):
        for values in [[0.]*13,[0.]*13+[-1.],[0.]*13+[np.nan]]:
            with self.assertRaises(ValueError):predict('2026-07-01',values,bundle={})

    def test_target_outcomes_are_not_features(self):
        self.assertFalse(set(NUMERIC)&{'target','target_total','label_end','Flood Type','Peak Flood Level (m)'})

    def test_missing_month_is_not_a_normal_month(self):
        frame=self.frame();frame.iloc[4,0]=np.nan
        analysis=historical_indicators(frame,baseline=self.frame())
        self.assertTrue(pd.isna(analysis['monthly'].iloc[0].rain_mm))

    def test_no_wet_baseline_gives_unknown_heavy_threshold(self):
        baseline=self.frame()*0
        result=historical_indicators(self.frame(),baseline)
        self.assertTrue(pd.isna(result['heavy_threshold_mm']))
        self.assertTrue(result['heavy_days'].isna().all())

    def test_default_anomaly_baseline_excludes_later_values(self):
        frame=pd.DataFrame({'rain_mm':np.arange(800,dtype=float)%11},
                            index=pd.date_range('2014-01-01',periods=800))
        first=historical_indicators(frame)
        changed=frame.copy();changed.loc[changed.index.year>2015,'rain_mm']=999.
        second=historical_indicators(changed)
        self.assertEqual(first['heavy_threshold_mm'],second['heavy_threshold_mm'])

    def test_sparse_gauge_is_not_assigned_low_or_high(self):
        rows=[]
        for gauge,severe in [('a',0),('b',5),('c',10)]:
            for i in range(10):rows.append({'GaugeID':gauge,'EventID':f'{gauge}{i}','target':int(i<severe),
                'Station':gauge,'State':'Test','Latitude':10.,'Longitude':72.,'T3d':20.})
        reference=pd.DataFrame(rows)
        summary,_=gauge_risk_summary(reference.iloc[:1],reference)
        self.assertEqual(summary.band.tolist(),['Limited records'])


if __name__=='__main__':unittest.main()
