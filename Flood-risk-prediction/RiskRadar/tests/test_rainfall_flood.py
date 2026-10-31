"""Semantic guards for rainfall matching, unknown labels and model leakage."""
import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd

from rainfall_flood import (RAIN,daily_rainfall,filter_events,filter_inventory,
                            rain_groups,state_tokens,severity_estimates)


class RainfallFloodTests(unittest.TestCase):
    def event(self):
        return {'Start Date':'2018-07-12','split':'test',**{key:float(i*(i+1)/2) for i,key in enumerate(RAIN,1)}}

    def test_daily_rainfall_is_past_only_and_reconstructs_all_cumulative_windows(self):
        row=self.event();daily=daily_rainfall(row)
        self.assertEqual(daily.date.max(),pd.Timestamp('2018-07-11'))
        self.assertEqual(daily.date.min(),pd.Timestamp('2018-07-02'))
        for day in range(1,11):self.assertAlmostEqual(daily.rain_mm.tail(day).sum(),row[f'T{day}d'])

    def test_non_monotone_cumulative_rain_is_rejected(self):
        row=self.event();row['T3d']=.1
        with self.assertRaises(ValueError):daily_rainfall(row)

    def test_missing_rain_is_not_a_dry_day(self):
        row=self.event();row['T3d']=np.nan
        with self.assertRaises(ValueError):daily_rainfall(row)

    def test_empty_rainfall_group_is_unknown_not_zero_risk(self):
        data=pd.DataFrame({'EventID':['a'],'T3d':[5.],'target':[1]})
        grouped=rain_groups(data,edges=[10,20,30])
        self.assertTrue(grouped.loc[grouped.events.eq(0),'severe_share'].isna().all())
        self.assertEqual(grouped.events.sum(),1)

    def test_multistate_inventory_is_not_duplicated_in_total(self):
        data=pd.DataFrame({'UEI':['a','b'],'State':['Maharashtra, Gujarat','Gujarat'],
                           'Start Date':pd.to_datetime(['2018-01-01','2019-01-01'])})
        self.assertEqual(len(filter_inventory(data)),2)
        self.assertEqual(len(filter_inventory(data,'Maharashtra')),1)
        self.assertEqual(len(filter_inventory(data,'Gujarat')),2)
        self.assertEqual(len(filter_inventory(data,'Maha')),0)

    def test_unknown_inventory_state_is_not_invented(self):
        self.assertEqual(state_tokens(np.nan),[])
        self.assertEqual(state_tokens('Orissa, Odisha'),['Odisha'])

    def test_time_filter_uses_event_onset_and_retains_real_labels(self):
        data=pd.DataFrame({'EventID':['a','b'],'State':['Maharashtra','Maharashtra'],
            'Start Date':pd.to_datetime(['2009-01-01','2018-01-01']),'Flood Type':['Flood','Severe Flood']})
        selected=filter_events(data,'Maharashtra',(2015,2019))
        self.assertEqual(selected['Flood Type'].tolist(),['Severe Flood'])

    def test_models_cannot_read_peak_outcome_or_event_label(self):
        fake=Mock();fake.predict_proba.return_value=np.array([[.7,.3]])
        bundle={'models':{'xgb_static':fake,'xgb_rainfall':fake},
                'features':{'xgb_static':['T3d'],'xgb_rainfall':['T3d']},
                'thresholds':{'xgb_static':.2,'xgb_rainfall':.4}}
        row={**self.event(),'Flood Type':'Flood','Peak Flood Level (m)':1,'target':0}
        first=severity_estimates(row,bundle)
        row.update({'Flood Type':'Severe Flood','Peak Flood Level (m)':9999,'target':1})
        self.assertEqual(first,severity_estimates(row,bundle))
        for call in fake.predict_proba.call_args_list:
            self.assertEqual(call.args[0].columns.tolist(),['T3d'])

    def test_training_events_are_not_shown_as_heldout_estimates(self):
        row=self.event();row['split']='train'
        with self.assertRaises(ValueError):severity_estimates(row,{})


if __name__=='__main__':unittest.main()
