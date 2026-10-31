"""Regression tests for the evaluation's leakage and evidence gates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from grouped_evaluation import (CATEGORIES,FEATURES,NUMERIC_MAP,RAW_INPUTS,
    bootstrap_confusion,evaluate,features,group_records,pipeline,split_groups,
    summarize,threshold_from_validation)


class GroupedEvaluationTests(unittest.TestCase):
    def raw(self,n=80):
        frame=pd.DataFrame({c:np.arange(n,dtype=float)+1 for c in RAW_INPUTS})
        for c in ['Climate Type','Landcover Type','Soil Type']:frame[c]='category'
        frame['gsim_code']=[f's{i}' for i in range(n)]
        frame['latitude']=np.arange(n)/10;frame['longitude']=np.arange(n)/10+10
        return frame

    def test_station_aliases_and_identical_profiles_cannot_cross_partitions(self):
        raw=self.raw()
        # Same station with different features, a different ID sharing its
        # coordinates, and a third ID with the same predictors form one group.
        raw.loc[1,'gsim_code']=raw.loc[0,'gsim_code']
        raw.loc[2,['latitude','longitude']]=raw.loc[1,['latitude','longitude']].values
        raw.loc[3,RAW_INPUTS]=raw.loc[2,RAW_INPUTS].values
        groups=group_records(raw)
        self.assertEqual(groups.iloc[:4].nunique(),1)
        parts=split_groups(groups)
        allocations={name:set(groups.iloc[index]) for name,index in parts.items()}
        self.assertFalse(allocations['train']&allocations['validation'])
        self.assertFalse(allocations['train']&allocations['test'])
        self.assertFalse(allocations['validation']&allocations['test'])

    def test_outcome_changes_do_not_change_group_or_split(self):
        raw=self.raw();raw['Annual peak discharge']=np.arange(len(raw))
        before=group_records(raw);parts=split_groups(before)
        raw['Annual peak discharge']=raw['Annual peak discharge'].iloc[::-1].to_numpy()*10000
        after=group_records(raw)
        pd.testing.assert_series_equal(before,after)
        for key,index in parts.items():np.testing.assert_array_equal(index,split_groups(after)[key])

    def test_preprocessing_does_not_learn_holdout_medians_or_categories(self):
        raw=self.raw(8);x=features(raw)
        x.loc[0,'basin_relief']=np.nan
        fit=pipeline({'family':'logistic','params':{'max_iter':1000},'weighted':False})
        fit.fit(x,np.array([0,0,0,0,1,1,1,1]))
        transform=fit['preprocess'];median=transform.named_transformers_['numeric']['impute'].statistics_.copy()
        categories=[a.copy() for a in transform.named_transformers_['categories'].categories_]
        holdout=x.copy();holdout['basin_relief']=1e12
        holdout[CATEGORIES]='NEVER_SEEN_IN_TRAIN'
        scores=fit.predict_proba(holdout)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(median,transform.named_transformers_['numeric']['impute'].statistics_)
        for a,b in zip(categories,transform.named_transformers_['categories'].categories_):np.testing.assert_array_equal(a,b)
        self.assertNotIn('Annual peak discharge',FEATURES)

    def test_threshold_ties_prefer_stricter_value(self):
        y=np.array([0,1,0,0,1]);p=np.array([.1,.3,.4,.5,.9])
        threshold=threshold_from_validation(y,p)
        self.assertAlmostEqual(threshold,.9)

    def test_report_uses_consistent_positive_class_and_all_four_confusion_counts(self):
        y=np.array([0,0,0,0,0,1,1,1])
        p=np.array([.1,.2,.3,.9,.8,.9,.8,.2])
        result=summarize(y,p,.5)
        self.assertEqual(result['confusion_matrix'],[[3.,2.],[1.,2.]])
        self.assertAlmostEqual(result['precision'],.5)
        self.assertAlmostEqual(result['recall'],2/3)
        self.assertAlmostEqual(result['accuracy'],5/8)

    def test_repeated_test_access_is_rejected_before_data_read(self):
        base=(Path(__file__).resolve().parents[1]/'evaluation').resolve()
        with tempfile.TemporaryDirectory(dir=base) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(base))
            run=Path(directory);(run/'test_opened.json').write_text('{}')
            with self.assertRaisesRegex(FileExistsError,'already been opened'):evaluate(run)

    def test_bootstrap_keeps_station_year_records_as_clusters(self):
        y=np.array([0,0,1,1]);prediction=y.copy();groups=np.array(['a','a','b','b'])
        intervals=bootstrap_confusion(y,prediction,groups,repetitions=100)
        self.assertEqual(intervals['accuracy'],{'lower':1.,'upper':1.})


if __name__=='__main__':unittest.main()
