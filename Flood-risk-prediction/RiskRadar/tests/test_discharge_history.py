"""Causal-history checks protect against current/future target leakage."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discharge_history import build_history_features, HISTORY_FEATURES


class HistoryLeakageTests(unittest.TestCase):
    def setUp(self):
        self.ref = pd.DataFrame({'gsim_code': ['A']*4, 'numeric_year': [1990, 1992, 1992, 1995],
                                 'source_peak_discharge': [10., 500., 600., 900.]})
        self.query = pd.DataFrame({'gsim_code': ['A'], 'numeric_year': [1992]})

    def test_same_year_records_are_excluded(self):
        row = build_history_features(self.query, self.ref, 100.).iloc[0]
        self.assertEqual(row.history_count, 1)
        self.assertEqual(row.history_last_ratio, .1)
        self.assertEqual(row.history_latest_year, 1990)

    def test_current_and_future_outcome_changes_cannot_change_features(self):
        changed = self.ref.copy()
        changed.loc[changed.numeric_year.ge(1992), 'source_peak_discharge'] = 1e12
        a = build_history_features(self.query, self.ref, 100.)[HISTORY_FEATURES]
        b = build_history_features(self.query, changed, 100.)[HISTORY_FEATURES]
        pd.testing.assert_frame_equal(a, b)

    def test_unseen_station_has_missing_flow_not_zero_flow(self):
        query = pd.DataFrame({'gsim_code': ['B'], 'numeric_year': [2000]})
        row = build_history_features(query, self.ref, 100.).iloc[0]
        self.assertEqual(row.history_count, 0)
        self.assertTrue(np.isnan(row.history_mean_ratio))

    def test_query_outcomes_are_never_used(self):
        query = pd.DataFrame({'gsim_code': ['A'], 'numeric_year': [2000], 'source_peak_discharge': [1e12]})
        first = build_history_features(query, self.ref, 100.)
        query.source_peak_discharge = 0.
        second = build_history_features(query, self.ref, 100.)
        pd.testing.assert_frame_equal(first[HISTORY_FEATURES], second[HISTORY_FEATURES])

    def test_future_rows_in_reference_are_ignored(self):
        ref = pd.concat([self.ref, pd.DataFrame({'gsim_code': ['A'], 'numeric_year': [2100],
                                                'source_peak_discharge': [1e12]})], ignore_index=True)
        a = build_history_features(self.query, self.ref, 100.)
        b = build_history_features(self.query, ref, 100.)
        pd.testing.assert_frame_equal(a[HISTORY_FEATURES], b[HISTORY_FEATURES])

    def test_unknown_target_year_cannot_access_entire_history(self):
        query = pd.DataFrame({'gsim_code': ['A'], 'numeric_year': [np.nan]})
        with self.assertRaises(ValueError):
            build_history_features(query, self.ref, 100.)


if __name__ == '__main__':
    unittest.main()
