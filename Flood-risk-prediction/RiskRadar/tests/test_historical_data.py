import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_data import (ROOT, aggregate_rainfall, default_station, fetch_weather,
                              history_path, read_weather, station_catalog, validate_weather)


class HistoricalDataTests(unittest.TestCase):
    def setUp(self):
        self.catalog = station_catalog()
        self.station = default_station(self.catalog)
        self.frame, self.document = read_weather(self.station)

    def test_source_catalog_and_audit(self):
        report = json.loads((ROOT / 'data/processed/data_audit.json').read_text())
        self.assertEqual(report['raw_rows'], 45932)
        self.assertEqual(report['raw_columns'], 93)
        self.assertEqual(len(self.catalog), 9710)
        self.assertEqual(self.catalog.country.nunique(), 57)
        self.assertEqual(report['invalid_year_records_excluded'], 22)
        self.assertFalse(report['contains_binary_flood_labels'])
        self.assertFalse(report['rainfall_is_time_series'])
        self.assertFalse(self.catalog.station_id.duplicated().any())

    def test_saved_real_archive_is_complete_and_dated(self):
        self.assertEqual(len(self.frame), 9497)
        self.assertEqual(self.frame.date.min(), pd.Timestamp('2000-01-01'))
        self.assertEqual(self.frame.date.max(), pd.Timestamp('2025-12-31'))
        self.assertEqual(self.frame.rain_mm.isna().sum(), 0)
        self.assertTrue(self.frame.rain_mm.ge(0).all())
        self.assertEqual(self.document['requested']['models'], 'era5')
        self.assertEqual(self.document['requested']['daily'], 'rain_sum')

    def test_annual_monthly_and_daily_totals_agree(self):
        yearly = aggregate_rainfall(self.frame, 'YS')
        monthly = aggregate_rainfall(self.frame, 'MS')
        self.assertEqual(yearly.notna().sum(), 26)
        self.assertEqual(monthly.notna().sum(), 312)
        self.assertAlmostEqual(yearly.sum(), self.frame.rain_mm.sum())
        self.assertAlmostEqual(yearly.sum(), monthly.sum())

    def test_missing_day_invalidates_period_without_zero_filling(self):
        frame = self.frame.copy()
        frame.loc[0, 'rain_mm'] = np.nan
        self.assertTrue(pd.isna(aggregate_rainfall(frame, 'MS').iloc[0]))
        self.assertTrue(pd.isna(aggregate_rainfall(frame, 'YS').iloc[0]))
        self.assertTrue(pd.notna(aggregate_rainfall(frame, 'YS').iloc[1]))

    def test_negative_and_non_numeric_rain_remain_missing(self):
        doc = copy.deepcopy(self.document)
        doc['response']['daily']['rain_sum'][:3] = [-1, None, 'bad']
        result = validate_weather(doc)
        self.assertTrue(result.rain_mm.iloc[:3].isna().all())

    def test_duplicate_dates_are_rejected(self):
        doc = copy.deepcopy(self.document)
        doc['response']['daily']['time'][1] = doc['response']['daily']['time'][0]
        with self.assertRaises(ValueError):
            validate_weather(doc)

    def test_units_and_lengths_are_validated(self):
        doc = copy.deepcopy(self.document)
        doc['response']['daily_units']['rain_sum'] = 'inch'
        with self.assertRaises(ValueError):
            validate_weather(doc)
        doc = copy.deepcopy(self.document)
        doc['response']['daily']['rain_sum'].pop()
        with self.assertRaises(ValueError):
            validate_weather(doc)

    def test_missing_dates_are_reindexed_as_missing(self):
        doc = copy.deepcopy(self.document)
        doc['response']['daily']['time'].pop(1)
        doc['response']['daily']['rain_sum'].pop(1)
        frame = validate_weather(doc)
        self.assertEqual(len(frame), 9497)
        self.assertTrue(pd.isna(frame.rain_mm.iloc[1]))

    def test_cached_history_works_offline(self):
        with patch('requests.get', side_effect=requests.ConnectionError('offline')):
            self.assertEqual(fetch_weather(self.station)['station_id'], self.station)

    def test_unknown_station_cannot_become_a_path(self):
        with self.assertRaises(ValueError):
            history_path('../../api.py')


if __name__ == '__main__':
    unittest.main()
