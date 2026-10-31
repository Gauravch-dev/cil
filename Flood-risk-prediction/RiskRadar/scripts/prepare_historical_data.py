"""Fetch the published source, audit it and cache ERA5 for one selected station."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_data import default_station, fetch_weather, prepare_source, station_catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', help='GloFlo station ID; defaults to a source station near Mumbai')
    args = parser.parse_args()
    report = prepare_source()
    selected = args.station or default_station(station_catalog())
    document = fetch_weather(selected)
    print(json.dumps({'source_rows': report['raw_rows'], 'stations': report['stations'],
                      'countries': report['countries'], 'cached_station': selected,
                      'weather_days': len(document['response']['daily']['time'])}, indent=2))
