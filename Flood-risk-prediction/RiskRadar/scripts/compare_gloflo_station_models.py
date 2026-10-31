"""Expanded catchment features with station-disjoint forward validation.

Stages: prepare, select, evaluate, verify. Preserves all prior experiments/app.
This is a follow-up on already examined data, not a new external test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import time

import catboost
from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import evaluate_gloflo_imbalance as base

RUN = base.ROOT/'experiments/gloflo_station_features_v1'
ADDITIONS = {
    'drainage_area': 'Drainage Area', 'basin_length': 'Basin Length', 'sinuosity_index': 'Sinuosity Index',
    'maximal_flow_length': 'Maximal Flow Length', 'downvalley_length': 'Downvalley Length',
    'stream_order': 'Stream Order', 'overland_flow_length': 'Length of Overland Flow',
    'basin_perimeter': 'Basin Perimeter', 'bifurcation_ratio': 'Weighted Mean Bifurcationratio',
    'form_factor': 'Form Factor', 'relief_ratio': 'Relief Ratio', 'elongation_ratio': 'Elongation Ratio',
    'circularity_ratio': 'Circularity Ratio', 'lemniscates_value': 'Lemniscates Value',
    'drainage_texture': 'Drainage Texture', 'compactness_coefficient': 'Compactness Coefficient',
    'wandering_ratio': 'Wandering Ratio', 'fitness_ratio': 'Fitness Ratio', 'basin_magnitude': 'Basin Magnitude',
    'channel_frequency': 'Channel Frequency', 'drainage_intensity': 'Drainage Intensity',
    'precipitation_wettest_month': 'Precipitation of Wettest Month',
    'precipitation_driest_month': 'Precipitation of Driest Month',
    'precipitation_seasonality': 'Precipitation Seasonality',
    'precipitation_wettest_quarter': 'Precipitation of Wettest Quarter',
    'precipitation_driest_quarter': 'Precipitation of Driest Quarter',
    'mean_diurnal_range': 'Mean Diurnal Range', 'isothermality': 'Isothermality',
    'temperature_annual_range': 'Temperature Annual Range', 'lithology_type': 'Lithology Type',
}
CATEGORIES = base.CATEGORIES+['lithology_type']
ALL_FEATURES = base.FEATURES+list(ADDITIONS)
VIEWS = ['legacy13', 'expanded', 'selected']
CONFIGS = {
    'catboost': [dict(iterations=300, depth=5, weight='none'), dict(iterations=500, depth=6, weight='sqrt')],
    'hist_gradient_boosting': [dict(max_iter=300, max_leaf_nodes=15, l2_regularization=10., weight='none'),
                               dict(max_iter=400, max_leaf_nodes=31, l2_regularization=5., weight='sqrt')],
    'random_forest': [dict(n_estimators=200, max_leaf_nodes=512, min_samples_leaf=5, weight='none'),
                      dict(n_estimators=300, max_leaf_nodes=1024, min_samples_leaf=2, weight='sqrt')],
}
WINDOWS = [(1959, 1960, 1979), (1979, 1980, 1989), (1989, 1990, 1999)]


def station_group(code):
    return int.from_bytes(sha256(str(code).encode()).digest()[:4], 'big') % 3


def prepare():
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Protocol frozen; use another prespecified experiment version.')
    old = base.load(base.RUN/'protocol.json')
    raw_path = base.ROOT/'data/raw/kuntla_flooddatabase.csv'
    assert base.digest(raw_path) == old['source_sha256']
    raw = pd.read_csv(raw_path, low_memory=False)
    (RUN/'data').mkdir(parents=True, exist_ok=True)
    for split in ['train', 'validation', 'test']:
        part = base.data(split, old)
        source = raw.iloc[part.source_row_index].reset_index(drop=True)
        assert np.array_equal(source.gsim_code.astype(str), part.gsim_code.astype(str))
        for name, column in ADDITIONS.items():
            part[name] = source[column].fillna('Unspecified').astype(str) if name in CATEGORIES else (
                pd.to_numeric(source[column], errors='coerce').replace([np.inf, -np.inf], np.nan))
        part.to_csv(RUN/f'data/{split}.csv', index=False)
    train = pd.read_csv(RUN/'data/train.csv')
    groups = train.gsim_code.map(station_group)
    folds, manifest = [], []
    for i, (end, start, stop) in enumerate(WINDOWS):
        a = train.loc[(train.numeric_year <= end) & (groups != i)]
        b = train.loc[train.numeric_year.between(start, stop) & (groups == i)]
        assert set(a.gsim_code).isdisjoint(b.gsim_code)
        assert a.numeric_year.max() < b.numeric_year.min()
        cutoff = float(a.source_peak_discharge.quantile(.95))
        folds.append({'fold': i, 'train_end': end, 'validation_start': start, 'validation_end': stop,
                      'cutoff': cutoff, 'train_rows': len(a), 'validation_rows': len(b),
                      'train_stations': a.gsim_code.nunique(), 'validation_stations': b.gsim_code.nunique(),
                      'positive_train': int(a.source_peak_discharge.gt(cutoff).sum()),
                      'positive_validation': int(b.source_peak_discharge.gt(cutoff).sum())})
        for role, frame in [('train', a), ('validation', b)]:
            for row in frame[['source_row_index', 'gsim_code', 'numeric_year']].to_dict('records'):
                manifest.append({'fold': i, 'role': role, **row})
    pd.DataFrame(manifest).to_csv(RUN/'data/station_time_fold_manifest.csv', index=False)
    protocol = {
        'task': old['task'], 'cutoff': old['cutoff'], 'chronology': old['chronology'],
        'source_sha256': old['source_sha256'], 'original_model_sha256': base.MODEL_SHA,
        'features': ALL_FEATURES, 'categories': CATEGORIES, 'additional_source_columns': ADDITIONS,
        'feature_views': VIEWS, 'configs': CONFIGS, 'seed': base.SEED, 'folds': folds,
        'fold_rule': 'SHA256(station code) first 4 bytes modulo 3, with group i held out in forward-time fold i. '
                     'All records from those stations are excluded from that fold training.',
        'fold_labels': 'Each fold derives the 95th percentile from its own training rows only. '
                       'Final models use the unchanged full-training cutoff 1129.84.',
        'feature_selection': 'Within each training fold: exclude additions >40% missing, constant, or '
                             'Pearson correlation >0.98 with retained numeric features. Retain legacy13. '
                             'For selected view retain top 12 additions by training-only, station-balanced '
                             'Random Forest grouped importance. Refit selector only on full training for final models.',
        'excluded': {'Annual peak discharge': 'target source', 'Unit peak discharge': 'derived from target',
                     'gsim_code/country/coordinates/year': 'identifiers/location memorisation or split fields',
                     'Number of Dams': 'historical commissioning/availability unknown'},
        'selection': 'Maximum mean fold average precision on unseen stations; ties minimum fold AP then pooled OOF F2. '
                     'Best configuration per learner/view and overall winner frozen before outer validation/test.',
        'threshold': 'From pooled station-disjoint forward OOF training predictions: maximise F2 subject to '
                     'precision >=0.50 and recall >=0.75. If infeasible, maximise F2 and disclose failure. '
                     'Scores rounded to 12 decimals for numerical stability. No outer validation/test threshold tuning.',
        'candidate_count': 18, 'cv_fits': 54, 'final_pipelines': 9,
        'split_hashes': {s: base.digest(RUN/f'data/{s}.csv') for s in ['train', 'validation', 'test']},
        'protected': {str(path.relative_to(base.ROOT)): base.digest(path) for path in [
            base.RUN/'protocol.json', base.RUN/'selection.json', base.RUN/'final_results.json',
            base.ROOT/'experiments/gloflo_learners_v1/selection.json',
            base.ROOT/'experiments/gloflo_learners_v1/final_results.json']},
        'versions': {**old['versions'], 'catboost': catboost.__version__},
        'limits': 'Previously inspected validation/test: exploratory follow-up, not fresh independent confirmation. '
                  'Contemporary static catchment/climate proxies are not historically issued weather forecasts. '
                  'The label is high annual peak discharge, not verified flood occurrence. '
                  'Unseen stations can remain spatially related to training catchments.',
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    base.dump(RUN/'protocol.json', protocol)
    print('Frozen station/time folds', folds, flush=True)


def protocol():
    p = base.load(RUN/'protocol.json')
    for path, digest in p['protected'].items():
        assert base.digest(base.ROOT/path) == digest
    assert base.digest(base.ROOT/'flood_model.pkl') == p['original_model_sha256']
    assert base.digest(base.ROOT/'data/raw/kuntla_flooddatabase.csv') == p['source_sha256']
    return p


def data(split, p):
    path = RUN/f'data/{split}.csv'
    assert base.digest(path) == p['split_hashes'][split]
    return pd.read_csv(path, low_memory=False)


def preprocessing(features, native=False):
    numeric = [c for c in features if c not in CATEGORIES]
    cats = [c for c in features if c in CATEGORIES]
    return ColumnTransformer([
        ('numeric', Pipeline([('median', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scale', StandardScaler())]), numeric),
        ('category', 'passthrough' if native else OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
    ], sparse_threshold=0)


def grouped_importance(prep, model, features):
    numeric = [c for c in features if c not in CATEGORIES]
    cats = [c for c in features if c in CATEGORIES]
    values = model.feature_importances_
    result = {c: float(values[i]) for i, c in enumerate(numeric)}
    offset = len(numeric)
    for col, levels in zip(cats, prep.named_transformers_['category'].categories_):
        result[col] = float(values[offset:offset+len(levels)].sum())
        offset += len(levels)
    return result


def feature_views(train, y):
    kept, dropped = list(base.FEATURES), {}
    for feature in ADDITIONS:
        series = train[feature]
        if series.isna().mean() > .4:
            dropped[feature] = 'training missing fraction > 0.40'
        elif series.nunique(dropna=True) < 2:
            dropped[feature] = 'constant in training'
        elif feature not in CATEGORIES:
            previous = [c for c in kept if c not in CATEGORIES]
            corr = train[previous].corrwith(series).abs()
            if corr.max() > .98:
                dropped[feature] = 'training Pearson correlation > 0.98 with '+str(corr.idxmax())
            else:
                kept.append(feature)
        else:
            kept.append(feature)
    prep = preprocessing(kept)
    x = prep.fit_transform(train[kept])
    counts = train.gsim_code.value_counts()
    sample_weight = train.gsim_code.map(lambda code: 1./counts[code]).to_numpy(copy=True)
    sample_weight /= sample_weight.mean()
    selector = RandomForestClassifier(n_estimators=120, max_depth=12, max_leaf_nodes=512,
                                      min_samples_leaf=5, class_weight='balanced', n_jobs=2, random_state=base.SEED)
    selector.fit(x, y, sample_weight=sample_weight)
    importance = grouped_importance(prep, selector, kept)
    additions = sorted([c for c in kept if c not in base.FEATURES], key=lambda c: (-importance[c], c))[:12]
    return {'legacy13': list(base.FEATURES), 'expanded': kept, 'selected': base.FEATURES+additions}, {
        'dropped': dropped, 'training_grouped_importance': importance, 'selected_additions': additions,
        'selection_fit_rows': len(train), 'selection_fit_stations': train.gsim_code.nunique()}


def make_model(learner, config, features, y):
    cfg = config.copy()
    weighting = cfg.pop('weight')
    weight = float(np.sqrt(np.sum(y == 0)/np.sum(y == 1))) if weighting == 'sqrt' else 1.
    if learner == 'catboost':
        numeric_count = sum(c not in CATEGORIES for c in features)
        model = CatBoostClassifier(**cfg, learning_rate=.05, l2_leaf_reg=8, loss_function='Logloss',
                                   class_weights=[1., weight], cat_features=list(range(numeric_count, len(features))),
                                   boosting_type='Ordered', has_time=True, thread_count=2,
                                   random_seed=base.SEED, verbose=False, allow_writing_files=False)
    elif learner == 'hist_gradient_boosting':
        model = HistGradientBoostingClassifier(**cfg, learning_rate=.05, min_samples_leaf=20,
                                               class_weight=None if weight == 1 else {0: 1., 1: weight},
                                               early_stopping=False, random_state=base.SEED)
    elif learner == 'random_forest':
        model = RandomForestClassifier(**cfg, class_weight=None if weight == 1 else {0: 1., 1: weight},
                                        max_features='sqrt', n_jobs=2, random_state=base.SEED)
    else:
        raise ValueError(learner)
    return Pipeline([('preprocessing', preprocessing(features, native=learner == 'catboost')), ('model', model)])


def threshold(y, scores):
    precision, recall, ts = precision_recall_curve(y, scores)
    f2 = np.divide(5*precision[:-1]*recall[:-1], 4*precision[:-1]+recall[:-1],
                   out=np.zeros_like(ts), where=(4*precision[:-1]+recall[:-1]) > 0)
    feasible = np.flatnonzero((precision[:-1] >= .5) & (recall[:-1] >= .75))
    options = feasible if len(feasible) else np.arange(len(ts))
    i = max(options, key=lambda j: (f2[j], precision[j], ts[j]))
    return {**base.metrics(y, scores, float(ts[i])), 'operating_constraints_met': bool(len(feasible))}


def candidate_key(item):
    return item['mean_fold_ap'], item['minimum_fold_ap'], item['oof_operating_point']['f2']


def select():
    if (RUN/'selection.json').exists():
        raise RuntimeError('Selection frozen. Do not retune.')
    p = protocol()
    train = data('train', p)
    groups = train.gsim_code.map(station_group)
    fold_data, feature_audits = [], {}
    for f in p['folds']:
        i = f['fold']
        a = train.loc[(train.numeric_year <= f['train_end']) & (groups != i)].copy()
        b = train.loc[train.numeric_year.between(f['validation_start'], f['validation_end']) & (groups == i)].copy()
        assert set(a.gsim_code).isdisjoint(b.gsim_code)
        assert a.numeric_year.max() < b.numeric_year.min()
        y = a.source_peak_discharge.gt(f['cutoff']).astype(int).to_numpy()
        yv = b.source_peak_discharge.gt(f['cutoff']).astype(int).to_numpy()
        views, audit = feature_views(a, y)
        feature_audits[str(i)] = {**audit, 'views': views}
        fold_data.append((a, b, y, yv, views))
    base.dump(RUN/'fold_feature_selection.json', feature_audits)
    candidates, best_cells = [], {}
    (RUN/'oof_predictions').mkdir(exist_ok=True)
    (RUN/'models').mkdir(exist_ok=True)
    for learner, configs in p['configs'].items():
        for view in p['feature_views']:
            for config_index, config in enumerate(configs):
                started = time.perf_counter()
                folds, oof = [], []
                for i, (a, b, y, yv, views) in enumerate(fold_data):
                    features = views[view]
                    pipe = make_model(learner, config, features, y)
                    pipe.fit(a[features], y)
                    scores = np.round(pipe.predict_proba(b[features])[:, 1], 12)
                    score = base.metrics(yv, scores, .5)
                    folds.append({'fold': i, 'features': features, 'metrics_at_0_5': score})
                    oof.append(pd.DataFrame({'source_row_index': b.source_row_index, 'gsim_code': b.gsim_code,
                                             'fold': i, 'target': yv, 'score': scores}))
                    print(f"CV {learner}/{view}/{config_index} fold {i+1}/3 AP={score['pr_auc_ap']:.4f}", flush=True)
                pooled = pd.concat(oof, ignore_index=True)
                point = threshold(pooled.target.to_numpy(), pooled.score.to_numpy())
                for f, frame in zip(folds, oof):
                    f['metrics_at_pooled_threshold'] = base.metrics(frame.target.to_numpy(), frame.score.to_numpy(), point['threshold'])
                candidate_id = f'{learner}__{view}__{config_index}'
                pooled.to_csv(RUN/f'oof_predictions/{candidate_id}.csv', index=False)
                item = {'id': candidate_id, 'learner': learner, 'view': view, 'config': config,
                        'folds': folds, 'mean_fold_ap': float(np.mean([f['metrics_at_0_5']['pr_auc_ap'] for f in folds])),
                        'minimum_fold_ap': min(f['metrics_at_0_5']['pr_auc_ap'] for f in folds),
                        'oof_operating_point': point, 'seconds': time.perf_counter()-started}
                candidates.append(item)
                cell = learner+'__'+view
                if cell not in best_cells or candidate_key(item) > candidate_key(best_cells[cell]):
                    best_cells[cell] = item
                base.dump(RUN/'candidate_progress.json', candidates)
                print(f"{len(candidates)}/18 {candidate_id}: mean AP={item['mean_fold_ap']:.4f}, "
                      f"OOF F2={point['f2']:.4f}, constraints={point['operating_constraints_met']}", flush=True)
    assert len(candidates) == p['candidate_count']
    winner = max(best_cells, key=lambda c: candidate_key(best_cells[c]))
    # Full-training selectors/models never fit on outer validation or test.
    views, audit = feature_views(train, train.target.to_numpy())
    base.dump(RUN/'full_training_feature_selection.json', {**audit, 'views': views})
    for cell, item in best_cells.items():
        features = views[item['view']]
        pipe = make_model(item['learner'], item['config'], features, train.target.to_numpy())
        pipe.fit(train[features], train.target.to_numpy())
        joblib.dump(pipe, RUN/f'models/{cell}.joblib', compress=3)
        item['features'] = features
        item['artifact_sha256'] = base.digest(RUN/f'models/{cell}.joblib')
        if item['learner'] == 'catboost':
            pipe.named_steps['model'].save_model(str(RUN/f'models/{cell}.cbm'))
        print('Saved full-training pipeline', cell, len(features), 'features', flush=True)
    selection = {'selected_cell': winner, 'cells': best_cells,
                 'selected_by': p['selection'], 'protocol_sha256': base.digest(RUN/'protocol.json'),
                 'candidate_count': len(candidates), 'cv_fits': p['cv_fits'],
                 'outer_validation_used_for_selection': False, 'test_used_for_selection': False,
                 'selected_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN/'validation_candidates.json', candidates)
    base.dump(RUN/'selection.json', selection)
    print('FROZEN WINNER', winner, best_cells[winner]['mean_fold_ap'],
          best_cells[winner]['oof_operating_point'], flush=True)


def split_evaluation(frame, train, scores, t):
    unseen = ~frame.gsim_code.isin(train.gsim_code)
    result = {'all': base.metrics(frame.target.to_numpy(), scores, t)}
    for name, mask in [('seen_stations', ~unseen), ('unseen_stations', unseen)]:
        result[name] = base.metrics(frame.loc[mask, 'target'].to_numpy(), scores[mask], t)
    return result


def evaluate():
    if (RUN/'final_results.json').exists():
        raise RuntimeError('Evaluation frozen; do not retune or overwrite.')
    p = protocol()
    s = base.load(RUN/'selection.json')
    assert s['protocol_sha256'] == base.digest(RUN/'protocol.json')
    train = data('train', p)
    results, predictions = {}, {}
    for split in ['validation', 'test']:
        frame = data(split, p)
        records = frame[['source_row_index', 'gsim_code', 'numeric_year', 'target']].copy()
        results[split] = {}
        for cell, item in s['cells'].items():
            path = RUN/f'models/{cell}.joblib'
            assert base.digest(path) == item['artifact_sha256']
            pipe = joblib.load(path)
            scores = np.round(pipe.predict_proba(frame[item['features']])[:, 1], 12)
            t = item['oof_operating_point']['threshold']
            results[split][cell] = split_evaluation(frame, train, scores, t)
            records[cell+'_score'] = scores
            records[cell+'_prediction'] = (scores >= t).astype(int)
        records.to_csv(RUN/f'{split}_predictions.csv', index=False)
        predictions[split] = records
    old_pred = pd.read_csv(base.ROOT/'experiments/gloflo_learners_v1/test_predictions.csv')
    old_result = base.load(base.ROOT/'experiments/gloflo_learners_v1/final_results.json')
    old_cell = old_result['selected_cell']
    test = data('test', p)
    unseen = ~test.gsim_code.isin(train.gsim_code)
    assert np.array_equal(old_pred.source_row_index, test.source_row_index)
    chosen = s['selected_cell']
    result = {'selected_cell': chosen, 'selection_sha256': base.digest(RUN/'selection.json'),
              'metrics': results, 'previous_selected': {
                  'cell': old_cell, 'all': old_result['test_metrics'][old_cell],
                  'unseen_stations': old_result['selected_subgroups']['unseen_stations']},
              'paired_station_bootstrap_f1_95ci': paired_bootstrap(
                  test, predictions['test'][chosen+'_prediction'].to_numpy(),
                  old_pred[old_cell+'_prediction'].to_numpy(), unseen.to_numpy()),
              'test_fitting_or_threshold_tuning': False, 'already_examined_data': True,
              'original_app_model_unchanged': True, 'evaluated_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN/'final_results.json', result)
    selected = s['cells'][chosen]
    base.dump(RUN/'selected_model_metadata.json', {
        'pipeline': f'models/{chosen}.joblib', 'pipeline_sha256': selected['artifact_sha256'],
        'features': selected['features'], 'target': p['task'], 'training_discharge_cutoff': p['cutoff'],
        'operating_threshold': selected['oof_operating_point']['threshold'], 'score_rounding_decimals': 12,
        'cv_mean_average_precision': selected['mean_fold_ap'],
        'oof_operating_point': selected['oof_operating_point'],
        'config': selected['config'], 'selection': p['selection'], 'threshold_selection': p['threshold'],
        'test_metrics': result['metrics']['test'][chosen], 'limitations': p['limits'],
        'deployed_to_application': False})
    rows = []
    for split, cells in results.items():
        for cell, groups in cells.items():
            for group, score in groups.items():
                rows.append({'split': split, 'cell': cell, 'group': group,
                             'cv_mean_ap': s['cells'][cell]['mean_fold_ap'], **score})
    pd.DataFrame(rows).to_csv(RUN/'comparison.csv', index=False)
    write_report(p, s, result)
    plot_results(s, result)
    protocol()
    print('FINAL', chosen, result['metrics']['test'][chosen],
          'INTERVALS', result['paired_station_bootstrap_f1_95ci'], flush=True)


def paired_bootstrap(test, chosen, previous, unseen):
    rng = np.random.default_rng(base.SEED)
    y, stations = test.target.to_numpy(), test.gsim_code.to_numpy()
    intervals = {}
    for name, mask in [('all', np.ones(len(test), dtype=bool)), ('unseen_stations', unseen)]:
        unique = np.unique(stations[mask])
        groups = {station: np.flatnonzero((stations == station) & mask) for station in unique}
        differences = []
        for _ in range(500):
            idx = np.concatenate([groups[station] for station in rng.choice(unique, len(unique), replace=True)])
            if len(np.unique(y[idx])) == 2:
                differences.append(f1_score(y[idx], chosen[idx], zero_division=0)-
                                   f1_score(y[idx], previous[idx], zero_division=0))
        intervals[name] = np.quantile(differences, [.025, .975]).tolist()
    return intervals


def write_report(p, s, r):
    chosen = s['selected_cell']
    selected = s['cells'][chosen]
    lines = ['# Expanded features and station-aware validation', '', p['limits'], '',
             '## Task and partitions', '',
             f"High annual peak discharge above the full-training 95th percentile ({p['cutoff']:.2f}, source units). "
             'This target is unchanged for final validation/test comparison.',
             'Train 1900–1999: 37,184 rows / 1,859 positives. Validation 2000–2005: 3,387 / 162. '
             'Test 2006–2015: 5,339 / 320. All source rows and final labels match previous studies.', '',
             '## Station-disjoint forward validation', '', p['fold_rule'], p['fold_labels'], '',
             '| Fold | Train through | Validate years | Train rows | Validation rows | Validation positives | Fold cutoff |',
             '|---|---:|---|---:|---:|---:|---:|']
    for f in p['folds']:
        lines.append(f"| {f['fold']} | {f['train_end']} | {f['validation_start']}–{f['validation_end']} | "
                     f"{f['train_rows']} | {f['validation_rows']} | {f['positive_validation']} | {f['cutoff']:.2f} |")
    lines += ['', 'No station appears on both sides of a fold; every training year precedes its validation years. '
              'Station groups and year windows are fixed independently of outcomes. These folds live entirely '
              'inside the original training partition. They do not constitute a spatially separated basin benchmark.', '',
              '## Features and model selection', '', p['feature_selection'],
              'Three views: legacy13, expanded after training-only missingness/correlation filtering, and selected '
              '(legacy13 plus at most 12 training-ranked additions). Extra candidates include drainage area, '
              'basin length, sinuosity, stream order and seasonal climate attributes. '
              'Raw discharge, unit discharge, station identifiers, location identifiers and dam counts are excluded.',
              'CatBoost uses native categorical inputs, ordered boosting and has_time=True on chronologically '
              'sorted training rows. Random Forest and HistGradientBoosting use training-fitted medians/scaling '
              'and one-hot categories. No random internal early stopping, SMOTE or held-out fitting is used.',
              'Eighteen candidates × three folds = 54 CV model fits, plus four training-only feature selectors '
              'and nine final training-only pipelines. The two configurations per learner vary complexity '
              'and weighting; this is a bounded search, not exhaustive optimisation.',
              p['selection'], p['threshold'],
              'The unseen-station outer validation subgroup has only 222 rows / 18 positives. '
              'It is reported as a separate check; neither it nor final test is used to choose models or thresholds.', '',
              f"**Frozen winner: {chosen}**, mean forward/station-disjoint AP {selected['mean_fold_ap']:.4f}, "
              f"OOF threshold {selected['oof_operating_point']['threshold']:.6f}, "
              f"OOF operating constraints met: {selected['oof_operating_point']['operating_constraints_met']}.",
              f"Final selected inputs ({len(selected['features'])}): "+', '.join(selected['features'])+'.', '',
              '## Chronological test: all stations', '',
              '| Learner / feature view | CV mean AP | Precision | Recall | F1 | F2 | PR AP | ROC-AUC | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cell in sorted(s['cells'], key=lambda c: candidate_key(s['cells'][c]), reverse=True):
        m = r['metrics']['test'][cell]['all']
        lines.append(f"| {cell} | {s['cells'][cell]['mean_fold_ap']:.4f} | "+
                     ' | '.join(f'{m[k]:.4f}' for k in ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap', 'roc_auc'])+
                     f" | {m['fp']} | {m['fn']} |")
    lines += ['', '## Chronological test: unseen stations', '',
              'Same 494 test rows / 35 positives as the earlier failure. No test stations are removed.', '',
              '| Learner / feature view | Precision | Recall | F1 | F2 | AP | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cell in sorted(s['cells'], key=lambda c: candidate_key(s['cells'][c]), reverse=True):
        m = r['metrics']['test'][cell]['unseen_stations']
        lines.append('| '+cell+' | '+' | '.join(f'{m[k]:.4f}' for k in
                     ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap'])+f" | {m['fp']} | {m['fn']} |")
    new, old = r['metrics']['test'][chosen], r['previous_selected']
    lines += ['', '## Selected model versus the previous frozen winner', '',
              '| Population / model | Precision | Recall | F1 | F2 | AP | TP | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['all', 'unseen_stations']:
        for label, m in [('previous', old[group]), ('new', new[group])]:
            lines.append('| '+group+' / '+label+' | '+' | '.join(f'{m[k]:.4f}' for k in
                         ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap'])+f" | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines += ['', 'Both the feature/model selection protocol and operating objective changed. '
              'Do not attribute the entire before/after difference to features or to CatBoost alone. '
              'Legacy13 controls in this experiment help distinguish feature effects under the same protocol.',
              f"**Assessment:** selected-model recall rises to {new['all']['recall']:.4f}, but precision "
              f"falls to {new['all']['precision']:.4f}; false positives rise from {old['all']['fp']} "
              f"to {new['all']['fp']} and all-station F1 falls from {old['all']['f1']:.4f} "
              f"to {new['all']['f1']:.4f}. This is not an overall fix. The precision operating target "
              'was not met in cross-validation; the candidate is retained as an experiment, not deployed.',
              f"Unseen-station average precision also falls from {old['unseen_stations']['pr_auc_ap']:.4f} "
              f"to {new['unseen_stations']['pr_auc_ap']:.4f}. Recovering more positives at a lower threshold "
              'does not establish better ranking or generalisation. Within the new protocol, expanded '
              'HistGradientBoosting reduces unseen-station false positives from 72 to 40 versus its '
              'legacy13 control at the same 29 detected positives, but its ranking AP is lower. '
              'The feature procedure and threshold selection have mixed effects.',
              f"Paired station-bootstrap 95% F1-difference intervals (new minus previous): "
              f"{r['paired_station_bootstrap_f1_95ci']}. These are exploratory intervals on previously examined data.", '',
              '## Untuned outer validation check', '',
              '| Selected model population | Rows | Positives | Precision | Recall | F1 | F2 | AP |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, m in r['metrics']['validation'][chosen].items():
        lines.append(f"| {group} | {m['n']} | {m['class_1']} | "+
                     ' | '.join(f'{m[k]:.4f}' for k in ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap'])+' |')
    lines += ['', '## Limits and artifacts', '',
              'Training-derived operating targets are not promises for future recall or precision. '
              'Minority and unseen-station samples are limited. Weighted scores are not established '
              'as calibrated flood probabilities. Static attributes cannot establish event timing. '
              'More useful catchment descriptors can improve high-discharge classification without '
              'turning these data into verified flood/non-flood observations.',
              'All nine inference pipelines, native CatBoost models, metadata, fold manifests, selected inputs, '
              'fold feature audits, out-of-fold predictions, outer validation/test predictions and metrics are saved. '
              'Previous studies and the website model are unchanged.', '',
              '## Reproduction', '', 'Install requirements-training.txt. Frozen stages refuse overwrite. '
              'Use a new prespecified version for another experiment.', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_station_models.py prepare',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_station_models.py select',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_station_models.py evaluate',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_station_models.py verify', '```', '',
              'References: [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common), '
              '[group/time validation](https://scikit-learn.org/stable/modules/cross_validation.html).', '']
    (RUN/'MODEL_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_results(s, r):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(RUN/'plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells = sorted(s['cells'], key=lambda c: candidate_key(s['cells'][c]), reverse=True)
    labels = [c.replace('hist_gradient_boosting', 'HistGB').replace('random_forest', 'RF').replace(
        'catboost', 'CatBoost').replace('__', ' / ') for c in cells]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, group, title in zip(axes, ['all', 'unseen_stations'], ['All test stations', 'Unseen test stations']):
        positions = np.arange(len(cells))
        for shift, metric, color in [(-.24, 'precision', '#176b87'), (0, 'recall', '#19a7a0'), (.24, 'f1', '#e9a23b')]:
            ax.barh(positions+shift, [r['metrics']['test'][c][group][metric] for c in cells],
                    height=.22, label=metric.title(), color=color)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Expanded-feature comparison — order fixed by station-aware CV average precision')
    fig.savefig(RUN/'model_comparison.png', dpi=170)
    plt.close(fig)


def verify():
    p = protocol()
    s, r = base.load(RUN/'selection.json'), base.load(RUN/'final_results.json')
    assert base.digest(RUN/'selection.json') == r['selection_sha256']
    assert base.digest(RUN/'protocol.json') == s['protocol_sha256']
    candidates = base.load(RUN/'validation_candidates.json')
    assert len(candidates) == p['candidate_count'] == 18
    assert candidate_key(max(candidates, key=candidate_key)) == candidate_key(s['cells'][s['selected_cell']])
    train = data('train', p)
    manifest = pd.read_csv(RUN/'data/station_time_fold_manifest.csv')
    for _, fold in manifest.groupby('fold'):
        a, b = fold.loc[fold.role == 'train'], fold.loc[fold.role == 'validation']
        assert set(a.gsim_code).isdisjoint(b.gsim_code)
        assert a.numeric_year.max() < b.numeric_year.min()
        assert set(fold.source_row_index).issubset(set(train.source_row_index))
    test = data('test', p)
    saved = pd.read_csv(RUN/'test_predictions.csv')
    assert np.array_equal(saved.source_row_index, test.source_row_index)
    assert np.array_equal(saved.target, test.target)
    max_error = 0.
    for cell, item in s['cells'].items():
        path = RUN/f'models/{cell}.joblib'
        assert base.digest(path) == item['artifact_sha256']
        assert set(item['features']).issubset(ALL_FEATURES)
        assert not set(item['features']).intersection(['source_peak_discharge', 'target', 'Unit peak discharge', 'gsim_code'])
        pipe = joblib.load(path)
        num = [c for c in item['features'] if c not in CATEGORIES]
        medians = pipe.named_steps['preprocessing'].named_transformers_['numeric'].named_steps['median'].statistics_
        np.testing.assert_allclose(medians, train[num].median().fillna(0).to_numpy())
        scores = np.round(pipe.predict_proba(test[item['features']])[:, 1], 12)
        error = float(np.max(np.abs(scores-saved[cell+'_score'].to_numpy())))
        assert error < 1e-7
        assert np.array_equal(scores >= item['oof_operating_point']['threshold'], saved[cell+'_prediction'])
        max_error = max(max_error, error)
    checks = {'pipelines_verified': 9, 'cv_candidates_verified': 18, 'station_time_folds_verified': 3,
              'outcome_features_excluded': True, 'train_only_medians_verified': True,
              'largest_saved_score_difference': max_error, 'previous_studies_and_app_preserved': True}
    base.dump(RUN/'verification.json', checks)
    print(checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'prepare': prepare, 'select': select, 'evaluate': evaluate, 'verify': verify}[args.stage]()
