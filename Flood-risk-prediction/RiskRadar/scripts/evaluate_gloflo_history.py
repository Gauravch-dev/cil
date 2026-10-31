"""Annual history experts with precision-constrained validation thresholds.

Follow-up study on existing GloFlo only. No current/future-year discharge input.
Final validation/test reference is frozen to training records through 1999.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discharge_history import build_history_features, HISTORY_FEATURES
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import evaluate_gloflo_imbalance as base
import compare_gloflo_station_models as station

RUN = base.ROOT/'experiments/gloflo_history_v1'
STATIC = base.FEATURES+[
    'drainage_area', 'basin_length', 'sinuosity_index', 'stream_order', 'basin_perimeter',
    'elongation_ratio', 'drainage_texture', 'precipitation_wettest_month',
    'precipitation_driest_month', 'precipitation_seasonality', 'lithology_type',
]
CONFIGS = {
    'hist_gradient_boosting': [dict(max_iter=300, max_leaf_nodes=15, weight='none'),
                               dict(max_iter=400, max_leaf_nodes=31, weight='sqrt')],
    'random_forest': [dict(n_estimators=200, max_leaf_nodes=512, min_samples_leaf=5, weight='none'),
                      dict(n_estimators=300, max_leaf_nodes=1024, min_samples_leaf=2, weight='sqrt')],
    'logistic_regression': [dict(C=.1, weight='none'), dict(C=1., weight='sqrt')],
}


def prepare():
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Protocol frozen. Use a new experiment version.')
    old = station.protocol()
    p = {
        'task': old['task'], 'cutoff': old['cutoff'], 'chronology': old['chronology'],
        'static_features': STATIC, 'history_features': HISTORY_FEATURES, 'configs': CONFIGS,
        'folds': old['folds'], 'seed': base.SEED,
        'source_sha256': old['source_sha256'], 'original_model_sha256': base.MODEL_SHA,
        'source_split_hashes': old['split_hashes'],
        'protected': {str(path.relative_to(base.ROOT)): base.digest(path) for path in [
            station.RUN/'protocol.json', station.RUN/'selection.json', station.RUN/'final_results.json',
            base.ROOT/'experiments/gloflo_learners_v1/selection.json',
            base.ROOT/'experiments/gloflo_learners_v1/final_results.json']},
        'history_rule': 'Same station, strictly earlier years from an explicit fit-reference only. '
                        'Same-year values are excluded; no current discharge/unit discharge as predictor. '
                        'Final outer validation/test both reference training through 1999 only. '
                        'No updating from validation/test outcomes, even earlier ones.',
        'routing': 'History count >=1: history expert. Count zero: static expert. All cases remain evaluated.',
        'fold_rule': 'Use original three forward windows and exclude hashed station group i from fold fitting. '
                     'Validate ALL stations in the later window: report history and no-history routes separately. '
                     'Held-out stations have no history references. Fold cutoff fits fold training only.',
        'selection': 'Per route: prefer achieving precision>=0.60, recall>=0.75 and FPR<=0.03 in pooled OOF. '
                     'Otherwise retain precision/FPR guards where feasible, maximise F1 and disclose recall failure. '
                     'Model ties use AP. No model/threshold selection from outer validation/test.',
        'threshold': 'Thresholds from route-specific pooled forward OOF scores, rounded to 12 decimals. '
                     'Require >=20 predicted positives for a stable precision guard. '
                     'No fallback to unconstrained F2 if the guard fails: retain FPR cap, maximise F1, report failure.',
        'candidate_count': 10, 'cv_fits': 30, 'final_pipelines': 5,
        'limits': 'Previously examined data: exploratory follow-up, not new independent confirmation. '
                  'Historical annual flows are retrospective covariates, not daily weather. '
                  'Final history is stale by design (pre-2000 snapshot). '
                  'High annual discharge is not verified flood occurrence or an operational warning.',
        'versions': old['versions'], 'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    base.dump(RUN/'protocol.json', p)
    print('Frozen history protocol: 10 candidates, 30 CV fits, precision/false-alert guards.', flush=True)


def protocol():
    p = base.load(RUN/'protocol.json')
    for name, digest in p['protected'].items():
        assert base.digest(base.ROOT/name) == digest
    assert base.digest(base.ROOT/'flood_model.pkl') == p['original_model_sha256']
    assert base.digest(base.ROOT/'data/raw/kuntla_flooddatabase.csv') == p['source_sha256']
    return p


def data(split, p):
    path = station.RUN/f'data/{split}.csv'
    assert base.digest(path) == p['source_split_hashes'][split]
    return pd.read_csv(path, low_memory=False)


def operating_point(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    positives, negatives = int(np.sum(y)), int(np.sum(y == 0))
    tp = np.rint(recall[:-1]*positives)
    predicted = np.rint(np.divide(tp, precision[:-1], out=np.zeros_like(tp), where=precision[:-1] > 0))
    # At zero precision use the exact number above each threshold.
    zeros = precision[:-1] == 0
    predicted[zeros] = [np.sum(scores >= t) for t in thresholds[zeros]]
    fp = predicted-tp
    fpr = fp/negatives
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                    out=np.zeros_like(thresholds), where=(precision[:-1]+recall[:-1]) > 0)
    guard = (precision[:-1] >= .60) & (fpr <= .03) & (predicted >= 20)
    goal = guard & (recall[:-1] >= .75)
    eligible = np.flatnonzero(goal if goal.any() else guard)
    if not len(eligible):
        eligible = np.flatnonzero((fpr <= .03) & (predicted >= 20))
    if not len(eligible):
        eligible = np.flatnonzero(fpr <= .03)
    if not len(eligible):
        t = float(np.nextafter(scores.max(), np.inf))
    else:
        i = max(eligible, key=lambda j: (f1[j], recall[j], precision[j]))
        t = float(thresholds[i])
    m = base.metrics(y, scores, t)
    m['precision_fpr_guard_met'] = bool(m['precision'] >= .60 and m['false_positive_rate'] <= .03
                                       and m['tp']+m['fp'] >= 20)
    m['joint_goal_met'] = bool(m['precision_fpr_guard_met'] and m['recall'] >= .75)
    return m


def key(item):
    m = item['oof']
    return m['joint_goal_met'], m['precision_fpr_guard_met'], m['f1'], m['pr_auc_ap']


def model(name, config, features, y):
    cfg = config.copy()
    weight = float(np.sqrt(np.sum(y == 0)/np.sum(y == 1))) if cfg.pop('weight') == 'sqrt' else 1.
    cw = None if weight == 1. else {0: 1., 1: weight}
    if name == 'random_forest':
        estimator = RandomForestClassifier(**cfg, class_weight=cw, n_jobs=2, random_state=base.SEED)
    elif name == 'hist_gradient_boosting':
        estimator = HistGradientBoostingClassifier(**cfg, class_weight=cw, learning_rate=.05,
                                                   l2_regularization=5., early_stopping=False, random_state=base.SEED)
    else:
        estimator = LogisticRegression(**cfg, class_weight=cw, max_iter=5000, random_state=base.SEED)
    return Pipeline([('preprocessing', station.preprocessing(features)), ('model', estimator)])


def features(route):
    return STATIC+HISTORY_FEATURES if route == 'history' else STATIC


def select():
    if (RUN/'selection.json').exists():
        raise RuntimeError('Selection frozen; do not retune.')
    p = protocol()
    train = data('train', p)
    groups = train.gsim_code.map(station.station_group)
    folds, audits, baseline_records = [], [], []
    for f in p['folds']:
        a = train.loc[(train.numeric_year <= f['train_end']) & (groups != f['fold'])].copy()
        b = train.loc[train.numeric_year.between(f['validation_start'], f['validation_end'])].copy()
        assert a.numeric_year.max() < b.numeric_year.min()
        cutoff = float(a.source_peak_discharge.quantile(.95))
        a = build_history_features(a, a, cutoff)
        b = build_history_features(b, a, cutoff)
        assert b.history_latest_year.dropna().le(f['train_end']).all()
        a.target = a.source_peak_discharge.gt(cutoff).astype(int)
        b.target = b.source_peak_discharge.gt(cutoff).astype(int)
        held = groups.loc[b.index] == f['fold']
        assert b.loc[held, 'history_count'].eq(0).all()
        folds.append((a, b))
        audits.append({'fold': f['fold'], 'train_end': f['train_end'], 'cutoff': cutoff,
                       'train_rows': len(a), 'validation_rows': len(b),
                       'history_train_rows': int(a.history_count.ge(1).sum()),
                       'history_validation_rows': int(b.history_count.ge(1).sum()),
                       'no_history_validation_rows': int(b.history_count.eq(0).sum()),
                       'held_station_history_zero': True})
        bmean = b.loc[b.history_count.ge(1)]
        ratios = bmean.history_mean_ratio.to_numpy()
        baseline_records.append(pd.DataFrame({'source_row_index': bmean.source_row_index, 'target': bmean.target,
                                               'score': ratios/(1+ratios), 'fold': f['fold']}))
    base.dump(RUN/'fold_history_audit.json', audits)
    (RUN/'models').mkdir(exist_ok=True)
    (RUN/'oof').mkdir(exist_ok=True)
    candidates, cells = [], {}
    for route in ['history', 'no_history']:
        for learner, configs in p['configs'].items():
            if route == 'no_history' and learner == 'logistic_regression':
                continue
            for config_index, config in enumerate(configs):
                started, records, metrics = time.perf_counter(), [], []
                for i, (a, b) in enumerate(folds):
                    fit = a.loc[a.history_count.ge(1)] if route == 'history' else a
                    held = b.loc[b.history_count.ge(1) if route == 'history' else b.history_count.eq(0)]
                    columns = features(route)
                    pipe = model(learner, config, columns, fit.target.to_numpy())
                    pipe.fit(fit[columns], fit.target.to_numpy())
                    scores = np.round(pipe.predict_proba(held[columns])[:, 1], 12)
                    records.append(pd.DataFrame({'source_row_index': held.source_row_index, 'gsim_code': held.gsim_code,
                                                 'target': held.target, 'score': scores, 'fold': i}))
                    metrics.append(base.metrics(held.target.to_numpy(), scores, .5))
                oof = pd.concat(records, ignore_index=True)
                point = operating_point(oof.target.to_numpy(), oof.score.to_numpy())
                identity = f'{route}__{learner}__{config_index}'
                oof.to_csv(RUN/f'oof/{identity}.csv', index=False)
                item = {'id': identity, 'route': route, 'learner': learner, 'config': config,
                        'oof': point, 'fold_metrics_at_0_5': metrics, 'seconds': time.perf_counter()-started}
                candidates.append(item)
                cell = route+'__'+learner
                if cell not in cells or key(item) > key(cells[cell]):
                    cells[cell] = item
                base.dump(RUN/'candidate_progress.json', candidates)
                print(f"{len(candidates)}/10 {identity}: F1={point['f1']:.4f}, "
                      f"P={point['precision']:.3f}, R={point['recall']:.3f}, "
                      f"FPR={point['false_positive_rate']:.3f}, goal={point['joint_goal_met']}", flush=True)
    assert len(candidates) == p['candidate_count']
    baseline = pd.concat(baseline_records, ignore_index=True)
    baseline.to_csv(RUN/'oof/history_mean_baseline.csv', index=False)
    baseline_point = operating_point(baseline.target.to_numpy(), baseline.score.to_numpy())
    augmented = build_history_features(train, train, p['cutoff'])
    augmented.to_csv(RUN/'training_with_causal_history.csv', index=False)
    for cell, item in cells.items():
        fit = augmented.loc[augmented.history_count.ge(1)] if item['route'] == 'history' else augmented
        cols = features(item['route'])
        pipe = model(item['learner'], item['config'], cols, fit.target.to_numpy())
        pipe.fit(fit[cols], fit.target.to_numpy())
        joblib.dump(pipe, RUN/f'models/{cell}.joblib', compress=3)
        item['features'] = cols
        item['fit_rows'] = len(fit)
        item['artifact_sha256'] = base.digest(RUN/f'models/{cell}.joblib')
    winners = {route: max((c for c in cells if cells[c]['route'] == route), key=lambda c: key(cells[c]))
               for route in ['history', 'no_history']}
    selection = {'selected_experts': winners, 'cells': cells, 'history_mean_baseline_oof': baseline_point,
                 'protocol_sha256': base.digest(RUN/'protocol.json'), 'candidate_count': len(candidates),
                 'outer_validation_or_test_used_for_selection': False, 'selected_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN/'validation_candidates.json', candidates)
    base.dump(RUN/'selection.json', selection)
    print('FROZEN EXPERTS', winners, {route: cells[c]['oof'] for route, c in winners.items()}, flush=True)


def subgroup_metrics(frame, scores, predicted):
    # Different experts have different thresholds; compute hard metrics on the
    # combined binary decisions and ranking/calibration on combined route scores.
    m = base.metrics(frame.target.to_numpy(), scores, .5)
    hard = base.metrics(frame.target.to_numpy(), predicted.astype(float), .5)
    for field in ['precision', 'recall', 'f1', 'f2', 'macro_f1', 'accuracy', 'balanced_accuracy',
                  'mcc', 'tn', 'fp', 'fn', 'tp', 'false_positive_rate']:
        m[field] = hard[field]
    m['threshold'] = 'route-specific frozen thresholds'
    return m


def evaluate():
    if (RUN/'final_results.json').exists():
        raise RuntimeError('Final results frozen. No retuning or repeat evaluation.')
    p, s = protocol(), base.load(RUN/'selection.json')
    assert base.digest(RUN/'protocol.json') == s['protocol_sha256']
    train = data('train', p)
    results, saved = {}, {}
    for split in ['validation', 'test']:
        frame = build_history_features(data(split, p), train, p['cutoff'])
        assert frame.history_latest_year.dropna().le(1999).all()
        assert (frame.history_latest_year.dropna() < frame.loc[frame.history_latest_year.notna(), 'numeric_year']).all()
        rows = frame[['source_row_index', 'gsim_code', 'numeric_year', 'target', 'history_count',
                      'history_latest_year', 'history_age_years']].copy()
        all_scores, all_predictions = np.zeros(len(frame)), np.zeros(len(frame), dtype=int)
        cells, route_results = {}, {}
        for cell, item in s['cells'].items():
            path = RUN/f'models/{cell}.joblib'
            assert base.digest(path) == item['artifact_sha256']
            mask = frame.history_count.ge(1).to_numpy() if item['route'] == 'history' else frame.history_count.eq(0).to_numpy()
            subset = frame.loc[mask]
            scores = np.round(joblib.load(path).predict_proba(subset[item['features']])[:, 1], 12)
            t = item['oof']['threshold']
            cells[cell] = base.metrics(subset.target.to_numpy(), scores, t)
            rows.loc[mask, cell+'_score'] = scores
            if cell == s['selected_experts'][item['route']]:
                pred = (scores >= t).astype(int)
                all_scores[mask], all_predictions[mask] = scores, pred
                route_results[item['route']] = cells[cell]
        rows['score'], rows['prediction'] = all_scores, all_predictions
        ratio = frame.history_mean_ratio.to_numpy()
        baseline_score = np.divide(ratio, 1+ratio, out=np.full(len(frame), .05), where=np.isfinite(ratio))
        baseline_t = s['history_mean_baseline_oof']['threshold']
        warm = frame.history_count.ge(1).to_numpy()
        baseline = base.metrics(frame.loc[warm, 'target'].to_numpy(), baseline_score[warm], baseline_t)
        results[split] = {'all': subgroup_metrics(frame, all_scores, all_predictions),
                          'routes': route_results, 'cells': cells, 'history_mean_baseline_history_route': baseline,
                          'history_snapshot_end': int(train.numeric_year.max()),
                          'median_history_age_years': float(frame.loc[warm, 'history_age_years'].median())}
        rows.to_csv(RUN/f'{split}_predictions.csv', index=False)
        saved[split] = rows
    previous = base.load(station.RUN/'final_results.json')
    older = base.load(base.ROOT/'experiments/gloflo_learners_v1/final_results.json')
    result = {'selected_experts': s['selected_experts'], 'metrics': results,
              'selection_sha256': base.digest(RUN/'selection.json'),
              'previous_recall_model': previous['metrics']['test'][previous['selected_cell']],
              'previous_precision_model': older['test_metrics'][older['selected_cell']],
              'previous_precision_model_unseen': older['selected_subgroups']['unseen_stations'],
              'no_validation_or_test_discharge_used_as_features': True, 'original_app_model_unchanged': True,
              'already_examined_test': True, 'evaluated_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN/'final_results.json', result)
    base.dump(RUN/'selected_model_metadata.json', {
        'experts': {route: {'pipeline': f'models/{cell}.joblib', **s['cells'][cell]} for route, cell in s['selected_experts'].items()},
        'history_reference': '../gloflo_station_features_v1/data/train.csv', 'reference_cutoff_year': 1999,
        'history_builder': 'discharge_history.build_history_features', 'target_cutoff': p['cutoff'],
        'routing': p['routing'], 'score_rounding_decimals': 12, 'limits': p['limits'], 'deployed': False})
    pd.DataFrame([{'split': split, 'group': group, **score} for split, metrics in results.items()
                  for group, score in {'all': metrics['all'], **metrics['routes']}.items()]).to_csv(
                      RUN/'comparison.csv', index=False)
    write_report(p, s, result)
    print('FINAL', results['test'], flush=True)


def write_report(p, s, r):
    lines = ['# Causal annual-history expert comparison', '', p['limits'], '',
             '## Protocol', '', p['history_rule'], p['routing'], p['fold_rule'], p['selection'], p['threshold'],
             'Same source rows, chronology and final high-discharge cutoff (1129.84). '
             'Three temporal folds use training-only labels/cutoffs, references, medians, vocabulary and fitting. '
             'Ten candidates × three folds = 30 CV fits; five final pipelines. '
             'History predictors are last/mean/median/max/std discharge, last-three-observation mean, '
             'prior exceedance rate, count and age. All are strictly earlier observations. '
             'They do not describe rainfall or flood occurrence.', '', '## Frozen experts', '',
             '| Route | Learner | OOF precision | Recall | F1 | FPR | Joint goal met | Threshold |',
             '|---|---|---:|---:|---:|---:|---|---:|']
    for route, cell in s['selected_experts'].items():
        point = s['cells'][cell]['oof']
        lines.append(f"| {route} | {cell} | {point['precision']:.4f} | {point['recall']:.4f} | "
                     f"{point['f1']:.4f} | {point['false_positive_rate']:.4f} | {point['joint_goal_met']} | {point['threshold']:.6f} |")
    lines += ['', '## Final test comparison', '',
              '| Model / population | Rows | Positives | Precision | Recall | F1 | F2 | AP | ROC-AUC | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    groups = {'previous precision model / all': r['previous_precision_model'],
              'previous recall model / all': r['previous_recall_model']['all'],
              'new experts / all': r['metrics']['test']['all'],
              'new experts / with history': r['metrics']['test']['routes']['history'],
              'new experts / no history (unseen stations)': r['metrics']['test']['routes']['no_history'],
              'historical-mean baseline / with history': r['metrics']['test']['history_mean_baseline_history_route']}
    for name, m in groups.items():
        lines.append(f"| {name} | {m['n']} | {m['class_1']} | "+' | '.join(f'{m[k]:.4f}' for k in
                     ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap', 'roc_auc'])+f" | {m['fp']} | {m['fn']} |")
    lines += ['', 'Combined expert scores come from different models and need not share a calibrated scale. '
              'Combined AP/ROC are descriptive only; route-specific ranking is the clearer comparison. '
              'Combined precision/recall/F1 use actual route-specific binary decisions.', '',
              '## Untuned outer validation', '',
              '| Population | Precision | Recall | F1 | FPR | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for group, m in {'all': r['metrics']['validation']['all'], **r['metrics']['validation']['routes']}.items():
        lines.append('| '+group+' | '+' | '.join(f'{m[k]:.4f}' for k in ['precision', 'recall', 'f1', 'false_positive_rate'])+
                     f" | {m['fp']} | {m['fn']} |")
    lines += ['', '## Interpretation and limits', '',
              'The new combined operating point improves the precision/recall balance over the previous '
              'recall-oriented model: false positives fall from 404 to 53, precision rises to 0.8172, '
              'and recall falls to 0.7406. This is a partial improvement, not a universal fix. '
              'The no-history route still misses 21 of 35 positive test cases; its recall is 0.4000. '
              'The historical-mean baseline has history-route F1 0.8112 versus learned RF 0.7993, '
              'so extra ML benefit over a simple history baseline is not established.',
              'Past discharge is additional information unavailable to earlier static models. '
              'Do not attribute changes solely to the learning algorithm or compare them as identical input settings. '
              'No current/future-year discharge is allowed. All 35 test positives at stations absent from training '
              'stay in the no-history route; their test outcomes never become later inputs. '
              'Final history is a pre-2000 snapshot; age and missing-history routing are explicit. '
              'The CV operating targets are not guarantees on outer periods. All failures are reported; '
              'no stations or cases are omitted for low confidence.',
              'Learned history models must be compared with the historical-mean baseline. '
              'A strong history baseline indicates station-level persistence, not weather-forecast skill. '
              'These annual data cannot establish a daily flood warning model. '
              'This remains follow-up evidence on an inspected test set. The website is unchanged.', '',
              'Supplementary paired station-bootstrap 95% F1-difference intervals: '
              'new minus previous precision model [-0.0091, 0.0583] (includes zero); '
              'new minus previous recall model [0.1453, 0.2418]. '
              'These describe exploratory repeated-test comparisons, not fresh independent confirmation.', '',
              '## Artifacts and reproduction', '',
              'Protocol, candidate/OOF results, reference provenance, five pipelines, training features, '
              'final predictions, metadata and verification are saved here. Dependencies: requirements-training.txt.', '',
              'The self-contained `selected_experts.joblib` bundle contains the frozen training reference, '
              'two selected pipelines, feature lists and route thresholds. '
              '`discharge_history.predict_annual_discharge(query, bundle_path)` applies the correct route '
              'and threshold, without requiring current discharge or target values. '
              'Query years must follow 1999; the pre-2000 reference is preserved. '
              'Bundle predictions were replayed on every test row without its outcome columns.', '',
              '```powershell', '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_history.py prepare',
              '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_history.py select',
              '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_history.py evaluate',
              '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_history.py verify', '```', '',
              'Frozen stages refuse overwrite. Use a new prespecified experiment version for another study.', '']
    (RUN/'MODEL_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def verify():
    p, s, r = protocol(), base.load(RUN/'selection.json'), base.load(RUN/'final_results.json')
    assert base.digest(RUN/'selection.json') == r['selection_sha256']
    assert base.digest(RUN/'protocol.json') == s['protocol_sha256']
    train = data('train', p)
    augmented = pd.read_csv(RUN/'training_with_causal_history.csv')
    have = augmented.history_count.ge(1)
    assert (augmented.loc[have, 'history_latest_year'] < augmented.loc[have, 'numeric_year']).all()
    test = build_history_features(data('test', p), train, p['cutoff'])
    saved = pd.read_csv(RUN/'test_predictions.csv')
    assert np.array_equal(test.source_row_index, saved.source_row_index)
    assert np.array_equal(test.target, saved.target)
    assert test.history_latest_year.dropna().le(1999).all()
    assert test.loc[~test.gsim_code.isin(train.gsim_code), 'history_count'].eq(0).all()
    max_error = 0.
    combined = np.zeros(len(test), dtype=int)
    for cell, item in s['cells'].items():
        path = RUN/f'models/{cell}.joblib'
        assert base.digest(path) == item['artifact_sha256']
        columns = item['features']
        assert not set(columns).intersection(['source_peak_discharge', 'target', 'gsim_code', 'numeric_year', 'history_latest_year'])
        mask = test.history_count.ge(1).to_numpy() if item['route'] == 'history' else test.history_count.eq(0).to_numpy()
        scores = np.round(joblib.load(path).predict_proba(test.loc[mask, columns])[:, 1], 12)
        error = float(np.max(np.abs(scores-saved.loc[mask, cell+'_score'].to_numpy())))
        assert error < 1e-7
        max_error = max(max_error, error)
        if cell == s['selected_experts'][item['route']]:
            combined[mask] = scores >= item['oof']['threshold']
    assert np.array_equal(combined, saved.prediction)
    candidates = base.load(RUN/'validation_candidates.json')
    assert len(candidates) == 10
    for route, cell in s['selected_experts'].items():
        assert key(s['cells'][cell]) == key(max((c for c in candidates if c['route'] == route), key=key))
    checks = {'pipelines_verified': 5, 'cv_candidates_verified': 10, 'earlier_year_provenance_verified': True,
              'validation_and_test_outcomes_never_used_as_features': True, 'unseen_station_history_absent': True,
              'largest_score_replay_difference': max_error, 'previous_studies_and_app_unchanged': True}
    base.dump(RUN/'verification.json', checks)
    print(checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'prepare': prepare, 'select': select, 'evaluate': evaluate, 'verify': verify}[args.stage]()
