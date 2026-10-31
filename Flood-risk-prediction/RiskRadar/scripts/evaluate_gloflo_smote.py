"""Training-only SMOTENC extension of the frozen GloFlo imbalance comparison.

The existing test has been examined: this is a disclosed follow-up comparison,
not a new independent confirmation. Never modifies earlier frozen artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import imblearn
from imblearn.over_sampling import SMOTENC
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import evaluate_gloflo_imbalance as base

ROOT = base.ROOT
RUN = ROOT / 'experiments/gloflo_smotenc_v1'
PREVIOUS = base.RUN
NUMERIC = [f for f in base.FEATURES if f not in base.CATEGORIES]
CONFIGS = [
    dict(n_estimators=300, max_depth=3, learning_rate=.05, min_child_weight=10, reg_lambda=10, reg_alpha=1),
    dict(n_estimators=500, max_depth=4, learning_rate=.05, min_child_weight=10, reg_lambda=10, reg_alpha=1),
    dict(n_estimators=300, max_depth=5, learning_rate=.08, min_child_weight=5, reg_lambda=5, reg_alpha=1),
]


def prepare():
    if (RUN / 'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; use a new experiment version.')
    previous = base.load(PREVIOUS / 'protocol.json')
    assert base.digest(ROOT / 'flood_model.pkl') == base.MODEL_SHA
    protocol = {
        'task': previous['task'], 'cutoff': previous['cutoff'],
        'chronology': previous['chronology'], 'features': base.FEATURES,
        'categorical_features': base.CATEGORIES, 'seed': base.SEED,
        'sampling_ratios': [.25, .5, 1.], 'k_neighbors': [3, 5], 'model_configs': CONFIGS,
        'candidate_fits': 18, 'positive_class_weight': 1.,
        'selection': 'Validation positive F1, ties AP then precision; validation-only decision threshold.',
        'preprocessing': 'Train-only numeric medians and scaling for SMOTENC distances; valid categorical modes. '
                         'Inverse numeric scaling, then original train-fitted one-hot model preprocessing.',
        'sampling': 'SMOTENC synthesises training minority rows only; validation/test remain natural.',
        'target_limit': 'High annual peak discharge, not verified flood occurrence or a lead-time forecast.',
        'followup_limit': 'Same 2006–2015 test was already examined in imbalance_v1. '
                          'This is a follow-up comparison, not fresh independent confirmation. '
                          'No test metrics are used by this selection stage.',
        'source_sha256': previous['source_sha256'], 'split_hashes': previous['split_hashes'],
        'previous_protocol_sha256': base.digest(PREVIOUS / 'protocol.json'),
        'previous_selection_sha256': base.digest(PREVIOUS / 'selection.json'),
        'previous_results_sha256': base.digest(PREVIOUS / 'final_results.json'),
        'original_model_sha256': base.MODEL_SHA,
        'versions': {**previous['versions'], 'imbalanced_learn': imblearn.__version__},
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    base.dump(RUN / 'protocol.json', protocol)
    print('Frozen SMOTENC protocol: 18 candidates; original chronology and target retained.', flush=True)


def verify_protocol():
    protocol = base.load(RUN / 'protocol.json')
    assert base.digest(PREVIOUS / 'protocol.json') == protocol['previous_protocol_sha256']
    assert base.digest(PREVIOUS / 'selection.json') == protocol['previous_selection_sha256']
    assert base.digest(PREVIOUS / 'final_results.json') == protocol['previous_results_sha256']
    assert base.digest(ROOT / 'flood_model.pkl') == protocol['original_model_sha256']
    return protocol


def key(item):
    v = item['validation']
    return v['f1'], v['pr_auc_ap'], v['precision']


def select():
    if (RUN / 'selection.json').exists():
        raise RuntimeError('Selection already frozen; do not retune.')
    protocol = verify_protocol()
    # Never open test rows or previous test scores during training/selection.
    train = base.data('train', protocol)
    val = base.data('validation', protocol)
    X = train[base.FEATURES]
    y = train.target.to_numpy()
    scaler = Pipeline([('median', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    distance_data = X.copy()
    distance_data[NUMERIC] = scaler.fit_transform(X[NUMERIC])
    preprocess = base.preprocessing()
    preprocess.fit(X)
    xv = preprocess.transform(val[base.FEATURES])
    candidates = []
    best = None
    (RUN / 'models').mkdir(parents=True, exist_ok=True)
    for ratio in protocol['sampling_ratios']:
        for neighbors in protocol['k_neighbors']:
            started = time.perf_counter()
            sampler = SMOTENC(categorical_features=base.CATEGORIES, sampling_strategy=ratio,
                              k_neighbors=neighbors, random_state=base.SEED)
            resampled, yr = sampler.fit_resample(distance_data, y)
            # Originals appear first; no original rows or labels may be altered.
            np.testing.assert_allclose(resampled.iloc[:len(X)][NUMERIC], distance_data[NUMERIC])
            assert resampled.iloc[:len(X)][base.CATEGORIES].equals(distance_data[base.CATEGORIES])
            assert np.array_equal(yr[:len(y)], y)
            assert np.sum(yr == 0) == np.sum(y == 0)
            for category in base.CATEGORIES:
                assert set(resampled[category]).issubset(set(X[category]))
            resampled[NUMERIC] = scaler.named_steps['scale'].inverse_transform(resampled[NUMERIC])
            assert np.isfinite(resampled[NUMERIC].to_numpy()).all()
            xt = preprocess.transform(resampled)
            for config in protocol['model_configs']:
                model = XGBClassifier(**config, objective='binary:logistic', tree_method='hist',
                                      eval_metric='logloss', subsample=.85, colsample_bytree=.85,
                                      scale_pos_weight=1., n_jobs=2, random_state=base.SEED)
                model.fit(xt, yr)
                p = model.predict_proba(xv)[:, 1]
                item = {'strategy': 'smotenc', 'sampling_ratio': ratio, 'k_neighbors': neighbors,
                        'class_weight': 1., 'config': config, 'fitted_rows': len(yr),
                        'fitted_positive_rows': int(np.sum(yr)), 'synthetic_rows': len(yr)-len(y),
                        'validation': base.threshold(val.target.to_numpy(), p)}
                candidates.append(item)
                if best is None or key(item) > key(best):
                    best = item
                    joblib.dump(Pipeline([('preprocessing', preprocess), ('model', model)]),
                                RUN / 'models/smotenc.joblib')
                    model.save_model(RUN / 'models/smotenc.ubj')
                    joblib.dump(scaler, RUN / 'models/sampling_numeric_preprocessing.joblib')
                    selected_data = resampled.copy()
                    selected_data['target'] = yr
                    selected_data['synthetic'] = np.arange(len(yr)) >= len(y)
                    selected_data.to_csv(RUN / 'selected_resampled_training.csv', index=False)
                    pd.DataFrame({'source_row_index': val.source_row_index, 'target': val.target,
                                  'score': p}).to_csv(RUN / 'validation_predictions.csv', index=False)
                    pd.DataFrame({'feature': preprocess.get_feature_names_out(),
                                  'gain_importance': model.feature_importances_}).sort_values(
                                      'gain_importance', ascending=False).to_csv(
                                          RUN / 'models/feature_importance.csv', index=False)
                print(f"Candidate {len(candidates)}/18: ratio={ratio}, k={neighbors}, "
                      f"depth={config['max_depth']}, validation F1={item['validation']['f1']:.4f}", flush=True)
            print(f'Sampling/configuration group completed in {time.perf_counter()-started:.1f}s', flush=True)
    assert len(candidates) == protocol['candidate_fits']
    best['artifact_sha256'] = base.digest(RUN / 'models/smotenc.joblib')
    previous_selection = base.load(PREVIOUS / 'selection.json')
    previous_winner = previous_selection['families'][previous_selection['selected_strategy']]
    selection = {'smotenc': best, 'candidate_fits': len(candidates),
                 'validation_winner_including_previous': 'smotenc' if key(best) > key(previous_winner)
                                                       else previous_selection['selected_strategy'],
                 'protocol_sha256': base.digest(RUN / 'protocol.json'), 'test_used_for_selection': False,
                 'selected_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN / 'validation_candidates.json', candidates)
    base.dump(RUN / 'selection.json', selection)
    print('FROZEN SELECTION', selection, flush=True)


def evaluate():
    if (RUN / 'final_results.json').exists():
        raise RuntimeError('Final evaluation already exists. Do not retune against it.')
    protocol = verify_protocol()
    selection = base.load(RUN / 'selection.json')
    assert selection['protocol_sha256'] == base.digest(RUN / 'protocol.json')
    assert selection['smotenc']['artifact_sha256'] == base.digest(RUN / 'models/smotenc.joblib')
    test = base.data('test', protocol)
    y = test.target.to_numpy()
    model = joblib.load(RUN / 'models/smotenc.joblib')
    p = model.predict_proba(test[base.FEATURES])[:, 1]
    t = selection['smotenc']['validation']['threshold']
    score = base.metrics(y, p, t)
    score['default_0_5'] = base.metrics(y, p, .5)
    previous = base.load(PREVIOUS / 'final_results.json')
    previous_predictions = pd.read_csv(PREVIOUS / 'test_predictions.csv')
    assert np.array_equal(previous_predictions.source_row_index, test.source_row_index)
    assert np.array_equal(previous_predictions.target, y)
    predictions = test[['source_row_index', 'gsim_code', 'numeric_year', 'target']].copy()
    predictions['score'] = p
    predictions['prediction'] = (p >= t).astype(int)
    predictions.to_csv(RUN / 'test_predictions.csv', index=False)
    precision, recall, _ = precision_recall_curve(y, p)
    base.dump(RUN / 'precision_recall_curve.json', {'precision': precision.tolist(), 'recall': recall.tolist()})
    train = base.data('train', protocol)
    subgroups = {}
    for name, mask in [('seen_stations', test.gsim_code.isin(train.gsim_code)),
                       ('unseen_stations', ~test.gsim_code.isin(train.gsim_code))]:
        subgroups[name] = base.metrics(y[mask], p[mask], t)
    rng = np.random.default_rng(base.SEED)
    stations = test.gsim_code.unique()
    station_ids = test.gsim_code.to_numpy()
    indices = {s: np.flatnonzero(station_ids == s) for s in stations}
    baseline = previous_predictions.untreated_prediction.to_numpy()
    chosen = (p >= t).astype(int)
    differences = []
    for _ in range(500):
        idx = np.concatenate([indices[s] for s in rng.choice(stations, len(stations), replace=True)])
        if len(np.unique(y[idx])) == 2:
            differences.append(f1_score(y[idx], chosen[idx])-f1_score(y[idx], baseline[idx]))
    result = {'smotenc': score, 'untreated': previous['test_metrics']['untreated'],
              'previous_balanced': previous['test_metrics'][previous['selected_strategy']],
              'selected_subgroups': subgroups,
              'station_bootstrap_95ci_f1_difference_vs_untreated': np.quantile(differences, [.025, .975]).tolist(),
              'validation_winner_including_previous': selection['validation_winner_including_previous'],
              'selection_sha256': base.digest(RUN / 'selection.json'),
              'test_fitting_or_resampling': False, 'already_examined_test': True,
              'original_application_model_unchanged': base.digest(ROOT / 'flood_model.pkl') == base.MODEL_SHA,
              'evaluated_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN / 'final_results.json', result)
    pd.DataFrame([{'strategy': strategy, **{k: v for k, v in result[strategy].items() if k != 'default_0_5'}}
                  for strategy in ['untreated', 'previous_balanced', 'smotenc']]).to_csv(
                      RUN / 'test_comparison.csv', index=False)
    write_report(protocol, selection, result)
    print(base.load(RUN / 'final_results.json'), flush=True)


def write_report(protocol, selection, result):
    best = selection['smotenc']
    lines = ['# Training-only SMOTE-NC comparison', '',
             'SMOTENC is the SMOTE variant for mixed numeric and categorical predictors. '
             'It interpolates numeric values and selects valid category values. '
             'This experiment does not manufacture fractional soil or climate categories.', '',
             '## Protocol', '',
             f"Target: {protocol['task']}. Training-only discharge cutoff: {protocol['cutoff']:.2f}.",
             'Train: 1900–1999, 37,184 rows (1,859 positive). Validation: 2000–2005, 3,387 rows (162 positive). '
             'Test: 2006–2015, 5,339 rows (320 positive). Same source, features, partitions and labels as imbalance_v1.',
             protocol['followup_limit'], '',
             'Train-only numeric medians and standardisation determine nearest neighbours. After sampling, '
             'numeric scaling is reversed and the original train-fitted model preprocessing is applied. '
             'All synthetic rows are training positives; validation/test are never sampled. '
             'Category values are checked against the original training vocabulary.',
             'Eighteen candidates: minority/majority ratios 0.25, 0.5, 1.0; k=3 or 5; '
             'the same three XGBoost configurations as the previous comparison. Positive class weight is 1 '
             'to isolate SMOTE from weighting. Thresholds and candidate selection use validation F1 only '
             '(ties AP, then precision).', '',
             '## Frozen validation selection', '',
             f"Best SMOTENC: ratio {best['sampling_ratio']}, k={best['k_neighbors']}, config {best['config']}, "
             f"threshold {best['validation']['threshold']:.6f}; validation F1 {best['validation']['f1']:.4f}.",
             f"Selected training rows: {best['fitted_rows']}, positives {best['fitted_positive_rows']}, "
             f"synthetic positives {best['synthetic_rows']}.",
             f"Validation winner when compared with previous frozen methods: {selection['validation_winner_including_previous']}.", '',
             '## Chronological test comparison', '',
             '| Strategy | Precision | Recall | F1 | F2 | PR AP | ROC-AUC | Accuracy | FP | FN |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['untreated', 'previous_balanced', 'smotenc']:
        s = result[name]
        lines.append('| '+name+' | '+' | '.join(f'{s[k]:.4f}' for k in
                     ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap', 'roc_auc', 'accuracy'])+
                     f" | {s['fp']} | {s['fn']} |")
    lines += ['', '## Interpretation and limits', '',
              f"SMOTENC raises precision from {result['untreated']['precision']:.4f} to "
              f"{result['smotenc']['precision']:.4f}, but recall falls from "
              f"{result['untreated']['recall']:.4f} to {result['smotenc']['recall']:.4f} and "
              f"F1 falls from {result['untreated']['f1']:.4f} to {result['smotenc']['f1']:.4f}. "
              'It does not improve the selected F1 objective. The previous balanced model remains the '
              'validation-selected winner; SMOTENC is saved as an experimental comparison, not deployed.',
              f"Station-cluster bootstrap 95% interval for SMOTENC minus untreated F1: "
              f"{result['station_bootstrap_95ci_f1_difference_vs_untreated']}.",
              f"Brier scores (lower is better): untreated {result['untreated']['brier']:.4f}, "
              f"SMOTENC {result['smotenc']['brier']:.4f}. Oversampling changes score scale; "
              'these scores are not calibrated flood probabilities. The 0.5 operating point is saved separately.',
              'The high-discharge label is not verified flood occurrence. Modern static catchment/climate '
              'attributes do not establish an operational weather forecast. Synthetic feature combinations '
              'remain statistical interpolations, not measured catchments or physically validated hydrological events. '
              'Repeated station attributes cross time; unseen-station performance is reported separately.', '',
              '| SMOTENC subgroup | Rows | Positives | Precision | Recall | F1 | AP |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in result['selected_subgroups'].items():
        lines.append(f"| {name} | {s['n']} | {s['class_1']} | {s['precision']:.4f} | "
                     f"{s['recall']:.4f} | {s['f1']:.4f} | {s['pr_auc_ap']:.4f} |")
    lines += ['', '## Saved artifacts', '',
              'Protocol, 18 validation candidates, frozen selection, fitted inference pipeline, native UBJ model, '
              'sampling preprocessor, selected resampled training table with synthetic flags, feature importance, '
              'test predictions and metrics are saved here. Prior experiment artifacts and the website model are unchanged.', '',
              '## Reproduction', '', 'Run from the RiskRadar directory in a new experiment version. '
              'Existing frozen protocol/selection/results refuse overwrite.', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe -m pip install -r requirements-training.txt',
              '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_smote.py prepare',
              '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_smote.py select',
              '.\\.venv\\Scripts\\python.exe scripts/evaluate_gloflo_smote.py evaluate', '```', '',
              'Method: [imbalanced-learn SMOTENC documentation]'
              '(https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html).', '']
    (RUN / 'SMOTE_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'evaluate'])
    args = parser.parse_args()
    {'prepare': prepare, 'select': select, 'evaluate': evaluate}[args.stage]()
