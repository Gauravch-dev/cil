"""Frozen follow-up comparison of learners and training-only sampling.

Run prepare, select, evaluate, verify. Existing studies and app are immutable.
The test period has already been examined in prior studies; no fresh-test claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import warnings

import imblearn
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import evaluate_gloflo_imbalance as base

RUN = base.ROOT / 'experiments/gloflo_learners_v1'
NUMERIC = [f for f in base.FEATURES if f not in base.CATEGORIES]
SAMPLERS = ['untreated', 'class_weight', 'random_oversampling', 'random_undersampling',
            'smotenc', 'smotenc_tomek', 'smotenc_enn']
CONFIGS = {
    'logistic_regression': [{'C': .1}, {'C': 1.}],
    'random_forest': [dict(n_estimators=200, max_depth=14, min_samples_leaf=5, max_leaf_nodes=512),
                      dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_leaf_nodes=1024)],
    'extra_trees': [dict(n_estimators=200, max_depth=16, min_samples_leaf=3, max_leaf_nodes=512),
                    dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_leaf_nodes=1024)],
    'hist_gradient_boosting': [dict(max_iter=300, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=10.),
                               dict(max_iter=400, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=5.)],
    'xgboost': [dict(n_estimators=500, max_depth=4, learning_rate=.05, min_child_weight=10, reg_lambda=10, reg_alpha=1),
                dict(n_estimators=300, max_depth=5, learning_rate=.08, min_child_weight=5, reg_lambda=5, reg_alpha=1)],
    'balanced_random_forest': [dict(n_estimators=200, max_depth=16, min_samples_leaf=3, sampling_strategy=.5),
                               dict(n_estimators=300, max_depth=None, min_samples_leaf=2, sampling_strategy=1.)],
    'balanced_bagging': [dict(n_estimators=80, depth=8, leaf=5, sampling_strategy=.5),
                         dict(n_estimators=120, depth=12, leaf=3, sampling_strategy=1.)],
    'easy_ensemble': [dict(n_estimators=10, boosting_rounds=50, learning_rate=.5, sampling_strategy=.5),
                      dict(n_estimators=20, boosting_rounds=80, learning_rate=.3, sampling_strategy=1.)],
}


def key(item):
    v = item['validation']
    return v['f1'], v['pr_auc_ap'], v['precision']


def prepare():
    if (RUN / 'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; choose a new prespecified version.')
    previous = base.load(base.RUN / 'protocol.json')
    protocol = {k: previous[k] for k in ['task', 'cutoff', 'chronology', 'source_sha256', 'split_hashes',
                                         'original_model_sha256', 'versions', 'static_feature_limit']}
    protocol.update({
        'features': base.FEATURES, 'categorical_features': base.CATEGORIES, 'seed': base.SEED,
        'samplers': SAMPLERS, 'configs': CONFIGS, 'candidate_fits': 76,
        'preprocessing': 'Numeric medians/standardisation and category vocabulary fitted on original training rows only. '
                         'All learners use the same scaled numeric plus one-hot representation.',
        'sampling_settings': {'random_oversampling_ratio': 1., 'random_undersampling_ratio': .25,
                              'smotenc_ratio': .5, 'smotenc_k': 3, 'cleaning_targets': 'majority only',
                              'enn_neighbors': 3, 'class_weight': 'positive weight sqrt(training neg/pos ratio)'},
        'selection': 'Max validation positive-class F1; ties average precision then precision. '
                     'Every operating threshold is chosen on validation only. No test selection/refit.',
        'followup_limit': 'The same 2006–2015 test and validation periods were examined in prior experiments. '
                          'This is an exploratory follow-up, not fresh independent confirmation. '
                          'Broad validation search can overfit validation; no test results are used in selection.',
        'protected_files': {str(path.relative_to(base.ROOT)): base.digest(path) for path in [
            base.RUN/'protocol.json', base.RUN/'selection.json', base.RUN/'final_results.json',
            base.ROOT/'experiments/gloflo_smotenc_v1/protocol.json',
            base.ROOT/'experiments/gloflo_smotenc_v1/selection.json',
            base.ROOT/'experiments/gloflo_smotenc_v1/final_results.json']},
        'created_utc': datetime.now(timezone.utc).isoformat(),
    })
    protocol['versions']['imbalanced_learn'] = imblearn.__version__
    base.dump(RUN/'protocol.json', protocol)
    print('Frozen protocol: 76 candidates, 8 learners, original chronology and target.', flush=True)


def verify_protocol():
    p = base.load(RUN/'protocol.json')
    for name, expected in p['protected_files'].items():
        assert base.digest(base.ROOT/name) == expected, name
    assert base.digest(base.ROOT/'flood_model.pkl') == p['original_model_sha256'] == base.MODEL_SHA
    assert base.digest(base.ROOT/'data/raw/kuntla_flooddatabase.csv') == p['source_sha256']
    return p


def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('median', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), base.CATEGORIES),
    ], sparse_threshold=0)


def estimator(name, config, weight):
    cfg = config.copy()
    cw = None if weight == 1. else {0: 1., 1: weight}
    if name == 'logistic_regression':
        return LogisticRegression(**cfg, class_weight=cw, max_iter=3000, solver='lbfgs', random_state=base.SEED)
    if name in ['random_forest', 'extra_trees']:
        cls = RandomForestClassifier if name == 'random_forest' else ExtraTreesClassifier
        return cls(**cfg, class_weight=cw, max_features='sqrt', n_jobs=2, random_state=base.SEED)
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(**cfg, class_weight=cw, learning_rate=.05,
                                               early_stopping=False, random_state=base.SEED)
    if name == 'xgboost':
        return XGBClassifier(**cfg, objective='binary:logistic', tree_method='hist', eval_metric='logloss',
                             subsample=.85, colsample_bytree=.85, scale_pos_weight=weight,
                             n_jobs=2, random_state=base.SEED)
    if name == 'balanced_random_forest':
        return BalancedRandomForestClassifier(**cfg, max_leaf_nodes=1024, replacement=True,
                                                bootstrap=False, n_jobs=2, random_state=base.SEED)
    if name == 'balanced_bagging':
        tree = DecisionTreeClassifier(max_depth=cfg.pop('depth'), min_samples_leaf=cfg.pop('leaf'))
        return BalancedBaggingClassifier(**cfg, estimator=tree, replacement=False,
                                          n_jobs=2, random_state=base.SEED)
    if name == 'easy_ensemble':
        boost = AdaBoostClassifier(n_estimators=cfg.pop('boosting_rounds'), learning_rate=cfg.pop('learning_rate'),
                                   random_state=base.SEED)
        return EasyEnsembleClassifier(**cfg, estimator=boost, replacement=False, n_jobs=2, random_state=base.SEED)
    raise ValueError(name)


def build_training_sets(train, prep, p):
    X, y = train[base.FEATURES], train.target.to_numpy()
    original = prep.transform(X)
    datasets = {'untreated': (original, y), 'class_weight': (original, y)}
    audits = {}
    settings = p['sampling_settings']
    for strategy, ratio in [('random_oversampling', settings['random_oversampling_ratio']),
                             ('random_undersampling', settings['random_undersampling_ratio'])]:
        idx = base.sampling_indices(y, strategy, ratio, base.SEED)
        datasets[strategy] = (original[idx], y[idx])
    distance = X.copy()
    distance[NUMERIC] = prep.named_transformers_['numeric'].transform(X[NUMERIC])
    smote = SMOTENC(categorical_features=base.CATEGORIES, sampling_strategy=settings['smotenc_ratio'],
                    k_neighbors=settings['smotenc_k'], random_state=base.SEED)
    generated, yr = smote.fit_resample(distance, y)
    np.testing.assert_allclose(generated.iloc[:len(y)][NUMERIC], distance[NUMERIC])
    assert generated.iloc[:len(y)][base.CATEGORIES].equals(distance[base.CATEGORIES])
    assert np.array_equal(yr[:len(y)], y)
    for col in base.CATEGORIES:
        assert set(generated[col]).issubset(set(X[col]))
    generated[NUMERIC] = prep.named_transformers_['numeric'].named_steps['scale'].inverse_transform(generated[NUMERIC])
    encoded = prep.transform(generated)
    assert np.isfinite(encoded).all()
    datasets['smotenc'] = (encoded, yr)
    for strategy, cleaner in [
        ('smotenc_tomek', TomekLinks(sampling_strategy='majority', n_jobs=2)),
        ('smotenc_enn', EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=3, kind_sel='all', n_jobs=2)),
    ]:
        started = time.perf_counter()
        # Distances use train-scaled numeric and one-hot categorical columns.
        cleaned, yc = cleaner.fit_resample(encoded, yr)
        assert np.sum(yc == 1) == np.sum(yr == 1), 'Cleaning must preserve all training minority examples.'
        datasets[strategy] = (cleaned, yc)
        audits[strategy] = {'removed_majority_rows': len(yr)-len(yc), 'retained_indices_sha256':
                           base.sha256(cleaner.sample_indices_.tobytes()).hexdigest()}
        print(f'{strategy}: retained {len(yc)} training rows ({time.perf_counter()-started:.1f}s).', flush=True)
    for strategy, (_, labels) in datasets.items():
        audits.setdefault(strategy, {})
        audits[strategy].update(rows=len(labels), positive=int(np.sum(labels)), negative=int(np.sum(labels == 0)))
    base.dump(RUN/'training_sampling_audit.json', audits)
    return datasets


def select():
    if (RUN/'selection.json').exists():
        raise RuntimeError('Selection already frozen; do not retune.')
    p = verify_protocol()
    # No test data or previous test scores are read during selection.
    train, val = base.data('train', p), base.data('validation', p)
    prep = preprocessing().fit(train[base.FEATURES])
    joblib.dump(prep, RUN/'training_preprocessing.joblib')
    datasets = build_training_sets(train, prep, p)
    xv, yv = prep.transform(val[base.FEATURES]), val.target.to_numpy()
    weight = float(np.sqrt(np.sum(train.target == 0)/np.sum(train.target == 1)))
    (RUN/'models').mkdir(exist_ok=True)
    candidates, cells = [], {}
    jobs = [(name, sampler, cfg) for name in list(CONFIGS)[:5] for sampler in p['samplers'] for cfg in p['configs'][name]]
    jobs += [(name, 'internal_balancing', cfg) for name in list(CONFIGS)[5:] for cfg in p['configs'][name]]
    assert len(jobs) == p['candidate_fits']
    for i, (name, sampler, config) in enumerate(jobs, 1):
        started = time.perf_counter()
        xt, yt = datasets['untreated' if sampler == 'internal_balancing' else sampler]
        class_weight = weight if sampler == 'class_weight' else 1.
        model = estimator(name, config, class_weight)
        # A failed/convergence-limited candidate aborts the experiment instead of silently disappearing.
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            model.fit(xt, yt)
            scores = model.predict_proba(xv)[:, 1]
        serious = [str(w.message) for w in observed if w.category.__name__ == 'ConvergenceWarning']
        if serious:
            raise RuntimeError(f'{name} failed convergence: {serious}')
        cell = name+'__'+sampler
        item = {'learner': name, 'sampler': sampler, 'cell': cell, 'config': config,
                'positive_weight': class_weight, 'fitted_rows': len(yt), 'positive_training_rows': int(np.sum(yt)),
                'internal_balancing': sampler == 'internal_balancing',
                'validation': base.threshold(yv, scores), 'fit_seconds': time.perf_counter()-started,
                'warnings': sorted(set(str(w.message) for w in observed))}
        candidates.append(item)
        if cell not in cells or key(item) > key(cells[cell]):
            path = RUN/f'models/{cell}.joblib'
            joblib.dump(Pipeline([('preprocessing', prep), ('model', model)]), path, compress=3)
            item['artifact_sha256'] = base.digest(path)
            cells[cell] = item.copy()
            pd.DataFrame({'source_row_index': val.source_row_index, 'target': yv, 'score': scores}).to_csv(
                RUN/f'{cell}_validation_predictions.csv', index=False)
        print(f"{i}/{len(jobs)} {cell}: validation F1={item['validation']['f1']:.4f}, "
              f"AP={item['validation']['pr_auc_ap']:.4f}, {item['fit_seconds']:.1f}s", flush=True)
        base.dump(RUN/'candidate_progress.json', candidates)
    winners = {name: max((cell for cell in cells if cells[cell]['learner'] == name), key=lambda c: key(cells[c]))
               for name in p['configs']}
    winner = max(cells, key=lambda c: key(cells[c]))
    old = base.load(base.RUN/'selection.json')
    old_winner = old['families'][old['selected_strategy']]
    prior_smote = base.load(base.ROOT/'experiments/gloflo_smotenc_v1/selection.json')['smotenc']
    overall = max([('new_study:'+winner, cells[winner]), ('previous:'+old['selected_strategy'], old_winner),
                   ('previous:smotenc', prior_smote)], key=lambda pair: key(pair[1]))[0]
    selection = {'selected_cell': winner, 'selected_including_prior_studies': overall,
                 'best_by_learner': winners, 'cells': cells, 'candidate_fits': len(candidates),
                 'protocol_sha256': base.digest(RUN/'protocol.json'), 'test_used_for_selection': False,
                 'selected_utc': datetime.now(timezone.utc).isoformat()}
    base.dump(RUN/'validation_candidates.json', candidates)
    base.dump(RUN/'selection.json', selection)
    print('FROZEN WINNER', winner, cells[winner]['validation'], 'OVERALL', overall, flush=True)


def evaluate():
    if (RUN/'final_results.json').exists():
        raise RuntimeError('Final results already frozen. No repeat evaluation or retuning.')
    p = verify_protocol()
    selection = base.load(RUN/'selection.json')
    assert base.digest(RUN/'protocol.json') == selection['protocol_sha256']
    test, train = base.data('test', p), base.data('train', p)
    y = test.target.to_numpy()
    predictions = test[['source_row_index', 'gsim_code', 'numeric_year', 'target']].copy()
    results, curves = {}, {}
    for cell, item in selection['cells'].items():
        path = RUN/f'models/{cell}.joblib'
        assert base.digest(path) == item['artifact_sha256']
        scores = joblib.load(path).predict_proba(test[base.FEATURES])[:, 1]
        t = item['validation']['threshold']
        results[cell] = base.metrics(y, scores, t)
        results[cell]['default_0_5'] = base.metrics(y, scores, .5)
        predictions[cell+'_score'] = scores
        predictions[cell+'_prediction'] = (scores >= t).astype(int)
        if cell in selection['best_by_learner'].values():
            precision, recall, _ = precision_recall_curve(y, scores)
            curves[cell] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    chosen = selection['selected_cell']
    subgroups = {}
    for name, mask in [('seen_stations', test.gsim_code.isin(train.gsim_code)),
                       ('unseen_stations', ~test.gsim_code.isin(train.gsim_code))]:
        subgroups[name] = base.metrics(y[mask], predictions.loc[mask, chosen+'_score'].to_numpy(),
                                      selection['cells'][chosen]['validation']['threshold'])
    prior_results = base.load(base.RUN/'final_results.json')
    prior_predictions = pd.read_csv(base.RUN/'test_predictions.csv')
    assert np.array_equal(prior_predictions.source_row_index, test.source_row_index)
    assert np.array_equal(prior_predictions.target, y)
    comparators = {'previous_untreated': prior_predictions.untreated_prediction.to_numpy(),
                   'previous_balanced': prior_predictions[prior_results['selected_strategy']+'_prediction'].to_numpy()}
    same_learner_baseline = selection['cells'][chosen]['learner']+'__untreated'
    if same_learner_baseline in results:
        comparators['same_learner_untreated'] = predictions[same_learner_baseline+'_prediction'].to_numpy()
    intervals = bootstrap(y, predictions[chosen+'_prediction'].to_numpy(), test.gsim_code.to_numpy(), comparators)
    result = {'selected_cell': chosen, 'selection_sha256': base.digest(RUN/'selection.json'),
              'selected_including_prior_studies': selection['selected_including_prior_studies'],
              'test_metrics': results, 'selected_subgroups': subgroups, 'paired_f1_bootstrap_95ci': intervals,
              'previous_untreated': prior_results['test_metrics']['untreated'],
              'previous_balanced': prior_results['test_metrics'][prior_results['selected_strategy']],
              'test_n': len(y), 'test_positive': int(np.sum(y)), 'already_examined_test': True,
              'test_fitting_or_resampling': False, 'original_app_model_unchanged': True,
              'evaluated_utc': datetime.now(timezone.utc).isoformat()}
    predictions.to_csv(RUN/'test_predictions.csv', index=False)
    base.dump(RUN/'precision_recall_curves.json', curves)
    base.dump(RUN/'final_results.json', result)
    base.dump(RUN/'selected_model_metadata.json', {
        'pipeline': f'models/{chosen}.joblib', 'pipeline_sha256': selection['cells'][chosen]['artifact_sha256'],
        'target': p['task'], 'training_discharge_cutoff': p['cutoff'], 'features': p['features'],
        'chronology': p['chronology'], 'selection': p['selection'],
        'learner_and_sampling': selection['cells'][chosen], 'test_metrics': results[chosen],
        'selection_including_prior_studies': selection['selected_including_prior_studies'],
        'limitations': [p['followup_limit'], p['static_feature_limit'],
                        'Not verified flood occurrence; not a calibrated operational flood probability.'],
        'deployed_to_application': False,
    })
    rows = []
    for cell, metrics in results.items():
        rows.append({'cell': cell, 'learner': selection['cells'][cell]['learner'],
                     'sampler': selection['cells'][cell]['sampler'],
                     'validation_f1': selection['cells'][cell]['validation']['f1'],
                     **{k: v for k, v in metrics.items() if k != 'default_0_5'}})
    pd.DataFrame(rows).to_csv(RUN/'test_comparison.csv', index=False)
    write_report(p, selection, result)
    verify_protocol()
    print('FINAL WINNER', chosen, results[chosen], 'INTERVALS', intervals, flush=True)


def bootstrap(y, pred, stations, comparators):
    unique = np.unique(stations)
    grouped = {station: np.flatnonzero(stations == station) for station in unique}
    rng = np.random.default_rng(base.SEED)
    differences = {name: [] for name in comparators}
    for _ in range(500):
        idx = np.concatenate([grouped[s] for s in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        current = f1_score(y[idx], pred[idx], zero_division=0)
        for name, other in comparators.items():
            differences[name].append(current-f1_score(y[idx], other[idx], zero_division=0))
    return {name: np.quantile(values, [.025, .975]).tolist() for name, values in differences.items()}


def write_report(p, s, r):
    chosen = s['selected_cell']
    lines = ['# Learning algorithms and sampling comparison', '',
             p['followup_limit'], '', '## Fixed target and data', '',
             f"{p['task']}. The training-only 95th percentile is {p['cutoff']:.2f} in source units.",
             'Train: 1900–1999, 37,184 rows / 1,859 positive. Validation: 2000–2005, '
             '3,387 rows / 162 positive. Test: 2006–2015, 5,339 rows / 320 positive. '
             'All source rows, labels, features and dates match the previous imbalance studies.', '',
             '## Training and selection', '', p['preprocessing'],
             'Five standard learners × seven treatments × two configurations = 70 fits. '
             'Balanced Random Forest, balanced bagging and EasyEnsemble each add two configurations: '
             '76 candidate fits, with 38 frozen learner/treatment pipelines.',
             'Treatments: untreated, positive class weight sqrt(training ratio), random oversampling to 1:1, '
             'random undersampling to 1:4, SMOTENC to 1:2 with k=3, SMOTENC then majority-only Tomek cleaning, '
             'and SMOTENC then majority-only ENN cleaning. Cleaning distances use scaled numeric and '
             'one-hot categorical columns. Minority examples are preserved by both cleaners.',
             'No resampling, cleaning or weight fitting is applied to validation/test. '
             'HistGradientBoosting disables random internal early stopping; no learner fits on validation. '
             'The three balanced ensembles perform their own training-only internal sampling.',
             p['selection'], '',
             f"New-study validation winner: **{chosen}**, validation F1 "
             f"{s['cells'][chosen]['validation']['f1']:.4f}, threshold "
             f"{s['cells'][chosen]['validation']['threshold']:.6f}.",
             f"Validation winner including prior frozen studies: **{s['selected_including_prior_studies']}**.", '',
             '## Best configuration per learner (chosen on validation)', '',
             '| Learner / treatment | Validation F1 | Test precision | Recall | F1 | F2 | PR AP | ROC-AUC | FP | FN |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for learner, cell in sorted(s['best_by_learner'].items(), key=lambda pair: key(s['cells'][pair[1]]), reverse=True):
        m = r['test_metrics'][cell]
        lines.append(f"| {cell} | {s['cells'][cell]['validation']['f1']:.4f} | "+
                     ' | '.join(f'{m[k]:.4f}' for k in ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap', 'roc_auc'])+
                     f" | {m['fp']} | {m['fn']} |")
    lines += ['', '## Previous baselines and uncertainty', '',
              '| Previous frozen baseline | Precision | Recall | F1 | F2 | PR AP | ROC-AUC | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['previous_untreated', 'previous_balanced']:
        m = r[name]
        lines.append('| '+name+' | '+' | '.join(f'{m[k]:.4f}' for k in
                     ['precision', 'recall', 'f1', 'f2', 'pr_auc_ap', 'roc_auc'])+f" | {m['fp']} | {m['fn']} |")
    lines += ['', 'Paired station-cluster bootstrap 95% intervals for selected-new-model F1 minus baseline:']
    for name, interval in r['paired_f1_bootstrap_95ci'].items():
        lines.append(f'- {name}: [{interval[0]:.4f}, {interval[1]:.4f}]')
    lines += ['', 'Intervals spanning zero do not establish reliable improvement. The winner is frozen '
              'using validation, never chosen by whichever test metric looks best.', '',
              '## What changed', '',
              f"The validation-selected model has test F1 {r['test_metrics'][chosen]['f1']:.4f}, compared with "
              f"{r['previous_balanced']['f1']:.4f} for the previous balanced XGBoost. Its precision increases "
              f"from {r['previous_balanced']['precision']:.4f} to {r['test_metrics'][chosen]['precision']:.4f}, "
              f"but recall decreases from {r['previous_balanced']['recall']:.4f} to "
              f"{r['test_metrics'][chosen]['recall']:.4f}. This is a precision/recall trade-off, not improvement everywhere.",
              'The untreated Random Forest has test F1 0.7744, above the selected oversampled version at 0.7532. '
              'Oversampling won validation but did not improve Random Forest test F1. '
              'Untreated HistGradientBoosting has the highest observed test F1 among these cells (0.7786); '
              'it is not substituted as the winner after inspecting test results. '
              'The results provide no consistent evidence that more elaborate sampling is better.', '',
              '## All learner / treatment cells', '',
              '| Cell | Validation F1 | Test precision | Recall | F1 | AP | FP | FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cell in sorted(s['cells'], key=lambda c: key(s['cells'][c]), reverse=True):
        m = r['test_metrics'][cell]
        lines.append(f"| {cell} | {s['cells'][cell]['validation']['f1']:.4f} | "+
                     ' | '.join(f'{m[k]:.4f}' for k in ['precision', 'recall', 'f1', 'pr_auc_ap'])+
                     f" | {m['fp']} | {m['fn']} |")
    lines += ['', '## Selected model transfer and calibration', '',
              '| Stations | Rows | Positives | Precision | Recall | F1 | AP |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in r['selected_subgroups'].items():
        lines.append(f"| {name} | {m['n']} | {m['class_1']} | "+
                     ' | '.join(f'{m[k]:.4f}' for k in ['precision', 'recall', 'f1', 'pr_auc_ap'])+' |')
    m = r['test_metrics'][chosen]
    unseen = r['selected_subgroups']['unseen_stations']
    lines += ['', f"**Transfer limitation:** the frozen winner detects {unseen['tp']} of "
              f"{unseen['class_1']} positive cases at unseen stations (recall {unseen['recall']:.4f}). "
              'Aggregate metrics must not be treated as evidence of reliable performance at new stations. '
              'The model is retained as an experiment and is not promoted to the website.',
              f"Selected Brier score: {m['brier']:.4f}; previous untreated: {r['previous_untreated']['brier']:.4f}. "
              'Lower Brier is better, but does not by itself establish probability calibration. '
              'Oversampled/weighted scores must not be labelled calibrated flood probabilities.',
              'Target: high annual peak discharge, not verified flood occurrence. Static contemporary attributes '
              'and repeated stations do not establish a lead-time weather forecast or new-location performance. '
              'Synthetic examples are statistical interpolations, not measured or physically validated catchments.',
              p['static_feature_limit'],
              'All 76 candidate validation scores, 38 pipelines, source/split hashes, sampling audit, '
              'test predictions and metrics are preserved. Previous studies and the website model are unchanged.', '',
              '## Reproduction', '', 'Dependencies: requirements-training.txt. Existing frozen files refuse overwrite. '
              'Use a new prespecified experiment version for a new study.', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_learners.py prepare',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_learners.py select',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_learners.py evaluate',
              '.\\.venv\\Scripts\\python.exe scripts/compare_gloflo_learners.py verify', '```', '',
              'Methods: [imbalanced-learn ensembles](https://imbalanced-learn.org/stable/ensemble.html), '
              '[neighbour cleaning](https://imbalanced-learn.org/stable/under_sampling.html).', '']
    (RUN/'COMPARISON_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def verify():
    p = verify_protocol()
    s, r = base.load(RUN/'selection.json'), base.load(RUN/'final_results.json')
    assert base.digest(RUN/'protocol.json') == s['protocol_sha256']
    assert base.digest(RUN/'selection.json') == r['selection_sha256']
    candidates = base.load(RUN/'validation_candidates.json')
    assert len(candidates) == p['candidate_fits'] == 76
    assert len(s['cells']) == 38
    assert key(max(candidates, key=key)) == key(s['cells'][s['selected_cell']])
    test, train = base.data('test', p), base.data('train', p)
    base.data('validation', p)
    recorded = pd.read_csv(RUN/'test_predictions.csv')
    assert np.array_equal(recorded.source_row_index, test.source_row_index)
    assert np.array_equal(recorded.target, test.target)
    worst = 0.
    for cell, item in s['cells'].items():
        path = RUN/f'models/{cell}.joblib'
        assert base.digest(path) == item['artifact_sha256']
        pipe = joblib.load(path)
        median = pipe.named_steps['preprocessing'].named_transformers_['numeric'].named_steps['median']
        np.testing.assert_allclose(median.statistics_, train[NUMERIC].median().to_numpy())
        scores = pipe.predict_proba(test[base.FEATURES])[:, 1]
        error = float(np.max(np.abs(scores-recorded[cell+'_score'].to_numpy())))
        assert error < 1e-7
        assert np.array_equal(scores >= item['validation']['threshold'], recorded[cell+'_prediction'])
        worst = max(worst, error)
    result = {'pipelines_verified': 38, 'candidate_selection_verified': 76,
              'largest_saved_score_difference': worst, 'train_only_medians_verified': True,
              'original_source_splits_prior_experiments_and_app_preserved': True}
    base.dump(RUN/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'prepare': prepare, 'select': select, 'evaluate': evaluate, 'verify': verify}[args.stage]()
