"""Recompute saved gold OOF results; never train or submit a Kaggle notebook."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix, roc_auc_score

from experiments.core import LABELS, SEEDS
from experiments.run_queue import verify_results
from experiments.weak_core import make_targets

ROOT = Path(__file__).resolve().parents[1]
DISPLAY_SEED = 42
THRESHOLD = .5


def bootstrap_pairs(available):
    from scripts.compare_confusions import PARENTS
    pairs=[('s13_ensemble','e09_protocols'),('s13_ensemble','s11_cross'),
           ('s10_gate','e09_protocols'),('s02_slice5','s03_slice10')]
    pairs.extend((eid,parent) for eid,parent in PARENTS.items() if eid.startswith(('r','t')))
    return [(a,b) for a,b in pairs if a in available and b in available]


def validate_oof(frame, gold, expected_folds=None):
    required = ['StudyInstanceUID', 'seed', 'fold', *LABELS]
    if frame.columns.tolist() != required or frame.duplicated(['StudyInstanceUID', 'seed']).any():
        raise ValueError('Invalid OOF columns or duplicate study/seed.')
    if set(frame.seed) != set(SEEDS):
        raise ValueError('Unexpected seeds.')
    values = frame[LABELS].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Invalid probabilities.')
    for _, group in frame.groupby('seed'):
        if set(group.StudyInstanceUID) != set(gold.index) or len(group) != len(gold):
            raise ValueError('OOF must contain each gold study exactly once per seed.')
    folds = frame.pivot(index='StudyInstanceUID', columns='seed', values='fold').loc[gold.index]
    if not np.isin(folds, [0, 1, 2]).all() or not (folds.to_numpy() == folds.iloc[:, :1].to_numpy()).all():
        raise ValueError('Inconsistent folds across seeds.')
    if expected_folds is not None and not np.array_equal(folds.iloc[:, 0], expected_folds):
        raise ValueError('Experiments use different gold folds.')
    return folds.iloc[:, 0].to_numpy()


def label_metrics(y, p, threshold=THRESHOLD):
    matrices = multilabel_confusion_matrix(y, p >= threshold)
    rows = []
    for j, label in enumerate(LABELS):
        tn, fp, fn, tp = map(int, matrices[j].ravel())
        rows.append(dict(label=label, positives=tp+fn, negatives=tn+fp, tn=tn, fp=fp, fn=fn, tp=tp,
                         recall=tp/(tp+fn), specificity=tn/(tn+fp),
                         precision=tp/(tp+fp) if tp+fp else np.nan,
                         f1=2*tp/(2*tp+fp+fn), auc=roc_auc_score(y[:, j], p[:, j]),
                         average_precision=average_precision_score(y[:, j], p[:, j]),
                         positive_mean_score=float(p[y[:, j] == 1, j].mean()),
                         negative_mean_score=float(p[y[:, j] == 0, j].mean())))
    return pd.DataFrame(rows)


def weighted_bootstrap_auc(y, predictions, counts):
    """Whole-study resampling, paired across models and seeds; ties count as 0.5."""
    scores = np.zeros(len(counts))
    valid = np.ones(len(counts), dtype=bool)
    for j in range(y.shape[1]):
        pos, neg = y[:, j] == 1, y[:, j] == 0
        wp, wn = counts[:, pos], counts[:, neg]
        denominator = wp.sum(1) * wn.sum(1)
        valid &= denominator > 0
        for p in predictions:
            difference = p[pos, j, None] - p[neg, j][None, :]
            wins = (difference > 0).astype(float) + .5*(difference == 0)
            numerator = np.einsum('bi,ij,bj->b', wp, wins, wn, optimize=True)
            scores += np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    scores /= y.shape[1] * len(predictions)
    scores[~valid] = np.nan
    return scores


def plots(output, summary, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    for eid in summary.experiment:
        rows = labels[(labels.experiment == eid) & (labels.seed == DISPLAY_SEED)].set_index('label').loc[LABELS]
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), layout='constrained')
        for label, ax in zip(LABELS, axes.ravel()):
            row = rows.loc[label]
            counts = row[['tn', 'fp', 'fn', 'tp']].to_numpy(int).reshape(2, 2)
            rates = counts/counts.sum(1, keepdims=True)
            im = ax.imshow(rates, cmap='Blues', vmin=0, vmax=1)
            for (i, j), count in np.ndenumerate(counts):
                tag = [['TN', 'FP'], ['FN', 'TP']][i][j]
                ax.text(j, i, f'{tag} {count}\n{rates[i,j]:.0%}', ha='center', va='center', fontsize=12,
                        color='white' if rates[i, j] > .55 else '#182333')
            ax.set(xticks=[0, 1], xticklabels=['Predicted 0', 'Predicted 1'], yticks=[0, 1],
                   yticklabels=['True 0', 'True 1'], title=f'{label} | AUC {row.auc:.3f}')
        fig.suptitle(f'{eid}: 12 independent confusion matrices\nGold58 OOF, seed 42, p >= 0.5; color and % normalized within each TRUE row', fontsize=16)
        fig.colorbar(im, ax=axes, shrink=.65, label='Fraction of each true class')
        fig.savefig(output/f'confusion_{eid}.png', dpi=160)
        fig.savefig(output/f'confusion_{eid}.svg')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 11), layout='constrained')
    colors = ['#236a9c' if eid.startswith('s') else '#568f75' if eid.startswith(('e06','e07','e08','e09','e10','e11','e12')) else '#959cab'
              for eid in summary.experiment]
    ax.barh(summary.experiment, summary.macro_auc, xerr=summary.seed_sd, color=colors, capsize=2)
    ax.invert_yaxis(); ax.set_xlim(.48, .83)
    ax.set(xlabel='Mean gold OOF macro ROC AUC (3 seeds); bars = seed SD, not CI',
           title='Completed experiments — training recipes differ between suites')
    for i, row in enumerate(summary.itertuples()):
        ax.text(row.macro_auc+row.seed_sd+.004, i, f'{row.macro_auc:.4f}', va='center', fontsize=9)
    fig.savefig(output/'experiment_auc.png', dpi=150); plt.close(fig)
    selected = ['e06_gold_control','e07_weak_labels','e09_protocols','s02_slice5','s10_gate','s11_cross','s12_finetune','s13_ensemble']
    matrix = labels.groupby(['label','experiment']).auc.mean().unstack().loc[LABELS, selected]
    fig, ax = plt.subplots(figsize=(13, 8), layout='constrained')
    im = ax.imshow(matrix, cmap='YlGnBu', vmin=.2, vmax=.95, aspect='auto')
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white' if value > .68 else '#182333')
    ax.set(xticks=range(len(selected)), xticklabels=[s.split('_')[0].upper() for s in selected],
           yticks=range(len(LABELS)), yticklabels=LABELS, title='Per-finding ROC AUC — mean of 3 seeds, Gold58 OOF')
    fig.colorbar(im, ax=ax, label='AUC'); fig.savefig(output/'category_auc.png', dpi=160); plt.close(fig)
    weak = pd.read_csv(output/'weak_label_support.csv').set_index('label').loc[LABELS]
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
    left = np.zeros(len(weak))
    for key, color in [('yes','#2d8068'),('no','#5897c6'),('unknown','#d4d9df')]:
        values = weak[key].to_numpy()/weak.weak_studies.to_numpy()
        ax.barh(LABELS, values, left=left, color=color, label=key.upper()); left += values
    ax.invert_yaxis(); ax.set(xlim=(0,1), xlabel='Fraction of 4,292 weak-label studies',
                            title='Report-label supervision coverage (UNK has zero training weight)')
    ax.legend(loc='lower right'); fig.savefig(output/'weak_label_coverage.png',dpi=150); plt.close(fig)


def main(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT/'data/raw/train.csv')
    gold = train.dropna(subset=LABELS).set_index('StudyInstanceUID').sort_index()
    if len(gold) != 58 or not gold.index.is_unique or not np.isin(gold[LABELS], [0,1]).all():
        raise ValueError('Unexpected gold population.')
    y = gold[LABELS].to_numpy(int)
    paths = list((ROOT/'outputs/experiments').glob('*/train/run/metrics.json'))
    paths += list((ROOT/'outputs/automation').glob('*/*/train/run/metrics.json'))
    summaries, detail, thresholds, errors, manifest, predictions = [], [], [], [], [], {}
    expected_folds = None
    for path in sorted(paths):
        metrics = json.loads(path.read_text(encoding='utf-8')); config = metrics['experiment']; eid = config['id']
        if eid in predictions:
            raise ValueError(f'Duplicate experiment: {eid}')
        verify_results(config, path.parent.parent, path.parent.parent.parent/'submit')
        oof_path = path.parent/'oof_predictions.csv'; frame = pd.read_csv(oof_path)
        expected_folds = validate_oof(frame, gold, expected_folds)
        by_seed, seed_aucs, seed_bces = [], [], []
        for run in metrics['runs']:
            seed = run['seed']; part = frame[frame.seed == seed].set_index('StudyInstanceUID').loc[gold.index]
            p = part[LABELS].to_numpy(float); by_seed.append((seed, p))
            auc = roc_auc_score(y, p, average='macro')
            clipped = np.clip(p, 1e-7, 1-1e-7)
            bce = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()
            if not np.isclose(auc, run['macro_auc'], atol=1e-7, rtol=0) or not np.isclose(bce, run['bce'], atol=1e-6, rtol=0):
                raise ValueError(f'Saved metrics differ from OOF: {eid}/{seed}')
            seed_aucs.append(auc); seed_bces.append(bce)
            rows = label_metrics(y, p); rows.insert(0, 'seed', seed); rows.insert(0, 'experiment', eid); detail.append(rows)
            if seed == DISPLAY_SEED:
                for threshold in (.3, .5, .7):
                    sweep = label_metrics(y, p, threshold); sweep.insert(0, 'threshold', threshold)
                    sweep.insert(0, 'experiment', eid); thresholds.append(sweep)
                for i, j in np.argwhere((p >= THRESHOLD) != y):
                    errors.append(dict(experiment=eid, StudyInstanceUID=gold.index[i], fold=int(part['fold'].iloc[i]),
                                       label=LABELS[j], truth=int(y[i,j]), probability=float(p[i,j]),
                                       error='FN' if y[i,j] else 'FP', seed=seed, threshold=THRESHOLD))
        predictions[eid] = np.stack([p for _, p in sorted(by_seed)])
        mean_auc = np.mean(seed_aucs)
        if (not np.isclose(mean_auc, metrics['macro_auc_mean'], atol=1e-7, rtol=0)
                or not np.isclose(np.std(seed_aucs), metrics['macro_auc_seed_std'], atol=1e-7, rtol=0)
                or not np.isclose(np.mean(seed_bces), metrics['bce_mean'], atol=1e-6, rtol=0)):
            raise ValueError('Summary AUC, seed SD or BCE mismatch.')
        summaries.append(dict(experiment=eid, macro_auc=mean_auc, seed_sd=metrics['macro_auc_seed_std'],
                              bce=metrics['bce_mean'], seed42_auc=roc_auc_score(y, predictions[eid][0]),
                              gold_studies=len(gold), training_studies=config.get('expected_training_studies',58),
                              head_epochs=config['epochs'], backbone=config['backbone']))
        for src in (path, oof_path):
            manifest.append(dict(path=src.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    summary = pd.DataFrame(summaries).sort_values('experiment')
    detail = pd.concat(detail, ignore_index=True)
    for ensemble, a, b in [('e12_ensemble','e10_class_mil','e11_finetune'),
                           ('s13_ensemble','s11_cross','s12_finetune')]:
        if not np.allclose(predictions[ensemble], .5*(predictions[a]+predictions[b]), atol=1e-7, rtol=0):
            raise ValueError(f'Incorrect saved ensemble probabilities: {ensemble}')
    summary.to_csv(output/'experiments.csv', index=False)
    detail.to_csv(output/'category_metrics_all_seeds.csv', index=False)
    pd.concat(thresholds, ignore_index=True).to_csv(output/'threshold_sensitivity_seed42.csv', index=False)
    errors = pd.DataFrame(errors)
    errors.to_csv(output/'errors_seed42.csv', index=False)
    common = errors[errors.experiment.isin(['e09_protocols','s10_gate','s13_ensemble'])]
    common = common.groupby(['StudyInstanceUID','label','error']).experiment.nunique().rename('models').reset_index()
    common[common.models==3].to_csv(output/'shared_errors_three_models.csv',index=False)
    worst = errors[errors.experiment=='s13_ensemble'].groupby(['StudyInstanceUID','fold']).agg(
        error_count=('label','size'), labels=('label',lambda s: '; '.join(s)), errors=('error',lambda s: '; '.join(s)))
    worst.sort_values('error_count',ascending=False).to_csv(output/'s13_study_review_queue.csv')
    main_rows = detail[(detail.experiment == 's13_ensemble') & (detail.seed == DISPLAY_SEED)].copy()
    for metric in ('auc', 'recall', 'precision'):
        grouped = detail[detail.experiment == 's13_ensemble'].groupby('label')[metric]
        main_rows[f'{metric}_seed_mean'] = main_rows.label.map(grouped.mean())
        main_rows[f'{metric}_seed_min'] = main_rows.label.map(grouped.min())
        main_rows[f'{metric}_seed_max'] = main_rows.label.map(grouped.max())
    main_rows.to_csv(output/'s13_categories.csv', index=False)
    source = pd.read_csv(ROOT/'data/weak_labels/report_labels_v2.csv')
    teacher = source.set_index('StudyInstanceUID').reindex(gold.index)
    teacher_rows=[]
    for j,label in enumerate(LABELS):
        verdict=teacher[label+'__verdict']; known=verdict.isin(['YES','NO']).to_numpy()
        truth=y[known,j]; guess=(verdict[known]=='YES').to_numpy()
        teacher_rows.append(dict(label=label,known_gold=int(known.sum()),unknown_or_missing=int((~known).sum()),
            tn=int(((truth==0)&~guess).sum()), fp=int(((truth==0)&guess).sum()),
            fn=int(((truth==1)&~guess).sum()), tp=int(((truth==1)&guess).sum())))
    pd.DataFrame(teacher_rows).to_csv(output/'report_vs_gold_known.csv',index=False)
    table, targets, weights, is_gold, _, audit = make_targets(train, source, {'weak_labels': True})
    source = source.set_index('StudyInstanceUID').loc[table.loc[~is_gold,'StudyInstanceUID']]
    weak_rows = []
    for j, label in enumerate(LABELS):
        verdict = source[label+'__verdict']; known = verdict != 'UNK'
        weak_rows.append(dict(label=label, weak_studies=len(source), yes=int((verdict=='YES').sum()),
                              no=int((verdict=='NO').sum()), unknown=int((~known).sum()),
                              known_fraction=float(known.mean()),
                              known_positive_fraction=float((verdict[known]=='YES').mean()),
                              gold_positive_fraction=float(y[:,j].mean()),
                              weak_weight_sum=float(weights[~is_gold,j].sum())))
    pd.DataFrame(weak_rows).to_csv(output/'weak_label_support.csv', index=False)
    # Resample study IDs, not 174 seed rows: seeds repeat the same 58 subjects.
    counts = np.random.default_rng(20261019).multinomial(len(gold), np.full(len(gold),1/len(gold)),size=5000)
    pairs=bootstrap_pairs(predictions)
    boot = {eid: weighted_bootstrap_auc(y, predictions[eid], counts) for eid in sorted({eid for pair in pairs for eid in pair})}
    intervals = []
    indexed = summary.set_index('experiment')
    for a, b in pairs:
        delta = boot[a]-boot[b]; valid = delta[np.isfinite(delta)]
        low, high = np.quantile(valid,[.025,.975])
        intervals.append(dict(a=a,b=b,mean_auc_difference=float(indexed.loc[a,'macro_auc']-indexed.loc[b,'macro_auc']),
                              percentile_95_low=float(low),percentile_95_high=float(high),valid_resamples=len(valid),
                              total_resamples=5000, note='Descriptive paired study bootstrap; ignores model selection and retraining variability.'))
    (output/'bootstrap_comparisons.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
    (output/'analysis_manifest.json').write_text(json.dumps(dict(gold_studies=58,experiments=len(summary),
        confusion_seed=DISPLAY_SEED,threshold=THRESHOLD,seeds=list(SEEDS),source_files=manifest,
        gold_sha256=hashlib.sha256((ROOT/'data/raw/train.csv').read_bytes()).hexdigest(),
        weak_source_sha256=hashlib.sha256((ROOT/'data/weak_labels/report_labels_v2.csv').read_bytes()).hexdigest(),
        validation='All OOF AUC/BCE, study IDs, folds, train/inference manifests and submission CSV formats verified.',
        teacher_independence=audit['teacher_gold_independence']),indent=2),encoding='utf-8')
    plots(output, summary, detail)
    from scripts.compare_confusions import write_comparisons
    write_comparisons(output)
    print(summary.to_string(index=False)); print('\nS13 seed42:'); print(main_rows.to_string(index=False))
    print('\nBootstrap:',json.dumps(intervals))
    print('\nWeak labels:'); print(pd.DataFrame(weak_rows).to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/analysis/2026-10-19')
    main(parser.parse_args().output)
