import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiments.core import LABELS, SEEDS
from scripts.analyze_experiments import label_metrics, validate_oof, weighted_bootstrap_auc


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_pairs_include_available_followups_and_parents(self):
        from scripts.analyze_experiments import bootstrap_pairs
        pairs=bootstrap_pairs({'s10_gate','r04_confidence','r05_label_loss','r06_warmup'})
        self.assertIn(('r04_confidence','s10_gate'),pairs)
        self.assertIn(('r05_label_loss','s10_gate'),pairs)
        self.assertNotIn(('r06_warmup','s12_finetune'),pairs)

    def test_comparison_keeps_seed42_counts_separate_from_mean_auc(self):
        from scripts.compare_confusions import comparison_tables
        y=np.tile(np.array([0,0,1,1])[:,None],(1,12))
        frames=[]
        for eid in ('s02_slice5','s03_slice10'):
            for seed in SEEDS:
                p=np.tile(np.array([.1,.8,.3,.5])[:,None],(1,12))
                if eid=='s03_slice10' and seed==42: p[2]=.7
                rows=label_metrics(y,p); rows['experiment']=eid;rows['seed']=seed;frames.append(rows)
        summary=pd.DataFrame(dict(experiment=['s02_slice5','s03_slice10'],macro_auc=[.7,.71]))
        aggregate,deltas=comparison_tables(summary,pd.concat(frames))
        self.assertEqual(aggregate.set_index('experiment').loc['s03_slice10','tp'],24)
        self.assertEqual(deltas.delta_fn.unique().tolist(),[-1])
        self.assertEqual(deltas.delta_tp.unique().tolist(),[1])

    def test_confusion_axes_and_inclusive_threshold(self):
        y=np.tile(np.array([0,0,1,1])[:,None],(1,12))
        p=np.tile(np.array([.1,.8,.3,.5])[:,None],(1,12))
        row=label_metrics(y,p).iloc[0]
        self.assertEqual(row[['tn','fp','fn','tp']].tolist(),[1,1,1,1])
        self.assertEqual(row.recall,.5)
        self.assertEqual(row.precision,.5)

    def test_bootstrap_weights_match_repeated_subjects_with_ties(self):
        y=np.tile(np.array([0,0,1,1])[:,None],(1,12))
        p=np.tile(np.array([.1,.5,.5,.8])[:,None],(1,12))
        counts=np.array([[1,1,1,1],[2,0,1,1],[1,2,0,1],[2,2,0,0]])
        scores=weighted_bootstrap_auc(y,np.stack([p,p]),counts)
        for i in range(3):
            indices=np.repeat(np.arange(4),counts[i])
            self.assertAlmostEqual(scores[i],roc_auc_score(y[indices],p[indices]))
        self.assertTrue(np.isnan(scores[-1]))

    def test_duplicate_missing_foreign_id_and_misaligned_fold_rejected(self):
        gold=pd.DataFrame(index=['a','b'])
        frame=pd.DataFrame([dict(StudyInstanceUID=uid,seed=seed,fold=i,**dict.fromkeys(LABELS,.5))
                            for seed in SEEDS for i,uid in enumerate(gold.index)])
        validate_oof(frame,gold)
        with self.assertRaises(ValueError): validate_oof(pd.concat([frame,frame.iloc[:1]]),gold)
        with self.assertRaises(ValueError): validate_oof(frame.iloc[1:],gold)
        changed=frame.copy(); changed.loc[0,'StudyInstanceUID']='foreign'
        with self.assertRaises(ValueError): validate_oof(changed,gold)
        changed=frame.copy(); changed.loc[0,'fold']=2
        with self.assertRaises(ValueError): validate_oof(changed,gold)
        changed=frame.copy(); changed.loc[0,LABELS[0]]=np.nan
        with self.assertRaises(ValueError): validate_oof(changed,gold)


if __name__=='__main__': unittest.main()
