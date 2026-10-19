import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.core import LABELS
from experiments import weak_core, spatial_train
from experiments.spatial_core import selected_windows, spatial_signature


class FollowupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_generated_nested_mixed_bundle_offline_csv(self):
        from experiments.followup_configs import followup_configs, saved_config
        from experiments.followup_notebooks import render_followup
        from experiments.weak_train import write_weak_manifest
        from experiments.spatial_core import SpatialHead
        config=followup_configs()[0]; nb=render_followup(config,'submit');ns={}
        for cell in nb['cells']:
            src=''.join(cell['source'])
            if cell['cell_type']=='code' and not src.startswith('submit_followup_experiment('): exec(src,ns)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);bundle=p/'bundle';expected=[]
            leaves=[('members/e09_protocols','e09_protocols',.5),
                ('members/s13_ensemble/members/s11_cross','s11_cross',.25),
                ('members/s13_ensemble/members/s12_finetune','s12_finetune',.25)]
            def wheels(path):
                (path/'wheels').mkdir(parents=True)
                for f in ('pydicom-3.0.2-test.whl','python_gdcm-3.2.6-test.whl'): (path/'wheels'/f).write_bytes(b'wheel')
            tokens=torch.randn(6,3,3);mask=torch.ones(6,3,dtype=torch.bool);coords=torch.zeros(6,3,3)
            coords[...,0]=torch.arange(3.);coords[...,2]=1
            for relative,eid,weight in leaves:
                path=bundle/relative;wheels(path)
                cfg=dict(saved_config(eid),feature_dim=3,hidden_dim=16,crop=4)
                enc=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(1))
                torch.jit.trace(enc,torch.randn(1,3,4,4)).save(str(path/'encoder.ts'))
                spatial=cfg.get('architecture')=='spatial_v1'
                head=(SpatialHead if spatial else weak_core.WeakHead)(cfg,torch.zeros(3),torch.ones(3)).eval()
                torch.save(dict(state=head.state_dict(),experiment=cfg,labels=LABELS),path/'model.pt')
                args=[tokens[None],mask[None]]+([coords[None]] if spatial else [])
                expected.append(weight*head(*args).sigmoid().detach().numpy()[0])
                write_weak_manifest(cfg,path,4350,58)
            nested=bundle/'members/s13_ensemble';wheels(nested)
            write_weak_manifest(saved_config('s13_ensemble'),nested,4350,58)
            wheels(bundle);write_weak_manifest(config,bundle,4350,58)
            pd.DataFrame(dict(StudyInstanceUID=['u'])).to_csv(p/'test.csv',index=False)
            pd.DataFrame(dict(StudyInstanceUID=['u'],SeriesInstanceUID=['s'])).to_csv(p/'test_series.csv',index=False)
            pd.DataFrame(columns=['StudyInstanceUID',*LABELS]).to_csv(p/'sample_submission.csv',index=False)
            ns['spatial_tokens']=ns['weak_tokens']=lambda *args:(tokens,mask,coords)
            ns['make_transform']=lambda c:lambda x:x
            with patch('subprocess.check_call') as install:
                result=ns['submit_followup_experiment'](config,p,bundle,p/'out')
            self.assertIn('--no-index',install.call_args.args[0])
            np.testing.assert_allclose(result[LABELS].iloc[0],sum(expected),rtol=1e-6)
            (nested/'members/s11_cross/model.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): weak_core.validate_weak_bundle(bundle,config)

    def test_range_and_cache_compatibility(self):
        cfg = dict(representation='single', neighbors=5, centers=5, crop=224,
                   resize=256, interpolation='bilinear', pipeline_sha256='old')
        original = selected_windows(100, cfg)
        self.assertEqual(original, selected_windows(100, dict(cfg, center_range=[.25,.75])))
        widened = selected_windows(100, dict(cfg, center_range=[.1,.9]))
        self.assertEqual((widened[0], widened[-1]), ((8,), (91,)))
        self.assertEqual(selected_windows(2, cfg), [(0,), (1,)])
        new = dict(cfg, pipeline_sha256='new', feature_pipeline_sha256='old', center_range=[.25,.75])
        self.assertEqual(spatial_signature(cfg), spatial_signature(new))
        self.assertNotEqual(spatial_signature(cfg), spatial_signature(dict(new, center_range=[.1,.9])))
        self.assertNotEqual(spatial_signature(cfg), spatial_signature(dict(new, centers=9)))
        with self.assertRaises(ValueError): selected_windows(10, dict(cfg, center_range=[.8,.2]))

    def test_confidence_mask_keeps_population_and_gold(self):
        train = pd.DataFrame([dict(StudyInstanceUID=uid, Report=uid,
            **{k: (1. if uid=='gold' else np.nan) for k in LABELS}) for uid in ('gold','weak')])
        source = pd.DataFrame([dict(StudyInstanceUID='weak', **{k+suffix:value for k in LABELS
            for suffix,value in [('',.7),('__conf',.7),('__verdict','YES')]})])
        a = weak_core.make_targets(train,source,dict(weak_labels=True))
        b = weak_core.make_targets(train,source,dict(weak_labels=True,weak_min_confidence=.8))
        self.assertEqual(a[0].StudyInstanceUID.tolist(),b[0].StudyInstanceUID.tolist())
        self.assertTrue((b[2][b[3]]==3).all())
        self.assertTrue((b[2][~b[3]]==0).all())
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(b[5]['confidence_filter']['removed_yes']['ACL'],1)

    def test_per_label_loss_mask_and_equal_contribution(self):
        fn = getattr(weak_core,'per_label_weighted_bce',None)
        self.assertTrue(callable(fn))
        logits = torch.zeros(2,3,requires_grad=True)
        weights = torch.tensor([[1.,9.,0.],[1.,0.,0.]])
        fn(logits,torch.ones_like(logits),weights).backward()
        torch.testing.assert_close(logits.grad.sum(0),torch.tensor([-.25,-.25,0.]))
        x = torch.randn(2,3,requires_grad=True)
        loss=fn(x,torch.zeros_like(x),torch.zeros_like(x)); loss.backward()
        self.assertEqual(loss.item(),0.)
        self.assertTrue((x.grad==0).all())

    def test_warmup_uses_only_weak_training_groups(self):
        split=getattr(spatial_train,'weak_finetune_split',None)
        self.assertTrue(callable(split))
        gold=np.array([True,False,False,False]); groups=np.array(['00000001','00000001','0000000a','00000001'])
        train,valid=split(gold,groups)
        np.testing.assert_array_equal(train,[1,3]); np.testing.assert_array_equal(valid,[2])
        self.assertFalse(gold[train].any()); self.assertFalse(gold[valid].any())

    def test_mixed_nested_ensemble_weights(self):
        from experiments.followup_submit import predictor_leaves
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for relative,cfg in [('',dict(ensemble=True,train_dependencies=['e','s'])),
                ('members/e',dict(id='e')),('members/s',dict(ensemble=True,train_dependencies=['s1','s2'])),
                ('members/s/members/s1',dict(id='s1')),('members/s/members/s2',dict(id='s2'))]:
                path=root/relative;path.mkdir(parents=True,exist_ok=True)
                (path/'assets.json').write_text(json.dumps(dict(experiment=cfg)))
            leaves=predictor_leaves(root)
            self.assertEqual([(c['id'],w) for _,c,w in leaves],[('e',.5),('s1',.25),('s2',.25)])


if __name__=='__main__': unittest.main()
