import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.spatial_core import SpatialHead, rotary, selected_windows, spatial_views
from experiments.spatial_configs import spatial_configs


class SpatialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **extra):
        return dict(feature_dim=8, hidden_dim=16, heads=4, sequence=True,
                    position='rope_index', plane_embedding=True, fusion='concat',
                    representation='single', neighbors=5, centers=5, crop=4,
                    mm_unit=3., epochs=2, batch_size=2, lr=.001, weight_decay=.01,
                    **extra)

    def test_neighbors_are_unique_ordered_and_triplets_do_not_expand_to_five_channels(self):
        for neighbors in (3, 5, 10):
            cfg = self.config(); cfg['neighbors'] = neighbors
            windows = selected_windows(41, cfg)
            indices = [w[0] for w in windows]
            self.assertEqual(indices, sorted(set(indices)))
            self.assertLessEqual(len(indices), 5 * neighbors)
            self.assertTrue(all(len(w) == 1 for w in windows))
            self.assertEqual(selected_windows(2, cfg), [(0,), (1,)])
        cfg.update(representation='triplet', neighbors=3)
        windows = selected_windows(41, cfg)
        self.assertEqual(windows[0], (9, 10, 11))
        self.assertEqual(len(windows), 5)

    def test_views_keep_coordinates_and_mask_missing_protocols(self):
        cfg = self.config()
        series = pd.DataFrame([dict(StudyInstanceUID='u', SeriesInstanceUID='s',
                                   Anatomical_Plane='Sagittal', Fluid_Sensitive=True, Fat_Suppression=True)])
        paths = [Path(str(i)) for i in range(5)]
        coords = np.array([[i, i * 2., 1.] for i in range(5)], dtype=np.float32)
        with patch('experiments.spatial_core.spatial_order', return_value=(paths, coords)), \
             patch('experiments.spatial_core.image_tensor', side_effect=lambda p,t:t(torch.full((3,4,4),float(p.name)))):
            images, mask, positions = spatial_views('u', series, Path('.'), lambda x:x, cfg)
        self.assertEqual(images.shape, (6,25,3,4,4))
        self.assertEqual(mask.sum().item(), 5)
        self.assertFalse(mask[1:].any())
        torch.testing.assert_close(positions[0,:5], torch.from_numpy(coords))

    def test_rotary_relative_shift_and_real_position_sensitivity(self):
        q, k = torch.randn(2,4,5,4), torch.randn(2,4,5,4)
        pos = torch.arange(5.).expand(2,-1)
        scores = rotary(q,pos) @ rotary(k,pos).transpose(-1,-2)
        shifted = rotary(q,pos+7) @ rotary(k,pos+7).transpose(-1,-2)
        torch.testing.assert_close(scores,shifted,atol=2e-5,rtol=2e-5)
        self.assertGreater((scores-q@k.transpose(-1,-2)).abs().max().item(), .1)

    def test_heads_use_positions_but_ignore_padding_and_joint_permutations(self):
        for fusion in ('concat','mlp','gate','cross'):
            cfg=self.config(); cfg['fusion']=fusion
            torch.manual_seed(7)
            head=SpatialHead(cfg,torch.zeros(8),torch.ones(8)).eval()
            x=torch.randn(2,6,5,8); mask=torch.ones(2,6,5,dtype=torch.bool); mask[:,2]=False
            coords=torch.zeros(2,6,5,3); coords[...,0]=torch.arange(5.); coords[...,1]=torch.arange(5.)*2; coords[...,2]=1
            with torch.no_grad():
                first=head(x,mask,coords)
                order=[4,0,3,1,2]
                torch.testing.assert_close(first,head(x[:,:,order],mask[:,:,order],coords[:,:,order]),atol=2e-5,rtol=2e-5)
                self.assertGreater((first-head(x,mask,coords[:,:,order])).abs().max().item(),1e-7)
                x[~mask]=float('nan'); coords[~mask]=float('nan')
                torch.testing.assert_close(first,head(x,mask,coords))
            self.assertEqual(first.shape,(2,12)); self.assertTrue(torch.isfinite(first).all())
            with self.assertRaises(ValueError): head(x,torch.zeros_like(mask),coords)

    def test_config_ablation_parents_and_nonoverlapping_ids(self):
        configs=spatial_configs()
        self.assertEqual(len(configs),14)
        seen=set()
        for cfg in configs:
            self.assertNotIn(cfg['id'],seen)
            self.assertTrue(set(cfg['train_dependencies'])<=seen)
            seen.add(cfg['id'])
        self.assertEqual(configs[-1]['train_dependencies'],['s11_cross','s12_finetune'])
        for cfg in configs[4:12]: self.assertEqual(cfg['neighbors'],5)

    def test_training_validation_independence_and_bundle_roundtrip(self):
        from experiments.spatial_train import fit_spatial_head
        from experiments.spatial_submit import load_spatial_predictor
        from experiments.weak_train import write_weak_manifest
        cfg=self.config(); cfg.update(suite='weak', id='test', ensemble=False)
        x=torch.randn(6,6,3,8); mask=torch.ones(6,6,3,dtype=torch.bool)
        coords=torch.zeros(6,6,3,3); coords[...,0]=torch.arange(3.); coords[...,2]=1
        y=torch.rand(6,12); w=torch.ones(6,12)
        first,_=fit_spatial_head(x,mask,coords,y,w,np.arange(4),42,cfg)
        x[4:]=1e5; coords[4:]=1e5; y[4:]=1
        second,_=fit_spatial_head(x,mask,coords,y,w,np.arange(4),42,cfg)
        for key,value in first.state_dict().items(): torch.testing.assert_close(value,second.state_dict()[key])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'wheels').mkdir()
            for f in ('pydicom-3.0.2-test.whl','python_gdcm-3.2.6-test.whl'): (p/'wheels'/f).write_bytes(b'wheel')
            encoder=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(1))
            torch.jit.trace(encoder,torch.rand(1,3,4,4)).save(str(p/'encoder.ts'))
            from experiments.core import LABELS
            torch.save({'state':first.state_dict(),'experiment':cfg,'labels':LABELS},p/'model.pt')
            write_weak_manifest(cfg,p,4350,58)
            with patch('experiments.spatial_submit.make_transform',return_value=lambda x:x):
                _,loaded,_,_=load_spatial_predictor(p,cfg,'cpu')
            torch.testing.assert_close(first(x[:2],mask[:2],coords[:2]),loaded(x[:2],mask[:2],coords[:2]))

    def test_standalone_notebooks_and_offline_submit(self):
        from experiments.spatial_notebooks import render_spatial
        for cfg in spatial_configs():
            for mode in ('train','submit'):
                nb=render_spatial(cfg,mode); ns={}
                for c in nb['cells']:
                    if c['cell_type']!='code': continue
                    source=''.join(c['source']); ast.parse(source)
                    self.assertNotIn('from experiments',source)
                    if not source.startswith(('import subprocess, sys','train_spatial_experiment(','submit_spatial_experiment(')):
                        exec(compile(source,'<standalone>','exec'),ns)
                self.assertEqual(ns['EXPERIMENT'],cfg)
                if mode=='submit': self.assertNotIn('fit_spatial_head',ns)

    def test_mm_order_fallback_and_cache_identity(self):
        from experiments.spatial_core import spatial_order, spatial_signature
        from experiments.spatial_train import compatible_spatial_cache
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for i in range(3): (p/f'{i}.dcm').touch()
            headers=[SimpleNamespace(ImagePositionPatient=[0,0,z],ImageOrientationPatient=[1,0,0,0,1,0]) for z in (8,2,5)]
            with patch('pydicom.dcmread',side_effect=headers):
                paths,coords=spatial_order(p)
            self.assertEqual([x.name for x in paths],['1.dcm','2.dcm','0.dcm'])
            np.testing.assert_array_equal(coords,[[0,0,1],[1,3,1],[2,6,1]])
            with patch('pydicom.dcmread',side_effect=[SimpleNamespace(InstanceNumber=i) for i in (9,1,3)]):
                _,fallback=spatial_order(p)
            np.testing.assert_array_equal(fallback[:,2],0)
            np.testing.assert_array_equal(fallback[:,1],0)
        cfg=spatial_configs()[2]
        cache=dict(view_signature=spatial_signature(cfg),encoder_sha256='a',ids=['x'])
        self.assertTrue(compatible_spatial_cache(cache,cfg,'a',['x']))
        self.assertFalse(compatible_spatial_cache(cache,cfg,'b',['x']))
        self.assertFalse(compatible_spatial_cache(cache,dict(cfg,neighbors=10),'a',['x']))

    def test_spatial_head_backpropagates_to_only_unfrozen_encoder_stage(self):
        from experiments.backbones import swin_t_encoder
        from experiments.weak_train import freeze_swin_except_last_stage
        encoder=swin_t_encoder(pretrained=False); trainable=freeze_swin_except_last_stage(encoder)
        cfg=self.config(); cfg.update(feature_dim=768,fusion='cross',position='rope_mm')
        head=SpatialHead(cfg,torch.zeros(768),torch.ones(768))
        features=encoder(torch.randn(2,3,32,32))
        mask=torch.zeros(1,6,2,dtype=torch.bool); mask[:,0]=True
        tokens=features.new_zeros(1,6,2,768); tokens[mask]=features
        coords=torch.zeros(1,6,2,3); coords[:,0,:,1]=torch.tensor([0.,3.]); coords[:,0,:,2]=1
        head(tokens,mask,coords).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in trainable))
        self.assertTrue(all(p.grad is None for p in encoder.features[0].parameters()))
        self.assertIsNotNone(head.queries.grad)

    def test_generated_offline_notebook_creates_valid_csv(self):
        from experiments.spatial_notebooks import render_spatial
        from experiments.weak_train import write_weak_manifest
        from experiments.core import LABELS
        cfg=spatial_configs()[11]; cfg.update(feature_dim=3,hidden_dim=16,crop=4)
        nb=render_spatial(cfg,'submit'); ns={}
        for c in nb['cells']:
            source=''.join(c['source'])
            if c['cell_type']=='code' and not source.startswith('submit_spatial_experiment('): exec(source,ns)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); bundle=p/'bundle'; bundle.mkdir(); (bundle/'wheels').mkdir()
            for f in ('pydicom-3.0.2-test.whl','python_gdcm-3.2.6-test.whl'): (bundle/'wheels'/f).write_bytes(b'wheel')
            encoder=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(1))
            torch.jit.trace(encoder,torch.rand(1,3,4,4)).save(str(bundle/'encoder.ts'))
            head=SpatialHead(cfg,torch.zeros(3),torch.ones(3)).eval()
            torch.save(dict(state=head.state_dict(),experiment=cfg,labels=LABELS),bundle/'model.pt')
            write_weak_manifest(cfg,bundle,4350,58)
            pd.DataFrame(dict(StudyInstanceUID=['u'])).to_csv(p/'test.csv',index=False)
            pd.DataFrame(dict(StudyInstanceUID=['u'],SeriesInstanceUID=['s'])).to_csv(p/'test_series.csv',index=False)
            pd.DataFrame(columns=['StudyInstanceUID',*LABELS]).to_csv(p/'sample_submission.csv',index=False)
            tokens=torch.randn(6,3,3); mask=torch.ones(6,3,dtype=torch.bool)
            coords=torch.zeros(6,3,3); coords[...,0]=torch.arange(3.); coords[...,2]=1
            ns['spatial_tokens']=lambda *args:(tokens,mask,coords)
            ns['make_transform']=lambda c:lambda x:x
            with patch('subprocess.check_call') as install:
                frame=ns['submit_spatial_experiment'](cfg,p,bundle,p/'out')
            self.assertIn('--no-index',install.call_args.args[0])
            expected=head(tokens[None],mask[None],coords[None]).sigmoid().detach().numpy()[0]
            np.testing.assert_allclose(frame[LABELS].to_numpy()[0],expected,rtol=1e-5)
            self.assertTrue((p/'out/submission.csv').is_file())


if __name__=='__main__': unittest.main()
