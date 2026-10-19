import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from experiments.anatomy_core import (quadrant_pool, quadrant_mean, physical_crop,
    canonical_orientation, AnatomyHead, reference_cdf)
from experiments.spatial_core import SpatialHead


class AnatomyCoreTests(unittest.TestCase):
    def test_quadrant_weighted_pool_preserves_global_average(self):
        x = torch.arange(2*7*7*8, dtype=torch.float32).reshape(2,7,7,8)
        torch.testing.assert_close(quadrant_mean(quadrant_pool(x)), x.mean((1,2)))

    def test_physical_crop_respects_anisotropic_spacing(self):
        # Two pixels vertically and four horizontally both represent 4 mm.
        y,x = torch.meshgrid(torch.arange(11.),torch.arange(21.),indexing='ij')
        result = physical_crop((y-5)*2+(x-10), [2.,1.], 8., 5)
        torch.testing.assert_close(result[2], torch.linspace(-4,4,5))
        torch.testing.assert_close(result[:,2], torch.linspace(-4,4,5))
        with self.assertRaises(ValueError): physical_crop(x,[0.,1.],8.,5)

    def test_orientation_recovers_flip_and_transpose(self):
        x=torch.arange(12.).reshape(3,4)
        base=[1,0,0,0,0,-1]
        actual, residual=canonical_orientation(x,base,'Coronal')
        torch.testing.assert_close(actual,x)
        actual,_=canonical_orientation(x.flip(1),[-1,0,0,0,0,-1],'Coronal')
        torch.testing.assert_close(actual,x)
        actual,_=canonical_orientation(x.T,[0,0,-1,1,0,0],'Coronal')
        torch.testing.assert_close(actual,x)
        self.assertLess(residual,1e-5)

    def config(self, patch):
        return dict(hidden_dim=16,feature_dim=8,heads=2,sequence=True,
            plane_embedding=True,position='rope_mm',mm_unit=3.,fusion='cross',patch_attention=patch)

    def test_mean_control_equals_existing_head(self):
        cfg=self.config(False); torch.manual_seed(7)
        old=SpatialHead(cfg,torch.zeros(8),torch.ones(8)).eval()
        new=AnatomyHead(cfg,torch.zeros(8),torch.ones(8)).eval()
        new.load_state_dict(old.state_dict())
        tokens=torch.randn(2,6,3,4,8); mask=torch.ones(2,6,3,dtype=torch.bool); coords=torch.randn(2,6,3,3)
        torch.testing.assert_close(new(tokens,mask,coords),old(quadrant_mean(tokens),mask,coords))

    def test_patch_mask_and_gradient(self):
        head=AnatomyHead(self.config(True),torch.zeros(8),torch.ones(8))
        x=torch.randn(2,6,3,4,8,requires_grad=True); mask=torch.ones(2,6,3,dtype=torch.bool);mask[:,1]=False
        coords=torch.zeros(2,6,3,3); out=head(x,mask,coords)
        corrupt=x.detach().clone();corrupt[:,1]=float('nan')
        torch.testing.assert_close(out,head(corrupt,mask,coords))
        self.assertEqual(out.shape,(2,12));out.sum().backward()
        self.assertGreater(x.grad[mask].abs().sum().item(),0)
        self.assertEqual(x.grad[~mask].abs().sum().item(),0)

    def test_cdf_uses_fixed_reference_and_is_batch_independent(self):
        reference=np.array([[.1,.2],[.3,.4],[.5,.6]])
        p=np.array([[.2,.7],[.4,.1]])
        full=reference_cdf(reference,p)
        np.testing.assert_allclose(full,np.concatenate([reference_cdf(reference,p[i:i+1]) for i in range(2)]))
        self.assertTrue(((full>0)&(full<1)).all())

    def test_head_training_statistics_exclude_heldout_extreme_values(self):
        from experiments.anatomy_train import fit_anatomy_head
        cfg=dict(self.config(True),epochs=1,batch_size=2,lr=.001,weight_decay=.01)
        x=torch.randn(4,6,2,4,8); x[3]=1e6
        mask=torch.ones(4,6,2,dtype=torch.bool);coords=torch.zeros(4,6,2,3)
        head,history=fit_anatomy_head(x,mask,coords,torch.zeros(4,12),torch.ones(4,12),[0,1,2],42,cfg)
        self.assertLess(head.mean.abs().max().item(),1.)
        self.assertEqual(history[0]['optimizer_steps'],2)

    def test_adaptation_segments_cover_only_weak_training_once(self):
        from experiments.anatomy_adapt import segment_indices,validate_checkpoint
        gold=np.array([True,False,False,False,False,False])
        groups=np.array(['00000000','00000001','00000002','00000003','00000004','0000000a'])
        pieces=[segment_indices(gold,groups,1,i)[0] for i in range(4)]
        np.testing.assert_array_equal(np.sort(np.concatenate(pieces)),[1,2,3,4])
        cfg=dict(pipeline_sha256='abc',labels_sha256='def')
        checkpoint=dict(recipe='bad',epoch=1,segment=0)
        with self.assertRaises(ValueError): validate_checkpoint(checkpoint,cfg,1,1)

    def test_ensemble_rejects_misaligned_validation_rows(self):
        import pandas as pd
        from experiments.anatomy_train import blend_oof
        from experiments.core import LABELS
        a=pd.DataFrame([['a',42,0,*([.2]*12)]],columns=['StudyInstanceUID','seed','fold',*LABELS])
        b=a.copy();b['fold']=1
        with self.assertRaises(ValueError): blend_oof([a,b])
        b=a.copy();b[LABELS]=.8
        np.testing.assert_allclose(blend_oof([a,b])[LABELS],.5)

    def test_dicom_control_matches_baseline_and_rejects_constant_image(self):
        from pydicom.dataset import FileDataset,FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        from experiments.anatomy_core import anatomy_image
        from experiments.core import image_tensor,make_transform
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.dcm';meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
            ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
            ds.Rows=32;ds.Columns=32;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
            ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
            ds.PixelData=np.arange(1024,dtype=np.uint16).tobytes();ds.save_as(path)
            cfg=dict(field_mm=None,canonical=False,crop=16,resize=18,interpolation='bicubic')
            torch.testing.assert_close(anatomy_image(path,'Coronal',cfg),image_tensor(path,make_transform(cfg)),rtol=0,atol=0)
            ds.PixelData=np.zeros(1024,dtype=np.uint16).tobytes();ds.save_as(path)
            with self.assertRaises(ValueError): anatomy_image(path,'Coronal',cfg)

    def test_quadrant_encoder_export_and_train_submit_roundtrip(self):
        from experiments.anatomy_core import QuadrantSwin
        from experiments.anatomy_train import save_anatomy_head,load_anatomy_head
        torch.set_num_threads(1);model=QuadrantSwin(False).eval()
        x=torch.randn(2,3,224,224)
        with torch.no_grad():
            traced=torch.jit.trace(model,x[:1]);torch.testing.assert_close(model(x),traced(x))
        cfg=self.config(True);head=AnatomyHead(cfg,torch.zeros(8),torch.ones(8)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'head.pt';save_anatomy_head(head,cfg,path);loaded=load_anatomy_head(path,cfg,'cpu')
            t=torch.randn(1,6,2,4,8);m=torch.ones(1,6,2,dtype=torch.bool);c=torch.zeros(1,6,2,3)
            torch.testing.assert_close(head(t,m,c),loaded(t,m,c))

    def test_synthetic_cv_saves_complete_folds_and_rank_predictions(self):
        import pandas as pd
        from experiments.anatomy_train import evaluate_anatomy_cv
        from experiments.core import LABELS
        torch.set_num_threads(1);torch.manual_seed(9)
        cfg=dict(self.config(True),id='synthetic',epochs=1,batch_size=16,lr=.001,weight_decay=.01)
        n=70;gold=np.array([True]*58+[False]*12);ids=[f'uid{i:03}' for i in range(n)]
        y=torch.from_numpy(np.random.default_rng(91).integers(0,2,(n,12)).astype(np.float32))
        tokens=torch.randn(n,6,2,4,8);mask=torch.ones(n,6,2,dtype=torch.bool);coords=torch.zeros(n,6,2,3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metrics=evaluate_anatomy_cv(tokens,mask,coords,y,torch.ones_like(y),gold,ids,cfg,root)
            a=pd.read_csv(root/'oof_predictions.csv');b=pd.read_csv(root/'oof_rank_predictions.csv')
            self.assertEqual(len(a),174);self.assertEqual(len(list((root/'fold_models').glob('*.pt'))),9)
            self.assertEqual(a.groupby('seed').StudyInstanceUID.nunique().tolist(),[58,58,58])
            self.assertTrue(np.isfinite(b[LABELS]).all().all());self.assertTrue(0<=metrics['macro_auc_mean']<=1)

    def test_tta_rejects_modified_parent_oof_before_gpu_extraction(self):
        import pandas as pd
        from unittest.mock import patch
        from experiments.anatomy_train import train_anatomy_fusion
        from experiments.core import file_hash,LABELS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'oof_predictions.csv'
            frame=pd.DataFrame([['uid',42,0,*([.2]*12)]],columns=['StudyInstanceUID','seed','fold',*LABELS])
            frame.to_csv(path,index=False);meta=dict(oof_sha256={'oof_predictions.csv':file_hash(path)})
            frame[LABELS]=.9;frame.to_csv(path,index=False)
            with patch('experiments.anatomy_train.checked_anatomy_run',return_value=(root,{},meta)):
                with self.assertRaisesRegex(ValueError,'Corrupt OOF'):
                    train_anatomy_fusion(dict(method='tta',train_dependencies=['parent']),root,root/'out')


if __name__=='__main__': unittest.main()
