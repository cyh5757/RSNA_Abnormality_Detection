import ast
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from experiments.anatomy_configs import anatomy_configs
from experiments.anatomy_notebooks import build_anatomy,render_anatomy,render_adaptation
from experiments.run_anatomy_queue import predecessor_ready,adaptation_stages


class AnatomyQueueTests(unittest.TestCase):
    def test_changed_source_blocks_builder_and_upload_after_quota_wait(self):
        from experiments.run_anatomy_queue import checked_builder,checked_gpu_wait
        configs=anatomy_configs();changed=[dict(c,pipeline_sha256='modified-on-disk') for c in configs]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'notebooks'
            with patch('experiments.run_anatomy_queue.anatomy_configs',return_value=changed):
                with self.assertRaisesRegex(ValueError,'source'): checked_builder([configs[0]],root)
            self.assertFalse(root.exists())
            with patch('experiments.run_anatomy_queue.anatomy_configs',side_effect=[configs,changed]), \
                    patch('experiments.run_anatomy_queue.wait_for_gpu'):
                with self.assertRaisesRegex(ValueError,'source'): checked_gpu_wait(None,configs[0],1.,Path(tmp)/'quota.json')

    def test_predecessor_missing_or_stopped_cannot_start_successor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertFalse(predecessor_ready(root))
            (root/'queue_status.json').write_text(json.dumps(dict(status='stopped')))
            with self.assertRaises(RuntimeError): predecessor_ready(root)

    def test_continuation_stage_sequence_does_not_repeat_first_epoch(self):
        cfgs=anatomy_configs()
        a=adaptation_stages(cfgs[4]);b=adaptation_stages(cfgs[5])
        self.assertEqual([(s['epoch'],s['segment']) for s in a],[(1,i) for i in range(4)])
        self.assertEqual([(s['epoch'],s['segment']) for s in b],[(e,i) for e in (2,3) for i in range(4)])
        self.assertEqual(b[0]['previous'],a[-1]['id'])

    def test_every_notebook_is_standalone_and_submit_is_offline(self):
        cfgs=anatomy_configs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build_anatomy(cfgs,root)
            for cfg in cfgs:
                heads=[]
                for mode in ('train','submit'):
                    nb=json.loads((root/cfg['id']/mode/(mode+'.ipynb')).read_text(encoding='utf-8'))
                    source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
                    tree=ast.parse(source);compile(tree,'notebook','exec')
                    self.assertFalse(any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('experiments.') for n in ast.walk(tree)))
                    heads.append(ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnatomyHead')))
                    meta=json.loads((root/cfg['id']/mode/'kernel-metadata.json').read_text())
                    self.assertTrue(meta['is_private']);self.assertTrue(meta['enable_gpu'])
                    if mode=='submit': self.assertFalse(meta['enable_internet'])
                self.assertEqual(*heads)
            for cfg in cfgs[4:6]:
                for stage in adaptation_stages(cfg):
                    nb=render_adaptation(cfg,stage)
                    compile('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'),'adapt','exec')


if __name__=='__main__': unittest.main()
