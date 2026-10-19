"""Read-only Kaggle source snapshots for review. Never execute downloaded code."""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/kaggle_code_20261019'


def collect(ref):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    from experiments.run_queue import retry_kaggle_read
    api=KaggleApi();api.authenticate()
    folder=OUT/ref.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
    if (folder/'snapshot.json').exists():
        return json.loads((folder/'snapshot.json').read_text(encoding='utf-8'))
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=ref.split('/')
    with api.build_kaggle_client() as client:
        response=retry_kaggle_read(lambda:client.kernels.kernels_api_client.get_kernel(req))
    meta=response.metadata
    if meta.is_private: raise ValueError('Expected public notebook: '+ref)
    source=response.blob.source
    fields=('id','ref','title','author','slug','last_run_time','language','kernel_type','is_private',
            'enable_gpu','enable_internet','total_votes','dataset_data_sources','kernel_data_sources',
            'competition_data_sources','model_data_sources','current_version_number','docker_image','machine_shape')
    data={k:getattr(meta,k,None) for k in fields}
    data.update(retrieved_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(source.encode()).hexdigest())
    (folder/'metadata.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    (folder/('source.ipynb' if meta.kernel_type=='notebook' else 'source.py')).write_text(source,encoding='utf-8')
    nb=json.loads(source) if meta.kernel_type=='notebook' else dict(cells=[dict(cell_type='code',source=source)])
    code=[]; markdown=[]; index=[]; embedded=[]
    for i,cell in enumerate(nb.get('cells',[])):
        src=cell.get('source','');src=''.join(src) if isinstance(src,list) else src
        kind=cell.get('cell_type','');target=code if kind=='code' else markdown
        target.append(f'\n### CELL {i} ({kind})\n'+src)
        index.append(dict(cell=i,kind=kind,lines=len(src.splitlines()),chars=len(src),preview=src[:160]))
        if kind=='code':
            try: tree=ast.parse(src)
            except SyntaxError: continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.count('\n')>8 and len(node.value)>500:
                    embedded.append(f'\n### CELL {i}, string at source line {node.lineno}\n'+node.value)
    (folder/'code.txt').write_text('\n'.join(code),encoding='utf-8')
    (folder/'markdown.md').write_text('\n'.join(markdown),encoding='utf-8')
    (folder/'embedded_strings.txt').write_text('\n'.join(embedded),encoding='utf-8')
    (folder/'cell_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    row=dict(ref=ref,title=meta.title,version=meta.current_version_number,votes=meta.total_votes,
             source_sha256=data['sha256'],source_chars=len(source),cells=len(index),folder=str(folder.relative_to(ROOT)))
    (folder/'snapshot.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Saved {ref} v{row['version']} ({len(index)} cells)",flush=True)
    return row


if __name__=='__main__':
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(collect,selection['selected']))
    (OUT/'snapshots.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Complete: {len(rows)} public source snapshots, no downloaded code executed.',flush=True)
