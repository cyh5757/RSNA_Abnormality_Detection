"""Same-study, same-threshold comparisons and a standalone Korean experiment viewer."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.core import LABELS

PARENTS=dict(e01_resnet18_5='e00_resnet18_3',e02a_dinov2_5='e01_resnet18_5',e02b_swin_5='e01_resnet18_5',
    e02c_convnext_5='e01_resnet18_5',e02d_vit_5='e01_resnet18_5',
    e03_attention='e02b_swin_5',e04_position='e03_attention',e05_plane='e04_position',
    e07_weak_labels='e06_gold_control',e08_triplet='e07_weak_labels',e09_protocols='e08_triplet',
    e10_class_mil='e09_protocols',e11_finetune='e10_class_mil',e12_ensemble='e10_class_mil',
    s01_slice3='s00_triplet',s02_slice5='s01_slice3',s03_slice10='s02_slice5',s04_sequence='s02_slice5',
    s05_position='s04_sequence',s06_rope_index='s04_sequence',s07_rope_mm='s04_sequence',s08_plane='s07_rope_mm',
    s09_mlp='s08_plane',s10_gate='s08_plane',s11_cross='s08_plane',s12_finetune='s11_cross',s13_ensemble='s11_cross',
    r01_e09_s13_blend='s13_ensemble',r02_wide='s10_gate',r03_dense='s10_gate',r04_confidence='s10_gate',
    r05_label_loss='s10_gate',r06_warmup='s12_finetune',r07_long_ft='r06_warmup',
    t00_position_control='s11_cross',t01_position_sum='t00_position_control',t02_position_joint='t01_position_sum',
    u00_control='s11_cross',u01_mm_crop='u00_control',u02_orientation='u01_mm_crop',u03_patch_attention='u02_orientation',
    u04_full_ft1='u03_patch_attention',u05_full_ft3='u04_full_ft1',u06_flip_tta='u05_full_ft3',
    u07_probability_blend='u05_full_ft3',u08_rank_blend='u07_probability_blend')


def comparison_tables(summary,detail):
    chosen=detail[detail.seed==42].copy()
    counts=chosen.groupby('experiment')[['tn','fp','fn','tp']].sum()
    counts['micro_recall']=counts.tp/(counts.tp+counts.fn)
    counts['micro_precision']=counts.tp/(counts.tp+counts.fp)
    aggregate=summary.merge(counts,on='experiment')
    indexed=chosen.set_index(['experiment','label']); means=detail.groupby(['experiment','label']).auc.mean()
    rows=[]
    for eid,parent in PARENTS.items():
        if eid not in chosen.experiment.values or parent not in chosen.experiment.values: continue
        for label in LABELS:
            a,b=indexed.loc[(eid,label)],indexed.loc[(parent,label)]
            row=dict(experiment=eid,parent=parent,label=label,delta_auc_mean=means[eid,label]-means[parent,label])
            row.update({f'delta_{k}':a[k]-b[k] for k in ('tn','fp','fn','tp','recall','precision')})
            rows.append(row)
    return aggregate,pd.DataFrame(rows)


def write_comparisons(output):
    output=Path(output)
    summary=pd.read_csv(output/'experiments.csv');detail=pd.read_csv(output/'category_metrics_all_seeds.csv')
    aggregate,deltas=comparison_tables(summary,detail)
    aggregate.to_csv(output/'confusion_comparison_seed42.csv',index=False)
    deltas.to_csv(output/'parent_category_deltas.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=detail[detail.seed==42]
    for value,title,filename in [('recall','Recall = TP / (TP + FN)','recall_all_experiments.png'),
                                 ('precision','Precision = TP / (TP + FP)','precision_all_experiments.png')]:
        matrix=chosen.pivot(index='label',columns='experiment',values=value).loc[LABELS,summary.experiment]
        fig,ax=plt.subplots(figsize=(18,8),layout='constrained')
        im=ax.imshow(matrix,aspect='auto',cmap='YlGnBu',vmin=0,vmax=1)
        ax.set(xticks=range(len(matrix.columns)),xticklabels=[x.split('_')[0].upper() for x in matrix.columns],
            yticks=range(12),yticklabels=LABELS,title=f'{title} | Gold58 OOF, seed42, p >= 0.5')
        for (i,j),v in np.ndenumerate(matrix.to_numpy()):
            ax.text(j,i,'–' if np.isnan(v) else f'{v*100:.0f}',ha='center',va='center',fontsize=7,color='white' if v>.65 else '#123')
        fig.colorbar(im,ax=ax);fig.savefig(output/filename,dpi=150);plt.close(fig)
    payload=dict(rows=json.loads(chosen.to_json(orient='records')),summary=json.loads(aggregate.to_json(orient='records')),
                 means=json.loads(detail.groupby(['experiment','label']).auc.mean().reset_index().to_json(orient='records')),
                 parents=PARENTS)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MRI 실험별 혼동행렬 비교</title><style>
body{font:16px/1.65 system-ui,sans-serif;margin:32px auto;max-width:1400px;padding:0 20px;color:#182333;background:#f5f7fa}h1{font-size:28px}h2{font-size:22px}
.card{background:white;padding:20px;border:1px solid #d9e1e8;border-radius:10px}select{font:inherit;padding:8px;max-width:100%}.controls{display:flex;gap:25px;flex-wrap:wrap;position:sticky;top:0;background:#f5f7fa;padding:16px 0}
#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}.matrix{display:grid;grid-template-columns:1fr 1fr;text-align:center}.matrix div{padding:8px;border:1px solid white}.good{background:#dbeddf}.bad{background:#fce4d7}.meta{font-size:13px;color:#435268}table{border-collapse:collapse;width:100%;background:white}td,th{padding:7px 12px;border-bottom:1px solid #dae2e9;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}a{color:#15599d}code{background:#e5eaf0;padding:2px 4px}</style>
<h1>MRI 실험 노트 · 소견별 혼동행렬 비교</h1>
<p>동일한 정답 검사 58건, 동일 3-fold OOF, seed 42, <code>확률 ≥ 0.5</code>를 사용합니다. AUC 평균은 3개 seed 평균입니다. 같은 검사가 반복되므로 174명의 독립 표본이 아닙니다. 아래 수치는 공개 리더보드 점수가 아닙니다.</p>
<div class="card"><b>행 = 실제 정답 / 열 = 예측</b><p>TN: 없는데 없다고 맞힘 · FP: 없는데 있다고 오탐 · FN: 있는데 없다고 놓침 · TP: 있는데 있다고 맞힘.</p>
<p><b>재현율 TP/(TP+FN)</b>: 실제 이상 중 얼마나 찾았나? <b>정밀도 TP/(TP+FP)</b>: 이상이라고 한 것 중 얼마나 맞았나? 예: 외측 반월상연골 S13은 TP 6, FN 17 → 재현율 26.1%; TP 6, FP 10 → 정밀도 37.5%.</p>
<p>0.5를 낮추면 일반적으로 놓침은 줄고 오탐은 늘어납니다. 이 판정 경계 조정 자체가 원래 확률의 AUC를 높이지는 않습니다. 12개 소견은 동시에 존재할 수 있어 각기 2×2로 봅니다.</p></div>
<div class="controls"><label>비교 A <select id="a"></select></label><label>비교 B <select id="b"></select></label></div>
<div id="grid"></div><h2>전체 실험 요약</h2><p>TP/FN/FP/TN 합계는 58×12=696개의 소견 판정입니다. 696명의 환자가 아닙니다. micro 재현율·정밀도는 소견 빈도의 영향을 받으며 대회 macro AUC와 다릅니다.</p>
<div class="scroll"><table id="summary"></table></div>
<script>const DATA=__DATA__;
const pct=x=>x==null?'정의 불가':(100*x).toFixed(1)+'%';const fmt=x=>x==null?'—':x.toFixed(4);
for(const id of ['a','b']){document.getElementById(id).innerHTML=DATA.summary.map(r=>`<option>${r.experiment}</option>`).join('');document.getElementById(id).onchange=render;}
document.getElementById('a').value='s02_slice5';document.getElementById('b').value='s03_slice10';
function panel(r){return `<b>${r.experiment.split('_')[0].toUpperCase()}</b><div class="meta">열: 예측 없음 / 있음<br>행: 실제 없음 / 있음</div><div class="matrix"><div class="good">TN ${r.tn}</div><div class="bad">FP ${r.fp}</div><div class="bad">FN ${r.fn}</div><div class="good">TP ${r.tp}</div></div><div>재현율 ${pct(r.recall)}<br>정밀도 ${pct(r.precision)}</div><div class="meta">AUC(seed42) ${fmt(r.auc)}<br>AUC(3-seed 평균) ${fmt(DATA.means.find(x=>x.experiment===r.experiment&&x.label===r.label).auc)}</div>`;}
function render(){let a=document.getElementById('a').value,b=document.getElementById('b').value;document.getElementById('grid').innerHTML=DATA.rows.filter(r=>r.experiment===a).map(x=>{let y=DATA.rows.find(r=>r.experiment===b&&r.label===x.label);return `<section class="card"><h2>${x.label}</h2><div class="pair"><div>${panel(x)}</div><div>${panel(y)}</div></div><p class="meta">B−A: FN ${y.fn-x.fn>=0?'+':''}${y.fn-x.fn}, FP ${y.fp-x.fp>=0?'+':''}${y.fp-x.fp} (음수면 해당 오류 감소)</p></section>`}).join('');}
document.getElementById('summary').innerHTML='<tr><th>실험</th><th>평균 AUC</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th><th>재현율 micro</th><th>정밀도 micro</th></tr>'+DATA.summary.map(r=>`<tr><td>${r.experiment}</td><td>${fmt(r.macro_auc)}</td><td>${r.tn}</td><td>${r.fp}</td><td>${r.fn}</td><td>${r.tp}</td><td>${pct(r.micro_recall)}</td><td>${pct(r.micro_precision)}</td></tr>`).join('');render();</script></html>'''
    (output/'confusion_comparison.html').write_text(page.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    lines=['# 전체 실험의 혼동행렬 비교','',
        '동일 Gold58 OOF, seed42, 임계값 0.5. AUC는 3개 seed 평균. 총 판정수 696 = 58검사 × 12소견.','',
        '[두 실험 나란히 비교하는 HTML](confusion_comparison.html) · [소견별 부모 대비 변화](parent_category_deltas.csv)','',
        '| 실험 | AUC | TN | FP | FN | TP | micro 재현율 | micro 정밀도 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate.itertuples():
        lines.append(f'| [{r.experiment}](confusion_{r.experiment}.png) | {r.macro_auc:.4f} | {r.tn} | {r.fp} | {r.fn} | {r.tp} | {r.micro_recall:.1%} | {r.micro_precision:.1%} |')
    (output/'confusion_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/analysis/2026-10-19'))
    write_comparisons(parser.parse_args().output)
