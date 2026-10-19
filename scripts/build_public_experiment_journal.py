"""Build the public, CPU-only experiment journal from aggregate results.

This script does not push to Kaggle, read patient-level predictions, or train models.
Only aggregate metrics and original model definitions are embedded in the notebook.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build(analysis_dir, output_dir):
    stats = pd.read_csv(analysis_dir / 'experiments.csv')
    categories = pd.read_csv(analysis_dir / 'category_metrics_all_seeds.csv')
    keep = ['experiment', 'seed', 'label', 'positives', 'negatives', 'tn', 'fp', 'fn', 'tp',
            'recall', 'specificity', 'precision', 'f1', 'auc', 'average_precision']
    categories = categories.loc[categories.seed == 42, keep].copy()
    assert len(stats) == 31, 'Update the dated narrative before publishing a new result snapshot.'
    assert len(categories) == 31 * 12 and not categories.duplicated(['experiment', 'label']).any()
    assert (categories[['tn', 'fp', 'fn', 'tp']].sum(axis=1) == 58).all()
    assert stats.gold_studies.eq(58).all()
    assert set(categories.experiment) == set(stats.experiment)
    assert abs(stats.set_index('experiment').loc['r01_e09_s13_blend', 'macro_auc'] - .7714318550906669) < 1e-10

    cells = []
    def md(s): cells.append(nbformat.v4.new_markdown_cell(s.strip()))
    def code(s): cells.append(nbformat.v4.new_code_cell(s.strip()))

    md('''# RSNA Knee: 31 Experiments and Honest OOF Analysis

**A controlled experiment journal — 2026-10-19 snapshot**

This notebook reports our own **Gold58 development out-of-fold (OOF) results**, not a 0.95 leaderboard solution. It covers weak report labels, frozen image features, slice sampling, positional encoding/RoPE, finding-specific attention, limited fine-tuning, and ensembling. All figures below are regenerated on CPU from aggregate statistics. The model example uses synthetic tensors, not MRI data.

이 실험의 질문은 **MRI의 정보가 어느 단계에서 사라지고, 어떤 변경이 실제로 12개 소견 예측에 도움이 되는가?** 입니다. 성공한 변경뿐 아니라 성능이 낮아진 실험도 기록합니다. 현재 snapshot의 최고 개발 OOF는 R01 **0.771432 ± 0.008916 (seed SD)**입니다. 이 값과 다른 사람의 public LB 0.95x는 서로 다른 평가이므로 직접 격차로 해석하지 않습니다.

이 Notebook은 방법론·오류 분석용 Code입니다. 새 GPU 학습이나 hidden-test 예측을 실행하지 않으며 `submission.csv`를 만들지 않습니다. 원 MRI, 보고서 원문, study 식별자, 환자별 예측, 타인의 모델 가중치를 배포하지 않습니다.''')

    md('''## 1. Task, targets and evaluation

한 무릎 MRI study에는 여러 series와 slice가 있습니다. 목표는 다음 **12개 소견 각각의 점수**를 출력하는 것입니다.

`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

여러 소견이 동시에 존재할 수 있는 **multi-label classification**입니다. 12개 중 하나를 고르는 softmax/MNIST 형태가 아니라, 12개 logit에 각각 sigmoid를 적용합니다. 정답은 study 단위이며 앞쪽 slice의 정답 이미지와 뒤쪽 test 이미지를 짝지어 비교하는 문제가 아닙니다.

학습은 소견별 BCEWithLogitsLoss를 기반으로 합니다. 알려진 weak label에는 confidence 가중치, gold에는 별도 가중치, UNK에는 0 mask를 적용합니다. 현재 known weak 가중치는 `0.25 + 0.75 × confidence`, gold 가중치는 3입니다. 평가 BCE는 gold의 unweighted 값이며 학습 loss의 절댓값과 직접 비교하지 않습니다.

대회 점수는 12개 소견별 ROC-AUC의 평균입니다. **AUC 0.95는 정확도 95%가 아닙니다.** 한 소견의 양성 study가 음성 study보다 높은 점수를 받는 순위를 측정합니다. 아래 혼동행렬은 별도로 threshold=0.5를 사용합니다. [Competition evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)''')

    md('''## 2. Validation contract and weak labels

- Gold 58개 study, 동일한 3-fold 체계, seed 42/2026/3407에서 OOF 예측을 평가했습니다. 표의 AUC는 seed별 전체 OOF AUC를 구한 뒤 평균한 값입니다. seed SD는 세 값의 변동이며 신뢰구간이 아닙니다.
- Gold만 쓰는 대조군과, 필터링한 report weak 4,292개를 추가하는 실험을 비교했습니다. 후자는 gold까지 총 4,350개 학습 후보입니다. 각 gold validation fold는 해당 모델의 gradient 학습에서 제외합니다.
- 약라벨은 판독 보고서를 해석해 만든 soft target입니다. 영상별로 전문가가 다시 확정한 정답과 같지 않습니다. 알려지지 않았다는 UNK는 정상이라는 NO와 구분합니다. 공개 teacher의 선택 과정과 gold에 대한 독립성 전체를 증명한 것은 아닙니다.
- 58개를 반복적으로 개발·선택에 사용했습니다. 따라서 이 OOF도 완전히 손대지 않은 최종 test가 아닙니다. 세 seed가 독립 환자 174명을 의미하지 않습니다. 외부 public 모델이 gold로 checkpoint를 골랐다면 그 gold 성적과도 평가 조건이 다릅니다.
- 배포용 모델은 full-fit seed42입니다. 세 seed/fold 평가를 했다는 사실이 배포 시 모든 fold/seed를 앙상블했다는 뜻은 아닙니다.

공개 약라벨 출처: [Pilkwang report labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels). 이 노트에는 해당 label table이나 보고서를 포함하지 않습니다.''')

    md('''## 3. Our current image-to-prediction pipeline

```text
DICOM study
  → 3 planes × 2 fat-suppression slots (up to 6)
  → order slices by ImagePositionPatient · cross(ImageOrientationPatient)
  → choose centres (default 25–75%) and neighbouring slices
  → deduplicate single-slice tokens; retain order and relative mm coordinates
  → intensity normalization, resize 232, centre crop 224
  → ImageNet Swin-T image encoder (mostly frozen)
  → global-pooled 768-d image feature
  → training-fold mean/std normalization → linear projection to 128-d
  → optional plane/protocol embeddings and within-slot slice self-attention/RoPE
  → class-specific slot pooling + gate (S10)
       OR 12 learned class queries cross-attending to all slice tokens (S11)
  → 12 logits → sigmoid → 12 independent scores
```

E00/E01은 ResNet18 대조군, E02에서 backbone 후보를 비교했습니다. 현재 주력은 Swin-T이며, Swin 자체의 이미지 내부 attention과 그 뒤에 추가한 **study/slice attention은 다른 단계**입니다.

S02의 `5`와 S03의 `10`은 각 중심 주변의 이웃 폭입니다. study 전체가 5장/10장이라는 뜻이 아닙니다. 겹치는 slice를 중복 제거하므로 token 수도 정확히 두 배가 되지 않습니다. S00/E08의 triplet은 인접 3장을 RGB 세 채널처럼 한 encoder 입력으로 넣는 방식이고, S01–S03은 각 slice를 별도 token으로 만드는 방식입니다.

**중요한 한계:** 현재 cross-attention 앞에는 이미 이미지 global pooling과 768→128 투영이 있습니다. cross-attention이 원래 사라진 patch 정보를 복원해 주지는 않습니다. 또한 mm RoPE를 쓴다고 PixelSpacing에 맞는 물리적 이미지 crop을 적용한 것은 아닙니다. 그 두 변경은 다음 실험에서 분리합니다.''')

    code('''import io, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import display

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
torch.set_num_threads(2)
print('Snapshot: 2026-10-19; CPU analysis; no patient-level records')''')
    code('stats = pd.read_json(io.StringIO(' + repr(stats.to_json(orient='records', double_precision=15)) + '))\n'
         + 'category = pd.read_json(io.StringIO(' + repr(categories.to_json(orient='records', double_precision=15)) + '))\n'
         + '''assert len(stats) == 31 and len(category) == 372
assert (category[['tn','fp','fn','tp']].sum(axis=1) == 58).all()
display(stats[['experiment','macro_auc','seed_sd','bce','training_studies']].round(6))
stats.to_csv(OUT / 'experiment_summary.csv', index=False)
category.to_csv(OUT / 'category_metrics_seed42.csv', index=False)''')

    md('''## 4. What changed, and what happened?

| Comparison | Mean OOF AUC | Interpretation within this development set |
|---|---|---|
| E06 gold → E07 weak labels | .585972 → .741941 | Largest improvement in this controlled comparison |
| E07 → E08 triplet → E09 protocol slots | .741941 → .742926 → .744113 | Small incremental changes; uncertainty matters |
| S01 neighbour3 → S02 neighbour5 → S03 neighbour10 | .716563 → .732541 → .724005 | Five won under this setting; not a universal optimal slice count |
| S02 → S04 sequence attention → S05 additive position | .732541 → .726607 → .713061 | More complex heads did not automatically help |
| S06 index RoPE → S07 physical-mm RoPE | .721530 → .711794 | Position encoding needs an effective representation and validation |
| S09 MLP → S10 gate / S11 cross | .724314 → .739986 / .736386 | Class-aware fusion was promising; cross did not beat gate |
| S11 → S12 limited fine-tuning | .736386 → .718888 | Current one-epoch setup underperformed; not proof against all fine-tuning |
| S13 → R01 | .748420 → .771432 | Complementary pathways helped in the development OOF |

R01 combines **0.5 E09 + 0.25 S11 + 0.25 S12**, using probabilities. It is not the probability average of three randomly named backbones. S12 can be a useful ensemble member even when its standalone score is weaker; this must be measured, not assumed.

S02→S03 actually changed unique image tokens from 409,239 to 526,504 (+28.7%) and feature extraction from 4.60h to 5.52h. Possible causes of the lower score include redundancy, extra background, model/data complexity, training settings, and sampling variation. These are hypotheses, not proven causes.''')
    code('''ordered = stats.sort_values('macro_auc')
fig, ax = plt.subplots(figsize=(10, 10))
colors = ['#cf6b36' if e.startswith('r01') else '#246d8c' for e in ordered.experiment]
ax.barh(ordered.experiment, ordered.macro_auc, xerr=ordered.seed_sd, color=colors, alpha=.9)
ax.set(xlim=(.45,.82), xlabel='Mean Gold58 OOF macro ROC-AUC (error bar: seed SD)',
       title='31 completed configurations — same development evaluation')
ax.grid(axis='x', alpha=.2)
fig.tight_layout(); fig.savefig(OUT/'experiment_auc.png', dpi=160); plt.show()''')

    md('''## 5. Confusion matrices: where do we miss findings?

각 소견별 **2×2 행렬**을 사용합니다. 하나의 12×12 혼동행렬을 쓰면 동시에 여러 소견이 가능한 문제를 잘못 표현합니다.

| | Predicted absent | Predicted present |
|---|---|---|
| Actually absent | TN: 정상 소견을 없다고 판단 | FP: 없는 소견을 있다고 판단 |
| Actually present | FN: 있는 소견을 놓침 | TP: 있는 소견을 맞힘 |

- **Recall / 재현율 = TP / (TP+FN)**: 실제 소견이 있는 study 중 얼마나 찾아냈는가? FN이 늘면 내려갑니다.
- **Precision / 정밀도 = TP / (TP+FP)**: 있다고 예측한 것 중 실제로 맞은 비율은 얼마인가? FP가 늘면 내려갑니다.
- TN이 많으면 accuracy가 높아도 양성 소견을 놓칠 수 있습니다. 희귀 소견은 양성 수가 작아 몇 건 차이가 recall을 크게 바꿉니다.
- AUC는 threshold 하나에 고정되지 않습니다. 따라서 AUC가 좋아도 threshold .5의 recall이 낮을 수 있습니다. threshold 선택은 별도 validation에서 해야 합니다.

아래는 seed42 OOF, threshold=.5입니다. S13→R01의 전체 FN은75→67, FP는144→132로 줄었지만 **MCL FN은6→7, Fracture FN은11→13**으로 늘었습니다. 평균 개선과 소견별 개선은 다릅니다.''')
    code('''def compare_experiments(before, after):
    fields = ['tn','fp','fn','tp','recall','precision','auc']
    a = category.loc[category.experiment.eq(before)].set_index('label')[fields]
    b = category.loc[category.experiment.eq(after)].set_index('label')[fields]
    assert len(a) == len(b) == 12
    result = a.add_suffix('_before').join(b.add_suffix('_after'))
    result['delta_fn'] = result.fn_after - result.fn_before
    result['delta_fp'] = result.fp_after - result.fp_before
    display(result.round(4))
    return result

comparison = compare_experiments('s13_ensemble', 'r01_e09_s13_blend')
fig, axes = plt.subplots(3, 4, figsize=(13, 9))
rows = category.loc[category.experiment.eq('r01_e09_s13_blend')]
for ax, row in zip(axes.flat, rows.itertuples()):
    matrix = np.array([[row.tn,row.fp],[row.fn,row.tp]])
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=58)
    for r in range(2):
        for c in range(2):
            name = [['TN','FP'],['FN','TP']][r][c]
            ax.text(c,r,f'{name} {matrix[r,c]}',ha='center',va='center',color='black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Pred -','Pred +'],yticklabels=['True -','True +'],
           title=f'{row.label}\\nRecall {row.recall:.2f} | Precision {row.precision:.2f}')
fig.suptitle('R01: 12 binary confusion matrices — Gold58 OOF, seed42, threshold 0.5')
fig.tight_layout(); fig.savefig(OUT/'r01_confusions.png', dpi=160); plt.show()

fig, axes = plt.subplots(1,2,figsize=(13,5))
for ax, field in zip(axes,['fn','fp']):
    comparison[[field+'_before',field+'_after']].plot.barh(ax=ax,color=['#84949f','#cf6b36'])
    ax.set_title(field.upper() + ': S13 vs R01'); ax.set_xlabel('Number of studies'); ax.grid(axis='x',alpha=.2)
fig.tight_layout(); fig.savefig(OUT/'s13_r01_error_changes.png', dpi=160); plt.show()
# To compare any two completed experiments, call compare_experiments with their IDs.''')

    md('''## 6. Original model code: slice tokens → twelve outputs

다음 정의는 우리 `experiments/spatial_core.py`에서 그대로 추출한 원본입니다. 이미지 encoder 다음의 head이며, MRI 전체 학습 파이프라인을 대체하지 않습니다. 실제 학습에서는 training fold의 feature mean/std와 유효 slice mask를 전달합니다. `selected_windows`는 이웃3/5/10과 중복 제거의 차이를 보여 줍니다.

실행 예제는 random feature/target을 사용하는 **shape·gradient 확인용**입니다. 의료 예측이나 성능 측정 결과가 아닙니다.''')
    source = (ROOT/'experiments/spatial_core.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    wanted = {'selected_windows','rotary','SliceBlock','SpatialHead'}
    definitions = [ast.get_source_segment(source,n) for n in tree.body
                   if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    assert len(definitions) == 4
    code('\n\n'.join(definitions))
    code('''torch.manual_seed(42)
demo_config = dict(feature_dim=768, hidden_dim=128, heads=4, sequence=True,
                   plane_embedding=True, position='rope_mm', mm_unit=3.0, fusion='cross')
model = SpatialHead(demo_config, torch.zeros(768), torch.ones(768))
B,S,T,D = 2,6,5,768
tokens = torch.randn(B,S,T,D)
mask = torch.ones(B,S,T,dtype=torch.bool)
mask[1,5] = False  # One missing series; padded tokens must not participate.
coords = torch.zeros(B,S,T,3)
coords[...,0] = torch.arange(T)
coords[...,1] = torch.arange(T) * 3.0
coords[...,2] = 1.0  # Physical DICOM coordinates available.
logits = model(tokens,mask,coords)
synthetic_targets = torch.rand(B,12)
synthetic_loss = nn.functional.binary_cross_entropy_with_logits(logits,synthetic_targets)
synthetic_loss.backward()
assert logits.shape == (B,12) and torch.isfinite(logits).all()
assert model.project.weight.grad is not None
assert torch.isfinite(model.project.weight.grad).all()
print('Synthetic shape check:',tuple(tokens.shape),'->',tuple(logits.shape))
print('BCE backward succeeded; these random scores have no performance meaning.')
for neighbours in (3,5,10):
    cfg = dict(centers=5, neighbors=neighbours, representation='single')
    picked = selected_windows(30,cfg)
    print('Neighbours',neighbours,':',len(picked),'unique tokens from a 30-slice series')''')

    md('''## 7. What public high-score notebooks add

We inspected **25 original source snapshots: 22 high-score candidates/derivatives and 3 training or architecture references**. This is not a claim that 22 independently reproduced 0.95 solutions were verified. Some current versions are unscored, reuse another author's score, require private extra weights, or only launch external code. Public figures below remain author reports.

1. **Input geometry and coverage:** the compact R7 manifest uses ConvNeXt-small, render384/crop352, physical spacing and knee-centred views. Raptor's commonly named 384 branch actually takes a320 crop from a384 rendered image and evaluates94 triplets over a96-slice stack. Copying a model name alone misses this contract.
2. **Image adaptation:** R7 records16epochs and4folds; the original Raptor training code adapts backbone+head with report soft targets. Our search mostly trained heads on frozen Swin features, with a limited one-epoch adaptation experiment.
3. **Pooling before information is lost:** R7's active spatial attention works on the image feature map before class-specific window pooling. Our cross-attention starts after image global pooling. This is a meaningful next ablation, not evidence that all Transformer heads must win.
4. **Complementarity:** the224 and320 Raptor branches use distinct trained weights and fields of view. Their rank blend is different from merely resizing one network's input. Anatomical mirror TTA must respect plane/channel order.
5. **Negative evidence matters:** goodpjw reports adding S6 reduced .958→.957; tuning12weights onGold58 reduced the reported LB to.955. hitarthjain0 reports a weak extra model lowered.950→.948. More members and more tuning do not guarantee improvement.
6. **Read execution, not labels:** several notebooks are the same inference family under different titles. We also found a four-model candidate explicitly marked unmeasured and a public.957 article without executable inference in that source version.

For AUC, a strictly increasing post-transform preserves order. Dividing a final score by a positive temperature alone cannot improve AUC. Rank or probit **before mixing models** can change the order, but neither is universally superior. Correlation-based clinical score adjustments also need validation; plausible stories are not causal evidence.''')

    snapshots = json.loads((ROOT/'outputs/research/kaggle_code_20261019/snapshots.json').read_text(encoding='utf-8'))
    lines = ['| # | Source snapshot | Version |','|---|---|---|']
    for i,r in enumerate(snapshots,1):
        title = r['ref'].replace('|','')
        lines.append(f"| {i} | [{title}](https://www.kaggle.com/code/{r['ref']}) | {r['version']} |")
    md('### Source ledger\n\n'+'\n'.join(lines)+'''

Model configuration evidence: [R7 manifest dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss). These links may later show newer versions; the table records the source versions inspected on2026-10-19. Original code/model licenses and dataset conditions remain with their authors. The source review was static; no third-party checkpoint or embedded script was executed for this journal.''')

    md('''## 8. Next experiments — hypotheses, not completed results

현재 별도 자동 queue에는 confidence weighting, 소견별 loss 균형, 중심 범위 확대, 중심 수 증가, head warmup 후 backbone fine-tuning 비교가 있습니다. 이 노트의 완료 결과표에는 포함하지 않았습니다.

| Priority | Controlled change | Evidence to collect |
|---|---|---|
| P00 | R01 probability blend vs fixed-weight rank blend | Same-fold OOF AUC/AP, error correlation, uncertainty; retain probabilities for calibration |
| P01 | Physical mm crop, then laterality normalization in separate steps | Header coverage, image contract, medial/lateral category errors |
| P02 | Broad coverage vs dense neighbouring slices, same unique-token budget | Coverage audit and paired OOF comparison |
| P03 | Global pooling vs class-specific spatial patch pooling | Same encoder/input/training data; memory/time and category AUC |
| P04 | Warmup then last-stage1/3/6epoch adaptation | Training curves, three seeds, deterioration on weak categories |
| P05 | Best checkpoint vs predeclared EMA/weight averaging | Same training run, fixed checkpoint-selection rule |
| P06 | Plain vs geometry-consistent mirror TTA | Same weights, OOF scores, inference cost |
| P07 | Backbone comparison after fixing input/training conditions | Swin/ConvNeXt/CoAtNet under matched budget |

채택은 평균 AUC 한 숫자만으로 결정하지 않습니다. seed 일관성, MCL/Fracture 등 취약 소견, paired uncertainty, 계산 비용을 함께 기록합니다. 58gold에 소견별 비율을 많이 맞추는 실험은 별도의 inner validation/독립 holdout 설계가 필요합니다. 외부 모델이 같은 gold로 선택되었다면 그 출력은 새로운 clean OOF라고 부르지 않습니다.

이 프로젝트에서 얻는 경험은 모델 이름을 바꾸는 것보다 **의료영상의 좌표·입력 계약, 약한 감독의 불확실성, 통제된 실험, 오류 분석, 재현 가능한 학습/추론 자동화**를 연결하는 데 있습니다. 대회 성능이 실제 진료 유효성을 검증하는 것은 아닙니다. 새 결과가 검증되면 이 노트를 버전별로 업데이트할 계획입니다.''')
    code('''summary = {'snapshot':'2026-10-19','completed_experiments':len(stats),
           'evaluation':'Gold58 development 3-fold OOF, three training seeds',
           'best_experiment':'r01_e09_s13_blend','mean_auc':float(stats.set_index('experiment').loc['r01_e09_s13_blend','macro_auc']),
           'official_leaderboard_score_claimed':False,
           'public_sources_inspected':25,'patient_level_records_included':False}
(OUT/'journal_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))''')

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = dict(display_name='Python 3',language='python',name='python3')
    nb.metadata.language_info = dict(name='python',version='3.11')
    nbformat.validate(nb)
    output_dir.mkdir(parents=True,exist_ok=True)
    path = output_dir/'rsna_knee_experiment_journal.ipynb'
    nbformat.write(nb,path)
    # Kaggle rejected public creation with this competition source on 2026-10-19.
    # Preserve the competition association and prepare a private reviewable version.
    metadata = dict(id='choiyoungwhan/rsna-knee-experiments-and-error-analysis',
        title='RSNA Knee Experiments and Error Analysis',
        code_file=path.name,language='python',kernel_type='notebook',is_private=True,
        enable_gpu=False,enable_internet=False,
        competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=[],kernel_sources=[])
    (output_dir/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    receipt = dict(notebook=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source_model_sha256=hashlib.sha256(source.encode()).hexdigest(),cells=len(cells),
                   experiments=len(stats),category_rows=len(categories),aggregate_only=True)
    (ROOT/'outputs/research/kaggle_code_20261019/public_journal_build.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir',type=Path,default=ROOT/'outputs/analysis/2026-10-19')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'notebooks/public_journal')
    args = parser.parse_args()
    build(args.analysis_dir,args.output_dir)
