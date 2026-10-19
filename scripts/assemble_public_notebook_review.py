"""Assemble the dated source-review index and Korean report without shell encoding loss."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research/kaggle_code_20261019'
snapshots = json.loads((OUT/'snapshots.json').read_text(encoding='utf-8'))
a = json.loads((OUT/'review_part_a.json').read_text(encoding='utf-8'))['records']
b = [
    ('.956→.957 reported', 'private extra branch; public base fallback'),
    ('.957 reference only', 'current V33 candidate explicitly unmeasured'),
    ('.957 reported', 'analytical article; no executable inference in v12'),
    ('.956 reported', 'Raptor + R7, reused inference recipe'),
    ('.954 reported', 'Raptor .7 + older compact .3'),
    ('.956 displayed', '224 branch addition; rank movement author report'),
    ('.956 displayed', '224 anatomical mirror; rank movement author report'),
    ('.950 historical', 'new V5 routing not independently scored; negative blending results'),
    ('.9564+ claimed', 'probit/cross-prior claims not independently verified'),
    ('>.956 claimed', 'same 3-branch family; gold table provenance incomplete'),
    ('.951+ claimed', 'two CoAt scales + 3-fold reader, class routing'),
    ('.954 title', 'old bundle; only two active branches'),
    ('training reference', 'DINOv2/protocol/geometry; not counted as independent >=.95'),
    ('.924 origin reference', 'requires own soft labels; gold checkpoint selection'),
    ('architecture reference', 'large inference graph; current v39 score not verified'),
]
ledger = []
for i, row in enumerate(snapshots):
    if i < 10:
        claim = str(a[i]['score_claim']) if a[i]['score_claim'] is not None else 'unverified'
        kind, note = a[i]['classification'], a[i]['score_evidence']
    else:
        claim, kind = b[i-10]
        note = kind
    ledger.append(dict(number=i+1, ref=row['ref'], version=row['version'],
        votes_at_collection=row['votes'], source_sha256=row['source_sha256'],
        score_statement=claim, evidence_note=note, classification=kind,
        official_score_verified=False))
assert len(ledger) == 25 and len({r['ref'] for r in ledger}) == 25
(OUT/'review_matrix.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'review_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=ledger[0].keys())
    writer.writeheader(); writer.writerows(ledger)

head = '''# Kaggle RSNA Knee 공개 Code 25개 검토와 우리 개선 계획

조사일: 2026-10-19. **0.95 이상을 보고하거나 그 계보에 속하는 후보 22개 + 학습·설계 원형 3개**, 총 25개 원문을 읽었다. 제목 검색에 그치지 않고 실제 실행 셀, 내장 모듈, 모델 manifest, 현재/과거 버전의 차이를 확인했다.

우리 방법도 [Kaggle 실험일지 v1](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-experiments-and-error-analysis)로 저장해 실행 완료·출력 일치를 확인했다. 현재는 **비공개**다. 공개 요청은 Kaggle의 대회 source 제한으로 거부됐으며, [게시 상태와 검증 기록](kaggle_experiment_journal_publication.md)에 정확한 응답을 남겼다.

**22개 모두의 0.95를 독립 재현한 결과가 아니다.** 아래에는 원저자 점수의 인용, 파생 코드, 최신 버전 미측정, 비공개 추가 가중치, 외부 실행기까지 구분했다. 상세 증거가 부족한 경우 그대로 표시했다. 공개 source는 실행하지 않았고 hidden-test 제출도 하지 않았다.

읽는 순서: 먼저 비교·실험 계획을 읽고, 이어지는 A1–A10 및 11–25번에서 개별 원문과 코드 근거를 확인하면 된다. [검토 목록 CSV](../outputs/research/kaggle_code_20261019/review_matrix.csv), [원문 버전·SHA 목록](../outputs/research/kaggle_code_20261019/snapshots.json)도 보존했다.

## 전체 검토 목록

| 번호 | 원본 | 버전 | 점수 표기 / 증거 범위 |
|---|---|---|---|
'''
for row in ledger:
    head += (f"| {row['number']} | [{row['ref']}](https://www.kaggle.com/code/{row['ref']}) "
             f"| v{row['version']} | {row['score_statement']} — {row['evidence_note']} |\n")
parts = [head]
for name in ['kaggle_095_comparison_and_plan.md','kaggle_095_review_part_a.md','kaggle_095_review_part_b.md']:
    parts.append((ROOT/'docs'/name).read_text(encoding='utf-8'))
text = '\n\n---\n\n'.join(parts)
assert '공개 Code 25개' in text and '???' not in text
(ROOT/'docs/kaggle_095_twenty_notebook_review.md').write_text(text,encoding='utf-8')
print('Assembled 25-source report:', len(text.splitlines()), 'lines; UTF-8 Korean verified.')
