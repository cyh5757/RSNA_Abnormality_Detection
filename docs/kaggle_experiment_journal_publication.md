# 우리 실험일지 Notebook와 게시 상태

작성일: 2026-10-19.

- Kaggle: [RSNA Knee Experiments and Error Analysis](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-experiments-and-error-analysis)
- 저장 버전: **v1, 비공개**, CPU, Internet OFF, 대회 데이터 source 연결.
- 로컬 원본: [rsna_knee_experiment_journal.ipynb](../notebooks/public_journal/rsna_knee_experiment_journal.ipynb)
- 비교 조사: [공개 Code 25개 종합 검토](kaggle_095_twenty_notebook_review.md)

## 내용

2026-10-19 분석 snapshot의 31개 완료 실험을 실험일지 형태로 담았다. 각 실험의 gold 개발 OOF AUC/seed SD/BCE, seed42의 372개 소견별 집계, R01 12개 혼동행렬, S13→R01 FP/FN 변화, TN/FP/FN/TP 및 recall/precision 해설을 포함한다.

우리 `SpatialHead`, `SliceBlock`, `rotary`, `selected_windows`의 실제 코드를 담았고 합성 tensor로 12개 출력과 BCE 역전파를 확인한다. 실제 MRI 학습 결과와 합성 예제를 구분한다. 25개 공개 source 목록·버전, 그 방식과의 차이, 다음 통제 실험도 포함했다.

이 노트는 CPU로 표·그림·모델 예제를 재생성하는 방법론 Code다. GPU 학습이나 hidden test 제출용이 아니며 `submission.csv`는 생성하지 않는다. 환자별 데이터나 원문 보고서, API 인증정보는 포함하지 않는다. 현재 진행 중인 후속 학습의 결과는 이 고정 snapshot에 자동으로 추가되지 않는다.

## 공개가 아직 완료되지 않은 이유

공개 게시를 요청했지만 Kaggle API가 다음 오류를 반환했다.

> Notebooks with competition rsna-knee-abnormality-detection as a source may not be made public until after the competition ends

이는 **이번 API 공개 요청의 실제 거부 응답**이다. 기존 공개 Notebook이 존재하지 않는다는 뜻이나 모든 공개 Code가 금지된다는 일반화는 하지 않는다. [공식 규칙](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)은 public code sharing을 설명하지만, 이번 요청에서는 플랫폼이 공개 저장을 허용하지 않았다. 대회 연결을 유지한 비공개본으로 준비했다. 공개 완료 또는 공개 Code 목록 노출을 주장하지 않는다.

거부 원문은 `outputs/research/kaggle_code_20261019/publication_rejection.json`, 비공개 저장 응답은 `private_publication_receipt.json`에 있다. 대회 종료 후 공개를 전환할 때도 최신 규칙/플랫폼 허용 여부와 노트 내용을 확인해야 한다. 자동 공개 예약은 설정하지 않았다.

## 로컬 재생성

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/build_public_experiment_journal.py
```

명령은 Notebook과 metadata를 생성하며 업로드하지 않는다. metadata는 현재 **비공개**로 생성된다. 새 결과 snapshot으로 바꿀 때는 31개 실험과 R01 기준값을 검사하는 assertion 및 본문의 날짜·수치를 함께 갱신해야 한다. 숫자만 늘려 과거 설명과 다른 내용을 게시하지 않도록 의도적으로 검사한다.

로컬 실행 검증은 `outputs/research/kaggle_code_20261019/journal_validation/validation.json`에 있다. Notebook 코드 7개 셀을 CPU에서 실행했고, 372개 집계의 합계, 모델 gradient, PNG 3개를 확인했다. Kaggle 실행 결과는 아래 기록으로 구분한다.

## Kaggle 실행 검증

**v1 COMPLETE**를 확인했다. CPU/Internet OFF/대회 연결/비공개 설정을 원격 metadata에서 확인했고, 원격 source의 17개 셀 내용이 로컬과 일치했다.

Kaggle 출력에는 PNG 3개, CSV 2개, JSON 1개가 생성됐다. CSV/JSON을 실제로 내려받아 31개 실험 AUC와 372개 소견 집계가 로컬 snapshot과 일치함을 검증했다. 실행 로그에서도 Notebook/HTML 변환 완료를 확인했다.

검증 기록: `outputs/research/kaggle_code_20261019/journal_kaggle_verification.json`. 내려받은 집계 출력: `outputs/research/kaggle_code_20261019/journal_kaggle_v1/`.
