# MRI 모델 단계별 비교 실험 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 한 번에 한 요소만 바꿔 어떤 변경이 검사별 12개 소견 예측에 도움이 되는지 확인한다.

**Architecture:** 기존 baseline과 제출 버전은 기준 자료로 보존한다. 별도의 Kaggle GPU 실험에서 입력 특징을 추출하고, 동일한 검사 분할과 분류기 학습 조건으로 비교한다. 이번 구현 범위는 E00 검증 기준 확립과 E01 슬라이스 3장 대 5장 비교다. 이후 단계는 결과를 보고 기준 모델을 선택한 다음 순서대로 진행한다.

**Tech Stack:** 기존 PyTorch, torchvision, pandas, numpy, scikit-learn, pydicom, Kaggle Notebook/Script. 추가 패키지 없음.

**Spec:** 사용자 요청: 슬라이스 수, 인코더, 위치·방향 정보를 점진적으로 바꾸고 성능을 비교한다. 대회 출력·제한은 [학습 및 지표 설명](../../training_and_metrics.md)의 공식 출처를 따른다.

E02의 모델 후보와 선정 근거는 [backbone 조사](../../backbone_research.md)를 참고한다. 이는 후보 조사이며, 새 backbone의 실제 비교 점수는 아직 없다.

## Global Constraints

- 정답이 완비된 58개 검사만 첫 실험에 사용한다. 미기입 정답을 0으로 바꾸지 않는다.
- 분할 단위는 `StudyInstanceUID`. 제공 CSV에 환자 식별자가 없어 환자 단위 독립성까지 보장할 수는 없다.
- 정규화 평균·표준편차와 양성 가중치는 매 fold의 학습 검사에서만 계산한다.
- 슬라이스 수를 바꿀 때 인코더, 전처리, 시리즈 선택, pooling, loss, optimizer, epoch를 함께 바꾸지 않는다.
- 검증 정답을 이용한 epoch 선택을 하지 않는다. 기존 pilot에서 선택됐던 5 epoch를 이번 두 모델에 동일하게 고정한다. 이는 이미 관찰한 데이터에 기반한 개발용 비교이며 새로운 독립 평가가 아니다.
- 원본 `train_baseline.py`, baseline 출력, 현재 작동하는 제출 Notebook을 덮어쓰지 않는다. 공식 제출은 이번 단계에 포함하지 않는다.
- 인증 정보는 코드·문서·출력에 저장하지 않는다. Kaggle 실험은 비공개로 만든다.

## Review Focus

- 특징과 정답을 검사 ID로 연결하는가? CSV/캐시의 행 순서를 가정하지 않는가?
- OOF 예측의 각 검사가 해당 모델의 학습에 포함되지 않았는가?
- 3장 재추출 특징이 기존 캐시와 일치하는가? 차이가 크면 비교를 중단하는가?
- 3장과 5장 비교에서 다른 설정이 동일한가?
- seed 간 표준편차를 새로운 데이터에 대한 신뢰구간처럼 설명하지 않는가?

## 단계와 의사결정

| 단계 | 기준 → 변경 | 고정할 요소 | 확인할 질문 |
| --- | --- | --- | --- |
| E00 | 기존 특징 → 고정 3-fold 평가 | 3장, frozen ResNet18, 평균 pooling | 반복 가능한 기준 점수는 얼마인가? |
| E01 | 방향별 3장 → 5장 | 나머지 E00 조건 전부 | 같은 범위 안에서 더 촘촘하게 보면 도움이 되는가? |
| E02 | ResNet18 → 사전학습 DINOv2 ViT-S/14 후보 | E01까지 선택한 장수, 평균 pooling, 동일 분할·loss | 인코더 교체가 도움이 되는가? 모델별 필수 전처리·특징 차원도 변경되므로 ‘인코더 구성’ 비교로 기록한다. |
| E03 | 평균 pooling → 작은 attention 통합기, 위치 정보 없음 | 선택한 인코더를 frozen으로 유지 | 슬라이스별 중요도를 학습하면 도움이 되는가? |
| E04 | E03 → 슬라이스 위치 인코딩 추가 | attention 크기·학습 조건 고정 | 영상 특징과 시리즈 내부 위치를 함께 알면 도움이 되는가? |
| E05 | E04 → 방향 임베딩 추가 | 위치 인코딩·나머지 조건 고정 | 시상·관상·축상 정보를 명시적으로 제공하면 도움이 되는가? |
| E06 | 선택 구조 → 판독문 기반 약한 정답 활용 | 검증 gold 검사·판독문을 약한 정답 학습에서 제외 | 4,349개 추가 검사의 영상으로 일반화가 나아지는가? 별도 정답 품질 점검 필요. |
| E07 | 선택 구조 → 인코더 일부 미세 조정 | 데이터·검증·구조 고정 | 충분한 학습 데이터에서 MRI에 적응시키면 좋아지는가? |
| E08 | 선택한 최종 모델 → offline 제출 패키지 | 학습/추론 전처리·모델 동일 | 실제 숨겨진 테스트 규모에서 9시간·메모리 조건을 충족하는가? |

E02 이후 세부 학습률·모델 크기·패키지 버전은 해당 단계 직전에 정하고 기록한다. DETR은 박스 정답이 없는 현재 분류 문제에서 우선 실험으로 넣지 않는다. E03~E05는 충분히 작은 모델부터 시작하며, 58건에서 과적합하면 구조를 더 키우지 않는다.

E03에서는 방향별 토큰을 같은 작은 attention 통합기로 처리한 뒤 기존 방향 순서로 concat한다. E04는 그 구조에 상대 위치를 추가한다. E05도 같은 통합 구조에 방향 임베딩만 추가한다. 방향들을 하나의 attention에 합치는 변경은 별도 실험이어야 한다. 위치를 더한 직후 평균내는 방식은 특징과 위치의 대응을 학습하지 못하므로 사용하지 않는다.

## E00 / E01 고정 프로토콜

| 항목 | 설정 |
| --- | --- |
| 검사 정렬·분할 | ID 오름차순, `KFold(3, shuffle=True, random_state=42)`; 각 fold의 모든 소견에 양성·음성이 있는지 검증. 다중 레이블 층화 분할은 아님 |
| fold 크기 | 검증 20 / 19 / 19건, 나머지 학습 |
| 반복 | 분할 고정, 분류기 초기화 seed `42, 2026, 3407` |
| 시리즈 선택 | 방향별 fluid-sensitive, fat-suppression 우선, ID로 동률 해소 |
| 3장 위치 | 정렬된 시리즈의 25%, 50%, 75% |
| 5장 위치 | 25%, 37.5%, 50%, 62.5%, 75% |
| 짧은 시리즈 | 반올림 후 중복 인덱스 제거. 실제 사용 장수 기록 |
| 인코더 | 기존 checkpoint의 frozen ImageNet ResNet18, eval 모드 |
| 영상 전처리 | 기존 DICOM 처리, 1–99 백분위 정규화, 3채널 복제, ImageNet resize/crop/정규화 |
| 통합·출력 | 방향별 평균 512개 → concat 1,536개 → Linear → 12 logits |
| 학습 | fold 학습 통계로 표준화, full-batch, AdamW lr=0.001 / weight_decay=0.1, 5 epoch |
| 학습 loss | `BCEWithLogitsLoss(pos_weight=학습 음성 수 / 학습 양성 수)` |
| 평가 | 주 지표 pooled OOF macro ROC AUC, 보조 fold별·소견별 AUC와 가중치 없는 BCE |

각 seed에서 전체 58개 검사에 OOF 예측을 한 번씩 만든다. seed별 점수와 5장−3장 차이를 모두 기록한다. 평균·표준편차는 초기화 민감도일 뿐 표본 불확실성이나 통계적 유의성을 뜻하지 않는다. 서로 다른 fold 모델의 점수를 합치는 pooled OOF AUC는 fold별 AUC 평균과 다를 수 있어 둘 다 남긴다.

3개 seed 모두 개선되고, 평균 fold AUC도 개선되면 5장을 다음 단계의 **잠정 후보**로 삼는다. 그 외에는 기존 3장을 유지하고 결과를 검토한다. 어느 경우에도 이 규칙이 leaderboard 향상을 보장하지 않는다. 시간이 늘어나는 정도도 함께 본다. cold/warm I/O가 섞인 단일 실행의 추출 시간은 정밀 벤치마크가 아니다.

## 파일과 실행

- `notebooks/experiments/slice_ablation/slice_ablation.py`: 독립 실행 가능한 첫 비교 실험. 기준 전처리를 그대로 옮기고 3장 재추출 결과를 기존 캐시와 대조한다. 기존 학습/제출을 리팩터링하지 않는다.
- `notebooks/experiments/slice_ablation/kernel-metadata.json`: 기존 학습 버전 4와 대회 데이터가 연결된 비공개 GPU 실험.
- `tests/test_slice_ablation.py`: 샘플링, 검사 ID 정합성, 분할 독립성, 학습 통계 누수 방지, OOF 완전성 테스트.
- `outputs/experiments/e00_local/`: 로컬 CPU에서 기존 3장 특징으로 프로토콜 점검.
- `outputs/experiments/e01_slice_ablation/`: Kaggle 결과와 실행 로그. feature cache, fold manifest, 설정·환경, OOF, history, 지표, 비교표.
- `docs/experiment_results.md`: 실제 결과, 해석, 다음 단계 결정.

```powershell
# 로컬: MRI를 재다운로드하지 않고 기존 특징으로 평가 코드 점검
.\.venv\Scripts\python.exe notebooks/experiments/slice_ablation/slice_ablation.py --cached-only --data data/raw --source outputs/baseline --out outputs/experiments/e00_local

# Kaggle: MRI 특징을 3장/5장으로 각각 추출하고 동일 조건에서 비교
.\.venv\Scripts\kaggle.exe kernels push -p notebooks/experiments/slice_ablation
.\.venv\Scripts\kaggle.exe kernels status choiyoungwhan/rsna-knee-slice-ablation
.\.venv\Scripts\kaggle.exe kernels output choiyoungwhan/rsna-knee-slice-ablation -p outputs/experiments/e01_slice_ablation
```

출력 경로가 이미 결과를 포함하면 실행을 중단한다. 재실험은 새 경로 또는 새 Kaggle 버전으로 수행한다.

## 구현 체크리스트

- [x] 현재 코드·출력·데이터·Kaggle 기준 버전 확인
- [x] 단계별 실험, 고정 변수, 비교 기준 문서화
- [x] 새 평가 로직의 작은 실패 테스트 작성·확인
- [x] 첫 비교 스크립트와 Kaggle 설정 구현, 테스트 통과
- [x] 기존 캐시로 로컬 CPU 실행 확인
- [x] Kaggle GPU 3장/5장 실행 완료, 출력 회수
- [x] 독립 리뷰와 관련 전체 테스트 확인
- [x] 실제 비교 결과와 다음 단계 선택 기록

## 진행 기록 / 판단

- 2026-10-16: 작업 폴더에 `.git`이 없어 worktree·commit 단계는 적용하지 않는다. 기존 파일은 보존하고 새 실험 경로를 사용한다.
- 기존 문서의 양성 가중치 유무 비교보다 사용자가 요청한 슬라이스·구조 개선을 먼저 진행한다. loss는 이번 비교에서 고정한다.
- 이번 단계에서는 전체 구조를 한꺼번에 구현하거나 최종 제출 모델을 교체하지 않는다. E00/E01 결과를 다음 단계의 출발점으로 사용한다.
- 로컬 pandas 3의 `stack()`은 결측값을 유지하므로 관측 레이블 검사에 `dropna()`를 명시했다. 결측 정답을 음성으로 바꾸지 않으며 gold 58건은 동일하다.
- 로컬 캐시 평가 완료: pooled OOF macro AUC 평균 0.590659, seed 표준편차 0.006534. 기존 15건 분할의 0.761852와 직접 비교할 수 없다.
- Kaggle 비공개 GPU 실험 `choiyoungwhan/rsna-knee-slice-ablation` 버전 1 실행 시작.
- 독립 구현 리뷰에서 필수 수정 사항 없음. 기존 제출 검증을 포함한 unittest 6개와 Python 구문 검사 통과. 실제 GPU 재추출 동등성은 실행 완료 결과로 별도 확인한다.
- Kaggle 버전 1 완료·출력 회수. 기존 3장 특징과의 최대 오차 0.0. 18개 저장 분류기의 예측을 재현하고 OOF 지표를 재계산해 일치 확인.
- E01 결과: OOF macro AUC 평균 3장 0.590659 → 5장 0.606793, 차이 +0.016134. 세 seed 모두 개선됐고 평균 fold AUC도 개선됐다. 사전 기준에 따라 E02의 잠정 기준을 5장으로 선택한다. [상세 결과](../../experiment_results.md).
- E00/E01 완료. 이후 단계는 아직 구현·실행하지 않았다. 검증 58건의 불확실성을 유지하고 공식 제출 버전은 변경하지 않는다.
