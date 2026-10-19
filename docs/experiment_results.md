# 단계별 실험 결과

[실험 계획과 고정 조건](superpowers/plans/2026-10-16-progressive-experiments.md) · [학습/지표 설명](training_and_metrics.md)

## E00: 평가 기준 확립

기존 58개 검사의 3장 특징으로 로컬 CPU 평가를 완료했다. 특징을 검사 ID에 맞춰 재정렬하고, 고정 3-fold와 초기화 seed 42·2026·3407을 사용했다. 각 fold는 학습 검사만으로 정규화·양성 가중치를 계산하고 분류기를 5 epoch 학습한다.

| 지표 | 로컬 3장 캐시 결과 |
| --- | ---: |
| pooled OOF macro AUC 평균 | 0.590659 |
| 초기화 seed 간 AUC 표준편차 | 0.006534 |
| fold별 macro AUC 평균 | 0.598796 |
| OOF BCE 평균 | 0.736453 |

이전 baseline의 0.761852는 별도 15건 검증 분할에서 epoch를 선택한 결과다. 분할·학습 인원·초기화·epoch 선택 방식이 바뀌었으므로 새 점수와 직접 비교해 성능이 떨어졌다고 해석하지 않는다. 이번 3장/5장 비교는 동일한 새 프로토콜끼리만 비교한다.

로컬 결과: `outputs/experiments/e00_local/`의 `config.json`, `folds.csv`, `comparison.csv`, `slices_3/metrics.json`, `slices_3/oof_predictions.csv`. seed 간 표준편차는 데이터 표본 불확실성을 나타내는 신뢰구간이 아니다.

## E01: 방향별 3장 대 5장

[Kaggle 실험 버전 1](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-slice-ablation?scriptVersionId=358895124)이 **완료**됐다. GPU Tesla T4에서 특징을 추출하고, 작은 분류기는 두 조건 모두 CPU에서 학습했다. Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 환경이다. 전체 패키지 버전·입력/코드 해시는 출력의 `config.json`에 저장했다.

두 입력 모두 시리즈의 25–75% 구간을 사용한다. 5장에는 37.5%·62.5% 지점을 더하므로 영상 범위 확대가 아니라 같은 범위의 샘플링 밀도 증가를 확인한다. 기존 ResNet18 가중치, 평균 pooling, 학습 loss와 학습 조건은 같다. 아직 위치·순서 정보를 학습하는 모델은 아니다.

3장 특징과 기존 캐시의 최대 절대 오차는 **0.0**이었다. 로컬과 Kaggle의 검사 분할이 같고, 3장 조건의 AUC도 일치했다. 58개 검사 모두 세 방향이 있어 실제 처리 영상은 3장 조건 522장, 5장 조건 870장이었다.

### 전체 결과

| 지표 | 방향별 3장 | 방향별 5장 | 5장−3장 |
| --- | ---: | ---: | ---: |
| pooled OOF macro AUC 평균 ↑ | 0.590659 | **0.606793** | **+0.016134** |
| 초기화 seed 간 AUC 표준편차 | 0.006534 | 0.006086 | — |
| fold별 macro AUC 평균 ↑ | 0.598796 | **0.620142** | +0.021345 |
| OOF BCE 평균 ↓ | 0.736453 | **0.734936** | −0.001517 |

평균 AUC는 0–1 척도에서 약 0.0161 상승했다. 세 seed에서 모두 AUC가 상승했고, seed를 평균한 세 fold 각각의 AUC도 상승했다. BCE는 평균적으로 소폭 개선됐지만 seed 3407에서는 나빠졌으므로 모든 예측 성질이 일관되게 개선됐다고 할 수는 없다.

| 초기화 seed | 3장 OOF AUC | 5장 OOF AUC | 차이 |
| --- | ---: | ---: | ---: |
| 42 | 0.583330 | 0.598211 | +0.014881 |
| 2026 | 0.589449 | 0.611644 | +0.022196 |
| 3407 | 0.599197 | 0.610524 | +0.011327 |

| 검증 fold | 3장 AUC: seed 평균 | 5장 AUC: seed 평균 |
| --- | ---: | ---: |
| 0: 20건 | 0.508754 | 0.552668 |
| 1: 19건 | 0.671504 | 0.684466 |
| 2: 19건 | 0.616131 | 0.623291 |

fold에 따른 차이가 seed에 따른 차이보다 크다. 현재 데이터에서 방향성은 긍정적이지만, 이 숫자는 독립 테스트 점수나 leaderboard 향상 보장이 아니다.

### 소견별 변화

아래는 각 seed의 pooled OOF AUC를 평균한 값이다.

| 소견 | 3장 | 5장 | 차이 |
| --- | ---: | ---: | ---: |
| ACL | 0.551879 | 0.577614 | +0.025735 |
| MCL | 0.581255 | 0.531368 | **−0.049887** |
| Medial Meniscus | 0.622997 | 0.575321 | **−0.047676** |
| Lateral Meniscus | 0.628986 | 0.635611 | +0.006625 |
| Medial OA | 0.670284 | 0.671835 | +0.001550 |
| Lateral OA | 0.630561 | 0.655706 | +0.025145 |
| PF OA | 0.562849 | 0.618619 | +0.055770 |
| Effusion | 0.564389 | 0.642236 | +0.077847 |
| Synovitis | 0.540422 | 0.585424 | +0.045002 |
| Baker's | 0.475845 | 0.468599 | −0.007246 |
| Contusion | 0.663068 | 0.691408 | +0.028340 |
| Fracture | 0.595370 | 0.627778 | +0.032407 |

12개 중 9개 소견의 평균 AUC가 높아졌고 3개는 낮아졌다. MCL·Medial Meniscus·Baker's는 다음 모델에서도 별도로 확인할 항목이다. 소수 표본에서 관찰한 결과만으로 특정 병변에 5장이 나쁘다고 일반화하지 않는다.

### 시간·메모리 해석

| 측정값 | 3장 | 5장 |
| --- | ---: | ---: |
| 58개 검사 특징 추출 | 61.61초 | 21.94초 |
| PyTorch 최대 GPU allocated 메모리 | 72.12 MiB | 76.18 MiB |

**5장이 더 빠른 모델이라는 뜻은 아니다.** 같은 실행에서 3장을 먼저 처리했으므로 후속 5장 실행은 파일 캐시와 CUDA 준비 상태의 영향을 받는다. 추출 시간은 실행 기록으로만 남긴다. 전체 추론 속도 비교는 warm-up, 실행 순서 교차, 반복 측정을 적용해 별도로 해야 한다. allocated 메모리도 CUDA context·reserved 메모리·CPU 메모리를 포함한 전체 사용량은 아니다. 이번 값으로 숨겨진 전체 테스트의 9시간 준수 여부를 확정하지 않는다.

### 검증과 산출물

- 관련 unittest 6개 및 Python 구문 검사 통과. 독립 구현 리뷰에서 필수 수정 사항 없음.
- 저장된 OOF 예측에서 AUC를 다시 계산해 지표 파일과 일치함을 확인했다.
- 두 조건 × 3 seeds × 3 folds의 **18개 분류기**를 다시 불러와 저장된 예측을 재현했다. 최대 절대 오차 약 `3.83e-7`.
- 저장된 학습/검증 ID가 분리됐고, 각 분류기의 정규화 통계가 해당 학습 특징에서 계산됐음을 확인했다.
- 기존 baseline과 제출 Notebook은 그대로 유지했다.

Kaggle 결과는 `outputs/experiments/e01_slice_ablation/slice_ablation/`에 있다.

| 파일 | 용도 |
| --- | --- |
| [comparison.csv](../outputs/experiments/e01_slice_ablation/slice_ablation/comparison.csv) | 두 조건의 평균 지표 |
| [paired_deltas.csv](../outputs/experiments/e01_slice_ablation/slice_ablation/paired_deltas.csv) | 동일 seed끼리의 점수 차이 |
| [per_label_comparison.csv](../outputs/experiments/e01_slice_ablation/slice_ablation/per_label_comparison.csv) | 소견별 비교 |
| [config.json](../outputs/experiments/e01_slice_ablation/slice_ablation/config.json) | 실행 조건·환경·파일 해시 |
| [verification.json](../outputs/experiments/e01_slice_ablation/slice_ablation/verification.json) | 결과 회수 후 재현 확인 |
| `folds.csv`, `slice_audit_*.csv` | 검사 분할, 실제 선택 시리즈·슬라이스 |
| `features_3.pt`, `features_5.pt`, `encoder.pt` | 재사용 가능한 영상 특징과 인코더 |
| `slices_3/`, `slices_5/` | 조건별 metrics, history, OOF 예측, fold 분류기 |

이 fold 분류기들은 비교·재현용이다. 현재 제출 코드에 그대로 연결할 최종 모델은 아니다.

## 다음 단계

사전에 정한 기준인 세 seed의 OOF AUC 개선과 평균 fold AUC 개선을 충족했으므로 **방향별 5장을 다음 단계의 잠정 기준으로 선택**한다. 선택 근거는 현재 개발용 교차 검증으로 한정한다.

다음 E02는 **5장 + 평균 pooling + 동일 검증/학습 조건**을 유지하면서 frozen ResNet18과 frozen DINOv2 ViT-S/14 후보를 비교한다. 인코더에 필수적인 전처리·특징 차원도 변경되므로 전체 인코더 구성의 효과로 해석한다. 이번에 얻은 5장 ResNet18 결과가 그 기준점이다. attention·위치·방향 임베딩은 각각 후속 단계로 분리한다.

E00/E01은 완료됐으며 E02 이후 구현과 공식 제출은 아직 실행하지 않았다.
