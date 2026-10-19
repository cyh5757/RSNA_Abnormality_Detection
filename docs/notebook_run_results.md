# 실험별 Kaggle 학습·제출 실행 결과

> **2026-10-19 전체 실험 결과:** 아래는 초기 backbone/E03~E05 기록이다. 이후 E06~E12·S00~S13까지 포함한 [30개 실험 결과와 소견별 오류 분석](experiment_error_analysis.md)을 별도로 작성했다.

[사용 방법과 전체 Notebook 목록](experiment_notebooks.md)

Backbone 비교 6개를 순서대로 Kaggle GPU에서 학습했다. 후속 E03~E05의 기준은 **Swin-T**로 선택했다. 이번 비교는 사전학습 backbone을 고정하고 12개 출력을 내는 분류기를 학습한 결과다.

## 실제 실행 버전

아래 링크는 이 계정의 비공개 Notebook이다. 학습과 제출 모두 GPU를 사용하며 제출 Notebook의 Internet OFF 설정을 확인했다. 완료된 제출 버전은 공개 테스트 3건의 `submission.csv`를 생성한다. 공식 리더보드 제출은 실행하지 않았다.

| 실험 | 학습 버전 | 오프라인 제출 버전 |
| --- | --- | --- |
| E00 ResNet18 3장 | [v2 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e00-resnet18-3-train) | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e00-resnet18-3-submit) |
| E01 ResNet18 5장 | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e01-resnet18-5-train) | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e01-resnet18-5-submit) |
| E02a DINOv2 ViT-S/14 | [v2 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02a-dinov2-5-train) | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02a-dinov2-5-submit) |
| E02b Swin-T | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02b-swin-5-train) | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02b-swin-5-submit) |
| E02c ConvNeXt-Tiny | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02c-convnext-5-train) | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02c-convnext-5-submit) |
| E02d ViT-B/16 | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02d-vit-5-train) | [v1 COMPLETE](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-e02d-vit-5-submit) |

Kaggle에서 제출할 때는 **`...-submit`의 완료된 버전**을 선택한다. 현재 비교에서 선택한 것은 E02b Swin-T다. 새로 재학습했다면 `kernel-metadata.json`의 `kernel_sources`에 새로 완료된 학습 버전을 고정한 뒤 제출 Notebook도 새로 실행해야 한다.

## 동일 조건 비교

확정 정답 58건, 동일 Study ID 정렬과 3-fold, seed 42/2026/3407, 5 epochs, train fold에서만 계산한 정규화·양성 가중치, weighted BCE, AdamW lr 0.001 / weight decay 0.1을 사용했다. 검증 후 제출용 분류기는 전체 58건으로 seed 42 재학습했다.

OOF AUC는 각 seed의 전체 58건 OOF 예측으로 계산한 macro AUC의 평균이다. fold AUC는 9개의 fold별 macro AUC 평균이다. 두 값은 계산 방식이 다르다. 표의 ±는 초기화 seed에 따른 표준편차이며 데이터 표본의 불확실성을 나타내는 신뢰구간이 아니다. BCE는 검증 시 가중치 없이 계산하며 낮을수록 좋다.

| 실험 | OOF macro AUC 평균 ± seed SD | fold AUC 평균 | 검증 BCE | 특징 추출 초 | 최대 CUDA 할당 MiB |
| --- | --- | --- | --- | --- | --- |
| E00 ResNet18 3장 | 0.590659 ± 0.006534 | 0.598796 | 0.736453 | 64.4 | 71.8 |
| E01 ResNet18 5장 | 0.606793 ± 0.006086 | 0.620142 | 0.734936 | 83.4 | 77.6 |
| E02a DINOv2 | 0.562762 ± 0.017212 | 0.569251 | 0.727786 | 66.9 | 113.7 |
| **E02b Swin-T** | **0.619577 ± 0.003212** | **0.625725** | 0.730335 | 62.3 | 173.4 |
| E02c ConvNeXt-Tiny | 0.597450 ± 0.004089 | 0.613418 | 0.747566 | 63.9 | 171.1 |
| E02d ViT-B/16 | 0.593018 ± 0.002950 | 0.604096 | 0.756686 | 100.9 | 366.4 |

시간과 메모리는 Tesla T4에서 58건을 한 번 추출한 관측값이다. 3장은 522개, 5장은 870개 이미지다. 준비·다운로드·모델 내보내기·head 학습 시간은 포함하지 않으며 반복 측정한 속도 벤치마크로 해석하지 않는다. backbone별 권장 resize/interpolation과 특징 차원도 달라진다.

Swin-T는 ResNet18 5장 대비 평균 OOF AUC가 **+0.012783** 높았다. 대회 지표인 AUC를 선택 기준으로 삼았다. DINOv2의 BCE가 더 낮아도 AUC 순위가 같지는 않다. 초기 baseline의 0.7619는 43/15 단일 분할 점수이므로 이 표의 OOF 점수와 평가 조건이 다르다. E00/E01은 [이전 슬라이스 비교](experiment_results.md)의 AUC를 재현했다.

## 소견별 비교

각 값은 3개 seed의 OOF AUC 평균이다. Swin-T가 모든 소견에서 개선된 것은 아니다.

| 소견 | ResNet18 5장 | Swin-T 5장 | 차이 |
| --- | --- | --- | --- |
| ACL | 0.5776 | 0.7108 | +0.1332 |
| MCL | 0.5314 | 0.4452 | -0.0862 |
| Medial Meniscus | 0.5753 | 0.5505 | -0.0248 |
| Lateral Meniscus | 0.6356 | 0.7118 | +0.0762 |
| Medial OA | 0.6718 | 0.7235 | +0.0517 |
| Lateral OA | 0.6557 | 0.6680 | +0.0123 |
| PF OA | 0.6186 | 0.6671 | +0.0485 |
| Effusion | 0.6422 | 0.8377 | +0.1954 |
| Synovitis | 0.5854 | 0.5890 | +0.0036 |
| Baker's | 0.4686 | 0.3448 | -0.1238 |
| Contusion | 0.6914 | 0.6820 | -0.0094 |
| Fracture | 0.6278 | 0.5046 | -0.1231 |

소견마다 양성 표본이 적고 후보 선택에도 같은 58건을 사용했다. 현재 결과는 다음 실험의 기준을 정하기 위한 개발 검증이다. 모델 선택의 효과를 확정하려면 추가 독립 검증이 필요하다.

## E03~E05 실행 결과

2026-10-16 16:06~16:19 KST에 shell 자동 큐로 세 실험의 학습 및 오프라인 제출 파일 생성을 완료했다. 모두 Swin-T, 방향별 5장, 58 gold, 3-fold·3-seed, 5 epoch다. 학습·제출 버전은 모두 v1이며 Kaggle에서 6개 세션의 COMPLETE 상태를 재확인했다.

1. **E03**: 평균 통합 → 방향별 작은 attention. [학습 코드](../notebooks/experiments/e03_attention/train/train.ipynb)
2. **E04**: E03 + 슬라이스 상대 위치. [학습 코드](../notebooks/experiments/e04_position/train/train.ipynb)
3. **E05**: E04 + 방향 임베딩. [학습 코드](../notebooks/experiments/e05_plane/train/train.ipynb)

| 실험 | OOF AUC 평균 ± seed SD | 검증 BCE |
| --- | --- | --- |
| E02b 평균 pooling | **0.619577 ± 0.003212** | 0.730335 |
| E03 attention | 0.583633 ± 0.030282 | 0.662931 |
| E04 + 위치 | 0.575507 ± 0.029291 | 0.659928 |
| E05 + 방향 | 0.575334 ± 0.029756 | 0.659805 |

fold 파일의 일치와 저장 OOF로 AUC를 재계산한 결과를 확인했다. 이 조건에서는 E02b의 AUC가 높다. backbone은 동결했고 head 하나당 full-batch 5번만 업데이트하므로, 각 구조의 충분히 학습된 성능 비교로 해석하지 않는다. 결과는 [자동 실행 비교표](../outputs/automation/next-experiments/summary.csv)에 있다.

다음 단계는 [보고서 약라벨 자동 실험 E06~E12](weak_label_experiments.md)다. E06에 새 학습 조건의 gold 대조군을 두고, 나머지 단계는 데이터·입력·pooling·backbone 학습을 분리해 비교한다.

## 검증과 결과 위치

- 전체 로컬 테스트 14개 통과: mask·순서/위치 대응·train-only 통계·다른 실험/손상된 bundle 거부·DICOM→export/reload→CSV 등.
- 18개 Notebook은 로컬 프로젝트 import가 없는 독립 코드이며, 각자 하나의 고정 실험만 포함한다.
- 6개 학습의 fold 파일이 바이트 단위로 같고, 저장된 OOF 예측에서 AUC를 별도로 재계산해 보고 점수와 일치함을 확인했다.
- 6개 오프라인 제출 모두 COMPLETE. 각 CSV의 테스트 ID 전체 일치, 12개 열 순서, 유한한 0~1 확률, 학습·제출의 실험 설정과 encoder/head 해시 일치를 검사했다. 실제 Kaggle 메타데이터에서도 GPU·비공개·Internet OFF를 확인했다.
- 학습 중 원 encoder와 내보낸 encoder 출력, 저장 후 재로딩을 확인했다. DINOv2 v1의 float32 비교 허용 오차 문제를 수정한 v2는 batch 1~5 최대 절대 오차 2.82e-5 이하로 통과했다. 학습 특징과 제출 특징은 모두 동일한 내보낸 encoder로 계산한다.
- 원시 결과: `outputs/experiments/<실험 ID>/train/run/`의 `metrics.json`, `folds.csv`, `oof_predictions.csv`, `history.csv`, `timings.json`, `bundle/assets.json`.
- 비교 CSV: [backbone_comparison.csv](../outputs/experiments/backbone_comparison.csv). 공개 테스트 출력: 각 `outputs/experiments/<실험 ID>/submit/submission.csv`와 `inference_summary.json`.

검증 성공은 공개 테스트의 파일 생성까지다. 실제 대회 채점·숨겨진 전체 테스트의 9시간 한도는 아직 확인하지 않았다.
