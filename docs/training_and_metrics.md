# RSNA Knee 대회 목적과 학습 안내

**프로젝트 의미와 회고:** [대회의 최종 목적, 얻을 수 있는 경험, 포트폴리오 정리](project_purpose_and_learning.md)

**후속 실험:** [전체 혼동행렬 비교, attention·약라벨·5장/10장 분석, R 자동 학습](followup_experiment_results.md)

**공개 해법 비교:** [0.95 계열 후보와 학습 원형 25개 원문 검토](kaggle_095_twenty_notebook_review.md) · [우리 실험일지 Notebook와 게시 상태](kaggle_experiment_journal_publication.md)

> **2026-10-19 최신 결과:** E/S 30개 실험이 모두 완료됐다. [전체 결과·소견별 혼동행렬·오류 분석](experiment_error_analysis.md)과 [후속 실험 계획](superpowers/plans/2026-10-19-error-driven-experiments.md)을 참조한다. 아래 본문은 원래 baseline 설명을 보존한 것이다.

이 문서의 모델 설명은 원래 baseline을 기준으로 한다. 이후 [3장/5장 비교 결과](experiment_results.md), [단계별 실험 계획](superpowers/plans/2026-10-16-progressive-experiments.md), [ViT·DINO·Swin 등 backbone 조사](backbone_research.md)를 별도로 기록했다.

## 현재 구현한 모델과 학습 설정 요약

| 항목 | 현재 코드에서 사용하는 것 |
| --- | --- |
| **학습 모델** | **ImageNet 사전학습 ResNet18 + `Linear(1536, 12)`**. ResNet18은 고정하고 마지막 선형 분류기만 학습 |
| **학습 데이터** | 무릎 MRI DICOM 영상과 검사 단위의 12개 이진 정답. 완전히 레이블된 58건을 학습 43건·검증 15건으로 분리. 판독문은 현재 모델에서 사용하지 않음 |
| **최종 학습 loss** | **양성 가중치를 적용한 `BCEWithLogitsLoss`**. 클래스별 `pos_weight = 학습 음성 수 / 학습 양성 수` |
| **평가지표** | **Macro ROC AUC**. 12개 소견의 AUC를 같은 비중으로 평균하며 높을수록 좋음 |

여기서 학습 loss, 검증 loss, 대회 점수는 구분해야 한다. 현재 **검증 loss는 가중치 없는 BCE**이며, **저장 모델을 고르는 기준은 검증 loss 최솟값**이다. 대회 점수는 macro ROC AUC다. 아래는 앞으로 개선할 모델이 아니라 현재 구현된 baseline의 설명이다.

## 대회의 목적

이 대회의 목표는 **무릎 MRI 검사에서 12가지 이상 소견이 있을 가능성을 자동으로 예측하는 모델을 만드는 것**이다. 무릎의 인대·반월상연골·뼈 등에 생긴 이상은 영상에서 미세하게 나타날 수 있고, 판독자에 따라 해석이 달라질 수 있다. 주최 측은 영상과 판독문을 함께 활용하는 학습을 통해 의료진의 판독을 더 정확하고 일관되며 빠르게 지원하는 기술을 개발하려 한다. [Kaggle 대회 소개](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

우리가 구현해야 할 기능은 **한 번의 무릎 MRI 검사에 포함된 여러 영상을 읽고, 그 검사에 대한 12개 예측 점수를 출력하는 것**이다. 학습 데이터에는 의료진이 작성한 판독문도 있어 영상과 글의 정보를 함께 활용할 수 있다. 최종 테스트에는 판독문이 제공되지 않으므로, 제출할 때는 영상과 제공된 시리즈 정보로 예측해야 한다. [Kaggle 데이터 설명](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

| 구분 | 이 프로젝트에서 의미하는 것 |
| --- | --- |
| 입력 단위 | MRI 검사 1건인 `StudyInstanceUID`. 검사마다 여러 촬영 시리즈와 영상 슬라이스가 포함됨 |
| 학습에 제공되는 정보 | MRI 영상, 촬영 방향 등의 메타데이터, 판독문, 일부 검사의 12개 정답 |
| 정답 | 소견별 `1`은 있음, `0`은 없음. 빈칸은 정답을 모르는 상태 |
| 모델 출력 | 검사마다 12개 소견에 대한 0~1 사이의 예측 점수 |
| 제출 결과 | `submission.csv`의 한 행에 검사 ID와 12개 점수를 기록 |
| 성능 비교 | 각 소견의 ROC AUC를 구한 뒤 12개를 평균한 점수 |

예를 들어 한 검사에서 `ACL=0.85`, `Effusion=0.70`, `Fracture=0.03`이라는 점수를 출력할 수 있다. 이는 설명용 가상 예측으로, 모델이 인대 손상과 관절액 증가에는 높은 점수를, 골절에는 낮은 점수를 준 상황이다. 실제 제출에는 나머지 소견까지 포함한 12개 점수가 모두 필요하다. 여러 소견이 동시에 존재할 수 있으므로 각각을 예측한다.

### 예측하는 12가지 소견

| 제출 열 이름 | 뜻 |
| --- | --- |
| `ACL` | 전방십자인대 손상 |
| `MCL` | 내측측부인대 손상 |
| `Medial Meniscus` | 내측 반월상연골 파열 |
| `Lateral Meniscus` | 외측 반월상연골 파열 |
| `Medial OA` | 무릎 안쪽 경골대퇴 구획의 골관절염 |
| `Lateral OA` | 무릎 바깥쪽 경골대퇴 구획의 골관절염 |
| `PF OA` | 슬개대퇴 관절의 골관절염 |
| `Effusion` | 관절액 증가 |
| `Synovitis` | 활막염 |
| `Baker's` | 베이커 낭종 |
| `Contusion` | 골좌상 |
| `Fracture` | 골절 |

위 소견과 열 이름은 [공식 데이터 정의](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)에 따른다. 모델 개선의 목표는 처음 보는 MRI 검사에서도 이 소견들을 잘 구별하게 만드는 것이다. 학습 loss와 검증 평가지표는 그 목표에 얼마나 가까워지고 있는지 확인하는 수단이다.

## 현재 프로젝트의 상태

현재 모델은 학습부터 제출 파일 생성까지 확인하기 위한 baseline이다. 로컬에 저장된 결과에서는 **5번째 epoch의 검증 macro ROC AUC가 0.7619**이며, 이후에는 학습 loss만 줄고 검증 성능은 나빠진다. 다음 실험은 **검증 안정화 → 판독문으로 학습 데이터 확대 → MRI 입력 범위 확대 → 모델 미세 조정** 순서가 적절하다.

분석 기준은 2026년 10월 16일 작업 폴더의 [학습 코드](../train_baseline.py), [metrics.json](../outputs/baseline/metrics.json), [history.csv](../outputs/baseline/history.csv)다. 아래 점수는 검증용 MRI 검사 15건에서 계산한 값이며, Kaggle 리더보드 점수는 아니다.

## 현재 모델이 학습하는 것

한 MRI 검사에서 12가지 이상 소견의 점수를 각각 예측한다. 여러 이상이 동시에 존재할 수 있는 **다중 레이블 분류**이므로, 12개 출력의 합이 1일 필요는 없다.

| 구성 | 현재 구현 | 의미와 한계 |
| --- | --- | --- |
| 데이터 | 전체 4,407건 중 12개 정답이 모두 있는 58건 사용 | 나머지 4,349건의 판독문과 영상은 현재 학습에 활용하지 않음 |
| 분할 | 학습 43건, 검증 15건 | 검사 ID로 분리함. 환자 단위 독립성이 확인된 분할은 아님 |
| 영상 선택 | Sagittal·Coronal·Axial에서 각각 시리즈 1개, 각 시리즈의 25·50·75% 위치 슬라이스 | 검사당 최대 9장만 사용하므로 병변이 있는 위치를 놓칠 수 있음 |
| 영상 특징 추출 | ImageNet 사전학습 ResNet18, 가중치 고정 | MRI에 맞춰 영상 인식 부분을 재학습하지 않음 |
| 특징 통합 | 방향별 슬라이스 특징을 평균낸 뒤 연결 | 검사당 1,536개 특징으로 요약 |
| 실제 학습 부분 | `Linear(1536, 12)` | 마지막 분류기 18,444개 파라미터만 학습 |
| 최적화 | AdamW, 학습률 0.001, weight decay 0.1, 100 epochs | 현재는 epoch마다 학습 43건을 한 번에 사용 |

`epoch`는 전체 학습 데이터를 한 차례 사용하는 단위다. `validation`은 가중치를 학습하는 데 쓰지 않은 데이터로 성능을 확인하는 과정이다. 다만 검증 결과로 최적 epoch를 고르므로, 검증 점수에도 선택에 따른 낙관적 편향이 생길 수 있다.

## 학습 데이터의 파일 형태와 텐서 형태

학습 샘플의 단위는 MRI 검사 1건이다. 한 검사 아래에 여러 촬영 시리즈가 있고, 각 시리즈 아래에 2차원 슬라이스 파일이 있다. 12개 정답은 검사 전체에 붙어 있다.

```text
대회 데이터 폴더/
  train.csv
  train_series.csv
  train_series/
    <StudyInstanceUID>/           검사 1건
      <SeriesInstanceUID>/       촬영 시리즈 1개
        <SOPInstanceUID>.dcm     영상 슬라이스 1장
```

| 파일 | 구조와 역할 |
| --- | --- |
| `train.csv` | 4,407행. 검사 ID, 자유 형식 판독문 `Report`, 소견별 0·1 또는 빈칸인 정답 열 12개 |
| `train_series.csv` | 24,371행. 검사 ID, 시리즈 ID, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `train_series/**/*.dcm` | MRI 픽셀과 DICOM 메타데이터. 원본 영상 크기와 슬라이스 수는 시리즈마다 다를 수 있음 |

현재 모델에서는 판독문을 입력하지 않고, 12개 정답이 모두 존재하는 58건의 영상만 처리한다. 나머지 4,349건의 빈 정답은 음성인 0으로 취급하지 않는다. 아래 `N`은 검사 수, `K`는 한 평면에서 선택한 슬라이스 수이며 최대 3이다.

| 처리 단계 | 텐서 형태 | 내용 |
| --- | --- | --- |
| 슬라이스 전처리 | `[K, 3, 224, 224]` | 흑백 영상을 같은 값의 3채널로 복제하고 크기·밝기를 전처리. 채널 3개는 촬영 방향 3개를 뜻하지 않음 |
| ResNet18 출력 | `[K, 512]` | 각 슬라이스에서 512개 특징 추출 |
| 평면별 평균 후 연결 | `[N, 1536]` | Sagittal·Coronal·Axial 각각의 512개 특징을 연결. 없는 평면은 0으로 채움 |
| 정규화 후 분류기 입력 | 학습 `[43, 1536]`, 검증 `[15, 1536]` | 학습 데이터로 계산한 평균·표준편차를 양쪽에 적용 |
| 정답 `y` | 학습 `[43, 12]`, 검증 `[15, 12]` | 소견별 0·1을 `float32`로 저장 |
| 분류기 출력 logit | `[N, 12]` | 아직 확률로 변환하지 않은 실수 점수. 학습 loss에 직접 전달 |
| 최종 예측 확률 | `[N, 12]` | logit에 sigmoid를 적용한 값. AUC 계산과 제출에 사용 |

즉, **MRI 슬라이스 → 고정된 ResNet18 → 검사별 1,536개 특징 → 학습하는 선형 분류기 → 12개 예측값**의 구조다. 여러 소견이 함께 존재할 수 있어 정답 벡터에는 1이 여러 개 들어갈 수 있다.

## Loss는 학습할 때 줄이는 오차

Loss는 **모델의 예측과 정답이 얼마나 어긋나는지 나타내는 벌점**이다. 현재 최종 학습 목적함수는 **양성 가중치를 적용한 이진 교차 엔트로피인 BCEWithLogitsLoss**이며, 이 값을 줄이는 방향으로 선형 분류기를 학습한다.

검사 `n`, 소견 `c`의 정답을 `y[n,c]`, logit을 `z[n,c]`, `p[n,c] = sigmoid(z[n,c])`라고 하면 현재 코드의 loss는 다음과 같다. `log`는 자연로그다.

```text
alpha[c] = 학습 데이터에서 소견 c의 음성 수 / 양성 수

L_train = -(1 / (N × 12)) × Σ_n Σ_c [
    alpha[c] × y[n,c] × log(p[n,c])
    + (1 - y[n,c]) × log(1 - p[n,c])
]
```

현재 학습은 전체 43건을 한 번에 사용하므로 `N=43`이며, 43×12개 항을 평균한다. 구현은 다음과 같다.

```python
positives = targets[train_idx].sum(dim=0)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=(len(train_idx) - positives) / positives
)
loss = criterion(head(X[train_idx]), targets[train_idx])
```

검증 loss는 위 식에서 `alpha[c]=1`로 놓은 가중치 없는 BCE다. 예를 들어 실제로 이상이 있는 사례에서, 가중치 없는 BCE는 `p=0.9`일 때 약 0.105, `p=0.1`일 때 약 2.303이다. 자신 있게 틀린 예측에 더 큰 벌점을 준다. BCE는 낮을수록 좋고 상한은 없으며, 틀린 사례의 비율을 뜻하지 않는다.

AdamW의 `weight_decay=0.1`은 가중치 업데이트에 적용하는 별도 설정이다. 현재 코드의 `loss` 값에는 별도의 정규화 항을 더하지 않는다.

코드의 `BCEWithLogitsLoss`는 sigmoid 변환과 BCE 계산을 안정적으로 묶어 처리한다. 따라서 입력에는 sigmoid를 적용하기 전의 점수인 **logit**을 넣는다. 확률을 출력할 때는 `logits.sigmoid()`를 사용한다. [PyTorch BCEWithLogitsLoss 문서](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)

### 학습 loss와 검증 loss의 계산이 다르다

| 기록 항목 | 계산 방식 | 읽는 방법 |
| --- | --- | --- |
| `train_loss` | `BCEWithLogitsLoss(pos_weight=음성 수 / 양성 수)` | 학습 데이터에서 클래스별 양성 가중치를 적용한 오차 |
| `valid_loss` | `binary_cross_entropy_with_logits(...)` | 검증 데이터에서 가중치 없이 계산한 오차 |

예를 들어 학습 MCL은 양성 4건, 음성 39건이다. `pos_weight=39/4=9.75`여서 MCL 양성 사례의 BCE 항에 9.75배 가중치를 준다. 가중치는 학습 데이터에서만 계산한다.

**두 loss의 절댓값 차이만으로 과적합을 판단하면 안 된다.** 현재는 계산 방식이 다르므로 각 값의 epoch별 추세와 검증 AUC를 함께 봐야 한다. 같은 기준으로 비교하려면 학습 데이터의 가중치 없는 BCE도 별도로 기록하면 된다.

## 대회 평가지표는 macro ROC AUC

대회는 **12개 소견 각각의 ROC AUC를 계산하고 동일한 비중으로 평균**한다. [Kaggle 공식 평가 기준](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)

```text
macro ROC AUC = (ACL의 AUC + MCL의 AUC + ... + Fracture의 AUC) / 12
```

ROC 곡선은 양성으로 판정하는 기준값을 바꿀 때 실제 양성을 찾아내는 비율과 실제 음성을 잘못 양성으로 판단하는 비율의 관계를 보여 준다. AUC는 그 곡선 아래의 면적이며, 직관적으로는 **이상이 있는 검사에 없는 검사보다 높은 점수를 주는 능력**이다.

| AUC | 의미 |
| --- | --- |
| 1.0 | 해당 평가 데이터에서 모든 양성이 모든 음성보다 높은 점수를 받음 |
| 0.5 | 무작위 순위와 같은 수준. 모든 점수가 같아도 0.5 |
| 0.5 미만 | 양성에 더 낮은 점수를 주는 경향 |

예를 들어 양성 두 건의 점수가 0.8·0.6이고 음성 두 건이 0.7·0.2라면, 양성·음성의 네 쌍 중 세 쌍에서 양성 점수가 높아 AUC는 0.75다. 동점은 절반의 성공으로 계산한다.

현재 코드의 `roc_auc_score(..., average="macro")`가 이 평균 방식을 사용한다. 빈도가 낮은 소견도 다른 소견과 똑같이 최종 점수의 1/12을 차지한다. **AUC 0.7619는 정확도 76.19%라는 뜻이 아니다.** [scikit-learn ROC AUC 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

Loss는 확률이 정답과 얼마나 맞는지도 반영하지만, AUC는 순서를 평가한다. 예측 확률의 순위를 유지한 채 값만 바꾸면 AUC는 같아도 BCE는 달라질 수 있다. 따라서 **loss 감소가 항상 AUC 상승을 뜻하지는 않는다.** 제출에는 0.5 기준으로 자른 0·1 대신 연속적인 예측 점수를 사용한다.

## 현재 학습 기록의 해석

| Epoch | 학습 loss ↓ | 검증 loss ↓ | 검증 macro AUC ↑ |
| --- | ---: | ---: | ---: |
| 1 | 0.9846 | 0.6971 | 0.6765 |
| **5** | **0.5487** | **0.6609** | **0.7619** |
| 20 | 0.1622 | 0.7518 | 0.7263 |
| 100 | 0.0364 | 1.0007 | 0.7161 |

5번째 epoch 이후 학습 데이터에 대한 오차는 줄어드는데, 검증 loss는 커지고 AUC도 낮아진다. 이는 학습 데이터에 지나치게 맞춰져 새로운 데이터에 대한 성능이 나빠지는 **과적합 패턴**이다. 이 실행에서는 학습 횟수를 더 늘리는 것이 개선책이 되지 않았다.

현재 코드는 **검증 loss가 가장 작은 epoch**를 저장한다. 이번 실행은 검증 loss 최저와 AUC 최고가 모두 epoch 5여서, 로컬 `baseline.pt`에는 epoch 100이 아니라 epoch 5의 분류기가 저장되어 있다. 조기 종료를 추가하면 불필요한 학습은 줄일 수 있지만, 이미 최적 epoch를 복원하므로 그것만으로 현재 저장 모델의 점수가 높아지지는 않는다.

### 소견별 검증 결과

| 소견 | AUC | 검증 양성 수 | 검증 음성 수 |
| --- | ---: | ---: | ---: |
| ACL | 0.6071 | 8 | 7 |
| MCL | 0.7800 | 5 | 10 |
| Medial Meniscus | 0.7963 | 6 | 9 |
| Lateral Meniscus | 0.8400 | 5 | 10 |
| Medial OA | 0.8333 | 3 | 12 |
| Lateral OA | 0.8611 | 3 | 12 |
| PF OA | 0.5741 | 6 | 9 |
| Effusion | 0.8182 | 11 | 4 |
| Synovitis | 0.8214 | 7 | 8 |
| Baker's | 0.7692 | 2 | 13 |
| Contusion | 0.6200 | 5 | 10 |
| Fracture | 0.8214 | 8 | 7 |

현재 분할에서는 PF OA·ACL·Contusion의 AUC가 낮아 오류 사례를 살펴볼 우선 후보이다. 다만 검증이 15건뿐이고 Baker's 양성은 2건이므로, 이 순서가 다른 데이터에서도 유지된다고 단정할 수 없다.

## 개선할 순서

**현재 실행 순서는 [단계별 실험 계획](superpowers/plans/2026-10-16-progressive-experiments.md)을 따른다.** 사용자 요청에 맞춰 고정 교차 검증 → 3장/5장 → 인코더 → attention → 위치 → 방향 정보 순으로 비교한다. 아래 표는 개선 후보의 배경 설명이다. 실제 실행 결과는 [실험 결과 기록](experiment_results.md)에 남긴다.

아래는 현재 코드와 결과에서 도출한 실험 우선순위다. 점수 향상 여부는 같은 검증 조건으로 비교해야 한다.

| 순서 | 개선할 내용 | 이유와 확인 방법 |
| --- | --- | --- |
| **1** | **검증을 안정화** | 58건을 여러 fold로 나눠 번갈아 검증하는 교차 검증을 구성한다. 클래스별 양성·음성이 각 fold에 있는지 확인하고, 동일 검사의 슬라이스·시리즈가 학습과 검증에 나뉘지 않게 한다. 환자 식별이 가능하면 환자 단위로 묶는다. |
| **2** | **4,349건의 판독문 활용** | 판독문에서 12개 소견의 임시 정답을 만들어 영상 학습에 활용한다. 부정 표현·불확실한 표현·언급 없음과 실제 음성을 구분하고, 확신할 수 없는 레이블은 loss 계산에서 제외한다. 일부 표본을 점검한 뒤 높은 신뢰도의 레이블부터 낮은 가중치로 추가한다. |
| **3** | **MRI에서 보는 범위 확대** | 현재 방향별 3장을 더 많은 슬라이스로 늘리는 실험부터 한다. 이후 여러 시리즈 활용이나 중요한 슬라이스에 가중치를 주는 통합 방법을 비교한다. AUC와 함께 추론 시간·메모리를 기록한다. |
| **4** | **영상 인식 모델 미세 조정** | 학습 데이터가 늘고 검증 방법이 안정되면 ResNet 마지막 블록부터 작은 학습률로 학습한다. 현재 43건으로 전체 모델을 바로 풀면 과적합 위험이 크다. |

교차 검증에서는 특징 정규화의 평균·표준편차와 `pos_weight`를 각 fold의 학습 데이터로 다시 계산해야 한다. 각 검사가 학습에 포함되지 않았을 때의 예측인 OOF 예측을 모으고, fold별 AUC와 소견별 AUC도 함께 확인한다. 이렇게 해도 확정 정답이 58건이라는 한계는 남는다.

판독문으로 만든 정답은 확정 정답과 구분해서 관리한다. 원래 CSV의 빈 정답을 일괄적으로 0으로 바꾸면 안 된다. 또한 검증 검사의 영상과 판독문은 모델 학습에서 제외하고, 검증 정답에 맞춰 임시 정답 생성 규칙을 반복 수정하지 않는다. 테스트에는 판독문이 없으므로 **최종 제출 모델은 MRI와 제공되는 검사 메타데이터만으로 예측할 수 있어야 한다.**

검증을 구성한 뒤에는 대회 지표에 맞춰 **검증 macro AUC가 가장 좋은 모델을 저장하는 방식**을 비교할 수 있다. BCE도 계속 기록해 과도하게 확신하는 예측이 늘어나는지 확인한다. 이번 실행에서는 두 기준이 같은 epoch를 선택하므로, 저장 기준 변경만으로 현재 결과가 달라지지는 않는다.

첫 실험은 **현재 고정 ResNet 특징으로 교차 검증을 구성하고, 방향별 3장과 5장을 비교하는 것**이다. 양성 가중치와 loss는 이 비교에서 고정한다. 기존 `gold_features.pt`의 가공 전 특징을 재사용해 로컬에서 평가 코드를 점검하고, Kaggle에서는 3장과 5장을 모두 다시 추출해 비교한다. 3장 재추출 결과는 기존 캐시와도 대조한다.

## 변경할 수 있는 모델 구조와 대회 제한

2026년 10월 16일 Kaggle MCP로 공식 Rules·Code Requirements·Evaluation·Timeline과 주최 측 공지를 확인했다. **ResNet18, 방향별 슬라이스 3장, 평균 pooling, BCE는 현재 baseline에서 선택한 구현이다. 대회가 지정한 구조가 아니다.** 확인한 규칙에는 특정 신경망 계열이나 슬라이스 수를 강제하는 조건이 없다. 따라서 아래 변경은 데이터·라이선스·실행 조건을 충족하는 범위에서 가능한 설계로 판단한다.

대회가 평가하는 최종 결과는 **검사별 12개 소견의 예측 점수**와 그에 대한 **macro ROC AUC**다. 모델 구조와 학습 loss를 바꿔도 이 출력 형식과 평가 기준은 그대로 유지한다. [공식 평가 기준](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), [공식 규칙](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

### 모델별로 가능한 변경

| 변경 | 적용 방법과 조건 |
| --- | --- |
| ResNet18 → ViT | 슬라이스별 특징 추출기를 교체하고 검사 단위 12개 분류 출력을 연결한다. 입력 전처리·특징 차원·가중치 로딩도 새 모델에 맞춘다. |
| DINO 계열 특징 추출기 | DINOv2처럼 자기지도학습으로 사전학습한 ViT를 특징 추출기로 쓸 수 있다. DINO는 하나의 고정된 아키텍처 이름이 아니므로 정확한 모델과 체크포인트를 지정해야 한다. |
| DETR 또는 객체 검출 DINO | 구조 사용은 가능하지만 기본 목적은 객체의 종류·박스 예측이다. 이 대회에는 검사 단위 정답이 제공되므로 기본 검출 loss를 그대로 적용할 수 없다. 특징 추출기나 query 구조를 활용하고 검사 단위 분류에 맞춰 출력·loss를 설계해야 한다. |
| 방향별 3장 → 5장 이상 | 가능하다. 정해진 장수 제한은 확인되지 않았다. 처리 시간·메모리를 측정하고 학습과 추론의 영상 선택 규칙을 맞춘다. |
| 위치 인코딩과 방향 임베딩 추가 | 가능하다. 평면 종류와 시리즈 내부 위치를 구분해서 모델에 전달하고, 위치와 영상 특징의 관계를 학습하도록 통합 단계를 바꾼다. |
| 평균 pooling·loss 교체 | 가능하다. 최종 출력은 동일한 12개 소견에 대응해야 하며 개선 여부는 같은 검증 조건의 macro AUC로 비교한다. |

ViT는 이미지 패치를 토큰으로 처리하는 영상 모델이다. DINOv2는 자기지도학습으로 얻은 영상 특징을 분류기에 활용할 수 있으며, 공식 기본 모델의 코드·가중치는 Apache 2.0으로 안내되어 있다. 같은 저장소의 파생 모델이나 다른 DINO 버전에도 같은 조건을 가정하지 말고 선택한 가중치의 조건을 확인한다. [ViT 논문](https://arxiv.org/abs/2010.11929), [DINOv2 공식 저장소](https://github.com/facebookresearch/dinov2)

별도로 `DINO: DETR with Improved DeNoising Anchor Boxes`라는 객체 검출 모델도 있다. DETR·검출 DINO를 사용하려면 검사 단위 정답과 맞는 학습 구조를 설계해야 한다는 판단은, 공식 모델의 객체 검출 목적과 이 대회의 검사 단위 레이블을 비교한 것이다. [DETR 공식 저장소](https://github.com/facebookresearch/detr), [검출 DINO 공식 저장소](https://github.com/IDEA-Research/DINO)

### 위치와 순서를 학습하는 설계 예시

다음은 가능한 개선안이며 아직 구현된 모델은 아니다.

```text
MRI 슬라이스를 공간적 위치로 정렬하고 방향별 5장 선택
  → 각 슬라이스를 사전학습 ViT 또는 DINOv2로 인코딩
  → 슬라이스별 특징을 평균내지 않고 유지
  → 각 특징에 시리즈 내부 위치 인코딩과 촬영 방향 임베딩 추가
  → 작은 Transformer로 슬라이스 간 관계 학습
  → 검사 전체 특징으로 통합
  → 12개 logit → sigmoid → 12개 예측 확률
```

ViT가 이미지 내부 패치에 부여하는 위치 정보와, 여러 MRI 슬라이스 사이의 위치 정보는 별개다. 시상·관상·축상 방향을 나타내는 임베딩도 시리즈 안의 앞뒤 위치를 대신하지 못한다. 또한 `mean(feature + position)`처럼 위치를 더하고 즉시 평균내면 `mean(feature) + mean(position)`이 되어 특징과 위치의 대응 관계를 구분할 수 없다. 위치를 활용하는 attention 등의 처리를 먼저 거쳐야 한다. Transformer가 토큰을 입력받는 배열 순서만으로 슬라이스 순서를 자동으로 알게 되는 것도 아니다.

시리즈 내부 위치는 DICOM 위치를 단면 법선에 투영한 좌표나 정규화된 상대 위치 등으로 표현할 수 있다. 서로 다른 검사의 같은 슬라이스 번호가 같은 해부학적 위치를 보장하지는 않는다. 이 부분은 대회의 요구사항이 아니라 모델 설계와 전처리에서 해결할 문제다.

### 제출과 데이터 사용 조건

| 항목 | 확인한 조건 |
| --- | --- |
| 제출 방식 | Kaggle Notebook으로 제출하며 출력 파일 이름은 `submission.csv` |
| 실행 시간 | 제출 Notebook의 CPU·GPU 실행 모두 최대 9시간 |
| 인터넷 | 제출 실행에서는 비활성화. 필요한 모델 가중치·코드·패키지를 입력으로 준비 |
| 외부 자료 | 무료로 공개된 외부 데이터와 사전학습 모델 허용. 규칙의 접근성·비용·라이선스 조건도 충족해야 함 |
| 테스트 입력 | 테스트에는 판독문이 없음. 샘플 3건은 채점 시 실제 테스트 데이터로 교체됨 |
| 제출 횟수 | 하루 최대 5회, 최종 심사용 제출 최대 2개 선택 |
| 팀 | 최대 5명. 팀 밖의 비공개 협업 목적 코드·데이터 공유 금지 |
| 정답 사용 | 평가용 validation/test 자료의 수작업 정답·사람 예측을 제출에 이용하는 행위 금지 |
| 수상 시 | 학습·추론 코드, 가중치, 실행 환경, 방법 설명 등의 제출·공개 의무 |

실행·외부 모델 조건은 [Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements), 테스트 입력은 [Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data), 제출 횟수·공유·수상 의무는 [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)에 따른다. 9시간은 제출 Notebook의 실행 한도다. 지금처럼 미리 학습한 모델을 첨부하고 제출 실행에서는 추론할 수 있다.

외부 데이터에 대해 주최 측은 10월 14일 **단순 가입과 라이선스 동의로 접근 가능한 자료는 접근성 기준을 충족하지만, 기관 연구 계약이나 기관 소속을 요구하는 자료는 충족하지 않는다**고 설명했다. 개별 데이터셋 전체에 대한 허용 목록은 제시하지 않았으므로 특정 MRI 데이터나 그 파생 가중치의 사용 가능성을 이름만으로 단정하지 않는다. [RSNA 주최 측 설명](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3538535)

학습 판독문에서 임시 정답을 만들기 위한 상용 LLM/API 사용은 주최 측이 허용했다. 합리적인 접근성·비용과 서비스 이용 조건을 충족해야 하며, 제출 시 인터넷 차단·판독문 미제공 조건은 그대로 적용된다. [주최 측 LLM 사용 안내](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)

최종 제출 마감은 **2026년 10월 23일 오전 8시 59분 한국시간**이다. 참가·팀 병합 마감은 10월 16일 오전 8시 59분 한국시간이며, 현재 연결된 계정은 참가 동의 완료로 확인됐다. 수상자 자료 제출 마감은 11월 6일 오전 8시 59분 한국시간이다. 일정은 변경될 수 있다. [공식 Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)

일반 순위는 macro AUC로 평가하며, 별도 Efficiency Prize는 성능과 실행 시간을 함께 평가한다. 따라서 큰 모델이나 많은 슬라이스를 사용할수록 전체 테스트 추론 시간을 확인해야 한다. [효율상 평가](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation)

허용 여부와 성능 향상은 별개다. 현재 확정 정답이 58건뿐이므로, 검증 분할을 고정하고 슬라이스 수·특징 추출기·위치 통합 방식을 한 번에 하나씩 바꿔 비교한다. 큰 ViT 전체를 처음부터 학습하는 것보다 사전학습 특징을 활용하는 작은 모델부터 비교하는 것이 현재 데이터 규모에 맞는 실험 출발점이다.

## 개선 모델을 제출에 반영하는 방법

새로운 단계별 비교는 [실험별 학습·제출 안내](experiment_notebooks.md)를 따른다. 각 실험에 독립적인 train/submit Notebook이 있으며 학습 Notebook이 오프라인 모델 묶음까지 만든다. 아래 내용은 기존 baseline의 별도 assets Notebook 방식이다.

현재 [제출 Notebook](../notebooks/submission/rsna_knee_submission.ipynb)은 학습을 하지 않고 저장된 모델을 불러와 `submission.csv`를 생성한다. 모델 개선 작업은 [학습 Notebook](../notebooks/rsna_knee_gpu_baseline.ipynb)에서 진행한다.

학습 결과를 새로 만들어도 제출 모델이 자동으로 바뀌지는 않는다. 현재 [assets 설정](../notebooks/offline_assets/kernel-metadata.json)은 학습 Notebook의 버전 4를, [제출 설정](../notebooks/submission/kernel-metadata.json)은 assets의 버전 1을 참조한다.

1. 학습 Notebook의 새 실행을 완료하고 모델·검증 결과를 확인한다.
2. Assets Notebook이 새 학습 버전을 참조하게 바꾸고 새 assets 버전을 만든다.
3. 제출 Notebook이 새 assets 버전을 참조하게 바꾼다. 입력 처리나 모델 구조가 바뀌었다면 추론 코드에도 동일하게 반영한다.
4. 인터넷을 끈 상태에서 제출 Notebook을 실행하고, 완료된 버전의 출력에 `submission.csv`가 생성됐는지 확인한 뒤 그 버전을 제출한다.

각 실험은 한 가지 변경부터 비교하고 학습 버전·데이터 분할·macro AUC·소견별 AUC·추론 시간을 함께 남기면, 어떤 변경이 도움이 되었는지 판단하기 쉽다.

방향과 slice 위치를 함께 학습하는 후속 구현은 [방향 × 깊이 실험노트](plane_depth_position_experiment.md)를 참조한다. 방향은 범주 embedding, 깊이는 연속 위치 신호로 처리하며, 이미지 내부 patch의 2D 위치와 구별한다. T00/T01/T02의 학습·오프라인 제출 notebook과 기존 큐 완료 후 실행하는 shell을 준비했다.
