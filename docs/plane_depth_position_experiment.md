# 방향 × slice 위치: 두 요소를 함께 학습하는 실험노트

작성: 2026-10-19. **구현·로컬 검증 완료, Kaggle 성능 결과는 아직 없음.** 사용자가 선택한 실행 순서는 기존 follow-up 큐 완료 → T00 → T01 → T02이다.

## 1. 제안의 의미와 기대 효과

제안은 타당하다. 한 장의 MRI slice에 **“어느 방향에서 촬영했는지”와 “그 방향의 series 안에서 어디에 있는지”**를 함께 알려주자는 것이다. 예를 들어 `(Sagittal, 30 mm)`와 `(Coronal, 30 mm)`는 서로 다른 정보를 가진 영상이다. 깊이 숫자가 같다고 같은 신체 위치의 영상이라고 간주하면 안 된다.

다만 여기서의 ‘2차원’은 가로·세로가 있는 일반적인 2D 이미지 좌표와 구별한다.

| 정보 | 성격 | 이번 구현 |
|---|---|---|
| 촬영 방향: Sagittal / Coronal / Axial | 범주. 방향 간 0→1→2라는 거리나 앞뒤 순서는 없음 | 학습 가능한 방향 embedding |
| 해당 series 안의 slice 위치 | 순서와 간격이 있는 연속 좌표 | 첫 slice 기준 mm 거리 / 3; 없으면 정렬 index |
| 한 slice 내부의 patch 위치 `(x, y)` | 이미지 안의 공간 좌표 | 이번 변경 대상 아님. 현재 저장된 feature는 slice마다 전역 벡터 |
| 환자 좌표계의 `(x, y, z)` 및 방향 벡터 | 실제 3D geometry | 원본 DICOM에는 정보가 있지만, 현재 cache는 이를 전부 보관하지 않음 |

따라서 이번에는 **방향 범주 × 깊이 좌표의 두 요소 positional encoding**으로 구현한다. 방향 번호를 연속 숫자로 놓고 ‘Coronal은 Sagittal과 Axial 사이’라는 거리를 강제로 부여하지 않는다.

위치 정보가 이미지 특징을 새로 추출하는 것은 아니다. Swin이 이미 각 slice의 영상 특징을 뽑고, 위치 encoding은 그 특징이 **어느 방향의 어느 깊이에서 관측됐는지 해석하는 데 도움**을 준다. 개선 가능성이 있지만 점수 상승은 실험으로 확인해야 한다.

## 2. 기존에도 무엇이 들어 있었나

기준 실험 S11에는 이미 다음이 있다.

1. DICOM의 방향·위치로 slice를 정렬한다.
2. 방향 3개 × 지방억제 유무 2개 = 최대 6개 slot을 만든다.
3. 각 slice의 Swin 특징 768차원을 128차원으로 투영한다.
4. 방향 embedding과 촬영 조건 embedding을 더한다.
5. 같은 slot 안에서 mm 기반 RoPE를 사용한 self-attention을 수행한다.
6. 소견별 12개 query가 전체 slot의 slice token에 cross-attention한다.
7. 소견마다 logit 하나씩 출력하고 sigmoid로 확률을 만든다.

즉 S11은 방향·순서를 전혀 모르는 모델이 아니며, 최종 결합도 단순 concat이 아니다. 다만 기존 RoPE의 상대 위치 효과는 주로 **같은 slot 안의 Q/K attention**에 들어간다. 이번에는 절대 깊이 신호를 token에도 추가하여 여러 방향을 읽는 class query가 직접 활용할 수 있게 한다.

기존 S11의 Gold58 OOF macro AUC 평균은 **0.736386**이다. 이는 이번 실험의 비교 기준이며, 공개 리더보드 0.95와 동일한 평가 집합의 점수가 아니다. 현재 더 높은 ensemble 점수가 있어도, 결합 방식의 효과를 분리하려고 S11 단일 구조를 기준으로 삼았다.

## 3. T00 / T01 / T02의 정확한 차이

`f`는 slice 특징, `p`는 방향 범주, `r`은 깊이 좌표이다. `S(r)`은 sin/cos 깊이 encoding, `E(p)`는 학습 가능한 방향 embedding이다. `Eprotocol`과 `Esource`는 각각 촬영 조건과 물리 좌표/대체 index 여부를 나타낸다.

```text
공통 기본 token = Linear(Normalize(f)) + E(p) + Eprotocol + Esource

T00: 기본 token
T01: 기본 token + S(r)
T02: 기본 token + S(r) + tanh(E(p)) ⊙ S(r)

세 실험 공통 후속 과정:
slot별 mm RoPE self-attention
→ 전체 방향의 token에 대한 12개 소견 query cross-attention
→ LayerNorm / residual FFN
→ 12개 logit → sigmoid 확률
```

`⊙`는 같은 차원의 원소끼리 곱하는 연산이다. T02에서는 방향별로 학습되는 값이 깊이 신호를 조절한다. 따라서 같은 `r`이라도 방향에 따라 다른 encoding을 줄 수 있다. `tanh`로 조절값을 -1~1에 제한한다. 위치 신호의 크기·가중 방식이 최적이라는 주장은 아니며, 우선 고정된 한 가지 가설을 검증한다.

| 실험 ID | 질문 | 직접 비교 |
|---|---|---|
| `t00_position_control` | 새 독립 notebook이 기존 S11을 재현하는가? | S11 vs T00 |
| `t01_position_sum` | 상대 위치 RoPE에 절대 깊이 신호를 추가하면 유리한가? | T00 vs T01 |
| `t02_position_joint` | 방향에 따라 깊이 신호를 다르게 반영하면 더 유리한가? | T01 vs T02 |

**파라미터는 새로 추가하지 않는다.** T02도 기존 방향 embedding을 사용한다. 세 실험의 파라미터 수와 초기화 순서가 같고, 데이터·loss·seed·fold·epoch·backbone을 고정한다. T00의 출력과 gradient가 기존 S11과 정확히 일치하는 로컬 테스트를 포함했다. 실제 GPU 재학습 결과에는 실행 환경의 수치 차이가 있을 수 있으므로 T00 OOF도 확인한다.

실제 생성된 세 head는 각각 **385,880 파라미터**이다.

여기서 ‘5장’은 검사 전체 5장이 아니다. 기본 설정은 **각 series에서 중심 5곳 × 중심 주변 이웃 5장**, 중복 index를 제거해서 최대 25개 token/slot이다. 최대 6 slot이므로 padding 포함 최대 150개 token/검사이다. 이 선택 방식도 세 실험에서 유지한다.

## 4. 2D RoPE를 바로 쓰지 않은 이유

이번 구현은 **방향 조건부 sin/cos encoding + 기존 1D 깊이 RoPE**이다. 새로운 2D RoPE를 구현했다고 부르지 않는다.

RoPE는 Q/K를 회전시켜 상대 위치를 attention에 반영한다. 현재 slot별 self-attention에서는 모든 token의 방향이 같다. 여기에 같은 방향 각도로 Q/K를 한 번 더 회전하면, 같은 회전의 영향은 내적에서 상쇄될 수 있다. 따라서 ‘방향 축 회전을 추가했다’는 사실만으로 방향 관계를 학습하게 되지는 않는다. [RoFormer 원 논문](https://arxiv.org/abs/2104.09864)

여러 방향의 token을 함께 self-attention하거나 실제 3D 좌표를 연결하는 실험은 별도로 가능하다. 하지만 이번에는 attention 구조까지 동시에 바꾸지 않고 방향×깊이 결합 자체를 비교한다. 현재 class query에는 특정 공간 좌표가 없으므로 임의의 좌표를 붙여 공간 RoPE를 적용하지 않는다.

## 5. 위치 정보의 한계

- 현재 `r`은 series의 첫 slice를 원점으로 한 깊이이다. 서로 다른 환자의 `30 mm`가 같은 해부학적 구조임을 보장하지 않는다. 촬영 범위, 무릎 크기와 자세가 달라질 수 있다.
- series별 정렬은 물리 좌표를 활용하지만, 방향별 FOV와 절단면의 완전한 3D 정합을 수행한 것은 아니다.
- 물리 정보가 없으면 정렬 index로 대체하고 좌표 출처 embedding으로 구별한다. S11 저장 cache에서는 물리 좌표 token 409,239개, index 대체 0개였다.
- backbone 출력은 slice마다 768차원의 전역 특징이다. 여기에 위치를 더해도 이미 pooling된 이미지 내부의 세부 patch 위치를 복구할 수는 없다. 미세 병변을 놓치는 문제가 지속되면 **patch token 유지 / ROI / backbone fine-tuning**을 별도 비교해야 한다.
- 라벨은 검사별 12개 소견이다. 특정 slice마다 정답 병변 위치가 제공되는 것은 아니다. Attention은 이 약한 감독 아래에서 유용한 slice를 찾아야 한다.

DICOM의 `ImagePositionPatient`는 환자 좌표계의 영상 위치, `ImageOrientationPatient`는 row/column 방향을 제공한다. 현재 정렬은 두 방향의 외적으로 normal을 구하고 영상 위치를 그 normal에 투영한다. 더 정밀한 3D encoding을 하려면 원본 위치·방향·spacing을 함께 유지하는 cache 확장이 필요하다. [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

## 6. 무엇을 보고 판단할까

공통 조건은 Swin 고정 특징, 학습 대상 4,350검사, Gold58의 동일 3 folds × 3 seeds, 기존 confidence-weighted BCE, head 20 epochs이다. Gold 검증 fold는 해당 fold 모델의 gradient와 특징 정규화 통계에 포함되지 않는다. 약라벨·확신도 기준·loss 집계 방식은 이번 실험에서 바꾸지 않는다.

1. **재현성:** T00 vs S11. 먼저 구현 차이나 cache 차이가 없는지 확인한다.
2. **주 지표:** seed별 및 평균 macro ROC-AUC, seed 표준편차, BCE.
3. **카테고리별 변화:** 12개 소견 AUC와 threshold 0.5의 TN/FP/FN/TP, 재현율·정밀도. 특히 MCL, Lateral Meniscus, Fracture의 FN을 확인한다.
4. **쌍 비교:** 같은 검사 단위 paired bootstrap으로 T01−T00, T02−T01 차이의 구간을 확인한다. 58검사의 작은 개발 집합을 반복해서 보므로 독립 최종 검증의 증거로 해석하지 않는다.
5. **선택 기준:** 평균 AUC +0.005 이상, 3 seeds 중 2개 이상 개선을 후보 채택의 기준으로 둔다. 주요 소견 AUC -0.03 초과 하락이나 FN 증가가 있으면 평균 점수와 함께 검토한다. 자동으로 ensemble에 편입하지 않는다.

결과를 읽는 방법:

| 관측 결과 | 가능한 해석 / 다음 검증 |
|---|---|
| T01 개선, T02 추가 개선 없음 | 절대 깊이 신호는 유용하나 명시적 방향×깊이 곱은 불필요할 수 있음 |
| T02가 T01보다 여러 seed에서 개선 | 방향에 따른 깊이 해석이 도움이 된다는 가설을 지지 |
| T01/T02 모두 하락 | 촬영 범위의 변동, 위치 신호의 크기, 기존 attention으로 이미 충분한 표현 가능성 등을 검토 |
| 특정 소견만 개선 | 방향별 관측에 의존하는 병변인지 확인. 곧바로 인과관계로 단정하지 않음 |

다음 단계 후보는 결과 확인 후 선택한다. (a) 물리 거리와 원본 series 길이를 함께 저장하여 상대 깊이 비교, (b) 한 방향씩 제외한 검증 추론으로 방향 의존성 측정, (c) 학습 시 위치를 섞는 음성 대조군, (d) patch token/실제 3D geometry 유지. **이 후보들은 이번 3개 실험에 섞지 않았고 아직 구현·실행한 결과도 없다.** 방향별 attention weight만 보고 어느 방향이 더 정확하다고 결론내릴 수는 없다.

## 7. 코드 위치와 실행

- [위치 encoding과 forward](../experiments/plane_depth_core.py): `PlaneDepthHead.position_features`, `forward`의 `NEW` 주석.
- [실험 설정](../experiments/plane_depth_configs.py): T00/T01/T02, 동일 S11 cache 고정.
- [전체 코드 notebook 생성](../experiments/plane_depth_notebooks.py): 학습·제출에 같은 head를 삽입한다.
- [자동 실행 shell](../scripts/run_plane_depth_experiments.ps1): 기존 큐 완료 확인 → private GPU 학습 → Internet OFF 제출 notebook 실행 → CSV 검증 → OOF/혼동행렬 분석.
- [동작 검증](../tests/test_plane_depth_experiments.py): 재현성, 순서·위치 연결, padding, 학습 gradient, held-out 독립성, offline CSV, 큐 순서.

```powershell
# 업로드 없이 입력 검증과 순서 확인
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_plane_depth_experiments.ps1 -DryRun

# 기존 follow-up 큐 완료를 기다린 뒤 자동 실행. 중단 후에도 같은 명령으로 재개.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_plane_depth_experiments.ps1
```

개별 실행용 전체 코드:

이 PC의 PowerShell 실행 정책 때문에 위처럼 해당 프로세스에서만 `Bypass`를 지정한다. 시스템 설정을 바꾸지는 않는다. Python으로 직접 실행해도 같다: `.\.venv\Scripts\python.exe -u -X utf8 -m experiments.run_plane_depth_queue`.

- [T00 train](../notebooks/plane_depth/t00_position_control/train/train.ipynb) / [submit](../notebooks/plane_depth/t00_position_control/submit/submit.ipynb)
- [T01 train](../notebooks/plane_depth/t01_position_sum/train/train.ipynb) / [submit](../notebooks/plane_depth/t01_position_sum/submit/submit.ipynb)
- [T02 train](../notebooks/plane_depth/t02_position_joint/train/train.ipynb) / [submit](../notebooks/plane_depth/t02_position_joint/submit/submit.ipynb)

각 train은 완료된 S11의 버전 고정 출력에서 feature cache를 재사용한다. 제출 notebook은 자신의 완료된 train 출력만 연결한다. 자동 실행기가 실제 버전 번호를 고정하며, 수동 실행에서는 해당 버전에 맞게 input을 연결해야 한다. GPU 시간은 학습 전 0.75시간, 제출 추론 전 0.5시간 이상의 여유를 요구하는 예약 기준이며 실제 사용량 보장은 아니다. 무료 quota가 부족하면 대기한다.

결과 저장:

```text
outputs/automation/plane-depth-20261019/
  queue_status.json, runner.log, inputs.json, state.json, summary.csv
  <실험ID>/train/run/metrics.json, oof_predictions.csv, timings.json
  <실험ID>/submit/submission.csv, inference_summary.json
outputs/analysis/plane-depth-20261019/
  experiments.csv, category_metrics_all_seeds.csv
  bootstrap_comparisons.json, 혼동행렬 그림 및 실험 비교 HTML
```

기존 큐의 lock이 풀렸다는 이유만으로 새 실험을 시작하지 않는다. R01/R04/R05/R02/R03/R06의 학습·제출 검증과 조건부 R07의 결정/완료까지 확인한다. 기존 큐가 실패해서 종료되면 새 큐도 원인을 기록하고 중단하므로, 기존 큐를 복구한 뒤 같은 shell로 재개한다. PC가 종료되거나 절전되면 로컬 자동 진행도 멈춘다.

이 자동화는 Kaggle에 private 학습·제출 notebook을 올려 `submission.csv`를 만드는 단계까지다. **대회 공식 채점 제출은 자동 수행하지 않는다.** 현재 계획과 실제 대기 상태는 [실행 기록](plane_depth_execution_log.md)에 구분해서 남긴다.
