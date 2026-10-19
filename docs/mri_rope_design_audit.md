# HTML 확장 설계와 현재 MRI 학습 코드 대조

확인일: 2026-10-16. 입력 문서: `C:\Users\USER\Downloads\MRI_학습_전체파이프라인_차원_RoPE_확장설계.html`. 해당 HTML의 입력·차원·RoPE·방향·fusion·학습·앙상블·로드맵을 현재 소스, 자동 실행 상태, 로컬 테스트와 대조했다.

**판정: 제안한 학습 방향은 구현 가능하지만, 현재 자동 실험은 일부만 구현한다.** 약라벨 확대, 인접 3장, 촬영 조건 분리, 소견별 MIL, 부분 fine-tuning, 확률 앙상블은 구현되어 있다. 현재 약라벨 모델에는 슬라이스 위치 입력, RoPE, 방향 간 cross-attention이 없다. 자동 큐가 정상 실행되는 것과 원하는 전체 아키텍처가 구현된 것은 구분해야 한다.

이번 작업에서는 학습 소스·실험 설정·Notebook·실행 큐를 변경하지 않았다. 이 파일은 확인 결과와 다음 실험 설계를 기록한다.

## 1. HTML은 원본 baseline 설명 + 확장 제안이다

HTML의 원본 설명은 `train_baseline.py`의 frozen ResNet18, 방향당 최대 3장, 방향별 평균과 선형 분류기를 대상으로 한다. HTML이 구분한 ‘확장 제안’은 완성된 실행 코드가 아니다.

현재 자동 큐의 실제 경로는 `scripts/run_weak_experiments.ps1` → `experiments/weak_configs.py` → 생성된 Kaggle train/submit Notebook이며, 핵심 코드는 `weak_core.py`, `weak_train.py`, `weak_submit.py`다. 현재는 Swin-T, 768차원 특징, 약라벨 실험을 사용하므로 HTML의 ResNet18·512차원 숫자를 그대로 적용하면 안 된다.

| 원하는 기능 | 현재 상태 | 코드 근거 |
| --- | --- | --- |
| 인접 영상 3장 | **E08부터 구현**. 같은 시리즈의 이전·현재·다음 영상을 3채널로 넣음 | `weak_core.py:99` |
| 인접 영상 5장·10장 | **미구현**. 채널 수와 인접 인덱스가 3으로 고정 | `weak_core.py:83`, `weak_core.py:100` |
| 한 시리즈에서 고르는 중심 위치 수 변경 | 코드가 `slices`를 받지만 현재 새 실험은 모두 **5개로 고정** | `weak_configs.py:19`, `weak_core.py:96` |
| 촬영 방향·조건 구분 | **E09부터 6개 슬롯**: 3방향 × 지방 억제 여부. 슬롯별 자리와 분류 가중치로 구분 | `weak_core.py:80`, `weak_core.py:137` |
| 슬라이스 위치 encoding | 예전 **E04/E05에는 구현**, 현재 E06~E12 `WeakHead`에서는 사용하지 않음 | `core.py:167`, `weak_core.py:143` |
| 학습 가능한 plane embedding | 예전 **E05에는 구현**, 현재 약라벨 모델에는 명시적 embedding 없음 | `core.py:157` |
| RoPE | **미구현**. 예전 E04의 sin/cos 덧셈은 RoPE가 아님 | `core.py:168` |
| 소견별 attention | **E10부터 구현**. 각 슬롯의 window 중요도를 소견별로 계산 | `weak_core.py:149` |
| 슬라이스 쌍의 순서·거리 비교 | 현재 약라벨 head에는 **없음**. 3채널 내부 순서가 고정되는 효과와 구분 | `weak_core.py:143` |
| 방향 간 cross-attention / label-query cross-attention | **미구현**. 슬롯별 pooling 후 연결·선형 합산 | `weak_core.py:151` |
| 정규화 | train 특징의 mean/std로 표준화. Swin 내부 정규화도 있으나 **새 fusion LayerNorm은 없음** | `weak_core.py:145`, `weak_core.py:166` |
| 부분 fine-tuning | **E11 구현, 실행 대기**. Swin 마지막 stage와 최종 norm, 첫 예산 1 epoch | `weak_train.py:57`, `weak_configs.py:22` |
| 앙상블 | **E12 구현, 실행 대기**. E10/E11의 예측 확률을 50:50 평균 | `weak_train.py:156`, `weak_submit.py:52` |

E04/E05의 위치·방향 기능이 E07 이후에 자동 누적된 구조는 아니다. 약라벨 실험은 별도 `WeakHead`로 진행하며, 이전 위치 실험 결과만으로 약라벨 데이터에서 위치 정보가 불필요하다고 결론낼 수 없다.

## 2. 현재 입력은 ‘5장’과 ‘인접 3장’이 다른 축이다

표기: B=검사 배치, S=촬영 슬롯(3 또는 6), M=중심 위치 수(현재 5), L=한 중심 주변의 인접 슬라이스 수(현재 3), D=encoder 특징 차원(768).

```text
이미 촬영된 sagittal / coronal / axial 시리즈 선택
  → DICOM 공간 위치 또는 InstanceNumber로 정렬
  → 각 시리즈의 중앙 25~75%에서 중심 위치 M=5개 선택
  → E08부터 각 중심에서 [이전, 현재, 다음] L=3장을 채널로 쌓음
  → images [B,S,5,3,224,224]
  → Swin-T: 3장 window 하나를 768차원으로 요약
  → tokens [B,S,5,768]
  → E07~E09: 슬롯 안에서 평균 → [B,S,768] → 연결 → 12 logits
  → E10/E11: 소견별 슬롯 내 가중합 → [B,S,12,768]
               → 소견별로 슬롯 연결 → 12 logits
  → sigmoid → 검사당 12개 확률
```

세 방향은 새로운 3D 재절단으로 생성하지 않고 이미 있는 촬영 시리즈를 선택한다. E09 이후에도 같은 슬롯 안의 여러 시리즈를 전부 쓰는 것은 아니며 우선순위로 하나를 고른다.

`slices=10`으로 바꾸면 **중심 10개 × 인접 3장**이 된다. ‘각 중심에서 인접 10장’이 되지 않는다. 6슬롯×5window는 최대 30개의 encoder 입력이며, 각 입력의 3개 채널이 원본 슬라이스 3장이다. 서로 겹치거나 경계에서 반복될 수 있어 최대 90개의 참조가 반드시 90개의 고유 DICOM을 뜻하지는 않는다.

## 3. 순서는 어디까지 반영되는가

현재 `ordered_slices()`는 물리 위치를 이용해 정렬한 뒤 위치를 시리즈별 0~1로 정규화한다. 위치 메타데이터가 없으면 InstanceNumber를 쓰지만, 반환값에는 어느 경로였는지 또는 실제 mm 간격이 남지 않는다. (`core.py:67`)

`study_views()`가 중심 위치를 반환하고 `tokens.pt`에도 저장하지만, `WeakHead.forward(tokens, mask)`에는 positions 인자가 없다. 학습과 제출 모두 이 위치를 전달하지 않는다. (`weak_train.py:258`, `weak_submit.py:49`)

따라서 구분해야 할 두 관계가 있다.

- **window 안의 이전/현재/다음:** 입력 채널에 순서가 고정되어 있어 encoder가 채널별 차이를 이용할 수 있다. 완전히 순서 정보가 없다고 말하면 부정확하다. 그러나 각각 독립된 token과 좌표로 비교하는 attention은 아니다.
- **다섯 window 사이의 앞/중앙/뒤:** 현재 head는 순서와 거리를 이용하지 않는다. E10은 window마다 내용에 기반한 12개 점수를 계산하고 슬롯 안에서 가중합한다. window 쌍의 Q/K 관계를 계산하는 self-attention이 아니다.

직접 확인한 E10 순서 교환 검사에서 최대 logit 차이는 **9.69×10⁻⁸**로, float 연산 오차 범위였다. 기존 테스트도 이 순서 불변성을 확인한다. 이는 버그가 아니라 현재 MIL 설계의 성질이며, 사용자가 원하는 순서 모델링은 아직 빠져 있다는 증거다.

## 4. 인접 5장·10장을 구현하는 두 경로

**채널 확대:** 현재 3채널을 5/10채널로 바꾸면 Swin의 입력 patch embedding 및 사전학습 가중치 초기화, 채널별 정규화를 함께 바꿔야 한다. 5→3 adapter를 넣는 방법도 있으나 그 단계에서 다시 압축된다. `slices` 설정 하나로 해결할 변경이 아니다.

**슬라이스 token 유지:** 각각의 흑백 영상을 encoder 입력용 3채널로 복제하되, 인접 영상마다 독립적인 특징 token을 만든다. 그러면 L=3/5/10 모두 같은 RGB 사전학습 encoder를 사용할 수 있고, 슬라이스별 위치를 attention에 넣기 쉽다. 사용자의 ‘인접 이미지도 순서대로 비교’ 의도에는 이 경로를 권한다. 먼저 L=3에서 현재 3채널 방식과 비교해야 입력 표현 변경 효과와 장수 효과가 섞이지 않는다.

```text
같은 S개 슬롯, 같은 M개 중심에서 주변 L장 선택
  → 이미지 [B,S,M,L,3,H,W]  # 마지막 3은 한 장을 복제한 RGB 입력
  → 특징 [B,S,M,L,768]     # 인접 장마다 token 유지
  → 같은 시리즈의 중복 slice 참조를 합치고 원래 순서로 정렬
  → padded tokens [B,S,T,768] + positions + valid_mask
```

T는 실제 선택한 고유 슬라이스 수이며 최대 M×L이다. 중복을 제거하지 않는 경로라면 겹치는 영상이 더 큰 가중치를 받는 효과를 별도로 통제해야 한다. 짧은 시리즈는 padding mask를 사용한다. 짝수 10장에서는 중심이 두 장 사이에 놓이므로 선택 기준을 고정하고 실제 위치를 저장한다.

M=5, S=6에서 L=3/5/10이면 중복 제거 전 각각 최대 90/150/300장의 개별 영상 encoding이 필요하다. 이는 현재 30개 window encoding보다 많다. 처음에는 frozen encoder의 개별 슬라이스 특징을 재사용하고, 실제 추론 시간·메모리를 측정한다. 처리 시간 증가율을 이 장수 비율과 같다고 단정하지 않는다.

## 5. 사용자가 원하는 후단 구조

아래는 **추가 구현 제안**이며 현재 실행되는 모델이 아니다. 먼저 backbone은 Swin-T로 고정한다. d=128 또는 256, 얕은 1-block 집계로 시작할 수 있다.

```text
슬라이스 특징                [B,S,T,768]
  → projection              [B,S,T,d]
  + plane/protocol embedding
  → 시리즈별 slice self-attention
       위치 없음 / 위치 덧셈 / index RoPE / mm RoPE 비교
       LayerNorm + residual + FFN
                             [B,S,T,d]  # 평균내지 않음
  → 모든 유효 token을 memory로 연결
                             [B,S*T,d]
  → 12개 소견 query가 memory에 cross-attention
       Q=[B,12,d], K/V=[B,S*T,d]
       residual + LayerNorm/FFN
                             [B,12,d]
  → 소견별 head              [B,12]
  → 현재의 confidence-weighted masked BCE
```

### RoPE의 역할과 위치

정식 이름은 **Rotary Position Embedding**이다. 시리즈 내 self-attention의 Q/K를 위치에 따른 각도로 회전시켜 상대 위치가 관계 계산에 반영되게 한다. 이미지 자체를 회전하는 증강이 아니며, 마지막 pooled vector에 위치를 더하는 것과도 다르다. [RoFormer 원 논문](https://arxiv.org/abs/2104.09864)

현재 0~1 좌표로 상대 위치 실험은 가능하지만, 실제 mm 기반 RoPE에는 **원본 투영 좌표, 좌표 유효 여부, 원본 slice index, 시리즈 식별자와 일관된 스케일**이 추가로 필요하다. 기존 normalized positions만으로 mm 간격을 복원할 수 없다. 먼저 시리즈 안에서 RoPE를 적용하고, 서로 다른 방향의 좌표를 하나의 1D 깊이축으로 이어 붙이지 않는다. 내용 기반 cross-attention 자체에는 서로 다른 방향의 정확한 영상 정합이 필수는 아니지만, 해부학적 위치 대응을 주장하려면 별도의 공통 좌표·정합 검증이 필요하다.

### 방향 식별과 방향 중요도는 다르다

plane/protocol embedding은 token의 출처를 알려 준다. 어떤 방향이 특정 소견에 유용한지는 **소견별 query/gate와 데이터에서 학습하는 관계**다. 시상·관상·축상을 0·1·2라는 가짜 연속 깊이로 취급하지 않는다.

현재도 concat의 고정 슬롯과 별도 분류 가중치 덕분에 방향별 기여를 다르게 줄 수 있다. 빠진 것은 token을 유지한 채 방향 사이를 조회하는 상호작용이다. 학습 후에는 동일 분할의 방향 단독 모델, 방향 제거 재학습, 소견별 AUC를 비교한다. attention이 높다는 사실만으로 그 방향이 정확도를 올렸다고 결론내리지 않는다. 추론 때만 방향을 지우는 검사는 분포 변화가 포함된 민감도 분석으로 구분한다.

### concat·cross-attention·정규화의 정확한 차이

특징은 이미 벡터다. concat은 벡터 성분을 이어 붙이므로 그 연산 자체가 정보를 평균내지는 않는다. 현재의 주요 압축은 **Swin의 전역 공간 pooling**과 **head의 슬롯 내 가중합**에서 일어난다. 실제 Swin 구성에 `AdaptiveAvgPool2d`가 남아 있고 출력은 window당 `[768]`이다.

cross-attention은 소견 query가 여러 영상 token을 선택적으로 참조하게 한다. 역시 가중합과 projection을 사용하므로 무손실 결합은 아니다. residual과 token 유지로 너무 이른 압축을 줄일 수 있다. LayerNorm은 수치 스케일을 안정화하며 이미 사라진 위치 정보를 복원하지 않는다. Q/K/V가 별도 입력인 구조는 [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)을 기준으로 구현할 수 있다.

슬라이스 내부의 작은 병변 위치까지 유지하려면 다음 단계에서 Swin의 global pooling 전 공간 token을 사용해야 한다. 현재 768차원 캐시에서 patch 위치를 복원할 수는 없다. 처음부터 모든 patch를 합치기보다는 슬라이스별 token 구조의 효과를 먼저 비교한다.

## 6. 비교 실험 순서

현재 E06~E12를 바꾸지 않고 결과를 남긴 뒤 다음 실험군을 별도로 만든다. 아래의 각 행도 가능하면 한 변경씩 분리한다.

| 순서 | 비교할 변경 | 고정 조건 |
| --- | --- | --- |
| 1 | 현재 3채널 window vs 한 장씩 독립 token(L=3) | 같은 원본 영상·슬롯·약라벨·Swin·검증 분할, 동일하게 정한 pooling |
| 2 | 인접 L=3/5/10 | M·샘플링 중심·encoder·집계·학습 예산. 중복과 padding 정책 고정 |
| 3 | 중심 M=5/10 또는 중앙/넓은 범위 | L 및 나머지 조건 고정. 장수와 범위는 별도 비교 |
| 4 | MIL vs 위치 없는 slice Transformer | 같은 token·후단 공통 차원. 학습 가능한 관계 모듈의 효과 |
| 5 | 위치 없음 → 위치 덧셈 → 원본 index RoPE → mm RoPE | 같은 Transformer 크기·token·loss·학습 예산. 위치 출처·스케일 기록 |
| 6 | plane/protocol embedding 유무 | 위치 방식과 fusion 고정 |
| 7 | concat+linear → concat+MLP → class gate → label-query cross-attention | 같은 encoder·slice representation. 비선형 head와 attention 효과 분리 |
| 8 | 선택한 구조의 frozen vs 마지막 stage fine-tuning | raw 이미지로 gradient 전달. encoder 변경 후 최종 특징 캐시 재생성 |
| 9 | 성능·오류가 다른 모델의 확률 평균 | 같은 OOF 정렬, 고정 혼합 비율, 모델별 전처리 유지 |

단계 5~7의 frozen encoder 실험은 같은 개별 슬라이스 특징을 재사용할 수 있어 비용을 줄일 수 있다. L/M 변경 때는 해당 슬라이스가 캐시에 있는지 확인하고, 입력·encoder hash가 다른 캐시는 재사용하지 않는다. 현재처럼 gold 58건을 반복 비교하는 평가는 개발용이며, 충분한 weak validation과 소견별 결과를 함께 기록해야 한다.

E11은 마지막 stage의 첫 1-epoch 예산이며 장시간 수렴을 검증한 설정은 아니다. E12는 두 모델의 확률 평균이며, seed/fold/backbone을 모두 섞는 대형 앙상블은 아니다. fine-tuning·앙상블을 이미 지원한다는 이유로 새 RoPE/fusion까지 학습되는 것으로 해석해서는 안 된다.

## 7. 구현 시 함께 변경해야 할 범위

- `experiments/core.py:67`, `weak_core.py:79`: 슬라이스 선택, 실제 index/mm/좌표 유효 여부 반환, 개별 영상 token 경로.
- `weak_core.py:126`: positions·slot metadata를 받는 후단, slice attention·RoPE·fusion, mask 처리.
- `weak_train.py:22`, `weak_train.py:64`, `weak_train.py:258`: CV·fine-tuning·최종 학습에 같은 좌표 전달, 캐시 스키마·hash 갱신.
- `weak_submit.py:9`, `weak_submit.py:49`: 학습과 동일 구조를 복원하고 오프라인 추론에도 위치 전달.
- 실험 config, `build_notebooks.py`, 실행 shell: 기존과 별도의 실험 ID·학습/제출 Notebook·결과 폴더. 원본 Python만 고치고 이미 생성된 Notebook을 그대로 두면 Kaggle 동작은 바뀌지 않는다.
- 테스트: 위치와 token을 함께 재배열했을 때의 일관성, 위치만 바꾼 경우의 동작, padding·방향 전체 결측, L=1/3/5/10, raw 이미지→해제한 encoder까지 gradient, export/reload·제출 일치.

RoPE를 위해 SDPA를 직접 쓰면 bool mask 의미를 주의해야 한다. `MultiheadAttention.key_padding_mask=True`는 제외, `scaled_dot_product_attention.attn_mask=True`는 참여다. 전부 결측인 슬롯은 attention을 건너뛰거나 명시적 정책으로 처리한다. [SDPA 공식 문서](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 8. 이번에 실제 확인한 것

- `python -X utf8 -m unittest tests.test_weak_experiments -v`: **9개 통과**. 인접 slice/시리즈 경계, 결측 mask, UNK zero gradient, train-only 통계, fine-tuning gradient 범위, 학습/제출 Notebook, bundle 복원, 앙상블 정렬을 포함한다.
- E10 token 순서를 섞어도 예측은 부동소수점 오차 이내로 동일했다. `forward`는 tokens와 mask만 받았다.
- E10 head는 token 개수 1/5/10에서 유한한 `[2,12]`를 출력했다. 이는 head의 shape 확인이며 인접 10장 입력 구현 또는 성능 검증이 아니다.
- 다운로드 없는 임의 초기화 Swin으로 `[1,3,224,224] → [1,768]`, global pooling 잔존을 확인했다.
- 자동 실행 로그에서 E06 train/submit 완료·CSV 검증, E07 실행 중을 확인했다. 로컬 queue 프로세스도 살아 있었고, Kaggle MCP로 최신 E07 상태 **RUNNING**을 재확인했다. E08~E12는 아직 실행 결과가 아니다.

새 RoPE·cross-attention의 실제 학습이나 성능 향상은 검증하지 않았다. 이번 판정은 기존 구현의 기능 범위·구조·실행 상태에 관한 것이며, HTML의 제안을 이미 구현했다는 의미가 아니다.
