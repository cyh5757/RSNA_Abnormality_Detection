# 무릎 MRI 특징 추출 backbone 조사

확인일: 2026-10-16. 공식 논문·모델 카드·구현·라이선스를 조사했다. 아래 모델 선택 순서는 이 프로젝트에 대한 **실험 제안**이며, 아직 해당 모델들의 무릎 MRI 성능을 측정한 결과가 아니다.

**사전학습 backbone으로 영상 특징을 추출하고, 여러 영상의 특징을 모아 마지막 분류기로 12개 소견을 예측하는 방식은 적절한 출발점이다.** 현재 ResNet18 baseline도 이 구조다. 먼저 DINOv2 ViT-S/14로 특징 추출기를 교체하고, Swin-T와 ConvNeXt-Tiny를 대조군으로 비교하는 것을 권한다. DINOv3도 실제 공개된 후보지만, 가중치 접근·라이선스 조건을 확인한 후 추가한다.

현재 기준은 [E01 결과](experiment_results.md)의 방향별 5장, frozen ResNet18, pooled OOF macro AUC 평균 **0.606793**이다. 정답이 완비된 검사는 58건이며, 아래 비교에도 같은 검사 분할을 사용한다.

## 사용자가 제안한 구조를 정확히 표현하면

```text
검사 B개, 검사마다 3방향 × 방향별 5장
입력                         [B, 3방향, 5장, 3채널, H, W]
  ↓ 같은 사전학습 backbone을 각 영상에 적용
영상별 특징                  [B, 3방향, 5장, D]
  ↓ 같은 방향의 5장 특징을 평균
방향별 특징                  [B, 3방향, D]
  ↓ 시상면·관상면·축상면 순서로 이어 붙임(concat)
검사별 특징                  [B, 3D]
  ↓ 학습 fold의 통계로 특징 표준화
Linear(3D, 12)               [B, 12] logits
  ↓ sigmoid: 추론·평가할 때
12개 소견별 예측 점수        [B, 12]
```

3채널은 MRI 한 장의 회색조를 RGB 입력 형식에 맞춰 복제한 것이다. 서로 다른 MRI 방향을 RGB 채널에 넣는다는 뜻이 아니다. 위 도식은 세 방향과 충분한 슬라이스가 있는 경우이며, 실제 구현에서는 중복 샘플 제거·결측 방향 처리도 유지해야 한다. 현재 구현은 [특징 통합 함수](../notebooks/experiments/slice_ablation/slice_ablation.py)의 `study_features`, 분류기 학습은 같은 파일의 `fit_head`에 있다.

예를 들어 DINOv2 ViT-S/14의 영상별 특징 `D=384`를 사용하면 방향별 평균도 384차원이고, 세 방향 concat은 1,152차원이다. 분류기는 `Linear(1152, 12)`가 된다. 384차원은 공식 모델 카드에 명시되어 있다. [DINOv2 모델 카드](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)

`Linear`는 특징 벡터를 12조각으로 잘라 내는 함수가 아니다. 각 소견이 **전체 특징 벡터에 서로 다른 가중치를 적용**한다. 검사 특징을 `h`라고 하면 소견 `c`의 logit은 `z_c = w_c · h + b_c`다. ACL과 Effusion이 같은 특징 일부를 함께 이용해도 된다.

“벡터 합”은 다음을 구분해야 한다.

- **같은 backbone으로 뽑은 여러 슬라이스:** 좌표의 의미를 공유하므로 평균 pooling을 사용할 수 있다. 단순 합보다 장수가 달라질 때 크기가 덜 흔들린다.
- **서로 다른 MRI 방향:** 현재는 방향별로 평균을 낸 뒤 concat한다. 모든 방향을 한 번에 더하면 어느 방향에서 나온 정보인지 잃는다. concat은 방향별 자리를 보존한다.
- **서로 다른 backbone의 특징:** 차원이 같아도 각 좌표의 의미가 정렬되어 있지 않으므로 바로 더하지 않는다. 필요하다면 나중에 특징 concat이나 모델별 예측 앙상블을 별도 실험한다.

## MNIST와 같은 점, 다른 점

영상 → 특징 → 학습되는 분류기라는 큰 구조는 같다. 다만 일반적인 MNIST는 0~9 중 하나인 단일 정답 문제이고, 여기서는 한 검사에 ACL 손상과 Effusion 등 여러 소견이 함께 존재할 수 있다.

| 항목 | 일반적인 MNIST | 이 프로젝트 |
| --- | --- | --- |
| 정답 | 숫자 하나 | 12개의 0/1 값 |
| 출력 | 서로 경쟁하는 10개 class 점수 | 소견별 12개 점수 |
| 최종 활성화 | softmax | 각 출력의 sigmoid |
| 학습 loss | cross entropy | `BCEWithLogitsLoss` |
| 해석 | 한 숫자를 선택 | 여러 소견이 동시에 높을 수 있음 |

`BCEWithLogitsLoss`에는 sigmoid 전 logits를 넣는다. 이 함수가 sigmoid와 binary cross entropy를 안정적으로 함께 계산한다. 현재 양성 가중치 `pos_weight`는 각 학습 fold에서만 계산하며, 첫 backbone 비교에서도 그대로 유지한다. [PyTorch BCEWithLogitsLoss 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=criterion)

따라서 사전학습 때 DINO를 사용했다고 대회 학습 loss까지 DINO loss로 바꾸는 것은 아니다. **기존 사전학습 가중치에서 특징을 추출하고, 무릎 소견 head는 BCE로 학습**한다. 처음에는 backbone을 고정하고 head만 학습한다. 이는 작은 정답 데이터에서 구현·계산량·과적합을 관리하기 위한 이번 프로젝트의 선택이다. 인코더 전체 미세 조정은 별도 단계다.

## 모델 후보와 선택 근거

ViT·Swin·ConvNeXt는 구조 이름이고, DINO는 자기지도 사전학습 방법 계열이다. 따라서 “ViT와 DINO 중 하나”라는 구분은 정확하지 않다. **DINOv2로 사전학습된 ViT**를 사용할 수 있다. DINOv3에는 ViT뿐 아니라 ConvNeXt 가중치도 있다. [DINO 공식 구현](https://github.com/facebookresearch/dino), [DINOv3 모델 카드](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)

| 후보 | 확인된 특징·출력 | 이 프로젝트에서의 판단 |
| --- | --- | --- |
| **DINOv2 ViT-S/14** | 자기지도 사전학습 ViT, 약 21M 파라미터, CLS 특징 384차원. 공식 문서는 고정 특징 위에 선형 분류기를 사용하는 방법을 설명한다. [공식 모델 카드](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) | **첫 번째 비교 대상.** 작은 모델로 frozen feature 방식의 효과를 확인하기 좋다. |
| **Swin-T** | 윈도 기반 attention과 단계별 해상도 축소를 사용하는 계층형 Transformer. torchvision ImageNet1K 가중치는 약 28.3M 파라미터, 224 입력 기준 4.49 GFLOPS다. [논문](https://arxiv.org/abs/2103.14030), [가중치 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html) | 국소 정보를 다루는 구조가 도움이 되는지 볼 Transformer 대조군. MRI 우위는 실험 전까지 가설이다. |
| **ConvNeXt-Tiny** | 현대화된 CNN. torchvision ImageNet1K 가중치는 약 28.6M 파라미터, 224 입력 기준 4.46 GFLOPS다. [공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) | 강한 CNN 대조군. Transformer 교체 자체가 필요한지 확인한다. |
| **일반 지도학습 ViT-B/16** | 이미지 패치를 Transformer에 입력하는 구조. torchvision `IMAGENET1K_V1`은 약 86.6M 파라미터, 224 입력 기준 17.56 GFLOPS다. [원 논문](https://arxiv.org/abs/2010.11929), [가중치 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) | 일반 ViT 비교에는 적합하나 우선 후보보다 크다. 작은 후보들을 본 뒤 추가한다. |
| **DINOv3 ViT-S/16** | 실제 제공되는 최신 계열 후보. 약 21M 파라미터, 384차원, 패치 크기 16, register token·RoPE 사용. [공식 모델 카드](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md) | 접근 조건을 충족하면 frozen 특징 추출 후보로 추가. 최신이라는 이유만으로 무릎 MRI 최선이라고 판단하지 않는다. |
| Swin V2-T | torchvision 가중치는 약 28.4M 파라미터, 256 입력 기준 5.94 GFLOPS다. [공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_v2_t.html) | Swin-T 다음의 선택 후보. 첫 비교에서 둘을 모두 넣어 실험 수를 늘릴 필요는 없다. |

표의 torchvision 파라미터 수와 GFLOPS는 **원래 ImageNet 분류기까지 포함한 공식 수치**다. 실제 특징 추출에서는 1,000-class 분류기를 제거하므로 파라미터 수가 조금 줄어든다. GPU 지연시간·메모리는 별도 측정해야 하며, 위 연산량으로 Kaggle T4 속도를 확정하지 않는다.

ViT-B/16, Swin-T/V2-T, ConvNeXt-Tiny의 마지막 분류기 직전 특징은 각각 768차원이다. Swin은 전역 평균 pooling, ConvNeXt는 pooling과 정규화 이후 표현, ViT는 CLS 표현을 사용한다. 각 구현에서 **분류 linear만 제거하고 필요한 정규화를 유지**해야 한다. [ViT 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html), [Swin 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/swin_transformer.html), [ConvNeXt 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/convnext.html)

## DETR와 이름이 같은 DINO는 어떻게 다른가

DETR는 이미지의 물체 종류와 박스를 예측하는 전체 detection 모델이다. 원 구현의 backbone은 ResNet50/101이며, 그 위에 Transformer와 detection head가 있다. 즉 DETR 자체를 ViT·Swin과 같은 수준의 “분류 backbone”으로 비교하면 서로 다른 구성 요소를 비교하게 된다. [DETR 공식 저장소](https://github.com/facebookresearch/detr)

또한 IDEA Research의 **DINO: DETR with Improved DeNoising Anchor Boxes**는 물체 검출 모델이다. Meta의 자기지도 DINO/DINOv2/DINOv3와 다른 연구다. 이름만 보고 detection DINO를 받아서는 안 된다. [Detection DINO 공식 저장소](https://github.com/IDEA-Research/DINO)

이 대회의 현재 학습표는 검사별 소견 정답이며 병변 박스 정답이 아니다. DETR 내부 특징을 재사용하거나 weak supervision으로 재설계하는 일은 가능하지만, 박스 검출 성능을 평가하는 문제가 아니므로 첫 backbone 실험에는 DINOv2·Swin·ConvNeXt가 더 직접적인 선택이다. 이는 금지 규정 때문이 아니라 현재 목표·정답 형태와 구현 비용을 고려한 판단이다.

## 가중치 접근과 전처리

일반 DINOv2 모델 카드는 Apache 2.0을 명시하며 공식 저장소에서 사전학습 가중치를 제공한다. 같은 저장소에 추가된 XRay-DINO·Cell-DINO의 조건을 일반 DINOv2 조건으로 혼동하지 않는다. 이번 후보는 **일반 `dinov2_vits14`**다. [DINOv2 모델 카드](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [공식 가중치 목록](https://github.com/facebookresearch/dinov2#pretrained-models)

DINOv3는 별도 **DINOv3 License**를 사용한다. 공식 안내는 접근 신청이 승인되면 가중치 URL을 이메일로 제공하는 절차를 설명한다. 따라서 현재 계정에서 이미 접근 가능하다고 가정하지 않는다. 실제 채택 전에는 사용·재배포 조건과 대회의 공개 외부 모델 조건을 함께 확인해야 한다. 이번 조사에서는 다운로드 승인·조건 동의·가중치 업로드를 수행하지 않았다. [공식 다운로드 안내](https://github.com/facebookresearch/dinov3#pretrained-models), [DINOv3 License 원문](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

영상 선택·DICOM 강도 처리는 동일하게 유지하되, backbone에 들어가는 영상의 resize·crop·정규화는 정확한 가중치에 맞춰 기록한다.

| 가중치 | 공식 평가/추론 변환 예시 |
| --- | --- |
| DINOv2 | 짧은 변 256 → 224 중앙 crop, bicubic, ImageNet 정규화. [공식 평가 변환 코드](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py) |
| Swin-T ImageNet1K V1 | 짧은 변 232 → 224 중앙 crop, bicubic. [공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html) |
| ConvNeXt-Tiny ImageNet1K V1 | 짧은 변 236 → 224 중앙 crop, bilinear. [공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) |
| ViT-B/16 ImageNet1K V1 | 짧은 변 256 → 224 중앙 crop, bilinear. [공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) |
| Swin V2-T ImageNet1K V1 | 짧은 변 260 → 256 중앙 crop, bicubic. [공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_v2_t.html) |
| DINOv3 LVD 가중치 | README 예시는 256×256 resize와 ImageNet 정규화. 위 후보들과 자동으로 같은 변환이라고 가정하지 않는다. [공식 README](https://github.com/facebookresearch/dinov3#pretrained-models) |

이 비교는 구조뿐 아니라 **사전학습 데이터·학습법·가중치·필수 전처리·특징 차원**이 함께 다른 “특징 추출 구성” 비교다. DINOv2가 좋아져도 그 차이를 전부 Transformer 구조 덕분이라고 해석할 수 없다. 모델별 공식 입력을 우선 사용하고, 해상도·crop을 통제하는 실험은 필요할 때 따로 한다.

## 권장 실험 순서와 비교 조건

1. **E02 첫 비교:** 기존 5장 ResNet18 대 DINOv2 ViT-S/14. 방향별 평균 + concat + linear head를 유지한다. DINO의 CLS/patch pooling까지 동시에 여러 방식으로 바꾸지 않는다.
2. **같은 방식의 추가 대조:** Swin-T, ConvNeXt-Tiny를 각각 독립적으로 비교한다. 이미 본 검증셋에 후보를 무한히 추가하지 않고 이 세 신규 후보에서 우선 결과를 정리한다.
3. **필요한 경우만 확대:** 지도학습 ViT-B/16 또는 접근이 확보된 DINOv3 ViT-S/16. Swin V2-T는 Swin 계열을 더 확인할 이유가 생겼을 때 추가한다.
4. **선택한 backbone을 고정한 후:** 평균 대신 작은 attention 통합기 → 동일 통합기에 슬라이스 위치 추가 → 방향 임베딩 추가 순으로 비교한다. [기존 단계별 계획](superpowers/plans/2026-10-16-progressive-experiments.md)을 따른다.

고정할 조건은 검사 ID별 3-fold, 초기화 seed 42·2026·3407, 방향별 같은 5개 슬라이스, train fold 통계만 쓰는 특징 표준화·양성 가중치, 동일한 linear head 학습 설정이다. 우선 기존 5 epoch 설정으로 비교하되, 이는 “고정 예산 아래의 첫 비교”다. 모든 backbone의 최적 학습 횟수라고 주장하지 않는다. 학습 곡선상 미수렴이 확인되면 전 후보에 공통으로 적용하는 추가 프로토콜을 먼저 정한 뒤 재비교한다.

기록할 결과는 seed별 pooled OOF macro AUC, fold별·소견별 AUC, unweighted BCE, 특징 추출 시간, GPU 메모리, 가중치·코드·전처리 버전이다. 파라미터 수가 다르면 head 크기도 다르므로 함께 기록한다. 58건에서 seed 간 일관성이 있어도 독립 테스트의 개선이나 통계적 유의성을 보장하지 않는다.

## 좋은 backbone도 슬라이스 순서를 자동으로 해결하지는 않는다

ViT의 patch 위치 인코딩은 **한 이미지 안의 패치 위치**를 다룬다. 5장의 MRI를 각각 인코딩한 뒤 평균하면 슬라이스 순서를 섞어도 평균은 같다. DINOv3의 2D RoPE를 쓰더라도 이 통합 구조에서는 MRI의 앞뒤 순서가 자동으로 생기지 않는다. [ViT 위치 인코딩 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html), [DINOv3 모델 카드](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)

예를 들어 위치 벡터를 더하고 곧바로 평균만 내면 `mean(feature_i + position_i) = mean(feature_i) + mean(position_i)`이므로 어느 특징이 어느 위치에 있었는지의 결합을 학습하지 못한다. 순서를 이용하려면 슬라이스별 특징을 유지한 채 위치와 함께 attention/sequence 통합기에 넣는 별도 단계가 필요하다.

또한 평균은 몇 장에만 나타나는 작은 병변의 신호를 희석할 수 있다. backbone 교체 후에도 이 문제가 남는다면 attention pooling이나 patch 특징 활용을 비교한다. 반대로 정답이 적은 상태에서 통합기를 크게 만들면 과적합할 수 있으므로, 좋은 backbone + 단순 head 결과를 먼저 확보하는 편이 해석하기 쉽다.

이번 작업은 조사와 설계 정리다. 신규 backbone 학습·DINOv3 접근 승인·공식 제출은 실행하지 않았다. 자연 이미지에서의 공개 성능만으로 이 대회의 최적 backbone을 확정할 수 없다.
