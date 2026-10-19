# 공개 Code 원문 검토: 11–25번

조사일: 2026-10-19. 아래 점수는 **저자의 본문·제목에 기재된 값**이다. 이번 조사에서 해당 가중치로 hidden test를 재실행하거나 제출 이력의 숫자 점수를 독립적으로 검증하지 않았다. 셀 번호는 다운로드한 원본의 **0부터 시작하는 번호**다. 소스는 실행하지 않고 읽었다. 제목보다 최종 실행 경로와 설정을 우선했다.

## 11. Knee-centred ConvNeXt 추가 — bliverrigpdoge, v1

[원본](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext)

- 저자 보고: R7+Raptor 기본 조합 **0.956 → 0.957**. 기본 결과에 자체 knee-centred 모델의 rank를 20% 섞는다.
- 셀 2–6은 Raptor SWA와 compact R7의 추론·rank 결합이다. 셀 7은 별도 `leg_main.py`와 `members.json`을 찾고 자체 모델을 실행한다. 셀 9의 실제 가중치는 `base=0.8, dm=0.2`다.
- 자체 branch의 ConvNeXt-small/EfficientNetV2-S 및 medial/lateral crop 설명은 저자 본문에 있다. **그 추가 가중치가 비공개**라는 사실도 명시돼 있다. 셀 7·9에서 해당 입력이 없으면 추가 branch를 제외하고 기본 CSV를 유지한다. 따라서 공개 fork가 저자의 0.957 모델 전체를 재현한다고 볼 수 없다.
- 우리에게 필요한 것은 해당 숫자의 복제가 아니라 **서로 다른 시야의 모델이 다른 오류를 내는지 확인하는 방법**이다. 작은 장기·부위 crop을 별도 모델로 학습하고 OOF 오류 상관과 blend 효과를 확인하는 후보로 삼는다. 성능이 낮은 모델이면 다양성만으로 도움이 되지 않는다.

## 12. Four-model ensemble — laimerbotrolenk, v1 / 내부 실험명 V33

[원본](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble)

- 기준 조합의 저자 보고 점수는 **0.957**. 그러나 **현재 mirror-mix 후보의 점수는 미측정**이라고 셀 0과 20이 명확히 구분한다. 이 후보를 0.957 개선 성공 사례로 집계하지 않는다.
- 셀 3의 기본 선택은 `mirror_mix_half`. R7 0.30 + S6 0.20 + Raptor384 TTA 0.325 + Raptor224 plain-rank 0.0875 + Raptor224 TTA-rank 0.0875다. 기준 조합은 마지막 두 항 대신 224 TTA-rank 0.175를 쓴다.
- 셀 11에서 이미 계산한 plain/mirror 출력을 보관하고 셀 20에서 조합한다. **확률 75:25 평균과 rank 공간 50:50 혼합은 다른 연산**이다. 셀 20은 두 후보 CSV와 `measured_public_auc=None`인 receipt를 남긴다. `REQUIRE_S6=True`면 S6 누락 시 네 모델이라는 이름으로 다른 조합을 내보내지 않는다.
- 우리가 배울 점은 후보/대조군을 같은 실행에서 만들고, 실제 사용된 branch·시간·실패 여부까지 기록하는 방식이다. 이미 동작하는 자동화의 검증 기록에 TTA 유효 여부를 추가하는 데 참고할 수 있다.

## 13. 0.957 Explained — leoprovorov, v12

[원본](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl)

- 저자는 10월 15일 public LB **0.957, 206위**와 기준 0.956을 보고한다. 설명된 조합은 Raptor384 0.325 + Raptor224 mirror 0.175 + R7 0.30 + S6 0.20이다.
- **이 v12 소스의 실행 셀은 HTML handbook 표시, 통계 계산, 그림 표시다.** 본문에 제출 코드가 포함됐다는 설명이 있지만 셀 22는 markdown 설명이며, 여기서 추론을 수행하거나 `submission.csv`를 생성하는 실행 셀은 확인되지 않았다. 실행 가능한 네 모델 추론 예시는 12번에서 따로 확인했다.
- Gold58에서 Raptor 계열을 선택했다는 평가 한계를 설명하는 부분은 유용하다. 반면 hidden public 표본수 약 640, 모델 오차의 동일상관 가정으로 구한 성능 상한 등은 **추정 모형**이다. 실제 test 크기·달성 가능한 상한으로 인용하지 않는다.
- 우리 노트에도 `실측 OOF / 저자 보고 LB / 아직 실행하지 않은 가설`을 분리한다. 외부 모델의 gold 점수가 높아도 그 gold가 checkpoint 선택에 쓰였다면 우리 OOF와 동일한 검증 조건이 아니다.

## 14. Raptor + ConvNeXt R7 — jenillangaliya, v1

[원본](https://www.kaggle.com/code/jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble)

- 제목/본문 보고 **0.956**. 셀 2의 CoAtNet Raptor SWA, 셀 4의 bundle `8c7b666e27dc0e8f`, 셀 6의 **Raptor 0.5 + R7 0.5 rank blend**가 실제 경로다.
- 독립적인 새 backbone 학습보다는 heliosli/nartaa의 공개 추론 조합과 설명을 제공한다. R7의 class별 spatial attention, fastMRI 보조 학습 등은 R7 manifest 및 원 training code를 함께 확인해야 한다. 이름에 OrthoFoundation이 있다고 최종 분류 backbone이 OFL Transformer인 것은 아니다. 이 경로는 ConvNeXt-small이다.
- Raptor 이름의 384는 원 렌더 크기이며 이 소스의 실제 입력은 320 center crop이다. 96개 stack에서 94개 triplet window를 사용한다. `5-fold`라는 잔존 주석과 달리 현재 `ARMS`는 SWA checkpoint 한 개다.
- 우리에게는 **Dense input + 학습된 backbone + 보완적인 두 모델**이라는 패키지 비교 대상이다. 이 결과만으로 CNN 대 Transformer 우열을 판정할 수 없다.

## 15. Dual SOTA 0.954 — jenillangaliya, v4

[원본](https://www.kaggle.com/code/jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble)

- 본문 보고 **0.954**. 셀 5의 구세대 compact bundle `209a32c8b50a6b1c`와 셀 7의 **Raptor 0.70 + compact S6 0.30** rank 결합을 확인했다. 14번 R7의 50:50과 다르다.
- 모델별 subprocess로 메모리를 반환하고 파일·체크포인트를 확인한다. 그러나 본문의 “OOM 위험 0”, “실패 없음”은 보장이 아니다. 실행 오류와 예측 실패는 여전히 발생할 수 있다.
- 세대와 결합 비율이 동시에 달라지므로 14번과의 0.002 차이를 R7 한 요소의 효과로 분리할 수 없다. 우리 실험은 같은 split·head·비율을 유지한 상태에서 한 요소씩 비교해야 한다.

## 16. Unused 224 checkpoint — varshithakolupuri, v1

[원본](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956)

- 저자 보고: 공개 표시 점수 **0.956 유지**, 같은 날짜 순위 233→156. 앞선 S6 조합에서는 0.954 표시를 유지하며 437→320이라고 설명한다. **순위 변화만으로 정확한 AUC 증가량이나 통계적 유의성을 확정하지 않는다.** 표시 동점 정렬 원인도 이 조사에서 별도 검증하지 않았다.
- 셀 9–12: 공개 가중치 묶음 안의 `c08_r224_swa.pt`를 추가로 실행한다. Raptor 부분만 384:224=0.65:0.35로 바꾼다. 전체 비율은 R7 0.5, Raptor384 0.325, Raptor224 0.175다.
- 224 branch는 시간 여유가 없거나 실패하면 생략된다. 이때 결과는 새 조합이 아니라 원래 두 모델 조합이다. 원본의 `R224_OK` 및 실행 로그 확인이 필수다.
- 우리 R01의 서로 다른 입력/head 결합과 가장 가까운 아이디어다. 우선 기존 OOF에서 확률 평균 대 rank 평균을 검증하고, 그다음 전처리·시야가 다른 학습 모델을 추가하는 순서가 경제적이다.

## 17. One free flip — varshithakolupuri, v1

[원본](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up)

- 저자 보고: **0.956 표시 유지**, 10월 14일 순위 357→289. 16번과 날짜가 다른 순위를 직접 효과량으로 비교하면 안 된다.
- 224 branch에 plain + mirror 확률 평균을 추가한다. `_mirror_tri`는 sagittal window에서 세 채널의 순서를 뒤집고, coronal/axial에서는 가로축을 뒤집는다. 그 후 rank를 계산한다. 따라서 모든 방향의 이미지를 똑같이 가로 반전하는 것과 다르다.
- 새 학습 없이 추론 pass를 늘린 사례다. 제목의 “free”는 추가 학습이 없다는 의미에 가깝고 **추론 비용은 증가**한다.
- 우리 적용 시 DICOM 방향 정규화와 각 token 좌표·plane metadata를 함께 변환해야 한다. 공개 recipe의 mirror 규칙을 현재 raw orientation에 그대로 붙이면 해부학적 좌우 반전이 아닐 수 있다. OOF에 같은 TTA를 적용해 먼저 평가한다.

## 18. CoAtNet + ConvNeXt MIL — hitarthjain0, v5

[원본](https://www.kaggle.com/code/hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950)

- 셀 12의 저자 결과표: 기본 **0.950**, 자체 EfficientNet-B3 10% 결합 **0.948**, 더 큰 자체 모델 비중 **0.944**, 자체 모델 단독 **0.902**. 새 모델의 다양성이 품질 부족을 반드시 보상하지 않는 실제 반례다.
- 셀 8 `KneeNet`: ConvNeXt의 이미지 vector → 384차원 → slot embedding + 위치 MLP → depth=2 TransformerEncoder → 12개 소견별 attention pooling. **사용자의 위치+방향+attention 구상과 유사한 구현이 이미 있다.** 이 구조가 0.950 전체 점수를 혼자 만든 것은 아니며 공개 CoAtNet과 섞는다.
- v5의 셀 9 실제 자체 reader TTA는 **256/320 두 크기, slice offset 0/+1, 가로 flip**이다. 본문의 384/512 표현과 구분해야 한다. 셀 10은 class별 V5 가중치와 작은 tie-break 항을 적용한다.
- 셀 11의 파일 모양 검사는 점수의 보장이 아니다. 특히 최신 V5의 목표 점수와 셀 12의 과거 0.950 실측 보고를 같은 것으로 보지 않는다. 모든 plane에 대한 가로 flip이 17번의 해부학적 mirror와 같은 것도 아니다.

## 19. Micro-boost tri-reader — rabari9999, v11

[원본](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader)

- 본문은 **0.9564+** 및 여러 단계의 CV/LB 상승표를 보고한다. 다만 해당 단계별 OOF 예측, split·checkpoint 선택 과정과 공식 제출 점수를 이 소스만으로 재검증하지 못했다. **저자 주장으로 보존하고 개선 효과의 증거로 확정하지 않는다.**
- 셀 15의 실제 코드는 Raptor384/224 + 3개 fold의 ConvNeXt reader를 class별로 섞는다. percentile→probit 변환, 온도 1.02, clipping, medial→lateral 12% 혼합, 다른 소견 score 기반 보정, 작은 tie-break 후 최종 rank 순서다.
- 수학적 검토: **양의 상수로 나누는 온도 조정만으로는 순위가 변하지 않아 AUC가 개선되지 않는다.** clipping은 동점을 만들 수 있다. 단조 변환 이전의 모델 결합이나 소견 간 혼합은 순위를 바꿀 수 있지만 개선은 보장되지 않는다. 따라서 “tail을 날카롭게 하면 AUC가 반드시 오른다”, “probit가 항상 낫다”는 설명을 채택하지 않는다.
- 우리에게는 우선순위가 낮다. 58 gold에 수십 개 class별 상수를 맞추면 개발 표본 과적합 위험이 크다. 임상적 인과 설명으로 가중치를 정당화하기보다 독립 검증 또는 nested 평가가 필요하다.

## 20. Multi-scale OrthoFoundation — yuvrajlamba01, v2

[원본](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956)

- 제목/본문 **>0.956**, 순위 156과 gold AUC 0.9512를 보고한다. 단계별 gold 표의 원 예측과 검증 설계는 이 소스에서 확인되지 않아 독립 CV로 취급하지 않는다.
- 셀 6은 R7 `8c7b666e…`, 셀 9는 224 crop, 셀 12는 `A224=.35`, Raptor:R7=.5:.5다. 실제 핵심은 **16번의 공개 3-branch 조합**이다. 새로 학습된 OrthoFoundation backbone의 별도 성공 사례로 집계하지 않는다.
- 최신 branch 수, fallback 여부, 실제 blend 계수는 다이어그램보다 코드로 확인해야 한다. 16번과 함께 한 계보로 묶고 증거를 두 배로 세지 않는다.

## 21. Dual-engine / triple-reader — rabari9999, v4

[원본](https://www.kaggle.com/code/rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951)

- 본문 보고 **0.951+**. 셀 3–5에서 Raptor224와384, 셀 7–9에서 2.5D reader를 정의하고 셀 10에서 결합한다.
- 실제 `W_C224=.3`, `W_OWN=0`이다. 후자는 ConvNeXt reader가 없다는 뜻이 아니라 **추가 own2 모델만 비활성**이라는 뜻이다. `cnxt_v0_fold*.pt` 세 개는 활성이다.
- CoAtNet 두 모델의 rank를 70:30으로 섞고 다시 rank, 이후 class별 ConvNeXt 비중(Baker's .35, Contusion .30 등)을 적용한다. 19번의 probit·cross-prior 방식과는 구분한다.
- 우리 코드와 닮은 study token Transformer 경로가 있지만, 공개 학습 가중치와 후처리를 함께 쓰는 구성이다. 우리 S11의 단일 head 변경 결과와 직접 대조하는 ablation은 아니다.

## 22. V15 tri-hybrid — yuvrajlamba01, v8

[원본](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954)

- 제목은 **0.954**, 본문에는 다른 높은 점수 표현도 있어 버전별 대응을 확정할 수 없다.
- 셀 5는 `BUNDLE_ID=209a32c8b50a6b1c`, 후보 dataset은 `compact-r6cnxfsss` / `compact-v1`이며 manifest ID를 검증한다. 따라서 dataset 이름에 r6가 있다고 최신 R6/R7와 동일하게 취급하면 안 된다.
- 셀 7의 실제 결합은 Raptor .5 + compact .5다. 224 추가 branch가 실행되지 않는다. **tri-hybrid라는 제목을 독립적인 세 backbone 앙상블의 근거로 쓰지 않는다.** 셀 8의 분포 그래프도 hidden 정답을 쓰는 성능 검증이 아니다.
- 얻을 교훈은 bundle hash와 활성 경로를 기록하는 것이다. 우리 자동화의 source/config/checkpoint pinning은 이미 이 문제를 다루고 있다.

## 23. 학습 원형: Pilkwang baseline — v15

[원본](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)

이 항목은 ≥0.95 독립 성공 건수에 넣지 않은 **학습·전처리 대조군**이다. 셀 13·19·21·25–33에서 확인한 핵심은 DINOv2, 130 mm crop, 좌우 정규화, protocol slot, class별 slot attention, backbone 마지막 6 block 적응이다. 기본 head LR=1e-3, backbone LR=8e-6, 10 epochs이며 feature를 영구 고정한 head 학습과 다르다. 3장은 spread된 선택 위치를 채널에 넣는 방식이므로 우리 adjacent triplet과도 구분한다.

공개 weight package가 붙으면 셀 30에서 **여러 사전학습 member 추론** 경로가 실행된다. 기본 학습 분기는 그 모든 배포 가중치의 재학습을 보증하지 않는다. label의 confidence floor와 unknown 처리도 우리의 UNK zero mask와 다르다. 검증 split·teacher 선택 과정까지 동일하다고 가정하지 않는다.

## 24. 학습 원형: Training the twelve-finding model — dreaddevelopment, v8

[원본](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)

이 역시 ≥0.95 건수에 넣지 않았다. 소스 머리말은 **0.924 추론 모델의 학습 원형**이라고 설명한다. `StudyWindows`, `RaptorClassifier`, `main`을 확인했다.

- 44×336×336 uint8 corpus에서 adjacent triplet을 뽑아 학습 때 random K, 검증 때 균등 K를 쓴다. 기본 CLI는 ViT-small/224/12epochs이고, CoAtNet384/16epochs는 제시된 실행 예시다. 예시를 기본값으로 잘못 적지 않는다.
- 이미지 backbone과 class별 attention-MIL을 함께 학습한다. AdamW backbone LR3e-5/head1e-3, OneCycleLR, BCEWithLogitsLoss(pos_weight), gradient clipping. ImageNet 초기화가 현재 기본이며 이름에 Raptor가 있다고 비공개 SSL 가중치가 반드시 필요한 것은 아니다.
- 58 gold는 gradient 학습에서 제외하지만 **best epoch·top3 checkpoint 선택에 사용**한다. top3 weight average와 gold 성능도 출력한다. 그래서 이 gold 점수는 최종 독립 평가가 아니다.
- report soft label 파일은 제공하지 않으며 사용자가 준비해야 한다. 없으면 정상 exit 0으로 끝난다. **프로세스 성공 = 학습 완료**가 아니라는 사례다.
- 우리에게 가장 직접적인 후속 방향은 고정 feature 한계 확인 후 제한된 backbone 해제, 적절한 학습률·warmup, 검증 기준을 고정한 weight averaging이다. 이 원형을 복사한다고 nartaa의 OAI 추가학습 최신 .949 가중치가 자동 재현되는 것은 아니다.

## 25. Speedy Raptors — mattiaangeli, v39

[원본](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original)

주요 역할은 **학습/추론 설계 대조군**이며 이 v39의 ≥0.95 새 점수를 확인한 것으로 집계하지 않는다. 이전 버전 분석 문서와 달라져 현재 활성 경로를 다시 확인했다.

- 셀 13·15의 DINOv2 20 member, 이후 A5/Rad 및 Raptor 경로, 셀 22의 추가 CoAt family가 누적되는 큰 추론 graph다. 셀 23은 DINO fingerprint 20개와 calibrator 적용을 확인한 뒤 마지막 CSV를 쓴다.
- 셀 22에는 public Raptor의 여러 resolution/span branch 외에 residual gated spatial head, Global96, D4 depth-zone, Repair-v1 child가 들어 있다. 성공한 member의 **확률 평균을 rank**하는 경로와, 확률 파일이 없을 때 기존 rank 혼합 경로가 구분돼 있다. child 실패 시 family가 달라질 수 있다.
- D4의 depth 구획, class별 spatial pooling과 보호된 global residual은 사용자가 제기한 **“이미지 feature가 global pooling에서 너무 사라지는가?”**에 직접 대응하는 아이디어다. 현재 ours의 cross-attention은 이미지별 pooled vector 다음에 있으므로 patch 내부 정보를 되살릴 수 없다.
- 다만 복잡한 전체 graph의 효과를 특정 head 하나의 개선으로 해석하면 안 된다. 우리는 patch pooling만 바꾼 단일 대조 실험을 먼저 하고, 성공한 경우에 depth-zone과 앙상블을 붙이는 편이 해석 가능하다. 속도 개선도 새 정확도 증거와 별개다.

## 검토 증거 보존

25개 소스의 버전·표 수·SHA256·셀 개수는 `outputs/research/kaggle_code_20261019/snapshots.json`, 각 원문과 cell index는 같은 디렉터리의 저자별 폴더에 있다. R7 설정은 `manifests/r7/manifest.json`을 별도 확보했다. 내려받은 타인 코드는 실행하지 않았다. 이 문서는 소스 수준 분석이며, 타인의 hidden-test 제출 점수나 공개 가중치의 전체 학습 과정을 재현한 결과는 아니다.
