# RSNA Knee Discussion: 고정 글·추천 상위 글 정리

조사일: **2026-10-16**. [대회 Discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion)의 Top·Hot 목록을 조회하고, 고정 글 5개와 학습·검증·규칙에 관련된 상위 글의 본문 및 주요 댓글을 확인했다. 전체 198개 게시물을 전수 조사한 것은 아니다. 투표 수는 조회 시점 값이며, 개인 경험·만화·단순 점수 공유 글보다 현재 실험에 활용할 수 있는 내용을 우선했다.

**지금 가장 중요한 내용은 보고서 약라벨의 불확실성, 실제 모델 입력의 품질, 작은 gold 검증셋의 한계다.** 좋은 backbone을 찾는 것만으로 해결되는 문제라는 합의는 없다. 아래에서 운영진의 확정 설명, 참가자의 실험 보고, 우리 프로젝트에 대한 제안을 구분한다.

## 1. 고정 게시물

| 투표 | 원문 | 핵심 |
| --- | --- | --- |
| 101 | [Challenge Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) | **HOST 공식 설명.** 12개 소견의 영상 판정 기준, MRI 방향·시퀀스, 양성/음성 예시. 약라벨이 무엇을 예측해야 하는지 확인할 첫 자료다. |
| 26 | [Use of Commercially Hosted LLMs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) | **HOST 공식 허용.** 접근성·비용·규칙·서비스 약관 조건을 충족하면 상용 LLM/API로 보고서에서 학습 라벨을 생성할 수 있다. |
| 19 | [Welcome](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733375) | MRI와 다국어 보고서로부터 학습하는 대회 목적, 효율성 상 소개. 구체적인 모델 처방은 없다. |
| 11 | [Getting started / Official Discord](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/730709) | **ADMIN 안내.** 공개 협업 채널 안내. 중요한 운영진 질문은 Kaggle 포럼에 남기며, 팀 밖 비공개 코드·데이터 공유와 공개 공유를 구분해야 한다. |
| 3 | [AI 활용 설문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749038) | 익명·10분 미만 설문. 순위·수상 자격에 영향 없으며 학습 방법 안내는 아니다. |

Overview의 핵심은 **이름이 비슷한 모든 경미한 소견을 양성으로 처리하지 않는다는 것**이다. 예를 들어 중등도 이상 Effusion과 경미한 액체 소견은 같은 라벨 정의가 아니다. 영상 판독의 경계성 음성 규칙을 보고서의 ‘미언급=음성’으로 확대해서도 안 된다. 자세한 공식 기준과 댓글은 [규칙 조사 노트](kaggle_discussion_rules.md)에 정리했다.

## 2. 추천이 많은 글

| 투표 | 게시물 | 읽을 이유 |
| --- | --- | --- |
| 114 | [Public notebooks 문제 제기](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181) | 높은 공개 점수·추천 수와 독창적인 학습 코드의 가치는 다를 수 있다. ADMIN 답변도 있다. |
| 97 | [외부 knee MRI 허용 범위](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416) | OAI·fastMRI 등의 접근성 논쟁. 10월 14일 HOST 답변이 중요하다. |
| 81 | [Best single-model score](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) | 작은 CNN부터 DINOv2까지 다양한 성공 보고. ‘single model’이 실제로는 5-fold 앙상블인 경우도 확인된다. |
| 62 | [“Not addressed” is a label too](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) | 보고서가 소견을 언급하지 않는 경우를 어떻게 학습할지 분석한다. |
| 41 | [One CoAtNet, public AUC 0.949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) | 약라벨, 2.5D, 소견별 attention, crop, 방향을 고려한 증강과 실패 실험을 공개한다. 외부 OAI 사용을 포함한다. |
| 35 | [Tricks for the competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861) | ROI 라벨링, 불확실한 사례 재검토, LLM 라벨 합의 등 아이디어와 시행착오. 검증된 단일 레시피는 아니다. |
| 30 | [보고서와 제공 라벨의 불일치](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826) | **HOST가 정답의 출처와 불일치 이유를 직접 설명한다.** 가장 먼저 읽을 댓글 중 하나다. |
| 21 | [LLM label-extraction benchmark](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658) | 보고서 라벨 추출 비교. 영상 LLM 실험은 일부 미완료 사례가 있어 결과 해석에 주의한다. |
| 20 | [Plateau at 0.903](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610) | 복잡한 파이프라인의 정체와 작은 모델·단일 변경 실험에 대한 구체적인 조언. 작성자는 후속 댓글에서 0.935를 보고했다. |

### A. 보고서와 정답이 다른 것은 대회의 핵심 난점이다

30표 글에서 **HOST Chen**은 다음을 명확히 답했다.

- 정답은 보고서에서 추출하지 않고 MRI를 독립적으로 판독하여 만들었다.
- 보고서와 영상 정답이 충돌하면 **영상 정답이 기준**이다.
- 제공된 정답의 0은 판독 기준에 따라 ‘없음’으로 표기한 값이다.
- 임상 보고서와 다중 판독자의 엄격한 영상 판정 사이에 차이가 생기는 것은 예상된 현상이다.
- 양쪽 무릎이 포함된 검사는 개별 검토하고 보고서 또는 DICOM 메타데이터를 조정하여 대상을 구분할 정보를 제공했다고 설명했다. 임의의 양측 OR 집계 규칙을 제시한 것은 아니다.

근거: [공식 답변 3510434](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826#3510434).

우리에게 의미하는 것은 **gold의 0과 약라벨의 UNK를 구분해야 한다는 것**이다. 보고서에 없다는 이유로 영상에서도 없다고 단정하면 잘못된 음성 감독을 줄 수 있다. 목표는 보고서 문장을 복제하는 것이 아니라 MRI에서 12개 소견을 예측하는 것이다.

### B. 약라벨: 미언급을 모두 같은 값으로 처리하지 말자는 제안

62표 글의 작성자는 자신의 추출 결과에서 전체 라벨 셀의 25.4%, Synovitis의 83.7%가 미언급이었다고 보고했다. ACL·반월상연골과 Synovitis는 보고서에서 다뤄지는 빈도가 크게 달랐다.

이 글의 숫자는 **라벨 추출값을 gold 58건과 비교한 AUC**다. MRI 모델의 리더보드 점수가 아니다. 작성자는 Synovitis의 미언급 셀만 Effusion 정보로 보완했을 때 라벨 macro AUC가 0.8780→0.8873으로 개선됐지만, 12개 소견 전체를 일괄 보완하면 0.8805로 낮아졌다고 보고한다. [원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)

**프로젝트 해석:** 현재 Pilkwang 라벨의 `UNK weight=0`은 명확한 비교 출발점이다. 이를 최적 정책이라고 단정하지 않는다. 후속 실험에서는 소견별 결측률과 유효 감독 수를 보고 mask·낮은 가중치·사전확률 보완을 각각 비교할 수 있다. Steven 라벨의 미언급 값 0.5와 우리가 사용하는 Pilkwang 라벨의 UNK score 0.28은 서로 다른 데이터셋의 인코딩이다.

라벨 AUC가 영상 모델 성능의 엄격한 상한이라는 주장은 채택하지 않는다. [단일 모델 토론의 참가자](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3518462)는 라벨 오류를 고쳐도 LB가 개선되지 않았고 영상 student가 label teacher보다 높은 평가를 받았다고 보고했다. 라벨 품질 개선과 모델 일반화 개선은 별도로 측정해야 한다.

### C. 거대한 Transformer나 대형 앙상블이 필수라는 근거는 없다

81표 글에는 DINOv2 단일 모델 0.915, 작은 ResNet 단일 fold 0.936, 다른 참가자의 단일 fold 0.949 등이 보고되어 있다. 모두 **참가자 자기 보고이며 재현 검증하지 않았다**. 0.943을 ‘single model’로 소개한 댓글은 뒤에서 **5-fold 앙상블**이라고 설명한다. 구조가 하나라는 뜻과 체크포인트가 하나라는 뜻을 구분해야 한다. [토론 원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)

20표 정체 사례의 주요 조언은 작은 CNN과 단순 pooling으로 출발하고, 라벨·입력·분할을 고정한 채 증강, 해상도, backbone을 차례로 비교하라는 것이다. 작성자는 후속에 0.935까지 개선했다고 답했으나 어떤 변경의 효과인지는 분리하지 않았다. [조언](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610#3523148), [후속 결과](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610#3531569)

같은 글에서 지적한 유용한 구분도 있다. **DINO는 사전학습 방법이며 DINOv2-ViT와 ImageNet-ResNet 비교는 구조·사전학습 목적·데이터가 함께 바뀐다.** 그 결과만으로 Transformer의 우월성을 결론낼 수 없다. [해당 댓글](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/740610#3523680)

### D. 0.949 CoAtNet에서 가져올 수 있는 것과 분리해야 할 것

작성자의 핵심 구성은 **3장 window → CoAtNet → 소견별 MIL attention → 12개 예측**이다. 4,349건의 보고서 약라벨과 외부 OAI 2,399건을 사용했으며, OAI에서 없는 소견은 음성으로 채우지 않고 mask했다. 학습은 16 epochs, 검사당 12 windows였고 최종 추론은 94 windows를 사용했다.

최종 0.949는 선택한 3개 epoch의 가중치 평균으로 만든 하나의 체크포인트에 원본·해부학적 좌우 반전 두 번을 적용한 값이다. sagittal은 슬라이스 채널 순서를 뒤집고 coronal/axial은 가로를 반전했다. 무조건 모든 방향에서 가로 flip하는 것과 다르다.

중요한 한계는 외부 데이터, 해상도, crop, 업데이트 횟수 등이 동시에 달라졌다는 것이다. gold 58건도 모델 선택에 반복 사용했다. 그러므로 전체 개선을 OAI나 CoAtNet 하나의 효과라고 볼 수 없다. **우리 실험에 참고할 것은 입력·mask·pooling·증강의 설계이며, 점수 자체가 재현 보장은 아니다.** [원문과 실패 실험 표](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

10월 14일의 [0.950 ConvNeXt 후속 글](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)(4표)도 확인했다. anatomy crop용 별도 체크포인트와 실패 시 CoAtNet fallback을 사용하므로, 제목만 보고 순수 ConvNeXt 한 모델의 효과로 비교하면 안 된다.

### E. 검증: 세 가지 AUC를 섞으면 안 된다

| 지표 | 비교하는 대상 | 한계 |
| --- | --- | --- |
| 라벨 추출 AUC | 보고서에서 만든 soft label ↔ gold 정답 | 보고서와 영상의 차이가 포함된다. 영상 모델 성능이 아니다. |
| 영상 모델 local CV | held-out MRI 예측 ↔ gold 또는 고정 약라벨 | gold 58건은 작고, 약라벨 검증은 추출기의 오류를 공유한다. 무엇을 정답으로 썼는지 함께 기록한다. |
| Public LB | 제출 모델의 숨겨진 공개 평가 부분 예측 ↔ 대회 정답 | 여러 번 보고 선택하면 공개 평가셋에도 과적합할 수 있다. private 성능을 보장하지 않는다. |

[Local CV or Public LB](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011)(12표)의 작성자는 자기 제출 15개의 상관계수를 gold CV와 LB에서 0.66, 약라벨 CV와 LB에서 0.87로 보고했다. 이는 해당 파이프라인의 관찰이며 약라벨 검증이 항상 우월하다는 증명은 아니다. 댓글의 노이즈 추정이나 작은 점수 차이의 임계값도 보편적인 기준으로 가져오지 않는다.

**프로젝트 제안:** 현재 고정 gold OOF 비교는 유지하되, 다음 실험군에서는 보고서 그룹을 분리한 고정 weak validation도 함께 기록한다. seed 반복만으로 58건 표본의 불확실성이 사라지는 것은 아니다. 학습에 사용한 약라벨을 고쳐 놓고 같은 라벨에 대한 평가 상승만으로 성능 향상을 주장하지 않는다.

### F. 고득점 공개 Notebook이 많아도 쉬운 대회라는 뜻은 아니다

114표 글은 공개 가중치의 유사한 앙상블·복제 Notebook이 높은 추천을 얻는 문제를 제기한다. **ADMIN Ryan Holbrook**도 공개 점수를 Notebook 품질의 기준으로 삼는 문제와 과적합 가능성을 인정하고, 출처를 지운 복제물 신고 및 마감 전 1주 공개 Notebook 잠금을 설명했다. [공식 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181#3537433)

이는 많은 참가자가 높은 점수에 접근할 수 있는 배경을 설명한다. 다만 실제 복제 비율이나 private 순위 하락을 확정한 통계는 아니다. 원본 학습 레시피와 단순 재조합을 구분해서 참고하는 것이 좋다.

### G. 외부 데이터는 최신 공식 답변까지 읽어야 한다

97표 글의 10월 14일 **HOST Chris Carr** 답변은 단순 등록·약관 동의와 기관 소속·연구계약 요구를 구분한다. 전자는 접근성 기준을 충족하지만 후자는 충족하지 않는다는 설명이며, 사용 데이터셋을 포함한 학습 방법 공개를 권고한다. 개별 데이터셋의 일괄 허용 목록을 제시한 것은 아니다. [공식 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3538535)

KneeCoT 불허, OAI 조건부 설명, 지역 제한과 파생 가중치에 대한 미해결 질문은 [규칙 조사 노트](kaggle_discussion_rules.md)에 시간순으로 정리했다. 높은 점수의 공개 가중치라는 이유만으로 학습 출처까지 승인됐다고 해석하지 않는다.

## 3. 현재 실험에 적용할 우선순위

아래는 Discussion을 바탕으로 한 제안이며, 이번 조사에서 실행 큐나 학습 코드를 변경한 것은 아니다. 현재 설정은 [약라벨 실험 계획](weak_label_experiments.md)을 기준으로 대조했다.

1. **현재 E06→E12 비교를 먼저 완료한다.** E07 약라벨, E08 인접 3장, E09 촬영 조건, E10 소견별 attention, E11 부분 fine-tuning, E12 앙상블의 목적은 토론의 주요 가설과 맞는다. 어떤 단계도 개선을 미리 보장하지 않는다.
2. **실제 입력을 확인한다.** 방향·지방 억제 슬롯·인접 순서가 맞는지, 병변 부위가 crop 또는 중앙 구간 선택으로 빠지지 않는지 본다. 이후 slice 범위·개수와 물리 crop을 각각 분리해 비교한다.
3. **약라벨을 소견별로 진단한다.** UNK 비율·유효 학습 셀·confidence·소견별 AUC를 함께 본다. Synovitis처럼 감독이 희박한 소견은 전체 평균만으로 판단하지 않는다.
4. **검증을 보완한 뒤 학습량을 늘린다.** 현재 E11의 1 epoch는 첫 실행 예산이지 충분한 학습량이라는 결론이 아니다. loss와 weak validation 추세로 추가 epoch를 판단한다. 58건 gold에서 잘 나온 조합을 계속 선택하는 방식은 피한다.
5. **다음 backbone 비교는 작은 CNN도 포함한다.** 동일 입력·라벨·분할·학습 조건으로 ResNet/EfficientNet/ConvNeXt와 현재 Swin을 비교한다. 외부 의료 MRI 가중치와 LLM 재라벨링을 동시에 추가하지 않아야 원인을 해석할 수 있다.

가장 먼저 읽을 순서는 **공식 Overview → 보고서/정답 불일치의 HOST 답변 → 미언급 라벨 분석 → 0.903 정체 글의 조언 → 0.949 CoAtNet 레시피**다. 현재 필요한 것은 새 backbone 후보를 더 많이 모으는 것보다 약라벨을 사용한 단일 모델을 제대로 평가하는 것이다.

공개 모델의 점수 재현, 외부 제공자 약관 전체, 이미지 첨부물 속 모든 수치는 검증하지 않았다. Discussion의 모델 점수는 작성자 보고이며, 공개 가중치 도입 전에는 그 학습 데이터의 출처와 허용 조건을 별도로 확인해야 한다.
