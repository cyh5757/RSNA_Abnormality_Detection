# 실험별 학습·제출 Notebook 설계

사용자 의도: 각 실험을 독립 실행하고, 학습이 끝나면 **그 실험 모델만** 사용하는 완전한 Kaggle 제출 Notebook을 선택할 수 있어야 한다. backbone 후보도 순서대로 실제 학습·검증한다.

## 결과물

각 실험 폴더에 `train/train.ipynb`, `train/kernel-metadata.json`, `submit/submit.ipynb`, `submit/kernel-metadata.json`을 둔다. Notebook에는 실행에 필요한 소스가 전부 들어가며 로컬 Python 모듈을 업로드할 필요가 없다. 한 Notebook은 하나의 고정된 실험만 실행한다.

학습 Notebook은 3-fold × 3 seeds 평가와 전체 gold 58건으로 최종 seed 42 재학습을 수행한다. 최종 모델·전처리 설정·오프라인 DICOM 패키지를 `bundle/`로 내보낸다. 제출 Notebook은 해당 학습의 완료 버전을 연결하고 인터넷 없이 현재 `test.csv` 전체를 예측해 `/kaggle/working/submission.csv`를 만든다. 실제 대회 제출 버튼/CLI 제출은 사용자가 선택할 수 있게 남겨 둔다.

## 실험 순서

| ID | 구성 |
| --- | --- |
| E00 | ResNet18, 방향별 3장, 평균 통합 |
| E01 | ResNet18, 방향별 5장, 평균 통합 |
| E02a | DINOv2 ViT-S/14, 5장, 평균 통합 |
| E02b | Swin-T, 5장, 평균 통합 |
| E02c | ConvNeXt-Tiny, 5장, 평균 통합 |
| E02d | ViT-B/16, 5장, 평균 통합 |
| E03 | backbone 비교에서 선택한 기준 + 작은 attention, 위치 정보 없음 |
| E04 | E03 + 시리즈 내부 상대 위치 |
| E05 | E04 + 방향 임베딩 |

E03~E05는 동일한 선택 backbone을 사용한다. 모델 선택은 새 비교의 seed 평균 OOF AUC를 우선 보고 fold/소견별 결과를 함께 기록한다. 이는 독립 테스트에 대한 보증이 아니다. DINOv3는 접근·라이선스가 선결되지 않았고, 판독문 약한 정답 단계는 정답 생성·품질 검토 설계가 별도로 필요하므로 이번 구현 목록에 임의로 끼우지 않는다.

## 구현 방식

여러 Notebook을 수동 복사하는 대신 프로젝트 안의 공통 소스와 고정 실험 설정으로 완전한 Notebook을 생성한다. 생성된 제출 Notebook에는 학습·온라인 가중치 다운로드 코드가 없다. 각 제출의 실험 ID·설정과 모델 번들의 ID·설정을 대조하여 다른 실험의 가중치를 잘못 붙이면 즉시 중단한다.

backbone은 학습 때 고정하고 입력 크기를 고정한 TorchScript로 내보낸다. 추적 전후 실제 모델 출력을 batch 1과 선택 장수에서 대조하고, **내보낸 encoder로** 학습 특징과 추론 특징을 추출한다. DINO 소스는 고정 commit을 사용하고 xFormers 의존성을 사용하지 않는다. 제출에는 외부 DINO 소스/인터넷이 필요 없다.

기존 DICOM 시리즈 선택·강도 처리·25–75% 구간을 유지한다. 토큰 캐시는 `[N,3,S,D]`, 유효 슬라이스 mask, 상대 위치를 포함한다. 평균 모델은 방향별 평균 후 concat, attention 모델은 방향별 동일한 작은 Transformer 후 평균·concat한다. 결측 방향·짧은 시리즈를 mask 처리한다.

## 비교와 안전한 출력

- 기존과 같은 ID 정렬, `KFold(3, shuffle=True, random_state=42)`, 초기화 seed 42/2026/3407, 5 epochs, weighted BCE, AdamW lr .001 / weight_decay .1.
- fold별 학습 데이터만으로 정규화와 양성 가중치를 계산한다. 전체 gold 재학습 결과는 검증 점수로 보고하지 않는다.
- encoder별 필수 resize/crop/정규화와 특징 차원은 번들에 기록하고 학습·추론에서 공유한다.
- 평가 산출물: fold manifest, OOF 예측, epoch history, macro/소견별/fold별 AUC, BCE, 시간·메모리, 환경·소스 해시.
- 학습/제출 Docker를 기존 Kaggle 이미지로 고정하고 wheel·모델 해시를 검증한다.
- 기존 baseline·제출 Notebook은 유지하고 새 비공개 Kaggle Notebook을 생성한다. 토큰을 Notebook에 넣지 않는다.

## 완료 기준

독립 Notebook 생성, 의미 있는 CPU 테스트, backbone 실험의 순차 Kaggle 학습 완료, 해당 모델별 인터넷 OFF 제출 실행 및 올바른 CSV 확인, 결과·사용법 기록. attention 단계는 완전한 코드와 입력/출력·순서·mask 검증을 갖추고, 실제 실행 여부를 결과 문서에 구분한다.
