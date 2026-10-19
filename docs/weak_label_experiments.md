# 보고서 약라벨을 이용한 다음 실험

2026-10-16. 사용자가 승인한 순서: 기존 E03~E05 완료 → Swin을 유지한 약라벨 확대 → 2.5D/촬영 조건/소견별 pooling 분리 → 부분 fine-tuning → 앙상블.

현재 코드의 train/submit Notebook 생성과 shell 큐를 확장한다. 기존 실험과 출력은 보존한다. 새 학습 조건 자체의 영향을 확인하기 위해 gold 대조군을 함께 실행한다.

| ID | 이전 단계와 달라지는 점 |
| --- | --- |
| E06 gold control | 기존 Swin/중앙 5장/3방향/동일 회색조 3채널/평균 pooling, 새 mini-batch 학습 조건의 gold 대조군 |
| E07 weak labels | E06 + 약라벨 4,292건 |
| E08 triplet | E07 + 같은 series 안의 인접 3장을 3채널로 입력 |
| E09 protocols | E08 + 방향 3종 × 지방 억제 여부 2종의 6개 슬롯; 없는 슬롯은 mask |
| E10 class MIL | E09 + 각 슬롯 안에서 12개 소견마다 다른 attention pooling |
| E11 fine-tune | E10 + gold와 겹치지 않는 약라벨만으로 Swin 마지막 stage 미세조정 후 동일 head CV |
| E12 ensemble | E10/E11의 소견별 예측을 고정 50:50 확률 평균; gold로 혼합 비율을 탐색하지 않음 |

## 데이터와 loss

[Pilkwang 공개 라벨](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) v1(CC0)의 `report_labels_v2.csv`를 사용한다. SHA256은 `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920`. 실제 파일의 4,406개 ID가 대회 train과 대응한다. 누락 1건은 gold라 실제 정답으로 대체된다.

YES/NO의 soft target과 confidence를 사용한다. **UNK는 가중치 0**이다. 이는 UNK에도 작은 가중치를 주는 공개 baseline과 다른, 사전에 고정한 정책이다. 사용 가능한 약라벨이 하나도 없는 검사와 gold의 정규화된 보고서와 같은 검사는 제외한다. 최종 입력은 gold 58 + weak 4,292 = **4,350건**이다. E06은 gold 58건만 사용한다.

Loss는 원소별 BCEWithLogitsLoss × 신뢰도 가중치를 유효 가중치 합으로 나눈다. gold weight=3, 알려진 weak weight=`0.25+0.75*confidence`. 새 실험은 `pos_weight`를 추가하지 않는다. E06 대조군으로 기존 loss/학습 예산 변경과 데이터 추가 효과를 구분한다. 공개 soft score는 보정된 실제 확률이 아니다.

## 고정 평가와 학습량

gold의 Study ID 정렬, 기존 3-fold와 seed 42/2026/3407을 유지한다. fold마다 gold train + 모든 적격 weak로 학습하고 held-out gold만 평가한다. 최종 모델은 gold 전체 + 적격 weak로 재학습한다. 정규화 통계는 해당 fold의 train만 사용한다. gold 정답은 약라벨보다 우선한다.

Head 학습은 20 epochs, batch 64, AdamW lr=0.001, weight decay=0.01로 고정한다. gold 검증으로 epoch를 고르지 않는다. epoch별 train loss와 마지막 gold OOF를 기록한다. 같은 58건을 계속 비교하므로 개발용 CV이며 독립적인 최종 성능 추정이 아니다. public teacher의 생성 과정까지 gold와 독립이라고 보증하지 않는다.

E11에서는 gold 및 같은 보고서의 weak를 backbone 학습에서 제외한다. 남은 weak의 보고서 그룹 hash로 약 10%를 backbone 검증에 사용한다. 첫 비교는 Kaggle 9시간 한도 안에서 처리량을 확인하기 위해 **1 epoch**로 시작한다. 유효 study batch=8이므로 수백 번 업데이트하며, 기존 gold head의 1 epoch=1회 업데이트와 다르다. epoch를 늘리는 후속 실행에서는 weak validation BCE가 가장 낮은 encoder를 선택한다. 마지막 Swin stage와 최종 LayerNorm만 학습하며 backbone lr=1e-5/head lr=1e-4이다. 그 encoder를 동결한 후 E10과 동일한 head CV를 수행한다. gold로 학습한 E10 head를 backbone 학습에 재사용하지 않는다.

## 입력과 재현

모든 단계에서 Swin-T, 224 px, 중앙 25~75%의 5개 중심 위치를 유지한다. 물리 crop·더 넓은 범위·새 positional encoding을 동시에 추가하지 않는다. E08 이후 세 채널은 같은 series의 직전/현재/직후 슬라이스다. 경계에서는 가장자리 슬라이스를 반복한다. E09의 슬롯에는 다른 촬영 조건을 복제해 채우지 않는다.

학습·제출 코드, label CSV hash, encoder/head hash, 데이터 분할, label audit, 실제 Kaggle 학습 버전을 기록한다. 제출은 Internet OFF이며 보고서나 label Dataset을 요구하지 않는다. E12는 각 멤버의 전처리와 bundle을 보존한다.

## 실행 및 확인

`scripts/run_weak_experiments.ps1`이 E06~E12를 순서대로 실행한다. 첫 단계가 실패하면 뒤 단계는 시작하지 않는다. 결과는 `outputs/automation/weak-labels/`에 저장한다. 기존 `run_experiments.ps1`의 기본 E03~E05 큐는 유지한다. 공식 리더보드 제출은 기존과 같이 수동이다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_weak_experiments.ps1
```

이미 실행 중이면 두 번째 실행은 lock으로 차단된다. 중단 후에는 같은 명령으로 재개한다. 새 실행은 `-RunName weak-labels-trial2`, 업로드 없는 확인은 `-DryRun`을 사용한다. 새 실행 이름은 E06부터 다시 학습하며 이전 결과 폴더를 덮어쓰지 않는다. 실행 중인 큐의 소스 코드는 변경하지 않는다.

E07은 E06의 동일한 영상 특징을 일부 재사용하고, E10은 E09의 전체 특징을 재사용한다. E08/E09는 입력이 달라 다시 추출하며 E11은 학습된 encoder로 다시 추출한다. 각 캐시는 Study ID, 입력 설정, encoder hash, cache 파일 hash가 맞아야 사용한다.

`runner.log`는 진행 상황, `summary.csv`는 완료 단계의 AUC/BCE, `state.json`은 재개와 정확한 버전 연결 정보다. 각 실험의 `train/run/label_audit.json`에서 사용한 label 수를, `history.csv`에서 실제 optimizer step 수를 확인할 수 있다. fine-tuning은 `backbone_audit.json`과 `backbone_history.csv`를 추가로 남긴다.

검증 범위는 라벨 결측/범위/중복, UNK의 zero gradient, gold와 중복 보고서 제외, 슬롯/인접 슬라이스 순서, padding mask, train-only 통계, 모델 export/reload, standalone Notebook 및 큐 버전 연결이다. 큰 가중치와 특징은 Kaggle 출력에 보관한다.

대규모 실험은 이전 58건 학습보다 오래 걸린다. 학습/제출 각 Notebook의 9시간 실행 한도와 계정 GPU 할당량은 실제 실행에서 확인한다. 부분 fine-tuning이 시간 한도에 걸리면 오류를 기록하고 멈추며, 미완성 모델로 성공 처리하지 않는다.

## 최초 실행 확인

2026-10-16 16:52 KST에 자동 큐를 시작했다. 로컬 33개 테스트와 실제 라벨 4,406행의 형식·ID·hash·학습 대상 수를 검증했다.

- E06 train v1, submit v1 완료 및 CSV 검증 통과. OOF AUC **0.585972 ± 0.000882**, BCE 0.753050. 새로운 loss·학습 예산의 gold 대조군 결과로, epoch 하나만 바꾼 실험은 아니다.
- E06의 3-fold × 3-seed × 20-epoch 기록 180행과 기존 gold 분할의 일치를 확인했다. Tesla T4에서 870개 영상의 특징 추출은 59.95초였다.
- 16:57 KST에 E07 train v1이 RUNNING으로 전환됐다. 후속 단계의 최신 상태는 [runner.log](../outputs/automation/weak-labels/runner.log), 완료된 결과는 [summary.csv](../outputs/automation/weak-labels/summary.csv)에서 확인한다. 이 문서는 최초 확인 시점의 기록이며 이후 자동 갱신되지는 않는다.

자동 실행기는 이 PC의 숨겨진 PowerShell 프로세스에서 동작한다. 다음 단계 업로드와 결과 회수가 이어지려면 PC를 켜 두어야 한다. 현재 Kaggle 작업은 PC가 꺼져도 계속되지만 뒤 단계는 같은 명령으로 재개해야 한다.
