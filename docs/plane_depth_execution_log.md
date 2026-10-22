# 방향 × slice 위치 실험 실행 기록

## 현재 실행 — 2026-10-20 17:26 KST

사용자 요청: **남은 시간으로 실행할 수 있는 실험은 진행하고, 시간이 부족한 실험은 제외한다.**
이 요청에 따라 아래 10월 19일의 R → T 대기 순서를 변경했다.

| 실행 순서 | 실험 | 비교할 내용 |
|---|---|---|
| 1 | T00 `t00_position_control` | S11의 방향 embedding + mm RoPE + 소견별 cross-attention 재현 |
| 2 | T01 `t01_position_sum` | T00에 절대 깊이 sin/cos 신호 추가 |
| 3 | T02 `t02_position_joint` | T01에 방향×깊이 상호작용 추가 |

- 시작 직전 Kaggle API 확인: **GPU 잔여 4.481785시간, 예약 0시간**.
- 세 실험의 실제 학습 의존성은 모두 **완료·검증된 S11 train v1**이다. R02 이후 실험은 데이터 의존성이 아니라 이전에 지정한 실행 순서였다. S11 산출물·버전·해시 검증은 그대로 적용한다.
- 같은 특징 캐시를 쓰는 기존 S11 학습 notebook은 약 12분 46초, R04/R05는 약 12분에 완료됐다. 이는 이번 실행 시간의 참고치이며 완료 시간을 보장하지 않는다.
- 학습 업로드 직전 **1.25시간**(캐시 학습 0.75 + 제출용 추론 0.5), 제출용 notebook 업로드 직전 **0.5시간**의 여유를 확인한다. 이는 시작 허용 기준이며 강제 실행 제한 시간은 아니다.
- 부족하면 `deferred_for_quota`를 기록하고 새 업로드 없이 종료한다. quota 갱신까지 자동 대기하지 않는다. 같은 명령으로 재개하면 저장된 버전을 이어 사용한다.
- 기존 R·T·U 대기 실행기 3개(각 launcher/child 포함 6개 프로세스)를 정확히 식별해 종료한 뒤 T 실행기 하나를 시작했다. 원격 학습을 취소하거나 완료 산출물을 삭제하지 않았다.
- **R02(6h), R03(7h), R06(11h), 조건부 R07, U 큐는 보류**했다. U00의 시작 요구량은 7h이며, 비용이 작은 후반 U 실험도 앞 단계 결과가 필요하다. R/U는 quota 갱신 시 자동 재시작하지 않는다.
- 17:26:40 Kaggle API: **T00 train v1 `RUNNING`**. 새 launcher PID `15700`. T01/T02는 앞 실험의 학습 → Internet OFF 제출용 추론 → 산출물 검증 후 순서대로 진행한다.
- 학습·추론 모델 소스는 변경하지 않았다. T/U source pin을 유지했다. 전체 테스트 **92개 통과**, 새 shell의 dry run 통과.

현재 실행 중인 명령의 재개 방법(동일 실행기가 살아 있으면 중복 실행은 잠금으로 차단):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_plane_depth_experiments.ps1 -AvailableQuotaOnly
```

확인 위치:

- 현재 단계: `outputs/automation/plane-depth-20261019/queue_status.json`
- 실행 기록: `outputs/automation/plane-depth-20261019/available-quota.stderr.log`
- 완료한 실험별 점수·제출 notebook 링크: 같은 폴더의 `summary.csv`
- 모든 단계 완료 후 혼동행렬·소견별 지표: `outputs/analysis/plane-depth-20261019/`
- T00 원격 작업: https://www.kaggle.com/code/choiyoungwhan/rsna-knee-t00-position-control-train
- 순서 변경과 종료 프로세스 기록: `outputs/automation/available-quota-20261020.json`

이 실행기는 제출 가능한 `submission.csv`까지 생성·검증한다. 공식 리더보드 제출은 수행하지 않는다.
Kaggle에 이미 올라간 작업은 원격에서 실행되지만, 다음 업로드·다운로드·분석을 이어가려면 이 PC가 켜져 있고 절전 상태가 아니어야 한다.

## 이전 실행 계획 — 2026-10-19

2026-10-19. 사용자 선택: **기존 큐 완료 후 새 위치 실험 실행**.

- 구현: T00 기존 S11 재현 → T01 절대 깊이 추가 → T02 방향×깊이 상호작용.
- 현재 학습 결과: 없음. 구현 테스트 성공을 Kaggle 성능 개선으로 해석하지 않는다.
- 기존 큐: `followup-20261019`. R05까지 학습·제출 완료, R02 시작 전 무료 GPU quota 대기. 확인 시점 잔여 약 4.51시간, R02 요구 6시간, 갱신 예정 2026-10-24 09:00 KST.
- 새 큐는 기존 실행을 중단하거나 우회하지 않는다. 기존 큐 및 조건부 R07이 완료되면 T00→T01→T02를 순서대로 학습·제출 검증한다.
- 실시간 상태: `outputs/automation/plane-depth-20261019/queue_status.json` 및 `runner.log`.
- 재개: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_plane_depth_experiments.ps1`.

## 실제 실행 확인 — 2026-10-19 11:41 KST

- 백그라운드 대기 프로세스 시작. launcher PID **36152**, 실행 중 확인.
- `queue_status.json`: **`waiting_for_existing_runner`**. 기존 follow-up runner가 shared lock을 보유하고 있다.
- `runner.log` 11:41:43: 기존 runner를 기다리는 중이며 새 notebook 업로드는 아직 하지 않았음을 확인.
- 세 head 모두 **385,880 파라미터**. 동일 S11 feature cache signature 검증 통과.
- 새 위치 실험 테스트 **8개 통과**. 그 전 실행한 공간/후속 실험·큐 회귀 검증 포함 32개도 통과했고, 이후 완료 조건부 R07 검증 테스트 1개를 추가하여 통과했다.
- 학습/제출 notebook **6개**의 구문, 독립 실행 정의, nbformat 및 metadata 검증 통과. 합성 입력으로 세 제출 notebook의 모델 재로딩·확률 일치·CSV 생성 확인. 실제 MRI 재학습 점수나 Kaggle 공식 제출 점수는 아직 없다.
- shell `-DryRun` exit 0, 완료된 S11 train v1 입력과 T00→T01→T02 순서 확인.

PC를 계속 켜 두면 기존 큐 완료 후 이어진다. 현재 기존 큐 자체가 무료 GPU quota를 기다리므로 새 실험 시작 시각은 확정할 수 없다.
