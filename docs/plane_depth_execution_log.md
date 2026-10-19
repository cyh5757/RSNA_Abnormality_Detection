# 방향 × slice 위치 실험 실행 기록

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
