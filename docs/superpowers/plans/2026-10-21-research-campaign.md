# 공개 방법 조사와 무인 실험 실행 계획

> Execution: `superpowers:executing-plans`, 현재 세션에서 구현·검증 후 Python 실행기에 인계한다. 사용자가 조사·계획 선정·실행을 모두 위임했다.

Goal: 공개 Notebook 방법과 기존 제안을 합친 유한한 실험 목록을 실행하고, LLM 호출 없이 무료 quota 대기·버전 고정·산출물 검증·비교 보고서 생성을 자동화한다.

Architecture: 기존 `run`, `resolve_verified_inputs`, `wait_for_gpu`, 독점 잠금, R/U 실행 함수를 재사용한다. 기존 source hash가 바뀌지 않도록 V 실험은 별도 모듈로 구현한다. 완료된 E/S/R/T 작업은 재학습하지 않는다.

## 확정 실행 목록

| 순서 | ID | 변경 / 대조군 |
|---|---|---|
| 1 | V00 | R01 0.8 + S10 0.2 고정 확률 결합 |
| 2 | V01 | R01 0.8 + T01 0.2; T01의 정확한 head 로더 사용 |
| 3 | V02 | R01 0.8 + S02 0.2 |
| 4 | V03 | S10 동일 cache/head에 학습 중 slice dropout 0.15; 활성 slot당 최소 1장 유지 |
| 5 | V04 | S10 head 3개 seed 확률 평균; 각 CV seed s에 s/s+1/s+2, 최종 42/43/44 |
| 6 | V05 | S10과 동일 입력·약라벨·head, encoder만 ConvNeXt-tiny로 교체; 특징 재추출 |
| 7 | V06 | R01 0.8 + V05 0.2 |
| 8 | R02/R03/R06 | 기존 보류 실험 재개; 기존 기준 충족 시 R07 |
| 9 | U00–U08 | 기존 공간 crop·방향·patch attention·전체 FT·TTA·확률/rank 실험 실행 |

V05는 같은 전처리를 유지하는 backbone 교체 실험이다. 공개 ConvNeXt-small/CoAtNet 모델을 그대로 재현하는 실험이 아니다. 동일 원본 모델을 여러 번 중복 가중하지 않는다. Gold58로 소견별 최적 가중치를 탐색하지 않는다.

## 범위와 운영 조건

- 무료 quota만 사용한다. 부족하면 10분마다 상태를 읽고 기다린다. 새 실험은 순차 실행한다.
- 공개 원문은 읽기 자료로만 저장한다. 공개 Notebook 코드를 실행하거나 외부 학습 가중치를 무검증으로 결합하지 않는다.
- 학습/추론 Notebook은 private. Internet OFF 제출용 CSV 생성까지 수행하며 공식 leaderboard 제출은 하지 않는다.
- 이전 입력은 local artifact hash와 remote version을 검증한다. 업로드 결과가 불명확하면 중복 업로드하지 않고 실패 상태를 기록한다.
- agent/LLM/API 토큰을 사용하는 자동 호출, 예약 Codex 작업, agent의 사후 감시는 만들지 않는다.
- 실행 중 source 변경은 중단 사유다. PC 종료/절전 시 원격 현재 작업은 계속되지만 로컬 후속 단계는 동일 명령으로 재개해야 한다.
- 현재 미커밋 변경 5개는 보존한다. 별도 worktree는 기존 local artifacts의 ROOT 경로와 큐 상태를 분리하므로 사용하지 않는다. 배포/commit/push는 범위에 없다.

## 구현 작업

- [x] 조사: Hotness/Most Votes 각 20개 metadata, 대표 12개 source/version/hash와 방법별 근거를 저장한다.
- [x] 모델/결합: `campaign_core.py`, `campaign_train.py`, `campaign_submit.py`; 가중 확률/ID 정렬, 중첩 leaf 가중치, T01 로더, dropout 마스크, seed 평균 roundtrip을 테스트 먼저 검증한다.
- [x] config/notebook: `campaign_configs.py`, `campaign_notebooks.py`; V00–V06 및 standalone train/submit 동등성·offline metadata 검증.
- [x] 실행기: `run_campaign_queue.py`, `run_research_campaign.ps1`; 완료 건 재사용, source pin, 잠금, quota, R07 결정, U adaptation 재사용, 단계별 실패 기록과 비교 보고서 생성.
- [x] 전체 테스트, dry run, 독립 코드 리뷰 후 숨김 프로세스로 실행. 시작·인계만 확인하고 agent 작업을 끝낸다.

## 검증 초점

1. CPU 앙상블 학습은 GPU 시간을 쓰지 않으며, 제출 추론은 GPU quota를 확인한다.
2. T01을 일반 SpatialHead로 읽거나 중첩 R01 비율을 잃지 않는다.
3. 멀티 seed에서 같은 fold의 validation gold가 학습에 섞이지 않으며 OOF는 같은 구성의 확률 평균이다.
4. 재개 시 완료된 remote versions를 덮어쓰거나 업로드를 중복하지 않는다.
5. 원격 오류/장기 quota 대기 후에도 실패·진행 상태와 재개 지점이 남는다.

## 실행 기록

- 계획 수립: 2026-10-21 KST. 기존 합의의 자동 실행 요청에 따라 추가 확인 없이 진행한다.
- 검증: 전체 105개 테스트 통과(14.953초), 14개 standalone Notebook 생성, dry-run 입력 8개 검증, `git diff --check` 통과. 로그: `outputs/analysis/campaign-setup-20261021/tests.log`.
- 독립 코드 리뷰: adaptation quota 대기 이후 campaign 전체 소스 확인 누락 1건을 수정하고, 변경 시 업로드가 차단되며 기존 guard가 복원되는 회귀 테스트를 추가했다. 나머지 가중치·T01·seed 재로딩·기존 R 재개에서 차단 결함 없음.
- 아직 측정하지 않은 항목: 실제 Kaggle 환경에서의 신규 모델 GPU 메모리/실행시간. quota 시간은 보수적인 실행 전 예산이며 완료시간 보장이 아니다.
- 인계: 2026-10-21 10:34:54 KST 숨김 Python 실행기 시작(PID 35436). V00 train 원격 버전 1 저장 및 Kaggle `RUNNING`을 시작 확인했다. 이후 agent의 자동 확인/예약 작업 없이 종료한다.
