# 실험별 Kaggle 학습·제출 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 각 실험을 완전한 학습/제출 Notebook 쌍으로 제공하고 backbone 비교를 실제 실행한다.
**Architecture:** 공통 MRI·분류 소스와 실험별 고정 설정을 Notebook 안에 포함한다. 학습은 평가와 최종 모델 bundle을 생성하고, 제출은 정확히 그 bundle을 오프라인에서 사용한다.
**Tech Stack:** 기존 Python/PyTorch/torchvision/pydicom/scikit-learn/Kaggle CLI. 런타임 패키지 추가 없음.
**Spec:** [설계](../specs/2026-10-16-experiment-notebooks-design.md)

## Global Constraints

기존 파일·출력 보존. 각 Notebook은 한 실험만 실행. 동일 3-fold/3 seeds/5 epochs 프로토콜, train-only 통계. 최종 재학습 seed 42. 제출 인터넷 OFF, test 전체 처리, 파일명 submission.csv, 비공개. 인증 정보 포함 금지.

## Review Focus

- 다른 실험의 bundle 연결을 거부하는가?
- encoder 추적 후 batch 크기가 달라도 원 모델과 같고, 온라인/오프라인 처리가 같은가?
- 짧은 시리즈·결측 방향·mask가 평균/attention에 NaN이나 padding 오염을 만들지 않는가?
- OOF 검증과 전체 gold 최종 모델을 구분하는가?
- 위치 없는 통합은 순열 불변이고 위치 추가 시 특징과 위치의 대응을 사용하는가?

## Task 1: MRI 토큰·head·검증 공통 코드

Create `experiments/core.py`, `tests/test_experiment_notebooks.py`.
Interfaces: `study_tokens(uid, series, image_root, encoder, transform, config, device)` → features/mask/positions; `StudyHead(config, mean, std)`; `fit_model(tokens, mask, positions, y, train_idx, seed, config)` → model/history. `validate_bundle(path, expected_config)` → manifest.

- [x] 의미 있는 실패 테스트: pooling/mask, fold normalization 누수, permutation/position, bundle ID/hash 불일치
- [x] 구현 및 테스트 통과

## Task 2: 실험별 독립 Notebook 생성

Create `experiments/backbones.py`, `experiments/train.py`, `experiments/submit.py`, `experiments/build_notebooks.py`, `notebooks/experiments/e*/{train,submit}/*`.
Interfaces: builder emits one literal `EXPERIMENT` config plus complete source, training includes only its selected backbone loader, submission consumes `bundle/assets.json`, `encoder.ts`, `model.pt`, wheels. All notebooks use shared core source by value, never import local project modules.

- [x] 실패 테스트: notebook compile/isolation, online code absent from submit, synthetic DICOM → exported model → reload → valid CSV
- [x] 생성기·train/CV/refit·offline submit 구현
- [x] CPU 테스트 및 Notebook 문법/동일 생성 확인

## Task 3: Kaggle 순차 실행과 결과 회수

- [x] E00/E01 학습·오프라인 제출 실행 검증
- [x] E02a/b/c/d 차례로 학습 및 오프라인 제출 실행 검증
- [x] 비교 결과에 따라 E03/04/05의 동일 기준 backbone 고정, Notebook 재생성
- [x] 실제 결과/버전/설정과 실행 상태를 문서화

## Task 4: 리뷰와 사용 안내

- [x] 독립 전체 구현 리뷰, 중요 수정의 회귀 테스트
- [x] 관련 전체 테스트와 생성물 검사
- [x] 실험별 위치·실행·재학습·제출 방법을 README에 작성

## 진행 기록

사용자는 새 계획 구현과 backbone들의 순차 실제 실행을 명시적으로 요청했다. 구현은 직접 수행하고 마지막에 독립 리뷰한다. 작업 폴더에 Git 저장소가 없으므로 worktree/commit 대신 기존 파일을 보존하고 새 경로에 작업한다. 조건이 미정인 DINOv3·판독문 약한 정답은 이번 학습 목록에 포함하지 않는다.

Task 1/2 완료: unittest 12개 통과. 합성 DICOM부터 encoder export/reload·CSV 생성까지 확인. 고정 224 입력·batch 1/S export 동등성은 각 실제 학습에서도 확인한다. E00 학습 v1 실행 시작. 독립 구현 리뷰 진행 중.

독립 리뷰 완료: bundle manifest에서 필수 모델/wheel 해시를 생략하면 검사를 우회할 수 있는 P2 한 건 발견. 누락 회귀 테스트가 실패하는 것을 확인한 뒤 두 모델·정확한 두 wheel의 해시를 필수화하여 전체 13개 테스트 통과. 리뷰어의 실제 ResNet/Swin/ConvNeXt/ViT CPU export/reload 및 batch 1~S 검증도 통과했다. 수정된 소스로 E00 v2 재실행하며 이후 비교는 이 버전을 사용한다.

E00 v2/E01 v1 학습 및 두 submit v1 COMPLETE. 점수는 이전 슬라이스 실험과 재현된다. DINOv2 v1은 export 비교에서 1/1920 특징의 절대 오차 1.37e-5가 atol 1e-5를 초과해 중단했다. float32 roundoff를 허용하되 실제 특징 변화는 거부하는 회귀 테스트 RED 확인 후 atol 1e-4/rtol 1e-4로 조정, batch 1~5별 최대 오차를 기록하도록 했다. 전체 14개 테스트 GREEN, DINOv2 v2 재검증 중. 학습과 추론 모두 내보낸 동일 encoder를 사용한다.

6개 backbone 비교 학습 모두 COMPLETE. DINOv2 v2 export 최대 절대 오차 2.82e-5 이하 확인. 최고 평균 OOF AUC 0.619577의 Swin-T를 E03/04/05 기본 backbone으로 고정했다. 전체 14개 테스트, 18개 실제 생성 Notebook 소스 일치·compile·버전 고정, 문서 링크 검사 통과. 결과 표와 소견별 비교를 docs/notebook_run_results.md에 기록했다. E03~E05는 코드/CPU 테스트까지이며 실제 Kaggle 실행 전이다.

완료: E00/E01/E02a/b/c/d의 학습 6회와 Internet OFF 제출 6회 모두 COMPLETE. 6개 CSV와 inference summary를 로컬에 회수하여 테스트 ID·열·확률 범위·실험 설정·모델 해시를 검증했다. 실제 공식 대회 제출은 하지 않았다. 코드 9쌍 제공 및 backbone 실제 순차 학습·검증이라는 이번 완료 기준을 충족했다.
