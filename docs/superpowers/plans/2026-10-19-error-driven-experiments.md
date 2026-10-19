# 오류 분석 기반 후속 실험 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 소견별 FN/FP와 감독 결측에 근거한 단일 변경 실험으로 MRI의 12개 소견 구별 성능을 높이고, 통신 오류에도 기존 Kaggle 작업을 안전하게 이어간다.

**Architecture:** E09·S10·S11·S12·S13의 고정 버전을 대조군/입력으로 재사용한다. 먼저 저비용 고정 앙상블을 확인한 뒤 S10을 기준으로 입력 범위·밀도·weak 감독·loss 정규화를 각각 분리한다. 마지막으로 현재 S12의 파인튜닝 초기화를 점검한다. 기존 모델/실행 기록은 보존하고 신규 R 실험군으로 구현한다.

**Tech Stack:** Python, PyTorch/torchvision Swin-T, numpy/pandas/scikit-learn, Kaggle SDK, PowerShell, unittest.

**Spec:** [2026-10-19 실험 결과·혼동행렬 분석](../../experiment_error_analysis.md). 원본 정답/예측 해시는 `outputs/analysis/2026-10-19/analysis_manifest.json`.

**상태:** 최초 기획 이후 사용자가 구현·자동 실행을 승인했다. R01~R06 및 조건부 R07을 구현했고 최종 검증/실제 실행을 진행한다. 최신 실행 상태는 [실행 기록](../../followup_execution_log.md), 해석은 [실험 노트](../../followup_experiment_results.md)를 참조한다. 기존 S13까지는 2026-10-19 10:01 KST에 모두 완료됐다.

## Global Constraints

- Gold58, 기존 Study ID별 3-fold, seed 42/2026/3407를 1차 비교에 그대로 사용한다. MRI·약라벨 source/hash를 고정한다.
- 1차 지표는 세 seed의 gold OOF macro ROC AUC 평균. BCE, 12개 소견 AUC/AP, seed42의 threshold 0.5 혼동행렬 및 다른 seed 범위를 함께 기록한다.
- 세 seed를 독립적인 174건으로 세지 않는다. 분석용 threshold 0.3/0.5/0.7은 최적 임계값 선정이 아니다.
- Gold 검증 fold는 학습·정규화·head 워밍업·checkpoint/threshold 선택에 사용하지 않는다. 공개 teacher의 독립성 미확립과 gold 반복 개발 한계를 결과에 명시한다.
- UNK는 정답 0이 아니다. source의 soft target은 유지하고 confidence/weight 정책 변경만 명시적으로 비교한다.
- Backbone Swin-T, 224px, 6슬롯, head 20 epoch를 기본으로 고정한다. 위치·장수·loss·epoch를 한 실험에서 함께 바꾸지 않는다.
- 기존 E/S 실험의 파일·상태·remote Notebook 버전을 덮어쓰지 않는다. 새 실행은 새로운 R slug와 실행 폴더를 사용한다.
- Kaggle 제출용 Notebook은 고정된 학습 버전과 offline 입력을 사용해 확률 `submission.csv`를 만든다. 자동 공식 채점 제출은 범위에 넣지 않는다.
- 원본 보고서/토큰/인증 정보는 새 로그에 출력하지 않는다. 로컬 분석 결과에는 검사 ID·정답·예측·오류만 둔다.

## Review Focus

1. **통신 오류 후 중복 업로드:** 읽기 재시도와 쓰기 결과 불명확 상태를 구분한다. 업로드 호출 자체는 자동 반복하지 않는다. Task 1에서 검증한다.
2. **Cross-suite 입력/캐시 오염:** E09/S13 등 서로 다른 queue의 모델, 다른 슬라이스 범위나 encoder hash를 잘못 섞지 않는다. Task 2/3에서 검증한다.
3. **희소/미상 소견의 loss:** batch에서 특정 소견의 유효 라벨이 0이더라도 NaN이나 가짜 음성 gradient가 생기지 않는다. Task 4에서 검증한다.
4. **Fine-tuning 초기화로 인한 gold 누수:** gold 전체를 본 S11 최종 head를 OOF backbone 미세 조정에 재사용하지 않는다. Task 5에서 검증한다.
5. **선택 편향과 threshold 과적합:** 같은 58건에서 최고의 모델/threshold를 선택한 점수를 독립적인 성능으로 발표하지 않는다. Task 6에서 검증한다.

## 실험 순서와 고정 비교 대상

| ID | 부모/대조 | 유일한 변경 | 주로 보는 문제 | 비용/우선순위 |
|---|---|---|---|---|
| R00 | 기존 결과 | 평가/입력 버전 감사, 읽기 재시도 | 실험 중단·비교 불일치 | 학습 없음, 먼저 |
| R01 | E09, S13 | 확률 고정 50:50 평균 | OA/골절 강점과 관절액/활막염 강점의 상호 보완 | 새 backbone 학습 없음 |
| R02 | S10 | 중심 범위 25~75% → 10~90% | 병변 범위를 놓치는 가설 | 특징 재추출 필요 |
| R03 | S10 | 중심 5개 → 9개, 범위 25~75% 유지 | 같은 범위에서 sampling 밀도 부족 가설 | 특징 재추출 필요 |
| R04 | S10 | known weak confidence <0.8은 weight 0 | 낮은 신뢰도 감독 영향 | 캐시 재사용, head 학습 |
| R05 | S10 | 소견별 weight 합으로 loss 정규화 | 유효 감독 수가 적은 소견의 기여 부족 | 캐시 재사용, head 학습 |
| R06 | S12 | weak-only head 1 epoch 워밍업 추가 후 기존 1 epoch FT | 새 head와 backbone을 동시에 갱신하는 초기 불안정 가설 | raw MRI 학습, 후순위 |
| R07 | R06 | backbone 1 → 3 epoch | FT 학습량/수렴 | R06 평가 후에만 |
| D01 | 선택된 고정 구성 | nested threshold/교정 진단 | 과다 양성·놓침의 trade-off | Kaggle AUC 실험과 분리 |

R02와 R03은 서로 누적하지 않는다. R04와 R05도 각각 S10과 비교한다. 성공한 변경을 합치는 실험은 이 단계의 결과를 보고 별도로 사전 정의한다. 소견마다 최고 모델/앙상블 비율을 58건에서 찾는 탐색은 수행하지 않는다.

## Task 1: 읽기 재시도와 결과 감사

**Files:** Modify `experiments/run_queue.py`; Test `tests/test_run_queue.py`; 기존 `scripts/analyze_experiments.py` 재사용.

**Interfaces:** `retry_kaggle_read(call, *, attempts=5, sleep=time.sleep)`는 인자 없는 읽기 callback 결과를 반환한다. 기존 `latest_version`, `wait_for_job`, 출력 다운로드의 호출 경계에 적용한다. `kernels_push`에는 적용하지 않는다.

- [ ] **Step 1:** `test_read_timeout_then_success`, `test_retry_exhaustion_preserves_state`, `test_nontransient_error_not_retried`, `test_push_is_never_retried`를 추가한다. Timeout/ConnectionError 후 성공은 동일 읽기만 반복, 401/403/404/config 불일치는 즉시 실패, 업로드 호출 수는 1임을 검증한다.
- [ ] **Step 2:** `python -m unittest tests.test_run_queue -v`로 해당 테스트 실패를 확인한다.
- [ ] **Step 3:** 일시적인 DNS/connection/timeout, HTTP 429/502/503/504만 최대 5회 시도한다. 대기는 5/10/20/40초, `Retry-After`가 있으면 300초 이내로 존중한다. 대기 중 로그를 남긴다. 재시도 후 버전 pin 확인을 유지한다. 다운로드는 동일 버전 파일로 재시도하고 검증 전 `downloaded/verified`를 기록하지 않는다.
- [ ] **Step 4:** 위 테스트와 `tests.test_spatial_queue`를 실행한다. 불명확한 push 결과는 재업로드하지 않고 remote version 확인이 필요하다는 오류를 유지한다. 모든 실험 OOF 재계산 명령도 실행해 기존 결과가 바뀌지 않았는지 확인한다.

## Task 2: 고정 앙상블과 새 queue의 기반

**Files:** Create `experiments/followup_configs.py`, `experiments/run_followup_queue.py`, `scripts/run_followup_experiments.ps1`; Modify `experiments/spatial_notebooks.py`의 신규 실험 생성 경로; Test `tests/test_followup_experiments.py`, `tests/test_followup_queue.py`.

**Interfaces:** `followup_configs()`는 R01~R07의 명시적 config 목록을 반환한다. `resolve_verified_inputs(run_roots, experiment_ids)`는 기존 state의 verified 입력과 assets/OOF hash를 검증하고 `{id: {ref, version, config, assets_sha256}}`를 반환한다. 실행 전 이를 새 폴더의 `inputs.json`에 저장한다.

- [ ] **Step 1:** `test_external_inputs_require_verified_matching_version`, `test_r01_probability_mean_and_alignment`, `test_resume_does_not_reupload_completed_stage`를 추가한다. seed·Study ID·fold 중 하나라도 다르면 앙상블 실패, 정확히 0.5/0.5일 때만 통과시킨다.
- [ ] **Step 2:** 테스트 실패를 확인한다.
- [ ] **Step 3:** 기존 `train_weak_ensemble`, offline ensemble 복원, `run_queue.run(..., builder=...)`, lock/manifest 검증을 재사용한다. 서로 다른 queue의 원본 결과를 수정하지 말고 검증된 외부 입력을 신규 실행의 의존성으로 등록한다. 새 ID/slug는 `r01_e09_s13_blend` 등 R 전용으로 만든다. 입력 Notebook 버전은 실제 완료 버전으로 pin한다.
- [ ] **Step 4:** R01은 gold와 무관하게 고정한 0.5/0.5로 OOF를 계산한다. 이 선택 자체가 기존 오류 분석 후의 탐색임을 표시한다. 소견별 최적 비율을 찾지 않는다. 입력 두 경로 중 하나라도 missing이면 중단한다.
- [ ] **Step 5:** shell `-DryRun`, standalone Notebook 구문, Internet OFF, 입력 버전, 중복 lock, 결과 재개를 테스트한다. 첫 실제 실행 범위는 R01 한 개로 제한하고 새 bundle→추론→CSV를 검증한다.

## Task 3: 범위와 밀도를 분리한 입력 실험

**Files:** Modify `experiments/spatial_core.py`, `experiments/spatial_configs.py`의 기존 기본값 호환 경로, `experiments/followup_configs.py`; Test `tests/test_spatial_experiments.py`, `tests/test_followup_experiments.py`.

**Interfaces:** 신규 config에 `center_range=[0.25,0.75]`를 명시한다. 기존 config에 값이 없으면 현재 동작과 같아야 한다. 중심 위치는 `linspace(start,end,centers)`이며 spatial signature에 범위·centers·neighbors를 모두 포함한다.

- [ ] **Step 1:** `test_center_range_reaches_expected_indices`, `test_old_config_selects_identical_slices`, `test_sampling_change_invalidates_feature_cache`, `test_short_series_deduplicates_and_masks`를 작성한다. 100장짜리 시리즈와 2장짜리 시리즈를 포함한다.
- [ ] **Step 2:** 신규 테스트 실패를 확인한다.
- [ ] **Step 3:** R02는 S10의 5중심×인접5장, RoPE/mm·방향 embedding·gate를 그대로 두고 범위만 10~90%로 바꾼다. R03은 범위25~75%와 인접5장을 유지하고 중심만9개로 늘린다. 방향마다 원본 slice order와 mm 좌표를 보존한다.
- [ ] **Step 4:** 기존 공간 테스트를 포함해 통과시킨다. audit에는 슬롯별 원본 장수·선택 고유 장수·선택 index 최소/최대·mask 비율을 기록한다. 시리즈 양 끝의 잡음 때문에 악화될 가능성도 결과에 기록한다.
- [ ] **Step 5:** R02/R03 각각 전체 weak+gold 특징을 재추출하고 동일 head CV를 수행한다. S10에 비해 장수/범위만 변했는지 assets와 cache signature를 확인한다.

## Task 4: 약라벨 품질과 소견별 loss 기여 비교

**Files:** Modify `experiments/weak_core.py`의 target/weight 경로, `experiments/spatial_train.py`의 loss 선택 경로, `experiments/followup_configs.py`; Test `tests/test_followup_experiments.py`.

**Interfaces:** 신규 `weak_min_confidence=0.8`, `loss_reduction='per_label'` config를 각각 R04/R05에서만 적용한다. 기존 기본값은 현재 수치와 동일하게 유지한다. 신규 `per_label_weighted_bce(logits, targets, weights)`는 유효 소견만 평균한다.

- [ ] **Step 1:** `test_confidence_mask_changes_weak_only`, `test_population_stays_fixed_when_all_labels_masked`, `test_per_label_bce_zero_mask_gradient`, `test_per_label_bce_equal_label_contribution`, `test_empty_batch_is_finite`를 작성한다. gold weight3 유지, UNK gradient0, 유효 감독 없는 소견 제외, 전부 mask인 batch의 differentiable zero를 확인한다.
- [ ] **Step 2:** 테스트 실패를 확인한다.
- [ ] **Step 3:** R04는 원래 확정한 4,350건 population과 이미지 캐시를 유지한다. known weak 중 confidence<0.8의 weight만 0으로 바꾼다. 모든 감독이0인 batch는 weight decay만 적용되는 일이 없도록 optimizer step도 건너뛴다. gold는 변경하지 않는다. 소견별로 제거된 YES/NO 수·남은 weight 합을 기록한다.
- [ ] **Step 4:** R05는 confidence 정책을 원래대로 유지한다. loss를 `mean_c(sum_b(w*BCE) / sum_b(w))`로 계산하되 weight 합0인 소견은 제외한다. class positive weighting, 샘플 oversampling, gold weight 변경을 동시에 넣지 않는다.
- [ ] **Step 5:** 합성 테스트와 기존 weak/spatial 테스트를 통과시킨다. R04/R05를 각각 S10과 비교한다. Synovitis 감독을 키워도 noisy-positive 편향이 악화될 수 있으므로 AUC뿐 아니라 FP·BCE를 확인한다.

## Task 5: gold 누수 없는 FT 초기화와 학습량 실험

**Files:** Modify `experiments/spatial_train.py`의 `finetune_spatial`, `experiments/followup_configs.py`; Test `tests/test_followup_experiments.py`.

**Interfaces:** `head_warmup_epochs=1`은 R06/R07에만 적용한다. warmup과 FT는 동일한 weak report-group train/validation 인덱스를 사용한다. `backbone_epochs`는 R06=1, R07=3이다.

- [ ] **Step 1:** `test_warmup_never_uses_gold`, `test_warmup_keeps_encoder_frozen`, `test_ft_updates_only_last_stage_and_norm`, `test_checkpoint_selected_on_weak_validation`, `test_encoder_change_invalidates_cache`를 추가한다.
- [ ] **Step 2:** 테스트 실패를 확인한다.
- [ ] **Step 3:** S11의 gold 전체를 본 최종 head를 로드하지 않는다. 새 head를 기존 weak train 3,835건의 frozen features로 **1 epoch**, AdamW lr0.001로 워밍업한다. weak validation457건은 워밍업/통계 계산에 사용하지 않는다.
- [ ] **Step 4:** 그 head와 기존 backbone으로 R06의 raw MRI FT 1 epoch를 수행한다. encoder lr1e-5/head lr1e-4, 마지막 stage+norm, effective batch8, 기존 loss/입력은 S12와 같게 유지한다. 이후 encoder를 고정하고 전체 특징을 새로 추출하여 동일 gold head CV를 수행한다.
- [ ] **Step 5:** R06의 결과·weak validation curve·처리량을 확인한 뒤 R07 실행 여부를 결정한다. R07은 FT epoch만3으로 바꾸고 weak validation BCE 최소 checkpoint를 선택한다. 시간 예산을 초과하면 임의로 epoch 수를 줄여 같은 실험 ID로 성공 처리하지 않는다.

## Task 6: 비교 보고서와 별도 교정 진단

**Files:** Reuse/extend `scripts/analyze_experiments.py`, `tests/test_experiment_analysis.py`; Create `docs/followup_experiment_results.md`. D01을 수행할 경우 Create `experiments/calibration_diagnostic.py`.

**Interfaces:** 신규 R 결과도 동일 `metrics.json`, `oof_predictions.csv`, `folds.csv`, `assets.json`, `inference_summary.json` 형식으로 기록한다. 외부 weak 규칙 변경이 있으면 소견별 유효 감독 audit도 필수다.

- [ ] **Step 1:** `test_followup_analysis_rejects_changed_gold_or_folds`, `test_threshold_does_not_replace_probability_submission`, `test_calibration_never_fits_on_outer_valid`를 추가한다.
- [ ] **Step 2:** 테스트 실패를 확인한다.
- [ ] **Step 3:** 각 R 결과에 부모 대비 AUC 차이, seed SD, 12개 소견 평균 AUC/AP, seed42 FN/FP, 처리 시간·메모리를 기록한다. seed별로 같은 검사가 반복됨을 유지한 paired bootstrap을 출력한다.
- [ ] **Step 4:** shortlist는 운영 기준으로 **부모 대비 평균 AUC +0.005 이상, 3개 seed 중 2개 이상 개선**을 사용한다. MCL/외측 반월상연골/골절의 평균 AUC가 0.03 이상 하락하면 전체 평균만으로 채택하지 않고 원인을 검토한다. 이 숫자는 통계적 유의성 기준이 아니다.
- [ ] **Step 5:** 선택된 후보 최대2개와 부모만, 미리 고정한 대체 3-fold split seed123에서 함께 재학습해 민감도를 확인한다. 각 fold의 양성/음성 존재 여부를 먼저 검사하고 실패하면 명시적으로 중단한다. 점수를 보고 좋은 split을 고르지 않는다. 같은 gold를 쓰므로 여전히 독립 검증은 아니다.
- [ ] **Step 6:** D01이 필요하면 각 outer train 내부에서 새 inner OOF를 만들어 threshold를 선택하고, untouched outer validation에 적용한다. 다른 outer fold의 OOF를 단순히 모아 교정 데이터를 만들면 그 모델이 평가 fold를 학습했을 수 있으므로 사용하지 않는다. 임계값 후보는 0.1~0.9 간격0.05, 목적은 balanced accuracy, 동률은0.5에 가까운 값을 사용한다. 원래 확률 AUC·BCE와 판정 지표를 분리해 보고한다.
- [ ] **Step 7:** 새 Notebook의 offline CSV 생성·ID/열/확률 범위·bundle hash를 확인한다. 실제 대회 채점은 별도 실행으로 남기고, 선택된 후보의 완료 버전을 명시한다.

## 실행·중단 기준

1. R00/R01 먼저 구현·검증한다. R01 이후 R04/R05는 캐시 재사용이 가능하므로 처리량 대비 효율적인 다음 후보다. R02/R03은 별도 입력 재추출이 필요하다.
2. R02~R05의 우선 실행 순서는 **R04 → R05 → R02 → R03**으로 한다. 비교 부모는 모두 S10이며, 앞 실험의 성패가 뒤 실험의 설정을 자동 변경하지 않는다.
3. R06/R07은 비싼 raw MRI 학습이다. 현재 S12 FT가 약9시간 걸린 기록을 근거로 GPU 예산을 사전에 확인하고, 예약된 무료 한도 안에서만 실행한다. 유료 quota를 자동 구매하지 않는다.
4. 반복 통신 장애가 읽기 재시도 한도를 넘으면 state를 보존하고 종료한다. 재개는 같은 실행 폴더/버전 pin에서 이어간다.
5. 신규 backbone 탐색, 외부 MRI 추가, 모든 변형의 무제한 조합은 이번 계획에 넣지 않는다. 현재 오류 분석은 입력 범위·소견별 감독·앙상블에 더 직접적인 근거를 준다.

## Self-review

- 요구된 전체 실험 비교, 12개 혼동행렬, 공통 오류, threshold 한계는 분석 문서에 완료했다.
- 각 후속 실험의 부모와 한 가지 변경을 고정했다. 범위와 밀도, confidence와 loss 정규화, 초기화와 epoch를 분리했다.
- 기존 재개 방식·공유 함수·Notebook builder를 재사용하며, 과거 fingerprint/결과를 덮어쓰지 않는 새 R 실행을 명시했다.
- 새 검증 gold를 만들거나 MRI를 실제 판독한 것으로 표현하지 않았다. 독립 teacher/환자 검증의 미확인 조건을 유지했다.
- 분석과 기획 완료는 신규 학습 실행 완료가 아니다. 실행 결과 표에는 아직 R 점수를 기입하지 않는다.
