# SDD ledger — plan: docs/superpowers/plans/2026-10-19-error-driven-experiments.md

2026-10-19. User authorized the comparison notebook, implementation, and automatic Kaggle runs.

- Ruling: workspace has no Git repository; use the existing project and source snapshots in `outputs/analysis/followup-source-snapshot` rather than create a worktree or invent commits.
- Pre-flight Task 1→2: reuse the runner with verified external inputs; never retry ambiguous uploads.
- Pre-flight Task 2→3/4/5: R01 mixes a WeakHead with a nested spatial ensemble, so recursive bundle validation and weighted leaf inference are required.
- Ruling: preserve old configs/artifacts; new feature signatures explicitly reuse the old pixel pipeline hash only when sampling/preprocessing are unchanged. Encoder hash and exact IDs still gate cache reuse.
- Ruling: R07 is conditional on R06 quality and measured duration fitting the Kaggle session budget. No unlimited GPU retry or paid quota purchase.
- Tasks: 1 retry; 2 queue/mixed ensemble; 3 sampling; 4 supervision/loss; 5 warmup; 6 comparisons/notebook/launch.

- Task 1 complete: retry tests RED→GREEN; `tests.test_run_queue` 12 passed. Push remains single-attempt with ambiguous-outcome state.
- Task 2 implementation complete: verified external pins, recursive .5/.25/.25 inference, standalone CPU R01; mixed synthetic bundle→offline CSV integration passed.
- Task 3 implementation complete: range/default compatibility tests RED→GREEN; encoder/hash/IDs gate caches; per-slot selection audit on re-extraction.
- Task 4 implementation complete: confidence/zero-gradient/equal-contribution tests RED→GREEN; gold and population retained; zero-supervision batches skip optimizer.
- Task 5 implementation complete: weak-only split test RED→GREEN; warmup starts fresh head, gold excluded; R07 runtime/quality gate coded.
- Task 6 analysis complete: all30 OOF/manifest checks passed; 30 PNG+SVG confusion grids, all-experiment recall/precision, standalone comparison HTML and notebook written. New remote results pending launch.
- Ruling: CPU R01, then R04→R05→R02→R03→R06, R07 conditional — free GPU remaining about4.93h; wait automatically for quota instead of launching a job with insufficient allowance. Cost if conservative: delayed learning, no extra purchased usage.
- Ruling: D01 calibration and alternate split123 stay conditional after candidate selection, not a blind additional queue — no fixed shortlist exists yet. Cost: robustness validation remains outstanding; no independent-performance claim.
- Verification: unittest discovery printed63 tests OK; explicit exit-code verification and whole-change review follow. PowerShell dry-run passed with per-process ExecutionPolicy Bypass (machine policy unchanged).
- Verification: explicit unittest discovery exit0, 63/63 passed. HTML JavaScript executed with DOM stubs: 30 rows,12 category panels, selection update; PNG visually inspected.
- Final review: independent read-only reviewer found no critical issue. R07 internal7h FT admission and dynamic R-parent bootstrap were important findings.
- Final: fixed R07 budget mismatch — `test_r07_respects_both_ft_and_total_session_budget` RED→GREEN.
- Final: fixed missing R-parent intervals — `test_bootstrap_pairs_include_available_followups_and_parents` RED→GREEN.
- Final: minor (deferred): full warmup orchestration is not exercised on a GPU locally; split/no-gold, freeze-last-stage gradients and offline restoration have tests. Real R06 completion remains the end-to-end evidence.
- Final: Ruling: review set aside live Kaggle behavior/throughput and actual GPU training — validate R01 remotely now, subsequent runs remain asynchronous; cost if wrong is a stopped job with preserved state, not a falsely completed experiment.
- Final: Ruling: future quality/calibration/independent clinical generalization were set aside — keep them unclaimed and conditional; cost is additional validation work before choosing a final model.
- Final: reviewer did not rerun63 tests — explicit parent-run log/exit0 is the verification evidence, not the review's authority. Full final suite follows fixes.

## 실제 실행 상태 — 2026-10-19 10:51 KST

- 최종 `python -m unittest discover -s tests -q`: **65/65 통과, exit0**. 로그: `outputs/analysis/followup-source-snapshot/final-tests.log`.
- 검토용 독립 Notebook7쌍 생성: `notebooks/followup/`.
- 자동 runner 시작: PID7684, `outputs/automation/followup-20261019/runner.pid`.
- **R01 train v1 업로드 성공, Kaggle API RUNNING 확인**. CPU 실행. [Kaggle R01 학습](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-r01-e09-s13-blend-train).
- 이후 R01 offline submit 검증 → R04 → R05 → R02 → R03 → R06 → 조건부R07. 무료 GPU 부족 시 자동 대기.
- R 점수/CSV 완료 여부는 아직 대기 중이며, 완료 전 성공으로 기록하지 않는다.

## 실제 실행 갱신 — 2026-10-19 10:58 KST

- **R01 train v1 + offline submit v1 모두 완료/검증 통과**. `state.json` verified=true, `summary.csv` 기록.
- R01 OOF macro AUC **0.7714318551**, seed SD0.0089159338, BCE0.5543932120.
- 공개 test3건×12확률의 `submission.csv` 형식·ID·확률 범위·학습 manifest 일치 검증. [R01 제출 노트북](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-r01-e09-s13-blend-submit).
- **R04 confidence train v1 RUNNING** (10:58:24 runner 확인). [R04 학습](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-r04-confidence-train).
- 이 프로세스를 그대로 유지해 R05→R02→R03→R06 및 조건부R07을 이어간다. 무료 quota 대기는 자동, PC 절전/종료 후에는 같은 shell로 재개.
- 최종 분석 재실행 exit0: 기존30+R01 **31개**, 소견별 confusion PNG/SVG31쌍, 비교 HTML·CSV 모두 갱신. R01−S13 paired구간+.00423~+.04226 (선택 편향을 제거한 독립 검증은 아님).
- 10:59 KST 최종 확인: runner PID7684 유지, R04 train v1 RUNNING. 아직 R04 이후의 성능/전체 GPU 완료를 주장하지 않는다.
