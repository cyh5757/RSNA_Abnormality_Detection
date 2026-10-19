# 상위권 방법 기반 U 실험 노트

2026-10-19 13:16 KST: 코드 구현·로컬 테스트 90개·notebook 30개 검증 완료. 자동 실행기 PID 47808을 시작했고 `waiting_for_predecessor`를 확인했다. **실제 Kaggle U 학습 결과는 아직 없다. 기존 R → T가 완료된 뒤 실행한다.**

시작 시점의 R 큐는 GPU 잔여 약 4.48시간 / 필요 6시간으로 대기 중이었다. T 큐도 기존 runner를 기다린다. 이 상태 때문에 U를 먼저 실행하지 않는다.

기존 모델은 이미 Swin feature에 slice RoPE와 소견별 cross attention을 적용한다. 다만 각 이미지의 Swin 공간 map을 먼저 평균했다. 이번에는 그 평균을 늦춰, 이미지 안의 위치 정보가 도움이 되는지 검증한다.

| 순서 | 변경 | 비교 대상 | 질문 |
|---|---|---|---|
| U00 | 기존 global 평균과 동등한 면적 가중 4영역 평균 | S11 | 새 저장·실행 경로의 대조군이 재현되는가? |
| U01 | PixelSpacing 기반 140 mm 영상 중심 crop | U00 | 촬영마다 다른 물리적 배율을 줄이면 좋은가? |
| U02 | DICOM IOP 기반 영상 내 transpose/flip | U01 | 같은 해부학 방향을 일관되게 보여주면 좋은가? |
| U03 | 2×2 영역 token + x/y sin/cos + slice mm RoPE + 소견별 cross attention | U02 | 평균 전 공간 정보를 유지하면 도움이 되는가? |
| U04 | 전체 Swin backbone 1 epoch fine-tuning | U03 | ImageNet feature를 MRI에 적응시키면 좋은가? |
| U05 | 같은 optimizer로 3 epoch까지 연속 학습 | U04 | 학습량 증가가 실제 개선으로 이어지는가? |
| U06 | U05 원본·영상 내 좌우 반전 예측 평균 | U05 | TTA가 예측 변동을 줄이는가? |
| U07 | U03 + U05 확률 50:50 | U05 | 고정 feature와 적응한 feature의 오류가 상호 보완적인가? |
| U08 | 같은 두 모델의 고정 기준 CDF 50:50 | U07 | 모델별 점수 범위 차이를 줄이면 순위가 개선되는가? |

약라벨은 보고서 기반 공개 예측 라벨이며 영상별 병변 위치 정답이 아니다. 4,350건 중 gold 58건은 3 folds × 3 seeds로 평가한다. 보고서 teacher 자체의 gold 독립성은 증명되지 않아 이 평가는 개발용 OOF다. 공개 리더보드 0.95와 같은 수치로 해석하지 않는다.

Backbone 학습은 gold를 전부 제외한다. 보고서 hash로 분리한 weak train/validation을 사용하며, head도 weak train만으로 새로 warmup한다. 기존 gold 학습 final head를 가져오지 않는다. Fold별 head의 정규화 통계는 해당 train fold에서만 계산한다. Rank 기준 분포도 해당 head가 학습한 weak 데이터로 만든다. 검증 정답으로 앙상블 가중치를 조정하지 않는다.

한 backbone epoch는 4개 Kaggle 작업으로 나뉜다. 각 작업은 모델·optimizer·AMP scaler·난수 상태를 저장하고 다음 작업이 이어받는다. U04에서 4개, U05에서 추가 8개 작업을 실행한다. GPU 메모리를 위해 backbone 학습 때는 촬영 slot당 최대 8장을 일정 간격으로 사용하며 gradient checkpointing을 적용한다. Head 학습과 제출 추론에서는 원래 선택한 장수를 모두 쓴다. 각 epoch 말 weak validation BCE가 가장 낮았던 backbone을 선택한다. 따라서 U05가 U04와 같은 epoch를 선택할 수도 있다.

해석상의 한계:

- 140 mm 영상 중심 crop은 nartaa의 segmentation 기반 무릎 ROI와 다르다. 병변을 잘라낼 가능성도 실험으로 확인해야 한다.
- IOP 처리는 영상 내 축의 swap/flip이다. 남은 oblique 각도를 기록하며 3D 재구성이나 좌우 무릎 통일을 수행하지 않는다.
- U03은 최종 7×7 map 전체 대신 4개 영역을 보존하는 첫 실험이다. 4영역의 크기 16/12/12/9를 반영한 평균은 원래 global 평균과 같지만 FP16 저장에 따른 작은 반올림 차이는 가능하다.
- U06은 영상 내 horizontal flip이다. 3D 해부학적 좌우 반사의 완전한 재현이 아니다. OOF에도 같은 변환을 적용해 효과를 측정한다.
- U08 값은 확률로 보정된 값이 아닌 순위 점수다. 0.5 혼동행렬과 BCE는 참고이며 macro AUC와 오류 사례를 우선 비교한다.
- 공개 0.95 코드의 16 epoch, 다른 backbone, segmentation ROI까지 그대로 재현한 실험은 아니다. 각 요소를 통제하여 현재 방법에 필요한 개선을 찾는 실험이다.

## 자동 실행과 결과

자동 실행기: [run_anatomy_experiments.ps1](../scripts/run_anatomy_experiments.ps1).

```powershell
# 예상 실행 순서만 확인
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_anatomy_experiments.ps1 -DryRun

# 실행 또는 중단된 동일 큐 재개 (이미 실행 중이면 중복 실행하지 않음)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_anatomy_experiments.ps1
```

PC의 로컬 runner가 켜져 있어야 다음 작업이 올라간다. 이미 올라간 Kaggle 작업은 PC가 꺼져도 계속되지만 후속 작업은 runner를 재개해야 한다. 기존 T 큐가 실패하거나 결과 검증이 안 되면 U도 업로드하지 않는다. GPU 잔량이 부족하면 무료 quota 갱신을 기다린다. 응답이 불명확한 업로드는 중복 업로드하지 않고 상태를 보존한다.

- 전체 대기 상태: `outputs/automation/anatomy-20261019/queue_status.json`
- 로그: `outputs/automation/anatomy-20261019/runner.log`
- 각 결과: `outputs/automation/anatomy-20261019-uXX_.../uXX_.../train/run/`
- 제출 CSV: 같은 실험의 `submit/submission.csv`
- 비교 결과: `outputs/analysis/anatomy-20261019/confusion_comparison.html` 및 `parent_category_deltas.csv`
- 수동 검토할 독립 notebook: `notebooks/anatomy/<실험 ID>/train/` 및 `submit/`

각 학습/제출 완료 후 실제 OOF로 12개 소견별 TN/FP/FN/TP, 재현율, 정밀도, AUC를 갱신한다. FN 감소가 FP 증가를 동반하는지, 평균 AUC 개선이 일부 소견 악화를 가리는지 함께 본다. 단일 seed 향상만으로 결론을 내리지 않는다.

Private Kaggle notebook 실행과 **Internet OFF 제출 notebook의 submission.csv 생성·검증**까지 자동화한다. 공식 리더보드 제출은 자동으로 하지 않는다. 출력 JSON이 아닌 `submission.csv`를 선택해야 한다.

근거 자료: [Hotness / Most Votes 확인](kaggle_hotness_votes_095_refresh.md), [0.95 점수 근거 검토](kaggle_095_evidence_audit.md), [구현 계획](superpowers/plans/2026-10-19-public-method-experiments.md).
