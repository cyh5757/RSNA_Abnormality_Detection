# 공개 상위권 방법을 검증하는 U 실험

Goal: 기존 R → T 완료 후 U00–U08을 자동 실행하고 독립적인 offline 제출 notebook과 OOF 혼동행렬을 남긴다.
Architecture: 기존 실행기/평가 함수를 재사용하고 `anatomy_*` 모듈만 추가한다. 기존 대기 큐가 사용하는 파일은 수정하지 않는다.
Tech stack: PyTorch/torchvision Swin-T, DICOM, Kaggle Python API, PowerShell.
Spec: 이 문서의 실험표 및 검증 계약. 사용자의 “알아서 실험까지 진행” 요청에 따른 자동 실행이다.

## 실험과 고정 조건

| 순서 | 변화 | 비교 대상 |
|---|---|---|
| U00 control | Swin 최종 7×7 map을 4개 영역으로 저장 후 면적 가중 평균; 기존 global pooling과 동일 | S11 |
| U01 mm crop | PixelSpacing을 이용한 영상 중심 140 mm 정사각형 resampling | U00 |
| U02 orientation | IOP 기반 영상 내 transpose/flip 정규화; oblique 잔여각 기록 | U01 |
| U03 patch attention | 평균 전에 2×2 영역 token을 보존하고 x/y encoding + slice RoPE + 소견별 cross attention | U02 |
| U04 full FT1 | weak-only로 Swin 전체 1 epoch 학습, 새 head warmup 1 epoch | U03 |
| U05 full FT3 | 같은 optimizer로 3 epoch까지 계속 학습 | U04 |
| U06 flip TTA | U05의 원본/영상 내 좌우 반전 예측 평균, OOF도 동일한 TTA | U05 |
| U07 probability blend | U03/U05 확률을 고정 0.5/0.5 평균 | U05 |
| U08 rank blend | 같은 두 모델을 학습 weak 데이터 기준 CDF로 변환하여 0.5/0.5 평균 | U07 |

Global Constraints:
- 동일 gold 58건, 3 folds × seeds 42/2026/3407, 약라벨/기본 head epoch/loss 유지. FT에는 gold를 일절 사용하지 않는다.
- U00–U02의 4영역 면적 가중 평균은 원래 global 평균과 같다. U03은 49개 patch 대신 4개 영역으로 공간 정보를 보존하는 제한된 첫 실험이다.
- 140 mm는 해부학 segmentation ROI가 아닌 영상 중심 crop. IOP 정규화는 3D registration/좌우 무릎 통일이 아니다. TTA는 영상 내 horizontal flip이며 해부학적 좌우 반사라고 주장하지 않는다.
- 전체 FT는 메모리/시간을 위해 slot당 최대 8장을 고정 표본으로 사용하고, 1 epoch를 4 segment로 나눈다. 추론은 기존 선택 장수를 모두 쓴다. 매 epoch 말 weak validation으로 backbone을 선택한다.
- 입력/소스/모델 SHA와 Kaggle version을 고정한다. 업로드 응답이 불명확하면 자동 재업로드하지 않는다. 실패한 선행 큐를 완료로 보지 않는다.
- 기존 R→T 순서를 지킨다. GPU 잔량이 부족하면 기다린다. private notebook 실행과 submission.csv 생성까지 자동화하며 공식 리더보드 제출은 하지 않는다.
- 공개 모델의 0.95 재현을 보장하지 않는다. 16 epoch/다중 backbone/segmentation ROI 재현은 이번 고정 실험과 다르다.

## 구현 작업

- [x] 1. geometry/quadrant/patch head: RED tests → 구현 → synthetic crop, orientation, pooling equivalence, mask invariance/gradient 검증.
  Interfaces: images → `[6,T,3,H,W]`; encoder → `[B,4,768]`; cache → `[N,6,T,4,768]`; head → `[B,12]`.
- [x] 2. 학습/FT/OOF/제출: fold heads 보존, weak-only segment checkpoint/optimizer 연속성, CDF training-reference 및 동일 TTA 평가.
  Interfaces: `run/{tokens.pt,fold_models,metrics.json,oof_predictions.csv,bundle}`; adaptation checkpoint + audit.
- [x] 3. 독립 notebook/자동 큐/shell: 선행 T 완료 확인, 원자적 상태 저장, 재개와 quota 대기, dry-run.
- [x] 4. 회귀 검증/새 reviewer/문서/백그라운드 대기 실행. 90 tests PASS, 30 notebook 검증; PID 47808, 2026-10-19 13:16 KST에 predecessor 대기 확인. GPU 결과는 아직 없음.

검증 명령: `.venv\Scripts\python.exe -m unittest discover -s tests -v` (기존 전체 suite 포함); 새 모듈 compile, 모든 notebook 코드 compile 및 train/submit class 일치. Expected: 모두 통과. live Kaggle 학습은 선행 큐와 quota를 기다리는 상태를 실제 상태 파일로 구분한다.

Review Focus: 유출 없는 fold normalization/FT/CDF, train-submit 전처리 동일성, odd map 가중 평균, 미지원 DICOM metadata 오류, empty slot masking, checkpoint provenance와 optimizer 연속성, crash/불명확 upload의 중복 방지, predecessor crash와 완료 구분.
