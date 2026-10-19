# 실험별 Kaggle 학습과 제출

각 실험에 **완전한 학습 Notebook 하나와 완전한 제출 Notebook 하나**가 있다. Notebook 하나가 여러 실험을 한꺼번에 실행하지 않는다. 공통 소스는 생성할 때 Notebook 안에 포함하므로 이 저장소를 Kaggle에 따로 업로드할 필요가 없다.

```text
실험 N / train.ipynb
  대회 MRI → 고정 encoder → 3-fold 평가
  → 전체 gold 58건으로 최종 분류기 재학습
  → run/bundle/{encoder.ts, model.pt, assets.json, wheels/}

실험 N / submit.ipynb (인터넷 OFF)
  실험 N의 완료된 학습 버전 연결
  → ID·설정·파일 해시 확인
  → 현재 테스트 MRI 전체 예측
  → /kaggle/working/submission.csv
```

교차 검증에는 각 fold의 학습 검사만 쓴다. 제출용 모델은 검증 이후 모든 gold로 재학습하며, 이 최종 모델의 학습 점수를 검증 성능으로 보고하지 않는다. 최종 분류기 seed는 42로 고정했다. 기존 제출 Notebook은 변경하지 않았다.

**이번 비교는 사전학습 backbone의 가중치를 고정하고 마지막 분류기를 학습하는 실험이다.** E03~E05에서는 attention도 학습한다. MRI로 backbone 전체를 미세조정한 결과는 아니다. 각 실험의 실제 Kaggle 버전·점수·제출 파일 생성 상태는 [실행 결과](notebook_run_results.md)에 기록한다.

## Notebook 목록

| 순번 / ID | 변경 | 학습 전체 코드 | 제출 전체 코드 |
| --- | --- | --- | --- |
| 1 / E00 | ResNet18, 3장, 평균 | [train](../notebooks/experiments/e00_resnet18_3/train/train.ipynb) | [submit](../notebooks/experiments/e00_resnet18_3/submit/submit.ipynb) |
| 2 / E01 | ResNet18, 5장, 평균 | [train](../notebooks/experiments/e01_resnet18_5/train/train.ipynb) | [submit](../notebooks/experiments/e01_resnet18_5/submit/submit.ipynb) |
| 3 / E02a | DINOv2 ViT-S/14, 5장 | [train](../notebooks/experiments/e02a_dinov2_5/train/train.ipynb) | [submit](../notebooks/experiments/e02a_dinov2_5/submit/submit.ipynb) |
| 4 / E02b | Swin-T, 5장 | [train](../notebooks/experiments/e02b_swin_5/train/train.ipynb) | [submit](../notebooks/experiments/e02b_swin_5/submit/submit.ipynb) |
| 5 / E02c | ConvNeXt-Tiny, 5장 | [train](../notebooks/experiments/e02c_convnext_5/train/train.ipynb) | [submit](../notebooks/experiments/e02c_convnext_5/submit/submit.ipynb) |
| 6 / E02d | ViT-B/16, 5장 | [train](../notebooks/experiments/e02d_vit_5/train/train.ipynb) | [submit](../notebooks/experiments/e02d_vit_5/submit/submit.ipynb) |
| 7 / E03 | 선택 backbone + attention | [train](../notebooks/experiments/e03_attention/train/train.ipynb) | [submit](../notebooks/experiments/e03_attention/submit/submit.ipynb) |
| 8 / E04 | E03 + 슬라이스 상대 위치 | [train](../notebooks/experiments/e04_position/train/train.ipynb) | [submit](../notebooks/experiments/e04_position/submit/submit.ipynb) |
| 9 / E05 | E04 + 방향 임베딩 | [train](../notebooks/experiments/e05_plane/train/train.ipynb) | [submit](../notebooks/experiments/e05_plane/submit/submit.ipynb) |

E03~E05의 backbone은 비교 결과로 선택한 동일한 Swin-T를 사용한다. 64차원 투영, Transformer layer 1개, head 4개, FFN 128, dropout 0으로 고정한다. 각 방향 안에서 attention을 적용한 뒤 concat하므로 위치/방향 임베딩을 추가할 때 통합 구조까지 바꾸지 않는다. 상대 위치는 DICOM 좌표를 시리즈 내부 0~1로 정규화하며, 좌표가 없으면 InstanceNumber를 사용한다. 검사 간 해부학적 위치 정합을 수행한 것은 아니다.

## VS Code에서 Kaggle로 실행

전체 과정을 자동으로 진행하려면 [PowerShell 자동 실행 안내](automatic_experiments.md)를 사용한다. 아래 명령들은 각 단계를 수동으로 실행하는 방법이다.

아래는 E02a 예시다. 다른 실험은 폴더 이름만 바꾼다.

```powershell
# 1. GPU 학습 실행 (인터넷 ON, 대회 데이터 자동 연결)
.\.venv\Scripts\kaggle.exe kernels push -p notebooks/experiments/e02a_dinov2_5/train

# 2. COMPLETE가 될 때까지 상태 확인
.\.venv\Scripts\kaggle.exe kernels status choiyoungwhan/rsna-knee-e02a-dinov2-5-train

# 3. 방금 push가 출력한 학습 버전 번호로 변경 (이번 검증 완료 버전은 2)
.\.venv\Scripts\python.exe -m experiments.build_notebooks --experiment e02a_dinov2_5 --training-version 2

# 4. 오프라인 제출 Notebook 실행 (인터넷 OFF)
.\.venv\Scripts\kaggle.exe kernels push -p notebooks/experiments/e02a_dinov2_5/submit

# 5. COMPLETE 상태와 실제 submission.csv 출력 확인
.\.venv\Scripts\kaggle.exe kernels status choiyoungwhan/rsna-knee-e02a-dinov2-5-submit
.\.venv\Scripts\kaggle.exe kernels output choiyoungwhan/rsna-knee-e02a-dinov2-5-submit -p outputs/experiments/e02a_dinov2_5/submit
```

그 다음 Kaggle 대회의 Submit Predictions에서 **완료된 `...-submit` Notebook 버전**을 선택한다. 학습용 `...-train` 버전은 `submission.csv`를 만들지 않으므로 선택하지 않는다. 제출 전 공개 샘플 CSV 생성 성공과 실제 숨겨진 테스트 채점 성공은 구분해야 한다.

브라우저에서 직접 작업하려면 해당 `train.ipynb`를 Import Notebook하고 GPU·Internet ON·대회 데이터 연결 후 실행한다. 이어 `submit.ipynb`를 Import하고 해당 학습 Notebook의 완료 출력과 대회 데이터를 Add Input으로 연결한 뒤 Internet OFF로 실행한다. 실험 설정이 다른 모델을 붙이면 오류가 나는 것이 정상이다.

## 코드 수정과 재생성

생성된 Notebook을 여러 개 수동으로 고치는 대신 아래 원본을 수정한다.

- [core.py](../experiments/core.py): DICOM 처리, 슬라이스 토큰/위치/mask, 평균·attention 분류기, loss, bundle 검사
- [backbones.py](../experiments/backbones.py): 정확한 사전학습 가중치와 encoder 생성
- [train.py](../experiments/train.py): encoder 내보내기 검증, 교차 검증, 최종 재학습, 모델·wheel 묶음
- [submit.py](../experiments/submit.py): 오프라인 로딩, 현재 테스트 전체 예측, CSV 생성
- [build_notebooks.py](../experiments/build_notebooks.py): 고정된 실험 설정과 독립 Notebook 생성

```powershell
.\.venv\Scripts\python.exe -m experiments.build_notebooks
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

공통 소스나 backbone 코드를 바꾸면 `pipeline_sha256`도 바뀐다. 새 Notebook과 이전 모델 bundle을 섞지 말고 다시 학습한 버전을 연결한다. 이미 완료된 원격 Kaggle 버전은 수정되지 않는다. 로컬 재생성은 생성된 Notebook 파일을 갱신한다.

attention 기준 backbone을 바꿀 때는 E03/04/05 모두 같은 이름을 사용하여 재생성한다. 기본값은 실제 비교에서 선택한 Swin-T이며 선택 근거는 [실행 결과](notebook_run_results.md)에 기록한다.

```powershell
.\.venv\Scripts\python.exe -m experiments.build_notebooks --attention-backbone swin_t
```

학습 출력 경로가 이미 있으면 이전 결과를 덮어쓰지 않도록 중단한다. Kaggle의 새로운 저장 실행을 사용한다. 생성된 Notebook은 Kaggle의 고정 Python 3.13/PyTorch 이미지용이다. 로컬 Python 3.14의 TorchScript 경고가 있으므로 로컬 smoke test와 실제 Kaggle 실행 검증을 구분한다.

## 비교 해석

모든 실험은 같은 58건·3-fold·3 seeds·5 epochs·weighted BCE·AdamW 설정을 사용한다. backbone별 필수 전처리와 특징 차원도 함께 달라지므로 아키텍처만의 인과 효과로 해석하지 않는다. 5 epochs는 고정 예산의 첫 비교이며 각 모델의 최적 학습 횟수라는 뜻은 아니다.

출력 `run/metrics.json`, `folds.csv`, `oof_predictions.csv`, `history.csv`, `timings.json`에 검증 기록이 있고, `run/bundle/assets.json`에 모델 설정·환경·가중치 해시·최종 학습 인원이 있다. `run/tokens.pt`에는 슬라이스별 특징·mask·위치가 있어 후속 분석에 활용할 수 있다.

확정 정답이 58건뿐이며 후보를 선택한 검증셋은 독립 테스트가 아니다. 또한 공개 3건의 추론 성공으로 숨겨진 전체 테스트의 9시간 실행 한도를 확정하지 않는다. DINOv3 접근 승인과 판독문 약한 정답 생성은 이번 코드 목록에 포함하지 않았다.
