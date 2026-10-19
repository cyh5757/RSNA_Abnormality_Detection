# 터미널에서 실험 자동 실행

VS Code의 PowerShell 터미널에서 프로젝트 폴더를 열고 한 번 실행한다. 이후에는 Codex나 다른 LLM을 호출하지 않으며 AI 토큰을 사용하지 않는다. 학습과 추론은 Kaggle GPU 사용량에 포함된다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1
```

기본 순서는 **E03 attention → E04 슬라이스 위치 → E05 방향 임베딩**이다. 모두 Swin-T, 방향별 5장을 사용한다. 한 실험의 학습과 제출 파일 검증이 끝나면 다음 실험을 시작한다.

```text
독립 Notebook 생성 → Kaggle에 비공개 학습 Notebook 업로드
→ 60초 간격으로 완료 확인 → 실제 학습 버전 번호 연결
→ Internet OFF 제출 Notebook 업로드·실행
→ 결과 다운로드·모델 해시/CSV 검증 → 다음 실험
```

기존 `.venv`와 Kaggle 인증을 그대로 사용한다. 새 패키지나 토큰 입력은 필요 없다. `ExecutionPolicy Bypass`는 이 PowerShell 프로세스에만 적용된다. 공식 리더보드 제출은 자동으로 하지 않으며, 생성된 `...-submit` 완료 버전을 대회 화면에서 선택한다.

## 실행 옵션

업로드 없이 대상과 순서만 확인:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1 -DryRun
```

실험 하나만 새로 실행:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1 -Experiments e03_attention -RunName e03-trial2
```

실험 목록과 순서를 직접 지정:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1 -Experiments "e04_position,e05_plane" -RunName position-plane
```

실험 ID는 [전체 목록](experiment_notebooks.md)을 따른다. `-PollSeconds 120`으로 확인 간격을 늘릴 수 있다. 기본 확인 간격은 60초이고, 한 Notebook에 최대 12시간을 기다린다.

## 중단과 재개

**같은 명령을 다시 실행하면 기록된 버전에서 이어간다.** 이미 업로드한 학습을 중복 업로드하지 않고, 검증까지 완료된 실험은 로컬 결과를 확인한 후 넘어간다. 새로 재학습하려면 다른 `-RunName`을 사용한다. 소스 코드나 실험 목록을 바꿨다면 새 실행 이름이 필요하다.

Ctrl+C로 중단해도 이미 시작된 Kaggle 작업은 계속 실행된다. 다음 실험의 업로드와 결과 수집을 자동으로 이어가려면 PC를 켜 두고 터미널을 유지한다. PC가 꺼졌거나 네트워크 오류가 발생하면 동일한 명령으로 재개한다. Kaggle 오류/취소/시간 초과는 다음 실험을 시작하지 않고 중단한다. 실패 원인은 `runner.log`, 내려받은 Kaggle 로그, 해당 Notebook에서 확인한다.

동시에 이 스크립트 두 개를 실행하는 것은 차단한다. 큐가 실행 중일 때 같은 Kaggle Notebook을 수동으로 새 버전 업로드하지 않는다. 원격 최신 버전이 기록된 버전과 달라지면 다른 실행의 결과가 섞이지 않도록 중단한다.

업로드 요청 직후 연결이 끊겨 `state.json`에 `version: null`이 남은 경우는 결과를 알 수 없으므로 자동 재업로드하지 않는다. Kaggle에서 해당 업로드가 완료됐는지 확인하고, 동일한 코드의 실행임을 확인한 버전 번호를 해당 train/submit 항목의 `version`에 기록한 후 재개한다. 업로드되지 않았다면 새 실행 이름으로 시작한다. 실패한 모델 코드를 수정한 뒤 재실행할 때도 새 실행 이름을 쓴다.

## 결과 위치

기본 실행의 결과는 `outputs/automation/next-experiments/`에 저장한다. `-RunName`을 지정하면 마지막 폴더 이름이 바뀐다.

| 경로 | 내용 |
| --- | --- |
| `runner.log` | 실행 진행과 오류 |
| `state.json` | 정확한 업로드 버전과 재개 상태 |
| `summary.csv` | 완료한 실험의 OOF AUC, BCE, 학습/제출 버전, Kaggle 링크 |
| `notebooks/<실험 ID>/train/`, `submit/` | 실제 업로드한 독립 Notebook과 메타데이터 |
| `<실험 ID>/train/run/` | metrics, folds, OOF 예측, history, timings, 모델 manifest |
| `<실험 ID>/submit/` | `submission.csv`, `inference_summary.json`, Kaggle 실행 로그 |

큰 모델 가중치는 Kaggle 학습 출력에 보존하고 제출 입력으로 연결한다. 로컬에는 비교와 검증에 필요한 작은 파일만 다운로드한다. 기존 `outputs/experiments/`의 수동 실험 결과는 보존한다. 자동 실행 결과는 위 `summary.csv`에 누적하며 기존 Markdown 결과표를 자동 수정하지 않는다.

진입점은 [run_experiments.ps1](../scripts/run_experiments.ps1), 실행 로직은 [run_queue.py](../experiments/run_queue.py)다. E03~E05의 실제 GPU 학습·오프라인 제출 파일 생성까지 완료했다. 이후 [약라벨 실험 E06~E12](weak_label_experiments.md)는 `scripts/run_weak_experiments.ps1`로 실행한다.
