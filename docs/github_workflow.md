# GitHub에 진행 내용 기록하기

저장소: <https://github.com/cyh5757/RSNA_Abnormality_Detection>

`main`에 현재 코드·실험 문서·집계 결과를 보관한다. 데이터, 모델 가중치, API 토큰, `.venv`, 개인 실행 상태는 `.gitignore`로 제외한다. Git 커밋 작성자 이메일은 이 저장소에만 GitHub noreply 주소를 설정했다.

## 이후 변경을 올리는 방법

프로젝트 폴더에서:

```powershell
git status
git add -A
git diff --cached --stat
git commit -m "docs: record the next experiment results"
git push
```

`git status`는 변경 내역, `git add`는 이번 기록에 넣을 파일 선택, `git commit`은 로컬 기록 생성, `git push`는 GitHub 업로드다. 특정 파일만 기록하려면 `git add docs/followup_experiment_results.md`처럼 경로를 지정한다. 처음 올린 상태에서 변경이 없으면 새 commit을 만들 필요가 없다.

VS Code에서는 왼쪽 **Source Control** → 변경 파일 검토 → `+`로 stage → 설명 입력 → **Commit** → **Sync Changes / Push** 순서로 진행할 수 있다. 로그인 요청이 뜨면 `cyh5757` 계정으로 브라우저 인증한다.

## 실험 결과를 갱신할 때

1. 실제 완료된 학습·제출 결과만 집계한다. 대기 중인 실험에는 결과 숫자를 적지 않는다.
2. `README.md`와 `docs/portfolio_results.md`에 비교 조건 및 관찰을 기록한다.
3. `docs/results/<날짜>/`에는 집계표·혼동행렬만 옮긴다. `errors_seed42.csv`, 검사별 OOF, 원문 보고서는 넣지 않는다.
4. 생성 파일을 올리기 전, CSV에 검사 식별자나 보고서 열이 없는지 확인한다.

현재 자동 큐는 소스 hash를 고정한다. 실험 모듈을 수정하면 다음 업로드 전에 변경을 감지하고 멈추므로, 실행 중인 실험의 코드 변경과 새 실험 버전을 구분해야 한다. `.gitattributes`는 기존 파일의 줄바꿈까지 보존하도록 설정했다. 문서 추가와 Git commit 자체는 학습 실행을 바꾸지 않는다.

## 공개 저장소의 재현 범위

저장소에는 과거 개인 Kaggle 실행 상태와 가중치가 없다. 기존 R/T/U 큐를 새 PC에서 바로 재개하려면 해당 선행 학습 artifact와 계정 접근 권한이 필요하다. 독립 notebook의 metadata에 들어 있는 계정과 input 버전도 실행 환경에 맞게 설정해야 한다.

공식 안내: [로컬 코드를 GitHub에 올리기](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
