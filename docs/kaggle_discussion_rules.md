# Kaggle 고정 토론·외부 데이터 규칙 확인

조사일: **2026-10-16**. Kaggle Discussion 원문과 대댓글을 API로 읽었다. 투표 수는 조회 시점 값이다. `HOST`·`ADMIN` 표시가 있는 운영진 발언과 참가자 해석을 구분했다. 현재 프로젝트 적용은 대회 MRI + 공개 Pilkwang 라벨 v1(약라벨 4,292건 + gold 58건), `UNK` loss 가중치 0, Swin ImageNet 사전학습을 사용하는 상태를 기준으로 한다.

**직접 적용할 내용은 공식 라벨 정의와 LLM 라벨 생성 허용이다. 외부 MRI는 일괄 허용 또는 일괄 금지로 결론낼 수 없다.** 10월 14일 운영진이 접근성 기준과 학습 데이터 공개를 다시 설명했지만, OAI 지역 제한 및 파생 체크포인트의 구체적인 판정 요청은 남아 있다. 아래 내용 때문에 현재 학습을 중단하거나 새 외부 데이터를 추가한 것은 아니다.

## 고정 게시물 5개

| 투표 | 게시물 / 작성자 역할 | 핵심과 현재 적용 |
| --- | --- | --- |
| 101 | [Knee Abnormality Detection AI Challenge Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) — Po-Hao “Howard” Chen, **HOST**; Dr. Jacob Kazam 제공 | 12개 소견의 판정 기준, 해부학, 촬영 방향·시퀀스, 양성/음성 예시. 라벨 품질 점검의 첫 기준이다. 검사 전체의 한쪽 무릎을 단위로 하며, 전문의 2인의 독립 판독과 제3자의 불일치 조정을 설명한다. |
| 26 | [Use of Commercially Hosted LLMs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965) — Chen, **HOST** | 보고서를 상용 LLM/API로 처리하여 라벨을 만드는 것은 허용된다. 합리적 접근성·최소 비용·대회 규칙·서비스 약관을 충족해야 한다. 이 처리 자체는 금지된 팀 외 비공개 공유로 보지 않는다. |
| 19 | [Welcome to Knee Abnormality Detection Challenge!](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733375) — Chen, **HOST** | MRI와 다국어 보고서를 함께 제공하는 대회 목적 및 효율성 상을 소개한다. 세부 모델 처방이나 새 규칙은 없다. |
| 11 | [How to get started + Competition’s Official Discord](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/730709) — María Cruz, **ADMIN** | 공식 Discord의 대회 채널은 공개 공간이다. 팀 외 비공개 코드·데이터 공유는 금지되며 코드 공유는 Kaggle 포럼/노트북에서 공개해야 한다. 운영진은 Discord를 모니터링하지 않으므로 중요한 질문은 Kaggle 포럼에 남긴다. |
| 3 | [AI 활용 설문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749038) — Chen, **HOST** | 익명·10분 미만 설문이며 순위와 수상 자격에 영향이 없다고 명시한다. 모델 학습과 직접 관련된 추가 지침은 없다. |

### 라벨 정의에서 놓치기 쉬운 부분

- ACL은 고도 부분 파열 또는 완전 파열, MCL은 고도 급성 파열을 목표로 한다. 경미한 변성·저등급 염좌를 모두 양성으로 확대하면 공식 기준과 어긋날 수 있다.
- 반월상연골은 표면에 닿는 신호가 적어도 2개 이미지에서 보이거나 형태 이상이 있어야 하며, 표면에 닿지 않는 내부 변성은 음성이다.
- 세 구획의 OA는 대략 1cm 이상의 영역에서 연골 두께 50% 초과의 고도 소실을 본다. Effusion과 Baker cyst는 중등도 이상이다.
- 공식 영상 판독에서 경계성 소견은 음성으로 처리한다. **이를 ‘보고서에 언급되지 않은 소견도 모두 음성’이라는 규칙으로 확대할 근거는 없다.** 현재 `UNK` 가중치 0은 보고서의 미언급/불확실성을 다루는 별도 학습 선택이다. 위 세부 기준과 이 구분의 근거는 [공식 Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)다.
- Chen, **HOST**는 08-08에 gold 라벨은 보고서에서 추출하지 않고 MRI에서 독립 판독했으며, 보고서와 충돌하면 영상 기반 라벨이 기준이라고 확인했다. gold의 음성은 미주석이 아니라 공식 방법으로 부재를 판정한 것이다. 보고서보다 다수 판독자의 엄격한 영상 기준을 쓰므로 불일치는 예상 가능하다는 설명이다. [공식 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826#3510434)
- 양측 무릎 검사·보고서는 각 사례를 개별 검토하고 참가자가 관련 검사를 구분할 수 있도록 공개 보고서 또는 DICOM 메타데이터를 필요한 만큼 조정했다고 같은 공식 답변에서 설명했다. 다만 양측 양성을 OR로 합친다는 집계 규칙은 명시하지 않았다. 이후 Overview에 올라온 구체적인 OR 집계 질문에도 확인한 댓글에는 추가 운영진 답변이 없다. [08-08 공식 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826#3510434), [10-07 후속 질문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343#3533980)

## 높은 투표의 규칙 토론과 최신 공식 답변

### 외부 MRI 데이터: 97표, 중요한 답변은 10월 14일 댓글

[Please clarify which external knee MRI datasets are allowed](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)는 **참가자 YYama의 질문**이다. 원문의 MRNet·SKM-TEA·fastMRI·OAI·KneeMRI·KneeXNet 목록은 운영진의 허용 목록이 아니다.

**Chris Carr, HOST / RSNA Director of Informatics**가 2026-10-14에 RSNA를 대표하여 답했다. 개별 데이터셋별 허용·금지 목록을 지정할 준비는 없으며, 단순 등록과 라이선스 동의만 요구하는 데이터는 접근성 기준을 충족하지만 기관 연구계약이나 기관 소속을 요구하면 충족하지 않는다는 설명이다. 그 밖에는 합리성 기준을 따르고 사용 데이터셋을 포함한 학습 방법을 투명하게 공개하라고 권고했다. [공식 댓글 3538535](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3538535)

관련 공식 발언의 시간순 정리는 다음과 같다.

| 일자 | 공식 발언 | 해석 한계 |
| --- | --- | --- |
| 08-25 | Chen, **HOST**: 기관과의 정식 데이터 이용 합의가 접근성 차이를 만들기 때문에 **KneeCoT 사용 불허**. [원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109#3516795) | 이용 정책이 바뀌면 입장을 바꿀 수 있다고 했지만, 확인한 원문에는 철회가 없다. 10월의 일반 원칙을 기존 개별 금지 철회로 읽지 않는다. |
| 08-27 | Chen, **HOST**: 상금 수령 자체로 상업적 이용이 되는 것은 아니며, 비상업적 제한만으로 대회에서 배제하지 않는다. 단순 등록은 대체로 가능하나 기관 승인·협상 계약·IRB·긴 자격 심사는 별도 문제다. [원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3517696) | 운영진은 외부 소유자의 약관을 해석하거나 면제할 권한이 없다. 수상 시 코드·모델 공개를 포함해 제공자 약관을 지킬 책임은 팀에 있다. |
| 09-21 | Chen, **HOST**: OAI가 무료이고 기관 심사·서명 또는 법무팀 없이 모든 참가자에게 일반적으로 접근 가능하다면 대회 규칙 위반이 아니다. [원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819#3526495) | **조건부 답변**이다. 제공자의 이용 약관 적합성까지 보장한 것은 아니다. |
| 10-14 | Carr, **HOST**: 단순 등록·약관 동의와 기관 소속·계약의 경계를 재확인하고 학습 데이터 공개를 권고. [원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3538535) | OAI의 실제 지역별 접근성이나 외부 데이터별 일괄 판정은 하지 않았다. |

[OAI 지역 제한 후속 글](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145)(15표, **참가자 Mobassir**)과 [10월 16일 후속 댓글](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416#3539588)은 지역 제한, OAI 기반 공개 가중치·의사라벨, 실격 적용 범위를 다시 묻는다. 조회한 댓글에는 이를 해결하는 추가 운영진 답변이 없다. 참가자의 접근 실패 보고, NDA 답신 전달, ‘공개 가중치면 괜찮다’는 추정은 대회 운영진의 판정과 구분해야 한다.

### LLM API 질문: 23표, 초기의 금지 추측은 해소됨

[Rules clarification: external knee-MRI datasets, and using an LLM API…](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652)는 **참가자 Fernando Faria의 질문**이다. 초기 댓글에 상용 API가 금지된다는 해석이 있지만, Chen이 08-09 공식 공지를 연결했으며 08-27에는 보고서를 읽어 라벨을 생성하는 LLM API 사용이 가능하다고 직접 재확인했다. 현재 공개 약라벨이 LLM으로 만들어졌다는 사실 자체를 금지 사유로 볼 근거는 없다. [운영진의 공지 연결](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652#3510695), [명시적 허용 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3517694)

다만 보고서 원문/일부를 별도 공개 Dataset으로 재배포해도 되는지 묻는 질문은 확인한 스레드에서 답이 없었다. **API 처리 허용을 원문 재배포 허용으로 확대하지 않는다.** 추론 시 라벨·임베딩 활용에 대한 운영진 답변도 실제 추론 입력을 확인하라는 조건을 붙인다. [재배포 질문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3521200), [추론 입력 관련 답변](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3517695)

## 공개 앙상블을 나중에 가져올 때 확인할 점

[What external data is inside the public 0.956/0.957 blends](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)(0표)은 **참가자 edteoh의 코드·manifest 조사**이며 공식 인증이 아니다. 작성자는 nartaa CoAtNet의 OAI, heliosli R7의 fastMRI/fastMRI+ 보조학습, `anatomy.pt`의 SKM-TEA를 지적한다. 첨부 Dataset의 이름만으로 실제 학습 출처를 판단할 수 없다는 실무적 점검 목록으로 유용하다. 각 원본 학습 이력과 라이선스를 직접 대조하기 전까지 해당 표를 확정 사실 목록으로 사용하지 않는다.

현재 프로젝트는 이런 공개 의료 MRI 체크포인트나 외부 MRI를 도입하지 않았다. 따라서 지금 필요한 일은 기존 공개 라벨의 출처·버전과 사용한 ImageNet 사전학습 가중치를 기록하고, 공식 판정 기준과 약라벨 의미를 혼동하지 않는 것이다. 추후 공개 앙상블을 도입할 때는 **모델 가중치·crop/segmentation 모델·의사라벨의 상류 데이터까지** 확인해야 한다. 이는 위 공개 앙상블 조사와 운영진의 학습 데이터 공개 요청에서 도출한 프로젝트 적용 제안이다.

공개 코드 실행, 새 데이터 다운로드, 학습·대기열 변경, 포럼 게시·설문 제출은 하지 않았다. 외부 데이터 제공자의 모든 이용 약관과 공개 앙상블의 전체 학습 출처는 별도로 검증하지 않았으므로, 이 노트는 특정 외부 가중치 사용을 보증하지 않는다.
