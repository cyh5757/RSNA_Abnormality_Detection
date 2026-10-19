# Kaggle 0.95 이상 후보 소스 검토 A — 처음 10개

검토일: 2026-10-19. `selection.json`의 처음 10개를 대상으로 **실제로 실행되는 코드, 내장된 Python 소스, 고정 모델 manifest**를 읽었다. 공개 노트북·내장 모듈·체크포인트는 실행하거나 로드하지 않았다. base64/zlib, base91/LZMA 자료는 바이트와 텍스트로만 복원했고 가능하면 원문 SHA256과 대조했다. 셀 번호는 JSON의 **0부터 시작하는 인덱스**다.

**핵심 결론:** 이 10개는 독립적인 10개 학습 성공 사례가 아니다. 대부분 nartaa의 CoAtNet/Raptor 두 체크포인트, heliosli의 R6 또는 R7, goodpjw2008의 2.5D reader를 다시 조합한 추론 노트북이다. 사용자의 방식과 가장 큰 차이는 물리 단위 영상 처리, 넓고 조밀한 슬라이스·국소 시야, 여러 epoch의 영상 encoder 학습, 보고서의 상태/정도 감독, 서로 다른 학습 계열의 혼합이다. 단순히 후단 attention을 추가해서 0.95가 된 사례로 읽으면 잘못이다.

## 점수와 버전의 증거 범위

수집된 Kaggle metadata에는 공식 제출의 숫자 점수가 없다. 아래 숫자는 **제목·본문·그림에 저자가 기재한 값**이다. 타인의 점수 인용, 과거 버전 점수, 현재 코드 점수를 구분했다. 특히 `.95`라는 숫자가 EMA·분위수 등 하이퍼파라미터에 등장하는 것은 성능 증거로 쓰지 않았다. 공식 제출 API를 통한 점수 검증이나 이 환경의 재현 실행을 했다는 뜻이 아니다.

| # | 노트북 / 고정 버전 | 원문 SHA256 | 점수 증거와 한계 |
|---|---|---|---|
| A1 | [goodpjw: CoAtNet + R6 + 2.5D, v27](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | `a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c` | 셀1: v25/v26 **.958**, 현재 v27은 v25 recipe의 문서 갱신이라고 명시 |
| A2 | [goodpjw: Does S6 help, v8](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957) | `64ac2a3070ef3b48dc14a433119b41d2812c14280922acb3ee1c50bb6db61c3f` | 셀1: v4/v5 .957, v6 .957, v7 .955; v8은 v6 코드와 결과 기록, 새 점수 주장 없음 |
| A3 | [heliosli: best public dataset, v7](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | `cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd` | 제목 **.9586**; 현재 v7의 공식 점수·각 변경의 개별 상승분은 원문으로 검증 불가 |
| A4 | [heliosli: old version .9573, v2](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) | `85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24` | 제목 **.9573**; 현재 bundle은 R7, 이름에 R6가 있는 dataset이어도 R6라고 읽으면 안 됨 |
| A5 | [heliosli: blend mine, v1](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) | `ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d` | 자체 제목에는 숫자 없음. A1/A2가 이 recipe를 **.958**이라고 명시적으로 인용 |
| A6 | [heliosli: .957 one-click, v1](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | `cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315` | 제목 약 **.957**, A1/A7 본문은 같은 R7+Raptor를 **.956**으로 설명. 둘을 합쳐 .957 확정으로 쓰지 않음 |
| A7 | [evgendvorkin: RSNA Versia 7, v22](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | `1b2282b6e5bfe83f77eb07e09b830d5ba1033733bb483487f6843abcff685bc2` | 셀1: 약 **.956은 원저자의 점수**, 자신의 재현 점수는 별도 확인해야 한다고 명시 |
| A8 | [prvsiyan: bee's knees, v47](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push) | `07a50c3e6e5af6616cd6cd41a647adb40d30141e33f80cf9822e1cf8f8c41d3e` | 셀145: upstream **.957**, **자신의 새 공식 점수 없음**. 셀150: .55/.45 변형은 검증되지 않은 대조 실험 |
| A9 | [kaiwalya: Competition Solution, v3](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-competition-solution) | `62a7587c2963f61c0dc5783c62a3509e3a25ec75d291208f6087cee7d818b09d` | 셀0 그림 **.958** 및 “v25/v26, current=v25”를 계승. 실제 수집 버전은 v3이므로 그 문구를 이 노트북의 버전 실험으로 해석 불가 |
| A10 | [kiyoshiohno: Template, v27](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template) | `66fbf17addfacbaca201153700220102ce8dda636d6e2b4851a8ac5150a7c66e` | 본문/코드에 0.95 이상 수치 근거 없음. 외부 pipeline 실행기라 이 소스만으로 모델·점수 확인 불가 |

원문·metadata·셀 인덱스는 [수집 디렉터리](../outputs/research/kaggle_code_20261019/selection.json)에 연결되어 있다. 위 후보 10개를 모두 읽었다는 사실과 “0.95 이상을 독립적으로 검증한 10개”라는 주장은 다르다.

## 공통 구성요소를 실제 경로로 분해

### R384: 이름은 384, 현재 추론 입력은 320

A1 셀3과 A3~A6 셀2의 실행 상수는 `IMG=384`, `INPUT_RES=320`, `CROP_MM=140`, `K_EVAL=94`다. 실제 `ARMS`에는 `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` **한 개**만 있다. 체크포인트 SHA256은 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`. 과거 주석의 “5-fold”, 본문의 64장/42창/336px를 현재 실행 설정으로 인용하면 안 된다. 모델 계열은 `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`이고 최종 임베딩을 `LayerNorm → class-specific tanh attention → class-specific linear readout`으로 모은다. `RaptorClassifier.head`에는 명시적인 slice-position embedding이나 study Transformer가 없다. [A1 셀3](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

5개 슬롯은 Sag fluid/nonfluid 26/22장, Cor fluid/nonfluid 18/12장, Ax 18장으로 총96장이다. `Fluid_Sensitive`를 우선하여 중복 series를 피한다. 안전하게 복원한 `_FASTREAD_B64`의 `make_reader_v2`, `build_study_v2`, `pick_indices`에서 확인한 활성 범위는 **2~98%**다. DICOM `ImagePositionPatient · cross(ImageOrientationPatient)`로 정렬하며 실패 시 InstanceNumber를 쓴다. PixelSpacing에 근거한 중심140mm crop을384로 만들고, 추론 시 중앙320을 취한다. 강도는 선택한 series 픽셀의 2/98분위수 창과 ImageNet 정규화다. 고정2.5D triplet으로94개 window를 점수화한다. 이 Raptor 구현에서 방향·좌우 표준화를 R6와 같은 수준으로 한다고 주장할 근거는 없고, 물리정렬/물리crop과 아래 mirror 처리는 확인된다. [A1 셀3의 내장 reader](../outputs/research/kaggle_code_20261019/goodpjw2008__rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/inspection/cell_3_embedded__FASTREAD_B64.txt)

`_mirror_tri`는 Sag 창의 세 채널을 역순으로, Cor/Ax는 가로방향으로 반전한다. **uint8 단계에서 반전한 뒤 강도 LUT를 적용**한다. `infer_pair_probs`는 plain/mirror 확률을 .5/.5 평균한다. R6의 클래스 permutation 및 logit 평균과 동일한 방식이 아니다. [A1 셀3 `_mirror_tri`, `infer_pair_probs`, `gpu_windows`](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

### R224: 별도 해상도로 학습된 두 번째 체크포인트

같은 nartaa dataset의 `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`를 사용한다. 384 크기로 만든140mm field의 **0.8 중심crop(약112mm)을224로 resize**한다. 단순히 R384 모델의 입력 해상도만 낮추는 방법이 아니다. A3/A4/A5/A8은 현재 실행상 **single view**, A1/A2/A9는224도 mirror 확률 평균을 한다. 내부 Raptor 혼합은 `.65 rank(R384) + .35 rank(R224)`다. A6/A7에는 R224 자체가 없다. [A1 셀6](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [A3 셀4](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

### C6/C7: 영상 전체를 학습한 compact ConvNeXt 계열

R6 `fe055406fad6a12f`의 고정 manifest는 A8 셀148에 든 package를 안전하게 복원하여 확보했다. SHA256 `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`. R7 `8c7b666e27dc0e8f` manifest는 공개 dataset에서 읽었다. 둘 다 **ConvNeXt-small,384 render/352crop,seed0의4fold,16epoch,train_windows176,weak pool4344,Gold58 학습 제외,backbone LR1e-4/head LR1e-3**를 기록한다. `freeze_blocks=16` 필드는 있지만 `Net` 구현에서 이는 **OFL encoder 전용**이다. 실제 ConvNeXt 경로 `_timm_encoder`를 frozen으로 해석하면 틀린다. [R6 manifest 복원본](../outputs/research/kaggle_code_20261019/prvsiyan__the-bee-s-knees-final-rsna-push/inspection/bee_R6_FROZEN_V1_MANIFEST.json.txt), [R7 manifest](../outputs/research/kaggle_code_20261019/manifests/r7/manifest.json), [R7 dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)

공통 입력은 `views=v4`,3-slice,`phys_mm=3.5`,eval_stride1이다. `px_cache.series_views`의 v4는 full field와100mm local을 함께 보며, Sag/Ax는 joint·patella·posterior, Cor는 medial/lateral local을 둔다. SKM-TEA anatomy checkpoint에서 얻은 knee/patella anchor와 DICOM geometry를 쓴다. 1.75mm보다 얇은3D stack은 약3.5mm slab로 평균하여 얇은 series가 대부분 버려지거나 과도하게 반복되지 않도록 한다. Sag right knee의 물리좌표 부호를 바꾸고, laterality가 불명확하면 side classifier로 재렌더링하는 경로도 활성 manifest에 포함된다. R6/R7 모두 단순 중앙5장보다 훨씬 촘촘하고 여러 시야를 사용한다. [A3 셀6 `px_cache.py:series_views/StudySlices`, `compact_train.py:window_rows`, `compact_infer.py:CStudies`](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

`Net._build_head/head`는 projection384차원, slot embedding10개, view embedding6개, **상대위치·log spacing·시야 extent·fluid metadata의 linear embedding**, class별 window attention을 쓴다. R7 manifest는 추가로 `spatial_attn=true`: backbone spatial map 위의 class별 query attention(`spatial_pool`,16차원2D sin/cos 좌표)을 활성화한다. 따라서 사용자의 Swin 전체벡터 후단 attention과 달리 **영상 내 위치에서도 소견별 pooling을 하는 경로**다. R7은 `extent=true`의 OA 면적 proxy, report-state head, `syn_decomp=true`도 활성이다. 이는 study self-attention/RoPE와는 다른 구조다. R6의 축약 manifest는 spatial/extent flag를 모두 보이지 않으므로 같은 generic 소스가 있다고 그 flag까지 동일하다고 추정하지 않았다. [R7 manifest](../outputs/research/kaggle_code_20261019/manifests/r7/manifest.json), [A3 셀6 `compact_train.py:Net`](../outputs/research/kaggle_code_20261019/heliosli__this-is-best-public-dataset-new-version-0-9586/inspection/cell_6_compact_train.py.txt)

R7 학습 설정은 `E_lrprior=lr:lr_prior:w`,report-state loss weight.5,matched-pair rank loss.5,rare boosting,mirror augmentation.5,EMA.999,coverage sampling stride3,fastMRI auxiliary weight.3/96windows를 명시한다. `lr_loss`는 likelihood-ratio 감독이며 사용자의 confidence-weighted soft BCE와 같지 않다. R7의 “fastMRI 활용”은 manifest와 generic 학습 코드 양쪽에서 확인되지만, **R6 대비 각 기법의 개별 효과를 보여주는 통제된 ablation은 여기 없다**. R6 축약 manifest의 run tag만으로 동일한 fastMRI 세부값을 확정하지 않는다. [R7 manifest](../outputs/research/kaggle_code_20261019/manifests/r7/manifest.json), [A6 셀0·4](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click)

배포 시4fold logit을 평균하고 Effusion은 이미지에서 산출한 fluid-volume stacker, Baker's는 report-state 분포 stacker로 보정한다. R6 flip-TTA는 Sag 채널순서+상대좌표 부호,Cor/Ax 가로반전,Cor local view2↔3을 함께 바꾸고 medial/lateral meniscus·OA 출력의 순서를 복구한 다음 **logit 평균**한다. MCL은 plain만 쓴다. R7 검토 노트북은 `flip_tta=false`다. “모든 영상에 단순 가로flip 후 평균”으로 복제하면 원문과 다르다. [A3 셀6 `compact_infer.py:mirror_windows/score_v4c/apply_stacker/apply_state_stacker`](../outputs/research/kaggle_code_20261019/heliosli__this-is-best-public-dataset-new-version-0-9586/inspection/cell_6_compact_infer.py.txt)

**학습 공개 범위:** `compact_train.py`230,308자와 관련 모듈은 소스에 포함되어 있어 학습 구조·loss·augmentation 구현을 읽을 수 있다. 그러나 현재 노트북이 training loop를 실행하는 것은 아니며, 실제 target table/state target 생성물·전체 run artifacts가 모두 담긴 것도 아니다. 원문도 report/label table을 제공하지 않는다고 명시한다. 따라서 “원학습 코드는 전혀 없다”도 부정확하고 “노트북만 Run All 하면 동일모델을 처음부터 재학습한다”도 부정확하다. [A3 셀1·6](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

### G: goodpjw 독립 2.5D ConvNeXt-Tiny reader

A1 셀15~17의 별도 모델이다.0.4mm/pixel·384×384=153.6mm field,IOP 기반 in-plane orientation과 IPP 물리순서,series당 최대64장,1/99.5분위수 intensity를 쓴다. six slots는3planes×fat-suppression이며 missing fs/nonfs를 counterpart로 채운다. 중심은 **4~96%에16개**,3-slice windows다. backbone 출력에384차원 projection과slot/연속위치 MLP embedding을 더하고 **2-layer study Transformer → class별 attention pooling**을 한다. 모델 입력 resize는 checkpoint의 `args.res`를 읽으므로 generic `forward(res=256)` 기본값만 보고 정확한 배포 res를 확정하지 않았다. [A1 셀15~17 `load_series`, `window_centres`, `KneeNet`, `load_models`](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

원저자는 report text로 묶은5fold 학습,4개 공개 soft-label table,Gold58 미학습을 기재한다. 배포는 `cnxt_v0_fold[012].pt`의 **3fold**이며 확률 평균이다. 공개 소스에는 preprocessing/dataset/model/inference가 있고 이 노트북에서는 학습하지 않는다. 학습 hyperparameter 전부가 노트북에 없으므로 저자 진술과 실행 소스 확인을 구분한다. 혼합 가중치는 Baker .035,Contusion .03,Medial OA/ACL .025,Medial Meniscus/MCL .02,Lateral Meniscus/Fracture/Lateral OA/PF OA .008,Effusion .004,Synovitis .003이다. **독립 모델 자체의 보고점수는 .929**이며 .958은 다른 대형 계열과의 혼합 결과다. [A1 셀1·18](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

## 개별 노트북에서 실제로 무엇이 달라졌나

### A1 — goodpjw CoAtNet + R6 + 2.5D, v27

활성 경로는 R384 mirror(셀3/4)+R224 mirror(6/7)+C6 flip(9/10)→소견별 rank blend(12)→G의 작은 보정(15~18)이다. Raptor share는 Synovitis.8,Lateral OA.2,PF OA.6,Baker.35,나머지.5. Raptor 내부는 .65/.35다. G stage가 시간/추론 오류로 실패하면 Part A 제출을 유지하는 구조이므로 결과 receipt 없이 모든 stage가 실제 실행됐다고 단정하면 안 된다. [A1 셀12·18](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

비교 증거는 비교적 구체적이다. 저자는 단일CoAtNet .949→G의 소견별 혼합 .950,이후 S6계열 .954→R7계열 .956→R6 flip .957→소견별 가중치 .958을 기록했다. 그러나 이 전체 경로는 학습·데이터·추론·가중치가 같이 바뀌므로 각각의 순수 인과효과가 아니다. **G full weight를 .956 base에 더하면 .954로 하락**,S6-distilled reader는 gold58 .931이어도 blend .953이었다고 기록한다. v25/v26은 R7을 추가해도 둘 다 .958이며 별도 이득이 입증되지 않았다. 채택할 교훈은 독립성·실제 holdout 검증이고, “모델을 많이 추가할수록 오른다”가 아니다. [A1 셀1](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

### A2 — Does S6 help next to R6, v8

제목에 S6가 있지만 **현재 v8 활성 코드는 R6만**이다. A1과 동일한 R384/R224/C6/G를 쓰고,셀12의 Raptor weight만 ACL.3,Effusion.7,Fracture.8로 바꾼다. 나머지는 A1의 가중치다. 이는 현재 소스에서 S6를 찾는 것과 과거 실험 결과를 읽는 것을 분리해야 하는 사례다. [A2 셀9·12](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

실험 기록: S6를 compact slot의30%로 추가한 v4/v5는 **.958→.957**. Gold58에서 큰 차이가 난3개 소견의 weight를 바꾼 v6도 **.957**. Gold58에서12개 weight를 최적화하고.1~.9로 제한한 v7은 **.955**. 본문은 Gold58의 소견별 양성9~35건에서 나온 차이로 weight를 맞추지 말라고 결론짓는다. 58건에서 내부 AUC가 높은 구성과 더 나은 LB 구성이 같지 않다는, 사용자의 후속실험 설계에 직접 중요한 **반증 자료**다. [A2 셀1](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

### A3 — heliosli best public dataset .9586, v7

R384 mirror+R224 single+C6 flip을 소견별 rank blend하고 마지막에 G를 A1과 동일한 작은 weight로 넣는다. 현재 셀8은 Raptor share **Lateral OA.15,PF OA.65,Synovitis.85,Baker.30**,나머지.5다. A5의 .2/.6/.8/.35와 다르다. 셀0의 일반적인 “.5/.5” 요약보다 실제 셀8을 우선했다. 셀9의 `_PB_FILES`에는 G의3개 파일이 들어 있고 SHA를 점검한 뒤 추론한다. [A3 셀2·4·6·8·9](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

새 학습을 하는 노트북이 아니라 이미 공개된 세 계열을 재조합한다. 제목의 .9586을 읽을 수 있지만 A5→A3에서 가중치4개 변화와 G 추가의 각각 독립된 before/after는 없다. “G 때문에 .0006 상승” 또는 “이 소견 가중치가 일반화 최적”으로 해석하면 안 된다.

### A4 — heliosli old version .9573, v2

R384 mirror+R224 single+C7 no-flip. 최종 `.5 C7 + .325 R384 + .175 R224`의 percentile ranks다. dataset slug는 `rsna-knee-r6-frozen-fe055406`인데 `BUNDLE_ID=8c7b666e27dc0e8f`,주석은 그 dataset v2가 R7이라고 명시한다. **이름·최신 dataset 선택만으로 R6/R7 재현을 보장할 수 없다.** manifest를 고정해야 한다. [A4 셀0·6·8](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use)

A6보다224 체크포인트를 더 사용하는 구조는 확인되지만 제목 .9573을 A6의 공식 점수와 같은 조건에서 잰 상승분으로 직접 빼지 않는다. 학습 스크립트는 C6/C7 공통 소스이고 실제 배포 bundle만 다르다.

### A5 — heliosli Blend mine, v1

R384 mirror+R224 single+C6 flip. 최종 Raptor share는 Synovitis.8,Lateral OA.2,PF OA.6,Baker.35,나머지.5이며 G는 없다. C6 bundle을 별도의 `rsna-knee-r6-frozen-fe055406-w1`로 지정하여 원래 dataset 최신버전이 R7로 바뀐 문제를 피한다. A1/A2는 이 recipe를 .958로 인용하지만 이 수집본의 metadata에서 공식 숫자를 검증한 것은 아니다. [A5 셀6·8](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters)

사용자가 참고할 수 있는 것은 서로 다른 모델의 소견별 강점을 이용하는 **고정 rank-blend 구조**다. 가중치가 어떤 독립 validation 과정으로 결정됐는지는 여기만으로 재현되지 않는다. 셀8은 Raptor가 Gold58에서 선택되었으므로 weight fitting에 쓸 깨끗한 labelled set이 없다고 주석을 남긴다. Gold58이 gradient 학습에서 빠졌다는 주장과 개발상 완전히 미접촉이었다는 주장을 혼동하면 안 된다.

### A6 — heliosli .957 latest one-click, v1

가장 단순한 C7 조합이다. **R384 mirror 하나+C7 4fold**,각 소견별 percentile rank를 .5/.5로 평균한다. R224/G는 없고,같은 데이터 decode를 공유하여 C7의4fold를 실행한다. 현재 recipe의 .956 설명과 제목 약 .957은 구분해야 한다. 활성R7 모델은 위 manifest의 spatial pooling·report-state·extent·fastMRI auxiliary 등을 갖지만 이 노트북이 그것들을 새로 학습하지 않는다. [A6 셀0·2·4·5·6](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click)

### A7 — evgendvorkin RSNA Versia 7, v22

긴 설명은9개 과거 공개 해법의 진화를 다루지만 현재 추론은 **A6와 같은 R384+C7 .5/.5**다. 셀3/7/9/11과 A6 셀2/4/5/6을 Python AST로 비교했을 때 주석/docstring을 제외한 실행 구조가 동일했다. 자체 학습 향상 사례로 한 표를 추가하지 않고 **문서화·재현 파생물**로 분류한다. 본문도 .956은 원저자의 점수이며 자신의 재현 점수는 별도 확인 대상이라고 명시한다. [A7 셀1·3·7·9·11](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7)

과거DINO·RadImageNet·S5/S6 등의 설명이 현재 활성 ensemble에 모두 포함된다고 읽으면 안 된다. 채택 가치는 계보 이해와 inference 재현성이다.

### A8 — prvsiyan bee's knees, v47

151개 셀 중 현재 `code`는 **셀148 하나**다. 과거DINO/A5/RadImageNet/Raptor·독립reader 실험은 raw 또는 markdown 셀로 보존되며 실행되지 않는다. 셀148의 base91/LZMA package를 바이트로 복원하고7개 payload SHA256을 모두 확인했다. 복원된 `runner.source.py`는 고정 upstream notebook의 셀을 사용하고 `blend55_patch.py`로 최종 가중치만 바꾼다. 이 현재 recipe는 **R384320px paired + R224224px0.8crop single + C6 flip**,final weights **C6.55,R384.2925,R224.1575**다. [A8 셀145·148·150](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push), [복원된 blend patch](../outputs/research/kaggle_code_20261019/prvsiyan__the-bee-s-knees-final-rsna-push/inspection/bee_blend55_patch.py.txt)

셀149에는224 mirror 설명이 남아 있지만 마지막 셀150과 활성 package는 **추가mirror inference 없이** 가중치만 바꾸는 실험을 명시한다. 셀145는 upstream `heliosli/...this-is-new-version-0-9576-in-one-click?scriptVersionId=358332966`의.957을 인용하면서 **자신의 새 공식 점수는 없다고 명시**한다. 따라서 고점수 후보 목록에서 상세검토했으나 새0.95 달성 사례로 세면 안 된다. 반복가능한 source pin과 baseline 대비 동일예측 비교를 중요하게 다룬다는 점은 참고할 만하다.

### A9 — kaiwalya Competition Solution, v3

R384 mirror+R224 mirror+C6 flip+G의 작은 가중치라는 A1 recipe다. A9 셀1/3/5/7/9/10/11/12가 A1 셀3/6/9/12/15/16/17/18과 **주석/docstring을 제거한 AST에서 동일**했다. figure는 “this notebook's reader”,“versions25/26 .958”라는 A1 계보의 설명까지 계승한다. 이 사용자의 독립적인 v25/v26 학습 실험이라고 볼 수 없다. [A9 셀0~12](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-competition-solution)

단순 제목 요약에서는 놓치기 쉬운 파생 관계다. 동일모델·동일혼합을 재실행한 노트북을 독립ensemble 다양성으로 추가하면 실질적인 새 정보가 없다.

### A10 — kiyoshiohno Template, v27

셀2는42줄 launcher다. `rsna_knee_manifest.json` 중 `bundle_id=rsna-knee-local-logic`,`runner=rsna_knee_pipeline.py`인 단일 dataset을 찾고 그 runner를 실행한다. 노트북 자체는 해상도·crop·pooling·학습·혼합을 정의하지 않으며 SHA 검증도 하지 않는다고 문서에 명시한다. metadata에는G/C6/nartaa/code dataset과 **빈 dataset ref**가 있다. 공개 launcher만으로 그 빈 항목의 실제 pipeline/버전/성능을 복원할 수 없다. [A10 셀0~2](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template), [metadata](../outputs/research/kaggle_code_20261019/kiyoshiohno__rsna-knee-template/metadata.json)

참고할 아이디어는 코드와 추론환경을 dataset으로 버전화하는 배포 패턴이다. 어떤 모델이 점수를 올렸는지 판단할 증거는 부족하다. 이 한계를 채우기 위해 무관한 private 입력에 접근하거나 임의 실행하지 않았다.

## 사용자의 현재 방식과 비교

사용자 R01 **.771432는 Gold58 개발 OOF**이고 위 공개값은 대부분 저자보고 public LB다. 둘을 빼서 “.187 차이”처럼 효과를 계산할 수 없다. 사용자 실험은 동일Gold58의3fold와3seed이고,외부 공개model의 training/selection split은 다르다. 공개모델을Gold58에 그대로 대입해 나온값도 기존OOF와 동일한 검증조건이라고 볼 수 없다. [현재 실험 결과](followup_experiment_results.md)

| 축 | 사용자 현재 | 반복하여 확인된 공개 고점수 recipe | 의미 / 다음 판단 |
|---|---|---|---|
| 영상학습 | Swin-T 대부분 고정특징,일부weak-only1epoch FT | C6/C7 ConvNeXt-small 영상encoder와 head16epoch,4fold | 현재의128차원 후단만 조정하는 탐색과 학습량·표현 적응의 규모가 다름. 같은 자료/OOF에서 단계적으로 비교할 근거 |
| 정보 범위 | 중심25~75%,5centers,각5single neighbors,224px | Raptor2~98%,96slice/94triplets; G4~96%/16triplets; C6/C7 physical dense coverage+full/local | MCL·lateral meniscus 등 주변 구조가 입력에서 잘리는지 먼저 확인. Raptor 상수를 복제하기보다 wider-span·physical coverage를 분리 실험 |
| 물리 크기 | IPP/IOP 순서는 사용,PixelSpacing 기반mm crop 없음 |140mm /153.6mm 표준FOV,100mm local,3.5mm through-plane | mm기반 위치RoPE를 썼다는 것과 입력의물리scale 정규화는 다른 개선축 |
| 좌우/시야 | 명시적 laterality normalization 부족 | C6/C7 side-canonical geometry와knee/patella anchor,불명확side분류; 해부학별mirror | 정렬방향·좌우·crop을 함께 일관되게 처리해야 함. 임의hflip을TTA로 추가하지 않음 |
| pooling/position |768→128,self-attention/RoPE/gate/cross;Swin global feature 이후 | Raptor class attention;G384dim2layerTransformer;R7 spatial+window class attention+geometry | 이미 attention은 있으므로 “attention의 유무”가 핵심 격차라는 근거 부족. 입력의 coverage와 spatial pooling 위치를 구분 |
| 약라벨 |Gold58+weak4292,UNK weight0,confidence-weighted softBCE |C6/C7 weak4344,Gold gradient 제외,report-state/정도/extent/ranking;Raptor는여러label source+OAI | 코드만으로teacher우열은 못정함. 같은 표본에서known/unknown·정도기준·label source차이를검증해야함 |
| 외부자료 |현재경로경쟁영상중심 |nartaa OAI,compact SKM-TEA anchors,R7 fastMRI auxiliary | 외부자료와모델훈련효과를후단구조효과로오인하지않기 |
| ensemble |R01확률50:50,실제E09.5/S11.25/S12.25 |독립학습계열의rankblend,224/320multi-res,TTA,class weights | 다른구조가아닌동일feature family만섞는것과차이. 다만A1/A2의추가모델악화사례로각arm검증필수 |

사용자 코드의 근거: [spatial_core.py](../experiments/spatial_core.py), [weak_core.py](../experiments/weak_core.py), [followup_experiment_results.md](followup_experiment_results.md). 기존R01의개선은같은OOF체계안의유효한관찰이며,공개점수와동일척도로직접비교하지않는것이그관찰을약화시키지않는다.

## 이 표본에서 채택할 것과 추정하면 안 되는 것

1. **우선할 증거 기반 변화:** 물리 FOV·좌우 정규화·넓은 slice coverage를 원영상 샘플로 확인하고, 변경별 OOF를 고정해 비교한다. 그다음 backbone 미세조정의 충분한 학습량, 가능하면 spatial feature pooling을 검토한다. 소스상 차이가 크다는 근거와 실제 점수 개선 예측은 별개다.
2. **의미 있는 구조 차이:** R7은 report-state/정도와 영상 내 class별 spatial pooling이 있다. 공개 입력 label/보조자료까지 같아야 그 학습을 정확히 재현할 수 있다. generic 코드에 있는 모든 옵션을 R6/R7에 적용됐다고 기재하지 않는다.
3. **앙상블은 독립성부터:** A1의 G 단독 .929가 좋은 초기 보완이었지만 강한 base에 큰 weight로 추가하면 오히려 하락했다. 복사 노트북의 개수를 독립 reader 수로 세지 않는다.
4. **Gold58에 대한 과적합 증거:** A2의 Gold 최적 weights가 LB .958→.955로 하락했다. 사용자의31개 개발 결과에서 소견별 최고모델을 선택해 혼합하는 것은 별도 검증 없이 성공으로 기록하기 어렵다.
5. **점수 provenance 유지:** 제목 .9586, 과거버전 .958, 타인 .956 인용, 새 점수 없음을 각각 유지한다. 버전·checkpoint hash·실제 사용 manifest·stage 성공 receipt가 없는 실행은 같은 recipe인지조차 불명확할 수 있다.

이는 새 훈련·공개모델 실행을 수행한 결과가 아니라, 고정된 공개소스와 manifest를 읽어 얻은 증거 검토다. 제안한 변경으로 사용자의 독립검증 성능이 얼마나 오를지는 이 자료만으로 산출할 수 없다.
