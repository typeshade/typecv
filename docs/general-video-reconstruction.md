# 일반 게임플레이·시네마틱 재구성: 감사와 단계별 개발

이 문서는 요청 전체의 완료 보고가 아니다. 먼저 현재 체크아웃을 감사하고,
기존 구현을 재사용할 수 있는 경계와 첫 구현 범위를 정한다. 동작을 직접 실행한
근거, 원격 소스를 읽은 근거, 제안만 있는 항목을 구분한다.

후속 P1의 첫 구현인 [보정된 상대 회전과 미삼각측량 관측 보존](rotation-tracking.md)은
별도 문서와 검증 기록으로 추가했다. 아래 표의 감사 당시 판정과 후속 구현의 지원
조건을 구분하며, 이 기능도 일반 영상 전체·전역 방향 최적화·무제한 관측 보존의
완료를 뜻하지 않는다.

감사는 2026-10-23의 `e40c878` 작업 트리에서 수행했다. 아래의 **로컬 판정과
기존 측정은 그 시점의 기록**이며, 이 문서가 반영된 최신 `main` 전체의 지원
상태를 뜻하지 않는다. 이후 머지 준비에서는 `2c45e7c` 기반의 별도 작업 트리에
감사 문서·fixture·평가 스크립트만 옮겼다. 이전 세션의 게임 프리셋·마스크·UI 변경은
원래 작업 트리에 보존했고 이 변경에 포함하지 않았다. 최신 구현의 일반 현황은
[기존 재구성 현황](reconstruction-status.md), [다중 영상](multi-video.md),
[KLT](klt-tracking.md), [MVS](dense-mvs.md) 문서와 함께 읽는다.

## 기준과 첫 단계

- 로컬 HEAD: `e40c878809370719faef7dac3e34b80a92c4ce3d`, 브랜치 `work`.
- 이전 세션의 게임 프리셋·HUD 제외·COLMAP 출력 변경이 미커밋 상태로 있다.
  따라서 HEAD와 작업 트리의 기능은 같지 않다. 이 변경은 보존한다.
- 조사한 원격 TypeCV: `2c45e7cc9146d675f9a3f999a6875124afe6e70b`.
  로컬보다 뒤의 KLT, 다중 영상, MVS 구현을 미구현으로 다시 만들지 않는다.
- 조사한 원격 Radiance: `3e4efb3f8b6ba2f1c5d9f21f070493c5c3fdfdd0`.
  로컬 Radiance 체크아웃은 없다. 소스 읽기와 실행 검증을 구분한다.
- 환경: Linux x86_64, Node `24.19.0`, npm `11.9.0`.
  로컬 TypeShade pin은 `212d26d2fb9fb6dd39b72674e537653fb0f7e888`이다.
  실제 GPU 여부와 실행 결과는 측정 기록에 적는다.

**첫 구현 범위는 감사·재현·평가 기반이다.** 기존 공개 API, mapper,
브라우저 입력과 GPU 커널의 동작을 바꾸지 않는다. 새로운 라이브러리 없이 기존
`reconstructIncremental`, `triangulateTrack`, CPU/GPU 특징점·매칭·기하 API와
`scripts/geometry-comparison.mjs`의 지도 단위 Sim(3) 평가를 재사용한다.
새 저장소, 이슈 등록, PR 생성·머지, 배포는 이 단계에 포함하지 않는다.

아래 판정은 이번 소스 조사와 실행 기록의 범위에 한정한다.

## 지원 상태 판정 규칙

`지원`은 표의 제한된 입력 계약에서 구현과 직접 실행 근거가 있는 경우다.
`부분 지원`은 일부 계약만 충족한 경우다. `미지원`은 조사한 API·제어 흐름에
필요한 경로가 없는 경우다. `확인 불가`는 소스·실행·실제 장치 근거가 부족한
경우다. 원격 소스의 존재와 PR의 테스트 설명은 로컬 실행 결과를 대신하지 않는다.

## 설계 원칙과 다음 단계 인터페이스

`Pose`, `Camera`, `FeatureTrack`, `Landmark`, `SparseReconstruction`을 유지한다.
기존 연속 배열과 TypeShade 소유권을 가로지르는 범용 장면 계층을 새로 만들지 않는다.
원격의 `HostScene`, `MapHistory`, view graph와 canonical ID를 기준으로 후속
관측 상태 메타데이터를 붙이는 방향을 먼저 검토한다. 아래 항목은 **제안**이며
새로운 공개 API가 이미 있다는 뜻이 아니다.

| 기존 관계 | 추가로 필요한 정보 | 확정하면 안 되는 정보 |
| --- | --- | --- |
| view / Camera | 원본 frame ID, decode PTS·time base, edit time, shot-local time, intrinsics group, calibration provenance | 컷 간 실제 사건 시간·속도 |
| 영상 좌표 / keypoints | 원본 크기, 유효 영역, source→analysis affine transform, projection region, mask provenance | 스코프 안팎의 단일 투영 |
| registered view / Pose | component ID, full pose / orientation only / unresolved, 근거·관측 분포 | orientation-only에 임의의 `t=0` 삽입 |
| track / Landmark | raw·verified·excluded 관측, deferred reason, 구조 생성에 쓴 view IDs, 재평가 요청 | 새 카메라만 이동했다는 이유로 미관측 점의 깊이 확정 |
| component | 독립 gauge, scale provenance, Sim(3)/SE(3) 연결 제약과 지지 관측 | 편집 순서·등장 인물만으로 지도 병합 |
| object ID | association, rigid/deforming/unresolved, local shape, relative pose, optional world pose | segmentation만으로 형상·월드 궤적 확정 |
| 결과 근거 | observed geometry / assumed calibration or constraint / learned prior / ground truth | 학습 사전을 관측 정답으로 표시 |

위 상태는 직교 축으로 둔다. `camera=registered`이면서 `new depth=deferred`일 수
있고, `object=identified`이면서 `shape=unresolved`, `world trajectory=ambiguous`일
수 있다. 보류·제외 이유는 관측을 삭제하는 이유와 구분한다. 공통 마스크와 프레임별
마스크는 원본 pixels를 보존하고 모델 소속 관측만 선택한다.

좌표 계약은 TypeCV의 기존 규약을 따른다: row-major `R`, world-to-camera
`q=RX+t`, camera center `-Rᵀt`, +Z 앞쪽, 픽셀 중심은 정수 좌표다. COLMAP
출력은 기존 half-pixel 변환을 쓴다. 새 깊이 fixture의 깊이는 카메라 +Z 좌표이며
광선 거리와 구분한다. 단안 지도 단위는 외부 스케일 근거가 없는 한 임의 단위다.

### 다음 API 변경의 최소 단위 (아직 구현하지 않음)

기존 `RegisteredView`/`Landmark`에 미확정 숫자를 넣지 않고, 기존 view/track ID를
참조하는 별도 evidence 메타데이터를 붙이는 방안을 우선 검토한다. 예를 들어
`orientation-only`는 reference view와 `R`을 갖고 **`t`나 depth는 갖지 않는다**.
`map-registered`만 기존 `RegisteredView`와 component ID를 참조한다. `unresolved`는
관측 ID와 보류 이유만 남긴다. 이 판정은 숫자 confidence 확률을 만들어 붙이는
작업과 구분한다. covariance/interval의 타당성 검증 전에는 uncertainty 수치를
생성하지 않는다. 기존 `IncrementalResult`의 필드를 삭제하거나 의미를 바꾸지 않는다.

P1은 다음 순서로 작게 나눈다.

1. 기존 calibrated unprojection과 TypeShade/CPU SO(3) 수학으로 rotation-compatible
   모델을 적합하고, 2D 잔차·분포와 E/H 모델의 적합을 함께 반환한다. 이 모델이 잘
   맞아도 병진 0이라고 증명한 것은 아니므로 `translation unobservable`을 허용한다.
2. 기존 지도에서 충분히 분포한 점의 PnP가 성립하면 full pose를 등록한다. 새 점의
   깊이 생성은 별도 decision이며 orientation-only pose를 삼각측량에 보내지 않는다.
3. raw/2D-verified/structure-verified edges를 구분해 관측을 보존한다. 더 뒤의 등록
   카메라와 **그 점의 재관측**이 생겼을 때만 기존 `triangulateTrack`을 재호출한다.
4. rotation-compatible edge에 temporal track이 유지되는지, 초기화를 지연한 상태가
   다음 입력을 받아 계속되는지 검사한다. 현재 offline 재실행 성공으로 이 게이트를
   대체하지 않는다. GPU 수치 경로는 기존 CPU/TypeShade oracle과 차등 검증한다.

P2의 첫 단위는 알려진 resize/crop transform과 한 개의 일관된 projection region
선택이다. 이후 구간별 공유 log-focal scale와 필요 시 smoothness를 검토하며, 검출된
조준 전환·컷을 가로질러 smoothness를 강제하지 않는다. 스코프 안팎을 별도 virtual
views로 만들더라도 같은 카메라 시점의 회전·중심을 묶지 않은 채 두 독립 카메라로
BA에 보내면 가짜 baseline을 만들 수 있다. 그래서 초기 범위는 단일 region 선택이며,
공동 region 최적화는 별도 제약 설계 이후다. 카메라 자세 이동은 K 변화와 별도로
허용한다. 잘 알려진 K 모드와 영상으로 K를 추정하는 모드는 결과 provenance가 다르다.

강체 객체의 영상은 `x ~ K · T_world_to_camera(t) · T_object_to_world(t) · X_object`로
다룬다. 대상만 보이면 주로 두 변환의 곱이 제약된다. 배경으로 카메라를 추정했어도
대상의 크기·깊이·운동에 별도 모호성이 남을 수 있으므로, world pose를 항상 채우지
않는다. 알려진 크기/지면/깊이/운동 사전의 사용 근거를 명시한다. 사람은 root movement와
joint/deformation state를 분리하고 전체를 하나의 rigid cloud로 승인하지 않는다.

## 단계별 순서와 완료 조건

| 우선순위 / 로컬 작업 단위 | 변경·기존 기능 재사용 | 합격 기준·주요 실패 조건 | 다음 단계 입력 |
| --- | --- | --- | --- |
| P0 현황 감사·재현·평가 (이번 단계) | 기존 API를 호출하는 fixture·probe, 원격 차이 기록 | 정답 누출 없음, 지도별 단일 정렬, 미지원·보류를 성공으로 집계하지 않음 | 재현 입력·계약·기준 측정 |
| P1 추적과 구조 생성 분리 | 원격 KLT·pair verification·PnP·MapHistory 재사용; calibrated rotation fit와 deferred observations 검토 | S02 방향과 깊이를 분리, S03 재관측된 점만 이후 삼각측량; H 적합만으로 회전 판정 금지 | 관측 역할·불확실성 상태 |
| P2 줌·조준·projection region | 기존 camera groups·shared focal·focal search·resize 변환 재사용 | S06 깊이 미정, S07–09 전진·줌 보상 방지; known K와 inferred K 분리 | 구간별 K·유효 영역·change points |
| P3 샷·장소·재지역화 | 기존 retrieval/view graph/component mapper를 확장 | S15 샷 간 공동 초기화, S16 독립 gauge, S17 반복 에셋 오병합 검사; Sim(3) 근거 없는 병합 금지 | edit timeline·component links |
| P4 강체 객체 | 배경·객체 관측 보존, 기존 geometry를 객체 좌표에 재사용 | S10 최대 RANSAC 지지를 배경으로 단정하지 않음; S11–13 상대 시차·별도 스케일 모호성 유지 | object-local shape / relative motion |
| P5 사람·변형·Radiance | 제한된 강체 부위부터; 골격/변형 사전 도입은 별도 의존성 평가 | S14 비강체를 단일 강체로 승인하지 않음; learned prior 근거 표시 | 동적 렌더 표현·관측/사전 구분 |
| P6 측정된 GPU 병목·실제 클립 | 원격의 streaming/chunks·kernel accounting·actual-device info 우선 재사용 | 단계·전체 시간, 전송·동기화·메모리, CPU/GPU 기하 오차; native/software 분리 | 실제 게임·시네마틱 통합 기록 |

동적 전경·HUD·관측 상태처럼 앞 단계의 카메라 추적을 보호하는 작업은 위 번호보다
먼저 들어갈 수 있다. P1 이후 구현은 오래된 로컬 mapper를 대규모 수정하기 전에
원격 구현과의 통합 기준을 먼저 정해야 한다. 이번 단계는 이를 자동 병합하지 않는다.

## 입력 현상별 관측 정책

| 현상 | 사용할 수 있는 근거·검출 | 보류·분리 기준 |
| --- | --- | --- |
| 순수 회전·먼 배경·평면 | calibrated rotation / H / E 적합, 관측 분포, 삼각측량 각도·부호·오차 비교 | H 우세는 회전의 증명이 아님; 새 깊이·병진은 식별 가능할 때만 |
| 전진·대상 접근·줌 | 배경과 대상의 독립 motion, intrinsics group, zoom change point | 전체 자유 K 최적화로 오류를 숨기지 않음; 확대율 UI를 정확한 f로 쓰지 않음 |
| 스코프·레터박스·크롭 | 일관된 투영 영역과 source→analysis 변환, per-region observations | 서로 다른 투영의 특징점을 섞지 않음; 무기·테두리는 배경과 분리 |
| HUD·자막·미니맵 | 공통/프레임별 마스크와 provenance, 화면 고정성과 기하 잔차 | RGB 보존; 화면 고정만으로 static world 판정 금지 |
| NPC·차량·추적 대상 | instance association + 실제 motion consistency | 최대 지지 집합=배경 가정 금지; 대상 상대 시점이 일정하면 형상 보류 |
| 사람·바퀴·문·의복 | 부위별 consistency, 관절/변형 모델의 적합성 | 비강체 부분을 rigid body에 합치지 않음 |
| 그림자·파티클·반사·투명·스카이박스·빌보드 | 시간/시점에 따른 consistency와 관측 지지 | semantic class만으로 일괄 제거하지 않음; finite geometry 근거가 없으면 보류 |
| 반복 텍스처·대칭·재사용 에셋 | retrieval 후보 이후 지역 분포·복수 지점의 기하 검증 | 외형/semantic 유사성만으로 같은 장소 확정 금지; 구별 불가능하면 독립 유지 |
| blur·temporal accumulation·frame generation·LOD·압축 | 검출/매칭 반복성, forward-backward consistency, 관측 품질 | 가짜 일관성·저품질 관측은 confidence 하향 또는 보류; 전부 발생한다고 가정하지 않음 |
| 컷·dissolve·섬광·급회전·가림 | appearance 후보 + 기하·tracking 연속성, 혼합 프레임 표시 | 후보 컷을 확정 컷으로 취급하지 않음; cut/dissolve에 속도·단일 카메라 강제 금지 |

## 검증 시나리오와 단계별 게이트

| ID | 입력 조건 | 첫 단계 / 후속 합격 기준 |
| --- | --- | --- |
| S01 | known K, 정적 비평면·텍스처, 이동 | 단일 component Sim(3) 이후 회전·위치·점 오차; correspondence와 RGB 실험 구분 |
| S02 | known K, 지도 없는 중심 고정 회전 | 잘못된 구조 초기화 0; orientation tracking은 별도 후속 게이트 |
| S03 | 지도 등록 가능한 회전 후 이동, 점 재관측 유/무 대조 | 지도 포즈 등록과 새 구조를 구분; 재관측 없는 점은 미확정 유지 |
| S04 | 정지 대상 주변 camera-center 공전 | 실제 중심 변화·삼각측량 근거를 사용; 게임 고유 반지름 가정 없음 |
| S05 | 평면·먼 배경·저시차를 별도 fixture | 가짜 병진·깊이 확정률과 보류 사유 기록 |
| S06 | 중심 고정 줌·회전+줌, known K / inferred K 별도 | 새 깊이 0; K를 제공한 결과를 줌 추정 성능으로 집계하지 않음 |
| S07 | 이동+연속 줌 | 각도·f·이동의 공동 오차와 observability; 부족하면 보류 |
| S08 | 급변 K+자세 변경 | change point를 가로지르는 잘못된 smoothness 강제 없음 |
| S09 | 내부/외부 projection 다른 스코프 | 영역별 K와 source transform round trip; 혼합 관측 기하 거부 |
| S10 | 동적 전경 우세·작은 배경 | foreground로 카메라가 고정되는 잘못된 확정 검출 |
| S11 | 차량과 동일한 상대 거리·각도 | 배경 시차와 객체 형상 근거를 구분 |
| S12 | 차량의 다른 상대 시점 | 객체 좌표 형상·상대 motion과 별도 scale 상태 |
| S13 | 대상만 관측 | 월드 camera/object 분리는 ambiguous로 유지 |
| S14 | 걷기·의복 변형 | 단일 강체 부적합, root·관절·사전 추정 구분 |
| S15 | 각 샷 내부 정지/회전, 샷 간 baseline | cross-shot 대응으로 공동 초기화; 샷별 SfM 성공을 요구하지 않음 |
| S16 | 공통 관측 없는 다른 장소 | component별 독립 gauge; 가짜 중간 경로 없음 |
| S17 | 재등장 장소와 다른 장소의 재사용 에셋 | false merge와 missed link 함께 측정 |
| S18 | dissolve/flash/pan/occlusion 대조 | 후보·확정·혼합 상태의 confusion matrix |
| S19 | time jump·재등장·속도 변경 | edit time·source time·object state 분리; 컷 간 실제 속도 미정 |
| S20 | 가림·저텍스처·재관측 | 원 관측을 보존하고 재평가; relocalization/component 결과 명시 |

대규모 데이터·실제 클립의 합격치는 첫 렌더 기준 측정 이후 고정한다. 사람·차량의
semantic 식별, 컷 검출 모델, 비강체 모델의 도입은 첫 단계에서 검증된 것으로
집계하지 않는다. 공개 합성 데이터의 GT는 evaluator에만 제공한다. MVS-Synth는
외부행렬의 det=-1 및 infinite depth를 확인하고, TartanAir도 원 camera/depth
규약을 변환한 뒤 사용해야 한다. 이번 단계에서는 두 데이터를 다운로드하거나
성능 근거로 사용하지 않는다.

## 실행 기록과 현황 표

### 최신 main에 반영하기 전 재검증

사용자의 머지 요청 후 `2c45e7cc9146d675f9a3f999a6875124afe6e70b` 기반의
별도 작업 트리에서 신규 회귀 테스트 5개를 그대로 실행해 통과했다. 기존 구현을
수정하지 않고 같은 API로 실행한 [기하 재검증](experiments/general-video/integration-geometry.json)도
12개 제한 게이트를 통과했으며 잘못된 구조 확정은 0개였다. 이는 전체 시나리오
지원 판정이 아니라 알려진 K·제공된 대응점 조건의 검사다.

[RGB 재검증](experiments/general-video/integration-rgb-software.json)에서도 이동
5/5프레임·234점을 얻었고, 지도별 단일 Sim(3) 이후 최대 회전 오차는 약 2.515°,
카메라 중심 RMSE는 0.003162 fixture 단위였다. 순수 회전과 회전+줌은 각각
초기화를 거부해 점을 생성하지 않았다. 알려진 K, 추정한 RGB 특징점·대응점,
마스크 없음, SwiftShader 소프트웨어 WebGPU 조건이다. 이 실행은 다른 검사와
동시에 수행했으므로 이전 측정과의 성능 비교나 가속 근거로 사용하지 않는다.

[머지 전 검증 기록](experiments/general-video/integration-validation.json)에
명령·소스/런타임 해시와 원래 작업 트리 보존을 기록했다. 코어 254개(신규 5개
포함), 타입·포맷 검사, 라이브러리·77페이지 웹사이트 빌드, Astro 검사와 정적
링크 검사를 통과했다. 변경하지 않은 전체 TypeShade CPU 모음의 로컬 재실행은
중단했으며 전체 `npm run check` 통과로 집계하지 않는다. 기준 SHA의 기존 CI는
Node 22/24와 WebGPU 모두 통과했다. 새 PR의 전체 CI 결과는 GitHub에서 별도로
확인하며, 이 로컬 기록으로 통과를 대신 주장하지 않는다.

### 감사 당시 작업 트리의 근거

근거 파일은 [기하 측정](experiments/general-video/geometry.json),
[RGB 측정](experiments/general-video/rgb-software.json),
[기존 영상 프레임의 cold/warm 측정](experiments/general-video/pyramid-software.json),
[원격 소스·PR 스냅샷](experiments/general-video/remote-evidence.json),
[실행 명령·소스 해시·기존 변경 보존 기록](experiments/general-video/validation.json)이다.
로컬 경로는 HEAD에 이전 세션의 미커밋 변경을 더한 작업 트리를 뜻한다.
원격 열의 코드는 조사한 SHA에서 읽었고 **실행하지 않았다**.

이번 단계에서 `npm test` 68개(신규 5개 포함), `npm run test:shade` 54개가
통과했다. 후자는 TypeShade CPU IR 검증이다. typecheck·format check·라이브러리
빌드·웹사이트 43개 페이지 빌드와 정적 링크 검사도 통과했다. 전체 UI 테스트와
전체 WebGPU 테스트 모음은 이번 단계에서 다시 실행하지 않았으며, 아래 RGB
측정 3개와 기존 파이프라인의 cold/warm 측정은 SwiftShader에서 직접 실행했다.

| 조사 항목 | 로컬 판정과 실제 경로·심볼 | 테스트·직접 실행 근거 | 원격에서 이미 있는 기능 / 남은 경계 |
| --- | --- | --- | --- |
| RGB 디코딩 | **부분 지원**: `website/src/components/SfmLab.tsx:sampleFrames`, HTMLVideo seek·첫 loadeddata 대기; `scripts/prepare-sfm-video.py:main`의 선택적 OpenCV 어댑터 | `tests/sfm-video.test.ts`; RGB probe에서 PNG decode 실제 실행. 웹 영상 디코딩은 앞 세션의 `website/site.test.mjs` 검증이며 이번 새 측정은 PNG 입력 | `website/src/workers/decode.worker.ts`·PR #111 있음. 이번 감사에서 전체 decoder 동작은 실행하지 않아 **확인 불가** |
| 프레임 선택·시간 | **부분 지원**: `website/src/lib/sfm-input.ts:planVideoFrames, videoPairSchedule`; 구간·균등 추출·temporal gaps. 저장 time은 목표 seek 시간이며 원본 decode PTS가 아님 | `tests/sfm-video.test.ts`의 구간·끝점·예산 검사; 편집 시간/사건 시간 자동 복원은 **미지원** | `video-sampling.ts:sampleTimestamps`와 phase/KLT gate #116/#130 존재. 픽셀 이동 gate를 관측 가능성 판정으로 대체해 쓰면 안 됨 |
| 특징점·매칭 | **지원**(계약 내): `src/gpu/features.ts:extractFeaturesGpuPyramidResident, matchResidentFeaturesGpu`; float descriptor 어댑터 `src/gpu/float-matching.ts` | `tests/typeshade/pyramid-features.test.mjs`, RGB probe에서 5개 프레임의 실제 검출·매칭·복원 | 원격은 SIFT/RootSIFT·ORB·guided matching도 구현. 로컬 FAST/BRIEF를 최종 frontend로 새로 표준화하지 않음 |
| 장기 2D 추적 | **부분 지원**: `src/reconstruction/tracks.ts:buildTracks`는 입력 pair의 conflict-safe 다중 관측 결합. descriptor 재검출을 잇는 graph이며 optical flow가 아님 | `tests/geometry.test.ts`의 track ownership; 이번 기하 fixture의 대응 ID는 합성으로 제공됨 | `src/gpu/klt-tracker.ts:createKltTracker`, `trackKltGpu`와 `tests/typeshade/klt-webgpu.test.mjs` 존재. GPU resident pyramid/track 상태 재사용 대상; 로컬 실행 근거는 없음 |
| E/H 기하 검증 | **지원**(보정·수치 범위 내): `src/geometry/image-pair.ts:verifyImagePair`, `src/gpu/image-pair.ts:verifyImagePairGpu` | `tests/image-pair.test.ts`; CPU fixture와 RGB probe. eight-point와 H 검사이며 강한 회전/이동 분류·five-point solver는 아님 | 원격 `verifyImagePairWithHomography`와 cheirality retry가 있음. H 우세를 순수 회전이라고 확정하는 API는 조사한 계약에 없음 |
| 초기화·PnP 등록 | **지원**(정적 기하·충분한 대응): `src/reconstruction/mapper.ts:incrementalSteps`, `src/geometry/pnp.ts:estimatePosePnP` | `tests/mapper.test.ts`, `tests/pnp.test.ts`, S01·S03·S04; 지도 점이 있으면 중심 고정 회전 view도 등록됨 | 원격은 `scene-mapper.ts`, `component-mapper.ts` 경로와 stalled registration 개선 #145가 있음 |
| 삼각측량·재평가 | **부분 지원**: `multiview.ts:triangulateTrack`, mapper `updateLandmarks`는 등록 뒤 기존 graph를 재삼각측량 | S03-unseen 40개, S03-reseen 60개; 양의 깊이·각도·잔차 검사 | 로컬은 매번 재구축하는 경로. 원격 #24의 persistent landmarks·`MapHistory`를 재구현하지 않음 |
| 저시차·평면·회전의 가짜 구조 방지 | **부분 지원**: `ReconstructionInitializationError`, `minInitialAngleDeg`, `minTriangulationAngleDeg`, H dominance | S02/S05/S06의 정확한 대응 fixture 5개와 RGB 회전/회전+줌 2개에서 구조 생성 0 | 보류 이유가 수치 오류 문자열로 섞임. 반복성·잔차만으로 uncertainty calibration을 충족한 것은 아님 |
| 지도 없는 방향 추적·초기화 지연 상태 | **미지원**: `ImagePairVerification`은 실패 시 빈 inlierRows/reason; `IncrementalResult`는 `RegisteredView`의 full `Pose`만 표현 | S02는 전체 초기화 거부이며 orientation-only R와 resumable deferred tracks를 반환하지 않음 | 원격 `types.ts`, `image-pair.ts`를 읽은 범위에서도 별도 상태를 확인하지 못함. KLT가 있다고 이 기능이 자동 구현되는 것은 아님 |
| BA·공유 초점거리 | **지원**(작은 정적 지도/공유 scale): `bundle-adjustment.ts:bundleAdjust`, `joint-focal.ts:bundleAdjustJointFocal`, GPU Schur/PCG | `tests/bundle-adjustment.test.ts`, `tests/typeshade/focal.test.mjs`의 known calibration·rollback 검사; 기존 전체 테스트 재실행 | 원격은 성장 기반 BA·streaming·cooperative products까지 구현. 최신 변경을 로컬에 새로 복제하지 않음 |
| 내부 파라미터 그룹 | **부분 지원**: view별 알려진 `Camera` 입력 가능, 자동 보정은 하나의 공유 focal scale | S06은 프레임별 **정답 K 제공** 조건; 추정 성능이 아님 | 원격 `component-mapper.ts:focalGroupsOf`, `joint-focal.ts:focalGroupIndex` 및 `tests/multi-video-mapper.test.ts`에 video별 CPU 그룹 최적화가 있음; GPU multi-group은 미확인/로드맵 잔여 |
| 연속 줌·조준·K change point | **미지원**: 로컬 solver가 구간별 연속/불연속 K를 추정하는 API 없음 | S06의 known K에서 새 깊이를 만들지 않는 검사만 수행; S07/S08 미실행 | 원격 자동 focal search #162는 정적 공유 f 검색. 연속 줌 추정의 대체 증거가 아님 |
| resize·crop·유효 영역 | **부분 지원**: `sfm-input.ts:cameraInSourcePixels`, worker의 fx/fy 비등방 rounding 처리, COLMAP half-pixel export | `tests/sfm-video.test.ts`, `tests/sfm-export.test.ts`의 rays·좌표·track 왕복 | resize는 있으나 로컬 crop offset·letterbox·projection region의 일반 계약은 없음. 소스/분석 좌표 변환 메타데이터 확장 제안 |
| 이중 투영 스코프 | **미지원**: 단일 view당 한 Camera, 공통 사각형 제외만 가능 | S09 계획만 있으며 현재 자동 분리·검증 테스트 없음 | 다른 투영 영역의 관측을 한 Camera에 넣지 않는 계약을 먼저 추가해야 함 |
| HUD·마스크 | **부분 지원**: 앞 세션 변경 `features.ts:excludedRegions`, `features.shade.ts:excludeFeatureScores`, `SfmMaskEditor` | GPU/CPU mask 테스트 존재; 이번 RGB probe는 **마스크 없음** | 공통 수동 사각형만 있음. 프레임별 instance mask·association·운동 모델은 별도 단계 |
| 정적/동적 모델·큰 전경 | **미지원**: `FeatureObservation`과 `Landmark`에 motion/object 소속 없음. E RANSAC 지지 집합의 static-world 보증 없음 | S10–14 미실행, 최대 지지 집합이 배경이라는 검증 없음 | 의미 분류·실제 motion·관측 보존을 분리하는 후속 설계 필요 |
| 강체 객체 로컬 형상·상대 운동 | **미지원**: local shape / object-to-world / camera-relative trajectory 계약 없음 | S11–13 계획. 추적 대상만으로 월드 궤적을 나누는 실험은 미실행 | 기존 기하를 객체 좌표에 재사용하되 독립 스케일·gauge 가정을 명시할 것 |
| 사람·관절·변형 | **미지원**: skeleton·deformation/association 모델 없음 | S14 계획. 사람을 전체 강체로 승인하는 테스트나 사전 모델 도입 없음 | TypeNN 모델·사전 도입은 목적·라이선스·입력·비용을 먼저 별도 평가 |
| 컷·디졸브·편집 시간 | **미지원**(로컬): shot ID·edit/source time·transition 상태·cut detector API 없음 | S15는 **대응점을 제공한** cross-shot 초기화 검사, 컷 검출 성능이 아님; S18/S19 미실행 | 원격 multi-video는 시간적으로 떨어진 입력의 연결 재사용 대상. 샷 경계·사건 시간 처리까지 지원한다고 확인하지 못함 |
| 다른 샷의 공동 초기화 | **부분 지원**: 로컬 mapper는 입력 순서가 아닌 모든 supplied pair에서 seed 선택 | S15: 정지에 가까운 두 샷을 가로지르는 seed `[0,2]`, 4/4 views·60 points | 원격 view graph/component mapper가 cross-video edges·공동 seed 선택 지원. shot별 SfM 선행조건을 새로 추가하면 퇴보 |
| 부분 지도·재지역화·병합 | **부분 지원**: 로컬은 단일 연결 map과 `unregistered`만 반환 | S16: 3/6 등록, 나머지 3개 unregistered, cross-component landmark 0. 두 번째 지도 생성은 하지 않음 | 원격 `buildViewGraph`·`reconstructComponent`·retrieval 존재, #137–143 merged. stalled-growth Sim(3) 병합은 별도 잔여; 임의 병합 금지 |
| 불확실성·근거 상태 | **부분 지원**: residual·parallax·positive depth·reason·unregistered. 확률적 depth uncertainty와 여러 직교 상태 없음 | 수치 rejection과 없는 깊이를 별도로 보고. confidence calibration·false-merge/association 평가 미실행 | 원격 `posed-evidence.ts`는 재현용 evidence로, 범용 불확실성/동적 상태의 대체 API가 아님 |
| CPU/GPU·readback | **지원**(명시한 backend): `ShadeDevice`, resident features, GPU geometry; graph/control/packing/result indexing은 CPU | 이번 SwiftShader RGB와 기존 프레임 cold/warm 측정. CPU fallback 없음, 모든 resident buffers dispose 뒤 0 | 원격 actual-device info·새 kernel accounting·streaming #167/#198/#208을 우선 재사용. 이 로컬의 오래된 kernel 시간 합을 최신 성능으로 쓰지 않음 |
| native GPU·WebGL2 | **확인 불가**(native 측정 없음) / **미지원**(로컬 공개 WebGL2 경로) | 실제 adapter `google swiftshader`; `ShadeDevice.backend`는 cpu/webgpu만 허용 | `ShadePlanReport.tier` union에 webgl2가 있다는 사실은 TypeCV 실행 지원 증거가 아님 |
| MVS·dense fusion/mesh | **미지원**(이 로컬 체크아웃) | 로컬 API/테스트 없음. 이번엔 dense 알고리즘 미실행 | 원격 `src/gpu/mvs.ts`, photometric/geometric passes와 `tests/typeshade/mvs-webgpu.test.mjs` 코드 존재; TSDF·cleanup·mesh도 있음. 실행·품질은 이번 세션 **확인 불가** |
| TypeShade·TypeNN 연결 | **지원**(TypeShade 컴퓨트/명시적 CPU oracle) / **부분 지원**(외부 float descriptors) | GPU frontend·geometry 실측, `reconstructIncrementalFloatGpu`의 same-runtime resident 계약 | TypeNN tensor zero-copy·실제 segmentation/pose 모델 통합은 조사한 로컬 코드에 없음 |
| Radiance·Gaussian 연결 | **미지원**(현재 direct adapter·GS 학습) / renderer 실행 **확인 불가** | 원격 `packages/radiance/src/index.ts`의 실제 exports, `PerspectiveCamera.ts`, 공개 surface 검토; 로컬 실행 없음 | #67은 **merged된 draft 설계 문서**, Gaussian 구현의 merge가 아님. 0011 `status: draft`; tree/API에 capture/GS 구현 없음 |

### 재현에서 확인한 조건과 원인 가설

정확한 대응점 fixture는 geometry의 입력 계약만 검사한다. 렌더 RGB에서는 특징점·
대응점도 실제로 추정했다. 두 결과의 등록률을 합산하거나 RGB-only 성능으로
바꿔 부르지 않는다.

| 사례 | 이번에 관측한 결과 | 원인/가설과 다음 실험 |
| --- | --- | --- |
| 지도 없는 회전 | exact 대응: 초기화 거부. RGB: 0/5 등록·0 points | depth를 억지 생성하지 않는 동작은 확인. 방향/2D track을 보존하는 상태가 없어 입력을 다시 평가해야 함; P1의 첫 대상 |
| 회전 후 이동·새 점 미재관측 | 5/5 camera 등록, 기존 40 points만 존재; 회전 중에만 본 20 points의 명시적 deferred 기록 0 | 입력 graph와 geometry rejection이 track 보존을 대신하지 못함. 초기화를 거부한 edge의 tracking 역할을 별도로 보존할 필요 |
| 같은 점을 이동 후 재관측 | 6/6 camera, 60 points | 기존 재삼각측량을 이미 재사용할 수 있음. 온라인 추적 연속성까지 검증한 것은 아님 |
| 순수 줌·회전+줌 | 프레임별 known K에서 초기화 거부; RGB 회전+줌도 0 points | 줌만으로 깊이를 만들지 않음. 영상만으로 K 변화를 추정한 결과는 아님 |
| 평면·매우 먼 점 | exact 대응 fixture 모두 초기화 거부 | 현재 보수적 gate. noisy planar / 작은 병진 / 동적 전경 혼합으로 확장해야 함 |
| 같은 장소의 정지 샷 두 개 | cross-shot seed로 4/4 views·60 points | 샷별 단독 SfM을 요구하면 이 사례를 잃음. 원격 view graph의 공동 seed 경로 재사용 |
| 다른 장소 두 개 | 한 map 3/6 views, 나머지는 미등록 | 잘못 이어 붙이지는 않지만 독립 지도 두 개의 완성은 아님. 이 부분은 원격 component mapper와 통합할 것 |
| 작은 RGB 정적 방 | 5/5 views·234 points, 지도 단위 정렬 후 center RMSE 0.00316 fixture unit, 최대 회전차 2.51° | 완벽한 정확도로 부르지 않음. nearly collinear 경로의 gauge 정렬 민감도, 정량화/매칭 오차의 기여를 후속 대조; reprojection만으로 성공 판정 금지 |
| 기존 피라미드 8장 | CPU/GPU 모두 4/8 views·97 points, 뒤 프레임의 2D–3D 대응 4/2/0/0개 | 입력 간격·frontend/track 단절이 원인 후보. 장면 깊이나 실제 카메라 GT가 없어 정확도 결론은 못 냄; KLT/SIFT 원격 경로와 동일 입력 비교 필요 |

기하 fixture의 S01은 한 지도 정렬 이후 center RMSE 약 `1.58e-12`, point RMSE
약 `2.44e-10`, 최대 회전차 약 `1.21e-6°`였다. 이는 noiseless known K와 제공된
대응점의 결과다. RGB 방의 결과와 동일한 난도의 실험이 아니다. S15는 카메라 중심이
두 개뿐이어서 기존 평가기가 Sim(3) 회전 정렬을 `unavailable`로 남긴다. 프레임별
정렬이나 임의 정렬로 숫자를 채우지 않았다.

### 실행 경계·측정과 병목

이번 장치는 **Google SwiftShader 소프트웨어 WebGPU**다. native GPU 속도·메모리나
native fallback의 성능을 측정했다고 주장하지 않는다. CPU는 독립 f64 기하 참조이며,
TypeShade CPU tier의 동일 IR 실행과도 구분한다.

noiseless geometry 회귀의 사전 게이트는 한 지도 gauge에서 회전행렬 Frobenius 오차
`<1e-6`, 중심 오차 `<1e-5`, 점 오차 `<1e-4 fixture unit`이다.
회전/평면/저시차/보정된 줌은 안정 seed가 없어야 하고, S03의 미재관측 점 생성은 0,
S16의 cross-component landmark는 0이어야 한다. RGB quality gate를 측정 뒤에
맞춰 만들지 않았다. 이번 RGB 수치는 **진단 기준선**이며, broad acceptance gate는
노이즈·경로·장면을 늘리고 고정한 뒤 다음 구현에 적용해야 한다.

RGB 정적 방의 첫 실행(5장, 320×240, max 1,000 pyramid FAST/BRIEF, known K):

| 단계/비용 | 관측값 | 범위 |
| --- | ---: | --- |
| PNG decode | 101.1 ms | 실제 영상 codec decode가 아님 |
| 특징점·업로드·keypoint readback | 718.3 ms | descriptor는 resident, compact keypoints/count를 CPU에서 읽음 |
| descriptor matching | 391.4 ms | matched indices/distances readback 포함 |
| mapping | 30,153.1 ms | 첫 shader compile·검증·PnP·삼각측량·BA·동기화·host control 포함 |
| 전체 | 31,364.1 ms | 위 파이프라인과 feature release. fixture 렌더링·video codec·GS 학습은 제외 |
| application upload / readback | 3,289,160 / 144,688 bytes | 실제 계측 counter; bus 전체 트래픽의 측정은 아님 |
| 3개 RGB case 전체의 peak managed live allocations | 3,466,200 bytes | allocator 직후 샘플; pool·브라우저/driver·decoded host pixels는 제외 |
| dispose 후 live buffers | 0 | 소유권 검사 |

기존 피라미드 8장 동일 입력으로 별도 측정한 cold/warm 결과:

| 실행 | features | matching | mapping | 전체 run |
| --- | ---: | ---: | ---: | ---: |
| cold | 2,217.1 ms | 272.2 ms | 32,934.4 ms | 35,427.2 ms |
| warm (1회) | 2,020.7 ms | 211.3 ms | 1,510.6 ms | 3,745.7 ms |

이 경우 첫 geometry shader compile 영향이 크다. warm에서는 특징점 단계가
약 54%다. 1회 warm 표본으로 최적 설정이나 native speedup을 주장하지 않는다.
같은 대응점의 CPU 기하는 556.8 ms였으나, 이는 features·매칭을 제외하므로
위 GPU 전체 시간과 직접적인 가속비를 만들지 않는다. CPU/GPU는 등록 집합·순서·
초기 pair·점/관측 개수가 같았고 평균 재투영 오차 차이는 약 `1.55e-6 px`였다.
이 비교는 CPU/GPU 수치 일관성이지 원본 장면의 정답 정확도는 아니다.

기존 renderer/worker는 descriptor를 매 pair마다 전체 readback하지 않지만,
keypoints·compact matches와 mapper의 graph/control·packing은 host에 있다.
새 RGB probe는 작은 5장 실험이라 모든 descriptor를 match 종료까지 보유한다.
운영 worker는 `videoPairSchedule.lastUse`에 따라 마지막 사용 뒤 해제한다.
probe의 lifetime을 운영용 무제한 영상 설계로 복사하면 안 된다.

원격 #198의 GPU 시간 계측 수정 이전 로컬이므로 kernel 시간 합은 사용하지 않았다.
현재 자료는 stage wall time·application counters다. 다음 최적화 실험은 원격 SIFT/KLT,
실제 장치 정보·streaming을 통합한 동일 입력에서 warm 표본·순서를 바꿔 반복하고,
geometry·등록·보류 상태 불변을 함께 확인해야 한다.

### 중복 작업 방지와 로컬 하위 작업 분해

GitHub 검색 시점의 공개 검색 결과에는 TypeCV 이슈 #160(장치 예산)이 있었고,
동일 목적의 일반 게임플레이 상위 이슈는 검색 결과에서 찾지 못했다. 검색은 접근
가능 범위의 유한 스냅샷이다. 이번에 외부 이슈를 새로 만들거나 수정하지 않았다.

- **G0 (이번 단계)**: 이 문서·fixtures·측정. 기존 PR #200의 현황 정리와 달리
  오래된 로컬 작업 트리 및 S01–S20 요구의 조건을 대조한다.
- **G1**: 원격 기준 통합 범위 정리. #24/#40의 scene/history, #116/#130의 gate/KLT,
  #137–143의 ID/retrieval/component mapper, #162의 focal search를 재사용한다.
  #15는 closed draft이고 **merged 아님**; 구현 완료 근거로 쓰지 않는다.
- **G2**: rotation tracking evidence와 deferred observations의 최소 계약. G1 뒤
  geometry·tracking edge의 역할을 분리하고 S02/S03·no-reobservation 대조를 통과시킨다.
- **G3**: intrinsics group/region·change point. 기존 CPU focal group을 확장하며
  GPU group·zoom smoothness는 지원 조건을 따로 고정한다.
- **G4**: cut 후보·혼합 프레임·edit timeline을 view graph에 연결. 독립 gauge와
  여러 샷의 공동 seed를 유지하고, 검증된 Sim(3) 제약만 병합에 사용한다.
- **G5**: foreground association·rigid local geometry와 object-relative motion.
  대상만 있는 경우 gauge/scale 모호성을 결과 상태로 남긴다.
- **G6**: skeletal/nonrigid priors 및 Radiance 연결. Radiance #67의 merged draft와
  실제 구현을 구분하고 `PerspectiveCamera`의 **수직 FOV**와 TypeCV K를 변환한다.
  현재 API의 중심 주점 pinhole 범위 밖 crop·distortion은 호환성 검사 후 보류한다.
- **G7**: 실제 GPU·실제 RGB 클립 평가. #160/#167/#198/#208/#210의 작업을 복제하지
  않고, 그 코드와 측정 체계를 재사용한다. PR #212는 이번 조사 시점 open이며
  최종 observation filtering/history 정합성 작업이다. 같은 수정 이슈를 중복 생성하지 않는다.

G0는 G1의 자동 merge/rebase를 요구하지 않는다. G2–G7은 설계·게이트이며 이번 구현의
지원 목록이 아니다. Radiance의 합성 렌더러를 새로 만드는 대신, 이번에 추가한
작은 analytic room은 테스트 데이터 생성 전용이다. 장기 RGB fixture 생성에는
Radiance의 기존 renderer를 사용할 수 있는 버전·카메라 계약을 먼저 맞출 것이다.

### 재현 명령과 정답 분리

```sh
npm run build
node --test tests/general-video.test.ts
node scripts/benchmark-general-video.mjs --render-dir=data/general-video-fixtures
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/probe-general-video-rgb.mjs
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-sfm-pyramid-e2e.mjs \
  --warm-runs=1 --timestamps=0 \
  --output=docs/experiments/general-video/pyramid-software.json
```

`data/general-video-fixtures/<case>/input/frames.json`과 PNG만 RGB 입력이다.
`known-calibration.json`은 **known K 실험에서만** 명시적으로 제공한다.
`truth/evaluation-only.json`과 `.f32` depth는 evaluator 전용이며 RGB inference
경계로 보내지 않는다. shape·depth·GT poses·합성 identity matches·샷 label도
RGB probe의 browser 인자로 보내지 않는다. geometry benchmark의 합성 대응점
모드는 따로 표기한다. RGB renderer의 실제 표면과 geometry-only 점군은 서로
다른 fixture이므로 점군 정답을 RGB 표면의 정답처럼 사용하지 않는다.

검증 범위 밖은 S07–S14, S17–S20의 실제 검출/추적/재지역화·동적 재구성,
unknown-intrinsics RGB zoom, 원격 main의 실행, 실제 게임 클립 및 native GPU다.
원본 게임 메시·토폴로지·머티리얼·로직 복원이나 Radiance 학습 완료를 주장하지 않는다.

MVS-Synth의 좌표/무한 깊이 주의사항은 [공식 데이터 형식](https://phuang17.github.io/DeepMVS/mvs-synth.html)을
확인했다. TartanAir의 stereo/depth/pose 제공은 [공식 소개](https://theairlab.org/tartanair-dataset/)로
확인했으며, 여기서 단안 RGB 입력만 사용할 실험은 그 modality들을 inference에서 배제해야 한다.
[OpenCV의 H 설명](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)과
[COLMAP의 보정·등록·병합 설명](https://colmap.github.io/faq.html)은 후속 설계 참고이며,
그 구현이나 논문의 성공을 이번 게임 영상의 성공률로 집계하지 않았다.
