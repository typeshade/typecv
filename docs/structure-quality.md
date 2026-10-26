# 초기 자세·깊이 안정성과 배경 관측 선택

실제 쇼츠의 관측 그래프 단절 수정과 방향 전용 관측의 남은 조건은
[c3086822 조사](shorts-reconstruction.md)에 별도로 기록했다.

`4264e97` 이후의 첫 품질 개선 단계다. 회전 관측 보존은 이미 기본 활성화되어 있고,
이번 변경은 불안정한 구조와 큰 전경이 만드는 오류를 다룬다. 실제 게임 전체 지원,
자동 동적 객체 검출이나 전체 SfM 불확실성 추정을 완료한 것은 아니다.

## 구현 확인과 변경 범위

| 기존 구현·근거 | 확인한 경계 | 이번 변경 |
| --- | --- | --- |
| `src/geometry/refinement.ts:refinePose`, `src/geometry/gpu/image-pair.ts` | CPU와 TypeShade에 상대 자세 정련이 이미 있으나 mapper는 0회로 고정 | `pairRefinementIterations`로 기존 연산을 선택해 재사용 |
| `src/geometry/multiview.ts:triangulateTrackDetailed`, [PR #18](https://github.com/typeshade/typecv/pull/18) | 강건한 합의·양의 깊이·재투영·시차 검사와 상태 0/1/3/4가 이미 있음 | 고정 자세·K에 조건부인 깊이 민감도와 보류 상태 5 추가 |
| `mapper.ts`, `scene-mapper.ts`, `src/geometry/gpu/observations.ts` | BA 후 필터는 재투영 오차를 검사 | 선택한 깊이 기준도 재검사하고 보류된 트랙을 보존 |
| 기존 작업 트리의 게임 HUD 제외 영역·픽셀 중심 규약 | 해당 미커밋 구현은 보존; 정적·동적 운동 분류기가 아님 | 같은 정규화 영역 규약으로 특징점 마스크를 만들고 합의 모델 추정 전에 적용 |
| Lab SIFT, KLT tracker/join, `website/src/lib/klt-join.ts` | 추적·매칭은 이미 존재 | 기존 경로를 RGB 검증에 그대로 사용; 특징점 ID와 원래 대응점 유지 |

기존 이슈·PR을 읽어 중복 구현을 확인했다. GPU 예산·청크 작업인
[#160](https://github.com/typeshade/typecv/issues/160), #60, #167은 이 변경에서 다시 구현하지 않는다.
새 라이브러리, 별도 저장소, 외부 이슈 등록·수정, 배포는 포함하지 않는다.

## API와 가정

```ts
const result = reconstructIncremental(images, pairs, {
  pairRefinementIterations: 10,
  maxRelativeDepthStdDev: 0.1,
  pixelNoiseStdDevPx: 1,
  cameraObservationMasks,
});
```

`pairRefinementIterations`는 0–100이며 기본값은 기존과 같은 0이다. 별도 기하 엔진을
추가하지 않고 기존 상대 자세 정련을 사용한다. 정련을 요청한 경우 이전 `preVerified`
판정을 그대로 사용하지 않고 재검증한다. 정련 횟수가 많으면 항상 정답에 가까워지는
것은 아니다. 이번 비교는 공급 K 모드이며 새 옵션과 자동 초점거리 탐색의 조합은 미검증이다.

깊이 기준은 기본 비활성화다. 옵션을 제공하면 합의에 포함된 관측의 기존 투영 Jacobian으로
`H = Σ JᵀJ`, `ΣX ≈ σpixel² H⁻¹`를 계산한다. 각 관측 카메라의 +Z 깊이에 대한
`stddev(Z) / Z` 중 최대값을 제한한다. 원근 DLT의 기존 직접 SVD는 바꾸지 않는다.

이 값은 **자세·내부 파라미터·합의 관측이 고정됐다는 조건부 선형 근사**다. 자세·보정값의
오차, 잘못된 대응, 동적 객체와 모델 오차까지 포함하는 전체 불확실성이나 확률이 아니다.
재투영 오차가 0이어도 지정한 픽셀 오차 가정에서는 깊이가 불안정할 수 있다. 1px 기본값은
명시적인 가정이며 잔차에서 잡음을 추정한 값이 아니다. 단안 지도의 스케일을 바꿔도
같은 판정을 유지한다. 특이 정보 행렬과 기준 초과는 상태 5로 보류하며 XYZ·inlier mask를
거부 결과에 노출하지 않는다. 상태 4의 기존 시차 부족과 구분한다.

`triangulateTrackDetailed`와 `assessTrackDepth`도 root API에서 사용할 수 있다.
mapper는 `tracking.depthDeferredTrackIds`로 이 기준 때문에 보류된 트랙을 표시한다.
ID는 활성 그래프의 로컬 번호다. 실제 재관측이 유효 시차를 더하면 기존 삼각측량으로
재평가한다. 새 카메라가 이동했다는 사실만으로 미재관측 점을 확정하지 않는다.

## 전경과 HUD 관측 선택

```ts
const mask = cameraMaskFromRegions(image, width, height, [
  { left: 0, top: 0, right: 1, bottom: 0.12 },
], 'manual', 6);

const result = reconstructIncremental(images, pairs, {
  cameraObservationMasks: [mask],
});

// 스트리밍에서는 특징점을 얻은 뒤 프레임별 마스크를 전달할 수 있다.
await session.add(image, newPairs, mask);
const evidence = session.observationSelection;
```

마스크는 원래 특징점 번호에 대응하는 `Uint8Array`다. 1은 배경 가설에 사용할 후보,
0은 카메라·정적 지도 추정에서 분리할 관측이며 실제 움직임의 판정이 아니다.
`source`는 `manual`, `geometry`, `learned`, `ground-truth` 중 제공한 출처를 기록한다.
이 이름들이 자동 분할기나 학습 모델을 구현했다는 뜻은 아니다.

영역 좌표는 [0,1]의 영상 가장자리 좌표이고, 정수 픽셀 중심은 `(x+0.5, y+0.5)`로
변환해 비교한다. 입력 영상 좌표와 크기가 일치해야 한다. padding은 설명자의 지지 영역을
고려하기 위한 호출자의 가정이며 모든 SIFT scale의 정확한 지지 영역을 자동으로 구하지 않는다.

두 끝점 중 하나가 제외되면 그 대응은 E/H/R 검증과 배경 트랙 생성에서 빠진다.
변경된 쌍의 `preVerified` 판정도 무효화한다. 최대 지지 집합을 자동으로 정적인 배경이라고
인정하는 정책을 추가하지 않는다. 마스크가 없으면 기존 배경 가정과 실패 조건이 남는다.

RGB·특징점·원래 입력 대응점을 수정하거나 번호를 다시 매기지 않는다.
`observationSelection.separatedPairs`에는 분리한 원래 행과 거리가 남는다. 이는 기하 검증된
객체 트랙, 객체 ID, 형상이나 월드 궤적이 아니다. `masks`, 사용 행 수와
`unappliedViewIds`도 기록한다. 미래·누락·퇴거한 뷰의 마스크를 적용했다고 표시하지 않는다.
초기화 실패에도 이 정보가 남고 세션 getter는 독립적인 스냅샷을 반환한다.
세션은 기존 8–64뷰 활성 윈도를 사용하며 분리 관측의 영구 보관소를 추가하지 않는다.

## 비교와 검증

[CPU 비교 기록](experiments/general-video/structure-quality-cpu-comparison.json)은 기존 12개
시나리오와 추가 교란·전경·깊이 보류 조건을 같은 입력에서 번갈아 실행한다.
각 모드 워밍업 1회와 측정 6회이며 다른 검증 작업과 동시에 측정하지 않는다.
합성 K·동일성 대응점을 공급한 기하 실험이다. 정답 포즈·구조는 평가에만 사용하고,
연결된 지도 전체에 초기 프레임의 정답 자세와 baseline 길이로 하나의 gauge를 적용한다.

| 합성 조건 | 기존 → 새 옵션 |
| --- | --- |
| 회전 시작, 좌표 교란 진폭 0.1px | 점 RMSE 0.05094 → 0.00704, 60점 유지 |
| 같은 조건, 0.3px | 점 RMSE 3.96881 → 0.02790, 60점 유지 |
| 같은 조건, 0.7px | 점 RMSE 47.74757 → 0.09845, 36 → 60점 |
| 같은 조건, 1px | 점 RMSE 0.96002 → 0.53067, 60점 유지; 잔여 오차 있음 |
| 배경 60개·움직이는 전경 180개·HUD 60개 특징점 | 정답 마스크로 전경이 섞인 지도 대신 배경 60점 복원; 2400 대응 행 분리 보존 |
| 낮은 baseline의 두 뷰, 조건부 기준 0.03 | 수치상 완벽한 점도 60개 보류; 유효 시점이 추가되면 60점 복원 |

정답 마스크를 사용한 결과를 자동 검출이나 RGB-only 분할 성능으로 해석하지 않는다.
정련을 적용한 교란 사례는 실행 시간이 약 2.7–3.4배로 늘었다. 강한 개선이 나온 입력과
비용을 함께 기록하며, 고비용 정련과 잡음 가정에 의존하는 깊이 기준을 전역 기본값으로
확대하지 않았다. 제공한 관측 마스크는 호출한 경로에서 기본 적용한다.

작은 렌더 장면의 [RGB 비교 기록](experiments/general-video/structure-quality-rgb-comparison.json)은
실제 Lab SIFT·KLT·join 병합을 사용한다.
입력은 RGB와 공급 K이며 합성 전경 조건은 정답 영역 마스크를 추가로 제공한다.
자세·깊이 정답은 브라우저 추론에 전달하지 않는다. GPU의 실제 어댑터, 처리 시간,
전송·read 호출·활성 할당·폐기를 따로 기록한다. 실게임·영상 코덱·원본 메시 복구·MVS·
Radiance 동적 복원이나 하드웨어 가속의 검증을 뜻하지 않는다.

| 렌더 RGB 조건 | 기존 → 새 옵션 | 정확도와 보류 |
| --- | --- | --- |
| 일반 이동, 정련·깊이 기준 적용 | 등록 5/5 유지, 111 → 99점, 깊이 기준으로 12개 보류 | 최대 회전 오차 0.210° → 0.160°; 카메라 중심 RMSE는 0.00613 → 0.00755로 악화 |
| 순수 회전 | 두 모드 모두 지도·점 0, 2D 트랙 163개 유지 | 잘못된 깊이 생성 없음 |
| 합성 전경·HUD, 정답 영역 마스크만 적용 | 등록 5/5 → 3/5, 63 → 38점, 원래 대응 153개 분리 | 공통 프레임의 최대 회전 오차 1.050° → 0.571°, 중심 RMSE 0.07210 → 0.03925 |
| 같은 입력, 정답 마스크와 정련·깊이 기준 함께 적용 | 등록 3/5, 38점 | 공통 프레임 최대 회전 오차 0.693°, 중심 RMSE 0.04700; 마스크 단독보다 나쁨 |

가려진 RGB 입력의 미등록 두 프레임을 성공으로 계산하지 않는다. 공통 프레임 비교도
각 지도 전체의 고정 gauge를 그대로 사용하며 프레임·부분 구간에 새로 정렬하지 않는다.
RGB 특징점의 정답 3D 동일성이 없으므로 이 기록은 점의 기하 정확도를 주장하지 않는다.
추가 정련이 모든 RGB 지표를 개선하지 않아 전역 기본 활성화의 근거로 삼지 않았다.

이 측정은 Google SwiftShader 소프트웨어 WebGPU, Chromium 154, Node 24.19.0이다.
일반 이동의 mapper 중앙값은 823 → 1049ms, 공통 프런트엔드를 더한 값은
6922 → 7148ms다. 전경 입력에서 마스크만 적용한 mapper는 857 → 634ms였지만
등록된 프레임 수도 줄었으므로 동일 작업의 가속률로 해석하지 않는다. 전체 프런트엔드는
입력별 한 번 측정했으며 매 반복마다 디코딩부터 다시 실행한 분포가 아니다.
같은 입력·순서를 번갈아 측정한 원시 기록에 단계별 시간·전송·read 호출 수가 있다.
활성 GPU 바이트는 각 mapper 실행 후 0, device 폐기 후 버퍼 수도 0이다.

GPU 경로의 새 깊이 계산과 BA 후 검사는 TypeShade에서 실행한다. 삼각측량의 기존 최종
readback은 유지하고, BA 후 깊이 검사는 point offsets와 점별 거부 상태를 추가한다.
이를 전송 비용이 없는 기능이라고 표현하지 않는다. 기존 CPU/scene 경로와 기하 판정을 비교한다.
마스크 행 선택은 기존에 호스트에 있는 특징점·대응점을 사용하는 CPU 제어 작업이다.
로컬 BA에서는 안정적인 점의 관측 소유권을 보존하고 불안정한 점 전체를 보류한다.
직렬화한 결과를 해석할 때 사용한 임계값과 픽셀 잡음 가정도 실행 옵션과 함께 보관해야 한다.

[검증 기록](experiments/general-video/structure-quality-validation.json)에 실제 커밋 기준,
소스 해시와 실행 명령을 남겼다. 핵심 테스트 283개(신규 10개), 영향받는 TypeShade CPU
11개, 소프트웨어 WebGPU 2개가 통과했다. 기존 전체 TypeShade/WebGPU/UI 묶음을
모두 재실행한 것은 아니다. 원래 작업 트리의 변경 파일 25개는 해시로 보존을 확인했다.
타입·포맷·core/GPU 빌드, Astro 검사(오류·경고 0, 기존 hint 4), 83페이지 생성과 정적 링크도 통과했다.

```sh
npm run build
node --test tests/structure-quality.test.ts
node --test tests/typeshade/structure-quality.test.mjs
TYPESHADE_CHROMIUM=/usr/bin/chromium node --test tests/typeshade/structure-quality-webgpu.test.mjs
node scripts/benchmark-structure-quality.mjs --repeats=6
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-structure-quality-rgb.mjs --repeats=6
```

후속 작업은 독립 입력의 잡음·오차 보정, 초기 자세의 안정성·다중 시점 가설 검증,
자동 운동 분리 후보와 관측 분포 검사다. 모르는 줌·복수 투영·컷 연결·강체 객체의 형상과
사람의 변형은 이번 단계에서 해결하지 않는다.
