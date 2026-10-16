export const origin = 'https://cv.typeshade.dev';
export const repository = 'https://github.com/typeshade/typecv';

export const copy = {
  en: {
    overview: 'Overview',
    docs: 'Documentation',
    playground: 'Playground',
    tagline: 'Computer vision, powered by TypeShade.',
    intro:
      'Reusable computer vision in TypeScript, with computation on the GPU. Start with geometric vision and sparse reconstruction, within a broader foundation for images, correspondence, motion and 3D.',
    start: 'Read the guide',
    try: 'Run in your browser',
    eyebrow: 'TypeScript → TypeShade → WebGPU',
    hero: 'A clearer view.\nFrom code to geometry.',
    caption:
      'One resident pair verification plan. Original correspondence indices and a relative-pose summary return to the host.',
    current: 'What runs today',
    currentText:
      'The current implementation concentrates on geometry and sparse 3D. Operators are reusable independently of an SfM graph.',
    ops: [
      [
        '01',
        'Verify correspondence',
        'Radial inversion, stable compaction, H/E consensus and relative-pose checks.',
        '/docs/gpu-initialization/',
        'verifyImagePairGpu',
      ],
      [
        '02',
        'Recover geometry',
        'P3P/PnP RANSAC, robust multi-view triangulation and observation filtering.',
        '/docs/gpu-geometry/',
        'estimatePosePnPGpu',
      ],
      [
        '03',
        'Refine a scene',
        'Fixed-intrinsics joint bundle adjustment with matrix-free Schur/PCG and explicit gauge control.',
        '/docs/incremental-sfm/',
        'bundleAdjustGpu',
      ],
    ],
    scope: 'CV is the scope. SfM is a subset.',
    scopeText:
      'Image processing, native feature extraction and matching, optical flow, dense depth and learned-vision integration are planned. The roadmap distinguishes available operators from future work.',
    roadmap: 'Explore the roadmap',
    evidence: 'Inspect the evidence',
    evidenceText:
      'The recorded pyramid experiment registers 8 of 8 views with 1,723 sparse points. Its calibration is assumed and it has no ground truth. Software WebGPU checks establish execution and numerical behavior; physical-GPU performance remains to be measured.',
    evidenceLink: 'Open the full experiment record',
    source: 'Start from source',
    sourceText:
      'TypeCV is early and has no published npm package. Build the library locally, then import the compiled entry points. The root entry is the independent CPU reference; the GPU entry uses TypeShade.',
    footer: 'Computer vision in TypeScript. Built on TypeShade.',
    englishDocs: 'Project documentation is currently written in English.',
    demoTitle: 'Test a pair. Inspect the result.',
    demoIntro:
      'Generate a calibrated synthetic scene, introduce pixel noise and mismatches, then run the actual compiled TypeCV operator. All input and computation stay in your browser.',
    backend: 'Execution',
    count: 'Correspondences',
    noise: 'Pixel noise',
    outliers: 'Mismatches',
    gpu: 'WebGPU',
    cpu: 'CPU reference (explicit)',
    run: 'Verify pair',
    running: 'Verifying…',
    ready: 'Choose an execution tier, then verify a pair.',
    noGpu:
      'WebGPU is unavailable in this browser. Choose the CPU reference explicitly, or use a browser with WebGPU enabled.',
    input: 'First image',
    output: 'Second image',
    legend: 'Blue: accepted by E · Red: rejected · Gray: not yet verified',
    synthetic: 'Synthetic fixture · seed 71 · pinhole cameras with radial distortion',
    status: 'Result',
    pose: 'Relative pose',
    metrics: 'Execution details',
    note: 'The point-quality count also applies positive-depth, parallax and reprojection checks. H support alone does not determine whether a pair is a suitable SfM seed. Timing includes the call and final reads; this is a correctness demo, not a hardware benchmark.',
    privacy: 'No uploads, account or API key required.',
  },
  ko: {
    overview: '소개',
    docs: '문서',
    playground: '플레이그라운드',
    tagline: 'TypeShade로 실행하는 컴퓨터 비전.',
    intro:
      'TypeScript로 작성하고 GPU에서 계산하는 컴퓨터 비전 라이브러리입니다. 이미지, 대응점, 모션, 3D를 위한 공통 기반 위에서 기하 비전과 희소 복원부터 구현하고 있습니다.',
    start: '가이드 읽기',
    try: '브라우저에서 실행',
    eyebrow: 'TypeScript → TypeShade → WebGPU',
    hero: '코드에서 기하로.\n더 선명한 시야.',
    caption:
      '이미지 쌍 검증을 하나의 GPU 실행 계획으로 연결합니다. 원본 대응점 인덱스와 상대 자세 요약을 반환합니다.',
    current: '지금 실행할 수 있는 연산',
    currentText:
      '현재는 기하 비전과 희소 3D에 집중합니다. 각 연산은 SfM 그래프와 독립적으로 사용할 수 있습니다.',
    ops: [
      [
        '01',
        '대응점 검증',
        '렌즈 역변환, 유효 행 압축, H/E 강건 추정과 상대 자세 검증을 실행합니다.',
        '/docs/gpu-initialization/',
        'verifyImagePairGpu',
      ],
      [
        '02',
        '기하 복원',
        'P3P/PnP RANSAC, 강건한 다중 시점 삼각측량과 관측 필터링을 제공합니다.',
        '/docs/gpu-geometry/',
        'estimatePosePnPGpu',
      ],
      [
        '03',
        '장면 최적화',
        '고정 내부 파라미터와 명시적인 게이지 제약 아래 Schur/PCG 기반 공동 번들 조정을 실행합니다.',
        '/docs/incremental-sfm/',
        'bundleAdjustGpu',
      ],
    ],
    scope: '컴퓨터 비전 전체를 향해. SfM은 그 일부입니다.',
    scopeText:
      '이미지 처리, 자체 특징 추출·매칭, 광류, 밀집 깊이와 학습 기반 비전 연동은 계획 단계입니다. 로드맵에서 구현된 연산과 앞으로의 작업을 구분합니다.',
    roadmap: '로드맵 보기',
    evidence: '측정 결과를 직접 확인하세요',
    evidenceText:
      '기록된 피라미드 영상 실험에서 8개 시점과 1,723개의 희소 3D 점을 복원했습니다. 가정한 카메라 보정을 사용하며 정답 데이터는 없습니다. 소프트웨어 WebGPU로 실행과 수치를 검증했으며, 물리 GPU 성능 측정은 남아 있습니다.',
    evidenceLink: '전체 실험 기록 보기',
    source: '소스에서 시작하기',
    sourceText:
      'TypeCV는 초기 개발 단계이며 npm에 배포되지 않았습니다. 로컬에서 라이브러리를 빌드한 뒤 컴파일된 진입점을 가져오세요. 기본 진입점은 독립적인 CPU 참조 구현이며 GPU 진입점은 TypeShade를 사용합니다.',
    footer: 'TypeScript로 구현하는 컴퓨터 비전. TypeShade 기반.',
    englishDocs: '프로젝트 상세 문서는 현재 영문으로 제공됩니다.',
    demoTitle: '대응점을 검증하고 결과를 확인하세요.',
    demoIntro:
      '보정된 합성 장면에 픽셀 노이즈와 잘못된 대응점을 추가한 뒤, 실제 컴파일된 TypeCV 연산을 실행합니다. 입력과 계산은 모두 브라우저에 머뭅니다.',
    backend: '실행 환경',
    count: '대응점 수',
    noise: '픽셀 노이즈',
    outliers: '잘못된 대응점',
    gpu: 'WebGPU',
    cpu: 'CPU 참조 구현 (명시적 선택)',
    run: '대응점 검증',
    running: '검증 중…',
    ready: '실행 환경을 선택하고 대응점을 검증하세요.',
    noGpu:
      '이 브라우저에서 WebGPU를 사용할 수 없습니다. CPU 참조 구현을 직접 선택하거나 WebGPU가 활성화된 브라우저를 사용하세요.',
    input: '첫 번째 이미지',
    output: '두 번째 이미지',
    legend: '파랑: E 추정에서 채택 · 빨강: 제외 · 회색: 검증 전',
    synthetic: '합성 입력 · 시드 71 · 방사 왜곡을 포함한 핀홀 카메라',
    status: '결과',
    pose: '상대 자세',
    metrics: '실행 정보',
    note: '유효 점 개수에는 양의 깊이, 시차와 재투영 오차 검사가 추가로 적용됩니다. H 지지만으로 SfM 초기 쌍의 적합성을 결정하지 않습니다. 시간은 연산과 최종 읽기를 포함하며, 이 데모는 하드웨어 성능 벤치마크가 아닙니다.',
    privacy: '업로드, 계정과 API 키가 필요하지 않습니다.',
  },
};

export const pairCode = `import { ShadeDevice, verifyImagePairGpu } from './dist/gpu.js';

const device = await ShadeDevice.open();
try {
  const pair = await verifyImagePairGpu(
    device, matches, firstCamera, secondCamera,
    { iterations: 256, threshold: 0.001, seed: 71 },
  );
  console.log(pair.inlierRows, pair.geometry?.R, pair.reason);
} finally {
  await device.dispose();
}`;

export const setupCode = `git clone https://github.com/typeshade/typecv.git
cd typecv
npm ci
npm run build
node examples/two-view.mjs`;
