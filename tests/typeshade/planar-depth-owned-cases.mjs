import { planarInput, planarMvsCase } from './planar-depth-cases.mjs';

const check = (condition, message) => {
  if (!condition) throw new Error(message);
};
const bytes = (array) => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
function equalBytes(a, b, message) {
  check(a.byteLength === b.byteLength, message + ': length');
  const left = bytes(a),
    right = bytes(b);
  for (let i = 0; i < left.length; i++)
    if (left[i] !== right[i]) throw new Error(message + ': byte ' + i);
}
function snapshot(input) {
  return input.views.map((view) => ({
    original: view,
    depth: view.depth.slice(),
    normal: view.normal.slice(),
    ...(view.confidence ? { confidence: view.confidence.slice() } : {}),
    rgba: view.rgba.slice(),
  }));
}
function unchanged(records, message) {
  for (const [index, record] of records.entries())
    for (const name of ['depth', 'normal', 'confidence', 'rgba']) {
      if (!record[name]) continue;
      equalBytes(record.original[name], record[name], `${message} view ${index} ${name}`);
    }
}
function sameMaps(actual, expected, label) {
  check(actual.input.views.length === expected.input.views.length, label + ': view count');
  check(actual.report.filled === expected.report.filled, label + ': total fill');
  check(
    JSON.stringify(actual.report.perView) === JSON.stringify(expected.report.perView),
    label + ': per-view fill',
  );
  for (const [index, view] of actual.input.views.entries()) {
    const other = expected.input.views[index];
    for (const name of ['depth', 'normal', 'confidence', 'rgba'])
      equalBytes(view[name], other[name], `${label} view ${index} ${name}`);
    check(JSON.stringify(view.center) === JSON.stringify(other.center), label + ': camera index');
  }
}
function planeView(center, template, hole = false) {
  const depth = Float32Array.from(
    { length: 256 },
    (_, i) =>
      (2 + 0.15 * center) /
      (1 - (0.15 * ((i % 16) - 7.5)) / 32 - (0.08 * (Math.floor(i / 16) - 7.5)) / 32),
  );
  if (hole) for (let y = 5; y < 11; y++) for (let x = 5; x < 11; x++) depth[16 * y + x] = 0;
  return {
    ...template,
    center: [center, 0, 0],
    rotation: [...template.rotation],
    depth,
    normal: template.normal.slice(),
    confidence: Float32Array.from(depth, (d) => (d > 0 ? 1 : 0)),
    rgba: template.rgba.slice(),
  };
}
function graphInput(count, holes = true) {
  const template = planarInput().input.views[1];
  return {
    width: 16,
    height: 16,
    views: Array.from({ length: count }, (_, index) =>
      planeView((index - 2) * 0.1, template, holes && index % 2 === 0),
    ),
  };
}
function frozenCase() {
  const input = graphInput(4, false);
  input.views[0] = planeView(-0.1, input.views[0], true);
  input.views[1] = planeView(0, input.views[1], true);
  input.views[2] = planeView(0.1, input.views[2]);
  input.views[3] = planeView(0.2, input.views[3]);
  return input;
}
function sharedCase() {
  const input = graphInput(5);
  for (const name of ['depth', 'normal', 'confidence', 'rgba']) {
    const ArrayType = input.views[0][name].constructor;
    const backing = new ArrayType((input.views.length + 2) * 256).fill(99);
    for (const [index, view] of input.views.entries()) {
      backing.set(view[name], (index + 1) * 256);
      view[name] = backing.subarray((index + 1) * 256, (index + 2) * 256);
    }
  }
  input.views[1].depth = input.views[0].depth;
  input.views[1].confidence = input.views[0].confidence;
  delete input.views[2].confidence;
  delete input.views[4].confidence;
  return input;
}

async function cancelledCase(api, backend, owned) {
  const input = planarInput().input;
  const before = snapshot(input);
  const slots = input.views.slice();
  const abort = new AbortController();
  const device = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  const originalPlan = device.planSteps;
  let plans = 0,
    aborted = false,
    result;
  device.planSteps = async function (...args) {
    const output = await originalPlan.apply(this, args);
    // Three independent groups: five plans per reference, so abort during a later reference.
    if (++plans === 6) abort.abort();
    return output;
  };
  try {
    try {
      result = await (owned ? api.completeOwnedPlanarDepths : api.completePlanarDepths)(
        device,
        input,
        { signal: abort.signal },
      );
    } catch (error) {
      aborted = error.name === 'AbortError';
    }
    check(
      aborted && result === undefined,
      'cancelled planar completion must reject without a result',
    );
    unchanged(before, owned ? 'cancel owned' : 'cancel immutable');
    if (!owned)
      for (let i = 0; i < slots.length; i++)
        check(input.views[i] === slots[i], 'immutable slots changed on cancellation');
    check(device.statistics.liveBytes === 0, 'cancelled planar completion leaked device arrays');
    return { plans, aborted, liveBytes: device.statistics.liveBytes };
  } finally {
    device.planSteps = originalPlan;
    await device.dispose();
  }
}

/** Exact same-device oracle checks: completed fills must never become new source evidence. */
export async function ownedPlanarCase(api, backend) {
  check(typeof api.completeOwnedPlanarDepths === 'function', 'missing owned planar entry');
  const cases = [
    ...[
      'plane',
      'rotated',
      'weak-support',
      'limited-overlap',
      'occluded',
      'one-source',
      'repeated-source',
      'color-edge',
      'depth-edge',
    ].map((name) => ({ name, make: () => planarInput(name).input })),
    // Later views also read view 1, making the frontier scheduler process 0 before 1.
    // View 1 must still see ORIGINAL view 0, rather than its freshly filled result.
    {
      name: 'frozen-two-holes',
      make: frozenCase,
      options: { sources: [[2, 3], [0, 2], [1], [1]] },
    },
    {
      name: 'cycles',
      make: () => graphInput(6),
      options: {
        sources: [
          [1, 2],
          [2, 3],
          [3, 4],
          [4, 5],
          [5, 0],
          [0, 1],
        ],
      },
    },
    {
      name: 'high-degree',
      make: () => graphInput(9),
      options: {
        sources: [
          [1, 2, 3, 4, 5, 6, 7, 8],
          [0, 2],
          [0, 3],
          [0, 4],
          [0, 5],
          [0, 6],
          [0, 7],
          [0, 8],
          [0, 0],
        ],
      },
    },
    {
      name: 'reordered-graph',
      make: () => graphInput(6),
      options: { sources: [[3, 4], [4, 5], [5, 3], [0], [1], [2]] },
    },
    {
      name: 'zero-sources',
      make: () => graphInput(6),
      options: { sources: [[], [], [], [], [], []] },
    },
    { name: 'shared-subviews-missing-confidence', make: sharedCase },
  ];
  const device = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  const rows = [];
  let bootstrapExtra = 0;
  try {
    for (const spec of cases) {
      const immutable = spec.make(),
        consumed = spec.make();
      const beforePublic = snapshot(immutable),
        beforeOwned = snapshot(consumed);
      const slots = immutable.views.slice();
      const oracle = await api.completePlanarDepths(device, immutable, spec.options);
      unchanged(beforePublic, spec.name + ' public');
      for (let i = 0; i < slots.length; i++)
        check(immutable.views[i] === slots[i], spec.name + ': public slot replacement');
      const owned = await api.completeOwnedPlanarDepths(device, consumed, spec.options);
      check(owned.input.views === consumed.views, spec.name + ': owned slot container changed');
      unchanged(beforeOwned, spec.name + ' original bytes');
      sameMaps(owned, oracle, spec.name);
      check(device.statistics.liveBytes === 0, spec.name + ': device leak');
      const memory = owned.report.hostMaps;
      check(
        memory && memory.inputBytes > 0 && memory.peakRetainedBytes >= memory.inputBytes,
        spec.name + ': missing host map accounting',
      );
      check(
        memory.retiredOriginalViews === consumed.views.length,
        spec.name + ': originals not fully retired',
      );
      check(
        memory.peakExtraOriginalViews >= 0 &&
          memory.peakExtraOriginalViews <= consumed.views.length,
        spec.name + ': invalid frontier',
      );
      check(
        memory.reorderedViews >= 0 && memory.reorderedViews <= consumed.views.length,
        spec.name + ': invalid reordered count',
      );
      if (spec.name === 'frozen-two-holes') {
        check(
          oracle.report.perView[0] === 36 && oracle.report.perView[1] < 36,
          'frozen two-hole oracle lacks unsupported region',
        );
        const tainted = spec.make();
        tainted.views[0] = oracle.input.views[0];
        const chained = await api.completePlanarDepths(device, tainted, spec.options);
        bootstrapExtra = chained.report.perView[1] - oracle.report.perView[1];
        check(bootstrapExtra > 0, 'fixture must detect using completed fills as evidence');
      }
      if (spec.name === 'zero-sources')
        check(owned.report.filled === 0, 'empty evidence filled a hole');
      if (spec.name === 'reordered-graph')
        check(memory.reorderedViews > 0, 'dependency fixture did not reorder references');
      rows.push({
        name: spec.name,
        filled: owned.report.filled,
        perView: [...owned.report.perView],
        hostMaps: memory,
      });
    }
  } finally {
    await device.dispose();
  }
  let callerMapsPreserved = false;
  const mvs = await planarMvsCase(
    {
      ...api,
      async reconstructDense(device, scene, images, options) {
        const given = options.depthMaps;
        const slots = Object.fromEntries(
          ['depth', 'normal', 'confidence'].map((name) => [name, given[name].slice()]),
        );
        const content = Object.fromEntries(
          ['depth', 'normal', 'confidence'].map((name) => [
            name,
            given[name].map((map) => map.slice()),
          ]),
        );
        const imageBytes = [...images].map(([id, image]) => [id, image.rgba.slice()]);
        const result = await api.reconstructDense(device, scene, images, options);
        for (const name of ['depth', 'normal', 'confidence'])
          for (let i = 0; i < slots[name].length; i++) {
            check(given[name][i] === slots[name][i], 'caller depth-map slot consumed');
            equalBytes(given[name][i], content[name][i], 'caller depth-map bytes changed');
          }
        for (const [id, bytes] of imageBytes)
          equalBytes(images.get(id).rgba, bytes, 'caller RGBA changed');
        callerMapsPreserved = true;
        return result;
      },
    },
    backend,
  );
  return {
    rows,
    bootstrapExtra,
    mvs,
    callerMapsPreserved,
    publicCancellation: await cancelledCase(api, backend, false),
    ownedCancellation: await cancelledCase(api, backend, true),
  };
}
