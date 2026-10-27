import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import {
  previousFp64Flavor,
  type FlavorAdapterSignal,
  type PreviousFlavor,
  type PreviousFlavorSignals,
} from '../website/src/lib/hardware-check/flavor.ts';

// The rule that chose the flavor of the emulated double before TypeShade change 0085, as the
// hardware check computes it (docs/lab-run-records.md, "Hardware check"). The table holds the
// copy to the answers of `recommendedFlavor` at 87724f9 (src/device/gpu/device.ts, lines 151 to
// 175) with TypeShade's selector at the pin 725b2969. The second test holds the pin's selector to
// the same table: a pin whose selector gives another answer fails it, and the inference of
// docs/library-structure.md, "0085's product work" (the same flavor as before), must then be read
// again. The third test runs the code of 87724f9 itself where the git history holds it.

const ROOT = join(import.meta.dirname, '..');
const UA = {
  iphone:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0.1 Mobile/15E148 Safari/604.1',
  ipad: 'Mozilla/5.0 (iPad; CPU OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0 Mobile/15E148 Safari/604.1',
  macSafari:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0 Safari/605.1.15',
  macChrome:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36',
  windowsChrome:
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36',
  linuxHeadless:
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.8010.12 Safari/537.36',
  android:
    'Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Mobile Safari/537.36',
  firefox: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0',
};
const info = (vendor: string, architecture: string): FlavorAdapterSignal => ({
  vendor,
  architecture,
});
/** Each row: the signals and the flavor that 87724f9 gives on them. */
const TABLE: readonly [string, PreviousFlavorSignals, PreviousFlavor][] = [
  [
    'the owner iPhone on the lab path',
    { handedDevice: true, deviceAdapterInfo: info('apple', 'common-3'), userAgent: UA.iphone },
    'integer',
  ],
  [
    'an iPhone whose device reports no adapter info',
    { handedDevice: true, deviceAdapterInfo: null, userAgent: UA.iphone },
    'integer',
  ],
  ['an iPad, no adapter', { handedDevice: false, userAgent: UA.ipad }, 'integer'],
  [
    'Safari on a Mac',
    {
      handedDevice: false,
      requestedAdapterInfo: info('apple', 'metal-3'),
      userAgent: UA.macSafari,
    },
    'integer',
  ],
  [
    'Chrome on a Mac with an Intel GPU: the user agent decides',
    { handedDevice: true, deviceAdapterInfo: info('intel', 'gen-12lp'), userAgent: UA.macChrome },
    'integer',
  ],
  [
    'the A1 device: an NVIDIA Blackwell on Windows',
    {
      handedDevice: true,
      deviceAdapterInfo: info('nvidia', 'blackwell'),
      userAgent: UA.windowsChrome,
    },
    'float',
  ],
  [
    'an NVIDIA Turing on Windows, default path',
    {
      handedDevice: false,
      requestedAdapterInfo: info('nvidia', 'turing'),
      userAgent: UA.windowsChrome,
    },
    'float',
  ],
  [
    'SwiftShader in headless Chromium',
    {
      handedDevice: true,
      deviceAdapterInfo: info('google', 'swiftshader'),
      userAgent: UA.linuxHeadless,
    },
    'float',
  ],
  [
    'Chrome on Android',
    { handedDevice: false, requestedAdapterInfo: info('qualcomm', ''), userAgent: UA.android },
    'float',
  ],
  [
    'Firefox on Windows',
    { handedDevice: false, requestedAdapterInfo: info('amd', 'rdna-3'), userAgent: UA.firefox },
    'float',
  ],
  [
    'an Apple vendor string with a capital',
    { handedDevice: false, requestedAdapterInfo: info('Apple Inc.', ''), userAgent: '' },
    'integer',
  ],
  [
    'a Metal architecture without a vendor',
    { handedDevice: true, deviceAdapterInfo: info('', 'metal-3'), userAgent: null },
    'integer',
  ],
  [
    'the lab path reads the device, not the requested adapter',
    {
      handedDevice: true,
      deviceAdapterInfo: info('nvidia', 'blackwell'),
      requestedAdapterInfo: info('apple', 'common-3'),
      userAgent: UA.windowsChrome,
    },
    'float',
  ],
  [
    'the default path reads the requested adapter, not a device',
    {
      handedDevice: false,
      deviceAdapterInfo: info('nvidia', 'blackwell'),
      requestedAdapterInfo: info('apple', 'common-3'),
      userAgent: UA.windowsChrome,
    },
    'integer',
  ],
  [
    'the CPU tier',
    { backend: 'cpu', handedDevice: false, requestedAdapterInfo: info('apple', 'common-3') },
    'float',
  ],
  ['no signal', { handedDevice: false }, 'float'],
];

test('the rule before change 0085 gives the flavor of 87724f9 on every row of the table', () => {
  for (const [name, signals, expected] of TABLE)
    assert.equal(previousFp64Flavor(signals), expected, name);
});

const SELECTOR = join(ROOT, 'node_modules/typeshade/src/core/fp64/flavor-select.ts');
const javascript = (source: string) =>
  ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
const load = (source: string) =>
  import('data:text/javascript,' + encodeURIComponent(source)) as Promise<Record<string, any>>;

test(
  "the pin's selector gives the same flavor on every row",
  { skip: !existsSync(SELECTOR) && 'the TypeShade source is not installed' },
  async () => {
    const { recommendFp64Flavor } = await load(javascript(readFileSync(SELECTOR, 'utf8')));
    for (const [name, signals, expected] of TABLE) {
      if ((signals.backend ?? 'webgpu') !== 'webgpu') continue;
      const adapterInfo = signals.handedDevice
        ? (signals.deviceAdapterInfo ?? null)
        : (signals.requestedAdapterInfo ?? null);
      assert.equal(
        recommendFp64Flavor({ adapterInfo, userAgent: signals.userAgent ?? null }),
        expected,
        `${name}: the pin's selector differs from the rule before change 0085`,
      );
    }
  },
);

/** The code of `recommendedFlavor` at 87724f9, or undefined without that history. */
function previousSource(): string | undefined {
  try {
    const text = execFileSync('git', ['show', '87724f9:src/device/gpu/device.ts'], {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: 'pipe',
    });
    const start = text.indexOf('async function recommendedFlavor(');
    return start < 0 ? undefined : text.slice(start, text.indexOf('\n}\n', start) + 3);
  } catch {
    return undefined;
  }
}
const previous = previousSource();

test(
  'the copy equals the code of 87724f9 on 5600 combinations of the signals',
  {
    skip:
      (!previous && 'the git history does not hold 87724f9') ||
      (!existsSync(SELECTOR) && 'the TypeShade source is not installed'),
  },
  async () => {
    const selector = javascript(readFileSync(SELECTOR, 'utf8')).replace(/^import .*$/gm, '');
    const { recommendedFlavor } = await load(
      `${selector}\n${javascript(previous!)}\nexport { recommendedFlavor };`,
    );
    const vendors = [undefined, '', 'apple', 'Apple Inc.', 'nvidia', 'intel', 'amd', 'qualcomm'];
    const architectures = [undefined, '', 'metal-3', 'apple-7', 'common-3', 'blackwell', 'turing'];
    const agents = [null, '', ...Object.values(UA)];
    const navigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
    let rows = 0;
    try {
      for (const backend of ['webgpu', 'cpu'] as const)
        for (const mode of ['handed', 'handed without info', 'requested', 'no adapter', 'no gpu'])
          for (const vendor of vendors)
            for (const architecture of architectures)
              for (const userAgent of agents) {
                const adapter =
                  vendor === undefined && architecture === undefined
                    ? null
                    : { vendor, architecture };
                const gpu =
                  mode === 'no gpu'
                    ? undefined
                    : {
                        requestAdapter: async () =>
                          mode === 'no adapter' ? null : { info: adapter },
                      };
                Object.defineProperty(globalThis, 'navigator', {
                  configurable: true,
                  writable: true,
                  value: { gpu, userAgent: userAgent ?? undefined },
                });
                const gpuDevice =
                  mode === 'handed'
                    ? { adapterInfo: adapter }
                    : mode === 'handed without info'
                      ? {}
                      : undefined;
                const expected = await recommendedFlavor(backend, gpuDevice);
                const copy = previousFp64Flavor({
                  backend,
                  handedDevice: gpuDevice !== undefined,
                  deviceAdapterInfo: gpuDevice?.adapterInfo ?? null,
                  requestedAdapterInfo: mode === 'requested' ? adapter : null,
                  userAgent: userAgent ?? null,
                });
                assert.equal(copy, expected, JSON.stringify({ backend, mode, adapter, userAgent }));
                rows++;
              }
    } finally {
      if (navigator) Object.defineProperty(globalThis, 'navigator', navigator);
    }
    assert.equal(rows, 2 * 5 * 8 * 7 * 10);
  },
);
