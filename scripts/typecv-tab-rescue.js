/*
 * Existing TypeCV Lab tab only: review this file, then run the whole file as a
 * DevTools Sources > Snippets snippet (or paste it into the Console).
 *
 * This does not reload the page, run reconstruction, send network requests, or
 * change application refs. A button asks for an output folder and streams copies
 * of the retained depth artifacts. React internals are version dependent: an
 * unsupported or ambiguous layout stops explicitly.
 */
(() => {
  'use strict';
  const pattern = /^typecv-depths-c[0-9]{3}(?:-[0-9]{4}\.tar\.gz|\.json)$/;
  const panelId = 'typecv-depth-recovery-panel';

  function artifactsFromDocument(doc) {
    const roots = doc.querySelectorAll('.lab-app');
    if (roots.length !== 1) throw new Error('Expected exactly one TypeCV Lab (.lab-app).');
    const root = roots[0];
    const keys = Object.getOwnPropertyNames(root).filter((key) => key.startsWith('__reactFiber$'));
    if (keys.length !== 1) throw new Error('Unsupported React host layout; no data was changed.');
    const candidates = new Set();
    const seenFibers = new Set();
    let fiber = root[keys[0]];
    for (let depth = 0; fiber && depth < 128 && !seenFibers.has(fiber); depth++) {
      seenFibers.add(fiber);
      const seenHooks = new Set();
      let hook = fiber.memoizedState;
      for (let i = 0; hook && typeof hook === 'object' && i < 1024 && !seenHooks.has(hook); i++) {
        seenHooks.add(hook);
        const state = hook.memoizedState;
        const descriptor =
          state && typeof state === 'object'
            ? Object.getOwnPropertyDescriptor(state, 'current')
            : undefined;
        const map = descriptor?.value;
        if (
          map instanceof Map &&
          map.size > 0 &&
          map.size <= 10000 &&
          [...map].every(
            ([name, data]) =>
              typeof name === 'string' &&
              pattern.test(name) &&
              (typeof data === 'string' || data instanceof Blob),
          )
        )
          candidates.add(map);
        hook = hook.next;
      }
      fiber = fiber.return;
    }
    if (candidates.size !== 1)
      throw new Error(
        candidates.size
          ? 'Ambiguous artifact refs; stopped without choosing one.'
          : 'No supported retained depth-artifact ref found. Save the run JSON and each component JSON/PLY using the existing export buttons.',
      );
    // Copies names and references only, never the Blob payloads or the app's Map.
    return new Map(candidates.values().next().value);
  }

  function sourceSummary(files) {
    return [...files]
      .filter(([name, data]) => name.endsWith('.json') && typeof data === 'string')
      .map(([name, data]) => {
        try {
          const value = JSON.parse(data);
          if (value.format !== 'typecv-depth-archive-v1' || !Array.isArray(value.archives))
            throw new Error('Unknown archive manifest');
          const invalid = value.archives.some(
            (part) => typeof part !== 'string' || !pattern.test(part),
          );
          return {
            name,
            captureComplete: value.complete === true && !invalid,
            captureOutcome: value.outcome,
            captureError: typeof value.error === 'string' ? value.error.slice(0, 500) : null,
            missingDuringCapture: Array.isArray(value.missing) ? value.missing.length : null,
            missingRetainedParts: invalid
              ? ['Invalid archive part list']
              : value.archives.filter((part) => !files.has(part)),
          };
        } catch (error) {
          return { name, captureComplete: false, manifestError: String(error).slice(0, 500) };
        }
      });
  }

  function scopeFromRecord(record) {
    const unknown = (reason) => ({ known: false, expectedComponentIds: [], reason });
    if (
      !record ||
      !/^typecv-run-record-v[456]$/.test(record.format ?? '') ||
      record.dense?.status !== 'completed'
    )
      return unknown('No supported completed Dense run record.');
    const videos = record.input?.videos;
    if (Array.isArray(videos) && videos.length) {
      const ids = new Set();
      for (const video of videos) {
        if (!Number.isSafeInteger(video?.registeredViews) || video.registeredViews < 0)
          return unknown('Invalid per-video registration count.');
        if (!video.registeredViews) continue;
        if (!Number.isSafeInteger(video.component) || video.component < 0 || video.component > 999)
          return unknown('Missing registered component identifier.');
        ids.add(video.component);
      }
      if (
        !ids.size ||
        !Array.isArray(record.result?.components) ||
        record.result.components.length !== ids.size
      )
        return unknown('Component scope does not match the run record.');
      return {
        known: true,
        expectedComponentIds: [...ids].sort((a, b) => a - b),
        reason: 'Registered video components in the completed Dense run record.',
      };
    }
    if (
      (videos !== undefined && !Array.isArray(videos)) ||
      !(record.result?.registeredViews >= 2) ||
      !(record.dense.views > 0) ||
      (Array.isArray(record.result.components) && record.result.components.length > 1)
    )
      return unknown('Cannot identify a single-run component scope.');
    return {
      known: true,
      expectedComponentIds: [0],
      reason: 'Completed single-run Dense record.',
    };
  }

  function scopeFromDocument(doc) {
    try {
      const text = doc.querySelector?.('.run-record details pre')?.textContent;
      if (!text || text.length > 256 * 1024)
        return { known: false, expectedComponentIds: [], reason: 'Run record JSON unavailable.' };
      return scopeFromRecord(JSON.parse(text));
    } catch {
      return { known: false, expectedComponentIds: [], reason: 'Run record JSON is unreadable.' };
    }
  }

  function sourceCompleteness(files, scope) {
    const archives = sourceSummary(files);
    const ids = archives.map((archive) =>
      Number(/^typecv-depths-c([0-9]{3})\.json$/.exec(archive.name)[1]),
    );
    const missingComponentManifests = scope.known
      ? scope.expectedComponentIds
          .filter((id) => !ids.includes(id))
          .map((id) => 'typecv-depths-c' + String(id).padStart(3, '0') + '.json')
      : [];
    const unexpectedComponentManifests = scope.known
      ? archives.filter((_, i) => !scope.expectedComponentIds.includes(ids[i])).map((a) => a.name)
      : [];
    return {
      sourceArchives: archives,
      archiveScope: { ...scope, missingComponentManifests, unexpectedComponentManifests },
      complete:
        scope.known &&
        archives.length > 0 &&
        missingComponentManifests.length === 0 &&
        unexpectedComponentManifests.length === 0 &&
        archives.every(
          (archive) => archive.captureComplete && archive.missingRetainedParts?.length === 0,
        ),
    };
  }

  const blobOf = (data) =>
    typeof data === 'string' ? new Blob([data], { type: 'application/json' }) : data;

  async function copyBlob(directory, name, data, signal) {
    const blob = blobOf(data);
    const file = await directory.getFileHandle(name, { create: true });
    const writable = await file.createWritable();
    let bytesWritten = 0;
    try {
      const counter = new TransformStream({
        transform(chunk, controller) {
          bytesWritten += chunk.byteLength;
          controller.enqueue(chunk);
        },
      });
      await blob.stream().pipeThrough(counter).pipeTo(writable, { signal });
      const written = await file.getFile();
      if (bytesWritten !== blob.size || written.size !== blob.size)
        throw new Error('Saved byte count does not match the retained artifact size.');
      return { bytesWritten, verification: 'stream-completed-and-byte-count-matched' };
    } catch (error) {
      // pipeTo aborts the writable on read/write failure. Never delete source data
      // or claim that an output left by an interrupted write is a complete file.
      try {
        await writable.abort(error);
      } catch {}
      throw error;
    }
  }

  async function saveEntries(files, directory, onFile = () => {}, signal) {
    const results = [];
    for (const [name, data] of files) {
      const expectedBytes = blobOf(data).size;
      let result;
      if (signal?.aborted)
        result = { name, expectedBytes, status: 'not-attempted', error: 'Backup cancelled.' };
      else {
        try {
          result = {
            name,
            expectedBytes,
            status: 'saved',
            ...(await copyBlob(directory, name, data, signal)),
          };
        } catch (error) {
          result = {
            name,
            expectedBytes,
            status: 'failed',
            error: String(error).slice(0, 500),
          };
        }
      }
      results.push(result);
      await onFile(result);
    }
    return results;
  }

  // Returning helpers only without a document permits focused Node stream tests.
  if (typeof document === 'undefined')
    return {
      artifactsFromDocument,
      sourceSummary,
      scopeFromRecord,
      scopeFromDocument,
      sourceCompleteness,
      copyBlob,
      saveEntries,
    };

  if (document.getElementById(panelId)) {
    document.getElementById(panelId).scrollIntoView();
    console.info('The existing recovery panel still holds its original snapshot.');
    return;
  }
  let files;
  try {
    files = artifactsFromDocument(document);
  } catch (error) {
    console.error('TypeCV recovery stopped:', error);
    alert(
      '깊이맵 복구 도구가 현재 탭의 참조를 확실히 찾지 못해 중단했습니다.\n' +
        String(error) +
        '\n기존 실행 기록 JSON과 컴포넌트별 JSON/PLY를 먼저 저장해주세요. 페이지는 변경하지 않았습니다.',
    );
    return;
  }

  const completeness = sourceCompleteness(files, scopeFromDocument(document));
  const results = new Map();
  let directory;
  let controller;
  let busy = false;
  const panel = document.createElement('section');
  panel.id = panelId;
  panel.setAttribute('role', 'region');
  panel.setAttribute('aria-label', 'TypeCV depth artifact recovery');
  Object.assign(panel.style, {
    position: 'fixed',
    top: '12px',
    right: '12px',
    zIndex: '2147483647',
    width: 'min(540px, calc(100vw - 24px))',
    maxHeight: '80vh',
    overflow: 'auto',
    padding: '16px',
    background: '#fff',
    color: '#111',
    border: '2px solid #175cd3',
    borderRadius: '8px',
    boxShadow: '0 4px 28px #0005',
    font: '14px/1.5 system-ui, sans-serif',
  });
  const add = (tag, text, parent = panel) => {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.append(node);
    return node;
  };
  add('strong', 'TypeCV 깊이맵 백업');
  const total = [...files.values()].reduce((n, data) => n + blobOf(data).size, 0);
  add(
    'p',
    files.size +
      '개 파일, 약 ' +
      (total / 1024 ** 3).toFixed(2) +
      ' GiB의 참조를 보존했습니다. 여유 공간이 있는 드라이브의 폴더를 선택하세요. 새 하위 폴더에 스트림으로 복사하며 앱의 원본은 지우지 않습니다.',
  );
  add(
    'p',
    '이 도구는 네트워크 전송을 하지 않습니다. 읽을 수 없어진 Blob은 복구를 보장할 수 없습니다. 실행 기록·컴포넌트 JSON/PLY는 기존 내보내기로 별도 저장하세요.',
  );
  if (!completeness.archiveScope.known)
    add(
      'p',
      '실행 기록에서 전체 컴포넌트 범위를 확인하지 못했습니다. 보관 중인 파일만 백업하며, 모든 깊이맵 묶음이 있다는 보장은 하지 않습니다.',
    );
  const status = add('p', '백업 전입니다. 파일 참조를 유지하려면 이 탭을 열어두세요.');
  status.setAttribute('role', 'status');
  const choose = add('button', '폴더 선택 후 전체 백업');
  choose.type = 'button';
  const cancel = add('button', '백업 중단');
  cancel.type = 'button';
  cancel.disabled = true;
  cancel.addEventListener('click', () => controller?.abort());
  const details = add('details', '');
  add('summary', '개별 파일 다운로드 / 실패 파일 확인', details);
  add('p', '일반 다운로드는 저장 성공을 확인할 수 없어 미검증으로 표시합니다.', details);
  const rows = new Map();
  const download = (name, blob) => {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  };
  for (const [name, data] of files) {
    const row = add('div', '', details);
    const label = add('span', name + ' — 미저장 ', row);
    const button = add('button', '다운로드 요청', row);
    button.type = 'button';
    button.addEventListener('click', () => {
      if (busy) return;
      download(name, blobOf(data));
      label.textContent = name + ' — 다운로드 요청됨 (저장 성공 미검증) ';
    });
    rows.set(name, { label, button });
  }
  const manifest = () => ({
    format: 'typecv-depth-recovery-v1',
    createdAt: new Date().toISOString(),
    retainedFiles: files.size,
    expectedBytes: total,
    allRetainedFilesSaved:
      results.size === files.size && [...results.values()].every((row) => row.status === 'saved'),
    sourceArchivesComplete: completeness.complete,
    archiveScope: completeness.archiveScope,
    sourceArchives: completeness.sourceArchives,
    files: [...files].map(
      ([name, data]) =>
        results.get(name) ?? { name, expectedBytes: blobOf(data).size, status: 'not-attempted' },
    ),
    note: 'Only saved entries have a completed stream and matching byte counts. No cryptographic checksum was measured. Failed outputs must not be treated as complete. Original application refs were not changed.',
  });
  const writeManifest = () =>
    copyBlob(directory, 'typecv-depth-recovery.json', JSON.stringify(manifest(), null, 2));
  const manifestButton = add('button', '백업 기록 JSON 다운로드');
  manifestButton.type = 'button';
  manifestButton.addEventListener('click', () =>
    download(
      'typecv-depth-recovery.json',
      new Blob([JSON.stringify(manifest(), null, 2)], { type: 'application/json' }),
    ),
  );
  choose.disabled = typeof window.showDirectoryPicker !== 'function';
  if (choose.disabled)
    status.textContent =
      '이 브라우저는 폴더 스트림 저장을 지원하지 않습니다. 아래 개별 파일 다운로드를 사용하세요. 다운로드 요청만으로 저장 성공이 확인되지는 않습니다.';
  choose.addEventListener('click', async () => {
    if (busy) return;
    let selected;
    try {
      // Must run directly inside the click handler for the browser's user gesture.
      selected = await window.showDirectoryPicker({ mode: 'readwrite' });
    } catch (error) {
      status.textContent = '폴더 선택이 완료되지 않았습니다. 원본 참조는 유지됩니다. ' + error;
      return;
    }
    busy = true;
    controller = new AbortController();
    choose.disabled = true;
    cancel.disabled = false;
    for (const row of rows.values()) row.button.disabled = true;
    results.clear();
    let manifestError = '';
    try {
      const suffix = Date.now() + '-' + Math.random().toString(36).slice(2, 10);
      directory = await selected.getDirectoryHandle('typecv-depth-recovery-' + suffix, {
        create: true,
      });
      await writeManifest();
      await saveEntries(
        files,
        directory,
        async (result) => {
          results.set(result.name, result);
          rows.get(result.name).label.textContent =
            result.name +
            ' — ' +
            (result.status === 'saved' ? '저장 확인됨' : result.status) +
            (result.error ? ': ' + result.error : '') +
            ' ';
          status.textContent =
            results.size + '/' + files.size + ' 처리됨. 원본 참조는 계속 보존됩니다.';
          try {
            await writeManifest();
          } catch (error) {
            manifestError = String(error).slice(0, 500);
          }
        },
        controller.signal,
      );
      await writeManifest();
      const summary = manifest();
      status.textContent = summary.allRetainedFilesSaved
        ? '보관 중이던 깊이맵 파일의 폴더 저장과 바이트 수 확인이 완료됐습니다. 실행 기록과 컴포넌트 JSON/PLY도 별도로 저장했는지 확인하세요.' +
          (summary.sourceArchivesComplete
            ? ''
            : ' 생성 단계의 누락 또는 미완료가 있으므로 백업 기록 JSON을 확인하세요.')
        : '일부 파일을 저장하지 못했습니다. 탭과 원본 참조를 유지하고 실패 파일을 확인하세요. 다시 시도하면 별도의 새 폴더를 만듭니다.';
    } catch (error) {
      status.textContent = '백업이 완료되지 않았습니다. 원본 참조는 유지됩니다. ' + error;
    } finally {
      if (manifestError)
        status.textContent += ' 백업 기록 저장 중 오류도 있었습니다: ' + manifestError;
      busy = false;
      controller = undefined;
      choose.disabled = false;
      cancel.disabled = true;
      for (const row of rows.values()) row.button.disabled = false;
    }
  });
  document.body.append(panel);
  console.info(
    'TypeCV recovery panel ready: retained artifact references only; no bytes read yet.',
  );
  return { panel: panelId, retainedFiles: files.size, expectedBytes: total };
})();
