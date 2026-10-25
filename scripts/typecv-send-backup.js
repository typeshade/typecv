/*
 * Existing TypeCV tab: review this file, then run it as a DevTools Sources > Snippets snippet.
 * Pick the existing backup subfolder containing typecv-depth-recovery.json. This sends only
 * verified saved depth archives through the CURRENT collector's ordinary POST/PUT endpoints.
 * No deployment, reconstruction, application-ref changes or whole-backup memory copy occurs.
 * Network requests start only after the explicit consent checkbox and Send button.
 */
(() => {
  'use strict';
  const NAME = /^typecv-depths-c[0-9]{3}(?:-[0-9]{4}\.tar\.gz|\.json)$/;
  const ID = /^[0-9]{17}-[0-9a-f]{8}$/;
  const MAX_FILE = 50 * 1024 ** 2;
  const MAX_TOTAL = 4 * 1024 ** 3;
  const MAX_COUNT = 1024;
  const MAX_RECORD = 256 * 1024;
  const MAX_MANIFEST = 128 * 1024;
  const WINDOW_MS = 60 * 60_000;
  const PANEL = 'typecv-send-backup-panel';
  const encoder = new TextEncoder();
  const bytesOf = (value) => encoder.encode(JSON.stringify(value)).byteLength;
  const object = (value) => !!value && typeof value === 'object' && !Array.isArray(value);
  const integer = (value) => Number.isSafeInteger(value) && value >= 0;
  const shortText = (value, max) => typeof value === 'string' && value.length <= max;
  const errorText = (error) => String(error).slice(0, 500);

  function validateRecord(record) {
    if (
      !object(record) ||
      !/^typecv-run-record-v[3456]$/.test(record.format ?? '') ||
      !['site', 'client', 'gpu', 'input', 'options', 'result', 'timing'].every((key) =>
        object(record[key]),
      ) ||
      !integer(record.input.frames) ||
      !integer(record.result.registeredViews) ||
      !Number.isFinite(record.timing.totalMs) ||
      record.timing.totalMs < 0 ||
      !Number.isFinite(Date.parse(record.createdAt)) ||
      bytesOf(record) > MAX_RECORD
    )
      throw new Error(
        '지원하는 작은 실행 기록 JSON을 확인하지 못했습니다. 기존 JSON 다운로드를 먼저 보관하세요.',
      );
    return record;
  }

  function recordFromDocument(doc) {
    if (doc.querySelectorAll('.lab-app').length !== 1)
      throw new Error('TypeCV Lab 탭에서만 실행할 수 있습니다.');
    const candidates = doc.querySelectorAll('.run-record details pre');
    if (
      candidates.length !== 1 ||
      !candidates[0].textContent ||
      candidates[0].textContent.length > MAX_RECORD
    )
      throw new Error('완료된 실행의 기록 JSON을 확실히 찾지 못했습니다.');
    return validateRecord(JSON.parse(candidates[0].textContent));
  }

  async function readRecordFile(file) {
    if (!file || !integer(file.size) || file.size > MAX_RECORD)
      throw new Error(
        '실행 기록 JSON은 256 KiB 이하여야 합니다. 큰 복원 JSON 대신 typecv-run-record.json을 고르세요.',
      );
    return validateRecord(JSON.parse(await file.text()));
  }

  function validateManifest(manifest) {
    if (
      !object(manifest) ||
      manifest.format !== 'typecv-depth-recovery-v1' ||
      !integer(manifest.retainedFiles) ||
      manifest.retainedFiles < 1 ||
      manifest.retainedFiles > 10000 ||
      !integer(manifest.expectedBytes) ||
      !Array.isArray(manifest.files) ||
      manifest.files.length !== manifest.retainedFiles ||
      typeof manifest.allRetainedFilesSaved !== 'boolean' ||
      typeof manifest.sourceArchivesComplete !== 'boolean' ||
      !object(manifest.archiveScope) ||
      typeof manifest.archiveScope.known !== 'boolean' ||
      !Array.isArray(manifest.archiveScope.expectedComponentIds) ||
      !Array.isArray(manifest.sourceArchives) ||
      !Number.isFinite(Date.parse(manifest.createdAt)) ||
      bytesOf(manifest) > MAX_MANIFEST
    )
      throw new Error('지원하지 않거나 너무 큰 typecv-depth-recovery.json 입니다.');
    const ids = manifest.archiveScope.expectedComponentIds;
    if (
      ids.length > 1000 ||
      new Set(ids).size !== ids.length ||
      ids.some((id) => !integer(id) || id > 999)
    )
      throw new Error('백업 컴포넌트 범위가 잘못되었습니다.');
    for (const key of ['missingComponentManifests', 'unexpectedComponentManifests']) {
      const list = manifest.archiveScope[key];
      if (
        list !== undefined &&
        (!Array.isArray(list) ||
          list.length > MAX_COUNT ||
          list.some(
            (name) => typeof name !== 'string' || !/^typecv-depths-c[0-9]{3}\.json$/.test(name),
          ))
      )
        throw new Error('백업 컴포넌트 manifest 경로가 잘못되었습니다.');
    }
    if (
      manifest.sourceArchives.length > 1000 ||
      manifest.sourceArchives.some(
        (source) =>
          !object(source) ||
          !/^typecv-depths-c[0-9]{3}\.json$/.test(source.name ?? '') ||
          typeof source.captureComplete !== 'boolean' ||
          (source.captureError != null && !shortText(source.captureError, 500)) ||
          (source.manifestError !== undefined && !shortText(source.manifestError, 500)) ||
          (source.missingDuringCapture != null && !integer(source.missingDuringCapture)) ||
          (source.missingRetainedParts !== undefined &&
            (!Array.isArray(source.missingRetainedParts) ||
              source.missingRetainedParts.length > 10000 ||
              source.missingRetainedParts.some((name) => !shortText(name, 200)))),
      )
    )
      throw new Error('원본 깊이맵 생성 상태가 잘못되었습니다.');
    const archiveNames = manifest.sourceArchives.map((source) => source.name);
    if (new Set(archiveNames).size !== archiveNames.length)
      throw new Error('원본 manifest 이름이 중복되었습니다.');
    const captureComplete =
      manifest.archiveScope.known &&
      manifest.sourceArchives.length > 0 &&
      ids.every((id) =>
        archiveNames.includes('typecv-depths-c' + String(id).padStart(3, '0') + '.json'),
      ) &&
      archiveNames.length === ids.length &&
      !manifest.archiveScope.missingComponentManifests?.length &&
      !manifest.archiveScope.unexpectedComponentManifests?.length &&
      manifest.sourceArchives.every(
        (source) =>
          source.captureComplete &&
          !source.captureError &&
          !source.manifestError &&
          !source.missingDuringCapture &&
          !source.missingRetainedParts?.length,
      );
    if (manifest.sourceArchivesComplete && !captureComplete)
      throw new Error('원본 캡처 완료 표시와 실제 누락 정보가 일치하지 않습니다.');
    const seen = new Set(),
      saved = [],
      excluded = [];
    let expected = 0,
      total = 0;
    for (const row of manifest.files) {
      if (
        !object(row) ||
        typeof row.name !== 'string' ||
        !NAME.test(row.name) ||
        seen.has(row.name) ||
        !integer(row.expectedBytes) ||
        !['saved', 'failed', 'not-attempted'].includes(row.status) ||
        (row.error !== undefined && !shortText(row.error, 500))
      )
        throw new Error(
          '백업 파일 이름·크기·상태가 잘못되었습니다. 경로나 중복 이름은 허용하지 않습니다.',
        );
      seen.add(row.name);
      expected += row.expectedBytes;
      if (!Number.isSafeInteger(expected)) throw new Error('백업 크기 합계가 잘못되었습니다.');
      if (row.status === 'saved') {
        if (
          row.bytesWritten !== row.expectedBytes ||
          row.verification !== 'stream-completed-and-byte-count-matched'
        )
          throw new Error(row.name + ': 저장 완료와 바이트 수 검증을 확인하지 못했습니다.');
        if (row.expectedBytes > MAX_FILE)
          throw new Error(
            row.name +
              ': 현재 서버의 파일당 50 MiB 한도를 넘습니다. 이 도구는 분할 이름으로 우회하지 않습니다.',
          );
        total += row.expectedBytes;
        saved.push(row);
      } else excluded.push(row);
    }
    if (
      expected !== manifest.expectedBytes ||
      manifest.allRetainedFilesSaved !== (excluded.length === 0)
    )
      throw new Error('백업 요약과 파일별 상태가 일치하지 않습니다.');
    if (!saved.length)
      throw new Error(
        '저장 확인된 파일이 없습니다. 백업 완료 후 생성된 실제 하위 폴더를 선택하세요.',
      );
    if (saved.length > MAX_COUNT || total > MAX_TOTAL)
      throw new Error('현재 서버의 1024개·4 GiB 첨부 한도를 넘습니다. 전송하지 않았습니다.');
    return { manifest, saved, excluded, total };
  }

  async function openBackup(directory) {
    const manifestFile = await (
      await directory.getFileHandle('typecv-depth-recovery.json')
    ).getFile();
    if (manifestFile.size > MAX_MANIFEST) throw new Error('백업 기록 JSON이 128 KiB를 넘습니다.');
    const inventory = validateManifest(JSON.parse(await manifestFile.text()));
    for (const row of inventory.saved) {
      const file = await (await directory.getFileHandle(row.name)).getFile();
      if (file.size !== row.expectedBytes)
        throw new Error(row.name + ': 실제 파일 크기가 백업 기록과 다릅니다.');
    }
    return { ...inventory, directory };
  }

  async function fileBytes(inventory, row) {
    const file = await (await inventory.directory.getFileHandle(row.name)).getFile();
    if (file.size !== row.expectedBytes || file.size > MAX_FILE)
      throw new Error(row.name + ': 파일 크기가 검증 이후 바뀌었습니다.');
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (bytes.byteLength !== row.expectedBytes)
      throw new Error(row.name + ': 파일을 전부 읽지 못했습니다.');
    return bytes;
  }
  async function digest(bytes) {
    const hash = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  }
  const checkConsent = (getConsent, signal) => {
    if (!getConsent()) throw new Error('전송 동의가 필요합니다. 네트워크 요청을 하지 않았습니다.');
    signal?.throwIfAborted();
  };

  function createTransfer({
    record,
    inventory,
    origin,
    fetcher,
    getConsent,
    onProgress = () => {},
    onReceipt = async () => {},
    now = () => Date.now(),
    previousReceiptIds = [],
  }) {
    validateRecord(record);
    const url = new URL(origin);
    if (
      url.origin !== origin ||
      !['https:', 'http:'].includes(url.protocol) ||
      url.username ||
      url.password
    )
      throw new Error('잘못된 현재 사이트 주소입니다.');
    if (
      !Array.isArray(previousReceiptIds) ||
      previousReceiptIds.length > 32 ||
      previousReceiptIds.some((id) => !ID.test(id))
    )
      throw new Error('이전 기록 ID가 잘못되었습니다.');
    let busy = false,
      controller,
      entries = [];
    const receipt = {
      format: 'typecv-depth-backup-send-v1',
      origin,
      scope:
        'saved-depth-archives-only; primary reconstruction/cloud/mesh exports are not included',
      createdAt: new Date(now()).toISOString(),
      id: null,
      receivedAt: null,
      uploadWindowEndsAt: null,
      backup: inventory.manifest,
      selected: inventory.saved.length,
      totalBytes: inventory.total,
      sourceArchivesComplete: inventory.manifest.sourceArchivesComplete,
      previousReceiptIds: [...previousReceiptIds],
      excluded: inventory.excluded,
      preparationFailures: [],
      files: [],
      status: 'not-sent',
    };
    const notify = async () => {
      try {
        await onReceipt(receipt);
      } catch (error) {
        receipt.receiptError = errorText(error);
      }
    };
    const request = (path, init) => {
      checkConsent(getConsent, controller.signal);
      return fetcher(new URL(path, origin).href, {
        ...init,
        credentials: 'same-origin',
        mode: 'same-origin',
        redirect: 'error',
        cache: 'no-store',
        signal: controller.signal,
      });
    };
    const expired = () =>
      receipt.uploadWindowEndsAt && now() >= Date.parse(receipt.uploadWindowEndsAt);

    async function uploadPending() {
      for (const entry of entries) {
        const row = receipt.files.find((file) => file.name === entry.name);
        if (row.status === 'stored') continue;
        if (expired()) {
          row.status = 'failed';
          row.error = '서버의 1시간 파일 전송 창이 끝났습니다. 새 기록 전송이 필요합니다.';
          continue;
        }
        checkConsent(getConsent, controller.signal);
        row.status = 'sending';
        for (let attempt = 0; attempt < 2; attempt++) {
          checkConsent(getConsent, controller.signal);
          row.attempts++;
          let bytes;
          try {
            const source = inventory.saved.find((file) => file.name === entry.name);
            bytes = await fileBytes(inventory, source);
            if ((await digest(bytes)) !== entry.sha256)
              throw new Error('검증 이후 파일 내용이 바뀌었습니다.');
            const response = await request('/api/records/' + receipt.id + '/files/' + entry.name, {
              method: 'PUT',
              headers: { 'content-type': 'application/octet-stream' },
              body: bytes,
            });
            const answer = await response.json().catch(() => ({}));
            if (
              response.status === 201 ||
              (response.status === 409 && answer.error === 'already stored')
            ) {
              row.status = 'stored';
              delete row.error;
              break;
            }
            row.error = 'HTTP ' + response.status + (answer.error ? ': ' + answer.error : '');
            if (response.status < 500 && response.status !== 429) break;
          } catch (error) {
            row.error = errorText(error);
            if (controller.signal.aborted || !getConsent()) throw error;
          } finally {
            bytes = undefined;
          }
        }
        if (row.status !== 'stored') row.status = 'failed';
        onProgress({
          stage: 'upload',
          done: receipt.files.filter((file) => file.status === 'stored').length,
          total: entries.length,
          name: entry.name,
        });
        await notify();
      }
      const failed = receipt.files.some((file) => file.status !== 'stored');
      receipt.status =
        failed || receipt.preparationFailures.length ? 'partial' : 'selected-depth-files-stored';
      await notify();
      return receipt;
    }

    async function start() {
      if (busy || receipt.id)
        throw new Error('이미 전송 중이거나 기록이 있습니다. 실패 파일 재시도를 사용하세요.');
      checkConsent(getConsent);
      busy = true;
      controller = new AbortController();
      try {
        entries = [];
        receipt.preparationFailures = [];
        receipt.status = 'preparing';
        for (const [index, row] of inventory.saved.entries()) {
          checkConsent(getConsent, controller.signal);
          let bytes;
          try {
            bytes = await fileBytes(inventory, row);
            entries.push({ name: row.name, bytes: bytes.byteLength, sha256: await digest(bytes) });
          } catch (error) {
            receipt.preparationFailures.push({ name: row.name, error: errorText(error) });
          } finally {
            bytes = undefined;
          }
          onProgress({
            stage: 'hash',
            done: index + 1,
            total: inventory.saved.length,
            name: row.name,
          });
        }
        const prepared = {
          ...record,
          files: entries,
          recovery: {
            format: receipt.format,
            scope: receipt.scope,
            backup: inventory.manifest,
            selectedSavedFiles: inventory.saved.length,
            preparedFiles: entries.length,
            preparationFailures: receipt.preparationFailures,
            previousReceiptIds: [...previousReceiptIds],
            incompleteOriginalCapture: !inventory.manifest.sourceArchivesComplete,
            missingRetainedFiles: inventory.excluded.map((row) => row.name),
          },
        };
        if (bytesOf(prepared) > MAX_RECORD)
          throw new Error(
            '기록과 백업 manifest가 서버의 256 KiB 한도를 넘습니다. 전송하지 않았습니다.',
          );
        const response = await request('/api/records/', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(prepared),
        });
        const answer = await response.json().catch(() => ({}));
        if (
          response.status !== 201 ||
          !ID.test(answer.id ?? '') ||
          !Number.isFinite(Date.parse(answer.receivedAt))
        )
          throw new Error(
            '기록 생성 실패: HTTP ' + response.status + (answer.error ? ': ' + answer.error : ''),
          );
        receipt.id = answer.id;
        receipt.receivedAt = answer.receivedAt;
        receipt.uploadWindowEndsAt = new Date(
          Date.parse(answer.receivedAt) + WINDOW_MS,
        ).toISOString();
        receipt.files = entries.map((entry) => ({ ...entry, status: 'pending', attempts: 0 }));
        await notify();
        return await uploadPending();
      } catch (error) {
        receipt.status = controller.signal.aborted ? 'cancelled' : 'failed';
        receipt.error = errorText(error);
        await notify();
        throw error;
      } finally {
        busy = false;
      }
    }
    async function retry() {
      if (busy || !receipt.id) throw new Error('재시도할 서버 기록 ID가 없습니다.');
      checkConsent(getConsent);
      if (expired())
        throw new Error(
          '1시간 전송 창이 끝났습니다. 남은 파일을 새 기록 ID로 전송 버튼을 사용하세요.',
        );
      busy = true;
      controller = new AbortController();
      try {
        return await uploadPending();
      } finally {
        busy = false;
      }
    }
    return { receipt, start, retry, cancel: () => controller?.abort() };
  }

  if (typeof document === 'undefined')
    return {
      validateRecord,
      recordFromDocument,
      readRecordFile,
      validateManifest,
      openBackup,
      createTransfer,
    };
  if (document.getElementById(PANEL)) {
    document.getElementById(PANEL).scrollIntoView();
    return;
  }
  let record;
  try {
    if (document.querySelectorAll('.lab-app').length !== 1)
      throw new Error('TypeCV Lab 탭에서만 실행할 수 있습니다.');
    if (document.querySelectorAll('.run-record details pre').length)
      record = recordFromDocument(document);
  } catch (error) {
    console.error('TypeCV backup sender stopped:', error);
    alert(String(error));
    return;
  }
  const origin = location.origin;
  const panel = document.createElement('section');
  panel.id = PANEL;
  panel.setAttribute('role', 'region');
  panel.setAttribute('aria-label', 'TypeCV saved depth backup sender');
  Object.assign(panel.style, {
    position: 'fixed',
    top: '12px',
    left: '12px',
    zIndex: '2147483647',
    width: 'min(570px, calc(100vw - 24px))',
    maxHeight: '85vh',
    overflow: 'auto',
    padding: '16px',
    border: '2px solid #175cd3',
    borderRadius: '8px',
    background: 'white',
    color: '#111',
    boxShadow: '0 4px 28px #0005',
    font: '14px/1.5 system-ui,sans-serif',
  });
  const add = (tag, text, parent = panel) => {
    const element = document.createElement(tag);
    element.textContent = text;
    parent.append(element);
    return element;
  };
  add('strong', 'TypeCV 저장된 깊이맵 백업 전송');
  const recordStatus = add(
    'p',
    record
      ? '현재 탭의 실행 기록을 읽었습니다. 저장한 실행 기록 JSON을 별도로 선택할 수도 있습니다.'
      : '실행 기록이 없는 새 Lab 탭입니다. 저장한 작은 typecv-run-record.json을 먼저 선택하세요. 다시 복원할 필요는 없습니다.',
  );
  const recordLabel = add('label', '저장한 실행 기록 JSON: ');
  const recordInput = add('input', '', recordLabel);
  recordInput.type = 'file';
  recordInput.accept = '.json,application/json';
  add(
    'p',
    '저장한 실행 기록 JSON과 백업 폴더를 유지하세요. 깊이 아카이브만 전송하며 복원 JSON·점군·메시는 포함하지 않습니다.',
  );
  add(
    'p',
    '서버 ID를 받은 뒤 이 백업 폴더에 전송 영수증 JSON을 기록합니다. 원본 깊이 파일과 백업 기록은 수정하지 않습니다.',
  );
  add(
    'p',
    '전송 대상: ' +
      origin +
      '/api/records/ · 파일당 50 MiB, 합계 4 GiB, 최대 1024개. 기록 생성 후 파일 전송 창은 1시간입니다.',
  );
  const choose = add('button', '저장된 백업 하위 폴더 선택');
  choose.type = 'button';
  const summary = add(
    'p',
    'typecv-depth-recovery.json이 있는 실제 백업 하위 폴더를 선택하세요. 폴더 선택만으로 전송하지 않습니다.',
  );
  const consentLabel = add('label', '');
  const consent = add('input', '', consentLabel);
  consent.type = 'checkbox';
  consent.checked = false;
  add(
    'span',
    ' 위 대상에 실행 기록, 저장 확인된 깊이 파일과 누락 상태를 보내는 데 동의합니다.',
    consentLabel,
  );
  const send = add('button', '선택 파일 검증 후 전송');
  send.type = 'button';
  send.disabled = true;
  const retry = add('button', '실패한 전송만 같은 ID로 재시도');
  retry.type = 'button';
  retry.disabled = true;
  const fresh = add('button', '남은 파일을 새 기록 ID로 전송');
  fresh.type = 'button';
  fresh.disabled = true;
  const stop = add('button', '전송 중단');
  stop.type = 'button';
  stop.disabled = true;
  const status = add('p', '전송 전입니다.');
  status.setAttribute('role', 'status');
  const receiptId = add('pre', '서버 기록 ID: 아직 없음');
  const save = add('button', '영수증 JSON을 백업 폴더에 저장');
  save.type = 'button';
  save.disabled = true;
  const download = add('button', '영수증 JSON 다운로드 요청');
  download.type = 'button';
  download.disabled = true;
  let inventory,
    transfer,
    busy = false;
  const previousReceiptIds = [];
  const update = () => {
    send.disabled = busy || !record || !inventory || !consent.checked || !!transfer?.receipt.id;
    retry.disabled = busy || !transfer?.receipt.id || !consent.checked;
    fresh.disabled =
      busy ||
      !transfer?.receipt.id ||
      !consent.checked ||
      !inventory?.saved.some(
        (row) =>
          !transfer.receipt.files.some(
            (file) => file.name === row.name && file.status === 'stored',
          ),
      );
    choose.disabled = busy || !record || typeof window.showDirectoryPicker !== 'function';
    recordInput.disabled = busy;
    stop.disabled = !busy;
    save.disabled = !transfer?.receipt.id;
    download.disabled = !transfer?.receipt.id;
  };
  const storeReceipt = async (receipt) => {
    receiptId.textContent =
      '서버 기록 ID: ' +
      (receipt.id ?? '아직 없음') +
      (receipt.uploadWindowEndsAt
        ? '\n파일 전송 창 종료: ' +
          new Date(receipt.uploadWindowEndsAt).toLocaleString('ko-KR', {
            timeZone: 'Asia/Seoul',
            hour12: false,
          }) +
          ' (한국시간)'
        : '');
    update();
    if (receipt.id && inventory) {
      try {
        const file = await inventory.directory.getFileHandle(
          'typecv-backup-send-' + receipt.id + '.json',
          { create: true },
        );
        const writable = await file.createWritable();
        await writable.write(JSON.stringify(receipt, null, 2));
        await writable.close();
      } catch (error) {
        receipt.receiptError = errorText(error);
        receiptId.textContent +=
          '\n영수증 폴더 저장 실패: 위 ID를 복사하세요. ' + receipt.receiptError;
      }
    }
  };
  const renderOutcome = (receipt) => {
    const stored = receipt.files.filter((row) => row.status === 'stored').length;
    status.textContent =
      '기록 ID ' +
      receipt.id +
      ' · 깊이 파일 ' +
      stored +
      '/' +
      receipt.selected +
      '개 저장됨. ' +
      '백업 제외 ' +
      receipt.excluded.length +
      '개, 준비 실패 ' +
      receipt.preparationFailures.length +
      '개. ' +
      '원본 캡처 ' +
      (receipt.sourceArchivesComplete ? '완료 표시' : '누락·미완료 있음') +
      '. 전체 복원 파일 전송 완료를 뜻하지 않습니다.';
  };
  recordInput.addEventListener('change', async () => {
    if (busy || !recordInput.files?.[0]) return;
    consent.checked = false;
    transfer = undefined;
    inventory = undefined;
    previousReceiptIds.length = 0;
    record = undefined;
    receiptId.textContent = '서버 기록 ID: 아직 없음';
    summary.textContent = '실행 기록이 바뀌었습니다. 같은 실행의 백업 하위 폴더를 다시 선택하세요.';
    update();
    try {
      record = await readRecordFile(recordInput.files[0]);
      recordStatus.textContent =
        '저장한 실행 기록 JSON을 읽었습니다. 복원·재실행 없이 이 기록을 전송합니다.';
      status.textContent = '기록 확인 완료. 아직 전송하지 않았습니다.';
    } catch (error) {
      recordStatus.textContent = '실행 기록을 읽지 못했습니다: ' + errorText(error);
      status.textContent = '전송하지 않았습니다. 올바른 작은 실행 기록 JSON을 선택하세요.';
    }
    update();
  });
  choose.addEventListener('click', async () => {
    if (busy) return;
    try {
      const directory = await window.showDirectoryPicker({ mode: 'readwrite' });
      inventory = await openBackup(directory);
      transfer = undefined;
      consent.checked = false;
      const gaps = inventory.manifest.sourceArchives.reduce(
        (n, row) => n + (row.missingDuringCapture ?? 0),
        0,
      );
      summary.textContent =
        '저장 확인된 깊이 파일 ' +
        inventory.saved.length +
        '개, ' +
        (inventory.total / 1024 ** 3).toFixed(2) +
        ' GiB. 제외 파일 ' +
        inventory.excluded.length +
        '개. 원본 캡처 누락 ' +
        gaps +
        '개. ' +
        (inventory.manifest.sourceArchivesComplete ? '' : '원본 캡처는 미완료입니다. ') +
        '동의 후 전송 버튼을 눌러야 네트워크 전송을 시작합니다.';
      status.textContent = '폴더 확인 완료. 아직 전송하지 않았습니다.';
    } catch (error) {
      inventory = undefined;
      status.textContent = errorText(error);
    }
    update();
  });
  consent.addEventListener('change', () => {
    if (!consent.checked && busy) transfer?.cancel();
    update();
  });
  const run = async (resume) => {
    if (busy || !record || !inventory || !consent.checked) return;
    busy = true;
    update();
    try {
      transfer ??= createTransfer({
        record,
        inventory,
        origin,
        fetcher: (...args) => fetch(...args),
        getConsent: () => consent.checked,
        onReceipt: storeReceipt,
        previousReceiptIds,
        onProgress: (value) => {
          status.textContent =
            (value.stage === 'hash' ? 'SHA 검증 ' : '서버 저장 ') +
            value.done +
            '/' +
            value.total +
            ' · ' +
            value.name;
        },
      });
      renderOutcome(await (resume ? transfer.retry() : transfer.start()));
    } catch (error) {
      status.textContent = errorText(error) + ' 백업과 원본 탭은 유지됩니다.';
    } finally {
      busy = false;
      update();
    }
  };
  send.addEventListener('click', () => run(false));
  retry.addEventListener('click', () => run(true));
  fresh.addEventListener('click', async () => {
    if (busy || !consent.checked || !transfer?.receipt.id || !inventory) return;
    const remaining = inventory.saved.filter(
      (row) =>
        !transfer.receipt.files.some((file) => file.name === row.name && file.status === 'stored'),
    );
    if (!remaining.length || previousReceiptIds.length >= 32) return;
    previousReceiptIds.push(transfer.receipt.id);
    inventory = {
      ...inventory,
      saved: remaining,
      total: remaining.reduce((sum, row) => sum + row.expectedBytes, 0),
    };
    transfer = undefined;
    await run(false);
  });
  stop.addEventListener('click', () => transfer?.cancel());
  save.addEventListener('click', async () => {
    if (!transfer?.receipt.id || !inventory) return;
    try {
      const name = 'typecv-backup-send-' + transfer.receipt.id + '-' + Date.now() + '.json';
      const file = await inventory.directory.getFileHandle(name, { create: true });
      const writable = await file.createWritable();
      await writable.write(JSON.stringify(transfer.receipt, null, 2));
      await writable.close();
      status.textContent = '전송 영수증 저장: ' + name + '. 백업 원본은 변경하지 않았습니다.';
    } catch (error) {
      status.textContent =
        '영수증 저장 실패: ' + errorText(error) + ' 위 서버 기록 ID를 복사해 보관하세요.';
    }
  });
  download.addEventListener('click', () => {
    if (!transfer?.receipt.id) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(transfer.receipt, null, 2)], { type: 'application/json' }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-backup-send-' + transfer.receipt.id + '.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  });
  if (typeof window.showDirectoryPicker !== 'function') {
    choose.disabled = true;
    status.textContent = '폴더 접근을 지원하는 Chrome이 필요합니다. 원본 탭을 새로고침하지 마세요.';
  }
  update();
  document.body.append(panel);
  return { panel: PANEL, destination: origin + '/api/records/', networkStarted: false };
})();
