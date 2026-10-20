"""Read a split zip archive (`name.z01`, `name.z02`, ..., `name.zip`) without joining its parts.

The standard library's zipfile refuses an archive whose central directory spans two parts,
and reads every local header offset as absolute, while a split archive stores it relative
to the part the entry starts on. This reader parses the end records (ZIP64 included) and
the central directory itself, keeps each entry's part number, and extracts stored and
deflated entries.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

EOCD = b'PK\x05\x06'
EOCD64 = b'PK\x06\x06'
EOCD64_LOCATOR = b'PK\x06\x07'
CENTRAL = b'PK\x01\x02'
LOCAL = b'PK\x03\x04'


@dataclass
class Entry:
    name: str
    method: int
    compressed: int
    size: int
    disk: int
    offset: int


class SplitZip:
    def __init__(self, parts: list[Path]) -> None:
        self.parts = parts
        self.sizes = [part.stat().st_size for part in parts]
        self.starts = [sum(self.sizes[:i]) for i in range(len(parts))]
        self.total = sum(self.sizes)
        self.handles = [part.open('rb') for part in parts]
        self.entries = self._read_directory()

    def close(self) -> None:
        for handle in self.handles:
            handle.close()

    def __enter__(self) -> 'SplitZip':
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_at(self, position: int, size: int) -> bytes:
        out = bytearray()
        while size > 0 and position < self.total:
            index = max(i for i, start in enumerate(self.starts) if start <= position)
            handle = self.handles[index]
            handle.seek(position - self.starts[index])
            chunk = handle.read(min(size, self.starts[index] + self.sizes[index] - position))
            if not chunk:
                break
            out += chunk
            position += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def _read_directory(self) -> list[Entry]:
        tail_size = min(self.total, 65536 + 22)
        tail_at = self.total - tail_size
        tail = self.read_at(tail_at, tail_size)
        at = tail.rfind(EOCD)
        if at < 0:
            raise ValueError('no end of central directory record')
        eocd_at = tail_at + at
        (disk, cd_disk, _this, entries, cd_size, cd_offset) = struct.unpack('<HHHHII', tail[at + 4 : at + 20])
        if 0xFFFF in (disk, cd_disk, entries) or 0xFFFFFFFF in (cd_size, cd_offset):
            locator = self.read_at(eocd_at - 20, 20)
            if locator[:4] != EOCD64_LOCATOR:
                raise ValueError('ZIP64 locator missing')
            z64_disk, z64_offset = struct.unpack('<IQ', locator[4:16])
            record = self.read_at(self.starts[z64_disk] + z64_offset, 56)
            if record[:4] != EOCD64:
                raise ValueError('ZIP64 end record missing')
            (_disk, cd_disk, _this, entries, cd_size, cd_offset) = struct.unpack('<IIQQQQ', record[16:56])
        directory = self.read_at(self.starts[cd_disk] + cd_offset, cd_size)
        out: list[Entry] = []
        at = 0
        for _ in range(entries):
            if directory[at : at + 4] != CENTRAL:
                raise ValueError('central directory entry expected')
            method, = struct.unpack('<H', directory[at + 10 : at + 12])
            compressed, size = struct.unpack('<II', directory[at + 20 : at + 28])
            name_len, extra_len, comment_len = struct.unpack('<HHH', directory[at + 28 : at + 34])
            entry_disk, = struct.unpack('<H', directory[at + 34 : at + 36])
            offset, = struct.unpack('<I', directory[at + 42 : at + 46])
            name = directory[at + 46 : at + 46 + name_len].decode('utf-8', 'replace')
            extra = directory[at + 46 + name_len : at + 46 + name_len + extra_len]
            # ZIP64 extra: the 8-byte fields appear, in order, only for those stored as all ones.
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, length = struct.unpack('<HH', extra[cursor : cursor + 4])
                body = extra[cursor + 4 : cursor + 4 + length]
                if tag == 0x0001:
                    fields = []
                    if size == 0xFFFFFFFF:
                        fields.append('size')
                    if compressed == 0xFFFFFFFF:
                        fields.append('compressed')
                    if offset == 0xFFFFFFFF:
                        fields.append('offset')
                    values = struct.unpack('<' + 'Q' * len(fields), body[: 8 * len(fields)])
                    for field, value in zip(fields, values):
                        if field == 'size':
                            size = value
                        elif field == 'compressed':
                            compressed = value
                        else:
                            offset = value
                    if entry_disk == 0xFFFF:
                        entry_disk, = struct.unpack('<I', body[8 * len(fields) : 8 * len(fields) + 4])
                cursor += 4 + length
            out.append(Entry(name, method, compressed, size, entry_disk, offset))
            at += 46 + name_len + extra_len + comment_len
        return out

    def names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    def extract(self, entry: Entry, directory: Path) -> Path:
        target = directory / entry.name
        if entry.name.endswith('/'):
            target.mkdir(parents=True, exist_ok=True)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        at = self.starts[entry.disk] + entry.offset
        header = self.read_at(at, 30)
        if header[:4] != LOCAL:
            raise ValueError(f'local header of {entry.name} not found')
        name_len, extra_len = struct.unpack('<HH', header[26:30])
        position = at + 30 + name_len + extra_len
        remaining = entry.compressed
        decoder = zlib.decompressobj(-15) if entry.method == 8 else None
        if entry.method not in (0, 8):
            raise ValueError(f'unsupported compression method {entry.method} for {entry.name}')
        with target.open('wb') as out:
            while remaining > 0:
                chunk = self.read_at(position, min(remaining, 8 << 20))
                if not chunk:
                    raise ValueError(f'{entry.name} is truncated')
                position += len(chunk)
                remaining -= len(chunk)
                out.write(decoder.decompress(chunk) if decoder else chunk)
            if decoder:
                out.write(decoder.flush())
        return target
