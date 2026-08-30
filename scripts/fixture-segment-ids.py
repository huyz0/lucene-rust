#!/usr/bin/env python3
"""Print the Lucene segment id stamped into every committed fixture index.

Fixtures written through a real `IndexWriter` are not byte-reproducible: Lucene
stamps a random 16-byte id (`StringHelper.randomId()`) into every index header,
so `gen-fixtures.sh --check` cannot diff those files against the committed tree.
That blind spot is exactly where the worst fixture accident lives -- a full
`gen-fixtures.sh` run replaces every index with different-but-plausible bytes,
and nothing byte-comparable changes in a way `--check` could name.

The fix is a committed baseline. This script derives the id of every fixture
index from the bytes themselves; `fixtures/segment-ids.txt` is that output,
checked in, and `gen-fixtures.sh --check` re-derives it and diffs. A regenerated
index therefore shows up as one named line ("segment id changed") rather than as
366 opaque binary diffs.

Only `*.si` and `segments_*` are read: both are written by
`CodecUtil.writeIndexHeader`, so the id's position is unambiguous, and every
index directory contains at least one of each. Other codec files carry the same
id, but distinguishing `writeIndexHeader` from `writeHeader` per format would
add a table to maintain for no extra detection.

Header layout (`CodecUtil.writeIndexHeader`):
    magic  0x3fd76c17          4 bytes, big endian
    name   vint length + UTF-8 bytes
    version                    4 bytes, big endian
    id                        16 bytes           <-- what we want

Usage: fixture-segment-ids.py [ROOT]      (default: fixtures/data)
Output: "<relative path> <32-hex id>", sorted, one per line.
"""

from __future__ import annotations

import os
import sys

CODEC_MAGIC = 0x3FD76C17


class MalformedHeader(Exception):
    """The file starts with the codec magic but does not carry an index id."""


def read_vint(data: bytes, pos: int) -> tuple[int, int]:
    """Java `DataInput.readVInt`: 7 bits per byte, high bit continues."""
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise MalformedHeader("truncated vint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, pos
        shift += 7
        if shift > 28:
            raise MalformedHeader("vint too long")


def segment_id(data: bytes) -> str:
    """The 32-hex-char index id from a `writeIndexHeader` prologue."""
    if len(data) < 4 or int.from_bytes(data[0:4], "big") != CODEC_MAGIC:
        raise MalformedHeader("not a Lucene codec header")
    name_len, pos = read_vint(data, 4)
    pos += name_len  # codec name
    pos += 4  # codec version
    if pos + 16 > len(data):
        raise MalformedHeader("truncated index header")
    return data[pos : pos + 16].hex()


def is_index_file(name: str) -> bool:
    return name.endswith(".si") or name.startswith("segments_")


def main(argv: list[str]) -> int:
    root = argv[1] if len(argv) > 1 else "fixtures/data"
    if not os.path.isdir(root):
        print(f"fixture-segment-ids: no such directory: {root}", file=sys.stderr)
        return 2

    rows: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not is_index_file(name):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as handle:
                head = handle.read(64)
            try:
                ident = segment_id(head)
            except MalformedHeader as exc:
                # A deliberately corrupt fixture is legitimate; say so rather
                # than failing, so the baseline still covers everything else.
                ident = f"<unreadable: {exc}>"
            rows.append((os.path.relpath(path, root), ident))

    for rel, ident in sorted(rows):
        print(f"{rel} {ident}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
