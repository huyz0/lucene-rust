# M2 sweep — batch `b1-util-store`

Scope: every function in `crates/lucene-util/src/{lib,base36,fixed_bit_set,
small_float,term_interner,zigzag}.rs` and `crates/lucene-store/src/{lib,
codec_util,data_input,data_output,directory,error,index_output}.rs`, compared
method-by-method against Lucene 10.5.0 at `/home/tuong/work/lucene`.

Gate: `cargo fmt --all`, `cargo clippy -p lucene-util -p lucene-store
--all-targets -- -D warnings`, `cargo test -p lucene-util -p lucene-store` —
all green. Per-file line coverage for the two crates (tests of these two
crates only): every swept file ≥ 97.7%.

Findings: **8 CORRECTNESS/MISSING fixed**, 3 PERF (2 fixed + measured, 1
recorded), 14 INTENTIONAL/scope.

---

## crates/lucene-util/src/lib.rs

Java counterpart: none (crate root; module declarations and re-exports only).

| Rust | Java | Status |
|---|---|---|
| `pub mod …`, `pub use FixedBitSet, TermId, TermInterner` | — | not-in-Java (glue) |

### Verdict
Swept-clean. No change.

---

## crates/lucene-util/src/base36.rs

Java counterparts: `java.lang.Long.toString(long, int)` /
`Long.parseLong(String, int)` as used by
`lucene/core/src/java/org/apache/lucene/index/SegmentInfos.java`
(`generationFromSegmentsFileName`, `getNextPendingGeneration`) and
`.../index/IndexFileNames.java` (`fileNameFromGeneration`).

| Rust | Java | Status |
|---|---|---|
| `to_base36(i64)` | `Long.toString(n, Character.MAX_RADIX)` | identical |
| `from_base36(&str)` | `Long.parseLong(s, 36)` | identical |

Checked explicitly: `i64::MIN` (`unsigned_abs`, so no overflow — Java produces
the same `-1y2p0ij32e8e8`), zero, leading `+`/`-` (both accepted by
`i64::from_str_radix` and `Long.parseLong`), uppercase digits (both accept),
empty string and overflow (both reject; Rust as `None`, Java as
`NumberFormatException`).

### Verdict
Swept-clean. No divergence found; no change.

---

## crates/lucene-util/src/zigzag.rs

Java counterpart: `lucene/core/src/java/org/apache/lucene/util/BitUtil.java`
(`zigZagEncode`/`zigZagDecode`, both overloads).

| Rust | Java | Status |
|---|---|---|
| `encode(i64) -> u64` | `zigZagEncode(long)` | identical |
| `decode(u64) -> i64` | `zigZagDecode(long)` | identical |
| `encode_i32(i32) -> u32` | `zigZagEncode(int)` | **added** (finding 1) |
| `decode_i32(u32) -> i32` | `zigZagDecode(int)` | **added** (finding 1) |

### 1. [MISSING] the 32-bit zigzag pair was not ported
Java has `zigZagEncode(int)`/`zigZagDecode(int)`, which is what
`DataInput.readZInt`/`DataOutput.writeZInt` are built on — a different
function from the 64-bit pair, not a narrowing of it (the sign bit is at bit
31). Consequence: reachable — `lucene-codecs/src/stored_fields.rs` had to
hand-roll both halves privately, so the primitive existed twice with no shared
tests. Resolution: **fixed** — `encode_i32`/`decode_i32` in `zigzag.rs`, with
round-trip, known-value (`-1 → 1`, `i32::MIN → u32::MAX`) and
"agrees with the 64-bit variant over the i32 range" tests.

### Verdict
Swept-clean after the fix. (The `codecs` copies still exist and should be
migrated to these in a later batch — noted as open, not a correctness risk.)

---

## crates/lucene-util/src/small_float.rs

Java counterpart:
`lucene/core/src/java/org/apache/lucene/util/SmallFloat.java`.

| Rust | Java | Status |
|---|---|---|
| `long_to_int4(u64)` | `longToInt4(long)` | identical (Java's `i < 0` throw is unrepresentable on `u64`) |
| `int4_to_long(u32)` | `int4ToLong(int)` | identical |
| `num_free_values()` | `NUM_FREE_VALUES` static | identical (derived, not hardcoded) |
| `int_to_byte4(u32)` | `intToByte4(int)` | divergent → fixed (finding 2) |
| `byte4_to_int(u8)` | `byte4ToInt(byte)` | identical |
| — | `floatToByte(float,int,int)`, `byteToFloat`, `floatToByte315`, `byteToFloat315` | not ported — scope (the general 24-bit-mantissa pair; no format this port reads uses it). Recorded, no caller |

### 2. [MISSING] `intToByte4` accepted input outside Java's domain and wrapped
Java throws `IllegalArgumentException` for negative input, so its reachable
domain is `0..=Integer.MAX_VALUE`, and `NUM_FREE_VALUES` is chosen so that the
top of that domain encodes to exactly 255. This port takes `u32`, which can
express values Java cannot: `int_to_byte4(4_000_000_000)` computed
`24 + 239 = 263` and truncated to byte `7` — i.e. a 4-billion-token field
would encode *shorter* than a 7-token one, inverting norm ordering. Not
reachable from today's callers (field lengths are `i32`-bounded), but the
signature invites it. Resolution: **fixed** — `debug_assert!` on Java's domain
plus a release-mode saturate to `i32::MAX` (keeps the encoding monotonic,
which is the one property every caller relies on); tests for the saturation
value and the debug assertion.

### Verdict
Swept-clean after the fix.

---

## crates/lucene-util/src/fixed_bit_set.rs

Java counterpart:
`lucene/core/src/java/org/apache/lucene/util/FixedBitSet.java` (and its
`BitSet`/`Bits` supertypes).

| Rust | Java | Status |
|---|---|---|
| `bits2words(usize)` | `bits2words(int)` | identical result (Rust's explicit `0 → 0` guard equals Java's `((0-1)>>6)+1 == 0`) |
| `FixedBitSet::from_words` | `FixedBitSet(long[], int)` | divergent → fixed (finding 3) |
| `new(num_bits)` | `FixedBitSet(int)` | identical |
| `len()` | `length()` | identical |
| `is_empty()` | — | not-in-Java (Rust idiom, pairs with `len`) |
| `get/set/clear(index)` | `get/set/clear(int)` | identical (same word/bit layout) |
| `cardinality()` | `cardinality()` | identical (whole-word popcount) |
| `words()` | `getBits()` | identical |
| — | `getAndSet`, `getAndClear`, `flip`, `set(from,to)`, `clear(from,to)`, `nextSetBit`, `nextClearBit`, `prevSetBit`, `or/and/xor/andNot`, `intersects`, `scanIsEmpty`, `copyOf`, `ensureCapacity`, `orRange/andRange`, `applyMask`, `forEach`, `intoArray`, `approximateCardinality`, `ramBytesUsed`, `equals/hashCode/clone` | not ported — scope. `.liv` decoding needs get/set/clear/cardinality only; nothing in this port can reach the rest. Recorded, no caller. `nextSetBit`/`intoArray` become relevant when live-docs-driven iteration lands |

### 3. [MISSING] no ghost-bit verification in the raw-words constructor
Java's `FixedBitSet(long[] storedBits, int numBits)` ends with
`assert verifyGhostBitsClear()`: bits above `numBits` in the final word must
be zero. That matters because `cardinality()` counts *whole words* in both
implementations — a `.liv` file whose trailing word carries junk silently
inflates the live-doc count instead of being rejected. Resolution: **fixed** —
`ghost_bits_clear()` (a direct port of `verifyGhostBitsClear`) as a
`debug_assert!` in `from_words`, with a positive test (a full 64-bit word is
legal) and a `#[should_panic]` negative test. Also documented the one
deliberate difference: Java allows `storedBits` *longer* than
`bits2words(numBits)`, this port requires the exact length.

### Verdict
Swept-clean after the fix; the unported API surface is scope, listed above.

---

## crates/lucene-util/src/term_interner.rs

Java counterpart: **none, deliberately.** The module doc already says it is
*not* a port of `util/BytesRefHash.java` (whose sort/compact/rehash machinery
is tied to `ByteBlockPool` and Lucene's int-allocation strategy). It is a
Rust-side primitive with the same purpose (stable integer handle for recurring
byte strings) and no on-disk format.

| Rust | Nearest Java | Status |
|---|---|---|
| `intern`, `intern_str`, `get`, `len`, `is_empty`, `with_capacity`, `TermId::index` | `BytesRefHash.add/get/size` | not-in-Java (different contract: Java's `add` returns `-(id+1)` for an existing entry; ours returns the id) |

### 4. [INTENTIONAL] not wired into any indexing path
Recorded, not a defect: the module doc states it and `PLAN.md` tracks the
integration. Worth revisiting when the indexing chain lands, at which point the
`BytesRefHash` comparison becomes a real design question (arena-backed storage
instead of `Box<[u8]>` per term).

### Verdict
Swept-clean (no Java counterpart to diverge from).

---

## crates/lucene-store/src/lib.rs

Java counterpart: none (crate root).

### Verdict
Swept-clean. No change (the new APIs are reachable via the already-public
`data_input`/`data_output`/`codec_util` modules).

---

## crates/lucene-store/src/error.rs

Java counterparts: `java.io.IOException`/`EOFException`,
`org.apache.lucene.index.CorruptIndexException`,
`IndexFormatTooOldException`/`IndexFormatTooNewException`.

| Rust | Java | Status |
|---|---|---|
| `Error::Eof` | `EOFException` | identical role |
| `Error::MalformedVarint` | (Java 10.5 has no such exception — see finding 5) | stricter |
| `Error::Corrupted(String)` | `CorruptIndexException` | identical role |
| `Error::Io` | `IOException` | identical role |
| — | `IndexFormatTooOld/TooNewException` | folded into `Corrupted` — INTENTIONAL: the message text is preserved and no caller in this port discriminates on the type |

### Verdict
Swept-clean. No change.

---

## crates/lucene-store/src/data_input.rs

Java counterparts: `lucene/core/src/java/org/apache/lucene/store/DataInput.java`,
`.../store/IndexInput.java`, `.../util/GroupVIntUtil.java`,
`.../codecs/CodecUtil.java` (`readBEInt`/`readBELong`).

| Rust | Java | Status |
|---|---|---|
| `read_byte`, `read_bytes` | `readByte`, `readBytes` | identical |
| `remaining` | (`IndexInput.length() - getFilePointer()`) | equivalent |
| `read_u32_le` / `read_u32s_le` | `readInt` / `readInts` | identical (SliceInput overrides with a bulk path, as `MemorySegmentIndexInput` does) |
| `read_vint` | `readVInt` | divergent → fixed (finding 5) |
| `read_zint` | `readZInt` | **added** (finding 6) |
| `read_vlong` | `readVLong` | identical byte consumption (≤10 bytes); errors instead of silently truncating a 10th continuation bit, which `writeVLong`/`writeSignedVLong` can never emit |
| `read_zlong` | `readZLong` | identical byte consumption; same stricter-tail note |
| `read_group_vints` | `GroupVIntUtil.readGroupVInts` | divergent → fixed (findings 7, 11) |
| `peek_u32_le`, `skip` | `RandomAccessInput.readInt(pos)` + `IndexInput.seek` / `skipBytes` | equivalent |
| `read_be_u32`/`read_be_u64`/`read_be_i32` | `CodecUtil.readBEInt`/`readBELong` | identical |
| `read_string` | `readString` | divergent → fixed (finding 8) |
| `read_i16` / `read_u16` | `readShort` / `Short.toUnsignedInt(readShort())` | identical |
| `read_i32` / `read_i64` | `readInt` / `readLong` | identical |
| `read_i64s` | `readLongs` | identical semantics; bulk override added (finding 12) |
| `read_map_of_strings` / `read_set_of_strings` | `readMapOfStrings` / `readSetOfStrings` | divergent → fixed (finding 8); returns `Vec` rather than `Map`/`Set` — INTENTIONAL (callers need order and don't need hashing) |
| `SliceInput::{new,position,len,is_empty,seek,slice,slice_input,as_slice,clone}` | `IndexInput.{getFilePointer,length,seek,slice,clone}` | identical semantics (independent file pointers verified by test) |
| — | `readShorts`, `readFloats`, `readBytes(…,useBuffer)`, `skipBytes(long)`, `readVInt$Baseline`-style helpers | not ported — scope. `readFloats` lands with the vector formats; `useBuffer` is a `BufferedIndexInput` concern this port has no analogue for; `skipBytes` is `skip(usize)` |
| — | `IndexInput.randomAccessSlice`, `prefetch`, `updateReadAdvice`, `ChecksumIndexInput`, `BufferedChecksum` | not ported — scope (see finding 14) |

### 5. [CORRECTNESS] `read_vint` could consume a 6th byte and desynchronize the stream
Java 10.5's `readVInt` is `for (int shift = 0; shift < 32; shift += 7)` — five
bytes maximum, then it stops whether or not the fifth byte still has its
continuation bit, silently ignoring the excess. This port's loop guarded on
`shift > 28 + 7`, so for input like `FF FF FF FF FF 00 …` it read a **sixth**
byte (and could return a value with no error at all, e.g. when that sixth byte
had no continuation bit). Consequence: on corrupt or adversarial input the
file pointer ends up one or more bytes ahead of where Java leaves it, so every
subsequent field in the same file decodes garbage — a silent mis-parse, worse
than the bogus value itself. Resolution: **fixed** — bound at five bytes
exactly, and report the impossible fifth continuation bit as
`MalformedVarint` (deliberately stricter than Java, and unreachable from data
`writeVInt` can produce, whose fifth byte is always `i >>> 28`). Tests: exact
position after a malformed 5-byte run, and that a junk-bearing fifth byte
(`0x7F`) decodes identically to the clean one (`0x0F`), matching Java's
`(b & 0x7F) << 28`.

### 6. [MISSING] `readZInt` was not ported
Reachable: `lucene-codecs/src/stored_fields.rs` hand-rolled `read_zint`
privately because the primitive was absent. Resolution: **fixed** —
`DataInput::read_zint` on top of the new `zigzag::decode_i32`, with a
boundary test (small negatives cost one byte, `i32::MIN`/`MAX` cost five) and
a `proptest` round-trip.

### 7. [CORRECTNESS] `read_group_vints` broke on any backend that cannot peek
The method documented that a backend unable to peek "may return Eof to force
the safe path", but the code was `self.peek_u32_le()? & MASKS[n]` — the `?`
propagated that Eof straight out, so such a backend got a spurious error
instead of the fallback. The in-module `PlainInput` test backend documents
exactly that contract and never exercised the group-vint path, so nothing
caught it. Resolution: **fixed** — the branchless path is now chosen once per
group by probing `peek_u32_le` (plus the remaining-bytes check of finding 11),
with a byte-at-a-time fallback otherwise. Tests: `PlainInput` and `SliceInput`
must decode the same group to the same values; last-group-of-file (<16 bytes
remaining, widths 4/4/4/1) decodes without over-reading; a truncated group is
`Eof`, not a silent zero.

### 8. [CORRECTNESS] untrusted lengths drove allocations
`read_string` did `read_vint()? as usize` then `vec![0u8; len]`, and
`read_map_of_strings`/`read_set_of_strings` did `Vec::with_capacity(count)` on
the same unvalidated count. A negative vint (Java: `NegativeArraySizeException`)
becomes ~2^64 here, and a 5-byte corrupt header could request a 2^31-entry
`Vec`. Consequence: a corrupt or hostile segment file aborts the process on
allocation failure — an abort, unlike a panic, cannot be intercepted by
`lucene-ffi`'s `catch_unwind`, so it takes the JVM down with it. Resolution:
**fixed** — a shared `read_length()` rejects negatives and any length beyond
`remaining()` (every element costs at least one byte, so this can never reject
valid input) before allocating. Tests for negative length, a 1 GiB claim over
a 3-byte input, and both collection readers.

### 9. [INTENTIONAL] `read_string` rejects invalid UTF-8
Java's `new String(bytes, UTF_8)` substitutes U+FFFD. Erroring is strictly
better here (a corrupt codec/segment name would otherwise fail later, at a
confusing comparison) and cannot differ on valid input. Pre-existing behaviour,
now documented on the method.

### 10. [INTENTIONAL] `read_vlong`/`read_zlong` reject an over-long tail
Java 10.5 exits its loop at `shift >= 64` without complaint. Both consume the
same ≤10 bytes as Java, so no desynchronization is possible; only the "silently
accept garbage" ending differs.

### 11. [PERF] the branchless group-varint path was decided per value, not per group
Java's `GroupVIntUtil.readGroupVInt` checks `length - pos >= 4 * Integer.BYTES`
**once** per group and then does four unchecked absolute reads; this port
checked `remaining() >= 4` before each of the four values. Same results, four
branches instead of one on the hottest postings decode loop. Resolution:
**fixed** in the same restructure as finding 7 and measured with the existing
criterion bench on the Java-produced `group_vint.bin` fixture (1024 values):
`data_input/read_group_vints_block` **827 ns → 743 ns, −10.4%**
(p < 0.05, change isolated to this hunk with everything else within ±0.3%).

### 12. [PERF] `read_i64s` had no bulk override
`SliceInput` overrode `read_u32s_le` with a single bounds check for the whole
run but left `read_i64s` on the default per-word loop — one bounds check and
one `Result` per word — even though `.liv` live-docs bitsets, BKD leaf blocks
and stored-fields bitsets all read hundreds of words at a time through it
(`live_docs.rs`, `points.rs`, `stored_fields.rs`). Java's
`MemorySegmentIndexInput.readLongs` is a bulk copy. Resolution: **fixed** —
bulk override mirroring `read_u32s_le`, plus a new bench. Measured on a
2048-word (16384-doc) bitset: `data_input/read_i64s_block`
**421 ns → 117 ns, −73%**. Tests: agreement with the default implementation,
all-or-nothing EOF, empty-slice no-op.

### 13. [PERF, recorded] `read_vlong` microbench moved +12% from code layout
After the finding-5 fix the `read_vlong` bench regressed ~12% even though
`read_vlong`'s source is byte-identical. Bisected: the regression appears when
*only* the loop-bound constant inside `read_vint` changes (`> 35` → `> 28`,
which changes how far LLVM unrolls that loop) and disappears in every variant
where that constant is left alone; `read_length`, `read_zint` and the
group-varint restructure are each neutral for `read_vlong` in isolation. So
this is bench-binary code alignment, not a cost in the vlong decoder — the
same edit also moved the (unmodified) group-vint bench by −5%. Recorded rather
than "fixed": the correctness fix is not negotiable, an alternative
Java-shaped `for shift in [0,7,14,21,28]` formulation measured *worse*
(read_vint +13%), and the artifact will re-shuffle with the next unrelated
edit. Control run confirmed the machine itself was stable (old code reproduced
its own baseline to within 0.4%).

### Verdict
Swept-clean after findings 5–8 and 11–12; finding 13 recorded with
measurements.

---

## crates/lucene-store/src/data_output.rs

Java counterpart: `lucene/core/src/java/org/apache/lucene/store/DataOutput.java`
(+ `util/GroupVIntUtil.writeGroupVInts`).

| Rust | Java | Status |
|---|---|---|
| `write_byte`, `write_bytes` | `writeByte`, `writeBytes` | identical |
| `write_vint` | `writeVInt` | identical (verified byte-for-byte against the Java fixture) |
| `write_vlong` | `writeVLong` | divergent → fixed (finding 15) |
| `write_zlong` / `write_vlong_raw_u64` | `writeZLong` / `writeSignedVLong` | identical (fixture-verified) |
| `write_zint` | `writeZInt` | **added** (finding 14) |
| `write_group_vints` | `writeGroupVInts` / `GroupVIntUtil.writeGroupVInts` | **added** (finding 14) |
| `write_be_u32`/`write_be_u64` | `CodecUtil.writeBEInt`/`writeBELong` | identical |
| `write_i16`/`write_i32`/`write_i64` | `writeShort`/`writeInt`/`writeLong` | identical |
| `write_string` | `writeString` | identical (`s.len()` is the UTF-8 byte length, as Java's `new BytesRef(s).length` is; Rust `&str` cannot hold the lone surrogates Java's `UnicodeUtil` has to sanitize) |
| `write_map_of_strings`/`write_set_of_strings` | `writeMapOfStrings`/`writeSetOfStrings` | identical |
| — | `copyBytes(DataInput, long)` | not ported — scope, no caller today (a CFS/merge writer will want it; it is a 6-line loop over `read_bytes`/`write_bytes` with no format content) |
| `VecDataOutput`, `impl DataOutput for Vec<u8>` | `ByteBuffersDataOutput` | equivalent role, simpler (single growable buffer vs. block list) — INTENTIONAL |

### 14. [MISSING] `writeZInt` and `writeGroupVInts` were not ported
Both are reachable and both had already been duplicated privately inside
`lucene-codecs` (`stored_fields.rs::write_zint`,
`postings.rs::write_group_vints`) — the clearest possible evidence that the
primitive belongs in `lucene-store`. Resolution: **fixed** — both added to the
`DataOutput` trait. `write_group_vints` is a direct port of
`GroupVIntUtil.writeGroupVInts` including its `numBytes(v) = 4 -
(numberOfLeadingZeros(v | 1) >> 3)` width rule and vint tail. Tested with a
**write-side differential test**: re-encoding the 1024 values from Java's own
`fixtures/data/group_vint.expected` must reproduce `group_vint.bin`
byte-for-byte — which it does. The same test now also pins `write_vint`,
`write_vlong` and `write_zlong` against their Java fixture bytes (previously
only the decoders were fixture-checked; a self-consistent but wrong encoder
would have passed the old round-trip tests).

### 15. [MISSING] `write_vlong` silently accepted negatives
Java throws `IllegalArgumentException("cannot write negative vLong")` —
the 10-byte encoding is reserved for `writeZLong`/internal use. Consequence:
a caller bug produced a 10-byte field where Java's writer would have failed
loudly. Resolution: **fixed** — `debug_assert!` with Java's message (release
builds still emit the same bytes Java's private `writeSignedVLong` would, and
`read_vlong` decodes them, so this never corrupts a stream); `#[should_panic]`
test. The whole workspace test suite passes with the assertion enabled, so no
existing writer violates it.

### Verdict
Swept-clean after findings 14–15.

---

## crates/lucene-store/src/codec_util.rs

Java counterpart: `lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java`.

| Rust | Java | Status |
|---|---|---|
| `CODEC_MAGIC`, `FOOTER_MAGIC`, `FOOTER_LENGTH`, `ID_LENGTH` | same constants / `footerLength()` / `StringHelper.ID_LENGTH` | identical |
| `check_header` | `checkHeader` | identical |
| `check_header_no_magic` | `checkHeaderNoMagic` | identical (version errors folded into `Corrupted`, see error.rs) |
| `check_index_header` | `checkIndexHeader` | identical |
| `check_index_header_id` | `checkIndexHeaderID` | identical |
| `check_index_header_suffix` | `checkIndexHeaderSuffix` | identical but rejects invalid UTF-8 (same call as finding 9) |
| `write_header` | `writeHeader` | divergent → fixed (finding 17) |
| `write_index_header` | `writeIndexHeader` | divergent → fixed (finding 17) |
| `write_footer` | `writeFooter` | identical bytes; takes `&mut Vec<u8>` because the CRC must cover what was already written — INTENTIONAL (Java gets it from `IndexOutput.getChecksum()`) |
| `check_footer` | `checkFooter(ChecksumIndexInput)` + `validateFooter` | identical checks; one merged message where Java has "file truncated?" / "file extended?" — INTENTIONAL |
| `retrieve_checksum` | `retrieveChecksum(IndexInput)` | identical |
| `retrieve_checksum_with_expected_length` | `retrieveChecksum(IndexInput, long)` | **added** (finding 16) |
| `header_length` / `index_header_length` | `headerLength` / `indexHeaderLength` | **added** (finding 16) |
| `check_whole_file_header`/`check_whole_file_footer` | — | not-in-Java (convenience over the whole-file `Input` model) |
| — | `checksumEntireFile`, `verifyAndCopyIndexHeader`, `readIndexHeader`, `readFooter`, `checkFooter(in, priorException)` | not ported — scope. `checksumEntireFile`'s role is covered by `lucene-index/src/checksum_verify.rs`; the rest are merge/copy-path helpers with no caller yet |

### 16. [MISSING] `headerLength`/`indexHeaderLength` and the `expectedLength` checksum overload
`headerLength` is reachable and was already duplicated:
`lucene-codecs/src/compound_format.rs` defines its own
`index_header_length(codec)`. `retrieveChecksum(in, expectedLength)` is the
variant that catches a file that is *longer* or *shorter* than the caller
expects (a truncated file whose tail happens to look like a footer otherwise
slips through). Resolution: **fixed** — all three added, with tests that assert
`header_length`/`index_header_length` equal the byte count
`write_header`/`write_index_header` actually emit (so they cannot drift), and
that the length variant rejects short, long and sub-footer-length claims.

### 17. [MISSING] the codec-name and suffix constraints were not enforced
Java throws `IllegalArgumentException` unless the codec name is simple ASCII
< 128 chars and the suffix simple ASCII < 256 chars. Those are not cosmetic:
`headerLength = 9 + codec.length()` is only correct while the name is ASCII
and its vint length is one byte, and the suffix length is written as a
**single byte**, so a 256-char suffix was silently written as length 0,
producing a file no reader can parse. Resolution: **fixed** — `debug_assert!`
in both writers with Java's messages (kept infallible: 79 call sites across the
workspace pass compile-time constants, so making these return `Result` would
be churn for a caller bug that cannot depend on input), plus `#[should_panic]`
tests for a non-ASCII codec name and a 256-byte suffix.

### 18. [INTENTIONAL] one-shot CRC instead of a running checksum
Java accumulates the CRC incrementally through `ChecksumIndexInput`/
`BufferedChecksum` as the file is read; this port hashes the covered byte
range in one `crc32fast::hash` call at `check_footer` time. Same result, and
`crc32fast` dispatches to the PCLMULQDQ/SSE4.2 path, so the single bulk pass
over already-resident bytes is competitive with Java's incremental
`java.util.zip.CRC32` — with no per-read update cost on the hot decode path,
which is where Java pays for it. Recorded, no change.

### Verdict
Swept-clean after findings 16–17.

---

## crates/lucene-store/src/directory.rs

Java counterparts: `store/Directory.java`, `store/FSDirectory.java`,
`store/MMapDirectory.java`, plus `index/SegmentInfos.java`
(`getLastCommitGeneration`, `generationFromSegmentsFileName`) and
`index/IndexFileNames.java` (`fileNameFromGeneration`).

| Rust | Java | Status |
|---|---|---|
| `Directory::list_all` / `list_all()` | `Directory.listAll` / `FSDirectory.listAll` | identical (sorted; byte order equals Java's UTF-16 order for the ASCII names Lucene produces) |
| `Directory::open` | `Directory.openInput` | equivalent, whole-file — INTENTIONAL (see finding 21) |
| `Directory::create_output` | `createOutput` | identical |
| `Directory::sync` | `sync` (+ `syncMetaData`'s directory fsync) | superset — INTENTIONAL, documented |
| `FsDirectory` / `MmapDirectory` | `NIOFSDirectory` / `MMapDirectory` | equivalent |
| `Input` (+ `Debug`, `Deref`) | `IndexInput` | equivalent role |
| `generation_from_segments_file_name` | `SegmentInfos.generationFromSegmentsFileName` | equivalent (finding 20) |
| `last_commit_generation` | `SegmentInfos.getLastCommitGeneration` | divergent → addressed (finding 19) |
| `segments_file_name` | `IndexFileNames.fileNameFromGeneration` | identical |
| `read_latest_commit` | (the `SegmentInfos.readLatestCommit` prologue) | fixed (finding 19) |
| — | `deleteFile`, `fileLength`, `rename`, `syncMetaData`, `copyFrom`, `createTempOutput`, `obtainLock`, `ensureOpen`, pending-delete bookkeeping | not ported — scope, already declared on the trait's doc comment and in `docs/parity.md` (commit lifecycle + locking). No caller can reach them |

### 19. [CORRECTNESS] an unparsable `segments*` name silently selected an older commit
Java's `getLastCommitGeneration` calls `generationFromSegmentsFileName` on
every `segments*` name and lets the exception escape, so a corrupt directory
fails fast. This port's `filter_map(...ok())` skipped the bad name and returned
the next-highest generation instead. Consequence: opening a directory that
contains, say, a half-written `segments_zzzzzzzzzzzzz` reads the *previous*
commit and every document committed since simply disappears — silent data
loss, the worst failure mode for this function. Resolution: **fixed** —
`read_latest_commit` now performs the strict scan itself and propagates the
error; `last_commit_generation` keeps its lenient contract (the write path
uses it only to pick the next generation, and the strict check has already run
on open), now documented as such. Test asserts the read path errors while the
lenient scan still returns 1, plus a test that the legacy `segments.gen`
pointer is still ignored by the read path. Deliberately **not** a signature
change: the two callers live in `lucene-index`, outside this batch's file set.

### 20. [INTENTIONAL] stricter `segments_` prefix parsing
Java takes `fileName.substring(1 + "segments".length())` for anything merely
*starting with* `segments`, so `segmentsX1` parses as generation 1; this port
requires the `segments_` separator and rejects it. Rust is stricter on names no
Lucene writer produces. Recorded, no change.

### 21. [PERF/INTENTIONAL] `Directory::open` reads whole files
Java returns a lazily-paged `IndexInput`. `MmapDirectory` (the intended
default, matching Lucene's own) is zero-copy and equivalent; `FsDirectory`
slurps the file, which is the price of the "safe, no `unsafe`" fallback and is
documented on the type. Recorded, no change.

### Verdict
Swept-clean after finding 19; the unported `Directory` surface is
scope, tracked in `docs/parity.md`.

---

## crates/lucene-store/src/index_output.rs

Java counterparts: `store/IndexOutput.java`,
`store/OutputStreamIndexOutput.java`, `store/FSDirectory.java`
(`fsync`/`sync`).

| Rust | Java | Status |
|---|---|---|
| `IndexOutput::name` | `getName()` | identical |
| `IndexOutput::file_pointer` | `getFilePointer()` | identical |
| `IndexOutput::checksum` | `getChecksum()` | identical (running CRC-32, zlib polynomial) |
| `FsIndexOutput::create` | `FSDirectory.createOutput` | identical (create/truncate) |
| `FsIndexOutput::close` | `close()` | identical contract (flush to OS, no fsync — durability is `Directory.sync`'s job); additionally surfaces the latched write error |
| `FsIndexOutput::path` | — | not-in-Java (needed by `sync`) |
| `sync(root, names)` | `FSDirectory.sync` + `syncMetaData` | superset (also fsyncs the directory entry, best effort) |
| `write_byte`/`write_bytes` | `writeByte`/`writeBytes` | identical, with the documented sticky-error model since `DataOutput` is infallible by signature |
| — | `alignFilePointer`/`alignOffset`, `Directory.createTempOutput`, `IndexOutput.getFilePointer` beyond `u64` | not ported — scope, no caller |

### 22. [INTENTIONAL] infallible `DataOutput` + sticky error
Already documented at the top of the module and in `docs/parity.md`. Re-checked
against Java: `IndexOutput.close()` does not fsync there either, so the
contract matches. No change.

### Verdict
Swept-clean. No divergence found beyond documented scope.

---

## Summary of changes made

Code:
- `lucene-util/src/zigzag.rs`: `encode_i32`/`decode_i32` (finding 1).
- `lucene-util/src/small_float.rs`: domain guard on `int_to_byte4` (2).
- `lucene-util/src/fixed_bit_set.rs`: ghost-bit verification in `from_words` (3).
- `lucene-store/src/data_input.rs`: 5-byte `read_vint` bound (5), `read_zint`
  (6), group-varint fallback + per-group branchless decision (7, 11),
  `read_length` guard for string/map/set (8), bulk `read_i64s` override (12),
  overflow-safe offset arithmetic in `read_bytes`/`read_u32s_le`.
- `lucene-store/src/data_output.rs`: `write_zint`, `write_group_vints` (14),
  negative-vlong assertion (15).
- `lucene-store/src/codec_util.rs`: `header_length`, `index_header_length`,
  `retrieve_checksum_with_expected_length` (16), codec-name/suffix
  constraints (17).
- `lucene-store/src/directory.rs`: strict generation scan in
  `read_latest_commit` (19).

Tests: 22 new unit tests + 1 new proptest across the two crates; 4 new
**write-side differential tests** in `crates/lucene-store/tests/java_fixtures.rs`
(re-encoding Java's own fixture values must reproduce Java's bytes for vint,
vlong, zlong and group-vint); 1 new criterion bench
(`data_input/read_i64s_block`).

Docs: `docs/parity.md` rows updated for `DataInput`, `GroupVIntUtil`,
`DataOutput`, `CodecUtil` and `retrieveChecksum`.

## Open items

- The `read_zint`/`write_zint` copies in `lucene-codecs/src/stored_fields.rs`
  and `write_group_vints` in `lucene-codecs/src/postings.rs`, and
  `index_header_length` in `lucene-codecs/src/compound_format.rs`, should be
  migrated to the now-existing `lucene-store` primitives. Not done here: those
  files are outside this batch and were being edited concurrently.
- `last_commit_generation` keeps Java's lenient-vs-strict split documented
  above; unifying it means changing a signature used from `lucene-index`.
- Finding 13 (bench-binary code-layout artifact on `read_vlong`) is recorded,
  not fixed.
